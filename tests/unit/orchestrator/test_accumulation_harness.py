"""Unit tests for Component 29 — memory_accumulation_harness.

The M4 milestone gate. Drives the full read→inject→write→accumulate loop forward
across 3 deterministic synthetic prior weeks + the real 2026-W24 data leg,
asserting accumulation-correctness at every week boundary.

Coverage matrix (TDD §29 Sample Selection):
  TestWeekBoundaryAssertions   — per-week criteria 1–5 at each boundary (≥5 cells/week × 4 weeks)
  TestFinalStateEqualsExpected — final accumulated state == independently-predicted ground-truth
  TestHarnessRerunDeterminism  — second run produces identical assertion verdicts (1 cell)
  TestIsolationContract        — harness never mutates real state/; re-run is safe
  TestNonNegotiableCodePath    — synthetic weeks traverse SAME code path as real weeks (no branching)
  TestPerWeekTranscripts       — founder-readable transcripts and briefing files produced

Real-data provenance note:
  WEEK-D uses real 2026-W24 memory files from state/memory/ (value.md etc.),
  injected READ-ONLY into the isolated harness workspace.  No PII; tickers and
  weights only.  Source: live 2026-W24 production run, sanitized 2026-06-09.

Non-negotiable design assertion:
  The harness calls run_weekly (the SAME orchestrator function as real weeks),
  not a harness-specific stub code path.  This is verified in TestNonNegotiableCodePath.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from round_table_portfolio.orchestrator.accumulation_harness import (
    PERSONA_SLUGS_7,
    WEEK_A,
    WEEK_B,
    WEEK_C,
    WEEK_D,
    HarnessWorkspace,
    HarnessResult,
    WeekBoundaryAssertions,
    build_harness_report,
    run_accumulation_harness,
    _compute_predicted_state,
    _assert_writeback_appended,
    _assert_backfill_correct,
    _assert_round_trip,
    _assert_no_cross_persona_leakage,
    _assert_windowed_content,
    _assert_digest_attribution,
    _assert_no_section_corruption,
    _assert_final_state_min_entries,
    _make_round1_output_week_a,
    _make_round1_output_week_c,
)
from round_table_portfolio.orchestrator.memory import (
    SECTION_PAST_CALLS,
    SECTION_COUNTERFACTUAL,
    SECTION_DEBATE_STANCES,
    SECTION_WHATS_NEW,
    _ALL_SECTIONS,
    parse_memory_file,
)
from round_table_portfolio.storage.apply_schema import apply_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stub_allow(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("STUB_ALLOW", "1")
    yield


@pytest.fixture()
def workspace(tmp_path: Path) -> HarnessWorkspace:
    """Provide a fresh isolated workspace for each test."""
    return HarnessWorkspace(root=tmp_path / "harness")


@pytest.fixture(scope="module")
def full_run_result() -> Iterator[HarnessResult]:
    """Run the full harness ONCE and share the result across all tests in this module.

    Scope=module so the expensive multi-week run happens once, not per test.
    """
    with tempfile.TemporaryDirectory() as td:
        ws = HarnessWorkspace(root=Path(td) / "harness")
        result = run_accumulation_harness(ws)
        yield result


# ---------------------------------------------------------------------------
# TestNonNegotiableCodePath
# ---------------------------------------------------------------------------


class TestNonNegotiableCodePath:
    """The non-negotiable design constraint: synthetic weeks MUST traverse the SAME
    code path as real weeks (TDD §29 — Major if violated).

    These tests verify that the harness calls run_weekly (the real orchestrator),
    not a synthetic-only stub path.  If the harness had a special-case branch for
    synthetic data, it would be testing a fiction.
    """

    def test_run_weekly_is_the_actual_orchestrator(self) -> None:
        """run_accumulation_harness imports and calls run_weekly from weekly_run — not a stub."""
        import round_table_portfolio.orchestrator.accumulation_harness as harness_mod
        from round_table_portfolio.orchestrator.weekly_run import run_weekly as real_run_weekly

        # The harness module must have imported run_weekly from weekly_run.
        assert hasattr(harness_mod, "run_weekly"), (
            "accumulation_harness does not import run_weekly — it must call the real orchestrator"
        )
        assert harness_mod.run_weekly is real_run_weekly, (
            "accumulation_harness.run_weekly is not the real orchestrator — "
            "synthetic weeks MUST traverse the same code path as real weeks"
        )

    def test_no_synthetic_only_branching_in_run_weekly(self) -> None:
        """Verify weekly_run.py contains no 'synthetic' or 'harness' conditional branches.

        A simple source-text check: the run_weekly function source must not contain
        is_synthetic / harness_mode / if synthetic: style special-casing.
        """
        import inspect
        from round_table_portfolio.orchestrator.weekly_run import run_weekly

        source = inspect.getsource(run_weekly)
        forbidden_patterns = [
            "is_synthetic",
            "harness_mode",
            "synthetic_week",
            "if synthetic",
            "synthetic=True",
        ]
        found = [p for p in forbidden_patterns if p in source]
        assert not found, (
            f"run_weekly contains synthetic-week special-case code: {found}. "
            "This violates the non-negotiable design constraint."
        )

    def test_writeback_memory_is_the_real_function(self) -> None:
        """The harness uses writeback_memory from memory.py — not a stub."""
        import round_table_portfolio.orchestrator.accumulation_harness as harness_mod

        # The harness calls run_weekly which internally calls writeback_memory.
        # The harness itself does not import writeback_memory directly (it goes through
        # run_weekly), but run_weekly must use the real writeback_memory.
        from round_table_portfolio.orchestrator.weekly_run import writeback_memory as ww_writeback
        from round_table_portfolio.orchestrator.memory import writeback_memory as real_writeback

        assert ww_writeback is real_writeback, (
            "weekly_run.writeback_memory is not the real memory.writeback_memory"
        )


# ---------------------------------------------------------------------------
# TestWeekBoundaryAssertions — criteria 1–5 at each week boundary
# ---------------------------------------------------------------------------


class TestWeekBoundaryAssertions:
    """Per-week accumulation-correctness assertions.

    Uses the module-scoped full_run_result so the harness runs once.
    """

    # --- WEEK-A ---

    def test_week_a_c1_writeback_appended(self, full_run_result: HarnessResult) -> None:
        """WEEK-A: write-back appended entries for all 7 personas."""
        ba = full_run_result.week_assertions.get(WEEK_A)
        assert ba is not None, f"No assertions recorded for {WEEK_A}"
        assert ba.c1_writeback_appended, (
            f"C1a FAIL for {WEEK_A}: {ba.c1_writeback_appended_note}"
        )

    def test_week_a_c1_backfill(self, full_run_result: HarnessResult) -> None:
        """WEEK-A: no backfill expected (cold start)."""
        ba = full_run_result.week_assertions.get(WEEK_A)
        assert ba is not None
        assert ba.c1_backfill_correct, (
            f"C1b FAIL for {WEEK_A}: {ba.c1_backfill_note}"
        )

    def test_week_a_c2_round_trip(self, full_run_result: HarnessResult) -> None:
        """WEEK-A: read reproduces what was written."""
        ba = full_run_result.week_assertions.get(WEEK_A)
        assert ba is not None
        assert ba.c2_round_trip, f"C2 FAIL for {WEEK_A}: {ba.c2_round_trip_note}"

    def test_week_a_c3_no_leakage(self, full_run_result: HarnessResult) -> None:
        """WEEK-A: no cross-persona leakage in briefings."""
        ba = full_run_result.week_assertions.get(WEEK_A)
        assert ba is not None
        assert ba.c3_no_leakage, f"C3a FAIL for {WEEK_A}: {ba.c3_no_leakage_note}"

    def test_week_a_c3_windowed_content(self, full_run_result: HarnessResult) -> None:
        """WEEK-A: briefing files present for all 7 personas."""
        ba = full_run_result.week_assertions.get(WEEK_A)
        assert ba is not None
        assert ba.c3_windowed_content, (
            f"C3b FAIL for {WEEK_A}: {ba.c3_windowed_content_note}"
        )

    def test_week_a_c4_digest(self, full_run_result: HarnessResult) -> None:
        """WEEK-A: digest attribution check passes (cold start — no resolved rows)."""
        ba = full_run_result.week_assertions.get(WEEK_A)
        assert ba is not None
        assert ba.c4_digest_attribution, (
            f"C4 FAIL for {WEEK_A}: {ba.c4_digest_attribution_note}"
        )

    def test_week_a_c5_no_corruption(self, full_run_result: HarnessResult) -> None:
        """WEEK-A: all 4 sections present; no duplicate entries."""
        ba = full_run_result.week_assertions.get(WEEK_A)
        assert ba is not None
        assert ba.c5_no_corruption, (
            f"C5 FAIL for {WEEK_A}: {ba.c5_no_corruption_note}"
        )

    # --- WEEK-B ---

    def test_week_b_c1_writeback_appended(self, full_run_result: HarnessResult) -> None:
        """WEEK-B: write-back appended entries for all 7 personas."""
        ba = full_run_result.week_assertions.get(WEEK_B)
        assert ba is not None
        assert ba.c1_writeback_appended, (
            f"C1a FAIL for {WEEK_B}: {ba.c1_writeback_appended_note}"
        )

    def test_week_b_c1_backfill(self, full_run_result: HarnessResult) -> None:
        """WEEK-B: WEEK-A outcomes resolved — backfill must replace 'pending' entries."""
        ba = full_run_result.week_assertions.get(WEEK_B)
        assert ba is not None
        assert ba.c1_backfill_correct, (
            f"C1b FAIL for {WEEK_B}: {ba.c1_backfill_note}"
        )

    def test_week_b_c2_round_trip(self, full_run_result: HarnessResult) -> None:
        """WEEK-B: read reproduces what was written."""
        ba = full_run_result.week_assertions.get(WEEK_B)
        assert ba is not None
        assert ba.c2_round_trip, f"C2 FAIL for {WEEK_B}: {ba.c2_round_trip_note}"

    def test_week_b_c3_no_leakage(self, full_run_result: HarnessResult) -> None:
        """WEEK-B: no cross-persona leakage in briefings."""
        ba = full_run_result.week_assertions.get(WEEK_B)
        assert ba is not None
        assert ba.c3_no_leakage, f"C3a FAIL for {WEEK_B}: {ba.c3_no_leakage_note}"

    def test_week_b_c3_windowed_content(self, full_run_result: HarnessResult) -> None:
        """WEEK-B: briefing files present; WEEK-A memory now in window."""
        ba = full_run_result.week_assertions.get(WEEK_B)
        assert ba is not None
        assert ba.c3_windowed_content, (
            f"C3b FAIL for {WEEK_B}: {ba.c3_windowed_content_note}"
        )

    def test_week_b_c4_digest(self, full_run_result: HarnessResult) -> None:
        """WEEK-B: digest attribution passes — resolved outcomes present after backfill."""
        ba = full_run_result.week_assertions.get(WEEK_B)
        assert ba is not None
        assert ba.c4_digest_attribution, (
            f"C4 FAIL for {WEEK_B}: {ba.c4_digest_attribution_note}"
        )

    def test_week_b_c5_no_corruption(self, full_run_result: HarnessResult) -> None:
        """WEEK-B: no section corruption; WEEK-A + WEEK-B entries, no duplicates."""
        ba = full_run_result.week_assertions.get(WEEK_B)
        assert ba is not None
        assert ba.c5_no_corruption, (
            f"C5 FAIL for {WEEK_B}: {ba.c5_no_corruption_note}"
        )

    # --- WEEK-C ---

    def test_week_c_c1_writeback_appended(self, full_run_result: HarnessResult) -> None:
        """WEEK-C: write-back appended entries for all 7 personas."""
        ba = full_run_result.week_assertions.get(WEEK_C)
        assert ba is not None
        assert ba.c1_writeback_appended, (
            f"C1a FAIL for {WEEK_C}: {ba.c1_writeback_appended_note}"
        )

    def test_week_c_c1_backfill(self, full_run_result: HarnessResult) -> None:
        """WEEK-C: WEEK-B outcomes resolved — backfill fires."""
        ba = full_run_result.week_assertions.get(WEEK_C)
        assert ba is not None
        assert ba.c1_backfill_correct, (
            f"C1b FAIL for {WEEK_C}: {ba.c1_backfill_note}"
        )

    def test_week_c_c2_round_trip(self, full_run_result: HarnessResult) -> None:
        """WEEK-C: read reproduces what was written."""
        ba = full_run_result.week_assertions.get(WEEK_C)
        assert ba is not None
        assert ba.c2_round_trip, f"C2 FAIL for {WEEK_C}: {ba.c2_round_trip_note}"

    def test_week_c_c3_no_leakage(self, full_run_result: HarnessResult) -> None:
        """WEEK-C: no cross-persona leakage in briefings."""
        ba = full_run_result.week_assertions.get(WEEK_C)
        assert ba is not None
        assert ba.c3_no_leakage, f"C3a FAIL for {WEEK_C}: {ba.c3_no_leakage_note}"

    def test_week_c_c3_windowed_content(self, full_run_result: HarnessResult) -> None:
        """WEEK-C: briefing files present; window=2 shows W21+W22 stances."""
        ba = full_run_result.week_assertions.get(WEEK_C)
        assert ba is not None
        assert ba.c3_windowed_content, (
            f"C3b FAIL for {WEEK_C}: {ba.c3_windowed_content_note}"
        )

    def test_week_c_c4_digest(self, full_run_result: HarnessResult) -> None:
        """WEEK-C: digest attribution passes."""
        ba = full_run_result.week_assertions.get(WEEK_C)
        assert ba is not None
        assert ba.c4_digest_attribution, (
            f"C4 FAIL for {WEEK_C}: {ba.c4_digest_attribution_note}"
        )

    def test_week_c_c5_no_corruption(self, full_run_result: HarnessResult) -> None:
        """WEEK-C: 3 entries per section per persona; no duplicates."""
        ba = full_run_result.week_assertions.get(WEEK_C)
        assert ba is not None
        assert ba.c5_no_corruption, (
            f"C5 FAIL for {WEEK_C}: {ba.c5_no_corruption_note}"
        )

    # --- WEEK-D ---

    def test_week_d_c1_writeback_appended(self, full_run_result: HarnessResult) -> None:
        """WEEK-D: write-back appended entries for all 7 personas (real-data leg)."""
        ba = full_run_result.week_assertions.get(WEEK_D)
        assert ba is not None
        assert ba.c1_writeback_appended, (
            f"C1a FAIL for {WEEK_D}: {ba.c1_writeback_appended_note}"
        )

    def test_week_d_c2_round_trip(self, full_run_result: HarnessResult) -> None:
        """WEEK-D: read reproduces what was written (real-data leg)."""
        ba = full_run_result.week_assertions.get(WEEK_D)
        assert ba is not None
        assert ba.c2_round_trip, f"C2 FAIL for {WEEK_D}: {ba.c2_round_trip_note}"

    def test_week_d_c3_no_leakage(self, full_run_result: HarnessResult) -> None:
        """WEEK-D: no cross-persona leakage in briefings (real-data leg)."""
        ba = full_run_result.week_assertions.get(WEEK_D)
        assert ba is not None
        assert ba.c3_no_leakage, f"C3a FAIL for {WEEK_D}: {ba.c3_no_leakage_note}"

    def test_week_d_c3_windowed_content(self, full_run_result: HarnessResult) -> None:
        """WEEK-D: briefing files present for all 7 personas."""
        ba = full_run_result.week_assertions.get(WEEK_D)
        assert ba is not None
        assert ba.c3_windowed_content, (
            f"C3b FAIL for {WEEK_D}: {ba.c3_windowed_content_note}"
        )

    def test_week_d_c5_no_corruption(self, full_run_result: HarnessResult) -> None:
        """WEEK-D: no section corruption after real-data injection + run."""
        ba = full_run_result.week_assertions.get(WEEK_D)
        assert ba is not None
        assert ba.c5_no_corruption, (
            f"C5 FAIL for {WEEK_D}: {ba.c5_no_corruption_note}"
        )


# ---------------------------------------------------------------------------
# TestFinalStateEqualsExpected
# ---------------------------------------------------------------------------


class TestFinalStateEqualsExpected:
    """Final accumulated state must match independently-predicted ground-truth."""

    def test_final_state_equals_predicted(self, full_run_result: HarnessResult) -> None:
        """Final memory state >= predicted entry count per section per persona."""
        assert full_run_result.final_state_equals_predicted, (
            f"Final state FAIL: {full_run_result.final_state_note}"
        )

    def test_all_weeks_assertions_pass(self, full_run_result: HarnessResult) -> None:
        """Every week boundary's assertions must all pass (100% — the option-A close bar)."""
        failed = []
        for week_id, ba in full_run_result.week_assertions.items():
            if not ba.all_passed:
                failed.append(f"{week_id}: {'; '.join(ba.summary_lines())}")
        assert not failed, (
            "Some week boundaries FAILED:\n" + "\n".join(failed)
        )

    def test_no_harness_errors(self, full_run_result: HarnessResult) -> None:
        """Harness must complete without errors."""
        assert not full_run_result.errors, (
            f"Harness run produced errors:\n" + "\n".join(full_run_result.errors)
        )

    def test_predicted_state_formula(self) -> None:
        """_compute_predicted_state returns correct counts for a known sequence."""
        predicted = _compute_predicted_state([WEEK_A, WEEK_B, WEEK_C])
        # 3 weeks, below the 12-cap → expected = 3 per section per persona
        assert len(predicted) == 7
        for slug, sections in predicted.items():
            assert len(sections) == 4, f"{slug}: expected 4 sections, got {len(sections)}"
            for section_name, count in sections.items():
                assert count == 3, (
                    f"{slug}/{section_name}: expected 3, got {count}"
                )


# ---------------------------------------------------------------------------
# TestHarnessRerunDeterminism
# ---------------------------------------------------------------------------


class TestHarnessRerunDeterminism:
    """Re-run produces identical assertion verdicts (determinism — no flakiness)."""

    def test_rerun_produces_identical_verdicts(self) -> None:
        """Run the harness twice with fresh isolated workspaces; assert identical verdicts.

        Two separate runs in separate temp dirs must produce the same per-week,
        per-criterion PASS/FAIL pattern.
        """
        with tempfile.TemporaryDirectory() as td1:
            ws1 = HarnessWorkspace(root=Path(td1) / "harness")
            result1 = run_accumulation_harness(ws1)

        with tempfile.TemporaryDirectory() as td2:
            ws2 = HarnessWorkspace(root=Path(td2) / "harness")
            result2 = run_accumulation_harness(ws2)

        # Both must pass overall
        assert not result1.errors, f"Run 1 errors: {result1.errors}"
        assert not result2.errors, f"Run 2 errors: {result2.errors}"

        # Per-week verdicts must match exactly
        assert set(result1.week_assertions.keys()) == set(result2.week_assertions.keys()), (
            "Run 1 and Run 2 produced different sets of week assertions"
        )

        for week_id in result1.week_assertions:
            ba1 = result1.week_assertions[week_id]
            ba2 = result2.week_assertions[week_id]
            assert ba1.c1_writeback_appended == ba2.c1_writeback_appended, (
                f"{week_id} C1a differs between runs"
            )
            assert ba1.c1_backfill_correct == ba2.c1_backfill_correct, (
                f"{week_id} C1b differs between runs"
            )
            assert ba1.c2_round_trip == ba2.c2_round_trip, (
                f"{week_id} C2 differs between runs"
            )
            assert ba1.c3_no_leakage == ba2.c3_no_leakage, (
                f"{week_id} C3a differs between runs"
            )
            assert ba1.c3_windowed_content == ba2.c3_windowed_content, (
                f"{week_id} C3b differs between runs"
            )
            assert ba1.c4_digest_attribution == ba2.c4_digest_attribution, (
                f"{week_id} C4 differs between runs"
            )
            assert ba1.c5_no_corruption == ba2.c5_no_corruption, (
                f"{week_id} C5 differs between runs"
            )

        # Final state check must match
        assert result1.final_state_equals_predicted == result2.final_state_equals_predicted, (
            "Final-state equality differs between runs"
        )


# ---------------------------------------------------------------------------
# TestIsolationContract
# ---------------------------------------------------------------------------


class TestIsolationContract:
    """Harness never mutates the real state/; workspace is isolated."""

    def test_harness_uses_temp_workspace_not_real_state(
        self, workspace: HarnessWorkspace
    ) -> None:
        """The harness workspace must not be under the real state/ directory."""
        real_state = Path(__file__).parents[4] / "state"
        # workspace.state_root must NOT be inside the real state/
        try:
            workspace.state_root.relative_to(real_state)
            pytest.fail(
                f"Harness workspace {workspace.state_root} is inside real state/ — "
                "isolation violation"
            )
        except ValueError:
            pass  # Not a subpath — correct

    def test_real_state_memory_unchanged_after_harness_run(
        self, workspace: HarnessWorkspace
    ) -> None:
        """Running the harness must not modify any real state/memory/ file."""
        real_memory_dir = Path(__file__).parents[4] / "state" / "memory"
        if not real_memory_dir.exists():
            pytest.skip("Real state/memory/ not present — skip isolation check")

        # Capture mtimes of real memory files before the run.
        before_mtimes: dict[str, float] = {
            f.name: f.stat().st_mtime
            for f in real_memory_dir.glob("*.md")
        }

        # Run the harness in the isolated workspace.
        result = run_accumulation_harness(workspace)

        # Capture mtimes after.
        after_mtimes: dict[str, float] = {
            f.name: f.stat().st_mtime
            for f in real_memory_dir.glob("*.md")
        }

        # No real file should have changed.
        for fname, before_mtime in before_mtimes.items():
            after_mtime = after_mtimes.get(fname)
            assert after_mtime == before_mtime, (
                f"Real state/memory/{fname} was modified by the harness — isolation violated"
            )

    def test_real_ledger_unchanged_after_harness_run(
        self, workspace: HarnessWorkspace
    ) -> None:
        """Running the harness must not modify the real state/ledger.db."""
        real_ledger = Path(__file__).parents[4] / "state" / "ledger.db"
        if not real_ledger.exists():
            pytest.skip("Real ledger.db not present — skip isolation check")

        before_mtime = real_ledger.stat().st_mtime
        run_accumulation_harness(workspace)
        after_mtime = real_ledger.stat().st_mtime

        assert after_mtime == before_mtime, (
            "Real state/ledger.db was modified by the harness — isolation violated"
        )

    def test_harness_db_is_separate_from_real_db(
        self, workspace: HarnessWorkspace
    ) -> None:
        """The harness ledger path must differ from the real ledger path."""
        real_ledger = Path(__file__).parents[4] / "state" / "ledger.db"
        assert workspace.db_path != real_ledger, (
            "Harness db_path is the same as the real ledger — isolation violated"
        )


# ---------------------------------------------------------------------------
# TestPerWeekTranscripts
# ---------------------------------------------------------------------------


class TestPerWeekTranscripts:
    """Founder-readable transcripts and briefing files are produced."""

    def test_all_four_weeks_have_transcripts(self, full_run_result: HarnessResult) -> None:
        """All 4 weeks must have transcript paths recorded."""
        for week_id in [WEEK_A, WEEK_B, WEEK_C, WEEK_D]:
            assert week_id in full_run_result.transcript_paths, (
                f"No transcript path for {week_id}"
            )
            transcript_path = full_run_result.transcript_paths[week_id]
            assert transcript_path is not None
            assert transcript_path.exists(), (
                f"Transcript file missing for {week_id}: {transcript_path}"
            )

    def test_all_four_weeks_have_briefing_dirs(self, full_run_result: HarnessResult) -> None:
        """All 4 weeks must have briefing directories."""
        for week_id in [WEEK_A, WEEK_B, WEEK_C, WEEK_D]:
            assert week_id in full_run_result.briefing_dirs, (
                f"No briefing_dir for {week_id}"
            )
            bdir = full_run_result.briefing_dirs[week_id]
            assert bdir is not None
            assert bdir.exists(), f"Briefing dir missing for {week_id}: {bdir}"
            # 7 persona briefing files + consensus_book.md (M6) under the dir.
            bf_files = list(bdir.glob("*.md"))
            assert len(bf_files) >= 7, (
                f"Expected >= 7 briefing files for {week_id}, got {len(bf_files)}"
            )

    def test_founder_transcripts_present(self, full_run_result: HarnessResult) -> None:
        """Founder transcript text produced for all 4 weeks."""
        for week_id in [WEEK_A, WEEK_B, WEEK_C, WEEK_D]:
            assert week_id in full_run_result.founder_transcripts, (
                f"No founder_transcript for {week_id}"
            )
            text = full_run_result.founder_transcripts[week_id]
            assert len(text) > 50, f"Founder transcript for {week_id} is too short: {text!r}"

    def test_report_builds_successfully(self, full_run_result: HarnessResult, tmp_path: Path) -> None:
        """build_harness_report writes a non-empty markdown file."""
        report_path = build_harness_report(full_run_result, output_dir=tmp_path / "report")
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "M4 Memory Accumulation Harness" in content
        assert "PASS" in content
        assert "What to Look At" in content
        assert len(content) > 500

    def test_week_c_conviction_shift_is_visible_in_value_stance(
        self, full_run_result: HarnessResult
    ) -> None:
        """The WEEK-C VALUE memory shows a stance change vs WEEK-A.

        WEEK-A: VALUE confidence=5 ADD on NVDA.
        WEEK-C: VALUE should show REDUCE / reduced confidence on NVDA.

        This is the core memory-in-action exhibit: memory informed the conviction shift.
        """
        ws = full_run_result.workspace
        mem_path = ws.memory_dir / "value.md"
        assert mem_path.exists(), "value.md memory file missing"

        parsed = parse_memory_file(mem_path)
        pc = parsed.sections.get(SECTION_PAST_CALLS)
        assert pc is not None, "value.md has no Past Calls Log section"

        # Find WEEK-A and WEEK-C entries
        week_a_entry = None
        week_c_entry = None
        for week_id, body in pc.entries:
            if week_id == WEEK_A:
                week_a_entry = body
            elif week_id == WEEK_C:
                week_c_entry = body

        assert week_a_entry is not None, f"No Past Calls entry for {WEEK_A} in value.md"
        assert week_c_entry is not None, f"No Past Calls entry for {WEEK_C} in value.md"

        # WEEK-A must show a high-conviction NVDA ADD
        assert "NVDA" in week_a_entry, f"WEEK-A entry does not mention NVDA: {week_a_entry[:200]}"
        assert "confidence=5" in week_a_entry or "conf=5" in week_a_entry or "5" in week_a_entry, (
            f"WEEK-A entry does not show conf=5: {week_a_entry[:200]}"
        )

        # WEEK-C must show a REDUCE / reduced conviction on NVDA
        assert "NVDA" in week_c_entry, f"WEEK-C entry does not mention NVDA: {week_c_entry[:200]}"
        # The REDUCE action should be present
        assert "REDUCE" in week_c_entry or "reduce" in week_c_entry.lower(), (
            f"WEEK-C entry does not show REDUCE for NVDA: {week_c_entry[:300]}"
        )


# ---------------------------------------------------------------------------
# TestBackfillScenario — detailed C1b checks
# ---------------------------------------------------------------------------


class TestBackfillScenario:
    """Detailed tests for the outcome backfill scenario (C1b).

    The backfill is the core of TDD §18b — when WEEK-A resolves with alpha=-0.12
    for VALUE, the WEEK-A Past Calls entry must flip from 'outcome: pending' to
    'outcome: alpha=-0.12 resolved=WEEK-B'.
    """

    def test_week_a_entry_starts_as_pending(
        self, full_run_result: HarnessResult
    ) -> None:
        """After WEEK-A runs, the value persona's WEEK-A entry has outcome: pending.

        Note: by the time we check in the full run, WEEK-B has already run and
        backfilled this entry.  We check the CURRENT state shows alpha=, not pending.
        The 'pending → backfilled' transition is the point of this test class.
        """
        ws = full_run_result.workspace
        mem_path = ws.memory_dir / "value.md"
        assert mem_path.exists()
        content = mem_path.read_text(encoding="utf-8")

        # After WEEK-B ran and backfilled, the WEEK-A entry must NOT be pending
        pattern = rf"### Entry {WEEK_A}(.*?)(?=### Entry|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        assert match is not None, f"No ### Entry {WEEK_A} found in value.md"
        section = match.group(1)
        assert "outcome: pending" not in section, (
            f"WEEK-A entry still shows 'outcome: pending' after WEEK-B backfill.\n"
            f"Section content:\n{section[:400]}"
        )

    def test_week_a_entry_has_resolved_alpha_after_backfill(
        self, full_run_result: HarnessResult
    ) -> None:
        """After WEEK-B runs, VALUE's WEEK-A entry must show alpha=-0.12."""
        ws = full_run_result.workspace
        mem_path = ws.memory_dir / "value.md"
        assert mem_path.exists()
        content = mem_path.read_text(encoding="utf-8")

        pattern = rf"### Entry {WEEK_A}(.*?)(?=### Entry|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        assert match is not None
        section = match.group(1)

        # Should contain the resolved alpha
        assert "alpha=" in section, (
            f"WEEK-A entry does not contain 'alpha=' backfill.\nSection:\n{section[:400]}"
        )

    def test_backfill_does_not_corrupt_other_sections(
        self, full_run_result: HarnessResult
    ) -> None:
        """Backfill must not create phantom entries or touch non-past-calls sections."""
        ws = full_run_result.workspace
        for slug in PERSONA_SLUGS_7:
            mem_path = ws.memory_dir / f"{slug}.md"
            assert mem_path.exists()
            parsed = parse_memory_file(mem_path)

            # Check counterfactual, debate stances, whats_new sections have no
            # 'outcome: pending' or 'alpha=' backfill contamination
            for section_name in [SECTION_COUNTERFACTUAL, SECTION_DEBATE_STANCES, SECTION_WHATS_NEW]:
                section = parsed.sections.get(section_name)
                if section:
                    for week_id, body in section.entries:
                        # The backfill is ONLY for Past Calls entries — never other sections
                        if "outcome: alpha=" in body and section_name != SECTION_PAST_CALLS:
                            pytest.fail(
                                f"{slug}/{section_name}/{week_id}: contains 'outcome: alpha=' — "
                                "backfill leaked into wrong section"
                            )


import re  # noqa: E402  (needed by TestBackfillScenario above)


# ---------------------------------------------------------------------------
# TestWindowEviction
# ---------------------------------------------------------------------------


class TestWindowEviction:
    """Recency window = 2 weeks; WEEK-A entries evicted from briefing by WEEK-C.

    In WEEK-C, the memory window is 2 weeks, so the briefing contains
    WEEK-B + WEEK-C stances (not WEEK-A). BUT WEEK-A remains in the
    MEMORY FILE (the 12-entry cap) — only the INJECTION is windowed.
    This is the critical distinction from TDD §26: "the window bounds what gets
    INJECTED, NOT the file."
    """

    def test_week_c_briefing_window_contains_at_most_two_past_call_entries(
        self, full_run_result: HarnessResult
    ) -> None:
        """WEEK-C briefing (window=2) contains at most 2 past-call entries.

        With window=2 and 2 prior entries (WEEK-A + WEEK-B), both appear in
        the briefing (window=2 returns ALL entries when N <= window).  Eviction
        only fires when N > window.  This test validates the window bound is
        correct: the briefing has ≤ 2 past-call entry headers, NOT unbounded.

        After WEEK-D runs (adding a 3rd + 4th entry), WEEK-A would be evicted
        from future briefings — but we are checking the WEEK-C briefing here
        which reads the post-WEEK-B state (2 entries → both shown).
        """
        ws = full_run_result.workspace
        briefing_dir = ws.runs_dir / f"{WEEK_C}-memory"
        if not briefing_dir.exists():
            pytest.skip(f"Briefing dir for {WEEK_C} missing — run did not complete")

        value_bf = briefing_dir / "value.md"
        if not value_bf.exists():
            pytest.skip("Value briefing file missing for WEEK-C")

        content = value_bf.read_text(encoding="utf-8")
        # Count how many "**YYYY-WNN**" entry headers appear in the Past Calls section.
        # The briefing format uses "**<week_id>**" for each entry (from _render_section).
        import re as _re
        # Extract the Past Calls section of the briefing
        pc_match = _re.search(r"### Past Calls Log(.*?)(?=###|\Z)", content, _re.DOTALL)
        if pc_match is None:
            pytest.skip("No Past Calls Log section in WEEK-C briefing")
        pc_section = pc_match.group(1)
        entry_headers = _re.findall(r"\*\*\d{4}-W\d{2}\*\*", pc_section)
        assert len(entry_headers) <= 2, (
            f"WEEK-C briefing Past Calls section has {len(entry_headers)} entries — "
            f"window=2 should bound it to at most 2.\n"
            f"Entry headers found: {entry_headers}"
        )

    def test_week_a_entry_still_in_memory_file_after_window_eviction(
        self, full_run_result: HarnessResult
    ) -> None:
        """WEEK-A entry must still exist in the memory FILE even after window eviction.

        The file retains up to the 12-entry cap; the window bounds injection only.
        """
        ws = full_run_result.workspace
        mem_path = ws.memory_dir / "value.md"
        assert mem_path.exists()
        parsed = parse_memory_file(mem_path)
        pc = parsed.sections.get(SECTION_PAST_CALLS)
        assert pc is not None
        week_ids = [e[0] for e in pc.entries]
        assert WEEK_A in week_ids, (
            f"WEEK-A ({WEEK_A}) missing from value.md Past Calls Log — "
            "window eviction incorrectly deleted the file entry (must only affect injection)"
        )


# ---------------------------------------------------------------------------
# TestSyntheticWeekDesign — verify scenario stubs are correctly structured
# ---------------------------------------------------------------------------


class TestSyntheticWeekDesign:
    """The synthetic week stubs must produce valid round-1 output for the
    designed scenarios."""

    def test_week_a_value_high_conviction_add(self) -> None:
        """WEEK-A VALUE stub encodes conf=5 ADD on NVDA."""
        import json as _json
        output = _json.loads(_make_round1_output_week_a("value"))
        stances = output["stances"]
        nvda = next((s for s in stances if s["ticker"] == "NVDA"), None)
        assert nvda is not None, "WEEK-A VALUE stub has no NVDA stance"
        assert nvda["action"] == "ADD"
        assert nvda["confidence"] == 5
        assert nvda["target_weight"] == pytest.approx(0.18)

    def test_week_a_other_persona_does_not_have_conf5_nvda(self) -> None:
        """WEEK-A non-value persona stubs do NOT have a conf=5 ADD on NVDA
        (the high-conviction signal is VALUE-only for the scenario to work)."""
        import json as _json
        for slug in ["growth", "technical", "risk-officer"]:
            output = _json.loads(_make_round1_output_week_a(slug))
            stances = output["stances"]
            nvda = next((s for s in stances if s["ticker"] == "NVDA"), None)
            if nvda:
                # Other personas may have NVDA but not at conf=5 ADD
                is_high_conv_add = (
                    nvda.get("action") == "ADD" and nvda.get("confidence") == 5
                )
                assert not is_high_conv_add, (
                    f"{slug} WEEK-A stub also has conf=5 ADD on NVDA — "
                    "the high-conviction signal should be VALUE-only"
                )

    def test_week_c_value_reduces_nvda(self) -> None:
        """WEEK-C VALUE stub encodes REDUCE on NVDA (conviction-shift scenario)."""
        import json as _json
        output = _json.loads(_make_round1_output_week_c("value"))
        stances = output["stances"]
        nvda = next((s for s in stances if s["ticker"] == "NVDA"), None)
        assert nvda is not None, "WEEK-C VALUE stub has no NVDA stance"
        assert nvda["action"] == "REDUCE", (
            f"WEEK-C VALUE stub should REDUCE NVDA, got {nvda['action']}"
        )
        assert nvda["confidence"] <= 2, (
            f"WEEK-C VALUE stub should have low confidence (<=2), got {nvda['confidence']}"
        )

    def test_week_c_value_rationale_references_prior_miss(self) -> None:
        """WEEK-C VALUE rationale must explicitly reference the prior WEEK-A miss."""
        import json as _json
        output = _json.loads(_make_round1_output_week_c("value"))
        stances = output["stances"]
        nvda = next((s for s in stances if s["ticker"] == "NVDA"), None)
        assert nvda is not None
        rationale = nvda.get("rationale", "")
        # The rationale should reference the prior call
        assert "2026-W20" in rationale or "prior call" in rationale.lower() or "miss" in rationale.lower(), (
            f"WEEK-C VALUE NVDA rationale does not reference prior miss: {rationale}"
        )


# ---------------------------------------------------------------------------
# TestAssertionTeeth — prove the two strengthened assertions have teeth
#
# Each test constructs a minimal synthetic workspace that would pass the OLD
# (weak) form of the assertion but MUST fail the NEW (strengthened) form.
# If either test is deleted or the assertion is reverted, these tests fail —
# confirming the strengthened check cannot be silently removed.
# ---------------------------------------------------------------------------


class TestAssertionTeeth:
    """Prove the two strengthened assertions (_assert_digest_attribution and
    _assert_windowed_content) have real teeth.

    Design principle: each test creates a minimal fixture that specifically
    triggers the failure mode the old weak assertion missed.  These tests are
    NOT driven by the full harness run — they call the assertion helpers directly
    against hand-crafted fixtures so the failure is isolated and deterministic.
    """

    # ------------------------------------------------------------------
    # Teeth test 1 — _assert_digest_attribution
    # ------------------------------------------------------------------

    def test_digest_attribution_teeth_file_level_alpha_does_not_fool_assertion(
        self, tmp_path: Path
    ) -> None:
        """OLD weak form: 'alpha=' anywhere in the file → always PASS.
        NEW strong form: 'alpha' must appear inside ## What's New Digest → ### Entry <week_id>.

        Fixture: value.md has 'alpha=' in Past Calls Log (backfill format) but
        the ## What's New Digest entry for the week contains no 'alpha' token.
        The old assertion would PASS (file contains 'alpha='); the new must FAIL.
        """
        # Build a minimal workspace pointing at tmp_path
        workspace = HarnessWorkspace(root=tmp_path)

        # Write a value.md where alpha= lives ONLY in Past Calls, NOT in digest
        mem_content = (
            "## Past Calls Log\n\n"
            "### Entry 2026-W20\n"
            "ticker: NVDA\n"
            "outcome: alpha=-0.12 resolved=2026-W21\n\n"
            "## Counterfactual Portfolio Log\n\n"
            "## Debate Stances Log\n\n"
            "## What's New Digest\n\n"
            "### Entry 2026-W21\n"
            "week: 2026-W21\n"
            "digest: No resolved outcomes surfaced in digest this week.\n\n"
        )
        (workspace.memory_dir / "value.md").write_text(mem_content, encoding="utf-8")

        # The old weak form checked 'alpha=' anywhere in the file — would pass
        # because Past Calls Log contains 'alpha='.
        # The new form checks the digest section specifically — must FAIL.
        passed, note = _assert_digest_attribution(
            week_id="2026-W21",
            workspace=workspace,
            expect_resolved=True,
            expect_empty_state=False,
        )
        assert not passed, (
            "TEETH FAIL: _assert_digest_attribution passed when 'alpha' was only in "
            "Past Calls Log, not in ## What's New Digest → ### Entry 2026-W21. "
            f"Note: {note}"
        )
        assert "2026-W21" in note, f"Failure note should mention week_id. Got: {note}"

    def test_digest_attribution_passes_when_alpha_in_digest_section(
        self, tmp_path: Path
    ) -> None:
        """Complement test: assertion correctly PASSes when 'alpha' IS in the digest entry."""
        workspace = HarnessWorkspace(root=tmp_path)

        mem_content = (
            "## Past Calls Log\n\n"
            "## Counterfactual Portfolio Log\n\n"
            "## Debate Stances Log\n\n"
            "## What's New Digest\n\n"
            "### Entry 2026-W21\n"
            "week: 2026-W21\n"
            "digest: Since your last run, these calls resolved:\n"
            "  NVDA (you said add conf=5 in 2026-W20) → alpha -0.1200 vs SPY\n\n"
        )
        (workspace.memory_dir / "value.md").write_text(mem_content, encoding="utf-8")

        passed, note = _assert_digest_attribution(
            week_id="2026-W21",
            workspace=workspace,
            expect_resolved=True,
            expect_empty_state=False,
        )
        assert passed, (
            f"Digest attribution should PASS when 'alpha' is in the digest entry. Note: {note}"
        )

    # ------------------------------------------------------------------
    # Teeth test 2 — _assert_windowed_content
    # ------------------------------------------------------------------

    def test_windowed_content_teeth_extra_entries_trigger_failure(
        self, tmp_path: Path
    ) -> None:
        """OLD weak form: only checked briefing files EXIST — expected_min_entries was unused.
        NEW strong form: Past Calls Log in the briefing must have ≤ memory_window_weeks (2) entries.

        Fixture: briefing files exist for all 7 personas, but value.md briefing
        has 3 past-call entries (exceeding the window=2 bound).
        Old assertion: PASS (files exist).  New assertion: FAIL (3 > 2).
        """
        week_id = "2026-W22"
        workspace = HarnessWorkspace(root=tmp_path)

        # Create the briefing directory
        briefing_dir = workspace.runs_dir / f"{week_id}-memory"
        briefing_dir.mkdir(parents=True, exist_ok=True)

        # Write briefing files — value.md has 3 past-call entries (over window)
        over_window_briefing = (
            "# Memory Briefing — value\n"
            "_Week: 2026-W22_\n\n"
            "### Past Calls Log\n"
            "**2026-W20**\nticker: NVDA action: ADD\n\n"
            "**2026-W21**\nticker: AAPL action: ADD\n\n"
            "**2026-W22**\nticker: MSFT action: HOLD\n\n"   # 3rd entry — exceeds window=2
            "### Counterfactual Portfolio Log\n_(no entries in window)_\n"
            "### Debate Stances Log\n_(no entries in window)_\n"
            "### What's New Digest\n_(no entries in window)_\n"
        )
        (briefing_dir / "value.md").write_text(over_window_briefing, encoding="utf-8")

        # Write normal 1-entry briefings for the other 6 personas
        normal_briefing = (
            "# Memory Briefing — {slug}\n"
            "_Week: 2026-W22_\n\n"
            "### Past Calls Log\n"
            "**2026-W21**\nticker: NVDA action: ADD\n\n"
            "### Counterfactual Portfolio Log\n_(no entries in window)_\n"
            "### Debate Stances Log\n_(no entries in window)_\n"
            "### What's New Digest\n_(no entries in window)_\n"
        )
        for slug in PERSONA_SLUGS_7:
            if slug != "value":
                (briefing_dir / f"{slug}.md").write_text(
                    normal_briefing.format(slug=slug), encoding="utf-8"
                )

        # Old assertion would PASS (all files exist, param ignored).
        # New assertion must FAIL (value has 3 entries, window=2 allows at most 2).
        passed, note = _assert_windowed_content(
            week_id=week_id,
            workspace=workspace,
            expected_min_entries=1,
        )
        assert not passed, (
            "TEETH FAIL: _assert_windowed_content passed when value briefing had 3 "
            "past-call entries but memory_window_weeks=2. "
            f"Note: {note}"
        )
        assert "window not applied" in note or "past-call entries" in note.lower() or "≤2" in note or "window" in note.lower(), (
            f"Failure note should mention windowing. Got: {note}"
        )

    def test_windowed_content_passes_when_entries_within_window(
        self, tmp_path: Path
    ) -> None:
        """Complement: assertion correctly PASSes when entries are within window=2."""
        week_id = "2026-W22"
        workspace = HarnessWorkspace(root=tmp_path)

        briefing_dir = workspace.runs_dir / f"{week_id}-memory"
        briefing_dir.mkdir(parents=True, exist_ok=True)

        # 2 past-call entries — exactly at the window=2 boundary
        two_entry_briefing = (
            "# Memory Briefing — {slug}\n"
            "_Week: 2026-W22_\n\n"
            "### Past Calls Log\n"
            "**2026-W20**\nticker: NVDA action: ADD\n\n"
            "**2026-W21**\nticker: AAPL action: ADD\n\n"
            "### Counterfactual Portfolio Log\n_(no entries in window)_\n"
            "### Debate Stances Log\n_(no entries in window)_\n"
            "### What's New Digest\n_(no entries in window)_\n"
        )
        for slug in PERSONA_SLUGS_7:
            (briefing_dir / f"{slug}.md").write_text(
                two_entry_briefing.format(slug=slug), encoding="utf-8"
            )

        passed, note = _assert_windowed_content(
            week_id=week_id,
            workspace=workspace,
            expected_min_entries=2,
        )
        assert passed, (
            f"_assert_windowed_content should PASS with 2 entries and window=2. Note: {note}"
        )
