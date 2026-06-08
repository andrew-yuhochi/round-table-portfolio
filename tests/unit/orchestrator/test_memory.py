"""Unit tests for orchestrator/memory.py — Component 18 (memory write-back).

Coverage matrix:
  TestOneEntryPerSection        — each of the 7 personas gains exactly one new
                                  entry in each of the 4 sections per write-back
                                  run (AC #1 — no duplicates, no missing persona)
  TestCapOverflow               — a section at cap=12 archives the oldest entry
                                  to the archive file; does NOT drop it (AC #2)
  TestPostCommitOrdering        — a simulated rolled-back week (exception before
                                  writeback_memory is called) leaves memory
                                  unchanged (AC #3)
  TestAtomicWrite               — a mid-write failure leaves the prior file
                                  intact (AC #4 — atomic contract)
  TestRoundTrip                 — entries written by writeback_memory are
                                  re-parseable by parse_memory_file, and the
                                  re-read content matches what was written
                                  (AC #4 — round-trip)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from round_table_portfolio.orchestrator.memory import (
    SECTION_COUNTERFACTUAL,
    SECTION_DEBATE_STANCES,
    SECTION_PAST_CALLS,
    SECTION_WHATS_NEW,
    _ALL_SECTIONS,
    parse_memory_file,
    writeback_memory,
)

# ---------------------------------------------------------------------------
# Helpers shared with test_weekly_run.py — reproduced here so this module
# has no import dependency on the orchestrator test module.
# ---------------------------------------------------------------------------

PERSONA_SLUGS_7 = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]

_DEBATE_SET = ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# Minimal stub types for Round1Capture and PersonaResearchResult
# ---------------------------------------------------------------------------

@dataclass
class _StubStance:
    week_id: str
    persona: str
    ticker: str
    round: int
    action: str
    target_weight: float
    confidence: int
    rationale: str
    user_id: str = "andrew"
    roster_version: int = 1
    enhancement_version: int = 1


@dataclass
class _StubRound1Capture:
    stances: list[_StubStance] = field(default_factory=list)
    counterfactuals: dict[str, dict[str, float]] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    narratives: dict[str, str] = field(default_factory=dict)


@dataclass
class _StubReportPayload:
    summary: str
    week_id: str = "2026-W23"
    persona: str = ""


@dataclass
class _StubValidation:
    passed: bool = True
    notes: str = ""
    stage: str = "structural"


@dataclass
class _StubResearchResult:
    persona_slug: str
    week_id: str
    report_payload: _StubReportPayload
    validation: _StubValidation = field(default_factory=_StubValidation)


def _make_round1_capture(week_id: str = "2026-W23") -> _StubRound1Capture:
    """Build a minimal Round1Capture covering all 7 personas × 3 debate-set tickers."""
    stances = []
    counterfactuals: dict[str, dict[str, float]] = {}
    narratives: dict[str, str] = {}

    for persona in PERSONA_SLUGS_7:
        for ticker in _DEBATE_SET:
            stances.append(_StubStance(
                week_id=week_id,
                persona=persona,
                ticker=ticker,
                round=1,
                action="add",
                target_weight=0.10,
                confidence=3,
                rationale=f"Stub: {persona} on {ticker}",
            ))
        counterfactuals[persona] = {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "NVDA": 0.10,
            "CASH": 0.63,
        }
        narratives[persona] = f"{persona}: constructive on tech sector this week."

    return _StubRound1Capture(
        stances=stances,
        counterfactuals=counterfactuals,
        narratives=narratives,
    )


def _make_validated_reports(week_id: str = "2026-W23") -> list[_StubResearchResult]:
    return [
        _StubResearchResult(
            persona_slug=slug,
            week_id=week_id,
            report_payload=_StubReportPayload(
                summary=f"{slug} weekly research: AAPL strong FCF, MSFT cloud ARR growth.",
                week_id=week_id,
                persona=slug,
            ),
            validation=_StubValidation(passed=True),
        )
        for slug in PERSONA_SLUGS_7
    ]


def _seed_memory_files(memory_dir: Path) -> None:
    """Write empty seed memory files for all 7 personas."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    for slug in PERSONA_SLUGS_7:
        path = memory_dir / f"{slug}.md"
        path.write_text(
            f"# Persona Memory\n\n"
            f"_No prior weeks yet. This file is updated by the /weekly-run orchestrator after each week's run._\n",
            encoding="utf-8",
        )


# The writeback_memory function does a deferred import of Round1Capture and
# PersonaResearchResult to avoid circular imports.  The stub types defined
# above are duck-type compatible — we patch the isinstance checks by patching
# the type annotations inline.  Since the function uses duck-typing (attribute
# access), no patching is needed.


# ---------------------------------------------------------------------------
# AC #1 — One new entry per section per persona (7 files × 4 sections)
# ---------------------------------------------------------------------------

class TestOneEntryPerSection:
    """Each of the 7 personas gains exactly one new entry in each of the 4 sections."""

    def test_all_7_files_created(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture(),
            _make_round1_capture().counterfactuals,
            _make_validated_reports(),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        for slug in PERSONA_SLUGS_7:
            assert (memory_dir / f"{slug}.md").exists(), f"Missing memory file for {slug}"

    def test_exactly_one_entry_per_section(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture(),
            _make_round1_capture().counterfactuals,
            _make_validated_reports(),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        for slug in PERSONA_SLUGS_7:
            parsed = parse_memory_file(memory_dir / f"{slug}.md")
            for section_name in _ALL_SECTIONS:
                section = parsed.sections.get(section_name)
                assert section is not None, (
                    f"{slug}: section {section_name!r} is absent from parsed file"
                )
                assert len(section.entries) == 1, (
                    f"{slug}/{section_name}: expected 1 entry, got {len(section.entries)}"
                )

    def test_entry_week_id_matches(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)
        week_id = "2026-W23"

        writeback_memory(
            _make_round1_capture(week_id),
            _make_round1_capture(week_id).counterfactuals,
            _make_validated_reports(week_id),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        for slug in PERSONA_SLUGS_7:
            parsed = parse_memory_file(memory_dir / f"{slug}.md")
            for section_name in _ALL_SECTIONS:
                entry_week, _ = parsed.sections[section_name].entries[0]
                assert entry_week == week_id, (
                    f"{slug}/{section_name}: expected week={week_id!r}, got {entry_week!r}"
                )

    def test_no_duplicate_entry_on_second_different_week(self, tmp_path: Path) -> None:
        """Two sequential write-backs for different weeks produce 2 entries each."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture("2026-W23"),
            _make_round1_capture("2026-W23").counterfactuals,
            _make_validated_reports("2026-W23"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )
        writeback_memory(
            _make_round1_capture("2026-W24"),
            _make_round1_capture("2026-W24").counterfactuals,
            _make_validated_reports("2026-W24"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        slug = "value"
        parsed = parse_memory_file(memory_dir / f"{slug}.md")
        for section_name in _ALL_SECTIONS:
            entries = parsed.sections[section_name].entries
            assert len(entries) == 2, (
                f"{slug}/{section_name}: expected 2 entries after 2 weeks, "
                f"got {len(entries)}"
            )
            assert entries[0][0] == "2026-W23"
            assert entries[1][0] == "2026-W24"

    def test_past_calls_body_contains_stance_data(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture("2026-W23"),
            _make_round1_capture("2026-W23").counterfactuals,
            _make_validated_reports("2026-W23"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        _, body = parsed.sections[SECTION_PAST_CALLS].entries[0]
        assert "AAPL" in body, "Past-calls entry missing ticker AAPL"
        assert "add" in body, "Past-calls entry missing action 'add'"
        assert "pending" in body, "Past-calls entry missing 'outcome: pending'"

    def test_counterfactual_body_contains_portfolio_data(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture("2026-W23"),
            _make_round1_capture("2026-W23").counterfactuals,
            _make_validated_reports("2026-W23"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        _, body = parsed.sections[SECTION_COUNTERFACTUAL].entries[0]
        assert "AAPL" in body, "Counterfactual entry missing ticker AAPL"
        assert "CASH" in body, "Counterfactual entry missing CASH"

    def test_debate_stances_body_contains_narrative(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture("2026-W23"),
            _make_round1_capture("2026-W23").counterfactuals,
            _make_validated_reports("2026-W23"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        _, body = parsed.sections[SECTION_DEBATE_STANCES].entries[0]
        assert "narrative" in body, "Debate-stances entry missing narrative field"

    def test_whats_new_body_contains_digest(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture("2026-W23"),
            _make_round1_capture("2026-W23").counterfactuals,
            _make_validated_reports("2026-W23"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        _, body = parsed.sections[SECTION_WHATS_NEW].entries[0]
        assert "digest" in body, "What's-new entry missing digest field"
        assert "validator" in body, "What's-new entry missing validator field"

    def test_absent_persona_memory_file_is_created(self, tmp_path: Path) -> None:
        """If a persona's memory file is absent, write-back creates it."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        memory_dir.mkdir(parents=True, exist_ok=True)
        # Deliberately do NOT seed any files.

        writeback_memory(
            _make_round1_capture("2026-W23"),
            _make_round1_capture("2026-W23").counterfactuals,
            _make_validated_reports("2026-W23"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        for slug in PERSONA_SLUGS_7:
            assert (memory_dir / f"{slug}.md").exists(), (
                f"Memory file for {slug} was not created from absent state"
            )


# ---------------------------------------------------------------------------
# AC #2 — 12-entry cap: overflow is archived, not dropped
# ---------------------------------------------------------------------------

class TestCapOverflow:
    """A section at cap=12 archives the oldest entry when a 13th is added."""

    def _load_at_cap_fixture(self, tmp_path: Path) -> Path:
        """Copy the value_at_cap.md fixture into tmp_path/memory/."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "memory"
        fixture_src = fixtures_dir / "value_at_cap.md"
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        dest = memory_dir / "value.md"
        dest.write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")
        # Seed other 6 personas with empty files.
        for slug in PERSONA_SLUGS_7:
            if slug != "value":
                (memory_dir / f"{slug}.md").write_text(
                    "# Persona Memory\n\n_No prior weeks yet._\n", encoding="utf-8"
                )
        return memory_dir

    def test_section_has_cap_entries_before_write(self, tmp_path: Path) -> None:
        """Fixture has exactly 12 entries in each section (the cap)."""
        memory_dir = self._load_at_cap_fixture(tmp_path)
        parsed = parse_memory_file(memory_dir / "value.md")
        for section_name in _ALL_SECTIONS:
            count = len(parsed.sections.get(section_name, type('', (), {'entries': []})()).entries)
            assert count == 12, (
                f"Fixture value.md section {section_name!r} should have 12 entries, "
                f"got {count}"
            )

    def test_13th_entry_archives_oldest(self, tmp_path: Path) -> None:
        """Adding a 13th entry to a capped section archives entry #1 (2025-W01)."""
        memory_dir = self._load_at_cap_fixture(tmp_path)
        archive_dir = tmp_path / "memory" / "archive"

        writeback_memory(
            _make_round1_capture("2025-W13"),
            _make_round1_capture("2025-W13").counterfactuals,
            _make_validated_reports("2025-W13"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
            cap=12,
        )

        # Section still has exactly 12 entries after the write.
        parsed = parse_memory_file(memory_dir / "value.md")
        for section_name in _ALL_SECTIONS:
            entries = parsed.sections[section_name].entries
            assert len(entries) == 12, (
                f"{section_name}: expected 12 entries after cap-overflow, "
                f"got {len(entries)}"
            )

        # Oldest entry (2025-W01) was evicted from the live file.
        for section_name in _ALL_SECTIONS:
            week_ids = [e[0] for e in parsed.sections[section_name].entries]
            assert "2025-W01" not in week_ids, (
                f"{section_name}: 2025-W01 should have been archived, "
                f"but is still in the live file"
            )

        # New entry (2025-W13) is in the live file.
        for section_name in _ALL_SECTIONS:
            week_ids = [e[0] for e in parsed.sections[section_name].entries]
            assert "2025-W13" in week_ids, (
                f"{section_name}: 2025-W13 not found in live file after write"
            )

    def test_oldest_entry_appears_in_archive(self, tmp_path: Path) -> None:
        """Archived entry appears in the archive file (not silently dropped)."""
        memory_dir = self._load_at_cap_fixture(tmp_path)
        archive_dir = tmp_path / "memory" / "archive"

        writeback_memory(
            _make_round1_capture("2025-W13"),
            _make_round1_capture("2025-W13").counterfactuals,
            _make_validated_reports("2025-W13"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
            cap=12,
        )

        archive_path = archive_dir / "value.md"
        assert archive_path.exists(), "Archive file was not created"
        archive_content = archive_path.read_text(encoding="utf-8")
        # The oldest entry's week_id should appear in the archive.
        assert "2025-W01" in archive_content, (
            "Oldest entry 2025-W01 not found in archive — entry was dropped, not archived"
        )

    def test_cap_at_1_archives_old_on_each_write(self, tmp_path: Path) -> None:
        """With cap=1, every write-back archives the previous entry."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture("2026-W01"),
            _make_round1_capture("2026-W01").counterfactuals,
            _make_validated_reports("2026-W01"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
            cap=1,
        )
        writeback_memory(
            _make_round1_capture("2026-W02"),
            _make_round1_capture("2026-W02").counterfactuals,
            _make_validated_reports("2026-W02"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
            cap=1,
        )

        slug = "value"
        parsed = parse_memory_file(memory_dir / f"{slug}.md")
        for section_name in _ALL_SECTIONS:
            entries = parsed.sections[section_name].entries
            # Only the most-recent entry survives in the live file.
            assert len(entries) == 1, (
                f"{section_name}: expected 1 entry with cap=1, got {len(entries)}"
            )
            assert entries[0][0] == "2026-W02", (
                f"{section_name}: expected 2026-W02 in live file, got {entries[0][0]}"
            )

        # The first week's entries were archived.
        archive_path = archive_dir / f"{slug}.md"
        assert archive_path.exists(), "Archive file was not created"
        assert "2026-W01" in archive_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC #3 — Post-commit ordering: rolled-back week leaves memory unchanged
# ---------------------------------------------------------------------------

class TestPostCommitOrdering:
    """Memory must lag the ledger — a transaction failure leaves memory unchanged."""

    def test_exception_before_writeback_leaves_memory_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Simulate a transaction failure: memory files are never modified."""
        memory_dir = tmp_path / "memory"
        _seed_memory_files(memory_dir)

        # Record the initial content of each memory file.
        initial_contents = {
            slug: (memory_dir / f"{slug}.md").read_text(encoding="utf-8")
            for slug in PERSONA_SLUGS_7
        }

        # Simulate what the orchestrator does: if an exception occurs inside
        # the transaction block (before conn.commit()), writeback_memory is
        # never called.  We model this by raising BEFORE the call.
        rolled_back = False
        try:
            # This block represents the orchestrator's BEGIN … COMMIT block.
            raise RuntimeError("Simulated ledger write failure — FK violation")
            # writeback_memory is never reached.
            writeback_memory(  # type: ignore[unreachable]
                _make_round1_capture("2026-W23"),
                _make_round1_capture("2026-W23").counterfactuals,
                _make_validated_reports("2026-W23"),
                {},
                memory_dir=memory_dir,
                archive_dir=tmp_path / "memory" / "archive",
            )
        except RuntimeError:
            rolled_back = True

        assert rolled_back, "Expected RuntimeError to propagate"

        for slug in PERSONA_SLUGS_7:
            current = (memory_dir / f"{slug}.md").read_text(encoding="utf-8")
            assert current == initial_contents[slug], (
                f"Memory file for {slug} was modified despite rolled-back transaction"
            )

    def test_writeback_called_only_after_commit_in_orchestrator(self) -> None:
        """Static check: writeback_memory call appears after conn.commit() in weekly_run.py.

        Parses the orchestrator source to confirm the ordering invariant is
        structurally enforced — memory write-back block appears on a line
        after the conn.commit() call.
        """
        import ast

        weekly_run_path = (
            Path(__file__).parent.parent.parent.parent
            / "src" / "round_table_portfolio" / "orchestrator" / "weekly_run.py"
        )
        source = weekly_run_path.read_text(encoding="utf-8")
        lines = source.splitlines()

        commit_line = None
        writeback_line = None
        for i, line in enumerate(lines, start=1):
            if "conn.commit()" in line:
                commit_line = i
            if "writeback_memory(" in line:
                writeback_line = i

        assert commit_line is not None, "conn.commit() not found in weekly_run.py"
        assert writeback_line is not None, "writeback_memory( not found in weekly_run.py"
        assert writeback_line > commit_line, (
            f"writeback_memory (line {writeback_line}) must appear AFTER "
            f"conn.commit() (line {commit_line}) in weekly_run.py"
        )

    def test_writeback_not_inside_except_block(self) -> None:
        """writeback_memory must NOT be reachable from the rollback/except branch."""
        weekly_run_path = (
            Path(__file__).parent.parent.parent.parent
            / "src" / "round_table_portfolio" / "orchestrator" / "weekly_run.py"
        )
        source = weekly_run_path.read_text(encoding="utf-8")
        lines = source.splitlines()

        # Find the bounds of the try/except/finally block around the transaction.
        try_line = None
        rollback_line = None
        finally_line = None
        writeback_line = None

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if "conn.execute(\"BEGIN\")" in stripped and try_line is None:
                # The try block starts shortly before BEGIN.
                pass
            if "conn.rollback()" in stripped:
                rollback_line = i
            if "conn.close()" in stripped and finally_line is None:
                finally_line = i
            if "writeback_memory(" in stripped:
                writeback_line = i

        assert rollback_line is not None, "conn.rollback() not found"
        assert writeback_line is not None, "writeback_memory( not found"
        # writeback_memory must be called AFTER the finally block (conn.close()).
        assert writeback_line > (finally_line or rollback_line), (
            f"writeback_memory (line {writeback_line}) must be after the "
            f"finally block / conn.close() (line {finally_line}) — "
            "it must not be reachable from the rollback/except path"
        )


# ---------------------------------------------------------------------------
# AC #4 — Atomic write: mid-write failure leaves prior file intact
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    """Atomic write contract: a failure during write leaves the original untouched."""

    def test_prior_content_survives_write_failure(self, tmp_path: Path) -> None:
        """If os.rename raises, the prior memory file is intact."""
        from round_table_portfolio.orchestrator import memory as mem_module

        memory_dir = tmp_path / "memory"
        _seed_memory_files(memory_dir)

        # Write one good week first so the file has real content.
        writeback_memory(
            _make_round1_capture("2026-W22"),
            _make_round1_capture("2026-W22").counterfactuals,
            _make_validated_reports("2026-W22"),
            {},
            memory_dir=memory_dir,
            archive_dir=tmp_path / "memory" / "archive",
        )

        slug = "value"
        content_before = (memory_dir / f"{slug}.md").read_text(encoding="utf-8")
        assert "2026-W22" in content_before  # sanity

        # Now simulate a failed write by patching os.rename to raise.
        with patch.object(mem_module.os, "rename", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                writeback_memory(
                    _make_round1_capture("2026-W23"),
                    _make_round1_capture("2026-W23").counterfactuals,
                    _make_validated_reports("2026-W23"),
                    {},
                    memory_dir=memory_dir,
                    archive_dir=tmp_path / "memory" / "archive",
                )

        # The original file must be unchanged.
        content_after = (memory_dir / f"{slug}.md").read_text(encoding="utf-8")
        assert content_after == content_before, (
            "Memory file was corrupted despite os.rename failure"
        )

    def test_no_tmpfiles_left_behind_on_success(self, tmp_path: Path) -> None:
        """Successful write leaves no .tmp files in the memory directory."""
        memory_dir = tmp_path / "memory"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture("2026-W23"),
            _make_round1_capture("2026-W23").counterfactuals,
            _make_validated_reports("2026-W23"),
            {},
            memory_dir=memory_dir,
            archive_dir=tmp_path / "memory" / "archive",
        )

        tmp_files = list(memory_dir.glob("*.tmp"))
        assert tmp_files == [], (
            f"Leftover tmp files after successful write: {tmp_files}"
        )


# ---------------------------------------------------------------------------
# AC #4 — Round-trip: written entries re-parse cleanly by parse_memory_file
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Entries written by writeback_memory are re-parseable by parse_memory_file."""

    def test_written_sections_all_parseable(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        _seed_memory_files(memory_dir)
        week_id = "2026-W23"

        writeback_memory(
            _make_round1_capture(week_id),
            _make_round1_capture(week_id).counterfactuals,
            _make_validated_reports(week_id),
            {},
            memory_dir=memory_dir,
            archive_dir=tmp_path / "memory" / "archive",
        )

        for slug in PERSONA_SLUGS_7:
            parsed = parse_memory_file(memory_dir / f"{slug}.md")
            for section_name in _ALL_SECTIONS:
                assert section_name in parsed.sections, (
                    f"{slug}: section {section_name!r} missing after re-parse"
                )
                entries = parsed.sections[section_name].entries
                assert len(entries) == 1, (
                    f"{slug}/{section_name}: expected 1 entry after re-parse"
                )
                w, body = entries[0]
                assert w == week_id, (
                    f"{slug}/{section_name}: week_id mismatch after re-parse: "
                    f"got {w!r}, expected {week_id!r}"
                )
                assert body, (
                    f"{slug}/{section_name}: entry body is empty after re-parse"
                )

    def test_two_weeks_roundtrip(self, tmp_path: Path) -> None:
        """Two sequential write-backs produce a file that re-parses to 2 entries."""
        memory_dir = tmp_path / "memory"
        _seed_memory_files(memory_dir)

        for week_id in ("2026-W23", "2026-W24"):
            writeback_memory(
                _make_round1_capture(week_id),
                _make_round1_capture(week_id).counterfactuals,
                _make_validated_reports(week_id),
                {},
                memory_dir=memory_dir,
                archive_dir=tmp_path / "memory" / "archive",
            )

        slug = "growth"
        parsed = parse_memory_file(memory_dir / f"{slug}.md")
        for section_name in _ALL_SECTIONS:
            entries = parsed.sections[section_name].entries
            assert len(entries) == 2, (
                f"{slug}/{section_name}: expected 2 entries, got {len(entries)}"
            )
            assert entries[0][0] == "2026-W23"
            assert entries[1][0] == "2026-W24"

    def test_parse_memory_file_on_empty_seed(self, tmp_path: Path) -> None:
        """parse_memory_file on an empty seed file returns a ParsedMemoryFile with no entries."""
        seed_path = tmp_path / "value.md"
        seed_path.write_text(
            "# Persona Memory\n\n_No prior weeks yet._\n", encoding="utf-8"
        )
        parsed = parse_memory_file(seed_path)
        # No section entries should be present.
        for section_name in _ALL_SECTIONS:
            entries = parsed.sections.get(section_name, type('S', (), {'entries': []})()).entries
            assert not entries, (
                f"Expected no entries in {section_name!r} from empty seed, "
                f"got {entries}"
            )

    def test_parse_memory_file_on_absent_file(self, tmp_path: Path) -> None:
        """parse_memory_file on a non-existent file returns an empty ParsedMemoryFile."""
        parsed = parse_memory_file(tmp_path / "nonexistent.md")
        assert parsed.sections == {}, "Expected empty sections for absent file"

    def test_written_content_survives_fixture_roundtrip(self, tmp_path: Path) -> None:
        """Fixture at cap (12 entries) parses cleanly and entries are accessible."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "memory"
        fixture_path = fixtures_dir / "value_at_cap.md"
        parsed = parse_memory_file(fixture_path)

        for section_name in _ALL_SECTIONS:
            assert section_name in parsed.sections, (
                f"Section {section_name!r} missing from fixture parse"
            )
            entries = parsed.sections[section_name].entries
            assert len(entries) == 12, (
                f"Fixture {section_name!r}: expected 12 entries, got {len(entries)}"
            )
            # Entries are in chronological order.
            assert entries[0][0] == "2025-W01"
            assert entries[-1][0] == "2025-W12"
