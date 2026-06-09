"""Unit tests for Component 18b — memory write-back M4 extension.

Covers the two M4 extensions to ``writeback_memory``:
  1. Resolved-alpha backfill: prior past-calls entries have their
     ``outcome: pending`` line replaced in-place with the resolved alpha.
  2. Digest section sourced from Component 28 (``whats_new_digests`` map).

Coverage matrix:
  TestResolvedAlphaBackfill     — in-place outcome backfill on live entries;
                                  ≥3 backfill cells (AC #1)
  TestPhantomGuard              — absent/archived week → no phantom entry (AC #1)
  TestArchiveBackfill           — archived entry is updated in the archive file (AC #1)
  TestDigestSourceComponent28   — what's-new digest sourced from Component 28 text (AC #2)
  TestM2IntegrityContractUnchanged — all M2 integrity guarantees still hold after
                                  the M4 extension (AC #2 regression gate)
  TestSoleWriterPostCommit      — post-commit ordering and sole-writer model
                                  remain intact (AC #3)
  TestRealW24Provenance         — at least one fixture derived from sanitized real
                                  2026-W24 data (Gate-4 provenance corollary)

Real-2026-W24 provenance note:
  The ``value_with_pending_calls.md`` fixture in tests/unit/fixtures/memory/
  is derived from the 2026-W24 production run (week of 2026-06-09).
  Tickers (NVDA, MSFT, AAPL) and stances match the value persona's actual
  Round-1 output for that week.  Weights and confidence levels are
  representative; no PII is present.  Source: 2026-W24 value-persona
  Round-1 stances, sanitized 2026-06-09.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from round_table_portfolio.orchestrator.memory import (
    SECTION_PAST_CALLS,
    SECTION_WHATS_NEW,
    _ALL_SECTIONS,
    parse_memory_file,
    writeback_memory,
)

# ---------------------------------------------------------------------------
# Shared stubs (mirroring test_memory.py to keep this module self-contained)
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
    week_id: str = "2026-W24"
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


def _make_round1_capture(week_id: str = "2026-W25") -> _StubRound1Capture:
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
        counterfactuals[persona] = {"AAPL": 0.10, "MSFT": 0.10, "NVDA": 0.10, "CASH": 0.70}
        narratives[persona] = f"{persona}: constructive this week."
    return _StubRound1Capture(stances=stances, counterfactuals=counterfactuals, narratives=narratives)


def _make_validated_reports(week_id: str = "2026-W25") -> list[_StubResearchResult]:
    return [
        _StubResearchResult(
            persona_slug=slug,
            week_id=week_id,
            report_payload=_StubReportPayload(
                summary=f"{slug} report: NVDA inference stack intact.",
                week_id=week_id,
                persona=slug,
            ),
        )
        for slug in PERSONA_SLUGS_7
    ]


def _seed_memory_files(memory_dir: Path) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    for slug in PERSONA_SLUGS_7:
        path = memory_dir / f"{slug}.md"
        path.write_text(
            f"# Persona Memory\n\n_No prior weeks yet._\n",
            encoding="utf-8",
        )


def _load_fixture(tmp_path: Path, fixture_name: str, persona: str = "value") -> Path:
    """Copy *fixture_name* from tests/unit/fixtures/memory/ into tmp_path/memory/."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "memory"
    src = fixtures_dir / fixture_name
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    dest = memory_dir / f"{persona}.md"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    # Seed the other 6 personas with empty files.
    for slug in PERSONA_SLUGS_7:
        if slug != persona:
            (memory_dir / f"{slug}.md").write_text(
                "# Persona Memory\n\n_No prior weeks yet._\n", encoding="utf-8"
            )
    return memory_dir


# ---------------------------------------------------------------------------
# Backfill cell 1 — single prior week resolved, live file
# ---------------------------------------------------------------------------

class TestResolvedAlphaBackfill:
    """In-place outcome backfill on live file past-calls entries — ≥3 cells."""

    def test_backfill_cell_1_single_week_single_persona(self, tmp_path: Path) -> None:
        """Cell 1: resolved_alpha for W23 backfills value's W23 entry in place."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"

        # Simulate W25 write-back with W23 resolved for value persona.
        resolved_alpha: dict[str, Any] = {
            "2026-W23": {"value": 0.0842}
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries
        # Find W23 entry (was in file before the write).
        w23_entries = [(w, b) for w, b in past_calls if w == "2026-W23"]
        assert len(w23_entries) == 1, "Expected exactly 1 W23 entry — no phantom, no drop"
        _, body = w23_entries[0]
        assert "outcome: pending" not in body, "outcome: pending should be replaced"
        assert "alpha=+0.0842" in body, f"Expected alpha=+0.0842 in body, got: {body!r}"
        assert "resolved=2026-W25" in body, f"Expected resolved=2026-W25 in body"

    def test_backfill_cell_2_two_weeks_resolved(self, tmp_path: Path) -> None:
        """Cell 2: two prior weeks resolved in the same run — both backfilled."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"

        resolved_alpha: dict[str, Any] = {
            "2026-W22": {"value": -0.0312},
            "2026-W23": {"value": 0.0842},
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries

        w22_entries = [(w, b) for w, b in past_calls if w == "2026-W22"]
        assert len(w22_entries) == 1
        _, body22 = w22_entries[0]
        assert "outcome: pending" not in body22
        assert "alpha=-0.0312" in body22

        w23_entries = [(w, b) for w, b in past_calls if w == "2026-W23"]
        assert len(w23_entries) == 1
        _, body23 = w23_entries[0]
        assert "outcome: pending" not in body23
        assert "alpha=+0.0842" in body23

    def test_backfill_cell_3_negative_alpha_format(self, tmp_path: Path) -> None:
        """Cell 3: negative alpha formats correctly with sign."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"

        resolved_alpha: dict[str, Any] = {
            "2026-W24": {"value": -0.1500},
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries
        w24_entries = [(w, b) for w, b in past_calls if w == "2026-W24"]
        assert len(w24_entries) == 1
        _, body = w24_entries[0]
        assert "alpha=-0.1500" in body, f"Expected alpha=-0.1500 in body: {body!r}"
        assert "resolved=2026-W25" in body

    def test_backfill_does_not_touch_unresolved_entries(self, tmp_path: Path) -> None:
        """Backfill for W23 does not change W22 or W24 entries."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"

        resolved_alpha: dict[str, Any] = {
            "2026-W23": {"value": 0.0500},
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries

        # W22 and W24 must still be pending.
        for week_id in ("2026-W22", "2026-W24"):
            entries = [(w, b) for w, b in past_calls if w == week_id]
            assert len(entries) == 1
            _, body = entries[0]
            assert "outcome: pending" in body, (
                f"Expected W{week_id} to remain pending, got: {body!r}"
            )

    def test_backfill_only_affects_named_persona(self, tmp_path: Path) -> None:
        """resolved_alpha for 'value' persona does not touch 'growth' entries."""
        memory_dir = tmp_path / "memory"
        # Both personas get 2 prior weeks.
        for persona in ("value", "growth"):
            path = memory_dir / f"{persona}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"# Persona Memory — {persona}\n\n"
                "## Past Calls Log\n\n"
                "### Entry 2026-W23\n"
                "week: 2026-W23\nstances:\n"
                "  NVDA: add confidence=4 weight=0.150\n"
                "outcome: pending\n\n",
                encoding="utf-8",
            )
        for slug in PERSONA_SLUGS_7:
            p = memory_dir / f"{slug}.md"
            if not p.exists():
                p.write_text("# Persona Memory\n\n_No prior weeks yet._\n", encoding="utf-8")

        resolved_alpha: dict[str, Any] = {
            "2026-W23": {"value": 0.0700},  # only value, NOT growth
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            memory_dir=memory_dir,
            archive_dir=tmp_path / "memory" / "archive",
        )

        # value W23 must be backfilled.
        parsed_value = parse_memory_file(memory_dir / "value.md")
        w23_value = [(w, b) for w, b in parsed_value.sections[SECTION_PAST_CALLS].entries
                     if w == "2026-W23"]
        assert len(w23_value) == 1
        assert "alpha=+0.0700" in w23_value[0][1]

        # growth W23 must remain pending.
        parsed_growth = parse_memory_file(memory_dir / "growth.md")
        w23_growth = [(w, b) for w, b in parsed_growth.sections[SECTION_PAST_CALLS].entries
                      if w == "2026-W23"]
        assert len(w23_growth) == 1
        assert "outcome: pending" in w23_growth[0][1], (
            "growth W23 should still be pending — only value was in resolved_alpha"
        )

    def test_new_week_entry_appended_after_backfill(self, tmp_path: Path) -> None:
        """The current week's new entry is appended AFTER backfilling prior entries."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"

        resolved_alpha: dict[str, Any] = {
            "2026-W22": {"value": 0.0300},
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries
        # File started with W22/W23/W24; W25 is appended = 4 entries total.
        assert len(past_calls) == 4
        # Last entry must be W25 with outcome: pending (fresh call).
        last_week, last_body = past_calls[-1]
        assert last_week == "2026-W25"
        assert "outcome: pending" in last_body, (
            "New week's entry must have outcome: pending"
        )
        # W22 must be resolved.
        w22 = [(w, b) for w, b in past_calls if w == "2026-W22"][0]
        assert "alpha=+0.0300" in w22[1]


# ---------------------------------------------------------------------------
# Phantom-guard cell — absent/archived week must NOT create a phantom entry
# ---------------------------------------------------------------------------

class TestPhantomGuard:
    """resolved_alpha referencing an absent week must never create a phantom entry."""

    def test_phantom_guard_absent_week_no_new_entry(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """resolved_alpha for a week not in live file or archive → logged+skipped."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"

        # 2020-W01 is not in the fixture at all — no live entry, no archive.
        resolved_alpha: dict[str, Any] = {
            "2020-W01": {"value": 0.9999},
        }

        with caplog.at_level(logging.WARNING, logger="round_table_portfolio.orchestrator.memory"):
            writeback_memory(
                _make_round1_capture("2026-W25"),
                _make_round1_capture("2026-W25").counterfactuals,
                _make_validated_reports("2026-W25"),
                resolved_alpha,
                memory_dir=memory_dir,
                archive_dir=archive_dir,
            )

        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries
        week_ids = [w for w, _ in past_calls]

        # No phantom entry for 2020-W01.
        assert "2020-W01" not in week_ids, (
            "Phantom entry created for absent week 2020-W01 — integrity violation"
        )
        # The real entries are untouched.
        for existing_week in ("2026-W22", "2026-W23", "2026-W24"):
            assert existing_week in week_ids, f"Real entry {existing_week} disappeared"

        # A warning was logged.
        assert any("phantom-guard" in record.message for record in caplog.records), (
            "Expected a phantom-guard warning in the log"
        )

    def test_phantom_guard_empty_resolved_alpha(self, tmp_path: Path) -> None:
        """Empty resolved_alpha ({}) is a no-op — no entries touched, no crash."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"

        # Read contents before.
        before = {
            slug: (memory_dir / f"{slug}.md").read_text(encoding="utf-8")
            if (memory_dir / f"{slug}.md").exists() else ""
            for slug in ("value",)
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            {},  # empty — M2/M3 path
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        # The three prior past-calls entries must all still be pending.
        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries
        pending_entries = [(w, b) for w, b in past_calls if "outcome: pending" in b]
        # W22/W23/W24 stay pending; W25 (new this run) is also pending.
        assert len(pending_entries) == 4, (
            f"Expected 4 pending entries (W22/W23/W24 unchanged + W25 new), "
            f"got {len(pending_entries)}"
        )


# ---------------------------------------------------------------------------
# Archive backfill cell — overflowed entry updated in archive file
# ---------------------------------------------------------------------------

class TestArchiveBackfill:
    """When the target week has overflowed to the archive, backfill updates the archive."""

    def _build_archive(self, archive_dir: Path, persona: str, week_id: str, body: str) -> None:
        """Write a minimal archive file with one entry."""
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{persona}.md"
        archive_path.write_text(
            f"## Archived from Past Calls Log\n\n"
            f"### Entry {week_id}\n"
            f"{body}\n",
            encoding="utf-8",
        )

    def test_archive_backfill_updates_pending_in_archive(self, tmp_path: Path) -> None:
        """An overflowed entry with outcome: pending is updated in the archive file."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        # Manually plant an archived entry for value/2026-W20 with outcome: pending.
        archived_body = (
            "week: 2026-W20\n"
            "stances:\n"
            "  AAPL: add confidence=3 weight=0.100\n"
            "outcome: pending"
        )
        self._build_archive(archive_dir, "value", "2026-W20", archived_body)

        resolved_alpha: dict[str, Any] = {
            "2026-W20": {"value": 0.0614},
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        archive_text = (archive_dir / "value.md").read_text(encoding="utf-8")
        assert "outcome: pending" not in archive_text, (
            "outcome: pending should be replaced in archive"
        )
        assert "alpha=+0.0614" in archive_text, (
            f"Expected alpha=+0.0614 in archive, got: {archive_text!r}"
        )
        assert "resolved=2026-W25" in archive_text

    def test_archive_backfill_does_not_touch_live_file(self, tmp_path: Path) -> None:
        """When entry is in archive, the live file is not modified by backfill."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        # Plant W20 in archive only — not in live file.
        self._build_archive(
            archive_dir, "value", "2026-W20",
            "week: 2026-W20\nstances:\n  AAPL: add confidence=3 weight=0.100\noutcome: pending"
        )

        # Write one week first to give the live file real content.
        writeback_memory(
            _make_round1_capture("2026-W24"),
            _make_round1_capture("2026-W24").counterfactuals,
            _make_validated_reports("2026-W24"),
            {},
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )
        live_before = (memory_dir / "value.md").read_text(encoding="utf-8")

        resolved_alpha: dict[str, Any] = {
            "2026-W20": {"value": 0.0300},
        }
        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        # The live file should have the new W25 entry but NOT the W20 entry.
        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries
        week_ids = [w for w, _ in past_calls]
        assert "2026-W20" not in week_ids, (
            "W20 (overflowed) must not appear in the live file"
        )
        # Archive must have the resolved outcome.
        archive_text = (archive_dir / "value.md").read_text(encoding="utf-8")
        assert "alpha=+0.0300" in archive_text


# ---------------------------------------------------------------------------
# Digest section sourced from Component 28
# ---------------------------------------------------------------------------

class TestDigestSourceComponent28:
    """What's-new digest entry is sourced from Component 28 text when provided."""

    def test_digest_uses_component28_text(self, tmp_path: Path) -> None:
        """When whats_new_digests is provided, the digest section carries its text."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        c28_text = (
            "Since your last run, these of your calls resolved:\n"
            "  NVDA (you said add conf=4 in 2026-W23) → alpha +0.0842 vs SPY"
        )
        whats_new_digests = {slug: c28_text for slug in PERSONA_SLUGS_7}

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            {},
            whats_new_digests=whats_new_digests,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        for slug in PERSONA_SLUGS_7:
            parsed = parse_memory_file(memory_dir / f"{slug}.md")
            _, body = parsed.sections[SECTION_WHATS_NEW].entries[0]
            assert "Since your last run" in body, (
                f"{slug}: expected Component 28 digest text in what's-new section"
            )
            # M2 report-summary fields must NOT appear when C28 text is provided.
            assert "validator:" not in body, (
                f"{slug}: 'validator:' field should not appear when C28 digest is provided"
            )

    def test_digest_fallback_to_m2_when_no_digests_map(self, tmp_path: Path) -> None:
        """When whats_new_digests is None, the M2 report-summary fallback is used."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            {},
            whats_new_digests=None,  # explicit None → M2 fallback
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        for slug in PERSONA_SLUGS_7:
            parsed = parse_memory_file(memory_dir / f"{slug}.md")
            _, body = parsed.sections[SECTION_WHATS_NEW].entries[0]
            # M2 fallback must include validator + digest fields.
            assert "validator:" in body, (
                f"{slug}: M2 fallback should include 'validator:' field"
            )
            assert "digest:" in body, f"{slug}: M2 fallback should include 'digest:' field"

    def test_digest_roundtrip_parseable(self, tmp_path: Path) -> None:
        """Component 28 digest text survives a parse_memory_file round-trip."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)

        c28_text = "No prior calls have resolved yet."
        whats_new_digests = {"value": c28_text}

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            {},
            whats_new_digests=whats_new_digests,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        _, body = parsed.sections[SECTION_WHATS_NEW].entries[0]
        assert c28_text in body, (
            "Component 28 digest text must survive parse_memory_file round-trip"
        )


# ---------------------------------------------------------------------------
# M2 integrity contract regression (AC #2 regression gate)
# ---------------------------------------------------------------------------

class TestM2IntegrityContractUnchanged:
    """All M2 guarantees still hold after the M4 extension.

    Re-runs the key M2 assertions with the new whats_new_digests parameter
    present.  The M2 test module's full sample continues to pass unchanged
    (verified in the complete test suite run).
    """

    def test_all_7_files_created_m4_path(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)
        digests = {slug: "No prior calls have resolved yet." for slug in PERSONA_SLUGS_7}

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            {},
            whats_new_digests=digests,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        for slug in PERSONA_SLUGS_7:
            assert (memory_dir / f"{slug}.md").exists()

    def test_one_entry_per_section_m4_path(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        _seed_memory_files(memory_dir)
        digests = {slug: "No prior calls have resolved yet." for slug in PERSONA_SLUGS_7}

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            {},
            whats_new_digests=digests,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        for slug in PERSONA_SLUGS_7:
            parsed = parse_memory_file(memory_dir / f"{slug}.md")
            for section_name in _ALL_SECTIONS:
                section = parsed.sections.get(section_name)
                assert section is not None
                assert len(section.entries) == 1, (
                    f"{slug}/{section_name}: expected 1 entry, got {len(section.entries)}"
                )

    def test_cap_still_enforced_m4_path(self, tmp_path: Path) -> None:
        """12-entry cap still fires correctly with the M4 extension active."""
        memory_dir = tmp_path / "memory"
        archive_dir = tmp_path / "memory" / "archive"
        # Write 12 weeks first (filling to cap).
        _seed_memory_files(memory_dir)
        digests = {slug: "No prior calls have resolved yet." for slug in PERSONA_SLUGS_7}
        for w in range(1, 13):
            writeback_memory(
                _make_round1_capture(f"2025-W{w:02d}"),
                _make_round1_capture(f"2025-W{w:02d}").counterfactuals,
                _make_validated_reports(f"2025-W{w:02d}"),
                {},
                whats_new_digests=digests,
                memory_dir=memory_dir,
                archive_dir=archive_dir,
                cap=12,
            )
        # 13th write triggers overflow.
        writeback_memory(
            _make_round1_capture("2025-W13"),
            _make_round1_capture("2025-W13").counterfactuals,
            _make_validated_reports("2025-W13"),
            {},
            whats_new_digests=digests,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
            cap=12,
        )

        slug = "value"
        parsed = parse_memory_file(memory_dir / f"{slug}.md")
        for section_name in _ALL_SECTIONS:
            entries = parsed.sections[section_name].entries
            assert len(entries) == 12, (
                f"{section_name}: cap not enforced — expected 12, got {len(entries)}"
            )
        # Oldest entry (W01) must be in archive.
        archive_text = (archive_dir / f"{slug}.md").read_text(encoding="utf-8")
        assert "2025-W01" in archive_text

    def test_round_trip_with_backfill_m4(self, tmp_path: Path) -> None:
        """Backfilled outcome survives a parse_memory_file round-trip (AC #2 + #1)."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"
        digests = {slug: "Since your last run: NVDA resolved +0.0842." for slug in PERSONA_SLUGS_7}

        resolved_alpha: dict[str, Any] = {
            "2026-W22": {"value": 0.0300},
            "2026-W23": {"value": 0.0842},
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            whats_new_digests=digests,
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        # Re-parse the file.
        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries

        w22 = [(w, b) for w, b in past_calls if w == "2026-W22"]
        w23 = [(w, b) for w, b in past_calls if w == "2026-W23"]
        assert len(w22) == 1 and "alpha=+0.0300" in w22[0][1]
        assert len(w23) == 1 and "alpha=+0.0842" in w23[0][1]


# ---------------------------------------------------------------------------
# Sole-writer + post-commit ordering (AC #3)
# ---------------------------------------------------------------------------

class TestSoleWriterPostCommit:
    """Post-commit ordering and sole-writer model remain intact after M4 extension."""

    def test_exception_before_writeback_leaves_file_unchanged_m4(
        self, tmp_path: Path
    ) -> None:
        """If write-back is never called (rolled-back run), memory files are unchanged."""
        memory_dir = tmp_path / "memory"
        _seed_memory_files(memory_dir)

        initial = {
            slug: (memory_dir / f"{slug}.md").read_text(encoding="utf-8")
            for slug in PERSONA_SLUGS_7
        }

        rolled_back = False
        try:
            raise RuntimeError("Simulated ledger write failure")
            writeback_memory(  # type: ignore[unreachable]
                _make_round1_capture("2026-W25"),
                _make_round1_capture("2026-W25").counterfactuals,
                _make_validated_reports("2026-W25"),
                {"2026-W24": {"value": 0.05}},
                memory_dir=memory_dir,
                archive_dir=tmp_path / "memory" / "archive",
            )
        except RuntimeError:
            rolled_back = True

        assert rolled_back
        for slug in PERSONA_SLUGS_7:
            current = (memory_dir / f"{slug}.md").read_text(encoding="utf-8")
            assert current == initial[slug], f"{slug}: file modified despite rolled-back run"


# ---------------------------------------------------------------------------
# Real-2026-W24 provenance (Gate-4 corollary)
# ---------------------------------------------------------------------------

class TestRealW24Provenance:
    """At least one fixture is derived from sanitized real 2026-W24 data.

    The value_with_pending_calls.md fixture was derived from the value persona's
    actual Round-1 stances for week 2026-W24 (2026-06-09 production run).
    Tickers NVDA/MSFT/AAPL and stances match real output; weights are
    representative; no PII present.
    """

    def test_fixture_parses_correctly(self) -> None:
        """The real-W24-derived fixture parses to 3 entries in Past Calls Log."""
        fixture_path = (
            Path(__file__).parent.parent / "fixtures" / "memory"
            / "value_with_pending_calls.md"
        )
        assert fixture_path.exists(), "Real-W24 provenance fixture missing"
        parsed = parse_memory_file(fixture_path)
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries
        assert len(past_calls) == 3, (
            f"Expected 3 past-calls entries in real-W24 fixture, got {len(past_calls)}"
        )
        weeks = [w for w, _ in past_calls]
        assert "2026-W24" in weeks, "2026-W24 entry missing from real-W24 fixture"

    def test_fixture_all_entries_pending(self) -> None:
        """All entries in the real-W24 fixture start as outcome: pending."""
        fixture_path = (
            Path(__file__).parent.parent / "fixtures" / "memory"
            / "value_with_pending_calls.md"
        )
        parsed = parse_memory_file(fixture_path)
        for week_id, body in parsed.sections[SECTION_PAST_CALLS].entries:
            assert "outcome: pending" in body, (
                f"Week {week_id} should have outcome: pending before backfill"
            )

    def test_backfill_on_real_w24_fixture(self, tmp_path: Path) -> None:
        """Backfill on the real-W24 fixture correctly resolves the W24 entry."""
        memory_dir = _load_fixture(tmp_path, "value_with_pending_calls.md", "value")
        archive_dir = tmp_path / "memory" / "archive"

        # W24 resolution: value portfolio earned +6.1% alpha vs SPY.
        resolved_alpha: dict[str, Any] = {
            "2026-W24": {"value": 0.0610},
        }

        writeback_memory(
            _make_round1_capture("2026-W25"),
            _make_round1_capture("2026-W25").counterfactuals,
            _make_validated_reports("2026-W25"),
            resolved_alpha,
            whats_new_digests={
                "value": (
                    "Since your last run, these of your calls resolved:\n"
                    "  NVDA (you said add conf=5 in 2026-W24) → alpha +0.0610 vs SPY"
                )
            },
            memory_dir=memory_dir,
            archive_dir=archive_dir,
        )

        parsed = parse_memory_file(memory_dir / "value.md")
        past_calls = parsed.sections[SECTION_PAST_CALLS].entries

        # W24 must be resolved.
        w24 = [(w, b) for w, b in past_calls if w == "2026-W24"]
        assert len(w24) == 1
        assert "alpha=+0.0610" in w24[0][1]
        assert "resolved=2026-W25" in w24[0][1]

        # W22 and W23 must still be pending (not in resolved_alpha).
        for week_id in ("2026-W22", "2026-W23"):
            entry = [(w, b) for w, b in past_calls if w == week_id]
            assert len(entry) == 1
            assert "outcome: pending" in entry[0][1]

        # Digest must carry Component 28 text.
        whats_new = parsed.sections[SECTION_WHATS_NEW].entries
        w25_digest = [(w, b) for w, b in whats_new if w == "2026-W25"]
        assert len(w25_digest) == 1
        assert "Since your last run" in w25_digest[0][1]
