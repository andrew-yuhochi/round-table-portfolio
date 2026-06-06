"""Unit tests for validation_harness.build_research_validation_report.

All tests are SKIP_LIVE-safe — no network calls, no subagent dispatch, no DB
writes.  Synthetic ``PersonaResearchResult`` objects are constructed inline.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from round_table_portfolio.personas.output_validator import (
    ReportValidationResult,
)
from round_table_portfolio.research.runner import (
    PersonaOutputSchema,
    PersonaReportPayload,
    PersonaResearchResult,
    PersonaShortlistRow,
    ShortlistEntry,
)
from round_table_portfolio.research.validation_harness import (
    _split_primaries_and_peers,
    build_research_validation_report,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FIXED_TS = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
_WEEK_ID = "2026-06-02"


def _make_shortlist_rows(
    persona: str,
    week_id: str,
    entries: list[tuple[str, list[str]]],
) -> list[PersonaShortlistRow]:
    """Build shortlist rows from (primary_ticker, [peer_ticker, ...]) tuples."""
    rows: list[PersonaShortlistRow] = []
    for primary, peers in entries:
        rows.append(
            PersonaShortlistRow(
                week_id=week_id,
                persona=persona,
                ticker=primary,
                is_cluster_peer=0,
                parent_ticker=None,
            )
        )
        for peer in peers:
            rows.append(
                PersonaShortlistRow(
                    week_id=week_id,
                    persona=persona,
                    ticker=peer,
                    is_cluster_peer=1,
                    parent_ticker=primary,
                )
            )
    return rows


def _make_parsed_output(
    web_searches: int = 5,
    data_tool_calls: int = 3,
    shortlist_entries: list[tuple[str, list[str]]] | None = None,
) -> PersonaOutputSchema:
    if shortlist_entries is None:
        shortlist_entries = [("BRK.B", ["LNC", "MET"])]
    shortlist = [
        ShortlistEntry(ticker=t, why="strong fundamentals", cluster=peers)
        for t, peers in shortlist_entries
    ]
    return PersonaOutputSchema(
        shortlist=shortlist,
        report="This is a detailed research report. Free-cash-flow yield is high.",
        web_searches_used=web_searches,
        data_tool_calls_used=data_tool_calls,
    )


def _make_budget_summary(
    web_searches_used: int = 5,
    web_cap: int = 10,
    data_calls_used: int = 3,
    data_cap: int = 8,
) -> dict:
    return {
        "web_searches": {
            "used": web_searches_used,
            "cap": web_cap,
            "exhausted": web_searches_used >= web_cap,
        },
        "data_tool_calls": {
            "used": data_calls_used,
            "cap": data_cap,
            "exhausted": data_calls_used >= data_cap,
        },
    }


def _make_result(
    persona_slug: str = "value",
    passed: bool = True,
    stage: str = "LLM_JUDGE",
    notes: str = "Report is on-mandate.",
    llm_justification: str = "Identified BRK.B as a deep-value play citing P/B of 1.1x.",
    web_searches: int = 5,
    data_tool_calls: int = 3,
    budget_overrun: bool = False,
    shortlist_entries: list[tuple[str, list[str]]] | None = None,
    week_id: str = _WEEK_ID,
) -> PersonaResearchResult:
    if shortlist_entries is None:
        shortlist_entries = [("BRK.B", ["LNC", "MET"]), ("C", [])]
    parsed = _make_parsed_output(
        web_searches=web_searches,
        data_tool_calls=data_tool_calls,
        shortlist_entries=shortlist_entries,
    )
    shortlist_rows = _make_shortlist_rows(persona_slug, week_id, shortlist_entries)
    return PersonaResearchResult(
        persona_slug=persona_slug,
        week_id=week_id,
        parsed_output=parsed,
        validation=ReportValidationResult(
            passed=passed,
            notes=notes,
            stage=stage,
            llm_justification=llm_justification,
        ),
        report_payload=PersonaReportPayload(
            week_id=week_id,
            persona=persona_slug,
            summary="Detailed value research summary.",
            validator_passed=1 if passed else 0,
            validator_notes=notes,
            full_report_path=f"state/reports/{week_id}/{persona_slug}.md",
        ),
        shortlist_rows=shortlist_rows,
        budget_summary=_make_budget_summary(
            web_searches_used=web_searches,
            data_calls_used=data_tool_calls,
        ),
        budget_overrun=budget_overrun,
    )


# ---------------------------------------------------------------------------
# Tests — cluster reconstruction helper
# ---------------------------------------------------------------------------


class TestSplitPrimariesAndPeers:
    def test_single_primary_with_peers(self) -> None:
        rows = _make_shortlist_rows("value", _WEEK_ID, [("BRK.B", ["LNC", "MET"])])
        primaries, peer_map = _split_primaries_and_peers(rows)
        assert [p.ticker for p in primaries] == ["BRK.B"]
        assert peer_map["BRK.B"] == ["LNC", "MET"]

    def test_multiple_primaries_no_peers(self) -> None:
        rows = _make_shortlist_rows("growth", _WEEK_ID, [("NVDA", []), ("MSFT", [])])
        primaries, peer_map = _split_primaries_and_peers(rows)
        assert {p.ticker for p in primaries} == {"NVDA", "MSFT"}
        assert peer_map.get("NVDA", []) == []
        assert peer_map.get("MSFT", []) == []

    def test_empty_rows(self) -> None:
        primaries, peer_map = _split_primaries_and_peers([])
        assert primaries == []
        assert peer_map == {}

    def test_peers_grouped_under_correct_parent(self) -> None:
        rows = _make_shortlist_rows(
            "technical", _WEEK_ID, [("AAPL", ["QCOM"]), ("MSFT", ["GOOGL", "META"])]
        )
        _, peer_map = _split_primaries_and_peers(rows)
        assert peer_map["AAPL"] == ["QCOM"]
        assert peer_map["MSFT"] == ["GOOGL", "META"]


# ---------------------------------------------------------------------------
# Tests — build_research_validation_report
# ---------------------------------------------------------------------------


class TestBuildResearchValidationReport:
    def test_writes_to_correct_path(self, tmp_path: Path) -> None:
        results = [_make_result("value")]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        assert path == tmp_path / "reports" / _WEEK_ID / "research_validation.md"
        assert path.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        results = [_make_result("value")]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        assert isinstance(path, Path)

    def test_report_contains_week_id(self, tmp_path: Path) -> None:
        results = [_make_result("value")]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert _WEEK_ID in text

    def test_report_contains_generation_timestamp(self, tmp_path: Path) -> None:
        results = [_make_result("value")]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "2026-06-02 12:00:00 UTC" in text

    def test_pass_rate_correct_all_pass(self, tmp_path: Path) -> None:
        results = [
            _make_result("value", passed=True),
            _make_result("growth", passed=True),
            _make_result("technical", passed=True),
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "3/3 PASS" in text

    def test_pass_rate_correct_with_one_fail(self, tmp_path: Path) -> None:
        results = [
            _make_result("value", passed=True),
            _make_result("growth", passed=False),
            _make_result("technical", passed=True),
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "2/3 PASS" in text

    def test_each_persona_present_in_report(self, tmp_path: Path) -> None:
        slugs = ["value", "growth", "technical"]
        results = [_make_result(s) for s in slugs]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        for slug in slugs:
            assert slug.upper() in text

    def test_fail_persona_shows_fail_badge(self, tmp_path: Path) -> None:
        results = [
            _make_result(
                "growth",
                passed=False,
                notes="STRUCTURAL GATE FAIL — missing metric terms",
                llm_justification="",
            )
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "FAIL" in text
        assert "STRUCTURAL GATE FAIL" in text

    def test_pass_persona_shows_pass_badge(self, tmp_path: Path) -> None:
        results = [_make_result("value", passed=True)]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "PASS" in text

    def test_llm_justification_included(self, tmp_path: Path) -> None:
        justification = "Identified BRK.B as deep-value citing P/B of 1.1x and FCF yield 9%."
        results = [_make_result("value", llm_justification=justification)]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert justification in text

    def test_notes_included(self, tmp_path: Path) -> None:
        notes = "Report is on-mandate with specific valuation metrics cited."
        results = [_make_result("value", notes=notes)]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert notes in text

    def test_primary_tickers_in_report(self, tmp_path: Path) -> None:
        results = [
            _make_result(
                "value",
                shortlist_entries=[("BRK.B", ["LNC"]), ("C", [])],
            )
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "BRK.B" in text
        assert "C" in text

    def test_cluster_peers_grouped_under_primary(self, tmp_path: Path) -> None:
        results = [
            _make_result(
                "value",
                shortlist_entries=[("BRK.B", ["LNC", "MET"])],
            )
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        # BRK.B row should be followed by its peers on the same table row
        assert "LNC" in text
        assert "MET" in text
        # Peers appear in the same row as their primary in the shortlist table.
        # The shortlist table uses **BRK.B** (bold), distinguishing it from the
        # validator section which may reference "BRK.B" in plain prose.
        brk_line = next(
            (ln for ln in text.splitlines() if "**BRK.B**" in ln and "|" in ln), None
        )
        assert brk_line is not None, "BRK.B not found as a bold shortlist table row"
        assert "LNC" in brk_line
        assert "MET" in brk_line

    def test_budget_numbers_in_report(self, tmp_path: Path) -> None:
        results = [_make_result("value", web_searches=7, data_tool_calls=4)]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "7" in text  # web searches used
        assert "4" in text  # data tool calls used

    def test_total_web_searches_summed(self, tmp_path: Path) -> None:
        results = [
            _make_result("value", web_searches=5),
            _make_result("growth", web_searches=8),
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "13" in text  # 5 + 8

    def test_total_data_tool_calls_summed(self, tmp_path: Path) -> None:
        results = [
            _make_result("value", data_tool_calls=3),
            _make_result("growth", data_tool_calls=6),
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "9" in text  # 3 + 6

    def test_budget_overrun_flag_shown(self, tmp_path: Path) -> None:
        results = [_make_result("value", budget_overrun=True)]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "BUDGET OVERRUN" in text

    def test_no_budget_overrun_flag_when_clean(self, tmp_path: Path) -> None:
        results = [_make_result("value", budget_overrun=False)]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "BUDGET OVERRUN" not in text

    def test_wall_clock_shown_when_provided(self, tmp_path: Path) -> None:
        results = [_make_result("value")]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            wall_clock_seconds={"value": 142.3},
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "142.3" in text

    def test_wall_clock_not_recorded_when_slug_absent(self, tmp_path: Path) -> None:
        results = [_make_result("value")]
        # wall_clock_seconds provided but "value" slug missing from it
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            wall_clock_seconds={"growth": 200.0},
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "not recorded" in text

    def test_wall_clock_not_recorded_when_none(self, tmp_path: Path) -> None:
        results = [_make_result("value")]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            wall_clock_seconds=None,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "not recorded" in text

    def test_no_crash_on_empty_llm_justification(self, tmp_path: Path) -> None:
        results = [_make_result("value", llm_justification="")]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        # Should render a placeholder, not crash
        assert "not run" in text

    def test_stable_persona_ordering(self, tmp_path: Path) -> None:
        # Insert out of order; report must follow roster order.
        results = [
            _make_result("technical"),
            _make_result("value"),
            _make_result("growth"),
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        value_pos = text.index("## VALUE")
        growth_pos = text.index("## GROWTH")
        technical_pos = text.index("## TECHNICAL")
        assert value_pos < growth_pos < technical_pos

    def test_unknown_slug_appended_alphabetically(self, tmp_path: Path) -> None:
        results = [
            _make_result("value"),
            _make_result("z-new-persona"),
        ]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        value_pos = text.index("## VALUE")
        new_pos = text.index("## Z-NEW-PERSONA")
        assert value_pos < new_pos

    def test_stage_shown_in_validator_section(self, tmp_path: Path) -> None:
        results = [_make_result("value", stage="LLM_JUDGE")]
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "LLM_JUDGE" in text

    def test_no_crash_on_empty_shortlist(self, tmp_path: Path) -> None:
        result = _make_result("value", shortlist_entries=[])
        path = build_research_validation_report(
            results=[result],
            week_id=_WEEK_ID,
            state_root=tmp_path,
            generated_at=_FIXED_TS,
        )
        text = path.read_text()
        assert "No shortlist entries" in text

    def test_creates_parent_dirs_if_missing(self, tmp_path: Path) -> None:
        results = [_make_result("value")]
        new_root = tmp_path / "deep" / "nested"
        path = build_research_validation_report(
            results=results,
            week_id=_WEEK_ID,
            state_root=new_root,
            generated_at=_FIXED_TS,
        )
        assert path.exists()
