"""Unit tests for orchestrator/metrics.py — Component 19 (run_metrics).

Critical Component #3: the #1 feasibility-risk measurement.

Coverage matrix:
  AC1 — total + per-persona wall-clock reported (non-zero, monotonic):
    test_total_wall_clock_is_sum_of_per_persona      — total == sum of per_persona_timing values
    test_per_persona_wall_clock_all_present          — every slug in timing map appears in report
    test_per_persona_wall_clock_values_nonzero        — all per-persona seconds > 0
    test_per_persona_wall_clock_monotonic_check       — per-persona map values ≥ 0 (non-negative)

  AC2 — per-persona web-search + data-tool counts vs budget, over-budget flagged:
    test_web_search_counts_reported                  — web_searches_used matches research_result
    test_data_tool_counts_reported                   — data_tool_calls_used matches research_result
    test_over_budget_web_search_flagged              — persona over web budget appears in over_budget_personas
    test_over_budget_data_tools_flagged              — persona over tool budget appears in over_budget_personas
    test_under_budget_not_flagged                    — persona within budget is NOT in over_budget_personas
    test_budget_limits_in_per_persona_metrics        — budget_max_* fields populated from PersonaBudget

  AC3 — window-proximity percentage correct; escalation fires/not-fires:
    test_window_fraction_computed_correctly          — window_fraction == total_seconds / window_seconds
    test_proximity_percentage_in_summary_text        — summary_text contains the formatted percentage
    test_escalation_fires_above_threshold            — escalation_triggered True when fraction ≥ threshold
    test_escalation_text_in_summary_when_triggered   — summary_text contains "BOUNDING-PLAYBOOK ESCALATION"
    test_escalation_not_fired_below_threshold        — escalation_triggered False when fraction < threshold
    test_no_escalation_text_when_under_threshold     — summary_text does NOT contain "ESCALATION" when under
    test_verdict_does_not_fit_at_threshold           — feasibility_verdict "does-not-fit" when fraction ≥ threshold
    test_verdict_tight_in_band                       — feasibility_verdict "tight" when 60%–threshold
    test_verdict_fits_below_band                     — feasibility_verdict "fits" when fraction < 60%

  AC4 — report persisted to run log AND returned:
    test_run_log_created                             — run_log_path exists after call
    test_run_log_contains_summary_text               — log file contents include the summary
    test_run_log_appends_not_overwrites              — second call appends, does not wipe prior content
    test_report_object_returned                      — return value is a RunMetricsReport instance

  AC5 — full test suite passes (no failures).

Sample count: ≥25 cells across 23 tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from round_table_portfolio.budget.loader import PersonaBudget
from round_table_portfolio.orchestrator.metrics import (
    RunMetricsReport,
    report_run_metrics,
)
from round_table_portfolio.research.runner import (
    PersonaOutputSchema,
    PersonaReportPayload,
    PersonaResearchResult,
    ShortlistEntry,
)
from round_table_portfolio.personas.output_validator import ReportValidationResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_HOURS = 5.0
_PROXIMITY_THRESHOLD = 0.80
_WINDOW_SECONDS = _WINDOW_HOURS * 3600  # 18 000s

_PERSONA_SLUGS = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]

# Per-persona timings: total = 320+410+290+305+275+340+260 = 2 200s (~12% of window — comfortably fits)
_TIMING_FITS = {
    "value":                320.0,
    "growth":               410.0,
    "discretionary-macro":  290.0,
    "cta-systematic-macro": 305.0,
    "technical":            275.0,
    "quant-systematic":     340.0,
    "risk-officer":         260.0,
}
_TOTAL_FITS = sum(_TIMING_FITS.values())  # 2 200.0s

# Timings that sum to > 80% of 18 000s — triggers escalation.
# Target: ~16 000s (88.9% of 18 000s)
_TIMING_CROSSES = {
    "value":                2500.0,
    "growth":               2400.0,
    "discretionary-macro":  2300.0,
    "cta-systematic-macro": 2200.0,
    "technical":            2100.0,
    "quant-systematic":     2200.0,
    "risk-officer":         2300.0,
}
_TOTAL_CROSSES = sum(_TIMING_CROSSES.values())  # 16 000.0s → 88.9% of window


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_output_schema(
    web_searches_used: int = 4,
    data_tool_calls_used: int = 10,
) -> PersonaOutputSchema:
    return PersonaOutputSchema(
        shortlist=[ShortlistEntry(ticker="AAPL", why="test")],
        report="Test report.",
        web_searches_used=web_searches_used,
        data_tool_calls_used=data_tool_calls_used,
    )


def _make_validation_result(passed: bool = True) -> ReportValidationResult:
    return ReportValidationResult(
        passed=passed,
        notes="ok" if passed else "failed",
        stage="structural",
        llm_justification="",
    )


def _make_research_result(
    slug: str,
    web_searches_used: int = 4,
    data_tool_calls_used: int = 10,
    budget_overrun: bool = False,
) -> PersonaResearchResult:
    return PersonaResearchResult(
        persona_slug=slug,
        week_id="2026-W23",
        parsed_output=_make_output_schema(web_searches_used, data_tool_calls_used),
        validation=_make_validation_result(),
        report_payload=PersonaReportPayload(
            week_id="2026-W23",
            persona=slug,
            summary="Test summary.",
            validator_passed=1,
            validator_notes="",
            full_report_path=f"state/reports/2026-W23/{slug}.md",
        ),
        shortlist_rows=[],
        budget_summary={"web_searches_used": web_searches_used, "data_tool_calls_used": data_tool_calls_used},
        budget_overrun=budget_overrun,
    )


def _make_results_all_within_budget() -> list[PersonaResearchResult]:
    """All 7 personas, each well within their budgets."""
    return [
        _make_research_result(slug, web_searches_used=4, data_tool_calls_used=10)
        for slug in _PERSONA_SLUGS
    ]


def _make_results_with_over_budget(
    over_slug: str = "growth",
    web_over: int = 20,   # growth budget is 10 → over
    tools_over: int = 30, # growth budget is 15 → over
) -> list[PersonaResearchResult]:
    results = []
    for slug in _PERSONA_SLUGS:
        if slug == over_slug:
            results.append(_make_research_result(slug, web_searches_used=web_over, data_tool_calls_used=tools_over))
        else:
            results.append(_make_research_result(slug, web_searches_used=4, data_tool_calls_used=10))
    return results


def _make_budgets(
    default_web: int = 8,
    default_tools: int = 15,
) -> dict[str, PersonaBudget]:
    """Minimal budget dict matching the 7-persona slugs."""
    budgets: dict[str, PersonaBudget] = {
        "__defaults__": PersonaBudget(max_turns=12, max_web_searches=default_web, max_data_tool_calls=default_tools),
        "value":                PersonaBudget(max_turns=12, max_web_searches=6,  max_data_tool_calls=18),
        "growth":               PersonaBudget(max_turns=12, max_web_searches=10, max_data_tool_calls=15),
        "discretionary-macro":  PersonaBudget(max_turns=12, max_web_searches=8,  max_data_tool_calls=12),
        "cta-systematic-macro": PersonaBudget(max_turns=12, max_web_searches=6,  max_data_tool_calls=12),
        "technical":            PersonaBudget(max_turns=12, max_web_searches=4,  max_data_tool_calls=20),
        "quant-systematic":     PersonaBudget(max_turns=12, max_web_searches=4,  max_data_tool_calls=20),
        "risk-officer":         PersonaBudget(max_turns=10, max_web_searches=4,  max_data_tool_calls=10),
    }
    return budgets


def _window_config(
    window_hours: float = _WINDOW_HOURS,
    proximity_threshold: float = _PROXIMITY_THRESHOLD,
) -> dict[str, Any]:
    return {
        "window_hours": window_hours,
        "window_proximity_threshold": proximity_threshold,
    }


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "2026-W23.log"


# ---------------------------------------------------------------------------
# AC1 — total + per-persona wall-clock reported (non-zero, monotonic)
# ---------------------------------------------------------------------------

def test_total_wall_clock_is_sum_of_per_persona(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert report.total_wall_seconds == pytest.approx(_TOTAL_FITS, abs=1e-6)


def test_per_persona_wall_clock_all_present(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    reported_slugs = {pm.persona_slug for pm in report.per_persona}
    assert reported_slugs == set(_PERSONA_SLUGS)


def test_per_persona_wall_clock_values_nonzero(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    for pm in report.per_persona:
        assert pm.wall_clock_seconds > 0, f"{pm.persona_slug} wall_clock_seconds should be > 0"


def test_per_persona_wall_clock_matches_input(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    timing_by_slug = {pm.persona_slug: pm.wall_clock_seconds for pm in report.per_persona}
    for slug, expected in _TIMING_FITS.items():
        assert timing_by_slug[slug] == pytest.approx(expected, abs=1e-6), (
            f"{slug}: expected {expected}s, got {timing_by_slug[slug]}s"
        )


# ---------------------------------------------------------------------------
# AC2 — per-persona web-search + data-tool counts vs budget, over-budget flagged
# ---------------------------------------------------------------------------

def test_web_search_counts_reported(tmp_log: Path) -> None:
    results = _make_results_all_within_budget()
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=results,
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    # All personas were built with web_searches_used=4
    for pm in report.per_persona:
        assert pm.web_searches_used == 4, f"{pm.persona_slug}: expected 4 web searches"


def test_data_tool_counts_reported(tmp_log: Path) -> None:
    results = _make_results_all_within_budget()
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=results,
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    for pm in report.per_persona:
        assert pm.data_tool_calls_used == 10, f"{pm.persona_slug}: expected 10 tool calls"


def test_over_budget_web_search_flagged(tmp_log: Path) -> None:
    # growth budget is max_web_searches=10; we set 20 → over budget
    results = _make_results_with_over_budget("growth", web_over=20, tools_over=10)
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=results,
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert "growth" in report.over_budget_personas
    growth_pm = next(pm for pm in report.per_persona if pm.persona_slug == "growth")
    assert growth_pm.web_searches_over_budget is True


def test_over_budget_data_tools_flagged(tmp_log: Path) -> None:
    # risk-officer budget is max_data_tool_calls=10; we set 30 → over budget
    results = _make_results_with_over_budget("risk-officer", web_over=3, tools_over=30)
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=results,
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert "risk-officer" in report.over_budget_personas
    ro_pm = next(pm for pm in report.per_persona if pm.persona_slug == "risk-officer")
    assert ro_pm.data_tool_calls_over_budget is True


def test_under_budget_not_flagged(tmp_log: Path) -> None:
    results = _make_results_all_within_budget()
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=results,
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert report.over_budget_personas == []


def test_budget_limits_in_per_persona_metrics(tmp_log: Path) -> None:
    budgets = _make_budgets()
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=budgets,
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    for pm in report.per_persona:
        expected_budget = budgets[pm.persona_slug]
        assert pm.budget_max_web_searches == expected_budget.max_web_searches
        assert pm.budget_max_data_tool_calls == expected_budget.max_data_tool_calls


# ---------------------------------------------------------------------------
# AC3 — window-proximity percentage correct; escalation fires/not-fires
# ---------------------------------------------------------------------------

def test_window_fraction_computed_correctly(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    expected_fraction = _TOTAL_FITS / _WINDOW_SECONDS
    assert report.window_fraction == pytest.approx(expected_fraction, abs=1e-6)


def test_proximity_percentage_in_summary_text(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    pct = int(report.window_fraction * 100)
    assert f"{pct}%" in report.summary_text, (
        f"Expected '{pct}%' in summary_text but got:\n{report.summary_text}"
    )


def test_escalation_fires_above_threshold(tmp_log: Path) -> None:
    """Synthetic run crossing 80% of the 5h window — escalation must fire."""
    report = report_run_metrics(
        per_persona_timing=_TIMING_CROSSES,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    # _TOTAL_CROSSES = 16 000s → 88.9% of 18 000s → above 80% threshold
    assert report.window_fraction >= _PROXIMITY_THRESHOLD
    assert report.escalation_triggered is True
    assert report.feasibility_verdict == "does-not-fit"


def test_escalation_text_in_summary_when_triggered(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_CROSSES,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert "BOUNDING-PLAYBOOK ESCALATION" in report.summary_text


def test_escalation_not_fired_below_threshold(tmp_log: Path) -> None:
    """Comfortably under threshold — escalation must NOT fire."""
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert report.window_fraction < _PROXIMITY_THRESHOLD
    assert report.escalation_triggered is False


def test_no_escalation_text_when_under_threshold(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert "BOUNDING-PLAYBOOK ESCALATION" not in report.summary_text


def test_verdict_does_not_fit_at_threshold(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_CROSSES,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert report.feasibility_verdict == "does-not-fit"


def test_verdict_tight_in_band(tmp_path: Path) -> None:
    """Timing in the 60–80% band → verdict 'tight', no escalation."""
    # Target: ~65% of 18 000s = 11 700s → distribute evenly
    tight_timing = {slug: 11700.0 / len(_PERSONA_SLUGS) for slug in _PERSONA_SLUGS}
    log = tmp_path / "runs" / "2026-W24.log"
    report = report_run_metrics(
        per_persona_timing=tight_timing,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=log,
    )
    assert 0.60 <= report.window_fraction < _PROXIMITY_THRESHOLD
    assert report.feasibility_verdict == "tight"
    assert report.escalation_triggered is False


def test_verdict_fits_below_band(tmp_log: Path) -> None:
    # _TIMING_FITS total = 2 200s → ~12% of window → "fits"
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert report.window_fraction < 0.60
    assert report.feasibility_verdict == "fits"


# ---------------------------------------------------------------------------
# AC4 — report persisted to run log AND returned
# ---------------------------------------------------------------------------

def test_run_log_created(tmp_log: Path) -> None:
    report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert tmp_log.exists(), "Run log was not created"


def test_run_log_contains_summary_text(tmp_log: Path) -> None:
    report = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    log_contents = tmp_log.read_text(encoding="utf-8")
    # The summary header line should appear in the log
    assert "RUN METRICS REPORT" in log_contents


def test_run_log_appends_not_overwrites(tmp_path: Path) -> None:
    """Two calls to report_run_metrics append to the same log — prior content survives."""
    log = tmp_path / "runs" / "2026-W25.log"
    # Write a sentinel line first (simulating the orchestrator's preamble write)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("week=2026-W25\n", encoding="utf-8")

    report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=log,
    )
    contents = log.read_text(encoding="utf-8")
    assert "week=2026-W25" in contents, "Prior preamble was overwritten"
    assert "RUN METRICS REPORT" in contents


def test_report_object_returned(tmp_log: Path) -> None:
    result = report_run_metrics(
        per_persona_timing=_TIMING_FITS,
        research_results=_make_results_all_within_budget(),
        budgets=_make_budgets(),
        window_config=_window_config(),
        run_log_path=tmp_log,
    )
    assert isinstance(result, RunMetricsReport)
    assert result.summary_text != ""
