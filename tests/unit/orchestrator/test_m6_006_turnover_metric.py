"""Unit tests for TASK-M6-006 — Consensus one-way turnover metric.

Coverage matrix (TDD Component 19 M6 Extension, Sample Selection):

  Formula fixtures (hand-computed, deterministic — Gate 4 ≥4 fixtures):
    test_identical_books_zero_turnover          — identical books → 0.0%
    test_full_replacement_100_pct_turnover      — complete swap → 100.0%
    test_partial_change_hand_computed           — partial change → exact hand-calc
    test_cash_move_included_in_turnover         — CASH 24%→0% contributes correctly
    test_first_week_na                          — prior={} → turnover_pct=None, N/A text

  Real-data anchor (W24→W25 committed fixture):
    test_w24_w25_anchor                         — reproduces the ~48.45% figure
    test_w24_w25_breakdown_counts               — added=19, removed=17, re-weighted=19

  Surfacing checks — preview line:
    test_preview_line_present_normal_week       — preview_line contains "Consensus turnover"
    test_preview_line_na_first_week             — first week preview_line says N/A

  Surfacing checks — run log:
    test_log_line_present_normal_week           — log_line contains turnover figure
    test_log_line_na_first_week                 — first week log_line says N/A

  Integration with report_run_metrics:
    test_report_run_metrics_turnover_field      — RunMetricsReport.turnover populated
    test_report_run_metrics_preview_in_summary  — summary_text contains turnover line
    test_report_run_metrics_log_file_has_turnover — run log file contains turnover line
    test_report_run_metrics_backward_compat     — omitting holdings → turnover=None (no crash)

  No-tripwire assertion:
    test_no_tripwire_run_completes_any_turnover — report returned regardless of high turnover;
                                                   no exception, no field blocking the return

Total deterministic cells: 16 (≥ TDD minimum of ≥4 fixtures + 1 anchor + 1 N/A + 2 surfacing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from round_table_portfolio.orchestrator.metrics import (
    RunMetricsReport,
    TurnoverResult,
    compute_consensus_turnover,
    report_run_metrics,
)
from round_table_portfolio.budget.loader import PersonaBudget
from round_table_portfolio.research.runner import (
    PersonaOutputSchema,
    PersonaReportPayload,
    PersonaResearchResult,
    ShortlistEntry,
)
from round_table_portfolio.personas.output_validator import ReportValidationResult

# Import the committed W24→W25 fixture.
from tests.unit.fixtures.consensus.w24_w25_consensus_pair import (
    W24_HOLDINGS,
    W25_HOLDINGS,
    EXPECTED_TURNOVER_PCT,
    EXPECTED_N_ADDED,
    EXPECTED_N_REMOVED,
    EXPECTED_N_REWEIGHTED,
)


# ---------------------------------------------------------------------------
# Hand-computed formula fixtures
# ---------------------------------------------------------------------------


def test_identical_books_zero_turnover() -> None:
    """Identical books → turnover = 0.0%."""
    book = {"AAPL": 0.30, "MSFT": 0.25, "CASH": 0.45}
    result = compute_consensus_turnover(book, book)
    assert result.turnover_pct == pytest.approx(0.0, abs=1e-9)
    assert result.n_added == 0
    assert result.n_removed == 0
    assert result.n_reweighted == 0


def test_full_replacement_100_pct_turnover() -> None:
    """Complete portfolio swap: sell A/B, buy C/D → one-way turnover = 100%.

    old = {A: 0.60, B: 0.40}   (no CASH for simplicity — fully invested)
    new = {C: 0.70, D: 0.30}
    |Δw|: A=0.60, B=0.40, C=0.70, D=0.30  → sum=2.00 → 0.5*2.00=1.00 → 100%
    """
    old = {"A": 0.60, "B": 0.40}
    new = {"C": 0.70, "D": 0.30}
    result = compute_consensus_turnover(old, new)
    assert result.turnover_pct == pytest.approx(100.0, abs=1e-6)


def test_partial_change_hand_computed() -> None:
    """Partial rebalance: keep some, add one, remove one.

    old = {AAPL: 0.40, MSFT: 0.30, CASH: 0.30}
    new = {AAPL: 0.50, GOOGL: 0.30, CASH: 0.20}

    MSFT: old=0.30, new=0.00  → |Δw|=0.30  (removed)
    AAPL: old=0.40, new=0.50  → |Δw|=0.10  (re-weighted)
    CASH: old=0.30, new=0.20  → |Δw|=0.10  (re-weighted)
    GOOGL: old=0.00, new=0.30 → |Δw|=0.30  (added)

    sum_abs_diff = 0.80 → one-way turnover = 0.5 * 0.80 = 0.40 → 40.0%
    """
    old = {"AAPL": 0.40, "MSFT": 0.30, "CASH": 0.30}
    new = {"AAPL": 0.50, "GOOGL": 0.30, "CASH": 0.20}
    result = compute_consensus_turnover(old, new)
    assert result.turnover_pct == pytest.approx(40.0, abs=1e-6)
    assert result.n_added == 1       # GOOGL
    assert result.n_removed == 1     # MSFT
    assert result.n_reweighted == 2  # AAPL, CASH


def test_cash_move_included_in_turnover() -> None:
    """CASH 24%→0% move contributes to turnover (this is the W24→W25 key behavior).

    old = {JPM: 0.50, CASH: 0.24, REST: 0.26}
    new = {JPM: 0.60, REST: 0.40, CASH: 0.00}

    JPM:  |0.60-0.50| = 0.10
    CASH: |0.00-0.24| = 0.24    ← CASH contributes
    REST: |0.40-0.26| = 0.14

    sum_abs_diff = 0.48 → turnover = 0.5 * 0.48 = 0.24 → 24.0%
    """
    old = {"JPM": 0.50, "CASH": 0.24, "REST": 0.26}
    new = {"JPM": 0.60, "REST": 0.40, "CASH": 0.00}
    result = compute_consensus_turnover(old, new)
    assert result.turnover_pct == pytest.approx(24.0, abs=1e-6)
    # CASH goes 0.24→0: it counts as REMOVED (old>0, new=0)
    assert result.n_removed == 1   # CASH
    assert result.n_reweighted == 2  # JPM, REST


# ---------------------------------------------------------------------------
# First-week N/A fixture
# ---------------------------------------------------------------------------


def test_first_week_na() -> None:
    """Prior book is empty (first-ever week) → turnover_pct=None, N/A text."""
    result = compute_consensus_turnover({}, {"AAPL": 0.60, "CASH": 0.40})
    assert result.turnover_pct is None
    assert result.n_added == 0
    assert result.n_removed == 0
    assert result.n_reweighted == 0
    assert "N/A" in result.preview_line
    assert "first week" in result.preview_line
    assert "N/A" in result.log_line


# ---------------------------------------------------------------------------
# Real-data anchor: W24→W25 committed fixture
# ---------------------------------------------------------------------------


def test_w24_w25_anchor() -> None:
    """W24→W25 consensus pair reproduces the ~48.45% figure that triggered M6.

    If this test fails with a materially different value, root-cause the formula
    against the hand calculation — do NOT just adjust the expected value.
    Exact expected: 48.4472...%; tolerance ±0.1% to absorb float rounding.
    """
    result = compute_consensus_turnover(W24_HOLDINGS, W25_HOLDINGS)
    assert result.turnover_pct is not None
    assert result.turnover_pct == pytest.approx(EXPECTED_TURNOVER_PCT, abs=0.1), (
        f"W24→W25 anchor mismatch: got {result.turnover_pct:.4f}% "
        f"expected ~{EXPECTED_TURNOVER_PCT:.2f}%. "
        "Root-cause the formula (CASH included? 0.5 one-way factor applied?) "
        "before adjusting the expected value."
    )


def test_w24_w25_breakdown_counts() -> None:
    """W24→W25 breakdown: added=19, removed=17, re-weighted=19."""
    result = compute_consensus_turnover(W24_HOLDINGS, W25_HOLDINGS)
    assert result.n_added == EXPECTED_N_ADDED, (
        f"n_added: got {result.n_added}, expected {EXPECTED_N_ADDED}"
    )
    assert result.n_removed == EXPECTED_N_REMOVED, (
        f"n_removed: got {result.n_removed}, expected {EXPECTED_N_REMOVED}"
    )
    assert result.n_reweighted == EXPECTED_N_REWEIGHTED, (
        f"n_reweighted: got {result.n_reweighted}, expected {EXPECTED_N_REWEIGHTED}"
    )


# ---------------------------------------------------------------------------
# Surfacing checks — preview_line and log_line
# ---------------------------------------------------------------------------


def test_preview_line_present_normal_week() -> None:
    """Normal week: preview_line contains 'Consensus turnover vs last week'."""
    old = {"AAPL": 0.60, "CASH": 0.40}
    new = {"AAPL": 0.50, "MSFT": 0.20, "CASH": 0.30}
    result = compute_consensus_turnover(old, new)
    assert "Consensus turnover vs last week" in result.preview_line
    assert "%" in result.preview_line
    assert "added" in result.preview_line
    assert "removed" in result.preview_line
    assert "re-weighted" in result.preview_line


def test_preview_line_na_first_week() -> None:
    """First week: preview_line clearly states N/A and 'first week'."""
    result = compute_consensus_turnover({}, {"AAPL": 1.0})
    assert "N/A" in result.preview_line
    assert "first week" in result.preview_line


def test_log_line_present_normal_week() -> None:
    """Normal week: log_line contains the turnover figure."""
    old = {"AAPL": 0.60, "CASH": 0.40}
    new = {"AAPL": 0.30, "MSFT": 0.30, "CASH": 0.40}
    result = compute_consensus_turnover(old, new)
    assert "%" in result.log_line
    assert "added" in result.log_line


def test_log_line_na_first_week() -> None:
    """First week: log_line contains N/A."""
    result = compute_consensus_turnover({}, {"AAPL": 1.0})
    assert "N/A" in result.log_line


# ---------------------------------------------------------------------------
# Integration with report_run_metrics
# ---------------------------------------------------------------------------

_PERSONA_SLUGS = [
    "value", "growth", "discretionary-macro", "cta-systematic-macro",
    "technical", "quant-systematic", "risk-officer",
]

_TIMING = {s: 300.0 for s in _PERSONA_SLUGS}


def _make_results() -> list[PersonaResearchResult]:
    def _res(slug: str) -> PersonaResearchResult:
        return PersonaResearchResult(
            persona_slug=slug,
            week_id="2026-W25",
            parsed_output=PersonaOutputSchema(
                shortlist=[ShortlistEntry(ticker="AAPL", why="test")],
                report="ok",
                web_searches_used=3,
                data_tool_calls_used=8,
            ),
            validation=ReportValidationResult(
                passed=True, notes="ok", stage="structural", llm_justification=""
            ),
            report_payload=PersonaReportPayload(
                week_id="2026-W25",
                persona=slug,
                summary="ok",
                validator_passed=1,
                validator_notes="",
                full_report_path=f"state/reports/2026-W25/{slug}.md",
            ),
            shortlist_rows=[],
            budget_summary={"web_searches_used": 3, "data_tool_calls_used": 8},
            budget_overrun=False,
        )
    return [_res(s) for s in _PERSONA_SLUGS]


def _make_budgets() -> dict[str, PersonaBudget]:
    return {
        "__defaults__": PersonaBudget(max_turns=12, max_web_searches=8, max_data_tool_calls=15),
        **{s: PersonaBudget(max_turns=12, max_web_searches=8, max_data_tool_calls=15)
           for s in _PERSONA_SLUGS},
    }


def _window_cfg() -> dict[str, Any]:
    return {"window_hours": 5.0, "window_proximity_threshold": 0.80}


def test_report_run_metrics_turnover_field(tmp_path: Path) -> None:
    """report_run_metrics populates RunMetricsReport.turnover when holdings supplied."""
    log = tmp_path / "runs" / "2026-W25.log"
    prior = {"AAPL": 0.40, "MSFT": 0.30, "CASH": 0.30}
    new   = {"AAPL": 0.50, "GOOGL": 0.30, "CASH": 0.20}
    report = report_run_metrics(
        per_persona_timing=_TIMING,
        research_results=_make_results(),
        budgets=_make_budgets(),
        window_config=_window_cfg(),
        run_log_path=log,
        prior_holdings=prior,
        new_holdings=new,
    )
    assert isinstance(report, RunMetricsReport)
    assert report.turnover is not None
    assert isinstance(report.turnover, TurnoverResult)
    assert report.turnover.turnover_pct == pytest.approx(40.0, abs=1e-6)


def test_report_run_metrics_preview_in_summary(tmp_path: Path) -> None:
    """summary_text contains the turnover preview line."""
    log = tmp_path / "runs" / "2026-W25.log"
    prior = {"AAPL": 0.40, "CASH": 0.60}
    new   = {"MSFT": 0.40, "CASH": 0.60}
    report = report_run_metrics(
        per_persona_timing=_TIMING,
        research_results=_make_results(),
        budgets=_make_budgets(),
        window_config=_window_cfg(),
        run_log_path=log,
        prior_holdings=prior,
        new_holdings=new,
    )
    assert "Consensus turnover vs last week" in report.summary_text


def test_report_run_metrics_log_file_has_turnover(tmp_path: Path) -> None:
    """The persisted run log file contains the turnover log_line."""
    log = tmp_path / "runs" / "2026-W25.log"
    prior = {"AAPL": 0.40, "CASH": 0.60}
    new   = {"MSFT": 0.40, "CASH": 0.60}
    report_run_metrics(
        per_persona_timing=_TIMING,
        research_results=_make_results(),
        budgets=_make_budgets(),
        window_config=_window_cfg(),
        run_log_path=log,
        prior_holdings=prior,
        new_holdings=new,
    )
    contents = log.read_text(encoding="utf-8")
    assert "consensus_turnover=" in contents
    assert "%" in contents


def test_report_run_metrics_backward_compat(tmp_path: Path) -> None:
    """Omitting prior_holdings and new_holdings → turnover=None, no crash."""
    log = tmp_path / "runs" / "2026-W24.log"
    report = report_run_metrics(
        per_persona_timing=_TIMING,
        research_results=_make_results(),
        budgets=_make_budgets(),
        window_config=_window_cfg(),
        run_log_path=log,
        # prior_holdings and new_holdings intentionally omitted
    )
    assert report.turnover is None


# ---------------------------------------------------------------------------
# No-tripwire assertion
# ---------------------------------------------------------------------------


def test_no_tripwire_run_completes_any_turnover(tmp_path: Path) -> None:
    """100% turnover does NOT raise, block, or alter the report — metric only.

    The run must complete and return a RunMetricsReport regardless of
    the turnover value.  No exception, no special return value.
    """
    log = tmp_path / "runs" / "2026-W25.log"
    # Full replacement → 100% one-way turnover
    prior = {"AAPL": 0.60, "CASH": 0.40}
    new   = {"MSFT": 0.70, "GOOGL": 0.30}
    report = report_run_metrics(
        per_persona_timing=_TIMING,
        research_results=_make_results(),
        budgets=_make_budgets(),
        window_config=_window_cfg(),
        run_log_path=log,
        prior_holdings=prior,
        new_holdings=new,
    )
    # Must still return a valid RunMetricsReport — not raise.
    assert isinstance(report, RunMetricsReport)
    assert report.turnover is not None
    assert report.turnover.turnover_pct == pytest.approx(100.0, abs=1e-6)
    # No escalation field change — turnover does NOT set escalation_triggered.
    # (escalation is window-based, not turnover-based.)
    assert report.escalation_triggered is False
