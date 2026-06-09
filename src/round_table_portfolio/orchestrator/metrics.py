"""run_metrics — Component 19: run-time + per-persona web-search-count measurement & report.

Critical Component #3 — the #1 feasibility-risk measurement.

ROADMAP §3.2 EXPLICIT task: measure + report the real agentic run-time and
per-persona web-search count for a single weekly run.  Compares total wall-clock
against the 5-hour rolling-window threshold and explicitly states the proximity
as a percentage.  If the run approaches or crosses the configured proximity
threshold, raises the bounding-playbook escalation (fewer personas / tighter
budgets / tier upgrade) as a LOUD, surfaced flag — never a silent note.

Usage::

    from round_table_portfolio.orchestrator.metrics import (
        RunMetricsReport,
        report_run_metrics,
    )

    metrics = report_run_metrics(
        per_persona_timing={"value": 320.1, "growth": 410.3, ...},
        per_round1_timing={"value": 180.0, "growth": 200.0, ...},
        per_judge_timing={"value": 60.0, "growth": 55.0, ...},
        per_round2_timing={"value": 300.0, "growth": 280.0},   # 2 outliers only
        research_results=persona_results,
        budgets=budgets,
        window_config={"window_hours": 5.0, "window_proximity_threshold": 0.80},
        run_log_path=Path("state/runs/2026-W23.log"),
    )
    print(metrics.summary_text)

Design notes (M3 update — DEF-003 full-cycle timing):
- The orchestrator measures per-persona wall-clock by capturing time.time()
  before/after each subagent dispatch.  Python cannot time across subagent
  boundaries from inside this helper — the orchestrator passes the measured
  maps in.  This is the same pattern as M1-011's wall_clock_seconds.
- M3 extends this to FOUR timing maps: research, Round-1, judges, Round-2.
  The headline total_run_time is now the SUM of all phases.
- The helper is a pure data assembler: no subagent dispatch, no DB writes.
  Side effect: appends to the run log file (state/runs/YYYY-WNN.log).
- FEASIBILITY VERDICT categories:
    fits:       total run < 60% of window
    tight:      60% <= total run < proximity_threshold of window
    does-not-fit: total run >= proximity_threshold of window
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional  # noqa: F401 — Optional used in report_run_metrics signature

from round_table_portfolio.budget.loader import PersonaBudget, get_budget
from round_table_portfolio.research.runner import PersonaResearchResult

logger = logging.getLogger(__name__)

# Default thresholds — overridden by window_config at call time.
_DEFAULT_WINDOW_HOURS = 5.0
_DEFAULT_PROXIMITY_THRESHOLD = 0.80   # 80% of the window triggers escalation

# The "tight" band starts at 60% — below this is comfortably "fits".
_TIGHT_BAND_START = 0.60


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class PersonaMetrics:
    """Per-persona slice of the run report."""
    persona_slug: str
    wall_clock_seconds: float
    web_searches_used: int
    data_tool_calls_used: int
    budget_max_web_searches: int
    budget_max_data_tool_calls: int
    web_searches_over_budget: bool
    data_tool_calls_over_budget: bool


@dataclass
class RunMetricsReport:
    """Full metrics report for one weekly run.

    The orchestrator prints ``summary_text`` to the session as part of the
    founder-facing M2-011 / M3-006 output.  The report is also persisted to
    ``state/runs/YYYY-WNN.log``.

    M3 change (DEF-003): total_wall_seconds is now the SUM of ALL phases
    (research + round1 + judges + round2), not research-only.  The per-phase
    breakdown fields let the founder see where time is spent.
    """
    week_label: str
    total_wall_seconds: float
    window_hours: float                     # configured 5-hour threshold
    window_fraction: float                  # 0–1, fraction of window consumed
    window_proximity_threshold: float       # configured escalation trigger (e.g. 0.80)
    feasibility_verdict: str                # "fits" | "tight" | "does-not-fit"
    escalation_triggered: bool              # True when window_fraction >= proximity_threshold
    per_persona: list[PersonaMetrics] = field(default_factory=list)
    over_budget_personas: list[str] = field(default_factory=list)
    summary_text: str = ""
    # Per-phase totals (M3 DEF-003 — zero when phase not run / timing not provided).
    research_total_seconds: float = 0.0
    round1_total_seconds: float = 0.0
    judges_total_seconds: float = 0.0
    round2_total_seconds: float = 0.0

    # Convenience accessor matching the stub's field name so callers that
    # used RunMetrics.total_wall_seconds continue to work unchanged.
    @property
    def total_wall_seconds_property(self) -> float:
        return self.total_wall_seconds


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xh Ym Zs' or 'Ym Zs' or 'Zs'."""
    secs = int(seconds)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _build_summary_text(
    week_label: str,
    total_wall_seconds: float,
    window_hours: float,
    window_fraction: float,
    proximity_threshold: float,
    feasibility_verdict: str,
    escalation_triggered: bool,
    per_persona: list[PersonaMetrics],
    over_budget_personas: list[str],
    *,
    research_total_seconds: float = 0.0,
    round1_total_seconds: float = 0.0,
    judges_total_seconds: float = 0.0,
    round2_total_seconds: float = 0.0,
) -> str:
    """Assemble the human-readable summary for session output.

    M3 DEF-003: total includes all phases; per-phase breakdown is printed so
    the founder sees where time is spent (research vs. Round 1 vs. judges vs.
    Round 2).
    """
    window_pct = window_fraction * 100.0
    duration_str = _fmt_duration(total_wall_seconds)
    window_label = _fmt_duration(window_hours * 3600)

    lines: list[str] = [
        "=" * 62,
        "  RUN METRICS REPORT",
        "=" * 62,
        f"  Week:          {week_label}",
        f"  Total run:     {duration_str}  ({window_pct:.0f}% of the {window_label} window)",
        f"  Verdict:       {feasibility_verdict.upper()}",
    ]

    # Per-phase breakdown — only print phases that consumed non-zero time.
    phase_breakdown: list[str] = []
    if research_total_seconds > 0:
        phase_breakdown.append(
            f"    Research:  {_fmt_duration(research_total_seconds)}"
        )
    if round1_total_seconds > 0:
        phase_breakdown.append(
            f"    Round-1:   {_fmt_duration(round1_total_seconds)}"
        )
    if judges_total_seconds > 0:
        phase_breakdown.append(
            f"    Judges:    {_fmt_duration(judges_total_seconds)}"
        )
    if round2_total_seconds > 0:
        phase_breakdown.append(
            f"    Round-2:   {_fmt_duration(round2_total_seconds)}"
        )
    if phase_breakdown:
        lines += ["", "  Phase breakdown (full-cycle total above):"]
        lines += phase_breakdown

    if escalation_triggered:
        lines += [
            "",
            "!! BOUNDING-PLAYBOOK ESCALATION !!",
            f"   Run consumed {window_pct:.0f}% of the {window_label} rolling window",
            f"   (threshold: {proximity_threshold*100:.0f}%).",
            "   Choose one: fewer personas / tighter budgets / tier upgrade.",
            "   This is a FOUNDER DECISION — do not proceed silently.",
        ]

    lines += [
        "",
        "  Per-persona breakdown:",
        f"  {'Persona':<28} {'Wall-clock':>10}  {'Web':>5}/{'' :<4}  {'Tools':>5}/{'':4}  Status",
        "  " + "-" * 58,
    ]

    for pm in per_persona:
        status_parts: list[str] = []
        if pm.web_searches_over_budget:
            status_parts.append(f"WEB OVER ({pm.web_searches_used}>{pm.budget_max_web_searches})")
        if pm.data_tool_calls_over_budget:
            status_parts.append(f"TOOLS OVER ({pm.data_tool_calls_used}>{pm.budget_max_data_tool_calls})")
        status = ", ".join(status_parts) if status_parts else "ok"
        dur = _fmt_duration(pm.wall_clock_seconds)
        lines.append(
            f"  {pm.persona_slug:<28} {dur:>10}"
            f"  {pm.web_searches_used:>5}/{pm.budget_max_web_searches:<4}"
            f"  {pm.data_tool_calls_used:>5}/{pm.budget_max_data_tool_calls:<4}"
            f"  {status}"
        )

    if over_budget_personas:
        lines += [
            "",
            f"  OVER-BUDGET PERSONAS: {', '.join(over_budget_personas)}",
            "  These personas systematically exceeded their budgets — root-cause",
            "  before next run (Critical Component #3, bounding playbook).",
        ]

    lines.append("=" * 62)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def report_run_metrics(
    per_persona_timing: dict[str, float],
    research_results: list[PersonaResearchResult],
    *,
    budgets: dict[str, PersonaBudget],
    window_config: dict[str, Any],
    run_log_path: Path,
    per_round1_timing: Optional[dict[str, float]] = None,
    per_judge_timing: Optional[dict[str, float]] = None,
    per_round2_timing: Optional[dict[str, float]] = None,
) -> RunMetricsReport:
    """Measure and report run-time + per-persona web-search/tool counts.

    M3 change (DEF-003): the headline total_run_time is now the SUM of ALL
    phases — research (per_persona_timing), Round-1 dispatches
    (per_round1_timing), output-validator judge dispatches (per_judge_timing),
    and the 2 Round-2 outlier dispatches (per_round2_timing).  Callers that
    do not yet supply the new maps (e.g. M2 tests, backward-compatible callers)
    receive a total equal to the research phase only — matching the M2 behaviour.

    Args:
        per_persona_timing:
            Mapping of persona_slug → wall-clock seconds for the research
            dispatch phase.  Measured by the orchestrator around each dispatch.
        research_results:
            The 7 ``PersonaResearchResult`` objects from the research loop.
        budgets:
            Loaded persona-budget dict (from ``budget.loader.load_budgets``).
        window_config:
            Dict with keys:
              ``window_hours`` (float, default 5.0) — the rolling-window limit.
              ``window_proximity_threshold`` (float, default 0.80) — fraction at
              which the bounding-playbook escalation fires.
        run_log_path:
            Path to append the report to.  ``state/runs/`` is mkdir-p'd.
        per_round1_timing:
            Mapping of persona_slug → wall-clock seconds for Round-1 dispatch.
            None if not yet measured (backward-compatible — treated as 0).
        per_judge_timing:
            Mapping of persona_slug → wall-clock seconds for judge dispatch.
            None if not yet measured (backward-compatible — treated as 0).
        per_round2_timing:
            Mapping of outlier_slug → wall-clock seconds for Round-2 dispatch.
            None if Round 2 has not run yet (backward-compatible — treated as 0).

    Returns:
        A ``RunMetricsReport`` with all metrics populated, ``summary_text``
        ready for session printing, and the report persisted to ``run_log_path``.

    Side effects:
        Appends (or creates) the run log at ``run_log_path``.
    """
    window_hours: float = float(window_config.get("window_hours", _DEFAULT_WINDOW_HOURS))
    proximity_threshold: float = float(
        window_config.get("window_proximity_threshold", _DEFAULT_PROXIMITY_THRESHOLD)
    )

    # Build per-persona metrics (research phase is the reference for budget).
    per_persona: list[PersonaMetrics] = []
    over_budget_personas: list[str] = []

    result_by_slug: dict[str, PersonaResearchResult] = {
        r.persona_slug: r for r in research_results
    }

    for slug, wall_secs in per_persona_timing.items():
        res = result_by_slug.get(slug)
        if res is None:
            logger.warning("per_persona_timing has slug %r with no matching research result.", slug)
            continue

        budget: PersonaBudget = get_budget(budgets, slug)
        web_used = res.parsed_output.web_searches_used
        tools_used = res.parsed_output.data_tool_calls_used
        web_over = web_used > budget.max_web_searches
        tools_over = tools_used > budget.max_data_tool_calls

        pm = PersonaMetrics(
            persona_slug=slug,
            wall_clock_seconds=wall_secs,
            web_searches_used=web_used,
            data_tool_calls_used=tools_used,
            budget_max_web_searches=budget.max_web_searches,
            budget_max_data_tool_calls=budget.max_data_tool_calls,
            web_searches_over_budget=web_over,
            data_tool_calls_over_budget=tools_over,
        )
        per_persona.append(pm)

        if web_over or tools_over:
            over_budget_personas.append(slug)

    # Per-phase totals (DEF-003 full-cycle timing).
    research_total = sum(per_persona_timing.values())
    round1_total = sum((per_round1_timing or {}).values())
    judges_total = sum((per_judge_timing or {}).values())
    round2_total = sum((per_round2_timing or {}).values())

    # Full-cycle total — the verdict is computed against this (DEF-003 fix).
    total_wall_seconds: float = research_total + round1_total + judges_total + round2_total

    window_seconds = window_hours * 3600.0
    window_fraction = total_wall_seconds / window_seconds if window_seconds > 0 else 0.0

    # Feasibility verdict.
    if window_fraction >= proximity_threshold:
        feasibility_verdict = "does-not-fit"
        escalation_triggered = True
    elif window_fraction >= _TIGHT_BAND_START:
        feasibility_verdict = "tight"
        escalation_triggered = False
    else:
        feasibility_verdict = "fits"
        escalation_triggered = False

    # Derive week_label from the run log path stem (e.g. "2026-W23").
    week_label = run_log_path.stem

    summary_text = _build_summary_text(
        week_label=week_label,
        total_wall_seconds=total_wall_seconds,
        window_hours=window_hours,
        window_fraction=window_fraction,
        proximity_threshold=proximity_threshold,
        feasibility_verdict=feasibility_verdict,
        escalation_triggered=escalation_triggered,
        per_persona=per_persona,
        over_budget_personas=over_budget_personas,
        research_total_seconds=research_total,
        round1_total_seconds=round1_total,
        judges_total_seconds=judges_total,
        round2_total_seconds=round2_total,
    )

    report = RunMetricsReport(
        week_label=week_label,
        total_wall_seconds=total_wall_seconds,
        window_hours=window_hours,
        window_fraction=window_fraction,
        window_proximity_threshold=proximity_threshold,
        feasibility_verdict=feasibility_verdict,
        escalation_triggered=escalation_triggered,
        per_persona=per_persona,
        over_budget_personas=over_budget_personas,
        summary_text=summary_text,
        research_total_seconds=research_total,
        round1_total_seconds=round1_total,
        judges_total_seconds=judges_total,
        round2_total_seconds=round2_total,
    )

    # Persist to run log — mkdir -p so state/runs/ does not need to pre-exist.
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log_entry = (
        f"\n--- run-metrics @ {timestamp} ---\n"
        f"{summary_text}\n"
    )
    with run_log_path.open("a", encoding="utf-8") as fh:
        fh.write(log_entry)

    logger.info(
        "Run metrics: total=%.1fs (research=%.1f r1=%.1f judges=%.1f r2=%.1f) "
        "window_fraction=%.1f%% verdict=%s escalation=%s",
        total_wall_seconds,
        research_total,
        round1_total,
        judges_total,
        round2_total,
        window_fraction * 100,
        feasibility_verdict,
        escalation_triggered,
    )

    if escalation_triggered:
        logger.warning(
            "BOUNDING-PLAYBOOK ESCALATION: run consumed %.0f%% of the %.1fh window. "
            "Raise to founder.",
            window_fraction * 100,
            window_hours,
        )

    return report
