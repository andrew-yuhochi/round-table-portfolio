"""M1 demo harness — AGGREGATE-AND-REPORT layer (Component 6).

Takes a list of already-run ``PersonaResearchResult`` objects (one per persona,
produced by the main session's ``run_persona_research`` calls) and renders a
single founder-readable markdown report at::

    state/reports/<week_id>/research_validation.md

The harness does NOT:
- Dispatch persona subagents (that is the main session's job).
- Re-invoke ``validate_persona_report`` or the on-mandate judge (the verdict is
  already on ``result.validation``).
- Write any ledger DB rows (that is the orchestrator's job).

It only reads the in-memory ``PersonaResearchResult`` list and writes one file.

Public entry point::

    path = build_research_validation_report(
        results=results,
        week_id="2026-06-02",
        wall_clock_seconds={"value": 142.3, "growth": 198.1, ...},
        state_root=Path("state"),
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from round_table_portfolio.research.runner import PersonaResearchResult, PersonaShortlistRow

logger = logging.getLogger(__name__)

# Stable display order — matches the 7-persona roster in order of archetype.
_PERSONA_ORDER = [
    "value",
    "growth",
    "technical",
    "quant-systematic",
    "discretionary-macro",
    "cta-systematic-macro",
    "risk-officer",
]


# ---------------------------------------------------------------------------
# Cluster reconstruction helpers
# ---------------------------------------------------------------------------


def _split_primaries_and_peers(
    shortlist_rows: list[PersonaShortlistRow],
) -> tuple[list[PersonaShortlistRow], dict[str, list[str]]]:
    """Split shortlist rows into primary entries and a peer-map.

    Returns:
        primaries: rows where ``is_cluster_peer == 0``, in insertion order.
        peer_map:  dict mapping each primary ticker to its list of peer tickers.
    """
    primaries: list[PersonaShortlistRow] = []
    peer_map: dict[str, list[str]] = {}

    for row in shortlist_rows:
        if row.is_cluster_peer == 0:
            primaries.append(row)
            if row.ticker not in peer_map:
                peer_map[row.ticker] = []
        else:
            parent = row.parent_ticker or "UNKNOWN"
            peer_map.setdefault(parent, []).append(row.ticker)

    return primaries, peer_map


# ---------------------------------------------------------------------------
# Per-persona section renderer
# ---------------------------------------------------------------------------


def _render_persona_section(
    result: PersonaResearchResult,
    wall_clock_seconds: Optional[float],
) -> str:
    """Render the markdown section for one persona."""
    lines: list[str] = []
    slug = result.persona_slug
    v = result.validation

    # Section heading
    verdict_badge = "PASS" if v.passed else "FAIL"
    lines.append(f"## {slug.upper()}  —  {verdict_badge}")
    lines.append("")

    # --- Validator verdict block ---
    lines.append("### Validator verdict")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Result | **{verdict_badge}** |")
    lines.append(f"| Stage | `{v.stage}` |")
    lines.append(f"| Notes | {v.notes} |")

    # llm_justification is the concrete on-mandate evidence — always include even
    # when empty so the founder can see at a glance whether the judge ran.
    justification_text = v.llm_justification.strip() if v.llm_justification else "(not run — structural stage only)"
    lines.append(f"| LLM justification | {justification_text} |")
    lines.append("")

    # --- Shortlist + cluster structure ---
    lines.append("### Shortlist + clusters")
    lines.append("")
    primaries, peer_map = _split_primaries_and_peers(result.shortlist_rows)

    if not primaries:
        lines.append("_No shortlist entries recorded._")
    else:
        lines.append("| Primary | Cluster peers |")
        lines.append("|---------|---------------|")
        for p in primaries:
            peers = peer_map.get(p.ticker, [])
            peers_str = ", ".join(peers) if peers else "—"
            lines.append(f"| **{p.ticker}** | {peers_str} |")
    lines.append("")

    # --- Budget consumption ---
    lines.append("### Budget consumption")
    lines.append("")
    bs = result.budget_summary

    # Web-search row — always present.
    ws = bs.get("web_searches", {})
    ws_used = ws.get("used", result.parsed_output.web_searches_used)
    ws_cap = ws.get("cap", "?")
    ws_exhausted = ws.get("exhausted", False)

    # Data-tool row — merged in by runner.
    dt = bs.get("data_tool_calls", {})
    dt_used = dt.get("used", result.parsed_output.data_tool_calls_used)
    dt_cap = dt.get("cap", "?")
    dt_exhausted = dt.get("exhausted", False)

    overrun_flag = " ⚠ BUDGET OVERRUN" if result.budget_overrun else ""

    lines.append(f"| Metric | Used | Cap | Exhausted |")
    lines.append(f"|--------|------|-----|-----------|")
    lines.append(f"| Web searches | {ws_used} | {ws_cap} | {'yes' if ws_exhausted else 'no'} |")
    lines.append(f"| Data tool calls | {dt_used} | {dt_cap} | {'yes' if dt_exhausted else 'no'} |")

    if wall_clock_seconds is not None:
        lines.append(f"| Wall clock (s) | {wall_clock_seconds:.1f} | — | — |")
    else:
        lines.append(f"| Wall clock (s) | not recorded | — | — |")

    if overrun_flag:
        lines.append("")
        lines.append(f"> **{overrun_flag.strip()}**")

    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_research_validation_report(
    results: list[PersonaResearchResult],
    week_id: str,
    wall_clock_seconds: dict[str, float] | None = None,
    *,
    state_root: Path = Path("state"),
    generated_at: Optional[datetime] = None,
) -> Path:
    """Aggregate per-persona research results into one founder-readable markdown report.

    Args:
        results:             List of ``PersonaResearchResult`` objects produced by
                             ``run_persona_research`` calls — one per persona.
        week_id:             ISO date string, e.g. ``"2026-06-02"``.  Used as the
                             sub-directory under ``state/reports/`` and as the
                             report header label.
        wall_clock_seconds:  Optional mapping of persona_slug → elapsed seconds
                             measured by the main session around each
                             ``run_persona_research`` call.  When ``None`` or a
                             slug is absent, wall-clock is rendered as
                             ``"not recorded"``.
        state_root:          Root of the ``state/`` directory.  Override in tests
                             to a ``tmp_path``.
        generated_at:        Timestamp to embed in the report header.  When
                             ``None``, the current UTC time is used.  Accepting
                             this as a parameter makes the output deterministic
                             in tests without monkeypatching datetime.

    Returns:
        Path to the written ``research_validation.md`` file.
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    wc = wall_clock_seconds or {}

    # Build a slug → result map for stable ordering.
    result_map: dict[str, PersonaResearchResult] = {r.persona_slug: r for r in results}

    # Stable order: roster order first, then any extras alphabetically.
    ordered_slugs = [s for s in _PERSONA_ORDER if s in result_map]
    extras = sorted(s for s in result_map if s not in _PERSONA_ORDER)
    ordered_slugs.extend(extras)

    # --- Compute header-level metrics ---
    n_personas = len(ordered_slugs)
    n_pass = sum(1 for s in ordered_slugs if result_map[s].validation.passed)
    total_web_searches = sum(
        result_map[s].parsed_output.web_searches_used for s in ordered_slugs
    )
    total_data_calls = sum(
        result_map[s].parsed_output.data_tool_calls_used for s in ordered_slugs
    )

    # --- Build report lines ---
    lines: list[str] = []

    lines.append(f"# Research Validation Report — week {week_id}")
    lines.append("")
    lines.append(f"**Generated**: {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Week ID**: `{week_id}`")
    lines.append(f"**Universe snapshot**: week of {week_id}")
    lines.append("")

    # Summary line — Critical Component #3 measurement lives here.
    pass_rate_str = f"{n_pass}/{n_personas} PASS"
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Personas run | {n_personas} of 7 |")
    lines.append(f"| Validator pass rate | **{pass_rate_str}** |")
    lines.append(f"| Total web searches (all personas) | {total_web_searches} |")
    lines.append(f"| Total data tool calls (all personas) | {total_data_calls} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-persona sections.
    for slug in ordered_slugs:
        result = result_map[slug]
        wall_clock = wc.get(slug)
        lines.append(_render_persona_section(result, wall_clock))

    report_text = "\n".join(lines)

    # Write file.
    report_dir = state_root / "reports" / week_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "research_validation.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Research validation report written: %s", report_path)

    return report_path
