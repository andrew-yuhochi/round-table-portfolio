"""weekly_run.py — Session driver for the /weekly-run end-to-end cycle.

The Claude session:
  1. Dispatches each of the 7 persona subagents and writes their raw JSON
     replies to  state/runs/<week>.persona_replies.json
  2. Calls this driver in prepare-round1 mode to compute the debate set and
     write state/runs/<week>.debate_set.json, which the session uses to build
     each persona's Round-1 prompt (commit-before-reveal):
       python scripts/weekly_run.py --mode prepare-round1 --week 2026-W23
  3. Dispatches the 7 Round-1 persona subagents over the debate set and writes
     their raw JSON replies to state/runs/<week>.round1_replies.json
  4. Dispatches the output-validator-judge subagent per persona, parses with
     parse_judge_response, and writes the verdicts to
     state/runs/<week>.judge_verdicts.json
  5. Records per-persona + per-phase wall-clock timing to
     state/runs/<week>.timing.json        (research phase)
     state/runs/<week>.round1_timing.json (Round-1 dispatch phase)
     state/runs/<week>.judge_timing.json  (judge dispatch phase)
  6. Calls this driver in prepare-round2 mode to receive the 2 outlier slugs
     and counterargument blocks printed to stdout, then dispatches those 2
     outlier subagents and writes their replies to
     state/runs/<week>.round2_replies.json:
       python scripts/weekly_run.py --mode prepare-round2 --week 2026-W23
  7. Calls this driver in preview mode to review the proposed consensus before
     committing:
       python scripts/weekly_run.py --mode preview [--week 2026-W23]
  8. Reviews the preview output, then commits with the founder reply:
       python scripts/weekly_run.py --mode commit --founder-reply "approve"

Modes
-----
prepare-round1 — Reads state/runs/<week>.persona_replies.json and computes the
           debate set.  Writes state/runs/<week>.debate_set.json.

prepare-round2 — Reads state/runs/<week>.persona_replies.json +
           state/runs/<week>.round1_replies.json and runs steps 6a–6c
           (dissent → outlier selection → counterargument assembly) deterministically.
           Prints the 2 outlier slugs + their counterargument blocks so the session
           can dispatch the Round-2 subagents.  Writes nothing to state/.

preview  — Runs the full engine (including Round-2 with stubbed replies if
           state/runs/<week>.round2_replies.json exists) against a THROWAWAY
           temp ledger.  Prints + writes state/runs/<week>.preview.md.

commit   — Runs the full engine against the REAL state/ledger.db with the
           supplied --founder-reply.

Input files (produced by the session before calling this driver)
---------------------------------------------------------------
state/runs/<week>.persona_replies.json        (7 entries — research phase)
state/runs/<week>.round1_replies.json         (7 entries — Round-1 phase)
state/runs/<week>.judge_verdicts.json         (7 entries — judge verdicts)
state/runs/<week>.timing.json                 (7 entries — research wall-clock)
state/runs/<week>.round1_timing.json          (7 entries — Round-1 wall-clock, optional)
state/runs/<week>.judge_timing.json           (7 entries — judge wall-clock, optional)
state/runs/<week>.round2_replies.json         (2 entries — Round-2 outlier replies, optional)
state/runs/<week>.round2_timing.json          (2 entries — Round-2 wall-clock, optional)

Output file written by prepare-round1
--------------------------------------
state/runs/<week>.debate_set.json
    {
      "debate_set": ["AAPL", "MSFT", ...],
      "persona_digest": { "<slug>": {"shortlist": [...], "report_excerpt": "..."}, ... }
    }
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, Optional

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

# This script lives at projects/round-table-portfolio/scripts/weekly_run.py.
# parents[1] is the project root (round-table-portfolio/).
_PROJECT_ROOT = Path(__file__).parents[1]

# Ensure the project src is importable when run directly (not via pytest).
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from round_table_portfolio.orchestrator.weekly_run import run_weekly, WeeklyRunResult
from round_table_portfolio.personas.output_validator import ReplayJudge, StubOnMandateJudge
from round_table_portfolio.storage.apply_schema import apply_schema
from round_table_portfolio.budget.loader import get_budget, load_budgets
from round_table_portfolio.personas.output_validator import load_validator_config
from round_table_portfolio.research.runner import run_persona_research
from round_table_portfolio.orchestrator.round1 import construct_debate_set
from round_table_portfolio.orchestrator.counterargument import load_counterargument_config
from round_table_portfolio.orchestrator.round2 import parse_round2_reply

# ---------------------------------------------------------------------------
# Default week
# ---------------------------------------------------------------------------


def _current_week_label() -> str:
    today = datetime.date.today()
    iso = today.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


# ---------------------------------------------------------------------------
# Input-file readers
# ---------------------------------------------------------------------------


def _load_persona_replies(week: str, state_root: Path) -> dict[str, str]:
    path = state_root / "runs" / f"{week}.persona_replies.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_round1_replies(week: str, state_root: Path) -> dict[str, str]:
    path = state_root / "runs" / f"{week}.round1_replies.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_judge_verdicts(week: str, state_root: Path) -> dict[str, tuple[bool, str]]:
    path = state_root / "runs" / f"{week}.judge_verdicts.json"
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {
        slug: (bool(v["passed"]), str(v["justification"]))
        for slug, v in raw.items()
    }


def _load_timing(week: str, state_root: Path) -> dict[str, float]:
    path = state_root / "runs" / f"{week}.timing.json"
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {slug: float(t) for slug, t in raw.items()}


def _load_timing_optional(week: str, state_root: Path, suffix: str) -> dict[str, float]:
    """Load a timing JSON file if it exists, returning {} otherwise."""
    path = state_root / "runs" / f"{week}.{suffix}.json"
    if not path.exists():
        return {}
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {slug: float(t) for slug, t in raw.items()}


def _load_round2_replies_optional(week: str, state_root: Path) -> dict[str, str]:
    """Load round2_replies.json if it exists, returning {} otherwise."""
    path = state_root / "runs" / f"{week}.round2_replies.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Config loaders (reused across modes)
# ---------------------------------------------------------------------------


def _load_personas_config(project_root: Path) -> Path:
    return project_root / "config" / "personas.yaml"


def _load_persona_slugs_from_yaml(project_root: Path) -> list[str]:
    import yaml
    personas_cfg = _load_personas_config(project_root)
    raw = yaml.safe_load(personas_cfg.read_text(encoding="utf-8")) or {}
    slugs = raw.get("personas", [])
    if not slugs:
        return list(_PERSONA_SLUGS)
    return [str(s["slug"]) if isinstance(s, dict) else str(s) for s in slugs]


# ---------------------------------------------------------------------------
# Prepare-round1 mode
# ---------------------------------------------------------------------------


def run_prepare_round1(week: str, state_root: Path) -> None:
    """Compute the debate set from persona_replies and write debate_set.json.

    Uses a TEMP state_root internally so run_persona_research side-effects
    (report file writes) are isolated from the real state/.  The only file
    written into the real state/ is state/runs/<week>.debate_set.json.

    Prints the debate set, the Round-1 reply schema contract, and the
    max_position_weight cap so the session has everything it needs to dispatch
    the Round-1 persona subagents.
    """
    import yaml

    persona_replies = _load_persona_replies(week, state_root)

    # Load the same configs the commit run uses so the debate set is identical.
    budget_config = _PROJECT_ROOT / "config" / "persona_budgets.yaml"
    thresholds_config = _PROJECT_ROOT / "config" / "thresholds.yaml"
    validator_config_path = _PROJECT_ROOT / "config" / "validator.yaml"

    budget_raw = yaml.safe_load(budget_config.read_text(encoding="utf-8")) or {}
    thresholds = yaml.safe_load(thresholds_config.read_text(encoding="utf-8")) or {}
    max_position_weight: float = float(thresholds.get("max_position_weight", 0.20))

    budgets = load_budgets(budget_config)
    v_config = load_validator_config(validator_config_path)

    # Use StubOnMandateJudge — the debate set depends only on shortlists, not
    # on the on-mandate verdict.  The real judge verdict is not needed here.
    judge = StubOnMandateJudge()

    # Run run_persona_research per persona in a TEMP state_root so no report
    # files or side-effects land in the real state/ directory.
    tmp_dir = Path(tempfile.mkdtemp(prefix="rtp_prep_"))
    try:
        tmp_state = tmp_dir / "state"
        tmp_state.mkdir()

        persona_results = []
        for slug in _PERSONA_SLUGS:
            raw = persona_replies.get(slug)
            if not raw:
                raise RuntimeError(
                    f"prepare-round1: no persona reply found for slug={slug!r}. "
                    f"Expected {len(_PERSONA_SLUGS)} entries in "
                    f"state/runs/{week}.persona_replies.json."
                )
            budget = get_budget(budgets, slug)
            result = run_persona_research(
                persona_slug=slug,
                week_id=week,
                raw_output=raw,
                mandate="",
                judge=judge,
                budget=budget,
                validator_config=v_config,
                state_root=tmp_state,
            )
            persona_results.append(result)
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # Build the debate set using the SAME config dict as the commit run.
    debate_cfg: dict = {
        "debate_set_ceiling": budget_raw.get("debate_set_ceiling", 40),
        "max_position_weight": max_position_weight,
    }
    debate_set = construct_debate_set(persona_results, debate_cfg)

    # Build per-persona digest: shortlist tickers (direct only) + report excerpt.
    persona_digest: dict[str, dict] = {}
    for result in persona_results:
        direct_shortlist = [
            row.ticker
            for row in result.shortlist_rows
            if row.is_cluster_peer == 0
        ]
        report_text = result.parsed_output.report if result.parsed_output else ""
        excerpt = report_text[:400].strip()
        persona_digest[result.persona_slug] = {
            "shortlist": direct_shortlist,
            "report_excerpt": excerpt,
        }

    # Write debate_set.json into the REAL state/runs/ (the only real-state write).
    debate_set_payload = {
        "debate_set": debate_set,
        "persona_digest": persona_digest,
    }
    runs_dir = state_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    debate_set_path = runs_dir / f"{week}.debate_set.json"
    debate_set_path.write_text(
        json.dumps(debate_set_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print the debate set summary and the Round-1 reply schema contract.
    print(f"\n{'='*62}")
    print(f"  DEBATE SET — {week}")
    print(f"{'='*62}")
    print(f"  Tickers ({len(debate_set)}): {', '.join(debate_set)}")
    print(f"  max_position_weight cap:   {max_position_weight}")
    print(f"  Action vocabulary:         add | reduce | hold | exit")
    print()
    print("  ROUND-1 REPLY SCHEMA (required from each persona):")
    print("  {")
    print('    "stances": [')
    print('      {')
    print('        "ticker":        "<one ticker from the debate set>",')
    print('        "action":        "ADD" | "REDUCE" | "HOLD" | "EXIT",')
    print(f'        "target_weight": <float 0.0 – {max_position_weight}>,')
    print('        "confidence":    <int 1–5>,')
    print('        "rationale":     "<text>"')
    print('      },')
    print(f'      ...  // one entry for EVERY ticker in the debate set ({len(debate_set)} required)')
    print('    ],')
    print('    "counterfactual_portfolio": {')
    print('      "<TICKER>": <float weight>,')
    print('      ...          // any subset of debate-set tickers')
    print(f'      "CASH":    <float>,   // explicit CASH key required')
    print('                   // all weights must be non-negative')
    print(f'                   // non-CASH weights each ≤ {max_position_weight}')
    print('                   // all weights must sum to 1.0')
    print('    },')
    print('    "narrative_summary": "<text>"')
    print("  }")
    print()
    print(f"  Written: {debate_set_path}")
    print()

    # Print per-persona shortlists for session reference.
    print("  Per-persona shortlists (direct picks only):")
    for slug, digest in persona_digest.items():
        tickers_str = ", ".join(digest["shortlist"]) if digest["shortlist"] else "(empty)"
        print(f"    {slug:<28} {tickers_str}")
    print()


# ---------------------------------------------------------------------------
# Prepare-round2 mode (M3 — prints outliers + counterarguments)
# ---------------------------------------------------------------------------


def run_prepare_round2(week: str, state_root: Path) -> None:
    """Run steps 6a–6c deterministically and print the Round-2 briefing.

    Reads persona_replies + round1_replies, runs the full engine up through
    counterargument assembly (all free Python — no dispatch), then prints:
      - The 2 selected outlier slugs + their divergence scores.
      - Each outlier's assembled counterargument block (verbatim quotes from
        opposing personas' Round-1 rationales).

    The session uses this output to dispatch the 2 Round-2 outlier subagents,
    capture their replies, and write state/runs/<week>.round2_replies.json.

    WRITES NOTHING to state/ — this is a read + print mode only.
    """
    import yaml
    from round_table_portfolio.orchestrator.dissent import (
        compute_dissent,
        load_dissent_config,
        select_outliers,
    )
    from round_table_portfolio.orchestrator.counterargument import (
        assemble_counterarguments,
        load_counterargument_config,
    )

    persona_replies = _load_persona_replies(week, state_root)
    round1_replies = _load_round1_replies(week, state_root)

    budget_config = _PROJECT_ROOT / "config" / "persona_budgets.yaml"
    thresholds_config = _PROJECT_ROOT / "config" / "thresholds.yaml"
    validator_config_path = _PROJECT_ROOT / "config" / "validator.yaml"

    budget_raw = yaml.safe_load(budget_config.read_text(encoding="utf-8")) or {}
    thresholds = yaml.safe_load(thresholds_config.read_text(encoding="utf-8")) or {}
    max_position_weight: float = float(thresholds.get("max_position_weight", 0.20))

    budgets = load_budgets(budget_config)
    v_config = load_validator_config(validator_config_path)
    judge = StubOnMandateJudge()

    tmp_dir = Path(tempfile.mkdtemp(prefix="rtp_r2prep_"))
    try:
        tmp_state = tmp_dir / "state"
        tmp_state.mkdir()

        persona_results = []
        for slug in _PERSONA_SLUGS:
            raw = persona_replies.get(slug)
            if not raw:
                raise RuntimeError(
                    f"prepare-round2: no persona reply for slug={slug!r}."
                )
            budget = get_budget(budgets, slug)
            result = run_persona_research(
                persona_slug=slug,
                week_id=week,
                raw_output=raw,
                mandate="",
                judge=judge,
                budget=budget,
                validator_config=v_config,
                state_root=tmp_state,
            )
            persona_results.append(result)

        debate_cfg: dict = {
            "debate_set_ceiling": budget_raw.get("debate_set_ceiling", 40),
            "max_position_weight": max_position_weight,
        }
        debate_set = construct_debate_set(persona_results, debate_cfg)

        from round_table_portfolio.orchestrator.round1 import capture_round1_stances
        from round_table_portfolio.personas.output_validator import validate_counterfactual_portfolio

        _r1_replies: dict[str, str] = round1_replies or persona_replies or {}
        round1 = capture_round1_stances(
            debate_set,
            persona_results,
            raw_round1_replies=_r1_replies,
            config={"max_position_weight": max_position_weight},
        )

        # 6a. Dissent
        dissent_cfg = load_dissent_config(thresholds_config)
        dissent_result = compute_dissent(round1.stances, debate_set, dissent_cfg)

        # 6b. Outlier selection
        outliers = select_outliers(dissent_result, round1.stances, dissent_cfg)

        # Collect R1 rationales for Component 23.
        r1_rationales: dict[str, dict[str, str]] = {}
        for s in round1.stances:
            p = s.persona; t = s.ticker; r = s.rationale
            r1_rationales.setdefault(p, {})[t] = r

        # 6c. Counterargument assembly
        ca_cfg = load_counterargument_config(thresholds_config)
        counterargument_blocks = assemble_counterarguments(
            outlier_slugs=outliers.selected,
            all_stances=round1.stances,
            rationales=r1_rationales,
            config=ca_cfg,
        )
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # Print the Round-2 briefing.
    print(f"\n{'='*62}")
    print(f"  ROUND-2 BRIEFING — {week}")
    print(f"{'='*62}")
    print(f"  Dissent score: {dissent_result.dissent_score:.4f}  "
          f"(contested={dissent_result.contested_week})")
    print(f"  Selected outliers (2):")
    for slug in outliers.selected:
        div = dissent_result.per_persona_divergence.get(slug, 0.0)
        print(f"    {slug:<30}  divergence={div:.4f}")
    print()
    print("  COUNTERARGUMENT BLOCKS (verbatim — source Round-1 rationales):")
    print()
    for slug in outliers.selected:
        cb = counterargument_blocks[slug]
        print(f"  --- {slug} ---")
        print(f"  Challenged tickers: {', '.join(cb.debated_tickers)}")
        print(f"  Source rationales used ({len(cb.source_rationales)}):")
        for src_persona, src_ticker, _ in cb.source_rationales:
            print(f"    [{src_persona} on {src_ticker}]")
        print()
        print("  Counterargument block (paste into Round-2 prompt):")
        print("  " + "\n  ".join(cb.block.splitlines()))
        print()

    print("  ROUND-2 OUTPUT SCHEMA (required from each outlier):")
    print("  {")
    print('    "round": 2,')
    print('    "rebuttal_narrative": "<how you responded to the counterargument>",')
    print('    "stances": [')
    print('      {')
    print('        "ticker":          "<one ticker from the debate set>",')
    print('        "action":          "ADD" | "REDUCE" | "HOLD" | "EXIT",')
    print(f'        "target_weight":   <float 0.0 – {max_position_weight}>,')
    print('        "confidence":      <int 1–5>,')
    print('        "rationale":       "<strengthened or revised rationale>",')
    print('        "position_change": "defended" | "revised"')
    print('      },')
    print(f'      ...  // restate stances for tickers where you respond to the counterargument')
    print('    ]')
    print("  }")
    print()
    print(f"  Write replies to: state/runs/{week}.round2_replies.json")
    print(f"  Write timing  to: state/runs/{week}.round2_timing.json")
    print()


# ---------------------------------------------------------------------------
# Preview data bundle — captured from temp dir BEFORE cleanup
# ---------------------------------------------------------------------------


class _PreviewData(NamedTuple):
    """All data needed by _render_preview, extracted before the temp dir is removed."""
    week: str
    # Consensus holdings: list of (ticker, weight) sorted by weight desc, CASH last.
    consensus_holdings: list
    # Transcript sections for dissent display (raw markdown text).
    transcript_text: str
    # Per-persona snapshot: slug -> {cash_weight, top_positions: [(ticker, weight), ...]}
    persona_snapshots: dict
    # WeeklyRunResult fields needed for metrics/debate blocks.
    debate_set: list
    num_portfolios_written: int
    num_stances_written: int
    decision_type: str
    metrics: Any
    # Real session timing (from state/runs/<week>.timing.json).
    session_timing: dict
    # M3 dissent / outlier / resynthesis (from WeeklyRunResult — used by Gap-B render).
    dissent: Any          # DissentResult — .dissent_score, .contested_week, .per_persona_divergence
    outliers: Any         # OutlierSelection — .selected (list[str])
    resynthesis: Any      # ResynthesisResult — .delta, .outlier_personas; None when R2 skipped
    num_round2_stances: int
    # Per-outlier defend/revise counts: {slug: {"defended": int, "revised": int}}.
    round2_defend_revise: dict
    # Per-phase timing maps for Gap-C full-cycle render.
    round1_timing: dict   # {} when absent
    judge_timing: dict    # {} when absent
    round2_timing: dict   # {} when absent


def _compute_round2_defend_revise(
    round2_replies_raw: dict[str, str],
    week: str = "unknown",
) -> dict[str, dict[str, int]]:
    """Parse round2_replies_raw and count defended/revised stances per outlier.

    Returns {slug: {"defended": N, "revised": N}}.  Silently skips unparseable
    replies (parse failure → 0/0 for that slug).
    """
    counts: dict[str, dict[str, int]] = {}
    for slug, raw in round2_replies_raw.items():
        try:
            reply = parse_round2_reply(raw, persona_slug=slug, week_id=week)
            defended = sum(1 for s in reply.stances if s.position_change == "defended")
            revised = sum(1 for s in reply.stances if s.position_change == "revised")
            counts[slug] = {"defended": defended, "revised": revised}
        except Exception:
            counts[slug] = {"defended": 0, "revised": 0}
    return counts


def _read_preview_data_from_temp(
    result: WeeklyRunResult,
    tmp_db: Path,
    week: str,
    session_timing: dict[str, float],
    round1_timing: dict[str, float],
    judge_timing: dict[str, float],
    round2_timing: dict[str, float],
    round2_defend_revise: dict[str, dict[str, int]],
) -> _PreviewData:
    """Extract all preview-render data from the temp DB and transcript file.

    Must be called BEFORE the temp directory is cleaned up.
    """
    # --- Consensus holdings from temp ledger ---
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    consensus_rows = conn.execute(
        """
        SELECT h.ticker, h.weight
        FROM portfolios p
        JOIN holdings h ON h.portfolio_id = p.portfolio_id
        WHERE p.week_id = ? AND p.type = 'consensus'
        ORDER BY h.weight DESC
        """,
        (week,),
    ).fetchall()

    # Separate CASH row; sort equity positions by weight desc; CASH at end.
    equity_rows = [(r["ticker"], r["weight"]) for r in consensus_rows if r["ticker"] != "CASH"]
    cash_rows = [(r["ticker"], r["weight"]) for r in consensus_rows if r["ticker"] == "CASH"]
    consensus_holdings = sorted(equity_rows, key=lambda x: -x[1]) + cash_rows

    # --- Per-persona counterfactual snapshots from temp ledger ---
    persona_snapshots: dict[str, dict] = {}
    for slug in _PERSONA_SLUGS:
        rows = conn.execute(
            """
            SELECT h.ticker, h.weight
            FROM portfolios p
            JOIN holdings h ON h.portfolio_id = p.portfolio_id
            WHERE p.week_id = ? AND p.type = ?
            ORDER BY h.weight DESC
            """,
            (week, slug),
        ).fetchall()
        if not rows:
            continue
        cash_w = next((r["weight"] for r in rows if r["ticker"] == "CASH"), 0.0)
        equity_positions = [
            (r["ticker"], r["weight"]) for r in rows if r["ticker"] != "CASH"
        ]
        top_positions = sorted(equity_positions, key=lambda x: -x[1])[:3]
        persona_snapshots[slug] = {
            "cash_weight": cash_w,
            "top_positions": top_positions,
        }
    conn.close()

    # --- Transcript text (for dissent section) ---
    transcript_text = ""
    if result.transcript_path and result.transcript_path.exists():
        transcript_text = result.transcript_path.read_text(encoding="utf-8")

    return _PreviewData(
        week=week,
        consensus_holdings=consensus_holdings,
        transcript_text=transcript_text,
        persona_snapshots=persona_snapshots,
        debate_set=result.debate_set,
        num_portfolios_written=result.num_portfolios_written,
        num_stances_written=result.num_stances_written,
        decision_type=result.decision_type,
        metrics=result.metrics,
        session_timing=session_timing,
        dissent=result.dissent,
        outliers=result.outliers,
        resynthesis=result.resynthesis,
        num_round2_stances=result.num_round2_stances,
        round2_defend_revise=round2_defend_revise,
        round1_timing=round1_timing,
        judge_timing=judge_timing,
        round2_timing=round2_timing,
    )


# ---------------------------------------------------------------------------
# Preview renderer
# ---------------------------------------------------------------------------


def _render_preview(data: _PreviewData) -> str:
    """Build the founder-readable preview from pre-captured temp-run data."""
    week = data.week
    lines: list[str] = [
        f"# Weekly Run Preview — {week}",
        "",
        "This preview was generated against a **throwaway ledger** (real state "
        "unchanged).  The run is deterministic: the commit run will produce "
        "identical portfolio weights.",
        "",
    ]

    # --- Proposed consensus holdings ---
    lines += ["## Proposed Consensus Portfolio", ""]
    if data.consensus_holdings:
        lines += ["| Ticker | Weight |", "|--------|--------|"]
        for ticker, weight in data.consensus_holdings:
            lines.append(f"| {ticker} | {weight:.4f} |")
        total = sum(w for _, w in data.consensus_holdings)
        lines += ["", f"*{len(data.consensus_holdings) - 1} equity positions + CASH  "
                  f"(sum = {total:.4f})*", ""]
    else:
        lines += ["_All personas EXIT — 100% CASH._", ""]

    # --- Round-1 Dissent (recalibrated M3 — from WeeklyRunResult.dissent/outliers) ---
    lines += ["## Round-1 Dissent", ""]
    if data.dissent is not None:
        d = data.dissent
        contested_label = "YES" if d.contested_week else "no"
        lines += [
            f"- **Dissent score:** {d.dissent_score:.4f}",
            f"- **Contested week** (threshold 0.50): {contested_label}",
        ]
        if data.outliers is not None and data.outliers.selected:
            lines += ["", "**Selected outliers (most divergent):**", ""]
            lines += ["| Persona | Divergence |", "|---------|------------|"]
            for slug in data.outliers.selected:
                div = d.per_persona_divergence.get(slug, 0.0)
                lines.append(f"| {slug} | {div:.4f} |")
        lines += [""]
    else:
        lines += ["*(dissent data not available)*", ""]

    # --- Round 2 summary ---
    lines += ["## Round 2", ""]
    if data.num_round2_stances > 0 and data.outliers is not None and data.resynthesis is not None:
        outlier_slugs = data.outliers.selected
        for slug in outlier_slugs:
            dr = data.round2_defend_revise.get(slug, {})
            defended = dr.get("defended", 0)
            revised = dr.get("revised", 0)
            lines.append(f"- **{slug}**: {defended} defended, {revised} revised")
        # Consensus shift summary.
        if data.resynthesis.delta:
            moved = sorted(
                data.resynthesis.delta.items(), key=lambda x: -abs(x[1])
            )
            moved_str = ", ".join(f"{t} ({w:+.4f})" for t, w in moved[:5])
            lines += ["", f"**Consensus shift:** {moved_str}"]
        else:
            lines += ["", "**Consensus shift:** no movement — all outliers defended"]
        lines += [
            "",
            f"*Full R1+R2 debate detail (counterarguments + rebuttal narratives) → "
            f"`state/runs/{week}.debate_transcript.md`*",
            "",
        ]
    else:
        lines += [
            "*(Round-2 data not available — no round2_replies.json provided)*",
            "",
        ]

    # --- Per-persona counterfactual snapshot ---
    lines += ["## Per-Persona Snapshot", ""]
    if data.persona_snapshots:
        lines += ["| Persona | CASH % | Top Positions |", "|---------|--------|---------------|"]
        for slug in _PERSONA_SLUGS:
            snap = data.persona_snapshots.get(slug)
            if snap is None:
                lines.append(f"| {slug} | — | — |")
                continue
            cash_pct = f"{snap['cash_weight']*100:.1f}%"
            top = ", ".join(
                f"{t} ({w*100:.1f}%)" for t, w in snap["top_positions"]
            ) or "—"
            lines.append(f"| {slug} | {cash_pct} | {top} |")
        lines += [""]
    else:
        lines += ["*(no per-persona data available)*", ""]

    # --- Vote tally / debate metrics ---
    lines += [
        "## Debate Metrics",
        f"- Debate set: {len(data.debate_set)} tickers — "
        f"{', '.join(data.debate_set[:10])}"
        + (f" ... (+{len(data.debate_set)-10} more)" if len(data.debate_set) > 10 else ""),
        f"- Portfolios generated: {data.num_portfolios_written} "
        "(1 consensus + 7 counterfactuals)",
        f"- Round-1 stances written: {data.num_stances_written}",
        f"- Decision type (preview): {data.decision_type}",
        "",
    ]

    # --- Engine metrics block ---
    if data.metrics:
        lines += ["## Run Metrics", "", data.metrics.summary_text, ""]

    # --- Full-cycle timing (DEF-003: research + R1 + judges + R2) ---
    if data.session_timing:
        import yaml as _yaml
        _ws_cfg_path = _PROJECT_ROOT / "config" / "web_search.yaml"
        _ws_cfg = _yaml.safe_load(_ws_cfg_path.read_text(encoding="utf-8")) if _ws_cfg_path.exists() else {}
        window_hours: float = float(_ws_cfg.get("window_hours", 5.0))
        proximity_threshold: float = float(_ws_cfg.get("window_proximity_threshold", 0.80))

        research_total = sum(data.session_timing.values())
        r1_total = sum(data.round1_timing.values()) if data.round1_timing else 0.0
        judge_total = sum(data.judge_timing.values()) if data.judge_timing else 0.0
        r2_total = sum(data.round2_timing.values()) if data.round2_timing else 0.0
        combined_total = research_total + r1_total + judge_total + r2_total

        window_secs = window_hours * 3600.0
        window_frac = combined_total / window_secs if window_secs > 0 else 0.0
        window_pct = window_frac * 100.0

        # Reuse the same verdict bands as report_run_metrics (TIGHT_BAND_START=0.60).
        _TIGHT_START = 0.60
        if window_frac >= proximity_threshold:
            verdict = "DOES-NOT-FIT"
        elif window_frac >= _TIGHT_START:
            verdict = "TIGHT"
        else:
            verdict = "FITS"

        lines += [
            "## Full-Cycle Timing (real — measured by session)",
            "",
            "| Phase | Wall-clock (s) | Notes |",
            "|-------|---------------|-------|",
            f"| Research (7 personas) | {research_total:.1f} | from timing.json |",
            f"| Round-1 dispatch | {r1_total:.1f} | "
            + ("from round1_timing.json" if data.round1_timing else "absent — treated as 0") + " |",
            f"| Judge dispatch | {judge_total:.1f} | "
            + ("from judge_timing.json" if data.judge_timing else "absent — treated as 0") + " |",
            f"| Round-2 dispatch | {r2_total:.1f} | "
            + ("from round2_timing.json" if data.round2_timing else "absent — treated as 0") + " |",
            f"| **COMBINED TOTAL** | **{combined_total:.1f}** | |",
            "",
            f"*Combined total: {combined_total:.0f}s "
            f"({combined_total/60:.1f} min) — "
            f"{window_pct:.1f}% of the {window_hours:.0f}-hour weekly window.*",
            "",
            f"**Feasibility verdict: {verdict}** "
            f"(FITS < 60% ≤ TIGHT < {proximity_threshold*100:.0f}% ≤ DOES-NOT-FIT)",
            "",
            "> Per-phase research breakdown (7 personas):",
        ]
        for slug, secs in data.session_timing.items():
            lines.append(f"> - {slug}: {secs:.1f}s")
        lines += [""]

    lines += [
        "---",
        "To commit this run, reply in the session with:",
        '```',
        f'python scripts/weekly_run.py --mode commit --week {week} --founder-reply "approve"',
        '```',
        "Or to override the consensus:",
        '```',
        f'python scripts/weekly_run.py --mode commit --week {week} '
        '--founder-reply "override: <your delta here>"',
        '```',
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Round-2 dispatcher factory
# ---------------------------------------------------------------------------


def _make_round2_dispatcher(
    round2_replies_raw: dict[str, str],
) -> Any:
    """Build a round2_dispatcher callable from pre-captured reply strings.

    The returned callable mirrors the live-session dispatch seam:
        (outlier_slug: str, counterargument_block: str) -> str (raw JSON)

    In the driver (preview + commit), the session has already captured the
    2 outlier replies and written them to round2_replies.json.  This factory
    replays them in order so run_weekly receives the canned JSON without a
    live dispatch.

    Only the outlier_slug argument is used for lookup; counterargument_block
    is accepted but ignored (the session already used it when dispatching).
    """
    def _dispatcher(outlier_slug: str, counterargument_block: str) -> str:
        raw = round2_replies_raw.get(outlier_slug)
        if raw is None:
            raise KeyError(
                f"round2_dispatcher: no pre-captured reply for slug={outlier_slug!r}. "
                f"Available: {sorted(round2_replies_raw)}. "
                "Check state/runs/<week>.round2_replies.json."
            )
        return raw
    return _dispatcher


# ---------------------------------------------------------------------------
# Preview mode
# ---------------------------------------------------------------------------


def run_preview(week: str, state_root: Path) -> None:
    """Execute a throwaway run and write the preview file."""
    persona_replies = _load_persona_replies(week, state_root)
    round1_replies = _load_round1_replies(week, state_root)
    judge_verdicts = _load_judge_verdicts(week, state_root)
    session_timing = _load_timing(week, state_root)
    round1_timing = _load_timing_optional(week, state_root, "round1_timing")
    judge_timing = _load_timing_optional(week, state_root, "judge_timing")
    round2_replies_raw = _load_round2_replies_optional(week, state_root)
    round2_timing = _load_timing_optional(week, state_root, "round2_timing")

    judge = ReplayJudge(judge_verdicts)

    # Build the round2_dispatcher from pre-captured replies (replay pattern).
    _r2_dispatcher = _make_round2_dispatcher(round2_replies_raw) if round2_replies_raw else None

    # Build a temp directory for the throwaway run.
    tmp_dir = Path(tempfile.mkdtemp(prefix="rtp_preview_"))
    preview_data: Optional[_PreviewData] = None
    try:
        tmp_state = tmp_dir / "state"
        tmp_state.mkdir()
        tmp_db = tmp_dir / "ledger.db"
        apply_schema(db_path=tmp_db)

        real_memory_dir = state_root / "memory"
        tmp_memory_dir = tmp_state / "memory"
        if real_memory_dir.exists():
            shutil.copytree(str(real_memory_dir), str(tmp_memory_dir))
        else:
            tmp_memory_dir.mkdir(parents=True)

        result = run_weekly(
            project="round-table-portfolio",
            week_id=week,
            persona_replies=persona_replies,
            round1_replies=round1_replies,
            founder_reply="approve",
            judge=judge,
            round2_dispatcher=_r2_dispatcher,
            per_round1_timing=round1_timing,
            per_judge_timing=judge_timing,
            personas_config=_PROJECT_ROOT / "config" / "personas.yaml",
            budget_config=_PROJECT_ROOT / "config" / "persona_budgets.yaml",
            thresholds_config=_PROJECT_ROOT / "config" / "thresholds.yaml",
            web_search_config=_PROJECT_ROOT / "config" / "web_search.yaml",
            state_root=tmp_state,
            db_path=tmp_db,
        )

        # Gap A: copy the R1+R2 debate transcript to a persistent path BEFORE cleanup.
        # The transcript lives in tmp_state/debates/<week>.md (written by
        # write_round1_transcript + append_round2_transcript).
        persistent_transcript: Optional[Path] = None
        tmp_transcript = result.transcript_path  # Path inside the temp dir
        if tmp_transcript is not None and tmp_transcript.exists():
            persistent_dir = state_root / "runs"
            persistent_dir.mkdir(parents=True, exist_ok=True)
            persistent_transcript = persistent_dir / f"{week}.debate_transcript.md"
            shutil.copy2(str(tmp_transcript), str(persistent_transcript))

        # Capture all preview data BEFORE the temp dir is removed.
        preview_data = _read_preview_data_from_temp(
            result,
            tmp_db,
            week,
            session_timing,
            round1_timing,
            judge_timing,
            round2_timing,
            _compute_round2_defend_revise(round2_replies_raw, week=week),
        )
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    if preview_data is None:
        return

    preview_text = _render_preview(preview_data)

    preview_path = state_root / "runs" / f"{week}.preview.md"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(preview_text, encoding="utf-8")

    print(preview_text)
    print(f"\nPreview written to: {preview_path}")
    if persistent_transcript is not None:
        print(f"Full R1+R2 debate transcript written to: {persistent_transcript}")


# ---------------------------------------------------------------------------
# Commit mode
# ---------------------------------------------------------------------------


def run_commit(week: str, founder_reply: str, state_root: Path) -> None:
    """Execute the real run against the live ledger."""
    persona_replies = _load_persona_replies(week, state_root)
    round1_replies = _load_round1_replies(week, state_root)
    judge_verdicts = _load_judge_verdicts(week, state_root)
    timing = _load_timing(week, state_root)
    round1_timing = _load_timing_optional(week, state_root, "round1_timing")
    judge_timing = _load_timing_optional(week, state_root, "judge_timing")
    round2_replies_raw = _load_round2_replies_optional(week, state_root)
    _round2_dispatcher = _make_round2_dispatcher(round2_replies_raw) if round2_replies_raw else None

    judge = ReplayJudge(judge_verdicts)
    real_db = state_root / "ledger.db"

    result = run_weekly(
        project="round-table-portfolio",
        week_id=week,
        persona_replies=persona_replies,
        round1_replies=round1_replies,
        founder_reply=founder_reply,
        judge=judge,
        round2_dispatcher=_round2_dispatcher,
        per_round1_timing=round1_timing,
        per_judge_timing=judge_timing,
        personas_config=_PROJECT_ROOT / "config" / "personas.yaml",
        budget_config=_PROJECT_ROOT / "config" / "persona_budgets.yaml",
        thresholds_config=_PROJECT_ROOT / "config" / "thresholds.yaml",
        web_search_config=_PROJECT_ROOT / "config" / "web_search.yaml",
        state_root=state_root,
        db_path=real_db,
    )

    # Verify the 8-portfolio invariant.
    assert result.num_portfolios_written == 8, (
        f"Expected 8 portfolios written, got {result.num_portfolios_written}. "
        "This is a fatal invariant violation — check the ledger transaction."
    )

    # Check memory, briefing, and validator-claim files.
    memory_dir = state_root / "memory"
    briefing_dir = state_root / "runs" / f"{week}-memory"
    claims_dir = state_root / "reports" / week / "validator_claims"

    memory_files = [slug for slug in _PERSONA_SLUGS if (memory_dir / f"{slug}.md").exists()]
    briefing_files = [slug for slug in _PERSONA_SLUGS if (briefing_dir / f"{slug}.md").exists()]
    claim_files = [slug for slug in _PERSONA_SLUGS if (claims_dir / f"{slug}.json").exists()]

    print(f"\n{'='*62}")
    print(f"  COMMIT COMPLETE — {week}")
    print(f"{'='*62}")
    print(f"  Decision:          {result.decision_type}")
    if result.decision_delta:
        print(f"  Delta:             {result.decision_delta}")
    print(f"  Portfolios:        {result.num_portfolios_written} (1 consensus + 7 counterfactual)")
    print(f"  Round-1 stances:   {result.num_stances_written}")
    print(f"  Round-2 stances:   {result.num_round2_stances} (2 outliers)")
    print(f"  Persona reports:   {result.num_persona_reports}")
    print(f"  Transcript:        {result.transcript_path}")
    print(f"  Memory updates:    {len(memory_files)}/7")
    print(f"  Briefing files:    {len(briefing_files)}/7  "
          f"({briefing_dir})")
    print(f"  Validator claims:  {len(claim_files)}/7")
    if result.resynthesis and result.resynthesis.delta:
        moved = sorted(
            result.resynthesis.delta.items(), key=lambda x: -abs(x[1])
        )
        delta_str = ", ".join(f"{t} ({d:+.4f})" for t, d in moved[:5])
        print(f"  Consensus shift:   {delta_str}")
    else:
        print(f"  Consensus shift:   none (outliers defended all positions)")
    print()

    if result.metrics:
        print(result.metrics.summary_text)

    # Session-measured timing (from the session-written file).
    total_session_time = sum(timing.values())
    print(f"\nSession-measured total wall time: {total_session_time:.1f}s")

    if len(memory_files) < 7:
        missing = set(_PERSONA_SLUGS) - set(memory_files)
        print(f"\nWARNING: Memory files missing for: {missing}")
    if len(claim_files) < 7:
        missing = set(_PERSONA_SLUGS) - set(claim_files)
        print(f"WARNING: Validator-claim files missing for: {missing}")


# ---------------------------------------------------------------------------
# Persona slug list (matches config/personas.yaml)
# ---------------------------------------------------------------------------

_PERSONA_SLUGS = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Session driver for the round-table /weekly-run cycle.",
    )
    parser.add_argument(
        "--week",
        default=_current_week_label(),
        help="ISO week label e.g. 2026-W23.  Defaults to current calendar week.",
    )
    parser.add_argument(
        "--mode",
        choices=["prepare-round1", "prepare-round2", "preview", "commit"],
        required=True,
        help=(
            "prepare-round1: compute debate set from persona_replies and write "
            "debate_set.json (no real state side-effects).  "
            "prepare-round2: run dissent + outlier selection + counterargument "
            "assembly deterministically, printing the 2 outlier slugs and their "
            "counterargument blocks so the session can dispatch Round-2.  "
            "preview: throwaway run (incl. Round-2 if round2_replies.json exists), "
            "prints founder-readable summary.  "
            "commit: real run against state/ledger.db."
        ),
    )
    parser.add_argument(
        "--founder-reply",
        default=None,
        help='Required for --mode commit.  e.g. "approve" or "override: reduce AAPL to 5%%".',
    )
    parser.add_argument(
        "--state-root",
        default=None,
        help="Override state/ directory (default: <project_root>/state/).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    state_root = Path(args.state_root) if args.state_root else _PROJECT_ROOT / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "prepare-round1":
        run_prepare_round1(args.week, state_root)
    elif args.mode == "prepare-round2":
        run_prepare_round2(args.week, state_root)
    elif args.mode == "preview":
        run_preview(args.week, state_root)
    elif args.mode == "commit":
        if not args.founder_reply:
            print("ERROR: --founder-reply is required for --mode commit.", file=sys.stderr)
            sys.exit(1)
        run_commit(args.week, args.founder_reply, state_root)


if __name__ == "__main__":
    main()
