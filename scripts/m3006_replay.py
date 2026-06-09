"""m3006_replay.py — One-off harness to produce the persistent R1+R2 debate
transcript for the M3-006 founder gate (2026-W24).

Problem context
---------------
`preview` mode in weekly_run.py runs the full R1+R2 cycle against a throwaway
temp dir and then shutil.rmtree's it, so the transcript is discarded.
`commit` mode can't run because 2026-W24 already exists in the live ledger
(INSERT-only write path → round=1 UNIQUE collision).

This script mirrors exactly what run_preview does but uses a PERSISTENT state
dir (state/runs/m3006/state/) so the transcript survives.  The live ledger
(state/ledger.db) is never touched.

Output
------
state/runs/m3006/state/debates/2026-W24.md   — transcript written by engine
state/runs/2026-W24.debate_transcript.md     — copy to founder-facing path
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — mirrors weekly_run.py
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Imports from production code (reuse driver helpers directly)
# ---------------------------------------------------------------------------

from round_table_portfolio.orchestrator.weekly_run import run_weekly, WeeklyRunResult
from round_table_portfolio.personas.output_validator import ReplayJudge
from round_table_portfolio.storage.apply_schema import apply_schema

# Reuse the driver's loader helpers and dispatcher factory by importing them
# from the weekly_run script module.  We add the scripts/ dir to sys.path for
# this purpose.
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from weekly_run import (  # type: ignore[import]
    _load_persona_replies,
    _load_round1_replies,
    _load_judge_verdicts,
    _load_timing,
    _load_timing_optional,
    _load_round2_replies_optional,
    _make_round2_dispatcher,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEEK = "2026-W24"

# Real state root (where all 2026-W24.*.json inputs live).
_REAL_STATE_ROOT = _PROJECT_ROOT / "state"

# Persistent replay state root — NEVER cleaned up.
_REPLAY_STATE = _PROJECT_ROOT / "state" / "runs" / "m3006" / "state"
_REPLAY_DB = _REPLAY_STATE / "ledger.db"

# Founder-facing output path.
_TRANSCRIPT_DEST = _PROJECT_ROOT / "state" / "runs" / f"{WEEK}.debate_transcript.md"

# Config paths.
_PERSONAS_CONFIG = _PROJECT_ROOT / "config" / "personas.yaml"
_BUDGET_CONFIG = _PROJECT_ROOT / "config" / "persona_budgets.yaml"
_THRESHOLDS_CONFIG = _PROJECT_ROOT / "config" / "thresholds.yaml"
_WEB_SEARCH_CONFIG = _PROJECT_ROOT / "config" / "web_search.yaml"


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"[m3006_replay] Loading inputs for week={WEEK}")

    # Load all replay inputs from the REAL state/runs/ directory.
    persona_replies = _load_persona_replies(WEEK, _REAL_STATE_ROOT)
    round1_replies = _load_round1_replies(WEEK, _REAL_STATE_ROOT)
    judge_verdicts = _load_judge_verdicts(WEEK, _REAL_STATE_ROOT)
    session_timing = _load_timing(WEEK, _REAL_STATE_ROOT)
    round1_timing = _load_timing_optional(WEEK, _REAL_STATE_ROOT, "round1_timing")
    judge_timing = _load_timing_optional(WEEK, _REAL_STATE_ROOT, "judge_timing")
    round2_replies_raw = _load_round2_replies_optional(WEEK, _REAL_STATE_ROOT)

    print(f"[m3006_replay] persona_replies: {sorted(persona_replies)}")
    print(f"[m3006_replay] round1_replies:  {sorted(round1_replies)}")
    print(f"[m3006_replay] judge_verdicts:  {sorted(judge_verdicts)}")
    print(f"[m3006_replay] round2_replies:  {sorted(round2_replies_raw)}")
    print(f"[m3006_replay] round1_timing:   {round1_timing}")
    print(f"[m3006_replay] judge_timing:    {judge_timing}")

    # Build judge and round2_dispatcher.
    judge = ReplayJudge(judge_verdicts)
    round2_dispatcher = _make_round2_dispatcher(round2_replies_raw) if round2_replies_raw else None

    # Prepare the persistent replay state dir.
    _REPLAY_STATE.mkdir(parents=True, exist_ok=True)

    # Remove stale DB if present (idempotent re-runs).
    if _REPLAY_DB.exists():
        _REPLAY_DB.unlink()
        print(f"[m3006_replay] Removed stale DB at {_REPLAY_DB}")

    apply_schema(db_path=_REPLAY_DB)
    print(f"[m3006_replay] Schema applied to {_REPLAY_DB}")

    # Mirror the memory copy that run_preview does.
    real_memory_dir = _REAL_STATE_ROOT / "memory"
    replay_memory_dir = _REPLAY_STATE / "memory"
    if replay_memory_dir.exists():
        shutil.rmtree(str(replay_memory_dir))
    if real_memory_dir.exists():
        shutil.copytree(str(real_memory_dir), str(replay_memory_dir))
        print(f"[m3006_replay] Memory dir copied: {real_memory_dir} -> {replay_memory_dir}")
    else:
        replay_memory_dir.mkdir(parents=True)
        print(f"[m3006_replay] No real memory dir found; created empty dir.")

    print(f"[m3006_replay] Running run_weekly (persistent state_root={_REPLAY_STATE})")

    result: WeeklyRunResult = run_weekly(
        project="round-table-portfolio",
        week_id=WEEK,
        persona_replies=persona_replies,
        round1_replies=round1_replies,
        founder_reply="approve",
        judge=judge,
        round2_dispatcher=round2_dispatcher,
        per_round1_timing=round1_timing,
        per_judge_timing=judge_timing,
        personas_config=_PERSONAS_CONFIG,
        budget_config=_BUDGET_CONFIG,
        thresholds_config=_THRESHOLDS_CONFIG,
        web_search_config=_WEB_SEARCH_CONFIG,
        state_root=_REPLAY_STATE,
        db_path=_REPLAY_DB,
    )

    print(f"[m3006_replay] run_weekly complete.")
    print(f"  decision_type:          {result.decision_type}")
    print(f"  num_portfolios_written: {result.num_portfolios_written}")
    print(f"  num_stances_written:    {result.num_stances_written}")
    print(f"  num_round2_stances:     {result.num_round2_stances}")
    print(f"  transcript_path:        {result.transcript_path}")

    # Locate the transcript (engine writes to state_root/debates/<week>.md).
    engine_transcript = _REPLAY_STATE / "debates" / f"{WEEK}.md"
    if not engine_transcript.exists():
        print(f"[m3006_replay] ERROR: transcript not found at {engine_transcript}")
        # Also check result.transcript_path in case the engine put it elsewhere.
        if result.transcript_path and result.transcript_path.exists():
            engine_transcript = result.transcript_path
            print(f"[m3006_replay] Found transcript via result.transcript_path: {engine_transcript}")
        else:
            print("[m3006_replay] FATAL: transcript missing from both expected paths.")
            sys.exit(1)

    # Copy to founder-facing path.
    _TRANSCRIPT_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(engine_transcript), str(_TRANSCRIPT_DEST))
    print(f"[m3006_replay] Transcript copied to: {_TRANSCRIPT_DEST}")

    # Print full-cycle timing summary (DEF-003).
    research_total = sum(session_timing.values())
    round1_total = sum(round1_timing.values()) if round1_timing else 0.0
    judge_total = sum(judge_timing.values()) if judge_timing else 0.0
    round2_timing_raw = _load_timing_optional(WEEK, _REAL_STATE_ROOT, "round2_timing")
    round2_total = sum(round2_timing_raw.values()) if round2_timing_raw else 0.0
    full_cycle_total = research_total + round1_total + judge_total + round2_total
    window_secs = 5.0 * 3600
    pct_window = (full_cycle_total / window_secs) * 100
    verdict = (
        "FITS" if pct_window < 80 else
        "TIGHT" if pct_window < 100 else
        "DOES-NOT-FIT"
    )

    print()
    print("=" * 62)
    print("  FULL-CYCLE METRICS (DEF-003)")
    print("=" * 62)
    print(f"  Research total:    {research_total:.1f}s  ({research_total/60:.1f} min)")
    print(f"  Round-1 total:     {round1_total:.1f}s  (timing file: {'found' if round1_timing else 'missing — 0s assumed'})")
    print(f"  Judge total:       {judge_total:.1f}s  (timing file: {'found' if judge_timing else 'missing — 0s assumed'})")
    print(f"  Round-2 total:     {round2_total:.1f}s  ({', '.join(f'{k}={v}s' for k,v in round2_timing_raw.items())})")
    print(f"  Full-cycle total:  {full_cycle_total:.1f}s  ({full_cycle_total/60:.1f} min)")
    print(f"  % of 5h window:    {pct_window:.1f}%")
    print(f"  Verdict:           {verdict}")
    print()

    if result.metrics:
        print("  RunMetricsReport (engine internal — phases near-zero due to wiring gap):")
        print("  " + "\n  ".join(result.metrics.summary_text.splitlines()))
        print()

    # Resynthesis / consensus shift.
    if result.resynthesis and result.resynthesis.delta:
        moved = sorted(result.resynthesis.delta.items(), key=lambda x: -abs(x[1]))
        print("  Consensus shift (provisional → final):")
        for t, d in moved:
            print(f"    {t}: {d:+.4f}")
    else:
        print("  Consensus shift: none — outliers defended all positions (no-op expected)")
    print()

    print(f"[m3006_replay] Done. Transcript at: {_TRANSCRIPT_DEST}")


if __name__ == "__main__":
    main()
