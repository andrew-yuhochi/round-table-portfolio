#!/usr/bin/env python3
"""Demo script: run the M4 memory accumulation harness end-to-end.

Executes the full 4-week sequence (WEEK-A synthetic → WEEK-B backfill →
WEEK-C conviction-shift → WEEK-D real 2026-W24) in an isolated temp workspace,
prints the assertion matrix, and writes the founder-readable report to
state/runs/m4-harness-<timestamp>/.

Usage (from project root with venv active):

    STUB_ALLOW=1 python scripts/run_accumulation_harness.py

The script does NOT touch state/memory/ or state/ledger.db.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project src is on the path when run directly.
_project_root = Path(__file__).parents[1]
sys.path.insert(0, str(_project_root / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("run_accumulation_harness")

os.environ["STUB_ALLOW"] = "1"


def main() -> int:
    from round_table_portfolio.orchestrator.accumulation_harness import (
        HarnessWorkspace,
        build_harness_report,
        run_accumulation_harness,
        WEEK_A, WEEK_B, WEEK_C, WEEK_D,
    )

    # Create a persistent output directory under state/runs/ for the founder to read.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_root = _project_root / "state" / "runs" / f"m4-harness-{ts}"
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("M4 MEMORY ACCUMULATION HARNESS")
    logger.info("Output: %s", output_root)
    logger.info("=" * 60)

    # Run in a temp workspace (isolated — never touches real state/).
    with tempfile.TemporaryDirectory() as td:
        ws = HarnessWorkspace(root=Path(td) / "harness")
        logger.info("Isolated workspace: %s", ws.root)

        result = run_accumulation_harness(ws)

        # Write the founder-readable report into the persistent output dir.
        report_path = build_harness_report(result, output_dir=output_root)

    # Print assertion matrix to stdout.
    print()
    print("=" * 60)
    print("ACCUMULATION-CORRECTNESS ASSERTION MATRIX")
    print("=" * 60)

    headers = ["Week", "C1a write-bk", "C1b backfill", "C2 round-trip",
               "C3a leakage", "C3b windowed", "C4 digest", "C5 corrupt", "All"]
    col_widths = [10, 13, 12, 13, 11, 12, 10, 10, 6]
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))

    for week_id in [WEEK_A, WEEK_B, WEEK_C, WEEK_D]:
        ba = result.week_assertions.get(week_id)
        if ba is None:
            print(f"{week_id:<10} | (not run)")
            continue

        def badge(b: bool) -> str:
            return "PASS" if b else "FAIL"

        cells = [
            week_id,
            badge(ba.c1_writeback_appended),
            badge(ba.c1_backfill_correct),
            badge(ba.c2_round_trip),
            badge(ba.c3_no_leakage),
            badge(ba.c3_windowed_content),
            badge(ba.c4_digest_attribution),
            badge(ba.c5_no_corruption),
            badge(ba.all_passed),
        ]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, col_widths)))

    print()
    final_badge = "PASS" if result.final_state_equals_predicted else "FAIL"
    print(f"Final state == predicted: {final_badge}")
    print(f"  Detail: {result.final_state_note}")
    print()

    overall = "ALL PASS" if result.all_passed else "FAIL"
    print(f"OVERALL: {overall}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")

    if result.errors:
        print()
        print("ERRORS:")
        for err in result.errors:
            print(f"  - {err}")

    print()
    print(f"Founder report: {report_path}")
    print()
    print("=" * 60)

    if not result.all_passed:
        print("FAIL — one or more assertions did not pass.")
        return 1

    print("SUCCESS — 100% of accumulation-correctness assertions pass.")
    print("Option-A close bar: MET.")
    print()
    print("Next step: founder reads the report above and records sign-off in")
    print("  docs/poc/quality-logs/TASK-M4-006.md  §AC-4 Founder Interpretability Gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
