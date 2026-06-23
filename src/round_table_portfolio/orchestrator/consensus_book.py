"""Component 12 M6 — consensus_book_loader.

Loads the most-recent prior consensus book from the ledger and renders a
structured prompt block for injection into each persona's turn-start prompt.

Design mirrors Component 27 (briefing_builder): Python BUILDS the block and
persists it to ``state/runs/<week>-memory/consensus_book.md``; the session
reads the file and INJECTS it alongside the memory briefing when dispatching
each persona.

Critical invariants
-------------------
1. **Commit-before-reveal boundary preserved.**  This module reads ONLY
   prior-week committed ``holdings`` / ``portfolios`` rows.  It has NO read
   path to the current week's ``agent_stances`` (which do not exist yet when
   this is called — Round 1 has not run).  Enforced structurally: the SQL
   ``WHERE p.week_id != :current_week_id`` guard makes any current-week leak
   a hard SQL-level miss.

2. **No new table, no new column.**  Reads ``portfolios`` + ``holdings``
   (already-committed rows, existing schema).

3. **No new subagent dispatch.**  Pure SQL read + string rendering — zero
   added window-budget cost (Critical Component #3).

4. **Week-one degradation.**  When no prior consensus exists (first-ever run),
   ``load_current_consensus_book`` returns an explicit week-one note, NOT
   silent ``None`` and NOT a crash.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sqlite3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@dataclass
class ConsensusBookResult:
    """Output of ``load_current_consensus_book``.

    ``week_set``       — the week_id the consensus was recorded (e.g. "2026-W25"),
                         or None when no prior consensus exists.
    ``holdings``       — dict mapping ticker → weight (includes 'CASH').
                         Empty when no prior consensus exists.
    ``block_text``     — the rendered markdown block, ready for session injection.
                         Contains the explicit week-one note when week_set is None.
    ``output_path``    — path where the block was persisted
                         (``state/runs/<week>-memory/consensus_book.md``), or None
                         when persist=False.
    """

    week_set: Optional[str]
    holdings: dict[str, float] = field(default_factory=dict)
    block_text: str = ""
    output_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# SQL — read the latest committed consensus portfolio + its holdings
# ---------------------------------------------------------------------------

_LATEST_CONSENSUS_WEEK_SQL = """
    SELECT p.week_id
    FROM   portfolios p
    WHERE  p.type    = 'consensus'
      AND  p.user_id = :user_id
      AND  p.week_id != :current_week_id
    ORDER BY p.week_id DESC
    LIMIT 1
"""

_CONSENSUS_HOLDINGS_SQL = """
    SELECT h.ticker, h.weight
    FROM   holdings   h
    JOIN   portfolios p ON h.portfolio_id = p.portfolio_id
    WHERE  p.type    = 'consensus'
      AND  p.user_id = :user_id
      AND  p.week_id = :week_id
    ORDER BY h.weight DESC, h.ticker ASC
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


_WEEK_ONE_NOTE = (
    "**No standing consensus book yet — this is the first week.**\n"
    "Propose holdings from scratch based on your mandate and research.\n"
)


def _render_book_block(week_set: str, holdings: dict[str, float]) -> str:
    """Render the consensus book as a markdown block for prompt injection.

    The block states: the week it was set, all holdings (sorted by weight
    descending, CASH last), and a one-line framing instruction.
    """
    lines = [
        f"# Standing Consensus Book (set {week_set})",
        "",
        "This is the portfolio the panel currently holds. "
        "Reason FROM it — argue what to KEEP, CHANGE, or EXIT and "
        "WHY the medium-term thesis supports the change.",
        "",
        "| Ticker | Weight |",
        "|--------|--------|",
    ]

    # Sort: equity positions (non-CASH) by weight desc, CASH at the end.
    equities = {t: w for t, w in holdings.items() if t != "CASH"}
    cash_weight = holdings.get("CASH", 0.0)

    for ticker, weight in sorted(equities.items(), key=lambda x: -x[1]):
        lines.append(f"| {ticker} | {weight:.1%} |")
    lines.append(f"| CASH | {cash_weight:.1%} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_current_consensus_book(
    conn: sqlite3.Connection,
    *,
    current_week_id: str,
    user_id: str = "andrew",
    runs_dir: Path = Path("state/runs"),
    persist: bool = True,
) -> ConsensusBookResult:
    """Load the latest committed consensus book and render the prompt block.

    Queries ``portfolios`` + ``holdings`` for the most-recent prior week's
    ``consensus`` portfolio (``week_id != current_week_id``, ordered DESC).

    Args:
        conn:             Read-only SQLite connection (caller owns it; this
                          function issues only SELECTs, never commits).
        current_week_id:  The week being run now.  Excluded from the search
                          to preserve commit-before-reveal: no current-week
                          rows can exist yet (they are written later in the
                          same transaction), but the guard makes the boundary
                          a hard SQL constraint, not just a timing assumption.
        user_id:          Owner filter (default "andrew").
        runs_dir:         Root of the state/runs tree (default ``state/runs``).
        persist:          When True (default) write the block to
                          ``<runs_dir>/<current_week_id>-memory/consensus_book.md``.

    Returns:
        ConsensusBookResult.  On first-ever week (no prior consensus): week_set=None,
        holdings={}, block_text=week-one note.  On normal week: week_set=prior week_id,
        holdings=full dict, block_text=rendered table.
    """
    # --- 1. Find the most-recent prior consensus week ---
    row = conn.execute(
        _LATEST_CONSENSUS_WEEK_SQL,
        {"user_id": user_id, "current_week_id": current_week_id},
    ).fetchone()

    if row is None:
        block_text = _WEEK_ONE_NOTE
        logger.info(
            "consensus_book: no prior consensus found for user=%s week=%s — week-one note emitted",
            user_id,
            current_week_id,
        )
        result = ConsensusBookResult(
            week_set=None,
            holdings={},
            block_text=block_text,
        )
    else:
        prior_week_id: str = row[0]

        # --- 2. Fetch all holdings for that consensus week ---
        holdings_rows = conn.execute(
            _CONSENSUS_HOLDINGS_SQL,
            {"user_id": user_id, "week_id": prior_week_id},
        ).fetchall()

        holdings: dict[str, float] = {ticker: weight for ticker, weight in holdings_rows}

        block_text = _render_book_block(prior_week_id, holdings)
        logger.info(
            "consensus_book: loaded prior consensus week=%s user=%s tickers=%d",
            prior_week_id,
            user_id,
            len(holdings),
        )
        result = ConsensusBookResult(
            week_set=prior_week_id,
            holdings=holdings,
            block_text=block_text,
        )

    # --- 3. Persist ---
    if persist:
        out_dir = runs_dir / f"{current_week_id}-memory"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "consensus_book.md"
        _atomic_write(out_path, result.block_text)
        result = ConsensusBookResult(
            week_set=result.week_set,
            holdings=result.holdings,
            block_text=result.block_text,
            output_path=out_path,
        )
        logger.debug(
            "consensus_book: persisted week=%s path=%s chars=%d",
            current_week_id,
            out_path,
            len(result.block_text),
        )

    return result


# ---------------------------------------------------------------------------
# Atomic write (mirrors briefing_builder._atomic_write)
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    dir_ = path.parent
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
