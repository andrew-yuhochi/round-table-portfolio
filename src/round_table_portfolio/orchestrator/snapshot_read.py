# snapshot_read.py — Read-only query helper over shortlist_price_snapshots (Component 39).
#
# Responsibility: single read layer over shortlist_price_snapshots, mirroring the
# derive-at-read discipline of Component 30 (ledger_read_api, built in M6).
#
# Every result is derived at query time from existing rows — NO new columns,
# NO writes back.  The caller MUST open the connection read-only (SQLite URI
# "?mode=ro"); this module issues only SELECTs and never calls conn.commit().
# Read-only is enforced by the caller's mode=ro connection (SQLite engine
# guarantee) and verified by the no-write-SQL + mode=ro tests in the suite.
#
# Consumed by:
#   - M6 dashboard "what we passed on" per-stock view (future)
#   - MVP Seasonal Review missed-opportunities analysis (future)
#   - TASK-M5-003 validation harness (immediate)

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickerSnapshot:
    """One week's price observation for a tracked ticker."""

    week_id: str
    snapshot_date: str
    price: float


@dataclass
class TickerTrack:
    """Full read-time view for one tracked ticker.

    Fields derived at read time (never stored as columns):
    - accepted: True if the consensus ever held this ticker.
    - return_since_first_proposed: (latest_price / first_price) - 1 when ≥2 snapshots
      exist; None when only one snapshot has been captured.
    """

    ticker: str
    first_proposed_week: str
    surfacing_personas: list[str]          # personas that surfaced ticker in its first week
    snapshots: list[TickerSnapshot]        # ordered by week_id ASC (chronological)
    accepted: bool                         # ever held by consensus
    return_since_first_proposed: Optional[float]  # derived at read time; None when <2 snapshots


@dataclass
class MissedOpportunity:
    """One entry in the missed-opportunities ranking.

    Only rejected tickers (never held by consensus) appear here, ranked by
    return_since_first_proposed descending.
    """

    ticker: str
    first_proposed_week: str
    surfacing_personas: list[str]
    return_since_first_proposed: float     # always present — ranking key


@dataclass
class SnapshotReadResult:
    """Full read-time result for all tracked tickers belonging to user_id."""

    ticker_tracks: dict[str, TickerTrack]           # keyed by ticker
    missed_opportunities: list[MissedOpportunity]   # rejected names, ranked by return DESC


# ---------------------------------------------------------------------------
# Internal query helpers
# ---------------------------------------------------------------------------


def _query_snapshots(
    conn: sqlite3.Connection,
    user_id: str,
) -> dict[str, list[TickerSnapshot]]:
    """Return all snapshots grouped by ticker, ordered chronologically.

    Key: ticker (upper-case string).
    Value: list of TickerSnapshot ordered by week_id ASC.
    """
    rows = conn.execute(
        """
        SELECT ticker, week_id, snapshot_date, price
        FROM   shortlist_price_snapshots
        WHERE  user_id = ?
        ORDER  BY ticker, week_id
        """,
        (user_id,),
    ).fetchall()

    result: dict[str, list[TickerSnapshot]] = {}
    for ticker, week_id, snapshot_date, price in rows:
        t = ticker.upper()
        if t not in result:
            result[t] = []
        result[t].append(TickerSnapshot(week_id=week_id, snapshot_date=snapshot_date, price=float(price)))
    return result


def _query_first_proposed(
    conn: sqlite3.Connection,
    user_id: str,
) -> dict[str, str]:
    """Return {ticker: first_week_id} — MIN(week_id) per ticker from snapshots."""
    rows = conn.execute(
        """
        SELECT ticker, MIN(week_id) AS first_week
        FROM   shortlist_price_snapshots
        WHERE  user_id = ?
        GROUP  BY ticker
        """,
        (user_id,),
    ).fetchall()
    return {ticker.upper(): first_week for ticker, first_week in rows}


def _query_surfacing_personas(
    conn: sqlite3.Connection,
    first_proposed: dict[str, str],
    user_id: str,
) -> dict[str, list[str]]:
    """Return {ticker: [persona, ...]} for the first-proposed week of each ticker.

    "Surfacing personas" = personas that listed the ticker in persona_shortlists
    in the ticker's first-proposed week.  Ordered alphabetically for stability.
    """
    if not first_proposed:
        return {}

    result: dict[str, list[str]] = {}
    for ticker, first_week in first_proposed.items():
        rows = conn.execute(
            """
            SELECT DISTINCT persona
            FROM   persona_shortlists
            WHERE  ticker = ?
              AND  week_id = ?
              AND  user_id = ?
            ORDER  BY persona
            """,
            (ticker, first_week, user_id),
        ).fetchall()
        result[ticker] = [row[0] for row in rows]
    return result


def _query_accepted_tickers(
    conn: sqlite3.Connection,
    user_id: str,
) -> set[str]:
    """Return the set of tickers the consensus ever held.

    A ticker is "accepted" when it appears in holdings for a consensus portfolio
    (portfolios.type = 'consensus') owned by user_id, with action IN ('add', 'hold').
    'reduce' and 'exit' rows are evidence the ticker was PREVIOUSLY held — still accepted.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT h.ticker
        FROM   holdings h
        JOIN   portfolios p ON h.portfolio_id = p.portfolio_id
        WHERE  p.type    = 'consensus'
          AND  p.user_id = ?
          AND  h.ticker != 'CASH'
        """,
        (user_id,),
    ).fetchall()
    return {row[0].upper() for row in rows}


# ---------------------------------------------------------------------------
# Derived computations
# ---------------------------------------------------------------------------


def _compute_return(snapshots: list[TickerSnapshot]) -> Optional[float]:
    """Compute (latest_price / first_price) - 1.  Returns None if < 2 snapshots."""
    if len(snapshots) < 2:
        return None
    first_price = snapshots[0].price
    latest_price = snapshots[-1].price
    if first_price <= 0:
        return None
    return (latest_price / first_price) - 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_snapshot_tracks(
    conn: sqlite3.Connection,
    user_id: str = "andrew",
) -> SnapshotReadResult:
    """Query all shortlisted/tracked tickers and derive read-time fields.

    Parameters
    ----------
    conn:
        Open SQLite connection.  For production use, open with
        ``sqlite3.connect("file:ledger.db?mode=ro", uri=True)`` so the
        underlying SQLite engine enforces the read-only contract.  This module
        issues only SELECT statements and never calls conn.commit().
    user_id:
        Owner identity for which to query snapshots and portfolio data.

    Returns
    -------
    SnapshotReadResult
        - ticker_tracks: per-ticker time series + derived fields.
        - missed_opportunities: rejected names ranked by return DESC.
    """
    snapshots_by_ticker = _query_snapshots(conn, user_id)
    if not snapshots_by_ticker:
        logger.info("snapshot_read: no snapshots found for user_id=%s", user_id)
        return SnapshotReadResult(ticker_tracks={}, missed_opportunities=[])

    first_proposed = _query_first_proposed(conn, user_id)
    surfacing_personas = _query_surfacing_personas(conn, first_proposed, user_id)
    accepted_tickers = _query_accepted_tickers(conn, user_id)

    ticker_tracks: dict[str, TickerTrack] = {}
    for ticker, snaps in snapshots_by_ticker.items():
        ret = _compute_return(snaps)
        track = TickerTrack(
            ticker=ticker,
            first_proposed_week=first_proposed.get(ticker, snaps[0].week_id),
            surfacing_personas=surfacing_personas.get(ticker, []),
            snapshots=snaps,
            accepted=ticker in accepted_tickers,
            return_since_first_proposed=ret,
        )
        ticker_tracks[ticker] = track

    # Missed opportunities: rejected tickers with ≥2 snapshots (return is computable),
    # ranked by return_since_first_proposed descending.
    missed: list[MissedOpportunity] = []
    for ticker, track in ticker_tracks.items():
        if not track.accepted and track.return_since_first_proposed is not None:
            missed.append(
                MissedOpportunity(
                    ticker=ticker,
                    first_proposed_week=track.first_proposed_week,
                    surfacing_personas=track.surfacing_personas,
                    return_since_first_proposed=track.return_since_first_proposed,
                )
            )

    missed.sort(key=lambda m: m.return_since_first_proposed, reverse=True)

    logger.info(
        "snapshot_read: user=%s | tracked=%d | accepted=%d | rejected=%d | "
        "missed_opps_with_return=%d",
        user_id,
        len(ticker_tracks),
        sum(1 for t in ticker_tracks.values() if t.accepted),
        sum(1 for t in ticker_tracks.values() if not t.accepted),
        len(missed),
    )

    return SnapshotReadResult(
        ticker_tracks=ticker_tracks,
        missed_opportunities=missed,
    )


def query_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    user_id: str = "andrew",
) -> Optional[TickerTrack]:
    """Convenience single-ticker query.  Returns None if ticker has no snapshots."""
    result = query_snapshot_tracks(conn, user_id)
    return result.ticker_tracks.get(ticker.upper())
