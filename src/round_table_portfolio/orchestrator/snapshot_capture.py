# snapshot_capture.py — Weekly per-stock price snapshot capture writer (Component 38).
#
# Responsibility: After every weekly ledger commit, capture a current-price
# snapshot for every ticker any persona surfaced that week — accepted by the
# consensus AND rejected by the debate — plus any previously-first-proposed
# tickers still inside their 104-week tracking window.  Writes exactly one
# shortlist_price_snapshots row per (week_id, ticker, user_id).
#
# Called by: Gate-4 test harness (this task); wired into the orchestrator
# post-commit step in TASK-M5-003.
#
# Key design decisions:
# - Tracking perimeter = (this week's shortlist ∪ debate-set tickers)
#   ∪ (prior-first-proposed tickers still inside 104-week window).
#   Bounded to ~100–200 unique tickers — NEVER the full ~500 universe.
# - ONE batched Alpaca multi-symbol call over the unique tracking set.
#   Reuses _alpaca_auth_headers, _ALPACA_BARS_URL from price_cache.py.
# - Alpaca misses: NAMED in the capture-summary log — never a NULL-price row.
# - snapshot_date: taken from the most-recent bar date Alpaca returns for
#   each ticker (may lag the run date by one trading day on weekends/holidays).

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from round_table_portfolio.data_tools.price_cache import (
    _ALPACA_BARS_URL,
    _ALPACA_429_BASE_DELAY_SECS,
    _ALPACA_429_MAX_RETRIES,
    _alpaca_auth_headers,
)

logger = logging.getLogger(__name__)

# Maximum weeks a ticker stays in the tracking window after first-proposed.
_TRACKING_WINDOW_WEEKS: int = 104

# Number of calendar days to look back when fetching the current price.
# We only need the most-recent bar; 10 days covers weekends + public holidays.
_SNAPSHOT_LOOKBACK_DAYS: int = 10

# Alpaca hard cap on symbols per batched request.
# The documented max is 1000; we use 200 to stay comfortably clear.
_ALPACA_BATCH_MAX_SYMBOLS: int = 200


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CaptureSummary:
    """Summary of one capture run — emitted to the log as a structured record."""

    week_id: str
    perimeter_count: int = 0
    newly_entered_count: int = 0
    still_tracking_count: int = 0
    success_count: int = 0
    miss_count: int = 0
    missed_tickers: list[str] = field(default_factory=list)

    def log(self) -> None:
        logger.info(
            "snapshot_capture summary | week=%s | perimeter=%d "
            "(new=%d still_tracking=%d) | Alpaca success=%d miss=%d%s",
            self.week_id,
            self.perimeter_count,
            self.newly_entered_count,
            self.still_tracking_count,
            self.success_count,
            self.miss_count,
            f" | missed_tickers={self.missed_tickers}" if self.missed_tickers else "",
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weeks_apart(week_a: str, week_b: str) -> int:
    """Return |weeks_a – week_b| in integer weeks.

    week_id format: 'YYYY-WNN' (ISO week label used throughout the project).
    Converts each to an approximate date using the ISO week Monday, then
    divides the day difference by 7.  Handles year-boundary weeks correctly
    via datetime.fromisocalendar.
    """
    def _to_date(week_id: str) -> datetime:
        parts = week_id.split("-W")
        year, week = int(parts[0]), int(parts[1])
        return datetime.fromisocalendar(year, week, 1)  # Monday of that ISO week

    d_a = _to_date(week_a)
    d_b = _to_date(week_b)
    return abs((d_a - d_b).days) // 7


def _compute_tracking_set(
    conn: sqlite3.Connection,
    week_id: str,
    shortlist_tickers: list[str],
    debate_set_tickers: list[str],
    user_id: str,
) -> tuple[set[str], set[str], set[str]]:
    """Return (tracking_set, new_entries, still_tracking_priors).

    tracking_set: unique tickers to snapshot this week.
    new_entries: tickers appearing for the first time this week.
    still_tracking_priors: tickers carried forward from prior weeks still
        within their 104-week window.
    """
    # This week's newly-surfaced perimeter (shortlist ∪ debate-set).
    this_week: set[str] = {t.upper().strip() for t in shortlist_tickers + debate_set_tickers if t}

    # Prior-first-proposed tickers still inside the 104-week window.
    # Derived from MIN(week_id) per ticker in existing shortlist_price_snapshots.
    first_proposed: dict[str, str] = {}
    rows = conn.execute(
        """
        SELECT ticker, MIN(week_id) AS first_week
        FROM shortlist_price_snapshots
        WHERE user_id = ?
        GROUP BY ticker
        """,
        (user_id,),
    ).fetchall()
    for ticker, first_week in rows:
        first_proposed[ticker.upper().strip()] = first_week

    still_tracking: set[str] = set()
    for ticker, first_week in first_proposed.items():
        if ticker in this_week:
            # Already in the new perimeter — will be counted as new or ongoing.
            continue
        weeks_elapsed = _weeks_apart(week_id, first_week)
        if weeks_elapsed <= _TRACKING_WINDOW_WEEKS:
            still_tracking.add(ticker)

    # New entries = this week's perimeter minus those with prior history.
    new_entries = this_week - set(first_proposed.keys())

    tracking_set = this_week | still_tracking
    return tracking_set, new_entries, still_tracking


def _fetch_current_prices_batched(
    tickers: list[str],
) -> dict[str, tuple[str, float]]:
    """Fetch the most-recent daily close for each ticker via one Alpaca batch call.

    Returns a dict {ticker: (snapshot_date, price)}.
    Tickers Alpaca cannot price are absent from the returned dict (logged as misses
    by the caller — never written as NULL-price rows).

    Uses _ALPACA_BARS_URL with multi-symbol batching (comma-joined symbols param).
    Falls back to exponential backoff on 429s up to _ALPACA_429_MAX_RETRIES.
    """
    if not tickers:
        return {}

    try:
        headers = _alpaca_auth_headers()
    except EnvironmentError as exc:
        logger.error("snapshot_capture: Alpaca auth missing — %s", exc)
        return {}

    start_date = (
        datetime.now(timezone.utc) - timedelta(days=_SNAPSHOT_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")

    results: dict[str, tuple[str, float]] = {}

    # Alpaca multi-symbol: pass a comma-joined list in the `symbols` param.
    # We batch in chunks of _ALPACA_BATCH_MAX_SYMBOLS to stay within API limits.
    for batch_start in range(0, len(tickers), _ALPACA_BATCH_MAX_SYMBOLS):
        batch = tickers[batch_start: batch_start + _ALPACA_BATCH_MAX_SYMBOLS]
        symbols_param = ",".join(batch)

        params: dict = {
            "symbols": symbols_param,
            "timeframe": "1Day",
            "start": start_date,
            "adjustment": "split",
            "limit": 1000,
        }

        all_bars: dict[str, list[dict]] = {}
        attempt = 0

        while True:
            try:
                resp = requests.get(
                    _ALPACA_BARS_URL,
                    params=params,
                    headers=headers,
                    timeout=30,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "snapshot_capture: Alpaca network error (batch %d–%d): %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                break  # skip this batch; tickers remain absent from results

            if resp.status_code == 429:
                if attempt < _ALPACA_429_MAX_RETRIES:
                    retry_after_hdr = resp.headers.get("Retry-After")
                    try:
                        delay = float(retry_after_hdr) if retry_after_hdr else (
                            _ALPACA_429_BASE_DELAY_SECS * (2 ** attempt)
                        )
                    except ValueError:
                        delay = _ALPACA_429_BASE_DELAY_SECS * (2 ** attempt)
                    logger.warning(
                        "snapshot_capture: Alpaca 429 (batch %d–%d) — retry %d/%d in %.0fs",
                        batch_start,
                        batch_start + len(batch),
                        attempt + 1,
                        _ALPACA_429_MAX_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                logger.warning(
                    "snapshot_capture: Alpaca 429 exhausted retries (batch %d–%d)",
                    batch_start,
                    batch_start + len(batch),
                )
                break

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                logger.warning(
                    "snapshot_capture: Alpaca HTTP error (batch %d–%d): %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                break

            try:
                payload = resp.json()
            except ValueError as exc:
                logger.warning(
                    "snapshot_capture: Alpaca JSON parse error (batch %d–%d): %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                break

            bars_by_symbol: dict = payload.get("bars") or {}
            for sym, bars in bars_by_symbol.items():
                if bars:
                    # Take the most-recent bar (last element, since Alpaca returns
                    # ascending order).
                    latest = bars[-1]
                    date_str = latest["t"][:10]   # "2026-06-06T00:00:00Z" → "2026-06-06"
                    close = latest.get("c")
                    if close is not None:
                        try:
                            price_val = float(close)
                            if price_val > 0:
                                all_bars[sym.upper()] = (date_str, price_val)
                        except (TypeError, ValueError):
                            pass  # absent from all_bars → counted as a miss

            next_token = payload.get("next_page_token")
            if not next_token:
                break  # all pages consumed for this batch

            params["page_token"] = next_token
            attempt = 0  # reset for next page

        results.update(all_bars)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def capture_shortlist_snapshots(
    conn: sqlite3.Connection,
    week_id: str,
    shortlist_tickers: list[str],
    debate_set_tickers: list[str],
    roster_version: int,
    user_id: str = "andrew",
    *,
    _price_fetcher: Optional[callable] = None,  # type: ignore[type-arg]
) -> CaptureSummary:
    """Capture a weekly price snapshot for every tracked shortlisted/debated ticker.

    Parameters
    ----------
    conn:
        Open SQLite connection with foreign_keys=ON (caller-managed; not closed here).
    week_id:
        ISO week label for the current run (e.g. "2026-W24").
    shortlist_tickers:
        All tickers any persona surfaced this week (from persona_shortlists).
        May contain duplicates — de-duped internally.
    debate_set_tickers:
        All tickers that entered the debate this week.
        May contain duplicates — de-duped internally.
    roster_version:
        Current roster version from weeks.roster_version (FK to roster_versions).
    user_id:
        Owner identity for the snapshot rows (default "andrew").
    _price_fetcher:
        Injectable price-fetch callable for testing.  Signature matches
        _fetch_current_prices_batched.  Defaults to the real Alpaca fetcher.

    Returns
    -------
    CaptureSummary
        Summary of the capture run (also emitted to the log).
    """
    if _price_fetcher is None:
        _price_fetcher = _fetch_current_prices_batched

    summary = CaptureSummary(week_id=week_id)

    # --- 1. Compute tracking set ---
    tracking_set, new_entries, still_tracking = _compute_tracking_set(
        conn, week_id, shortlist_tickers, debate_set_tickers, user_id
    )
    summary.perimeter_count = len(tracking_set)
    summary.newly_entered_count = len(new_entries)
    summary.still_tracking_count = len(still_tracking)

    if not tracking_set:
        logger.info(
            "snapshot_capture: no tickers to snapshot for week=%s (empty tracking set)",
            week_id,
        )
        summary.log()
        return summary

    # --- 2. Fetch current prices (one batched Alpaca call) ---
    tickers_sorted = sorted(tracking_set)  # deterministic order for batching
    price_map = _price_fetcher(tickers_sorted)

    # --- 3. Write one row per ticker; log misses, never write NULL-price rows ---
    rows_written = 0
    missed: list[str] = []

    for ticker in tickers_sorted:
        if ticker not in price_map:
            logger.warning(
                "snapshot_capture: Alpaca miss for ticker=%s week=%s — "
                "skipped (no NULL-price row written)",
                ticker,
                week_id,
            )
            missed.append(ticker)
            continue

        snapshot_date, price = price_map[ticker]
        try:
            conn.execute(
                """
                INSERT INTO shortlist_price_snapshots
                    (week_id, ticker, snapshot_date, price, roster_version, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (week_id, ticker, snapshot_date, price, roster_version, user_id),
            )
            rows_written += 1
        except sqlite3.IntegrityError as exc:
            # UNIQUE constraint fires on a duplicate (week_id, ticker, user_id).
            # This is a Major-tier caller bug — log loudly but don't crash the run.
            logger.error(
                "snapshot_capture: UNIQUE violation for (week=%s, ticker=%s, user=%s) — %s",
                week_id,
                ticker,
                user_id,
                exc,
            )

    conn.commit()

    summary.success_count = rows_written
    summary.miss_count = len(missed)
    summary.missed_tickers = missed
    summary.log()

    return summary
