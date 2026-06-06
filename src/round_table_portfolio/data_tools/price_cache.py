# price_cache.py — Lazy weekly price cache backed by Alpaca Markets.
#
# Design: LAZY — the first request for a ticker in a given ISO week triggers
# ONE Alpaca /v2/stocks/bars fetch for that ticker, then caches the result to
# state/cache/prices/<week_id>/<ticker>.parquet.  Every subsequent request for
# that ticker that week (from any persona) is served from the parquet file —
# zero network calls.
#
# Source: Alpaca Markets data API (authenticated, SIP consolidated tape).
#   Endpoint: GET https://data.alpaca.markets/v2/stocks/bars
#   Auth:     APCA-API-KEY-ID / APCA-API-SECRET-KEY headers
#   Params:   symbols, timeframe=1Day, start (2y back), adjustment=split
#   Batching: multi-symbol supported but called per-ticker here to match the
#             lazy cache key structure (one parquet per ticker per week).
#   Pagination: next_page_token followed automatically.
#
# Adjustment choice: `adjustment=split` — corrects for stock splits so
# technical indicators (RSI, MACD, SMA-200) don't see false price jumps,
# without also adjusting for dividends (which would distort absolute price
# levels used by value personas).
#
# 429 handling: Alpaca returns 429 + Retry-After when you exceed 200/min.
# With authenticated requests and per-ticker lazy caching, you won't normally
# hit this; the retry honors it when it does occur.
#
# Cache key: ISO week (e.g. "2026-W23") + ticker.
# Cache location: state/cache/prices/<week_id>/<ticker>.parquet
#
# Fallback: Finnhub quote() remains in finnhub_tools.get_prices() as the final
# live-spot fallback (single bar).  price_cache.py only handles the Alpaca path.
#
# Migrated from Yahoo Finance v8 → Alpaca Markets 2026-06-03 (BUG-001/BLG-003).
# Yahoo's unofficial endpoint was IP-banned under bulk authenticated use.

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.environ.get("STATE_DIR", "state"))
_CACHE_DIR = _STATE_DIR / "cache" / "prices"

# No cold-fetch pacing needed: Alpaca is authenticated (200 req/min quota).
# The module constant is kept so the test fixture that zeroes it still compiles.
_COLD_FETCH_PACE_SECS: float = 0.0

# Alpaca Markets data API
_ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"

# 429 retry: honor Retry-After header; fall back to exponential backoff.
_ALPACA_429_MAX_RETRIES: int = 3
_ALPACA_429_BASE_DELAY_SECS: float = 2.0

# Look-back: ~2 years of daily bars (Alpaca free tier supports this)
_ALPACA_LOOKBACK_DAYS: int = 730

# Canonical fetch depth: every cold fetch stores at least this many calendar
# days of history so that SMA-200 (needs ≥200 trading days ≈ 280 calendar days)
# and other long-lookback indicators are always available in the cache.
# Short "days" requests are SLICED from this canonical entry — they never cap it.
_CANONICAL_DAYS: int = 365


def _alpaca_auth_headers() -> dict[str, str]:
    """Build Alpaca auth headers from environment. Raises if keys are missing."""
    key_id = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key_id or not secret:
        raise EnvironmentError(
            "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set. "
            "Add them to .env (see .env.example)."
        )
    return {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# Cache path helpers (per-ticker parquet files)
# ---------------------------------------------------------------------------

def _ticker_cache_path(week_id: str, ticker: str) -> Path:
    return _CACHE_DIR / week_id / f"{ticker}.parquet"


def _load_ticker_cache(week_id: str, ticker: str) -> Optional[pd.DataFrame]:
    """Return the cached OHLCV DataFrame for *ticker* in *week_id*, or None on miss."""
    path = _ticker_cache_path(week_id, ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        logger.info("price_cache: hit  %s/%s (%d rows)", week_id, ticker, len(df))
        return df
    except Exception as exc:
        logger.warning("price_cache: read failed %s/%s: %s — will re-fetch", week_id, ticker, exc)
        return None


def _save_ticker_cache(week_id: str, ticker: str, df: pd.DataFrame) -> None:
    """Write the per-ticker OHLCV DataFrame to parquet."""
    path = _ticker_cache_path(week_id, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("price_cache: written %s/%s (%d rows)", week_id, ticker, len(df))


# ---------------------------------------------------------------------------
# Alpaca fetch
# ---------------------------------------------------------------------------

def _fetch_alpaca(ticker: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV for *ticker* from the Alpaca Markets /v2/stocks/bars endpoint.

    Uses the SIP consolidated tape (default feed), 1Day timeframe, split
    adjustment, ~2-year lookback.  Follows pagination via next_page_token.

    Returns a DataFrame indexed by date (str "YYYY-MM-DD") with columns
    open/high/low/close/volume, or None on any hard failure.
    """
    try:
        headers = _alpaca_auth_headers()
    except EnvironmentError as exc:
        logger.error("price_cache: Alpaca auth missing — %s", exc)
        return None

    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    params: dict = {
        "symbols": ticker,
        "timeframe": "1Day",
        "start": start_date,
        "adjustment": "split",
        "limit": 1000,
    }

    rows: list[dict] = []
    attempt = 0

    while True:
        try:
            resp = requests.get(
                _ALPACA_BARS_URL,
                params=params,
                headers=headers,
                timeout=20,
            )
        except requests.RequestException as exc:
            logger.warning("price_cache: Alpaca network error for %s: %s", ticker, exc)
            return None

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
                    "price_cache: Alpaca 429 for %s — retry %d/%d in %.0fs",
                    ticker, attempt + 1, _ALPACA_429_MAX_RETRIES, delay,
                )
                time.sleep(delay)
                attempt += 1
                continue
            logger.warning(
                "price_cache: Alpaca 429 for %s — exhausted %d retries, giving up",
                ticker, _ALPACA_429_MAX_RETRIES,
            )
            return None

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            logger.warning("price_cache: Alpaca HTTP error for %s: %s", ticker, exc)
            return None

        try:
            payload = resp.json()
        except ValueError as exc:
            logger.warning("price_cache: Alpaca JSON parse error for %s: %s", ticker, exc)
            return None

        bars_by_symbol = payload.get("bars") or {}
        ticker_bars = bars_by_symbol.get(ticker, [])
        for bar in ticker_bars:
            date_str = bar["t"][:10]  # "2024-01-02T00:00:00Z" → "2024-01-02"
            rows.append({
                "date":   date_str,
                "open":   _safe_float(bar.get("o")),
                "high":   _safe_float(bar.get("h")),
                "low":    _safe_float(bar.get("l")),
                "close":  _safe_float(bar.get("c")),
                "volume": _safe_float(bar.get("v")),
            })

        next_token = payload.get("next_page_token")
        if not next_token:
            break  # all pages consumed

        # Pass the page token on the next request
        params["page_token"] = next_token
        attempt = 0  # reset retry counter for the next page

    if not rows:
        logger.warning("price_cache: Alpaca: 0 bars returned for %s", ticker)
        return None

    df = pd.DataFrame(rows).set_index("date")
    df.attrs["source"] = "alpaca"
    logger.info("price_cache: Alpaca OK — %s: %d rows", ticker, len(df))
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_to_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return the most-recent *days* calendar days from *df* (tail slice).

    The index is expected to be date strings "YYYY-MM-DD" in ascending order.
    We include all rows whose date falls within *days* calendar days before the
    last row's date.  If the DataFrame is already shorter than *days*, it is
    returned as-is (no padding).

    df.attrs are preserved on the returned slice.
    """
    if df.empty or days <= 0:
        return df
    try:
        from datetime import datetime, timedelta
        last_date = datetime.strptime(str(df.index[-1]), "%Y-%m-%d")
        cutoff = (last_date - timedelta(days=days)).strftime("%Y-%m-%d")
        sliced = df[df.index >= cutoff]
    except Exception:
        # Index format unexpected — return full frame; indicator will see all bars
        sliced = df
    # Preserve attrs (source tag etc.)
    sliced.attrs = df.attrs
    return sliced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cached_prices(
    ticker: str,
    *,
    week_id: str,
    days: int = 365,
    all_tickers: Optional[list[str]] = None,  # kept for API compat; not used in lazy path
    _pace: bool = True,
) -> Optional[pd.DataFrame]:
    """Return a per-ticker OHLCV DataFrame from the lazy weekly cache.

    Cache always stores the canonical depth (_CANONICAL_DAYS, or the requested
    window if larger) so that long-lookback indicators (SMA-200) are always
    available.  A short-window request is sliced from the canonical entry —
    it never causes the cache to store fewer bars than a later long request needs.

    On cache miss: fetches *ticker* at max(days, _CANONICAL_DAYS) calendar days,
    writes the full canonical parquet, then slices the return value to *days*.

    On cache hit: reads the full canonical parquet and slices to *days*.

    The Finnhub-quote single-bar fallback is NOT stored in this cache; callers
    in finnhub_tools return it directly to avoid poisoning history reads.

    Args:
        ticker:      Ticker to retrieve.
        week_id:     ISO week key (e.g. "2026-W23") — determines cache freshness.
        days:        Requested look-back window in calendar days.  The cache may
                     hold MORE bars (up to _CANONICAL_DAYS); the caller gets at
                     most *days* worth sliced from the tail.
        all_tickers: Accepted for API compatibility with old callers; not used.
        _pace:       Accepted for API compatibility; Alpaca needs no pacing.

    Returns:
        DataFrame indexed by date (str) with columns open/high/low/close/volume,
        sliced to at most *days* calendar days from the tail of the canonical
        series, or None if the Alpaca fetch fails.
    """
    ticker = ticker.upper().strip()

    # --- Cache hit path ---
    cached = _load_ticker_cache(week_id, ticker)
    if cached is not None:
        return _slice_to_days(cached, days)

    # --- Cache miss: fetch at canonical depth to future-proof long-lookback callers ---
    fetch_days = max(days, _CANONICAL_DAYS)
    logger.info(
        "price_cache: miss  %s/%s — Alpaca fetch (fetch_days=%d, requested=%d)",
        week_id, ticker, fetch_days, days,
    )

    df = _fetch_alpaca(ticker, fetch_days)
    if df is None:
        logger.error("price_cache: Alpaca failed for %s/%s", week_id, ticker)
        return None

    # Store the full canonical fetch; slice for this caller.
    _save_ticker_cache(week_id, ticker, df)
    return _slice_to_days(df, days)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None
