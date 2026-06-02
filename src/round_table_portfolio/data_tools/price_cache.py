# price_cache.py — Lazy weekly price cache.
#
# Design: LAZY — the first request for a ticker in a given ISO week triggers
# ONE raw Yahoo v8 fetch for that ticker, then caches the result to
# state/cache/prices/<week_id>/<ticker>.parquet.  Every subsequent request for
# that ticker that week (from any persona) is served from the parquet file —
# zero network calls.
#
# Cold-path pacing: a short sleep between successive cold fetches keeps the
# cold path itself under Yahoo's per-IP rate limit.  The pacing is applied
# ONLY on cache misses (warm reads have no sleep).
#
# Cache key: ISO week (e.g. "2026-W23") + ticker.
# Cache location: state/cache/prices/<week_id>/<ticker>.parquet
#
# Fallback: Finnhub quote() remains in finnhub_tools.get_prices() as the final
# live-spot fallback (single bar).  price_cache.py only handles the raw-Yahoo
# path — it does NOT call Finnhub.
#
# Dead paths removed (per founder direction 2026-06-02):
#   - yfinance.download() batch (got 429d worse than raw)
#   - Stooq via pandas-datareader (broken on Python 3.14)
#
# Per TDD Component 2 / founder Option A directive 2026-06-02.

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import certifi
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.environ.get("STATE_DIR", "state"))
_CACHE_DIR = _STATE_DIR / "cache" / "prices"

# Seconds to sleep between cold per-ticker fetches — keeps cold path under Yahoo limit.
_COLD_FETCH_PACE_SECS: float = 0.5

# Yahoo Finance v8 chart endpoint — confirmed working 2026-06-02.
_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
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
# Raw Yahoo v8 fetch (the one path that actually works — confirmed 200)
# ---------------------------------------------------------------------------

def _fetch_raw_yahoo(ticker: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV for *ticker* from the Yahoo Finance v8 chart endpoint.

    Returns a DataFrame indexed by date (str "YYYY-MM-DD") with columns
    open/high/low/close/volume, or None on any failure.

    Confirmed working 2026-06-02 on Python 3.14 / macOS with certifi bundle.
    """
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    url = _YAHOO_CHART_URL.format(ticker=ticker)
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    try:
        resp = requests.get(
            url,
            params=params,
            headers=_YAHOO_HEADERS,
            verify=certifi.where(),
            timeout=15,
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning("price_cache: Yahoo HTTP error for %s: %s", ticker, exc)
        return None
    except requests.RequestException as exc:
        logger.warning("price_cache: Yahoo network error for %s: %s", ticker, exc)
        return None

    try:
        payload = resp.json()
        result = payload["chart"]["result"]
        if not result:
            logger.warning("price_cache: Yahoo returned empty result for %s", ticker)
            return None

        chart = result[0]
        timestamps = chart.get("timestamp", [])
        indicators = chart.get("indicators", {})
        quote = indicators.get("quote", [{}])[0]
        adjclose_list = indicators.get("adjclose", [{}])
        adjclose = adjclose_list[0].get("adjclose", []) if adjclose_list else []

        if not timestamps:
            logger.warning("price_cache: Yahoo: no timestamps for %s", ticker)
            return None

        rows: list[dict] = []
        closes_raw = quote.get("close", [])
        # Prefer adjclose when available
        close_src = adjclose if len(adjclose) == len(timestamps) else closes_raw

        for i, ts in enumerate(timestamps):
            close_val = _safe_float(_nth(close_src, i))
            if close_val is None:
                continue
            rows.append({
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                "open":   _safe_float(_nth(quote.get("open", []), i)),
                "high":   _safe_float(_nth(quote.get("high", []), i)),
                "low":    _safe_float(_nth(quote.get("low", []), i)),
                "close":  close_val,
                "volume": _safe_float(_nth(quote.get("volume", []), i)),
            })

        if not rows:
            logger.warning("price_cache: Yahoo: 0 valid rows for %s", ticker)
            return None

        df = pd.DataFrame(rows).set_index("date")
        df.attrs["source"] = "raw_yahoo"
        logger.info("price_cache: Yahoo OK — %s: %d rows", ticker, len(df))
        return df

    except (KeyError, IndexError, ValueError, TypeError) as exc:
        logger.warning("price_cache: Yahoo parse error for %s: %s", ticker, exc)
        return None


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

    On cache miss: fetches *ticker* via one raw Yahoo v8 request, writes the
    parquet, and returns the result.  A short sleep (_COLD_FETCH_PACE_SECS) is
    applied before each cold fetch to stay under Yahoo's rate limit.

    On cache hit: returns the cached parquet immediately — no network call.

    Args:
        ticker:      Ticker to retrieve.
        week_id:     ISO week key (e.g. "2026-W23") — determines cache freshness.
        days:        Look-back window in calendar days (used on cold fetch only).
        all_tickers: Accepted for API compatibility with old callers; not used.
                     The lazy design fetches each ticker individually on first miss.
        _pace:       If True (default), sleep before cold fetch.  Set False in tests.

    Returns:
        DataFrame indexed by date (str) with columns open/high/low/close/volume,
        or None if Yahoo fetch fails.
    """
    ticker = ticker.upper().strip()

    # --- Cache hit path ---
    cached = _load_ticker_cache(week_id, ticker)
    if cached is not None:
        return cached

    # --- Cache miss: one raw Yahoo fetch for this ticker ---
    logger.info("price_cache: miss  %s/%s — raw Yahoo fetch", week_id, ticker)

    if _pace:
        time.sleep(_COLD_FETCH_PACE_SECS)

    df = _fetch_raw_yahoo(ticker, days)
    if df is None:
        logger.error("price_cache: raw Yahoo failed for %s/%s", week_id, ticker)
        return None

    _save_ticker_cache(week_id, ticker, df)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nth(lst: list, i: int):
    """Return lst[i] or None if out of range."""
    try:
        return lst[i]
    except (IndexError, TypeError):
        return None


def _safe_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None
