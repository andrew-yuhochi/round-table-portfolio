# prenarrow.py — Optional cheap deterministic pre_narrow() tool.
#
# Fetches a lightweight prices+fundamentals snapshot over the full ~500-name
# S&P 500 universe, ranks by a simple composite score, and caches the result
# once per week to state/prenarrow/<week_id>/prenarrow.parquet.
#
# Subsequent calls within the same week return from cache — no re-pull.
# This avoids 7 × ~500-name Finnhub calls per weekly run.
#
# Per TDD Component 2 §Data Stored + §Quality Criteria:
#   - Cache written to state/prenarrow/<week_id>/prenarrow.parquet.
#   - Cache reuse verified by parquet mtime + single fetch in tool-call manifest.
#   - Second persona calling pre_narrow() reads cache (not a fresh pull).

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from round_table_portfolio.data_tools.models import PreNarrowEntry, PreNarrowResult
from round_table_portfolio.data_tools.rate_limiter import FINNHUB_LIMITER
# Top-level imports so patch("...prenarrow.get_prices/get_fundamentals") works in tests.
from round_table_portfolio.data_tools.finnhub_tools import get_prices, get_fundamentals
from round_table_portfolio.config.universe import load_universe, _DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.environ.get("STATE_DIR", "state"))
_PRENARROW_DIR = _STATE_DIR / "prenarrow"


def _cache_path(week_id: str) -> Path:
    return _PRENARROW_DIR / week_id / "prenarrow.parquet"


def _load_cache(week_id: str) -> Optional[PreNarrowResult]:
    """Return cached result for the week, or None if not cached."""
    path = _cache_path(week_id)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        entries = []
        for _, row in df.iterrows():
            entries.append(
                PreNarrowEntry(
                    symbol=str(row.get("symbol", "")),
                    name=str(row.get("name", "")),
                    sector=str(row.get("sector", "")),
                    price=_float_or_none(row.get("price")),
                    pe_ratio=_float_or_none(row.get("pe_ratio")),
                    pb_ratio=_float_or_none(row.get("pb_ratio")),
                    roe=_float_or_none(row.get("roe")),
                    revenue_ttm=_float_or_none(row.get("revenue_ttm")),
                    week_52_high=_float_or_none(row.get("week_52_high")),
                    week_52_low=_float_or_none(row.get("week_52_low")),
                    momentum_4w=_float_or_none(row.get("momentum_4w")),
                )
            )
        fetched_at = str(df.attrs.get("fetched_at", "unknown"))
        tickers_attempted = int(df.attrs.get("tickers_attempted", len(df)))
        tickers_succeeded = int(df.attrs.get("tickers_succeeded", len(df)))
        logger.info(
            "pre_narrow: cache hit for %s (%d entries from %s)",
            week_id, len(entries), fetched_at,
        )
        return PreNarrowResult(
            week_id=week_id,
            entries=entries,
            fetched_at=fetched_at,
            cache_hit=True,
            tickers_attempted=tickers_attempted,
            tickers_succeeded=tickers_succeeded,
        )
    except Exception as exc:
        logger.warning("pre_narrow: cache read failed for %s: %s — will re-fetch", week_id, exc)
        return None


def _save_cache(week_id: str, result: PreNarrowResult) -> None:
    """Write the pre_narrow result to parquet cache."""
    path = _cache_path(week_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "symbol": e.symbol,
            "name": e.name,
            "sector": e.sector,
            "price": e.price,
            "pe_ratio": e.pe_ratio,
            "pb_ratio": e.pb_ratio,
            "roe": e.roe,
            "revenue_ttm": e.revenue_ttm,
            "week_52_high": e.week_52_high,
            "week_52_low": e.week_52_low,
            "momentum_4w": e.momentum_4w,
        }
        for e in result.entries
    ]
    df = pd.DataFrame(rows)
    df.attrs["fetched_at"] = result.fetched_at
    df.attrs["tickers_attempted"] = result.tickers_attempted
    df.attrs["tickers_succeeded"] = result.tickers_succeeded
    df.to_parquet(path, index=False)
    logger.info("pre_narrow: cache written to %s (%d entries)", path, len(rows))


def pre_narrow(
    week_id: str,
    *,
    config_path: Optional[Path] = None,
    max_tickers: Optional[int] = None,
) -> PreNarrowResult:
    """Fetch a lightweight prices+fundamentals snapshot over the S&P 500 universe.

    Cached once per week to state/prenarrow/<week_id>/prenarrow.parquet.
    Subsequent calls within the same week serve from cache.

    The result is ranked by a simple composite (lower P/E + higher ROE +
    positive momentum) so personas can use it as a starting shortlist.

    Args:
        week_id:     ISO week label (e.g. '2026-W23') — determines cache key.
        config_path: Override path to sp500_universe.yaml (for testing).
        max_tickers: Limit the number of tickers fetched (for testing).

    Returns:
        PreNarrowResult — ranked list with cache_hit=True if served from cache.

    Raises:
        RuntimeError: Finnhub calls failed for the majority of tickers.
    """
    # --- Cache check ---
    cached = _load_cache(week_id)
    if cached is not None:
        return cached

    logger.info("pre_narrow: no cache for %s — fetching from Finnhub", week_id)

    # --- Load universe ---
    universe = load_universe(config_path or _DEFAULT_CONFIG_PATH)
    if max_tickers:
        universe = universe[:max_tickers]

    # --- Fetch prices + fundamentals for each ticker ---
    entries: list[PreNarrowEntry] = []
    tickers_attempted = len(universe)
    tickers_succeeded = 0
    fetched_at = datetime.now(timezone.utc).isoformat()

    for ticker_entry in universe:
        ticker = ticker_entry.symbol
        try:
            # Prices (recent — 30 days for momentum_4w)
            candle = get_prices(ticker, days=30)
            price: Optional[float] = candle.c[-1] if candle.s == "ok" and candle.c else None
            momentum_4w: Optional[float] = None
            if candle.s == "ok" and len(candle.c) >= 2:
                try:
                    momentum_4w = (candle.c[-1] - candle.c[0]) / candle.c[0] * 100.0
                except ZeroDivisionError:
                    pass

            # Fundamentals
            try:
                FINNHUB_LIMITER.acquire()
                fund = get_fundamentals(ticker)
                pe = fund.pe_ratio
                pb = fund.pb_ratio
                roe = fund.roe
                rev = fund.revenue_ttm
                hi52 = fund.week_52_high
                lo52 = fund.week_52_low
            except Exception as fund_exc:
                logger.debug("pre_narrow fundamentals failed for %s: %s", ticker, fund_exc)
                pe = pb = roe = rev = hi52 = lo52 = None

            entries.append(PreNarrowEntry(
                symbol=ticker,
                name=ticker_entry.name,
                sector=ticker_entry.sector,
                price=price,
                pe_ratio=pe,
                pb_ratio=pb,
                roe=roe,
                revenue_ttm=rev,
                week_52_high=hi52,
                week_52_low=lo52,
                momentum_4w=momentum_4w,
            ))
            tickers_succeeded += 1

        except Exception as exc:
            logger.warning("pre_narrow: %s failed — skipping: %s", ticker, exc)
            # Add skeleton entry so the ticker is still in the universe
            entries.append(PreNarrowEntry(
                symbol=ticker,
                name=ticker_entry.name,
                sector=ticker_entry.sector,
            ))

    # --- Rank (simple composite: lower P/E better, higher ROE better, positive momentum) ---
    entries = _rank_entries(entries)

    result = PreNarrowResult(
        week_id=week_id,
        entries=entries,
        fetched_at=fetched_at,
        cache_hit=False,
        tickers_attempted=tickers_attempted,
        tickers_succeeded=tickers_succeeded,
    )

    _save_cache(week_id, result)
    return result


def _rank_entries(entries: list[PreNarrowEntry]) -> list[PreNarrowEntry]:
    """Rank entries by a simple composite score.

    Score = (1/PE if PE > 0 else 0) + (ROE/100 if ROE else 0) + (momentum_4w/100 if positive else 0)

    Higher score → better rank. Entries with all-None fundamentals go to the end.
    """
    def _score(e: PreNarrowEntry) -> float:
        score = 0.0
        if e.pe_ratio and e.pe_ratio > 0:
            score += 1.0 / e.pe_ratio
        if e.roe is not None:
            score += max(e.roe, 0.0) / 100.0
        if e.momentum_4w is not None and e.momentum_4w > 0:
            score += e.momentum_4w / 100.0
        return score

    return sorted(entries, key=_score, reverse=True)


def _float_or_none(val: object) -> Optional[float]:
    """Convert value to float, returning None on failure or NaN."""
    if val is None:
        return None
    import math
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
