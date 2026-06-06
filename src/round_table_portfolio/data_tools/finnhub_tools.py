# finnhub_tools.py — Finnhub data tools: prices, news, transcripts, peers, fundamentals.
#
# Free-tier Finnhub endpoints (confirmed 2026-06-01):
#   WORKS:   quote(), company_news(), company_basic_financials(), company_peers()
#   PREMIUM: stock_candles(), transcripts(), transcripts_list()
#
# Price source (updated 2026-06-03 — migrated Yahoo → Alpaca, BUG-001/BLG-003):
#   get_prices():  Served from the LAZY WEEKLY PRICE CACHE (price_cache.py).
#                  Cache miss → one Alpaca /v2/stocks/bars fetch per ticker → cached.
#                  Subsequent calls that week are served from parquet — zero network.
#                  Final fallback: Finnhub quote() for a live single-bar snapshot.
#   get_transcript(): Routes entirely to EDGAR 8-K fallback (Finnhub transcripts = premium).
#
# TDD Component 2, DATA-SOURCES.md §Finnhub.

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import finnhub
from dotenv import load_dotenv
from pydantic import ValidationError

from round_table_portfolio.data_tools.models import (
    FinnhubCandle,
    FinnhubBasicFinancials,
    FinnhubNewsList,
    FinnhubNewsItem,
    FinnhubTranscript,
    FinnhubPeers,
)
from round_table_portfolio.data_tools.rate_limiter import FINNHUB_LIMITER
from round_table_portfolio.data_tools.price_cache import get_cached_prices

load_dotenv()

logger = logging.getLogger(__name__)


def _get_client() -> finnhub.Client:
    """Build a Finnhub client from the environment. Raises if key is missing."""
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FINNHUB_API_KEY not set. Add it to .env (see .env.example)."
        )
    return finnhub.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------

def get_prices(
    ticker: str,
    *,
    days: int = 365,
    week_id: Optional[str] = None,
    all_tickers: Optional[list[str]] = None,
) -> FinnhubCandle:
    """Fetch daily OHLCV candles for *ticker* from the lazy weekly price cache.

    Primary path: lazy weekly cache (price_cache.py).  On cache miss, ONE Alpaca
    /v2/stocks/bars fetch is made for *ticker* (split-adjusted), then cached to
    parquet.  All subsequent calls for that ticker that week read from the
    parquet — no network.

    Final fallback: Finnhub quote() for a live single-bar spot price when the
    Alpaca fetch also fails.

    Args:
        ticker:      Uppercase ticker symbol.
        days:        Look-back window in calendar days (default 365).
        week_id:     ISO week string (e.g. "2026-W23").  Defaults to current week.
        all_tickers: Accepted for API compatibility; not used in the lazy path.

    Returns:
        FinnhubCandle with source='alpaca' or 'finnhub_quote'.

    Raises:
        RuntimeError: All sources failed.
    """
    ticker = ticker.upper().strip()

    if week_id is None:
        week_id = datetime.now(timezone.utc).strftime("%G-W%V")

    # --- Primary: lazy weekly price cache (raw Yahoo v8 on miss) ---
    cache_df = get_cached_prices(ticker, week_id=week_id, days=days, all_tickers=all_tickers)

    if cache_df is not None and not cache_df.empty:
        # Source tag is stored in df.attrs by price_cache
        source_tag = cache_df.attrs.get("source", "raw_yahoo")

        # Build timestamp list from date strings (convert to unix epoch)
        timestamps: list[int] = []
        for date_str in cache_df.index:
            try:
                ts = int(datetime.strptime(str(date_str), "%Y-%m-%d")
                         .replace(tzinfo=timezone.utc).timestamp())
                timestamps.append(ts)
            except Exception:
                timestamps.append(0)

        closes = [v for v in cache_df["close"].tolist()]
        highs  = [v if v is not None else closes[i] for i, v in enumerate(cache_df["high"].tolist())]
        lows   = [v if v is not None else closes[i] for i, v in enumerate(cache_df["low"].tolist())]
        opens  = [v if v is not None else closes[i] for i, v in enumerate(cache_df["open"].tolist())]
        vols   = [v if v is not None else 0.0 for v in cache_df["volume"].tolist()]

        # Drop rows where close is None
        rows = [
            (t, c, h, l, o, v)
            for t, c, h, l, o, v in zip(timestamps, closes, highs, lows, opens, vols)
            if c is not None
        ]
        if rows:
            ts_l, c_l, h_l, l_l, o_l, v_l = zip(*rows)
            candle = FinnhubCandle(
                symbol=ticker,
                c=list(c_l), h=list(h_l), l=list(l_l),
                o=list(o_l), t=list(ts_l), v=list(v_l),
                s="ok", source=source_tag,
            )
            logger.debug(
                "get_prices %s: %d bars from cache (source=%s, week=%s)",
                ticker, len(rows), source_tag, week_id,
            )
            return candle

    cache_error = f"cache returned no valid rows for {ticker}"
    logger.warning("get_prices %s: %s", ticker, cache_error)

    # --- Final fallback: Finnhub quote() — live spot price, single bar ---
    logger.info("get_prices %s: falling back to Finnhub quote()", ticker)
    try:
        FINNHUB_LIMITER.acquire()
        client = _get_client()
        q = client.quote(ticker)
        if q and q.get("c") and q.get("t"):
            date_str = datetime.fromtimestamp(int(q["t"]), tz=timezone.utc).strftime("%Y-%m-%d")
            import pandas as pd
            quote_df = pd.DataFrame([{
                "open":   float(q.get("o", q["c"])),
                "high":   float(q.get("h", q["c"])),
                "low":    float(q.get("l", q["c"])),
                "close":  float(q["c"]),
                "volume": 0.0,
            }], index=pd.Index([date_str], name="date"))
            quote_df.attrs["source"] = "finnhub_quote"

            # NOT written to the history cache — a single-bar spot quote must not
            # poison the per-ticker parquet that later callers expect to hold full
            # multi-bar Alpaca history (needed for SMA-200, MACD, etc.).
            # A later get_cached_prices call will see a cache miss and re-attempt Alpaca.

            candle = FinnhubCandle(
                symbol=ticker,
                c=[float(q["c"])],
                h=[float(q.get("h", q["c"]))],
                l=[float(q.get("l", q["c"]))],
                o=[float(q.get("o", q["c"]))],
                t=[int(q["t"])],
                v=[0.0],
                s="ok",
                source="finnhub_quote",
            )
            logger.info("Finnhub quote() fallback succeeded for %s (not cached — spot only)", ticker)
            return candle
        raise RuntimeError(f"Finnhub quote() returned no data for {ticker}: {q}")

    except Exception as fh_exc:
        raise RuntimeError(
            f"All price sources failed for {ticker}. "
            f"Cache error: {cache_error}. Finnhub error: {fh_exc}"
        ) from fh_exc


# ---------------------------------------------------------------------------
# get_fundamentals
# ---------------------------------------------------------------------------

def get_fundamentals(ticker: str) -> FinnhubBasicFinancials:
    """Fetch valuation / quality metrics from Finnhub company_basic_financials.

    Args:
        ticker: Uppercase ticker symbol.

    Returns:
        FinnhubBasicFinancials (fields may be None when Finnhub lacks data).

    Raises:
        RuntimeError: Finnhub call failed.
    """
    ticker = ticker.upper().strip()
    try:
        FINNHUB_LIMITER.acquire()
        client = _get_client()
        raw = client.company_basic_financials(ticker, "all")
        logger.debug("Finnhub basic_financials %s: keys=%s", ticker, list(raw.keys()))
    except Exception as exc:
        raise RuntimeError(f"Finnhub get_fundamentals failed for {ticker}: {exc}") from exc

    metric = raw.get("metric", {}) or {}

    return FinnhubBasicFinancials(
        symbol=ticker,
        pe_ratio=_float_or_none(metric.get("peNormalizedAnnual") or metric.get("peBasicExclExtraTTM")),
        pb_ratio=_float_or_none(metric.get("pbAnnual")),
        ps_ratio=_float_or_none(metric.get("psTTM")),
        ev_ebitda=_float_or_none(metric.get("evEbitdaTTM")),
        roe=_float_or_none(metric.get("roeTTM")),
        roa=_float_or_none(metric.get("roaTTM")),
        revenue_ttm=_float_or_none(metric.get("revenueTTM")),
        eps_ttm=_float_or_none(metric.get("epsTTM")),
        debt_equity=_float_or_none(metric.get("totalDebt/totalEquityAnnual")),
        current_ratio=_float_or_none(metric.get("currentRatioAnnual")),
        gross_margin=_float_or_none(metric.get("grossMarginTTM")),
        fcf_yield=_float_or_none(metric.get("fcfYieldTTM")),
        beta=_float_or_none(metric.get("beta")),
        week_52_high=_float_or_none(metric.get("52WeekHigh")),
        week_52_low=_float_or_none(metric.get("52WeekLow")),
    )


# ---------------------------------------------------------------------------
# get_company_news
# ---------------------------------------------------------------------------

def get_company_news(ticker: str, *, days: int = 30) -> FinnhubNewsList:
    """Fetch recent company news headlines from Finnhub.

    NULL thresholds: headline/url/datetime → 0%; summary → ≤10%.
    Malformed items are rejected and logged (never silently passed).

    Args:
        ticker: Uppercase ticker symbol.
        days:   Look-back window in days (≤30 per DATA-SOURCES.md integration note).

    Returns:
        FinnhubNewsList (may be empty if no news in the window).

    Raises:
        RuntimeError: API call failed.
    """
    ticker = ticker.upper().strip()
    date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_from = (datetime.now(timezone.utc) - timedelta(days=min(days, 30))).strftime("%Y-%m-%d")

    try:
        FINNHUB_LIMITER.acquire()
        client = _get_client()
        raw_items = client.company_news(ticker, _from=date_from, to=date_to)
        logger.debug("Finnhub company_news %s: %d raw items", ticker, len(raw_items))
    except Exception as exc:
        raise RuntimeError(f"Finnhub get_company_news failed for {ticker}: {exc}") from exc

    items: list[FinnhubNewsItem] = []
    rejected = 0
    for raw in raw_items:
        try:
            item = FinnhubNewsItem(
                headline=raw.get("headline", "") or "",
                url=raw.get("url", "") or "",
                datetime=int(raw.get("datetime", 0) or 0),
                summary=raw.get("summary") or None,
                source=raw.get("source") or None,
            )
            items.append(item)
        except (ValidationError, ValueError) as exc:
            rejected += 1
            logger.warning(
                "Finnhub news item rejected for %s (validation): %s — raw=%s",
                ticker, exc, {k: raw.get(k) for k in ("headline", "url", "datetime")},
            )

    if rejected:
        logger.warning(
            "Finnhub company_news %s: %d/%d items rejected by schema validation",
            ticker, rejected, len(raw_items),
        )

    return FinnhubNewsList(symbol=ticker, items=items)


# ---------------------------------------------------------------------------
# get_earnings_transcript
# ---------------------------------------------------------------------------

def get_earnings_transcript(ticker: str) -> FinnhubTranscript:
    """Fetch the most recent earnings transcript via EDGAR 8-K fallback.

    Finnhub transcript endpoints (transcripts() / transcripts_list()) require
    a premium subscription (403 on free tier, confirmed 2026-06-01).
    This function routes entirely to the EDGAR 8-K Item 2.02 fallback.

    NULL threshold: transcript ≤30% across the portfolio (some tickers may not
    have a recent 8-K with earnings text — this is expected coverage gap, not
    a code bug).

    Args:
        ticker: Uppercase ticker symbol.

    Returns:
        FinnhubTranscript with source='edgar_8k'. transcript may be None on
        coverage gap (counted toward the ≤30% NULL threshold).
    """
    ticker = ticker.upper().strip()
    logger.info(
        "get_earnings_transcript %s: routing to EDGAR 8-K "
        "(Finnhub transcripts = premium endpoint, not available on free tier)",
        ticker,
    )
    try:
        from round_table_portfolio.data_tools.edgar_tools import _get_8k_item_202_text
        text = _get_8k_item_202_text(ticker)
        if text:
            return FinnhubTranscript(symbol=ticker, transcript=text, source="edgar_8k")
        logger.info("EDGAR 8-K: no earnings text found for %s (coverage gap)", ticker)
        return FinnhubTranscript(symbol=ticker, transcript=None, source="edgar_8k")
    except Exception as exc:
        logger.error("EDGAR 8-K fallback failed for %s: %s", ticker, exc)
        return FinnhubTranscript(symbol=ticker, transcript=None, source="edgar_8k")


# ---------------------------------------------------------------------------
# get_peers
# ---------------------------------------------------------------------------

def get_peers(ticker: str) -> FinnhubPeers:
    """Fetch peer companies from Finnhub /stock/peers.

    Args:
        ticker: Uppercase ticker symbol.

    Returns:
        FinnhubPeers (validated list of peer tickers, may be empty).

    Raises:
        RuntimeError: API call failed.
    """
    ticker = ticker.upper().strip()
    try:
        FINNHUB_LIMITER.acquire()
        client = _get_client()
        raw = client.company_peers(ticker)
        logger.debug("Finnhub peers %s: %s", ticker, raw)
    except Exception as exc:
        raise RuntimeError(f"Finnhub get_peers failed for {ticker}: {exc}") from exc

    peers_raw = raw if isinstance(raw, list) else []
    return FinnhubPeers(
        symbol=ticker,
        peers=[p for p in peers_raw if p and p.upper() != ticker],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_or_none(val: object) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
