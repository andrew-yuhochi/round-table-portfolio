# finnhub_tools.py — Finnhub data tools: prices, news, transcripts, peers, fundamentals.
#
# Free-tier Finnhub endpoints (confirmed 2026-06-01):
#   WORKS:   quote(), company_news(), company_basic_financials(), company_peers()
#   PREMIUM: stock_candles(), transcripts(), transcripts_list()
#
# Adaptations for free tier:
#   get_prices():       Uses Yahoo Finance v8 API directly via requests + certifi
#                       (yfinance library has rate-limit handling issues; bypass it).
#                       Falls back to Finnhub quote() for a current-price-only snapshot.
#   get_transcript():   Routes entirely to EDGAR 8-K fallback (Finnhub transcripts = premium).
#
# TDD Component 2, DATA-SOURCES.md §Finnhub.

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import certifi
import finnhub
import requests
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
from round_table_portfolio.data_tools.rate_limiter import FINNHUB_LIMITER, YFINANCE_LIMITER

load_dotenv()

logger = logging.getLogger(__name__)

_YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


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
) -> FinnhubCandle:
    """Fetch daily OHLCV candles for *ticker*.

    Primary: Yahoo Finance v8 API via requests + certifi (bypasses yfinance's
    broken rate-limit handling).  Falls back to a Finnhub quote()-based
    single-bar snapshot when Yahoo is unavailable.

    Args:
        ticker: Uppercase ticker symbol.
        days:   Look-back window in calendar days (default 365 for technicals).

    Returns:
        FinnhubCandle with source='yfinance' (Yahoo) or 'finnhub_quote' (fallback).

    Raises:
        RuntimeError: Both Yahoo and Finnhub quote failed (per PRD NFR #5).
    """
    ticker = ticker.upper().strip()
    range_str = f"{days}d" if days <= 730 else "2y"

    # --- Primary: Yahoo Finance v8 API ---
    yf_error: Optional[str] = None
    try:
        YFINANCE_LIMITER.acquire()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        resp = requests.get(
            url,
            params={"interval": "1d", "range": range_str},
            headers=_YF_HEADERS,
            verify=certifi.where(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if result:
            chart = result[0]
            timestamps = chart.get("timestamp", [])
            indicators = chart.get("indicators", {})
            quote = indicators.get("quote", [{}])[0]
            closes = [float(v) if v is not None else None for v in quote.get("close", [])]
            highs  = [float(v) if v is not None else None for v in quote.get("high", [])]
            lows   = [float(v) if v is not None else None for v in quote.get("low", [])]
            opens  = [float(v) if v is not None else None for v in quote.get("open", [])]
            vols   = [float(v) if v is not None else None for v in quote.get("volume", [])]

            # Drop rows where close is None (market-closed / pre-market rows)
            rows = [
                (t, c, h, l, o, v)
                for t, c, h, l, o, v in zip(timestamps, closes, highs, lows, opens, vols)
                if c is not None
            ]
            if rows:
                ts, c, h, l, o, v = zip(*rows)
                candle = FinnhubCandle(
                    symbol=ticker,
                    c=list(c), h=list(h), l=list(l),
                    o=list(o), t=list(ts), v=list(v),
                    s="ok", source="yfinance",
                )
                logger.debug("Yahoo Finance %s: %d bars", ticker, len(rows))
                return candle

        yf_error = f"Yahoo Finance returned empty chart for {ticker}"
        logger.warning("%s", yf_error)

    except requests.RequestException as exc:
        yf_error = str(exc)
        logger.warning("Yahoo Finance get_prices failed for %s: %s", ticker, exc)
    except Exception as exc:
        yf_error = str(exc)
        logger.warning("Yahoo Finance parse failed for %s: %s", ticker, exc)

    # --- Fallback: Finnhub quote() — single-bar snapshot ---
    logger.info("get_prices %s: falling back to Finnhub quote() (Yahoo: %s)", ticker, yf_error)
    try:
        FINNHUB_LIMITER.acquire()
        client = _get_client()
        q = client.quote(ticker)
        # quote() returns: c (current), h (high), l (low), o (open), pc (prev close), t (ts)
        if q and q.get("c") and q.get("t"):
            candle = FinnhubCandle(
                symbol=ticker,
                c=[float(q["c"])],
                h=[float(q.get("h", q["c"]))],
                l=[float(q.get("l", q["c"]))],
                o=[float(q.get("o", q["c"]))],
                t=[int(q["t"])],
                v=[0.0],  # quote() doesn't provide volume
                s="ok",
                source="finnhub_quote",
            )
            logger.info("Finnhub quote() fallback succeeded for %s", ticker)
            return candle
        raise RuntimeError(f"Finnhub quote() returned no data for {ticker}: {q}")

    except Exception as fh_exc:
        raise RuntimeError(
            f"Both Yahoo Finance and Finnhub quote() failed for {ticker}. "
            f"Yahoo error: {yf_error}. Finnhub error: {fh_exc}"
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
