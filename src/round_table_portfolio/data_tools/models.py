# models.py — Pydantic validation models for every data tool return type.
#
# Each tool returns one of these validated models (or raises).
# "Schema validation on every external response" — TDD Component 2 §Responsibility.
# No silent coercion: validators raise ValidationError on unexpected shapes.

from __future__ import annotations

from typing import Optional
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Finnhub models
# ---------------------------------------------------------------------------

class FinnhubCandle(BaseModel):
    """Validated Finnhub stock_candles response.

    Per TDD NULL threshold: if s=='ok', all arrays (c, h, l, o, t, v) must be
    present — 0% NULL allowed.
    """

    symbol: str
    c: list[float]   # close prices
    h: list[float]   # highs
    l: list[float]   # lows
    o: list[float]   # opens
    t: list[int]     # unix timestamps
    v: list[float]   # volumes
    s: str           # status: 'ok' | 'no_data'
    source: str = "finnhub"  # 'finnhub' or 'yfinance' (fallback)

    @model_validator(mode="after")
    def validate_ok_arrays(self) -> "FinnhubCandle":
        if self.s == "ok":
            length = len(self.c)
            for field_name in ("h", "l", "o", "t", "v"):
                arr = getattr(self, field_name)
                if len(arr) != length:
                    raise ValueError(
                        f"FinnhubCandle: array length mismatch for '{field_name}' "
                        f"(expected {length}, got {len(arr)})"
                    )
        return self


class FinnhubNewsItem(BaseModel):
    """A single Finnhub company news item.

    NULL thresholds: headline/url/datetime → 0%; summary → ≤10%.
    """

    headline: str
    url: str
    datetime: int     # unix timestamp
    summary: Optional[str] = None
    source: Optional[str] = None

    @field_validator("headline", "url")
    @classmethod
    def must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("headline and url must be non-empty")
        return v

    @field_validator("datetime")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("datetime must be a positive unix timestamp")
        return v


class FinnhubNewsList(BaseModel):
    """Validated list of news items for a ticker."""

    symbol: str
    items: list[FinnhubNewsItem]


class FinnhubTranscript(BaseModel):
    """Validated Finnhub earnings call transcript.

    NULL threshold: transcript ≤30% (coverage gap per TDD); when missing the
    EDGAR 8-K Item 2.02 fallback is invoked and source is set accordingly.
    """

    symbol: str
    transcript: Optional[str] = None   # None triggers EDGAR fallback
    time: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[int] = None
    source: str = "finnhub"  # 'finnhub' | 'edgar_8k'


class FinnhubPeers(BaseModel):
    """Validated Finnhub /stock/peers response."""

    symbol: str
    peers: list[str]

    @field_validator("peers")
    @classmethod
    def peers_are_strings(cls, v: list[str]) -> list[str]:
        return [p.strip().upper() for p in v if p and p.strip()]


class FinnhubBasicFinancials(BaseModel):
    """Key valuation / quality fields from Finnhub company_basic_financials.

    Used by get_fundamentals() and pre_narrow().
    Fields may be None when Finnhub doesn't have them for a ticker; that is
    logged and counted in the per-field NULL audit.
    """

    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    roe: Optional[float] = None          # return on equity
    roa: Optional[float] = None          # return on assets
    revenue_ttm: Optional[float] = None  # trailing twelve months revenue
    eps_ttm: Optional[float] = None
    debt_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    fcf_yield: Optional[float] = None
    beta: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None


# ---------------------------------------------------------------------------
# EDGAR models
# ---------------------------------------------------------------------------

class EDGARFiling(BaseModel):
    """Metadata for a single SEC EDGAR filing."""

    ticker: str
    form: str             # '10-K', '10-Q', '8-K', etc.
    filing_date: str      # ISO date string
    accession_no: str
    cik: Optional[str] = None


class EDGARFilingSet(BaseModel):
    """Validated set of recent filings for a ticker, plus key financials."""

    ticker: str
    filings: list[EDGARFiling]
    # Financial statement DataFrames are held as dicts (JSON-serialisable);
    # balance_sheet / income_stmt / cash_flow may be None if parsing fails.
    balance_sheet_columns: Optional[list[str]] = None
    income_stmt_columns: Optional[list[str]] = None
    cash_flow_columns: Optional[list[str]] = None
    # Raw DataFrames stored out-of-band (not in the Pydantic model) because
    # pandas DataFrames are not JSON-serialisable. Accessed via .df_cache dict.
    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# FRED models
# ---------------------------------------------------------------------------

class FREDSeriesObservation(BaseModel):
    """A single FRED series observation.

    value may be None (NaN) for not-yet-released observations — valid data.
    NULL threshold: ≤20% across the series list per TDD.
    """

    series_id: str
    date: str             # ISO date string
    value: Optional[float] = None


class FREDSeries(BaseModel):
    """Validated FRED series result (most-recent N observations)."""

    series_id: str
    description: str
    observations: list[FREDSeriesObservation]
    latest_value: Optional[float] = None
    latest_date: Optional[str] = None


class FREDMacroSnapshot(BaseModel):
    """All configured FRED series for a weekly run."""

    week_id: str
    series: list[FREDSeries]


# ---------------------------------------------------------------------------
# RSS models
# ---------------------------------------------------------------------------

class RSSEntry(BaseModel):
    """A single RSS feed entry.

    NULL thresholds: title/link/published → 0%; summary → ≤20%.
    Malformed entries (missing required fields) are skipped with a warning.
    """

    title: str
    link: str
    published: str
    summary: Optional[str] = None
    feed_source: str

    @field_validator("title", "link", "published")
    @classmethod
    def must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title, link, published must be non-empty")
        return v


class RSSHeadlines(BaseModel):
    """Validated RSS headlines from all configured feeds."""

    entries: list[RSSEntry]
    feeds_attempted: int
    feeds_succeeded: int
    feeds_with_bozo: int  # feedparser bozo flag count (logged warning)


# ---------------------------------------------------------------------------
# Technical indicators model
# ---------------------------------------------------------------------------

class TechnicalIndicators(BaseModel):
    """Locally-computed technical indicators for a ticker.

    Per TDD NULL threshold: early bars (bar_index < lookback) are legitimately
    None; 0% NULL allowed on bars where bar_index ≥ lookback.
    """

    ticker: str
    # Most-recent bar values (None means insufficient history for lookback)
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr_14: Optional[float] = None
    adx_14: Optional[float] = None
    obv: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    sma_200: Optional[float] = None
    bars_available: int = 0   # number of OHLCV bars used


# ---------------------------------------------------------------------------
# Pre-narrow model
# ---------------------------------------------------------------------------

class PreNarrowEntry(BaseModel):
    """A single ticker entry in the pre_narrow() lightweight universe dataset."""

    symbol: str
    name: str
    sector: str
    price: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    revenue_ttm: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    # Momentum signal: % change over last 4 weeks (None if prices unavailable)
    momentum_4w: Optional[float] = None


class PreNarrowResult(BaseModel):
    """Cached result of pre_narrow() for a given week."""

    week_id: str
    entries: list[PreNarrowEntry]
    fetched_at: str         # ISO datetime
    cache_hit: bool = False
    tickers_attempted: int = 0
    tickers_succeeded: int = 0
