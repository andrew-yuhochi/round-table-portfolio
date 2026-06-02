# technical_tools.py — Locally-computed technical indicators via pandas-ta-classic.
#
# Computes RSI, MACD, Bollinger Bands, ATR, ADX, OBV, EMA, SMA from OHLCV
# price candles fetched via get_prices().  No external API call — pure local
# computation.
#
# Per DATA-SOURCES.md (CRITICAL naming note):
#   Package: pandas-ta-classic (pip install pandas-ta-classic)
#   Import:  import pandas_ta_classic as ta   ← NOT pandas_ta
#
# Per TDD NULL threshold:
#   - Early bars (bar_index < indicator lookback) are legitimately None.
#   - 0% NULL allowed on bars where bar_index >= lookback.
#   - bars_available is recorded so callers can judge adequacy.

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pandas_ta_classic as ta
from pydantic import ValidationError

from round_table_portfolio.data_tools.models import TechnicalIndicators

logger = logging.getLogger(__name__)

# Minimum bars needed for SMA-200 (longest lookback used)
_MIN_BARS_FOR_FULL_INDICATORS = 200


def compute_technicals(
    ticker: str,
    *,
    prices_df: Optional[pd.DataFrame] = None,
    candle=None,  # FinnhubCandle — accept either form
) -> TechnicalIndicators:
    """Compute technical indicators for *ticker* from OHLCV price data.

    Accepts either a pre-fetched pandas DataFrame or a FinnhubCandle object.
    When neither is provided, fetches prices via get_prices() internally.

    Per TDD: indicators are computed locally (no rate limit); import failure
    at startup → abort (dependency issue, not runtime data issue).

    Args:
        ticker:     Uppercase ticker symbol.
        prices_df:  Optional OHLCV DataFrame with columns open/high/low/close/volume.
        candle:     Optional FinnhubCandle object (alternative to prices_df).

    Returns:
        TechnicalIndicators with most-recent bar values (None for early bars).

    Raises:
        RuntimeError: Price data unavailable and fetch failed.
        ImportError:  pandas_ta_classic not installed (deploy-time issue).
    """
    ticker = ticker.upper().strip()

    # --- Build the OHLCV DataFrame ---
    df = _build_ohlcv_df(ticker, prices_df, candle)

    if df is None or df.empty:
        logger.warning(
            "compute_technicals %s: no price data available — returning empty indicators",
            ticker,
        )
        return TechnicalIndicators(ticker=ticker, bars_available=0)

    bars = len(df)
    logger.debug("compute_technicals %s: %d bars", ticker, bars)

    if bars < 14:
        # Below minimum lookback for even RSI-14 — return empty
        logger.warning(
            "compute_technicals %s: only %d bars (need ≥14 for RSI) — returning empty",
            ticker, bars,
        )
        return TechnicalIndicators(ticker=ticker, bars_available=bars)

    # --- Compute indicators ---
    # pandas-ta-classic appends columns to df when append=True
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.obv(append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.sma(length=200, append=True)
    except Exception as exc:
        logger.error("pandas-ta-classic computation failed for %s: %s", ticker, exc)
        raise RuntimeError(f"Technical indicator computation failed for {ticker}: {exc}") from exc

    # --- Extract most-recent bar ---
    last = df.iloc[-1]

    def _get(col_patterns: list[str]) -> Optional[float]:
        """Find the first matching column and return the last row value."""
        for pat in col_patterns:
            # Try exact match first
            if pat in df.columns:
                val = last.get(pat)
                return None if (val is None or pd.isna(val)) else float(val)
            # Try case-insensitive prefix match
            matches = [c for c in df.columns if c.upper().startswith(pat.upper())]
            if matches:
                val = last.get(matches[0])
                return None if (val is None or pd.isna(val)) else float(val)
        return None

    return TechnicalIndicators(
        ticker=ticker,
        bars_available=bars,
        rsi_14=_get(["RSI_14"]),
        macd=_get(["MACD_12_26_9"]),
        macd_signal=_get(["MACDs_12_26_9"]),
        macd_hist=_get(["MACDh_12_26_9"]),
        bb_upper=_get(["BBU_20_2.0", "BBU_20"]),
        bb_middle=_get(["BBM_20_2.0", "BBM_20"]),
        bb_lower=_get(["BBL_20_2.0", "BBL_20"]),
        atr_14=_get(["ATRr_14", "ATR_14"]),
        adx_14=_get(["ADX_14"]),
        obv=_get(["OBV"]),
        ema_20=_get(["EMA_20"]),
        ema_50=_get(["EMA_50"]),
        sma_200=_get(["SMA_200"]),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ohlcv_df(
    ticker: str,
    prices_df: Optional[pd.DataFrame],
    candle,
) -> Optional[pd.DataFrame]:
    """Convert price input to a standard OHLCV DataFrame."""
    if prices_df is not None:
        return _normalise_df(prices_df)

    if candle is not None:
        # Convert FinnhubCandle to DataFrame
        if candle.s != "ok" or not candle.c:
            return None
        df = pd.DataFrame({
            "open": candle.o,
            "high": candle.h,
            "low": candle.l,
            "close": candle.c,
            "volume": candle.v,
        })
        return df

    # No price data provided — fetch via get_prices()
    try:
        from round_table_portfolio.data_tools.finnhub_tools import get_prices
        candle = get_prices(ticker)
        if candle.s == "ok" and candle.c:
            df = pd.DataFrame({
                "open": candle.o,
                "high": candle.h,
                "low": candle.l,
                "close": candle.c,
                "volume": candle.v,
            })
            return df
    except Exception as exc:
        logger.error("compute_technicals %s: get_prices failed: %s", ticker, exc)

    return None


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase open/high/low/close/volume."""
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume",
                     "adj close", "adj_close"):
            col_map[col] = lower.replace(" ", "_").replace("adj_close", "close")
    df = df.rename(columns=col_map)
    # Ensure required columns exist
    for req in ("open", "high", "low", "close", "volume"):
        if req not in df.columns:
            # Try uppercase
            upper = req.upper()
            if upper in df.columns:
                df = df.rename(columns={upper: req})
    return df
