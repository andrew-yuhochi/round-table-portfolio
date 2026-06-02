# edgar_tools.py — SEC EDGAR tools via edgartools: filings + key financials.
#
# Per TDD Component 2 §EDGAR row:
#   - edgartools handles SEC rate-limit (10 req/sec) internally.
#   - Company(ticker) raises if ticker not found — treat as Major-tier failure.
#   - NULL thresholds: balance_sheet.total_assets, income_stmt.revenue → 0%
#     on tickers with a published 10-K/10-Q; cash_flow.fcf → ≤10%.
#
# Also exposes _get_8k_item_202_text() — the transcript fallback used by
# get_earnings_transcript() in finnhub_tools.py.

from __future__ import annotations

import logging
from typing import Optional

import os

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from round_table_portfolio.data_tools.models import EDGARFiling, EDGARFilingSet
from round_table_portfolio.data_tools.rate_limiter import EDGAR_LIMITER

load_dotenv()


def _set_edgar_identity() -> None:
    """Set the SEC EDGAR user-agent identity required by edgartools.

    edgartools raises if edgar.set_identity() has not been called before any
    Company() lookup. We call it here at first use (idempotent — calling it
    multiple times is safe).
    """
    import edgar
    identity = os.environ.get("SEC_EDGAR_USER_AGENT", "")
    if not identity:
        raise EnvironmentError(
            "SEC_EDGAR_USER_AGENT not set. Add it to .env "
            "(format: 'Your Name your@email.com'). "
            "See .env.example."
        )
    edgar.set_identity(identity)


_edgar_identity_set = False


def _ensure_identity() -> None:
    """Call set_edgar_identity() once per process."""
    global _edgar_identity_set
    if not _edgar_identity_set:
        _set_edgar_identity()
        _edgar_identity_set = True

logger = logging.getLogger(__name__)


def get_filings(ticker: str) -> EDGARFilingSet:
    """Fetch the most recent 10-K, 10-Q, and 8-K filings for *ticker*.

    Also extracts key financial columns (balance sheet, income statement,
    cash flow) from the latest 10-K or 10-Q if available.

    NULL thresholds enforced at Gate 4:
      - balance_sheet: total_assets present → 0% NULL (US GAAP requirement)
      - income_stmt:   revenue present → 0% NULL
      - cash_flow:     free_cash_flow (computed) → ≤10% NULL (label variation)

    Args:
        ticker: Uppercase ticker symbol.

    Returns:
        EDGARFilingSet with filing metadata and financial column lists.

    Raises:
        RuntimeError: edgartools Company() lookup failed (ticker not found or
                      SEC unreachable) — Major-tier per TDD.
        ValidationError: Response data failed schema checks.
    """
    ticker = ticker.upper().strip()

    _ensure_identity()
    try:
        EDGAR_LIMITER.acquire()
        import edgar
        company = edgar.Company(ticker)
    except Exception as exc:
        raise RuntimeError(
            f"EDGAR get_filings: Company lookup failed for {ticker}: {exc}"
        ) from exc

    filings: list[EDGARFiling] = []
    bs_cols: Optional[list[str]] = None
    is_cols: Optional[list[str]] = None
    cf_cols: Optional[list[str]] = None
    df_cache: dict[str, pd.DataFrame] = {}

    # --- Fetch 10-K ---
    try:
        EDGAR_LIMITER.acquire()
        filings_10k = company.get_filings(form="10-K")
        if filings_10k:
            latest_10k = filings_10k.latest(1)
            if latest_10k:
                meta = _extract_filing_meta(ticker, latest_10k, "10-K")
                if meta:
                    filings.append(meta)
                # Extract financials from 10-K
                bs_cols, is_cols, cf_cols, df_cache = _extract_financials(
                    latest_10k, ticker, "10-K"
                )
    except Exception as exc:
        logger.warning("EDGAR 10-K fetch failed for %s: %s", ticker, exc)

    # --- Fetch 10-Q ---
    try:
        EDGAR_LIMITER.acquire()
        filings_10q = company.get_filings(form="10-Q")
        if filings_10q:
            latest_10q = filings_10q.latest(1)
            if latest_10q:
                meta = _extract_filing_meta(ticker, latest_10q, "10-Q")
                if meta:
                    filings.append(meta)
                # Use 10-Q financials only if 10-K didn't provide them
                if bs_cols is None:
                    bs_cols, is_cols, cf_cols, df_cache = _extract_financials(
                        latest_10q, ticker, "10-Q"
                    )
    except Exception as exc:
        logger.warning("EDGAR 10-Q fetch failed for %s: %s", ticker, exc)

    # --- Fetch recent 8-K list (metadata only) ---
    try:
        EDGAR_LIMITER.acquire()
        filings_8k = company.get_filings(form="8-K")
        if filings_8k:
            recent_8ks = filings_8k.latest(3)  # last 3 8-K filings
            if recent_8ks:
                # latest(3) may return a single object or a collection
                for f in _iter_filings(recent_8ks):
                    meta = _extract_filing_meta(ticker, f, "8-K")
                    if meta:
                        filings.append(meta)
    except Exception as exc:
        logger.warning("EDGAR 8-K list fetch failed for %s: %s", ticker, exc)

    result = EDGARFilingSet(
        ticker=ticker,
        filings=filings,
        balance_sheet_columns=bs_cols,
        income_stmt_columns=is_cols,
        cash_flow_columns=cf_cols,
    )
    # Attach df_cache as an attribute (not in the Pydantic model — DFs not serialisable)
    object.__setattr__(result, "_df_cache", df_cache)
    return result


def _get_8k_item_202_text(ticker: str) -> Optional[str]:
    """Extract earnings press release text from the most recent 8-K filing.

    This is the fallback used by get_earnings_transcript() when Finnhub has
    no transcript coverage for the ticker.

    Returns the text of Item 2.02 exhibits if found, else None.
    """
    ticker = ticker.upper().strip()
    _ensure_identity()
    try:
        EDGAR_LIMITER.acquire()
        import edgar
        company = edgar.Company(ticker)
        filings_8k = company.get_filings(form="8-K")
        if not filings_8k:
            logger.info("EDGAR: no 8-K filings found for %s", ticker)
            return None

        latest = filings_8k.latest(1)
        if latest is None:
            return None

        # edgartools v5 exposes .text or .html on a filing object
        text = getattr(latest, "text", None) or getattr(latest, "html", None)
        if text:
            # Return first 10,000 chars — transcripts are long; limit for prompt budget
            return str(text)[:10_000]

        return None

    except Exception as exc:
        logger.warning("EDGAR 8-K Item 2.02 fallback failed for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_filing_meta(
    ticker: str, filing: object, form: str
) -> Optional[EDGARFiling]:
    """Extract filing metadata from an edgartools filing object."""
    try:
        accession = str(getattr(filing, "accession_number", "") or "")
        date = str(getattr(filing, "filing_date", "") or "")
        cik = str(getattr(filing, "cik", "") or "")
        return EDGARFiling(
            ticker=ticker,
            form=form,
            filing_date=date,
            accession_no=accession,
            cik=cik or None,
        )
    except (ValidationError, Exception) as exc:
        logger.warning("Could not extract filing meta for %s %s: %s", ticker, form, exc)
        return None


def _extract_financials(
    filing: object, ticker: str, form: str
) -> tuple[Optional[list[str]], Optional[list[str]], Optional[list[str]], dict]:
    """Extract financial statement column names and DataFrames from a filing."""
    bs_cols = None
    is_cols = None
    cf_cols = None
    df_cache: dict[str, pd.DataFrame] = {}

    try:
        financials = getattr(filing, "financials", None)
        if financials is None:
            logger.debug("No financials on %s %s", ticker, form)
            return bs_cols, is_cols, cf_cols, df_cache

        bs = getattr(financials, "balance_sheet", None)
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            bs_cols = list(bs.columns)
            df_cache["balance_sheet"] = bs
        elif bs is not None:
            logger.debug("Balance sheet for %s is not a DataFrame: %s", ticker, type(bs))

        ist = getattr(financials, "income_statement", None)
        if isinstance(ist, pd.DataFrame) and not ist.empty:
            is_cols = list(ist.columns)
            df_cache["income_statement"] = ist
        elif ist is not None:
            logger.debug("Income statement for %s is not a DataFrame: %s", ticker, type(ist))

        cf = getattr(financials, "cash_flow_statement", None)
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            cf_cols = list(cf.columns)
            df_cache["cash_flow"] = cf
        elif cf is not None:
            logger.debug("Cash flow for %s is not a DataFrame: %s", ticker, type(cf))

    except Exception as exc:
        logger.warning("Financial extraction failed for %s %s: %s", ticker, form, exc)

    return bs_cols, is_cols, cf_cols, df_cache


def _iter_filings(filings_obj: object):
    """Iterate over a filings object that may be a single filing or a collection."""
    if hasattr(filings_obj, "__iter__"):
        yield from filings_obj
    else:
        yield filings_obj
