# data_tools/__init__.py — On-demand data tool layer for agentic personas.
#
# Exposes callable tools the personas invoke by judgment during research.
# Each tool wraps a source client, validates the response against an explicit
# schema, and returns typed data.  A central shared rate limiter enforces
# per-source quotas across all concurrent persona calls.
#
# TDD Component 2 (data_tool_layer) — TASK-M1-003.

from round_table_portfolio.data_tools.finnhub_tools import (
    get_prices,
    get_fundamentals,
    get_company_news,
    get_earnings_transcript,
    get_peers,
)
from round_table_portfolio.data_tools.edgar_tools import get_filings
from round_table_portfolio.data_tools.fred_tools import get_macro_series
from round_table_portfolio.data_tools.rss_tools import get_rss_headlines
from round_table_portfolio.data_tools.technical_tools import compute_technicals
from round_table_portfolio.data_tools.prenarrow import pre_narrow
from round_table_portfolio.data_tools.manifest import record_tool_call

__all__ = [
    "get_prices",
    "get_fundamentals",
    "get_company_news",
    "get_earnings_transcript",
    "get_peers",
    "get_filings",
    "get_macro_series",
    "get_rss_headlines",
    "compute_technicals",
    "pre_narrow",
    "record_tool_call",
]
