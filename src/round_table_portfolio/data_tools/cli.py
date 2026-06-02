# cli.py — Command-line wrapper over the data-tool layer.
#
# Usage:
#   python -m round_table_portfolio.data_tools.cli <command> [args...]
#
# Each command prints a JSON result to stdout and exits 0.
# On error: prints {"error": "<message>"} to stdout and exits 1.
#
# This is the tool interface persona subagents use via Bash.
# It imports and calls existing tool functions — no logic is reimplemented here.

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command handlers — each returns a JSON-serialisable dict or list
# ---------------------------------------------------------------------------

def _cmd_quote(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.finnhub_tools import get_prices
    candle = get_prices(args.ticker, days=1)
    return candle.model_dump()


def _cmd_prices(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.finnhub_tools import get_prices
    candle = get_prices(args.ticker, days=args.days)
    return candle.model_dump()


def _cmd_news(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.finnhub_tools import get_company_news
    result = get_company_news(args.ticker)
    return result.model_dump()


def _cmd_peers(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.finnhub_tools import get_peers
    result = get_peers(args.ticker)
    return result.model_dump()


def _cmd_fundamentals(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.finnhub_tools import get_fundamentals
    result = get_fundamentals(args.ticker)
    return result.model_dump()


def _cmd_macro(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.fred_tools import get_macro_series
    # week_id derived from today so the call is self-contained
    week_id = datetime.now(timezone.utc).strftime("%G-W%V")
    snapshot = get_macro_series(week_id)
    # Filter to the requested series_id if provided; otherwise return full snapshot
    payload = snapshot.model_dump()
    if args.series_id:
        sid = args.series_id.upper()
        payload["series"] = [s for s in payload.get("series", []) if s["series_id"] == sid]
    return payload


def _cmd_technicals(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.technical_tools import compute_technicals
    result = compute_technicals(args.ticker)
    return result.model_dump()


def _cmd_prenarrow(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.prenarrow import pre_narrow
    week_id = datetime.now(timezone.utc).strftime("%G-W%V")
    result = pre_narrow(week_id)
    payload = result.model_dump()
    # Slice to top-N if requested
    if args.n is not None:
        payload["entries"] = payload["entries"][: args.n]
    return payload


def _cmd_rss(args: argparse.Namespace) -> object:
    from round_table_portfolio.data_tools.rss_tools import get_rss_headlines
    result = get_rss_headlines()
    return result.model_dump()


def _cmd_universe(args: argparse.Namespace) -> object:
    from round_table_portfolio.config.universe import load_universe
    entries = load_universe()
    return [e.__dict__ for e in entries]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m round_table_portfolio.data_tools.cli",
        description=(
            "Data-tool CLI — each command fetches data and prints JSON to stdout.\n"
            "Intended for use by persona subagents via Bash.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # quote
    p = sub.add_parser("quote", help="Current price snapshot for TICKER")
    p.add_argument("ticker", metavar="TICKER")

    # prices
    p = sub.add_parser("prices", help="Daily OHLCV candles for TICKER")
    p.add_argument("ticker", metavar="TICKER")
    p.add_argument("days", metavar="DAYS", type=int, nargs="?", default=365)

    # news
    p = sub.add_parser("news", help="Recent company news for TICKER")
    p.add_argument("ticker", metavar="TICKER")

    # peers
    p = sub.add_parser("peers", help="Peer tickers for TICKER")
    p.add_argument("ticker", metavar="TICKER")

    # fundamentals
    p = sub.add_parser("fundamentals", help="Valuation/quality metrics for TICKER")
    p.add_argument("ticker", metavar="TICKER")

    # macro
    p = sub.add_parser("macro", help="FRED macro snapshot (optionally filtered to SERIES_ID, e.g. DGS10)")
    p.add_argument("series_id", metavar="SERIES_ID", nargs="?", default=None)

    # technicals
    p = sub.add_parser("technicals", help="Technical indicators (RSI, MACD, BB, …) for TICKER")
    p.add_argument("ticker", metavar="TICKER")

    # prenarrow
    p = sub.add_parser("prenarrow", help="Cached ranked S&P 500 universe snapshot")
    p.add_argument("n", metavar="N", type=int, nargs="?", default=None,
                   help="Return top-N entries only (default: all)")

    # rss
    sub.add_parser("rss", help="RSS market headlines from configured feeds")

    # universe
    sub.add_parser("universe", help="List the full S&P 500 universe from config")

    return parser


# ---------------------------------------------------------------------------
# Dispatch table — maps command name → handler
# ---------------------------------------------------------------------------

_HANDLERS = {
    "quote": _cmd_quote,
    "prices": _cmd_prices,
    "news": _cmd_news,
    "peers": _cmd_peers,
    "fundamentals": _cmd_fundamentals,
    "macro": _cmd_macro,
    "technicals": _cmd_technicals,
    "prenarrow": _cmd_prenarrow,
    "rss": _cmd_rss,
    "universe": _cmd_universe,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = _HANDLERS.get(args.command)
    if handler is None:
        # argparse sub.required=True should make this unreachable, but guard anyway
        print(json.dumps({"error": f"unknown command: {args.command}"}))
        sys.exit(1)

    try:
        result = handler(args)
        print(json.dumps(result, default=str))
    except Exception as exc:
        logger.error("CLI command %r failed: %s", args.command, exc, exc_info=True)
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
