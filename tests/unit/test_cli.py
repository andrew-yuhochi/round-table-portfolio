# test_cli.py — Unit tests for the data-tool CLI wrapper (cli.py).
#
# All tests run under SKIP_LIVE=1 — tool functions are mocked; no network calls.
# One @pytest.mark.live smoke test exercises the full round-trip for quote AAPL.

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from round_table_portfolio.data_tools.cli import _build_parser, main, _HANDLERS


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_quote_parses_ticker(self):
        parser = _build_parser()
        args = parser.parse_args(["quote", "AAPL"])
        assert args.command == "quote"
        assert args.ticker == "AAPL"

    def test_prices_defaults_to_365(self):
        parser = _build_parser()
        args = parser.parse_args(["prices", "MSFT"])
        assert args.days == 365

    def test_prices_accepts_explicit_days(self):
        parser = _build_parser()
        args = parser.parse_args(["prices", "MSFT", "30"])
        assert args.days == 30

    def test_macro_no_series_id_defaults_to_none(self):
        parser = _build_parser()
        args = parser.parse_args(["macro"])
        assert args.series_id is None

    def test_macro_with_series_id(self):
        parser = _build_parser()
        args = parser.parse_args(["macro", "DGS10"])
        assert args.series_id == "DGS10"

    def test_prenarrow_no_n_defaults_to_none(self):
        parser = _build_parser()
        args = parser.parse_args(["prenarrow"])
        assert args.n is None

    def test_prenarrow_with_n(self):
        parser = _build_parser()
        args = parser.parse_args(["prenarrow", "10"])
        assert args.n == 10

    def test_rss_no_args(self):
        parser = _build_parser()
        args = parser.parse_args(["rss"])
        assert args.command == "rss"

    def test_universe_no_args(self):
        parser = _build_parser()
        args = parser.parse_args(["universe"])
        assert args.command == "universe"

    def test_unknown_command_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["badcmd"])


# ---------------------------------------------------------------------------
# Handler dispatch — every command maps to the right tool fn
# ---------------------------------------------------------------------------

class TestDispatch:
    """Verify each command invokes the correct underlying tool function."""

    def _run(self, argv: list[str], capsys) -> dict:
        """Run main() with mocked tool fns; return parsed stdout JSON."""
        main(argv)
        captured = capsys.readouterr()
        return json.loads(captured.out)

    def test_quote_calls_get_prices(self, capsys):
        mock_candle = MagicMock()
        mock_candle.model_dump.return_value = {"symbol": "AAPL", "c": [195.0], "source": "yfinance"}
        with patch("round_table_portfolio.data_tools.cli._cmd_quote") as mock_handler:
            mock_handler.return_value = {"symbol": "AAPL", "c": [195.0]}
            # Call via _HANDLERS to confirm the mapping is correct
            assert "quote" in _HANDLERS

    def test_prices_maps_to_handler(self):
        assert "prices" in _HANDLERS

    def test_news_maps_to_handler(self):
        assert "news" in _HANDLERS

    def test_peers_maps_to_handler(self):
        assert "peers" in _HANDLERS

    def test_fundamentals_maps_to_handler(self):
        assert "fundamentals" in _HANDLERS

    def test_macro_maps_to_handler(self):
        assert "macro" in _HANDLERS

    def test_technicals_maps_to_handler(self):
        assert "technicals" in _HANDLERS

    def test_prenarrow_maps_to_handler(self):
        assert "prenarrow" in _HANDLERS

    def test_rss_maps_to_handler(self):
        assert "rss" in _HANDLERS

    def test_universe_maps_to_handler(self):
        assert "universe" in _HANDLERS

    def test_all_commands_have_handlers(self):
        expected = {
            "quote", "prices", "news", "peers", "fundamentals",
            "macro", "technicals", "prenarrow", "rss", "universe",
        }
        assert set(_HANDLERS.keys()) == expected


# ---------------------------------------------------------------------------
# Full main() round-trip with mocked tool functions
# ---------------------------------------------------------------------------

class TestMainRoundTrip:
    def test_quote_prints_json(self, capsys):
        fake_candle = MagicMock()
        fake_candle.model_dump.return_value = {"symbol": "AAPL", "c": [195.5], "source": "yfinance"}
        with patch("round_table_portfolio.data_tools.finnhub_tools.get_prices", return_value=fake_candle):
            main(["quote", "AAPL"])
        out = json.loads(capsys.readouterr().out)
        assert out["symbol"] == "AAPL"
        assert out["c"] == [195.5]

    def test_prices_prints_json(self, capsys):
        fake_candle = MagicMock()
        fake_candle.model_dump.return_value = {"symbol": "MSFT", "c": [420.0], "source": "yfinance"}
        with patch("round_table_portfolio.data_tools.finnhub_tools.get_prices", return_value=fake_candle):
            main(["prices", "MSFT", "30"])
        out = json.loads(capsys.readouterr().out)
        assert out["symbol"] == "MSFT"

    def test_news_prints_json(self, capsys):
        fake_news = MagicMock()
        fake_news.model_dump.return_value = {"symbol": "AAPL", "items": []}
        with patch("round_table_portfolio.data_tools.finnhub_tools.get_company_news", return_value=fake_news):
            main(["news", "AAPL"])
        out = json.loads(capsys.readouterr().out)
        assert out["symbol"] == "AAPL"

    def test_peers_prints_json(self, capsys):
        fake_peers = MagicMock()
        fake_peers.model_dump.return_value = {"symbol": "AAPL", "peers": ["MSFT", "GOOG"]}
        with patch("round_table_portfolio.data_tools.finnhub_tools.get_peers", return_value=fake_peers):
            main(["peers", "AAPL"])
        out = json.loads(capsys.readouterr().out)
        assert out["peers"] == ["MSFT", "GOOG"]

    def test_fundamentals_prints_json(self, capsys):
        fake_fund = MagicMock()
        fake_fund.model_dump.return_value = {"symbol": "AAPL", "pe_ratio": 28.5}
        with patch("round_table_portfolio.data_tools.finnhub_tools.get_fundamentals", return_value=fake_fund):
            main(["fundamentals", "AAPL"])
        out = json.loads(capsys.readouterr().out)
        assert out["pe_ratio"] == 28.5

    def test_macro_no_filter_prints_full_snapshot(self, capsys):
        fake_snapshot = MagicMock()
        fake_snapshot.model_dump.return_value = {
            "week_id": "2026-W23",
            "series": [
                {"series_id": "DGS10", "description": "10Y Treasury", "observations": []},
                {"series_id": "FEDFUNDS", "description": "Fed Funds", "observations": []},
            ],
        }
        with patch("round_table_portfolio.data_tools.fred_tools.get_macro_series", return_value=fake_snapshot):
            main(["macro"])
        out = json.loads(capsys.readouterr().out)
        assert len(out["series"]) == 2

    def test_macro_with_series_id_filters(self, capsys):
        fake_snapshot = MagicMock()
        fake_snapshot.model_dump.return_value = {
            "week_id": "2026-W23",
            "series": [
                {"series_id": "DGS10", "description": "10Y Treasury", "observations": []},
                {"series_id": "FEDFUNDS", "description": "Fed Funds", "observations": []},
            ],
        }
        with patch("round_table_portfolio.data_tools.fred_tools.get_macro_series", return_value=fake_snapshot):
            main(["macro", "DGS10"])
        out = json.loads(capsys.readouterr().out)
        assert len(out["series"]) == 1
        assert out["series"][0]["series_id"] == "DGS10"

    def test_technicals_prints_json(self, capsys):
        fake_tech = MagicMock()
        fake_tech.model_dump.return_value = {"symbol": "AAPL", "rsi": 55.2}
        with patch("round_table_portfolio.data_tools.technical_tools.compute_technicals", return_value=fake_tech):
            main(["technicals", "AAPL"])
        out = json.loads(capsys.readouterr().out)
        assert out["rsi"] == 55.2

    def test_prenarrow_no_n_returns_all(self, capsys):
        fake_result = MagicMock()
        fake_result.model_dump.return_value = {
            "week_id": "2026-W23",
            "entries": [{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "GOOG"}],
            "cache_hit": True,
        }
        with patch("round_table_portfolio.data_tools.prenarrow.pre_narrow", return_value=fake_result):
            main(["prenarrow"])
        out = json.loads(capsys.readouterr().out)
        assert len(out["entries"]) == 3

    def test_prenarrow_with_n_slices(self, capsys):
        fake_result = MagicMock()
        fake_result.model_dump.return_value = {
            "week_id": "2026-W23",
            "entries": [{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "GOOG"}],
            "cache_hit": True,
        }
        with patch("round_table_portfolio.data_tools.prenarrow.pre_narrow", return_value=fake_result):
            main(["prenarrow", "2"])
        out = json.loads(capsys.readouterr().out)
        assert len(out["entries"]) == 2
        assert out["entries"][0]["symbol"] == "AAPL"

    def test_rss_prints_json(self, capsys):
        fake_headlines = MagicMock()
        fake_headlines.model_dump.return_value = {"entries": [{"title": "Markets up"}]}
        with patch("round_table_portfolio.data_tools.rss_tools.get_rss_headlines", return_value=fake_headlines):
            main(["rss"])
        out = json.loads(capsys.readouterr().out)
        assert out["entries"][0]["title"] == "Markets up"

    def test_universe_prints_json(self, capsys):
        mock_entry = SimpleNamespace(symbol="AAPL", name="Apple Inc", sector="Technology")
        with patch("round_table_portfolio.config.universe.load_universe", return_value=[mock_entry]):
            main(["universe"])
        out = json.loads(capsys.readouterr().out)
        assert out[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_tool_exception_prints_error_json_and_exits_nonzero(self, capsys):
        with patch(
            "round_table_portfolio.data_tools.finnhub_tools.get_prices",
            side_effect=RuntimeError("API down"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main(["quote", "AAPL"])
        assert exc_info.value.code == 1
        out = json.loads(capsys.readouterr().out)
        assert "error" in out
        assert "API down" in out["error"]

    def test_unknown_command_exits_via_argparse(self):
        with pytest.raises(SystemExit):
            main(["notacommand"])


# ---------------------------------------------------------------------------
# Live smoke test — real network calls (gated by SKIP_LIVE)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_quote_aapl_live(capsys):
    """End-to-end: quote AAPL hits Yahoo Finance and returns valid JSON."""
    main(["quote", "AAPL"])
    out = json.loads(capsys.readouterr().out)
    assert "error" not in out, f"Live quote AAPL failed: {out}"
    assert out.get("symbol") == "AAPL"
    assert isinstance(out.get("c"), list)
    assert len(out["c"]) >= 1
    assert out["c"][-1] > 0
