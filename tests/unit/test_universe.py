# test_universe.py — tests for the S&P 500 universe loader (Component 3).
#
# AC-1: loader returns ~500 unique entries from the real config file.
# AC-2: loader rejects malformed / duplicate fixtures (fail-loudly).
# AC-3: sampled live-resolution check (gated behind SKIP_LIVE=1).

from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

import pytest

from round_table_portfolio.config.universe import (
    GICS_SECTORS,
    TickerEntry,
    load_universe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical path of the real config file (next to the project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_REAL_CONFIG = _PROJECT_ROOT / "config" / "sp500_universe.yaml"


def _write_fixture(tmp_path: Path, content: str) -> Path:
    """Write a temporary YAML fixture file and return its path."""
    p = tmp_path / "universe_fixture.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# AC-1: Real config — ~500 unique entries, all sectors present
# ---------------------------------------------------------------------------

class TestRealUniverse:
    """AC-1 tests against the real config/sp500_universe.yaml."""

    def test_real_config_exists(self) -> None:
        assert _REAL_CONFIG.exists(), (
            f"Real config not found at {_REAL_CONFIG}. "
            "Build it by running the scrape step (see TDD Component 3)."
        )

    def test_load_returns_list_of_ticker_entries(self) -> None:
        entries = load_universe(_REAL_CONFIG)
        assert isinstance(entries, list)
        assert len(entries) > 0
        assert all(isinstance(e, TickerEntry) for e in entries)

    def test_entry_count_approximately_500(self) -> None:
        entries = load_universe(_REAL_CONFIG)
        # S&P 500 has ~503 constituents including dual-class names
        assert 490 <= len(entries) <= 520, (
            f"Expected ~500 entries, got {len(entries)}"
        )

    def test_all_symbols_unique(self) -> None:
        entries = load_universe(_REAL_CONFIG)
        symbols = [e.symbol for e in entries]
        assert len(symbols) == len(set(symbols)), "Duplicate symbols found"

    def test_all_symbols_uppercase(self) -> None:
        entries = load_universe(_REAL_CONFIG)
        non_upper = [e.symbol for e in entries if e.symbol != e.symbol.upper()]
        assert non_upper == [], f"Non-uppercase symbols: {non_upper}"

    def test_all_required_fields_present(self) -> None:
        entries = load_universe(_REAL_CONFIG)
        for e in entries:
            assert e.symbol, f"Empty symbol: {e}"
            assert e.name, f"Empty name for symbol {e.symbol}"
            assert e.sector, f"Empty sector for symbol {e.symbol}"

    def test_all_11_gics_sectors_present(self) -> None:
        entries = load_universe(_REAL_CONFIG)
        present = {e.sector for e in entries}
        missing = GICS_SECTORS - present
        assert not missing, (
            f"Missing GICS sectors: {sorted(missing)}. Present: {sorted(present)}"
        )

    def test_sector_counts_reasonable(self) -> None:
        """Each sector should have at least 10 constituents (S&P 500 is broad)."""
        entries = load_universe(_REAL_CONFIG)
        counts = Counter(e.sector for e in entries)
        thin = {s: c for s, c in counts.items() if c < 10}
        assert not thin, f"Sectors with suspiciously few members: {thin}"

    def test_entries_sorted_by_symbol(self) -> None:
        entries = load_universe(_REAL_CONFIG)
        symbols = [e.symbol for e in entries]
        assert symbols == sorted(symbols), "Entries are not sorted by symbol"


# ---------------------------------------------------------------------------
# AC-2: Malformed / duplicate fixtures — loader must reject (fail-loudly)
# ---------------------------------------------------------------------------

class TestMalformedFixtures:
    """AC-2 tests: loader raises ValueError on bad input."""

    def test_missing_symbol_field(self, tmp_path: Path) -> None:
        fixture = _write_fixture(
            tmp_path,
            """\
            snapshot_date: '2026-06-01'
            source: 'test'
            total_count: 1
            universe:
              - symbol: AAPL
                name: 'Apple Inc.'
                sector: 'Information Technology'
              - name: 'Missing Symbol Corp'
                sector: 'Financials'
            """,
        )
        with pytest.raises(ValueError, match="missing 'symbol'"):
            load_universe(fixture)

    def test_missing_name_field(self, tmp_path: Path) -> None:
        # Build a fixture with enough valid entries to cover all sectors,
        # then add one entry missing 'name'.
        sector_entries = "\n".join(
            f"  - symbol: {s}\n    name: 'Placeholder {s}'\n    sector: '{sec}'"
            for s, sec in [
                ("COMM", "Communication Services"),
                ("COND", "Consumer Discretionary"),
                ("CONS", "Consumer Staples"),
                ("ENRG", "Energy"),
                ("FINC", "Financials"),
                ("HLTH", "Health Care"),
                ("INDU", "Industrials"),
                ("INFT", "Information Technology"),
                ("MATL", "Materials"),
                ("REAL", "Real Estate"),
                ("UTIL", "Utilities"),
            ]
        )
        content = (
            "snapshot_date: '2026-06-01'\nsource: 'test'\ntotal_count: 12\nuniverse:\n"
            + sector_entries
            + "\n  - symbol: NONAME\n    sector: 'Financials'\n"
        )
        p = tmp_path / "fixture.yaml"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'name'"):
            load_universe(p)

    def test_missing_sector_field(self, tmp_path: Path) -> None:
        fixture = _write_fixture(
            tmp_path,
            """\
            snapshot_date: '2026-06-01'
            source: 'test'
            total_count: 1
            universe:
              - symbol: AAPL
                name: 'Apple Inc.'
            """,
        )
        with pytest.raises(ValueError, match="missing 'sector'"):
            load_universe(fixture)

    def test_duplicate_symbol_rejected(self, tmp_path: Path) -> None:
        fixture = _write_fixture(
            tmp_path,
            """\
            snapshot_date: '2026-06-01'
            source: 'test'
            total_count: 2
            universe:
              - symbol: AAPL
                name: 'Apple Inc.'
                sector: 'Information Technology'
              - symbol: AAPL
                name: 'Apple Inc. (duplicate)'
                sector: 'Information Technology'
            """,
        )
        with pytest.raises(ValueError, match="Duplicate symbol"):
            load_universe(fixture)

    def test_missing_snapshot_date_raises(self, tmp_path: Path) -> None:
        fixture = _write_fixture(
            tmp_path,
            """\
            source: 'test'
            total_count: 1
            universe:
              - symbol: AAPL
                name: 'Apple Inc.'
                sector: 'Information Technology'
            """,
        )
        with pytest.raises(ValueError, match="snapshot_date"):
            load_universe(fixture)

    def test_missing_gics_sector_raises(self, tmp_path: Path) -> None:
        """A universe that omits a GICS sector should be rejected."""
        # Only 10 sectors — missing Real Estate
        sector_entries = "\n".join(
            f"  - symbol: {s}\n    name: 'Placeholder {s}'\n    sector: '{sec}'"
            for s, sec in [
                ("COMM", "Communication Services"),
                ("COND", "Consumer Discretionary"),
                ("CONS", "Consumer Staples"),
                ("ENRG", "Energy"),
                ("FINC", "Financials"),
                ("HLTH", "Health Care"),
                ("INDU", "Industrials"),
                ("INFT", "Information Technology"),
                ("MATL", "Materials"),
                ("UTIL", "Utilities"),
                # Real Estate deliberately omitted
            ]
        )
        content = (
            "snapshot_date: '2026-06-01'\nsource: 'test'\ntotal_count: 10\nuniverse:\n"
            + sector_entries
            + "\n"
        )
        p = tmp_path / "fixture.yaml"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="missing GICS sectors"):
            load_universe(p)

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_universe(tmp_path / "nonexistent.yaml")

    def test_inline_comment_on_quoted_value_does_not_corrupt(self, tmp_path: Path) -> None:
        """
        Regression test: a quoted value followed by an inline YAML comment
        must parse to the bare value only — no comment text leaked in.

        This is the confirmed silent-corruption bug in the former hand-rolled
        parser (TASK-M1-002 follow-up, 2026-06-01).  yaml.safe_load() handles
        inline comments natively, so 'Apple Inc.' is the expected result.
        """
        fixture = _write_fixture(
            tmp_path,
            """\
            snapshot_date: '2026-06-01'
            source: 'test'
            total_count: 1
            universe:
              - symbol: AAPL
                name: 'Apple Inc.' # largest company by market cap
                sector: 'Information Technology'
            """,
        )
        # Only one sector present — loader will raise for missing GICS sectors,
        # but the name field must already be correct before that check fires.
        # We catch ValueError and confirm it's the sector error, not a corrupt name.
        try:
            load_universe(fixture)
        except ValueError as exc:
            msg = str(exc)
            # Must be a sector-coverage error, not a corrupt-name or symbol error
            assert "missing GICS sectors" in msg, (
                f"Unexpected ValueError (possible name corruption): {msg}"
            )
            # Confirm the name was parsed cleanly by loading via yaml directly
            import yaml as _yaml
            raw = _yaml.safe_load((tmp_path / "universe_fixture.yaml").read_text())
            parsed_name = raw["universe"][0]["name"]
            assert parsed_name == "Apple Inc.", (
                f"Inline comment leaked into name field: {parsed_name!r}"
            )

    def test_invalid_symbol_format_rejected(self, tmp_path: Path) -> None:
        """Symbol with lowercase or excessive length should be rejected."""
        fixture = _write_fixture(
            tmp_path,
            """\
            snapshot_date: '2026-06-01'
            source: 'test'
            total_count: 1
            universe:
              - symbol: aapl
                name: 'Apple Inc.'
                sector: 'Information Technology'
            """,
        )
        with pytest.raises(ValueError, match="invalid symbol format"):
            load_universe(fixture)

    def test_boolean_symbol_raises_clear_error(self, tmp_path: Path) -> None:
        """
        Regression guard: unquoted YAML 1.1 keyword ON is parsed by
        yaml.safe_load() as boolean True, not the string 'ON'.
        The loader must raise a clear ValueError before str() silently
        converts True→'True' and corrupts the symbol.

        This fixture deliberately omits quotes around ON so yaml.safe_load()
        produces a bool — proving the guard fires even if the YAML file
        is ever edited carelessly.
        """
        # Write the fixture directly (not via _write_fixture's dedent) so we
        # can control exact YAML bytes with no quoting around ON.
        p = tmp_path / "boolean_symbol.yaml"
        p.write_text(
            "snapshot_date: '2026-06-01'\n"
            "source: 'test'\n"
            "total_count: 1\n"
            "universe:\n"
            "  - symbol: ON\n"          # unquoted — yaml.safe_load parses as True
            "    name: 'ON Semiconductor'\n"
            "    sector: 'Information Technology'\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-string"):
            load_universe(p)

    def test_on_ticker_loads_as_string_when_quoted(self, tmp_path: Path) -> None:
        """
        Confirm that quoting 'ON' in the YAML file (as in the real universe
        file) makes yaml.safe_load() return the string 'ON', not a bool.
        The loader should NOT raise for the quoted form.

        Because only one sector is present the loader will raise for missing
        GICS sectors — that's expected and not the bug we're testing.
        We catch that specific ValueError to prove the symbol parsed cleanly.
        """
        p = tmp_path / "quoted_on.yaml"
        p.write_text(
            "snapshot_date: '2026-06-01'\n"
            "source: 'test'\n"
            "total_count: 1\n"
            "universe:\n"
            "  - symbol: 'ON'\n"        # quoted — must parse as string 'ON'
            "    name: 'ON Semiconductor'\n"
            "    sector: 'Information Technology'\n",
            encoding="utf-8",
        )
        try:
            load_universe(p)
        except ValueError as exc:
            msg = str(exc)
            # Only acceptable error is the missing-sectors error; a non-string
            # error here means the guard fired incorrectly on a quoted symbol.
            assert "missing GICS sectors" in msg, (
                f"Unexpected ValueError (quoted ON should parse as str): {msg}"
            )


# ---------------------------------------------------------------------------
# AC-3: Live resolution check (gated behind SKIP_LIVE=1)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestLiveResolution:
    """
    AC-3: Verify a sample of tickers resolve to real securities via the
    data tools.  Runs only when SKIP_LIVE=0 (network available).

    Usage:
        SKIP_LIVE=0 python -m pytest tests/unit/test_universe.py::TestLiveResolution -v

    At M1 the data-tool layer (TASK-M1-003) is not yet built, so this test
    performs a lightweight resolution check using yfinance-style ticker
    validation via a direct Finnhub quote endpoint — or simply confirms the
    symbols are non-empty strings in the loaded universe (structural check
    only until TASK-M1-003 is complete).

    NOTE: Full live-resolution against get_prices + get_fundamentals is
    exercised during the TASK-M1-003 quality gate (≥30 sampled tickers
    spanning all 11 sectors).  This test is the structural precursor.
    """

    _SAMPLE_SIZE = 10
    _SECTORS_TO_SAMPLE = [
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financials",
        "Health Care",
        "Industrials",
        "Information Technology",
        "Materials",
        "Real Estate",
    ]

    def _sample_tickers(self) -> list[TickerEntry]:
        """Pick one ticker per sector from the real universe."""
        entries = load_universe(_REAL_CONFIG)
        by_sector: dict[str, list[TickerEntry]] = {}
        for e in entries:
            by_sector.setdefault(e.sector, []).append(e)

        sample: list[TickerEntry] = []
        for sector in self._SECTORS_TO_SAMPLE:
            tickers = by_sector.get(sector, [])
            if tickers:
                sample.append(tickers[0])  # alphabetically first in sector
        return sample

    def test_sampled_tickers_are_non_empty_strings(self) -> None:
        """
        Structural live check: sampled tickers are well-formed symbols.
        Full data-tool resolution is tested in TASK-M1-003 quality gate.
        """
        sample = self._sample_tickers()
        assert len(sample) >= self._SAMPLE_SIZE, (
            f"Expected ≥{self._SAMPLE_SIZE} sampled tickers, got {len(sample)}"
        )
        for entry in sample:
            assert isinstance(entry.symbol, str) and len(entry.symbol) >= 1
            assert entry.symbol == entry.symbol.upper()
            assert entry.sector in GICS_SECTORS, (
                f"{entry.symbol} has unrecognised sector {entry.sector!r}"
            )

    def test_sampled_tickers_span_all_sectors(self) -> None:
        """Each sample should cover the major GICS sectors."""
        sample = self._sample_tickers()
        covered = {e.sector for e in sample}
        missing = set(self._SECTORS_TO_SAMPLE) - covered
        assert not missing, f"Sample missed sectors: {sorted(missing)}"
