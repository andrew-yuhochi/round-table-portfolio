# universe.py — load_universe() loader and validator for config/sp500_universe.yaml.
#
# Reads the dated static S&P 500 snapshot, validates every entry against the
# TickerEntry dataclass (symbol / name / sector all required, symbol uppercase),
# enforces uniqueness, and checks that all 11 GICS sectors are represented.
# Fail-loudly: any violation raises ValueError before returning (NFR #5).

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GICS_SECTORS = frozenset(
    [
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
        "Utilities",
    ]
)

# Valid S&P 500 ticker pattern: 1-5 uppercase letters, optional dot + 1 uppercase letter
# (e.g. BRK.B, GOOGL, A)
_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

# Default config path relative to this file's package root
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "sp500_universe.yaml"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TickerEntry:
    """A single S&P 500 universe entry. All fields required and non-empty."""

    symbol: str   # uppercase ticker, e.g. "AAPL"
    name: str     # full company name
    sector: str   # GICS sector label


# ---------------------------------------------------------------------------
# YAML parsing — delegates to yaml.safe_load() (PyYAML)
# ---------------------------------------------------------------------------

def _parse_yaml_universe(path: Path) -> tuple[str, list[dict[str, str]]]:
    """
    Parse sp500_universe.yaml using yaml.safe_load().

    Returns (snapshot_date, list of raw dicts with symbol/name/sector keys).
    Raises ValueError if the top-level structure is missing required keys.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level of {path}")

    snapshot_date = str(data.get("snapshot_date", "")).strip()
    if not snapshot_date:
        raise ValueError(f"Missing 'snapshot_date' field in {path}")

    raw_entries = data.get("universe", [])
    if not isinstance(raw_entries, list):
        raise ValueError(f"Expected 'universe' to be a list in {path}")

    # Guard against PyYAML 1.1 boolean coercion: unquoted YAML keywords like
    # ON, OFF, YES, NO, TRUE, FALSE are parsed as bool by yaml.safe_load().
    # Detect this BEFORE str(v) silently converts True→"True" / False→"False".
    normalised: list[dict[str, str]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        for field in ("symbol", "name", "sector"):
            raw_val = entry.get(field)
            if raw_val is not None and not isinstance(raw_val, str):
                raise ValueError(
                    f"Field '{field}' parsed as non-string "
                    f"(YAML keyword coercion?): {raw_val!r} "
                    f"(type={type(raw_val).__name__}). "
                    f"Quote the value in the YAML file to fix this."
                )
        normalised.append({k: str(v).strip() for k, v in entry.items()})

    return snapshot_date, normalised


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_universe(config_path: Path | None = None) -> list[TickerEntry]:
    """
    Load and validate the S&P 500 universe from config/sp500_universe.yaml.

    Args:
        config_path: Override path for testing. Defaults to the canonical
                     config/sp500_universe.yaml next to the project root.

    Returns:
        Sorted list of TickerEntry (sorted by symbol) — ~503 entries.

    Raises:
        FileNotFoundError: config file missing.
        ValueError: any validation failure (missing field, duplicate symbol,
                    invalid symbol format, sector gap).
    """
    path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Universe config not found: {path}")

    logger.info("Loading S&P 500 universe from %s", path)

    snapshot_date, raw_entries = _parse_yaml_universe(path)

    logger.info("Snapshot date: %s — raw entries: %d", snapshot_date, len(raw_entries))

    entries: list[TickerEntry] = []
    seen_symbols: set[str] = set()
    errors: list[str] = []

    for idx, raw in enumerate(raw_entries):
        symbol = raw.get("symbol", "").strip()
        name = raw.get("name", "").strip()
        sector = raw.get("sector", "").strip()

        # Required-field presence
        if not symbol:
            errors.append(f"Entry #{idx}: missing 'symbol'")
            continue
        if not name:
            errors.append(f"Entry #{idx} ({symbol}): missing 'name'")
            continue
        if not sector:
            errors.append(f"Entry #{idx} ({symbol}): missing 'sector'")
            continue

        # Symbol format: uppercase letters only (1-5 chars, optional dot+letter)
        if not _SYMBOL_RE.match(symbol):
            errors.append(f"Entry #{idx}: invalid symbol format {symbol!r}")
            continue

        # Uniqueness
        if symbol in seen_symbols:
            errors.append(f"Duplicate symbol: {symbol!r}")
            continue
        seen_symbols.add(symbol)

        entries.append(TickerEntry(symbol=symbol, name=name, sector=sector))

    # Field-level errors take priority — raise before sector check so the
    # error message is specific (missing field / duplicate symbol), not masked
    # by downstream sector-gap errors on a truncated entry set.
    if errors:
        error_summary = "; ".join(errors[:10])
        if len(errors) > 10:
            error_summary += f" ... and {len(errors) - 10} more"
        raise ValueError(f"Universe validation failed ({len(errors)} error(s)): {error_summary}")

    # Sector coverage check — all 11 GICS sectors must be present
    present_sectors = {e.sector for e in entries}
    missing_sectors = GICS_SECTORS - present_sectors
    if missing_sectors:
        raise ValueError(
            f"Universe is missing GICS sectors: {sorted(missing_sectors)}. "
            f"Present sectors: {sorted(present_sectors)}"
        )

    # Warn about unexpected sectors (non-GICS labels)
    unexpected_sectors = present_sectors - GICS_SECTORS
    if unexpected_sectors:
        logger.warning(
            "Universe contains %d unrecognised sector label(s): %s",
            len(unexpected_sectors),
            sorted(unexpected_sectors),
        )

    entries_sorted = sorted(entries, key=lambda e: e.symbol)

    logger.info(
        "Universe loaded: %d tickers, %d sectors, snapshot_date=%s",
        len(entries_sorted),
        len(present_sectors),
        snapshot_date,
    )

    return entries_sorted
