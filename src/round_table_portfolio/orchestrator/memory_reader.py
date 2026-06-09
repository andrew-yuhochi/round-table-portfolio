"""Component 26 — memory_reader.

READ half of the per-persona memory loop (Component 18 owns the write half).

For each persona, this module:
  1. Parses ``state/memory/<persona>.md`` via the existing ``parse_memory_file``
     (never reimplements parsing — round-trip contract lives in memory.py).
  2. Applies the recency window: keeps the *most-recent* ``memory_window_weeks``
     entries **per section** for injection, newest-last (chronological).
     The window bounds what gets INJECTED, NOT the file — the file always
     retains up to the 12-entry cap (Component 18's write contract).
  3. Runs the resolved-outcomes query against ``weekly_returns`` joined through
     ``portfolios`` / ``holdings`` to identify which of this persona's prior
     counterfactual calls have resolved since its last run, and what alpha they
     achieved vs SPY.

Returns, per persona:
  - A ``WindowedMemory`` with four recency-windowed sections.
  - A ``resolved_alpha`` map (ticker → alpha) for Component 18b's backfill.
  - The raw ``Sequence[ResolvedRow]`` for Component 28's digest.

Read-only contract:
  Component 26 NEVER writes to the memory files, to the archive, or to the
  ledger.  The sole writer is Component 18 / 18b.  Any code path in this
  module that could write would be a Major integrity violation.

Cold-start:
  A persona whose memory file is absent or empty returns four empty sections
  and an empty resolved set — no crash.  This is the week-1 bootstrap case.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import yaml

from round_table_portfolio.orchestrator.digest import ResolvedRow
from round_table_portfolio.orchestrator.memory import (
    SECTION_COUNTERFACTUAL,
    SECTION_DEBATE_STANCES,
    SECTION_PAST_CALLS,
    SECTION_WHATS_NEW,
    MemorySection,
    parse_memory_file,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(os.environ.get("THRESHOLDS_CONFIG", "config/thresholds.yaml"))

_DEFAULT_MEMORY_WINDOW_WEEKS: int = 8


@dataclass(frozen=True)
class MemoryReaderConfig:
    """Typed view of the memory-reader section of thresholds.yaml."""

    memory_window_weeks: int


def load_memory_reader_config(
    config_path: Optional[Path] = None,
) -> MemoryReaderConfig:
    """Read thresholds.yaml and return a MemoryReaderConfig.

    Falls back to built-in defaults for any missing key so runs before the
    config key is present work without error.
    """
    path = config_path or _CONFIG_PATH
    raw: dict = {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logger.warning(
            "thresholds.yaml not found at %s — using built-in defaults.", path
        )

    return MemoryReaderConfig(
        memory_window_weeks=int(
            raw.get("memory_window_weeks", _DEFAULT_MEMORY_WINDOW_WEEKS)
        ),
    )


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowedMemory:
    """Per-persona recency-windowed view of all four memory sections.

    Each section contains only the most-recent ``memory_window_weeks`` entries,
    preserved in chronological (oldest-first) order — "newest-last."

    This is what Components 27 and 28 consume.  It is NOT the full memory file;
    the file retains up to the 12-entry cap independently.
    """

    persona: str
    past_calls: list[tuple[str, str]]         # (week_id, body_text)
    counterfactual: list[tuple[str, str]]     # (week_id, body_text)
    debate_stances: list[tuple[str, str]]     # (week_id, body_text)
    whats_new: list[tuple[str, str]]          # (week_id, body_text)


@dataclass(frozen=True)
class PersonaMemoryResult:
    """All Component 26 outputs for one persona in one run.

    Fields:
        windowed_memory: The four recency-windowed sections for this persona.
        resolved_alpha:  Map of ticker → alpha for Component 18b backfill.
        resolved_rows:   Raw ResolvedRow sequence for Component 28's digest.
    """

    windowed_memory: WindowedMemory
    resolved_alpha: dict[str, float]
    resolved_rows: list[ResolvedRow]


# ---------------------------------------------------------------------------
# Recency window
# ---------------------------------------------------------------------------

def _apply_window(section: MemorySection, window: int) -> list[tuple[str, str]]:
    """Return the most-recent ``window`` entries from *section*, newest-last.

    The full entry list in *section* is unchanged — this function returns a
    projection (a slice) and never mutates the section or its backing file.

    If the section has N ≤ window entries, all N are returned.
    """
    entries = section.entries  # already chronological (oldest-first) per memory.py
    return list(entries[-window:]) if len(entries) > window else list(entries)


# ---------------------------------------------------------------------------
# Resolved-outcomes query
# ---------------------------------------------------------------------------

# The persona-type column in ``portfolios`` uses the canonical slug strings
# defined in the schema CHECK constraint — these ARE the persona slugs.
_PERSONA_SLUGS_7 = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]


def _query_resolved_rows(
    conn: sqlite3.Connection,
    persona: str,
    last_run_week_id: Optional[str],
) -> list[ResolvedRow]:
    """Query ``weekly_returns`` for resolved outcomes since the persona's last run.

    "Resolved since last run" means:
      - The portfolio is the persona's counterfactual (``portfolios.type = persona``).
      - A ``weekly_returns`` row exists for that portfolio (alpha was marked to
        market for ``as_of_week_id``).
      - ``as_of_week_id`` is strictly AFTER ``last_run_week_id`` (ISO week
        string lexicographic comparison — valid because all week_ids are
        'YYYY-WNN' and the format sorts correctly as strings).

    When ``last_run_week_id`` is None (persona has never run), the condition
    ``as_of_week_id > ''`` is always true — all resolved rows are returned,
    which is the correct bootstrap behaviour (first call, sees everything).

    Alpha definition: the ``alpha`` column in ``weekly_returns`` — continuous
    annualised alpha stored by the M2 mark-to-market step.  This is the same
    definition used by the M2/M3 counterfactual capture (never recomputed here).

    The join path:
        weekly_returns ← portfolios ← holdings
    ``holdings.action`` is retrieved for the ticker via a sub-join on
    ``holdings.portfolio_id = portfolios.portfolio_id AND holdings.ticker = ...``.

    Args:
        conn:             Read-only SQLite connection.
        persona:          Persona slug to query.
        last_run_week_id: The most-recent week_id for which this persona already
                          has a memory entry.  Rows with ``as_of_week_id`` strictly
                          after this value are "new."  None → all rows.

    Returns:
        List of ResolvedRow, one per (portfolio, as_of_week, ticker) tuple.
    """
    since = last_run_week_id or ""

    sql = """
        SELECT
            p.type          AS persona,
            h.ticker        AS ticker,
            p.week_id       AS call_week_id,
            wr.as_of_week_id,
            wr.alpha,
            h.action        AS action
        FROM weekly_returns wr
        JOIN portfolios p  ON wr.portfolio_id = p.portfolio_id
        JOIN holdings   h  ON h.portfolio_id  = p.portfolio_id
        WHERE p.type = ?
          AND h.ticker != 'CASH'
          AND wr.alpha IS NOT NULL
          AND wr.as_of_week_id > ?
        ORDER BY p.week_id, h.ticker, wr.as_of_week_id
    """
    try:
        rows = conn.execute(sql, (persona, since)).fetchall()
    except sqlite3.Error:
        logger.exception(
            "resolved-outcomes query failed: persona=%s since=%s", persona, since
        )
        raise

    result: list[ResolvedRow] = []
    for persona_val, ticker, call_week_id, as_of_week_id, alpha, action in rows:
        result.append(
            ResolvedRow(
                persona=persona_val,
                ticker=ticker,
                call_week_id=call_week_id,
                as_of_week_id=as_of_week_id,
                alpha=float(alpha),
                action=action,
            )
        )
    return result


def _resolved_alpha_map(resolved_rows: list[ResolvedRow]) -> dict[str, float]:
    """Collapse resolved rows to a ticker → alpha map for Component 18b.

    When the same ticker resolves across multiple weeks (multiple ``as_of_week``
    values), the MOST-RECENT alpha (latest ``as_of_week_id``) wins, because the
    most recent mark-to-market is the most informative.  ISO week strings
    compare lexicographically correctly so max() works directly.

    Args:
        resolved_rows: Output of ``_query_resolved_rows``.

    Returns:
        ``{ticker: alpha}`` — one entry per ticker, latest alpha.
    """
    # Group by ticker; keep the row with the max as_of_week_id.
    latest: dict[str, ResolvedRow] = {}
    for row in resolved_rows:
        existing = latest.get(row.ticker)
        if existing is None or row.as_of_week_id > existing.as_of_week_id:
            latest[row.ticker] = row
    return {ticker: row.alpha for ticker, row in latest.items()}


# ---------------------------------------------------------------------------
# "Last run" derivation from parsed memory
# ---------------------------------------------------------------------------

def _last_run_week_id(windowed: WindowedMemory) -> Optional[str]:
    """Derive the latest week_id from the FULL windowed memory's past-calls log.

    Uses ``past_calls`` entries because they are written for every run by
    Component 18 immediately after the ledger commits.  The most-recent entry
    is the canonical "persona last ran in week X" signal.

    Returns None when the past-calls log is empty (cold-start / week-1).
    """
    if not windowed.past_calls:
        return None
    # entries are chronological (oldest-first); the last entry is most recent.
    return windowed.past_calls[-1][0]


# ---------------------------------------------------------------------------
# Per-persona reader
# ---------------------------------------------------------------------------

def read_persona_memory(
    persona: str,
    conn: sqlite3.Connection,
    *,
    memory_dir: Path = Path("state/memory"),
    config: Optional[MemoryReaderConfig] = None,
) -> PersonaMemoryResult:
    """Read, window, and query memory for one persona.

    This is the per-persona entry point called by ``read_all_personas_memory``.

    Args:
        persona:    Persona slug (e.g. "value").
        conn:       Read-only SQLite connection over the ledger.
        memory_dir: Root of the memory files (default ``state/memory/``).
        config:     MemoryReaderConfig.  Loaded from thresholds.yaml if None.

    Returns:
        PersonaMemoryResult with windowed memory, resolved_alpha map, and
        raw resolved rows.
    """
    cfg = config or load_memory_reader_config()
    window = cfg.memory_window_weeks

    memory_path = memory_dir / f"{persona}.md"
    parsed = parse_memory_file(memory_path)  # tolerant of missing file

    windowed = WindowedMemory(
        persona=persona,
        past_calls=_apply_window(
            parsed.get_section(SECTION_PAST_CALLS), window
        ),
        counterfactual=_apply_window(
            parsed.get_section(SECTION_COUNTERFACTUAL), window
        ),
        debate_stances=_apply_window(
            parsed.get_section(SECTION_DEBATE_STANCES), window
        ),
        whats_new=_apply_window(
            parsed.get_section(SECTION_WHATS_NEW), window
        ),
    )

    # The "since last run" predicate is derived from the FULL section (before
    # windowing) because archiving or a small window should not shift the
    # resolved-since marker.  Use the raw parsed section, not the windowed view.
    full_past_calls = parsed.get_section(SECTION_PAST_CALLS).entries
    last_run: Optional[str] = (
        full_past_calls[-1][0] if full_past_calls else None
    )

    resolved_rows = _query_resolved_rows(conn, persona, last_run)
    alpha_map = _resolved_alpha_map(resolved_rows)

    logger.debug(
        "memory_reader: persona=%s window=%d past_calls=%d resolved=%d last_run=%s",
        persona, window,
        len(windowed.past_calls),
        len(resolved_rows),
        last_run,
    )

    return PersonaMemoryResult(
        windowed_memory=windowed,
        resolved_alpha=alpha_map,
        resolved_rows=resolved_rows,
    )


# ---------------------------------------------------------------------------
# All-personas entry point
# ---------------------------------------------------------------------------

def read_all_personas_memory(
    conn: sqlite3.Connection,
    *,
    personas: Optional[list[str]] = None,
    memory_dir: Path = Path("state/memory"),
    config: Optional[MemoryReaderConfig] = None,
) -> dict[str, PersonaMemoryResult]:
    """Read memory for all (or specified) personas.

    Args:
        conn:       Read-only SQLite connection over the ledger.
        personas:   List of persona slugs to read.  Defaults to the canonical
                    7-persona list from the schema.
        memory_dir: Root of the memory files (default ``state/memory/``).
        config:     MemoryReaderConfig.  Loaded from thresholds.yaml if None.

    Returns:
        Dict mapping persona slug → PersonaMemoryResult.
    """
    cfg = config or load_memory_reader_config()
    target_personas = personas if personas is not None else _PERSONA_SLUGS_7

    results: dict[str, PersonaMemoryResult] = {}
    for persona in target_personas:
        results[persona] = read_persona_memory(
            persona,
            conn,
            memory_dir=memory_dir,
            config=cfg,
        )

    return results
