"""Unit tests for orchestrator/memory_reader.py — Component 26
(memory reader + recency window + resolved-outcomes query).

Coverage matrix (Gate 4 ACs):

  AC-1 — Recency window applied correctly (deterministic ≥95%; ≥4 window cells
          + 1 cold-start):
    TestWindowApplication      — 4+ cells: N=0, N=1, N=W, N>W per section;
                                  assert windowed count = min(N, W);
                                  chronological order preserved (newest-last).
    TestColdStart              — absent/empty file → 4 empty sections, no crash.

  AC-2 — Resolved-outcomes query returns correct rows (≥3 query cells):
    TestResolvedQuery          — ≥3 cells: correct rows returned; "since last run"
                                  predicate correct; no double-count; empty case.

  AC-3 — Round-trip with the writer (parse(write(x)) = x, 7 personas):
    TestRoundTrip              — 7 cells (one per persona): writeback_memory writes
                                  week n; memory_reader reads back week n+1 and
                                  reproduces the content written for week n.

  AC-4 — Read-only (no writes to ledger or memory files):
    TestReadOnly               — window does NOT modify the file; no ledger writes.

  Config:
    TestMemoryReaderConfig     — load_memory_reader_config falls back to default=8
                                  when key absent; override works.

  Real-2026-W24 fixture (Gate-4 provenance corollary):
    TestRealW24Fixture         — at least one fixture derived from sanitized real
                                  2026-W24 ledger data (stances_2026_w24_round1.json);
                                  tickers, actions, and confidence values are real;
                                  resolution week and alpha values are synthetic.

Provenance note (real-W24 cells):
  Stances sourced from tests/fixtures/stances_2026_w24_round1.json — the real
  Round-1 stances recorded from the live 2026-W24 run on 2026-06-09.
  Tickers, actions, and confidence values are real; no PII is present.
  Resolution week "2026-W25" and alpha values are SYNTHETIC (deterministic).
  No real data is committed under tests/fixtures/real/.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from round_table_portfolio.orchestrator.digest import ResolvedRow
from round_table_portfolio.orchestrator.memory import (
    SECTION_COUNTERFACTUAL,
    SECTION_DEBATE_STANCES,
    SECTION_PAST_CALLS,
    SECTION_WHATS_NEW,
    ParsedMemoryFile,
    MemorySection,
    parse_memory_file,
    writeback_memory,
)
from round_table_portfolio.orchestrator.memory_reader import (
    MemoryReaderConfig,
    PersonaMemoryResult,
    WindowedMemory,
    _apply_window,
    _last_run_week_id,
    _query_resolved_rows,
    _resolved_alpha_map,
    load_memory_reader_config,
    read_all_personas_memory,
    read_persona_memory,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PERSONA_SLUGS_7 = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]

_W8_CFG = MemoryReaderConfig(memory_window_weeks=8)
_W3_CFG = MemoryReaderConfig(memory_window_weeks=3)
_W1_CFG = MemoryReaderConfig(memory_window_weeks=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_section(n_entries: int, week_prefix: str = "2026-W") -> MemorySection:
    """Build a MemorySection with n_entries, weeks '2026-W01' … '2026-WNN'."""
    s = MemorySection(name=SECTION_PAST_CALLS)
    for i in range(1, n_entries + 1):
        s.entries.append((f"{week_prefix}{i:02d}", f"body for week {i}"))
    return s


def _make_memory_file(n_per_section: int, week_prefix: str = "2026-W") -> ParsedMemoryFile:
    """Build a ParsedMemoryFile with n_per_section entries in every section."""
    pmf = ParsedMemoryFile()
    for sec_name in [
        SECTION_PAST_CALLS,
        SECTION_COUNTERFACTUAL,
        SECTION_DEBATE_STANCES,
        SECTION_WHATS_NEW,
    ]:
        sec = pmf.get_section(sec_name)
        for i in range(1, n_per_section + 1):
            sec.entries.append((f"{week_prefix}{i:02d}", f"{sec_name} body {i}"))
    return pmf


def _past_calls_body(week_id: str, stances: list[tuple[str, str, int]]) -> str:
    """Build a past-calls body matching memory.py format."""
    stance_lines = "\n".join(
        f"  {ticker}: {action} confidence={conf} weight=0.1000"
        for ticker, action, conf in sorted(stances, key=lambda x: x[0])
    )
    return f"week: {week_id}\nstances:\n{stance_lines}\noutcome: pending"


# ---------------------------------------------------------------------------
# Minimal in-memory SQLite ledger for resolved-query tests
# ---------------------------------------------------------------------------

_SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS weeks (
    week_id  TEXT PRIMARY KEY,
    run_date TEXT NOT NULL,
    notes    TEXT,
    user_id  TEXT NOT NULL DEFAULT 'andrew'
);

CREATE TABLE IF NOT EXISTS roster_versions (
    roster_version INTEGER PRIMARY KEY,
    description    TEXT NOT NULL,
    created_date   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enhancement_versions (
    enhancement_version INTEGER PRIMARY KEY,
    description         TEXT NOT NULL,
    created_date        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id             TEXT NOT NULL REFERENCES weeks(week_id),
    type                TEXT NOT NULL,
    user_id             TEXT NOT NULL DEFAULT 'andrew',
    roster_version      INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    enhancement_version INTEGER NOT NULL REFERENCES enhancement_versions(enhancement_version),
    created_at          TEXT NOT NULL,
    UNIQUE(week_id, type, user_id)
);

CREATE TABLE IF NOT EXISTS holdings (
    holding_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id   INTEGER NOT NULL REFERENCES portfolios(portfolio_id),
    ticker         TEXT    NOT NULL,
    weight         REAL    NOT NULL,
    action         TEXT    NOT NULL,
    entry_date     TEXT    NOT NULL,
    user_id        TEXT    NOT NULL DEFAULT 'andrew',
    roster_version INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    UNIQUE(portfolio_id, ticker)
);

CREATE TABLE IF NOT EXISTS weekly_returns (
    return_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id        INTEGER NOT NULL REFERENCES portfolios(portfolio_id),
    as_of_week_id       TEXT    NOT NULL REFERENCES weeks(week_id),
    realized_return     REAL,
    unrealized_return   REAL,
    spy_return          REAL,
    alpha               REAL,
    user_id             TEXT    NOT NULL DEFAULT 'andrew',
    roster_version      INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    enhancement_version INTEGER NOT NULL REFERENCES enhancement_versions(enhancement_version),
    UNIQUE(portfolio_id, as_of_week_id)
);
"""

_SEED_SQL = """
INSERT OR IGNORE INTO roster_versions VALUES (1, 'test', date('now'));
INSERT OR IGNORE INTO enhancement_versions VALUES (1, 'test', date('now'));
"""


def _make_db() -> sqlite3.Connection:
    """Create and seed an in-memory SQLite ledger for tests."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA_DDL)
    conn.executescript(_SEED_SQL)
    conn.commit()
    return conn


def _seed_week(conn: sqlite3.Connection, week_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO weeks (week_id, run_date) VALUES (?, ?)",
        (week_id, "2026-01-01"),
    )


def _seed_portfolio_with_returns(
    conn: sqlite3.Connection,
    persona: str,
    call_week: str,
    as_of_week: str,
    tickers: list[tuple[str, str, float]],  # (ticker, action, alpha)
) -> None:
    """Seed one persona portfolio + holdings + weekly_returns for a resolved call.

    Args:
        conn:      In-memory connection.
        persona:   Portfolio type = persona slug.
        call_week: The week the portfolio was created (call_week_id).
        as_of_week: The week of the return mark (as_of_week_id).
        tickers:   List of (ticker, action, alpha).
    """
    _seed_week(conn, call_week)
    _seed_week(conn, as_of_week)

    conn.execute(
        """
        INSERT OR IGNORE INTO portfolios
          (week_id, type, user_id, roster_version, enhancement_version, created_at)
        VALUES (?, ?, 'andrew', 1, 1, '2026-01-01T00:00:00')
        """,
        (call_week, persona),
    )
    portfolio_id = conn.execute(
        "SELECT portfolio_id FROM portfolios WHERE week_id=? AND type=? AND user_id='andrew'",
        (call_week, persona),
    ).fetchone()[0]

    for ticker, action, alpha in tickers:
        conn.execute(
            """
            INSERT OR IGNORE INTO holdings
              (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
            VALUES (?, ?, 0.1, ?, '2026-01-01', 'andrew', 1)
            """,
            (portfolio_id, ticker, action),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO weekly_returns
              (portfolio_id, as_of_week_id, realized_return, unrealized_return,
               spy_return, alpha, user_id, roster_version, enhancement_version)
            VALUES (?, ?, 0.0, 0.0, 0.0, ?, 'andrew', 1, 1)
            """,
            (portfolio_id, as_of_week, alpha),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# AC-1 — Window application (≥4 window cells + 1 cold-start)
# ---------------------------------------------------------------------------

class TestWindowApplication:
    """Assert recency window keeps min(N, W) most-recent entries, newest-last."""

    def test_window_zero_entries(self) -> None:
        """Window cell 1: N=0 entries → windowed count = 0."""
        section = MemorySection(name=SECTION_PAST_CALLS)
        result = _apply_window(section, window=8)
        assert len(result) == 0

    def test_window_one_entry_less_than_window(self) -> None:
        """Window cell 2: N=1 < W=8 → windowed count = 1."""
        section = _make_section(1)
        result = _apply_window(section, window=8)
        assert len(result) == 1
        assert result[0][0] == "2026-W01"

    def test_window_exactly_w_entries(self) -> None:
        """Window cell 3: N=W=3 → windowed count = 3 (all entries)."""
        section = _make_section(3)
        result = _apply_window(section, window=3)
        assert len(result) == 3
        # Order preserved: oldest-first (newest-last)
        assert [e[0] for e in result] == ["2026-W01", "2026-W02", "2026-W03"]

    def test_window_more_than_w_entries_keeps_most_recent(self) -> None:
        """Window cell 4: N=10 > W=3 → windowed count = 3, most-recent entries kept."""
        section = _make_section(10)
        result = _apply_window(section, window=3)
        assert len(result) == 3
        # Most recent 3 of 10: W08, W09, W10
        assert result[0][0] == "2026-W08"
        assert result[1][0] == "2026-W09"
        assert result[2][0] == "2026-W10"

    def test_window_newest_last_order(self) -> None:
        """Window cell 5: entries returned in chronological order (oldest-first, newest-last)."""
        section = _make_section(5)
        result = _apply_window(section, window=3)
        week_ids = [e[0] for e in result]
        assert week_ids == sorted(week_ids), "Entries must be chronological (oldest-first)"

    def test_window_does_not_mutate_section(self) -> None:
        """Window cell 6: _apply_window does not modify the source section."""
        section = _make_section(10)
        original_count = len(section.entries)
        _apply_window(section, window=3)
        assert len(section.entries) == original_count

    def test_full_memory_file_window_all_sections(self, tmp_path: Path) -> None:
        """Window cell 7: all 4 sections windowed correctly for N=10 > W=3."""
        pmf = _make_memory_file(10)
        # Write file and re-parse to use actual parse_memory_file path.
        from round_table_portfolio.orchestrator.memory import _render_memory_file
        content = _render_memory_file("value", pmf)
        mem_path = tmp_path / "value.md"
        mem_path.write_text(content, encoding="utf-8")

        conn = _make_db()
        result = read_persona_memory(
            "value", conn, memory_dir=tmp_path, config=_W3_CFG
        )
        wm = result.windowed_memory
        assert len(wm.past_calls) == 3
        assert len(wm.counterfactual) == 3
        assert len(wm.debate_stances) == 3
        assert len(wm.whats_new) == 3

    def test_window_8_with_less_than_8_entries(self, tmp_path: Path) -> None:
        """Window cell 8: N=5 < W=8 → all 5 entries returned."""
        pmf = _make_memory_file(5)
        from round_table_portfolio.orchestrator.memory import _render_memory_file
        content = _render_memory_file("growth", pmf)
        mem_path = tmp_path / "growth.md"
        mem_path.write_text(content, encoding="utf-8")

        conn = _make_db()
        result = read_persona_memory(
            "growth", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        wm = result.windowed_memory
        assert len(wm.past_calls) == 5
        assert len(wm.counterfactual) == 5

    def test_windowed_content_matches_most_recent_entries(self, tmp_path: Path) -> None:
        """Window cell 9: windowed entries match the N most-recent entries in the file."""
        pmf = _make_memory_file(6)
        from round_table_portfolio.orchestrator.memory import _render_memory_file
        content = _render_memory_file("technical", pmf)
        mem_path = tmp_path / "technical.md"
        mem_path.write_text(content, encoding="utf-8")

        conn = _make_db()
        result = read_persona_memory(
            "technical", conn, memory_dir=tmp_path, config=_W3_CFG
        )
        wm = result.windowed_memory
        # N=6, W=3 → keep entries 4, 5, 6 (2026-W04, 2026-W05, 2026-W06)
        assert wm.past_calls[0][0] == "2026-W04"
        assert wm.past_calls[2][0] == "2026-W06"


# ---------------------------------------------------------------------------
# AC-1 — Cold-start (1 cell)
# ---------------------------------------------------------------------------

class TestColdStart:
    """Absent or empty memory file → four empty sections, no crash."""

    def test_absent_file_returns_four_empty_sections(self, tmp_path: Path) -> None:
        """Cold-start cell: no file → WindowedMemory with 4 empty sections."""
        conn = _make_db()
        result = read_persona_memory(
            "value", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        wm = result.windowed_memory
        assert wm.persona == "value"
        assert len(wm.past_calls) == 0
        assert len(wm.counterfactual) == 0
        assert len(wm.debate_stances) == 0
        assert len(wm.whats_new) == 0
        assert result.resolved_alpha == {}
        assert result.resolved_rows == []

    def test_empty_file_returns_four_empty_sections(self, tmp_path: Path) -> None:
        """Cold-start cell 2: file exists but empty → 4 empty sections."""
        mem_path = tmp_path / "risk-officer.md"
        mem_path.write_text("", encoding="utf-8")
        conn = _make_db()
        result = read_persona_memory(
            "risk-officer", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        wm = result.windowed_memory
        assert len(wm.past_calls) == 0
        assert len(wm.counterfactual) == 0

    def test_cold_start_resolved_rows_empty(self, tmp_path: Path) -> None:
        """Cold-start: no last_run → resolved query uses '' as since, gets all rows."""
        conn = _make_db()
        # Seed one resolved row in the ledger
        _seed_portfolio_with_returns(
            conn, "value", "2026-W23", "2026-W24", [("NVDA", "add", 0.10)]
        )
        # No memory file → last_run = None → resolved_rows includes all returned rows
        result = read_persona_memory(
            "value", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        # Should return the row (as_of > '' is always true)
        assert len(result.resolved_rows) >= 1
        assert any(r.ticker == "NVDA" for r in result.resolved_rows)


# ---------------------------------------------------------------------------
# AC-2 — Resolved-outcomes query (≥3 query cells)
# ---------------------------------------------------------------------------

class TestResolvedQuery:
    """Assert resolved query returns the correct rows with correct field values."""

    def test_query_returns_row_after_last_run(self, tmp_path: Path) -> None:
        """Query cell 1: row with as_of_week_id > last_run → included."""
        conn = _make_db()
        _seed_portfolio_with_returns(
            conn, "value", "2026-W23", "2026-W25", [("AAPL", "add", 0.07)]
        )
        # last_run = "2026-W24" → row with as_of="2026-W25" included
        rows = _query_resolved_rows(conn, "value", "2026-W24")
        assert len(rows) == 1
        r = rows[0]
        assert r.persona == "value"
        assert r.ticker == "AAPL"
        assert r.call_week_id == "2026-W23"
        assert r.as_of_week_id == "2026-W25"
        assert abs(r.alpha - 0.07) < 1e-9
        assert r.action == "add"

    def test_query_excludes_row_at_or_before_last_run(self) -> None:
        """Query cell 2: row with as_of_week_id == last_run → excluded (strictly after)."""
        conn = _make_db()
        _seed_portfolio_with_returns(
            conn, "growth", "2026-W23", "2026-W24", [("MSFT", "add", 0.05)]
        )
        # last_run = "2026-W24" → row with as_of="2026-W24" excluded
        rows = _query_resolved_rows(conn, "growth", "2026-W24")
        assert len(rows) == 0

    def test_query_returns_multiple_tickers(self) -> None:
        """Query cell 3: multiple tickers in same portfolio → all returned."""
        conn = _make_db()
        _seed_portfolio_with_returns(
            conn, "technical", "2026-W22", "2026-W25", [
                ("NVDA", "add", 0.15),
                ("AMD",  "hold", -0.03),
                ("TSLA", "reduce", 0.01),
            ]
        )
        rows = _query_resolved_rows(conn, "technical", "2026-W24")
        tickers = {r.ticker for r in rows}
        assert tickers == {"NVDA", "AMD", "TSLA"}

    def test_query_excludes_cash_ticker(self) -> None:
        """Query cell 4: CASH holding excluded from resolved rows."""
        conn = _make_db()
        _seed_week(conn, "2026-W23")
        _seed_week(conn, "2026-W25")
        conn.execute(
            "INSERT OR IGNORE INTO portfolios (week_id, type, user_id, roster_version, enhancement_version, created_at) VALUES (?, 'quant-systematic', 'andrew', 1, 1, '2026-01-01')",
            ("2026-W23",),
        )
        pid = conn.execute(
            "SELECT portfolio_id FROM portfolios WHERE week_id='2026-W23' AND type='quant-systematic'",
        ).fetchone()[0]
        # CASH holding + one real ticker
        conn.execute(
            "INSERT OR IGNORE INTO holdings (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version) VALUES (?, 'CASH', 0.6, 'hold', '2026-01-01', 'andrew', 1)",
            (pid,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO holdings (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version) VALUES (?, 'GOOG', 0.4, 'add', '2026-01-01', 'andrew', 1)",
            (pid,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO weekly_returns (portfolio_id, as_of_week_id, realized_return, unrealized_return, spy_return, alpha, user_id, roster_version, enhancement_version) VALUES (?, '2026-W25', 0, 0, 0, 0.08, 'andrew', 1, 1)",
            (pid,),
        )
        conn.commit()
        rows = _query_resolved_rows(conn, "quant-systematic", "2026-W24")
        tickers = {r.ticker for r in rows}
        assert "CASH" not in tickers
        assert "GOOG" in tickers

    def test_query_no_rows_returns_empty_list(self) -> None:
        """Query cell 5: empty ledger → empty list."""
        conn = _make_db()
        rows = _query_resolved_rows(conn, "value", "2026-W24")
        assert rows == []

    def test_query_excludes_null_alpha(self) -> None:
        """Query cell 6: rows with NULL alpha excluded (not yet marked to market)."""
        conn = _make_db()
        _seed_week(conn, "2026-W23")
        _seed_week(conn, "2026-W25")
        conn.execute(
            "INSERT OR IGNORE INTO portfolios (week_id, type, user_id, roster_version, enhancement_version, created_at) VALUES ('2026-W23', 'risk-officer', 'andrew', 1, 1, '2026-01-01')"
        )
        pid = conn.execute(
            "SELECT portfolio_id FROM portfolios WHERE week_id='2026-W23' AND type='risk-officer'"
        ).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO holdings (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version) VALUES (?, 'TLT', 0.3, 'add', '2026-01-01', 'andrew', 1)",
            (pid,),
        )
        # NULL alpha — not yet marked
        conn.execute(
            "INSERT OR IGNORE INTO weekly_returns (portfolio_id, as_of_week_id, realized_return, unrealized_return, spy_return, alpha, user_id, roster_version, enhancement_version) VALUES (?, '2026-W25', 0, 0, 0, NULL, 'andrew', 1, 1)",
            (pid,),
        )
        conn.commit()
        rows = _query_resolved_rows(conn, "risk-officer", "2026-W24")
        assert len(rows) == 0

    def test_resolved_alpha_map_latest_wins(self) -> None:
        """Resolved-alpha map cell: same ticker resolves in 2 weeks → latest alpha wins."""
        rows = [
            ResolvedRow("value", "NVDA", "2026-W22", "2026-W24", 0.05, "add"),
            ResolvedRow("value", "NVDA", "2026-W22", "2026-W25", 0.12, "add"),  # later
        ]
        alpha_map = _resolved_alpha_map(rows)
        assert abs(alpha_map["NVDA"] - 0.12) < 1e-9

    def test_resolved_alpha_map_multiple_tickers(self) -> None:
        """Resolved-alpha map cell 2: distinct tickers get distinct alpha values."""
        rows = [
            ResolvedRow("value", "AAPL", "2026-W22", "2026-W25", 0.03, "hold"),
            ResolvedRow("value", "MSFT", "2026-W22", "2026-W25", -0.02, "reduce"),
        ]
        alpha_map = _resolved_alpha_map(rows)
        assert "AAPL" in alpha_map
        assert "MSFT" in alpha_map
        assert abs(alpha_map["AAPL"] - 0.03) < 1e-9
        assert abs(alpha_map["MSFT"] - (-0.02)) < 1e-9


# ---------------------------------------------------------------------------
# AC-2 — Last-run derivation
# ---------------------------------------------------------------------------

class TestLastRunWeekId:
    """_last_run_week_id returns the most-recent week_id from past-calls entries."""

    def test_empty_windowed_returns_none(self) -> None:
        wm = WindowedMemory("value", [], [], [], [])
        assert _last_run_week_id(wm) is None

    def test_single_entry(self) -> None:
        wm = WindowedMemory(
            "value", [("2026-W24", "body")], [], [], []
        )
        assert _last_run_week_id(wm) == "2026-W24"

    def test_multiple_entries_returns_last(self) -> None:
        wm = WindowedMemory(
            "value",
            [("2026-W22", "b1"), ("2026-W23", "b2"), ("2026-W24", "b3")],
            [], [], [],
        )
        assert _last_run_week_id(wm) == "2026-W24"


# ---------------------------------------------------------------------------
# AC-3 — Round-trip with writer (7 personas)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """parse(write(x)) reproduces content for all 7 personas.

    The invariant: what writeback_memory wrote last week is what
    memory_reader reads back this week.  Uses existing M2 writer path
    (writeback_memory / _build_past_calls_entry) as the write half.
    """

    # Minimal stubs for writeback_memory call
    @dataclass
    class _Stance:
        persona: str
        ticker: str
        action: str
        confidence: int
        target_weight: float
        week_id: str

    @dataclass
    class _Round1Capture:
        stances: list
        narratives: dict
        counterfactuals: dict

    @dataclass
    class _ValidationResult:
        passed: bool
        notes: str

    @dataclass
    class _ReportPayload:
        summary: str

    @dataclass
    class _PersonaResearchResult:
        persona_slug: str
        week_id: str
        report_payload: Any
        validation: Any

    def _make_round1_for_persona(self, persona: str, week_id: str) -> "_Round1Capture":
        stances = [
            self._Stance(persona, "NVDA", "add", 5, 0.10, week_id),
            self._Stance(persona, "AAPL", "hold", 3, 0.08, week_id),
        ]
        return self._Round1Capture(
            stances=stances,
            narratives={persona: f"{persona} narrative for {week_id}"},
            counterfactuals={persona: {"NVDA": 0.10, "AAPL": 0.08, "CASH": 0.82}},
        )

    def _make_reports_for_persona(self, persona: str, week_id: str) -> list:
        return [
            self._PersonaResearchResult(
                persona_slug=persona,
                week_id=week_id,
                report_payload=self._ReportPayload(summary=f"Research for {persona} {week_id}"),
                validation=self._ValidationResult(passed=True, notes=""),
            )
        ]

    @pytest.mark.parametrize("persona", PERSONA_SLUGS_7)
    def test_round_trip_one_persona(self, tmp_path: Path, persona: str) -> None:
        """Round-trip cell for {persona}: write week n → read back at week n+1."""
        week_n = "2026-W24"
        write_capture = self._make_round1_for_persona(persona, week_n)
        reports = self._make_reports_for_persona(persona, week_n)

        # Write using the M2/Component 18 writer.
        writeback_memory(
            write_capture,
            write_capture.counterfactuals,
            reports,
            {},  # no resolved_alpha yet (same as real PoC state)
            memory_dir=tmp_path,
            archive_dir=tmp_path / "archive",
        )

        # Read back with memory_reader (week n+1 context; no resolved rows yet).
        conn = _make_db()
        result = read_persona_memory(persona, conn, memory_dir=tmp_path, config=_W8_CFG)
        wm = result.windowed_memory

        assert wm.persona == persona
        # One entry per section was written.
        assert len(wm.past_calls) == 1
        assert len(wm.counterfactual) == 1
        assert len(wm.debate_stances) == 1
        assert len(wm.whats_new) == 1

        # The week_id round-trips correctly.
        assert wm.past_calls[0][0] == week_n
        assert wm.counterfactual[0][0] == week_n

        # NVDA and AAPL appear in the past-calls body.
        past_body = wm.past_calls[0][1]
        assert "NVDA" in past_body
        assert "AAPL" in past_body
        assert "add" in past_body


# ---------------------------------------------------------------------------
# AC-4 — Read-only (no writes to file or ledger)
# ---------------------------------------------------------------------------

class TestReadOnly:
    """Component 26 never writes to memory files or the ledger."""

    def test_window_does_not_modify_file(self, tmp_path: Path) -> None:
        """Read-only cell 1: file contents unchanged after read_persona_memory."""
        pmf = _make_memory_file(10)
        from round_table_portfolio.orchestrator.memory import _render_memory_file
        content = _render_memory_file("value", pmf)
        mem_path = tmp_path / "value.md"
        mem_path.write_text(content, encoding="utf-8")
        original_content = mem_path.read_text(encoding="utf-8")
        original_mtime = mem_path.stat().st_mtime

        conn = _make_db()
        read_persona_memory("value", conn, memory_dir=tmp_path, config=_W3_CFG)

        assert mem_path.read_text(encoding="utf-8") == original_content
        assert mem_path.stat().st_mtime == original_mtime

    def test_no_new_files_created(self, tmp_path: Path) -> None:
        """Read-only cell 2: absent file stays absent after read_persona_memory."""
        conn = _make_db()
        before = set(tmp_path.iterdir())
        read_persona_memory("growth", conn, memory_dir=tmp_path, config=_W8_CFG)
        after = set(tmp_path.iterdir())
        assert after == before, "read_persona_memory must not create files"

    def test_no_ledger_writes(self) -> None:
        """Read-only cell 3: ledger row counts unchanged after read_persona_memory."""
        conn = _make_db()
        _seed_portfolio_with_returns(
            conn, "value", "2026-W23", "2026-W25", [("NVDA", "add", 0.10)]
        )
        before_rows = conn.execute("SELECT COUNT(*) FROM weekly_returns").fetchone()[0]
        before_port = conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            read_persona_memory(
                "value", conn, memory_dir=Path(tmp), config=_W8_CFG
            )

        after_rows = conn.execute("SELECT COUNT(*) FROM weekly_returns").fetchone()[0]
        after_port = conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]
        assert before_rows == after_rows
        assert before_port == after_port


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestMemoryReaderConfig:
    """load_memory_reader_config: fallback defaults + override."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Config reads memory_window_weeks correctly."""
        cfg_file = tmp_path / "thresholds.yaml"
        cfg_file.write_text("memory_window_weeks: 12\n", encoding="utf-8")
        cfg = load_memory_reader_config(cfg_file)
        assert cfg.memory_window_weeks == 12

    def test_default_when_key_absent(self, tmp_path: Path) -> None:
        """Config falls back to memory_window_weeks=8 when key absent."""
        cfg_file = tmp_path / "thresholds.yaml"
        cfg_file.write_text("max_position_weight: 0.20\n", encoding="utf-8")
        cfg = load_memory_reader_config(cfg_file)
        assert cfg.memory_window_weeks == 8

    def test_default_when_file_missing(self, tmp_path: Path) -> None:
        """Config returns default=8 when thresholds.yaml is absent."""
        cfg = load_memory_reader_config(tmp_path / "nonexistent.yaml")
        assert cfg.memory_window_weeks == 8

    def test_config_used_in_reader(self, tmp_path: Path) -> None:
        """Config window_weeks controls how many entries are injected."""
        pmf = _make_memory_file(6)
        from round_table_portfolio.orchestrator.memory import _render_memory_file
        content = _render_memory_file("value", pmf)
        (tmp_path / "value.md").write_text(content, encoding="utf-8")
        conn = _make_db()
        result = read_persona_memory(
            "value", conn, memory_dir=tmp_path, config=MemoryReaderConfig(memory_window_weeks=2)
        )
        assert len(result.windowed_memory.past_calls) == 2

    def test_window_weeks_no_literal_in_reader(self, tmp_path: Path) -> None:
        """Config-override test: window=1 (not hardcoded 8) keeps only 1 entry."""
        pmf = _make_memory_file(5)
        from round_table_portfolio.orchestrator.memory import _render_memory_file
        content = _render_memory_file("value", pmf)
        (tmp_path / "value.md").write_text(content, encoding="utf-8")
        conn = _make_db()
        result = read_persona_memory(
            "value", conn, memory_dir=tmp_path, config=_W1_CFG
        )
        assert len(result.windowed_memory.past_calls) == 1


# ---------------------------------------------------------------------------
# read_all_personas_memory — 7-persona sweep
# ---------------------------------------------------------------------------

class TestReadAllPersonas:
    """read_all_personas_memory returns a result for each of the 7 personas."""

    def test_all_seven_personas_returned(self, tmp_path: Path) -> None:
        """All 7 slugs are keys in the result dict."""
        conn = _make_db()
        results = read_all_personas_memory(conn, memory_dir=tmp_path, config=_W8_CFG)
        assert set(results.keys()) == set(PERSONA_SLUGS_7)

    def test_custom_persona_list(self, tmp_path: Path) -> None:
        """Passing a custom personas list returns only those slugs."""
        conn = _make_db()
        results = read_all_personas_memory(
            conn,
            personas=["value", "growth"],
            memory_dir=tmp_path,
            config=_W8_CFG,
        )
        assert set(results.keys()) == {"value", "growth"}

    def test_each_result_is_persona_memory_result(self, tmp_path: Path) -> None:
        """Each value in the result dict is a PersonaMemoryResult."""
        conn = _make_db()
        results = read_all_personas_memory(conn, memory_dir=tmp_path, config=_W8_CFG)
        for slug, res in results.items():
            assert isinstance(res, PersonaMemoryResult), f"{slug}: expected PersonaMemoryResult"
            assert res.windowed_memory.persona == slug


# ---------------------------------------------------------------------------
# Real-2026-W24 derived fixture (Gate-4 provenance corollary)
#
# Provenance: stances sourced from the real 2026-W24 live run on 2026-06-09.
#   Tickers, actions, and confidence values are real (from the live round-1
#   output); resolution week and alpha values are SYNTHETIC + deterministic.
#   No PII; no data committed under tests/fixtures/real/.
# ---------------------------------------------------------------------------

class TestRealW24Fixture:
    """Gate-4 provenance: real 2026-W24 cta-systematic-macro stances + synthetic resolution."""

    # Real 2026-W24 cta-systematic-macro stances (from the live run):
    _CTA_W24_STANCES = [
        ("DELL", "add", 5),
        ("FTNT", "add", 5),
        ("LLY",  "add", 5),
        ("NTAP", "add", 5),
        ("ELV",  "add", 4),
        ("CVS",  "add", 4),
        ("UNH",  "add", 4),
        ("MAR",  "add", 4),
        ("STX",  "add", 3),
        ("C",    "add", 3),
    ]

    # Synthetic deterministic portfolio-level alpha for resolution week 2026-W25.
    # The weekly_returns table stores ONE return per (portfolio_id, as_of_week_id),
    # not one per ticker.  Per-ticker alpha is not in the schema — the alpha column
    # is a portfolio-level mark.  We use a single positive value so tests can
    # assert presence of resolved rows without depending on per-ticker values.
    _CTA_W25_ALPHA: float = 0.07   # single portfolio-level alpha for W25

    def _seed_cta_w24_ledger(self, conn: sqlite3.Connection) -> None:
        """Seed real W24 cta portfolio + a single synthetic W25 portfolio return.

        All 10 tickers go into the same portfolio; weekly_returns has ONE row
        for that portfolio at as_of_week_id='2026-W25' (schema UNIQUE constraint).
        Each ticker in resolved_rows receives the same portfolio alpha — correct
        per the schema design (alpha is portfolio-level, not per-ticker).
        """
        # All tickers get the same portfolio-level alpha (schema stores per portfolio).
        tickers = [
            (t, a, self._CTA_W25_ALPHA)
            for t, a, _ in self._CTA_W24_STANCES
        ]
        _seed_portfolio_with_returns(
            conn, "cta-systematic-macro", "2026-W24", "2026-W25", tickers
        )

    def _write_cta_w24_memory(self, memory_dir: Path) -> None:
        """Write a single-week memory file for cta-systematic-macro using real stances."""
        body = _past_calls_body("2026-W24", self._CTA_W24_STANCES)
        content = (
            "# Persona Memory — cta-systematic-macro\n\n"
            "## Past Calls Log\n\n"
            f"### Entry 2026-W24\n{body}\n\n"
            "## Counterfactual Portfolio Log\n\n"
            "_No entries yet._\n\n"
            "## Debate Stances Log\n\n"
            "_No entries yet._\n\n"
            "## What's New Digest\n\n"
            "_No entries yet._\n"
        )
        (memory_dir / "cta-systematic-macro.md").write_text(content, encoding="utf-8")

    def test_real_w24_window_applies_correctly(self, tmp_path: Path) -> None:
        """Real W24 cell 1: 1 entry < W=8 → all 1 entry returned."""
        self._write_cta_w24_memory(tmp_path)
        conn = _make_db()
        self._seed_cta_w24_ledger(conn)
        result = read_persona_memory(
            "cta-systematic-macro", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        assert len(result.windowed_memory.past_calls) == 1
        assert result.windowed_memory.past_calls[0][0] == "2026-W24"

    def test_real_w24_resolved_rows_include_ntap(self, tmp_path: Path) -> None:
        """Real W24 cell 2: NTAP (best alpha=+0.12) in resolved_rows."""
        self._write_cta_w24_memory(tmp_path)
        conn = _make_db()
        self._seed_cta_w24_ledger(conn)
        # last_run = W24 → rows with as_of > W24 → W25 rows
        result = read_persona_memory(
            "cta-systematic-macro", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        tickers = {r.ticker for r in result.resolved_rows}
        assert "NTAP" in tickers

    def test_real_w24_resolved_rows_alpha_values_correct(self, tmp_path: Path) -> None:
        """Real W24 cell 3: all resolved rows carry the portfolio-level alpha (0.07).

        The weekly_returns table stores one alpha per (portfolio_id, as_of_week_id),
        not one per ticker.  Every ticker in the resolved rows for the same
        portfolio + as_of_week receives the same portfolio-level alpha value.
        """
        self._write_cta_w24_memory(tmp_path)
        conn = _make_db()
        self._seed_cta_w24_ledger(conn)
        result = read_persona_memory(
            "cta-systematic-macro", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        ntap_row = next(r for r in result.resolved_rows if r.ticker == "NTAP")
        assert abs(ntap_row.alpha - self._CTA_W25_ALPHA) < 1e-9

    def test_real_w24_resolved_alpha_map_ntap(self, tmp_path: Path) -> None:
        """Real W24 cell 4: resolved_alpha map contains NTAP → portfolio alpha."""
        self._write_cta_w24_memory(tmp_path)
        conn = _make_db()
        self._seed_cta_w24_ledger(conn)
        result = read_persona_memory(
            "cta-systematic-macro", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        assert "NTAP" in result.resolved_alpha
        assert abs(result.resolved_alpha["NTAP"] - self._CTA_W25_ALPHA) < 1e-9

    def test_real_w24_resolved_rows_all_10_tickers_present(self, tmp_path: Path) -> None:
        """Real W24 cell 5: all 10 real W24 tickers appear in resolved_rows.

        With own_misses context: the query surfaces every ticker in the portfolio
        that resolved; whether it is a 'miss' is determined downstream by the
        digest builder using the alpha sign.
        """
        self._write_cta_w24_memory(tmp_path)
        conn = _make_db()
        self._seed_cta_w24_ledger(conn)
        result = read_persona_memory(
            "cta-systematic-macro", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        expected_tickers = {t for t, _, _ in self._CTA_W24_STANCES}
        actual_tickers = {r.ticker for r in result.resolved_rows}
        assert expected_tickers == actual_tickers

    def test_real_w24_resolved_rows_correct_persona_field(self, tmp_path: Path) -> None:
        """Real W24 cell 6: all resolved rows carry persona='cta-systematic-macro'."""
        self._write_cta_w24_memory(tmp_path)
        conn = _make_db()
        self._seed_cta_w24_ledger(conn)
        result = read_persona_memory(
            "cta-systematic-macro", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        for row in result.resolved_rows:
            assert row.persona == "cta-systematic-macro", (
                f"Unexpected persona: {row.persona}"
            )

    def test_real_w24_resolved_rows_feed_digest_builder(self, tmp_path: Path) -> None:
        """Real W24 cell 7: Component 26 → Component 28 seam holds end-to-end.

        resolved_rows from memory_reader plug into build_whats_new_digest
        without conversion — the ResolvedRow type is shared (imported from
        digest.py, not redefined here).

        All 10 tickers carry the same portfolio-level alpha (0.07), so the
        digest tiebreaks by ticker ascending and caps at 5.  The seam test
        checks structural correctness (header present, real W24 tickers present,
        action attribution from past-calls body), not specific ordering.
        """
        from round_table_portfolio.orchestrator.digest import (
            DigestConfig,
            build_whats_new_digest,
        )

        self._write_cta_w24_memory(tmp_path)
        conn = _make_db()
        self._seed_cta_w24_ledger(conn)
        result = read_persona_memory(
            "cta-systematic-macro", conn, memory_dir=tmp_path, config=_W8_CFG
        )

        past_calls_entries = result.windowed_memory.past_calls
        cfg = DigestConfig(digest_max_items=5, own_misses_in_digest=True)

        digest = build_whats_new_digest(
            "cta-systematic-macro",
            result.resolved_rows,   # raw ResolvedRow sequence from Component 26
            past_calls_entries,
            cfg,
        )

        # The seam holds: header present, real W24 tickers present (at least
        # the ticker-ascending first 5: C, CVS, DELL, ELV, FTNT)
        assert "Since your last run" in digest
        assert "C " in digest or "C\n" in digest or "C)" in digest
        assert "you said add" in digest       # action attribution works
        assert "2026-W24" in digest           # call week attribution works
        assert f"+{self._CTA_W25_ALPHA:.4f}" in digest  # alpha formatted

    def test_real_w24_no_rows_excluded_by_wrong_last_run(self, tmp_path: Path) -> None:
        """Real W24 cell 8: 'since last run' predicate uses W24 (from memory file)
        so W25 returns are included; W24-or-earlier returns would be excluded."""
        self._write_cta_w24_memory(tmp_path)
        conn = _make_db()
        # Seed a W24 return (should be excluded — not strictly after W24)
        _seed_portfolio_with_returns(
            conn, "cta-systematic-macro", "2026-W23", "2026-W24",
            [("ORCL", "add", 0.20)]
        )
        # Seed a W25 return (should be included)
        _seed_portfolio_with_returns(
            conn, "cta-systematic-macro", "2026-W24", "2026-W25",
            [("NTAP", "add", 0.12)]
        )
        result = read_persona_memory(
            "cta-systematic-macro", conn, memory_dir=tmp_path, config=_W8_CFG
        )
        # last_run = W24 (from memory file with 1 entry for W24)
        # ORCL resolved at W24 → excluded; NTAP resolved at W25 → included
        tickers = {r.ticker for r in result.resolved_rows}
        assert "ORCL" not in tickers
        assert "NTAP" in tickers
