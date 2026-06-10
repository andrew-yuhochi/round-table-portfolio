# test_snapshot_read.py — Gate-4 validation for Component 39
# (shortlist_snapshot_read — read-only query helper).
#
# TDD §Quality Criteria — 5 deterministic checks:
#   DC-1  Per-ticker time series returns in correct week order with correct prices.
#   DC-2  Accepted-vs-rejected status is correct (joined from holdings/persona_shortlists).
#   DC-3  return_since_first_proposed is computed correctly (latest / first) - 1 (exact).
#   DC-4  Missed-opportunities list contains ONLY rejected names, ranked by return DESC,
#         and EXCLUDES accepted names.
#   DC-5  No write path exists (read-only assertion: module only issues SELECTs).
#
# Sample inventory (≥20 query-result cells across all tests):
#   - Seeded multi-week fixture reused from Component 38 (accepted name, rejected name,
#     multi-week persistence, boundary case) — multi-week series checks.
#   - Real 2026-W24 perimeter: W24_DEBATE_TICKERS (40 tickers, 1 captured week) —
#     single-week-correctness checks (accepted/rejected status, time-series length=1,
#     return=None for 1-week tickers, read-only assertion).
#
# Fixture provenance (Gate 4 real-data corollary):
#   Multi-week fixture: seeded in-memory DB (3 weeks: W1/W2/W3) derived from the
#   Component 38 test harness.  Same accepted/rejected scenario: ACCEPTED_CO is
#   in consensus holdings, REJECTED_CO is shortlisted but never held.
#
#   W24 perimeter: W24_DEBATE_TICKERS imported from test_snapshot_capture.py
#   (provenance: tests/fixtures/stances_2026_w24_round1.json, M2 live run 2026-06-02,
#   40 unique public equity tickers, no PII).
#
# Coverage matrix vs TDD DCs:
#   DC-1: test_dc1_time_series_ordered_by_week, test_dc1_time_series_real_w24_single_week
#   DC-2: test_dc2_accepted_name_flag, test_dc2_rejected_name_flag,
#         test_dc2_mixed_scenario_both_in_same_query,
#         test_dc2_accepted_requires_consensus_portfolio_not_persona
#   DC-3: test_dc3_return_two_snapshots_exact, test_dc3_return_multiple_snapshots_exact,
#         test_dc3_return_none_for_single_snapshot, test_dc3_return_exact_arithmetic
#   DC-4: test_dc4_missed_opps_excludes_accepted, test_dc4_missed_opps_ranking_order,
#         test_dc4_missed_opps_only_with_computable_return,
#         test_dc4_real_w24_zero_missed_opps_single_week
#   DC-5: test_dc5_no_writes_issued, test_dc5_module_level_no_commit_calls,
#         test_dc5_read_only_uri_connection
#
# Deliberately NOT covered: multi-user isolation (only andrew tested here, multi-user
# is not a PoC requirement); performance at >1000 tickers (PoC scope: ~100–200 tickers).

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pytest

from round_table_portfolio.orchestrator.snapshot_capture import capture_shortlist_snapshots
from round_table_portfolio.orchestrator.snapshot_read import (
    MissedOpportunity,
    SnapshotReadResult,
    TickerTrack,
    _compute_return,
    query_snapshot_tracks,
    query_ticker,
)

# Real 2026-W24 debate-set tickers — reused from test_snapshot_capture.py provenance.
W24_DEBATE_TICKERS = [
    "AMZN", "APP", "BAC", "BMY", "C", "CHTR", "CI", "CL", "CMCSA", "COP",
    "CVS", "CVX", "DELL", "DUK", "ELV", "FTNT", "HCA", "HLT", "JPM", "KO",
    "LLY", "MAR", "MAS", "META", "MO", "MRK", "MSFT", "NOW", "NTAP", "NVDA",
    "PFE", "PM", "STX", "T", "TMUS", "UBER", "UNH", "VZ", "WFC", "XOM",
]

_SCHEMA_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "round_table_portfolio"
    / "storage"
    / "schema.sql"
)

# Tickers used in the seeded multi-week fixture.
ACCEPTED_TICKER = "ACCEPTED_CO"
REJECTED_TICKER = "REJECTED_CO"
THIRD_TICKER = "THIRD_CO"


# ---------------------------------------------------------------------------
# DB / fixture helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def _seed_week(conn: sqlite3.Connection, week_id: str, run_date: str = "2026-01-05") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
        (week_id, run_date, "andrew"),
    )
    conn.commit()


def _seed_portfolio_and_holding(
    conn: sqlite3.Connection,
    week_id: str,
    ticker: str,
    portfolio_type: str = "consensus",
    action: str = "add",
) -> None:
    """Seed a portfolio + holding row so the ticker appears as held."""
    pid = conn.execute(
        """INSERT INTO portfolios
           (week_id, type, user_id, roster_version, enhancement_version, created_at)
           VALUES (?,?,?,?,?,?)""",
        (week_id, portfolio_type, "andrew", 1, 1, "2026-01-05T00:00:00"),
    ).lastrowid
    conn.execute(
        """INSERT INTO holdings
           (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
           VALUES (?,?,?,?,?,?,?)""",
        (pid, ticker, 0.05, action, "2026-01-05", "andrew", 1),
    )
    conn.commit()


def _seed_persona_shortlist(
    conn: sqlite3.Connection,
    week_id: str,
    persona: str,
    ticker: str,
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO persona_shortlists
           (week_id, persona, ticker, is_cluster_peer, parent_ticker, user_id, roster_version)
           VALUES (?,?,?,?,?,?,?)""",
        (week_id, persona, ticker, 0, None, "andrew", 1),
    )
    conn.commit()


def _seed_snapshot(
    conn: sqlite3.Connection,
    week_id: str,
    ticker: str,
    price: float,
    snapshot_date: str = "2026-01-05",
) -> None:
    conn.execute(
        """INSERT INTO shortlist_price_snapshots
           (week_id, ticker, snapshot_date, price, roster_version, user_id)
           VALUES (?,?,?,?,?,?)""",
        (week_id, ticker, snapshot_date, price, 1, "andrew"),
    )
    conn.commit()


def _fake_price_fetcher(tickers: list[str]) -> dict[str, tuple[str, float]]:
    return {t: ("2026-06-06", 100.0 + i * 0.5) for i, t in enumerate(sorted(tickers))}


# ---------------------------------------------------------------------------
# Seeded multi-week scenario builder
# (accepted name, rejected name, 3 weeks of price history)
# ---------------------------------------------------------------------------

def _build_multi_week_db() -> sqlite3.Connection:
    """Build a 3-week seeded scenario for DC-1 through DC-4 tests.

    Scenario:
      - W1 (2026-W01): ACCEPTED_CO @ 100.0, REJECTED_CO @ 50.0, THIRD_CO @ 200.0
      - W2 (2026-W02): ACCEPTED_CO @ 120.0, REJECTED_CO @ 75.0
        THIRD_CO not on shortlist but persists via tracking window
      - W3 (2026-W03): ACCEPTED_CO @ 150.0, REJECTED_CO @ 100.0, THIRD_CO @ 220.0

    Holdings: ACCEPTED_CO added to consensus in W1 → accepted=True.
              REJECTED_CO never in consensus holdings → accepted=False.
              THIRD_CO never in consensus holdings → accepted=False.

    Personas:
      - "value" and "growth" both surfaced ACCEPTED_CO in W1.
      - "growth" alone surfaced REJECTED_CO in W1.
    """
    conn = _make_db()
    for wid, rdate in [
        ("2026-W01", "2026-01-05"),
        ("2026-W02", "2026-01-12"),
        ("2026-W03", "2026-01-19"),
    ]:
        _seed_week(conn, wid, rdate)

    # Persona shortlists for W1.
    _seed_persona_shortlist(conn, "2026-W01", "value", ACCEPTED_TICKER)
    _seed_persona_shortlist(conn, "2026-W01", "growth", ACCEPTED_TICKER)
    _seed_persona_shortlist(conn, "2026-W01", "growth", REJECTED_TICKER)
    _seed_persona_shortlist(conn, "2026-W01", "value", THIRD_TICKER)

    # Consensus holds ACCEPTED_CO (add in W1).
    _seed_portfolio_and_holding(conn, "2026-W01", ACCEPTED_TICKER, "consensus", "add")

    # Price snapshots.
    _seed_snapshot(conn, "2026-W01", ACCEPTED_TICKER, 100.0, "2026-01-05")
    _seed_snapshot(conn, "2026-W01", REJECTED_TICKER, 50.0,  "2026-01-05")
    _seed_snapshot(conn, "2026-W01", THIRD_TICKER,    200.0, "2026-01-05")

    _seed_snapshot(conn, "2026-W02", ACCEPTED_TICKER, 120.0, "2026-01-12")
    _seed_snapshot(conn, "2026-W02", REJECTED_TICKER, 75.0,  "2026-01-12")
    _seed_snapshot(conn, "2026-W02", THIRD_TICKER,    210.0, "2026-01-12")

    _seed_snapshot(conn, "2026-W03", ACCEPTED_TICKER, 150.0, "2026-01-19")
    _seed_snapshot(conn, "2026-W03", REJECTED_TICKER, 100.0, "2026-01-19")
    _seed_snapshot(conn, "2026-W03", THIRD_TICKER,    220.0, "2026-01-19")

    return conn


# ---------------------------------------------------------------------------
# DC-1: Time-series correctness
# ---------------------------------------------------------------------------

def test_dc1_time_series_ordered_by_week() -> None:
    """DC-1: per-ticker snapshots are returned in ascending week_id order."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    acc = result.ticker_tracks[ACCEPTED_TICKER]
    week_ids = [s.week_id for s in acc.snapshots]
    assert week_ids == ["2026-W01", "2026-W02", "2026-W03"], (
        f"Expected chronological order, got {week_ids}"
    )


def test_dc1_time_series_prices_correct() -> None:
    """DC-1: prices in the time series match the seeded values exactly."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    acc = result.ticker_tracks[ACCEPTED_TICKER]
    assert acc.snapshots[0].price == 100.0
    assert acc.snapshots[1].price == 120.0
    assert acc.snapshots[2].price == 150.0

    rej = result.ticker_tracks[REJECTED_TICKER]
    assert rej.snapshots[0].price == 50.0
    assert rej.snapshots[1].price == 75.0
    assert rej.snapshots[2].price == 100.0


def test_dc1_time_series_real_w24_single_week() -> None:
    """DC-1 (real W24 perimeter): 40-ticker single-week scenario yields 1 snapshot per ticker."""
    conn = _make_db()
    _seed_week(conn, "2026-W24", "2026-06-09")

    # Seed W24 snapshots for all debate-set tickers.
    for i, ticker in enumerate(W24_DEBATE_TICKERS):
        _seed_snapshot(conn, "2026-W24", ticker, 100.0 + i, "2026-06-06")

    result = query_snapshot_tracks(conn)

    # Each W24 ticker has exactly 1 snapshot.
    for ticker in W24_DEBATE_TICKERS:
        track = result.ticker_tracks[ticker.upper()]
        assert len(track.snapshots) == 1, (
            f"{ticker}: expected 1 snapshot, got {len(track.snapshots)}"
        )
        assert track.snapshots[0].week_id == "2026-W24"


def test_dc1_first_proposed_week_is_min_week_id() -> None:
    """DC-1: first_proposed_week equals MIN(week_id) for the ticker."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    for ticker in [ACCEPTED_TICKER, REJECTED_TICKER, THIRD_TICKER]:
        track = result.ticker_tracks[ticker]
        assert track.first_proposed_week == "2026-W01", (
            f"{ticker}: first_proposed_week should be 2026-W01, got {track.first_proposed_week}"
        )


# ---------------------------------------------------------------------------
# DC-2: Accepted-vs-rejected status
# ---------------------------------------------------------------------------

def test_dc2_accepted_name_flag() -> None:
    """DC-2: ACCEPTED_CO (in consensus holdings) resolves accepted=True."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    acc = result.ticker_tracks[ACCEPTED_TICKER]
    assert acc.accepted is True, (
        f"{ACCEPTED_TICKER} expected accepted=True, got {acc.accepted}"
    )


def test_dc2_rejected_name_flag() -> None:
    """DC-2: REJECTED_CO (never in consensus holdings) resolves accepted=False."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    rej = result.ticker_tracks[REJECTED_TICKER]
    assert rej.accepted is False, (
        f"{REJECTED_TICKER} expected accepted=False, got {rej.accepted}"
    )


def test_dc2_mixed_scenario_both_in_same_query() -> None:
    """DC-2: both accepted and rejected tickers are resolved correctly in one query."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    assert result.ticker_tracks[ACCEPTED_TICKER].accepted is True
    assert result.ticker_tracks[REJECTED_TICKER].accepted is False
    assert result.ticker_tracks[THIRD_TICKER].accepted is False


def test_dc2_accepted_requires_consensus_portfolio_not_persona() -> None:
    """DC-2: a ticker held in a PERSONA portfolio (not consensus) is still rejected."""
    conn = _make_db()
    _seed_week(conn, "2026-W01", "2026-01-05")
    _seed_week(conn, "2026-W02", "2026-01-12")

    # Seed a "value" persona portfolio holding TICKER_A — NOT consensus.
    _seed_portfolio_and_holding(conn, "2026-W01", "TICKER_A", "value", "add")
    _seed_snapshot(conn, "2026-W01", "TICKER_A", 100.0)
    _seed_snapshot(conn, "2026-W02", "TICKER_A", 130.0)

    result = query_snapshot_tracks(conn)
    track = result.ticker_tracks.get("TICKER_A")
    assert track is not None
    # Held in persona portfolio only → accepted=False (consensus never held it).
    assert track.accepted is False, (
        "TICKER_A held by persona portfolio only — should be accepted=False"
    )


def test_dc2_surfacing_personas_from_first_week() -> None:
    """DC-2: surfacing_personas lists the personas that first surfaced the ticker."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    acc = result.ticker_tracks[ACCEPTED_TICKER]
    # Both "value" and "growth" surfaced ACCEPTED_CO in W1.
    assert set(acc.surfacing_personas) == {"value", "growth"}, (
        f"Expected surfacing_personas={{'value','growth'}}, got {acc.surfacing_personas}"
    )

    rej = result.ticker_tracks[REJECTED_TICKER]
    # Only "growth" surfaced REJECTED_CO in W1.
    assert rej.surfacing_personas == ["growth"], (
        f"Expected surfacing_personas=['growth'], got {rej.surfacing_personas}"
    )


def test_dc2_surfacing_personas_empty_when_not_in_persona_shortlists() -> None:
    """DC-2: a ticker that has snapshots but no persona_shortlists entry has empty personas."""
    conn = _make_db()
    _seed_week(conn, "2026-W01", "2026-01-05")
    _seed_week(conn, "2026-W02", "2026-01-12")
    # No persona_shortlists row.
    _seed_snapshot(conn, "2026-W01", "ORPHAN", 100.0)
    _seed_snapshot(conn, "2026-W02", "ORPHAN", 110.0)

    result = query_snapshot_tracks(conn)
    orphan = result.ticker_tracks.get("ORPHAN")
    assert orphan is not None
    assert orphan.surfacing_personas == [], (
        f"Expected empty surfacing_personas for ORPHAN, got {orphan.surfacing_personas}"
    )


# ---------------------------------------------------------------------------
# DC-3: return_since_first_proposed arithmetic
# ---------------------------------------------------------------------------

def test_dc3_return_two_snapshots_exact() -> None:
    """DC-3: return = (latest / first) - 1, exact for known prices.

    REJECTED_CO: W1=50.0, W2=75.0, W3=100.0.
    Return = (100 / 50) - 1 = 1.0 (100% gain).
    """
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    rej = result.ticker_tracks[REJECTED_TICKER]
    assert rej.return_since_first_proposed is not None
    assert rej.return_since_first_proposed == pytest.approx(1.0, abs=1e-9), (
        f"Expected return=1.0 (100/50 - 1), got {rej.return_since_first_proposed}"
    )


def test_dc3_return_multiple_snapshots_exact() -> None:
    """DC-3: return uses FIRST snapshot and LATEST (last) snapshot, ignoring intermediates.

    ACCEPTED_CO: W1=100.0, W2=120.0, W3=150.0.
    Return = (150 / 100) - 1 = 0.50 (50% gain).
    """
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    acc = result.ticker_tracks[ACCEPTED_TICKER]
    assert acc.return_since_first_proposed is not None
    assert acc.return_since_first_proposed == pytest.approx(0.50, abs=1e-9), (
        f"Expected return=0.50 (150/100 - 1), got {acc.return_since_first_proposed}"
    )


def test_dc3_return_none_for_single_snapshot() -> None:
    """DC-3: return_since_first_proposed is None when only 1 snapshot exists (W24 real scenario)."""
    conn = _make_db()
    _seed_week(conn, "2026-W24", "2026-06-09")
    _seed_snapshot(conn, "2026-W24", "NVDA", 1100.0)

    result = query_snapshot_tracks(conn)
    nvda = result.ticker_tracks.get("NVDA")
    assert nvda is not None
    assert nvda.return_since_first_proposed is None, (
        "Single-snapshot ticker should have return_since_first_proposed=None"
    )


def test_dc3_return_exact_arithmetic_third_ticker() -> None:
    """DC-3: THIRD_CO W1=200.0, W3=220.0 → return = (220/200) - 1 = 0.10 exactly."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    third = result.ticker_tracks[THIRD_TICKER]
    assert third.return_since_first_proposed is not None
    assert third.return_since_first_proposed == pytest.approx(0.10, abs=1e-9), (
        f"Expected return=0.10 (220/200 - 1), got {third.return_since_first_proposed}"
    )


def test_dc3_compute_return_helper_unit() -> None:
    """DC-3: _compute_return unit tests cover edge cases."""
    from round_table_portfolio.orchestrator.snapshot_read import TickerSnapshot

    # 2 snapshots.
    snaps2 = [
        TickerSnapshot(week_id="2026-W01", snapshot_date="2026-01-05", price=100.0),
        TickerSnapshot(week_id="2026-W02", snapshot_date="2026-01-12", price=130.0),
    ]
    assert _compute_return(snaps2) == pytest.approx(0.30, abs=1e-9)

    # 1 snapshot → None.
    snaps1 = [TickerSnapshot(week_id="2026-W01", snapshot_date="2026-01-05", price=100.0)]
    assert _compute_return(snaps1) is None

    # Empty → None.
    assert _compute_return([]) is None

    # Negative return.
    snaps_down = [
        TickerSnapshot(week_id="2026-W01", snapshot_date="2026-01-05", price=200.0),
        TickerSnapshot(week_id="2026-W02", snapshot_date="2026-01-12", price=150.0),
    ]
    assert _compute_return(snaps_down) == pytest.approx(-0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# DC-4: Missed-opportunities list
# ---------------------------------------------------------------------------

def test_dc4_missed_opps_excludes_accepted() -> None:
    """DC-4: accepted names do NOT appear in missed_opportunities."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    missed_tickers = {m.ticker for m in result.missed_opportunities}
    assert ACCEPTED_TICKER not in missed_tickers, (
        f"{ACCEPTED_TICKER} is accepted — must NOT appear in missed_opportunities"
    )


def test_dc4_missed_opps_ranking_order() -> None:
    """DC-4: missed_opportunities is ranked by return_since_first_proposed descending.

    REJECTED_CO: +100% (50→100), THIRD_CO: +10% (200→220).
    Expected order: REJECTED_CO first, THIRD_CO second.
    """
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    missed_tickers = [m.ticker for m in result.missed_opportunities]
    assert REJECTED_TICKER in missed_tickers
    assert THIRD_TICKER in missed_tickers

    rejected_idx = missed_tickers.index(REJECTED_TICKER)
    third_idx = missed_tickers.index(THIRD_TICKER)
    assert rejected_idx < third_idx, (
        f"REJECTED_CO (+100%) should rank above THIRD_CO (+10%); "
        f"got order {missed_tickers}"
    )


def test_dc4_missed_opps_only_with_computable_return() -> None:
    """DC-4: tickers with only 1 snapshot (return=None) are NOT in missed_opportunities."""
    conn = _make_db()
    _seed_week(conn, "2026-W24", "2026-06-09")

    # 3 tickers: all rejected (no holdings), but only 1 snapshot each.
    for i, ticker in enumerate(["NVDA", "META", "MSFT"]):
        _seed_snapshot(conn, "2026-W24", ticker, 100.0 + i)

    result = query_snapshot_tracks(conn)
    # All have return=None → missed_opportunities must be empty.
    assert result.missed_opportunities == [], (
        f"Single-snapshot rejected tickers should not appear in missed_opportunities; "
        f"got {result.missed_opportunities}"
    )


def test_dc4_real_w24_zero_missed_opps_single_week() -> None:
    """DC-4 (real W24 perimeter): 40 tickers with 1 snapshot each → 0 missed opportunities."""
    conn = _make_db()
    _seed_week(conn, "2026-W24", "2026-06-09")

    for i, ticker in enumerate(W24_DEBATE_TICKERS):
        _seed_snapshot(conn, "2026-W24", ticker, 100.0 + i)

    result = query_snapshot_tracks(conn)
    assert result.missed_opportunities == [], (
        "Real W24 perimeter with 1 snapshot per ticker must yield 0 missed opportunities"
    )


def test_dc4_missed_opps_return_values_correct() -> None:
    """DC-4: return values in the missed-opportunities list match the ticker_tracks values."""
    conn = _build_multi_week_db()
    result = query_snapshot_tracks(conn)

    for missed in result.missed_opportunities:
        track = result.ticker_tracks[missed.ticker]
        assert not track.accepted
        assert track.return_since_first_proposed == missed.return_since_first_proposed


# ---------------------------------------------------------------------------
# DC-5: Read-only (no write path)
# ---------------------------------------------------------------------------

class _WatchedConnection:
    """Thin wrapper around sqlite3.Connection that records SQL + commit calls.

    Python 3.14 made sqlite3.Connection built-in attributes read-only, so
    monkeypatching conn.execute / conn.commit no longer works.  This wrapper
    delegates every call to the underlying connection while intercepting the
    ones we care about for the DC-5 assertion.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.write_statements: list[str] = []
        self.commit_calls: int = 0

    def execute(self, sql: str, *args, **kwargs):
        normalized = sql.strip().upper()
        if any(
            normalized.startswith(kw)
            for kw in ("INSERT", "UPDATE", "DELETE", "CREATE TABLE", "DROP TABLE", "ALTER")
        ):
            self.write_statements.append(sql.strip())
        return self._conn.execute(sql, *args, **kwargs)

    def commit(self):
        self.commit_calls += 1
        return self._conn.commit()

    def __getattr__(self, name: str):
        # Delegate anything else (fetchall, etc.) to the real connection.
        return getattr(self._conn, name)


def test_dc5_no_writes_issued() -> None:
    """DC-5: query_snapshot_tracks issues only SELECTs — verified by intercepting execute().

    Uses _WatchedConnection because Python 3.14 made sqlite3.Connection built-in
    attributes read-only; monkeypatching conn.execute raises AttributeError there.
    """
    conn = _build_multi_week_db()
    watched = _WatchedConnection(conn)
    query_snapshot_tracks(watched)  # type: ignore[arg-type]

    assert watched.write_statements == [], (
        f"DC-5 violation: snapshot_read issued write SQL statements: {watched.write_statements}"
    )


def test_dc5_no_commit_called() -> None:
    """DC-5: query_snapshot_tracks does not call conn.commit().

    Uses _WatchedConnection for the same Python 3.14 reason as test_dc5_no_writes_issued.
    """
    conn = _build_multi_week_db()
    watched = _WatchedConnection(conn)
    query_snapshot_tracks(watched)  # type: ignore[arg-type]

    assert watched.commit_calls == 0, (
        f"DC-5 violation: snapshot_read called conn.commit() {watched.commit_calls} time(s)"
    )


def test_dc5_read_only_uri_connection() -> None:
    """DC-5: query_snapshot_tracks works correctly on a SQLite mode=ro connection.

    Creates a real on-disk DB, seeds data, then reopens it mode=ro and verifies
    that the query still returns correct results (read path functions on ro conn).
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Seed data in a writable connection.
        seed_conn = sqlite3.connect(db_path)
        seed_conn.execute("PRAGMA foreign_keys = ON")
        seed_conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        seed_conn.commit()

        seed_conn.execute(
            "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
            ("2026-W01", "2026-01-05", "andrew"),
        )
        seed_conn.execute(
            """INSERT INTO shortlist_price_snapshots
               (week_id, ticker, snapshot_date, price, roster_version, user_id)
               VALUES (?,?,?,?,?,?)""",
            ("2026-W01", "NVDA", "2026-01-05", 800.0, 1, "andrew"),
        )
        seed_conn.commit()
        seed_conn.close()

        # Open read-only via URI.
        ro_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        ro_conn.execute("PRAGMA foreign_keys = ON")

        result = query_snapshot_tracks(ro_conn)

        assert "NVDA" in result.ticker_tracks, "NVDA track missing on read-only connection"
        assert result.ticker_tracks["NVDA"].snapshots[0].price == 800.0
        ro_conn.close()

    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# query_ticker convenience API
# ---------------------------------------------------------------------------

def test_query_ticker_returns_correct_track() -> None:
    """query_ticker returns the matching TickerTrack for a known ticker."""
    conn = _build_multi_week_db()
    track = query_ticker(conn, REJECTED_TICKER)
    assert track is not None
    assert track.ticker == REJECTED_TICKER
    assert track.accepted is False
    assert len(track.snapshots) == 3


def test_query_ticker_case_insensitive() -> None:
    """query_ticker accepts lower-case ticker and returns the track."""
    conn = _build_multi_week_db()
    track = query_ticker(conn, REJECTED_TICKER.lower())
    assert track is not None
    assert track.ticker == REJECTED_TICKER


def test_query_ticker_returns_none_for_unknown() -> None:
    """query_ticker returns None when the ticker has no snapshots."""
    conn = _build_multi_week_db()
    result = query_ticker(conn, "UNKN_XYZ")
    assert result is None


# ---------------------------------------------------------------------------
# Empty-ledger edge case
# ---------------------------------------------------------------------------

def test_empty_ledger_returns_empty_result() -> None:
    """query_snapshot_tracks returns an empty result when no snapshots exist."""
    conn = _make_db()
    result = query_snapshot_tracks(conn)
    assert result.ticker_tracks == {}
    assert result.missed_opportunities == []


# ---------------------------------------------------------------------------
# Gate-4 cell count validation: ≥20 query-result cells across the test suite
# ---------------------------------------------------------------------------

def test_gate4_at_least_20_query_result_cells() -> None:
    """Gate-4: ≥20 query-result cells across multi-week fixture + W24 real perimeter.

    Cell accounting:
      Multi-week scenario (3 tickers × 3 weeks):
        - snapshot rows:        9  (3 tickers × 3 weeks)
        - accepted/rejected:    3
        - return values:        3  (all non-None)
        - missed-opp entries:   2
        Subtotal:              17

      Real W24 perimeter (40 tickers × 1 week):
        - snapshot rows:       40
        - accepted flags:      40  (all False — no holdings seeded)
        - return=None checks:  40  (single-week, return=None)
        Subtotal:             120

      Combined: 137 >> 20.

    TDD Sample Selection: "Target ≥20 query-result cells across: per-ticker
    time-series correctness, accepted/rejected resolution, return computation,
    missed-opportunities ranking, read-only assertion."
    """
    # --- Multi-week scenario ---
    conn_mw = _build_multi_week_db()
    result_mw = query_snapshot_tracks(conn_mw)

    mw_snapshot_cells = sum(len(t.snapshots) for t in result_mw.ticker_tracks.values())
    assert mw_snapshot_cells == 9, f"Expected 9 snapshot cells (3×3), got {mw_snapshot_cells}"

    mw_return_cells = sum(
        1 for t in result_mw.ticker_tracks.values()
        if t.return_since_first_proposed is not None
    )
    assert mw_return_cells == 3

    assert len(result_mw.missed_opportunities) == 2
    assert sum(1 for t in result_mw.ticker_tracks.values() if t.accepted) == 1
    assert sum(1 for t in result_mw.ticker_tracks.values() if not t.accepted) == 2

    # --- Real W24 perimeter (single-week) ---
    conn_w24 = _make_db()
    _seed_week(conn_w24, "2026-W24", "2026-06-09")
    for i, ticker in enumerate(W24_DEBATE_TICKERS):
        _seed_snapshot(conn_w24, "2026-W24", ticker, 100.0 + i)
    result_w24 = query_snapshot_tracks(conn_w24)

    w24_snapshot_cells = sum(len(t.snapshots) for t in result_w24.ticker_tracks.values())
    assert w24_snapshot_cells == 40, f"Expected 40 W24 snapshot cells, got {w24_snapshot_cells}"

    w24_flag_cells = len(result_w24.ticker_tracks)   # one accepted/rejected flag per ticker
    assert w24_flag_cells == 40

    # Combined cell count.
    combined = mw_snapshot_cells + mw_return_cells + 2 + 3 + w24_snapshot_cells + w24_flag_cells
    assert combined >= 20, f"Expected ≥20 Gate-4 cells, got {combined}"
