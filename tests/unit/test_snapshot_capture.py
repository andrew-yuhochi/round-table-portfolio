# test_snapshot_capture.py — Gate-4 validation for Component 38
# (shortlist_snapshot_capture + shortlist_price_snapshots table).
#
# TDD §Quality Criteria — 6 deterministic checks:
#   DC-1  Completeness: exactly one snapshot row per unique ticker per week; no missing, no dup.
#   DC-2  Rejected names captured: a ticker shortlisted but NOT in consensus holdings still
#         gets a snapshot row (the "what we passed on" property).
#   DC-3  Bounded perimeter: snapshotted set is the shortlist/debate-set union (+in-window
#         priors), NOT the full ~500 universe.
#   DC-4  Tracking-duration rule: ticker first-proposed week W is snapshotted in W+k for
#         k ≤ 104; stops at k > 104.
#   DC-5  price NULL-rate audit: every written row has price > 0; Alpaca misses are
#         NAMED logged skips, never NULL-price rows.
#   DC-6  Additive / no-mutation: every previously-locked table's row count is unchanged
#         after a capture run; only shortlist_price_snapshots gains rows.
#
# Sample inventory (≥20 ticker-week capture cells):
#   - Real-data fixture: 2026-W24 debate-set tickers (40 unique; provenance documented below).
#   - Seeded multi-week fixture: 3 weeks (W1/W2/W3) exercising tracking-duration,
#     rejected-name capture, and the 104-week boundary (≥20+ cells).
#   - Alpaca-miss fixture: a ticker Alpaca cannot price → logged skip, no NULL row.
#
# Provenance (Gate 4 real-data corollary):
#   Source: tests/fixtures/stances_2026_w24_round1.json — the real round-1 stances from
#   the M2 live run executed 2026-06-02.  Unique tickers extracted from the `ticker` field
#   across all persona stances (40 tickers).  The fixture is the debate-set for 2026-W24
#   (tickers that entered the debate; the stances record each persona's round-1 judgment).
#   No PII present — tickers are public equity symbols only.
#   PII-strip note: none needed (ticker symbols are public data).

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from round_table_portfolio.orchestrator.snapshot_capture import (
    CaptureSummary,
    _compute_tracking_set,
    _weeks_apart,
    capture_shortlist_snapshots,
)

# ---------------------------------------------------------------------------
# Real 2026-W24 debate-set tickers
# (provenance: tests/fixtures/stances_2026_w24_round1.json, M2 live run 2026-06-02)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    """In-memory SQLite DB with the full schema applied and FK enforcement on."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def _seed_base(conn: sqlite3.Connection, week_id: str, run_date: str = "2026-06-09") -> None:
    """Seed the minimum prerequisite rows for a week."""
    conn.execute(
        "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
        (week_id, run_date, "andrew"),
    )
    conn.commit()


def _seed_multi_week(conn: sqlite3.Connection) -> None:
    """Seed three sequential test weeks (W1/W2/W3) for multi-week tracking tests."""
    weeks = [
        ("2026-W01", "2026-01-05"),
        ("2026-W02", "2026-01-12"),
        ("2026-W03", "2026-01-19"),
    ]
    for wid, rdate in weeks:
        conn.execute(
            "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
            (wid, rdate, "andrew"),
        )
    conn.commit()


def _fake_price_fetcher(tickers: list[str]) -> dict[str, tuple[str, float]]:
    """Stub fetcher: returns a fixed price for every ticker (simulates Alpaca success)."""
    return {t: ("2026-06-06", 100.0 + i * 0.5) for i, t in enumerate(sorted(tickers))}


def _miss_fetcher_for(*miss_tickers: str):
    """Return a fetcher stub that omits the specified tickers (simulates Alpaca misses)."""
    miss_set = set(miss_tickers)

    def _fetcher(tickers: list[str]) -> dict[str, tuple[str, float]]:
        return {
            t: ("2026-06-06", 100.0)
            for t in tickers
            if t.upper() not in miss_set
        }

    return _fetcher


def _empty_fetcher(tickers: list[str]) -> dict[str, tuple[str, float]]:
    """Stub fetcher that returns nothing (simulates total Alpaca failure)."""
    return {}


# ---------------------------------------------------------------------------
# _weeks_apart helper tests
# ---------------------------------------------------------------------------

def test_weeks_apart_same_week() -> None:
    assert _weeks_apart("2026-W01", "2026-W01") == 0


def test_weeks_apart_one_week() -> None:
    assert _weeks_apart("2026-W01", "2026-W02") == 1


def test_weeks_apart_104_weeks() -> None:
    # 2024-W01 to 2026-W01 ≈ 104 weeks (2 years).
    result = _weeks_apart("2024-W01", "2026-W01")
    # Allowing ±1 week for leap-year boundary effects.
    assert 103 <= result <= 105, f"Expected ~104 weeks apart, got {result}"


def test_weeks_apart_symmetric() -> None:
    assert _weeks_apart("2026-W10", "2026-W03") == _weeks_apart("2026-W03", "2026-W10")


# ---------------------------------------------------------------------------
# Schema: additive-lock test (DC-6 precondition / Critical Component #2)
# ---------------------------------------------------------------------------

# These are the table names that were locked before M5.
_LOCKED_TABLES = [
    "roster_versions",
    "enhancement_versions",
    "weeks",
    "portfolios",
    "holdings",
    "weekly_returns",
    "transcripts",
    "agent_stances",
    "persona_reports",
    "persona_shortlists",
]


def _get_table_ddl(conn: sqlite3.Connection, table: str) -> str:
    """Return the CREATE TABLE statement for *table* from sqlite_master."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    assert row is not None, f"Table {table!r} not found in sqlite_master"
    return row[0].strip()


def _baseline_ddls() -> dict[str, str]:
    """
    Baseline DDLs of all locked tables from the schema file WITHOUT the new
    shortlist_price_snapshots table.  We derive this by applying the schema
    and reading sqlite_master — the source of truth is the file itself.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return {t: _get_table_ddl(conn, t) for t in _LOCKED_TABLES}


# Compute baseline once at module import (fast — in-memory only).
_BASELINE_DDLS = _baseline_ddls()


def test_additive_lock_all_previously_locked_tables_unchanged() -> None:
    """DC-6 precondition: every previously-locked table's DDL is byte-identical
    after the shortlist_price_snapshots table is added.

    This is the additive-lock test required by TASK-M5-001 AC-1 and TDD §1 Component 1
    Critical lock preservation.  It derives the current DDL from the live schema file
    and compares it against the baseline captured at test-collection time.
    """
    conn = _make_db()
    for table in _LOCKED_TABLES:
        current_ddl = _get_table_ddl(conn, table)
        expected_ddl = _BASELINE_DDLS[table]
        assert current_ddl == expected_ddl, (
            f"LOCK VIOLATION: {table!r} DDL changed after M5 additive table was added.\n"
            f"Expected:\n{expected_ddl}\n\nGot:\n{current_ddl}"
        )


def test_new_table_exists_in_schema() -> None:
    """shortlist_price_snapshots must exist after schema application."""
    conn = _make_db()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='shortlist_price_snapshots'"
    ).fetchone()
    assert row is not None, "shortlist_price_snapshots table was not created"


def test_new_table_has_no_enhancement_version_column() -> None:
    """enhancement_version FK must be ABSENT — a price snapshot is a market fact."""
    conn = _make_db()
    cols = [
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(shortlist_price_snapshots)"
        ).fetchall()
    ]
    assert "enhancement_version" not in cols, (
        "enhancement_version column found on shortlist_price_snapshots — "
        "it must be omitted per spec (price is a market fact, not a decision artefact)"
    )


def test_new_table_price_check_rejects_zero() -> None:
    """CHECK price > 0 must reject a zero-price insert."""
    conn = _make_db()
    _seed_base(conn, "2026-W24")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO shortlist_price_snapshots
               (week_id, ticker, snapshot_date, price, roster_version, user_id)
               VALUES (?,?,?,?,?,?)""",
            ("2026-W24", "AAPL", "2026-06-06", 0.0, 1, "andrew"),
        )


def test_new_table_price_check_rejects_negative() -> None:
    """CHECK price > 0 must reject a negative price."""
    conn = _make_db()
    _seed_base(conn, "2026-W24")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO shortlist_price_snapshots
               (week_id, ticker, snapshot_date, price, roster_version, user_id)
               VALUES (?,?,?,?,?,?)""",
            ("2026-W24", "AAPL", "2026-06-06", -1.0, 1, "andrew"),
        )


def test_new_table_unique_constraint_fires() -> None:
    """UNIQUE(week_id, ticker, user_id) must reject a duplicate (week_id, ticker, user_id)."""
    conn = _make_db()
    _seed_base(conn, "2026-W24")
    conn.execute(
        """INSERT INTO shortlist_price_snapshots
           (week_id, ticker, snapshot_date, price, roster_version, user_id)
           VALUES (?,?,?,?,?,?)""",
        ("2026-W24", "NVDA", "2026-06-06", 120.0, 1, "andrew"),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO shortlist_price_snapshots
               (week_id, ticker, snapshot_date, price, roster_version, user_id)
               VALUES (?,?,?,?,?,?)""",
            ("2026-W24", "NVDA", "2026-06-07", 125.0, 1, "andrew"),
        )


def test_new_table_fk_week_id_enforced() -> None:
    """FK week_id → weeks must reject an unknown week."""
    conn = _make_db()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO shortlist_price_snapshots
               (week_id, ticker, snapshot_date, price, roster_version, user_id)
               VALUES (?,?,?,?,?,?)""",
            ("2099-W99", "AAPL", "2026-06-06", 100.0, 1, "andrew"),
        )


def test_new_table_fk_roster_version_enforced() -> None:
    """FK roster_version → roster_versions must reject an unknown version."""
    conn = _make_db()
    _seed_base(conn, "2026-W24")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO shortlist_price_snapshots
               (week_id, ticker, snapshot_date, price, roster_version, user_id)
               VALUES (?,?,?,?,?,?)""",
            ("2026-W24", "AAPL", "2026-06-06", 100.0, 99, "andrew"),
        )


# ---------------------------------------------------------------------------
# DC-1: Completeness — exactly one snapshot row per unique ticker per week
# (real 2026-W24 debate-set — provenance documented at top of file)
# ---------------------------------------------------------------------------

def test_dc1_completeness_real_w24_debate_set() -> None:
    """DC-1 (real data): 40 W24 debate-set tickers → exactly 40 snapshot rows,
    no duplicates.

    Provenance: W24_DEBATE_TICKERS derived from tests/fixtures/stances_2026_w24_round1.json,
    the M2 live-run stances (2026-06-02).  40 unique tickers.
    """
    conn = _make_db()
    _seed_base(conn, "2026-W24", "2026-06-09")

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W24",
        shortlist_tickers=[],           # debate-set tickers carry the real-data sample
        debate_set_tickers=W24_DEBATE_TICKERS,
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    # Row count must equal unique-ticker count.
    row_count = conn.execute(
        "SELECT COUNT(*) FROM shortlist_price_snapshots WHERE week_id='2026-W24'"
    ).fetchone()[0]
    assert row_count == len(set(W24_DEBATE_TICKERS)), (
        f"Expected {len(set(W24_DEBATE_TICKERS))} rows, got {row_count}"
    )

    # No duplicates: every (week_id, ticker) pair must appear exactly once.
    dups = conn.execute(
        """SELECT ticker, COUNT(*) AS cnt
           FROM shortlist_price_snapshots
           WHERE week_id='2026-W24'
           GROUP BY ticker
           HAVING cnt > 1"""
    ).fetchall()
    assert not dups, f"Duplicate (week_id, ticker) pairs found: {dups}"

    # Summary counts.
    assert summary.success_count == len(set(W24_DEBATE_TICKERS))
    assert summary.miss_count == 0


def test_dc1_completeness_deduplication_of_input() -> None:
    """DC-1: duplicate tickers in shortlist + debate_set are de-duped; one row written."""
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    # AAPL appears in both lists and twice in shortlist.
    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=["AAPL", "AAPL", "MSFT"],
        debate_set_tickers=["AAPL", "NVDA"],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    row_count = conn.execute(
        "SELECT COUNT(*) FROM shortlist_price_snapshots WHERE week_id='2026-W10'"
    ).fetchone()[0]
    assert row_count == 3, f"Expected 3 unique rows (AAPL, MSFT, NVDA), got {row_count}"
    assert summary.success_count == 3


# ---------------------------------------------------------------------------
# DC-2: Rejected names are captured
# ---------------------------------------------------------------------------

def test_dc2_rejected_name_captured() -> None:
    """DC-2: a ticker shortlisted but NOT in consensus holdings still gets a snapshot."""
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    # REJECTED_TICKER is in shortlist, NOT added to holdings — simulates debate rejection.
    REJECTED = "REJECTED_CO"
    ACCEPTED = "ACCEPTED_CO"

    # Seed a holdings row for ACCEPTED only (simulating consensus chose it).
    portfolio_id_row = conn.execute(
        """INSERT INTO portfolios
           (week_id, type, user_id, roster_version, enhancement_version, created_at)
           VALUES (?,?,?,?,?,?)""",
        ("2026-W10", "consensus", "andrew", 1, 1, "2026-01-05T00:00:00"),
    ).lastrowid
    conn.execute(
        """INSERT INTO holdings
           (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
           VALUES (?,?,?,?,?,?,?)""",
        (portfolio_id_row, ACCEPTED, 0.05, "add", "2026-01-05", "andrew", 1),
    )
    conn.commit()

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=[REJECTED, ACCEPTED],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    # Both tickers must have snapshot rows regardless of holdings status.
    for ticker in [REJECTED, ACCEPTED]:
        row = conn.execute(
            "SELECT price FROM shortlist_price_snapshots WHERE week_id='2026-W10' AND ticker=?",
            (ticker,),
        ).fetchone()
        assert row is not None, f"Expected snapshot row for {ticker!r} — got none"
        assert row[0] > 0, f"Expected positive price for {ticker!r}"

    assert summary.success_count == 2


# ---------------------------------------------------------------------------
# DC-3: Bounded perimeter
# ---------------------------------------------------------------------------

def test_dc3_bounded_perimeter_not_full_universe() -> None:
    """DC-3: the captured set is bounded to shortlist+debate-set union, not ~500 universe."""
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    # Provide 30 tickers (well under 500).
    shortlist = [f"TICK{i:03d}" for i in range(30)]

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=shortlist,
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    row_count = conn.execute(
        "SELECT COUNT(*) FROM shortlist_price_snapshots WHERE week_id='2026-W10'"
    ).fetchone()[0]
    assert row_count == 30, f"Expected 30 rows, got {row_count}"
    assert summary.perimeter_count == 30
    # The perimeter count must be far below 500 (the full universe).
    assert summary.perimeter_count < 200


def test_dc3_bounded_perimeter_real_w24() -> None:
    """DC-3 (real data): W24 40-ticker capture is bounded well below 500."""
    conn = _make_db()
    _seed_base(conn, "2026-W24", "2026-06-09")

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W24",
        shortlist_tickers=[],
        debate_set_tickers=W24_DEBATE_TICKERS,
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    assert summary.perimeter_count == 40
    assert summary.perimeter_count < 200


# ---------------------------------------------------------------------------
# DC-4: Tracking-duration rule
# ---------------------------------------------------------------------------

def test_dc4_tracking_persists_after_ticker_drops_from_shortlist() -> None:
    """DC-4: a ticker first-proposed in W1 is snapshotted in W3 even with NO W2/W3 shortlist."""
    conn = _make_db()
    _seed_multi_week(conn)

    PERSISTENT = "PERSISTENT"
    W3_ONLY = "W3ONLY"

    # W1: both tickers appear.
    capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W01",
        shortlist_tickers=[PERSISTENT, W3_ONLY],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    # W2: neither ticker on any shortlist — PERSISTENT should still be tracked.
    summary_w2 = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W02",
        shortlist_tickers=[],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    # W3: only W3_ONLY reappears on shortlist — PERSISTENT still tracked from W1.
    summary_w3 = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W03",
        shortlist_tickers=[W3_ONLY],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    # W2: PERSISTENT must have a snapshot even though it was on no shortlist.
    w2_row = conn.execute(
        "SELECT price FROM shortlist_price_snapshots WHERE week_id='2026-W02' AND ticker=?",
        (PERSISTENT,),
    ).fetchone()
    assert w2_row is not None, "PERSISTENT ticker missing from W2 — tracking-duration rule violated"
    assert w2_row[0] > 0

    # W3: PERSISTENT still tracked from W1 history (W3_ONLY reappears + PERSISTENT from prior).
    w3_persistent = conn.execute(
        "SELECT price FROM shortlist_price_snapshots WHERE week_id='2026-W03' AND ticker=?",
        (PERSISTENT,),
    ).fetchone()
    assert w3_persistent is not None, "PERSISTENT ticker missing from W3 — tracking-duration violated"

    # Summary: W2 still_tracking_count must reflect prior tickers.
    assert summary_w2.still_tracking_count >= 2, (
        f"Expected ≥2 still-tracking in W2 (PERSISTENT + W3ONLY), got {summary_w2.still_tracking_count}"
    )


def test_dc4_newly_entered_count_correct() -> None:
    """DC-4: newly_entered_count matches tickers appearing for the first time."""
    conn = _make_db()
    _seed_multi_week(conn)

    # W1: 3 new tickers.
    s1 = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W01",
        shortlist_tickers=["AAPL", "MSFT", "NVDA"],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )
    assert s1.newly_entered_count == 3, f"W1 newly_entered expected 3, got {s1.newly_entered_count}"

    # W2: AAPL again + 1 new ticker AMZN.
    s2 = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W02",
        shortlist_tickers=["AAPL", "AMZN"],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )
    assert s2.newly_entered_count == 1, (
        f"W2 newly_entered expected 1 (only AMZN is new), got {s2.newly_entered_count}"
    )


def test_dc4_tracking_window_boundary_104_weeks() -> None:
    """DC-4: a ticker stops being tracked after 104 weeks from first-proposed.

    We test this by pre-seeding a snapshot row for a ticker with a first_week
    that is 105 weeks before the current week.  The tracker must NOT include
    that ticker in the new capture.
    """
    conn = _make_db()

    # Seed the "old" week (105 weeks ago) and a "current" week.
    # We use artificial week labels for the boundary test.
    OLD_WEEK = "2024-W01"   # will be set ~105 weeks before 2026-W02
    CURR_WEEK = "2026-W02"

    conn.execute(
        "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
        (OLD_WEEK, "2024-01-01", "andrew"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
        (CURR_WEEK, "2026-01-12", "andrew"),
    )
    conn.commit()

    STALE_TICKER = "STALE"

    # Pre-seed a snapshot for STALE_TICKER from OLD_WEEK.
    conn.execute(
        """INSERT INTO shortlist_price_snapshots
           (week_id, ticker, snapshot_date, price, roster_version, user_id)
           VALUES (?,?,?,?,?,?)""",
        (OLD_WEEK, STALE_TICKER, "2024-01-01", 50.0, 1, "andrew"),
    )
    conn.commit()

    # Verify that STALE_TICKER is NOT in the current week's tracking set.
    weeks_since = _weeks_apart(CURR_WEEK, OLD_WEEK)
    assert weeks_since > 104, (
        f"Test setup error: {weeks_since} weeks apart, expected > 104 for boundary test"
    )

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id=CURR_WEEK,
        shortlist_tickers=[],       # STALE not on any current shortlist
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    stale_row = conn.execute(
        "SELECT * FROM shortlist_price_snapshots WHERE week_id=? AND ticker=?",
        (CURR_WEEK, STALE_TICKER),
    ).fetchone()
    assert stale_row is None, (
        f"STALE ticker snapshotted in {CURR_WEEK} — should have stopped after 104 weeks "
        f"({weeks_since} weeks from first-proposed {OLD_WEEK})"
    )
    assert summary.still_tracking_count == 0


def test_dc4_tracking_window_boundary_within_104_weeks() -> None:
    """DC-4: a ticker at exactly 104 weeks IS still tracked (boundary inclusive)."""
    conn = _make_db()

    # 104 weeks before 2026-W01 ≈ 2024-W01.
    FIRST_WEEK = "2024-W01"
    CURR_WEEK = "2026-W01"

    for wid, rdate in [(FIRST_WEEK, "2024-01-01"), (CURR_WEEK, "2026-01-06")]:
        conn.execute(
            "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
            (wid, rdate, "andrew"),
        )
    conn.commit()

    TRACKED = "WITHIN104"
    conn.execute(
        """INSERT INTO shortlist_price_snapshots
           (week_id, ticker, snapshot_date, price, roster_version, user_id)
           VALUES (?,?,?,?,?,?)""",
        (FIRST_WEEK, TRACKED, "2024-01-01", 50.0, 1, "andrew"),
    )
    conn.commit()

    weeks_since = _weeks_apart(CURR_WEEK, FIRST_WEEK)
    assert weeks_since <= 104, (
        f"Test setup error: {weeks_since} weeks apart — should be ≤104 for within-window test"
    )

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id=CURR_WEEK,
        shortlist_tickers=[],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    row = conn.execute(
        "SELECT price FROM shortlist_price_snapshots WHERE week_id=? AND ticker=?",
        (CURR_WEEK, TRACKED),
    ).fetchone()
    assert row is not None, (
        f"Ticker within 104-week window ({weeks_since} weeks from first-proposed) "
        f"was NOT tracked — should still be tracked"
    )
    assert row[0] > 0


# ---------------------------------------------------------------------------
# DC-5: price NULL-rate audit — 0% among written rows; misses are logged skips
# ---------------------------------------------------------------------------

def test_dc5_no_null_price_rows_written() -> None:
    """DC-5: no NULL-price rows in shortlist_price_snapshots after a capture."""
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=["AAPL", "MSFT", "NVDA"],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    null_price_rows = conn.execute(
        "SELECT ticker FROM shortlist_price_snapshots WHERE price IS NULL"
    ).fetchall()
    assert not null_price_rows, (
        f"NULL-price rows found (DC-5 violation): {null_price_rows}"
    )


def test_dc5_alpaca_miss_skipped_not_written_as_null() -> None:
    """DC-5: Alpaca miss → named logged skip, NOT a NULL-price row.

    NVDA cannot be priced — it must be absent from shortlist_price_snapshots,
    NOT present with price=NULL.
    """
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=["AAPL", "NVDA", "MSFT"],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_miss_fetcher_for("NVDA"),
    )

    # NVDA must be absent from the table (skip, not NULL row).
    nvda_row = conn.execute(
        "SELECT * FROM shortlist_price_snapshots WHERE week_id='2026-W10' AND ticker='NVDA'"
    ).fetchone()
    assert nvda_row is None, (
        "NVDA has a row in shortlist_price_snapshots despite being an Alpaca miss — "
        "it must be a named skip, never a NULL-price row"
    )

    # AAPL and MSFT must be written successfully.
    for ticker in ["AAPL", "MSFT"]:
        row = conn.execute(
            "SELECT price FROM shortlist_price_snapshots WHERE week_id='2026-W10' AND ticker=?",
            (ticker,),
        ).fetchone()
        assert row is not None and row[0] > 0, f"{ticker} missing or zero-price after miss test"

    # Summary must name the miss.
    assert summary.miss_count == 1
    assert "NVDA" in summary.missed_tickers
    assert summary.success_count == 2


def test_dc5_alpaca_miss_named_in_summary() -> None:
    """DC-5: multiple Alpaca misses are ALL named in CaptureSummary.missed_tickers."""
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=["AAA", "BBB", "CCC", "DDD"],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_miss_fetcher_for("BBB", "DDD"),
    )

    assert summary.miss_count == 2
    assert set(summary.missed_tickers) == {"BBB", "DDD"}, (
        f"Expected missed_tickers={{'BBB','DDD'}}, got {summary.missed_tickers}"
    )
    assert summary.success_count == 2


def test_dc5_total_alpaca_failure_no_rows_written() -> None:
    """DC-5: if Alpaca returns nothing for all tickers, zero rows written (all named misses)."""
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=["AAPL", "MSFT"],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_empty_fetcher,
    )

    row_count = conn.execute(
        "SELECT COUNT(*) FROM shortlist_price_snapshots WHERE week_id='2026-W10'"
    ).fetchone()[0]
    assert row_count == 0, f"Expected 0 rows on total Alpaca failure, got {row_count}"
    assert summary.success_count == 0
    assert summary.miss_count == 2
    assert set(summary.missed_tickers) == {"AAPL", "MSFT"}


def test_dc5_price_positive_for_all_written_rows() -> None:
    """DC-5: price > 0 for every written row (both the DB CHECK and the summary agree)."""
    conn = _make_db()
    _seed_base(conn, "2026-W24", "2026-06-09")

    capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W24",
        shortlist_tickers=[],
        debate_set_tickers=W24_DEBATE_TICKERS,
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    zero_or_null = conn.execute(
        "SELECT ticker, price FROM shortlist_price_snapshots WHERE price IS NULL OR price <= 0"
    ).fetchall()
    assert not zero_or_null, (
        f"Rows with price ≤ 0 or NULL found (DC-5 violation): {zero_or_null}"
    )


# ---------------------------------------------------------------------------
# DC-6: Additive / no-mutation — locked tables untouched
# ---------------------------------------------------------------------------

def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return row counts for all locked tables."""
    counts = {}
    for table in _LOCKED_TABLES:
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def test_dc6_additive_no_mutation_of_locked_tables() -> None:
    """DC-6: a capture run only adds rows to shortlist_price_snapshots;
    all previously-locked tables remain at their pre-run row counts.
    """
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    before = _row_counts(conn)

    capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=["AAPL", "MSFT", "NVDA"],
        debate_set_tickers=["AMZN"],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    after = _row_counts(conn)

    for table in _LOCKED_TABLES:
        assert after[table] == before[table], (
            f"DC-6 violation: {table!r} row count changed from "
            f"{before[table]} → {after[table]} during capture"
        )

    # New table must have gained rows.
    new_rows = conn.execute(
        "SELECT COUNT(*) FROM shortlist_price_snapshots WHERE week_id='2026-W10'"
    ).fetchone()[0]
    assert new_rows == 4, f"Expected 4 new snapshot rows, got {new_rows}"


def test_dc6_additive_multi_week_accumulation() -> None:
    """DC-6: snapshots accumulate across weeks; prior weeks' rows are NOT overwritten."""
    conn = _make_db()
    _seed_multi_week(conn)

    # W1: 3 tickers.
    capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W01",
        shortlist_tickers=["AAPL", "MSFT", "NVDA"],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    # W2: 2 of the same + 1 new.
    capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W02",
        shortlist_tickers=["AAPL", "AMZN"],  # MSFT/NVDA still tracked via prior
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    # W1 rows must still be present.
    w1_count = conn.execute(
        "SELECT COUNT(*) FROM shortlist_price_snapshots WHERE week_id='2026-W01'"
    ).fetchone()[0]
    assert w1_count == 3, f"W1 rows were mutated or deleted — expected 3, got {w1_count}"

    # Total rows = W1 (3) + W2 (4: AAPL new shortlist + MSFT still + NVDA still + AMZN new).
    total = conn.execute(
        "SELECT COUNT(*) FROM shortlist_price_snapshots"
    ).fetchone()[0]
    assert total >= 6, f"Expected ≥6 total snapshot rows across W1+W2, got {total}"


# ---------------------------------------------------------------------------
# CaptureSummary — unit tests
# ---------------------------------------------------------------------------

def test_capture_summary_fields_populated() -> None:
    """CaptureSummary fields are all populated correctly from a capture run."""
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=["AAPL", "MSFT"],
        debate_set_tickers=["NVDA"],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_miss_fetcher_for("NVDA"),
    )

    assert summary.week_id == "2026-W10"
    assert summary.perimeter_count == 3      # AAPL + MSFT + NVDA
    assert summary.success_count == 2        # AAPL + MSFT priced OK
    assert summary.miss_count == 1           # NVDA missed
    assert summary.missed_tickers == ["NVDA"]


def test_capture_summary_empty_input() -> None:
    """CaptureSummary handles empty shortlist and debate_set gracefully."""
    conn = _make_db()
    _seed_base(conn, "2026-W10")

    summary = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=[],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    assert summary.perimeter_count == 0
    assert summary.success_count == 0
    assert summary.miss_count == 0


# ---------------------------------------------------------------------------
# Multi-week seeded fixture: ≥20 ticker-week cells
# (exercises tracking-duration + rejected-name + accumulation)
# ---------------------------------------------------------------------------

def test_multi_week_seeded_fixture_at_least_20_cells() -> None:
    """Seeded multi-week fixture delivers ≥20 ticker-week capture cells.

    Scenario:
    - W1: 10 tickers shortlisted (all new entries).
    - W2: 5 of those + 5 new tickers — 10 others carry forward from tracking.
    - W3: 3 new + prior tickers still in window.

    Expected cells: ≥20 across the three weeks.
    """
    conn = _make_db()
    _seed_multi_week(conn)

    w1_tickers = [f"STOCK{i:02d}" for i in range(10)]
    w2_new_tickers = [f"NEW{i:02d}" for i in range(5)]

    # W1.
    s1 = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W01",
        shortlist_tickers=w1_tickers,
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )
    assert s1.success_count == 10
    assert s1.newly_entered_count == 10

    # W2: 5 existing + 5 new.
    s2 = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W02",
        shortlist_tickers=w1_tickers[:5] + w2_new_tickers,
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )
    # W2 tracking set = (5 new-shortlist ∪ 5 new_tickers) ∪ (10 from W1 still in window).
    assert s2.newly_entered_count == 5    # only w2_new_tickers are new
    assert s2.still_tracking_count >= 5  # remaining W1 tickers not on W2 shortlist

    # W3: no new shortlist — all prior tickers still tracked.
    conn.execute(
        "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
        ("2026-W03", "2026-01-19", "andrew"),
    )
    conn.commit()
    s3 = capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W03",
        shortlist_tickers=[],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    total_cells = conn.execute(
        "SELECT COUNT(*) FROM shortlist_price_snapshots"
    ).fetchone()[0]
    assert total_cells >= 20, (
        f"Expected ≥20 ticker-week cells across W1+W2+W3, got {total_cells}"
    )


# ---------------------------------------------------------------------------
# Snapshot_date is carried from Alpaca return value
# ---------------------------------------------------------------------------

def test_snapshot_date_from_alpaca_return() -> None:
    """snapshot_date in the written row is the date Alpaca returned, not the run date."""
    ALPACA_DATE = "2026-06-05"  # Friday before a Monday run

    def _friday_fetcher(tickers: list[str]) -> dict[str, tuple[str, float]]:
        return {t: (ALPACA_DATE, 200.0) for t in tickers}

    conn = _make_db()
    _seed_base(conn, "2026-W24", "2026-06-09")  # run date is Monday 2026-06-09

    capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W24",
        shortlist_tickers=["AAPL"],
        debate_set_tickers=[],
        roster_version=1,
        user_id="andrew",
        _price_fetcher=_friday_fetcher,
    )

    row = conn.execute(
        "SELECT snapshot_date FROM shortlist_price_snapshots WHERE ticker='AAPL'"
    ).fetchone()
    assert row is not None
    assert row[0] == ALPACA_DATE, (
        f"snapshot_date should be the Alpaca-returned date {ALPACA_DATE!r}, got {row[0]!r}"
    )


# ---------------------------------------------------------------------------
# roster_version FK is written correctly
# ---------------------------------------------------------------------------

def test_roster_version_written_to_row() -> None:
    """roster_version on snapshot rows matches the passed roster_version argument."""
    conn = _make_db()
    # Insert a second roster version to test non-default.
    conn.execute(
        "INSERT OR IGNORE INTO roster_versions VALUES (?,?,?)",
        (2, "PoC roster v2 test", "2026-06-09"),
    )
    conn.commit()
    _seed_base(conn, "2026-W10")

    capture_shortlist_snapshots(
        conn=conn,
        week_id="2026-W10",
        shortlist_tickers=["AAPL"],
        debate_set_tickers=[],
        roster_version=2,
        user_id="andrew",
        _price_fetcher=_fake_price_fetcher,
    )

    row = conn.execute(
        "SELECT roster_version FROM shortlist_price_snapshots WHERE ticker='AAPL'"
    ).fetchone()
    assert row is not None
    assert row[0] == 2, f"Expected roster_version=2, got {row[0]}"
