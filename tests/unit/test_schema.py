# test_schema.py — Constraint tests for the round-table-portfolio SQLite ledger schema.
#
# Fixture inventory (≥21 per TDD Component 1 Sample Selection):
#   FK violations    : ≥10 cases (one per declared FK, including both version FKs)
#   CHECK violations : ≥5  cases (action='short', weight=1.5, confidence=0/6, round=3, type='momentum')
#   UNIQUE violations: ≥3  cases (portfolios, holdings, agent_stances)
#   Happy-path       : ≥3  cases (full week insert: 1 consensus + 7 persona portfolios + holdings)
#   Contract test    : 1   counterfactual-row contract (8th persona type rejected)
#   persona_reports  : 2+  cases (validator_passed CHECK, UNIQUE, full_report_path NOT NULL,
#                          version FK violations on persona_reports + persona_shortlists)
#
# All tests are deterministic (no external calls), so SKIP_LIVE does not apply here.
# Each test creates its own in-memory SQLite DB to stay fully isolated.

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "round_table_portfolio"
    / "storage"
    / "schema.sql"
)


def _make_db() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with foreign keys enabled and
    the full ledger schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def _seed_prerequisites(conn: sqlite3.Connection) -> None:
    """Insert the minimum rows needed as FK parents for most tests.

    Seeds:
    - roster_versions(1)  — already seeded by schema.sql INSERT OR IGNORE;
      included here defensively.
    - enhancement_versions(1) — same.
    - weeks('W-TEST')
    - portfolios(1)  — type='consensus' for W-TEST / user_id='andrew'
    """
    conn.executemany(
        "INSERT OR IGNORE INTO roster_versions VALUES (?,?,?)",
        [(1, "PoC initial 7-persona roster", "2026-06-01")],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO enhancement_versions VALUES (?,?,?)",
        [(1, "PoC initial state", "2026-06-01")],
    )
    conn.execute(
        "INSERT OR IGNORE INTO weeks(week_id, run_date, user_id) VALUES (?,?,?)",
        ("W-TEST", "2026-06-01", "andrew"),
    )
    conn.execute(
        """INSERT OR IGNORE INTO portfolios
           (week_id, type, user_id, roster_version, enhancement_version, created_at)
           VALUES (?,?,?,?,?,?)""",
        ("W-TEST", "consensus", "andrew", 1, 1, "2026-06-01T00:00:00"),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# PRAGMA check
# ---------------------------------------------------------------------------


def test_pragma_foreign_keys_is_on() -> None:
    """PRAGMA foreign_keys must return 1 — proves FK enforcement is active."""
    conn = _make_db()
    result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert result == 1, f"Expected foreign_keys=1, got {result}"


# ---------------------------------------------------------------------------
# Seeded lookup rows
# ---------------------------------------------------------------------------


def test_seed_rows_present() -> None:
    """schema.sql must seed roster_versions(1) and enhancement_versions(1)."""
    conn = _make_db()
    rv = conn.execute(
        "SELECT roster_version FROM roster_versions WHERE roster_version = 1"
    ).fetchone()
    ev = conn.execute(
        "SELECT enhancement_version FROM enhancement_versions WHERE enhancement_version = 1"
    ).fetchone()
    assert rv is not None, "roster_versions seed row (1) missing"
    assert ev is not None, "enhancement_versions seed row (1) missing"


# ---------------------------------------------------------------------------
# FK VIOLATION FIXTURES (≥10)
# ---------------------------------------------------------------------------


def test_fk_holdings_bad_portfolio_id() -> None:
    """FK-1: holdings.portfolio_id → portfolios must fire on unknown portfolio."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO holdings
               (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            (9999, "AAPL", 0.1, "add", "2026-06-01", "andrew", 1),
        )


def test_fk_holdings_bad_roster_version() -> None:
    """FK-2: holdings.roster_version → roster_versions must fire on unknown version."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO holdings
               (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            (1, "AAPL", 0.1, "add", "2026-06-01", "andrew", 99),
        )


def test_fk_portfolios_bad_week_id() -> None:
    """FK-3: portfolios.week_id → weeks must fire on unknown week."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO portfolios
               (week_id, type, user_id, roster_version, enhancement_version, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("W-NOPE", "value", "andrew", 1, 1, "2026-06-01T00:00:00"),
        )


def test_fk_portfolios_bad_roster_version() -> None:
    """FK-4: portfolios.roster_version → roster_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO portfolios
               (week_id, type, user_id, roster_version, enhancement_version, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("W-TEST", "value", "andrew", 99, 1, "2026-06-01T00:00:00"),
        )


def test_fk_portfolios_bad_enhancement_version() -> None:
    """FK-5: portfolios.enhancement_version → enhancement_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO portfolios
               (week_id, type, user_id, roster_version, enhancement_version, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("W-TEST", "growth", "andrew", 1, 99, "2026-06-01T00:00:00"),
        )


def test_fk_weekly_returns_bad_portfolio_id() -> None:
    """FK-6: weekly_returns.portfolio_id → portfolios must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO weekly_returns
               (portfolio_id, as_of_week_id, realized_return, unrealized_return,
                spy_return, alpha, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (9999, "W-TEST", 0.01, 0.02, 0.015, 0.005, "andrew", 1, 1),
        )


def test_fk_weekly_returns_bad_roster_version() -> None:
    """FK-7: weekly_returns.roster_version → roster_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO weekly_returns
               (portfolio_id, as_of_week_id, realized_return, unrealized_return,
                spy_return, alpha, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (1, "W-TEST", 0.01, 0.02, 0.015, 0.005, "andrew", 99, 1),
        )


def test_fk_weekly_returns_bad_enhancement_version() -> None:
    """FK-8: weekly_returns.enhancement_version → enhancement_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO weekly_returns
               (portfolio_id, as_of_week_id, realized_return, unrealized_return,
                spy_return, alpha, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (1, "W-TEST", 0.01, 0.02, 0.015, 0.005, "andrew", 1, 99),
        )


def test_fk_agent_stances_bad_roster_version() -> None:
    """FK-9: agent_stances.roster_version → roster_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO agent_stances
               (week_id, persona, ticker, round, action, target_weight,
                confidence, rationale, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 1, "add", 0.05, 4, "cheap", "andrew", 99, 1),
        )


def test_fk_agent_stances_bad_enhancement_version() -> None:
    """FK-10: agent_stances.enhancement_version → enhancement_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO agent_stances
               (week_id, persona, ticker, round, action, target_weight,
                confidence, rationale, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 1, "add", 0.05, 4, "cheap", "andrew", 1, 99),
        )


def test_fk_persona_reports_bad_roster_version() -> None:
    """FK-11: persona_reports.roster_version → roster_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO persona_reports
               (week_id, persona, summary, validator_passed, validator_notes,
                full_report_path, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "ok", 1, None, "state/r.md", "andrew", 99, 1),
        )


def test_fk_persona_reports_bad_enhancement_version() -> None:
    """FK-12: persona_reports.enhancement_version → enhancement_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO persona_reports
               (week_id, persona, summary, validator_passed, validator_notes,
                full_report_path, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "ok", 1, None, "state/r.md", "andrew", 1, 99),
        )


def test_fk_persona_shortlists_bad_roster_version() -> None:
    """FK-13: persona_shortlists.roster_version → roster_versions must fire."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO persona_shortlists
               (week_id, persona, ticker, is_cluster_peer, parent_ticker,
                user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 0, None, "andrew", 99),
        )


# ---------------------------------------------------------------------------
# CHECK VIOLATION FIXTURES (≥5)
# ---------------------------------------------------------------------------


def test_check_holdings_action_short() -> None:
    """CHECK-1: action='short' is not in the allowed set for holdings."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO holdings
               (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            (1, "AAPL", 0.1, "short", "2026-06-01", "andrew", 1),
        )


def test_check_holdings_weight_above_one() -> None:
    """CHECK-2: weight=1.5 violates weight >= 0 AND weight <= 1."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO holdings
               (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            (1, "AAPL", 1.5, "add", "2026-06-01", "andrew", 1),
        )


def test_check_holdings_weight_below_zero() -> None:
    """CHECK-3: weight=-0.1 violates weight >= 0."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO holdings
               (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            (1, "AAPL", -0.1, "add", "2026-06-01", "andrew", 1),
        )


def test_check_agent_stances_action_short() -> None:
    """CHECK-4: action='short' is not in the allowed set for agent_stances."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO agent_stances
               (week_id, persona, ticker, round, action, target_weight,
                confidence, rationale, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 1, "short", 0.05, 4, "nope", "andrew", 1, 1),
        )


def test_check_agent_stances_confidence_zero() -> None:
    """CHECK-5: confidence=0 violates confidence >= 1."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO agent_stances
               (week_id, persona, ticker, round, action, target_weight,
                confidence, rationale, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 1, "add", 0.05, 0, "nope", "andrew", 1, 1),
        )


def test_check_agent_stances_confidence_six() -> None:
    """CHECK-6: confidence=6 violates confidence <= 5."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO agent_stances
               (week_id, persona, ticker, round, action, target_weight,
                confidence, rationale, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 1, "add", 0.05, 6, "nope", "andrew", 1, 1),
        )


def test_check_agent_stances_round_three() -> None:
    """CHECK-7: round=3 is not in (1, 2)."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO agent_stances
               (week_id, persona, ticker, round, action, target_weight,
                confidence, rationale, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 3, "add", 0.05, 4, "nope", "andrew", 1, 1),
        )


def test_check_portfolios_type_momentum() -> None:
    """CHECK-8: type='momentum' is not in the allowed persona type list."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO portfolios
               (week_id, type, user_id, roster_version, enhancement_version, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("W-TEST", "momentum", "andrew", 1, 1, "2026-06-01T00:00:00"),
        )


def test_check_persona_reports_validator_passed_invalid() -> None:
    """CHECK-9: validator_passed=2 is not in (0, 1)."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO persona_reports
               (week_id, persona, summary, validator_passed, validator_notes,
                full_report_path, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "ok", 2, None, "state/r.md", "andrew", 1, 1),
        )


def test_check_persona_shortlists_is_cluster_peer_invalid() -> None:
    """CHECK-10: is_cluster_peer=2 is not in (0, 1)."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO persona_shortlists
               (week_id, persona, ticker, is_cluster_peer, parent_ticker,
                user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 2, None, "andrew", 1),
        )


def test_check_persona_reports_full_report_path_not_null() -> None:
    """NOT NULL-1: full_report_path NOT NULL must fire when omitted."""
    conn = _make_db()
    _seed_prerequisites(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO persona_reports
               (week_id, persona, summary, validator_passed, validator_notes,
                full_report_path, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "ok", 1, None, None, "andrew", 1, 1),
        )


# ---------------------------------------------------------------------------
# UNIQUE VIOLATION FIXTURES (≥3)
# ---------------------------------------------------------------------------


def test_unique_portfolios_same_week_type_user() -> None:
    """UNIQUE-1: inserting a second portfolio with the same (week_id, type, user_id) fails."""
    conn = _make_db()
    _seed_prerequisites(conn)
    # First insert — the seeded consensus row already exists; try to insert another.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO portfolios
               (week_id, type, user_id, roster_version, enhancement_version, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("W-TEST", "consensus", "andrew", 1, 1, "2026-06-02T00:00:00"),
        )


def test_unique_holdings_same_portfolio_ticker() -> None:
    """UNIQUE-2: inserting a second holding for the same (portfolio_id, ticker) fails."""
    conn = _make_db()
    _seed_prerequisites(conn)
    conn.execute(
        """INSERT INTO holdings
           (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
           VALUES (?,?,?,?,?,?,?)""",
        (1, "MSFT", 0.05, "add", "2026-06-01", "andrew", 1),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO holdings
               (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            (1, "MSFT", 0.07, "hold", "2026-06-01", "andrew", 1),
        )


def test_unique_agent_stances_same_week_persona_ticker_round_user() -> None:
    """UNIQUE-3: duplicate (week_id, persona, ticker, round, user_id) on agent_stances fails."""
    conn = _make_db()
    _seed_prerequisites(conn)
    conn.execute(
        """INSERT INTO agent_stances
           (week_id, persona, ticker, round, action, target_weight,
            confidence, rationale, user_id, roster_version, enhancement_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        ("W-TEST", "value", "AAPL", 1, "add", 0.05, 4, "first", "andrew", 1, 1),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO agent_stances
               (week_id, persona, ticker, round, action, target_weight,
                confidence, rationale, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "AAPL", 1, "hold", 0.03, 3, "second", "andrew", 1, 1),
        )


def test_unique_persona_reports_same_week_persona_user() -> None:
    """UNIQUE-4: duplicate (week_id, persona, user_id) on persona_reports fails."""
    conn = _make_db()
    _seed_prerequisites(conn)
    conn.execute(
        """INSERT INTO persona_reports
           (week_id, persona, summary, validator_passed, validator_notes,
            full_report_path, user_id, roster_version, enhancement_version)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        ("W-TEST", "value", "first", 1, None, "state/r1.md", "andrew", 1, 1),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO persona_reports
               (week_id, persona, summary, validator_passed, validator_notes,
                full_report_path, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "second", 1, None, "state/r2.md", "andrew", 1, 1),
        )


def test_unique_persona_shortlists_same_week_persona_ticker_user() -> None:
    """UNIQUE-5: duplicate (week_id, persona, ticker, user_id) on persona_shortlists fails."""
    conn = _make_db()
    _seed_prerequisites(conn)
    conn.execute(
        """INSERT INTO persona_shortlists
           (week_id, persona, ticker, is_cluster_peer, parent_ticker, user_id, roster_version)
           VALUES (?,?,?,?,?,?,?)""",
        ("W-TEST", "value", "NVDA", 0, None, "andrew", 1),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO persona_shortlists
               (week_id, persona, ticker, is_cluster_peer, parent_ticker, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            ("W-TEST", "value", "NVDA", 1, "NVDA", "andrew", 1),
        )


# ---------------------------------------------------------------------------
# HAPPY-PATH FIXTURES (≥3)
# ---------------------------------------------------------------------------

_SEVEN_PERSONAS = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]


def test_happy_path_full_week_portfolios_accepted() -> None:
    """Happy-1: 1 consensus + 7 persona portfolios for a single week_id all accepted."""
    conn = _make_db()
    _seed_prerequisites(conn)

    # The consensus row is already inserted by _seed_prerequisites.
    for persona in _SEVEN_PERSONAS:
        conn.execute(
            """INSERT INTO portfolios
               (week_id, type, user_id, roster_version, enhancement_version, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("W-TEST", persona, "andrew", 1, 1, "2026-06-01T00:00:00"),
        )
    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM portfolios WHERE week_id = 'W-TEST'"
    ).fetchone()[0]
    assert count == 8, f"Expected 8 portfolios (1 consensus + 7 persona), got {count}"


def test_happy_path_holdings_accepted() -> None:
    """Happy-2: 5 holdings per portfolio insert without error (FKs + CHECKs all pass)."""
    conn = _make_db()
    _seed_prerequisites(conn)

    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    for ticker in tickers:
        conn.execute(
            """INSERT INTO holdings
               (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            (1, ticker, 0.10, "add", "2026-06-01", "andrew", 1),
        )
    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM holdings WHERE portfolio_id = 1"
    ).fetchone()[0]
    assert count == 5


def test_happy_path_persona_reports_and_shortlists_accepted() -> None:
    """Happy-3: 7 persona_reports rows + shortlists for a single week insert cleanly."""
    conn = _make_db()
    _seed_prerequisites(conn)

    for persona in _SEVEN_PERSONAS:
        conn.execute(
            """INSERT INTO persona_reports
               (week_id, persona, summary, validator_passed, validator_notes,
                full_report_path, user_id, roster_version, enhancement_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                "W-TEST",
                persona,
                f"{persona} report summary",
                1,
                None,
                f"state/reports/W-TEST/{persona}.md",
                "andrew",
                1,
                1,
            ),
        )
        # Each persona shortlists 2 names (1 direct + 1 cluster peer).
        conn.execute(
            """INSERT INTO persona_shortlists
               (week_id, persona, ticker, is_cluster_peer, parent_ticker, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            ("W-TEST", persona, "AAPL", 0, None, "andrew", 1),
        )
        conn.execute(
            """INSERT INTO persona_shortlists
               (week_id, persona, ticker, is_cluster_peer, parent_ticker, user_id, roster_version)
               VALUES (?,?,?,?,?,?,?)""",
            ("W-TEST", persona, "AMD", 1, "NVDA", "andrew", 1),
        )
    conn.commit()

    report_count = conn.execute(
        "SELECT COUNT(*) FROM persona_reports WHERE week_id = 'W-TEST'"
    ).fetchone()[0]
    shortlist_count = conn.execute(
        "SELECT COUNT(*) FROM persona_shortlists WHERE week_id = 'W-TEST'"
    ).fetchone()[0]
    assert report_count == 7, f"Expected 7 reports, got {report_count}"
    assert shortlist_count == 14, f"Expected 14 shortlist rows (7×2), got {shortlist_count}"


# ---------------------------------------------------------------------------
# COUNTERFACTUAL-ROW CONTRACT TEST
# ---------------------------------------------------------------------------


def test_counterfactual_row_contract() -> None:
    """Contract test: 1 consensus + 7 named-persona portfolios succeeds;
    an 8th portfolio with type='event-driven' is REJECTED by the type CHECK.

    This is the roster-lock proof per TDD Component 1 Quality Criteria #6.
    """
    conn = _make_db()
    _seed_prerequisites(conn)

    # consensus already inserted by _seed_prerequisites.
    for persona in _SEVEN_PERSONAS:
        conn.execute(
            """INSERT INTO portfolios
               (week_id, type, user_id, roster_version, enhancement_version, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("W-TEST", persona, "andrew", 1, 1, "2026-06-01T00:00:00"),
        )
    conn.commit()

    # Verify the 8 legal rows were accepted.
    count = conn.execute(
        "SELECT COUNT(*) FROM portfolios WHERE week_id = 'W-TEST'"
    ).fetchone()[0]
    assert count == 8, f"Expected 8, got {count}"

    # The 8th persona type 'event-driven' must be REJECTED.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO portfolios
               (week_id, type, user_id, roster_version, enhancement_version, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("W-TEST", "event-driven", "andrew2", 1, 1, "2026-06-01T00:00:00"),
        )


# ---------------------------------------------------------------------------
# Multi-tenant user_id column presence
# ---------------------------------------------------------------------------


def test_user_id_column_present_and_defaults() -> None:
    """Every domain table must have a user_id column defaulting to 'andrew'."""
    conn = _make_db()
    _seed_prerequisites(conn)

    # Insert a holdings row WITHOUT specifying user_id — default should kick in.
    conn.execute(
        """INSERT INTO holdings
           (portfolio_id, ticker, weight, action, entry_date, roster_version)
           VALUES (?,?,?,?,?,?)""",
        (1, "TSLA", 0.05, "add", "2026-06-01", 1),
    )
    conn.commit()

    row = conn.execute(
        "SELECT user_id FROM holdings WHERE ticker = 'TSLA'"
    ).fetchone()
    assert row is not None
    assert row[0] == "andrew", f"Expected user_id='andrew', got {row[0]!r}"
