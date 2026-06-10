-- schema.sql — Canonical SQLite ledger DDL for round-table-portfolio.
-- Locked at Week 1 (PoC M1-001). Schema correctness is irreversible once
-- weekly-run data exists; any post-Week-1 change is a Directional decision
-- requiring explicit founder approval (TDD §1.5 Failure Handling, Component 1).
--
-- IMPORTANT: apply_schema.py sets PRAGMA foreign_keys = ON on every connection.
-- SQLite does NOT enforce foreign keys by default — the PRAGMA is mandatory.

-- ============================================================
-- Lookup / version tables (seeded on first apply)
-- ============================================================

CREATE TABLE IF NOT EXISTS roster_versions (
    roster_version  INTEGER PRIMARY KEY,
    description     TEXT    NOT NULL,
    created_date    TEXT    NOT NULL   -- ISO-8601 date string
);

CREATE TABLE IF NOT EXISTS enhancement_versions (
    enhancement_version  INTEGER PRIMARY KEY,
    description          TEXT    NOT NULL,
    created_date         TEXT    NOT NULL   -- ISO-8601 date string
);

-- ============================================================
-- Core time dimension
-- ============================================================

CREATE TABLE IF NOT EXISTS weeks (
    week_id   TEXT PRIMARY KEY,   -- ISO week label e.g. '2026-W23'
    run_date  TEXT NOT NULL,      -- ISO-8601 date of the weekly run
    notes     TEXT,
    user_id   TEXT NOT NULL DEFAULT 'andrew'
);

-- ============================================================
-- Portfolio (1 consensus + 7 persona counterfactuals per week)
-- ============================================================

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id             TEXT    NOT NULL REFERENCES weeks(week_id),
    type                TEXT    NOT NULL CHECK(type IN (
                            'consensus',
                            'value',
                            'growth',
                            'discretionary-macro',
                            'cta-systematic-macro',
                            'technical',
                            'quant-systematic',
                            'risk-officer'
                        )),
    user_id             TEXT    NOT NULL DEFAULT 'andrew',
    roster_version      INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    enhancement_version INTEGER NOT NULL REFERENCES enhancement_versions(enhancement_version),
    created_at          TEXT    NOT NULL,   -- ISO-8601 datetime
    UNIQUE(week_id, type, user_id)
);

-- ============================================================
-- Holdings (one row per ticker per portfolio)
-- ============================================================

CREATE TABLE IF NOT EXISTS holdings (
    holding_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(portfolio_id),
    ticker          TEXT    NOT NULL,
    weight          REAL    NOT NULL CHECK(weight >= 0 AND weight <= 1),
    action          TEXT    NOT NULL CHECK(action IN ('add', 'reduce', 'hold', 'exit')),
    entry_date      TEXT    NOT NULL,   -- ISO-8601 date
    user_id         TEXT    NOT NULL DEFAULT 'andrew',
    roster_version  INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    UNIQUE(portfolio_id, ticker)
);

-- ============================================================
-- Weekly returns (mark-to-market on all 8 portfolios per week)
-- ============================================================

CREATE TABLE IF NOT EXISTS weekly_returns (
    return_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id        INTEGER NOT NULL REFERENCES portfolios(portfolio_id),
    as_of_week_id       TEXT    NOT NULL REFERENCES weeks(week_id),
    realized_return     REAL,
    unrealized_return   REAL,
    spy_return          REAL,
    alpha               REAL,   -- continuous annualized alpha; never collapsed to win/loss boolean
    user_id             TEXT    NOT NULL DEFAULT 'andrew',
    roster_version      INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    enhancement_version INTEGER NOT NULL REFERENCES enhancement_versions(enhancement_version),
    UNIQUE(portfolio_id, as_of_week_id)
);

-- ============================================================
-- Transcripts (one per week × user — file pointer to full debate log)
-- ============================================================

CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id         TEXT    NOT NULL REFERENCES weeks(week_id),
    summary         TEXT,
    vote_tally      TEXT,   -- JSON string
    key_contention  TEXT,
    full_log_path   TEXT    NOT NULL,   -- path to state/debates/YYYY-WNN.md
    user_id         TEXT    NOT NULL DEFAULT 'andrew',
    UNIQUE(week_id, user_id)
);

-- ============================================================
-- Agent stances (on DEBATED names only; ~15-40 tickers per week)
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_stances (
    stance_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id             TEXT    NOT NULL REFERENCES weeks(week_id),
    persona             TEXT    NOT NULL,
    ticker              TEXT    NOT NULL,
    round               INTEGER NOT NULL CHECK(round IN (1, 2)),
    action              TEXT    NOT NULL CHECK(action IN ('add', 'reduce', 'hold', 'exit')),
    target_weight       REAL,
    confidence          INTEGER NOT NULL CHECK(confidence >= 1 AND confidence <= 5),
    rationale           TEXT,
    user_id             TEXT    NOT NULL DEFAULT 'andrew',
    roster_version      INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    enhancement_version INTEGER NOT NULL REFERENCES enhancement_versions(enhancement_version),
    UNIQUE(week_id, persona, ticker, round, user_id)
);

-- ============================================================
-- Persona reports (one per week × persona — the WHY record)
-- Added 2026-06-01 terminal: replaces screener_scores
-- ============================================================

CREATE TABLE IF NOT EXISTS persona_reports (
    report_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id             TEXT    NOT NULL REFERENCES weeks(week_id),
    persona             TEXT    NOT NULL,
    summary             TEXT,
    validator_passed    INTEGER NOT NULL CHECK(validator_passed IN (0, 1)),
    validator_notes     TEXT,
    full_report_path    TEXT    NOT NULL,   -- path to state/reports/<week>/<persona>.md; NOT NULL enforced
    user_id             TEXT    NOT NULL DEFAULT 'andrew',
    roster_version      INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    enhancement_version INTEGER NOT NULL REFERENCES enhancement_versions(enhancement_version),
    UNIQUE(week_id, persona, user_id)
);

-- ============================================================
-- Persona shortlists (one row per surfaced ticker per persona per week)
-- Added 2026-06-01 terminal: replaces screener_scores
-- ============================================================

CREATE TABLE IF NOT EXISTS persona_shortlists (
    shortlist_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id         TEXT    NOT NULL REFERENCES weeks(week_id),
    persona         TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    is_cluster_peer INTEGER NOT NULL CHECK(is_cluster_peer IN (0, 1)),
    parent_ticker   TEXT,   -- NULL when is_cluster_peer=0; names the shortlisted parent when =1
    user_id         TEXT    NOT NULL DEFAULT 'andrew',
    roster_version  INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    UNIQUE(week_id, persona, ticker, user_id)
);

-- ============================================================
-- Per-stock counterfactual price snapshots (additive — M5, 2026-06-09)
-- One row per shortlisted/debate-set ticker per weekly run.
-- Captures EVERY surfaced stock (accepted AND rejected) for the
-- "what we passed on" performance curve.
-- NOTE: no enhancement_version FK — a price snapshot is a market fact,
--       not a decision-version-dependent artefact (ALIGNMENT-LOG 2026-06-09).
-- NOTE: no FK to persona_shortlists — the surfaced-by join is recovered at
--       read time (Component 39) so tracking persists beyond shortlist lifetime.
-- ============================================================

CREATE TABLE IF NOT EXISTS shortlist_price_snapshots (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    week_id         TEXT    NOT NULL REFERENCES weeks(week_id),
    ticker          TEXT    NOT NULL,
    snapshot_date   TEXT    NOT NULL,   -- actual price date returned by Alpaca (may lag run_date by 1 trading day)
    price           REAL    NOT NULL CHECK(price > 0),
    roster_version  INTEGER NOT NULL REFERENCES roster_versions(roster_version),
    user_id         TEXT    NOT NULL DEFAULT 'andrew',
    UNIQUE(week_id, ticker, user_id)
);

-- ============================================================
-- Seed data: PoC initial lookup rows
-- INSERT OR IGNORE so apply_schema.py is idempotent on re-runs.
-- ============================================================

INSERT OR IGNORE INTO roster_versions (roster_version, description, created_date)
VALUES (1, 'PoC initial 7-persona roster', date('now'));

INSERT OR IGNORE INTO enhancement_versions (enhancement_version, description, created_date)
VALUES (1, 'PoC initial state — no seasonal reviews executed', date('now'));
