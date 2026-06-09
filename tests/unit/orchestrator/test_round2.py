"""Unit tests for orchestrator/round2.py — Component 24 (deterministic parts).

Coverage matrix:
  Parser — clean fixtures:
    test_parse_clean_defend            — valid defend reply: all stances position_change=defended
    test_parse_clean_revise            — valid revise reply: stances changed, position_change=revised
    test_parse_mixed_defend_revise     — some stances defended, some revised (valid)
    test_parse_uppercase_action        — ADD/REDUCE/EXIT/HOLD normalised to lowercase
    test_parse_markdown_fenced_json    — ```json ... ``` stripped before parsing

  Parser — domain-edge fixtures (fail-loud, no coercion):
    test_parse_malformed_missing_rebuttal_narrative — missing rebuttal_narrative → Round2ParseError
    test_parse_malformed_missing_stances           — missing stances key → Round2ParseError
    test_parse_malformed_not_json                  — garbage string → Round2ParseError
    test_parse_confidence_zero_rejected            — confidence=0 → Round2ParseError
    test_parse_confidence_six_rejected             — confidence=6 → Round2ParseError
    test_parse_confidence_boundary_valid           — confidence=1 and confidence=5 both valid
    test_parse_target_weight_over_cap_rejected     — target_weight > max_position_weight → Round2ParseError
    test_parse_target_weight_zero_valid            — target_weight=0.0 valid (exit stance)
    test_parse_bad_action_rejected                 — unknown action verb → Round2ParseError
    test_parse_bad_position_change_rejected        — position_change="unchanged" → Round2ParseError
    test_parse_empty_rebuttal_narrative_rejected   — empty string → Round2ParseError
    test_parse_empty_stances_array_rejected        — stances=[] → Round2ParseError
    test_parse_missing_ticker_rejected             — stance with no ticker → Round2ParseError

  Writer — payload construction:
    test_writer_produces_round2_rows               — build_round2_stance_payloads → round=2 payloads
    test_writer_row_count_matches_stances          — one row per stance
    test_writer_round_field_is_2                   — all payloads have round=2

  Writer — UNIQUE coexistence (round=2 distinct from round=1 under constraint):
    test_writer_coexists_with_round1_in_sqlite     — round=2 rows INSERT alongside round=1 rows
                                                     without violating UNIQUE(week_id,persona,ticker,round,user_id)
    test_writer_round2_distinct_from_round1        — same (persona,ticker), different round → 2 rows
    test_writer_duplicate_round2_violates_unique   — inserting same round=2 row twice → IntegrityError

  capture_round2_stances — convenience wrapper:
    test_capture_two_outliers_returns_both         — 2 slugs in → 2 entries out
    test_capture_not_two_raises                    — 1 or 3 slugs → ValueError (cost contract)

  Real-data-anchored writer test:
    test_writer_2026_w24_outliers_coexist          — growth + cta-systematic-macro from W24 fixture;
                                                     round=2 rows coexist with round=1 under UNIQUE
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from round_table_portfolio.orchestrator.round2 import (
    Round2ParseError,
    Round2Reply,
    Round2Stance,
    build_round2_stance_payloads,
    capture_round2_stances,
    parse_round2_reply,
)
from round_table_portfolio.orchestrator.round1 import AgentStancePayload
from round_table_portfolio.storage.apply_schema import apply_schema

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEEK_ID = "2026-W24"          # real W24 week from the fixture
MAX_WEIGHT = 0.20
PERSONA_GROWTH = "growth"
PERSONA_CTA = "cta-systematic-macro"

# The 2026-W24 fixture lives here (real-data-anchored writer test).
_FIXTURE_PATH = (
    Path(__file__).parents[3]
    / "tests"
    / "fixtures"
    / "stances_2026_w24_round1.json"
)


# ---------------------------------------------------------------------------
# JSON builders
# ---------------------------------------------------------------------------

def _make_r2_json(
    persona: str,
    tickers: list[str],
    *,
    action: str = "hold",
    target_weight: float = 0.0,
    confidence: int = 3,
    rationale_suffix: str = "",
    position_change: str = "defended",
    rebuttal_narrative: str = "The counterargument raises valid macro concerns but my mandate's momentum signals are intact.",
    extra_fields: dict | None = None,
) -> str:
    """Build a valid ROUND 2 OUTPUT SCHEMA JSON string."""
    stances = [
        {
            "ticker": t,
            "action": action,
            "target_weight": target_weight,
            "confidence": confidence,
            "rationale": f"{persona} on {t}: position maintained{rationale_suffix}.",
            "position_change": position_change,
        }
        for t in tickers
    ]
    doc: dict = {
        "round": 2,
        "rebuttal_narrative": rebuttal_narrative,
        "stances": stances,
    }
    if extra_fields:
        doc.update(extra_fields)
    return json.dumps(doc)


def _make_r2_json_revise(
    persona: str,
    tickers: list[str],
    *,
    new_action: str = "reduce",
    new_weight: float = 0.05,
    confidence: int = 4,
) -> str:
    """Build a Round-2 revise reply (stances changed)."""
    stances = [
        {
            "ticker": t,
            "action": new_action,
            "target_weight": new_weight,
            "confidence": confidence,
            "rationale": f"{persona} on {t}: revised after counterargument moved me on valuation.",
            "position_change": "revised",
        }
        for t in tickers
    ]
    return json.dumps({
        "round": 2,
        "rebuttal_narrative": (
            "The panel's counterargument on stretched multiples was compelling; "
            "reducing exposure on the highest-P/E names."
        ),
        "stances": stances,
    })


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _make_temp_db(tmp_path: Path) -> Path:
    """Create a fresh in-memory-equivalent SQLite DB with the canonical schema."""
    db_path = tmp_path / "test_ledger.db"
    apply_schema(db_path)
    return db_path


def _seed_round1_row(conn: sqlite3.Connection, persona: str, ticker: str, week_id: str = WEEK_ID) -> None:
    """Insert one round=1 agent_stances row (seeding the 'pre-existing' round-1 context)."""
    conn.execute(
        """
        INSERT INTO agent_stances
          (week_id, persona, ticker, round, action, target_weight, confidence,
           rationale, user_id, roster_version, enhancement_version)
        VALUES (?, ?, ?, 1, 'hold', 0.0, 3, 'round-1 rationale', 'andrew', 1, 1)
        """,
        (week_id, persona, ticker),
    )


def _insert_payload(conn: sqlite3.Connection, p: AgentStancePayload) -> None:
    conn.execute(
        """
        INSERT INTO agent_stances
          (week_id, persona, ticker, round, action, target_weight, confidence,
           rationale, user_id, roster_version, enhancement_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            p.week_id, p.persona, p.ticker, p.round,
            p.action, p.target_weight, p.confidence, p.rationale,
            p.user_id, p.roster_version, p.enhancement_version,
        ),
    )


# ---------------------------------------------------------------------------
# Parser — clean fixtures
# ---------------------------------------------------------------------------

class TestParseCleanFixtures:
    """Valid Round-2 replies parse without errors."""

    def test_parse_clean_defend(self) -> None:
        """Clean defend: all stances position_change=defended, rationale strengthened."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL", "MSFT"], position_change="defended")
        reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
        assert reply.persona == PERSONA_GROWTH
        assert len(reply.stances) == 2
        assert all(s.position_change == "defended" for s in reply.stances)
        assert reply.rebuttal_narrative.startswith("The counterargument")

    def test_parse_clean_revise(self) -> None:
        """Clean revise: stances changed, position_change=revised."""
        raw = _make_r2_json_revise(PERSONA_CTA, ["NVDA", "META"])
        reply = parse_round2_reply(raw, persona_slug=PERSONA_CTA, week_id=WEEK_ID)
        assert reply.persona == PERSONA_CTA
        assert len(reply.stances) == 2
        assert all(s.position_change == "revised" for s in reply.stances)
        # Verify action was normalized from uppercase if needed.
        assert all(s.action in {"add", "reduce", "hold", "exit"} for s in reply.stances)

    def test_parse_mixed_defend_revise(self) -> None:
        """Some stances defended, some revised — both are valid in one reply."""
        stances = [
            {"ticker": "AAPL", "action": "hold", "target_weight": 0.0,
             "confidence": 3, "rationale": "unchanged", "position_change": "defended"},
            {"ticker": "NVDA", "action": "reduce", "target_weight": 0.05,
             "confidence": 4, "rationale": "moved by counterargument", "position_change": "revised"},
        ]
        raw = json.dumps({"round": 2, "rebuttal_narrative": "Partial agreement.", "stances": stances})
        reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
        pc_values = {s.ticker: s.position_change for s in reply.stances}
        assert pc_values["AAPL"] == "defended"
        assert pc_values["NVDA"] == "revised"

    def test_parse_uppercase_action(self) -> None:
        """ADD/REDUCE/EXIT/HOLD are normalised to lowercase — same as Component 14."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], action="ADD", target_weight=0.10)
        reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
        assert reply.stances[0].action == "add"

    def test_parse_markdown_fenced_json(self) -> None:
        """```json ... ``` code fences are stripped before parsing."""
        inner = _make_r2_json(PERSONA_GROWTH, ["AAPL"])
        raw = f"```json\n{inner}\n```"
        reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
        assert len(reply.stances) == 1


# ---------------------------------------------------------------------------
# Parser — domain-edge fixtures (fail-loud)
# ---------------------------------------------------------------------------

class TestParseDomainEdges:
    """Malformed or out-of-domain replies raise Round2ParseError immediately."""

    def test_parse_malformed_missing_rebuttal_narrative(self) -> None:
        """Missing rebuttal_narrative key → Round2ParseError."""
        stances = [{"ticker": "AAPL", "action": "hold", "target_weight": 0.0,
                    "confidence": 3, "rationale": "r", "position_change": "defended"}]
        raw = json.dumps({"round": 2, "stances": stances})
        with pytest.raises(Round2ParseError, match="rebuttal_narrative"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_malformed_missing_stances(self) -> None:
        """Missing stances key → Round2ParseError."""
        raw = json.dumps({"round": 2, "rebuttal_narrative": "ok"})
        with pytest.raises(Round2ParseError, match="stances"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_malformed_not_json(self) -> None:
        """Garbage string → Round2ParseError with JSON error context."""
        with pytest.raises(Round2ParseError, match="not valid JSON"):
            parse_round2_reply("this is not json", persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_confidence_zero_rejected(self) -> None:
        """confidence=0 is outside 1..5 — rejected with no coercion."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], confidence=0)
        with pytest.raises(Round2ParseError, match="outside 1..5"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_confidence_six_rejected(self) -> None:
        """confidence=6 is outside 1..5 — rejected."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], confidence=6)
        with pytest.raises(Round2ParseError, match="outside 1..5"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_confidence_boundary_valid(self) -> None:
        """confidence=1 and confidence=5 are both valid boundary values."""
        for conf in (1, 5):
            raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], confidence=conf)
            reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
            assert reply.stances[0].confidence == conf

    def test_parse_target_weight_over_cap_rejected(self) -> None:
        """target_weight > max_position_weight → Round2ParseError, no coercion."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], action="add", target_weight=0.25)
        with pytest.raises(Round2ParseError, match="outside"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID, max_position_weight=0.20)

    def test_parse_target_weight_zero_valid(self) -> None:
        """target_weight=0.0 is valid (exit/hold stance)."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], action="exit", target_weight=0.0)
        reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
        assert reply.stances[0].target_weight == 0.0

    def test_parse_bad_action_rejected(self) -> None:
        """Unknown action verb → Round2ParseError, no coercion."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], action="BUY")
        with pytest.raises(Round2ParseError, match="4-value vocabulary"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_bad_position_change_rejected(self) -> None:
        """position_change outside {defended, revised} → Round2ParseError."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], position_change="unchanged")
        with pytest.raises(Round2ParseError, match="position_change"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_empty_rebuttal_narrative_rejected(self) -> None:
        """Empty rebuttal_narrative string → Round2ParseError."""
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], rebuttal_narrative="")
        with pytest.raises(Round2ParseError, match="rebuttal_narrative"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_empty_stances_array_rejected(self) -> None:
        """stances=[] (no stances) → Round2ParseError."""
        raw = json.dumps({"round": 2, "rebuttal_narrative": "ok", "stances": []})
        with pytest.raises(Round2ParseError, match="empty"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)

    def test_parse_missing_ticker_rejected(self) -> None:
        """Stance without a ticker → Round2ParseError."""
        stances = [{"action": "hold", "target_weight": 0.0,
                    "confidence": 3, "rationale": "r", "position_change": "defended"}]
        raw = json.dumps({"round": 2, "rebuttal_narrative": "ok", "stances": stances})
        with pytest.raises(Round2ParseError, match="missing 'ticker'"):
            parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)


# ---------------------------------------------------------------------------
# Writer — payload construction
# ---------------------------------------------------------------------------

class TestWriterPayloads:
    """build_round2_stance_payloads produces correct AgentStancePayload rows."""

    def _make_reply(self, persona: str, tickers: list[str], pc: str = "defended") -> Round2Reply:
        raw = _make_r2_json(persona, tickers, position_change=pc)
        return parse_round2_reply(raw, persona_slug=persona, week_id=WEEK_ID)

    def test_writer_produces_round2_rows(self) -> None:
        """build_round2_stance_payloads returns AgentStancePayload instances."""
        reply = self._make_reply(PERSONA_GROWTH, ["AAPL", "MSFT"])
        payloads = build_round2_stance_payloads(reply, week_id=WEEK_ID)
        assert all(isinstance(p, AgentStancePayload) for p in payloads)

    def test_writer_row_count_matches_stances(self) -> None:
        """One payload per restated stance."""
        tickers = ["AAPL", "MSFT", "NVDA"]
        reply = self._make_reply(PERSONA_GROWTH, tickers)
        payloads = build_round2_stance_payloads(reply, week_id=WEEK_ID)
        assert len(payloads) == 3

    def test_writer_round_field_is_2(self) -> None:
        """All payloads have round=2 — the DB column that distinguishes from Round 1."""
        reply = self._make_reply(PERSONA_CTA, ["AMZN", "META"])
        payloads = build_round2_stance_payloads(reply, week_id=WEEK_ID)
        assert all(p.round == 2 for p in payloads)

    def test_writer_fields_match_parsed_stances(self) -> None:
        """Payload fields mirror the parsed Round2Stance values."""
        stances_data = [
            {"ticker": "AAPL", "action": "add", "target_weight": 0.12,
             "confidence": 4, "rationale": "strong momentum", "position_change": "revised"},
        ]
        raw = json.dumps({"round": 2, "rebuttal_narrative": "moved by valuation.", "stances": stances_data})
        reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
        payloads = build_round2_stance_payloads(reply, week_id=WEEK_ID)
        p = payloads[0]
        assert p.ticker == "AAPL"
        assert p.action == "add"
        assert p.target_weight == pytest.approx(0.12)
        assert p.confidence == 4
        assert p.rationale == "strong momentum"
        assert p.round == 2
        assert p.persona == PERSONA_GROWTH


# ---------------------------------------------------------------------------
# Writer — UNIQUE coexistence with round=1 rows
# ---------------------------------------------------------------------------

class TestWriterUniqueCoexistence:
    """Round-2 rows coexist with round-1 rows under UNIQUE(week_id,persona,ticker,round,user_id)."""

    def test_writer_coexists_with_round1_in_sqlite(self, tmp_path: Path) -> None:
        """INSERT round=2 rows alongside existing round=1 rows — no constraint violation."""
        db_path = _make_temp_db(tmp_path)
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL", "MSFT"], position_change="defended")
        reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
        payloads = build_round2_stance_payloads(reply, week_id=WEEK_ID)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        # Seed weeks row (FK parent of agent_stances).
        conn.execute(
            "INSERT OR IGNORE INTO weeks (week_id, run_date, notes, user_id) VALUES (?, ?, ?, ?)",
            (WEEK_ID, "2026-06-08", "test", "andrew"),
        )
        # Seed round=1 rows for the same (persona, ticker) pairs.
        for ticker in ["AAPL", "MSFT"]:
            _seed_round1_row(conn, PERSONA_GROWTH, ticker)
        conn.commit()

        # Now insert round=2 rows — must not raise.
        for p in payloads:
            _insert_payload(conn, p)
        conn.commit()

        rows = conn.execute(
            "SELECT round, COUNT(*) FROM agent_stances WHERE persona=? AND week_id=? GROUP BY round",
            (PERSONA_GROWTH, WEEK_ID),
        ).fetchall()
        conn.close()
        round_counts = {r: cnt for r, cnt in rows}
        assert round_counts.get(1) == 2, "Expected 2 round=1 rows"
        assert round_counts.get(2) == 2, "Expected 2 round=2 rows"

    def test_writer_round2_distinct_from_round1(self, tmp_path: Path) -> None:
        """Same (persona, ticker) yields 2 DB rows: one round=1, one round=2."""
        db_path = _make_temp_db(tmp_path)
        raw = _make_r2_json(PERSONA_CTA, ["NVDA"], position_change="revised",
                            action="reduce", target_weight=0.05)
        reply = parse_round2_reply(raw, persona_slug=PERSONA_CTA, week_id=WEEK_ID)
        payloads = build_round2_stance_payloads(reply, week_id=WEEK_ID)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT OR IGNORE INTO weeks (week_id, run_date, notes, user_id) VALUES (?, ?, ?, ?)",
            (WEEK_ID, "2026-06-08", "test", "andrew"),
        )
        _seed_round1_row(conn, PERSONA_CTA, "NVDA")
        conn.commit()

        for p in payloads:
            _insert_payload(conn, p)
        conn.commit()

        rows = conn.execute(
            "SELECT round, action FROM agent_stances WHERE persona=? AND ticker=? AND week_id=? ORDER BY round",
            (PERSONA_CTA, "NVDA", WEEK_ID),
        ).fetchall()
        conn.close()
        assert len(rows) == 2, "Expected 2 rows: one round=1, one round=2"
        assert rows[0][0] == 1
        assert rows[1][0] == 2
        # round=1 action was 'hold' (seeded); round=2 action should be 'reduce' (revised).
        assert rows[0][1] == "hold"
        assert rows[1][1] == "reduce"

    def test_writer_duplicate_round2_violates_unique(self, tmp_path: Path) -> None:
        """Inserting the same round=2 row twice violates UNIQUE — confirms constraint is active."""
        db_path = _make_temp_db(tmp_path)
        raw = _make_r2_json(PERSONA_GROWTH, ["AAPL"], position_change="defended")
        reply = parse_round2_reply(raw, persona_slug=PERSONA_GROWTH, week_id=WEEK_ID)
        payloads = build_round2_stance_payloads(reply, week_id=WEEK_ID)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT OR IGNORE INTO weeks (week_id, run_date, notes, user_id) VALUES (?, ?, ?, ?)",
            (WEEK_ID, "2026-06-08", "test", "andrew"),
        )
        conn.commit()
        _insert_payload(conn, payloads[0])
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            _insert_payload(conn, payloads[0])
            conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# capture_round2_stances — convenience wrapper
# ---------------------------------------------------------------------------

class TestCaptureRound2Stances:
    """capture_round2_stances is the orchestrator's single entry point."""

    def test_capture_two_outliers_returns_both(self) -> None:
        """2 slugs in → 2 (Round2Reply, payloads) entries out."""
        raw_replies = {
            PERSONA_GROWTH: _make_r2_json(PERSONA_GROWTH, ["AAPL", "MSFT"]),
            PERSONA_CTA: _make_r2_json_revise(PERSONA_CTA, ["NVDA", "META"]),
        }
        result = capture_round2_stances(raw_replies, week_id=WEEK_ID)
        assert set(result.keys()) == {PERSONA_GROWTH, PERSONA_CTA}
        for slug, (reply, payloads) in result.items():
            assert isinstance(reply, Round2Reply)
            assert all(p.round == 2 for p in payloads)
            assert reply.persona == slug

    def test_capture_not_two_raises(self) -> None:
        """1 or 3 outlier replies → ValueError (cost-contract guard)."""
        with pytest.raises(ValueError, match="exactly 2"):
            capture_round2_stances(
                {PERSONA_GROWTH: _make_r2_json(PERSONA_GROWTH, ["AAPL"])},
                week_id=WEEK_ID,
            )
        with pytest.raises(ValueError, match="exactly 2"):
            capture_round2_stances(
                {
                    PERSONA_GROWTH: _make_r2_json(PERSONA_GROWTH, ["AAPL"]),
                    PERSONA_CTA: _make_r2_json(PERSONA_CTA, ["MSFT"]),
                    "value": _make_r2_json("value", ["NVDA"]),
                },
                week_id=WEEK_ID,
            )


# ---------------------------------------------------------------------------
# Real-data-anchored writer test (W24 fixture)
# ---------------------------------------------------------------------------

class TestWriterRealDataW24:
    """Use the 2026-W24 round=1 fixture to anchor the coexistence test.

    Source: tests/fixtures/stances_2026_w24_round1.json (real session data,
    PII-free, provenance: 2026-W24 live run, exported 2026-06-08).
    The 2 W24 outliers were growth + cta-systematic-macro (confirmed by M3-001
    dissent computation over this fixture).
    """

    def _load_r1_fixture(self) -> list[dict]:
        if not _FIXTURE_PATH.exists():
            pytest.skip(f"W24 fixture not found at {_FIXTURE_PATH}")
        import json
        return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_writer_2026_w24_outliers_coexist(self, tmp_path: Path) -> None:
        """growth + cta-systematic-macro round=2 rows coexist with their round=1 rows."""
        fixture_rows = self._load_r1_fixture()
        growth_tickers = [r["ticker"] for r in fixture_rows if r["persona"] == "growth"]
        cta_tickers = [r["ticker"] for r in fixture_rows if r["persona"] == "cta-systematic-macro"]

        assert growth_tickers, "growth tickers must be present in W24 fixture"
        assert cta_tickers, "cta-systematic-macro tickers must be present in W24 fixture"

        # Build stubbed round=2 replies using the real W24 tickers.
        # Defend stance: growth defends all positions (no change).
        growth_r2_raw = _make_r2_json(
            "growth", growth_tickers, position_change="defended",
            rebuttal_narrative=(
                "The counterargument highlights valuation risk, but my mandate is to find "
                "companies with durable multi-year earnings growth — the momentum signals "
                "remain intact and I maintain my Round-1 positions."
            ),
        )
        # Revise stance: cta-systematic-macro revises two positions.
        cta_r2_raw = _make_r2_json_revise("cta-systematic-macro", cta_tickers)

        raw_replies = {
            "growth": growth_r2_raw,
            "cta-systematic-macro": cta_r2_raw,
        }
        result = capture_round2_stances(raw_replies, week_id=WEEK_ID)

        db_path = _make_temp_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT OR IGNORE INTO weeks (week_id, run_date, notes, user_id) VALUES (?, ?, ?, ?)",
            (WEEK_ID, "2026-06-08", "test", "andrew"),
        )
        # Seed all round=1 rows from the real fixture.
        for row in fixture_rows:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_stances
                  (week_id, persona, ticker, round, action, target_weight,
                   confidence, rationale, user_id, roster_version, enhancement_version)
                VALUES (?, ?, ?, 1, ?, ?, ?, 'r1 rationale', 'andrew', 1, 1)
                """,
                (WEEK_ID, row["persona"], row["ticker"], row["action"],
                 row["target_weight"], row["confidence"]),
            )
        conn.commit()

        # Insert round=2 rows for the 2 W24 outliers.
        for slug, (reply, payloads) in result.items():
            for p in payloads:
                _insert_payload(conn, p)
        conn.commit()

        # Verify: for each outlier persona, both round=1 and round=2 rows exist.
        for persona in ("growth", "cta-systematic-macro"):
            rounds_present = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT round FROM agent_stances WHERE persona=? AND week_id=?",
                    (persona, WEEK_ID),
                ).fetchall()
            }
            assert 1 in rounds_present, f"{persona}: round=1 rows missing"
            assert 2 in rounds_present, f"{persona}: round=2 rows missing"

        # Verify row counts: round=2 count must equal the number of restated tickers.
        for persona, tickers in [("growth", growth_tickers), ("cta-systematic-macro", cta_tickers)]:
            r2_count = conn.execute(
                "SELECT COUNT(*) FROM agent_stances WHERE persona=? AND week_id=? AND round=2",
                (persona, WEEK_ID),
            ).fetchone()[0]
            assert r2_count == len(tickers), (
                f"{persona}: expected {len(tickers)} round=2 rows, got {r2_count}"
            )

        conn.close()
