"""Unit tests for orchestrator/weekly_run.py — Component 12.

All tests use SKIP_LIVE=1-compatible execution: no live subagent dispatches,
no real ledger DB (tmp_path), canned persona_replies, StubOnMandateJudge.
STUB_ALLOW=1 is set in every fixture so the sibling-task stubs are callable.

Coverage matrix:
  full_sequence          — 1 weeks + 8 portfolios + round-1-only agent_stances
                           + transcripts + 7 persona_reports (AC #1)
  rollback_fk_violation  — forced FK violation on Nth portfolio rolls back ALL
                           rows; zero rows for that week_id survive (AC #2)
  no_round2_reachable    — orchestrator source contains no Round-2 dispatch (AC #3)
  approve_parse          — "approve" → panel_approved (AC #4)
  override_parse         — "override: reduce AAPL to 5%" → founder_override (AC #4)
  ambiguous_reprompt     — ambiguous → one re-prompt → resolved (AC #4)
  ambiguous_double       — ambiguous twice → defaults to panel_approved (AC #4)
"""

from __future__ import annotations

import inspect
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterator

import pytest

from round_table_portfolio.orchestrator.weekly_run import (
    _parse_founder_reply,
    _parse_reply_with_reprompt,
    run_weekly,
)
from round_table_portfolio.personas.output_validator import (
    PersonaConfig,
    StructuralConfig,
    StubOnMandateJudge,
    ValidatorConfig,
)
from round_table_portfolio.storage.apply_schema import apply_schema


# ---------------------------------------------------------------------------
# Helpers
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

# A valid RESEARCH OUTPUT SCHEMA that passes all structural gates.
# 3 shortlisted tickers with cluster peers; substantive report text.
def _make_persona_output(slug: str) -> str:
    """Produce a valid RESEARCH OUTPUT SCHEMA JSON for a given persona slug."""
    report_body = (
        f"The {slug} analysis identifies three compelling opportunities. "
        "AAPL trades at 25× earnings with strong FCF yield of 4.5% and balance-sheet strength. "
        "MSFT shows revenue growth of 15% YoY with cloud ARR acceleration. "
        "GOOGL offers search dominance and AI optionality at a P/E of 20×. "
        "Technical indicators: RSI 52, MACD neutral. Valuation metrics: P/E, FCF, EPS growth. "
        "Data sources consulted: EDGAR 10-K, FRED macro series, price history via Alpaca. "
        "Risk considerations: concentration in mega-cap tech; macro regime shift risk. "
        "Conviction level: high for AAPL and MSFT; moderate for GOOGL pending antitrust outcome. "
        "Portfolio weight recommendation: AAPL 15%, MSFT 12%, GOOGL 10%, CASH 63%. "
        "This allocation reflects a fully-invested posture within the persona mandate."
    )
    schema = {
        "shortlist": [
            {"ticker": "AAPL", "why": "Strong FCF, capital return program.", "cluster": ["QCOM"]},
            {"ticker": "MSFT", "why": "Cloud moat, durable revenue.", "cluster": ["GOOGL"]},
            {"ticker": "NVDA", "why": "AI infrastructure leader.", "cluster": ["AMD", "INTC"]},
        ],
        "report": report_body,
        "web_searches_used": 4,
        "data_tool_calls_used": 8,
    }
    return json.dumps(schema)


# Debate set derived from _make_persona_output shortlists:
# Direct (is_cluster_peer=0): AAPL, MSFT, NVDA
# Cluster peers:              QCOM (under AAPL), GOOGL (under MSFT),
#                             AMD and INTC (under NVDA)
# Note: GOOGL appears as a cluster peer of MSFT; it is not directly shortlisted.
_DEBATE_SET_FROM_FIXTURE = ["AAPL", "MSFT", "NVDA", "QCOM", "GOOGL", "AMD", "INTC"]


def _make_round1_output(slug: str) -> str:
    """Produce a valid ROUND 1 OUTPUT SCHEMA JSON for the fixture debate set.

    Every debate-set ticker gets a stance; weights sum to exactly 1.0 with CASH.
    Weights respect max_position_weight=0.20.
    """
    # 7 tickers: give each a 0.10 weight (total 0.70), CASH = 0.30.
    stances = [
        {
            "ticker": t,
            "action": "ADD",
            "target_weight": 0.10,
            "confidence": 3,
            "rationale": f"Stub Round-1 rationale for {t} by {slug}.",
        }
        for t in _DEBATE_SET_FROM_FIXTURE
    ]
    # Counterfactual: best portfolio from this persona's own shortlist.
    # Use AAPL + MSFT + NVDA at 0.15 each; rest CASH.
    counterfactual = {
        "AAPL": 0.15,
        "MSFT": 0.12,
        "NVDA": 0.10,
        "CASH": 0.63,
    }
    schema = {
        "stances": stances,
        "counterfactual_portfolio": counterfactual,
        "narrative_summary": f"{slug}: constructive on tech sector; high conviction AAPL/MSFT.",
    }
    return json.dumps(schema)


def _make_validator_config() -> ValidatorConfig:
    """Minimal ValidatorConfig that passes the canned reports."""
    structural = StructuralConfig(
        min_report_chars=100,
        min_ticker_references=2,
        min_metric_terms=1,
        metric_terms=("p/e", "fcf", "yield", "eps", "revenue"),
        data_source_signals=("edgar", "fred", "alpaca", "valuation", "price"),
    )
    personas = {slug: PersonaConfig((), ()) for slug in PERSONA_SLUGS_7}
    return ValidatorConfig(structural=structural, personas=personas)


def _make_personas_yaml(tmp_path: Path) -> Path:
    content = "slugs:\n" + "".join(f"  - {s}\n" for s in PERSONA_SLUGS_7)
    p = tmp_path / "personas.yaml"
    p.write_text(content)
    return p


def _make_thresholds_yaml(tmp_path: Path) -> Path:
    content = (
        "max_position_weight: 0.20\n"
        "dissent_std_dev_threshold: 0.08\n"
        "run_window_hours: 5.0\n"
        "contested_week_threshold: 0.50\n"
        "action_direction_map:\n"
        "  add: 1.0\n"
        "  hold: 0.0\n"
        "  reduce: -0.5\n"
        "  exit: -1.0\n"
        "n_outliers: 2\n"
        "divergence_tiebreak: alpha_asc\n"
    )
    p = tmp_path / "thresholds.yaml"
    p.write_text(content)
    return p


def _make_db(tmp_path: Path) -> Path:
    """Create a fresh ledger.db with the full schema applied."""
    db_path = tmp_path / "ledger.db"
    apply_schema(db_path=db_path)
    return db_path


def _row_counts(db_path: Path, week_id: str) -> dict[str, int]:
    """Return a dict of table → row count for the given week_id."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    tables = {
        "weeks": "SELECT COUNT(*) FROM weeks WHERE week_id=?",
        "portfolios": "SELECT COUNT(*) FROM portfolios WHERE week_id=?",
        "agent_stances": "SELECT COUNT(*) FROM agent_stances WHERE week_id=?",
        "persona_reports": "SELECT COUNT(*) FROM persona_reports WHERE week_id=?",
        "persona_shortlists": "SELECT COUNT(*) FROM persona_shortlists WHERE week_id=?",
        "transcripts": "SELECT COUNT(*) FROM transcripts WHERE week_id=?",
    }
    counts = {}
    for table, sql in tables.items():
        counts[table] = conn.execute(sql, (week_id,)).fetchone()[0]
    conn.close()
    return counts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def stub_allow(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Allow stubs to be called in all tests in this module."""
    monkeypatch.setenv("STUB_ALLOW", "1")
    yield


@pytest.fixture()
def week_id() -> str:
    return "2026-W23"


@pytest.fixture()
def persona_replies() -> dict[str, str]:
    return {slug: _make_persona_output(slug) for slug in PERSONA_SLUGS_7}


@pytest.fixture()
def round1_replies() -> dict[str, str]:
    """Valid ROUND 1 OUTPUT SCHEMA JSON for each persona, covering the fixture debate set."""
    return {slug: _make_round1_output(slug) for slug in PERSONA_SLUGS_7}


@pytest.fixture()
def run_env(tmp_path: Path) -> dict:
    """All paths and configs needed for a run_weekly call."""
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / "memory").mkdir()
    # Seed memory files (needed by the real runner; stubs don't read them but
    # the filesystem layout should be correct).
    for slug in PERSONA_SLUGS_7:
        (state_root / "memory" / f"{slug}.md").write_text(f"# {slug} memory\nNo prior weeks.\n")

    db_path = _make_db(tmp_path)
    personas_yaml = _make_personas_yaml(tmp_path)
    thresholds_yaml = _make_thresholds_yaml(tmp_path)
    v_config = _make_validator_config()

    return {
        "state_root": state_root,
        "db_path": db_path,
        "personas_config": personas_yaml,
        "thresholds_config": thresholds_yaml,
        "validator_config_obj": v_config,
        # Round-1 replies: valid ROUND 1 OUTPUT SCHEMA JSON for each persona.
        # Included here so all run_weekly callers in this module can pass
        # round1_replies=run_env["round1_replies"] without changing test
        # signatures one-by-one.
        "round1_replies": {slug: _make_round1_output(slug) for slug in PERSONA_SLUGS_7},
    }


# ---------------------------------------------------------------------------
# AC #1 — Full sequence row counts
# ---------------------------------------------------------------------------

class TestFullSequence:
    """AC #1: One /weekly-run produces the correct row counts in every table."""

    def test_row_counts_weeks(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        result = run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        counts = _row_counts(run_env["db_path"], week_id)
        assert counts["weeks"] == 1, f"Expected 1 weeks row, got {counts['weeks']}"

    def test_row_counts_portfolios(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        counts = _row_counts(run_env["db_path"], week_id)
        assert counts["portfolios"] == 8, (
            f"Expected 8 portfolios (1 consensus + 7 named), got {counts['portfolios']}"
        )

    def test_row_counts_agent_stances_round1_only(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        db = run_env["db_path"]
        conn = sqlite3.connect(str(db))
        # All stances must be round=1.
        round2_count = conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=2",
            (week_id,),
        ).fetchone()[0]
        round1_count = conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=1",
            (week_id,),
        ).fetchone()[0]
        conn.close()

        assert round2_count == 0, f"Round-2 stances must not exist in M2, got {round2_count}"
        assert round1_count > 0, "Expected at least one round-1 stance"

    def test_row_counts_persona_reports(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        counts = _row_counts(run_env["db_path"], week_id)
        assert counts["persona_reports"] == 7, (
            f"Expected 7 persona_reports, got {counts['persona_reports']}"
        )

    def test_transcripts_row_non_null_path(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        db = run_env["db_path"]
        conn = sqlite3.connect(str(db))
        row = conn.execute(
            "SELECT full_log_path FROM transcripts WHERE week_id=?", (week_id,)
        ).fetchone()
        conn.close()

        assert row is not None, "No transcripts row found"
        log_path = Path(row[0])
        assert log_path.exists(), f"Transcript file does not exist: {log_path}"
        assert log_path.stat().st_size > 0, "Transcript file is empty"

    def test_result_object_shape(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        result = run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        assert result.week_id == week_id
        assert result.num_portfolios_written == 8
        assert result.num_persona_reports == 7
        assert result.num_stances_written > 0
        assert result.transcript_path is not None


# ---------------------------------------------------------------------------
# AC #2 — Transactional rollback on forced FK violation
# ---------------------------------------------------------------------------

class TestTransactionalRollback:
    """AC #2: A forced mid-write failure leaves ZERO rows for the week_id."""

    def test_fk_violation_rollback_leaves_zero_rows(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a bad portfolio payload whose type violates the CHECK constraint."""
        from round_table_portfolio.orchestrator import _stubs as stubs_module

        original_materialize = stubs_module.materialize_portfolios

        def broken_materialize(
            counterfactuals,
            consensus_weights,
            *,
            prior_portfolios=None,
            week_id,
            config=None,
            entry_date="",
        ):
            payloads = original_materialize(
                counterfactuals,
                consensus_weights,
                prior_portfolios=prior_portfolios,
                week_id=week_id,
                config=config,
                entry_date=entry_date,
            )
            # Inject a payload with an invalid type to trigger the CHECK constraint.
            from round_table_portfolio.orchestrator._stubs import PortfolioPayload
            bad = PortfolioPayload(
                type="INVALID_TYPE_TRIGGERS_CHECK_FAIL",
                week_id=week_id,
                roster_version=1,
                enhancement_version=1,
                user_id="andrew",
                holdings=[],
            )
            return payloads + [bad]

        monkeypatch.setattr(
            "round_table_portfolio.orchestrator.weekly_run.materialize_portfolios",
            broken_materialize,
        )

        with pytest.raises(Exception):
            run_weekly(
                "round-table-portfolio",
                week_id=week_id,
                persona_replies=persona_replies,
                founder_reply="approve",
                judge=StubOnMandateJudge(),
                **run_env,
            )

        counts = _row_counts(run_env["db_path"], week_id)
        assert counts["weeks"] == 0, (
            f"Rollback failed: weeks row survived (got {counts['weeks']})"
        )
        assert counts["portfolios"] == 0, (
            f"Rollback failed: portfolio rows survived (got {counts['portfolios']})"
        )
        assert counts["agent_stances"] == 0, (
            f"Rollback failed: agent_stances rows survived (got {counts['agent_stances']})"
        )
        assert counts["persona_reports"] == 0, (
            f"Rollback failed: persona_reports survived (got {counts['persona_reports']})"
        )
        assert counts["transcripts"] == 0, (
            f"Rollback failed: transcripts row survived (got {counts['transcripts']})"
        )

    def test_holdings_fk_violation_rollback(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a holdings row with an impossible weight (> 1) to trigger CHECK."""
        from round_table_portfolio.orchestrator import _stubs as stubs_module
        from round_table_portfolio.orchestrator._stubs import HoldingPayload, PortfolioPayload

        original_materialize = stubs_module.materialize_portfolios

        def materialize_with_bad_holding(
            counterfactuals,
            consensus_weights,
            *,
            prior_portfolios=None,
            week_id,
            config=None,
            entry_date="",
        ):
            payloads = original_materialize(
                counterfactuals,
                consensus_weights,
                prior_portfolios=prior_portfolios,
                week_id=week_id,
                config=config,
                entry_date=entry_date,
            )
            # Replace the 6th portfolio's first holding with weight=1.5 (> 1 violates CHECK).
            if len(payloads) >= 6:
                p = payloads[5]
                bad_holding = HoldingPayload(
                    ticker="BADTICKER",
                    weight=1.5,  # violates CHECK(weight >= 0 AND weight <= 1)
                    action="add",
                    entry_date=entry_date or week_id,
                )
                payloads[5] = PortfolioPayload(
                    type=p.type,
                    week_id=p.week_id,
                    roster_version=p.roster_version,
                    enhancement_version=p.enhancement_version,
                    user_id=p.user_id,
                    holdings=[bad_holding],
                )
            return payloads

        monkeypatch.setattr(
            "round_table_portfolio.orchestrator.weekly_run.materialize_portfolios",
            materialize_with_bad_holding,
        )

        with pytest.raises(Exception):
            run_weekly(
                "round-table-portfolio",
                week_id=week_id,
                persona_replies=persona_replies,
                founder_reply="approve",
                judge=StubOnMandateJudge(),
                **run_env,
            )

        counts = _row_counts(run_env["db_path"], week_id)
        assert counts["portfolios"] == 0, (
            f"Rollback failed: portfolio rows survived after holdings CHECK violation "
            f"(got {counts['portfolios']})"
        )


# ---------------------------------------------------------------------------
# AC #3 — No Round-2 dispatch is reachable in M2
# ---------------------------------------------------------------------------

class TestNoRound2:
    """AC #3: The orchestrator source contains no Round-2 dispatch path."""

    def test_no_round2_dispatch_in_source(self) -> None:
        """Inspect the weekly_run module source for any Round-2 dispatch call.

        "round=2" legitimately appears in SQL that writes round=1 agent_stances
        rows (the column is named 'round') and in comment text.  What we must
        not find is a *dispatch* function call that would invoke Round 2 logic —
        function names that would only exist if Round-2 code were wired in.
        """
        from round_table_portfolio.orchestrator import weekly_run as wr_module
        source = inspect.getsource(wr_module)

        # These function-call patterns would indicate an actual Round-2 dispatch.
        # Simple column references ("round=2" in SQL, comments) are permitted.
        forbidden_call_patterns = [
            "dispatch_round2(",
            "capture_round2(",
            "capture_round_2(",
            "run_round2(",
            "run_round_2(",
            "round_two(",
        ]
        violations = [p for p in forbidden_call_patterns if p.lower() in source.lower()]
        assert not violations, (
            f"Round-2 dispatch call found in weekly_run source — M2 scope violation. "
            f"Patterns found: {violations}"
        )

    def test_no_round2_stances_in_db(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        db = run_env["db_path"]
        conn = sqlite3.connect(str(db))
        round2 = conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=2",
            (week_id,),
        ).fetchone()[0]
        conn.close()
        assert round2 == 0, f"Round-2 stances must not exist in M2, got {round2}"


# ---------------------------------------------------------------------------
# AC #4 — Approve / override / ambiguous parse paths
# ---------------------------------------------------------------------------

class TestFounderReplyParser:
    """AC #4: parse_founder_reply and _parse_reply_with_reprompt unit tests."""

    # --- _parse_founder_reply ---

    @pytest.mark.parametrize("reply", [
        "approve",
        "Approve",
        "APPROVE",
        "approved",
        "yes",
        "looks good",
        "confirm",
        "confirmed",
        "go ahead",
    ])
    def test_approve_variants(self, reply: str) -> None:
        decision, delta = _parse_founder_reply(reply)
        assert decision == "panel_approved", f"Expected panel_approved for {reply!r}"
        assert delta == ""

    @pytest.mark.parametrize("reply,expected_delta", [
        ("override: reduce AAPL to 5%", "reduce AAPL to 5%"),
        ("Override: add MSFT at 8% instead", "add MSFT at 8% instead"),
        ("OVERRIDE: swap GOOGL for META", "swap GOOGL for META"),
        ("override:drop NVDA entirely", "drop NVDA entirely"),
    ])
    def test_override_variants(self, reply: str, expected_delta: str) -> None:
        decision, delta = _parse_founder_reply(reply)
        assert decision == "founder_override", f"Expected founder_override for {reply!r}"
        assert delta == expected_delta

    @pytest.mark.parametrize("reply", [
        "maybe",
        "looks reasonable but not sure",
        "what about TSLA",
        "I need to think about this",
        "",
        "   ",
    ])
    def test_ambiguous_replies(self, reply: str) -> None:
        decision, _ = _parse_founder_reply(reply)
        assert decision == "ambiguous", f"Expected ambiguous for {reply!r}"

    # --- _parse_reply_with_reprompt ---

    def test_approve_no_reprompt_needed(self) -> None:
        reprompt_called = []

        def reprompt() -> str:
            reprompt_called.append(True)
            return "approve"

        decision, delta = _parse_reply_with_reprompt("approve", reprompt)
        assert decision == "panel_approved"
        assert delta == ""
        assert not reprompt_called, "Re-prompt must not be called for a clear approve"

    def test_override_no_reprompt_needed(self) -> None:
        reprompt_called = []

        def reprompt() -> str:
            reprompt_called.append(True)
            return "approve"

        decision, delta = _parse_reply_with_reprompt("override: reduce AAPL to 5%", reprompt)
        assert decision == "founder_override"
        assert delta == "reduce AAPL to 5%"
        assert not reprompt_called

    def test_ambiguous_then_approve(self) -> None:
        reprompt_calls = []

        def reprompt() -> str:
            reprompt_calls.append("approve")
            return "approve"

        decision, delta = _parse_reply_with_reprompt("hmm not sure", reprompt)
        assert decision == "panel_approved"
        assert delta == ""
        assert len(reprompt_calls) == 1, "Exactly one re-prompt must be issued"

    def test_ambiguous_then_override(self) -> None:
        def reprompt() -> str:
            return "override: cut AAPL to 3%"

        decision, delta = _parse_reply_with_reprompt("I'll think about it", reprompt)
        assert decision == "founder_override"
        assert delta == "cut AAPL to 3%"

    def test_ambiguous_twice_defaults_to_approved(self) -> None:
        """Two ambiguous replies → defaults to panel_approved with a warning."""
        reprompt_calls = []

        def reprompt() -> str:
            reprompt_calls.append("still ambiguous")
            return "I'm still not sure"

        decision, delta = _parse_reply_with_reprompt("not certain", reprompt)
        assert decision == "panel_approved", (
            "After two ambiguous replies, must default to panel_approved"
        )
        assert delta == ""
        assert len(reprompt_calls) == 1, "Re-prompt called exactly once (not twice)"

    # --- Integration: decision recorded in DB ---

    def test_approve_written_to_weeks(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        conn = sqlite3.connect(str(run_env["db_path"]))
        row = conn.execute(
            "SELECT notes FROM weeks WHERE week_id=?", (week_id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert "panel_approved" in (row[0] or "")

    def test_override_written_to_weeks(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        result = run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="override: reduce AAPL to 5%",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        assert result.decision_type == "founder_override"
        assert "reduce AAPL to 5%" in result.decision_delta

    def test_ambiguous_reprompt_integration(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        reprompt_calls = []

        def reprompt() -> str:
            reprompt_calls.append(True)
            return "approve"

        result = run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="not sure",  # ambiguous
            reprompt_fn=reprompt,
            judge=StubOnMandateJudge(),
            **run_env,
        )
        assert result.decision_type == "panel_approved"
        assert len(reprompt_calls) == 1, "Exactly one re-prompt must have been issued"


# ---------------------------------------------------------------------------
# AC #1 (M2-010) — Exactly 7 validator-claim files produced per run
# ---------------------------------------------------------------------------

class TestValidatorClaimsProduced:
    """M2-010 AC: one durable validator_claim JSON per persona, no more, no less."""

    def test_exactly_7_claim_files_produced(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        claims_dir = run_env["state_root"] / "reports" / week_id / "validator_claims"
        assert claims_dir.is_dir(), "validator_claims/ subdir not created"
        claim_files = sorted(claims_dir.glob("*.json"))
        assert len(claim_files) == 7, (
            f"Expected exactly 7 claim files (one per persona), got {len(claim_files)}: "
            f"{[f.name for f in claim_files]}"
        )

    def test_claim_file_names_match_persona_slugs(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        claims_dir = run_env["state_root"] / "reports" / week_id / "validator_claims"
        actual_slugs = {f.stem for f in claims_dir.glob("*.json")}
        assert actual_slugs == set(PERSONA_SLUGS_7), (
            f"Claim file slugs don't match expected personas.\n"
            f"Expected: {sorted(PERSONA_SLUGS_7)}\n"
            f"Got:      {sorted(actual_slugs)}"
        )

    def test_each_claim_contains_required_fields(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        import json as _json

        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        claims_dir = run_env["state_root"] / "reports" / week_id / "validator_claims"
        required_fields = {"week_id", "persona", "passed", "notes", "stage", "llm_justification"}
        for claim_file in claims_dir.glob("*.json"):
            payload = _json.loads(claim_file.read_text(encoding="utf-8"))
            missing = required_fields - payload.keys()
            assert not missing, (
                f"Claim for {claim_file.stem} is missing fields: {missing}"
            )
            # passed must be a bool (not the _stub sentinel string)
            assert isinstance(payload["passed"], bool), (
                f"Claim for {claim_file.stem}: 'passed' must be a bool, got {type(payload['passed'])}"
            )

    def test_claims_are_round_trippable(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        """Re-load each claim JSON back into a ReportValidationResult-shaped object."""
        from round_table_portfolio.personas.output_validator import (
            ReportValidationResult,
            load_validator_claim,
        )

        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        claims_dir = run_env["state_root"] / "reports" / week_id / "validator_claims"
        for claim_file in sorted(claims_dir.glob("*.json")):
            loaded = load_validator_claim(claim_file)
            assert isinstance(loaded, ReportValidationResult), (
                f"load_validator_claim({claim_file.name}) did not return a ReportValidationResult"
            )
            assert isinstance(loaded.passed, bool)
            assert isinstance(loaded.notes, str)
            assert isinstance(loaded.stage, str)
            assert isinstance(loaded.llm_justification, str)

    def test_no_stub_sentinel_in_claims(
        self,
        run_env: dict,
        persona_replies: dict,
        week_id: str,
    ) -> None:
        """The real writer must not leave the '_stub' key from the old stub."""
        import json as _json

        run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=persona_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )
        claims_dir = run_env["state_root"] / "reports" / week_id / "validator_claims"
        for claim_file in claims_dir.glob("*.json"):
            payload = _json.loads(claim_file.read_text(encoding="utf-8"))
            assert "_stub" not in payload, (
                f"Claim {claim_file.name} still contains '_stub' key — stub was not replaced"
            )


# ---------------------------------------------------------------------------
# AC #5 — Full test suite passes (covered by running all the above)
# ---------------------------------------------------------------------------
# The test suite itself IS the AC #5 gate — pytest must report zero failures.
# No additional test class needed; the above classes cover all deterministic ACs.


# ---------------------------------------------------------------------------
# Regression: TASK-M2-011 — Layer-2 must NOT re-run report-prose gates
# ---------------------------------------------------------------------------

def _make_persona_output_sparse_opening(slug: str) -> str:
    """Persona report whose FIRST PARAGRAPH names zero tickers.

    The opening sentence is deliberately ticker-free to reproduce the TASK-M2-011
    failure: Layer-2 received the 500-char summary (= first paragraph) and ran
    the structural ticker gate on it, false-failing personas with sparse openings.

    The FULL report (later paragraphs) still contains ≥2 tickers and passes all
    Layer-1 structural gates — so this fixture is valid at Layer-1 and must also
    pass Layer-2 after the fix.
    """
    # First paragraph: no tickers, just narrative prose — deliberately >500 chars
    # so _extract_summary(max_chars=500) returns only this ticker-free text.
    opening = (
        "The current macro environment presents compelling opportunities for "
        "disciplined investors willing to look through near-term volatility. "
        "Valuation dispersion across sectors is at a multi-year high, and "
        "capital discipline has improved markedly since the rate-shock of 2022. "
        "Central bank policy is on hold, credit spreads remain contained, and "
        "earnings revision breadth has turned positive for the first time in "
        "three quarters, suggesting the fundamental backdrop is improving even "
        "as headline indices hover near all-time highs. "
        "This report identifies three high-conviction ideas supported by "
        "fundamental and quantitative research conducted across the full universe."
    )
    # Sanity-check: opening must exceed 500 chars so the summary window is
    # entirely ticker-free (regression guard for this fixture itself).
    assert len(opening) > 500, (
        f"Opening paragraph is only {len(opening)} chars — must exceed 500 so "
        "_extract_summary returns a ticker-free summary. Extend the prose."
    )
    # Subsequent paragraphs: tickers appear, passing the structural ≥2 gate on
    # the FULL report but NOT on the first-paragraph summary alone.
    body = (
        " AAPL trades at 25× earnings with a strong FCF yield of 4.5% and "
        "robust balance-sheet strength. MSFT shows revenue growth of 15% YoY "
        "with cloud ARR acceleration. GOOGL offers search dominance and AI "
        "optionality at a P/E of 20×. "
        "Technical indicators: RSI 52, MACD neutral. Valuation metrics: P/E, "
        "FCF, EPS growth. "
        "Data sources consulted: EDGAR 10-K, FRED macro series, price history "
        "via Alpaca. "
        "Risk considerations: concentration in mega-cap tech. "
        "Conviction: high for AAPL and MSFT; moderate for GOOGL. "
        "Portfolio: AAPL 15%, MSFT 12%, GOOGL 10%, CASH 63%."
    )
    report_body = opening + body
    schema = {
        "shortlist": [
            {"ticker": "AAPL", "why": "Strong FCF, capital return.", "cluster": ["QCOM"]},
            {"ticker": "MSFT", "why": "Cloud moat.", "cluster": ["GOOGL"]},
            {"ticker": "NVDA", "why": "AI infrastructure.", "cluster": ["AMD", "INTC"]},
        ],
        "report": report_body,
        "web_searches_used": 4,
        "data_tool_calls_used": 8,
    }
    return json.dumps(schema)


class TestLayer2DoesNotReRunProseGates:
    """Regression: TASK-M2-011.

    A persona whose report's first paragraph contains 0 tickers (but whose FULL
    report is structurally valid with ≥2 tickers) must PASS Layer-2 when its
    counterfactual portfolio is fully invested.

    Before the fix, Layer-2 passed `report_payload.summary` (= 500-char first
    paragraph) to `validate_persona_report`, which re-ran the structural ticker
    gate on that truncated text and raised RuntimeError even though Layer-1
    already passed the full report.
    """

    def test_sparse_opening_persona_passes_layer2(
        self,
        run_env: dict,
        week_id: str,
    ) -> None:
        """All 7 personas with ticker-free opening paragraphs must run to completion."""
        # Persona replies: valid full reports but first paragraphs have 0 tickers.
        sparse_replies = {
            slug: _make_persona_output_sparse_opening(slug) for slug in PERSONA_SLUGS_7
        }

        # This must NOT raise RuntimeError about Layer-2 structural gate failure.
        result = run_weekly(
            "round-table-portfolio",
            week_id=week_id,
            persona_replies=sparse_replies,
            founder_reply="approve",
            judge=StubOnMandateJudge(),
            **run_env,
        )

        # The run completed — 8 portfolios written, 7 persona reports.
        counts = _row_counts(run_env["db_path"], week_id)
        assert counts["portfolios"] == 8, (
            f"Expected 8 portfolios after sparse-opening run, got {counts['portfolios']}. "
            "Layer-2 may have incorrectly re-run the report-prose structural gate."
        )
        assert counts["persona_reports"] == 7, (
            f"Expected 7 persona_reports, got {counts['persona_reports']}."
        )

    def test_sparse_opening_summary_would_fail_structural_gate(
        self,
        run_env: dict,
    ) -> None:
        """Confirm the first paragraph of the sparse-opening fixture actually contains
        fewer than 2 tickers — proving the regression fixture is exercising the right path.
        """
        import json as _json
        import re

        _TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
        _TICKER_EXCLUDE = frozenset({
            "THE", "AND", "FOR", "NOT", "BUT", "NOR", "YET", "SO", "OR",
            "WITH", "FROM", "THAT", "THIS", "PASS", "FAIL",
            "FCF", "RSI", "TAM", "CPI", "PCE", "FED", "SEC", "ETF",
            "CEO", "CFO", "COO", "IPO", "GDP", "EPS", "ROE", "ROA",
            "FRED", "ISM", "VIX", "YTD", "YOY", "TTM",
            "MACD", "ROIC", "ARR", "AUM",
            "REIT", "SPAC", "AI", "ML", "US", "UK", "EU", "USD",
            "GPU", "FSD", "AWS", "PBM",
        })

        raw = _make_persona_output_sparse_opening("value")
        report_text = _json.loads(raw)["report"]
        # Extract first paragraph (up to first blank line or 500 chars — mirrors
        # _extract_summary behaviour).
        first_para = report_text[:500].split("\n\n")[0]

        tickers_in_summary = {
            tok for tok in _TICKER_RE.findall(first_para)
            if tok not in _TICKER_EXCLUDE
        }
        assert len(tickers_in_summary) < 2, (
            f"Fixture first paragraph contains {len(tickers_in_summary)} tickers "
            f"({tickers_in_summary}) — it should contain <2 to reproduce the bug. "
            "Adjust _make_persona_output_sparse_opening."
        )
