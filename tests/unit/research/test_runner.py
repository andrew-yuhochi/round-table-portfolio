"""Unit tests for research/runner.py — Component 9.

Fixture provenance: the realistic RESEARCH OUTPUT SCHEMA JSON is adapted from
the Value persona's own schema example (value.md RESEARCH OUTPUT SCHEMA block),
with plausible tickers/rationale representative of a real value-investor run.
No live subagent is dispatched; StubOnMandateJudge is injected throughout.
All tests run with SKIP_LIVE=1 (no network calls).

Coverage matrix:
  happy-path              — parse + report written + shortlist rows + budget + validator verdict
  cluster expansion       — cluster peers produce is_cluster_peer=1 rows
  budget overrun          — web-search breach flagged
  data tool overrun       — data_tool_calls_used > cap flagged in summary
  validator pass          — validator_passed=1 in payload
  validator fail          — validator structural fail propagated into payload
  empty shortlist         — Major-tier warning, result still returned
  malformed JSON          — PersonaOutputParseError raised
  missing required key    — PersonaOutputParseError raised
  wrong type counts       — PersonaOutputParseError raised
  markdown fence stripping — code-fenced JSON parsed correctly
  demo_scaffold_insert    — rows written to a temp ledger.db with correct shape
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from round_table_portfolio.budget.loader import PersonaBudget
from round_table_portfolio.personas.output_validator import (
    StubOnMandateJudge,
    ValidatorConfig,
    StructuralConfig,
    PersonaConfig,
)
from round_table_portfolio.research.runner import (
    PersonaOutputParseError,
    PersonaResearchResult,
    _build_shortlist_rows,
    _parse_persona_output,
    demo_scaffold_insert,
    run_persona_research,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Realistic RESEARCH OUTPUT SCHEMA — adapted from value.md's own example.
# This is a representative sample from a value-investor run on 2026-06-02.
# Tickers chosen from S&P 500 large-cap value names; rationale matches the
# value mandate (FCF yield, balance-sheet, margin-of-safety language).
VALID_VALUE_OUTPUT: dict = {
    "shortlist": [
        {
            "ticker": "BRK",
            "why": (
                "Trading at ~1.3× book with $150B cash fortress and no debt. "
                "Normalized FCF yield ~8%. Intrinsic value estimate: $380-420/share. "
                "Current price $330 represents a 15–25% discount — solid margin of safety."
            ),
            "cluster": ["MKL", "FNMA"],
        },
        {
            "ticker": "GOOG",
            "why": (
                "Search + Cloud moat intact. TTM FCF yield 5.2% on enterprise value. "
                "P/E 20× on normalized earnings with $100B net cash. "
                "Market paying tech-narrative premium; I see a durable cash machine at discount."
            ),
            "cluster": ["META", "SNAP"],
        },
        {
            "ticker": "CVX",
            "why": (
                "Integrated oil at 6× normalized earnings, 4% dividend yield, "
                "AA balance sheet. Capex discipline post-2020 reset. "
                "Energy inflation hedge; deep discount to NAV on 10-year oil curve."
            ),
            "cluster": ["XOM", "COP"],
        },
    ],
    "report": (
        "## Executive Summary\n\n"
        "This week's value screen surfaces three names with defensible margins of safety: "
        "BRK, GOOG, and CVX. Each trades at a discount to a conservative intrinsic-value "
        "estimate, with strong balance sheets and durable earnings power.\n\n"
        "## BRK — Berkshire Hathaway\n\n"
        "Berkshire trades at 1.3× book value, a historically cheap entry for a business "
        "compounding book at 10–12% per annum. The $150B cash balance provides both a "
        "margin-of-safety cushion and optionality for a large acquisition. FCF yield "
        "on operating earnings is approximately 8%. My intrinsic-value range is $380–420 "
        "per B-share equivalent; at $330, the discount is 15–25%. I would ADD at current "
        "prices and increase position if the stock falls further.\n\n"
        "Sources: 2025 10-K (EDGAR), Berkshire Q4 earnings release (web search), "
        "Finnhub fundamentals (P/B, net cash).\n\n"
        "## GOOG — Alphabet\n\n"
        "Alphabet's Search moat remains intact despite AI-search noise. The business "
        "generates $70B+ in annual FCF; at $1.6T enterprise value the FCF yield is 5.2%. "
        "Net cash position of ~$100B provides downside protection. Normalized P/E of 20× "
        "is reasonable for a business with this FCF conversion rate. The market's fear "
        "of AI disruption is, in my view, creating a valuation gap. ADD on weakness.\n\n"
        "Sources: 10-Q filings (EDGAR), web search for AI competition narrative.\n\n"
        "## CVX — Chevron\n\n"
        "Chevron's integrated model and AA balance sheet make it the quality anchor in "
        "energy. At 6× normalized earnings (using $65 Brent assumption) and a 4% "
        "dividend yield, CVX offers both income and upside. The balance sheet has been "
        "rebuilt since 2020: net debt/EBITDA is sub-1×. I am building a position "
        "incrementally given oil-price uncertainty.\n\n"
        "Sources: Finnhub fundamentals, FRED macro (oil price series), web search.\n\n"
        "## Risk Factors\n\n"
        "Primary risk: macro slowdown compresses earnings across all three names. "
        "Secondary risk: GOOG faces regulatory action. I am comfortable with these "
        "risks at current valuations — wider safety margins than usual.\n\n"
        "## Conclusion\n\n"
        "Top conviction: BRK (largest discount, best balance sheet). "
        "Second: CVX (income + value). Third: GOOG (growth-at-a-reasonable-price).\n"
    ),
    "web_searches_used": 4,
    "data_tool_calls_used": 11,
}

VALID_VALUE_JSON: str = json.dumps(VALID_VALUE_OUTPUT)


@pytest.fixture()
def value_budget() -> PersonaBudget:
    return PersonaBudget(max_turns=12, max_web_searches=6, max_data_tool_calls=18)


@pytest.fixture()
def validator_config() -> ValidatorConfig:
    """Minimal ValidatorConfig sufficient for happy-path tests."""
    return ValidatorConfig(
        structural=StructuralConfig(
            min_report_chars=200,
            min_ticker_references=3,
            min_metric_terms=3,
            metric_terms=("p/e", "fcf", "earnings", "margin", "valuation"),
            data_source_signals=("p/e", "fcf", "10-k", "earnings", "finnhub", "edgar"),
        ),
        personas={
            "value": PersonaConfig(
                on_mandate_concepts=("fcf yield", "margin of safety", "p/e", "intrinsic value"),
                off_mandate_signals=("momentum", "rsi", "macd", "trend"),
            )
        },
    )


@pytest.fixture()
def stub_judge_pass() -> StubOnMandateJudge:
    """Judge that always returns PASS for the value fixture."""
    key = ("value", VALID_VALUE_OUTPUT["report"][:50])
    return StubOnMandateJudge({key: (True, "On-mandate: all reasoning anchored in valuation.")})


@pytest.fixture()
def stub_judge_fail() -> StubOnMandateJudge:
    """Judge that always returns FAIL for the value fixture."""
    key = ("value", VALID_VALUE_OUTPUT["report"][:50])
    return StubOnMandateJudge({key: (False, "Off-mandate: report argues through momentum lens.")})


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------


class TestParsePersonaOutput:
    def test_happy_path_parses_all_fields(self):
        parsed = _parse_persona_output(VALID_VALUE_JSON)
        assert len(parsed.shortlist) == 3
        assert parsed.shortlist[0].ticker == "BRK"
        assert "FCF" in parsed.shortlist[0].why
        assert parsed.shortlist[0].cluster == ["MKL", "FNMA"]
        assert parsed.web_searches_used == 4
        assert parsed.data_tool_calls_used == 11

    def test_tickers_uppercased(self):
        lowercased = json.dumps({**VALID_VALUE_OUTPUT,
            "shortlist": [{"ticker": "brk", "why": "x", "cluster": ["mkl"]}]})
        parsed = _parse_persona_output(lowercased)
        assert parsed.shortlist[0].ticker == "BRK"
        assert parsed.shortlist[0].cluster == ["MKL"]

    def test_strips_markdown_code_fence(self):
        fenced = "```json\n" + VALID_VALUE_JSON + "\n```"
        parsed = _parse_persona_output(fenced)
        assert parsed.shortlist[0].ticker == "BRK"

    def test_strips_plain_code_fence(self):
        fenced = "```\n" + VALID_VALUE_JSON + "\n```"
        parsed = _parse_persona_output(fenced)
        assert len(parsed.shortlist) == 3

    def test_empty_cluster_allowed(self):
        data = {**VALID_VALUE_OUTPUT,
                "shortlist": [{"ticker": "AAPL", "why": "good", "cluster": []}]}
        parsed = _parse_persona_output(json.dumps(data))
        assert parsed.shortlist[0].cluster == []

    def test_raises_on_invalid_json(self):
        with pytest.raises(PersonaOutputParseError, match="not valid JSON"):
            _parse_persona_output("{not valid json}")

    def test_raises_on_missing_shortlist(self):
        data = {k: v for k, v in VALID_VALUE_OUTPUT.items() if k != "shortlist"}
        with pytest.raises(PersonaOutputParseError, match="Missing required keys"):
            _parse_persona_output(json.dumps(data))

    def test_raises_on_missing_report(self):
        data = {k: v for k, v in VALID_VALUE_OUTPUT.items() if k != "report"}
        with pytest.raises(PersonaOutputParseError, match="Missing required keys"):
            _parse_persona_output(json.dumps(data))

    def test_raises_on_missing_counts(self):
        data = {k: v for k, v in VALID_VALUE_OUTPUT.items()
                if k not in ("web_searches_used", "data_tool_calls_used")}
        with pytest.raises(PersonaOutputParseError, match="Missing required keys"):
            _parse_persona_output(json.dumps(data))

    def test_raises_on_non_list_shortlist(self):
        data = {**VALID_VALUE_OUTPUT, "shortlist": "not a list"}
        with pytest.raises(PersonaOutputParseError, match="must be a JSON array"):
            _parse_persona_output(json.dumps(data))

    def test_raises_on_shortlist_item_without_ticker(self):
        data = {**VALID_VALUE_OUTPUT,
                "shortlist": [{"why": "no ticker", "cluster": []}]}
        with pytest.raises(PersonaOutputParseError, match="missing 'ticker'"):
            _parse_persona_output(json.dumps(data))

    def test_raises_on_non_integer_counts(self):
        data = {**VALID_VALUE_OUTPUT, "web_searches_used": "not an int"}
        with pytest.raises(PersonaOutputParseError, match="must be integers"):
            _parse_persona_output(json.dumps(data))


# ---------------------------------------------------------------------------
# Shortlist rows builder tests
# ---------------------------------------------------------------------------


class TestBuildShortlistRows:
    def test_direct_entries_are_not_cluster_peers(self):
        parsed = _parse_persona_output(VALID_VALUE_JSON)
        rows = _build_shortlist_rows(parsed, "value", "2026-06-02")
        direct = {r.ticker: r for r in rows if r.is_cluster_peer == 0}
        assert set(direct.keys()) == {"BRK", "GOOG", "CVX"}
        for r in direct.values():
            assert r.parent_ticker is None

    def test_cluster_peers_reference_parent(self):
        parsed = _parse_persona_output(VALID_VALUE_JSON)
        rows = _build_shortlist_rows(parsed, "value", "2026-06-02")
        peers = {r.ticker: r for r in rows if r.is_cluster_peer == 1}
        # BRK cluster: MKL, FNMA; GOOG cluster: META, SNAP; CVX cluster: XOM, COP
        assert peers["MKL"].parent_ticker == "BRK"
        assert peers["FNMA"].parent_ticker == "BRK"
        assert peers["META"].parent_ticker == "GOOG"
        assert peers["SNAP"].parent_ticker == "GOOG"
        assert peers["XOM"].parent_ticker == "CVX"
        assert peers["COP"].parent_ticker == "CVX"

    def test_direct_entry_wins_over_cluster_peer_dedup(self):
        """If GOOG appears as both a direct shortlist entry and a cluster peer, direct wins."""
        data = {
            **VALID_VALUE_OUTPUT,
            "shortlist": [
                {"ticker": "GOOG", "why": "direct", "cluster": []},
                {"ticker": "MSFT", "why": "direct2", "cluster": ["GOOG"]},
            ],
        }
        parsed = _parse_persona_output(json.dumps(data))
        rows = _build_shortlist_rows(parsed, "value", "2026-06-02")
        goog_row = next(r for r in rows if r.ticker == "GOOG")
        assert goog_row.is_cluster_peer == 0
        assert goog_row.parent_ticker is None

    def test_total_row_count(self):
        parsed = _parse_persona_output(VALID_VALUE_JSON)
        rows = _build_shortlist_rows(parsed, "value", "2026-06-02")
        # 3 direct + 6 cluster peers (2 per direct)
        assert len(rows) == 9


# ---------------------------------------------------------------------------
# run_persona_research integration tests
# ---------------------------------------------------------------------------


class TestRunPersonaResearch:
    def test_report_written_to_correct_path(self, tmp_path, value_budget, validator_config,
                                            stub_judge_pass):
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        expected = tmp_path / "reports" / "2026-06-02" / "value.md"
        assert expected.exists()
        assert result.parsed_output.report in expected.read_text(encoding="utf-8")

    def test_shortlist_payload_correct(self, tmp_path, value_budget, validator_config,
                                       stub_judge_pass):
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        tickers = {r.ticker for r in result.shortlist_rows}
        assert {"BRK", "GOOG", "CVX"}.issubset(tickers)
        # cluster peers present
        assert "MKL" in tickers
        assert "META" in tickers

    def test_validator_pass_propagated_to_payload(self, tmp_path, value_budget,
                                                   validator_config, stub_judge_pass):
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        assert result.report_payload.validator_passed == 1
        assert result.validation.passed is True

    def test_validator_fail_propagated_to_payload(self, tmp_path, value_budget,
                                                   validator_config, stub_judge_fail):
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_fail,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        assert result.report_payload.validator_passed == 0
        assert result.validation.passed is False

    def test_budget_counts_recorded(self, tmp_path, value_budget, validator_config,
                                    stub_judge_pass):
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        # web_searches_used=4 < cap=6 → no breach
        assert result.budget_summary["web_searches"]["used"] == 4
        assert result.budget_summary["web_searches"]["breach"] is False
        assert result.budget_overrun is False

    def test_web_search_overrun_flagged(self, tmp_path, validator_config, stub_judge_pass):
        budget = PersonaBudget(max_turns=12, max_web_searches=2, max_data_tool_calls=18)
        # web_searches_used=4 > cap=2 → breach
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        assert result.budget_summary["web_searches"]["breach"] is True
        assert result.budget_overrun is True

    def test_data_tool_overrun_flagged(self, tmp_path, validator_config, stub_judge_pass):
        budget = PersonaBudget(max_turns=12, max_web_searches=6, max_data_tool_calls=5)
        # data_tool_calls_used=11 > cap=5 → flagged
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        assert result.budget_summary["data_tool_calls"]["used"] == 11
        assert result.budget_overrun is True

    def test_empty_shortlist_returns_result_with_warning(self, tmp_path, value_budget,
                                                          validator_config, stub_judge_pass):
        """Empty shortlist is Major-tier but the runner still returns a result."""
        data = {**VALID_VALUE_OUTPUT, "shortlist": []}
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=json.dumps(data),
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        assert result.shortlist_rows == []
        # The run still completes; caller decides whether to re-prompt.
        assert result.persona_slug == "value"

    def test_malformed_json_raises(self, tmp_path, value_budget, validator_config,
                                   stub_judge_pass):
        with pytest.raises(PersonaOutputParseError):
            run_persona_research(
                persona_slug="value",
                week_id="2026-06-02",
                raw_output="{bad json",
                mandate="Research the universe for value.",
                judge=stub_judge_pass,
                budget=value_budget,
                validator_config=validator_config,
                state_root=tmp_path,
            )

    def test_report_payload_fields_populated(self, tmp_path, value_budget,
                                              validator_config, stub_judge_pass):
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        rp = result.report_payload
        assert rp.week_id == "2026-06-02"
        assert rp.persona == "value"
        assert len(rp.summary) > 0
        assert "value.md" in rp.full_report_path
        assert rp.roster_version == 1
        assert rp.enhancement_version == 1
        assert rp.user_id == "andrew"

    def test_structural_fail_short_circuits_judge(self, tmp_path, value_budget,
                                                   stub_judge_pass):
        """A structurally-failing report never reaches the judge."""
        # Config with very high min_report_chars so the fixture fails structurally.
        strict_config = ValidatorConfig(
            structural=StructuralConfig(
                min_report_chars=999_999,
                min_ticker_references=3,
                min_metric_terms=0,
                metric_terms=(),
                data_source_signals=("fcf",),
            ),
            personas={},
        )
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,  # would return PASS if called; should NOT be called
            budget=value_budget,
            validator_config=strict_config,
            state_root=tmp_path,
        )
        # Structural fail → validator_passed=0, stage="structural"
        assert result.report_payload.validator_passed == 0
        assert result.validation.stage == "structural"

    def test_markdown_fenced_output_parsed(self, tmp_path, value_budget,
                                           validator_config, stub_judge_pass):
        fenced = "```json\n" + VALID_VALUE_JSON + "\n```"
        result = run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=fenced,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )
        assert result.parsed_output.shortlist[0].ticker == "BRK"


# ---------------------------------------------------------------------------
# Demo scaffold insert tests
# ---------------------------------------------------------------------------


class TestDemoScaffoldInsert:
    @pytest.fixture()
    def fresh_ledger(self, tmp_path) -> Path:
        """Apply the project schema to a fresh in-memory DB file."""
        schema_path = (
            Path(__file__).parents[3]
            / "src" / "round_table_portfolio" / "storage" / "schema.sql"
        )
        db_path = tmp_path / "ledger.db"
        sql = schema_path.read_text(encoding="utf-8")
        with sqlite3.connect(db_path) as conn:
            conn.executescript(sql)
        return db_path

    def _make_result(self, tmp_path, value_budget, validator_config,
                     stub_judge_pass) -> PersonaResearchResult:
        return run_persona_research(
            persona_slug="value",
            week_id="2026-06-02",
            raw_output=VALID_VALUE_JSON,
            mandate="Research the universe for value.",
            judge=stub_judge_pass,
            budget=value_budget,
            validator_config=validator_config,
            state_root=tmp_path,
        )

    def test_persona_reports_row_inserted(self, tmp_path, fresh_ledger, value_budget,
                                           validator_config, stub_judge_pass):
        result = self._make_result(tmp_path, value_budget, validator_config, stub_judge_pass)
        demo_scaffold_insert(result, fresh_ledger)

        with sqlite3.connect(fresh_ledger) as conn:
            row = conn.execute(
                "SELECT persona, validator_passed, validator_notes, full_report_path "
                "FROM persona_reports WHERE week_id=? AND persona=?",
                ("2026-06-02", "value"),
            ).fetchone()
        assert row is not None
        persona, vp, vn, frp = row
        assert persona == "value"
        assert vp == 1  # stub_judge_pass → PASS
        assert "value.md" in frp

    def test_persona_shortlist_rows_inserted(self, tmp_path, fresh_ledger, value_budget,
                                              validator_config, stub_judge_pass):
        result = self._make_result(tmp_path, value_budget, validator_config, stub_judge_pass)
        demo_scaffold_insert(result, fresh_ledger)

        with sqlite3.connect(fresh_ledger) as conn:
            rows = conn.execute(
                "SELECT ticker, is_cluster_peer, parent_ticker "
                "FROM persona_shortlists WHERE week_id=? AND persona=?",
                ("2026-06-02", "value"),
            ).fetchall()
        tickers = {r[0] for r in rows}
        assert "BRK" in tickers
        assert "GOOG" in tickers
        assert "MKL" in tickers  # cluster peer of BRK

        # Check is_cluster_peer shape.
        direct = {r[0]: r for r in rows if r[1] == 0}
        peers = {r[0]: r for r in rows if r[1] == 1}
        assert "BRK" in direct
        assert "MKL" in peers
        assert peers["MKL"][2] == "BRK"  # parent_ticker

    def test_idempotent_insert(self, tmp_path, fresh_ledger, value_budget,
                                validator_config, stub_judge_pass):
        """Inserting twice does not create duplicate rows (INSERT OR REPLACE)."""
        result = self._make_result(tmp_path, value_budget, validator_config, stub_judge_pass)
        demo_scaffold_insert(result, fresh_ledger)
        demo_scaffold_insert(result, fresh_ledger)

        with sqlite3.connect(fresh_ledger) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM persona_reports WHERE week_id=? AND persona=?",
                ("2026-06-02", "value"),
            ).fetchone()[0]
        assert count == 1

    def test_schema_conformance_all_columns_populated(self, tmp_path, fresh_ledger,
                                                       value_budget, validator_config,
                                                       stub_judge_pass):
        """No NOT NULL violation; FK columns carry expected values."""
        result = self._make_result(tmp_path, value_budget, validator_config, stub_judge_pass)
        demo_scaffold_insert(result, fresh_ledger)

        with sqlite3.connect(fresh_ledger) as conn:
            row = conn.execute(
                "SELECT user_id, roster_version, enhancement_version "
                "FROM persona_reports WHERE week_id=? AND persona=?",
                ("2026-06-02", "value"),
            ).fetchone()
        user_id, rv, ev = row
        assert user_id == "andrew"
        assert rv == 1
        assert ev == 1
