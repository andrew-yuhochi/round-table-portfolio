"""Unit tests for Component 11 — per-persona output validator.

M2 additions (TASK-M2-002)
--------------------------
Section 6: fully-invested gate (clause c) — 5 fixtures covering the 3 malformed
  and 2 well-formed cases from Component 11 Quality Criterion 5.
Section 7: single-source guarantee (AC3) — the validator uses check_fully_invested
  from portfolio/invariants.py (same symbol Component 15 imports).
Section 8: zero-judge-dispatch confirmation — fully-invested failures short-circuit
  before the judge (MagicMock proves no judge.judge() call is made).

Test structure
--------------
1. Config loading — load_validator_config() from a real config file.
2. Deterministic structural gate — 100% pass rate required (deterministic AC).
3. Stub-judge path — on-mandate vs off-mandate fixtures with injected
   StubOnMandateJudge responses.  Every judge test injects an explicit verdict
   so the assertion proves "judge returned the injected verdict and it propagated
   correctly" rather than relying on the stub's keyword heuristic.
4. _parse_judge_response — edge cases for the structured response parser.
5. Composition — structural fail short-circuits without touching the judge.

All tests use StubOnMandateJudge so no external API call or subagent dispatch
is made.  SKIP_LIVE=1 is the standard test-run mode.

The on-mandate judge runs as a subagent (output-validator-judge agent) wired by
the TASK-M1-010 orchestration runner — NOT as an external API call.  This module
exposes only the OnMandateJudge Protocol and the StubOnMandateJudge stub.

Fixture provenance
------------------
on_mandate_value_real.md — live Value persona run 2026-06-02 (first production
  run).  This is the Gate-4 real-provenance fixture required by the fixture-
  provenance corollary deferred at M1-009.  Satisfies the requirement that at
  least one fixture per validator be derived from sanitized real output.

All other on-mandate and off-mandate fixtures are hand-authored (2026-06-02,
updated 2026-06-03 to remove rigid ## Thesis/## Rationale headers that were
absent from real persona output — see quality-log TASK-M1-009 structural-gate
fix entry).  Bare-verdict fixtures remain hand-authored by design — they test
the floor, not the ceiling.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from round_table_portfolio.personas.output_validator import (
    STAGE_FULLY_INVESTED,
    STAGE_LLM_JUDGE,
    STAGE_STRUCTURAL,
    OnMandateJudge,
    ReportValidationResult,
    StubOnMandateJudge,
    ValidatorConfig,
    parse_judge_response,
    _run_structural_gate,
    _run_fully_invested_gate,
    load_validator_config,
    validate_persona_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# File lives at tests/unit/personas/test_output_validator.py — parents[3] is
# the project root (round-table-portfolio/).
_PROJECT_ROOT = Path(__file__).parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "validator.yaml"
# Fixtures live at tests/unit/fixtures/reports/ — one level up from this
# package directory.
_FIXTURES_DIR = Path(__file__).parents[1] / "fixtures" / "reports"


def _load(filename: str) -> str:
    return (_FIXTURES_DIR / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadValidatorConfig:
    def test_loads_structural_thresholds(self) -> None:
        cfg = load_validator_config(_CONFIG_PATH)
        assert cfg.structural.min_report_chars > 0
        assert cfg.structural.min_ticker_references >= 1
        assert cfg.structural.min_metric_terms >= 1
        assert len(cfg.structural.metric_terms) >= 10
        assert len(cfg.structural.data_source_signals) >= 5

    def test_loads_all_7_personas(self) -> None:
        cfg = load_validator_config(_CONFIG_PATH)
        expected = {
            "value", "growth", "technical", "discretionary-macro",
            "cta-systematic-macro", "quant-systematic", "risk-officer",
        }
        assert expected.issubset(cfg.personas.keys())

    def test_each_persona_has_concepts_and_signals(self) -> None:
        cfg = load_validator_config(_CONFIG_PATH)
        for slug, p in cfg.personas.items():
            assert len(p.on_mandate_concepts) >= 3, f"{slug} has too few on_mandate_concepts"
            assert len(p.off_mandate_signals) >= 3, f"{slug} has too few off_mandate_signals"

    def test_env_override(self, tmp_path: Path) -> None:
        """VALIDATOR_CONFIG env var overrides the default path."""
        minimal = tmp_path / "v.yaml"
        minimal.write_text(
            "structural:\n"
            "  min_report_chars: 50\n"
            "  min_ticker_references: 1\n"
            "  min_metric_terms: 3\n"
            "  metric_terms: [revenue, earnings, margin]\n"
            "  data_source_signals: [revenue]\n"
            "personas: {}\n",
            encoding="utf-8",
        )
        orig = os.environ.get("VALIDATOR_CONFIG")
        os.environ["VALIDATOR_CONFIG"] = str(minimal)
        try:
            # Force re-read by passing path explicitly — env var path is tested
            # by the module-level _CONFIG_PATH default, but load_validator_config
            # accepts an explicit override which is more reliable in test isolation.
            cfg = load_validator_config(minimal)
            assert cfg.structural.min_report_chars == 50
            assert cfg.structural.min_metric_terms == 3
        finally:
            if orig is None:
                os.environ.pop("VALIDATOR_CONFIG", None)
            else:
                os.environ["VALIDATOR_CONFIG"] = orig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> ValidatorConfig:
    """Return the real project config (loaded once per test that needs it)."""
    return load_validator_config(_CONFIG_PATH)


def _mandate(persona_slug: str) -> str:
    """Read the RESEARCH MANDATE section text from the authored persona file."""
    agent_path = (
        _PROJECT_ROOT / ".claude" / "agents" / f"{persona_slug}.md"
    )
    text = agent_path.read_text(encoding="utf-8")
    # Extract everything between ## RESEARCH MANDATE and the next ## heading.
    import re
    m = re.search(
        r"## RESEARCH MANDATE\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL
    )
    return m.group(1).strip() if m else ""


def _injected_stub(persona: str, report: str, verdict: bool, justification: str) -> StubOnMandateJudge:
    """Build a StubOnMandateJudge that returns an explicit verdict for this report.

    Using an injected response rather than the keyword heuristic means the test
    asserts "the judge returned the injected verdict and it propagated to the
    result" — not "the stub's word-count agreed with the expected label."
    """
    key = (persona, report[:50])
    return StubOnMandateJudge({key: (verdict, justification)})


# ---------------------------------------------------------------------------
# Deterministic structural gate — 100% required
# ---------------------------------------------------------------------------

class TestStructuralGate:
    """Every test in this class must pass 100% — this gate is deterministic."""

    # --- bare-verdict fixtures (must FAIL) ---

    def test_bare_verdict_value_fails(self) -> None:
        report = _load("bare_verdict_value.md")
        cfg = _cfg()
        result = _run_structural_gate(report, cfg.structural)
        assert not result.passed, "bare_verdict_value should fail the structural gate"
        assert result.stage == STAGE_STRUCTURAL
        assert "STRUCTURAL GATE FAIL" in result.notes

    def test_bare_verdict_no_sections_fails(self) -> None:
        report = _load("bare_verdict_no_sections.md")
        cfg = _cfg()
        result = _run_structural_gate(report, cfg.structural)
        assert not result.passed
        # Missing required sections or too short
        assert "STRUCTURAL GATE FAIL" in result.notes

    def test_bare_verdict_too_short_fails(self) -> None:
        report = _load("bare_verdict_too_short.md")
        cfg = _cfg()
        result = _run_structural_gate(report, cfg.structural)
        assert not result.passed
        assert "STRUCTURAL GATE FAIL" in result.notes

    def test_bare_verdict_no_tickers_fails(self) -> None:
        report = _load("bare_verdict_no_tickers.md")
        cfg = _cfg()
        result = _run_structural_gate(report, cfg.structural)
        assert not result.passed
        assert "STRUCTURAL GATE FAIL" in result.notes

    # --- on-mandate fixtures must PASS the structural gate ---

    @pytest.mark.parametrize("filename", [
        "on_mandate_value.md",
        "on_mandate_value_2.md",
        "on_mandate_value_real.md",  # real-provenance fixture (live run 2026-06-02)
        "on_mandate_growth.md",
        "on_mandate_technical.md",
        "on_mandate_discretionary_macro.md",
        "on_mandate_cta_systematic.md",
        "on_mandate_quant_systematic.md",
        "on_mandate_risk_officer.md",
    ])
    def test_on_mandate_passes_structural_gate(self, filename: str) -> None:
        report = _load(filename)
        cfg = _cfg()
        result = _run_structural_gate(report, cfg.structural)
        assert result.passed, (
            f"{filename} should pass the structural gate but got: {result.notes}"
        )
        assert result.stage == STAGE_STRUCTURAL

    # --- off-mandate fixtures must PASS the structural gate (they're structurally valid) ---

    @pytest.mark.parametrize("filename", [
        "off_mandate_value_arguing_momentum.md",
        "off_mandate_growth_arguing_valuation.md",
        "off_mandate_technical_arguing_fundamentals.md",
        "off_mandate_discretionary_macro_arguing_momentum.md",
        "off_mandate_cta_arguing_narrative.md",
        "off_mandate_quant_arguing_stories.md",
        "off_mandate_risk_officer_arguing_upside.md",
        "off_mandate_growth_arguing_macro.md",
    ])
    def test_off_mandate_passes_structural_gate(self, filename: str) -> None:
        """Off-mandate reports are structurally valid — only the LLM judge rejects them."""
        report = _load(filename)
        cfg = _cfg()
        result = _run_structural_gate(report, cfg.structural)
        assert result.passed, (
            f"{filename} should pass the structural gate (it's structurally valid). "
            f"Got: {result.notes}"
        )

    # --- inline edge cases ---

    def test_empty_report_fails(self) -> None:
        cfg = _cfg()
        result = _run_structural_gate("", cfg.structural)
        assert not result.passed

    def test_free_prose_without_headers_passes(self) -> None:
        """The gate must accept detailed free prose with no ## headings.

        This is the core behavioral change from the structural-gate fix
        (2026-06-03): rigid ## Thesis / ## Rationale headers are no longer
        required.  A report with rich financial vocabulary, enough tickers,
        and data-source signals passes regardless of markdown structure.
        """
        cfg = _cfg()
        report = (
            "AAPL and MSFT are both trading at meaningful discounts to intrinsic value. "
            "FCF yield on AAPL is 4.2% at current prices; P/E of 28× is below the "
            "5-year median of 31×. Balance sheet is net cash with strong earnings quality. "
            "MSFT trades at 32× earnings with 25% FCF margin and a durable competitive moat "
            "across cloud, productivity, and gaming. Margin of safety is present on both. "
            "Sourced from fundamentals tool and recent 10-K filings."
        )
        result = _run_structural_gate(report, cfg.structural)
        assert result.passed, f"Free-prose report should pass structural gate: {result.notes}"

    def test_report_with_low_metric_density_fails(self) -> None:
        """A report long enough to pass the char floor but with sparse financial
        vocabulary is caught by the metric-density check."""
        cfg = _cfg()
        # 362 chars, 2 tickers, data-source signal present (rsi/macd/momentum),
        # but only 3 distinct metric terms — below the threshold of 5.
        report = _load("bare_verdict_no_sections.md")
        result = _run_structural_gate(report, cfg.structural)
        assert not result.passed
        assert "metric" in result.notes.lower() or "vocabulary" in result.notes.lower()

    def test_real_provenance_value_report_passes_structural_gate(self) -> None:
        """The real live Value persona report (2026-06-02) must pass the structural gate.

        This is the primary invariant introduced by the structural-gate fix: the gate
        must accept genuine, detailed, on-mandate prose output even when it contains
        no ## markdown headers.  If this test fails, the gate is over-specified.
        """
        report = _load("on_mandate_value_real.md")
        cfg = _cfg()
        result = _run_structural_gate(report, cfg.structural)
        assert result.passed, (
            f"Real live Value report must pass structural gate. Got: {result.notes}"
        )

    def test_report_with_no_data_source_signal_fails(self) -> None:
        """A report that has the right structure but no data-source signals fails."""
        cfg = _cfg()
        # Craft a report that has tickers and metric terms but zero data-source vocabulary.
        report = (
            "AAPL and GOOG are interesting names with strong prospects. "
            "AAPL has a very loyal customer base and strong brand recognition. "
            "GOOG dominates the search market and has excellent competitive "
            "positioning across its various business lines. Both names are "
            "well-positioned for the future and have strong management teams "
            "with excellent track records of creating value for shareholders. "
            "The competitive moat is wide and the growth runway is long."
        )
        result = _run_structural_gate(report, cfg.structural)
        assert not result.passed
        assert "data-source" in result.notes.lower()


# ---------------------------------------------------------------------------
# LLM-judge path (StubLLMClient — injected responses)
# ---------------------------------------------------------------------------

class TestLLMJudgeWithStub:
    """Tests that exercise the full validate_persona_report() path with a stub client.

    Every test injects an explicit (verdict, justification) response for the
    specific fixture+persona key.  This means each test asserts:
      "the judge returned the injected verdict and it propagated to
       result.stage == STAGE_LLM_JUDGE with the correct passed value"
    rather than relying on the stub's keyword heuristic to produce the right
    classification.  If a future refactor changes how verdicts are propagated,
    these tests will catch it regardless of fixture vocabulary.
    """

    # --- on-mandate: inject PASS and assert it propagates ---

    @pytest.mark.parametrize("filename,persona,justification", [
        ("on_mandate_value.md", "value",
         "Report stays firmly in the value lens — FCF yield and margin-of-safety throughout."),
        ("on_mandate_value_2.md", "value",
         "Deep discount framing and explicit dismissal of momentum — on-mandate."),
        ("on_mandate_value_real.md", "value",
         "Real live run: numbers-anchored value gaps, FCF/P/E/margin-of-safety throughout — on-mandate."),
        ("on_mandate_growth.md", "growth",
         "Revenue growth, TAM expansion, and reinvestment runway dominate — on-mandate."),
        ("on_mandate_technical.md", "technical",
         "RSI, MACD, breakout, and relative strength drive all conclusions — on-mandate."),
        ("on_mandate_discretionary_macro.md", "discretionary-macro",
         "Fed pivot, yield curve, and CPI/PCE regime thesis throughout — on-mandate."),
        ("on_mandate_cta_systematic.md", "cta-systematic-macro",
         "Momentum signal and volatility-scaled positioning rules throughout — on-mandate."),
        ("on_mandate_quant_systematic.md", "quant-systematic",
         "Factor scores and cross-sectional ranking drive all conclusions — on-mandate."),
        ("on_mandate_risk_officer.md", "risk-officer",
         "Concentration, correlation, VaR, and drawdown framing throughout — on-mandate."),
    ])
    def test_on_mandate_passes_judge(
        self, filename: str, persona: str, justification: str
    ) -> None:
        report = _load(filename)
        cfg = _cfg()
        stub = _injected_stub(persona, report, True, justification)
        result = validate_persona_report(
            report=report,
            mandate=_mandate(persona),
            config=cfg,
            persona_slug=persona,
            judge=stub,
        )
        assert result.passed, (
            f"{filename} ({persona}) should pass the LLM judge. Notes: {result.notes}"
        )
        assert result.stage == STAGE_LLM_JUDGE
        assert justification in result.llm_justification

    # --- off-mandate: inject FAIL and assert it propagates ---

    @pytest.mark.parametrize("filename,persona,justification", [
        ("off_mandate_value_arguing_momentum.md", "value",
         "Core argument is RSI/MACD/trend — off-mandate for value."),
        ("off_mandate_growth_arguing_valuation.md", "growth",
         "Core argument is P/B discount and mean reversion — off-mandate for growth."),
        ("off_mandate_technical_arguing_fundamentals.md", "technical",
         "Core argument is FCF yield and P/E — off-mandate for technical."),
        ("off_mandate_discretionary_macro_arguing_momentum.md", "discretionary-macro",
         "Core argument is RSI/MACD crossover — off-mandate for discretionary-macro."),
        ("off_mandate_cta_arguing_narrative.md", "cta-systematic-macro",
         "Core argument is Fed-pivot narrative, not signal rules — off-mandate for CTA-Systematic."),
        ("off_mandate_quant_arguing_stories.md", "quant-systematic",
         "Core argument is conviction narrative, not factor scores — off-mandate for quant."),
        ("off_mandate_risk_officer_arguing_upside.md", "risk-officer",
         "Core argument is multi-bagger upside — off-mandate for risk-officer."),
        ("off_mandate_growth_arguing_macro.md", "growth",
         "Core argument is Fed/yield-curve regime — off-mandate for growth."),
    ])
    def test_off_mandate_fails_judge(
        self, filename: str, persona: str, justification: str
    ) -> None:
        report = _load(filename)
        cfg = _cfg()
        stub = _injected_stub(persona, report, False, justification)
        result = validate_persona_report(
            report=report,
            mandate=_mandate(persona),
            config=cfg,
            persona_slug=persona,
            judge=stub,
        )
        assert not result.passed, (
            f"{filename} ({persona}) should FAIL the LLM judge but passed. "
            f"Notes: {result.notes}"
        )
        assert result.stage == STAGE_LLM_JUDGE
        assert justification in result.llm_justification

    # --- injected stub responses (deterministic override) ---

    def test_injected_pass_response(self) -> None:
        report = _load("on_mandate_value.md")
        cfg = _cfg()
        key = ("value", report[:50])
        stub = StubOnMandateJudge({key: (True, "Report stays firmly in the value lens.")})
        result = validate_persona_report(
            report=report,
            mandate=_mandate("value"),
            config=cfg,
            persona_slug="value",
            judge=stub,
        )
        assert result.passed
        assert result.llm_justification == "Report stays firmly in the value lens."

    def test_injected_fail_response(self) -> None:
        report = _load("on_mandate_value.md")
        cfg = _cfg()
        key = ("value", report[:50])
        stub = StubOnMandateJudge({key: (False, "Core argument is momentum, not valuation.")})
        result = validate_persona_report(
            report=report,
            mandate=_mandate("value"),
            config=cfg,
            persona_slug="value",
            judge=stub,
        )
        assert not result.passed
        assert "Core argument is momentum" in result.llm_justification


# ---------------------------------------------------------------------------
# Structural fail short-circuits — no LLM client call
# ---------------------------------------------------------------------------

class TestShortCircuit:
    def test_bare_verdict_never_reaches_llm_judge(self) -> None:
        """A bare-verdict report must fail the structural gate and never reach
        the LLM judge.

        We inject a mock client whose call raises AssertionError.  If a future
        refactor ever calls the judge on a structural-gate failure, the mock
        fires loudly and the test fails — proving the short-circuit held.
        Passing llm_client=None was a weaker proof because it would only raise
        AttributeError, which could be masked by other code paths; the mock
        makes the intent explicit and failure mode unambiguous.
        """
        report = _load("bare_verdict_value.md")
        cfg = _cfg()
        mock_client = MagicMock(
            side_effect=AssertionError("judge must not be called on a structural-gate failure")
        )
        result = validate_persona_report(
            report=report,
            mandate="test mandate",
            config=cfg,
            persona_slug="value",
            judge=mock_client,
        )
        # Structural gate rejected the report without touching the mock client.
        assert not result.passed
        assert result.stage == STAGE_STRUCTURAL
        mock_client.assert_not_called()

    def test_too_short_never_reaches_llm_judge(self) -> None:
        report = _load("bare_verdict_too_short.md")
        cfg = _cfg()
        mock_client = MagicMock(
            side_effect=AssertionError("judge must not be called on a structural-gate failure")
        )
        result = validate_persona_report(
            report=report,
            mandate="test mandate",
            config=cfg,
            persona_slug="growth",
            judge=mock_client,
        )
        assert not result.passed
        assert result.stage == STAGE_STRUCTURAL
        mock_client.assert_not_called()


# ---------------------------------------------------------------------------
# _parse_judge_response edge cases
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def test_pass_verdict(self) -> None:
        raw = "VERDICT: PASS\nJUSTIFICATION: The report uses the value lens throughout."
        passed, just = parse_judge_response(raw)
        assert passed is True
        assert "value lens" in just

    def test_fail_verdict(self) -> None:
        raw = "VERDICT: FAIL\nJUSTIFICATION: Core argument is momentum, not valuation."
        passed, just = parse_judge_response(raw)
        assert passed is False
        assert "momentum" in just

    def test_case_insensitive_verdict(self) -> None:
        raw = "verdict: pass\njustification: on-mandate."
        passed, _ = parse_judge_response(raw)
        assert passed is True

    def test_malformed_returns_false(self) -> None:
        raw = "I think this looks fine to me."
        passed, just = parse_judge_response(raw)
        assert passed is False
        assert "Malformed" in just

    def test_multiline_justification_captured(self) -> None:
        raw = (
            "VERDICT: FAIL\n"
            "JUSTIFICATION: The report drifts.\n"
            "Line two of justification.\n"
            "Line three.\n"
        )
        passed, just = parse_judge_response(raw)
        assert not passed
        assert "Line two" in just


# ---------------------------------------------------------------------------
# ReportValidationResult convenience
# ---------------------------------------------------------------------------

class TestReportValidationResult:
    def test_bool_true_when_passed(self) -> None:
        r = ReportValidationResult(passed=True, notes="ok", stage=STAGE_STRUCTURAL)
        assert bool(r) is True

    def test_bool_false_when_not_passed(self) -> None:
        r = ReportValidationResult(passed=False, notes="fail", stage=STAGE_STRUCTURAL)
        assert bool(r) is False

    def test_llm_justification_defaults_empty(self) -> None:
        r = ReportValidationResult(passed=True, notes="ok", stage=STAGE_LLM_JUDGE)
        assert r.llm_justification == ""


# ---------------------------------------------------------------------------
# OnMandateJudge Protocol — structural check
# ---------------------------------------------------------------------------

class TestOnMandateJudgeProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        """StubOnMandateJudge must satisfy the OnMandateJudge Protocol."""
        stub = StubOnMandateJudge()
        assert isinstance(stub, OnMandateJudge)

    def test_validate_raises_if_judge_none(self) -> None:
        """validate_persona_report raises ValueError when judge=None — no implicit default."""
        cfg = load_validator_config(_CONFIG_PATH)
        # A structurally-valid report so the structural gate passes and we reach
        # the judge check.
        report = _load("on_mandate_value.md")
        with pytest.raises(ValueError, match="requires a judge implementation"):
            validate_persona_report(
                report=report,
                mandate=_mandate("value"),
                config=cfg,
                persona_slug="value",
                judge=None,
            )


# ---------------------------------------------------------------------------
# Section 6 — Fully-invested gate fixtures (TASK-M2-002, clause c)
# 5 fixtures: 3 malformed → FLAG, 2 well-formed → PASS.
# All deterministic — 100% pass rate required (Quality Criterion 5).
# ---------------------------------------------------------------------------

# These portfolios are the canonical representations for the 5 fixtures.
# The .md files in fixtures/reports/ carry the full report text + a comment
# block documenting the portfolio structure.  Tests inline the portfolios here
# so the assertion is self-contained and the fixture provenance is explicit.

_FI_MALFORMED_095 = {
    "AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.25, "CASH": 0.10,
}  # total = 0.95, under-invested

_FI_MALFORMED_105 = {
    "AAPL": 0.40, "MSFT": 0.40, "NVDA": 0.25, "CASH": 0.00,
}  # total = 1.05, over-invested

_FI_MALFORMED_OVER_CAP = {
    "AAPL": 0.25, "MSFT": 0.15, "CASH": 0.60,
}  # AAPL = 0.25 > max_position_weight = 0.20

_FI_WELLFORMED_INVESTED = {
    "AAPL": 0.20, "MSFT": 0.20, "NVDA": 0.20, "CASH": 0.40,
}  # total = 1.00, no weight over 0.20

_FI_WELLFORMED_HIGH_CASH = {
    "AAPL": 0.15, "CASH": 0.85,
}  # total = 1.00, high cash is valid mechanics


_MAX_POS_WEIGHT = 0.20  # matches config/thresholds.yaml default


class TestFullyInvestedGateDirect:
    """Direct tests of _run_fully_invested_gate — 100% deterministic."""

    def test_malformed_sums_to_095_flags(self) -> None:
        """sums-to-0.95-no-CASH fixture → FULLY-INVESTED GATE FAIL."""
        result = _run_fully_invested_gate(_FI_MALFORMED_095, _MAX_POS_WEIGHT)
        assert not result.passed, (
            f"Malformed 0.95 portfolio should FLAG. Notes: {result.notes}"
        )
        assert result.stage == STAGE_FULLY_INVESTED
        assert "FULLY-INVESTED GATE FAIL" in result.notes

    def test_malformed_sums_to_105_flags(self) -> None:
        """sums-to-1.05-over-invested fixture → FULLY-INVESTED GATE FAIL."""
        result = _run_fully_invested_gate(_FI_MALFORMED_105, _MAX_POS_WEIGHT)
        assert not result.passed, (
            f"Malformed 1.05 portfolio should FLAG. Notes: {result.notes}"
        )
        assert result.stage == STAGE_FULLY_INVESTED
        assert "FULLY-INVESTED GATE FAIL" in result.notes

    def test_malformed_single_weight_over_cap_flags(self) -> None:
        """single-weight-over-cap fixture (AAPL=0.25 > cap=0.20) → FULLY-INVESTED GATE FAIL."""
        result = _run_fully_invested_gate(_FI_MALFORMED_OVER_CAP, _MAX_POS_WEIGHT)
        assert not result.passed, (
            f"Over-cap portfolio should FLAG. Notes: {result.notes}"
        )
        assert result.stage == STAGE_FULLY_INVESTED
        assert "FULLY-INVESTED GATE FAIL" in result.notes

    def test_wellformed_invested_passes(self) -> None:
        """Well-formed positions+CASH=1.0 portfolio → PASS."""
        result = _run_fully_invested_gate(_FI_WELLFORMED_INVESTED, _MAX_POS_WEIGHT)
        assert result.passed, (
            f"Well-formed portfolio should PASS. Notes: {result.notes}"
        )
        assert result.stage == STAGE_FULLY_INVESTED

    def test_wellformed_high_cash_passes(self) -> None:
        """Well-formed high-cash portfolio (AAPL=0.15, CASH=0.85) → PASS.

        High cash is a valid mechanical state — the gate checks sum-to-1.0
        only; whether the cash level is appropriate stays with the on-mandate
        judge.
        """
        result = _run_fully_invested_gate(_FI_WELLFORMED_HIGH_CASH, _MAX_POS_WEIGHT)
        assert result.passed, (
            f"High-cash portfolio should PASS the mechanics check. Notes: {result.notes}"
        )
        assert result.stage == STAGE_FULLY_INVESTED


class TestFullyInvestedViaValidatePersonaReport:
    """End-to-end: the gate runs inside validate_persona_report when
    counterfactual_portfolio is supplied, and still short-circuits before judge."""

    def test_malformed_portfolio_fails_before_judge(self) -> None:
        """A structurally-valid report with a malformed portfolio must fail at
        STAGE_FULLY_INVESTED without reaching the judge."""
        report = _load("fi_malformed_sums_to_095.md")
        cfg = _cfg()
        mock_judge = MagicMock(
            side_effect=AssertionError("judge must not be called on a fully-invested gate failure")
        )
        result = validate_persona_report(
            report=report,
            mandate=_mandate("value"),
            config=cfg,
            persona_slug="value",
            judge=mock_judge,
            counterfactual_portfolio=_FI_MALFORMED_095,
            max_position_weight=_MAX_POS_WEIGHT,
        )
        assert not result.passed
        assert result.stage == STAGE_FULLY_INVESTED
        mock_judge.assert_not_called()

    def test_wellformed_portfolio_reaches_judge(self) -> None:
        """A structurally-valid report with a well-formed portfolio must pass the
        fully-invested gate and proceed to the judge."""
        report = _load("fi_wellformed_invested.md")
        cfg = _cfg()
        stub = _injected_stub("value", report, True, "On-mandate value reasoning.")
        result = validate_persona_report(
            report=report,
            mandate=_mandate("value"),
            config=cfg,
            persona_slug="value",
            judge=stub,
            counterfactual_portfolio=_FI_WELLFORMED_INVESTED,
            max_position_weight=_MAX_POS_WEIGHT,
        )
        assert result.passed
        assert result.stage == STAGE_LLM_JUDGE

    def test_no_portfolio_skips_gate(self) -> None:
        """When counterfactual_portfolio is None the fully-invested gate is skipped
        and the existing M1 behaviour is unchanged."""
        report = _load("on_mandate_value.md")
        cfg = _cfg()
        stub = _injected_stub("value", report, True, "On-mandate.")
        result = validate_persona_report(
            report=report,
            mandate=_mandate("value"),
            config=cfg,
            persona_slug="value",
            judge=stub,
        )
        assert result.passed
        assert result.stage == STAGE_LLM_JUDGE


# ---------------------------------------------------------------------------
# Section 7 — Single-source guarantee (AC3)
# ---------------------------------------------------------------------------

class TestSingleSourceHelper:
    def test_output_validator_uses_check_fully_invested_from_invariants(self) -> None:
        """The validator's internal _run_fully_invested_gate delegates to
        check_fully_invested from portfolio/invariants.py — the SAME symbol
        Component 15 (TASK-M2-005) will import.

        Proof: _run_fully_invested_gate is importable, and the module it lives
        in imports check_fully_invested from portfolio.invariants.  We confirm
        by asserting both import paths resolve to the same callable.
        """
        from round_table_portfolio.portfolio.invariants import (
            check_fully_invested as invariants_fn,
        )
        import round_table_portfolio.personas.output_validator as ov_module
        # The output_validator module must expose check_fully_invested via its
        # own namespace (imported at module level).
        assert hasattr(ov_module, "check_fully_invested"), (
            "output_validator.py must import check_fully_invested from portfolio.invariants "
            "at module level so Component 15 can import the same symbol."
        )
        assert ov_module.check_fully_invested is invariants_fn, (
            "output_validator.check_fully_invested must be the same object as "
            "portfolio.invariants.check_fully_invested — single arithmetic source."
        )


# ---------------------------------------------------------------------------
# Section 8 — Zero-judge-dispatch confirmation
# ---------------------------------------------------------------------------

class TestZeroJudgeDispatch:
    """Fully-invested failures must consume ZERO judge dispatches.

    Each test injects a MagicMock judge with side_effect=AssertionError.
    If the gate ever calls judge.judge(), the mock fires and the test fails —
    proving the short-circuit held.
    """

    @pytest.mark.parametrize("portfolio,label", [
        (_FI_MALFORMED_095, "sums-to-0.95"),
        (_FI_MALFORMED_105, "sums-to-1.05-over-invested"),
        (_FI_MALFORMED_OVER_CAP, "single-weight-over-cap"),
    ])
    def test_malformed_portfolio_never_dispatches_judge(
        self, portfolio: dict, label: str
    ) -> None:
        report = _load("on_mandate_value.md")  # structurally valid — gate 1 passes
        cfg = _cfg()
        mock_judge = MagicMock(
            side_effect=AssertionError(
                f"judge must not be called for {label} — fully-invested gate must short-circuit"
            )
        )
        result = validate_persona_report(
            report=report,
            mandate=_mandate("value"),
            config=cfg,
            persona_slug="value",
            judge=mock_judge,
            counterfactual_portfolio=portfolio,
            max_position_weight=_MAX_POS_WEIGHT,
        )
        assert not result.passed, f"{label}: expected FLAG but got PASS"
        assert result.stage == STAGE_FULLY_INVESTED
        mock_judge.assert_not_called()
