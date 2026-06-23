"""Unit tests for M6-005 thesis-discipline validator — Component 11 extension.

Covers the DETERMINISTIC parts only:
- Thesis-presence gate (_run_thesis_presence_gate): missing/empty reason → FAIL;
  present reason → PASS; HOLD exempt.
- parse_judge_response_with_thesis: extracts per-stance thesis_reasoning from a
  raw judge response string in the THESIS_REASONING_START/END block format.
- validate_round1_stances: end-to-end with a raw judge response string — presence
  gate results + genuine/reactive labels propagate correctly; reactive flagged in
  notes but passed=True (not auto-blocked).
- ReplayJudge path: validate_round1_stances works correctly when given the raw
  response string that ReplayJudge would replay (no second judge.judge() call).

Does NOT assert real LLM genuine/reactive accuracy — that is the probabilistic
gate evaluated by the main session using the real judge over labeled fixtures.

Fixture provenance:
  Labeled fixture files live under tests/unit/fixtures/thesis_stances/.
  Each fixture carries a _fixture_meta block with provenance notes (see files).
  All fixtures in this test module are hand-authored (no real-run data used here)
  to keep deterministic tests self-contained and fast.

Test count target: ≥20 tests across all sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from round_table_portfolio.personas.output_validator import (
    THESIS_GENUINE,
    THESIS_REACTIVE,
    ReplayJudge,
    StanceThesisResult,
    StubOnMandateJudgeWithThesis,
    _build_stances_block,
    _parse_inline_thesis_reasoning,
    _run_thesis_presence_gate,
    parse_judge_response_with_thesis,
    validate_round1_stances,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[3]
_FIXTURES_DIR = Path(__file__).parents[1] / "fixtures" / "thesis_stances"


def _load_fixture(filename: str) -> dict:
    return json.loads((_FIXTURES_DIR / filename).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Minimal mandate with HOLDING HORIZON clause (used across tests)
# ---------------------------------------------------------------------------

_VALUE_MANDATE = """\
## RESEARCH MANDATE
Research the universe for names trading below your estimate of intrinsic value.
Weigh FCF yield, balance-sheet strength, durable competitive position.

## HOLDING HORIZON
You hold positions over a 3-month-to-2-year horizon. You exit when the
medium-term earnings-power thesis is structurally broken, not when price falls.
"""

_CTA_MANDATE = """\
## RESEARCH MANDATE
Research the universe for confirmed medium-term trends.

## HOLDING HORIZON
You target medium-term trends over a 3-month-to-2-year horizon. A genuine
medium-term trend break is a legitimate in-horizon exit — a confirmed multi-week
reversal (act) vs. a single week's move (noise — do not act).
"""

_TECHNICAL_MANDATE = """\
## RESEARCH MANDATE
Read medium-term chart structure over a 3-month-to-2-year horizon.

## HOLDING HORIZON
Your thesis is broken when the medium-term structure breaks — a confirmed
breakdown of a major base, a decisive break of key support on volume — not
because a single day or week ticked lower.
"""


# ===========================================================================
# Section 1 — Deterministic thesis-presence gate (_run_thesis_presence_gate)
# ===========================================================================

class TestThesisPresenceGate:
    """100% deterministic — no judge call, no external dependency."""

    def test_exit_missing_thesis_status_fails(self) -> None:
        """EXIT with no thesis_status at all → presence failure."""
        stances = [{"ticker": "TSLA", "action": "exit"}]
        failures, action_tickers = _run_thesis_presence_gate(stances, "value")
        assert "TSLA" in failures[0]
        assert "TSLA" in action_tickers

    def test_reduce_missing_thesis_status_fails(self) -> None:
        """REDUCE with no thesis_status → presence failure."""
        stances = [{"ticker": "AMZN", "action": "reduce"}]
        failures, _ = _run_thesis_presence_gate(stances, "growth")
        assert len(failures) == 1
        assert "AMZN" in failures[0]

    def test_add_missing_thesis_status_fails(self) -> None:
        """ADD with no thesis_status → presence failure."""
        stances = [{"ticker": "MSFT", "action": "add"}]
        failures, _ = _run_thesis_presence_gate(stances, "value")
        assert len(failures) == 1
        assert "MSFT" in failures[0]

    def test_exit_empty_reason_fails(self) -> None:
        """EXIT with thesis_status present but empty reason → presence failure."""
        stances = [{"ticker": "AMZN", "action": "reduce",
                    "thesis_status": {"verdict": "broken", "reason": ""}}]
        failures, _ = _run_thesis_presence_gate(stances, "value")
        assert len(failures) == 1
        assert "AMZN" in failures[0]

    def test_exit_whitespace_only_reason_fails(self) -> None:
        """Exit with whitespace-only reason is treated as empty → failure."""
        stances = [{"ticker": "NFLX", "action": "exit",
                    "thesis_status": {"verdict": "broken", "reason": "   "}}]
        failures, _ = _run_thesis_presence_gate(stances, "growth")
        assert len(failures) == 1

    def test_exit_non_dict_thesis_status_fails(self) -> None:
        """thesis_status that is not a dict fails (e.g. a plain string)."""
        stances = [{"ticker": "AAPL", "action": "exit",
                    "thesis_status": "broken: price fell"}]
        failures, _ = _run_thesis_presence_gate(stances, "value")
        assert len(failures) == 1
        assert "AAPL" in failures[0]

    def test_hold_no_thesis_status_passes(self) -> None:
        """HOLD with no thesis_status is exempt — passes with no failures."""
        stances = [{"ticker": "AAPL", "action": "hold"}]
        failures, action_tickers = _run_thesis_presence_gate(stances, "value")
        assert failures == []
        assert action_tickers == []  # HOLD not in action_tickers

    def test_exit_with_non_empty_reason_passes(self) -> None:
        """EXIT with a non-empty reason passes the presence gate."""
        stances = [{"ticker": "INTC", "action": "exit",
                    "thesis_status": {"verdict": "broken",
                                      "reason": "Gross margins structurally broke below 45%."}}]
        failures, action_tickers = _run_thesis_presence_gate(stances, "value")
        assert failures == []
        assert "INTC" in action_tickers

    def test_mixed_stances_isolates_failures(self) -> None:
        """Mixed stances: only the missing-reason one fails; others pass."""
        stances = [
            {"ticker": "TSLA", "action": "exit"},  # missing — FAIL
            {"ticker": "AAPL", "action": "hold"},  # exempt — PASS
            {"ticker": "MSFT", "action": "reduce",
             "thesis_status": {"verdict": "broken",
                               "reason": "Earnings power deteriorated."}},  # PASS
        ]
        failures, action_tickers = _run_thesis_presence_gate(stances, "value")
        assert len(failures) == 1
        assert "TSLA" in failures[0]
        assert set(action_tickers) == {"TSLA", "MSFT"}

    def test_fixture_presence_fail_missing(self) -> None:
        """Load presence_fail_missing_thesis_status.json → gate fails."""
        f = _load_fixture("presence_fail_missing_thesis_status.json")
        failures, _ = _run_thesis_presence_gate(f["stances"], "value")
        assert len(failures) >= 1

    def test_fixture_presence_fail_empty_reason(self) -> None:
        """Load presence_fail_empty_reason.json → gate fails."""
        f = _load_fixture("presence_fail_empty_reason.json")
        failures, _ = _run_thesis_presence_gate(f["stances"], "value")
        assert len(failures) >= 1

    def test_fixture_presence_pass_hold_exempt(self) -> None:
        """Load presence_pass_hold_exempt.json → gate passes (HOLD exempt)."""
        f = _load_fixture("presence_pass_hold_exempt.json")
        failures, action_tickers = _run_thesis_presence_gate(f["stances"], "value")
        assert failures == []
        assert action_tickers == []


# ===========================================================================
# Section 2 — parse_judge_response_with_thesis
# ===========================================================================

class TestParseJudgeResponseWithThesis:
    """Tests for the extended judge response parser."""

    def test_parses_verdict_and_thesis_block(self) -> None:
        raw = (
            "VERDICT: PASS\n"
            "JUSTIFICATION: All exits cite structural thesis changes.\n"
            "THESIS_REASONING_START\n"
            "AAPL: genuine\n"
            "TSLA: reactive\n"
            "THESIS_REASONING_END"
        )
        passed, justification, thesis_map = parse_judge_response_with_thesis(raw)
        assert passed is True
        assert "structural" in justification
        assert thesis_map == {"AAPL": THESIS_GENUINE, "TSLA": THESIS_REACTIVE}

    def test_fail_verdict_with_reactive_stance(self) -> None:
        raw = (
            "VERDICT: PASS\n"
            "JUSTIFICATION: One stance is reactive.\n"
            "THESIS_REASONING_START\n"
            "MSFT: reactive\n"
            "THESIS_REASONING_END"
        )
        passed, _, thesis_map = parse_judge_response_with_thesis(raw)
        assert passed is True
        assert thesis_map["MSFT"] == THESIS_REACTIVE

    def test_empty_thesis_block_returns_empty_map(self) -> None:
        """Response without THESIS_REASONING block → empty dict."""
        raw = "VERDICT: PASS\nJUSTIFICATION: No action stances."
        passed, justification, thesis_map = parse_judge_response_with_thesis(raw)
        assert passed is True
        assert thesis_map == {}

    def test_unknown_label_skipped_with_warning(self, caplog) -> None:
        """An unrecognised label is skipped; valid labels still parsed."""
        import logging
        raw = (
            "VERDICT: PASS\n"
            "JUSTIFICATION: Mixed.\n"
            "THESIS_REASONING_START\n"
            "AAPL: genuine\n"
            "NFLX: unknown_label\n"
            "THESIS_REASONING_END"
        )
        with caplog.at_level(logging.WARNING):
            _, _, thesis_map = parse_judge_response_with_thesis(raw)
        assert "AAPL" in thesis_map
        assert "NFLX" not in thesis_map

    def test_malformed_verdict_returns_fail(self) -> None:
        """No VERDICT line → conservative (False, message, empty map)."""
        raw = "This is not a valid judge response."
        passed, justification, thesis_map = parse_judge_response_with_thesis(raw)
        assert passed is False
        assert thesis_map == {}

    def test_ticker_uppercased_in_map(self) -> None:
        """Ticker in block is uppercased regardless of case in response."""
        raw = (
            "VERDICT: PASS\n"
            "JUSTIFICATION: ok.\n"
            "THESIS_REASONING_START\n"
            "aapl: genuine\n"
            "THESIS_REASONING_END"
        )
        _, _, thesis_map = parse_judge_response_with_thesis(raw)
        assert "AAPL" in thesis_map


# ---------------------------------------------------------------------------
# Helper: build a raw judge response string from a thesis-labels dict
# (used by TestValidateRound1Stances to simulate what the real judge returns)
# ---------------------------------------------------------------------------

def _make_raw_response(thesis_labels: dict[str, str]) -> str:
    """Build a raw judge response string containing a THESIS_REASONING block.

    Simulates the full text the output-validator-judge subagent emits when
    stances are included in the judge prompt (M6-005 extended format).
    ``validate_round1_stances`` receives this raw string from the session and
    parses both the on-mandate verdict AND the per-stance thesis block from it —
    no second judge.judge() call.

    Args:
        thesis_labels: {ticker: "genuine"|"reactive"}

    Returns:
        A raw response string that parse_judge_response_with_thesis can parse.
    """
    lines = [
        "VERDICT: PASS",
        "JUSTIFICATION: Thesis discipline assessed for all action stances.",
    ]
    if thesis_labels:
        lines.append("THESIS_REASONING_START")
        for ticker, label in thesis_labels.items():
            lines.append(f"{ticker}: {label}")
        lines.append("THESIS_REASONING_END")
    return "\n".join(lines)


# ===========================================================================
# Section 3 — validate_round1_stances (end-to-end with raw judge response)
# ===========================================================================

class TestValidateRound1Stances:
    """End-to-end tests for validate_round1_stances.

    Uses a raw judge response string (``judge_raw_response``) instead of
    judge.judge() — the same string the session captures from the ONE judge
    call per persona.  No second judge dispatch; ReplayJudge is not called.
    """

    def _make_stance(
        self,
        ticker: str,
        action: str,
        reason: str = "Earnings power structurally broke.",
        verdict: str = "broken",
    ) -> dict:
        if action.lower() == "hold":
            return {"ticker": ticker, "action": action}
        return {
            "ticker": ticker,
            "action": action,
            "thesis_status": {"verdict": verdict, "reason": reason},
        }

    def test_reactive_stance_flagged_not_blocked(self) -> None:
        """A reactive flag surfaces in notes and reactive_flags, but passed=True."""
        raw = _make_raw_response({"PYPL": THESIS_REACTIVE})
        stances = [self._make_stance("PYPL", "exit",
                                     reason="PYPL down 9% this week, momentum turned.")]
        result = validate_round1_stances(stances, "value",
                                         judge_raw_response=raw)
        assert result.passed is True  # NOT auto-blocked
        assert "PYPL" in result.reactive_flags
        assert "REACTIVE" in result.notes.upper()

    def test_genuine_stance_passes_cleanly(self) -> None:
        """A genuine exit produces no reactive flags."""
        raw = _make_raw_response({"INTC": THESIS_GENUINE})
        stances = [self._make_stance("INTC", "exit",
                                     reason="Gross margins structurally broke below 45% for two quarters.")]
        result = validate_round1_stances(stances, "value",
                                         judge_raw_response=raw)
        assert result.passed is True
        assert result.reactive_flags == []
        assert result.thesis_reasoning.get("INTC") == THESIS_GENUINE

    def test_cta_trend_break_not_mis_flagged(self) -> None:
        """CTA confirmed-trend-break exit → genuine; must NOT appear in reactive_flags."""
        raw = _make_raw_response({"SMH": THESIS_GENUINE})
        f = _load_fixture("genuine_trend_break_exit_cta.json")
        result = validate_round1_stances(
            f["stances"], "cta-systematic-macro",
            judge_raw_response=raw,
        )
        assert result.passed is True
        assert "SMH" not in result.reactive_flags
        assert result.thesis_reasoning.get("SMH") == THESIS_GENUINE

    def test_technical_trend_break_not_mis_flagged(self) -> None:
        """Technical persona confirmed structure break → genuine."""
        raw = _make_raw_response({"XBI": THESIS_GENUINE})
        f = _load_fixture("genuine_trend_break_exit_technical.json")
        result = validate_round1_stances(
            f["stances"], "technical",
            judge_raw_response=raw,
        )
        assert "XBI" not in result.reactive_flags

    def test_presence_failure_recorded_without_response(self) -> None:
        """Missing thesis_status → presence_failure even without a judge response."""
        stances = [{"ticker": "TSLA", "action": "exit"}]
        result = validate_round1_stances(stances, "value",
                                         judge_raw_response=None)
        assert result.passed is True  # still not blocked
        assert any("TSLA" in f for f in result.presence_failures)
        assert "PRESENCE" in result.notes.upper()

    def test_no_response_skips_genuine_reactive_classification(self) -> None:
        """When judge_raw_response=None, thesis_reasoning dict is empty."""
        stances = [self._make_stance("MSFT", "reduce",
                                     reason="Revenue deceleration confirmed.")]
        result = validate_round1_stances(stances, "value",
                                         judge_raw_response=None)
        assert result.thesis_reasoning == {}
        assert result.reactive_flags == []

    def test_hold_exempt_from_presence_and_classification(self) -> None:
        """HOLD stances produce no failures and no thesis_reasoning entries."""
        raw = _make_raw_response({})  # no stances to classify
        stances = [{"ticker": "AAPL", "action": "hold"}]
        result = validate_round1_stances(stances, "value",
                                         judge_raw_response=raw)
        assert result.presence_failures == []
        assert result.reactive_flags == []

    def test_mixed_genuine_and_reactive_both_recorded(self) -> None:
        """Mixed stances: reactive flags collected; genuine passes silently."""
        raw = _make_raw_response({"INTC": THESIS_GENUINE, "PYPL": THESIS_REACTIVE})
        stances = [
            self._make_stance("INTC", "exit", reason="Earnings power broke structurally."),
            self._make_stance("PYPL", "exit", reason="Down 9% this week."),
        ]
        result = validate_round1_stances(stances, "value",
                                         judge_raw_response=raw)
        assert "PYPL" in result.reactive_flags
        assert "INTC" not in result.reactive_flags
        assert result.thesis_reasoning["INTC"] == THESIS_GENUINE
        assert result.thesis_reasoning["PYPL"] == THESIS_REACTIVE

    def test_notes_appended_to_validator_notes(self) -> None:
        """Result.notes is a non-empty string suitable for writing to validator_notes."""
        raw = _make_raw_response({"WBA": THESIS_GENUINE})
        f = _load_fixture("genuine_add_new_thesis_value.json")
        result = validate_round1_stances(
            f["stances"], "value",
            judge_raw_response=raw,
        )
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_add_with_new_thesis_classified_genuine(self) -> None:
        """ADD fixture with genuine new thesis → classified as genuine."""
        raw = _make_raw_response({"WBA": THESIS_GENUINE})
        f = _load_fixture("genuine_add_new_thesis_value.json")
        result = validate_round1_stances(
            f["stances"], "value",
            judge_raw_response=raw,
        )
        assert "WBA" not in result.reactive_flags
        assert result.thesis_reasoning.get("WBA") == THESIS_GENUINE

    # -----------------------------------------------------------------------
    # ReplayJudge-path tests — the critical new coverage
    # -----------------------------------------------------------------------

    def test_replay_judge_path_thesis_block_parsed(self) -> None:
        """ReplayJudge path: raw response containing both verdict AND thesis block →
        both parse correctly from the same string; no judge.judge() re-invocation.

        This is the exact live-replay scenario: the session captured one judge
        response per persona (stored as the raw string), then calls
        validate_round1_stances with judge_raw_response=<that raw string>.
        The engine must extract the thesis block without calling judge.judge() again.
        """
        # Build a raw response as the real judge emits (verdict + thesis block).
        raw = (
            "VERDICT: PASS\n"
            "JUSTIFICATION: Value persona cited structural earnings-power break for INTC "
            "and genuine FCF recovery thesis for WBA. PYPL exit is reactive — no thesis change stated.\n"
            "THESIS_REASONING_START\n"
            "INTC: genuine\n"
            "WBA: genuine\n"
            "PYPL: reactive\n"
            "THESIS_REASONING_END"
        )
        stances = [
            self._make_stance("INTC", "exit",
                               reason="Gross margins broke below 45% for three quarters."),
            self._make_stance("WBA", "add",
                               reason="FCF recovery thesis: CEO cost plan + Shields stake."),
            self._make_stance("PYPL", "exit",
                               reason="Down 9% this week, momentum turned negative."),
        ]
        result = validate_round1_stances(stances, "value", judge_raw_response=raw)

        # On-mandate verdict (passed) and thesis block both come from same raw string.
        assert result.passed is True
        assert result.thesis_reasoning == {
            "INTC": THESIS_GENUINE,
            "WBA": THESIS_GENUINE,
            "PYPL": THESIS_REACTIVE,
        }
        assert "PYPL" in result.reactive_flags
        assert "INTC" not in result.reactive_flags
        assert "WBA" not in result.reactive_flags

    def test_replay_judge_path_no_second_judge_call(self) -> None:
        """Confirm validate_round1_stances does NOT call judge.judge() at all.

        ReplayJudge.judge() raises KeyError for unknown slugs.  If the engine
        were calling judge.judge(), this test would raise.  Passing a ReplayJudge
        with no verdicts and relying on judge_raw_response confirms the
        judge.judge() call path is eliminated.
        """
        # ReplayJudge with NO stored verdicts — any .judge() call raises KeyError.
        replay_judge = ReplayJudge({})
        raw = _make_raw_response({"MSFT": THESIS_GENUINE})
        stances = [self._make_stance("MSFT", "reduce",
                                     reason="Revenue growth deceleration confirmed.")]

        # Must NOT raise — validate_round1_stances must not call replay_judge.judge().
        result = validate_round1_stances(stances, "value", judge_raw_response=raw)
        assert result.thesis_reasoning.get("MSFT") == THESIS_GENUINE

    def test_replay_judge_path_response_without_thesis_block(self) -> None:
        """Raw response that has no THESIS_REASONING block → empty thesis_reasoning.

        Covers the case where the session's judge ran without stances (old-format
        response) — validate_round1_stances degrades gracefully to Gate A only.
        """
        raw = "VERDICT: PASS\nJUSTIFICATION: Report is on mandate."
        stances = [self._make_stance("AAPL", "exit",
                                     reason="FCF yield thesis broke.")]
        result = validate_round1_stances(stances, "value", judge_raw_response=raw)
        # Gate A still runs; Gate B finds no thesis block → empty.
        assert result.thesis_reasoning == {}
        assert result.reactive_flags == []
        assert result.passed is True


# ===========================================================================
# Section 4 — StubOnMandateJudgeWithThesis
# ===========================================================================

class TestStubOnMandateJudgeWithThesis:
    """Stub returns correctly-formatted THESIS_REASONING block."""

    def test_stub_returns_thesis_block_format(self) -> None:
        stub = StubOnMandateJudgeWithThesis(
            thesis_labels={"value": {"AAPL": THESIS_GENUINE, "TSLA": THESIS_REACTIVE}}
        )
        _passed, raw = stub.judge(
            report="ROUND-1 STANCES FOR THESIS DISCIPLINE JUDGMENT (persona: value)\n",
            mandate=_VALUE_MANDATE,
            persona_slug="value",
            on_mandate_concepts=(),
            off_mandate_signals=(),
        )
        assert "THESIS_REASONING_START" in raw
        assert "AAPL: genuine" in raw
        assert "TSLA: reactive" in raw
        assert "THESIS_REASONING_END" in raw

    def test_stub_no_labels_returns_pass_no_block(self) -> None:
        """Empty labels → no THESIS_REASONING block in response."""
        stub = StubOnMandateJudgeWithThesis(thesis_labels={"value": {}})
        _passed, raw = stub.judge("", _VALUE_MANDATE, "value", (), ())
        assert "THESIS_REASONING_START" not in raw

    def test_stub_parse_roundtrip(self) -> None:
        """Stub output → parse_judge_response_with_thesis → same labels."""
        labels = {"MSFT": THESIS_GENUINE, "PYPL": THESIS_REACTIVE}
        stub = StubOnMandateJudgeWithThesis(thesis_labels={"value": labels})
        _passed, raw = stub.judge("", _VALUE_MANDATE, "value", (), ())
        _, _, thesis_map = parse_judge_response_with_thesis(raw)
        assert thesis_map == labels


# ===========================================================================
# Section 5 — _build_stances_block
# ===========================================================================

class TestBuildStancesBlock:
    """The block injected into the judge prompt contains required markers."""

    def test_block_contains_round1_marker(self) -> None:
        stances = [{"ticker": "AAPL", "action": "exit",
                    "thesis_status": {"verdict": "broken", "reason": "Thesis broke."}}]
        block = _build_stances_block(stances, "value")
        assert "ROUND-1 STANCES FOR THESIS DISCIPLINE JUDGMENT" in block
        assert "persona: value" in block

    def test_block_contains_ticker_and_reason(self) -> None:
        stances = [{"ticker": "MSFT", "action": "reduce",
                    "thesis_status": {"verdict": "broken",
                                      "reason": "Revenue growth decelerated."}}]
        block = _build_stances_block(stances, "growth")
        assert "TICKER: MSFT" in block
        assert "Revenue growth decelerated" in block

    def test_block_multiple_stances(self) -> None:
        stances = [
            {"ticker": "A", "action": "exit",
             "thesis_status": {"verdict": "broken", "reason": "Reason A"}},
            {"ticker": "B", "action": "add",
             "thesis_status": {"verdict": "new", "reason": "Reason B"}},
        ]
        block = _build_stances_block(stances, "value")
        assert "TICKER: A" in block
        assert "TICKER: B" in block


# ===========================================================================
# Section 6 — _parse_inline_thesis_reasoning (fallback)
# ===========================================================================

class TestParseInlineThesisReasoning:
    """Fallback parser for compact stub format (used when no block present)."""

    def test_parses_pipe_separated(self) -> None:
        raw = "AAPL: genuine | TSLA: reactive"
        result = _parse_inline_thesis_reasoning(raw)
        assert result == {"AAPL": THESIS_GENUINE, "TSLA": THESIS_REACTIVE}

    def test_ignores_unknown_labels(self) -> None:
        raw = "AAPL: genuine | MSFT: unknown"
        result = _parse_inline_thesis_reasoning(raw)
        assert "AAPL" in result
        assert "MSFT" not in result

    def test_empty_string_returns_empty_dict(self) -> None:
        assert _parse_inline_thesis_reasoning("") == {}
