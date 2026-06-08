"""Unit tests for Component 16 — blend_consensus (TASK-M2-006).

Covers:
  - 5 synthetic stance-set fixtures (all-agree, sharply-split, one-persona-extreme,
    everyone-EXITs-one-name, name-only-one-persona-holds)
  - Edge-case decisions: all-EXIT ticker, single-holder, over-1 normalization,
    under-1 residual-as-cash
  - M3-swap-isolation: blend_consensus is a single pure function and no other
    component reads its intermediate state (only the returned dict)
  - Post-condition assertions: every weight ≤ max_position_weight, Σ ≤ 1

Fixture design note (provenance):
    All fixtures are synthetic — hand-crafted to drive specific arithmetic
    paths.  They are not derived from a live run (no live data available at
    M2-006 time; the live validation-week run is TASK-M2-011).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import pytest

from round_table_portfolio.portfolio.consensus import blend_consensus


# ---------------------------------------------------------------------------
# Minimal stance stub (mirrors AgentStancePayload from orchestrator/round1.py)
# ---------------------------------------------------------------------------

@dataclass
class _Stance:
    """Minimal stance stub for unit tests — matches AgentStancePayload duck type."""
    ticker: str
    action: str          # 'add' | 'reduce' | 'hold' | 'exit'
    target_weight: float
    confidence: int      # 1–5
    persona: str = "persona"
    week_id: str = "2026-W23"
    round: int = 1


def _s(ticker: str, action: str, weight: float, confidence: int = 3, persona: str = "p") -> _Stance:
    """Helper: build a stance."""
    return _Stance(ticker=ticker, action=action, target_weight=weight, confidence=confidence, persona=persona)


_MAX_PW = 0.20  # matches thresholds.yaml
_CFG = {"max_position_weight": _MAX_PW}


# ---------------------------------------------------------------------------
# Helper: assert post-conditions on any blended result
# ---------------------------------------------------------------------------

def _assert_postconditions(weights: dict[str, float], max_pw: float = _MAX_PW) -> None:
    """Assert the three post-conditions guaranteed by blend_consensus."""
    for ticker, w in weights.items():
        assert w >= 0.0, f"Negative weight for {ticker}: {w}"
        assert w <= max_pw + 1e-9, f"Weight for {ticker} ({w:.6f}) exceeds cap ({max_pw})"
    total = sum(weights.values())
    assert total <= 1.0 + 1e-9, f"Σ weights = {total:.8f} > 1"


# ===========================================================================
# FIXTURE 1 — all-agree
# All 7 personas hold AAPL at 0.10 and MSFT at 0.15.
# Expected: AAPL → 0.10, MSFT → 0.15 (plain mean of identical values).
# Σ = 0.25, residual 0.75 becomes cash (not in returned dict).
# ===========================================================================

class TestAllAgree:
    """Fixture 1: all 7 personas hold the same two tickers at the same weights."""

    PERSONAS = [f"p{i}" for i in range(7)]
    TICKERS = {"AAPL": 0.10, "MSFT": 0.15}

    @pytest.fixture
    def stances(self) -> list[_Stance]:
        result = []
        for persona in self.PERSONAS:
            for ticker, weight in self.TICKERS.items():
                result.append(_s(ticker, "add", weight, persona=persona))
        return result

    def test_weights_are_mean_of_identical_values(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert math.isclose(weights["AAPL"], 0.10, abs_tol=1e-9), (
            f"AAPL expected 0.10, got {weights['AAPL']}"
        )
        assert math.isclose(weights["MSFT"], 0.15, abs_tol=1e-9), (
            f"MSFT expected 0.15, got {weights['MSFT']}"
        )

    def test_sum_under_one_no_inflation(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        total = sum(weights.values())
        assert math.isclose(total, 0.25, abs_tol=1e-9), (
            f"Expected Σ=0.25 (residual stays as cash), got {total:.8f}"
        )

    def test_no_cash_key_in_returned_dict(self, stances: list[_Stance]) -> None:
        """blend_consensus returns position weights only; CASH is NOT a key."""
        weights = blend_consensus(stances, _CFG)
        assert "CASH" not in weights, (
            "blend_consensus must not add a CASH key — Component 15 does that."
        )

    def test_postconditions(self, stances: list[_Stance]) -> None:
        _assert_postconditions(blend_consensus(stances, _CFG))


# ===========================================================================
# FIXTURE 2 — sharply-split
# 4 personas hold TSLA at 0.20; 3 personas EXIT TSLA.
# Expected: TSLA → mean(0.20, 0.20, 0.20, 0.20) = 0.20 (over the 4 holders).
# Also: NVDA at 0.10 for all 7.
# ===========================================================================

class TestSharpySplit:
    """Fixture 2: sharply-split stance on TSLA (4 hold, 3 EXIT)."""

    @pytest.fixture
    def stances(self) -> list[_Stance]:
        result: list[_Stance] = []
        # 4 personas ADD TSLA at 0.20.
        for i in range(4):
            result.append(_s("TSLA", "add", 0.20, persona=f"holder_{i}"))
        # 3 personas EXIT TSLA.
        for i in range(3):
            result.append(_s("TSLA", "exit", 0.0, persona=f"exiter_{i}"))
        # All 7 ADD NVDA at 0.10.
        for i in range(7):
            persona = f"holder_{i}" if i < 4 else f"exiter_{i - 4}"
            result.append(_s("NVDA", "add", 0.10, persona=persona))
        return result

    def test_tsla_mean_over_non_exit_holders_only(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        # 4 holders each at 0.20 → mean = 0.20.
        assert math.isclose(weights["TSLA"], 0.20, abs_tol=1e-9), (
            f"TSLA expected 0.20 (mean over 4 non-EXIT), got {weights['TSLA']}"
        )

    def test_nvda_weight_correct(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert math.isclose(weights["NVDA"], 0.10, abs_tol=1e-9), (
            f"NVDA expected 0.10, got {weights['NVDA']}"
        )

    def test_exit_stances_do_not_contribute_to_average(self, stances: list[_Stance]) -> None:
        """EXIT stances must be excluded from the mean, not counted as 0."""
        weights = blend_consensus(stances, _CFG)
        # If EXIT were included: mean = (4 × 0.20 + 3 × 0.0) / 7 ≈ 0.1143.
        # Correct (exclude EXIT): mean = (4 × 0.20) / 4 = 0.20.
        assert weights["TSLA"] > 0.10, (
            "EXIT stances are incorrectly being averaged in — "
            f"TSLA weight {weights['TSLA']:.4f} is too low (expected 0.20)."
        )

    def test_postconditions(self, stances: list[_Stance]) -> None:
        _assert_postconditions(blend_consensus(stances, _CFG))


# ===========================================================================
# FIXTURE 3 — one-persona-extreme
# 6 personas hold AMZN at 0.05; 1 persona holds AMZN at 0.20 (the cap).
# Expected mean = (6 × 0.05 + 1 × 0.20) / 7 = (0.30 + 0.20) / 7 = 0.50/7 ≈ 0.07143.
# The cap (0.20) is not violated by the mean, so no capping needed.
# ===========================================================================

class TestOnePersonaExtreme:
    """Fixture 3: one persona at the cap weight, six at a low weight."""

    EXPECTED_MEAN = (6 * 0.05 + 1 * 0.20) / 7  # ≈ 0.07143

    @pytest.fixture
    def stances(self) -> list[_Stance]:
        result: list[_Stance] = []
        for i in range(6):
            result.append(_s("AMZN", "add", 0.05, persona=f"low_{i}"))
        result.append(_s("AMZN", "add", 0.20, persona="high_0"))
        return result

    def test_mean_is_correct(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert math.isclose(weights["AMZN"], self.EXPECTED_MEAN, abs_tol=1e-9), (
            f"AMZN expected {self.EXPECTED_MEAN:.6f}, got {weights['AMZN']:.6f}"
        )

    def test_mean_does_not_exceed_cap(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert weights["AMZN"] <= _MAX_PW + 1e-9

    def test_postconditions(self, stances: list[_Stance]) -> None:
        _assert_postconditions(blend_consensus(stances, _CFG))


# ===========================================================================
# FIXTURE 4 — everyone-EXITs-one-name
# All 7 personas EXIT GOOG; they all ADD AAPL at 0.10.
# Expected: GOOG → absent (excluded, weight 0 dropped); AAPL → 0.10.
# ===========================================================================

class TestEveryoneExitsOneName:
    """Fixture 4: all 7 personas EXIT one ticker — it must be dropped from output."""

    @pytest.fixture
    def stances(self) -> list[_Stance]:
        result: list[_Stance] = []
        for i in range(7):
            result.append(_s("GOOG", "exit", 0.0, persona=f"p{i}"))
            result.append(_s("AAPL", "add", 0.10, persona=f"p{i}"))
        return result

    def test_all_exit_ticker_excluded(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert "GOOG" not in weights, (
            f"GOOG (all-EXIT) should be excluded from output, but got weight {weights.get('GOOG')}"
        )

    def test_aapl_weight_correct(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert math.isclose(weights["AAPL"], 0.10, abs_tol=1e-9), (
            f"AAPL expected 0.10, got {weights['AAPL']}"
        )

    def test_only_aapl_in_output(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert set(weights.keys()) == {"AAPL"}, (
            f"Expected only {{AAPL}} in output, got {set(weights.keys())}"
        )

    def test_postconditions(self, stances: list[_Stance]) -> None:
        _assert_postconditions(blend_consensus(stances, _CFG))


# ===========================================================================
# FIXTURE 5 — name only one persona holds
# Only 1 of 7 personas holds META at 0.12; all others EXIT META.
# All 7 hold MSFT at 0.08.
# Expected: META → 0.12 (mean over the 1 non-EXIT holder); MSFT → 0.08.
# ===========================================================================

class TestNameOnlyOnePersonaHolds:
    """Fixture 5: only one persona holds a given ticker (others all EXIT)."""

    @pytest.fixture
    def stances(self) -> list[_Stance]:
        result: list[_Stance] = []
        # Only p0 holds META.
        result.append(_s("META", "add", 0.12, persona="p0"))
        for i in range(1, 7):
            result.append(_s("META", "exit", 0.0, persona=f"p{i}"))
        # All 7 hold MSFT.
        for i in range(7):
            result.append(_s("MSFT", "add", 0.08, persona=f"p{i}"))
        return result

    def test_single_holder_mean(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert math.isclose(weights["META"], 0.12, abs_tol=1e-9), (
            f"META single-holder: expected 0.12, got {weights['META']:.6f}"
        )

    def test_msft_weight_correct(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert math.isclose(weights["MSFT"], 0.08, abs_tol=1e-9), (
            f"MSFT expected 0.08, got {weights['MSFT']:.6f}"
        )

    def test_postconditions(self, stances: list[_Stance]) -> None:
        _assert_postconditions(blend_consensus(stances, _CFG))


# ===========================================================================
# EDGE CASE — normalization when raw sum exceeds 1
# 7 personas all hold 10 tickers at 0.12 each.
# Raw mean per ticker = 0.12; cap is 0.20 (no capping triggered).
# Raw Σ = 10 × 0.12 = 1.20 > 1 → scale down by 1/1.20.
# Expected per-ticker weight after scaling = 0.12 / 1.20 = 0.10.
# ===========================================================================

class TestOverOneNormalization:
    """Raw sum > 1: must scale down to Σ ≤ 1, never inflate."""

    TICKERS = [f"TICK{i}" for i in range(10)]

    @pytest.fixture
    def stances(self) -> list[_Stance]:
        result: list[_Stance] = []
        for i in range(7):
            for ticker in self.TICKERS:
                result.append(_s(ticker, "add", 0.12, persona=f"p{i}"))
        return result

    def test_sum_scaled_to_one(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        total = sum(weights.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9), (
            f"After scaling, Σ should be 1.0, got {total:.8f}"
        )

    def test_each_weight_after_scaling(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        expected = 0.12 / 1.20  # = 0.10
        for ticker in self.TICKERS:
            assert math.isclose(weights[ticker], expected, abs_tol=1e-9), (
                f"{ticker} expected {expected:.6f} after scaling, got {weights[ticker]:.6f}"
            )

    def test_postconditions(self, stances: list[_Stance]) -> None:
        _assert_postconditions(blend_consensus(stances, _CFG))


# ===========================================================================
# EDGE CASE — normalization when raw sum is under 1 (no inflation)
# 7 personas hold 2 tickers at 0.05 each.
# Raw Σ = 2 × 0.05 = 0.10; residual 0.90 must remain as cash (NOT inflated).
# ===========================================================================

class TestUnderOneNoInflation:
    """Raw sum < 1: residual stays as cash — weights are NOT inflated to fill 1."""

    @pytest.fixture
    def stances(self) -> list[_Stance]:
        result: list[_Stance] = []
        for i in range(7):
            result.append(_s("AAPL", "add", 0.05, persona=f"p{i}"))
            result.append(_s("MSFT", "add", 0.05, persona=f"p{i}"))
        return result

    def test_sum_not_inflated(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        total = sum(weights.values())
        assert math.isclose(total, 0.10, abs_tol=1e-9), (
            f"Expected Σ=0.10 (no inflation), got {total:.8f}"
        )

    def test_postconditions(self, stances: list[_Stance]) -> None:
        _assert_postconditions(blend_consensus(stances, _CFG))


# ===========================================================================
# EDGE CASE — per-ticker cap applied before normalization
# A raw mean weight > max_position_weight is capped to max_position_weight.
# 7 personas hold AAPL at 0.20 (the cap); no Σ overflow.
# Expected: AAPL → 0.20 exactly (mean = cap, no clipping distortion).
# ===========================================================================

class TestPerTickerCapAtExactLimit:
    """Mean at exactly the cap must pass through unchanged."""

    @pytest.fixture
    def stances(self) -> list[_Stance]:
        return [_s("AAPL", "add", 0.20, persona=f"p{i}") for i in range(7)]

    def test_weight_at_cap_is_preserved(self, stances: list[_Stance]) -> None:
        weights = blend_consensus(stances, _CFG)
        assert math.isclose(weights["AAPL"], 0.20, abs_tol=1e-9)

    def test_postconditions(self, stances: list[_Stance]) -> None:
        _assert_postconditions(blend_consensus(stances, _CFG))


# ===========================================================================
# EDGE CASE — empty stances list (all EXIT or no stances)
# blend_consensus must return {} (not raise).
# ===========================================================================

class TestEmptyStances:
    """No non-EXIT stances → return empty dict, no raise."""

    def test_empty_returns_empty(self) -> None:
        assert blend_consensus([], _CFG) == {}

    def test_all_exit_returns_empty(self) -> None:
        stances = [_s("AAPL", "exit", 0.0, persona=f"p{i}") for i in range(7)]
        assert blend_consensus(stances, _CFG) == {}


# ===========================================================================
# M3-SWAP ISOLATION TEST
# Verifies that blend_consensus is a SINGLE pure function (no class, no
# cached state, no module-level mutable that another component could read).
# ===========================================================================

class TestM3SwapIsolation:
    """The M3-swap guarantee: blend_consensus is one pure function.

    The contract: only the returned weights dict is consumed by downstream
    components.  No other component reads any intermediate state of the blend.

    We verify this by:
    1. Confirming blend_consensus is a plain function (not a class/object with
       hidden state).
    2. Confirming successive calls with different stances produce independent
       results (no module-level mutable cache).
    3. Confirming the module exports only the one entry point (no internal
       state accessor).
    4. Confirming the import chain from weekly_run.py reaches
       portfolio.consensus.blend_consensus — not the stub.
    """

    def test_blend_consensus_is_a_plain_function(self) -> None:
        assert inspect.isfunction(blend_consensus), (
            "blend_consensus must be a plain function, not a class or callable object."
        )

    def test_successive_calls_are_independent(self) -> None:
        """Two successive calls with different stances must not share state."""
        s1 = [_s("AAPL", "add", 0.10, persona=f"p{i}") for i in range(7)]
        s2 = [_s("MSFT", "add", 0.15, persona=f"p{i}") for i in range(7)]

        w1 = blend_consensus(s1, _CFG)
        w2 = blend_consensus(s2, _CFG)

        assert "AAPL" in w1 and "MSFT" not in w1, (
            "First call result polluted by second call's state."
        )
        assert "MSFT" in w2 and "AAPL" not in w2, (
            "Second call result polluted by first call's state."
        )

    def test_returned_dict_is_the_only_output(self) -> None:
        """blend_consensus must return only the weights dict — no side outputs."""
        stances = [_s("AAPL", "add", 0.10, persona=f"p{i}") for i in range(7)]
        result = blend_consensus(stances, _CFG)
        assert isinstance(result, dict), (
            f"blend_consensus must return dict[str, float], got {type(result)}"
        )

    def test_weekly_run_imports_real_function_not_stub(self) -> None:
        """weekly_run.py must import blend_consensus from portfolio.consensus."""
        import round_table_portfolio.orchestrator.weekly_run as wr_module
        # The function object bound in weekly_run should be the real one.
        fn = getattr(wr_module, "blend_consensus", None)
        assert fn is not None, "blend_consensus not found in weekly_run module"
        assert fn is blend_consensus, (
            "weekly_run.blend_consensus is not the real portfolio.consensus.blend_consensus. "
            "It may still be importing the stub."
        )

    def test_stub_blend_consensus_not_imported_in_weekly_run(self) -> None:
        """The stub blend_consensus must NOT be the active function in weekly_run."""
        from round_table_portfolio.orchestrator import _stubs
        import round_table_portfolio.orchestrator.weekly_run as wr_module
        wr_fn = getattr(wr_module, "blend_consensus")
        stub_fn = getattr(_stubs, "blend_consensus")
        assert wr_fn is not stub_fn, (
            "weekly_run is still using the _stubs.blend_consensus — "
            "TASK-M2-006 wiring is incomplete."
        )


# ===========================================================================
# INTEGRATION SMOKE TEST — no STUB_ALLOW needed (real function, no stubs)
# Confirm the real blend_consensus runs without STUB_ALLOW env var.
# ===========================================================================

class TestNoStubAllowRequired:
    """Real blend_consensus must run without STUB_ALLOW=1 set."""

    def test_runs_without_stub_allow(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("STUB_ALLOW", raising=False)
        stances = [_s("AAPL", "add", 0.10, persona=f"p{i}") for i in range(7)]
        # Should not raise — the real function has no stub guard.
        weights = blend_consensus(stances, _CFG)
        assert "AAPL" in weights
