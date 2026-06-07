"""Unit tests for portfolio/invariants.py — check_fully_invested.

Every test here is deterministic — the helper performs exact arithmetic with
a float tolerance and has no external dependencies.  100% pass rate required.

Violation types covered:
- Well-formed pass case (positions + CASH = 1.0, all weights in-range)
- cash < 0 (over-invested, negative residual)
- Σ(positions) + cash ≠ 1.0 — sums to 0.95 (under-invested without CASH)
- Σ(positions) + cash ≠ 1.0 — sums to 1.05 (over-invested)
- Single position weight > max_position_weight
- Negative position weight
- All-cash portfolio (positions empty, cash = 1.0) — edge case, must pass
- Exactly-at-cap is allowed; just-over-cap is flagged
- Multiple violations produce multiple reason strings
"""

from __future__ import annotations

import pytest

from round_table_portfolio.portfolio.invariants import check_fully_invested


# ---------------------------------------------------------------------------
# Well-formed pass cases
# ---------------------------------------------------------------------------

class TestWellFormed:
    def test_simple_pass(self) -> None:
        """Three positions + CASH summing to exactly 1.0."""
        positions = {"AAPL": 0.20, "MSFT": 0.15, "NVDA": 0.10}
        cash = 0.55
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.20)
        assert passed, f"Expected pass but got reasons: {reasons}"
        assert reasons == []

    def test_all_cash(self) -> None:
        """Empty positions, cash = 1.0 — a fully-cash portfolio is valid."""
        passed, reasons = check_fully_invested({}, cash=1.0, max_position_weight=0.20)
        assert passed, f"All-cash portfolio should pass: {reasons}"
        assert reasons == []

    def test_exactly_at_cap(self) -> None:
        """A position exactly at max_position_weight is allowed (not > cap)."""
        positions = {"AAPL": 0.20, "MSFT": 0.20}
        cash = 0.60
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.20)
        assert passed, f"At-cap position should pass: {reasons}"
        assert reasons == []

    def test_zero_cash_fully_invested(self) -> None:
        """Positions sum to exactly 1.0, cash = 0.0 — valid (fully deployed)."""
        positions = {"AAPL": 0.20, "MSFT": 0.15, "NVDA": 0.10, "AMZN": 0.10, "GOOGL": 0.45}
        cash = 0.0
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.45)
        assert passed, f"Zero-cash fully-invested portfolio should pass: {reasons}"
        assert reasons == []

    def test_within_float_tolerance(self) -> None:
        """Tiny float rounding that lands inside 1e-6 tolerance should still pass."""
        positions = {"AAPL": 1 / 3, "MSFT": 1 / 3}
        cash = 1 - (2 / 3)  # floating-point residual, very close to 1/3
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.40)
        assert passed, f"Within-tolerance rounding should pass: {reasons}"


# ---------------------------------------------------------------------------
# Violation: cash < 0
# ---------------------------------------------------------------------------

class TestNegativeCash:
    def test_negative_cash_flagged(self) -> None:
        """Positions sum > 1.0 forces cash < 0 — must flag."""
        positions = {"AAPL": 0.60, "MSFT": 0.50}
        cash = -0.10  # over-invested
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.60)
        assert not passed
        assert any("negative" in r.lower() or "over-invested" in r.lower() for r in reasons)

    def test_negative_cash_reason_mentions_cash(self) -> None:
        positions = {"AAPL": 0.55, "MSFT": 0.55}
        cash = -0.10
        _, reasons = check_fully_invested(positions, cash, max_position_weight=0.60)
        assert any("cash" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Violation: Σ ≠ 1.0
# ---------------------------------------------------------------------------

class TestSumNotOne:
    def test_sums_to_095_no_cash(self) -> None:
        """Positions sum to 0.85, cash = 0.10 → total 0.95 — must FLAG.

        Corresponds to the 'sums-to-0.95-no-CASH' malformed fixture in the
        output-validator test suite.
        """
        positions = {"AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.25}
        cash = 0.10  # total = 0.95
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.30)
        assert not passed
        assert any("1.0" in r or "sum" in r.lower() or "deviation" in r.lower() for r in reasons)

    def test_sums_to_105_over_invested(self) -> None:
        """Positions sum to 1.05, cash = 0.0 — over-invested, must FLAG.

        Corresponds to the 'sums-to-1.05-over-invested' malformed fixture.
        """
        positions = {"AAPL": 0.40, "MSFT": 0.40, "NVDA": 0.25}
        cash = 0.0  # total = 1.05
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.40)
        assert not passed
        assert any("1.0" in r or "deviation" in r.lower() for r in reasons)

    def test_sums_to_110_flagged(self) -> None:
        """Extreme over-investment is still a single sum-check violation."""
        positions = {"AAPL": 0.60, "MSFT": 0.50}
        cash = 0.0  # total = 1.10
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.60)
        assert not passed


# ---------------------------------------------------------------------------
# Violation: single position > max_position_weight
# ---------------------------------------------------------------------------

class TestConcentrationCap:
    def test_single_weight_over_cap_flagged(self) -> None:
        """One position at 0.25 with cap 0.20 — must FLAG.

        Corresponds to the 'single-weight-over-cap' malformed fixture.
        """
        positions = {"AAPL": 0.25, "MSFT": 0.15}
        cash = 0.60
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.20)
        assert not passed
        assert any("AAPL" in r for r in reasons)
        assert any("max_position_weight" in r or "0.2000" in r for r in reasons)

    def test_just_over_cap_flagged(self) -> None:
        """0.20 + 2e-6 is clearly beyond the 1e-6 tolerance band and must be flagged."""
        over = 0.20 + 2e-6
        positions = {"AAPL": over, "MSFT": 0.10}
        cash = round(1.0 - over - 0.10, 10)
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.20)
        assert not passed

    def test_multiple_positions_over_cap(self) -> None:
        """Two positions over cap produce two separate reason strings."""
        positions = {"AAPL": 0.30, "MSFT": 0.25}
        cash = 0.45
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.20)
        assert not passed
        over_cap_reasons = [r for r in reasons if "max_position_weight" in r or "exceeds" in r.lower()]
        assert len(over_cap_reasons) == 2


# ---------------------------------------------------------------------------
# Violation: negative position weight
# ---------------------------------------------------------------------------

class TestNegativePositionWeight:
    def test_negative_position_flagged(self) -> None:
        """A short position (negative weight) is not allowed."""
        positions = {"AAPL": -0.05, "MSFT": 0.30}
        cash = 0.75
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.30)
        assert not passed
        assert any("AAPL" in r for r in reasons)
        assert any("negative" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Multiple violations at once
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    def test_over_invested_and_over_cap_both_reported(self) -> None:
        """A portfolio that is both over-invested AND has an over-cap position
        should produce at least two reason strings."""
        positions = {"AAPL": 0.70, "MSFT": 0.50}
        cash = -0.20  # forced negative by over-investment
        passed, reasons = check_fully_invested(positions, cash, max_position_weight=0.60)
        assert not passed
        assert len(reasons) >= 2


# ---------------------------------------------------------------------------
# Single-source guarantee (AC3)
# ---------------------------------------------------------------------------

class TestSingleSource:
    def test_check_fully_invested_importable_from_invariants(self) -> None:
        """check_fully_invested must be importable from portfolio.invariants.

        This test is the AC3 single-source assertion: both the validator
        (Component 11, Layer 2) and the materializer (Component 15, Layer 3)
        import from this exact module path.  If the symbol is moved or renamed,
        this test breaks and forces an explicit reconciliation.
        """
        from round_table_portfolio.portfolio.invariants import check_fully_invested as fn
        assert callable(fn)
        # Confirm it's the same object imported at module load.
        assert fn is check_fully_invested
