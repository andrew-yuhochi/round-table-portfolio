"""Unit tests for Component 25 — re-synthesis (resynthesis.py).

Test plan (≥7 deterministic cells + 1 real-W24 anchored case):

Synthetic fixtures
  F-A  Both outliers DEFEND → final_weights == provisional_weights (no-op merge)
  F-B  One outlier REVISES one ticker → consensus moves on exactly that ticker
  F-C  Both outliers REVISE toward each other (divergent positions converge)
  F-D  Both outliers REVISE apart (aligned positions diverge)

Proof cells
  P-1  AC1 blend-reuse proof: reconstruct merged set externally, call
       blend_consensus directly, assert identical final_weights (no parallel math)
  P-2  AC2 invariant: final_weights passes check_fully_invested (Σ + CASH = 1,
       every weight ≤ max_position_weight, cash ≥ 0)

AC-3 cells (defend → no-op; revise → moves exactly those names)
  A3-a Defend path: both outliers defend → delta is empty (final == provisional)
  A3-b Revise path: outlier revises T1 only → delta carries T1 only; all other
       tickers unchanged

Real-W24 anchored case
  W24  Uses stances_2026_w24_round1.json (280 rows, 7×40 tickers).
       Outliers: growth + cta-systematic-macro (algorithmic result from M3-001).
       Stubbed round=2 stances: growth revises DELL (0.10→0.05) + NVDA (hold→add 0.08);
       cta-systematic-macro revises CHTR (exit→hold 0.04) + DELL (0.10→0.15).
       Asserts: provisional ≠ final; delta carries exactly the revised tickers;
       final passes invariant; final == blend_consensus(merged, config) (reuse proof).

Edge / error cases
  E-1  Empty outlier_personas raises ValueError
  E-2  All 7 personas exit every ticker → empty final_weights (100% cash)
  E-3  Round=2 stances for a ticker NOT in round=1 → appended + warning (no crash)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest

from round_table_portfolio.orchestrator.resynthesis import (
    ResynthesisResult,
    resynthesize_consensus,
    verify_blend_reuse,
)
from round_table_portfolio.orchestrator.round1 import AgentStancePayload
from round_table_portfolio.portfolio.consensus import blend_consensus
from round_table_portfolio.portfolio.invariants import check_fully_invested

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEEK = "2026-W99"
_MAX_W = 0.20
_CFG = {"max_position_weight": _MAX_W}

_ALL_PERSONAS = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]


def _stance(
    persona: str,
    ticker: str,
    action: str,
    target_weight: float,
    confidence: int = 3,
    round_: int = 1,
) -> AgentStancePayload:
    return AgentStancePayload(
        week_id=_WEEK,
        persona=persona,
        ticker=ticker,
        round=round_,
        action=action,
        target_weight=target_weight,
        confidence=confidence,
        rationale="stub",
        user_id="andrew",
        roster_version=1,
        enhancement_version=1,
    )


def _make_round1_stances(
    tickers: list[str],
    weights_by_persona: dict[str, dict[str, tuple[str, float]]],
) -> list[AgentStancePayload]:
    """Build round=1 stance list from a {persona: {ticker: (action, weight)}} dict."""
    stances: list[AgentStancePayload] = []
    for persona in _ALL_PERSONAS:
        pw = weights_by_persona.get(persona, {})
        for ticker in tickers:
            action, tw = pw.get(ticker, ("hold", 0.0))
            stances.append(_stance(persona, ticker, action, tw))
    return stances


def _invariant_passes(weights: dict[str, float]) -> tuple[bool, list[str]]:
    """Check that a weights dict (no CASH key) satisfies the fully-invested invariant."""
    cash = 1.0 - sum(weights.values())
    return check_fully_invested(dict(weights), cash, _MAX_W)


# ---------------------------------------------------------------------------
# Fixture F-A: both outliers defend → final == provisional
# ---------------------------------------------------------------------------

class TestBothDefend:
    """F-A + A3-a: both outliers defend — merge is a no-op."""

    TICKERS = ["AAPL", "MSFT", "GOOG"]
    OUTLIERS = {"growth", "cta-systematic-macro"}

    def _build(self):
        # Simple setup: each persona adds one ticker at 0.10 or holds.
        w: dict[str, dict[str, tuple[str, float]]] = {
            "value":                  {"AAPL": ("add", 0.10), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0)},
            "growth":                 {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.12), "GOOG": ("hold", 0.0)},
            "discretionary-macro":    {"AAPL": ("add", 0.08), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0)},
            "cta-systematic-macro":   {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0), "GOOG": ("add", 0.15)},
            "technical":              {"AAPL": ("add", 0.10), "MSFT": ("add", 0.10), "GOOG": ("hold", 0.0)},
            "quant-systematic":       {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.11), "GOOG": ("hold", 0.0)},
            "risk-officer":           {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0)},
        }
        round1 = _make_round1_stances(self.TICKERS, w)
        provisional = blend_consensus(round1, _CFG)

        # Defend = round=2 stances are IDENTICAL to round=1 (same weight/action).
        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona in self.OUTLIERS:
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        return round1, round2, provisional

    def test_final_equals_provisional(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert result.final_weights == pytest.approx(provisional, abs=1e-9), (
            "Both outliers defended — final consensus must equal provisional."
        )

    def test_delta_is_empty(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        # delta omits zero-change tickers — so it must be empty when final == provisional.
        assert result.delta == {}, (
            "Both outliers defended — delta must be empty (no weight moved)."
        )

    def test_merged_stances_count(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        # Merged set must have the same count as round=1 (7 × 3 tickers = 21).
        assert len(result.merged_stances) == len(round1)

    def test_invariant_passes(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        passed, reasons = _invariant_passes(result.final_weights)
        assert passed, f"Invariant failed: {reasons}"

    def test_blend_reuse_proof(self):
        """AC1 proof: same function, no parallel math."""
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert verify_blend_reuse(result.merged_stances, result.final_weights, _CFG), (
            "blend_consensus(merged_stances) must exactly reproduce final_weights."
        )


# ---------------------------------------------------------------------------
# Fixture F-B: one outlier revises one ticker → consensus moves on exactly that name
# ---------------------------------------------------------------------------

class TestOneRevises:
    """F-B + A3-b: one outlier revises exactly one ticker."""

    TICKERS = ["AAPL", "MSFT", "GOOG"]
    OUTLIERS = {"growth", "cta-systematic-macro"}

    def _build(self, growth_msft_r2_weight: float):
        # Round 1: growth adds MSFT at 0.10; round 2: growth revises MSFT weight.
        w: dict[str, dict[str, tuple[str, float]]] = {
            "value":                  {"AAPL": ("add", 0.10), "MSFT": ("hold", 0.0),  "GOOG": ("hold", 0.0)},
            "growth":                 {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.10),  "GOOG": ("hold", 0.0)},
            "discretionary-macro":    {"AAPL": ("add", 0.08), "MSFT": ("hold", 0.0),  "GOOG": ("hold", 0.0)},
            "cta-systematic-macro":   {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0),  "GOOG": ("add", 0.15)},
            "technical":              {"AAPL": ("add", 0.10), "MSFT": ("add", 0.10),  "GOOG": ("hold", 0.0)},
            "quant-systematic":       {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.11),  "GOOG": ("hold", 0.0)},
            "risk-officer":           {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0),  "GOOG": ("hold", 0.0)},
        }
        round1 = _make_round1_stances(self.TICKERS, w)
        provisional = blend_consensus(round1, _CFG)

        # growth revises MSFT; cta defends all.
        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona == "growth" and s.ticker == "MSFT":
                # revised weight
                action = "add" if growth_msft_r2_weight > 0 else "hold"
                round2.append(_stance("growth", "MSFT", action, growth_msft_r2_weight, round_=2))
            elif s.persona == "growth":
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))
            elif s.persona == "cta-systematic-macro":
                # defend all
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        return round1, round2, provisional

    def test_final_differs_from_provisional(self):
        round1, round2, provisional = self._build(growth_msft_r2_weight=0.18)
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert result.final_weights != pytest.approx(provisional, abs=1e-9), (
            "One outlier revised — final must differ from provisional."
        )

    def test_delta_contains_only_revised_ticker(self):
        round1, round2, provisional = self._build(growth_msft_r2_weight=0.18)
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        # Only MSFT changed — delta must carry MSFT only.
        assert set(result.delta.keys()) == {"MSFT"}, (
            f"Delta must carry only MSFT (the revised ticker), got {set(result.delta.keys())}."
        )

    def test_delta_direction_correct(self):
        """Increasing growth's MSFT weight must raise the MSFT consensus weight."""
        round1_r, round2_r, provisional = self._build(growth_msft_r2_weight=0.18)
        result = resynthesize_consensus(round1_r, round2_r, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert result.delta["MSFT"] > 0, (
            "growth raised MSFT weight in round=2 → MSFT delta must be positive."
        )

    def test_invariant_passes(self):
        round1, round2, provisional = self._build(growth_msft_r2_weight=0.18)
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        passed, reasons = _invariant_passes(result.final_weights)
        assert passed, f"Invariant failed: {reasons}"

    def test_blend_reuse_proof(self):
        round1, round2, provisional = self._build(growth_msft_r2_weight=0.18)
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert verify_blend_reuse(result.merged_stances, result.final_weights, _CFG)

    def test_non_outlier_stances_unchanged_in_merged(self):
        """Non-outlier round=1 stances must appear verbatim in merged_stances."""
        round1, round2, provisional = self._build(growth_msft_r2_weight=0.18)
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        non_outlier_r1 = {(s.persona, s.ticker): s for s in round1
                          if s.persona not in self.OUTLIERS}
        non_outlier_merged = {(s.persona, s.ticker): s for s in result.merged_stances
                               if s.persona not in self.OUTLIERS}
        assert non_outlier_r1 == non_outlier_merged, (
            "Non-outlier stances must be identical between round=1 and merged set."
        )


# ---------------------------------------------------------------------------
# Fixture F-C: both outliers revise toward each other (divergent positions converge)
# ---------------------------------------------------------------------------

class TestBothReviseConverge:
    """F-C: outliers held opposite extreme positions; round=2 they converge."""

    TICKERS = ["AAPL", "MSFT"]
    OUTLIERS = {"growth", "cta-systematic-macro"}

    def _build(self):
        # Round 1: growth all-in AAPL (0.20), cta all-in MSFT (0.20).
        # All other 5 personas hold both.
        w: dict[str, dict[str, tuple[str, float]]] = {
            "value":                  {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0)},
            "growth":                 {"AAPL": ("add", 0.20), "MSFT": ("hold", 0.0)},
            "discretionary-macro":    {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0)},
            "cta-systematic-macro":   {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.20)},
            "technical":              {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0)},
            "quant-systematic":       {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0)},
            "risk-officer":           {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0)},
        }
        round1 = _make_round1_stances(self.TICKERS, w)
        provisional = blend_consensus(round1, _CFG)

        # Round 2: growth reduces AAPL to 0.10, cta reduces MSFT to 0.10 — converge.
        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona == "growth":
                new_w = 0.10 if s.ticker == "AAPL" else 0.0
                action = "add" if new_w > 0 else "hold"
                round2.append(_stance("growth", s.ticker, action, new_w, round_=2))
            elif s.persona == "cta-systematic-macro":
                new_w = 0.10 if s.ticker == "MSFT" else 0.0
                action = "add" if new_w > 0 else "hold"
                round2.append(_stance("cta-systematic-macro", s.ticker, action, new_w, round_=2))

        return round1, round2, provisional

    def test_both_tickers_in_delta(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        # Both AAPL and MSFT changed → both in delta.
        assert "AAPL" in result.delta and "MSFT" in result.delta, (
            f"Both AAPL and MSFT should appear in delta; got {set(result.delta.keys())}."
        )

    def test_aapl_weight_decreases(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert result.delta["AAPL"] < 0, "growth reduced AAPL weight → AAPL delta must be negative."

    def test_msft_weight_decreases(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert result.delta["MSFT"] < 0, "cta reduced MSFT weight → MSFT delta must be negative."

    def test_invariant_passes(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        passed, reasons = _invariant_passes(result.final_weights)
        assert passed, f"Invariant failed: {reasons}"

    def test_blend_reuse_proof(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert verify_blend_reuse(result.merged_stances, result.final_weights, _CFG)


# ---------------------------------------------------------------------------
# Fixture F-D: both outliers revise apart (aligned positions diverge)
# ---------------------------------------------------------------------------

class TestBothReviseApart:
    """F-D: outliers started aligned; round=2 they move apart."""

    TICKERS = ["AAPL", "MSFT", "GOOG"]
    OUTLIERS = {"growth", "cta-systematic-macro"}

    def _build(self):
        # Round 1: growth + cta both add AAPL at 0.08 (aligned).
        # Round 2: growth raises AAPL to 0.18; cta drops to 0.0 exit.
        w: dict[str, dict[str, tuple[str, float]]] = {
            "value":                  {"AAPL": ("add", 0.08), "MSFT": ("add", 0.05), "GOOG": ("hold", 0.0)},
            "growth":                 {"AAPL": ("add", 0.08), "MSFT": ("hold", 0.0),  "GOOG": ("add", 0.07)},
            "discretionary-macro":    {"AAPL": ("add", 0.06), "MSFT": ("add", 0.06), "GOOG": ("hold", 0.0)},
            "cta-systematic-macro":   {"AAPL": ("add", 0.08), "MSFT": ("add", 0.08), "GOOG": ("hold", 0.0)},
            "technical":              {"AAPL": ("add", 0.09), "MSFT": ("hold", 0.0),  "GOOG": ("add", 0.06)},
            "quant-systematic":       {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.07), "GOOG": ("hold", 0.0)},
            "risk-officer":           {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0),  "GOOG": ("hold", 0.0)},
        }
        round1 = _make_round1_stances(self.TICKERS, w)
        provisional = blend_consensus(round1, _CFG)

        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona == "growth":
                if s.ticker == "AAPL":
                    round2.append(_stance("growth", "AAPL", "add", 0.18, round_=2))
                else:
                    round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))
            elif s.persona == "cta-systematic-macro":
                if s.ticker == "AAPL":
                    round2.append(_stance("cta-systematic-macro", "AAPL", "exit", 0.0, round_=2))
                else:
                    round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        return round1, round2, provisional

    def test_aapl_in_delta(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert "AAPL" in result.delta, "AAPL changed in round=2 → must appear in delta."

    def test_other_tickers_not_in_delta(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        # MSFT and GOOG were not revised by either outlier → should not be in delta.
        # (Growth defended MSFT and GOOG by repeating r1 values; cta defended them too.)
        for ticker in ["MSFT", "GOOG"]:
            assert ticker not in result.delta, (
                f"{ticker} was not revised → must not appear in delta; "
                f"got delta={result.delta}"
            )

    def test_invariant_passes(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        passed, reasons = _invariant_passes(result.final_weights)
        assert passed, f"Invariant failed: {reasons}"

    def test_blend_reuse_proof(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert verify_blend_reuse(result.merged_stances, result.final_weights, _CFG)


# ---------------------------------------------------------------------------
# P-1: Explicit blend-reuse proof (AC1 verification)
# ---------------------------------------------------------------------------

class TestAC1BlendReuseProof:
    """Prove that resynthesize_consensus uses blend_consensus and not parallel math.

    Method: call resynthesize_consensus to get final_weights + merged_stances,
    then independently call blend_consensus(merged_stances, config) and assert
    the two weight dicts are identical.  If resynthesis used any alternative
    formula, the weights would diverge.
    """

    def test_recomputed_weights_match_final(self):
        tickers = ["AAPL", "MSFT", "GOOG"]
        w = {
            "value":                  {"AAPL": ("add", 0.10), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0)},
            "growth":                 {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.15), "GOOG": ("hold", 0.0)},
            "discretionary-macro":    {"AAPL": ("add", 0.08), "MSFT": ("hold", 0.0), "GOOG": ("add", 0.09)},
            "cta-systematic-macro":   {"AAPL": ("add", 0.12), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0)},
            "technical":              {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.10), "GOOG": ("hold", 0.0)},
            "quant-systematic":       {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.11), "GOOG": ("hold", 0.0)},
            "risk-officer":           {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0)},
        }
        round1 = _make_round1_stances(tickers, w)
        provisional = blend_consensus(round1, _CFG)

        outliers = {"growth", "cta-systematic-macro"}
        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona == "growth" and s.ticker == "MSFT":
                round2.append(_stance("growth", "MSFT", "add", 0.18, round_=2))
            elif s.persona == "growth":
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))
            elif s.persona == "cta-systematic-macro" and s.ticker == "AAPL":
                round2.append(_stance("cta-systematic-macro", "AAPL", "add", 0.05, round_=2))
            elif s.persona == "cta-systematic-macro":
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        result = resynthesize_consensus(round1, round2, outlier_personas=outliers,
                                        provisional_weights=provisional, config=_CFG)

        # Direct call to blend_consensus on the same merged set.
        direct = blend_consensus(result.merged_stances, _CFG)
        assert direct == pytest.approx(result.final_weights, abs=1e-9), (
            "blend_consensus(merged_stances) must reproduce final_weights exactly — "
            "proving no parallel blend math."
        )

    def test_verify_blend_reuse_helper_returns_true(self):
        tickers = ["AAPL", "MSFT"]
        w = {
            p: {"AAPL": ("add", 0.10) if p == "growth" else ("hold", 0.0),
                "MSFT": ("add", 0.10) if p == "value" else ("hold", 0.0)}
            for p in _ALL_PERSONAS
        }
        round1 = _make_round1_stances(tickers, w)
        provisional = blend_consensus(round1, _CFG)
        outliers = {"growth", "cta-systematic-macro"}
        round2 = [_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2)
                  for s in round1 if s.persona in outliers]

        result = resynthesize_consensus(round1, round2, outlier_personas=outliers,
                                        provisional_weights=provisional, config=_CFG)
        assert verify_blend_reuse(result.merged_stances, result.final_weights, _CFG)

    def test_verify_blend_reuse_returns_false_on_tampered_weights(self):
        """Verify that verify_blend_reuse catches a manipulated weights dict."""
        tickers = ["AAPL", "MSFT"]
        w = {
            p: {"AAPL": ("add", 0.10) if p == "growth" else ("hold", 0.0),
                "MSFT": ("add", 0.10) if p == "value" else ("hold", 0.0)}
            for p in _ALL_PERSONAS
        }
        round1 = _make_round1_stances(tickers, w)
        provisional = blend_consensus(round1, _CFG)
        outliers = {"growth", "cta-systematic-macro"}
        round2 = [_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2)
                  for s in round1 if s.persona in outliers]

        result = resynthesize_consensus(round1, round2, outlier_personas=outliers,
                                        provisional_weights=provisional, config=_CFG)

        # Tamper with the final_weights.
        tampered = dict(result.final_weights)
        if tampered:
            first_ticker = next(iter(tampered))
            tampered[first_ticker] = tampered[first_ticker] + 0.05

        assert not verify_blend_reuse(result.merged_stances, tampered, _CFG), (
            "verify_blend_reuse must return False when weights were tampered."
        )


# ---------------------------------------------------------------------------
# P-2: Invariant checks across all fixture variants
# ---------------------------------------------------------------------------

class TestInvariantAcrossFixtures:
    """Systematic invariant check: every final_weights passes check_fully_invested."""

    def _run(self, tickers, w_by_persona, outliers_set, r2_overrides):
        """Build, resynthesize, return (result, passed, reasons)."""
        round1 = _make_round1_stances(tickers, w_by_persona)
        provisional = blend_consensus(round1, _CFG)
        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona in outliers_set:
                override = r2_overrides.get((s.persona, s.ticker))
                if override:
                    action, tw = override
                    round2.append(_stance(s.persona, s.ticker, action, tw, round_=2))
                else:
                    round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))
        result = resynthesize_consensus(round1, round2, outlier_personas=outliers_set,
                                        provisional_weights=provisional, config=_CFG)
        passed, reasons = _invariant_passes(result.final_weights)
        return result, passed, reasons

    def test_max_weight_cap_respected_after_revision(self):
        """If an outlier revises to max_position_weight, Σ still ≤ 1."""
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        w = {
            "value":               {"AAPL": ("add", 0.18), "MSFT": ("hold", 0.0),  "GOOG": ("hold", 0.0), "AMZN": ("hold", 0.0)},
            "growth":              {"AAPL": ("add", 0.10), "MSFT": ("add", 0.10),  "GOOG": ("hold", 0.0), "AMZN": ("hold", 0.0)},
            "discretionary-macro": {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.12),  "GOOG": ("add", 0.08), "AMZN": ("hold", 0.0)},
            "cta-systematic-macro":{"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0),  "GOOG": ("add", 0.10), "AMZN": ("add", 0.15)},
            "technical":           {"AAPL": ("add", 0.10), "MSFT": ("hold", 0.0),  "GOOG": ("add", 0.10), "AMZN": ("hold", 0.0)},
            "quant-systematic":    {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.09),  "GOOG": ("hold", 0.0), "AMZN": ("add", 0.11)},
            "risk-officer":        {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0),  "GOOG": ("hold", 0.0), "AMZN": ("hold", 0.0)},
        }
        r2_overrides = {
            ("growth", "AAPL"): ("add", 0.20),  # revise up to cap
            ("growth", "MSFT"): ("add", 0.10),
            ("growth", "GOOG"): ("hold", 0.0),
            ("growth", "AMZN"): ("hold", 0.0),
        }
        result, passed, reasons = self._run(tickers, w, {"growth", "cta-systematic-macro"}, r2_overrides)
        assert passed, f"Invariant failed with cap-revision: {reasons}"

    def test_all_exit_yields_100pct_cash(self):
        """If ALL 7 personas exit every ticker, final_weights is empty → 100% cash."""
        tickers = ["AAPL", "MSFT"]
        w = {p: {"AAPL": ("exit", 0.0), "MSFT": ("exit", 0.0)} for p in _ALL_PERSONAS}
        round1 = _make_round1_stances(tickers, w)
        provisional = blend_consensus(round1, _CFG)
        assert provisional == {}  # all exit → empty provisional
        round2 = [_stance(s.persona, s.ticker, "exit", 0.0, round_=2)
                  for s in round1 if s.persona in {"growth", "cta-systematic-macro"}]
        result = resynthesize_consensus(round1, round2,
                                        outlier_personas={"growth", "cta-systematic-macro"},
                                        provisional_weights=provisional, config=_CFG)
        assert result.final_weights == {}
        # cash = 1 - 0 = 1.0 — invariant passes trivially (no positions to check).
        passed, reasons = check_fully_invested({}, 1.0, _MAX_W)
        assert passed


# ---------------------------------------------------------------------------
# AC-3 explicit cells (defend → no-op; revise → moves EXACTLY those names)
# ---------------------------------------------------------------------------

class TestAC3ReviseMovesExactly:
    """Explicit AC3 tests: final differs in EXACTLY the names the outlier moved."""

    TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN"]
    OUTLIERS = {"growth", "cta-systematic-macro"}

    def _round1(self):
        w = {
            "value":                  {"AAPL": ("add", 0.10), "MSFT": ("add", 0.08), "GOOG": ("hold", 0.0), "AMZN": ("hold", 0.0)},
            "growth":                 {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.12), "GOOG": ("add", 0.10), "AMZN": ("hold", 0.0)},
            "discretionary-macro":    {"AAPL": ("add", 0.07), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0), "AMZN": ("add", 0.09)},
            "cta-systematic-macro":   {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0), "AMZN": ("add", 0.11)},
            "technical":              {"AAPL": ("add", 0.09), "MSFT": ("add", 0.07), "GOOG": ("hold", 0.0), "AMZN": ("hold", 0.0)},
            "quant-systematic":       {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.10), "GOOG": ("add", 0.08), "AMZN": ("hold", 0.0)},
            "risk-officer":           {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0), "GOOG": ("hold", 0.0), "AMZN": ("hold", 0.0)},
        }
        return _make_round1_stances(self.TICKERS, w)

    def test_both_defend_final_equals_provisional(self):
        round1 = self._round1()
        provisional = blend_consensus(round1, _CFG)
        round2 = [_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2)
                  for s in round1 if s.persona in self.OUTLIERS]
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert result.final_weights == pytest.approx(provisional, abs=1e-9)
        assert result.delta == {}

    def test_growth_revises_goog_only_delta_is_goog(self):
        """growth revises GOOG; cta defends all → delta must be {GOOG} only."""
        round1 = self._round1()
        provisional = blend_consensus(round1, _CFG)

        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona == "growth" and s.ticker == "GOOG":
                round2.append(_stance("growth", "GOOG", "add", 0.18, round_=2))
            elif s.persona in self.OUTLIERS:
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert set(result.delta.keys()) == {"GOOG"}, (
            f"Only GOOG was revised — delta must be {{GOOG}}, got {set(result.delta.keys())}."
        )

    def test_cta_revises_amzn_only_delta_is_amzn(self):
        """cta revises AMZN; growth defends all → delta must be {AMZN} only."""
        round1 = self._round1()
        provisional = blend_consensus(round1, _CFG)

        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona == "cta-systematic-macro" and s.ticker == "AMZN":
                round2.append(_stance("cta-systematic-macro", "AMZN", "add", 0.05, round_=2))
            elif s.persona in self.OUTLIERS:
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert set(result.delta.keys()) == {"AMZN"}, (
            f"Only AMZN was revised — delta must be {{AMZN}}, got {set(result.delta.keys())}."
        )

    def test_both_revise_different_tickers_delta_is_both(self):
        """growth revises GOOG; cta revises AMZN → delta is {GOOG, AMZN}."""
        round1 = self._round1()
        provisional = blend_consensus(round1, _CFG)

        round2: list[AgentStancePayload] = []
        for s in round1:
            if s.persona == "growth" and s.ticker == "GOOG":
                round2.append(_stance("growth", "GOOG", "add", 0.18, round_=2))
            elif s.persona == "cta-systematic-macro" and s.ticker == "AMZN":
                round2.append(_stance("cta-systematic-macro", "AMZN", "add", 0.05, round_=2))
            elif s.persona in self.OUTLIERS:
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert set(result.delta.keys()) == {"GOOG", "AMZN"}, (
            f"GOOG + AMZN revised → delta must be {{GOOG, AMZN}}, got {set(result.delta.keys())}."
        )


# ---------------------------------------------------------------------------
# Edge / error cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_outlier_personas_raises_value_error(self):
        tickers = ["AAPL"]
        w = {p: {"AAPL": ("add", 0.10)} for p in _ALL_PERSONAS}
        round1 = _make_round1_stances(tickers, w)
        provisional = blend_consensus(round1, _CFG)
        with pytest.raises(ValueError, match="outlier_personas must be non-empty"):
            resynthesize_consensus(round1, [], outlier_personas=set(),
                                   provisional_weights=provisional, config=_CFG)

    def test_round2_stance_for_extra_ticker_appended_with_warning(self, caplog):
        """A round=2 stance for a ticker absent from round=1 is appended with a warning."""
        tickers = ["AAPL"]
        w = {p: {"AAPL": ("add", 0.10) if p == "growth" else ("hold", 0.0)}
             for p in _ALL_PERSONAS}
        round1 = _make_round1_stances(tickers, w)
        provisional = blend_consensus(round1, _CFG)

        # round=2 includes a stance for NVDA which is NOT in round=1.
        round2_growth_aapl = _stance("growth", "AAPL", "add", 0.10, round_=2)
        round2_growth_nvda = _stance("growth", "NVDA", "add", 0.08, round_=2)
        round2_cta = [_stance("cta-systematic-macro", "AAPL", "hold", 0.0, round_=2)]

        with caplog.at_level(logging.WARNING, logger="round_table_portfolio.orchestrator.resynthesis"):
            result = resynthesize_consensus(
                round1,
                [round2_growth_aapl, round2_growth_nvda] + round2_cta,
                outlier_personas={"growth", "cta-systematic-macro"},
                provisional_weights=provisional,
                config=_CFG,
            )
        assert any("NVDA" in msg for msg in caplog.messages), (
            "A warning must be logged for a round=2 stance with no matching round=1 ticker."
        )
        # NVDA should appear in merged_stances.
        merged_tickers = {(s.persona, s.ticker) for s in result.merged_stances}
        assert ("growth", "NVDA") in merged_tickers

    def test_result_carries_outlier_personas_frozenset(self):
        tickers = ["AAPL"]
        w = {p: {"AAPL": ("add", 0.10) if p == "growth" else ("hold", 0.0)}
             for p in _ALL_PERSONAS}
        round1 = _make_round1_stances(tickers, w)
        provisional = blend_consensus(round1, _CFG)
        outliers = {"growth", "cta-systematic-macro"}
        round2 = [_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2)
                  for s in round1 if s.persona in outliers]
        result = resynthesize_consensus(round1, round2, outlier_personas=outliers,
                                        provisional_weights=provisional, config=_CFG)
        assert result.outlier_personas == frozenset(outliers)
        assert isinstance(result.outlier_personas, frozenset)

    def test_provisional_weights_stored_unchanged(self):
        tickers = ["AAPL", "MSFT"]
        w = {
            "value":               {"AAPL": ("add", 0.10), "MSFT": ("add", 0.10)},
            "growth":              {"AAPL": ("add", 0.15), "MSFT": ("hold", 0.0)},
            "discretionary-macro": {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.12)},
            "cta-systematic-macro":{"AAPL": ("hold", 0.0), "MSFT": ("add", 0.08)},
            "technical":           {"AAPL": ("add", 0.09), "MSFT": ("hold", 0.0)},
            "quant-systematic":    {"AAPL": ("hold", 0.0), "MSFT": ("add", 0.11)},
            "risk-officer":        {"AAPL": ("hold", 0.0), "MSFT": ("hold", 0.0)},
        }
        round1 = _make_round1_stances(tickers, w)
        provisional = blend_consensus(round1, _CFG)
        original_prov = dict(provisional)

        outliers = {"growth", "cta-systematic-macro"}
        round2 = []
        for s in round1:
            if s.persona == "growth" and s.ticker == "AAPL":
                round2.append(_stance("growth", "AAPL", "add", 0.19, round_=2))
            elif s.persona in outliers:
                round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        result = resynthesize_consensus(round1, round2, outlier_personas=outliers,
                                        provisional_weights=provisional, config=_CFG)
        assert result.provisional_weights == pytest.approx(original_prov, abs=1e-9)


# ---------------------------------------------------------------------------
# Real-W24 anchored case (provenance: 2026-W24 ledger run, sanitized)
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
_W24_FIXTURE = _FIXTURES_DIR / "stances_2026_w24_round1.json"


def _load_w24_stances() -> list[AgentStancePayload]:
    """Load the W24 round=1 fixture into AgentStancePayload objects."""
    rows = json.loads(_W24_FIXTURE.read_text())
    return [
        AgentStancePayload(
            week_id="2026-W24",
            persona=r["persona"],
            ticker=r["ticker"],
            round=1,
            action=r["action"],
            target_weight=float(r["target_weight"]),
            confidence=int(r["confidence"]),
            rationale="",
            user_id="andrew",
            roster_version=1,
            enhancement_version=1,
        )
        for r in rows
    ]


@pytest.mark.skipif(not _W24_FIXTURE.exists(), reason="W24 fixture not found")
class TestRealW24:
    """Real-W24 anchored case.

    Provenance: agent_stances WHERE week_id='2026-W24' AND round=1, 280 rows
    (7 × 40 tickers).  Source: live 2026-W24 run, state/ledger.db.
    PII: none — equity research / market data only.

    Outliers (from M3-001): growth + cta-systematic-macro.

    Stubbed round=2 stances (designed to show a visible shift):
      growth revises:
        DELL  0.10 → 0.05  (growth reduces conviction)
        NVDA  hold 0.0 → add 0.08  (growth adds a new position)
      cta revises:
        CHTR  exit 0.0 → hold 0.04  (cta softens exit)
        DELL  0.10 → 0.15  (cta raises conviction — moves opposite to growth)
      All other tickers: defended (round=2 = round=1 weights).
    """

    OUTLIERS = {"growth", "cta-systematic-macro"}
    # Expected tickers that should appear in delta given the stub revisions above.
    EXPECTED_DELTA_TICKERS = {"DELL", "NVDA", "CHTR"}

    def _build(self):
        round1 = _load_w24_stances()
        provisional = blend_consensus(round1, _CFG)

        # Build a lookup of round=1 stances by (persona, ticker) for stub construction.
        r1_lookup: dict[tuple[str, str], AgentStancePayload] = {
            (s.persona, s.ticker): s for s in round1
        }

        round2: list[AgentStancePayload] = []
        # Stub revisions for growth.
        growth_r2_overrides = {
            "DELL": ("add", 0.05),    # reduce from 0.10
            "NVDA": ("add", 0.08),    # new add (was hold 0.0)
        }
        for s in round1:
            if s.persona == "growth":
                if s.ticker in growth_r2_overrides:
                    action, tw = growth_r2_overrides[s.ticker]
                    round2.append(_stance("growth", s.ticker, action, tw, round_=2))
                else:
                    round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        # Stub revisions for cta-systematic-macro.
        cta_r2_overrides = {
            "CHTR": ("hold", 0.04),   # soften from exit 0.0
            "DELL": ("add", 0.15),    # raise from 0.10
        }
        for s in round1:
            if s.persona == "cta-systematic-macro":
                if s.ticker in cta_r2_overrides:
                    action, tw = cta_r2_overrides[s.ticker]
                    round2.append(_stance("cta-systematic-macro", s.ticker, action, tw, round_=2))
                else:
                    round2.append(_stance(s.persona, s.ticker, s.action, s.target_weight, round_=2))

        return round1, round2, provisional

    def test_provisional_not_equal_final(self):
        """Revisions exist → final must differ from provisional."""
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        # At least one weight must have moved.
        all_tickers = set(provisional) | set(result.final_weights)
        any_moved = any(
            abs(result.final_weights.get(t, 0.0) - provisional.get(t, 0.0)) > 1e-9
            for t in all_tickers
        )
        assert any_moved, "Stub revisions exist → final must differ from provisional."

    def test_delta_contains_revised_tickers(self):
        """Delta must carry at minimum the tickers that were revised."""
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        for ticker in self.EXPECTED_DELTA_TICKERS:
            assert ticker in result.delta, (
                f"{ticker} was revised in stub round=2 → must appear in delta; "
                f"delta keys={set(result.delta.keys())}."
            )

    def test_invariant_passes(self):
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        passed, reasons = _invariant_passes(result.final_weights)
        assert passed, f"W24 invariant failed: {reasons}"

    def test_blend_reuse_proof(self):
        """AC1: same blend_consensus function, no parallel math."""
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert verify_blend_reuse(result.merged_stances, result.final_weights, _CFG), (
            "W24: blend_consensus(merged_stances) must reproduce final_weights exactly."
        )

    def test_merged_set_size_is_280(self):
        """Merged set must be the same size as round=1 (280 rows, 7×40)."""
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        assert len(result.merged_stances) == 280

    def test_non_outlier_stances_unchanged(self):
        """The 5 non-outlier personas' stances must be identical between round=1 and merged."""
        round1, round2, provisional = self._build()
        result = resynthesize_consensus(round1, round2, outlier_personas=self.OUTLIERS,
                                        provisional_weights=provisional, config=_CFG)
        non_outlier_r1 = {(s.persona, s.ticker): (s.action, s.target_weight)
                          for s in round1 if s.persona not in self.OUTLIERS}
        non_outlier_merged = {(s.persona, s.ticker): (s.action, s.target_weight)
                               for s in result.merged_stances if s.persona not in self.OUTLIERS}
        assert non_outlier_r1 == non_outlier_merged, (
            "Non-outlier stances must be identical in round=1 and merged set."
        )
