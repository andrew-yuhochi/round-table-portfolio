"""Unit tests for Components 21 + 22 — dissent_metric and outlier_selection.

Coverage plan
-------------
A. 20-cell mapping table (4 actions × 5 confidence levels) — parametrized.
B. 5+ synthetic stance-set fixtures × expected dissent_score + contested flag.
C. Component 22 selection cells (≥4 fixtures + determinism + brute-force cross-check).
D. contested_week threshold-from-config assertion (no literal 0.50 in code path).
E. Real-data DEF-004 regression: 2026-W24 Round-1 stances (sanitized fixture).
   Provenance: live 2026-W24 run, quality-logs/TASK-M2-011.md.
   PII: none — equity research / market data only; sanitized snapshot.
"""
from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from round_table_portfolio.orchestrator.dissent import (
    DissentConfig,
    DissentResult,
    OutlierSelection,
    compute_dissent,
    load_dissent_config,
    select_outliers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "stances_2026_w24_round1.json"
)

# Default config used for most tests — matches thresholds.yaml defaults.
_DEFAULT_CFG = DissentConfig(
    contested_week_threshold=0.50,
    action_direction_map={"add": 1.0, "hold": 0.0, "reduce": -0.5, "exit": -1.0},
    n_outliers=2,
    divergence_tiebreak="alpha_asc",
)


def _make_stances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows as plain dicts — compute_dissent accepts dict or object."""
    return rows


def _debate_set_from(stances: list[dict]) -> list[str]:
    return list(dict.fromkeys(s["ticker"] for s in stances))


# ---------------------------------------------------------------------------
# A. 20-cell mapping: _signed_score via compute_dissent (public API)
# ---------------------------------------------------------------------------

# Hand-computed expected values: s = direction × (confidence / 5)
# add:    +1.0;  hold: 0.0;  reduce: -0.5;  exit: -1.0
_MAPPING_CASES: list[tuple[str, int, float]] = [
    ("add",    1, +0.2),
    ("add",    2, +0.4),
    ("add",    3, +0.6),
    ("add",    4, +0.8),
    ("add",    5, +1.0),
    ("hold",   1,  0.0),
    ("hold",   2,  0.0),
    ("hold",   3,  0.0),
    ("hold",   4,  0.0),
    ("hold",   5,  0.0),
    ("reduce", 1, -0.1),
    ("reduce", 2, -0.2),
    ("reduce", 3, -0.3),
    ("reduce", 4, -0.4),
    ("reduce", 5, -0.5),
    ("exit",   1, -0.2),
    ("exit",   2, -0.4),
    ("exit",   3, -0.6),
    ("exit",   4, -0.8),
    ("exit",   5, -1.0),
]


@pytest.mark.parametrize("action,conf,expected_s", _MAPPING_CASES)
def test_signed_score_mapping(action: str, conf: int, expected_s: float) -> None:
    """Single-ticker two-persona fixture isolates one action×confidence cell.

    Using two personas with equal and opposite scores to back out the individual
    score from the per-ticker σ would add indirection — instead we use two
    identical stances so σ=0 and can then verify the panel_mean equals expected_s
    by checking that a third persona at s=0 (HOLD) contributes divergence = |expected_s|.
    Simpler direct path: build a 1-ticker 2-persona stances set where both personas
    take the same action+confidence, verify per_persona_divergence is 0 (mean==each),
    and verify per_ticker_sigma == 0.  Then for the second persona use action+conf
    that gives s=0 (hold@any), so the per-ticker σ equals |expected_s| / sqrt(2) * sqrt(1).
    Actually the cleanest approach: use a 2-persona set, one at the target, one at hold@1
    (s=0), and derive expected σ analytically.
    """
    # Two personas: target action+conf (s=expected_s) and hold@1 (s=0.0).
    stances = [
        {"persona": "alpha", "ticker": "X", "action": action, "confidence": conf},
        {"persona": "beta",  "ticker": "X", "action": "hold", "confidence": 1},
    ]
    result = compute_dissent(stances, ["X"], _DEFAULT_CFG)

    # Per-ticker mean_s = (expected_s + 0) / 2
    mean_s = expected_s / 2.0
    # Per-ticker σ = sqrt(((expected_s - mean_s)^2 + (0 - mean_s)^2) / 2)
    expected_sigma = math.sqrt(
        ((expected_s - mean_s) ** 2 + (0.0 - mean_s) ** 2) / 2.0
    )
    assert abs(result.per_ticker_sigma["X"] - expected_sigma) < 1e-9

    # Also verify alpha's divergence = |expected_s - mean_s|
    expected_alpha_div = abs(expected_s - mean_s)
    assert abs(result.per_persona_divergence["alpha"] - expected_alpha_div) < 1e-9
    assert abs(result.per_persona_divergence["beta"] - abs(0.0 - mean_s)) < 1e-9


# ---------------------------------------------------------------------------
# B. Synthetic stance-set fixtures × expected dissent_score + contested flag
# ---------------------------------------------------------------------------


class TestSyntheticFixtures:
    """Five synthetic stance sets covering the spec's required cases."""

    def test_all_agree_add5(self) -> None:
        """All personas ADD@5 on all tickers → every σ = 0 → dissent ≈ 0, not contested."""
        tickers = ["A", "B", "C"]
        personas = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        stances = [
            {"persona": p, "ticker": t, "action": "add", "confidence": 5}
            for p in personas
            for t in tickers
        ]
        result = compute_dissent(stances, tickers, _DEFAULT_CFG)
        assert result.dissent_score == pytest.approx(0.0, abs=1e-9)
        assert result.contested_week is False

    def test_maximally_split_half_add5_half_exit5(self) -> None:
        """4 ADD@5 (+1.0) and 3 EXIT@5 (-1.0) on every ticker.

        mean_s = (4×1 + 3×(-1)) / 7 = 1/7
        variance = (4×(1 - 1/7)^2 + 3×(-1 - 1/7)^2) / 7
               = (4×(6/7)^2 + 3×(-8/7)^2) / 7
               = (4×36/49 + 3×64/49) / 7
               = (144 + 192) / (49 × 7)
               = 336 / 343
        σ = sqrt(336/343) ≈ 0.9899
        dissent_score = 0.9899 (same for every ticker) → contested (≥ 0.50).
        """
        tickers = ["T1", "T2"]
        stances = []
        for t in tickers:
            for i in range(4):
                stances.append({"persona": f"adder_{i}", "ticker": t, "action": "add", "confidence": 5})
            for i in range(3):
                stances.append({"persona": f"exiter_{i}", "ticker": t, "action": "exit", "confidence": 5})
        result = compute_dissent(stances, tickers, _DEFAULT_CFG)
        expected_sigma = math.sqrt(336 / 343)
        assert result.dissent_score == pytest.approx(expected_sigma, abs=1e-6)
        assert result.contested_week is True

    def test_one_outlier_vs_six_aligned(self) -> None:
        """6 ADD@5 (+1.0), 1 EXIT@5 (-1.0) on one ticker.

        mean_s = (6×1 + 1×(-1)) / 7 = 5/7
        variance = (6×(1 - 5/7)^2 + 1×(-1 - 5/7)^2) / 7
               = (6×(2/7)^2 + 1×(12/7)^2) / 7
               = (6×4/49 + 144/49) / 7
               = (24 + 144) / (49 × 7)
               = 168 / 343
        σ = sqrt(168/343) ≈ 0.7003
        dissent_score = σ → contested.
        The lone EXIT@5 persona has highest divergence (= |(-1) - 5/7| = 12/7).
        """
        tickers = ["X"]
        stances = [
            {"persona": f"aligned_{i}", "ticker": "X", "action": "add", "confidence": 5}
            for i in range(6)
        ]
        stances.append({"persona": "outlier", "ticker": "X", "action": "exit", "confidence": 5})
        result = compute_dissent(stances, tickers, _DEFAULT_CFG)
        expected_sigma = math.sqrt(168 / 343)
        assert result.dissent_score == pytest.approx(expected_sigma, abs=1e-6)
        assert result.contested_week is True
        # Outlier has highest divergence
        max_persona = max(result.per_persona_divergence, key=lambda p: result.per_persona_divergence[p])
        assert max_persona == "outlier"

    def test_split_on_cash_not_names(self) -> None:
        """Two tickers: all HOLD on T1, split ADD/EXIT on T2.

        T1: all HOLD → σ=0.
        T2: 4 ADD@3 (s=0.6), 3 EXIT@3 (s=-0.6).
        mean_s_T2 = (4×0.6 + 3×(-0.6)) / 7 = 0.6/7
        dissent_score = (0 + σ_T2) / 2

        σ_T2: mean = 0.6/7 ≈ 0.08571
        var = (4×(0.6 - 0.0857)^2 + 3×(-0.6 - 0.0857)^2) / 7
        """
        personas_add = [f"pa_{i}" for i in range(4)]
        personas_exit = [f"pe_{i}" for i in range(3)]
        all_personas = personas_add + personas_exit
        stances = []
        # T1: everyone HOLD
        for p in all_personas:
            stances.append({"persona": p, "ticker": "T1", "action": "hold", "confidence": 3})
        # T2: split ADD/EXIT
        for p in personas_add:
            stances.append({"persona": p, "ticker": "T2", "action": "add", "confidence": 3})
        for p in personas_exit:
            stances.append({"persona": p, "ticker": "T2", "action": "exit", "confidence": 3})

        result = compute_dissent(stances, ["T1", "T2"], _DEFAULT_CFG)
        assert result.per_ticker_sigma["T1"] == pytest.approx(0.0, abs=1e-9)
        assert result.per_ticker_sigma["T2"] > 0.0
        # dissent_score = mean of [0, σ_T2]
        assert result.dissent_score == pytest.approx(result.per_ticker_sigma["T2"] / 2, abs=1e-9)
        # Not contested because T1 pulls the mean down
        # (σ_T2 for 4×0.6 + 3×-0.6 case is < 1.0; mean with 0 will be < 0.50)

    def test_everyone_hold(self) -> None:
        """All personas HOLD on all tickers → every s=0 → σ=0 → dissent=0, not contested."""
        tickers = ["A", "B"]
        stances = [
            {"persona": f"p{i}", "ticker": t, "action": "hold", "confidence": c}
            for i in range(7)
            for t, c in [("A", 3), ("B", 2)]
        ]
        result = compute_dissent(stances, tickers, _DEFAULT_CFG)
        assert result.dissent_score == pytest.approx(0.0, abs=1e-9)
        assert result.contested_week is False
        for sigma in result.per_ticker_sigma.values():
            assert sigma == pytest.approx(0.0, abs=1e-9)

    def test_empty_debate_set_returns_zero(self) -> None:
        stances = [{"persona": "p", "ticker": "X", "action": "add", "confidence": 3}]
        result = compute_dissent(stances, [], _DEFAULT_CFG)
        assert result.dissent_score == 0.0
        assert result.contested_week is False
        assert result.per_ticker_sigma == {}
        assert result.per_persona_divergence == {}


# ---------------------------------------------------------------------------
# C. Component 22 — outlier_selection cells
# ---------------------------------------------------------------------------


def _cfg_n(n: int) -> DissentConfig:
    return DissentConfig(
        contested_week_threshold=0.50,
        action_direction_map=_DEFAULT_CFG.action_direction_map,
        n_outliers=n,
        divergence_tiebreak="alpha_asc",
    )


def _dissent_from_divergences(divergences: dict[str, float]) -> DissentResult:
    """Build a minimal DissentResult from a divergence map (for selection tests)."""
    return DissentResult(
        dissent_score=0.5,
        contested_week=True,
        per_ticker_sigma={},
        per_persona_divergence=divergences,
    )


class TestOutlierSelection:
    """Component 22 selection cells — ≥4 fixtures + determinism + brute-force cross-check."""

    def test_clear_top_two(self) -> None:
        """Unambiguous top-2 by divergence."""
        divs = {
            "alpha": 0.9,
            "beta":  0.7,
            "gamma": 0.3,
            "delta": 0.2,
            "epsilon": 0.1,
            "zeta": 0.05,
            "eta": 0.04,
        }
        stances = [{"persona": p, "ticker": "X", "action": "hold", "confidence": 3} for p in divs]
        result = select_outliers(_dissent_from_divergences(divs), stances, _DEFAULT_CFG)
        assert result.selected == ["alpha", "beta"]
        assert set(result.stances_by_persona.keys()) == {"alpha", "beta"}

    def test_tie_at_second_position_alpha_asc_tiebreak(self) -> None:
        """When #2 and #3 divergence are equal, alpha_asc picks the alphabetically first."""
        divs = {
            "zeus":  0.9,   # clear #1
            "alpha": 0.5,   # tied for #2
            "omega": 0.5,   # tied for #3 — alpha < omega so alpha wins
            "gamma": 0.2,
            "delta": 0.1,
            "eta":   0.1,
            "iota":  0.05,
        }
        stances = [{"persona": p, "ticker": "X", "action": "hold", "confidence": 3} for p in divs]
        result = select_outliers(_dissent_from_divergences(divs), stances, _DEFAULT_CFG)
        assert result.selected[0] == "zeus"
        assert result.selected[1] == "alpha"   # alpha_asc picks "alpha" over "omega"

    def test_all_equal_tiebreak_picks_alphabetically_first_two(self) -> None:
        """All divergences equal → alpha_asc tie-break → first two alphabetically."""
        personas = ["zeta", "beta", "gamma", "alpha", "delta", "epsilon", "eta"]
        divs = {p: 0.4 for p in personas}
        stances = [{"persona": p, "ticker": "X", "action": "hold", "confidence": 3} for p in personas]
        result = select_outliers(_dissent_from_divergences(divs), stances, _DEFAULT_CFG)
        sorted_alpha = sorted(personas)
        assert result.selected == sorted_alpha[:2]

    def test_one_dominant_outlier_flat_rest(self) -> None:
        """One persona with much-higher divergence, rest equal."""
        divs = {
            "dominant": 0.95,
            "a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1, "e": 0.1, "f": 0.1,
        }
        stances = [{"persona": p, "ticker": "X", "action": "hold", "confidence": 3} for p in divs]
        result = select_outliers(_dissent_from_divergences(divs), stances, _DEFAULT_CFG)
        assert result.selected[0] == "dominant"
        assert result.selected[1] == "a"   # alpha_asc among the equal flat rest

    def test_determinism_same_input_same_output(self) -> None:
        """Running selection twice on identical input returns the same pair."""
        divs = {"gamma": 0.8, "alpha": 0.6, "delta": 0.4, "beta": 0.4, "e": 0.3, "f": 0.2, "g": 0.1}
        stances = [{"persona": p, "ticker": "X", "action": "hold", "confidence": 3} for p in divs]
        dr = _dissent_from_divergences(divs)
        r1 = select_outliers(dr, stances, _DEFAULT_CFG)
        r2 = select_outliers(dr, stances, _DEFAULT_CFG)
        assert r1.selected == r2.selected

    def test_brute_force_max_cross_check(self) -> None:
        """Brute-force: selected[0] has the highest divergence of all personas."""
        import random
        rng = random.Random(42)
        personas = [f"p{i}" for i in range(7)]
        divs = {p: rng.random() for p in personas}
        stances = [{"persona": p, "ticker": "X", "action": "hold", "confidence": 3} for p in personas]
        result = select_outliers(_dissent_from_divergences(divs), stances, _DEFAULT_CFG)
        max_persona = max(divs, key=lambda p: (divs[p], -ord(p[0])))  # rough brute-force
        assert divs[result.selected[0]] == max(divs.values())

    def test_stances_by_persona_contains_full_round1_stances(self) -> None:
        """stances_by_persona carries ALL tickers for the selected persona.

        Setup: 3 personas, 2 tickers.
        - "outlier" ADD@5 (+1.0) on both tickers.
        - "neg_outlier" EXIT@5 (-1.0) on both tickers.
        - "aligned" HOLD@1 (0.0) on both tickers.

        Panel mean per ticker = (1.0 + -1.0 + 0.0) / 3 = 0.
        Divergences: outlier=1.0, neg_outlier=1.0, aligned=0.0.
        Tie between outlier + neg_outlier → alpha_asc picks "neg_outlier" (n < o).
        But we only select n_outliers=1; the test checks stances_by_persona content,
        not which persona wins.  Use a single extreme persona with no competitor:
        2 personas only: outlier ADD@5, aligned HOLD@1.
        mean_ticker = (1.0 + 0.0)/2 = 0.5
        outlier divergence = |1.0 - 0.5| = 0.5
        aligned divergence = |0.0 - 0.5| = 0.5   ← tie
        alpha_asc tie-break: "aligned" < "outlier" → "aligned" selected.
        Use names so "outlier" sorts before "aligned" alphabetically:
        rename "aligned" → "zoomed" so "outlier" < "zoomed".
        """
        stances = [
            {"persona": "outlier", "ticker": "A", "action": "add",  "confidence": 5},
            {"persona": "outlier", "ticker": "B", "action": "add",  "confidence": 5},
            {"persona": "zoomed",  "ticker": "A", "action": "hold", "confidence": 1},
            {"persona": "zoomed",  "ticker": "B", "action": "hold", "confidence": 1},
        ]
        dr = compute_dissent(stances, ["A", "B"], _DEFAULT_CFG)
        result = select_outliers(dr, stances, DissentConfig(
            contested_week_threshold=0.0,
            action_direction_map=_DEFAULT_CFG.action_direction_map,
            n_outliers=1,
            divergence_tiebreak="alpha_asc",
        ))
        # Both divergences equal at 0.5; alpha_asc picks "outlier" < "zoomed"
        assert result.selected[0] == "outlier"
        assert len(result.stances_by_persona["outlier"]) == 2

    def test_raises_when_fewer_personas_than_n_outliers(self) -> None:
        divs = {"only_one": 0.5}
        stances = [{"persona": "only_one", "ticker": "X", "action": "hold", "confidence": 3}]
        with pytest.raises(RuntimeError, match="n_outliers=2"):
            select_outliers(_dissent_from_divergences(divs), stances, _DEFAULT_CFG)


# ---------------------------------------------------------------------------
# D. contested_week threshold comes from config, not a literal 0.50
# ---------------------------------------------------------------------------


class TestThresholdFromConfig:
    """The contested_week flag must respect the config threshold, not a hardcoded value."""

    def test_threshold_from_config_fires_at_boundary(self) -> None:
        """dissent_score exactly at threshold → contested=True; just below → False."""
        # Build a 2-persona, 1-ticker fixture where we can control exact σ.
        # Two personas: s_a and s_b.  σ = |s_a - s_b| / 2.
        # To hit threshold T exactly: we want σ = T.
        # Use s_a = T, s_b = -T so mean=0, σ = T.
        # T = 0.50 → ADD@5 (+1.0) and EXIT@5 (-1.0) but that gives σ=1.0 for 2 personas.
        # Actually for N=2: σ = sqrt(((s_a-mean)^2 + (s_b-mean)^2)/2) = |s_a - s_b|/2.
        # So to hit exactly 0.50: need |s_a - s_b| = 1.0 → e.g. ADD@5=+1 and HOLD@1=0 → σ=0.5.
        stances_at = [
            {"persona": "a", "ticker": "T", "action": "add",  "confidence": 5},  # s=1.0
            {"persona": "b", "ticker": "T", "action": "hold", "confidence": 1},  # s=0.0
        ]
        # σ = |1.0 - 0.0| / 2 = 0.5 exactly

        cfg_exact = DissentConfig(
            contested_week_threshold=0.50,
            action_direction_map=_DEFAULT_CFG.action_direction_map,
            n_outliers=2,
            divergence_tiebreak="alpha_asc",
        )
        result_at = compute_dissent(stances_at, ["T"], cfg_exact)
        assert result_at.dissent_score == pytest.approx(0.50, abs=1e-9)
        assert result_at.contested_week is True

        # Raise threshold slightly → same score is no longer contested
        cfg_high = DissentConfig(
            contested_week_threshold=0.51,
            action_direction_map=_DEFAULT_CFG.action_direction_map,
            n_outliers=2,
            divergence_tiebreak="alpha_asc",
        )
        result_below = compute_dissent(stances_at, ["T"], cfg_high)
        assert result_below.contested_week is False

    def test_load_dissent_config_reads_threshold_from_yaml(self, tmp_path: Path) -> None:
        """load_dissent_config honours a non-default threshold from a YAML file."""
        cfg_yaml = tmp_path / "thresholds.yaml"
        cfg_yaml.write_text(
            textwrap.dedent("""\
                contested_week_threshold: 0.75
                action_direction_map:
                  add: 1.0
                  hold: 0.0
                  reduce: -0.5
                  exit: -1.0
                n_outliers: 3
                divergence_tiebreak: alpha_desc
            """),
            encoding="utf-8",
        )
        cfg = load_dissent_config(cfg_yaml)
        assert cfg.contested_week_threshold == pytest.approx(0.75)
        assert cfg.n_outliers == 3
        assert cfg.divergence_tiebreak == "alpha_desc"

    def test_load_dissent_config_falls_back_to_defaults_on_missing_file(self) -> None:
        """Missing thresholds.yaml → no exception, built-in defaults used."""
        cfg = load_dissent_config(Path("/nonexistent/thresholds.yaml"))
        assert cfg.contested_week_threshold == pytest.approx(0.50)
        assert cfg.n_outliers == 2
        assert cfg.action_direction_map["add"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# E. DEF-004 regression — real 2026-W24 Round-1 stances
# ---------------------------------------------------------------------------
# Fixture provenance: live 2026-W24 run (quality-logs/TASK-M2-011.md §4 Real ledger summary).
# Source: agent_stances WHERE week_id='2026-W24' AND round=1, 280 rows (7 × 40 tickers).
# PII: none — equity research / market data only.  Sanitized snapshot committed to tests/fixtures/.
#
# DEF-004 finding (quality-logs/TASK-M2-011.md §10):
#   The M2 raw-weight σ (threshold 0.08) flagged ZERO contested names on 2026-W24, yet
#   the panel plainly disagreed: Growth was all-tech / 3% cash while other personas held
#   mixed/defensive positions.  Disagreement lived in direction-of-view, not per-ticker
#   weight variance.  The recalibrated metric (signed-score σ) must produce a materially
#   higher dissent_score, proving the under-read is fixed.
#
# Actual computed results (verified against live ledger):
#   NEW dissent_score (signed-score σ): 0.405222
#   OLD raw-weight σ:                  0.032234
#   contested_week: False  (0.405 < 0.50 threshold — the week is measurably divided
#                           but not above the contested threshold; honest result)
#   Top-2 outliers by divergence: growth (0.5618), cta-systematic-macro (0.3532)
#
# Note: TDD §22 Sample Selection references "Growth and Risk-Officer" as the motivating
# narrative (DEF-004 description), but the algorithmic result on the live data is
# growth + cta-systematic-macro.  risk-officer is #3 (0.3229).  The DEF-004 fix is
# validated by the 12.6× improvement in dissent_score (0.405 vs 0.032), not by matching
# a specific named pair.  The outlier pair is probabilistic and will be presented to the
# founder at the M3-006 gate for judgment.


class TestDEF004RealDataRegression:
    """Real 2026-W24 stance set — DEF-004 regression."""

    @pytest.fixture(scope="class")
    def w24_stances(self) -> list[dict]:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        return data

    @pytest.fixture(scope="class")
    def w24_debate_set(self, w24_stances: list[dict]) -> list[str]:
        return list(dict.fromkeys(s["ticker"] for s in w24_stances))

    @pytest.fixture(scope="class")
    def w24_result(self, w24_stances: list[dict], w24_debate_set: list[str]) -> DissentResult:
        return compute_dissent(w24_stances, w24_debate_set, _DEFAULT_CFG)

    def test_fixture_has_expected_shape(self, w24_stances: list[dict]) -> None:
        assert len(w24_stances) == 280  # 7 personas × 40 tickers
        personas = {s["persona"] for s in w24_stances}
        assert len(personas) == 7
        tickers = {s["ticker"] for s in w24_stances}
        assert len(tickers) == 40

    def test_new_dissent_score_materially_higher_than_old_weight_sigma(
        self,
        w24_stances: list[dict],
        w24_debate_set: list[str],
        w24_result: DissentResult,
    ) -> None:
        """NEW metric is materially higher than old raw-weight σ (DEF-004 fix proof)."""
        # Recompute old raw-weight σ inline (old helper deleted per Gate 8).
        old_sigma_per_ticker: dict[str, float] = {}
        for ticker in w24_debate_set:
            weights = [s["target_weight"] for s in w24_stances if s["ticker"] == ticker]
            if len(weights) < 2:
                old_sigma_per_ticker[ticker] = 0.0
                continue
            mean_w = sum(weights) / len(weights)
            var_w = sum((w - mean_w) ** 2 for w in weights) / len(weights)
            old_sigma_per_ticker[ticker] = math.sqrt(var_w)
        old_dissent = sum(old_sigma_per_ticker.values()) / len(w24_debate_set)

        # New metric is at least 5× higher — DEF-004 shows a ~12.6× improvement.
        assert w24_result.dissent_score > old_dissent * 5, (
            f"Expected new_score ({w24_result.dissent_score:.6f}) > 5 × "
            f"old_score ({old_dissent:.6f}); ratio was "
            f"{w24_result.dissent_score / old_dissent:.1f}×"
        )

    def test_new_dissent_score_exact_value(self, w24_result: DissentResult) -> None:
        """Regression lock: dissent_score matches the value computed from the live ledger."""
        assert w24_result.dissent_score == pytest.approx(0.405222, abs=1e-4)

    def test_old_weight_sigma_near_zero(
        self, w24_stances: list[dict], w24_debate_set: list[str]
    ) -> None:
        """Old metric was near-zero on this plainly-divided week (the DEF-004 bug)."""
        old_per_ticker: dict[str, float] = {}
        for ticker in w24_debate_set:
            weights = [s["target_weight"] for s in w24_stances if s["ticker"] == ticker]
            mean_w = sum(weights) / len(weights)
            var_w = sum((w - mean_w) ** 2 for w in weights) / len(weights)
            old_per_ticker[ticker] = math.sqrt(var_w)
        old_dissent = sum(old_per_ticker.values()) / len(w24_debate_set)
        assert old_dissent == pytest.approx(0.032234, abs=1e-4)
        # Threshold was 0.08 — old metric would flag 0 / 40 tickers as contested.
        flagged = sum(1 for v in old_per_ticker.values() if v >= 0.08)
        assert flagged == 0

    def test_contested_week_false(self, w24_result: DissentResult) -> None:
        """2026-W24 dissent_score (0.405) is below the 0.50 threshold → not contested.

        This is the honest algorithmic result: the week shows measurable divergence
        but not enough to cross the contested threshold.  The DEF-004 narrative
        (Growth vs Risk-Officer) motivated the metric recalibration; the threshold
        judgment is the founder's at the M3-006 gate.
        """
        assert w24_result.contested_week is False

    def test_growth_is_top_outlier(self, w24_result: DissentResult) -> None:
        """Growth persona has the highest divergence score on 2026-W24 (algorithmic result)."""
        top = max(w24_result.per_persona_divergence, key=lambda p: w24_result.per_persona_divergence[p])
        assert top == "growth"

    def test_top_two_outliers(
        self, w24_stances: list[dict], w24_result: DissentResult
    ) -> None:
        """Top-2 outliers are growth + cta-systematic-macro (live data algorithmic result).

        Note: TDD §22 references 'Growth and Risk-Officer' as the motivating DEF-004
        narrative; the live algorithmic result is growth + cta-systematic-macro.
        risk-officer is #3.  The actual pair is presented to the founder at M3-006
        for probabilistic judgment (AC#2).
        """
        selection = select_outliers(w24_result, w24_stances, _DEFAULT_CFG)
        assert selection.selected == ["growth", "cta-systematic-macro"]

    def test_all_seven_personas_have_divergence_scores(self, w24_result: DissentResult) -> None:
        assert len(w24_result.per_persona_divergence) == 7

    def test_all_forty_tickers_have_sigma(self, w24_result: DissentResult) -> None:
        assert len(w24_result.per_ticker_sigma) == 40

    def test_per_persona_divergence_ordering(self, w24_result: DissentResult) -> None:
        """Divergence ranking is growth > cta-systematic-macro > risk-officer."""
        div = w24_result.per_persona_divergence
        assert div["growth"] > div["cta-systematic-macro"]
        assert div["cta-systematic-macro"] > div["risk-officer"]

    def test_selection_stances_by_persona_has_40_stances_each(
        self, w24_stances: list[dict], w24_result: DissentResult
    ) -> None:
        selection = select_outliers(w24_result, w24_stances, _DEFAULT_CFG)
        for persona in selection.selected:
            assert len(selection.stances_by_persona[persona]) == 40

    def test_out_of_domain_action_raises(self, w24_debate_set: list[str]) -> None:
        """A stance with an unrecognised action raises ValueError (fail-loud spec).

        Two personas needed so the <2-stances guard doesn't skip the ticker
        before _signed_score is reached.
        """
        bad_stances = [
            {"persona": "p1", "ticker": "X", "action": "short", "confidence": 3},
            {"persona": "p2", "ticker": "X", "action": "hold",  "confidence": 3},
        ]
        with pytest.raises(ValueError, match="Out-of-domain action"):
            compute_dissent(bad_stances, ["X"], _DEFAULT_CFG)

    def test_out_of_domain_confidence_raises(self) -> None:
        """Confidence outside 1-5 raises ValueError (fail-loud spec).

        Two personas needed for the same reason as test_out_of_domain_action_raises.
        """
        bad_stances = [
            {"persona": "p1", "ticker": "X", "action": "add", "confidence": 6},
            {"persona": "p2", "ticker": "X", "action": "hold", "confidence": 3},
        ]
        with pytest.raises(ValueError, match="Out-of-domain confidence"):
            compute_dissent(bad_stances, ["X"], _DEFAULT_CFG)
