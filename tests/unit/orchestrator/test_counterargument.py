"""Unit tests — Component 23: counterargument_assembly.

Covers:
  - 4 synthetic scenario fixtures (AC1/AC2/AC3 deterministic)
  - 1 zero-dispatch assertion (AC3 — by construction + import check)
  - 1 real-data fixture: 2026-W24 outliers (growth + cta-systematic-macro)

Provenance of real-data fixture
--------------------------------
stances: tests/fixtures/stances_2026_w24_round1.json
  Source: agent_stances WHERE week_id='2026-W24' AND round=1, 280 rows.
  Live 2026-W24 run, state/ledger.db. PII: none (equity research only).
rationales: tests/fixtures/rationales_2026_w24_round1.json
  Source: agent_stances.rationale WHERE week_id='2026-W24' AND round=1, 280 rows.
  Extracted 2026-06-08. Same provenance as stances fixture.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from round_table_portfolio.orchestrator.counterargument import (
    CounterargumentBlock,
    CounterargumentConfig,
    assemble_counterargument,
    assemble_counterarguments,
    load_counterargument_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures"

_ACTION_DIRECTION: dict[str, float] = {
    "add": 1.0,
    "hold": 0.0,
    "reduce": -0.5,
    "exit": -1.0,
}

_DEFAULT_CFG = CounterargumentConfig(
    counterargument_max_rationales=3,
    counterargument_agree_tolerance=0.05,
    action_direction_map=_ACTION_DIRECTION,
)


def _stance(persona: str, ticker: str, action: str, confidence: int) -> dict[str, Any]:
    return {"persona": persona, "ticker": ticker, "action": action, "confidence": confidence}


def _rats(*entries: tuple[str, str, str]) -> dict[str, dict[str, str]]:
    """Build rationales dict from (persona, ticker, text) tuples."""
    result: dict[str, dict[str, str]] = {}
    for persona, ticker, text in entries:
        result.setdefault(persona, {})[ticker] = text
    return result


def _all_sources_are_substrings(block: CounterargumentBlock) -> bool:
    """AC1 check: every segment in block.block is a substring of a source rationale."""
    for _persona, _ticker, text in block.source_rationales:
        if text not in block.block:
            return False
    return True


def _no_novel_sentences(block: CounterargumentBlock, all_rationale_texts: list[str]) -> bool:
    """Stricter AC1: every quoted segment in the block exists verbatim in the
    corpus of source rationale texts (not just the capped selection)."""
    for _persona, _ticker, text in block.source_rationales:
        if text not in all_rationale_texts:
            return False
    return True


# ---------------------------------------------------------------------------
# AC3 — Zero-dispatch assertion (by construction)
# ---------------------------------------------------------------------------


class TestZeroDispatch:
    """Component 23 makes no Agent/Task dispatch — pure Python, no external calls."""

    def test_counterargument_module_imports_no_dispatch_machinery(self) -> None:
        """The module must not import anthropic, Agent, Task, or any dispatch
        surface.  If it did, a dispatch COULD occur — by construction it cannot.
        """
        # Reload to get a clean module object.
        mod = importlib.import_module(
            "round_table_portfolio.orchestrator.counterargument"
        )
        source = Path(mod.__file__).read_text(encoding="utf-8")  # type: ignore[arg-type]

        # Check for actual import-level dispatch machinery, not prose mentions.
        forbidden = ["import anthropic", "from anthropic", "Agent(", "Task("]
        for term in forbidden:
            assert term not in source, (
                f"counterargument.py contains dispatch import/call {term!r} — "
                "Component 23 must be zero-dispatch by construction."
            )

    def test_assemble_counterargument_is_pure_python(self) -> None:
        """assemble_counterargument returns a CounterargumentBlock with no I/O side
        effects (no network, no subprocess, no file write beyond what's passed in).
        This test confirms it executes to completion using only in-memory inputs."""
        stances = [
            _stance("growth", "MSFT", "add", 5),
            _stance("value", "MSFT", "hold", 3),
            _stance("risk-officer", "MSFT", "exit", 4),
        ]
        rats = _rats(
            ("value", "MSFT", "Expensive by book value; no margin of safety."),
            ("risk-officer", "MSFT", "Concentration risk at 15% weight; tail risk."),
        )
        result = assemble_counterargument("growth", stances, rats, _DEFAULT_CFG)
        assert isinstance(result, CounterargumentBlock)
        # No exception = no dispatch attempted; pure Python path confirmed.


# ---------------------------------------------------------------------------
# Synthetic fixture 1 — outlier exits a panel favourite
# ---------------------------------------------------------------------------


class TestOutlierExitsPanelFavourite:
    """growth (outlier) EXITs NVDA; the panel strongly ADDs."""

    STANCES = [
        _stance("growth", "NVDA", "exit", 5),       # outlier: s = -1.0
        _stance("value", "NVDA", "add", 4),          # panel: s = +0.8
        _stance("technical", "NVDA", "add", 5),      # panel: s = +1.0
        _stance("risk-officer", "NVDA", "add", 3),   # panel: s = +0.6
        _stance("quant-systematic", "NVDA", "add", 4),# panel: s = +0.8
    ]
    RATIONALES = _rats(
        ("value", "NVDA", "Deep value: NVDA trades at 28x earnings with dominant moat."),
        ("technical", "NVDA", "Perfect MA stack, ADX 62, RSI 68 — strongest uptrend in set."),
        ("risk-officer", "NVDA", "Risk-adjusted add: concentration capped at 12% for tail exposure."),
        ("quant-systematic", "NVDA", "Factor score: momentum +2.1σ, quality +1.8σ."),
    )

    def test_targets_nvda_as_most_divergent(self) -> None:
        result = assemble_counterargument("growth", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert "NVDA" in result.debated_tickers

    def test_block_composed_only_from_existing_rationale_text(self) -> None:
        result = assemble_counterargument("growth", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert _all_sources_are_substrings(result)

    def test_attribution_is_correct(self) -> None:
        result = assemble_counterargument("growth", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        # Every source persona in the block should be a non-outlier panel member.
        for persona, ticker, _ in result.source_rationales:
            assert persona != "growth", "Outlier's own rationale must not appear in its counterargument."
            assert ticker == "NVDA"

    def test_block_does_not_challenge_agreed_tickers(self) -> None:
        # Add a ticker where growth and the single panel member agree exactly.
        # growth@add@4 → s=+0.8; value@add@4 → s=+0.8.
        # panel_mean (excluding growth) = +0.8; divergence = 0.0 ≤ _AGREE_TOL.
        stances = self.STANCES + [
            _stance("growth", "MSFT", "add", 4),
            _stance("value", "MSFT", "add", 4),
        ]
        rats = dict(self.RATIONALES)
        rats.setdefault("value", {})["MSFT"] = "MSFT: strong moat."
        result = assemble_counterargument("growth", stances, rats, _DEFAULT_CFG)
        assert "MSFT" not in result.debated_tickers

    def test_length_cap_respected(self) -> None:
        cfg = CounterargumentConfig(counterargument_max_rationales=2, counterargument_agree_tolerance=0.05, action_direction_map=_ACTION_DIRECTION)
        result = assemble_counterargument("growth", self.STANCES, self.RATIONALES, cfg)
        assert len(result.source_rationales) <= 2


# ---------------------------------------------------------------------------
# Synthetic fixture 2 — outlier adds a name the panel avoids
# ---------------------------------------------------------------------------


class TestOutlierAddsPanelAvoids:
    """cta-systematic-macro (outlier) ADDs T (telecom); panel exits/reduces."""

    STANCES = [
        _stance("cta-systematic-macro", "T", "add", 5),   # outlier: s = +1.0
        _stance("growth", "T", "exit", 5),                 # panel: s = -1.0
        _stance("value", "T", "exit", 4),                  # panel: s = -0.8
        _stance("discretionary-macro", "T", "reduce", 4),  # panel: s = -0.4
        _stance("risk-officer", "T", "exit", 3),           # panel: s = -0.6
    ]
    RATIONALES = _rats(
        ("growth", "T", "No-growth, debt-heavy telecom; yield story tied to a saturated market."),
        ("value", "T", "Yield trap: dividend yield elevated because growth is zero; book destroyed."),
        ("discretionary-macro", "T", "Stagflationary environment kills leveraged balance sheets."),
        ("risk-officer", "T", "Debt load 2.3x book; refinancing risk at elevated rates."),
    )

    def test_targets_t_as_most_divergent(self) -> None:
        result = assemble_counterargument("cta-systematic-macro", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert "T" in result.debated_tickers

    def test_opposing_rationales_are_from_exit_reduce_personas(self) -> None:
        result = assemble_counterargument("cta-systematic-macro", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        for persona, _ticker, _text in result.source_rationales:
            assert persona in {"growth", "value", "discretionary-macro", "risk-officer"}

    def test_block_composed_only_from_existing_rationale_text(self) -> None:
        result = assemble_counterargument("cta-systematic-macro", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert _all_sources_are_substrings(result)

    def test_outlier_slug_not_in_source_rationales(self) -> None:
        result = assemble_counterargument("cta-systematic-macro", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        for persona, _, _ in result.source_rationales:
            assert persona != "cta-systematic-macro"


# ---------------------------------------------------------------------------
# Synthetic fixture 3 — outlier diverges on cash level (all-hold context)
# ---------------------------------------------------------------------------


class TestOutlierDivergesOnCashLevel:
    """risk-officer (outlier) HOLDs everything (zero score, low conviction);
    panel strongly ADDs a name.  The outlier's HOLD is implicitly a low-conviction
    non-participation in a panel consensus ADD — divergence comes from the signed
    score gap (0.0 vs panel mean ~+0.7)."""

    STANCES = [
        _stance("risk-officer", "CVX", "hold", 2),      # outlier: s = 0.0
        _stance("growth", "CVX", "add", 4),              # panel: s = +0.8
        _stance("value", "CVX", "add", 5),               # panel: s = +1.0
        _stance("technical", "CVX", "add", 4),           # panel: s = +0.8
        _stance("quant-systematic", "CVX", "add", 3),    # panel: s = +0.6
    ]
    RATIONALES = _rats(
        ("growth", "CVX", "Energy supercycle intact; FCF yield 8%, buyback 4%."),
        ("value", "CVX", "Book value 1.2x; FCF 12% yield at $75 oil."),
        ("technical", "CVX", "MA stack bullish, ADX 38, RSI 61."),
        ("quant-systematic", "CVX", "Factor screen: value +1.9σ, quality +1.4σ."),
    )

    def test_targets_cvx(self) -> None:
        result = assemble_counterargument("risk-officer", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert "CVX" in result.debated_tickers

    def test_block_non_empty(self) -> None:
        result = assemble_counterargument("risk-officer", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert result.block.strip() != ""

    def test_block_composed_only_from_existing_rationale_text(self) -> None:
        result = assemble_counterargument("risk-officer", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert _all_sources_are_substrings(result)

    def test_correct_attribution(self) -> None:
        result = assemble_counterargument("risk-officer", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        source_personas = {p for p, _, _ in result.source_rationales}
        assert "risk-officer" not in source_personas


# ---------------------------------------------------------------------------
# Synthetic fixture 4 — outlier aligned on most names, diverges on one
# ---------------------------------------------------------------------------


class TestOutlierDivergesOnOneOnly:
    """value (outlier) agrees with panel on 4 tickers but diverges sharply on MO."""

    STANCES = [
        # Agreed tickers (both value and panel ADD these).
        _stance("value", "KO", "add", 3),
        _stance("growth", "KO", "add", 3),
        _stance("value", "MRK", "add", 4),
        _stance("growth", "MRK", "add", 4),
        _stance("value", "PM", "add", 3),
        _stance("growth", "PM", "add", 3),
        # Divergent ticker: value ADDS MO; panel exits.
        _stance("value", "MO", "add", 5),       # outlier: s = +1.0
        _stance("growth", "MO", "exit", 5),     # panel: s = -1.0
        _stance("technical", "MO", "exit", 4),  # panel: s = -0.8
    ]
    RATIONALES = _rats(
        ("growth", "MO", "Declining-volume tobacco — high yield masking structural decline."),
        ("technical", "MO", "Confirmed downtrend: price below SMA200, MACD deeply negative."),
    )

    def test_only_mo_is_debated(self) -> None:
        result = assemble_counterargument("value", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert "MO" in result.debated_tickers
        # KO, MRK, PM should NOT be debated (outlier agrees with panel there).
        for agreed in ("KO", "MRK", "PM"):
            assert agreed not in result.debated_tickers

    def test_block_composed_only_from_existing_rationale_text(self) -> None:
        result = assemble_counterargument("value", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        assert _all_sources_are_substrings(result)

    def test_attribution_targets_mo(self) -> None:
        result = assemble_counterargument("value", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        for _persona, ticker, _ in result.source_rationales:
            assert ticker == "MO"

    def test_no_self_attribution(self) -> None:
        result = assemble_counterargument("value", self.STANCES, self.RATIONALES, _DEFAULT_CFG)
        for persona, _, _ in result.source_rationales:
            assert persona != "value"


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


class TestAssembleCounterarguments:
    """assemble_counterarguments produces one block per outlier slug."""

    STANCES = [
        _stance("growth", "NVDA", "add", 5),
        _stance("value", "NVDA", "exit", 4),
        _stance("technical", "NVDA", "exit", 5),
        _stance("cta-systematic-macro", "MO", "add", 4),
        _stance("growth", "MO", "hold", 2),
        _stance("value", "MO", "exit", 4),
    ]
    RATIONALES = _rats(
        ("value", "NVDA", "Expensive by any traditional metric; no margin of safety."),
        ("technical", "NVDA", "Breakdown confirmed: ADX falling, RSI below 40."),
        ("growth", "MO", "Tobacco is structural decline; no growth runway here."),
        ("value", "MO", "High yield but zero growth — yield trap."),
    )

    def test_returns_one_block_per_slug(self) -> None:
        results = assemble_counterarguments(
            ["growth", "cta-systematic-macro"], self.STANCES, self.RATIONALES, _DEFAULT_CFG
        )
        assert set(results.keys()) == {"growth", "cta-systematic-macro"}

    def test_each_block_is_counterargument_block(self) -> None:
        results = assemble_counterarguments(
            ["growth", "cta-systematic-macro"], self.STANCES, self.RATIONALES, _DEFAULT_CFG
        )
        for block in results.values():
            assert isinstance(block, CounterargumentBlock)

    def test_deterministic_same_inputs_same_output(self) -> None:
        r1 = assemble_counterarguments(
            ["growth", "cta-systematic-macro"], self.STANCES, self.RATIONALES, _DEFAULT_CFG
        )
        r2 = assemble_counterarguments(
            ["growth", "cta-systematic-macro"], self.STANCES, self.RATIONALES, _DEFAULT_CFG
        )
        for slug in ("growth", "cta-systematic-macro"):
            assert r1[slug].block == r2[slug].block
            assert r1[slug].debated_tickers == r2[slug].debated_tickers


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_stances_returns_empty_block(self) -> None:
        result = assemble_counterargument("growth", [], {}, _DEFAULT_CFG)
        assert result.block == ""
        assert result.source_rationales == []

    def test_no_opposing_rationales_returns_empty_block(self) -> None:
        """When the opposing stances exist but their rationale text is missing,
        the block is empty (skipped quietly — Gate 5: not a failure, just no text)."""
        stances = [
            _stance("growth", "NVDA", "exit", 5),
            _stance("value", "NVDA", "add", 4),
        ]
        result = assemble_counterargument("growth", stances, {}, _DEFAULT_CFG)
        # No rationale text for value on NVDA → nothing to include.
        assert result.block == ""

    def test_max_rationales_zero_produces_empty_block(self) -> None:
        cfg = CounterargumentConfig(counterargument_max_rationales=0, counterargument_agree_tolerance=0.05, action_direction_map=_ACTION_DIRECTION)
        stances = [
            _stance("growth", "NVDA", "exit", 5),
            _stance("value", "NVDA", "add", 4),
        ]
        rats = _rats(("value", "NVDA", "Strong uptrend."))
        result = assemble_counterargument("growth", stances, rats, cfg)
        assert result.block == ""
        assert result.source_rationales == []

    def test_no_duplicate_source_rationales(self) -> None:
        """Each (persona, ticker) pair appears at most once in source_rationales."""
        stances = [
            _stance("growth", "NVDA", "exit", 5),
            _stance("value", "NVDA", "add", 5),
            _stance("technical", "NVDA", "add", 4),
            _stance("risk-officer", "NVDA", "add", 3),
        ]
        rats = _rats(
            ("value", "NVDA", "Value rationale."),
            ("technical", "NVDA", "Technical rationale."),
            ("risk-officer", "NVDA", "Risk rationale."),
        )
        result = assemble_counterargument("growth", stances, rats, _DEFAULT_CFG)
        seen: set[tuple[str, str]] = set()
        for persona, ticker, _ in result.source_rationales:
            key = (persona, ticker)
            assert key not in seen, f"Duplicate source rationale for ({persona}, {ticker})"
            seen.add(key)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestLoadCounterargumentConfig:
    def test_missing_file_falls_back_to_defaults(self, tmp_path: Path) -> None:
        cfg = load_counterargument_config(config_path=tmp_path / "missing.yaml")
        assert cfg.counterargument_max_rationales == 3
        assert cfg.counterargument_agree_tolerance == 0.05
        assert "add" in cfg.action_direction_map

    def test_reads_counterargument_max_rationales_from_yaml(self, tmp_path: Path) -> None:
        yaml_text = "counterargument_max_rationales: 5\n"
        p = tmp_path / "thresholds.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        cfg = load_counterargument_config(config_path=p)
        assert cfg.counterargument_max_rationales == 5

    def test_reads_counterargument_agree_tolerance_from_yaml(self, tmp_path: Path) -> None:
        yaml_text = "counterargument_agree_tolerance: 0.30\n"
        p = tmp_path / "thresholds.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        cfg = load_counterargument_config(config_path=p)
        assert cfg.counterargument_agree_tolerance == 0.30

    def test_agree_tolerance_from_config_changes_challenged_set(self, tmp_path: Path) -> None:
        """A wider tolerance suppresses tickers where divergence is small.

        Setup: outlier score = 0.2, panel mean = 0.0 → divergence = 0.2.
        With tight tolerance (0.05): ticker IS challenged.
        With wide tolerance (0.25): ticker IS NOT challenged (0.2 ≤ 0.25).
        """
        stances = [
            _stance("growth", "MSFT", "add", 1),   # s = +0.2 (add × 1/5)
            _stance("value", "MSFT", "hold", 3),   # s = 0.0
            _stance("technical", "MSFT", "hold", 4),  # s = 0.0
        ]
        rats = _rats(
            ("value", "MSFT", "Fair value; no catalyst."),
            ("technical", "MSFT", "Flat MA stack; neutral signal."),
        )

        tight_cfg = CounterargumentConfig(
            counterargument_max_rationales=3,
            counterargument_agree_tolerance=0.05,
            action_direction_map=_ACTION_DIRECTION,
        )
        wide_cfg = CounterargumentConfig(
            counterargument_max_rationales=3,
            counterargument_agree_tolerance=0.25,
            action_direction_map=_ACTION_DIRECTION,
        )

        tight_result = assemble_counterargument("growth", stances, rats, tight_cfg)
        wide_result = assemble_counterargument("growth", stances, rats, wide_cfg)

        assert "MSFT" in tight_result.debated_tickers, (
            "With tight tolerance (0.05), divergence=0.2 should be debated"
        )
        assert "MSFT" not in wide_result.debated_tickers, (
            "With wide tolerance (0.25), divergence=0.2 should be treated as agreed"
        )

    def test_reads_action_direction_map_from_yaml(self, tmp_path: Path) -> None:
        yaml_text = (
            "action_direction_map:\n"
            "  add: 1.0\n"
            "  hold: 0.0\n"
            "  reduce: -0.5\n"
            "  exit: -1.0\n"
        )
        p = tmp_path / "thresholds.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        cfg = load_counterargument_config(config_path=p)
        assert cfg.action_direction_map["add"] == 1.0
        assert cfg.action_direction_map["exit"] == -1.0


# ---------------------------------------------------------------------------
# Real-data fixture — 2026-W24 (Gate 4 real-data requirement)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FIXTURE_DIR / "stances_2026_w24_round1.json").exists()
    or not (FIXTURE_DIR / "rationales_2026_w24_round1.json").exists(),
    reason="2026-W24 fixtures not present",
)
class TestRealData2026W24:
    """Gate 4 real-data fixture.

    Provenance:
      stances: agent_stances WHERE week_id='2026-W24' AND round=1, 280 rows.
        Source: live 2026-W24 run, state/ledger.db. Sanitized snapshot.
      rationales: agent_stances.rationale, same 280 rows, extracted 2026-06-08.
    Outliers (from M3-001 algorithmic result): growth (#1, divergence=0.562)
      and cta-systematic-macro (#2, divergence=0.353).
    """

    @pytest.fixture(scope="class")
    def stances(self) -> list[dict]:
        return json.loads((FIXTURE_DIR / "stances_2026_w24_round1.json").read_text())

    @pytest.fixture(scope="class")
    def rationales(self) -> dict[str, dict[str, str]]:
        return json.loads((FIXTURE_DIR / "rationales_2026_w24_round1.json").read_text())

    @pytest.fixture(scope="class")
    def growth_block(self, stances, rationales) -> CounterargumentBlock:
        return assemble_counterargument("growth", stances, rationales, _DEFAULT_CFG)

    @pytest.fixture(scope="class")
    def cta_block(self, stances, rationales) -> CounterargumentBlock:
        return assemble_counterargument(
            "cta-systematic-macro", stances, rationales, _DEFAULT_CFG
        )

    # --- growth outlier ---

    def test_growth_block_non_empty(self, growth_block: CounterargumentBlock) -> None:
        assert growth_block.block.strip() != ""

    def test_growth_composed_only_from_existing_rationale_text(
        self, growth_block: CounterargumentBlock, rationales: dict
    ) -> None:
        all_texts = [
            text
            for persona_rats in rationales.values()
            for text in persona_rats.values()
        ]
        for _persona, _ticker, text in growth_block.source_rationales:
            assert text in all_texts, (
                f"Text not found in any Round-1 rationale: {text[:60]!r}"
            )

    def test_growth_correct_attribution(
        self, growth_block: CounterargumentBlock
    ) -> None:
        for persona, ticker, text in growth_block.source_rationales:
            assert persona != "growth", "Outlier's own rationale must not appear."
            # Attribution in block uses "[persona on ticker]:" format.
            assert f"[{persona} on {ticker}]:" in growth_block.block

    def test_growth_length_cap_respected(
        self, growth_block: CounterargumentBlock
    ) -> None:
        assert len(growth_block.source_rationales) <= _DEFAULT_CFG.counterargument_max_rationales

    def test_growth_targets_divergent_tickers_not_agreed_ones(
        self, growth_block: CounterargumentBlock, stances: list[dict], rationales: dict
    ) -> None:
        """Debated tickers are those where growth diverges — not where it agrees."""
        from round_table_portfolio.orchestrator.counterargument import (
            _ticker_scores,
            _panel_mean,
        )
        tol = _DEFAULT_CFG.counterargument_agree_tolerance
        for ticker in growth_block.debated_tickers:
            scores = _ticker_scores(stances, ticker, _ACTION_DIRECTION)
            if "growth" not in scores:
                continue
            panel_mean = _panel_mean(scores, "growth")
            divergence = abs(scores["growth"] - panel_mean)
            assert divergence > tol, (
                f"growth ticker {ticker} is in debated_tickers but divergence={divergence:.3f} ≤ {tol}"
            )

    def test_growth_block_is_deterministic(
        self, stances: list[dict], rationales: dict
    ) -> None:
        b1 = assemble_counterargument("growth", stances, rationales, _DEFAULT_CFG)
        b2 = assemble_counterargument("growth", stances, rationales, _DEFAULT_CFG)
        assert b1.block == b2.block

    # --- cta-systematic-macro outlier ---

    def test_cta_block_non_empty(self, cta_block: CounterargumentBlock) -> None:
        assert cta_block.block.strip() != ""

    def test_cta_composed_only_from_existing_rationale_text(
        self, cta_block: CounterargumentBlock, rationales: dict
    ) -> None:
        all_texts = [
            text
            for persona_rats in rationales.values()
            for text in persona_rats.values()
        ]
        for _persona, _ticker, text in cta_block.source_rationales:
            assert text in all_texts, (
                f"Text not found in any Round-1 rationale: {text[:60]!r}"
            )

    def test_cta_correct_attribution(self, cta_block: CounterargumentBlock) -> None:
        for persona, ticker, text in cta_block.source_rationales:
            assert persona != "cta-systematic-macro"
            assert f"[{persona} on {ticker}]:" in cta_block.block

    def test_cta_length_cap_respected(self, cta_block: CounterargumentBlock) -> None:
        assert len(cta_block.source_rationales) <= _DEFAULT_CFG.counterargument_max_rationales

    def test_cta_targets_divergent_tickers_not_agreed_ones(
        self, cta_block: CounterargumentBlock, stances: list[dict], rationales: dict
    ) -> None:
        from round_table_portfolio.orchestrator.counterargument import (
            _ticker_scores,
            _panel_mean,
        )
        tol = _DEFAULT_CFG.counterargument_agree_tolerance
        for ticker in cta_block.debated_tickers:
            scores = _ticker_scores(stances, ticker, _ACTION_DIRECTION)
            if "cta-systematic-macro" not in scores:
                continue
            panel_mean = _panel_mean(scores, "cta-systematic-macro")
            divergence = abs(scores["cta-systematic-macro"] - panel_mean)
            assert divergence > tol, (
                f"cta ticker {ticker} is in debated_tickers but divergence={divergence:.3f} ≤ {tol}"
            )

    def test_cta_block_is_deterministic(
        self, stances: list[dict], rationales: dict
    ) -> None:
        b1 = assemble_counterargument("cta-systematic-macro", stances, rationales, _DEFAULT_CFG)
        b2 = assemble_counterargument("cta-systematic-macro", stances, rationales, _DEFAULT_CFG)
        assert b1.block == b2.block

    def test_batch_assembly_produces_both_outlier_blocks(
        self, stances: list[dict], rationales: dict
    ) -> None:
        results = assemble_counterarguments(
            ["growth", "cta-systematic-macro"], stances, rationales, _DEFAULT_CFG
        )
        assert "growth" in results
        assert "cta-systematic-macro" in results
        assert results["growth"].block.strip() != ""
        assert results["cta-systematic-macro"].block.strip() != ""
