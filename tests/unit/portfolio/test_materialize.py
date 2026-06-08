"""Unit tests for Component 15 — materialize_portfolios (TASK-M2-005).

Coverage matrix
---------------
AC1  8-payload production          — 8 payloads returned (7 personas + consensus)
AC2  Explicit CASH row per payload — exactly 1 CASH row per payload, weight=1-Σpositions
AC2  Exactly-100% edge case        — CASH row weight=0.0 still present
AC3  >100% rejection (3 fixtures)  — positions summing to 1.01/1.05/1.50 raise Major
AC3  Never clipped/rescaled        — rejection, not silent fix
AC4  Action derivation             — first-week all-add; later-week reduce/hold/exit
AC4  Phantom-ticker rejection       — ticker outside debate_set + prior raises
AC5  Single-source helper          — materialize imports check_fully_invested from
                                     portfolio/invariants (same object as Layer-2)
AC6  Full test suite passes        — covered by pytest run (see quality log)

End-to-end DB check:
     SELECT SUM(weight) FROM holdings WHERE portfolio_id=? = 1.0 for all 8.

Fixture provenance:
    Synthetic — hand-crafted to drive specific arithmetic paths.
    Real-data validation is TASK-M2-011 (live weekly run).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from round_table_portfolio.portfolio.materialize import (
    HoldingPayload,
    PortfolioPayload,
    materialize_portfolios,
)
from round_table_portfolio.portfolio.invariants import check_fully_invested
import round_table_portfolio.portfolio.materialize as _materialize_module

from tests.unit.fixtures.portfolios.fixtures import (
    EXACT_100_FIXTURES,
    OVER_INVESTED_FIXTURES,
    PRIOR_FOR_SINGLE_PERSONA,
    PRIOR_WITH_EXITED_TICKER,
    VALID_SINGLE_PERSONA,
    EXPECTED_ACTIONS_WITH_PRIOR,
    EXACT_100_CASH_ZERO,
    EXACT_100_EXPLICIT_ZERO_CASH,
    EXACT_100_NO_CASH_KEY,
    PARTIAL_NO_CASH_KEY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WEEK_ID = "2026-W23"
_DATE = "2026-06-07"
_MAX_PW = 0.20
_CFG = {"max_position_weight": _MAX_PW}
_CFG_WITH_DEBATE = {
    "max_position_weight": _MAX_PW,
    "debate_set": ["AAPL", "MSFT", "NVDA", "QCOM", "GOOGL", "AMD", "INTC"],
}

_PERSONA_SLUGS_7 = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_valid_counterfactual(
    tickers: list[str] | None = None,
    pos_weight: float = 0.10,
    cash: float | None = None,
) -> dict[str, float]:
    """Build a valid counterfactual dict summing to 1.0."""
    tickers = tickers or ["AAPL", "MSFT", "NVDA"]
    positions = {t: pos_weight for t in tickers}
    actual_cash = cash if cash is not None else round(1.0 - sum(positions.values()), 10)
    positions["CASH"] = actual_cash
    return positions


def _make_all_7_counterfactuals(
    tickers: list[str] | None = None,
    pos_weight: float = 0.10,
) -> dict[str, dict[str, float]]:
    """Build a full set of 7 valid per-persona counterfactuals."""
    return {
        slug: _make_valid_counterfactual(tickers, pos_weight)
        for slug in _PERSONA_SLUGS_7
    }


def _make_consensus_weights(
    tickers: list[str] | None = None,
    pos_weight: float = 0.10,
) -> dict[str, float]:
    """Build consensus weights (no CASH key — Component 16 contract)."""
    tickers = tickers or ["AAPL", "MSFT", "NVDA"]
    return {t: pos_weight for t in tickers}


def _holdings_by_ticker(payload: PortfolioPayload) -> dict[str, HoldingPayload]:
    return {h.ticker: h for h in payload.holdings}


# ---------------------------------------------------------------------------
# AC5 — Single-source-helper test (same check_fully_invested object as Layer-2)
# ---------------------------------------------------------------------------

class TestSingleSourceHelper:
    """AC5: The backstop uses the SAME check_fully_invested as the Layer-2 validator."""

    def test_materialize_imports_from_invariants(self) -> None:
        """The `check_fully_invested` used inside materialize.py is imported from
        portfolio/invariants.py — confirmed by inspecting the module's namespace."""
        # The materialize module must expose check_fully_invested via its imports.
        assert hasattr(_materialize_module, "check_fully_invested"), (
            "materialize.py must import check_fully_invested from portfolio/invariants"
        )

    def test_same_function_object(self) -> None:
        """The function object in materialize's namespace IS the same object as
        the one imported directly from portfolio.invariants."""
        assert _materialize_module.check_fully_invested is check_fully_invested, (
            "materialize.check_fully_invested must be the SAME object as "
            "portfolio.invariants.check_fully_invested — one shared arithmetic."
        )


# ---------------------------------------------------------------------------
# AC1 — 8-payload production
# ---------------------------------------------------------------------------

class TestEightPayloadProduction:
    """AC1: Exactly 8 PortfolioPayload objects returned (7 personas + 1 consensus)."""

    def test_returns_eight_payloads(self) -> None:
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        assert len(payloads) == 8

    def test_payload_types_are_correct(self) -> None:
        """The 7 persona slugs + 'consensus' are all present exactly once."""
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        types = [p.type for p in payloads]
        for slug in _PERSONA_SLUGS_7:
            assert slug in types, f"Missing persona type {slug!r}"
        assert "consensus" in types
        assert len(types) == len(set(types)), "Duplicate portfolio types"

    def test_all_payloads_are_PortfolioPayload(self) -> None:
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        for p in payloads:
            assert isinstance(p, PortfolioPayload)

    def test_week_id_on_all_payloads(self) -> None:
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        for p in payloads:
            assert p.week_id == _WEEK_ID


# ---------------------------------------------------------------------------
# AC2 — Explicit CASH row per portfolio
# ---------------------------------------------------------------------------

class TestExplicitCashRow:
    """AC2: Every portfolio has exactly one CASH row; weight = 1 − Σ(positions) ≥ 0."""

    def test_exactly_one_cash_row_per_payload(self) -> None:
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        for p in payloads:
            cash_rows = [h for h in p.holdings if h.ticker == "CASH"]
            assert len(cash_rows) == 1, (
                f"Portfolio {p.type!r}: expected 1 CASH row, got {len(cash_rows)}"
            )

    def test_cash_action_is_hold(self) -> None:
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        for p in payloads:
            cash = next(h for h in p.holdings if h.ticker == "CASH")
            assert cash.action == "hold", (
                f"Portfolio {p.type!r}: CASH action must be 'hold', got {cash.action!r}"
            )

    def test_cash_weight_equals_residual(self) -> None:
        """cash.weight == 1 − Σ(position weights) within 1e-9."""
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        for p in payloads:
            cash_h = next(h for h in p.holdings if h.ticker == "CASH")
            pos_sum = sum(h.weight for h in p.holdings if h.ticker != "CASH")
            expected_cash = 1.0 - pos_sum
            assert abs(cash_h.weight - expected_cash) < 1e-9, (
                f"Portfolio {p.type!r}: CASH weight={cash_h.weight} ≠ 1−Σpos={expected_cash}"
            )

    def test_cash_weight_non_negative(self) -> None:
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        for p in payloads:
            cash = next(h for h in p.holdings if h.ticker == "CASH")
            assert cash.weight >= 0.0, (
                f"Portfolio {p.type!r}: CASH weight {cash.weight} is negative"
            )

    def test_sum_of_all_holdings_equals_1(self) -> None:
        """The invariant: Σ(all holdings including CASH) == 1.0 within 1e-6."""
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        for p in payloads:
            total = sum(h.weight for h in p.holdings)
            assert abs(total - 1.0) < 1e-6, (
                f"Portfolio {p.type!r}: Σ holdings = {total} ≠ 1.0"
            )

    @pytest.mark.parametrize("label,raw_weights", EXACT_100_FIXTURES)
    def test_exactly_100_produces_cash_row_weight_zero(
        self, label: str, raw_weights: dict[str, float]
    ) -> None:
        """Positions summing to exactly 1.0 → CASH row still present with weight=0.0."""
        # For a single-persona test, wrap in all-7 using the same weights.
        cf = {slug: dict(raw_weights) for slug in _PERSONA_SLUGS_7}
        # Consensus: strip CASH from weights (Component 16 contract).
        # When positions sum to 1.0, consensus receives full weights and the
        # materialize function derives cash = 1 - 1.0 = 0.0.
        cw = {t: w for t, w in raw_weights.items() if t != "CASH"}
        # Include all position tickers in debate_set to avoid phantom-ticker error.
        all_tickers = [t for t in raw_weights if t != "CASH"]
        # max_position_weight must be >= max position weight in fixture (all are 0.20).
        config = {"max_position_weight": _MAX_PW, "debate_set": all_tickers}

        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=config,
        )
        assert len(payloads) == 8, f"{label}: expected 8 payloads"
        for p in payloads:
            cash_rows = [h for h in p.holdings if h.ticker == "CASH"]
            assert len(cash_rows) == 1, (
                f"{label}, portfolio {p.type!r}: missing CASH row"
            )
            assert abs(cash_rows[0].weight - 0.0) < 1e-9, (
                f"{label}, portfolio {p.type!r}: CASH weight={cash_rows[0].weight} ≠ 0.0"
            )
            total = sum(h.weight for h in p.holdings)
            assert abs(total - 1.0) < 1e-6, (
                f"{label}, portfolio {p.type!r}: Σ={total} ≠ 1.0"
            )


# ---------------------------------------------------------------------------
# AC3 — >100% rejection (never clipped/rescaled)
# ---------------------------------------------------------------------------

class TestOverInvestedRejection:
    """AC3: positions summing to >100% → RuntimeError raised, NEVER clipped."""

    @pytest.mark.parametrize("label,bad_weights", OVER_INVESTED_FIXTURES)
    def test_over_invested_raises_runtime_error(
        self, label: str, bad_weights: dict[str, float]
    ) -> None:
        """A portfolio whose positions sum to >100% raises RuntimeError (Major)."""
        cf = {slug: dict(bad_weights) for slug in _PERSONA_SLUGS_7}
        cw = _make_consensus_weights()

        with pytest.raises(RuntimeError, match="Layer-3 fully-invested backstop FAILED"):
            materialize_portfolios(
                cf, cw,
                week_id=_WEEK_ID,
                entry_date=_DATE,
                config=_CFG,
            )

    @pytest.mark.parametrize("label,bad_weights", OVER_INVESTED_FIXTURES)
    def test_error_names_the_portfolio_type(
        self, label: str, bad_weights: dict[str, float]
    ) -> None:
        """The error message names the portfolio type ('value' is first alphabetically)."""
        first_slug = _PERSONA_SLUGS_7[0]
        cf = {slug: dict(bad_weights) for slug in _PERSONA_SLUGS_7}
        cw = _make_consensus_weights()

        with pytest.raises(RuntimeError) as exc_info:
            materialize_portfolios(
                cf, cw,
                week_id=_WEEK_ID,
                entry_date=_DATE,
                config=_CFG,
            )
        assert first_slug in str(exc_info.value) or "portfolio type=" in str(exc_info.value)

    @pytest.mark.parametrize("label,bad_weights", OVER_INVESTED_FIXTURES)
    def test_error_flags_layer_escape(
        self, label: str, bad_weights: dict[str, float]
    ) -> None:
        """The error message explicitly flags a Layer-1/2 ESCAPE."""
        cf = {slug: dict(bad_weights) for slug in _PERSONA_SLUGS_7}
        cw = _make_consensus_weights()

        with pytest.raises(RuntimeError) as exc_info:
            materialize_portfolios(
                cf, cw,
                week_id=_WEEK_ID,
                entry_date=_DATE,
                config=_CFG,
            )
        msg = str(exc_info.value)
        assert "Layer-1/2 ESCAPE" in msg, (
            "Error must flag this as a Layer-1/2 escape to direct the user upstream"
        )

    @pytest.mark.parametrize("label,bad_weights", OVER_INVESTED_FIXTURES)
    def test_no_clipping_or_rescaling(
        self, label: str, bad_weights: dict[str, float]
    ) -> None:
        """Verify zero payloads are returned — the call raises, not silently clips."""
        cf = {slug: dict(bad_weights) for slug in _PERSONA_SLUGS_7}
        cw = _make_consensus_weights()

        # Must raise — not return any (possibly clipped) payloads.
        with pytest.raises(RuntimeError):
            materialize_portfolios(
                cf, cw,
                week_id=_WEEK_ID,
                entry_date=_DATE,
                config=_CFG,
            )


# ---------------------------------------------------------------------------
# AC4 — Action derivation
# ---------------------------------------------------------------------------

class TestActionDerivation:
    """AC4: First-week all-add; later-week reduce/hold/exit per weight change."""

    def test_first_week_all_positions_are_add(self) -> None:
        """No prior_portfolios → all position rows are 'add'."""
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw,
            prior_portfolios=None,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        for p in payloads:
            for h in p.holdings:
                if h.ticker == "CASH":
                    assert h.action == "hold"
                else:
                    assert h.action == "add", (
                        f"Portfolio {p.type!r}, ticker {h.ticker!r}: "
                        f"expected 'add' on first week, got {h.action!r}"
                    )

    def test_later_week_hold_reduce_add(self) -> None:
        """With a prior portfolio, derive correct per-ticker action."""
        # One persona with specific prior history.
        slug = "value"
        cf = {slug: dict(VALID_SINGLE_PERSONA)}
        # Add the other 6 personas (different weights, first-week).
        for s in _PERSONA_SLUGS_7:
            if s != slug:
                cf[s] = _make_valid_counterfactual()
        cw = _make_consensus_weights()

        prior = {slug: dict(PRIOR_FOR_SINGLE_PERSONA)}
        payloads = materialize_portfolios(
            cf, cw,
            prior_portfolios=prior,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )

        value_payload = next(p for p in payloads if p.type == slug)
        holdings = _holdings_by_ticker(value_payload)

        assert holdings["AAPL"].action == "hold",  "AAPL unchanged → hold"
        assert holdings["MSFT"].action == "add",   "MSFT increased → add"
        assert holdings["NVDA"].action == "reduce", "NVDA decreased → reduce"
        assert holdings["CASH"].action == "hold",  "CASH always hold"

    def test_exit_when_prior_ticker_now_absent(self) -> None:
        """A ticker held in prior but absent (weight 0) in current → 'exit'.

        Note: exit positions are weight=0 entries — they should appear in the
        counterfactual as 0.0 (or be absent, triggering 'add' for a NEW ticker).
        This test exercises the case where the position is explicitly 0.0.
        """
        slug = "value"
        # Current portfolio: QCOM was held before but now has weight 0.
        current = {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "NVDA": 0.10,
            "QCOM": 0.0,   # previously held, now exited
            "CASH": 0.63,
        }
        cf = {slug: current}
        for s in _PERSONA_SLUGS_7:
            if s != slug:
                cf[s] = _make_valid_counterfactual()
        cw = _make_consensus_weights()
        prior = {slug: dict(PRIOR_WITH_EXITED_TICKER)}

        payloads = materialize_portfolios(
            cf, cw,
            prior_portfolios=prior,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=_CFG,
        )
        value_payload = next(p for p in payloads if p.type == slug)
        holdings = _holdings_by_ticker(value_payload)

        assert holdings["QCOM"].action == "exit", (
            "QCOM was held in prior but now weight=0 → should be 'exit'"
        )

    def test_new_ticker_not_in_prior_is_add(self) -> None:
        """A ticker not in the prior portfolio at all → 'add' (not exit)."""
        slug = "value"
        debate_set = ["AAPL", "MSFT", "INTC"]
        # INTC is not in the prior portfolio.
        current = {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "INTC": 0.10,   # brand new
            "CASH": 0.63,
        }
        prior_for_slug = {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "CASH": 0.73,
        }
        # Other 6 personas use only tickers that are in the debate set.
        cf = {slug: current}
        for s in _PERSONA_SLUGS_7:
            if s != slug:
                cf[s] = _make_valid_counterfactual(tickers=["AAPL", "MSFT", "INTC"])
        cw = _make_consensus_weights(tickers=["AAPL", "MSFT"])
        prior = {slug: prior_for_slug}
        config = {
            "max_position_weight": _MAX_PW,
            "debate_set": debate_set,
        }

        payloads = materialize_portfolios(
            cf, cw,
            prior_portfolios=prior,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=config,
        )
        value_payload = next(p for p in payloads if p.type == slug)
        holdings = _holdings_by_ticker(value_payload)
        assert holdings["INTC"].action == "add"


# ---------------------------------------------------------------------------
# AC4 — Phantom-ticker rejection
# ---------------------------------------------------------------------------

class TestPhantomTickerRejection:
    """AC4: A ticker outside the debate set AND prior → RuntimeError."""

    def test_phantom_ticker_raises(self) -> None:
        """Ticker 'PHANTOM' is not in debate_set or prior → hard rejection."""
        slug = "value"
        bad_cf = {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "PHANTOM": 0.10,   # not in debate_set
            "CASH": 0.63,
        }
        cf = {slug: bad_cf}
        for s in _PERSONA_SLUGS_7:
            if s != slug:
                cf[s] = _make_valid_counterfactual()
        cw = _make_consensus_weights()
        config = {
            "max_position_weight": _MAX_PW,
            "debate_set": ["AAPL", "MSFT", "NVDA"],   # PHANTOM not included
        }

        with pytest.raises(RuntimeError, match="phantom ticker"):
            materialize_portfolios(
                cf, cw,
                prior_portfolios=None,
                week_id=_WEEK_ID,
                entry_date=_DATE,
                config=config,
            )

    def test_valid_ticker_in_debate_set_not_rejected(self) -> None:
        """Ticker present in debate_set passes the phantom-ticker guard."""
        cf = _make_all_7_counterfactuals(tickers=["AAPL", "MSFT", "NVDA"])
        cw = _make_consensus_weights(tickers=["AAPL", "MSFT", "NVDA"])
        config = {
            "max_position_weight": _MAX_PW,
            "debate_set": ["AAPL", "MSFT", "NVDA"],
        }
        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=config,
        )
        assert len(payloads) == 8

    def test_ticker_in_prior_but_not_debate_set_is_allowed(self) -> None:
        """A ticker from the prior portfolio (not in current debate_set) is valid."""
        slug = "value"
        # QCOM was in prior, not in current debate_set.
        current = {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "QCOM": 0.10,
            "CASH": 0.63,
        }
        # Other 6 personas: only use tickers that ARE in the debate set ["AAPL","MSFT"].
        cf = {slug: current}
        for s in _PERSONA_SLUGS_7:
            if s != slug:
                cf[s] = _make_valid_counterfactual(tickers=["AAPL", "MSFT"])
        cw = _make_consensus_weights(tickers=["AAPL"])
        prior = {slug: {"QCOM": 0.15, "CASH": 0.85}}
        config = {
            "max_position_weight": _MAX_PW,
            "debate_set": ["AAPL", "MSFT"],   # QCOM NOT in debate set
        }

        # Should not raise — QCOM is in the prior for slug='value'.
        payloads = materialize_portfolios(
            cf, cw,
            prior_portfolios=prior,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=config,
        )
        assert any(p.type == slug for p in payloads)


# ---------------------------------------------------------------------------
# Consensus portfolio specifics
# ---------------------------------------------------------------------------

class TestConsensusPortfolio:
    """The consensus portfolio behaves exactly like any other — same invariant."""

    def test_consensus_payload_present(self) -> None:
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw, week_id=_WEEK_ID, entry_date=_DATE, config=_CFG
        )
        types = [p.type for p in payloads]
        assert "consensus" in types

    def test_consensus_cash_row_is_residual(self) -> None:
        """consensus CASH = 1 − Σ(consensus_weights)."""
        cf = _make_all_7_counterfactuals()
        cw = {"AAPL": 0.15, "MSFT": 0.12, "NVDA": 0.10}  # sum=0.37, cash=0.63
        payloads = materialize_portfolios(
            cf, cw, week_id=_WEEK_ID, entry_date=_DATE, config=_CFG
        )
        cp = next(p for p in payloads if p.type == "consensus")
        cash = next(h for h in cp.holdings if h.ticker == "CASH")
        assert abs(cash.weight - 0.63) < 1e-9

    def test_consensus_sum_to_one(self) -> None:
        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw, week_id=_WEEK_ID, entry_date=_DATE, config=_CFG
        )
        cp = next(p for p in payloads if p.type == "consensus")
        total = sum(h.weight for h in cp.holdings)
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end DB test — SELECT SUM(weight) = 1.0 for all 8 portfolio_ids
# ---------------------------------------------------------------------------

class TestEndToEndDBInvariant:
    """Write payloads into a real SQLite DB and verify the cash invariant."""

    def test_sum_weight_equals_one_for_all_eight(self, tmp_path: Path) -> None:
        """SELECT SUM(weight) FROM holdings WHERE portfolio_id=? = 1.0 for each."""
        from round_table_portfolio.storage.apply_schema import apply_schema

        db_path = tmp_path / "ledger.db"
        apply_schema(db_path=db_path)

        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw, week_id=_WEEK_ID, entry_date=_DATE, config=_CFG
        )
        assert len(payloads) == 8

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN")

        # Insert weeks row (required FK).
        conn.execute(
            "INSERT INTO weeks (week_id, run_date, notes, user_id) VALUES (?, ?, ?, ?)",
            (_WEEK_ID, _DATE, "test", "andrew"),
        )

        portfolio_ids: list[int] = []
        for pp in payloads:
            import datetime as _dt
            conn.execute(
                """INSERT INTO portfolios
                   (week_id, type, user_id, roster_version, enhancement_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pp.week_id, pp.type, pp.user_id, pp.roster_version,
                 pp.enhancement_version, _dt.datetime.now(_dt.timezone.utc).isoformat()),
            )
            pid = conn.execute(
                "SELECT portfolio_id FROM portfolios WHERE week_id=? AND type=? AND user_id=?",
                (pp.week_id, pp.type, pp.user_id),
            ).fetchone()[0]
            portfolio_ids.append(pid)

            for h in pp.holdings:
                conn.execute(
                    """INSERT INTO holdings
                       (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (pid, h.ticker, h.weight, h.action, h.entry_date, h.user_id, h.roster_version),
                )

        conn.commit()

        # Verify SUM(weight) = 1.0 for each portfolio_id.
        for pid in portfolio_ids:
            row = conn.execute(
                "SELECT SUM(weight) FROM holdings WHERE portfolio_id=?", (pid,)
            ).fetchone()
            total = row[0] or 0.0
            assert abs(total - 1.0) < 1e-6, (
                f"portfolio_id={pid}: SUM(weight)={total} ≠ 1.0 "
                "(cash invariant violated in DB)"
            )

        conn.close()

    def test_exactly_one_cash_row_per_portfolio_in_db(self, tmp_path: Path) -> None:
        """In the DB, exactly one ticker='CASH' row per portfolio_id."""
        from round_table_portfolio.storage.apply_schema import apply_schema
        import datetime as _dt

        db_path = tmp_path / "ledger.db"
        apply_schema(db_path=db_path)

        cf = _make_all_7_counterfactuals()
        cw = _make_consensus_weights()
        payloads = materialize_portfolios(
            cf, cw, week_id=_WEEK_ID, entry_date=_DATE, config=_CFG
        )

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO weeks (week_id, run_date, notes, user_id) VALUES (?, ?, ?, ?)",
            (_WEEK_ID, _DATE, "test", "andrew"),
        )

        portfolio_ids = []
        for pp in payloads:
            conn.execute(
                """INSERT INTO portfolios
                   (week_id, type, user_id, roster_version, enhancement_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pp.week_id, pp.type, pp.user_id, pp.roster_version,
                 pp.enhancement_version, _dt.datetime.now(_dt.timezone.utc).isoformat()),
            )
            pid = conn.execute(
                "SELECT portfolio_id FROM portfolios WHERE week_id=? AND type=? AND user_id=?",
                (pp.week_id, pp.type, pp.user_id),
            ).fetchone()[0]
            portfolio_ids.append(pid)
            for h in pp.holdings:
                conn.execute(
                    """INSERT INTO holdings
                       (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (pid, h.ticker, h.weight, h.action, h.entry_date, h.user_id, h.roster_version),
                )
        conn.commit()

        for pid in portfolio_ids:
            cash_count = conn.execute(
                "SELECT COUNT(*) FROM holdings WHERE portfolio_id=? AND ticker='CASH'", (pid,)
            ).fetchone()[0]
            assert cash_count == 1, (
                f"portfolio_id={pid}: expected 1 CASH row, got {cash_count}"
            )

        conn.close()


# ---------------------------------------------------------------------------
# Integration: weekly_run orchestrator uses the real materialize_portfolios
# ---------------------------------------------------------------------------

class TestOrchestratorUsesRealMaterialize:
    """Confirm the orchestrator no longer imports materialize_portfolios from _stubs."""

    def test_weekly_run_imports_materialize_from_portfolio(self) -> None:
        """weekly_run.materialize_portfolios must come from portfolio/materialize,
        not from orchestrator/_stubs."""
        import round_table_portfolio.orchestrator.weekly_run as weekly_run_mod
        from round_table_portfolio.portfolio.materialize import (
            materialize_portfolios as real_fn,
        )
        assert weekly_run_mod.materialize_portfolios is real_fn, (
            "weekly_run.materialize_portfolios must be the real implementation "
            "from portfolio/materialize.py, not the stub."
        )

    def test_stubs_materialize_not_imported_in_weekly_run(self) -> None:
        """The stub materialize_portfolios from _stubs should not be accessible
        via the weekly_run module namespace after TASK-M2-005 wiring."""
        import round_table_portfolio.orchestrator._stubs as stubs_mod
        import round_table_portfolio.orchestrator.weekly_run as weekly_run_mod

        stub_fn = stubs_mod.materialize_portfolios
        real_fn = weekly_run_mod.materialize_portfolios
        assert stub_fn is not real_fn, (
            "weekly_run must use the real materialize_portfolios, not the stub."
        )


# ---------------------------------------------------------------------------
# Absent-CASH-key paths (follow-up 2026-06-07)
# ---------------------------------------------------------------------------

class TestAbsentCashKey:
    """Covers the two absent-CASH-key code paths in _build_portfolio_payload.

    When a persona dict has no 'CASH' key, cash_weight initialises to 0.0.
    The backstop arithmetic then determines whether the book is valid:

      * Positions sum == 1.0 + no CASH key → cash_weight=0.0 → passes backstop
        → CASH row synthesized at weight=0.0 (same output as explicit CASH=0.0)

      * Positions sum < 1.0 + no CASH key → cash_weight=0.0 → Σ+cash ≠ 1.0
        → backstop raises RuntimeError (Layer-3 ESCAPE); residual is NOT synthesized.
    """

    def test_absent_cash_key_exact_100_produces_cash_row_weight_zero(self) -> None:
        """EXACT_100_NO_CASH_KEY: positions sum to 1.0, CASH key absent.

        Expected: backstop passes; CASH row synthesized at weight=0.0;
        Σ(all holdings) == 1.0.  This exercises the absent-key code path
        and confirms the output is identical to the explicit-zero path.
        """
        raw_weights = dict(EXACT_100_NO_CASH_KEY)  # no CASH key
        assert "CASH" not in raw_weights, "fixture must have no CASH key"

        all_tickers = list(raw_weights.keys())
        cf = {slug: dict(raw_weights) for slug in _PERSONA_SLUGS_7}
        cw = {t: w for t, w in raw_weights.items()}  # no CASH key needed for consensus
        config = {"max_position_weight": _MAX_PW, "debate_set": all_tickers}

        payloads = materialize_portfolios(
            cf, cw,
            week_id=_WEEK_ID,
            entry_date=_DATE,
            config=config,
        )
        assert len(payloads) == 8
        for p in payloads:
            cash_rows = [h for h in p.holdings if h.ticker == "CASH"]
            assert len(cash_rows) == 1, (
                f"Portfolio {p.type!r}: expected exactly 1 CASH row, got {len(cash_rows)}"
            )
            assert abs(cash_rows[0].weight - 0.0) < 1e-9, (
                f"Portfolio {p.type!r}: CASH weight={cash_rows[0].weight} — "
                "expected 0.0 when positions sum to 1.0 with no CASH key"
            )
            total = sum(h.weight for h in p.holdings)
            assert abs(total - 1.0) < 1e-6, (
                f"Portfolio {p.type!r}: Σ holdings={total} ≠ 1.0"
            )

    def test_absent_cash_key_partial_book_raises_backstop_error(self) -> None:
        """PARTIAL_NO_CASH_KEY: positions sum to 0.70, CASH key absent.

        Expected: materializer does NOT synthesize the residual (0.30 is not
        auto-filled).  Instead, check_fully_invested sees 0.70 + 0.0 ≠ 1.0
        and raises RuntimeError flagging a Layer-1/2 ESCAPE.
        """
        raw_weights = dict(PARTIAL_NO_CASH_KEY)  # positions sum to 0.70, no CASH key
        assert "CASH" not in raw_weights, "fixture must have no CASH key"

        all_tickers = list(raw_weights.keys())
        cf = {slug: dict(raw_weights) for slug in _PERSONA_SLUGS_7}
        cw = {t: w for t, w in raw_weights.items()}
        config = {"max_position_weight": _MAX_PW, "debate_set": all_tickers}

        with pytest.raises(RuntimeError, match="Layer-3 fully-invested backstop FAILED"):
            materialize_portfolios(
                cf, cw,
                week_id=_WEEK_ID,
                entry_date=_DATE,
                config=config,
            )
