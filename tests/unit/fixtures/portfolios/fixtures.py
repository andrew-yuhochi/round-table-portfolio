"""Portfolio materialize fixtures for TASK-M2-005.

Three >100%-rejection fixtures, two exactly-100% explicit-zero-cash fixtures,
and two absent-CASH-key fixtures (CASH key truly omitted from the dict).

Provenance: synthetic — hand-crafted to drive specific arithmetic paths in
``materialize_portfolios``.  Tickers chosen from the standard fixture debate
set (AAPL, MSFT, NVDA) used across the M2 test suite.

Fixture naming:
    OVER_101                  — positions sum to 1.01 (CASH = -0.01) → REJECTED (Major)
    OVER_105                  — positions sum to 1.05 (CASH = -0.05) → REJECTED (Major)
    OVER_150                  — positions sum to 1.50 (CASH = -0.50) → REJECTED (Major)
    EXACT_100_EXPLICIT_ZERO_CASH — positions sum to exactly 1.0, CASH key present as 0.0
                                   → CASH row with weight=0.0 must still appear
    EXACT_100_CASH_ZERO          — same logic, different weight split; second explicit-zero path
    EXACT_100_NO_CASH_KEY        — positions sum to exactly 1.0, NO CASH key in dict at all
                                   → materializer defaults cash_weight=0.0; backstop passes;
                                      CASH row synthesized at weight=0.0
    PARTIAL_NO_CASH_KEY          — positions sum to 0.70, NO CASH key in dict at all
                                   → materializer sees cash_weight=0.0; backstop FAILS
                                      (0.70 + 0.0 ≠ 1.0); RuntimeError raised
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# >100% rejection fixtures (3)
# ---------------------------------------------------------------------------

# positions sum to 0.20 + 0.45 + 0.36 = 1.01; CASH = -0.01
OVER_101: dict[str, float] = {
    "AAPL": 0.20,
    "MSFT": 0.45,
    "NVDA": 0.36,
    "CASH": -0.01,
}

# positions sum to 0.20 + 0.45 + 0.40 = 1.05; CASH = -0.05
OVER_105: dict[str, float] = {
    "AAPL": 0.20,
    "MSFT": 0.45,
    "NVDA": 0.40,
    "CASH": -0.05,
}

# positions sum to 0.50 + 0.60 + 0.40 = 1.50; CASH = -0.50
OVER_150: dict[str, float] = {
    "AAPL": 0.50,
    "MSFT": 0.60,
    "NVDA": 0.40,
    "CASH": -0.50,
}

# List of all >100% fixtures with their expected violation description
OVER_INVESTED_FIXTURES: list[tuple[str, dict[str, float]]] = [
    ("over_101", OVER_101),
    ("over_105", OVER_105),
    ("over_150", OVER_150),
]

# ---------------------------------------------------------------------------
# Exactly-100%-no-cash fixtures (2)
# ---------------------------------------------------------------------------

# Positions sum to exactly 1.0; CASH key IS present as explicit 0.0.
# All position weights ≤ 0.20 (respects max_position_weight cap).
# materialize_portfolios must still emit a CASH row with weight=0.0.
EXACT_100_EXPLICIT_ZERO_CASH: dict[str, float] = {
    "AAPL": 0.20,
    "MSFT": 0.20,
    "NVDA": 0.20,
    "GOOGL": 0.20,
    "AMD": 0.20,
    "CASH": 0.0,   # explicit 0.0 key present
}

# Different weight split (5 positions × 0.20 = 1.0), still exactly 1.0 with CASH=0.0 explicit.
EXACT_100_CASH_ZERO: dict[str, float] = {
    "AAPL": 0.20,
    "MSFT": 0.20,
    "NVDA": 0.20,
    "QCOM": 0.20,
    "INTC": 0.20,
    "CASH": 0.0,
}

# List of exactly-100% fixtures (both have explicit CASH=0.0 key present)
EXACT_100_FIXTURES: list[tuple[str, dict[str, float]]] = [
    ("exact_100_explicit_zero_cash", EXACT_100_EXPLICIT_ZERO_CASH),
    ("exact_100_cash_zero", EXACT_100_CASH_ZERO),
]

# ---------------------------------------------------------------------------
# Absent-CASH-key fixtures (CASH key truly omitted)
# ---------------------------------------------------------------------------

# Positions sum to exactly 1.0; NO "CASH" key in the dict at all.
# Behavior: _build_portfolio_payload initialises cash_weight=0.0 (the default).
# check_fully_invested sees Σpositions(1.0) + cash(0.0) = 1.0 → PASSES.
# The materializer then synthesizes the CASH row at weight=0.0 — same output
# as EXACT_100_EXPLICIT_ZERO_CASH, but via the absent-key code path.
EXACT_100_NO_CASH_KEY: dict[str, float] = {
    "AAPL": 0.20,
    "MSFT": 0.20,
    "NVDA": 0.20,
    "GOOGL": 0.20,
    "AMD": 0.20,
    # No "CASH" key — cash_weight defaults to 0.0 inside _build_portfolio_payload
}

# Positions sum to 0.70; NO "CASH" key in the dict at all.
# Behavior: cash_weight=0.0 (default), Σpositions(0.70) + cash(0.0) = 0.70 ≠ 1.0.
# check_fully_invested FAILS → RuntimeError raised.
# The materializer does NOT synthesize the residual — that is the persona's job.
PARTIAL_NO_CASH_KEY: dict[str, float] = {
    "AAPL": 0.20,
    "MSFT": 0.25,
    "NVDA": 0.25,
    # No "CASH" key — residual 0.30 is silently lost; backstop catches it
}

# ---------------------------------------------------------------------------
# Standard valid fixture (used for action-derivation and general tests)
# ---------------------------------------------------------------------------

# A well-formed single-persona counterfactual — 3 positions + CASH = 1.0.
VALID_SINGLE_PERSONA: dict[str, float] = {
    "AAPL": 0.15,
    "MSFT": 0.12,
    "NVDA": 0.10,
    "CASH": 0.63,
}

# A prior portfolio for the same type (used for action-derivation tests).
# AAPL stays the same → 'hold'; MSFT increases → 'add'; NVDA decreases → 'reduce'.
PRIOR_FOR_SINGLE_PERSONA: dict[str, float] = {
    "AAPL": 0.15,   # same  → hold
    "MSFT": 0.08,   # lower → current 0.12 > prior 0.08 → add
    "NVDA": 0.14,   # higher → current 0.10 < prior 0.14 → reduce
    "CASH": 0.63,
}

# Expected actions for VALID_SINGLE_PERSONA vs PRIOR_FOR_SINGLE_PERSONA.
EXPECTED_ACTIONS_WITH_PRIOR: dict[str, str] = {
    "AAPL": "hold",
    "MSFT": "add",
    "NVDA": "reduce",
    "CASH": "hold",
}

# A prior portfolio where a ticker was held but is now absent (weight 0 = exit).
PRIOR_WITH_EXITED_TICKER: dict[str, float] = {
    "AAPL": 0.15,
    "MSFT": 0.12,
    "NVDA": 0.10,
    "QCOM": 0.10,   # held previously
    "CASH": 0.53,
}
