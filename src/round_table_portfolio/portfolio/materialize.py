"""Component 15 — counterfactual_portfolio_capture (Layer-3 BACKSTOP).

Converts per-persona counterfactual dicts + consensus weights into
PortfolioPayload objects ready for the ledger transaction (TASK-M2-005).

Layer-3 backstop:
    Before returning each portfolio, ``check_fully_invested`` (the SAME helper
    that Component 11 / Layer-2 validator uses) is called.  A failure here
    means Layer 1 or Layer 2 let a malformed book through — the error message
    says so explicitly.  Nothing is clipped or rescaled; the caller receives a
    raised exception and zero rows for that portfolio.

CASH representation:
    Every portfolio payload carries exactly ONE explicit ``ticker='CASH'`` row
    with ``action='hold'`` and ``weight = 1 − Σ(position weights)``.  This
    is per the Component-15 / schema decision recorded in TASK-M2-001 and
    confirmed in TASK-M2-002's Layer-1 contract.

Action derivation:
    Each position's ``action`` is derived by comparing the current weight
    against the same portfolio type's prior week holdings (``prior_portfolios``).
    Rules:
      - If there is no prior portfolio for this type → all positions are 'add'.
      - If the ticker was not in the prior portfolio → 'add'.
      - If weight > prior weight (with tol) → 'add'.
      - If weight < prior weight (with tol) → 'reduce'.
      - If weight ≈ prior weight (within tol) → 'hold'.
      - If the prior held the ticker but the current weight is 0 → 'exit'.
    The CASH row is ALWAYS 'hold'.

Phantom-ticker guard:
    Every non-CASH position ticker must appear either in the debate set OR in
    the prior portfolio for that type.  A ticker that is in neither is a
    phantom — the function raises immediately (fail-loudly, no silent discard).

Entry point::

    from round_table_portfolio.portfolio.materialize import materialize_portfolios
    payloads = materialize_portfolios(
        counterfactuals,
        consensus_weights,
        prior_portfolios=prior,
        week_id="2026-W23",
        config={"max_position_weight": 0.20, "debate_set": ["AAPL", "MSFT"]},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from round_table_portfolio.portfolio.invariants import check_fully_invested

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_POSITION_WEIGHT: float = 0.20
_ACTION_TOL: float = 1e-6  # weight-change tolerance for hold vs add/reduce
_CONSENSUS_TYPE: str = "consensus"

# The 7 named-persona slugs in canonical order (matches schema CHECK constraint).
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
# Payload dataclasses — SAME shapes as the orchestrator writes.
# Defined here as the canonical source; _stubs.py types are parallel
# (kept as stubs only) — the orchestrator imports from here after TASK-M2-005.
# ---------------------------------------------------------------------------

@dataclass
class HoldingPayload:
    """One row payload for the ``holdings`` table."""
    ticker: str
    weight: float
    action: str     # 'add' | 'reduce' | 'hold' | 'exit'
    entry_date: str
    user_id: str = "andrew"
    roster_version: int = 1


@dataclass
class PortfolioPayload:
    """One portfolio + its holdings, ready for ledger write."""
    type: str
    week_id: str
    roster_version: int
    enhancement_version: int
    user_id: str
    holdings: list[HoldingPayload] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def materialize_portfolios(
    counterfactuals: dict[str, dict[str, float]],
    consensus_weights: dict[str, float],
    *,
    prior_portfolios: dict[str, dict[str, float]] | None = None,
    week_id: str,
    config: dict[str, Any] | None = None,
    entry_date: str = "",
    user_id: str = "andrew",
    roster_version: int = 1,
    enhancement_version: int = 1,
) -> list[PortfolioPayload]:
    """Materialize 8 PortfolioPayload objects (7 personas + 1 consensus).

    For each of the 8 portfolios this function:
      1. Separates positions from the CASH entry.
      2. Asserts no phantom tickers.
      3. Derives ``action`` per position against ``prior_portfolios``.
      4. Calls ``check_fully_invested`` (Layer-3 BACKSTOP) — raises on failure.
      5. Appends the explicit CASH row (``action='hold'``).
      6. Returns the completed PortfolioPayload.

    Args:
        counterfactuals:   Per-persona portfolio dicts.
                           Each dict maps ticker → weight INCLUDING a 'CASH' key.
                           Keys are persona slugs matching the 7 _PERSONA_SLUGS_7.
        consensus_weights: Weights dict from ``blend_consensus`` — does NOT
                           include a 'CASH' key (Component 16 contract).
        prior_portfolios:  Optional mapping of portfolio type → prior week's
                           weights dict (WITH CASH).  Pass ``None`` or ``{}``
                           for the first week (all positions become 'add').
        week_id:           ISO week label, e.g. "2026-W23".
        config:            Optional dict.  Recognised keys:
                             ``max_position_weight`` (float, default 0.20),
                             ``debate_set``          (list[str], optional).
        entry_date:        ISO date string for HoldingPayload.entry_date.
                           Falls back to ``week_id`` when empty.
        user_id:           Owner field (default "andrew").
        roster_version:    Ledger version field (default 1).
        enhancement_version: Ledger version field (default 1).

    Returns:
        A list of exactly 8 PortfolioPayload objects — one per portfolio type.
        The list order is: 7 named-persona slugs (in _PERSONA_SLUGS_7 order),
        then the consensus.  Every payload carries exactly one CASH row.

    Raises:
        RuntimeError (Major):  If any portfolio fails the Layer-3 backstop
            check (over-invested, negative cash, sum ≠ 100%, over-cap position).
            The error message names the portfolio type and flags it as a
            Layer-1/2 escape.  Zero rows are returned for that portfolio.
        RuntimeError (phantom-ticker): If any non-CASH position ticker is
            neither in the current debate set nor in the prior portfolio for
            that type.
    """
    cfg = config or {}
    max_position_weight: float = float(
        cfg.get("max_position_weight", _DEFAULT_MAX_POSITION_WEIGHT)
    )
    debate_set_set: set[str] = {
        t.upper() for t in cfg.get("debate_set", [])
    }
    prior = prior_portfolios or {}
    _entry_date = entry_date or week_id

    payloads: list[PortfolioPayload] = []

    # -----------------------------------------------------------------------
    # 7 named-persona portfolios.
    # -----------------------------------------------------------------------
    for slug in _PERSONA_SLUGS_7:
        if slug not in counterfactuals:
            logger.warning(
                "materialize_portfolios: no counterfactual for persona=%r — skipping.",
                slug,
            )
            continue

        raw_weights = counterfactuals[slug]
        payload = _build_portfolio_payload(
            portfolio_type=slug,
            raw_weights=raw_weights,
            prior_weights=prior.get(slug, {}),
            debate_set=debate_set_set,
            week_id=week_id,
            entry_date=_entry_date,
            max_position_weight=max_position_weight,
            user_id=user_id,
            roster_version=roster_version,
            enhancement_version=enhancement_version,
        )
        payloads.append(payload)

    # -----------------------------------------------------------------------
    # 1 consensus portfolio.
    # consensus_weights does NOT carry CASH (Component 16 contract) — the cash
    # residual is computed here as 1 − Σ(positions).
    # -----------------------------------------------------------------------
    consensus_raw = dict(consensus_weights)  # copy; no CASH key yet
    cash_weight = 1.0 - sum(consensus_raw.values())
    consensus_raw["CASH"] = cash_weight

    payload = _build_portfolio_payload(
        portfolio_type=_CONSENSUS_TYPE,
        raw_weights=consensus_raw,
        prior_weights=prior.get(_CONSENSUS_TYPE, {}),
        debate_set=debate_set_set,
        week_id=week_id,
        entry_date=_entry_date,
        max_position_weight=max_position_weight,
        user_id=user_id,
        roster_version=roster_version,
        enhancement_version=enhancement_version,
    )
    payloads.append(payload)

    logger.info(
        "materialize_portfolios: produced %d PortfolioPayload objects for week=%s",
        len(payloads),
        week_id,
    )
    return payloads


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_portfolio_payload(
    *,
    portfolio_type: str,
    raw_weights: dict[str, float],
    prior_weights: dict[str, float],
    debate_set: set[str],
    week_id: str,
    entry_date: str,
    max_position_weight: float,
    user_id: str,
    roster_version: int,
    enhancement_version: int,
) -> PortfolioPayload:
    """Build one PortfolioPayload from a raw weights dict (includes CASH key).

    Raises RuntimeError on backstop failure or phantom-ticker detection.
    """
    # Separate positions from CASH.
    positions: dict[str, float] = {}
    cash_weight: float = 0.0

    for ticker, weight in raw_weights.items():
        t_upper = ticker.upper()
        if t_upper == "CASH":
            cash_weight = float(weight)
        else:
            positions[t_upper] = float(weight)

    # -----------------------------------------------------------------------
    # Phantom-ticker guard.
    # A position ticker is valid if it is in the debate set OR was held in the
    # prior portfolio for this type.  An unrecognised ticker is a phantom.
    # -----------------------------------------------------------------------
    prior_tickers: set[str] = {
        t.upper() for t in prior_weights if t.upper() != "CASH"
    }
    allowed_tickers: set[str] = debate_set | prior_tickers

    if allowed_tickers:
        # Only enforce when we actually know the debate set or have prior data.
        for ticker in positions:
            if ticker not in allowed_tickers:
                raise RuntimeError(
                    f"MAJOR — phantom ticker detected in portfolio type={portfolio_type!r}: "
                    f"ticker={ticker!r} is neither in the debate set "
                    f"({sorted(debate_set)}) nor in the prior portfolio "
                    f"({sorted(prior_tickers)}).  "
                    "Root-cause: the persona introduced a ticker outside the "
                    "debate set without prior authorization."
                )

    # -----------------------------------------------------------------------
    # Layer-3 BACKSTOP — same arithmetic as Layer-2 (Component 11).
    # -----------------------------------------------------------------------
    passed, reasons = check_fully_invested(
        positions,
        cash_weight,
        max_position_weight,
    )
    if not passed:
        violations = "; ".join(reasons)
        raise RuntimeError(
            f"MAJOR — Layer-3 fully-invested backstop FAILED for "
            f"portfolio type={portfolio_type!r}: {violations}.  "
            "This is a Layer-1/2 ESCAPE: Layer 1 (persona contract) or "
            "Layer 2 (output validator) failed to catch a malformed book "
            "before it reached ledger-write.  Root-cause the upstream layers — "
            "do NOT clip or rescale here."
        )

    # -----------------------------------------------------------------------
    # Action derivation against prior portfolio of same type.
    # -----------------------------------------------------------------------
    prior_pos: dict[str, float] = {
        t.upper(): float(w)
        for t, w in prior_weights.items()
        if t.upper() != "CASH"
    }
    first_week_for_type = not bool(prior_weights)

    holdings: list[HoldingPayload] = []

    for ticker, weight in sorted(positions.items()):  # sorted for determinism
        action = _derive_action(
            ticker=ticker,
            weight=weight,
            prior_pos=prior_pos,
            first_week=first_week_for_type,
        )
        holdings.append(HoldingPayload(
            ticker=ticker,
            weight=weight,
            action=action,
            entry_date=entry_date,
            user_id=user_id,
            roster_version=roster_version,
        ))

    # Explicit CASH row — always action='hold'.
    holdings.append(HoldingPayload(
        ticker="CASH",
        weight=cash_weight,
        action="hold",
        entry_date=entry_date,
        user_id=user_id,
        roster_version=roster_version,
    ))

    logger.debug(
        "Portfolio type=%r: %d position(s) + CASH=%.6f; backstop PASSED.",
        portfolio_type,
        len(positions),
        cash_weight,
    )

    return PortfolioPayload(
        type=portfolio_type,
        week_id=week_id,
        roster_version=roster_version,
        enhancement_version=enhancement_version,
        user_id=user_id,
        holdings=holdings,
    )


def _derive_action(
    *,
    ticker: str,
    weight: float,
    prior_pos: dict[str, float],
    first_week: bool,
) -> str:
    """Derive the action for a single position vs the prior portfolio.

    Rules:
      - first_week=True OR ticker not in prior → 'add'
      - weight ≈ prior weight (within _ACTION_TOL) → 'hold'
      - weight > prior weight → 'add'
      - weight < prior weight and weight > 0 → 'reduce'
      - weight == 0 but prior weight > 0 → 'exit'
        (zero-weight positions should not appear as holdings, but guard anyway)

    Returns one of: 'add' | 'reduce' | 'hold' | 'exit'
    """
    if first_week or ticker not in prior_pos:
        return "add"

    prior_weight = prior_pos[ticker]
    diff = weight - prior_weight

    if abs(diff) <= _ACTION_TOL:
        return "hold"
    if diff > 0:
        return "add"
    # diff < 0
    if weight <= _ACTION_TOL:
        return "exit"
    return "reduce"
