"""Component 16 — consensus_equivalent (M2 simple blend; M3-swap-isolated).

Public entry point::

    from round_table_portfolio.portfolio.consensus import blend_consensus
    weights = blend_consensus(stances, config)

M2 SIMPLE BLEND ALGORITHM
--------------------------
For each debate-set ticker, compute the mean ``target_weight`` across the 7
personas that hold a NON-EXIT stance on that ticker (a plain unweighted
average — NOT conviction-weighted).  Then normalize so every ticker respects
``max_position_weight`` and Σ(weights) ≤ 1.  The residual (1 − Σ) becomes
the ``CASH`` row written by Component 15 (materialize_portfolios).

This function does NOT write any rows.  It returns a raw weights dict only.

M3-SWAP CONTRACT
----------------
This is a SINGLE pure function (no side effects, no hidden state).  In M3 the
body will be replaced with a conviction-weighted formula — the signature (full
stance set incl. ``confidence`` + ``target_weight`` in, weights dict out) is
frozen between M2 and M3.  No other component reads intermediate blend state;
only the returned weights dict is consumed downstream.

EDGE-CASE DECISIONS (documented for quality log)
-------------------------------------------------
1. All-EXIT ticker: weight → 0.0, ticker excluded from returned dict.
2. Single-holder ticker: mean over the 1 non-EXIT persona (that one's weight).
3. Over-1 raw sum after capping: scale entire dict down uniformly so Σ ≤ 1.
4. Under-1 raw sum: leave residual as cash — do NOT inflate.  Cash is natural.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_MAX_POSITION_WEIGHT: float = 0.20


def blend_consensus(
    stances: Sequence[Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Blend 7 personas' Round-1 stances into a consensus weights dict.

    M2 body: simple mean of ``target_weight`` over non-EXIT stances per ticker,
    then normalize to respect ``max_position_weight`` and Σ ≤ 1.

    M3 will replace ONLY the body of this function.  The signature and the
    returned type (``dict[str, float]``) are the M3-stable contract.

    Args:
        stances: The flat list of AgentStancePayload objects from Round1Capture
                 (7 personas × |debate_set| rows, round=1 only).  Each stance
                 must expose ``.ticker``, ``.action``, ``.target_weight``, and
                 ``.confidence`` attributes.  Consumed READ-ONLY — no mutation.
        config:  Optional dict.  Recognised key: ``max_position_weight`` (float).
                 Falls back to ``_DEFAULT_MAX_POSITION_WEIGHT`` (0.20) when
                 absent or None.

    Returns:
        A ``dict[str, float]`` mapping each debate-set ticker with at least one
        non-EXIT stance to its consensus weight.  Tickers where ALL personas
        EXIT are omitted (weight 0 → excluded).  The dict does NOT include a
        ``CASH`` key — Component 15 (materialize_portfolios) appends cash as
        ``1 − Σ(weights)``.

        Post-conditions guaranteed by this function:
        - Every value ≥ 0.
        - Every value ≤ ``max_position_weight``.
        - Σ(values) ≤ 1.0 (within floating-point precision; enforced by
          scaling-down when necessary).

    Note:
        This function is PURE — it has no side effects and writes no rows.
        Only the returned dict is consumed by any downstream component.
        No intermediate state is exposed or read by any other component
        (the M3-swap guarantee).
    """
    cfg = config or {}
    max_position_weight: float = float(
        cfg.get("max_position_weight", _DEFAULT_MAX_POSITION_WEIGHT)
    )

    # ------------------------------------------------------------------
    # Phase 1: Collect non-EXIT target_weights per ticker.
    # ------------------------------------------------------------------
    ticker_weights: dict[str, list[float]] = {}
    for s in stances:
        if s.action != "exit":
            ticker_weights.setdefault(s.ticker, []).append(s.target_weight)
        # EXIT stances contribute 0 to the mean — they are simply excluded.
        # If ALL personas EXIT a ticker, the ticker ends up with no entry in
        # ticker_weights → omitted from the output (weight 0, excluded).

    if not ticker_weights:
        logger.warning(
            "blend_consensus received zero non-EXIT stances — "
            "returning empty weights dict (100%% cash)."
        )
        return {}

    # ------------------------------------------------------------------
    # Phase 2: Simple mean per ticker.
    # ------------------------------------------------------------------
    blended: dict[str, float] = {}
    for ticker, weights in ticker_weights.items():
        mean_w = sum(weights) / len(weights)
        # Apply per-ticker cap immediately after averaging.
        capped_w = min(mean_w, max_position_weight)
        blended[ticker] = capped_w

    # ------------------------------------------------------------------
    # Phase 3: Normalize if the raw sum exceeds 1.
    #          If sum ≤ 1, leave the residual as cash (do NOT inflate).
    # ------------------------------------------------------------------
    raw_sum = sum(blended.values())

    if raw_sum > 1.0 + 1e-9:
        # Scale all weights down uniformly so Σ = 1.0.
        # This preserves relative proportions while respecting the invariant.
        scale = 1.0 / raw_sum
        blended = {t: w * scale for t, w in blended.items()}
        logger.info(
            "blend_consensus: raw sum %.6f > 1 — scaled down by %.6f; "
            "Σ after scaling = %.6f",
            raw_sum,
            scale,
            sum(blended.values()),
        )
    else:
        # Under-1 or exactly-1: leave as-is.  The residual is cash.
        logger.info(
            "blend_consensus: raw sum %.6f ≤ 1 — residual %.6f becomes cash.",
            raw_sum,
            1.0 - raw_sum,
        )

    # ------------------------------------------------------------------
    # Phase 4: Re-assert the cap after scaling (floating-point safety).
    # ------------------------------------------------------------------
    for ticker in list(blended):
        if blended[ticker] > max_position_weight + 1e-9:
            # Should not happen after proper scaling, but guard loudly.
            raise RuntimeError(
                f"MAJOR: blend_consensus post-scaling cap violation for "
                f"ticker={ticker!r}: weight={blended[ticker]:.6f} > "
                f"max_position_weight={max_position_weight:.4f}.  "
                "Root-cause the normalization step."
            )

    logger.info(
        "blend_consensus complete: %d tickers, Σ=%.6f, max_position_weight=%.4f",
        len(blended),
        sum(blended.values()),
        max_position_weight,
    )

    return blended
