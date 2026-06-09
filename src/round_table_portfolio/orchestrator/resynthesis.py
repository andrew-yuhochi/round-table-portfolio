"""Component 25 — re-synthesis (re-blend revised Round-2 stances).

After Round 2, the 2 outliers have either defended or revised their stances.
This module builds the MERGED stance set (5 non-outlier Round-1 stances + 2
outlier Round-2 stances), reblends via the SAME ``blend_consensus`` function
used in M2's step 4, and computes the provisional → final DELTA so the shift
is visible in the transcript.

Public entry point::

    from round_table_portfolio.orchestrator.resynthesis import resynthesize_consensus
    result = resynthesize_consensus(
        round1_stances=round1.stances,         # flat list, all 7 personas, round=1
        round2_payloads=round2_stance_payloads, # list of AgentStancePayload, round=2
        outlier_personas={"growth", "cta-systematic-macro"},
        provisional_weights=consensus_weights,  # dict[str,float] from M2 step 4
        config={"max_position_weight": 0.20},
    )
    # result.final_weights   — dict[str, float], ready for Component 15
    # result.delta           — dict[str, float], per-ticker weight change

Design contract (M3-swap guarantee)
-------------------------------------
This module NEVER reimplements blend math.  It imports and calls
``blend_consensus`` from ``round_table_portfolio.portfolio.consensus``.
All normalization, cap enforcement, and Σ ≤ 1 guarantees are delegated there.

MERGE RULE
----------
For every stance in the debate set × 7 personas:
  - If the persona IS an outlier  → use the round=2 stance (by ticker).
  - If the persona is NOT outlier → use the round=1 stance (unchanged).

If an outlier emitted round=2 stances for only a SUBSET of the debate-set
tickers, the remaining tickers for that outlier fall back to their round=1
stances (the outlier implicitly defended those).  This is consistent with the
position_change='defended' semantic.

DELTA
-----
delta[ticker] = final_weights.get(ticker, 0.0) - provisional_weights.get(ticker, 0.0)

Tickers that appear in only one dict (entered or exited the consensus) carry a
positive or negative delta equal to their final weight (or the negative of their
provisional weight).

FAILURE MODES (Gate 5 — root-cause-first)
------------------------------------------
- If ``final_weights == provisional_weights`` when at least one outlier REVISED
  at least one ticker: caller should treat this as a merge bug (Major tier —
  a round-1 stance was not replaced, or a non-outlier's stance changed).
- If ``blend_consensus`` raises a RuntimeError (cap violation after scaling):
  do NOT catch — propagate immediately so the caller can diagnose (Major tier).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from round_table_portfolio.orchestrator.round1 import AgentStancePayload
from round_table_portfolio.portfolio.consensus import blend_consensus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ResynthesisResult:
    """Output of ``resynthesize_consensus``.

    Attributes:
        final_weights:       Post-Round-2 consensus weights dict (no CASH key).
                             Ready to pass to Component 15 ``materialize_portfolios``
                             as the ``consensus_weights`` argument.
        provisional_weights: The Round-1 provisional weights passed in
                             (stored for reference / transcript writing).
        delta:               Per-ticker weight change: final − provisional.
                             Tickers present in one dict only carry ± their full
                             weight as the delta.
        merged_stances:      The merged stance list that was fed into
                             ``blend_consensus`` — stored for test assertions and
                             transcript writing.
        outlier_personas:    Frozenset of the outlier persona slugs whose
                             round=2 stances replaced their round=1 stances.
    """
    final_weights: dict[str, float]
    provisional_weights: dict[str, float]
    delta: dict[str, float]
    merged_stances: list[AgentStancePayload]
    outlier_personas: frozenset[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resynthesize_consensus(
    round1_stances: Sequence[AgentStancePayload],
    round2_payloads: Sequence[AgentStancePayload],
    *,
    outlier_personas: set[str] | frozenset[str],
    provisional_weights: dict[str, float],
    config: dict[str, Any] | None = None,
) -> ResynthesisResult:
    """Merge Round-2 outlier stances into the Round-1 stance set and reblend.

    Calls the SAME ``blend_consensus(merged_stances, config)`` used in M2.
    No blend math is re-implemented here.

    Args:
        round1_stances:      Flat list of AgentStancePayload (round=1).
                             Contains stances for ALL 7 personas × debate-set.
        round2_payloads:     Flat list of AgentStancePayload (round=2).
                             Contains restated stances from the 2 outlier personas.
                             Each entry has .persona, .ticker, .action,
                             .target_weight, .confidence — same shape as
                             AgentStancePayload (same dataclass, round=2).
        outlier_personas:    Set of exactly 2 persona slugs whose round=2
                             stances replace their round=1 stances in the merge.
        provisional_weights: Round-1 consensus weights (output of the M2
                             ``blend_consensus`` call — NO CASH key).
                             Used as the baseline for delta computation.
        config:              Optional config dict passed through to
                             ``blend_consensus``.  Recognised key:
                             ``max_position_weight`` (float, default 0.20).

    Returns:
        ResynthesisResult with final_weights, delta, merged_stances, and a
        copy of provisional_weights.

    Raises:
        ValueError:     If ``outlier_personas`` is empty — merge would be a
                        no-op with no semantic content.
        RuntimeError:   Propagated from ``blend_consensus`` on cap violation
                        (Major tier — do not catch here).
    """
    if not outlier_personas:
        raise ValueError(
            "resynthesize_consensus: outlier_personas must be non-empty. "
            "An empty set means no round=2 stances exist — call blend_consensus "
            "directly on round=1 stances instead."
        )

    _outliers = frozenset(outlier_personas)

    # ------------------------------------------------------------------
    # Build a lookup: (persona, ticker) → round=2 AgentStancePayload
    # for each of the 2 outliers.
    # ------------------------------------------------------------------
    round2_by_key: dict[tuple[str, str], AgentStancePayload] = {}
    for s in round2_payloads:
        if s.persona in _outliers:
            round2_by_key[(s.persona, s.ticker)] = s

    logger.info(
        "resynthesize_consensus: outliers=%s, round2 stances captured=%d",
        sorted(_outliers),
        len(round2_by_key),
    )

    # ------------------------------------------------------------------
    # Merge: for each round=1 stance, replace with round=2 if it exists
    # for an outlier persona on that ticker.  Non-outliers are untouched.
    # Outlier tickers NOT restated in round=2 fall back to round=1 (the
    # outlier implicitly defended those).
    # ------------------------------------------------------------------
    merged: list[AgentStancePayload] = []
    replaced_count = 0

    for s in round1_stances:
        if s.persona in _outliers:
            r2 = round2_by_key.get((s.persona, s.ticker))
            if r2 is not None:
                merged.append(r2)
                replaced_count += 1
            else:
                # Outlier did not restate this ticker → keep round=1 (implicit defend).
                merged.append(s)
        else:
            merged.append(s)

    # Also append any round=2 stances for tickers that were NOT in round=1
    # (the outlier added a NEW ticker in round=2 that was not in the debate set).
    # This is theoretically prevented by the parser (only debate-set tickers
    # can appear in round=2), but guard it defensively: only add if the key
    # was not already consumed above.
    consumed_keys: set[tuple[str, str]] = {
        (s.persona, s.ticker)
        for s in round1_stances
        if s.persona in _outliers
    }
    for key, s in round2_by_key.items():
        if key not in consumed_keys:
            logger.warning(
                "resynthesize_consensus: round=2 stance for (%s, %s) has no "
                "matching round=1 stance — appending as new (check debate set).",
                key[0], key[1],
            )
            merged.append(s)

    logger.info(
        "resynthesize_consensus: merged set size=%d, replaced=%d round=1→round=2",
        len(merged),
        replaced_count,
    )

    # ------------------------------------------------------------------
    # Reblend using the SAME blend_consensus function (no re-implementation).
    # ------------------------------------------------------------------
    final_weights = blend_consensus(merged, config)

    # ------------------------------------------------------------------
    # Compute provisional → final DELTA per ticker.
    # ------------------------------------------------------------------
    all_tickers = set(provisional_weights) | set(final_weights)
    delta: dict[str, float] = {}
    for ticker in all_tickers:
        d = final_weights.get(ticker, 0.0) - provisional_weights.get(ticker, 0.0)
        if abs(d) > 1e-12:  # omit zero-change tickers for clarity
            delta[ticker] = d

    logger.info(
        "resynthesize_consensus: provisional Σ=%.6f → final Σ=%.6f; "
        "%d tickers moved (delta != 0)",
        sum(provisional_weights.values()),
        sum(final_weights.values()),
        len(delta),
    )

    return ResynthesisResult(
        final_weights=final_weights,
        provisional_weights=dict(provisional_weights),
        delta=delta,
        merged_stances=merged,
        outlier_personas=_outliers,
    )


# ---------------------------------------------------------------------------
# Convenience: verify that final_weights was produced by blend_consensus
# (used by test_ac1_blend_reuse_proof — the test reconstructs the merged set
# independently and calls blend_consensus directly, then asserts equality).
# ---------------------------------------------------------------------------

def verify_blend_reuse(
    merged_stances: Sequence[AgentStancePayload],
    claimed_final_weights: dict[str, float],
    config: dict[str, Any] | None = None,
) -> bool:
    """Return True iff blend_consensus(merged_stances, config) == claimed_final_weights.

    This is the AC1 proof helper: reconstruct the merged set externally, call
    blend_consensus directly, and assert the weights are identical — confirming
    no parallel blend math was used.

    Args:
        merged_stances:        The merged stance list (from ResynthesisResult).
        claimed_final_weights: The final_weights from ResynthesisResult.
        config:                Same config passed to resynthesize_consensus.

    Returns:
        True if they match exactly (within 1e-9 per ticker).
    """
    recomputed = blend_consensus(merged_stances, config)

    all_keys = set(recomputed) | set(claimed_final_weights)
    for ticker in all_keys:
        r = recomputed.get(ticker, 0.0)
        c = claimed_final_weights.get(ticker, 0.0)
        if abs(r - c) > 1e-9:
            logger.error(
                "verify_blend_reuse MISMATCH: ticker=%r recomputed=%.8f claimed=%.8f",
                ticker, r, c,
            )
            return False
    return True
