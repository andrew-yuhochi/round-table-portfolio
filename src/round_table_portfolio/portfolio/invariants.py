"""Portfolio invariant helpers — Component 11 (Layer 2) and Component 15 (Layer 3).

This is the SINGLE arithmetic source for the fully-invested check.
Both the output validator (Component 11, TASK-M2-002) and the ledger-write
backstop (Component 15, TASK-M2-005) import ``check_fully_invested`` from
here — never from each other and never re-implemented inline.

Entry point::

    from round_table_portfolio.portfolio.invariants import check_fully_invested
    passed, reasons = check_fully_invested(positions, cash, max_position_weight)

NEVER rescales or clips weights.  Reports violations only — the caller decides
what to do (flag, raise, or surface to the founder).
"""

from __future__ import annotations


def check_fully_invested(
    positions: dict[str, float],
    cash: float,
    max_position_weight: float,
    *,
    tol: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Check that a portfolio satisfies the fully-invested invariant.

    Rules checked (in order):
    1. ``cash >= 0`` — cash cannot be negative.
    2. All position weights ``>= 0`` — no short positions.
    3. ``Σ(positions) + cash == 1.0`` within ``tol`` — book sums to 100%.
    4. No single position weight ``> max_position_weight`` — concentration cap.

    Args:
        positions: Mapping of ticker → weight for the non-cash positions.
                   The ``CASH`` key must NOT appear here; pass it via ``cash``.
        cash:      The explicit cash weight (``1 − Σ(positions)`` in a
                   well-formed book).
        max_position_weight: Per-ticker hard ceiling from ``config/thresholds.yaml``.
        tol:       Float tolerance for the sum-to-1.0 check (default 1e-6).

    Returns:
        ``(passed, reasons)`` where ``passed`` is ``True`` iff all rules hold
        and ``reasons`` is a (possibly empty) list of human-readable violation
        strings — one string per violated rule.

    Notes:
        - This function NEVER rescales or clips.  The caller must not silence
          the violation; surfacing it is the whole point.
        - When ``positions`` is empty and ``cash == 1.0`` the book is fully in
          cash and passes (a valid — if extreme — portfolio state).
    """
    reasons: list[str] = []

    # Rule 1 — cash cannot be negative.
    if cash < -tol:
        reasons.append(
            f"Cash weight is negative ({cash:.6f}): portfolio is over-invested."
        )

    # Rule 2 — no negative position weights.
    for ticker, weight in positions.items():
        if weight < -tol:
            reasons.append(
                f"Position weight for {ticker!r} is negative ({weight:.6f})."
            )

    # Rule 3 — Σ(positions) + cash must equal 1.0.
    total = sum(positions.values()) + cash
    if abs(total - 1.0) > tol:
        reasons.append(
            f"Portfolio does not sum to 1.0: Σ(positions) + cash = {total:.8f} "
            f"(deviation {total - 1.0:+.2e}, tolerance {tol:.0e})."
        )

    # Rule 4 — per-ticker concentration cap.
    for ticker, weight in positions.items():
        if weight > max_position_weight + tol:
            reasons.append(
                f"Position weight for {ticker!r} ({weight:.4f}) exceeds "
                f"max_position_weight ({max_position_weight:.4f})."
            )

    return (len(reasons) == 0, reasons)
