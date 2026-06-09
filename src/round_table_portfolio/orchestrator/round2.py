"""Component 24 (deterministic parts) — Round-2 stance capture: schema, parser, writer.

This module handles the DETERMINISTIC plumbing half of Component 24:
  1. ROUND 2 OUTPUT SCHEMA — the dataclass contract the live outlier subagent emits.
  2. Fail-loud PARSER — parses each outlier's Round-2 JSON, reusing Component 14's
     action-vocabulary normalization and domain-enforcement logic from round1.py.
  3. round=2 WRITER — produces AgentStancePayload rows (round=2) the orchestrator
     inserts inside its week transaction, coexisting with round=1 rows under
     UNIQUE(week_id, persona, ticker, round, user_id).

The LIVE DISPATCH of the 2 outlier subagents happens at TASK-M3-006 (the
engine/session split: a plain Python module cannot spawn Claude Code subagents).
This module does not perform any dispatch.

Reuse contract (Gate 8 — no duplicate domain enforcement):
    _ACTION_NORMALISE, _VALID_ACTIONS, and the per-field validation rules are
    imported directly from round1.py.  Round 2 adds only the new fields
    (rebuttal_narrative, position_change) on top of the same stance shape.

Fail-loudly principle (TDD §1.4 / §1.5 / Gate 5):
    Out-of-domain action, confidence outside 1..5, target_weight outside
    [0, max_position_weight], invalid position_change, or missing rebuttal_narrative
    all raise Round2ParseError immediately.  No silent coercion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from round_table_portfolio.orchestrator.round1 import (
    AgentStancePayload,
    _ACTION_NORMALISE,      # noqa: PLC2701 — intentional reuse of C14 normalization map
    _VALID_ACTIONS,         # noqa: PLC2701 — intentional reuse of C14 valid-action set
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_POSITION_CHANGES = frozenset({"defended", "revised"})


# ---------------------------------------------------------------------------
# Round-2 schema types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Round2Stance:
    """One restated stance from a Round-2 outlier reply.

    Same shape as Round 1 (action/target_weight/confidence/rationale) plus an
    explicit position_change marker indicating whether the outlier defended
    or revised its Round-1 position on this ticker.
    """
    ticker: str
    action: str          # 'add' | 'reduce' | 'hold' | 'exit'
    target_weight: float
    confidence: int      # 1–5
    rationale: str
    position_change: str  # 'defended' | 'revised'


@dataclass
class Round2Reply:
    """Parsed and validated Round-2 reply from one outlier persona.

    rebuttal_narrative: how the outlier responded to the counterargument overall.
    stances:            per-ticker restated stances with position_change markers.
    """
    persona: str
    rebuttal_narrative: str
    stances: list[Round2Stance] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class Round2ParseError(ValueError):
    """Raised when an outlier's Round-2 JSON violates the output contract.

    Never caught internally — the caller (orchestrator) decides whether to
    re-prompt within budget or abort.  Silent coercion would mask mandate
    violations (TDD §1.4).
    """


# ---------------------------------------------------------------------------
# Parser — fail-loud (Gate 5)
# ---------------------------------------------------------------------------

def parse_round2_reply(
    raw: str,
    *,
    persona_slug: str,
    week_id: str,
    max_position_weight: float = 0.20,
) -> Round2Reply:
    """Parse one outlier's Round-2 JSON into a Round2Reply.

    ROUND 2 OUTPUT SCHEMA (from _persona_template.md §"ROUND 2 OUTPUT SCHEMA")::

        {
          "round": 2,
          "rebuttal_narrative": "<how you responded to the counterargument>",
          "stances": [
            {
              "ticker": "<T>",
              "action": "<ADD|REDUCE|EXIT|HOLD>",
              "target_weight": 0.0,
              "confidence": 3,
              "rationale": "<strengthened or revised rationale>",
              "position_change": "<defended|revised>"
            }
          ]
        }

    Reuses Component 14's action vocabulary normalization and domain rules.
    Adds validation of the new fields: rebuttal_narrative (non-empty str)
    and position_change per stance (defended | revised).

    Args:
        raw:                  Raw JSON string from the outlier subagent.
        persona_slug:         Persona identifier (used in error messages).
        week_id:              Week label (used when building AgentStancePayload rows).
        max_position_weight:  Upper bound for target_weight; from thresholds.yaml.

    Returns:
        Round2Reply with validated stances and rebuttal_narrative.

    Raises:
        Round2ParseError: On any contract violation.  Never silently coerces.
    """
    stripped = raw.strip()
    # Strip markdown code fences if present (same handling as Component 14).
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        stripped = "\n".join(inner).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise Round2ParseError(
            f"[{persona_slug}] Round-2 output is not valid JSON: {exc}"
        ) from exc

    # Required top-level keys.
    for key in ("rebuttal_narrative", "stances"):
        if key not in data:
            raise Round2ParseError(
                f"[{persona_slug}] Round-2 JSON missing required key {key!r}."
            )

    # rebuttal_narrative must be a non-empty string.
    rebuttal_narrative = str(data["rebuttal_narrative"]).strip()
    if not rebuttal_narrative:
        raise Round2ParseError(
            f"[{persona_slug}] Round-2 JSON 'rebuttal_narrative' is empty. "
            "The outlier must describe how it responded to the counterargument."
        )

    stances_raw = data["stances"]
    if not isinstance(stances_raw, list):
        raise Round2ParseError(
            f"[{persona_slug}] Round-2 'stances' must be a JSON array."
        )
    if not stances_raw:
        raise Round2ParseError(
            f"[{persona_slug}] Round-2 'stances' array is empty. "
            "The outlier must restate at least one stance."
        )

    parsed_stances: list[Round2Stance] = []
    for i, item in enumerate(stances_raw):
        if not isinstance(item, dict):
            raise Round2ParseError(
                f"[{persona_slug}] stances[{i}] is not a JSON object."
            )

        raw_ticker = str(item.get("ticker", "")).strip().upper()
        if not raw_ticker:
            raise Round2ParseError(
                f"[{persona_slug}] stances[{i}] missing 'ticker'."
            )

        # Normalize action — reuse Component 14's map; fail loudly on unknown.
        raw_action = str(item.get("action", "")).strip()
        action = _ACTION_NORMALISE.get(raw_action)
        if action is None:
            raise Round2ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"action={raw_action!r} is not in the 4-value vocabulary "
                f"{sorted(_VALID_ACTIONS)}.  No silent coercion."
            )

        # Validate confidence ∈ 1..5 — same rule as Component 14.
        try:
            confidence = int(item.get("confidence", 0))
        except (TypeError, ValueError) as exc:
            raise Round2ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"confidence must be an integer: {exc}"
            ) from exc
        if confidence not in range(1, 6):
            raise Round2ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"confidence={confidence} is outside 1..5.  No silent coercion."
            )

        # Validate target_weight ∈ [0, max_position_weight] — same rule as Component 14.
        try:
            target_weight = float(item.get("target_weight", -1.0))
        except (TypeError, ValueError) as exc:
            raise Round2ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"target_weight must be a float: {exc}"
            ) from exc
        if target_weight < 0.0 or target_weight > max_position_weight + 1e-9:
            raise Round2ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"target_weight={target_weight:.4f} is outside "
                f"[0, {max_position_weight}].  No silent coercion."
            )

        # Validate position_change ∈ {defended, revised} — Round-2-specific.
        raw_pc = str(item.get("position_change", "")).strip().lower()
        if raw_pc not in _VALID_POSITION_CHANGES:
            raise Round2ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"position_change={raw_pc!r} is not in "
                f"{sorted(_VALID_POSITION_CHANGES)}.  No silent coercion."
            )

        rationale = str(item.get("rationale", "")).strip()

        parsed_stances.append(Round2Stance(
            ticker=raw_ticker,
            action=action,
            target_weight=target_weight,
            confidence=confidence,
            rationale=rationale,
            position_change=raw_pc,
        ))

    logger.info(
        "[%s] Round-2 parsed: %d restated stances; rebuttal_narrative length=%d chars",
        persona_slug,
        len(parsed_stances),
        len(rebuttal_narrative),
    )

    return Round2Reply(
        persona=persona_slug,
        rebuttal_narrative=rebuttal_narrative,
        stances=parsed_stances,
    )


# ---------------------------------------------------------------------------
# Writer — produces round=2 AgentStancePayload rows
# ---------------------------------------------------------------------------

def build_round2_stance_payloads(
    reply: Round2Reply,
    *,
    week_id: str,
    user_id: str = "andrew",
    roster_version: int = 1,
    enhancement_version: int = 1,
) -> list[AgentStancePayload]:
    """Build the round=2 AgentStancePayload rows from a parsed Round2Reply.

    Mirrors the round=1 writer pattern in weekly_run._write_week_transaction:
    one AgentStancePayload per (persona × restated ticker), with round=2 so
    they coexist with the round=1 rows under UNIQUE(week_id, persona, ticker,
    round, user_id) without collision.

    The orchestrator (Component 12) inserts these inside its week transaction —
    this function only constructs the in-memory payload; it does not touch the DB.

    Args:
        reply:               Validated Round2Reply from parse_round2_reply.
        week_id:             ISO week label (e.g. "2026-W24").
        user_id:             Owner field; default "andrew".
        roster_version:      Matches the round=1 rows for this week.
        enhancement_version: Matches the round=1 rows for this week.

    Returns:
        List of AgentStancePayload objects with round=2, one per restated stance.
    """
    payloads: list[AgentStancePayload] = []
    for stance in reply.stances:
        payloads.append(AgentStancePayload(
            week_id=week_id,
            persona=reply.persona,
            ticker=stance.ticker,
            round=2,
            action=stance.action,
            target_weight=stance.target_weight,
            confidence=stance.confidence,
            rationale=stance.rationale,
            user_id=user_id,
            roster_version=roster_version,
            enhancement_version=enhancement_version,
        ))
    return payloads


# ---------------------------------------------------------------------------
# Convenience: parse + build in one call (used by the orchestrator at M3-006)
# ---------------------------------------------------------------------------

def capture_round2_stances(
    raw_round2_replies: dict[str, str],
    *,
    week_id: str,
    config: dict[str, Any] | None = None,
    user_id: str = "andrew",
    roster_version: int = 1,
    enhancement_version: int = 1,
) -> dict[str, tuple[Round2Reply, list[AgentStancePayload]]]:
    """Parse 2 outlier Round-2 replies and produce (Round2Reply, payload list) pairs.

    This is the single entry point the orchestrator (Component 12) calls at
    M3-006 after capturing the 2 live outlier replies.  At M3-003 it is
    exercised via stubbed fixtures in the test suite.

    Args:
        raw_round2_replies: Mapping of persona_slug → raw Round-2 JSON string.
                            Must contain EXACTLY 2 entries (the selected outliers).
        week_id:            ISO week label.
        config:             Optional dict; may carry ``max_position_weight``.
        user_id:            Owner field.
        roster_version:     Matches the round=1 rows for this week.
        enhancement_version: Matches the round=1 rows for this week.

    Returns:
        Dict mapping persona_slug → (Round2Reply, list[AgentStancePayload]).
        The AgentStancePayload list is what the orchestrator inserts as round=2 rows.

    Raises:
        Round2ParseError: If any reply is malformed or out-of-domain.
        ValueError:       If the number of replies is not exactly 2 (cost contract).
    """
    n = len(raw_round2_replies)
    if n != 2:
        raise ValueError(
            f"capture_round2_stances expects exactly 2 outlier replies; got {n}. "
            "More or fewer than 2 Round-2 dispatches is a cost-contract violation "
            "(TDD Component 24, Gate 5 Major tier)."
        )

    cfg = config or {}
    max_position_weight: float = float(cfg.get("max_position_weight", 0.20))

    result: dict[str, tuple[Round2Reply, list[AgentStancePayload]]] = {}
    for slug, raw in raw_round2_replies.items():
        reply = parse_round2_reply(
            raw,
            persona_slug=slug,
            week_id=week_id,
            max_position_weight=max_position_weight,
        )
        payloads = build_round2_stance_payloads(
            reply,
            week_id=week_id,
            user_id=user_id,
            roster_version=roster_version,
            enhancement_version=enhancement_version,
        )
        result[slug] = (reply, payloads)

    return result
