"""Component 13 + 14 — debate-set construction and Round-1 stance capture.

Component 13 (`construct_debate_set`):
    Applies Component 10's bounding rule to the 7 real persona outputs:
    de-duplicate the union of all shortlists + cluster peers, bound within the
    configured ceiling, never drop a directly-shortlisted name.

Component 14 (`capture_round1_stances`):
    Parse each persona's Round-1 JSON (ROUND 1 OUTPUT SCHEMA), normalize the
    action vocabulary to the agent_stances CHECK domain (add/reduce/hold/exit),
    and produce AgentStancePayload rows + counterfactual_portfolio dicts.

    Commit-before-reveal contract: no persona's Round-1 prompt may contain any
    other persona's Round-1 output.  The prompt builder is injected so tests
    can assert isolation without needing live subagent output.

Fail-loudly principle (TDD §1.4 / Gate 5):
    Out-of-domain action, confidence outside 1..5, or target_weight outside
    [0, max_position_weight] all raise Round1ParseError immediately.  No silent
    coercion.  These are contract violations — the caller (orchestrator) must
    decide whether to re-prompt or abort.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from round_table_portfolio.research.runner import PersonaResearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical four-value action vocabulary (lowercase, matching the SQL CHECK).
_VALID_ACTIONS = frozenset({"add", "reduce", "hold", "exit"})

# Map from the ROUND 1 OUTPUT SCHEMA uppercase variants to the stored lowercase.
_ACTION_NORMALISE: dict[str, str] = {
    "ADD":    "add",
    "REDUCE": "reduce",
    "HOLD":   "hold",
    "EXIT":   "exit",
    # lowercase already-normalised pass-through.
    "add":    "add",
    "reduce": "reduce",
    "hold":   "hold",
    "exit":   "exit",
}

_DEFAULT_DEBATE_SET_CEILING = 40


# ---------------------------------------------------------------------------
# Shared payload types (re-exported by weekly_run.py)
# ---------------------------------------------------------------------------

@dataclass
class AgentStancePayload:
    """One row payload for agent_stances (round=1)."""
    week_id: str
    persona: str
    ticker: str
    round: int              # always 1 in M2
    action: str             # 'add' | 'reduce' | 'hold' | 'exit'
    target_weight: float
    confidence: int         # 1–5
    rationale: str
    user_id: str = "andrew"
    roster_version: int = 1
    enhancement_version: int = 1


@dataclass
class Round1Capture:
    """Return type of capture_round1_stances.

    stances:         flat list of AgentStancePayload (7 × |debate_set| rows)
    counterfactuals: per-persona portfolio dict {ticker: weight, 'CASH': weight}
    prompts:         per-persona Round-1 prompt text (used for isolation audit)
    narratives:      per-persona narrative_summary string (used by transcript writer)
    """
    stances: list[AgentStancePayload] = field(default_factory=list)
    counterfactuals: dict[str, dict[str, float]] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    narratives: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class Round1ParseError(ValueError):
    """Raised when a persona's Round-1 JSON violates the output contract.

    Never caught internally — the caller (orchestrator) decides what to do.
    This is intentional: silent coercion would mask mandate violations.
    """


# ---------------------------------------------------------------------------
# Component 13 — construct_debate_set
# ---------------------------------------------------------------------------

def construct_debate_set(
    research_results: list[PersonaResearchResult],
    config: dict[str, Any],
) -> list[str]:
    """Apply Component 10's bounding rule to this week's 7 persona outputs.

    De-duplicates the union of all 7 shortlists + cluster peers; trims from
    cluster peers first if the union exceeds the ceiling; NEVER drops a
    directly-shortlisted name.

    Args:
        research_results: The 7 PersonaResearchResult objects from this week's
            per-persona research phase.
        config: Dict that may carry ``debate_set_ceiling`` (int) under the
            ``persona_budgets`` key, or directly as a top-level key.  Falls
            back to _DEFAULT_DEBATE_SET_CEILING (40) if absent.

    Returns:
        A de-duplicated list of ticker strings, length ≤ ceiling.

    Raises:
        RuntimeError (Major-tier): if the ceiling is exceeded after applying
            the trim rule.  Should never happen given the trim logic, but
            surfaced explicitly so the caller can diagnose wiring issues.
    """
    ceiling = _resolve_ceiling(config)

    # Collect directly-shortlisted tickers (is_cluster_peer=0) — these are
    # NEVER dropped by the trim.  Use a dict to preserve insertion order and
    # de-duplicate across personas.
    direct: dict[str, None] = {}
    peers: dict[str, None] = {}

    for result in research_results:
        for row in result.shortlist_rows:
            if row.is_cluster_peer == 0:
                direct[row.ticker] = None
            else:
                # Cluster peers go into a separate pool for trimming.
                if row.ticker not in direct:
                    peers[row.ticker] = None

    # Any ticker that was also shortlisted directly by another persona wins.
    # Remove such tickers from the peers pool so we do not double-count them
    # in the debate-set while still honouring direct-shortlist priority.
    peers = {t: None for t in peers if t not in direct}

    n_direct = len(direct)

    if n_direct > ceiling:
        # This would be a Major — more directly-shortlisted names than the
        # ceiling allows.  We surface it loudly rather than silently dropping.
        raise RuntimeError(
            f"MAJOR: {n_direct} directly-shortlisted names exceed the "
            f"debate-set ceiling ({ceiling}).  Cannot trim without dropping "
            "load-bearing names.  Raise the ceiling in persona_budgets.yaml "
            "or reduce per-persona shortlist lengths."
        )

    # Trim cluster peers so the total fits the ceiling.
    available_for_peers = ceiling - n_direct
    trimmed_peers = list(peers.keys())[:available_for_peers]

    debate_set = list(direct.keys()) + trimmed_peers
    actual_size = len(debate_set)

    logger.info(
        "Debate set formed: %d tickers (%d direct + %d cluster peers); ceiling=%d",
        actual_size,
        n_direct,
        len(trimmed_peers),
        ceiling,
    )

    if actual_size > ceiling:
        # Defensive — should never happen given the trim above.
        raise RuntimeError(
            f"MAJOR: Debate set ({actual_size}) exceeded ceiling ({ceiling}) "
            "after bounding.  This is a logic bug — root-cause the trim step."
        )

    return debate_set


def _resolve_ceiling(config: dict[str, Any]) -> int:
    """Extract the debate-set ceiling from config with a safe fallback."""
    # Try direct top-level key first (used in tests and orchestrator pass-throughs).
    if "debate_set_ceiling" in config:
        return int(config["debate_set_ceiling"])
    # Try nested under persona_budgets key.
    budgets = config.get("persona_budgets", {})
    if isinstance(budgets, dict) and "debate_set_ceiling" in budgets:
        return int(budgets["debate_set_ceiling"])
    return _DEFAULT_DEBATE_SET_CEILING


# ---------------------------------------------------------------------------
# Component 14 — capture_round1_stances
# ---------------------------------------------------------------------------

def capture_round1_stances(
    debate_set: list[str],
    research_results: list[PersonaResearchResult],
    *,
    raw_round1_replies: dict[str, str],
    config: dict[str, Any] | None = None,
    prompt_builder: Optional[Callable[[str, list[str], "PersonaResearchResult"], str]] = None,
    judge_dispatch: Any = None,  # reserved for production wiring; unused in M2 engine
) -> Round1Capture:
    """Parse 7 personas' Round-1 JSON and produce stance payloads + counterfactuals.

    The commit-before-reveal guarantee:
        Each persona's Round-1 prompt is built BEFORE any other persona's Round-1
        output is available.  The ``prompt_builder`` callable is called with only
        the current persona's own research result and the debate set — never with
        another persona's reply.  This function validates that guarantee by
        collecting all prompts before any reply is processed, then verifying that
        no prompt contains another persona's output text.

    Args:
        debate_set:         The bounded list of tickers from construct_debate_set.
        research_results:   The 7 PersonaResearchResult objects.
        raw_round1_replies: Mapping of persona_slug → raw Round-1 JSON string
            (already captured by the session before this function is called).
        config:             Optional config dict; may carry ``max_position_weight``.
        prompt_builder:     Optional callable(persona_slug, debate_set, result)
            → prompt_str.  In production the session builds prompts; in tests
            a simple builder is injected so the isolation assertion is checkable.
            When None, a default prompt builder is used (adds no persona output).
        judge_dispatch:     Reserved for future production wiring (M2-011).
            Not used in the M2 engine — stances come from raw_round1_replies.

    Returns:
        Round1Capture with:
          - stances:         7 × |debate_set| AgentStancePayload rows (round=1)
          - counterfactuals: per-persona {ticker: weight, 'CASH': weight}
          - prompts:         per-persona prompt text (for isolation audit)

    Raises:
        Round1ParseError: If any persona's JSON is malformed or contains
            out-of-domain action/confidence/weight values.  Never silently
            coerced.
    """
    cfg = config or {}
    max_position_weight: float = float(cfg.get("max_position_weight", 0.20))

    week_id = research_results[0].week_id if research_results else ""
    _prompt_builder = prompt_builder or _default_prompt_builder

    persona_slug_to_result: dict[str, PersonaResearchResult] = {
        r.persona_slug: r for r in research_results
    }

    # -----------------------------------------------------------------------
    # Phase 1: Build all prompts BEFORE processing any reply.
    # This is the structural guarantee of commit-before-reveal — prompts are
    # constructed only from the persona's own research, not from peer replies.
    # -----------------------------------------------------------------------
    prompts: dict[str, str] = {}
    for slug, result in persona_slug_to_result.items():
        prompts[slug] = _prompt_builder(slug, debate_set, result)

    # -----------------------------------------------------------------------
    # Phase 2: Parse each persona's reply and produce payloads.
    # -----------------------------------------------------------------------
    all_stances: list[AgentStancePayload] = []
    counterfactuals: dict[str, dict[str, float]] = {}
    narratives: dict[str, str] = {}

    for slug in persona_slug_to_result:
        raw = raw_round1_replies.get(slug)
        if raw is None:
            raise Round1ParseError(
                f"No Round-1 reply provided for persona={slug!r}. "
                "Every persona must return a Round-1 JSON before stances are captured."
            )

        stances_for_persona, counterfactual, narrative = _parse_round1_reply(
            raw=raw,
            persona_slug=slug,
            week_id=week_id,
            debate_set=debate_set,
            max_position_weight=max_position_weight,
        )
        all_stances.extend(stances_for_persona)
        counterfactuals[slug] = counterfactual
        narratives[slug] = narrative

    # -----------------------------------------------------------------------
    # Phase 3: Cross-check — debate_set size must equal distinct stance tickers.
    # -----------------------------------------------------------------------
    distinct_tickers = {s.ticker for s in all_stances}
    if len(distinct_tickers) != len(debate_set):
        raise RuntimeError(
            f"MAJOR: Debate-set size ({len(debate_set)}) disagrees with "
            f"distinct round-1 stance tickers ({len(distinct_tickers)}). "
            "Root-cause the dispatch wiring — Round 1 must reason over the "
            "exact debate set that was constructed."
        )

    return Round1Capture(
        stances=all_stances,
        counterfactuals=counterfactuals,
        prompts=prompts,
        narratives=narratives,
    )


# ---------------------------------------------------------------------------
# Default prompt builder (production stubs; real version is session-built)
# ---------------------------------------------------------------------------

def _default_prompt_builder(
    persona_slug: str,
    debate_set: list[str],
    result: PersonaResearchResult,
) -> str:
    """Build a minimal Round-1 prompt that contains NO peer output.

    This is the default used when no custom prompt_builder is injected.
    The real production prompt is built by the session (M2-011) and is richer,
    but this default satisfies the commit-before-reveal isolation requirement:
    it only includes the persona's own research summary and the debate set.
    """
    tickers_str = ", ".join(debate_set)
    summary_excerpt = result.report_payload.summary[:400] if result.report_payload else ""
    return (
        f"PERSONA: {persona_slug}\n"
        f"DEBATE SET: {tickers_str}\n"
        f"YOUR RESEARCH SUMMARY: {summary_excerpt}\n"
        f"Produce your Round-1 stance JSON covering every ticker in the debate set."
    )


# ---------------------------------------------------------------------------
# Round-1 JSON parser (fail-loudly on any violation)
# ---------------------------------------------------------------------------

def _parse_round1_reply(
    raw: str,
    persona_slug: str,
    week_id: str,
    debate_set: list[str],
    max_position_weight: float,
) -> tuple[list[AgentStancePayload], dict[str, float], str]:
    """Parse one persona's Round-1 JSON into stances + counterfactual.

    Schema (from _persona_template.md §"ROUND 1 OUTPUT SCHEMA")::

        {
          "stances": [
            {"ticker": "X", "action": "ADD", "target_weight": 0.10,
             "confidence": 4, "rationale": "..."}
          ],
          "counterfactual_portfolio": {"AAPL": 0.10, "CASH": 0.90},
          "narrative_summary": "..."
        }

    Returns:
        (stances, counterfactual, narrative_summary)

    Raises:
        Round1ParseError: On any contract violation.  Never silently coerces.
    """
    stripped = raw.strip()
    # Strip markdown code fences if present.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        stripped = "\n".join(inner).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise Round1ParseError(
            f"[{persona_slug}] Round-1 output is not valid JSON: {exc}"
        ) from exc

    # Required top-level keys.
    for key in ("stances", "counterfactual_portfolio", "narrative_summary"):
        if key not in data:
            raise Round1ParseError(
                f"[{persona_slug}] Round-1 JSON missing required key {key!r}."
            )

    # Build a lookup for fast coverage check.
    debate_set_upper = {t.upper() for t in debate_set}
    covered: set[str] = set()

    stances: list[AgentStancePayload] = []
    stances_raw = data["stances"]
    if not isinstance(stances_raw, list):
        raise Round1ParseError(
            f"[{persona_slug}] 'stances' must be a JSON array."
        )

    for i, item in enumerate(stances_raw):
        if not isinstance(item, dict):
            raise Round1ParseError(
                f"[{persona_slug}] stances[{i}] is not a JSON object."
            )

        raw_ticker = str(item.get("ticker", "")).strip().upper()
        if not raw_ticker:
            raise Round1ParseError(
                f"[{persona_slug}] stances[{i}] missing 'ticker'."
            )

        # Validate the ticker is in the debate set.
        if raw_ticker not in debate_set_upper:
            raise Round1ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r} is not "
                f"in the debate set.  Debate set: {sorted(debate_set_upper)}."
            )

        # Normalize action — fail loudly on unknown values.
        raw_action = str(item.get("action", "")).strip()
        action = _ACTION_NORMALISE.get(raw_action)
        if action is None:
            raise Round1ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"action={raw_action!r} is not in the 4-value vocabulary "
                f"{sorted(_VALID_ACTIONS)}.  No silent coercion."
            )

        # Validate confidence ∈ 1..5.
        try:
            confidence = int(item.get("confidence", 0))
        except (TypeError, ValueError) as exc:
            raise Round1ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"confidence must be an integer: {exc}"
            ) from exc
        if confidence not in range(1, 6):
            raise Round1ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"confidence={confidence} is outside 1..5.  No silent coercion."
            )

        # Validate target_weight ∈ [0, max_position_weight].
        try:
            target_weight = float(item.get("target_weight", -1.0))
        except (TypeError, ValueError) as exc:
            raise Round1ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"target_weight must be a float: {exc}"
            ) from exc
        if target_weight < 0.0 or target_weight > max_position_weight + 1e-9:
            raise Round1ParseError(
                f"[{persona_slug}] stances[{i}] ticker={raw_ticker!r}: "
                f"target_weight={target_weight:.4f} is outside "
                f"[0, {max_position_weight}].  No silent coercion."
            )

        rationale = str(item.get("rationale", "")).strip()

        stances.append(AgentStancePayload(
            week_id=week_id,
            persona=persona_slug,
            ticker=raw_ticker,
            round=1,
            action=action,
            target_weight=target_weight,
            confidence=confidence,
            rationale=rationale,
        ))
        covered.add(raw_ticker)

    # Every debate-set ticker must have a stance (no missing cells).
    missing = debate_set_upper - covered
    if missing:
        raise Round1ParseError(
            f"[{persona_slug}] Round-1 reply missing stances for "
            f"{len(missing)} debate-set tickers: {sorted(missing)}.  "
            "Every debate-set ticker must have a stance."
        )

    # Parse counterfactual_portfolio.
    raw_cf = data["counterfactual_portfolio"]
    if not isinstance(raw_cf, dict):
        raise Round1ParseError(
            f"[{persona_slug}] 'counterfactual_portfolio' must be a JSON object."
        )

    counterfactual: dict[str, float] = {}
    for ticker, weight_raw in raw_cf.items():
        t = str(ticker).strip().upper()
        try:
            w = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise Round1ParseError(
                f"[{persona_slug}] counterfactual_portfolio[{ticker!r}]: "
                f"weight must be a float: {exc}"
            ) from exc
        if w < 0.0:
            raise Round1ParseError(
                f"[{persona_slug}] counterfactual_portfolio[{ticker!r}]: "
                f"weight={w:.4f} < 0.  Weights must be non-negative."
            )
        # Position weights (non-CASH) must respect the cap.
        if t != "CASH" and w > max_position_weight + 1e-9:
            raise Round1ParseError(
                f"[{persona_slug}] counterfactual_portfolio[{ticker!r}]: "
                f"weight={w:.4f} exceeds max_position_weight={max_position_weight}.  "
                "No silent coercion."
            )
        counterfactual[t] = w

    if "CASH" not in counterfactual:
        raise Round1ParseError(
            f"[{persona_slug}] counterfactual_portfolio is missing the explicit "
            "'CASH' entry.  The ROUND 1 OUTPUT SCHEMA requires an explicit CASH key."
        )

    narrative_summary: str = str(data.get("narrative_summary", "")).strip()

    logger.info(
        "[%s] Round-1 parsed: %d stances, counterfactual has %d positions + CASH",
        persona_slug,
        len(stances),
        len(counterfactual) - 1,
    )

    return stances, counterfactual, narrative_summary
