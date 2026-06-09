"""Components 21 + 22 — recalibrated dissent metric and outlier selection.

Component 21 — dissent_metric
------------------------------
Replaces the M2 raw-weight σ (DEF-004).  Maps each Round-1 stance to a signed
normalized score ``s ∈ [−1, +1]`` that captures BOTH direction and conviction:

    s = direction(action) × (confidence / 5)

    direction map (from config):
        add    → +1.0
        hold   →  0.0   (no signal — confidence irrelevant)
        reduce → −0.5
        exit   → −1.0

Per-ticker, compute the POPULATION σ of the 7 personas' ``s`` values.
Per-week ``dissent_score`` = mean of those per-ticker σs across the debate set.

Rationale: the theoretical maximum per-ticker σ on this axis is 1.0 (half
personas at ADD@5 = +1.0, half at EXIT@5 = −1.0).  The M2 raw-weight σ
collapsed to ~0 because all personas held identical 0.0 weights on most
tickers while disagreeeing on *which* tickers to weight — the new metric
captures that directional disagreement.

Component 22 — outlier_selection
----------------------------------
Ranks the 7 personas by per-persona divergence (mean |s − panel_mean_s| across
all debate tickers) and selects the top ``n_outliers`` (default 2) for Round 2.

Tie-break (from config): alphabetical ascending by persona slug.  This is
deterministic and stable — same Round-1 stances → same 2 outliers.

All thresholds + the action→direction map are read from config (never literals).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader — mirrors budget/loader.py pattern
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(os.environ.get("THRESHOLDS_CONFIG", "config/thresholds.yaml"))

# Built-in defaults used when config keys are absent (keeps the module
# self-contained for tests that don't provide a full thresholds.yaml).
_DEFAULT_CONTESTED_WEEK_THRESHOLD: float = 0.50
_DEFAULT_N_OUTLIERS: int = 2
_DEFAULT_DIVERGENCE_TIEBREAK: str = "alpha_asc"

# The action→direction map lives here as a fallback; the canonical copy is
# in thresholds.yaml under ``action_direction_map``.
_DEFAULT_ACTION_DIRECTION: dict[str, float] = {
    "add":    +1.0,
    "hold":    0.0,
    "reduce": -0.5,
    "exit":   -1.0,
}


@dataclass(frozen=True)
class DissentConfig:
    """Typed view of the dissent-relevant section of thresholds.yaml."""
    contested_week_threshold: float
    action_direction_map: dict[str, float]
    n_outliers: int
    divergence_tiebreak: str  # "alpha_asc" | "alpha_desc"


def load_dissent_config(config_path: Optional[Path] = None) -> DissentConfig:
    """Read thresholds.yaml and return a DissentConfig.

    Falls back to built-in defaults for any missing key so the module is
    usable in tests that provide a minimal thresholds.yaml.
    """
    path = config_path or _CONFIG_PATH
    raw: dict[str, Any] = {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logger.warning(
            "thresholds.yaml not found at %s — using built-in defaults.", path
        )

    action_dir_raw: dict[str, Any] = raw.get(
        "action_direction_map", _DEFAULT_ACTION_DIRECTION
    )
    # Ensure all four actions are present; fill missing from defaults.
    action_direction: dict[str, float] = {
        k: float(v) for k, v in _DEFAULT_ACTION_DIRECTION.items()
    }
    action_direction.update({k: float(v) for k, v in action_dir_raw.items()})

    return DissentConfig(
        contested_week_threshold=float(
            raw.get("contested_week_threshold", _DEFAULT_CONTESTED_WEEK_THRESHOLD)
        ),
        action_direction_map=action_direction,
        n_outliers=int(raw.get("n_outliers", _DEFAULT_N_OUTLIERS)),
        divergence_tiebreak=str(
            raw.get("divergence_tiebreak", _DEFAULT_DIVERGENCE_TIEBREAK)
        ),
    )


# ---------------------------------------------------------------------------
# Component 21 output shape
# ---------------------------------------------------------------------------


@dataclass
class DissentResult:
    """Full output of Component 21.

    All values are deterministic given fixed Round-1 stances + config.
    ``per_ticker_sigma`` and ``per_persona_divergence`` carry the intermediate
    values that Component 22 and the orchestrator need.
    """
    dissent_score: float           # mean per-ticker σ across debate set
    contested_week: bool           # dissent_score ≥ contested_week_threshold
    per_ticker_sigma: dict[str, float]   # ticker → population σ of signed scores
    per_persona_divergence: dict[str, float]  # persona → mean |s − panel_mean|


# ---------------------------------------------------------------------------
# Component 22 output shape
# ---------------------------------------------------------------------------


@dataclass
class OutlierSelection:
    """Output of Component 22.

    ``selected`` is an ordered list (most divergent first) of exactly
    ``n_outliers`` persona slugs together with their Round-1 stances +
    rationales.  The stances are the ``AgentStancePayload``-compatible dicts
    (or objects with .ticker / .action / .confidence / .target_weight /
    .rationale / .persona attributes) passed in by the caller.
    """
    selected: list[str]                  # ordered by descending divergence
    stances_by_persona: dict[str, list[Any]]  # persona → its Round-1 stances


# ---------------------------------------------------------------------------
# Component 21 — dissent_metric
# ---------------------------------------------------------------------------


def _signed_score(action: str, confidence: int, action_direction: dict[str, float]) -> float:
    """Map one stance to s = direction(action) × (confidence / 5).

    Raises:
        ValueError: if action is not in the direction map (out-of-domain input
                    from Component 14 — fail loudly, per TDD §1.4 / Gate 5).
    """
    if action not in action_direction:
        raise ValueError(
            f"Out-of-domain action {action!r} — not in action_direction_map "
            f"{sorted(action_direction)}. Fix upstream at Component 14."
        )
    direction = action_direction[action]
    # HOLD has direction 0.0; confidence is irrelevant but must still be valid.
    if not (1 <= confidence <= 5):
        raise ValueError(
            f"Out-of-domain confidence {confidence!r} — must be 1–5. "
            "Fix upstream at Component 14."
        )
    return direction * (confidence / 5.0)


def compute_dissent(
    stances: list[Any],
    debate_set: list[str],
    config: DissentConfig,
) -> DissentResult:
    """Compute the recalibrated dissent metric (Component 21).

    Args:
        stances:    Round-1 ``AgentStancePayload``-compatible objects with
                    ``.persona``, ``.ticker``, ``.action``, ``.confidence``
                    attributes (or equivalent dicts with those keys).
        debate_set: Ordered list of tickers to score over.
        config:     DissentConfig loaded from thresholds.yaml.

    Returns:
        DissentResult with dissent_score, contested_week flag, per-ticker σ,
        and per-persona divergence scores.

    Raises:
        ValueError: on out-of-domain action or confidence (Gate 5 / fail-loudly).
    """
    if not debate_set:
        return DissentResult(
            dissent_score=0.0,
            contested_week=False,
            per_ticker_sigma={},
            per_persona_divergence={},
        )

    # Normalize stances to attribute access regardless of dict vs dataclass.
    def _get(obj: Any, attr: str) -> Any:
        if isinstance(obj, dict):
            return obj[attr]
        return getattr(obj, attr)

    # Collect all persona slugs that appear in the stances.
    all_personas: list[str] = list(dict.fromkeys(_get(s, "persona") for s in stances))

    per_ticker_sigma: dict[str, float] = {}
    # Accumulator: persona → list of |s − panel_mean| values, one per ticker.
    per_persona_abs_dev: dict[str, list[float]] = {p: [] for p in all_personas}

    for ticker in debate_set:
        ticker_stances = [s for s in stances if _get(s, "ticker") == ticker]
        if len(ticker_stances) < 2:
            per_ticker_sigma[ticker] = 0.0
            # For personas that did take a stance, contribute a 0 deviation.
            for s in ticker_stances:
                per_persona_abs_dev[_get(s, "persona")].append(0.0)
            continue

        # Compute signed scores for all personas on this ticker.
        scores: list[tuple[str, float]] = []
        for s in ticker_stances:
            persona = _get(s, "persona")
            action = _get(s, "action")
            confidence = _get(s, "confidence")
            sc = _signed_score(action, confidence, config.action_direction_map)
            scores.append((persona, sc))

        # Population σ of s values for this ticker.
        s_vals = [sc for _, sc in scores]
        mean_s = sum(s_vals) / len(s_vals)
        variance = sum((sc - mean_s) ** 2 for sc in s_vals) / len(s_vals)
        per_ticker_sigma[ticker] = math.sqrt(variance)

        # Per-persona: accumulate |s − panel_mean| for divergence.
        for persona, sc in scores:
            per_persona_abs_dev[persona].append(abs(sc - mean_s))

    # Per-week dissent_score = mean of per-ticker σs.
    dissent_score = (
        sum(per_ticker_sigma.values()) / len(per_ticker_sigma)
        if per_ticker_sigma
        else 0.0
    )
    contested_week = dissent_score >= config.contested_week_threshold

    # Per-persona divergence = mean absolute distance across all debate tickers.
    per_persona_divergence: dict[str, float] = {
        persona: (sum(devs) / len(devs) if devs else 0.0)
        for persona, devs in per_persona_abs_dev.items()
    }

    return DissentResult(
        dissent_score=dissent_score,
        contested_week=contested_week,
        per_ticker_sigma=per_ticker_sigma,
        per_persona_divergence=per_persona_divergence,
    )


# ---------------------------------------------------------------------------
# Component 22 — outlier_selection
# ---------------------------------------------------------------------------


def select_outliers(
    dissent_result: DissentResult,
    stances: list[Any],
    config: DissentConfig,
) -> OutlierSelection:
    """Select the top ``n_outliers`` most-divergent personas for Round 2.

    Tie-break rule (from config, default "alpha_asc"): when two personas have
    identical divergence scores, the one that sorts first alphabetically
    (ascending) is ranked higher.  This guarantees a total order: same
    Round-1 stances → same 2 outliers, every time.

    Args:
        dissent_result: Output of ``compute_dissent`` for this week.
        stances:        The same Round-1 stances passed to ``compute_dissent``.
                        Each selected persona's full stance list is carried
                        forward for Components 23/24.
        config:         DissentConfig (provides n_outliers + divergence_tiebreak).

    Returns:
        OutlierSelection with ``selected`` (ordered list, most divergent first)
        and ``stances_by_persona`` mapping for the selected personas.

    Raises:
        RuntimeError: if fewer personas ran than n_outliers (at PoC the run
                      aborts on any persona failure, TDD §1.5 — surface loudly).
    """
    n = config.n_outliers
    divergences = dissent_result.per_persona_divergence

    if len(divergences) < n:
        raise RuntimeError(
            f"Only {len(divergences)} persona(s) have divergence scores but "
            f"n_outliers={n}. At PoC, all {n} personas must complete Round 1 "
            "before Round 2 can proceed (TDD §1.5 abort rule)."
        )

    # Tie-break: secondary sort key.
    reverse_secondary = config.divergence_tiebreak == "alpha_desc"
    # Sort: primary = descending divergence; secondary = alphabetical (configurable).
    sorted_personas = sorted(
        divergences.keys(),
        key=lambda p: (-divergences[p], p if not reverse_secondary else ""),
        reverse=False,
    )
    # For "alpha_desc" the secondary sort is reversed; re-sort properly.
    if reverse_secondary:
        sorted_personas = sorted(
            divergences.keys(),
            key=lambda p: (-divergences[p], [-ord(c) for c in p]),
        )

    selected = sorted_personas[:n]

    # Normalize stances to attribute access.
    def _get(obj: Any, attr: str) -> Any:
        if isinstance(obj, dict):
            return obj[attr]
        return getattr(obj, attr)

    stances_by_persona: dict[str, list[Any]] = {
        persona: [s for s in stances if _get(s, "persona") == persona]
        for persona in selected
    }

    return OutlierSelection(
        selected=selected,
        stances_by_persona=stances_by_persona,
    )
