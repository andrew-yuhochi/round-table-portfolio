"""Component 23 — counterargument_assembly (deterministic, zero-dispatch).

Builds each outlier's counterargument block entirely from EXISTING Round-1
rationale text — selection + attribution, no novel generation, no subagent
dispatch of any kind.

Design
------
For each outlier:
1. Identify the tickers where the outlier diverges MOST from the panel, using
   the per-ticker signed scores already computed by Component 21.
2. For each such ticker, collect the opposing personas' Round-1 rationales —
   those whose signed score is on the opposite side of the panel mean from the
   outlier's position.
3. Rank opposing rationales by |their_score - panel_mean| (most opposing first).
4. Assemble into a verbatim-quoted, attributed block bounded by
   ``counterargument_max_rationales`` (from config).

The output is a plain string ready to be embedded verbatim into a Round-2 prompt
(Component 24 — a later task).  No paraphrasing.  No novel sentences.
Every segment in the assembled block is a substring of a real Round-1 rationale.

Failure modes
-------------
- If the outlier and panel AGREE on a ticker (|outlier_score - panel_mean| ≈ 0),
  that ticker is never challenged (``_AGREE_TOL`` guards this).
- If no opposing rationales are found for a ticker (all personas held the same
  direction), that ticker is silently skipped.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(os.environ.get("THRESHOLDS_CONFIG", "config/thresholds.yaml"))

_DEFAULT_MAX_RATIONALES: int = 3
# Fallback when counterargument_agree_tolerance is absent from thresholds.yaml.
# The comparison paths always read from CounterargumentConfig — this constant
# is the documented default only.
_DEFAULT_AGREE_TOL: float = 0.05


@dataclass(frozen=True)
class CounterargumentConfig:
    """Typed view of the counterargument section of thresholds.yaml."""

    counterargument_max_rationales: int
    counterargument_agree_tolerance: float
    # Inherited for signed-score recomputation (must match DissentConfig values).
    action_direction_map: dict[str, float]


def load_counterargument_config(
    config_path: Optional[Path] = None,
) -> CounterargumentConfig:
    """Read thresholds.yaml and return a CounterargumentConfig.

    Falls back to built-in defaults for any missing key.
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
        "action_direction_map",
        {"add": 1.0, "hold": 0.0, "reduce": -0.5, "exit": -1.0},
    )
    action_direction: dict[str, float] = {
        k: float(v) for k, v in action_dir_raw.items()
    }

    return CounterargumentConfig(
        counterargument_max_rationales=int(
            raw.get("counterargument_max_rationales", _DEFAULT_MAX_RATIONALES)
        ),
        counterargument_agree_tolerance=float(
            raw.get("counterargument_agree_tolerance", _DEFAULT_AGREE_TOL)
        ),
        action_direction_map=action_direction,
    )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@dataclass
class CounterargumentBlock:
    """Per-outlier output of Component 23.

    ``outlier_slug``       — the persona being challenged.
    ``debated_tickers``    — tickers targeted (where the outlier most diverges).
    ``block``              — the assembled verbatim text, attributed, bounded by
                             the config length cap.
    ``source_rationales``  — list of (persona, ticker, rationale_text) tuples
                             whose text was used; every sentence in ``block`` is
                             a substring of one of these.
    """

    outlier_slug: str
    debated_tickers: list[str]
    block: str
    source_rationales: list[tuple[str, str, str]]  # (persona, ticker, text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, attr: str) -> Any:
    """Uniform attribute access for both dict and dataclass stance objects."""
    if isinstance(obj, dict):
        return obj[attr]
    return getattr(obj, attr)


def _signed_score(
    action: str, confidence: int, action_direction: dict[str, float]
) -> float:
    """Recompute the signed score.  Mirrors dissent.py._signed_score."""
    direction = action_direction.get(action)
    if direction is None:
        raise ValueError(
            f"Out-of-domain action {action!r} — not in action_direction_map."
        )
    if not (1 <= confidence <= 5):
        raise ValueError(
            f"Out-of-domain confidence {confidence!r} — must be 1–5."
        )
    return direction * (confidence / 5.0)


def _ticker_scores(
    stances: list[Any],
    ticker: str,
    action_direction: dict[str, float],
) -> dict[str, float]:
    """Return {persona: signed_score} for all personas that took a stance on ticker."""
    result: dict[str, float] = {}
    for s in stances:
        if _get(s, "ticker") == ticker:
            persona = _get(s, "persona")
            action = _get(s, "action")
            confidence = _get(s, "confidence")
            result[persona] = _signed_score(action, confidence, action_direction)
    return result


def _panel_mean(scores: dict[str, float], exclude: str) -> float:
    """Mean signed score of all personas EXCEPT the outlier."""
    panel_scores = [v for k, v in scores.items() if k != exclude]
    if not panel_scores:
        return 0.0
    return sum(panel_scores) / len(panel_scores)


# ---------------------------------------------------------------------------
# Component 23 — assemble_counterargument
# ---------------------------------------------------------------------------


def assemble_counterargument(
    outlier_slug: str,
    all_stances: list[Any],
    rationales: dict[str, dict[str, str]],
    config: CounterargumentConfig,
    n_target_tickers: int = 3,
) -> CounterargumentBlock:
    """Assemble a verbatim counterargument block for one outlier.

    Parameters
    ----------
    outlier_slug:
        The persona being challenged (e.g. "growth").
    all_stances:
        ALL Round-1 stances for the week (all 7 personas) — either dicts with
        keys {persona, ticker, action, confidence} or objects with those
        attributes.
    rationales:
        Mapping {persona: {ticker: rationale_text}} for Round 1.  Every entry
        whose text appears in the output block must be present here.
        Tickers with no rationale text are silently skipped.
    config:
        CounterargumentConfig loaded from thresholds.yaml.
    n_target_tickers:
        How many most-divergent tickers to challenge.  Bounded by actual
        divergent tickers found.

    Returns
    -------
    CounterargumentBlock
        ``block`` is composed ONLY from existing rationale text — verbatim
        quotes with attribution.  Every sentence is a substring of a
        source_rationales entry.
    """
    all_tickers: list[str] = list(
        dict.fromkeys(_get(s, "ticker") for s in all_stances)
    )

    # Step 1 — compute per-ticker divergence for the outlier.
    ticker_divergence: list[tuple[str, float, float, float]] = []
    # (ticker, outlier_score, panel_mean, |outlier - panel_mean|)
    for ticker in all_tickers:
        scores = _ticker_scores(all_stances, ticker, config.action_direction_map)
        if outlier_slug not in scores:
            continue
        outlier_score = scores[outlier_slug]
        panel_mean = _panel_mean(scores, outlier_slug)
        divergence = abs(outlier_score - panel_mean)
        if divergence <= config.counterargument_agree_tolerance:
            # Outlier and panel agree here — do not challenge.
            continue
        ticker_divergence.append((ticker, outlier_score, panel_mean, divergence))

    # Sort: most divergent first.
    ticker_divergence.sort(key=lambda x: -x[3])
    target_tickers = [td[0] for td in ticker_divergence[:n_target_tickers]]

    # Step 2 — for each target ticker, collect opposing rationales.
    # "Opposing" means: persona whose score is on the OPPOSITE side of the
    # panel mean from the outlier (or more extreme in the panel direction).
    collected_rationales: list[tuple[str, str, str, float]] = []
    # (persona, ticker, rationale_text, |their_score - panel_mean|) — for ranking

    ticker_meta: dict[str, tuple[float, float]] = {
        td[0]: (td[1], td[2]) for td in ticker_divergence
    }

    for ticker in target_tickers:
        outlier_score, panel_mean = ticker_meta[ticker]
        scores = _ticker_scores(all_stances, ticker, config.action_direction_map)
        outlier_side = math.copysign(1.0, outlier_score - panel_mean)

        for persona, score in scores.items():
            if persona == outlier_slug:
                continue
            persona_dev = score - panel_mean
            # Opposing: this persona is on the other side of the panel mean
            # from the outlier, OR is the most extreme in the opposing direction.
            if math.copysign(1.0, persona_dev) == -outlier_side and abs(persona_dev) > config.counterargument_agree_tolerance:
                rat_text = rationales.get(persona, {}).get(ticker)
                if rat_text:
                    collected_rationales.append(
                        (persona, ticker, rat_text, abs(persona_dev))
                    )

    # Step 3 — rank by opposition strength (highest |persona_dev| first),
    # then deduplicate by (persona, ticker) keeping highest-ranked.
    collected_rationales.sort(key=lambda x: -x[3])
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str, str]] = []
    for persona, ticker, rat_text, _ in collected_rationales:
        key = (persona, ticker)
        if key not in seen:
            seen.add(key)
            deduped.append((persona, ticker, rat_text))

    # Step 4 — apply length cap from config.
    capped = deduped[: config.counterargument_max_rationales]

    # Step 5 — compose attributed block.
    # Format: "[{PERSONA} on {TICKER}]: {rationale_text}"
    # No paraphrasing.  The rationale_text is the verbatim stored text.
    lines: list[str] = []
    for persona, ticker, rat_text in capped:
        lines.append(f"[{persona} on {ticker}]: {rat_text}")

    block = "\n\n".join(lines)

    return CounterargumentBlock(
        outlier_slug=outlier_slug,
        debated_tickers=target_tickers,
        block=block,
        source_rationales=capped,
    )


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def assemble_counterarguments(
    outlier_slugs: list[str],
    all_stances: list[Any],
    rationales: dict[str, dict[str, str]],
    config: CounterargumentConfig,
    n_target_tickers: int = 3,
) -> dict[str, CounterargumentBlock]:
    """Assemble counterargument blocks for all selected outliers.

    Parameters
    ----------
    outlier_slugs:
        Ordered list of persona slugs (from Component 22's
        ``OutlierSelection.selected``).
    all_stances, rationales, config, n_target_tickers:
        Forwarded to ``assemble_counterargument`` for each outlier.

    Returns
    -------
    dict[str, CounterargumentBlock]
        Keyed by persona slug.  Deterministic given fixed stances + rationales
        + config.
    """
    return {
        slug: assemble_counterargument(
            outlier_slug=slug,
            all_stances=all_stances,
            rationales=rationales,
            config=config,
            n_target_tickers=n_target_tickers,
        )
        for slug in outlier_slugs
    }
