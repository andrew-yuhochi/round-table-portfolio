"""Component 27 — memory_briefing_builder.

Renders each persona's windowed memory into a self-contained, size-bounded
markdown briefing block that the SESSION injects into that persona's research /
Round-1 / Round-2 dispatch prompt.

Design mirrors the M3 counterargument-block seam (Component 23): Python BUILDS
the block; the session INJECTS it.  This module never dispatches a subagent.

Critical invariants
-------------------
1. **Zero cross-persona leakage.** Every briefing is scoped to exactly ONE
   persona's own ``WindowedMemory``.  The caller must supply per-persona inputs;
   this module never receives all-persona data and never scans across personas.
   Breaching this contaminates commit-before-reveal independence (Critical
   Component #1) — it is a thesis-level correctness failure, not a minor gap.

2. **Size-bounded.** The recency window (Component 26) already bounds entry
   count.  This builder additionally truncates any narrative body that would push
   the finished block over ``memory_briefing_max_chars`` (from thresholds.yaml).
   Truncation is NOTED inline (an appended ``[truncated]`` marker) so the session
   and the audit trail both know content was cut.

3. **Persisted.** The block is written to
   ``state/runs/<week>-memory/<persona>.md`` as the durable record of EXACTLY
   what each persona was shown this week.  The session reads these files to inject
   them; they are also the audit trail for "what did this persona know."
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml

from round_table_portfolio.orchestrator.digest import DigestConfig, ResolvedRow
from round_table_portfolio.orchestrator.memory_reader import WindowedMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(os.environ.get("THRESHOLDS_CONFIG", "config/thresholds.yaml"))

# Default size budget in characters for one persona's briefing block.
# Rationale: a typical persona prompt is ~2 000–4 000 chars; 3 000 chars of
# memory context leaves ≈50–60 % of a 6 000-char context headroom for the
# round-specific instruction and debate-set material (Critical Component #3).
# Set conservatively so even a full 8-week window never overflows the window.
_DEFAULT_BRIEFING_MAX_CHARS: int = 3_000

# Per-section narrative truncation budget: if the entire block would exceed the
# ceiling, individual entry bodies are truncated to this length first.
# 280 chars ≈ 2–3 short sentences — enough to convey the stance, not a full essay.
_DEFAULT_ENTRY_TRUNCATE_AT: int = 280
_TRUNCATION_MARKER = " [truncated]"


@dataclass(frozen=True)
class BriefingConfig:
    """Typed view of the briefing section of thresholds.yaml."""

    memory_briefing_max_chars: int
    own_misses_in_digest: bool  # re-read here so the callout respects the same flag


def load_briefing_config(
    config_path: Optional[Path] = None,
) -> BriefingConfig:
    """Read thresholds.yaml and return a BriefingConfig.

    Falls back to built-in defaults for any missing key so early-week runs
    before the config key is present work without error.
    """
    path = config_path or _CONFIG_PATH
    raw: dict = {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logger.warning(
            "thresholds.yaml not found at %s — using built-in defaults.", path
        )

    return BriefingConfig(
        memory_briefing_max_chars=int(
            raw.get("memory_briefing_max_chars", _DEFAULT_BRIEFING_MAX_CHARS)
        ),
        own_misses_in_digest=bool(
            raw.get("own_misses_in_digest", True)
        ),
    )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@dataclass
class BriefingResult:
    """Output of ``build_persona_briefing`` for one persona.

    ``persona``         — persona slug (e.g. "value").
    ``briefing_text``   — the rendered markdown block, ready for session injection.
    ``truncated``       — True when at least one entry body was truncated to fit
                          under ``memory_briefing_max_chars``.
    ``output_path``     — the path where the briefing was persisted
                          (``state/runs/<week>-memory/<persona>.md``), or None
                          when ``persist=False``.
    """

    persona: str
    briefing_text: str
    truncated: bool
    output_path: Optional[Path]


# ---------------------------------------------------------------------------
# Internal render helpers
# ---------------------------------------------------------------------------


def _truncate_body(body: str, limit: int) -> tuple[str, bool]:
    """Truncate *body* to *limit* chars if needed.

    Returns (possibly-truncated body, was_truncated).  The ``[truncated]``
    marker is appended to the body so every reader (session + audit log) knows
    content was cut.
    """
    if len(body) <= limit:
        return body, False
    return body[:limit] + _TRUNCATION_MARKER, True


def _render_section(
    section_title: str,
    entries: Sequence[tuple[str, str]],
    entry_truncate_at: int,
) -> tuple[str, bool]:
    """Render one memory section to markdown.

    Returns (rendered_text, any_truncated).
    """
    if not entries:
        return f"### {section_title}\n_(no entries in window)_\n", False

    lines: list[str] = [f"### {section_title}"]
    any_truncated = False
    for week_id, body in entries:
        truncated_body, was_truncated = _truncate_body(body.strip(), entry_truncate_at)
        if was_truncated:
            any_truncated = True
        lines.append(f"\n**{week_id}**\n{truncated_body}")
    return "\n".join(lines) + "\n", any_truncated


def _build_own_misses_callout(
    resolved_rows: Sequence[ResolvedRow],
    persona: str,
) -> str:
    """Build the own-misses callout from resolved-negative-alpha rows.

    Lists only calls where alpha < 0 (resolved below SPY) for the given persona.
    Returns an empty string when there are no such calls.
    """
    misses = [r for r in resolved_rows if r.persona == persona and r.alpha < 0.0]
    if not misses:
        return ""

    # Sort by alpha ascending (worst miss first), tiebreak ticker ascending.
    misses.sort(key=lambda r: (r.alpha, r.ticker))

    lines = ["### Your Past Calls That Resolved Below SPY"]
    for row in misses:
        lines.append(
            f"  - {row.ticker}: {row.action} in {row.call_week_id} "
            f"→ alpha {row.alpha:+.4f} vs SPY"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public entry point — single-persona
# ---------------------------------------------------------------------------


def build_persona_briefing(
    windowed_memory: WindowedMemory,
    whats_new_digest: str,
    resolved_rows: Sequence[ResolvedRow],
    *,
    week_id: str,
    config: Optional[BriefingConfig] = None,
    runs_dir: Path = Path("state/runs"),
    persist: bool = True,
) -> BriefingResult:
    """Build and persist the memory briefing block for one persona.

    Args:
        windowed_memory:  The recency-windowed four-section memory from Component 26.
                          MUST be scoped to the target persona — the caller guarantees
                          this; the function enforces it by reading only
                          ``windowed_memory.persona``.
        whats_new_digest: The "what's new" digest text from Component 28.
        resolved_rows:    Raw ResolvedRow sequence from Component 26 (used for the
                          own-misses callout).  Only rows whose ``persona`` field
                          matches ``windowed_memory.persona`` are used — any stray
                          rows from other personas are silently ignored (defence-in-
                          depth for the leakage invariant).
        week_id:          ISO week string for this run (e.g. "2026-W24").
                          Used to name the output directory.
        config:           BriefingConfig.  Loaded from thresholds.yaml if None.
        runs_dir:         Root of the state/runs tree (default ``state/runs``).
        persist:          When True (default) write the briefing to
                          ``<runs_dir>/<week_id>-memory/<persona>.md``.

    Returns:
        BriefingResult with the rendered text, truncation flag, and output path.
    """
    cfg = config or load_briefing_config()
    persona = windowed_memory.persona

    # Defence-in-depth: filter resolved_rows to own-persona only before any use.
    # This is the per-persona scoping guard — do NOT filter downstream.
    own_resolved = [r for r in resolved_rows if r.persona == persona]

    # Entry-level truncation budget: target each entry body to at most 1/6 of
    # the overall budget.  With 4 sections × up to 8 entries each that is 32
    # potential bodies; 1/6 of 3 000 = 500 chars per body.  We cap at
    # _DEFAULT_ENTRY_TRUNCATE_AT (280) as the hard floor so tiny overall budgets
    # don't produce nonsense.
    entry_budget = max(
        _DEFAULT_ENTRY_TRUNCATE_AT,
        cfg.memory_briefing_max_chars // 6,
    )

    any_truncated = False

    # --- Header ---
    header = (
        f"# Memory Briefing — {persona}\n"
        f"_Week: {week_id} | This briefing contains only your own memory._\n\n"
    )

    # --- Four memory sections ---
    past_calls_text, t1 = _render_section(
        "Past Calls Log", windowed_memory.past_calls, entry_budget
    )
    counterfactual_text, t2 = _render_section(
        "Counterfactual Portfolio Log", windowed_memory.counterfactual, entry_budget
    )
    debate_stances_text, t3 = _render_section(
        "Debate Stances Log", windowed_memory.debate_stances, entry_budget
    )
    whats_new_text, t4 = _render_section(
        "What's New Digest", windowed_memory.whats_new, entry_budget
    )
    any_truncated = any_truncated or t1 or t2 or t3 or t4

    # --- Fresh digest ---
    digest_section = f"### Latest Digest\n{whats_new_digest}\n"

    # --- Own-misses callout (only when config flag is on) ---
    own_misses_section = ""
    if cfg.own_misses_in_digest:
        own_misses_section = _build_own_misses_callout(own_resolved, persona)

    # --- Assemble ---
    parts = [
        header,
        past_calls_text,
        counterfactual_text,
        debate_stances_text,
        whats_new_text,
        digest_section,
    ]
    if own_misses_section:
        parts.append(own_misses_section)

    briefing = "\n".join(parts)

    # --- Global size gate: if still over budget, truncate the whole block ---
    if len(briefing) > cfg.memory_briefing_max_chars:
        briefing = briefing[: cfg.memory_briefing_max_chars] + _TRUNCATION_MARKER
        any_truncated = True
        logger.warning(
            "briefing_builder: persona=%s briefing exceeded %d chars — hard-truncated.",
            persona,
            cfg.memory_briefing_max_chars,
        )

    # --- Persist ---
    output_path: Optional[Path] = None
    if persist:
        out_dir = runs_dir / f"{week_id}-memory"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{persona}.md"
        _atomic_write(output_path, briefing)
        logger.debug(
            "briefing_builder: persisted persona=%s week=%s path=%s chars=%d truncated=%s",
            persona,
            week_id,
            output_path,
            len(briefing),
            any_truncated,
        )

    return BriefingResult(
        persona=persona,
        briefing_text=briefing,
        truncated=any_truncated,
        output_path=output_path,
    )


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* atomically via tmpfile + rename.

    Mirrors the atomic-write contract from memory.py (Component 18) — a
    mid-write failure leaves the prior file intact.
    """
    dir_ = path.parent
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        # Clean up the orphaned tmpfile; re-raise so the caller sees the failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Batch entry point — all personas
# ---------------------------------------------------------------------------


def build_all_briefings(
    per_persona_inputs: dict[
        str,
        tuple[WindowedMemory, str, Sequence[ResolvedRow]],
    ],
    *,
    week_id: str,
    config: Optional[BriefingConfig] = None,
    runs_dir: Path = Path("state/runs"),
    persist: bool = True,
) -> dict[str, BriefingResult]:
    """Build briefings for all personas.

    Args:
        per_persona_inputs: Mapping from persona slug to a 3-tuple:
            (windowed_memory, whats_new_digest, resolved_rows).
            Each windowed_memory.persona must equal the key — callers should
            use the ``read_all_personas_memory`` output to build this dict.
        week_id, config, runs_dir, persist: Forwarded to ``build_persona_briefing``.

    Returns:
        Dict mapping persona slug → BriefingResult.
    """
    cfg = config or load_briefing_config()
    results: dict[str, BriefingResult] = {}
    for persona, (windowed, digest_text, resolved) in per_persona_inputs.items():
        results[persona] = build_persona_briefing(
            windowed,
            digest_text,
            resolved,
            week_id=week_id,
            config=cfg,
            runs_dir=runs_dir,
            persist=persist,
        )
    return results
