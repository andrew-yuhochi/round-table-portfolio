"""Component 17 — Round-1 transcript persistence.

Public entry point::

    from round_table_portfolio.orchestrator.transcript import write_round1_transcript

    transcript_path = write_round1_transcript(
        round1_capture,
        consensus_weights,
        std_devs,
        decision_type,
        week_id="2026-W23",
        state_root=Path("state"),
    )

FILE-FIRST CONTRACT
-------------------
The markdown file is written BEFORE the caller writes the ``transcripts`` DB row.
This guarantees the ``full_log_path NOT NULL`` pointer always resolves to an
existing file — a dangling pointer would break the Debate Archive (M5).

ATOMIC WRITE
------------
The file is written to a temp path (``<target>.tmp``) first, then renamed
atomically via ``os.rename``.  A failure between write and rename leaves any
prior transcript at the target path intact — no partial file at the target.

M3-APPEND CONTRACT
------------------
The markdown structure uses a clear ``## Round 1`` heading so M3 can append a
``## Round 2`` section without reformatting.  The M2 scope boundary is enforced
by the absence of any Round-2 content — asserted in the test suite.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from round_table_portfolio.orchestrator.round1 import AgentStancePayload, Round1Capture

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS_PATH = Path(
    os.environ.get("THRESHOLDS_CONFIG", "config/thresholds.yaml")
)
_DEFAULT_DISSENT_THRESHOLD = 0.08


# ---------------------------------------------------------------------------
# Return payload — the orchestrator writes the transcripts row from this
# ---------------------------------------------------------------------------

@dataclass
class TranscriptPayload:
    """Row-ready payload for the ``transcripts`` table + the resolved file path.

    The orchestrator receives this from ``write_round1_transcript`` and uses it
    to populate the ``transcripts`` INSERT — the file has already been written
    before this object is returned, so ``full_log_path`` always resolves.
    """
    full_log_path: Path
    summary: str
    vote_tally: str        # JSON string — per-ticker action counts
    key_contention: str


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def write_round1_transcript(
    round1_capture: Round1Capture,
    consensus: dict[str, float],
    std_dev: dict[str, float],
    decision: str,
    *,
    week_id: str,
    state_root: Path = Path("state"),
    thresholds_path: Path | None = None,
) -> Path:
    """Assemble + atomically write the Round-1 transcript markdown.

    Writes ``state/debates/YYYY-WNN.md`` first, then returns the resolved path
    so the orchestrator can set ``transcripts.full_log_path`` to a path that
    already exists.

    Args:
        round1_capture:   The ``Round1Capture`` from ``capture_round1_stances``.
                          Provides per-persona stances + narratives.
        consensus:        Consensus weights dict from ``blend_consensus``
                          (ticker → weight, no CASH key).
        std_dev:          Per-ticker stance std-dev dict from the orchestrator.
        decision:         Founder decision string, e.g. ``"panel_approved"`` or
                          ``"founder_override"``.
        week_id:          ISO week label, e.g. ``"2026-W23"``.
        state_root:       Root of the runtime state directory (default ``state/``).
        thresholds_path:  Override path to ``thresholds.yaml``; falls back to the
                          ``THRESHOLDS_CONFIG`` env var or ``config/thresholds.yaml``.

    Returns:
        The resolved ``Path`` to the written markdown file.

    Raises:
        OSError: If the atomic write or rename fails.  Any prior transcript at
                 the target path is left intact on failure.
    """
    _thresholds = _load_thresholds(thresholds_path)
    dissent_threshold: float = float(
        _thresholds.get("dissent_std_dev_threshold", _DEFAULT_DISSENT_THRESHOLD)
    )

    debates_dir = state_root / "debates"
    debates_dir.mkdir(parents=True, exist_ok=True)
    target_path = debates_dir / f"{week_id}.md"

    vote_tally = _build_vote_tally(round1_capture.stances)
    key_contention = _build_key_contention(std_dev, dissent_threshold)
    summary = _build_summary(round1_capture.stances, consensus, week_id)

    markdown = _render_markdown(
        round1_capture=round1_capture,
        consensus=consensus,
        std_dev=std_dev,
        decision=decision,
        week_id=week_id,
        dissent_threshold=dissent_threshold,
        key_contention=key_contention,
    )

    _atomic_write(target_path, markdown)

    logger.info(
        "Round-1 transcript written: %s  (%d personas, %d stance rows)",
        target_path,
        len(set(s.persona for s in round1_capture.stances)),
        len(round1_capture.stances),
    )

    # Attach the payload attributes the orchestrator needs for the DB row.
    # Return only the path per the spec signature — the orchestrator already
    # computes summary/vote_tally/key_contention itself via the helpers in
    # weekly_run.py.  We store them as module-level cache in case the caller
    # wants them, but the contract return is Path only.
    write_round1_transcript._last_payload = TranscriptPayload(  # type: ignore[attr-defined]
        full_log_path=target_path,
        summary=summary,
        vote_tally=vote_tally,
        key_contention=key_contention,
    )

    return target_path


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _render_markdown(
    *,
    round1_capture: Round1Capture,
    consensus: dict[str, float],
    std_dev: dict[str, float],
    decision: str,
    week_id: str,
    dissent_threshold: float,
    key_contention: str,
) -> str:
    """Build the full transcript markdown string.

    Structure (M3-append-safe):
        # Transcript — <week_id>
        ## Round 1
        ### Persona stances (×7)
        ### Consensus
        ### Dissent note
        ### Founder decision
        <!-- ROUND-2-INSERT-POINT -->

    M3 appends ``## Round 2`` after the insert-point comment without touching
    the Round-1 section.
    """
    lines: list[str] = []

    lines.append(f"# Transcript — {week_id}")
    lines.append("")

    # ------------------------------------------------------------------
    # Round 1 section
    # ------------------------------------------------------------------
    lines.append("## Round 1")
    lines.append("")

    # Group stances by persona (preserving insertion order).
    persona_order: list[str] = list(dict.fromkeys(s.persona for s in round1_capture.stances))

    for persona in persona_order:
        narrative = round1_capture.narratives.get(persona, "")
        persona_stances = [s for s in round1_capture.stances if s.persona == persona]

        lines.append(f"### {persona}")
        lines.append("")
        if narrative:
            lines.append(f"**Narrative:** {narrative}")
            lines.append("")

        lines.append("| Ticker | Action | Weight | Confidence | Rationale |")
        lines.append("|--------|--------|--------|------------|-----------|")
        for s in sorted(persona_stances, key=lambda x: x.ticker):
            rationale_cell = s.rationale.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {s.ticker} | {s.action} | {s.target_weight:.4f} "
                f"| {s.confidence} | {rationale_cell} |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------
    lines.append("### Consensus")
    lines.append("")
    if consensus:
        cash = round(1.0 - sum(consensus.values()), 6)
        lines.append("| Ticker | Proposed Weight |")
        lines.append("|--------|----------------|")
        for ticker, weight in sorted(consensus.items()):
            lines.append(f"| {ticker} | {weight:.4f} |")
        lines.append(f"| CASH | {max(0.0, cash):.4f} |")
    else:
        lines.append("_All personas EXIT — 100% cash._")
    lines.append("")

    # ------------------------------------------------------------------
    # Dissent note (std-dev driven)
    # ------------------------------------------------------------------
    lines.append("### Dissent Note")
    lines.append("")
    lines.append(f"**Threshold:** σ ≥ {dissent_threshold:.3f}")
    lines.append("")
    lines.append(f"**Key contention:** {key_contention}")
    lines.append("")

    contested = [
        (t, v) for t, v in std_dev.items() if v >= dissent_threshold
    ]
    if contested:
        lines.append("| Ticker | Std Dev (σ) |")
        lines.append("|--------|------------|")
        for ticker, sigma in sorted(contested, key=lambda x: -x[1]):
            lines.append(f"| {ticker} | {sigma:.4f} |")
        lines.append("")
    else:
        lines.append("_No tickers exceeded the dissent threshold._")
        lines.append("")

    # ------------------------------------------------------------------
    # Founder decision
    # ------------------------------------------------------------------
    lines.append("### Founder Decision")
    lines.append("")
    lines.append(f"**Decision:** {decision}")
    lines.append("")

    # M3-append anchor — M3 inserts ## Round 2 after this comment.
    lines.append("<!-- ROUND-2-INSERT-POINT -->")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vote-tally builder
# ---------------------------------------------------------------------------

def _build_vote_tally(stances: list[AgentStancePayload]) -> str:
    """Build the vote_tally JSON string: per-ticker action counts (round=1 only).

    Returns:
        A JSON string mapping ticker → {"add": n, "reduce": n, "hold": n, "exit": n}.
    """
    tally: dict[str, dict[str, int]] = {}
    for s in stances:
        if s.ticker not in tally:
            tally[s.ticker] = {"add": 0, "reduce": 0, "hold": 0, "exit": 0}
        tally[s.ticker][s.action] = tally[s.ticker].get(s.action, 0) + 1
    return json.dumps(tally, sort_keys=True)


# ---------------------------------------------------------------------------
# Key-contention builder
# ---------------------------------------------------------------------------

def _build_key_contention(std_devs: dict[str, float], threshold: float) -> str:
    """Return a human-readable dissent note for the most contested tickers."""
    contested = sorted(
        ((t, v) for t, v in std_devs.items() if v >= threshold),
        key=lambda x: -x[1],
    )
    if not contested:
        return "No tickers above dissent threshold."
    top = contested[:3]
    parts = [f"{t} (σ={v:.3f})" for t, v in top]
    suffix = "." if len(contested) <= 3 else f" (+{len(contested) - 3} more)."
    return "Contested: " + ", ".join(parts) + suffix


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    stances: list[AgentStancePayload],
    consensus: dict[str, float],
    week_id: str,
) -> str:
    """Build a 1–3 line summary string for the transcripts row."""
    n_add = sum(1 for s in stances if s.action == "add")
    n_reduce = sum(1 for s in stances if s.action == "reduce")
    n_hold = sum(1 for s in stances if s.action == "hold")
    n_exit = sum(1 for s in stances if s.action == "exit")
    n_tickers = len(set(s.ticker for s in stances))
    consensus_size = len(consensus)
    return (
        f"Round-1 consensus for {week_id}. "
        f"{n_tickers} debate-set tickers; consensus holds {consensus_size} positions. "
        f"Stance breakdown — add: {n_add}, reduce: {n_reduce}, hold: {n_hold}, exit: {n_exit}."
    )


# ---------------------------------------------------------------------------
# Atomic file write
# ---------------------------------------------------------------------------

def _atomic_write(target: Path, content: str) -> None:
    """Write content to target atomically via tmpfile → os.rename.

    If the write itself fails (e.g. disk full), the .tmp file is cleaned up
    and any prior file at target is left intact.  os.rename is atomic on POSIX.
    """
    tmp = target.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.rename(str(tmp), str(target))
    except Exception:
        # Best-effort cleanup of the partial temp file.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_thresholds(override: Path | None = None) -> dict[str, Any]:
    path = override or _DEFAULT_THRESHOLDS_PATH
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logger.warning(
            "thresholds.yaml not found at %s — using built-in defaults.", path
        )
        return {}
