"""Component 18 — per-persona memory write-back (`state/memory/<persona>.md`).

Appends this week's entries to the four sections of each persona's memory file:
  1. Past-calls log       — this week's Round-1 stance summary + (future) alpha outcome
  2. Counterfactual log   — this week's counterfactual portfolio snapshot
  3. Debate-stances log   — this week's Round-1 narrative summary
  4. What's-new digest    — fresh summary of this week's research highlights

Design contracts:
  - Write-back happens AFTER the ledger transaction commits.  The orchestrator
    (weekly_run.py) is the sole caller; it calls writeback_memory as the last
    step after conn.commit() — never before.  A rolled-back week leaves memory
    unchanged (memory lags the ledger, never leads it).
  - Atomic write: content is written to a tmpfile then os.rename'd over the
    destination.  A mid-write failure leaves the prior file intact.
  - 12-entry-per-section cap: when a section reaches the cap, the OLDEST entry
    is moved to state/memory/archive/<persona>.md (appended there) before the
    new entry is added.  Entries are never dropped.
  - If a persona's memory file is absent, it is created on first write.

Round-trip contract:
  The section format written here is the canonical format for both writing and
  reading.  The orchestrator (M1-010 research dispatch + memory injection) reads
  memory files using ``parse_memory_file`` exported from this module.  Any
  change to the section markers or entry format must preserve the round-trip.

File format::

    # Persona Memory

    ## Past Calls Log

    ### Entry YYYY-WNN
    <content lines>

    ### Entry YYYY-WNN
    <content lines>

    ## Counterfactual Portfolio Log

    ### Entry YYYY-WNN
    <content lines>

    ## Debate Stances Log

    ### Entry YYYY-WNN
    <content lines>

    ## What's New Digest

    ### Entry YYYY-WNN
    <content lines>
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section names — canonical, used for both write and parse
# ---------------------------------------------------------------------------

SECTION_PAST_CALLS = "Past Calls Log"
SECTION_COUNTERFACTUAL = "Counterfactual Portfolio Log"
SECTION_DEBATE_STANCES = "Debate Stances Log"
SECTION_WHATS_NEW = "What's New Digest"

_ALL_SECTIONS = [
    SECTION_PAST_CALLS,
    SECTION_COUNTERFACTUAL,
    SECTION_DEBATE_STANCES,
    SECTION_WHATS_NEW,
]

_ENTRY_PREFIX = "### Entry "

_DEFAULT_CAP = 12


# ---------------------------------------------------------------------------
# Parsed memory file representation
# ---------------------------------------------------------------------------

@dataclass
class MemorySection:
    """Ordered list of (week_id, content_lines) entries for one section."""
    name: str
    entries: list[tuple[str, str]] = field(default_factory=list)
    # Each entry: (week_id, body_text) where body_text is the text after the
    # "### Entry <week_id>" header line (stripped of surrounding blank lines).


@dataclass
class ParsedMemoryFile:
    """All four sections of one persona's memory file."""
    sections: dict[str, MemorySection] = field(default_factory=dict)

    def get_section(self, name: str) -> MemorySection:
        return self.sections.setdefault(name, MemorySection(name=name))


# ---------------------------------------------------------------------------
# Parser (round-trip reader)
# ---------------------------------------------------------------------------

def parse_memory_file(path: Path) -> ParsedMemoryFile:
    """Parse a memory file into a ParsedMemoryFile.

    Tolerant of missing sections (returns them as empty).  Unknown sections
    between known h2 headers are ignored — future-safe.

    The parser is the canonical read path used by the orchestrator before each
    weekly run (M1-010 / Component 12 prompt construction).

    Args:
        path: Path to a ``state/memory/<persona>.md`` file.

    Returns:
        A ``ParsedMemoryFile`` with each section's entries in chronological
        (file) order.
    """
    result = ParsedMemoryFile()
    if not path.exists():
        return result

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    current_section: Optional[str] = None
    current_entry_week: Optional[str] = None
    current_entry_lines: list[str] = []

    def _flush_entry() -> None:
        if current_section is not None and current_entry_week is not None:
            body = "\n".join(current_entry_lines).strip()
            result.get_section(current_section).entries.append(
                (current_entry_week, body)
            )

    for line in lines:
        # Detect ## section headers.
        if line.startswith("## "):
            _flush_entry()
            current_entry_week = None
            current_entry_lines = []
            section_name = line[3:].strip()
            if section_name in _ALL_SECTIONS:
                current_section = section_name
            else:
                current_section = None
            continue

        # Detect ### Entry <week_id> headers within a known section.
        if current_section is not None and line.startswith(_ENTRY_PREFIX):
            _flush_entry()
            current_entry_week = line[len(_ENTRY_PREFIX):].strip()
            current_entry_lines = []
            continue

        # Accumulate body lines for the current entry.
        if current_section is not None and current_entry_week is not None:
            current_entry_lines.append(line)

    _flush_entry()
    return result


# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------

def _render_memory_file(persona: str, parsed: ParsedMemoryFile) -> str:
    """Render a ParsedMemoryFile back to the canonical markdown string."""
    lines: list[str] = [f"# Persona Memory — {persona}", ""]

    for section_name in _ALL_SECTIONS:
        section = parsed.sections.get(section_name, MemorySection(name=section_name))
        lines.append(f"## {section_name}")
        lines.append("")
        if not section.entries:
            lines.append("_No entries yet._")
            lines.append("")
        else:
            for week_id, body in section.entries:
                lines.append(f"{_ENTRY_PREFIX}{week_id}")
                if body:
                    lines.append(body)
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Archive writer
# ---------------------------------------------------------------------------

def _append_to_archive(
    persona: str,
    section_name: str,
    week_id: str,
    body: str,
    archive_dir: Path,
) -> None:
    """Append a single overflowed entry to the archive file for this persona."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{persona}.md"
    entry_text = (
        f"\n## Archived from {section_name}\n"
        f"\n{_ENTRY_PREFIX}{week_id}\n"
        f"{body}\n"
    )
    with archive_path.open("a", encoding="utf-8") as fh:
        fh.write(entry_text)
    logger.debug(
        "Archived overflow entry: persona=%s section=%r week=%s → %s",
        persona, section_name, week_id, archive_path,
    )


# ---------------------------------------------------------------------------
# Cap enforcer
# ---------------------------------------------------------------------------

def _enforce_cap(
    section: MemorySection,
    cap: int,
    persona: str,
    archive_dir: Path,
) -> None:
    """If section has ``cap`` entries, archive the oldest before appending.

    Called BEFORE the new entry is added, so after this function returns the
    section has at most ``cap - 1`` entries and there is room for one more.

    When the section already has ``cap`` entries:
      - The oldest entry (index 0) is moved to the archive file.
      - It is removed from ``section.entries``.
    """
    while len(section.entries) >= cap:
        oldest_week, oldest_body = section.entries.pop(0)
        _append_to_archive(
            persona=persona,
            section_name=section.name,
            week_id=oldest_week,
            body=oldest_body,
            archive_dir=archive_dir,
        )
        logger.info(
            "Cap overflow archived: persona=%s section=%r evicted=%s",
            persona, section.name, oldest_week,
        )


# ---------------------------------------------------------------------------
# Atomic file writer
# ---------------------------------------------------------------------------

def _write_atomically(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via tmpfile + os.rename.

    A crash or exception during the write leaves the prior file intact.
    Uses a sibling tmpfile (same directory) so os.rename is an atomic
    filesystem operation (same mount point).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file, then rename.
    fd, tmp_path_str = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.rename(tmp_path_str, str(path))
    except Exception:
        # Clean up the temp file; leave the original untouched.
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Entry builders
# ---------------------------------------------------------------------------

def _build_past_calls_entry(
    persona: str,
    week_id: str,
    round1_capture: object,
) -> str:
    """Build the past-calls log entry for one persona.

    Records the Round-1 stances (action + confidence) for all debate-set
    tickers.  Alpha outcome is not yet known at write-back time (look-forward
    window is open); the field is left as ``outcome: pending``.
    """
    from round_table_portfolio.orchestrator.round1 import Round1Capture
    capture: Round1Capture = round1_capture  # type: ignore[assignment]

    persona_stances = [s for s in capture.stances if s.persona == persona]
    if not persona_stances:
        return f"week: {week_id}\nstances: (none)\noutcome: pending"

    stance_lines = "\n".join(
        f"  {s.ticker}: {s.action} confidence={s.confidence} weight={s.target_weight:.3f}"
        for s in sorted(persona_stances, key=lambda s: s.ticker)
    )
    return f"week: {week_id}\nstances:\n{stance_lines}\noutcome: pending"


def _build_counterfactual_entry(
    persona: str,
    week_id: str,
    counterfactuals: dict[str, dict[str, float]],
) -> str:
    """Build the counterfactual-portfolio log entry for one persona."""
    portfolio = counterfactuals.get(persona, {})
    if not portfolio:
        return f"week: {week_id}\nportfolio: (none)"

    position_lines = "\n".join(
        f"  {ticker}: {weight:.4f}"
        for ticker, weight in sorted(portfolio.items(), key=lambda x: -x[1])
    )
    return f"week: {week_id}\nportfolio:\n{position_lines}"


def _build_debate_stances_entry(
    persona: str,
    week_id: str,
    round1_capture: object,
) -> str:
    """Build the debate-stances log entry (Round-1 narrative summary)."""
    from round_table_portfolio.orchestrator.round1 import Round1Capture
    capture: Round1Capture = round1_capture  # type: ignore[assignment]

    narrative = capture.narratives.get(persona, "")
    if not narrative:
        narrative = "(no narrative recorded)"
    return f"week: {week_id}\nnarrative: {narrative}"


def _build_whats_new_entry(
    persona: str,
    week_id: str,
    validated_reports: list[object],
) -> str:
    """Build the what's-new digest entry from the persona's validated report summary."""
    from round_table_portfolio.research.runner import PersonaResearchResult
    for res in validated_reports:
        r: PersonaResearchResult = res  # type: ignore[assignment]
        if r.persona_slug == persona:
            summary = r.report_payload.summary[:300] if r.report_payload else ""
            validator_flag = "PASS" if r.validation.passed else "FAIL"
            return (
                f"week: {week_id}\n"
                f"validator: {validator_flag}\n"
                f"digest: {summary}"
            )
    return f"week: {week_id}\ndigest: (no report found)"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def writeback_memory(
    round1_capture: object,
    counterfactuals: dict[str, dict[str, float]],
    validated_reports: list[object],
    resolved_alpha: dict[str, object],
    *,
    memory_dir: Path = Path("state/memory"),
    archive_dir: Path = Path("state/memory/archive"),
    cap: int = _DEFAULT_CAP,
) -> None:
    """Write per-persona memory files after the week's ledger transaction commits.

    For each of the 7 personas, appends one entry to each of the four sections:
      1. Past-calls log       — Round-1 stances + pending outcome field
      2. Counterfactual log   — counterfactual portfolio snapshot
      3. Debate-stances log   — Round-1 narrative summary
      4. What's-new digest    — research summary + validator verdict

    Enforces a ``cap``-entry-per-section limit.  When a section reaches
    ``cap``, the oldest entry is archived to ``archive_dir/<persona>.md``
    BEFORE the new entry is appended.  Entries are never dropped.

    Each persona's file is written atomically (tmpfile → os.rename) so a
    mid-write failure leaves the prior file intact.

    ORDERING CONTRACT (enforced by the orchestrator):
        This function MUST be called AFTER ``conn.commit()`` in weekly_run.py.
        It must never be called inside the transaction block or on a rolled-back
        path.  A rolled-back week leaves memory unchanged.

    Args:
        round1_capture:    ``Round1Capture`` from ``capture_round1_stances``.
        counterfactuals:   Per-persona counterfactual portfolio dicts
                           ``{persona: {ticker: weight, 'CASH': weight}}``.
        validated_reports: List of ``PersonaResearchResult`` from this week's
                           per-persona research phase.
        resolved_alpha:    Dict of any newly-resolved alpha outcomes whose
                           look-forward window closed this week.  May be empty
                           at PoC (no resolved weeks yet on first run).
        memory_dir:        Root of the memory files (default ``state/memory/``).
        archive_dir:       Root of the archive files (default
                           ``state/memory/archive/``).
        cap:               Max entries per section before archiving (default 12).

    Raises:
        OSError: If an atomic write fails (the prior memory file is left intact
                 per the atomic-write contract; the exception propagates so the
                 orchestrator can log and flag it as a Major-tier failure).
    """
    from round_table_portfolio.orchestrator.round1 import Round1Capture
    capture: Round1Capture = round1_capture  # type: ignore[assignment]

    # Derive the week_id from the first stance (all stances carry the same week).
    week_id = ""
    if capture.stances:
        week_id = capture.stances[0].week_id
    elif validated_reports:
        from round_table_portfolio.research.runner import PersonaResearchResult
        week_id = validated_reports[0].week_id  # type: ignore[union-attr]

    if not week_id:
        logger.warning("writeback_memory: could not determine week_id — skipping write-back.")
        return

    # Collect the persona slugs from the round1 capture (the authoritative list).
    persona_slugs: list[str] = sorted(
        {s.persona for s in capture.stances}
    )
    if not persona_slugs:
        # Fall back to validated_reports persona slugs.
        from round_table_portfolio.research.runner import PersonaResearchResult
        persona_slugs = sorted(
            {r.persona_slug for r in validated_reports}  # type: ignore[union-attr]
        )

    memory_dir.mkdir(parents=True, exist_ok=True)

    for persona in persona_slugs:
        memory_path = memory_dir / f"{persona}.md"

        # Load (or create) the existing memory file.
        parsed = parse_memory_file(memory_path)

        # Build the four new entries for this week.
        past_calls_entry = _build_past_calls_entry(persona, week_id, capture)
        counterfactual_entry = _build_counterfactual_entry(persona, week_id, counterfactuals)
        debate_stances_entry = _build_debate_stances_entry(persona, week_id, capture)
        whats_new_entry = _build_whats_new_entry(persona, week_id, validated_reports)

        # Append to each section, enforcing the cap.
        for section_name, entry_body in [
            (SECTION_PAST_CALLS, past_calls_entry),
            (SECTION_COUNTERFACTUAL, counterfactual_entry),
            (SECTION_DEBATE_STANCES, debate_stances_entry),
            (SECTION_WHATS_NEW, whats_new_entry),
        ]:
            section = parsed.get_section(section_name)
            _enforce_cap(section, cap, persona, archive_dir)
            section.entries.append((week_id, entry_body))

        # Render and write atomically.
        content = _render_memory_file(persona, parsed)
        _write_atomically(memory_path, content)
        logger.info(
            "Memory write-back complete: persona=%s week=%s path=%s",
            persona, week_id, memory_path,
        )
