"""Component 18 / 18b — per-persona memory write-back (`state/memory/<persona>.md`).

Appends this week's entries to the four sections of each persona's memory file:
  1. Past-calls log       — this week's Round-1 stance summary + (future) alpha outcome
  2. Counterfactual log   — this week's counterfactual portfolio snapshot
  3. Debate-stances log   — this week's Round-1 narrative summary
  4. What's-new digest    — Component 28 resolved-outcomes digest (M4); report summary (M2)

Design contracts (M2, unchanged by M4):
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
  - Sole-writer: only writeback_memory writes memory files; readers are
    read-only.

M4 extension (Component 18b):
  - Resolved-alpha backfill: for each (week, persona) in ``resolved_alpha``,
    the past-calls ``### Entry <week>`` for that persona has its
    ``outcome: pending`` line replaced in-place with
    ``outcome: alpha=<value> resolved=<this_week>``.  This is an UPDATE of a
    prior entry, never a new append.  If the entry has overflowed to the archive
    the archive copy is updated instead.  If neither is found a Minor warning is
    logged and backfill is skipped — no phantom entry is ever fabricated.
  - Digest source: the what's-new digest entry is now sourced from Component
    28's ``build_whats_new_digest`` text (a per-persona map passed via
    ``whats_new_digests``).  Falls back to the M2 report-summary path when the
    map is absent or has no entry for the persona.

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
# Resolved-alpha in-place backfill helpers (M4 / Component 18b)
# ---------------------------------------------------------------------------

_OUTCOME_PENDING = "outcome: pending"


def _outcome_resolved_line(alpha: float, resolved_week: str) -> str:
    """Render the resolved outcome line for a backfilled past-calls entry."""
    return f"outcome: alpha={alpha:+.4f} resolved={resolved_week}"


def _backfill_outcome_in_body(body: str, alpha: float, resolved_week: str) -> str:
    """Replace the ``outcome: pending`` line in *body* with the resolved value.

    Returns the modified body.  If the body does not contain ``outcome: pending``
    (already resolved or unusual format), returns it unchanged.
    """
    resolved_line = _outcome_resolved_line(alpha, resolved_week)
    return body.replace(_OUTCOME_PENDING, resolved_line, 1)


def _backfill_in_parsed(
    parsed: ParsedMemoryFile,
    persona: str,
    call_week: str,
    alpha: float,
    resolved_week: str,
) -> bool:
    """Update the ``outcome:`` line of the past-calls entry for *call_week* in place.

    Operates on the in-memory ``ParsedMemoryFile``.  Returns True when the
    entry was found and updated, False when not present (overflowed to archive
    or genuinely absent).

    Never creates a new entry — this is an in-place update only.
    """
    section = parsed.sections.get(SECTION_PAST_CALLS)
    if section is None:
        return False
    for idx, (week_id, body) in enumerate(section.entries):
        if week_id == call_week:
            new_body = _backfill_outcome_in_body(body, alpha, resolved_week)
            section.entries[idx] = (week_id, new_body)
            logger.debug(
                "Backfilled outcome in live file: persona=%s week=%s alpha=%+.4f",
                persona, call_week, alpha,
            )
            return True
    return False


def _backfill_in_archive(
    archive_path: Path,
    persona: str,
    call_week: str,
    alpha: float,
    resolved_week: str,
) -> bool:
    """Update the ``outcome:`` line of *call_week*'s entry in the archive file.

    Reads the archive file, replaces the first ``outcome: pending`` that appears
    inside the ``### Entry <call_week>`` block, then rewrites the file atomically.
    Returns True when the entry was found and the file updated, False when not
    present.

    The archive is append-only from the cap-overflow path; this function applies
    a targeted line replacement inside the relevant block without altering any
    other content.
    """
    if not archive_path.exists():
        return False

    text = archive_path.read_text(encoding="utf-8")
    entry_header = f"{_ENTRY_PREFIX}{call_week}\n"
    start = text.find(entry_header)
    if start == -1:
        return False

    # Find the end of this entry's block: next "### Entry" or end of file.
    body_start = start + len(entry_header)
    next_entry = text.find(_ENTRY_PREFIX, body_start)
    block_end = next_entry if next_entry != -1 else len(text)

    block = text[body_start:block_end]
    if _OUTCOME_PENDING not in block:
        # Already resolved or no outcome line — nothing to do.
        return False

    resolved_line = _outcome_resolved_line(alpha, resolved_week)
    new_block = block.replace(_OUTCOME_PENDING, resolved_line, 1)
    new_text = text[:body_start] + new_block + text[block_end:]

    _write_atomically(archive_path, new_text)
    logger.debug(
        "Backfilled outcome in archive: persona=%s week=%s alpha=%+.4f path=%s",
        persona, call_week, alpha, archive_path,
    )
    return True


def _apply_resolved_alpha_backfill(
    parsed: ParsedMemoryFile,
    persona: str,
    resolved_week: str,
    resolved_alpha: dict[str, object],
    archive_dir: Path,
) -> None:
    """Apply all resolved-alpha backfills for *persona* from *resolved_alpha*.

    ``resolved_alpha`` shape: ``{week_id: {persona_slug: alpha_value}}``.

    For each (call_week, alpha) where this persona appears:
      1. Try to update the entry in the live parsed file (in-memory).
      2. If not found there (overflowed to archive), try the archive file.
      3. If still not found, log a Minor warning and skip — no phantom entry.
    """
    for call_week, per_persona in resolved_alpha.items():  # type: ignore[union-attr]
        if not isinstance(per_persona, dict):
            continue
        alpha_val = per_persona.get(persona)
        if alpha_val is None:
            continue
        alpha = float(alpha_val)

        found_in_live = _backfill_in_parsed(
            parsed, persona, call_week, alpha, resolved_week
        )
        if found_in_live:
            continue

        archive_path = archive_dir / f"{persona}.md"
        found_in_archive = _backfill_in_archive(
            archive_path, persona, call_week, alpha, resolved_week
        )
        if not found_in_archive:
            logger.warning(
                "resolved_alpha backfill skipped — entry not found in live file "
                "or archive (phantom-guard): persona=%s call_week=%s alpha=%+.4f",
                persona, call_week, alpha,
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
    whats_new_digests: dict[str, str] | None = None,
) -> str:
    """Build the what's-new digest entry.

    M4 path (preferred): when *whats_new_digests* contains an entry for
    *persona*, that Component 28 text is used verbatim as the digest body.

    M2 fallback: when *whats_new_digests* is absent or has no entry for
    *persona*, the report summary is used (preserves M2 behaviour).
    """
    # M4 path — Component 28 digest text.
    if whats_new_digests is not None:
        digest_text = whats_new_digests.get(persona)
        if digest_text is not None:
            return f"week: {week_id}\ndigest: {digest_text}"

    # M2 fallback — report summary.
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
    whats_new_digests: dict[str, str] | None = None,
    memory_dir: Path = Path("state/memory"),
    archive_dir: Path = Path("state/memory/archive"),
    cap: int = _DEFAULT_CAP,
) -> None:
    """Write per-persona memory files after the week's ledger transaction commits.

    For each of the 7 personas, appends one entry to each of the four sections:
      1. Past-calls log       — Round-1 stances + pending outcome field
      2. Counterfactual log   — counterfactual portfolio snapshot
      3. Debate-stances log   — Round-1 narrative summary
      4. What's-new digest    — Component 28 digest (M4) or report summary (M2)

    M4 extensions (Component 18b):
      - Resolved-alpha backfill: entries in ``resolved_alpha`` (shape
        ``{week_id: {persona: alpha_value}}``) are applied IN PLACE to prior
        past-calls entries, replacing ``outcome: pending`` with the resolved
        value.  Live-file entries are updated before the new week is appended;
        archived entries are updated in the archive file.  No phantom entries
        are ever created.
      - Digest source: when ``whats_new_digests`` carries an entry for a
        persona, that Component 28 text is written as the digest body instead
        of the report summary.

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
        resolved_alpha:    Map of newly-resolved alpha outcomes whose
                           look-forward window closed this week, shape
                           ``{week_id: {persona: alpha_value}}``.  May be
                           empty (``{}``) at M2/M3; populated at M4+.
        whats_new_digests: Optional per-persona Component 28 digest texts,
                           shape ``{persona: digest_text}``.  When provided,
                           these replace the M2 report-summary source for the
                           what's-new digest section.
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
    archive_dir.mkdir(parents=True, exist_ok=True)

    for persona in persona_slugs:
        memory_path = memory_dir / f"{persona}.md"

        # Load (or create) the existing memory file.
        parsed = parse_memory_file(memory_path)

        # M4: backfill any resolved outcomes onto prior past-calls entries
        # BEFORE appending this week's new entry.  This updates the parsed
        # in-memory structure for live-file entries; archive entries are
        # updated directly on disk inside _apply_resolved_alpha_backfill.
        if resolved_alpha:
            _apply_resolved_alpha_backfill(
                parsed, persona, week_id, resolved_alpha, archive_dir
            )

        # Build the four new entries for this week.
        past_calls_entry = _build_past_calls_entry(persona, week_id, capture)
        counterfactual_entry = _build_counterfactual_entry(persona, week_id, counterfactuals)
        debate_stances_entry = _build_debate_stances_entry(persona, week_id, capture)
        whats_new_entry = _build_whats_new_entry(
            persona, week_id, validated_reports, whats_new_digests
        )

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
