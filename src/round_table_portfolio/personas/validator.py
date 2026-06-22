"""Persona-definition conformance validator.

Reads a persona ``.md`` file (frontmatter + prompt body) and asserts it conforms
to the framework convention documented at ``.claude/agents/_FRAMEWORK.md``
(TDD Part 2 Component 5).

This validator enforces the STRUCTURE of a persona definition file. It does NOT
judge report quality / on-mandate-ness at runtime — that is the output validator
(Component 11), a separate component.

Usage::

    from round_table_portfolio.personas.validator import validate_persona_definition
    result = validate_persona_definition(Path(".claude/agents/value.md"))
    if not result.ok:
        for v in result.violations:
            print(v)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Required prompt-body sections, matched on H2 heading text (case-insensitive).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "MANDATE",
    "RESEARCH MANDATE",
    "HOLDING HORIZON",
    "RESEARCH ACCESS",
    "ALLOWED ACTIONS",
    "MEMORY",
    "RESEARCH OUTPUT SCHEMA",
    "ROUND 1 OUTPUT SCHEMA",
    "ROUND 2 OUTPUT SCHEMA",
)

# Tools that MUST appear in the frontmatter `tools` list.
REQUIRED_TOOLS: tuple[str, ...] = ("Bash", "WebSearch")

# Minimum substantive length (chars) for the MANDATE / RESEARCH MANDATE bodies —
# guards against thin one-line placeholders. The mandate is the moat.
MIN_SUBSTANTIVE_CHARS = 120

# Minimum substantive length (chars) for the HOLDING HORIZON body — guards against
# one-line blanket stubs like "hold for the medium term". The per-archetype clause
# must be specific enough to convey WHY this persona's horizon works the way it does.
MIN_HORIZON_CHARS = 150

# Words that signal the explicit "ignore" clause required in RESEARCH MANDATE.
_IGNORE_WORDS = ("ignore", "ignores", "avoid", "avoids", "exclude", "excludes", "do not", "does not")

# Terms that must appear in HOLDING HORIZON to confirm (a) the 3mo–2yr band is named
# and (b) the thesis-change-not-price-move principle is stated.
# Check (a): at least one of these must appear in the section text (case-insensitive).
_HORIZON_BAND_TERMS = ("3-month", "3 month", "2-year", "2 year", "medium-term", "medium term")
# Check (b): the clause must contain a thesis-change or anti-price-move signal.
# Covers the full range of per-archetype voices: value/growth use "thesis" + "price move";
# cta/quant/technical use "price move" or "rationale must"; risk-officer uses "tape"
# and "rationale must" (its principle is "driven by risk picture, not the weekly tape").
_HORIZON_THESIS_TERMS = (
    "thesis",
    "price move",
    "price-move",
    "price action",
    "not because",
    "not a price",
    "not on a price",
    "not on price",
    "rationale must",
    "tape",
)


@dataclass
class PersonaValidationResult:
    """Structured outcome of validating one persona file."""

    path: Path
    ok: bool
    violations: list[str] = field(default_factory=list)
    name: str | None = None

    def __bool__(self) -> bool:  # convenience: `if result:`
        return self.ok


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    """Return (frontmatter_block, body). Frontmatter is the leading ``---`` fence."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return None, text
    return m.group(1), m.group(2)


def _parse_frontmatter_tools(fm: str) -> list[str] | None:
    """Extract the `tools:` list from a frontmatter block (inline-list form only)."""
    m = re.search(r"^tools:\s*\[(.*?)\]\s*$", fm, re.MULTILINE)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return []
    return [t.strip().strip("'\"") for t in inner.split(",") if t.strip()]


def _parse_frontmatter_field(fm: str, key: str) -> str | None:
    """Extract a simple scalar `key: value` field from frontmatter."""
    m = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", fm, re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def _section_bodies(body: str) -> dict[str, str]:
    """Map upper-cased H2 heading text -> the text under that heading (until next H2)."""
    sections: dict[str, str] = {}
    # Find all H2 headings and their spans.
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", body, re.MULTILINE))
    for i, m in enumerate(matches):
        heading = m.group(1).strip().upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[heading] = body[start:end].strip()
    return sections


def validate_persona_definition(path: str | Path) -> PersonaValidationResult:
    """Validate a single persona ``.md`` file against the framework convention.

    Returns a :class:`PersonaValidationResult`. ``ok is True`` only when every
    conformance check passes. Each failure appends a human-readable violation.
    """
    path = Path(path)
    if not path.exists():
        return PersonaValidationResult(path=path, ok=False, violations=[f"File not found: {path}"])

    text = path.read_text(encoding="utf-8")
    violations: list[str] = []

    # --- Frontmatter -------------------------------------------------------
    fm, body = _split_frontmatter(text)
    name: str | None = None
    if fm is None:
        violations.append("Missing or malformed YAML frontmatter (expected leading `---` fence).")
    else:
        name = _parse_frontmatter_field(fm, "name")
        description = _parse_frontmatter_field(fm, "description")
        tools = _parse_frontmatter_tools(fm)

        if not name:
            violations.append("Frontmatter missing non-empty `name`.")
        if not description:
            violations.append("Frontmatter missing non-empty `description`.")
        if tools is None:
            violations.append("Frontmatter missing `tools:` inline list.")
        else:
            for required in REQUIRED_TOOLS:
                if required not in tools:
                    violations.append(f"Frontmatter `tools` is missing required tool `{required}`.")

    # --- Required sections present ----------------------------------------
    sections = _section_bodies(body)
    for heading in REQUIRED_SECTIONS:
        if heading.upper() not in sections:
            violations.append(f"Missing required section: `## {heading}`.")

    # --- MANDATE / RESEARCH MANDATE substantive ---------------------------
    for heading in ("MANDATE", "RESEARCH MANDATE"):
        text_body = sections.get(heading.upper(), "")
        if heading.upper() in sections and len(text_body) < MIN_SUBSTANTIVE_CHARS:
            violations.append(
                f"`## {heading}` is too thin ({len(text_body)} chars < {MIN_SUBSTANTIVE_CHARS}); "
                "the mandate is the moat — author substantive prose."
            )

    # --- RESEARCH MANDATE has an explicit ignore clause -------------------
    rm = sections.get("RESEARCH MANDATE", "").lower()
    if "RESEARCH MANDATE" in {s.upper() for s in sections} or rm:
        if rm and not any(w in rm for w in _IGNORE_WORDS):
            violations.append(
                "`## RESEARCH MANDATE` has no explicit ignore/avoid/exclude clause; "
                "the 'what it ignores' clause is what makes the persona distinguishable."
            )

    # --- HOLDING HORIZON substantive (M6-002) -----------------------------
    # The section-presence check fires above (REQUIRED_SECTIONS). Here we
    # enforce: (1) non-trivial length — blanket one-liners fail; (2) the
    # 3mo–2yr band is named — confirms the medium-term window is explicit;
    # (3) the thesis-change-not-price-move principle appears — confirms the
    # exit discipline is stated. All three are deterministic text checks; no
    # LLM dispatch (Critical Component #3).
    hh = sections.get("HOLDING HORIZON", "")
    if "HOLDING HORIZON" in {s.upper() for s in sections} or hh:
        hh_lower = hh.lower()
        if hh and len(hh) < MIN_HORIZON_CHARS:
            violations.append(
                f"`## HOLDING HORIZON` is too thin ({len(hh)} chars < {MIN_HORIZON_CHARS}); "
                "author an archetype-specific clause — a blanket one-liner fails."
            )
        else:
            if hh and not any(t in hh_lower for t in _HORIZON_BAND_TERMS):
                violations.append(
                    "`## HOLDING HORIZON` does not reference the 3-month-to-2-year band; "
                    "the medium-term window must be explicitly named (e.g. '3-month', '2-year', "
                    "'medium-term')."
                )
            if hh and not any(t in hh_lower for t in _HORIZON_THESIS_TERMS):
                violations.append(
                    "`## HOLDING HORIZON` omits the thesis-change-not-price-move principle; "
                    "the clause must state that an EXIT/REDUCE/ADD is driven by a change in "
                    "the medium-term thesis, not by a price move."
                )

    # --- ALLOWED ACTIONS must not contain SHORT ---------------------------
    actions = sections.get("ALLOWED ACTIONS", "")
    if re.search(r"\bSHORT\b", actions, re.IGNORECASE):
        violations.append("`## ALLOWED ACTIONS` names `SHORT` — long-only vocabulary only (NFR #3).")

    # --- ROUND 1 SCHEMA must include counterfactual_portfolio -------------
    r1 = sections.get("ROUND 1 OUTPUT SCHEMA", "")
    if "ROUND 1 OUTPUT SCHEMA" in {s.upper() for s in sections}:
        if "counterfactual_portfolio" not in r1:
            violations.append(
                "`## ROUND 1 OUTPUT SCHEMA` omits `counterfactual_portfolio` "
                "(Critical Component #2 anchor)."
            )
        # The explicit-CASH fully-invested clause (Layer 1 of the three-layer
        # fully-invested rule, TASK-M2-001). The counterfactual_portfolio must
        # carry an explicit `CASH` entry so positions + CASH = exactly 100% by
        # design — not a residual the framework computes after the fact. The
        # quoted JSON key `"CASH"` is the deterministic conformance marker.
        elif '"CASH"' not in r1:
            violations.append(
                "`## ROUND 1 OUTPUT SCHEMA` omits the explicit `\"CASH\"` entry in "
                "`counterfactual_portfolio`; the book must be fully specified so "
                "positions + CASH = 100% by design (Layer 1, three-layer "
                "fully-invested rule — TASK-M2-001)."
            )

    # --- RESEARCH OUTPUT SCHEMA must include shortlist + cluster ----------
    ros = sections.get("RESEARCH OUTPUT SCHEMA", "")
    if "RESEARCH OUTPUT SCHEMA" in {s.upper() for s in sections}:
        if "shortlist" not in ros:
            violations.append("`## RESEARCH OUTPUT SCHEMA` omits `shortlist`.")
        if "cluster" not in ros:
            violations.append("`## RESEARCH OUTPUT SCHEMA` omits `cluster`.")

    # --- ROUND 2 SCHEMA must be the NEW shape, not the stale one ----------
    # Heading-present is checked above (REQUIRED_SECTIONS); here we enforce the
    # section's SHAPE so a stale/old-shape block can no longer pass silently
    # (TASK-M3-003 follow-up — the new shape is what parse_round2_reply expects).
    r2 = sections.get("ROUND 2 OUTPUT SCHEMA", "")
    if "ROUND 2 OUTPUT SCHEMA" in {s.upper() for s in sections}:
        for marker in ("stances", "position_change", "rebuttal_narrative"):
            if marker not in r2:
                violations.append(
                    f"`## ROUND 2 OUTPUT SCHEMA` omits `{marker}`; the Round-2 "
                    "reply must use the new shape parsed by `parse_round2_reply` "
                    "(Component 24)."
                )
        for stale in ("addresses_persona", "revised_stances"):
            if stale in r2:
                violations.append(
                    f"`## ROUND 2 OUTPUT SCHEMA` still carries the old-shape field "
                    f"`{stale}`; the Round-2 schema was migrated to "
                    "`stances`/`position_change`/`rebuttal_narrative` "
                    "(TASK-M3-003). Remove the stale block."
                )

    ok = not violations
    if not ok:
        logger.warning("Persona %s failed validation: %d violation(s)", path.name, len(violations))
    return PersonaValidationResult(path=path, ok=ok, violations=violations, name=name)
