"""Tests for validate_persona_definition() — framework conformance (TASK-M1-007).

Covers:
- The shipped `_persona_template.md` PASSES.
- The shipped `_FRAMEWORK.md` exists.
- A programmatically-built conforming persona PASSES.
- >=3 deliberate-failure fixtures each FAIL on exactly one check:
    1. missing-section (no ## MEMORY)
    2. SHORT-in-actions
    3. WebSearch-absent-from-tools
    4. missing counterfactual_portfolio in ROUND 1 schema
    5. thin-mandate (one-line MANDATE)
    6. no-ignore-clause in RESEARCH MANDATE
"""

from __future__ import annotations

from pathlib import Path

import pytest

from round_table_portfolio.personas.validator import (
    REQUIRED_SECTIONS,
    validate_persona_definition,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = PROJECT_ROOT / ".claude" / "agents"
TEMPLATE = AGENTS_DIR / "_persona_template.md"
FRAMEWORK = AGENTS_DIR / "_FRAMEWORK.md"


# ---------------------------------------------------------------------------
# A minimal-but-conforming persona, built in code so fixtures mutate one facet.
# ---------------------------------------------------------------------------

_MANDATE = (
    "The Value persona represents disciplined intrinsic-value investing and covers "
    "the value-style slice of the AUM breakdown. It buys durable businesses below a "
    "conservative estimate of worth and holds them with patience.\n\n"
    "- Primacy of margin of safety over short-term price action.\n"
    "- Cash flow is trusted over reported earnings.\n"
    "- Quality and balance-sheet strength gate every name considered."
)

_RESEARCH_MANDATE = (
    "Research the universe for names trading below your estimate of intrinsic value. "
    "Attend to free-cash-flow yield, balance-sheet strength, and durable competitive "
    "position; weigh those against the price paid. You explicitly IGNORE price-chart "
    "momentum and narrative hype — if a name is up 80% on a story, that is a reason for "
    "caution, not interest. Use fundamentals and filings via the data tools; use web "
    "search only for recent context."
)

_RESEARCH_ACCESS = (
    "You may use WebSearch and the data tools via the CLI:\n\n"
    "```bash\n"
    "cd /Users/andrew.yu/personal/new-structure/projects/round-table-portfolio "
    "&& source .venv/bin/activate "
    "&& python -m round_table_portfolio.data_tools.cli <cmd> <args>\n"
    "```\n\n"
    "Commands: universe, quote, prices, news, peers, fundamentals, technicals, macro, "
    "prenarrow, rss. Budget caps: max turns 12, max web searches 6, max data tool calls 18."
)

_ALLOWED_ACTIONS = (
    "Long-only vocabulary:\n\n"
    "- `ADD <ticker> <weight>`\n"
    "- `REDUCE <ticker> <weight>`\n"
    "- `EXIT <ticker>`\n"
    "- `HOLD <ticker>`\n\n"
    "Weight constraint: 0 <= weight <= max_position_weight."
)

_MEMORY = "Read `state/memory/value.md` before forming this week's stance and reference past calls."

_RESEARCH_OUTPUT_SCHEMA = (
    "```json\n"
    "{\n"
    '  "shortlist": [{"ticker": "<T>", "why": "<why>", "cluster": ["<peer1>"]}],\n'
    '  "report": "<full report>",\n'
    '  "web_searches_used": 0,\n'
    '  "data_tool_calls_used": 0\n'
    "}\n"
    "```"
)

_ROUND1_SCHEMA = (
    "```json\n"
    "{\n"
    '  "round": 1,\n'
    '  "stances": [{"ticker": "<T>", "action": "HOLD", "target_weight": 0.0, '
    '"confidence": 3, "rationale": "<...>"}],\n'
    '  "counterfactual_portfolio": {"<ticker>": 0.0},\n'
    '  "narrative_summary": "<thesis>"\n'
    "}\n"
    "```"
)

_ROUND2_SCHEMA = (
    "```json\n"
    "{\n"
    '  "round": 2,\n'
    '  "addresses_persona": "<name>",\n'
    '  "addresses_position": "<summary>",\n'
    '  "response": "<counterargument>",\n'
    '  "revised_stances": []\n'
    "}\n"
    "```"
)

_FRONTMATTER = (
    "---\n"
    "name: value\n"
    "description: Disciplined intrinsic-value investor.\n"
    "tools: [Read, Bash, WebSearch]\n"
    "model: claude-opus-4-7\n"
    "---\n"
)

_BODY_SECTIONS: dict[str, str] = {
    "MANDATE": _MANDATE,
    "RESEARCH MANDATE": _RESEARCH_MANDATE,
    "RESEARCH ACCESS": _RESEARCH_ACCESS,
    "ALLOWED ACTIONS": _ALLOWED_ACTIONS,
    "MEMORY": _MEMORY,
    "RESEARCH OUTPUT SCHEMA": _RESEARCH_OUTPUT_SCHEMA,
    "ROUND 1 OUTPUT SCHEMA": _ROUND1_SCHEMA,
    "ROUND 2 OUTPUT SCHEMA": _ROUND2_SCHEMA,
}


def _build_persona(
    frontmatter: str = _FRONTMATTER,
    sections: dict[str, str] | None = None,
    drop: str | None = None,
) -> str:
    secs = dict(_BODY_SECTIONS if sections is None else sections)
    if drop is not None:
        secs.pop(drop, None)
    body = "\n\n".join(f"## {h}\n\n{secs[h]}" for h in _BODY_SECTIONS if h in secs)
    return frontmatter + "\n" + body + "\n"


def _write(tmp_path: Path, content: str, name: str = "p.md") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# PASS cases
# ---------------------------------------------------------------------------

def test_template_passes() -> None:
    """The shipped fill-in template is itself a conforming file."""
    result = validate_persona_definition(TEMPLATE)
    assert result.ok, f"Template should pass; violations: {result.violations}"


def test_framework_doc_exists() -> None:
    assert FRAMEWORK.exists(), "_FRAMEWORK.md convention doc must exist."


def test_built_conforming_persona_passes(tmp_path: Path) -> None:
    p = _write(tmp_path, _build_persona())
    result = validate_persona_definition(p)
    assert result.ok, f"Conforming persona should pass; violations: {result.violations}"
    assert result.name == "value"


def test_required_sections_constant_has_eight() -> None:
    assert len(REQUIRED_SECTIONS) == 8


# ---------------------------------------------------------------------------
# Deliberate-failure fixtures (each violates exactly one check)
# ---------------------------------------------------------------------------

def test_fixture_missing_section_fails(tmp_path: Path) -> None:
    """Fixture 1 — drop the MEMORY section."""
    p = _write(tmp_path, _build_persona(drop="MEMORY"))
    result = validate_persona_definition(p)
    assert not result.ok
    assert any("MEMORY" in v for v in result.violations)


def test_fixture_short_in_actions_fails(tmp_path: Path) -> None:
    """Fixture 2 — ALLOWED ACTIONS lists SHORT (NFR #3 violation)."""
    secs = dict(_BODY_SECTIONS)
    secs["ALLOWED ACTIONS"] = _ALLOWED_ACTIONS + "\n- `SHORT <ticker> <weight>`"
    p = _write(tmp_path, _build_persona(sections=secs))
    result = validate_persona_definition(p)
    assert not result.ok
    assert any("SHORT" in v for v in result.violations)


def test_fixture_websearch_absent_fails(tmp_path: Path) -> None:
    """Fixture 3 — tools list omits WebSearch."""
    fm = _FRONTMATTER.replace("tools: [Read, Bash, WebSearch]", "tools: [Read, Bash]")
    p = _write(tmp_path, _build_persona(frontmatter=fm))
    result = validate_persona_definition(p)
    assert not result.ok
    assert any("WebSearch" in v for v in result.violations)


def test_fixture_missing_counterfactual_fails(tmp_path: Path) -> None:
    """Fixture 4 — ROUND 1 schema omits counterfactual_portfolio."""
    secs = dict(_BODY_SECTIONS)
    secs["ROUND 1 OUTPUT SCHEMA"] = _ROUND1_SCHEMA.replace(
        '  "counterfactual_portfolio": {"<ticker>": 0.0},\n', ""
    )
    p = _write(tmp_path, _build_persona(sections=secs))
    result = validate_persona_definition(p)
    assert not result.ok
    assert any("counterfactual_portfolio" in v for v in result.violations)


def test_fixture_thin_mandate_fails(tmp_path: Path) -> None:
    """Fixture 5 — MANDATE is a thin one-liner."""
    secs = dict(_BODY_SECTIONS)
    secs["MANDATE"] = "Value investor."
    p = _write(tmp_path, _build_persona(sections=secs))
    result = validate_persona_definition(p)
    assert not result.ok
    assert any("MANDATE" in v and "thin" in v for v in result.violations)


def test_fixture_no_ignore_clause_fails(tmp_path: Path) -> None:
    """Fixture 6 — RESEARCH MANDATE has no explicit ignore/avoid clause."""
    secs = dict(_BODY_SECTIONS)
    secs["RESEARCH MANDATE"] = (
        "Research the universe for names trading below intrinsic value. Attend to "
        "free-cash-flow yield, balance-sheet strength, and durable competitive position; "
        "weigh those against the price paid. Use fundamentals and filings via the data "
        "tools and web search for recent context to confirm the thesis each week."
    )
    p = _write(tmp_path, _build_persona(sections=secs))
    result = validate_persona_definition(p)
    assert not result.ok
    assert any("ignore" in v.lower() for v in result.violations)


def test_missing_file_fails() -> None:
    result = validate_persona_definition(Path("/nonexistent/persona.md"))
    assert not result.ok
    assert any("not found" in v.lower() for v in result.violations)


def test_only_one_violation_per_single_fault_fixture(tmp_path: Path) -> None:
    """Each single-fault fixture should trip exactly one violation (clean isolation)."""
    # WebSearch-absent
    fm = _FRONTMATTER.replace("tools: [Read, Bash, WebSearch]", "tools: [Read, Bash]")
    r = validate_persona_definition(_write(tmp_path, _build_persona(frontmatter=fm)))
    assert len(r.violations) == 1, r.violations


# Also persist the fixtures to disk for provenance / manual inspection.
def test_fixtures_written_to_disk() -> None:
    fdir = PROJECT_ROOT / "tests" / "unit" / "fixtures" / "personas"
    expected = {
        "fail_missing_section.md",
        "fail_short_in_actions.md",
        "fail_websearch_absent.md",
    }
    present = {p.name for p in fdir.glob("*.md")}
    assert expected.issubset(present), f"Missing on-disk fixtures: {expected - present}"
    # Each on-disk fixture must FAIL validation.
    for name in expected:
        r = validate_persona_definition(fdir / name)
        assert not r.ok, f"On-disk fixture {name} should fail validation."
