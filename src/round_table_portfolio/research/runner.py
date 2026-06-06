"""Per-persona agentic research runner — Component 9 (`per_persona_research`).

Harness that processes the structured JSON reply a persona subagent produces
(RESEARCH OUTPUT SCHEMA) into ledger payloads, a written report file, and
budget accounting.  The runner does NOT dispatch the subagent — that is the
orchestrating Claude Code session's job.  It receives the raw JSON string and
does everything around the dispatch:

1. Parses the RESEARCH OUTPUT SCHEMA reply into report + shortlist + counts.
2. Writes the full report to ``state/reports/<week>/<persona>.md``.
3. Records web-search + turn counts into a ``PersonaBudgetTracker`` and flags
   budget overrun (data-tool-call hard enforcement runs inside the subagent via
   ``guarded_tools.py`` — the runner only post-records the reported counts).
4. Invokes ``validate_persona_report`` with the injected on-mandate judge.
5. Returns ``PersonaResearchResult`` carrying the ``persona_reports`` payload,
   the ``persona_shortlists`` payload, and the budget summary — all in-memory.
   The orchestrator (M2 ``/weekly-run``) writes the DB rows.  The demo scaffold
   at the bottom of this module writes rows directly for AC verification only.

Subagent-dispatch seam
----------------------
The runner's public entry point ``run_persona_research`` accepts the raw
RESEARCH OUTPUT SCHEMA string that a persona subagent has already produced.
The concrete ``OnMandateJudge`` injected in production builds the judge prompt
(using ``JUDGE_PROMPT_TEMPLATE``) and delegates the actual subagent dispatch
back to the session callback.  Unit tests inject ``StubOnMandateJudge`` — no
subagent is spawned and no external service is called.

Usage::

    from round_table_portfolio.research.runner import run_persona_research
    from round_table_portfolio.personas.output_validator import StubOnMandateJudge

    result = run_persona_research(
        persona_slug="value",
        week_id="2026-06-02",
        raw_output=<json-string from subagent>,
        mandate="You research the universe hunting for...",
        judge=StubOnMandateJudge(),
        budget=get_budget(budgets, "value"),
        validator_config=load_validator_config(),
        state_root=Path("state"),
    )
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from round_table_portfolio.budget.loader import PersonaBudget
from round_table_portfolio.budget.tracker import PersonaBudgetTracker
from round_table_portfolio.personas.output_validator import (
    JUDGE_PROMPT_TEMPLATE,  # noqa: F401 — exported for session callers
    OnMandateJudge,
    ReportValidationResult,
    StubOnMandateJudge,  # noqa: F401 — re-exported for test convenience
    ValidatorConfig,
    parse_judge_response,  # noqa: F401 — re-exported for session callers
    validate_persona_report,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsed RESEARCH OUTPUT SCHEMA
# ---------------------------------------------------------------------------


@dataclass
class ShortlistEntry:
    """One name from the persona's shortlist."""

    ticker: str
    why: str
    cluster: list[str] = field(default_factory=list)


@dataclass
class PersonaOutputSchema:
    """Parsed RESEARCH OUTPUT SCHEMA produced by a persona subagent."""

    shortlist: list[ShortlistEntry]
    report: str
    web_searches_used: int
    data_tool_calls_used: int


# ---------------------------------------------------------------------------
# Ledger payload dataclasses (returned to the orchestrator — not written here)
# ---------------------------------------------------------------------------


@dataclass
class PersonaReportPayload:
    """Row payload for ``persona_reports`` — filled by the runner, written by orchestrator."""

    week_id: str
    persona: str
    summary: str
    validator_passed: int  # 0 or 1
    validator_notes: str
    full_report_path: str  # relative path string, e.g. "state/reports/2026-06-02/value.md"
    roster_version: int = 1
    enhancement_version: int = 1
    user_id: str = "andrew"


@dataclass
class PersonaShortlistRow:
    """One row payload for ``persona_shortlists``."""

    week_id: str
    persona: str
    ticker: str
    is_cluster_peer: int  # 0 = directly shortlisted; 1 = cluster peer
    parent_ticker: Optional[str]  # NULL when is_cluster_peer=0
    roster_version: int = 1
    user_id: str = "andrew"


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


@dataclass
class PersonaResearchResult:
    """Everything the runner produces for one persona in one week.

    The orchestrator consumes ``report_payload`` + ``shortlist_rows`` for DB
    writes.  The runner writes the report file; the orchestrator does not need
    to touch the filesystem.
    """

    persona_slug: str
    week_id: str
    parsed_output: PersonaOutputSchema
    validation: ReportValidationResult
    report_payload: PersonaReportPayload
    shortlist_rows: list[PersonaShortlistRow]
    budget_summary: dict  # from PersonaBudgetTracker.summary()
    budget_overrun: bool  # True if any breach detected


# ---------------------------------------------------------------------------
# Output schema parser
# ---------------------------------------------------------------------------

class PersonaOutputParseError(ValueError):
    """Raised when the raw persona output cannot be parsed into the expected schema."""


def _parse_persona_output(raw: str) -> PersonaOutputSchema:
    """Parse raw RESEARCH OUTPUT SCHEMA JSON into a typed object.

    The schema from each persona file::

        {
          "shortlist": [
            {"ticker": "<T>", "why": "<...>", "cluster": ["<peer1>", ...]}
          ],
          "report": "<full text>",
          "web_searches_used": <int>,
          "data_tool_calls_used": <int>
        }

    Raises:
        PersonaOutputParseError: On JSON decode failure, missing required keys,
            or wrong types.  Never silently accepts a malformed reply.
    """
    # Strip markdown code fences if the subagent wrapped the JSON.
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening ```json fence and the closing ``` fence.
        inner_lines = [
            ln for ln in lines[1:]
            if not ln.strip().startswith("```")
        ]
        stripped = "\n".join(inner_lines).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise PersonaOutputParseError(
            f"RESEARCH OUTPUT SCHEMA is not valid JSON: {exc}"
        ) from exc

    # Required top-level keys.
    missing = [k for k in ("shortlist", "report", "web_searches_used", "data_tool_calls_used")
               if k not in data]
    if missing:
        raise PersonaOutputParseError(
            f"Missing required keys in RESEARCH OUTPUT SCHEMA: {missing}"
        )

    shortlist_raw = data["shortlist"]
    if not isinstance(shortlist_raw, list):
        raise PersonaOutputParseError("'shortlist' must be a JSON array.")

    entries: list[ShortlistEntry] = []
    for i, item in enumerate(shortlist_raw):
        if not isinstance(item, dict):
            raise PersonaOutputParseError(f"shortlist[{i}] is not a JSON object.")
        ticker = item.get("ticker", "").strip().upper()
        why = item.get("why", "").strip()
        cluster = [str(p).strip().upper() for p in item.get("cluster", [])]
        if not ticker:
            raise PersonaOutputParseError(f"shortlist[{i}] is missing 'ticker'.")
        entries.append(ShortlistEntry(ticker=ticker, why=why, cluster=cluster))

    report = str(data["report"]).strip()

    try:
        web_searches_used = int(data["web_searches_used"])
        data_tool_calls_used = int(data["data_tool_calls_used"])
    except (TypeError, ValueError) as exc:
        raise PersonaOutputParseError(
            f"'web_searches_used' and 'data_tool_calls_used' must be integers: {exc}"
        ) from exc

    return PersonaOutputSchema(
        shortlist=entries,
        report=report,
        web_searches_used=web_searches_used,
        data_tool_calls_used=data_tool_calls_used,
    )


# ---------------------------------------------------------------------------
# Shortlist rows builder
# ---------------------------------------------------------------------------


def _build_shortlist_rows(
    parsed: PersonaOutputSchema,
    persona_slug: str,
    week_id: str,
    roster_version: int = 1,
) -> list[PersonaShortlistRow]:
    """Expand a parsed output schema into flat ``persona_shortlists`` rows.

    Each directly-shortlisted ticker → ``is_cluster_peer=0, parent_ticker=NULL``.
    Each cluster peer under that ticker → ``is_cluster_peer=1, parent_ticker=<parent>``.
    De-duplicates: if a ticker appears as both a direct entry and a cluster peer
    of another name, the direct entry wins (is_cluster_peer=0 preserved).
    """
    rows: dict[str, PersonaShortlistRow] = {}

    for entry in parsed.shortlist:
        parent = entry.ticker
        # Direct shortlist entry — wins any earlier cluster-peer entry.
        rows[parent] = PersonaShortlistRow(
            week_id=week_id,
            persona=persona_slug,
            ticker=parent,
            is_cluster_peer=0,
            parent_ticker=None,
            roster_version=roster_version,
        )
        for peer in entry.cluster:
            if peer not in rows:
                # Only add as cluster-peer if not already a direct entry.
                rows[peer] = PersonaShortlistRow(
                    week_id=week_id,
                    persona=persona_slug,
                    ticker=peer,
                    is_cluster_peer=1,
                    parent_ticker=parent,
                    roster_version=roster_version,
                )

    return list(rows.values())


# ---------------------------------------------------------------------------
# Summary extractor
# ---------------------------------------------------------------------------


def _extract_summary(report: str, max_chars: int = 500) -> str:
    """Return the first paragraph of the report as a summary, truncated to max_chars."""
    for line in report.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:max_chars]
    return report[:max_chars]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_persona_research(
    persona_slug: str,
    week_id: str,
    raw_output: str,
    mandate: str,
    judge: OnMandateJudge,
    budget: PersonaBudget,
    validator_config: ValidatorConfig,
    *,
    state_root: Path = Path("state"),
    roster_version: int = 1,
    enhancement_version: int = 1,
    user_id: str = "andrew",
) -> PersonaResearchResult:
    """Process one persona's RESEARCH OUTPUT SCHEMA reply into payloads + validation.

    This function is the harness that wraps the actual subagent dispatch. The
    caller (a Claude Code session or the M1-011 demo harness) has already
    dispatched the persona subagent and captured its raw JSON output. This
    function does all the post-dispatch work.

    Args:
        persona_slug: e.g. "value", "technical".
        week_id:      ISO date string used as the ledger key, e.g. "2026-06-02".
        raw_output:   The raw RESEARCH OUTPUT SCHEMA JSON string from the subagent.
        mandate:      The persona's RESEARCH MANDATE section text (for the judge).
        judge:        An ``OnMandateJudge`` implementation.  In tests, pass
                      ``StubOnMandateJudge()``.  In production, pass the
                      session-backed judge that dispatches the judge subagent.
        budget:       The persona's ``PersonaBudget`` (from ``get_budget``).
        validator_config: Loaded ``ValidatorConfig`` (from ``load_validator_config``).
        state_root:   Root of the ``state/`` directory.  Override in tests to
                      a tmp_path.
        roster_version:      FK value for ``persona_reports.roster_version``.
        enhancement_version: FK value for ``persona_reports.enhancement_version``.
        user_id:             Owner field, default "andrew".

    Returns:
        ``PersonaResearchResult`` with the report payload, shortlist rows, and
        budget summary.  DB writes are the orchestrator's job (see note on the
        demo-scaffold insert below).

    Raises:
        PersonaOutputParseError: If the raw output cannot be parsed.

    Failure handling (TDD §9 Gate 5 tiers):
        - Empty/degenerate shortlist → logged as Major-tier warning; result is
          still returned so the caller can decide whether to re-prompt.
        - Budget overrun → flagged in result.budget_overrun; logged as warning.
        - Malformed raw output → PersonaOutputParseError raised (caller handles).
    """
    # Step 1 — parse the raw RESEARCH OUTPUT SCHEMA.
    parsed = _parse_persona_output(raw_output)

    # Step 2 — write the report to state/reports/<week>/<persona>.md.
    report_dir = state_root / "reports" / week_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{persona_slug}.md"
    report_path.write_text(parsed.report, encoding="utf-8")
    logger.info("Report written: %s", report_path)

    # Step 3 — record budget counts; flag overrun.
    tracker = PersonaBudgetTracker(persona=persona_slug, budget=budget)
    tracker.record("web_searches", count=parsed.web_searches_used)
    tracker.record("turns", count=1)  # one subagent dispatch = one turn block
    # data_tool_calls are reported by the subagent; post-record without hard-stop
    # (hard enforcement ran inside the subagent via guarded_tools.py).
    # We record against the budget for NFR-#6 cost logging; if reported count
    # exceeds the cap that is a feasibility signal, not a runtime error here.
    tracker_for_data_tools = PersonaBudgetTracker(persona=persona_slug, budget=budget)
    data_cap = budget.max_data_tool_calls
    safe_data_calls = min(parsed.data_tool_calls_used, data_cap)
    if parsed.data_tool_calls_used > data_cap:
        logger.warning(
            "Budget breach[%s] data_tool_calls_used=%d > cap=%d — feasibility signal",
            persona_slug,
            parsed.data_tool_calls_used,
            data_cap,
        )
    tracker_for_data_tools.record("data_tool_calls", count=safe_data_calls)

    budget_overrun = (
        tracker.web_search_breach
        or parsed.data_tool_calls_used > data_cap
    )
    budget_summary = tracker.summary()
    # Merge data-tool count into the summary for completeness.
    budget_summary["data_tool_calls"] = {
        "used": parsed.data_tool_calls_used,
        "cap": data_cap,
        "exhausted": parsed.data_tool_calls_used >= data_cap,
    }

    # Step 4 — empty shortlist detection (TDD §9 failure handling: Major tier).
    if not parsed.shortlist:
        logger.warning(
            "MAJOR: persona '%s' produced an empty shortlist for week '%s'. "
            "Root-cause: mandate too vague, tools failing, or budget too tight to converge.",
            persona_slug,
            week_id,
        )

    # Step 5 — validate the report (structural gate + on-mandate judge).
    validation = validate_persona_report(
        report=parsed.report,
        mandate=mandate,
        config=validator_config,
        persona_slug=persona_slug,
        judge=judge,
    )
    logger.info(
        "Validator result for %s: passed=%s stage=%s",
        persona_slug,
        validation.passed,
        validation.stage,
    )

    # Step 6 — build ledger payloads (returned; NOT written here per §1.1).
    report_payload = PersonaReportPayload(
        week_id=week_id,
        persona=persona_slug,
        summary=_extract_summary(parsed.report),
        validator_passed=1 if validation.passed else 0,
        validator_notes=validation.notes,
        full_report_path=str(report_path),
        roster_version=roster_version,
        enhancement_version=enhancement_version,
        user_id=user_id,
    )

    shortlist_rows = _build_shortlist_rows(
        parsed=parsed,
        persona_slug=persona_slug,
        week_id=week_id,
        roster_version=roster_version,
    )

    return PersonaResearchResult(
        persona_slug=persona_slug,
        week_id=week_id,
        parsed_output=parsed,
        validation=validation,
        report_payload=report_payload,
        shortlist_rows=shortlist_rows,
        budget_summary=budget_summary,
        budget_overrun=budget_overrun,
    )


# ---------------------------------------------------------------------------
# DEMO SCAFFOLD — single-persona DB insert for AC #2 verification
# ---------------------------------------------------------------------------
# This insert path exists SOLELY to satisfy the M1-010 Demo Artifact AC:
# "Output rows write correctly to persona_reports + persona_shortlists."
# It is NOT the production write path. The production path is the M2
# /weekly-run orchestrator consuming the runner's returned payloads.
# Label preserved so it cannot be mistaken for the orchestrator writer.
# ---------------------------------------------------------------------------


def demo_scaffold_insert(
    result: PersonaResearchResult,
    db_path: Path,
) -> None:
    """DEMO SCAFFOLD — write one persona's result to the ledger.

    Used only by the M1-010 single-persona demo run. The production orchestrator
    (M2 /weekly-run) writes rows using its own writer, not this function.

    Inserts / replaces:
    - One row into ``persona_reports``.
    - One row per ticker into ``persona_shortlists``.

    Ensures the ``weeks`` row for the week exists first (idempotent insert)
    so the FK constraint does not fire on the demo run.

    Args:
        result:  The ``PersonaResearchResult`` from ``run_persona_research``.
        db_path: Path to the SQLite ledger (e.g. ``Path("state/ledger.db")``).
    """
    rp = result.report_payload

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # Ensure the week row exists so the FK is satisfied.
        conn.execute(
            "INSERT OR IGNORE INTO weeks (week_id, run_date, user_id) VALUES (?, date('now'), ?)",
            (rp.week_id, rp.user_id),
        )

        # persona_reports insert.
        conn.execute(
            """
            INSERT OR REPLACE INTO persona_reports
              (week_id, persona, summary, validator_passed, validator_notes,
               full_report_path, user_id, roster_version, enhancement_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rp.week_id,
                rp.persona,
                rp.summary,
                rp.validator_passed,
                rp.validator_notes,
                rp.full_report_path,
                rp.user_id,
                rp.roster_version,
                rp.enhancement_version,
            ),
        )

        # persona_shortlists insert (one row per ticker).
        for row in result.shortlist_rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO persona_shortlists
                  (week_id, persona, ticker, is_cluster_peer, parent_ticker,
                   user_id, roster_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.week_id,
                    row.persona,
                    row.ticker,
                    row.is_cluster_peer,
                    row.parent_ticker,
                    row.user_id,
                    row.roster_version,
                ),
            )

        conn.commit()

    logger.info(
        "DEMO SCAFFOLD: inserted persona_reports + %d shortlist rows for %s/%s",
        len(result.shortlist_rows),
        result.week_id,
        result.persona_slug,
    )
