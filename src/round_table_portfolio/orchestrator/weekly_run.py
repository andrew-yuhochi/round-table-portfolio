"""weekly_run_orchestrator — Component 12, the sole ledger writer.

The spine of M3.  Sequences ONE week end-to-end (Round 1 + Round 2) and is the
ONLY writer of:
  weeks, portfolios, holdings, agent_stances, transcripts,
  persona_reports, persona_shortlists,
  state/memory/, state/reports/, state/debates/

Entry point::

    from round_table_portfolio.orchestrator.weekly_run import run_weekly

    result = run_weekly(
        project="round-table-portfolio",
        week_id="2026-W23",
        persona_replies={"value": "<raw JSON>", ...},  # 7 entries
        founder_reply="approve",
        judge=StubOnMandateJudge(),
        round2_dispatcher=my_stub_dispatcher,  # callable(outlier_slug, counterarg_block) -> str
    )

Engine / session split (TDD §1.1)
----------------------------------
- The Claude session dispatches each persona subagent and captures raw JSON.
- ``run_weekly`` receives those raw strings via ``persona_replies`` and does all
  post-dispatch work: parse → validate → debate-set → Round-1 stances →
  Round-2 (2 outlier dispatches via injected ``round2_dispatcher``) →
  re-synthesis → consensus → ledger write → transcript → memory write-back →
  metrics.
- In tests, inject canned ``persona_replies`` + a ``StubOnMandateJudge`` +
  a stub ``round2_dispatcher`` returning canned JSON + set ``STUB_ALLOW=1``.
  No live subagent is spawned.
- In production (TASK-M3-006), the session collects real replies and passes them
  in; ``round2_dispatcher`` is backed by real Agent dispatches.

Round-2 dispatch seam (TDD §1.1 M3 change)
--------------------------------------------
``round2_dispatcher`` is a callable with signature:
    (outlier_slug: str, counterargument_block: str) -> str
It receives the outlier persona slug and the assembled counterargument text
(Component 23 output), and must return the raw Round-2 JSON reply string.
The live session backs this with a real Agent dispatch; tests back it with a
stub that returns canned JSON.  This mirrors the M2 ``persona_replies`` seam
exactly — Python cannot spawn Claude Code subagents; the session does it.
Exactly 2 calls are made (one per selected outlier) — the cost contract.

Transaction boundary (TDD §1.5)
--------------------------------
All 8 portfolios (1 FINAL consensus + 7 named counterfactuals) + their holdings
+ round=1 agent_stances + round=2 agent_stances (2 outliers) + persona_reports
+ persona_shortlists + weeks + transcripts (R1+R2) are written in ONE SQLite
transaction.  Any write failure triggers ROLLBACK — zero rows for that week_id
survive.  Memory write-back happens AFTER the commit so memory never references
a week that does not exist in the ledger.
The canonical 8th portfolio is the post-Round-2 FINAL consensus; the Round-1
provisional consensus is recorded in the transcript only (no duplicate portfolio
row).
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from round_table_portfolio.budget.loader import PersonaBudget, get_budget, load_budgets
from round_table_portfolio.personas.output_validator import (
    OnMandateJudge,
    StubOnMandateJudge,
    ValidatorConfig,
    load_validator_config,
    persist_validator_claim,
    validate_counterfactual_portfolio,
    validate_persona_report,
)
from round_table_portfolio.research.runner import PersonaResearchResult, run_persona_research

# Component 13 + 14 — real implementations (TASK-M2-004).
from round_table_portfolio.orchestrator.round1 import (
    AgentStancePayload,       # noqa: F401 — shape used by tests
    Round1Capture,            # noqa: F401 — shape used by tests
    capture_round1_stances,
    construct_debate_set,
)

# Component 16 — real implementation (TASK-M2-006).
from round_table_portfolio.portfolio.consensus import blend_consensus

# Component 15 — real implementation (TASK-M2-005).
from round_table_portfolio.portfolio.materialize import (
    HoldingPayload,      # noqa: F401 — shape used by tests
    PortfolioPayload,    # noqa: F401 — shape used by tests
    materialize_portfolios,
)

# Component 17 — real implementation (TASK-M2-007 / M3 extended).
from round_table_portfolio.orchestrator.transcript import (
    append_round2_transcript,
    write_round1_transcript,
)

# Components 21 + 22 — real implementation (TASK-M3-001).
from round_table_portfolio.orchestrator.dissent import (
    DissentResult,    # noqa: F401 — shape used by tests
    OutlierSelection, # noqa: F401 — shape used by tests
    compute_dissent,
    load_dissent_config,
    select_outliers,
)

# Component 23 — real implementation (TASK-M3-002).
from round_table_portfolio.orchestrator.counterargument import (
    CounterargumentBlock,  # noqa: F401 — shape used by tests
    assemble_counterarguments,
    load_counterargument_config,
)

# Component 24 — real implementation (TASK-M3-003).
from round_table_portfolio.orchestrator.round2 import (
    Round2Reply,        # noqa: F401 — shape used by tests
    capture_round2_stances,
)

# Component 25 — real implementation (TASK-M3-004).
from round_table_portfolio.orchestrator.resynthesis import (
    ResynthesisResult,  # noqa: F401 — shape used by tests
    resynthesize_consensus,
)

# Component 18 — real implementation (TASK-M2-008).
from round_table_portfolio.orchestrator.memory import writeback_memory

# Component 19 — real implementation (TASK-M2-009).
from round_table_portfolio.orchestrator.metrics import (
    RunMetricsReport,    # noqa: F401 — shape used by tests
    report_run_metrics,
)

# All sibling-task stubs have now been replaced by real implementations.
# persist_validator_claim — replaced by TASK-M2-010 (real impl in
#   personas/output_validator.py, imported above).

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

_PERSONAS_CONFIG = Path(os.environ.get("PERSONAS_CONFIG", "config/personas.yaml"))
_THRESHOLDS_CONFIG = Path(os.environ.get("THRESHOLDS_CONFIG", "config/thresholds.yaml"))
_BUDGET_CONFIG = Path(os.environ.get("BUDGET_CONFIG", "config/persona_budgets.yaml"))
_VALIDATOR_CONFIG = Path(os.environ.get("VALIDATOR_CONFIG", "config/validator.yaml"))
_WEB_SEARCH_CONFIG = Path(os.environ.get("WEB_SEARCH_CONFIG", "config/web_search.yaml"))

_PERSONA_SLUGS_7 = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]


def _load_persona_slugs(config_path: Optional[Path] = None) -> list[str]:
    path = config_path or _PERSONAS_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    slugs = raw.get("slugs", [])
    if not slugs:
        raise ValueError(f"personas.yaml at {path} contains no slugs.")
    return list(slugs)


def _load_thresholds(config_path: Optional[Path] = None) -> dict[str, Any]:
    path = config_path or _THRESHOLDS_CONFIG
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_web_search_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    path = config_path or _WEB_SEARCH_CONFIG
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# Week-label resolver
# ---------------------------------------------------------------------------


def _resolve_week_id(week_id: Optional[str]) -> str:
    """Return the canonical ISO week label for the run.

    Format: YYYY-WNN  (e.g. "2026-W23").
    If week_id is given it is returned as-is (caller's responsibility to
    validate format).  Otherwise the current calendar week is used.
    """
    if week_id:
        return week_id
    today = datetime.date.today()
    iso = today.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


# ---------------------------------------------------------------------------
# Approve / override parser
# ---------------------------------------------------------------------------

_APPROVE_RE = re.compile(
    r"^\s*(approve[ds]?|looks?\s+good|yes|confirm[ed]*|go\s+ahead)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_OVERRIDE_RE = re.compile(r"^\s*override\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)


def _parse_founder_reply(reply: str) -> tuple[str, str]:
    """Parse a founder reply into (decision_type, delta).

    Returns:
        ("panel_approved", "")          — unambiguous approve
        ("founder_override", "<delta>") — unambiguous override with delta text
        ("ambiguous", "")               — neither pattern matched

    The caller is responsible for re-prompting on "ambiguous" (up to one time
    per TDD §1.5 / AC #4).
    """
    stripped = reply.strip()
    if _APPROVE_RE.match(stripped):
        return "panel_approved", ""
    m = _OVERRIDE_RE.match(stripped)
    if m:
        delta = m.group(1).strip()
        return "founder_override", delta
    return "ambiguous", ""


def _parse_reply_with_reprompt(
    initial_reply: str,
    reprompt_fn: Any,  # callable() -> str; the re-prompt callback
) -> tuple[str, str]:
    """Attempt to resolve a founder reply, re-prompting ONCE on ambiguity.

    Args:
        initial_reply: The first reply text the founder provided.
        reprompt_fn:   A zero-argument callable that returns the founder's
                       second reply.  Called at most once.

    Returns:
        (decision_type, delta) — guaranteed not "ambiguous" after re-prompt;
        if still ambiguous after one re-prompt, defaults to "panel_approved"
        with a warning log (safe default per §1.5: do not abort on ambiguity).
    """
    decision_type, delta = _parse_founder_reply(initial_reply)
    if decision_type != "ambiguous":
        return decision_type, delta

    logger.info(
        "Founder reply ambiguous (%r); issuing one re-prompt.", initial_reply[:80]
    )
    second_reply = reprompt_fn()
    decision_type, delta = _parse_founder_reply(second_reply)

    if decision_type == "ambiguous":
        logger.warning(
            "Founder reply still ambiguous after re-prompt (%r); "
            "defaulting to panel_approved.",
            second_reply[:80],
        )
        return "panel_approved", ""

    return decision_type, delta


# ---------------------------------------------------------------------------
# Ledger writer
# ---------------------------------------------------------------------------


def _write_week_transaction(
    conn: sqlite3.Connection,
    *,
    week_id: str,
    run_date: str,
    portfolio_payloads: list[Any],
    stances: list[Any],
    round2_stances: list[Any],
    persona_results: list[PersonaResearchResult],
    transcript_path: Path,
    transcript_summary: str,
    transcript_vote_tally: str,
    transcript_key_contention: str,
    decision_type: str,
    decision_delta: str,
    user_id: str = "andrew",
) -> None:
    """Write all rows for one week inside the caller's transaction.

    Writes round=1 stances (all 7 personas) + round=2 stances (2 outliers) in
    one atomic block.  The caller must have already called conn.execute("BEGIN")
    and must call conn.commit() or conn.rollback() afterwards.

    Raises:
        sqlite3.Error: On any constraint violation or I/O error.  The caller
            rolls back on any exception from this function.
    """
    # weeks row.
    conn.execute(
        "INSERT OR IGNORE INTO weeks (week_id, run_date, notes, user_id) "
        "VALUES (?, ?, ?, ?)",
        (week_id, run_date, f"decision={decision_type}", user_id),
    )

    # 8 portfolios + their holdings.
    for pp in portfolio_payloads:
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO portfolios
              (week_id, type, user_id, roster_version, enhancement_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pp.week_id, pp.type, pp.user_id, pp.roster_version, pp.enhancement_version, created_at),
        )
        portfolio_id = conn.execute(
            "SELECT portfolio_id FROM portfolios WHERE week_id=? AND type=? AND user_id=?",
            (pp.week_id, pp.type, pp.user_id),
        ).fetchone()[0]

        for h in pp.holdings:
            conn.execute(
                """
                INSERT INTO holdings
                  (portfolio_id, ticker, weight, action, entry_date, user_id, roster_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    portfolio_id,
                    h.ticker,
                    h.weight,
                    h.action,
                    h.entry_date,
                    h.user_id,
                    h.roster_version,
                ),
            )

    # agent_stances rows — round=1 (all 7 personas) + round=2 (2 outliers).
    for s in stances + round2_stances:
        conn.execute(
            """
            INSERT INTO agent_stances
              (week_id, persona, ticker, round, action, target_weight, confidence,
               rationale, user_id, roster_version, enhancement_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                s.week_id,
                s.persona,
                s.ticker,
                s.round,
                s.action,
                s.target_weight,
                s.confidence,
                s.rationale,
                s.user_id,
                s.roster_version,
                s.enhancement_version,
            ),
        )

    # persona_reports + persona_shortlists rows.
    for res in persona_results:
        rp = res.report_payload
        conn.execute(
            """
            INSERT INTO persona_reports
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
        for row in res.shortlist_rows:
            conn.execute(
                """
                INSERT INTO persona_shortlists
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

    # transcripts row (file already written before the transaction).
    conn.execute(
        """
        INSERT INTO transcripts
          (week_id, summary, vote_tally, key_contention, full_log_path, user_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            week_id,
            transcript_summary,
            transcript_vote_tally,
            transcript_key_contention,
            str(transcript_path),
            user_id,
        ),
    )


def _vote_tally_json(stances: list[Any], debate_set: list[str]) -> str:
    """Build the vote_tally JSON string from round-1 stances."""
    import json

    tally: dict[str, dict[str, int]] = {}
    for ticker in debate_set:
        counts: dict[str, int] = {"add": 0, "reduce": 0, "hold": 0, "exit": 0}
        for s in stances:
            if s.ticker == ticker:
                counts[s.action] = counts.get(s.action, 0) + 1
        tally[ticker] = counts
    return json.dumps(tally)


# ---------------------------------------------------------------------------
# WeeklyRunResult
# ---------------------------------------------------------------------------


@dataclass
class WeeklyRunResult:
    """Everything produced by a single run_weekly call.

    All ledger rows have been written (or rolled back on failure).
    M3 additions: num_round2_stances, resynthesis, provisional_weights.
    """

    week_id: str
    decision_type: str            # "panel_approved" | "founder_override"
    decision_delta: str           # empty string unless founder_override
    debate_set: list[str]
    num_portfolios_written: int   # should always be 8 on success
    num_stances_written: int      # 7 × |debate_set| round-1 stances
    num_round2_stances: int       # 2 outliers × their debate-set stances
    num_persona_reports: int      # 7
    transcript_path: Optional[Path]
    metrics: Any                  # RunMetricsReport from metrics.py (Component 19)
    dissent: Any                  # DissentResult from dissent.py (Component 21)
    outliers: Any                 # OutlierSelection from dissent.py (Component 22)
    resynthesis: Any              # ResynthesisResult from resynthesis.py (Component 25)
    provisional_weights: dict     # Round-1 consensus weights (for transcript/metrics)
    persona_results: list[PersonaResearchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_weekly(
    project: str,
    *,
    week_id: Optional[str] = None,
    persona_replies: Optional[dict[str, str]] = None,
    round1_replies: Optional[dict[str, str]] = None,
    founder_reply: Optional[str] = None,
    reprompt_fn: Optional[Any] = None,
    judge: Optional[OnMandateJudge] = None,
    # Round-2 dispatch seam (M3 — mirrors the M2 persona_replies seam).
    # Signature: (outlier_slug: str, counterargument_block: str) -> str (raw JSON).
    # Live session backs this with a real Agent dispatch.
    # Tests back this with a stub returning canned round-2 JSON.
    # When None, no Round-2 is performed (backward-compatible for legacy tests).
    round2_dispatcher: Optional[Any] = None,
    # Per-phase timing maps for full-cycle metrics (M3 DEF-003).
    # When None, treated as {} — that phase's time contributes 0 to the total.
    per_round1_timing: Optional[dict[str, float]] = None,
    per_judge_timing: Optional[dict[str, float]] = None,
    # Config overrides (for testing).
    personas_config: Optional[Path] = None,
    budget_config: Optional[Path] = None,
    validator_config_obj: Optional[ValidatorConfig] = None,
    validator_config_path: Optional[Path] = None,
    thresholds_config: Optional[Path] = None,
    web_search_config: Optional[Path] = None,
    # Paths (for testing — override default state/ and ledger locations).
    state_root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    user_id: str = "andrew",
) -> WeeklyRunResult:
    """Run the full single-week cycle (Round 1 + Round 2 + re-synthesis).

    Args:
        project:          Project name (informational; used in log messages).
        week_id:          ISO week label e.g. "2026-W23".  Derived from current
                          date when not provided.
        persona_replies:  Mapping of persona_slug → raw RESEARCH OUTPUT SCHEMA
                          JSON string.  When None (production live run), the
                          caller (session) must have dispatched each persona
                          and provide the replies here.  In tests, pass canned
                          JSON strings.
        round1_replies:   Mapping of persona_slug → raw ROUND 1 OUTPUT SCHEMA
                          JSON string.  The session captures these after dispatching
                          each persona over the debate set (commit-before-reveal).
                          When None, falls back to ``persona_replies`` for
                          backward-compatible test runs where both phases use the
                          same canned JSON.
        founder_reply:    The founder's approve/override text.  When None,
                          ``reprompt_fn`` is called to obtain it (production
                          interactive path).  In tests, pass the reply directly.
        reprompt_fn:      Zero-argument callable that returns the founder's reply
                          interactively.  Used only when founder_reply is None,
                          or for the second re-prompt on ambiguity.
        judge:            OnMandateJudge implementation.  Pass StubOnMandateJudge
                          in tests.  In production the session wires a real judge.
        round2_dispatcher: Optional callable (outlier_slug: str, counterarg: str) → str.
                          The seam for the 2 Round-2 outlier dispatches.  The live
                          session backs this with a real Agent dispatch; tests use a
                          stub returning canned JSON.  When None, Round-2 is skipped
                          (backward-compatible path — provisional R1 consensus is used
                          as the final consensus, no round=2 stances are written).
        per_round1_timing: Dict[slug, float] — wall-clock seconds per persona for
                          Round-1 dispatch.  None → treated as {}, contributing 0s
                          to the full-cycle total (DEF-003).
        per_judge_timing:  Dict[slug, float] — wall-clock seconds per persona for
                          judge dispatch.  None → treated as {}, contributing 0s.
        personas_config:  Override path to personas.yaml (for tests).
        budget_config:    Override path to persona_budgets.yaml (for tests).
        validator_config_obj: Pre-built ValidatorConfig (for tests — avoids file I/O).
        validator_config_path: Override path to validator.yaml (for tests).
        thresholds_config: Override path to thresholds.yaml (for tests).
        state_root:       Override for state/ directory (for tests — use tmp_path).
        db_path:          Override for ledger DB path (for tests — use tmp_path).
        user_id:          Owner field; default "andrew".

    Returns:
        WeeklyRunResult with the week_id, decision, row counts, and M3 outputs.

    Raises:
        RuntimeError:   If any persona's research/validation fails (per TDD §1.5
                        abort rule — no partial runs at PoC), or if
                        capture_round2_stances receives != 2 replies.
        sqlite3.Error:  If the ledger transaction fails (caller sees the rolled-
                        back exception; zero rows for this week_id survive).
    """
    # ------------------------------------------------------------------
    # 0. Resolve paths and load configs.
    # ------------------------------------------------------------------
    _state = state_root or Path("state")
    _db = db_path or (_state / "ledger.db")
    _state.mkdir(parents=True, exist_ok=True)
    (_state / "runs").mkdir(parents=True, exist_ok=True)

    week_label = _resolve_week_id(week_id)
    run_date = datetime.date.today().isoformat()
    logger.info("[%s] Starting weekly run for week=%s", project, week_label)

    persona_slugs = _load_persona_slugs(personas_config)
    budgets = load_budgets(budget_config)
    thresholds = _load_thresholds(thresholds_config)
    max_position_weight: float = float(thresholds.get("max_position_weight", 0.20))
    _ws_cfg = _load_web_search_config(web_search_config)
    window_config = {
        "window_hours": float(_ws_cfg.get("window_hours", 5.0)),
        "window_proximity_threshold": float(_ws_cfg.get("window_proximity_threshold", 0.80)),
    }

    if validator_config_obj is not None:
        v_config = validator_config_obj
    else:
        v_config = load_validator_config(validator_config_path)

    _judge = judge or StubOnMandateJudge()

    # ------------------------------------------------------------------
    # 1. Per-persona research + validation + persist claim.
    # ------------------------------------------------------------------
    persona_results: list[PersonaResearchResult] = []
    # Timing map for Component 19 — measured around each dispatch (Python
    # cannot time across subagent boundaries from inside a helper; the
    # orchestrator is the only place that can do this).
    per_persona_timing: dict[str, float] = {}
    for slug in persona_slugs:
        raw = (persona_replies or {}).get(slug)
        if not raw:
            raise RuntimeError(
                f"No persona reply provided for slug={slug!r}. "
                "In production, the session must dispatch the persona subagent "
                "and pass the raw JSON reply via persona_replies."
            )

        budget: PersonaBudget = get_budget(budgets, slug)

        # Mandate: the orchestrator passes an empty string in tests / stubs;
        # the production session reads the mandate from the agent file.
        mandate = ""

        _persona_t0 = time.time()
        result = run_persona_research(
            persona_slug=slug,
            week_id=week_label,
            raw_output=raw,
            mandate=mandate,
            judge=_judge,
            budget=budget,
            validator_config=v_config,
            state_root=_state,
            user_id=user_id,
        )
        per_persona_timing[slug] = time.time() - _persona_t0

        if result.budget_overrun:
            logger.warning(
                "[%s] Budget overrun for persona=%s — feasibility signal.",
                project, slug,
            )

        # Layer-2 validation with counterfactual_portfolio (if extractable from
        # the stub/real round1 output — not yet available here; deferred to after
        # capture_round1_stances).  The structural + judge gate already ran
        # inside run_persona_research above.

        # Persist validator claim — Component 20 (real impl, TASK-M2-010).
        persist_validator_claim(result.validation, week_label, slug, state_root=_state)

        persona_results.append(result)
        logger.info("[%s] Persona %s: validation=%s", project, slug, result.validation.passed)

    # ------------------------------------------------------------------
    # 2. Form the debate set (Component 13 — TASK-M2-004 real implementation).
    # ------------------------------------------------------------------
    _budget_raw = yaml.safe_load(
        (budget_config or _BUDGET_CONFIG).read_text(encoding="utf-8")
    ) or {}
    debate_cfg: dict[str, Any] = {
        "debate_set_ceiling": _budget_raw.get("debate_set_ceiling", 40),
        "max_position_weight": max_position_weight,
    }
    debate_set = construct_debate_set(persona_results, debate_cfg)
    logger.info("[%s] Debate set: %d tickers", project, len(debate_set))

    # ------------------------------------------------------------------
    # 3. Round-1 stances + counterfactuals (Component 14 — TASK-M2-004 real).
    #    Layer-2 validation of each counterfactual runs here.
    # ------------------------------------------------------------------
    # In production the session dispatches each persona over the debate set
    # (commit-before-reveal) and provides the raw Round-1 JSON strings via
    # round1_replies.  In tests, round1_replies falls back to persona_replies
    # so existing canned fixtures work without modification.
    _r1_replies: dict[str, str] = round1_replies or persona_replies or {}
    round1 = capture_round1_stances(
        debate_set,
        persona_results,
        raw_round1_replies=_r1_replies,
        config={"max_position_weight": max_position_weight},
    )

    # Layer-2 validate each counterfactual (M2-002 portfolio-arithmetic gate only).
    # Layer-1 already ran the report-prose structural + on-mandate gates on the
    # FULL report text inside run_persona_research.  Layer-2's sole job is the
    # fully-invested / CASH / per-position-cap invariant on the Round-1 portfolio.
    # Passing the truncated summary to validate_persona_report re-ran the
    # structural ticker gate on 500 chars and false-failed personas whose opening
    # paragraph named fewer than 2 tickers (TASK-M2-011 bug).
    for slug, portfolio in round1.counterfactuals.items():
        validation = validate_counterfactual_portfolio(
            counterfactual_portfolio=portfolio,
            max_position_weight=max_position_weight,
        )
        if not validation.passed:
            raise RuntimeError(
                f"Layer-2 counterfactual validation FAILED for persona={slug!r}: "
                f"{validation.notes}"
            )

    # ------------------------------------------------------------------
    # 4. Round-1 provisional consensus (Component 16 — blend_consensus).
    #    This is the PROVISIONAL consensus used as the Round-2 baseline.
    # ------------------------------------------------------------------
    provisional_weights = blend_consensus(
        round1.stances,
        config={"max_position_weight": max_position_weight},
    )

    # ------------------------------------------------------------------
    # 5. Steps 6a–6e: Round-2 phase (M3).  If no round2_dispatcher is
    #    provided, skip Round-2 and use the provisional consensus as final.
    # ------------------------------------------------------------------

    # 6a. Compute recalibrated dissent score (Component 21).
    dissent_cfg = load_dissent_config(thresholds_config)
    dissent_result = compute_dissent(round1.stances, debate_set, dissent_cfg)

    # 6b. Select the 2 most-divergent personas (Component 22).
    outliers = select_outliers(dissent_result, round1.stances, dissent_cfg)
    logger.info(
        "[%s] Dissent: score=%.4f contested=%s outliers=%s",
        project, dissent_result.dissent_score, dissent_result.contested_week,
        outliers.selected,
    )

    # Collect R1 rationales for Component 23.
    # rationales[persona][ticker] = rationale text from round1 stances.
    def _get_attr(obj: Any, attr: str) -> Any:
        return obj[attr] if isinstance(obj, dict) else getattr(obj, attr)

    r1_rationales: dict[str, dict[str, str]] = {}
    for s in round1.stances:
        persona = _get_attr(s, "persona")
        ticker = _get_attr(s, "ticker")
        rationale = _get_attr(s, "rationale")
        r1_rationales.setdefault(persona, {})[ticker] = rationale

    # 6c. Assemble counterarguments deterministically (Component 23 — NO dispatch).
    ca_cfg = load_counterargument_config(thresholds_config)
    counterargument_blocks = assemble_counterarguments(
        outlier_slugs=outliers.selected,
        all_stances=round1.stances,
        rationales=r1_rationales,
        config=ca_cfg,
    )

    # 6d. Dispatch 2 Round-2 outliers (seam — session backs with real Agent calls;
    #     tests back with a stub).  Exactly 2 calls — the cost contract.
    round2_raw_replies: dict[str, str] = {}
    per_round2_timing: dict[str, float] = {}

    if round2_dispatcher is not None:
        for outlier_slug in outliers.selected:
            cb = counterargument_blocks[outlier_slug]
            _r2_t0 = time.time()
            raw_r2 = round2_dispatcher(outlier_slug, cb.block)
            per_round2_timing[outlier_slug] = time.time() - _r2_t0
            round2_raw_replies[outlier_slug] = raw_r2
            logger.info(
                "[%s] Round-2 reply captured for outlier=%s (%.1fs)",
                project, outlier_slug, per_round2_timing[outlier_slug],
            )

    # Parse + build round=2 AgentStancePayload rows (Component 24).
    round2_stances_all: list[Any] = []
    round2_replies: dict[str, Any] = {}
    if round2_raw_replies:
        r2_captures = capture_round2_stances(
            round2_raw_replies,
            week_id=week_label,
            config={"max_position_weight": max_position_weight},
            user_id=user_id,
        )
        for slug, (r2_reply, r2_payloads) in r2_captures.items():
            round2_stances_all.extend(r2_payloads)
            round2_replies[slug] = r2_reply

    # 6e. Re-synthesize consensus via SAME blend_consensus over updated stance set
    #     (Component 25).  If no Round-2 ran, use the provisional weights as final.
    if round2_stances_all:
        resynth = resynthesize_consensus(
            round1_stances=round1.stances,
            round2_payloads=round2_stances_all,
            outlier_personas=set(outliers.selected),
            provisional_weights=provisional_weights,
            config={"max_position_weight": max_position_weight},
        )
        final_weights = resynth.final_weights
    else:
        # No Round-2 (no dispatcher provided) — provisional is final.
        from round_table_portfolio.orchestrator.resynthesis import ResynthesisResult
        resynth = ResynthesisResult(
            final_weights=provisional_weights,
            provisional_weights=provisional_weights,
            delta={},
            merged_stances=list(round1.stances),
            outlier_personas=frozenset(),
        )
        final_weights = provisional_weights

    # ------------------------------------------------------------------
    # 6. Materialize portfolios (Component 15) using the FINAL consensus weights.
    # ------------------------------------------------------------------
    portfolio_payloads = materialize_portfolios(
        round1.counterfactuals,
        final_weights,
        prior_portfolios=None,   # first week — no prior; M4+ will supply this
        week_id=week_label,
        entry_date=run_date,
        config={
            "max_position_weight": max_position_weight,
            "debate_set": debate_set,
        },
    )

    # Per-ticker σ (new signed-score basis) feeds the transcript dissent note.
    std_devs = dissent_result.per_ticker_sigma
    vote_tally = _vote_tally_json(round1.stances, debate_set)

    # ------------------------------------------------------------------
    # 7. Write transcript FILE first (atomic — so the path exists before
    #    the DB row is written, satisfying full_log_path NOT NULL).
    #    7a: Round-1 section  →  7b: Round-2 section appended.
    # ------------------------------------------------------------------
    # Resolve founder decision before writing transcript.
    if founder_reply is not None:
        _initial_reply = founder_reply
    elif reprompt_fn is not None:
        _initial_reply = reprompt_fn()
    else:
        raise RuntimeError(
            "run_weekly requires either founder_reply or reprompt_fn. "
            "In tests, pass founder_reply='approve' (or 'override: <delta>'). "
            "In production, the session captures the founder's conversational reply."
        )

    def _second_reprompt() -> str:
        if reprompt_fn is not None:
            return reprompt_fn()
        return "approve"

    decision_type, decision_delta = _parse_reply_with_reprompt(
        _initial_reply, _second_reprompt
    )

    transcript_path = write_round1_transcript(
        round1,
        provisional_weights,
        std_devs,
        decision_type,
        week_id=week_label,
        state_root=_state,
    )

    # 7b. Append Round-2 section if Round-2 was performed.
    if round2_stances_all:
        append_round2_transcript(
            transcript_path,
            dissent_result=dissent_result,
            outliers=outliers,
            counterargument_blocks=counterargument_blocks,
            round2_replies=round2_replies,
            resynthesis_result=resynth,
        )

    transcript_summary = (
        f"R1+R2 consensus for {week_label}. Decision: {decision_type}."
        if round2_stances_all
        else f"Round-1 consensus for {week_label}. Decision: {decision_type}."
    )

    # ------------------------------------------------------------------
    # 8. ONE-TRANSACTION ledger write.  Roll back ALL rows if any write fails.
    #    Writes: round=1 stances + round=2 stances + FINAL consensus portfolio
    #    + 7 counterfactuals + persona_reports + weeks + transcripts.
    # ------------------------------------------------------------------
    conn = sqlite3.connect(str(_db))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN")
        _write_week_transaction(
            conn,
            week_id=week_label,
            run_date=run_date,
            portfolio_payloads=portfolio_payloads,
            stances=round1.stances,
            round2_stances=round2_stances_all,
            persona_results=persona_results,
            transcript_path=transcript_path,
            transcript_summary=transcript_summary,
            transcript_vote_tally=vote_tally,
            transcript_key_contention=(
                f"dissent_score={dissent_result.dissent_score:.4f} "
                f"contested={dissent_result.contested_week} "
                f"outliers={outliers.selected}"
            ),
            decision_type=decision_type,
            decision_delta=decision_delta,
            user_id=user_id,
        )
        conn.commit()
        logger.info("[%s] Ledger write committed for week=%s", project, week_label)
    except Exception:
        conn.rollback()
        logger.error(
            "[%s] Ledger write FAILED — rolling back all rows for week=%s",
            project, week_label,
        )
        raise
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # 9. Memory write-back AFTER commit (TASK-M2-008 real implementation).
    #    Memory must not advance if the ledger rolled back (TDD §1.5).
    #    Correctness invariant: this block is unreachable on any rolled-back
    #    path — the transaction block above re-raises on failure, so execution
    #    never reaches here unless conn.commit() succeeded.
    # ------------------------------------------------------------------
    writeback_memory(
        round1,
        round1.counterfactuals,
        persona_results,
        {},                  # resolved_alpha: empty at PoC (no closed windows yet)
        memory_dir=_state / "memory",
        archive_dir=_state / "memory" / "archive",
    )

    # ------------------------------------------------------------------
    # 10. Metrics — Component 19 (M3 DEF-003: full-cycle timing).
    #     Passes all 4 timing maps so the total covers research + R1 + judges
    #     + Round-2.  run_log_path is written with a summary preamble first;
    #     report_run_metrics appends the full report below it.
    # ------------------------------------------------------------------
    run_log_path = _state / "runs" / f"{week_label}.log"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_log_path.write_text(
        f"week={week_label}\n"
        f"decision={decision_type}\n"
        f"delta={decision_delta!r}\n"
        f"portfolios={len(portfolio_payloads)}\n"
        f"stances={len(round1.stances)}\n"
        f"round2_stances={len(round2_stances_all)}\n"
        f"persona_reports={len(persona_results)}\n"
        f"debate_set_size={len(debate_set)}\n",
        encoding="utf-8",
    )
    metrics = report_run_metrics(
        per_persona_timing=per_persona_timing,
        research_results=persona_results,
        budgets=budgets,
        window_config=window_config,
        run_log_path=run_log_path,
        per_round1_timing=per_round1_timing or {},
        per_judge_timing=per_judge_timing or {},
        per_round2_timing=per_round2_timing,
    )

    return WeeklyRunResult(
        week_id=week_label,
        decision_type=decision_type,
        decision_delta=decision_delta,
        debate_set=debate_set,
        num_portfolios_written=len(portfolio_payloads),
        num_stances_written=len(round1.stances),
        num_round2_stances=len(round2_stances_all),
        num_persona_reports=len(persona_results),
        transcript_path=transcript_path,
        metrics=metrics,
        dissent=dissent_result,
        outliers=outliers,
        resynthesis=resynth,
        provisional_weights=provisional_weights,
        persona_results=persona_results,
    )
