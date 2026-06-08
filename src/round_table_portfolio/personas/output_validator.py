"""Per-persona output validator — Component 11.

Hybrid validation: a cheap deterministic structural gate runs first; only
structurally-valid reports proceed to the on-mandate judge.

Entry point::

    from round_table_portfolio.personas.output_validator import validate_persona_report
    result = validate_persona_report(report, mandate, config, judge=judge)
    # result.passed bool, result.notes str, result.stage str

    # From M2: pass counterfactual_portfolio to add the fully-invested gate (clause c).
    result = validate_persona_report(
        report, mandate, config, judge=judge,
        counterfactual_portfolio={"AAPL": 0.10, "MSFT": 0.10, "CASH": 0.80},
        max_position_weight=0.20,
    )

The concrete on-mandate judge (``OnMandateJudge`` implementor) is wired at the
orchestration layer (TASK-M1-010 runner), NOT inside this module.  This module
exposes:

- ``OnMandateJudge``  — Protocol/interface that the orchestration layer implements
                        as a subagent dispatch.
- ``StubOnMandateJudge`` — Deterministic stub for unit tests.
- ``validate_persona_report`` — Composes structural gate + judge.

This module does NOT write DB rows.  The research runner (TASK-M1-010) writes
``persona_reports.validator_passed`` and ``persona_reports.validator_notes``
from the returned ``ReportValidationResult``.

Component 5 (persona definition validator) != Component 11 (output validator).
They share a package but address different concerns.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import yaml

from round_table_portfolio.portfolio.invariants import check_fully_invested

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(os.environ.get("VALIDATOR_CONFIG", "config/validator.yaml"))


@dataclass(frozen=True)
class StructuralConfig:
    min_report_chars: int
    min_ticker_references: int
    min_metric_terms: int
    metric_terms: tuple[str, ...]
    data_source_signals: tuple[str, ...]


@dataclass(frozen=True)
class PersonaConfig:
    on_mandate_concepts: tuple[str, ...]
    off_mandate_signals: tuple[str, ...]


@dataclass(frozen=True)
class ValidatorConfig:
    structural: StructuralConfig
    personas: dict[str, PersonaConfig]


def load_validator_config(config_path: Optional[Path] = None) -> ValidatorConfig:
    """Read validator.yaml and return a typed config object.

    Args:
        config_path: Override for testing; defaults to VALIDATOR_CONFIG env var
                     or ``config/validator.yaml`` relative to the working dir.
    """
    path = config_path or _CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    s = raw["structural"]
    structural = StructuralConfig(
        min_report_chars=int(s["min_report_chars"]),
        min_ticker_references=int(s["min_ticker_references"]),
        min_metric_terms=int(s.get("min_metric_terms", 0)),
        metric_terms=tuple(t.lower() for t in s.get("metric_terms", [])),
        data_source_signals=tuple(sig.lower() for sig in s["data_source_signals"]),
    )

    personas: dict[str, PersonaConfig] = {}
    for slug, p in (raw.get("personas") or {}).items():
        personas[slug] = PersonaConfig(
            on_mandate_concepts=tuple(p.get("on_mandate_concepts", [])),
            off_mandate_signals=tuple(p.get("off_mandate_signals", [])),
        )

    return ValidatorConfig(structural=structural, personas=personas)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

STAGE_STRUCTURAL = "structural"
STAGE_FULLY_INVESTED = "fully_invested"
STAGE_LLM_JUDGE = "llm_judge"


@dataclass
class ReportValidationResult:
    """Outcome of validating one persona report.

    Mirrors the shape written to ``persona_reports`` by the runner:
    - ``passed``  -> ``validator_passed`` (0/1)
    - ``notes``   -> ``validator_notes``
    - ``stage``   -> informational only (which gate made the call)
    """

    passed: bool
    notes: str
    stage: str  # STAGE_STRUCTURAL or STAGE_LLM_JUDGE
    llm_justification: str = ""  # populated by judge path only

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# Deterministic structural gate
# ---------------------------------------------------------------------------

# Pattern that matches uppercase ticker-like tokens (2-5 capital letters,
# optionally surrounded by word boundaries or punctuation).
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")

# Common English words and financial acronyms that are NOT equity tickers.
# Maintained here so the exclude set is defined once and tested explicitly.
_TICKER_EXCLUDE: frozenset[str] = frozenset({
    # Common English words and markdown tokens.
    "THE", "AND", "FOR", "NOT", "BUT", "NOR", "YET", "SO", "OR",
    "WITH", "FROM", "THAT", "THIS", "PASS", "FAIL",
    # Financial / macro acronyms that are NOT equity tickers.
    "FCF", "RSI", "TAM", "CPI", "PCE", "FED", "SEC", "ETF",
    "CEO", "CFO", "COO", "IPO", "GDP", "EPS", "ROE", "ROA",
    "FRED", "ISM", "VIX", "VaR", "YTD", "YOY", "TTM",
    "MACD", "ROIC", "ARR", "AUM", "LBO", "EBITDA",
    "REIT", "SPAC", "AI", "ML", "US", "UK", "EU", "USD",
    "GPU", "FSD", "AWS", "PBM",
})


def _run_fully_invested_gate(
    counterfactual_portfolio: dict[str, float],
    max_position_weight: float,
) -> ReportValidationResult:
    """Clause (c) of the deterministic structural gate — fully-invested check.

    Splits ``counterfactual_portfolio`` into ``positions`` (everything except
    the ``CASH`` key) and ``cash``, then delegates all arithmetic to
    ``check_fully_invested`` from ``portfolio/invariants.py``.

    This is the SINGLE call site for the Layer-2 invariant check.  Component 15
    (TASK-M2-005) calls the same ``check_fully_invested`` helper directly —
    both sites share identical arithmetic (the "intentionally identical
    arithmetic" guarantee).

    Does NOT judge cash level — only mechanical sum-to-100% is checked here.
    """
    cash = counterfactual_portfolio.get("CASH", 0.0)
    positions = {k: v for k, v in counterfactual_portfolio.items() if k != "CASH"}

    passed, reasons = check_fully_invested(positions, cash, max_position_weight)

    if not passed:
        notes = "FULLY-INVESTED GATE FAIL — " + " | ".join(reasons)
        logger.debug("Fully-invested gate failed: %s", notes)
        return ReportValidationResult(passed=False, notes=notes, stage=STAGE_FULLY_INVESTED)

    return ReportValidationResult(
        passed=True,
        notes="Fully-invested gate passed.",
        stage=STAGE_FULLY_INVESTED,
    )


def _run_structural_gate(
    report: str,
    cfg: StructuralConfig,
) -> ReportValidationResult:
    """Deterministic checks — no judge call.

    Checks in order (fail-fast):
    1. Minimum total length.
    2. Minimum ticker references.
    3. At least one data-source signal present.
    4. Metric-term density floor (catches long-ish bare verdicts with no
       substantive financial vocabulary).

    Deliberately does NOT require any specific markdown heading structure.
    The persona RESEARCH OUTPUT SCHEMA (Component 5) specifies free-text
    detailed-rationale prose — rigid ``## Thesis`` / ``## Rationale`` headers
    are never mandated and real persona output does not include them.
    """
    failures: list[str] = []

    # 1. Total length floor.
    if len(report) < cfg.min_report_chars:
        failures.append(
            f"Report too short ({len(report)} chars < {cfg.min_report_chars}): "
            "no detailed rationale present."
        )

    # 2. Ticker references.
    tickers = _TICKER_RE.findall(report)
    real_tickers = [t for t in tickers if t not in _TICKER_EXCLUDE]
    if len(real_tickers) < cfg.min_ticker_references:
        failures.append(
            f"Fewer than {cfg.min_ticker_references} ticker references found "
            f"(found {len(real_tickers)}): per-name rationale absent."
        )

    # 3. Data-source signal.
    report_lower = report.lower()
    has_signal = any(sig in report_lower for sig in cfg.data_source_signals)
    if not has_signal:
        failures.append(
            "No data-source signal found (no reference to valuation metrics, "
            "technicals, macro series, or fundamentals)."
        )

    # 4. Metric-term density.
    if cfg.min_metric_terms > 0 and cfg.metric_terms:
        distinct_hits = sum(1 for t in cfg.metric_terms if t in report_lower)
        if distinct_hits < cfg.min_metric_terms:
            failures.append(
                f"Insufficient financial vocabulary: {distinct_hits} distinct "
                f"metric terms found, {cfg.min_metric_terms} required — "
                "report lacks substantive per-name data rationale."
            )

    if failures:
        notes = "STRUCTURAL GATE FAIL — " + " | ".join(failures)
        logger.debug("Structural gate failed: %s", notes)
        return ReportValidationResult(passed=False, notes=notes, stage=STAGE_STRUCTURAL)

    return ReportValidationResult(passed=True, notes="Structural gate passed.", stage=STAGE_STRUCTURAL)


# ---------------------------------------------------------------------------
# OnMandateJudge interface
# ---------------------------------------------------------------------------

@runtime_checkable
class OnMandateJudge(Protocol):
    """Interface for the on-mandate judge.

    The concrete implementation (wired in TASK-M1-010 orchestration) dispatches
    the ``output-validator-judge`` subagent with the report + mandate text and
    parses its structured VERDICT/JUSTIFICATION response.

    Unit tests inject a ``StubOnMandateJudge`` — no subagent is spawned and
    no external service is called.
    """

    def judge(
        self,
        report: str,
        mandate: str,
        persona_slug: str,
        on_mandate_concepts: tuple[str, ...],
        off_mandate_signals: tuple[str, ...],
    ) -> tuple[bool, str]:
        """Return (passed, one_paragraph_justification).

        Args:
            report: The full persona report text.
            mandate: The persona's RESEARCH MANDATE section text.
            persona_slug: e.g. "value", "technical".
            on_mandate_concepts: Key concepts that should appear in on-mandate reasoning.
            off_mandate_signals: Phrases that suggest drift into another persona's lens.

        Returns:
            (passed, justification) — ``passed=True`` means reasoning stayed
            on-mandate; justification is a one-paragraph human-readable explanation.
        """
        ...


# ---------------------------------------------------------------------------
# Judge prompt template (used by the orchestration layer to build the subagent
# dispatch message — kept here so prompt and interface stay co-located)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are a strict mandate-compliance judge for a multi-persona investment research panel.

## PERSONA MANDATE
{mandate}

## ON-MANDATE CONCEPTS (what this persona SHOULD reason through)
{on_mandate_concepts}

## OFF-MANDATE SIGNALS (phrases that suggest drift into another persona's lens)
{off_mandate_signals}

## REPORT TO EVALUATE
{report}

## TASK
Determine whether the report's CORE ARGUMENT and REASONING stay on-mandate for this persona.

Rules:
- A Value persona MAY briefly mention momentum to dismiss it — that is on-mandate.
  A Value persona whose CORE ARGUMENT is "this stock is going up" — that is off-mandate.
- Focus on the reasoning lens, not the conclusion.
- Be strict: a report that merely uses on-mandate vocabulary but argues through an
  off-mandate lens should FAIL.

Respond with EXACTLY this format (no extra text):
VERDICT: PASS  (or FAIL)
JUSTIFICATION: <one paragraph explaining why, citing specific phrases from the report>
"""


class StubOnMandateJudge:
    """Deterministic stub for unit tests and SKIP_LIVE environments.

    Accepts a mapping of ``(persona_slug, first_50_chars_of_report) -> (passed, justification)``
    so tests can inject precise, fixture-matched responses.

    If a fixture key is not found, falls back to a keyword heuristic — checks
    whether any ``off_mandate_signals`` appear prominently in the report core.
    The heuristic is NOT production-quality; it exists only so tests that do not
    inject an explicit response still get a deterministic answer.
    """

    def __init__(
        self,
        responses: Optional[dict[tuple[str, str], tuple[bool, str]]] = None,
    ) -> None:
        self._responses: dict[tuple[str, str], tuple[bool, str]] = responses or {}

    def judge(
        self,
        report: str,
        mandate: str,
        persona_slug: str,
        on_mandate_concepts: tuple[str, ...],
        off_mandate_signals: tuple[str, ...],
    ) -> tuple[bool, str]:
        key = (persona_slug, report[:50])
        if key in self._responses:
            return self._responses[key]

        # Keyword-heuristic fallback for un-keyed fixtures.
        report_lower = report.lower()
        # More off-mandate signal than on-mandate signal -> likely drifted.
        # Using strict > (not >=) so a tie is treated as on-mandate (benefit of doubt).
        off_hits = sum(1 for s in off_mandate_signals if s.lower() in report_lower)
        on_hits = sum(1 for c in on_mandate_concepts if c.lower() in report_lower)
        if off_hits > on_hits:
            return (
                False,
                f"Stub heuristic: {off_hits} off-mandate signals vs "
                f"{on_hits} on-mandate concepts — reasoning appears drifted.",
            )
        return (
            True,
            f"Stub heuristic: {on_hits} on-mandate concepts vs "
            f"{off_hits} off-mandate signals — appears on-mandate.",
        )


# ---------------------------------------------------------------------------
# Validator-claim persistence (M1-014 / Component 20)
# ---------------------------------------------------------------------------

def persist_validator_claim(
    result: "ReportValidationResult",
    week_id: str,
    persona_slug: str,
    *,
    state_root: Path = Path("state"),
) -> Path:
    """Serialize a ReportValidationResult to a durable JSON claim file.

    Writes ``state/reports/<week_id>/validator_claims/<persona_slug>.json``
    (option-b path-convention — no schema change; the path is deterministically
    reconstructable from ``week_id`` + ``persona_slug``).

    The JSON contains every field on ``ReportValidationResult``:
      - ``passed`` (bool) — PASS / FAIL
      - ``notes`` (str)  — deterministic-gate flags or off-mandate description
      - ``stage`` (str)  — which gate made the call
      - ``llm_justification`` (str) — on-mandate judge justification (empty when
        the structural or fully-invested gate short-circuited before the judge)

    Note on "cited report-excerpt evidence": the judge justification
    (``llm_justification``) contains the judge's cited phrases from the
    report.  There is no separate ``evidence`` field on ``ReportValidationResult``
    — the justification IS the evidence carrier.  This gap is documented in
    the TASK-M2-010 quality log.

    The file is re-loadable: ``ReportValidationResult(**json.load(f))`` reconstructs
    the object (confirmed by round-trip tests).

    Args:
        result:       The validation verdict from ``validate_persona_report``.
        week_id:      ISO week label e.g. "2026-W23".
        persona_slug: e.g. "value", "technical".
        state_root:   Root of the runtime state tree; defaults to Path("state").
                      Tests pass tmp_path here.

    Returns:
        The Path of the written claim file.
    """
    import json as _json

    claims_dir = state_root / "reports" / week_id / "validator_claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    claim_path = claims_dir / f"{persona_slug}.json"
    payload = {
        "week_id": week_id,
        "persona": persona_slug,
        "passed": result.passed,
        "notes": result.notes,
        "stage": result.stage,
        "llm_justification": result.llm_justification,
    }
    claim_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    logger.debug(
        "Validator claim written: week=%s persona=%s passed=%s path=%s",
        week_id, persona_slug, result.passed, claim_path,
    )
    return claim_path


def load_validator_claim(claim_path: Path) -> "ReportValidationResult":
    """Re-load a persisted claim JSON into a ReportValidationResult.

    Provides the round-trip path: write via ``persist_validator_claim``,
    read back via this helper.

    Args:
        claim_path: Path to the ``.json`` file written by ``persist_validator_claim``.

    Returns:
        A ``ReportValidationResult`` reconstructed from the JSON payload.

    Raises:
        FileNotFoundError: If the claim file does not exist.
        KeyError: If the JSON is missing a required field.
    """
    import json as _json

    raw = _json.loads(claim_path.read_text(encoding="utf-8"))
    return ReportValidationResult(
        passed=bool(raw["passed"]),
        notes=str(raw["notes"]),
        stage=str(raw["stage"]),
        llm_justification=str(raw.get("llm_justification", "")),
    )


def parse_judge_response(raw: str) -> tuple[bool, str]:
    """Parse the structured judge response into (passed, justification).

    Used by the orchestration layer after receiving the subagent's text output.
    """
    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", raw, re.IGNORECASE)
    just_match = re.search(r"JUSTIFICATION:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)

    if not verdict_match:
        # Malformed response — conservative fail.
        logger.warning("Judge returned malformed response; defaulting to FAIL. Raw: %r", raw[:200])
        return False, f"Malformed judge response: {raw[:200]}"

    passed = verdict_match.group(1).upper() == "PASS"
    justification = just_match.group(1).strip() if just_match else raw
    return passed, justification


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def validate_persona_report(
    report: str,
    mandate: str,
    config: ValidatorConfig,
    persona_slug: str = "",
    judge: Optional[OnMandateJudge] = None,
    *,
    counterfactual_portfolio: Optional[dict[str, float]] = None,
    max_position_weight: float = 0.20,
) -> ReportValidationResult:
    """Validate a persona report against its mandate.

    Stage 1 — Deterministic structural gate (no judge call):
        Checks minimum length, required sections, ticker count, data-source
        signals.  Failure short-circuits; no judge call is made.

    Stage 1b — Fully-invested gate (clause c, M2, optional):
        When ``counterfactual_portfolio`` is provided, the deterministic gate
        additionally asserts the portfolio sums to 1.0 within 1e-6, cash ≥ 0,
        all weights ≥ 0, and no single position weight > ``max_position_weight``.
        Uses ``check_fully_invested`` from ``portfolio/invariants.py`` — the
        SAME helper Component 15 (Layer-3 backstop) calls.  A portfolio failure
        short-circuits to ``passed=False`` without dispatching the judge.
        Does NOT judge cash level — only mechanical sum-to-100%.

    Stage 2 — On-mandate judge (only if stages 1/1b pass):
        The injected judge reads the report + mandate and returns pass/fail +
        one-paragraph justification.  In production this is a subagent dispatch
        (wired by the TASK-M1-010 runner); in tests it is a StubOnMandateJudge.

    Args:
        report: The full persona report text.
        mandate: The persona's RESEARCH MANDATE section text.
        config: Loaded ``ValidatorConfig`` (from ``load_validator_config()``).
        persona_slug: e.g. "value", "technical".  Used to look up per-persona
                      concept lists in ``config.personas``.  Falls back to
                      empty lists if the slug is not configured.
        judge: Injectable on-mandate judge.  Pass a ``StubOnMandateJudge`` in
               tests.  In production the runner passes its subagent-dispatch
               implementation.  Raises ``ValueError`` if None — the orchestration
               layer must always wire a concrete judge; there is no default.
        counterfactual_portfolio: Optional mapping of ticker → weight including
               an explicit ``"CASH"`` key.  When present, the fully-invested
               structural check (clause c) runs before the LLM judge.
        max_position_weight: Per-ticker hard ceiling; read from
               ``config/thresholds.yaml`` by the caller and passed in.
               Defaults to 0.20 but callers should always pass the configured
               value — the default is a safety net, not the source of truth.

    Returns:
        ``ReportValidationResult`` with ``passed``, ``notes``, ``stage``, and
        (for the judge path) ``llm_justification``.

    Raises:
        ValueError: If ``judge`` is None (no default judge exists — must be
                    wired by the caller).
    """
    # Stage 1 — structural gate.
    structural_result = _run_structural_gate(report, config.structural)
    if not structural_result.passed:
        return structural_result

    # Stage 1b — fully-invested gate (clause c, M2).
    if counterfactual_portfolio is not None:
        fi_result = _run_fully_invested_gate(counterfactual_portfolio, max_position_weight)
        if not fi_result.passed:
            return fi_result

    # Stage 2 — on-mandate judge.
    if judge is None:
        raise ValueError(
            "validate_persona_report requires a judge implementation. "
            "Pass a StubOnMandateJudge for tests or wire the subagent-dispatch "
            "implementation from the TASK-M1-010 runner."
        )

    persona_cfg = config.personas.get(persona_slug, PersonaConfig((), ()))

    logger.debug("On-mandate judge dispatched for persona=%r", persona_slug)
    passed, justification = judge.judge(
        report=report,
        mandate=mandate,
        persona_slug=persona_slug,
        on_mandate_concepts=persona_cfg.on_mandate_concepts,
        off_mandate_signals=persona_cfg.off_mandate_signals,
    )

    if passed:
        notes = "On-mandate: reasoning stays within the persona's lens."
    else:
        notes = f"OFF-MANDATE (judge): {justification[:300]}"

    return ReportValidationResult(
        passed=passed,
        notes=notes,
        stage=STAGE_LLM_JUDGE,
        llm_justification=justification,
    )
