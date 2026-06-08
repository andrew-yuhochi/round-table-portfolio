"""Minimal stubs for sibling-task helpers not yet implemented.

Every function here is marked with the TASK that replaces it.  They return
correctly-shaped placeholder data so the orchestrator spine can be tested
end-to-end without the real implementations.

Do NOT call these in production — the live /weekly-run (TASK-M2-011) wires the
real helpers.  Each stub raises AssertionError if called outside a test
environment (i.e. when STUB_ALLOW env var is not set), so a production mistake
fails loudly rather than silently returning placeholder data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _assert_stub_allowed(name: str) -> None:
    """Guard: fail loudly if a stub is called outside a test context."""
    if os.environ.get("STUB_ALLOW") != "1":
        raise RuntimeError(
            f"STUB called in non-test context: {name!r}. "
            "Set STUB_ALLOW=1 only in tests. "
            "Production must use the real implementation."
        )


# ---------------------------------------------------------------------------
# Component 13 — debate_set_construction  (STUB → replaced by TASK-M2-004)
# ---------------------------------------------------------------------------

def construct_debate_set(
    research_results: list[Any],
    config: dict[str, Any],
) -> list[str]:
    """STUB — replaced by TASK-M2-004 (orchestrator/round1.py).

    Returns a minimal de-duplicated list of tickers from all persona shortlists.
    """
    _assert_stub_allowed("construct_debate_set")
    tickers: dict[str, None] = {}
    for result in research_results:
        for row in result.shortlist_rows:
            tickers[row.ticker] = None
    return list(tickers.keys()) if tickers else ["AAPL", "MSFT", "GOOGL"]


# ---------------------------------------------------------------------------
# Component 14 — round1_stance_capture  (STUB → replaced by TASK-M2-004)
# ---------------------------------------------------------------------------

@dataclass
class AgentStancePayload:
    """One row payload for agent_stances (round=1)."""
    week_id: str
    persona: str
    ticker: str
    round: int  # always 1 in M2
    action: str  # 'add' | 'reduce' | 'hold' | 'exit'
    target_weight: float
    confidence: int  # 1–5
    rationale: str
    user_id: str = "andrew"
    roster_version: int = 1
    enhancement_version: int = 1


@dataclass
class Round1Capture:
    """Return type of capture_round1_stances.

    stances: flat list of AgentStancePayload (7 personas × |debate_set| rows)
    counterfactuals: per-persona portfolio dict {ticker: weight, 'CASH': weight}
    """
    stances: list[AgentStancePayload] = field(default_factory=list)
    counterfactuals: dict[str, dict[str, float]] = field(default_factory=dict)


def capture_round1_stances(
    debate_set: list[str],
    research_results: list[Any],
    *,
    persona_replies: dict[str, str] | None = None,
    judge: Any = None,
    config: dict[str, Any] | None = None,
) -> Round1Capture:
    """STUB — replaced by TASK-M2-004 (orchestrator/round1.py).

    Generates one stance per persona per debate-set ticker, all round=1.
    Counterfactual portfolios are evenly distributed (minus small cash residual).
    """
    _assert_stub_allowed("capture_round1_stances")
    personas = [r.persona_slug for r in research_results]
    stances: list[AgentStancePayload] = []
    counterfactuals: dict[str, dict[str, float]] = {}

    if not debate_set:
        debate_set = ["AAPL", "MSFT", "GOOGL"]

    # Evenly split weight across tickers, leave 10% cash.
    n = len(debate_set)
    position_weight = round(0.90 / n, 6) if n else 0.0
    cash = round(1.0 - position_weight * n, 6)

    for persona in personas:
        portfolio: dict[str, float] = {}
        for ticker in debate_set:
            stances.append(AgentStancePayload(
                week_id=research_results[0].week_id,
                persona=persona,
                ticker=ticker,
                round=1,
                action="add",
                target_weight=position_weight,
                confidence=3,
                rationale=f"Stub stance: {persona} on {ticker}",
            ))
            portfolio[ticker] = position_weight
        portfolio["CASH"] = cash
        counterfactuals[persona] = portfolio

    return Round1Capture(stances=stances, counterfactuals=counterfactuals)


# ---------------------------------------------------------------------------
# Component 15 — materialize_portfolios  (STUB → replaced by TASK-M2-005)
# ---------------------------------------------------------------------------

@dataclass
class HoldingPayload:
    """One row payload for holdings."""
    ticker: str
    weight: float
    action: str
    entry_date: str
    user_id: str = "andrew"
    roster_version: int = 1


@dataclass
class PortfolioPayload:
    """One portfolio + its holdings, ready for ledger write."""
    type: str
    week_id: str
    roster_version: int
    enhancement_version: int
    user_id: str
    holdings: list[HoldingPayload] = field(default_factory=list)


def materialize_portfolios(
    counterfactuals: dict[str, dict[str, float]],
    consensus_weights: dict[str, float],
    *,
    prior_portfolios: dict[str, Any] | None = None,
    week_id: str,
    config: dict[str, Any] | None = None,
    entry_date: str = "",
) -> list[PortfolioPayload]:
    """STUB — replaced by TASK-M2-005 (portfolio/materialize.py).

    Converts counterfactual dicts (including CASH key) + consensus weights
    into PortfolioPayload objects ready for the ledger transaction.
    """
    _assert_stub_allowed("materialize_portfolios")
    payloads: list[PortfolioPayload] = []
    _date = entry_date or week_id

    # 7 persona portfolios.
    for persona, weights in counterfactuals.items():
        holdings = [
            HoldingPayload(
                ticker=t,
                weight=w,
                action="hold" if t == "CASH" else "add",
                entry_date=_date,
            )
            for t, w in weights.items()
        ]
        payloads.append(PortfolioPayload(
            type=persona,
            week_id=week_id,
            roster_version=1,
            enhancement_version=1,
            user_id="andrew",
            holdings=holdings,
        ))

    # Consensus portfolio.
    consensus_holdings = [
        HoldingPayload(
            ticker=t,
            weight=w,
            action="hold" if t == "CASH" else "add",
            entry_date=_date,
        )
        for t, w in consensus_weights.items()
    ]
    payloads.append(PortfolioPayload(
        type="consensus",
        week_id=week_id,
        roster_version=1,
        enhancement_version=1,
        user_id="andrew",
        holdings=consensus_holdings,
    ))

    return payloads


# ---------------------------------------------------------------------------
# Component 16 — blend_consensus  (STUB → replaced by TASK-M2-006)
# ---------------------------------------------------------------------------

def blend_consensus(
    stances: list[AgentStancePayload],
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """STUB — replaced by TASK-M2-006 (portfolio/consensus.py).

    Simple mean of target_weight per ticker across non-EXIT stances.
    Returns weights dict including a CASH key.
    """
    _assert_stub_allowed("blend_consensus")
    ticker_weights: dict[str, list[float]] = {}
    for s in stances:
        if s.action != "exit":
            ticker_weights.setdefault(s.ticker, []).append(s.target_weight)

    blended: dict[str, float] = {}
    for ticker, weights in ticker_weights.items():
        blended[ticker] = round(sum(weights) / len(weights), 6) if weights else 0.0

    total = sum(blended.values())
    blended["CASH"] = round(max(0.0, 1.0 - total), 6)
    return blended


# ---------------------------------------------------------------------------
# Component 17 — write_round1_transcript  (STUB → replaced by TASK-M2-007)
# ---------------------------------------------------------------------------

def write_round1_transcript(
    round1_capture: Round1Capture,
    consensus: dict[str, float],
    std_dev: dict[str, float],
    decision: str,
    *,
    week_id: str,
    state_root: Path = Path("state"),
) -> Path:
    """STUB — replaced by TASK-M2-007 (orchestrator/transcript.py).

    Writes a minimal placeholder transcript file and returns its path.
    """
    _assert_stub_allowed("write_round1_transcript")
    debates_dir = state_root / "debates"
    debates_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = debates_dir / f"{week_id}.md"
    transcript_path.write_text(
        f"# Round-1 Transcript — {week_id}\n\n"
        f"## Decision\n{decision}\n\n"
        f"## Consensus\n{consensus}\n\n"
        f"(STUB — replaced by TASK-M2-007)\n",
        encoding="utf-8",
    )
    return transcript_path


# ---------------------------------------------------------------------------
# Component 18 — writeback_memory  (STUB → replaced by TASK-M2-008)
# ---------------------------------------------------------------------------

def writeback_memory(
    research_results: list[Any],
    round1_capture: Round1Capture,
    consensus: dict[str, float],
    *,
    week_id: str,
    state_root: Path = Path("state"),
) -> None:
    """STUB — replaced by TASK-M2-008.

    No-op placeholder; real implementation updates state/memory/<persona>.md.
    """
    _assert_stub_allowed("writeback_memory")
    # No-op: memory write-back happens AFTER the ledger commit (TDD §1.5).


# ---------------------------------------------------------------------------
# Component 19 — report_run_metrics  (STUB → replaced by TASK-M2-009)
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Metrics returned by report_run_metrics."""
    total_wall_seconds: float
    per_persona_seconds: dict[str, float]
    per_persona_web_searches: dict[str, int]
    window_fraction: float  # 0–1, fraction of the 5-hour window consumed


def report_run_metrics(
    research_results: list[Any],
    *,
    start_time: float,
    config: dict[str, Any] | None = None,
) -> RunMetrics:
    """STUB — replaced by TASK-M2-009.

    Returns a zeroed-out metrics placeholder.
    """
    _assert_stub_allowed("report_run_metrics")
    import time
    elapsed = time.time() - start_time
    return RunMetrics(
        total_wall_seconds=elapsed,
        per_persona_seconds={r.persona_slug: 0.0 for r in research_results},
        per_persona_web_searches={
            r.persona_slug: r.parsed_output.web_searches_used
            for r in research_results
        },
        window_fraction=elapsed / (5.0 * 3600),
    )


# ---------------------------------------------------------------------------
# Component 20 — persist_validator_claim  (STUB → replaced by TASK-M2-010)
# ---------------------------------------------------------------------------

def persist_validator_claim(
    result: Any,
    week_id: str,
    persona_slug: str,
    *,
    state_root: Path = Path("state"),
) -> None:
    """STUB — replaced by TASK-M2-010 (real impl in personas/output_validator.py).

    Writes the durable validator claim JSON at
    state/reports/<week_id>/validator_claims/<persona>.json.
    """
    _assert_stub_allowed("persist_validator_claim")
    claims_dir = state_root / "reports" / week_id / "validator_claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    claim_path = claims_dir / f"{persona_slug}.json"
    import json
    claim = {
        "week_id": week_id,
        "persona": persona_slug,
        "passed": result.validation.passed,
        "notes": result.validation.notes,
        "stage": result.validation.stage,
        "_stub": "TASK-M2-011 replaces this",
    }
    claim_path.write_text(json.dumps(claim, indent=2), encoding="utf-8")
