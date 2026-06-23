"""Unit tests for orchestrator/round1.py — Components 13 + 14.

Coverage matrix:
  Component 13 — construct_debate_set:
    test_dedup_union_equals_direct_plus_peers      — debate set = dedup union of shortlists + peers
    test_strict_superset_of_directly_shortlisted   — direct names always present (AC1)
    test_ceiling_bound_trims_peers_first           — ceiling trim removes peers, not direct (AC1)
    test_ceiling_exactly_met                       — set size == ceiling when peers fill it
    test_no_peers_within_ceiling                   — direct-only set, no peers
    test_direct_exceeds_ceiling_raises             — more direct names than ceiling → RuntimeError
    test_duplicate_tickers_across_personas         — same ticker in multiple personas → once
    test_cluster_peer_promoted_by_direct           — peer promoted to direct if another persona shortlists it
    test_empty_personas                            — zero results → empty debate set

  Component 13 / Component 14 cross-check:
    test_debate_set_size_equals_distinct_stance_tickers — |debate set| == distinct stance tickers (AC2)

  Component 14 — capture_round1_stances:
    test_parse_canned_json_full_coverage           — all tickers covered, actions normalised, counts correct (AC3/AC4)
    test_stance_count_7_times_debate_set           — count == 7 × |debate set|, all round=1 (AC4)
    test_action_vocabulary_normalised_uppercase    — ADD/REDUCE/EXIT/HOLD → add/reduce/exit/hold
    test_out_of_domain_action_raises               — unknown action raises Round1ParseError, no coercion (AC3)
    test_out_of_domain_confidence_raises           — confidence 6 raises Round1ParseError (AC3)
    test_confidence_boundary_valid                 — confidence 1 and 5 are both valid
    test_out_of_domain_weight_raises               — weight > max_position_weight raises (AC3)
    test_weight_zero_valid                         — weight 0.0 is allowed (EXIT stance)
    test_missing_cash_in_counterfactual_raises     — no CASH key → Round1ParseError
    test_missing_ticker_coverage_raises            — persona omits a debate-set ticker → Round1ParseError
    test_extra_ticker_not_in_debate_set_raises     — stance for non-debate ticker → Round1ParseError
    test_counterfactual_negative_weight_raises     — negative weight in counterfactual → Round1ParseError
    test_counterfactual_position_over_cap_raises   — non-CASH weight > cap → Round1ParseError
    test_missing_persona_reply_raises              — no reply for a persona → Round1ParseError

  Component 14 — commit-before-reveal (AC5):
    test_prompts_built_before_any_reply_processed  — prompts dict populated for all personas
    test_prompt_isolation_no_peer_output           — no persona's prompt contains another's Round-1 output
    test_custom_prompt_builder_injected            — prompt_builder callable is called per persona

  Component 14 — thesis_status contract (M6-004):
    test_exit_with_thesis_status_parses_clean      — valid EXIT + thesis_status accepted
    test_reduce_with_thesis_status_parses_clean    — valid REDUCE + thesis_status accepted
    test_add_with_thesis_status_parses_clean       — valid ADD + thesis_status accepted
    test_hold_without_thesis_status_parses_clean   — HOLD with no thesis_status exempt
    test_hold_with_thesis_status_parses_clean      — HOLD with optional thesis_status accepted
    test_exit_missing_thesis_status_raises         — EXIT missing thesis_status → parse error
    test_reduce_missing_thesis_status_raises       — REDUCE missing thesis_status → parse error
    test_add_missing_thesis_status_raises          — ADD missing thesis_status → parse error
    test_exit_empty_reason_raises                  — EXIT with empty reason → parse error
    test_exit_invalid_verdict_raises               — EXIT with verdict=new → parse error
    test_add_invalid_verdict_raises                — ADD with verdict=broken → parse error
    test_thesis_status_serialized_into_rationale   — content folded into agent_stances.rationale
    test_thesis_status_present_in_rationale_text   — 'THESIS STATUS:' label in rationale
    test_hold_rationale_unchanged_no_thesis_status — HOLD rationale clean when no thesis_status
    test_intact_verdict_allowed_on_exit            — EXIT verdict=intact is valid parse (C11 flags)
    test_intact_verdict_allowed_on_reduce          — REDUCE verdict=intact is valid parse
    test_multiple_action_stances_all_require_ts    — per-stance enforcement (not just first)

  Component 14 — Round1Capture shape:
    test_round1_capture_has_prompts_field          — prompts dict keyed by persona slug
    test_counterfactuals_keyed_by_persona          — 7 entries, each has CASH key
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from round_table_portfolio.orchestrator.round1 import (
    AgentStancePayload,
    Round1Capture,
    Round1ParseError,
    capture_round1_stances,
    construct_debate_set,
)
from round_table_portfolio.research.runner import (
    PersonaResearchResult,
    PersonaReportPayload,
    PersonaShortlistRow,
    PersonaOutputSchema,
    ShortlistEntry,
)
from round_table_portfolio.personas.output_validator import (
    ReportValidationResult,
    StubOnMandateJudge,
)
from round_table_portfolio.budget.tracker import PersonaBudgetTracker
from round_table_portfolio.budget.loader import PersonaBudget


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSONA_SLUGS_7 = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]

WEEK_ID = "2026-W23"
MAX_WEIGHT = 0.20


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_budget() -> PersonaBudget:
    return PersonaBudget(
        max_turns=12,
        max_web_searches=8,
        max_data_tool_calls=15,
    )


def _make_report_payload(persona: str, week_id: str = WEEK_ID) -> PersonaReportPayload:
    return PersonaReportPayload(
        week_id=week_id,
        persona=persona,
        summary=f"{persona} summary: strong FCF names identified.",
        validator_passed=1,
        validator_notes="",
        full_report_path=f"state/reports/{week_id}/{persona}.md",
    )


def _make_validation_result() -> ReportValidationResult:
    return ReportValidationResult(passed=True, notes="", stage="structural")


def _make_persona_result(
    persona: str,
    direct_tickers: list[str],
    cluster_map: dict[str, list[str]] | None = None,
    week_id: str = WEEK_ID,
) -> PersonaResearchResult:
    """Build a PersonaResearchResult with the given direct shortlist + cluster peers."""
    cluster_map = cluster_map or {}
    shortlist_rows: list[PersonaShortlistRow] = []
    shortlist_entries: list[ShortlistEntry] = []

    for ticker in direct_tickers:
        shortlist_rows.append(PersonaShortlistRow(
            week_id=week_id,
            persona=persona,
            ticker=ticker,
            is_cluster_peer=0,
            parent_ticker=None,
        ))
        peers = cluster_map.get(ticker, [])
        shortlist_entries.append(ShortlistEntry(ticker=ticker, why="test", cluster=peers))
        for peer in peers:
            # Only add as peer if not already direct.
            if peer not in direct_tickers:
                shortlist_rows.append(PersonaShortlistRow(
                    week_id=week_id,
                    persona=persona,
                    ticker=peer,
                    is_cluster_peer=1,
                    parent_ticker=ticker,
                ))

    parsed_output = PersonaOutputSchema(
        shortlist=shortlist_entries,
        report=f"{persona} report text with P/E FCF yield data.",
        web_searches_used=3,
        data_tool_calls_used=5,
    )
    budget = _make_budget()
    tracker = PersonaBudgetTracker(persona=persona, budget=budget)
    tracker.record("web_searches", count=3)
    tracker.record("turns", count=1)

    return PersonaResearchResult(
        persona_slug=persona,
        week_id=week_id,
        parsed_output=parsed_output,
        validation=_make_validation_result(),
        report_payload=_make_report_payload(persona, week_id),
        shortlist_rows=shortlist_rows,
        budget_summary=tracker.summary(),
        budget_overrun=False,
    )


def _make_7_results(
    ticker_sets: list[tuple[str, list[str], dict[str, list[str]]]],
) -> list[PersonaResearchResult]:
    """Build exactly 7 PersonaResearchResult objects.

    ticker_sets: list of (persona_slug, direct_tickers, cluster_map) tuples.
    """
    assert len(ticker_sets) == 7
    return [
        _make_persona_result(slug, directs, clusters)
        for slug, directs, clusters in ticker_sets
    ]


def _uniform_ticker_sets(
    directs: list[str],
    clusters: dict[str, list[str]] | None = None,
) -> list[tuple[str, list[str], dict[str, list[str]]]]:
    """All 7 personas share the same direct + cluster tickers."""
    return [(slug, directs, clusters or {}) for slug in PERSONA_SLUGS_7]


# ---------------------------------------------------------------------------
# Round-1 JSON builders
# ---------------------------------------------------------------------------

def _make_round1_json(
    persona: str,
    debate_set: list[str],
    *,
    action: str = "ADD",
    confidence: int = 3,
    weight: float = 0.10,
    counterfactual: dict[str, float] | None = None,
) -> str:
    """Build a valid ROUND 1 OUTPUT SCHEMA JSON string for the given debate set.

    Includes thesis_status on every EXIT/REDUCE/ADD stance (M6-004 contract).
    HOLD stances are produced without thesis_status (exempt by spec).
    """
    _action_upper = action.upper()
    _action_lower = action.lower()

    def _stance(t: str) -> dict:
        base: dict = {
            "ticker": t,
            "action": action,
            "target_weight": weight,
            "confidence": confidence,
            "rationale": f"{persona} on {t}: momentum indicators positive.",
        }
        # HOLD is exempt; all other actions require thesis_status.
        if _action_upper != "HOLD":
            if _action_upper == "ADD":
                base["thesis_status"] = {
                    "verdict": "new",
                    "reason": f"Initiating medium-term thesis on {t}: secular growth driver identified.",
                }
            else:  # EXIT or REDUCE
                base["thesis_status"] = {
                    "verdict": "broken",
                    "reason": f"Medium-term thesis on {t} broken: key growth driver no longer intact.",
                }
        return base

    stances = [_stance(t) for t in debate_set]
    cf = counterfactual or {debate_set[0]: 0.15, "CASH": 0.85}
    schema = {
        "stances": stances,
        "counterfactual_portfolio": cf,
        "narrative_summary": f"{persona}: constructive on debate set.",
    }
    return json.dumps(schema)


def _make_all_round1_replies(
    debate_set: list[str],
    personas: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    slugs = personas or PERSONA_SLUGS_7
    return {slug: _make_round1_json(slug, debate_set, **kwargs) for slug in slugs}


# ---------------------------------------------------------------------------
# Component 13 — construct_debate_set
# ---------------------------------------------------------------------------

class TestConstructDebateSet:
    """Tests for Component 13 — the bounding rule instantiation."""

    _cfg = {"debate_set_ceiling": 40}

    def test_dedup_union_equals_direct_plus_peers(self) -> None:
        """Debate set = union of all direct tickers + cluster peers, de-duplicated."""
        ticker_sets = _uniform_ticker_sets(
            directs=["AAPL", "MSFT"],
            clusters={"AAPL": ["QCOM"], "MSFT": ["GOOGL"]},
        )
        results = _make_7_results(ticker_sets)
        debate = construct_debate_set(results, self._cfg)
        # AAPL, MSFT direct; QCOM, GOOGL peers.
        assert set(debate) == {"AAPL", "MSFT", "QCOM", "GOOGL"}

    def test_strict_superset_of_directly_shortlisted(self) -> None:
        """Debate set must contain all directly-shortlisted names (AC1)."""
        ticker_sets = _uniform_ticker_sets(
            directs=["AAPL", "MSFT", "NVDA"],
            clusters={"AAPL": ["QCOM"], "NVDA": ["AMD", "INTC"]},
        )
        results = _make_7_results(ticker_sets)
        debate = construct_debate_set(results, self._cfg)
        for direct in ["AAPL", "MSFT", "NVDA"]:
            assert direct in debate, f"Directly-shortlisted name {direct!r} missing from debate set"

    def test_ceiling_bound_trims_peers_first(self) -> None:
        """When union > ceiling, cluster peers are trimmed; direct names kept (AC1)."""
        # 3 direct + 5 peers = 8 total. Ceiling = 5 → must drop 3 peers.
        ticker_sets = _uniform_ticker_sets(
            directs=["AAPL", "MSFT", "NVDA"],
            clusters={
                "AAPL": ["QCOM", "AVGO"],
                "MSFT": ["GOOGL"],
                "NVDA": ["AMD", "INTC"],
            },
        )
        results = _make_7_results(ticker_sets)
        cfg = {"debate_set_ceiling": 5}
        debate = construct_debate_set(results, cfg)
        assert len(debate) <= 5
        # All direct names must survive.
        for direct in ["AAPL", "MSFT", "NVDA"]:
            assert direct in debate, f"Direct name {direct!r} was dropped — violation of bounding rule"

    def test_ceiling_exactly_met(self) -> None:
        """Ceiling == number of direct names + peers fills exactly to ceiling."""
        ticker_sets = _uniform_ticker_sets(
            directs=["AAPL", "MSFT"],
            clusters={"AAPL": ["QCOM", "AVGO", "BROADCOM"]},
        )
        results = _make_7_results(ticker_sets)
        cfg = {"debate_set_ceiling": 5}
        debate = construct_debate_set(results, cfg)
        assert len(debate) <= 5
        assert "AAPL" in debate
        assert "MSFT" in debate

    def test_no_peers_within_ceiling(self) -> None:
        """Direct-only set (no cluster peers) stays intact."""
        ticker_sets = _uniform_ticker_sets(directs=["AAPL", "MSFT", "NVDA"])
        results = _make_7_results(ticker_sets)
        debate = construct_debate_set(results, self._cfg)
        assert set(debate) == {"AAPL", "MSFT", "NVDA"}

    def test_direct_exceeds_ceiling_raises(self) -> None:
        """More directly-shortlisted names than ceiling → RuntimeError (Major-tier)."""
        ticker_sets = _uniform_ticker_sets(directs=["AAPL", "MSFT", "NVDA", "TSLA"])
        results = _make_7_results(ticker_sets)
        cfg = {"debate_set_ceiling": 3}
        with pytest.raises(RuntimeError, match="MAJOR"):
            construct_debate_set(results, cfg)

    def test_duplicate_tickers_across_personas(self) -> None:
        """Same ticker in multiple personas' shortlists appears once in the debate set."""
        # All 7 personas shortlist AAPL and MSFT.
        ticker_sets = _uniform_ticker_sets(directs=["AAPL", "MSFT"])
        results = _make_7_results(ticker_sets)
        debate = construct_debate_set(results, self._cfg)
        assert debate.count("AAPL") == 1
        assert debate.count("MSFT") == 1
        assert len(debate) == 2

    def test_cluster_peer_promoted_by_direct(self) -> None:
        """A ticker that appears as a peer of one persona and direct of another → counts as direct."""
        # Persona 1 shortlists AAPL with MSFT as a peer.
        # Persona 2 shortlists MSFT directly.
        # MSFT should be in direct pool, not peer pool.
        r1 = _make_persona_result("value", ["AAPL"], {"AAPL": ["MSFT"]})
        r2 = _make_persona_result("growth", ["MSFT"], {})
        # Fill remaining 5 with non-conflicting tickers.
        fillers = [
            _make_persona_result(slug, [f"T{i}"], {})
            for i, slug in enumerate(PERSONA_SLUGS_7[2:], start=1)
        ]
        results = [r1, r2] + fillers
        debate = construct_debate_set(results, self._cfg)
        # MSFT must appear exactly once.
        assert debate.count("MSFT") == 1
        assert "MSFT" in debate

    def test_empty_personas(self) -> None:
        """No research results → empty debate set (not a crash)."""
        debate = construct_debate_set([], self._cfg)
        assert debate == []


# ---------------------------------------------------------------------------
# Component 13/14 cross-check — AC2
# ---------------------------------------------------------------------------

class TestDebateSetCrossCheck:
    """AC2: |debate set| == count of distinct round-1 stance tickers."""

    def test_debate_set_size_equals_distinct_stance_tickers(self) -> None:
        ticker_sets = _uniform_ticker_sets(
            directs=["AAPL", "MSFT"],
            clusters={"AAPL": ["QCOM"]},
        )
        results = _make_7_results(ticker_sets)
        cfg = {"debate_set_ceiling": 40}
        debate = construct_debate_set(results, cfg)

        replies = _make_all_round1_replies(debate)
        capture = capture_round1_stances(
            debate,
            results,
            raw_round1_replies=replies,
            config={"max_position_weight": MAX_WEIGHT},
        )

        distinct_tickers = {s.ticker for s in capture.stances}
        assert len(distinct_tickers) == len(debate), (
            f"Cross-check failed: debate set has {len(debate)} tickers, "
            f"but {len(distinct_tickers)} distinct tickers in stances."
        )


# ---------------------------------------------------------------------------
# Component 14 — capture_round1_stances — parsing + normalization
# ---------------------------------------------------------------------------

class TestCaptureRound1StancesParsing:
    """Tests for Round-1 JSON parsing, normalization, and coverage checks."""

    _debate = ["AAPL", "MSFT", "NVDA"]
    _cfg = {"max_position_weight": MAX_WEIGHT}

    def _results(self) -> list[PersonaResearchResult]:
        ticker_sets = _uniform_ticker_sets(directs=self._debate)
        return _make_7_results(ticker_sets)

    def test_parse_canned_json_full_coverage(self) -> None:
        """Every persona has a stance for every debate-set ticker (AC3)."""
        results = self._results()
        replies = _make_all_round1_replies(self._debate)
        capture = capture_round1_stances(
            self._debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        covered = {(s.persona, s.ticker) for s in capture.stances}
        for slug in PERSONA_SLUGS_7:
            for ticker in self._debate:
                assert (slug, ticker) in covered, (
                    f"Missing stance for persona={slug!r} ticker={ticker!r}"
                )

    def test_stance_count_7_times_debate_set(self) -> None:
        """Total stances == 7 × |debate_set|; all round=1 (AC4)."""
        results = self._results()
        replies = _make_all_round1_replies(self._debate)
        capture = capture_round1_stances(
            self._debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        assert len(capture.stances) == 7 * len(self._debate), (
            f"Expected {7 * len(self._debate)} stances, got {len(capture.stances)}"
        )
        assert all(s.round == 1 for s in capture.stances), (
            "All stances must carry round=1 in M2"
        )

    def test_action_vocabulary_normalised_uppercase(self) -> None:
        """Uppercase ADD/REDUCE/EXIT/HOLD are normalised to lowercase."""
        results = self._results()
        # Use lowercase action in JSON — the schema uses uppercase but lowercase must work too.
        debate = ["AAPL"]
        for raw_action, expected in [
            ("ADD", "add"), ("REDUCE", "reduce"), ("EXIT", "exit"), ("HOLD", "hold"),
            ("add", "add"), ("reduce", "reduce"), ("exit", "exit"), ("hold", "hold"),
        ]:
            replies = {
                slug: _make_round1_json(slug, debate, action=raw_action, weight=0.10)
                for slug in PERSONA_SLUGS_7
            }
            ticker_sets = _uniform_ticker_sets(directs=debate)
            results_local = _make_7_results(ticker_sets)
            capture = capture_round1_stances(
                debate, results_local,
                raw_round1_replies=replies,
                config=self._cfg,
            )
            for s in capture.stances:
                assert s.action == expected, (
                    f"action={raw_action!r} should normalise to {expected!r}, got {s.action!r}"
                )

    def test_out_of_domain_action_raises(self) -> None:
        """Unknown action raises Round1ParseError — no silent coercion (AC3)."""
        results = self._results()
        bad_replies = dict(_make_all_round1_replies(self._debate))
        # Inject an invalid action for one persona.
        bad_json = _make_round1_json("value", self._debate)
        data = json.loads(bad_json)
        data["stances"][0]["action"] = "BUY"   # not in vocabulary
        bad_replies["value"] = json.dumps(data)

        with pytest.raises(Round1ParseError, match="not in the 4-value vocabulary"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=bad_replies,
                config=self._cfg,
            )

    def test_out_of_domain_confidence_raises(self) -> None:
        """Confidence outside 1..5 raises Round1ParseError (AC3)."""
        results = self._results()
        bad_replies = dict(_make_all_round1_replies(self._debate))
        data = json.loads(bad_replies["value"])
        data["stances"][0]["confidence"] = 6     # invalid
        bad_replies["value"] = json.dumps(data)

        with pytest.raises(Round1ParseError, match="outside 1..5"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=bad_replies,
                config=self._cfg,
            )

    def test_confidence_boundary_valid(self) -> None:
        """Confidence values 1 and 5 are both valid."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        for conf in [1, 5]:
            replies = _make_all_round1_replies(debate, confidence=conf)
            capture = capture_round1_stances(
                debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )
            assert all(s.confidence == conf for s in capture.stances)

    def test_out_of_domain_weight_raises(self) -> None:
        """target_weight > max_position_weight raises Round1ParseError (AC3)."""
        results = self._results()
        bad_replies = dict(_make_all_round1_replies(self._debate))
        data = json.loads(bad_replies["value"])
        data["stances"][0]["target_weight"] = 0.99  # far exceeds 0.20 cap
        bad_replies["value"] = json.dumps(data)

        with pytest.raises(Round1ParseError, match="outside"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=bad_replies,
                config=self._cfg,
            )

    def test_weight_zero_valid(self) -> None:
        """target_weight = 0.0 is valid (e.g. EXIT stance)."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = _make_all_round1_replies(debate, action="EXIT", weight=0.0)
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        assert all(s.target_weight == 0.0 for s in capture.stances)

    def test_missing_cash_in_counterfactual_raises(self) -> None:
        """counterfactual_portfolio without CASH key raises Round1ParseError."""
        results = self._results()
        bad_replies = dict(_make_all_round1_replies(self._debate))
        data = json.loads(bad_replies["value"])
        # Remove CASH from counterfactual.
        data["counterfactual_portfolio"] = {"AAPL": 0.15, "MSFT": 0.10}
        bad_replies["value"] = json.dumps(data)

        with pytest.raises(Round1ParseError, match="CASH"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=bad_replies,
                config=self._cfg,
            )

    def test_missing_ticker_coverage_raises(self) -> None:
        """Persona omits a debate-set ticker → Round1ParseError (no missing cells)."""
        results = self._results()
        bad_replies = dict(_make_all_round1_replies(self._debate))
        data = json.loads(bad_replies["value"])
        # Remove the last stance so NVDA is uncovered.
        data["stances"] = [s for s in data["stances"] if s["ticker"] != "NVDA"]
        bad_replies["value"] = json.dumps(data)

        with pytest.raises(Round1ParseError, match="missing stances"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=bad_replies,
                config=self._cfg,
            )

    def test_extra_ticker_not_in_debate_set_raises(self) -> None:
        """Stance for a ticker not in the debate set raises Round1ParseError."""
        results = self._results()
        bad_replies = dict(_make_all_round1_replies(self._debate))
        data = json.loads(bad_replies["value"])
        # Add a stance for a ticker not in the debate set.
        data["stances"].append({
            "ticker": "TSLA",
            "action": "ADD",
            "target_weight": 0.05,
            "confidence": 3,
            "rationale": "Extra ticker not in debate set.",
        })
        bad_replies["value"] = json.dumps(data)

        with pytest.raises(Round1ParseError, match="not in the debate set"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=bad_replies,
                config=self._cfg,
            )

    def test_counterfactual_negative_weight_raises(self) -> None:
        """Negative weight in counterfactual_portfolio raises Round1ParseError."""
        results = self._results()
        bad_replies = dict(_make_all_round1_replies(self._debate))
        data = json.loads(bad_replies["value"])
        data["counterfactual_portfolio"]["AAPL"] = -0.05
        bad_replies["value"] = json.dumps(data)

        with pytest.raises(Round1ParseError, match="< 0"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=bad_replies,
                config=self._cfg,
            )

    def test_counterfactual_position_over_cap_raises(self) -> None:
        """Non-CASH position weight > max_position_weight raises Round1ParseError."""
        results = self._results()
        bad_replies = dict(_make_all_round1_replies(self._debate))
        data = json.loads(bad_replies["value"])
        data["counterfactual_portfolio"]["AAPL"] = 0.90   # far over 0.20 cap
        bad_replies["value"] = json.dumps(data)

        with pytest.raises(Round1ParseError, match="exceeds max_position_weight"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=bad_replies,
                config=self._cfg,
            )

    def test_missing_persona_reply_raises(self) -> None:
        """No reply provided for a persona → Round1ParseError."""
        results = self._results()
        replies = _make_all_round1_replies(self._debate)
        # Remove one persona's reply.
        del replies["value"]

        with pytest.raises(Round1ParseError, match="No Round-1 reply"):
            capture_round1_stances(
                self._debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )


# ---------------------------------------------------------------------------
# Component 14 — thesis_status contract (M6-004)
# ---------------------------------------------------------------------------

class TestThesisStatusContract:
    """Tests for M6-004: thesis_status required on EXIT/REDUCE/ADD; HOLD exempt.

    Coverage matrix:
      test_exit_with_thesis_status_parses_clean          — valid EXIT + thesis_status accepted
      test_reduce_with_thesis_status_parses_clean        — valid REDUCE + thesis_status accepted
      test_add_with_thesis_status_parses_clean           — valid ADD + thesis_status accepted
      test_hold_without_thesis_status_parses_clean       — HOLD with no thesis_status accepted (exempt)
      test_hold_with_thesis_status_parses_clean          — HOLD with thesis_status also accepted
      test_exit_missing_thesis_status_raises             — EXIT missing thesis_status → parse error
      test_reduce_missing_thesis_status_raises           — REDUCE missing thesis_status → parse error
      test_add_missing_thesis_status_raises              — ADD missing thesis_status → parse error
      test_exit_empty_reason_raises                      — EXIT with empty reason → parse error
      test_exit_invalid_verdict_raises                   — EXIT with verdict=new → parse error
      test_add_invalid_verdict_raises                    — ADD with verdict=broken → parse error
      test_thesis_status_serialized_into_rationale       — thesis_status folded into agent_stances.rationale
      test_hold_rationale_unchanged_no_thesis_status     — HOLD rationale not modified when no thesis_status
      test_thesis_status_present_in_rationale_text       — THESIS STATUS: label present in rationale
      test_intact_verdict_allowed_on_exit                — EXIT with verdict=intact is a valid parse (flag for C11, not parse error)
      test_intact_verdict_allowed_on_reduce              — REDUCE with verdict=intact is a valid parse
      test_multiple_action_stances_all_require_ts        — all EXIT/REDUCE/ADD in one reply need thesis_status
    """

    _debate = ["AAPL", "MSFT"]
    _cfg = {"max_position_weight": MAX_WEIGHT}

    def _results(self) -> list[PersonaResearchResult]:
        ticker_sets = _uniform_ticker_sets(directs=self._debate)
        return _make_7_results(ticker_sets)

    def _make_replies_with_override(self, persona: str, debate: list[str], override: dict) -> dict[str, str]:
        """All 7 personas return valid JSON; one persona gets an overridden stance."""
        replies = _make_all_round1_replies(debate)
        base_data = json.loads(replies[persona])
        # Apply overrides to first stance
        base_data["stances"][0].update(override)
        replies[persona] = json.dumps(base_data)
        return replies

    def _single_stance_reply(
        self,
        persona: str,
        ticker: str,
        action: str,
        thesis_status: dict | None = None,
        include_ts: bool = True,
    ) -> dict[str, str]:
        """Build replies where one persona has a single-ticker debate with custom thesis_status."""
        debate = [ticker]
        replies = {}
        for slug in PERSONA_SLUGS_7:
            stance: dict = {
                "ticker": ticker,
                "action": action,
                "target_weight": 0.10,
                "confidence": 3,
                "rationale": f"{slug} on {ticker}: thesis evaluation.",
            }
            if include_ts and thesis_status is not None:
                stance["thesis_status"] = thesis_status
            elif not include_ts:
                pass  # explicitly no thesis_status
            stances_list = [stance]
            replies[slug] = json.dumps({
                "stances": stances_list,
                "counterfactual_portfolio": {ticker: 0.10, "CASH": 0.90},
                "narrative_summary": f"{slug}: evaluating {ticker}.",
            })
        return replies

    # --- Happy-path: valid thesis_status on action stances ---

    def test_exit_with_thesis_status_parses_clean(self) -> None:
        """Valid EXIT stance with thesis_status accepted."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "value", "AAPL", "EXIT",
            thesis_status={"verdict": "broken", "reason": "Core moat eroded by regulatory change."},
        )
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        exit_stances = [s for s in capture.stances if s.action == "exit"]
        assert len(exit_stances) == 7

    def test_reduce_with_thesis_status_parses_clean(self) -> None:
        """Valid REDUCE stance with thesis_status accepted."""
        debate = ["MSFT"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "growth", "MSFT", "REDUCE",
            thesis_status={"verdict": "intact", "reason": "Thesis intact but position sized for risk."},
        )
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        reduce_stances = [s for s in capture.stances if s.action == "reduce"]
        assert len(reduce_stances) == 7

    def test_add_with_thesis_status_parses_clean(self) -> None:
        """Valid ADD stance with thesis_status (verdict=new) accepted."""
        debate = ["NVDA"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "technical", "NVDA", "ADD",
            thesis_status={"verdict": "new", "reason": "Initiating: AI infrastructure cycle thesis."},
        )
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        add_stances = [s for s in capture.stances if s.action == "add"]
        assert len(add_stances) == 7

    def test_hold_without_thesis_status_parses_clean(self) -> None:
        """HOLD stance with no thesis_status is exempt — must parse cleanly."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "value", "AAPL", "HOLD",
            thesis_status=None,
            include_ts=False,
        )
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        hold_stances = [s for s in capture.stances if s.action == "hold"]
        assert len(hold_stances) == 7

    def test_hold_with_thesis_status_parses_clean(self) -> None:
        """HOLD stance WITH thesis_status also accepted (optional, not enforced)."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "value", "AAPL", "HOLD",
            thesis_status={"verdict": "intact", "reason": "Monitoring for re-entry trigger."},
        )
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        hold_stances = [s for s in capture.stances if s.action == "hold"]
        assert len(hold_stances) == 7

    # --- Failure cases: missing / empty / invalid thesis_status ---

    def test_exit_missing_thesis_status_raises(self) -> None:
        """EXIT stance missing thesis_status → Round1ParseError (fail-loudly)."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply("value", "AAPL", "EXIT", include_ts=False)
        with pytest.raises(Round1ParseError, match="requires 'thesis_status'"):
            capture_round1_stances(
                debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )

    def test_reduce_missing_thesis_status_raises(self) -> None:
        """REDUCE stance missing thesis_status → Round1ParseError."""
        debate = ["MSFT"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply("growth", "MSFT", "REDUCE", include_ts=False)
        with pytest.raises(Round1ParseError, match="requires 'thesis_status'"):
            capture_round1_stances(
                debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )

    def test_add_missing_thesis_status_raises(self) -> None:
        """ADD stance missing thesis_status → Round1ParseError."""
        debate = ["NVDA"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply("technical", "NVDA", "ADD", include_ts=False)
        with pytest.raises(Round1ParseError, match="requires 'thesis_status'"):
            capture_round1_stances(
                debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )

    def test_exit_empty_reason_raises(self) -> None:
        """EXIT with thesis_status.reason='' → Round1ParseError."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "value", "AAPL", "EXIT",
            thesis_status={"verdict": "broken", "reason": ""},
        )
        with pytest.raises(Round1ParseError, match="reason is empty"):
            capture_round1_stances(
                debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )

    def test_exit_invalid_verdict_raises(self) -> None:
        """EXIT with verdict='new' (ADD-only) → Round1ParseError."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "value", "AAPL", "EXIT",
            thesis_status={"verdict": "new", "reason": "Should not be 'new' on EXIT."},
        )
        with pytest.raises(Round1ParseError, match="not valid for action"):
            capture_round1_stances(
                debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )

    def test_add_invalid_verdict_raises(self) -> None:
        """ADD with verdict='broken' (EXIT/REDUCE-only) → Round1ParseError."""
        debate = ["NVDA"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "technical", "NVDA", "ADD",
            thesis_status={"verdict": "broken", "reason": "Should not be 'broken' on ADD."},
        )
        with pytest.raises(Round1ParseError, match="not valid for action"):
            capture_round1_stances(
                debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )

    # --- Serialization into rationale ---

    def test_thesis_status_serialized_into_rationale(self) -> None:
        """thesis_status content appears in the written agent_stances.rationale."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        ts_reason = "Secular demand shift thesis broken: PC refresh cycle stalled."
        replies = self._single_stance_reply(
            "value", "AAPL", "EXIT",
            thesis_status={"verdict": "broken", "reason": ts_reason},
        )
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        exit_stances = [s for s in capture.stances if s.action == "exit"]
        for s in exit_stances:
            assert ts_reason in s.rationale, (
                f"thesis_status reason not found in rationale for {s.persona}: {s.rationale!r}"
            )
            assert "broken" in s.rationale, (
                f"thesis_status verdict not found in rationale for {s.persona}: {s.rationale!r}"
            )

    def test_thesis_status_present_in_rationale_text(self) -> None:
        """'THESIS STATUS:' label is present in rationale for action stances."""
        debate = ["MSFT"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "growth", "MSFT", "ADD",
            thesis_status={"verdict": "new", "reason": "Cloud margins expanding: new thesis initiated."},
        )
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        for s in capture.stances:
            assert "THESIS STATUS:" in s.rationale, (
                f"'THESIS STATUS:' label missing from rationale for {s.persona}: {s.rationale!r}"
            )

    def test_hold_rationale_unchanged_no_thesis_status(self) -> None:
        """HOLD with no thesis_status: rationale is the base text, no THESIS STATUS label."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply("value", "AAPL", "HOLD", include_ts=False)
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        for s in capture.stances:
            assert "THESIS STATUS:" not in s.rationale, (
                f"'THESIS STATUS:' unexpectedly present on HOLD rationale: {s.rationale!r}"
            )

    # --- Verdict boundary cases ---

    def test_intact_verdict_allowed_on_exit(self) -> None:
        """EXIT with verdict=intact is a valid parse (signals for C11, not a parse error)."""
        debate = ["AAPL"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "value", "AAPL", "EXIT",
            thesis_status={"verdict": "intact", "reason": "Position exited for risk management, thesis intact."},
        )
        # Must NOT raise — intact on EXIT is allowed at parse layer; C11 flags it.
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        exit_stances = [s for s in capture.stances if s.action == "exit"]
        assert len(exit_stances) == 7

    def test_intact_verdict_allowed_on_reduce(self) -> None:
        """REDUCE with verdict=intact is a valid parse."""
        debate = ["MSFT"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies = self._single_stance_reply(
            "growth", "MSFT", "REDUCE",
            thesis_status={"verdict": "intact", "reason": "Trimming for position sizing; thesis unchanged."},
        )
        capture = capture_round1_stances(
            debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        reduce_stances = [s for s in capture.stances if s.action == "reduce"]
        assert len(reduce_stances) == 7

    def test_multiple_action_stances_all_require_ts(self) -> None:
        """All EXIT/REDUCE/ADD stances in a multi-ticker reply need thesis_status.

        Verifies enforcement is per-stance, not just on the first one.
        """
        # Build a reply with 2 tickers: AAPL=EXIT (has TS), MSFT=ADD (missing TS).
        debate = ["AAPL", "MSFT"]
        ticker_sets = _uniform_ticker_sets(directs=debate)
        results = _make_7_results(ticker_sets)
        replies: dict[str, str] = {}
        for slug in PERSONA_SLUGS_7:
            stances = [
                {
                    "ticker": "AAPL",
                    "action": "EXIT",
                    "target_weight": 0.0,
                    "confidence": 3,
                    "rationale": "Exiting AAPL.",
                    "thesis_status": {"verdict": "broken", "reason": "Valuation thesis broken by margin compression."},
                },
                {
                    "ticker": "MSFT",
                    "action": "ADD",
                    "target_weight": 0.10,
                    "confidence": 4,
                    "rationale": "Adding MSFT.",
                    # Deliberately missing thesis_status on this ADD
                },
            ]
            replies[slug] = json.dumps({
                "stances": stances,
                "counterfactual_portfolio": {"MSFT": 0.10, "CASH": 0.90},
                "narrative_summary": f"{slug}: mixed conviction.",
            })
        with pytest.raises(Round1ParseError, match="requires 'thesis_status'"):
            capture_round1_stances(
                debate, results,
                raw_round1_replies=replies,
                config=self._cfg,
            )


# ---------------------------------------------------------------------------
# Component 14 — commit-before-reveal (AC5)
# ---------------------------------------------------------------------------

class TestCommitBeforeReveal:
    """AC5: no persona's Round-1 prompt contains another persona's Round-1 output."""

    _debate = ["AAPL", "MSFT", "NVDA"]
    _cfg = {"max_position_weight": MAX_WEIGHT}

    def _results(self) -> list[PersonaResearchResult]:
        ticker_sets = _uniform_ticker_sets(directs=self._debate)
        return _make_7_results(ticker_sets)

    def test_prompts_built_before_any_reply_processed(self) -> None:
        """Round1Capture.prompts has an entry for every persona slug."""
        results = self._results()
        replies = _make_all_round1_replies(self._debate)
        capture = capture_round1_stances(
            self._debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        assert set(capture.prompts.keys()) == set(PERSONA_SLUGS_7), (
            "prompts dict must have an entry for every persona"
        )

    def test_prompt_isolation_no_peer_output(self) -> None:
        """No persona's prompt contains any other persona's Round-1 reply text.

        This is the structural guarantee of commit-before-reveal (AC5).
        The prompt is built from the persona's own research only; the raw Round-1
        replies (which contain actual output text) must not appear in any prompt.
        """
        results = self._results()
        # Make each persona's reply contain a distinctive sentinel string.
        sentinels: dict[str, str] = {
            slug: f"UNIQUE_SENTINEL_{slug.upper().replace('-','_')}"
            for slug in PERSONA_SLUGS_7
        }
        replies: dict[str, str] = {}
        for slug in PERSONA_SLUGS_7:
            data = json.loads(_make_round1_json(slug, self._debate))
            data["narrative_summary"] = sentinels[slug]
            replies[slug] = json.dumps(data)

        capture = capture_round1_stances(
            self._debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )

        for slug in PERSONA_SLUGS_7:
            prompt = capture.prompts[slug]
            for other_slug, sentinel in sentinels.items():
                if other_slug == slug:
                    continue   # a persona's own output may appear in its own prompt
                assert sentinel not in prompt, (
                    f"COMMIT-BEFORE-REVEAL VIOLATION: persona {slug!r}'s prompt "
                    f"contains {other_slug!r}'s Round-1 output sentinel {sentinel!r}. "
                    "This defeats the anti-consensus mechanic."
                )

    def test_custom_prompt_builder_injected(self) -> None:
        """The prompt_builder callable is invoked once per persona."""
        results = self._results()
        replies = _make_all_round1_replies(self._debate)
        called_for: list[str] = []

        def tracking_builder(
            slug: str,
            debate_set: list[str],
            result: Any,
        ) -> str:
            called_for.append(slug)
            return f"CUSTOM_PROMPT_FOR_{slug}"

        capture = capture_round1_stances(
            self._debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
            prompt_builder=tracking_builder,
        )
        assert set(called_for) == set(PERSONA_SLUGS_7), (
            "prompt_builder must be called exactly once per persona"
        )
        for slug in PERSONA_SLUGS_7:
            assert capture.prompts[slug] == f"CUSTOM_PROMPT_FOR_{slug}"


# ---------------------------------------------------------------------------
# Round1Capture shape
# ---------------------------------------------------------------------------

class TestRound1CaptureShape:
    """Verify Round1Capture has the correct structure."""

    _debate = ["AAPL", "MSFT"]
    _cfg = {"max_position_weight": MAX_WEIGHT}

    def _results(self) -> list[PersonaResearchResult]:
        ticker_sets = _uniform_ticker_sets(directs=self._debate)
        return _make_7_results(ticker_sets)

    def test_round1_capture_has_prompts_field(self) -> None:
        """Round1Capture.prompts is a dict keyed by persona slug."""
        results = self._results()
        replies = _make_all_round1_replies(self._debate)
        capture = capture_round1_stances(
            self._debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        assert isinstance(capture.prompts, dict)
        assert all(isinstance(k, str) for k in capture.prompts)

    def test_counterfactuals_keyed_by_persona(self) -> None:
        """counterfactuals dict has 7 entries; each has a CASH key."""
        results = self._results()
        replies = _make_all_round1_replies(self._debate)
        capture = capture_round1_stances(
            self._debate, results,
            raw_round1_replies=replies,
            config=self._cfg,
        )
        assert set(capture.counterfactuals.keys()) == set(PERSONA_SLUGS_7)
        for slug, cf in capture.counterfactuals.items():
            assert "CASH" in cf, f"counterfactual for {slug!r} missing CASH key"
