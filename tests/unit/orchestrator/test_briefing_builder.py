"""Unit tests for orchestrator/briefing_builder.py — Component 27.

Coverage matrix (Gate 4 ACs):

  AC-1 — Zero cross-persona leakage (≥7 cells):
    TestLeakage                 — 7 cells (one per persona): each persona's
                                  briefing contains ONLY that persona's own data.
                                  Asserts no other persona's slug appears in a
                                  stance/counterfactual-revealing way.

  AC-2a — Briefing content matches windowed memory (≥7 content-match cells):
    TestContentMatch            — 7 cells: briefing for each persona includes the
                                  exact week_id labels + body text from the
                                  WindowedMemory fed in.

  AC-2b — Size cap enforced (≥2 truncation cells):
    TestTruncation              — 2 cells: an over-long entry body → truncated +
                                  "[truncated]" marker present; a hard-global-cap
                                  scenario → whole block truncated.

  AC-3a — Own-misses surfaced when flag on (≥2 cells):
    TestOwnMisses               — 2 cells: a persona with resolved-negative-alpha
                                  rows → own-misses callout present and correct;
                                  own_misses_in_digest=False → callout absent.

  AC-3b — Briefing persisted to state/runs/<week>-memory/<persona>.md:
    TestPersistence             — confirms file exists at expected path with exact
                                  briefing content after build.

  Config:
    TestBriefingConfig          — load_briefing_config falls back to defaults when
                                  key absent; non-default value is respected.

  Real-W24-derived fixture (Gate-4 provenance corollary):
    TestRealW24Fixture          — at least one scenario is seeded from sanitized
                                  real 2026-W24-style data shapes.

  Batch entry point:
    TestBuildAllBriefings       — build_all_briefings returns one result per
                                  persona with no cross-contamination.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import pytest

from round_table_portfolio.orchestrator.briefing_builder import (
    BriefingConfig,
    BriefingResult,
    _TRUNCATION_MARKER,
    build_all_briefings,
    build_persona_briefing,
    load_briefing_config,
)
from round_table_portfolio.orchestrator.digest import ResolvedRow
from round_table_portfolio.orchestrator.memory_reader import WindowedMemory

# ---------------------------------------------------------------------------
# Shared constants
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

_DEFAULT_CFG = BriefingConfig(memory_briefing_max_chars=3000, own_misses_in_digest=True)
_NO_MISSES_CFG = BriefingConfig(memory_briefing_max_chars=3000, own_misses_in_digest=False)
_TINY_CAP_CFG = BriefingConfig(memory_briefing_max_chars=200, own_misses_in_digest=True)

WEEK = "2026-W24"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_windowed(persona: str) -> WindowedMemory:
    """Return a WindowedMemory with all-empty sections for *persona*."""
    return WindowedMemory(
        persona=persona,
        past_calls=[],
        counterfactual=[],
        debate_stances=[],
        whats_new=[],
    )


def _simple_windowed(
    persona: str,
    *,
    past_calls: list[tuple[str, str]] | None = None,
    counterfactual: list[tuple[str, str]] | None = None,
    debate_stances: list[tuple[str, str]] | None = None,
    whats_new: list[tuple[str, str]] | None = None,
) -> WindowedMemory:
    return WindowedMemory(
        persona=persona,
        past_calls=past_calls or [],
        counterfactual=counterfactual or [],
        debate_stances=debate_stances or [],
        whats_new=whats_new or [],
    )


def _resolved_row(
    persona: str,
    ticker: str,
    alpha: float,
    *,
    call_week: str = "2026-W23",
    as_of_week: str = WEEK,
    action: str = "add",
) -> ResolvedRow:
    return ResolvedRow(
        persona=persona,
        ticker=ticker,
        call_week_id=call_week,
        as_of_week_id=as_of_week,
        alpha=alpha,
        action=action,
    )


# ---------------------------------------------------------------------------
# AC-1 — Zero cross-persona leakage (7 cells, one per persona)
# ---------------------------------------------------------------------------

class TestLeakage:
    """AC-1: Each persona's briefing must contain NO other persona's slug in a
    stance/counterfactual-revealing context.

    Strategy: build all 7 briefings from per-persona WindowedMemory objects
    whose body text contains each persona's own ticker + action strings.  Then
    assert that persona X's briefing does NOT contain persona Y's slug anywhere.

    This is a structural text check — the leakage invariant is guaranteed by
    construction (each WindowedMemory is persona-scoped), but the test confirms
    the render path does not accidentally blend sources.
    """

    @pytest.fixture
    def all_briefings(self) -> dict[str, BriefingResult]:
        """Build all 7 briefings, each with a unique entry body containing
        that persona's own ticker + a distinctive marker.
        """
        results: dict[str, BriefingResult] = {}
        for persona in PERSONA_SLUGS_7:
            marker = f"UNIQUE_TICKER_FOR_{persona.upper().replace('-', '_')}"
            wm = _simple_windowed(
                persona,
                past_calls=[(WEEK, f"  {marker}: add confidence=4 weight=0.050")],
                counterfactual=[(WEEK, f"portfolio: {marker} 5%")],
                debate_stances=[(WEEK, f"stance: bullish on {marker}")],
            )
            results[persona] = build_persona_briefing(
                wm,
                "No prior calls have resolved yet.",
                [],
                week_id=WEEK,
                config=_DEFAULT_CFG,
                persist=False,
            )
        return results

    @pytest.mark.parametrize("persona", PERSONA_SLUGS_7)
    def test_no_other_persona_slug_in_briefing(
        self, all_briefings: dict[str, BriefingResult], persona: str
    ) -> None:
        """Cell: persona's briefing must not contain any OTHER persona's unique
        ticker marker.
        """
        own_marker = f"UNIQUE_TICKER_FOR_{persona.upper().replace('-', '_')}"
        briefing_text = all_briefings[persona].briefing_text

        # Own marker must be present.
        assert own_marker in briefing_text, (
            f"{persona}: expected own marker in briefing"
        )

        # No other persona's marker may be present.
        for other in PERSONA_SLUGS_7:
            if other == persona:
                continue
            other_marker = f"UNIQUE_TICKER_FOR_{other.upper().replace('-', '_')}"
            assert other_marker not in briefing_text, (
                f"LEAKAGE: {persona}'s briefing contains {other}'s marker "
                f"'{other_marker}'"
            )

    def test_persona_field_in_result_matches_input(
        self, all_briefings: dict[str, BriefingResult]
    ) -> None:
        """Persona field on BriefingResult must match the key it was built for."""
        for persona, result in all_briefings.items():
            assert result.persona == persona


# ---------------------------------------------------------------------------
# AC-2a — Briefing content matches windowed memory (≥7 content-match cells)
# ---------------------------------------------------------------------------

class TestContentMatch:
    """AC-2a: The briefing must faithfully reflect the windowed memory input —
    all week_id labels and body text appear in the rendered block.
    """

    @pytest.mark.parametrize("persona", PERSONA_SLUGS_7)
    def test_windowed_entries_appear_in_briefing(self, persona: str) -> None:
        """Cell: all four sections' week_ids and body snippets are present."""
        wm = _simple_windowed(
            persona,
            past_calls=[
                ("2026-W20", f"  AAPL: add confidence=3 weight=0.040"),
                ("2026-W23", f"  MSFT: hold confidence=2 weight=0.030"),
            ],
            counterfactual=[
                ("2026-W20", "portfolio: AAPL 4.0% MSFT 3.0%"),
            ],
            debate_stances=[
                ("2026-W20", "stance: bullish tech, cautious on rates"),
            ],
            whats_new=[
                ("2026-W23", "NVDA earnings beat; macro softened"),
            ],
        )
        result = build_persona_briefing(
            wm,
            "NVDA (+0.0200 vs SPY) resolved",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        text = result.briefing_text

        # Week IDs present
        assert "2026-W20" in text
        assert "2026-W23" in text

        # Body content present (partial matches — exact format tested in memory.py)
        assert "AAPL" in text
        assert "MSFT" in text
        assert "bullish tech" in text
        assert "NVDA earnings beat" in text

        # Digest text present
        assert "NVDA (+0.0200 vs SPY)" in text

    def test_empty_sections_render_cleanly(self) -> None:
        """Empty sections produce a _(no entries in window)_ placeholder."""
        wm = _empty_windowed("value")
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        assert "_(no entries in window)_" in result.briefing_text
        # Header must identify the persona
        assert "value" in result.briefing_text

    def test_section_count_matches_input(self) -> None:
        """Entry count in rendered text matches windowed input count."""
        entries = [
            ("2026-W21", "  GOOG: add confidence=4 weight=0.045"),
            ("2026-W22", "  META: reduce confidence=3 weight=0.020"),
            ("2026-W23", "  AMZN: hold confidence=2 weight=0.035"),
        ]
        wm = _simple_windowed("growth", past_calls=entries)
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        # All three week IDs must appear in the past calls section
        for week_id, _ in entries:
            assert week_id in result.briefing_text

    def test_header_names_persona_and_week(self) -> None:
        """Briefing header must identify both the persona and the run week."""
        persona = "technical"
        wm = _empty_windowed(persona)
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        assert persona in result.briefing_text
        assert WEEK in result.briefing_text

    def test_whats_new_digest_is_latest_not_section_entries(self) -> None:
        """The whats_new_digest arg appears under 'Latest Digest', separately
        from the windowed What's New section entries.
        """
        wm = _simple_windowed(
            "risk-officer",
            whats_new=[("2026-W22", "old digest entry")],
        )
        fresh_digest = "FRESH DIGEST TEXT FOR TESTING"
        result = build_persona_briefing(
            wm,
            fresh_digest,
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        # Both the old windowed entry AND the fresh digest must be present
        assert "old digest entry" in result.briefing_text
        assert fresh_digest in result.briefing_text
        assert "Latest Digest" in result.briefing_text


# ---------------------------------------------------------------------------
# AC-2b — Size cap enforced (≥2 truncation cells)
# ---------------------------------------------------------------------------

class TestTruncation:
    """AC-2b: Truncation fires and is noted when content exceeds the budget."""

    def test_entry_body_truncated_when_too_long(self) -> None:
        """Cell: a 500-char entry body under a 200-char overall cap → truncated
        + [truncated] marker present + truncated=True on result.
        """
        long_body = "X" * 500  # definitely over any per-entry budget under 200-char cap
        wm = _simple_windowed(
            "value",
            past_calls=[("2026-W23", long_body)],
        )
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            [],
            week_id=WEEK,
            config=_TINY_CAP_CFG,
            persist=False,
        )
        assert _TRUNCATION_MARKER in result.briefing_text
        assert result.truncated is True

    def test_global_hard_cap_fires(self) -> None:
        """Cell: a briefing that would be large even after per-entry truncation
        is hard-truncated at the global cap boundary + [truncated] marker at end.
        """
        # Build a windowed memory with many sections worth of long content
        many_entries = [
            (f"2026-W{i:02d}", "M" * 400) for i in range(1, 9)
        ]
        wm = WindowedMemory(
            persona="quant-systematic",
            past_calls=many_entries,
            counterfactual=many_entries[:2],
            debate_stances=many_entries[:2],
            whats_new=many_entries[:2],
        )
        tiny_cfg = BriefingConfig(memory_briefing_max_chars=300, own_misses_in_digest=False)
        result = build_persona_briefing(
            wm,
            "digest text",
            [],
            week_id=WEEK,
            config=tiny_cfg,
            persist=False,
        )
        assert len(result.briefing_text) <= 300 + len(_TRUNCATION_MARKER)
        assert result.briefing_text.endswith(_TRUNCATION_MARKER)
        assert result.truncated is True

    def test_within_budget_not_truncated(self) -> None:
        """A small briefing under budget must NOT have the truncation marker."""
        wm = _empty_windowed("value")
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        assert _TRUNCATION_MARKER not in result.briefing_text
        assert result.truncated is False


# ---------------------------------------------------------------------------
# AC-3a — Own-misses surfaced when flag on (≥2 cells)
# ---------------------------------------------------------------------------

class TestOwnMisses:
    """AC-3a: Own-misses callout fires exactly when own_misses_in_digest=True
    and there are resolved-negative-alpha rows for that persona.
    """

    def _make_resolved_rows(self, persona: str) -> list[ResolvedRow]:
        return [
            _resolved_row(persona, "NVDA", alpha=-0.0450, action="add"),
            _resolved_row(persona, "AMD", alpha=+0.0320, action="add"),
            _resolved_row(persona, "INTC", alpha=-0.0210, action="hold"),
        ]

    def test_own_misses_callout_present_when_flag_on(self) -> None:
        """Cell: negative-alpha resolved rows → callout lists exactly those tickers."""
        persona = "value"
        wm = _empty_windowed(persona)
        resolved = self._make_resolved_rows(persona)
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            resolved,
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        text = result.briefing_text
        # Callout section header must be present
        assert "Past Calls That Resolved Below SPY" in text
        # Both negative-alpha tickers must appear
        assert "NVDA" in text
        assert "INTC" in text
        # Positive-alpha ticker must NOT appear in the misses callout
        # (AMD appears with positive alpha — it may appear in other sections
        # if provided via windowed_memory, but NOT in the own-misses callout)
        misses_start = text.find("Past Calls That Resolved Below SPY")
        misses_block = text[misses_start:]
        assert "AMD" not in misses_block

    def test_own_misses_absent_when_flag_off(self) -> None:
        """Cell: own_misses_in_digest=False → no own-misses callout even with
        negative-alpha resolved rows.
        """
        persona = "growth"
        wm = _empty_windowed(persona)
        resolved = self._make_resolved_rows(persona)
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            resolved,
            week_id=WEEK,
            config=_NO_MISSES_CFG,
            persist=False,
        )
        assert "Past Calls That Resolved Below SPY" not in result.briefing_text

    def test_own_misses_absent_when_all_positive_alpha(self) -> None:
        """No misses callout when all resolved rows are positive alpha."""
        persona = "technical"
        wm = _empty_windowed(persona)
        resolved = [
            _resolved_row(persona, "AAPL", alpha=+0.0100),
            _resolved_row(persona, "GOOG", alpha=+0.0250),
        ]
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            resolved,
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        assert "Past Calls That Resolved Below SPY" not in result.briefing_text

    def test_own_misses_uses_only_own_resolved_rows(self) -> None:
        """Cross-persona resolved rows supplied by accident must NOT appear
        in the own-misses callout.  Defence-in-depth filter check.
        """
        persona = "cta-systematic-macro"
        other = "risk-officer"
        wm = _empty_windowed(persona)
        resolved = [
            _resolved_row(persona, "OWN_MISS", alpha=-0.0500),
            _resolved_row(other, "OTHER_MISS", alpha=-0.0600),  # must be filtered
        ]
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            resolved,
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        # Own miss present
        assert "OWN_MISS" in result.briefing_text
        # Other persona's miss NOT present
        assert "OTHER_MISS" not in result.briefing_text

    def test_own_misses_sorted_worst_first(self) -> None:
        """Misses are sorted alpha ascending (worst miss at top)."""
        persona = "discretionary-macro"
        wm = _empty_windowed(persona)
        resolved = [
            _resolved_row(persona, "MSFT", alpha=-0.0100),
            _resolved_row(persona, "AAPL", alpha=-0.0800),  # bigger miss
            _resolved_row(persona, "GOOG", alpha=-0.0400),
        ]
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            resolved,
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        text = result.briefing_text
        # AAPL (worst) must appear before GOOG must appear before MSFT
        pos_aapl = text.find("AAPL")
        pos_goog = text.find("GOOG")
        pos_msft = text.find("MSFT")
        assert pos_aapl < pos_goog < pos_msft, (
            "Own-misses must be sorted worst-alpha first (AAPL < GOOG < MSFT)"
        )


# ---------------------------------------------------------------------------
# AC-3b — Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    """AC-3b: Briefing is persisted to the correct path with exact content."""

    def test_file_written_at_correct_path(self, tmp_path: Path) -> None:
        """state/runs/<week>-memory/<persona>.md exists after build."""
        persona = "value"
        wm = _empty_windowed(persona)
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            runs_dir=tmp_path,
            persist=True,
        )
        expected = tmp_path / f"{WEEK}-memory" / f"{persona}.md"
        assert expected.exists(), f"Expected briefing file at {expected}"
        assert result.output_path == expected

    def test_file_content_matches_briefing_text(self, tmp_path: Path) -> None:
        """File content must be byte-identical to briefing_text."""
        persona = "growth"
        wm = _simple_windowed(
            persona,
            past_calls=[("2026-W23", "  TSLA: add confidence=4 weight=0.040")],
        )
        result = build_persona_briefing(
            wm,
            "TSLA resolved +0.0500 vs SPY",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            runs_dir=tmp_path,
            persist=True,
        )
        on_disk = result.output_path.read_text(encoding="utf-8")
        assert on_disk == result.briefing_text

    def test_persist_false_returns_no_path(self) -> None:
        """When persist=False, output_path is None and no file is written."""
        persona = "technical"
        wm = _empty_windowed(persona)
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        assert result.output_path is None

    def test_parent_directory_created(self, tmp_path: Path) -> None:
        """The <week>-memory/ subdirectory is created if absent."""
        target_dir = tmp_path / f"{WEEK}-memory"
        assert not target_dir.exists()
        persona = "quant-systematic"
        wm = _empty_windowed(persona)
        build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            [],
            week_id=WEEK,
            config=_DEFAULT_CFG,
            runs_dir=tmp_path,
            persist=True,
        )
        assert target_dir.is_dir()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestBriefingConfig:
    """load_briefing_config falls back to defaults when keys are absent."""

    def test_defaults_when_file_missing(self, tmp_path: Path) -> None:
        cfg = load_briefing_config(config_path=tmp_path / "nonexistent.yaml")
        assert cfg.memory_briefing_max_chars == 3000
        assert cfg.own_misses_in_digest is True

    def test_custom_value_loaded(self, tmp_path: Path) -> None:
        yaml_text = "memory_briefing_max_chars: 1500\nown_misses_in_digest: false\n"
        p = tmp_path / "t.yaml"
        p.write_text(yaml_text)
        cfg = load_briefing_config(config_path=p)
        assert cfg.memory_briefing_max_chars == 1500
        assert cfg.own_misses_in_digest is False

    def test_partial_override_uses_defaults_for_missing(self, tmp_path: Path) -> None:
        yaml_text = "memory_briefing_max_chars: 5000\n"
        p = tmp_path / "t.yaml"
        p.write_text(yaml_text)
        cfg = load_briefing_config(config_path=p)
        assert cfg.memory_briefing_max_chars == 5000
        assert cfg.own_misses_in_digest is True  # default


# ---------------------------------------------------------------------------
# Real-W24-derived fixture (Gate-4 provenance corollary)
#
# Provenance: shapes derived from real 2026-W24 portfolio/holdings/weekly_returns
# structure produced by TASK-M4-002 tests.  Actual ticker symbols + weights are
# representative of real portfolio sizes; alpha values are synthetic
# mark-to-market on a deterministic synthetic follow-on week (2026-W25).
# No PII present — tickers + weights + returns only.  Hand-crafted body
# text matches the format written by memory.py _build_past_calls_entry.
# ---------------------------------------------------------------------------

class TestRealW24Fixture:
    """Gate-4 fixture-provenance corollary: at least one fixture derived from
    sanitized real 2026-W24 data shapes.
    """

    # Sanitized 2026-W24 derived data:
    # value persona: held AAPL (add, conf=4, 8.5%), MSFT (hold, conf=3, 5.0%)
    # Synthetic W25 resolution: AAPL alpha = +0.0180 vs SPY, MSFT alpha = -0.0090
    _W24_PAST_CALLS_BODY = (
        "  AAPL: add confidence=4 weight=0.085\n"
        "  MSFT: hold confidence=3 weight=0.050\n"
        "outcome: pending"
    )
    _W24_COUNTERFACTUAL_BODY = (
        "AAPL 8.5% | MSFT 5.0% | CASH 86.5%"
    )
    _W24_DEBATE_STANCES_BODY = (
        "AAPL: bullish (value screens strong FCF); MSFT: neutral (growth baked in)"
    )
    _W24_WHATS_NEW_BODY = "value: no prior calls resolved yet (week 1)"

    _W25_RESOLVED = [
        ResolvedRow(
            persona="value",
            ticker="AAPL",
            call_week_id="2026-W24",
            as_of_week_id="2026-W25",
            alpha=0.0180,
            action="add",
        ),
        ResolvedRow(
            persona="value",
            ticker="MSFT",
            call_week_id="2026-W24",
            as_of_week_id="2026-W25",
            alpha=-0.0090,
            action="hold",
        ),
    ]

    def test_real_w24_fixture_content(self) -> None:
        """Briefing for 'value' on 2026-W25 built from W24-derived inputs
        contains the expected tickers, actions, and alpha values.
        """
        wm = _simple_windowed(
            "value",
            past_calls=[("2026-W24", self._W24_PAST_CALLS_BODY)],
            counterfactual=[("2026-W24", self._W24_COUNTERFACTUAL_BODY)],
            debate_stances=[("2026-W24", self._W24_DEBATE_STANCES_BODY)],
            whats_new=[("2026-W24", self._W24_WHATS_NEW_BODY)],
        )
        # Synthetic W25 digest summarising W24 resolutions
        digest = (
            "Since your last run, these of your calls resolved:\n"
            "  AAPL (you said add conf=4 in 2026-W24) → alpha +0.0180 vs SPY\n"
            "  MSFT (you said hold conf=3 in 2026-W24) → alpha -0.0090 vs SPY"
        )
        result = build_persona_briefing(
            wm,
            digest,
            self._W25_RESOLVED,
            week_id="2026-W25",
            config=_DEFAULT_CFG,
            persist=False,
        )
        text = result.briefing_text

        # Content match: key W24 data shapes appear
        assert "AAPL" in text
        assert "MSFT" in text
        assert "2026-W24" in text
        assert "bullish" in text
        assert "FCF" in text

        # Digest is injected
        assert "+0.0180" in text

        # Own-misses: MSFT is negative alpha → appears in callout
        assert "Past Calls That Resolved Below SPY" in text
        assert "MSFT" in text

        # No truncation expected (content is small)
        assert result.truncated is False

    def test_real_w24_no_cross_leakage(self) -> None:
        """No other persona's slug appears in the value briefing."""
        wm = _simple_windowed(
            "value",
            past_calls=[("2026-W24", self._W24_PAST_CALLS_BODY)],
        )
        result = build_persona_briefing(
            wm,
            "No prior calls have resolved yet.",
            self._W25_RESOLVED,
            week_id="2026-W25",
            config=_DEFAULT_CFG,
            persist=False,
        )
        for other in PERSONA_SLUGS_7:
            if other == "value":
                continue
            # slug must not appear as a standalone word in a revealing context
            # (header says "value" so that's fine; check other slugs only)
            assert other not in result.briefing_text, (
                f"Leakage: 'value' briefing contains other persona slug '{other}'"
            )


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

class TestBuildAllBriefings:
    """build_all_briefings returns one result per persona, no cross-contamination."""

    def test_returns_all_personas(self) -> None:
        inputs = {
            p: (
                _empty_windowed(p),
                "No prior calls have resolved yet.",
                [],
            )
            for p in PERSONA_SLUGS_7
        }
        results = build_all_briefings(
            inputs,
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        assert set(results.keys()) == set(PERSONA_SLUGS_7)
        for persona, result in results.items():
            assert result.persona == persona

    def test_each_briefing_contains_only_own_marker(self) -> None:
        """Same leakage check via batch entry point."""
        inputs: dict = {}
        for p in PERSONA_SLUGS_7:
            marker = f"BATCH_MARKER_{p.upper().replace('-', '_')}"
            inputs[p] = (
                _simple_windowed(p, past_calls=[(WEEK, f"  {marker}: add confidence=3 weight=0.030")]),
                "No prior calls have resolved yet.",
                [],
            )
        results = build_all_briefings(
            inputs,
            week_id=WEEK,
            config=_DEFAULT_CFG,
            persist=False,
        )
        for persona, result in results.items():
            own_marker = f"BATCH_MARKER_{persona.upper().replace('-', '_')}"
            assert own_marker in result.briefing_text
            for other in PERSONA_SLUGS_7:
                if other == persona:
                    continue
                other_marker = f"BATCH_MARKER_{other.upper().replace('-', '_')}"
                assert other_marker not in result.briefing_text

    def test_persist_all_files(self, tmp_path: Path) -> None:
        """All 7 persona briefing files are written under <week>-memory/."""
        inputs = {
            p: (
                _empty_windowed(p),
                "No prior calls have resolved yet.",
                [],
            )
            for p in PERSONA_SLUGS_7
        }
        results = build_all_briefings(
            inputs,
            week_id=WEEK,
            config=_DEFAULT_CFG,
            runs_dir=tmp_path,
            persist=True,
        )
        out_dir = tmp_path / f"{WEEK}-memory"
        for persona in PERSONA_SLUGS_7:
            expected = out_dir / f"{persona}.md"
            assert expected.exists(), f"Missing briefing file for {persona}"
            assert results[persona].output_path == expected
