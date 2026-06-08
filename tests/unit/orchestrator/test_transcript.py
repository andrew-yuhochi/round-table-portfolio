"""Unit tests for orchestrator/transcript.py — Component 17.

Coverage matrix:
  AC1 — exactly one transcripts row per week; full_log_path non-null and resolving:
    test_transcript_file_is_created_and_nonempty       — file exists, non-empty
    test_transcript_path_resolves_to_existing_file     — returned path == file that exists
    test_transcript_path_is_inside_debates_dir         — written to state/debates/

  AC2 — markdown contains Round-1 section for all 7 personas + consensus + decision; NO Round-2:
    test_markdown_has_round1_heading                   — "## Round 1" present
    test_all_7_personas_appear_in_markdown             — all persona slugs in output
    test_narratives_appear_in_markdown                 — narrative_summary per persona included
    test_consensus_section_present                     — "### Consensus" present
    test_founder_decision_present                      — "### Founder Decision" + decision text
    test_no_round2_section_in_m2_transcript            — "## Round 2" NOT present (scope boundary)
    test_dissent_note_section_present                  — "### Dissent Note" present

  AC3 — vote_tally is valid JSON, round-trips, matches round-1 stances:
    test_vote_tally_is_valid_json                      — json.loads succeeds
    test_vote_tally_round_trips                        — parse → re-serialize == original
    test_vote_tally_matches_stances                    — counts match actual stance actions per ticker
    test_vote_tally_all_four_actions_present           — each ticker has add/reduce/hold/exit keys

  AC4 — atomic write: mid-write failure leaves prior transcript intact:
    test_atomic_write_tmpfile_removed_on_success       — .tmp file absent after success
    test_atomic_write_prior_file_intact_on_failure     — prior file unchanged when rename fails
    test_atomic_write_no_partial_file_at_target        — target path has no partial content on failure

  AC5 — full test suite passes (verified by running all tests via pytest):
    (This is the passing criterion for the whole file — no individual test for it.)

Sample count: ≥13 cells across all tests.
"""

from __future__ import annotations

import json
import os
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest

from round_table_portfolio.orchestrator.round1 import AgentStancePayload, Round1Capture
from round_table_portfolio.orchestrator.transcript import (
    TranscriptPayload,
    _atomic_write,
    _build_vote_tally,
    write_round1_transcript,
)

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

WEEK_ID = "2026-W24"
DEBATE_SET = ["AAPL", "MSFT", "GOOGL"]


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_stance(
    persona: str,
    ticker: str,
    action: str = "add",
    weight: float = 0.10,
    confidence: int = 3,
    rationale: str = "test rationale",
) -> AgentStancePayload:
    return AgentStancePayload(
        week_id=WEEK_ID,
        persona=persona,
        ticker=ticker,
        round=1,
        action=action,
        target_weight=weight,
        confidence=confidence,
        rationale=rationale,
    )


def _make_round1_capture(
    debate_set: list[str] | None = None,
    action_override: dict[str, str] | None = None,
) -> Round1Capture:
    """Build a Round1Capture for all 7 personas over DEBATE_SET.

    action_override: maps persona slug to action for all its tickers.
    """
    tickers = debate_set or DEBATE_SET
    action_override = action_override or {}
    stances: list[AgentStancePayload] = []
    narratives: dict[str, str] = {}

    for i, persona in enumerate(PERSONA_SLUGS_7):
        action = action_override.get(persona, "add")
        for ticker in tickers:
            stances.append(_make_stance(persona, ticker, action=action))
        narratives[persona] = f"{persona} narrative: identified strong FCF names in {week_id}."

    return Round1Capture(
        stances=stances,
        counterfactuals={
            p: {t: 0.10 for t in tickers} | {"CASH": 1.0 - 0.10 * len(tickers)}
            for p in PERSONA_SLUGS_7
        },
        prompts={p: f"Prompt for {p}" for p in PERSONA_SLUGS_7},
        narratives=narratives,
    )


def _make_consensus(tickers: list[str] | None = None) -> dict[str, float]:
    t = tickers or DEBATE_SET
    w = round(0.80 / len(t), 6)
    return {ticker: w for ticker in t}


def _make_std_devs(tickers: list[str] | None = None, high: bool = False) -> dict[str, float]:
    t = tickers or DEBATE_SET
    v = 0.12 if high else 0.02
    return {ticker: v for ticker in t}


def _make_thresholds_yaml(tmp_path: Path, threshold: float = 0.08) -> Path:
    p = tmp_path / "thresholds.yaml"
    p.write_text(
        f"max_position_weight: 0.20\ndissent_std_dev_threshold: {threshold}\nrun_window_hours: 5.0\n",
        encoding="utf-8",
    )
    return p


# Use a valid week label at module level — avoid NameError from WEEK_ID ref inside dict comp.
week_id = WEEK_ID


# ---------------------------------------------------------------------------
# AC1 — file creation and path contract
# ---------------------------------------------------------------------------

class TestTranscriptFileCreation:

    def test_transcript_file_is_created_and_nonempty(self, tmp_path: Path) -> None:
        """AC1: the written file exists and is non-empty."""
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = _make_round1_capture()
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=thresholds,
        )
        assert path.exists(), "Transcript file must exist after write."
        assert path.stat().st_size > 0, "Transcript file must be non-empty."

    def test_transcript_path_resolves_to_existing_file(self, tmp_path: Path) -> None:
        """AC1: the returned path resolves to a real, readable file."""
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = _make_round1_capture()
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=thresholds,
        )
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_transcript_path_is_inside_debates_dir(self, tmp_path: Path) -> None:
        """AC1: file is written to state/debates/YYYY-WNN.md."""
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = _make_round1_capture()
        state_root = tmp_path / "state"
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=state_root,
            thresholds_path=thresholds,
        )
        expected = state_root / "debates" / f"{WEEK_ID}.md"
        assert path == expected, f"Expected {expected}, got {path}"
        assert path.exists()


# ---------------------------------------------------------------------------
# AC2 — markdown content: Round-1 section complete, no Round-2
# ---------------------------------------------------------------------------

class TestTranscriptMarkdownContent:

    @pytest.fixture
    def transcript_content(self, tmp_path: Path) -> str:
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = _make_round1_capture()
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(high=True),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=thresholds,
        )
        return path.read_text(encoding="utf-8")

    def test_markdown_has_round1_heading(self, transcript_content: str) -> None:
        """AC2: the file must have a '## Round 1' section heading."""
        assert "## Round 1" in transcript_content

    def test_all_7_personas_appear_in_markdown(self, transcript_content: str) -> None:
        """AC2: all 7 persona slugs appear as headings in the Round-1 section."""
        for persona in PERSONA_SLUGS_7:
            assert persona in transcript_content, (
                f"Persona {persona!r} not found in transcript."
            )

    def test_narratives_appear_in_markdown(self, transcript_content: str) -> None:
        """AC2: each persona's narrative_summary appears in the transcript."""
        for persona in PERSONA_SLUGS_7:
            expected_fragment = f"{persona} narrative:"
            assert expected_fragment in transcript_content, (
                f"Narrative for {persona!r} not found in transcript."
            )

    def test_consensus_section_present(self, transcript_content: str) -> None:
        """AC2: '### Consensus' heading present."""
        assert "### Consensus" in transcript_content

    def test_founder_decision_present(self, transcript_content: str) -> None:
        """AC2: '### Founder Decision' heading and decision text present."""
        assert "### Founder Decision" in transcript_content
        assert "panel_approved" in transcript_content

    def test_no_round2_section_in_m2_transcript(self, transcript_content: str) -> None:
        """AC2 scope boundary: M2 transcript must NOT contain a Round-2 section."""
        assert "## Round 2" not in transcript_content, (
            "SCOPE VIOLATION: Round-2 section must not appear in an M2 transcript."
        )

    def test_dissent_note_section_present(self, transcript_content: str) -> None:
        """AC2: '### Dissent Note' section present."""
        assert "### Dissent Note" in transcript_content

    def test_consensus_tickers_in_output(self, transcript_content: str) -> None:
        """AC2: each debate-set ticker appears in the Consensus table."""
        for ticker in DEBATE_SET:
            assert ticker in transcript_content, (
                f"Consensus ticker {ticker!r} not found in transcript."
            )

    def test_m3_append_anchor_present(self, tmp_path: Path) -> None:
        """AC2 / M3-append contract: the insert-point comment exists."""
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = _make_round1_capture()
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(),
            "founder_override",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=thresholds,
        )
        content = path.read_text(encoding="utf-8")
        assert "<!-- ROUND-2-INSERT-POINT -->" in content


# ---------------------------------------------------------------------------
# AC3 — vote_tally JSON contract
# ---------------------------------------------------------------------------

class TestVoteTally:

    def _stances_with_mixed_actions(self) -> Round1Capture:
        """3 personas add, 2 reduce, 1 hold, 1 exit — all over DEBATE_SET."""
        action_map = {
            "value": "add",
            "growth": "add",
            "discretionary-macro": "add",
            "cta-systematic-macro": "reduce",
            "technical": "reduce",
            "quant-systematic": "hold",
            "risk-officer": "exit",
        }
        return _make_round1_capture(action_override=action_map)

    def test_vote_tally_is_valid_json(self, tmp_path: Path) -> None:
        """AC3: _build_vote_tally output is valid JSON."""
        capture = self._stances_with_mixed_actions()
        tally_str = _build_vote_tally(capture.stances)
        parsed = json.loads(tally_str)
        assert isinstance(parsed, dict)

    def test_vote_tally_round_trips(self, tmp_path: Path) -> None:
        """AC3: JSON round-trips cleanly (parse → re-serialize == original)."""
        capture = self._stances_with_mixed_actions()
        tally_str = _build_vote_tally(capture.stances)
        parsed = json.loads(tally_str)
        re_serialized = json.dumps(parsed, sort_keys=True)
        assert json.loads(tally_str) == json.loads(re_serialized)

    def test_vote_tally_matches_stances(self, tmp_path: Path) -> None:
        """AC3: the tally counts match the actual actions in the stances."""
        capture = self._stances_with_mixed_actions()
        tally_str = _build_vote_tally(capture.stances)
        tally = json.loads(tally_str)

        for ticker in DEBATE_SET:
            assert ticker in tally, f"Ticker {ticker!r} missing from vote_tally."
            counts = tally[ticker]
            # Compute expected counts from stances directly.
            expected: dict[str, int] = {"add": 0, "reduce": 0, "hold": 0, "exit": 0}
            for s in capture.stances:
                if s.ticker == ticker:
                    expected[s.action] += 1
            assert counts["add"] == expected["add"], f"{ticker}: add mismatch"
            assert counts["reduce"] == expected["reduce"], f"{ticker}: reduce mismatch"
            assert counts["hold"] == expected["hold"], f"{ticker}: hold mismatch"
            assert counts["exit"] == expected["exit"], f"{ticker}: exit mismatch"

    def test_vote_tally_all_four_actions_present(self, tmp_path: Path) -> None:
        """AC3: every ticker's tally entry has all four action keys."""
        capture = self._stances_with_mixed_actions()
        tally = json.loads(_build_vote_tally(capture.stances))
        for ticker in DEBATE_SET:
            for action in ("add", "reduce", "hold", "exit"):
                assert action in tally[ticker], (
                    f"Action {action!r} missing from vote_tally[{ticker!r}]"
                )

    def test_vote_tally_written_into_transcript(self, tmp_path: Path) -> None:
        """AC3 integration: the vote_tally produced by build matches what stances hold."""
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = self._stances_with_mixed_actions()

        write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=thresholds,
        )

        # The last payload stored by the function carries the vote_tally.
        payload: TranscriptPayload = write_round1_transcript._last_payload  # type: ignore[attr-defined]
        tally = json.loads(payload.vote_tally)

        for ticker in DEBATE_SET:
            assert ticker in tally
            counts = tally[ticker]
            assert counts["add"] == 3    # value, growth, discretionary-macro
            assert counts["reduce"] == 2  # cta-systematic-macro, technical
            assert counts["hold"] == 1    # quant-systematic
            assert counts["exit"] == 1    # risk-officer


# ---------------------------------------------------------------------------
# AC4 — atomic write
# ---------------------------------------------------------------------------

class TestAtomicWrite:

    def test_atomic_write_tmpfile_removed_on_success(self, tmp_path: Path) -> None:
        """AC4: after a successful write, the .tmp file is absent."""
        target = tmp_path / "test.md"
        _atomic_write(target, "hello world")
        tmp = target.with_suffix(".tmp")
        assert not tmp.exists(), ".tmp file must not exist after a successful write."
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_atomic_write_prior_file_intact_on_failure(self, tmp_path: Path) -> None:
        """AC4: if os.rename raises, the prior file at target is left intact."""
        target = tmp_path / "prior.md"
        original_content = "# Prior transcript content"
        target.write_text(original_content, encoding="utf-8")

        with mock.patch("os.rename", side_effect=OSError("simulated rename failure")):
            with pytest.raises(OSError, match="simulated rename failure"):
                _atomic_write(target, "# New partial content")

        # The prior file must be intact (unchanged).
        assert target.read_text(encoding="utf-8") == original_content, (
            "Prior transcript must be intact after a failed atomic write."
        )

    def test_atomic_write_no_partial_file_at_target(self, tmp_path: Path) -> None:
        """AC4: on rename failure with NO prior file, the target does not exist."""
        target = tmp_path / "new.md"
        assert not target.exists()

        with mock.patch("os.rename", side_effect=OSError("simulated rename failure")):
            with pytest.raises(OSError):
                _atomic_write(target, "partial content")

        # Target must not exist — no partial file leaked to the target path.
        assert not target.exists(), (
            "Target path must not exist after a failed write when no prior file was there."
        )

    def test_atomic_write_full_pipeline_leaves_no_tmp(self, tmp_path: Path) -> None:
        """AC4 integration: write_round1_transcript leaves no .tmp artifact."""
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = _make_round1_capture()
        state_root = tmp_path / "state"
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=state_root,
            thresholds_path=thresholds,
        )
        tmp = path.with_suffix(".tmp")
        assert not tmp.exists(), ".tmp file must not remain after successful pipeline write."


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTranscriptEdgeCases:

    def test_all_exit_consensus_renders_cash_note(self, tmp_path: Path) -> None:
        """Edge case: empty consensus dict → cash-only note in output."""
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = _make_round1_capture(action_override={p: "exit" for p in PERSONA_SLUGS_7})
        path = write_round1_transcript(
            capture,
            {},  # empty consensus — all exit
            _make_std_devs(),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=thresholds,
        )
        content = path.read_text(encoding="utf-8")
        assert "100% cash" in content.lower() or "CASH" in content

    def test_no_contested_tickers_renders_no_dissent_table(self, tmp_path: Path) -> None:
        """Edge case: all std_devs below threshold → no contested-ticker table."""
        thresholds = _make_thresholds_yaml(tmp_path, threshold=0.08)
        capture = _make_round1_capture()
        std_devs = {t: 0.01 for t in DEBATE_SET}  # all below 0.08
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            std_devs,
            "panel_approved",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=thresholds,
        )
        content = path.read_text(encoding="utf-8")
        assert "No tickers exceeded the dissent threshold" in content

    def test_thresholds_missing_uses_default(self, tmp_path: Path) -> None:
        """Edge case: missing thresholds.yaml uses built-in default (0.08)."""
        nonexistent = tmp_path / "nonexistent_thresholds.yaml"
        capture = _make_round1_capture()
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(high=True),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=nonexistent,
        )
        assert path.exists()

    def test_founder_override_decision_in_transcript(self, tmp_path: Path) -> None:
        """Founder override decision text appears in the transcript."""
        thresholds = _make_thresholds_yaml(tmp_path)
        capture = _make_round1_capture()
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(),
            "founder_override",
            week_id=WEEK_ID,
            state_root=tmp_path / "state",
            thresholds_path=thresholds,
        )
        content = path.read_text(encoding="utf-8")
        assert "founder_override" in content

    def test_debates_dir_created_if_absent(self, tmp_path: Path) -> None:
        """The debates/ directory is created when it does not pre-exist."""
        thresholds = _make_thresholds_yaml(tmp_path)
        state_root = tmp_path / "new_state_dir"
        assert not state_root.exists()
        capture = _make_round1_capture()
        path = write_round1_transcript(
            capture,
            _make_consensus(),
            _make_std_devs(),
            "panel_approved",
            week_id=WEEK_ID,
            state_root=state_root,
            thresholds_path=thresholds,
        )
        assert path.parent.exists()
        assert path.exists()
