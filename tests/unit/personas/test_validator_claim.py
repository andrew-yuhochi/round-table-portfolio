"""Unit tests for the validator-claim persistence layer — M1-014 / Component 20.

Tests the ``persist_validator_claim`` writer and the ``load_validator_claim``
round-trip reader in ``personas/output_validator.py``.

Coverage matrix
---------------
serialize_pass          — PASS result → JSON contains all required fields
serialize_fail          — FAIL result → passed=False persisted correctly
serialize_structural    — structural-gate result (no llm_justification)
serialize_fully_invested— fully-invested gate result (no llm_justification)
serialize_llm_judge     — judge-path result carries llm_justification
round_trip_pass         — write then load → ReportValidationResult round-trips
round_trip_fail         — same for FAIL result
mkdir_behavior          — validator_claims/ subdir created when absent
path_convention         — path is state/reports/<week_id>/validator_claims/<slug>.json
edge_empty_justification— llm_justification='' persisted as empty string, not None
edge_overwrite          — second write to same path overwrites (idempotent per run)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from round_table_portfolio.personas.output_validator import (
    STAGE_FULLY_INVESTED,
    STAGE_LLM_JUDGE,
    STAGE_STRUCTURAL,
    ReportValidationResult,
    load_validator_claim,
    persist_validator_claim,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WEEK_ID = "2026-W23"
_PERSONA = "value"


def _pass_result(stage: str = STAGE_LLM_JUDGE, justification: str = "Value thesis confirmed — FCF yield cited.") -> ReportValidationResult:
    return ReportValidationResult(
        passed=True,
        notes="On-mandate: reasoning stays within the persona's lens.",
        stage=stage,
        llm_justification=justification,
    )


def _fail_result(stage: str = STAGE_LLM_JUDGE, justification: str = "Report argues momentum, not value.") -> ReportValidationResult:
    return ReportValidationResult(
        passed=False,
        notes="OFF-MANDATE (judge): Report argues momentum, not value.",
        stage=stage,
        llm_justification=justification,
    )


# ---------------------------------------------------------------------------
# Serialization correctness
# ---------------------------------------------------------------------------

class TestSerialize:
    """Claim JSON contains all required fields for every result variant."""

    def test_serialize_pass_contains_all_fields(self, tmp_path: Path) -> None:
        result = _pass_result()
        path = persist_validator_claim(result, _WEEK_ID, _PERSONA, state_root=tmp_path)

        assert path.exists(), "Claim file was not created"
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["week_id"] == _WEEK_ID
        assert payload["persona"] == _PERSONA
        assert payload["passed"] is True
        assert "notes" in payload
        assert "stage" in payload
        assert "llm_justification" in payload

    def test_serialize_fail_passed_is_false(self, tmp_path: Path) -> None:
        result = _fail_result()
        path = persist_validator_claim(result, _WEEK_ID, _PERSONA, state_root=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["passed"] is False

    def test_serialize_structural_gate(self, tmp_path: Path) -> None:
        result = ReportValidationResult(
            passed=False,
            notes="STRUCTURAL GATE FAIL — Report too short",
            stage=STAGE_STRUCTURAL,
            llm_justification="",
        )
        path = persist_validator_claim(result, _WEEK_ID, "technical", state_root=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["stage"] == STAGE_STRUCTURAL
        assert payload["passed"] is False
        assert "STRUCTURAL" in payload["notes"]
        # No judge justification for structural failures.
        assert payload["llm_justification"] == ""

    def test_serialize_fully_invested_gate(self, tmp_path: Path) -> None:
        result = ReportValidationResult(
            passed=False,
            notes="FULLY-INVESTED GATE FAIL — sum exceeds 1.0",
            stage=STAGE_FULLY_INVESTED,
            llm_justification="",
        )
        path = persist_validator_claim(result, _WEEK_ID, "quant-systematic", state_root=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["stage"] == STAGE_FULLY_INVESTED
        assert payload["llm_justification"] == ""

    def test_serialize_llm_judge_carries_justification(self, tmp_path: Path) -> None:
        justification = "The report cites FCF yield 4.5% and P/E 18× — core value lens. PASS."
        result = _pass_result(justification=justification)
        path = persist_validator_claim(result, _WEEK_ID, "growth", state_root=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["llm_justification"] == justification

    def test_edge_empty_justification_persisted_as_empty_string(self, tmp_path: Path) -> None:
        result = ReportValidationResult(
            passed=True,
            notes="Structural gate passed.",
            stage=STAGE_STRUCTURAL,
            llm_justification="",
        )
        path = persist_validator_claim(result, _WEEK_ID, "risk-officer", state_root=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Must be an empty string, not None / absent.
        assert payload["llm_justification"] == ""
        assert payload["llm_justification"] is not None


# ---------------------------------------------------------------------------
# Round-trip (re-load)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Write then load produces an identical ReportValidationResult."""

    def test_round_trip_pass(self, tmp_path: Path) -> None:
        original = _pass_result()
        path = persist_validator_claim(original, _WEEK_ID, _PERSONA, state_root=tmp_path)
        loaded = load_validator_claim(path)

        assert loaded.passed == original.passed
        assert loaded.notes == original.notes
        assert loaded.stage == original.stage
        assert loaded.llm_justification == original.llm_justification

    def test_round_trip_fail(self, tmp_path: Path) -> None:
        original = _fail_result()
        path = persist_validator_claim(original, _WEEK_ID, _PERSONA, state_root=tmp_path)
        loaded = load_validator_claim(path)

        assert loaded.passed == original.passed
        assert loaded.notes == original.notes
        assert loaded.stage == original.stage
        assert loaded.llm_justification == original.llm_justification

    def test_round_trip_structural_no_justification(self, tmp_path: Path) -> None:
        original = ReportValidationResult(
            passed=False,
            notes="STRUCTURAL GATE FAIL — too short",
            stage=STAGE_STRUCTURAL,
            llm_justification="",
        )
        path = persist_validator_claim(original, _WEEK_ID, "discretionary-macro", state_root=tmp_path)
        loaded = load_validator_claim(path)
        assert isinstance(loaded, ReportValidationResult)
        assert loaded.passed is False
        assert loaded.llm_justification == ""

    def test_load_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_validator_claim(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Path-convention and mkdir behavior
# ---------------------------------------------------------------------------

class TestPathConvention:
    """Option-b path-convention: state/reports/<week_id>/validator_claims/<slug>.json"""

    def test_path_convention(self, tmp_path: Path) -> None:
        result = _pass_result()
        path = persist_validator_claim(result, _WEEK_ID, _PERSONA, state_root=tmp_path)
        expected = tmp_path / "reports" / _WEEK_ID / "validator_claims" / f"{_PERSONA}.json"
        assert path == expected

    def test_validator_claims_subdir_created(self, tmp_path: Path) -> None:
        claims_dir = tmp_path / "reports" / _WEEK_ID / "validator_claims"
        assert not claims_dir.exists(), "Pre-condition: subdir should not exist"
        persist_validator_claim(_pass_result(), _WEEK_ID, _PERSONA, state_root=tmp_path)
        assert claims_dir.is_dir(), "validator_claims/ subdir was not created"

    def test_overwrite_is_idempotent(self, tmp_path: Path) -> None:
        """Two writes to the same path — second one overwrites (no error)."""
        persist_validator_claim(_pass_result(), _WEEK_ID, _PERSONA, state_root=tmp_path)
        result2 = _fail_result()
        path = persist_validator_claim(result2, _WEEK_ID, _PERSONA, state_root=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Second write (FAIL) should overwrite the first (PASS).
        assert payload["passed"] is False

    def test_returns_path_object(self, tmp_path: Path) -> None:
        path = persist_validator_claim(_pass_result(), _WEEK_ID, _PERSONA, state_root=tmp_path)
        assert isinstance(path, Path)
        assert path.suffix == ".json"
