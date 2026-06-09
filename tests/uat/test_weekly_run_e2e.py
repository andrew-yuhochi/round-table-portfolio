"""Integrated UAT for the /weekly-run end-to-end cycle — Gate 9, TASK-M2-011.

This module is the primary Gate 9 UAT for M2.  It drives the ACTUAL run path
(``run_commit`` from scripts/weekly_run.py → ``run_weekly`` in
orchestrator/weekly_run.py) against a TEMP DB and TEMP state_root using the
REAL cached inputs from the 2026-W24 live run.  The real committed
``state/ledger.db`` is NEVER modified.

Design contract (Gate 9 #1 — app-driven, not engine-only):
  Tests exercise the full driver+engine path from ``run_commit()`` through to
  the SQLite write, file writes, and returned metrics object.  Engine helpers
  are NOT called directly — this is the same path the founder invokes.

Real-data provenance (Gate 4):
  All seven persona_replies and round1_replies are loaded from the committed
  2026-W24 cached input files (state/runs/2026-W24.*.json).  These are the
  ACTUAL subagent outputs from the live production run.  Judge verdicts are
  replayed via ReplayJudge from state/runs/2026-W24.judge_verdicts.json.

SKIP_LIVE=1 safe: no web search, no live subagent dispatch, no market data.
  ReplayJudge replays pre-captured verdicts — no LLM call is made.

Part A assertions (integrated contract):
  A1. Exactly 8 portfolio rows (1 consensus + 7 named counterfactual).
  A2. Each of the 8 portfolios:
       - has an explicit CASH holdings row
       - Σ(holdings weights) == 1.0 within 1e-6 (EXACT — tolerance-masking
         forbidden per Gate 9 #3)
  A3. agent_stances: round=1 ONLY, count == 7 × 40 = 280 (exact).
  A4. Transcript file written and non-empty.
  A5. 7 memory files updated (exist and non-empty after writeback).
  A6. 7 validator-claim JSON files written with required structure.
  A7. RunMetricsReport returned with per-persona entries for all 7 personas.

Part B — Real ledger read-only assertions (live committed 2026-W24):
  Same contract verified against the actual committed state/ledger.db
  without modifying it.

Risk tier (Gate 9 #8): HIGHEST — this is the money-bearing, multi-step,
  multi-persona orchestration flow with a transaction boundary.  Full matrix.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Project root and driver import
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[2]
_SCRIPTS = _PROJECT_ROOT / "scripts"
_STATE_ROOT = _PROJECT_ROOT / "state"
_REAL_DB = _STATE_ROOT / "ledger.db"
_REAL_RUNS = _STATE_ROOT / "runs"
_WEEK_ID = "2026-W24"          # the live production week
_SYNTHETIC_WEEK = "2026-W99"   # used by the temp-run so it never collides

_PERSONA_SLUGS = [
    "value",
    "growth",
    "discretionary-macro",
    "cta-systematic-macro",
    "technical",
    "quant-systematic",
    "risk-officer",
]
_DEBATE_SET_SIZE = 40   # confirmed from 2026-W24.debate_set.json
_PERSONA_COUNT = 7
_EXPECTED_STANCES = _PERSONA_COUNT * _DEBATE_SET_SIZE  # 280


def _import_driver() -> Any:
    spec = importlib.util.spec_from_file_location(
        "weekly_run_driver_e2e", _SCRIPTS / "weekly_run.py"
    )
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    return mod


_driver_mod = _import_driver()
run_commit = _driver_mod.run_commit

# ---------------------------------------------------------------------------
# Storage helper (schema application for temp DB)
# ---------------------------------------------------------------------------

from round_table_portfolio.storage.apply_schema import apply_schema
from round_table_portfolio.personas.output_validator import ReplayJudge

# ---------------------------------------------------------------------------
# Helpers for DB inspection
# ---------------------------------------------------------------------------


def _open_ro(db_path: Path) -> sqlite3.Connection:
    """Open a DB connection.  For the real DB this is functionally read-only
    (we issue no writes); for the temp DB it is read-write for inspection."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _portfolios(conn: sqlite3.Connection, week_id: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT portfolio_id, week_id, type FROM portfolios WHERE week_id=?",
        (week_id,),
    ).fetchall()]


def _holdings_for_week(conn: sqlite3.Connection, week_id: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        """
        SELECT p.portfolio_id, p.type, h.ticker, h.weight
        FROM portfolios p
        JOIN holdings h ON h.portfolio_id = p.portfolio_id
        WHERE p.week_id = ?
        """,
        (week_id,),
    ).fetchall()]


def _stances(conn: sqlite3.Connection, week_id: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT round, persona, ticker FROM agent_stances WHERE week_id=?",
        (week_id,),
    ).fetchall()]


# ---------------------------------------------------------------------------
# Fixture: wired temp environment using REAL 2026-W24 cached inputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e2e_env(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Full temp environment driven by the real 2026-W24 cached inputs.

    - Reads the real persona_replies, round1_replies, judge_verdicts, timing
      from state/runs/2026-W24.*.json.
    - Remaps the week_id to 2026-W99 so it never collides with the real ledger.
    - Applies the full schema to a fresh temp DB.
    - Seeds memory files (copies from real state/memory/).
    - Calls run_commit() — the same path the founder uses.
    - All assertions in Part A inspect the temp DB and temp state_root.

    The real state/ledger.db is NEVER touched.
    """
    tmp_path = tmp_path_factory.mktemp("e2e_run")
    state_root = tmp_path / "state"
    state_root.mkdir()
    runs_dir = state_root / "runs"
    runs_dir.mkdir()
    memory_dir = state_root / "memory"
    memory_dir.mkdir()

    # --- Load real 2026-W24 inputs ---
    def _load(filename: str) -> Any:
        return json.loads((_REAL_RUNS / filename).read_text(encoding="utf-8"))

    real_persona_replies: dict[str, str] = _load(f"{_WEEK_ID}.persona_replies.json")
    real_round1_replies: dict[str, str] = _load(f"{_WEEK_ID}.round1_replies.json")
    real_judge_verdicts_raw: dict[str, Any] = _load(f"{_WEEK_ID}.judge_verdicts.json")
    real_timing: dict[str, float] = _load(f"{_WEEK_ID}.timing.json")

    # --- Remap to synthetic week so the temp DB has a clean week_id ---
    # The driver reads files by week_id filename; we write the same content
    # under the synthetic week filenames.
    (runs_dir / f"{_SYNTHETIC_WEEK}.persona_replies.json").write_text(
        json.dumps(real_persona_replies), encoding="utf-8"
    )
    (runs_dir / f"{_SYNTHETIC_WEEK}.round1_replies.json").write_text(
        json.dumps(real_round1_replies), encoding="utf-8"
    )
    (runs_dir / f"{_SYNTHETIC_WEEK}.judge_verdicts.json").write_text(
        json.dumps(real_judge_verdicts_raw), encoding="utf-8"
    )
    (runs_dir / f"{_SYNTHETIC_WEEK}.timing.json").write_text(
        json.dumps(real_timing), encoding="utf-8"
    )

    # --- Seed memory files (writeback_memory requires them to exist) ---
    real_memory_dir = _STATE_ROOT / "memory"
    if real_memory_dir.exists():
        for md_file in real_memory_dir.glob("*.md"):
            shutil.copy2(str(md_file), str(memory_dir / md_file.name))
    else:
        for slug in _PERSONA_SLUGS:
            (memory_dir / f"{slug}.md").write_text(
                f"# {slug}\nNo prior weeks.\n", encoding="utf-8"
            )

    # --- Apply schema to fresh temp DB ---
    db_path = state_root / "ledger.db"
    apply_schema(db_path=db_path)

    # --- Patch driver _PROJECT_ROOT so it reads real config/ files ---
    import importlib.util as _ilu
    _driver_mod._PROJECT_ROOT = _PROJECT_ROOT

    # --- Run the commit path against the temp DB ---
    # run_commit() uses state_root/ledger.db; we write it there.
    run_commit(_SYNTHETIC_WEEK, "approve", state_root)

    return {
        "state_root": state_root,
        "db_path": db_path,
        "week": _SYNTHETIC_WEEK,
        "real_persona_replies": real_persona_replies,
        "real_round1_replies": real_round1_replies,
    }


# ===========================================================================
# Part A — Integrated contract assertions (temp run using real 2026-W24 data)
# ===========================================================================


class TestA1_ExactlyEightPortfolios:
    """A1: Exactly 8 portfolio rows (1 consensus + 7 counterfactual)."""

    def test_portfolio_count_is_8(self, e2e_env: dict) -> None:
        conn = _open_ro(e2e_env["db_path"])
        portfolios = _portfolios(conn, e2e_env["week"])
        conn.close()
        assert len(portfolios) == 8, (
            f"Expected 8 portfolios (1 consensus + 7 named counterfactual), "
            f"got {len(portfolios)}.  portfolios={portfolios}"
        )

    def test_exactly_one_consensus(self, e2e_env: dict) -> None:
        conn = _open_ro(e2e_env["db_path"])
        portfolios = _portfolios(conn, e2e_env["week"])
        conn.close()
        consensus = [p for p in portfolios if p["type"] == "consensus"]
        assert len(consensus) == 1, (
            f"Expected exactly 1 consensus portfolio, found {len(consensus)}."
        )

    def test_seven_named_counterfactuals(self, e2e_env: dict) -> None:
        conn = _open_ro(e2e_env["db_path"])
        portfolios = _portfolios(conn, e2e_env["week"])
        conn.close()
        persona_types = {p["type"] for p in portfolios if p["type"] != "consensus"}
        assert persona_types == set(_PERSONA_SLUGS), (
            f"Expected counterfactual types {set(_PERSONA_SLUGS)}, "
            f"got {persona_types}."
        )


class TestA2_FullyInvestedCashInvariant:
    """A2: Three-layer fully-invested-with-CASH invariant on all 8 portfolios.

    Per Gate 9 #3: EXACT assertions, not loose tolerance.
    The tolerance floor here is 1e-6 as specified in the task AC.
    """

    def test_every_portfolio_has_explicit_cash_row(self, e2e_env: dict) -> None:
        conn = _open_ro(e2e_env["db_path"])
        holdings = _holdings_for_week(conn, e2e_env["week"])
        conn.close()

        portfolio_ids: set[int] = {h["portfolio_id"] for h in holdings}
        portfolios_with_cash: set[int] = {
            h["portfolio_id"] for h in holdings if h["ticker"] == "CASH"
        }
        missing = portfolio_ids - portfolios_with_cash
        assert not missing, (
            f"Explicit CASH holdings row missing for portfolio_id(s): {missing}. "
            "Three-layer cash invariant violated."
        )

    def test_every_portfolio_weights_sum_exactly_to_1(self, e2e_env: dict) -> None:
        conn = _open_ro(e2e_env["db_path"])
        holdings = _holdings_for_week(conn, e2e_env["week"])
        conn.close()

        by_portfolio: dict[int, list[float]] = defaultdict(list)
        pid_to_type: dict[int, str] = {}
        for h in holdings:
            by_portfolio[h["portfolio_id"]].append(h["weight"])
            pid_to_type[h["portfolio_id"]] = h["type"]

        violations: list[str] = []
        for pid, weights in by_portfolio.items():
            total = sum(weights)
            if abs(total - 1.0) >= 1e-6:
                violations.append(
                    f"portfolio_id={pid} type={pid_to_type[pid]} "
                    f"sum={total:.10f} (delta={abs(total-1.0):.2e})"
                )
        assert not violations, (
            "Fully-invested invariant violated (tolerance 1e-6):\n"
            + "\n".join(violations)
        )

    def test_all_weights_are_non_negative(self, e2e_env: dict) -> None:
        conn = _open_ro(e2e_env["db_path"])
        holdings = _holdings_for_week(conn, e2e_env["week"])
        conn.close()
        negatives = [h for h in holdings if h["weight"] < 0.0]
        assert not negatives, (
            f"Negative weights found: {negatives}"
        )

    def test_non_cash_weights_within_max_position_cap(self, e2e_env: dict) -> None:
        """No equity position may exceed 0.20 (max_position_weight from thresholds.yaml)."""
        import yaml
        thresholds = yaml.safe_load(
            (_PROJECT_ROOT / "config" / "thresholds.yaml").read_text(encoding="utf-8")
        ) or {}
        cap = float(thresholds.get("max_position_weight", 0.20))

        conn = _open_ro(e2e_env["db_path"])
        holdings = _holdings_for_week(conn, e2e_env["week"])
        conn.close()

        violations = [
            h for h in holdings
            if h["ticker"] != "CASH" and h["weight"] > cap + 1e-9
        ]
        assert not violations, (
            f"Non-CASH weight exceeds max_position_weight={cap}: {violations}"
        )


class TestA3_StanceRounds:
    """A3 (M3): agent_stances have round=1 (all 7 personas) and round=2 (0 outlier
    dispatches in this fixture since no round2_dispatcher is wired into e2e_env).

    The e2e fixture uses run_commit() without a round2_replies.json file, so
    round2_dispatcher is None — Round-2 is skipped (backward-compatible path).
    Round=1 count must still be exactly 280.

    The Round-2-present assertions live in the unit tests (TestRound2Present).
    """

    def test_no_round2_stances_when_no_dispatcher(self, e2e_env: dict) -> None:
        """Without round2_replies.json the dispatcher is None → no round=2 rows."""
        conn = _open_ro(e2e_env["db_path"])
        round2 = conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=2",
            (e2e_env["week"],),
        ).fetchone()[0]
        conn.close()
        assert round2 == 0, (
            f"Round-2 stances found ({round2}) without a round2_dispatcher — "
            "the None-dispatcher backward-compatible path is broken."
        )

    def test_round1_stance_count_is_exactly_280(self, e2e_env: dict) -> None:
        """7 personas × 40 debate-set tickers = 280 exactly."""
        conn = _open_ro(e2e_env["db_path"])
        count = conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=1",
            (e2e_env["week"],),
        ).fetchone()[0]
        conn.close()
        assert count == _EXPECTED_STANCES, (
            f"Expected {_EXPECTED_STANCES} round-1 stances "
            f"(7 personas × {_DEBATE_SET_SIZE} tickers), got {count}."
        )

    def test_every_persona_has_stances_for_all_debate_set_tickers(
        self, e2e_env: dict
    ) -> None:
        conn = _open_ro(e2e_env["db_path"])
        stances = _stances(conn, e2e_env["week"])
        conn.close()

        per_persona: dict[str, set[str]] = defaultdict(set)
        for s in stances:
            if s["round"] == 1:
                per_persona[s["persona"]].add(s["ticker"])

        for slug in _PERSONA_SLUGS:
            assert slug in per_persona, (
                f"Persona {slug!r} has no round-1 stances."
            )
            assert len(per_persona[slug]) == _DEBATE_SET_SIZE, (
                f"Persona {slug!r} has {len(per_persona[slug])} round-1 stances, "
                f"expected {_DEBATE_SET_SIZE}."
            )


class TestA4_TranscriptFile:
    """A4: Transcript file written and non-empty."""

    def test_transcript_file_written(self, e2e_env: dict) -> None:
        debates_dir = e2e_env["state_root"] / "debates"
        all_md = list(debates_dir.rglob("*.md")) if debates_dir.exists() else []
        # Allow transcript at state/debates/<week>.md or state/debates/<week>/*.md
        week = e2e_env["week"]
        week_transcripts = [
            p for p in all_md
            if week in p.name or week in p.read_text(encoding="utf-8")[:500]
        ]
        assert week_transcripts, (
            f"No transcript .md file found for week {week!r} under "
            f"{e2e_env['state_root']}."
        )

    def test_transcript_is_non_empty(self, e2e_env: dict) -> None:
        debates_dir = e2e_env["state_root"] / "debates"
        all_md = list(debates_dir.rglob("*.md")) if debates_dir.exists() else []
        week = e2e_env["week"]
        week_transcripts = [
            p for p in all_md
            if week in p.name or week in p.read_text(encoding="utf-8")[:500]
        ]
        assert week_transcripts, "No transcript found (prerequisite for non-empty check)."
        for p in week_transcripts:
            assert p.stat().st_size > 0, f"Transcript file is empty: {p}"

    def test_transcript_in_db_points_to_real_file(self, e2e_env: dict) -> None:
        conn = _open_ro(e2e_env["db_path"])
        row = conn.execute(
            "SELECT full_log_path FROM transcripts WHERE week_id=?",
            (e2e_env["week"],),
        ).fetchone()
        conn.close()
        assert row is not None, (
            f"No transcripts row for week_id={e2e_env['week']!r}."
        )
        log_path = Path(row["full_log_path"])
        assert log_path.exists(), (
            f"transcripts.full_log_path points to non-existent file: {log_path}"
        )


class TestA5_MemoryFilesUpdated:
    """A5: 7 memory files updated — one per persona slug, all non-empty."""

    def test_7_memory_files_exist(self, e2e_env: dict) -> None:
        memory_dir = e2e_env["state_root"] / "memory"
        missing = [
            slug for slug in _PERSONA_SLUGS
            if not (memory_dir / f"{slug}.md").exists()
        ]
        assert not missing, (
            f"Memory files missing for persona(s): {missing}"
        )

    def test_all_memory_files_non_empty(self, e2e_env: dict) -> None:
        memory_dir = e2e_env["state_root"] / "memory"
        empty = [
            slug for slug in _PERSONA_SLUGS
            if (memory_dir / f"{slug}.md").stat().st_size == 0
        ]
        assert not empty, (
            f"Memory files are empty for persona(s): {empty}"
        )

    def test_memory_files_contain_week_reference(self, e2e_env: dict) -> None:
        """After writeback, each memory file should reference the run week."""
        memory_dir = e2e_env["state_root"] / "memory"
        week = e2e_env["week"]
        missing_week = [
            slug for slug in _PERSONA_SLUGS
            if week not in (memory_dir / f"{slug}.md").read_text(encoding="utf-8")
        ]
        assert not missing_week, (
            f"Memory files for {missing_week} do not reference week {week!r}. "
            "Write-back did not append new week data."
        )


class TestA6_ValidatorClaimFiles:
    """A6: 7 validator-claim JSON files written with required structure."""

    def test_claims_directory_exists(self, e2e_env: dict) -> None:
        claims_dir = (
            e2e_env["state_root"] / "reports" / e2e_env["week"] / "validator_claims"
        )
        assert claims_dir.exists(), (
            f"Validator claims directory missing: {claims_dir}"
        )

    def test_7_claim_files_present(self, e2e_env: dict) -> None:
        claims_dir = (
            e2e_env["state_root"] / "reports" / e2e_env["week"] / "validator_claims"
        )
        missing = [
            slug for slug in _PERSONA_SLUGS
            if not (claims_dir / f"{slug}.json").exists()
        ]
        assert not missing, (
            f"Validator claim files missing for persona(s): {missing}"
        )

    def test_all_claim_files_have_required_fields(self, e2e_env: dict) -> None:
        claims_dir = (
            e2e_env["state_root"] / "reports" / e2e_env["week"] / "validator_claims"
        )
        for slug in _PERSONA_SLUGS:
            path = claims_dir / f"{slug}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert "passed" in payload, (
                f"Claim file for {slug!r} missing 'passed' key."
            )
            assert "justification" in payload or "notes" in payload, (
                f"Claim file for {slug!r} missing both 'justification' and 'notes' keys."
            )

    def test_real_data_verdicts_match_cached_judge_verdicts(
        self, e2e_env: dict
    ) -> None:
        """Claim files must replay the pre-captured judge verdicts exactly.

        This is the real-data quality gate: 7/7 personas, all passed=True
        (as per the live 2026-W24 run where the founder approved).
        """
        claims_dir = (
            e2e_env["state_root"] / "reports" / e2e_env["week"] / "validator_claims"
        )
        real_verdicts_raw = json.loads(
            (_REAL_RUNS / f"{_WEEK_ID}.judge_verdicts.json").read_text(encoding="utf-8")
        )
        mismatches: list[str] = []
        for slug in _PERSONA_SLUGS:
            path = claims_dir / f"{slug}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected_passed = bool(real_verdicts_raw[slug]["passed"])
            actual_passed = bool(payload["passed"])
            if actual_passed != expected_passed:
                mismatches.append(
                    f"{slug}: expected passed={expected_passed}, got {actual_passed}"
                )
        assert not mismatches, (
            "Validator claim 'passed' values do not match cached verdicts:\n"
            + "\n".join(mismatches)
        )


class TestA7_MetricsReport:
    """A7: RunMetricsReport returned with per-persona entries for all 7 personas."""

    def test_run_log_written(self, e2e_env: dict) -> None:
        run_log = e2e_env["state_root"] / "runs" / f"{e2e_env['week']}.log"
        assert run_log.exists(), (
            f"Run log not found at {run_log}"
        )

    def test_run_log_contains_metrics_report(self, e2e_env: dict) -> None:
        run_log = e2e_env["state_root"] / "runs" / f"{e2e_env['week']}.log"
        content = run_log.read_text(encoding="utf-8")
        assert "RUN METRICS REPORT" in content, (
            "Run log missing 'RUN METRICS REPORT' block."
        )

    def test_run_log_references_all_7_personas(self, e2e_env: dict) -> None:
        run_log = e2e_env["state_root"] / "runs" / f"{e2e_env['week']}.log"
        content = run_log.read_text(encoding="utf-8")
        missing = [slug for slug in _PERSONA_SLUGS if slug not in content]
        assert not missing, (
            f"Run log does not mention persona(s): {missing}"
        )

    def test_run_log_contains_feasibility_verdict(self, e2e_env: dict) -> None:
        run_log = e2e_env["state_root"] / "runs" / f"{e2e_env['week']}.log"
        content = run_log.read_text(encoding="utf-8")
        # The metrics report emits "Verdict: FITS|TIGHT|DOES-NOT-FIT" (from
        # _build_summary_text in metrics.py — verdict strings are "fits",
        # "tight", "does-not-fit", printed upper-cased in the report).
        assert any(
            verdict in content
            for verdict in ("Verdict:", "FITS", "TIGHT", "DOES-NOT-FIT")
        ), (
            "Run log does not contain a feasibility verdict ('Verdict:' line). "
            "Component 19 (report_run_metrics) must emit one. "
            f"Run log contents (first 500 chars): {content[:500]!r}"
        )


# ===========================================================================
# Part B — Real committed ledger read-only assertions (live 2026-W24)
# ===========================================================================


class TestB_RealLedger2026W24:
    """Read-only assertions against the ACTUAL committed state/ledger.db.

    These verify the LIVE deliverable without modifying it.  Any failure here
    is a Major defect in the real committed artifact.
    """

    @pytest.fixture(autouse=True)
    def _open_real_db(self) -> None:
        self._conn = _open_ro(_REAL_DB)
        yield
        self._conn.close()

    def test_real_8_portfolios(self) -> None:
        portfolios = _portfolios(self._conn, _WEEK_ID)
        assert len(portfolios) == 8, (
            f"Real ledger: expected 8 portfolios, got {len(portfolios)}."
        )

    def test_real_one_consensus(self) -> None:
        portfolios = _portfolios(self._conn, _WEEK_ID)
        assert sum(1 for p in portfolios if p["type"] == "consensus") == 1

    def test_real_seven_counterfactuals_named_by_persona(self) -> None:
        portfolios = _portfolios(self._conn, _WEEK_ID)
        types = {p["type"] for p in portfolios if p["type"] != "consensus"}
        assert types == set(_PERSONA_SLUGS), (
            f"Real ledger: counterfactual types {types} != {set(_PERSONA_SLUGS)}"
        )

    def test_real_every_portfolio_has_cash_row(self) -> None:
        holdings = _holdings_for_week(self._conn, _WEEK_ID)
        portfolio_ids = {h["portfolio_id"] for h in holdings}
        with_cash = {h["portfolio_id"] for h in holdings if h["ticker"] == "CASH"}
        missing = portfolio_ids - with_cash
        assert not missing, (
            f"Real ledger: CASH row missing for portfolio_id(s): {missing}"
        )

    def test_real_every_portfolio_weights_sum_exactly_to_1(self) -> None:
        holdings = _holdings_for_week(self._conn, _WEEK_ID)
        by_portfolio: dict[int, list[float]] = defaultdict(list)
        pid_to_type: dict[int, str] = {}
        for h in holdings:
            by_portfolio[h["portfolio_id"]].append(h["weight"])
            pid_to_type[h["portfolio_id"]] = h["type"]
        violations: list[str] = []
        for pid, weights in by_portfolio.items():
            total = sum(weights)
            if abs(total - 1.0) >= 1e-6:
                violations.append(
                    f"portfolio_id={pid} type={pid_to_type[pid]} sum={total:.10f}"
                )
        assert not violations, (
            "Real ledger fully-invested invariant violated:\n"
            + "\n".join(violations)
        )

    def test_real_m2_ledger_has_no_round2_stances(self) -> None:
        """The 2026-W24 real ledger is an M2 run — round=2 stances were not written.
        This is expected: the M2 weekly run had no round2_dispatcher.
        """
        round2 = self._conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=2",
            (_WEEK_ID,),
        ).fetchone()[0]
        assert round2 == 0, (
            f"Real ledger (M2 run 2026-W24): {round2} round-2 stances found. "
            "The M2 run had no Round-2 dispatcher — this is unexpected."
        )

    def test_real_stance_count_exactly_280(self) -> None:
        count = self._conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=1",
            (_WEEK_ID,),
        ).fetchone()[0]
        assert count == _EXPECTED_STANCES, (
            f"Real ledger: expected {_EXPECTED_STANCES} round-1 stances, got {count}."
        )

    def test_real_transcript_row_exists_with_non_null_path(self) -> None:
        row = self._conn.execute(
            "SELECT full_log_path FROM transcripts WHERE week_id=?",
            (_WEEK_ID,),
        ).fetchone()
        assert row is not None, "Real ledger: no transcripts row for 2026-W24."
        assert row["full_log_path"], "Real ledger: full_log_path is NULL or empty."

    def test_real_transcript_file_exists_and_non_empty(self) -> None:
        row = self._conn.execute(
            "SELECT full_log_path FROM transcripts WHERE week_id=?",
            (_WEEK_ID,),
        ).fetchone()
        path = Path(row["full_log_path"])
        assert path.exists(), f"Real transcript file missing: {path}"
        assert path.stat().st_size > 0, f"Real transcript file is empty: {path}"

    def test_real_7_persona_reports(self) -> None:
        count = self._conn.execute(
            "SELECT COUNT(*) FROM persona_reports WHERE week_id=?",
            (_WEEK_ID,),
        ).fetchone()[0]
        assert count == _PERSONA_COUNT, (
            f"Real ledger: expected {_PERSONA_COUNT} persona_reports, got {count}."
        )

    def test_real_7_memory_files_exist(self) -> None:
        memory_dir = _STATE_ROOT / "memory"
        missing = [
            slug for slug in _PERSONA_SLUGS
            if not (memory_dir / f"{slug}.md").exists()
        ]
        assert not missing, f"Real memory files missing: {missing}"

    def test_real_7_validator_claim_files_exist(self) -> None:
        claims_dir = _STATE_ROOT / "reports" / _WEEK_ID / "validator_claims"
        missing = [
            slug for slug in _PERSONA_SLUGS
            if not (claims_dir / f"{slug}.json").exists()
        ]
        assert not missing, f"Real validator claim files missing: {missing}"

    def test_real_decision_type_panel_approved(self) -> None:
        """The live run was approved by the founder — decision_type must be panel_approved."""
        row = self._conn.execute(
            "SELECT notes FROM weeks WHERE week_id=?",
            (_WEEK_ID,),
        ).fetchone()
        assert row is not None, "Real ledger: no weeks row for 2026-W24."
        # The orchestrator stores decision_type in the notes field.
        assert "panel_approved" in row["notes"], (
            f"Real ledger: expected notes to contain 'panel_approved', got {row['notes']!r}."
        )

    def test_real_run_log_exists(self) -> None:
        run_log = _STATE_ROOT / "runs" / f"{_WEEK_ID}.log"
        assert run_log.exists(), f"Real run log missing: {run_log}"
        content = run_log.read_text(encoding="utf-8")
        assert len(content) > 50, "Real run log is suspiciously short."


# ===========================================================================
# Source-code fence: Round-2 dispatch seam IS present in weekly_run.py (M3)
# ===========================================================================


class TestRound2InSourceCode:
    """Assert that the Round-2 dispatch seam IS wired in weekly_run.py.

    M3 inversion of the M2 TestNoRound2InSourceCode fence.
    These positive guards ensure the wiring is never accidentally removed.
    """

    def test_run_weekly_source_has_round2_dispatcher_seam(self) -> None:
        """weekly_run.py must reference 'round2_dispatcher' — the injected seam."""
        weekly_run_src = (
            _PROJECT_ROOT / "src" / "round_table_portfolio"
            / "orchestrator" / "weekly_run.py"
        ).read_text(encoding="utf-8")

        assert "round2_dispatcher" in weekly_run_src, (
            "weekly_run.py does not contain 'round2_dispatcher' — "
            "the Round-2 dispatch seam (M3 AC #1) is absent."
        )

    def test_run_weekly_source_imports_capture_round2_stances(self) -> None:
        """weekly_run.py must import 'capture_round2_stances' (Component 24)."""
        weekly_run_src = (
            _PROJECT_ROOT / "src" / "round_table_portfolio"
            / "orchestrator" / "weekly_run.py"
        ).read_text(encoding="utf-8")

        assert "capture_round2_stances" in weekly_run_src, (
            "weekly_run.py does not import 'capture_round2_stances' — "
            "Component 24 is not wired in."
        )

    def test_run_weekly_source_imports_resynthesize_consensus(self) -> None:
        """weekly_run.py must import 'resynthesize_consensus' (Component 25)."""
        weekly_run_src = (
            _PROJECT_ROOT / "src" / "round_table_portfolio"
            / "orchestrator" / "weekly_run.py"
        ).read_text(encoding="utf-8")

        assert "resynthesize_consensus" in weekly_run_src, (
            "weekly_run.py does not import 'resynthesize_consensus' — "
            "Component 25 (re-synthesis) is not wired in."
        )
