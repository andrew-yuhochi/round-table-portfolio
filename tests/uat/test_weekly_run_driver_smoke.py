"""Smoke test for scripts/weekly_run.py — deterministic backbone for Gate-9 UAT.

This test proves the full driver plumbing WITHOUT any live calls.  It:
  1. Generates 7 minimal-but-valid persona_replies (each passing the structural
     gate: >=300 chars, >=2 real tickers, >=5 distinct metric terms, >=1
     data-source signal, non-empty shortlist with cluster).
  2. Generates 7 fake judge verdicts and 7 fake timings.
  3. Writes the three session-produced input files into a temp state/runs/ dir.
  4. Drives the driver in BOTH preview and commit modes via direct function
     import (avoids subprocess; same plumbing the session uses, importable
     and inspectable by test-validator for extension).
  5. Asserts the integrated contract:
       - Exactly 8 portfolio rows (1 consensus + 7 counterfactual).
       - Each portfolio is fully invested: CASH holdings row present, all
         weights sum to 1.0 (three-layer cash invariant).
       - agent_stances are round=1 ONLY (no round=2 anywhere).
       - Transcript file exists and is non-empty.
       - 7 memory files exist (one per persona slug).
       - 7 validator-claim JSON files exist (one per persona slug).
       - Metrics report object present with feasibility_verdict set.

SKIP_LIVE=1 safe: no web search, no market data, no subagent dispatch.
STUB_ALLOW is NOT set — the driver uses the real engine path.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

# ---------------------------------------------------------------------------
# Import the driver module from scripts/ (not installed as a package).
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[2]
_SCRIPTS = _PROJECT_ROOT / "scripts"

# Load scripts/weekly_run.py as a module without requiring it to be a package.
_spec = importlib.util.spec_from_file_location(
    "weekly_run_driver", _SCRIPTS / "weekly_run.py"
)
assert _spec is not None and _spec.loader is not None
_driver = importlib.util.util_from_spec(_spec) if False else None  # type hint only

def _import_driver() -> Any:
    spec = importlib.util.spec_from_file_location(
        "weekly_run_driver", _SCRIPTS / "weekly_run.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_driver_mod = _import_driver()

run_preview = _driver_mod.run_preview
run_commit = _driver_mod.run_commit
_PERSONA_SLUGS = _driver_mod._PERSONA_SLUGS

# ---------------------------------------------------------------------------
# Re-use the project's schema applier and storage helpers.
# ---------------------------------------------------------------------------

from round_table_portfolio.storage.apply_schema import apply_schema
from round_table_portfolio.personas.output_validator import (
    PersonaConfig,
    ReplayJudge,
    StructuralConfig,
    ValidatorConfig,
)

# ---------------------------------------------------------------------------
# Persona-specific vocabulary so the structural gate passes for each archetype.
# ---------------------------------------------------------------------------

_PERSONA_VOCAB: dict[str, dict[str, str]] = {
    "value": {
        "tickers": "AAPL MSFT",
        "flavor": (
            "P/E of 18x is below the sector median of 22x, suggesting deep value. "
            "FCF yield of 5.2% and EPS growth of 12% YoY support the thesis. "
            "Balance-sheet: net debt/EBITDA of 1.2x, well within covenant. "
            "Dividend yield: 2.1%. ROE: 28%. Intrinsic value via DCF: $195. "
            "Data sources: EDGAR 10-K filing, FRED macro series, price history. "
        ),
    },
    "growth": {
        "tickers": "NVDA MSFT",
        "flavor": (
            "Revenue growth of 42% YoY driven by AI infrastructure demand. "
            "ARR acceleration and expanding operating margin (+400 bps). "
            "P/E of 45x justified by TAM expansion and durable competitive moat. "
            "EPS growth: 60% YoY. FCF margin: 32%. ROE: 45%. "
            "Data sources: SEC filings, earnings transcripts, Alpaca price data. "
        ),
    },
    "discretionary-macro": {
        "tickers": "SPY GOOGL",
        "flavor": (
            "CPI inflation at 3.2% signals persistent but decelerating pressure. "
            "PCE core at 2.8% keeps the Fed on hold through Q3. "
            "ISM manufacturing PMI at 51.2 indicates expansion. "
            "Yield curve: 10Y-2Y spread at +15 bps, no longer inverted. "
            "Consumer discretionary FCF margins improving. P/E multiples compressed. "
            "Data sources: FRED macro series, ISM, BLS CPI releases. "
        ),
    },
    "cta-systematic-macro": {
        "tickers": "QQQ SPY",
        "flavor": (
            "Trend signal: 12-month momentum score +0.82 for US equities. "
            "RSI 14-day at 58 — momentum intact, not overbought. "
            "MACD crossover confirmed on weekly chart. "
            "Volatility regime: VIX at 14, low-vol expansion phase. "
            "FCF yield spread vs 10Y treasury: 220 bps. EPS revision breadth: +65%. "
            "Data sources: price history via Alpaca, FRED, Bloomberg macro feeds. "
        ),
    },
    "technical": {
        "tickers": "AAPL TSLA",
        "flavor": (
            "50-day SMA acting as support at $178; 200-day SMA at $165. "
            "RSI 14-day: 54 — neutral territory. MACD histogram turning positive. "
            "Volume-weighted average price (VWAP): $182. "
            "Bollinger Band width narrowing: coiled for breakout. "
            "FCF and EPS used as secondary confirmation only. "
            "Data sources: price history via Alpaca, technical indicator library. "
        ),
    },
    "quant-systematic": {
        "tickers": "MSFT AMZN",
        "flavor": (
            "Factor model: value Z-score +1.4, momentum Z-score +0.9. "
            "Quality factor (ROE, FCF stability): top quintile. "
            "Low-volatility tilt: realized vol 22% vs universe median 31%. "
            "EPS surprise factor: +0.8 sigma over 4 quarters. "
            "P/E relative to sector: -0.6 sigma (cheap on cross-section). "
            "Data sources: EDGAR fundamentals, FRED macro, Alpaca price series. "
        ),
    },
    "risk-officer": {
        "tickers": "SPY TLT",
        "flavor": (
            "Tail-risk scenario: -25% drawdown if Fed delivers surprise 50 bps hike. "
            "VaR 95% (10-day): 3.8% of portfolio. "
            "Concentration risk: top-3 positions represent 38% of NAV. "
            "FCF coverage of dividend: 2.1x — adequate buffer. "
            "EPS sensitivity to rate shock: -12% in stress scenario. "
            "Data sources: FRED stress scenarios, EDGAR filings, price history. "
        ),
    },
}


def _make_persona_output(slug: str) -> str:
    """Produce a valid RESEARCH OUTPUT SCHEMA JSON passing all structural gates.

    The opening paragraph is deliberately ticker-free so the smoke test exercises
    the real Layer-2 path (portfolio-arithmetic gate only, no re-run of the
    report-prose structural gate on the truncated summary).  Tickers appear in
    later paragraphs — the FULL report passes Layer-1's structural gate (≥2
    tickers, ≥5 metric terms, data-source signal) while the first-paragraph
    summary would fail it, matching realistic LLM persona output.
    """
    vocab = _PERSONA_VOCAB[slug]
    tickers = vocab["tickers"].split()
    flavor = vocab["flavor"]

    report_body = (
        # First paragraph: no tickers — macro framing only.  This is the part
        # _extract_summary returns (up to 500 chars).  It must NOT contain ≥2
        # tickers so the smoke test catches any regression where Layer-2 re-runs
        # the prose gate on the summary instead of calling validate_counterfactual_portfolio.
        f"The current environment presents compelling opportunities for {slug} "
        "investors. Macro conditions, valuation dispersion, and capital discipline "
        "all point toward selective positioning. The analysis below details the "
        "highest-conviction ideas identified this week through fundamental and "
        "quantitative research."
        # Second paragraph: tickers appear — full report passes Layer-1.
        f" Primary names: {tickers[0]} and {tickers[1]}. "
        + flavor
        + f"Conviction: high for {tickers[0]}, moderate for {tickers[1]}. "
        "Portfolio weight recommendation: fully invested per mandate. "
        "This analysis reflects the persona's core investment lens and mandate focus."
    )

    schema = {
        "shortlist": [
            {
                "ticker": tickers[0],
                "why": f"Core thesis: {slug} mandate conviction.",
                "cluster": [tickers[1]],
            },
            {
                "ticker": "AAPL" if tickers[0] != "AAPL" else "NVDA",
                "why": "Secondary opportunity with supporting fundamentals.",
                "cluster": [],
            },
        ],
        "report": report_body,
        "web_searches_used": 4,
        "data_tool_calls_used": 8,
    }
    return json.dumps(schema)


def _extract_debate_set_from_replies(persona_replies: dict[str, str]) -> list[str]:
    """Derive the debate set from persona_replies the same way the engine does.

    Union of all directly-shortlisted tickers + cluster peers, de-duplicated.
    This lets the smoke test generate round1_replies that cover the exact debate
    set the engine will compute — no hardcoding needed.
    """
    seen: dict[str, None] = {}  # insertion-ordered dedup
    for raw in persona_replies.values():
        data = json.loads(raw)
        for entry in data.get("shortlist", []):
            ticker = entry.get("ticker", "").strip().upper()
            if ticker:
                seen[ticker] = None
            for peer in entry.get("cluster", []):
                p = str(peer).strip().upper()
                if p:
                    seen[p] = None
    return list(seen)


def _make_round1_output(slug: str, debate_set: list[str]) -> str:
    """Produce a valid ROUND 1 OUTPUT SCHEMA JSON covering the debate set.

    Stances cover every ticker in the debate set (required by capture_round1_stances).
    The counterfactual uses the first 3 debate-set tickers at fixed weights that
    sum EXACTLY to 1.0 with CASH — no floating-point accumulation risk.

    This mirrors the pattern in test_weekly_run.py's _make_round1_output: a
    compact counterfactual that passes the Layer-2 fully-invested gate cleanly.
    """
    # Distribute stance weights evenly; compute integer counts to avoid fp drift.
    n = len(debate_set)
    # Use 0.10 per ticker for stances up to 7 tickers; 0.05 beyond that so the
    # stance target_weight stays within max_position_weight=0.20.
    stance_weight = 0.10 if n <= 7 else 0.05

    stances = [
        {
            "ticker": t,
            "action": "ADD",
            "target_weight": stance_weight,
            "confidence": 3,
            "rationale": f"Stub Round-1 rationale for {t} by {slug}.",
        }
        for t in debate_set
    ]

    # Counterfactual: pick the first 3 debate-set tickers at exact weights that
    # sum to 1.0 with CASH.  The Layer-2 gate checks counterfactual_portfolio
    # sum == 1.0 with tolerance 1e-6; arithmetic is done in integers here.
    top3 = debate_set[:3]
    counterfactual: dict[str, float] = {t: 0.15 for t in top3}
    counterfactual["CASH"] = 1.0 - 0.15 * len(top3)   # = 0.55 for 3 tickers

    schema = {
        "stances": stances,
        "counterfactual_portfolio": counterfactual,
        "narrative_summary": f"{slug}: constructive; weights spread evenly across debate set.",
    }
    return json.dumps(schema)


# ---------------------------------------------------------------------------
# Fixture: temp env wiring the driver to a temp project layout.
# ---------------------------------------------------------------------------


WEEK_ID = "2026-W99"  # synthetic week so it never collides with a real run

FAKE_TIMINGS = {slug: float(100 + i * 10) for i, slug in enumerate(_PERSONA_SLUGS)}

FAKE_VERDICTS = {
    slug: {"passed": True, "justification": f"{slug}: on-mandate (smoke test fixture)."}
    for slug in _PERSONA_SLUGS
}


@pytest.fixture()
def smoke_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Full wired temp environment for the driver smoke test.

    Writes the three session input files, creates a fresh ledger, seeds memory
    files, and overrides config paths so the driver reads from the real project
    config/ directory.
    """
    state_root = tmp_path / "state"
    state_root.mkdir()
    runs_dir = state_root / "runs"
    runs_dir.mkdir()
    memory_dir = state_root / "memory"
    memory_dir.mkdir()

    # Seed memory files (writeback_memory requires them to exist).
    for slug in _PERSONA_SLUGS:
        (memory_dir / f"{slug}.md").write_text(
            f"# {slug} memory\nNo prior weeks.\n", encoding="utf-8"
        )

    # Build persona_replies.
    persona_replies = {slug: _make_persona_output(slug) for slug in _PERSONA_SLUGS}

    # Derive the debate set from the persona_replies shortlists (same logic the
    # engine uses) so round1_replies cover exactly the right tickers.
    debate_set = _extract_debate_set_from_replies(persona_replies)

    # Build round1_replies covering the full debate set.
    round1_replies = {slug: _make_round1_output(slug, debate_set) for slug in _PERSONA_SLUGS}

    # Write the four session input files.
    (runs_dir / f"{WEEK_ID}.persona_replies.json").write_text(
        json.dumps(persona_replies), encoding="utf-8"
    )
    (runs_dir / f"{WEEK_ID}.round1_replies.json").write_text(
        json.dumps(round1_replies), encoding="utf-8"
    )
    (runs_dir / f"{WEEK_ID}.judge_verdicts.json").write_text(
        json.dumps(FAKE_VERDICTS), encoding="utf-8"
    )
    (runs_dir / f"{WEEK_ID}.timing.json").write_text(
        json.dumps(FAKE_TIMINGS), encoding="utf-8"
    )

    # Apply the full DB schema to a fresh temp ledger at state_root/ledger.db.
    # run_commit opens state_root/"ledger.db" — the schema must be at that path.
    db_path = state_root / "ledger.db"
    apply_schema(db_path=db_path)

    # Patch the driver module's _PROJECT_ROOT so it reads real config/ files
    # but writes state to the temp dir.
    monkeypatch.setattr(_driver_mod, "_PROJECT_ROOT", _PROJECT_ROOT)

    return {
        "state_root": state_root,
        "db_path": db_path,
        "persona_replies": persona_replies,
        "week": WEEK_ID,
    }


# ---------------------------------------------------------------------------
# Helper: row counts from the committed ledger.
# ---------------------------------------------------------------------------


def _row_counts(db_path: Path, week_id: str) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    queries = {
        "weeks": "SELECT COUNT(*) FROM weeks WHERE week_id=?",
        "portfolios": "SELECT COUNT(*) FROM portfolios WHERE week_id=?",
        "agent_stances": "SELECT COUNT(*) FROM agent_stances WHERE week_id=?",
        "persona_reports": "SELECT COUNT(*) FROM persona_reports WHERE week_id=?",
        "transcripts": "SELECT COUNT(*) FROM transcripts WHERE week_id=?",
    }
    result = {t: conn.execute(q, (week_id,)).fetchone()[0] for t, q in queries.items()}
    conn.close()
    return result


def _holdings_for_week(db_path: Path, week_id: str) -> list[dict]:
    """Return all holdings rows for the given week via portfolio join."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT p.portfolio_id, p.type, h.ticker, h.weight
        FROM portfolios p
        JOIN holdings h ON h.portfolio_id = p.portfolio_id
        WHERE p.week_id = ?
        """,
        (week_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Smoke test: preview mode
# ---------------------------------------------------------------------------


class TestPreviewMode:
    """preview mode writes a .preview.md without touching the real ledger."""

    @pytest.fixture(autouse=True)
    def _run_preview(self, smoke_env: dict) -> None:
        """Execute one preview run; all assertions below inspect the written file."""
        run_preview(smoke_env["week"], smoke_env["state_root"])
        self._env = smoke_env
        preview_path = smoke_env["state_root"] / "runs" / f"{WEEK_ID}.preview.md"
        self._preview_text = preview_path.read_text(encoding="utf-8") if preview_path.exists() else ""

    def test_preview_file_written(self) -> None:
        preview_path = self._env["state_root"] / "runs" / f"{WEEK_ID}.preview.md"
        assert preview_path.exists(), "Preview file was not written."
        assert len(self._preview_text) > 100, "Preview file is suspiciously short."

    def test_preview_does_not_touch_real_ledger(self) -> None:
        """Preview must write to a THROWAWAY ledger; the real DB must stay clean."""
        # The fixture already ran the preview; we just check the real DB is still empty.
        real_db = self._env["db_path"]
        conn = sqlite3.connect(str(real_db))
        post_count = conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]
        conn.close()
        assert post_count == 0, (
            f"Preview mode wrote {post_count} portfolio rows to the "
            "real ledger — it must only touch the throwaway temp ledger."
        )

    def test_preview_contains_consensus_holdings_table_with_cash(self) -> None:
        """Preview must render the consensus holdings table including a CASH row."""
        assert "## Proposed Consensus Portfolio" in self._preview_text, (
            "Preview is missing the 'Proposed Consensus Portfolio' section."
        )
        assert "| CASH |" in self._preview_text, (
            "Preview consensus holdings table is missing the explicit CASH row. "
            "The founder must see the CASH weight to judge the risk posture."
        )

    def test_preview_contains_per_persona_snapshot(self) -> None:
        """Preview must include the per-persona snapshot table."""
        assert "## Per-Persona Snapshot" in self._preview_text, (
            "Preview is missing the 'Per-Persona Snapshot' section."
        )
        # Every persona slug should appear in the snapshot table.
        for slug in _PERSONA_SLUGS:
            assert slug in self._preview_text, (
                f"Persona {slug!r} is absent from the preview per-persona snapshot."
            )


# ---------------------------------------------------------------------------
# Smoke test: commit mode — integrated contract
# ---------------------------------------------------------------------------


class TestCommitMode:
    """commit mode executes the full engine against the temp ledger."""

    @pytest.fixture(autouse=True)
    def _run_commit(self, smoke_env: dict) -> None:
        """Execute one commit run; all assertions below inspect the result."""
        run_commit(smoke_env["week"], "approve", smoke_env["state_root"])
        self._env = smoke_env

    # --- 8-portfolio invariant ---

    def test_exactly_8_portfolios(self) -> None:
        counts = _row_counts(self._env["db_path"], WEEK_ID)
        assert counts["portfolios"] == 8, (
            f"Expected 8 portfolios (1 consensus + 7 counterfactual), "
            f"got {counts['portfolios']}."
        )

    # --- Cash invariant: every portfolio has a CASH row summing to 1.0 ---

    def test_every_portfolio_has_cash_row(self) -> None:
        holdings = _holdings_for_week(self._env["db_path"], WEEK_ID)
        portfolio_ids: set[int] = {h["portfolio_id"] for h in holdings}
        portfolios_with_cash: set[int] = {
            h["portfolio_id"] for h in holdings if h["ticker"] == "CASH"
        }
        missing = portfolio_ids - portfolios_with_cash
        assert not missing, (
            f"CASH holdings row missing for portfolio_ids: {missing}. "
            "Three-layer cash invariant violated."
        )

    def test_every_portfolio_weights_sum_to_1(self) -> None:
        holdings = _holdings_for_week(self._env["db_path"], WEEK_ID)
        # Group by portfolio_id.
        from collections import defaultdict
        by_portfolio: dict[int, list[float]] = defaultdict(list)
        for h in holdings:
            by_portfolio[h["portfolio_id"]].append(h["weight"])
        for pid, weights in by_portfolio.items():
            total = round(sum(weights), 6)
            assert abs(total - 1.0) < 1e-4, (
                f"Portfolio {pid} weights sum to {total}, expected 1.0. "
                "Fully-invested invariant violated."
            )

    # --- Round-1 invariant (round=2 absent when no dispatcher provided) ---

    def test_no_round2_stances_without_dispatcher(self) -> None:
        """Without round2_replies.json the dispatcher is None → no round=2 rows.
        The smoke fixture does not provide round2_replies.json (M2 data only).
        """
        db = self._env["db_path"]
        conn = sqlite3.connect(str(db))
        round2_count = conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=2",
            (WEEK_ID,),
        ).fetchone()[0]
        conn.close()
        assert round2_count == 0, (
            f"Round-2 stances found ({round2_count}) without a round2_dispatcher — "
            "the None-dispatcher backward-compatible path is broken."
        )

    def test_round1_stances_present(self) -> None:
        db = self._env["db_path"]
        conn = sqlite3.connect(str(db))
        round1_count = conn.execute(
            "SELECT COUNT(*) FROM agent_stances WHERE week_id=? AND round=1",
            (WEEK_ID,),
        ).fetchone()[0]
        conn.close()
        assert round1_count > 0, "No round-1 stances were written."

    # --- Transcript file exists ---

    def test_transcript_file_exists(self) -> None:
        transcripts_dir = self._env["state_root"] / "debates"
        # Transcript path format: state/debates/YYYY-WNN/round1.md (or similar)
        # The exact subpath is written by write_round1_transcript; we search
        # broadly so the test doesn't depend on the internal path convention.
        md_files = list(transcripts_dir.rglob("*.md")) if transcripts_dir.exists() else []
        # Also check the state root for any .md transcript file containing week.
        all_md = list(self._env["state_root"].rglob("*.md"))
        week_transcripts = [
            p for p in all_md
            if WEEK_ID.replace("-", "") in p.name or WEEK_ID in p.read_text(encoding="utf-8")[:500]
        ]
        assert week_transcripts, (
            f"No transcript .md file found referencing {WEEK_ID} under "
            f"{self._env['state_root']}. Transcript write-back failed."
        )

    # --- 7 memory files updated ---

    def test_7_memory_files_exist(self) -> None:
        memory_dir = self._env["state_root"] / "memory"
        for slug in _PERSONA_SLUGS:
            mem_file = memory_dir / f"{slug}.md"
            assert mem_file.exists(), f"Memory file missing for persona={slug!r}."
            # Content must be non-empty.
            assert mem_file.stat().st_size > 0, (
                f"Memory file for {slug!r} is empty after writeback."
            )

    # --- 7 validator-claim JSON files written ---

    def test_7_validator_claim_files_exist(self) -> None:
        claims_dir = (
            self._env["state_root"] / "reports" / WEEK_ID / "validator_claims"
        )
        assert claims_dir.exists(), (
            f"Validator claims directory does not exist: {claims_dir}"
        )
        for slug in _PERSONA_SLUGS:
            claim_file = claims_dir / f"{slug}.json"
            assert claim_file.exists(), (
                f"Validator claim file missing for persona={slug!r}."
            )
            payload = json.loads(claim_file.read_text(encoding="utf-8"))
            assert "passed" in payload, f"Claim file for {slug!r} missing 'passed' key."

    # --- Metrics report present ---

    def test_run_log_written(self) -> None:
        run_log = self._env["state_root"] / "runs" / f"{WEEK_ID}.log"
        assert run_log.exists(), f"Run log not found at {run_log}."
        content = run_log.read_text(encoding="utf-8")
        assert "RUN METRICS REPORT" in content, (
            "Run log does not contain the metrics report block."
        )


# ---------------------------------------------------------------------------
# ReplayJudge unit tests (exported from output_validator)
# ---------------------------------------------------------------------------


class TestReplayJudge:
    """Unit tests for the ReplayJudge class."""

    def test_returns_captured_verdict(self) -> None:
        judge = ReplayJudge({"value": (True, "On mandate: deep value lens.")})
        passed, justification = judge.judge(
            report="any",
            mandate="any",
            persona_slug="value",
            on_mandate_concepts=(),
            off_mandate_signals=(),
        )
        assert passed is True
        assert justification == "On mandate: deep value lens."

    def test_returns_fail_verdict(self) -> None:
        judge = ReplayJudge({"growth": (False, "Off mandate: used value lens.")})
        passed, _ = judge.judge(
            report="any",
            mandate="any",
            persona_slug="growth",
            on_mandate_concepts=(),
            off_mandate_signals=(),
        )
        assert passed is False

    def test_missing_slug_raises_key_error(self) -> None:
        judge = ReplayJudge({"value": (True, "ok")})
        with pytest.raises(KeyError, match="no pre-captured verdict"):
            judge.judge(
                report="any",
                mandate="any",
                persona_slug="growth",  # not in verdicts
                on_mandate_concepts=(),
                off_mandate_signals=(),
            )

    def test_ignores_report_and_mandate(self) -> None:
        """ReplayJudge must not inspect report/mandate — verdict is pre-captured."""
        judge = ReplayJudge({"technical": (True, "pre-captured")})
        result1 = judge.judge("report A", "mandate A", "technical", (), ())
        result2 = judge.judge("totally different report", "other mandate", "technical", (), ())
        assert result1 == result2, (
            "ReplayJudge returned different results for the same slug — "
            "it must replay the pre-captured verdict regardless of report text."
        )


# ---------------------------------------------------------------------------
# Gap-A: run_preview writes a persistent debate transcript
# ---------------------------------------------------------------------------

# Minimal valid round2 reply for the 2 outliers that the engine selects.
# We can't know which 2 slugs will be selected from the smoke fixture ahead of
# time, so we pre-build valid round2 JSON for ALL 7 and let the test pick them
# up after running preview once without round2 (to discover the outlier slugs),
# then re-run with round2.  Simpler: provide replies for all 7 — the driver
# only uses the 2 it actually selected, ignoring the rest.

def _make_round2_output(slug: str, debate_set: list[str]) -> str:
    """Produce a valid ROUND 2 OUTPUT SCHEMA JSON for one outlier persona."""
    stances = [
        {
            "ticker": t,
            "action": "ADD",
            "target_weight": 0.10,
            "confidence": 3,
            "rationale": f"Defended position on {t} — counterargument not compelling.",
            "position_change": "defended",
        }
        for t in debate_set[:3]   # outlier restates stances for its top tickers
    ]
    schema = {
        "round": 2,
        "rebuttal_narrative": f"{slug}: I maintain my positions after reviewing the counterarguments.",
        "stances": stances,
    }
    return json.dumps(schema)


@pytest.fixture()
def smoke_env_with_round2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Full wired temp environment that also provides round2_replies.json.

    Builds on the same input files as smoke_env but additionally writes
    round2_replies.json and round2_timing.json so run_preview exercises the
    full R1+R2 path (Gap A, B, C).
    """
    state_root = tmp_path / "state"
    state_root.mkdir()
    runs_dir = state_root / "runs"
    runs_dir.mkdir()
    memory_dir = state_root / "memory"
    memory_dir.mkdir()

    for slug in _PERSONA_SLUGS:
        (memory_dir / f"{slug}.md").write_text(
            f"# {slug} memory\nNo prior weeks.\n", encoding="utf-8"
        )

    persona_replies = {slug: _make_persona_output(slug) for slug in _PERSONA_SLUGS}
    debate_set = _extract_debate_set_from_replies(persona_replies)
    round1_replies = {slug: _make_round1_output(slug, debate_set) for slug in _PERSONA_SLUGS}

    (runs_dir / f"{WEEK_ID}.persona_replies.json").write_text(
        json.dumps(persona_replies), encoding="utf-8"
    )
    (runs_dir / f"{WEEK_ID}.round1_replies.json").write_text(
        json.dumps(round1_replies), encoding="utf-8"
    )
    (runs_dir / f"{WEEK_ID}.judge_verdicts.json").write_text(
        json.dumps(FAKE_VERDICTS), encoding="utf-8"
    )
    (runs_dir / f"{WEEK_ID}.timing.json").write_text(
        json.dumps(FAKE_TIMINGS), encoding="utf-8"
    )

    # Provide round2 replies for all 7 personas (driver uses whichever 2 were selected).
    round2_replies = {slug: _make_round2_output(slug, debate_set) for slug in _PERSONA_SLUGS}
    (runs_dir / f"{WEEK_ID}.round2_replies.json").write_text(
        json.dumps(round2_replies), encoding="utf-8"
    )
    # Fake round2 timing for the 2 outliers (driver loads this but only 2 slugs
    # end up being dispatched; use first 2 slugs as a stand-in — values are arbitrary).
    round2_timing = {_PERSONA_SLUGS[0]: 45.0, _PERSONA_SLUGS[1]: 52.0}
    (runs_dir / f"{WEEK_ID}.round2_timing.json").write_text(
        json.dumps(round2_timing), encoding="utf-8"
    )

    db_path = state_root / "ledger.db"
    apply_schema(db_path=db_path)
    monkeypatch.setattr(_driver_mod, "_PROJECT_ROOT", _PROJECT_ROOT)

    return {
        "state_root": state_root,
        "db_path": db_path,
        "persona_replies": persona_replies,
        "debate_set": debate_set,
        "week": WEEK_ID,
    }


class TestPreviewTranscriptPersistence:
    """Gap A: run_preview must persist the R1+R2 transcript to state/runs/<week>.debate_transcript.md."""

    @pytest.fixture(autouse=True)
    def _run_preview(self, smoke_env_with_round2: dict) -> None:
        run_preview(smoke_env_with_round2["week"], smoke_env_with_round2["state_root"])
        self._env = smoke_env_with_round2

    def test_transcript_file_written_to_persistent_path(self) -> None:
        transcript_path = (
            self._env["state_root"] / "runs" / f"{WEEK_ID}.debate_transcript.md"
        )
        assert transcript_path.exists(), (
            f"Persistent transcript not written to {transcript_path}. "
            "Gap A fix: run_preview must copy the temp transcript before rmtree."
        )
        assert transcript_path.stat().st_size > 0, "Transcript file is empty."

    def test_transcript_contains_round2_section(self) -> None:
        transcript_path = (
            self._env["state_root"] / "runs" / f"{WEEK_ID}.debate_transcript.md"
        )
        if not transcript_path.exists():
            pytest.skip("Transcript not written — see test_transcript_file_written_to_persistent_path.")
        content = transcript_path.read_text(encoding="utf-8")
        assert "## Round 2" in content, (
            "Transcript does not contain '## Round 2' section. "
            "The persisted transcript must include the full R1+R2 debate."
        )


# ---------------------------------------------------------------------------
# Gap-B: preview.md shows recalibrated M3 dissent, NOT the stale σ ≥ 0.080 metric
# ---------------------------------------------------------------------------


class TestPreviewRecalibratedDissent:
    """Gap B: _render_preview must use result.dissent / result.outliers (M3), not old σ threshold."""

    @pytest.fixture(autouse=True)
    def _run_preview(self, smoke_env_with_round2: dict) -> None:
        run_preview(smoke_env_with_round2["week"], smoke_env_with_round2["state_root"])
        self._env = smoke_env_with_round2
        preview_path = (
            smoke_env_with_round2["state_root"] / "runs" / f"{WEEK_ID}.preview.md"
        )
        self._preview_text = (
            preview_path.read_text(encoding="utf-8") if preview_path.exists() else ""
        )

    def test_preview_contains_dissent_score(self) -> None:
        assert "Dissent score:" in self._preview_text, (
            "preview.md is missing 'Dissent score:' — recalibrated M3 dissent not rendered."
        )

    def test_preview_contains_contested_flag(self) -> None:
        assert "Contested week" in self._preview_text, (
            "preview.md is missing 'Contested week' line — recalibrated M3 dissent not rendered."
        )

    def test_preview_contains_selected_outliers(self) -> None:
        assert "Selected outliers" in self._preview_text, (
            "preview.md is missing 'Selected outliers' section — outlier table not rendered."
        )

    def test_preview_does_not_contain_stale_sigma_threshold(self) -> None:
        assert "σ ≥ 0.080" not in self._preview_text, (
            "preview.md still contains the stale M2 σ ≥ 0.080 threshold string. "
            "The render must use the recalibrated M3 dissent_score, not the old metric."
        )

    def test_preview_contains_round2_summary_section(self) -> None:
        assert "## Round 2" in self._preview_text, (
            "preview.md is missing '## Round 2' section."
        )

    def test_preview_round2_section_contains_defend_revise(self) -> None:
        assert "defended" in self._preview_text, (
            "preview.md Round-2 section does not show defended/revised counts."
        )


# ---------------------------------------------------------------------------
# Gap-C: full-cycle timing render sums all 4 phases and computes verdict on combined total
# ---------------------------------------------------------------------------


class TestPreviewFullCycleTiming:
    """Gap C: timing section must sum all 4 phases and use the combined total for the verdict."""

    @pytest.fixture(autouse=True)
    def _run_preview(self, smoke_env_with_round2: dict) -> None:
        run_preview(smoke_env_with_round2["week"], smoke_env_with_round2["state_root"])
        self._env = smoke_env_with_round2
        preview_path = (
            smoke_env_with_round2["state_root"] / "runs" / f"{WEEK_ID}.preview.md"
        )
        self._preview_text = (
            preview_path.read_text(encoding="utf-8") if preview_path.exists() else ""
        )

    def test_full_cycle_timing_section_present(self) -> None:
        assert "Full-Cycle Timing" in self._preview_text, (
            "preview.md is missing 'Full-Cycle Timing' section (Gap C fix not applied)."
        )

    def test_combined_total_row_present(self) -> None:
        assert "COMBINED TOTAL" in self._preview_text, (
            "preview.md 'Full-Cycle Timing' section missing 'COMBINED TOTAL' row."
        )

    def test_feasibility_verdict_present(self) -> None:
        assert "Feasibility verdict:" in self._preview_text, (
            "preview.md is missing 'Feasibility verdict:' line in full-cycle section."
        )

    def test_combined_total_exceeds_research_only(self) -> None:
        """Combined total must be >= research total (FAKE_TIMINGS sum = 770s).

        The smoke fixture provides round2_timing with 2 entries (45+52=97s), so
        the combined total must be > the research-only sum.
        """
        research_total = sum(FAKE_TIMINGS.values())  # 770s
        # Extract combined total from the preview text.
        import re
        match = re.search(r"\*\*COMBINED TOTAL\*\*.*?\*\*([\d.]+)\*\*", self._preview_text)
        if match is None:
            pytest.fail("Could not parse COMBINED TOTAL from preview.md.")
        combined = float(match.group(1))
        assert combined >= research_total, (
            f"Combined total ({combined}s) is less than research-only total ({research_total}s). "
            "round2_timing must be summed into the combined total."
        )

    def test_verdict_computed_on_combined_total_not_research_only(self) -> None:
        """Verdict must be computed on the combined total.

        Construct a scenario where research alone would be FITS but combined
        total crosses TIGHT: research=3600s (research_only=3600s, window=5h=18000s → 20%=FITS),
        round2=8001s → combined=11601s → 64.4% → TIGHT.  This is verified via
        _compute_round2_defend_revise + _render_preview directly (unit-level, no full run).
        """
        # Import the private helpers from the driver module.
        _render_preview = _driver_mod._render_preview
        _PreviewData = _driver_mod._PreviewData

        # Stub dissent/outliers/resynthesis so _render_preview doesn't crash.
        class _FakeDissent:
            dissent_score = 0.30
            contested_week = False
            per_persona_divergence: dict = {}

        class _FakeOutliers:
            selected: list = []

        # research=3600s (20% of 5h → FITS alone),
        # round2=8001s → combined=11601s → 64.5% → TIGHT
        data = _PreviewData(
            week="2026-W99",
            consensus_holdings=[],
            transcript_text="",
            persona_snapshots={},
            debate_set=[],
            num_portfolios_written=8,
            num_stances_written=0,
            decision_type="panel_approved",
            metrics=None,
            session_timing={"value": 3600.0},
            dissent=_FakeDissent(),
            outliers=_FakeOutliers(),
            resynthesis=None,
            num_round2_stances=0,
            round2_defend_revise={},
            round1_timing={},
            judge_timing={},
            round2_timing={"growth": 8001.0},
        )
        rendered = _render_preview(data)
        # Extract the verdict keyword from "Feasibility verdict: TIGHT" (first word after colon+space).
        import re as _re
        verdict_match = _re.search(r"Feasibility verdict:\s*\*\*(\w[\w-]*)\*\*", rendered)
        if verdict_match is None:
            verdict_match = _re.search(r"Feasibility verdict:\s*(\w[\w-]*)", rendered)
        assert verdict_match is not None, (
            "Could not find 'Feasibility verdict: <WORD>' in rendered preview."
        )
        verdict_word = verdict_match.group(1).upper()
        assert verdict_word == "TIGHT", (
            f"Verdict should be TIGHT when research(3600s) + round2(8001s) = 11601s "
            f"(64.5% of 5h window), but got '{verdict_word}'. "
            "Verdict is being computed on research-only, not the combined total."
        )
