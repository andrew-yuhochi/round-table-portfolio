"""Smoke test for --mode prepare-round1 in scripts/weekly_run.py.

Verifies:
1. debate_set.json is written to state/runs/<week>.debate_set.json.
2. debate_set list is non-empty and equals what construct_debate_set produces
   over the same persona_replies inputs.
3. prepare-round1 wrote NOTHING into the real state/ directory beyond
   state/runs/<week>.debate_set.json (temp state_root was used internally).
4. The per-persona digest contains exactly 7 entries (one per persona slug).

SKIP_LIVE=1 safe — no web search, no market data, no subagent dispatch.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from round_table_portfolio.storage.apply_schema import apply_schema

# ---------------------------------------------------------------------------
# Import the driver module (same loader pattern as the existing smoke test).
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[2]
_SCRIPTS = _PROJECT_ROOT / "scripts"


def _import_driver() -> Any:
    spec = importlib.util.spec_from_file_location(
        "weekly_run_driver_prep", _SCRIPTS / "weekly_run.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_driver_mod = _import_driver()

run_prepare_round1 = _driver_mod.run_prepare_round1
_PERSONA_SLUGS = _driver_mod._PERSONA_SLUGS

# ---------------------------------------------------------------------------
# Reuse the same per-persona output builder from the existing smoke test so
# fixtures are consistent across the two test files.
# ---------------------------------------------------------------------------

from round_table_portfolio.orchestrator.round1 import construct_debate_set
from round_table_portfolio.research.runner import run_persona_research
from round_table_portfolio.personas.output_validator import StubOnMandateJudge, load_validator_config
from round_table_portfolio.budget.loader import get_budget, load_budgets

_PERSONA_VOCAB: dict[str, dict[str, str]] = {
    "value": {
        "tickers": "AAPL MSFT",
        "flavor": (
            "P/E of 18x is below the sector median of 22x, suggesting deep value. "
            "FCF yield of 5.2% and EPS growth of 12% YoY support the thesis. "
            "Balance-sheet: net debt/EBITDA of 1.2x. Dividend yield: 2.1%. ROE: 28%. "
            "Intrinsic value via DCF: $195. Data sources: EDGAR 10-K, FRED, price history. "
        ),
    },
    "growth": {
        "tickers": "NVDA MSFT",
        "flavor": (
            "Revenue growth of 42% YoY driven by AI infrastructure demand. "
            "ARR acceleration and expanding operating margin (+400 bps). "
            "P/E of 45x justified by TAM expansion. EPS growth: 60% YoY. FCF margin: 32%. "
            "Data sources: SEC filings, earnings transcripts, Alpaca price data. "
        ),
    },
    "discretionary-macro": {
        "tickers": "SPY GOOGL",
        "flavor": (
            "CPI inflation at 3.2%. PCE core at 2.8% keeps the Fed on hold through Q3. "
            "ISM manufacturing PMI at 51.2. Yield curve: 10Y-2Y spread at +15 bps. "
            "FCF margins improving. P/E multiples compressed. "
            "Data sources: FRED macro series, ISM, BLS CPI releases. "
        ),
    },
    "cta-systematic-macro": {
        "tickers": "QQQ SPY",
        "flavor": (
            "Trend signal: 12-month momentum score +0.82. RSI 14-day at 58. "
            "MACD crossover confirmed. VIX at 14, low-vol expansion phase. "
            "FCF yield spread vs 10Y treasury: 220 bps. EPS revision breadth: +65%. "
            "Data sources: price history via Alpaca, FRED, Bloomberg macro feeds. "
        ),
    },
    "technical": {
        "tickers": "AAPL TSLA",
        "flavor": (
            "50-day SMA acting as support at $178; 200-day SMA at $165. "
            "RSI 14-day: 54 neutral. MACD histogram turning positive. VWAP: $182. "
            "FCF and EPS used as secondary confirmation. "
            "Data sources: price history via Alpaca, technical indicator library. "
        ),
    },
    "quant-systematic": {
        "tickers": "MSFT AMZN",
        "flavor": (
            "Factor model: value Z-score +1.4, momentum Z-score +0.9. "
            "Quality factor (ROE, FCF stability): top quintile. Low-vol tilt: 22% vs 31%. "
            "EPS surprise factor: +0.8 sigma. P/E relative sector: -0.6 sigma. "
            "Data sources: EDGAR fundamentals, FRED macro, Alpaca price series. "
        ),
    },
    "risk-officer": {
        "tickers": "SPY TLT",
        "flavor": (
            "Tail-risk scenario: -25% drawdown if Fed delivers surprise 50 bps hike. "
            "VaR 95% (10-day): 3.8% of portfolio. Concentration risk: top-3 = 38% NAV. "
            "FCF coverage of dividend: 2.1x. EPS sensitivity to rate shock: -12%. "
            "Data sources: FRED stress scenarios, EDGAR filings, price history. "
        ),
    },
}


def _make_persona_output(slug: str) -> str:
    vocab = _PERSONA_VOCAB[slug]
    tickers = vocab["tickers"].split()
    flavor = vocab["flavor"]
    report_body = (
        f"The {slug} analysis identifies compelling opportunities. "
        f"Primary names: {tickers[0]} and {tickers[1]}. "
        + flavor
        + f"Conviction: high for {tickers[0]}, moderate for {tickers[1]}. "
        "Portfolio weight recommendation: fully invested per mandate."
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
                "why": "Secondary opportunity.",
                "cluster": [],
            },
        ],
        "report": report_body,
        "web_searches_used": 4,
        "data_tool_calls_used": 8,
    }
    return json.dumps(schema)


# ---------------------------------------------------------------------------
# Reference compute: what construct_debate_set produces over these inputs.
# Used as ground truth for assertion #2.
# ---------------------------------------------------------------------------

def _reference_debate_set(persona_replies: dict[str, str], tmp_path: Path) -> list[str]:
    """Recompute the debate set the same way the engine does, in a temp dir."""
    import yaml

    budget_config = _PROJECT_ROOT / "config" / "persona_budgets.yaml"
    thresholds_config = _PROJECT_ROOT / "config" / "thresholds.yaml"
    validator_config_path = _PROJECT_ROOT / "config" / "validator.yaml"

    budget_raw = yaml.safe_load(budget_config.read_text(encoding="utf-8")) or {}
    thresholds = yaml.safe_load(thresholds_config.read_text(encoding="utf-8")) or {}
    max_position_weight = float(thresholds.get("max_position_weight", 0.20))

    budgets = load_budgets(budget_config)
    v_config = load_validator_config(validator_config_path)
    judge = StubOnMandateJudge()

    tmp_state = tmp_path / "ref_state"
    tmp_state.mkdir()

    persona_results = []
    for slug in _PERSONA_SLUGS:
        raw = persona_replies[slug]
        budget = get_budget(budgets, slug)
        result = run_persona_research(
            persona_slug=slug,
            week_id="2026-W99",
            raw_output=raw,
            mandate="",
            judge=judge,
            budget=budget,
            validator_config=v_config,
            state_root=tmp_state,
        )
        persona_results.append(result)

    debate_cfg = {
        "debate_set_ceiling": budget_raw.get("debate_set_ceiling", 40),
        "max_position_weight": max_position_weight,
    }
    return construct_debate_set(persona_results, debate_cfg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WEEK_ID = "2026-W99"  # synthetic week — never collides with a real run


@pytest.fixture()
def prep_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Minimal temp env for the prepare-round1 mode.

    Writes only persona_replies.json (the sole input for this mode).
    Does NOT write round1_replies / judge_verdicts / timing — not needed.
    """
    state_root = tmp_path / "state"
    state_root.mkdir()
    runs_dir = state_root / "runs"
    runs_dir.mkdir()

    persona_replies = {slug: _make_persona_output(slug) for slug in _PERSONA_SLUGS}

    (runs_dir / f"{WEEK_ID}.persona_replies.json").write_text(
        json.dumps(persona_replies), encoding="utf-8"
    )

    # prepare-round1 now emits consensus_book.md by opening state/ledger.db.
    # Seed an empty (schema-only, no prior consensus) ledger so the week-one
    # path runs without crashing.
    apply_schema(db_path=state_root / "ledger.db")

    # Patch _PROJECT_ROOT so the driver reads real config/ files.
    monkeypatch.setattr(_driver_mod, "_PROJECT_ROOT", _PROJECT_ROOT)

    return {
        "state_root": state_root,
        "runs_dir": runs_dir,
        "persona_replies": persona_replies,
        "week": WEEK_ID,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrepareRound1:
    """Validates the prepare-round1 mode contract."""

    def test_debate_set_json_written(self, prep_env: dict) -> None:
        """debate_set.json must exist after the mode completes."""
        run_prepare_round1(prep_env["week"], prep_env["state_root"])

        debate_path = prep_env["runs_dir"] / f"{WEEK_ID}.debate_set.json"
        assert debate_path.exists(), (
            f"debate_set.json was not written at {debate_path}"
        )

    def test_debate_set_is_nonempty_list(self, prep_env: dict) -> None:
        """debate_set must be a non-empty list of ticker strings."""
        run_prepare_round1(prep_env["week"], prep_env["state_root"])

        debate_path = prep_env["runs_dir"] / f"{WEEK_ID}.debate_set.json"
        payload = json.loads(debate_path.read_text(encoding="utf-8"))

        assert "debate_set" in payload, "debate_set.json missing 'debate_set' key."
        tickers = payload["debate_set"]
        assert isinstance(tickers, list), "'debate_set' must be a list."
        assert len(tickers) > 0, "'debate_set' must be non-empty."

    def test_debate_set_matches_reference(self, prep_env: dict, tmp_path: Path) -> None:
        """The emitted debate_set must equal construct_debate_set over the same inputs."""
        run_prepare_round1(prep_env["week"], prep_env["state_root"])

        debate_path = prep_env["runs_dir"] / f"{WEEK_ID}.debate_set.json"
        payload = json.loads(debate_path.read_text(encoding="utf-8"))
        emitted = payload["debate_set"]

        reference = _reference_debate_set(prep_env["persona_replies"], tmp_path)

        assert emitted == reference, (
            f"Emitted debate_set {emitted} differs from reference {reference}. "
            "prepare-round1 must use the same construct_debate_set path as commit."
        )

    def test_no_side_effects_in_real_state(self, prep_env: dict) -> None:
        """prepare-round1 must write NOTHING into real state/ except known outputs.

        Known outputs after M6-003 fix-forward:
          - runs/<week>.debate_set.json       (debate set)
          - runs/<week>-memory/consensus_book.md  (emitted for Round-1 injection)

        The persona_replies.json is pre-existing input (written by the fixture).
        """
        state_root = prep_env["state_root"]

        run_prepare_round1(prep_env["week"], state_root)

        # Enumerate every file in state/ after the run.
        all_files = [
            p for p in state_root.rglob("*")
            if p.is_file()
        ]
        # Allow: debate_set.json (primary output), persona_replies.json (pre-existing
        # fixture input), consensus_book.md (M6-003 fix: emitted for Round-1 injection),
        # and ledger.db (created by prep_env for the book emission call).
        memory_dir = prep_env["runs_dir"] / f"{WEEK_ID}-memory"
        allowed = {
            prep_env["runs_dir"] / f"{WEEK_ID}.debate_set.json",
            prep_env["runs_dir"] / f"{WEEK_ID}.persona_replies.json",
            memory_dir / "consensus_book.md",
            state_root / "ledger.db",
        }
        unexpected = [p for p in all_files if p not in allowed]
        assert not unexpected, (
            f"prepare-round1 wrote unexpected files into real state/:\n"
            + "\n".join(f"  {p}" for p in unexpected)
        )

    def test_persona_digest_has_7_entries(self, prep_env: dict) -> None:
        """persona_digest must have exactly 7 entries — one per persona slug."""
        run_prepare_round1(prep_env["week"], prep_env["state_root"])

        debate_path = prep_env["runs_dir"] / f"{WEEK_ID}.debate_set.json"
        payload = json.loads(debate_path.read_text(encoding="utf-8"))

        assert "persona_digest" in payload, (
            "debate_set.json missing 'persona_digest' key."
        )
        digest = payload["persona_digest"]
        assert len(digest) == 7, (
            f"persona_digest has {len(digest)} entries, expected 7."
        )
        assert set(digest.keys()) == set(_PERSONA_SLUGS), (
            f"persona_digest keys {set(digest.keys())} != expected {set(_PERSONA_SLUGS)}"
        )

    def test_persona_digest_structure(self, prep_env: dict) -> None:
        """Each digest entry must have 'shortlist' (list) and 'report_excerpt' (str)."""
        run_prepare_round1(prep_env["week"], prep_env["state_root"])

        debate_path = prep_env["runs_dir"] / f"{WEEK_ID}.debate_set.json"
        payload = json.loads(debate_path.read_text(encoding="utf-8"))

        for slug, entry in payload["persona_digest"].items():
            assert "shortlist" in entry, f"digest[{slug!r}] missing 'shortlist'."
            assert isinstance(entry["shortlist"], list), (
                f"digest[{slug!r}]['shortlist'] must be a list."
            )
            assert "report_excerpt" in entry, f"digest[{slug!r}] missing 'report_excerpt'."
            assert isinstance(entry["report_excerpt"], str), (
                f"digest[{slug!r}]['report_excerpt'] must be a string."
            )


# ---------------------------------------------------------------------------
# Helpers for consensus_book emission tests (M6-003 fix-forward)
# ---------------------------------------------------------------------------


def _seed_consensus_week(
    db_path: Path,
    week_id: str,
    tickers: list[str],
    equity_weight: float = 0.15,
    cash_weight: float = 0.55,
    user_id: str = "andrew",
) -> None:
    """Seed a committed consensus portfolio + holdings into db_path for week_id."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("BEGIN")
    try:
        conn.execute(
            "INSERT OR IGNORE INTO roster_versions (roster_version, description) "
            "VALUES (1, 'seed')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO enhancement_versions (enhancement_version, description) "
            "VALUES (1, 'seed')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO weeks (week_id, run_date, notes, user_id) "
            "VALUES (?, '2026-01-01', 'seeded', ?)",
            (week_id, user_id),
        )
        conn.execute(
            "INSERT INTO portfolios "
            "(week_id, type, user_id, roster_version, enhancement_version, created_at) "
            "VALUES (?, 'consensus', ?, 1, 1, '2026-01-01T00:00:00Z')",
            (week_id, user_id),
        )
        port_id = conn.execute(
            "SELECT portfolio_id FROM portfolios "
            "WHERE week_id=? AND type='consensus' AND user_id=?",
            (week_id, user_id),
        ).fetchone()[0]
        for ticker in tickers:
            conn.execute(
                "INSERT INTO holdings "
                "(portfolio_id, ticker, weight, action, entry_date, user_id, roster_version) "
                "VALUES (?, ?, ?, 'add', ?, ?, 1)",
                (port_id, ticker, equity_weight, week_id, user_id),
            )
        conn.execute(
            "INSERT INTO holdings "
            "(portfolio_id, ticker, weight, action, entry_date, user_id, roster_version) "
            "VALUES (?, 'CASH', ?, 'hold', ?, ?, 1)",
            (port_id, cash_weight, week_id, user_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# TestConsensusbookEmissionAtPrepareRound1 (M6-003 fix-forward)
# ---------------------------------------------------------------------------


class TestConsensusbookEmissionAtPrepareRound1:
    """Validates that prepare-round1 emits consensus_book.md before Round-1 dispatch.

    Root cause (TASK-M6-003 follow-up): the book was only generated inside
    run_weekly() at Stage 6–7 (preview/commit), so it was never available when
    the session dispatched Round-1 persona subagents at Stage 3.  The fix emits
    it from prepare-round1 — the last driver step before Stage 3.

    Coverage:
      1. consensus_book_written         — file exists at <week>-memory/consensus_book.md
      2. week_one_file_written          — week-one note written when no prior consensus
      3. prior_tickers_in_book          — prior-week tickers appear in emitted file
      4. commit_before_reveal_at_prep   — current-week row excluded; prior-week row used
    """

    @pytest.fixture()
    def env_with_ledger(self, prep_env: dict) -> dict:
        """prep_env already seeds an empty ledger (week-one path). Return as-is."""
        return prep_env

    @pytest.fixture()
    def env_with_prior(self, prep_env: dict) -> dict:
        """Extend prep_env's already-created ledger with a prior consensus week."""
        db_path = prep_env["state_root"] / "ledger.db"
        _seed_consensus_week(db_path, "2026-W98", ["NVDA", "MSFT", "AAPL"])
        return prep_env

    def test_consensus_book_written(self, env_with_ledger: dict) -> None:
        """consensus_book.md must exist at <week>-memory/ after prepare-round1."""
        run_prepare_round1(env_with_ledger["week"], env_with_ledger["state_root"])

        expected = (
            env_with_ledger["runs_dir"]
            / f"{WEEK_ID}-memory"
            / "consensus_book.md"
        )
        assert expected.exists(), (
            f"consensus_book.md not found at {expected} — "
            "prepare-round1 must emit it so the session can inject it at Round-1 dispatch"
        )

    def test_week_one_file_written(self, env_with_ledger: dict) -> None:
        """With no prior consensus, prepare-round1 emits the week-one note (no crash)."""
        run_prepare_round1(env_with_ledger["week"], env_with_ledger["state_root"])

        book_path = (
            env_with_ledger["runs_dir"]
            / f"{WEEK_ID}-memory"
            / "consensus_book.md"
        )
        assert book_path.exists(), "consensus_book.md must be written even on week one"
        content = book_path.read_text(encoding="utf-8")
        # The week-one note is the explicit sentinel (not empty, not a ticker table).
        assert len(content) > 0, "consensus_book.md must not be empty on week one"

    def test_prior_tickers_in_book(self, env_with_prior: dict) -> None:
        """Prior-week tickers appear in the emitted consensus_book.md."""
        run_prepare_round1(env_with_prior["week"], env_with_prior["state_root"])

        book_path = (
            env_with_prior["runs_dir"]
            / f"{WEEK_ID}-memory"
            / "consensus_book.md"
        )
        assert book_path.exists()
        content = book_path.read_text(encoding="utf-8")
        for ticker in ("NVDA", "MSFT", "AAPL", "CASH"):
            assert ticker in content, (
                f"Prior-week ticker {ticker} missing from consensus_book.md"
            )

    def test_commit_before_reveal_at_prepare_round1(self, prep_env: dict) -> None:
        """Current-week consensus row is excluded; prior-week row is used.

        Seeds both W98 (prior) and WEEK_ID (current — simulating a pre-existing
        run) so the SQL guard in load_current_consensus_book is exercised at the
        prepare-round1 call site.
        """
        db_path = prep_env["state_root"] / "ledger.db"  # already created by prep_env
        _seed_consensus_week(db_path, "2026-W98", ["NVDA", "MSFT"])
        # Current-week row — must be excluded by the commit-before-reveal guard.
        _seed_consensus_week(db_path, WEEK_ID, ["TSLA", "META"])

        run_prepare_round1(prep_env["week"], prep_env["state_root"])

        book_path = (
            prep_env["runs_dir"]
            / f"{WEEK_ID}-memory"
            / "consensus_book.md"
        )
        assert book_path.exists()
        content = book_path.read_text(encoding="utf-8")

        # Prior-week tickers must appear; current-week tickers must not.
        assert "NVDA" in content, "Prior-week NVDA missing from book"
        assert "MSFT" in content, "Prior-week MSFT missing from book"
        assert "TSLA" not in content, "Current-week TSLA leaked into book (commit-before-reveal violated)"
        assert "META" not in content, "Current-week META leaked into book (commit-before-reveal violated)"
