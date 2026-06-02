# test_budget.py — Unit tests for TASK-M1-005 per-persona budget enforcement.
#
# All deterministic.  No network calls, no file I/O beyond a tmp YAML fixture.
# Tests cover:
#   - load_budgets(): per-persona overrides + defaults fallback
#   - PersonaBudgetTracker: counting, remaining(), is_exhausted()
#   - Hard stop: data_tool_calls past cap → BudgetExhaustedError raised, call refused
#   - Web-search post-count: breach flag set and logged; no raise
#   - make_guarded(): guarded wrapper refuses calls past cap
#   - record_web_searches() / record_turns(): post-run bulk recording

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from round_table_portfolio.budget import (
    BudgetExhaustedError,
    PersonaBudget,
    PersonaBudgetTracker,
    get_budget,
    guarded_data_tool,
    load_budgets,
    make_guarded,
    record_turns,
    record_web_searches,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUDGET_YAML = textwrap.dedent("""\
    defaults:
      max_turns: 10
      max_web_searches: 5
      max_data_tool_calls: 8

    personas:
      value:
        max_turns: 12
        max_web_searches: 6
        max_data_tool_calls: 18
      technical:
        max_turns: 12
        max_web_searches: 4
        max_data_tool_calls: 20
""")


@pytest.fixture()
def budget_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "persona_budgets.yaml"
    p.write_text(BUDGET_YAML, encoding="utf-8")
    return p


@pytest.fixture()
def budgets(budget_yaml: Path) -> dict[str, PersonaBudget]:
    return load_budgets(config_path=budget_yaml)


@pytest.fixture()
def value_tracker(budgets: dict[str, PersonaBudget]) -> PersonaBudgetTracker:
    return PersonaBudgetTracker("value", get_budget(budgets, "value"))


@pytest.fixture()
def default_tracker(budgets: dict[str, PersonaBudget]) -> PersonaBudgetTracker:
    """Tracker for a persona NOT listed in the yaml — should get defaults."""
    return PersonaBudgetTracker("quant-systematic", get_budget(budgets, "quant-systematic"))


# ---------------------------------------------------------------------------
# load_budgets — per-persona overrides + defaults fallback
# ---------------------------------------------------------------------------

class TestLoadBudgets:
    def test_listed_persona_overrides(self, budgets: dict[str, PersonaBudget]) -> None:
        b = get_budget(budgets, "value")
        assert b.max_turns == 12
        assert b.max_web_searches == 6
        assert b.max_data_tool_calls == 18

    def test_second_listed_persona(self, budgets: dict[str, PersonaBudget]) -> None:
        b = get_budget(budgets, "technical")
        assert b.max_data_tool_calls == 20
        assert b.max_web_searches == 4

    def test_unlisted_persona_gets_defaults(self, budgets: dict[str, PersonaBudget]) -> None:
        # "quant-systematic" is not listed in the fixture yaml
        b = get_budget(budgets, "quant-systematic")
        assert b.max_turns == 10
        assert b.max_web_searches == 5
        assert b.max_data_tool_calls == 8

    def test_unknown_slug_never_raises(self, budgets: dict[str, PersonaBudget]) -> None:
        b = get_budget(budgets, "nonexistent-persona")
        assert isinstance(b, PersonaBudget)

    def test_defaults_key_present(self, budgets: dict[str, PersonaBudget]) -> None:
        assert "__defaults__" in budgets
        assert budgets["__defaults__"].max_turns == 10

    def test_loads_real_config(self) -> None:
        """Smoke-test against the actual config/persona_budgets.yaml."""
        real = load_budgets(config_path=Path("config/persona_budgets.yaml"))
        b = get_budget(real, "value")
        assert b.max_data_tool_calls > 0
        assert b.max_turns > 0


# ---------------------------------------------------------------------------
# PersonaBudgetTracker — counting, remaining, is_exhausted
# ---------------------------------------------------------------------------

class TestTrackerCounting:
    def test_starts_at_zero(self, value_tracker: PersonaBudgetTracker) -> None:
        assert value_tracker.counts == {"data_tool_calls": 0, "web_searches": 0, "turns": 0}

    def test_remaining_full_at_start(self, value_tracker: PersonaBudgetTracker) -> None:
        assert value_tracker.remaining("data_tool_calls") == 18
        assert value_tracker.remaining("web_searches") == 6
        assert value_tracker.remaining("turns") == 12

    def test_record_data_tool_decrements_remaining(
        self, value_tracker: PersonaBudgetTracker
    ) -> None:
        value_tracker.record("data_tool_calls")
        assert value_tracker.remaining("data_tool_calls") == 17
        assert value_tracker.counts["data_tool_calls"] == 1

    def test_record_web_search_decrements_remaining(
        self, value_tracker: PersonaBudgetTracker
    ) -> None:
        value_tracker.record("web_searches")
        assert value_tracker.remaining("web_searches") == 5

    def test_record_turns_bulk(self, value_tracker: PersonaBudgetTracker) -> None:
        value_tracker.record("turns", count=7)
        assert value_tracker.counts["turns"] == 7
        assert value_tracker.remaining("turns") == 5

    def test_is_not_exhausted_below_cap(self, value_tracker: PersonaBudgetTracker) -> None:
        for _ in range(17):
            value_tracker.record("data_tool_calls")
        assert not value_tracker.is_exhausted("data_tool_calls")

    def test_is_exhausted_at_cap(self, value_tracker: PersonaBudgetTracker) -> None:
        for _ in range(18):
            value_tracker.record("data_tool_calls")
        assert value_tracker.is_exhausted("data_tool_calls")
        assert value_tracker.remaining("data_tool_calls") == 0

    def test_invalid_kind_raises_value_error(
        self, value_tracker: PersonaBudgetTracker
    ) -> None:
        with pytest.raises(ValueError, match="Unknown budget kind"):
            value_tracker.record("invalid_kind")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hard stop: data_tool_calls past cap → BudgetExhaustedError, call refused
# ---------------------------------------------------------------------------

class TestHardStop:
    def test_raises_on_next_call_past_cap(self, value_tracker: PersonaBudgetTracker) -> None:
        for _ in range(18):
            value_tracker.record("data_tool_calls")
        with pytest.raises(BudgetExhaustedError) as exc_info:
            value_tracker.record("data_tool_calls")
        err = exc_info.value
        assert err.persona == "value"
        assert err.kind == "data_tool_calls"
        assert err.cap == 18
        assert err.attempted == 19

    def test_counter_not_incremented_on_refusal(
        self, value_tracker: PersonaBudgetTracker
    ) -> None:
        for _ in range(18):
            value_tracker.record("data_tool_calls")
        with pytest.raises(BudgetExhaustedError):
            value_tracker.record("data_tool_calls")
        # Counter must stay at cap — no phantom increment.
        assert value_tracker.counts["data_tool_calls"] == 18

    def test_guarded_data_tool_refuses_past_cap(
        self, value_tracker: PersonaBudgetTracker
    ) -> None:
        calls: list[str] = []

        def fake_tool(symbol: str) -> str:
            calls.append(symbol)
            return f"data:{symbol}"

        for _ in range(18):
            guarded_data_tool(fake_tool, value_tracker, "AAPL")
        assert len(calls) == 18

        with pytest.raises(BudgetExhaustedError):
            guarded_data_tool(fake_tool, value_tracker, "MSFT")

        # The underlying fn was NOT called on the refused attempt.
        assert len(calls) == 18

    def test_default_tracker_hard_stop_at_8(
        self, default_tracker: PersonaBudgetTracker
    ) -> None:
        """Unlisted persona (defaults cap=8) is also hard-stopped."""
        for _ in range(8):
            default_tracker.record("data_tool_calls")
        with pytest.raises(BudgetExhaustedError) as exc_info:
            default_tracker.record("data_tool_calls")
        assert exc_info.value.cap == 8


# ---------------------------------------------------------------------------
# Web-search post-count: breach flag, no raise
# ---------------------------------------------------------------------------

class TestWebSearchPostCount:
    def test_no_breach_within_cap(self, value_tracker: PersonaBudgetTracker) -> None:
        record_web_searches(value_tracker, 6)
        assert not value_tracker.web_search_breach
        assert value_tracker.counts["web_searches"] == 6

    def test_breach_flag_set_over_cap(self, value_tracker: PersonaBudgetTracker) -> None:
        record_web_searches(value_tracker, 9)  # cap is 6
        assert value_tracker.web_search_breach
        assert value_tracker.remaining("web_searches") == -3  # negative = over cap

    def test_breach_does_not_raise(self, value_tracker: PersonaBudgetTracker) -> None:
        # Must NOT raise — web search is a soft cap (no hard intercept).
        record_web_searches(value_tracker, 100)
        assert value_tracker.web_search_breach

    def test_incremental_breach_detection(self, value_tracker: PersonaBudgetTracker) -> None:
        record_web_searches(value_tracker, 5)
        assert not value_tracker.web_search_breach
        record_web_searches(value_tracker, 2)  # now at 7 > cap 6
        assert value_tracker.web_search_breach


# ---------------------------------------------------------------------------
# make_guarded — wraps a tool fn, same hard-stop behaviour
# ---------------------------------------------------------------------------

class TestMakeGuarded:
    def test_guarded_fn_passes_through_within_cap(
        self, default_tracker: PersonaBudgetTracker
    ) -> None:
        results: list[int] = []

        def real_tool(x: int) -> int:
            results.append(x)
            return x * 2

        guarded = make_guarded(real_tool, default_tracker)
        assert guarded(5) == 10
        assert results == [5]

    def test_guarded_fn_refused_past_cap(
        self, default_tracker: PersonaBudgetTracker
    ) -> None:
        guarded = make_guarded(lambda: None, default_tracker)
        for _ in range(8):
            guarded()
        with pytest.raises(BudgetExhaustedError):
            guarded()

    def test_guarded_preserves_fn_name(
        self, default_tracker: PersonaBudgetTracker
    ) -> None:
        def my_tool() -> None:
            pass

        guarded = make_guarded(my_tool, default_tracker)
        assert guarded.__name__ == "my_tool"


# ---------------------------------------------------------------------------
# record_turns — post-run bulk recording
# ---------------------------------------------------------------------------

class TestRecordTurns:
    def test_record_turns_bulk(self, value_tracker: PersonaBudgetTracker) -> None:
        record_turns(value_tracker, 11)
        assert value_tracker.counts["turns"] == 11

    def test_record_turns_over_cap_does_not_raise(
        self, value_tracker: PersonaBudgetTracker
    ) -> None:
        # Turns are dispatch-bounded, not hard-interceptable mid-subagent.
        record_turns(value_tracker, 99)
        assert value_tracker.is_exhausted("turns")


# ---------------------------------------------------------------------------
# summary() / counts — cost-log output shape
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_shape(self, value_tracker: PersonaBudgetTracker) -> None:
        value_tracker.record("data_tool_calls")
        record_web_searches(value_tracker, 3)
        record_turns(value_tracker, 8)

        s = value_tracker.summary()
        assert s["persona"] == "value"
        assert s["data_tool_calls"]["used"] == 1
        assert s["data_tool_calls"]["cap"] == 18
        assert s["web_searches"]["used"] == 3
        assert s["web_searches"]["cap"] == 6
        assert not s["web_searches"]["breach"]
        assert s["turns"]["used"] == 8
        assert s["turns"]["cap"] == 12
