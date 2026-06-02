# budget/guarded_tools.py — Budget-guarded wrappers for the data-tool layer.
#
# Each wrapper checks the persona's PersonaBudgetTracker BEFORE forwarding a
# call to the underlying data tool.  If the tracker raises BudgetExhaustedError
# the wrapper catches it, logs it, and re-raises — the call never reaches the
# network.
#
# HARD vs SOFT enforcement summary (documented here per TDD §1.4 bounding):
#
#   data_tool_calls — HARD (enforced here before network).
#   web_searches    — SOFT: cap is surfaced in the persona prompt; post-run
#                     count is recorded via record_web_searches(); breach flag
#                     is set on the tracker.  Native WebSearch is not routable
#                     through this layer.
#   turns           — SOFT: max_turns is the dispatch-level subagent limit
#                     (M2 orchestrator).  Actual count recorded post-run.
#
# TDD Component 2 bounding + TASK-M1-005.

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from round_table_portfolio.budget.tracker import BudgetExhaustedError, PersonaBudgetTracker

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def guarded_data_tool(
    fn: Callable[..., Any],
    tracker: PersonaBudgetTracker,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call *fn* only if the persona's data-tool budget is not exhausted.

    Raises:
        BudgetExhaustedError: propagated from the tracker when the cap is hit.
            The underlying *fn* is NOT called — no network request is made.
    """
    # record() raises BudgetExhaustedError before incrementing when over cap.
    tracker.record("data_tool_calls")
    return fn(*args, **kwargs)


def make_guarded(fn: Callable[..., Any], tracker: PersonaBudgetTracker) -> Callable[..., Any]:
    """Return a budget-guarded version of *fn* bound to *tracker*.

    The returned callable has the same signature as *fn* but prefixes every
    call with a budget check.  Use this to wrap tools before handing them to a
    persona subagent context so every call is accounted for without the caller
    needing to think about budget accounting.

    Example::

        budgets = load_budgets()
        tracker = PersonaBudgetTracker("value", get_budget(budgets, "value"))
        guarded_get_prices = make_guarded(get_prices, tracker)
        # Now: guarded_get_prices("AAPL") — counts against budget or raises.
    """
    import functools

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return guarded_data_tool(fn, tracker, *args, **kwargs)

    return _wrapper


def record_web_searches(tracker: PersonaBudgetTracker, count: int) -> None:
    """Record *count* web searches against *tracker* post-run.

    Called by the orchestrator (M2) after a persona subagent completes, using
    the web-search count parsed from the persona's result transcript.  This
    feeds NFR-#6 cost logging and sets the breach flag if the persona exceeded
    its cap.

    Args:
        tracker: The persona's tracker for this run.
        count:   Total web searches the persona made (from transcript parse).
    """
    tracker.record("web_searches", count=count)
    if tracker.web_search_breach:
        logger.warning(
            "Web-search breach for persona '%s': used %d, cap %d — "
            "included in cost log; no hard stop (native tool not interceptable).",
            tracker.persona,
            tracker.counts["web_searches"],
            tracker.budget.max_web_searches,
        )


def record_turns(tracker: PersonaBudgetTracker, count: int) -> None:
    """Record actual turn count post-run (for cost log; not a hard stop).

    Args:
        tracker: The persona's tracker for this run.
        count:   Number of turns the subagent used (from subagent result).
    """
    tracker.record("turns", count=count)
