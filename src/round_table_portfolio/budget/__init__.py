# budget/__init__.py — Per-persona budget enforcement module.
#
# Public API:
#   load_budgets()           — read persona_budgets.yaml → dict[str, PersonaBudget]
#   get_budget()             — look up a budget, with defaults fallback
#   PersonaBudget            — frozen dataclass (max_turns, max_web_searches,
#                              max_data_tool_calls)
#   PersonaBudgetTracker     — per-persona counter + hard/soft enforcement
#   BudgetExhaustedError     — raised on hard-capped data-tool calls over cap
#   guarded_data_tool()      — call a data tool after budget check
#   make_guarded()           — wrap a tool fn for a given tracker
#   record_web_searches()    — post-run web-search count recording
#   record_turns()           — post-run turn count recording

from round_table_portfolio.budget.loader import PersonaBudget, load_budgets, get_budget
from round_table_portfolio.budget.tracker import PersonaBudgetTracker, BudgetExhaustedError
from round_table_portfolio.budget.guarded_tools import (
    guarded_data_tool,
    make_guarded,
    record_web_searches,
    record_turns,
)

__all__ = [
    "PersonaBudget",
    "load_budgets",
    "get_budget",
    "PersonaBudgetTracker",
    "BudgetExhaustedError",
    "guarded_data_tool",
    "make_guarded",
    "record_web_searches",
    "record_turns",
]
