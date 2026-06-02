# budget/tracker.py — Per-persona budget tracker.
#
# Counts data_tool_calls, web_searches, and turns for one persona during a
# weekly run.  Hard enforcement model:
#
#   data_tool_calls — HARD CAP.  The data-tool layer (guarded_tools.py) checks
#       this tracker before every call and raises BudgetExhaustedError when the
#       cap is hit.  No call reaches the network.
#
#   web_searches — DISPATCH-BOUNDED + COUNTED.  Native WebSearch does not route
#       through this layer, so it cannot be intercepted mid-subagent.  The cap
#       is exposed as max_web_searches for inclusion in the persona's research
#       prompt (soft instruction the persona is told to respect), and the count
#       is recorded here after the run via record_web_searches() for NFR-#6
#       cost logging and breach detection.
#
#   turns — DISPATCH-BOUNDED + COUNTED.  max_turns is the value the orchestrator
#       passes as the subagent's turn limit at dispatch time (M2).  This tracker
#       records the actual turn count post-run for cost logging.
#
# TDD Component 2 bounding + TASK-M1-005.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from round_table_portfolio.budget.loader import PersonaBudget

logger = logging.getLogger(__name__)

BudgetKind = Literal["data_tool_calls", "web_searches", "turns"]


class BudgetExhaustedError(RuntimeError):
    """Raised when a persona attempts a data-tool call past its hard cap.

    Attributes:
        persona:   Persona slug.
        kind:      Budget dimension that was exhausted.
        cap:       The configured cap.
        attempted: Count at the time of the violation.
    """

    def __init__(self, persona: str, kind: BudgetKind, cap: int, attempted: int) -> None:
        self.persona = persona
        self.kind = kind
        self.cap = cap
        self.attempted = attempted
        super().__init__(
            f"Budget exhausted for persona '{persona}': "
            f"{kind} cap={cap}, attempted call #{attempted}"
        )


@dataclass
class PersonaBudgetTracker:
    """Track one persona's resource consumption during a weekly run.

    Usage::

        tracker = PersonaBudgetTracker("value", budget)
        tracker.record("data_tool_calls")   # raises BudgetExhaustedError if over cap
        tracker.record("turns", count=12)   # post-run bulk record (no raise)
        tracker.is_exhausted("web_searches")
        tracker.remaining("data_tool_calls")
    """

    persona: str
    budget: PersonaBudget

    _data_tool_calls: int = field(default=0, init=False, repr=False)
    _web_searches: int = field(default=0, init=False, repr=False)
    _turns: int = field(default=0, init=False, repr=False)
    _web_search_breach: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def record(self, kind: BudgetKind, *, count: int = 1) -> None:
        """Record *count* units of *kind* being consumed.

        For ``"data_tool_calls"``: raises :exc:`BudgetExhaustedError` BEFORE
        incrementing the counter when the cap would be exceeded.  This is the
        hard-stop point wired in guarded_tools.py.

        For ``"web_searches"`` and ``"turns"``: always increments (these are
        counted, not interceptable).  Sets the ``_web_search_breach`` flag when
        web searches exceed the cap.

        Args:
            kind:  Budget dimension.
            count: How many units to add (default 1; use >1 for post-run bulk).
        """
        if kind == "data_tool_calls":
            # Hard enforcement — raise before the call goes to the network.
            new_total = self._data_tool_calls + count
            if new_total > self.budget.max_data_tool_calls:
                raise BudgetExhaustedError(
                    self.persona,
                    "data_tool_calls",
                    self.budget.max_data_tool_calls,
                    new_total,
                )
            self._data_tool_calls = new_total
            logger.debug(
                "Budget[%s] data_tool_calls=%d/%d",
                self.persona,
                self._data_tool_calls,
                self.budget.max_data_tool_calls,
            )

        elif kind == "web_searches":
            self._web_searches += count
            if self._web_searches > self.budget.max_web_searches:
                self._web_search_breach = True
                logger.warning(
                    "Budget breach[%s] web_searches=%d > cap=%d",
                    self.persona,
                    self._web_searches,
                    self.budget.max_web_searches,
                )

        elif kind == "turns":
            self._turns += count

        else:
            raise ValueError(f"Unknown budget kind: {kind!r}")

    def remaining(self, kind: BudgetKind) -> int:
        """Return units remaining before the cap for *kind*.

        A negative value means the cap was already exceeded (only possible for
        web_searches and turns, which are counted not intercepted).
        """
        if kind == "data_tool_calls":
            return self.budget.max_data_tool_calls - self._data_tool_calls
        if kind == "web_searches":
            return self.budget.max_web_searches - self._web_searches
        if kind == "turns":
            return self.budget.max_turns - self._turns
        raise ValueError(f"Unknown budget kind: {kind!r}")

    def is_exhausted(self, kind: BudgetKind) -> bool:
        """Return True if *kind* is at or past its cap."""
        return self.remaining(kind) <= 0

    # ------------------------------------------------------------------
    # Post-run web-search breach check (for orchestrator + cost log)
    # ------------------------------------------------------------------

    @property
    def web_search_breach(self) -> bool:
        """True if web searches exceeded the cap (detected post-run)."""
        return self._web_search_breach

    # ------------------------------------------------------------------
    # Read-only counts (for NFR-#6 cost log)
    # ------------------------------------------------------------------

    @property
    def counts(self) -> dict[str, int]:
        """Return a snapshot of all counters for logging."""
        return {
            "data_tool_calls": self._data_tool_calls,
            "web_searches": self._web_searches,
            "turns": self._turns,
        }

    def summary(self) -> dict[str, object]:
        """Return a loggable summary including caps and breach flags."""
        return {
            "persona": self.persona,
            "data_tool_calls": {
                "used": self._data_tool_calls,
                "cap": self.budget.max_data_tool_calls,
                "exhausted": self.is_exhausted("data_tool_calls"),
            },
            "web_searches": {
                "used": self._web_searches,
                "cap": self.budget.max_web_searches,
                "breach": self._web_search_breach,
            },
            "turns": {
                "used": self._turns,
                "cap": self.budget.max_turns,
            },
        }
