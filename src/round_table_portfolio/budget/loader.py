# budget/loader.py — Load per-persona turn/tool-call/web-search budgets from config.
#
# Reads config/persona_budgets.yaml and returns a typed dict keyed by persona slug.
# Unlisted personas receive the `defaults` values — never a KeyError.
#
# TDD Component 2 bounding + TASK-M1-005.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_CONFIG_PATH = Path(os.environ.get("BUDGET_CONFIG", "config/persona_budgets.yaml"))


@dataclass(frozen=True)
class PersonaBudget:
    max_turns: int
    max_web_searches: int
    max_data_tool_calls: int


def load_budgets(config_path: Optional[Path] = None) -> dict[str, PersonaBudget]:
    """Read persona_budgets.yaml and return a mapping of slug → PersonaBudget.

    Personas not listed in the ``personas`` section inherit the ``defaults``
    block so callers always get a valid budget regardless of slug.

    Args:
        config_path: Override for testing; defaults to BUDGET_CONFIG env var or
                     ``config/persona_budgets.yaml`` relative to the working dir.

    Returns:
        A dict mapping every explicitly-listed persona slug to its budget, plus
        a ``"__defaults__"`` key carrying the fallback budget used for unlisted
        personas.  Callers should use :func:`get_budget` rather than accessing
        the dict directly.
    """
    path = config_path or _CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    defaults_raw = raw.get("defaults", {})
    defaults = PersonaBudget(
        max_turns=int(defaults_raw["max_turns"]),
        max_web_searches=int(defaults_raw["max_web_searches"]),
        max_data_tool_calls=int(defaults_raw["max_data_tool_calls"]),
    )

    budgets: dict[str, PersonaBudget] = {"__defaults__": defaults}
    for slug, overrides in (raw.get("personas") or {}).items():
        budgets[slug] = PersonaBudget(
            max_turns=int(overrides.get("max_turns", defaults_raw["max_turns"])),
            max_web_searches=int(
                overrides.get("max_web_searches", defaults_raw["max_web_searches"])
            ),
            max_data_tool_calls=int(
                overrides.get("max_data_tool_calls", defaults_raw["max_data_tool_calls"])
            ),
        )

    return budgets


def get_budget(budgets: dict[str, PersonaBudget], persona: str) -> PersonaBudget:
    """Return the budget for *persona*, falling back to defaults for unknown slugs."""
    return budgets.get(persona, budgets["__defaults__"])
