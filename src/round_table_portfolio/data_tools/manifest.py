# manifest.py — Per-week tool-call manifest writer.
#
# Records every tool call's source / target / timestamp / success-or-failure /
# response-shape validation status to state/runs/<week_id>_toolcalls.json.
# Includes per-persona web-search count for NFR #6 monitoring.
#
# TDD Component 2 §Data Stored at This Step.

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.environ.get("STATE_DIR", "state"))
_RUNS_DIR = _STATE_DIR / "runs"
_LOCK = threading.Lock()


def _manifest_path(week_id: str) -> Path:
    return _RUNS_DIR / f"{week_id}_toolcalls.json"


def _load_manifest(week_id: str) -> dict:
    path = _manifest_path(week_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Manifest at %s is corrupt — starting fresh", path)
    return {"week_id": week_id, "calls": [], "web_searches": {}}


def _save_manifest(week_id: str, data: dict) -> None:
    path = _manifest_path(week_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_tool_call(
    *,
    week_id: str,
    persona: str,
    source: str,
    target: str,
    success: bool,
    validation_passed: bool,
    error: Optional[str] = None,
    is_fallback: bool = False,
) -> None:
    """Append one tool-call record to the week's manifest.

    Args:
        week_id:           ISO week label (e.g. '2026-W23').
        persona:           Persona slug or 'orchestrator'.
        source:            Data source name ('finnhub', 'edgar', 'fred', etc.).
        target:            Ticker symbol or series ID.
        success:           Whether the call returned data.
        validation_passed: Whether schema validation passed on the response.
        error:             Short error message if success=False.
        is_fallback:       True when this call is a fallback (e.g. yfinance
                           after Finnhub failure).
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "persona": persona,
        "source": source,
        "target": target,
        "success": success,
        "validation_passed": validation_passed,
        "is_fallback": is_fallback,
        "error": error,
    }
    with _LOCK:
        data = _load_manifest(week_id)
        data["calls"].append(entry)
        _save_manifest(week_id, data)


def record_web_search(*, week_id: str, persona: str) -> None:
    """Increment the web-search count for a persona in the week's manifest.

    The actual web search is a native subagent tool (not in this layer).
    The persona's research runner calls this to log the count for NFR #6.
    """
    with _LOCK:
        data = _load_manifest(week_id)
        searches = data.setdefault("web_searches", {})
        searches[persona] = searches.get(persona, 0) + 1
        _save_manifest(week_id, data)


def get_manifest(week_id: str) -> dict:
    """Read the manifest for a week (for reporting / testing)."""
    with _LOCK:
        return _load_manifest(week_id)
