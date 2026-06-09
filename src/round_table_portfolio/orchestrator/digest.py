"""Component 28 — whats_new_digest_builder.

Produces a short text digest for ONE persona summarising which of that persona's
OWN prior counterfactual calls have resolved since its last run, and how they
did vs SPY.

Design contracts:
  - **Own-calls-only**: the digest is strictly scoped to the persona's OWN past
    calls.  No peer persona's calls ever appear here.  Breaching this would
    contaminate the commit-before-reveal independence that is Critical
    Component #1.
  - **Deterministic**: identical resolved-rows input → byte-identical output.
    Tiebreak on equal |alpha|: ticker ascending.
  - **Empty-state safe**: when no calls have resolved yet (common on early
    weeks before any look-forward window closes), emits the canonical
    "No prior calls have resolved yet." string — never crashes.
  - **No side effects**: pure function; no DB writes, no file writes.  The
    caller (Component 26 / memory_reader) supplies all inputs; the caller
    (Component 18b / memory_writeback and Component 27 / briefing_builder)
    consumes the returned string.

Input types:
  - ``resolved_rows``: sequence of ``ResolvedRow`` — one row per ticker that
    resolved for this persona.  These come from the Component 26 query against
    ``weekly_returns`` joined through ``portfolios``/``holdings``.
  - ``past_call_entries``: the persona's Past Calls Log entries as parsed by
    ``parse_memory_file`` — list of ``(week_id, body_text)`` pairs.
  - ``config``: a ``DigestConfig`` loaded from ``config/thresholds.yaml``.

The builder joins resolved_rows to past_call_entries on (persona, ticker,
call_week) to attribute each resolved alpha to the original call's
action/confidence.  The join is ticker + week based: for each resolved row
(ticker resolved in ``as_of_week_id``, originating from ``call_week_id``), we
look up the past-calls body for ``call_week_id`` and extract the ticker's
``action`` + ``confidence`` fields.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(os.environ.get("THRESHOLDS_CONFIG", "config/thresholds.yaml"))

_DEFAULT_DIGEST_MAX_ITEMS: int = 5
_DEFAULT_OWN_MISSES_IN_DIGEST: bool = True


@dataclass(frozen=True)
class DigestConfig:
    """Typed view of the digest section of thresholds.yaml."""

    digest_max_items: int
    own_misses_in_digest: bool


def load_digest_config(
    config_path: Optional[Path] = None,
) -> DigestConfig:
    """Read thresholds.yaml and return a DigestConfig.

    Falls back to built-in defaults for any missing key so early-week runs
    before the config key is present work without error.
    """
    path = config_path or _CONFIG_PATH
    raw: dict = {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logger.warning(
            "thresholds.yaml not found at %s — using built-in defaults.", path
        )

    return DigestConfig(
        digest_max_items=int(
            raw.get("digest_max_items", _DEFAULT_DIGEST_MAX_ITEMS)
        ),
        own_misses_in_digest=bool(
            raw.get("own_misses_in_digest", _DEFAULT_OWN_MISSES_IN_DIGEST)
        ),
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedRow:
    """One resolved return record for a single ticker in a persona's portfolio.

    Sourced from ``weekly_returns`` joined through ``portfolios``/``holdings``
    (Component 26's query output).  The ``call_week_id`` field is the week the
    original call was made (the portfolio's ``week_id``); ``as_of_week_id`` is
    the week the return was marked to market.
    """

    persona: str          # persona slug (e.g. "value")
    ticker: str           # equity ticker (e.g. "NVDA")
    call_week_id: str     # week the call was originally made (e.g. "2026-W24")
    as_of_week_id: str    # week the return resolved (e.g. "2026-W25")
    alpha: float          # alpha vs SPY (annualised continuous; can be negative)
    action: str           # action from holdings at call time (add/hold/reduce/exit)


@dataclass(frozen=True)
class _DigestItem:
    """Internal ranked item before text rendering."""

    ticker: str
    action: str
    confidence: Optional[int]  # None when not found in past-calls log
    call_week: str
    alpha: float


# ---------------------------------------------------------------------------
# Past-calls body parser helpers
# ---------------------------------------------------------------------------

# Regex to extract a single ticker line from the past-calls body.
# Format written by _build_past_calls_entry in memory.py:
#   "  TICKER: action confidence=N weight=W.WWW"
_STANCE_LINE_RE = re.compile(
    r"^\s{2}(?P<ticker>[A-Z0-9.\-]+):\s+"
    r"(?P<action>\w+)\s+"
    r"confidence=(?P<confidence>\d+)",
    re.MULTILINE,
)


def _parse_confidence_from_body(body: str, ticker: str) -> Optional[int]:
    """Extract confidence for *ticker* from a past-calls body string.

    The body is the text stored under a ``### Entry <week_id>`` in the
    Past Calls Log section.  Returns None if the ticker is not found.
    """
    for m in _STANCE_LINE_RE.finditer(body):
        if m.group("ticker") == ticker:
            return int(m.group("confidence"))
    return None


def _build_lookup(
    past_call_entries: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Build a ``(week_id, ticker) → body`` lookup from past-calls entries.

    Args:
        past_call_entries: list of ``(week_id, body_text)`` from a persona's
            Past Calls Log section (as returned by ``parse_memory_file``).

    Returns:
        Dict mapping ``(week_id, ticker)`` to the full body for that week.
        The same body is stored once per (week_id, ticker) pair so the caller
        can extract confidence without re-scanning.
    """
    lookup: dict[tuple[str, str], str] = {}
    for week_id, body in past_call_entries:
        for m in _STANCE_LINE_RE.finditer(body):
            key = (week_id, m.group("ticker"))
            lookup[key] = body
    return lookup


# ---------------------------------------------------------------------------
# Digest line renderer
# ---------------------------------------------------------------------------

def _render_item(item: _DigestItem) -> str:
    """Render one resolved call to a single digest line.

    Format: "<TICKER> (you said <action> conf=<N> in <week>) → alpha <value> vs SPY"
    If confidence is unavailable (call not found in past-calls log), the
    conf= field is omitted.
    """
    alpha_str = f"{item.alpha:+.4f}"
    if item.confidence is not None:
        call_detail = (
            f"you said {item.action} conf={item.confidence} in {item.call_week}"
        )
    else:
        call_detail = f"you said {item.action} in {item.call_week}"
    return f"{item.ticker} ({call_detail}) → alpha {alpha_str} vs SPY"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_EMPTY_STATE_MSG = "No prior calls have resolved yet."


def build_whats_new_digest(
    persona: str,
    resolved_rows: Sequence[ResolvedRow],
    past_call_entries: Sequence[tuple[str, str]],
    config: DigestConfig,
) -> str:
    """Build the "what's new" digest for one persona.

    Args:
        persona:           The persona slug being digested.  Used only for
                           own-calls enforcement: any row whose ``persona``
                           field does not match is silently dropped.
        resolved_rows:     Sequence of ResolvedRow from Component 26's query.
                           Should already be pre-filtered to this persona by
                           the caller, but this function enforces it
                           defensively to guarantee the own-calls invariant.
        past_call_entries: ``(week_id, body_text)`` pairs from this persona's
                           Past Calls Log section.  Used to look up the
                           original action + confidence for each resolved call.
        config:            DigestConfig loaded from thresholds.yaml.

    Returns:
        A multi-line string: a header + one ranked line per resolved call
        (capped at ``config.digest_max_items``), or the canonical empty-state
        message if nothing resolved.
    """
    # --- Own-calls enforcement (Critical Component #1 invariant) -----------
    own_rows = [r for r in resolved_rows if r.persona == persona]

    if not own_rows:
        return _EMPTY_STATE_MSG

    # --- Build (week_id, ticker) → body lookup from past-calls entries -----
    body_lookup = _build_lookup(past_call_entries)

    # --- Assemble ranked items ---------------------------------------------
    items: list[_DigestItem] = []
    for row in own_rows:
        # Skip own_misses when config flag is off.
        if not config.own_misses_in_digest and row.alpha < 0:
            continue
        confidence = _parse_confidence_from_body(
            body_lookup.get((row.call_week_id, row.ticker), ""),
            row.ticker,
        )
        items.append(
            _DigestItem(
                ticker=row.ticker,
                action=row.action,
                confidence=confidence,
                call_week=row.call_week_id,
                alpha=row.alpha,
            )
        )

    if not items:
        return _EMPTY_STATE_MSG

    # --- Rank: |alpha| descending; tiebreak: ticker ascending (determinism) -
    ranked = sorted(items, key=lambda x: (-abs(x.alpha), x.ticker))

    # --- Cap ---------------------------------------------------------------
    capped = ranked[: config.digest_max_items]

    # --- Render ------------------------------------------------------------
    lines = ["Since your last run, these of your calls resolved:"]
    for item in capped:
        lines.append(f"  {_render_item(item)}")

    return "\n".join(lines)
