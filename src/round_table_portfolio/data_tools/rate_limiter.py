# rate_limiter.py — Central shared rate limiter for all data tool calls.
#
# A SINGLE module-level limiter instance enforces per-source quotas across
# ALL persona tool calls in the same Python process.  Persona calls are
# sequential within a run (the orchestrator dispatches personas one at a time),
# so this provides the correct cross-persona enforcement.
#
# Per TDD Component 2 §Rate handling:
#   Finnhub: 60 calls/min, 30 calls/sec
#   EDGAR:   10 req/sec  (enforced internally by edgartools; we still sleep)
#   FRED:    generous (no documented limit; we impose 2/sec as courtesy)
#   RSS:     no limit
#   Alpaca:  200 req/min (free data plan); lazy per-ticker cache means we
#            rarely approach this — limiter is a safety net only
#
# Implementation: token-bucket via a simple per-source deque of timestamps.
# stdlib-only — no extra dependency.

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)


class _SourceLimiter:
    """Token-bucket rate limiter for one data source.

    Tracks call timestamps and blocks (sleep) when the bucket is full.
    Thread-safe via a per-instance lock.
    """

    def __init__(
        self,
        name: str,
        calls_per_minute: int | None = None,
        calls_per_second: int | None = None,
    ) -> None:
        self.name = name
        self._per_min = calls_per_minute
        self._per_sec = calls_per_second
        self._lock = threading.Lock()
        # Sliding windows: deque of monotonic call timestamps
        self._min_window: Deque[float] = deque()
        self._sec_window: Deque[float] = deque()

    def acquire(self) -> None:
        """Block until a call slot is available, then claim it."""
        with self._lock:
            now = time.monotonic()

            # --- per-second enforcement ---
            if self._per_sec is not None:
                # Evict timestamps older than 1 second
                while self._sec_window and now - self._sec_window[0] >= 1.0:
                    self._sec_window.popleft()
                if len(self._sec_window) >= self._per_sec:
                    sleep_s = 1.0 - (now - self._sec_window[0])
                    if sleep_s > 0:
                        logger.debug(
                            "%s rate limiter: per-second bucket full (%d/%d), sleeping %.3fs",
                            self.name,
                            len(self._sec_window),
                            self._per_sec,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        now = time.monotonic()
                        # Re-evict after sleeping
                        while self._sec_window and now - self._sec_window[0] >= 1.0:
                            self._sec_window.popleft()
                self._sec_window.append(now)

            # --- per-minute enforcement ---
            if self._per_min is not None:
                while self._min_window and now - self._min_window[0] >= 60.0:
                    self._min_window.popleft()
                if len(self._min_window) >= self._per_min:
                    sleep_s = 60.0 - (now - self._min_window[0])
                    if sleep_s > 0:
                        logger.warning(
                            "%s rate limiter: per-minute bucket full (%d/%d), sleeping %.1fs",
                            self.name,
                            len(self._min_window),
                            self._per_min,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        now = time.monotonic()
                        while self._min_window and now - self._min_window[0] >= 60.0:
                            self._min_window.popleft()
                self._min_window.append(now)

    def call_count_last_minute(self) -> int:
        with self._lock:
            now = time.monotonic()
            while self._min_window and now - self._min_window[0] >= 60.0:
                self._min_window.popleft()
            return len(self._min_window)


# ---------------------------------------------------------------------------
# Module-level singleton limiters — shared across all tool calls in process
# ---------------------------------------------------------------------------

FINNHUB_LIMITER = _SourceLimiter(
    name="Finnhub",
    calls_per_minute=60,
    calls_per_second=30,
)

EDGAR_LIMITER = _SourceLimiter(
    name="EDGAR",
    calls_per_second=8,   # stay safely under SEC's 10/sec policy
)

FRED_LIMITER = _SourceLimiter(
    name="FRED",
    calls_per_second=2,   # courtesy cap; FRED has no documented hard limit
)

ALPACA_LIMITER = _SourceLimiter(
    name="Alpaca",
    calls_per_minute=200,  # free data plan hard cap; lazy cache means rarely hit
)

# RSS has no rate limit — no limiter needed


def get_limiter(source: str) -> _SourceLimiter | None:
    """Return the shared limiter for a given source name.

    Returns None for sources with no rate limit (RSS, technicals).
    """
    mapping = {
        "finnhub": FINNHUB_LIMITER,
        "edgar": EDGAR_LIMITER,
        "fred": FRED_LIMITER,
        "alpaca": ALPACA_LIMITER,
    }
    return mapping.get(source.lower())
