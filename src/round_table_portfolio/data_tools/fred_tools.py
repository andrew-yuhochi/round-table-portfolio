# fred_tools.py — FRED macro series tool.
#
# Fetches configured macro series (FEDFUNDS, yield curve, CPI, PCE, etc.)
# from the Federal Reserve Economic Data API.
#
# Uses requests + certifi directly instead of fredapi because fredapi relies
# on Python's urllib which does not pick up the certifi certificate bundle,
# causing SSL verification failures on macOS Python 3.14 (diagnosed 2026-06-01).
#
# Per TDD Component 2 §FRED row:
#   - Series list lives in config/fred_series.yaml (not hardcoded).
#   - value may be NaN for unreleased observations — valid data, not NULL bug.
#   - NULL threshold: ≤20% across the series list at the most-recent observation.
#   - If FRED is unavailable: run aborts (no fallback at PoC).

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import certifi
import requests
import yaml
from dotenv import load_dotenv

from round_table_portfolio.data_tools.models import (
    FREDSeries,
    FREDSeriesObservation,
    FREDMacroSnapshot,
)
from round_table_portfolio.data_tools.rate_limiter import FRED_LIMITER

load_dotenv()

logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred"
_DEFAULT_FRED_CONFIG = (
    Path(__file__).parent.parent.parent.parent.parent / "config" / "fred_series.yaml"
)


def _load_series_config(config_path: Optional[Path] = None) -> list[dict]:
    """Load fred_series.yaml and return list of {id, description} dicts."""
    path = config_path or _DEFAULT_FRED_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"FRED series config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    series = data.get("series", [])
    if not series:
        raise ValueError(f"No series defined in {path}")
    return series


def _fred_get(endpoint: str, params: dict, *, _retries: int = 3) -> dict:
    """Make a GET request to the FRED API using requests + certifi.

    Retries up to *_retries* times on 429 (rate-limit) with exponential
    back-off.  Raises RuntimeError on persistent failure or other HTTP errors.
    """
    import time as _time

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Add it to .env (see .env.example)."
        )
    params = {**params, "api_key": api_key, "file_type": "json"}
    delay = 2.0
    for attempt in range(1, _retries + 1):
        try:
            resp = requests.get(
                f"{_FRED_BASE}/{endpoint}",
                params=params,
                verify=certifi.where(),
                timeout=15,
            )
            if resp.status_code == 429:
                if attempt < _retries:
                    logger.warning(
                        "FRED 429 rate-limit on %s (attempt %d/%d) — sleeping %.1fs",
                        endpoint, attempt, _retries, delay,
                    )
                    _time.sleep(delay)
                    delay *= 2
                    continue
                raise requests.HTTPError(response=resp)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"FRED API HTTP error on {endpoint}: "
                f"{exc.response.status_code} {exc.response.text[:200]}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"FRED API network error on {endpoint}: {exc}") from exc
    raise RuntimeError(f"FRED API: exhausted {_retries} retries on {endpoint}")


def get_macro_series(
    week_id: str,
    *,
    config_path: Optional[Path] = None,
    observation_limit: int = 12,
) -> FREDMacroSnapshot:
    """Fetch all configured FRED macro series for the current weekly run.

    Fetches the most recent *observation_limit* observations per series.
    NaN / missing values (not-yet-released monthly data) are retained as None —
    valid data per TDD; not a NULL bug.

    Args:
        week_id:           ISO week label (e.g. '2026-W23') for the snapshot.
        config_path:       Override path to fred_series.yaml (for testing).
        observation_limit: How many recent observations to return per series.

    Returns:
        FREDMacroSnapshot with all configured series.

    Raises:
        EnvironmentError: FRED_API_KEY not set.
        FileNotFoundError: fred_series.yaml missing.
        RuntimeError: FRED API call failed (run aborts — no fallback at PoC).
    """
    series_configs = _load_series_config(config_path)
    results: list[FREDSeries] = []

    for cfg in series_configs:
        series_id = cfg["id"]
        description = cfg.get("description", series_id)

        FRED_LIMITER.acquire()
        try:
            data = _fred_get(
                "series/observations",
                {
                    "series_id": series_id,
                    "limit": observation_limit,
                    "sort_order": "desc",   # newest first
                },
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"FRED get_series({series_id!r}) failed — run aborts. Error: {exc}"
            ) from exc

        raw_obs = data.get("observations", [])
        # FRED returns newest-first; reverse to chronological order
        raw_obs = list(reversed(raw_obs))

        observations: list[FREDSeriesObservation] = []
        for obs in raw_obs:
            raw_val = obs.get("value", ".")
            # FRED uses "." for missing/unreleased values
            if raw_val == "." or raw_val is None:
                obs_value: Optional[float] = None
            else:
                try:
                    parsed = float(raw_val)
                    obs_value = None if math.isnan(parsed) else parsed
                except (ValueError, TypeError):
                    obs_value = None

            observations.append(
                FREDSeriesObservation(
                    series_id=series_id,
                    date=obs.get("date", ""),
                    value=obs_value,
                )
            )

        # Latest non-None value
        latest_val: Optional[float] = None
        latest_date: Optional[str] = None
        for obs in reversed(observations):
            if obs.value is not None:
                latest_val = obs.value
                latest_date = obs.date
                break

        results.append(
            FREDSeries(
                series_id=series_id,
                description=description,
                observations=observations,
                latest_value=latest_val,
                latest_date=latest_date,
            )
        )
        logger.info(
            "FRED %s: %d obs, latest=%s on %s",
            series_id, len(observations), latest_val, latest_date,
        )

    return FREDMacroSnapshot(week_id=week_id, series=results)
