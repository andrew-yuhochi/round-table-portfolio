# rss_tools.py — RSS headline tool using feedparser.
#
# Per TDD Component 2 §RSS row:
#   - Feed URLs live in config/rss_feeds.yaml (not hardcoded).
#   - Per-feed failure: log warning, continue with remaining feeds.
#   - ALL feeds failing: log + proceed (news is supplementary, not load-bearing).
#   - If feed.bozo is set: log warning, accept the parse (feedparser is permissive).
#   - NULL thresholds: title/link/published → 0%; summary → ≤20%.
#   - Malformed entries (missing required fields) are skipped with a warning.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import feedparser
import requests
import certifi
import yaml
from pydantic import ValidationError

_RSS_HEADERS = {"User-Agent": "round-table-portfolio/1.0 (research tool; contact andrewyu0517.ca@gmail.com)"}

from round_table_portfolio.data_tools.models import RSSEntry, RSSHeadlines

logger = logging.getLogger(__name__)

_DEFAULT_RSS_CONFIG = (
    Path(__file__).parent.parent.parent.parent.parent / "config" / "rss_feeds.yaml"
)


def _load_feeds_config(config_path: Optional[Path] = None) -> list[dict]:
    """Load rss_feeds.yaml and return list of {url, name} dicts."""
    path = config_path or _DEFAULT_RSS_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"RSS feeds config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    feeds = data.get("feeds", [])
    if not feeds:
        raise ValueError(f"No feeds defined in {path}")
    return feeds


def get_rss_headlines(
    *,
    config_path: Optional[Path] = None,
    max_entries_per_feed: int = 25,
) -> RSSHeadlines:
    """Fetch headlines from all configured RSS feeds.

    Per-feed failure is logged and skipped (news is supplementary).
    ALL feeds failing is logged and an empty-entries result is returned
    (does not abort the run).

    NULL thresholds: title/link/published → 0% (RFC-mandated; missing entries
    are rejected/skipped with a warning); summary → ≤20%.

    Args:
        config_path:          Override path to rss_feeds.yaml (for testing).
        max_entries_per_feed: Max headline entries to keep per feed.

    Returns:
        RSSHeadlines (may have empty entries if all feeds failed).
    """
    feed_configs = _load_feeds_config(config_path)
    all_entries: list[RSSEntry] = []
    feeds_attempted = 0
    feeds_succeeded = 0
    feeds_with_bozo = 0
    rejected_total = 0

    for cfg in feed_configs:
        url = cfg["url"]
        feed_name = cfg.get("name", url)
        feeds_attempted += 1

        try:
            # Pre-fetch with requests (User-Agent + certifi) then pass raw content
            # to feedparser.  feedparser.parse(url) doesn't set User-Agent and
            # doesn't use the certifi bundle — both cause silent failures on many feeds.
            resp = requests.get(url, headers=_RSS_HEADERS, verify=certifi.where(), timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)

            if getattr(feed, "bozo", False):
                feeds_with_bozo += 1
                logger.warning(
                    "RSS feed %s has bozo flag set (malformed XML): %s",
                    feed_name,
                    getattr(feed, "bozo_exception", "unknown"),
                )

            entries = feed.get("entries", [])[:max_entries_per_feed]
            accepted = 0
            rejected = 0

            for entry in entries:
                # Extract fields with safe fallbacks
                title = entry.get("title", "") or ""
                link = entry.get("link", "") or ""
                published = (
                    entry.get("published", "")
                    or entry.get("updated", "")
                    or ""
                )
                summary = entry.get("summary") or entry.get("description") or None

                try:
                    rss_entry = RSSEntry(
                        title=title,
                        link=link,
                        published=published,
                        summary=summary,
                        feed_source=feed_name,
                    )
                    all_entries.append(rss_entry)
                    accepted += 1
                except (ValidationError, ValueError) as exc:
                    rejected += 1
                    rejected_total += 1
                    logger.warning(
                        "RSS entry rejected from %s (validation): %s — "
                        "title=%r link=%r published=%r",
                        feed_name, exc, title[:60], link[:60], published[:60],
                    )

            feeds_succeeded += 1
            logger.info(
                "RSS %s: %d accepted, %d rejected from %d entries",
                feed_name, accepted, rejected, len(entries),
            )

        except Exception as exc:
            logger.warning("RSS feed %s failed: %s — skipping", feed_name, exc)

    if feeds_succeeded == 0:
        logger.warning(
            "All %d RSS feeds failed. Proceeding without news (supplementary source).",
            feeds_attempted,
        )

    if rejected_total > 0:
        logger.warning(
            "RSS: %d entries total rejected by schema validation across all feeds",
            rejected_total,
        )

    return RSSHeadlines(
        entries=all_entries,
        feeds_attempted=feeds_attempted,
        feeds_succeeded=feeds_succeeded,
        feeds_with_bozo=feeds_with_bozo,
    )
