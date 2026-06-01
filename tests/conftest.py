# conftest.py — pytest configuration for round-table-portfolio tests.
#
# SKIP_LIVE=1 skips all tests marked @pytest.mark.live (integration tests
# that make real API calls).  All tests in tests/unit/ run unconditionally.

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: marks tests that make live API calls — skipped when SKIP_LIVE=1",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if os.environ.get("SKIP_LIVE") == "1":
        skip_live = pytest.mark.skip(reason="SKIP_LIVE=1 — live API tests skipped")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
