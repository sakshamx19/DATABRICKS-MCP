"""Bounded waiting for long-running Databricks operations.

Tools never block indefinitely: an optional wait is clamped to
``DBX_MCP_MAX_WAIT_SECONDS`` and kept at least ``TOOL_TIMEOUT_MARGIN`` seconds
inside the per-call tool timeout, so a tool can still return a ``pending``
result (with ids to poll) instead of being cut off by the timeout.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from dbx_mcp.server.config import Settings

T = TypeVar("T")

TOOL_TIMEOUT_MARGIN = 15


def wait_cap(settings: Settings) -> int:
    """The longest any single tool call may wait on an operation."""
    return max(1, min(settings.max_wait_seconds, settings.tool_timeout_seconds - TOOL_TIMEOUT_MARGIN))


def clamp_wait(settings: Settings, requested: int | None, default: int = 0) -> tuple[int, str | None]:
    """Clamp a requested wait; returns (seconds, note-if-reduced)."""
    value = requested if requested and requested > 0 else default
    cap = wait_cap(settings)
    if value > cap:
        return cap, f"wait reduced to {cap}s (server limit)."
    return max(0, value), None


def poll(
    fetch: Callable[[], T],
    done: Callable[[T], bool],
    wait_seconds: float,
    *,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
) -> tuple[T, bool]:
    """Fetch once, then poll with backoff until ``done`` or the deadline. Returns (last, finished)."""
    deadline = time.monotonic() + max(0.0, wait_seconds)
    delay = initial_delay
    obj = fetch()
    while not done(obj):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.5, max_delay)
        obj = fetch()
    return obj, done(obj)


