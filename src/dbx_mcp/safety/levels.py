"""Safety classification for tool actions."""

from __future__ import annotations

from enum import Enum


class SafetyLevel(str, Enum):
    """How dangerous an operation is.

    An action may carry several levels (e.g. revoking a grant is both
    DESTRUCTIVE and SECURITY_SENSITIVE).
    """

    READ_ONLY = "READ_ONLY"
    """Inspects state; never changes anything."""

    WRITE = "WRITE"
    """Creates or modifies resources in a recoverable way."""

    DESTRUCTIVE = "DESTRUCTIVE"
    """Deletes, drops, terminates, revokes or otherwise loses state/data."""

    EXECUTION = "EXECUTION"
    """Runs user-supplied code/queries or triggers compute that costs money."""

    SECURITY_SENSITIVE = "SECURITY_SENSITIVE"
    """Changes who can access what, shares data externally, or mints credentials."""


READ = frozenset({SafetyLevel.READ_ONLY})
WRITE = frozenset({SafetyLevel.WRITE})
DESTRUCTIVE = frozenset({SafetyLevel.DESTRUCTIVE})
EXECUTION = frozenset({SafetyLevel.EXECUTION})
SECURITY_SENSITIVE = frozenset({SafetyLevel.SECURITY_SENSITIVE})

# Common combinations
WRITE_SECURITY = WRITE | SECURITY_SENSITIVE
DESTRUCTIVE_SECURITY = DESTRUCTIVE | SECURITY_SENSITIVE
READ_SECURITY = READ | SECURITY_SENSITIVE

CONFIRMATION_LEVELS = frozenset({SafetyLevel.DESTRUCTIVE, SafetyLevel.SECURITY_SENSITIVE})


_MUTATING = frozenset({SafetyLevel.WRITE, SafetyLevel.DESTRUCTIVE})


def is_read_action(levels: frozenset[SafetyLevel] | set[SafetyLevel]) -> bool:
    """True if the action only reads state.

    READ_ONLY may be combined with EXECUTION (a SELECT runs on a warehouse) or
    SECURITY_SENSITIVE (reading grants) and is still a read.
    """
    levels = set(levels)
    return SafetyLevel.READ_ONLY in levels and not (levels & _MUTATING)


def needs_confirmation(
    levels: frozenset[SafetyLevel] | set[SafetyLevel], *, confirm_execution: bool = False
) -> bool:
    """Whether an action must be explicitly confirmed (two-step) before running.

    Reads never need confirmation. Destructive or security-sensitive changes always
    do; non-read EXECUTION does when ``confirm_execution`` is enabled.
    """
    levels = set(levels)
    if is_read_action(levels):
        return False
    if levels & CONFIRMATION_LEVELS:
        return True
    return confirm_execution and SafetyLevel.EXECUTION in levels
