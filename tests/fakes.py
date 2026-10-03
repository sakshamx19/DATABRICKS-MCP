"""Fake credentials for tests.

Built at runtime so the source never contains a token-shaped literal: secret scanners (e.g.
GitHub push protection) flag anything matching the Databricks PAT format, even obvious fakes.
"""

from __future__ import annotations

_PREFIX = "da" + "pi"


def fake_pat(fill: str = "0") -> str:
    """A string in Databricks PAT format (prefix + 32 hex chars) that is not a real token."""
    return _PREFIX + (fill * 32)[:32]


FAKE_PAT = _PREFIX + "0123456789abcdef" * 2
