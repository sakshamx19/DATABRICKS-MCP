"""Uniform pagination over SDK list iterators.

The Databricks SDK exposes list endpoints as lazy iterators that follow the
service's own page tokens internally. To give MCP clients a uniform contract we
wrap them with an opaque cursor (``page_token``) that encodes how many items have
already been returned. Requesting page N re-reads the iterator up to that offset,
which is simple and correct; for very large collections prefer server-side
filters (name/prefix parameters) to keep this cheap.
"""

from __future__ import annotations

import base64
import itertools
import json
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from dbx_mcp.models.common import PageInfo
from dbx_mcp.utils.errors import ValidationFailed

T = TypeVar("T")


def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"o": offset}).encode()).decode().rstrip("=")


def decode_cursor(token: str | None) -> int:
    if not token:
        return 0
    try:
        padded = token + "=" * (-len(token) % 4)
        offset = int(json.loads(base64.urlsafe_b64decode(padded.encode()))["o"])
    except Exception as exc:
        raise ValidationFailed("Invalid page_token; pass the next_page_token from a previous response.") from exc
    if offset < 0:
        raise ValidationFailed("Invalid page_token.")
    return offset


def clamp_page_size(page_size: int | None, default: int, maximum: int) -> int:
    if page_size is None:
        return default
    if page_size < 1:
        raise ValidationFailed("page_size must be >= 1")
    return min(page_size, maximum)


def paginate(
    items: Iterable[T],
    *,
    page_size: int,
    page_token: str | None,
    transform: Callable[[T], Any] | None = None,
) -> tuple[list[Any], PageInfo]:
    """Return one page of ``items`` plus pagination metadata."""
    offset = decode_cursor(page_token)
    window = list(itertools.islice(iter(items), offset, offset + page_size + 1))
    has_more = len(window) > page_size
    page = window[:page_size]
    if transform is not None:
        page = [transform(item) for item in page]
    info = PageInfo(
        page_size=page_size,
        returned=len(page),
        offset=offset,
        has_more=has_more,
        next_page_token=encode_cursor(offset + page_size) if has_more else None,
    )
    return page, info
