"""Shared parameter types and helpers for tool implementations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Annotated, Any

from pydantic import Field

from dbx_mcp.models.common import PageInfo, ToolResponse
from dbx_mcp.server.context import AppContext, get_context
from dbx_mcp.utils.errors import ValidationFailed
from dbx_mcp.utils.pagination import clamp_page_size, paginate
from dbx_mcp.utils.serialization import to_jsonable

# --- reusable parameter annotations -------------------------------------------------------------

Confirm = Annotated[
    bool,
    Field(
        description=(
            "Set to true ONLY after the user has reviewed the plan returned by a previous call "
            "with status 'confirmation_required'. Required for destructive/security-sensitive actions."
        )
    ),
]
DryRun = Annotated[bool, Field(description="If true, validate and return the planned change without executing it.")]
PageSize = Annotated[int | None, Field(description="Max items to return (server caps this).", ge=1)]
PageToken = Annotated[str | None, Field(description="next_page_token from a previous response.")]
Spec = Annotated[
    dict[str, Any] | None,
    Field(
        description=(
            "Request body fields for create/update, using the Databricks REST API field names "
            "(snake_case). Unknown fields are rejected."
        )
    ),
]


def ctx() -> AppContext:
    return get_context()


def ok(
    summary: str,
    data: Any = None,
    *,
    status: str = "success",
    page: PageInfo | None = None,
    warnings: Iterable[str] | None = None,
    next_steps: Iterable[str] | None = None,
) -> ToolResponse:
    """Build a successful response. ``tool``/``action``/``safety`` are filled in by the wrapper."""
    return ToolResponse(
        status=status,  # type: ignore[arg-type]
        tool="",
        summary=summary,
        data=to_jsonable(data),
        page=page,
        warnings=[w for w in (warnings or []) if w],
        next_steps=list(next_steps or []),
    )


def require(value: Any, name: str, action: str | None = None) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        suffix = f" for action '{action}'" if action else ""
        raise ValidationFailed(f"Parameter '{name}' is required{suffix}")
    return value


def list_page(
    items: Iterable[Any],
    page_size: int | None,
    page_token: str | None,
    transform: Callable[[Any], Any] | None = None,
) -> tuple[list[Any], PageInfo]:
    settings = ctx().settings
    size = clamp_page_size(page_size, settings.default_page_size, settings.max_page_size)
    return paginate(items, page_size=size, page_token=page_token, transform=transform or to_jsonable)


def paged_response(
    noun: str,
    items: Iterable[Any],
    page_size: int | None,
    page_token: str | None,
    transform: Callable[[Any], Any] | None = None,
    *,
    warnings: Iterable[str] | None = None,
) -> ToolResponse:
    page, info = list_page(items, page_size, page_token, transform)
    more = " (more available - pass next_page_token)" if info.has_more else ""
    return ok(f"Returned {info.returned} {noun}{more}.", page, page=info, warnings=warnings)
