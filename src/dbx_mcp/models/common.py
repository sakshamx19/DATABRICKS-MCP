"""Common, strongly-typed response models shared by all tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dbx_mcp.safety.levels import SafetyLevel

ResponseStatus = Literal[
    "success",
    "pending",
    "dry_run",
    "confirmation_required",
    "partial_failure",
    "failed",
]


class PageInfo(BaseModel):
    """Pagination metadata. Pass ``next_page_token`` back as ``page_token`` to continue."""

    page_size: int
    returned: int
    offset: int = 0
    has_more: bool
    next_page_token: str | None = None


class OperationPlan(BaseModel):
    """A description of a change that will be (or would have been) made."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    action: str | None = None
    safety: list[SafetyLevel]
    target: dict[str, Any] = Field(default_factory=dict, description="Identifiers of the affected resource(s).")
    description: str = Field(description="Human-readable summary of the change.")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional facts about the change/resource.")
    warnings: list[str] = Field(default_factory=list)
    reversible: bool | None = Field(default=None, description="Whether the change can be undone.")


class ToolResponse(BaseModel):
    """Envelope returned by every tool.

    * ``summary`` is a short human-readable sentence.
    * ``data`` holds the machine-readable result.
    * ``status`` distinguishes completed work from previews (``dry_run``), two-step
      confirmations (``confirmation_required``) and still-running operations (``pending``).
    """

    status: ResponseStatus = "success"
    tool: str
    action: str | None = None
    summary: str
    safety: list[SafetyLevel] = Field(default_factory=list)
    data: Any = None
    page: PageInfo | None = None
    plan: OperationPlan | None = None
    warnings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list, description="Suggested follow-up calls.")
    request_id: str | None = None

    # Keys exempted from secret redaction. Set ONLY by tools whose explicit, confirmed
    # purpose is returning a credential (e.g. Lakebase credential with reveal=true).
    _unredacted_keys: set[str] = PrivateAttr(default_factory=set)
