"""Typed response models for SQL tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from dbx_mcp.models.common import ToolResponse


class SqlColumn(BaseModel):
    name: str
    type: str | None = None
    position: int | None = None


class SqlResultSet(BaseModel):
    """Rows returned by the statement (the *query result*)."""

    columns: list[SqlColumn]
    row_format: Literal["arrays", "objects"] = "arrays"
    rows: list[Any] = Field(description="Arrays aligned with `columns`, or objects keyed by column name.")
    row_count: int = Field(description="Rows included in this response.")
    total_row_count: int | None = Field(default=None, description="Total rows produced, when reported by Databricks.")
    truncated: bool = Field(description="True if more rows exist than were returned (raise max_rows or add LIMIT).")


class WarehouseSelection(BaseModel):
    warehouse_id: str
    name: str | None = None
    state: str | None = None
    reason: str
    candidates_considered: list[dict[str, Any]] = Field(default_factory=list)


class SqlExecutionMetadata(BaseModel):
    """Facts about *how* the statement ran (distinct from its result)."""

    statement_id: str | None = None
    state: str | None = None
    statement_kind: str = Field(description="read | write | destructive | security | unknown")
    classification_reasons: list[str] = Field(default_factory=list)
    warehouse: WarehouseSelection | None = None
    duration_ms: float | None = None
    error: dict[str, Any] | None = None
    sql_state: str | None = None


class SqlExecution(BaseModel):
    result: SqlResultSet | None = None
    execution: SqlExecutionMetadata


class SqlExecutionResponse(ToolResponse):
    data: SqlExecution | None = None


class StatementOutcome(BaseModel):
    index: int
    statement: str
    statement_kind: str
    status: Literal["succeeded", "failed", "pending", "skipped"]
    result: SqlResultSet | None = None
    statement_id: str | None = None
    error: dict[str, Any] | None = None


class SqlMultiExecution(BaseModel):
    warehouse: WarehouseSelection | None = None
    succeeded: int
    failed: int
    pending: int
    skipped: int
    statements: list[StatementOutcome]
    note: str = (
        "Statements run sequentially and independently: there is no transaction, so earlier "
        "successful statements are NOT rolled back when a later one fails."
    )


class SqlMultiResponse(ToolResponse):
    data: SqlMultiExecution | None = None
