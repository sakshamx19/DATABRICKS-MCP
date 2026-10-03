"""SQL execution via the Databricks SQL Statement Execution API (``/api/2.0/sql/statements``).

Shared by ``execute_sql`` and by tools that are implemented with SQL DDL because
that is the documented interface (tags, row filters/column masks, metric views).

Behaviour:
* Results are fetched INLINE as JSON arrays, following result chunks until the
  configured row cap is reached (``truncated`` is reported).
* The call waits at most ``wait_timeout_seconds`` (5-50s, an API limit); if the
  statement is still running it is NOT cancelled - the statement id is returned
  so the caller can poll with ``manage_sql_statement``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from databricks.sdk.service import sql as sql_svc

from dbx_mcp.server.context import AppContext
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, ValidationFailed

_STATE_RANK = {"RUNNING": 0, "STARTING": 1, "STOPPED": 2, "STOPPING": 3}


@dataclass
class WarehouseChoice:
    warehouse_id: str
    name: str | None
    state: str | None
    reason: str
    warnings: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "warehouse_id": self.warehouse_id,
            "name": self.name,
            "state": self.state,
            "reason": self.reason,
            "candidates_considered": self.candidates,
        }


def _value(enum_or_none: Any) -> str | None:
    return getattr(enum_or_none, "value", enum_or_none)


def rank_warehouses(warehouses: list[sql_svc.EndpointInfo]) -> list[sql_svc.EndpointInfo]:
    """Order: running first, then starting, then stopped; within a state prefer serverless, then PRO."""

    def key(w: sql_svc.EndpointInfo) -> tuple[int, int, int, str]:
        state = _value(w.state) or ""
        return (
            _STATE_RANK.get(state, 9),
            0 if w.enable_serverless_compute else 1,
            0 if _value(w.warehouse_type) == "PRO" else 1,
            (w.name or "").lower(),
        )

    usable = [w for w in warehouses if _value(w.state) not in {"DELETED", "DELETING"}]
    return sorted(usable, key=key)


def select_warehouse(c: AppContext, requested: str | None = None) -> WarehouseChoice:
    """Pick the SQL warehouse to use, explaining why (never silently)."""
    if requested:
        return WarehouseChoice(requested, None, None, "explicitly requested by the caller")
    if c.default_warehouse_id:
        source = (
            "connection default (X-Databricks-Warehouse-Id header)"
            if c.clients.request_mode
            else "configured default (DBX_MCP_DEFAULT_WAREHOUSE_ID)"
        )
        return WarehouseChoice(c.default_warehouse_id, None, None, source)
    if c.settings.warehouse_selection == "configured_only":
        raise DbxToolError(
            ErrorCategory.CONFIGURATION,
            "No warehouse_id given and no default configured (DBX_MCP_WAREHOUSE_SELECTION=configured_only).",
            hint="Pass warehouse_id or set DBX_MCP_DEFAULT_WAREHOUSE_ID.",
        )
    ranked = rank_warehouses(list(c.w.warehouses.list()))
    if not ranked:
        raise DbxToolError(
            ErrorCategory.NOT_FOUND,
            "No SQL warehouses are visible to the current user.",
            hint="Create one with manage_sql_warehouse or ask an admin for CAN_USE on a warehouse.",
        )
    best = ranked[0]
    state = _value(best.state)
    candidates = [
        {"id": w.id, "name": w.name, "state": _value(w.state), "serverless": bool(w.enable_serverless_compute)}
        for w in ranked[:10]
    ]
    warnings = []
    if state != "RUNNING":
        warnings.append(
            f"No running warehouse found; using {best.name!r} ({state}). Running a statement will start it, "
            "which takes time and incurs cost."
        )
    reason = (
        "automatic selection (DBX_MCP_WAREHOUSE_SELECTION=prefer_running): ranked running > starting > stopped, "
        "then serverless > pro > classic, then by name"
    )
    return WarehouseChoice(best.id or "", best.name, state, reason, warnings, candidates)


@dataclass
class SqlResult:
    statement_id: str | None
    state: str | None
    columns: list[dict[str, Any]] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    total_row_count: int | None = None
    truncated: bool = False
    error: dict[str, Any] | None = None
    sql_state: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.state == "SUCCEEDED"

    @property
    def pending(self) -> bool:
        return self.state in {"PENDING", "RUNNING"}

    def records(self) -> list[dict[str, Any]]:
        names = [col["name"] for col in self.columns]
        return [dict(zip(names, row, strict=False)) for row in self.rows]


def _parameters(params: dict[str, Any] | None) -> list[sql_svc.StatementParameterListItem] | None:
    if not params:
        return None
    items = []
    for name, value in params.items():
        if not name.isidentifier():
            raise ValidationFailed(f"Invalid SQL parameter name {name!r}; use :name markers with identifier names")
        items.append(sql_svc.StatementParameterListItem(name=name, value=None if value is None else str(value)))
    return items


def collect_result(c: AppContext, response: sql_svc.StatementResponse, max_rows: int) -> SqlResult:
    status = response.status
    state = _value(status.state) if status else None
    result = SqlResult(statement_id=response.statement_id, state=state)
    if status and status.error:
        result.error = {"error_code": _value(status.error.error_code), "message": status.error.message}
        result.sql_state = status.sql_state
    manifest = response.manifest
    if manifest and manifest.schema and manifest.schema.columns:
        result.columns = [
            {"name": col.name, "type": col.type_text or _value(col.type_name), "position": col.position}
            for col in manifest.schema.columns
        ]
    if manifest:
        result.total_row_count = manifest.total_row_count
        result.truncated = bool(manifest.truncated)
    chunk = response.result
    while chunk is not None:
        for row in chunk.data_array or []:
            if len(result.rows) >= max_rows:
                result.truncated = True
                break
            result.rows.append(row)
        if len(result.rows) >= max_rows or chunk.next_chunk_index is None or not response.statement_id:
            if chunk.next_chunk_index is not None:
                result.truncated = True
            break
        chunk = c.w.statement_execution.get_statement_result_chunk_n(response.statement_id, chunk.next_chunk_index)
    if result.total_row_count is not None and result.total_row_count > len(result.rows):
        result.truncated = True
    result.row_count = len(result.rows)
    return result


def run_statement(
    c: AppContext,
    statement: str,
    *,
    warehouse_id: str,
    catalog: str | None = None,
    schema: str | None = None,
    parameters: dict[str, Any] | None = None,
    max_rows: int | None = None,
    wait_timeout_seconds: int | None = None,
) -> SqlResult:
    limit = min(max_rows or c.settings.sql_max_rows, c.settings.sql_max_rows)
    wait = min(max(wait_timeout_seconds or c.settings.sql_wait_timeout_seconds, 5), 50)
    response = c.w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema,
        parameters=_parameters(parameters),
        row_limit=limit + 1,  # +1 lets us detect truncation precisely
        disposition=sql_svc.Disposition.INLINE,
        format=sql_svc.Format.JSON_ARRAY,
        wait_timeout=f"{wait}s",
        on_wait_timeout=sql_svc.ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    return collect_result(c, response, limit)


def run_ddl(c: AppContext, statement: str, *, warehouse_id: str | None = None) -> tuple[SqlResult, WarehouseChoice]:
    """Run a single DDL/metadata statement and raise a categorized error if it fails."""
    choice = select_warehouse(c, warehouse_id)
    result = run_statement(c, statement, warehouse_id=choice.warehouse_id, wait_timeout_seconds=50)
    if result.pending:
        raise DbxToolError(
            ErrorCategory.TIMEOUT,
            f"Statement still running after 50s (statement_id={result.statement_id}).",
            hint="Poll with manage_sql_statement action=get.",
        )
    if not result.succeeded:
        raise sql_error(result)
    return result, choice


def sql_error(result: SqlResult) -> DbxToolError:
    err = result.error or {}
    code = (err.get("error_code") or "").upper()
    message = err.get("message") or f"Statement ended in state {result.state}"
    category = ErrorCategory.SERVICE_ERROR
    lowered = message.lower()
    if "permission" in lowered or "not authorized" in lowered or "insufficient" in lowered or "PERMISSION" in code:
        category = ErrorCategory.AUTHORIZATION
    elif "not found" in lowered or "cannot be found" in lowered or "does not exist" in lowered or "NOT_FOUND" in code:
        category = ErrorCategory.NOT_FOUND
    elif "already exists" in lowered:
        category = ErrorCategory.CONFLICT
    elif "PARSE" in lowered.upper() or "syntax" in lowered or "BAD_REQUEST" in code or "INVALID" in code:
        category = ErrorCategory.INVALID_PARAMETER
    suffix = f" (sql_state={result.sql_state}, statement_id={result.statement_id})"
    return DbxToolError(category, message + suffix, hint="")
