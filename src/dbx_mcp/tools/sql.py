"""SQL & data exploration: execute_sql, execute_sql_multi, manage_sql_statement,
get_table_stats_and_schema."""

from __future__ import annotations

import contextlib
import json
import time
from typing import Annotated, Any, Literal

from pydantic import Field

from dbx_mcp.databricks.sql_runner import (
    SqlResult,
    WarehouseChoice,
    collect_result,
    rank_warehouses,
    run_statement,
    select_warehouse,
    sql_error,
)
from dbx_mcp.models.common import ToolResponse
from dbx_mcp.models.sql import (
    SqlColumn,
    SqlExecution,
    SqlExecutionMetadata,
    SqlExecutionResponse,
    SqlMultiExecution,
    SqlMultiResponse,
    SqlResultSet,
    StatementOutcome,
    WarehouseSelection,
)
from dbx_mcp.safety.levels import EXECUTION, READ, SafetyLevel
from dbx_mcp.safety.validation import (
    StatementClassification,
    classify_sql,
    quote_full_name,
    split_full_name,
)
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, ctx, list_page, ok
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import ValidationFailed, normalize_exception
from dbx_mcp.utils.serialization import pick

ALL_LEVELS = frozenset(SafetyLevel)

WarehouseId = Annotated[
    str | None,
    Field(description="SQL warehouse id. If omitted: DBX_MCP_DEFAULT_WAREHOUSE_ID, else automatic selection "
          "(reported in the response)."),
]
MaxRows = Annotated[int | None, Field(description="Maximum rows to return (capped by DBX_MCP_SQL_MAX_ROWS).", ge=1)]
RowFormat = Annotated[
    Literal["arrays", "objects"],
    Field(description="'arrays' (compact, aligned with columns) or 'objects' (one dict per row)."),
]

_NUMERIC_INT = {"TINYINT", "SMALLINT", "INT", "INTEGER", "BIGINT", "LONG", "BYTE", "SHORT"}
_NUMERIC_FLOAT = {"FLOAT", "DOUBLE", "REAL"}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _typed(value: Any, type_text: str | None) -> Any:
    """The JSON_ARRAY format returns every value as a string; convert simple scalar types."""
    if value is None or not isinstance(value, str) or not type_text:
        return value
    base = type_text.upper().split("(")[0].split("<")[0].strip()
    try:
        if base in _NUMERIC_INT:
            return int(value)
        if base in _NUMERIC_FLOAT:
            return float(value)
        if base == "BOOLEAN":
            return value.lower() == "true"
    except ValueError:
        return value
    return value


def _result_set(result: SqlResult, row_format: str) -> SqlResultSet | None:
    if not result.columns and not result.rows:
        return None
    columns = [SqlColumn(**col) for col in result.columns]
    types = [col.type for col in columns]
    rows: list[Any] = [[_typed(v, types[i] if i < len(types) else None) for i, v in enumerate(row)] for row in result.rows]
    if row_format == "objects":
        names = [col.name for col in columns]
        rows = [dict(zip(names, row, strict=False)) for row in rows]
    return SqlResultSet(
        columns=columns,
        row_format=row_format,  # type: ignore[arg-type]
        rows=rows,
        row_count=result.row_count,
        total_row_count=result.total_row_count,
        truncated=result.truncated,
    )


def _selection(choice: WarehouseChoice) -> WarehouseSelection:
    return WarehouseSelection(**choice.as_dict())


def _sql_levels(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    statement = args.get("statement")
    if isinstance(statement, str) and statement.strip():
        return classify_sql(statement)[1]
    sql = _multi_sql(args)
    if sql.strip():
        return classify_sql(sql)[1]
    raise ValidationFailed("A SQL statement is required")


def _multi_sql(args: dict[str, Any]) -> str:
    if isinstance(args.get("script"), str):
        return args["script"]
    statements = args.get("statements")
    if isinstance(statements, list):
        return ";\n".join(s for s in statements if isinstance(s, str))
    return ""


def _describe_classification(items: list[StatementClassification]) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "kind": c.kind,
            "keyword": c.keyword,
            "safety": sorted(level.value for level in c.levels),
            "reasons": c.reasons,
            "statement": c.statement if len(c.statement) <= 2000 else c.statement[:2000] + " ...",
        }
        for i, c in enumerate(items)
    ]


def _sql_preview(args: dict[str, Any]) -> PlanInfo:
    sql = args.get("statement") or _multi_sql(args)
    items, levels = classify_sql(sql)
    warnings = []
    if SafetyLevel.DESTRUCTIVE in levels:
        warnings.append("Contains statements that delete, drop, overwrite or replace data/objects.")
    if SafetyLevel.SECURITY_SENSITIVE in levels:
        warnings.append("Contains statements that change permissions, sharing, credentials or security policies.")
    if any(c.kind == "unknown" for c in items):
        warnings.append("Contains statements the server could not classify; treated as destructive.")
    choice = None
    try:
        choice = select_warehouse(ctx(), args.get("warehouse_id"))
        warnings.extend(choice.warnings)
    except Exception as exc:  # preview should still work without a warehouse
        warnings.append(f"Warehouse selection failed: {exc}")
    return PlanInfo(
        description=f"Execute {len(items)} SQL statement(s) on warehouse "
        f"{choice.warehouse_id if choice else '<unresolved>'}.",
        target={"warehouse_id": choice.warehouse_id if choice else None,
                "catalog": args.get("catalog"), "schema": args.get("schema")},
        details={"statements": _describe_classification(items)},
        warnings=warnings,
        reversible=SafetyLevel.DESTRUCTIVE not in levels,
    )


# ----------------------------------------------------------------------------------------------
# execute_sql
# ----------------------------------------------------------------------------------------------

@tool(
    toolset="sql",
    title="Execute SQL",
    safety=_sql_levels,
    possible_levels=ALL_LEVELS,
    preview=_sql_preview,
)
def execute_sql(
    statement: Annotated[str, Field(description="A single SQL statement (SELECT, DDL or DML). Use "
                                    "execute_sql_multi for scripts.")],
    warehouse_id: WarehouseId = None,
    catalog: Annotated[str | None, Field(description="Default catalog for unqualified names.")] = None,
    schema: Annotated[str | None, Field(description="Default schema for unqualified names.")] = None,
    parameters: Annotated[
        dict[str, Any] | None,
        Field(description="Named parameters referenced as :name in the statement (values are bound "
              "server-side, never interpolated)."),
    ] = None,
    max_rows: MaxRows = None,
    row_format: RowFormat = "arrays",
    wait_timeout_seconds: Annotated[
        int | None, Field(description="Seconds to wait (5-50) before returning a pending statement id.", ge=5, le=50)
    ] = None,
    confirm: Confirm = False,
    dry_run: DryRun = False,
) -> SqlExecutionResponse:
    """Execute one SQL statement on a Databricks SQL warehouse via the Statement Execution API.

    The statement is classified before running: SELECT/SHOW/DESCRIBE are reads; INSERT/CREATE
    are writes; DROP/DELETE/TRUNCATE/UPDATE/MERGE/OR REPLACE/INSERT OVERWRITE are destructive and
    GRANT/REVOKE/ownership/row-filter/mask changes are security-sensitive. Destructive and
    security-sensitive statements require confirm=true. The response separates `data.result`
    (columns, rows, truncation) from `data.execution` (statement id, state, warehouse used and why).
    Rows are capped by max_rows. If the statement is still running after wait_timeout_seconds the
    response has status 'pending' - poll with manage_sql_statement."""
    c = ctx()
    items, _ = classify_sql(statement)
    if len(items) > 1:
        raise ValidationFailed(
            f"execute_sql accepts exactly one statement; got {len(items)}. Use execute_sql_multi."
        )
    classification = items[0]
    choice = select_warehouse(c, warehouse_id)
    started = time.monotonic()
    result = run_statement(
        c,
        statement,
        warehouse_id=choice.warehouse_id,
        catalog=catalog,
        schema=schema,
        parameters=parameters,
        max_rows=max_rows,
        wait_timeout_seconds=wait_timeout_seconds,
    )
    meta = SqlExecutionMetadata(
        statement_id=result.statement_id,
        state=result.state,
        statement_kind=classification.kind,
        classification_reasons=classification.reasons,
        warehouse=_selection(choice),
        duration_ms=round((time.monotonic() - started) * 1000, 1),
        error=result.error,
        sql_state=result.sql_state,
    )
    if result.pending:
        return SqlExecutionResponse(
            status="pending",
            tool="execute_sql",
            summary=f"Statement is still {result.state} on warehouse {choice.warehouse_id}.",
            data=SqlExecution(result=None, execution=meta),
            warnings=choice.warnings,
            next_steps=[f"Call manage_sql_statement with action='get' and statement_id='{result.statement_id}'."],
        )
    if not result.succeeded:
        raise sql_error(result)
    result_set = _result_set(result, row_format)
    if result_set is None:
        summary = f"{classification.kind.capitalize()} statement succeeded (no result rows)."
    else:
        more = " (truncated - more rows available)" if result_set.truncated else ""
        summary = f"Statement succeeded: {result_set.row_count} row(s), {len(result_set.columns)} column(s){more}."
    return SqlExecutionResponse(
        tool="execute_sql",
        summary=summary,
        data=SqlExecution(result=result_set, execution=meta),
        warnings=choice.warnings,
    )


# ----------------------------------------------------------------------------------------------
# execute_sql_multi
# ----------------------------------------------------------------------------------------------

@tool(
    toolset="sql",
    title="Execute multiple SQL statements",
    safety=_sql_levels,
    possible_levels=ALL_LEVELS,
    preview=_sql_preview,
)
def execute_sql_multi(
    statements: Annotated[
        list[str] | None,
        Field(description="Statements to run in order. Alternatively pass `script`."),
    ] = None,
    script: Annotated[
        str | None, Field(description="A SQL script; split on top-level semicolons (comments/literals respected).")
    ] = None,
    continue_on_error: Annotated[
        bool, Field(description="Keep executing after a failed statement (default: stop at first failure).")
    ] = False,
    warehouse_id: WarehouseId = None,
    catalog: Annotated[str | None, Field(description="Default catalog.")] = None,
    schema: Annotated[str | None, Field(description="Default schema.")] = None,
    max_rows_per_statement: Annotated[int, Field(description="Row cap per statement result.", ge=1)] = 100,
    row_format: RowFormat = "arrays",
    confirm: Confirm = False,
    dry_run: DryRun = False,
) -> SqlMultiResponse:
    """Execute several SQL statements sequentially, preserving order, and report success/failure
    per statement with statement-level errors. Stops at the first failure unless
    continue_on_error=true (remaining statements are reported as 'skipped'). There is no
    transaction: completed statements are not rolled back. Safety is the union of all statements'
    classifications (any destructive statement requires confirm=true)."""
    if statements is not None and script is not None:
        raise ValidationFailed("Pass either `statements` or `script`, not both")
    return _run_multi(
        statements, script, continue_on_error, warehouse_id, catalog, schema, max_rows_per_statement, row_format
    )


def _run_multi(
    statements: list[str] | None,
    script: str | None,
    continue_on_error: bool,
    warehouse_id: str | None,
    catalog: str | None,
    schema: str | None,
    max_rows: int,
    row_format: str,
) -> SqlMultiResponse:
    c = ctx()
    sql = script if script is not None else ";\n".join(statements or [])
    if not sql.strip():
        raise ValidationFailed("No SQL statements provided")
    items, _ = classify_sql(sql)
    if len(items) > 100:
        raise ValidationFailed("At most 100 statements per call")
    choice = select_warehouse(c, warehouse_id)
    outcomes: list[StatementOutcome] = []
    stop = False
    for index, item in enumerate(items):
        if stop:
            outcomes.append(StatementOutcome(index=index, statement=item.statement, statement_kind=item.kind,
                                             status="skipped"))
            continue
        try:
            result = run_statement(c, item.statement, warehouse_id=choice.warehouse_id, catalog=catalog,
                                   schema=schema, max_rows=max_rows)
        except Exception as exc:  # API-level failure for this statement
            err = normalize_exception(exc, debug=c.settings.debug)
            outcomes.append(StatementOutcome(index=index, statement=item.statement, statement_kind=item.kind,
                                             status="failed", error={"category": err.category.value, "message": str(err)}))
            stop = not continue_on_error
            continue
        if result.pending:
            outcomes.append(StatementOutcome(index=index, statement=item.statement, statement_kind=item.kind,
                                             status="pending", statement_id=result.statement_id))
            stop = True  # order must be preserved: never start the next statement while one is running
            continue
        if not result.succeeded:
            outcomes.append(StatementOutcome(index=index, statement=item.statement, statement_kind=item.kind,
                                             status="failed", statement_id=result.statement_id,
                                             error={**(result.error or {}), "sql_state": result.sql_state}))
            stop = not continue_on_error
            continue
        outcomes.append(StatementOutcome(index=index, statement=item.statement, statement_kind=item.kind,
                                         status="succeeded", statement_id=result.statement_id,
                                         result=_result_set(result, row_format)))

    counts = {s: sum(1 for o in outcomes if o.status == s) for s in ("succeeded", "failed", "pending", "skipped")}
    status = "success"
    if counts["pending"]:
        status = "pending"
    elif counts["failed"]:
        status = "partial_failure" if counts["succeeded"] else "failed"
    summary = (
        f"{counts['succeeded']}/{len(items)} statement(s) succeeded, {counts['failed']} failed, "
        f"{counts['pending']} still running, {counts['skipped']} skipped."
    )
    next_steps = []
    if counts["pending"]:
        pending = next(o for o in outcomes if o.status == "pending")
        next_steps.append(
            f"Statement {pending.index} is still running: poll manage_sql_statement action='get' "
            f"statement_id='{pending.statement_id}', then re-run the skipped statements."
        )
    return SqlMultiResponse(
        status=status,  # type: ignore[arg-type]
        tool="execute_sql_multi",
        summary=summary,
        data=SqlMultiExecution(warehouse=_selection(choice), statements=outcomes, **counts),
        warnings=choice.warnings,
        next_steps=next_steps,
    )


# ----------------------------------------------------------------------------------------------
# manage_sql_statement
# ----------------------------------------------------------------------------------------------

@tool(
    toolset="sql",
    title="Inspect or cancel a SQL statement",
    safety={"get": READ, "cancel": EXECUTION},
)
def manage_sql_statement(
    action: Annotated[Literal["get", "cancel"], Field(description="get: status and results; cancel: stop it.")],
    statement_id: Annotated[str, Field(description="Statement id returned by execute_sql.")],
    max_rows: MaxRows = None,
    row_format: RowFormat = "arrays",
    confirm: Confirm = False,
    dry_run: DryRun = False,
) -> SqlExecutionResponse:
    """Poll a previously submitted SQL statement (status and, once finished, its results) or
    cancel it. Use after execute_sql returned status 'pending'."""
    c = ctx()
    if action == "cancel":
        c.w.statement_execution.cancel_execution(statement_id)
        return SqlExecutionResponse(
            tool="manage_sql_statement",
            summary=f"Cancellation requested for statement {statement_id}.",
            data=SqlExecution(execution=SqlExecutionMetadata(statement_id=statement_id, state="CANCEL_REQUESTED",
                                                             statement_kind="unknown")),
        )
    response = c.w.statement_execution.get_statement(statement_id)
    limit = min(max_rows or c.settings.sql_max_rows, c.settings.sql_max_rows)
    result = collect_result(c, response, limit)
    meta = SqlExecutionMetadata(statement_id=statement_id, state=result.state, statement_kind="unknown",
                                error=result.error, sql_state=result.sql_state)
    if result.pending:
        return SqlExecutionResponse(
            status="pending", tool="manage_sql_statement",
            summary=f"Statement {statement_id} is still {result.state}.",
            data=SqlExecution(execution=meta),
            next_steps=["Poll again shortly, or cancel with action='cancel'."],
        )
    if not result.succeeded:
        return SqlExecutionResponse(
            status="failed", tool="manage_sql_statement",
            summary=f"Statement {statement_id} ended in state {result.state}: "
            f"{(result.error or {}).get('message', 'no error message')}",
            data=SqlExecution(execution=meta),
        )
    result_set = _result_set(result, row_format)
    return SqlExecutionResponse(
        tool="manage_sql_statement",
        summary=f"Statement {statement_id} succeeded"
        + (f": {result_set.row_count} row(s)." if result_set else " (no result rows)."),
        data=SqlExecution(result=result_set, execution=meta),
    )


# ----------------------------------------------------------------------------------------------
# get_table_stats_and_schema
# ----------------------------------------------------------------------------------------------

_TABLE_LIST_KEYS = ("name", "full_name", "table_type", "data_source_format", "comment", "owner", "updated_at")


def _stats_levels(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    if (args.get("stats") or "auto") == "none":
        return READ
    return READ | EXECUTION


def _column(col: Any) -> dict[str, Any]:
    return {
        "name": col.name,
        "type": col.type_text or getattr(col.type_name, "value", None),
        "nullable": col.nullable,
        "comment": col.comment,
        "position": col.position,
        "partition_index": col.partition_index,
        "has_mask": col.mask is not None,
    }


def _parse_describe_detail(result: SqlResult) -> dict[str, Any]:
    if not result.rows:
        return {}
    record = result.records()[0]
    detail: dict[str, Any] = {}
    for key in ("format", "numFiles", "sizeInBytes", "partitionColumns", "clusteringColumns", "lastModified",
                "minReaderVersion", "minWriterVersion", "tableFeatures", "statistics"):
        value = record.get(key)
        if isinstance(value, str) and value[:1] in "[{":
            with contextlib.suppress(ValueError):
                value = json.loads(value)
        if value is not None:
            detail[key] = value
    for key in ("numFiles", "sizeInBytes"):
        if isinstance(detail.get(key), str) and detail[key].isdigit():
            detail[key] = int(detail[key])
    return detail


def _running_warehouse(c: Any, requested: str | None) -> WarehouseChoice | None:
    if requested or c.default_warehouse_id:
        return select_warehouse(c, requested)
    for wh in rank_warehouses(list(c.w.warehouses.list())):
        if getattr(wh.state, "value", wh.state) == "RUNNING":
            return WarehouseChoice(wh.id or "", wh.name, "RUNNING", "first running warehouse (no startup cost)")
    return None


@tool(
    toolset="sql",
    title="Table/schema details and statistics",
    safety=_stats_levels,
    possible_levels=READ | EXECUTION,
)
def get_table_stats_and_schema(
    name: Annotated[
        str,
        Field(description="Fully qualified table 'catalog.schema.table' for one table, or 'catalog.schema' "
              "to list the tables in a schema."),
    ],
    stats: Annotated[
        Literal["auto", "none", "metadata", "exact_count"],
        Field(description="none: Unity Catalog metadata only. metadata: also DESCRIBE DETAIL (files, size, "
              "partitioning) on a warehouse. exact_count: also SELECT COUNT(*) (scans the table). auto: "
              "metadata only if a warehouse is already running (never starts one)."),
    ] = "auto",
    warehouse_id: WarehouseId = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
) -> ToolResponse:
    """Inspect a Unity Catalog table (catalog, schema, type, format, columns with types,
    nullability, comments, partition columns, location, owner, properties, row filter/masks
    presence) plus optional statistics (file count, size, partitioning, row count). Given a
    two-part 'catalog.schema' name, lists the tables in that schema (paginated)."""
    c = ctx()
    parts = split_full_name(name, parts=(2, 3))
    if len(parts) == 2:
        catalog_name, schema_name = parts
        tables = c.w.tables.list(catalog_name=catalog_name, schema_name=schema_name)
        page, info = list_page(tables, page_size, page_token, lambda t: pick(t.as_dict(), _TABLE_LIST_KEYS))
        return ok(f"Schema {catalog_name}.{schema_name}: {info.returned} table(s) on this page.", page, page=info)

    table = c.w.tables.get(".".join(parts))
    columns = [_column(col) for col in (table.columns or [])]
    data: dict[str, Any] = {
        "catalog": table.catalog_name,
        "schema": table.schema_name,
        "table": table.name,
        "full_name": table.full_name,
        "table_type": getattr(table.table_type, "value", table.table_type),
        "data_source_format": getattr(table.data_source_format, "value", table.data_source_format),
        "comment": table.comment,
        "owner": table.owner,
        "storage_location": table.storage_location,
        "created_at": table.created_at,
        "updated_at": table.updated_at,
        "columns": columns,
        "partition_columns": [col["name"] for col in sorted(
            (col for col in columns if col["partition_index"] is not None), key=lambda col: col["partition_index"])],
        "has_row_filter": table.row_filter is not None,
        "view_definition": table.view_definition,
        "properties": table.properties or {},
        "statistics": None,
    }
    warnings: list[str] = []
    is_view = data["table_type"] in {"VIEW", "MATERIALIZED_VIEW", "METRIC_VIEW"}
    if stats != "none":
        choice = _running_warehouse(c, warehouse_id) if stats == "auto" else select_warehouse(c, warehouse_id)
        if choice is None:
            warnings.append("No running SQL warehouse; statistics skipped (use stats='metadata' to start one).")
        else:
            warnings.extend(choice.warnings)
            data["statistics"] = _table_statistics(c, table.full_name or name, choice, stats, is_view, warnings)
    return ok(
        f"{data['full_name']}: {data['table_type']} with {len(columns)} column(s)"
        + (f", partitioned by {', '.join(data['partition_columns'])}" if data["partition_columns"] else "")
        + ".",
        data,
        warnings=warnings,
    )


def _table_statistics(c: Any, full_name: str, choice: WarehouseChoice, mode: str, is_view: bool,
                      warnings: list[str]) -> dict[str, Any]:
    quoted = quote_full_name(full_name, parts=3)
    out: dict[str, Any] = {"warehouse_id": choice.warehouse_id, "source": []}
    if is_view:
        warnings.append("File/size statistics are not available for views (no DESCRIBE DETAIL); "
                        "use stats='exact_count' for a row count.")
    else:
        detail = run_statement(c, f"DESCRIBE DETAIL {quoted}", warehouse_id=choice.warehouse_id, max_rows=1,
                               wait_timeout_seconds=30)
        if detail.succeeded:
            out.update(_parse_describe_detail(detail))
            out["source"].append("DESCRIBE DETAIL")
        else:
            warnings.append(f"DESCRIBE DETAIL unavailable: {(detail.error or {}).get('message', detail.state)}")
    if mode == "exact_count":
        count = run_statement(c, f"SELECT COUNT(*) AS row_count FROM {quoted}", warehouse_id=choice.warehouse_id,
                              max_rows=1, wait_timeout_seconds=50)
        if count.succeeded and count.rows:
            out["row_count"] = int(count.rows[0][0])
            out["source"].append("SELECT COUNT(*)")
        elif count.pending:
            warnings.append(f"Row count still running (statement_id={count.statement_id}); poll manage_sql_statement.")
        else:
            warnings.append(f"Row count failed: {(count.error or {}).get('message', count.state)}")
    return out

