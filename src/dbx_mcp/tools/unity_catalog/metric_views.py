"""Unity Catalog metric views: manage_metric_views.

The Databricks SDK has no dedicated metric-view API. The documented interface is SQL:

* ``CREATE [OR REPLACE] VIEW <name> WITH METRICS LANGUAGE YAML AS $$ <yaml> $$``
* ``DROP VIEW <name>``
* ``SELECT <dims>, MEASURE(<measure>) FROM <view> GROUP BY <dims>``

Metadata is read through the Unity Catalog Tables API: metric views are tables with
``TableInfo.table_type == TableType.METRIC_VIEW`` and their YAML definition is in
``TableInfo.view_definition``.
"""

from __future__ import annotations

import difflib
import re
from typing import Annotated, Any, Literal

from databricks.sdk.service import catalog
from pydantic import Field

from dbx_mcp.databricks.sql_runner import run_ddl, run_statement, select_warehouse, sql_error
from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, EXECUTION, READ, WRITE
from dbx_mcp.safety.validation import quote_full_name, split_full_name
from dbx_mcp.server.context import AppContext
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.tools.unity_catalog.security_policies import audit_block, column_ident
from dbx_mcp.utils.errors import ValidationFailed
from dbx_mcp.utils.serialization import pick, to_jsonable

_SAFETY = {
    "create": WRITE,
    "get": READ,
    "list": READ,
    "update": WRITE | DESTRUCTIVE,
    "delete": WRITE | DESTRUCTIVE,
    "query": READ | EXECUTION,
}

_BAD_YAML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_OPS = {"=", "!=", "<>", "<", "<=", ">", ">=", "LIKE", "NOT LIKE", "IS NULL", "IS NOT NULL"}
_NO_VALUE_OPS = {"IS NULL", "IS NOT NULL"}


def _name(full_name: str | None, action: str) -> str:
    require(full_name, "full_name", action)
    split_full_name(full_name, parts=3)
    return full_name  # type: ignore[return-value]


def _yaml(definition: Any, action: str) -> str:
    if not isinstance(definition, str):
        raise ValidationFailed(f"'yaml_definition' must be a string containing the metric view YAML for {action}")
    body = definition.strip("\n")
    if not body.strip():
        raise ValidationFailed(f"'yaml_definition' is required for {action}")
    if "$$" in body:
        raise ValidationFailed("'yaml_definition' must not contain '$$' (it delimits the YAML body in SQL)")
    if _BAD_YAML_CHARS.search(body):
        raise ValidationFailed("'yaml_definition' contains control characters")
    return body


def _create_sql(name: str, body: str, *, replace: bool) -> str:
    verb = "CREATE OR REPLACE VIEW" if replace else "CREATE VIEW"
    return f"{verb} {quote_full_name(name, parts=3)}\nWITH METRICS\nLANGUAGE YAML\nAS $$\n{body}\n$$"


def _is_metric_view(info: catalog.TableInfo) -> bool:
    return info.table_type == catalog.TableType.METRIC_VIEW or to_jsonable(info.table_type) == "METRIC_VIEW"


def _get_metric_view(c: AppContext, name: str) -> catalog.TableInfo:
    info = c.w.tables.get(name)
    if not _is_metric_view(info):
        raise ValidationFailed(
            f"{name} is a {to_jsonable(info.table_type) or 'unknown object type'}, not a metric view; "
            "refusing to change it with this tool."
        )
    return info


def _describe(info: catalog.TableInfo) -> dict[str, Any]:
    data = pick(
        to_jsonable(info),
        [
            "full_name",
            "catalog_name",
            "schema_name",
            "name",
            "table_type",
            "owner",
            "comment",
            "created_at",
            "created_by",
            "updated_at",
            "updated_by",
            "properties",
            "view_dependencies",
        ],
    )
    data["definition_yaml"] = info.view_definition
    data["columns"] = [
        pick(to_jsonable(col), ["name", "type_text", "type_name", "comment", "nullable", "position"])
        for col in info.columns or []
    ]
    return data


def _preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in {"create", "update", "delete"}:
        return None
    c = ctx()
    name = _name(args.get("full_name"), action)
    if action == "create":
        body = _yaml(args.get("yaml_definition"), action)
        sql = _create_sql(name, body, replace=False)
        return PlanInfo(
            description=f"Create metric view {name}.",
            target={"full_name": name},
            details={"statement": sql},
            reversible=True,
        )
    c.safety.check_protected("metric view", name, None, operation=f"{action}")
    info = _get_metric_view(c, name)
    old = info.view_definition or ""
    if action == "update":
        body = _yaml(args.get("yaml_definition"), action)
        diff = list(
            difflib.unified_diff(old.splitlines(), body.splitlines(), "current", "new", lineterm="")
        )
        return PlanInfo(
            description=f"Replace the definition of metric view {name} (CREATE OR REPLACE VIEW).",
            target={"full_name": name},
            details={
                "current_definition": old,
                "new_definition": body,
                "diff": diff,
                "statement": _create_sql(name, body, replace=True),
            },
            warnings=[
                "The existing definition is replaced. Queries, dashboards or Genie spaces that use dimensions or "
                "measures removed by this change will fail. current_definition is included for rollback."
            ],
            reversible=True,
        )
    return PlanInfo(
        description=f"Drop metric view {name}.",
        target={"full_name": name},
        details={"current_definition": old, "statement": f"DROP VIEW {quote_full_name(name, parts=3)}"},
        warnings=["Anything querying this metric view will fail. current_definition is included to recreate it."],
        reversible=False,
    )


def _query_sql(
    view: str,
    dimensions: list[str] | None,
    measures: list[str] | None,
    filters: list[dict[str, Any]] | None,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    dims = [column_ident(d, "dimensions") for d in dimensions or []]
    meas = [column_ident(m, "measures") for m in measures or []]
    if not dims and not meas:
        raise ValidationFailed("query needs at least one of 'dimensions' or 'measures'")
    select = dims + [f"MEASURE({m}) AS {m}" for m in meas]
    sql = f"SELECT {', '.join(select)}\nFROM {quote_full_name(view, parts=3)}"
    params: dict[str, Any] = {}
    clauses = []
    for i, flt in enumerate(filters or []):
        if not isinstance(flt, dict):
            raise ValidationFailed(f"filters[{i}] must be an object {{dimension, op, value}}")
        col = column_ident(flt.get("dimension"), f"filters[{i}].dimension")
        op = str(flt.get("op", "=")).upper().strip()
        if op not in _OPS:
            raise ValidationFailed(f"filters[{i}].op must be one of {sorted(_OPS)}")
        if op in _NO_VALUE_OPS:
            clauses.append(f"{col} {op}")
            continue
        if "value" not in flt or flt["value"] is None:
            raise ValidationFailed(f"filters[{i}].value is required for op {op!r}")
        params[f"p{i}"] = flt["value"]
        clauses.append(f"{col} {op} :p{i}")
    if clauses:
        sql += "\nWHERE " + " AND ".join(clauses)
    if dims and meas:
        sql += f"\nGROUP BY {', '.join(dims)}"
    sql += f"\nLIMIT {int(limit)}"
    return sql, params


@tool(
    toolset="unity_catalog",
    title="Unity Catalog metric views",
    safety=_SAFETY,
    preview=_preview,
)
def manage_metric_views(
    action: Annotated[
        Literal["create", "get", "list", "update", "delete", "query"],
        Field(
            description=(
                "create / update (CREATE OR REPLACE) / delete a metric view from YAML; get: definition + metadata; "
                "list: metric views in a schema; query: SELECT dimensions + MEASURE(measures)."
            )
        ),
    ],
    full_name: Annotated[str | None, Field(description="Metric view name catalog.schema.view.")] = None,
    yaml_definition: Annotated[
        str | None,
        Field(
            description=(
                "create/update: the metric view YAML (e.g. version, source, dimensions[{name, expr}], "
                "measures[{name, expr}], optional filter/joins). Must not contain '$$'."
            )
        ),
    ] = None,
    catalog_name: Annotated[str | None, Field(description="list: catalog.")] = None,
    schema_name: Annotated[str | None, Field(description="list: schema.")] = None,
    dimensions: Annotated[list[str] | None, Field(description="query: dimension names to group by.")] = None,
    measures: Annotated[list[str] | None, Field(description="query: measure names (wrapped in MEASURE()).")] = None,
    filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description=(
                "query: [{dimension, op, value}] combined with AND; op in =, !=, <>, <, <=, >, >=, LIKE, NOT LIKE, "
                "IS NULL, IS NOT NULL. Values are bound as parameters."
            )
        ),
    ] = None,
    limit: Annotated[int, Field(description="query: max rows.", ge=1, le=10000)] = 100,
    warehouse_id: Annotated[str | None, Field(description="SQL warehouse (default: configured/auto-selected).")] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Unity Catalog metric views (semantic layer) - implemented with documented SQL DDL.

    - create(full_name, yaml_definition): CREATE VIEW ... WITH METRICS LANGUAGE YAML AS $$...$$
    - get(full_name): YAML definition, columns and metadata.
    - list(catalog_name, schema_name): metric views in a schema.
    - update(full_name, yaml_definition): CREATE OR REPLACE - destructive, plan shows old vs new definition.
    - delete(full_name): DROP VIEW - destructive, needs confirm.
    - query(full_name, dimensions, measures, filters?, limit?): SELECT dims, MEASURE(m) ... GROUP BY dims.
    DDL and queries run on a SQL warehouse (warehouse_id optional)."""
    c = ctx()

    if action == "list":
        require(catalog_name, "catalog_name", action)
        require(schema_name, "schema_name", action)
        it = (
            t
            for t in c.w.tables.list(catalog_name, schema_name, omit_columns=True, omit_properties=True)
            if _is_metric_view(t)
        )
        return paged_response(
            "metric views",
            it,
            page_size,
            page_token,
            lambda t: pick(to_jsonable(t), ["full_name", "name", "owner", "comment", "created_at", "updated_at"]),
        )

    name = _name(full_name, action)

    if action == "get":
        info = c.w.tables.get(name)
        if not _is_metric_view(info):
            raise ValidationFailed(f"{name} is a {to_jsonable(info.table_type)}, not a metric view")
        data = _describe(info)
        warnings = [] if info.view_definition else ["Unity Catalog returned no view_definition for this metric view."]
        return ok(f"Metric view {name}.", data, warnings=warnings)

    if action == "query":
        _get_metric_view(c, name)
        sql, params = _query_sql(name, dimensions, measures, filters, limit)
        choice = select_warehouse(c, warehouse_id)
        result = run_statement(c, sql, warehouse_id=choice.warehouse_id, parameters=params or None, max_rows=limit)
        if result.pending:
            return ok(
                f"Query still running (statement_id={result.statement_id}).",
                {"statement_id": result.statement_id, "statement": sql},
                status="pending",
                next_steps=["Poll with manage_sql_statement action=get."],
            )
        if not result.succeeded:
            raise sql_error(result)
        return ok(
            f"Returned {result.row_count} row(s) from metric view {name}.",
            {
                "statement": sql,
                "parameters": params,
                "warehouse": choice.as_dict(),
                "columns": result.columns,
                "rows": result.records(),
                "truncated": result.truncated,
            },
            warnings=choice.warnings,
        )

    if action == "create":
        body = _yaml(yaml_definition, action)
        sql = _create_sql(name, body, replace=False)
        _result, choice = run_ddl(c, sql, warehouse_id=warehouse_id)
        return ok(
            f"Created metric view {name}.",
            {"full_name": name, "statement": sql, "audit": audit_block(c, statement=sql, warehouse_id=choice.warehouse_id)},
            warnings=choice.warnings,
        )

    c.safety.check_protected("metric view", name, None, operation=action)
    info = _get_metric_view(c, name)  # never replace/drop a table or regular view by accident

    if action == "update":
        body = _yaml(yaml_definition, action)
        sql = _create_sql(name, body, replace=True)
        _result, choice = run_ddl(c, sql, warehouse_id=warehouse_id)
        return ok(
            f"Replaced the definition of metric view {name}.",
            {
                "full_name": name,
                "previous_definition": info.view_definition,
                "statement": sql,
                "audit": audit_block(c, statement=sql, warehouse_id=choice.warehouse_id),
            },
            warnings=choice.warnings,
        )

    # delete
    sql = f"DROP VIEW {quote_full_name(name, parts=3)}"
    _result, choice = run_ddl(c, sql, warehouse_id=warehouse_id)
    return ok(
        f"Dropped metric view {name}.",
        {
            "full_name": name,
            "previous_definition": info.view_definition,
            "statement": sql,
            "audit": audit_block(c, statement=sql, warehouse_id=choice.warehouse_id),
        },
        warnings=choice.warnings,
    )
