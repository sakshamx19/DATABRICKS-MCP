"""Unity Catalog fine-grained access control: manage_uc_security_policies.

Two mechanisms are covered:

* **Row filters / column masks** bound directly to a table. The documented interface
  is SQL DDL (``ALTER TABLE ... SET ROW FILTER`` / ``ALTER COLUMN ... SET MASK``), so
  changes run through the SQL Statement Execution API on a warehouse. Current state is
  read from ``w.tables.get`` (``TableInfo.row_filter`` / ``ColumnInfo.mask``).
* **ABAC policies** (attribute-based row-filter / column-mask policies defined on a
  catalog, schema or table) via the ``w.policies`` API (``PoliciesAPI``).

Every change is SECURITY_SENSITIVE and requires confirmation; the plan shows the
current state next to the requested state, and every change response carries an
``audit`` block (who / what / when).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service import catalog
from pydantic import Field

from dbx_mcp.databricks.sql_runner import run_ddl
from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import (
    DESTRUCTIVE_SECURITY,
    READ_SECURITY,
    WRITE,
    WRITE_SECURITY,
)
from dbx_mcp.safety.validation import quote_full_name, quote_ident, split_full_name
from dbx_mcp.server.context import AppContext
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import ValidationFailed
from dbx_mcp.utils.serialization import parse_sdk_object, pick, to_jsonable

# ----------------------------------------------------------------------------------------------
# Shared audit helpers (also used by the other governance tools in this package)
# ----------------------------------------------------------------------------------------------

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def current_user_name(c: AppContext) -> str | None:
    """Best-effort name of the authenticated principal (cached)."""
    cached = c.cached_user_name()
    if isinstance(cached, str):
        return cached
    try:
        name = c.w.current_user.me().user_name
    except Exception:
        return None
    if isinstance(name, str):
        c.remember_user_name(name)
        return name
    return None


def audit_block(
    c: AppContext,
    *,
    statement: str | None = None,
    api_call: str | None = None,
    warehouse_id: str | None = None,
) -> dict[str, Any]:
    """Who executed what, when - attached to every governance change response."""
    block: dict[str, Any] = {
        "who": current_user_name(c),
        "when": datetime.now(timezone.utc).isoformat(),
    }
    if statement is not None:
        block["what"] = statement
        block["via"] = "SQL Statement Execution API"
    if api_call is not None:
        block["what"] = api_call
        block["via"] = "Databricks REST API"
    if warehouse_id:
        block["warehouse_id"] = warehouse_id
    return block


def column_ident(name: str | None, what: str = "column_name") -> str:
    """Validate and backtick-quote a single column identifier."""
    if not isinstance(name, str) or not name.strip():
        raise ValidationFailed(f"'{what}' must be a non-empty column name")
    if _CONTROL_CHARS.search(name):
        raise ValidationFailed(f"'{what}' contains control characters")
    return quote_ident(name)


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

_ACTIONS = Literal[
    "get",
    "list_policies",
    "get_policy",
    "set_row_filter",
    "drop_row_filter",
    "set_column_mask",
    "drop_column_mask",
    "create_policy",
    "update_policy",
    "delete_policy",
]

_SAFETY = {
    "get": READ_SECURITY,
    "list_policies": READ_SECURITY,
    "get_policy": READ_SECURITY,
    "set_row_filter": WRITE_SECURITY,
    "set_column_mask": WRITE_SECURITY,
    "drop_row_filter": WRITE | DESTRUCTIVE_SECURITY,
    "drop_column_mask": WRITE | DESTRUCTIVE_SECURITY,
    "create_policy": WRITE_SECURITY,
    "update_policy": WRITE_SECURITY,
    "delete_policy": WRITE | DESTRUCTIVE_SECURITY,
}

# Securable types on which ABAC policies can be defined (subset of catalog.SecurableType).
_POLICY_SECURABLES = ("CATALOG", "SCHEMA", "TABLE")
_POLICY_READ_ONLY_FIELDS = {"id", "created_at", "created_by", "updated_at", "updated_by"}
_POLICY_ID_FIELDS = {"name", "on_securable_type", "on_securable_fullname"}


def _table(name: str | None, action: str) -> str:
    require(name, "table_name", action)
    split_full_name(name, parts=3)
    return name  # type: ignore[return-value]


def _function(name: str | None, action: str) -> str:
    require(name, "function_name", action)
    split_full_name(name, parts=3)  # fully qualified: an unqualified UDF would resolve against session defaults
    return name  # type: ignore[return-value]


def _columns(cols: list[str] | None, what: str) -> list[str]:
    return [column_ident(c, what) for c in cols or []]


def _current_controls(info: catalog.TableInfo) -> dict[str, Any]:
    masks = [
        {"column": col.name, "mask": to_jsonable(col.mask)}
        for col in info.columns or []
        if col.mask is not None
    ]
    return {"row_filter": to_jsonable(info.row_filter), "column_masks": masks}


def _column_info(info: catalog.TableInfo, column: str) -> catalog.ColumnInfo | None:
    for col in info.columns or []:
        if col.name is not None and col.name.lower() == column.lower():
            return col
    return None


def _build_sql(args: dict[str, Any]) -> str:
    action = args["action"]
    table = quote_full_name(_table(args.get("table_name"), action), parts=3)
    if action == "set_row_filter":
        fn = quote_full_name(_function(args.get("function_name"), action), parts=3)
        if args.get("using_columns") is None:
            raise ValidationFailed(
                "Parameter 'using_columns' is required for set_row_filter: the table columns passed to the "
                "filter function, in parameter order (use [] for a function without parameters)"
            )
        cols = ", ".join(_columns(args.get("using_columns"), "using_columns"))
        return f"ALTER TABLE {table} SET ROW FILTER {fn} ON ({cols})"
    if action == "drop_row_filter":
        return f"ALTER TABLE {table} DROP ROW FILTER"
    column = column_ident(require(args.get("column_name"), "column_name", action))
    if action == "set_column_mask":
        fn = quote_full_name(_function(args.get("function_name"), action), parts=3)
        sql = f"ALTER TABLE {table} ALTER COLUMN {column} SET MASK {fn}"
        extra = _columns(args.get("using_columns"), "using_columns")
        if extra:
            sql += f" USING COLUMNS ({', '.join(extra)})"
        return sql
    if action == "drop_column_mask":
        return f"ALTER TABLE {table} ALTER COLUMN {column} DROP MASK"
    raise ValidationFailed(f"Action {action!r} is not a SQL action")  # pragma: no cover


def _restore_sql(table: str, action: str, args: dict[str, Any], info: catalog.TableInfo) -> str | None:
    """SQL that would re-apply the control being dropped (for the audit trail / rollback)."""
    qt = quote_full_name(table, parts=3)
    if action == "drop_row_filter" and info.row_filter and info.row_filter.function_name:
        cols = ", ".join(quote_ident(c) for c in info.row_filter.input_column_names or [])
        return f"ALTER TABLE {qt} SET ROW FILTER {quote_full_name(info.row_filter.function_name)} ON ({cols})"
    if action == "drop_column_mask":
        col = _column_info(info, args["column_name"])
        if col and col.mask and col.mask.function_name:
            sql = f"ALTER TABLE {qt} ALTER COLUMN {quote_ident(col.name or '')} SET MASK {quote_full_name(col.mask.function_name)}"
            if col.mask.using_column_names:
                sql += f" USING COLUMNS ({', '.join(quote_ident(u) for u in col.mask.using_column_names)})"
            return sql
    return None


def _sql_plan(c: AppContext, args: dict[str, Any]) -> tuple[PlanInfo, catalog.TableInfo, str]:
    action = args["action"]
    table = _table(args.get("table_name"), action)
    sql = _build_sql(args)
    info = c.w.tables.get(table)
    current = _current_controls(info)
    warnings: list[str] = []
    details: dict[str, Any] = {"statement": sql}

    if action in {"drop_row_filter", "drop_column_mask"}:
        c.safety.check_protected("table", table, None, operation=action.replace("_", " ") + " on")

    if action == "set_row_filter":
        new = {"function_name": args["function_name"], "input_column_names": list(args.get("using_columns") or [])}
        details.update({"current_row_filter": current["row_filter"], "new_row_filter": new})
        if current["row_filter"]:
            warnings.append(
                f"Table already has row filter {current['row_filter'].get('function_name')!r}; it will be "
                "changed to the new function (if Databricks rejects this, drop the existing filter first)."
            )
        warnings.append("Rows not accepted by the filter function become invisible to affected users immediately.")
        description = f"Set row filter {args['function_name']} on table {table}."
        reversible = True
    elif action == "drop_row_filter":
        details.update({"current_row_filter": current["row_filter"], "new_row_filter": None})
        if not current["row_filter"]:
            warnings.append("The table currently has no row filter according to Unity Catalog metadata.")
        warnings.append(
            "Removing the row filter BROADENS access: every user with SELECT on the table will see all rows."
        )
        details["restore_statement"] = _restore_sql(table, action, args, info)
        description = f"Drop the row filter from table {table}."
        reversible = True
    else:
        column = args["column_name"]
        col = _column_info(info, column)
        if info.columns and col is None:
            raise ValidationFailed(f"Column {column!r} does not exist in table {table}")
        current_mask = to_jsonable(col.mask) if col else None
        details["column"] = column
        details["current_mask"] = current_mask
        if action == "set_column_mask":
            details["new_mask"] = {
                "function_name": args["function_name"],
                "using_column_names": list(args.get("using_columns") or []),
            }
            if current_mask:
                warnings.append(
                    f"Column already has mask {current_mask.get('function_name')!r}; it will be replaced by the new mask."
                )
            description = f"Set column mask {args['function_name']} on {table}.{column}."
        else:
            details["new_mask"] = None
            if not current_mask:
                warnings.append("The column currently has no mask according to Unity Catalog metadata.")
            warnings.append(
                f"Removing the mask BROADENS access: users with SELECT will see unmasked values of column {column!r}."
            )
            details["restore_statement"] = _restore_sql(table, action, args, info)
            description = f"Drop the column mask from {table}.{column}."
        reversible = True

    details["all_current_controls_on_table"] = current
    plan = PlanInfo(
        description=description,
        target={"table": table, **({"column": args["column_name"]} if args.get("column_name") else {})},
        details=details,
        warnings=warnings,
        reversible=reversible,
    )
    return plan, info, sql


def _securable(args: dict[str, Any], action: str) -> tuple[str, str]:
    stype = require(args.get("securable_type"), "securable_type", action).upper()
    if stype not in _POLICY_SECURABLES:
        raise ValidationFailed(f"securable_type must be one of {', '.join(_POLICY_SECURABLES)}, got {stype!r}")
    fullname = require(args.get("securable_fullname"), "securable_fullname", action)
    split_full_name(fullname, parts={"CATALOG": 1, "SCHEMA": 2, "TABLE": 3}[stype])
    return stype, fullname


def _policy_body(args: dict[str, Any], action: str) -> tuple[str, str, str | None, dict[str, Any]]:
    stype, fullname = _securable(args, action)
    spec = dict(args.get("spec") or {})
    bad = sorted(set(spec) & (_POLICY_READ_ONLY_FIELDS | _POLICY_ID_FIELDS))
    if bad:
        raise ValidationFailed(
            f"Field(s) {', '.join(bad)} cannot be set in spec; use policy_name / securable_type / "
            "securable_fullname parameters (output-only fields are rejected)."
        )
    name = args.get("policy_name")
    if action == "create_policy":
        require(name, "policy_name", action)
        missing = [f for f in ("to_principals", "for_securable_type", "policy_type") if not spec.get(f)]
        if missing:
            raise ValidationFailed(f"spec is missing required policy field(s): {', '.join(missing)}")
    elif action == "update_policy":
        require(name, "policy_name", action)
        if not spec:
            raise ValidationFailed("spec with the fields to change is required for update_policy")
    else:
        require(name, "policy_name", action)
        if spec:
            raise ValidationFailed(f"spec is not accepted for {action}")
    return stype, fullname, name, spec


def _policy_info(stype: str, fullname: str, name: str | None, spec: dict[str, Any]) -> catalog.PolicyInfo:
    body = {**spec, "on_securable_type": stype, "on_securable_fullname": fullname}
    if name:
        body["name"] = name
    return parse_sdk_object(catalog.PolicyInfo, body)


def _policy_summary(p: Any) -> dict[str, Any]:
    return pick(
        to_jsonable(p),
        [
            "name",
            "policy_type",
            "on_securable_type",
            "on_securable_fullname",
            "for_securable_type",
            "to_principals",
            "except_principals",
            "when_condition",
            "comment",
        ],
    )


def _policy_plan(c: AppContext, args: dict[str, Any]) -> PlanInfo:
    action = args["action"]
    stype, fullname, name, spec = _policy_body(args, action)
    target = {"securable_type": stype, "securable_fullname": fullname, "policy_name": name}
    if action == "create_policy":
        info = _policy_info(stype, fullname, name, spec)
        return PlanInfo(
            description=f"Create ABAC policy {name!r} on {stype.lower()} {fullname}.",
            target=target,
            details={"new_policy": to_jsonable(info), "api_call": "policies.create_policy"},
            warnings=[
                f"The policy applies to {fullname} and ALL its descendants (for securable type "
                f"{spec.get('for_securable_type')}), for principals {spec.get('to_principals')}."
            ],
            reversible=True,
        )
    current = to_jsonable(c.w.policies.get_policy(stype, fullname, name))
    if action == "update_policy":
        changes = {k: {"current": current.get(k), "new": to_jsonable(v)} for k, v in spec.items()}
        mask = args.get("update_mask") or ",".join(spec)
        return PlanInfo(
            description=f"Update ABAC policy {name!r} on {stype.lower()} {fullname} (fields: {mask}).",
            target=target,
            details={"changes": changes, "update_mask": mask, "current_policy": current, "api_call": "policies.update_policy"},
            warnings=["Fields listed in update_mask but absent from spec will be cleared."] if args.get("update_mask") else [],
            reversible=True,
        )
    # delete_policy
    c.safety.check_protected("policy", name, None, operation="delete")
    c.safety.check_protected(stype.lower(), fullname, None, operation="delete a policy on")
    policy_type = current.get("policy_type")
    warnings = ["Deleting a policy cannot be undone automatically; the current definition is included to recreate it."]
    if policy_type in {"POLICY_TYPE_ROW_FILTER", "POLICY_TYPE_COLUMN_MASK", "POLICY_TYPE_DENY"}:
        warnings.append("This policy restricts access; deleting it BROADENS what affected principals can see.")
    return PlanInfo(
        description=f"Delete ABAC policy {name!r} ({policy_type}) on {stype.lower()} {fullname}.",
        target=target,
        details={"current_policy": current, "new_policy": None, "api_call": "policies.delete_policy"},
        warnings=warnings,
        reversible=False,
    )


def _preview(args: dict[str, Any]) -> PlanInfo | None:
    c = ctx()
    action = args.get("action")
    if action in {"set_row_filter", "drop_row_filter", "set_column_mask", "drop_column_mask"}:
        return _sql_plan(c, args)[0]
    if action in {"create_policy", "update_policy", "delete_policy"}:
        return _policy_plan(c, args)
    return None


# ----------------------------------------------------------------------------------------------
# Tool
# ----------------------------------------------------------------------------------------------

@tool(
    toolset="unity_catalog",
    title="Unity Catalog row filters, column masks & ABAC policies",
    safety=_SAFETY,
    preview=_preview,
)
def manage_uc_security_policies(
    action: Annotated[
        _ACTIONS,
        Field(
            description=(
                "get: current row filter, column masks and ABAC policies on table_name; "
                "list_policies / get_policy: ABAC policies on a securable; "
                "set_row_filter / drop_row_filter / set_column_mask / drop_column_mask: table-bound UDF "
                "filters/masks (SQL DDL on a warehouse); create_policy / update_policy / delete_policy: ABAC policies."
            )
        ),
    ],
    table_name: Annotated[str | None, Field(description="Table full name catalog.schema.table.")] = None,
    column_name: Annotated[str | None, Field(description="Column for set_column_mask / drop_column_mask.")] = None,
    function_name: Annotated[
        str | None,
        Field(description="Fully qualified SQL UDF catalog.schema.function used as row filter or column mask."),
    ] = None,
    using_columns: Annotated[
        list[str] | None,
        Field(
            description=(
                "set_row_filter: table columns passed to the filter UDF, in order ([] for none). "
                "set_column_mask: additional columns passed after the masked column (USING COLUMNS)."
            )
        ),
    ] = None,
    securable_type: Annotated[
        Literal["CATALOG", "SCHEMA", "TABLE"] | None,
        Field(description="ABAC policies: type of the securable the policy is defined on."),
    ] = None,
    securable_fullname: Annotated[
        str | None, Field(description="ABAC policies: full name of that catalog / schema / table.")
    ] = None,
    policy_name: Annotated[str | None, Field(description="ABAC policy name (get/update/delete/create).")] = None,
    include_inherited: Annotated[
        bool | None,
        Field(description="list_policies/get: include policies inherited from parent schema/catalog (get defaults to true)."),
    ] = None,
    spec: Spec = None,
    update_mask: Annotated[
        str | None,
        Field(description="update_policy: comma-separated fields to update (default: the keys present in spec)."),
    ] = None,
    warehouse_id: Annotated[
        str | None, Field(description="SQL warehouse for filter/mask DDL (default: configured/auto-selected).")
    ] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Unity Catalog fine-grained access control.

    Actions:
    - get(table_name): current row filter + column masks (from table metadata) and ABAC policies in effect.
    - set_row_filter(table_name, function_name, using_columns) / drop_row_filter(table_name)
    - set_column_mask(table_name, column_name, function_name, using_columns?) / drop_column_mask(table_name, column_name)
      These run ALTER TABLE DDL on a SQL warehouse (warehouse_id optional).
    - list_policies(securable_type, securable_fullname, include_inherited?) / get_policy(+policy_name)
    - create_policy(securable_type, securable_fullname, policy_name, spec) - spec uses PolicyInfo fields:
      to_principals, for_securable_type, policy_type (POLICY_TYPE_ROW_FILTER|POLICY_TYPE_COLUMN_MASK),
      row_filter {function_name, using}, column_mask {function_name, on_column, using}, match_columns,
      when_condition, except_principals, comment.
    - update_policy(..., policy_name, spec, update_mask?) / delete_policy(..., policy_name)
    All changes are security-sensitive: call without confirm to get a plan showing current vs new state,
    then repeat with confirm=true. Change responses include an audit block (who/what/when)."""
    c = ctx()
    args = {
        "action": action,
        "table_name": table_name,
        "column_name": column_name,
        "function_name": function_name,
        "using_columns": using_columns,
        "securable_type": securable_type,
        "securable_fullname": securable_fullname,
        "policy_name": policy_name,
        "spec": spec,
        "update_mask": update_mask,
    }

    if action == "get":
        table = _table(table_name, action)
        info = c.w.tables.get(table)
        data: dict[str, Any] = {"table": info.full_name or table, **_current_controls(info)}
        warnings: list[str] = []
        try:
            policies = []
            for i, p in enumerate(
                c.w.policies.list_policies("TABLE", table, include_inherited=True if include_inherited is None else include_inherited)
            ):
                if i >= 100:
                    warnings.append("More than 100 ABAC policies; use list_policies to page through all of them.")
                    break
                policies.append(_policy_summary(p))
            data["abac_policies"] = policies
        except DatabricksError as exc:
            data["abac_policies"] = None
            warnings.append(f"Could not list ABAC policies for this table: {exc}")
        n_masks = len(data["column_masks"])
        summary = (
            f"{table}: row filter {'present' if data['row_filter'] else 'none'}, {n_masks} masked column(s)"
            + (f", {len(data['abac_policies'])} ABAC policy(ies)." if data.get("abac_policies") is not None else ".")
        )
        return ok(summary, data, warnings=warnings)

    if action == "list_policies":
        stype, fullname = _securable(args, action)
        it = c.w.policies.list_policies(stype, fullname, include_inherited=include_inherited)
        return paged_response("policies", it, page_size, page_token, _policy_summary)

    if action == "get_policy":
        stype, fullname = _securable(args, action)
        require(policy_name, "policy_name", action)
        policy = c.w.policies.get_policy(stype, fullname, policy_name)
        return ok(f"Policy {policy_name!r} on {stype.lower()} {fullname}.", policy)

    if action in {"set_row_filter", "drop_row_filter", "set_column_mask", "drop_column_mask"}:
        plan, _info, sql = _sql_plan(c, args)
        _result, choice = run_ddl(c, sql, warehouse_id=warehouse_id)
        data = {
            "table": table_name,
            "previous": {k: v for k, v in plan.details.items() if k.startswith("current")},
            "applied": {k: v for k, v in plan.details.items() if k.startswith("new")},
            "restore_statement": plan.details.get("restore_statement"),
            "audit": audit_block(c, statement=sql, warehouse_id=choice.warehouse_id),
        }
        return ok(f"Done: {plan.description}", data, warnings=[*choice.warnings])

    # ABAC policy changes
    stype, fullname, name, body = _policy_body(args, action)
    if action == "create_policy":
        info = _policy_info(stype, fullname, name, body)
        created = c.w.policies.create_policy(info)
        data = {"policy": to_jsonable(created), "audit": audit_block(c, api_call=f"policies.create_policy {stype} {fullname} {name}")}
        return ok(f"Created ABAC policy {name!r} on {stype.lower()} {fullname}.", data)

    if action == "update_policy":
        previous = to_jsonable(c.w.policies.get_policy(stype, fullname, name))
        info = _policy_info(stype, fullname, name, body)
        mask = update_mask or ",".join(body)
        updated = c.w.policies.update_policy(stype, fullname, name, info, update_mask=mask)
        data = {
            "previous": previous,
            "policy": to_jsonable(updated),
            "update_mask": mask,
            "audit": audit_block(c, api_call=f"policies.update_policy {stype} {fullname} {name} mask={mask}"),
        }
        return ok(f"Updated ABAC policy {name!r} on {stype.lower()} {fullname}.", data)

    # delete_policy
    c.safety.check_protected("policy", name, None, operation="delete")
    c.safety.check_protected(stype.lower(), fullname, None, operation="delete a policy on")
    previous = to_jsonable(c.w.policies.get_policy(stype, fullname, name))
    c.w.policies.delete_policy(stype, fullname, name)
    data = {
        "deleted_policy": previous,
        "audit": audit_block(c, api_call=f"policies.delete_policy {stype} {fullname} {name}"),
    }
    return ok(f"Deleted ABAC policy {name!r} on {stype.lower()} {fullname}.", data)
