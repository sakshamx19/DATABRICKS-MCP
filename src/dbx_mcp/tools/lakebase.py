"""Lakebase (PostgreSQL on Databricks): manage_lakebase_database, manage_lakebase_branch,
manage_lakebase_sync, generate_lakebase_credential.

Two SDK services back these tools (verified against databricks-sdk 0.146):

* ``w.database`` - *provisioned* Lakebase: database instances, database catalogs (UC
  registration), synced database tables (reverse ETL) and credential generation. Instance
  creation returns a ``Wait``; everything else is synchronous.
* ``w.postgres`` - Lakebase *autoscaling*: projects, branches, compute endpoints, catalogs,
  synced tables and credential generation. Every write returns a long-running operation
  object exposing ``name()``; its state is read with ``w.postgres.get_operation(name)``.

Long-running work returns immediately with status ``pending`` (plus the operation name /
resource state) unless ``wait_seconds`` asks for a bounded wait.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Annotated, Any, Literal
from urllib.parse import quote

from databricks.sdk.common.types.fieldmask import FieldMask
from databricks.sdk.service import database as dbsvc
from databricks.sdk.service import postgres as pgsvc
from google.protobuf.duration_pb2 import Duration
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import (
    DESTRUCTIVE,
    EXECUTION,
    READ,
    WRITE,
    WRITE_SECURITY,
)
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import SafetyBlockedError, UnsupportedOperation, ValidationFailed
from dbx_mcp.utils.polling import clamp_wait
from dbx_mcp.utils.serialization import parse_sdk_object, pick, to_jsonable, wait_response

TOOLSET = "lakebase"

PG_PORT = 5432
DEFAULT_PG_DATABASE = "databricks_postgres"
_POLL_INTERVAL_SECONDS = 5.0

COST_WARNING = (
    "Lakebase compute is billed while it runs (provisioned capacity / autoscaling compute units). "
    "Confirm the size with the user and stop or delete resources that are no longer needed."
)

Kind = Annotated[
    Literal["provisioned", "autoscaling"],
    Field(
        description="provisioned: Lakebase database instances (w.database). autoscaling: Lakebase "
        "autoscaling projects/branches/endpoints (w.postgres)."
    ),
]
WaitSeconds = Annotated[
    int,
    Field(
        description="Seconds to wait for a long-running create/update/delete to finish. 0 (default) returns "
        "immediately with status 'pending'. Capped by DBX_MCP_MAX_WAIT_SECONDS and the tool timeout.",
        ge=0,
    ),
]
OperationName = Annotated[
    str | None,
    Field(description="Autoscaling operation name returned by a previous call (for action='get_operation')."),
]
UpdateMask = Annotated[
    str | None,
    Field(
        description="Comma-separated field paths to update. Default: derived from the keys of `spec` "
        "(autoscaling resources use 'spec.<field>' paths)."
    ),
]

_PENDING_INSTANCE_STATES = {"STARTING", "UPDATING", "DELETING", "FAILING_OVER"}
_PROVISIONING_SYNC_STATES = {
    "SYNCED_TABLE_PROVISIONING",
    "SYNCED_TABLE_PROVISIONING_INITIAL_SNAPSHOT",
    "SYNCED_TABLE_PROVISIONING_PIPELINE_RESOURCES",
}


# ----------------------------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------------------------

def _value(enum_or_none: Any) -> Any:
    return getattr(enum_or_none, "value", enum_or_none)


def _ts(value: Any) -> str | None:
    """protobuf Timestamp -> RFC3339 string (SDK Timestamps have no as_dict)."""
    if value is None:
        return None
    to_json = getattr(value, "ToJsonString", None)
    return to_json() if callable(to_json) else str(value)


def _last(name: str) -> str:
    return name.rstrip("/").rsplit("/", 1)[-1]


def _wait_budget(wait_seconds: int | None) -> int:
    return clamp_wait(ctx().settings, wait_seconds)[0]


def _track(resource_type: str, resource_id: str, name: str | None, tool_name: str, **metadata: Any) -> str | None:
    c = ctx()
    return c.manifest.safe_track(
        resource_type=resource_type,
        resource_id=resource_id,
        name=name,
        created_by_tool=tool_name,
        workspace_host=c.host,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def _no_spec_keys(spec: dict[str, Any] | None, keys: tuple[str, ...]) -> None:
    clash = [k for k in keys if k in (spec or {})]
    if clash:
        raise ValidationFailed(
            f"Field(s) {', '.join(clash)} must be passed as dedicated tool parameters, not in spec"
        )


# --- autoscaling resource names ---------------------------------------------------------------

def _project_name(project: str | None, action: str | None = None) -> str:
    require(project, "project", action)
    p = project.strip().strip("/")
    return p if p.startswith("projects/") else f"projects/{p}"


def _branch_name(project: str | None, branch: str | None, action: str | None = None) -> str:
    require(branch, "branch", action)
    b = branch.strip().strip("/")
    if b.startswith("projects/"):
        return b
    return f"{_project_name(project, action)}/branches/{b}"


def _endpoint_name(project: str | None, branch: str | None, endpoint: str | None, action: str | None = None) -> str:
    require(endpoint, "endpoint", action)
    e = endpoint.strip().strip("/")
    if e.startswith("projects/"):
        return e
    return f"{_branch_name(project, branch, action)}/endpoints/{e}"


def _pg_catalog_name(catalog: str) -> str:
    return catalog if catalog.startswith("catalogs/") else f"catalogs/{catalog}"


def _pg_synced_name(table: str) -> str:
    return table if table.startswith("synced_tables/") else f"synced_tables/{table}"


def _pg_mask(spec: dict[str, Any], *, nested: bool, override: str | None) -> FieldMask:
    """FieldMask for w.postgres updates. ``nested`` = spec holds the resource's inner ``spec`` fields."""
    if override:
        paths = [p.strip() for p in override.split(",") if p.strip()]
    elif nested:
        paths = [f"spec.{k}" for k in spec]
    else:
        paths = []
        for key, value in spec.items():
            if key == "spec" and isinstance(value, dict) and value:
                paths.extend(f"spec.{k}" for k in value)
            else:
                paths.append(key)
    if not paths:
        raise ValidationFailed("Nothing to update: provide at least one field in spec (or update_mask)")
    return FieldMask(field_mask=paths)


# --- long-running operations (w.postgres) -----------------------------------------------------

def _poll_operation(op_name: str, wait_seconds: int | None) -> pgsvc.Operation:
    w = ctx().w
    deadline = time.monotonic() + _wait_budget(wait_seconds)
    while True:
        operation = w.postgres.get_operation(name=op_name)
        remaining = deadline - time.monotonic()
        if operation.done or remaining <= 0:
            return operation
        time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))


def _operation_data(op_name: str | None, operation: pgsvc.Operation | None) -> dict[str, Any]:
    data: dict[str, Any] = {"name": op_name, "done": bool(operation and operation.done)}
    if operation is not None:
        if operation.metadata:
            data["metadata"] = operation.metadata
        if operation.error:
            data["error"] = {"error_code": _value(operation.error.error_code), "message": operation.error.message}
    return data


def _lro_response(
    op: Any,
    *,
    what: str,
    result_cls: type | None,
    wait_seconds: int | None,
    warnings: list[str | None] | None = None,
    poll_hint: str,
) -> ToolResponse:
    """Turn a w.postgres long-running operation into a ToolResponse (bounded wait)."""
    op_name = op.name()
    warnings = list(warnings or [])
    if not op_name:
        return ok(
            f"{what} submitted; the service returned no operation name to poll.",
            {"operation": {"name": None, "done": False}},
            status="pending",
            warnings=warnings,
            next_steps=[poll_hint],
        )
    operation = _poll_operation(op_name, wait_seconds)
    data: dict[str, Any] = {"operation": _operation_data(op_name, operation)}
    if operation.done and operation.error:
        message = operation.error.message or "unknown error"
        return ok(f"{what} failed: {message}", data, status="failed", warnings=warnings)
    if operation.done:
        if operation.response and result_cls is not None:
            data["result"] = result_cls.from_dict(operation.response)
        return ok(f"{what} completed.", data, warnings=warnings)
    return ok(
        f"{what} started and is still running (operation {op_name}).",
        data,
        status="pending",
        warnings=warnings,
        next_steps=[f"Poll with action='get_operation', operation_name='{op_name}'.", poll_hint],
    )


def _get_operation(operation_name: str | None) -> ToolResponse:
    require(operation_name, "operation_name", "get_operation")
    operation = ctx().w.postgres.get_operation(name=operation_name)
    state = "done" if operation.done else "running"
    if operation.done and operation.error:
        state = f"failed ({operation.error.message or 'unknown error'})"
    return ok(f"Operation {operation_name} is {state}.", operation)


# --- compact list views ------------------------------------------------------------------------

def _instance_summary(instance: dbsvc.DatabaseInstance) -> dict[str, Any]:
    return pick(
        to_jsonable(instance),
        [
            "name", "state", "capacity", "effective_capacity", "effective_node_count", "pg_version",
            "effective_stopped", "read_write_dns", "creator", "creation_time", "uid",
        ],
    )


def _project_summary(project: pgsvc.Project) -> dict[str, Any]:
    d = to_jsonable(project)
    status = d.get("status") or {}
    out = {
        "name": d.get("name"),
        "project_id": status.get("project_id") or d.get("project_id"),
        "display_name": status.get("display_name"),
        "pg_version": status.get("pg_version"),
        "owner": status.get("owner"),
        "default_branch": status.get("default_branch"),
        "create_time": d.get("create_time"),
        "delete_time": d.get("delete_time"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _branch_summary(branch: pgsvc.Branch) -> dict[str, Any]:
    d = to_jsonable(branch)
    status = d.get("status") or {}
    out = {
        "name": d.get("name"),
        "branch_id": status.get("branch_id") or d.get("branch_id"),
        "state": status.get("current_state"),
        "pending_state": status.get("pending_state"),
        "default": status.get("default"),
        "is_protected": status.get("is_protected"),
        "source_branch": status.get("source_branch"),
        "logical_size_bytes": status.get("logical_size_bytes"),
        "expire_time": status.get("expire_time"),
        "create_time": d.get("create_time"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _endpoint_summary(endpoint: pgsvc.Endpoint) -> dict[str, Any]:
    d = to_jsonable(endpoint)
    status = d.get("status") or {}
    out = {
        "name": d.get("name"),
        "endpoint_id": status.get("endpoint_id") or d.get("endpoint_id"),
        "endpoint_type": status.get("endpoint_type"),
        "state": status.get("current_state"),
        "autoscaling_limit_min_cu": status.get("autoscaling_limit_min_cu"),
        "autoscaling_limit_max_cu": status.get("autoscaling_limit_max_cu"),
        "disabled": status.get("disabled"),
        "host": (status.get("hosts") or {}).get("host"),
        "last_active_time": status.get("last_active_time"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _catalog_summary(catalog: dbsvc.DatabaseCatalog) -> dict[str, Any]:
    return pick(to_jsonable(catalog), ["name", "database_instance_name", "database_name", "uid"])


def _synced_summary(table: dbsvc.SyncedDatabaseTable) -> dict[str, Any]:
    d = to_jsonable(table)
    spec = d.get("spec") or {}
    status = d.get("data_synchronization_status") or {}
    out = {
        "name": d.get("name"),
        "database_instance_name": d.get("effective_database_instance_name") or d.get("database_instance_name"),
        "logical_database_name": d.get("effective_logical_database_name") or d.get("logical_database_name"),
        "source_table_full_name": spec.get("source_table_full_name"),
        "scheduling_policy": spec.get("scheduling_policy"),
        "state": status.get("detailed_state"),
        "pipeline_id": status.get("pipeline_id"),
        "unity_catalog_provisioning_state": d.get("unity_catalog_provisioning_state"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _instance_tags(instance: dbsvc.DatabaseInstance) -> dict[str, str]:
    tags = instance.effective_custom_tags or instance.custom_tags or []
    return {t.key: t.value or "" for t in tags if t.key}


def _project_tags(project: pgsvc.Project) -> dict[str, str]:
    tags = (project.status.custom_tags if project.status else None) or (
        project.spec.custom_tags if project.spec else None
    ) or []
    return {t.key: t.value or "" for t in tags if t.key}


# ==============================================================================================
# manage_lakebase_database
# ==============================================================================================

_DB_SAFETY = {
    "list": READ,
    "get": READ,
    "create": WRITE,
    "update": WRITE,
    "delete": DESTRUCTIVE,
    "undelete": WRITE,
    "get_operation": READ,
    "list_catalogs": READ,
    "get_catalog": READ,
    "create_catalog": WRITE,
    "delete_catalog": DESTRUCTIVE,
}


def _protect_instance(instance: dbsvc.DatabaseInstance, operation: str) -> None:
    ctx().safety.check_protected(
        "Lakebase database instance", instance.name, _instance_tags(instance), operation=operation
    )


def _protect_project(project: pgsvc.Project, full_name: str, operation: str) -> None:
    safety = ctx().safety
    tags = _project_tags(project)
    safety.check_protected("Lakebase project", _last(full_name), tags, operation=operation)
    display = project.status.display_name if project.status else None
    if display:
        safety.check_protected("Lakebase project", display, operation=operation)


def _db_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    kind = args.get("kind") or "provisioned"
    name = args.get("name")
    w = ctx().w

    if action == "delete" and kind == "provisioned":
        require(name, "name", action)
        instance = w.database.get_database_instance(name=name)
        _protect_instance(instance, "delete")
        children = [r.name for r in (instance.child_instance_refs or []) if r.name]
        warnings = [
            "Deleting a database instance permanently removes the Postgres server and ALL databases and "
            "data on it. Synced tables and database catalogs that use it will stop working.",
        ]
        if children and not args.get("force"):
            warnings.append(
                f"The instance has descendant (point-in-time) instances {children}; the delete will be rejected "
                "unless force=true, which deletes them too."
            )
        elif children:
            warnings.append(f"force=true also deletes the descendant instances {children}.")
        return PlanInfo(
            description=f"Permanently delete Lakebase database instance '{instance.name}'.",
            target={"kind": "provisioned", "name": instance.name, "uid": instance.uid},
            details={
                "state": _value(instance.state),
                "capacity": instance.effective_capacity or instance.capacity,
                "node_count": instance.effective_node_count or instance.node_count,
                "pg_version": instance.pg_version,
                "read_write_dns": instance.read_write_dns,
                "creator": instance.creator,
                "child_instances": children,
                "force": bool(args.get("force")),
            },
            warnings=warnings,
            reversible=False,
        )

    if action == "delete" and kind == "autoscaling":
        full = _project_name(name, action)
        project = w.postgres.get_project(name=full)
        _protect_project(project, full, "delete")
        purge = bool(args.get("purge"))
        status = project.status
        return PlanInfo(
            description=f"{'Permanently delete (purge)' if purge else 'Soft-delete'} Lakebase autoscaling "
            f"project '{full}' including all of its branches, compute endpoints and databases.",
            target={"kind": "autoscaling", "name": full},
            details={
                "display_name": status.display_name if status else None,
                "owner": status.owner if status else None,
                "default_branch": status.default_branch if status else None,
                "pg_version": status.pg_version if status else None,
                "synthetic_storage_size_bytes": status.synthetic_storage_size_bytes if status else None,
                "purge": purge,
            },
            warnings=[
                "All branches, endpoints and data in the project become unavailable; connected applications fail.",
                "purge=true is a hard delete and cannot be undone."
                if purge
                else "Soft delete: the project can be restored with action='undelete' until it is purged.",
            ],
            reversible=not purge,
        )

    if action == "delete_catalog":
        catalog_name = require(args.get("catalog_name"), "catalog_name", action)
        ctx().safety.check_protected("Lakebase database catalog", catalog_name, operation="delete")
        if kind == "provisioned":
            catalog = w.database.get_database_catalog(name=catalog_name)
            details = {
                "database_instance_name": catalog.database_instance_name,
                "database_name": catalog.database_name,
                "uid": catalog.uid,
            }
        else:
            pg_catalog = w.postgres.get_catalog(name=_pg_catalog_name(catalog_name))
            status = pg_catalog.status
            details = {
                "project": status.project if status else None,
                "branch": status.branch if status else None,
                "postgres_database": status.postgres_database if status else None,
            }
        return PlanInfo(
            description=f"Remove the Unity Catalog registration '{catalog_name}' of a Lakebase Postgres database.",
            target={"kind": kind, "catalog_name": catalog_name},
            details=details,
            warnings=[
                "Unity Catalog objects, grants and queries that reference this catalog stop working.",
                "Verify in the Lakebase documentation whether the underlying Postgres database is retained "
                "before relying on it; this tool does not drop it explicitly.",
            ],
            reversible=False,
        )

    if action in ("create", "update", "undelete"):
        what = "database instance" if kind == "provisioned" else "autoscaling project"
        return PlanInfo(
            description=f"{action.capitalize()} Lakebase {what} '{name}'.",
            target={"kind": kind, "name": name},
            details={"spec": args.get("spec"), "update_mask": args.get("update_mask")},
            warnings=[COST_WARNING] if action != "undelete" else [],
            reversible=True,
        )
    return None


@tool(toolset=TOOLSET, title="Manage Lakebase databases", safety=_DB_SAFETY, preview=_db_preview)
def manage_lakebase_database(
    action: Annotated[
        Literal[
            "list", "get", "create", "update", "delete", "undelete", "get_operation",
            "list_catalogs", "get_catalog", "create_catalog", "delete_catalog",
        ],
        Field(description="Operation to perform. *_catalog actions register/unregister a Lakebase Postgres "
              "database as a Unity Catalog catalog."),
    ],
    kind: Kind = "provisioned",
    name: Annotated[
        str | None,
        Field(description="provisioned: database instance name. autoscaling: project id or 'projects/<id>'."),
    ] = None,
    spec: Spec = None,
    update_mask: UpdateMask = None,
    force: Annotated[bool, Field(description="provisioned delete: also delete descendant point-in-time "
                                 "instances (otherwise the delete is rejected if any exist).")] = False,
    purge: Annotated[bool, Field(description="autoscaling delete: hard delete (irreversible). Default is a "
                                 "soft delete restorable with action='undelete'.")] = False,
    show_deleted: Annotated[bool, Field(description="autoscaling list: include soft-deleted projects.")] = False,
    catalog_name: Annotated[str | None, Field(description="Unity Catalog catalog name for *_catalog actions.")] = None,
    database_name: Annotated[str | None, Field(description="create_catalog: Postgres database to register.")] = None,
    branch: Annotated[str | None, Field(description="autoscaling create_catalog: branch id or full branch name "
                                       "(default: the project's default branch).")] = None,
    create_database_if_missing: Annotated[bool, Field(description="create_catalog: create the Postgres "
                                                      "database if it does not exist.")] = False,
    operation_name: OperationName = None,
    wait_seconds: WaitSeconds = 0,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Lakebase (Postgres) databases.

    kind='provisioned' manages database instances: list, get, create (spec = DatabaseInstance fields,
    e.g. {"capacity": "CU_1"}), update (spec = fields to change, e.g. {"stopped": true} or
    {"capacity": "CU_2"}), delete (force=true also removes point-in-time children).
    kind='autoscaling' manages projects: list, get, create (spec = Project fields, e.g.
    {"spec": {"display_name": "My app", "pg_version": 17}}), update (e.g. {"spec": {"display_name": "x"}}),
    delete (soft unless purge=true), undelete, get_operation.
    Catalog actions register a Postgres database in Unity Catalog: list_catalogs (provisioned, name =
    instance), get_catalog, create_catalog (catalog_name, database_name, name/branch), delete_catalog.
    Compute is billed; long-running work returns status 'pending' unless wait_seconds is set."""
    if action == "get_operation":
        return _get_operation(operation_name)
    if kind == "provisioned":
        return _provisioned_database(
            action, name, spec, update_mask, force, catalog_name, database_name,
            create_database_if_missing, wait_seconds, page_size, page_token,
        )
    return _autoscaling_project(
        action, name, spec, update_mask, purge, show_deleted, catalog_name, database_name, branch,
        create_database_if_missing, wait_seconds, page_size, page_token,
    )


def _provisioned_database(
    action: str,
    name: str | None,
    spec: dict[str, Any] | None,
    update_mask: str | None,
    force: bool,
    catalog_name: str | None,
    database_name: str | None,
    create_database_if_missing: bool,
    wait_seconds: int,
    page_size: int | None,
    page_token: str | None,
) -> ToolResponse:
    w = ctx().w
    tool_name = "manage_lakebase_database"

    if action == "list":
        return paged_response(
            "Lakebase database instances", w.database.list_database_instances(), page_size, page_token,
            _instance_summary,
        )

    if action == "get":
        require(name, "name", action)
        instance = w.database.get_database_instance(name=name)
        return ok(f"Database instance '{instance.name}' is {_value(instance.state) or 'in an unknown state'}.", instance)

    if action == "create":
        require(name, "name", action)
        _no_spec_keys(spec, ("name",))
        instance = parse_sdk_object(dbsvc.DatabaseInstance, {**(spec or {}), "name": name})
        waiter = w.database.create_database_instance(database_instance=instance)
        result = wait_response(waiter)
        warnings: list[str | None] = [COST_WARNING]
        budget = _wait_budget(wait_seconds)
        if budget > 0:
            try:
                result = w.database.wait_get_database_instance_database_available(
                    name=name, timeout=timedelta(seconds=budget)
                )
            except TimeoutError:
                warnings.append(f"Instance was not AVAILABLE after waiting {budget}s; it is still provisioning.")
        warnings.append(_track("lakebase_instance", name, name, tool_name,
                               capacity=getattr(result, "capacity", None) or instance.capacity))
        state = _value(getattr(result, "state", None))
        done = state == "AVAILABLE"
        return ok(
            f"Database instance '{name}' {'is AVAILABLE' if done else f'is being created (state={state})'}.",
            result,
            status="success" if done else "pending",
            warnings=warnings,
            next_steps=[] if done else [f"Poll with action='get', name='{name}' until state is AVAILABLE."],
        )

    if action == "update":
        require(name, "name", action)
        require(spec, "spec", action)
        _no_spec_keys(spec, ("name",))
        mask = update_mask or ",".join(spec)
        if not mask:
            raise ValidationFailed("Nothing to update: provide at least one field in spec")
        instance = parse_sdk_object(dbsvc.DatabaseInstance, {**spec, "name": name})
        result = w.database.update_database_instance(name=name, database_instance=instance, update_mask=mask)
        state = _value(result.state)
        pending = state in _PENDING_INSTANCE_STATES
        return ok(
            f"Updated database instance '{name}' ({mask})" + (f"; state is {state}." if state else "."),
            result,
            status="pending" if pending else "success",
            warnings=[COST_WARNING],
            next_steps=[f"Poll with action='get', name='{name}'."] if pending else [],
        )

    if action == "delete":
        require(name, "name", action)
        instance = w.database.get_database_instance(name=name)
        _protect_instance(instance, "delete")
        w.database.delete_database_instance(name=name, force=True if force else None)
        ctx().manifest.safe_untrack("lakebase_instance", name)
        return ok(
            f"Deletion of database instance '{name}' requested.",
            {"name": name, "force": force, "previous_state": _value(instance.state)},
            status="pending",
            next_steps=[f"action='get', name='{name}' returns NOT_FOUND once deletion has finished."],
        )

    if action == "undelete":
        raise UnsupportedOperation(
            "Provisioned database instances cannot be undeleted (deletion is permanent). "
            "undelete is only available for kind='autoscaling' projects."
        )

    if action == "list_catalogs":
        require(name, "name", action)
        return paged_response(
            "database catalogs", w.database.list_database_catalogs(instance_name=name), page_size, page_token,
            _catalog_summary,
        )

    if action == "get_catalog":
        require(catalog_name, "catalog_name", action)
        return ok(f"Database catalog '{catalog_name}'.", w.database.get_database_catalog(name=catalog_name))

    if action == "create_catalog":
        require(catalog_name, "catalog_name", action)
        require(name, "name", action)
        require(database_name, "database_name", action)
        catalog = dbsvc.DatabaseCatalog(
            name=catalog_name,
            database_instance_name=name,
            database_name=database_name,
            create_database_if_not_exists=True if create_database_if_missing else None,
        )
        result = w.database.create_database_catalog(catalog=catalog)
        warning = _track("lakebase_database_catalog", catalog_name, catalog_name, tool_name, instance=name)
        return ok(
            f"Registered Postgres database '{database_name}' on instance '{name}' as catalog '{catalog_name}'.",
            result,
            warnings=[warning],
        )

    if action == "delete_catalog":
        require(catalog_name, "catalog_name", action)
        ctx().safety.check_protected("Lakebase database catalog", catalog_name, operation="delete")
        w.database.delete_database_catalog(name=catalog_name)
        ctx().manifest.safe_untrack("lakebase_database_catalog", catalog_name)
        return ok(f"Deleted database catalog '{catalog_name}'.", {"catalog_name": catalog_name})

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover - guarded by safety map


def _autoscaling_project(
    action: str,
    name: str | None,
    spec: dict[str, Any] | None,
    update_mask: str | None,
    purge: bool,
    show_deleted: bool,
    catalog_name: str | None,
    database_name: str | None,
    branch: str | None,
    create_database_if_missing: bool,
    wait_seconds: int,
    page_size: int | None,
    page_token: str | None,
) -> ToolResponse:
    w = ctx().w
    tool_name = "manage_lakebase_database"

    if action == "list":
        return paged_response(
            "Lakebase projects", w.postgres.list_projects(show_deleted=True if show_deleted else None),
            page_size, page_token, _project_summary,
        )

    if action == "get":
        full = _project_name(name, action)
        project = w.postgres.get_project(name=full)
        return ok(f"Lakebase project '{full}'.", project)

    if action == "create":
        full = _project_name(name, action)
        _no_spec_keys(spec, ("name", "project_id"))
        project = parse_sdk_object(pgsvc.Project, dict(spec or {}))
        op = w.postgres.create_project(project=project, project_id=_last(full))
        warning = _track("lakebase_project", full, _last(full), tool_name)
        return _lro_response(
            op, what=f"Creating project '{full}'", result_cls=pgsvc.Project, wait_seconds=wait_seconds,
            warnings=[COST_WARNING, warning], poll_hint=f"Or action='get', kind='autoscaling', name='{full}'.",
        )

    if action == "update":
        full = _project_name(name, action)
        require(spec, "spec", action)
        _no_spec_keys(spec, ("name", "project_id"))
        mask = _pg_mask(spec, nested=False, override=update_mask)
        project = parse_sdk_object(pgsvc.Project, {**spec, "name": full})
        op = w.postgres.update_project(name=full, project=project, update_mask=mask)
        return _lro_response(
            op, what=f"Updating project '{full}' ({mask.ToJsonString()})", result_cls=pgsvc.Project,
            wait_seconds=wait_seconds, warnings=[COST_WARNING],
            poll_hint=f"Or action='get', kind='autoscaling', name='{full}'.",
        )

    if action == "delete":
        full = _project_name(name, action)
        project = w.postgres.get_project(name=full)
        _protect_project(project, full, "delete")
        op = w.postgres.delete_project(name=full, purge=True if purge else None)
        ctx().manifest.safe_untrack("lakebase_project", full)
        return _lro_response(
            op, what=f"{'Purging' if purge else 'Deleting'} project '{full}'", result_cls=None,
            wait_seconds=wait_seconds,
            warnings=[None if purge else "Soft delete: restore with action='undelete' until the project is purged."],
            poll_hint="Deletion is complete when the operation reports done=true.",
        )

    if action == "undelete":
        full = _project_name(name, action)
        op = w.postgres.undelete_project(name=full)
        return _lro_response(
            op, what=f"Restoring project '{full}'", result_cls=pgsvc.Project, wait_seconds=wait_seconds,
            poll_hint=f"Or action='get', kind='autoscaling', name='{full}'.",
        )

    if action == "list_catalogs":
        raise UnsupportedOperation(
            "The Lakebase autoscaling API (w.postgres) has no list operation for catalogs.",
            hint="Use action='get_catalog' with a known catalog_name, or list Unity Catalog catalogs instead.",
        )

    if action == "get_catalog":
        require(catalog_name, "catalog_name", action)
        full_catalog = _pg_catalog_name(catalog_name)
        return ok(f"Lakebase catalog '{full_catalog}'.", w.postgres.get_catalog(name=full_catalog))

    if action == "create_catalog":
        require(catalog_name, "catalog_name", action)
        require(database_name, "database_name", action)
        if branch:
            branch_full = _branch_name(name, branch, action)
        else:
            full = _project_name(name, action)
            project = w.postgres.get_project(name=full)
            default_branch = project.status.default_branch if project.status else None
            if not default_branch:
                raise ValidationFailed(
                    f"Project '{full}' reports no default branch; pass `branch` explicitly."
                )
            branch_full = _branch_name(full, default_branch, action)
        catalog = pgsvc.Catalog(
            spec=pgsvc.CatalogCatalogSpec(
                postgres_database=database_name,
                branch=branch_full,
                create_database_if_missing=True if create_database_if_missing else None,
            )
        )
        op = w.postgres.create_catalog(catalog=catalog, catalog_id=catalog_name)
        warning = _track("lakebase_catalog", _pg_catalog_name(catalog_name), catalog_name, tool_name,
                         branch=branch_full)
        return _lro_response(
            op, what=f"Registering '{database_name}' on '{branch_full}' as catalog '{catalog_name}'",
            result_cls=pgsvc.Catalog, wait_seconds=wait_seconds, warnings=[warning],
            poll_hint=f"Or action='get_catalog', kind='autoscaling', catalog_name='{catalog_name}'.",
        )

    if action == "delete_catalog":
        require(catalog_name, "catalog_name", action)
        ctx().safety.check_protected("Lakebase database catalog", catalog_name, operation="delete")
        full_catalog = _pg_catalog_name(catalog_name)
        op = w.postgres.delete_catalog(name=full_catalog)
        ctx().manifest.safe_untrack("lakebase_catalog", full_catalog)
        return _lro_response(
            op, what=f"Deleting catalog '{full_catalog}'", result_cls=None, wait_seconds=wait_seconds,
            poll_hint="Deletion is complete when the operation reports done=true.",
        )

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover - guarded by safety map


# ==============================================================================================
# manage_lakebase_branch (autoscaling only)
# ==============================================================================================

_BRANCH_SAFETY = {
    "list": READ,
    "get": READ,
    "create": WRITE,
    "update": WRITE,
    "delete": DESTRUCTIVE,
    "undelete": WRITE,
    "list_endpoints": READ,
    "get_endpoint": READ,
    "create_endpoint": WRITE,
    "update_endpoint": WRITE,
    "delete_endpoint": DESTRUCTIVE,
    "get_operation": READ,
}


def _check_branch_delete(branch: pgsvc.Branch, full: str, allow_default_branch: bool) -> None:
    ctx().safety.check_protected("Lakebase branch", _last(full), operation="delete")
    if branch.status and branch.status.default and not allow_default_branch:
        raise SafetyBlockedError(
            f"Refusing to delete '{full}': it is the project's default (primary) branch.",
            hint="Make another branch the default first, or pass allow_default_branch=true if this is intended.",
        )


def _branch_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    w = ctx().w
    if action == "delete":
        full = _branch_name(args.get("project"), args.get("branch"), action)
        branch = w.postgres.get_branch(name=full)
        _check_branch_delete(branch, full, bool(args.get("allow_default_branch")))
        purge = bool(args.get("purge"))
        status = branch.status
        warnings = [
            "The branch's databases and data become unavailable and its compute endpoints are removed.",
            "purge=true is a hard delete and cannot be undone."
            if purge
            else "Soft delete: restore with action='undelete' until the branch is purged.",
        ]
        if status and status.is_protected:
            warnings.append("The branch is protected; the service will reject the delete until it is unprotected.")
        if status and status.default:
            warnings.append("This is the project's DEFAULT branch (allow_default_branch=true was given).")
        return PlanInfo(
            description=f"{'Permanently delete (purge)' if purge else 'Soft-delete'} Lakebase branch '{full}'.",
            target={"name": full},
            details={
                "state": _value(status.current_state) if status else None,
                "default": status.default if status else None,
                "is_protected": status.is_protected if status else None,
                "source_branch": status.source_branch if status else None,
                "logical_size_bytes": status.logical_size_bytes if status else None,
                "purge": purge,
            },
            warnings=warnings,
            reversible=not purge,
        )
    if action == "delete_endpoint":
        full = _endpoint_name(args.get("project"), args.get("branch"), args.get("endpoint"), action)
        ctx().safety.check_protected("Lakebase endpoint", _last(full), operation="delete")
        endpoint = w.postgres.get_endpoint(name=full)
        summary = _endpoint_summary(endpoint)
        return PlanInfo(
            description=f"Delete Lakebase compute endpoint '{full}'.",
            target={"name": full},
            details=summary,
            warnings=[
                "Open connections through this endpoint are dropped and its host name stops resolving. "
                "Branch data is not deleted; a new endpoint can be created on the branch.",
            ],
            reversible=False,
        )
    if action in ("create", "update", "create_endpoint", "update_endpoint", "undelete"):
        target = args.get("endpoint") if "endpoint" in action else args.get("branch")
        return PlanInfo(
            description=f"{action.replace('_', ' ').capitalize()} '{target}' in project '{args.get('project')}'.",
            target={"project": args.get("project"), "branch": args.get("branch"), "endpoint": args.get("endpoint")},
            details={k: args.get(k) for k in ("spec", "update_mask", "source_branch", "source_branch_time",
                                              "source_branch_lsn") if args.get(k) is not None},
            warnings=[COST_WARNING] if action != "undelete" else [],
            reversible=True,
        )
    return None


@tool(toolset=TOOLSET, title="Manage Lakebase branches", safety=_BRANCH_SAFETY, preview=_branch_preview)
def manage_lakebase_branch(
    action: Annotated[
        Literal[
            "list", "get", "create", "update", "delete", "undelete",
            "list_endpoints", "get_endpoint", "create_endpoint", "update_endpoint", "delete_endpoint",
            "get_operation",
        ],
        Field(description="Branch lifecycle, plus compute endpoints (*_endpoint) of a branch."),
    ],
    project: Annotated[str | None, Field(description="Project id or 'projects/<id>'.")] = None,
    branch: Annotated[str | None, Field(description="Branch id or full name 'projects/<p>/branches/<b>'.")] = None,
    endpoint: Annotated[str | None, Field(description="Endpoint id or full endpoint name.")] = None,
    spec: Annotated[
        dict[str, Any] | None,
        Field(description="BranchSpec fields (create/update, e.g. {\"ttl\": \"86400s\"}, {\"no_expiry\": true}, "
              "{\"is_protected\": true}) or EndpointSpec fields (create_endpoint/update_endpoint, e.g. "
              "{\"endpoint_type\": \"ENDPOINT_TYPE_READ_WRITE\", \"autoscaling_limit_min_cu\": 0.5, "
              "\"autoscaling_limit_max_cu\": 2}). Unknown fields are rejected."),
    ] = None,
    update_mask: UpdateMask = None,
    source_branch: Annotated[str | None, Field(description="create: parent branch id/name to branch from "
                                               "(default: the project's default branch).")] = None,
    source_branch_time: Annotated[str | None, Field(description="create: point in time of the parent branch "
                                                    "(RFC3339, e.g. 2025-01-31T12:00:00Z).")] = None,
    source_branch_lsn: Annotated[str | None, Field(description="create: Postgres LSN of the parent branch to "
                                                   "branch from.")] = None,
    purge: Annotated[bool, Field(description="delete: hard delete (irreversible). Default is a soft delete "
                                 "restorable with action='undelete'.")] = False,
    allow_default_branch: Annotated[bool, Field(description="delete: permit deleting the project's default "
                                                "branch (refused otherwise).")] = False,
    show_deleted: Annotated[bool, Field(description="list: include soft-deleted branches.")] = False,
    operation_name: OperationName = None,
    wait_seconds: WaitSeconds = 0,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Lakebase autoscaling branches (copy-on-write Postgres branches) and their compute endpoints.

    Branch actions: list (project), get, create (project, branch id, optional source_branch,
    source_branch_time for point-in-time, source_branch_lsn, spec), update (spec, e.g.
    {"is_protected": true}), delete (soft unless purge=true; the default branch is refused unless
    allow_default_branch=true), undelete.
    Endpoint actions: list_endpoints, get_endpoint, create_endpoint (endpoint id + spec with
    endpoint_type), update_endpoint (e.g. CU limits, {"disabled": true}), delete_endpoint.
    get_operation polls a long-running operation. Writes return status 'pending' unless wait_seconds."""
    w = ctx().w
    tool_name = "manage_lakebase_branch"

    if action == "get_operation":
        return _get_operation(operation_name)

    if action == "list":
        parent = _project_name(project, action)
        return paged_response(
            "branches", w.postgres.list_branches(parent=parent, show_deleted=True if show_deleted else None),
            page_size, page_token, _branch_summary,
        )

    if action == "get":
        full = _branch_name(project, branch, action)
        result = w.postgres.get_branch(name=full)
        state = _value(result.status.current_state) if result.status else None
        return ok(f"Branch '{full}'" + (f" is {state}." if state else "."), result)

    if action == "create":
        parent = _project_name(project, action)
        require(branch, "branch", action)
        if branch.startswith("projects/"):
            raise ValidationFailed("For create, pass the new branch id (e.g. 'dev'), not a full resource name")
        body: dict[str, Any] = dict(spec or {})
        dedicated = {"source_branch": source_branch, "source_branch_time": source_branch_time,
                     "source_branch_lsn": source_branch_lsn}
        _no_spec_keys(body, tuple(k for k, v in dedicated.items() if v))
        parent_branch = source_branch or body.get("source_branch")
        if not parent_branch:
            proj = w.postgres.get_project(name=parent)
            parent_branch = proj.status.default_branch if proj.status else None
            if not parent_branch:
                raise ValidationFailed(f"Project '{parent}' reports no default branch; pass source_branch.")
        body["source_branch"] = _branch_name(parent, parent_branch, action)
        if source_branch_time:
            body["source_branch_time"] = source_branch_time
        if source_branch_lsn:
            body["source_branch_lsn"] = source_branch_lsn
        new_branch = parse_sdk_object(pgsvc.Branch, {"spec": body})
        op = w.postgres.create_branch(parent=parent, branch=new_branch, branch_id=branch)
        full = f"{parent}/branches/{branch}"
        warning = _track("lakebase_branch", full, branch, tool_name, source_branch=body["source_branch"])
        return _lro_response(
            op, what=f"Creating branch '{full}' from '{body['source_branch']}'", result_cls=pgsvc.Branch,
            wait_seconds=wait_seconds, warnings=[warning],
            poll_hint=f"Or action='get', project='{project}', branch='{branch}'.",
        )

    if action == "update":
        full = _branch_name(project, branch, action)
        require(spec, "spec", action)
        mask = _pg_mask(spec, nested=True, override=update_mask)
        updated = parse_sdk_object(pgsvc.Branch, {"name": full, "spec": spec})
        op = w.postgres.update_branch(name=full, branch=updated, update_mask=mask)
        return _lro_response(
            op, what=f"Updating branch '{full}' ({mask.ToJsonString()})", result_cls=pgsvc.Branch,
            wait_seconds=wait_seconds, poll_hint=f"Or action='get', branch='{full}'.",
        )

    if action == "delete":
        full = _branch_name(project, branch, action)
        current = w.postgres.get_branch(name=full)
        _check_branch_delete(current, full, allow_default_branch)
        op = w.postgres.delete_branch(name=full, purge=True if purge else None)
        ctx().manifest.safe_untrack("lakebase_branch", full)
        return _lro_response(
            op, what=f"{'Purging' if purge else 'Deleting'} branch '{full}'", result_cls=None,
            wait_seconds=wait_seconds,
            warnings=[None if purge else "Soft delete: restore with action='undelete' until the branch is purged."],
            poll_hint="Deletion is complete when the operation reports done=true.",
        )

    if action == "undelete":
        full = _branch_name(project, branch, action)
        op = w.postgres.undelete_branch(name=full)
        return _lro_response(
            op, what=f"Restoring branch '{full}'", result_cls=pgsvc.Branch, wait_seconds=wait_seconds,
            poll_hint=f"Or action='get', branch='{full}'.",
        )

    if action == "list_endpoints":
        parent = _branch_name(project, branch, action)
        return paged_response(
            "endpoints", w.postgres.list_endpoints(parent=parent), page_size, page_token, _endpoint_summary
        )

    if action == "get_endpoint":
        full = _endpoint_name(project, branch, endpoint, action)
        result = w.postgres.get_endpoint(name=full)
        summary = _endpoint_summary(result)
        return ok(
            f"Endpoint '{full}'" + (f" is {summary['state']}." if summary.get("state") else "."), result
        )

    if action == "create_endpoint":
        parent = _branch_name(project, branch, action)
        require(endpoint, "endpoint", action)
        if endpoint.startswith("projects/"):
            raise ValidationFailed("For create_endpoint, pass the new endpoint id, not a full resource name")
        require(spec, "spec", action)
        new_endpoint = parse_sdk_object(pgsvc.Endpoint, {"spec": spec})
        op = w.postgres.create_endpoint(parent=parent, endpoint=new_endpoint, endpoint_id=endpoint)
        full = f"{parent}/endpoints/{endpoint}"
        warning = _track("lakebase_endpoint", full, endpoint, tool_name)
        return _lro_response(
            op, what=f"Creating endpoint '{full}'", result_cls=pgsvc.Endpoint, wait_seconds=wait_seconds,
            warnings=[COST_WARNING, warning], poll_hint=f"Or action='get_endpoint', endpoint='{full}'.",
        )

    if action == "update_endpoint":
        full = _endpoint_name(project, branch, endpoint, action)
        require(spec, "spec", action)
        mask = _pg_mask(spec, nested=True, override=update_mask)
        updated = parse_sdk_object(pgsvc.Endpoint, {"name": full, "spec": spec})
        op = w.postgres.update_endpoint(name=full, endpoint=updated, update_mask=mask)
        return _lro_response(
            op, what=f"Updating endpoint '{full}' ({mask.ToJsonString()})", result_cls=pgsvc.Endpoint,
            wait_seconds=wait_seconds, warnings=[COST_WARNING],
            poll_hint=f"Or action='get_endpoint', endpoint='{full}'.",
        )

    if action == "delete_endpoint":
        full = _endpoint_name(project, branch, endpoint, action)
        ctx().safety.check_protected("Lakebase endpoint", _last(full), operation="delete")
        op = w.postgres.delete_endpoint(name=full)
        ctx().manifest.safe_untrack("lakebase_endpoint", full)
        return _lro_response(
            op, what=f"Deleting endpoint '{full}'", result_cls=None, wait_seconds=wait_seconds,
            poll_hint="Deletion is complete when the operation reports done=true.",
        )

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover - guarded by safety map


# ==============================================================================================
# manage_lakebase_sync (synced tables: Delta -> Lakebase)
# ==============================================================================================

_SYNC_SAFETY = {
    "list": READ,
    "get": READ,
    "create": WRITE,
    "update": WRITE,
    "delete": DESTRUCTIVE,
    "trigger": EXECUTION,
    "get_operation": READ,
}

_SYNC_UPDATE_UNSUPPORTED = (
    "Updating a synced table is not supported: the provisioned API marks update_synced_database_table as "
    "'currently unimplemented' and the autoscaling API (w.postgres) has no update operation."
)


def _sync_info(kind: str, table_name: str) -> dict[str, Any]:
    """Fetch a synced table and return the facts needed by preview/trigger/delete."""
    w = ctx().w
    if kind == "provisioned":
        table = w.database.get_synced_database_table(name=table_name)
        status = table.data_synchronization_status
        spec = table.spec
        return {
            "object": table,
            "name": table.name,
            "source_table_full_name": spec.source_table_full_name if spec else None,
            "scheduling_policy": _value(spec.scheduling_policy) if spec else None,
            "state": _value(status.detailed_state) if status else None,
            "pipeline_id": status.pipeline_id if status else None,
            "database_instance_name": table.effective_database_instance_name or table.database_instance_name,
            "logical_database_name": table.effective_logical_database_name or table.logical_database_name,
        }
    table = w.postgres.get_synced_table(name=_pg_synced_name(table_name))
    status = table.status
    spec = table.spec
    return {
        "object": table,
        "name": table.name,
        "source_table_full_name": spec.source_table_full_name if spec else None,
        "scheduling_policy": _value(spec.scheduling_policy) if spec else None,
        "state": _value(status.detailed_state) if status else None,
        "pipeline_id": status.pipeline_id if status else None,
        "branch": spec.branch if spec else None,
        "postgres_database": spec.postgres_database if spec else None,
    }


def _sync_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    kind = args.get("kind") or "provisioned"
    if action == "update":
        raise UnsupportedOperation(_SYNC_UPDATE_UNSUPPORTED)
    if action == "delete":
        table_name = require(args.get("table_name"), "table_name", action)
        ctx().safety.check_protected("Lakebase synced table", table_name, operation="delete")
        info = _sync_info(kind, table_name)
        details = {k: v for k, v in info.items() if k != "object"}
        purge_data = bool(args.get("purge_data"))
        warnings = ["Synchronization from the source Delta table stops and the synced table is removed from "
                    "Unity Catalog. The source Delta table is not affected."]
        if kind == "provisioned":
            details["purge_data"] = purge_data
            warnings.append(
                "purge_data=true also DROPS the Postgres table and its data."
                if purge_data
                else "The Postgres table is kept (purge_data=false) but no longer updated."
            )
        return PlanInfo(
            description=f"Delete synced table '{table_name}' ({kind}).",
            target={"kind": kind, "table_name": table_name},
            details=details,
            warnings=warnings,
            reversible=False,
        )
    if action == "trigger":
        table_name = require(args.get("table_name"), "table_name", action)
        info = _sync_info(kind, table_name)
        return PlanInfo(
            description=f"Start a sync of '{table_name}' by starting an update of its pipeline "
            f"{info.get('pipeline_id')!r}.",
            target={"kind": kind, "table_name": table_name, "pipeline_id": info.get("pipeline_id")},
            details={k: v for k, v in info.items() if k != "object"},
            warnings=["Starting the pipeline consumes compute."],
            reversible=None,
        )
    if action == "create":
        return PlanInfo(
            description=f"Create synced table '{args.get('table_name')}' ({kind}) that continuously or "
            "periodically copies a Delta table into Lakebase Postgres.",
            target={"kind": kind, "table_name": args.get("table_name")},
            details={k: args.get(k) for k in ("instance_name", "logical_database_name", "spec")
                     if args.get(k) is not None},
            warnings=["The sync runs on a managed pipeline that consumes compute (continuously for CONTINUOUS)."],
            reversible=True,
        )
    return None


@tool(toolset=TOOLSET, title="Manage Lakebase synced tables", safety=_SYNC_SAFETY, preview=_sync_preview)
def manage_lakebase_sync(
    action: Annotated[
        Literal["list", "get", "create", "update", "delete", "trigger", "get_operation"],
        Field(description="Synced table operation. trigger starts a sync for TRIGGERED/SNAPSHOT policies."),
    ],
    kind: Kind = "provisioned",
    table_name: Annotated[
        str | None,
        Field(description="Full Unity Catalog name of the synced table: catalog.schema.table."),
    ] = None,
    instance_name: Annotated[
        str | None,
        Field(description="provisioned: database instance (required for list; for create unless the target "
              "catalog is a registered database catalog)."),
    ] = None,
    logical_database_name: Annotated[
        str | None, Field(description="provisioned create: target Postgres database name."),
    ] = None,
    spec: Annotated[
        dict[str, Any] | None,
        Field(description="create: synced table spec, e.g. {\"source_table_full_name\": \"main.sales.orders\", "
              "\"primary_key_columns\": [\"order_id\"], \"scheduling_policy\": \"TRIGGERED\"} (SNAPSHOT | "
              "TRIGGERED | CONTINUOUS; optional new_pipeline_spec / existing_pipeline_id, timeseries_key, "
              "create_database_objects_if_missing). Autoscaling specs also take branch and postgres_database. "
              "Unknown fields are rejected."),
    ] = None,
    purge_data: Annotated[bool, Field(description="provisioned delete: also DROP the Postgres table.")] = False,
    operation_name: OperationName = None,
    wait_seconds: WaitSeconds = 0,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Lakebase synced tables (reverse ETL: Unity Catalog Delta table -> Lakebase Postgres table).

    Actions: list (provisioned; instance_name), get, create (table_name + spec with source_table_full_name,
    primary_key_columns, scheduling_policy SNAPSHOT/TRIGGERED/CONTINUOUS), delete (purge_data=true also
    drops the Postgres table), trigger (starts the synced table's managed pipeline via
    pipelines.start_update; not for CONTINUOUS), get_operation (autoscaling). update is not supported by
    the Databricks API. kind='autoscaling' uses w.postgres synced tables (no list)."""
    w = ctx().w
    tool_name = "manage_lakebase_sync"

    if action == "get_operation":
        return _get_operation(operation_name)

    if action == "update":
        raise UnsupportedOperation(_SYNC_UPDATE_UNSUPPORTED, hint="Delete and re-create the synced table instead.")

    if action == "list":
        if kind == "autoscaling":
            raise UnsupportedOperation(
                "The Lakebase autoscaling API (w.postgres) has no list operation for synced tables.",
                hint="Use action='get' with a known table_name, or browse the catalog in Unity Catalog.",
            )
        require(instance_name, "instance_name", action)
        return paged_response(
            "synced tables", w.database.list_synced_database_tables(instance_name=instance_name),
            page_size, page_token, _synced_summary,
        )

    require(table_name, "table_name", action)

    if action == "get":
        info = _sync_info(kind, table_name)
        return ok(
            f"Synced table '{table_name}' is {info.get('state') or 'in an unknown state'} "
            f"(policy {info.get('scheduling_policy')}).",
            info["object"],
        )

    if action == "create":
        require(spec, "spec", action)
        if kind == "provisioned":
            table = parse_sdk_object(
                dbsvc.SyncedDatabaseTable,
                {
                    "name": table_name,
                    "database_instance_name": instance_name,
                    "logical_database_name": logical_database_name,
                    "spec": spec,
                },
            )
            result = w.database.create_synced_database_table(synced_table=table)
            status = result.data_synchronization_status
            state = _value(status.detailed_state) if status else None
            warning = _track("lakebase_synced_table", table_name, table_name, tool_name,
                             instance=instance_name, pipeline_id=status.pipeline_id if status else None)
            pending = state is None or state in _PROVISIONING_SYNC_STATES
            return ok(
                f"Created synced table '{table_name}'" + (f" (state {state})." if state else "."),
                result,
                status="pending" if pending else "success",
                warnings=[warning],
                next_steps=[f"Poll with action='get', table_name='{table_name}'."] if pending else [],
            )
        if instance_name or logical_database_name:
            raise ValidationFailed(
                "instance_name/logical_database_name apply to kind='provisioned'; for autoscaling put "
                "branch and postgres_database in spec."
            )
        table = parse_sdk_object(pgsvc.SyncedTable, {"spec": spec})
        op = w.postgres.create_synced_table(synced_table=table, synced_table_id=table_name)
        warning = _track("lakebase_synced_table", _pg_synced_name(table_name), table_name, tool_name)
        return _lro_response(
            op, what=f"Creating synced table '{table_name}'", result_cls=pgsvc.SyncedTable,
            wait_seconds=wait_seconds, warnings=[warning],
            poll_hint=f"Or action='get', kind='autoscaling', table_name='{table_name}'.",
        )

    if action == "delete":
        ctx().safety.check_protected("Lakebase synced table", table_name, operation="delete")
        if kind == "provisioned":
            w.database.delete_synced_database_table(name=table_name, purge_data=True if purge_data else None)
            ctx().manifest.safe_untrack("lakebase_synced_table", table_name)
            return ok(
                f"Deleted synced table '{table_name}'" + (" and its Postgres table." if purge_data else "."),
                {"table_name": table_name, "purge_data": purge_data},
            )
        if purge_data:
            raise ValidationFailed("purge_data is only supported for kind='provisioned'")
        op = w.postgres.delete_synced_table(name=_pg_synced_name(table_name))
        ctx().manifest.safe_untrack("lakebase_synced_table", _pg_synced_name(table_name))
        return _lro_response(
            op, what=f"Deleting synced table '{table_name}'", result_cls=None, wait_seconds=wait_seconds,
            poll_hint="Deletion is complete when the operation reports done=true.",
        )

    if action == "trigger":
        info = _sync_info(kind, table_name)
        policy = info.get("scheduling_policy")
        if policy == "CONTINUOUS":
            raise ValidationFailed(
                f"Synced table '{table_name}' uses the CONTINUOUS policy and is updated automatically; "
                "there is nothing to trigger."
            )
        pipeline_id = info.get("pipeline_id")
        if not pipeline_id:
            raise UnsupportedOperation(
                f"Synced table '{table_name}' does not expose a pipeline_id (state "
                f"{info.get('state')}); the SDK has no direct sync trigger, so it cannot be started here.",
                hint="Wait until provisioning finishes, then retry.",
            )
        response = w.pipelines.start_update(pipeline_id=pipeline_id)
        return ok(
            f"Started sync of '{table_name}' (pipeline {pipeline_id}, update {response.update_id}).",
            {"table_name": table_name, "pipeline_id": pipeline_id, "update_id": response.update_id,
             "scheduling_policy": policy},
            status="pending",
            next_steps=[f"Poll with action='get', table_name='{table_name}' (or the pipelines tools with "
                        f"pipeline_id='{pipeline_id}')."],
        )

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover - guarded by safety map


# ==============================================================================================
# generate_lakebase_credential
# ==============================================================================================

def _current_user_name() -> str | None:
    c = ctx()
    cached = c.cached_user_name()
    if cached:
        return cached
    try:
        user_name = c.w.current_user.me().user_name
    except Exception:
        return None
    if user_name:
        c.remember_user_name(user_name)
    return user_name


def _connection(host: str | None, user: str | None, source: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "source": source,
        "host": host,
        "port": PG_PORT,
        "database": DEFAULT_PG_DATABASE,
        "user": user,
        "sslmode": "require",
        "auth": "Use the generated OAuth token as the Postgres password.",
    }
    if host and user:
        info["uri"] = (
            f"postgresql://{quote(user, safe='')}@{host}:{PG_PORT}/{DEFAULT_PG_DATABASE}?sslmode=require"
        )
    return info


def _credential_preview(args: dict[str, Any]) -> PlanInfo:
    kind = args.get("kind") or "provisioned"
    reveal = bool(args.get("reveal_token"))
    target = (
        {"kind": kind, "instance_names": args.get("instance_names")}
        if kind == "provisioned"
        else {"kind": kind, "endpoint": args.get("endpoint")}
    )
    warnings = ["The credential is a short-lived OAuth token (about one hour) carrying your Databricks identity."]
    if reveal:
        warnings.append("reveal_token=true: the token will be included in the response. It is a secret and must "
                        "not be logged, shared or persisted.")
    return PlanInfo(
        description="Generate a Lakebase Postgres OAuth credential for the current identity"
        + (" and RETURN the token." if reveal else "; the token itself will NOT be returned (reveal_token=false)."),
        target=target,
        details={"claims": args.get("claims"), "ttl_seconds": args.get("ttl_seconds")},
        warnings=warnings,
        reversible=None,
    )


@tool(
    toolset=TOOLSET,
    title="Generate Lakebase credential",
    safety=WRITE_SECURITY,
    preview=_credential_preview,
)
def generate_lakebase_credential(
    kind: Kind = "provisioned",
    instance_names: Annotated[
        list[str] | None,
        Field(description="provisioned: database instance names the credential is for."),
    ] = None,
    endpoint: Annotated[
        str | None,
        Field(description="autoscaling: endpoint resource name projects/<p>/branches/<b>/endpoints/<e>."),
    ] = None,
    claims: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Optional Unity Catalog claims scoping the token, e.g. [{\"permission_set\": "
              "\"READ_ONLY\", \"resources\": [{\"table_name\": \"cat.schema.table\"}]}]."),
    ] = None,
    request_id: Annotated[str | None, Field(description="provisioned: optional idempotency request id.")] = None,
    ttl_seconds: Annotated[
        int | None, Field(description="autoscaling: token lifetime in seconds (300-3600).", ge=300, le=3600)
    ] = None,
    reveal_token: Annotated[
        bool,
        Field(description="Return the token itself. Default false: only expiry and connection details are "
              "returned. Requires confirm=true."),
    ] = False,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Generate a short-lived OAuth credential for connecting to Lakebase Postgres as the current identity.

    kind='provisioned' (instance_names and/or claims) or kind='autoscaling' (endpoint, optional ttl_seconds).
    By default the token is NOT returned - only its expiration and connection details (host, port 5432,
    database databricks_postgres, user, sslmode=require). Pass reveal_token=true (with confirm=true) to
    receive the token in data.token; treat it as a secret and never log or store it."""
    w = ctx().w
    user = _current_user_name()
    warnings: list[str] = []

    if kind == "provisioned":
        if endpoint or ttl_seconds:
            raise ValidationFailed("endpoint/ttl_seconds apply to kind='autoscaling'")
        if not instance_names and not claims:
            raise ValidationFailed("Provide instance_names and/or claims for kind='provisioned'")
        parsed_claims = [
            parse_sdk_object(dbsvc.RequestedClaims, c, f"claims[{i}]") for i, c in enumerate(claims or [])
        ]
        credential = w.database.generate_database_credential(
            instance_names=instance_names or None, claims=parsed_claims or None, request_id=request_id
        )
        expiration = credential.expiration_time
        connections = []
        for instance_name in instance_names or []:
            host = None
            try:
                host = w.database.get_database_instance(name=instance_name).read_write_dns
            except Exception as exc:  # connection info is best-effort; the credential was minted
                warnings.append(f"Could not read instance '{instance_name}' for its host: {type(exc).__name__}")
            connections.append(_connection(host, user, instance_name))
        target = {"instance_names": instance_names or []}
    else:
        if instance_names or request_id:
            raise ValidationFailed("instance_names/request_id apply to kind='provisioned'")
        require(endpoint, "endpoint")
        parsed_claims = [
            parse_sdk_object(pgsvc.RequestedClaims, c, f"claims[{i}]") for i, c in enumerate(claims or [])
        ]
        credential = w.postgres.generate_database_credential(
            endpoint=endpoint,
            claims=parsed_claims or None,
            ttl=Duration(seconds=ttl_seconds) if ttl_seconds else None,
        )
        expiration = _ts(credential.expire_time)
        host = None
        try:
            ep = w.postgres.get_endpoint(name=endpoint)
            host = ep.status.hosts.host if ep.status and ep.status.hosts else None
        except Exception as exc:
            warnings.append(f"Could not read endpoint '{endpoint}' for its host: {type(exc).__name__}")
        connections = [_connection(host, user, endpoint)]
        target = {"endpoint": endpoint}

    data: dict[str, Any] = {
        "kind": kind,
        **target,
        "expiration_time": expiration,
        "user": user,
        "connections": connections,
        "token_returned": bool(reveal_token),
    }
    token = credential.token
    del credential
    if reveal_token:
        data["token"] = token
        warnings.append(
            "data.token is a short-lived secret OAuth token: use it only as the Postgres password for this "
            "connection; never log, print, commit or persist it."
        )
        summary = f"Generated a Lakebase credential (expires {expiration}); token included in data.token."
    else:
        data["how_to_obtain"] = (
            "The token was generated but withheld. Call again with reveal_token=true and confirm=true to "
            "receive it, or have the application mint its own token at connect time via the Databricks SDK "
            "(w.database.generate_database_credential / w.postgres.generate_database_credential) or CLI."
        )
        summary = f"Generated a Lakebase credential (expires {expiration}); token withheld (reveal_token=false)."
    token = None  # noqa: F841 - drop our reference as early as possible

    response = ok(summary, data, warnings=warnings)
    if reveal_token:
        response._unredacted_keys = {"token"}
    return response
