"""AI/BI (Lakeview) dashboards: manage_dashboard."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from databricks.sdk.service.dashboards import Dashboard
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, READ, WRITE, WRITE_SECURITY, SafetyLevel
from dbx_mcp.safety.validation import validate_workspace_path
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, ValidationFailed
from dbx_mcp.utils.serialization import pick, to_jsonable

DashboardAction = Literal["create", "get", "list", "update", "delete", "publish", "unpublish", "get_published"]

_ACTION_LEVELS: dict[str, frozenset[SafetyLevel]] = {
    "create": WRITE,
    "get": READ,
    "list": READ,
    "update": WRITE,
    "delete": DESTRUCTIVE,
    "publish": WRITE,
    "unpublish": DESTRUCTIVE,
    "get_published": READ,
}

_SUMMARY_KEYS = (
    "dashboard_id", "display_name", "path", "parent_path", "lifecycle_state", "warehouse_id", "create_time",
    "update_time",
)


def _levels(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action not in _ACTION_LEVELS:
        raise DbxToolError(
            ErrorCategory.INVALID_PARAMETER,
            f"Unknown action {action!r} for manage_dashboard. Valid actions: {', '.join(_ACTION_LEVELS)}",
        )
    if action == "publish" and args.get("embed_credentials"):
        # Viewers run the dashboard's queries with the publisher's credentials -> shares data access.
        return WRITE_SECURITY
    return _ACTION_LEVELS[action]


def _serialized(value: str | dict[str, Any] | None) -> str | None:
    """Accept the dashboard definition as a JSON string or object; return a JSON string."""
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValidationFailed(f"serialized_dashboard is not valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise ValidationFailed("serialized_dashboard must be a JSON object (with 'pages' and optionally 'datasets')")
    return value


def _parent_path(path: str | None) -> str | None:
    if path is None:
        return None
    clean = validate_workspace_path(path)
    ctx().safety.check_workspace_path(clean)
    return clean


def _summary(d: Dashboard) -> dict[str, Any]:
    return pick(to_jsonable(d), _SUMMARY_KEYS)


def _get_dashboard(dashboard_id: str) -> Dashboard:
    return ctx().w.lakeview.get(dashboard_id)


def _preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in {"delete", "unpublish", "publish", "update"}:
        return None
    dashboard_id = require(args.get("dashboard_id"), "dashboard_id", action)
    c = ctx()
    d = _get_dashboard(dashboard_id)
    name = d.display_name
    target = {"dashboard_id": dashboard_id, "display_name": name, "path": d.path}
    warnings: list[str] = []
    details: dict[str, Any] = {"dashboard": _summary(d)}
    if action == "delete":
        c.safety.check_protected("dashboard", name, operation="delete")
        try:
            published = c.w.lakeview.get_published(dashboard_id)
            details["published"] = to_jsonable(published)
            warnings.append("The dashboard is published; viewers will lose access once it is trashed.")
        except (NotFound, ResourceDoesNotExist):
            details["published"] = None
        return PlanInfo(
            description=f"Move dashboard {name!r} ({dashboard_id}) at {d.path} to the trash.",
            target=target,
            details=details,
            warnings=warnings
            + ["Trashed dashboards can be restored from the workspace Trash until they are permanently purged."],
            reversible=True,
        )
    if action == "unpublish":
        c.safety.check_protected("dashboard", name, operation="unpublish")
        try:
            details["published"] = to_jsonable(c.w.lakeview.get_published(dashboard_id))
        except (NotFound, ResourceDoesNotExist):
            details["published"] = None
            warnings.append("The dashboard does not appear to be published.")
        return PlanInfo(
            description=f"Unpublish dashboard {name!r} ({dashboard_id}); viewers of the published version lose access.",
            target=target,
            details=details,
            warnings=warnings + ["The draft is kept; it can be published again (the current draft would be published)."],
            reversible=True,
        )
    if action == "publish":
        embed = bool(args.get("embed_credentials"))
        if embed:
            warnings.append(
                "embed_credentials=true: viewers will run the dashboard's queries with YOUR credentials, "
                "seeing any data you can access through them."
            )
        return PlanInfo(
            description=f"Publish the current draft of dashboard {name!r} ({dashboard_id})"
            + (" with embedded publisher credentials." if embed else " without embedded credentials."),
            target=target,
            details={**details, "warehouse_id": args.get("warehouse_id") or d.warehouse_id, "embed_credentials": embed},
            warnings=warnings,
            reversible=True,
        )
    changes = {k: args.get(k) for k in ("display_name", "warehouse_id", "dataset_catalog", "dataset_schema") if args.get(k)}
    if args.get("serialized_dashboard") is not None:
        changes["serialized_dashboard"] = "(replaced)"
        warnings.append("serialized_dashboard replaces the entire draft definition (pages, datasets, widgets).")
    return PlanInfo(
        description=f"Update draft dashboard {name!r} ({dashboard_id}).",
        target=target,
        details={**details, "changes": changes},
        warnings=warnings,
        reversible=False,
    )


@tool(
    toolset="dashboards",
    title="Manage AI/BI dashboards",
    safety=_levels,
    possible_levels=READ | WRITE_SECURITY | DESTRUCTIVE,
    preview=_preview,
)
def manage_dashboard(
    action: Annotated[
        DashboardAction,
        Field(
            description="create | get | list | update (draft) | delete (move to trash) | publish | unpublish "
            "| get_published"
        ),
    ],
    dashboard_id: Annotated[str | None, Field(description="Dashboard id (all actions except create/list).")] = None,
    display_name: Annotated[str | None, Field(description="create/update: dashboard name.")] = None,
    parent_path: Annotated[
        str | None, Field(description="create: workspace folder for the dashboard, e.g. /Users/me@x.com/dashboards.")
    ] = None,
    warehouse_id: Annotated[
        str | None, Field(description="create/update: SQL warehouse for the draft; publish: override warehouse.")
    ] = None,
    serialized_dashboard: Annotated[
        str | dict[str, Any] | None,
        Field(description="create/update: the dashboard definition (JSON string or object, as exported by Databricks)."),
    ] = None,
    dataset_catalog: Annotated[str | None, Field(description="create/update: default catalog for all datasets.")] = None,
    dataset_schema: Annotated[str | None, Field(description="create/update: default schema for all datasets.")] = None,
    etag: Annotated[str | None, Field(description="update: etag from get, to fail if the draft changed meanwhile.")] = None,
    embed_credentials: Annotated[
        bool,
        Field(
            description="publish: run viewers' queries with the publisher's credentials (SECURITY_SENSITIVE; "
            "requires confirm). Default false: viewers use their own credentials."
        ),
    ] = False,
    show_trashed: Annotated[bool, Field(description="list: include dashboards in the trash.")] = False,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage AI/BI (Lakeview) dashboards. Actions: create (display_name, optional parent_path,
    warehouse_id, serialized_dashboard), get, list (show_trashed), update (draft fields; etag for
    optimistic concurrency), delete (moves to trash; recoverable), publish (embed_credentials,
    warehouse_id), unpublish, get_published. Created dashboards are tracked in the project manifest."""
    c = ctx()
    if action == "list":
        return paged_response(
            "dashboards",
            c.w.lakeview.list(show_trashed=show_trashed or None),
            page_size,
            page_token,
            transform=_summary,
        )

    if action == "create":
        require(display_name, "display_name", action)
        dashboard = Dashboard(
            display_name=display_name,
            parent_path=_parent_path(parent_path),
            warehouse_id=warehouse_id,
            serialized_dashboard=_serialized(serialized_dashboard),
        )
        created = c.w.lakeview.create(dashboard, dataset_catalog=dataset_catalog, dataset_schema=dataset_schema)
        warning = c.manifest.safe_track(
            resource_type="dashboard",
            resource_id=created.dashboard_id,
            name=created.display_name,
            created_by_tool="manage_dashboard",
            workspace_host=c.host,
            metadata={"path": created.path},
        )
        return ok(
            f"Created draft dashboard {created.display_name!r} ({created.dashboard_id}) at {created.path}.",
            _summary(created),
            warnings=[warning] if warning else None,
            next_steps=["Publish it with manage_dashboard action=publish when ready."],
        )

    dashboard_id = require(dashboard_id, "dashboard_id", action)

    if action == "get":
        return ok(f"Dashboard {dashboard_id}.", _get_dashboard(dashboard_id))

    if action == "get_published":
        published = c.w.lakeview.get_published(dashboard_id)
        return ok(f"Published version of dashboard {dashboard_id} ({published.display_name}).", published)

    if action == "update":
        if parent_path is not None:
            raise ValidationFailed("parent_path cannot be changed with update; move the dashboard in the workspace instead")
        fields = {
            "display_name": display_name,
            "warehouse_id": warehouse_id,
            "serialized_dashboard": _serialized(serialized_dashboard),
            "etag": etag,
        }
        if not any(v is not None for k, v in fields.items() if k != "etag") and not (dataset_catalog or dataset_schema):
            raise ValidationFailed(
                "update requires at least one of display_name, warehouse_id, serialized_dashboard, "
                "dataset_catalog, dataset_schema"
            )
        updated = c.w.lakeview.update(
            dashboard_id,
            Dashboard(**{k: v for k, v in fields.items() if v is not None}),
            dataset_catalog=dataset_catalog,
            dataset_schema=dataset_schema,
        )
        return ok(f"Updated draft dashboard {updated.display_name!r} ({dashboard_id}).", _summary(updated))

    if action == "delete":
        d = _get_dashboard(dashboard_id)
        c.safety.check_protected("dashboard", d.display_name, operation="delete")
        c.w.lakeview.trash(dashboard_id)
        c.manifest.safe_untrack("dashboard", dashboard_id)
        return ok(
            f"Moved dashboard {d.display_name!r} ({dashboard_id}) to the trash.",
            {"dashboard_id": dashboard_id, "display_name": d.display_name, "lifecycle_state": "TRASHED"},
            next_steps=["It can be restored from the workspace Trash if this was a mistake."],
        )

    if action == "publish":
        published = c.w.lakeview.publish(
            dashboard_id, embed_credentials=embed_credentials, warehouse_id=warehouse_id
        )
        return ok(
            f"Published dashboard {dashboard_id}"
            + (" with embedded credentials." if embed_credentials else " (viewers use their own credentials)."),
            published,
        )

    if action == "unpublish":
        d = _get_dashboard(dashboard_id)
        c.safety.check_protected("dashboard", d.display_name, operation="unpublish")
        c.w.lakeview.unpublish(dashboard_id)
        return ok(f"Unpublished dashboard {d.display_name!r} ({dashboard_id}); the draft is kept.",
                  {"dashboard_id": dashboard_id, "published": False})

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover - Literal guards this
