"""Project manifest tools: list_tracked_resources, delete_tracked_resource.

The manifest is a local JSON record of resources created through this server
(see :mod:`dbx_mcp.server.manifest`). These tools only read/modify that local
record; they never delete anything in Databricks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import READ, WRITE
from dbx_mcp.server.manifest import TrackedResource, normalize_host
from dbx_mcp.tools.common import DryRun, PageSize, PageToken, ctx, list_page, ok, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, categorize

NOT_DELETED_NOTE = (
    "The Databricks resource itself was NOT deleted - only its entry in the local project manifest was removed. "
    "Use the matching manage_* tool (e.g. manage_jobs, manage_dashboard, manage_app) to delete the resource."
)


def _job(w: Any, rid: str) -> Any:
    try:
        job_id = int(rid)
    except ValueError:
        raise ValueError(f"job id {rid!r} is not numeric") from None
    return w.jobs.get(job_id=job_id)


# resource_type -> cheap existence check (raises NotFound when the resource is gone)
_VERIFIERS: dict[str, Callable[[Any, str], Any]] = {
    "job": _job,
    "pipeline": lambda w, rid: w.pipelines.get(pipeline_id=rid),
    "dashboard": lambda w, rid: w.lakeview.get(dashboard_id=rid),
    "app": lambda w, rid: w.apps.get(name=rid),
    "cluster": lambda w, rid: w.clusters.get(cluster_id=rid),
    "warehouse": lambda w, rid: w.warehouses.get(id=rid),
}


def _same_host(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return True
    return a.rstrip("/").lower() == b.rstrip("/").lower()


def _verify(item: dict[str, Any], host: str | None) -> dict[str, Any]:
    rtype, rid = item["resource_type"], item["resource_id"]
    if not _same_host(item.get("workspace_host"), host):
        return {"status": "other_workspace", "detail": f"Tracked in {item.get('workspace_host')}, not the current workspace."}
    verifier = _VERIFIERS.get(rtype)
    if verifier is None:
        return {"status": "not_verified", "detail": f"Existence checks are not implemented for type {rtype!r}."}
    try:
        obj = verifier(ctx().w, rid)
    except (NotFound, ResourceDoesNotExist):
        return {"status": "missing", "detail": "Not found in Databricks (deleted outside this server?)."}
    except Exception as exc:
        return {"status": "unknown", "detail": f"Check failed: {categorize(exc).value}"}
    lifecycle = getattr(obj, "lifecycle_state", None)
    if getattr(lifecycle, "value", lifecycle) == "TRASHED":
        return {"status": "trashed", "detail": "The dashboard is in the trash."}
    return {"status": "exists"}


@tool(toolset="manifest", title="List tracked resources", safety=READ)
def list_tracked_resources(
    resource_type: Annotated[
        str | None, Field(description="Filter by type, e.g. job, pipeline, dashboard, app, cluster, warehouse.")
    ] = None,
    verify: Annotated[
        bool,
        Field(
            description="Check whether each returned resource still exists in Databricks (one GET per item; "
            "supported for job, pipeline, dashboard, app, cluster, warehouse)."
        ),
    ] = False,
    page_size: PageSize = None,
    page_token: PageToken = None,
) -> ToolResponse:
    """List resources recorded in the local project manifest (created through this server):
    type, id, name, creating tool, creation time and workspace. With verify=true, each returned
    item is checked against Databricks and missing ones are reported. Paginated."""
    c = ctx()
    items = _visible(c.manifest.list(resource_type or None))
    page, info = list_page(items, page_size, page_token, transform=lambda r: r.model_dump())
    warnings: list[str] = []
    missing: list[dict[str, str]] = []
    if verify:
        host = c.host
        for item in page:
            item["verification"] = _verify(item, host)
            if item["verification"]["status"] in {"missing", "trashed"}:
                missing.append({"resource_type": item["resource_type"], "resource_id": item["resource_id"]})
        if missing:
            warnings.append(
                f"{len(missing)} tracked resource(s) no longer exist (or are trashed) in Databricks; "
                "remove them from the manifest with delete_tracked_resource if no longer needed."
            )
    data: dict[str, Any] = {"manifest_path": str(c.manifest.path), "resources": page}
    if verify:
        data["missing"] = missing
    more = " (more available - pass next_page_token)" if info.has_more else ""
    noun = f"tracked {resource_type} resource(s)" if resource_type else "tracked resource(s)"
    return ok(f"Returned {info.returned} {noun}{more}.", data, page=info, warnings=warnings)


def _visible(items: list[TrackedResource]) -> list[TrackedResource]:
    """In request-auth mode a caller only sees entries of the workspace it authenticated to."""
    c = ctx()
    if not c.clients.request_mode:
        return items
    host = normalize_host(c.host)
    return [r for r in items if normalize_host(r.workspace_host) == host]


def _find(resource_type: str, resource_id: str) -> TrackedResource | None:
    return next((r for r in _visible(ctx().manifest.list(resource_type)) if r.resource_id == resource_id), None)


def _untrack_preview(args: dict[str, Any]) -> PlanInfo:
    rtype = require(args.get("resource_type"), "resource_type")
    rid = require(args.get("resource_id"), "resource_id")
    entry = _find(rtype, rid)
    warnings = ["This does NOT delete the Databricks resource; it only stops tracking it locally."]
    if entry is None:
        warnings.append("No such entry is currently tracked; the call will fail.")
    return PlanInfo(
        description=f"Remove {rtype} {rid!r} from the local project manifest. The Databricks resource is NOT deleted.",
        target={"resource_type": rtype, "resource_id": rid},
        details={"entry": entry.model_dump() if entry else None, "manifest_path": str(ctx().manifest.path)},
        warnings=warnings,
        reversible=False,
    )


@tool(toolset="manifest", title="Stop tracking a resource", safety=WRITE, preview=_untrack_preview)
def delete_tracked_resource(
    resource_type: Annotated[str, Field(description="Type of the tracked entry, e.g. job, dashboard, app.")],
    resource_id: Annotated[str, Field(description="Id of the tracked entry (as shown by list_tracked_resources).")],
    dry_run: DryRun = False,
) -> ToolResponse:
    """Remove an entry from the local project manifest (stop tracking it). This does NOT delete
    the Databricks resource itself - use the matching manage_* tool for that."""
    require(resource_type, "resource_type")
    require(resource_id, "resource_id")
    c = ctx()
    if c.clients.request_mode:
        removed = c.manifest.untrack(resource_type, resource_id, c.host) if _find(resource_type, resource_id) else None
    else:
        removed = c.manifest.untrack(resource_type, resource_id)
    if removed is None:
        raise DbxToolError(
            ErrorCategory.NOT_FOUND,
            f"No tracked {resource_type} with id {resource_id!r} in the project manifest.",
            hint="Use list_tracked_resources to see tracked entries.",
        )
    return ok(
        f"Stopped tracking {resource_type} {resource_id!r}"
        + (f" ({removed.name})" if removed.name else "")
        + f". {NOT_DELETED_NOTE}",
        {"removed": removed.model_dump(), "databricks_resource_deleted": False},
        warnings=[NOT_DELETED_NOTE],
    )
