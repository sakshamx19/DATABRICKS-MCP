"""Compute: manage_cluster, manage_sql_warehouse, manage_warehouse, list_compute."""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal

from databricks.sdk.service import compute as compute_svc
from pydantic import Field

from dbx_mcp.databricks.sql_runner import rank_warehouses, select_warehouse
from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, READ, WRITE
from dbx_mcp.tools.common import (
    Confirm,
    DryRun,
    PageSize,
    PageToken,
    Spec,
    ctx,
    list_page,
    ok,
    paged_response,
    require,
)
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import ValidationFailed
from dbx_mcp.utils.polling import wait_cap
from dbx_mcp.utils.serialization import (
    call_with_spec,
    coerce_kwargs,
    parse_sdk_object,
    pick,
    to_jsonable,
    wait_response,
)

Wait = Annotated[bool, Field(description="Wait (bounded) for the operation to reach a steady state.")]
TimeoutSeconds = Annotated[
    int | None, Field(description="Max seconds to wait when wait=true (capped by DBX_MCP_MAX_WAIT_SECONDS).", ge=1)
]


def _state(value: Any) -> str | None:
    return getattr(value, "value", value)


def _wait_timeout(timeout_seconds: int | None) -> dt.timedelta:
    cap = wait_cap(ctx().settings)
    return dt.timedelta(seconds=min(timeout_seconds or cap, cap))


# ==============================================================================================
# Clusters
# ==============================================================================================

_CLUSTER_SUMMARY = (
    "cluster_id", "cluster_name", "state", "spark_version", "node_type_id", "num_workers", "autoscale",
    "creator_user_name", "cluster_source", "data_security_mode", "autotermination_minutes", "custom_tags",
)


def _cluster_summary(cluster: compute_svc.ClusterDetails) -> dict[str, Any]:
    return pick(to_jsonable(cluster), _CLUSTER_SUMMARY)


def _cluster_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in {"update", "restart", "terminate", "delete", "resize", "start"}:
        return None
    cluster_id = require(args.get("cluster_id"), "cluster_id", action)
    c = ctx()
    cluster = c.w.clusters.get(cluster_id)
    state = _state(cluster.state)
    c.safety.check_protected("cluster", cluster.cluster_name, cluster.custom_tags, operation=action)
    warnings: list[str] = []
    running = state in {"RUNNING", "RESIZING", "PENDING", "RESTARTING"}
    descriptions = {
        "terminate": "Terminate (stop) the cluster. Its configuration is kept and it can be started again.",
        "delete": "PERMANENTLY delete the cluster. Its configuration is removed and cannot be recovered.",
        "restart": "Restart the cluster.",
        "update": "Update the cluster configuration.",
        "resize": "Resize the cluster.",
        "start": "Start the cluster (incurs compute cost).",
    }
    if running and action in {"terminate", "delete", "restart"}:
        warnings.append("The cluster is active: running commands, notebooks and jobs on it will be interrupted.")
    if running and action == "update":
        warnings.append("Updating a running cluster may restart it, interrupting running workloads.")
    if cluster.cluster_source and _state(cluster.cluster_source) == "JOB":
        warnings.append("This is a job cluster managed by the Jobs service.")
    details: dict[str, Any] = {"current": _cluster_summary(cluster)}
    if action == "update" and args.get("spec"):
        details["changes"] = args["spec"]
    return PlanInfo(
        description=f"{descriptions[action]} Cluster {cluster.cluster_name!r} ({cluster_id}) is currently {state}.",
        target={"cluster_id": cluster_id, "cluster_name": cluster.cluster_name},
        details=details,
        warnings=warnings,
        reversible=action != "delete",
    )


@tool(
    toolset="compute",
    title="Manage clusters",
    safety={
        "list": READ,
        "get": READ,
        "events": READ,
        "create": WRITE,
        "update": WRITE,
        "resize": WRITE,
        "start": WRITE,
        "restart": DESTRUCTIVE,
        "terminate": DESTRUCTIVE,
        "delete": DESTRUCTIVE,
    },
    preview=_cluster_preview,
)
def manage_cluster(
    action: Annotated[
        Literal["list", "get", "events", "create", "update", "resize", "start", "restart", "terminate", "delete"],
        Field(description="terminate = stop (restartable); delete = permanent removal."),
    ],
    cluster_id: Annotated[str | None, Field(description="Cluster id (all actions except list/create).")] = None,
    spec: Spec = None,
    num_workers: Annotated[int | None, Field(description="For resize: fixed worker count.", ge=0)] = None,
    autoscale_min_workers: Annotated[int | None, Field(description="For resize: autoscale minimum.", ge=0)] = None,
    autoscale_max_workers: Annotated[int | None, Field(description="For resize: autoscale maximum.", ge=1)] = None,
    states: Annotated[
        list[str] | None,
        Field(description="For list: filter by states, e.g. ['RUNNING','PENDING']."),
    ] = None,
    wait: Wait = False,
    timeout_seconds: TimeoutSeconds = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    confirm: Confirm = False,
    dry_run: DryRun = False,
) -> ToolResponse:
    """Manage all-purpose clusters.

    Actions: list (optionally filtered by state), get, events (recent cluster events), create
    (spec = Clusters API create body, e.g. {"cluster_name","spark_version","node_type_id",
    "num_workers" or "autoscale","autotermination_minutes"}), update (partial update: spec holds
    only the fields to change), resize, start, restart, terminate (stop; restartable) and delete
    (permanent). restart/terminate/delete require confirm=true and are refused for clusters whose
    name/tags match the protected (production) patterns. Lifecycle actions return immediately
    with the current state unless wait=true."""
    c = ctx()
    w = c.w
    if action == "list":
        filter_by = None
        if states:
            filter_by = parse_sdk_object(compute_svc.ListClustersFilterBy, {"cluster_states": states}, "states")
        return paged_response("cluster(s)", w.clusters.list(filter_by=filter_by), page_size, page_token,
                              _cluster_summary)

    if action == "create":
        kwargs = coerce_kwargs(w.clusters.create, spec)
        if kwargs.get("autotermination_minutes") in (None, 0):
            warning = "No autotermination_minutes set: the cluster will run (and bill) until stopped."
        else:
            warning = None
        waiter = w.clusters.create(**kwargs)
        cluster_id = wait_response(waiter).cluster_id
        warnings = [warning, c.manifest.safe_track(resource_type="cluster", resource_id=cluster_id,
                                                   name=kwargs.get("cluster_name"), created_by_tool="manage_cluster",
                                                   workspace_host=c.host)]
        return _lifecycle_result(w, cluster_id, "created", wait, timeout_seconds, warnings)

    cluster_id = require(cluster_id, "cluster_id", action)
    if action == "get":
        return ok(f"Cluster {cluster_id}.", w.clusters.get(cluster_id))
    if action == "events":
        events = w.clusters.events(cluster_id)
        return paged_response("event(s)", events, page_size, page_token)

    cluster = w.clusters.get(cluster_id)
    if action in {"update", "restart", "terminate", "delete", "resize"}:
        c.safety.check_protected("cluster", cluster.cluster_name, cluster.custom_tags, operation=action)

    if action == "update":
        if not spec:
            raise ValidationFailed("spec with the fields to change is required for update")
        resource = parse_sdk_object(compute_svc.UpdateClusterResource, spec)
        w.clusters.update(cluster_id, update_mask=",".join(sorted(spec)), cluster=resource)
        return _lifecycle_result(w, cluster_id, f"updated ({', '.join(sorted(spec))})", wait, timeout_seconds)
    if action == "resize":
        if num_workers is None and autoscale_max_workers is None:
            raise ValidationFailed("resize needs num_workers or autoscale_min_workers/autoscale_max_workers")
        autoscale = None
        if autoscale_max_workers is not None:
            autoscale = compute_svc.AutoScale(min_workers=autoscale_min_workers or 0,
                                              max_workers=autoscale_max_workers)
        w.clusters.resize(cluster_id, num_workers=num_workers if autoscale is None else None, autoscale=autoscale)
        return _lifecycle_result(w, cluster_id, "resize requested", wait, timeout_seconds)
    if action == "start":
        w.clusters.start(cluster_id)
        return _lifecycle_result(w, cluster_id, "start requested", wait, timeout_seconds)
    if action == "restart":
        w.clusters.restart(cluster_id)
        return _lifecycle_result(w, cluster_id, "restart requested", wait, timeout_seconds)
    if action == "terminate":
        w.clusters.delete(cluster_id)  # the API's "delete" terminates; permanent_delete removes it
        return _lifecycle_result(w, cluster_id, "termination requested", wait, timeout_seconds, terminal=True)
    # delete (permanent)
    w.clusters.permanent_delete(cluster_id)
    c.manifest.safe_untrack("cluster", cluster_id)
    return ok(f"Cluster {cluster.cluster_name!r} ({cluster_id}) permanently deleted.",
              {"cluster_id": cluster_id, "deleted": True})


def _lifecycle_result(
    w: Any,
    cluster_id: str,
    what: str,
    wait: bool,
    timeout_seconds: int | None,
    warnings: list[str | None] | None = None,
    terminal: bool = False,
) -> ToolResponse:
    if wait:
        try:
            if terminal:
                cluster = w.clusters.wait_get_cluster_terminated(cluster_id, timeout=_wait_timeout(timeout_seconds))
            else:
                cluster = w.clusters.wait_get_cluster_running(cluster_id, timeout=_wait_timeout(timeout_seconds))
        except TimeoutError:
            cluster = w.clusters.get(cluster_id)
            return ok(f"Cluster {cluster_id} {what}; still {_state(cluster.state)} after waiting.",
                      _cluster_summary(cluster), status="pending", warnings=warnings,
                      next_steps=["Poll with manage_cluster action='get'."])
        return ok(f"Cluster {cluster_id} {what}; now {_state(cluster.state)}.", _cluster_summary(cluster),
                  warnings=warnings)
    cluster = w.clusters.get(cluster_id)
    state = _state(cluster.state)
    steady = state in {"RUNNING", "TERMINATED", "ERROR"}
    return ok(
        f"Cluster {cluster_id} {what}; current state {state}.",
        _cluster_summary(cluster),
        status="success" if steady else "pending",
        warnings=warnings,
        next_steps=[] if steady else ["Poll with manage_cluster action='get' (or pass wait=true)."],
    )


# ==============================================================================================
# SQL warehouses
# ==============================================================================================

_WAREHOUSE_SUMMARY = (
    "id", "name", "state", "cluster_size", "warehouse_type", "enable_serverless_compute", "min_num_clusters",
    "max_num_clusters", "num_clusters", "num_active_sessions", "auto_stop_mins", "creator_name", "health", "tags",
)


def _warehouse_summary(wh: Any) -> dict[str, Any]:
    return pick(to_jsonable(wh), _WAREHOUSE_SUMMARY)


def _warehouse_tags(wh: Any) -> dict[str, str]:
    pairs = getattr(getattr(wh, "tags", None), "custom_tags", None) or []
    return {p.key: p.value for p in pairs if p.key}


def _warehouse_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in {"update", "start", "stop", "delete"}:
        return None
    warehouse_id = require(args.get("warehouse_id"), "warehouse_id", action)
    c = ctx()
    wh = c.w.warehouses.get(warehouse_id)
    c.safety.check_protected("SQL warehouse", wh.name, _warehouse_tags(wh), operation=action)
    state = _state(wh.state)
    warnings = []
    if action in {"stop", "delete"} and state in {"RUNNING", "STARTING"}:
        warnings.append(f"The warehouse is {state} with {wh.num_active_sessions or 0} active session(s); "
                        "running queries and dashboards using it will fail.")
    descriptions = {
        "update": "Update the warehouse configuration.",
        "start": "Start the warehouse (incurs cost).",
        "stop": "Stop the warehouse. It can be started again.",
        "delete": "DELETE the warehouse. Queries, dashboards and alerts using it will break.",
    }
    details: dict[str, Any] = {"current": _warehouse_summary(wh)}
    if action == "update":
        details["changes"] = args.get("spec")
    return PlanInfo(
        description=f"{descriptions[action]} Warehouse {wh.name!r} ({warehouse_id}) is currently {state}.",
        target={"warehouse_id": warehouse_id, "name": wh.name},
        details=details,
        warnings=warnings,
        reversible=action != "delete",
    )


@tool(
    toolset="compute",
    title="Manage SQL warehouses",
    safety={
        "list": READ,
        "get": READ,
        "create": WRITE,
        "update": WRITE,
        "start": WRITE,
        "stop": DESTRUCTIVE,
        "delete": DESTRUCTIVE,
    },
    preview=_warehouse_preview,
)
def manage_sql_warehouse(
    action: Annotated[Literal["list", "get", "create", "update", "start", "stop", "delete"], Field()],
    warehouse_id: Annotated[str | None, Field(description="Warehouse id (all actions except list/create).")] = None,
    spec: Spec = None,
    wait: Wait = False,
    timeout_seconds: TimeoutSeconds = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    confirm: Confirm = False,
    dry_run: DryRun = False,
) -> ToolResponse:
    """Manage SQL warehouses.

    Actions: list, get, create (spec e.g. {"name","cluster_size":"2X-Small","max_num_clusters":1,
    "auto_stop_mins":10,"enable_serverless_compute":true,"warehouse_type":"PRO"}), update (spec holds
    only fields to change; merged onto the current configuration), start, stop and delete. stop and
    delete require confirm=true and are refused for production-marked warehouses."""
    c = ctx()
    w = c.w
    if action == "list":
        return paged_response("warehouse(s)", w.warehouses.list(), page_size, page_token, _warehouse_summary)
    if action == "create":
        kwargs = coerce_kwargs(w.warehouses.create, spec)
        warnings: list[str | None] = []
        if not kwargs.get("auto_stop_mins"):
            warnings.append("auto_stop_mins not set: the warehouse may keep running (and billing) when idle.")
        waiter = w.warehouses.create(**kwargs)
        new_id = wait_response(waiter).id
        warnings.append(c.manifest.safe_track(resource_type="warehouse", resource_id=new_id,
                                              name=kwargs.get("name"), created_by_tool="manage_sql_warehouse",
                                              workspace_host=c.host))
        return _warehouse_result(w, new_id, "created", wait, timeout_seconds, warnings, target="RUNNING")

    warehouse_id = require(warehouse_id, "warehouse_id", action)
    if action == "get":
        return ok(f"Warehouse {warehouse_id}.", w.warehouses.get(warehouse_id))

    wh = w.warehouses.get(warehouse_id)
    if action in {"update", "stop", "delete"}:
        c.safety.check_protected("SQL warehouse", wh.name, _warehouse_tags(wh), operation=action)

    if action == "update":
        if not spec:
            raise ValidationFailed("spec with the fields to change is required for update")
        # The edit API takes the full configuration: merge the requested changes onto the current one.
        current = {k: v for k, v in to_jsonable(wh).items() if k in _EDITABLE_WAREHOUSE_FIELDS}
        unknown = sorted(set(spec) - _EDITABLE_WAREHOUSE_FIELDS)
        if unknown:
            raise ValidationFailed(f"Unknown/non-editable warehouse field(s): {', '.join(unknown)}. "
                                   f"Editable: {', '.join(sorted(_EDITABLE_WAREHOUSE_FIELDS))}")
        call_with_spec(w.warehouses.edit, {**current, **spec}, fixed={"id": warehouse_id})
        return _warehouse_result(w, warehouse_id, f"updated ({', '.join(sorted(spec))})", wait, timeout_seconds)
    if action == "start":
        w.warehouses.start(warehouse_id)
        return _warehouse_result(w, warehouse_id, "start requested", wait, timeout_seconds, target="RUNNING")
    if action == "stop":
        w.warehouses.stop(warehouse_id)
        return _warehouse_result(w, warehouse_id, "stop requested", wait, timeout_seconds, target="STOPPED")
    w.warehouses.delete(warehouse_id)
    c.manifest.safe_untrack("warehouse", warehouse_id)
    return ok(f"Warehouse {wh.name!r} ({warehouse_id}) deleted.", {"warehouse_id": warehouse_id, "deleted": True})


_EDITABLE_WAREHOUSE_FIELDS = {
    "auto_stop_mins", "channel", "cluster_size", "creator_name", "enable_photon", "enable_serverless_compute",
    "instance_profile_arn", "max_num_clusters", "min_num_clusters", "name", "spot_instance_policy", "tags",
    "warehouse_type",
}


def _warehouse_result(
    w: Any,
    warehouse_id: str,
    what: str,
    wait: bool,
    timeout_seconds: int | None,
    warnings: list[str | None] | None = None,
    target: str | None = None,
) -> ToolResponse:
    if wait and target:
        try:
            if target == "RUNNING":
                wh = w.warehouses.wait_get_warehouse_running(warehouse_id, timeout=_wait_timeout(timeout_seconds))
            else:
                wh = w.warehouses.wait_get_warehouse_stopped(warehouse_id, timeout=_wait_timeout(timeout_seconds))
        except TimeoutError:
            wh = w.warehouses.get(warehouse_id)
            return ok(f"Warehouse {warehouse_id} {what}; still {_state(wh.state)} after waiting.",
                      _warehouse_summary(wh), status="pending", warnings=warnings,
                      next_steps=["Poll with manage_sql_warehouse action='get'."])
        return ok(f"Warehouse {warehouse_id} {what}; now {_state(wh.state)}.", _warehouse_summary(wh),
                  warnings=warnings)
    wh = w.warehouses.get(warehouse_id)
    state = _state(wh.state)
    steady = target is None or state == target
    return ok(f"Warehouse {warehouse_id} {what}; current state {state}.", _warehouse_summary(wh),
              status="success" if steady else "pending", warnings=warnings,
              next_steps=[] if steady else ["Poll with manage_sql_warehouse action='get' (or pass wait=true)."])


# ==============================================================================================
# Warehouse utilities
# ==============================================================================================

@tool(toolset="compute", title="Warehouse status & selection", safety=READ)
def manage_warehouse(
    action: Annotated[
        Literal["list", "status", "select"],
        Field(description="list: warehouses ranked for SQL execution; status: state/health of one "
              "warehouse; select: which warehouse SQL tools will use and why."),
    ] = "select",
    warehouse_id: Annotated[str | None, Field(description="Warehouse id for status (or to validate in select).")] = None,
    require_running: Annotated[bool, Field(description="For select: only accept a RUNNING warehouse.")] = False,
) -> ToolResponse:
    """Inspect SQL warehouses and the server's warehouse-selection logic. Selection is transparent
    and configurable: an explicit warehouse_id wins, then DBX_MCP_DEFAULT_WAREHOUSE_ID, then (with
    DBX_MCP_WAREHOUSE_SELECTION=prefer_running) the best visible warehouse ranked running > starting
    > stopped, then serverless > pro > classic, then name."""
    c = ctx()
    if action == "status":
        warehouse_id = require(warehouse_id, "warehouse_id", action)
        wh = c.w.warehouses.get(warehouse_id)
        return ok(f"Warehouse {wh.name!r} is {_state(wh.state)}.", _warehouse_summary(wh))
    if action == "list":
        ranked = rank_warehouses(list(c.w.warehouses.list()))
        data = [{"rank": i + 1, **_warehouse_summary(wh)} for i, wh in enumerate(ranked)]
        running = sum(1 for wh in ranked if _state(wh.state) == "RUNNING")
        return ok(f"{len(ranked)} warehouse(s), {running} running (ranked by selection preference).", data)
    choice = select_warehouse(c, warehouse_id)
    if choice.state is None:
        wh = c.w.warehouses.get(choice.warehouse_id)
        choice.name, choice.state = wh.name, _state(wh.state)
    if require_running and choice.state != "RUNNING":
        raise ValidationFailed(
            f"Selected warehouse {choice.name!r} is {choice.state}, not RUNNING.",
            hint="Start it with manage_sql_warehouse action='start' or choose another warehouse.",
        )
    return ok(f"SQL tools will use warehouse {choice.name!r} ({choice.warehouse_id}, {choice.state}): {choice.reason}.",
              choice.as_dict(), warnings=choice.warnings)


# ==============================================================================================
# list_compute
# ==============================================================================================

@tool(toolset="compute", title="List compute resources", safety=READ)
def list_compute(
    resource: Annotated[
        Literal["summary", "clusters", "warehouses", "node_types", "spark_versions"],
        Field(description="summary: clusters + warehouses with states; or one resource type."),
    ] = "summary",
    filter: Annotated[
        str | None, Field(description="Case-insensitive substring filter on name/id (node_types, spark_versions).")
    ] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
) -> ToolResponse:
    """List compute: all-purpose clusters and SQL warehouses with current state, available node
    types (cores, memory, GPUs, Photon support) and Databricks Runtime (Spark) versions."""
    w = ctx().w
    needle = (filter or "").lower()
    if resource == "clusters":
        return paged_response("cluster(s)", w.clusters.list(), page_size, page_token, _cluster_summary)
    if resource == "warehouses":
        return paged_response("warehouse(s)", w.warehouses.list(), page_size, page_token, _warehouse_summary)
    if resource == "node_types":
        node_types = [nt for nt in (w.clusters.list_node_types().node_types or [])
                      if not nt.is_deprecated and not nt.is_hidden
                      and (not needle or needle in (nt.node_type_id or "").lower()
                           or needle in (nt.description or "").lower())]
        node_types.sort(key=lambda nt: (nt.num_cores or 0, nt.memory_mb or 0))
        return paged_response("node type(s)", node_types, page_size, page_token, lambda nt: {
            "node_type_id": nt.node_type_id, "cores": nt.num_cores, "memory_gb": round((nt.memory_mb or 0) / 1024, 1),
            "gpus": nt.num_gpus, "category": nt.category, "photon_worker_capable": nt.photon_worker_capable,
        })
    if resource == "spark_versions":
        versions = [v for v in (w.clusters.spark_versions().versions or [])
                    if not needle or needle in (v.key or "").lower() or needle in (v.name or "").lower()]
        versions.sort(key=lambda v: v.key or "", reverse=True)
        return paged_response("runtime version(s)", versions, page_size, page_token, lambda v: {
            "key": v.key, "name": v.name, "lts": "LTS" in (v.name or ""), "ml": "-ml-" in (v.key or ""),
        })

    clusters, cluster_page = list_page(w.clusters.list(), page_size, None, _cluster_summary)
    warehouses = [_warehouse_summary(wh) for wh in w.warehouses.list()]
    data = {
        "clusters": clusters,
        "clusters_truncated": cluster_page.has_more,
        "warehouses": warehouses,
        "counts": {
            "clusters_running": sum(1 for cl in clusters if cl.get("state") == "RUNNING"),
            "clusters_listed": len(clusters),
            "warehouses_running": sum(1 for wh in warehouses if wh.get("state") == "RUNNING"),
            "warehouses_total": len(warehouses),
        },
    }
    counts = data["counts"]
    return ok(
        f"{counts['clusters_listed']} cluster(s) ({counts['clusters_running']} running), "
        f"{counts['warehouses_total']} warehouse(s) ({counts['warehouses_running']} running).",
        data,
        next_steps=["Use resource='node_types' or 'spark_versions' for creation options."],
    )
