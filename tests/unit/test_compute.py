"""Compute toolset tests (clusters, warehouses, selection, list_compute)."""

from __future__ import annotations

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service import compute, sql
from databricks.sdk.service._internal import Wait

TOOLSETS = ("compute",)


def _cluster(cid="c1", name="dev-cluster", state="RUNNING", tags=None):
    return compute.ClusterDetails(cluster_id=cid, cluster_name=name, state=compute.State(state),
                                  spark_version="15.4.x-scala2.12", node_type_id="i3.xlarge", num_workers=1,
                                  custom_tags=tags)


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=TOOLSETS)


async def test_list_clusters_paginated_and_filtered(h):
    h.w.clusters.list.return_value = [_cluster(f"c{i}", f"n{i}") for i in range(3)]
    out = await h.call("manage_cluster", {"action": "list", "page_size": 2, "states": ["RUNNING"]})
    assert [c["cluster_id"] for c in out["data"]] == ["c0", "c1"]
    assert out["page"]["has_more"]
    filter_by = h.w.clusters.list.call_args.kwargs["filter_by"]
    assert filter_by.cluster_states == [compute.State.RUNNING]


async def test_list_clusters_invalid_state(h):
    err = await h.call_error("manage_cluster", {"action": "list", "states": ["SLEEPING"]})
    assert "invalid" in err and "cluster_states" in err


async def test_create_cluster_validates_and_tracks(h, tmp_path):
    h.w.clusters.create.return_value = Wait(lambda **kw: None, response=compute.CreateClusterResponse(cluster_id="new-1"))
    h.w.clusters.get.return_value = _cluster("new-1", "my-cluster", "PENDING")
    out = await h.call("manage_cluster", {"action": "create", "spec": {
        "cluster_name": "my-cluster", "spark_version": "15.4.x-scala2.12", "node_type_id": "i3.xlarge",
        "num_workers": 1, "autotermination_minutes": 30}})
    assert out["status"] == "pending"
    assert out["data"]["cluster_id"] == "new-1"
    assert h.settings.manifest_path.exists()
    assert "new-1" in h.settings.manifest_path.read_text()

    err = await h.call_error("manage_cluster", {"action": "create", "spec": {"spark_version": "x", "nodes": 3}})
    assert "Unknown field(s) in spec: nodes" in err


async def test_create_cluster_without_autotermination_warns(h):
    h.w.clusters.create.return_value = Wait(lambda **kw: None, response=compute.CreateClusterResponse(cluster_id="n2"))
    h.w.clusters.get.return_value = _cluster("n2", state="PENDING")
    out = await h.call("manage_cluster", {"action": "create", "spec": {"spark_version": "x", "num_workers": 0}})
    assert any("autotermination" in w for w in out["warnings"])


async def test_terminate_requires_confirmation(h):
    h.w.clusters.get.return_value = _cluster()
    out = await h.call("manage_cluster", {"action": "terminate", "cluster_id": "c1"})
    assert out["status"] == "confirmation_required"
    assert out["plan"]["reversible"] is True
    assert any("interrupted" in w for w in out["plan"]["warnings"])
    h.w.clusters.delete.assert_not_called()

    out = await h.call("manage_cluster", {"action": "terminate", "cluster_id": "c1", "confirm": True})
    h.w.clusters.delete.assert_called_once_with("c1")
    h.w.clusters.permanent_delete.assert_not_called()


async def test_permanent_delete_is_irreversible_and_untracks(h):
    h.w.clusters.get.return_value = _cluster(state="TERMINATED")
    plan = (await h.call("manage_cluster", {"action": "delete", "cluster_id": "c1"}))["plan"]
    assert plan["reversible"] is False and "PERMANENTLY" in plan["description"]
    out = await h.call("manage_cluster", {"action": "delete", "cluster_id": "c1", "confirm": True})
    h.w.clusters.permanent_delete.assert_called_once_with("c1")
    assert out["data"]["deleted"] is True


@pytest.mark.parametrize("name,tags", [("prod-etl", None), ("etl", {"env": "production"}), ("Analytics PROD", None)])
async def test_protected_cluster_is_refused(h, name, tags):
    h.w.clusters.get.return_value = _cluster(name=name, tags=tags)
    err = await h.call_error("manage_cluster", {"action": "terminate", "cluster_id": "c1"})
    assert "BLOCKED_BY_SAFETY_POLICY" in err and "protected" in err
    err = await h.call_error("manage_cluster", {"action": "terminate", "cluster_id": "c1", "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in err
    h.w.clusters.delete.assert_not_called()


async def test_protected_override(make_harness):
    h = make_harness(toolsets=TOOLSETS, allow_protected_changes=True)
    h.w.clusters.get.return_value = _cluster(name="prod-etl")
    await h.call("manage_cluster", {"action": "terminate", "cluster_id": "c1", "confirm": True})
    h.w.clusters.delete.assert_called_once()


async def test_non_prod_names_not_protected(h):
    h.w.clusters.get.return_value = _cluster(name="product-analytics")
    out = await h.call("manage_cluster", {"action": "terminate", "cluster_id": "c1", "confirm": True})
    assert out["status"] in {"success", "pending"}


async def test_update_cluster_partial(h):
    h.w.clusters.get.return_value = _cluster(state="TERMINATED")
    out = await h.call("manage_cluster", {"action": "update", "cluster_id": "c1",
                                          "spec": {"autotermination_minutes": 20, "num_workers": 2}})
    args = h.w.clusters.update.call_args
    assert args.kwargs["update_mask"] == "autotermination_minutes,num_workers"
    assert args.kwargs["cluster"].num_workers == 2
    assert out["status"] == "success"


async def test_resize_and_wait_timeout(h):
    h.w.clusters.get.return_value = _cluster(state="RESIZING")
    h.w.clusters.wait_get_cluster_running.side_effect = TimeoutError("timed out")
    out = await h.call("manage_cluster", {"action": "resize", "cluster_id": "c1", "autoscale_min_workers": 1,
                                          "autoscale_max_workers": 4, "wait": True, "timeout_seconds": 5})
    assert out["status"] == "pending"
    autoscale = h.w.clusters.resize.call_args.kwargs["autoscale"]
    assert (autoscale.min_workers, autoscale.max_workers) == (1, 4)
    assert h.w.clusters.wait_get_cluster_running.call_args.kwargs["timeout"].total_seconds() == 5


async def test_get_missing_cluster(h):
    h.w.clusters.get.side_effect = NotFound("Cluster c9 does not exist")
    err = await h.call_error("manage_cluster", {"action": "get", "cluster_id": "c9"})
    assert "[NOT_FOUND]" in err


async def test_missing_cluster_id(h):
    err = await h.call_error("manage_cluster", {"action": "get"})
    assert "cluster_id" in err and "required" in err


async def test_read_only_blocks_cluster_changes(make_harness):
    h = make_harness(toolsets=TOOLSETS, read_only=True)
    h.w.clusters.list.return_value = []
    assert (await h.call("manage_cluster", {"action": "list"}))["status"] == "success"
    for action in ("create", "start", "terminate"):
        err = await h.call_error("manage_cluster", {"action": action, "cluster_id": "c1", "confirm": True})
        assert "BLOCKED_BY_SAFETY_POLICY" in err


# ---------------------------------------------------------------------------------------------- warehouses


def _warehouse(state="RUNNING", name="analytics", sessions=0):
    return sql.GetWarehouseResponse(id="w1", name=name, state=sql.State(state), cluster_size="Small",
                                    auto_stop_mins=10, max_num_clusters=1, min_num_clusters=1,
                                    num_active_sessions=sessions, enable_serverless_compute=True,
                                    warehouse_type=sql.GetWarehouseResponseWarehouseType.PRO)


async def test_warehouse_update_merges_current_config(h):
    h.w.warehouses.get.return_value = _warehouse()
    await h.call("manage_sql_warehouse", {"action": "update", "warehouse_id": "w1", "spec": {"auto_stop_mins": 5}})
    kwargs = h.w.warehouses.edit.call_args.kwargs
    assert kwargs["id"] == "w1" and kwargs["auto_stop_mins"] == 5
    assert kwargs["cluster_size"] == "Small" and kwargs["name"] == "analytics"  # preserved
    assert kwargs["warehouse_type"].value == "PRO"


async def test_warehouse_update_rejects_unknown(h):
    h.w.warehouses.get.return_value = _warehouse()
    err = await h.call_error("manage_sql_warehouse", {"action": "update", "warehouse_id": "w1", "spec": {"size": 1}})
    assert "non-editable" in err


async def test_warehouse_stop_confirmation_mentions_sessions(h):
    h.w.warehouses.get.return_value = _warehouse(sessions=3)
    out = await h.call("manage_sql_warehouse", {"action": "stop", "warehouse_id": "w1"})
    assert out["status"] == "confirmation_required"
    assert any("3 active session" in w for w in out["plan"]["warnings"])
    h.w.warehouses.stop.assert_not_called()


async def test_warehouse_delete_protected(h):
    h.w.warehouses.get.return_value = _warehouse(name="prod warehouse")
    err = await h.call_error("manage_sql_warehouse", {"action": "delete", "warehouse_id": "w1", "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in err
    h.w.warehouses.delete.assert_not_called()


async def test_warehouse_create(h):
    h.w.warehouses.create.return_value = Wait(lambda **kw: None, response=sql.CreateWarehouseResponse(id="w-new"))
    h.w.warehouses.get.return_value = _warehouse(state="STARTING")
    out = await h.call("manage_sql_warehouse", {"action": "create", "spec": {
        "name": "adhoc", "cluster_size": "2X-Small", "max_num_clusters": 1, "auto_stop_mins": 10,
        "enable_serverless_compute": True, "warehouse_type": "PRO"}})
    assert out["status"] == "pending"
    assert h.w.warehouses.create.call_args.kwargs["warehouse_type"].value == "PRO"


async def test_manage_warehouse_select_and_list(h):
    h.w.warehouses.list.return_value = [
        sql.EndpointInfo(id="a", name="stopped", state=sql.State.STOPPED),
        sql.EndpointInfo(id="b", name="running", state=sql.State.RUNNING),
    ]
    out = await h.call("manage_warehouse", {"action": "select"})
    assert out["data"]["warehouse_id"] == "b" and "ranked" in out["data"]["reason"]
    out = await h.call("manage_warehouse", {"action": "list"})
    assert [w["id"] for w in out["data"]] == ["b", "a"]


async def test_manage_warehouse_select_require_running(make_harness):
    h = make_harness(toolsets=TOOLSETS, default_warehouse_id="w1")
    h.w.warehouses.get.return_value = _warehouse(state="STOPPED")
    err = await h.call_error("manage_warehouse", {"action": "select", "require_running": True})
    assert "not RUNNING" in err


async def test_configured_only_selection(make_harness):
    h = make_harness(toolsets=TOOLSETS, warehouse_selection="configured_only")
    err = await h.call_error("manage_warehouse", {"action": "select"})
    assert "CONFIGURATION_ERROR" in err


# ---------------------------------------------------------------------------------------------- list_compute


async def test_list_compute_summary(h):
    h.w.clusters.list.return_value = [_cluster(), _cluster("c2", "other", "TERMINATED")]
    h.w.warehouses.list.return_value = [sql.EndpointInfo(id="w", name="wh", state=sql.State.RUNNING)]
    out = await h.call("list_compute")
    assert out["data"]["counts"] == {"clusters_running": 1, "clusters_listed": 2, "warehouses_running": 1,
                                     "warehouses_total": 1}


async def test_list_compute_node_types_and_versions(h):
    h.w.clusters.list_node_types.return_value = compute.ListNodeTypesResponse(node_types=[
        compute.NodeType(node_type_id="big", memory_mb=65536, num_cores=16, description="big", instance_type_id="x", category="General"),
        compute.NodeType(node_type_id="small", memory_mb=8192, num_cores=4, description="small", instance_type_id="y", category="General"),
        compute.NodeType(node_type_id="old", memory_mb=1, num_cores=1, description="old", instance_type_id="z", category="General",
                         is_deprecated=True),
    ])
    out = await h.call("list_compute", {"resource": "node_types"})
    assert [n["node_type_id"] for n in out["data"]] == ["small", "big"]
    assert out["data"][1]["memory_gb"] == 64.0

    h.w.clusters.spark_versions.return_value = compute.GetSparkVersionsResponse(versions=[
        compute.SparkVersion(key="14.3.x-scala2.12", name="14.3 LTS (includes Apache Spark 3.5.0, Scala 2.12)"),
        compute.SparkVersion(key="15.4.x-cpu-ml-scala2.12", name="15.4 LTS ML"),
    ])
    out = await h.call("list_compute", {"resource": "spark_versions", "filter": "lts"})
    assert out["data"][0] == {"key": "15.4.x-cpu-ml-scala2.12", "name": "15.4 LTS ML", "lts": True, "ml": True}
