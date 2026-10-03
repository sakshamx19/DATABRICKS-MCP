"""Manifest toolset tests: list_tracked_resources, delete_tracked_resource."""

from __future__ import annotations

import pytest
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service.dashboards import Dashboard, LifecycleState

from dbx_mcp.server.context import get_context
from dbx_mcp.server.manifest import TrackedResource

TOOLSETS = ("manifest",)
HOST = "https://example.cloud.databricks.com"


@pytest.fixture
def h(make_harness):
    harness = make_harness(toolsets=TOOLSETS)
    store = get_context().manifest
    for rtype, rid, name in [
        ("job", "101", "nightly"),
        ("job", "102", "gone-job"),
        ("dashboard", "d1", "sales"),
        ("app", "my-app", "my-app"),
        ("vector_index", "c.s.i", "idx"),
    ]:
        store.track(TrackedResource(resource_type=rtype, resource_id=rid, name=name, workspace_host=HOST,
                                    created_by_tool="test"))
    store.track(TrackedResource(resource_type="cluster", resource_id="c9", workspace_host="https://other.example.com"))
    return harness


async def test_list_and_filter(h):
    out = await h.call("list_tracked_resources")
    assert len(out["data"]["resources"]) == 6
    out = await h.call("list_tracked_resources", {"resource_type": "job"})
    assert [r["resource_id"] for r in out["data"]["resources"]] == ["101", "102"]
    assert "verification" not in out["data"]["resources"][0]
    h.w.jobs.get.assert_not_called()


async def test_list_paginated(h):
    out = await h.call("list_tracked_resources", {"page_size": 4})
    assert len(out["data"]["resources"]) == 4 and out["page"]["has_more"]
    nxt = await h.call("list_tracked_resources", {"page_size": 4, "page_token": out["page"]["next_page_token"]})
    assert len(nxt["data"]["resources"]) == 2


async def test_verify_reports_missing(h):
    def job_get(job_id):
        assert isinstance(job_id, int)
        if job_id == 102:
            raise NotFound("Job 102 does not exist")
        return object()

    h.w.jobs.get.side_effect = job_get
    h.w.lakeview.get.return_value = Dashboard(dashboard_id="d1", lifecycle_state=LifecycleState.TRASHED)
    h.w.apps.get.side_effect = PermissionDenied("nope")
    out = await h.call("list_tracked_resources", {"verify": True})
    status = {r["resource_id"]: r["verification"]["status"] for r in out["data"]["resources"]}
    assert status == {
        "101": "exists",
        "102": "missing",
        "d1": "trashed",
        "my-app": "unknown",
        "c.s.i": "not_verified",
        "c9": "other_workspace",
    }
    assert {m["resource_id"] for m in out["data"]["missing"]} == {"102", "d1"}
    assert out["warnings"]
    h.w.clusters.get.assert_not_called()


async def test_delete_tracked_resource_states_not_deleted(h):
    out = await h.call("delete_tracked_resource", {"resource_type": "job", "resource_id": "101"})
    assert out["status"] == "success"
    assert "NOT deleted" in out["summary"]
    assert out["data"]["databricks_resource_deleted"] is False
    h.w.jobs.delete.assert_not_called()
    remaining = await h.call("list_tracked_resources", {"resource_type": "job"})
    assert [r["resource_id"] for r in remaining["data"]["resources"]] == ["102"]


async def test_delete_tracked_resource_dry_run_and_missing(h):
    out = await h.call("delete_tracked_resource", {"resource_type": "job", "resource_id": "101", "dry_run": True})
    assert out["status"] == "dry_run" and "NOT deleted" in out["plan"]["description"]
    assert len(get_context().manifest.list("job")) == 2
    err = await h.call_error("delete_tracked_resource", {"resource_type": "job", "resource_id": "999"})
    assert "[NOT_FOUND]" in err


async def test_read_only_blocks_untrack(make_harness):
    h = make_harness(toolsets=TOOLSETS, read_only=True)
    err = await h.call_error("delete_tracked_resource", {"resource_type": "job", "resource_id": "1"})
    assert "read-only" in err
