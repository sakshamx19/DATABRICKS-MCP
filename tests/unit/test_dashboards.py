"""Dashboards toolset tests: manage_dashboard (AI/BI / Lakeview)."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.dashboards import Dashboard, LifecycleState, PublishedDashboard

from dbx_mcp.server.context import get_context

TOOLSETS = ("dashboards",)
DEF = {"pages": [{"name": "p1", "displayName": "Page 1"}], "datasets": []}


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=TOOLSETS)


def _dash(did="d1", name="Sales", **kw):
    return Dashboard(dashboard_id=did, display_name=name, path=f"/Users/me/{name}.lvdash.json",
                     parent_path="/Users/me", lifecycle_state=LifecycleState.ACTIVE, warehouse_id="wh1", **kw)


async def test_create_tracks_and_serializes(h):
    h.w.lakeview.create.return_value = _dash()
    out = await h.call("manage_dashboard", {"action": "create", "display_name": "Sales", "parent_path": "/Users/me",
                                            "warehouse_id": "wh1", "serialized_dashboard": DEF})
    assert out["status"] == "success" and out["data"]["dashboard_id"] == "d1"
    sent = h.w.lakeview.create.call_args.args[0]
    assert isinstance(sent, Dashboard)
    assert json.loads(sent.serialized_dashboard) == DEF and sent.parent_path == "/Users/me"
    tracked = get_context().manifest.list("dashboard")
    assert [(t.resource_id, t.name) for t in tracked] == [("d1", "Sales")]


async def test_create_validation(h):
    err = await h.call_error("manage_dashboard", {"action": "create", "display_name": "x", "serialized_dashboard": "{bad"})
    assert "not valid JSON" in err
    err = await h.call_error("manage_dashboard", {"action": "create", "display_name": "x", "serialized_dashboard": "[1]"})
    assert "JSON object" in err or "serialized_dashboard" in err  # rejected by us or by argument validation
    err = await h.call_error("manage_dashboard", {"action": "create", "display_name": "x", "parent_path": "/Users/../etc"})
    assert "[INVALID_PARAMETER]" in err
    err = await h.call_error("manage_dashboard", {"action": "create"})
    assert "display_name" in err
    h.w.lakeview.create.assert_not_called()


async def test_parent_path_allowlist(make_harness):
    h = make_harness(toolsets=TOOLSETS, allowed_workspace_prefixes=("/Shared/team",))
    err = await h.call_error("manage_dashboard", {"action": "create", "display_name": "x", "parent_path": "/Users/me"})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err


async def test_list_and_get(h):
    h.w.lakeview.list.return_value = iter([_dash("d1"), _dash("d2", "Ops", serialized_dashboard="{}")])
    out = await h.call("manage_dashboard", {"action": "list", "page_size": 1})
    assert len(out["data"]) == 1
    assert out["data"][0]["dashboard_id"] == "d1" and "serialized_dashboard" not in out["data"][0]
    assert out["page"]["has_more"]

    h.w.lakeview.get.return_value = _dash(serialized_dashboard=json.dumps(DEF))
    out = await h.call("manage_dashboard", {"action": "get", "dashboard_id": "d1"})
    assert out["data"]["serialized_dashboard"]


async def test_get_not_found(h):
    h.w.lakeview.get.side_effect = NotFound("Dashboard d9 not found")
    err = await h.call_error("manage_dashboard", {"action": "get", "dashboard_id": "d9"})
    assert "[NOT_FOUND]" in err


async def test_update(h):
    h.w.lakeview.update.return_value = _dash(name="Renamed")
    out = await h.call("manage_dashboard", {"action": "update", "dashboard_id": "d1", "display_name": "Renamed",
                                            "etag": "e1"})
    assert out["status"] == "success"
    args = h.w.lakeview.update.call_args
    assert args.args[0] == "d1" and args.args[1].display_name == "Renamed" and args.args[1].etag == "e1"
    err = await h.call_error("manage_dashboard", {"action": "update", "dashboard_id": "d1"})
    assert "at least one" in err


async def test_delete_requires_confirm_then_trashes(h):
    h.w.lakeview.get.return_value = _dash()
    h.w.lakeview.get_published.side_effect = NotFound("not published")
    get_context().manifest.safe_track(resource_type="dashboard", resource_id="d1", name="Sales")
    args = {"action": "delete", "dashboard_id": "d1"}
    out = await h.call("manage_dashboard", args)
    assert out["status"] == "confirmation_required"
    assert "Sales" in out["plan"]["description"] and out["plan"]["reversible"] is True
    h.w.lakeview.trash.assert_not_called()

    out = await h.call("manage_dashboard", {**args, "confirm": True})
    assert out["status"] == "success"
    h.w.lakeview.trash.assert_called_once_with("d1")
    assert get_context().manifest.list("dashboard") == []


async def test_delete_protected_dashboard_blocked(make_harness):
    import re
    h = make_harness(toolsets=TOOLSETS, protected_name_patterns=(re.compile(r"(?i)prod"),))
    h.w.lakeview.get.return_value = _dash(name="prod-kpis")
    err = await h.call_error("manage_dashboard", {"action": "delete", "dashboard_id": "d1"})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err
    err = await h.call_error("manage_dashboard", {"action": "delete", "dashboard_id": "d1", "confirm": True})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err
    h.w.lakeview.trash.assert_not_called()


async def test_publish_without_and_with_embedded_credentials(h):
    h.w.lakeview.publish.return_value = PublishedDashboard(display_name="Sales", embed_credentials=False)
    out = await h.call("manage_dashboard", {"action": "publish", "dashboard_id": "d1"})
    assert out["status"] == "success"
    assert h.w.lakeview.publish.call_args.kwargs["embed_credentials"] is False

    h.w.lakeview.get.return_value = _dash()
    h.w.lakeview.publish.reset_mock()
    args = {"action": "publish", "dashboard_id": "d1", "embed_credentials": True}
    out = await h.call("manage_dashboard", args)
    assert out["status"] == "confirmation_required" and "SECURITY_SENSITIVE" in out["safety"]
    h.w.lakeview.publish.assert_not_called()
    out = await h.call("manage_dashboard", {**args, "confirm": True})
    assert out["status"] == "success"
    assert h.w.lakeview.publish.call_args.kwargs["embed_credentials"] is True


async def test_unpublish_and_get_published(h):
    h.w.lakeview.get.return_value = _dash()
    h.w.lakeview.get_published.return_value = PublishedDashboard(display_name="Sales")
    out = await h.call("manage_dashboard", {"action": "get_published", "dashboard_id": "d1"})
    assert out["data"]["display_name"] == "Sales"
    out = await h.call("manage_dashboard", {"action": "unpublish", "dashboard_id": "d1"})
    assert out["status"] == "confirmation_required"
    h.w.lakeview.unpublish.assert_not_called()
    await h.call("manage_dashboard", {"action": "unpublish", "dashboard_id": "d1", "confirm": True})
    h.w.lakeview.unpublish.assert_called_once_with("d1")


async def test_read_only_blocks_changes(make_harness):
    h = make_harness(toolsets=TOOLSETS, read_only=True)
    err = await h.call_error("manage_dashboard", {"action": "create", "display_name": "x"})
    assert "read-only" in err
    h.w.lakeview.list.return_value = iter([])
    out = await h.call("manage_dashboard", {"action": "list"})
    assert out["status"] == "success"
