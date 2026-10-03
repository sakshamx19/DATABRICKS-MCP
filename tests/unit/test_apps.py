"""Apps toolset tests: manage_app."""

from __future__ import annotations

import re

import pytest
from databricks.sdk.errors import NotFound, OperationFailed
from databricks.sdk.service._internal import Wait
from databricks.sdk.service.apps import (
    App,
    AppDeployment,
    AppDeploymentMode,
    AppDeploymentState,
    AppDeploymentStatus,
    AppUpdate,
    AppUpdateUpdateStatus,
    AppUpdateUpdateStatusUpdateState,
    ComputeState,
    ComputeStatus,
)

from dbx_mcp.server.context import get_context

TOOLSETS = ("apps",)


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=TOOLSETS)


def _app(name="my-app", state=ComputeState.STARTING):
    return App(name=name, id="a1", url=f"https://{name}.example.databricksapps.com",
               compute_status=ComputeStatus(state=state))


def _wait(response, result=None, exc=None):
    def waiter(**kwargs):
        if exc is not None:
            raise exc
        return result
    return Wait(waiter, response=response)


async def test_create_returns_pending_and_tracks(h):
    h.w.apps.create.return_value = _wait(_app())
    out = await h.call("manage_app", {"action": "create", "name": "my-app", "app": {"description": "demo"}})
    assert out["status"] == "pending"
    assert out["data"]["compute_state"] == "STARTING" and out["data"]["url"]
    sent = h.w.apps.create.call_args.args[0]
    assert isinstance(sent, App) and sent.name == "my-app" and sent.description == "demo"
    tracked = get_context().manifest.list("app")
    assert [t.resource_id for t in tracked] == ["my-app"]


async def test_create_with_bounded_wait(h):
    h.w.apps.create.return_value = _wait(_app(), result=_app(state=ComputeState.ACTIVE))
    out = await h.call("manage_app", {"action": "create", "name": "my-app", "wait": True, "timeout_seconds": 5})
    assert out["status"] == "success" and out["data"]["compute_state"] == "ACTIVE"


async def test_create_wait_timeout_returns_pending(h):
    h.w.apps.create.return_value = _wait(_app(), exc=TimeoutError("timed out"))
    h.w.apps.get.return_value = _app()
    out = await h.call("manage_app", {"action": "create", "name": "my-app", "wait": True, "timeout_seconds": 1})
    assert out["status"] == "pending" and "Still in progress" in out["summary"]


async def test_create_validation(h):
    err = await h.call_error("manage_app", {"action": "create", "name": "My_App"})
    assert "lowercase" in err
    err = await h.call_error("manage_app", {"action": "create", "name": "ok-app", "app": {"bogus_field": 1}})
    assert "bogus_field" in err
    h.w.apps.create.assert_not_called()


async def test_create_with_resources_is_security_sensitive(h):
    h.w.apps.create.return_value = _wait(_app())
    args = {"action": "create", "name": "my-app",
            "app": {"resources": [{"name": "wh", "sql_warehouse": {"id": "w1", "permission": "CAN_USE"}}]}}
    out = await h.call("manage_app", args)
    assert out["status"] == "confirmation_required" and "SECURITY_SENSITIVE" in out["safety"]
    h.w.apps.create.assert_not_called()
    out = await h.call("manage_app", {**args, "confirm": True})
    assert out["status"] == "pending"


async def test_get_list_and_not_found(h):
    h.w.apps.list.return_value = iter([_app("a-1"), _app("a-2")])
    out = await h.call("manage_app", {"action": "list"})
    assert [a["name"] for a in out["data"]] == ["a-1", "a-2"]
    h.w.apps.get.side_effect = NotFound("App does not exist")
    err = await h.call_error("manage_app", {"action": "get", "name": "missing"})
    assert "[NOT_FOUND]" in err


async def test_update_is_partial_via_update_mask(h):
    upd = AppUpdate(description="new", status=AppUpdateUpdateStatus(state=AppUpdateUpdateStatusUpdateState.IN_PROGRESS))
    h.w.apps.create_update.return_value = _wait(upd)
    out = await h.call("manage_app", {"action": "update", "name": "my-app", "app": {"description": "new"}})
    assert out["status"] == "pending"
    call = h.w.apps.create_update.call_args
    assert call.args[:2] == ("my-app", "description")
    assert call.kwargs["app"].description == "new"
    err = await h.call_error("manage_app", {"action": "update", "name": "my-app", "app": {"name": "other"}})
    assert "Renaming" in err


async def test_deploy(h):
    dep = AppDeployment(deployment_id="dep1", source_code_path="/Workspace/Users/me/app",
                        status=AppDeploymentStatus(state=AppDeploymentState.IN_PROGRESS))
    h.w.apps.get.return_value = _app()
    h.w.apps.deploy.return_value = _wait(dep)
    out = await h.call("manage_app", {"action": "deploy", "name": "my-app",
                                      "source_code_path": "/Workspace/Users/me/app", "mode": "SNAPSHOT"})
    assert out["status"] == "pending" and out["data"]["deployment_id"] == "dep1"
    sent = h.w.apps.deploy.call_args.args[1]
    assert sent.source_code_path == "/Workspace/Users/me/app" and sent.mode == AppDeploymentMode.SNAPSHOT

    err = await h.call_error("manage_app", {"action": "deploy", "name": "my-app", "source_code_path": "/Users/../x"})
    assert "[INVALID_PARAMETER]" in err


async def test_deploy_wait_failure(h):
    dep = AppDeployment(deployment_id="dep1", status=AppDeploymentStatus(state=AppDeploymentState.IN_PROGRESS))
    failed = AppDeployment(deployment_id="dep1", status=AppDeploymentStatus(state=AppDeploymentState.FAILED,
                                                                            message="bad requirements"))
    h.w.apps.deploy.return_value = _wait(dep, exc=OperationFailed("failed to reach SUCCEEDED"))
    h.w.apps.get_deployment.return_value = failed
    out = await h.call("manage_app", {"action": "deploy", "name": "my-app", "wait": True, "timeout_seconds": 1})
    assert out["status"] == "failed" and out["data"]["message"] == "bad requirements"


async def test_deployments(h):
    h.w.apps.list_deployments.return_value = iter([AppDeployment(deployment_id="d1"), AppDeployment(deployment_id="d2")])
    out = await h.call("manage_app", {"action": "list_deployments", "name": "my-app"})
    assert [d["deployment_id"] for d in out["data"]] == ["d1", "d2"]
    h.w.apps.get_deployment.return_value = AppDeployment(
        deployment_id="d1", status=AppDeploymentStatus(state=AppDeploymentState.SUCCEEDED))
    out = await h.call("manage_app", {"action": "get_deployment", "name": "my-app", "deployment_id": "d1"})
    assert "SUCCEEDED" in out["summary"]


async def test_start_stop(h):
    h.w.apps.start.return_value = _wait(_app())
    out = await h.call("manage_app", {"action": "start", "name": "my-app"})
    assert out["status"] == "pending"
    h.w.apps.stop.return_value = _wait(_app(state=ComputeState.STOPPING), result=_app(state=ComputeState.STOPPED))
    out = await h.call("manage_app", {"action": "stop", "name": "my-app", "wait": True})
    assert out["status"] == "success" and out["data"]["compute_state"] == "STOPPED"


async def test_delete_requires_confirm_and_untracks(h):
    get_context().manifest.safe_track(resource_type="app", resource_id="my-app", name="my-app")
    h.w.apps.get.return_value = _app()
    out = await h.call("manage_app", {"action": "delete", "name": "my-app"})
    assert out["status"] == "confirmation_required"
    assert "my-app" in out["plan"]["description"] and out["plan"]["reversible"] is False
    h.w.apps.delete.assert_not_called()
    h.w.apps.delete.return_value = _app()
    out = await h.call("manage_app", {"action": "delete", "name": "my-app", "confirm": True})
    assert out["status"] == "success"
    h.w.apps.delete.assert_called_once_with("my-app")
    assert get_context().manifest.list("app") == []


async def test_delete_protected_app_blocked(make_harness):
    h = make_harness(toolsets=TOOLSETS, protected_name_patterns=(re.compile(r"(^|-)prod($|-)"),))
    h.w.apps.get.return_value = _app("sales-prod")
    err = await h.call_error("manage_app", {"action": "delete", "name": "sales-prod", "confirm": True})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err
    h.w.apps.delete.assert_not_called()


async def test_logs_unsupported(h):
    err = await h.call_error("manage_app", {"action": "logs", "name": "my-app"})
    assert "[UNSUPPORTED_OPERATION]" in err


async def test_read_only_blocks(make_harness):
    h = make_harness(toolsets=TOOLSETS, read_only=True)
    err = await h.call_error("manage_app", {"action": "deploy", "name": "my-app"})
    assert "read-only" in err
    h.w.apps.deploy.assert_not_called()
