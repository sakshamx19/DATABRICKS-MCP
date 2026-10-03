"""Unit tests for manage_pipeline / manage_pipeline_run (mocked WorkspaceClient)."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service import pipelines as pl
from databricks.sdk.service._internal import Wait

import dbx_mcp.tools.pipelines as pipeline_tools


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(pipeline_tools, "POLL_INTERVAL_SECONDS", 0.05)


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=("pipelines",))


def _pipeline(name="sales-dev", tags=None, state="IDLE") -> pl.GetPipelineResponse:
    return pl.GetPipelineResponse(
        pipeline_id="p1",
        name=name,
        state=pl.PipelineState(state),
        creator_user_name="me@example.com",
        last_modified=1234,
        spec=pl.PipelineSpec(name=name, catalog="main", schema="sales", development=True, tags=tags,
                             libraries=[pl.PipelineLibrary(notebook=pl.NotebookLibrary(path="/Users/me/dlt"))]),
        latest_updates=[pl.UpdateStateInfo(update_id="u1", state=pl.UpdateStateInfoState.COMPLETED)],
    )


def _update(state: str, update_id="u2") -> pl.GetUpdateResponse:
    return pl.GetUpdateResponse(update=pl.UpdateInfo(update_id=update_id, pipeline_id="p1",
                                                     state=pl.UpdateInfoState(state)))


# ---------------------------------------------------------------------------------------------
# manage_pipeline
# ---------------------------------------------------------------------------------------------

async def test_tools_registered(h):
    assert {"manage_pipeline", "manage_pipeline_run"} <= h.tool_names()


async def test_list_paginates_and_filters(h):
    items = [pl.PipelineStateInfo(pipeline_id=f"p{i}", name=f"n{i}", state=pl.PipelineState.IDLE) for i in range(3)]
    h.w.pipelines.list_pipelines.side_effect = lambda **kw: iter(items)
    result = await h.call("manage_pipeline", {"action": "list", "name_contains": "sales", "page_size": 2})
    assert [p["pipeline_id"] for p in result["data"]] == ["p0", "p1"]
    assert result["data"][0]["state"] == "IDLE"
    assert result["page"]["has_more"] is True
    assert h.w.pipelines.list_pipelines.call_args.kwargs["filter"] == "name LIKE '%sales%'"
    second = await h.call("manage_pipeline", {"action": "list", "page_size": 2,
                                              "page_token": result["page"]["next_page_token"]})
    assert [p["pipeline_id"] for p in second["data"]] == ["p2"]
    assert second["page"]["has_more"] is False


async def test_list_rejects_quote_injection(h):
    message = await h.call_error("manage_pipeline", {"action": "list", "name_contains": "x' OR '1'='1"})
    assert "[INVALID_PARAMETER]" in message


async def test_get(h):
    h.w.pipelines.get.return_value = _pipeline()
    result = await h.call("manage_pipeline", {"action": "get", "pipeline_id": "p1"})
    assert result["data"]["spec"]["catalog"] == "main"


async def test_get_not_found(h):
    h.w.pipelines.get.side_effect = NotFound("no such pipeline")
    assert "[NOT_FOUND]" in await h.call_error("manage_pipeline", {"action": "get", "pipeline_id": "zz"})


async def test_create_and_track(h):
    h.w.pipelines.create.return_value = pl.CreatePipelineResponse(pipeline_id="new1")
    spec = {"name": "orders", "catalog": "main", "schema": "bronze", "serverless": True,
            "libraries": [{"glob": {"include": "/Users/me/orders/**"}}]}
    result = await h.call("manage_pipeline", {"action": "create", "spec": spec})
    assert result["data"]["pipeline_id"] == "new1"
    kwargs = h.w.pipelines.create.call_args.kwargs
    assert isinstance(kwargs["libraries"][0], pl.PipelineLibrary)
    manifest = json.loads(h.settings.manifest_path.read_text())
    assert manifest["resources"][0]["resource_type"] == "pipeline"


async def test_create_rejects_unknown_field(h):
    message = await h.call_error("manage_pipeline", {"action": "create", "spec": {"name": "x", "nope": True}})
    assert "[INVALID_PARAMETER]" in message and "nope" in message
    h.w.pipelines.create.assert_not_called()


async def test_update_merges_onto_current_spec(h):
    h.w.pipelines.get.return_value = _pipeline()
    result = await h.call("manage_pipeline", {"action": "update", "pipeline_id": "p1",
                                              "spec": {"development": False, "catalog": None}})
    assert result["status"] == "success"
    kwargs = h.w.pipelines.update.call_args.kwargs
    assert kwargs["pipeline_id"] == "p1"
    assert kwargs["name"] == "sales-dev"           # preserved
    assert kwargs["development"] is False          # changed
    assert "catalog" not in kwargs                 # removed via null
    assert kwargs["expected_last_modified"] == 1234
    assert isinstance(kwargs["libraries"][0], pl.PipelineLibrary)


async def test_update_dry_run(h):
    result = await h.call("manage_pipeline", {"action": "update", "pipeline_id": "p1",
                                              "spec": {"development": False}, "dry_run": True})
    assert result["status"] == "dry_run"
    h.w.pipelines.update.assert_not_called()


async def test_delete_requires_confirmation_and_warns_cascade(h):
    h.w.pipelines.get.return_value = _pipeline()
    result = await h.call("manage_pipeline", {"action": "delete", "pipeline_id": "p1"})
    assert result["status"] == "confirmation_required"
    assert any("cascade" in w for w in result["plan"]["warnings"])
    h.w.pipelines.delete.assert_not_called()
    result = await h.call("manage_pipeline", {"action": "delete", "pipeline_id": "p1", "cascade": False, "confirm": True})
    assert result["status"] == "success"
    h.w.pipelines.delete.assert_called_once_with("p1", cascade=False, force=None)


async def test_delete_protected_blocked(h):
    h.w.pipelines.get.return_value = _pipeline(name="sales_prod")
    message = await h.call_error("manage_pipeline", {"action": "delete", "pipeline_id": "p1", "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.pipelines.delete.assert_not_called()


async def test_read_only_blocks_create(make_harness):
    h = make_harness(read_only=True, toolsets=("pipelines",))
    message = await h.call_error("manage_pipeline", {"action": "create", "spec": {"name": "x"}})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.pipelines.create.assert_not_called()


async def test_clone(h):
    h.w.pipelines.clone.return_value = pl.ClonePipelineResponse(pipeline_id="clone1")
    result = await h.call("manage_pipeline", {"action": "clone", "pipeline_id": "p1",
                                              "spec": {"catalog": "main", "target": "s", "clone_mode": "MIGRATE_TO_UC"}})
    assert result["status"] == "pending"
    kwargs = h.w.pipelines.clone.call_args.kwargs
    assert kwargs["pipeline_id"] == "p1" and kwargs["clone_mode"] == pl.CloneMode.MIGRATE_TO_UC


# ---------------------------------------------------------------------------------------------
# manage_pipeline_run
# ---------------------------------------------------------------------------------------------

async def test_start_returns_pending(h):
    h.w.pipelines.start_update.return_value = pl.StartUpdateResponse(update_id="u2")
    result = await h.call("manage_pipeline_run", {"action": "start", "pipeline_id": "p1",
                                                  "refresh_selection": ["orders"]})
    assert result["status"] == "pending"
    assert result["data"]["update_id"] == "u2"
    kwargs = h.w.pipelines.start_update.call_args.kwargs
    assert kwargs["refresh_selection"] == ["orders"]


async def test_full_refresh_requires_confirmation(h):
    h.w.pipelines.get.return_value = _pipeline()
    result = await h.call("manage_pipeline_run", {"action": "start", "pipeline_id": "p1", "full_refresh": True})
    assert result["status"] == "confirmation_required"
    assert "DESTRUCTIVE" in result["safety"]
    h.w.pipelines.start_update.assert_not_called()

    h.w.pipelines.start_update.return_value = pl.StartUpdateResponse(update_id="u3")
    result = await h.call("manage_pipeline_run",
                          {"action": "start", "pipeline_id": "p1", "full_refresh": True, "confirm": True})
    assert result["status"] == "pending"
    assert h.w.pipelines.start_update.call_args.kwargs["full_refresh"] is True


async def test_validate_only_full_refresh_is_not_destructive(h):
    h.w.pipelines.start_update.return_value = pl.StartUpdateResponse(update_id="u4")
    result = await h.call("manage_pipeline_run", {"action": "start", "pipeline_id": "p1",
                                                  "full_refresh": True, "validate_only": True})
    assert result["status"] == "pending"


async def test_full_refresh_on_protected_pipeline_blocked(h):
    h.w.pipelines.get.return_value = _pipeline(name="orders-prod")
    message = await h.call_error("manage_pipeline_run",
                                 {"action": "start", "pipeline_id": "p1", "full_refresh": True, "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.pipelines.start_update.assert_not_called()


async def test_stop_requires_confirmation(h):
    h.w.pipelines.get.return_value = _pipeline(state="RUNNING")
    result = await h.call("manage_pipeline_run", {"action": "stop", "pipeline_id": "p1"})
    assert result["status"] == "confirmation_required"
    h.w.pipelines.stop.assert_not_called()
    h.w.pipelines.stop.return_value = Wait(lambda **kw: None)
    result = await h.call("manage_pipeline_run", {"action": "stop", "pipeline_id": "p1", "confirm": True})
    assert result["status"] == "success"
    h.w.pipelines.stop.assert_called_once_with("p1")


async def test_wait_failed_update_surfaces_error_events(h):
    h.w.pipelines.get_update.return_value = _update("FAILED")
    events = [
        pl.PipelineEvent(id="e1", level=pl.EventLevel.ERROR, message="other update failed",
                         origin=pl.Origin(update_id="u-other")),
        pl.PipelineEvent(id="e2", level=pl.EventLevel.ERROR, message="Table orders: column x not found",
                         origin=pl.Origin(update_id="u2", flow_name="orders"),
                         error=pl.ErrorDetail(fatal=True, exceptions=[
                             pl.SerializedException(class_name="AnalysisException", message="column x not found")])),
    ]
    h.w.pipelines.list_pipeline_events.return_value = iter(events)
    result = await h.call("manage_pipeline_run", {"action": "wait", "pipeline_id": "p1", "update_id": "u2",
                                                  "timeout_seconds": 5})
    assert result["status"] == "failed"
    errors = result["data"]["errors"]
    assert [e["id"] for e in errors] == ["e2"]
    assert errors[0]["error"]["exceptions"][0]["class_name"] == "AnalysisException"
    assert h.w.pipelines.list_pipeline_events.call_args.kwargs["filter"] == "level='ERROR'"


async def test_wait_defaults_to_latest_update_and_times_out_pending(h):
    h.w.pipelines.get.return_value = _pipeline()
    h.w.pipelines.get_update.return_value = _update("RUNNING", update_id="u1")
    result = await h.call("manage_pipeline_run", {"action": "wait", "pipeline_id": "p1", "timeout_seconds": 1})
    assert result["status"] == "pending"
    h.w.pipelines.get_update.assert_called_with("p1", "u1")


async def test_get_update_completed(h):
    h.w.pipelines.get_update.return_value = _update("COMPLETED")
    result = await h.call("manage_pipeline_run", {"action": "get_update", "pipeline_id": "p1", "update_id": "u2"})
    assert result["data"]["update"]["state"] == "COMPLETED"
    assert "errors" not in result["data"]


async def test_list_updates_follows_server_pages(h):
    pages = {
        None: pl.ListUpdatesResponse(updates=[pl.UpdateInfo(update_id="a"), pl.UpdateInfo(update_id="b")],
                                     next_page_token="t2"),
        "t2": pl.ListUpdatesResponse(updates=[pl.UpdateInfo(update_id="c")]),
    }
    h.w.pipelines.list_updates.side_effect = lambda pid, max_results=None, page_token=None, **kw: pages[page_token]
    result = await h.call("manage_pipeline_run", {"action": "list_updates", "pipeline_id": "p1", "page_size": 3})
    assert [u["update_id"] for u in result["data"]] == ["a", "b", "c"]


async def test_list_events_level_filter(h):
    h.w.pipelines.list_pipeline_events.return_value = iter(
        [pl.PipelineEvent(id="e1", level=pl.EventLevel.ERROR, message="boom", event_type="update_progress")]
    )
    result = await h.call("manage_pipeline_run", {"action": "list_events", "pipeline_id": "p1", "level": "ERROR"})
    assert result["data"][0]["message"] == "boom"
    assert h.w.pipelines.list_pipeline_events.call_args.kwargs["filter"] == "level='ERROR'"


async def test_read_only_allows_monitoring_but_blocks_start(make_harness):
    h = make_harness(read_only=True, toolsets=("pipelines",))
    message = await h.call_error("manage_pipeline_run", {"action": "start", "pipeline_id": "p1"})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.pipelines.get_update.return_value = _update("COMPLETED")
    result = await h.call("manage_pipeline_run", {"action": "get_update", "pipeline_id": "p1", "update_id": "u2"})
    assert result["status"] == "success"
