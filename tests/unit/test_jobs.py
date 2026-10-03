"""Unit tests for manage_jobs / manage_job_runs (mocked WorkspaceClient)."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service import jobs
from databricks.sdk.service._internal import Wait

import dbx_mcp.tools.jobs as jobs_tools


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(jobs_tools, "POLL_INTERVAL_SECONDS", 0.05)


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=("jobs",))


def _wait(response) -> Wait:
    return Wait(lambda **kwargs: None, response=response)


def _job(job_id=1, name="etl-dev", tags=None) -> jobs.Job:
    return jobs.Job(
        job_id=job_id,
        creator_user_name="me@example.com",
        created_time=1700000000000,
        settings=jobs.JobSettings(
            name=name,
            tags=tags,
            tasks=[jobs.Task(task_key="t1", notebook_task=jobs.NotebookTask(notebook_path="/Users/me/nb"))],
            max_concurrent_runs=1,
        ),
    )


def _run(run_id=7, life="TERMINATED", result="SUCCESS", tasks=None, message="") -> jobs.Run:
    return jobs.Run(
        run_id=run_id,
        job_id=1,
        run_name="etl-dev",
        state=jobs.RunState(
            life_cycle_state=jobs.RunLifeCycleState(life),
            result_state=jobs.RunResultState(result) if result else None,
            state_message=message,
        ),
        tasks=tasks,
        run_page_url="https://example/run/7",
    )


# ---------------------------------------------------------------------------------------------
# manage_jobs
# ---------------------------------------------------------------------------------------------

async def test_tools_registered(h):
    assert {"manage_jobs", "manage_job_runs"} <= h.tool_names()


async def test_list_paginates(h):
    all_jobs = [jobs.BaseJob(job_id=i, settings=jobs.JobSettings(name=f"job{i}")) for i in range(5)]
    h.w.jobs.list.side_effect = lambda **kw: iter(all_jobs)
    first = await h.call("manage_jobs", {"action": "list", "page_size": 2})
    assert first["status"] == "success"
    assert [j["job_id"] for j in first["data"]] == [0, 1]
    assert first["data"][0]["name"] == "job0"
    assert first["page"]["has_more"] is True
    token = first["page"]["next_page_token"]
    second = await h.call("manage_jobs", {"action": "list", "page_size": 2, "page_token": token})
    assert [j["job_id"] for j in second["data"]] == [2, 3]


async def test_list_name_filter_is_server_side(h):
    h.w.jobs.list.return_value = iter([])
    await h.call("manage_jobs", {"action": "list", "name": "nightly"})
    h.w.jobs.list.assert_called_once_with(name="nightly")


async def test_get_returns_full_job(h):
    h.w.jobs.get.return_value = _job()
    result = await h.call("manage_jobs", {"action": "get", "job_id": 1})
    assert result["data"]["settings"]["name"] == "etl-dev"
    assert result["data"]["settings"]["tasks"][0]["task_key"] == "t1"


async def test_get_not_found_is_mapped(h):
    h.w.jobs.get.side_effect = NotFound("Job 99 does not exist")
    message = await h.call_error("manage_jobs", {"action": "get", "job_id": 99})
    assert "[NOT_FOUND]" in message


async def test_create_converts_spec_and_tracks(h):
    h.w.jobs.create.return_value = jobs.CreateResponse(job_id=42)
    spec = {
        "name": "nightly",
        "tasks": [{"task_key": "a", "notebook_task": {"notebook_path": "/Users/me/nb"}, "existing_cluster_id": "c1"}],
        "schedule": {"quartz_cron_expression": "0 0 2 * * ?", "timezone_id": "UTC"},
    }
    result = await h.call("manage_jobs", {"action": "create", "spec": spec})
    assert result["status"] == "success"
    assert result["data"]["job_id"] == 42
    kwargs = h.w.jobs.create.call_args.kwargs
    assert isinstance(kwargs["tasks"][0], jobs.Task)
    assert isinstance(kwargs["schedule"], jobs.CronSchedule)
    manifest = json.loads(h.settings.manifest_path.read_text())
    assert manifest["resources"][0]["resource_id"] == "42"
    assert manifest["resources"][0]["resource_type"] == "job"


async def test_create_rejects_unknown_field(h):
    message = await h.call_error("manage_jobs", {"action": "create", "spec": {"name": "x", "bogus_field": 1}})
    assert "[INVALID_PARAMETER]" in message and "bogus_field" in message
    h.w.jobs.create.assert_not_called()


async def test_create_rejects_unknown_nested_field(h):
    spec = {"name": "x", "tasks": [{"task_key": "a", "notebook_task": {"path": "/x"}}]}
    message = await h.call_error("manage_jobs", {"action": "create", "spec": spec})
    assert "[INVALID_PARAMETER]" in message
    h.w.jobs.create.assert_not_called()


async def test_create_with_run_as_requires_confirmation(h):
    spec = {"name": "x", "run_as": {"service_principal_name": "sp-1"}}
    result = await h.call("manage_jobs", {"action": "create", "spec": spec})
    assert result["status"] == "confirmation_required"
    assert "SECURITY_SENSITIVE" in result["safety"]
    h.w.jobs.create.assert_not_called()


async def test_update_is_partial(h):
    result = await h.call(
        "manage_jobs",
        {"action": "update", "job_id": 1, "spec": {"max_concurrent_runs": 3}, "fields_to_remove": ["schedule"]},
    )
    assert result["status"] == "success"
    kwargs = h.w.jobs.update.call_args.kwargs
    assert kwargs["new_settings"].as_dict() == {"max_concurrent_runs": 3}
    assert kwargs["fields_to_remove"] == ["schedule"]


async def test_update_dry_run_changes_nothing(h):
    result = await h.call("manage_jobs", {"action": "update", "job_id": 1, "spec": {"name": "y"}, "dry_run": True})
    assert result["status"] == "dry_run"
    h.w.jobs.update.assert_not_called()


async def test_reset_requires_confirmation_and_lists_removed_fields(h):
    h.w.jobs.get.return_value = _job()
    result = await h.call("manage_jobs", {"action": "reset", "job_id": 1, "spec": {"name": "etl-dev"}})
    assert result["status"] == "confirmation_required"
    assert "tasks" in result["plan"]["details"]["fields_removed"]
    h.w.jobs.reset.assert_not_called()

    result = await h.call("manage_jobs", {"action": "reset", "job_id": 1, "spec": {"name": "etl-dev"}, "confirm": True})
    assert result["status"] == "success"
    h.w.jobs.reset.assert_called_once()
    assert h.w.jobs.reset.call_args.kwargs["new_settings"].as_dict() == {"name": "etl-dev"}


async def test_delete_requires_confirmation(h):
    h.w.jobs.get.return_value = _job()
    result = await h.call("manage_jobs", {"action": "delete", "job_id": 1})
    assert result["status"] == "confirmation_required"
    assert result["plan"]["target"]["name"] == "etl-dev"
    assert result["plan"]["reversible"] is False
    h.w.jobs.delete.assert_not_called()


async def test_delete_with_confirm_deletes_and_untracks(h):
    h.w.jobs.create.return_value = jobs.CreateResponse(job_id=1)
    await h.call("manage_jobs", {"action": "create", "spec": {"name": "etl-dev"}})
    h.w.jobs.get.return_value = _job()
    result = await h.call("manage_jobs", {"action": "delete", "job_id": 1, "confirm": True})
    assert result["status"] == "success"
    h.w.jobs.delete.assert_called_once_with(1)
    manifest = json.loads(h.settings.manifest_path.read_text())
    assert manifest["resources"] == []


async def test_delete_protected_job_blocked(h):
    h.w.jobs.get.return_value = _job(name="sales-prod")
    message = await h.call_error("manage_jobs", {"action": "delete", "job_id": 1, "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.jobs.delete.assert_not_called()


async def test_delete_protected_by_tag_blocked_even_in_preview(h):
    h.w.jobs.get.return_value = _job(name="etl", tags={"env": "prod"})
    message = await h.call_error("manage_jobs", {"action": "delete", "job_id": 1})
    assert "BLOCKED_BY_SAFETY_POLICY" in message


async def test_read_only_blocks_writes(make_harness):
    h = make_harness(read_only=True, toolsets=("jobs",))
    message = await h.call_error("manage_jobs", {"action": "create", "spec": {"name": "x"}})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.jobs.create.assert_not_called()
    message = await h.call_error("manage_jobs", {"action": "run_now", "job_id": 1})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    # reads still work
    h.w.jobs.get.return_value = _job()
    assert (await h.call("manage_jobs", {"action": "get", "job_id": 1}))["status"] == "success"


async def test_run_now_returns_pending(h):
    h.w.jobs.run_now.return_value = _wait(jobs.RunNowResponse(run_id=7, number_in_job=3))
    result = await h.call("manage_jobs", {"action": "run_now", "job_id": 1, "spec": {"job_parameters": {"d": "1"}}})
    assert result["status"] == "pending"
    assert result["data"]["run_id"] == 7
    assert h.w.jobs.run_now.call_args.kwargs == {"job_id": 1, "job_parameters": {"d": "1"}}
    assert any("run_id=7" in s for s in result["next_steps"])


async def test_run_now_wait_surfaces_task_errors(h):
    h.w.jobs.run_now.return_value = _wait(jobs.RunNowResponse(run_id=7))
    failed_task = jobs.RunTask(
        task_key="t1",
        run_id=70,
        state=jobs.RunState(life_cycle_state=jobs.RunLifeCycleState.TERMINATED,
                            result_state=jobs.RunResultState.FAILED, state_message="Workload failed"),
    )
    h.w.jobs.get_run.return_value = _run(result="FAILED", tasks=[failed_task], message="Task t1 failed")
    h.w.jobs.get_run_output.return_value = jobs.RunOutput(error="ZeroDivisionError: division by zero",
                                                          error_trace="Traceback...")
    result = await h.call("manage_jobs", {"action": "run_now", "job_id": 1, "wait": True, "timeout_seconds": 5})
    assert result["status"] == "failed"
    errors = result["data"]["errors"]
    assert any(e.get("error") == "ZeroDivisionError: division by zero" for e in errors)
    h.w.jobs.get_run_output.assert_called_once_with(70)


async def test_run_now_wait_times_out_as_pending(h):
    h.w.jobs.run_now.return_value = _wait(jobs.RunNowResponse(run_id=7))
    h.w.jobs.get_run.return_value = _run(life="RUNNING", result=None)
    result = await h.call("manage_jobs", {"action": "run_now", "job_id": 1, "wait": True, "timeout_seconds": 1})
    assert result["status"] == "pending"
    assert result["data"]["run"]["life_cycle_state"] == "RUNNING"


# ---------------------------------------------------------------------------------------------
# manage_job_runs
# ---------------------------------------------------------------------------------------------

async def test_runs_list_paginates(h):
    runs = [jobs.BaseRun(run_id=i, job_id=1, state=jobs.RunState(life_cycle_state=jobs.RunLifeCycleState.TERMINATED))
            for i in range(3)]
    h.w.jobs.list_runs.side_effect = lambda **kw: iter(runs)
    result = await h.call("manage_job_runs", {"action": "list", "job_id": 1, "page_size": 2})
    assert [r["run_id"] for r in result["data"]] == [0, 1]
    assert result["page"]["next_page_token"]
    assert h.w.jobs.list_runs.call_args.kwargs["job_id"] == 1


async def test_runs_list_rejects_conflicting_filters(h):
    message = await h.call_error("manage_job_runs", {"action": "list", "active_only": True, "completed_only": True})
    assert "[INVALID_PARAMETER]" in message


async def test_get_run_includes_task_states(h):
    task = jobs.RunTask(task_key="t1", run_id=70,
                        state=jobs.RunState(life_cycle_state=jobs.RunLifeCycleState.RUNNING))
    h.w.jobs.get_run.return_value = _run(life="RUNNING", result=None, tasks=[task])
    result = await h.call("manage_job_runs", {"action": "get", "run_id": 7})
    assert result["data"]["run"]["tasks"][0]["life_cycle_state"] == "RUNNING"
    assert result["data"]["run"]["terminal"] is False


async def test_get_output_expands_multi_task_runs(h):
    tasks = [jobs.RunTask(task_key="a", run_id=71), jobs.RunTask(task_key="b", run_id=72)]
    h.w.jobs.get_run.return_value = _run(tasks=tasks)
    h.w.jobs.get_run_output.side_effect = lambda rid: jobs.RunOutput(
        notebook_output=jobs.NotebookOutput(result=f"out-{rid}"))
    result = await h.call("manage_job_runs", {"action": "get_output", "run_id": 7})
    outputs = result["data"]["outputs"]
    assert [(o["task_key"], o["notebook_output"]["result"]) for o in outputs] == [("a", "out-71"), ("b", "out-72")]


async def test_get_output_single_task_filter(h):
    tasks = [jobs.RunTask(task_key="a", run_id=71), jobs.RunTask(task_key="b", run_id=72)]
    h.w.jobs.get_run.return_value = _run(tasks=tasks)
    h.w.jobs.get_run_output.return_value = jobs.RunOutput(logs="hello")
    result = await h.call("manage_job_runs", {"action": "get_output", "run_id": 7, "task_key": "b"})
    h.w.jobs.get_run_output.assert_called_once_with(72)
    assert result["data"]["outputs"][0]["logs"] == "hello"


async def test_submit_returns_pending(h):
    h.w.jobs.submit.return_value = _wait(jobs.SubmitRunResponse(run_id=99))
    spec = {"run_name": "adhoc", "tasks": [{"task_key": "a", "spark_python_task": {"python_file": "/Workspace/a.py"},
                                             "environment_key": "default"}],
            "environments": [{"environment_key": "default", "spec": {"environment_version": "2"}}]}
    result = await h.call("manage_job_runs", {"action": "submit", "spec": spec})
    assert result["status"] == "pending"
    assert result["data"]["run_id"] == 99
    assert isinstance(h.w.jobs.submit.call_args.kwargs["tasks"][0], jobs.SubmitTask)


async def test_cancel_requires_confirmation(h):
    h.w.jobs.get_run.return_value = _run(life="RUNNING", result=None)
    h.w.jobs.get.return_value = _job()
    result = await h.call("manage_job_runs", {"action": "cancel", "run_id": 7})
    assert result["status"] == "confirmation_required"
    h.w.jobs.cancel_run.assert_not_called()
    result = await h.call("manage_job_runs", {"action": "cancel", "run_id": 7, "confirm": True})
    assert result["status"] == "success"
    h.w.jobs.cancel_run.assert_called_once_with(7)


async def test_cancel_all_requires_scope(h):
    message = await h.call_error("manage_job_runs", {"action": "cancel_all", "confirm": True})
    assert "[INVALID_PARAMETER]" in message
    h.w.jobs.cancel_all_runs.assert_not_called()


async def test_delete_run_protected_blocked(h):
    h.w.jobs.get_run.return_value = _run()
    h.w.jobs.get.return_value = _job(name="prod_etl")
    message = await h.call_error("manage_job_runs", {"action": "delete_run", "run_id": 7, "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.jobs.delete_run.assert_not_called()


async def test_repair_validates_and_calls_sdk(h):
    message = await h.call_error("manage_job_runs", {"action": "repair", "run_id": 7})
    assert "[INVALID_PARAMETER]" in message
    h.w.jobs.repair_run.return_value = _wait(jobs.RepairRunResponse(repair_id=5))
    result = await h.call("manage_job_runs",
                          {"action": "repair", "run_id": 7, "spec": {"rerun_all_failed_tasks": True}})
    assert result["status"] == "pending"
    assert result["data"]["repair_id"] == 5
    h.w.jobs.repair_run.assert_called_once_with(run_id=7, rerun_all_failed_tasks=True)


async def test_wait_success(h):
    h.w.jobs.get_run.return_value = _run()
    result = await h.call("manage_job_runs", {"action": "wait", "run_id": 7, "timeout_seconds": 5})
    assert result["status"] == "success"
    assert result["data"]["run"]["result_state"] == "SUCCESS"


async def test_wait_timeout_is_capped(make_harness):
    h = make_harness(toolsets=("jobs",), max_wait_seconds=30)
    h.w.jobs.get_run.return_value = _run()
    result = await h.call("manage_job_runs", {"action": "wait", "run_id": 7, "timeout_seconds": 10_000})
    assert any("reduced to 30s" in w for w in result["warnings"])
