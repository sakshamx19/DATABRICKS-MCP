"""Lakeflow Jobs: manage_jobs (job definitions) and manage_job_runs (runs, outputs, repair).

All calls go through the official ``databricks-sdk`` ``w.jobs`` service (Jobs API 2.2).
Long-running work (run_now, submit, repair) returns immediately with the run id and
``status="pending"`` unless ``wait=true`` is passed, in which case the run is polled for
a bounded time (never longer than the server's ``max_wait_seconds``/tool timeout).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from databricks.sdk.service import jobs as jobs_svc
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import (
    DESTRUCTIVE,
    EXECUTION,
    READ,
    SECURITY_SENSITIVE,
    WRITE,
    SafetyLevel,
)
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import ValidationFailed
from dbx_mcp.utils.polling import clamp_wait
from dbx_mcp.utils.serialization import call_with_spec, parse_sdk_object, to_jsonable, wait_response

# Seconds between polls when wait=true. Module-level so tests can set it to 0.
POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_WAIT_SECONDS = 120
_MAX_TEXT_CHARS = 20_000
_MAX_TASK_OUTPUTS = 20

_TERMINAL_LIFECYCLE = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}
_SUCCESS_RESULTS = {"SUCCESS", "SUCCESS_WITH_FAILURES"}
_FAILED_RESULTS = {"FAILED", "TIMEDOUT", "CANCELED", "MAXIMUM_CONCURRENT_RUNS_REACHED", "UPSTREAM_FAILED", "UPSTREAM_CANCELED"}

# Fields in a job spec that change who the job runs as / who can access it.
_SECURITY_FIELDS = ("run_as", "access_control_list")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _v(value: Any) -> Any:
    return getattr(value, "value", value)


def _truncate(text: str | None, limit: int = _MAX_TEXT_CHARS) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def wait_budget(requested: int | None, default: int = _DEFAULT_WAIT_SECONDS) -> tuple[int, str | None]:
    """Clamp a requested wait to the server limits (max_wait_seconds and the tool timeout)."""
    return clamp_wait(ctx().settings, requested, default)


def _poll_run(run_id: int, timeout_seconds: int) -> jobs_svc.Run:
    w = ctx().w
    deadline = time.monotonic() + timeout_seconds
    while True:
        run = w.jobs.get_run(run_id)
        if run_is_terminal(run) or time.monotonic() >= deadline:
            return run
        time.sleep(min(POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))


def run_is_terminal(run: jobs_svc.Run | jobs_svc.BaseRun) -> bool:
    if run.state and _v(run.state.life_cycle_state) in _TERMINAL_LIFECYCLE:
        return True
    return bool(run.status and _v(run.status.state) == "TERMINATED")


def _state_dict(state: jobs_svc.RunState | None, status: jobs_svc.RunStatus | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if state:
        out["life_cycle_state"] = _v(state.life_cycle_state)
        out["result_state"] = _v(state.result_state)
        out["state_message"] = state.state_message or None
        if state.user_cancelled_or_timedout:
            out["user_cancelled_or_timedout"] = True
    if status:
        out["status"] = _v(status.state)
        if status.termination_details:
            td = status.termination_details
            out["termination"] = {"code": _v(td.code), "type": _v(td.type), "message": td.message}
        if status.queue_details:
            out["queue"] = {"code": _v(status.queue_details.code), "message": status.queue_details.message}
    return {k: v for k, v in out.items() if v is not None}


def _task_failed(task: jobs_svc.RunTask) -> bool:
    if task.state and _v(task.state.result_state) in _FAILED_RESULTS:
        return True
    if task.state and _v(task.state.life_cycle_state) == "INTERNAL_ERROR":
        return True
    td = task.status.termination_details if task.status else None
    return bool(td and _v(td.type) not in (None, "SUCCESS"))


def _task_summary(task: jobs_svc.RunTask) -> dict[str, Any]:
    out = {
        "task_key": task.task_key,
        "run_id": task.run_id,
        **_state_dict(task.state, task.status),
        "attempt_number": task.attempt_number,
        "start_time": task.start_time,
        "end_time": task.end_time,
        "execution_duration_ms": task.execution_duration,
        "existing_cluster_id": task.existing_cluster_id,
        "job_cluster_key": task.job_cluster_key,
        "environment_key": task.environment_key,
        "run_page_url": task.run_page_url,
    }
    return {k: v for k, v in out.items() if v is not None}


def run_summary(run: jobs_svc.Run | jobs_svc.BaseRun, *, include_tasks: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "run_id": run.run_id,
        "job_id": run.job_id,
        "run_name": run.run_name,
        "run_type": _v(run.run_type),
        **_state_dict(run.state, run.status),
        "terminal": run_is_terminal(run),
        "start_time": run.start_time,
        "end_time": run.end_time or None,
        "run_duration_ms": run.run_duration,
        "trigger": _v(run.trigger),
        "creator_user_name": run.creator_user_name,
        "run_page_url": run.run_page_url,
    }
    if include_tasks and run.tasks:
        out["tasks"] = [_task_summary(t) for t in run.tasks]
    return {k: v for k, v in out.items() if v is not None}


def run_errors(run: jobs_svc.Run, *, fetch_task_output: bool = True) -> list[dict[str, Any]]:
    """Collect error information for a finished run, including per-task error traces."""
    w = ctx().w
    errors: list[dict[str, Any]] = []
    state = _state_dict(run.state, run.status)
    if state.get("result_state") in _FAILED_RESULTS or state.get("life_cycle_state") == "INTERNAL_ERROR" or (
        state.get("termination", {}).get("type") not in (None, "SUCCESS")
    ):
        msg = state.get("state_message") or state.get("termination", {}).get("message")
        if msg:
            errors.append({"scope": "run", "run_id": run.run_id, "message": msg})
    fetched = 0
    for task in run.tasks or []:
        if not _task_failed(task):
            continue
        ts = _state_dict(task.state, task.status)
        entry: dict[str, Any] = {
            "scope": "task",
            "task_key": task.task_key,
            "run_id": task.run_id,
            "result_state": ts.get("result_state"),
            "message": ts.get("state_message") or ts.get("termination", {}).get("message"),
        }
        upstream = ts.get("result_state") in {"UPSTREAM_FAILED", "UPSTREAM_CANCELED"}
        if fetch_task_output and task.run_id and not upstream and fetched < 10:
            fetched += 1
            try:
                output = w.jobs.get_run_output(task.run_id)
                entry["error"] = output.error
                entry["error_trace"] = _truncate(output.error_trace, 8000)
            except Exception as exc:  # best effort: the run itself was fetched fine
                entry["output_unavailable"] = str(exc)[:300]
        errors.append({k: v for k, v in entry.items() if v is not None})
    return errors


def _run_outcome_response(run: jobs_svc.Run, prefix: str, warnings: list[str]) -> ToolResponse:
    """Response for a run we waited on (or just started): pending / success / failed."""
    prefix = f"{prefix} " if prefix else ""
    summary = run_summary(run)
    if not run_is_terminal(run):
        return ok(
            f"{prefix}Run {run.run_id} is {summary.get('life_cycle_state') or summary.get('status')}.",
            {"run": summary},
            status="pending",
            warnings=warnings,
            next_steps=[
                f"Poll with manage_job_runs action='get' run_id={run.run_id} (or action='wait').",
                f"Fetch results when finished with manage_job_runs action='get_output' run_id={run.run_id}.",
            ],
        )
    result = summary.get("result_state")
    if result in _SUCCESS_RESULTS or (result is None and summary.get("termination", {}).get("type") == "SUCCESS"):
        if result == "SUCCESS_WITH_FAILURES":
            warnings.append("Run finished SUCCESS_WITH_FAILURES: some tasks failed but were allowed to.")
        return ok(
            f"{prefix}Run {run.run_id} finished: {result or 'SUCCESS'}.",
            {"run": summary},
            warnings=warnings,
            next_steps=[f"Get task outputs with manage_job_runs action='get_output' run_id={run.run_id}."],
        )
    errors = run_errors(run)
    first = next((e.get("error") or e.get("message") for e in errors if e.get("error") or e.get("message")), None)
    return ok(
        f"{prefix}Run {run.run_id} ended {result or summary.get('life_cycle_state')}"
        + (f": {_truncate(first, 300)}" if first else "."),
        {"run": summary, "errors": errors},
        status="failed",
        warnings=warnings,
        next_steps=[
            f"Inspect with manage_job_runs action='get_output' run_id={run.run_id}.",
            f"Re-run failed tasks with manage_job_runs action='repair' run_id={run.run_id} "
            "spec={'rerun_all_failed_tasks': true}.",
        ],
    )


def _spec_touches_security(spec: dict[str, Any] | None) -> list[str]:
    return [f for f in _SECURITY_FIELDS if spec and spec.get(f) not in (None, [], {})]


def _job_row(job: jobs_svc.BaseJob) -> dict[str, Any]:
    s = job.settings
    row: dict[str, Any] = {
        "job_id": job.job_id,
        "name": s.name if s else None,
        "creator_user_name": job.creator_user_name,
        "created_time": job.created_time,
    }
    if s:
        if s.schedule:
            row["schedule"] = {
                "quartz_cron_expression": s.schedule.quartz_cron_expression,
                "timezone_id": s.schedule.timezone_id,
                "pause_status": _v(s.schedule.pause_status),
            }
        if s.continuous:
            row["continuous"] = to_jsonable(s.continuous)
        if s.trigger:
            row["has_trigger"] = True
        if s.tags:
            row["tags"] = s.tags
        if s.tasks:
            row["task_keys"] = [t.task_key for t in s.tasks]
    return {k: v for k, v in row.items() if v is not None}


def _run_row(run: jobs_svc.BaseRun) -> dict[str, Any]:
    return run_summary(run, include_tasks=False)


def _job_name_and_tags(job_id: int) -> tuple[jobs_svc.Job, str | None, dict[str, str]]:
    job = ctx().w.jobs.get(job_id)
    settings = job.settings
    return job, (settings.name if settings else None), dict((settings.tags if settings else None) or {})


# ----------------------------------------------------------------------------------------------
# manage_jobs
# ----------------------------------------------------------------------------------------------

_JOB_LEVELS: dict[str, frozenset[SafetyLevel]] = {
    "create": WRITE,
    "get": READ,
    "list": READ,
    "update": WRITE,
    "reset": WRITE | DESTRUCTIVE,
    "delete": DESTRUCTIVE,
    "run_now": EXECUTION,
}


def _jobs_safety(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action not in _JOB_LEVELS:
        raise ValidationFailed(f"Unknown action {action!r} for manage_jobs. Valid: {', '.join(_JOB_LEVELS)}")
    levels = _JOB_LEVELS[action]
    if action in {"create", "update", "reset"} and _spec_touches_security(args.get("spec")):
        levels = levels | SECURITY_SENSITIVE
    return levels


def _jobs_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    job_id = args.get("job_id")
    security = _spec_touches_security(args.get("spec"))
    sec_warning = (
        [f"Spec sets {', '.join(security)}: this changes the identity the job runs as and/or who can access it."]
        if security
        else []
    )
    if action == "delete":
        require(job_id, "job_id", action)
        job, name, tags = _job_name_and_tags(job_id)
        ctx().safety.check_protected("job", name, tags, operation="delete")
        return PlanInfo(
            description=f"Permanently delete job {job_id} ({name!r}) and its schedule/triggers. "
            "Active runs are cancelled; run history becomes inaccessible through the job.",
            target={"job_id": job_id, "name": name},
            details={
                "creator_user_name": job.creator_user_name,
                "created_time": job.created_time,
                "task_keys": [t.task_key for t in (job.settings.tasks or [])] if job.settings else [],
                "schedule": to_jsonable(job.settings.schedule) if job.settings else None,
            },
            warnings=["Deleting a job cannot be undone; recreate it from its settings if needed."],
            reversible=False,
        )
    if action == "reset":
        require(job_id, "job_id", action)
        job, name, tags = _job_name_and_tags(job_id)
        ctx().safety.check_protected("job", name, tags, operation="reset (overwrite) the settings of")
        current = to_jsonable(job.settings) or {}
        new = args.get("spec") or {}
        dropped = sorted(set(current) - set(new))
        return PlanInfo(
            description=f"Replace ALL settings of job {job_id} ({name!r}) with the given spec. "
            "Settings not present in the spec are removed.",
            target={"job_id": job_id, "name": name},
            details={"fields_removed": dropped, "fields_set": sorted(new)},
            warnings=[*sec_warning, *([f"These current settings will be removed: {', '.join(dropped)}"] if dropped else [])],
            reversible=False,
        )
    if action in {"create", "update"} and security:
        target = {"job_id": job_id} if job_id else {"name": (args.get("spec") or {}).get("name")}
        return PlanInfo(
            description=f"manage_jobs will '{action}' a job with security-relevant settings.",
            target=target,
            details={"fields": sorted(args.get("spec") or {})},
            warnings=sec_warning,
            reversible=True,
        )
    return None


@tool(
    toolset="jobs",
    title="Manage jobs",
    safety=_jobs_safety,
    possible_levels=READ | WRITE | DESTRUCTIVE | EXECUTION | SECURITY_SENSITIVE,
    preview=_jobs_preview,
)
def manage_jobs(
    action: Annotated[
        Literal["create", "get", "list", "update", "reset", "delete", "run_now"],
        Field(
            description="create: new job from spec; get: full job definition; list: jobs (optional name filter); "
            "update: partial change (spec = fields to set, fields_to_remove); reset: replace ALL settings with spec; "
            "delete: delete the job; run_now: trigger a run (spec = run parameters)."
        ),
    ],
    job_id: Annotated[int | None, Field(description="Job id (get/update/reset/delete/run_now).")] = None,
    spec: Spec = None,
    fields_to_remove: Annotated[
        list[str] | None,
        Field(description="update only: top-level settings to remove, or 'tasks/<task_key>' / 'job_clusters/<key>'."),
    ] = None,
    name: Annotated[str | None, Field(description="list only: exact job name filter (server-side).")] = None,
    wait: Annotated[bool, Field(description="run_now only: poll until the run finishes (bounded).")] = False,
    timeout_seconds: Annotated[
        int | None, Field(description="run_now with wait=true: max seconds to wait (capped by server).", ge=1)
    ] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Create, inspect, change, delete and trigger Databricks Lakeflow Jobs.

    - create: spec = JobSettings fields (name, tasks, job_clusters, environments, schedule, trigger,
      continuous, parameters, email_notifications, webhook_notifications, tags, queue, max_concurrent_runs,
      timeout_seconds, git_source, run_as, access_control_list, ...). Each task needs task_key and one task type
      (notebook_task, spark_python_task, python_wheel_task, sql_task, pipeline_task, run_job_task, ...) plus
      compute (existing_cluster_id, job_cluster_key, new_cluster, or environment_key for serverless).
    - get (job_id), list (name filter, paginated).
    - update (job_id, spec and/or fields_to_remove): partial; top-level fields in spec replace existing ones,
      tasks/job_clusters are merged by key.
    - reset (job_id, spec): full overwrite of all settings (DESTRUCTIVE, needs confirm).
    - delete (job_id): DESTRUCTIVE, needs confirm.
    - run_now (job_id, spec: job_parameters, notebook_params, python_params, only, queue, performance_target,
      idempotency_token, ...): returns the run_id immediately (status 'pending'); wait=true polls (bounded).
    Specs setting run_as/access_control_list are additionally SECURITY_SENSITIVE (confirm required).
    """
    c = ctx()
    w = c.w

    if action == "list":
        items: Iterator[jobs_svc.BaseJob] = w.jobs.list(name=name) if name else w.jobs.list()
        return paged_response("job(s)", items, page_size, page_token, _job_row)

    if action == "get":
        require(job_id, "job_id", action)
        job = w.jobs.get(job_id)
        nm = job.settings.name if job.settings else None
        return ok(f"Job {job_id} ({nm!r}).", job)

    if action == "create":
        require(spec, "spec", action)
        if not spec.get("name"):
            raise ValidationFailed("spec.name is required to create a job")
        response = call_with_spec(w.jobs.create, spec)
        warning = c.manifest.safe_track(
            resource_type="job",
            resource_id=str(response.job_id),
            name=spec.get("name"),
            created_by_tool="manage_jobs",
            workspace_host=c.host,
        )
        return ok(
            f"Created job {response.job_id} ({spec.get('name')!r}).",
            {"job_id": response.job_id, "name": spec.get("name")},
            warnings=[warning] if warning else None,
            next_steps=[f"Trigger it with manage_jobs action='run_now' job_id={response.job_id}."],
        )

    if action == "update":
        require(job_id, "job_id", action)
        if not spec and not fields_to_remove:
            raise ValidationFailed("update needs 'spec' (fields to set) and/or 'fields_to_remove'")
        new_settings = parse_sdk_object(jobs_svc.JobSettings, spec) if spec else None
        w.jobs.update(job_id, new_settings=new_settings, fields_to_remove=fields_to_remove or None)
        return ok(
            f"Updated job {job_id}.",
            {"job_id": job_id, "fields_set": sorted(spec or {}), "fields_removed": fields_to_remove or []},
            next_steps=[f"Review with manage_jobs action='get' job_id={job_id}."],
        )

    if action == "reset":
        require(job_id, "job_id", action)
        require(spec, "spec", action)
        _, nm, tags = _job_name_and_tags(job_id)
        c.safety.check_protected("job", nm, tags, operation="reset (overwrite) the settings of")
        w.jobs.reset(job_id, new_settings=parse_sdk_object(jobs_svc.JobSettings, spec))
        return ok(f"Replaced all settings of job {job_id}.", {"job_id": job_id, "fields_set": sorted(spec)})

    if action == "delete":
        require(job_id, "job_id", action)
        _, nm, tags = _job_name_and_tags(job_id)
        c.safety.check_protected("job", nm, tags, operation="delete")
        w.jobs.delete(job_id)
        c.manifest.safe_untrack("job", str(job_id))
        return ok(f"Deleted job {job_id} ({nm!r}).", {"job_id": job_id, "name": nm, "deleted": True})

    # run_now
    require(job_id, "job_id", action)
    response = wait_response(call_with_spec(w.jobs.run_now, spec, fixed={"job_id": job_id}))
    run_id = response.run_id
    warnings: list[str] = []
    if wait:
        budget, note = wait_budget(timeout_seconds)
        if note:
            warnings.append(note)
        return _run_outcome_response(_poll_run(run_id, budget), f"Triggered job {job_id}.", warnings)
    return ok(
        f"Triggered job {job_id}: run {run_id} started.",
        {"job_id": job_id, "run_id": run_id, "number_in_job": getattr(response, "number_in_job", None)},
        status="pending",
        next_steps=[
            f"Poll with manage_job_runs action='get' run_id={run_id}, or action='wait' run_id={run_id}.",
            f"When finished: manage_job_runs action='get_output' run_id={run_id}.",
        ],
    )


# ----------------------------------------------------------------------------------------------
# manage_job_runs
# ----------------------------------------------------------------------------------------------

_RUN_LEVELS: dict[str, frozenset[SafetyLevel]] = {
    "submit": EXECUTION,
    "list": READ,
    "get": READ,
    "get_output": READ,
    "wait": READ,
    "cancel": DESTRUCTIVE,
    "cancel_all": DESTRUCTIVE,
    "repair": EXECUTION,
    "delete_run": DESTRUCTIVE,
}


def _runs_safety(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action not in _RUN_LEVELS:
        raise ValidationFailed(f"Unknown action {action!r} for manage_job_runs. Valid: {', '.join(_RUN_LEVELS)}")
    levels = _RUN_LEVELS[action]
    if action == "submit" and _spec_touches_security(args.get("spec")):
        levels = levels | SECURITY_SENSITIVE
    return levels


def _run_job_name(run: jobs_svc.Run) -> tuple[str | None, dict[str, str]]:
    """Name/tags used for protected-resource checks: the parent job's, else the run name."""
    if run.job_id:
        try:
            _, name, tags = _job_name_and_tags(run.job_id)
            return name or run.run_name, tags
        except Exception:
            pass
    return run.run_name, {}


def _runs_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    c = ctx()
    if action in {"cancel", "delete_run"}:
        run_id = require(args.get("run_id"), "run_id", action)
        run = c.w.jobs.get_run(run_id)
        name, tags = _run_job_name(run)
        verb = "cancel" if action == "cancel" else "delete"
        c.safety.check_protected("job run", name, tags, operation=verb)
        summary = run_summary(run, include_tasks=False)
        if action == "cancel":
            desc = (
                f"Cancel run {run_id} ({name!r}, currently {summary.get('life_cycle_state')}). "
                "Running tasks are stopped asynchronously; in-flight work is lost."
            )
            warnings = [] if not run_is_terminal(run) else ["The run has already finished; cancel is a no-op."]
            reversible = False
        else:
            desc = f"Delete the record of run {run_id} ({name!r}). Only non-active runs can be deleted."
            warnings = ["The run's history and outputs will no longer be retrievable."]
            if not run_is_terminal(run):
                warnings.append("The run is still active; Databricks rejects deleting active runs.")
            reversible = False
        return PlanInfo(description=desc, target={"run_id": run_id, "job_id": run.job_id}, details=summary,
                        warnings=warnings, reversible=reversible)
    if action == "cancel_all":
        job_id = args.get("job_id")
        all_queued = bool(args.get("all_queued_runs"))
        if not job_id and not all_queued:
            raise ValidationFailed("cancel_all needs job_id, or all_queued_runs=true to cancel queued runs of all jobs")
        active = []
        name = None
        if job_id:
            _, name, tags = _job_name_and_tags(job_id)
            c.safety.check_protected("job", name, tags, operation="cancel all runs of")
            for r in c.w.jobs.list_runs(job_id=job_id, active_only=True):
                active.append(_run_row(r))
                if len(active) >= 50:
                    break
        scope = f"job {job_id} ({name!r})" if job_id else "ALL jobs in the workspace"
        what = "queued runs" if all_queued else "active runs"
        return PlanInfo(
            description=f"Cancel all {what} of {scope}.",
            target={"job_id": job_id, "all_queued_runs": all_queued},
            details={"active_runs_sample": active} if job_id else {},
            warnings=["In-flight work of every affected run is lost."],
            reversible=False,
        )
    if action == "submit" and _spec_touches_security(args.get("spec")):
        return PlanInfo(
            description="Submit a one-time run with run_as/access_control_list settings.",
            target={"run_name": (args.get("spec") or {}).get("run_name")},
            warnings=["The run will execute as/with the specified identity/permissions."],
        )
    return None


def _task_output(run_id: int, task_key: str | None) -> dict[str, Any]:
    out = ctx().w.jobs.get_run_output(run_id)
    data = to_jsonable(out) or {}
    data.pop("metadata", None)  # the run itself; use action='get' for that
    if "logs" in data:
        data["logs"] = _truncate(data["logs"])
    if "error_trace" in data:
        data["error_trace"] = _truncate(data["error_trace"])
    nb = data.get("notebook_output")
    if isinstance(nb, dict) and isinstance(nb.get("result"), str):
        nb["result"] = _truncate(nb["result"], 200_000)
    return {"task_key": task_key, "run_id": run_id, **data}


@tool(
    toolset="jobs",
    title="Manage job runs",
    safety=_runs_safety,
    possible_levels=READ | EXECUTION | DESTRUCTIVE | SECURITY_SENSITIVE,
    preview=_runs_preview,
)
def manage_job_runs(
    action: Annotated[
        Literal["submit", "list", "get", "get_output", "wait", "cancel", "cancel_all", "repair", "delete_run"],
        Field(
            description="submit: one-time run (spec = runs/submit body); list: runs (filters); get: run with task "
            "states and errors; get_output: outputs/errors per task; wait: poll until finished (bounded); cancel: one "
            "run; cancel_all: all active runs of a job; repair: re-run failed/selected tasks; delete_run: delete a "
            "finished run record."
        ),
    ],
    run_id: Annotated[int | None, Field(description="Run id (get/get_output/wait/cancel/repair/delete_run).")] = None,
    job_id: Annotated[int | None, Field(description="list/cancel_all: restrict to this job.")] = None,
    spec: Spec = None,
    task_key: Annotated[str | None, Field(description="get_output: only this task's output.")] = None,
    active_only: Annotated[bool | None, Field(description="list: only active runs.")] = None,
    completed_only: Annotated[bool | None, Field(description="list: only completed runs.")] = None,
    start_time_from: Annotated[int | None, Field(description="list: runs started at/after (epoch ms).")] = None,
    start_time_to: Annotated[int | None, Field(description="list: runs started at/before (epoch ms).")] = None,
    all_queued_runs: Annotated[
        bool | None, Field(description="cancel_all: cancel queued runs (of all jobs when job_id is omitted).")
    ] = None,
    wait: Annotated[bool, Field(description="submit/repair: poll until the run finishes (bounded).")] = False,
    timeout_seconds: Annotated[int | None, Field(description="Max seconds to wait (capped by server).", ge=1)] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Start, monitor, inspect, cancel and repair Databricks job runs.

    - submit: spec = one-time run (run_name, tasks [task_key + task type + compute], environments,
      git_source, timeout_seconds, idempotency_token, ...). Returns run_id with status 'pending'.
    - list (job_id, active_only/completed_only, start_time_from/to), get (run_id: state, per-task states,
      error messages), wait (run_id, timeout_seconds: bounded poll), get_output (run_id[, task_key]:
      notebook exit values, logs, errors/stack traces; multi-task runs are expanded per task).
    - cancel (run_id), cancel_all (job_id or all_queued_runs), delete_run (run_id): DESTRUCTIVE, need confirm.
    - repair (run_id, spec: rerun_all_failed_tasks | rerun_tasks, rerun_dependent_tasks, latest_repair_id,
      job_parameters, ...): EXECUTION.
    """
    c = ctx()
    w = c.w

    if action == "list":
        if active_only and completed_only:
            raise ValidationFailed("active_only and completed_only are mutually exclusive")
        items = w.jobs.list_runs(
            job_id=job_id,
            active_only=active_only,
            completed_only=completed_only,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
        )
        return paged_response("run(s)", items, page_size, page_token, _run_row)

    if action in {"get", "wait"}:
        require(run_id, "run_id", action)
        warnings: list[str] = []
        if action == "wait":
            budget, note = wait_budget(timeout_seconds)
            if note:
                warnings.append(note)
            return _run_outcome_response(_poll_run(run_id, budget), "", warnings)
        run = w.jobs.get_run(run_id)
        summary = run_summary(run)
        data: dict[str, Any] = {"run": summary}
        failed = run_is_terminal(run) and summary.get("result_state") not in _SUCCESS_RESULTS
        if failed:
            data["errors"] = run_errors(run)
        data["details"] = run
        state = summary.get("result_state") or summary.get("life_cycle_state") or summary.get("status")
        return ok(
            f"Run {run_id} ({run.run_name!r}): {state}.",
            data,
            next_steps=[] if summary["terminal"] else [f"Poll again or use action='wait' run_id={run_id}."],
        )

    if action == "get_output":
        require(run_id, "run_id", action)
        run = w.jobs.get_run(run_id)
        tasks = list(run.tasks or [])
        warnings = []
        if task_key:
            tasks = [t for t in tasks if t.task_key == task_key]
            if not tasks:
                raise ValidationFailed(f"Run {run_id} has no task with task_key {task_key!r}")
        outputs: list[dict[str, Any]] = []
        if not tasks:
            outputs.append(_task_output(run_id, None))
        else:
            if len(tasks) > _MAX_TASK_OUTPUTS:
                warnings.append(f"Run has {len(tasks)} tasks; returning the first {_MAX_TASK_OUTPUTS}. Use task_key.")
            for task in tasks[:_MAX_TASK_OUTPUTS]:
                if not task.run_id:
                    continue
                try:
                    outputs.append(_task_output(task.run_id, task.task_key))
                except Exception as exc:
                    outputs.append({"task_key": task.task_key, "run_id": task.run_id, "output_unavailable": str(exc)[:500]})
        if not run_is_terminal(run):
            warnings.append("The run has not finished yet; outputs may be incomplete.")
        return ok(
            f"Output of run {run_id} ({len(outputs)} task output(s)).",
            {"run": run_summary(run, include_tasks=False), "outputs": outputs},
            warnings=warnings,
        )

    if action == "submit":
        require(spec, "spec", action)
        if not spec.get("tasks"):
            raise ValidationFailed("spec.tasks is required for submit")
        response = wait_response(call_with_spec(w.jobs.submit, spec))
        new_run_id = response.run_id
        if wait:
            budget, note = wait_budget(timeout_seconds)
            return _run_outcome_response(_poll_run(new_run_id, budget), "Submitted one-time run.", [note] if note else [])
        return ok(
            f"Submitted one-time run {new_run_id}.",
            {"run_id": new_run_id},
            status="pending",
            next_steps=[
                f"Poll with manage_job_runs action='get' run_id={new_run_id} or action='wait'.",
                f"When finished: manage_job_runs action='get_output' run_id={new_run_id}.",
            ],
        )

    if action == "repair":
        require(run_id, "run_id", action)
        if not spec or not (spec.get("rerun_all_failed_tasks") or spec.get("rerun_tasks")):
            raise ValidationFailed("repair needs spec.rerun_all_failed_tasks=true or spec.rerun_tasks=[task keys]")
        response = wait_response(call_with_spec(w.jobs.repair_run, spec, fixed={"run_id": run_id}))
        repair_id = getattr(response, "repair_id", None)
        if wait:
            budget, note = wait_budget(timeout_seconds)
            return _run_outcome_response(
                _poll_run(run_id, budget), f"Repair {repair_id} started.", [note] if note else []
            )
        return ok(
            f"Started repair {repair_id} of run {run_id}.",
            {"run_id": run_id, "repair_id": repair_id},
            status="pending",
            next_steps=[f"Poll with manage_job_runs action='wait' run_id={run_id}."],
        )

    if action == "cancel":
        require(run_id, "run_id", action)
        run = w.jobs.get_run(run_id)
        name, tags = _run_job_name(run)
        c.safety.check_protected("job run", name, tags, operation="cancel")
        w.jobs.cancel_run(run_id)
        return ok(
            f"Cancellation of run {run_id} requested (asynchronous).",
            {"run_id": run_id, "cancel_requested": True},
            next_steps=[f"Confirm with manage_job_runs action='get' run_id={run_id}."],
        )

    if action == "cancel_all":
        if not job_id and not all_queued_runs:
            raise ValidationFailed("cancel_all needs job_id, or all_queued_runs=true")
        if job_id:
            _, name, tags = _job_name_and_tags(job_id)
            c.safety.check_protected("job", name, tags, operation="cancel all runs of")
        w.jobs.cancel_all_runs(job_id=job_id, all_queued_runs=all_queued_runs)
        scope = f"job {job_id}" if job_id else "all jobs"
        return ok(
            f"Cancellation of all {'queued' if all_queued_runs else 'active'} runs of {scope} requested.",
            {"job_id": job_id, "all_queued_runs": bool(all_queued_runs), "cancel_requested": True},
        )

    # delete_run
    require(run_id, "run_id", action)
    run = w.jobs.get_run(run_id)
    name, tags = _run_job_name(run)
    c.safety.check_protected("job run", name, tags, operation="delete")
    w.jobs.delete_run(run_id)
    return ok(f"Deleted run {run_id}.", {"run_id": run_id, "deleted": True})
