"""Lakeflow Spark Declarative Pipelines: manage_pipeline and manage_pipeline_run.

Uses the official ``databricks-sdk`` ``w.pipelines`` service (Pipelines API 2.0).
Pipeline updates are long-running: ``start`` returns the update id immediately
(``status="pending"``); ``wait`` polls for a bounded time. Failed updates surface
the ERROR events recorded in the pipeline event log for that update.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from databricks.sdk.service import pipelines as pl
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
from dbx_mcp.utils.serialization import call_with_spec, to_jsonable

# Seconds between polls when waiting. Module-level so tests can set it to 0.
POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_WAIT_SECONDS = 120
_TERMINAL_UPDATE_STATES = {"COMPLETED", "FAILED", "CANCELED"}
_MAX_EVENT_SCAN = 1000
_MAX_MESSAGE_CHARS = 4000


def _v(value: Any) -> Any:
    return getattr(value, "value", value)


def _truncate(text: str | None, limit: int = _MAX_MESSAGE_CHARS) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _wait_budget(requested: int | None) -> tuple[int, str | None]:
    """Clamp a requested wait to the server limits (max_wait_seconds and the tool timeout)."""
    return clamp_wait(ctx().settings, requested, _DEFAULT_WAIT_SECONDS)


def _pipeline_row(p: pl.PipelineStateInfo) -> dict[str, Any]:
    latest = (p.latest_updates or [None])[0]
    row = {
        "pipeline_id": p.pipeline_id,
        "name": p.name,
        "state": _v(p.state),
        "health": _v(p.health),
        "creator_user_name": p.creator_user_name,
        "run_as_user_name": p.run_as_user_name,
        "cluster_id": p.cluster_id,
        "latest_update": (
            {"update_id": latest.update_id, "state": _v(latest.state), "creation_time": latest.creation_time}
            if latest
            else None
        ),
    }
    return {k: v for k, v in row.items() if v is not None}


def _update_row(u: pl.UpdateInfo) -> dict[str, Any]:
    row = {
        "update_id": u.update_id,
        "state": _v(u.state),
        "cause": _v(u.cause),
        "creation_time": u.creation_time,
        "full_refresh": u.full_refresh,
        "full_refresh_selection": u.full_refresh_selection,
        "refresh_selection": u.refresh_selection,
        "validate_only": u.validate_only,
        "cluster_id": u.cluster_id,
    }
    return {k: v for k, v in row.items() if v not in (None, [])}


def _event_row(e: pl.PipelineEvent) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": e.id,
        "timestamp": e.timestamp,
        "level": _v(e.level),
        "event_type": e.event_type,
        "message": _truncate(e.message),
    }
    if e.origin:
        row["update_id"] = e.origin.update_id
        row["flow_name"] = e.origin.flow_name
        row["dataset_name"] = e.origin.dataset_name
    if e.error:
        row["error"] = {
            "fatal": e.error.fatal,
            "exceptions": [
                {"class_name": ex.class_name, "message": _truncate(ex.message)}
                for ex in (e.error.exceptions or [])[:5]
            ],
        }
    return {k: v for k, v in row.items() if v is not None}


def _update_errors(pipeline_id: str, update_id: str | None, limit: int = 10) -> list[dict[str, Any]]:
    """ERROR events of the pipeline (optionally of one update), newest first. Best effort."""
    try:
        events = ctx().w.pipelines.list_pipeline_events(pipeline_id, filter="level='ERROR'")
        out: list[dict[str, Any]] = []
        for event in itertools.islice(events, _MAX_EVENT_SCAN):
            if update_id and (not event.origin or event.origin.update_id != update_id):
                continue
            out.append(_event_row(event))
            if len(out) >= limit:
                break
        return out
    except Exception as exc:
        return [{"error_events_unavailable": str(exc)[:300]}]


def _get_pipeline(pipeline_id: str) -> tuple[pl.GetPipelineResponse, dict[str, str]]:
    p = ctx().w.pipelines.get(pipeline_id)
    tags = dict((p.spec.tags if p.spec else None) or {})
    return p, tags


def _spec_security(spec: dict[str, Any] | None) -> bool:
    return bool(spec and spec.get("run_as"))


# ----------------------------------------------------------------------------------------------
# manage_pipeline
# ----------------------------------------------------------------------------------------------

_PIPELINE_LEVELS: dict[str, frozenset[SafetyLevel]] = {
    "create": WRITE,
    "get": READ,
    "list": READ,
    "update": WRITE,
    "delete": DESTRUCTIVE,
    "clone": WRITE | EXECUTION,
}


def _pipeline_safety(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action not in _PIPELINE_LEVELS:
        raise ValidationFailed(f"Unknown action {action!r} for manage_pipeline. Valid: {', '.join(_PIPELINE_LEVELS)}")
    levels = _PIPELINE_LEVELS[action]
    if action in {"create", "update", "clone"} and _spec_security(args.get("spec")):
        levels = levels | SECURITY_SENSITIVE
    return levels


def _pipeline_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action == "delete":
        pipeline_id = require(args.get("pipeline_id"), "pipeline_id", action)
        p, tags = _get_pipeline(pipeline_id)
        ctx().safety.check_protected("pipeline", p.name, tags, operation="delete")
        cascade = args.get("cascade")
        warnings = ["Deleting a pipeline cannot be undone."]
        if cascade is not False:
            warnings.append(
                "cascade defaults to true: for Unity Catalog pipelines ALL materialized views, streaming tables "
                "and views defined by the pipeline are deleted with it. Pass cascade=false to keep the datasets."
            )
        return PlanInfo(
            description=f"Delete pipeline {pipeline_id} ({p.name!r}, state {_v(p.state)}).",
            target={"pipeline_id": pipeline_id, "name": p.name},
            details={
                "creator_user_name": p.creator_user_name,
                "run_as_user_name": p.run_as_user_name,
                "catalog": p.spec.catalog if p.spec else None,
                "schema": (p.spec.schema or p.spec.target) if p.spec else None,
                "cascade": cascade if cascade is not None else "server default (true)",
                "force": args.get("force"),
            },
            warnings=warnings,
            reversible=False,
        )
    if action in {"create", "update", "clone"} and _spec_security(args.get("spec")):
        return PlanInfo(
            description=f"manage_pipeline will '{action}' a pipeline and set run_as.",
            target={"pipeline_id": args.get("pipeline_id"), "name": (args.get("spec") or {}).get("name")},
            details={"run_as": (args.get("spec") or {}).get("run_as")},
            warnings=["run_as changes the identity the pipeline's updates execute as."],
            reversible=True,
        )
    return None


@tool(
    toolset="pipelines",
    title="Manage pipelines",
    safety=_pipeline_safety,
    possible_levels=READ | WRITE | DESTRUCTIVE | EXECUTION | SECURITY_SENSITIVE,
    preview=_pipeline_preview,
)
def manage_pipeline(
    action: Annotated[
        Literal["create", "get", "list", "update", "delete", "clone"],
        Field(
            description="create: new pipeline from spec; get: full definition and state; list: pipelines; "
            "update: change settings (merged onto the current spec); delete: delete pipeline; clone: copy a "
            "Hive-metastore pipeline to Unity Catalog (starts an update on the clone)."
        ),
    ],
    pipeline_id: Annotated[str | None, Field(description="Pipeline id (get/update/delete/clone).")] = None,
    spec: Spec = None,
    name_contains: Annotated[
        str | None, Field(description="list: only pipelines whose name contains this text (server-side LIKE).")
    ] = None,
    filter: Annotated[
        str | None,
        Field(description="list: raw server filter, e.g. \"notebook='/Users/me/nb'\" or \"name LIKE '%sales%'\"."),
    ] = None,
    cascade: Annotated[
        bool | None,
        Field(description="delete: false keeps the pipeline's tables/views (server default true deletes them)."),
    ] = None,
    force: Annotated[bool | None, Field(description="delete: proceed even if resource cleanup fails.")] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Create, inspect, change, clone and delete Lakeflow Spark Declarative Pipelines (DLT).

    - create: spec = pipeline settings (name, catalog, schema, libraries [{notebook:{path}} | {file:{path}} |
      {glob:{include}}], root_path, serverless, clusters, configuration, continuous, development, channel,
      edition, photon, notifications, tags, trigger, environment, event_log, run_as, ...).
    - get (pipeline_id), list (name_contains or filter; paginated).
    - update (pipeline_id, spec): the given top-level fields are merged onto the current settings (set a field to
      null to remove it); uses expected_last_modified to avoid overwriting concurrent edits.
    - delete (pipeline_id[, cascade, force]): DESTRUCTIVE, needs confirm. By default tables are deleted too.
    - clone (pipeline_id, spec: catalog, schema/target, clone_mode='MIGRATE_TO_UC', ...): HMS -> UC copy.
    Run/monitor updates with manage_pipeline_run. Specs with run_as are SECURITY_SENSITIVE.
    """
    c = ctx()
    w = c.w

    if action == "list":
        if name_contains and filter:
            raise ValidationFailed("Pass either name_contains or filter, not both (composite filters are not supported)")
        flt = filter
        if name_contains:
            if "'" in name_contains:
                raise ValidationFailed("name_contains must not contain single quotes")
            flt = f"name LIKE '%{name_contains}%'"
        items = w.pipelines.list_pipelines(filter=flt, max_results=100)
        return paged_response("pipeline(s)", items, page_size, page_token, _pipeline_row)

    if action == "get":
        require(pipeline_id, "pipeline_id", action)
        p = w.pipelines.get(pipeline_id)
        warnings = []
        if _v(p.state) == "FAILED" or _v(p.health) == "UNHEALTHY":
            warnings.append("Pipeline is FAILED/UNHEALTHY; inspect errors with manage_pipeline_run action='list_events' level='ERROR'.")
        return ok(f"Pipeline {pipeline_id} ({p.name!r}): {_v(p.state)}.", p, warnings=warnings)

    if action == "create":
        require(spec, "spec", action)
        if not spec.get("name"):
            raise ValidationFailed("spec.name is required to create a pipeline")
        response = call_with_spec(w.pipelines.create, spec)
        new_id = response.pipeline_id
        warning = None
        if new_id:
            warning = c.manifest.safe_track(
                resource_type="pipeline",
                resource_id=str(new_id),
                name=spec.get("name"),
                created_by_tool="manage_pipeline",
                workspace_host=c.host,
            )
        return ok(
            f"Created pipeline {new_id} ({spec.get('name')!r})." if new_id else "Pipeline spec validated (API dry_run).",
            {"pipeline_id": new_id, "name": spec.get("name"), "effective_settings": response.effective_settings},
            warnings=[warning] if warning else None,
            next_steps=[f"Start it with manage_pipeline_run action='start' pipeline_id={new_id}."] if new_id else [],
        )

    if action == "update":
        require(pipeline_id, "pipeline_id", action)
        require(spec, "spec", action)
        current = w.pipelines.get(pipeline_id)
        merged: dict[str, Any] = dict(to_jsonable(current.spec) or {})
        for key, value in spec.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        if "expected_last_modified" not in spec and current.last_modified:
            merged["expected_last_modified"] = current.last_modified
        call_with_spec(w.pipelines.update, merged, fixed={"pipeline_id": pipeline_id})
        return ok(
            f"Updated pipeline {pipeline_id} ({merged.get('name') or current.name!r}).",
            {"pipeline_id": pipeline_id, "fields_changed": sorted(spec)},
            next_steps=[f"Apply the change by starting an update: manage_pipeline_run action='start' pipeline_id={pipeline_id}."],
        )

    if action == "delete":
        require(pipeline_id, "pipeline_id", action)
        p, tags = _get_pipeline(pipeline_id)
        c.safety.check_protected("pipeline", p.name, tags, operation="delete")
        w.pipelines.delete(pipeline_id, cascade=cascade, force=force)
        c.manifest.safe_untrack("pipeline", str(pipeline_id))
        return ok(
            f"Deleted pipeline {pipeline_id} ({p.name!r}).",
            {"pipeline_id": pipeline_id, "name": p.name, "deleted": True, "cascade": cascade},
        )

    # clone
    require(pipeline_id, "pipeline_id", action)
    response = call_with_spec(w.pipelines.clone, spec, fixed={"pipeline_id": pipeline_id})
    new_id = response.pipeline_id
    warning = None
    if new_id:
        warning = c.manifest.safe_track(
            resource_type="pipeline",
            resource_id=str(new_id),
            name=(spec or {}).get("name"),
            created_by_tool="manage_pipeline",
            workspace_host=c.host,
            metadata={"cloned_from": pipeline_id},
        )
    return ok(
        f"Cloned pipeline {pipeline_id} -> {new_id}; an update was started on the clone.",
        {"source_pipeline_id": pipeline_id, "pipeline_id": new_id},
        status="pending",
        warnings=[warning] if warning else None,
        next_steps=[f"Monitor with manage_pipeline_run action='wait' pipeline_id={new_id}."],
    )


# ----------------------------------------------------------------------------------------------
# manage_pipeline_run
# ----------------------------------------------------------------------------------------------

_RUN_LEVELS: dict[str, frozenset[SafetyLevel]] = {
    "start": EXECUTION,
    "stop": DESTRUCTIVE,
    "get_update": READ,
    "list_updates": READ,
    "list_events": READ,
    "wait": READ,
}


def _is_full_refresh(args: dict[str, Any]) -> bool:
    return bool(args.get("full_refresh") or args.get("full_refresh_selection"))


def _pipeline_run_safety(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action not in _RUN_LEVELS:
        raise ValidationFailed(f"Unknown action {action!r} for manage_pipeline_run. Valid: {', '.join(_RUN_LEVELS)}")
    levels = _RUN_LEVELS[action]
    if action == "start" and _is_full_refresh(args) and not args.get("validate_only"):
        levels = levels | DESTRUCTIVE
    return levels


def _pipeline_run_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in {"start", "stop"}:
        return None
    pipeline_id = require(args.get("pipeline_id"), "pipeline_id", action)
    p, tags = _get_pipeline(pipeline_id)
    latest = (p.latest_updates or [None])[0]
    details = {
        "state": _v(p.state),
        "latest_update": {"update_id": latest.update_id, "state": _v(latest.state)} if latest else None,
        "serverless": p.spec.serverless if p.spec else None,
        "continuous": p.spec.continuous if p.spec else None,
    }
    if action == "stop":
        ctx().safety.check_protected("pipeline", p.name, tags, operation="stop")
        return PlanInfo(
            description=f"Stop the active update of pipeline {pipeline_id} ({p.name!r}).",
            target={"pipeline_id": pipeline_id, "name": p.name},
            details=details,
            warnings=["The running update is cancelled; tables may be left partially refreshed until the next update."],
            reversible=False,
        )
    full = _is_full_refresh(args) and not args.get("validate_only")
    if full:
        ctx().safety.check_protected("pipeline", p.name, tags, operation="full-refresh")
    scope = "all tables" if args.get("full_refresh") else f"tables {args.get('full_refresh_selection')}"
    return PlanInfo(
        description=(
            f"Start an update of pipeline {pipeline_id} ({p.name!r})"
            + (" in validate-only mode." if args.get("validate_only") else ".")
            + (f" FULL REFRESH of {scope}: their state is reset and data recomputed from source." if full else "")
        ),
        target={"pipeline_id": pipeline_id, "name": p.name},
        details={
            **details,
            "full_refresh": args.get("full_refresh"),
            "full_refresh_selection": args.get("full_refresh_selection"),
            "refresh_selection": args.get("refresh_selection"),
            "validate_only": args.get("validate_only"),
        },
        warnings=(
            ["Full refresh truncates streaming tables; data no longer available in the source cannot be recovered."]
            if full
            else []
        ),
        reversible=not full,
    )


def _iter_updates(pipeline_id: str) -> Iterator[pl.UpdateInfo]:
    w = ctx().w
    token: str | None = None
    while True:
        response = w.pipelines.list_updates(pipeline_id, max_results=100, page_token=token)
        yield from response.updates or []
        token = response.next_page_token
        if not token:
            return


def _latest_update_id(pipeline_id: str) -> str:
    p = ctx().w.pipelines.get(pipeline_id)
    latest = (p.latest_updates or [None])[0]
    if not latest or not latest.update_id:
        raise ValidationFailed(f"Pipeline {pipeline_id} has no updates yet; start one with action='start'")
    return latest.update_id


def _update_response(pipeline_id: str, update: pl.UpdateInfo, prefix: str, warnings: list[str]) -> ToolResponse:
    row = _update_row(update)
    state = row.get("state")
    prefix = f"{prefix} " if prefix else ""
    if state not in _TERMINAL_UPDATE_STATES:
        return ok(
            f"{prefix}Update {update.update_id} of pipeline {pipeline_id} is {state}.",
            {"pipeline_id": pipeline_id, "update": row},
            status="pending",
            warnings=warnings,
            next_steps=[
                f"Poll with manage_pipeline_run action='wait' pipeline_id={pipeline_id} update_id={update.update_id}.",
                "Stop it with action='stop' if needed.",
            ],
        )
    if state == "COMPLETED":
        return ok(
            f"{prefix}Update {update.update_id} of pipeline {pipeline_id} COMPLETED.",
            {"pipeline_id": pipeline_id, "update": row},
            warnings=warnings,
        )
    errors = _update_errors(pipeline_id, update.update_id)
    first = next((e.get("message") for e in errors if e.get("message")), None)
    return ok(
        f"{prefix}Update {update.update_id} of pipeline {pipeline_id} {state}" + (f": {_truncate(first, 300)}" if first else "."),
        {"pipeline_id": pipeline_id, "update": row, "errors": errors},
        status="failed",
        warnings=warnings,
        next_steps=[f"See all events: manage_pipeline_run action='list_events' pipeline_id={pipeline_id} update_id={update.update_id}."],
    )


def _poll_update(pipeline_id: str, update_id: str, timeout_seconds: int) -> pl.UpdateInfo:
    w = ctx().w
    deadline = time.monotonic() + timeout_seconds
    while True:
        update = w.pipelines.get_update(pipeline_id, update_id).update
        if update is None:
            raise ValidationFailed(f"Update {update_id} of pipeline {pipeline_id} was not returned by the API")
        if _v(update.state) in _TERMINAL_UPDATE_STATES or time.monotonic() >= deadline:
            return update
        time.sleep(min(POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))


@tool(
    toolset="pipelines",
    title="Run and monitor pipelines",
    safety=_pipeline_run_safety,
    possible_levels=READ | EXECUTION | DESTRUCTIVE,
    preview=_pipeline_run_preview,
)
def manage_pipeline_run(
    action: Annotated[
        Literal["start", "stop", "get_update", "list_updates", "list_events", "wait"],
        Field(
            description="start: start an update; stop: stop the active update; get_update: one update's state "
            "(+errors if failed); list_updates: update history; list_events: event log (use level='ERROR' for "
            "errors); wait: poll an update until it finishes (bounded)."
        ),
    ],
    pipeline_id: Annotated[str, Field(description="Pipeline id.")],
    update_id: Annotated[
        str | None, Field(description="get_update/wait (defaults to the latest update); list_events: filter.")
    ] = None,
    full_refresh: Annotated[bool | None, Field(description="start: reset ALL tables before running (DESTRUCTIVE).")] = None,
    refresh_selection: Annotated[list[str] | None, Field(description="start: tables to refresh (incremental).")] = None,
    full_refresh_selection: Annotated[
        list[str] | None, Field(description="start: tables to fully refresh (DESTRUCTIVE).")
    ] = None,
    validate_only: Annotated[
        bool | None, Field(description="start: only validate the source code; materialize nothing.")
    ] = None,
    parameters: Annotated[dict[str, str] | None, Field(description="start: key/value pipeline parameters.")] = None,
    level: Annotated[
        Literal["ERROR", "WARN", "INFO", "METRICS"] | None, Field(description="list_events: only events of this level.")
    ] = None,
    filter: Annotated[
        str | None, Field(description="list_events: raw filter, e.g. \"timestamp > '2025-01-01T00:00:00Z'\".")
    ] = None,
    wait: Annotated[bool, Field(description="start: poll until the update finishes (bounded).")] = False,
    timeout_seconds: Annotated[int | None, Field(description="Max seconds to wait (capped by server).", ge=1)] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Run and monitor Spark Declarative Pipeline updates and surface pipeline errors.

    - start (pipeline_id[, full_refresh, refresh_selection, full_refresh_selection, validate_only, parameters,
      wait, timeout_seconds]): EXECUTION; returns update_id with status 'pending'. Full refreshes are also
      DESTRUCTIVE (confirm required) because table state is reset.
    - stop (pipeline_id): stops the active update (DESTRUCTIVE, confirm required).
    - get_update / wait (pipeline_id[, update_id] - default latest): state; failed updates include ERROR events.
    - list_updates (pipeline_id): update history, newest first.
    - list_events (pipeline_id[, level, update_id, filter]): event log, newest first.
    """
    c = ctx()
    w = c.w

    if action == "list_updates":
        return paged_response("update(s)", _iter_updates(pipeline_id), page_size, page_token, _update_row)

    if action == "list_events":
        clauses = [f"({filter})"] if filter else []
        if level:
            clauses.append(f"level='{level}'")
        events: Iterator[pl.PipelineEvent] = w.pipelines.list_pipeline_events(
            pipeline_id, filter=" AND ".join(clauses) or None
        )
        warnings = []
        if update_id:
            events = (
                e
                for e in itertools.islice(events, _MAX_EVENT_SCAN)
                if e.origin is not None and e.origin.update_id == update_id
            )
            warnings.append(f"Filtering by update_id is client-side over the newest {_MAX_EVENT_SCAN} events.")
        return paged_response("event(s)", events, page_size, page_token, _event_row, warnings=warnings)

    if action in {"get_update", "wait"}:
        uid = update_id or _latest_update_id(pipeline_id)
        if action == "wait":
            budget, note = _wait_budget(timeout_seconds)
            return _update_response(pipeline_id, _poll_update(pipeline_id, uid, budget), "", [note] if note else [])
        update = w.pipelines.get_update(pipeline_id, uid).update
        if update is None:
            raise ValidationFailed(f"Update {uid} of pipeline {pipeline_id} was not returned by the API")
        row = _update_row(update)
        data: dict[str, Any] = {"pipeline_id": pipeline_id, "update": row, "details": update}
        if row.get("state") in {"FAILED", "CANCELED"}:
            data["errors"] = _update_errors(pipeline_id, uid)
        return ok(f"Update {uid} of pipeline {pipeline_id}: {row.get('state')}.", data)

    if action == "stop":
        p, tags = _get_pipeline(pipeline_id)
        c.safety.check_protected("pipeline", p.name, tags, operation="stop")
        w.pipelines.stop(pipeline_id)
        return ok(
            f"Stop requested for pipeline {pipeline_id} ({p.name!r}).",
            {"pipeline_id": pipeline_id, "stop_requested": True},
            next_steps=[f"Check with manage_pipeline action='get' pipeline_id={pipeline_id}."],
        )

    # start
    if _is_full_refresh({"full_refresh": full_refresh, "full_refresh_selection": full_refresh_selection}) and not validate_only:
        p, tags = _get_pipeline(pipeline_id)
        c.safety.check_protected("pipeline", p.name, tags, operation="full-refresh")
    response = w.pipelines.start_update(
        pipeline_id,
        full_refresh=full_refresh,
        refresh_selection=refresh_selection,
        full_refresh_selection=full_refresh_selection,
        validate_only=validate_only,
        parameters=parameters,
    )
    uid = response.update_id
    if wait and uid:
        budget, note = _wait_budget(timeout_seconds)
        return _update_response(
            pipeline_id, _poll_update(pipeline_id, uid, budget), "Started update.", [note] if note else []
        )
    return ok(
        f"Started update {uid} of pipeline {pipeline_id}"
        + (" (validate only)." if validate_only else "."),
        {"pipeline_id": pipeline_id, "update_id": uid, "full_refresh": bool(full_refresh)},
        status="pending",
        next_steps=[
            f"Poll with manage_pipeline_run action='wait' pipeline_id={pipeline_id} update_id={uid}.",
            f"Errors: manage_pipeline_run action='list_events' pipeline_id={pipeline_id} level='ERROR'.",
        ],
    )
