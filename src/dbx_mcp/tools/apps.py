"""Databricks Apps: manage_app."""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Any, Literal

from databricks.sdk.errors import OperationFailed
from databricks.sdk.service.apps import App, AppDeployment
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, EXECUTION, READ, SECURITY_SENSITIVE, WRITE, SafetyLevel
from dbx_mcp.safety.validation import validate_workspace_path
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, UnsupportedOperation, ValidationFailed
from dbx_mcp.utils.polling import wait_cap
from dbx_mcp.utils.serialization import parse_sdk_object, pick, to_jsonable, wait_response

AppAction = Literal[
    "create", "get", "list", "update", "delete", "deploy", "get_deployment", "list_deployments", "start", "stop", "logs"
]

_ACTION_LEVELS: dict[str, frozenset[SafetyLevel]] = {
    "create": WRITE | EXECUTION,  # starts app compute unless no_compute=true
    "get": READ,
    "list": READ,
    "update": WRITE,
    "delete": DESTRUCTIVE,
    "deploy": WRITE | EXECUTION,  # runs the deployed source code
    "get_deployment": READ,
    "list_deployments": READ,
    "start": WRITE | EXECUTION,
    "stop": WRITE,
    "logs": READ,
}

# App fields that grant the app's service principal / user token access to other resources.
_SECURITY_FIELDS = {"resources", "user_api_scopes", "forward_user_access_token"}
_APP_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")

_APP_SUMMARY = (
    "name", "id", "description", "url", "compute_size", "creator", "create_time", "update_time",
    "default_source_code_path", "service_principal_name",
)
_DEPLOYMENT_SUMMARY = ("deployment_id", "source_code_path", "mode", "create_time", "creator", "update_time")

Wait = Annotated[bool, Field(description="Wait (bounded) for the operation to reach a steady state.")]
TimeoutSeconds = Annotated[
    int | None, Field(description="Max seconds to wait when wait=true (capped by DBX_MCP_MAX_WAIT_SECONDS).", ge=1)
]


def _levels(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action not in _ACTION_LEVELS:
        raise DbxToolError(
            ErrorCategory.INVALID_PARAMETER,
            f"Unknown action {action!r} for manage_app. Valid actions: {', '.join(_ACTION_LEVELS)}",
        )
    levels = _ACTION_LEVELS[action]
    if action in {"create", "update"} and _SECURITY_FIELDS & set(args.get("app") or {}):
        levels = levels | SECURITY_SENSITIVE
    if action == "create" and args.get("no_compute"):
        levels = WRITE | (levels & SECURITY_SENSITIVE)
    return levels


def _state(value: Any) -> str | None:
    return getattr(value, "value", value)


def _app_summary(app: App) -> dict[str, Any]:
    data = pick(to_jsonable(app), _APP_SUMMARY)
    data["app_state"] = _state(app.app_status.state) if app.app_status else None
    data["compute_state"] = _state(app.compute_status.state) if app.compute_status else None
    if app.active_deployment:
        data["active_deployment_id"] = app.active_deployment.deployment_id
    if app.pending_deployment:
        data["pending_deployment_id"] = app.pending_deployment.deployment_id
    return data


def _deployment_summary(d: AppDeployment) -> dict[str, Any]:
    data = pick(to_jsonable(d), _DEPLOYMENT_SUMMARY)
    if d.status:
        data["state"] = _state(d.status.state)
        data["message"] = d.status.message
    return data


def _app_name(name: str | None, action: str) -> str:
    require(name, "name", action)
    if not _APP_NAME.match(name):  # type: ignore[arg-type]
        raise ValidationFailed(
            f"Invalid app name {name!r}: app names may contain only lowercase letters, numbers and hyphens"
        )
    return name  # type: ignore[return-value]


def _source_path(path: str | None) -> str | None:
    if path is None:
        return None
    clean = validate_workspace_path(path)
    ctx().safety.check_workspace_path(clean)
    return clean


def _wait_budget(timeout_seconds: int | None) -> dt.timedelta:
    cap = wait_cap(ctx().settings)
    return dt.timedelta(seconds=min(timeout_seconds or cap, cap))


def _finish(waiter: Any, wait: bool, timeout_seconds: int | None, refresh: Any) -> tuple[Any, str, str | None]:
    """Return (object, status, note) after an optional bounded wait on an SDK ``Wait``."""
    initial = wait_response(waiter)
    if not wait:
        return initial, "pending", None
    try:
        return waiter.result(timeout=_wait_budget(timeout_seconds)), "success", None
    except TimeoutError:
        return refresh(), "pending", "Still in progress after the wait budget; poll with get/get_deployment."
    except OperationFailed as exc:
        return refresh(), "failed", str(exc)


def _preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    c = ctx()
    if action == "create":
        name = _app_name(args.get("name"), action)
        spec = dict(args.get("app") or {})
        parse_sdk_object(App, {**spec, "name": name}, "app")  # validate field names/types up front
        warnings = []
        if _SECURITY_FIELDS & set(spec):
            warnings.append(
                "The app is granted access to other resources/scopes "
                f"({', '.join(sorted(_SECURITY_FIELDS & set(spec)))}); review them."
            )
        if not args.get("no_compute"):
            warnings.append("App compute will start after creation and incurs cost until stopped.")
        return PlanInfo(
            description=f"Create Databricks App {args.get('name')!r}.",
            target={"name": args.get("name")},
            details={"app": spec, "no_compute": bool(args.get("no_compute"))},
            warnings=warnings,
            reversible=True,
        )
    if action not in {"delete", "stop", "update", "deploy", "start"}:
        return None
    name = _app_name(args.get("name"), action)
    app = c.w.apps.get(name)
    summary = _app_summary(app)
    target = {"name": name, "url": app.url}
    if action in {"delete", "stop"}:
        c.safety.check_protected("app", name, operation=action)
    if action == "delete":
        return PlanInfo(
            description=f"PERMANENTLY delete Databricks App {name!r} ({app.url or 'no url'}); its URL stops working "
            "and its deployments are removed.",
            target=target,
            details={"app": summary},
            warnings=["The app's source code in the workspace is not deleted, but the app itself cannot be restored."],
            reversible=False,
        )
    if action == "stop":
        return PlanInfo(
            description=f"Stop app {name!r}; it becomes unavailable to users until started again.",
            target=target,
            details={"app": summary},
            reversible=True,
        )
    if action == "update":
        spec = dict(args.get("app") or {})
        warnings = []
        if _SECURITY_FIELDS & set(spec):
            warnings.append(f"Changes access-related fields: {', '.join(sorted(_SECURITY_FIELDS & set(spec)))}.")
        return PlanInfo(
            description=f"Update app {name!r} fields: {', '.join(sorted(spec)) or '(none)'}.",
            target=target,
            details={"app": summary, "changes": spec},
            warnings=warnings,
            reversible=True,
        )
    if action == "deploy":
        return PlanInfo(
            description=f"Deploy app {name!r} from {args.get('source_code_path') or 'its default source'}"
            f" (mode {args.get('mode') or 'default'}); the new code will run with the app's identity.",
            target=target,
            details={"app": summary, "deployment": args.get("deployment") or {}},
            reversible=True,
        )
    return PlanInfo(description=f"Start app {name!r} (incurs compute cost).", target=target,
                    details={"app": summary}, reversible=True)


@tool(
    toolset="apps",
    title="Manage Databricks Apps",
    safety=_levels,
    possible_levels=READ | WRITE | EXECUTION | DESTRUCTIVE | SECURITY_SENSITIVE,
    preview=_preview,
)
def manage_app(
    action: Annotated[
        AppAction,
        Field(
            description="create | get | list | update | delete | deploy | get_deployment | list_deployments | "
            "start | stop | logs"
        ),
    ],
    name: Annotated[str | None, Field(description="App name (lowercase letters, numbers, hyphens).")] = None,
    app: Annotated[
        dict[str, Any] | None,
        Field(
            description="create/update: App fields (REST names), e.g. description, resources, compute_size, "
            "user_api_scopes, budget_policy_id. update changes only the fields given."
        ),
    ] = None,
    no_compute: Annotated[bool, Field(description="create: do not start app compute after creation.")] = False,
    source_code_path: Annotated[
        str | None, Field(description="deploy: workspace folder with the app source, e.g. /Workspace/Users/me/app.")
    ] = None,
    mode: Annotated[
        Literal["SNAPSHOT", "AUTO_SYNC"] | None,
        Field(description="deploy: SNAPSHOT (copy source now) or AUTO_SYNC (keep syncing from source_code_path)."),
    ] = None,
    deployment: Annotated[
        dict[str, Any] | None,
        Field(description="deploy: extra AppDeployment fields (e.g. git_source, command, env_vars)."),
    ] = None,
    deployment_id: Annotated[str | None, Field(description="get_deployment: deployment id.")] = None,
    wait: Wait = False,
    timeout_seconds: TimeoutSeconds = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Databricks Apps. Actions: create (name, app fields, no_compute), get, list, update
    (partial: only the given app fields), delete, deploy (source_code_path, mode SNAPSHOT|AUTO_SYNC,
    extra deployment fields), get_deployment, list_deployments, start, stop. create/deploy/start/stop
    return immediately with status 'pending' unless wait=true (bounded). `logs` is not available via
    the API/SDK. Created apps are tracked in the project manifest."""
    c = ctx()
    if action == "list":
        return paged_response("apps", c.w.apps.list(), page_size, page_token, transform=_app_summary)

    if action == "logs":
        raise UnsupportedOperation(
            "App logs are not exposed by the Databricks SDK/REST API used by this server.",
            hint="View logs in the app's 'Logs' tab in the Databricks UI.",
        )

    name = _app_name(name, action)

    if action == "get":
        return ok(f"App {name}.", c.w.apps.get(name))

    if action == "list_deployments":
        return paged_response(
            f"deployments of app {name}", c.w.apps.list_deployments(name), page_size, page_token,
            transform=_deployment_summary,
        )

    if action == "get_deployment":
        require(deployment_id, "deployment_id", action)
        d = c.w.apps.get_deployment(name, deployment_id)
        return ok(f"Deployment {deployment_id} of app {name}: {_state(d.status.state) if d.status else 'unknown'}.", d)

    if action == "create":
        spec = dict(app or {})
        if "name" in spec and spec["name"] != name:
            raise ValidationFailed("Pass the app name via the 'name' parameter, not inside 'app'")
        spec["name"] = name
        app_obj = parse_sdk_object(App, spec, "app")
        waiter = c.w.apps.create(app_obj, no_compute=no_compute or None)
        if no_compute:
            wait = False
        result, status, note = _finish(waiter, wait, timeout_seconds, lambda: c.w.apps.get(name))
        if no_compute:
            status = "success"
        warning = c.manifest.safe_track(
            resource_type="app",
            resource_id=name,
            name=name,
            created_by_tool="manage_app",
            workspace_host=c.host,
            metadata={"id": getattr(result, "id", None), "url": getattr(result, "url", None)},
        )
        summary = _app_summary(result)
        text = f"Created app {name!r}" + (f" ({summary.get('url')})" if summary.get("url") else "") + (
            "; compute is starting." if status == "pending" and not no_compute else "."
        )
        if note:
            text += f" {note}"
        return ok(
            text, summary, status=status, warnings=[warning] if warning else None,
            next_steps=[f"Deploy code with manage_app action=deploy name={name} source_code_path=..."],
        )

    if action == "update":
        spec = dict(app or {})
        if spec.get("name") == name:
            del spec["name"]
        if "name" in spec:
            raise ValidationFailed("Renaming apps is not supported; 'name' identifies the app")
        if not spec:
            raise ValidationFailed("update requires 'app' with at least one field to change")
        app_obj = parse_sdk_object(App, {**spec, "name": name}, "app")
        waiter = c.w.apps.create_update(name, ",".join(sorted(spec)), app=app_obj)
        result, status, note = _finish(waiter, wait, timeout_seconds, lambda: c.w.apps.get_update(name))
        update_state = _state(result.status.state) if getattr(result, "status", None) else None
        if update_state == "SUCCEEDED":
            status = "success"
        elif update_state == "FAILED":
            status = "failed"
        text = f"Update of app {name!r} ({', '.join(sorted(spec))}): {update_state or status}."
        if note:
            text += f" {note}"
        return ok(text, result, status=status)

    if action == "deploy":
        body = dict(deployment or {})
        for key, value in (("source_code_path", _source_path(source_code_path)), ("mode", mode)):
            if value is not None:
                if key in body:
                    raise ValidationFailed(f"Pass {key} as a dedicated parameter, not inside 'deployment'")
                body[key] = value
        if "source_code_path" in body and source_code_path is None:
            body["source_code_path"] = _source_path(body["source_code_path"])
        app_deployment = parse_sdk_object(AppDeployment, body, "deployment")
        waiter = c.w.apps.deploy(name, app_deployment)
        initial = wait_response(waiter)
        dep_id = getattr(initial, "deployment_id", None)
        result, status, note = _finish(
            waiter, wait, timeout_seconds,
            lambda: c.w.apps.get_deployment(name, dep_id) if dep_id else initial,
        )
        data = _deployment_summary(result)
        if data.get("state") == "SUCCEEDED":
            status = "success"
        elif data.get("state") in {"FAILED", "CANCELLED"}:
            status = "failed"
        text = f"Deployment {data.get('deployment_id')} of app {name!r}: {data.get('state') or status}."
        if note:
            text += f" {note}"
        return ok(text, data, status=status,
                  next_steps=[f"Poll with manage_app action=get_deployment name={name} deployment_id={dep_id}"]
                  if status == "pending" else None)

    if action in {"start", "stop"}:
        if action == "stop":
            c.safety.check_protected("app", name, operation="stop")
        waiter = c.w.apps.start(name) if action == "start" else c.w.apps.stop(name)
        result, status, note = _finish(waiter, wait, timeout_seconds, lambda: c.w.apps.get(name))
        summary = _app_summary(result)
        text = f"{'Starting' if action == 'start' else 'Stopping'} app {name!r}: compute {summary.get('compute_state')}."
        if status == "success":
            text = f"App {name!r} {'started' if action == 'start' else 'stopped'} (compute {summary.get('compute_state')})."
        if note:
            text += f" {note}"
        return ok(text, summary, status=status)

    if action == "delete":
        c.safety.check_protected("app", name, operation="delete")
        deleted = c.w.apps.delete(name)
        c.manifest.safe_untrack("app", name)
        return ok(f"Deleted app {name!r}.", _app_summary(deleted) if deleted else {"name": name})

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover - Literal guards this
