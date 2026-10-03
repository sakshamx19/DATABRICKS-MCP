"""Tool registration framework.

Tools are plain (synchronous) functions decorated with :func:`tool`. The decorator
only records a :class:`ToolSpec`; :func:`register_tools` later wraps each enabled
spec with the cross-cutting behaviour every tool must have, so no individual tool
can forget it:

* safety-policy enforcement (read-only mode, blocked levels)
* ``dry_run`` previews and two-step ``confirm`` for destructive/security changes
* a timeout (the SDK call runs in a worker thread)
* secret redaction of the response
* error normalization into categorized MCP tool errors
* structured logging with a per-call request id
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import time
import typing
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import anyio
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from dbx_mcp.databricks.request_auth import credentials_from_headers, request_credentials_var
from dbx_mcp.models.common import OperationPlan, ToolResponse
from dbx_mcp.safety.levels import SafetyLevel, is_read_action, needs_confirmation
from dbx_mcp.server.context import get_context
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, normalize_exception
from dbx_mcp.utils.logging import get_logger, request_id_var
from dbx_mcp.utils.redaction import redact, redact_text

log = get_logger("tools")

Levels = frozenset[SafetyLevel]
SafetySpec = Levels | Mapping[str, Levels] | Callable[[dict[str, Any]], Levels]


@dataclass
class PlanInfo:
    """What a tool's ``preview`` hook returns to describe a pending change."""

    description: str
    target: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reversible: bool | None = None


PreviewFn = Callable[[dict[str, Any]], PlanInfo | None]


@dataclass
class ToolSpec:
    name: str
    toolset: str
    fn: Callable[..., ToolResponse]
    title: str
    safety: SafetySpec
    possible_levels: Levels
    preview: PreviewFn | None = None
    timeout_seconds: int | None = None
    action_param: str = "action"

    @property
    def description(self) -> str:
        return inspect.cleandoc(self.fn.__doc__ or self.title)

    @functools.cached_property
    def response_model(self) -> type[ToolResponse]:
        """The tool's declared ToolResponse subclass (used for gate responses too)."""
        try:
            hint = typing.get_type_hints(self.fn).get("return")
        except Exception:
            hint = None
        if inspect.isclass(hint) and issubclass(hint, ToolResponse):
            return hint
        return ToolResponse

    def levels_for(self, args: Mapping[str, Any]) -> Levels:
        if isinstance(self.safety, frozenset):
            return self.safety
        if isinstance(self.safety, Mapping):
            action = args.get(self.action_param)
            if action not in self.safety:
                raise DbxToolError(
                    ErrorCategory.INVALID_PARAMETER,
                    f"Unknown action {action!r} for {self.name}. Valid actions: {', '.join(self.safety)}",
                )
            return self.safety[action]
        return self.safety(dict(args))

    def safety_summary(self) -> str:
        def fmt(levels: Levels) -> str:
            return "+".join(sorted(level.value for level in levels))

        if isinstance(self.safety, frozenset):
            text = fmt(self.safety)
        elif isinstance(self.safety, Mapping):
            groups: dict[str, list[str]] = {}
            for action, levels in self.safety.items():
                groups.setdefault(fmt(levels), []).append(action)
            text = "; ".join(f"{', '.join(actions)} = {lv}" for lv, actions in groups.items())
        else:
            text = "depends on input (" + ", ".join(sorted(level.value for level in self.possible_levels)) + ")"
        return text

    def annotations(self) -> ToolAnnotations:
        levels = self.possible_levels
        return ToolAnnotations(
            title=self.title,
            read_only_hint=is_read_action(levels),
            destructive_hint=SafetyLevel.DESTRUCTIVE in levels,
            open_world_hint=True,
        )


_REGISTRY: dict[str, ToolSpec] = {}

# Name of the hidden parameter through which MCPServer injects its request Context into the
# wrapper (used to read HTTP headers in request-auth mode). Never part of a tool's input schema.
CONTEXT_KWARG = "mcp_ctx"


def tool(
    *,
    toolset: str,
    title: str,
    safety: SafetySpec,
    name: str | None = None,
    possible_levels: Levels | None = None,
    preview: PreviewFn | None = None,
    timeout_seconds: int | None = None,
) -> Callable[[Callable[..., ToolResponse]], Callable[..., ToolResponse]]:
    """Declare an MCP tool. See module docstring."""

    def decorator(fn: Callable[..., ToolResponse]) -> Callable[..., ToolResponse]:
        tool_name = name or fn.__name__
        if isinstance(safety, frozenset):
            levels = safety
        elif isinstance(safety, Mapping):
            levels = frozenset().union(*safety.values())
        else:
            if possible_levels is None:
                raise TypeError(f"{tool_name}: callable safety requires possible_levels")
            levels = possible_levels
        spec = ToolSpec(
            name=tool_name,
            toolset=toolset,
            fn=fn,
            title=title,
            safety=safety,
            possible_levels=possible_levels or levels,
            preview=preview,
            timeout_seconds=timeout_seconds,
        )
        _validate_signature(spec)
        if tool_name in _REGISTRY and _REGISTRY[tool_name].fn.__module__ != fn.__module__:
            raise ValueError(f"Duplicate tool name {tool_name!r}")
        _REGISTRY[tool_name] = spec
        return fn

    return decorator


def _all_action_levels(spec: ToolSpec) -> list[Levels]:
    if isinstance(spec.safety, Mapping):
        return list(spec.safety.values())
    return [spec.possible_levels]


def _validate_signature(spec: ToolSpec) -> None:
    """Every tool that can change state must expose dry_run (and confirm where needed)."""
    params = inspect.signature(spec.fn).parameters
    action_levels = _all_action_levels(spec)
    if any(not is_read_action(lv) for lv in action_levels) and "dry_run" not in params:
        raise TypeError(f"Tool {spec.name} has non-read actions and must accept a 'dry_run' parameter")
    if any(needs_confirmation(lv, confirm_execution=True) for lv in action_levels) and "confirm" not in params:
        raise TypeError(f"Tool {spec.name} has destructive/security/execution actions and must accept 'confirm'")
    if isinstance(spec.safety, Mapping) and spec.action_param not in params:
        raise TypeError(f"Tool {spec.name} declares per-action safety but has no '{spec.action_param}' parameter")


def registered_tools() -> dict[str, ToolSpec]:
    return dict(_REGISTRY)


# ----------------------------------------------------------------------------------------------
# Wrapping
# ----------------------------------------------------------------------------------------------

def _generic_plan(spec: ToolSpec, args: dict[str, Any]) -> PlanInfo:
    shown = {k: v for k, v in args.items() if k not in {"confirm", "dry_run"} and v is not None}
    action = args.get(spec.action_param)
    return PlanInfo(
        description=f"{spec.name}" + (f" will perform '{action}'" if action else " will run") + " with the given arguments.",
        target=redact(shown),
    )


def _build_plan(spec: ToolSpec, args: dict[str, Any], levels: Levels) -> OperationPlan:
    info = spec.preview(args) if spec.preview else None
    if info is None:
        info = _generic_plan(spec, args)
    return OperationPlan(
        tool=spec.name,
        action=args.get(spec.action_param),
        safety=sorted(levels, key=lambda lv: lv.value),
        target=redact(info.target),
        description=info.description,
        details=redact(info.details),
        warnings=info.warnings,
        reversible=info.reversible,
    )


def make_handler(spec: ToolSpec) -> Callable[..., Any]:
    fn = spec.fn

    @functools.wraps(fn)
    async def handler(**kwargs: Any) -> ToolResponse:
        mcp_ctx = kwargs.pop(CONTEXT_KWARG, None)
        request_id = uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        creds_token = None
        workspace_host = None
        started = time.monotonic()
        action = kwargs.get(spec.action_param)
        outcome, category = "success", None
        try:
            ctx = get_context()
            if ctx.clients.request_mode:
                # Per-request credentials from HTTP headers; propagates into worker threads.
                creds = credentials_from_headers(_headers(mcp_ctx), ctx.settings.allowed_workspace_hosts)
                creds_token = request_credentials_var.set(creds)
                workspace_host = creds.host
            levels = spec.levels_for(kwargs)
            ctx.safety.check_allowed(spec.name, action, levels)
            sorted_levels = sorted(levels, key=lambda lv: lv.value)

            if not is_read_action(levels):
                wants_dry_run = bool(kwargs.get("dry_run"))
                wants_confirm = ctx.safety.requires_confirmation(levels) and not kwargs.get("confirm")
                if wants_dry_run or wants_confirm:
                    plan = await anyio.to_thread.run_sync(
                        functools.partial(_build_plan, spec, kwargs, levels), abandon_on_cancel=True
                    )
                    outcome = "dry_run" if wants_dry_run else "confirmation_required"
                    response_cls = spec.response_model
                    if wants_dry_run:
                        return response_cls(
                            status="dry_run",
                            tool=spec.name,
                            action=action,
                            safety=sorted_levels,
                            summary=f"DRY RUN - nothing was changed. {plan.description}",
                            plan=plan,
                            warnings=plan.warnings,
                            request_id=request_id,
                        )
                    return response_cls(
                        status="confirmation_required",
                        tool=spec.name,
                        action=action,
                        safety=sorted_levels,
                        summary=(
                            f"CONFIRMATION REQUIRED - nothing was changed. This action is "
                            f"{'/'.join(lv.value for lv in sorted_levels)}. {plan.description}"
                        ),
                        plan=plan,
                        warnings=plan.warnings,
                        next_steps=[
                            "Review the plan with the user. To proceed, call this tool again with the "
                            "same arguments plus confirm=true."
                        ],
                        request_id=request_id,
                    )

            timeout = spec.timeout_seconds or ctx.settings.tool_timeout_seconds
            try:
                with anyio.fail_after(timeout):
                    result = await anyio.to_thread.run_sync(
                        functools.partial(fn, **kwargs), abandon_on_cancel=True
                    )
            except TimeoutError as exc:
                raise DbxToolError(
                    ErrorCategory.TIMEOUT,
                    f"{spec.name} did not finish within {timeout}s.",
                ) from exc

            if not isinstance(result, ToolResponse):  # pragma: no cover - authoring error
                raise TypeError(f"{spec.name} must return ToolResponse, got {type(result).__name__}")
            result.tool = spec.name
            result.action = result.action or action
            if not result.safety:
                result.safety = sorted_levels
            result.request_id = request_id
            allow = getattr(result, "_unredacted_keys", set())
            if isinstance(result.data, BaseModel):
                result.data = type(result.data).model_validate(
                    redact(result.data.model_dump(mode="json"), allow_keys=allow)
                )
            else:
                result.data = redact(result.data, allow_keys=allow)
            result.summary = redact_text(result.summary)
            result.warnings = [redact_text(w) for w in result.warnings]
            result.next_steps = [redact_text(s) for s in result.next_steps]
            outcome = result.status
            return result
        except Exception as exc:
            ctx_debug = False
            with contextlib.suppress(Exception):
                ctx_debug = get_context().settings.debug
            err = normalize_exception(exc, debug=ctx_debug)
            outcome, category = "error", err.category.value
            if err.category == ErrorCategory.INTERNAL:
                log.error("tool crashed", exc_info=exc, extra={"tool": spec.name, "action": action})
            raise err from exc
        finally:
            log.info(
                "tool call",
                extra={
                    "tool": spec.name,
                    "action": action,
                    "toolset": spec.toolset,
                    "outcome": outcome,
                    "error_category": category,
                    "duration_ms": round((time.monotonic() - started) * 1000, 1),
                    "user": _cached_user(),
                    "workspace_host": workspace_host,
                },
            )
            if creds_token is not None:
                request_credentials_var.reset(creds_token)
            request_id_var.reset(token)

    # Ask MCPServer to inject its Context: it detects the parameter through type hints.
    handler.__annotations__ = {**fn.__annotations__, CONTEXT_KWARG: Context}
    return handler


def _headers(mcp_ctx: Any) -> Any:
    if mcp_ctx is None:
        return None
    try:
        return mcp_ctx.headers
    except Exception:  # Context raises outside of a request
        return None


def _cached_user() -> str | None:
    try:
        return get_context().cached_user_name()
    except Exception:
        return None


def register_tools(server: MCPServer, *, toolsets: tuple[str, ...], disabled: frozenset[str]) -> list[ToolSpec]:
    enabled: list[ToolSpec] = []
    for spec in sorted(_REGISTRY.values(), key=lambda s: (s.toolset, s.name)):
        if spec.toolset not in toolsets or spec.name in disabled:
            continue
        description = f"{spec.description}\n\nSafety classification: {spec.safety_summary()}."
        server.add_tool(
            make_handler(spec),
            name=spec.name,
            title=spec.title,
            description=description,
            annotations=spec.annotations(),
            structured_output=True,
        )
        enabled.append(spec)
    return enabled
