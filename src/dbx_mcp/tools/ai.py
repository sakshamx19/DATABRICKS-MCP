"""AI tools: model serving endpoints, Knowledge Assistants, Supervisor Agents and Genie.

Every SDK method, dataclass and enum used here was verified by introspection against
databricks-sdk (``w.serving_endpoints``, ``w.knowledge_assistants``,
``w.supervisor_agents``, ``w.genie``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Annotated, Any, Literal

from databricks.sdk.common.types.fieldmask import FieldMask
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.dashboards import MessageStatus
from databricks.sdk.service.knowledgeassistants import Example as KaExample
from databricks.sdk.service.knowledgeassistants import (
    KnowledgeAssistant,
    KnowledgeAssistantState,
    KnowledgeSource,
)
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady
from databricks.sdk.service.supervisoragents import Example as MasExample
from databricks.sdk.service.supervisoragents import SupervisorAgent
from databricks.sdk.service.supervisoragents import Tool as MasTool
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import (
    DESTRUCTIVE,
    EXECUTION,
    READ,
    READ_SECURITY,
    WRITE,
    WRITE_SECURITY,
)
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import UnsupportedOperation, ValidationFailed
from dbx_mcp.utils.polling import clamp_wait, poll
from dbx_mcp.utils.redaction import REDACTED
from dbx_mcp.utils.serialization import call_with_spec, parse_sdk_object, pick, to_jsonable, wait_response

TOOLSET = "ai"

# ----------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------

WaitSeconds = Annotated[
    int | None,
    Field(
        description="Optionally wait up to this many seconds for the operation to finish (capped by the "
        "server's max wait). Default: return immediately with status 'pending'.",
        ge=0,
    ),
]


def _wait_budget(requested: int | None) -> int:
    """Clamp a requested wait to the server limits (0 = do not wait)."""
    return clamp_wait(ctx().settings, requested)[0]


def _poll(fetch: Callable[[], Any], done: Callable[[Any], bool], wait_seconds: int) -> tuple[Any, bool]:
    return poll(fetch, done, wait_seconds)


def _token_pages(fetch: Callable[[str | None], Any], items_attr: str) -> Iterator[Any]:
    """Lazily iterate an API that returns ``{<items_attr>: [...], next_page_token}`` pages."""
    token: str | None = None
    while True:
        resp = fetch(token)
        yield from (getattr(resp, items_attr, None) or [])
        token = getattr(resp, "next_page_token", None)
        if not token:
            return


def _last_segment(resource_name: str | None) -> str | None:
    return resource_name.rstrip("/").rsplit("/", 1)[-1] if resource_name else None


def _field_mask(spec: dict[str, Any], update_mask: str | None, allowed: set[str] | None, what: str) -> FieldMask:
    paths = [p.strip() for p in update_mask.split(",") if p.strip()] if update_mask else sorted(spec)
    if not paths:
        raise ValidationFailed(f"Nothing to update: provide {what} fields in 'spec'.")
    if allowed is not None:
        bad = sorted(set(paths) - allowed)
        if bad:
            raise ValidationFailed(
                f"Field(s) {', '.join(bad)} cannot be updated on a {what}. Updatable: {', '.join(sorted(allowed))}"
            )
    return FieldMask(paths)


# ----------------------------------------------------------------------------------------------
# manage_serving_endpoint
# ----------------------------------------------------------------------------------------------

STRIPPED = REDACTED  # same marker as the global response redaction
# External-model provider configs carry API keys / secrets (both secret references and plaintext).
_CREDENTIAL_KEY = re.compile(
    r"(api_?key|api_token|secret|private_key|_plaintext$|^token$|access_key_id|password|credential_json)",
    re.IGNORECASE,
)
_SECRET_REF = re.compile(r"^\{\{\s*secrets/[^}]+\}\}$")


def _strip_credentials(value: Any, parent: str | None = None) -> Any:
    """Remove credentials from serving-endpoint data (external model configs, env vars, auth blocks)."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            secretish = bool(_CREDENTIAL_KEY.search(key)) or (parent in {"api_key_auth", "bearer_token_auth"} and key in {"value", "token"})
            if secretish and item not in (None, "", [], {}):
                out[key] = STRIPPED
            elif key == "environment_vars" and isinstance(item, dict):
                # Values may be plaintext secrets; only {{secrets/scope/key}} references are safe to show.
                out[key] = {k: (v if isinstance(v, str) and _SECRET_REF.match(v.strip()) else STRIPPED) for k, v in item.items()}
            else:
                out[key] = _strip_credentials(item, key)
        return out
    if isinstance(value, list):
        return [_strip_credentials(v, parent) for v in value]
    return value


def _endpoint_data(endpoint: Any) -> dict[str, Any]:
    return _strip_credentials(to_jsonable(endpoint))


def _endpoint_summary(endpoint: Any) -> dict[str, Any]:
    d = _endpoint_data(endpoint)
    entities = []
    for entity in (d.get("config") or {}).get("served_entities") or []:
        item = pick(entity, ("name", "entity_name", "entity_version"))
        if entity.get("external_model"):
            item["external_model"] = pick(entity["external_model"], ("provider", "name", "task"))
        if entity.get("foundation_model"):
            item["foundation_model"] = pick(entity["foundation_model"], ("name", "display_name"))
        entities.append(item)
    out = pick(d, ("name", "id", "task", "creator", "description", "state", "creation_timestamp", "last_updated_timestamp"))
    if entities:
        out["served_entities"] = entities
    if d.get("tags"):
        out["tags"] = {t.get("key"): t.get("value") for t in d["tags"]}
    return out


def _endpoint_tags(endpoint: Any) -> dict[str, str]:
    return {t.key: t.value or "" for t in (getattr(endpoint, "tags", None) or []) if getattr(t, "key", None)}


def _endpoint_settled(endpoint: Any) -> bool:
    state = getattr(endpoint, "state", None)
    update = getattr(state, "config_update", None) if state else None
    return update is not None and update != EndpointStateConfigUpdate.IN_PROGRESS


def _endpoint_outcome(endpoint: Any, verb: str, warnings: list[str]) -> ToolResponse:
    name = getattr(endpoint, "name", None)
    state = getattr(endpoint, "state", None)
    update = getattr(state, "config_update", None) if state else None
    ready = getattr(state, "ready", None) if state else None
    data = _endpoint_data(endpoint)
    poll = f"Poll with manage_serving_endpoint(action='get', name='{name}')."
    if update in (EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED):
        return ok(
            f"Serving endpoint '{name}' {verb}, but its config update ended in state {update.value}.",
            data,
            status="failed",
            warnings=warnings,
            next_steps=[f"Inspect build logs: manage_serving_endpoint(action='get_build_logs', name='{name}', served_model_name=...)."],
        )
    if update == EndpointStateConfigUpdate.NOT_UPDATING and ready == EndpointStateReady.READY:
        return ok(f"Serving endpoint '{name}' {verb} and is READY.", data, warnings=warnings)
    return ok(
        f"Serving endpoint '{name}' {verb}; provisioning is in progress "
        f"(ready={getattr(ready, 'value', ready)}, config_update={getattr(update, 'value', update)}).",
        data,
        status="pending",
        warnings=warnings,
        next_steps=[poll],
    )


def _query_text(resp: dict[str, Any]) -> str | None:
    for choice in resp.get("choices") or []:
        message = choice.get("message") or {}
        if message.get("content"):
            return message["content"]
        if choice.get("text"):
            return choice["text"]
    return None


def _output_limit(max_output_chars: int | None) -> int:
    return max(1_000, min(max_output_chars or 20_000, 200_000))


def _serving_preview(args: dict[str, Any]) -> PlanInfo | None:
    action, name = args.get("action"), args.get("name")
    if not name:
        return None
    w = ctx().w
    if action == "delete":
        endpoint = w.serving_endpoints.get(name=name)
        ctx().safety.check_protected("serving endpoint", name, _endpoint_tags(endpoint), operation="delete")
        return PlanInfo(
            description=f"Permanently delete serving endpoint '{name}'. Clients calling it will start failing.",
            target={"name": name},
            details=_endpoint_summary(endpoint),
            warnings=["Deleting an endpoint cannot be undone; its URL and configuration are lost."],
            reversible=False,
        )
    if action == "update_config":
        endpoint = w.serving_endpoints.get(name=name)
        ctx().safety.check_protected("serving endpoint", name, _endpoint_tags(endpoint), operation="update the config of")
        version = getattr(getattr(endpoint, "config", None), "config_version", None)
        return PlanInfo(
            description=f"Replace the served entities/traffic config of serving endpoint '{name}' "
            f"(current config_version={version}). The endpoint keeps serving the old config until the new one is ready.",
            target={"name": name},
            details={"current": _endpoint_summary(endpoint), "new_config": args.get("spec") or {}},
            reversible=True,
        )
    if action == "query":
        return PlanInfo(
            description=f"Send an inference request to serving endpoint '{name}' (may incur model/compute cost).",
            target={"name": name},
            details={"request": args.get("request") or {}},
            reversible=None,
        )
    return None


@tool(
    toolset=TOOLSET,
    title="Model serving endpoints",
    safety={
        "list": READ,
        "get": READ,
        "get_build_logs": READ,
        "get_logs": READ,
        "create": WRITE,
        "update_config": WRITE,
        "update_ai_gateway": WRITE,
        "delete": DESTRUCTIVE,
        "query": EXECUTION,
    },
    preview=_serving_preview,
)
def manage_serving_endpoint(
    action: Annotated[
        Literal["list", "get", "create", "update_config", "update_ai_gateway", "delete", "query", "get_build_logs", "get_logs"],
        Field(description="Operation to perform."),
    ],
    name: Annotated[str | None, Field(description="Serving endpoint name (all actions except list).")] = None,
    spec: Spec = None,
    request: Annotated[
        dict[str, Any] | None,
        Field(
            description="query: request body. Chat: {'messages': [{'role': 'user', 'content': '...'}], 'max_tokens': 256}; "
            "completions: {'prompt': '...'}; embeddings: {'input': ['...']}; custom models: "
            "{'dataframe_records': [...]} / {'dataframe_split': {...}} / {'instances': [...]} / {'inputs': ...}. "
            "Streaming is not supported."
        ),
    ] = None,
    served_model_name: Annotated[str | None, Field(description="Served model/entity name for get_build_logs/get_logs.")] = None,
    wait_seconds: WaitSeconds = None,
    max_output_chars: Annotated[
        int | None, Field(description="Cap on returned query/log text size (default 20000, max 200000).", ge=1)
    ] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage and query Databricks Model Serving endpoints.

    Actions:
    - list / get: endpoints with state and served entities. Credentials of external-model
      providers (API keys, secrets, tokens, plaintext env vars) are always stripped.
    - create: spec = create body (config, ai_gateway, tags, route_optimized, budget_policy_id,
      description, email_notifications, rate_limits, ...); name is a dedicated parameter.
    - update_config: spec = {served_entities, traffic_config, auto_capture_config, served_models}.
    - update_ai_gateway: spec = {guardrails, inference_table_config, rate_limits, usage_tracking_config, fallback_config}.
    - delete: permanently delete (requires confirm).
    - query: invoke the endpoint with `request` (chat messages / prompt / embeddings input / dataframe).
    - get_build_logs / get_logs: build or server logs for `served_model_name` (tail, size-capped).
    create/update_config are long-running: they return status 'pending' unless `wait_seconds` is set.
    """
    c = ctx()
    w = c.w
    se = w.serving_endpoints

    if action == "list":
        return paged_response("serving endpoint(s)", se.list(), page_size, page_token, _endpoint_summary)

    require(name, "name", action)

    if action == "get":
        endpoint = se.get(name=name)
        return ok(f"Serving endpoint '{name}'.", _endpoint_data(endpoint))

    if action == "create":
        waiter = call_with_spec(se.create, spec, fixed={"name": name})
        endpoint = wait_response(waiter)
        warnings = [c.manifest.safe_track(
            resource_type="serving_endpoint",
            resource_id=name,
            name=name,
            created_by_tool="manage_serving_endpoint",
            workspace_host=c.host,
        )]
        budget = _wait_budget(wait_seconds)
        if budget:
            endpoint, _ = _poll(lambda: se.get(name=name), _endpoint_settled, budget)
        return _endpoint_outcome(endpoint, "was created", [w_ for w_ in warnings if w_])

    if action == "update_config":
        current = se.get(name=name)
        c.safety.check_protected("serving endpoint", name, _endpoint_tags(current), operation="update the config of")
        waiter = call_with_spec(se.update_config, spec, fixed={"name": name})
        endpoint = wait_response(waiter)
        budget = _wait_budget(wait_seconds)
        if budget:
            endpoint, _ = _poll(lambda: se.get(name=name), _endpoint_settled, budget)
        return _endpoint_outcome(endpoint, "config update was submitted", [])

    if action == "update_ai_gateway":
        result = call_with_spec(se.put_ai_gateway, spec, fixed={"name": name})
        return ok(f"AI Gateway configuration of serving endpoint '{name}' updated.", _strip_credentials(to_jsonable(result)))

    if action == "delete":
        endpoint = se.get(name=name)
        c.safety.check_protected("serving endpoint", name, _endpoint_tags(endpoint), operation="delete")
        se.delete(name=name)
        c.manifest.safe_untrack("serving_endpoint", name)
        return ok(f"Serving endpoint '{name}' deleted.", {"name": name, "deleted": True})

    if action == "query":
        require(request, "request", action)
        result = call_with_spec(se.query, request, fixed={"name": name}, exclude=("stream",))
        data = to_jsonable(result)
        text = _query_text(data) if isinstance(data, dict) else None
        limit = _output_limit(max_output_chars)
        serialized = json.dumps(data, default=str)
        warnings: list[str] = []
        if len(serialized) > limit:
            warnings.append(f"Response was {len(serialized)} characters; truncated to {limit}. Raise max_output_chars to see more.")
            data = {
                "truncated": True,
                "original_size_chars": len(serialized),
                "text": text[:limit] if text else None,
                "response_preview": serialized[:limit],
            }
        else:
            data = {"truncated": False, "text": text, "response": data}
        summary = f"Queried serving endpoint '{name}'."
        if text:
            summary += f" Model output (model-generated): {text[:200]}{'...' if len(text) > 200 else ''}"
        return ok(summary, data, warnings=warnings)

    if action in ("get_build_logs", "get_logs"):
        require(served_model_name, "served_model_name", action)
        fetch = se.build_logs if action == "get_build_logs" else se.logs
        logs = fetch(name=name, served_model_name=served_model_name).logs or ""
        limit = _output_limit(max_output_chars)
        truncated = len(logs) > limit
        kind = "Build" if action == "get_build_logs" else "Server"
        return ok(
            f"{kind} logs for '{served_model_name}' on endpoint '{name}' ({len(logs)} chars"
            f"{', showing the last ' + str(limit) if truncated else ''}).",
            {"logs": logs[-limit:] if truncated else logs, "truncated": truncated, "total_chars": len(logs)},
        )

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover - guarded by Literal


# ----------------------------------------------------------------------------------------------
# manage_ka - Knowledge Assistants
# ----------------------------------------------------------------------------------------------

_KA_PREFIX = "knowledge-assistants/"
_KA_UPDATABLE = {"display_name", "description", "instructions"}
_KS_UPDATABLE = {"display_name", "description"}
_EXAMPLE_UPDATABLE = {"question", "guidelines"}
_KS_TYPES = {"files": "files", "index": "index", "file_table": "file_table"}


def _ka_name(value: str | None) -> str:
    value = require(value, "knowledge_assistant_id")
    return value if value.startswith(_KA_PREFIX) else f"{_KA_PREFIX}{value}"


def _ka_child(parent: str, collection: str, child: str | None, param: str) -> str:
    child = require(child, param)
    return child if child.startswith(_KA_PREFIX) else f"{parent}/{collection}/{child}"


def _ka_summary(ka: Any) -> dict[str, Any]:
    d = to_jsonable(ka)
    out = pick(d, ("name", "display_name", "description", "state", "endpoint_name", "creator", "create_time"))
    out["knowledge_assistant_id"] = _last_segment(d.get("name")) or d.get("id")
    return out


def _source_summary(src: Any) -> dict[str, Any]:
    d = to_jsonable(src)
    return pick(d, ("name", "display_name", "description", "source_type", "state", "files", "index", "file_table", "knowledge_cutoff_time"))


def _example_summary(example: Any) -> dict[str, Any]:
    return pick(to_jsonable(example), ("name", "example_id", "question", "guidelines"))


def _validate_source(spec: dict[str, Any]) -> None:
    for key in ("display_name", "description", "source_type"):
        require(spec.get(key), f"spec.{key}", "add_source")
    source_type = spec["source_type"]
    if source_type not in _KS_TYPES:
        raise ValidationFailed(f"spec.source_type must be one of {', '.join(_KS_TYPES)}, got {source_type!r}")
    if not spec.get(_KS_TYPES[source_type]):
        raise ValidationFailed(f"source_type '{source_type}' requires spec.{_KS_TYPES[source_type]}")
    path = (spec.get("files") or {}).get("path") if isinstance(spec.get("files"), dict) else None
    if path:
        ctx().safety.check_volume_path(path.rstrip("/"))


def _ka_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in {"delete", "delete_source", "delete_example", "update_permissions", "sync_sources"} or not args.get("knowledge_assistant_id"):
        return None
    w = ctx().w
    parent = _ka_name(args.get("knowledge_assistant_id"))
    ka = w.knowledge_assistants.get_knowledge_assistant(name=parent)
    if action == "delete":
        ctx().safety.check_protected("knowledge assistant", ka.display_name, operation="delete")
        return PlanInfo(
            description=f"Permanently delete Knowledge Assistant '{ka.display_name}' ({parent}), its knowledge "
            "sources configuration, examples and its agent endpoint.",
            target={"name": parent},
            details=_ka_summary(ka),
            warnings=["Applications querying this assistant's endpoint will stop working."],
            reversible=False,
        )
    if action == "delete_source":
        name = _ka_child(parent, "knowledge-sources", args.get("source_id"), "source_id")
        src = w.knowledge_assistants.get_knowledge_source(name=name)
        ctx().safety.check_protected("knowledge assistant", ka.display_name, operation="remove a knowledge source from")
        return PlanInfo(
            description=f"Remove knowledge source '{src.display_name}' from Knowledge Assistant '{ka.display_name}'. "
            "The underlying documents/tables are not deleted.",
            target={"name": name},
            details=_source_summary(src),
            reversible=False,
        )
    if action == "delete_example":
        name = _ka_child(parent, "examples", args.get("example_id"), "example_id")
        return PlanInfo(
            description=f"Delete example {name} from Knowledge Assistant '{ka.display_name}'.",
            target={"name": name},
            reversible=False,
        )
    if action == "sync_sources":
        return PlanInfo(
            description=f"Re-ingest all non-index knowledge sources of Knowledge Assistant '{ka.display_name}' (uses compute).",
            target={"name": parent},
            reversible=None,
        )
    return PlanInfo(
        description=f"Update permissions on Knowledge Assistant '{ka.display_name}'.",
        target={"knowledge_assistant_id": _last_segment(parent)},
        details={"access_control_list": (args.get("spec") or {}).get("access_control_list")},
        reversible=True,
    )


@tool(
    toolset=TOOLSET,
    title="Knowledge Assistants",
    safety={
        "list": READ,
        "get": READ,
        "create": WRITE,
        "update": WRITE,
        "delete": DESTRUCTIVE,
        "list_sources": READ,
        "get_source": READ,
        "add_source": WRITE,
        "update_source": WRITE,
        "delete_source": DESTRUCTIVE,
        "sync_sources": WRITE | EXECUTION,
        "list_examples": READ,
        "get_example": READ,
        "add_example": WRITE,
        "update_example": WRITE,
        "delete_example": DESTRUCTIVE,
        "get_permissions": READ_SECURITY,
        "update_permissions": WRITE_SECURITY,
    },
    preview=_ka_preview,
)
def manage_ka(
    action: Annotated[
        Literal[
            "list", "get", "create", "update", "delete",
            "list_sources", "get_source", "add_source", "update_source", "delete_source", "sync_sources",
            "list_examples", "get_example", "add_example", "update_example", "delete_example",
            "get_permissions", "update_permissions",
        ],
        Field(description="Operation to perform."),
    ],
    knowledge_assistant_id: Annotated[
        str | None, Field(description="Knowledge Assistant id or resource name 'knowledge-assistants/{id}'.")
    ] = None,
    source_id: Annotated[str | None, Field(description="Knowledge source id (or full resource name).")] = None,
    example_id: Annotated[str | None, Field(description="Example id (or full resource name).")] = None,
    spec: Spec = None,
    update_mask: Annotated[
        str | None, Field(description="Comma-separated fields to update; defaults to the keys present in spec.")
    ] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Knowledge Assistants (Agent Bricks document Q&A agents over UC volumes, tables or vector indexes).

    Actions:
    - list / get / delete; create (spec: display_name, description, instructions);
      update (spec fields among display_name, description, instructions).
    - list_sources / get_source / delete_source; add_source (spec: display_name, description,
      source_type 'files'|'index'|'file_table' plus files={path:'/Volumes/...'} or
      index={index_name,text_col,doc_uri_col} or file_table={table_name,file_col});
      update_source (display_name, description); sync_sources re-ingests non-index sources.
    - list_examples / get_example / add_example (spec: question, guidelines) / update_example / delete_example.
    - get_permissions / update_permissions (spec: access_control_list).
    Query an assistant through its serving endpoint (manage_serving_endpoint action=query).
    """
    c = ctx()
    api = c.w.knowledge_assistants

    if action == "list":
        return paged_response("knowledge assistant(s)", api.list_knowledge_assistants(), page_size, page_token, _ka_summary)

    if action == "create":
        body = dict(require(spec, "spec", action))
        for key in ("display_name", "description"):
            require(body.get(key), f"spec.{key}", action)
        ka = api.create_knowledge_assistant(knowledge_assistant=parse_sdk_object(KnowledgeAssistant, body))
        ka_id = _last_segment(ka.name) or ka.id
        warn = c.manifest.safe_track(
            resource_type="knowledge_assistant",
            resource_id=ka_id or ka.display_name,
            name=ka.display_name,
            created_by_tool="manage_ka",
            workspace_host=c.host,
            metadata={"resource_name": ka.name, "endpoint_name": ka.endpoint_name},
        )
        creating = ka.state in (None, KnowledgeAssistantState.CREATING)
        return ok(
            f"Knowledge Assistant '{ka.display_name}' created (id={ka_id}, state={getattr(ka.state, 'value', ka.state)}).",
            to_jsonable(ka),
            status="pending" if creating else "success",
            warnings=[warn] if warn else None,
            next_steps=[
                f"Add documents: manage_ka(action='add_source', knowledge_assistant_id='{ka_id}', spec={{...}}).",
                f"Check readiness: manage_ka(action='get', knowledge_assistant_id='{ka_id}').",
            ],
        )

    name = _ka_name(knowledge_assistant_id)
    ka_id = _last_segment(name)

    if action == "get":
        ka = api.get_knowledge_assistant(name=name)
        return ok(f"Knowledge Assistant '{ka.display_name}' (state={getattr(ka.state, 'value', ka.state)}).", ka)

    if action == "update":
        body = dict(require(spec, "spec", action))
        mask = _field_mask(body, update_mask, _KA_UPDATABLE, "Knowledge Assistant")
        ka = api.update_knowledge_assistant(
            name=name, knowledge_assistant=parse_sdk_object(KnowledgeAssistant, body), update_mask=mask
        )
        return ok(f"Knowledge Assistant '{ka.display_name}' updated ({mask.ToJsonString()}).", ka)

    if action == "delete":
        ka = api.get_knowledge_assistant(name=name)
        c.safety.check_protected("knowledge assistant", ka.display_name, operation="delete")
        api.delete_knowledge_assistant(name=name)
        c.manifest.safe_untrack("knowledge_assistant", ka_id)
        return ok(f"Knowledge Assistant '{ka.display_name}' deleted.", {"name": name, "deleted": True})

    # --- knowledge sources -----------------------------------------------------------------
    if action == "list_sources":
        return paged_response(
            "knowledge source(s)", api.list_knowledge_sources(parent=name), page_size, page_token, _source_summary
        )
    if action == "get_source":
        return ok("Knowledge source.", api.get_knowledge_source(name=_ka_child(name, "knowledge-sources", source_id, "source_id")))
    if action == "add_source":
        body = dict(require(spec, "spec", action))
        _validate_source(body)
        src = api.create_knowledge_source(parent=name, knowledge_source=parse_sdk_object(KnowledgeSource, body))
        return ok(
            f"Knowledge source '{src.display_name}' added to {name} (state={getattr(src.state, 'value', src.state)}).",
            src,
            next_steps=[
                f"Ingestion runs asynchronously; check with manage_ka(action='list_sources', knowledge_assistant_id='{ka_id}'). "
                f"Re-ingest changed files with action='sync_sources'."
            ],
        )
    if action == "update_source":
        body = dict(require(spec, "spec", action))
        mask = _field_mask(body, update_mask, _KS_UPDATABLE, "knowledge source")
        src = api.update_knowledge_source(
            name=_ka_child(name, "knowledge-sources", source_id, "source_id"),
            knowledge_source=parse_sdk_object(KnowledgeSource, body),
            update_mask=mask,
        )
        return ok(f"Knowledge source '{src.display_name}' updated ({mask.ToJsonString()}).", src)
    if action == "delete_source":
        source_name = _ka_child(name, "knowledge-sources", source_id, "source_id")
        ka = api.get_knowledge_assistant(name=name)
        c.safety.check_protected("knowledge assistant", ka.display_name, operation="remove a knowledge source from")
        api.delete_knowledge_source(name=source_name)
        return ok(f"Knowledge source {source_name} removed.", {"name": source_name, "deleted": True})
    if action == "sync_sources":
        api.sync_knowledge_sources(name=name)
        return ok(
            f"Sync of non-index knowledge sources of {name} started.",
            {"name": name, "sync_requested": True},
            status="pending",
            next_steps=[f"Check source states: manage_ka(action='list_sources', knowledge_assistant_id='{ka_id}')."],
        )

    # --- examples --------------------------------------------------------------------------
    if action == "list_examples":
        return paged_response("example(s)", api.list_examples(parent=name), page_size, page_token, _example_summary)
    if action == "get_example":
        return ok("Example.", api.get_example(name=_ka_child(name, "examples", example_id, "example_id")))
    if action == "add_example":
        body = dict(require(spec, "spec", action))
        require(body.get("question"), "spec.question", action)
        example = api.create_example(parent=name, example=parse_sdk_object(KaExample, body))
        return ok("Example added.", example)
    if action == "update_example":
        body = dict(require(spec, "spec", action))
        mask = _field_mask(body, update_mask, _EXAMPLE_UPDATABLE, "example")
        example = api.update_example(
            name=_ka_child(name, "examples", example_id, "example_id"),
            example=parse_sdk_object(KaExample, body),
            update_mask=mask,
        )
        return ok(f"Example updated ({mask.ToJsonString()}).", example)
    if action == "delete_example":
        example_name = _ka_child(name, "examples", example_id, "example_id")
        api.delete_example(name=example_name)
        return ok(f"Example {example_name} deleted.", {"name": example_name, "deleted": True})

    # --- permissions -----------------------------------------------------------------------
    if action == "get_permissions":
        return ok(f"Permissions of Knowledge Assistant {ka_id}.", api.get_permissions(knowledge_assistant_id=ka_id))
    if action == "update_permissions":
        result = call_with_spec(api.update_permissions, require(spec, "spec", action), fixed={"knowledge_assistant_id": ka_id})
        return ok(f"Permissions of Knowledge Assistant {ka_id} updated.", result)

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover


# ----------------------------------------------------------------------------------------------
# manage_mas - Supervisor Agents (multi-agent supervisors)
# ----------------------------------------------------------------------------------------------

_MAS_PREFIX = "supervisor-agents/"
# Tool types whose spec fields exist on the SDK's Tool dataclass (other documented types such as
# dashboard/table/vector_search_index/web_search have no field in this SDK version).
_MAS_TOOL_TYPES = {"genie_space", "knowledge_assistant", "uc_function", "uc_connection", "app", "volume"}


def _mas_name(value: str | None) -> str:
    value = require(value, "supervisor_agent_id")
    return value if value.startswith(_MAS_PREFIX) else f"{_MAS_PREFIX}{value}"


def _mas_child(parent: str, collection: str, child: str | None, param: str) -> str:
    child = require(child, param)
    return child if child.startswith(_MAS_PREFIX) else f"{parent}/{collection}/{child}"


def _mas_summary(agent: Any) -> dict[str, Any]:
    d = to_jsonable(agent)
    out = pick(d, ("name", "supervisor_agent_id", "display_name", "description", "endpoint_name", "creator", "create_time"))
    out.setdefault("supervisor_agent_id", _last_segment(d.get("name")) or d.get("id"))
    return out


def _mas_tool_summary(t: Any) -> dict[str, Any]:
    d = to_jsonable(t)
    out = pick(d, ("name", "tool_id", "tool_type", "description"))
    kind = d.get("tool_type")
    if kind and d.get(kind):
        out[kind] = d[kind]
    return out


def _validate_mas_tool(spec: dict[str, Any]) -> None:
    tool_type = require(spec.get("tool_type"), "spec.tool_type", "add_tool")
    if tool_type not in _MAS_TOOL_TYPES:
        raise UnsupportedOperation(
            f"tool_type {tool_type!r} cannot be configured through the installed Databricks SDK: its Tool model only "
            f"carries specs for {', '.join(sorted(_MAS_TOOL_TYPES))}.",
            hint="Configure this tool type in the Databricks UI, or upgrade the databricks-sdk.",
        )
    if not spec.get(tool_type):
        raise ValidationFailed(f"tool_type '{tool_type}' requires spec.{tool_type} (e.g. {{'{tool_type}': {{...}}}}).")


def _mas_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in {"delete", "delete_tool", "delete_example", "update_permissions"} or not args.get("supervisor_agent_id"):
        return None
    w = ctx().w
    parent = _mas_name(args.get("supervisor_agent_id"))
    agent = w.supervisor_agents.get_supervisor_agent(name=parent)
    if action == "delete":
        ctx().safety.check_protected("supervisor agent", agent.display_name, operation="delete")
        return PlanInfo(
            description=f"Permanently delete Supervisor Agent '{agent.display_name}' ({parent}), its tools, "
            "examples and serving endpoint. The sub-agents/resources it orchestrates are not deleted.",
            target={"name": parent},
            details=_mas_summary(agent),
            reversible=False,
        )
    if action == "delete_tool":
        name = _mas_child(parent, "tools", args.get("tool_id"), "tool_id")
        t = w.supervisor_agents.get_tool(name=name)
        ctx().safety.check_protected("supervisor agent", agent.display_name, operation="remove a tool from")
        return PlanInfo(
            description=f"Remove tool {name} from Supervisor Agent '{agent.display_name}'. The target resource is not deleted.",
            target={"name": name},
            details=_mas_tool_summary(t),
            reversible=True,
        )
    if action == "delete_example":
        name = _mas_child(parent, "examples", args.get("example_id"), "example_id")
        return PlanInfo(description=f"Delete example {name} from Supervisor Agent '{agent.display_name}'.", target={"name": name}, reversible=False)
    return PlanInfo(
        description=f"Update permissions on Supervisor Agent '{agent.display_name}'.",
        target={"supervisor_agent_id": _last_segment(parent)},
        details={"access_control_list": (args.get("spec") or {}).get("access_control_list")},
        reversible=True,
    )


@tool(
    toolset=TOOLSET,
    title="Supervisor (multi-agent) agents",
    safety={
        "list": READ,
        "get": READ,
        "create": WRITE,
        "update": WRITE,
        "delete": DESTRUCTIVE,
        "list_tools": READ,
        "get_tool": READ,
        "add_tool": WRITE,
        "update_tool": WRITE,
        "delete_tool": DESTRUCTIVE,
        "list_examples": READ,
        "get_example": READ,
        "add_example": WRITE,
        "update_example": WRITE,
        "delete_example": DESTRUCTIVE,
        "get_permissions": READ_SECURITY,
        "update_permissions": WRITE_SECURITY,
    },
    preview=_mas_preview,
)
def manage_mas(
    action: Annotated[
        Literal[
            "list", "get", "create", "update", "delete",
            "list_tools", "get_tool", "add_tool", "update_tool", "delete_tool",
            "list_examples", "get_example", "add_example", "update_example", "delete_example",
            "get_permissions", "update_permissions",
        ],
        Field(description="Operation to perform."),
    ],
    supervisor_agent_id: Annotated[
        str | None, Field(description="Supervisor Agent id or resource name 'supervisor-agents/{id}'.")
    ] = None,
    tool_id: Annotated[str | None, Field(description="Tool id (add_tool: the id to assign; others: id or full resource name).")] = None,
    example_id: Annotated[str | None, Field(description="Example id (or full resource name).")] = None,
    spec: Spec = None,
    update_mask: Annotated[
        str | None, Field(description="Comma-separated fields to update; defaults to the keys present in spec.")
    ] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Supervisor Agents (Agent Bricks multi-agent orchestrators that route to Genie spaces,
    Knowledge Assistants, UC functions, UC connections (MCP), apps and volumes).

    Actions:
    - list / get / delete; create (spec: display_name, description, instructions); update (spec fields).
    - list_tools / get_tool / delete_tool; add_tool (tool_id + spec: tool_type, description and the matching
      block, e.g. {'tool_type':'genie_space','genie_space':{'id':'...'},'description':'...'} or
      {'tool_type':'knowledge_assistant','knowledge_assistant':{'knowledge_assistant_id':'...'}});
      update_tool (only description can change).
    - list_examples / get_example / add_example (spec: question, guidelines) / update_example / delete_example.
    - get_permissions / update_permissions (spec: access_control_list).
    Query a supervisor through its serving endpoint (manage_serving_endpoint action=query).
    """
    c = ctx()
    api = c.w.supervisor_agents

    if action == "list":
        return paged_response("supervisor agent(s)", api.list_supervisor_agents(), page_size, page_token, _mas_summary)

    if action == "create":
        body = dict(require(spec, "spec", action))
        require(body.get("display_name"), "spec.display_name", action)
        agent = api.create_supervisor_agent(supervisor_agent=parse_sdk_object(SupervisorAgent, body))
        agent_id = agent.supervisor_agent_id or _last_segment(agent.name) or agent.id
        warn = c.manifest.safe_track(
            resource_type="supervisor_agent",
            resource_id=agent_id or agent.display_name,
            name=agent.display_name,
            created_by_tool="manage_mas",
            workspace_host=c.host,
            metadata={"resource_name": agent.name, "endpoint_name": agent.endpoint_name},
        )
        return ok(
            f"Supervisor Agent '{agent.display_name}' created (id={agent_id}).",
            agent,
            warnings=[warn] if warn else None,
            next_steps=[f"Attach tools: manage_mas(action='add_tool', supervisor_agent_id='{agent_id}', tool_id='...', spec={{...}})."],
        )

    name = _mas_name(supervisor_agent_id)
    agent_id = _last_segment(name)

    if action == "get":
        agent = api.get_supervisor_agent(name=name)
        return ok(f"Supervisor Agent '{agent.display_name}'.", agent)

    if action == "update":
        body = dict(require(spec, "spec", action))
        mask = _field_mask(body, update_mask, None, "Supervisor Agent")
        agent = api.update_supervisor_agent(
            name=name, supervisor_agent=parse_sdk_object(SupervisorAgent, body), update_mask=mask
        )
        return ok(f"Supervisor Agent '{agent.display_name}' updated ({mask.ToJsonString()}).", agent)

    if action == "delete":
        agent = api.get_supervisor_agent(name=name)
        c.safety.check_protected("supervisor agent", agent.display_name, operation="delete")
        api.delete_supervisor_agent(name=name)
        c.manifest.safe_untrack("supervisor_agent", agent_id)
        return ok(f"Supervisor Agent '{agent.display_name}' deleted.", {"name": name, "deleted": True})

    # --- tools -----------------------------------------------------------------------------
    if action == "list_tools":
        return paged_response("tool(s)", api.list_tools(parent=name), page_size, page_token, _mas_tool_summary)
    if action == "get_tool":
        return ok("Supervisor Agent tool.", api.get_tool(name=_mas_child(name, "tools", tool_id, "tool_id")))
    if action == "add_tool":
        require(tool_id, "tool_id", action)
        body = dict(require(spec, "spec", action))
        _validate_mas_tool(body)
        created = api.create_tool(parent=name, tool=parse_sdk_object(MasTool, body), tool_id=tool_id)
        return ok(f"Tool '{tool_id}' ({created.tool_type}) added to {name}.", created)
    if action == "update_tool":
        body = dict(require(spec, "spec", action))
        mask = _field_mask(body, update_mask, {"description"}, "Supervisor Agent tool")
        updated = api.update_tool(
            name=_mas_child(name, "tools", tool_id, "tool_id"), tool=parse_sdk_object(MasTool, body), update_mask=mask
        )
        return ok("Tool updated.", updated)
    if action == "delete_tool":
        tool_name = _mas_child(name, "tools", tool_id, "tool_id")
        agent = api.get_supervisor_agent(name=name)
        c.safety.check_protected("supervisor agent", agent.display_name, operation="remove a tool from")
        api.delete_tool(name=tool_name)
        return ok(f"Tool {tool_name} removed.", {"name": tool_name, "deleted": True})

    # --- examples --------------------------------------------------------------------------
    if action == "list_examples":
        return paged_response("example(s)", api.list_examples(parent=name), page_size, page_token, _example_summary)
    if action == "get_example":
        return ok("Example.", api.get_example(name=_mas_child(name, "examples", example_id, "example_id")))
    if action == "add_example":
        body = dict(require(spec, "spec", action))
        require(body.get("question"), "spec.question", action)
        example = api.create_example(parent=name, example=parse_sdk_object(MasExample, body))
        return ok("Example added.", example)
    if action == "update_example":
        body = dict(require(spec, "spec", action))
        mask = _field_mask(body, update_mask, _EXAMPLE_UPDATABLE, "example")
        example = api.update_example(
            name=_mas_child(name, "examples", example_id, "example_id"),
            example=parse_sdk_object(MasExample, body),
            update_mask=mask,
        )
        return ok(f"Example updated ({mask.ToJsonString()}).", example)
    if action == "delete_example":
        example_name = _mas_child(name, "examples", example_id, "example_id")
        api.delete_example(name=example_name)
        return ok(f"Example {example_name} deleted.", {"name": example_name, "deleted": True})

    # --- permissions -----------------------------------------------------------------------
    if action == "get_permissions":
        return ok(f"Permissions of Supervisor Agent {agent_id}.", api.get_permissions(supervisor_agent_id=agent_id))
    if action == "update_permissions":
        result = call_with_spec(api.update_permissions, require(spec, "spec", action), fixed={"supervisor_agent_id": agent_id})
        return ok(f"Permissions of Supervisor Agent {agent_id} updated.", result)

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover


# ----------------------------------------------------------------------------------------------
# manage_genie
# ----------------------------------------------------------------------------------------------


def _space_summary(space: Any) -> dict[str, Any]:
    return pick(to_jsonable(space), ("space_id", "title", "description", "warehouse_id", "parent_path", "create_time", "update_time"))


def _genie_spec(spec: dict[str, Any] | None) -> dict[str, Any]:
    body = dict(spec or {})
    if isinstance(body.get("serialized_space"), dict | list):
        body["serialized_space"] = json.dumps(body["serialized_space"])
    return body


def _genie_preview(args: dict[str, Any]) -> PlanInfo | None:
    action, space_id = args.get("action"), args.get("space_id")
    if not space_id:
        return None
    w = ctx().w
    if action == "delete":
        space = w.genie.get_space(space_id=space_id)
        ctx().safety.check_protected("Genie space", space.title, operation="trash")
        return PlanInfo(
            description=f"Move Genie space '{space.title}' ({space_id}) to the trash. Users lose access to it and its conversations.",
            target={"space_id": space_id},
            details=_space_summary(space),
            warnings=["The space can typically be restored from the workspace Trash for a limited time."],
            reversible=True,
        )
    if action == "delete_conversation":
        require(args.get("conversation_id"), "conversation_id", action)
        return PlanInfo(
            description=f"Permanently delete conversation {args['conversation_id']} in Genie space {space_id}.",
            target={"space_id": space_id, "conversation_id": args["conversation_id"]},
            reversible=False,
        )
    if action == "update":
        space = w.genie.get_space(space_id=space_id)
        body = _genie_spec(args.get("spec"))
        warnings = []
        if "serialized_space" in body:
            warnings.append("serialized_space is a FULL replacement of the space's tables, instructions and sample questions.")
        return PlanInfo(
            description=f"Update Genie space '{space.title}' ({space_id}): {', '.join(sorted(body)) or 'no fields'}.",
            target={"space_id": space_id},
            details={"current": _space_summary(space), "changes": sorted(body)},
            warnings=warnings,
            reversible=True,
        )
    return None


@tool(
    toolset=TOOLSET,
    title="Genie spaces",
    safety={
        "list": READ,
        "get": READ,
        "create": WRITE,
        "update": WRITE,
        "delete": DESTRUCTIVE,
        "list_conversations": READ,
        "list_messages": READ,
        "delete_conversation": DESTRUCTIVE,
    },
    preview=_genie_preview,
)
def manage_genie(
    action: Annotated[
        Literal["list", "get", "create", "update", "delete", "list_conversations", "list_messages", "delete_conversation"],
        Field(description="Operation to perform."),
    ],
    space_id: Annotated[str | None, Field(description="Genie space id (all actions except list/create).")] = None,
    conversation_id: Annotated[str | None, Field(description="Conversation id for list_messages/delete_conversation.")] = None,
    spec: Spec = None,
    include_serialized_space: Annotated[
        bool, Field(description="get: include the serialized space definition (requires CAN EDIT).")
    ] = False,
    include_all: Annotated[
        bool, Field(description="list_conversations: include all users' conversations (requires CAN MANAGE).")
    ] = False,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage AI/BI Genie spaces (natural-language-to-SQL over Unity Catalog tables).

    Actions:
    - list / get (include_serialized_space for the full definition).
    - create: spec = {warehouse_id, serialized_space (JSON string or object), title, description, parent_path}.
      Tip: get an existing space with include_serialized_space=true to see the serialized_space format.
    - update: spec with any of title, description, warehouse_id, parent_path, serialized_space (full replacement), etag.
    - delete: move the space to trash (requires confirm).
    - list_conversations / list_messages (conversation_id) / delete_conversation (requires confirm).
    Use ask_genie to ask questions.
    """
    c = ctx()
    genie = c.w.genie

    if action == "list":
        items = _token_pages(lambda token: genie.list_spaces(page_token=token), "spaces")
        return paged_response("Genie space(s)", items, page_size, page_token, _space_summary)

    if action == "create":
        body = _genie_spec(require(spec, "spec", action))
        space = call_with_spec(genie.create_space, body)
        warn = c.manifest.safe_track(
            resource_type="genie_space",
            resource_id=space.space_id,
            name=space.title,
            created_by_tool="manage_genie",
            workspace_host=c.host,
        )
        return ok(
            f"Genie space '{space.title}' created (space_id={space.space_id}).",
            space,
            warnings=[warn] if warn else None,
            next_steps=[f"Ask a question: ask_genie(space_id='{space.space_id}', question='...')."],
        )

    require(space_id, "space_id", action)

    if action == "get":
        space = genie.get_space(space_id=space_id, include_serialized_space=include_serialized_space or None)
        return ok(f"Genie space '{space.title}'.", space)

    if action == "update":
        body = _genie_spec(require(spec, "spec", action))
        if not body:
            raise ValidationFailed("Nothing to update: provide fields in 'spec'.")
        space = call_with_spec(genie.update_space, body, fixed={"space_id": space_id})
        return ok(f"Genie space '{space.title}' updated.", space)

    if action == "delete":
        space = genie.get_space(space_id=space_id)
        c.safety.check_protected("Genie space", space.title, operation="trash")
        genie.trash_space(space_id=space_id)
        c.manifest.safe_untrack("genie_space", space_id)
        return ok(f"Genie space '{space.title}' moved to trash.", {"space_id": space_id, "trashed": True})

    if action == "list_conversations":
        items = _token_pages(
            lambda token: genie.list_conversations(space_id=space_id, include_all=include_all or None, page_token=token),
            "conversations",
        )
        return paged_response("conversation(s)", items, page_size, page_token)

    if action == "list_messages":
        require(conversation_id, "conversation_id", action)
        items = _token_pages(
            lambda token: genie.list_conversation_messages(space_id=space_id, conversation_id=conversation_id, page_token=token),
            "messages",
        )
        return paged_response("message(s)", items, page_size, page_token, _message_summary)

    if action == "delete_conversation":
        require(conversation_id, "conversation_id", action)
        genie.delete_conversation(space_id=space_id, conversation_id=conversation_id)
        return ok(f"Conversation {conversation_id} deleted.", {"space_id": space_id, "conversation_id": conversation_id, "deleted": True})

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover


def _message_summary(msg: Any) -> dict[str, Any]:
    d = to_jsonable(msg)
    out = pick(d, ("message_id", "content", "status", "created_timestamp", "error"))
    sql = [a["query"].get("query") for a in d.get("attachments") or [] if a.get("query")]
    if sql:
        out["generated_sql"] = sql
    return out


# ----------------------------------------------------------------------------------------------
# ask_genie
# ----------------------------------------------------------------------------------------------

_GENIE_TERMINAL = {
    MessageStatus.COMPLETED,
    MessageStatus.FAILED,
    MessageStatus.CANCELLED,
    MessageStatus.QUERY_RESULT_EXPIRED,
}
SQL_RESULT_NOTE = (
    "Rows produced by executing the Genie-generated SQL on the warehouse. The SQL itself is model-generated; "
    "verify the SQL before relying on it."
)


def _statement_rows(statement: Any, max_rows: int) -> dict[str, Any]:
    manifest = getattr(statement, "manifest", None)
    schema = getattr(manifest, "schema", None) if manifest else None
    columns = [col.name for col in (getattr(schema, "columns", None) or [])]
    result = getattr(statement, "result", None)
    data_array = (getattr(result, "data_array", None) or []) if result else []
    rows = [dict(zip(columns, row, strict=False)) if columns else list(row) for row in data_array[:max_rows]]
    total = getattr(manifest, "total_row_count", None) if manifest else None
    status = getattr(statement, "status", None)
    state = getattr(status, "state", None) if status else None
    out: dict[str, Any] = {
        "statement_id": getattr(statement, "statement_id", None),
        "state": getattr(state, "value", state),
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "total_row_count": total,
        "truncated": len(data_array) > max_rows
        or bool(total and total > len(rows))
        or bool(getattr(manifest, "truncated", False)),
    }
    error = getattr(status, "error", None) if status else None
    if error is not None:
        out["error"] = to_jsonable(error)
    return out


def _genie_answer(space_id: str, msg: Any, max_rows: int) -> tuple[dict[str, Any], list[str]]:
    genie = ctx().w.genie
    warnings: list[str] = []
    texts: list[str] = []
    follow_ups: list[str] = []
    suggested: list[str] = []
    queries: list[dict[str, Any]] = []
    for att in msg.attachments or []:
        if att.text is not None and att.text.content:
            purpose = getattr(att.text.purpose, "value", None)
            (follow_ups if purpose == "FOLLOW_UP_QUESTION" else texts).append(att.text.content)
        if att.suggested_questions is not None:
            suggested.extend(att.suggested_questions.questions or [])
        if att.query is not None:
            q = att.query
            entry: dict[str, Any] = {
                "attachment_id": att.attachment_id,
                "title": q.title,
                "description": q.description,
                "generated_sql": q.query,
                "sql_is_model_generated": True,
                "statement_id": q.statement_id,
            }
            if q.parameters:
                entry["parameters"] = to_jsonable(q.parameters)
            if q.query_result_metadata is not None:
                entry["result_metadata"] = to_jsonable(q.query_result_metadata)
            if att.attachment_id and msg.status == MessageStatus.COMPLETED:
                try:
                    res = genie.get_message_attachment_query_result(
                        space_id=space_id,
                        conversation_id=msg.conversation_id,
                        message_id=msg.message_id,
                        attachment_id=att.attachment_id,
                    )
                    if res.statement_response is not None:
                        entry["sql_result"] = _statement_rows(res.statement_response, max_rows)
                        entry["sql_result_note"] = SQL_RESULT_NOTE
                        if entry["sql_result"]["truncated"]:
                            warnings.append(
                                f"Query result for attachment {att.attachment_id} truncated to {max_rows} row(s)."
                            )
                except DatabricksError as exc:
                    entry["sql_result_error"] = f"{type(exc).__name__}: {exc}"
                    warnings.append(f"Could not fetch the query result for attachment {att.attachment_id}: {exc}")
            queries.append(entry)
    data: dict[str, Any] = {
        "answer_is_model_generated": True,
        "answer_source": "model_generated",
        "answer_note": "Genie's text and SQL are generated by an AI model and may be wrong. "
        "Only the sql_result rows come from the database, and only as the output of the generated SQL.",
        "space_id": space_id,
        "conversation_id": msg.conversation_id,
        "message_id": msg.message_id,
        "question": msg.content,
        "genie_status": getattr(msg.status, "value", msg.status),
        "text_response": "\n\n".join(texts) or None,
        "queries": queries,
    }
    if queries:
        data["sql_result_note"] = SQL_RESULT_NOTE
    if follow_ups:
        data["follow_up_questions"] = follow_ups
    if suggested:
        data["suggested_questions"] = suggested
    if msg.error is not None:
        data["error"] = to_jsonable(msg.error)
    return data, warnings


@tool(toolset=TOOLSET, title="Ask Genie", safety=READ | EXECUTION)
def ask_genie(
    space_id: Annotated[str, Field(description="Genie space id.")],
    question: Annotated[
        str | None, Field(description="Natural-language question. Omit when polling an existing message_id.")
    ] = None,
    conversation_id: Annotated[
        str | None, Field(description="Continue this conversation (follow-up question), or poll a message in it.")
    ] = None,
    message_id: Annotated[
        str | None, Field(description="Poll mode: with conversation_id, fetch status/result of a previous question.")
    ] = None,
    wait_seconds: Annotated[
        int, Field(description="How long to wait for Genie to finish (default 60s, capped by server max wait).", ge=0)
    ] = 60,
    max_rows: Annotated[
        int | None, Field(description="Max result rows to return per query (capped by the server SQL row limit).", ge=1)
    ] = None,
) -> ToolResponse:
    """Ask a natural-language question in a Genie space and return Genie's answer.

    Starts a new conversation (or a follow-up when conversation_id is given), waits up to
    wait_seconds, and returns: the model-generated text answer, the generated SQL with its
    description, the rows produced by running that SQL (capped), status and ids. If Genie is
    still working, returns status 'pending' with conversation_id/message_id - call again with
    those ids (and no question) to poll. The answer and SQL are MODEL-GENERATED, not authoritative data.
    """
    c = ctx()
    genie = c.w.genie
    settings = c.settings
    row_cap = min(max_rows or settings.sql_max_rows, settings.sql_max_rows)

    if message_id:
        require(conversation_id, "conversation_id")
        if question:
            raise ValidationFailed("Pass either 'question' (to ask) or 'message_id' (to poll), not both.")
    else:
        require(question, "question")
        if conversation_id:
            sent = wait_response(genie.create_message(space_id=space_id, conversation_id=conversation_id, content=question))
            message_id = sent.message_id
        else:
            started = wait_response(genie.start_conversation(space_id=space_id, content=question))
            conversation_id, message_id = started.conversation_id, started.message_id

    msg, done = _poll(
        lambda: genie.get_message(space_id=space_id, conversation_id=conversation_id, message_id=message_id),
        lambda m: m.status in _GENIE_TERMINAL,
        _wait_budget(wait_seconds),
    )
    ids = {"space_id": space_id, "conversation_id": conversation_id, "message_id": message_id}
    status_value = getattr(msg.status, "value", msg.status)

    if not done:
        return ok(
            f"Genie is still working on the question (status={status_value}).",
            {**ids, "genie_status": status_value, "answer_is_model_generated": True},
            status="pending",
            next_steps=[
                f"Poll: ask_genie(space_id='{space_id}', conversation_id='{conversation_id}', message_id='{message_id}')."
            ],
        )

    data, warnings = _genie_answer(space_id, msg, row_cap)
    if msg.status == MessageStatus.COMPLETED:
        parts = ["Genie answer (MODEL-GENERATED - verify before relying on it)."]
        if data["text_response"]:
            text = data["text_response"]
            parts.append(f"Text: {text[:300]}{'...' if len(text) > 300 else ''}")
        for q in data["queries"]:
            result = q.get("sql_result")
            if result is not None:
                parts.append(
                    f"Generated SQL{' (' + q['title'] + ')' if q.get('title') else ''} returned {result['row_count']} row(s)"
                    f"{' (truncated)' if result['truncated'] else ''}."
                )
            else:
                parts.append("Generated SQL included (no result rows fetched).")
        return ok(
            " ".join(parts),
            data,
            warnings=warnings,
            next_steps=[f"Ask a follow-up: ask_genie(space_id='{space_id}', conversation_id='{conversation_id}', question='...')."],
        )

    if msg.status == MessageStatus.QUERY_RESULT_EXPIRED:
        return ok(
            "Genie's answer is available but its query result has expired.",
            data,
            warnings=warnings,
            next_steps=["Ask the question again to regenerate and re-run the query."],
        )

    error = data.get("error") or {}
    return ok(
        f"Genie could not answer (status={status_value}): {error.get('error') or 'no error details'}.",
        data,
        status="failed",
        warnings=warnings,
        next_steps=["Rephrase the question or check that the space's tables and warehouse are accessible."],
    )

