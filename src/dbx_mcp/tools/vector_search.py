"""Vector Search tools: endpoints, indexes, similarity queries and direct-access data.

SDK surface (verified by introspection): ``w.vector_search_endpoints`` (create_endpoint,
get_endpoint, list_endpoints, delete_endpoint, patch_endpoint, update_endpoint_budget_policy,
update_endpoint_custom_tags) and ``w.vector_search_indexes`` (create_index, get_index,
list_indexes, delete_index, sync_index, query_index, query_next_page, scan_index,
upsert_data_vector_index, delete_data_vector_index).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, Literal

from databricks.sdk.service.vectorsearch import (
    CustomTag,
    EndpointStatusState,
    EndpointType,
    PipelineType,
    VectorIndexType,
)
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, EXECUTION, READ, WRITE
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import UnsupportedOperation, ValidationFailed
from dbx_mcp.utils.polling import clamp_wait, poll
from dbx_mcp.utils.serialization import call_with_spec, pick, to_jsonable, wait_response

TOOLSET = "vector_search"

WaitSeconds = Annotated[
    int | None,
    Field(
        description="Optionally wait up to this many seconds for the endpoint to come ONLINE (capped by the "
        "server's max wait). Default: return immediately with status 'pending'.",
        ge=0,
    ),
]


def _wait_budget(requested: int | None) -> int:
    """Clamp a requested wait to the server limits (0 = do not wait)."""
    return clamp_wait(ctx().settings, requested)[0]


def _poll(fetch: Callable[[], Any], done: Callable[[Any], bool], wait_seconds: int) -> tuple[Any, bool]:
    return poll(fetch, done, wait_seconds)


def _result_cap(requested: int | None, default: int = 10) -> int:
    return max(1, min(requested or default, ctx().settings.sql_max_rows))


# ----------------------------------------------------------------------------------------------
# manage_vs_endpoint
# ----------------------------------------------------------------------------------------------

_ENDPOINT_UPDATABLE = {"target_qps", "budget_policy_id", "custom_tags"}


def _endpoint_state(endpoint: Any) -> EndpointStatusState | None:
    status = getattr(endpoint, "endpoint_status", None)
    return getattr(status, "state", None) if status else None


def _endpoint_summary(endpoint: Any) -> dict[str, Any]:
    d = to_jsonable(endpoint)
    out = pick(d, ("name", "id", "endpoint_type", "endpoint_status", "num_indexes", "creator", "budget_policy_id",
                   "effective_budget_policy_id", "scaling_info", "last_updated_timestamp"))
    if d.get("custom_tags"):
        out["custom_tags"] = {t.get("key"): t.get("value") for t in d["custom_tags"]}
    return out


def _endpoint_tags(endpoint: Any) -> dict[str, str]:
    return {t.key: t.value or "" for t in (getattr(endpoint, "custom_tags", None) or []) if getattr(t, "key", None)}


def _custom_tags(value: Any) -> list[CustomTag]:
    if isinstance(value, dict):
        return [CustomTag(key=str(k), value=None if v is None else str(v)) for k, v in value.items()]
    if isinstance(value, list):
        tags = []
        for i, item in enumerate(value):
            if not isinstance(item, dict) or not item.get("key") or set(item) - {"key", "value"}:
                raise ValidationFailed(f"spec.custom_tags[{i}] must be an object with 'key' and optional 'value'")
            tags.append(CustomTag(key=item["key"], value=item.get("value")))
        return tags
    raise ValidationFailed("spec.custom_tags must be a {key: value} object or a list of {key, value}")


def _vs_endpoint_preview(args: dict[str, Any]) -> PlanInfo | None:
    if args.get("action") != "delete" or not args.get("name"):
        return None
    name = args["name"]
    endpoint = ctx().w.vector_search_endpoints.get_endpoint(endpoint_name=name)
    ctx().safety.check_protected("vector search endpoint", name, _endpoint_tags(endpoint), operation="delete")
    warnings = []
    if endpoint.num_indexes:
        warnings.append(f"{endpoint.num_indexes} index(es) are hosted on this endpoint and will no longer be queryable.")
    return PlanInfo(
        description=f"Permanently delete Vector Search endpoint '{name}'.",
        target={"endpoint_name": name},
        details=_endpoint_summary(endpoint),
        warnings=warnings,
        reversible=False,
    )


@tool(
    toolset=TOOLSET,
    title="Vector Search endpoints",
    safety={"list": READ, "get": READ, "create": WRITE, "update": WRITE, "delete": DESTRUCTIVE},
    preview=_vs_endpoint_preview,
)
def manage_vs_endpoint(
    action: Annotated[Literal["list", "get", "create", "update", "delete"], Field(description="Operation to perform.")],
    name: Annotated[str | None, Field(description="Endpoint name (all actions except list).")] = None,
    endpoint_type: Annotated[
        Literal["STANDARD", "STORAGE_OPTIMIZED"], Field(description="create: endpoint type.")
    ] = "STANDARD",
    spec: Spec = None,
    wait_seconds: WaitSeconds = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Vector Search endpoints (the compute that hosts vector indexes).

    Actions:
    - list / get: endpoint state, type, number of indexes, tags.
    - create: name + endpoint_type; optional spec {budget_policy_id, target_qps, usage_policy_id}.
      Provisioning is long-running: returns status 'pending' unless wait_seconds is set.
    - update: spec with any of target_qps, budget_policy_id, custom_tags ({key: value} - replaces all tags).
    - delete: permanently delete the endpoint (requires confirm).
    """
    c = ctx()
    api = c.w.vector_search_endpoints

    if action == "list":
        return paged_response("vector search endpoint(s)", api.list_endpoints(), page_size, page_token, _endpoint_summary)

    require(name, "name", action)

    if action == "get":
        endpoint = api.get_endpoint(endpoint_name=name)
        return ok(f"Vector Search endpoint '{name}' (state={getattr(_endpoint_state(endpoint), 'value', None)}).", endpoint)

    if action == "create":
        waiter = call_with_spec(api.create_endpoint, spec, fixed={"name": name, "endpoint_type": EndpointType(endpoint_type)})
        endpoint = wait_response(waiter)
        warn = c.manifest.safe_track(
            resource_type="vector_search_endpoint",
            resource_id=name,
            name=name,
            created_by_tool="manage_vs_endpoint",
            workspace_host=c.host,
            metadata={"endpoint_type": endpoint_type},
        )
        budget = _wait_budget(wait_seconds)
        terminal = {EndpointStatusState.ONLINE, EndpointStatusState.OFFLINE, EndpointStatusState.RED_STATE}
        if budget:
            endpoint, _ = _poll(lambda: api.get_endpoint(endpoint_name=name), lambda e: _endpoint_state(e) in terminal, budget)
        state = _endpoint_state(endpoint)
        warnings = [warn] if warn else []
        if state == EndpointStatusState.ONLINE:
            return ok(f"Vector Search endpoint '{name}' created and ONLINE.", endpoint, warnings=warnings)
        if state in (EndpointStatusState.OFFLINE, EndpointStatusState.RED_STATE):
            return ok(f"Vector Search endpoint '{name}' was created but is {state.value}.", endpoint, status="failed", warnings=warnings)
        return ok(
            f"Vector Search endpoint '{name}' is being provisioned (state={getattr(state, 'value', state)}).",
            endpoint,
            status="pending",
            warnings=warnings,
            next_steps=[f"Poll: manage_vs_endpoint(action='get', name='{name}')."],
        )

    if action == "update":
        body = dict(require(spec, "spec", action))
        unknown = sorted(set(body) - _ENDPOINT_UPDATABLE)
        if unknown:
            raise ValidationFailed(
                f"Field(s) {', '.join(unknown)} cannot be updated. Updatable: {', '.join(sorted(_ENDPOINT_UPDATABLE))}"
            )
        if not body:
            raise ValidationFailed("Nothing to update: provide fields in 'spec'.")
        results: dict[str, Any] = {}
        if "target_qps" in body:
            results["target_qps"] = api.patch_endpoint(endpoint_name=name, target_qps=body["target_qps"])
        if "budget_policy_id" in body:
            results["budget_policy"] = api.update_endpoint_budget_policy(
                endpoint_name=name, budget_policy_id=require(body["budget_policy_id"], "spec.budget_policy_id")
            )
        if "custom_tags" in body:
            results["custom_tags"] = api.update_endpoint_custom_tags(endpoint_name=name, custom_tags=_custom_tags(body["custom_tags"]))
        return ok(f"Vector Search endpoint '{name}' updated ({', '.join(sorted(body))}).", results)

    if action == "delete":
        endpoint = api.get_endpoint(endpoint_name=name)
        c.safety.check_protected("vector search endpoint", name, _endpoint_tags(endpoint), operation="delete")
        api.delete_endpoint(endpoint_name=name)
        c.manifest.safe_untrack("vector_search_endpoint", name)
        return ok(f"Vector Search endpoint '{name}' deleted.", {"name": name, "deleted": True})

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover


# ----------------------------------------------------------------------------------------------
# manage_vs_index
# ----------------------------------------------------------------------------------------------

_INDEX_NO_UPDATE = (
    "The Vector Search API has no index update operation. To change the primary key, columns, embedding "
    "model or source, delete the index and create it again. To refresh data use action='sync' (Delta Sync "
    "indexes) or manage_vs_data action='upsert'/'delete' (Direct Vector Access indexes)."
)


def _index_summary(index: Any) -> dict[str, Any]:
    return pick(to_jsonable(index), ("name", "endpoint_name", "index_type", "index_subtype", "primary_key", "creator", "status"))


def _get_index(index_name: str) -> Any:
    return ctx().w.vector_search_indexes.get_index(index_name=index_name)


def _require_index_type(index: Any, expected: VectorIndexType, operation: str) -> None:
    if index.index_type != expected:
        actual = getattr(index.index_type, "value", index.index_type)
        if expected == VectorIndexType.DIRECT_ACCESS:
            hint = ("Delta Sync indexes mirror their source Delta table: change the rows in the source table, "
                    "then run manage_vs_data action='sync'.")
        else:
            hint = "Direct Vector Access indexes have no source table; write data with manage_vs_data action='upsert'."
        raise UnsupportedOperation(
            f"{operation} requires a {expected.value} index, but '{index.name}' is {actual}.", hint=hint
        )


def _sync(index_name: str) -> ToolResponse:
    index = _get_index(index_name)
    _require_index_type(index, VectorIndexType.DELTA_SYNC, "sync")
    ctx().w.vector_search_indexes.sync_index(index_name=index_name)
    spec = getattr(index, "delta_sync_index_spec", None)
    warnings = []
    if spec is not None and spec.pipeline_type == PipelineType.CONTINUOUS:
        warnings.append("This index uses a CONTINUOUS pipeline, which syncs automatically.")
    return ok(
        f"Sync of index '{index_name}' triggered; it runs asynchronously.",
        {"index_name": index_name, "sync_requested": True},
        status="pending",
        warnings=warnings,
        next_steps=[f"Check progress: manage_vs_index(action='get', index_name='{index_name}') (status.indexed_row_count, status.ready)."],
    )


def _sync_plan(index_name: str) -> PlanInfo:
    index = _get_index(index_name)
    _require_index_type(index, VectorIndexType.DELTA_SYNC, "sync")
    return PlanInfo(
        description=f"Trigger a sync of Delta Sync index '{index_name}' from its source table (runs a pipeline; uses compute).",
        target={"index_name": index_name},
        details=_index_summary(index),
        reversible=None,
    )


def _vs_index_preview(args: dict[str, Any]) -> PlanInfo | None:
    action, index_name = args.get("action"), args.get("index_name")
    if action == "update":
        raise UnsupportedOperation(_INDEX_NO_UPDATE)
    if not index_name:
        return None
    if action == "sync":
        return _sync_plan(index_name)
    if action != "delete":
        return None
    index = _get_index(index_name)
    ctx().safety.check_protected("vector search index", index_name, operation="delete")
    if index.index_type == VectorIndexType.DELTA_SYNC:
        note = "The source Delta table is NOT affected; the index can be rebuilt by recreating it."
    else:
        note = "All vectors and rows stored in this Direct Vector Access index are permanently lost."
    return PlanInfo(
        description=f"Delete vector search index '{index_name}'.",
        target={"index_name": index_name},
        details=_index_summary(index),
        warnings=[note],
        reversible=False,
    )


@tool(
    toolset=TOOLSET,
    title="Vector Search indexes",
    safety={
        "list": READ,
        "get": READ,
        "create": WRITE,
        "update": WRITE,
        "delete": DESTRUCTIVE,
        "sync": WRITE | EXECUTION,
    },
    preview=_vs_index_preview,
)
def manage_vs_index(
    action: Annotated[
        Literal["list", "get", "create", "update", "delete", "sync"], Field(description="Operation to perform.")
    ],
    index_name: Annotated[
        str | None, Field(description="Full index name catalog.schema.index (all actions except list).")
    ] = None,
    endpoint_name: Annotated[str | None, Field(description="Vector Search endpoint (required for list and create).")] = None,
    spec: Spec = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Vector Search indexes.

    Actions:
    - list (endpoint_name) / get (index_name): type, primary key, status and readiness.
    - create: index_name + endpoint_name + spec {primary_key, index_type: DELTA_SYNC|DIRECT_ACCESS,
      index_subtype?, delta_sync_index_spec: {source_table, pipeline_type: TRIGGERED|CONTINUOUS,
      embedding_source_columns: [{name, embedding_model_endpoint_name}] or embedding_vector_columns,
      columns_to_sync?} | direct_access_index_spec: {embedding_vector_columns: [{name, embedding_dimension}],
      schema_json}}.
    - sync: trigger a Delta Sync index refresh. delete: delete the index (requires confirm).
    - update: not supported by the API (recreate the index instead).
    """
    c = ctx()
    api = c.w.vector_search_indexes

    if action == "list":
        require(endpoint_name, "endpoint_name", action)
        return paged_response(
            f"index(es) on endpoint '{endpoint_name}'", api.list_indexes(endpoint_name=endpoint_name), page_size, page_token, _index_summary
        )
    if action == "update":
        raise UnsupportedOperation(_INDEX_NO_UPDATE)

    require(index_name, "index_name", action)

    if action == "get":
        index = api.get_index(index_name=index_name)
        ready = getattr(index.status, "ready", None) if index.status else None
        return ok(f"Index '{index_name}' (ready={ready}).", index)

    if action == "create":
        require(endpoint_name, "endpoint_name", action)
        index = call_with_spec(api.create_index, spec, fixed={"name": index_name, "endpoint_name": endpoint_name})
        warn = c.manifest.safe_track(
            resource_type="vector_search_index",
            resource_id=index_name,
            name=index_name,
            created_by_tool="manage_vs_index",
            workspace_host=c.host,
            metadata={"endpoint_name": endpoint_name},
        )
        ready = bool(getattr(getattr(index, "status", None), "ready", False))
        return ok(
            f"Index '{index_name}' created on endpoint '{endpoint_name}'" + ("." if ready else "; initial build in progress."),
            index,
            status="success" if ready else "pending",
            warnings=[warn] if warn else None,
            next_steps=[] if ready else [f"Poll: manage_vs_index(action='get', index_name='{index_name}') until status.ready is true."],
        )

    if action == "delete":
        _get_index(index_name)
        c.safety.check_protected("vector search index", index_name, operation="delete")
        api.delete_index(index_name=index_name)
        c.manifest.safe_untrack("vector_search_index", index_name)
        return ok(f"Index '{index_name}' deleted.", {"index_name": index_name, "deleted": True})

    if action == "sync":
        return _sync(index_name)

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover


# ----------------------------------------------------------------------------------------------
# query_vs_index
# ----------------------------------------------------------------------------------------------


def _query_result(resp: Any) -> dict[str, Any]:
    manifest = getattr(resp, "manifest", None)
    columns = [col.name for col in (getattr(manifest, "columns", None) or [])] if manifest else []
    result = getattr(resp, "result", None)
    rows = (getattr(result, "data_array", None) or []) if result else []
    records = [dict(zip(columns, row, strict=False)) for row in rows]
    out: dict[str, Any] = {
        "columns": columns,
        "records": records,
        "row_count": len(records),
        "next_page_token": getattr(resp, "next_page_token", None),
    }
    if "score" in columns:
        out["scores"] = [r.get("score") for r in records]
        out["score_note"] = "'score' is the similarity/relevance score appended by Vector Search (higher is more relevant)."
    facets = getattr(resp, "facet_result", None)
    if facets is not None:
        out["facets"] = to_jsonable(facets)
        facet_cols = getattr(manifest, "facet_columns", None) if manifest else None
        if facet_cols:
            out["facet_columns"] = [col.name for col in facet_cols]
    return out


@tool(toolset=TOOLSET, title="Query a vector index", safety=READ | EXECUTION)
def query_vs_index(
    index_name: Annotated[str, Field(description="Full index name catalog.schema.index.")],
    columns: Annotated[list[str] | None, Field(description="Columns to return (required unless page_token is given).")] = None,
    query_text: Annotated[
        str | None, Field(description="Text query (indexes with a model-computed embedding, or HYBRID/FULL_TEXT).")
    ] = None,
    query_vector: Annotated[
        list[float] | None, Field(description="Query embedding (Direct Access or self-managed-embedding indexes).")
    ] = None,
    num_results: Annotated[int | None, Field(description="Results to return (default 10, server-capped).", ge=1)] = None,
    filters: Annotated[
        dict[str, Any] | str | None,
        Field(description="Filter object (sent as filters_json), e.g. {'category': 'news', 'id >': 5, 'tag': ['a', 'b']}."),
    ] = None,
    query_type: Annotated[
        Literal["ANN", "HYBRID", "FULL_TEXT"] | None, Field(description="Search type (default ANN).")
    ] = None,
    options: Annotated[
        dict[str, Any] | None,
        Field(
            description="Extra query_index fields: score_threshold, query_columns, sort_columns, facets, "
            "columns_to_rerank, reranker. Unknown fields are rejected."
        ),
    ] = None,
    page_token: Annotated[
        str | None, Field(description="next_page_token from a previous query_vs_index response to fetch the next page.")
    ] = None,
    endpoint_name: Annotated[str | None, Field(description="Endpoint name (optional, used with page_token).")] = None,
) -> ToolResponse:
    """Run a similarity / hybrid / full-text search against a Vector Search index.

    Returns the matching records as a list of {column: value} objects, their scores (the
    'score' column), the column list, facets (if requested) and query information. Pass the
    returned next_page_token as page_token to continue.
    """
    api = ctx().w.vector_search_indexes
    if page_token:
        resp = api.query_next_page(index_name=index_name, endpoint_name=endpoint_name, page_token=page_token)
        data = _query_result(resp)
        data["query"] = {"index_name": index_name, "page_token": page_token}
        return _query_response(index_name, data)

    if not columns:
        raise ValidationFailed("Parameter 'columns' is required (list of columns to return).")
    if not query_text and not query_vector:
        raise ValidationFailed("Provide query_text or query_vector.")
    filters_json = json.dumps(filters) if isinstance(filters, dict) else filters
    if isinstance(filters_json, str):
        try:
            json.loads(filters_json)
        except json.JSONDecodeError as exc:
            raise ValidationFailed(f"'filters' is not valid JSON: {exc}") from exc
    limit = _result_cap(num_results)
    fixed: dict[str, Any] = {"index_name": index_name, "columns": columns, "num_results": limit}
    for key, value in (("query_text", query_text), ("query_vector", query_vector), ("filters_json", filters_json), ("query_type", query_type)):
        if value is not None:
            fixed[key] = value
    resp = call_with_spec(api.query_index, options, fixed=fixed)
    data = _query_result(resp)
    data["query"] = {
        "index_name": index_name,
        "query_type": query_type or "ANN",
        "query_text": query_text,
        "query_vector_dimension": len(query_vector) if query_vector else None,
        "filters": filters,
        "num_results": limit,
        **({"options": options} if options else {}),
    }
    warnings = [f"num_results capped at {limit}."] if num_results and num_results > limit else None
    return _query_response(index_name, data, warnings)


def _query_response(index_name: str, data: dict[str, Any], warnings: list[str] | None = None) -> ToolResponse:
    next_steps = []
    if data.get("next_page_token"):
        next_steps.append(f"More results: query_vs_index(index_name='{index_name}', page_token='<next_page_token>').")
    return ok(f"{data['row_count']} matching record(s) from index '{index_name}'.", data, warnings=warnings, next_steps=next_steps)


# ----------------------------------------------------------------------------------------------
# manage_vs_data
# ----------------------------------------------------------------------------------------------


def _value(v: Any) -> Any:
    if v is None:
        return None
    if v.struct_value is not None:
        return _struct(v.struct_value)
    if v.list_value is not None:
        return [_value(x) for x in v.list_value.values or []]
    if v.string_value is not None:
        return v.string_value
    if v.number_value is not None:
        return v.number_value
    return v.bool_value


def _struct(s: Any) -> dict[str, Any]:
    return {entry.key: _value(entry.value) for entry in (s.fields or [])}


def _compact_vectors(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, list) and len(value) > 32 and all(isinstance(x, int | float) for x in value):
            out[key] = f"<vector dim={len(value)}>"
        else:
            out[key] = value
    return out


def _write_status(resp: Any) -> str:
    status = getattr(getattr(resp, "status", None), "value", None)
    return {"SUCCESS": "success", "PARTIAL_SUCCESS": "partial_failure", "FAILURE": "failed"}.get(status or "", "success")


def _vs_data_preview(args: dict[str, Any]) -> PlanInfo | None:
    action, index_name = args.get("action"), args.get("index_name")
    if not index_name:
        return None
    if action == "sync":
        return _sync_plan(index_name)
    if action == "delete":
        keys = args.get("primary_keys") or []
        index = _get_index(index_name)
        _require_index_type(index, VectorIndexType.DIRECT_ACCESS, "delete")
        ctx().safety.check_protected("vector search index", index_name, operation="delete rows from")
        return PlanInfo(
            description=f"Delete {len(keys)} row(s) by primary key from Direct Access index '{index_name}'.",
            target={"index_name": index_name},
            details={"primary_key_column": index.primary_key, "primary_keys_sample": keys[:20], "count": len(keys)},
            reversible=False,
        )
    if action == "upsert":
        index = _get_index(index_name)
        _require_index_type(index, VectorIndexType.DIRECT_ACCESS, "upsert")
        rows = _upsert_rows(args.get("records"), args.get("inputs_json"))
        return PlanInfo(
            description=f"Insert or overwrite {len(rows)} row(s) (matched on '{index.primary_key}') in Direct Access index '{index_name}'.",
            target={"index_name": index_name},
            details={"row_count": len(rows), "columns": sorted({k for r in rows if isinstance(r, dict) for k in r})},
            reversible=False,
        )
    return None


def _upsert_rows(records: list[dict[str, Any]] | None, inputs_json: str | list[Any] | None) -> list[Any]:
    if records and inputs_json:
        raise ValidationFailed("Pass either 'records' or 'inputs_json', not both.")
    if records:
        return records
    require(inputs_json, "records or inputs_json", "upsert")
    rows: Any = inputs_json
    if isinstance(inputs_json, str):
        try:
            rows = json.loads(inputs_json)
        except json.JSONDecodeError as exc:
            raise ValidationFailed(f"inputs_json is not valid JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValidationFailed("inputs_json must be a non-empty JSON array of row objects.")
    return rows


@tool(
    toolset=TOOLSET,
    title="Vector index data",
    safety={"scan": READ, "upsert": WRITE, "delete": DESTRUCTIVE, "sync": WRITE | EXECUTION},
    preview=_vs_data_preview,
)
def manage_vs_data(
    action: Annotated[Literal["scan", "upsert", "delete", "sync"], Field(description="Operation to perform.")],
    index_name: Annotated[str, Field(description="Full index name catalog.schema.index.")],
    records: Annotated[
        list[dict[str, Any]] | None,
        Field(description="upsert: rows to write, each including the primary key and the vector column(s)."),
    ] = None,
    inputs_json: Annotated[
        str | list[dict[str, Any]] | None,
        Field(description="upsert: alternative to records - a JSON array (string or already-parsed array)."),
    ] = None,
    primary_keys: Annotated[list[str] | None, Field(description="delete: primary key values of rows to delete.")] = None,
    last_primary_key: Annotated[
        str | None, Field(description="scan: continue after this primary key (from a previous scan).")
    ] = None,
    num_results: Annotated[int | None, Field(description="scan: rows to return (default 10, server-capped).", ge=1)] = None,
    include_vectors: Annotated[
        bool, Field(description="scan: return full embedding vectors (default: summarized as <vector dim=N>).")
    ] = False,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Read and write the data inside a Vector Search index.

    Actions:
    - scan: page through stored rows (last_primary_key to continue).
    - upsert: insert/overwrite rows in a Direct Vector Access index (records or inputs_json).
    - delete: delete rows by primary key from a Direct Vector Access index (requires confirm).
    - sync: trigger a refresh of a Delta Sync index from its source table.
    Delta Sync indexes cannot be written directly: modify the source table and sync instead.
    """
    c = ctx()
    api = c.w.vector_search_indexes

    if action == "scan":
        limit = _result_cap(num_results)
        resp = api.scan_index(index_name=index_name, last_primary_key=last_primary_key, num_results=limit)
        rows = [_struct(s) for s in (resp.data or [])]
        if not include_vectors:
            rows = [_compact_vectors(r) for r in rows]
        data = {"rows": rows, "row_count": len(rows), "last_primary_key": resp.last_primary_key}
        next_steps = []
        if resp.last_primary_key and len(rows) >= limit:
            next_steps.append(
                f"Next page: manage_vs_data(action='scan', index_name='{index_name}', last_primary_key='{resp.last_primary_key}')."
            )
        return ok(f"Scanned {len(rows)} row(s) from index '{index_name}'.", data, next_steps=next_steps)

    if action == "sync":
        return _sync(index_name)

    index = _get_index(index_name)

    if action == "upsert":
        _require_index_type(index, VectorIndexType.DIRECT_ACCESS, "upsert")
        rows = _upsert_rows(records, inputs_json)
        resp = api.upsert_data_vector_index(index_name=index_name, inputs_json=json.dumps(rows))
        status = _write_status(resp)
        result = to_jsonable(resp.result) or {}
        failed = result.get("failed_primary_keys") or []
        return ok(
            f"Upsert into '{index_name}': {result.get('success_row_count', 0)} row(s) written"
            + (f", {len(failed)} failed." if failed else "."),
            resp,
            status=status,
            warnings=[f"Failed primary keys: {failed[:20]}"] if failed else None,
        )

    if action == "delete":
        keys = require(primary_keys, "primary_keys", action)
        _require_index_type(index, VectorIndexType.DIRECT_ACCESS, "delete")
        c.safety.check_protected("vector search index", index_name, operation="delete rows from")
        resp = api.delete_data_vector_index(index_name=index_name, primary_keys=[str(k) for k in keys])
        status = _write_status(resp)
        result = to_jsonable(resp.result) or {}
        failed = result.get("failed_primary_keys") or []
        return ok(
            f"Delete from '{index_name}': {result.get('success_row_count', 0)} row(s) deleted"
            + (f", {len(failed)} failed." if failed else "."),
            resp,
            status=status,
            warnings=[f"Failed primary keys: {failed[:20]}"] if failed else None,
        )

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover
