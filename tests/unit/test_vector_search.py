"""Tests for the vector_search toolset."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service._internal import Wait
from databricks.sdk.service.vectorsearch import (
    ColumnInfo,
    CustomTag,
    DeleteDataResult,
    DeleteDataStatus,
    DeleteDataVectorIndexResponse,
    EndpointInfo,
    EndpointStatus,
    EndpointStatusState,
    EndpointType,
    ListValue,
    MapStringValueEntry,
    MiniVectorIndex,
    QueryVectorIndexResponse,
    ResultData,
    ResultManifest,
    ScanVectorIndexResponse,
    Struct,
    UpsertDataResult,
    UpsertDataStatus,
    UpsertDataVectorIndexResponse,
    Value,
    VectorIndex,
    VectorIndexType,
)

TOOLSETS = ("vector_search",)


@pytest.fixture
def vs(make_harness):
    def factory(**overrides):
        h = make_harness(toolsets=TOOLSETS, **overrides)
        return h

    return factory


def _index(index_type=VectorIndexType.DIRECT_ACCESS, name="main.rag.docs_idx"):
    return VectorIndex(name=name, endpoint_name="vs1", index_type=index_type, primary_key="id")


# ------------------------------------------------------------------------------------------
# manage_vs_endpoint
# ------------------------------------------------------------------------------------------


async def test_endpoint_list_paginates(vs):
    h = vs()
    h.w.vector_search_endpoints.list_endpoints.return_value = [EndpointInfo(name=f"vs{i}") for i in range(3)]
    result = await h.call("manage_vs_endpoint", {"action": "list", "page_size": 2})
    assert [e["name"] for e in result["data"]] == ["vs0", "vs1"]
    assert result["page"]["has_more"] is True


async def test_endpoint_create_pending_and_tracked(vs):
    h = vs()
    info = EndpointInfo(name="vs-new", endpoint_status=EndpointStatus(state=EndpointStatusState.PROVISIONING))
    h.w.vector_search_endpoints.create_endpoint.return_value = Wait(lambda **kw: None, response=info, endpoint_name="vs-new")
    result = await h.call("manage_vs_endpoint", {"action": "create", "name": "vs-new", "spec": {"budget_policy_id": "bp1"}})
    assert result["status"] == "pending"
    h.w.vector_search_endpoints.create_endpoint.assert_called_once_with(
        name="vs-new", endpoint_type=EndpointType.STANDARD, budget_policy_id="bp1"
    )
    manifest = json.loads(h.settings.manifest_path.read_text())
    assert manifest["resources"][0]["resource_type"] == "vector_search_endpoint"


async def test_endpoint_create_wait_online(vs):
    h = vs()
    h.w.vector_search_endpoints.create_endpoint.return_value = Wait(
        lambda **kw: None, response=EndpointInfo(name="vs-new"), endpoint_name="vs-new"
    )
    h.w.vector_search_endpoints.get_endpoint.return_value = EndpointInfo(
        name="vs-new", endpoint_status=EndpointStatus(state=EndpointStatusState.ONLINE)
    )
    result = await h.call("manage_vs_endpoint", {"action": "create", "name": "vs-new", "wait_seconds": 10})
    assert result["status"] == "success"


async def test_endpoint_create_rejects_unknown_field(vs):
    h = vs()
    msg = await h.call_error("manage_vs_endpoint", {"action": "create", "name": "x", "spec": {"num_replicas": 2}})
    assert "Unknown field" in msg
    h.w.vector_search_endpoints.create_endpoint.assert_not_called()


async def test_endpoint_update_tags_and_qps(vs):
    h = vs()
    await h.call("manage_vs_endpoint", {"action": "update", "name": "vs1", "spec": {"custom_tags": {"team": "ml"}, "target_qps": 50}})
    h.w.vector_search_endpoints.update_endpoint_custom_tags.assert_called_once_with(
        endpoint_name="vs1", custom_tags=[CustomTag(key="team", value="ml")]
    )
    h.w.vector_search_endpoints.patch_endpoint.assert_called_once_with(endpoint_name="vs1", target_qps=50)
    msg = await h.call_error("manage_vs_endpoint", {"action": "update", "name": "vs1", "spec": {"endpoint_type": "STANDARD"}})
    assert "cannot be updated" in msg


async def test_endpoint_delete_confirm_and_protection(vs):
    h = vs()
    h.w.vector_search_endpoints.get_endpoint.return_value = EndpointInfo(name="vs-dev", num_indexes=2)
    result = await h.call("manage_vs_endpoint", {"action": "delete", "name": "vs-dev"})
    assert result["status"] == "confirmation_required"
    assert any("2 index" in w for w in result["plan"]["warnings"])
    h.w.vector_search_endpoints.delete_endpoint.assert_not_called()
    await h.call("manage_vs_endpoint", {"action": "delete", "name": "vs-dev", "confirm": True})
    h.w.vector_search_endpoints.delete_endpoint.assert_called_once_with(endpoint_name="vs-dev")

    h.w.vector_search_endpoints.get_endpoint.return_value = EndpointInfo(name="vs-prod")
    msg = await h.call_error("manage_vs_endpoint", {"action": "delete", "name": "vs-prod", "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg


async def test_endpoint_get_not_found(vs):
    h = vs()
    h.w.vector_search_endpoints.get_endpoint.side_effect = NotFound("endpoint nope not found")
    msg = await h.call_error("manage_vs_endpoint", {"action": "get", "name": "nope"})
    assert "[NOT_FOUND]" in msg


# ------------------------------------------------------------------------------------------
# manage_vs_index
# ------------------------------------------------------------------------------------------


async def test_index_list_requires_endpoint(vs):
    h = vs()
    msg = await h.call_error("manage_vs_index", {"action": "list"})
    assert "endpoint_name" in msg
    h.w.vector_search_indexes.list_indexes.return_value = [MiniVectorIndex(name="a.b.c", index_type=VectorIndexType.DELTA_SYNC)]
    result = await h.call("manage_vs_index", {"action": "list", "endpoint_name": "vs1"})
    assert result["data"] == [{"name": "a.b.c", "index_type": "DELTA_SYNC"}]


async def test_index_create_spec(vs):
    h = vs()
    h.w.vector_search_indexes.create_index.return_value = _index(VectorIndexType.DELTA_SYNC)
    spec = {
        "primary_key": "id",
        "index_type": "DELTA_SYNC",
        "delta_sync_index_spec": {
            "source_table": "main.rag.docs",
            "pipeline_type": "TRIGGERED",
            "embedding_source_columns": [{"name": "text", "embedding_model_endpoint_name": "databricks-gte-large-en"}],
        },
    }
    result = await h.call("manage_vs_index", {"action": "create", "index_name": "main.rag.docs_idx", "endpoint_name": "vs1", "spec": spec})
    assert result["status"] == "pending"
    kwargs = h.w.vector_search_indexes.create_index.call_args.kwargs
    assert kwargs["name"] == "main.rag.docs_idx" and kwargs["endpoint_name"] == "vs1"
    assert json.loads(h.settings.manifest_path.read_text())["resources"][0]["resource_type"] == "vector_search_index"

    msg = await h.call_error(
        "manage_vs_index",
        {"action": "create", "index_name": "x", "endpoint_name": "vs1", "spec": {**spec, "replicas": 3}},
    )
    assert "Unknown field" in msg


async def test_index_update_unsupported(vs):
    h = vs()
    msg = await h.call_error("manage_vs_index", {"action": "update", "index_name": "a.b.c", "spec": {"primary_key": "x"}})
    assert "UNSUPPORTED_OPERATION" in msg and "delete the index and create it again" in msg


async def test_index_delete_dry_run_and_confirm(vs):
    h = vs()
    h.w.vector_search_indexes.get_index.return_value = _index(VectorIndexType.DELTA_SYNC)
    result = await h.call("manage_vs_index", {"action": "delete", "index_name": "main.rag.docs_idx", "dry_run": True})
    assert result["status"] == "dry_run"
    assert any("NOT affected" in w for w in result["plan"]["warnings"])
    result = await h.call("manage_vs_index", {"action": "delete", "index_name": "main.rag.docs_idx"})
    assert result["status"] == "confirmation_required"
    h.w.vector_search_indexes.delete_index.assert_not_called()
    await h.call("manage_vs_index", {"action": "delete", "index_name": "main.rag.docs_idx", "confirm": True})
    h.w.vector_search_indexes.delete_index.assert_called_once_with(index_name="main.rag.docs_idx")


async def test_index_sync_requires_delta_sync(vs):
    h = vs()
    h.w.vector_search_indexes.get_index.return_value = _index(VectorIndexType.DIRECT_ACCESS)
    msg = await h.call_error("manage_vs_index", {"action": "sync", "index_name": "main.rag.docs_idx"})
    assert "UNSUPPORTED_OPERATION" in msg
    h.w.vector_search_indexes.sync_index.assert_not_called()

    h.w.vector_search_indexes.get_index.return_value = _index(VectorIndexType.DELTA_SYNC)
    result = await h.call("manage_vs_index", {"action": "sync", "index_name": "main.rag.docs_idx"})
    assert result["status"] == "pending"
    h.w.vector_search_indexes.sync_index.assert_called_once_with(index_name="main.rag.docs_idx")


# ------------------------------------------------------------------------------------------
# query_vs_index
# ------------------------------------------------------------------------------------------


def _query_response(next_token=None):
    return QueryVectorIndexResponse(
        manifest=ResultManifest(columns=[ColumnInfo(name="id"), ColumnInfo(name="text"), ColumnInfo(name="score")]),
        result=ResultData(data_array=[["1", "alpha", 0.92], ["2", "beta", 0.81]], row_count=2),
        next_page_token=next_token,
    )


async def test_query_zips_records_and_scores(vs):
    h = vs()
    h.w.vector_search_indexes.query_index.return_value = _query_response("tok2")
    result = await h.call(
        "query_vs_index",
        {
            "index_name": "main.rag.docs_idx",
            "columns": ["id", "text"],
            "query_text": "what is alpha",
            "filters": {"category": "news"},
            "query_type": "HYBRID",
            "num_results": 5,
        },
    )
    data = result["data"]
    assert data["records"] == [{"id": "1", "text": "alpha", "score": 0.92}, {"id": "2", "text": "beta", "score": 0.81}]
    assert data["scores"] == [0.92, 0.81]
    assert data["next_page_token"] == "tok2"
    assert data["query"]["query_type"] == "HYBRID"
    assert result["next_steps"]
    kwargs = h.w.vector_search_indexes.query_index.call_args.kwargs
    assert kwargs["filters_json"] == json.dumps({"category": "news"})
    assert kwargs["num_results"] == 5 and kwargs["query_type"] == "HYBRID"


async def test_query_caps_results_and_validates_options(vs):
    h = vs(sql_max_rows=3)
    h.w.vector_search_indexes.query_index.return_value = _query_response()
    result = await h.call("query_vs_index", {"index_name": "i", "columns": ["id"], "query_vector": [0.1, 0.2], "num_results": 100})
    assert h.w.vector_search_indexes.query_index.call_args.kwargs["num_results"] == 3
    assert result["warnings"]
    msg = await h.call_error("query_vs_index", {"index_name": "i", "columns": ["id"], "query_text": "x", "options": {"bogus": 1}})
    assert "Unknown field" in msg
    msg = await h.call_error("query_vs_index", {"index_name": "i", "columns": ["id"]})
    assert "query_text or query_vector" in msg


async def test_query_next_page(vs):
    h = vs()
    h.w.vector_search_indexes.query_next_page.return_value = _query_response()
    result = await h.call("query_vs_index", {"index_name": "i", "page_token": "tok2", "endpoint_name": "vs1"})
    assert result["data"]["row_count"] == 2
    h.w.vector_search_indexes.query_next_page.assert_called_once_with(index_name="i", endpoint_name="vs1", page_token="tok2")
    h.w.vector_search_indexes.query_index.assert_not_called()


async def test_query_allowed_in_read_only_but_data_writes_blocked(vs):
    h = vs(read_only=True)
    h.w.vector_search_indexes.query_index.return_value = _query_response()
    result = await h.call("query_vs_index", {"index_name": "i", "columns": ["id"], "query_text": "x"})
    assert result["status"] == "success"
    msg = await h.call_error("manage_vs_data", {"action": "upsert", "index_name": "i", "records": [{"id": "1"}]})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.vector_search_indexes.upsert_data_vector_index.assert_not_called()


# ------------------------------------------------------------------------------------------
# manage_vs_data
# ------------------------------------------------------------------------------------------


async def test_data_upsert_direct_access(vs):
    h = vs()
    h.w.vector_search_indexes.get_index.return_value = _index()
    h.w.vector_search_indexes.upsert_data_vector_index.return_value = UpsertDataVectorIndexResponse(
        status=UpsertDataStatus.SUCCESS, result=UpsertDataResult(success_row_count=2)
    )
    records = [{"id": "1", "vec": [0.1, 0.2]}, {"id": "2", "vec": [0.3, 0.4]}]
    result = await h.call("manage_vs_data", {"action": "upsert", "index_name": "main.rag.docs_idx", "records": records})
    assert result["status"] == "success"
    h.w.vector_search_indexes.upsert_data_vector_index.assert_called_once_with(
        index_name="main.rag.docs_idx", inputs_json=json.dumps(records)
    )


async def test_data_upsert_partial_failure(vs):
    h = vs()
    h.w.vector_search_indexes.get_index.return_value = _index()
    h.w.vector_search_indexes.upsert_data_vector_index.return_value = UpsertDataVectorIndexResponse(
        status=UpsertDataStatus.PARTIAL_SUCCESS, result=UpsertDataResult(success_row_count=1, failed_primary_keys=["2"])
    )
    result = await h.call("manage_vs_data", {"action": "upsert", "index_name": "i", "inputs_json": '[{"id": "1"}, {"id": "2"}]'})
    assert result["status"] == "partial_failure"


async def test_data_upsert_rejected_for_delta_sync(vs):
    h = vs()
    h.w.vector_search_indexes.get_index.return_value = _index(VectorIndexType.DELTA_SYNC)
    msg = await h.call_error("manage_vs_data", {"action": "upsert", "index_name": "i", "records": [{"id": "1"}]})
    assert "UNSUPPORTED_OPERATION" in msg and "source" in msg
    h.w.vector_search_indexes.upsert_data_vector_index.assert_not_called()


async def test_data_delete_requires_confirm(vs):
    h = vs()
    h.w.vector_search_indexes.get_index.return_value = _index()
    h.w.vector_search_indexes.delete_data_vector_index.return_value = DeleteDataVectorIndexResponse(
        status=DeleteDataStatus.SUCCESS, result=DeleteDataResult(success_row_count=2)
    )
    args = {"action": "delete", "index_name": "main.rag.docs_idx", "primary_keys": ["1", "2"]}
    result = await h.call("manage_vs_data", args)
    assert result["status"] == "confirmation_required"
    assert result["plan"]["details"]["count"] == 2
    h.w.vector_search_indexes.delete_data_vector_index.assert_not_called()
    result = await h.call("manage_vs_data", {**args, "confirm": True})
    assert result["status"] == "success"
    h.w.vector_search_indexes.delete_data_vector_index.assert_called_once_with(
        index_name="main.rag.docs_idx", primary_keys=["1", "2"]
    )


async def test_data_scan_converts_structs(vs):
    h = vs()
    vector = Value(list_value=ListValue(values=[Value(number_value=float(i)) for i in range(64)]))
    row = Struct(
        fields=[
            MapStringValueEntry(key="id", value=Value(string_value="1")),
            MapStringValueEntry(key="flag", value=Value(bool_value=True)),
            MapStringValueEntry(key="vec", value=vector),
        ]
    )
    h.w.vector_search_indexes.scan_index.return_value = ScanVectorIndexResponse(data=[row], last_primary_key="1")
    result = await h.call("manage_vs_data", {"action": "scan", "index_name": "i", "num_results": 1})
    assert result["data"]["rows"] == [{"id": "1", "flag": True, "vec": "<vector dim=64>"}]
    assert result["next_steps"]
    full = await h.call("manage_vs_data", {"action": "scan", "index_name": "i", "include_vectors": True})
    assert len(full["data"]["rows"][0]["vec"]) == 64
