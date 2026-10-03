"""SQL toolset tests (Statement Execution API mocked)."""

from __future__ import annotations

import pytest
from databricks.sdk.service import catalog, sql

TOOLSETS = ("sql", "compute")


def _wh(id_, name, state, serverless=False, wtype="PRO"):
    return sql.EndpointInfo(id=id_, name=name, state=sql.State(state), enable_serverless_compute=serverless,
                            warehouse_type=sql.EndpointInfoWarehouseType(wtype))


def _resp(state="SUCCEEDED", columns=None, rows=None, total=None, next_chunk=None, error=None, sid="stmt-1"):
    manifest = None
    if columns is not None:
        manifest = sql.ResultManifest(
            schema=sql.ResultSchema(columns=[sql.ColumnInfo(name=n, type_text=t, position=i)
                                             for i, (n, t) in enumerate(columns)]),
            total_row_count=total if total is not None else len(rows or []),
            truncated=False,
        )
    return sql.StatementResponse(
        statement_id=sid,
        status=sql.StatementStatus(state=sql.StatementState(state), error=error),
        manifest=manifest,
        result=sql.ResultData(data_array=rows or [], next_chunk_index=next_chunk) if rows is not None else None,
    )


@pytest.fixture
def h(make_harness):
    h = make_harness(toolsets=TOOLSETS)
    h.w.warehouses.list.return_value = [
        _wh("stopped1", "b-stopped", "STOPPED", serverless=True),
        _wh("run-classic", "z-classic", "RUNNING", wtype="CLASSIC"),
        _wh("run-serverless", "a-serverless", "RUNNING", serverless=True),
    ]
    return h


async def test_select_returns_typed_rows_and_selection(h):
    h.w.statement_execution.execute_statement.return_value = _resp(
        columns=[("id", "bigint"), ("name", "string"), ("ok", "boolean"), ("score", "double")],
        rows=[["1", "a", "true", "1.5"], ["2", None, "false", "2"]],
    )
    out = await h.call("execute_sql", {"statement": "SELECT * FROM t", "row_format": "objects"})
    assert out["status"] == "success"
    assert out["safety"] == ["EXECUTION", "READ_ONLY"]
    result, meta = out["data"]["result"], out["data"]["execution"]
    assert result["rows"][0] == {"id": 1, "name": "a", "ok": True, "score": 1.5}
    assert result["rows"][1]["name"] is None
    assert meta["warehouse"]["warehouse_id"] == "run-serverless"
    assert "prefer_running" in meta["warehouse"]["reason"]
    assert meta["statement_kind"] == "read"
    kwargs = h.w.statement_execution.execute_statement.call_args.kwargs
    assert kwargs["warehouse_id"] == "run-serverless"
    assert kwargs["on_wait_timeout"] == sql.ExecuteStatementRequestOnWaitTimeout.CONTINUE


async def test_parameters_are_bound_not_interpolated(h):
    h.w.statement_execution.execute_statement.return_value = _resp(columns=[("x", "int")], rows=[["1"]])
    await h.call("execute_sql", {"statement": "SELECT :v AS x", "parameters": {"v": "1; DROP TABLE t"}})
    kwargs = h.w.statement_execution.execute_statement.call_args.kwargs
    assert kwargs["statement"] == "SELECT :v AS x"
    assert kwargs["parameters"][0].value == "1; DROP TABLE t"


async def test_row_cap_and_chunk_following(make_harness):
    h = make_harness(toolsets=TOOLSETS, sql_max_rows=3, default_warehouse_id="wh")
    h.w.statement_execution.execute_statement.return_value = _resp(
        columns=[("n", "int")], rows=[["1"], ["2"]], total=10, next_chunk=1)
    h.w.statement_execution.get_statement_result_chunk_n.return_value = sql.ResultData(
        data_array=[["3"], ["4"]], next_chunk_index=2)
    out = await h.call("execute_sql", {"statement": "SELECT n FROM t"})
    result = out["data"]["result"]
    assert result["rows"] == [[1], [2], [3]]
    assert result["truncated"] is True and result["total_row_count"] == 10
    assert "configured default" in out["data"]["execution"]["warehouse"]["reason"]


async def test_pending_statement(h):
    h.w.statement_execution.execute_statement.return_value = _resp(state="RUNNING", sid="s-9")
    out = await h.call("execute_sql", {"statement": "SELECT * FROM big"})
    assert out["status"] == "pending"
    assert out["data"]["execution"]["statement_id"] == "s-9"
    assert "manage_sql_statement" in out["next_steps"][0]


async def test_failed_statement_error_category(h):
    h.w.statement_execution.execute_statement.return_value = _resp(
        state="FAILED", error=sql.ServiceError(message="[TABLE_OR_VIEW_NOT_FOUND] The table or view `x` cannot be found."))
    err = await h.call_error("execute_sql", {"statement": "SELECT * FROM x"})
    assert "[NOT_FOUND]" in err


async def test_destructive_requires_confirmation(h):
    out = await h.call("execute_sql", {"statement": "DROP TABLE main.default.t"})
    assert out["status"] == "confirmation_required"
    assert "DESTRUCTIVE" in out["safety"]
    assert out["plan"]["details"]["statements"][0]["kind"] == "destructive"
    h.w.statement_execution.execute_statement.assert_not_called()

    h.w.statement_execution.execute_statement.return_value = _resp()
    out = await h.call("execute_sql", {"statement": "DROP TABLE main.default.t", "confirm": True})
    assert out["status"] == "success"
    h.w.statement_execution.execute_statement.assert_called_once()


async def test_write_runs_without_confirmation_but_respects_read_only(h, make_harness):
    h.w.statement_execution.execute_statement.return_value = _resp()
    out = await h.call("execute_sql", {"statement": "INSERT INTO t VALUES (1)"})
    assert out["status"] == "success"

    ro = make_harness(toolsets=TOOLSETS, read_only=True, default_warehouse_id="wh")
    ro.w.statement_execution.execute_statement.return_value = _resp(columns=[("a", "int")], rows=[["1"]])
    assert (await ro.call("execute_sql", {"statement": "SELECT 1 AS a"}))["status"] == "success"
    err = await ro.call_error("execute_sql", {"statement": "INSERT INTO t VALUES (1)"})
    assert "BLOCKED_BY_SAFETY_POLICY" in err


async def test_execute_sql_rejects_multiple_statements(h):
    err = await h.call_error("execute_sql", {"statement": "SELECT 1; SELECT 2"})
    assert "execute_sql_multi" in err


async def test_dry_run_sql(h):
    out = await h.call("execute_sql", {"statement": "GRANT SELECT ON TABLE t TO `x`", "dry_run": True})
    assert out["status"] == "dry_run"
    assert "SECURITY_SENSITIVE" in out["safety"]
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_multi_stops_at_first_failure(h):
    h.w.statement_execution.execute_statement.side_effect = [
        _resp(),
        _resp(state="FAILED", error=sql.ServiceError(message="syntax error")),
        _resp(),
    ]
    out = await h.call("execute_sql_multi", {"script": "CREATE TABLE a (x INT); INSERT INTO a VALUES (; SELECT 1"})
    assert out["status"] == "partial_failure"
    statuses = [s["status"] for s in out["data"]["statements"]]
    assert statuses == ["succeeded", "failed", "skipped"]
    assert out["data"]["statements"][1]["error"]["message"] == "syntax error"
    assert h.w.statement_execution.execute_statement.call_count == 2
    assert "not rolled back" in out["data"]["note"].lower() or "NOT rolled back" in out["data"]["note"]


async def test_multi_continue_on_error_and_order(h):
    h.w.statement_execution.execute_statement.side_effect = [
        _resp(state="FAILED", error=sql.ServiceError(message="boom")),
        _resp(columns=[("a", "int")], rows=[["7"]]),
    ]
    out = await h.call("execute_sql_multi", {"statements": ["SELECT bad", "SELECT 7 AS a"], "continue_on_error": True})
    assert [s["status"] for s in out["data"]["statements"]] == ["failed", "succeeded"]
    assert out["data"]["statements"][1]["result"]["rows"] == [[7]]
    sent = [c.kwargs["statement"] for c in h.w.statement_execution.execute_statement.call_args_list]
    assert sent == ["SELECT bad", "SELECT 7 AS a"]


async def test_multi_with_destructive_needs_confirm(h):
    out = await h.call("execute_sql_multi", {"statements": ["SELECT 1", "TRUNCATE TABLE t"]})
    assert out["status"] == "confirmation_required"
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_manage_sql_statement_get_and_cancel(h):
    h.w.statement_execution.get_statement.return_value = _resp(columns=[("a", "int")], rows=[["1"]])
    out = await h.call("manage_sql_statement", {"action": "get", "statement_id": "s1"})
    assert out["data"]["result"]["rows"] == [[1]]
    out = await h.call("manage_sql_statement", {"action": "cancel", "statement_id": "s1"})
    h.w.statement_execution.cancel_execution.assert_called_once_with("s1")
    assert out["status"] == "success"


async def test_no_warehouse_available(make_harness):
    h = make_harness(toolsets=TOOLSETS)
    h.w.warehouses.list.return_value = []
    err = await h.call_error("execute_sql", {"statement": "SELECT 1"})
    assert "[NOT_FOUND]" in err and "No SQL warehouses" in err


async def test_stopped_only_warehouse_warns(make_harness):
    h = make_harness(toolsets=TOOLSETS)
    h.w.warehouses.list.return_value = [_wh("s", "only", "STOPPED")]
    h.w.statement_execution.execute_statement.return_value = _resp(columns=[("a", "int")], rows=[["1"]])
    out = await h.call("execute_sql", {"statement": "SELECT 1 AS a"})
    assert any("will start it" in w for w in out["warnings"])


# ---------------------------------------------------------------------------------------------- table stats


def _table():
    return catalog.TableInfo(
        catalog_name="main", schema_name="sales", name="orders", full_name="main.sales.orders",
        table_type=catalog.TableType.MANAGED, data_source_format=catalog.DataSourceFormat.DELTA,
        owner="data-eng", storage_location="s3://bucket/orders", comment="Orders",
        columns=[
            catalog.ColumnInfo(name="id", type_text="bigint", nullable=False, position=0),
            catalog.ColumnInfo(name="day", type_text="date", nullable=True, position=1, partition_index=0,
                               comment="partition"),
        ],
    )


async def test_table_schema_without_stats(h):
    h.w.tables.get.return_value = _table()
    out = await h.call("get_table_stats_and_schema", {"name": "main.sales.orders", "stats": "none"})
    data = out["data"]
    assert data["table_type"] == "MANAGED" and data["data_source_format"] == "DELTA"
    assert data["partition_columns"] == ["day"]
    assert data["columns"][0] == {"name": "id", "type": "bigint", "nullable": False, "comment": None,
                                  "position": 0, "partition_index": None, "has_mask": False}
    assert data["statistics"] is None
    h.w.statement_execution.execute_statement.assert_not_called()
    assert out["safety"] == ["READ_ONLY"]


async def test_table_stats_auto_uses_running_warehouse_and_quotes(h):
    h.w.tables.get.return_value = _table()
    h.w.statement_execution.execute_statement.side_effect = [
        _resp(columns=[("format", "string"), ("numFiles", "bigint"), ("sizeInBytes", "bigint"),
                       ("partitionColumns", "array<string>")],
              rows=[["delta", "4", "1024", '["day"]']]),
        _resp(columns=[("row_count", "bigint")], rows=[["99"]]),
    ]
    out = await h.call("get_table_stats_and_schema", {"name": "main.sales.orders", "stats": "exact_count"})
    stats = out["data"]["statistics"]
    assert stats["numFiles"] == 4 and stats["sizeInBytes"] == 1024 and stats["partitionColumns"] == ["day"]
    assert stats["row_count"] == 99
    sent = [c.kwargs["statement"] for c in h.w.statement_execution.execute_statement.call_args_list]
    assert sent == ["DESCRIBE DETAIL `main`.`sales`.`orders`", "SELECT COUNT(*) AS row_count FROM `main`.`sales`.`orders`"]


async def test_table_stats_auto_skips_without_running_warehouse(make_harness):
    h = make_harness(toolsets=TOOLSETS)
    h.w.warehouses.list.return_value = [_wh("s", "stopped", "STOPPED")]
    h.w.tables.get.return_value = _table()
    out = await h.call("get_table_stats_and_schema", {"name": "main.sales.orders"})
    assert out["data"]["statistics"] is None
    assert any("No running SQL warehouse" in w for w in out["warnings"])
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_schema_listing_paginates(h):
    h.w.tables.list.return_value = [catalog.TableInfo(name=f"t{i}", full_name=f"main.s.t{i}") for i in range(5)]
    out = await h.call("get_table_stats_and_schema", {"name": "main.s", "page_size": 2})
    assert [t["name"] for t in out["data"]] == ["t0", "t1"]
    assert out["page"]["has_more"]
    out2 = await h.call("get_table_stats_and_schema", {"name": "main.s", "page_size": 2,
                                                       "page_token": out["page"]["next_page_token"]})
    assert [t["name"] for t in out2["data"]] == ["t2", "t3"]
    h.w.tables.list.assert_called_with(catalog_name="main", schema_name="s")


async def test_table_name_validation(h):
    err = await h.call_error("get_table_stats_and_schema", {"name": "just_one_part"})
    assert "[INVALID_PARAMETER]" in err
