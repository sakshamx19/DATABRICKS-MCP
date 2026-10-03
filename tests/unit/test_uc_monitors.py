"""manage_uc_monitors: data quality monitors via w.data_quality."""

from __future__ import annotations

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service import catalog
from databricks.sdk.service import dataquality as dq
from databricks.sdk.service import sql as sql_svc

TOOL = "manage_uc_monitors"
TABLE = "main.sales.orders"


def _monitor(**cfg):
    base = {
        "output_schema_id": "sid",
        "monitored_table_name": TABLE,
        "profile_metrics_table_name": "main.mon.orders_profile_metrics",
        "drift_metrics_table_name": "main.mon.orders_drift_metrics",
        "dashboard_id": "dash1",
        "status": dq.DataProfilingStatus.DATA_PROFILING_STATUS_ACTIVE,
        "snapshot": dq.SnapshotConfig(),
    }
    base.update(cfg)
    return dq.Monitor(object_type="table", object_id="tid", data_profiling_config=dq.DataProfilingConfig(**base))


@pytest.fixture
def h(make_harness):
    harness = make_harness(toolsets=("unity_catalog",))
    harness.w.tables.get.return_value = catalog.TableInfo(full_name=TABLE, table_id="tid")
    harness.w.schemas.get.return_value = catalog.SchemaInfo(full_name="main.mon", schema_id="sid")
    harness.w.data_quality.get_monitor.return_value = _monitor()
    return harness


async def test_create_resolves_ids_and_builds_monitor(h):
    h.w.data_quality.create_monitor.return_value = _monitor()
    res = await h.call(
        TOOL,
        {
            "action": "create",
            "full_name": TABLE,
            "spec": {
                "output_schema_name": "main.mon",
                "time_series": {"timestamp_column": "ts", "granularities": ["AGGREGATION_GRANULARITY_1_DAY"]},
                "schedule": {"quartz_cron_expression": "0 0 12 * * ?", "timezone_id": "UTC"},
            },
        },
    )
    assert res["status"] == "success"
    h.w.tables.get.assert_called_with(TABLE)
    h.w.schemas.get.assert_called_with("main.mon")
    sent = h.w.data_quality.create_monitor.call_args.args[0]
    assert isinstance(sent, dq.Monitor)
    assert (sent.object_type, sent.object_id) == ("table", "tid")
    cfg = sent.data_profiling_config
    assert cfg.output_schema_id == "sid"
    assert cfg.time_series.timestamp_column == "ts"
    assert cfg.schedule.quartz_cron_expression == "0 0 12 * * ?"


async def test_create_validates_profile_type_and_fields(h):
    msg = await h.call_error(TOOL, {"action": "create", "full_name": TABLE, "spec": {"output_schema_name": "main.mon"}})
    assert "exactly one profile type" in msg
    msg = await h.call_error(
        TOOL,
        {"action": "create", "full_name": TABLE, "spec": {"output_schema_name": "main.mon", "snapshot": {}, "bogus": 1}},
    )
    assert "bogus" in msg
    msg = await h.call_error(
        TOOL,
        {"action": "create", "full_name": TABLE, "spec": {"output_schema_id": "s", "snapshot": {}, "dashboard_id": "x"}},
    )
    assert "Output-only" in msg
    h.w.data_quality.create_monitor.assert_not_called()


async def test_get(h):
    res = await h.call(TOOL, {"action": "get", "full_name": TABLE})
    assert res["data"]["data_profiling_config"]["dashboard_id"] == "dash1"
    h.w.data_quality.get_monitor.assert_called_once_with("table", "tid")


async def test_update_builds_mask_and_preview_diff(h):
    spec = {"slicing_exprs": ["region"]}
    res = await h.call(TOOL, {"action": "update", "full_name": TABLE, "spec": spec, "dry_run": True})
    assert res["status"] == "dry_run"
    assert res["plan"]["details"]["changes"] == {"slicing_exprs": {"current": None, "new": ["region"]}}
    h.w.data_quality.update_monitor.assert_not_called()

    h.w.data_quality.update_monitor.return_value = _monitor(slicing_exprs=["region"])
    res = await h.call(TOOL, {"action": "update", "full_name": TABLE, "spec": spec})
    assert res["status"] == "success"
    args = h.w.data_quality.update_monitor.call_args.args
    assert args[0:2] == ("table", "tid")
    assert args[2].data_profiling_config.slicing_exprs == ["region"]
    assert args[3] == "data_profiling_config.slicing_exprs"


async def test_delete_requires_confirmation(h):
    res = await h.call(TOOL, {"action": "delete", "full_name": TABLE})
    assert res["status"] == "confirmation_required"
    assert any("NOT deleted" in w for w in res["warnings"])
    assert res["plan"]["details"]["outputs"]["profile_metrics_table_name"] == "main.mon.orders_profile_metrics"
    h.w.data_quality.delete_monitor.assert_not_called()

    res = await h.call(TOOL, {"action": "delete", "full_name": TABLE, "confirm": True})
    assert res["status"] == "success"
    h.w.data_quality.delete_monitor.assert_called_once_with("table", "tid")


async def test_read_only_blocks_delete(make_harness):
    h = make_harness(read_only=True, toolsets=("unity_catalog",))
    msg = await h.call_error(TOOL, {"action": "delete", "full_name": TABLE, "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.data_quality.delete_monitor.assert_not_called()


async def test_refresh_and_refresh_history(h):
    h.w.data_quality.create_refresh.return_value = dq.Refresh(
        object_type="table", object_id="tid", refresh_id=7, state=dq.RefreshState.MONITOR_REFRESH_STATE_PENDING
    )
    res = await h.call(TOOL, {"action": "refresh", "full_name": TABLE})
    assert res["status"] == "pending"
    call = h.w.data_quality.create_refresh.call_args
    assert call.args[0:2] == ("table", "tid")
    assert isinstance(call.args[2], dq.Refresh)

    h.w.data_quality.list_refresh.return_value = iter(
        [dq.Refresh(object_type="table", object_id="tid", refresh_id=i, state=dq.RefreshState.MONITOR_REFRESH_STATE_SUCCESS) for i in range(3)]
    )
    res = await h.call(TOOL, {"action": "list_refreshes", "full_name": TABLE})
    assert [r["refresh_id"] for r in res["data"]] == [0, 1, 2]

    h.w.data_quality.get_refresh.return_value = dq.Refresh(
        object_type="table", object_id="tid", refresh_id=7, state=dq.RefreshState.MONITOR_REFRESH_STATE_RUNNING
    )
    res = await h.call(TOOL, {"action": "get_refresh", "full_name": TABLE, "refresh_id": 7})
    assert "MONITOR_REFRESH_STATE_RUNNING" in res["summary"]
    h.w.data_quality.get_refresh.assert_called_once_with("table", "tid", 7)

    msg = await h.call_error(TOOL, {"action": "get_refresh", "full_name": TABLE})
    assert "refresh_id" in msg


async def test_metrics_returns_table_names_and_quoted_queries(h):
    res = await h.call(TOOL, {"action": "metrics", "full_name": TABLE})
    data = res["data"]
    assert data["profile_metrics_table_name"] == "main.mon.orders_profile_metrics"
    assert data["drift_metrics_table_name"] == "main.mon.orders_drift_metrics"
    assert data["dashboard_id"] == "dash1"
    assert data["example_queries"]["profile"] == "SELECT * FROM `main`.`mon`.`orders_profile_metrics` LIMIT 100"


async def test_query_metrics_runs_limited_select(h):
    h.w.warehouses.list.return_value = [sql_svc.EndpointInfo(id="wh1", name="wh", state=sql_svc.State.RUNNING)]
    h.w.statement_execution.execute_statement.return_value = sql_svc.StatementResponse(
        statement_id="s1",
        status=sql_svc.StatementStatus(state=sql_svc.StatementState.SUCCEEDED),
        manifest=sql_svc.ResultManifest(
            schema=sql_svc.ResultSchema(columns=[sql_svc.ColumnInfo(name="column_name", type_text="string", position=0)]),
            total_row_count=1,
        ),
        result=sql_svc.ResultData(data_array=[["region"]]),
    )
    res = await h.call(TOOL, {"action": "query_metrics", "full_name": TABLE, "metrics_table": "drift", "sample_rows": 5})
    stmt = h.w.statement_execution.execute_statement.call_args.kwargs["statement"]
    assert stmt == "SELECT * FROM `main`.`mon`.`orders_drift_metrics` LIMIT 5"
    assert res["data"]["rows"] == [{"column_name": "region"}]


async def test_schema_monitor_and_table_only_actions(h):
    h.w.data_quality.create_monitor.return_value = dq.Monitor(object_type="schema", object_id="sid")
    await h.call(
        TOOL,
        {"action": "create", "object_type": "schema", "full_name": "main.sales", "spec": {"excluded_table_full_names": ["main.sales.tmp"]}},
    )
    sent = h.w.data_quality.create_monitor.call_args.args[0]
    assert (sent.object_type, sent.object_id) == ("schema", "sid")
    assert sent.anomaly_detection_config.excluded_table_full_names == ["main.sales.tmp"]

    msg = await h.call_error(TOOL, {"action": "refresh", "object_type": "schema", "full_name": "main.sales"})
    assert "only supported for table monitors" in msg


async def test_not_found(h):
    h.w.tables.get.side_effect = NotFound("Table does not exist")
    msg = await h.call_error(TOOL, {"action": "get", "full_name": TABLE})
    assert "[NOT_FOUND]" in msg
