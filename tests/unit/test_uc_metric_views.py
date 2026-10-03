"""manage_metric_views: SQL-based metric view management."""

from __future__ import annotations

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service import catalog, iam
from databricks.sdk.service import sql as sql_svc

TOOL = "manage_metric_views"
VIEW = "main.sales.orders_mv"
QV = "`main`.`sales`.`orders_mv`"
OLD = "version: 0.1\nsource: main.sales.orders\ndimensions:\n  - name: region\n    expr: region\nmeasures:\n  - name: revenue\n    expr: SUM(amount)"
NEW = OLD + "\n  - name: orders\n    expr: COUNT(1)"


@pytest.fixture
def h(make_harness):
    harness = make_harness(toolsets=("unity_catalog",))
    harness.w.current_user.me.return_value = iam.User(user_name="alice@example.com")
    harness.w.warehouses.list.return_value = [sql_svc.EndpointInfo(id="wh1", name="wh", state=sql_svc.State.RUNNING)]
    harness.w.statement_execution.execute_statement.return_value = sql_svc.StatementResponse(
        statement_id="st1", status=sql_svc.StatementStatus(state=sql_svc.StatementState.SUCCEEDED)
    )
    harness.w.tables.get.return_value = catalog.TableInfo(
        full_name=VIEW,
        name="orders_mv",
        table_type=catalog.TableType.METRIC_VIEW,
        view_definition=OLD,
        owner="alice@example.com",
        columns=[catalog.ColumnInfo(name="region", type_text="string"), catalog.ColumnInfo(name="revenue", type_text="double")],
    )
    return harness


def executed(h):
    return [c.kwargs["statement"] for c in h.w.statement_execution.execute_statement.call_args_list]


async def test_create_runs_documented_ddl(h):
    res = await h.call(TOOL, {"action": "create", "full_name": VIEW, "yaml_definition": OLD})
    assert res["status"] == "success"
    sql = f"CREATE VIEW {QV}\nWITH METRICS\nLANGUAGE YAML\nAS $$\n{OLD}\n$$"
    assert executed(h) == [sql]
    assert res["data"]["audit"]["what"] == sql
    assert res["data"]["audit"]["who"] == "alice@example.com"


async def test_create_dry_run(h):
    res = await h.call(TOOL, {"action": "create", "full_name": VIEW, "yaml_definition": OLD, "dry_run": True})
    assert res["status"] == "dry_run"
    assert res["plan"]["details"]["statement"].startswith(f"CREATE VIEW {QV}")
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_yaml_validation(h):
    msg = await h.call_error(TOOL, {"action": "create", "full_name": VIEW, "yaml_definition": "a: 1\n$$; DROP TABLE x; $$"})
    assert "$$" in msg and "[INVALID_PARAMETER]" in msg
    msg = await h.call_error(TOOL, {"action": "create", "full_name": VIEW, "yaml_definition": "   "})
    assert "[INVALID_PARAMETER]" in msg
    msg = await h.call_error(TOOL, {"action": "create", "full_name": VIEW})
    assert "yaml_definition" in msg
    msg = await h.call_error(TOOL, {"action": "create", "full_name": "orders_mv", "yaml_definition": OLD})
    assert "[INVALID_PARAMETER]" in msg
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_get_returns_definition_and_metadata(h):
    res = await h.call(TOOL, {"action": "get", "full_name": VIEW})
    data = res["data"]
    assert data["definition_yaml"] == OLD
    assert data["table_type"] == "METRIC_VIEW"
    assert [c["name"] for c in data["columns"]] == ["region", "revenue"]


async def test_get_rejects_non_metric_view(h):
    h.w.tables.get.return_value = catalog.TableInfo(full_name=VIEW, table_type=catalog.TableType.VIEW)
    msg = await h.call_error(TOOL, {"action": "get", "full_name": VIEW})
    assert "not a metric view" in msg


async def test_list_filters_metric_views(h):
    h.w.tables.list.return_value = iter(
        [
            catalog.TableInfo(full_name="main.sales.orders", name="orders", table_type=catalog.TableType.MANAGED),
            catalog.TableInfo(full_name=VIEW, name="orders_mv", table_type=catalog.TableType.METRIC_VIEW),
            catalog.TableInfo(full_name="main.sales.v", name="v", table_type=catalog.TableType.VIEW),
        ]
    )
    res = await h.call(TOOL, {"action": "list", "catalog_name": "main", "schema_name": "sales"})
    assert res["data"] == [{"full_name": VIEW, "name": "orders_mv"}]
    h.w.tables.list.assert_called_once_with("main", "sales", omit_columns=True, omit_properties=True)


async def test_update_requires_confirmation_and_shows_diff(h):
    args = {"action": "update", "full_name": VIEW, "yaml_definition": NEW}
    res = await h.call(TOOL, args)
    assert res["status"] == "confirmation_required"
    assert set(res["safety"]) == {"DESTRUCTIVE", "WRITE"}
    details = res["plan"]["details"]
    assert details["current_definition"] == OLD
    assert details["new_definition"] == NEW
    assert "+  - name: orders" in details["diff"]
    h.w.statement_execution.execute_statement.assert_not_called()

    res = await h.call(TOOL, {**args, "confirm": True})
    assert executed(h) == [f"CREATE OR REPLACE VIEW {QV}\nWITH METRICS\nLANGUAGE YAML\nAS $$\n{NEW}\n$$"]
    assert res["data"]["previous_definition"] == OLD


async def test_update_refuses_regular_view(h):
    h.w.tables.get.return_value = catalog.TableInfo(full_name=VIEW, table_type=catalog.TableType.VIEW, view_definition="SELECT 1")
    msg = await h.call_error(TOOL, {"action": "update", "full_name": VIEW, "yaml_definition": NEW, "confirm": True})
    assert "not a metric view" in msg
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_delete(h):
    res = await h.call(TOOL, {"action": "delete", "full_name": VIEW})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["current_definition"] == OLD
    h.w.statement_execution.execute_statement.assert_not_called()
    await h.call(TOOL, {"action": "delete", "full_name": VIEW, "confirm": True})
    assert executed(h) == [f"DROP VIEW {QV}"]


async def test_read_only_blocks_writes_but_allows_query(make_harness):
    h = make_harness(read_only=True, toolsets=("unity_catalog",))
    msg = await h.call_error(TOOL, {"action": "create", "full_name": VIEW, "yaml_definition": OLD})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_query_builds_quoted_sql_with_bound_parameters(h):
    h.w.statement_execution.execute_statement.return_value = sql_svc.StatementResponse(
        statement_id="q1",
        status=sql_svc.StatementStatus(state=sql_svc.StatementState.SUCCEEDED),
        manifest=sql_svc.ResultManifest(
            schema=sql_svc.ResultSchema(
                columns=[
                    sql_svc.ColumnInfo(name="region", type_text="string", position=0),
                    sql_svc.ColumnInfo(name="revenue", type_text="double", position=1),
                ]
            ),
            total_row_count=1,
        ),
        result=sql_svc.ResultData(data_array=[["EU", "10.5"]]),
    )
    res = await h.call(
        TOOL,
        {
            "action": "query",
            "full_name": VIEW,
            "dimensions": ["region"],
            "measures": ["revenue", "x` FROM y; --"],
            "filters": [{"dimension": "region", "op": "=", "value": "EU' OR 1=1 --"}, {"dimension": "region", "op": "is not null"}],
            "limit": 10,
        },
    )
    call = h.w.statement_execution.execute_statement.call_args.kwargs
    assert call["statement"] == (
        "SELECT `region`, MEASURE(`revenue`) AS `revenue`, MEASURE(`x`` FROM y; --`) AS `x`` FROM y; --`\n"
        f"FROM {QV}\n"
        "WHERE `region` = :p0 AND `region` IS NOT NULL\n"
        "GROUP BY `region`\n"
        "LIMIT 10"
    )
    params = call["parameters"]
    assert [(p.name, p.value) for p in params] == [("p0", "EU' OR 1=1 --")]
    assert res["data"]["rows"] == [{"region": "EU", "revenue": "10.5"}]


async def test_query_rejects_bad_operator(h):
    msg = await h.call_error(
        TOOL,
        {"action": "query", "full_name": VIEW, "measures": ["revenue"], "filters": [{"dimension": "region", "op": "; DROP", "value": 1}]},
    )
    assert "[INVALID_PARAMETER]" in msg
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_not_found(h):
    h.w.tables.get.side_effect = NotFound("Table 'main.sales.orders_mv' does not exist")
    msg = await h.call_error(TOOL, {"action": "get", "full_name": VIEW})
    assert "[NOT_FOUND]" in msg
