"""manage_uc_security_policies: row filters, column masks and ABAC policies."""

from __future__ import annotations

import pytest
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service import catalog, iam
from databricks.sdk.service import sql as sql_svc

TOOL = "manage_uc_security_policies"
TABLE = "main.sales.orders"
QT = "`main`.`sales`.`orders`"


@pytest.fixture
def h(make_harness):
    harness = make_harness(toolsets=("unity_catalog",))
    harness.w.current_user.me.return_value = iam.User(user_name="alice@example.com")
    harness.w.warehouses.list.return_value = [
        sql_svc.EndpointInfo(id="wh1", name="shared", state=sql_svc.State.RUNNING)
    ]
    harness.w.statement_execution.execute_statement.return_value = sql_svc.StatementResponse(
        statement_id="st1", status=sql_svc.StatementStatus(state=sql_svc.StatementState.SUCCEEDED)
    )
    harness.w.tables.get.return_value = _table()
    harness.w.policies.list_policies.return_value = iter([])
    return harness


def _table(row_filter=None, masks=None):
    masks = masks or {}
    return catalog.TableInfo(
        full_name=TABLE,
        row_filter=row_filter,
        columns=[
            catalog.ColumnInfo(name=n, type_text="string", mask=masks.get(n))
            for n in ("id", "region", "email", "c` ; DROP")
        ],
    )


def executed(h):
    return [c.kwargs["statement"] for c in h.w.statement_execution.execute_statement.call_args_list]


async def test_get_reports_filters_masks_and_policies(h):
    h.w.tables.get.return_value = _table(
        row_filter=catalog.TableRowFilter(function_name="main.sec.region_filter", input_column_names=["region"]),
        masks={"email": catalog.ColumnMask(function_name="main.sec.mask_email", using_column_names=["region"])},
    )
    h.w.policies.list_policies.return_value = iter(
        [
            catalog.PolicyInfo(
                name="pii",
                to_principals=["analysts"],
                for_securable_type=catalog.SecurableType.TABLE,
                policy_type=catalog.PolicyType.POLICY_TYPE_COLUMN_MASK,
                on_securable_type=catalog.SecurableType.SCHEMA,
                on_securable_fullname="main.sales",
            )
        ]
    )
    res = await h.call(TOOL, {"action": "get", "table_name": TABLE})
    assert res["status"] == "success"
    data = res["data"]
    assert data["row_filter"]["function_name"] == "main.sec.region_filter"
    assert data["column_masks"] == [
        {"column": "email", "mask": {"function_name": "main.sec.mask_email", "using_column_names": ["region"]}}
    ]
    assert data["abac_policies"][0]["name"] == "pii"
    h.w.policies.list_policies.assert_called_once_with("TABLE", TABLE, include_inherited=True)


async def test_get_tolerates_policy_api_failure(h):
    h.w.policies.list_policies.side_effect = PermissionDenied("no MANAGE")
    res = await h.call(TOOL, {"action": "get", "table_name": TABLE})
    assert res["data"]["abac_policies"] is None
    assert any("ABAC" in w for w in res["warnings"])


async def test_set_row_filter_requires_confirmation_and_shows_diff(h):
    h.w.tables.get.return_value = _table(
        row_filter=catalog.TableRowFilter(function_name="main.sec.old", input_column_names=["id"])
    )
    args = {
        "action": "set_row_filter",
        "table_name": TABLE,
        "function_name": "main.sec.region_filter",
        "using_columns": ["region"],
    }
    res = await h.call(TOOL, args)
    assert res["status"] == "confirmation_required"
    details = res["plan"]["details"]
    assert details["current_row_filter"]["function_name"] == "main.sec.old"
    assert details["new_row_filter"] == {"function_name": "main.sec.region_filter", "input_column_names": ["region"]}
    assert details["statement"] == f"ALTER TABLE {QT} SET ROW FILTER `main`.`sec`.`region_filter` ON (`region`)"
    assert "SECURITY_SENSITIVE" in res["safety"]
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_set_row_filter_confirmed_executes_with_audit(h):
    args = {
        "action": "set_row_filter",
        "table_name": TABLE,
        "function_name": "main.sec.region_filter",
        "using_columns": ["region", "id"],
        "confirm": True,
    }
    res = await h.call(TOOL, args)
    assert res["status"] == "success"
    sql = f"ALTER TABLE {QT} SET ROW FILTER `main`.`sec`.`region_filter` ON (`region`, `id`)"
    assert executed(h) == [sql]
    audit = res["data"]["audit"]
    assert audit["who"] == "alice@example.com"
    assert audit["what"] == sql
    assert audit["when"] and audit["warehouse_id"] == "wh1"
    assert res["data"]["previous"]["current_row_filter"] is None


async def test_set_row_filter_requires_using_columns(h):
    msg = await h.call_error(
        TOOL,
        {"action": "set_row_filter", "table_name": TABLE, "function_name": "main.sec.f", "confirm": True},
    )
    assert "using_columns" in msg
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_dry_run_executes_nothing(h):
    res = await h.call(
        TOOL, {"action": "drop_row_filter", "table_name": TABLE, "dry_run": True, "confirm": True}
    )
    assert res["status"] == "dry_run"
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_drop_row_filter_preview_warns_and_has_restore_statement(h):
    h.w.tables.get.return_value = _table(
        row_filter=catalog.TableRowFilter(function_name="main.sec.region_filter", input_column_names=["region"])
    )
    res = await h.call(TOOL, {"action": "drop_row_filter", "table_name": TABLE})
    assert res["status"] == "confirmation_required"
    assert "DESTRUCTIVE" in res["safety"]
    assert any("BROADENS" in w for w in res["warnings"])
    assert res["plan"]["details"]["restore_statement"] == (
        f"ALTER TABLE {QT} SET ROW FILTER `main`.`sec`.`region_filter` ON (`region`)"
    )
    res = await h.call(TOOL, {"action": "drop_row_filter", "table_name": TABLE, "confirm": True})
    assert executed(h) == [f"ALTER TABLE {QT} DROP ROW FILTER"]


async def test_set_and_drop_column_mask(h):
    res = await h.call(
        TOOL,
        {
            "action": "set_column_mask",
            "table_name": TABLE,
            "column_name": "email",
            "function_name": "main.sec.mask_email",
            "using_columns": ["region"],
        },
    )
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["current_mask"] is None
    assert res["plan"]["details"]["new_mask"]["function_name"] == "main.sec.mask_email"

    await h.call(
        TOOL,
        {
            "action": "set_column_mask",
            "table_name": TABLE,
            "column_name": "email",
            "function_name": "main.sec.mask_email",
            "using_columns": ["region"],
            "confirm": True,
        },
    )
    await h.call(TOOL, {"action": "drop_column_mask", "table_name": TABLE, "column_name": "email", "confirm": True})
    assert executed(h) == [
        f"ALTER TABLE {QT} ALTER COLUMN `email` SET MASK `main`.`sec`.`mask_email` USING COLUMNS (`region`)",
        f"ALTER TABLE {QT} ALTER COLUMN `email` DROP MASK",
    ]


async def test_column_mask_unknown_column_rejected(h):
    msg = await h.call_error(
        TOOL,
        {"action": "drop_column_mask", "table_name": TABLE, "column_name": "nope", "confirm": True},
    )
    assert "[INVALID_PARAMETER]" in msg and "nope" in msg
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_injection_attempts_are_quoted_or_rejected(h):
    # Backticks in a column name are escaped, never closing the identifier.
    await h.call(
        TOOL,
        {
            "action": "drop_column_mask",
            "table_name": TABLE,
            "column_name": "c` ; DROP",
            "confirm": True,
        },
    )
    assert executed(h) == [f"ALTER TABLE {QT} ALTER COLUMN `c`` ; DROP` DROP MASK"]

    # An unbalanced backtick in a table name is rejected before anything runs.
    msg = await h.call_error(
        TOOL, {"action": "drop_row_filter", "table_name": "main.sales.orders`; DROP TABLE x; --", "confirm": True}
    )
    assert "[INVALID_PARAMETER]" in msg
    # Unqualified function names are rejected (would resolve against session defaults).
    msg = await h.call_error(
        TOOL,
        {"action": "set_row_filter", "table_name": TABLE, "function_name": "f", "using_columns": [], "confirm": True},
    )
    assert "[INVALID_PARAMETER]" in msg
    assert len(executed(h)) == 1


async def test_read_only_mode_blocks_changes(make_harness):
    h = make_harness(read_only=True, toolsets=("unity_catalog",))
    msg = await h.call_error(TOOL, {"action": "drop_row_filter", "table_name": TABLE, "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_not_found_mapping(h):
    h.w.tables.get.side_effect = NotFound("Table main.sales.orders does not exist")
    msg = await h.call_error(TOOL, {"action": "get", "table_name": TABLE})
    assert "[NOT_FOUND]" in msg


async def test_list_policies(h):
    h.w.policies.list_policies.return_value = iter(
        [
            catalog.PolicyInfo(
                name=f"p{i}",
                to_principals=["g"],
                for_securable_type=catalog.SecurableType.TABLE,
                policy_type=catalog.PolicyType.POLICY_TYPE_ROW_FILTER,
            )
            for i in range(3)
        ]
    )
    res = await h.call(
        TOOL, {"action": "list_policies", "securable_type": "SCHEMA", "securable_fullname": "main.sales"}
    )
    assert [p["name"] for p in res["data"]] == ["p0", "p1", "p2"]
    h.w.policies.list_policies.assert_called_once_with("SCHEMA", "main.sales", include_inherited=None)


async def test_create_policy_flow(h):
    spec = {
        "to_principals": ["analysts"],
        "for_securable_type": "TABLE",
        "policy_type": "POLICY_TYPE_ROW_FILTER",
        "row_filter": {"function_name": "main.sec.region_filter"},
        "when_condition": "hasTag('pii')",
    }
    args = {
        "action": "create_policy",
        "securable_type": "CATALOG",
        "securable_fullname": "main",
        "policy_name": "region_rf",
        "spec": spec,
    }
    res = await h.call(TOOL, args)
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["new_policy"]["row_filter"]["function_name"] == "main.sec.region_filter"
    h.w.policies.create_policy.assert_not_called()

    h.w.policies.create_policy.return_value = catalog.PolicyInfo(
        name="region_rf",
        to_principals=["analysts"],
        for_securable_type=catalog.SecurableType.TABLE,
        policy_type=catalog.PolicyType.POLICY_TYPE_ROW_FILTER,
    )
    res = await h.call(TOOL, {**args, "confirm": True})
    sent = h.w.policies.create_policy.call_args.args[0]
    assert isinstance(sent, catalog.PolicyInfo)
    assert sent.name == "region_rf"
    assert sent.on_securable_type == catalog.SecurableType.CATALOG
    assert sent.on_securable_fullname == "main"
    assert sent.policy_type == catalog.PolicyType.POLICY_TYPE_ROW_FILTER
    assert sent.row_filter.function_name == "main.sec.region_filter"
    assert res["data"]["audit"]["who"] == "alice@example.com"


async def test_create_policy_rejects_identity_fields_and_unknown_fields(h):
    base = {"action": "create_policy", "securable_type": "CATALOG", "securable_fullname": "main", "policy_name": "x"}
    msg = await h.call_error(TOOL, {**base, "spec": {"name": "other", "to_principals": ["a"]}, "confirm": True})
    assert "[INVALID_PARAMETER]" in msg
    msg = await h.call_error(
        TOOL,
        {
            **base,
            "spec": {
                "to_principals": ["a"],
                "for_securable_type": "TABLE",
                "policy_type": "POLICY_TYPE_ROW_FILTER",
                "bogus": 1,
            },
            "confirm": True,
        },
    )
    assert "bogus" in msg
    h.w.policies.create_policy.assert_not_called()


async def test_update_and_delete_policy(h):
    current = catalog.PolicyInfo(
        name="pii",
        to_principals=["analysts"],
        for_securable_type=catalog.SecurableType.TABLE,
        policy_type=catalog.PolicyType.POLICY_TYPE_COLUMN_MASK,
        comment="old",
    )
    h.w.policies.get_policy.return_value = current
    base = {"securable_type": "SCHEMA", "securable_fullname": "main.sales", "policy_name": "pii"}

    res = await h.call(TOOL, {**base, "action": "update_policy", "spec": {"comment": "new"}})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["changes"] == {"comment": {"current": "old", "new": "new"}}

    h.w.policies.update_policy.return_value = current
    await h.call(TOOL, {**base, "action": "update_policy", "spec": {"comment": "new"}, "confirm": True})
    call = h.w.policies.update_policy.call_args
    assert call.args[:3] == ("SCHEMA", "main.sales", "pii")
    assert call.args[3].comment == "new"
    assert call.kwargs["update_mask"] == "comment"

    res = await h.call(TOOL, {**base, "action": "delete_policy"})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["reversible"] is False
    assert any("BROADENS" in w for w in res["warnings"])
    h.w.policies.delete_policy.assert_not_called()
    res = await h.call(TOOL, {**base, "action": "delete_policy", "confirm": True})
    h.w.policies.delete_policy.assert_called_once_with("SCHEMA", "main.sales", "pii")
    assert res["data"]["deleted_policy"]["comment"] == "old"


async def test_protected_table_blocks_drop(make_harness):
    import re

    h = make_harness(toolsets=("unity_catalog",), protected_name_patterns=(re.compile("prod"),))
    msg = await h.call_error(
        TOOL, {"action": "drop_row_filter", "table_name": "prod.sales.orders", "confirm": True}
    )
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.statement_execution.execute_statement.assert_not_called()
