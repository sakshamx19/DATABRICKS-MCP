"""manage_uc_tags: entity tag assignments + comments (SDK and quoted SQL)."""

from __future__ import annotations

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import CatalogInfo, ColumnInfo, EntityTagAssignment, TableInfo
from databricks.sdk.service.sql import EndpointInfo, State, StatementResponse, StatementState, StatementStatus

TOOL = "manage_uc_tags"


@pytest.fixture
def h(make_harness):
    harness = make_harness(toolsets=("unity_catalog",))
    harness.w.warehouses.list.return_value = [EndpointInfo(id="wh1", name="wh", state=State.RUNNING)]
    harness.w.statement_execution.execute_statement.return_value = StatementResponse(
        statement_id="st1", status=StatementStatus(state=StatementState.SUCCEEDED)
    )
    return harness


def _tag(key: str, value: str | None, etype: str = "tables", name: str = "main.s.t") -> EntityTagAssignment:
    return EntityTagAssignment(entity_name=name, tag_key=key, entity_type=etype, tag_value=value)


def _sql(h) -> str:
    return h.w.statement_execution.execute_statement.call_args.kwargs["statement"]


async def test_get_tags_and_comment(h):
    h.w.entity_tag_assignments.list.return_value = iter([_tag("pii", "email"), _tag("team", "sales")])
    h.w.tables.get.return_value = TableInfo(full_name="main.s.t", comment="orders")
    res = await h.call(TOOL, {"action": "get", "entity_type": "table", "full_name": "main.s.t"})
    assert res["data"]["comment"] == "orders"
    assert res["data"]["tags"] == [{"tag_key": "pii", "tag_value": "email"}, {"tag_key": "team", "tag_value": "sales"}]
    h.w.entity_tag_assignments.list.assert_called_once_with(entity_type="tables", entity_name="main.s.t")


async def test_get_column_tags(h):
    h.w.entity_tag_assignments.list.return_value = iter([_tag("pii", "ssn", "columns", "main.s.t.ssn")])
    h.w.tables.get.return_value = TableInfo(full_name="main.s.t", columns=[ColumnInfo(name="ssn", comment="social")])
    res = await h.call(TOOL, {"action": "get", "entity_type": "column", "full_name": "main.s.t.ssn"})
    assert res["data"]["comment"] == "social"
    h.w.entity_tag_assignments.list.assert_called_once_with(entity_type="columns", entity_name="main.s.t.ssn")


async def test_column_requires_four_parts(h):
    msg = await h.call_error(TOOL, {"action": "get", "entity_type": "column", "full_name": "main.s.t"})
    assert "4-part" in msg


async def test_get_not_found(h):
    h.w.entity_tag_assignments.list.side_effect = NotFound("Table 'main.s.x' does not exist.")
    msg = await h.call_error(TOOL, {"action": "get", "entity_type": "table", "full_name": "main.s.x"})
    assert "[NOT_FOUND]" in msg


async def test_add_tags(h):
    h.w.entity_tag_assignments.create.side_effect = lambda tag_assignment: tag_assignment
    res = await h.call(TOOL, {"action": "add", "entity_type": "schema", "full_name": "main.s", "tags": {"pii": "none", "certified": None}})
    assert res["status"] == "success"
    assert res["data"]["added"] == ["pii", "certified"]
    calls = [c.kwargs["tag_assignment"] for c in h.w.entity_tag_assignments.create.call_args_list]
    assert calls[0] == EntityTagAssignment(entity_name="main.s", tag_key="pii", entity_type="schemas", tag_value="none")
    assert calls[1].tag_value is None


async def test_update_tags_dry_run_shows_diff(h):
    h.w.entity_tag_assignments.list.return_value = iter([_tag("pii", "none")])
    res = await h.call(
        TOOL, {"action": "update", "entity_type": "table", "full_name": "main.s.t", "tags": {"pii": "email"}, "dry_run": True}
    )
    assert res["status"] == "dry_run"
    assert res["plan"]["details"]["changes"]["pii"] == {"before": "none", "after": "email"}
    h.w.entity_tag_assignments.update.assert_not_called()


async def test_update_tags_calls_update_with_mask(h):
    res = await h.call(TOOL, {"action": "update", "entity_type": "volume", "full_name": "main.s.v", "tags": {"pii": "email"}})
    assert res["status"] == "success"
    h.w.entity_tag_assignments.update.assert_called_once_with(
        entity_type="volumes",
        entity_name="main.s.v",
        tag_key="pii",
        tag_assignment=EntityTagAssignment(entity_name="main.s.v", tag_key="pii", entity_type="volumes", tag_value="email"),
        update_mask="tag_value",
    )


async def test_partial_failure_reported(h):
    def create(tag_assignment):
        if tag_assignment.tag_key == "bad":
            raise NotFound("Tag policy not found")
        return tag_assignment

    h.w.entity_tag_assignments.create.side_effect = create
    res = await h.call(TOOL, {"action": "add", "entity_type": "table", "full_name": "main.s.t", "tags": {"ok": "1", "bad": "2"}})
    assert res["status"] == "partial_failure"
    assert res["data"]["added"] == ["ok"]
    assert res["data"]["failed"][0]["tag_key"] == "bad"


async def test_remove_requires_confirmation(h):
    h.w.entity_tag_assignments.list.side_effect = lambda **kw: iter([_tag("pii", "email"), _tag("team", "x")])
    res = await h.call(TOOL, {"action": "remove", "entity_type": "table", "full_name": "main.s.t", "tag_keys": ["pii"]})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["removing"] == {"pii": "email"}
    assert res["plan"]["details"]["tags_after"] == {"team": "x"}
    h.w.entity_tag_assignments.delete.assert_not_called()

    res = await h.call(
        TOOL, {"action": "remove", "entity_type": "table", "full_name": "main.s.t", "tag_keys": ["pii"], "confirm": True}
    )
    assert res["status"] == "success"
    h.w.entity_tag_assignments.delete.assert_called_once_with(entity_type="tables", entity_name="main.s.t", tag_key="pii")


async def test_remove_blocked_on_prod_tagged_entity(h):
    h.w.entity_tag_assignments.list.side_effect = lambda **kw: iter([_tag("env", "prod")])
    msg = await h.call_error(
        TOOL, {"action": "remove", "entity_type": "table", "full_name": "main.s.t", "tag_keys": ["env"], "confirm": True}
    )
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.entity_tag_assignments.delete.assert_not_called()


async def test_set_comment_catalog_uses_sdk(h):
    h.w.catalogs.update.return_value = CatalogInfo(name="main", comment="hi")
    res = await h.call(TOOL, {"action": "set_comment", "entity_type": "catalog", "full_name": "main", "comment": "hi"})
    assert res["status"] == "success"
    h.w.catalogs.update.assert_called_once_with("main", comment="hi")
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_set_comment_table_sql_is_quoted(h):
    res = await h.call(
        TOOL,
        {"action": "set_comment", "entity_type": "table", "full_name": "main.s.orders",
         "comment": "it's x'; DROP TABLE y; --", "warehouse_id": "wh9"},
    )
    assert res["status"] == "success"
    assert _sql(h) == r"COMMENT ON TABLE `main`.`s`.`orders` IS 'it\'s x\'; DROP TABLE y; --'"
    assert h.w.statement_execution.execute_statement.call_args.kwargs["warehouse_id"] == "wh9"


async def test_set_comment_column_injection_in_identifiers(h):
    await h.call(
        TOOL,
        {"action": "set_comment", "entity_type": "column", "full_name": "main.s.`t``; DROP TABLE z; --`.`c`",
         "comment": "back\\slash"},
    )
    assert _sql(h) == (
        "ALTER TABLE `main`.`s`.`t``; DROP TABLE z; --` ALTER COLUMN `c` COMMENT 'back\\\\slash'"
    )
    # warehouse auto-selected from the running one
    assert h.w.statement_execution.execute_statement.call_args.kwargs["warehouse_id"] == "wh1"


async def test_set_comment_table_clear(h):
    await h.call(TOOL, {"action": "set_comment", "entity_type": "table", "full_name": "main.s.t", "comment": ""})
    assert _sql(h) == "COMMENT ON TABLE `main`.`s`.`t` IS NULL"


async def test_set_comment_dry_run_shows_sql(h):
    h.w.tables.get.return_value = TableInfo(full_name="main.s.t", comment="old")
    res = await h.call(
        TOOL, {"action": "set_comment", "entity_type": "table", "full_name": "main.s.t", "comment": "new", "dry_run": True}
    )
    assert res["status"] == "dry_run"
    assert res["plan"]["details"]["before"] == "old"
    assert res["plan"]["details"]["sql"] == "COMMENT ON TABLE `main`.`s`.`t` IS 'new'"
    h.w.statement_execution.execute_statement.assert_not_called()


async def test_read_only_blocks_tag_changes(make_harness):
    h = make_harness(read_only=True, toolsets=("unity_catalog",))
    msg = await h.call_error(TOOL, {"action": "add", "entity_type": "table", "full_name": "a.b.c", "tags": {"k": "v"}})
    assert "read-only" in msg
    h.w.entity_tag_assignments.create.assert_not_called()
