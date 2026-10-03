"""manage_uc_objects: catalogs, schemas, tables, volumes, functions."""

from __future__ import annotations

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    CatalogInfo,
    FunctionInfo,
    SchemaInfo,
    TableInfo,
    TableType,
    VolumeInfo,
    VolumeType,
)

TOOL = "manage_uc_objects"


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=("unity_catalog",))


async def test_tools_registered(h):
    assert {"manage_uc_objects", "manage_uc_grants", "manage_uc_storage", "manage_uc_connections", "manage_uc_tags"} <= h.tool_names()


# --- hierarchy ---------------------------------------------------------------------------------

async def test_list_schemas_requires_catalog(h):
    msg = await h.call_error(TOOL, {"action": "list", "object_type": "schema"})
    assert "catalog_name" in msg
    h.w.schemas.list.assert_not_called()


async def test_list_tables_requires_catalog_and_schema(h):
    msg = await h.call_error(TOOL, {"action": "list", "object_type": "table", "catalog_name": "main"})
    assert "schema_name" in msg
    h.w.tables.list.assert_not_called()


async def test_list_tables_uses_parent_and_omits_columns(h):
    h.w.tables.list.return_value = iter([TableInfo(full_name="main.s.t", name="t", table_type=TableType.MANAGED, columns=[])])
    res = await h.call(TOOL, {"action": "list", "object_type": "table", "catalog_name": "main", "schema_name": "s"})
    assert res["data"] == [{"full_name": "main.s.t", "name": "t", "table_type": "MANAGED"}]
    h.w.tables.list.assert_called_once_with(catalog_name="main", schema_name="s", omit_columns=True, omit_properties=True)


async def test_list_volumes_accepts_parent_full_name(h):
    h.w.volumes.list.return_value = iter([VolumeInfo(full_name="main.s.v", name="v", volume_type=VolumeType.MANAGED)])
    res = await h.call(TOOL, {"action": "list", "object_type": "volume", "full_name": "main.s"})
    assert res["data"][0]["full_name"] == "main.s.v"
    h.w.volumes.list.assert_called_once_with(catalog_name="main", schema_name="s")


async def test_table_get_requires_three_part_name(h):
    msg = await h.call_error(TOOL, {"action": "get", "object_type": "table", "full_name": "main.s"})
    assert "INVALID_PARAMETER" in msg
    h.w.tables.get.assert_not_called()


async def test_conflicting_catalog_name(h):
    msg = await h.call_error(TOOL, {"action": "get", "object_type": "schema", "full_name": "a.b", "catalog_name": "x"})
    assert "conflicts" in msg


async def test_url_unsafe_name_rejected(h):
    msg = await h.call_error(TOOL, {"action": "get", "object_type": "catalog", "full_name": "a/b?x"})
    assert "must not contain" in msg
    h.w.catalogs.get.assert_not_called()


# --- list pagination ---------------------------------------------------------------------------

async def test_list_catalogs_paginates(h):
    h.w.catalogs.list.side_effect = lambda **kw: iter([CatalogInfo(name=f"c{i}", owner="me") for i in range(5)])
    first = await h.call(TOOL, {"action": "list", "object_type": "catalog", "page_size": 2})
    assert [c["name"] for c in first["data"]] == ["c0", "c1"]
    assert first["page"]["has_more"] is True
    second = await h.call(
        TOOL, {"action": "list", "object_type": "catalog", "page_size": 2, "page_token": first["page"]["next_page_token"]}
    )
    assert [c["name"] for c in second["data"]] == ["c2", "c3"]


# --- get ----------------------------------------------------------------------------------------

async def test_get_volume_uses_read(h):
    h.w.volumes.read.return_value = VolumeInfo(full_name="main.s.v", name="v", volume_type=VolumeType.EXTERNAL)
    res = await h.call(TOOL, {"action": "get", "object_type": "volume", "catalog_name": "main", "schema_name": "s", "name": "v"})
    assert res["data"]["volume_type"] == "EXTERNAL"
    h.w.volumes.read.assert_called_once_with("main.s.v")


async def test_get_backtick_name(h):
    h.w.functions.get.return_value = FunctionInfo(full_name="main.s.f", name="f")
    await h.call(TOOL, {"action": "get", "object_type": "function", "full_name": "`main`.`s`.`f`"})
    h.w.functions.get.assert_called_once_with("main.s.f")


async def test_not_found_maps(h):
    h.w.catalogs.get.side_effect = NotFound("Catalog 'nope' does not exist.")
    msg = await h.call_error(TOOL, {"action": "get", "object_type": "catalog", "full_name": "nope"})
    assert "[NOT_FOUND]" in msg


# --- create -------------------------------------------------------------------------------------

async def test_create_schema(h):
    h.w.schemas.create.return_value = SchemaInfo(full_name="main.sales", name="sales", catalog_name="main")
    res = await h.call(TOOL, {"action": "create", "object_type": "schema", "full_name": "main.sales", "spec": {"comment": "x"}})
    assert res["status"] == "success"
    h.w.schemas.create.assert_called_once_with(comment="x", catalog_name="main", name="sales")


async def test_create_unknown_field_rejected(h):
    msg = await h.call_error(TOOL, {"action": "create", "object_type": "catalog", "full_name": "c", "spec": {"colour": "red"}})
    assert "Unknown field" in msg and "colour" in msg
    h.w.catalogs.create.assert_not_called()


async def test_create_spec_cannot_override_identifier(h):
    msg = await h.call_error(TOOL, {"action": "create", "object_type": "catalog", "full_name": "c", "spec": {"name": "other"}})
    assert "dedicated" in msg
    h.w.catalogs.create.assert_not_called()


async def test_create_volume_defaults_managed(h):
    h.w.volumes.create.return_value = VolumeInfo(full_name="main.s.v", volume_type=VolumeType.MANAGED)
    res = await h.call(TOOL, {"action": "create", "object_type": "volume", "full_name": "main.s.v"})
    kwargs = h.w.volumes.create.call_args.kwargs
    assert kwargs["volume_type"] == VolumeType.MANAGED and kwargs["name"] == "v"
    assert any("MANAGED" in w for w in res["warnings"])


async def test_create_managed_table_unsupported(h):
    msg = await h.call_error(
        TOOL, {"action": "create", "object_type": "table", "full_name": "main.s.t", "spec": {"table_type": "MANAGED"}}
    )
    assert "UNSUPPORTED_OPERATION" in msg and "execute_sql" in msg
    h.w.tables.create.assert_not_called()


async def test_create_external_delta_table(h):
    h.w.tables.create.return_value = TableInfo(full_name="main.s.t")
    await h.call(
        TOOL,
        {
            "action": "create",
            "object_type": "table",
            "full_name": "main.s.t",
            "spec": {"table_type": "EXTERNAL", "data_source_format": "DELTA", "storage_location": "s3://b/p"},
        },
    )
    kwargs = h.w.tables.create.call_args.kwargs
    assert kwargs["table_type"] == TableType.EXTERNAL and kwargs["schema_name"] == "s"


async def test_create_function_missing_fields(h):
    msg = await h.call_error(TOOL, {"action": "create", "object_type": "function", "full_name": "main.s.f", "spec": {}})
    assert "Missing required function field" in msg
    h.w.functions.create.assert_not_called()


async def test_create_tracks_manifest(h):
    h.w.catalogs.create.return_value = CatalogInfo(name="sandbox")
    await h.call(TOOL, {"action": "create", "object_type": "catalog", "name": "sandbox"})
    from dbx_mcp.server.context import get_context

    assert [r.resource_id for r in get_context().manifest.list("uc_catalog")] == ["sandbox"]


# --- update -------------------------------------------------------------------------------------

async def test_update_comment_is_plain_write(h):
    h.w.catalogs.update.return_value = CatalogInfo(name="c", comment="new")
    res = await h.call(TOOL, {"action": "update", "object_type": "catalog", "full_name": "c", "spec": {"comment": "new"}})
    assert res["status"] == "success"
    assert res["safety"] == ["WRITE"]
    h.w.catalogs.update.assert_called_once_with(comment="new", name="c")


async def test_update_owner_needs_confirmation_with_diff(h):
    h.w.schemas.get.return_value = SchemaInfo(full_name="main.s", owner="alice")
    res = await h.call(TOOL, {"action": "update", "object_type": "schema", "full_name": "main.s", "spec": {"owner": "bob"}})
    assert res["status"] == "confirmation_required"
    assert "SECURITY_SENSITIVE" in res["safety"]
    assert res["plan"]["details"]["changes"]["owner"] == {"before": "alice", "after": "bob"}
    h.w.schemas.update.assert_not_called()

    h.w.schemas.update.return_value = SchemaInfo(full_name="main.s", owner="bob")
    res = await h.call(
        TOOL, {"action": "update", "object_type": "schema", "full_name": "main.s", "spec": {"owner": "bob"}, "confirm": True}
    )
    assert res["status"] == "success"
    h.w.schemas.update.assert_called_once_with(owner="bob", full_name="main.s")


async def test_update_table_only_owner(h):
    msg = await h.call_error(
        TOOL, {"action": "update", "object_type": "table", "full_name": "main.s.t", "spec": {"comment": "x"}}
    )
    assert "owner" in msg and "manage_uc_tags" in msg
    h.w.tables.update.assert_not_called()


# --- delete -------------------------------------------------------------------------------------

async def test_delete_requires_confirmation(h):
    h.w.tables.get.return_value = TableInfo(full_name="main.s.t", table_type=TableType.MANAGED, owner="me")
    res = await h.call(TOOL, {"action": "delete", "object_type": "table", "full_name": "main.s.t"})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["reversible"] is False
    assert any("MANAGED" in w for w in res["plan"]["warnings"])
    h.w.tables.delete.assert_not_called()


async def test_delete_with_confirm(h):
    h.w.tables.get.return_value = TableInfo(full_name="main.s.t", table_type=TableType.EXTERNAL)
    res = await h.call(TOOL, {"action": "delete", "object_type": "table", "full_name": "main.s.t", "confirm": True})
    assert res["status"] == "success"
    h.w.tables.delete.assert_called_once_with("main.s.t")


async def test_force_delete_catalog_plan_counts_children(h):
    h.w.catalogs.get.return_value = CatalogInfo(name="sandbox", owner="me")
    h.w.schemas.list.return_value = iter(
        [SchemaInfo(name="information_schema"), SchemaInfo(name="a"), SchemaInfo(name="b")]
    )
    res = await h.call(TOOL, {"action": "delete", "object_type": "catalog", "full_name": "sandbox", "force": True, "dry_run": True})
    assert res["status"] == "dry_run"
    assert res["plan"]["details"]["contained_objects"] == {"schemas": 2}
    assert res["plan"]["warnings"][0].startswith("FORCE DELETE")
    h.w.catalogs.delete.assert_not_called()


async def test_force_delete_confirmed_passes_force(h):
    h.w.catalogs.get.return_value = CatalogInfo(name="sandbox")
    await h.call(TOOL, {"action": "delete", "object_type": "catalog", "full_name": "sandbox", "force": True, "confirm": True})
    h.w.catalogs.delete.assert_called_once_with("sandbox", force=True)


async def test_force_not_allowed_for_volume(h):
    msg = await h.call_error(TOOL, {"action": "delete", "object_type": "volume", "full_name": "a.b.c", "force": True})
    assert "force" in msg
    h.w.volumes.delete.assert_not_called()


async def test_delete_protected_name_blocked(h):
    h.w.catalogs.get.return_value = CatalogInfo(name="prod")
    msg = await h.call_error(TOOL, {"action": "delete", "object_type": "catalog", "full_name": "prod", "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.catalogs.delete.assert_not_called()


# --- read-only mode -----------------------------------------------------------------------------

async def test_read_only_blocks_changes_but_allows_reads(make_harness):
    h = make_harness(read_only=True, toolsets=("unity_catalog",))
    msg = await h.call_error(TOOL, {"action": "create", "object_type": "catalog", "full_name": "c"})
    assert "read-only" in msg
    h.w.catalogs.create.assert_not_called()
    h.w.catalogs.get.return_value = CatalogInfo(name="c")
    res = await h.call(TOOL, {"action": "get", "object_type": "catalog", "full_name": "c"})
    assert res["data"]["name"] == "c"
