"""Unit tests for the Lakebase toolset (mocked WorkspaceClient; no real workspace)."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.common.types.fieldmask import FieldMask
from databricks.sdk.errors import NotFound
from databricks.sdk.service import database as db
from databricks.sdk.service import postgres as pg
from databricks.sdk.service._internal import Wait
from databricks.sdk.service.iam import User
from databricks.sdk.service.pipelines import StartUpdateResponse

SECRET = "lakebase-oauth-secret-value-0123456789"
OP = "projects/app/operations/op-1"


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=("lakebase",))


def _instance(name="dev-db", state=db.DatabaseInstanceState.AVAILABLE, **kw):
    return db.DatabaseInstance(name=name, state=state, capacity="CU_1", read_write_dns=f"{name}.pg.example", **kw)


def _pending_op(h, op_cls, done=False, response=None, error=None):
    h.w.postgres.get_operation.return_value = pg.Operation(name=OP, done=done, response=response, error=error)
    return op_cls(h.w.postgres, pg.Operation(name=OP, done=False))


def _manifest(h):
    path = h.settings.manifest_path
    return json.loads(path.read_text(encoding="utf-8"))["resources"] if path.exists() else []


# ----------------------------------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------------------------------

def test_tools_registered(h):
    assert h.tool_names() == {
        "manage_lakebase_database",
        "manage_lakebase_branch",
        "manage_lakebase_sync",
        "generate_lakebase_credential",
    }


# ----------------------------------------------------------------------------------------------
# manage_lakebase_database - provisioned
# ----------------------------------------------------------------------------------------------

async def test_list_instances_compact(h):
    h.w.database.list_database_instances.return_value = iter(
        [_instance("a", creator="me@x.com", uid="u1"), _instance("b", state=db.DatabaseInstanceState.STOPPED)]
    )
    res = await h.call("manage_lakebase_database", {"action": "list"})
    assert res["status"] == "success"
    assert [i["name"] for i in res["data"]] == ["a", "b"]
    assert res["data"][1]["state"] == "STOPPED"
    assert res["page"]["returned"] == 2


async def test_get_instance(h):
    h.w.database.get_database_instance.return_value = _instance()
    res = await h.call("manage_lakebase_database", {"action": "get", "name": "dev-db"})
    assert res["data"]["read_write_dns"] == "dev-db.pg.example"
    assert "AVAILABLE" in res["summary"]
    h.w.database.get_database_instance.assert_called_once_with(name="dev-db")


async def test_get_instance_not_found(h):
    h.w.database.get_database_instance.side_effect = NotFound("Database instance 'nope' does not exist")
    msg = await h.call_error("manage_lakebase_database", {"action": "get", "name": "nope"})
    assert "[NOT_FOUND]" in msg


async def test_get_requires_name(h):
    msg = await h.call_error("manage_lakebase_database", {"action": "get"})
    assert "[INVALID_PARAMETER]" in msg and "name" in msg


async def test_create_instance_rejects_unknown_field(h):
    msg = await h.call_error(
        "manage_lakebase_database",
        {"action": "create", "name": "dev-db", "spec": {"capacity": "CU_1", "bogus_field": 1}},
    )
    assert "[INVALID_PARAMETER]" in msg and "bogus_field" in msg
    h.w.database.create_database_instance.assert_not_called()


async def test_create_instance_returns_pending_and_tracks(h):
    starting = _instance(state=db.DatabaseInstanceState.STARTING)
    h.w.database.create_database_instance.return_value = Wait(lambda **kw: starting, response=starting, name="dev-db")
    res = await h.call(
        "manage_lakebase_database", {"action": "create", "name": "dev-db", "spec": {"capacity": "CU_1"}}
    )
    assert res["status"] == "pending"
    assert any("billed" in w for w in res["warnings"])
    sent = h.w.database.create_database_instance.call_args.kwargs["database_instance"]
    assert isinstance(sent, db.DatabaseInstance) and sent.name == "dev-db" and sent.capacity == "CU_1"
    h.w.database.wait_get_database_instance_database_available.assert_not_called()
    assert any(r["resource_type"] == "lakebase_instance" and r["resource_id"] == "dev-db" for r in _manifest(h))


async def test_create_instance_bounded_wait(h):
    starting = _instance(state=db.DatabaseInstanceState.STARTING)
    h.w.database.create_database_instance.return_value = Wait(lambda **kw: starting, response=starting)
    h.w.database.wait_get_database_instance_database_available.return_value = _instance()
    res = await h.call(
        "manage_lakebase_database",
        {"action": "create", "name": "dev-db", "spec": {"capacity": "CU_1"}, "wait_seconds": 30},
    )
    assert res["status"] == "success"
    timeout = h.w.database.wait_get_database_instance_database_available.call_args.kwargs["timeout"]
    assert timeout.total_seconds() == 30


async def test_create_instance_dry_run(h):
    res = await h.call(
        "manage_lakebase_database",
        {"action": "create", "name": "dev-db", "spec": {"capacity": "CU_1"}, "dry_run": True},
    )
    assert res["status"] == "dry_run"
    assert any("billed" in w for w in res["plan"]["warnings"])
    h.w.database.create_database_instance.assert_not_called()


async def test_update_instance_derives_mask(h):
    h.w.database.update_database_instance.return_value = _instance(
        state=db.DatabaseInstanceState.UPDATING, stopped=True
    )
    res = await h.call("manage_lakebase_database", {"action": "update", "name": "dev-db", "spec": {"stopped": True}})
    kwargs = h.w.database.update_database_instance.call_args.kwargs
    assert kwargs["update_mask"] == "stopped"
    assert kwargs["database_instance"].stopped is True and kwargs["database_instance"].name == "dev-db"
    assert res["status"] == "pending"


async def test_read_only_blocks_writes(make_harness):
    h = make_harness(read_only=True, toolsets=("lakebase",))
    msg = await h.call_error("manage_lakebase_database", {"action": "create", "name": "x", "spec": {}})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in msg
    h.w.database.create_database_instance.assert_not_called()
    h.w.database.list_database_instances.return_value = iter([])
    res = await h.call("manage_lakebase_database", {"action": "list"})
    assert res["status"] == "success"


async def test_delete_instance_requires_confirmation(h):
    h.w.database.get_database_instance.return_value = _instance(
        child_instance_refs=[db.DatabaseInstanceRef(name="dev-db-pitr")]
    )
    res = await h.call("manage_lakebase_database", {"action": "delete", "name": "dev-db"})
    assert res["status"] == "confirmation_required"
    plan = res["plan"]
    assert plan["target"]["name"] == "dev-db"
    assert plan["details"]["state"] == "AVAILABLE" and plan["details"]["capacity"] == "CU_1"
    assert plan["reversible"] is False
    assert any("force=true" in w for w in plan["warnings"])
    h.w.database.delete_database_instance.assert_not_called()


async def test_delete_instance_with_confirm(h):
    h.w.database.get_database_instance.return_value = _instance()
    res = await h.call("manage_lakebase_database", {"action": "delete", "name": "dev-db", "confirm": True})
    assert res["status"] == "pending"
    h.w.database.delete_database_instance.assert_called_once_with(name="dev-db", force=None)


async def test_delete_protected_instance_blocked(h):
    h.w.database.get_database_instance.return_value = _instance("prod-db")
    msg = await h.call_error("manage_lakebase_database", {"action": "delete", "name": "prod-db", "confirm": True})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in msg
    h.w.database.delete_database_instance.assert_not_called()


async def test_provisioned_undelete_unsupported(h):
    msg = await h.call_error("manage_lakebase_database", {"action": "undelete", "name": "dev-db"})
    assert "[UNSUPPORTED_OPERATION]" in msg


async def test_create_and_delete_database_catalog(h):
    h.w.database.create_database_catalog.return_value = db.DatabaseCatalog(
        name="lb_cat", database_instance_name="dev-db", database_name="appdb"
    )
    res = await h.call(
        "manage_lakebase_database",
        {"action": "create_catalog", "name": "dev-db", "catalog_name": "lb_cat", "database_name": "appdb",
         "create_database_if_missing": True},
    )
    assert res["status"] == "success"
    cat = h.w.database.create_database_catalog.call_args.kwargs["catalog"]
    assert cat.create_database_if_not_exists is True and cat.database_instance_name == "dev-db"

    h.w.database.get_database_catalog.return_value = cat
    res = await h.call("manage_lakebase_database", {"action": "delete_catalog", "catalog_name": "lb_cat"})
    assert res["status"] == "confirmation_required"
    h.w.database.delete_database_catalog.assert_not_called()
    await h.call("manage_lakebase_database", {"action": "delete_catalog", "catalog_name": "lb_cat", "confirm": True})
    h.w.database.delete_database_catalog.assert_called_once_with(name="lb_cat")


# ----------------------------------------------------------------------------------------------
# manage_lakebase_database - autoscaling
# ----------------------------------------------------------------------------------------------

async def test_list_projects(h):
    h.w.postgres.list_projects.return_value = iter(
        [pg.Project(name="projects/app", status=pg.ProjectStatus(display_name="App", default_branch="projects/app/branches/main"))]
    )
    res = await h.call("manage_lakebase_database", {"action": "list", "kind": "autoscaling"})
    assert res["data"] == [
        {"name": "projects/app", "display_name": "App", "default_branch": "projects/app/branches/main"}
    ]


async def test_create_project_pending(h):
    h.w.postgres.create_project.return_value = _pending_op(h, pg.CreateProjectOperation)
    res = await h.call(
        "manage_lakebase_database",
        {"action": "create", "kind": "autoscaling", "name": "app",
         "spec": {"spec": {"display_name": "App", "pg_version": 17}}},
    )
    assert res["status"] == "pending"
    assert res["data"]["operation"] == {"name": OP, "done": False}
    kwargs = h.w.postgres.create_project.call_args.kwargs
    assert kwargs["project_id"] == "app" and kwargs["project"].spec.pg_version == 17
    h.w.postgres.get_operation.assert_called_once_with(name=OP)


async def test_create_project_rejects_unknown_nested_field(h):
    msg = await h.call_error(
        "manage_lakebase_database",
        {"action": "create", "kind": "autoscaling", "name": "app", "spec": {"spec": {"nope": 1}}},
    )
    assert "[INVALID_PARAMETER]" in msg and "nope" in msg
    h.w.postgres.create_project.assert_not_called()


async def test_update_project_field_mask_and_done(h):
    h.w.postgres.update_project.return_value = _pending_op(
        h, pg.UpdateProjectOperation, done=True, response={"name": "projects/app", "status": {"display_name": "New"}}
    )
    res = await h.call(
        "manage_lakebase_database",
        {"action": "update", "kind": "autoscaling", "name": "projects/app", "spec": {"spec": {"display_name": "New"}}},
    )
    assert res["status"] == "success"
    assert res["data"]["result"]["status"]["display_name"] == "New"
    kwargs = h.w.postgres.update_project.call_args.kwargs
    assert kwargs["update_mask"] == FieldMask(field_mask=["spec.display_name"])
    assert kwargs["name"] == "projects/app"


async def test_failed_operation_reported(h):
    err = pg.DatabricksServiceExceptionWithDetailsProto(error_code=pg.ErrorCode.QUOTA_EXCEEDED, message="quota")
    h.w.postgres.create_project.return_value = _pending_op(h, pg.CreateProjectOperation, done=True, error=err)
    res = await h.call("manage_lakebase_database", {"action": "create", "kind": "autoscaling", "name": "app"})
    assert res["status"] == "failed"
    assert res["data"]["operation"]["error"]["error_code"] == "QUOTA_EXCEEDED"


async def test_delete_project_confirm_flow(h):
    h.w.postgres.get_project.return_value = pg.Project(
        name="projects/app", status=pg.ProjectStatus(display_name="App", owner="me@x.com")
    )
    res = await h.call("manage_lakebase_database", {"action": "delete", "kind": "autoscaling", "name": "app"})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["reversible"] is True
    h.w.postgres.delete_project.assert_not_called()

    h.w.postgres.delete_project.return_value = _pending_op(h, pg.DeleteProjectOperation, done=True)
    res = await h.call(
        "manage_lakebase_database",
        {"action": "delete", "kind": "autoscaling", "name": "app", "purge": True, "confirm": True},
    )
    assert res["status"] == "success"
    h.w.postgres.delete_project.assert_called_once_with(name="projects/app", purge=True)


async def test_get_operation(h):
    h.w.postgres.get_operation.return_value = pg.Operation(name=OP, done=True, response={"name": "projects/app"})
    res = await h.call("manage_lakebase_database", {"action": "get_operation", "operation_name": OP})
    assert res["data"]["done"] is True and "done" in res["summary"]


async def test_autoscaling_list_catalogs_unsupported(h):
    msg = await h.call_error("manage_lakebase_database", {"action": "list_catalogs", "kind": "autoscaling"})
    assert "[UNSUPPORTED_OPERATION]" in msg


# ----------------------------------------------------------------------------------------------
# manage_lakebase_branch
# ----------------------------------------------------------------------------------------------

def _branch(branch_id="dev", default=False, **status):
    return pg.Branch(
        name=f"projects/app/branches/{branch_id}",
        status=pg.BranchStatus(branch_id=branch_id, default=default, current_state=pg.BranchStatusState.READY, **status),
    )


async def test_list_branches(h):
    h.w.postgres.list_branches.return_value = iter([_branch("main", default=True), _branch("dev")])
    res = await h.call("manage_lakebase_branch", {"action": "list", "project": "app"})
    assert [b["branch_id"] for b in res["data"]] == ["main", "dev"]
    assert res["data"][0]["default"] is True and res["data"][0]["state"] == "READY"
    h.w.postgres.list_branches.assert_called_once_with(parent="projects/app", show_deleted=None)


async def test_create_branch_defaults_to_project_default_and_point_in_time(h):
    h.w.postgres.get_project.return_value = pg.Project(
        name="projects/app", status=pg.ProjectStatus(default_branch="projects/app/branches/main")
    )
    h.w.postgres.create_branch.return_value = _pending_op(h, pg.CreateBranchOperation)
    res = await h.call(
        "manage_lakebase_branch",
        {"action": "create", "project": "app", "branch": "dev",
         "source_branch_time": "2025-01-31T12:00:00Z", "spec": {"ttl": "86400s"}},
    )
    assert res["status"] == "pending"
    kwargs = h.w.postgres.create_branch.call_args.kwargs
    assert kwargs["parent"] == "projects/app" and kwargs["branch_id"] == "dev"
    spec = kwargs["branch"].spec
    assert spec.source_branch == "projects/app/branches/main"
    assert spec.source_branch_time.ToJsonString() == "2025-01-31T12:00:00Z"
    assert spec.ttl.seconds == 86400


async def test_create_branch_rejects_unknown_field(h):
    msg = await h.call_error(
        "manage_lakebase_branch",
        {"action": "create", "project": "app", "branch": "dev", "source_branch": "main", "spec": {"size": 1}},
    )
    assert "[INVALID_PARAMETER]" in msg
    h.w.postgres.create_branch.assert_not_called()


async def test_update_branch_mask(h):
    h.w.postgres.update_branch.return_value = _pending_op(h, pg.UpdateBranchOperation)
    await h.call(
        "manage_lakebase_branch",
        {"action": "update", "project": "app", "branch": "dev", "spec": {"is_protected": True}},
    )
    kwargs = h.w.postgres.update_branch.call_args.kwargs
    assert kwargs["update_mask"] == FieldMask(field_mask=["spec.is_protected"])
    assert kwargs["branch"].name == "projects/app/branches/dev"


async def test_delete_branch_confirmation_then_execute(h):
    h.w.postgres.get_branch.return_value = _branch("dev", logical_size_bytes=1024)
    res = await h.call("manage_lakebase_branch", {"action": "delete", "project": "app", "branch": "dev"})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["logical_size_bytes"] == 1024
    h.w.postgres.delete_branch.assert_not_called()

    h.w.postgres.delete_branch.return_value = _pending_op(h, pg.DeleteBranchOperation)
    res = await h.call("manage_lakebase_branch", {"action": "delete", "project": "app", "branch": "dev", "confirm": True})
    assert res["status"] == "pending"
    h.w.postgres.delete_branch.assert_called_once_with(name="projects/app/branches/dev", purge=None)


async def test_delete_default_branch_refused(h):
    h.w.postgres.get_branch.return_value = _branch("main", default=True)
    msg = await h.call_error(
        "manage_lakebase_branch", {"action": "delete", "project": "app", "branch": "main", "confirm": True}
    )
    assert "[BLOCKED_BY_SAFETY_POLICY]" in msg and "default" in msg
    h.w.postgres.delete_branch.assert_not_called()

    h.w.postgres.delete_branch.return_value = _pending_op(h, pg.DeleteBranchOperation)
    await h.call(
        "manage_lakebase_branch",
        {"action": "delete", "project": "app", "branch": "main", "confirm": True, "allow_default_branch": True},
    )
    h.w.postgres.delete_branch.assert_called_once()


async def test_create_endpoint_and_invalid_enum(h):
    msg = await h.call_error(
        "manage_lakebase_branch",
        {"action": "create_endpoint", "project": "app", "branch": "dev", "endpoint": "rw",
         "spec": {"endpoint_type": "BIG"}},
    )
    assert "[INVALID_PARAMETER]" in msg
    h.w.postgres.create_endpoint.assert_not_called()

    h.w.postgres.create_endpoint.return_value = _pending_op(h, pg.CreateEndpointOperation)
    res = await h.call(
        "manage_lakebase_branch",
        {"action": "create_endpoint", "project": "app", "branch": "dev", "endpoint": "rw",
         "spec": {"endpoint_type": "ENDPOINT_TYPE_READ_WRITE", "autoscaling_limit_max_cu": 2}},
    )
    assert res["status"] == "pending" and any("billed" in w for w in res["warnings"])
    kwargs = h.w.postgres.create_endpoint.call_args.kwargs
    assert kwargs["parent"] == "projects/app/branches/dev" and kwargs["endpoint_id"] == "rw"
    assert kwargs["endpoint"].spec.endpoint_type == pg.EndpointType.ENDPOINT_TYPE_READ_WRITE


async def test_delete_endpoint_requires_confirm(h):
    h.w.postgres.get_endpoint.return_value = pg.Endpoint(
        name="projects/app/branches/dev/endpoints/rw",
        status=pg.EndpointStatus(current_state=pg.EndpointStatusState.ACTIVE,
                                 hosts=pg.EndpointHosts(host="ep.example")),
    )
    res = await h.call(
        "manage_lakebase_branch", {"action": "delete_endpoint", "project": "app", "branch": "dev", "endpoint": "rw"}
    )
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["host"] == "ep.example"
    h.w.postgres.delete_endpoint.assert_not_called()


# ----------------------------------------------------------------------------------------------
# manage_lakebase_sync
# ----------------------------------------------------------------------------------------------

def _synced(policy=db.SyncedTableSchedulingPolicy.TRIGGERED, pipeline_id="pipe-1",
            state=db.SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE):
    return db.SyncedDatabaseTable(
        name="cat.sch.orders_synced",
        database_instance_name="dev-db",
        spec=db.SyncedTableSpec(source_table_full_name="main.sales.orders", primary_key_columns=["id"],
                                scheduling_policy=policy),
        data_synchronization_status=db.SyncedTableStatus(detailed_state=state, pipeline_id=pipeline_id),
    )


async def test_sync_list_and_get(h):
    h.w.database.list_synced_database_tables.return_value = iter([_synced()])
    res = await h.call("manage_lakebase_sync", {"action": "list", "instance_name": "dev-db"})
    assert res["data"][0]["pipeline_id"] == "pipe-1"
    assert res["data"][0]["scheduling_policy"] == "TRIGGERED"

    h.w.database.get_synced_database_table.return_value = _synced()
    res = await h.call("manage_lakebase_sync", {"action": "get", "table_name": "cat.sch.orders_synced"})
    assert res["data"]["spec"]["source_table_full_name"] == "main.sales.orders"


async def test_sync_create_validates_enum_and_creates(h):
    msg = await h.call_error(
        "manage_lakebase_sync",
        {"action": "create", "table_name": "cat.sch.t", "instance_name": "dev-db",
         "spec": {"source_table_full_name": "main.s.t", "scheduling_policy": "HOURLY"}},
    )
    assert "[INVALID_PARAMETER]" in msg
    h.w.database.create_synced_database_table.assert_not_called()

    h.w.database.create_synced_database_table.return_value = _synced(
        state=db.SyncedTableState.SYNCED_TABLE_PROVISIONING
    )
    res = await h.call(
        "manage_lakebase_sync",
        {"action": "create", "table_name": "cat.sch.orders_synced", "instance_name": "dev-db",
         "logical_database_name": "appdb",
         "spec": {"source_table_full_name": "main.sales.orders", "primary_key_columns": ["id"],
                  "scheduling_policy": "SNAPSHOT"}},
    )
    assert res["status"] == "pending"
    table = h.w.database.create_synced_database_table.call_args.kwargs["synced_table"]
    assert table.spec.scheduling_policy == db.SyncedTableSchedulingPolicy.SNAPSHOT
    assert table.logical_database_name == "appdb"


async def test_sync_delete_confirm(h):
    h.w.database.get_synced_database_table.return_value = _synced()
    res = await h.call(
        "manage_lakebase_sync", {"action": "delete", "table_name": "cat.sch.orders_synced", "purge_data": True}
    )
    assert res["status"] == "confirmation_required"
    assert any("DROPS" in w for w in res["plan"]["warnings"])
    h.w.database.delete_synced_database_table.assert_not_called()

    await h.call(
        "manage_lakebase_sync",
        {"action": "delete", "table_name": "cat.sch.orders_synced", "purge_data": True, "confirm": True},
    )
    h.w.database.delete_synced_database_table.assert_called_once_with(name="cat.sch.orders_synced", purge_data=True)


async def test_sync_trigger_starts_pipeline(h):
    h.w.database.get_synced_database_table.return_value = _synced()
    h.w.pipelines.start_update.return_value = StartUpdateResponse(update_id="upd-9")
    res = await h.call("manage_lakebase_sync", {"action": "trigger", "table_name": "cat.sch.orders_synced"})
    assert res["status"] == "pending"
    assert res["data"]["update_id"] == "upd-9"
    h.w.pipelines.start_update.assert_called_once_with(pipeline_id="pipe-1")


async def test_sync_trigger_continuous_and_missing_pipeline(h):
    h.w.database.get_synced_database_table.return_value = _synced(policy=db.SyncedTableSchedulingPolicy.CONTINUOUS)
    msg = await h.call_error("manage_lakebase_sync", {"action": "trigger", "table_name": "cat.sch.orders_synced"})
    assert "CONTINUOUS" in msg

    h.w.database.get_synced_database_table.return_value = _synced(pipeline_id=None)
    msg = await h.call_error("manage_lakebase_sync", {"action": "trigger", "table_name": "cat.sch.orders_synced"})
    assert "[UNSUPPORTED_OPERATION]" in msg
    h.w.pipelines.start_update.assert_not_called()


async def test_sync_update_unsupported(h):
    msg = await h.call_error(
        "manage_lakebase_sync", {"action": "update", "table_name": "cat.sch.t", "spec": {"timeseries_key": "ts"}}
    )
    assert "[UNSUPPORTED_OPERATION]" in msg


async def test_sync_autoscaling_create(h):
    h.w.postgres.create_synced_table.return_value = _pending_op(h, pg.CreateSyncedTableOperation)
    res = await h.call(
        "manage_lakebase_sync",
        {"action": "create", "kind": "autoscaling", "table_name": "cat.sch.t",
         "spec": {"source_table_full_name": "main.s.t", "primary_key_columns": ["id"],
                  "scheduling_policy": "TRIGGERED", "branch": "projects/app/branches/main",
                  "postgres_database": "databricks_postgres"}},
    )
    assert res["status"] == "pending"
    kwargs = h.w.postgres.create_synced_table.call_args.kwargs
    assert kwargs["synced_table_id"] == "cat.sch.t"
    assert kwargs["synced_table"].spec.branch == "projects/app/branches/main"


# ----------------------------------------------------------------------------------------------
# generate_lakebase_credential
# ----------------------------------------------------------------------------------------------

def _cred_mocks(h):
    h.w.current_user.me.return_value = User(user_name="me@example.com")
    h.w.database.generate_database_credential.return_value = db.DatabaseCredential(
        expiration_time="2026-10-03T13:00:00Z", token=SECRET
    )
    h.w.database.get_database_instance.return_value = _instance()


async def test_credential_requires_confirmation(h):
    _cred_mocks(h)
    res = await h.call("generate_lakebase_credential", {"instance_names": ["dev-db"]})
    assert res["status"] == "confirmation_required"
    h.w.database.generate_database_credential.assert_not_called()
    assert SECRET not in json.dumps(res)


async def test_credential_dry_run(h):
    _cred_mocks(h)
    res = await h.call("generate_lakebase_credential", {"instance_names": ["dev-db"], "dry_run": True})
    assert res["status"] == "dry_run"
    h.w.database.generate_database_credential.assert_not_called()


async def test_credential_without_reveal_withholds_token(h):
    _cred_mocks(h)
    res = await h.call("generate_lakebase_credential", {"instance_names": ["dev-db"], "confirm": True})
    assert res["status"] == "success"
    assert SECRET not in json.dumps(res)
    assert "token" not in res["data"]
    data = res["data"]
    assert data["expiration_time"] == "2026-10-03T13:00:00Z"
    conn = data["connections"][0]
    assert conn["host"] == "dev-db.pg.example" and conn["port"] == 5432
    assert conn["database"] == "databricks_postgres" and conn["user"] == "me@example.com"
    assert conn["sslmode"] == "require"
    assert "reveal_token" in data["how_to_obtain"]
    h.w.database.generate_database_credential.assert_called_once_with(
        instance_names=["dev-db"], claims=None, request_id=None
    )


async def test_credential_reveal_returns_token(h):
    _cred_mocks(h)
    res = await h.call(
        "generate_lakebase_credential", {"instance_names": ["dev-db"], "reveal_token": True, "confirm": True}
    )
    assert res["data"]["token"] == SECRET
    assert any("secret" in w for w in res["warnings"])


async def test_credential_read_only_blocked(make_harness):
    h = make_harness(read_only=True, toolsets=("lakebase",))
    msg = await h.call_error("generate_lakebase_credential", {"instance_names": ["dev-db"], "confirm": True})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in msg
    h.w.database.generate_database_credential.assert_not_called()


async def test_credential_autoscaling_with_ttl_and_claims(h):
    from google.protobuf.timestamp_pb2 import Timestamp

    expire = Timestamp()
    expire.FromJsonString("2026-10-03T13:00:00Z")
    h.w.current_user.me.return_value = User(user_name="me@example.com")
    h.w.postgres.generate_database_credential.return_value = pg.DatabaseCredential(expire_time=expire, token=SECRET)
    endpoint = "projects/app/branches/main/endpoints/rw"
    h.w.postgres.get_endpoint.return_value = pg.Endpoint(
        name=endpoint, status=pg.EndpointStatus(hosts=pg.EndpointHosts(host="ep.example"))
    )
    res = await h.call(
        "generate_lakebase_credential",
        {"kind": "autoscaling", "endpoint": endpoint, "ttl_seconds": 900, "confirm": True,
         "claims": [{"permission_set": "READ_ONLY", "resources": [{"table_name": "c.s.t"}]}]},
    )
    assert SECRET not in json.dumps(res)
    assert res["data"]["expiration_time"] == "2026-10-03T13:00:00Z"
    assert res["data"]["connections"][0]["host"] == "ep.example"
    kwargs = h.w.postgres.generate_database_credential.call_args.kwargs
    assert kwargs["endpoint"] == endpoint and kwargs["ttl"].seconds == 900
    assert kwargs["claims"][0].permission_set == pg.RequestedClaimsPermissionSet.READ_ONLY


async def test_credential_rejects_unknown_claim_field(h):
    _cred_mocks(h)
    msg = await h.call_error(
        "generate_lakebase_credential",
        {"instance_names": ["dev-db"], "confirm": True, "claims": [{"permission_set": "READ_ONLY", "x": 1}]},
    )
    assert "[INVALID_PARAMETER]" in msg
    h.w.database.generate_database_credential.assert_not_called()
