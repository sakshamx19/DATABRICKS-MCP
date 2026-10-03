"""manage_uc_storage and manage_uc_connections."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    AwsIamRoleResponse,
    AzureServicePrincipal,
    CatalogInfo,
    ConnectionInfo,
    ConnectionType,
    ExternalLocationInfo,
    StorageCredentialInfo,
    ValidateStorageCredentialResponse,
    ValidationResult,
    ValidationResultOperation,
    ValidationResultResult,
)

STORAGE = "manage_uc_storage"
CONN = "manage_uc_connections"


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=("unity_catalog",))


def _azure_cred(name: str = "cred") -> StorageCredentialInfo:
    return StorageCredentialInfo(
        name=name,
        owner="me",
        azure_service_principal=AzureServicePrincipal(directory_id="dir", application_id="app", client_secret="S3CRET-VALUE"),
    )


# ===== storage =================================================================================

async def test_storage_get_strips_secrets(h):
    h.w.storage_credentials.get.return_value = _azure_cred()
    res = await h.call(STORAGE, {"action": "get", "resource": "storage_credential", "name": "cred"})
    assert "S3CRET-VALUE" not in json.dumps(res)
    assert "client_secret" not in res["data"]["azure_service_principal"]
    assert res["data"]["azure_service_principal"]["application_id"] == "app"


async def test_storage_list_credentials(h):
    h.w.storage_credentials.list.return_value = iter(
        [_azure_cred("a"), StorageCredentialInfo(name="b", aws_iam_role=AwsIamRoleResponse(role_arn="arn:aws:iam::1:role/r"))]
    )
    res = await h.call(STORAGE, {"action": "list", "resource": "storage_credential"})
    assert "S3CRET-VALUE" not in json.dumps(res)
    assert res["data"][0]["credential_kind"] == "azure_service_principal"
    assert res["data"][1]["aws_iam_role"]["role_arn"].startswith("arn:aws")


async def test_external_location_create_needs_confirm(h):
    args = {"action": "create", "resource": "external_location", "name": "loc",
            "spec": {"url": "s3://bucket/path", "credential_name": "cred"}}
    res = await h.call(STORAGE, args)
    assert res["status"] == "confirmation_required"
    h.w.external_locations.create.assert_not_called()
    h.w.external_locations.create.return_value = ExternalLocationInfo(name="loc", url="s3://bucket/path")
    res = await h.call(STORAGE, {**args, "confirm": True})
    assert res["status"] == "success"
    h.w.external_locations.create.assert_called_once_with(url="s3://bucket/path", credential_name="cred", name="loc")


async def test_storage_create_unknown_field(h):
    msg = await h.call_error(
        STORAGE, {"action": "create", "resource": "storage_credential", "name": "c", "spec": {"bogus": 1}, "confirm": True}
    )
    assert "Unknown field" in msg
    h.w.storage_credentials.create.assert_not_called()


async def test_storage_create_plan_hides_secret(h):
    res = await h.call(
        STORAGE,
        {"action": "create", "resource": "storage_credential", "name": "c", "dry_run": True,
         "spec": {"azure_service_principal": {"directory_id": "d", "application_id": "a", "client_secret": "TOPSECRET"}}},
    )
    assert res["status"] == "dry_run"
    assert "TOPSECRET" not in json.dumps(res)


async def test_storage_credential_delete_plan_counts_dependents(h):
    h.w.storage_credentials.get.return_value = _azure_cred()
    h.w.external_locations.list.return_value = iter(
        [ExternalLocationInfo(name="l1", credential_name="cred"), ExternalLocationInfo(name="l2", credential_name="other")]
    )
    res = await h.call(STORAGE, {"action": "delete", "resource": "storage_credential", "name": "cred"})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["dependent_external_locations"] == 1
    assert "S3CRET-VALUE" not in json.dumps(res)
    h.w.storage_credentials.delete.assert_not_called()


async def test_storage_delete_confirmed(h):
    res = await h.call(
        STORAGE, {"action": "delete", "resource": "external_location", "name": "loc", "force": True, "confirm": True}
    )
    assert res["status"] == "success"
    h.w.external_locations.delete.assert_called_once_with("loc", force=True)


async def test_storage_update_owner_diff(h):
    h.w.external_locations.get.return_value = ExternalLocationInfo(name="loc", owner="alice", url="s3://b")
    res = await h.call(
        STORAGE, {"action": "update", "resource": "external_location", "name": "loc", "spec": {"owner": "bob"}}
    )
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["changes"]["owner"] == {"before": "alice", "after": "bob"}
    h.w.external_locations.update.assert_not_called()


async def test_validate_external_location(h):
    h.w.external_locations.get.return_value = ExternalLocationInfo(name="loc", url="s3://b/p", credential_name="cred")
    h.w.storage_credentials.validate.return_value = ValidateStorageCredentialResponse(
        is_dir=True,
        results=[
            ValidationResult(operation=ValidationResultOperation.READ, result=ValidationResultResult.PASS),
            ValidationResult(operation=ValidationResultOperation.WRITE, result=ValidationResultResult.FAIL, message="denied"),
        ],
    )
    res = await h.call(STORAGE, {"action": "validate", "resource": "external_location", "name": "loc"})
    assert res["status"] == "success" and "1 failed" in res["summary"]
    h.w.storage_credentials.validate.assert_called_once_with(
        storage_credential_name="cred", external_location_name="loc", url="s3://b/p"
    )


async def test_validate_credential_requires_target(h):
    msg = await h.call_error(STORAGE, {"action": "validate", "resource": "storage_credential", "name": "cred"})
    assert "url" in msg
    h.w.storage_credentials.validate.assert_not_called()


async def test_storage_read_only_mode(make_harness):
    h = make_harness(read_only=True, toolsets=("unity_catalog",))
    msg = await h.call_error(STORAGE, {"action": "delete", "resource": "storage_credential", "name": "c", "confirm": True})
    assert "read-only" in msg
    h.w.storage_credentials.delete.assert_not_called()


async def test_storage_not_found(h):
    h.w.storage_credentials.get.side_effect = NotFound("Storage Credential 'x' does not exist.")
    msg = await h.call_error(STORAGE, {"action": "get", "resource": "storage_credential", "name": "x"})
    assert "[NOT_FOUND]" in msg


# ===== connections =============================================================================

def _conn(**options) -> ConnectionInfo:
    return ConnectionInfo(name="sf", connection_type=ConnectionType.SNOWFLAKE, owner="me", options=options)


async def test_connection_get_hides_secret_options(h):
    h.w.connections.get.return_value = _conn(host="acct.snowflakecomputing.com", port="443", user="svc_user",
                                             password="P@ssw0rd!", sfWarehouse="WH")
    res = await h.call(CONN, {"action": "get", "name": "sf"})
    dumped = json.dumps(res)
    assert "P@ssw0rd!" not in dumped and "svc_user" not in dumped
    assert res["data"]["options"] == {"host": "acct.snowflakecomputing.com", "port": "443"}
    assert res["data"]["hidden_option_keys"] == ["password", "sfWarehouse", "user"]


async def test_connection_list_hides_options(h):
    h.w.connections.list.return_value = iter([_conn(host="h", password="pw-123456")])
    res = await h.call(CONN, {"action": "list"})
    assert "pw-123456" not in json.dumps(res)
    assert res["data"][0]["name"] == "sf"


async def test_connection_create_passes_secrets_but_never_returns_them(h):
    args = {"action": "create", "name": "pg", "connection_type": "postgresql",
            "options": {"host": "db.example.com", "port": 5432, "user": "admin_user", "password": "hunter2-secret"},
            "spec": {"comment": "federation"}}
    res = await h.call(CONN, args)
    assert res["status"] == "confirmation_required"
    dumped = json.dumps(res)
    assert "hunter2-secret" not in dumped and "admin_user" not in dumped
    assert res["plan"]["details"]["hidden_option_keys"] == ["password", "user"]
    h.w.connections.create.assert_not_called()

    h.w.connections.create.return_value = ConnectionInfo(
        name="pg", connection_type=ConnectionType.POSTGRESQL,
        options={"host": "db.example.com", "port": "5432", "user": "admin_user", "password": "hunter2-secret"},
    )
    res = await h.call(CONN, {**args, "confirm": True})
    assert res["status"] == "success"
    assert "hunter2-secret" not in json.dumps(res) and "admin_user" not in json.dumps(res)
    kwargs = h.w.connections.create.call_args.kwargs
    assert kwargs["options"] == {"host": "db.example.com", "port": "5432", "user": "admin_user", "password": "hunter2-secret"}
    assert kwargs["connection_type"] == ConnectionType.POSTGRESQL
    assert kwargs["comment"] == "federation"


async def test_connection_invalid_type(h):
    msg = await h.call_error(CONN, {"action": "create", "name": "x", "connection_type": "FTP", "options": {"host": "h"}})
    assert "Unknown connection_type" in msg and "SNOWFLAKE" in msg


async def test_connection_update_requires_options(h):
    msg = await h.call_error(CONN, {"action": "update", "name": "sf", "spec": {"owner": "bob"}, "confirm": True})
    assert "options" in msg
    h.w.connections.update.assert_not_called()


async def test_connection_options_cannot_be_in_spec(h):
    msg = await h.call_error(
        CONN,
        {"action": "update", "name": "sf", "options": {"host": "h"}, "spec": {"options": {"password": "zzz"}}, "confirm": True},
    )
    assert "dedicated" in msg and "zzz" not in msg
    h.w.connections.update.assert_not_called()


async def test_connection_delete_plan_lists_foreign_catalogs(h):
    h.w.connections.get.return_value = _conn(host="h", password="pw-abcdef")
    h.w.catalogs.list.return_value = iter([CatalogInfo(name="sf_cat", connection_name="sf"), CatalogInfo(name="main")])
    res = await h.call(CONN, {"action": "delete", "name": "sf"})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["details"]["dependent_foreign_catalogs"] == 1
    assert "pw-abcdef" not in json.dumps(res)
    h.w.connections.delete.assert_not_called()
    res = await h.call(CONN, {"action": "delete", "name": "sf", "confirm": True})
    assert res["status"] == "success"
    h.w.connections.delete.assert_called_once_with("sf")
