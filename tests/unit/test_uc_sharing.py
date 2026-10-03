"""manage_uc_sharing: shares, recipients, providers - secrets never leave the server."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service import iam, sharing

TOOL = "manage_uc_sharing"
ACTIVATION = "https://example.cloud.databricks.com/delta_sharing/retrieve_config.html?abcSECRETxyz"
PROFILE = '{"shareCredentialsVersion":1,"bearerToken":"PROVIDER-BEARER-SECRET","endpoint":"https://x"}'


@pytest.fixture
def h(make_harness):
    harness = make_harness(toolsets=("unity_catalog",))
    harness.w.current_user.me.return_value = iam.User(user_name="alice@example.com")
    harness.w.shares.get.return_value = sharing.ShareInfo(
        name="s1",
        owner="alice@example.com",
        objects=[sharing.SharedDataObject(name="main.sales.orders", data_object_type=sharing.SharedDataObjectDataObjectType.TABLE)],
    )
    harness.w.shares.share_permissions.return_value = sharing.GetSharePermissionsResponse(
        privilege_assignments=[sharing.PrivilegeAssignment(principal="acme", privileges=[sharing.Privilege.SELECT])]
    )
    return harness


def _recipient():
    return sharing.RecipientInfo(
        name="acme",
        authentication_type=sharing.AuthenticationType.TOKEN,
        activated=False,
        activation_url=ACTIVATION,
        sharing_code="SHARING-CODE-SECRET",
        tokens=[
            sharing.RecipientTokenInfo(id="tok1", activation_url=ACTIVATION, expiration_time=1999999999000, created_by="alice")
        ],
    )


def _no_secrets(payload):
    text = json.dumps(payload)
    assert "abcSECRETxyz" not in text
    assert "activation_url" not in text
    assert "SHARING-CODE-SECRET" not in text
    assert "PROVIDER-BEARER-SECRET" not in text
    assert "recipient_profile_str" not in text or "***" in text


async def test_recipient_get_strips_tokens_and_activation_links(h):
    h.w.recipients.get.return_value = _recipient()
    res = await h.call(TOOL, {"resource": "recipient", "action": "get", "name": "acme"})
    _no_secrets(res)
    data = res["data"]
    assert "tokens" not in data
    assert data["token_metadata"] == [{"id": "tok1", "created_by": "alice", "expiration_time": 1999999999000}]
    assert data["name"] == "acme"


async def test_recipient_list_is_compact_and_clean(h):
    h.w.recipients.list.return_value = iter([_recipient()])
    res = await h.call(TOOL, {"resource": "recipient", "action": "list"})
    _no_secrets(res)
    assert res["data"] == [{"name": "acme", "authentication_type": "TOKEN", "activated": False}]


async def test_recipient_create_requires_confirmation_and_strips_output(h):
    args = {"resource": "recipient", "action": "create", "name": "acme", "spec": {"authentication_type": "TOKEN"}}
    res = await h.call(TOOL, args)
    assert res["status"] == "confirmation_required"
    assert "SECURITY_SENSITIVE" in res["safety"]
    h.w.recipients.create.assert_not_called()

    h.w.recipients.create.return_value = _recipient()
    res = await h.call(TOOL, {**args, "confirm": True})
    _no_secrets(res)
    h.w.recipients.create.assert_called_once_with(name="acme", authentication_type=sharing.AuthenticationType.TOKEN)
    assert res["data"]["audit"]["who"] == "alice@example.com"
    assert res["data"]["audit"]["what"] == "recipients.create acme"


async def test_recipient_create_rejects_unknown_spec_fields(h):
    msg = await h.call_error(
        TOOL,
        {"resource": "recipient", "action": "create", "name": "acme", "spec": {"authentication_type": "TOKEN", "bogus": 1}, "confirm": True},
    )
    assert "bogus" in msg
    h.w.recipients.create.assert_not_called()


async def test_rotate_token_destructive_and_stripped(h):
    h.w.recipients.get.return_value = _recipient()
    args = {"resource": "recipient", "action": "rotate_token", "name": "acme", "existing_token_expire_in_seconds": 0}
    res = await h.call(TOOL, args)
    assert res["status"] == "confirmation_required"
    assert {"DESTRUCTIVE", "SECURITY_SENSITIVE"} <= set(res["safety"])
    _no_secrets(res)
    h.w.recipients.rotate_token.assert_not_called()

    h.w.recipients.rotate_token.return_value = _recipient()
    res = await h.call(TOOL, {**args, "confirm": True})
    h.w.recipients.rotate_token.assert_called_once_with("acme", 0)
    _no_secrets(res)


async def test_add_objects_warns_about_external_exposure(h):
    args = {
        "resource": "share",
        "action": "add_objects",
        "name": "s1",
        "objects": [{"name": "main.hr.salaries", "data_object_type": "TABLE"}],
    }
    res = await h.call(TOOL, args)
    assert res["status"] == "confirmation_required"
    warning = " ".join(res["warnings"])
    assert "EXTERNAL DATA EXPOSURE" in warning and "main.hr.salaries" in warning and "acme" in warning
    details = res["plan"]["details"]
    assert details["current_objects"] == ["main.sales.orders"]
    assert details["objects_after"] == ["main.sales.orders", "main.hr.salaries"]
    h.w.shares.update.assert_not_called()

    h.w.shares.update.return_value = sharing.ShareInfo(name="s1")
    res = await h.call(TOOL, {**args, "confirm": True})
    kwargs = h.w.shares.update.call_args.kwargs
    assert kwargs["name"] == "s1"
    (upd,) = kwargs["updates"]
    assert upd.action == sharing.SharedDataObjectUpdateAction.ADD
    assert upd.data_object.name == "main.hr.salaries"
    assert upd.data_object.data_object_type == sharing.SharedDataObjectDataObjectType.TABLE
    assert "ADD main.hr.salaries" in res["data"]["audit"]["what"]


async def test_remove_objects_is_destructive(h):
    args = {"resource": "share", "action": "remove_objects", "name": "s1", "objects": ["main.sales.orders"]}
    res = await h.call(TOOL, args)
    assert "DESTRUCTIVE" in res["safety"]
    assert any("lose access" in w for w in res["warnings"])
    assert res["plan"]["details"]["objects_after"] == []


async def test_share_update_with_remove_is_destructive(h):
    res = await h.call(
        TOOL,
        {
            "resource": "share",
            "action": "update",
            "name": "s1",
            "spec": {"updates": [{"action": "REMOVE", "data_object": {"name": "main.sales.orders"}}]},
            "dry_run": True,
        },
    )
    assert res["status"] == "dry_run"
    assert "DESTRUCTIVE" in res["safety"]


async def test_update_permissions_grant_and_revoke(h):
    args = {
        "resource": "share",
        "action": "update_permissions",
        "name": "s1",
        "changes": [{"principal": "globex", "add": ["SELECT"]}],
    }
    res = await h.call(TOOL, args)
    assert res["status"] == "confirmation_required"
    assert "DESTRUCTIVE" not in res["safety"]
    assert any("globex" in w and "main.sales.orders" in w for w in res["warnings"])

    h.w.shares.update_permissions.return_value = sharing.UpdateSharePermissionsResponse()
    await h.call(TOOL, {**args, "confirm": True})
    change = h.w.shares.update_permissions.call_args.kwargs["changes"][0]
    assert isinstance(change, sharing.PermissionsChange)
    assert (change.principal, change.add) == ("globex", ["SELECT"])

    res = await h.call(TOOL, {**args, "changes": [{"principal": "acme", "remove": ["SELECT"]}]})
    assert "DESTRUCTIVE" in res["safety"]


async def test_share_get_and_permissions(h):
    res = await h.call(TOOL, {"resource": "share", "action": "get", "name": "s1"})
    assert res["data"]["objects"][0]["name"] == "main.sales.orders"
    h.w.shares.get.assert_called_with("s1", include_shared_data=True)
    res = await h.call(TOOL, {"resource": "share", "action": "get_permissions", "name": "s1"})
    assert res["data"]["privilege_assignments"] == [{"principal": "acme", "privileges": ["SELECT"]}]


async def test_delete_share_confirmation(h):
    res = await h.call(TOOL, {"resource": "share", "action": "delete", "name": "s1"})
    assert res["status"] == "confirmation_required"
    assert res["plan"]["reversible"] is False
    h.w.shares.delete.assert_not_called()
    await h.call(TOOL, {"resource": "share", "action": "delete", "name": "s1", "confirm": True})
    h.w.shares.delete.assert_called_once_with("s1")


async def test_provider_credentials_never_returned(h):
    h.w.providers.get.return_value = sharing.ProviderInfo(
        name="p1",
        authentication_type=sharing.AuthenticationType.TOKEN,
        recipient_profile_str=PROFILE,
        recipient_profile=sharing.RecipientProfile(bearer_token="PROVIDER-BEARER-SECRET", endpoint="https://x"),
    )
    res = await h.call(TOOL, {"resource": "provider", "action": "get", "name": "p1"})
    _no_secrets(res)
    assert "recipient_profile" not in res["data"]

    args = {
        "resource": "provider",
        "action": "create",
        "name": "p2",
        "spec": {"authentication_type": "TOKEN", "recipient_profile_str": PROFILE},
    }
    res = await h.call(TOOL, args)
    assert res["status"] == "confirmation_required"
    _no_secrets(res)
    h.w.providers.create.return_value = h.w.providers.get.return_value
    res = await h.call(TOOL, {**args, "confirm": True})
    _no_secrets(res)
    assert h.w.providers.create.call_args.kwargs["recipient_profile_str"] == PROFILE


async def test_invalid_resource_action_combination(h):
    msg = await h.call_error(TOOL, {"resource": "provider", "action": "rotate_token", "name": "p1"})
    assert "[INVALID_PARAMETER]" in msg and "not valid for resource" in msg


async def test_read_only_mode(make_harness):
    h = make_harness(read_only=True, toolsets=("unity_catalog",))
    msg = await h.call_error(
        TOOL, {"resource": "share", "action": "create", "name": "s", "confirm": True}
    )
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.shares.create.assert_not_called()
    h.w.shares.list_shares.return_value = iter([sharing.ShareInfo(name="s1")])
    res = await h.call(TOOL, {"resource": "share", "action": "list"})
    assert res["data"] == [{"name": "s1"}]


async def test_not_found(h):
    h.w.recipients.get.side_effect = NotFound("Recipient 'x' does not exist")
    msg = await h.call_error(TOOL, {"resource": "recipient", "action": "get", "name": "x"})
    assert "[NOT_FOUND]" in msg
