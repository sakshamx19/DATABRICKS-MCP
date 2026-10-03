"""manage_uc_grants: direct/effective grants, grant, revoke."""

from __future__ import annotations

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    EffectivePermissionsList,
    EffectivePrivilege,
    EffectivePrivilegeAssignment,
    GetPermissionsResponse,
    PermissionsChange,
    Privilege,
    PrivilegeAssignment,
    SecurableType,
    UpdatePermissionsResponse,
)

TOOL = "manage_uc_grants"


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=("unity_catalog",))


def _perms(principal: str, *privs: Privilege) -> GetPermissionsResponse:
    return GetPermissionsResponse(privilege_assignments=[PrivilegeAssignment(principal=principal, privileges=list(privs))])


async def test_get_direct_grants(h):
    h.w.grants.get.return_value = GetPermissionsResponse(
        privilege_assignments=[
            PrivilegeAssignment(principal="analysts", privileges=[Privilege.SELECT]),
            PrivilegeAssignment(principal="eng", privileges=[Privilege.MODIFY, Privilege.SELECT]),
        ]
    )
    res = await h.call(TOOL, {"action": "get", "securable_type": "TABLE", "full_name": "main.s.t"})
    assert res["status"] == "success"
    assert res["data"][1] == {"principal": "eng", "privileges": ["MODIFY", "SELECT"]}
    h.w.grants.get.assert_called_once_with("table", "main.s.t", principal=None)


async def test_view_alias_and_backticks(h):
    h.w.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])
    await h.call(TOOL, {"action": "get", "securable_type": "view", "full_name": "`main`.`s`.`v`"})
    h.w.grants.get.assert_called_once_with("table", "main.s.v", principal=None)


async def test_get_effective(h):
    h.w.grants.get_effective.return_value = EffectivePermissionsList(
        privilege_assignments=[
            EffectivePrivilegeAssignment(
                principal="analysts",
                privileges=[
                    EffectivePrivilege(
                        privilege=Privilege.USE_CATALOG,
                        inherited_from_type=SecurableType.CATALOG,
                        inherited_from_name="main",
                    )
                ],
            )
        ]
    )
    res = await h.call(TOOL, {"action": "get_effective", "securable_type": "schema", "full_name": "main.s", "principal": "analysts"})
    assert res["data"][0]["privileges"][0]["inherited_from_name"] == "main"
    h.w.grants.get_effective.assert_called_once_with("schema", "main.s", principal="analysts")


async def test_unknown_securable_type(h):
    msg = await h.call_error(TOOL, {"action": "get", "securable_type": "bucket", "full_name": "x"})
    assert "Unknown securable_type" in msg and "external_location" in msg


async def test_get_not_found(h):
    h.w.grants.get.side_effect = NotFound("Table 'main.s.x' does not exist.")
    msg = await h.call_error(TOOL, {"action": "get", "securable_type": "table", "full_name": "main.s.x"})
    assert "[NOT_FOUND]" in msg


async def test_grant_requires_confirmation_with_diff(h):
    h.w.grants.get.return_value = _perms("analysts", Privilege.USE_SCHEMA)
    res = await h.call(
        TOOL,
        {"action": "grant", "securable_type": "schema", "full_name": "main.s", "principal": "analysts",
         "privileges": ["select", "USE SCHEMA"]},
    )
    assert res["status"] == "confirmation_required"
    diff = res["plan"]["details"]["direct_privileges"]
    assert diff["before"] == ["USE_SCHEMA"]
    assert diff["after"] == ["SELECT", "USE_SCHEMA"]
    assert diff["added"] == ["SELECT"]
    assert diff["already_granted"] == ["USE_SCHEMA"]
    h.w.grants.get.assert_called_with("schema", "main.s", principal="analysts")
    h.w.grants.update.assert_not_called()


async def test_grant_confirmed_calls_update(h):
    h.w.grants.get.return_value = _perms("analysts")
    h.w.grants.update.return_value = UpdatePermissionsResponse(
        privilege_assignments=[PrivilegeAssignment(principal="analysts", privileges=[Privilege.SELECT])]
    )
    res = await h.call(
        TOOL,
        {"action": "grant", "securable_type": "table", "full_name": "main.s.t", "principal": "analysts",
         "privileges": ["SELECT"], "confirm": True},
    )
    assert res["status"] == "success"
    assert res["data"]["before"] == [] and res["data"]["after"] == ["SELECT"]
    h.w.grants.update.assert_called_once_with(
        "table", "main.s.t", changes=[PermissionsChange(principal="analysts", add=[Privilege.SELECT])]
    )


async def test_grant_all_privileges_rejected(h):
    msg = await h.call_error(
        TOOL,
        {"action": "grant", "securable_type": "catalog", "full_name": "main", "principal": "eng",
         "privileges": ["ALL PRIVILEGES"], "confirm": True},
    )
    assert "ALL_PRIVILEGES" in msg and "allow_all_privileges" in msg
    h.w.grants.update.assert_not_called()


async def test_grant_all_privileges_with_opt_in_warns(h):
    h.w.grants.get.return_value = _perms("eng")
    res = await h.call(
        TOOL,
        {"action": "grant", "securable_type": "catalog", "full_name": "main", "principal": "eng",
         "privileges": ["ALL_PRIVILEGES"], "allow_all_privileges": True},
    )
    assert res["status"] == "confirmation_required"
    assert any("every current and future privilege" in w for w in res["plan"]["warnings"])


async def test_grant_to_account_users_warns(h):
    h.w.grants.get.return_value = _perms("account users")
    res = await h.call(
        TOOL,
        {"action": "grant", "securable_type": "table", "full_name": "main.s.t", "principal": "account users",
         "privileges": ["SELECT"], "dry_run": True},
    )
    assert res["status"] == "dry_run"
    assert any("ALL users" in w for w in res["warnings"])
    h.w.grants.update.assert_not_called()


async def test_invalid_privilege(h):
    msg = await h.call_error(
        TOOL, {"action": "grant", "securable_type": "table", "full_name": "a.b.c", "principal": "x", "privileges": ["READ_ALL"]}
    )
    assert "Unknown privilege" in msg


async def test_grant_requires_principal(h):
    msg = await h.call_error(
        TOOL, {"action": "grant", "securable_type": "table", "full_name": "a.b.c", "privileges": ["SELECT"]}
    )
    assert "principal" in msg


async def test_revoke_without_confirm_not_executed(h):
    h.w.grants.get.return_value = _perms("analysts", Privilege.SELECT, Privilege.MODIFY)
    h.w.grants.get_effective.return_value = EffectivePermissionsList(
        privilege_assignments=[
            EffectivePrivilegeAssignment(
                principal="analysts",
                privileges=[
                    EffectivePrivilege(privilege=Privilege.SELECT, inherited_from_type=SecurableType.SCHEMA,
                                       inherited_from_name="main.s"),
                ],
            )
        ]
    )
    res = await h.call(
        TOOL,
        {"action": "revoke", "securable_type": "table", "full_name": "main.s.t", "principal": "analysts",
         "privileges": ["SELECT"]},
    )
    assert res["status"] == "confirmation_required"
    assert set(res["safety"]) == {"DESTRUCTIVE", "SECURITY_SENSITIVE"}
    diff = res["plan"]["details"]["direct_privileges"]
    assert diff["before"] == ["MODIFY", "SELECT"] and diff["after"] == ["MODIFY"] and diff["removed"] == ["SELECT"]
    assert any("inherited from SCHEMA 'main.s'" in w for w in res["plan"]["warnings"])
    h.w.grants.update.assert_not_called()


async def test_revoke_confirmed(h):
    h.w.grants.get.return_value = _perms("analysts", Privilege.SELECT)
    h.w.grants.update.return_value = UpdatePermissionsResponse(privilege_assignments=[])
    res = await h.call(
        TOOL,
        {"action": "revoke", "securable_type": "table", "full_name": "main.s.t", "principal": "analysts",
         "privileges": ["SELECT"], "confirm": True},
    )
    assert res["status"] == "success"
    h.w.grants.update.assert_called_once_with(
        "table", "main.s.t", changes=[PermissionsChange(principal="analysts", remove=[Privilege.SELECT])]
    )


async def test_revoke_on_protected_blocked(h):
    msg = await h.call_error(
        TOOL,
        {"action": "revoke", "securable_type": "catalog", "full_name": "prod", "principal": "a",
         "privileges": ["USE_CATALOG"], "confirm": True},
    )
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.grants.update.assert_not_called()


async def test_read_only_blocks_grant_allows_get(make_harness):
    h = make_harness(read_only=True, toolsets=("unity_catalog",))
    msg = await h.call_error(
        TOOL,
        {"action": "grant", "securable_type": "table", "full_name": "a.b.c", "principal": "x", "privileges": ["SELECT"],
         "confirm": True},
    )
    assert "read-only" in msg
    h.w.grants.update.assert_not_called()
    h.w.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])
    res = await h.call(TOOL, {"action": "get", "securable_type": "table", "full_name": "a.b.c"})
    assert res["status"] == "success"
