"""Unity Catalog privileges (grants) via the SDK ``w.grants`` service.

* ``get``           -> ``w.grants.get``            (direct grants only)
* ``get_effective`` -> ``w.grants.get_effective``  (includes inherited privileges)
* ``grant``/``revoke`` -> ``w.grants.update`` with ``PermissionsChange(add=..., remove=...)``

Never broadens access silently: every change is previewed as a before/after diff of
the principal's direct privileges, ``ALL_PRIVILEGES`` needs an explicit opt-in, and
grants to the all-users groups are flagged.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from databricks.sdk.service import catalog as uc
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE_SECURITY, READ_SECURITY, WRITE_SECURITY
from dbx_mcp.safety.validation import split_full_name
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.tools.unity_catalog.objects import check_url_safe, enum_value
from dbx_mcp.utils.errors import ValidationFailed

SECURABLE_TYPES: tuple[str, ...] = tuple(sorted(m.value.lower() for m in uc.SecurableType))
_ALIASES = {"view": "table", "materialized_view": "table", "streaming_table": "table"}
_BROAD_PRINCIPALS = {"account users", "users", "all users"}


def normalize_securable_type(value: str | None) -> str:
    raw = require(value, "securable_type").strip().lower().replace(" ", "_").replace("-", "_")
    raw = _ALIASES.get(raw, raw)
    if raw not in SECURABLE_TYPES:
        raise ValidationFailed(f"Unknown securable_type {value!r}. Valid: {', '.join(SECURABLE_TYPES)} (views use 'table').")
    return raw


def normalize_privileges(values: list[str] | None) -> list[str]:
    if not values:
        raise ValidationFailed("Parameter 'privileges' must list at least one privilege")
    valid = {m.value for m in uc.Privilege}
    out: list[str] = []
    for value in values:
        norm = str(value).strip().upper().replace(" ", "_")
        if norm not in valid:
            raise ValidationFailed(f"Unknown privilege {value!r}. Valid: {', '.join(sorted(valid))}")
        if norm not in out:
            out.append(norm)
    return out


def _full_name(full_name: str | None) -> str:
    name = require(full_name, "full_name").strip()
    # Accept backtick-quoted input; the REST API takes the plain dotted name.
    return ".".join(check_url_safe(p, "full_name") for p in split_full_name(name, parts=(1, 2, 3)))


def _direct_privileges(securable_type: str, full_name: str, principal: str) -> list[str]:
    """Current direct privileges of ``principal`` on the securable."""
    resp = ctx().w.grants.get(securable_type, full_name, principal=principal)
    privileges: set[str] = set()
    for assignment in resp.privilege_assignments or []:
        if (assignment.principal or "").casefold() == principal.casefold():
            privileges.update(enum_value(p) for p in assignment.privileges or [])
    return sorted(privileges)


def _validate_change(args: dict[str, Any]) -> tuple[str, str, str, list[str], list[str]]:
    """Shared validation for grant/revoke (used by both preview and body)."""
    action = args.get("action")
    securable_type = normalize_securable_type(args.get("securable_type"))
    full_name = _full_name(args.get("full_name"))
    principal = require(args.get("principal"), "principal", action).strip()
    privileges = normalize_privileges(args.get("privileges"))
    warnings: list[str] = []
    if action == "grant":
        if "ALL_PRIVILEGES" in privileges and not args.get("allow_all_privileges"):
            raise ValidationFailed(
                "Refusing to grant ALL_PRIVILEGES: it grants every current and future privilege on the securable "
                "and its children.",
                hint="Grant only the specific privileges needed, or pass allow_all_privileges=true after the user "
                "explicitly asked for ALL_PRIVILEGES.",
            )
        if "ALL_PRIVILEGES" in privileges:
            warnings.append("ALL_PRIVILEGES grants every current and future privilege on this securable and its children.")
        if principal.casefold() in _BROAD_PRINCIPALS:
            warnings.append(
                f"Principal {principal!r} is a built-in group containing ALL users; this grants access to everyone."
            )
        if "MANAGE" in privileges:
            warnings.append("MANAGE lets the principal manage privileges and ownership-like operations on the object.")
    return securable_type, full_name, principal, privileges, warnings


def _plan_change(action: str, before: list[str], privileges: list[str]) -> dict[str, Any]:
    if action == "grant":
        changed = [p for p in privileges if p not in before]
        noop = [p for p in privileges if p in before]
        after = sorted(set(before) | set(privileges))
        return {"before": before, "after": after, "added": changed, "already_granted": noop}
    changed = [p for p in privileges if p in before]
    noop = [p for p in privileges if p not in before]
    after = sorted(set(before) - set(privileges))
    return {"before": before, "after": after, "removed": changed, "not_directly_granted": noop}


def _inherited_warnings(securable_type: str, full_name: str, principal: str, privileges: list[str]) -> list[str]:
    try:
        resp = ctx().w.grants.get_effective(securable_type, full_name, principal=principal)
    except Exception as exc:
        return [f"Could not check inherited privileges: {type(exc).__name__}"]
    out = []
    for assignment in resp.privilege_assignments or []:
        if (assignment.principal or "").casefold() != principal.casefold():
            continue
        for eff in assignment.privileges or []:
            priv = enum_value(eff.privilege)
            if eff.inherited_from_name and (priv in privileges or "ALL_PRIVILEGES" in privileges):
                out.append(
                    f"{principal!r} also has {priv} inherited from {enum_value(eff.inherited_from_type)} "
                    f"{eff.inherited_from_name!r}; revoking here does not remove that inherited access."
                )
    return out


def _grants_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in ("grant", "revoke"):
        return None
    securable_type, full_name, principal, privileges, warnings = _validate_change(args)
    if action == "revoke":
        ctx().safety.check_protected(securable_type, full_name, operation="revoke privileges on")
    before = _direct_privileges(securable_type, full_name, principal)
    diff = _plan_change(action, before, privileges)
    if action == "grant" and diff["already_granted"]:
        warnings.append(f"Already granted (no-op): {', '.join(diff['already_granted'])}")
    if action == "revoke":
        if diff["not_directly_granted"]:
            warnings.append(f"Not directly granted (no-op): {', '.join(diff['not_directly_granted'])}")
        warnings.extend(_inherited_warnings(securable_type, full_name, principal, privileges))
    verb = "Grant" if action == "grant" else "Revoke"
    prep = "to" if action == "grant" else "from"
    return PlanInfo(
        description=f"{verb} {', '.join(privileges)} on {securable_type} {full_name!r} {prep} {principal!r}.",
        target={"securable_type": securable_type, "full_name": full_name, "principal": principal},
        details={"privileges": privileges, "direct_privileges": diff},
        warnings=warnings,
        reversible=True,
    )


@tool(
    toolset="unity_catalog",
    title="Manage Unity Catalog grants",
    safety={
        "get": READ_SECURITY,
        "get_effective": READ_SECURITY,
        "grant": WRITE_SECURITY,
        "revoke": DESTRUCTIVE_SECURITY,
    },
    preview=_grants_preview,
)
def manage_uc_grants(
    action: Annotated[
        Literal["get", "get_effective", "grant", "revoke"],
        Field(description="get: direct grants; get_effective: incl. inherited; grant / revoke privileges."),
    ],
    securable_type: Annotated[
        str,
        Field(description="Securable type: " + ", ".join(SECURABLE_TYPES) + ". Views use 'table'."),
    ],
    full_name: Annotated[str, Field(description="Full name of the securable, e.g. 'main.sales.orders' "
                                    "(metastore: the metastore id).")],
    principal: Annotated[str | None, Field(description="User email, group name or service principal application id. "
                                           "Required for grant/revoke; optional filter for get.")] = None,
    privileges: Annotated[list[str] | None, Field(description="Privileges for grant/revoke, e.g. ['SELECT', "
                                                  "'USE_SCHEMA'] (spaces allowed: 'USE CATALOG').")] = None,
    allow_all_privileges: Annotated[bool, Field(description="Must be true to grant ALL_PRIVILEGES.")] = False,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Show, grant and revoke Unity Catalog privileges on any securable (catalog, schema, table/view, volume,
    function, external_location, storage_credential, connection, share, metastore, ...).

    get returns direct grants, get_effective includes privileges inherited from parents. grant/revoke require
    principal + privileges and always show the principal's before/after direct privileges in the plan.
    ALL_PRIVILEGES is rejected unless allow_all_privileges=true; grants to 'account users' are flagged."""
    c = ctx()
    if action in ("get", "get_effective"):
        st = normalize_securable_type(securable_type)
        name = _full_name(full_name)
        if action == "get":
            resp = c.w.grants.get(st, name, principal=principal)
            noun = "direct privilege assignment(s)"
        else:
            resp = c.w.grants.get_effective(st, name, principal=principal)
            noun = "effective privilege assignment(s)"
        return paged_response(f"{noun} on {st} {name}", resp.privilege_assignments or [], page_size, page_token)

    args = {
        "action": action,
        "securable_type": securable_type,
        "full_name": full_name,
        "principal": principal,
        "privileges": privileges,
        "allow_all_privileges": allow_all_privileges,
    }
    st, name, who, privs, warnings = _validate_change(args)
    if action == "revoke":
        c.safety.check_protected(st, name, operation="revoke privileges on")
    before = _direct_privileges(st, name, who)
    enums = [uc.Privilege(p) for p in privs]
    change = (
        uc.PermissionsChange(principal=who, add=enums)
        if action == "grant"
        else uc.PermissionsChange(principal=who, remove=enums)
    )
    resp = c.w.grants.update(st, name, changes=[change])
    after: set[str] = set()
    found = False
    for assignment in resp.privilege_assignments or []:
        if (assignment.principal or "").casefold() == who.casefold():
            found = True
            after.update(enum_value(p) for p in assignment.privileges or [])
    after_list = sorted(after) if (found or resp.privilege_assignments) else _direct_privileges(st, name, who)
    verb = "Granted" if action == "grant" else "Revoked"
    prep = "to" if action == "grant" else "from"
    data = {
        "securable_type": st,
        "full_name": name,
        "principal": who,
        "privileges": privs,
        "before": before,
        "after": after_list,
    }
    return ok(f"{verb} {', '.join(privs)} on {st} {name} {prep} {who}.", data, warnings=warnings)
