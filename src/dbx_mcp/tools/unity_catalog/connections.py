"""Lakehouse Federation connections via the SDK ``w.connections`` service.

Connection ``options`` carry credentials (user, password, tokens, private keys, ...).
They are passed through to Databricks on create/update but never echoed: responses
and plans expose only an allowlist of non-secret option keys (host, port, ...),
plus the *names* of the hidden keys.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from databricks.sdk.service import catalog as uc
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE_SECURITY, READ, WRITE_SECURITY
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.tools.unity_catalog.objects import bounded_count, check_url_safe, sdk_kwargs, strip_secrets
from dbx_mcp.utils.errors import ValidationFailed
from dbx_mcp.utils.serialization import pick, to_jsonable

CONNECTION_TYPES: tuple[str, ...] = tuple(m.value for m in uc.ConnectionType if m.value != "UNKNOWN_CONNECTION_TYPE")

# Option keys that are known not to carry credentials. Everything else is hidden.
_SAFE_OPTION_KEYS = {
    "host",
    "port",
    "httppath",
    "database",
    "warehouse",
    "sfrole",
    "role",
    "catalog",
    "schema",
    "trustservercertificate",
    "encrypt",
    "sslmode",
    "authenticationmethod",
    "region",
    "projectid",
    "account",
    "instance",
    "baseurl",
}
_LIST_FIELDS = ["name", "connection_type", "credential_type", "owner", "comment", "read_only", "url", "full_name"]


def _is_safe_option(key: str) -> bool:
    return key.replace("_", "").replace("-", "").lower() in _SAFE_OPTION_KEYS


def safe_connection(info: Any) -> dict[str, Any]:
    """Connection info with only non-secret option keys; hidden key *names* are listed."""
    data = to_jsonable(info) or {}
    options = data.pop("options", None) or {}
    data = strip_secrets(data)
    data["options"] = {k: v for k, v in options.items() if _is_safe_option(k)}
    hidden = sorted(k for k in options if not _is_safe_option(k))
    if hidden:
        data["hidden_option_keys"] = hidden
    return data


def _options(options: dict[str, Any] | None, action: str) -> dict[str, str]:
    """Validate the options map WITHOUT echoing any value in error messages."""
    if options is None:
        raise ValidationFailed(
            f"Parameter 'options' is required for action '{action}'"
            + (" (Databricks requires the full options map on update, including credentials)." if action == "update" else ".")
        )
    out: dict[str, str] = {}
    for key, value in options.items():
        if value is None or isinstance(value, dict | list):
            raise ValidationFailed(f"options[{key!r}] must be a string/number/boolean value")
        out[str(key)] = str(value).lower() if isinstance(value, bool) else str(value)
    return out


def _connection_type(value: str | None) -> str:
    raw = require(value, "connection_type", "create").strip().upper()
    if raw not in CONNECTION_TYPES:
        raise ValidationFailed(f"Unknown connection_type {value!r}. Valid: {', '.join(CONNECTION_TYPES)}")
    return raw


def _name(name: str | None, action: str) -> str:
    return check_url_safe(require(name, "name", action).strip(), "name")


def _option_summary(options: dict[str, Any] | None) -> dict[str, Any]:
    options = options or {}
    return {
        "options": {k: v for k, v in options.items() if _is_safe_option(k)},
        "hidden_option_keys": sorted(k for k in options if not _is_safe_option(k)),
    }


def _connections_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in ("create", "update", "delete"):
        return None
    name = _name(args.get("name"), action)
    spec = strip_secrets(args.get("spec") or {})
    target = {"name": name}
    warnings: list[str] = []

    if action == "create":
        ctype = _connection_type(args.get("connection_type"))
        _options(args.get("options"), action)
        warnings.append(
            "The connection stores credentials in Unity Catalog; anyone granted USE_CONNECTION (or a foreign catalog "
            "on it) can query the external system with those credentials."
        )
        return PlanInfo(
            description=f"Create {ctype} connection {name!r}.",
            target={**target, "connection_type": ctype},
            details={**_option_summary(args.get("options")), "spec": spec},
            warnings=warnings,
            reversible=True,
        )

    current = safe_connection(ctx().w.connections.get(name))
    if action == "update":
        _options(args.get("options"), action)
        if "owner" in spec:
            warnings.append(f"Ownership moves from {current.get('owner')!r} to {spec['owner']!r}.")
        if "new_name" in spec:
            warnings.append("Renaming breaks foreign catalogs and queries that reference the old name.")
        warnings.append("The options map (including credentials) is replaced with the values supplied.")
        return PlanInfo(
            description=f"Update connection {name!r}.",
            target=target,
            details={
                "current": {k: current.get(k) for k in ("connection_type", "owner", "options", "hidden_option_keys")},
                "new": {**_option_summary(args.get("options")), "spec": spec},
            },
            warnings=warnings,
            reversible=False,
        )

    ctx().safety.check_protected("connection", name, operation="delete")
    details: dict[str, Any] = {"connection_type": current.get("connection_type"), "owner": current.get("owner")}
    try:
        count, truncated = bounded_count(ctx().w.catalogs.list(), predicate=lambda cat: cat.connection_name == name)
        details["dependent_foreign_catalogs"] = f">{count}" if truncated else count
        if count:
            warnings.append(f"{count} foreign catalog(s) use this connection and will stop working.")
    except Exception as exc:
        warnings.append(f"Could not count dependent foreign catalogs: {type(exc).__name__}")
    warnings.append("The stored credentials are discarded; recreating the connection requires them again.")
    return PlanInfo(
        description=f"Delete connection {name!r}.",
        target=target,
        details=details,
        warnings=warnings,
        reversible=False,
    )


@tool(
    toolset="unity_catalog",
    title="Manage Lakehouse Federation connections",
    safety={
        "get": READ,
        "list": READ,
        "create": WRITE_SECURITY,
        "update": WRITE_SECURITY,
        "delete": DESTRUCTIVE_SECURITY,
    },
    preview=_connections_preview,
)
def manage_uc_connections(
    action: Annotated[Literal["create", "get", "list", "update", "delete"], Field(description="create | get | list | update | delete")],
    name: Annotated[str | None, Field(description="Connection name (required except for list).")] = None,
    connection_type: Annotated[
        str | None, Field(description="create only: " + ", ".join(CONNECTION_TYPES))
    ] = None,
    options: Annotated[
        dict[str, Any] | None,
        Field(description="Connection options, e.g. {host, port, user, password} (Snowflake also sfWarehouse...). "
              "Required for create and update. Secret values are never returned."),
    ] = None,
    spec: Spec = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Unity Catalog Lakehouse Federation connections (Snowflake, PostgreSQL, MySQL, SQL Server, Redshift,
    BigQuery, Oracle, Teradata, Databricks, ...).

    create needs name, connection_type and options; spec may add comment, properties, read_only. update needs
    the full options map (Databricks replaces it) and spec may set owner/new_name. Credentials in options are
    sent to Databricks but never returned: responses show only non-secret option keys (host, port, ...).
    All changes are SECURITY_SENSITIVE; delete is also DESTRUCTIVE."""
    c = ctx()
    w = c.w

    if action == "list":
        return paged_response(
            "connection(s)",
            w.connections.list(),
            page_size,
            page_token,
            lambda conn: pick(safe_connection(conn), _LIST_FIELDS),
        )

    target = _name(name, action)

    if action == "get":
        return ok(f"Retrieved connection {target}.", safe_connection(w.connections.get(target)))

    if action == "create":
        ctype = _connection_type(connection_type)
        kwargs = sdk_kwargs(
            uc.ConnectionsAPI.create,
            spec,
            fixed={"name": target, "connection_type": ctype, "options": _options(options, action)},
        )
        created = w.connections.create(**kwargs)
        note = c.manifest.safe_track(
            resource_type="uc_connection",
            resource_id=target,
            name=target,
            created_by_tool="manage_uc_connections",
            workspace_host=c.host,
        )
        return ok(f"Created {ctype} connection {target}.", safe_connection(created), warnings=[note or ""])

    if action == "update":
        kwargs = sdk_kwargs(uc.ConnectionsAPI.update, spec, fixed={"name": target, "options": _options(options, action)})
        updated = w.connections.update(**kwargs)
        return ok(f"Updated connection {target}.", safe_connection(updated))

    c.safety.check_protected("connection", target, operation="delete")
    w.connections.delete(target)
    c.manifest.safe_untrack("uc_connection", target)
    return ok(f"Deleted connection {target}.", {"name": target, "deleted": True})
