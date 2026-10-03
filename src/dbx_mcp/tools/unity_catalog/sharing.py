"""Delta Sharing: manage_uc_sharing (shares, recipients, providers, share permissions).

SDK services: ``w.shares`` (SharesAPI), ``w.recipients`` (RecipientsAPI),
``w.providers`` (ProvidersAPI). Request bodies are validated against the SDK method
signatures (``coerce_kwargs`` on the API class) before anything is sent.

Every non-read action is SECURITY_SENSITIVE (sharing exposes data outside the
metastore) and needs confirmation; removals, revocations, deletes and token rotation
are also DESTRUCTIVE.

Secrets: recipient activation links / bearer tokens and provider credential files are
NEVER returned. They are stripped explicitly here (``tokens`` -> non-secret
``token_metadata``; ``activation_url``, ``sharing_code``, ``recipient_profile_str``,
``recipient_profile`` removed), in addition to the server-wide redaction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from databricks.sdk.service import sharing
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import (
    DESTRUCTIVE_SECURITY,
    READ,
    READ_SECURITY,
    WRITE,
    WRITE_SECURITY,
    SafetyLevel,
)
from dbx_mcp.server.context import AppContext
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.tools.unity_catalog.security_policies import audit_block
from dbx_mcp.utils.errors import ValidationFailed
from dbx_mcp.utils.serialization import coerce_kwargs, parse_sdk_object, pick, to_jsonable

Levels = frozenset[SafetyLevel]

_ACTIONS: dict[str, tuple[str, ...]] = {
    "share": (
        "list", "get", "create", "update", "delete", "add_objects", "remove_objects",
        "get_permissions", "update_permissions",
    ),
    "recipient": ("list", "get", "create", "update", "delete", "get_permissions", "rotate_token"),
    "provider": ("list", "get", "create", "update", "delete", "list_shares"),
}
_READS = {"list", "get", "list_shares"}
_DESTRUCTIVE = {"delete", "remove_objects", "rotate_token"}

# Keys whose values grant access to shared data - removed from every response/plan.
_STRIP_KEYS = {"activation_url", "sharing_code", "recipient_profile_str", "recipient_profile", "bearer_token"}
_TOKEN_META_KEYS = ("id", "created_at", "created_by", "expiration_time", "updated_at", "updated_by")


# ----------------------------------------------------------------------------------------------
# Safety classification
# ----------------------------------------------------------------------------------------------

def _validate_combo(resource: Any, action: Any) -> None:
    if resource not in _ACTIONS:
        raise ValidationFailed(f"resource must be one of {', '.join(_ACTIONS)}, got {resource!r}")
    if action not in _ACTIONS[resource]:
        raise ValidationFailed(
            f"Action {action!r} is not valid for resource {resource!r}. Valid: {', '.join(_ACTIONS[resource])}"
        )


def _spec_removes_objects(spec: Any) -> bool:
    updates = (spec or {}).get("updates") if isinstance(spec, dict) else None
    return any(isinstance(u, dict) and str(u.get("action", "")).upper() == "REMOVE" for u in updates or [])


def _changes_revoke(changes: Any) -> bool:
    return any(isinstance(ch, dict) and ch.get("remove") for ch in changes or [])


def _levels(args: dict[str, Any]) -> Levels:
    resource, action = args.get("resource"), args.get("action")
    _validate_combo(resource, action)
    if action in _READS:
        return READ
    if action == "get_permissions":
        return READ_SECURITY
    if action in _DESTRUCTIVE:
        return WRITE | DESTRUCTIVE_SECURITY
    if action == "update_permissions" and _changes_revoke(args.get("changes")):
        return WRITE | DESTRUCTIVE_SECURITY
    if action == "update" and resource == "share" and _spec_removes_objects(args.get("spec")):
        return WRITE | DESTRUCTIVE_SECURITY
    return WRITE_SECURITY


# ----------------------------------------------------------------------------------------------
# Output sanitization
# ----------------------------------------------------------------------------------------------

def sanitize(obj: Any) -> Any:
    """JSON-convert and strip activation links, tokens and credential files."""
    data = to_jsonable(obj)

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if key in _STRIP_KEYS:
                    continue
                if key == "tokens" and isinstance(item, list):
                    out["token_metadata"] = [
                        {k: t[k] for k in _TOKEN_META_KEYS if isinstance(t, dict) and t.get(k) is not None}
                        for t in item
                    ]
                    continue
                out[key] = walk(item)
            return out
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk(data)


def _mask_spec(spec: dict[str, Any] | None) -> dict[str, Any]:
    masked = dict(spec or {})
    for key in ("recipient_profile_str", "sharing_code"):
        if masked.get(key):
            masked[key] = "***provided (not shown)***"
    return masked


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

_API_CLASS = {"share": sharing.SharesAPI, "recipient": sharing.RecipientsAPI, "provider": sharing.ProvidersAPI}


def _service(c: AppContext, resource: str) -> Any:
    return {"share": c.w.shares, "recipient": c.w.recipients, "provider": c.w.providers}[resource]


def _kwargs(resource: str, method: str, spec: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """Validate spec against the real SDK method signature and convert to SDK types."""
    fn = getattr(_API_CLASS[resource], method)
    return coerce_kwargs(fn, spec, fixed={"name": name})


def _collect(fetch: Callable[..., Any], attr: str, max_pages: int = 50) -> list[Any]:
    items: list[Any] = []
    token: str | None = None
    for _ in range(max_pages):
        resp = fetch(page_token=token) if token else fetch()
        items.extend(getattr(resp, attr, None) or [])
        token = getattr(resp, "next_page_token", None)
        if not isinstance(token, str) or not token:
            break
    return items


def _share_recipients(c: AppContext, share: str) -> list[dict[str, Any]]:
    assignments = _collect(lambda **kw: c.w.shares.share_permissions(share, **kw), "privilege_assignments")
    return [sanitize(a) for a in assignments]


def _share_object_names(c: AppContext, share: str) -> list[str]:
    info = c.w.shares.get(share, include_shared_data=True)
    return [o.name for o in info.objects or [] if o.name]


def _object_updates(objects: list[Any] | None, update_action: str, op: str) -> list[dict[str, Any]]:
    if not objects:
        raise ValidationFailed(f"Parameter 'objects' (non-empty list) is required for {op}")
    updates = []
    for i, obj in enumerate(objects):
        data_object = {"name": obj} if isinstance(obj, str) else dict(obj)
        if not isinstance(data_object.get("name"), str) or not data_object["name"].strip():
            raise ValidationFailed(f"objects[{i}].name is required (e.g. catalog.schema.table)")
        updates.append({"action": update_action, "data_object": data_object})
    return updates


def _changes(changes: list[dict[str, Any]] | None) -> list[sharing.PermissionsChange]:
    if not changes:
        raise ValidationFailed("Parameter 'changes' is required: [{principal, add: [...], remove: [...]}]")
    parsed = []
    for i, ch in enumerate(changes):
        if not isinstance(ch, dict) or not ch.get("principal"):
            raise ValidationFailed(f"changes[{i}].principal (recipient name) is required")
        if not ch.get("add") and not ch.get("remove"):
            raise ValidationFailed(f"changes[{i}] must contain 'add' and/or 'remove' privileges")
        parsed.append(parse_sdk_object(sharing.PermissionsChange, ch, f"changes[{i}]"))
    return parsed


def _summary_transform(resource: str) -> Callable[[Any], dict[str, Any]]:
    keys = {
        "share": ["name", "owner", "comment", "storage_root", "created_at", "updated_at"],
        "recipient": [
            "name", "authentication_type", "activated", "owner", "comment", "cloud", "region",
            "data_recipient_global_metastore_id", "created_at",
        ],
        "provider": [
            "name", "authentication_type", "owner", "comment", "cloud", "region",
            "data_provider_global_metastore_id", "created_at",
        ],
    }[resource]
    return lambda item: pick(sanitize(item), keys)


# ----------------------------------------------------------------------------------------------
# Preview
# ----------------------------------------------------------------------------------------------

def _exposure_warning(objects: list[str], recipients: list[dict[str, Any]]) -> str:
    who = [r.get("principal") for r in recipients if r.get("principal")]
    if who:
        return (
            f"EXTERNAL DATA EXPOSURE: {objects} will become readable by share recipients {who} "
            "(and by any recipient granted access to this share later)."
        )
    return (
        f"EXTERNAL DATA EXPOSURE: {objects} will be added to the share. No recipient currently has access, "
        "but any recipient granted SELECT on this share will be able to read them."
    )


def _preview(args: dict[str, Any]) -> PlanInfo | None:
    resource, action = args.get("resource"), args.get("action")
    _validate_combo(resource, action)
    if action in _READS or action == "get_permissions":
        return None
    c = ctx()
    name = require(args.get("name"), "name", action)
    spec = args.get("spec")
    target = {"resource": resource, "name": name}

    if action == "create":
        kwargs = _kwargs(resource, "create", spec, name)  # validates before showing the plan
        warnings = []
        if resource == "recipient":
            warnings.append(
                "Creating a recipient enables sharing data outside this metastore. For TOKEN recipients the "
                "activation link is NOT returned by this tool; retrieve it in Catalog Explorer."
            )
        if resource == "provider":
            warnings.append("Creating a provider lets this metastore read data shared by an external party.")
        return PlanInfo(
            description=f"Create {resource} {name!r}.",
            target=target,
            details={"request": _mask_spec({k: to_jsonable(v) for k, v in kwargs.items()})},
            warnings=warnings,
            reversible=True,
        )

    if action in {"delete", "remove_objects", "rotate_token"}:
        c.safety.check_protected(resource, name, None, operation=action.replace("_", " ") + " on")

    if resource == "share":
        current = sanitize(c.w.shares.get(name, include_shared_data=True))
        recipients = _share_recipients(c, name)
        current_objects = [o.get("name") for o in current.get("objects") or []]
        details: dict[str, Any] = {"current_objects": current_objects, "current_recipients": recipients}
        warnings: list[str] = []
        updates: list[dict[str, Any]] = []
        if action == "add_objects":
            updates = _object_updates(args.get("objects"), "ADD", action)
        elif action == "remove_objects":
            updates = _object_updates(args.get("objects"), "REMOVE", action)
        elif action == "update":
            updates = list((spec or {}).get("updates") or [])
            _kwargs("share", "update", spec, name)
            details["changes"] = {
                k: {"current": current.get(k), "new": v} for k, v in (spec or {}).items() if k != "updates"
            }
        if updates:
            details["object_updates"] = updates
            added = [u["data_object"].get("name") for u in updates if str(u.get("action")).upper() == "ADD"]
            removed = [u["data_object"].get("name") for u in updates if str(u.get("action")).upper() == "REMOVE"]
            if added:
                warnings.append(_exposure_warning(added, recipients))
            if removed:
                warnings.append(f"Recipients will immediately lose access to {removed}.")
            details["objects_after"] = [o for o in current_objects if o not in removed] + [
                a for a in added if a not in current_objects
            ]
        if action == "delete":
            warnings.append(
                f"Deleting the share revokes access for all recipients {[r.get('principal') for r in recipients]} "
                f"to {current_objects}. This cannot be undone."
            )
        if action == "update_permissions":
            changes = [to_jsonable(ch) for ch in _changes(args.get("changes"))]
            details["permission_changes"] = changes
            for ch in changes:
                if ch.get("add"):
                    warnings.append(
                        f"EXTERNAL DATA EXPOSURE: recipient {ch['principal']!r} gains {ch['add']} on share "
                        f"{name!r}, which contains {current_objects}."
                    )
                if ch.get("remove"):
                    warnings.append(f"Recipient {ch['principal']!r} loses {ch['remove']} on share {name!r}.")
        reversible = action not in {"delete"}
        return PlanInfo(
            description=f"{action.replace('_', ' ').capitalize()} on share {name!r}.",
            target=target,
            details=details,
            warnings=warnings,
            reversible=reversible,
        )

    if resource == "recipient":
        current = sanitize(c.w.recipients.get(name))
        details = {"current": current}
        warnings = []
        if action == "update":
            _kwargs("recipient", "update", spec, name)
            details["changes"] = {k: {"current": current.get(k), "new": v} for k, v in (spec or {}).items()}
        elif action == "delete":
            shares = _collect(lambda **kw: c.w.recipients.share_permissions(name, **kw), "permissions_out")
            details["current_share_access"] = [sanitize(s) for s in shares]
            warnings.append(
                f"Recipient {name!r} immediately loses access to all shares "
                f"{[getattr(s, 'share_name', None) for s in shares]}. This cannot be undone."
            )
        elif action == "rotate_token":
            seconds = args.get("existing_token_expire_in_seconds")
            if seconds is None:
                raise ValidationFailed("existing_token_expire_in_seconds is required for rotate_token (0 = expire now)")
            warnings.append(
                f"The recipient's existing token will expire in {seconds} second(s); the recipient must use the "
                "new activation link (retrieve it in Catalog Explorer - it is never returned by this tool)."
            )
            details["existing_token_expire_in_seconds"] = seconds
        return PlanInfo(
            description=f"{action.replace('_', ' ').capitalize()} recipient {name!r}.",
            target=target,
            details=details,
            warnings=warnings,
            reversible=action == "update",
        )

    # provider
    current = sanitize(c.w.providers.get(name))
    details = {"current": current}
    warnings = []
    if action == "update":
        _kwargs("provider", "update", spec, name)
        details["changes"] = {
            k: {"current": current.get(k), "new": v} for k, v in _mask_spec(spec).items()
        }
    else:
        warnings.append(
            f"Deleting provider {name!r} breaks catalogs created from its shares in this metastore. "
            "This cannot be undone without the provider's credential."
        )
    return PlanInfo(
        description=f"{action.capitalize()} provider {name!r}.",
        target=target,
        details=details,
        warnings=warnings,
        reversible=action == "update",
    )


# ----------------------------------------------------------------------------------------------
# Tool
# ----------------------------------------------------------------------------------------------

@tool(
    toolset="unity_catalog",
    title="Delta Sharing: shares, recipients, providers",
    safety=_levels,
    possible_levels=READ | WRITE | DESTRUCTIVE_SECURITY,
    preview=_preview,
)
def manage_uc_sharing(
    resource: Annotated[Literal["share", "recipient", "provider"], Field(description="Delta Sharing object type.")],
    action: Annotated[
        Literal[
            "list", "get", "create", "update", "delete", "add_objects", "remove_objects",
            "get_permissions", "update_permissions", "rotate_token", "list_shares",
        ],
        Field(
            description=(
                "All: list, get, create, update, delete. share: add_objects, remove_objects, get_permissions "
                "(recipients with access), update_permissions. recipient: get_permissions (shares it can "
                "access), rotate_token. provider: list_shares."
            )
        ),
    ],
    name: Annotated[str | None, Field(description="Share / recipient / provider name.")] = None,
    spec: Spec = None,
    objects: Annotated[
        list[dict[str, Any] | str] | None,
        Field(
            description=(
                "add_objects/remove_objects: data objects, e.g. {name: 'cat.sch.tbl', data_object_type: 'TABLE', "
                "shared_as?, cdf_enabled?, history_data_sharing_status?, partitions?, comment?}; "
                "remove_objects also accepts plain names."
            )
        ),
    ] = None,
    changes: Annotated[
        list[dict[str, Any]] | None,
        Field(description="update_permissions: [{principal: <recipient>, add: ['SELECT'], remove: [...]}]."),
    ] = None,
    existing_token_expire_in_seconds: Annotated[
        int | None,
        Field(description="rotate_token: seconds until the current token expires (0 = immediately).", ge=0),
    ] = None,
    include_shared_data: Annotated[bool, Field(description="share get: include the shared objects.")] = True,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Delta Sharing shares, recipients and providers.

    - share: list | get(name) | create(name, spec{comment, storage_root}) | update(name, spec{comment,
      new_name, owner, storage_root, updates}) | delete | add_objects / remove_objects(name, objects) |
      get_permissions(name) | update_permissions(name, changes=[{principal, add, remove}]).
    - recipient: list | get | create(name, spec{authentication_type: TOKEN|DATABRICKS|OIDC_FEDERATION|...,
      data_recipient_global_metastore_id, comment, ip_access_list, expiration_time, owner,
      properties_kvpairs}) | update | delete | get_permissions (shares it can read) |
      rotate_token(existing_token_expire_in_seconds).
    - provider: list | get | create(name, spec{authentication_type, recipient_profile_str, comment}) |
      update | delete | list_shares.
    All changes are security-sensitive and need confirm=true after reviewing the plan (adding objects or
    granting recipients is external data exposure). Activation links, tokens and provider credentials
    are never returned."""
    c = ctx()
    _validate_combo(resource, action)
    svc = _service(c, resource)

    if action == "list":
        it = {"share": lambda: c.w.shares.list_shares(), "recipient": lambda: c.w.recipients.list(),
              "provider": lambda: c.w.providers.list()}[resource]()
        return paged_response(f"{resource}s", it, page_size, page_token, _summary_transform(resource))

    name = require(name, "name", action)

    if action == "get":
        item = c.w.shares.get(name, include_shared_data=include_shared_data) if resource == "share" else svc.get(name)
        return ok(f"{resource.capitalize()} {name!r}.", sanitize(item))

    if action == "list_shares":
        it = c.w.providers.list_shares(name)
        return paged_response(f"shares from provider {name!r}", it, page_size, page_token, lambda s: sanitize(s))

    if action == "get_permissions":
        if resource == "share":
            items = _share_recipients(c, name)
            return ok(f"{len(items)} recipient privilege assignment(s) on share {name!r}.", {"privilege_assignments": items})
        items = [sanitize(s) for s in _collect(lambda **kw: c.w.recipients.share_permissions(name, **kw), "permissions_out")]
        return ok(f"Recipient {name!r} has access to {len(items)} share(s).", {"permissions_out": items})

    # ---- changes ------------------------------------------------------------------------------
    audit_what = f"{resource}s.{action} {name}"
    if action == "create":
        result = svc.create(**_kwargs(resource, "create", spec, name))
        steps = []
        if resource == "recipient":
            steps.append(
                "Grant the recipient access with manage_uc_sharing resource=share action=update_permissions. "
                "For TOKEN recipients, share the activation link from Catalog Explorer (never returned here)."
            )
        data = {resource: sanitize(result), "audit": audit_block(c, api_call=f"{resource}s.create {name}")}
        return ok(f"Created {resource} {name!r}.", data, next_steps=steps)

    if action == "update":
        if not spec:
            raise ValidationFailed("spec with the fields to change is required for update")
        result = svc.update(**_kwargs(resource, "update", spec, name))
        return ok(f"Updated {resource} {name!r}.", {resource: sanitize(result), "audit": audit_block(c, api_call=audit_what)})

    if action == "delete":
        c.safety.check_protected(resource, name, None, operation="delete")
        svc.delete(name)
        return ok(f"Deleted {resource} {name!r}.", {"name": name, "audit": audit_block(c, api_call=audit_what)})

    if action in {"add_objects", "remove_objects"}:
        if action == "remove_objects":
            c.safety.check_protected("share", name, None, operation="remove objects from")
        updates = _object_updates(objects, "ADD" if action == "add_objects" else "REMOVE", action)
        result = c.w.shares.update(**_kwargs("share", "update", {"updates": updates}, name))
        data = {
            "share": sanitize(result),
            "object_updates": updates,
            "audit": audit_block(c, api_call=f"shares.update {name} " + ", ".join(
                f"{u['action']} {u['data_object']['name']}" for u in updates)),
        }
        return ok(f"{'Added' if action == 'add_objects' else 'Removed'} {len(updates)} object(s) on share {name!r}.", data)

    if action == "update_permissions":
        parsed = _changes(changes)
        result = c.w.shares.update_permissions(name, changes=parsed)
        data = {
            "result": sanitize(result),
            "changes": [to_jsonable(ch) for ch in parsed],
            "audit": audit_block(c, api_call=f"shares.update_permissions {name}"),
        }
        return ok(f"Updated recipient permissions on share {name!r}.", data)

    # rotate_token
    if existing_token_expire_in_seconds is None:
        raise ValidationFailed("existing_token_expire_in_seconds is required for rotate_token (0 = expire now)")
    c.safety.check_protected("recipient", name, None, operation="rotate the token of")
    result = c.w.recipients.rotate_token(name, existing_token_expire_in_seconds)
    return ok(
        f"Rotated the token of recipient {name!r}; the previous token expires in "
        f"{existing_token_expire_in_seconds}s.",
        {"recipient": sanitize(result), "audit": audit_block(c, api_call=f"recipients.rotate_token {name}")},
        next_steps=["Send the recipient the new activation link from Catalog Explorer (not returned by this tool)."],
    )
