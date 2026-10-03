"""Unity Catalog storage credentials and external locations.

SDK services: ``w.storage_credentials`` (create/get/list/update/delete/validate) and
``w.external_locations`` (create/get/list/update/delete). External locations have no
validate endpoint of their own; they are validated through
``w.storage_credentials.validate`` with the location's credential, url and name.

Secret fields (Azure client secrets, Cloudflare secret access keys, ...) are dropped
from every response and plan.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from databricks.sdk.service import catalog as uc
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE_SECURITY, READ, WRITE_SECURITY
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.tools.unity_catalog.objects import (
    bounded_count,
    check_url_safe,
    field_diff,
    sdk_kwargs,
    strip_secrets,
)
from dbx_mcp.utils.errors import ValidationFailed
from dbx_mcp.utils.serialization import pick

Resource = Literal["storage_credential", "external_location"]

_CREDENTIAL_KINDS = (
    "aws_iam_role",
    "azure_managed_identity",
    "azure_service_principal",
    "databricks_gcp_service_account",
    "cloudflare_api_token",
)
_CREDENTIAL_LIST_FIELDS = ["name", "owner", "comment", "read_only", "isolation_mode", "used_for_managed_storage"]
_LOCATION_LIST_FIELDS = ["name", "url", "credential_name", "owner", "comment", "read_only", "isolation_mode", "fallback"]


def _credential_summary(info: Any) -> dict[str, Any]:
    data = strip_secrets(info)
    out = pick(data, _CREDENTIAL_LIST_FIELDS)
    for kind in _CREDENTIAL_KINDS:
        if data.get(kind) is not None:
            out["credential_kind"] = kind
            out[kind] = data[kind]
            break
    return out


def _get(resource: str, name: str) -> Any:
    w = ctx().w
    if resource == "storage_credential":
        return w.storage_credentials.get(name)
    return w.external_locations.get(name)


def _name(name: str | None, action: str) -> str:
    return check_url_safe(require(name, "name", action).strip(), "name")


def _storage_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in ("create", "update", "delete"):
        return None
    resource = args.get("resource")
    label = resource.replace("_", " ") if resource else "resource"
    name = _name(args.get("name"), action)
    spec = args.get("spec") or {}
    force = bool(args.get("force"))
    target = {"resource": resource, "name": name}
    warnings: list[str] = []

    if action == "create":
        if spec.get("skip_validation"):
            warnings.append("skip_validation=true: the cloud credential/path is not checked before creation.")
        if resource == "storage_credential":
            warnings.append("A storage credential grants Databricks access to cloud storage; control who can use it.")
        else:
            warnings.append(
                f"Principals granted privileges on this external location can read/write files under {spec.get('url')!r}."
            )
        return PlanInfo(
            description=f"Create {label} {name!r}.",
            target=target,
            details={"spec": strip_secrets(spec)},
            warnings=warnings,
            reversible=True,
        )

    current = strip_secrets(_get(resource, name))
    if action == "update":
        if not spec and not force:
            raise ValidationFailed("update requires a non-empty spec")
        if "owner" in spec:
            warnings.append(f"Ownership moves from {current.get('owner')!r} to {spec['owner']!r}.")
        for key, message in (
            ("url", "Changing the URL changes which storage path every dependent table/volume points at."),
            ("credential_name", "Changing the credential changes the cloud identity used to access the path."),
            ("isolation_mode", "Changing isolation_mode changes which workspaces can use this object."),
            ("new_name", "Renaming breaks references to the old name."),
        ):
            if key in spec:
                warnings.append(message)
        if spec.get("read_only") is False:
            warnings.append("read_only=false allows WRITE access through this object.")
        if force:
            warnings.append("force=true applies the update even if dependent objects exist.")
        return PlanInfo(
            description=f"Update {label} {name!r}: {', '.join(sorted(spec)) or 'force'}.",
            target=target,
            details={"changes": strip_secrets(field_diff(current, spec)), "force": force},
            warnings=warnings,
            reversible=True,
        )

    # delete
    ctx().safety.check_protected(resource, name, operation="delete")
    details: dict[str, Any] = {"force": force, "owner": current.get("owner")}
    if resource == "storage_credential":
        try:
            count, truncated = bounded_count(
                ctx().w.external_locations.list(), predicate=lambda loc: loc.credential_name == name
            )
            details["dependent_external_locations"] = f">{count}" if truncated else count
            if count and not force:
                warnings.append(f"{count} external location(s) use this credential; delete fails unless force=true.")
            elif count:
                warnings.append(f"force=true: {count} external location(s) using this credential will stop working.")
        except Exception as exc:
            warnings.append(f"Could not count dependent external locations: {type(exc).__name__}")
    else:
        details["url"] = current.get("url")
        warnings.append(
            "Tables and volumes defined on this location lose governed access to it"
            + (" (force=true deletes even with dependents)." if force else "; delete fails if dependents exist unless force=true.")
        )
    warnings.append("Files in cloud storage are not deleted.")
    return PlanInfo(
        description=f"Delete {label} {name!r}" + (" with force=true" if force else "") + ".",
        target=target,
        details=details,
        warnings=warnings,
        reversible=False,
    )


def _validation_response(resp: uc.ValidateStorageCredentialResponse) -> dict[str, Any]:
    results = [
        {"operation": getattr(r.operation, "value", r.operation), "result": getattr(r.result, "value", r.result), "message": r.message}
        for r in resp.results or []
    ]
    return {"is_dir": resp.is_dir, "results": results}


@tool(
    toolset="unity_catalog",
    title="Manage UC storage credentials & external locations",
    safety={
        "get": READ,
        "list": READ,
        "validate": READ,
        "create": WRITE_SECURITY,
        "update": WRITE_SECURITY,
        "delete": DESTRUCTIVE_SECURITY,
    },
    preview=_storage_preview,
)
def manage_uc_storage(
    action: Annotated[
        Literal["create", "get", "list", "update", "delete", "validate"],
        Field(description="create | get | list | update | delete | validate"),
    ],
    resource: Annotated[Resource, Field(description="storage_credential | external_location")],
    name: Annotated[str | None, Field(description="Name of the storage credential / external location.")] = None,
    spec: Spec = None,
    url: Annotated[str | None, Field(description="validate (storage_credential): cloud URL to test access against.")] = None,
    force: Annotated[bool, Field(description="delete/update: proceed even if dependent objects exist.")] = False,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Unity Catalog storage credentials and external locations.

    create/update use `spec` with Databricks API fields - storage_credential: aws_iam_role {role_arn},
    azure_managed_identity {access_connector_id}, databricks_gcp_service_account {}, comment, read_only,
    skip_validation (update also owner, new_name, isolation_mode); external_location: url, credential_name,
    comment, read_only, skip_validation (update also owner, new_name, isolation_mode). validate tests cloud
    access (storage_credential: with `url` or spec.external_location_name; external_location: its own url).
    All changes are SECURITY_SENSITIVE, delete is also DESTRUCTIVE. Secret fields are never returned."""
    c = ctx()
    w = c.w
    is_cred = resource == "storage_credential"
    label = resource.replace("_", " ")

    if action == "list":
        if is_cred:
            return paged_response("storage credential(s)", w.storage_credentials.list(), page_size, page_token, _credential_summary)
        return paged_response(
            "external location(s)",
            w.external_locations.list(),
            page_size,
            page_token,
            lambda loc: pick(strip_secrets(loc), _LOCATION_LIST_FIELDS),
        )

    if action == "validate":
        if is_cred:
            fixed = {"storage_credential_name": _name(name, action) if name else None, "url": url}
            kwargs = sdk_kwargs(uc.StorageCredentialsAPI.validate, spec, fixed=fixed)
            if not (kwargs.get("url") or kwargs.get("external_location_name")):
                raise ValidationFailed("validate needs `url` or spec.external_location_name to test against")
            if not (kwargs.get("storage_credential_name") or any(kwargs.get(k) for k in _CREDENTIAL_KINDS)):
                raise ValidationFailed("validate needs `name` (an existing credential) or a cloud credential in spec")
            resp = w.storage_credentials.validate(**kwargs)
            subject = kwargs.get("storage_credential_name") or "inline credential"
        else:
            loc_name = _name(name, action)
            if spec:
                raise ValidationFailed("validate for an external_location takes only `name`")
            loc = w.external_locations.get(loc_name)
            resp = w.storage_credentials.validate(
                storage_credential_name=loc.credential_name, external_location_name=loc_name, url=loc.url
            )
            subject = loc_name
        data = _validation_response(resp)
        failed = [r for r in data["results"] if r["result"] == "FAIL"]
        summary = f"Validated {label} {subject}: {len(data['results']) - len(failed)} check(s) passed/skipped, {len(failed)} failed."
        return ok(summary, data)

    target = _name(name, action)
    if force and action not in ("update", "delete"):
        raise ValidationFailed("force is only valid for update and delete")

    if action == "get":
        return ok(f"Retrieved {label} {target}.", strip_secrets(_get(resource, target)))

    if action == "create":
        if is_cred:
            created = w.storage_credentials.create(**sdk_kwargs(uc.StorageCredentialsAPI.create, spec, fixed={"name": target}))
        else:
            created = w.external_locations.create(**sdk_kwargs(uc.ExternalLocationsAPI.create, spec, fixed={"name": target}))
        note = c.manifest.safe_track(
            resource_type=f"uc_{resource}",
            resource_id=target,
            name=target,
            created_by_tool="manage_uc_storage",
            workspace_host=c.host,
        )
        return ok(f"Created {label} {target}.", strip_secrets(created), warnings=[note or ""])

    if action == "update":
        if not spec and not force:
            raise ValidationFailed("update requires a non-empty spec")
        fixed = {"name": target, "force": True if force else None}
        api = uc.StorageCredentialsAPI.update if is_cred else uc.ExternalLocationsAPI.update
        method = w.storage_credentials.update if is_cred else w.external_locations.update
        updated = method(**sdk_kwargs(api, spec, fixed=fixed))
        return ok(f"Updated {label} {target}.", strip_secrets(updated))

    # delete
    c.safety.check_protected(resource, target, operation="delete")
    if is_cred:
        w.storage_credentials.delete(target, force=True if force else None)
    else:
        w.external_locations.delete(target, force=True if force else None)
    c.manifest.safe_untrack(f"uc_{resource}", target)
    return ok(
        f"Deleted {label} {target}" + (" (force=true)" if force else "") + ".",
        {"resource": resource, "name": target, "deleted": True, "force": force},
    )
