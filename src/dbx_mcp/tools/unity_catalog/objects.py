"""Unity Catalog securable objects: catalogs, schemas, tables, volumes and functions.

Implemented with the official SDK services ``w.catalogs``, ``w.schemas``,
``w.tables``, ``w.volumes`` (``read`` is its "get") and ``w.functions``.

Small helpers shared by the other Unity Catalog tool modules live here too
(name parsing, spec validation against the real SDK signature, secret stripping).
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, Any, Literal

from databricks.sdk.service import catalog as uc
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, READ, WRITE, WRITE_SECURITY, SafetyLevel
from dbx_mcp.safety.validation import split_full_name
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import UnsupportedOperation, ValidationFailed
from dbx_mcp.utils.redaction import is_secret_key
from dbx_mcp.utils.serialization import coerce_kwargs, parse_sdk_object, pick, to_jsonable

# ----------------------------------------------------------------------------------------------
# Shared helpers (also imported by grants/storage/connections/tags)
# ----------------------------------------------------------------------------------------------

_URL_UNSAFE = ("/", "?", "#")


def check_url_safe(value: str, what: str) -> str:
    """Names are interpolated into REST paths by the SDK; refuse characters that would alter the URL."""
    if any(ch in value for ch in _URL_UNSAFE):
        raise ValidationFailed(f"{what} {value!r} must not contain '/', '?' or '#'")
    return value


def parse_uc_name(full_name: str, parts: int, what: str) -> list[str]:
    """Split a (possibly backtick-quoted) dotted name into exactly ``parts`` parts, URL-safe."""
    result = split_full_name(full_name, parts=parts)
    for part in result:
        check_url_safe(part, what)
    return result


def sdk_kwargs(
    unbound: Callable[..., Any],
    spec: Mapping[str, Any] | None,
    *,
    fixed: Mapping[str, Any] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate ``spec`` against the *real* SDK method signature (the unbound class function),
    so validation is independent of the client instance (which may be a test double)."""
    return coerce_kwargs(unbound, spec, fixed={k: v for k, v in (fixed or {}).items() if v is not None}, exclude=exclude)


def strip_secrets(value: Any) -> Any:
    """JSON-convert ``value`` and DROP (not mask) every secret-valued key, recursively."""
    data = to_jsonable(value)
    if isinstance(data, dict):
        return {k: strip_secrets(v) for k, v in data.items() if not (isinstance(k, str) and is_secret_key(k))}
    if isinstance(data, list):
        return [strip_secrets(v) for v in data]
    return data


def bounded_count(items: Iterable[Any], limit: int = 1000, predicate: Callable[[Any], bool] | None = None) -> tuple[int, bool]:
    """Count up to ``limit`` items; returns (count, truncated)."""
    count = 0
    for item in itertools.islice(iter(items), limit + 1):
        if predicate is None or predicate(item):
            count += 1
    return min(count, limit), count > limit


def field_diff(current: Mapping[str, Any] | None, changes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    current = current or {}
    return {key: {"before": current.get(key), "after": to_jsonable(value)} for key, value in changes.items()}


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


# ----------------------------------------------------------------------------------------------
# manage_uc_objects
# ----------------------------------------------------------------------------------------------

ObjectType = Literal["catalog", "schema", "table", "volume", "function"]
_PARTS = {"catalog": 1, "schema": 2, "table": 3, "volume": 3, "function": 3}
_SECURITY_UPDATE_FIELDS = {"owner", "isolation_mode"}
_LIST_FIELDS = {
    "catalog": ["name", "catalog_type", "owner", "comment", "isolation_mode", "connection_name", "provider_name", "share_name"],
    "schema": ["full_name", "name", "owner", "comment"],
    "table": ["full_name", "name", "table_type", "data_source_format", "owner", "comment"],
    "volume": ["full_name", "name", "volume_type", "owner", "comment", "storage_location"],
    "function": ["full_name", "name", "full_data_type", "routine_body", "owner", "comment"],
}
_FORCE_SUPPORTED = {"catalog", "schema", "function"}


def _resolve(
    object_type: str,
    full_name: str | None,
    catalog_name: str | None,
    schema_name: str | None,
    name: str | None,
) -> list[str]:
    """Resolve the target object's name parts, enforcing catalog -> schema -> object."""
    n = _PARTS[object_type]
    if full_name:
        parts = parse_uc_name(full_name, n, f"{object_type} name")
        given = [catalog_name, schema_name][: n - 1]
        for supplied, actual, label in zip(given, parts, ("catalog_name", "schema_name"), strict=False):
            if supplied and supplied != actual:
                raise ValidationFailed(f"{label}={supplied!r} conflicts with full_name={full_name!r}")
        return parts
    if object_type == "catalog":
        leaf = name or catalog_name
        if not leaf:
            raise ValidationFailed("Provide full_name (or name) of the catalog")
        return [check_url_safe(leaf, "catalog name")]
    if object_type == "schema":
        leaf = name or schema_name
        if not (catalog_name and leaf):
            raise ValidationFailed("A schema is identified by catalog_name + name, or full_name='catalog.schema'")
        return [check_url_safe(catalog_name, "catalog_name"), check_url_safe(leaf, "schema name")]
    if not (catalog_name and schema_name and name):
        raise ValidationFailed(
            f"A {object_type} is identified by catalog_name + schema_name + name, or full_name='catalog.schema.{object_type}'"
        )
    return [check_url_safe(p, "name") for p in (catalog_name, schema_name, name)]


def _list_parent(object_type: str, catalog_name: str | None, schema_name: str | None, full_name: str | None) -> list[str]:
    """Parent for list: none for catalogs, catalog for schemas, catalog+schema for the rest."""
    depth = _PARTS[object_type] - 1
    if depth == 0:
        return []
    if full_name and not catalog_name:
        return parse_uc_name(full_name, depth, "parent name")
    if depth == 1:
        return [check_url_safe(require(catalog_name, "catalog_name", f"list {object_type}s"), "catalog_name")]
    require(catalog_name, "catalog_name", f"list {object_type}s")
    require(schema_name, "schema_name", f"list {object_type}s")
    return [check_url_safe(catalog_name, "catalog_name"), check_url_safe(schema_name, "schema_name")]  # type: ignore[arg-type]


def _object_levels(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action in ("get", "list"):
        return READ
    if action == "create":
        return WRITE
    if action == "update":
        return WRITE_SECURITY if set(args.get("spec") or {}) & _SECURITY_UPDATE_FIELDS else WRITE
    if action == "delete":
        return DESTRUCTIVE
    raise ValidationFailed(f"Unknown action {action!r}. Valid: create, get, list, update, delete")


def _get(object_type: str, full: str) -> Any:
    w = ctx().w
    if object_type == "catalog":
        return w.catalogs.get(full)
    if object_type == "schema":
        return w.schemas.get(full)
    if object_type == "table":
        return w.tables.get(full)
    if object_type == "volume":
        return w.volumes.read(full)
    return w.functions.get(full)


def _child_counts(object_type: str, parts: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Count children that a forced delete would remove (bounded)."""
    w = ctx().w
    counts: dict[str, Any] = {}
    warnings: list[str] = []

    def record(label: str, fn: Callable[[], tuple[int, bool]]) -> None:
        try:
            count, truncated = fn()
            counts[label] = f">{count}" if truncated else count
        except Exception as exc:  # counting is advisory; never block the plan on it
            warnings.append(f"Could not count {label}: {type(exc).__name__}: {exc}")

    if object_type == "catalog":
        record(
            "schemas",
            lambda: bounded_count(w.schemas.list(catalog_name=parts[0]), predicate=lambda s: s.name != "information_schema"),
        )
    elif object_type == "schema":
        record("tables", lambda: bounded_count(w.tables.list(catalog_name=parts[0], schema_name=parts[1], omit_columns=True, omit_properties=True)))
        record("volumes", lambda: bounded_count(w.volumes.list(catalog_name=parts[0], schema_name=parts[1])))
        record("functions", lambda: bounded_count(w.functions.list(catalog_name=parts[0], schema_name=parts[1])))
    return counts, warnings


def _target(args: dict[str, Any]) -> tuple[str, list[str], str]:
    object_type = require(args.get("object_type"), "object_type")
    parts = _resolve(object_type, args.get("full_name"), args.get("catalog_name"), args.get("schema_name"), args.get("name"))
    return object_type, parts, ".".join(parts)


def _objects_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in ("create", "update", "delete"):
        return None
    object_type, parts, full = _target(args)
    target = {"object_type": object_type, "full_name": full}
    spec = args.get("spec") or {}

    if action == "create":
        return PlanInfo(
            description=f"Create {object_type} {full!r}.",
            target=target,
            details={"spec": strip_secrets(spec)},
            reversible=True,
        )

    if action == "update":
        current = to_jsonable(_get(object_type, full))
        warnings = []
        if "owner" in spec:
            warnings.append(
                f"Ownership of {object_type} {full!r} moves from {current.get('owner')!r} to {spec['owner']!r}. "
                "The owner has full control (including granting privileges); the previous owner keeps only "
                "privileges granted explicitly."
            )
        if "isolation_mode" in spec:
            warnings.append("Changing isolation_mode changes which workspaces can access this catalog.")
        if "new_name" in spec:
            warnings.append("Renaming breaks queries, jobs and views that reference the old name.")
        return PlanInfo(
            description=f"Update {object_type} {full!r}: {', '.join(sorted(spec)) or 'no fields'}.",
            target=target,
            details={"changes": field_diff(current, spec)},
            warnings=warnings,
            reversible=True,
        )

    # delete
    force = bool(args.get("force"))
    if force and object_type not in _FORCE_SUPPORTED:
        raise ValidationFailed("force is only valid for action=delete on a catalog, schema or function")
    obj = to_jsonable(_get(object_type, full))
    ctx().safety.check_protected(object_type, full, obj.get("properties") if isinstance(obj.get("properties"), dict) else None, operation="delete")
    counts, warnings = _child_counts(object_type, parts)
    details: dict[str, Any] = {"force": force, "owner": obj.get("owner")}
    if counts:
        details["contained_objects"] = counts
    non_empty = any(v not in (0, None) for v in counts.values())
    if object_type in ("catalog", "schema"):
        if force:
            warnings.insert(
                0,
                f"FORCE DELETE: {object_type} {full!r} and EVERYTHING inside it ({counts or 'unknown contents'}) "
                "will be dropped, including data of managed tables and volumes.",
            )
        elif non_empty:
            warnings.append(f"{object_type} {full!r} is not empty; the delete will fail unless force=true.")
    if object_type == "table":
        details["table_type"] = obj.get("table_type")
        if obj.get("table_type") == "MANAGED":
            warnings.append(
                "Deleting a MANAGED table removes its data. It may be recoverable with UNDROP TABLE within the "
                "retention period; otherwise the data is lost."
            )
        else:
            warnings.append("The table definition is removed; for EXTERNAL tables the underlying files are not deleted.")
    if object_type == "volume" and obj.get("volume_type") == "MANAGED":
        warnings.append("Deleting a MANAGED volume deletes all files stored in it.")
    return PlanInfo(
        description=f"Delete {object_type} {full!r}" + (" with force=true (recursive)" if force else "") + ".",
        target=target,
        details=details,
        warnings=warnings,
        reversible=False,
    )


def _create(object_type: str, parts: list[str], spec: dict[str, Any]) -> tuple[Any, list[str]]:
    w = ctx().w
    warnings: list[str] = []
    if object_type == "catalog":
        return w.catalogs.create(**sdk_kwargs(uc.CatalogsAPI.create, spec, fixed={"name": parts[0]})), warnings
    if object_type == "schema":
        kwargs = sdk_kwargs(uc.SchemasAPI.create, spec, fixed={"catalog_name": parts[0], "name": parts[1]})
        return w.schemas.create(**kwargs), warnings
    fixed = {"catalog_name": parts[0], "schema_name": parts[1], "name": parts[2]}
    if object_type == "volume":
        spec = dict(spec)
        if "volume_type" not in spec:
            spec["volume_type"] = "MANAGED"
            warnings.append("volume_type not given; created a MANAGED volume.")
        return w.volumes.create(**sdk_kwargs(uc.VolumesAPI.create, spec, fixed=fixed)), warnings
    if object_type == "table":
        table_type = str(spec.get("table_type", "")).upper()
        fmt = str(spec.get("data_source_format", "")).upper()
        if table_type != "EXTERNAL" or fmt != "DELTA" or not spec.get("storage_location"):
            raise UnsupportedOperation(
                "The Unity Catalog Create Table API only supports EXTERNAL Delta tables "
                "(spec: table_type='EXTERNAL', data_source_format='DELTA', storage_location='<cloud url>', columns=[...]). "
                "Create managed tables, views and other table types with SQL (CREATE TABLE / CREATE VIEW) via execute_sql.",
                hint="Use execute_sql with a CREATE TABLE statement.",
            )
        kwargs = sdk_kwargs(uc.TablesAPI.create, spec, fixed=fixed)
        warnings.append(
            "The Create Table API does not validate the column spec; prefer SQL DDL for anything non-trivial."
        )
        return w.tables.create(**kwargs), warnings
    # function
    overlap = sorted(set(spec) & set(fixed))
    if overlap:
        raise ValidationFailed(f"Field(s) {', '.join(overlap)} must be passed as dedicated tool parameters, not in spec")
    body = {**spec, **fixed}
    required = [
        f.name
        for f in dataclasses.fields(uc.CreateFunction)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.name not in body
    ]
    if required:
        raise ValidationFailed(
            f"Missing required function field(s): {', '.join(required)}",
            hint="The Functions API needs the full CreateFunction body; CREATE FUNCTION via execute_sql is usually simpler.",
        )
    info = parse_sdk_object(uc.CreateFunction, body)
    return w.functions.create(function_info=info), warnings


def _update(object_type: str, full: str, spec: dict[str, Any]) -> Any:
    w = ctx().w
    if not spec:
        raise ValidationFailed("update requires a non-empty spec (e.g. {'comment': '...'} or {'owner': '...'})")
    if object_type in ("table", "function") and set(spec) - {"owner"}:
        raise ValidationFailed(
            f"The {object_type}s API only supports changing 'owner'; got {', '.join(sorted(set(spec) - {'owner'}))}.",
            hint="Use manage_uc_tags action=set_comment for comments, or ALTER statements via execute_sql.",
        )
    if object_type == "catalog":
        return w.catalogs.update(**sdk_kwargs(uc.CatalogsAPI.update, spec, fixed={"name": full}))
    if object_type == "schema":
        return w.schemas.update(**sdk_kwargs(uc.SchemasAPI.update, spec, fixed={"full_name": full}))
    if object_type == "table":
        w.tables.update(**sdk_kwargs(uc.TablesAPI.update, spec, fixed={"full_name": full}))
        return w.tables.get(full)  # tables.update returns nothing
    if object_type == "volume":
        return w.volumes.update(**sdk_kwargs(uc.VolumesAPI.update, spec, fixed={"name": full}))
    return w.functions.update(**sdk_kwargs(uc.FunctionsAPI.update, spec, fixed={"name": full}))


def _delete(object_type: str, full: str, force: bool) -> None:
    w = ctx().w
    flag = True if force else None
    if object_type == "catalog":
        w.catalogs.delete(full, force=flag)
    elif object_type == "schema":
        w.schemas.delete(full, force=flag)
    elif object_type == "table":
        w.tables.delete(full)
    elif object_type == "volume":
        w.volumes.delete(full)
    else:
        w.functions.delete(full, force=flag)


def _list(object_type: str, parent: list[str]) -> Iterable[Any]:
    w = ctx().w
    if object_type == "catalog":
        return w.catalogs.list()
    if object_type == "schema":
        return w.schemas.list(catalog_name=parent[0])
    if object_type == "table":
        return w.tables.list(catalog_name=parent[0], schema_name=parent[1], omit_columns=True, omit_properties=True)
    if object_type == "volume":
        return w.volumes.list(catalog_name=parent[0], schema_name=parent[1])
    return w.functions.list(catalog_name=parent[0], schema_name=parent[1])


@tool(
    toolset="unity_catalog",
    title="Manage Unity Catalog objects",
    safety=_object_levels,
    possible_levels=READ | WRITE | DESTRUCTIVE | WRITE_SECURITY,
    preview=_objects_preview,
)
def manage_uc_objects(
    action: Annotated[
        Literal["create", "get", "list", "update", "delete"],
        Field(description="create | get | list | update | delete"),
    ],
    object_type: Annotated[ObjectType, Field(description="catalog | schema | table | volume | function")],
    full_name: Annotated[
        str | None,
        Field(description="Target name: 'catalog', 'catalog.schema' or 'catalog.schema.object' (backticks allowed). "
              "For list, may give the parent ('catalog' or 'catalog.schema')."),
    ] = None,
    catalog_name: Annotated[str | None, Field(description="Parent catalog (required to list schemas/tables/volumes/functions).")] = None,
    schema_name: Annotated[str | None, Field(description="Parent schema (required to list tables/volumes/functions).")] = None,
    name: Annotated[str | None, Field(description="Object name relative to its parent (alternative to full_name).")] = None,
    spec: Spec = None,
    force: Annotated[
        bool,
        Field(description="delete only (catalog/schema/function): drop even if not empty - RECURSIVELY deletes all contents."),
    ] = False,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Create, inspect, list, update and delete Unity Catalog catalogs, schemas, tables, volumes and functions.

    Hierarchy is catalog -> schema -> object: list schemas needs catalog_name; list tables/volumes/functions
    need catalog_name + schema_name. Identify a target by full_name or by catalog_name/schema_name/name.
    create/update take `spec` with Databricks API fields (e.g. catalog: comment, storage_root, properties;
    volume: volume_type, storage_location, comment; update: comment, owner, new_name, properties).
    Tables: only EXTERNAL Delta tables can be created via the API (use execute_sql for CREATE TABLE/VIEW);
    table/function update supports only 'owner'. delete is DESTRUCTIVE; force=true on catalog/schema deletes
    all contents. Changing owner/isolation_mode is SECURITY_SENSITIVE."""
    c = ctx()

    if action == "list":
        parent = _list_parent(object_type, catalog_name, schema_name, full_name)
        fields = _LIST_FIELDS[object_type]
        return paged_response(
            f"{object_type}(s)" + (f" in {'.'.join(parent)}" if parent else ""),
            _list(object_type, parent),
            page_size,
            page_token,
            lambda item: pick(to_jsonable(item), fields),
        )

    parts = _resolve(object_type, full_name, catalog_name, schema_name, name)
    full = ".".join(parts)

    if force and (action != "delete" or object_type not in _FORCE_SUPPORTED):
        raise ValidationFailed("force is only valid for action=delete on a catalog, schema or function")

    if action == "get":
        return ok(f"Retrieved {object_type} {full}.", _get(object_type, full))

    if action == "create":
        created, warnings = _create(object_type, parts, dict(spec or {}))
        note = c.manifest.safe_track(
            resource_type=f"uc_{object_type}",
            resource_id=full,
            name=full,
            created_by_tool="manage_uc_objects",
            workspace_host=c.host,
        )
        return ok(f"Created {object_type} {full}.", created, warnings=[*warnings, note or ""])

    if action == "update":
        updated = _update(object_type, full, dict(spec or {}))
        return ok(f"Updated {object_type} {full} ({', '.join(sorted(spec or {}))}).", updated)

    # delete
    obj = to_jsonable(_get(object_type, full))
    props = obj.get("properties") if isinstance(obj.get("properties"), dict) else None
    c.safety.check_protected(object_type, full, props, operation="delete")
    _delete(object_type, full, force)
    c.manifest.safe_untrack(f"uc_{object_type}", full)
    return ok(
        f"Deleted {object_type} {full}" + (" (force=true, including all contents)" if force else "") + ".",
        {"object_type": object_type, "full_name": full, "deleted": True, "force": force},
    )
