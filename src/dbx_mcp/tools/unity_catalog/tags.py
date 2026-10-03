"""Unity Catalog tags and comments.

Tags use the official Entity Tag Assignments API (``w.entity_tag_assignments``:
create/get/list/update/delete), which supports catalogs, schemas, tables (incl.
views), columns and volumes. Entity types are passed as the API's plural names
(``catalogs``, ``schemas``, ``tables``, ``columns``, ``volumes``).

Comments use the SDK where it exposes them (``catalogs.update``, ``schemas.update``,
``volumes.update`` with ``comment``). There is no SDK call for table/column comments,
so those run a single quoted DDL statement on a SQL warehouse:

* ``COMMENT ON TABLE <quoted name> IS '<literal>'``
* ``ALTER TABLE <quoted table> ALTER COLUMN <quoted column> COMMENT '<literal>'``
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from databricks.sdk.service import catalog as uc
from pydantic import Field

from dbx_mcp.databricks.sql_runner import run_ddl
from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, EXECUTION, READ, WRITE, SafetyLevel
from dbx_mcp.safety.validation import quote_ident, quote_string_literal
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, ctx, list_page, ok, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.tools.unity_catalog.objects import parse_uc_name
from dbx_mcp.utils.errors import DbxToolError, ValidationFailed, normalize_exception
from dbx_mcp.utils.serialization import pick, to_jsonable

EntityType = Literal["catalog", "schema", "table", "column", "volume"]
_PARTS = {"catalog": 1, "schema": 2, "table": 3, "column": 4, "volume": 3}
_API_ENTITY = {"catalog": "catalogs", "schema": "schemas", "table": "tables", "column": "columns", "volume": "volumes"}
_SQL_COMMENT_TYPES = {"table", "column"}
_TAG_FIELDS = ["tag_key", "tag_value", "source_type", "updated_by", "update_time"]


def _tag_levels(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action == "get":
        return READ
    if action == "remove":
        return DESTRUCTIVE
    if action == "set_comment" and args.get("entity_type") in _SQL_COMMENT_TYPES:
        return WRITE | EXECUTION  # runs DDL on a SQL warehouse
    if action in ("add", "update", "set_comment"):
        return WRITE
    raise ValidationFailed(f"Unknown action {action!r}. Valid: get, add, update, remove, set_comment")


def _entity(entity_type: str | None, full_name: str | None) -> tuple[str, list[str], str]:
    etype = require(entity_type, "entity_type")
    n = _PARTS[etype]
    example = ["catalog", "catalog.schema", "catalog.schema.table", "catalog.schema.table.column"][n - 1]
    if not full_name:
        raise ValidationFailed(f"Parameter 'full_name' is required (for a {etype}: '{example}')")
    parts = parse_uc_name(full_name, n, f"{etype} name")
    return etype, parts, ".".join(parts)


def _validate_tags(tags: dict[str, Any] | None, action: str) -> dict[str, str | None]:
    if not tags:
        raise ValidationFailed(f"Parameter 'tags' (a {{key: value}} map) is required for action '{action}'")
    out: dict[str, str | None] = {}
    for key, value in tags.items():
        if not str(key).strip():
            raise ValidationFailed("Tag keys must be non-empty")
        if isinstance(value, dict | list):
            raise ValidationFailed(f"Tag {key!r}: value must be a string (or null for a key-only tag)")
        out[str(key)] = None if value is None else str(value)
    return out


def _tag_keys(tag_keys: list[str] | None, tags: dict[str, Any] | None) -> list[str]:
    keys = list(tag_keys or []) or list((tags or {}).keys())
    if not keys:
        raise ValidationFailed("Parameter 'tag_keys' is required for action 'remove'")
    return [str(k) for k in keys]


def _current_tags(etype: str, name: str) -> dict[str, str | None]:
    return {
        t.tag_key: t.tag_value
        for t in ctx().w.entity_tag_assignments.list(entity_type=_API_ENTITY[etype], entity_name=name)
    }


def _comment_sql(etype: str, parts: list[str], comment: str) -> str:
    table = ".".join(quote_ident(p) for p in parts[:3])
    if etype == "table":
        value = quote_string_literal(comment) if comment != "" else "NULL"
        return f"COMMENT ON TABLE {table} IS {value}"
    return f"ALTER TABLE {table} ALTER COLUMN {quote_ident(parts[3])} COMMENT {quote_string_literal(comment)}"


def _current_comment(etype: str, parts: list[str], name: str) -> str | None:
    w = ctx().w
    if etype == "catalog":
        return w.catalogs.get(name).comment
    if etype == "schema":
        return w.schemas.get(name).comment
    if etype == "volume":
        return w.volumes.read(name).comment
    table = w.tables.get(".".join(parts[:3]))
    if etype == "table":
        return table.comment
    for col in table.columns or []:
        if col.name == parts[3]:
            return col.comment
    raise ValidationFailed(f"Column {parts[3]!r} not found in table {'.'.join(parts[:3])!r}")


def _tags_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action == "get":
        return None
    etype, parts, name = _entity(args.get("entity_type"), args.get("full_name"))
    target = {"entity_type": etype, "full_name": name}

    if action == "set_comment":
        comment = args.get("comment")
        if comment is None:
            raise ValidationFailed("Parameter 'comment' is required for set_comment ('' clears it)")
        details: dict[str, Any] = {"before": _current_comment(etype, parts, name), "after": comment}
        warnings = []
        if etype in _SQL_COMMENT_TYPES:
            details["sql"] = _comment_sql(etype, parts, comment)
            warnings.append("Runs one DDL statement on a SQL warehouse (may start it).")
        return PlanInfo(description=f"Set the comment on {etype} {name!r}.", target=target, details=details,
                        warnings=warnings, reversible=True)

    current = _current_tags(etype, name)
    if action == "remove":
        keys = _tag_keys(args.get("tag_keys"), args.get("tags"))
        ctx().safety.check_protected(etype, name, {k: v or "" for k, v in current.items()}, operation="remove tags from")
        missing = [k for k in keys if k not in current]
        return PlanInfo(
            description=f"Remove tag(s) {', '.join(keys)} from {etype} {name!r}.",
            target=target,
            details={"removing": {k: current[k] for k in keys if k in current}, "tags_before": current,
                     "tags_after": {k: v for k, v in current.items() if k not in keys}},
            warnings=[f"Not currently set (no-op): {', '.join(missing)}"] if missing else [],
            reversible=True,
        )

    tags = _validate_tags(args.get("tags"), action)
    warnings = []
    if action == "add":
        existing = [k for k in tags if k in current]
        if existing:
            warnings.append(f"Already set (use action=update to change): {', '.join(existing)}")
    else:
        missing = [k for k in tags if k not in current]
        if missing:
            warnings.append(f"Not currently set (use action=add): {', '.join(missing)}")
    return PlanInfo(
        description=f"{'Add' if action == 'add' else 'Update'} tag(s) {', '.join(tags)} on {etype} {name!r}.",
        target=target,
        details={"changes": {k: {"before": current.get(k), "after": v} for k, v in tags.items()}},
        warnings=warnings,
        reversible=True,
    )


def _per_key(keys: list[str], fn: Any, verb: str, etype: str, name: str, extra: dict[str, Any]) -> ToolResponse:
    """Apply ``fn(key)`` per tag key; report partial failures instead of hiding them."""
    done: list[str] = []
    failed: list[dict[str, str]] = []
    first_error: Exception | None = None
    for key in keys:
        try:
            fn(key)
            done.append(key)
        except Exception as exc:
            first_error = first_error or exc
            failed.append({"tag_key": key, "error": str(normalize_exception(exc))})
    if not done and first_error is not None:
        raise first_error
    data = {"entity_type": etype, "full_name": name, verb: done, "failed": failed, **extra}
    if failed:
        return ok(f"{verb.capitalize()} {len(done)} tag(s) on {etype} {name}; {len(failed)} failed.", data,
                  status="partial_failure")
    return ok(f"{verb.capitalize()} tag(s) {', '.join(done)} on {etype} {name}.", data)


@tool(
    toolset="unity_catalog",
    title="Manage Unity Catalog tags & comments",
    safety=_tag_levels,
    possible_levels=READ | WRITE | DESTRUCTIVE | EXECUTION,
    preview=_tags_preview,
)
def manage_uc_tags(
    action: Annotated[
        Literal["get", "add", "update", "remove", "set_comment"],
        Field(description="get: tags + comment; add/update: set tag values; remove: delete tag keys; set_comment."),
    ],
    entity_type: Annotated[EntityType, Field(description="catalog | schema | table (incl. views) | column | volume")],
    full_name: Annotated[str, Field(description="'catalog', 'catalog.schema', 'catalog.schema.table|volume' or "
                                    "'catalog.schema.table.column' (backticks allowed).")],
    tags: Annotated[dict[str, str | None] | None, Field(description="add/update: {tag_key: tag_value}; null value "
                                                        "= key-only tag. E.g. {'pii': 'email', 'owner_team': 'sales'}.")] = None,
    tag_keys: Annotated[list[str] | None, Field(description="remove: tag keys to remove.")] = None,
    comment: Annotated[str | None, Field(description="set_comment: the new comment ('' clears it).")] = None,
    warehouse_id: Annotated[str | None, Field(description="SQL warehouse for table/column comments (auto-selected if omitted).")] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Read, add, update and remove Unity Catalog tags (business metadata, PII classification, ...) and set
    comments on catalogs, schemas, tables/views, columns and volumes.

    Tags use the Entity Tag Assignments API (governed tags may need ASSIGN permission on the tag policy).
    remove is DESTRUCTIVE and needs confirm. Table/column comments run one safely-quoted DDL statement on a
    SQL warehouse (warehouse_id optional); catalog/schema/volume comments use the API."""
    c = ctx()
    w = c.w
    etype, parts, name = _entity(entity_type, full_name)
    api_type = _API_ENTITY[etype]

    if action == "get":
        items, info = list_page(
            w.entity_tag_assignments.list(entity_type=api_type, entity_name=name),
            page_size,
            page_token,
            lambda t: pick(to_jsonable(t), _TAG_FIELDS),
        )
        warnings = []
        try:
            current_comment = _current_comment(etype, parts, name)
        except DbxToolError as exc:
            current_comment = None
            warnings.append(str(exc))
        data = {"entity_type": etype, "full_name": name, "comment": current_comment, "tags": items}
        more = " (more available - pass next_page_token)" if info.has_more else ""
        return ok(f"{etype} {name} has {info.returned} tag(s){more}.", data, page=info, warnings=warnings)

    if action == "set_comment":
        if comment is None:
            raise ValidationFailed("Parameter 'comment' is required for set_comment ('' clears it)")
        extra: dict[str, Any] = {}
        if etype == "catalog":
            w.catalogs.update(name, comment=comment)
        elif etype == "schema":
            w.schemas.update(name, comment=comment)
        elif etype == "volume":
            w.volumes.update(name, comment=comment)
        else:
            statement = _comment_sql(etype, parts, comment)
            result, choice = run_ddl(c, statement, warehouse_id=warehouse_id)
            extra = {"sql": statement, "statement_id": result.statement_id, "warehouse": choice.as_dict()}
        return ok(
            f"{'Cleared' if comment == '' else 'Set'} the comment on {etype} {name}.",
            {"entity_type": etype, "full_name": name, "comment": comment, **extra},
        )

    if action == "remove":
        keys = _tag_keys(tag_keys, tags)
        current = _current_tags(etype, name)
        c.safety.check_protected(etype, name, {k: v or "" for k, v in current.items()}, operation="remove tags from")
        return _per_key(
            keys,
            lambda key: w.entity_tag_assignments.delete(entity_type=api_type, entity_name=name, tag_key=key),
            "removed",
            etype,
            name,
            {"removed_values": {k: current.get(k) for k in keys if k in current}},
        )

    values = _validate_tags(tags, action)

    def assignment(key: str) -> uc.EntityTagAssignment:
        return uc.EntityTagAssignment(entity_name=name, tag_key=key, entity_type=api_type, tag_value=values[key])

    if action == "add":
        return _per_key(list(values), lambda key: w.entity_tag_assignments.create(tag_assignment=assignment(key)),
                        "added", etype, name, {"tags": values})
    return _per_key(
        list(values),
        lambda key: w.entity_tag_assignments.update(
            entity_type=api_type, entity_name=name, tag_key=key, tag_assignment=assignment(key), update_mask="tag_value"
        ),
        "updated",
        etype,
        name,
        {"tags": values},
    )
