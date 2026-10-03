"""Lakehouse Monitoring (data quality monitors): manage_uc_monitors.

API choice: ``w.data_quality`` (``DataQualityAPI``, REST ``/api/data-quality/v1/monitors``).
The older ``w.quality_monitors`` (``QualityMonitorsAPI``) is marked *Deprecated* in the
SDK in favour of this API. The data quality API addresses monitors by
``object_type`` (``table`` | ``schema``) and the object's UUID, so this tool resolves the
UUID from the full name via ``w.tables.get`` / ``w.schemas.get``.

Supported by the SDK and used here: create_monitor, get_monitor, update_monitor (with an
update mask), delete_monitor, create_refresh, list_refresh, get_refresh, cancel_refresh.
``list_monitor`` / ``update_refresh`` / ``delete_refresh`` are documented as
*(Unimplemented)* in the SDK and are therefore not exposed.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from databricks.sdk.service import dataquality as dq
from pydantic import Field

from dbx_mcp.databricks.sql_runner import run_statement, select_warehouse, sql_error
from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, EXECUTION, READ, WRITE
from dbx_mcp.safety.validation import quote_full_name, split_full_name
from dbx_mcp.server.context import AppContext
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, Spec, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, ValidationFailed
from dbx_mcp.utils.serialization import parse_sdk_object, pick, to_jsonable

_SAFETY = {
    "create": WRITE,
    "get": READ,
    "update": WRITE,
    "delete": WRITE | DESTRUCTIVE,
    "refresh": EXECUTION,
    "list_refreshes": READ,
    "get_refresh": READ,
    "cancel_refresh": WRITE,
    "metrics": READ,
    "query_metrics": READ | EXECUTION,
}

# DataProfilingConfig fields that are computed by the service and cannot be supplied.
_OUTPUT_ONLY = {
    "dashboard_id",
    "drift_metrics_table_name",
    "profile_metrics_table_name",
    "effective_warehouse_id",
    "latest_monitor_failure_message",
    "monitor_version",
    "monitored_table_name",
    "status",
}
_PROFILE_TYPES = ("snapshot", "time_series", "inference_log")


def _resolve(c: AppContext, object_type: str, full_name: str | None, action: str) -> str:
    """Return the UUID the data quality API needs for a table/schema full name."""
    require(full_name, "full_name", action)
    if object_type == "table":
        split_full_name(full_name, parts=3)
        object_id = c.w.tables.get(full_name).table_id
    else:
        split_full_name(full_name, parts=2)
        object_id = c.w.schemas.get(full_name).schema_id
    if not object_id:
        raise DbxToolError(
            ErrorCategory.SERVICE_ERROR, f"Unity Catalog did not return an id for {object_type} {full_name!r}."
        )
    return object_id


def _config_key(object_type: str) -> str:
    return "data_profiling_config" if object_type == "table" else "anomaly_detection_config"


def _profiling_spec(c: AppContext, spec: dict[str, Any], *, creating: bool) -> dict[str, Any]:
    spec = dict(spec)
    bad = sorted(set(spec) & _OUTPUT_ONLY)
    if bad:
        raise ValidationFailed(f"Output-only field(s) cannot be set: {', '.join(bad)}")
    schema_name = spec.pop("output_schema_name", None)
    if schema_name:
        if spec.get("output_schema_id"):
            raise ValidationFailed("Pass either output_schema_name or output_schema_id, not both")
        split_full_name(schema_name, parts=2)
        spec["output_schema_id"] = c.w.schemas.get(schema_name).schema_id
    if creating:
        if not spec.get("output_schema_id"):
            raise ValidationFailed(
                "spec.output_schema_name (catalog.schema for the metric tables) or spec.output_schema_id is required"
            )
        present = [p for p in _PROFILE_TYPES if spec.get(p) is not None]
        if len(present) != 1:
            raise ValidationFailed(
                "spec must contain exactly one profile type: snapshot ({}), time_series "
                "({timestamp_column, granularities: [AGGREGATION_GRANULARITY_1_DAY, ...]}) or inference_log "
                "({problem_type, timestamp_column, granularities, prediction_column, model_id_column})"
            )
    return spec


def _monitor(c: AppContext, object_type: str, object_id: str, spec: dict[str, Any], *, creating: bool) -> dq.Monitor:
    if object_type == "table":
        cfg = parse_sdk_object(dq.DataProfilingConfig, _profiling_spec(c, spec, creating=creating), "spec")
        return dq.Monitor(object_type="table", object_id=object_id, data_profiling_config=cfg)
    cfg = parse_sdk_object(dq.AnomalyDetectionConfig, spec, "spec")
    return dq.Monitor(object_type="schema", object_id=object_id, anomaly_detection_config=cfg)


def _outputs(monitor: dq.Monitor) -> dict[str, Any]:
    cfg = monitor.data_profiling_config
    if cfg is None:
        return {}
    return pick(
        to_jsonable(cfg),
        [
            "monitored_table_name",
            "status",
            "profile_metrics_table_name",
            "drift_metrics_table_name",
            "dashboard_id",
            "output_schema_id",
            "assets_dir",
            "latest_monitor_failure_message",
        ],
    )


def _preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action not in {"update", "delete"}:
        return None
    c = ctx()
    object_type = args.get("object_type") or "table"
    full_name = args.get("full_name")
    object_id = _resolve(c, object_type, full_name, action)
    current = c.w.data_quality.get_monitor(object_type, object_id)
    current_cfg = to_jsonable(getattr(current, _config_key(object_type))) or {}
    target = {"object_type": object_type, "full_name": full_name, "object_id": object_id}
    if action == "delete":
        c.safety.check_protected(object_type, full_name, None, operation="delete the monitor of")
        return PlanInfo(
            description=f"Delete the data quality monitor on {object_type} {full_name}.",
            target=target,
            details={"current_monitor": to_jsonable(current), "outputs": _outputs(current)},
            warnings=[
                "Monitoring and scheduled refreshes stop. The metric tables and dashboard are NOT deleted by this "
                "call; drop them separately if no longer needed."
            ],
            reversible=False,
        )
    spec = dict(args.get("spec") or {})
    if not spec:
        raise ValidationFailed("spec with the fields to change is required for update")
    if object_type == "table":
        spec = _profiling_spec(c, spec, creating=False)
    key = _config_key(object_type)
    changes = {k: {"current": current_cfg.get(k), "new": v} for k, v in spec.items()}
    return PlanInfo(
        description=f"Update the data quality monitor on {object_type} {full_name}.",
        target=target,
        details={"changes": changes, "update_mask": ",".join(f"{key}.{k}" for k in spec)},
        reversible=True,
    )


def _refresh_summary(r: Any) -> dict[str, Any]:
    return pick(to_jsonable(r), ["refresh_id", "state", "trigger", "start_time_ms", "end_time_ms", "message"])


@tool(
    toolset="unity_catalog",
    title="Data quality monitors (Lakehouse Monitoring)",
    safety=_SAFETY,
    preview=_preview,
)
def manage_uc_monitors(
    action: Annotated[
        Literal[
            "create",
            "get",
            "update",
            "delete",
            "refresh",
            "list_refreshes",
            "get_refresh",
            "cancel_refresh",
            "metrics",
            "query_metrics",
        ],
        Field(
            description=(
                "create/get/update/delete a monitor; refresh: start a metrics refresh; list_refreshes/get_refresh/"
                "cancel_refresh; metrics: names of the profile/drift metric tables and dashboard; "
                "query_metrics: sample rows from a metric table via a SQL warehouse."
            )
        ),
    ],
    full_name: Annotated[
        str | None,
        Field(description="Monitored object: table catalog.schema.table (or catalog.schema when object_type=schema)."),
    ] = None,
    object_type: Annotated[
        Literal["table", "schema"],
        Field(description="table: data profiling monitor; schema: anomaly detection monitor."),
    ] = "table",
    spec: Spec = None,
    refresh_id: Annotated[int | None, Field(description="Refresh id for get_refresh / cancel_refresh.")] = None,
    metrics_table: Annotated[
        Literal["profile", "drift"], Field(description="query_metrics: which metric table to sample.")
    ] = "profile",
    sample_rows: Annotated[int, Field(description="query_metrics: rows to return.", ge=1, le=1000)] = 20,
    warehouse_id: Annotated[str | None, Field(description="SQL warehouse for query_metrics.")] = None,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Unity Catalog data quality monitors (Lakehouse Monitoring) via the Data Quality API.

    Actions (full_name identifies the table, or schema with object_type=schema):
    - create(spec): table monitors take DataProfilingConfig fields - output_schema_name (catalog.schema,
      or output_schema_id), exactly one of snapshot {} | time_series {timestamp_column, granularities:
      ["AGGREGATION_GRANULARITY_1_DAY", ...]} |
      inference_log {...}, plus optional schedule {quartz_cron_expression, timezone_id}, slicing_exprs,
      custom_metrics, baseline_table_name, assets_dir, warehouse_id, notification_settings,
      skip_builtin_dashboard. Schema monitors take AnomalyDetectionConfig fields (excluded_table_full_names).
    - get, update(spec: only the fields to change), delete (metric tables/dashboard are kept).
    - refresh (starts compute), list_refreshes, get_refresh(refresh_id), cancel_refresh(refresh_id).
    - metrics: profile_metrics_table_name, drift_metrics_table_name, dashboard_id - query them with SQL.
    - query_metrics(metrics_table=profile|drift, sample_rows, warehouse_id?): sample rows of a metric table.
    Listing all monitors is not available (the SDK marks list_monitor as unimplemented)."""
    c = ctx()
    if action in {"refresh", "list_refreshes", "get_refresh", "cancel_refresh", "metrics", "query_metrics"} and (
        object_type != "table"
    ):
        raise ValidationFailed(f"Action {action!r} is only supported for table monitors (object_type=table)")
    object_id = _resolve(c, object_type, full_name, action)
    label = f"{object_type} {full_name}"

    if action == "create":
        if not spec:
            raise ValidationFailed("spec is required for create")
        monitor = _monitor(c, object_type, object_id, spec, creating=True)
        created = c.w.data_quality.create_monitor(monitor)
        return ok(
            f"Created data quality monitor on {label}.",
            created,
            next_steps=["Use action=metrics to get the metric table names once the first refresh completes."],
        )

    if action == "get":
        monitor = c.w.data_quality.get_monitor(object_type, object_id)
        return ok(f"Data quality monitor on {label}.", monitor)

    if action == "update":
        if not spec:
            raise ValidationFailed("spec with the fields to change is required for update")
        monitor = _monitor(c, object_type, object_id, spec, creating=False)
        key = _config_key(object_type)
        sent = to_jsonable(getattr(monitor, key)) or {}
        mask = ",".join(f"{key}.{k}" for k in sent)
        if not mask:
            raise ValidationFailed("spec contains no updatable fields")
        updated = c.w.data_quality.update_monitor(object_type, object_id, monitor, mask)
        return ok(f"Updated data quality monitor on {label} (fields: {mask}).", updated)

    if action == "delete":
        c.safety.check_protected(object_type, full_name, None, operation="delete the monitor of")
        c.w.data_quality.delete_monitor(object_type, object_id)
        return ok(
            f"Deleted data quality monitor on {label}.",
            {"object_type": object_type, "full_name": full_name, "object_id": object_id},
            warnings=["Metric tables and the dashboard were not deleted."],
        )

    if action == "refresh":
        refresh = c.w.data_quality.create_refresh(
            object_type, object_id, dq.Refresh(object_type=object_type, object_id=object_id)
        )
        return ok(
            f"Queued refresh {refresh.refresh_id} for {label}.",
            refresh,
            status="pending",
            next_steps=[f"Poll with action=get_refresh refresh_id={refresh.refresh_id}."],
        )

    if action == "list_refreshes":
        it = c.w.data_quality.list_refresh(object_type, object_id)
        return paged_response("refreshes", it, page_size, page_token, _refresh_summary)

    if action in {"get_refresh", "cancel_refresh"}:
        require(refresh_id, "refresh_id", action)
        if action == "cancel_refresh":
            c.w.data_quality.cancel_refresh(object_type, object_id, refresh_id)
            return ok(f"Requested cancellation of refresh {refresh_id} on {label}.", {"refresh_id": refresh_id})
        refresh = c.w.data_quality.get_refresh(object_type, object_id, refresh_id)
        return ok(f"Refresh {refresh_id} on {label}: {to_jsonable(refresh.state)}.", refresh)

    # metrics / query_metrics
    monitor = c.w.data_quality.get_monitor(object_type, object_id)
    outputs = _outputs(monitor)
    if action == "metrics":
        queries = {
            kind: f"SELECT * FROM {quote_full_name(outputs[f'{kind}_metrics_table_name'], parts=3)} LIMIT 100"
            for kind in ("profile", "drift")
            if outputs.get(f"{kind}_metrics_table_name")
        }
        return ok(
            f"Metric outputs of the monitor on {label}.",
            {**outputs, "example_queries": queries},
            next_steps=["Run example_queries with execute_sql, or use action=query_metrics for a quick sample."],
        )

    table = outputs.get(f"{metrics_table}_metrics_table_name")
    if not table:
        raise DbxToolError(
            ErrorCategory.NOT_FOUND,
            f"The monitor on {label} has no {metrics_table} metrics table yet.",
            hint="Wait for the first refresh to complete (action=list_refreshes).",
        )
    sql = f"SELECT * FROM {quote_full_name(table, parts=3)} LIMIT {int(sample_rows)}"
    choice = select_warehouse(c, warehouse_id)
    result = run_statement(c, sql, warehouse_id=choice.warehouse_id, max_rows=sample_rows)
    if result.pending:
        return ok(
            f"Query on {table} is still running (statement_id={result.statement_id}).",
            {"statement_id": result.statement_id, "statement": sql},
            status="pending",
            next_steps=["Poll with manage_sql_statement action=get."],
        )
    if not result.succeeded:
        raise sql_error(result)
    return ok(
        f"Returned {result.row_count} row(s) from {table}.",
        {
            "metrics_table": table,
            "statement": sql,
            "warehouse": choice.as_dict(),
            "columns": result.columns,
            "rows": result.records(),
            "truncated": result.truncated,
        },
        warnings=choice.warnings,
    )
