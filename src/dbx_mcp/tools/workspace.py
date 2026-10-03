"""Workspace files/notebooks (manage_workspace_files) and code execution (execute_code).

* manage_workspace_files uses the Workspace API (``w.workspace``): list, get-status,
  export, import, mkdirs, delete.
* execute_code runs code
  - on a classic all-purpose cluster through the Command Execution API 1.2
    (``w.command_execution``: create context -> execute -> poll -> destroy context), or
  - (Python only) on serverless jobs compute: a temporary notebook is uploaded to
    ``/Users/<me>/.dbx_mcp/tmp/``, run once with ``w.jobs.submit`` (a notebook task with
    no cluster spec runs on serverless compute when it is enabled for the workspace),
    and its captured stdout/stderr is returned through ``dbutils.notebook.exit``.
Both wait a bounded time and otherwise return ``status="pending"`` with ids to poll.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import re
import time
import uuid
from typing import Annotated, Any, Literal

from databricks.sdk.service import compute as compute_svc
from databricks.sdk.service import jobs as jobs_svc
from databricks.sdk.service import workspace as ws
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, EXECUTION, READ, WRITE, SafetyLevel
from dbx_mcp.safety.validation import resolve_local_path, validate_workspace_path
from dbx_mcp.tools.common import Confirm, DryRun, PageSize, PageToken, ctx, ok, paged_response, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, ValidationFailed
from dbx_mcp.utils.polling import clamp_wait
from dbx_mcp.utils.serialization import wait_response

# Seconds between polls. Module-level so tests can set it to 0.
POLL_INTERVAL_SECONDS = 2.0
_DEFAULT_WAIT_SECONDS = 60
_IMPORT_LIMIT_BYTES = 10 * 1024 * 1024  # Workspace import API limit
_MAX_OUTPUT_CHARS = 100_000
_SERVERLESS_CAPTURE_CHARS = 500_000
_SERVERLESS_RUN_TIMEOUT_SECONDS = 3600
_TMP_MARKER = "/.dbx_mcp/tmp/execute_code_"
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_COMMAND_TERMINAL = {"Finished", "Error", "Cancelled"}
_RUN_TERMINAL = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}


def _v(value: Any) -> Any:
    return getattr(value, "value", value)


def _truncate(text: str | None, limit: int = _MAX_OUTPUT_CHARS) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _wait_budget(requested: int | None) -> tuple[int, str | None]:
    """Clamp a requested wait to the server limits (max_wait_seconds and the tool timeout)."""
    return clamp_wait(ctx().settings, requested, _DEFAULT_WAIT_SECONDS)


def _checked_path(path: str | None, action: str) -> str:
    clean = validate_workspace_path(require(path, "path", action))
    ctx().safety.check_workspace_path(clean)
    return clean


def _check_protected_path(path: str, operation: str) -> None:
    """Apply protected-name patterns to every segment of the path (e.g. /Shared/prod/...)."""
    safety = ctx().safety
    for segment in [s for s in path.split("/") if s]:
        safety.check_protected("workspace object", segment, operation=operation)


# ----------------------------------------------------------------------------------------------
# manage_workspace_files
# ----------------------------------------------------------------------------------------------

_FILE_LEVELS: dict[str, frozenset[SafetyLevel]] = {
    "list": READ,
    "get_status": READ,
    "export": READ,
    "import": WRITE,
    "mkdirs": WRITE,
    "delete": DESTRUCTIVE,
}


def _files_safety(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action not in _FILE_LEVELS:
        raise ValidationFailed(f"Unknown action {action!r} for manage_workspace_files. Valid: {', '.join(_FILE_LEVELS)}")
    levels = _FILE_LEVELS[action]
    if action == "import" and args.get("overwrite"):
        levels = levels | DESTRUCTIVE
    return levels


def _object_row(o: ws.ObjectInfo) -> dict[str, Any]:
    row = {
        "path": o.path,
        "object_type": _v(o.object_type),
        "language": _v(o.language),
        "object_id": o.object_id,
        "size": o.size,
        "modified_at": o.modified_at,
    }
    return {k: v for k, v in row.items() if v is not None}


def _existing(path: str) -> ws.ObjectInfo | None:
    from databricks.sdk.errors import NotFound, ResourceDoesNotExist

    try:
        return ctx().w.workspace.get_status(path)
    except (NotFound, ResourceDoesNotExist):
        return None


def _files_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action == "delete":
        path = _checked_path(args.get("path"), action)
        _check_protected_path(path, "delete")
        info = ctx().w.workspace.get_status(path)
        otype = _v(info.object_type)
        details: dict[str, Any] = _object_row(info)
        warnings = ["Deleted workspace objects can't be restored through the API."]
        if otype in {"DIRECTORY", "REPO"}:
            children = []
            for child in ctx().w.workspace.list(path):
                children.append(child.path)
                if len(children) >= 1000:
                    break
            details["direct_children"] = len(children) if len(children) < 1000 else "1000+"
            details["children_sample"] = children[:20]
            if children and not args.get("recursive"):
                warnings.append("Directory is not empty: the delete will fail unless recursive=true.")
            elif children:
                warnings.append(f"recursive=true: the directory and ALL {details['direct_children']} entries "
                                "(and their descendants) will be deleted.")
        return PlanInfo(
            description=f"Delete {otype} {path}" + (" recursively." if args.get("recursive") else "."),
            target={"path": path, "object_type": otype, "object_id": info.object_id},
            details=details,
            warnings=warnings,
            reversible=False,
        )
    if action == "import":
        path = _checked_path(args.get("path"), action)
        existing = _existing(path) if args.get("overwrite") else None
        if args.get("overwrite"):
            _check_protected_path(path, "overwrite")
        size = None
        with contextlib.suppress(DbxToolError):
            size = len(_import_bytes(args))
        return PlanInfo(
            description=f"Import {size if size is not None else '?'} bytes to {path}"
            + (" overwriting the existing object." if existing else "."),
            target={"path": path},
            details={
                "format": _import_format(args.get("format"), args.get("language")).value,
                "language": args.get("language"),
                "overwrite": bool(args.get("overwrite")),
                "existing_object": _object_row(existing) if existing else None,
            },
            warnings=["The existing object's content will be replaced."] if existing else [],
            reversible=not existing,
        )
    return None


def _import_format(fmt: str | None, language: str | None) -> ws.ImportFormat:
    if fmt:
        return ws.ImportFormat(fmt)
    return ws.ImportFormat.SOURCE if language else ws.ImportFormat.AUTO


def _import_bytes(args: dict[str, Any]) -> bytes:
    sources = [k for k in ("content", "content_base64", "local_path") if args.get(k) is not None]
    if len(sources) != 1:
        raise ValidationFailed("import needs exactly one of: content (text), content_base64, local_path")
    source = sources[0]
    if source == "content":
        data = str(args["content"]).encode("utf-8")
    elif source == "content_base64":
        try:
            data = base64.b64decode(args["content_base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValidationFailed(f"content_base64 is not valid base64: {exc}") from exc
    else:
        local = resolve_local_path(ctx().settings.local_file_root, args["local_path"], must_exist=True)
        if local.stat().st_size > _IMPORT_LIMIT_BYTES:
            raise ValidationFailed(f"Local file is larger than the 10 MB workspace import limit: {args['local_path']!r}")
        data = local.read_bytes()
    if len(data) > _IMPORT_LIMIT_BYTES:
        raise ValidationFailed("Content exceeds the 10 MB workspace import limit")
    return data


@tool(
    toolset="workspace",
    title="Manage workspace files and notebooks",
    safety=_files_safety,
    possible_levels=READ | WRITE | DESTRUCTIVE,
    preview=_files_preview,
)
def manage_workspace_files(
    action: Annotated[
        Literal["list", "get_status", "export", "import", "mkdirs", "delete"],
        Field(
            description="list: directory contents; get_status: object metadata; export: download content "
            "(inline or to local_path); import: upload/create/update a file or notebook; mkdirs: create "
            "directories; delete: delete object (recursive for non-empty directories)."
        ),
    ],
    path: Annotated[str, Field(description="Absolute workspace path, e.g. /Users/me@x.com/project/nb.")],
    recursive: Annotated[
        bool | None, Field(description="list: walk sub-directories (files only); delete: delete non-empty directory.")
    ] = None,
    format: Annotated[
        Literal["SOURCE", "HTML", "JUPYTER", "DBC", "R_MARKDOWN", "AUTO", "RAW"] | None,
        Field(
            description="export/import format. Export default: SOURCE for notebooks, AUTO otherwise. Import "
            "default: SOURCE when language is set (notebook), else AUTO (file, or notebook if the content has a "
            "notebook header). RAW is import-only."
        ),
    ] = None,
    language: Annotated[
        Literal["PYTHON", "SQL", "SCALA", "R"] | None,
        Field(description="import: notebook language (required for a single SOURCE notebook)."),
    ] = None,
    content: Annotated[str | None, Field(description="import: text content (UTF-8).")] = None,
    content_base64: Annotated[str | None, Field(description="import: binary content, base64-encoded.")] = None,
    local_path: Annotated[
        str | None,
        Field(description="import: read from / export: write to this path relative to DBX_MCP_LOCAL_FILE_ROOT."),
    ] = None,
    overwrite: Annotated[
        bool, Field(description="import: replace an existing object (DESTRUCTIVE); export: replace a local file.")
    ] = False,
    create_parents: Annotated[bool, Field(description="import: create missing parent directories.")] = True,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Manage Databricks workspace files, notebooks and folders (Workspace API).

    - list (path[, recursive]), get_status (path): metadata (type, language, size, object_id).
    - export (path[, format, local_path]): text content inline (UTF-8) or base64 for binary; capped by
      DBX_MCP_MAX_INLINE_DOWNLOAD_BYTES; with local_path the file is written under DBX_MCP_LOCAL_FILE_ROOT.
    - import (path, content | content_base64 | local_path[, language, format, overwrite]): create or update a
      file or notebook (10 MB limit). Notebooks: language=PYTHON|SQL|SCALA|R with format SOURCE (default
      when language is set) or JUPYTER (.ipynb content). overwrite=true is DESTRUCTIVE (confirm required).
    - mkdirs (path): create directory and parents.
    - delete (path[, recursive]): DESTRUCTIVE, confirm required.
    """
    c = ctx()
    w = c.w
    clean = _checked_path(path, action)

    if action == "list":
        items = w.workspace.list(clean, recursive=bool(recursive))
        warnings = ["Recursive listing returns files/notebooks only (directories are traversed)."] if recursive else None
        return paged_response("object(s)", items, page_size, page_token, _object_row, warnings=warnings)

    if action == "get_status":
        info = w.workspace.get_status(clean)
        return ok(f"{_v(info.object_type)} {clean}.", info)

    if action == "export":
        if format == "RAW":
            raise ValidationFailed("format RAW is only valid for import")
        info = w.workspace.get_status(clean)
        otype = _v(info.object_type)
        fmt = ws.ExportFormat(format) if format else (
            ws.ExportFormat.SOURCE if otype == "NOTEBOOK" else ws.ExportFormat.AUTO
        )
        if otype == "DIRECTORY" and fmt not in {ws.ExportFormat.SOURCE, ws.ExportFormat.DBC, ws.ExportFormat.AUTO}:
            raise ValidationFailed("Directories can only be exported with format SOURCE, DBC or AUTO")
        cap = c.settings.max_inline_download_bytes
        if local_path is None and info.size and info.size > cap:
            raise ValidationFailed(
                f"{clean} is {info.size} bytes, above the inline limit of {cap} bytes",
                hint="Pass local_path to write it under DBX_MCP_LOCAL_FILE_ROOT, or raise DBX_MCP_MAX_INLINE_DOWNLOAD_BYTES.",
            )
        response = w.workspace.export(clean, format=fmt)
        raw = base64.b64decode(response.content or "")
        meta = {"path": clean, "object_type": otype, "language": _v(info.language), "format": fmt.value,
                "file_type": response.file_type, "size_bytes": len(raw)}
        if local_path is not None:
            target = resolve_local_path(c.settings.local_file_root, local_path, must_exist=False)
            if target.exists() and not overwrite:
                raise ValidationFailed(f"Local file {local_path!r} exists; pass overwrite=true to replace it")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            return ok(f"Exported {clean} ({len(raw)} bytes) to local file {local_path}.", {**meta, "local_path": local_path})
        if len(raw) > cap:
            raise ValidationFailed(
                f"Exported content is {len(raw)} bytes, above the inline limit of {cap} bytes",
                hint="Pass local_path to write it under DBX_MCP_LOCAL_FILE_ROOT.",
            )
        try:
            text = raw.decode("utf-8")
            body = {**meta, "encoding": "utf-8", "content": text}
        except UnicodeDecodeError:
            body = {**meta, "encoding": "base64", "content": base64.b64encode(raw).decode("ascii")}
        return ok(f"Exported {otype} {clean} ({len(raw)} bytes, {body['encoding']}).", body)

    if action == "import":
        args = {"content": content, "content_base64": content_base64, "local_path": local_path}
        data = _import_bytes(args)
        fmt = _import_format(format, language)
        if fmt == ws.ImportFormat.SOURCE and not language:
            raise ValidationFailed("format SOURCE requires 'language' (PYTHON, SQL, SCALA or R) for a single notebook")
        if overwrite:
            _check_protected_path(clean, "overwrite")
        parent = clean.rsplit("/", 1)[0]
        if create_parents and len([s for s in parent.split("/") if s]) >= 2:
            w.workspace.mkdirs(parent)
        w.workspace.import_(
            clean,
            content=base64.b64encode(data).decode("ascii"),
            format=fmt,
            language=ws.Language(language) if language else None,
            overwrite=overwrite,
        )
        return ok(
            f"Imported {len(data)} bytes to {clean} (format {fmt.value}).",
            {"path": clean, "format": fmt.value, "language": language, "size_bytes": len(data), "overwrite": overwrite},
        )

    if action == "mkdirs":
        w.workspace.mkdirs(clean)
        return ok(f"Directory {clean} exists.", {"path": clean})

    # delete
    _check_protected_path(clean, "delete")
    w.workspace.delete(clean, recursive=recursive)
    return ok(f"Deleted {clean}.", {"path": clean, "deleted": True, "recursive": bool(recursive)})


# ----------------------------------------------------------------------------------------------
# execute_code
# ----------------------------------------------------------------------------------------------

_EXEC_LEVELS: dict[str, frozenset[SafetyLevel]] = {"run": EXECUTION, "get_status": READ, "cancel": WRITE}


def _exec_safety(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action") or "run"
    if action not in _EXEC_LEVELS:
        raise ValidationFailed(f"Unknown action {action!r} for execute_code. Valid: {', '.join(_EXEC_LEVELS)}")
    return _EXEC_LEVELS[action]


def _exec_preview(args: dict[str, Any]) -> PlanInfo | None:
    if (args.get("action") or "run") != "run":
        return None
    code = args.get("code") or ""
    mode = args.get("compute") or "cluster"
    target: dict[str, Any] = {"compute": mode, "language": args.get("language") or "python"}
    if mode == "cluster":
        target["cluster_id"] = args.get("cluster_id") or ctx().default_cluster_id
    return PlanInfo(
        description=f"Execute {len(code)} characters of {target['language']} code on "
        + ("serverless jobs compute (one-time run)." if mode == "serverless" else f"cluster {target.get('cluster_id')}."),
        target=target,
        details={"code_preview": code[:1000]},
        warnings=["Code runs with the permissions of the authenticated identity (or the cluster's access mode) "
                  "and may read, modify or delete data and incur compute cost."],
        reversible=False,
    )


def _strip_ansi(text: str | None) -> str | None:
    return _ANSI.sub("", text) if text else text


def _command_output(resp: compute_svc.CommandStatusResponse) -> dict[str, Any]:
    out: dict[str, Any] = {"command_status": _v(resp.status)}
    r = resp.results
    if not r:
        return out
    rtype = _v(r.result_type)
    out["result_type"] = rtype
    if rtype == "error":
        out["error"] = {"summary": _truncate(_strip_ansi(r.summary), 2000), "cause": _truncate(_strip_ansi(r.cause), 20_000)}
    elif rtype == "text":
        out["output"] = _truncate(_strip_ansi(r.data if isinstance(r.data, str) else str(r.data or "")))
    elif rtype == "table":
        rows = list(r.data or []) if isinstance(r.data, list) else []
        cap = ctx().settings.sql_max_rows
        out["columns"] = [{"name": col.get("name"), "type": col.get("type")} for col in (r.schema or [])]
        out["rows"] = rows[:cap]
        out["row_count"] = min(len(rows), cap)
        if len(rows) > cap:
            out["truncated"] = True
    elif rtype in {"image", "images"}:
        out["note"] = "Image output is not returned by this tool."
        out["file_names"] = r.file_names or ([r.file_name] if r.file_name else [])
    if r.truncated:
        out["truncated"] = True
    return out


def _command_response(resp: compute_svc.CommandStatusResponse, compute_info: dict[str, Any],
                      warnings: list[str]) -> ToolResponse:
    out = _command_output(resp)
    data = {"compute": compute_info, **out}
    status = out["command_status"]
    if status == "Finished" and out.get("result_type") != "error":
        return ok(f"Code finished on {compute_info.get('description')}.", data, warnings=warnings)
    summary = (out.get("error") or {}).get("summary") or f"command {status}"
    return ok(f"Code failed on {compute_info.get('description')}: {_truncate(summary, 300)}", data,
              status="failed", warnings=warnings)


def _destroy_context(cluster_id: str, context_id: str) -> str | None:
    try:
        ctx().w.command_execution.destroy(cluster_id, context_id)
        return None
    except Exception as exc:
        return f"Could not destroy execution context {context_id}: {str(exc)[:200]}"


def _resolve_cluster(cluster_id: str | None) -> tuple[str, dict[str, Any]]:
    c = ctx()
    cid = cluster_id or c.default_cluster_id
    if not cid:
        raise DbxToolError(
            ErrorCategory.CONFIGURATION,
            "No cluster_id given and no default cluster configured.",
            hint="Pass cluster_id, set DBX_MCP_DEFAULT_CLUSTER_ID, or use compute='serverless' for Python.",
        )
    cluster = c.w.clusters.get(cid)
    state = _v(cluster.state)
    info = {
        "type": "cluster",
        "cluster_id": cid,
        "cluster_name": cluster.cluster_name,
        "state": state,
        "spark_version": cluster.spark_version,
        "data_security_mode": _v(cluster.data_security_mode),
        "source": "parameter" if cluster_id else "default (DBX_MCP_DEFAULT_CLUSTER_ID)",
        "description": f"cluster {cluster.cluster_name!r} ({cid})",
    }
    if state not in {"RUNNING", "RESIZING"}:
        raise DbxToolError(
            ErrorCategory.CONFLICT,
            f"Cluster {cid} ({cluster.cluster_name!r}) is {state}; code can only run on a RUNNING cluster.",
            hint="Start the cluster (manage_cluster action='start') and retry once it is RUNNING, "
            "or use compute='serverless' for Python.",
        )
    return cid, info


def _run_on_cluster(code: str, language: str, cluster_id: str | None, timeout_seconds: int | None) -> ToolResponse:
    w = ctx().w
    cid, info = _resolve_cluster(cluster_id)
    budget, note = _wait_budget(timeout_seconds)
    warnings = [note] if note else []
    deadline = time.monotonic() + budget
    lang = compute_svc.Language(language)

    context_id = wait_response(w.command_execution.create(cluster_id=cid, language=lang)).id
    try:
        while True:
            cs = w.command_execution.context_status(cid, context_id)
            if _v(cs.status) == "Running":
                break
            if _v(cs.status) == "Error":
                raise DbxToolError(ErrorCategory.SERVICE_ERROR, f"Execution context on cluster {cid} failed to start.")
            if time.monotonic() >= deadline:
                raise DbxToolError(ErrorCategory.TIMEOUT, f"Execution context on cluster {cid} not ready within {budget}s.")
            time.sleep(POLL_INTERVAL_SECONDS)
        command_id = wait_response(
            w.command_execution.execute(cluster_id=cid, context_id=context_id, language=lang, command=code)
        ).id
        while True:
            resp = w.command_execution.command_status(cid, context_id, command_id)
            if _v(resp.status) in _COMMAND_TERMINAL or time.monotonic() >= deadline:
                break
            time.sleep(min(POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))
    except BaseException:
        _destroy_context(cid, context_id)
        raise

    if _v(resp.status) not in _COMMAND_TERMINAL:
        return ok(
            f"Code still {_v(resp.status)} on {info['description']} after {budget}s.",
            {"compute": info, "cluster_id": cid, "context_id": context_id, "command_id": command_id,
             "command_status": _v(resp.status)},
            status="pending",
            warnings=warnings,
            next_steps=[
                f"Poll: execute_code action='get_status' cluster_id='{cid}' context_id='{context_id}' "
                f"command_id='{command_id}'.",
                "Or stop it: the same ids with action='cancel'. The execution context is released once the "
                "command finishes and is polled, or cancelled.",
            ],
        )
    destroy_warning = _destroy_context(cid, context_id)
    if destroy_warning:
        warnings.append(destroy_warning)
    return _command_response(resp, info, warnings)


_NOTEBOOK_TEMPLATE = '''# Databricks notebook source
# Generated by dbx-mcp execute_code (serverless). Temporary; safe to delete.
import base64 as _dbx_b64, contextlib as _dbx_cl, io as _dbx_io, json as _dbx_json, traceback as _dbx_tb

_dbx_code = _dbx_b64.b64decode("__CODE_B64__").decode("utf-8")
_dbx_out, _dbx_err = _dbx_io.StringIO(), _dbx_io.StringIO()
_dbx_result = {"status": "ok"}
try:
    with _dbx_cl.redirect_stdout(_dbx_out), _dbx_cl.redirect_stderr(_dbx_err):
        exec(compile(_dbx_code, "<execute_code>", "exec"), {"__name__": "__main__", "spark": spark, "dbutils": dbutils})
except BaseException as _dbx_e:
    _dbx_result = {"status": "error", "error": f"{type(_dbx_e).__name__}: {_dbx_e}", "traceback": _dbx_tb.format_exc()}
_dbx_limit = __LIMIT__
for _dbx_name, _dbx_buf in (("stdout", _dbx_out), ("stderr", _dbx_err)):
    _dbx_text = _dbx_buf.getvalue()
    _dbx_result[_dbx_name] = _dbx_text[:_dbx_limit]
    if len(_dbx_text) > _dbx_limit:
        _dbx_result[_dbx_name + "_truncated"] = True

# COMMAND ----------

dbutils.notebook.exit(_dbx_json.dumps(_dbx_result))
'''


def build_serverless_notebook(code: str) -> str:
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return _NOTEBOOK_TEMPLATE.replace("__CODE_B64__", encoded).replace("__LIMIT__", str(_SERVERLESS_CAPTURE_CHARS))


def _tmp_notebook_of(run: jobs_svc.Run) -> str | None:
    for task in run.tasks or []:
        nb = task.notebook_task
        if nb and nb.notebook_path and _TMP_MARKER in nb.notebook_path:
            return nb.notebook_path
    return None


def _delete_tmp_notebook(path: str | None) -> str | None:
    if not path or _TMP_MARKER not in path:
        return None
    try:
        ctx().w.workspace.delete(path)
        return None
    except Exception as exc:
        return f"Temporary notebook {path} could not be deleted: {str(exc)[:200]}"


def _serverless_info(run: jobs_svc.Run) -> dict[str, Any]:
    return {
        "type": "serverless",
        "run_id": run.run_id,
        "run_page_url": run.run_page_url,
        "description": f"serverless jobs compute (run {run.run_id})",
    }


def _serverless_response(run: jobs_svc.Run, warnings: list[str]) -> ToolResponse:

    w = ctx().w
    info = _serverless_info(run)
    if not (run.state and _v(run.state.life_cycle_state) in _RUN_TERMINAL):
        return ok(
            f"Code still running on {info['description']}.",
            {"compute": info, "run_id": run.run_id, "life_cycle_state": _v(run.state.life_cycle_state) if run.state else None},
            status="pending",
            warnings=warnings,
            next_steps=[
                f"Poll: execute_code action='get_status' run_id={run.run_id}.",
                f"Stop: execute_code action='cancel' run_id={run.run_id}.",
            ],
        )
    task = (run.tasks or [None])[0]
    task_run_id = (task.run_id if task else None) or run.run_id
    output = w.jobs.get_run_output(task_run_id)
    result_state = _v(run.state.result_state) if run.state else None
    data: dict[str, Any] = {"compute": info, "run_id": run.run_id, "result_state": result_state}
    captured = None
    if output.notebook_output and output.notebook_output.result:
        try:
            captured = json.loads(output.notebook_output.result)
        except ValueError:
            data["exit_value"] = _truncate(output.notebook_output.result)
    if isinstance(captured, dict):
        data["stdout"] = _truncate(captured.get("stdout"))
        data["stderr"] = _truncate(captured.get("stderr"))
        if captured.get("stdout_truncated") or captured.get("stderr_truncated"):
            data["truncated"] = True
        if captured.get("status") == "error":
            data["error"] = {"summary": captured.get("error"), "cause": _truncate(captured.get("traceback"), 20_000)}
    if output.error:
        data["error"] = {"summary": output.error, "cause": _truncate(_strip_ansi(output.error_trace), 20_000)}
    cleanup = _delete_tmp_notebook(_tmp_notebook_of(run))
    if cleanup:
        warnings.append(cleanup)
    if result_state == "SUCCESS" and "error" not in data:
        return ok(f"Code finished on {info['description']}.", data, warnings=warnings)
    msg = (data.get("error") or {}).get("summary") or (run.state.state_message if run.state else None) or result_state
    return ok(f"Code failed on {info['description']}: {_truncate(str(msg), 300)}", data, status="failed", warnings=warnings)


def _run_serverless(code: str, timeout_seconds: int | None) -> ToolResponse:
    c = ctx()
    w = c.w
    user = w.current_user.me().user_name
    if not user:
        raise DbxToolError(ErrorCategory.CONFIGURATION, "Could not determine the current user's home folder.")
    token = uuid.uuid4().hex[:12]
    tmp_dir = f"/Users/{user}/.dbx_mcp/tmp"
    nb_path = f"{tmp_dir}/execute_code_{token}"
    c.safety.check_workspace_path(nb_path)
    budget, note = _wait_budget(timeout_seconds)
    warnings = [note] if note else []
    w.workspace.mkdirs(tmp_dir)
    w.workspace.import_(
        nb_path,
        content=base64.b64encode(build_serverless_notebook(code).encode("utf-8")).decode("ascii"),
        format=ws.ImportFormat.SOURCE,
        language=ws.Language.PYTHON,
        overwrite=True,
    )
    try:
        submitted = wait_response(
            w.jobs.submit(
                run_name=f"dbx-mcp execute_code {token}",
                tasks=[
                    jobs_svc.SubmitTask(
                        task_key="execute_code",
                        notebook_task=jobs_svc.NotebookTask(notebook_path=nb_path, source=jobs_svc.Source.WORKSPACE),
                    )
                ],
                timeout_seconds=_SERVERLESS_RUN_TIMEOUT_SECONDS,
            )
        )
    except BaseException:
        _delete_tmp_notebook(nb_path)
        raise
    run_id = submitted.run_id
    deadline = time.monotonic() + budget
    while True:
        run = w.jobs.get_run(run_id)
        if (run.state and _v(run.state.life_cycle_state) in _RUN_TERMINAL) or time.monotonic() >= deadline:
            break
        time.sleep(min(POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))
    return _serverless_response(run, warnings)


def _own_serverless_run(run_id: int) -> jobs_svc.Run:
    run = ctx().w.jobs.get_run(run_id)
    if not _tmp_notebook_of(run):
        raise ValidationFailed(
            f"Run {run_id} was not started by execute_code; use manage_job_runs to inspect or cancel it."
        )
    return run


@tool(
    toolset="workspace",
    title="Execute code",
    safety=_exec_safety,
    possible_levels=READ | WRITE | EXECUTION,
    preview=_exec_preview,
)
def execute_code(
    action: Annotated[
        Literal["run", "get_status", "cancel"],
        Field(description="run: execute code; get_status: poll a pending execution; cancel: stop it."),
    ] = "run",
    code: Annotated[str | None, Field(description="run: the code to execute.")] = None,
    language: Annotated[Literal["python", "sql", "scala", "r"], Field(description="run: code language.")] = "python",
    compute: Annotated[
        Literal["cluster", "serverless"],
        Field(description="run: 'cluster' (classic all-purpose cluster; any language) or 'serverless' "
              "(Python only, one-time serverless job run; slower to start)."),
    ] = "cluster",
    cluster_id: Annotated[
        str | None, Field(description="Cluster id (default DBX_MCP_DEFAULT_CLUSTER_ID); also for get_status/cancel.")
    ] = None,
    timeout_seconds: Annotated[
        int | None, Field(description="run: max seconds to wait before returning 'pending' (capped by server).", ge=1)
    ] = None,
    context_id: Annotated[str | None, Field(description="get_status/cancel (cluster): execution context id.")] = None,
    command_id: Annotated[str | None, Field(description="get_status/cancel (cluster): command id.")] = None,
    run_id: Annotated[int | None, Field(description="get_status/cancel (serverless): run id.")] = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Execute Python, SQL, Scala or R code on Databricks compute and return its output.

    - run (code, language[, compute, cluster_id, timeout_seconds]): on a RUNNING classic cluster via the
      Command Execution API (a fresh execution context per call; no state is kept between calls), or, for
      Python, on serverless jobs compute (temporary notebook in ~/.dbx_mcp/tmp, one-time run; stdout/stderr
      captured). Returns status success/failed with output (text or table rows/columns) and error
      summary/stack trace, and which compute was used. If not finished within timeout_seconds it returns
      status 'pending' with ids to poll.
    - get_status (cluster_id+context_id+command_id, or run_id): poll a pending execution.
    - cancel (same ids): stop a pending execution.
    For SQL on a SQL warehouse prefer execute_sql. Classified EXECUTION: code can change data and costs money.
    """
    c = ctx()
    w = c.w

    if action == "run":
        require(code, "code", action)
        if compute == "serverless":
            if language != "python":
                raise ValidationFailed("compute='serverless' supports language='python' only; "
                                       "use compute='cluster' or execute_sql for SQL")
            return _run_serverless(code, timeout_seconds)
        return _run_on_cluster(code, language, cluster_id, timeout_seconds)

    if run_id is not None:
        run = _own_serverless_run(run_id)
        if action == "get_status":
            return _serverless_response(run, [])
        w.jobs.cancel_run(run_id)
        warning = _delete_tmp_notebook(_tmp_notebook_of(run))
        return ok(f"Cancellation of serverless run {run_id} requested.", {"run_id": run_id, "cancel_requested": True},
                  warnings=[warning] if warning else None)

    cid = cluster_id or c.default_cluster_id
    require(cid, "cluster_id", action)
    require(context_id, "context_id", action)
    require(command_id, "command_id", action)
    info = {"type": "cluster", "cluster_id": cid, "description": f"cluster {cid}"}
    if action == "get_status":
        resp = w.command_execution.command_status(cid, context_id, command_id)
        if _v(resp.status) not in _COMMAND_TERMINAL:
            return ok(
                f"Code still {_v(resp.status)} on cluster {cid}.",
                {"compute": info, "context_id": context_id, "command_id": command_id, "command_status": _v(resp.status)},
                status="pending",
                next_steps=["Poll again later, or cancel with action='cancel'."],
            )
        warning = _destroy_context(cid, context_id)
        return _command_response(resp, info, [warning] if warning else [])

    # cancel (cluster)
    w.command_execution.cancel(cluster_id=cid, context_id=context_id, command_id=command_id)
    warning = _destroy_context(cid, context_id)
    return ok(
        f"Cancelled command {command_id} on cluster {cid}.",
        {"compute": info, "context_id": context_id, "command_id": command_id, "cancel_requested": True},
        warnings=[warning] if warning else None,
    )
