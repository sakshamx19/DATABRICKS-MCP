"""Unity Catalog Volume files: get_volume_folder_details, manage_volume_files.

All remote access goes through the Databricks Files API (``w.files``). Every path is
validated with :func:`parse_volume_path` (rejects traversal, backslashes, control
characters, anything outside ``/Volumes/<catalog>/<schema>/<volume>``) and checked
against ``DBX_MCP_ALLOWED_VOLUME_PREFIXES``. Local filesystem access is disabled
unless ``DBX_MCP_LOCAL_FILE_ROOT`` is set, and is then confined to that directory.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import shutil
from collections import deque
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, READ, WRITE, SafetyLevel
from dbx_mcp.safety.validation import VolumePath, parse_volume_path, resolve_local_path
from dbx_mcp.tools.common import (
    Confirm,
    DryRun,
    PageSize,
    PageToken,
    ctx,
    list_page,
    ok,
    paged_response,
    require,
)
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, ValidationFailed

_NOT_FOUND = (NotFound, ResourceDoesNotExist)

#: Hard caps that keep a single call bounded regardless of user input.
MAX_SCAN_ENTRIES = 10_000
MAX_SCAN_DEPTH = 10
MAX_DELTA_PROBES = 25
MAX_RECURSIVE_DELETE_ENTRIES = 10_000

# ----------------------------------------------------------------------------------------------
# Format detection
# ----------------------------------------------------------------------------------------------

_COMPRESSION_SUFFIXES = {"gz", "gzip", "bz2", "zst", "zstd", "lz4", "snappy", "deflate", "xz", "br"}
_FORMATS: dict[str, str] = {
    "parquet": "parquet",
    "pq": "parquet",
    "csv": "csv",
    "tsv": "tsv",
    "json": "json",
    "jsonl": "json",
    "ndjson": "json",
    "avro": "avro",
    "orc": "orc",
    "txt": "text",
    "log": "text",
    "md": "markdown",
    "xml": "xml",
    "xlsx": "excel",
    "xls": "excel",
    "pdf": "pdf",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "svg": "image",
    "webp": "image",
    "zip": "archive",
    "tar": "archive",
    "tgz": "archive",
    "whl": "python_wheel",
    "jar": "jar",
    "py": "python",
    "sql": "sql",
    "ipynb": "notebook",
    "yaml": "yaml",
    "yml": "yaml",
    "html": "html",
    "htm": "html",
    "crc": "checksum",
}


def detect_file_format(name: str) -> tuple[str, str | None]:
    """Return ``(format, compression)`` inferred from a file name's extensions.

    ``part-0.snappy.parquet`` -> ("parquet", "snappy"); ``data.csv.gz`` -> ("csv", "gz").
    """
    parts = name.lower().rsplit("/", 1)[-1].split(".")
    if len(parts) < 2:
        return "unknown", None
    exts = parts[1:]
    compression: str | None = None
    while exts and exts[-1] in _COMPRESSION_SUFFIXES:  # "data.csv.gz"
        compression = compression or exts[-1]
        exts.pop()
    if not exts:
        return ("compressed", compression) if compression else ("unknown", None)
    if len(exts) >= 2 and exts[-2] in _COMPRESSION_SUFFIXES:  # "part-0.snappy.parquet"
        compression = compression or exts[-2]
    if exts[-1] in _FORMATS:
        return _FORMATS[exts[-1]], compression
    return "unknown", compression


def _iso_ms(value: int | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def _entry_dict(entry: Any, depth: int = 1) -> dict[str, Any]:
    is_dir = bool(entry.is_directory)
    name = entry.name or (entry.path or "").rstrip("/").rsplit("/", 1)[-1]
    out: dict[str, Any] = {
        "name": name,
        "path": (entry.path or "").rstrip("/") if is_dir else entry.path,
        "type": "directory" if is_dir else "file",
    }
    if is_dir:
        out["format"] = "directory"
    else:
        fmt, compression = detect_file_format(name)
        out["format"] = fmt
        if compression:
            out["compression"] = compression
        out["size_bytes"] = entry.file_size
    if entry.last_modified is not None:
        out["modified_at"] = _iso_ms(entry.last_modified)
        out["modified_epoch_ms"] = entry.last_modified
    if depth > 1:
        out["depth"] = depth
    return out


# ----------------------------------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------------------------------

def _vpath(path: str | None, *, require_file: bool = False, what: str = "path") -> VolumePath:
    require(path, what)
    vp = parse_volume_path(path, require_file=require_file)  # type: ignore[arg-type]
    ctx().safety.check_volume_path(vp.full)
    return vp


def _is_directory(full: str) -> bool:
    """True if ``full`` is an existing directory (HEAD request), False if not found."""
    try:
        ctx().w.files.get_directory_metadata(full)
        return True
    except _NOT_FOUND:
        return False


def _file_metadata(full: str) -> dict[str, Any]:
    meta = ctx().w.files.get_metadata(full)
    name = full.rsplit("/", 1)[-1]
    fmt, compression = detect_file_format(name)
    out: dict[str, Any] = {
        "path": full,
        "name": name,
        "type": "file",
        "format": fmt,
        "size_bytes": meta.content_length,
        "content_type": meta.content_type,
        "last_modified": meta.last_modified,
    }
    if compression:
        out["compression"] = compression
    return out


def _list(full: str) -> Iterator[Any]:
    return iter(ctx().w.files.list_directory_contents(full))


def _walk(root: str, *, max_depth: int, max_entries: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Breadth-first listing below ``root``.

    Directories that contain ``_delta_log`` are reported as Delta tables and not descended into.
    Returns ``(entries, info)`` where info has ``truncated``/``depth_limited``/``root_is_delta``.
    """
    entries: list[dict[str, Any]] = []
    info: dict[str, Any] = {"truncated": False, "depth_limited": False, "root_is_delta": False, "expanded": set()}
    by_path: dict[str, dict[str, Any]] = {}
    queue: deque[tuple[str, int]] = deque([(root, 1)])
    while queue:
        directory, depth = queue.popleft()
        info["expanded"].add(directory)
        children = []
        for entry in _list(directory):
            children.append(entry)
            if len(entries) + len(children) > max_entries:
                break
        is_delta = any(c.is_directory and (c.name or "").rstrip("/") == "_delta_log" for c in children)
        if is_delta:
            if directory == root:
                info["root_is_delta"] = True
            else:
                parent = by_path.get(directory)
                if parent is not None:
                    parent["format"] = "delta"
                    parent["is_delta_table"] = True
                continue  # do not expand a Delta table's internals
        for child in children:
            if len(entries) >= max_entries:
                info["truncated"] = True
                return entries, info
            item = _entry_dict(child, depth)
            entries.append(item)
            if item["type"] == "directory":
                by_path[item["path"]] = item
                if item["name"] == "_delta_log":
                    continue
                if depth < max_depth:
                    queue.append((item["path"], depth + 1))
                else:
                    info["depth_limited"] = True
    return entries, info


def _probe_delta(entries: list[dict[str, Any]], expanded: set[str]) -> int:
    """For directories whose contents weren't listed, check for ``_delta_log`` (HEAD, capped)."""
    probes = 0
    for item in entries:
        if item["type"] != "directory" or item.get("format") == "delta" or item["name"] == "_delta_log":
            continue
        if item["path"] in expanded:
            continue
        if probes >= MAX_DELTA_PROBES:
            break
        probes += 1
        try:
            if _is_directory(f"{item['path']}/_delta_log"):
                item["format"] = "delta"
                item["is_delta_table"] = True
        except Exception:  # pragma: no cover - permission errors etc.; detection is best-effort
            pass
    return probes


def _summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_format: dict[str, dict[str, int]] = {}
    files = dirs = total = 0
    for item in entries:
        if item["type"] == "directory":
            dirs += 1
            if item.get("format") == "delta":
                by_format.setdefault("delta", {"count": 0, "total_bytes": 0})["count"] += 1
            continue
        files += 1
        size = item.get("size_bytes") or 0
        total += size
        bucket = by_format.setdefault(item["format"], {"count": 0, "total_bytes": 0})
        bucket["count"] += 1
        bucket["total_bytes"] += size
    return {
        "file_count": files,
        "directory_count": dirs,
        "total_file_bytes": total,
        "by_format": dict(sorted(by_format.items(), key=lambda kv: -kv[1]["count"])),
    }


# ----------------------------------------------------------------------------------------------
# get_volume_folder_details
# ----------------------------------------------------------------------------------------------

@tool(toolset="volumes", title="Volume folder details", safety=READ)
def get_volume_folder_details(
    path: Annotated[str, Field(description="Volume path: /Volumes/<catalog>/<schema>/<volume>[/sub/path].")],
    recursive: Annotated[bool, Field(description="Also list sub-directories (bounded by max_depth/max_entries).")] = False,
    max_depth: Annotated[int, Field(description=f"Max directory depth when recursive (1-{MAX_SCAN_DEPTH}).", ge=1)] = 3,
    max_entries: Annotated[
        int, Field(description=f"Max entries scanned in total (capped at {MAX_SCAN_ENTRIES}).", ge=1)
    ] = 1000,
    detect_delta: Annotated[
        bool, Field(description="Probe listed sub-directories for a _delta_log folder (Delta tables).")
    ] = True,
    page_size: PageSize = None,
    page_token: PageToken = None,
) -> ToolResponse:
    """Inspect a Unity Catalog Volume path. For a directory: entries with type (file/directory),
    size, modification time and detected format (parquet, csv, json, delta, avro, orc, text, ...),
    plus summary counts/total size by format; `recursive` walks sub-directories within max_depth /
    max_entries caps and reports directories containing `_delta_log` as Delta tables. For a file:
    its metadata (size, content type, last modified, format). Entries are paginated."""
    vp = _vpath(path)
    full = vp.full
    if vp.relative and not _is_directory(full):
        data = _file_metadata(full)  # raises NOT_FOUND if neither a file nor a directory
        return ok(f"{full} is a {data['format']} file of {data['size_bytes']} bytes.", data)

    depth = min(max_depth, MAX_SCAN_DEPTH) if recursive else 1
    cap = min(max_entries, MAX_SCAN_ENTRIES)
    entries, info = _walk(full, max_depth=depth, max_entries=cap)
    page, page_info = list_page(entries, page_size, page_token, transform=lambda x: x)
    if detect_delta:
        _probe_delta(page, info["expanded"])
    summary = _summarize(entries)
    warnings: list[str] = []
    if info["truncated"]:
        warnings.append(f"Scan stopped after {cap} entries; summary covers only the scanned entries.")
    if info["depth_limited"]:
        warnings.append(f"Sub-directories deeper than {depth} level(s) were not expanded.")
    data = {
        "path": full,
        "type": "directory",
        "volume": vp.volume_full_name,
        "is_delta_table": info["root_is_delta"],
        "recursive": recursive,
        "summary": {**summary, "truncated": info["truncated"], "depth_limited": info["depth_limited"]},
        "entries": page,
    }
    kind = "Delta table directory" if info["root_is_delta"] else "directory"
    more = " (more available - pass next_page_token)" if page_info.has_more else ""
    text = (
        f"{full} is a {kind}: {summary['file_count']} file(s), {summary['directory_count']} dir(s), "
        f"{summary['total_file_bytes']} bytes scanned. Returned {page_info.returned} entries{more}."
    )
    return ok(text, data, page=page_info, warnings=warnings)


# ----------------------------------------------------------------------------------------------
# manage_volume_files
# ----------------------------------------------------------------------------------------------

VolumeAction = Literal["list", "get_metadata", "upload", "download", "delete", "delete_directory", "create_directory"]

_ACTION_LEVELS: dict[str, frozenset[SafetyLevel]] = {
    "list": READ,
    "get_metadata": READ,
    "download": READ,
    "upload": WRITE,
    "create_directory": WRITE,
    "delete": DESTRUCTIVE,
    "delete_directory": DESTRUCTIVE,
}


def _volume_levels(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    action = args.get("action")
    if action not in _ACTION_LEVELS:
        raise DbxToolError(
            ErrorCategory.INVALID_PARAMETER,
            f"Unknown action {action!r} for manage_volume_files. Valid actions: {', '.join(_ACTION_LEVELS)}",
        )
    if action == "upload" and args.get("overwrite"):
        return WRITE | DESTRUCTIVE
    return _ACTION_LEVELS[action]


def _inline_payload(content: str | None, content_base64: str | None) -> bytes | None:
    if content is not None and content_base64 is not None:
        raise ValidationFailed("Pass only one of 'content' or 'content_base64'")
    if content is not None:
        return content.encode("utf-8")
    if content_base64 is not None:
        try:
            return base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValidationFailed(f"'content_base64' is not valid base64: {exc}") from None
    return None


def _upload_source(args: dict[str, Any]) -> tuple[bytes | None, Any, int]:
    """Validate the upload source; returns (inline_bytes, local_path, size)."""
    payload = _inline_payload(args.get("content"), args.get("content_base64"))
    local = args.get("local_path")
    if payload is not None and local:
        raise ValidationFailed("Pass either inline content or 'local_path', not both")
    if payload is None and not local:
        raise ValidationFailed("upload requires 'content', 'content_base64' or 'local_path'")
    settings = ctx().settings
    if payload is not None:
        if len(payload) > settings.max_inline_download_bytes:
            raise ValidationFailed(
                f"Inline content is {len(payload)} bytes; the inline limit is {settings.max_inline_download_bytes}.",
                hint="Use local_path (with DBX_MCP_LOCAL_FILE_ROOT) for large files.",
            )
        return payload, None, len(payload)
    local_file = resolve_local_path(settings.local_file_root, local, must_exist=True)
    return None, local_file, local_file.stat().st_size


def _dir_to_delete(path: str | None) -> VolumePath:
    vp = _vpath(path)
    if not vp.relative:
        raise ValidationFailed(f"Refusing to delete the volume root {vp.full}; volumes are managed in Unity Catalog.")
    return vp


def _existing_file(full: str) -> dict[str, Any] | None:
    try:
        return _file_metadata(full)
    except _NOT_FOUND:
        return None


def _scan_for_delete(full: str) -> tuple[list[dict[str, Any]], list[str], int, bool]:
    """Collect every file and sub-directory below ``full`` (deepest directories last)."""
    files: list[dict[str, Any]] = []
    dirs: list[str] = []
    total = 0
    queue: deque[str] = deque([full])
    seen = 0
    while queue:
        directory = queue.popleft()
        for entry in _list(directory):
            seen += 1
            if seen > MAX_RECURSIVE_DELETE_ENTRIES:
                return files, dirs, total, True
            if entry.is_directory:
                child = (entry.path or "").rstrip("/")
                dirs.append(child)
                queue.append(child)
            else:
                files.append({"path": entry.path, "size_bytes": entry.file_size})
                total += entry.file_size or 0
    return files, dirs, total, False


def _volume_preview(args: dict[str, Any]) -> PlanInfo | None:
    action = args.get("action")
    if action == "upload":
        vp = _vpath(args.get("path"), require_file=True)
        _, local, size = _upload_source(args)
        existing = _existing_file(vp.full)
        target = {"path": vp.full}
        details: dict[str, Any] = {"upload_bytes": size, "source": str(local.name) if local else "inline"}
        warnings: list[str] = []
        if existing:
            details["existing_file"] = existing
            if args.get("overwrite"):
                warnings.append(
                    f"The existing file ({existing.get('size_bytes')} bytes, modified {existing.get('last_modified')}) "
                    "will be REPLACED and cannot be recovered."
                )
                desc = f"Overwrite {vp.full} ({existing.get('size_bytes')} bytes) with {size} new bytes."
            else:
                warnings.append("A file already exists at this path; the upload will fail unless overwrite=true.")
                desc = f"Upload {size} bytes to {vp.full} (will fail: file exists and overwrite=false)."
        else:
            desc = f"Upload {size} bytes to new file {vp.full}."
        return PlanInfo(description=desc, target=target, details=details, warnings=warnings, reversible=not existing)

    if action == "delete":
        vp = _vpath(args.get("path"), require_file=True)
        meta = _file_metadata(vp.full)
        return PlanInfo(
            description=f"PERMANENTLY delete file {vp.full} ({meta['size_bytes']} bytes, format {meta['format']}).",
            target={"path": vp.full},
            details=meta,
            warnings=["Deleted volume files cannot be recovered through this server."],
            reversible=False,
        )

    if action == "delete_directory":
        vp = _dir_to_delete(args.get("path"))
        files, dirs, total, truncated = _scan_for_delete(vp.full)
        recursive = bool(args.get("recursive"))
        details = {
            "file_count": len(files),
            "subdirectory_count": len(dirs),
            "total_file_bytes": total,
            "sample_files": [f["path"] for f in files[:20]],
            "recursive": recursive,
        }
        warnings = []
        if truncated:
            warnings.append(
                f"Directory has more than {MAX_RECURSIVE_DELETE_ENTRIES} entries; recursive delete will be refused."
            )
        empty = not files and not dirs
        if empty:
            desc = f"Delete the empty directory {vp.full}."
        elif recursive:
            desc = (
                f"PERMANENTLY delete directory {vp.full} and ALL its contents: {len(files)} file(s) "
                f"({total} bytes) in {len(dirs)} sub-directorie(s)."
            )
            warnings.append(f"{len(files)} file(s) totalling {total} bytes will be permanently deleted.")
        else:
            desc = (
                f"Delete directory {vp.full}. It is NOT empty ({len(files)} file(s), {len(dirs)} sub-directorie(s)); "
                "the Files API only deletes empty directories, so this will fail unless recursive=true."
            )
            warnings.append("Directory is not empty; pass recursive=true to delete its contents too.")
        return PlanInfo(description=desc, target={"path": vp.full}, details=details, warnings=warnings, reversible=False)
    return None


@tool(
    toolset="volumes",
    title="Manage volume files",
    safety=_volume_levels,
    possible_levels=READ | WRITE | DESTRUCTIVE,
    preview=_volume_preview,
)
def manage_volume_files(
    action: Annotated[
        VolumeAction,
        Field(
            description="list: directory entries; get_metadata: file (or directory) metadata; upload: write a file; "
            "download: read a file; delete: delete a file; delete_directory: delete a directory "
            "(recursive=true deletes its contents too); create_directory: create a directory (and parents)."
        ),
    ],
    path: Annotated[str, Field(description="Volume path: /Volumes/<catalog>/<schema>/<volume>/...")],
    content: Annotated[str | None, Field(description="upload: UTF-8 text content.")] = None,
    content_base64: Annotated[str | None, Field(description="upload: binary content, base64-encoded.")] = None,
    local_path: Annotated[
        str | None,
        Field(
            description="upload: source file / download: destination file, relative to DBX_MCP_LOCAL_FILE_ROOT "
            "(local file access is disabled unless that is set)."
        ),
    ] = None,
    overwrite: Annotated[
        bool, Field(description="upload: replace an existing file (DESTRUCTIVE; requires confirm).")
    ] = False,
    recursive: Annotated[
        bool, Field(description="delete_directory: also delete all files and sub-directories inside it.")
    ] = False,
    page_size: PageSize = None,
    page_token: PageToken = None,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Unity Catalog Volume file operations: list, get_metadata, upload (inline `content` /
    `content_base64`, or `local_path`; `overwrite` replaces an existing file and is DESTRUCTIVE),
    download (returned inline up to DBX_MCP_MAX_INLINE_DOWNLOAD_BYTES - as text when UTF-8, else
    base64 - or saved to `local_path`), delete (file), delete_directory (empty dirs; `recursive`
    deletes contents after confirmation), create_directory. Paths are validated against traversal
    and the configured volume allowlist."""
    c = ctx()
    if action == "list":
        vp = _vpath(path)
        return paged_response(
            f"entries in {vp.full}", _list(vp.full), page_size, page_token, transform=_entry_dict
        )

    if action == "get_metadata":
        vp = _vpath(path)
        if not vp.relative:
            c.w.files.get_directory_metadata(vp.full)
            return ok(f"{vp.full} is a volume root directory.", {"path": vp.full, "type": "directory"})
        try:
            data = _file_metadata(vp.full)
        except _NOT_FOUND:
            if _is_directory(vp.full):
                return ok(f"{vp.full} is a directory.", {"path": vp.full, "type": "directory"})
            raise
        return ok(f"{vp.full}: {data['size_bytes']} bytes, format {data['format']}.", data)

    if action == "upload":
        vp = _vpath(path, require_file=True)
        payload, local, size = _upload_source(
            {"content": content, "content_base64": content_base64, "local_path": local_path}
        )
        if payload is not None:
            c.w.files.upload(vp.full, io.BytesIO(payload), overwrite=overwrite)
        else:
            with open(local, "rb") as fh:
                c.w.files.upload(vp.full, fh, overwrite=overwrite)
        data = {"path": vp.full, "size_bytes": size, "overwrite": overwrite, "source": "local_file" if local else "inline"}
        return ok(f"Uploaded {size} bytes to {vp.full}.", data)

    if action == "download":
        vp = _vpath(path, require_file=True)
        return _download(vp, local_path)

    if action == "delete":
        vp = _vpath(path, require_file=True)
        c.w.files.delete(vp.full)
        return ok(f"Deleted file {vp.full}.", {"path": vp.full, "deleted": True})

    if action == "delete_directory":
        vp = _dir_to_delete(path)
        if not recursive:
            c.w.files.delete_directory(vp.full)
            return ok(f"Deleted empty directory {vp.full}.", {"path": vp.full, "deleted": True})
        return _delete_recursive(vp.full)

    if action == "create_directory":
        vp = _vpath(path)
        c.w.files.create_directory(vp.full)
        return ok(f"Directory {vp.full} exists (created if missing).", {"path": vp.full, "type": "directory"})

    raise ValidationFailed(f"Unknown action {action!r}")  # pragma: no cover - Literal guards this


def _download(vp: VolumePath, local_path: str | None) -> ToolResponse:
    c = ctx()
    settings = c.settings
    if local_path:
        target = resolve_local_path(settings.local_file_root, local_path, must_exist=False)
        if target.exists():
            raise ValidationFailed(
                f"Local file {local_path!r} already exists; choose a new name (downloads never overwrite local files)."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        resp = c.w.files.download(vp.full)
        try:
            with open(target, "xb") as out:
                shutil.copyfileobj(resp.contents, out, length=1024 * 1024)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        size = os.path.getsize(target)
        data = {"path": vp.full, "local_path": str(target), "size_bytes": size, "content_type": resp.content_type}
        return ok(f"Downloaded {vp.full} ({size} bytes) to {local_path}.", data)

    cap = settings.max_inline_download_bytes
    meta = c.w.files.get_metadata(vp.full)
    if meta.content_length is not None and meta.content_length > cap:
        raise ValidationFailed(
            f"{vp.full} is {meta.content_length} bytes, above the inline download limit of {cap} bytes.",
            hint="Set DBX_MCP_LOCAL_FILE_ROOT and pass local_path to save it locally instead.",
        )
    resp = c.w.files.download(vp.full)
    try:
        raw = resp.contents.read(cap + 1) if resp.contents is not None else b""
    finally:
        close = getattr(resp.contents, "close", None)
        if callable(close):
            close()
    if len(raw) > cap:
        raise ValidationFailed(
            f"{vp.full} exceeds the inline download limit of {cap} bytes.",
            hint="Set DBX_MCP_LOCAL_FILE_ROOT and pass local_path to save it locally instead.",
        )
    fmt, _ = detect_file_format(vp.full)
    data: dict[str, Any] = {
        "path": vp.full,
        "size_bytes": len(raw),
        "content_type": resp.content_type or meta.content_type,
        "format": fmt,
    }
    try:
        if b"\x00" in raw:
            raise UnicodeDecodeError("utf-8", raw, 0, 1, "NUL byte")
        data["encoding"] = "text"
        data["content"] = raw.decode("utf-8")
    except UnicodeDecodeError:
        data["encoding"] = "base64"
        data["content"] = base64.b64encode(raw).decode("ascii")
    return ok(f"Downloaded {vp.full} ({len(raw)} bytes, returned as {data['encoding']}).", data)


def _delete_recursive(full: str) -> ToolResponse:
    c = ctx()
    files, dirs, total, truncated = _scan_for_delete(full)
    if truncated:
        raise ValidationFailed(
            f"{full} contains more than {MAX_RECURSIVE_DELETE_ENTRIES} entries; refusing a recursive delete this large.",
            hint="Delete sub-directories individually.",
        )
    deleted_files = deleted_dirs = 0
    try:
        for item in files:
            c.w.files.delete(item["path"])
            deleted_files += 1
        for directory in [*reversed(dirs), full]:  # deepest first, the target last
            c.w.files.delete_directory(directory)
            deleted_dirs += 1
    except Exception as exc:
        data = {
            "path": full,
            "deleted_files": deleted_files,
            "deleted_directories": deleted_dirs,
            "remaining_files": len(files) - deleted_files,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return ok(
            f"Recursive delete of {full} stopped after deleting {deleted_files}/{len(files)} file(s) and "
            f"{deleted_dirs} director(ies): {exc}",
            data,
            status="partial_failure",
        )
    data = {"path": full, "deleted_files": deleted_files, "deleted_directories": deleted_dirs, "deleted_bytes": total}
    return ok(
        f"Deleted {full} with {deleted_files} file(s) ({total} bytes) and {deleted_dirs - 1} sub-directorie(s).", data
    )
