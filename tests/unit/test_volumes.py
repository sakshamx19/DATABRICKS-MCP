"""Volumes toolset tests: get_volume_folder_details, manage_volume_files."""

from __future__ import annotations

import base64
import io

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.files import DirectoryEntry, DownloadResponse, GetMetadataResponse

from dbx_mcp.tools.volumes import detect_file_format

TOOLSETS = ("volumes",)
ROOT = "/Volumes/main/default/v"


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=TOOLSETS)


def _f(path, size=10, modified=1_700_000_000_000):
    return DirectoryEntry(path=path, name=path.rsplit("/", 1)[-1], is_directory=False, file_size=size,
                          last_modified=modified)


def _d(path):
    return DirectoryEntry(path=path + "/", name=path.rsplit("/", 1)[-1], is_directory=True)


def _listing(tree):
    def list_directory_contents(path, **_):
        return iter(tree.get(path.rstrip("/"), []))
    return list_directory_contents


# --- format detection --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "fmt", "compression"),
    [
        ("a.parquet", "parquet", None),
        ("part-0000.snappy.parquet", "parquet", "snappy"),
        ("data.csv.gz", "csv", "gz"),
        ("x.json", "json", None),
        ("x.ndjson", "json", None),
        ("y.avro", "avro", None),
        ("z.orc", "orc", None),
        ("notes.txt", "text", None),
        ("README", "unknown", None),
    ],
)
def test_detect_file_format(name, fmt, compression):
    assert detect_file_format(name) == (fmt, compression)


# --- path validation -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad",
    ["/Volumes/a/b/c/../x", "/etc/passwd", "\\Volumes\\a\\b\\c", "/Volumes/a/b/c/x\\..\\y", "/Volumes/a/b", "Volumes/a/b/c"],
)
async def test_path_traversal_rejected(h, bad):
    err = await h.call_error("manage_volume_files", {"action": "list", "path": bad})
    assert "[INVALID_PARAMETER]" in err
    err = await h.call_error("get_volume_folder_details", {"path": bad})
    assert "[INVALID_PARAMETER]" in err
    h.w.files.list_directory_contents.assert_not_called()


async def test_allowlist_blocks_other_volumes(make_harness):
    h = make_harness(toolsets=TOOLSETS, allowed_volume_prefixes=(ROOT,))
    h.w.files.list_directory_contents.side_effect = _listing({ROOT: [_f(f"{ROOT}/a.csv")]})
    out = await h.call("manage_volume_files", {"action": "list", "path": ROOT})
    assert out["data"][0]["format"] == "csv"
    err = await h.call_error("manage_volume_files", {"action": "list", "path": "/Volumes/main/default/other"})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err
    # prefix must match on a path boundary
    err = await h.call_error("manage_volume_files", {"action": "list", "path": "/Volumes/main/default/v2"})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err


# --- get_volume_folder_details -----------------------------------------------------------------

async def test_folder_details_summary_and_delta(h):
    tree = {
        ROOT: [_f(f"{ROOT}/a.parquet", 100), _f(f"{ROOT}/b.csv", 50), _f(f"{ROOT}/c.csv.gz", 5), _d(f"{ROOT}/sales"),
               _d(f"{ROOT}/raw")],
        f"{ROOT}/sales": [_d(f"{ROOT}/sales/_delta_log"), _f(f"{ROOT}/sales/part-0.snappy.parquet", 999)],
        f"{ROOT}/raw": [_f(f"{ROOT}/raw/e.json", 7)],
    }
    h.w.files.list_directory_contents.side_effect = _listing(tree)
    out = await h.call("get_volume_folder_details", {"path": ROOT + "/", "recursive": True})
    data = out["data"]
    by_name = {e["name"]: e for e in data["entries"]}
    assert by_name["sales"]["format"] == "delta" and by_name["sales"]["is_delta_table"]
    assert "part-0.snappy.parquet" not in by_name  # delta internals are not expanded
    assert by_name["e.json"]["depth"] == 2
    assert by_name["a.parquet"]["modified_at"].startswith("2023-11-14")
    summary = data["summary"]
    assert summary["file_count"] == 4
    assert summary["total_file_bytes"] == 162
    assert summary["by_format"]["csv"] == {"count": 2, "total_bytes": 55}
    assert summary["by_format"]["delta"]["count"] == 1
    assert data["is_delta_table"] is False


async def test_folder_details_non_recursive_probes_delta(h):
    h.w.files.list_directory_contents.side_effect = _listing({ROOT: [_d(f"{ROOT}/t1"), _d(f"{ROOT}/plain")]})

    def head(path):
        if path != f"{ROOT}/t1/_delta_log":
            raise NotFound("no")
    h.w.files.get_directory_metadata.side_effect = head
    out = await h.call("get_volume_folder_details", {"path": ROOT})
    formats = {e["name"]: e["format"] for e in out["data"]["entries"]}
    assert formats == {"t1": "delta", "plain": "directory"}


async def test_folder_details_root_is_delta_table(h):
    tree = {ROOT + "/tbl": [_d(f"{ROOT}/tbl/_delta_log"), _f(f"{ROOT}/tbl/p.parquet")]}
    h.w.files.list_directory_contents.side_effect = _listing(tree)
    out = await h.call("get_volume_folder_details", {"path": ROOT + "/tbl", "detect_delta": False})
    assert out["data"]["is_delta_table"] is True
    assert "Delta table" in out["summary"]


async def test_folder_details_caps_and_pagination(h):
    h.w.files.list_directory_contents.side_effect = _listing({ROOT: [_f(f"{ROOT}/f{i}.csv") for i in range(30)]})
    out = await h.call("get_volume_folder_details", {"path": ROOT, "max_entries": 20, "page_size": 5})
    assert out["data"]["summary"]["truncated"] is True
    assert out["data"]["summary"]["file_count"] == 20
    assert len(out["data"]["entries"]) == 5 and out["page"]["has_more"]
    assert any("stopped after 20" in w for w in out["warnings"])
    nxt = await h.call("get_volume_folder_details",
                       {"path": ROOT, "max_entries": 20, "page_size": 5, "page_token": out["page"]["next_page_token"]})
    assert nxt["data"]["entries"][0]["name"] == "f5.csv"


async def test_folder_details_for_file(h):
    h.w.files.get_directory_metadata.side_effect = NotFound("not a dir")
    h.w.files.get_metadata.return_value = GetMetadataResponse(content_length=42, content_type="text/csv",
                                                              last_modified="Tue, 01 Oct 2024 00:00:00 GMT")
    out = await h.call("get_volume_folder_details", {"path": f"{ROOT}/x.csv"})
    assert out["data"]["type"] == "file" and out["data"]["format"] == "csv" and out["data"]["size_bytes"] == 42


async def test_folder_details_not_found(h):
    h.w.files.get_directory_metadata.side_effect = NotFound("nope")
    h.w.files.get_metadata.side_effect = NotFound("File not found")
    err = await h.call_error("get_volume_folder_details", {"path": f"{ROOT}/missing"})
    assert "[NOT_FOUND]" in err


# --- manage_volume_files -----------------------------------------------------------------------

async def test_list_and_get_metadata(h):
    h.w.files.list_directory_contents.side_effect = _listing({ROOT: [_f(f"{ROOT}/a.json", 3), _d(f"{ROOT}/d")]})
    out = await h.call("manage_volume_files", {"action": "list", "path": ROOT})
    assert [(e["name"], e["type"]) for e in out["data"]] == [("a.json", "file"), ("d", "directory")]
    assert out["data"][1]["path"] == f"{ROOT}/d"

    h.w.files.get_metadata.return_value = GetMetadataResponse(content_length=3, content_type="application/json")
    out = await h.call("manage_volume_files", {"action": "get_metadata", "path": f"{ROOT}/a.json"})
    assert out["data"]["size_bytes"] == 3 and out["data"]["format"] == "json"


async def test_upload_inline_text_and_base64(h):
    out = await h.call("manage_volume_files", {"action": "upload", "path": f"{ROOT}/a.txt", "content": "héllo"})
    assert out["status"] == "success" and out["data"]["size_bytes"] == len("héllo".encode())
    args, kwargs = h.w.files.upload.call_args
    assert args[0] == f"{ROOT}/a.txt" and args[1].read() == "héllo".encode() and kwargs["overwrite"] is False

    payload = bytes(range(256))
    await h.call("manage_volume_files",
                 {"action": "upload", "path": f"{ROOT}/b.bin", "content_base64": base64.b64encode(payload).decode()})
    assert h.w.files.upload.call_args.args[1].read() == payload

    err = await h.call_error("manage_volume_files", {"action": "upload", "path": f"{ROOT}/c", "content_base64": "@@"})
    assert "not valid base64" in err
    err = await h.call_error("manage_volume_files", {"action": "upload", "path": ROOT, "content": "x"})
    assert "[INVALID_PARAMETER]" in err


async def test_upload_overwrite_is_destructive(h):
    h.w.files.get_metadata.return_value = GetMetadataResponse(content_length=500, last_modified="yesterday")
    args = {"action": "upload", "path": f"{ROOT}/a.txt", "content": "new", "overwrite": True}
    out = await h.call("manage_volume_files", args)
    assert out["status"] == "confirmation_required"
    assert "DESTRUCTIVE" in out["safety"]
    assert "500 bytes" in out["plan"]["description"]
    h.w.files.upload.assert_not_called()
    out = await h.call("manage_volume_files", {**args, "confirm": True})
    assert out["status"] == "success"
    assert h.w.files.upload.call_args.kwargs["overwrite"] is True


async def test_upload_dry_run(h):
    h.w.files.get_metadata.side_effect = NotFound("no file")
    out = await h.call("manage_volume_files",
                       {"action": "upload", "path": f"{ROOT}/a.txt", "content": "abc", "dry_run": True})
    assert out["status"] == "dry_run"
    assert "3 bytes" in out["plan"]["description"]
    h.w.files.upload.assert_not_called()


async def test_local_file_root_disabled_by_default(h):
    err = await h.call_error("manage_volume_files", {"action": "upload", "path": f"{ROOT}/a", "local_path": "x.csv"})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err and "Local filesystem access is disabled" in err
    err = await h.call_error("manage_volume_files", {"action": "download", "path": f"{ROOT}/a", "local_path": "x"})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err


async def test_local_upload_and_download_within_root(make_harness, tmp_path):
    root = (tmp_path / "files").resolve()
    root.mkdir()
    (root / "in.csv").write_bytes(b"a,b\n1,2\n")
    h = make_harness(toolsets=TOOLSETS, local_file_root=root)
    out = await h.call("manage_volume_files", {"action": "upload", "path": f"{ROOT}/in.csv", "local_path": "in.csv"})
    assert out["data"]["size_bytes"] == 8 and out["data"]["source"] == "local_file"

    err = await h.call_error("manage_volume_files",
                             {"action": "upload", "path": f"{ROOT}/x", "local_path": "../outside.txt"})
    assert "escapes" in err

    h.w.files.download.return_value = DownloadResponse(contents=io.BytesIO(b"payload"), content_type="text/plain")
    out = await h.call("manage_volume_files", {"action": "download", "path": f"{ROOT}/in.csv", "local_path": "out/x.csv"})
    assert (root / "out" / "x.csv").read_bytes() == b"payload"
    err = await h.call_error("manage_volume_files",
                             {"action": "download", "path": f"{ROOT}/in.csv", "local_path": "out/x.csv"})
    assert "already exists" in err


async def test_inline_download_text_binary_and_cap(make_harness):
    h = make_harness(toolsets=TOOLSETS, max_inline_download_bytes=1024)
    h.w.files.get_metadata.return_value = GetMetadataResponse(content_length=5, content_type="text/plain")
    h.w.files.download.return_value = DownloadResponse(contents=io.BytesIO(b"hello"))
    out = await h.call("manage_volume_files", {"action": "download", "path": f"{ROOT}/a.txt"})
    assert out["data"]["encoding"] == "text" and out["data"]["content"] == "hello"

    h.w.files.download.return_value = DownloadResponse(contents=io.BytesIO(b"\x89PNG\x00\xff"))
    out = await h.call("manage_volume_files", {"action": "download", "path": f"{ROOT}/a.png"})
    assert out["data"]["encoding"] == "base64"
    assert base64.b64decode(out["data"]["content"]) == b"\x89PNG\x00\xff"

    h.w.files.download.reset_mock()
    h.w.files.get_metadata.return_value = GetMetadataResponse(content_length=5000)
    err = await h.call_error("manage_volume_files", {"action": "download", "path": f"{ROOT}/big.bin"})
    assert "inline download limit" in err
    h.w.files.download.assert_not_called()

    # metadata may under-report; the stream read is capped too
    h.w.files.get_metadata.return_value = GetMetadataResponse(content_length=None)
    h.w.files.download.return_value = DownloadResponse(contents=io.BytesIO(b"x" * 2000))
    err = await h.call_error("manage_volume_files", {"action": "download", "path": f"{ROOT}/big.bin"})
    assert "inline download limit" in err


async def test_delete_requires_confirmation(h):
    h.w.files.get_metadata.return_value = GetMetadataResponse(content_length=77)
    args = {"action": "delete", "path": f"{ROOT}/old.parquet"}
    out = await h.call("manage_volume_files", args)
    assert out["status"] == "confirmation_required"
    assert "77 bytes" in out["plan"]["description"] and out["plan"]["reversible"] is False
    h.w.files.delete.assert_not_called()

    out = await h.call("manage_volume_files", {**args, "dry_run": True, "confirm": True})
    assert out["status"] == "dry_run"
    h.w.files.delete.assert_not_called()

    out = await h.call("manage_volume_files", {**args, "confirm": True})
    assert out["status"] == "success"
    h.w.files.delete.assert_called_once_with(f"{ROOT}/old.parquet")


async def test_delete_missing_file_maps_not_found(h):
    h.w.files.get_metadata.side_effect = NotFound("File does not exist")
    err = await h.call_error("manage_volume_files", {"action": "delete", "path": f"{ROOT}/nope"})
    assert "[NOT_FOUND]" in err


async def test_delete_directory_recursive_preview_counts_and_executes(h):
    d = f"{ROOT}/dir"
    tree = {d: [_f(f"{d}/a.csv", 10), _d(f"{d}/sub")], f"{d}/sub": [_f(f"{d}/sub/b.csv", 20), _f(f"{d}/sub/c.csv", 30)]}
    h.w.files.list_directory_contents.side_effect = _listing(tree)
    args = {"action": "delete_directory", "path": d, "recursive": True}
    out = await h.call("manage_volume_files", args)
    assert out["status"] == "confirmation_required"
    assert out["plan"]["details"]["file_count"] == 3
    assert "3 file(s)" in out["plan"]["description"] and "60 bytes" in out["plan"]["description"]
    h.w.files.delete.assert_not_called()
    h.w.files.delete_directory.assert_not_called()

    out = await h.call("manage_volume_files", {**args, "confirm": True})
    assert out["data"]["deleted_files"] == 3
    deleted_dirs = [c.args[0] for c in h.w.files.delete_directory.call_args_list]
    assert deleted_dirs == [f"{d}/sub", d]


async def test_delete_directory_non_recursive_warns_when_not_empty(h):
    d = f"{ROOT}/dir"
    h.w.files.list_directory_contents.side_effect = _listing({d: [_f(f"{d}/a.csv")]})
    out = await h.call("manage_volume_files", {"action": "delete_directory", "path": d})
    assert out["status"] == "confirmation_required"
    assert any("not empty" in w for w in out["warnings"])
    out = await h.call("manage_volume_files", {"action": "delete_directory", "path": d, "confirm": True})
    h.w.files.delete_directory.assert_called_once_with(d)
    h.w.files.delete.assert_not_called()


async def test_delete_volume_root_refused(h):
    err = await h.call_error("manage_volume_files", {"action": "delete_directory", "path": ROOT, "confirm": True})
    assert "volume root" in err


async def test_recursive_delete_partial_failure(h):
    d = f"{ROOT}/dir"
    h.w.files.list_directory_contents.side_effect = _listing({d: [_f(f"{d}/a"), _f(f"{d}/b")]})
    h.w.files.delete.side_effect = [None, RuntimeError("boom")]
    out = await h.call("manage_volume_files",
                       {"action": "delete_directory", "path": d, "recursive": True, "confirm": True})
    assert out["status"] == "partial_failure"
    assert out["data"]["deleted_files"] == 1 and out["data"]["remaining_files"] == 1


async def test_create_directory(h):
    out = await h.call("manage_volume_files", {"action": "create_directory", "path": f"{ROOT}/new/dir"})
    assert out["status"] == "success"
    h.w.files.create_directory.assert_called_once_with(f"{ROOT}/new/dir")


async def test_read_only_mode_blocks_writes_allows_reads(make_harness):
    h = make_harness(toolsets=TOOLSETS, read_only=True)
    for args in (
        {"action": "upload", "path": f"{ROOT}/a", "content": "x"},
        {"action": "delete", "path": f"{ROOT}/a", "confirm": True},
        {"action": "create_directory", "path": f"{ROOT}/d"},
    ):
        err = await h.call_error("manage_volume_files", args)
        assert "read-only" in err
    h.w.files.upload.assert_not_called()
    h.w.files.list_directory_contents.side_effect = _listing({ROOT: []})
    out = await h.call("manage_volume_files", {"action": "list", "path": ROOT})
    assert out["status"] == "success"
