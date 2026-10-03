"""PDF toolset tests: generate_and_upload_pdf."""

from __future__ import annotations

import builtins
import hashlib
import socket

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.files import GetMetadataResponse

from dbx_mcp.tools.pdf import MAX_HTML_BYTES, build_html, markdown_to_html

pytest.importorskip("xhtml2pdf")

TOOLSETS = ("pdf",)
DEST = "/Volumes/main/reports/files/report.pdf"


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=TOOLSETS)


@pytest.fixture
def no_network(monkeypatch):
    """Fail on any non-loopback connection (the asyncio event loop uses a loopback socket pair)."""
    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"network access attempted: {address}")
        return real_connect(self, address, *args, **kwargs)

    def refuse(address, *args, **kwargs):
        raise AssertionError(f"network access attempted: {address}")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def _uploaded(h) -> bytes:
    args, kwargs = h.w.files.upload.call_args
    assert args[0] == DEST
    return args[1].getvalue()


async def test_generates_pdf_and_uploads(h, no_network):
    out = await h.call("generate_and_upload_pdf", {"destination_path": DEST, "html": "<h1>Q3</h1><p>Revenue up</p>"})
    assert out["status"] == "success"
    pdf = _uploaded(h)
    assert pdf.startswith(b"%PDF")
    data = out["data"]
    assert data["path"] == DEST and data["size_bytes"] == len(pdf)
    assert data["sha256"] == hashlib.sha256(pdf).hexdigest()
    assert data["page_count"] == 1
    assert h.w.files.upload.call_args.kwargs["overwrite"] is False


async def test_blocks_remote_and_file_resources(h, no_network):
    html = (
        '<html><head><link rel="stylesheet" href="https://evil.example.com/s.css">'
        '<style>@import url("http://evil.example.com/i.css");</style></head><body>'
        '<img src="https://evil.example.com/a.png"/><img src="file:///etc/passwd"/><img src="../secret.png"/>'
        "<p>ok</p></body></html>"
    )
    out = await h.call("generate_and_upload_pdf", {"destination_path": DEST, "html": html})
    assert _uploaded(h).startswith(b"%PDF")
    blocked = out["data"]["blocked_resources"]
    assert "https://evil.example.com/a.png" in blocked
    assert "file:///etc/passwd" in blocked
    assert any("blocked" in w for w in out["warnings"])


async def test_markdown_and_text_are_escaped(h, no_network):
    out = await h.call("generate_and_upload_pdf",
                       {"destination_path": DEST, "markdown": "# Title\n\n- <b>x</b>\n\n![i](http://x/y.png)"})
    assert out["status"] == "success"
    html = markdown_to_html("- <script>alert(1)</script>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    doc = build_html(None, None, "a < b\n\nnext", "T&C")
    assert "a &lt; b" in doc and "T&amp;C" in doc
    out = await h.call("generate_and_upload_pdf", {"destination_path": DEST, "text": "plain text"})
    assert _uploaded(h).startswith(b"%PDF")


@pytest.mark.parametrize(
    ("dest", "fragment"),
    [
        ("/Volumes/main/reports/files/report.txt", "must end with .pdf"),
        ("/Volumes/main/reports/files/../x.pdf", "INVALID_PARAMETER"),
        ("/tmp/report.pdf", "INVALID_PARAMETER"),
        ("/Volumes/main/reports/files", "INVALID_PARAMETER"),
    ],
)
async def test_destination_validation(h, dest, fragment):
    err = await h.call_error("generate_and_upload_pdf", {"destination_path": dest, "html": "<p>x</p>"})
    assert fragment in err
    h.w.files.upload.assert_not_called()


async def test_destination_allowlist(make_harness):
    h = make_harness(toolsets=TOOLSETS, allowed_volume_prefixes=("/Volumes/main/default/v",))
    err = await h.call_error("generate_and_upload_pdf", {"destination_path": DEST, "html": "<p>x</p>"})
    assert "[BLOCKED_BY_SAFETY_POLICY]" in err


async def test_exactly_one_content_source(h):
    err = await h.call_error("generate_and_upload_pdf", {"destination_path": DEST})
    assert "exactly one" in err
    err = await h.call_error("generate_and_upload_pdf", {"destination_path": DEST, "html": "<p/>", "text": "x"})
    assert "exactly one" in err


async def test_html_size_cap(h):
    err = await h.call_error("generate_and_upload_pdf", {"destination_path": DEST, "html": "x" * (MAX_HTML_BYTES + 1)})
    assert "limit" in err


async def test_overwrite_requires_confirmation(h, no_network):
    h.w.files.get_metadata.return_value = GetMetadataResponse(content_length=1234)
    args = {"destination_path": DEST, "html": "<p>x</p>", "overwrite": True}
    out = await h.call("generate_and_upload_pdf", args)
    assert out["status"] == "confirmation_required" and "DESTRUCTIVE" in out["safety"]
    assert any("1234 bytes" in w for w in out["warnings"])
    h.w.files.upload.assert_not_called()
    out = await h.call("generate_and_upload_pdf", {**args, "confirm": True})
    assert out["status"] == "success"
    assert h.w.files.upload.call_args.kwargs["overwrite"] is True


async def test_dry_run_does_not_upload(h):
    h.w.files.get_metadata.side_effect = NotFound("no")
    out = await h.call("generate_and_upload_pdf", {"destination_path": DEST, "html": "<p>x</p>", "dry_run": True})
    assert out["status"] == "dry_run"
    h.w.files.upload.assert_not_called()


async def test_read_only_blocks(make_harness):
    h = make_harness(toolsets=TOOLSETS, read_only=True)
    err = await h.call_error("generate_and_upload_pdf", {"destination_path": DEST, "html": "<p>x</p>"})
    assert "read-only" in err


async def test_missing_dependency_is_configuration_error(h, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("xhtml2pdf"):
            raise ImportError("No module named 'xhtml2pdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    err = await h.call_error("generate_and_upload_pdf", {"destination_path": DEST, "html": "<p>x</p>"})
    assert "[CONFIGURATION_ERROR]" in err and "dbx-mcp[pdf]" in err
    h.w.files.upload.assert_not_called()
