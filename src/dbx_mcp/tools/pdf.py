"""PDF generation: generate_and_upload_pdf.

HTML (or Markdown / plain text converted to simple, escaped HTML) is rendered with
``xhtml2pdf`` - an optional dependency (``pip install "dbx-mcp[pdf]"``) - and uploaded
to a Unity Catalog Volume. Rendering never fetches external resources: every link the
renderer tries to load (stylesheets, images, fonts, ``url()``) goes through a callback
that only lets inline ``data:`` URIs through and replaces everything else (http(s),
``file:``, relative paths) with an empty resource.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import inspect
import io
import re
from typing import Annotated, Any

from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, WRITE, SafetyLevel
from dbx_mcp.safety.validation import VolumePath, parse_volume_path
from dbx_mcp.tools.common import Confirm, DryRun, ctx, ok, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory, ValidationFailed

#: Max size of the HTML document (after Markdown/text conversion) accepted for rendering.
MAX_HTML_BYTES = 5 * 1024 * 1024
_EMPTY_RESOURCE = "data:,"


# ----------------------------------------------------------------------------------------------
# Content -> HTML
# ----------------------------------------------------------------------------------------------

def _inline_markdown(text: str) -> str:
    """Escape, then apply a minimal inline syntax: `code`, **bold**, *italic*."""
    out = html_lib.escape(text, quote=True)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\*)", r"<i>\1</i>", out)
    return out


def markdown_to_html(markdown: str) -> str:
    """A deliberately small Markdown subset: headings, bullet/numbered lists, fenced code,
    horizontal rules and paragraphs. All text is HTML-escaped; links/images are not rendered
    as fetchable elements."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []
    list_tag: str | None = None
    in_code = False
    code: list[str] = []

    def flush_para() -> None:
        if para:
            out.append("<p>" + "<br/>".join(_inline_markdown(p) for p in para) + "</p>")
            para.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre>" + html_lib.escape("\n".join(code)) + "</pre>")
                code.clear()
                in_code = False
            else:
                flush_para()
                close_list()
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if not stripped:
            flush_para()
            close_list()
        elif heading:
            flush_para()
            close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline_markdown(heading.group(2))}</h{level}>")
        elif re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            flush_para()
            close_list()
            out.append("<hr/>")
        elif bullet or numbered:
            flush_para()
            tag = "ul" if bullet else "ol"
            if list_tag != tag:
                close_list()
                out.append(f"<{tag}>")
                list_tag = tag
            item = (bullet or numbered).group(1)  # type: ignore[union-attr]
            out.append(f"<li>{_inline_markdown(item)}</li>")
        else:
            close_list()
            para.append(stripped)
    if in_code:
        out.append("<pre>" + html_lib.escape("\n".join(code)) + "</pre>")
    flush_para()
    close_list()
    return "\n".join(out)


def text_to_html(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return "\n".join(
        "<p>" + "<br/>".join(html_lib.escape(line) for line in p.split("\n")) + "</p>" for p in paragraphs if p.strip()
    )


def _wrap_document(body: str, title: str | None) -> str:
    head_title = f"<title>{html_lib.escape(title)}</title>" if title else ""
    heading = f"<h1>{html_lib.escape(title)}</h1>\n" if title else ""
    style = (
        "<style>body{font-family:Helvetica;font-size:11pt;line-height:1.4}"
        "pre{font-family:Courier;font-size:9pt;background-color:#f4f4f4;padding:4pt}"
        "code{font-family:Courier}td,th{border:1px solid #999;padding:3pt}</style>"
    )
    return f'<html><head><meta charset="utf-8"/>{head_title}{style}</head><body>\n{heading}{body}\n</body></html>'


def build_html(html: str | None, markdown: str | None, text: str | None, title: str | None) -> str:
    given = [name for name, value in (("html", html), ("markdown", markdown), ("text", text)) if value is not None]
    if len(given) != 1:
        raise ValidationFailed("Provide exactly one of 'html', 'markdown' or 'text'")
    if html is not None:
        document = html if re.search(r"<html[\s>]", html, re.IGNORECASE) else _wrap_document(html, title)
    elif markdown is not None:
        document = _wrap_document(markdown_to_html(markdown), title)
    else:
        document = _wrap_document(text_to_html(text or ""), title)
    if not document.strip():
        raise ValidationFailed("The document is empty")
    size = len(document.encode("utf-8"))
    if size > MAX_HTML_BYTES:
        raise ValidationFailed(f"The HTML document is {size} bytes; the limit is {MAX_HTML_BYTES} bytes.")
    return document


# ----------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------

def _load_pisa() -> Any:
    try:
        from xhtml2pdf import pisa
    except ImportError as exc:
        raise DbxToolError(
            ErrorCategory.CONFIGURATION,
            "PDF generation requires the optional 'xhtml2pdf' package, which is not installed.",
            hint='Install it with: pip install "dbx-mcp[pdf]" (or pip install xhtml2pdf) and restart the server.',
        ) from exc
    return pisa


def _deny_all_policy() -> Any:
    """xhtml2pdf >= 0.2.17 also has a resource policy; use it as a second layer when present."""
    try:
        from xhtml2pdf.config.resources import ResourceAccessPolicy  # type: ignore[import-not-found]

        return ResourceAccessPolicy(allow_remote=False, base_dir=None)
    except Exception:
        return None


def render_pdf(document: str) -> tuple[bytes, list[str]]:
    """Render HTML to PDF bytes. Returns (pdf_bytes, blocked_resource_uris)."""
    pisa = _load_pisa()
    blocked: list[str] = []

    def link_callback(uri: str, rel: str | None = None) -> str:
        if isinstance(uri, str) and uri.strip().lower().startswith("data:"):
            return uri
        blocked.append(str(uri)[:200])
        return _EMPTY_RESOURCE

    kwargs: dict[str, Any] = {"dest": io.BytesIO(), "link_callback": link_callback, "encoding": "utf-8"}
    try:
        params = inspect.signature(pisa.CreatePDF).parameters
    except (TypeError, ValueError):  # pragma: no cover
        params = {}
    policy = _deny_all_policy() if "resource_policy" in params else None
    if policy is not None:
        kwargs["resource_policy"] = policy
    dest = kwargs["dest"]
    status = pisa.CreatePDF(document, **kwargs)
    pdf = dest.getvalue()
    if not pdf.startswith(b"%PDF"):
        errors = getattr(status, "err", None)
        raise ValidationFailed(f"PDF rendering failed (renderer reported {errors} error(s)); check the HTML.")
    return pdf, blocked


def _page_count(pdf: bytes) -> int | None:
    try:
        from pypdf import PdfReader  # installed with xhtml2pdf

        return len(PdfReader(io.BytesIO(pdf)).pages)
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------
# Tool
# ----------------------------------------------------------------------------------------------

def _destination(path: str | None) -> VolumePath:
    require(path, "destination_path")
    vp = parse_volume_path(path, require_file=True)  # type: ignore[arg-type]
    if not vp.relative.lower().endswith(".pdf"):
        raise ValidationFailed(f"destination_path must end with .pdf, got {vp.full!r}")
    ctx().safety.check_volume_path(vp.full)
    return vp


def _pdf_levels(args: dict[str, Any]) -> frozenset[SafetyLevel]:
    return WRITE | DESTRUCTIVE if args.get("overwrite") else WRITE


def _pdf_preview(args: dict[str, Any]) -> PlanInfo:
    vp = _destination(args.get("destination_path"))
    document = build_html(args.get("html"), args.get("markdown"), args.get("text"), args.get("title"))
    warnings: list[str] = []
    details: dict[str, Any] = {"html_bytes": len(document.encode("utf-8"))}
    existing = None
    try:
        meta = ctx().w.files.get_metadata(vp.full)
        existing = {"size_bytes": meta.content_length, "last_modified": meta.last_modified}
    except (NotFound, ResourceDoesNotExist):
        pass
    if existing:
        details["existing_file"] = existing
        if args.get("overwrite"):
            warnings.append(f"The existing PDF ({existing['size_bytes']} bytes) will be REPLACED and cannot be recovered.")
        else:
            warnings.append("A file already exists at this path; the upload will fail unless overwrite=true.")
    return PlanInfo(
        description=f"Render the document to PDF and upload it to {vp.full}"
        + (" (overwriting the existing file)." if existing and args.get("overwrite") else "."),
        target={"destination_path": vp.full},
        details=details,
        warnings=warnings,
        reversible=not existing,
    )


@tool(
    toolset="pdf",
    title="Generate and upload PDF",
    safety=_pdf_levels,
    possible_levels=WRITE | DESTRUCTIVE,
    preview=_pdf_preview,
)
def generate_and_upload_pdf(
    destination_path: Annotated[
        str, Field(description="Target file in a Unity Catalog Volume, e.g. /Volumes/main/reports/files/q3.pdf")
    ],
    html: Annotated[str | None, Field(description="HTML to render (external resources are never fetched).")] = None,
    markdown: Annotated[
        str | None, Field(description="Markdown (headings, lists, code blocks, bold/italic) to render instead of html.")
    ] = None,
    text: Annotated[str | None, Field(description="Plain text to render instead of html.")] = None,
    title: Annotated[str | None, Field(description="Optional document title (used for markdown/text/HTML fragments).")] = None,
    overwrite: Annotated[bool, Field(description="Replace an existing file (DESTRUCTIVE; requires confirm).")] = False,
    dry_run: DryRun = False,
    confirm: Confirm = False,
) -> ToolResponse:
    """Render HTML (or Markdown / plain text, converted to escaped HTML) to a PDF and upload it to a
    Unity Catalog Volume path ending in .pdf. Remote URLs, file: links and relative resources in the
    HTML are blocked (only inline data: URIs are used). Returns the path, size in bytes, page count
    and SHA-256. Requires the optional xhtml2pdf dependency (pip install "dbx-mcp[pdf]")."""
    vp = _destination(destination_path)
    document = build_html(html, markdown, text, title)
    pdf, blocked = render_pdf(document)
    ctx().w.files.upload(vp.full, io.BytesIO(pdf), overwrite=overwrite)
    data = {
        "path": vp.full,
        "size_bytes": len(pdf),
        "page_count": _page_count(pdf),
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "html_bytes": len(document.encode("utf-8")),
        "blocked_resources": blocked[:20],
    }
    warnings = []
    if blocked:
        warnings.append(
            f"{len(blocked)} external/local resource reference(s) were blocked and rendered empty "
            "(embed images as data: URIs instead)."
        )
    pages = f", {data['page_count']} page(s)" if data["page_count"] is not None else ""
    return ok(f"Uploaded PDF to {vp.full} ({len(pdf)} bytes{pages}).", data, warnings=warnings)
