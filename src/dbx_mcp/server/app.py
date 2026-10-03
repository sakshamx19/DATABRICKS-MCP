"""MCP server assembly."""

from __future__ import annotations

import importlib

from mcp.server.mcpserver import MCPServer

from dbx_mcp import __version__
from dbx_mcp.databricks.client import ClientProvider
from dbx_mcp.server.config import ALL_TOOLSETS, Settings
from dbx_mcp.server.context import AppContext, set_context
from dbx_mcp.tools.registry import ToolSpec, register_tools
from dbx_mcp.utils.logging import get_logger

log = get_logger("server")

# toolset name -> module that defines it
TOOL_MODULES: dict[str, str] = {name: f"dbx_mcp.tools.{name}" for name in ALL_TOOLSETS}

INSTRUCTIONS = """\
Tools for a Databricks workspace (SQL, compute, jobs, pipelines, Unity Catalog, volumes, AI/BI, \
vector search, Lakebase, apps).

Conventions:
- Every response has `status`, `summary`, `data`, and optional `page`, `plan`, `warnings`, `next_steps`.
- List actions are paginated: pass `next_page_token` back as `page_token`.
- Changes that are DESTRUCTIVE or SECURITY_SENSITIVE return status `confirmation_required` with a \
`plan` and change nothing. Show the plan to the user and only re-call with `confirm=true` after \
they explicitly approve. Never set confirm=true on your own initiative.
- `dry_run=true` previews any change without executing it.
- Long-running operations return quickly with an id and state (status `pending`); poll with the \
matching get/status action instead of waiting.
- Genie answers are model-generated; treat them as suggestions, not authoritative data.
"""


def load_tool_modules(toolsets: tuple[str, ...]) -> None:
    for toolset in toolsets:
        importlib.import_module(TOOL_MODULES[toolset])


def build_server(settings: Settings, clients: ClientProvider | None = None) -> tuple[MCPServer, list[ToolSpec]]:
    ctx = AppContext.create(settings, clients)
    set_context(ctx)
    load_tool_modules(settings.toolsets)
    server = MCPServer(
        name="dbx-mcp",
        title="Databricks MCP Server",
        version=__version__,
        instructions=INSTRUCTIONS,
        log_level=settings.log_level if settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO",
    )
    enabled = register_tools(server, toolsets=settings.toolsets, disabled=settings.disabled_tools)
    log.info(
        "server built",
        extra={
            "version": __version__,
            "tools": len(enabled),
            "toolsets": list(settings.toolsets),
            "read_only": settings.read_only,
            "blocked_levels": sorted(level.value for level in settings.blocked_safety_levels),
        },
    )
    return server, enabled
