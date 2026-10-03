"""End-to-end over the real MCP stdio protocol: spawn `python -m dbx_mcp` and talk to it.

No Databricks credentials are needed: listing tools and a read-only-mode refusal never call
Databricks. This also proves nothing but protocol messages is written to stdout.
"""

from __future__ import annotations

import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.fakes import fake_pat


async def test_stdio_handshake_list_and_call(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DBX_MCP_", "DATABRICKS_"))}
    env.update({
        "DBX_MCP_READ_ONLY": "true",
        "DBX_MCP_MANIFEST_PATH": str(tmp_path / "manifest.json"),
        "DATABRICKS_HOST": "https://example.invalid",
        "DATABRICKS_TOKEN": fake_pat("0"),
        "DATABRICKS_CONFIG_FILE": str(tmp_path / "nonexistent.cfg"),
    })
    params = StdioServerParameters(command=sys.executable, args=["-m", "dbx_mcp"], env=env)
    with open(tmp_path / "stderr.log", "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = {t.name for t in tools}
            assert len(tools) == 45
            assert {"execute_sql", "manage_cluster", "manage_uc_grants", "ask_genie"} <= names

            result = await session.call_tool("manage_cluster", {"action": "delete", "cluster_id": "x", "confirm": True})
            assert result.is_error
            assert "BLOCKED_BY_SAFETY_POLICY" in result.content[0].text

    log = (tmp_path / "stderr.log").read_text(encoding="utf-8")
    assert "server built" in log  # structured logs go to stderr, never stdout
    assert fake_pat("0") not in log
