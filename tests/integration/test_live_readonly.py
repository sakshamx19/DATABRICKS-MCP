"""Live, read-only integration tests against a real Databricks workspace.

Opt-in only: set DBX_MCP_RUN_INTEGRATION=1 plus normal Databricks auth (DATABRICKS_HOST +
DATABRICKS_TOKEN, or a profile). Optionally INTEGRATION_ENV_FILE=path/to/.env.

The server is forced into read-only mode, so these tests cannot create, change or delete
anything. SQL is only executed if a warehouse is already RUNNING (never starts one).
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("DBX_MCP_RUN_INTEGRATION") != "1", reason="set DBX_MCP_RUN_INTEGRATION=1"),
]


@pytest.fixture
def live(tmp_path):
    env_file = os.environ.get("INTEGRATION_ENV_FILE")
    if env_file:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    from dbx_mcp.server.app import build_server
    from dbx_mcp.server.config import Settings

    settings = replace(
        Settings.from_env(),
        read_only=True,
        toolsets=("identity", "sql", "compute"),
        manifest_path=tmp_path / "manifest.json",
    )
    server, _ = build_server(settings)
    return server


async def _call(server, name, args=None):
    result = await server.call_tool(name, args or {})
    assert not result.is_error, result.content
    return result.structured_content


async def test_current_user(live):
    out = await _call(live, "get_current_user")
    assert out["data"]["user_name"]
    assert "token" not in str(out).lower() or "REDACTED" in str(out)


async def test_workspace_info(live):
    out = await _call(live, "manage_workspace", {"action": "info"})
    assert out["data"]["host"].startswith("https://")
    assert out["data"]["server"]["read_only"] is True


async def test_list_compute(live):
    out = await _call(live, "list_compute")
    assert "counts" in out["data"]


async def test_write_is_blocked(live):
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="BLOCKED_BY_SAFETY_POLICY"):
        await live.call_tool("manage_cluster", {"action": "create", "spec": {"spark_version": "x"}})


async def test_select_on_running_warehouse(live):
    warehouses = (await _call(live, "manage_warehouse", {"action": "list"}))["data"]
    running = [w for w in warehouses if w.get("state") == "RUNNING"]
    if not running:
        pytest.skip("no RUNNING warehouse; not starting one")
    out = await _call(live, "execute_sql", {"statement": "SELECT 1 AS one", "warehouse_id": running[0]["id"]})
    assert out["data"]["result"]["rows"] == [[1]]
