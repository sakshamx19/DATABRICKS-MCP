"""Request-auth mode: per-request workspace URL + PAT from HTTP headers (multi-workspace server)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from databricks.sdk.service import iam, sql

from dbx_mcp.databricks.client import _RequestScopedConfig
from dbx_mcp.databricks.request_auth import credentials_from_headers, normalize_host
from dbx_mcp.server.config import DEFAULT_ALLOWED_WORKSPACE_HOSTS
from dbx_mcp.server.context import get_context
from dbx_mcp.server.manifest import TrackedResource
from dbx_mcp.tools.registry import make_handler, registered_tools
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory
from tests.fakes import fake_pat

ALLOWED = DEFAULT_ALLOWED_WORKSPACE_HOSTS
HOST_A = "https://adb-111.1.azuredatabricks.net"
HOST_B = "https://dbc-222.cloud.databricks.com"
TOKEN_A = fake_pat("a")
TOKEN_B = fake_pat("b")


# ---------------------------------------------------------------------------------------------- headers


def test_parse_bearer_and_optional_ids():
    creds = credentials_from_headers(
        {"X-Databricks-Host": HOST_A + "/", "Authorization": f"Bearer {TOKEN_A}",
         "X-Databricks-Warehouse-Id": "wh1", "X-Databricks-Cluster-Id": "0101-abc"},
        ALLOWED,
    )
    assert (creds.host, creds.token, creds.warehouse_id, creds.cluster_id) == (HOST_A, TOKEN_A, "wh1", "0101-abc")
    assert TOKEN_A not in repr(creds) and TOKEN_A not in creds.key


def test_parse_token_header_and_case_insensitive():
    creds = credentials_from_headers({"x-databricks-host": "adb-111.1.azuredatabricks.net",
                                      "x-databricks-token": TOKEN_A}, ALLOWED)
    assert creds.host == HOST_A and creds.token == TOKEN_A


@pytest.mark.parametrize(
    "headers,category,fragment",
    [
        ({"Authorization": f"Bearer {TOKEN_A}"}, ErrorCategory.AUTHENTICATION, "X-Databricks-Host"),
        ({"X-Databricks-Host": HOST_A}, ErrorCategory.AUTHENTICATION, "token"),
        ({"X-Databricks-Host": HOST_A, "Authorization": f"Basic {TOKEN_A}"}, ErrorCategory.AUTHENTICATION, "Bearer"),
        ({"X-Databricks-Host": "http://adb-111.1.azuredatabricks.net", "Authorization": f"Bearer {TOKEN_A}"},
         ErrorCategory.AUTHENTICATION, "https"),
        ({"X-Databricks-Host": HOST_A + "/api/2.0", "Authorization": f"Bearer {TOKEN_A}"},
         ErrorCategory.AUTHENTICATION, "root URL"),
        ({"X-Databricks-Host": HOST_A + ":8443", "Authorization": f"Bearer {TOKEN_A}"},
         ErrorCategory.AUTHENTICATION, "port"),
        ({"X-Databricks-Host": "https://user:pw@adb-1.azuredatabricks.net", "Authorization": f"Bearer {TOKEN_A}"},
         ErrorCategory.AUTHENTICATION, "credentials"),
        ({"X-Databricks-Host": "https://169.254.169.254", "Authorization": f"Bearer {TOKEN_A}"},
         ErrorCategory.SAFETY_BLOCKED, "not an allowed"),
        ({"X-Databricks-Host": "https://evil.com", "Authorization": f"Bearer {TOKEN_A}"},
         ErrorCategory.SAFETY_BLOCKED, "not an allowed"),
        ({"X-Databricks-Host": "https://azuredatabricks.net.evil.com", "Authorization": f"Bearer {TOKEN_A}"},
         ErrorCategory.SAFETY_BLOCKED, "not an allowed"),
        ({"X-Databricks-Host": HOST_A, "Authorization": "Bearer has space"}, ErrorCategory.AUTHENTICATION, "malformed"),
        ({"X-Databricks-Host": HOST_A, "Authorization": f"Bearer {TOKEN_A}", "X-Databricks-Warehouse-Id": "a b"},
         ErrorCategory.INVALID_PARAMETER, "warehouse"),
    ],
)
def test_header_validation(headers, category, fragment):
    with pytest.raises(DbxToolError) as info:
        credentials_from_headers(headers, ALLOWED)
    assert info.value.category == category
    assert fragment.lower() in str(info.value).lower()
    assert TOKEN_A not in str(info.value)


def test_no_headers_means_wrong_transport():
    with pytest.raises(DbxToolError) as info:
        credentials_from_headers(None, ALLOWED)
    assert info.value.category == ErrorCategory.CONFIGURATION


def test_host_allowlist_override():
    assert normalize_host("https://internal.example.com", (".example.com",)) == "https://internal.example.com"
    assert normalize_host("https://anything.io", ("*",)) == "https://anything.io"


# ---------------------------------------------------------------------------------------------- SDK isolation


def test_request_scoped_config_ignores_server_credentials(monkeypatch, tmp_path):
    """Pins SDK-internal behaviour: env vars and ~/.databrickscfg must never leak into a request client."""
    cfg_file = tmp_path / "databrickscfg"
    cfg_file.write_text("[DEFAULT]\nhost = https://other.cloud.databricks.com\ntoken = dapiPROFILE\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg_file))
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiENV")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "sp-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "sp-secret")
    from databricks.sdk.credentials_provider import pat_auth

    cfg = _RequestScopedConfig(host=HOST_A, token=TOKEN_A, credentials_strategy=pat_auth)
    assert cfg.host == HOST_A
    assert cfg.client_id is None and cfg.client_secret is None and cfg.profile is None
    assert cfg.authenticate()["Authorization"] == f"Bearer {TOKEN_A}"


# ---------------------------------------------------------------------------------------------- wrapper / tools


def _me(user):
    return iam.User(user_name=user, id="1", display_name=user, active=True, groups=[], emails=[])


@pytest.fixture
def multi(make_harness):
    """A request-mode harness whose client factory returns a separate mock per (host, token)."""
    from tests.conftest import autospec_workspace_client

    created: dict[str, object] = {}

    def factory(settings, creds):
        w = autospec_workspace_client()
        w.config.host = creds.host
        w.current_user.me.return_value = _me("alice@a.com" if creds.host == HOST_A else "bob@b.com")
        created[creds.host] = w
        return w

    h = make_harness(auth_mode="request", toolsets=("identity", "sql", "manifest"))
    get_context().clients._request_factory = factory
    h.created = created
    return h


async def _call(tool, headers, **args):
    handler = make_handler(registered_tools()[tool])
    return await handler(mcp_ctx=SimpleNamespace(headers=headers), **args)


def _hdr(host, token, **extra):
    return {"X-Databricks-Host": host, "Authorization": f"Bearer {token}", **extra}


async def test_each_request_uses_its_own_workspace(multi):
    a = await _call("get_current_user", _hdr(HOST_A, TOKEN_A))
    b = await _call("get_current_user", _hdr(HOST_B, TOKEN_B))
    a2 = await _call("get_current_user", _hdr(HOST_A, TOKEN_A))
    assert a.data["user_name"] == "alice@a.com" and a.data["workspace"]["host"] == HOST_A
    assert b.data["user_name"] == "bob@b.com" and b.data["workspace"]["host"] == HOST_B
    assert a2.data["user_name"] == "alice@a.com"
    assert set(multi.created) == {HOST_A, HOST_B}  # client cached: A built once
    for response in (a, b):
        dumped = response.model_dump_json()
        assert TOKEN_A not in dumped and TOKEN_B not in dumped


async def test_missing_headers_rejected(multi):
    with pytest.raises(DbxToolError) as info:
        await _call("get_current_user", {})
    assert info.value.category == ErrorCategory.AUTHENTICATION
    with pytest.raises(DbxToolError) as info:
        await _call("get_current_user", None)
    assert info.value.category == ErrorCategory.CONFIGURATION


async def test_profile_switching_unsupported(multi):
    with pytest.raises(DbxToolError) as info:
        await _call("manage_workspace", _hdr(HOST_A, TOKEN_A), action="switch_profile", profile="x")
    assert info.value.category == ErrorCategory.UNSUPPORTED


async def test_warehouse_default_comes_from_connection_header(multi):
    from dbx_mcp.databricks.request_auth import request_credentials_var

    # A server-wide default would belong to one workspace only, so it is ignored in request mode.
    object.__setattr__(get_context().settings, "default_warehouse_id", "server-wh")
    resp = sql.StatementResponse(statement_id="s", status=sql.StatementStatus(state=sql.StatementState.SUCCEEDED))
    await _call("get_current_user", _hdr(HOST_A, TOKEN_A))  # creates the client for A
    multi.created[HOST_A].statement_execution.execute_statement.return_value = resp

    out = await _call("execute_sql", _hdr(HOST_A, TOKEN_A, **{"X-Databricks-Warehouse-Id": "conn-wh"}),
                      statement="SELECT 1")
    assert out.data.execution.warehouse.warehouse_id == "conn-wh"
    assert "X-Databricks-Warehouse-Id" in out.data.execution.warehouse.reason
    assert request_credentials_var.get() is None  # reset after the call


async def test_manifest_is_scoped_per_workspace(multi):
    store = get_context().manifest
    store.track(TrackedResource(resource_type="job", resource_id="42", name="a-job", workspace_host=HOST_A))
    store.track(TrackedResource(resource_type="job", resource_id="42", name="b-job", workspace_host=HOST_B))
    assert len(store.list()) == 2  # same id in two workspaces no longer collides

    listed = await _call("list_tracked_resources", _hdr(HOST_A, TOKEN_A))
    assert [r["name"] for r in listed.data["resources"]] == ["a-job"]

    await _call("delete_tracked_resource", _hdr(HOST_B, TOKEN_B), resource_type="job", resource_id="42")
    assert [r.name for r in store.list()] == ["a-job"]


def test_client_cache_is_bounded(make_harness):
    from dbx_mcp.databricks.request_auth import RequestCredentials, request_credentials_var

    make_harness(auth_mode="request", toolsets=("identity",), request_client_cache_size=2)
    clients = get_context().clients
    clients._request_factory = lambda s, c: object()
    for i in range(4):
        token = request_credentials_var.set(RequestCredentials(host=HOST_A, token=f"dapi{i}"))
        try:
            clients.workspace()
        finally:
            request_credentials_var.reset(token)
    assert len(clients._request_clients) == 2


# ---------------------------------------------------------------------------------------------- CLI + HTTP e2e


def test_cli_rejects_request_mode_on_stdio():
    env = {k: v for k, v in os.environ.items() if not k.startswith("DBX_MCP_")}
    proc = subprocess.run([sys.executable, "-m", "dbx_mcp", "--auth-mode", "request"], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "HTTP transport" in proc.stderr


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_http_end_to_end_without_server_credentials(tmp_path):
    """Real streamable-HTTP server in request mode with NO Databricks credentials configured."""
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DBX_MCP_", "DATABRICKS_"))}
    env.update({"DBX_MCP_MANIFEST_PATH": str(tmp_path / "m.json"),
                "DATABRICKS_CONFIG_FILE": str(tmp_path / "none.cfg")})
    proc = subprocess.Popen([sys.executable, "-m", "dbx_mcp", "--auth-mode", "request", "--transport",
                             "streamable-http", "--port", str(port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.5)

        async def call(headers, tool="get_current_user", args=None):
            async with httpx2.AsyncClient(headers=headers, timeout=30) as http, \
                    streamable_http_client(url, http_client=http) as (read, write, *_), \
                    ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                result = await session.call_tool(tool, args or {})
                return len(tools), result

        n_tools, result = await call({})
        assert n_tools == 45  # discovery works without credentials
        assert result.is_error and "AUTHENTICATION_FAILED" in result.content[0].text

        _, result = await call(_hdr("https://evil.example.com", TOKEN_A))
        assert result.is_error and "BLOCKED_BY_SAFETY_POLICY" in result.content[0].text
        assert TOKEN_A not in json.dumps(result.model_dump(mode="json"))
    finally:
        proc.terminate()
        proc.wait(timeout=10)
