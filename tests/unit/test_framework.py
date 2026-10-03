"""Tool framework tests: registration rules, safety gates, timeouts, redaction, MCP schemas."""

from __future__ import annotations

import time

import pytest
from databricks.sdk.service import iam

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import DESTRUCTIVE, READ, WRITE, SafetyLevel
from dbx_mcp.tools import registry
from dbx_mcp.tools.common import Confirm, DryRun, ok
from dbx_mcp.tools.registry import tool
from tests.fakes import FAKE_PAT

CORE = ("identity", "sql", "compute")


@pytest.fixture
def core(make_harness):
    return make_harness(toolsets=CORE)


def _me(user="someone@example.com"):
    return iam.User(user_name=user, id="42", display_name="Some One", active=True,
                    groups=[iam.ComplexValue(display="admins")], emails=[iam.ComplexValue(value=user)])


# ---------------------------------------------------------------------------------------------- registration


def test_registration_requires_dry_run_and_confirm():
    with pytest.raises(TypeError, match="dry_run"):
        @tool(toolset="identity", title="t", safety=WRITE, name="_bad_write")
        def _bad_write() -> ToolResponse:  # pragma: no cover
            return ok("x")

    with pytest.raises(TypeError, match="confirm"):
        @tool(toolset="identity", title="t", safety=DESTRUCTIVE, name="_bad_destructive")
        def _bad_destructive(dry_run: DryRun = False) -> ToolResponse:  # pragma: no cover
            return ok("x")

    with pytest.raises(TypeError, match="possible_levels"):
        @tool(toolset="identity", title="t", safety=lambda a: READ, name="_bad_callable")
        def _bad_callable() -> ToolResponse:  # pragma: no cover
            return ok("x")


async def test_all_tools_have_schemas_descriptions_and_annotations(harness):  # all toolsets
    tools = await harness.server.list_tools()
    assert len(tools) == len(harness.enabled) > 0
    for t in tools:
        assert t.description and "Safety classification:" in t.description, t.name
        assert t.output_schema, f"{t.name} has no output schema"
        assert t.input_schema.get("type") == "object", t.name
        assert t.annotations is not None and t.annotations.title, t.name
        props = t.input_schema.get("properties", {})
        spec = registry.registered_tools()[t.name]
        if not t.annotations.read_only_hint:
            assert "dry_run" in props, t.name
        if t.annotations.destructive_hint:
            assert "confirm" in props, t.name
        if isinstance(spec.safety, dict):
            assert set(props["action"].get("enum", [])) == set(spec.safety), t.name


# ---------------------------------------------------------------------------------------------- gates (identity tools)


async def test_read_tool_works_and_caches_user(core):
    core.w.current_user.me.return_value = _me()
    out = await core.call("get_current_user")
    assert out["status"] == "success"
    assert out["data"]["user_name"] == "someone@example.com"
    assert out["data"]["home_path"] == "/Users/someone@example.com"
    assert out["data"]["is_workspace_admin"] is True
    assert out["safety"] == ["READ_ONLY"]
    assert out["request_id"]


async def test_dry_run_returns_plan_without_executing(core, monkeypatch):
    called = []
    import dbx_mcp.tools.identity as identity

    monkeypatch.setattr(identity, "_profiles", lambda: called.append(1) or [{"profile": "dev"}])
    out = await core.call("manage_workspace", {"action": "switch_profile", "profile": "dev", "dry_run": True})
    assert out["status"] == "dry_run"
    assert out["plan"]["target"] == {"profile": "dev"}
    assert core.w.current_user.me.call_count == 0


async def test_read_only_mode_blocks_writes_allows_reads(make_harness):
    h = make_harness(read_only=True, toolsets=CORE)
    h.w.current_user.me.return_value = _me()
    assert (await h.call("get_current_user"))["status"] == "success"
    err = await h.call_error("manage_workspace", {"action": "switch_profile", "profile": "dev"})
    assert "BLOCKED_BY_SAFETY_POLICY" in err and "read-only" in err


async def test_blocked_levels(make_harness):
    h = make_harness(blocked_safety_levels=frozenset({SafetyLevel.EXECUTION}), toolsets=CORE)
    h.w.warehouses.list.return_value = []
    err = await h.call_error("execute_sql", {"statement": "SELECT 1"})
    assert "BLOCKED_BY_SAFETY_POLICY" in err


async def test_sdk_errors_are_normalized(core):
    from databricks.sdk.errors import PermissionDenied

    core.w.current_user.me.side_effect = PermissionDenied("User lacks access")
    err = await core.call_error("get_current_user")
    assert "[PERMISSION_DENIED]" in err and "User lacks access" in err


async def test_auth_configuration_error(tmp_path):
    from dataclasses import replace

    from dbx_mcp.databricks.client import ClientProvider
    from dbx_mcp.server.app import build_server
    from dbx_mcp.server.config import Settings

    def broken(settings, profile):
        raise ValueError("default auth: cannot configure default credentials")

    settings = replace(Settings.from_env(), manifest_path=tmp_path / "m.json", toolsets=("identity",))
    server, _ = build_server(settings, ClientProvider(settings, factory=broken))
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="AUTHENTICATION_FAILED"):
        await server.call_tool("get_current_user", {})


async def test_output_is_redacted(core):
    me = _me()
    me.display_name = "token=supersecretvalue123"
    core.w.current_user.me.return_value = me
    out = await core.call("get_current_user")
    assert "supersecretvalue123" not in str(out)


async def test_tool_timeout(make_harness, monkeypatch):
    h = make_harness(tool_timeout_seconds=1, max_wait_seconds=0, toolsets=CORE)
    h.w.current_user.me.side_effect = lambda: time.sleep(3)
    err = await h.call_error("get_current_user")
    assert "[TIMEOUT]" in err


async def test_unknown_profile_rejected(core, monkeypatch):
    import dbx_mcp.tools.identity as identity

    monkeypatch.setattr(identity, "_profiles", lambda: [{"profile": "dev"}])
    err = await core.call_error("manage_workspace", {"action": "switch_profile", "profile": "nope"})
    assert "not found" in err


async def test_workspace_info_has_no_secrets(core):
    core.w.config.token = FAKE_PAT
    out = await core.call("manage_workspace", {"action": "info"})
    assert out["data"]["host"] == "https://example.cloud.databricks.com"
    assert out["data"]["workspace_id"] == 1234567890
    assert "dapi" not in str(out)


def test_confirm_annotation_is_bool():
    assert Confirm.__origin__ is bool  # type: ignore[attr-defined]
