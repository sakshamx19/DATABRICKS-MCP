"""Shared fixtures: a fully-wired MCP server backed by a mocked WorkspaceClient.

Usage in tests::

    async def test_something(harness):
        harness.w.clusters.get.return_value = ClusterDetails(cluster_id="c1", cluster_name="dev")
        result = await harness.call("manage_cluster", {"action": "get", "cluster_id": "c1"})
        assert result["status"] == "success"

``harness.call`` returns the structured ToolResponse dict; use ``harness.call_error``
to assert a tool error and get its message.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from dbx_mcp.databricks.client import ClientProvider
from dbx_mcp.server.app import build_server
from dbx_mcp.server.config import Settings
from dbx_mcp.server.context import set_context


def _service_classes() -> dict[str, type]:
    """Map each WorkspaceClient service property (e.g. 'clusters') to its API class."""
    import inspect
    import typing

    from databricks.sdk import WorkspaceClient

    out: dict[str, type] = {}
    for name, prop in inspect.getmembers(WorkspaceClient, lambda v: isinstance(v, property)):
        try:
            cls = typing.get_type_hints(prop.fget).get("return")
        except Exception:
            continue
        if inspect.isclass(cls) and cls.__module__.startswith("databricks.sdk"):
            out[name] = cls
    return out


_SERVICES = _service_classes()


def autospec_workspace_client() -> MagicMock:
    """A WorkspaceClient mock whose services are autospecced from the real SDK classes.

    Calling a service method with a wrong name or wrong keyword argument raises, so unit
    tests also verify that tools use the real SDK API (no hallucinated methods/params).
    """
    return _LazyWorkspaceMock(name="WorkspaceClient")


class _LazyWorkspaceMock(MagicMock):
    """Autospecs a service on first access (autospeccing all ~140 services up front is slow)."""

    def __getattr__(self, name: str) -> Any:
        if name in _SERVICES:
            service = create_autospec(_SERVICES[name], instance=True)
            setattr(self, name, service)
            return service
        return super().__getattr__(name)

    def _get_child_mock(self, **kwargs: Any) -> MagicMock:
        return MagicMock(**kwargs)


class Harness:
    def __init__(self, tmp_path, **overrides: Any):
        self.w = autospec_workspace_client()
        self.w.config.host = "https://example.cloud.databricks.com"
        self.w.config.auth_type = "pat"
        self.w.config.profile = None
        self.w.get_workspace_id.return_value = 1234567890
        base = Settings.from_env()
        self.settings = replace(base, manifest_path=tmp_path / "manifest.json", **overrides)
        self.server, self.enabled = build_server(self.settings, ClientProvider(self.settings, factory=lambda s, p: self.w))

    async def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        result = await self.server.call_tool(name, args or {})
        assert not result.is_error, result.content
        return result.structured_content

    async def call_error(self, name: str, args: dict[str, Any] | None = None) -> str:
        with pytest.raises(ToolError) as info:
            result = await self.server.call_tool(name, args or {})
            if result.is_error:  # some MCP versions return instead of raising
                raise ToolError(result.content[0].text)
        return str(info.value)

    def tool_names(self) -> set[str]:
        return {spec.name for spec in self.enabled}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("DBX_MCP_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_CLUSTER_ID", raising=False)
    yield
    set_context(None)


@pytest.fixture
def harness(tmp_path) -> Harness:
    return Harness(tmp_path)


@pytest.fixture
def make_harness(tmp_path):
    """Build a harness with Settings overrides, e.g. make_harness(read_only=True)."""

    def factory(**overrides: Any) -> Harness:
        return Harness(tmp_path, **overrides)

    return factory
