"""Process-wide server context shared by all tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from databricks.sdk import WorkspaceClient

from dbx_mcp.databricks.client import ClientProvider
from dbx_mcp.safety.guard import SafetyPolicy
from dbx_mcp.server.config import Settings
from dbx_mcp.server.manifest import ManifestStore


@dataclass
class AppContext:
    settings: Settings
    clients: ClientProvider
    safety: SafetyPolicy
    manifest: ManifestStore
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def w(self) -> WorkspaceClient:
        return self.clients.workspace()

    @property
    def host(self) -> str | None:
        creds = self.clients.current_credentials()
        if creds is not None:
            return creds.host
        try:
            return self.w.config.host
        except Exception:
            return None

    @property
    def default_warehouse_id(self) -> str | None:
        """Per-connection header in request mode; otherwise the server-wide setting.

        A server-wide default is ignored in request mode: it belongs to one workspace and
        would be wrong for every other workspace the server serves.
        """
        if self.clients.request_mode:
            creds = self.clients.current_credentials()
            return creds.warehouse_id if creds else None
        return self.settings.default_warehouse_id

    @property
    def default_cluster_id(self) -> str | None:
        if self.clients.request_mode:
            creds = self.clients.current_credentials()
            return creds.cluster_id if creds else None
        return self.settings.default_cluster_id

    # The current user's name is cached per credential, so one tenant never sees another's.
    def cached_user_name(self) -> str | None:
        return self.cache.get(f"user:{self.clients.credential_key()}")

    def remember_user_name(self, user_name: str | None) -> None:
        if user_name:
            self.cache[f"user:{self.clients.credential_key()}"] = user_name

    def forget_user_name(self) -> None:
        self.cache.pop(f"user:{self.clients.credential_key()}", None)

    @classmethod
    def create(cls, settings: Settings, clients: ClientProvider | None = None) -> AppContext:
        app = cls(
            settings=settings,
            clients=clients or ClientProvider(settings),
            safety=SafetyPolicy(settings),
            manifest=ManifestStore(settings.manifest_path),
        )
        app.manifest = ManifestStore(settings.manifest_path, current_host=lambda: app.host)
        return app


_CONTEXT: AppContext | None = None


def set_context(ctx: AppContext | None) -> None:
    global _CONTEXT
    _CONTEXT = ctx


def get_context() -> AppContext:
    if _CONTEXT is None:
        raise RuntimeError("Server context is not initialised; call set_context() first.")
    return _CONTEXT
