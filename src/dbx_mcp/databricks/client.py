"""Centralized Databricks client creation.

Tools never construct clients or handle credentials themselves - they ask the
:class:`ClientProvider` for the shared :class:`~databricks.sdk.WorkspaceClient`.

Authentication uses the Databricks SDK's unified auth chain, so every mechanism
the SDK supports works unchanged: PAT (``DATABRICKS_TOKEN``), OAuth M2M
(``DATABRICKS_CLIENT_ID``/``DATABRICKS_CLIENT_SECRET``), OAuth U2M via the
Databricks CLI, ``~/.databrickscfg`` profiles (``DATABRICKS_CONFIG_PROFILE``),
Azure CLI / MSI / service principals, and GCP credentials.

Retries with backoff on 429/503 and transient errors are handled by the SDK's
API client, bounded by ``retry_timeout_seconds``; ``rate_limit`` applies a
client-side requests-per-second cap.

In request auth mode (``DBX_MCP_AUTH_MODE=request``) there are no server credentials:
each request's own workspace URL + PAT (see :mod:`dbx_mcp.databricks.request_auth`) gets
its own isolated client, cached by a hash of host + token.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.credentials_provider import pat_auth

from dbx_mcp import __version__
from dbx_mcp.databricks.request_auth import RequestCredentials, request_credentials_var
from dbx_mcp.server.config import Settings
from dbx_mcp.utils.errors import DbxToolError, ErrorCategory

ClientFactory = Callable[[Settings, "str | None"], WorkspaceClient]
RequestClientFactory = Callable[[Settings, RequestCredentials], WorkspaceClient]

ENV_CREDENTIAL_KEY = "env"


def default_client_factory(settings: Settings, profile: str | None = None) -> WorkspaceClient:
    kwargs: dict[str, Any] = {
        "product": "dbx-mcp",
        "product_version": __version__,
        "http_timeout_seconds": settings.http_timeout_seconds,
        "retry_timeout_seconds": settings.retry_timeout_seconds,
    }
    if settings.rate_limit_per_second:
        kwargs["rate_limit"] = settings.rate_limit_per_second
    if profile:
        kwargs["profile"] = profile
    return WorkspaceClient(config=Config(**kwargs))


class _RequestScopedConfig(Config):
    """A Config built ONLY from explicit arguments.

    The default Config also reads DATABRICKS_* environment variables and ~/.databrickscfg,
    which in a multi-tenant server would mix the server machine's credentials into a
    user's request. This subclass disables both sources and skips the OIDC host-metadata
    probe (irrelevant for PAT auth). tests/unit/test_request_auth.py pins this behaviour.
    """

    @classmethod
    def attributes(cls):  # the base implementation only scans cls.__dict__
        return Config.attributes()

    def _load_from_env(self) -> None:
        return None

    def _known_file_config_loader(self) -> None:
        return None

    def _resolve_host_metadata(self) -> None:
        return None


def default_request_client_factory(settings: Settings, creds: RequestCredentials) -> WorkspaceClient:
    kwargs: dict[str, Any] = {
        "host": creds.host,
        "token": creds.token,
        "credentials_strategy": pat_auth,
        "product": "dbx-mcp",
        "product_version": __version__,
        "http_timeout_seconds": settings.http_timeout_seconds,
        "retry_timeout_seconds": settings.retry_timeout_seconds,
    }
    if settings.rate_limit_per_second:
        kwargs["rate_limit"] = settings.rate_limit_per_second
    return WorkspaceClient(config=_RequestScopedConfig(**kwargs))


class ClientProvider:
    """Creates and caches WorkspaceClients (thread-safe).

    * env mode: one lazily-created client from the server's own configuration.
    * request mode: one client per distinct (host, token), LRU-cached.
    """

    def __init__(
        self,
        settings: Settings,
        factory: ClientFactory | None = None,
        request_factory: RequestClientFactory | None = None,
    ):
        self._settings = settings
        self._factory = factory or default_client_factory
        self._request_factory = request_factory or default_request_client_factory
        self._client: WorkspaceClient | None = None
        self._profile: str | None = None
        self._request_clients: OrderedDict[str, WorkspaceClient] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def request_mode(self) -> bool:
        return self._settings.auth_mode == "request"

    def current_credentials(self) -> RequestCredentials | None:
        return request_credentials_var.get() if self.request_mode else None

    def credential_key(self) -> str:
        """Identifies whose credentials are in use (a hash in request mode, never the token)."""
        creds = self.current_credentials()
        return creds.key if creds else ENV_CREDENTIAL_KEY

    @property
    def profile_override(self) -> str | None:
        return self._profile

    def use_profile(self, profile: str | None) -> WorkspaceClient:
        """Switch to a ~/.databrickscfg profile (None = back to environment defaults)."""
        if self.request_mode:
            raise DbxToolError(
                ErrorCategory.UNSUPPORTED,
                "Profiles are not used in request-auth mode; each client connection sends its own workspace.",
                hint="Point your MCP client's X-Databricks-Host / Authorization headers at another workspace.",
            )
        with self._lock:
            previous_client, previous_profile = self._client, self._profile
            self._profile, self._client = profile, None
        try:
            return self.workspace()
        except Exception:
            with self._lock:
                self._client, self._profile = previous_client, previous_profile
            raise

    def workspace(self) -> WorkspaceClient:
        if self.request_mode:
            return self._request_client()
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        self._client = self._factory(self._settings, self._profile)
                    except Exception as exc:  # the SDK raises ValueError for unresolved auth
                        raise DbxToolError(
                            ErrorCategory.AUTHENTICATION,
                            f"Could not configure Databricks authentication: {exc}",
                            hint="Set DATABRICKS_HOST plus credentials (e.g. DATABRICKS_TOKEN or a "
                            "DATABRICKS_CONFIG_PROFILE). See README 'Authentication'.",
                        ) from exc
        return self._client

    def _request_client(self) -> WorkspaceClient:
        creds = request_credentials_var.get()
        if creds is None:
            raise DbxToolError(
                ErrorCategory.AUTHENTICATION,
                "No Databricks credentials were supplied with this request.",
                hint="Send 'X-Databricks-Host' and 'Authorization: Bearer <PAT>' headers from your MCP client.",
            )
        key = creds.key
        with self._lock:
            client = self._request_clients.get(key)
            if client is not None:
                self._request_clients.move_to_end(key)
                return client
        try:
            client = self._request_factory(self._settings, creds)
        except Exception as exc:
            raise DbxToolError(
                ErrorCategory.AUTHENTICATION,
                f"Could not configure a Databricks client for {creds.host}: {type(exc).__name__}",
            ) from exc
        with self._lock:
            self._request_clients[key] = client
            self._request_clients.move_to_end(key)
            while len(self._request_clients) > self._settings.request_client_cache_size:
                self._request_clients.popitem(last=False)
        return client

    def reset(self) -> None:
        with self._lock:
            self._client = None
            self._request_clients.clear()

    def describe(self) -> dict[str, Any]:
        """Non-secret description of the active connection/auth configuration."""
        cfg = self.workspace().config
        info: dict[str, Any] = {
            "host": cfg.host,
            "auth_type": cfg.auth_type,
            "profile": None if self.request_mode else cfg.profile,
            "config_file": cfg.config_file if cfg.profile and not self.request_mode else None,
            "auth_mode": self._settings.auth_mode,
            "cloud": None,
            "is_account_client": False,
        }
        for attr, key in (("is_azure", "azure"), ("is_aws", "aws"), ("is_gcp", "gcp")):
            try:
                if getattr(cfg, attr):
                    info["cloud"] = key
            except Exception:  # pragma: no cover - attribute availability varies by SDK version
                continue
        return {k: v for k, v in info.items() if v is not None}
