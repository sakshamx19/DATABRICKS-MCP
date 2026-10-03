"""Per-request Databricks credentials (``DBX_MCP_AUTH_MODE=request``).

In request mode the server stores no Databricks credentials. Every MCP request made over an
HTTP transport carries its own workspace and token in headers, typically set once in the MCP
client's JSON config::

    X-Databricks-Host: https://adb-123.4.azuredatabricks.net
    Authorization: Bearer dapi...            (or X-Databricks-Token: dapi...)
    X-Databricks-Warehouse-Id: abc123        (optional per-connection default)
    X-Databricks-Cluster-Id: 0101-...        (optional per-connection default)

One server can therefore serve many workspaces and many users at once. The credentials of
the request being handled live in a context variable, which propagates into the worker thread
that runs the tool, so tools stay unaware of where their credentials came from.

Security notes:
* The host must be ``https`` and match an allowed Databricks domain suffix
  (``DBX_MCP_ALLOWED_WORKSPACE_HOSTS``). This stops the server being used to send requests to
  arbitrary or internal URLs.
* Tokens are never logged or returned; caches are keyed by a SHA-256 digest of host + token.
* Headers are only as private as the transport: serve over HTTPS (e.g. behind a TLS-terminating
  reverse proxy) whenever clients are not on the same machine.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from dbx_mcp.utils.errors import DbxToolError, ErrorCategory

HOST_HEADER = "x-databricks-host"
TOKEN_HEADER = "x-databricks-token"
WAREHOUSE_HEADER = "x-databricks-warehouse-id"
CLUSTER_HEADER = "x-databricks-cluster-id"

_HOSTNAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")
_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_MAX_TOKEN_LENGTH = 4096

_AUTH_HINT = (
    "This server runs in request-auth mode: configure your MCP client to send the headers "
    "'X-Databricks-Host: https://<workspace-url>' and 'Authorization: Bearer <PAT>'."
)


@dataclass(frozen=True)
class RequestCredentials:
    host: str
    token: str = field(repr=False)
    warehouse_id: str | None = None
    cluster_id: str | None = None

    @property
    def key(self) -> str:
        """Stable, non-reversible identifier for caching (never contains the token)."""
        return hashlib.sha256(f"{self.host}\n{self.token}".encode()).hexdigest()

    @property
    def short_key(self) -> str:
        return self.key[:12]


request_credentials_var: ContextVar[RequestCredentials | None] = ContextVar(
    "dbx_mcp_request_credentials", default=None
)


def _auth_error(message: str) -> DbxToolError:
    return DbxToolError(ErrorCategory.AUTHENTICATION, message, hint=_AUTH_HINT)


def normalize_host(raw: str, allowed_suffixes: tuple[str, ...]) -> str:
    """Validate a workspace URL from a header and return ``https://<hostname>``."""
    value = raw.strip()
    if "://" not in value:
        value = "https://" + value
    parts = urlsplit(value)
    if parts.scheme.lower() != "https":
        raise _auth_error("X-Databricks-Host must be an https:// URL.")
    if parts.username or parts.password or parts.port not in (None, 443):
        raise _auth_error("X-Databricks-Host must not contain credentials or a custom port.")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise _auth_error("X-Databricks-Host must be the workspace root URL (no path or query).")
    hostname = (parts.hostname or "").lower().rstrip(".")
    if not _HOSTNAME.match(hostname):
        raise _auth_error(f"X-Databricks-Host has an invalid hostname {hostname!r}.")
    if "*" not in allowed_suffixes and not any(
        hostname.endswith(suffix) if suffix.startswith(".") else hostname == suffix for suffix in allowed_suffixes
    ):
        raise DbxToolError(
            ErrorCategory.SAFETY_BLOCKED,
            f"Workspace host {hostname!r} is not an allowed Databricks domain.",
            hint="Ask the server operator to add it to DBX_MCP_ALLOWED_WORKSPACE_HOSTS.",
        )
    return f"https://{hostname}"


def credentials_from_headers(
    headers: Mapping[str, str] | None, allowed_suffixes: tuple[str, ...]
) -> RequestCredentials:
    """Extract and validate per-request credentials from HTTP headers."""
    if headers is None:
        raise DbxToolError(
            ErrorCategory.CONFIGURATION,
            "Request-auth mode needs an HTTP transport (streamable-http or sse); this request has no headers.",
            hint="Start the server with --transport streamable-http, or use DBX_MCP_AUTH_MODE=env for stdio.",
        )
    lowered = {k.lower(): v for k, v in headers.items()}
    raw_host = (lowered.get(HOST_HEADER) or "").strip()
    token = (lowered.get(TOKEN_HEADER) or "").strip()
    authorization = (lowered.get("authorization") or "").strip()
    if not token and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() != "bearer" or not value.strip():
            raise _auth_error("The Authorization header must be 'Bearer <PAT>'.")
        token = value.strip()
    if not raw_host:
        raise _auth_error("Missing the X-Databricks-Host header.")
    if not token:
        raise _auth_error("Missing the Databricks token (Authorization: Bearer <PAT> or X-Databricks-Token).")
    if len(token) > _MAX_TOKEN_LENGTH or any(ch.isspace() for ch in token):
        raise _auth_error("The Databricks token is malformed.")

    def optional_id(header: str) -> str | None:
        value = (lowered.get(header) or "").strip()
        if not value:
            return None
        if not _ID.match(value):
            raise DbxToolError(ErrorCategory.INVALID_PARAMETER, f"Header {header} has an invalid value.")
        return value

    return RequestCredentials(
        host=normalize_host(raw_host, allowed_suffixes),
        token=token,
        warehouse_id=optional_id(WAREHOUSE_HEADER),
        cluster_id=optional_id(CLUSTER_HEADER),
    )
