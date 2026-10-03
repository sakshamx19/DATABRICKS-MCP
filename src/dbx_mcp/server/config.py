"""Server configuration, loaded from environment variables.

Databricks *authentication* is deliberately NOT configured here: it is delegated
entirely to the Databricks SDK's unified authentication (DATABRICKS_HOST,
DATABRICKS_TOKEN, DATABRICKS_CONFIG_PROFILE, OAuth M2M/U2M, Azure, GCP, ...).
This module only holds MCP-server behaviour settings, all prefixed ``DBX_MCP_``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dbx_mcp.safety.levels import SafetyLevel

ENV_PREFIX = "DBX_MCP_"

ALL_TOOLSETS: tuple[str, ...] = (
    "identity",
    "sql",
    "compute",
    "workspace",
    "jobs",
    "pipelines",
    "unity_catalog",
    "volumes",
    "dashboards",
    "ai",
    "vector_search",
    "lakebase",
    "apps",
    "manifest",
    "pdf",
)


# Host suffixes accepted for X-Databricks-Host in request auth mode (Databricks workspace domains).
# Restricting hosts stops the server being used to send requests to arbitrary/internal URLs (SSRF).
DEFAULT_ALLOWED_WORKSPACE_HOSTS: tuple[str, ...] = (
    ".cloud.databricks.com",
    ".azuredatabricks.net",
    ".gcp.databricks.com",
    ".databricks.azure.us",
    ".databricks.azure.cn",
    ".cloud.databricks.us",
)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{ENV_PREFIX}{name} must be a boolean (true/false), got {value!r}")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{ENV_PREFIX}{name} must be >= {minimum}, got {parsed}")
    return parsed


def _env_list(name: str) -> list[str]:
    value = _env(name)
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- capability selection -------------------------------------------------
    toolsets: tuple[str, ...] = ALL_TOOLSETS
    disabled_tools: frozenset[str] = frozenset()

    # --- safety -----------------------------------------------------------------
    read_only: bool = False
    blocked_safety_levels: frozenset[SafetyLevel] = frozenset()
    require_confirmation: bool = True
    confirm_execution: bool = False
    protected_name_patterns: tuple[re.Pattern[str], ...] = ()
    allow_protected_changes: bool = False
    allowed_volume_prefixes: tuple[str, ...] = ()
    allowed_workspace_prefixes: tuple[str, ...] = ()
    local_file_root: Path | None = None

    # --- defaults ---------------------------------------------------------------
    default_warehouse_id: str | None = None
    warehouse_selection: str = "prefer_running"  # "prefer_running" | "configured_only"
    default_cluster_id: str | None = None
    sql_max_rows: int = 1000
    sql_wait_timeout_seconds: int = 30
    max_page_size: int = 100
    default_page_size: int = 50
    max_inline_download_bytes: int = 10 * 1024 * 1024
    max_wait_seconds: int = 240

    # --- authentication ---------------------------------------------------------
    # "env": the server's own Databricks credentials (SDK unified auth) - one workspace.
    # "request": every HTTP request brings its own workspace URL + PAT in headers; the server
    #            stores no credentials and can serve any number of workspaces/users.
    auth_mode: str = "env"
    allowed_workspace_hosts: tuple[str, ...] = DEFAULT_ALLOWED_WORKSPACE_HOSTS
    request_client_cache_size: int = 64

    # --- runtime ----------------------------------------------------------------
    tool_timeout_seconds: int = 300
    http_timeout_seconds: int = 60
    retry_timeout_seconds: int = 300
    rate_limit_per_second: int | None = None
    manifest_path: Path = field(default_factory=lambda: Path(".databricks_mcp") / "manifest.json")
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        toolsets_raw = _env_list("TOOLSETS")
        if not toolsets_raw or toolsets_raw == ["all"]:
            toolsets: tuple[str, ...] = ALL_TOOLSETS
        else:
            unknown = sorted(set(toolsets_raw) - set(ALL_TOOLSETS))
            if unknown:
                raise ValueError(
                    f"{ENV_PREFIX}TOOLSETS contains unknown toolsets {unknown}. Valid: {', '.join(ALL_TOOLSETS)}"
                )
            toolsets = tuple(t for t in ALL_TOOLSETS if t in toolsets_raw)

        blocked: set[SafetyLevel] = set()
        for raw in _env_list("BLOCKED_SAFETY_LEVELS"):
            try:
                blocked.add(SafetyLevel(raw.upper()))
            except ValueError as exc:
                valid = ", ".join(level.value for level in SafetyLevel)
                raise ValueError(f"Unknown safety level {raw!r} in {ENV_PREFIX}BLOCKED_SAFETY_LEVELS. Valid: {valid}") from exc
        if SafetyLevel.READ_ONLY in blocked:
            raise ValueError("READ_ONLY cannot be blocked; disable toolsets instead.")

        patterns_raw = _env("PROTECTED_NAME_PATTERNS", r"(?i)(^|[-_ .])prod(uction)?($|[-_ .])")
        patterns: list[re.Pattern[str]] = []
        if patterns_raw and patterns_raw.lower() != "none":
            for raw in patterns_raw.split(","):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    patterns.append(re.compile(raw))
                except re.error as exc:
                    raise ValueError(f"Invalid regex {raw!r} in {ENV_PREFIX}PROTECTED_NAME_PATTERNS: {exc}") from exc

        local_root = _env("LOCAL_FILE_ROOT")
        selection = (_env("WAREHOUSE_SELECTION", "prefer_running") or "prefer_running").lower()
        if selection not in {"prefer_running", "configured_only"}:
            raise ValueError(f"{ENV_PREFIX}WAREHOUSE_SELECTION must be 'prefer_running' or 'configured_only'")

        auth_mode = (_env("AUTH_MODE", "env") or "env").lower()
        if auth_mode not in {"env", "request"}:
            raise ValueError(f"{ENV_PREFIX}AUTH_MODE must be 'env' or 'request', got {auth_mode!r}")
        hosts = _env_list("ALLOWED_WORKSPACE_HOSTS")
        allowed_hosts = tuple(h.lower() for h in hosts) if hosts else DEFAULT_ALLOWED_WORKSPACE_HOSTS

        rate_limit = _env("RATE_LIMIT_PER_SECOND")
        max_page = _env_int("MAX_PAGE_SIZE", 100, minimum=1)
        tool_timeout = _env_int("TOOL_TIMEOUT_SECONDS", 300, minimum=10)
        max_wait = _env_int("MAX_WAIT_SECONDS", min(240, tool_timeout - 30), minimum=5)
        if max_wait > tool_timeout - 10:
            raise ValueError(
                f"{ENV_PREFIX}MAX_WAIT_SECONDS ({max_wait}) must be at least 10s below "
                f"{ENV_PREFIX}TOOL_TIMEOUT_SECONDS ({tool_timeout})"
            )

        return cls(
            toolsets=toolsets,
            disabled_tools=frozenset(_env_list("DISABLED_TOOLS")),
            read_only=_env_bool("READ_ONLY", False),
            blocked_safety_levels=frozenset(blocked),
            require_confirmation=_env_bool("REQUIRE_CONFIRMATION", True),
            confirm_execution=_env_bool("CONFIRM_EXECUTION", False),
            protected_name_patterns=tuple(patterns),
            allow_protected_changes=_env_bool("ALLOW_PROTECTED_CHANGES", False),
            allowed_volume_prefixes=tuple(p.rstrip("/") for p in _env_list("ALLOWED_VOLUME_PREFIXES")),
            allowed_workspace_prefixes=tuple(p.rstrip("/") for p in _env_list("ALLOWED_WORKSPACE_PREFIXES")),
            local_file_root=Path(local_root).resolve() if local_root else None,
            default_warehouse_id=_env("DEFAULT_WAREHOUSE_ID") or os.environ.get("DATABRICKS_WAREHOUSE_ID") or None,
            warehouse_selection=selection,
            default_cluster_id=_env("DEFAULT_CLUSTER_ID") or os.environ.get("DATABRICKS_CLUSTER_ID") or None,
            sql_max_rows=_env_int("SQL_MAX_ROWS", 1000, minimum=1),
            sql_wait_timeout_seconds=min(max(_env_int("SQL_WAIT_TIMEOUT_SECONDS", 30, minimum=5), 5), 50),
            max_page_size=max_page,
            default_page_size=min(_env_int("DEFAULT_PAGE_SIZE", 50, minimum=1), max_page),
            max_inline_download_bytes=_env_int("MAX_INLINE_DOWNLOAD_BYTES", 10 * 1024 * 1024, minimum=1024),
            max_wait_seconds=max_wait,
            auth_mode=auth_mode,
            allowed_workspace_hosts=allowed_hosts,
            request_client_cache_size=_env_int("REQUEST_CLIENT_CACHE_SIZE", 64, minimum=1),
            tool_timeout_seconds=tool_timeout,
            http_timeout_seconds=_env_int("HTTP_TIMEOUT_SECONDS", 60, minimum=5),
            retry_timeout_seconds=_env_int("RETRY_TIMEOUT_SECONDS", 300, minimum=0),
            rate_limit_per_second=int(rate_limit) if rate_limit else None,
            manifest_path=Path(_env("MANIFEST_PATH", str(Path(".databricks_mcp") / "manifest.json")) or ""),
            debug=_env_bool("DEBUG", False),
            log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
        )
