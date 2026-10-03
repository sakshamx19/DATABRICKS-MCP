"""Identity & workspace context: get_current_user, manage_workspace."""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from dbx_mcp.models.common import ToolResponse
from dbx_mcp.safety.levels import READ, WRITE
from dbx_mcp.tools.common import DryRun, ctx, ok, require
from dbx_mcp.tools.registry import PlanInfo, tool
from dbx_mcp.utils.errors import UnsupportedOperation, ValidationFailed


def _home_path(user_name: str | None) -> str | None:
    return f"/Users/{user_name}" if user_name else None


@tool(toolset="identity", title="Get current user", safety=READ)
def get_current_user() -> ToolResponse:
    """Return the Databricks identity this server is authenticated as: username, user id,
    display name, group memberships, home folder and workspace. Never returns credentials."""
    c = ctx()
    me = c.w.current_user.me()
    c.remember_user_name(me.user_name)
    groups = [g.display for g in (me.groups or []) if g.display]
    try:
        workspace_id = c.w.get_workspace_id()
    except Exception:
        workspace_id = None
    data = {
        "user_name": me.user_name,
        "user_id": me.id,
        "display_name": me.display_name,
        "active": me.active,
        "emails": [e.value for e in (me.emails or []) if e.value],
        "groups": groups,
        "is_workspace_admin": "admins" in groups,
        "home_path": _home_path(me.user_name),
        "workspace": {"host": c.host, "workspace_id": workspace_id},
    }
    return ok(f"Authenticated as {me.user_name} ({me.display_name or 'no display name'}).", data)


def _profiles() -> list[dict[str, str | None]]:
    path = Path(os.environ.get("DATABRICKS_CONFIG_FILE", "~/.databrickscfg")).expanduser()
    if not path.exists():
        return []
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    out = []
    for section in ["DEFAULT", *parser.sections()]:
        if section == "DEFAULT" and not parser.defaults():
            continue
        items = parser[section]
        # Only non-secret keys are exposed.
        out.append(
            {
                "profile": section,
                "host": items.get("host"),
                "auth_type": items.get("auth_type"),
                "account_id": items.get("account_id"),
            }
        )
    return out


def _workspace_preview(args: dict) -> PlanInfo | None:
    if args.get("action") == "switch_profile":
        return PlanInfo(
            description=f"Switch this server's Databricks connection to profile {args.get('profile')!r} "
            "from the Databricks config file. All subsequent tool calls will use that workspace/identity.",
            target={"profile": args.get("profile")},
            reversible=True,
        )
    return None


@tool(
    toolset="identity",
    title="Workspace context",
    safety={"info": READ, "list_profiles": READ, "switch_profile": WRITE},
    preview=_workspace_preview,
)
def manage_workspace(
    action: Annotated[
        Literal["info", "list_profiles", "switch_profile"],
        Field(description="info: current workspace/auth context; list_profiles: profiles in ~/.databrickscfg "
              "(names and hosts only); switch_profile: reconnect using another profile."),
    ] = "info",
    profile: Annotated[str | None, Field(description="Profile name for switch_profile (use 'env' to revert "
                                         "to environment-variable configuration).")] = None,
    dry_run: DryRun = False,
) -> ToolResponse:
    """Identify or change the Databricks workspace this server talks to: workspace URL,
    workspace id, active profile and auth type (never tokens), and available config profiles."""
    c = ctx()
    if action in {"list_profiles", "switch_profile"} and c.clients.request_mode:
        # Profiles live on the server machine; exposing or switching them would leak/mix tenants.
        raise UnsupportedOperation(
            f"'{action}' is not available in request-auth mode: each client connection sends its own "
            "workspace URL and token.",
            hint="Use a different X-Databricks-Host / Authorization header pair to reach another workspace.",
        )
    if action == "list_profiles":
        profiles = _profiles()
        return ok(f"Found {len(profiles)} profile(s) in the Databricks config file.", profiles)

    if action == "switch_profile":
        require(profile, "profile", action)
        target = None if profile == "env" else profile
        if target and target not in {p["profile"] for p in _profiles()}:
            raise ValidationFailed(f"Profile {profile!r} not found in the Databricks config file")
        c.clients.use_profile(target)
        c.forget_user_name()
        return _info(c, f"Switched to {'environment configuration' if target is None else f'profile {target!r}'}.")

    return _info(c, None)


def _info(c, prefix: str | None) -> ToolResponse:
    auth = c.clients.describe()
    try:
        workspace_id = c.w.get_workspace_id()
    except Exception:
        workspace_id = None
    data = {
        **auth,
        "workspace_id": workspace_id,
        "active_profile": c.clients.profile_override or auth.get("profile") or os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        "server": {
            "read_only": c.settings.read_only,
            "require_confirmation": c.settings.require_confirmation,
            "blocked_safety_levels": sorted(level.value for level in c.settings.blocked_safety_levels),
            "toolsets": list(c.settings.toolsets),
            "default_warehouse_id": c.default_warehouse_id,
            "default_cluster_id": c.default_cluster_id,
        },
    }
    summary = f"Connected to {auth.get('host')} (workspace_id={workspace_id}, auth={auth.get('auth_type')})."
    return ok(f"{prefix} {summary}" if prefix else summary, data)
