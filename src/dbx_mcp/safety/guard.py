"""Safety policy enforcement.

Three layers, applied in order by the tool wrapper (see :mod:`dbx_mcp.tools.registry`):

1. **Policy** - is this safety level allowed at all? (read-only mode, blocked levels)
2. **Dry run** - for any non-read action, ``dry_run=true`` returns the plan and stops.
3. **Confirmation** - DESTRUCTIVE / SECURITY_SENSITIVE changes require ``confirm=true``.
   Without it the tool returns ``status="confirmation_required"`` with a plan
   describing exactly what would change, and does nothing.

Tools additionally call :meth:`SafetyPolicy.check_protected` before changing a
resource whose name/tags mark it as production.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from dbx_mcp.safety.levels import SafetyLevel, is_read_action, needs_confirmation
from dbx_mcp.server.config import Settings
from dbx_mcp.utils.errors import SafetyBlockedError


class SafetyPolicy:
    def __init__(self, settings: Settings):
        self.settings = settings

    # -- layer 1 ---------------------------------------------------------------------------
    def check_allowed(self, tool: str, action: str | None, levels: frozenset[SafetyLevel]) -> None:
        label = f"{tool}" + (f".{action}" if action else "")
        if self.settings.read_only and not is_read_action(levels):
            raise SafetyBlockedError(
                f"{label} is classified {sorted(level.value for level in levels)} but the server runs in read-only mode.",
                hint="Unset DBX_MCP_READ_ONLY to allow changes.",
            )
        blocked = levels & self.settings.blocked_safety_levels
        if blocked:
            raise SafetyBlockedError(
                f"{label} is classified {sorted(level.value for level in blocked)}, which is blocked by server policy.",
                hint="Adjust DBX_MCP_BLOCKED_SAFETY_LEVELS to permit it.",
            )

    # -- layer 3 ---------------------------------------------------------------------------
    def requires_confirmation(self, levels: frozenset[SafetyLevel]) -> bool:
        return self.settings.require_confirmation and needs_confirmation(
            levels, confirm_execution=self.settings.confirm_execution
        )

    # -- protected resources --------------------------------------------------------------
    def protected_reason(self, name: str | None, tags: Mapping[str, str] | None = None) -> str | None:
        candidates: list[tuple[str, str]] = []
        if name:
            candidates.append(("name", name))
        for key, value in (tags or {}).items():
            candidates.append((f"tag {key}", f"{key}={value}"))
            candidates.append((f"tag {key}", str(value)))
        for pattern in self.settings.protected_name_patterns:
            for label, text in candidates:
                if pattern.search(text):
                    return f"{label} {text!r} matches protected pattern {pattern.pattern!r}"
        return None

    def check_protected(
        self,
        resource: str,
        name: str | None,
        tags: Mapping[str, str] | None = None,
        *,
        operation: str,
    ) -> None:
        """Refuse destructive/disruptive changes to production-marked resources."""
        if self.settings.allow_protected_changes:
            return
        reason = self.protected_reason(name, tags)
        if reason:
            raise SafetyBlockedError(
                f"Refusing to {operation} {resource}: it looks like a protected/production resource ({reason}).",
                hint="Set DBX_MCP_ALLOW_PROTECTED_CHANGES=true (or adjust DBX_MCP_PROTECTED_NAME_PATTERNS) if this is intended.",
            )

    # -- path allowlists -------------------------------------------------------------------
    @staticmethod
    def _within(path: str, prefixes: Iterable[str]) -> bool:
        return any(path == p or path.startswith(p + "/") for p in prefixes)

    def check_volume_path(self, path: str) -> None:
        prefixes = self.settings.allowed_volume_prefixes
        if prefixes and not self._within(path, prefixes):
            raise SafetyBlockedError(
                f"Path {path!r} is outside the allowed volume prefixes {list(prefixes)}.",
                hint="Adjust DBX_MCP_ALLOWED_VOLUME_PREFIXES.",
            )

    def check_workspace_path(self, path: str) -> None:
        prefixes = self.settings.allowed_workspace_prefixes
        if prefixes and not self._within(path, prefixes):
            raise SafetyBlockedError(
                f"Path {path!r} is outside the allowed workspace prefixes {list(prefixes)}.",
                hint="Adjust DBX_MCP_ALLOWED_WORKSPACE_PREFIXES.",
            )
