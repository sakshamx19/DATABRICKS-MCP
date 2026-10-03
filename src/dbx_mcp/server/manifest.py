"""Project manifest: a local JSON record of resources created through this server.

Tracking is bookkeeping only. Removing an entry from the manifest never deletes
the Databricks resource, and deleting a resource through a ``manage_*`` tool
removes its manifest entry automatically.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dbx_mcp.utils.logging import get_logger

log = get_logger("manifest")

MANIFEST_VERSION = 1


class TrackedResource(BaseModel):
    resource_type: str = Field(description="e.g. job, pipeline, dashboard, app, cluster, warehouse")
    resource_id: str
    name: str | None = None
    created_by_tool: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    workspace_host: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        # Ids are only unique within a workspace (job 42 can exist in several workspaces).
        return (self.resource_type, self.resource_id, normalize_host(self.workspace_host))


def normalize_host(host: str | None) -> str:
    return (host or "").rstrip("/").lower()


_ANY_HOST = object()


class ManifestStore:
    def __init__(self, path: Path, current_host: Callable[[], str | None] | None = None):
        self.path = path
        self._lock = threading.Lock()
        # Returns the workspace of the request being served; deletes untrack only that workspace's entry.
        self._current_host = current_host or (lambda: None)

    def _load(self) -> list[TrackedResource]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Manifest file {self.path} is unreadable: {exc}") from exc
        return [TrackedResource.model_validate(item) for item in raw.get("resources", [])]

    def _save(self, resources: list[TrackedResource]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": MANIFEST_VERSION, "resources": [r.model_dump() for r in resources]}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".manifest-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def list(self, resource_type: str | None = None) -> list[TrackedResource]:
        with self._lock:
            items = self._load()
        if resource_type:
            items = [r for r in items if r.resource_type == resource_type]
        return items

    def track(self, resource: TrackedResource) -> None:
        with self._lock:
            items = [r for r in self._load() if r.key != resource.key]
            items.append(resource)
            self._save(items)

    def untrack(
        self, resource_type: str, resource_id: str, workspace_host: str | None | object = _ANY_HOST
    ) -> TrackedResource | None:
        """Remove one entry. ``workspace_host`` restricts the match to that workspace (default: any)."""

        def matches(r: TrackedResource) -> bool:
            if (r.resource_type, r.resource_id) != (resource_type, resource_id):
                return False
            if workspace_host is _ANY_HOST or not r.workspace_host:  # entries without a host match any workspace
                return True
            return normalize_host(r.workspace_host) == normalize_host(workspace_host)  # type: ignore[arg-type]

        with self._lock:
            items = self._load()
            removed = next((r for r in items if matches(r)), None)
            if removed is not None:
                self._save([r for r in items if r.key != removed.key])
            return removed

    # Best-effort helpers used by tools: manifest problems must never fail a Databricks operation.
    def safe_track(self, **kwargs: Any) -> str | None:
        try:
            self.track(TrackedResource(**kwargs))
            return None
        except Exception as exc:
            log.warning("manifest track failed", extra={"error": str(exc)})
            return f"Resource created but could not be recorded in the project manifest: {exc}"

    def safe_untrack(self, resource_type: str, resource_id: str) -> None:
        """Called after a resource is deleted: untrack it in the current workspace only."""
        try:
            host = self._current_host()
            self.untrack(resource_type, resource_id, host if host else _ANY_HOST)
        except Exception as exc:
            log.warning("manifest untrack failed", extra={"error": str(exc)})
