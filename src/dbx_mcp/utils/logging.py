"""Structured JSON logging to stderr.

stdout is reserved for the MCP stdio transport, so ALL logging goes to stderr.
Log records are passed through secret redaction before being written.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from dbx_mcp.utils.redaction import redact, redact_text

request_id_var: ContextVar[str | None] = ContextVar("dbx_mcp_request_id", default=None)

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact_text(record.getMessage()),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(redact(payload), default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("dbx_mcp")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    # The SDK logs request/response bodies at DEBUG; keep it quiet unless asked.
    sdk_logger = logging.getLogger("databricks.sdk")
    sdk_logger.handlers[:] = [handler]
    sdk_logger.setLevel(logging.WARNING if level != "DEBUG" else logging.INFO)
    sdk_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"dbx_mcp.{name}")
