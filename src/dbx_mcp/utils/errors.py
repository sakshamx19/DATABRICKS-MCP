"""Error normalization: Databricks/SDK/validation errors -> categorized MCP tool errors."""

from __future__ import annotations

import traceback
from enum import Enum

from databricks.sdk import errors as dbx_errors
from mcp.server.mcpserver.exceptions import ToolError

from dbx_mcp.utils.redaction import redact_text


class ErrorCategory(str, Enum):
    AUTHENTICATION = "AUTHENTICATION_FAILED"
    AUTHORIZATION = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    CONFLICT = "CONFLICT"
    RATE_LIMIT = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    SERVICE_ERROR = "DATABRICKS_SERVICE_ERROR"
    UNSUPPORTED = "UNSUPPORTED_OPERATION"
    SAFETY_BLOCKED = "BLOCKED_BY_SAFETY_POLICY"
    CONFIGURATION = "CONFIGURATION_ERROR"
    INTERNAL = "INTERNAL_ERROR"


_HINTS: dict[ErrorCategory, str] = {
    ErrorCategory.AUTHENTICATION: "Check DATABRICKS_HOST and your credentials (token/profile/OAuth).",
    ErrorCategory.AUTHORIZATION: "The authenticated principal lacks the required privilege; ask an admin for access.",
    ErrorCategory.NOT_FOUND: "Verify the identifier/name and that you can see the resource.",
    ErrorCategory.INVALID_PARAMETER: "Fix the indicated parameter and retry.",
    ErrorCategory.CONFLICT: "The resource already exists or is in a conflicting state; inspect it first.",
    ErrorCategory.RATE_LIMIT: "Databricks rate-limited the request; retry later.",
    ErrorCategory.TIMEOUT: "The operation did not finish in time; it may still complete server-side - poll its status.",
    ErrorCategory.UNSUPPORTED: "This operation is not supported by the server/workspace.",
}


class DbxToolError(ToolError):
    """A categorized, user-safe tool error."""

    def __init__(self, category: ErrorCategory, message: str, *, hint: str | None = None, request_id: str | None = None):
        self.category = category
        self.hint = hint if hint is not None else _HINTS.get(category)
        self.request_id = request_id
        text = f"[{category.value}] {redact_text(message)}"
        if self.hint:
            text += f" Hint: {self.hint}"
        if request_id:
            text += f" (request_id={request_id})"
        super().__init__(text)


class SafetyBlockedError(DbxToolError):
    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(ErrorCategory.SAFETY_BLOCKED, message, hint=hint)


class ValidationFailed(DbxToolError):
    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(ErrorCategory.INVALID_PARAMETER, message, hint=hint)


class UnsupportedOperation(DbxToolError):
    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(ErrorCategory.UNSUPPORTED, message, hint=hint)


_SDK_MAP: list[tuple[type[BaseException], ErrorCategory]] = [
    (dbx_errors.Unauthenticated, ErrorCategory.AUTHENTICATION),
    (dbx_errors.PermissionDenied, ErrorCategory.AUTHORIZATION),
    (dbx_errors.NotFound, ErrorCategory.NOT_FOUND),
    (dbx_errors.ResourceDoesNotExist, ErrorCategory.NOT_FOUND),
    (dbx_errors.InvalidParameterValue, ErrorCategory.INVALID_PARAMETER),
    (dbx_errors.BadRequest, ErrorCategory.INVALID_PARAMETER),
    (dbx_errors.AlreadyExists, ErrorCategory.CONFLICT),
    (dbx_errors.ResourceAlreadyExists, ErrorCategory.CONFLICT),
    (dbx_errors.ResourceConflict, ErrorCategory.CONFLICT),
    (dbx_errors.InvalidState, ErrorCategory.CONFLICT),
    (dbx_errors.Aborted, ErrorCategory.CONFLICT),
    (dbx_errors.TooManyRequests, ErrorCategory.RATE_LIMIT),
    (dbx_errors.RequestLimitExceeded, ErrorCategory.RATE_LIMIT),
    (dbx_errors.ResourceExhausted, ErrorCategory.RATE_LIMIT),
    (dbx_errors.DeadlineExceeded, ErrorCategory.TIMEOUT),
    (dbx_errors.OperationTimeout, ErrorCategory.TIMEOUT),
    (dbx_errors.NotImplemented, ErrorCategory.UNSUPPORTED),
    (dbx_errors.TemporarilyUnavailable, ErrorCategory.SERVICE_ERROR),
    (dbx_errors.InternalError, ErrorCategory.SERVICE_ERROR),
]


def categorize(exc: BaseException) -> ErrorCategory:
    if isinstance(exc, DbxToolError):
        return exc.category
    if isinstance(exc, dbx_errors.PermissionDenied) and "invalid access token" in str(exc).lower():
        return ErrorCategory.AUTHENTICATION  # Databricks answers a bad/expired PAT with HTTP 403
    for exc_type, category in _SDK_MAP:
        if isinstance(exc, exc_type):
            return category
    if isinstance(exc, dbx_errors.DatabricksError):
        message = str(exc).lower()
        if "credential" in message or "authenticat" in message or "default auth" in message:
            return ErrorCategory.AUTHENTICATION
        return ErrorCategory.SERVICE_ERROR
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, ValueError | TypeError | KeyError):
        return ErrorCategory.INVALID_PARAMETER
    return ErrorCategory.INTERNAL


def _databricks_request_id(exc: BaseException) -> str | None:
    for detail in getattr(exc, "details", None) or []:
        request_id = getattr(detail, "request_id", None) or (
            detail.get("request_id") if isinstance(detail, dict) else None
        )
        if request_id:
            return str(request_id)
    return None


def normalize_exception(exc: BaseException, *, debug: bool = False) -> DbxToolError:
    """Convert any exception into a :class:`DbxToolError` safe to show the model."""
    if isinstance(exc, DbxToolError):
        return exc
    if isinstance(exc, ToolError):
        return DbxToolError(ErrorCategory.INVALID_PARAMETER, str(exc), hint="")
    category = categorize(exc)
    if isinstance(exc, dbx_errors.DatabricksError):
        code = getattr(exc, "error_code", None)
        message = f"{code}: {exc}" if code else str(exc)
    elif category == ErrorCategory.INTERNAL and not debug:
        message = f"Unexpected {type(exc).__name__}. Enable DBX_MCP_DEBUG=true for details."
    else:
        message = f"{type(exc).__name__}: {exc}"
    if debug:
        message += "\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return DbxToolError(category, message, request_id=_databricks_request_id(exc))
