"""Input validation: paths, identifiers, and SQL statement classification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from dbx_mcp.safety.levels import SafetyLevel
from dbx_mcp.utils.errors import SafetyBlockedError, ValidationFailed

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


# ----------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------

def _check_segments(path: str, what: str) -> list[str]:
    if _CONTROL_CHARS.search(path):
        raise ValidationFailed(f"{what} contains control characters")
    if "\\" in path:
        raise ValidationFailed(f"{what} must use forward slashes")
    if path == "/":
        return []
    segments = path.split("/")[1:]
    if any(seg in ("", ".", "..") for seg in segments):
        raise ValidationFailed(f"{what} must not contain empty, '.' or '..' segments: {path!r}")
    return segments


@dataclass(frozen=True)
class VolumePath:
    catalog: str
    schema: str
    volume: str
    relative: str  # path inside the volume, "" for the volume root

    @property
    def full(self) -> str:
        base = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"
        return f"{base}/{self.relative}" if self.relative else base

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    @property
    def volume_full_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.volume}"


def parse_volume_path(path: str, *, require_file: bool = False) -> VolumePath:
    """Validate a Unity Catalog Volume path: ``/Volumes/<catalog>/<schema>/<volume>[/...]``.

    Rejects traversal (``..``), relative paths, backslashes, control characters,
    and anything outside ``/Volumes``. ``dbfs:/Volumes/...`` is accepted and normalized.
    """
    if not path or not isinstance(path, str):
        raise ValidationFailed("A volume path is required")
    raw = path.strip()
    if raw.startswith("dbfs:"):
        raw = raw[len("dbfs:"):]
    if len(raw) > 1 and raw.endswith("/"):
        raw = raw.rstrip("/")
    if not raw.startswith("/Volumes/"):
        raise ValidationFailed(
            f"Volume paths must start with /Volumes/<catalog>/<schema>/<volume>, got {path!r}"
        )
    segments = _check_segments(raw, "Volume path")
    if len(segments) < 4 or any(not s for s in segments):
        raise ValidationFailed(f"Volume path must include catalog, schema and volume: {path!r}")
    _, catalog, schema, volume, *rest = segments
    if require_file and not rest:
        raise ValidationFailed(f"Expected a file path inside the volume, got the volume root {path!r}")
    return VolumePath(catalog, schema, volume, "/".join(rest))


def validate_workspace_path(path: str) -> str:
    """Validate an absolute workspace path (``/Users/...``, ``/Shared/...``, ``/Workspace/...``)."""
    if not path or not isinstance(path, str):
        raise ValidationFailed("A workspace path is required")
    raw = path.strip()
    if len(raw) > 1:
        raw = raw.rstrip("/")
    if not raw.startswith("/"):
        raise ValidationFailed(f"Workspace paths must be absolute (start with '/'), got {path!r}")
    if raw.startswith("/Volumes") or raw.startswith("/dbfs"):
        raise ValidationFailed("Use the volume tools for /Volumes paths; this tool manages workspace objects")
    _check_segments(raw, "Workspace path")
    return raw


def resolve_local_path(root: Path | None, relative: str, *, must_exist: bool) -> Path:
    """Resolve a path inside the configured local file root, refusing escapes."""
    if root is None:
        raise SafetyBlockedError(
            "Local filesystem access is disabled.",
            hint="Set DBX_MCP_LOCAL_FILE_ROOT to a directory to enable local upload/download, or pass content inline.",
        )
    if _CONTROL_CHARS.search(relative):
        raise ValidationFailed("Local path contains control characters")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise SafetyBlockedError(f"Local path {relative!r} escapes DBX_MCP_LOCAL_FILE_ROOT")
    if must_exist and not candidate.is_file():
        raise ValidationFailed(f"Local file not found: {relative!r}")
    return candidate


# ----------------------------------------------------------------------------------------------
# Identifiers
# ----------------------------------------------------------------------------------------------

def split_full_name(full_name: str, *, parts: int | tuple[int, ...]) -> list[str]:
    """Split ``catalog.schema.object`` honoring backtick quoting; validate part count."""
    if not full_name or not isinstance(full_name, str):
        raise ValidationFailed("A name is required")
    if _CONTROL_CHARS.search(full_name):
        raise ValidationFailed("Name contains control characters")
    result: list[str] = []
    buf: list[str] = []
    quoted = False
    i = 0
    while i < len(full_name):
        ch = full_name[i]
        if ch == "`":
            if quoted and i + 1 < len(full_name) and full_name[i + 1] == "`":
                buf.append("`")
                i += 2
                continue
            quoted = not quoted
        elif ch == "." and not quoted:
            result.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if quoted:
        raise ValidationFailed(f"Unbalanced backticks in {full_name!r}")
    result.append("".join(buf))
    allowed = (parts,) if isinstance(parts, int) else parts
    if len(result) not in allowed or any(not p.strip() for p in result):
        expected = " or ".join(str(p) for p in allowed)
        raise ValidationFailed(f"Expected a {expected}-part name (e.g. catalog.schema.object), got {full_name!r}")
    return result


def quote_ident(part: str) -> str:
    return "`" + part.replace("`", "``") + "`"


def quote_full_name(full_name: str, *, parts: int | tuple[int, ...] = (1, 2, 3)) -> str:
    return ".".join(quote_ident(p) for p in split_full_name(full_name, parts=parts))


def quote_string_literal(value: str) -> str:
    """Quote a SQL string literal (single quotes, backslash-escaped as Databricks SQL expects)."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


# ----------------------------------------------------------------------------------------------
# SQL classification
# ----------------------------------------------------------------------------------------------

_READ_KEYWORDS = {"SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "LIST", "VALUES", "TABLE", "FROM", "USE", "SET", "LIMIT"}
_WRITE_KEYWORDS = {
    "INSERT", "CREATE", "ALTER", "COMMENT", "COPY", "OPTIMIZE", "ANALYZE", "REFRESH", "CACHE", "UNCACHE",
    "MSCK", "REPAIR", "UNDROP", "SYNC", "CLONE", "REORG", "CLEAR", "RESET", "DECLARE", "ADD", "INSTALL",
}
_DESTRUCTIVE_KEYWORDS = {"DROP", "DELETE", "TRUNCATE", "UPDATE", "MERGE", "VACUUM", "RESTORE", "REPLACE", "REMOVE"}
_SECURITY_KEYWORDS = {"GRANT", "REVOKE", "DENY"}

_SECURITY_PATTERNS = [
    re.compile(r"\bOWNER\s+TO\b"),
    re.compile(r"\bSET\s+ROW\s+FILTER\b"),
    re.compile(r"\bDROP\s+ROW\s+FILTER\b"),
    re.compile(r"\bSET\s+MASK\b"),
    re.compile(r"\bDROP\s+MASK\b"),
    re.compile(r"\b(SHARE|RECIPIENT|PROVIDER|CONNECTION|STORAGE\s+CREDENTIAL|EXTERNAL\s+LOCATION|CREDENTIAL)\b"),
    re.compile(r"\bPOLICY\b"),
]
_DESTRUCTIVE_PATTERNS = [
    (re.compile(r"\bINSERT\s+OVERWRITE\b"), "INSERT OVERWRITE replaces existing data"),
    (re.compile(r"\bOR\s+REPLACE\b"), "OR REPLACE replaces an existing object"),
    (re.compile(r"\bDROP\b"), "contains DROP"),
    (re.compile(r"\bREPLACE\s+WHERE\b"), "REPLACE WHERE overwrites matching data"),
    (re.compile(r"\bTRUNCATE\b"), "contains TRUNCATE"),
    (re.compile(r"\bUNSET\b"), "UNSET removes properties/tags"),
]
_WITH_DML = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE)\b")


@dataclass
class StatementClassification:
    statement: str
    keyword: str
    kind: str  # read | write | destructive | security | unknown
    levels: frozenset[SafetyLevel]
    reasons: list[str] = field(default_factory=list)


def _strip_comments_and_literals(sql: str) -> str:
    """Remove comments and replace string/identifier literals with placeholders."""
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":
            j = sql.find("\n", i)
            i = n if j == -1 else j
        elif ch == "/" and nxt == "*":
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            out.append(" ")
        elif ch in ("'", '"', "`"):
            quote = ch
            i += 1
            while i < n:
                if sql[i] == "\\" and quote != "`":
                    i += 2
                    continue
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        i += 2
                        continue
                    break
                i += 1
            i += 1
            out.append(" _lit_ " if quote != "`" else " _ident_ ")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def split_sql_statements(sql: str) -> list[str]:
    """Split a SQL script on top-level semicolons (ignoring those in comments/literals)."""
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":
            j = sql.find("\n", i)
            j = n if j == -1 else j
            buf.append(sql[i:j])
            i = j
            continue
        if ch == "/" and nxt == "*":
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            buf.append(sql[i:j])
            i = j
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            j = i + 1
            while j < n:
                if sql[j] == "\\" and quote != "`":
                    j += 2
                    continue
                if sql[j] == quote:
                    if j + 1 < n and sql[j + 1] == quote:
                        j += 2
                        continue
                    break
                j += 1
            buf.append(sql[i : j + 1])
            i = j + 1
            continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if _strip_comments_and_literals(stmt).strip():
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail and _strip_comments_and_literals(tail).strip():
        statements.append(tail)
    return statements


def classify_statement(statement: str) -> StatementClassification:
    """Lexically classify a single SQL statement.

    This is a conservative *guardrail*, not a security boundary: anything not
    recognised is treated as potentially destructive. Real enforcement is done by
    Databricks permissions on the authenticated principal.
    """
    cleaned = _strip_comments_and_literals(statement).upper()
    tokens = re.findall(r"[A-Z_][A-Z0-9_]*", cleaned)
    if not tokens:
        raise ValidationFailed("Empty SQL statement")
    keyword = tokens[0]
    if keyword == "(":
        keyword = "SELECT"
    reasons: list[str] = []
    levels: set[SafetyLevel] = {SafetyLevel.EXECUTION}

    if keyword in _SECURITY_KEYWORDS:
        levels |= {SafetyLevel.SECURITY_SENSITIVE, SafetyLevel.WRITE}
        if keyword in {"REVOKE", "DENY"}:
            levels.add(SafetyLevel.DESTRUCTIVE)
        reasons.append(f"{keyword} changes permissions")
        kind = "security"
    elif keyword in _DESTRUCTIVE_KEYWORDS:
        levels |= {SafetyLevel.DESTRUCTIVE, SafetyLevel.WRITE}
        reasons.append(f"{keyword} removes or overwrites data/objects")
        kind = "destructive"
    elif keyword == "WITH":
        if _WITH_DML.search(cleaned):
            levels |= {SafetyLevel.DESTRUCTIVE, SafetyLevel.WRITE}
            reasons.append("WITH ... followed by data modification")
            kind = "destructive"
        else:
            levels.add(SafetyLevel.READ_ONLY)
            kind = "read"
    elif keyword in _READ_KEYWORDS:
        levels.add(SafetyLevel.READ_ONLY)
        kind = "read"
        if keyword == "SET" and re.search(r"\bSET\s+(TAGS|OWNER|ROW|MASK|TBLPROPERTIES)\b", cleaned):
            levels = {SafetyLevel.EXECUTION, SafetyLevel.WRITE}
            kind = "write"
    elif keyword in _WRITE_KEYWORDS:
        levels.add(SafetyLevel.WRITE)
        kind = "write"
    else:
        levels |= {SafetyLevel.WRITE, SafetyLevel.DESTRUCTIVE}
        reasons.append(f"Unrecognised statement type {keyword!r}; treated as potentially destructive")
        kind = "unknown"

    if kind in {"write", "security"}:
        for pattern, reason in _DESTRUCTIVE_PATTERNS:
            if pattern.search(cleaned):
                levels.add(SafetyLevel.DESTRUCTIVE)
                reasons.append(reason)
                if kind == "write":
                    kind = "destructive"
                break
    if kind in {"write", "destructive"}:
        for pattern in _SECURITY_PATTERNS:
            if pattern.search(cleaned):
                levels.add(SafetyLevel.SECURITY_SENSITIVE)
                reasons.append("affects access control, sharing, credentials or security policies")
                break

    return StatementClassification(statement, keyword, kind, frozenset(levels), reasons)


def classify_sql(sql: str) -> tuple[list[StatementClassification], frozenset[SafetyLevel]]:
    statements = split_sql_statements(sql)
    if not statements:
        raise ValidationFailed("No SQL statement provided")
    classified = [classify_statement(s) for s in statements]
    combined: set[SafetyLevel] = set()
    for c in classified:
        combined |= c.levels
    if any(SafetyLevel.READ_ONLY not in c.levels for c in classified):
        combined.discard(SafetyLevel.READ_ONLY)
    return classified, frozenset(combined)
