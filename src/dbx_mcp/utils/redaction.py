"""Secret redaction for tool output and logs.

Defense in depth: every tool response passes through :func:`redact` before it
leaves the server, so a field the Databricks API happens to echo back (a
connection password, a token, a client secret, ...) is never forwarded to the
MCP client.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "***REDACTED***"

# Key names (case-insensitive, compared after removing '-' and '_') whose values are secrets.
_SECRET_KEYS = {
    "token",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "oauthtoken",
    "bearertoken",
    "sessiontoken",
    "sastoken",
    "pat",
    "password",
    "passwd",
    "pwd",
    "secret",
    "clientsecret",
    "secretvalue",
    "stringvalue",  # Databricks secret scope values
    "bytesvalue",
    "privatekey",
    "privatekeyid",
    "privatekeycontent",
    "pemprivatekey",
    "secretaccesskey",
    "awssecretaccesskey",
    "accountkey",
    "connectionstring",
    "apikey",
    "authorization",
    "credentialjson",
    "googlecredentials",
    "jsoncredentials",
    "sharedaccesssignature",
    "activationurl",  # Delta Sharing recipient activation links grant access
    "sharingcode",
    "recipientprofile",
    "recipientprofilestr",
    "bearer",
    "personalaccesstoken",
    "tokenvalue",
    "encryptedvalue",
    "sslkey",
}

# Suffix-based detection, e.g. "aws_secret_access_key", "openai_api_key", "pg_password".
_SECRET_SUFFIXES = ("password", "secret", "apikey", "privatekey", "accesskey", "token", "credentials")

# Keys that end in a secret suffix but are NOT secrets.
_SAFE_KEYS = {
    "tokenid",
    "tokentype",
    "nextpagetoken",
    "pagetoken",
    "prevpagetoken",
    "previouspagetoken",
    "maxtokens",
    "idempotencytoken",
    "tokenexpirytime",
    "tokenexpiration",
    "expirationtime",
    "secretscope",
    "secretkey",  # a *reference* {{secrets/scope/key}}, not the value
    "accesskeyid",
    "awsaccesskeyid",
    "storagecredentials",
    "numtokens",
    "totaltokens",
    "inputtokens",
    "outputtokens",
    "prompttokens",
    "completiontokens",
    "usagetokens",
    "credentialname",
}

_INLINE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"dapi[0-9a-f]{32}(-\d)?", re.IGNORECASE), REDACTED),  # Databricks PAT
    (re.compile(r"dose[0-9a-f]{32}", re.IGNORECASE), REDACTED),  # Databricks OAuth secret
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]{20,}=*"), r"\g<1>" + REDACTED),
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), REDACTED),  # JWT
    (re.compile(r"(?i)\b(password|pwd|secret|token)=([^&\s;'\"]+)"), r"\g<1>=" + REDACTED),
]


def _normalise(key: str) -> str:
    return key.replace("-", "").replace("_", "").lower()


def is_secret_key(key: str) -> bool:
    norm = _normalise(key)
    if norm in _SAFE_KEYS:
        return False
    if norm in _SECRET_KEYS:
        return True
    return any(norm.endswith(suffix) for suffix in _SECRET_SUFFIXES)


def redact_text(text: str) -> str:
    """Mask secret-looking substrings (PATs, JWTs, bearer tokens, key=value secrets)."""
    out = text
    for pattern, replacement in _INLINE_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact(value: Any, allow_keys: Iterable[str] = (), _depth: int = 0) -> Any:
    """Return a deep copy of ``value`` with secret-valued keys masked.

    ``allow_keys`` exempts specific key names (used only by tools whose explicit,
    confirmed purpose is to hand a credential to the caller).
    """
    allowed = {_normalise(k) for k in allow_keys}
    if _depth > 64:
        return value
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            if (
                isinstance(key, str)
                and is_secret_key(key)
                and _normalise(key) not in allowed
                and item not in (None, "", [], {})
                and not isinstance(item, bool)
            ):
                result[key] = REDACTED
            else:
                result[key] = redact(item, allowed, _depth + 1)
        return result
    if isinstance(value, list | tuple):
        return [redact(item, allowed, _depth + 1) for item in value]
    if isinstance(value, str):
        return redact_text(value) if not allowed else value
    return value
