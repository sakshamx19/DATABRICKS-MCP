"""Conversion between Databricks SDK objects and JSON-compatible data.

Two directions:

* :func:`to_jsonable` - SDK dataclasses / enums / iterators -> plain JSON data.
* :func:`coerce_kwargs` / :func:`call_with_spec` - a JSON ``spec`` supplied by the
  MCP client -> correctly-typed keyword arguments for an SDK method, validated
  against that method's real signature.

The SDK's own ``from_dict`` silently drops unknown keys and invalid enum values;
we detect that and raise a validation error instead, so a typo in a spec never
results in a silently different request.
"""

from __future__ import annotations

import base64
import dataclasses
import inspect
import typing
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

# ``Wait`` is the documented return type of the SDK's long-running operations.
from databricks.sdk.service._internal import Wait

from dbx_mcp.utils.errors import ValidationFailed


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert SDK objects (dataclasses with ``as_dict``), enums, etc. to JSON data."""
    if _depth > 50:
        return str(obj)
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        try:
            return to_jsonable(as_dict(), _depth + 1)
        except Exception:  # pragma: no cover - defensive; fall through to dataclass handling
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name), _depth + 1)
            for f in dataclasses.fields(obj)
            if getattr(obj, f.name) is not None
        }
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return str(obj)


def pick(data: Mapping[str, Any] | None, keys: Iterable[str]) -> dict[str, Any]:
    """Select a subset of keys (skipping missing/None) - used for compact list views."""
    if not data:
        return {}
    return {k: data[k] for k in keys if k in data and data[k] is not None}


# ----------------------------------------------------------------------------------------------
# spec -> SDK kwargs
# ----------------------------------------------------------------------------------------------

def _hints_from_class(klass: type, name: str) -> dict[str, Any]:
    # Mixin classes sometimes live in modules that don't import every annotation name;
    # walk the MRO to the same-named method in the generated service module.
    for base in klass.__mro__:
        candidate = base.__dict__.get(name)
        if candidate is None:
            continue
        try:
            hints = typing.get_type_hints(candidate)
        except Exception:
            continue
        if hints:
            return hints
    return {}


def _type_hints_for(method: Callable[..., Any]) -> dict[str, Any]:
    func = getattr(method, "__func__", method)
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}
    if hints:
        return hints
    owner = getattr(method, "__self__", None)
    if owner is not None and hasattr(func, "__name__"):
        return _hints_from_class(type(owner), func.__name__)
    # unittest.mock autospec of an SDK service (used in tests): resolve the real method.
    spec_class = getattr(getattr(method, "_mock_parent", None), "_spec_class", None)
    if spec_class is not None:
        return _hints_from_class(spec_class, getattr(method, "_mock_name", ""))
    return {}


def _unwrap_optional(tp: Any) -> Any:
    if typing.get_origin(tp) is typing.Union or type(tp).__name__ == "UnionType":
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _convert(value: Any, tp: Any, path: str) -> Any:
    tp = _unwrap_optional(tp)
    if value is None:
        return None
    origin = typing.get_origin(tp)
    if origin in (list, typing.List, Iterable):  # noqa: UP006
        (item_tp,) = typing.get_args(tp) or (Any,)
        if not isinstance(value, list):
            raise ValidationFailed(f"'{path}' must be a list")
        return [_convert(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, typing.Dict, Mapping):  # noqa: UP006
        args = typing.get_args(tp)
        val_tp = args[1] if len(args) == 2 else Any
        if not isinstance(value, dict):
            raise ValidationFailed(f"'{path}' must be an object")
        return {k: _convert(v, val_tp, f"{path}.{k}") for k, v in value.items()}
    if inspect.isclass(tp) and issubclass(tp, Enum):
        if isinstance(value, tp):
            return value
        try:
            return tp(value)
        except ValueError:
            allowed = ", ".join(str(m.value) for m in tp)
            raise ValidationFailed(f"'{path}' has invalid value {value!r}. Allowed: {allowed}") from None
    from_dict = getattr(tp, "from_dict", None) if inspect.isclass(tp) else None
    if callable(from_dict):
        if not isinstance(value, dict):
            raise ValidationFailed(f"'{path}' must be an object ({tp.__name__})")
        obj = from_dict(value)
        dropped = _dropped_keys(value, to_jsonable(obj), path)
        if dropped:
            raise ValidationFailed(
                f"Unknown or invalid field(s) for {tp.__name__}: {', '.join(dropped)}",
                hint="Check field names/enum values against the Databricks API reference.",
            )
        return obj
    if tp is bool and not isinstance(value, bool):
        raise ValidationFailed(f"'{path}' must be a boolean")
    if tp is int and (isinstance(value, bool) or not isinstance(value, int)):
        if isinstance(value, str) and value.isdigit():
            return int(value)
        raise ValidationFailed(f"'{path}' must be an integer")
    return value


def _dropped_keys(original: Any, parsed: Any, path: str) -> list[str]:
    """Keys present in ``original`` that did not survive SDK parsing."""
    dropped: list[str] = []
    if isinstance(original, dict):
        parsed_dict = parsed if isinstance(parsed, dict) else {}
        for key, value in original.items():
            if value in (None, [], {}, ""):
                continue
            child = f"{path}.{key}" if path else key
            if key not in parsed_dict:
                dropped.append(child)
            else:
                dropped.extend(_dropped_keys(value, parsed_dict[key], child))
    elif isinstance(original, list) and isinstance(parsed, list) and len(original) == len(parsed):
        for i, (o, p) in enumerate(zip(original, parsed, strict=False)):
            dropped.extend(_dropped_keys(o, p, f"{path}[{i}]"))
    return dropped


def coerce_kwargs(
    method: Callable[..., Any],
    spec: Mapping[str, Any] | None,
    *,
    fixed: Mapping[str, Any] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate ``spec`` against ``method``'s signature and convert values to SDK types.

    ``fixed`` arguments (e.g. an id taken from a dedicated tool parameter) are merged
    in and may not be overridden by ``spec``. ``exclude`` lists parameter names the
    caller must not set through ``spec`` (e.g. ``timeout``, ``callback``).
    """
    spec = dict(spec or {})
    fixed = dict(fixed or {})
    signature = inspect.signature(method)
    params = {
        name: p
        for name, p in signature.parameters.items()
        if name not in {"self", "timeout", "callback", *exclude} and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    hints = _type_hints_for(method)

    unknown = sorted(set(spec) - set(params))
    if unknown:
        allowed = ", ".join(sorted(set(params) - set(fixed)))
        raise ValidationFailed(f"Unknown field(s) in spec: {', '.join(unknown)}. Allowed: {allowed}")
    overlap = sorted(set(spec) & set(fixed))
    if overlap:
        raise ValidationFailed(f"Field(s) {', '.join(overlap)} must be passed as dedicated tool parameters, not in spec")

    kwargs: dict[str, Any] = {}
    for name, value in {**spec, **fixed}.items():
        kwargs[name] = _convert(value, hints.get(name, Any), name)

    missing = [
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty and name not in kwargs
    ]
    if missing:
        raise ValidationFailed(f"Missing required field(s): {', '.join(missing)}")
    return kwargs


def call_with_spec(
    method: Callable[..., Any],
    spec: Mapping[str, Any] | None,
    *,
    fixed: Mapping[str, Any] | None = None,
    exclude: Iterable[str] = (),
) -> Any:
    return method(**coerce_kwargs(method, spec, fixed=fixed, exclude=exclude))


def parse_sdk_object(cls: type, data: Mapping[str, Any], path: str = "spec") -> Any:
    """Parse a dict into an SDK dataclass, rejecting unknown/invalid fields."""
    return _convert(dict(data), cls, path)


def wait_response(waiter: Any) -> Any:
    """Return the immediate response carried by an SDK ``Wait`` object (or the object itself)."""
    return waiter.response if isinstance(waiter, Wait) else waiter
