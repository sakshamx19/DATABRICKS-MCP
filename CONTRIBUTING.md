# Contributing

Thanks for helping! Ground rules:

1. **Independent implementation.** Build on the official Databricks SDK, the REST API docs and the
   MCP SDK. Do not copy code from other Databricks MCP projects.
2. **No invented APIs.** Before using an SDK method, field or enum value, verify it by
   introspection (`inspect.signature`, `dataclasses.fields`) against the pinned SDK version.
   Unit tests autospec the SDK, so a wrong method or argument name fails CI. If a capability has
   no official API, raise `UnsupportedOperation` with an explanation instead of emulating it.
3. **Safety first.** Every action needs a safety classification. Mutating tools need `dry_run`;
   destructive, security-sensitive and execution tools need `confirm`, plus a `preview` that
   explains exactly what will change. Use `check_protected` for named resources.
4. **Tests.** Add unit tests for every action: happy path, validation errors, confirmation gating,
   read-only blocking, and error mapping.
5. **Docs.** Run `python scripts/gen_tool_docs.py` and commit the updated `docs/TOOLS.md`.

```bash
uv venv && uv pip install -e ".[dev,pdf]"
ruff check src tests && python -m pytest && python scripts/gen_tool_docs.py --check
```
