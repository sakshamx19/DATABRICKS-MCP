# dbx-mcp: a safety-first Databricks MCP server

`dbx-mcp` is an open-source [Model Context Protocol](https://modelcontextprotocol.io) server. It
lets AI agents and assistants work with a Databricks workspace: SQL, clusters and warehouses,
notebooks, Jobs, Lakeflow pipelines, Unity Catalog, Volumes, AI/BI dashboards, Genie, model
serving, Vector Search, Lakebase and Apps.

It is built on the official [Databricks SDK for Python](https://github.com/databricks/databricks-sdk-py)
and the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk). It does not
use or copy any other Databricks MCP implementation.

**Design goals**

- **Safe by default.** Every action has a safety class: read, write, destructive, execution or
  security-sensitive.
  - Destructive and security-sensitive changes use two steps. The first call returns a plan and
    changes nothing; the change runs only when the call is repeated with `confirm=true`.
  - Resources whose name or tags mark them as production are protected.
  - The server can run fully read-only.
- **Honest.** Every SDK call is checked against the real SDK. Features with no official API raise
  `UNSUPPORTED_OPERATION` with an explanation; nothing is faked.
- **Machine-readable.** Every tool has a typed input schema and a typed output envelope, plus a
  human-readable summary.
- **No secret leakage.** Responses and logs pass through secret redaction. Credentials are never
  returned unless a tool exists for that purpose and the user explicitly asks.

---

## Contents

1. [Quick start](#quick-start)
2. [Installation](#installation)
3. [Authentication](#authentication)
4. [Configuration (environment variables)](#configuration)
5. [MCP client setup](#mcp-client-setup)
6. [Tools](#tools)
7. [Security model](#security-model)
8. [Example tool calls](#example-tool-calls)
9. [Architecture](#architecture)
10. [Development](#development)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)
13. [Known limitations](#known-limitations)

---

## Quick start

```bash
git clone <this repo> dbx-mcp && cd dbx-mcp
uv venv && uv pip install -e .            # or: python -m venv .venv && pip install -e .
cp .env.example .env                       # set DATABRICKS_HOST + credentials
dbx-mcp --env-file .env --list-tools       # verify configuration
dbx-mcp --env-file .env --read-only        # start (stdio) in read-only mode
```

## Installation

You need Python 3.10 or newer.

```bash
pip install -e .            # core
pip install -e ".[pdf]"     # adds HTML->PDF conversion (xhtml2pdf) for generate_and_upload_pdf
pip install -e ".[dev]"     # tests + linters
```

For reproducible installs with the exact tested versions, use the pinned files (generated from
`pyproject.toml`):

```bash
pip install -r requirements.txt && pip install --no-deps -e .        # runtime (incl. PDF support)
pip install -r requirements-dev.txt && pip install --no-deps -e .    # + tests and linters
```

To regenerate the pinned files after changing dependencies:
`uv pip compile pyproject.toml --extra pdf --python-version 3.10 -o requirements.txt`
(add `--extra dev` and `-o requirements-dev.txt` for the dev file).

The installed entry points are `dbx-mcp` and `python -m dbx_mcp`.

```
dbx-mcp [--transport stdio|streamable-http|sse] [--host 127.0.0.1] [--port 8765]
        [--env-file PATH] [--read-only] [--list-tools] [--version]
```

`stdio` is the default and the recommended transport. The HTTP transports bind to `127.0.0.1` by
default. Exposing them on a network gives anyone who can reach the port your Databricks
permissions, so put an authenticating proxy in front.

## Authentication

Authentication is handled entirely by the Databricks SDK's
[unified authentication](https://docs.databricks.com/en/dev-tools/auth/unified-auth.html). The
server never reads, stores or returns credentials itself. Supported methods:

| Method | Environment |
|---|---|
| Personal access token | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` |
| OAuth M2M (service principal) | `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` |
| OAuth U2M (browser login) | run `databricks auth login --host ...` once, then `DATABRICKS_CONFIG_PROFILE` |
| Config profile | `DATABRICKS_CONFIG_PROFILE` (from `~/.databrickscfg`) |
| Azure (CLI, MSI, service principal) | `DATABRICKS_HOST` plus `ARM_*` / Azure CLI login |
| Google Cloud | `DATABRICKS_HOST` plus `GOOGLE_CREDENTIALS` / `DATABRICKS_GOOGLE_SERVICE_ACCOUNT` |

**Least privilege.** The server can do anything the authenticated principal can do. For agents,
prefer a dedicated **service principal** granted only the Unity Catalog privileges and workspace
entitlements it needs. Add `DBX_MCP_READ_ONLY=true` for exploration-only use.

`get_current_user` and `manage_workspace action=info` show which identity and workspace are
active (never tokens). `manage_workspace action=switch_profile` reconnects using another profile.

## Configuration

Server behaviour is configured with `DBX_MCP_*` environment variables. All of them are optional.

| Variable | Default | Purpose |
|---|---|---|
| `DBX_MCP_AUTH_MODE` | `env` | `env`: the server's own credentials (one workspace). `request`: each HTTP request sends its workspace URL + PAT in headers (multi-workspace; see below). CLI: `--auth-mode`. |
| `DBX_MCP_ALLOWED_WORKSPACE_HOSTS` | Databricks domains | Request mode: allowed host suffixes (e.g. `.azuredatabricks.net`), or `*` for any host. |
| `DBX_MCP_REQUEST_CLIENT_CACHE_SIZE` | `64` | Request mode: number of per-credential SDK clients kept in memory. |
| `DBX_MCP_TOOLSETS` | `all` | Comma list of toolsets to enable (see [docs/TOOLS.md](docs/TOOLS.md)). |
| `DBX_MCP_DISABLED_TOOLS` | | Comma list of individual tools to hide. |
| `DBX_MCP_READ_ONLY` | `false` | Allow only read actions (SELECTs are allowed; writes, DDL and code execution are not). |
| `DBX_MCP_BLOCKED_SAFETY_LEVELS` | | Block classes entirely, e.g. `DESTRUCTIVE,SECURITY_SENSITIVE,EXECUTION`. |
| `DBX_MCP_REQUIRE_CONFIRMATION` | `true` | Two-step `confirm=true` for destructive or security-sensitive changes. |
| `DBX_MCP_CONFIRM_EXECUTION` | `false` | Also require confirmation for code/job execution. |
| `DBX_MCP_PROTECTED_NAME_PATTERNS` | `(?i)(^\|[-_ .])prod(uction)?($\|[-_ .])` | Regexes for resource names and tags that must not be deleted, terminated or changed. `none` disables. |
| `DBX_MCP_ALLOW_PROTECTED_CHANGES` | `false` | Allow changes to protected resources (still requires confirmation). |
| `DBX_MCP_ALLOWED_VOLUME_PREFIXES` | | Restrict volume file tools to these `/Volumes/...` prefixes. |
| `DBX_MCP_ALLOWED_WORKSPACE_PREFIXES` | | Restrict workspace file tools to these paths. |
| `DBX_MCP_LOCAL_FILE_ROOT` | (disabled) | Directory the server may read from or write to for local uploads and downloads. |
| `DBX_MCP_DEFAULT_WAREHOUSE_ID` | `DATABRICKS_WAREHOUSE_ID` | Warehouse for SQL tools. |
| `DBX_MCP_WAREHOUSE_SELECTION` | `prefer_running` | `prefer_running` (automatic, explained in every response) or `configured_only`. |
| `DBX_MCP_DEFAULT_CLUSTER_ID` | `DATABRICKS_CLUSTER_ID` | Cluster for `execute_code`. |
| `DBX_MCP_SQL_MAX_ROWS` | `1000` | Hard cap on rows returned by SQL tools. |
| `DBX_MCP_SQL_WAIT_TIMEOUT_SECONDS` | `30` | How long SQL waits (5-50) before returning a pending statement id. |
| `DBX_MCP_DEFAULT_PAGE_SIZE` / `DBX_MCP_MAX_PAGE_SIZE` | `50` / `100` | Pagination. |
| `DBX_MCP_MAX_INLINE_DOWNLOAD_BYTES` | `10485760` | Max file bytes returned inline. |
| `DBX_MCP_TOOL_TIMEOUT_SECONDS` | `300` | Per-call timeout. |
| `DBX_MCP_MAX_WAIT_SECONDS` | `240` | Cap for `wait=true` on long-running operations (must be below the tool timeout). |
| `DBX_MCP_HTTP_TIMEOUT_SECONDS` | `60` | Per HTTP request to Databricks. |
| `DBX_MCP_RETRY_TIMEOUT_SECONDS` | `300` | SDK retry budget for 429/503/transient errors. |
| `DBX_MCP_RATE_LIMIT_PER_SECOND` | | Client-side request rate limit. |
| `DBX_MCP_MANIFEST_PATH` | `.databricks_mcp/manifest.json` | Project manifest file. |
| `DBX_MCP_LOG_LEVEL` | `INFO` | JSON logs to stderr. |
| `DBX_MCP_DEBUG` | `false` | Include stack traces in errors (development only). |

## MCP client setup

**Claude Code**

```bash
claude mcp add databricks -- dbx-mcp --env-file /absolute/path/to/.env
```

**Claude Desktop / Cursor / any client using `mcpServers` JSON**

```json
{
  "mcpServers": {
    "databricks": {
      "command": "dbx-mcp",
      "args": ["--env-file", "/absolute/path/to/.env"],
      "env": {
        "DBX_MCP_TOOLSETS": "identity,sql,compute,unity_catalog,volumes",
        "DBX_MCP_READ_ONLY": "true"
      }
    }
  }
}
```

If `dbx-mcp` is not on the client's `PATH`, use the absolute path to the virtualenv's
executable, e.g. `/path/to/repo/.venv/bin/dbx-mcp` (Windows: `.venv\\Scripts\\dbx-mcp.exe`).

**VS Code** (`.vscode/mcp.json`)

```json
{
  "servers": {
    "databricks": { "type": "stdio", "command": "dbx-mcp", "args": ["--env-file", "${workspaceFolder}/.env"] }
  }
}
```

**LangGraph / LangChain agents:** see [examples/langgraph_agent](examples/langgraph_agent/README.md).
It is a working agent that connects through `langchain-mcp-adapters` and adds a human-approval
gate for `confirm=true` calls.

**HTTP transport** (for clients that connect by URL): `dbx-mcp --transport streamable-http --port 8765`,
then connect to `http://127.0.0.1:8765/mcp`.

### One server, many workspaces (request-auth mode)

In request-auth mode the server stores **no Databricks credentials**. Each client sends its own
workspace URL and PAT as HTTP headers, so one running server can serve any number of
workspaces and users at once:

```bash
dbx-mcp --auth-mode request --transport streamable-http --host 0.0.0.0 --port 8765
```

Client config. Add one entry per workspace; all entries point at the same server:

```json
{
  "mcpServers": {
    "databricks-prod-eu": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "X-Databricks-Host": "https://adb-1111111111111111.1.azuredatabricks.net",
        "Authorization": "Bearer dapi...",
        "X-Databricks-Warehouse-Id": "optional-default-warehouse"
      }
    },
    "databricks-dev-us": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "X-Databricks-Host": "https://dbc-2222.cloud.databricks.com",
        "Authorization": "Bearer dapi..."
      }
    }
  }
}
```

| Header | Required | Meaning |
|---|---|---|
| `X-Databricks-Host` | yes | Workspace URL (`https://...`, root only) |
| `Authorization: Bearer <PAT>` (or `X-Databricks-Token`) | yes | That workspace's personal access token |
| `X-Databricks-Warehouse-Id` | no | Default SQL warehouse for this connection |
| `X-Databricks-Cluster-Id` | no | Default cluster for `execute_code` on this connection |

How it behaves:

- **Isolation:** each request's credentials get their own SDK client, built only from the
  headers. The server machine's env vars and `~/.databrickscfg` are never mixed in.
- **Secrets:** clients are cached by a SHA-256 hash of host + token; tokens are never logged or
  returned.
- **Allowed hosts:** only Databricks domains are accepted (`*.azuredatabricks.net`,
  `*.cloud.databricks.com`, `*.gcp.databricks.com`, ...), which stops the server being used to
  reach arbitrary or internal URLs. Adjust with `DBX_MCP_ALLOWED_WORKSPACE_HOSTS`.
- **Not available in this mode:** server-side profiles (`list_profiles`, `switch_profile`) and
  server-wide default warehouse/cluster ids. Use the per-connection headers instead.
- **Manifest:** entries are scoped per workspace, so tenants only see their own.
- **Transport:** request mode needs an HTTP transport; stdio is refused.
- **Use HTTPS beyond localhost.** PATs travel in headers, so put the server behind a
  TLS-terminating reverse proxy (nginx, Caddy, a cloud load balancer). A request without valid
  headers simply fails; there is no shared server identity to fall back on.

## Tools

Tools are grouped into toolsets that can be enabled independently. See **[docs/TOOLS.md](docs/TOOLS.md)**
for the complete reference: every parameter, the safety class of every action, and the capability matrix.

| Toolset | Tools |
|---|---|
| `identity` | `get_current_user`, `manage_workspace` |
| `sql` | `execute_sql`, `execute_sql_multi`, `manage_sql_statement`, `get_table_stats_and_schema` |
| `compute` | `manage_cluster`, `manage_sql_warehouse`, `manage_warehouse`, `list_compute` |
| `workspace` | `execute_code`, `manage_workspace_files` |
| `jobs` | `manage_jobs`, `manage_job_runs` |
| `pipelines` | `manage_pipeline`, `manage_pipeline_run` |
| `unity_catalog` | `manage_uc_objects`, `manage_uc_grants`, `manage_uc_storage`, `manage_uc_connections`, `manage_uc_tags`, `manage_uc_security_policies`, `manage_uc_monitors`, `manage_uc_sharing`, `manage_metric_views` |
| `volumes` | `get_volume_folder_details`, `manage_volume_files` |
| `dashboards` | `manage_dashboard` |
| `ai` | `manage_serving_endpoint`, `manage_ka`, `manage_mas`, `manage_genie`, `ask_genie` |
| `vector_search` | `manage_vs_endpoint`, `manage_vs_index`, `query_vs_index`, `manage_vs_data` |
| `lakebase` | `manage_lakebase_database`, `manage_lakebase_branch`, `manage_lakebase_sync`, `generate_lakebase_credential` |
| `apps` | `manage_app` |
| `manifest` | `list_tracked_resources`, `delete_tracked_resource` |
| `pdf` | `generate_and_upload_pdf` |

### Response envelope

Every tool returns:

```jsonc
{
  "status": "success | pending | dry_run | confirmation_required | partial_failure | failed",
  "tool": "manage_cluster", "action": "terminate",
  "summary": "Human-readable one-liner.",
  "safety": ["DESTRUCTIVE"],
  "data": { /* machine-readable result */ },
  "page": { "page_size": 50, "returned": 50, "has_more": true, "next_page_token": "..." },
  "plan": { /* what will change: present for dry_run / confirmation_required */ },
  "warnings": [], "next_steps": [], "request_id": "4f1c..."
}
```

Errors are MCP tool errors with a category prefix, for example `[NOT_FOUND]`, `[PERMISSION_DENIED]`,
`[AUTHENTICATION_FAILED]`, `[INVALID_PARAMETER]`, `[CONFLICT]`, `[RATE_LIMITED]`, `[TIMEOUT]`,
`[DATABRICKS_SERVICE_ERROR]`, `[UNSUPPORTED_OPERATION]` or `[BLOCKED_BY_SAFETY_POLICY]`. Each
category comes with a hint and, when Databricks provides one, a request id.

## Security model

| Control | Behaviour |
|---|---|
| Safety classification | Every action is READ_ONLY, WRITE, DESTRUCTIVE, EXECUTION and/or SECURITY_SENSITIVE. The class is shown in the tool description and in MCP tool annotations (`readOnlyHint`, `destructiveHint`). |
| Central enforcement | A single wrapper applies policy, dry-run, confirmation, timeout, redaction and error handling to every tool, so an individual tool cannot skip them. Registration fails if a mutating tool lacks `dry_run` or `confirm`. |
| Two-step confirmation | DESTRUCTIVE and SECURITY_SENSITIVE changes first return `confirmation_required` with a plan (target, current state, diff, warnings, reversibility). Nothing runs until the call is repeated with `confirm=true`. Agents are instructed to get user approval first. |
| Dry run | `dry_run=true` previews any change. |
| Read-only mode and blocked classes | `DBX_MCP_READ_ONLY`, `DBX_MCP_BLOCKED_SAFETY_LEVELS`. |
| SQL classification | Statements are classified lexically (comments and literals stripped), so `DROP`, `DELETE`, `TRUNCATE`, `UPDATE`, `MERGE`, `INSERT OVERWRITE`, `CREATE OR REPLACE`, `GRANT`/`REVOKE`, ownership, row-filter and mask changes are detected. Unknown statements count as destructive. **This is a guardrail, not a security boundary;** real enforcement is Databricks permissions on the principal. |
| Production protection | Deleting, terminating, stopping or changing resources whose name or tags match `DBX_MCP_PROTECTED_NAME_PATTERNS` is refused unless explicitly allowed. |
| Grants | Grant and revoke plans show a before/after diff. `ALL_PRIVILEGES` needs an explicit flag, and broad principals produce a warning. |
| Paths | Volume and workspace paths are validated (no `..`, no relative paths, no control characters, no backslashes), with optional prefix allowlists. Local filesystem access is off unless `DBX_MCP_LOCAL_FILE_ROOT` is set, and is confined to it. |
| SQL injection | Values go through bound statement parameters. Identifiers and literals that tools build into DDL are quoted and escaped. |
| Secrets | Responses (data, summary, warnings) and logs are redacted by key name and by pattern (PATs, JWTs, bearer tokens, `password=`). Connection options and recipient activation links are stripped. The Lakebase credential token is returned only with `reveal_token=true` plus confirmation. |
| Logging | Structured JSON on **stderr** (stdout is the MCP channel): tool, action, request id, user, duration, outcome and error category. No arguments, tokens or query results are logged. |
| Errors | Normalized and categorized. Stack traces are shown only with `DBX_MCP_DEBUG=true`. |

## Example tool calls

```jsonc
// Who am I?
{"name": "get_current_user", "arguments": {}}

// Explore
{"name": "manage_uc_objects", "arguments": {"action": "list", "object_type": "schema", "catalog_name": "main"}}
{"name": "get_table_stats_and_schema", "arguments": {"name": "main.sales.orders", "stats": "metadata"}}

// Query (parameters are bound server-side)
{"name": "execute_sql", "arguments": {
  "statement": "SELECT region, SUM(amount) AS total FROM main.sales.orders WHERE day >= :d GROUP BY region",
  "parameters": {"d": "2026-01-01"}, "max_rows": 100, "row_format": "objects"}}

// Destructive: the first call returns a plan; repeat with confirm=true after user approval
{"name": "execute_sql", "arguments": {"statement": "DROP TABLE main.sandbox.tmp_orders"}}
// -> {"status": "confirmation_required", "plan": {...}, "next_steps": ["... confirm=true"]}
{"name": "execute_sql", "arguments": {"statement": "DROP TABLE main.sandbox.tmp_orders", "confirm": true}}

// Long-running: returns quickly; poll for status
{"name": "manage_job_runs", "arguments": {"action": "get", "run_id": 123}}

// Pagination
{"name": "manage_cluster", "arguments": {"action": "list", "page_size": 20, "page_token": "eyJvIjogMjB9"}}

// Genie (the answer is model-generated; check the SQL)
{"name": "ask_genie", "arguments": {"space_id": "01ef...", "question": "Top 5 products by revenue last month?"}}
```

## Architecture

```
src/dbx_mcp/
  __main__.py            CLI entry point (transport, --env-file, --read-only, --list-tools)
  server/
    app.py               builds the MCPServer and registers enabled toolsets
    config.py            Settings from DBX_MCP_* env vars
    context.py           process-wide context (settings, client, safety policy, manifest)
    manifest.py          local JSON project manifest
  databricks/
    client.py            the only place a WorkspaceClient is created (auth, retries, timeouts)
    sql_runner.py        Statement Execution API, warehouse selection, result chunking
  safety/
    levels.py            SafetyLevel enum and semantics
    guard.py             policy, confirmation, production protection, path allowlists
    validation.py        path/identifier validation, quoting, SQL statement classifier
  tools/
    registry.py          @tool decorator + the wrapper applying all cross-cutting behaviour
    common.py            shared parameter types and response helpers
    identity.py sql.py compute.py workspace.py jobs.py pipelines.py volumes.py
    dashboards.py ai.py vector_search.py lakebase.py apps.py manifest.py pdf.py
    unity_catalog/       objects, grants, storage, connections, tags, security_policies,
                         monitors, sharing, metric_views
  models/                typed pydantic response models
  utils/                 errors, logging, redaction, pagination, serialization
```

Key mechanisms:

- **`call_with_spec`.** Create and update tools accept a `spec` using Databricks REST field names.
  It is validated against the real SDK method signature and converted to SDK dataclasses. Unknown
  fields and invalid enum values are rejected; the SDK alone would silently drop them.
- **Pagination.** Every list action returns an opaque `next_page_token`.
- **Long-running operations** return immediately with an id and state (`status: pending`). An
  optional `wait=true` is bounded by `DBX_MCP_MAX_WAIT_SECONDS`.
- **Retries and rate limits** use the SDK's built-in retry/backoff (`DBX_MCP_RETRY_TIMEOUT_SECONDS`)
  plus an optional client-side rate limit.

## Development

```bash
uv venv && uv pip install -e ".[dev,pdf]"
ruff check src tests
python -m pytest                        # unit tests (no Databricks needed)
python scripts/gen_tool_docs.py         # regenerate docs/TOOLS.md after changing tools
```

**Adding a tool:**

1. Write a synchronous function in a module under `tools/`.
2. Decorate it with `@tool(toolset=..., title=..., safety={action: levels})`.
3. Give it typed `Annotated[..., Field(description=...)]` parameters, and add `dry_run` / `confirm`
   if it mutates (registration enforces this).
4. Return `ok(...)` or `paged_response(...)`.
5. For destructive or security-sensitive actions, add a `preview` function that describes exactly
   what will change, and call `ctx().safety.check_protected(...)`.
6. **Verify every SDK method, field and enum value by introspection** before using it.

## Testing

```bash
python -m pytest tests/unit                    # fast, fully mocked
DBX_MCP_RUN_INTEGRATION=1 INTEGRATION_ENV_FILE=.env python -m pytest tests/integration
```

- **Unit tests** mock the `WorkspaceClient` with services *autospecced from the real SDK
  classes*. A tool that calls a non-existent SDK method or passes a wrong keyword argument fails
  its tests.
- **Integration tests** never run unless `DBX_MCP_RUN_INTEGRATION=1` is set. The default suite is
  **read-only**: the server is forced into read-only mode, and SQL runs only on an
  already-running warehouse.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[AUTHENTICATION_FAILED] ... cannot configure default credentials` | Set `DATABRICKS_HOST` plus credentials, or pass `--env-file`. Check with `databricks auth describe`. |
| `[PERMISSION_DENIED]` | The principal lacks a privilege (UC grant, warehouse `CAN_USE`, cluster `CAN_ATTACH_TO`, ...). Run `get_current_user` to see the identity. |
| `[BLOCKED_BY_SAFETY_POLICY] ... read-only mode` | Unset `DBX_MCP_READ_ONLY`, or adjust `DBX_MCP_BLOCKED_SAFETY_LEVELS`. |
| `... protected/production resource` | Intended. Set `DBX_MCP_ALLOW_PROTECTED_CHANGES=true` or adjust `DBX_MCP_PROTECTED_NAME_PATTERNS` if appropriate. |
| SQL says `No SQL warehouses are visible` | Create or grant a warehouse, or set `DBX_MCP_DEFAULT_WAREHOUSE_ID`. |
| SQL returns `status: pending` | The query is still running. Poll `manage_sql_statement action=get`. |
| Responses are truncated | Raise `max_rows` (up to `DBX_MCP_SQL_MAX_ROWS`) or add `LIMIT`/filters. |
| Client shows no tools or garbled output | Something printed to stdout. The server logs only to stderr; check wrappers or shell profile output. |
| `generate_and_upload_pdf` reports missing dependency | `pip install "dbx-mcp[pdf]"`. |
| Need details of an unexpected error | Set `DBX_MCP_DEBUG=true` and `DBX_MCP_LOG_LEVEL=DEBUG` (development only). |

## Known limitations

See the "Limitations" notes in [docs/TOOLS.md](docs/TOOLS.md) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
Capabilities without a stable official API are reported as `UNSUPPORTED_OPERATION` instead of being
emulated.

## License

Apache-2.0. See [LICENSE](LICENSE).
