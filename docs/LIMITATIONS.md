# Known limitations and unsupported operations

Following the project rule "never invent an API", anything without a reliable, documented
interface in `databricks-sdk` 0.146 is reported as `UNSUPPORTED_OPERATION` with an explanation,
not emulated. Items marked *unverified* are implemented from SDK and REST documentation but have
not yet been exercised against a live workspace by an integration test.

## Verified live (read-only sweep, Azure workspace)

The following list/get paths ran successfully against a real workspace through
`scripts/live_readonly_sweep.py`:

- identity, compute, warehouses, workspace files, jobs, job runs, pipelines
- UC catalogs, schemas, tables and volumes; grants and effective grants (catalog, schema, table)
- UC tags (catalog, table), row filters and masks, metric views, storage credentials, external
  locations, connections, Delta Sharing (shares, recipients, providers)
- dashboards, serving endpoints, Genie spaces, Knowledge Assistants, supervisor agents,
  Vector Search endpoints, Lakebase (provisioned and autoscaling), apps
- `execute_sql` on a running warehouse; read-only mode blocking destructive calls

## SQL

- Statement classification is lexical. It is a guardrail, **not a security boundary**; Databricks
  permissions are the real control.
- Results are fetched inline as JSON arrays: at most 25 MiB per chunk, and rows are capped by
  `DBX_MCP_SQL_MAX_ROWS`. Use `LIMIT` or filters for large results.
- `execute_sql_multi` has no transaction: completed statements are not rolled back.

## Compute and code

- `execute_code` with `compute="cluster"` needs a running classic cluster (Command Execution API).
- `execute_code` with `compute="serverless"` is Python only, via a one-time serverless job run of
  a temporary notebook. Executor-side output and `display()` are not captured. *Unverified live.*

## Jobs and pipelines

- Pipeline `update` merges your fields onto the current spec because the API replaces the full
  spec. `run_as` is not carried over automatically.
- Pipeline events filtered by `update_id` are filtered client-side over the newest 1000 events.
- Cancelling a job run and stopping a pipeline are classified DESTRUCTIVE, so they need
  confirmation.

## Unity Catalog

- **Tables:** the SDK can only create EXTERNAL Delta tables; create other tables with
  `execute_sql`. Table and function `update` supports `owner` only.
- **Connections:** `update` must include the full `options` map (including credentials), as the
  API requires.
- **External locations:** validation is done through the storage-credential validate endpoint.
- **Comments:** table and column comments are set with SQL (`COMMENT ON`/`ALTER COLUMN`); column
  comments on views may not be supported.
- **Monitors:** these use the `data_quality` API (`quality_monitors` is deprecated). There is no
  "list all monitors"; the SDK marks it unimplemented.
- **ABAC policies:** securable types are sent uppercase as in the SDK enum. *Unverified live for
  writes.*
- **Metric views:** these use documented SQL DDL (`WITH METRICS LANGUAGE YAML`); a comment must be
  part of the YAML.

## Volumes and PDF

- Recursive directory delete lists and deletes the contents (the API only removes empty
  directories), capped at 10,000 entries. Contents may change between preview and execution.
- Downloaded text passes through secret redaction, so secret-looking strings in file content are
  masked.
- The PDF Markdown converter supports a basic subset. Remote resources in HTML are never fetched.

## AI and Vector Search

- Supervisor agent tools support the types exposed by the SDK `Tool` dataclass: genie_space,
  knowledge_assistant, uc_function, uc_connection, app and volume.
- Not implemented: KA/MAS permission *set* (only update), provisioned-throughput endpoint
  creation, Genie feedback/comments/evals, and re-executing an expired Genie query.
- Vector Search indexes have no update API (recreate the index). Upsert and delete work on
  Direct Access indexes only.
- Genie answers are model-generated and are labelled as such.

## Lakebase

- Autoscaling: no list API for catalogs or synced tables; synced tables cannot be updated; the
  update-mask format (`spec.<field>`) is inferred (an explicit `update_mask` override exists).
- Provisioned: instances cannot be undeleted.
- Credential tokens are returned only with `reveal_token=true` plus confirmation.

## Apps and dashboards

- There is no app logs API in the SDK (`logs` returns UNSUPPORTED_OPERATION).
- Dashboard delete moves the dashboard to trash.
