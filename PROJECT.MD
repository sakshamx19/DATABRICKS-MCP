You are a senior Python/MCP engineer specializing in Databricks, Unity Catalog, and Model Context Protocol (MCP).

I want you to design and implement a production-ready Databricks MCP Server.

## 1. Objective

Build an MCP server that exposes Databricks workspace, SQL, compute, Jobs, Pipelines, Unity Catalog, Volumes, AI/BI, serving, Lakebase, Apps, and related capabilities as MCP tools.

The MCP server should allow an MCP-compatible AI agent/client to safely discover, inspect, query, create, update, execute, and manage Databricks resources through well-defined MCP tools.

The implementation should be modular, maintainable, strongly typed, and production-oriented.

---

# 2. IMPORTANT CONSTRAINT

DO NOT use, copy, vendor, import, or depend on the implementation/components from:

https://github.com/databricks-solutions/ai-dev-kit

We are aware that this repository contains Databricks-related MCP/AI tooling.

You may use publicly available Databricks API/SDK documentation as technical reference, but:

- Do not copy its source code.
- Do not reuse its MCP tool implementations.
- Do not import its packages as dependencies.
- Do not reproduce its internal architecture merely by renaming components.
- Build our MCP server independently.
- Prefer official Databricks REST APIs, SDKs, SQL APIs, and documented interfaces.

If there is uncertainty about whether a component originates from the AI Dev Kit, implement an independent version using the official Databricks APIs instead.

---

# 3. Core Requirements

Use MCP as the interface layer.

The server should:

- Expose Databricks capabilities as MCP tools.
- Use structured input/output schemas.
- Validate all tool inputs.
- Return machine-readable results where possible.
- Return useful human-readable summaries alongside structured data where appropriate.
- Handle Databricks API errors cleanly.
- Support pagination.
- Support long-running operations.
- Provide clear error messages.
- Avoid exposing secrets, access tokens, passwords, or credentials in tool output.
- Follow least-privilege principles.
- Make destructive operations explicit.
- Keep the implementation modular so individual capabilities can be enabled/disabled.

Prefer Python unless there is a strong technical reason to use another language.

---

# 4. Databricks Authentication

Support standard Databricks authentication mechanisms supported by the official Databricks SDK/API.

The implementation should avoid hard-coding credentials.

Configuration should be supplied through environment variables, Databricks configuration, or another secure mechanism.

The server should expose:

- Current authenticated user
- Connected workspace information
- Workspace URL/profile information where appropriate

Never return secrets or authentication tokens through MCP responses.

---

# 5. MCP Tool Categories

Implement the following capability groups.

## A. SQL & Data Exploration

### execute_sql
Execute a SQL query or DDL statement.

Requirements:

- Support SELECT queries.
- Support DDL/DML where permissions allow.
- Return columns, rows, affected-row information, and execution metadata where available.
- Handle large result sets safely.
- Support configurable result limits.
- Clearly distinguish query results from execution metadata.

### execute_sql_multi
Execute multiple SQL statements.

Requirements:

- Preserve execution order.
- Return success/failure for each statement.
- Provide statement-level errors.
- Avoid silently continuing after failures unless explicitly requested.

### get_table_stats_and_schema
Inspect tables and schemas.

Return information such as:

- Catalog
- Schema
- Table
- Table type
- Columns
- Data types
- Nullable information
- Comments
- Row counts where available
- Table statistics where available
- Partition information
- Location where appropriate

### get_volume_folder_details
Inspect Unity Catalog Volume paths.

Support:

- File listing
- Directory listing
- File type
- Size
- Modification time
- Basic metadata

Recognize common formats such as:

- Parquet
- CSV
- JSON
- Delta

---

# 6. Compute — Clusters & SQL Warehouses

## manage_cluster

Support:

- Create
- Get
- List
- Edit/update
- Start
- Terminate
- Delete

Include validation around destructive operations.

Do not accidentally terminate or delete production resources.

## manage_sql_warehouse

Support:

- Create
- Get
- List
- Edit/update
- Start
- Stop
- Delete

## manage_warehouse

Provide utilities to:

- List warehouses
- Inspect warehouse state
- Select an appropriate running warehouse for SQL execution

The selection logic should be transparent and configurable rather than hidden.

## list_compute

List:

- Clusters
- SQL warehouses
- Node types
- Spark versions
- Runtime information
- Current state

---

# 7. Code & Workspace

## execute_code

Allow execution of:

- Python
- SQL
- Scala
- R

Support execution against appropriate Databricks compute/serverless environments.

Requirements:

- Capture stdout/stderr where available.
- Return execution status.
- Return errors and stack traces safely.
- Support execution timeouts.
- Avoid leaking secrets.
- Clearly identify the compute environment used.

Code execution must be treated as a potentially dangerous operation.

## manage_workspace_files

Support Workspace files/notebooks:

- Upload
- Download where supported
- List
- Get metadata
- Delete
- Create/update notebooks
- Manage paths under `/Workspace`

Prevent accidental deletion through validation/confirmation mechanisms where appropriate.

---

# 8. Jobs & Pipelines

## manage_jobs

Support:

- Create
- Get
- List
- Update
- Delete
- Trigger/run where appropriate

Support common job configuration such as:

- Tasks
- Schedules
- Parameters
- Libraries
- Compute
- Notifications

## manage_job_runs

Support:

- Start
- List
- Get
- Monitor
- Cancel
- Retrieve outputs
- Retrieve task/run status

Long-running operations should expose useful state instead of blocking indefinitely.

---

# 9. Spark Declarative Pipelines

## manage_pipeline

Support:

- Create
- Get
- List
- Update
- Delete

## manage_pipeline_run

Support:

- Start/run
- Get status
- Monitor
- Stop/cancel where supported
- Retrieve run/update information
- Surface pipeline errors

---

# 10. Unity Catalog

## manage_uc_objects

Support management of:

- Catalogs
- Schemas
- Tables where applicable
- Volumes
- Functions

Operations:

- Create
- Get
- List
- Update
- Delete

Respect Unity Catalog hierarchy:

catalog → schema → object

## manage_uc_grants

Manage privileges on:

- Catalogs
- Schemas
- Tables
- Views
- Volumes
- Functions
- Other supported Unity Catalog securables

Support:

- Grant
- Revoke
- Show/list privileges

Never silently broaden permissions.

## manage_uc_storage

Manage:

- Storage credentials
- External locations

Support appropriate CRUD operations.

## manage_uc_connections

Manage Lakehouse Federation connections.

Support connections such as:

- Snowflake
- PostgreSQL
- Other Databricks-supported external systems

Do not expose connection secrets.

## manage_uc_tags

Support:

- Add tags
- Update tags
- Remove tags
- Read tags
- Add comments/descriptions where supported

Tags may include business metadata and PII classification.

## manage_uc_security_policies

Manage:

- Row-level filters
- Column masking
- UDF-based security policies

Make policy changes explicit and auditable.

## manage_uc_monitors

Manage data quality monitors for Delta tables.

Support:

- Create
- Get
- Update
- Delete
- Inspect monitor results/metrics where available

## manage_uc_sharing

Manage Delta Sharing:

- Shares
- Recipients
- Providers

Support:

- Create
- Get
- List
- Update
- Delete

Treat sharing operations as security-sensitive.

---

# 11. Semantic / Metrics Layer

## manage_metric_views

Support management of metric views.

Operations should include, where supported:

- Create
- Get
- List
- Update
- Delete

Return definitions and metadata in structured form.

---

# 12. Volumes & Files

## manage_volume_files

Support Unity Catalog Volume file operations:

- List
- Upload
- Download
- Delete
- Create directory
- Get metadata

Support safe path handling.

Prevent path traversal and access outside the intended Volume.

---

# 13. AI / BI / Dashboards / Agents

## manage_dashboard

Support AI/BI dashboards:

- Create
- Get
- List
- Update
- Delete
- Publish where supported

## manage_serving_endpoint

Support model-serving endpoints:

- List
- Get
- Create/update where supported
- Query/invoke
- Delete where supported

Do not expose credentials or internal authentication details.

## manage_ka

Manage Knowledge Assistants over document volumes.

Support documented lifecycle operations.

## manage_mas

Manage supervisor/orchestrator agents.

Support documented lifecycle and configuration operations.

## manage_genie

Manage Genie Spaces.

Support:

- Create
- Get
- List
- Update
- Delete
- Configure/query where supported

## ask_genie

Allow natural-language questions against a Genie Space.

Return:

- Generated response
- Generated/underlying SQL where available
- Query status
- Relevant metadata
- Errors

Clearly distinguish generated answers from authoritative raw database results.

---

# 14. Vector Search

## manage_vs_endpoint

Manage Vector Search endpoints.

Support appropriate:

- Create
- Get
- List
- Update
- Delete

## manage_vs_index

Manage Vector Search indexes.

Support:

- Create
- Get
- List
- Update
- Delete

## query_vs_index

Query Vector Search indexes.

Return:

- Matching records
- Scores
- Metadata
- Query information

## manage_vs_data

Support:

- Upsert
- Delete
- Inspect
- Sync/index data where supported

---

# 15. Lakebase — PostgreSQL on Databricks

## manage_lakebase_database

Manage provisioned/autoscaling Lakebase databases.

Support:

- Create
- Get
- List
- Update
- Delete

## manage_lakebase_branch

Manage autoscaling branches.

Support lifecycle operations.

## manage_lakebase_sync

Manage reverse ETL/synchronization from Delta to Lakebase.

Support:

- Create
- Get
- List
- Update
- Delete
- Trigger/sync where supported

## generate_lakebase_credential

Generate OAuth credentials/tokens required for database connectivity.

IMPORTANT:

Never expose generated secrets unnecessarily in logs or persisted MCP responses.

---

# 16. Databricks Apps & Identity

## manage_app

Manage Databricks Apps.

Support:

- Create
- Deploy
- Get
- List
- Update
- Delete

## get_current_user

Return:

- Current Databricks user identity
- Username
- User ID where supported
- Home path
- Relevant workspace identity information

Do not expose authentication secrets.

## manage_workspace

Identify/manage the current Databricks workspace context.

Return:

- Workspace URL
- Workspace identifier where available
- Active profile/configuration
- Relevant workspace metadata

---

# 17. Project Manifest

## list_tracked_resources

List resources tracked in the project manifest.

Examples:

- Jobs
- Dashboards
- Pipelines
- Other supported resources

## delete_tracked_resource

Remove a resource from project tracking.

Make it clear that removing tracking is not necessarily the same as deleting the Databricks resource itself.

---

# 18. PDF Generation

## generate_and_upload_pdf

Support:

1. Accepting HTML/content.
2. Converting it to PDF.
3. Uploading the resulting PDF to a specified Unity Catalog Volume.
4. Returning the destination path and metadata.

Validate the destination path and prevent unsafe filesystem operations.

---

# 19. Tool Design Requirements

Every MCP tool should have:

- Clear name
- Clear description
- Strongly typed input schema
- Strongly typed output schema
- Required vs optional parameters
- Validation
- Useful errors
- Appropriate timeout
- Pagination where applicable
- Idempotency considerations
- Safety classification

Categorize operations internally as:

READ_ONLY
WRITE
DESTRUCTIVE
EXECUTION
SECURITY_SENSITIVE

For example:

READ_ONLY:
- get_current_user
- list_compute
- get_table_stats_and_schema

WRITE:
- create catalog
- create job
- upload file

DESTRUCTIVE:
- delete cluster
- delete job
- delete table
- revoke grants

EXECUTION:
- execute_sql
- execute_code
- run_job
- query_serving_endpoint

SECURITY_SENSITIVE:
- manage_uc_grants
- manage_uc_storage
- manage_uc_security_policies
- manage_uc_sharing

---

# 20. Safety Requirements

The MCP server will have access to potentially powerful Databricks operations.

Implement safeguards for:

- DELETE operations
- DROP operations
- TERMINATE operations
- REVOKE operations
- Permission changes
- Production compute changes
- Data modification
- Code execution
- External sharing
- Credential generation

The MCP server must never silently perform destructive operations when the requested action is ambiguous.

Where appropriate, support a `dry_run` mode.

For destructive operations, return a clear summary of what will be changed.

---

# 21. Architecture

Use a modular architecture similar to:

mcp_server/
    server/
        app.py
        config.py
        context.py

    tools/
        sql.py
        compute.py
        workspace.py
        jobs.py
        pipelines.py
        unity_catalog.py
        volumes.py
        dashboards.py
        ai.py
        vector_search.py
        lakebase.py
        apps.py
        manifest.py
        pdf.py

    databricks/
        client.py
        auth.py
        sql_client.py
        workspace_client.py
        unity_catalog_client.py
        jobs_client.py
        compute_client.py
        volume_client.py

    models/
        common.py
        sql.py
        compute.py
        jobs.py
        unity_catalog.py
        volumes.py

    safety/
        permissions.py
        validation.py
        destructive_ops.py

    utils/
        pagination.py
        errors.py
        logging.py

    tests/
        unit/
        integration/

Do not blindly follow this directory structure if a better architecture is technically justified.

---

# 22. Databricks API Integration

Prefer official Databricks SDK/API clients.

Do not implement raw HTTP calls when an official SDK/API abstraction already provides the required functionality, unless there is a documented capability gap.

Centralize Databricks client creation so tools do not independently create authentication/configuration logic.

Implement:

- Retry handling
- Rate-limit handling
- Timeouts
- Pagination
- API error normalization
- Request correlation IDs where possible

---

# 23. Error Handling

Normalize Databricks errors into useful MCP errors.

Examples:

- Authentication failure
- Authorization failure
- Resource not found
- Invalid parameter
- Conflict
- Rate limit
- Timeout
- Databricks service error
- Unsupported operation

Do not return raw stack traces to users unless explicitly configured for development/debug mode.

---

# 24. Logging & Observability

Implement structured logging.

Log:

- Tool name
- Request ID
- User/context
- Databricks operation
- Duration
- Success/failure
- Error category

Do NOT log:

- Access tokens
- Passwords
- OAuth tokens
- Secrets
- Sensitive query results
- Sensitive PII

---

# 25. Testing

Create unit tests for every major capability.

Create integration tests for Databricks-backed operations where practical.

At minimum test:

- Authentication
- SQL execution
- Pagination
- Error handling
- Permission failures
- Resource creation
- Resource deletion
- Job execution
- Unity Catalog operations
- Volume operations
- File upload/download
- MCP schema validation
- Destructive-operation safeguards

Mock Databricks APIs for unit tests.

Integration tests should be configurable and should not run automatically without explicit Databricks credentials/configuration.

---

# 26. Documentation

Produce:

1. README
2. Installation instructions
3. Configuration instructions
4. Authentication setup
5. MCP client configuration examples
6. Complete tool reference
7. Environment variables
8. Security model
9. Development instructions
10. Testing instructions
11. Example MCP tool calls
12. Troubleshooting guide

Include a capability matrix showing:

| Capability | MCP Tool | Read | Write | Destructive | Execution | Security Sensitive |
|------------|----------|------|-------|-------------|-----------|--------------------|

---

# 27. Implementation Strategy

Do not try to implement everything as one giant module.

First establish:

1. MCP server foundation
2. Databricks authentication/client abstraction
3. Error handling
4. Tool registration framework
5. Common models/utilities
6. SQL capabilities
7. Workspace/compute capabilities
8. Jobs/pipelines
9. Unity Catalog
10. Volumes/files
11. AI/BI capabilities
12. Vector Search
13. Lakebase
14. Apps/manifest/PDF
15. Testing and documentation

Prioritize correctness and maintainability over implementing every tool superficially.

If a Databricks capability is not supported by the currently available official API/SDK, explicitly mark it as unsupported rather than creating a fake implementation.

---

# 28. Important Engineering Rule

Do not hallucinate Databricks APIs, SDK methods, endpoints, request fields, or response fields.

Before implementing a capability:

- Verify the official Databricks API/SDK supports it.
- Use the correct API version.
- Confirm request/response schemas.
- Clearly document any limitations.

If an operation cannot be implemented reliably, say so and isolate it behind an explicit TODO/interface rather than inventing an API.

---

# 29. Expected Deliverable

Produce a complete working Databricks MCP Server with:

- Production-quality source code
- MCP tool definitions
- Databricks client abstraction
- Authentication/configuration
- Validation
- Safety controls
- Error handling
- Logging
- Tests
- Documentation
- Example configuration

The final implementation should be independently developed and must NOT depend on or copy the Databricks AI Dev Kit repository mentioned above.

Before writing substantial code, inspect the existing project structure and identify what already exists. Reuse existing project conventions where appropriate, but do not compromise the architecture or safety requirements above.