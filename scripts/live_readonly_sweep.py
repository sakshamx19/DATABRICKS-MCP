"""Read-only live sweep: call list/get actions of every toolset against a real workspace.

Usage: python scripts/live_readonly_sweep.py --env-file .env
The server is forced into read-only mode; nothing that executes SQL, code or compute is called.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import replace


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    from dbx_mcp.server.app import build_server
    from dbx_mcp.server.config import Settings

    settings = replace(Settings.from_env(), read_only=True,
                       manifest_path=__import__("pathlib").Path(tempfile.mkdtemp()) / "m.json")
    server, _ = build_server(settings)

    results: list[tuple[str, str, str]] = []

    async def call(name: str, arguments: dict) -> dict | None:
        label = f"{name} {json.dumps(arguments)[:90]}"
        try:
            r = await server.call_tool(name, arguments)
            if r.is_error:
                results.append(("ERR", label, r.content[0].text[:220]))
                return None
            out = r.structured_content
            results.append(("OK ", label, out["summary"][:120]))
            return out
        except Exception as exc:  # noqa: BLE001
            results.append(("ERR", label, str(exc)[:220]))
            return None

    me = await call("get_current_user", {})
    await call("manage_workspace", {"action": "info"})
    await call("list_compute", {})
    await call("list_compute", {"resource": "spark_versions", "filter": "LTS", "page_size": 3})
    await call("manage_warehouse", {"action": "list"})
    await call("manage_cluster", {"action": "list", "page_size": 5})
    await call("manage_sql_warehouse", {"action": "list", "page_size": 5})
    if me:
        home = me["data"]["home_path"]
        await call("manage_workspace_files", {"action": "list", "path": home, "page_size": 5})
    await call("manage_jobs", {"action": "list", "page_size": 5})
    await call("manage_job_runs", {"action": "list", "page_size": 5})
    await call("manage_pipeline", {"action": "list", "page_size": 5})

    cats = await call("manage_uc_objects", {"action": "list", "object_type": "catalog", "page_size": 20})
    catalog = None
    if cats and cats["data"]:
        names = [c.get("name") for c in cats["data"]]
        catalog = next((n for n in names if n not in ("system", "samples", "hive_metastore")), names[0])
    if catalog:
        await call("manage_uc_objects", {"action": "get", "object_type": "catalog", "full_name": catalog})
        await call("manage_uc_grants", {"action": "get", "securable_type": "catalog", "full_name": catalog})
        await call("manage_uc_grants", {"action": "get_effective", "securable_type": "catalog", "full_name": catalog})
        await call("manage_uc_tags", {"action": "get", "entity_type": "catalog", "name": catalog})
        schemas = await call("manage_uc_objects", {"action": "list", "object_type": "schema",
                                                     "catalog_name": catalog, "page_size": 20})
        schema = next((s["name"] for s in (schemas or {}).get("data", []) if s.get("name") != "information_schema"), None)
        if schema:
            await call("manage_uc_grants", {"action": "get", "securable_type": "schema",
                                            "full_name": f"{catalog}.{schema}"})
            tables = await call("manage_uc_objects", {"action": "list", "object_type": "table",
                                                      "catalog_name": catalog, "schema_name": schema, "page_size": 5})
            await call("manage_uc_objects", {"action": "list", "object_type": "volume",
                                             "catalog_name": catalog, "schema_name": schema, "page_size": 5})
            await call("get_table_stats_and_schema", {"name": f"{catalog}.{schema}", "stats": "none", "page_size": 5})
            if tables and tables["data"]:
                t = tables["data"][0]
                full = t.get("full_name") or f"{catalog}.{schema}.{t['name']}"
                await call("get_table_stats_and_schema", {"name": full, "stats": "none"})
                await call("manage_uc_grants", {"action": "get", "securable_type": "table", "full_name": full})
                await call("manage_uc_tags", {"action": "get", "entity_type": "table", "name": full})
                await call("manage_uc_security_policies", {"action": "get", "table_name": full})
            await call("manage_metric_views", {"action": "list", "catalog_name": catalog, "schema_name": schema})
    await call("manage_uc_storage", {"action": "list", "resource": "storage_credential", "page_size": 5})
    await call("manage_uc_storage", {"action": "list", "resource": "external_location", "page_size": 5})
    await call("manage_uc_connections", {"action": "list", "page_size": 5})
    await call("manage_uc_sharing", {"action": "list", "resource": "share", "page_size": 5})
    await call("manage_uc_sharing", {"action": "list", "resource": "recipient", "page_size": 5})
    await call("manage_uc_sharing", {"action": "list", "resource": "provider", "page_size": 5})
    await call("manage_dashboard", {"action": "list", "page_size": 5})
    await call("manage_serving_endpoint", {"action": "list", "page_size": 5})
    await call("manage_genie", {"action": "list", "page_size": 5})
    await call("manage_ka", {"action": "list", "page_size": 5})
    await call("manage_mas", {"action": "list", "page_size": 5})
    await call("manage_vs_endpoint", {"action": "list", "page_size": 5})
    await call("manage_lakebase_database", {"action": "list", "kind": "provisioned", "page_size": 5})
    await call("manage_lakebase_database", {"action": "list", "kind": "autoscaling", "page_size": 5})
    await call("manage_app", {"action": "list", "page_size": 5})
    await call("list_tracked_resources", {})
    # safety: a write must be refused in read-only mode
    await call("manage_jobs", {"action": "delete", "job_id": 1, "confirm": True})

    for status, label, msg in results:
        print(f"{status} | {label}\n      -> {msg}")
    errors = sum(1 for r in results if r[0] == "ERR")
    print(f"\n{len(results) - errors} ok, {errors} error(s)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
