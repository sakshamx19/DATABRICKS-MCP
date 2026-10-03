"""Command-line entry point: ``dbx-mcp`` / ``python -m dbx_mcp``."""

from __future__ import annotations

import argparse
import sys

from dbx_mcp import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dbx-mcp", description="Databricks MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP transports")
    parser.add_argument("--port", type=int, default=8765, help="Bind port for HTTP transports")
    parser.add_argument("--env-file", help="Load environment variables from this .env file first")
    parser.add_argument("--read-only", action="store_true", help="Shortcut for DBX_MCP_READ_ONLY=true")
    parser.add_argument(
        "--auth-mode",
        choices=["env", "request"],
        help="env: use the server's own Databricks credentials (default). request: every HTTP request "
        "sends its own workspace URL + PAT in headers; the server stores no credentials. "
        "Shortcut for DBX_MCP_AUTH_MODE.",
    )
    parser.add_argument("--list-tools", action="store_true", help="Print enabled tools and exit")
    parser.add_argument("--version", action="version", version=f"dbx-mcp {__version__}")
    args = parser.parse_args(argv)

    if args.env_file:
        from dotenv import load_dotenv

        if not load_dotenv(args.env_file, override=False):
            print(f"warning: could not load env file {args.env_file}", file=sys.stderr)
    import os

    if args.read_only:
        os.environ["DBX_MCP_READ_ONLY"] = "true"
    if args.auth_mode:
        os.environ["DBX_MCP_AUTH_MODE"] = args.auth_mode

    from dbx_mcp.server.app import build_server
    from dbx_mcp.server.config import Settings
    from dbx_mcp.utils.logging import configure_logging

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if settings.auth_mode == "request" and args.transport == "stdio" and not args.list_tools:
        print(
            "configuration error: --auth-mode request needs an HTTP transport (credentials arrive in "
            "request headers). Use --transport streamable-http, or --auth-mode env for stdio.",
            file=sys.stderr,
        )
        return 2
    configure_logging(settings.log_level)
    server, enabled = build_server(settings)

    if args.list_tools:
        for spec in enabled:
            print(f"{spec.toolset:<14} {spec.name:<34} {spec.safety_summary()}")
        return 0

    if args.transport == "stdio":
        server.run("stdio")
    else:
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            if settings.auth_mode == "request":
                print(
                    "note: request-auth mode - clients send their own Databricks PAT in headers. Serve "
                    "over HTTPS (TLS-terminating reverse proxy) so tokens are not sent in clear text.",
                    file=sys.stderr,
                )
            else:
                print(
                    "warning: binding to a non-loopback address exposes your Databricks credentials' "
                    "capabilities to the network. Put an authenticating proxy in front of it, or use "
                    "--auth-mode request so each client brings its own credentials.",
                    file=sys.stderr,
                )
        server.run(args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
