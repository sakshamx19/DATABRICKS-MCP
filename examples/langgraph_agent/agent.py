"""LangGraph agent that uses the dbx-mcp Databricks MCP server as its tool provider.

The LLM is an Azure AI Foundry (OpenAI v1-compatible endpoint) deployment; the tools come
from `dbx-mcp`, spawned as a stdio subprocess through langchain-mcp-adapters.

Safety: the server already refuses destructive/security-sensitive changes unless the call
carries `confirm=true`. This agent adds a human gate on top: any tool call with
`confirm=true` is shown in the terminal and runs only if YOU type "yes" - the model can never
approve a destructive change by itself.

Run (from the repo root):
    examples/langgraph_agent/.venv/Scripts/python examples/langgraph_agent/agent.py
    ... agent.py --read-only                      # server refuses every change
    ... agent.py --toolsets identity,sql,unity_catalog
    ... agent.py -q "Which SQL warehouses are running?"   # one question, then exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from mcp.types import CallToolResult, TextContent

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

SYSTEM_PROMPT = """\
You are a Databricks assistant. You act on the user's Databricks workspace through the
`dbx-mcp` tools.

How the tools behave:
- Every tool returns JSON with `status`, `summary`, `data`, `warnings`, `next_steps`.
- `status: "confirmation_required"` means NOTHING was changed. Show the user the `plan`
  (what will change, warnings, whether it is reversible) and ask whether to proceed. Only
  after they agree, call the same tool again with the same arguments plus `confirm: true`.
- `status: "pending"` means a long-running operation was started; tell the user and poll
  with the matching get/status action if they want to wait.
- Use `dry_run: true` when the user wants to preview a change.
- Lists are paginated; pass `next_page_token` back as `page_token` for more.
- Genie answers are model-generated; say so and show the SQL it used.

Prefer reading/inspecting before changing anything. Be concise; use tables for tabular data.
"""


def build_llm() -> ChatOpenAI:
    """Azure AI Foundry deployment via its OpenAI v1-compatible endpoint (no api-version needed)."""
    missing = [k for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL_NAME") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)} (set them in .env)")
    return ChatOpenAI(
        model=os.environ["LLM_MODEL_NAME"],  # the Foundry deployment name, e.g. gpt-5.1
        base_url=os.environ["OPENAI_BASE_URL"],  # https://<resource>.services.ai.azure.com/openai/v1/
        api_key=os.environ["OPENAI_API_KEY"],
        use_responses_api=True,
        timeout=120,
        max_retries=2,
    )


def server_command() -> str:
    """Path to the dbx-mcp executable in the repo's own virtualenv (override with DBX_MCP_COMMAND)."""
    if os.environ.get("DBX_MCP_COMMAND"):
        return os.environ["DBX_MCP_COMMAND"]
    exe = "dbx-mcp.exe" if os.name == "nt" else "dbx-mcp"
    candidate = REPO_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / exe
    if not candidate.exists():
        sys.exit(f"dbx-mcp not found at {candidate}. Install the server (pip install -e .) or set DBX_MCP_COMMAND.")
    return str(candidate)


def mcp_connections(env_file: Path, read_only: bool, toolsets: str | None) -> dict:
    server_env = {
        # Inherit the minimal environment the server needs (PATH, user profile dirs on Windows, ...).
        **{k: v for k, v in os.environ.items() if k.upper() in {
            "PATH", "SYSTEMROOT", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "APPDATA",
            "LOCALAPPDATA", "TEMP", "TMP", "PROGRAMDATA",
        }},
        "DBX_MCP_LOG_LEVEL": "WARNING",
    }
    if read_only:
        server_env["DBX_MCP_READ_ONLY"] = "true"
    if toolsets:
        server_env["DBX_MCP_TOOLSETS"] = toolsets
    return {
        "databricks": {
            "transport": "stdio",
            "command": server_command(),
            # Databricks credentials are read by the server from the env file; they are never
            # passed to (or visible to) the LLM.
            "args": ["--env-file", str(env_file)],
            "env": server_env,
        }
    }


async def human_approval_gate(request: MCPToolCallRequest, handler):
    """Interceptor: any confirmed (destructive/security-sensitive) call needs a human 'yes'."""
    if request.args.get("confirm") is True:
        print("\n" + "=" * 70)
        print(f"APPROVAL NEEDED: {request.name}")
        print(json.dumps({k: v for k, v in request.args.items() if k != "confirm"}, indent=2, default=str))
        print("=" * 70)
        answer = await asyncio.to_thread(input, "Type 'yes' to execute, anything else to cancel: ")
        if answer.strip().lower() != "yes":
            return CallToolResult(
                content=[TextContent(type="text", text="The user DENIED this action. It was not executed. "
                                                       "Do not retry it unless the user asks again.")],
                isError=True,
            )
    return await handler(request)


async def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file, override=False)
    client = MultiServerMCPClient(
        mcp_connections(args.env_file, args.read_only, args.toolsets),
        tool_interceptors=[human_approval_gate],
    )
    tools = await client.get_tools()
    print(f"Connected to dbx-mcp: {len(tools)} tools loaded"
          + (" (read-only mode)" if args.read_only else "") + ".")

    agent = create_agent(
        model=build_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # keeps the conversation across turns
    )
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 40}

    async def ask(question: str) -> None:
        async for step in agent.astream({"messages": [{"role": "user", "content": question}]},
                                        config=config, stream_mode="updates"):
            for node, update in step.items():
                for msg in (update or {}).get("messages", []):
                    if node == "model" and getattr(msg, "tool_calls", None):
                        for call in msg.tool_calls:
                            print(f"  -> {call['name']}({json.dumps(call['args'], default=str)[:160]})")
                    elif node == "model" and msg.text:
                        print(f"\nAssistant: {msg.text}\n")

    if args.question:
        await ask(args.question)
        return
    print("Ask about your Databricks workspace (type 'exit' to quit).\n")
    while True:
        try:
            question = (await asyncio.to_thread(input, "You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            await ask(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agent over the dbx-mcp Databricks MCP server")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--read-only", action="store_true", help="Start the server in read-only mode")
    parser.add_argument("--toolsets", help="Comma-separated toolsets to enable, e.g. identity,sql,compute")
    parser.add_argument("-q", "--question", help="Ask one question and exit")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
