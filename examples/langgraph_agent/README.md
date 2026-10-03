# LangGraph agent over dbx-mcp

A terminal chat agent built with LangGraph (`langchain.agents.create_agent`). It uses the
`dbx-mcp` server as its toolbox, connected over stdio through `langchain-mcp-adapters`. The LLM
is an Azure AI Foundry deployment reached through Foundry's OpenAI v1-compatible endpoint.

## Setup

The agent gets its own virtualenv, separate from the server: `langchain-mcp-adapters` pins
`mcp<2`, while the server uses `mcp` 2.x. The two only talk over stdio, so the versions never
conflict.

```bash
# from the repo root; the server itself must already be installed in ./.venv
uv venv examples/langgraph_agent/.venv
uv pip install --python examples/langgraph_agent/.venv/Scripts/python.exe -r examples/langgraph_agent/requirements.txt
```

`.env` (repo root) needs these variables. The Databricks credentials are read only by the
server; the LLM never sees them.

```
DATABRICKS_HOST=...
DATABRICKS_TOKEN=...
OPENAI_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1/
OPENAI_API_KEY=...
LLM_MODEL_NAME=gpt-5.1          # your Foundry deployment name
```

## Run

```bash
python examples/langgraph_agent/agent.py                  # interactive chat
python examples/langgraph_agent/agent.py --read-only      # server refuses all changes
python examples/langgraph_agent/agent.py --toolsets identity,sql,compute,unity_catalog
python examples/langgraph_agent/agent.py -q "List my running clusters"
```

Use the agent venv's Python (`examples/langgraph_agent/.venv/Scripts/python.exe`).
`DBX_MCP_COMMAND` overrides the path to the `dbx-mcp` executable.

## Safety

There are two independent gates on destructive and security-sensitive changes:

1. **The server** returns `confirmation_required` with a plan and changes nothing until the
   call is repeated with `confirm=true`.
2. **This agent** intercepts every call that has `confirm=true` and prints it. It runs only if
   you type `yes` in the terminal, so the model cannot approve a change by itself.
