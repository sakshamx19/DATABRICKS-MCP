"""Tests for the 'ai' toolset: serving endpoints, Knowledge Assistants, Supervisor Agents."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.common.types.fieldmask import FieldMask
from databricks.sdk.errors import NotFound
from databricks.sdk.service._internal import Wait
from databricks.sdk.service.knowledgeassistants import (
    KnowledgeAssistant,
    KnowledgeAssistantState,
    KnowledgeSource,
)
from databricks.sdk.service.serving import (
    BuildLogsResponse,
    ChatMessage,
    ChatMessageRole,
    EndpointCoreConfigOutput,
    EndpointState,
    EndpointStateConfigUpdate,
    EndpointStateReady,
    EndpointTag,
    ExternalModel,
    ExternalModelProvider,
    OpenAiConfig,
    QueryEndpointResponse,
    ServedEntityOutput,
    ServingEndpoint,
    ServingEndpointDetailed,
    V1ResponseChoiceElement,
)
from databricks.sdk.service.supervisoragents import SupervisorAgent, Tool

TOOLSETS = ("ai",)


@pytest.fixture
def ai(make_harness):
    def factory(**overrides):
        h = make_harness(toolsets=TOOLSETS, **overrides)
        return h

    return factory


def _endpoint(name="chat", update=EndpointStateConfigUpdate.NOT_UPDATING, ready=EndpointStateReady.READY, tags=None):
    return ServingEndpointDetailed(
        name=name,
        state=EndpointState(config_update=update, ready=ready),
        tags=tags,
        config=EndpointCoreConfigOutput(
            config_version=3,
            served_entities=[
                ServedEntityOutput(
                    name="gpt",
                    environment_vars={"PLAIN": "hunter2-value", "REF": "{{secrets/scope/key}}"},
                    external_model=ExternalModel(
                        provider=ExternalModelProvider.OPENAI,
                        name="gpt-4o",
                        task="llm/v1/chat",
                        openai_config=OpenAiConfig(
                            openai_api_key="{{secrets/scope/openai}}",
                            openai_api_key_plaintext="sk-live-very-secret-123",
                            microsoft_entra_client_secret_plaintext="entra-secret-xyz",
                            openai_api_base="https://example.openai.azure.com",
                        ),
                    ),
                )
            ],
        ),
    )


# ------------------------------------------------------------------------------------------
# manage_serving_endpoint
# ------------------------------------------------------------------------------------------


async def test_serving_list_paginates(ai):
    h = ai()
    h.w.serving_endpoints.list.return_value = [ServingEndpoint(name=f"ep{i}") for i in range(5)]
    first = await h.call("manage_serving_endpoint", {"action": "list", "page_size": 2})
    assert [e["name"] for e in first["data"]] == ["ep0", "ep1"]
    assert first["page"]["has_more"] is True
    second = await h.call(
        "manage_serving_endpoint", {"action": "list", "page_size": 2, "page_token": first["page"]["next_page_token"]}
    )
    assert [e["name"] for e in second["data"]] == ["ep2", "ep3"]


async def test_serving_get_strips_external_model_credentials(ai):
    h = ai()
    h.w.serving_endpoints.get.return_value = _endpoint()
    result = await h.call("manage_serving_endpoint", {"action": "get", "name": "chat"})
    text = json.dumps(result)
    assert "sk-live-very-secret-123" not in text
    assert "entra-secret-xyz" not in text
    assert "hunter2-value" not in text
    entity = result["data"]["config"]["served_entities"][0]
    cfg = entity["external_model"]["openai_config"]
    assert cfg["openai_api_key"] == "***REDACTED***"
    assert cfg["openai_api_key_plaintext"] == "***REDACTED***"
    assert cfg["openai_api_base"] == "https://example.openai.azure.com"
    assert entity["environment_vars"] == {"PLAIN": "***REDACTED***", "REF": "{{secrets/scope/key}}"}


async def test_serving_list_summary_has_no_credentials(ai):
    h = ai()
    h.w.serving_endpoints.list.return_value = [_endpoint()]
    result = await h.call("manage_serving_endpoint", {"action": "list"})
    assert "sk-live" not in json.dumps(result)
    assert result["data"][0]["served_entities"][0]["external_model"]["provider"] == "openai"


async def test_serving_create_rejects_unknown_field(ai):
    h = ai()
    msg = await h.call_error("manage_serving_endpoint", {"action": "create", "name": "x", "spec": {"bogus": 1}})
    assert "Unknown field" in msg and "bogus" in msg
    h.w.serving_endpoints.create.assert_not_called()


async def test_serving_create_returns_pending_and_tracks(ai):
    h = ai()
    pending = _endpoint(name="new-ep", update=EndpointStateConfigUpdate.IN_PROGRESS, ready=EndpointStateReady.NOT_READY)
    h.w.serving_endpoints.create.return_value = Wait(lambda **kw: None, response=pending, name="new-ep")
    result = await h.call(
        "manage_serving_endpoint",
        {"action": "create", "name": "new-ep", "spec": {"description": "demo"}},
    )
    assert result["status"] == "pending"
    assert any("action='get'" in s for s in result["next_steps"])
    h.w.serving_endpoints.create.assert_called_once_with(name="new-ep", description="demo")
    manifest = json.loads(h.settings.manifest_path.read_text())
    assert manifest["resources"][0]["resource_type"] == "serving_endpoint"
    assert manifest["resources"][0]["resource_id"] == "new-ep"
    assert "sk-live" not in json.dumps(result)


async def test_serving_create_bounded_wait_reaches_ready(ai):
    h = ai()
    pending = _endpoint(name="new-ep", update=EndpointStateConfigUpdate.IN_PROGRESS, ready=EndpointStateReady.NOT_READY)
    h.w.serving_endpoints.create.return_value = Wait(lambda **kw: None, response=pending, name="new-ep")
    h.w.serving_endpoints.get.return_value = _endpoint(name="new-ep")
    result = await h.call("manage_serving_endpoint", {"action": "create", "name": "new-ep", "wait_seconds": 30})
    assert result["status"] == "success"
    assert "READY" in result["summary"]


async def test_serving_delete_requires_confirmation(ai):
    h = ai()
    h.w.serving_endpoints.get.return_value = _endpoint(name="dev-chat")
    result = await h.call("manage_serving_endpoint", {"action": "delete", "name": "dev-chat"})
    assert result["status"] == "confirmation_required"
    assert result["plan"]["reversible"] is False
    h.w.serving_endpoints.delete.assert_not_called()

    result = await h.call("manage_serving_endpoint", {"action": "delete", "name": "dev-chat", "confirm": True})
    assert result["status"] == "success"
    h.w.serving_endpoints.delete.assert_called_once_with(name="dev-chat")


async def test_serving_delete_dry_run(ai):
    h = ai()
    h.w.serving_endpoints.get.return_value = _endpoint(name="dev-chat")
    result = await h.call("manage_serving_endpoint", {"action": "delete", "name": "dev-chat", "dry_run": True, "confirm": True})
    assert result["status"] == "dry_run"
    h.w.serving_endpoints.delete.assert_not_called()


async def test_serving_delete_protected_endpoint_blocked(ai):
    h = ai()
    h.w.serving_endpoints.get.return_value = _endpoint(name="chat", tags=[EndpointTag(key="env", value="prod")])
    msg = await h.call_error("manage_serving_endpoint", {"action": "delete", "name": "chat", "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    h.w.serving_endpoints.delete.assert_not_called()


async def test_serving_query_chat(ai):
    h = ai()
    h.w.serving_endpoints.query.return_value = QueryEndpointResponse(
        choices=[V1ResponseChoiceElement(index=0, message=ChatMessage(role=ChatMessageRole.ASSISTANT, content="Hello there"))]
    )
    result = await h.call(
        "manage_serving_endpoint",
        {"action": "query", "name": "chat", "request": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}},
    )
    assert result["status"] == "success"
    assert result["data"]["text"] == "Hello there"
    assert result["data"]["truncated"] is False
    kwargs = h.w.serving_endpoints.query.call_args.kwargs
    assert kwargs["name"] == "chat" and kwargs["max_tokens"] == 20


async def test_serving_query_rejects_stream_and_unknown(ai):
    h = ai()
    msg = await h.call_error("manage_serving_endpoint", {"action": "query", "name": "chat", "request": {"stream": True}})
    assert "Unknown field" in msg
    h.w.serving_endpoints.query.assert_not_called()


async def test_serving_query_output_capped(ai):
    h = ai()
    h.w.serving_endpoints.query.return_value = QueryEndpointResponse(predictions=[list(range(50)) for _ in range(200)])
    result = await h.call(
        "manage_serving_endpoint",
        {"action": "query", "name": "m", "request": {"dataframe_records": [{"a": 1}]}, "max_output_chars": 1000},
    )
    assert result["data"]["truncated"] is True
    assert len(result["data"]["response_preview"]) == 1000
    assert result["warnings"]


async def test_serving_query_blocked_in_read_only(ai):
    h = ai(read_only=True)
    msg = await h.call_error("manage_serving_endpoint", {"action": "query", "name": "m", "request": {"input": "x"}})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg
    # reads still work
    h.w.serving_endpoints.list.return_value = []
    assert (await h.call("manage_serving_endpoint", {"action": "list"}))["status"] == "success"


async def test_serving_build_logs_tail(ai):
    h = ai()
    h.w.serving_endpoints.build_logs.return_value = BuildLogsResponse(logs="x" * 5000 + "END")
    result = await h.call(
        "manage_serving_endpoint",
        {"action": "get_build_logs", "name": "m", "served_model_name": "m-1", "max_output_chars": 1000},
    )
    assert result["data"]["truncated"] is True
    assert result["data"]["logs"].endswith("END") and len(result["data"]["logs"]) == 1000


async def test_serving_not_found(ai):
    h = ai()
    h.w.serving_endpoints.get.side_effect = NotFound("Endpoint missing does not exist")
    msg = await h.call_error("manage_serving_endpoint", {"action": "get", "name": "missing"})
    assert "[NOT_FOUND]" in msg


async def test_serving_requires_name(ai):
    h = ai()
    msg = await h.call_error("manage_serving_endpoint", {"action": "get"})
    assert "INVALID_PARAMETER" in msg and "name" in msg


# ------------------------------------------------------------------------------------------
# manage_ka
# ------------------------------------------------------------------------------------------


async def test_ka_list(ai):
    h = ai()
    h.w.knowledge_assistants.list_knowledge_assistants.return_value = [
        KnowledgeAssistant(display_name=f"KA {i}", description="d", name=f"knowledge-assistants/id{i}") for i in range(3)
    ]
    result = await h.call("manage_ka", {"action": "list", "page_size": 2})
    assert [k["knowledge_assistant_id"] for k in result["data"]] == ["id0", "id1"]
    assert result["page"]["has_more"] is True


async def test_ka_create_tracks_and_is_pending(ai):
    h = ai()
    h.w.knowledge_assistants.create_knowledge_assistant.return_value = KnowledgeAssistant(
        display_name="HR docs", description="Answers HR questions", name="knowledge-assistants/abc",
        state=KnowledgeAssistantState.CREATING,
    )
    result = await h.call(
        "manage_ka",
        {"action": "create", "spec": {"display_name": "HR docs", "description": "Answers HR questions", "instructions": "Be brief"}},
    )
    assert result["status"] == "pending"
    sent = h.w.knowledge_assistants.create_knowledge_assistant.call_args.kwargs["knowledge_assistant"]
    assert isinstance(sent, KnowledgeAssistant) and sent.instructions == "Be brief"
    manifest = json.loads(h.settings.manifest_path.read_text())
    assert manifest["resources"][0]["resource_type"] == "knowledge_assistant"
    assert manifest["resources"][0]["resource_id"] == "abc"


async def test_ka_create_validation(ai):
    h = ai()
    msg = await h.call_error("manage_ka", {"action": "create", "spec": {"display_name": "x", "description": "y", "colour": "red"}})
    assert "colour" in msg
    msg = await h.call_error("manage_ka", {"action": "create", "spec": {"display_name": "x"}})
    assert "description" in msg
    h.w.knowledge_assistants.create_knowledge_assistant.assert_not_called()


async def test_ka_update_derives_mask(ai):
    h = ai()
    h.w.knowledge_assistants.update_knowledge_assistant.return_value = KnowledgeAssistant(
        display_name="HR", description="d", name="knowledge-assistants/abc"
    )
    await h.call("manage_ka", {"action": "update", "knowledge_assistant_id": "abc", "spec": {"instructions": "Cite sources"}})
    kwargs = h.w.knowledge_assistants.update_knowledge_assistant.call_args.kwargs
    assert kwargs["name"] == "knowledge-assistants/abc"
    assert kwargs["update_mask"] == FieldMask(["instructions"])
    assert kwargs["knowledge_assistant"].instructions == "Cite sources"

    msg = await h.call_error("manage_ka", {"action": "update", "knowledge_assistant_id": "abc", "spec": {"endpoint_name": "x"}})
    assert "cannot be updated" in msg


async def test_ka_add_files_source(ai):
    h = ai()
    h.w.knowledge_assistants.create_knowledge_source.return_value = KnowledgeSource(
        display_name="Policies", description="PDFs", source_type="files", name="knowledge-assistants/abc/knowledge-sources/s1"
    )
    spec = {"display_name": "Policies", "description": "PDFs", "source_type": "files", "files": {"path": "/Volumes/hr/docs/policies"}}
    result = await h.call("manage_ka", {"action": "add_source", "knowledge_assistant_id": "abc", "spec": spec})
    assert result["status"] == "success"
    kwargs = h.w.knowledge_assistants.create_knowledge_source.call_args.kwargs
    assert kwargs["parent"] == "knowledge-assistants/abc"
    assert kwargs["knowledge_source"].files.path == "/Volumes/hr/docs/policies"

    msg = await h.call_error(
        "manage_ka",
        {"action": "add_source", "knowledge_assistant_id": "abc", "spec": {**spec, "source_type": "files", "files": None}},
    )
    assert "requires spec.files" in msg


async def test_ka_delete_confirm_flow(ai):
    h = ai()
    h.w.knowledge_assistants.get_knowledge_assistant.return_value = KnowledgeAssistant(
        display_name="HR docs", description="d", name="knowledge-assistants/abc"
    )
    result = await h.call("manage_ka", {"action": "delete", "knowledge_assistant_id": "abc"})
    assert result["status"] == "confirmation_required"
    h.w.knowledge_assistants.delete_knowledge_assistant.assert_not_called()
    result = await h.call("manage_ka", {"action": "delete", "knowledge_assistant_id": "abc", "confirm": True})
    assert result["status"] == "success"
    h.w.knowledge_assistants.delete_knowledge_assistant.assert_called_once_with(name="knowledge-assistants/abc")


async def test_ka_sync_sources(ai):
    h = ai()
    result = await h.call("manage_ka", {"action": "sync_sources", "knowledge_assistant_id": "knowledge-assistants/abc"})
    assert result["status"] == "pending"
    h.w.knowledge_assistants.sync_knowledge_sources.assert_called_once_with(name="knowledge-assistants/abc")


async def test_ka_writes_blocked_in_read_only(ai):
    h = ai(read_only=True)
    msg = await h.call_error("manage_ka", {"action": "create", "spec": {"display_name": "x", "description": "y"}})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg


# ------------------------------------------------------------------------------------------
# manage_mas
# ------------------------------------------------------------------------------------------


async def test_mas_create_and_add_genie_tool(ai):
    h = ai()
    h.w.supervisor_agents.create_supervisor_agent.return_value = SupervisorAgent(
        display_name="Ops", name="supervisor-agents/sa1", supervisor_agent_id="sa1"
    )
    result = await h.call("manage_mas", {"action": "create", "spec": {"display_name": "Ops", "description": "routes"}})
    assert result["status"] == "success"
    assert json.loads(h.settings.manifest_path.read_text())["resources"][0]["resource_type"] == "supervisor_agent"

    h.w.supervisor_agents.create_tool.return_value = Tool(tool_type="genie_space", name="supervisor-agents/sa1/tools/sales")
    await h.call(
        "manage_mas",
        {
            "action": "add_tool",
            "supervisor_agent_id": "sa1",
            "tool_id": "sales",
            "spec": {"tool_type": "genie_space", "genie_space": {"id": "g1"}, "description": "Sales data"},
        },
    )
    kwargs = h.w.supervisor_agents.create_tool.call_args.kwargs
    assert kwargs["parent"] == "supervisor-agents/sa1" and kwargs["tool_id"] == "sales"
    assert kwargs["tool"].genie_space.id == "g1"


async def test_mas_unsupported_tool_type(ai):
    h = ai()
    msg = await h.call_error(
        "manage_mas",
        {"action": "add_tool", "supervisor_agent_id": "sa1", "tool_id": "t", "spec": {"tool_type": "vector_search_index"}},
    )
    assert "UNSUPPORTED_OPERATION" in msg
    h.w.supervisor_agents.create_tool.assert_not_called()


async def test_mas_update_tool_only_description(ai):
    h = ai()
    msg = await h.call_error(
        "manage_mas",
        {"action": "update_tool", "supervisor_agent_id": "sa1", "tool_id": "t", "spec": {"tool_type": "app"}},
    )
    assert "cannot be updated" in msg


async def test_mas_delete_tool_requires_confirm(ai):
    h = ai()
    h.w.supervisor_agents.get_supervisor_agent.return_value = SupervisorAgent(display_name="Ops", name="supervisor-agents/sa1")
    h.w.supervisor_agents.get_tool.return_value = Tool(tool_type="app", name="supervisor-agents/sa1/tools/t")
    result = await h.call("manage_mas", {"action": "delete_tool", "supervisor_agent_id": "sa1", "tool_id": "t"})
    assert result["status"] == "confirmation_required"
    h.w.supervisor_agents.delete_tool.assert_not_called()
    await h.call("manage_mas", {"action": "delete_tool", "supervisor_agent_id": "sa1", "tool_id": "t", "confirm": True})
    h.w.supervisor_agents.delete_tool.assert_called_once_with(name="supervisor-agents/sa1/tools/t")


async def test_mas_get_not_found(ai):
    h = ai()
    h.w.supervisor_agents.get_supervisor_agent.side_effect = NotFound("no such agent")
    msg = await h.call_error("manage_mas", {"action": "get", "supervisor_agent_id": "nope"})
    assert "[NOT_FOUND]" in msg
