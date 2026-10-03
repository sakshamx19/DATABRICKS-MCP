"""Tests for manage_genie and ask_genie."""

from __future__ import annotations

import json

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service import sql
from databricks.sdk.service._internal import Wait
from databricks.sdk.service.dashboards import (
    GenieAttachment,
    GenieGetMessageQueryResultResponse,
    GenieListSpacesResponse,
    GenieMessage,
    GenieQueryAttachment,
    GenieSpace,
    GenieStartConversationResponse,
    MessageError,
    MessageStatus,
    TextAttachment,
)

TOOLSETS = ("ai",)


@pytest.fixture
def genie(make_harness):
    def factory(**overrides):
        h = make_harness(toolsets=TOOLSETS, **overrides)
        return h

    return factory


def _message(status=MessageStatus.COMPLETED, attachments=None, error=None):
    return GenieMessage(
        space_id="sp1",
        conversation_id="c1",
        message_id="m1",
        content="How many orders per region?",
        status=status,
        attachments=attachments,
        error=error,
    )


def _completed_with_query():
    return _message(
        attachments=[
            GenieAttachment(
                attachment_id="a1",
                query=GenieQueryAttachment(
                    title="Orders per region",
                    description="Counts orders grouped by region",
                    query="SELECT region, count(*) AS n FROM orders GROUP BY region",
                    statement_id="st1",
                ),
            ),
            GenieAttachment(attachment_id="a2", text=TextAttachment(content="EU has 3 orders and US has 5.")),
        ]
    )


def _query_result(rows):
    return GenieGetMessageQueryResultResponse(
        statement_response=sql.StatementResponse(
            statement_id="st1",
            status=sql.StatementStatus(state=sql.StatementState.SUCCEEDED),
            manifest=sql.ResultManifest(
                schema=sql.ResultSchema(columns=[sql.ColumnInfo(name="region"), sql.ColumnInfo(name="n")]),
                total_row_count=len(rows),
            ),
            result=sql.ResultData(data_array=rows),
        )
    )


def _start(h):
    h.w.genie.start_conversation.return_value = Wait(
        lambda **kw: None,
        response=GenieStartConversationResponse(conversation_id="c1", message_id="m1"),
        conversation_id="c1",
        message_id="m1",
        space_id="sp1",
    )


# ------------------------------------------------------------------------------------------
# ask_genie
# ------------------------------------------------------------------------------------------


async def test_ask_genie_returns_labelled_model_generated_answer(genie):
    h = genie()
    _start(h)
    h.w.genie.get_message.return_value = _completed_with_query()
    h.w.genie.get_message_attachment_query_result.return_value = _query_result([["EU", "3"], ["US", "5"]])

    result = await h.call("ask_genie", {"space_id": "sp1", "question": "How many orders per region?"})

    assert result["status"] == "success"
    assert "MODEL-GENERATED" in result["summary"]
    data = result["data"]
    assert data["answer_is_model_generated"] is True
    assert data["answer_source"] == "model_generated"
    assert "verify the SQL" in data["sql_result_note"]
    assert data["text_response"] == "EU has 3 orders and US has 5."
    query = data["queries"][0]
    assert query["generated_sql"].startswith("SELECT region")
    assert query["description"] == "Counts orders grouped by region"
    assert query["sql_is_model_generated"] is True
    assert query["sql_result"]["rows"] == [{"region": "EU", "n": "3"}, {"region": "US", "n": "5"}]
    assert query["sql_result"]["truncated"] is False
    assert (data["conversation_id"], data["message_id"]) == ("c1", "m1")
    h.w.genie.start_conversation.assert_called_once_with(space_id="sp1", content="How many orders per region?")
    h.w.genie.get_message_attachment_query_result.assert_called_once_with(
        space_id="sp1", conversation_id="c1", message_id="m1", attachment_id="a1"
    )


async def test_ask_genie_pending_returns_ids(genie):
    h = genie()
    _start(h)
    h.w.genie.get_message.return_value = _message(status=MessageStatus.EXECUTING_QUERY)
    result = await h.call("ask_genie", {"space_id": "sp1", "question": "q", "wait_seconds": 0})
    assert result["status"] == "pending"
    assert result["data"]["conversation_id"] == "c1" and result["data"]["message_id"] == "m1"
    assert "message_id='m1'" in result["next_steps"][0]
    h.w.genie.get_message_attachment_query_result.assert_not_called()


async def test_ask_genie_poll_mode(genie):
    h = genie()
    h.w.genie.get_message.return_value = _completed_with_query()
    h.w.genie.get_message_attachment_query_result.return_value = _query_result([["EU", "3"]])
    result = await h.call("ask_genie", {"space_id": "sp1", "conversation_id": "c1", "message_id": "m1"})
    assert result["status"] == "success"
    h.w.genie.start_conversation.assert_not_called()
    h.w.genie.create_message.assert_not_called()


async def test_ask_genie_follow_up_uses_create_message(genie):
    h = genie()
    h.w.genie.create_message.return_value = Wait(
        lambda **kw: None, response=_message(status=MessageStatus.SUBMITTED), conversation_id="c1", message_id="m1"
    )
    h.w.genie.get_message.return_value = _message(attachments=[GenieAttachment(text=TextAttachment(content="Yes."))])
    result = await h.call("ask_genie", {"space_id": "sp1", "conversation_id": "c1", "question": "And in 2024?"})
    assert result["data"]["text_response"] == "Yes."
    h.w.genie.create_message.assert_called_once_with(space_id="sp1", conversation_id="c1", content="And in 2024?")
    h.w.genie.start_conversation.assert_not_called()


async def test_ask_genie_failed_message(genie):
    h = genie()
    _start(h)
    h.w.genie.get_message.return_value = _message(status=MessageStatus.FAILED, error=MessageError(error="Table not found"))
    result = await h.call("ask_genie", {"space_id": "sp1", "question": "q"})
    assert result["status"] == "failed"
    assert "Table not found" in result["summary"]


async def test_ask_genie_rows_capped(genie):
    h = genie(sql_max_rows=1)
    _start(h)
    h.w.genie.get_message.return_value = _completed_with_query()
    h.w.genie.get_message_attachment_query_result.return_value = _query_result([["EU", "3"], ["US", "5"]])
    result = await h.call("ask_genie", {"space_id": "sp1", "question": "q", "max_rows": 50})
    res = result["data"]["queries"][0]["sql_result"]
    assert res["rows"] == [{"region": "EU", "n": "3"}]
    assert res["truncated"] is True
    assert result["warnings"]


async def test_ask_genie_requires_question_or_message(genie):
    h = genie()
    msg = await h.call_error("ask_genie", {"space_id": "sp1"})
    assert "question" in msg


async def test_ask_genie_allowed_in_read_only_but_writes_blocked(genie):
    h = genie(read_only=True)
    _start(h)
    h.w.genie.get_message.return_value = _message(attachments=[GenieAttachment(text=TextAttachment(content="42"))])
    result = await h.call("ask_genie", {"space_id": "sp1", "question": "q"})
    assert result["status"] == "success"
    msg = await h.call_error("manage_genie", {"action": "create", "spec": {"warehouse_id": "w", "serialized_space": "{}"}})
    assert "BLOCKED_BY_SAFETY_POLICY" in msg


# ------------------------------------------------------------------------------------------
# manage_genie
# ------------------------------------------------------------------------------------------


async def test_genie_list_follows_server_pages(genie):
    h = genie()
    pages = {
        None: GenieListSpacesResponse(spaces=[GenieSpace(space_id="s1", title="A"), GenieSpace(space_id="s2", title="B")], next_page_token="p2"),
        "p2": GenieListSpacesResponse(spaces=[GenieSpace(space_id="s3", title="C")]),
    }
    h.w.genie.list_spaces.side_effect = lambda page_token=None, page_size=None: pages[page_token]
    first = await h.call("manage_genie", {"action": "list", "page_size": 2})
    assert [s["space_id"] for s in first["data"]] == ["s1", "s2"]
    assert first["page"]["has_more"] is True
    second = await h.call("manage_genie", {"action": "list", "page_size": 2, "page_token": first["page"]["next_page_token"]})
    assert [s["space_id"] for s in second["data"]] == ["s3"]


async def test_genie_create_serializes_space_and_tracks(genie):
    h = genie()
    h.w.genie.create_space.return_value = GenieSpace(space_id="new", title="Sales")
    result = await h.call(
        "manage_genie",
        {"action": "create", "spec": {"warehouse_id": "wh1", "title": "Sales", "serialized_space": {"version": 1}}},
    )
    assert result["status"] == "success"
    kwargs = h.w.genie.create_space.call_args.kwargs
    assert kwargs["serialized_space"] == json.dumps({"version": 1})
    assert json.loads(h.settings.manifest_path.read_text())["resources"][0]["resource_type"] == "genie_space"


async def test_genie_create_rejects_unknown_field(genie):
    h = genie()
    msg = await h.call_error("manage_genie", {"action": "create", "spec": {"warehouse_id": "w", "serialized_space": "{}", "owner": "me"}})
    assert "Unknown field" in msg and "owner" in msg
    h.w.genie.create_space.assert_not_called()


async def test_genie_delete_requires_confirm(genie):
    h = genie()
    h.w.genie.get_space.return_value = GenieSpace(space_id="s1", title="Sandbox")
    result = await h.call("manage_genie", {"action": "delete", "space_id": "s1"})
    assert result["status"] == "confirmation_required"
    h.w.genie.trash_space.assert_not_called()
    result = await h.call("manage_genie", {"action": "delete", "space_id": "s1", "confirm": True})
    assert result["status"] == "success"
    h.w.genie.trash_space.assert_called_once_with(space_id="s1")


async def test_genie_get_not_found(genie):
    h = genie()
    h.w.genie.get_space.side_effect = NotFound("space missing")
    msg = await h.call_error("manage_genie", {"action": "get", "space_id": "nope"})
    assert "[NOT_FOUND]" in msg
