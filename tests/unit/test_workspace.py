"""Unit tests for manage_workspace_files / execute_code (mocked WorkspaceClient)."""

from __future__ import annotations

import ast
import base64
import json

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service import compute, iam, jobs
from databricks.sdk.service import workspace as ws
from databricks.sdk.service._internal import Wait

import dbx_mcp.tools.workspace as ws_tools


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(ws_tools, "POLL_INTERVAL_SECONDS", 0.05)


@pytest.fixture
def h(make_harness):
    return make_harness(toolsets=("workspace",))


def _wait(response=None) -> Wait:
    return Wait(lambda **kwargs: None, response=response)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------------------------
# manage_workspace_files
# ---------------------------------------------------------------------------------------------

async def test_tools_registered(h):
    assert {"manage_workspace_files", "execute_code"} <= h.tool_names()


async def test_list_paginates(h):
    objs = [ws.ObjectInfo(path=f"/Users/me/f{i}", object_type=ws.ObjectType.FILE, size=i) for i in range(3)]
    h.w.workspace.list.side_effect = lambda path, **kw: iter(objs)
    result = await h.call("manage_workspace_files", {"action": "list", "path": "/Users/me/", "page_size": 2})
    assert [o["path"] for o in result["data"]] == ["/Users/me/f0", "/Users/me/f1"]
    assert result["data"][0]["object_type"] == "FILE"
    assert result["page"]["has_more"] is True
    h.w.workspace.list.assert_called_with("/Users/me", recursive=False)


async def test_get_status(h):
    h.w.workspace.get_status.return_value = ws.ObjectInfo(path="/Users/me/nb", object_type=ws.ObjectType.NOTEBOOK,
                                                          language=ws.Language.PYTHON, object_id=5)
    result = await h.call("manage_workspace_files", {"action": "get_status", "path": "/Users/me/nb"})
    assert result["data"]["language"] == "PYTHON"


async def test_get_status_not_found(h):
    h.w.workspace.get_status.side_effect = NotFound("RESOURCE_DOES_NOT_EXIST")
    message = await h.call_error("manage_workspace_files", {"action": "get_status", "path": "/Users/me/missing"})
    assert "[NOT_FOUND]" in message


@pytest.mark.parametrize("bad", ["relative/path", "/Users/me/../other", "/Volumes/a/b/c/f", "C:\\x"])
async def test_invalid_paths_rejected(h, bad):
    message = await h.call_error("manage_workspace_files", {"action": "get_status", "path": bad})
    assert "[INVALID_PARAMETER]" in message
    h.w.workspace.get_status.assert_not_called()


async def test_workspace_prefix_allowlist(make_harness):
    h = make_harness(toolsets=("workspace",), allowed_workspace_prefixes=("/Users/me",))
    message = await h.call_error("manage_workspace_files", {"action": "get_status", "path": "/Shared/x"})
    assert "BLOCKED_BY_SAFETY_POLICY" in message


async def test_export_notebook_as_text(h):
    h.w.workspace.get_status.return_value = ws.ObjectInfo(path="/Users/me/nb", object_type=ws.ObjectType.NOTEBOOK,
                                                          language=ws.Language.PYTHON)
    h.w.workspace.export.return_value = ws.ExportResponse(content=_b64(b"print('hi')\n"), file_type="py")
    result = await h.call("manage_workspace_files", {"action": "export", "path": "/Users/me/nb"})
    assert result["data"]["encoding"] == "utf-8"
    assert result["data"]["content"] == "print('hi')\n"
    h.w.workspace.export.assert_called_once_with("/Users/me/nb", format=ws.ExportFormat.SOURCE)


async def test_export_binary_file_as_base64(h):
    h.w.workspace.get_status.return_value = ws.ObjectInfo(path="/Users/me/a.bin", object_type=ws.ObjectType.FILE)
    payload = bytes([0xFF, 0xFE, 0x00, 0x81])
    h.w.workspace.export.return_value = ws.ExportResponse(content=_b64(payload))
    result = await h.call("manage_workspace_files", {"action": "export", "path": "/Users/me/a.bin"})
    assert result["data"]["encoding"] == "base64"
    assert base64.b64decode(result["data"]["content"]) == payload
    h.w.workspace.export.assert_called_once_with("/Users/me/a.bin", format=ws.ExportFormat.AUTO)


async def test_export_too_large_for_inline(make_harness):
    h = make_harness(toolsets=("workspace",), max_inline_download_bytes=1024)
    h.w.workspace.get_status.return_value = ws.ObjectInfo(path="/Users/me/big", object_type=ws.ObjectType.FILE,
                                                          size=5000)
    message = await h.call_error("manage_workspace_files", {"action": "export", "path": "/Users/me/big"})
    assert "[INVALID_PARAMETER]" in message and "inline limit" in message
    h.w.workspace.export.assert_not_called()


async def test_export_to_local_file(make_harness, tmp_path):
    root = tmp_path / "local"
    root.mkdir()
    h = make_harness(toolsets=("workspace",), local_file_root=root.resolve())
    h.w.workspace.get_status.return_value = ws.ObjectInfo(path="/Users/me/a.txt", object_type=ws.ObjectType.FILE)
    h.w.workspace.export.return_value = ws.ExportResponse(content=_b64(b"data"))
    result = await h.call("manage_workspace_files",
                          {"action": "export", "path": "/Users/me/a.txt", "local_path": "out/a.txt"})
    assert result["status"] == "success"
    assert (root / "out" / "a.txt").read_bytes() == b"data"


async def test_import_notebook_source(h):
    result = await h.call("manage_workspace_files", {
        "action": "import", "path": "/Users/me/proj/nb", "content": "print(1)", "language": "PYTHON"})
    assert result["status"] == "success"
    h.w.workspace.mkdirs.assert_called_once_with("/Users/me/proj")
    h.w.workspace.import_.assert_called_once_with(
        "/Users/me/proj/nb", content=_b64(b"print(1)"), format=ws.ImportFormat.SOURCE,
        language=ws.Language.PYTHON, overwrite=False)


async def test_import_plain_file_defaults_to_auto(h):
    await h.call("manage_workspace_files", {"action": "import", "path": "/Users/me/data.json",
                                            "content_base64": _b64(b"{}"), "create_parents": False})
    h.w.workspace.mkdirs.assert_not_called()
    kwargs = h.w.workspace.import_.call_args.kwargs
    assert kwargs["format"] == ws.ImportFormat.AUTO and kwargs["language"] is None


async def test_import_source_without_language_rejected(h):
    message = await h.call_error("manage_workspace_files", {"action": "import", "path": "/Users/me/nb",
                                                            "content": "x", "format": "SOURCE"})
    assert "[INVALID_PARAMETER]" in message
    h.w.workspace.import_.assert_not_called()


async def test_import_requires_exactly_one_source(h):
    message = await h.call_error("manage_workspace_files", {"action": "import", "path": "/Users/me/nb",
                                                            "content": "x", "content_base64": _b64(b"x")})
    assert "[INVALID_PARAMETER]" in message


async def test_import_local_file_disabled_by_default(h):
    message = await h.call_error("manage_workspace_files", {"action": "import", "path": "/Users/me/a.py",
                                                            "local_path": "a.py"})
    assert "BLOCKED_BY_SAFETY_POLICY" in message


async def test_import_local_file_escape_blocked(make_harness, tmp_path):
    h = make_harness(toolsets=("workspace",), local_file_root=tmp_path.resolve())
    message = await h.call_error("manage_workspace_files", {"action": "import", "path": "/Users/me/a.py",
                                                            "local_path": "../../etc/passwd"})
    assert "BLOCKED_BY_SAFETY_POLICY" in message or "[INVALID_PARAMETER]" in message
    h.w.workspace.import_.assert_not_called()


async def test_import_overwrite_requires_confirmation(h):
    h.w.workspace.get_status.return_value = ws.ObjectInfo(path="/Users/me/nb", object_type=ws.ObjectType.NOTEBOOK)
    args = {"action": "import", "path": "/Users/me/nb", "content": "x", "language": "SQL", "overwrite": True}
    result = await h.call("manage_workspace_files", args)
    assert result["status"] == "confirmation_required"
    assert result["plan"]["details"]["existing_object"]["object_type"] == "NOTEBOOK"
    h.w.workspace.import_.assert_not_called()
    result = await h.call("manage_workspace_files", {**args, "confirm": True})
    assert result["status"] == "success"
    assert h.w.workspace.import_.call_args.kwargs["overwrite"] is True


async def test_read_only_blocks_import(make_harness):
    h = make_harness(read_only=True, toolsets=("workspace",))
    message = await h.call_error("manage_workspace_files", {"action": "import", "path": "/Users/me/x", "content": "x"})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.workspace.import_.assert_not_called()


async def test_mkdirs_dry_run(h):
    result = await h.call("manage_workspace_files", {"action": "mkdirs", "path": "/Users/me/new", "dry_run": True})
    assert result["status"] == "dry_run"
    h.w.workspace.mkdirs.assert_not_called()


async def test_delete_directory_confirmation_flow(h):
    h.w.workspace.get_status.return_value = ws.ObjectInfo(path="/Users/me/dir", object_type=ws.ObjectType.DIRECTORY)
    h.w.workspace.list.side_effect = lambda path, **kw: iter([ws.ObjectInfo(path="/Users/me/dir/a")])
    result = await h.call("manage_workspace_files", {"action": "delete", "path": "/Users/me/dir"})
    assert result["status"] == "confirmation_required"
    assert any("recursive=true" in w for w in result["plan"]["warnings"])
    h.w.workspace.delete.assert_not_called()
    result = await h.call("manage_workspace_files",
                          {"action": "delete", "path": "/Users/me/dir", "recursive": True, "confirm": True})
    assert result["status"] == "success"
    h.w.workspace.delete.assert_called_once_with("/Users/me/dir", recursive=True)


async def test_delete_protected_path_blocked(h):
    message = await h.call_error("manage_workspace_files",
                                 {"action": "delete", "path": "/Shared/prod/etl", "confirm": True})
    assert "BLOCKED_BY_SAFETY_POLICY" in message
    h.w.workspace.delete.assert_not_called()


# ---------------------------------------------------------------------------------------------
# execute_code - classic cluster
# ---------------------------------------------------------------------------------------------

def _cluster(state="RUNNING") -> compute.ClusterDetails:
    return compute.ClusterDetails(cluster_id="c1", cluster_name="dev-cluster", state=compute.State(state),
                                  spark_version="15.4.x-scala2.12")


def _setup_cluster(h, final: compute.CommandStatusResponse):
    h.w.clusters.get.return_value = _cluster()
    h.w.command_execution.create.return_value = _wait(compute.Created(id="ctx1"))
    h.w.command_execution.context_status.return_value = compute.ContextStatusResponse(
        id="ctx1", status=compute.ContextStatus.RUNNING)
    h.w.command_execution.execute.return_value = _wait(compute.Created(id="cmd1"))
    h.w.command_execution.command_status.return_value = final


async def test_execute_python_text_output(h):
    _setup_cluster(h, compute.CommandStatusResponse(
        id="cmd1", status=compute.CommandStatus.FINISHED,
        results=compute.Results(result_type=compute.ResultType.TEXT, data="hello")))
    result = await h.call("execute_code", {"code": "print('hello')", "cluster_id": "c1"})
    assert result["status"] == "success"
    assert result["data"]["output"] == "hello"
    assert result["data"]["compute"]["cluster_name"] == "dev-cluster"
    assert "EXECUTION" in result["safety"]
    h.w.command_execution.execute.assert_called_once_with(
        cluster_id="c1", context_id="ctx1", language=compute.Language.PYTHON, command="print('hello')")
    h.w.command_execution.destroy.assert_called_once_with("c1", "ctx1")


async def test_execute_sql_table_output(h):
    _setup_cluster(h, compute.CommandStatusResponse(
        id="cmd1", status=compute.CommandStatus.FINISHED,
        results=compute.Results(result_type=compute.ResultType.TABLE, data=[[1, "a"], [2, "b"]],
                                schema=[{"name": "id", "type": "int"}, {"name": "v", "type": "string"}])))
    result = await h.call("execute_code", {"code": "SELECT 1", "language": "sql", "cluster_id": "c1"})
    assert result["data"]["columns"] == [{"name": "id", "type": "int"}, {"name": "v", "type": "string"}]
    assert result["data"]["rows"] == [[1, "a"], [2, "b"]]
    assert h.w.command_execution.create.call_args.kwargs["language"] == compute.Language.SQL


async def test_execute_error_is_failed_with_clean_trace(h):
    _setup_cluster(h, compute.CommandStatusResponse(
        id="cmd1", status=compute.CommandStatus.FINISHED,
        results=compute.Results(result_type=compute.ResultType.ERROR, summary="NameError: x",
                                cause="\x1b[0;31mNameError\x1b[0m: name 'x' is not defined")))
    result = await h.call("execute_code", {"code": "x", "cluster_id": "c1"})
    assert result["status"] == "failed"
    assert result["data"]["error"]["cause"] == "NameError: name 'x' is not defined"
    h.w.command_execution.destroy.assert_called_once()


async def test_execute_pending_keeps_context(h):
    _setup_cluster(h, compute.CommandStatusResponse(id="cmd1", status=compute.CommandStatus.RUNNING))
    result = await h.call("execute_code", {"code": "import time; time.sleep(999)", "cluster_id": "c1",
                                           "timeout_seconds": 1})
    assert result["status"] == "pending"
    assert result["data"]["context_id"] == "ctx1" and result["data"]["command_id"] == "cmd1"
    h.w.command_execution.destroy.assert_not_called()


async def test_get_status_finished_destroys_context(h):
    h.w.command_execution.command_status.return_value = compute.CommandStatusResponse(
        id="cmd1", status=compute.CommandStatus.FINISHED,
        results=compute.Results(result_type=compute.ResultType.TEXT, data="done"))
    result = await h.call("execute_code", {"action": "get_status", "cluster_id": "c1", "context_id": "ctx1",
                                           "command_id": "cmd1"})
    assert result["status"] == "success" and result["data"]["output"] == "done"
    h.w.command_execution.destroy.assert_called_once_with("c1", "ctx1")


async def test_cancel_command(h):
    h.w.command_execution.cancel.return_value = _wait()
    result = await h.call("execute_code", {"action": "cancel", "cluster_id": "c1", "context_id": "ctx1",
                                           "command_id": "cmd1"})
    assert result["status"] == "success"
    h.w.command_execution.cancel.assert_called_once_with(cluster_id="c1", context_id="ctx1", command_id="cmd1")


async def test_execute_on_terminated_cluster_errors(h):
    h.w.clusters.get.return_value = _cluster("TERMINATED")
    message = await h.call_error("execute_code", {"code": "1", "cluster_id": "c1"})
    assert "[CONFLICT]" in message and "TERMINATED" in message
    h.w.command_execution.create.assert_not_called()


async def test_execute_without_cluster_is_configuration_error(h):
    message = await h.call_error("execute_code", {"code": "1"})
    assert "[CONFIGURATION_ERROR]" in message


async def test_execute_uses_default_cluster(make_harness):
    h = make_harness(toolsets=("workspace",), default_cluster_id="c1")
    _setup_cluster(h, compute.CommandStatusResponse(
        id="cmd1", status=compute.CommandStatus.FINISHED,
        results=compute.Results(result_type=compute.ResultType.TEXT, data="")))
    result = await h.call("execute_code", {"code": "1"})
    assert result["data"]["compute"]["source"].startswith("default")
    h.w.clusters.get.assert_called_once_with("c1")


async def test_read_only_blocks_execution(make_harness):
    h = make_harness(read_only=True, toolsets=("workspace",))
    message = await h.call_error("execute_code", {"code": "1", "cluster_id": "c1"})
    assert "BLOCKED_BY_SAFETY_POLICY" in message


async def test_confirm_execution_setting_requires_confirm(make_harness):
    h = make_harness(toolsets=("workspace",), confirm_execution=True)
    result = await h.call("execute_code", {"code": "print(1)", "cluster_id": "c1"})
    assert result["status"] == "confirmation_required"
    assert result["plan"]["details"]["code_preview"] == "print(1)"
    h.w.command_execution.create.assert_not_called()


# ---------------------------------------------------------------------------------------------
# execute_code - serverless
# ---------------------------------------------------------------------------------------------

TMP_NB = "/Users/me@example.com/.dbx_mcp/tmp/execute_code_abc"


def _serverless_run(life="TERMINATED", result="SUCCESS") -> jobs.Run:
    return jobs.Run(
        run_id=9,
        state=jobs.RunState(life_cycle_state=jobs.RunLifeCycleState(life),
                            result_state=jobs.RunResultState(result) if result else None),
        tasks=[jobs.RunTask(task_key="execute_code", run_id=10,
                            notebook_task=jobs.NotebookTask(notebook_path=TMP_NB))],
    )


async def test_serverless_python_captures_stdout_and_cleans_up(h):
    h.w.current_user.me.return_value = iam.User(user_name="me@example.com")
    h.w.jobs.submit.return_value = _wait(jobs.SubmitRunResponse(run_id=9))
    h.w.jobs.get_run.return_value = _serverless_run()
    h.w.jobs.get_run_output.return_value = jobs.RunOutput(notebook_output=jobs.NotebookOutput(
        result=json.dumps({"status": "ok", "stdout": "hi\n", "stderr": ""})))
    result = await h.call("execute_code", {"code": "print('hi')", "compute": "serverless", "timeout_seconds": 5})
    assert result["status"] == "success"
    assert result["data"]["stdout"] == "hi\n"
    assert result["data"]["compute"]["type"] == "serverless"
    import_kwargs = h.w.workspace.import_.call_args
    assert import_kwargs.args[0].startswith("/Users/me@example.com/.dbx_mcp/tmp/execute_code_")
    assert import_kwargs.kwargs["format"] == ws.ImportFormat.SOURCE
    submit_task = h.w.jobs.submit.call_args.kwargs["tasks"][0]
    assert submit_task.notebook_task.notebook_path == import_kwargs.args[0]
    assert submit_task.new_cluster is None and submit_task.existing_cluster_id is None
    h.w.jobs.get_run_output.assert_called_once_with(10)
    h.w.workspace.delete.assert_called_once_with(TMP_NB)


async def test_serverless_user_error_is_failed(h):
    h.w.current_user.me.return_value = iam.User(user_name="me@example.com")
    h.w.jobs.submit.return_value = _wait(jobs.SubmitRunResponse(run_id=9))
    h.w.jobs.get_run.return_value = _serverless_run()
    h.w.jobs.get_run_output.return_value = jobs.RunOutput(notebook_output=jobs.NotebookOutput(result=json.dumps(
        {"status": "error", "error": "ZeroDivisionError: division by zero", "traceback": "Traceback...",
         "stdout": "", "stderr": ""})))
    result = await h.call("execute_code", {"code": "1/0", "compute": "serverless", "timeout_seconds": 5})
    assert result["status"] == "failed"
    assert result["data"]["error"]["summary"] == "ZeroDivisionError: division by zero"


async def test_serverless_pending_then_get_status(h):
    h.w.current_user.me.return_value = iam.User(user_name="me@example.com")
    h.w.jobs.submit.return_value = _wait(jobs.SubmitRunResponse(run_id=9))
    h.w.jobs.get_run.return_value = _serverless_run(life="RUNNING", result=None)
    result = await h.call("execute_code", {"code": "1", "compute": "serverless", "timeout_seconds": 1})
    assert result["status"] == "pending"
    assert result["data"]["run_id"] == 9
    h.w.workspace.delete.assert_not_called()

    h.w.jobs.get_run.return_value = _serverless_run()
    h.w.jobs.get_run_output.return_value = jobs.RunOutput(notebook_output=jobs.NotebookOutput(
        result=json.dumps({"status": "ok", "stdout": "x", "stderr": ""})))
    result = await h.call("execute_code", {"action": "get_status", "run_id": 9})
    assert result["status"] == "success" and result["data"]["stdout"] == "x"


async def test_get_status_rejects_foreign_runs(h):
    h.w.jobs.get_run.return_value = jobs.Run(run_id=5, tasks=[jobs.RunTask(
        task_key="t", notebook_task=jobs.NotebookTask(notebook_path="/Users/me/real_job"))])
    message = await h.call_error("execute_code", {"action": "cancel", "run_id": 5})
    assert "[INVALID_PARAMETER]" in message
    h.w.jobs.cancel_run.assert_not_called()


async def test_serverless_rejects_non_python(h):
    message = await h.call_error("execute_code", {"code": "SELECT 1", "language": "sql", "compute": "serverless"})
    assert "[INVALID_PARAMETER]" in message
    h.w.jobs.submit.assert_not_called()


def test_serverless_notebook_is_valid_python_and_embeds_code():
    code = 'print("quotes \' and \\"")\n"""triple"""'
    source = ws_tools.build_serverless_notebook(code)
    ast.parse(source)
    encoded = source.split('b64decode("')[1].split('")')[0]
    assert base64.b64decode(encoded).decode() == code
    assert "dbutils.notebook.exit" in source
