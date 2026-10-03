"""Foundation tests: config, redaction, validation, SQL classification, serialization, pagination, errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from databricks.sdk import errors as dbx_errors
from databricks.sdk.service import compute, jobs

from dbx_mcp.safety.levels import SafetyLevel, is_read_action, needs_confirmation
from dbx_mcp.safety.validation import (
    classify_sql,
    classify_statement,
    parse_volume_path,
    quote_full_name,
    quote_string_literal,
    resolve_local_path,
    split_full_name,
    split_sql_statements,
    validate_workspace_path,
)
from dbx_mcp.server.config import ALL_TOOLSETS, Settings
from dbx_mcp.utils.errors import (
    DbxToolError,
    ErrorCategory,
    SafetyBlockedError,
    ValidationFailed,
    normalize_exception,
)
from dbx_mcp.utils.pagination import decode_cursor, encode_cursor, paginate
from dbx_mcp.utils.redaction import REDACTED, redact, redact_text
from dbx_mcp.utils.serialization import coerce_kwargs, parse_sdk_object, to_jsonable
from tests.fakes import FAKE_PAT

# ---------------------------------------------------------------------------------------------- config


def test_settings_defaults():
    s = Settings.from_env()
    assert s.toolsets == ALL_TOOLSETS
    assert not s.read_only
    assert s.require_confirmation
    assert s.protected_name_patterns  # production protection on by default
    assert s.max_wait_seconds < s.tool_timeout_seconds


def test_settings_parsing(monkeypatch):
    monkeypatch.setenv("DBX_MCP_TOOLSETS", "sql,compute")
    monkeypatch.setenv("DBX_MCP_READ_ONLY", "true")
    monkeypatch.setenv("DBX_MCP_BLOCKED_SAFETY_LEVELS", "destructive,security_sensitive")
    s = Settings.from_env()
    assert s.toolsets == ("sql", "compute")
    assert s.read_only
    assert s.blocked_safety_levels == {SafetyLevel.DESTRUCTIVE, SafetyLevel.SECURITY_SENSITIVE}


@pytest.mark.parametrize(
    "key,value",
    [
        ("DBX_MCP_TOOLSETS", "sql,bogus"),
        ("DBX_MCP_READ_ONLY", "maybe"),
        ("DBX_MCP_BLOCKED_SAFETY_LEVELS", "READ_ONLY"),
        ("DBX_MCP_PROTECTED_NAME_PATTERNS", "(unclosed"),
        ("DBX_MCP_MAX_WAIT_SECONDS", "999"),
    ],
)
def test_settings_rejects_invalid(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        Settings.from_env()


# ---------------------------------------------------------------------------------------------- safety levels


def test_level_semantics():
    assert is_read_action({SafetyLevel.READ_ONLY})
    assert is_read_action({SafetyLevel.READ_ONLY, SafetyLevel.EXECUTION})
    assert is_read_action({SafetyLevel.READ_ONLY, SafetyLevel.SECURITY_SENSITIVE})
    assert not is_read_action({SafetyLevel.EXECUTION})
    assert not needs_confirmation({SafetyLevel.READ_ONLY, SafetyLevel.SECURITY_SENSITIVE})
    assert needs_confirmation({SafetyLevel.DESTRUCTIVE})
    assert needs_confirmation({SafetyLevel.WRITE, SafetyLevel.SECURITY_SENSITIVE})
    assert not needs_confirmation({SafetyLevel.EXECUTION})
    assert needs_confirmation({SafetyLevel.EXECUTION}, confirm_execution=True)


# ---------------------------------------------------------------------------------------------- redaction


def test_redact_nested_secrets():
    data = {
        "name": "conn",
        "options": {"host": "db", "password": "hunter2", "user": "me"},
        "token_value": "abc",
        "next_page_token": "keep-me",
        "items": [{"client_secret": "s3cr3t", "client_id": "id"}],
        "enabled": True,
    }
    out = redact(data)
    assert out["options"]["password"] == REDACTED
    assert out["options"]["host"] == "db"
    assert out["token_value"] == REDACTED
    assert out["next_page_token"] == "keep-me"
    assert out["items"][0]["client_secret"] == REDACTED
    assert out["items"][0]["client_id"] == "id"
    assert data["options"]["password"] == "hunter2"  # original untouched


def test_redact_allow_keys_and_text():
    assert redact({"token": "x"}, allow_keys={"token"}) == {"token": "x"}
    text = f"auth Bearer abcdefghijklmnopqrstuvwxyz0123 and {FAKE_PAT} password=foo"
    cleaned = redact_text(text)
    assert "abcdefghijklmnopqrstuvwxyz0123" not in cleaned
    assert FAKE_PAT not in cleaned
    assert "password=foo" not in cleaned


# ---------------------------------------------------------------------------------------------- paths & names


def test_volume_paths():
    vp = parse_volume_path("dbfs:/Volumes/main/raw/files/a/b.csv")
    assert (vp.catalog, vp.schema, vp.volume, vp.relative) == ("main", "raw", "files", "a/b.csv")
    assert vp.full == "/Volumes/main/raw/files/a/b.csv"
    assert parse_volume_path("/Volumes/main/raw/files/").relative == ""
    for bad in [
        "/Volumes/main/raw/files/../../etc",
        "/Volumes/main/raw",
        "/etc/passwd",
        "Volumes/main/raw/files",
        "/Volumes/main/raw/files/a\\b",
        "/Volumes/main//files/x",
        "/Volumes/main/raw/files/x\x00y",
    ]:
        with pytest.raises(ValidationFailed):
            parse_volume_path(bad)
    with pytest.raises(ValidationFailed):
        parse_volume_path("/Volumes/main/raw/files", require_file=True)


def test_workspace_paths():
    assert validate_workspace_path("/Users/a@b.com/nb/") == "/Users/a@b.com/nb"
    for bad in ["relative/x", "/Users/../etc", "/Volumes/a/b/c", "/Users//x"]:
        with pytest.raises(ValidationFailed):
            validate_workspace_path(bad)


def test_local_path_resolution(tmp_path: Path):
    (tmp_path / "f.txt").write_text("hi")
    assert resolve_local_path(tmp_path, "f.txt", must_exist=True) == (tmp_path / "f.txt").resolve()
    with pytest.raises(SafetyBlockedError):
        resolve_local_path(tmp_path, "../outside.txt", must_exist=False)
    with pytest.raises(SafetyBlockedError):
        resolve_local_path(None, "f.txt", must_exist=True)


def test_identifier_quoting():
    assert split_full_name("`my.cat`.s.t", parts=3) == ["my.cat", "s", "t"]
    assert quote_full_name("main.default.`my``tbl`") == "`main`.`default`.`my``tbl`"
    assert quote_full_name("main.default.t; DROP TABLE x") == "`main`.`default`.`t; DROP TABLE x`"
    with pytest.raises(ValidationFailed, match="Unbalanced"):
        quote_full_name("main.default.my`tbl")
    assert quote_string_literal("it's \\ fine") == "'it\\'s \\\\ fine'"
    with pytest.raises(ValidationFailed):
        split_full_name("a.b", parts=3)
    with pytest.raises(ValidationFailed):
        split_full_name("`a.b", parts=(1, 2))


# ---------------------------------------------------------------------------------------------- SQL classification


@pytest.mark.parametrize(
    "sql,kind,must_have,must_not_have",
    [
        ("SELECT * FROM t", "read", {SafetyLevel.READ_ONLY}, {SafetyLevel.WRITE}),
        ("  -- note\n select 1", "read", {SafetyLevel.READ_ONLY}, set()),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "read", {SafetyLevel.READ_ONLY}, set()),
        ("SHOW TABLES IN main.default", "read", {SafetyLevel.READ_ONLY}, set()),
        ("SELECT 'DROP TABLE x' AS s", "read", {SafetyLevel.READ_ONLY}, {SafetyLevel.DESTRUCTIVE}),
        ("INSERT INTO t VALUES (1)", "write", {SafetyLevel.WRITE}, {SafetyLevel.DESTRUCTIVE}),
        ("CREATE TABLE t (a INT)", "write", {SafetyLevel.WRITE}, {SafetyLevel.DESTRUCTIVE}),
        ("CREATE OR REPLACE TABLE t AS SELECT 1", "destructive", {SafetyLevel.DESTRUCTIVE}, set()),
        ("INSERT OVERWRITE t SELECT 1", "destructive", {SafetyLevel.DESTRUCTIVE}, set()),
        ("DROP TABLE t", "destructive", {SafetyLevel.DESTRUCTIVE}, {SafetyLevel.READ_ONLY}),
        ("DELETE FROM t WHERE a=1", "destructive", {SafetyLevel.DESTRUCTIVE}, set()),
        ("UPDATE t SET a=1", "destructive", {SafetyLevel.DESTRUCTIVE}, set()),
        ("MERGE INTO t USING s ON t.id=s.id WHEN MATCHED THEN DELETE", "destructive", {SafetyLevel.DESTRUCTIVE}, set()),
        ("ALTER TABLE t DROP COLUMN a", "destructive", {SafetyLevel.DESTRUCTIVE}, set()),
        ("GRANT SELECT ON TABLE t TO `x`", "security", {SafetyLevel.SECURITY_SENSITIVE}, {SafetyLevel.DESTRUCTIVE}),
        ("REVOKE SELECT ON TABLE t FROM `x`", "security", {SafetyLevel.SECURITY_SENSITIVE, SafetyLevel.DESTRUCTIVE}, set()),
        ("ALTER TABLE t SET ROW FILTER f ON (a)", "write", {SafetyLevel.SECURITY_SENSITIVE}, set()),
        ("ALTER TABLE t OWNER TO `x`", "write", {SafetyLevel.SECURITY_SENSITIVE}, set()),
        ("FROBNICATE everything", "unknown", {SafetyLevel.DESTRUCTIVE}, set()),
    ],
)
def test_classify_statement(sql, kind, must_have, must_not_have):
    c = classify_statement(sql)
    assert c.kind == kind
    assert must_have <= c.levels
    assert not (must_not_have & c.levels)
    assert SafetyLevel.EXECUTION in c.levels


def test_split_statements_respects_literals_and_comments():
    script = "SELECT ';' AS a; -- c;omment\nSELECT 2 /* ; */;\n\n;  "
    assert split_sql_statements(script) == ["SELECT ';' AS a", "-- c;omment\nSELECT 2 /* ; */"]
    items, levels = classify_sql("SELECT 1; DROP TABLE t")
    assert len(items) == 2
    assert SafetyLevel.DESTRUCTIVE in levels and SafetyLevel.READ_ONLY not in levels


# ---------------------------------------------------------------------------------------------- serialization


def test_coerce_kwargs_converts_and_validates():
    from databricks.sdk.service.compute import ClustersAPI

    api = ClustersAPI(api_client=None)
    kwargs = coerce_kwargs(
        api.create,
        {"spark_version": "15.4.x-scala2.12", "autoscale": {"min_workers": 1, "max_workers": 2},
         "data_security_mode": "SINGLE_USER"},
    )
    assert isinstance(kwargs["autoscale"], compute.AutoScale)
    assert kwargs["data_security_mode"] is compute.DataSecurityMode.SINGLE_USER

    with pytest.raises(ValidationFailed, match="Unknown field"):
        coerce_kwargs(api.create, {"spark_version": "x", "clustr_name": "typo"})
    with pytest.raises(ValidationFailed, match="Missing required"):
        coerce_kwargs(api.create, {"cluster_name": "x"})
    with pytest.raises(ValidationFailed, match="invalid value"):
        coerce_kwargs(api.create, {"spark_version": "x", "data_security_mode": "NOPE"})
    with pytest.raises(ValidationFailed, match="autoscale.nope"):
        coerce_kwargs(api.create, {"spark_version": "x", "autoscale": {"min_workers": 1, "nope": 2}})
    with pytest.raises(ValidationFailed, match="dedicated tool parameters"):
        coerce_kwargs(api.edit, {"cluster_id": "a", "spark_version": "x"}, fixed={"cluster_id": "b"})


def test_parse_sdk_object_nested_list():
    settings = parse_sdk_object(
        jobs.JobSettings,
        {"name": "j", "tasks": [{"task_key": "a", "notebook_task": {"notebook_path": "/x"}}]},
    )
    assert settings.tasks[0].notebook_task.notebook_path == "/x"
    with pytest.raises(ValidationFailed, match=r"tasks\[0\].bogus"):
        parse_sdk_object(jobs.JobSettings, {"tasks": [{"task_key": "a", "bogus": 1}]})


def test_to_jsonable():
    obj = compute.ClusterDetails(cluster_id="c", state=compute.State.RUNNING, autoscale=compute.AutoScale(1, 2))
    assert to_jsonable(obj) == {"cluster_id": "c", "state": "RUNNING", "autoscale": {"max_workers": 1, "min_workers": 2}}


# ---------------------------------------------------------------------------------------------- pagination


def test_paginate_round_trip():
    items = list(range(7))
    page1, info1 = paginate(items, page_size=3, page_token=None)
    assert page1 == [0, 1, 2] and info1.has_more and info1.next_page_token
    page2, info2 = paginate(items, page_size=3, page_token=info1.next_page_token)
    assert page2 == [3, 4, 5]
    page3, info3 = paginate(items, page_size=3, page_token=info2.next_page_token)
    assert page3 == [6] and not info3.has_more and info3.next_page_token is None
    assert decode_cursor(encode_cursor(42)) == 42
    with pytest.raises(ValidationFailed):
        decode_cursor("garbage!!")


# ---------------------------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "exc,category",
    [
        (dbx_errors.NotFound("nope"), ErrorCategory.NOT_FOUND),
        (dbx_errors.PermissionDenied("no"), ErrorCategory.AUTHORIZATION),
        (dbx_errors.PermissionDenied("403: Invalid access token."), ErrorCategory.AUTHENTICATION),
        (dbx_errors.Unauthenticated("bad token"), ErrorCategory.AUTHENTICATION),
        (dbx_errors.InvalidParameterValue("bad"), ErrorCategory.INVALID_PARAMETER),
        (dbx_errors.ResourceAlreadyExists("dup"), ErrorCategory.CONFLICT),
        (dbx_errors.TooManyRequests("slow down"), ErrorCategory.RATE_LIMIT),
        (dbx_errors.DeadlineExceeded("late"), ErrorCategory.TIMEOUT),
        (dbx_errors.NotImplemented("no"), ErrorCategory.UNSUPPORTED),
        (dbx_errors.InternalError("boom"), ErrorCategory.SERVICE_ERROR),
        (TimeoutError("t"), ErrorCategory.TIMEOUT),
        (RuntimeError("x"), ErrorCategory.INTERNAL),
    ],
)
def test_error_normalization(exc, category):
    err = normalize_exception(exc)
    assert isinstance(err, DbxToolError)
    assert err.category == category
    assert str(err).startswith(f"[{category.value}]")


def test_internal_errors_hide_details_unless_debug():
    err = normalize_exception(RuntimeError(f"secret internals {FAKE_PAT}"))
    assert "secret internals" not in str(err)
    debug = normalize_exception(RuntimeError("visible detail"), debug=True)
    assert "visible detail" in str(debug)
    leaked = normalize_exception(dbx_errors.BadRequest(f"token {FAKE_PAT} bad"))
    assert FAKE_PAT not in str(leaked)
