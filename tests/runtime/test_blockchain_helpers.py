"""Tests for scripts.runtime.blockchain_helpers — env file helpers and utility functions."""

import json
import os

import pytest

from scripts.runtime.blockchain_helpers import (
    _load_env_file,
    _parse_bulk_errors,
    _set_env_value,
    _trainer_identifier,
    resolve_auth_secret,
)


class TestLoadEnvFile:
    def test_simple_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value\n")
        result = _load_env_file(env_file)
        assert result == {"KEY": "value"}

    def test_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('NAME="quoted"\nOTHER=\'single\'\n')
        result = _load_env_file(env_file)
        assert result["NAME"] == "quoted"
        assert result["OTHER"] == "single"

    def test_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n")
        result = _load_env_file(env_file)
        assert result == {"KEY": "val"}

    def test_handles_export_prefix(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("export MY_VAR=123\n")
        result = _load_env_file(env_file)
        assert result["MY_VAR"] == "123"

    def test_missing_file_returns_empty(self, tmp_path):
        result = _load_env_file(tmp_path / "nonexistent")
        assert result == {}

    def test_values_with_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("URL=http://host:9000?foo=bar\n")
        result = _load_env_file(env_file)
        assert result["URL"] == "http://host:9000?foo=bar"


class TestSetEnvValue:
    def test_updates_existing_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=old\nOTHER=keep\n")
        _set_env_value(env_file, "KEY", "new")
        result = _load_env_file(env_file)
        assert result["KEY"] == "new"
        assert result["OTHER"] == "keep"

    def test_appends_new_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=yes\n")
        _set_env_value(env_file, "NEW_KEY", "added")
        result = _load_env_file(env_file)
        assert result["NEW_KEY"] == "added"
        assert result["EXISTING"] == "yes"

    def test_creates_file_if_missing(self, tmp_path):
        env_file = tmp_path / ".env"
        _set_env_value(env_file, "FIRST", "value")
        assert env_file.exists()
        result = _load_env_file(env_file)
        assert result["FIRST"] == "value"

    def test_preserves_export_prefix(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("export SECRET=old\n")
        _set_env_value(env_file, "SECRET", "new")
        content = env_file.read_text()
        assert "export SECRET=new" in content


class TestResolveAuthSecret:
    def test_from_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTH_JWT_SECRET", "env-secret")
        paths = {"api_gateway": tmp_path}
        assert resolve_auth_secret(paths) == "env-secret"

    def test_from_env_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AUTH_JWT_SECRET", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("AUTH_JWT_SECRET=file-secret\n")
        paths = {"api_gateway": tmp_path}
        assert resolve_auth_secret(paths) == "file-secret"

    def test_missing_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AUTH_JWT_SECRET", raising=False)
        paths = {"api_gateway": tmp_path}
        with pytest.raises(SystemExit, match="AUTH_JWT_SECRET"):
            resolve_auth_secret(paths)


class TestTrainerIdentifier:
    def test_jwt_sub(self):
        assert _trainer_identifier({"jwt_sub": "node-1"}) == "node-1"

    def test_node_id(self):
        assert _trainer_identifier({"nodeId": "n-42"}) == "n-42"

    def test_did(self):
        assert _trainer_identifier({"did": "did:example:123"}) == "did:example:123"

    def test_unknown_fallback(self):
        assert _trainer_identifier({}) == "unknown"

    def test_priority_order(self):
        entry = {"jwt_sub": "first", "nodeId": "second", "did": "third"}
        assert _trainer_identifier(entry) == "first"


class TestParseBulkErrors:
    def test_empty_body(self):
        assert _parse_bulk_errors("") == []

    def test_all_ok(self):
        body = json.dumps({"results": [{"status": "ok"}, {"status": "OK"}]})
        assert _parse_bulk_errors(body) == []

    def test_extracts_errors(self):
        body = json.dumps({
            "results": [
                {"status": "ok", "jwt_sub": "n1"},
                {"status": "error", "jwt_sub": "n2", "error": "duplicate"},
            ]
        })
        errors = _parse_bulk_errors(body)
        assert len(errors) == 1
        assert errors[0]["id"] == "n2"
        assert errors[0]["error"] == "duplicate"

    def test_invalid_json_returns_empty(self):
        assert _parse_bulk_errors("not json") == []

    def test_no_results_key_returns_empty(self):
        assert _parse_bulk_errors(json.dumps({"other": []})) == []
