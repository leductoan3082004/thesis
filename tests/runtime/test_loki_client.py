"""Tests for scripts.runtime.loki_client — LogQL building, time parsing, and entry formatting."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.runtime.loki_client import (
    _extract_entries,
    _ns_to_iso,
    _parse_time,
    _to_nanoseconds,
    build_logql,
    format_entries,
)


class TestBuildLogql:
    def test_no_filters_uses_wildcard(self):
        q = build_logql()
        assert q == '{service=~".+"}'

    def test_service_filter(self):
        q = build_logql(service="fl_node")
        assert 'service="fl_node"' in q

    def test_node_id_filter(self):
        q = build_logql(node_id="trainer-node-001")
        assert 'node_id="trainer-node-001"' in q

    def test_level_filter(self):
        q = build_logql(service="ttp", level="ERROR")
        assert '|~ "(?i)ERROR"' in q

    def test_contains_filter(self):
        q = build_logql(contains="accuracy")
        assert '|~ "accuracy"' in q

    def test_combined_filters(self):
        q = build_logql(service="fl_node", node_id="n1", level="WARN", contains="round")
        assert 'service="fl_node"' in q
        assert 'node_id="n1"' in q
        assert '|~ "(?i)WARN"' in q
        assert '|~ "round"' in q


class TestParseTime:
    def test_minutes(self):
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_time("30m", now)
        assert result == now - timedelta(minutes=30)

    def test_hours(self):
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_time("2h", now)
        assert result == now - timedelta(hours=2)

    def test_days(self):
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_time("7d", now)
        assert result == now - timedelta(days=7)

    def test_iso_timestamp(self):
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_time("2025-01-01T10:00:00", now)
        assert result.hour == 10

    def test_whitespace_stripped(self):
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_time("  5m  ", now)
        assert result == now - timedelta(minutes=5)


class TestToNanoseconds:
    def test_conversion(self):
        dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ns = _to_nanoseconds(dt)
        assert ns == str(int(dt.timestamp() * 1e9))


class TestNsToIso:
    def test_valid_timestamp(self):
        ns = str(int(datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp() * 1e9))
        iso = _ns_to_iso(ns)
        assert "2025-06-15" in iso
        assert "10:30:00" in iso

    def test_invalid_value_returns_input(self):
        assert _ns_to_iso("not-a-number") == "not-a-number"


class TestExtractEntries:
    def test_empty_body(self):
        assert _extract_entries({}) == []

    def test_single_stream(self):
        body = {
            "data": {
                "result": [
                    {
                        "stream": {"service": "fl_node", "node_index": "0"},
                        "values": [
                            ["1700000000000000000", "training round 1"],
                            ["1700000001000000000", "training round 2"],
                        ],
                    }
                ]
            }
        }
        entries = _extract_entries(body)
        assert len(entries) == 2
        assert entries[0]["service"] == "fl_node"
        assert entries[0]["node_id"] == "0"
        assert entries[0]["line"] == "training round 1"

    def test_entries_sorted_by_timestamp(self):
        body = {
            "data": {
                "result": [
                    {
                        "stream": {"service": "ttp"},
                        "values": [
                            ["1700000002000000000", "second"],
                            ["1700000000000000000", "first"],
                        ],
                    }
                ]
            }
        }
        entries = _extract_entries(body)
        assert entries[0]["line"] == "first"
        assert entries[1]["line"] == "second"

    def test_multiple_streams_merged(self):
        body = {
            "data": {
                "result": [
                    {"stream": {"service": "a"}, "values": [["1700000000000000000", "from a"]]},
                    {"stream": {"service": "b"}, "values": [["1700000001000000000", "from b"]]},
                ]
            }
        }
        entries = _extract_entries(body)
        assert len(entries) == 2
        assert entries[0]["service"] == "a"
        assert entries[1]["service"] == "b"


class TestFormatEntries:
    def test_plain_format(self):
        entries = [
            {"timestamp": "2025-01-01T00:00:00Z", "service": "fl_node", "node_id": "0", "line": "hello"},
        ]
        output = format_entries(entries)
        assert "[2025-01-01T00:00:00Z]" in output
        assert "[fl_node]" in output
        assert "hello" in output

    def test_json_format(self):
        entries = [{"timestamp": "t1", "service": "s", "node_id": "", "line": "msg"}]
        output = format_entries(entries, as_json=True)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert parsed[0]["line"] == "msg"

    def test_empty_entries(self):
        assert format_entries([]) == ""
        assert format_entries([], as_json=True) == "[]"
