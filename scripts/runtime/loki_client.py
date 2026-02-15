"""Loki HTTP API client for log querying from the CLI."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

DEFAULT_LOKI_URL = "http://localhost:3100"


def build_logql(
    *,
    service: Optional[str] = None,
    node_id: Optional[str] = None,
    level: Optional[str] = None,
    contains: Optional[str] = None,
) -> str:
    """Construct a LogQL filter expression from the given parameters."""
    matchers: List[str] = []
    if service:
        matchers.append(f'service="{service}"')
    if node_id:
        matchers.append(f'node_id="{node_id}"')
    if not matchers:
        matchers.append('service=~".+"')

    query = "{" + ", ".join(matchers) + "}"

    filters: List[str] = []
    if level:
        filters.append(f'|~ "(?i){level}"')
    if contains:
        filters.append(f'|~ "{contains}"')
    if filters:
        query += " " + " ".join(filters)

    return query


def query_range(
    query: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100,
    loki_url: str = DEFAULT_LOKI_URL,
) -> List[Dict[str, Any]]:
    """Execute a range query against Loki and return parsed log entries."""
    now = datetime.now(timezone.utc)
    if start:
        start_ts = _parse_time(start, now)
    else:
        start_ts = now - timedelta(hours=1)
    if end:
        end_ts = _parse_time(end, now)
    else:
        end_ts = now

    params = urllib_parse.urlencode(
        {
            "query": query,
            "start": _to_nanoseconds(start_ts),
            "end": _to_nanoseconds(end_ts),
            "limit": str(limit),
            "direction": "backward",
        }
    )
    url = f"{loki_url.rstrip('/')}/loki/api/v1/query_range?{params}"

    try:
        with urllib_request.urlopen(url, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except urllib_error.URLError as exc:
        raise SystemExit(f"Failed to query Loki at {loki_url}: {exc}") from exc

    return _extract_entries(body)


def tail(
    query: str,
    *,
    loki_url: str = DEFAULT_LOKI_URL,
    delay_for: int = 0,
) -> None:
    """Stream logs via Loki tail endpoint using polling (WebSocket-free fallback)."""
    start_ts = datetime.now(timezone.utc)
    print(f"Tailing logs (Ctrl+C to stop)...\n")
    try:
        while True:
            end_ts = datetime.now(timezone.utc)
            params = urllib_parse.urlencode(
                {
                    "query": query,
                    "start": _to_nanoseconds(start_ts),
                    "end": _to_nanoseconds(end_ts),
                    "limit": "50",
                    "direction": "forward",
                }
            )
            url = f"{loki_url.rstrip('/')}/loki/api/v1/query_range?{params}"
            try:
                with urllib_request.urlopen(url, timeout=10) as resp:
                    body = json.loads(resp.read().decode())
                entries = _extract_entries(body)
                for entry in entries:
                    _print_entry(entry)
                if entries:
                    # Move start past the last entry to avoid duplicates.
                    last_ts = entries[-1].get("timestamp_ns", "")
                    if last_ts:
                        start_ts = datetime.fromtimestamp(int(last_ts) / 1e9, tz=timezone.utc) + timedelta(microseconds=1)
                    else:
                        start_ts = end_ts
                else:
                    start_ts = end_ts
            except (urllib_error.URLError, OSError):
                pass
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopped tailing.")


def format_entries(entries: List[Dict[str, Any]], as_json: bool = False) -> str:
    if as_json:
        return json.dumps(entries, indent=2)
    lines: List[str] = []
    for entry in entries:
        _append_formatted(lines, entry)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_entries(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    data = body.get("data", {})
    for stream in data.get("result", []):
        labels = stream.get("stream", {})
        for ts, line in stream.get("values", []):
            results.append(
                {
                    "timestamp_ns": ts,
                    "timestamp": _ns_to_iso(ts),
                    "service": labels.get("service", ""),
                    "node_id": labels.get("node_id", labels.get("node_index", "")),
                    "line": line,
                }
            )
    results.sort(key=lambda e: e["timestamp_ns"])
    return results


def _print_entry(entry: Dict[str, Any]) -> None:
    ts = entry.get("timestamp", "")
    svc = entry.get("service", "")
    node = entry.get("node_id", "")
    line = entry.get("line", "")
    prefix = f"[{ts}]"
    if svc:
        prefix += f" [{svc}]"
    if node:
        prefix += f" [{node}]"
    print(f"{prefix} {line}")


def _append_formatted(lines: List[str], entry: Dict[str, Any]) -> None:
    ts = entry.get("timestamp", "")
    svc = entry.get("service", "")
    node = entry.get("node_id", "")
    line = entry.get("line", "")
    prefix = f"[{ts}]"
    if svc:
        prefix += f" [{svc}]"
    if node:
        prefix += f" [{node}]"
    lines.append(f"{prefix} {line}")


def _parse_time(value: str, now: datetime) -> datetime:
    """Parse a duration like '30m', '1h', '2d' or an ISO timestamp."""
    value = value.strip()
    if value.endswith("m"):
        return now - timedelta(minutes=int(value[:-1]))
    if value.endswith("h"):
        return now - timedelta(hours=int(value[:-1]))
    if value.endswith("d"):
        return now - timedelta(days=int(value[:-1]))
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _to_nanoseconds(dt: datetime) -> str:
    return str(int(dt.timestamp() * 1e9))


def _ns_to_iso(ns_str: str) -> str:
    try:
        ts = int(ns_str) / 1e9
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except (ValueError, OSError):
        return ns_str
