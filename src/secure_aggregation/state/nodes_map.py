"""Helpers for parsing the hierarchical nodes map."""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


@dataclass
class NodesMapMetadata:
    """Represents scope/node relationships derived from nodes-map.json."""

    rosters: Dict[str, Dict[str, List[str]]]
    child_map: Dict[Tuple[str, str], Dict[str, List[str]]]
    memberships: Dict[str, Dict[str, str]]


def _candidate_keys(scope_name: str) -> List[str]:
    base = str(scope_name or "").strip()
    lower = base.lower()
    plural = f"{base}s"
    plural_lower = f"{lower}s"
    return [base, lower, plural, plural_lower]


def _extract_scope_entries(container: Any, scope_name: str) -> List[Mapping[str, Any]]:
    """Return a list of mapping entries for the requested scope name."""
    if isinstance(container, list):
        return [entry for entry in container if isinstance(entry, Mapping)]
    if not isinstance(container, Mapping):
        return []
    for key in _candidate_keys(scope_name):
        value = container.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, Mapping)]
    return []


def _extract_scope_id(entry: Mapping[str, Any], scope_name: str, fallback: Optional[str]) -> Optional[str]:
    """Derive the identifier for a scope entry."""
    candidates = [
        f"{scope_name}_id",
        f"{scope_name.lower()}_id",
        "scope_id",
        "id",
    ]
    for key in candidates:
        value = entry.get(key)
        if value:
            value_str = str(value).strip()
            if value_str:
                return value_str
    return str(fallback) if fallback else None


def _normalize_nodes(value: Any) -> List[str]:
    """Normalize a nodes list."""
    if not isinstance(value, Iterable):
        return []
    nodes: List[str] = []
    for item in value:
        node_id = str(item).strip()
        if node_id:
            nodes.append(node_id)
    return nodes


def _empty_metadata(scope_order: Sequence[str]) -> NodesMapMetadata:
    rosters = {name.lower(): OrderedDict() for name in scope_order}
    return NodesMapMetadata(rosters=rosters, child_map={}, memberships={})


def parse_nodes_map(
    data: Mapping[str, Any],
    scope_order: Sequence[str],
) -> NodesMapMetadata:
    """
    Parse a JSON mapping describing scope membership.

    Args:
        data: JSON object loaded from nodes-map.json.
        scope_order: Sequence of scope names ordered from highest (largest scope_index)
                     to lowest (closest to the cluster level).
    """
    if not scope_order:
        return NodesMapMetadata(rosters={}, child_map={}, memberships={})
    rosters: Dict[str, MutableMapping[str, List[str]]] = {
        name.lower(): OrderedDict() for name in scope_order
    }
    roster_sets: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    child_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    memberships: Dict[str, Dict[str, str]] = {}

    def assign_nodes(path: List[Tuple[str, str]], nodes: List[str]) -> None:
        if not nodes or not path:
            return
        for node_id in nodes:
            scope_members = memberships.setdefault(node_id, {})
            for scope_name, scope_id in path:
                scope_members.setdefault(scope_name, scope_id)
        for scope_name, scope_id in path:
            scope_key = scope_name.lower()
            bucket = rosters.setdefault(scope_key, OrderedDict()).setdefault(scope_id, [])
            seen = roster_sets.setdefault((scope_key, scope_id), set())
            for node_id in nodes:
                if node_id in seen:
                    continue
                bucket.append(node_id)
                seen.add(node_id)

    def visit_level(level_idx: int, container: Mapping[str, Any], ancestors: List[Tuple[str, str]]) -> None:
        scope_name = scope_order[level_idx]
        entries = _extract_scope_entries(container, scope_name)
        if not entries:
            return
        for entry in entries:
            scope_id = _extract_scope_id(entry, scope_name, None)
            if not scope_id:
                continue
            current_path = ancestors + [(scope_name.lower(), scope_id)]
            if ancestors:
                parent_scope, parent_id = ancestors[-1]
                key = (parent_scope, parent_id)
                child_scopes = child_map.setdefault(key, {})
                bucket = child_scopes.setdefault(scope_name.lower(), [])
                if scope_id not in bucket:
                    bucket.append(scope_id)
            if level_idx == len(scope_order) - 1:
                nodes = _normalize_nodes(entry.get("nodes") or [])
                if nodes:
                    assign_nodes(current_path, nodes)
            else:
                visit_level(level_idx + 1, entry, current_path)

    root_container: Mapping[str, Any]
    if isinstance(data, list):
        root_container = {scope_order[0]: data}
    else:
        root_container = data
    visit_level(0, root_container, [])
    return NodesMapMetadata(rosters=rosters, child_map=child_map, memberships=memberships)


def load_nodes_map(path: Path, scope_order: Sequence[str]) -> NodesMapMetadata:
    """Load nodes-map.json from disk and parse it."""
    if not path.exists():
        return _empty_metadata(scope_order)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # noqa: TRY003
        raise ValueError(f"Invalid JSON in nodes map {path}: {exc}") from exc
    return parse_nodes_map(data, scope_order)
