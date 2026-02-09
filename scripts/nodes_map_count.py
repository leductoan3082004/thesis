#!/usr/bin/env python3
"""Helper script to count nodes defined in a hierarchical nodes map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nodes_map", type=Path, help="Path to nodes-map JSON file")
    return parser.parse_args()


def _count_nodes(payload: object) -> int:
    if isinstance(payload, list):
        return sum(_count_nodes(item) for item in payload)
    if isinstance(payload, dict):
        total = 0
        for key, value in payload.items():
            if key == "nodes":
                if not isinstance(value, list):
                    raise SystemExit(f"'nodes' entries must be lists (found {type(value)!r})")
                total += len(value)
            else:
                total += _count_nodes(value)
        return total
    return 0


def main() -> None:
    args = parse_args()
    if not args.nodes_map.exists():
        raise SystemExit(f"Nodes map file not found: {args.nodes_map}")
    try:
        data = json.loads(args.nodes_map.read_text())
    except json.JSONDecodeError as exc:  # noqa: TRY003
        raise SystemExit(f"Invalid JSON in nodes map {args.nodes_map}: {exc}") from exc
    total = _count_nodes(data)
    if total <= 0:
        raise SystemExit(f"Nodes map {args.nodes_map} does not include any node entries.")
    print(total)


if __name__ == "__main__":
    main()
