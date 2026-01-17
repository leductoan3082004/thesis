#!/usr/bin/env python3
"""Helper script to count nodes defined in a state-map JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_map", type=Path, help="Path to state-map JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.state_map.exists():
        raise SystemExit(f"State map file not found: {args.state_map}")
    try:
        data = json.loads(args.state_map.read_text())
    except json.JSONDecodeError as exc:  # noqa: TRY003
        raise SystemExit(f"Invalid JSON in state map {args.state_map}: {exc}") from exc
    states = data.get("states") or []
    if not states:
        raise SystemExit(f"State map {args.state_map} must contain at least one state entry.")

    total = 0
    for entry in states:
        nodes = entry.get("nodes")
        if nodes:
            if not isinstance(nodes, list):
                raise SystemExit(f"'nodes' for state entry must be a list: {entry}")
            total += len(nodes)
            continue
        count = entry.get("count")
        if count is None:
            raise SystemExit(f"State entry missing 'nodes' and 'count': {entry}")
        try:
            count_val = int(count)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid count {count!r} for entry {entry}") from exc
        if count_val <= 0:
            raise SystemExit(f"State entry must have count >= 1 (found {count_val}): {entry}")
        total += count_val

    print(total)


if __name__ == "__main__":
    main()
