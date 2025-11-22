"""
Generate a Docker Compose file with one service per participant in a scenario.

Usage:
  python scripts/generate_compose_from_scenario.py --scenario config/scenario.sample.json --output docker/docker-compose.generated.yml
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml


VOLUME_BINDS = ["../config:/app/config", "../data:/app/data", "../logs:/app/logs", "../checkpoints:/app/checkpoints"]


def _node_service(node_id: str, scenario_path: str) -> Dict:
    return {
        "build": {"context": "..", "dockerfile": "docker/node.Dockerfile"},
        "command": ["python", "-m", "secure_aggregation.node.service"],
        "container_name": node_id,
        "environment": [f"NODE_ID={node_id}", f"SCENARIO_PATH=/app/{scenario_path}", "PORT=8000"],
        "networks": ["secureagg"],
        "volumes": VOLUME_BINDS,
    }


def generate_compose(participants: List[str], scenario_path: str) -> Dict:
    services: Dict[str, Dict] = {}
    services["ttp"] = {
        "build": {"context": "..", "dockerfile": "docker/ttp.Dockerfile"},
        "command": ["python", "-c", "print('TTP placeholder')"],
        "container_name": "ttp",
        "networks": ["secureagg"],
        "volumes": VOLUME_BINDS,
    }
    for node_id in participants:
        services[node_id] = _node_service(node_id, scenario_path)
    return {"version": "3.9", "services": services, "networks": {"secureagg": {}}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON (relative to repo root).")
    parser.add_argument("--output", required=True, help="Path to write the generated docker-compose YAML.")
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    scenario = json.loads(scenario_path.read_text())
    participants = scenario.get("participants", [])
    if not participants:
        raise SystemExit("Scenario has no participants")
    compose = generate_compose(participants, scenario_path.as_posix())
    out_path = Path(args.output)
    out_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    print(f"Wrote {out_path} for participants: {participants}")


if __name__ == "__main__":
    main()
