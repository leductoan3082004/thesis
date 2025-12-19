#!/usr/bin/env python3
"""Generate per-node configs/Compose file and launch Docker with a dynamic node count."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'PyYAML'. Install project requirements first, e.g. "
        "`python3 -m pip install -e '.[mnist]'`.",
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COMPOSE_TEMPLATE = ROOT_DIR / "docker" / "docker-compose.yml"
DEFAULT_COMPOSE_OUTPUT = ROOT_DIR / "docker" / "docker-compose.auto.yml"
NODE_TEMPLATE_PATH = ROOT_DIR / "config" / "node.config.template.json"
NODES_DIR = ROOT_DIR / "config" / "nodes"
SYSTEM_CONFIG_FILENAME = "system-config.json"
SYSTEM_CONFIG_ENV_VAR = "SYSTEM_CONFIG_PATH"


DEFAULT_NODE_SERVICE: Dict[str, Any] = {
    "build": {"context": "..", "dockerfile": "docker/node.Dockerfile"},
    "networks": ["secureagg"],
    "command": [
        "sh",
        "-c",
        "sleep 10 && python -m secure_aggregation.communication.node_service --config /app/config/nodes/node_0.json",
    ],
    "depends_on": {
        "ttp": {"condition": "service_healthy"},
        "registry": {"condition": "service_healthy"},
    },
    "env_file": ["../.env"],
    "environment": ["PYTHONUNBUFFERED=1", "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"],
    "volumes": [
        "../config:/app/config",
        "../data:/app/data",
        "../logs:/app/logs",
        "../checkpoints:/app/checkpoints",
        "../src:/app/src",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        "--node",
        dest="nodes",
        type=int,
        help="Number of node containers/configs to launch (overrides system config).",
    )
    parser.add_argument(
        "--compose-template",
        type=Path,
        default=DEFAULT_COMPOSE_TEMPLATE,
        help="Path to the base docker-compose template.",
    )
    parser.add_argument(
        "--compose-output",
        type=Path,
        default=DEFAULT_COMPOSE_OUTPUT,
        help="Path to write the generated docker-compose file.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate configs/compose file but skip running docker compose.",
    )
    parser.add_argument(
        "--detach",
        "-d",
        action="store_true",
        help="Run docker compose in detached mode.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip passing --build to docker compose up.",
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        help="Override path to system-config.json when deriving the node count.",
    )
    return parser.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def load_node_template() -> Dict[str, Any]:
    if not NODE_TEMPLATE_PATH.exists():
        raise SystemExit(f"Missing node template at {NODE_TEMPLATE_PATH}")
    return json.loads(NODE_TEMPLATE_PATH.read_text())


def _ipfs_sort_key(service_name: str) -> Tuple[int, str]:
    """Sort IPFS service names numerically when possible."""
    try:
        suffix = service_name.rsplit("-", 1)[-1]
        return int(suffix), service_name
    except ValueError:
        return sys.maxsize, service_name


def _select_ipfs_service(node_index: int, services: List[str]) -> str:
    if not services:
        raise SystemExit("No IPFS services defined in docker-compose template.")
    return services[node_index % len(services)]


def _apply_ipfs_distribution(ipfs_service: str, config: Dict[str, Any]) -> None:
    inter_cluster = config.setdefault("inter_cluster", {})
    ipfs_section = inter_cluster.setdefault("ipfs", {})
    ipfs_section["api_url"] = f"http://{ipfs_service}:5001"


def _apply_blockchain_identity(node_index: int, config: Dict[str, Any]) -> None:
    suffix = f"{node_index + 1:03d}"
    identity = f"trainer-node-{suffix}"
    inter_cluster = config.setdefault("inter_cluster", {})
    blockchain = inter_cluster.setdefault("blockchain", {})
    blockchain["identity"] = identity
    blockchain["private_key_path"] = f"config/keys/{identity}_sk.pem"
    blockchain["state_path"] = f"data/blockchain/{identity}.json"


def _ensure_blockchain_dir() -> None:
    blockchain_dir = ROOT_DIR / "data" / "blockchain"
    blockchain_dir.mkdir(parents=True, exist_ok=True)


def resolve_system_config_path(cli_path: Optional[Path]) -> Path:
    if cli_path:
        return _resolve_repo_path(cli_path).resolve()
    env_value = os.getenv(SYSTEM_CONFIG_ENV_VAR)
    if env_value:
        env_path = Path(env_value)
        if not env_path.is_absolute():
            env_path = (ROOT_DIR / env_path).resolve()
        return env_path
    return (ROOT_DIR / "config" / SYSTEM_CONFIG_FILENAME).resolve()


def _extract_node_count(config_data: Dict[str, Any]) -> Optional[int]:
    candidate = config_data.get("number_of_nodes")
    if candidate is not None:
        return candidate
    deployment = config_data.get("deployment")
    if isinstance(deployment, dict):
        return deployment.get("number_of_nodes")
    return None


def determine_node_count(cli_nodes: Optional[int], system_config_path: Path) -> int:
    if cli_nodes is not None:
        if cli_nodes < 1:
            raise SystemExit("Number of nodes must be >= 1")
        return cli_nodes

    if not system_config_path.exists():
        raise SystemExit(
            f"System config not found at {system_config_path}. "
            "Pass --nodes or create the file with number_of_nodes defined.",
        )

    try:
        config_data = json.loads(system_config_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in system config {system_config_path}: {exc}") from exc

    node_count = _extract_node_count(config_data)
    if node_count is None:
        raise SystemExit(
            f"number_of_nodes not found in {system_config_path}. "
            "Specify --nodes or add the field to the system config.",
        )

    try:
        node_count_int = int(node_count)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid number_of_nodes={node_count!r} in {system_config_path}; expected integer.",
        ) from exc

    if node_count_int < 1:
        raise SystemExit(f"number_of_nodes must be >= 1 (found {node_count_int}).")

    return node_count_int


def load_compose_template(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Docker compose template not found at {path}")
    return yaml.safe_load(path.read_text())


def _extract_ipfs_services(compose_data: Dict[str, Any]) -> List[str]:
    services = compose_data.get("services", {})
    ipfs_services = [name for name in services if name.startswith("ipfs-node")]
    return sorted(ipfs_services, key=_ipfs_sort_key)


def _locate_node_template(compose_data: Dict[str, Any]) -> Dict[str, Any]:
    services = compose_data.get("services", {})
    for name, service in services.items():
        if name.startswith("node_"):
            return deepcopy(service)
    return deepcopy(DEFAULT_NODE_SERVICE)


def _merge_node_service(
    node_name: str,
    base_template: Dict[str, Any],
    ipfs_service: str,
) -> Dict[str, Any]:
    service = deepcopy(base_template)
    service["container_name"] = node_name
    service["command"] = [
        "sh",
        "-c",
        f"sleep 10 && python -m secure_aggregation.communication.node_service --config /app/config/nodes/{node_name}.json",
    ]
    depends_on = {k: v for k, v in service.get("depends_on", {}).items() if not k.startswith("ipfs-node")}
    depends_on.setdefault("ttp", {"condition": "service_healthy"})
    depends_on.setdefault("registry", {"condition": "service_healthy"})
    depends_on[ipfs_service] = {"condition": "service_healthy"}
    service["depends_on"] = depends_on
    service.setdefault("build", DEFAULT_NODE_SERVICE["build"])
    service.setdefault("networks", DEFAULT_NODE_SERVICE["networks"])
    service.setdefault("env_file", DEFAULT_NODE_SERVICE["env_file"])
    service.setdefault("environment", DEFAULT_NODE_SERVICE["environment"])
    service.setdefault("volumes", DEFAULT_NODE_SERVICE["volumes"])
    return service


def update_compose_file(
    compose_data: Dict[str, Any],
    num_nodes: int,
    ipfs_services: List[str],
) -> Dict[str, Any]:
    services = compose_data.setdefault("services", {})
    base_node_service = _locate_node_template(compose_data)
    # Remove any pre-existing node definitions.
    for key in list(services.keys()):
        if key.startswith("node_"):
            services.pop(key)
    for idx in range(num_nodes):
        node_name = f"node_{idx}"
        ipfs_service = _select_ipfs_service(idx, ipfs_services)
        services[node_name] = _merge_node_service(node_name, base_node_service, ipfs_service)
    return compose_data


def write_compose_file(data: Dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def run_docker_compose(compose_path: Path, detach: bool, build: bool) -> int:
    cmd = ["docker", "compose", "-f", compose_path.name, "up"]
    if build:
        cmd.append("--build")
    if detach:
        cmd.append("-d")
    result = subprocess.run(cmd, cwd=compose_path.parent, check=False)
    return result.returncode


def main() -> None:
    args = parse_args()

    compose_template_path = _resolve_repo_path(args.compose_template)
    compose_output_path = _resolve_repo_path(args.compose_output)
    system_config_path = resolve_system_config_path(args.system_config)
    node_count = determine_node_count(args.nodes, system_config_path)

    template_config = load_node_template()
    compose_template = load_compose_template(compose_template_path)
    ipfs_services = _extract_ipfs_services(compose_template)

    _ensure_blockchain_dir()
    NODES_DIR.mkdir(parents=True, exist_ok=True)
    # Generate configs with the correct IPFS/blockchain layout.
    for idx in range(node_count):
        config = deepcopy(template_config)
        node_id = f"node_{idx}"
        config["node_id"] = node_id
        ipfs_service = _select_ipfs_service(idx, ipfs_services)
        _apply_ipfs_distribution(ipfs_service, config)
        _apply_blockchain_identity(idx, config)
        config_path = NODES_DIR / f"{node_id}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    updated_compose = update_compose_file(compose_template, node_count, ipfs_services)
    compose_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_compose_file(updated_compose, compose_output_path)
    print(f"Generated docker compose file with {node_count} nodes -> {compose_output_path}")
    print(f"(node count source: {'--nodes CLI' if args.nodes is not None else system_config_path})")

    if args.generate_only:
        print("Skipped docker compose up (generate-only mode).")
        print(f"Run: (cd docker && docker compose -f {compose_output_path.name} up --build)")
        return

    build = not args.no_build
    return_code = run_docker_compose(compose_output_path, args.detach, build)
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
