"""Generate per-node isolated process-runtime layout and configuration files."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
from secure_aggregation.topology import generate_preliminary_topology
from secure_aggregation.state.nodes_map import load_nodes_map

NODE_TEMPLATE_PATH = ROOT_DIR / "config" / "node.config.template.json"
NODES_MAP_FILE = ROOT_DIR / "config" / "nodes-map.json"
SYSTEM_CONFIG_FILENAME = "system-config.json"
SYSTEM_CONFIG_ENV_VAR = "SYSTEM_CONFIG_PATH"
DEFAULT_IPFS_PROCESS_CONFIG = ROOT_DIR / "config" / "ipfs-process.json"


@dataclass(frozen=True)
class IPFSTarget:
    name: str
    api_url: str


# ---------------------------------------------------------------------------
# Template and config loading
# ---------------------------------------------------------------------------


def load_node_template() -> Dict[str, Any]:
    if not NODE_TEMPLATE_PATH.exists():
        raise SystemExit(f"Missing node template at {NODE_TEMPLATE_PATH}")
    return json.loads(NODE_TEMPLATE_PATH.read_text())


def resolve_system_config_path(cli_path: Optional[Path]) -> Path:
    if cli_path:
        p = cli_path if cli_path.is_absolute() else ROOT_DIR / cli_path
        return p.resolve()
    env_value = os.getenv(SYSTEM_CONFIG_ENV_VAR)
    if env_value:
        env_path = Path(env_value)
        if not env_path.is_absolute():
            env_path = (ROOT_DIR / env_path).resolve()
        return env_path
    return (ROOT_DIR / "config" / SYSTEM_CONFIG_FILENAME).resolve()


def _load_system_config_data(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"System config not found at {path}. "
            "Pass --nodes or create the file with hierarchy/roster settings.",
        )
    return json.loads(path.read_text())


def extract_scope_names(config_data: Dict[str, Any]) -> List[str]:
    levels = config_data.get("hierarchy_levels")
    if not isinstance(levels, list):
        return ["state"]

    def _order(level: Dict[str, Any]) -> int:
        try:
            return int(level.get("scope_index") or level.get("scopeIndex"))
        except (TypeError, ValueError):
            return sys.maxsize

    ordered = sorted(levels, key=_order)
    scopes: List[str] = []
    seen: set[str] = set()
    for level in ordered:
        scope_name = level.get("scope_name") or level.get("scopeName")
        if not scope_name:
            continue
        scope_str = str(scope_name)
        if scope_str in seen:
            continue
        seen.add(scope_str)
        scopes.append(scope_str)
    return scopes or ["state"]


# ---------------------------------------------------------------------------
# Node count determination
# ---------------------------------------------------------------------------


def determine_node_count(
    cli_nodes: Optional[int],
    system_config_path: Path,
    config_data: Optional[Dict[str, Any]] = None,
) -> int:
    if cli_nodes is not None:
        if cli_nodes < 1:
            raise SystemExit("Number of nodes must be >= 1")
        return cli_nodes
    if config_data is None:
        config_data = _load_system_config_data(system_config_path)
    candidate = config_data.get("number_of_nodes")
    if candidate is None:
        deployment = config_data.get("deployment")
        if isinstance(deployment, dict):
            candidate = deployment.get("number_of_nodes")
    if candidate is None:
        raise SystemExit(
            f"number_of_nodes not found in {system_config_path}. "
            "Specify --nodes or add the field to the system config.",
        )
    try:
        count = int(candidate)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid number_of_nodes={candidate!r}; expected integer.") from exc
    if count < 1:
        raise SystemExit(f"number_of_nodes must be >= 1 (found {count}).")
    return count


# ---------------------------------------------------------------------------
# Nodes-map and assignment helpers
# ---------------------------------------------------------------------------


def _normalize_trainer_id(raw: Optional[str], seq_index: int) -> str:
    base = (raw or "").strip()
    if base:
        match = re.search(r"(\d+)(?!.*\d)", base)
        if match:
            number = int(match.group(1))
            return f"trainer-node-{number:03d}"
        sanitized = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
        if sanitized:
            return sanitized
    return f"trainer-node-{seq_index + 1:03d}"


def load_nodes_map_assignments(
    nodes_map_path: Optional[Path],
    scope_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if nodes_map_path is None:
        return []
    if not nodes_map_path.exists():
        raise SystemExit(f"Nodes map file not found: {nodes_map_path}")
    scope_chain = [name.lower() for name in reversed(scope_names or ["state"])]
    if not scope_chain:
        scope_chain = ["state"]
    try:
        metadata = load_nodes_map(nodes_map_path, scope_chain)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    lowest_scope = scope_names[0] if scope_names else scope_chain[-1]
    roster = metadata.rosters.get(lowest_scope.lower())
    if not roster:
        raise SystemExit(
            f"Nodes map {nodes_map_path} must define at least one '{lowest_scope}' entry."
        )
    assignments: List[Dict[str, Any]] = []
    next_index = 0
    for scope_id, node_aliases in roster.items():
        for alias in node_aliases:
            alias_str = str(alias).strip()
            if not alias_str:
                raise SystemExit(f"Invalid node alias under scope {scope_id!r}: {alias!r}")
            trainer_id = _normalize_trainer_id(alias_str, next_index)
            membership = metadata.memberships.get(alias_str, {})
            scope_map: Dict[str, str] = {}
            for scope in scope_names:
                scope_value = membership.get(scope.lower())
                if scope_value:
                    scope_map[scope] = scope_value
            state_identifier = scope_map.get("state") or scope_map.get(lowest_scope.lower()) or scope_id
            assignments.append(
                {
                    "node_index": next_index,
                    "state_id": str(state_identifier),
                    "trainer_id": trainer_id,
                    "scopes": scope_map,
                }
            )
            next_index += 1
    return assignments


# ---------------------------------------------------------------------------
# IPFS target loading (process-mode only)
# ---------------------------------------------------------------------------


def load_process_ipfs_targets(config_path: Path) -> List[IPFSTarget]:
    if not config_path.exists():
        raise SystemExit(f"IPFS process config not found at {config_path}")
    data = json.loads(config_path.read_text())
    nodes = data.get("nodes") or []
    if not nodes:
        raise SystemExit(f"IPFS process config {config_path} must define at least one node entry.")
    targets: List[IPFSTarget] = []
    for idx, node in enumerate(nodes):
        name = str(node.get("name") or f"ipfs-process-{idx + 1}")
        api_port = node.get("api_port")
        if api_port is None:
            raise SystemExit(f"IPFS process entry '{name}' missing required field 'api_port'.")
        client_host = str(node.get("client_host") or node.get("api_host") or "127.0.0.1").strip()
        scheme = str(node.get("api_scheme") or "http").strip() or "http"
        api_url = str(node.get("api_url") or "").strip()
        if not api_url:
            api_url = f"{scheme}://{client_host}:{int(api_port)}"
        targets.append(IPFSTarget(name=name, api_url=api_url))
    return targets


def _select_ipfs_target(node_index: int, targets: List[IPFSTarget]) -> IPFSTarget:
    if not targets:
        raise SystemExit("No IPFS endpoints defined.")
    return targets[node_index % len(targets)]


# ---------------------------------------------------------------------------
# Config application helpers
# ---------------------------------------------------------------------------


def _apply_ipfs_distribution(
    ipfs_target: IPFSTarget,
    config: Dict[str, Any],
    all_targets: List[IPFSTarget],
) -> None:
    inter_cluster = config.setdefault("inter_cluster", {})
    ipfs_section = inter_cluster.setdefault("ipfs", {})
    ipfs_section["api_url"] = ipfs_target.api_url
    replicas = [t.api_url for t in all_targets if t is not ipfs_target]
    if replicas:
        ipfs_section["replica_api_urls"] = replicas
    elif "replica_api_urls" in ipfs_section:
        del ipfs_section["replica_api_urls"]


def _apply_blockchain_identity(
    node_index: int,
    config: Dict[str, Any],
    identity_override: Optional[str] = None,
) -> None:
    suffix = f"{node_index + 1:03d}"
    identity = identity_override or f"trainer-node-{suffix}"
    inter_cluster = config.setdefault("inter_cluster", {})
    blockchain = inter_cluster.setdefault("blockchain", {})
    blockchain["identity"] = identity
    blockchain["private_key_path"] = f"config/keys/{identity}_sk.pem"
    blockchain["state_path"] = f"data/blockchain/{identity}.json"


# ---------------------------------------------------------------------------
# Process-layout generation (the main entry point)
# ---------------------------------------------------------------------------


def generate_process_layout(
    *,
    num_nodes: Optional[int] = None,
    runtime_dir: Path,
    nodes_map_path: Optional[Path] = None,
    system_config_path: Optional[Path] = None,
    clique_size: int = 3,
    ipfs_config_path: Path = DEFAULT_IPFS_PROCESS_CONFIG,
    gateway_url: str = "http://127.0.0.1:9000",
    copy_data: bool = False,
    ttp_port: int = 50051,
    base_node_port: int = 51000,
    base_metrics_port: int = 61000,
) -> Dict[str, Any]:
    """Create the full process-runtime layout and return a summary dict.

    Side-effects:
    - creates ``runtime_dir/nodes/node_<i>/`` subdirs with config, data symlink, logs, etc.
    - writes ``runtime_dir/datasets.json`` with absolute host paths
    - writes ``runtime_dir/topology.json``
    - writes per-node JSON config files
    """
    sys_cfg_path = resolve_system_config_path(system_config_path)
    sys_cfg_data: Dict[str, Any] = {}
    if sys_cfg_path.exists():
        sys_cfg_data = _load_system_config_data(sys_cfg_path)
    scope_names = extract_scope_names(sys_cfg_data) if sys_cfg_data else ["state"]
    primary_scope = scope_names[0] if scope_names else "state"

    # Skip nodes-map auto-discovery when an explicit node count is provided.
    resolved_map = _resolve_nodes_map(nodes_map_path) if num_nodes is None else nodes_map_path
    assignments = load_nodes_map_assignments(resolved_map, scope_names)

    if assignments:
        node_count = len(assignments)
    else:
        node_count = determine_node_count(num_nodes, sys_cfg_path, sys_cfg_data or None)
        assignments = [
            {
                "node_index": i,
                "trainer_id": _normalize_trainer_id(None, i),
                "state_id": "",
                "scopes": {},
            }
            for i in range(node_count)
        ]

    ipfs_targets = load_process_ipfs_targets(ipfs_config_path)
    template = load_node_template()

    # Ensure runtime directories exist.
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_dir = runtime_dir / "config"
    runtime_config_dir.mkdir(parents=True, exist_ok=True)

    # Ensure nodes find the shared system configuration next to their configs.
    destination = runtime_config_dir / SYSTEM_CONFIG_FILENAME
    if sys_cfg_path.exists():
        shutil.copy2(sys_cfg_path, destination)
    elif destination.exists():
        destination.unlink()

    nodes_map_dest = runtime_config_dir / "nodes-map.json"
    nodes_map_source = resolved_map
    if nodes_map_source and nodes_map_source.exists():
        shutil.copy2(nodes_map_source, nodes_map_dest)
    elif nodes_map_dest.exists():
        nodes_map_dest.unlink()

    # Write process-mode datasets.json with absolute host paths.
    datasets_path = runtime_dir / "datasets.json"
    _write_datasets_json(datasets_path)

    topology_path = runtime_dir / "topology.json"
    topology_data = generate_preliminary_topology(node_count, clique_size)
    topology_path.write_text(json.dumps(topology_data, indent=2) + "\n")

    # Also write to config/topology.json so TTP can find it.
    config_topology = ROOT_DIR / "config" / "topology.json"
    config_topology.parent.mkdir(parents=True, exist_ok=True)
    config_topology.write_text(json.dumps(topology_data, indent=2) + "\n")

    node_configs: List[Path] = []
    nodes_config_dir = runtime_dir / "config" / "nodes"
    nodes_config_dir.mkdir(parents=True, exist_ok=True)

    for assignment in assignments:
        idx = assignment["node_index"]
        trainer_id = assignment["trainer_id"]
        state_id = assignment.get("state_id") or ""
        scopes = dict(assignment.get("scopes") or {})

        node_dir = runtime_dir / "nodes" / f"node_{idx}"
        for subdir in ("config", "logs", "checkpoints", "pids"):
            (node_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Dataset: symlink by default, copy if requested.
        data_link = node_dir / "data"
        data_source = ROOT_DIR / "data"
        if data_link.is_symlink():
            data_link.unlink()
        elif data_link.is_dir():
            import shutil as _shutil
            _shutil.rmtree(data_link)
        elif data_link.exists():
            data_link.unlink()
        if copy_data and data_source.exists():
            shutil.copytree(data_source, data_link)
        elif data_source.exists():
            data_link.symlink_to(data_source)
        else:
            data_link.mkdir(parents=True, exist_ok=True)

        service_port = base_node_port + idx
        metrics_port = base_metrics_port + idx

        config = deepcopy(template)
        config["node_id"] = trainer_id
        config["trainer_id"] = trainer_id
        config["network_host"] = "127.0.0.1"
        config["port"] = service_port
        config["metrics_port"] = metrics_port
        config["ttp_address"] = f"127.0.0.1:{ttp_port}"

        if state_id:
            config["state_id"] = state_id
        else:
            fallback = scopes.get("state") or scopes.get(primary_scope)
            config["state_id"] = fallback or "default"

        if scopes:
            config["scope_assignments"] = scopes
            for scope_key, scope_value in scopes.items():
                config[scope_key] = scope_value

        if scope_names:
            config["scope_hierarchy"] = list(scope_names)

        # Dataset section: point to process-mode datasets.json.
        dataset_section = config.setdefault("dataset", {})
        dataset_section["config_path"] = str(datasets_path)

        # Inter-cluster: IPFS distribution.
        ipfs_target = _select_ipfs_target(idx, ipfs_targets)
        _apply_ipfs_distribution(ipfs_target, config, ipfs_targets)

        # Inter-cluster: blockchain identity.
        _apply_blockchain_identity(idx, config, identity_override=trainer_id)

        # Inter-cluster: topology file and gateway URL.  Point each node to
        # the runtime copy so nodes don't share a repo-level file.
        inter_cluster = config.setdefault("inter_cluster", {})
        inter_cluster["topology_file"] = str(topology_path)
        blockchain = inter_cluster.setdefault("blockchain", {})
        blockchain["gateway_url"] = gateway_url

        # Write per-node config.
        config_path = nodes_config_dir / f"node_{idx}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        node_configs.append(config_path)

    print(f"Generated process layout for {node_count} nodes -> {runtime_dir}")
    return {
        "node_count": node_count,
        "runtime_dir": str(runtime_dir),
        "node_configs": [str(p) for p in node_configs],
        "topology_path": str(topology_path),
        "datasets_path": str(datasets_path),
        "assignments": assignments,
        "clique_size": clique_size,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_nodes_map(cli_path: Optional[Path]) -> Optional[Path]:
    if cli_path:
        p = cli_path if cli_path.is_absolute() else ROOT_DIR / cli_path
        return p.resolve()
    if NODES_MAP_FILE.exists():
        return NODES_MAP_FILE
    return None


def _write_datasets_json(dest: Path) -> None:
    """Write a datasets.json that references absolute host paths."""
    data_root = str(ROOT_DIR / "data")
    datasets = {
        "mnist": {
            "type": "torchvision",
            "class": "MNIST",
            "root": data_root,
            "num_classes": 10,
            "input_shape": [1, 28, 28],
        },
        "fashion_mnist": {
            "type": "torchvision",
            "class": "FashionMNIST",
            "root": data_root,
            "num_classes": 10,
            "input_shape": [1, 28, 28],
        },
        "cifar10": {
            "type": "torchvision",
            "class": "CIFAR10",
            "root": data_root,
            "num_classes": 10,
            "input_shape": [3, 32, 32],
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(datasets, indent=2) + "\n")
