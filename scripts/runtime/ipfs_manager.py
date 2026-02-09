"""Manage IPFS Kubo daemons as host processes."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, IO, List, Optional, Tuple

from scripts.runtime.process_registry import ManagedProcess

ROOT_DIR = Path(__file__).resolve().parents[2]
BOOTSTRAP_SCRIPT = ROOT_DIR / "docker" / "ipfs-bootstrap.sh"
DEFAULT_CONFIG = ROOT_DIR / "config" / "ipfs-process.json"


@dataclass
class IPFSNode:
    name: str
    repo: Path
    api_host: str
    api_port: int
    gateway_host: str
    gateway_port: int
    swarm_host: str
    swarm_port: int
    advertise_host: str
    advertise_proto: str


@dataclass
class IPFSLayout:
    peer_registry: Path
    nodes: List[IPFSNode]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_layout(config_path: Path) -> IPFSLayout:
    if not config_path.exists():
        raise SystemExit(f"IPFS process config not found at {config_path}")
    raw = json.loads(config_path.read_text())
    peer_registry = _resolve_path(raw.get("peer_registry", "data/ipfs/peers"))
    nodes_data = raw.get("nodes") or []
    if not nodes_data:
        raise SystemExit(f"{config_path} does not define any IPFS nodes.")
    nodes: List[IPFSNode] = []
    for idx, entry in enumerate(nodes_data):
        name = str(entry.get("name") or f"ipfs-process-{idx + 1}")
        repo = _resolve_path(entry.get("repo", f"data/ipfs/node-{idx + 1}"))
        api_port = int(entry.get("api_port", 0))
        gateway_port = int(entry.get("gateway_port", 0))
        swarm_port = int(entry.get("swarm_port", 0))
        if not api_port or not gateway_port or not swarm_port:
            raise SystemExit(f"IPFS node '{name}' is missing api_port, gateway_port, or swarm_port.")
        nodes.append(
            IPFSNode(
                name=name,
                repo=repo,
                api_host=str(entry.get("api_host") or "127.0.0.1"),
                api_port=api_port,
                gateway_host=str(entry.get("gateway_host") or "127.0.0.1"),
                gateway_port=gateway_port,
                swarm_host=str(entry.get("swarm_host") or "0.0.0.0"),
                swarm_port=swarm_port,
                advertise_host=str(entry.get("advertise_host") or "127.0.0.1"),
                advertise_proto=str(entry.get("advertise_proto") or "dns4"),
            )
        )
    return IPFSLayout(peer_registry=peer_registry, nodes=nodes)


def _locate_ipfs_binary(binary: str = "ipfs") -> str:
    resolved = shutil.which(binary) if not Path(binary).is_file() else str(Path(binary).resolve())
    if resolved is None:
        raise SystemExit(f"Unable to locate IPFS binary '{binary}'. Install Kubo and ensure the command is on PATH.")
    return resolved


def start_ipfs(
    config_path: Path,
    runtime_dir: Path,
    ipfs_binary: str = "ipfs",
) -> List[ManagedProcess]:
    """Launch IPFS daemons and return managed-process entries for the registry."""
    layout = load_layout(config_path)
    log_dir = runtime_dir / "logs" / "ipfs"
    log_dir.mkdir(parents=True, exist_ok=True)
    layout.peer_registry.mkdir(parents=True, exist_ok=True)
    # Clear stale peer addresses.
    for addr_file in layout.peer_registry.glob("*.addr"):
        addr_file.unlink()
    for node in layout.nodes:
        node.repo.mkdir(parents=True, exist_ok=True)

    binary = _locate_ipfs_binary(ipfs_binary)
    base_env = os.environ.copy()
    base_env["IPFS_BINARY"] = binary
    cluster_names = ",".join(n.name for n in layout.nodes)

    managed: List[ManagedProcess] = []
    for node in layout.nodes:
        log_path = log_dir / f"{node.name}.log"
        env = dict(base_env)
        env.update(
            {
                "IPFS_PATH": str(node.repo),
                "NODE_NAME": node.name,
                "IPFS_CLUSTER_NODES": cluster_names,
                "PEER_REGISTRY": str(layout.peer_registry),
                "API_HOST": node.api_host,
                "API_PORT": str(node.api_port),
                "GATEWAY_HOST": node.gateway_host,
                "GATEWAY_PORT": str(node.gateway_port),
                "SWARM_HOST": node.swarm_host,
                "SWARM_PORT": str(node.swarm_port),
                "NODE_ADVERTISE_HOST": node.advertise_host,
                "NODE_ADVERTISE_PROTO": node.advertise_proto,
            }
        )
        log_handle = log_path.open("ab")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"=== Starting {node.name} at {timestamp} ===\n".encode())
        log_handle.flush()
        proc = subprocess.Popen(
            ["/bin/sh", str(BOOTSTRAP_SCRIPT)],
            cwd=ROOT_DIR,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        log_handle.close()
        pid_file = runtime_dir / "pids" / f"{node.name}.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{proc.pid}\n")
        ports = [node.api_port, node.gateway_port, node.swarm_port]
        managed.append(
            ManagedProcess(
                name=node.name,
                pid=proc.pid,
                pid_file=str(pid_file),
                ports=ports,
                log_file=str(log_path),
                component_type="infrastructure",
            )
        )
        print(
            f"[ipfs] {node.name} -> API {node.api_host}:{node.api_port}, "
            f"Gateway {node.gateway_host}:{node.gateway_port}, "
            f"Swarm {node.swarm_host}:{node.swarm_port} (pid {proc.pid})"
        )
    return managed
