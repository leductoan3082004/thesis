#!/usr/bin/env python3
"""Launch the IPFS cluster described in config/ipfs-process.json as host processes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, IO, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "config" / "ipfs-process.json"
DEFAULT_LOG_DIR = ROOT_DIR / "logs" / "ipfs"
BOOTSTRAP_SCRIPT = ROOT_DIR / "docker" / "ipfs-bootstrap.sh"


@dataclass
class ProcessNode:
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
class ProcessLayout:
    peer_registry: Path
    nodes: List[ProcessNode]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_layout(config_path: Path) -> ProcessLayout:
    if not config_path.exists():
        raise SystemExit(f"IPFS process config not found at {config_path}")
    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse {config_path}: {exc}") from exc
    peer_registry = _resolve_path(raw.get("peer_registry", "data/ipfs/peers"))
    nodes_data = raw.get("nodes") or []
    if not nodes_data:
        raise SystemExit(f"{config_path} does not define any IPFS nodes.")
    nodes: List[ProcessNode] = []
    for idx, entry in enumerate(nodes_data):
        name = str(entry.get("name") or f"ipfs-process-{idx + 1}")
        repo = _resolve_path(entry.get("repo", f"data/ipfs/node-{idx + 1}"))
        api_port = int(entry.get("api_port", 0))
        gateway_port = int(entry.get("gateway_port", 0))
        swarm_port = int(entry.get("swarm_port", 0))
        if not api_port or not gateway_port or not swarm_port:
            raise SystemExit(f"IPFS node '{name}' is missing api_port, gateway_port, or swarm_port.")
        nodes.append(
            ProcessNode(
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
    return ProcessLayout(peer_registry=peer_registry, nodes=nodes)


def prepare_directories(layout: ProcessLayout, log_dir: Path) -> None:
    layout.peer_registry.mkdir(parents=True, exist_ok=True)
    for addr_file in layout.peer_registry.glob("*.addr"):
        addr_file.unlink()
    for node in layout.nodes:
        node.repo.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the IPFS process layout JSON (default: config/ipfs-process.json).",
    )
    parser.add_argument(
        "--ipfs-binary",
        type=str,
        default="ipfs",
        help="Path to the ipfs binary (default: found on PATH).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for per-node log files (default: logs/ipfs).",
    )
    return parser.parse_args()


def locate_ipfs(binary: str) -> str:
    resolved = shutil.which(binary) if not Path(binary).is_file() else str(Path(binary).resolve())
    if resolved is None:
        raise SystemExit(f"Unable to locate IPFS binary '{binary}'. Install Kubo and ensure the command is on PATH.")
    return resolved


def _build_env(
    base_env: Dict[str, str],
    node: ProcessNode,
    cluster_names: str,
    layout: ProcessLayout,
) -> Dict[str, str]:
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
    return env


def start_processes(
    layout: ProcessLayout,
    ipfs_binary: str,
    log_dir: Path,
) -> List[Tuple[ProcessNode, subprocess.Popen, Path, IO[bytes]]]:
    base_env = os.environ.copy()
    base_env["IPFS_BINARY"] = ipfs_binary
    processes: List[Tuple[ProcessNode, subprocess.Popen, Path, IO[bytes]]] = []
    cluster_names = ",".join(node.name for node in layout.nodes)
    for node in layout.nodes:
        log_path = log_dir / f"{node.name}.log"
        env = _build_env(base_env, node, cluster_names, layout)
        log_handle = log_path.open("ab")
        log_handle.write(f"=== Starting {node.name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode("utf-8"))
        log_handle.flush()
        proc = subprocess.Popen(
            ["/bin/sh", str(BOOTSTRAP_SCRIPT)],
            cwd=ROOT_DIR,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((node, proc, log_path, log_handle))
        print(
            f"[ipfs] {node.name} -> API {node.api_host}:{node.api_port}, "
            f"Gateway {node.gateway_host}:{node.gateway_port}, Swarm {node.swarm_host}:{node.swarm_port} "
            f"(log: {log_path})"
        )
    return processes


def shutdown(processes: List[Tuple[ProcessNode, subprocess.Popen, Path, IO[bytes]]]) -> None:
    for _, proc, _, _ in processes:
        if proc.poll() is None:
            proc.terminate()
    for node, proc, log_path, handle in processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        handle.write(f"=== Stopped {node.name} ===\n".encode("utf-8"))
        handle.flush()
        handle.close()


def monitor(processes: List[Tuple[ProcessNode, subprocess.Popen, Path, IO[bytes]]]) -> None:
    try:
        while True:
            for node, proc, _, _ in processes:
                ret = proc.poll()
                if ret is not None:
                    raise SystemExit(f"IPFS process {node.name} exited with code {ret}. Check its log for details.")
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping IPFS processes...")


def main() -> None:
    args = parse_args()
    layout = load_layout(args.config)
    prepare_directories(layout, args.log_dir)
    ipfs_binary = locate_ipfs(args.ipfs_binary)
    processes = start_processes(layout, ipfs_binary, args.log_dir)

    def _handle_signal(signum, _frame):
        print(f"Received signal {signum}; shutting down IPFS processes...")
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)
    try:
        monitor(processes)
    finally:
        shutdown(processes)


if __name__ == "__main__":
    main()
