#!/usr/bin/env python3
"""Unified control CLI for the secure-aggregation process-only runtime.

Usage:
    secureagg_ctl.py start   [--nodes N | --nodes-map PATH] [--clique-size K] ...
    secureagg_ctl.py stop
    secureagg_ctl.py status
    secureagg_ctl.py logs    [--node NAME] [--service SVC] [--follow] ...
    secureagg_ctl.py cleanup [--purge-logs]
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from scripts.runtime.port_allocator import PortMap, allocate_ports, check_conflicts
from scripts.runtime.process_registry import ManagedProcess, ProcessRegistry, SENTINEL_PID
from scripts.runtime.config_generator import generate_process_layout
from scripts.runtime.ipfs_manager import start_ipfs
from scripts.runtime import blockchain_helpers
from scripts.runtime import observability
from scripts.runtime import loki_client

PROCESS_RUNTIME_DIR = ROOT_DIR / "process-runtime"
REGISTRY_PATH = PROCESS_RUNTIME_DIR / "registry.json"
DEFAULT_IPFS_CONFIG = ROOT_DIR / "config" / "ipfs-process.json"
DEFAULT_GATEWAY_URL = os.environ.get("BLOCKCHAIN_GATEWAY_URL", "http://localhost:9000")
TTP_SCRIPT = ROOT_DIR / "scripts" / "run_ttp_with_topology.py"
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


def _cmd_start(args: argparse.Namespace) -> None:
    registry = ProcessRegistry(REGISTRY_PATH)

    # If there are still-running processes, refuse to start.
    alive = [r for r in registry.status_all() if r["alive"]]
    if alive:
        names = ", ".join(r["name"] for r in alive)
        raise SystemExit(
            f"Processes still running: {names}. Run 'secureagg_ctl.py stop' first."
        )
    registry.cleanup_stale()

    gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL
    nodes_map_path = Path(args.nodes_map) if args.nodes_map else None

    # Generate process layout.
    layout = generate_process_layout(
        num_nodes=args.nodes,
        runtime_dir=PROCESS_RUNTIME_DIR,
        nodes_map_path=nodes_map_path,
        clique_size=args.clique_size,
        ipfs_config_path=Path(args.ipfs_config),
        gateway_url=gateway_url,
        copy_data=args.copy_data,
        ttp_port=args.ttp_port,
        base_node_port=args.base_node_port,
        base_metrics_port=args.base_metrics_port,
    )
    node_count = layout["node_count"]

    port_map = allocate_ports(
        node_count,
        ttp_port=args.ttp_port,
        base_node=args.base_node_port,
        base_metrics=args.base_metrics_port,
    )

    # Check port conflicts before proceeding.
    conflicts = check_conflicts(port_map)
    if conflicts:
        detail = ", ".join(f"{label}={port}" for label, port in conflicts)
        raise SystemExit(f"Port conflicts detected: {detail}. Free them before starting.")

    started: List[str] = []

    blockchain_started = False

    def _rollback() -> None:
        if started:
            print("Rolling back: stopping already-started processes...")
            registry.stop_all()
        if blockchain_started:
            blockchain_helpers.stop_blockchain()

    try:
        # 1) IPFS
        if not args.skip_ipfs:
            print("Starting IPFS processes...")
            ipfs_procs = start_ipfs(
                config_path=Path(args.ipfs_config),
                runtime_dir=PROCESS_RUNTIME_DIR,
            )
            for proc in ipfs_procs:
                registry.register(proc)
                started.append(proc.name)
            time.sleep(3)

        # 2) Blockchain
        if not args.skip_blockchain:
            blockchain_paths = blockchain_helpers.require_blockchain_repo_paths()
            auth_secret = blockchain_helpers.resolve_auth_secret(blockchain_paths)

            print("Stopping any previous blockchain processes...")
            blockchain_helpers.stop_blockchain()
            print("Clearing blockchain runtime state...")
            blockchain_helpers.clear_blockchain_runtime(blockchain_paths)
            print("Preparing blockchain artifacts (process-mode CA)...")
            blockchain_helpers.prepare_blockchain_artifacts(blockchain_paths, auth_secret)
            print("Starting blockchain stack (orderer + peers + gateway)...")
            blockchain_helpers.start_blockchain(blockchain_paths, auth_secret, gateway_url)
            blockchain_started = True

            # The blockchain is managed by manage.sh externally; register a
            # sentinel entry so status/stop can track it.  Use SENTINEL_PID
            # so _pid_alive/_kill_pid never target PID 0 (the process group).
            bc_pid_file = PROCESS_RUNTIME_DIR / "pids" / "blockchain.pid"
            bc_pid_file.parent.mkdir(parents=True, exist_ok=True)
            bc_pid_file.write_text(f"{SENTINEL_PID}\n")
            registry.register(
                ManagedProcess(
                    name="blockchain",
                    pid=SENTINEL_PID,
                    pid_file=str(bc_pid_file),
                    ports=[7050, 7051, 8051, 9051, 9000],
                    component_type="infrastructure",
                )
            )
            started.append("blockchain")

            print(f"Waiting for gateway health at {gateway_url}...")
            blockchain_helpers.wait_for_gateway_health(gateway_url)
            print("Registering trainers...")
            blockchain_helpers.bulk_register_trainers(blockchain_paths, gateway_url)

        # 3) TTP
        print("Starting TTP service...")
        ttp_log = PROCESS_RUNTIME_DIR / "logs" / "ttp.log"
        ttp_log.parent.mkdir(parents=True, exist_ok=True)
        ttp_pid_file = PROCESS_RUNTIME_DIR / "pids" / "ttp.pid"
        ttp_pid_file.parent.mkdir(parents=True, exist_ok=True)
        ttp_proc = _start_ttp(layout, args, ttp_log)
        registry.register(
            ManagedProcess(
                name="ttp",
                pid=ttp_proc.pid,
                pid_file=str(ttp_pid_file),
                ports=[args.ttp_port],
                log_file=str(ttp_log),
                component_type="ttp",
            )
        )
        started.append("ttp")
        time.sleep(3)
        if ttp_proc.poll() is not None:
            raise SystemExit(f"TTP exited immediately (code {ttp_proc.returncode}). Check {ttp_log}")

        # 4) FL nodes
        print(f"Starting {node_count} FL nodes...")
        for idx in range(node_count):
            config_path = layout["node_configs"][idx]
            node_dir = PROCESS_RUNTIME_DIR / "nodes" / f"node_{idx}"
            node_log = node_dir / "logs" / "node.log"
            node_log.parent.mkdir(parents=True, exist_ok=True)
            node_pid_file = node_dir / "pids" / "node.pid"
            node_pid_file.parent.mkdir(parents=True, exist_ok=True)

            service_port = args.base_node_port + idx
            metrics_port = args.base_metrics_port + idx
            agg_port = service_port + 1000
            bridge_port = service_port + 2000

            node_proc = _start_fl_node(config_path, node_log)
            trainer_id = layout["assignments"][idx]["trainer_id"]
            registry.register(
                ManagedProcess(
                    name=f"node_{idx}",
                    pid=node_proc.pid,
                    pid_file=str(node_pid_file),
                    ports=[service_port, agg_port, bridge_port, metrics_port],
                    log_file=str(node_log),
                    component_type="training",
                )
            )
            started.append(f"node_{idx}")
            print(f"  [{trainer_id}] pid={node_proc.pid} service={service_port} metrics={metrics_port}")

        # 5) Monitoring stack (Loki -> Promtail -> Prometheus -> Grafana)
        if not args.skip_monitoring:
            print("Starting monitoring stack...")
            loki_proc = observability.start_loki(PROCESS_RUNTIME_DIR)
            registry.register(loki_proc)
            started.append(loki_proc.name)
            time.sleep(2)

            node_ids = [a["trainer_id"] for a in layout["assignments"]]
            promtail_proc = observability.start_promtail(PROCESS_RUNTIME_DIR, node_count, node_ids=node_ids)
            registry.register(promtail_proc)
            started.append(promtail_proc.name)

            prometheus_proc = observability.start_prometheus(
                PROCESS_RUNTIME_DIR, node_count, args.base_metrics_port,
            )
            registry.register(prometheus_proc)
            started.append(prometheus_proc.name)

            grafana_proc = observability.start_grafana(PROCESS_RUNTIME_DIR, node_count, node_ids=node_ids)
            registry.register(grafana_proc)
            started.append(grafana_proc.name)

            print("  Loki      -> http://localhost:3100")
            print("  Promtail  -> http://localhost:9080")
            print("  Prometheus-> http://localhost:9090")
            print("  Grafana   -> http://localhost:3000 (admin/admin)")

        print(f"\nAll {node_count} nodes started. Use 'secureagg_ctl.py status' to check health.")

    except (Exception, KeyboardInterrupt):
        _rollback()
        raise


def _start_ttp(layout: Dict, args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    node_count = layout["node_count"]
    topology_path = layout["topology_path"]
    cmd = [
        PYTHON, str(TTP_SCRIPT),
        "--port", str(args.ttp_port),
        "--num-clients", str(node_count),
        "--clique-size", str(layout["clique_size"]),
        "--topology-output", topology_path,
        "--data-dir", str(ROOT_DIR / "data"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["PYTHONUNBUFFERED"] = "1"
    # Point NODE_CONFIG_DIR to the generated per-node configs.
    env["NODE_CONFIG_DIR"] = str(PROCESS_RUNTIME_DIR / "config" / "nodes")
    log_handle = log_path.open("ab")
    return subprocess.Popen(cmd, cwd=ROOT_DIR, env=env, stdout=log_handle, stderr=subprocess.STDOUT)


def _start_fl_node(config_path: str, log_path: Path) -> subprocess.Popen:
    cmd = [PYTHON, "-m", "secure_aggregation.communication.node_service", "--config", config_path]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["PYTHONUNBUFFERED"] = "1"
    log_handle = log_path.open("ab")
    return subprocess.Popen(cmd, cwd=ROOT_DIR, env=env, stdout=log_handle, stderr=subprocess.STDOUT)


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def _cmd_stop(_args: argparse.Namespace) -> None:
    registry = ProcessRegistry(REGISTRY_PATH)
    if registry.names:
        print(f"Stopping {len(registry.names)} managed processes...")
        registry.stop_all()
    else:
        print("No managed processes found in registry.")
    # Always attempt blockchain shutdown — manage.sh is external to the
    # registry and may be running even when the registry file is stale.
    blockchain_helpers.stop_blockchain()
    print("All processes stopped.")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _cmd_status(_args: argparse.Namespace) -> None:
    registry = ProcessRegistry(REGISTRY_PATH)
    rows = registry.status_all()
    if not rows:
        print("No managed processes registered.")
        return
    header = f"{'NAME':<25} {'PID':>7} {'STATUS':<10} {'TYPE':<15} {'PORTS'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        status = "running" if row["alive"] else "dead"
        ports_str = ",".join(str(p) for p in row["ports"]) if row["ports"] else "-"
        print(f"{row['name']:<25} {row['pid']:>7} {status:<10} {row['type']:<15} {ports_str}")


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def _loki_available(loki_url: str) -> bool:
    """Return True if Loki is reachable at the given URL."""
    from urllib import request as urllib_request, error as urllib_error
    try:
        with urllib_request.urlopen(f"{loki_url}/ready", timeout=2) as resp:
            return resp.status == 200
    except (urllib_error.URLError, OSError):
        return False


def _cmd_logs(args: argparse.Namespace) -> None:
    """Query logs via Loki when available, falling back to local log files."""
    loki_url = args.loki_url

    if _loki_available(loki_url):
        _cmd_logs_loki(args, loki_url)
    else:
        _cmd_logs_files(args)


def _cmd_logs_loki(args: argparse.Namespace, loki_url: str) -> None:
    query = loki_client.build_logql(
        service=args.service,
        node_id=args.node,
        level="error" if args.errors else args.level,
        contains=args.contains,
    )

    if args.follow:
        loki_client.tail(query, loki_url=loki_url)
        return

    entries = loki_client.query_range(
        query,
        start=args.since,
        end=args.until,
        limit=args.limit,
        loki_url=loki_url,
    )

    if not entries:
        print("No log entries found.")
        return

    print(loki_client.format_entries(entries, as_json=args.json))


def _cmd_logs_files(args: argparse.Namespace) -> None:
    """Fallback: read logs directly from process-runtime log files."""
    log_dir = PROCESS_RUNTIME_DIR
    if not log_dir.exists():
        print("No process-runtime directory found. Start the stack first.")
        return

    log_files: List[Path] = []
    if args.node:
        for node_dir in sorted((log_dir / "nodes").glob("node_*")):
            node_log = node_dir / "logs" / "node.log"
            if node_log.exists():
                config_dir = PROCESS_RUNTIME_DIR / "config" / "nodes"
                config_file = config_dir / f"{node_dir.name}.json"
                if config_file.exists():
                    import json
                    cfg = json.loads(config_file.read_text())
                    if cfg.get("node_id") == args.node or cfg.get("trainer_id") == args.node:
                        log_files.append(node_log)
                        break
                if node_dir.name == args.node:
                    log_files.append(node_log)
                    break
    elif args.service:
        if args.service == "ttp":
            ttp_log = log_dir / "logs" / "ttp.log"
            if ttp_log.exists():
                log_files.append(ttp_log)
        elif args.service == "ipfs":
            ipfs_log_dir = log_dir / "logs" / "ipfs"
            if ipfs_log_dir.exists():
                log_files.extend(sorted(ipfs_log_dir.glob("*.log")))
    else:
        for node_dir in sorted((log_dir / "nodes").glob("node_*")):
            node_log = node_dir / "logs" / "node.log"
            if node_log.exists():
                log_files.append(node_log)
        ttp_log = log_dir / "logs" / "ttp.log"
        if ttp_log.exists():
            log_files.append(ttp_log)

    if not log_files:
        print("No log files found.")
        return

    if args.follow:
        cmd = ["tail", "-f"] + [str(f) for f in log_files]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
    else:
        limit = args.limit or 100
        for log_file in log_files:
            print(f"\n=== {log_file.relative_to(ROOT_DIR)} ===")
            lines = log_file.read_text().splitlines()
            for line in lines[-limit:]:
                print(line)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def _cmd_cleanup(args: argparse.Namespace) -> None:
    from scripts.runtime.process_registry import lsof_port_pids

    registry = ProcessRegistry(REGISTRY_PATH)
    killed = registry.kill_stale()
    if killed:
        print(f"Killed {killed} stale process(es).")
    blockchain_helpers.stop_blockchain()

    # Clean up PID files.
    pids_dir = PROCESS_RUNTIME_DIR / "pids"
    if pids_dir.exists():
        for pid_file in pids_dir.glob("*.pid"):
            pid_file.unlink()

    if args.purge_logs:
        import shutil
        logs_dir = PROCESS_RUNTIME_DIR / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        for node_dir in (PROCESS_RUNTIME_DIR / "nodes").glob("node_*"):
            node_logs = node_dir / "logs"
            if node_logs.exists():
                shutil.rmtree(node_logs)
        obs_dir = PROCESS_RUNTIME_DIR / "observability"
        if obs_dir.exists():
            shutil.rmtree(obs_dir)
        print("Purged all log and observability data.")

    # Verify well-known managed ports are free.
    managed_ports = [3000, 3100, 9080, 9090, 50051]
    busy = [(p, lsof_port_pids(p)) for p in managed_ports if lsof_port_pids(p)]
    if busy:
        for port, pids in busy:
            print(f"Warning: port {port} still occupied by PID(s) {pids}")
    else:
        print("All managed ports verified free.")
    print("Cleanup complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="secureagg_ctl.py",
        description="Unified control CLI for the secure-aggregation process-only runtime.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # -- start ----------------------------------------------------------------
    start_p = subs.add_parser("start", help="Launch the full process-mode stack.")
    start_p.add_argument("--nodes", type=int, help="Number of FL nodes.")
    start_p.add_argument("--nodes-map", type=str, help="Path to hierarchical nodes-map JSON.")
    start_p.add_argument("--clique-size", type=int, default=3, help="D-Cliques clique size (default: 3).")
    start_p.add_argument("--ipfs-config", type=str, default=str(DEFAULT_IPFS_CONFIG), help="IPFS process config.")
    start_p.add_argument("--gateway-url", type=str, help="Blockchain gateway URL.")
    start_p.add_argument("--skip-ipfs", action="store_true", help="Skip starting IPFS.")
    start_p.add_argument("--skip-blockchain", action="store_true", help="Skip starting blockchain.")
    start_p.add_argument("--skip-monitoring", action="store_true", help="Skip starting monitoring stack (Loki, Promtail, Prometheus, Grafana).")
    start_p.add_argument("--copy-data", action="store_true", help="Copy datasets instead of symlinking.")
    start_p.add_argument("--ttp-port", type=int, default=50051, help="TTP service port (default: 50051).")
    start_p.add_argument("--base-node-port", type=int, default=51000, help="Base port for FL node services.")
    start_p.add_argument("--base-metrics-port", type=int, default=61000, help="Base port for metrics endpoints.")

    # -- stop -----------------------------------------------------------------
    subs.add_parser("stop", help="Stop all managed processes.")

    # -- status ---------------------------------------------------------------
    subs.add_parser("status", help="Show status of all managed processes.")

    # -- logs -----------------------------------------------------------------
    logs_p = subs.add_parser("logs", help="View logs from managed processes.")
    logs_p.add_argument("--node", type=str, help="Filter by node ID (e.g. trainer-node-001).")
    logs_p.add_argument("--service", type=str, help="Filter by service name (fl_node, ttp, ipfs).")
    logs_p.add_argument("--level", type=str, help="Filter by log level (e.g. ERROR, WARNING).")
    logs_p.add_argument("--contains", type=str, help="Filter lines containing this text.")
    logs_p.add_argument("--since", type=str, help="Start time: duration (30m, 1h, 2d) or ISO timestamp.")
    logs_p.add_argument("--until", type=str, help="End time: duration or ISO timestamp.")
    logs_p.add_argument("--follow", "-f", action="store_true", help="Stream logs in real-time.")
    logs_p.add_argument("--limit", type=int, default=100, help="Number of recent entries to show.")
    logs_p.add_argument("--errors", action="store_true", help="Show only error-level logs.")
    logs_p.add_argument("--json", action="store_true", help="Output logs as JSON.")
    logs_p.add_argument("--loki-url", type=str, default="http://localhost:3100", help="Loki endpoint URL.")

    # -- cleanup --------------------------------------------------------------
    cleanup_p = subs.add_parser("cleanup", help="Remove stale PIDs and optionally purge logs.")
    cleanup_p.add_argument("--purge-logs", action="store_true", help="Also delete log files.")

    return parser


def _install_sigint_handler() -> None:
    """Ensure SIGINT during startup triggers a clean shutdown of all managed processes."""

    def _handler(signum: int, frame: object) -> None:
        print("\nSIGINT received — stopping all managed processes...")
        registry = ProcessRegistry(REGISTRY_PATH)
        registry.stop_all()
        blockchain_helpers.stop_blockchain()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handler)


def main() -> None:
    _install_sigint_handler()
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "start": _cmd_start,
        "stop": _cmd_stop,
        "status": _cmd_status,
        "logs": _cmd_logs,
        "cleanup": _cmd_cleanup,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(1)
    handler(args)


if __name__ == "__main__":
    main()
