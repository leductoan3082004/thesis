#!/usr/bin/env python3
"""Manage the process-mode infrastructure (IPFS + blockchain/api-gateway)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional
import shutil

ROOT_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
IPFS_RUNNER_SCRIPT = ROOT_DIR / "scripts" / "run_ipfs_processes.py"
DEFAULT_IPFS_CONFIG = ROOT_DIR / "config" / "ipfs-process.json"
DEFAULT_IPFS_LOG_DIR = ROOT_DIR / "logs" / "ipfs"
PROCESS_RUNTIME_DIR = ROOT_DIR / "process-runtime"
IPFS_PID_FILE = PROCESS_RUNTIME_DIR / "ipfs-runner.pid"
IPFS_RUNNER_LOG = PROCESS_RUNTIME_DIR / "ipfs-runner.log"
BLOCKCHAIN_REPO_DIR = PARENT_DIR / "thesis-blockchain"
BLOCKCHAIN_API_GATEWAY_DIR = BLOCKCHAIN_REPO_DIR / "api-gateway"
BLOCKCHAIN_PROCESS_RUNNER = BLOCKCHAIN_API_GATEWAY_DIR / "process-runner" / "manage.sh"
BLOCKCHAIN_PROCESS_DIR = BLOCKCHAIN_PROCESS_RUNNER.parent
BLOCKCHAIN_PROCESS_RUNTIME_DIR = BLOCKCHAIN_PROCESS_DIR / "runtime"
BLOCKCHAIN_PROCESS_LOG_DIR = BLOCKCHAIN_PROCESS_RUNTIME_DIR / "logs"

try:
    from scripts.run_docker_with_nodes import (
        DEFAULT_GATEWAY_URL as DOCKER_DEFAULT_GATEWAY_URL,
        _bulk_register_trainers,
        _read_admin_public_key,
        _require_blockchain_repo_paths,
        _resolve_auth_secret,
        _wait_for_gateway_health,
    )
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Unable to import helpers from run_docker_with_nodes.py. "
        "Run this script from the repository root after installing project dependencies.",
    ) from exc


def _ensure_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"{description} not found at {path}.")


def _ensure_runtime_dir() -> None:
    PROCESS_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _listening_pids(port: int) -> Iterable[int]:
    try:
        result = subprocess.run(
            ["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode not in (0, 1):
        return []
    values = (result.stdout or "").strip().splitlines()
    return [int(value) for value in values if value.strip()]


def _command_for_pid(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return (result.stdout or "").strip()


def _ensure_port_free(port: int, label: str) -> None:
    pids = list(_listening_pids(port))
    if not pids:
        return
    for pid in pids:
        cmdline = _command_for_pid(pid)
        if "ipfs" not in cmdline.lower():
            raise SystemExit(
                f"Port {port} required for {label} is already in use by PID {pid} ({cmdline or 'unknown'}). "
                "Stop that process or free the port before continuing.",
            )
        print(f"Stopping leftover {label} listener on port {port} (pid {pid})")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 15
    while time.time() < deadline:
        if not list(_listening_pids(port)):
            return
        time.sleep(1)
    raise SystemExit(f"Port {port} is still busy after terminating leftover {label} processes.")


def _collect_ipfs_ports(config_path: Path) -> Iterable[int]:
    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse IPFS process config {config_path}: {exc}") from exc
    nodes = raw.get("nodes") or []
    if not nodes:
        raise SystemExit(f"{config_path} does not define any IPFS nodes.")
    ports: set[int] = set()
    for entry in nodes:
        for key in ("api_port", "gateway_port", "swarm_port"):
            value = entry.get(key)
            if value:
                ports.add(int(value))
    return sorted(ports)


def _ensure_ipfs_ports_free(config_path: Path) -> None:
    ports = list(_collect_ipfs_ports(config_path))
    for port in ports:
        _ensure_port_free(port, "IPFS")


def _stop_ipfs_runner(verbose: bool = True, timeout: int = 30) -> None:
    pid = _read_pid(IPFS_PID_FILE)
    if not pid:
        if verbose:
            print("IPFS process runner is not running.")
        return
    if verbose:
        print(f"Stopping IPFS process runner (pid {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        IPFS_PID_FILE.unlink(missing_ok=True)
        if verbose:
            print("IPFS runner pid file was stale; nothing to stop.")
        return
    for _ in range(timeout):
        if not _pid_alive(pid):
            break
        time.sleep(1)
    else:
        print("IPFS runner did not exit after SIGTERM; sending SIGKILL.")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    IPFS_PID_FILE.unlink(missing_ok=True)


def _start_ipfs_runner(config: Path, log_dir: Path, ipfs_binary: Optional[str]) -> None:
    _ensure_path_exists(IPFS_RUNNER_SCRIPT, "IPFS process runner script")
    _ensure_path_exists(config, "IPFS process configuration")
    _ensure_runtime_dir()
    existing_pid = _read_pid(IPFS_PID_FILE)
    if existing_pid and _pid_alive(existing_pid):
        raise SystemExit(
            f"IPFS process runner is already active (pid {existing_pid}). "
            "Run 'python scripts/run_process_mode.py stop' before starting again.",
        )
    cmd = [sys.executable, str(IPFS_RUNNER_SCRIPT), "--config", str(config), "--log-dir", str(log_dir)]
    if ipfs_binary:
        cmd.extend(["--ipfs-binary", ipfs_binary])
    PROCESS_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    IPFS_RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with IPFS_RUNNER_LOG.open("ab") as log_handle:
        log_handle.write(f"=== Launching IPFS runner at {timestamp} ===\n".encode("utf-8"))
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    IPFS_PID_FILE.write_text(f"{proc.pid}\n")
    time.sleep(3)
    if proc.poll() is not None:
        IPFS_PID_FILE.unlink(missing_ok=True)
        raise SystemExit(
            f"IPFS runner failed to start (exit code {proc.returncode}). "
            f"Check {IPFS_RUNNER_LOG} for details.",
        )
    print(f"Started IPFS runner (pid {proc.pid}); logs -> {IPFS_RUNNER_LOG}")


def _run_process_runner(action: str, env: Optional[Dict[str, str]], check: bool = True) -> int:
    _ensure_path_exists(BLOCKCHAIN_PROCESS_RUNNER, "Blockchain process runner script")
    result = subprocess.run(
        ["./manage.sh", action],
        cwd=BLOCKCHAIN_PROCESS_RUNNER.parent,
        env=env,
        check=False,
    )
    if check and result.returncode != 0:
        raise SystemExit(
            f"Blockchain process runner command '{action}' failed with exit code {result.returncode}.",
        )
    return result.returncode


def _start_blockchain_processes(paths: Dict[str, Path], auth_secret: str, gateway_url: str) -> None:
    admin_public_key = _read_admin_public_key(paths)
    env = os.environ.copy()
    env["AUTH_JWT_SECRET"] = auth_secret
    env["ADMIN_PUBLIC_KEY"] = admin_public_key
    env.setdefault("BLOCKCHAIN_GATEWAY_URL", gateway_url)
    print(f"Starting blockchain process stack (orderer + peers + API gateway)... logs -> {BLOCKCHAIN_PROCESS_LOG_DIR}")
    _run_process_runner("start", env, check=True)


def _stop_blockchain_processes() -> None:
    if not BLOCKCHAIN_PROCESS_RUNNER.exists():
        return
    env = os.environ.copy()
    _run_process_runner("stop", env, check=False)


def _clear_blockchain_runtime(paths: Dict[str, Path]) -> None:
    if BLOCKCHAIN_PROCESS_RUNTIME_DIR.exists():
        shutil.rmtree(BLOCKCHAIN_PROCESS_RUNTIME_DIR)
    BLOCKCHAIN_PROCESS_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    trainer_db = paths.get("trainer_db")
    if trainer_db:
        trainer_db.parent.mkdir(parents=True, exist_ok=True)
        trainer_db.write_text("[\n]\n")
    print(f"Cleared blockchain runtime data -> {BLOCKCHAIN_PROCESS_RUNTIME_DIR}")


def _start_fl_docker_stack(compose_file: Path, detach: bool, build: bool) -> None:
    compose_file = compose_file.resolve()
    if not compose_file.exists():
        raise SystemExit(f"FL docker compose file not found at {compose_file}.")
    compose_dir = compose_file.parent
    compose_name = compose_file.name
    cmd = ["docker", "compose", "-f", compose_name, "up"]
    if build:
        cmd.append("--build")
    if detach:
        cmd.append("-d")
    print(
        f"Starting FL docker stack via {compose_file} "
        f"(build={'yes' if build else 'no'}, detach={'yes' if detach else 'no'})...",
    )
    result = subprocess.run(cmd, cwd=compose_dir, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"docker compose up for FL stack exited with code {result.returncode}.",
        )


def _stop_fl_docker_stack(compose_file: Path) -> None:
    compose_file = compose_file.resolve()
    if not compose_file.exists():
        return
    compose_dir = compose_file.parent
    compose_name = compose_file.name
    cmd = ["docker", "compose", "-f", compose_name, "down", "-v"]
    subprocess.run(cmd, cwd=compose_dir, check=False)


def _resolve_gateway_url(cli_value: Optional[str]) -> str:
    return cli_value or os.environ.get("BLOCKCHAIN_GATEWAY_URL") or DOCKER_DEFAULT_GATEWAY_URL


def _start(args: argparse.Namespace) -> None:
    blockchain_paths = _require_blockchain_repo_paths()
    auth_secret = _resolve_auth_secret(blockchain_paths)
    gateway_url = _resolve_gateway_url(args.gateway_url)
    ipfs_started = False
    try:
        if args.skip_ipfs:
            print("Skipping IPFS processes (per --skip-ipfs).")
        else:
            print("Ensuring previous IPFS processes are stopped...")
            _stop_ipfs_runner(verbose=False)
            _ensure_path_exists(args.ipfs_config, "IPFS process configuration")
            _ensure_ipfs_ports_free(args.ipfs_config)
            _start_ipfs_runner(args.ipfs_config, args.ipfs_log_dir, args.ipfs_binary)
            ipfs_started = True
        if args.skip_blockchain:
            print("Skipping blockchain stack (per --skip-blockchain).")
            return
        print("Ensuring blockchain process stack is stopped...")
        _stop_blockchain_processes()
        print("Resetting blockchain runtime state...")
        _clear_blockchain_runtime(blockchain_paths)
        _start_blockchain_processes(blockchain_paths, auth_secret, gateway_url)
        print(f"Waiting for blockchain gateway health -> {gateway_url.rstrip('/')}/health")
        _wait_for_gateway_health(gateway_url)
        print("Gateway is healthy; registering trainers via /auth/register-trainers ...")
        _bulk_register_trainers(blockchain_paths, gateway_url)
        _summarize_trainer_db(blockchain_paths)
        print("Blockchain stack is running in process mode and trainer nodes are registered.")
        if args.skip_ipfs:
            print("Note: IPFS processes were not started; run 'run_process_mode.py start' without --skip-ipfs if needed.")
        else:
            print("IPFS processes are running in the background (managed by run_ipfs_processes.py).")
        if args.fl_compose_file:
            print("Ensuring previous FL docker stack is stopped...")
            _stop_fl_docker_stack(args.fl_compose_file)
            _start_fl_docker_stack(
                args.fl_compose_file,
                detach=args.fl_detach,
                build=not args.fl_no_build,
            )
    except Exception:
        if ipfs_started and not args.keep_ipfs_on_failure:
            _stop_ipfs_runner(verbose=False)
        raise


def _stop(args: argparse.Namespace) -> None:
    if not args.skip_blockchain:
        _stop_blockchain_processes()
    if not args.skip_ipfs:
        _stop_ipfs_runner()


def _summarize_trainer_db(paths: Dict[str, Path]) -> None:
    trainer_db = paths.get("trainer_db")
    if not trainer_db or not trainer_db.exists():
        print(f"Trainer database not found at {trainer_db}.")
        return
    try:
        entries = json.loads(trainer_db.read_text() or "[]")
    except json.JSONDecodeError as exc:
        print(f"Trainer database {trainer_db} is not valid JSON: {exc}")
        return
    count = len(entries) if isinstance(entries, list) else 0
    print(f"Trainer whitelist updated -> {trainer_db} ({count} entries)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Launch IPFS + blockchain stacks as host processes.")
    start_parser.add_argument(
        "--ipfs-config",
        type=Path,
        default=DEFAULT_IPFS_CONFIG,
        help=f"Path to the IPFS process layout (default: {DEFAULT_IPFS_CONFIG}).",
    )
    start_parser.add_argument(
        "--ipfs-log-dir",
        type=Path,
        default=DEFAULT_IPFS_LOG_DIR,
        help=f"Directory for per-node IPFS logs (default: {DEFAULT_IPFS_LOG_DIR}).",
    )
    start_parser.add_argument(
        "--ipfs-binary",
        type=str,
        default=None,
        help="Optional override for the Kubo binary path (defaults to 'ipfs').",
    )
    start_parser.add_argument(
        "--gateway-url",
        type=str,
        help="Override the API gateway base URL (default: env BLOCKCHAIN_GATEWAY_URL or http://localhost:9000).",
    )
    start_parser.add_argument(
        "--skip-ipfs",
        action="store_true",
        help="Start only the blockchain stack and leave IPFS untouched.",
    )
    start_parser.add_argument(
        "--skip-blockchain",
        action="store_true",
        help="Start only IPFS processes and skip the blockchain stack.",
    )
    start_parser.add_argument(
        "--keep-ipfs-on-failure",
        action="store_true",
        help="Do not stop the IPFS process runner if blockchain startup fails.",
    )
    start_parser.add_argument(
        "--fl-compose-file",
        type=Path,
        help="Optional docker compose file to start the FL stack after blockchain registration.",
    )
    start_parser.add_argument(
        "--fl-detach",
        dest="fl_detach",
        action="store_true",
        default=True,
        help="Run the FL docker stack in detached mode (default).",
    )
    start_parser.add_argument(
        "--fl-no-detach",
        dest="fl_detach",
        action="store_false",
        help="Run the FL docker stack in the foreground (implies --fl-detach=0).",
    )
    start_parser.add_argument(
        "--fl-no-build",
        action="store_true",
        help="Skip passing --build to docker compose when launching the FL stack.",
    )

    stop_parser = subparsers.add_parser("stop", help="Stop IPFS and blockchain process-mode stacks.")
    stop_parser.add_argument(
        "--skip-ipfs",
        action="store_true",
        help="Skip stopping IPFS processes.",
    )
    stop_parser.add_argument(
        "--skip-blockchain",
        action="store_true",
        help="Skip stopping the blockchain stack.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "start":
        _start(args)
    elif args.command == "stop":
        _stop(args)
    else:  # pragma: no cover - argparse guarantees start/stop
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
