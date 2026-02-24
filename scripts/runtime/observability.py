"""Observability config generation and process management for Loki, Promtail, Prometheus, and Grafana."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]

from scripts.runtime.process_registry import ManagedProcess

ROOT_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# YAML helper (fallback if PyYAML not installed)
# ---------------------------------------------------------------------------


def _dump_yaml(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    else:
        path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Loki
# ---------------------------------------------------------------------------


def generate_loki_config(runtime_dir: Path) -> Path:
    loki_dir = runtime_dir / "observability" / "loki"
    loki_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "auth_enabled": False,
        "server": {"http_listen_port": 3100},
        "common": {
            "path_prefix": str(loki_dir / "data"),
            "storage": {"filesystem": {"chunks_directory": str(loki_dir / "chunks"), "rules_directory": str(loki_dir / "rules")}},
            "replication_factor": 1,
            "ring": {"kvstore": {"store": "inmemory"}},
        },
        "schema_config": {
            "configs": [
                {
                    "from": "2024-01-01",
                    "store": "tsdb",
                    "object_store": "filesystem",
                    "schema": "v13",
                    "index": {"prefix": "index_", "period": "24h"},
                }
            ]
        },
    }
    config_path = runtime_dir / "observability" / "loki.yml"
    _dump_yaml(config, config_path)
    return config_path


def start_loki(runtime_dir: Path) -> ManagedProcess:
    config_path = generate_loki_config(runtime_dir)
    log_path = runtime_dir / "logs" / "loki.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = runtime_dir / "pids" / "loki.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    loki_bin = shutil.which("loki")
    if not loki_bin:
        raise SystemExit("Loki binary not found in PATH. Install Loki and ensure 'loki' is available.")

    log_handle = log_path.open("ab")
    proc = subprocess.Popen(
        [loki_bin, "-config.file", str(config_path)],
        cwd=runtime_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    log_handle.close()
    return ManagedProcess(
        name="loki",
        pid=proc.pid,
        pid_file=str(pid_file),
        ports=[3100],
        log_file=str(log_path),
        component_type="monitoring",
    )


# ---------------------------------------------------------------------------
# Promtail
# ---------------------------------------------------------------------------


def generate_promtail_config(
    runtime_dir: Path,
    node_count: int,
    node_ids: Optional[List[str]] = None,
) -> Path:
    promtail_dir = runtime_dir / "observability" / "promtail"
    promtail_dir.mkdir(parents=True, exist_ok=True)

    scrape_configs: List[Dict[str, Any]] = []

    # Scrape FL node logs through a single file target and derive node labels
    # from the concrete filename to keep target count small.
    pipeline_stages: List[Dict[str, Any]] = [
        {
            "regex": {
                "source": "filename",
                "expression": r".*/node_(?P<node_index>[0-9]+)/logs/.*",
            }
        },
        {"labels": {"node_index": "node_index"}},
    ]
    if node_ids:
        for idx, node_id in enumerate(node_ids):
            pipeline_stages.append(
                {
                    "match": {
                        "selector": '{service="fl_node", node_index="' + str(idx) + '"}',
                        "stages": [{"static_labels": {"node_id": node_id}}],
                    }
                }
            )
    fl_job = {
        "job_name": "fl_nodes",
        "pipeline_stages": pipeline_stages,
        "static_configs": [
            {
                "targets": ["localhost"],
                "labels": {
                    "service": "fl_node",
                    "__path__": str(runtime_dir / "nodes" / "node_*" / "logs" / "*.log"),
                },
            }
        ],
    }
    scrape_configs.append(fl_job)

    # Scrape TTP logs.
    ttp_log = runtime_dir / "logs" / "ttp.log"
    scrape_configs.append(
        {
            "job_name": "ttp",
            "static_configs": [
                {
                    "targets": ["localhost"],
                    "labels": {
                        "service": "ttp",
                        "__path__": str(ttp_log),
                    },
                }
            ],
        }
    )

    # Scrape IPFS logs.
    scrape_configs.append(
        {
            "job_name": "ipfs",
            "static_configs": [
                {
                    "targets": ["localhost"],
                    "labels": {
                        "service": "ipfs",
                        "__path__": str(runtime_dir / "logs" / "ipfs" / "*.log"),
                    },
                }
            ],
        }
    )

    config = {
        "server": {"http_listen_port": 9080, "grpc_listen_port": 0},
        "positions": {"filename": str(promtail_dir / "positions.yml")},
        "clients": [{"url": "http://localhost:3100/loki/api/v1/push"}],
        "scrape_configs": scrape_configs,
    }
    config_path = runtime_dir / "observability" / "promtail.yml"
    _dump_yaml(config, config_path)
    return config_path


def start_promtail(
    runtime_dir: Path,
    node_count: int,
    node_ids: Optional[List[str]] = None,
) -> ManagedProcess:
    # Promtail opens one FD per file target; scale soft limit with cluster size.
    fd_target = max(8192, node_count * 64)
    new_limit = _ensure_nofile_limit(fd_target)
    if new_limit is not None:
        print(f"[observability] RLIMIT_NOFILE={new_limit} (target {fd_target})")
    config_path = generate_promtail_config(runtime_dir, node_count, node_ids=node_ids)
    log_path = runtime_dir / "logs" / "promtail.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = runtime_dir / "pids" / "promtail.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    promtail_bin = shutil.which("promtail")
    if not promtail_bin:
        raise SystemExit("Promtail binary not found in PATH. Install Promtail and ensure it is available.")

    log_handle = log_path.open("ab")
    cmd = [promtail_bin, "-config.file", str(config_path)]
    if os.name != "nt":
        quoted_bin = shlex.quote(promtail_bin)
        quoted_cfg = shlex.quote(str(config_path))
        cmd = ["bash", "-lc", f"ulimit -n {fd_target} || true; exec {quoted_bin} -config.file {quoted_cfg}"]
    proc = subprocess.Popen(
        cmd,
        cwd=runtime_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    log_handle.close()
    # Fail fast so callers can rollback when promtail cannot initialize.
    time.sleep(1)
    if proc.poll() is not None:
        tail = _tail_file(log_path)
        details = f"\nRecent promtail log output:\n{tail}" if tail else ""
        raise SystemExit(f"Promtail exited immediately (code {proc.returncode}).{details}")
    return ManagedProcess(
        name="promtail",
        pid=proc.pid,
        pid_file=str(pid_file),
        ports=[9080],
        log_file=str(log_path),
        component_type="monitoring",
    )


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------


def generate_prometheus_config(runtime_dir: Path, node_count: int, base_metrics_port: int = 61000) -> Path:
    targets = [f"localhost:{base_metrics_port + i}" for i in range(node_count)]
    config = {
        "global": {"scrape_interval": "5s", "evaluation_interval": "5s"},
        "scrape_configs": [
            {
                "job_name": "fl_nodes",
                "static_configs": [{"targets": targets, "labels": {"group": "training_nodes"}}],
            }
        ],
    }
    config_path = runtime_dir / "observability" / "prometheus.yml"
    _dump_yaml(config, config_path)
    return config_path


def start_prometheus(runtime_dir: Path, node_count: int, base_metrics_port: int = 61000) -> ManagedProcess:
    config_path = generate_prometheus_config(runtime_dir, node_count, base_metrics_port)
    data_dir = runtime_dir / "observability" / "prometheus" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "logs" / "prometheus.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = runtime_dir / "pids" / "prometheus.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    prom_bin = shutil.which("prometheus")
    if not prom_bin:
        raise SystemExit("Prometheus binary not found in PATH.")

    log_handle = log_path.open("ab")
    proc = subprocess.Popen(
        [
            prom_bin,
            f"--config.file={config_path}",
            f"--storage.tsdb.path={data_dir}",
            "--web.listen-address=:9090",
        ],
        cwd=runtime_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    log_handle.close()
    return ManagedProcess(
        name="prometheus",
        pid=proc.pid,
        pid_file=str(pid_file),
        ports=[9090],
        log_file=str(log_path),
        component_type="monitoring",
    )


# ---------------------------------------------------------------------------
# Grafana
# ---------------------------------------------------------------------------


def generate_grafana_config(
    runtime_dir: Path,
    node_count: int = 0,
    node_ids: Optional[List[str]] = None,
) -> Path:
    grafana_dir = runtime_dir / "observability" / "grafana"
    prov_dir = grafana_dir / "provisioning" / "datasources"
    prov_dir.mkdir(parents=True, exist_ok=True)
    dash_prov_dir = grafana_dir / "provisioning" / "dashboards"
    dash_prov_dir.mkdir(parents=True, exist_ok=True)
    dash_dir = grafana_dir / "dashboards"
    dash_dir.mkdir(parents=True, exist_ok=True)

    datasources = {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": "http://localhost:9090",
                "isDefault": True,
            },
            {
                "name": "Loki",
                "type": "loki",
                "access": "proxy",
                "url": "http://localhost:3100",
            },
        ],
    }
    _dump_yaml(datasources, prov_dir / "datasources.yml")

    dashboard_prov = {
        "apiVersion": 1,
        "providers": [
            {
                "name": "default",
                "orgId": 1,
                "folder": "",
                "type": "file",
                "options": {"path": str(dash_dir)},
            }
        ],
    }
    _dump_yaml(dashboard_prov, dash_prov_dir / "dashboards.yml")

    # Copy base dashboards and inject per-node panels for the logs dashboard.
    docker_dash_dir = ROOT_DIR / "docker" / "grafana" / "dashboards"
    if docker_dash_dir.is_dir():
        for src in docker_dash_dir.glob("*.json"):
            if src.name == "fl_logs.json" and node_ids:
                _write_logs_dashboard_with_per_node(src, dash_dir / src.name, node_ids)
            else:
                shutil.copy2(src, dash_dir / src.name)

    return grafana_dir


def _write_logs_dashboard_with_per_node(
    base_path: Path,
    dest_path: Path,
    node_ids: List[str],
) -> None:
    """Load the base logs dashboard and append per-node log panels."""
    dashboard = json.loads(base_path.read_text())
    panels = dashboard["panels"]

    max_id = _max_panel_id(panels)
    max_y = _max_panel_y(panels)

    next_id = max_id + 1
    y = max_y + 1

    # Build individual log panels (two columns, each 12 units wide).
    node_panels: List[Dict[str, Any]] = []
    inner_y = y + 1
    for i, nid in enumerate(node_ids):
        col = (i % 2) * 12
        if i > 0 and i % 2 == 0:
            inner_y += 7
        node_panels.append({
            "id": next_id + 1 + i,
            "type": "logs",
            "title": nid,
            "description": f"Logs for {nid}",
            "gridPos": {"x": col, "y": inner_y, "w": 12, "h": 7},
            "datasource": {"type": "loki", "uid": "${loki_ds}"},
            "targets": [
                {
                    "expr": '{service="fl_node", node_id="' + nid + '"}',
                    "refId": "A",
                }
            ],
            "options": {
                "showTime": True,
                "showLabels": True,
                "showCommonLabels": False,
                "wrapLogMessage": True,
                "prettifyLogMessage": False,
                "enableLogDetails": True,
                "sortOrder": "Descending",
                "dedupStrategy": "none",
            },
        })

    row_panel = {
        "id": next_id,
        "type": "row",
        "title": f"Per-Node Logs ({len(node_ids)} nodes)",
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "collapsed": True,
        "panels": node_panels,
    }
    panels.append(row_panel)

    dest_path.write_text(json.dumps(dashboard, indent=2) + "\n")


def _max_panel_id(panels: List[Dict[str, Any]]) -> int:
    result = 0
    for p in panels:
        result = max(result, p.get("id", 0))
        for child in p.get("panels", []):
            result = max(result, child.get("id", 0))
    return result


def _max_panel_y(panels: List[Dict[str, Any]]) -> int:
    result = 0
    for p in panels:
        gp = p.get("gridPos", {})
        result = max(result, gp.get("y", 0) + gp.get("h", 0))
    return result


def start_grafana(
    runtime_dir: Path,
    node_count: int = 0,
    node_ids: Optional[List[str]] = None,
) -> ManagedProcess:
    grafana_dir = runtime_dir / "observability" / "grafana"
    # Always start from a clean Grafana state to avoid persisting
    # dashboards/users/sessions between runs.
    if grafana_dir.exists():
        shutil.rmtree(grafana_dir)
    grafana_dir = generate_grafana_config(runtime_dir, node_count, node_ids)
    data_dir = grafana_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "logs" / "grafana.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = runtime_dir / "pids" / "grafana.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    grafana_bin = shutil.which("grafana") or shutil.which("grafana-server")
    if not grafana_bin:
        raise SystemExit("Grafana binary not found in PATH.")

    # Build a minimal env to avoid hitting the OS argument-list-too-long
    # limit when the parent environment is large.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "GF_PATHS_DATA": str(data_dir),
        "GF_PATHS_PROVISIONING": str(grafana_dir / "provisioning"),
        "GF_PATHS_LOGS": str(runtime_dir / "logs"),
        "GF_SERVER_HTTP_PORT": "3000",
        "GF_SECURITY_ADMIN_USER": "admin",
        "GF_SECURITY_ADMIN_PASSWORD": "admin",
    }

    homepath = _grafana_homepath(grafana_bin)
    log_handle = log_path.open("ab")
    proc = subprocess.Popen(
        [grafana_bin, "server", f"--homepath={homepath}"],
        cwd=grafana_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    log_handle.close()
    return ManagedProcess(
        name="grafana",
        pid=proc.pid,
        pid_file=str(pid_file),
        ports=[3000],
        log_file=str(log_path),
        component_type="monitoring",
    )


def _grafana_homepath(binary_path: str) -> str:
    """Heuristic to find Grafana's home directory from its binary location."""
    bin_dir = Path(binary_path).resolve().parent
    # Common layouts: /usr/share/grafana, /opt/homebrew/share/grafana, etc.
    for candidate in [
        bin_dir.parent / "share" / "grafana",
        bin_dir.parent,
        Path("/usr/share/grafana"),
    ]:
        if (candidate / "public").exists():
            return str(candidate)
    return str(bin_dir.parent)


def _tail_file(path: Path, line_count: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _ensure_nofile_limit(min_soft_limit: int) -> Optional[int]:
    """Best-effort bump of RLIMIT_NOFILE so promtail can watch many log files."""
    if resource is None:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return None
    if soft >= min_soft_limit:
        return soft
    new_soft = min_soft_limit
    if hard != resource.RLIM_INFINITY and hard < new_soft:
        new_soft = hard
    if new_soft > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            soft = new_soft
        except (ValueError, OSError):
            pass
    return soft
