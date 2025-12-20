#!/usr/bin/env python3
"""
Run a federated learning cluster with configurable number of nodes.

Usage:
    python scripts/run_cluster.py --nodes 6
    python scripts/run_cluster.py --nodes 10 --clique-size 5
    python scripts/run_cluster.py --nodes 6 --generate-only
    python scripts/run_cluster.py --stop
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
NODES_DIR = CONFIG_DIR / "nodes"
KEYS_DIR = CONFIG_DIR / "keys"
DOCKER_DIR = PROJECT_ROOT / "docker"


def generate_topology(num_nodes: int, clique_size: int) -> dict:
    """Generate topology.json with nodes distributed into cliques."""
    num_cliques = math.ceil(num_nodes / clique_size)
    cliques = []
    inter_edges = []

    for i in range(num_cliques):
        start_idx = i * clique_size
        end_idx = min(start_idx + clique_size, num_nodes)
        clique_nodes = [f"node_{j}" for j in range(start_idx, end_idx)]
        cliques.append(clique_nodes)

    # Create inter-cluster edges (connect first node of each clique to first node of next).
    for i in range(num_cliques - 1):
        if cliques[i] and cliques[i + 1]:
            inter_edges.append([cliques[i][0], cliques[i + 1][0]])
            # Add secondary edge if cliques have enough nodes.
            if len(cliques[i]) > 1 and len(cliques[i + 1]) > 1:
                inter_edges.append([cliques[i][1], cliques[i + 1][1]])

    return {
        "num_cliques": num_cliques,
        "cliques": cliques,
        "inter_edges": inter_edges,
    }


def generate_node_config(node_idx: int, num_nodes: int, clique_size: int) -> dict:
    """Generate configuration for a single node."""
    threshold = max(2, clique_size - 1)

    return {
        "node_id": f"node_{node_idx}",
        "role": "hybrid",
        "ttp_address": "ttp:50051",
        "port": 50052,
        "dataset": {
            "name": "mnist",
            "num_clients": num_nodes,
            "alpha": 0.5,
            "seed": 42,
        },
        "training": {
            "local_epochs": 1,
            "batch_size": 64,
            "rounds": 3,
        },
        "secure_agg": {
            "threshold": threshold,
            "scale": 1000000.0,
        },
        "inter_cluster": {
            "enabled": True,
            "use_mock": False,
            "topology_file": "/app/config/topology.json",
            "ipfs": {"api_url": "http://ipfs-node-1:5001"},
            "blockchain": {
                "gateway_url": "http://gateway:9000",
                "identity": f"trainer-node-{node_idx:03d}",
                "private_key_path": f"/app/config/keys/trainer-node-{node_idx:03d}_sk.pem",
                "state_path": f"data/blockchain/trainer-node-{node_idx:03d}.json",
                "jwt_role": "trainer",
                "jwt_state": "system",
            },
            "window_size": 10,
            "alpha": 0.5,
            "base_gamma": 0.2,
            "freshness_window": 300.0,
            "max_neighbors": 4,
        },
    }


def generate_docker_compose(num_nodes: int, clique_size: int) -> str:
    """Generate docker-compose.yml content for the specified number of nodes."""
    lines = [
        'version: "3.9"',
        "",
        "services:",
        "  prometheus:",
        "    image: prom/prometheus:v2.47.0",
        "    container_name: prometheus",
        "    hostname: prometheus",
        "    command:",
        "      - '--config.file=/etc/prometheus/prometheus.yml'",
        "      - '--storage.tsdb.path=/prometheus'",
        "      - '--web.enable-lifecycle'",
        "      - '--storage.tsdb.retention.time=1h'",
        "    volumes:",
        "      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml",
        "    ports:",
        '      - "9090:9090"',
        "    networks:",
        "      - secureagg",
        "",
        "  dashboard-generator:",
        "    image: python:3.11-slim",
        "    container_name: dashboard-generator",
        "    volumes:",
        "      - ../config:/app/config:ro",
        "      - ../scripts:/app/scripts:ro",
        "      - ./grafana/dashboards:/app/output",
        "    working_dir: /app",
        '    command: ["python", "scripts/generate_grafana_dashboard.py"]',
        "    networks:",
        "      - secureagg",
        "",
        "  grafana:",
        "    image: grafana/grafana:10.0.0",
        "    container_name: grafana",
        "    restart: always",
        "    hostname: grafana",
        "    environment:",
        "      - GF_SECURITY_ADMIN_USER=admin",
        "      - GF_SECURITY_ADMIN_PASSWORD=admin",
        "      - GF_USERS_ALLOW_SIGN_UP=false",
        "    volumes:",
        "      - ./grafana/provisioning:/etc/grafana/provisioning",
        "      - ./grafana/dashboards:/var/lib/grafana/dashboards",
        "      - grafana_data:/var/lib/grafana",
        "    ports:",
        '      - "3000:3000"',
        "    networks:",
        "      - secureagg",
        "    depends_on:",
        "      dashboard-generator:",
        "        condition: service_completed_successfully",
        "      prometheus:",
        "        condition: service_started",
        "",
        "  ipfs-node-1:",
        "    image: ipfs/kubo:v0.29.0",
        "    container_name: ipfs-node-1",
        "    restart: always",
        "    hostname: ipfs-node-1",
        "    environment:",
        '      NODE_NAME: "ipfs-node-1"',
        "    volumes:",
        "      - ../data/ipfs/node-1:/data/ipfs",
        "      - ./ipfs-bootstrap.sh:/container-bootstrap.sh",
        "    networks:",
        "      - secureagg",
        '    entrypoint: ["/bin/sh", "/container-bootstrap.sh"]',
        "    healthcheck:",
        '      test: ["CMD", "ipfs", "id"]',
        "      interval: 5s",
        "      timeout: 3s",
        "      retries: 10",
        "",
        "  registry:",
        "    build:",
        "      context: ..",
        "      dockerfile: docker/registry.Dockerfile",
        "    container_name: model-registry",
        "    restart: always",
        "    hostname: registry",
        "    environment:",
        '      REGISTRY_STORAGE_PATH: "/data/registry.json"',
        "    volumes:",
        "      - ../data/registry:/data",
        "    networks:",
        "      - secureagg",
        "    healthcheck:",
        '      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]',
        "      interval: 5s",
        "      timeout: 3s",
        "      retries: 5",
        "",
        "  gateway:",
        "    build:",
        "      context: ..",
        "      dockerfile: docker/gateway.Dockerfile",
        "    container_name: blockchain-gateway",
        "    restart: always",
        "    hostname: gateway",
        "    environment:",
        "      - GATEWAY_STORAGE_PATH=/data/blockchain_gateway.json",
        "      - GATEWAY_PUBLIC_KEYS_DIR=/app/config/keys",
        "    volumes:",
        "      - ../data/gateway:/data",
        "      - ../config/keys:/app/config/keys",
        "      - ../src:/app/src",
        "    networks:",
        "      - secureagg",
        "    healthcheck:",
        '      test: ["CMD", "curl", "-f", "http://localhost:9000/health"]',
        "      interval: 5s",
        "      timeout: 3s",
        "      retries: 5",
        "",
        "  ttp:",
        "    build:",
        "      context: ..",
        "      dockerfile: docker/node.Dockerfile",
        "    container_name: ttp",
        f'    command: ["python", "scripts/run_ttp_with_topology.py", "--num-clients", "{num_nodes}", "--clique-size", "{clique_size}", "--topology-output", "/app/config/topology.json"]',
        "    networks:",
        "      - secureagg",
        "    ports:",
        '      - "50051:50051"',
        "    environment:",
        "      - PYTHONUNBUFFERED=1",
        f"      - NUM_CLIENTS={num_nodes}",
        f"      - CLIQUE_SIZE={clique_size}",
        "      - ALPHA=0.5",
        "      - SEED=42",
        "    volumes:",
        "      - ../data:/app/data",
        "      - ../logs:/app/logs",
        "      - ../config:/app/config",
        "      - ../src:/app/src",
        "    healthcheck:",
        '      test: ["CMD", "python", "-c", "import os; exit(0 if os.path.exists(\'/app/config/topology.json\') else 1)"]',
        "      interval: 2s",
        "      timeout: 5s",
        "      retries: 30",
        "",
    ]

    # Generate node services.
    for i in range(num_nodes):
        lines.extend(
            [
                f"  node_{i}:",
                "    build:",
                "      context: ..",
                "      dockerfile: docker/node.Dockerfile",
                f"    container_name: node_{i}",
                f'    command: ["sh", "-c", "sleep 10 && python -m secure_aggregation.communication.node_service --config /app/config/nodes/node_{i}.json"]',
                "    networks:",
                "      - secureagg",
                "    depends_on:",
                "      ttp:",
                "        condition: service_healthy",
                "      ipfs-node-1:",
                "        condition: service_healthy",
                "      registry:",
                "        condition: service_healthy",
                "      gateway:",
                "        condition: service_healthy",
                "    environment:",
                "      - PYTHONUNBUFFERED=1",
                "      - MAX_TRAINING_ROUNDS=10000",
                "      - PROMETHEUS_PORT=8000",
                "    volumes:",
                "      - ../config:/app/config",
                "      - ../data:/app/data",
                "      - ../logs:/app/logs",
                "      - ../checkpoints:/app/checkpoints",
                "      - ../src:/app/src",
                "",
            ]
        )

    lines.extend(
        [
            "networks:",
            "  secureagg:",
            "    driver: bridge",
            "",
            "volumes:",
            "  grafana_data:",
            "",
        ]
    )

    return "\n".join(lines)


def generate_keys(num_nodes: int) -> None:
    """Generate Ed25519 key pairs for all nodes."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError:
        print("Warning: cryptography not installed, skipping key generation")
        print("Run: pip install cryptography")
        return

    KEYS_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(num_nodes):
        identity = f"trainer-node-{i:03d}"
        sk_path = KEYS_DIR / f"{identity}_sk.pem"
        pk_path = KEYS_DIR / f"{identity}_pk.pem"

        # Skip if key already exists.
        if sk_path.exists() and pk_path.exists():
            continue

        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        sk_path.write_bytes(private_pem)
        pk_path.write_bytes(public_pem)
        print(f"  Generated keys for {identity}")


def generate_prometheus_config(num_nodes: int) -> str:
    """Generate prometheus.yml with all node targets."""
    targets = [f"node_{i}:8000" for i in range(num_nodes)]
    targets_str = "\n".join(f"          - '{t}'" for t in targets)

    return f"""global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'fl_nodes'
    static_configs:
      - targets:
{targets_str}
        labels:
          group: 'training_nodes'
"""


def write_configs(num_nodes: int, clique_size: int) -> None:
    """Write all configuration files."""
    NODES_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old node configs.
    for old_config in NODES_DIR.glob("node_*.json"):
        old_config.unlink()

    # Generate keys for all nodes.
    print("Generating keys...")
    generate_keys(num_nodes)

    # Generate topology.
    topology = generate_topology(num_nodes, clique_size)
    topology_path = CONFIG_DIR / "topology.json"
    with open(topology_path, "w") as f:
        json.dump(topology, f, indent=2)
    print(f"Generated {topology_path}")

    # Generate node configs.
    for i in range(num_nodes):
        config = generate_node_config(i, num_nodes, clique_size)
        config_path = NODES_DIR / f"node_{i}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    print(f"Generated {num_nodes} node configs in {NODES_DIR}")

    # Generate docker-compose.
    compose_content = generate_docker_compose(num_nodes, clique_size)
    compose_path = DOCKER_DIR / "docker-compose.generated.yml"
    with open(compose_path, "w") as f:
        f.write(compose_content)
    print(f"Generated {compose_path}")

    # Generate prometheus config.
    prometheus_content = generate_prometheus_config(num_nodes)
    prometheus_path = DOCKER_DIR / "prometheus" / "prometheus.yml"
    prometheus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prometheus_path, "w") as f:
        f.write(prometheus_content)
    print(f"Generated {prometheus_path}")


def generate_grafana_dashboard() -> None:
    """Generate the Grafana dashboard from topology."""
    script_path = PROJECT_ROOT / "scripts" / "generate_grafana_dashboard.py"
    subprocess.run([sys.executable, str(script_path)], check=True)


def start_cluster(compose_file: Path) -> None:
    """Start the Docker Compose cluster."""
    print("\n=== Starting Docker Compose ===")
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d", "--build"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("\n=== Services Started ===")
    print("Grafana:    http://localhost:3000 (admin/admin)")
    print("Prometheus: http://localhost:9090")
    print(f"\nUse 'docker compose -f {compose_file} logs -f' to view logs")


def stop_cluster(quiet: bool = False, remove_volumes: bool = True) -> None:
    """Stop all running clusters and remove volumes to ensure fresh dashboard on restart."""
    compose_files = [
        DOCKER_DIR / "docker-compose.generated.yml",
        DOCKER_DIR / "docker-compose.6nodes.yml",
    ]
    for compose_file in compose_files:
        if compose_file.exists():
            if not quiet:
                print(f"Stopping {compose_file}...")
            cmd = ["docker", "compose", "-f", str(compose_file), "down"]
            if remove_volumes:
                cmd.append("-v")
            subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=quiet)


def clean_data() -> None:
    """Remove data directories for a fresh start."""
    data_dirs = [
        PROJECT_ROOT / "data" / "ipfs",
        PROJECT_ROOT / "data" / "registry",
        PROJECT_ROOT / "data" / "gateway",
        PROJECT_ROOT / "data" / "blockchain",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "checkpoints",
    ]
    for data_dir in data_dirs:
        if data_dir.exists():
            print(f"Removing {data_dir}...")
            shutil.rmtree(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a federated learning cluster with configurable nodes."
    )
    parser.add_argument(
        "--nodes",
        "-n",
        type=int,
        default=6,
        help="Number of nodes to run (default: 6)",
    )
    parser.add_argument(
        "--clique-size",
        "-c",
        type=int,
        default=3,
        help="Number of nodes per clique/cluster (default: 3)",
    )
    parser.add_argument(
        "--generate-only",
        "-g",
        action="store_true",
        help="Only generate configs, don't start the cluster",
    )
    parser.add_argument(
        "--stop",
        "-s",
        action="store_true",
        help="Stop running cluster",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean data directories before starting (fresh start)",
    )

    args = parser.parse_args()

    if args.stop:
        stop_cluster()
        return

    num_nodes = args.nodes
    clique_size = args.clique_size

    if clique_size > num_nodes:
        clique_size = num_nodes
        print(f"Adjusted clique_size to {clique_size} (cannot exceed num_nodes)")

    num_cliques = math.ceil(num_nodes / clique_size)
    print(f"=== Cluster Configuration ===")
    print(f"Nodes:        {num_nodes}")
    print(f"Clique size:  {clique_size}")
    print(f"Num cliques:  {num_cliques}")
    print()

    print("=== Generating Configurations ===")
    write_configs(num_nodes, clique_size)

    print("\n=== Generating Grafana Dashboard ===")
    generate_grafana_dashboard()

    if args.generate_only:
        print("\nConfigs generated. Use --start to run the cluster.")
        return

    # Stop any previously running cluster before starting new one.
    print("\n=== Stopping Previous Cluster (if any) ===")
    stop_cluster(quiet=True)

    if args.clean:
        print("\n=== Cleaning Data Directories ===")
        clean_data()

    compose_file = DOCKER_DIR / "docker-compose.generated.yml"
    start_cluster(compose_file)


if __name__ == "__main__":
    main()
