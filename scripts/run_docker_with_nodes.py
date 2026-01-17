#!/usr/bin/env python3
"""Generate per-node configs/Compose file and launch Docker with a dynamic node count."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import shlex
from copy import deepcopy
from http import client as http_client
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'PyYAML'. Install project requirements first, e.g. "
        "`python3 -m pip install -e '.[mnist]'`.",
    ) from exc

# Add src to path for topology import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secure_aggregation.topology import generate_preliminary_topology


ROOT_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = ROOT_DIR.parent
DEFAULT_COMPOSE_TEMPLATE = ROOT_DIR / "docker" / "docker-compose.yml"
DEFAULT_COMPOSE_OUTPUT = ROOT_DIR / "docker" / "docker-compose.auto.yml"
NODE_TEMPLATE_PATH = ROOT_DIR / "config" / "node.config.template.json"
NODES_DIR = ROOT_DIR / "config" / "nodes"
KEYS_DIR = ROOT_DIR / "config" / "keys"
TOPOLOGY_FILE = ROOT_DIR / "config" / "topology.json"
STATE_MAP_FILE = ROOT_DIR / "config" / "state-map.json"
PROMETHEUS_CONFIG = ROOT_DIR / "docker" / "prometheus" / "prometheus.yml"
SYSTEM_CONFIG_FILENAME = "system-config.json"
SYSTEM_CONFIG_ENV_VAR = "SYSTEM_CONFIG_PATH"
BLOCKCHAIN_REPO_DIR = PARENT_DIR / "thesis-blockchain"
BLOCKCHAIN_API_GATEWAY_DIR = BLOCKCHAIN_REPO_DIR / "api-gateway"
BLOCKCHAIN_IDENTITY_SCRIPT = BLOCKCHAIN_API_GATEWAY_DIR / "scripts" / "generate-trainer-identities.js"
BLOCKCHAIN_ENROLL_SCRIPT = BLOCKCHAIN_API_GATEWAY_DIR / "scripts" / "enroll-trainer-identities.js"
BLOCKCHAIN_SIGN_VC_SCRIPT = BLOCKCHAIN_API_GATEWAY_DIR / "scripts" / "sign-trainer-vcs.js"
BLOCKCHAIN_BUILD_BULK_SCRIPT = BLOCKCHAIN_API_GATEWAY_DIR / "scripts" / "build-bulk-register-payload.js"
BLOCKCHAIN_SETUP_ROOT = BLOCKCHAIN_API_GATEWAY_DIR / "nodes-setup"
BLOCKCHAIN_SETUP_NODES_DIR = BLOCKCHAIN_SETUP_ROOT / "nodes"
BLOCKCHAIN_SETUP_KEYS_DIR = BLOCKCHAIN_SETUP_ROOT / "keys"
BLOCKCHAIN_SETUP_UNSIGNED_DIR = BLOCKCHAIN_SETUP_ROOT / "vc-unsigned"
BLOCKCHAIN_SETUP_SIGNED_DIR = BLOCKCHAIN_SETUP_ROOT / "vc-signed"
BLOCKCHAIN_SETUP_TOKENS_DIR = BLOCKCHAIN_SETUP_ROOT / "tokens"
BLOCKCHAIN_BULK_OUTPUT = BLOCKCHAIN_SETUP_ROOT / "bulk-register.json"
BLOCKCHAIN_API_JWT_SCRIPT = BLOCKCHAIN_API_GATEWAY_DIR / "jwt.js"
BLOCKCHAIN_VCTOOL = BLOCKCHAIN_API_GATEWAY_DIR / "api" / "vctool"
BLOCKCHAIN_ORG_DIR = BLOCKCHAIN_API_GATEWAY_DIR / "organizations" / "peerOrganizations" / "org1.nebula.com"
BLOCKCHAIN_CA_DIR = BLOCKCHAIN_ORG_DIR / "ca"
BLOCKCHAIN_USERS_DIR = BLOCKCHAIN_ORG_DIR / "users"
BLOCKCHAIN_ADMIN_HOME = BLOCKCHAIN_USERS_DIR / "Admin@org1.nebula.com"
BLOCKCHAIN_CA_ADMIN_HOME = BLOCKCHAIN_ORG_DIR / "ca-admin"
BLOCKCHAIN_CA_CERT = BLOCKCHAIN_ORG_DIR / "msp" / "cacerts" / "ca.org1.nebula.com-cert.pem"
BLOCKCHAIN_MSP_TEMPLATE = BLOCKCHAIN_ORG_DIR / "msp" / "config.yaml"
BLOCKCHAIN_COMPOSE_FILE = BLOCKCHAIN_API_GATEWAY_DIR / "docker-compose.yaml"
BLOCKCHAIN_ADMIN_PUBKEY_FILE = BLOCKCHAIN_API_GATEWAY_DIR / "admin_public_key.b64"
BLOCKCHAIN_ADMIN_KEY_FILE = BLOCKCHAIN_API_GATEWAY_DIR / "admin_ed25519_sk.pem"
BLOCKCHAIN_ENV_FILE = BLOCKCHAIN_API_GATEWAY_DIR / ".env"
BLOCKCHAIN_ENV_EXAMPLE = BLOCKCHAIN_API_GATEWAY_DIR / ".env.example"
BLOCKCHAIN_ORGANIZATIONS_DIR = BLOCKCHAIN_API_GATEWAY_DIR / "organizations"
BLOCKCHAIN_SYSTEM_GENESIS_DIR = BLOCKCHAIN_API_GATEWAY_DIR / "system-genesis-block"
BLOCKCHAIN_CHANNEL_ARTIFACTS_DIR = BLOCKCHAIN_API_GATEWAY_DIR / "channel-artifacts"
BLOCKCHAIN_CRYPTO_CONFIG = BLOCKCHAIN_API_GATEWAY_DIR / "crypto-config.yaml"
BLOCKCHAIN_CONFIGTX_DIR = BLOCKCHAIN_API_GATEWAY_DIR / "configtx"
BLOCKCHAIN_TRAINER_DB = BLOCKCHAIN_API_GATEWAY_DIR / "data" / "trainers.json"
CA_CONTAINER_NAME = "ca-org1.nebula.com"
CA_IMAGE = "hyperledger/fabric-ca:1.5"
CA_PORT = "7054"
DEFAULT_GATEWAY_URL = os.environ.get("BLOCKCHAIN_GATEWAY_URL", "http://localhost:9000")
GATEWAY_HEALTH_PATH = "/health"
GATEWAY_BULK_PATH = "/auth/register-trainers"
NODE_IMAGE_TAG = os.environ.get("SECUREAGG_NODE_IMAGE", "secureagg-node:latest")
NODE_DOCKERFILE = ROOT_DIR / "docker" / "node.Dockerfile"


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
        default=True,
        help="Run docker compose in detached mode (default: True).",
    )
    parser.add_argument(
        "--no-detach",
        action="store_true",
        help="Run docker compose in foreground mode (disables --detach).",
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
    parser.add_argument(
        "--clique-size",
        type=int,
        default=3,
        help="Size of each clique in the D-Cliques topology (default: 3).",
    )
    parser.add_argument(
        "--state-map",
        type=Path,
        help="Optional JSON file describing which nodes belong to which state. "
        "Overrides --nodes and system-config counts when provided.",
    )
    parser.add_argument(
        "--blockchain-only",
        action="store_true",
        help="Prepare artifacts and start only the blockchain/api-gateway stack.",
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


def _apply_ipfs_distribution(ipfs_service: str, config: Dict[str, Any], all_services: List[str]) -> None:
    inter_cluster = config.setdefault("inter_cluster", {})
    ipfs_section = inter_cluster.setdefault("ipfs", {})
    ipfs_section["api_url"] = f"http://{ipfs_service}:5001"
    replicas = [
        f"http://{service}:5001"
        for service in all_services
        if service != ipfs_service
    ]
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


def _ensure_blockchain_dir() -> None:
    blockchain_dir = ROOT_DIR / "data" / "blockchain"
    blockchain_dir.mkdir(parents=True, exist_ok=True)


def _ensure_ipfs_dir() -> None:
    ipfs_dir = ROOT_DIR / "data" / "ipfs"
    ipfs_dir.mkdir(parents=True, exist_ok=True)


def _reset_nodes_dir() -> None:
    if NODES_DIR.exists():
        shutil.rmtree(NODES_DIR)
    NODES_DIR.mkdir(parents=True, exist_ok=True)


def _clear_runtime_state(paths: Dict[str, Path]) -> None:
    for subdir in ("blockchain", "ipfs"):
        target = ROOT_DIR / "data" / subdir
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    trainer_db = paths.get("trainer_db")
    if trainer_db:
        trainer_db.parent.mkdir(parents=True, exist_ok=True)
        trainer_db.write_text("[\n]\n")


def _copy_directory(source: Path, destination: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Source directory not found: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_command(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    desc = description or "Command"
    kwargs: Dict[str, Any] = {"cwd": cwd, "env": env}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0:
        extra = ""
        if capture_output:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            extra = f"\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        raise SystemExit(f"{desc} failed with exit code {result.returncode}.{extra}")
    return result


def _require_blockchain_repo_paths() -> Dict[str, Path]:
    errors = []
    if not BLOCKCHAIN_REPO_DIR.exists():
        errors.append(
            f"Blockchain repo not found at {BLOCKCHAIN_REPO_DIR}. "
            "Clone thesis-blockchain alongside this repo (../thesis-blockchain).",
        )
    if not BLOCKCHAIN_API_GATEWAY_DIR.exists():
        errors.append(f"Missing api-gateway directory at {BLOCKCHAIN_API_GATEWAY_DIR}.")
    if not BLOCKCHAIN_IDENTITY_SCRIPT.exists():
        errors.append(
            "Trainer identity generator not found at "
            f"{BLOCKCHAIN_IDENTITY_SCRIPT}. Make sure the repo is up to date.",
        )
    if not BLOCKCHAIN_ENROLL_SCRIPT.exists():
        errors.append(f"Trainer enrollment script missing at {BLOCKCHAIN_ENROLL_SCRIPT}.")
    if not BLOCKCHAIN_SIGN_VC_SCRIPT.exists():
        errors.append(f"VC signing script missing at {BLOCKCHAIN_SIGN_VC_SCRIPT}.")
    if not BLOCKCHAIN_BUILD_BULK_SCRIPT.exists():
        errors.append(f"Bulk registration script missing at {BLOCKCHAIN_BUILD_BULK_SCRIPT}.")
    if not BLOCKCHAIN_SETUP_ROOT.exists():
        errors.append(f"nodes-setup directory missing at {BLOCKCHAIN_SETUP_ROOT}.")
    if not BLOCKCHAIN_VCTOOL.exists():
        errors.append(f"vctool binary missing at {BLOCKCHAIN_VCTOOL}. Build it first.")
    if not BLOCKCHAIN_API_JWT_SCRIPT.exists():
        errors.append(f"JWT helper missing at {BLOCKCHAIN_API_JWT_SCRIPT}.")
    if not BLOCKCHAIN_COMPOSE_FILE.exists():
        errors.append(f"Docker compose file missing at {BLOCKCHAIN_COMPOSE_FILE}.")
    if not BLOCKCHAIN_ENV_EXAMPLE.exists():
        errors.append(f".env.example missing at {BLOCKCHAIN_ENV_EXAMPLE}.")
    if not BLOCKCHAIN_CRYPTO_CONFIG.exists():
        errors.append(f"crypto-config.yaml missing at {BLOCKCHAIN_CRYPTO_CONFIG}.")
    if not BLOCKCHAIN_CONFIGTX_DIR.exists():
        errors.append(f"configtx directory missing at {BLOCKCHAIN_CONFIGTX_DIR}.")
    if errors:
        raise SystemExit("\n".join(errors))
    return {
        "api_gateway": BLOCKCHAIN_API_GATEWAY_DIR,
        "generate_script": BLOCKCHAIN_IDENTITY_SCRIPT,
        "enroll_script": BLOCKCHAIN_ENROLL_SCRIPT,
        "sign_script": BLOCKCHAIN_SIGN_VC_SCRIPT,
        "bulk_script": BLOCKCHAIN_BUILD_BULK_SCRIPT,
        "nodes_dir": BLOCKCHAIN_SETUP_NODES_DIR,
        "keys_dir": BLOCKCHAIN_SETUP_KEYS_DIR,
        "unsigned_dir": BLOCKCHAIN_SETUP_UNSIGNED_DIR,
        "signed_dir": BLOCKCHAIN_SETUP_SIGNED_DIR,
        "tokens_dir": BLOCKCHAIN_SETUP_TOKENS_DIR,
        "bulk_output": BLOCKCHAIN_BULK_OUTPUT,
        "org_dir": BLOCKCHAIN_ORG_DIR,
        "ca_dir": BLOCKCHAIN_CA_DIR,
        "users_dir": BLOCKCHAIN_USERS_DIR,
        "admin_home": BLOCKCHAIN_ADMIN_HOME,
        "ca_admin_home": BLOCKCHAIN_CA_ADMIN_HOME,
        "ca_cert": BLOCKCHAIN_CA_CERT,
        "msp_template": BLOCKCHAIN_MSP_TEMPLATE,
        "vctool": BLOCKCHAIN_VCTOOL,
        "jwt_script": BLOCKCHAIN_API_JWT_SCRIPT,
        "admin_jwt_path": BLOCKCHAIN_API_GATEWAY_DIR / "admin.jwt",
        "compose_file": BLOCKCHAIN_COMPOSE_FILE,
        "admin_public_key_file": BLOCKCHAIN_ADMIN_PUBKEY_FILE,
        "admin_private_key_file": BLOCKCHAIN_ADMIN_KEY_FILE,
        "env_file": BLOCKCHAIN_ENV_FILE,
        "env_example": BLOCKCHAIN_ENV_EXAMPLE,
        "organizations_dir": BLOCKCHAIN_ORGANIZATIONS_DIR,
        "system_genesis_dir": BLOCKCHAIN_SYSTEM_GENESIS_DIR,
        "channel_artifacts_dir": BLOCKCHAIN_CHANNEL_ARTIFACTS_DIR,
        "crypto_config": BLOCKCHAIN_CRYPTO_CONFIG,
        "configtx_dir": BLOCKCHAIN_CONFIGTX_DIR,
        "trainer_db": BLOCKCHAIN_TRAINER_DB,
    }


def _load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def _set_env_value(env_path: Path, key: str, value: str) -> None:
    if env_path.exists():
        lines = env_path.read_text().splitlines()
    else:
        lines = []
    found = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prefix = ""
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
            prefix = "export "
        if "=" not in stripped:
            continue
        current_key, _ = stripped.split("=", 1)
        if current_key.strip() == key:
            if stripped == f"{key}={value}":
                return
            lines[idx] = f"{prefix}{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n")


def _ensure_env_file(paths: Dict[str, Path], auth_secret: str, admin_public_key: str) -> None:
    env_path = paths["env_file"]
    if not env_path.exists():
        shutil.copy(paths["env_example"], env_path)
    _set_env_value(env_path, "AUTH_JWT_SECRET", auth_secret)
    _set_env_value(env_path, "ADMIN_PUBLIC_KEY", admin_public_key)


def _derive_admin_public_key(private_key_path: Path) -> str:
    result = subprocess.run(
        ["openssl", "pkey", "-in", str(private_key_path), "-pubout", "-outform", "DER"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Failed to extract admin public key: {result.stderr.decode().strip()}",
        )
    public_der = result.stdout
    if len(public_der) < 32:
        raise SystemExit("Unexpected admin public key DER output.")
    raw = public_der[-32:]
    return base64.b64encode(raw).decode("ascii")


def _ensure_admin_keypair(paths: Dict[str, Path]) -> str:
    private_key_path = paths["admin_private_key_file"]
    public_key_path = paths["admin_public_key_file"]
    if not private_key_path.exists():
        _run_command(
            ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private_key_path)],
            description="Generate admin Ed25519 keypair",
        )
    if not public_key_path.exists():
        pub_value = _derive_admin_public_key(private_key_path)
        public_key_path.write_text(f"{pub_value}\n")
    pub_text = public_key_path.read_text().strip()
    if not pub_text:
        pub_text = _derive_admin_public_key(private_key_path)
        public_key_path.write_text(f"{pub_text}\n")
    paths["keys_dir"].mkdir(parents=True, exist_ok=True)
    dest_key = paths["keys_dir"] / private_key_path.name
    dest_pub = paths["keys_dir"] / public_key_path.name
    shutil.copy2(private_key_path, dest_key)
    shutil.copy2(public_key_path, dest_pub)
    return pub_text


def _ensure_binary(name: str) -> None:
    result = subprocess.run([name, "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Required binary '{name}' is not available in PATH.")


def _ensure_fabric_tools_available() -> None:
    _ensure_binary("cryptogen")
    _ensure_binary("configtxgen")


def _teardown_blockchain_stack(paths: Dict[str, Path]) -> None:
    subprocess.run(
        ["docker", "compose", "down", "-v"],
        cwd=paths["api_gateway"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _teardown_system_stack(compose_dir: Path, compose_filename: str) -> None:
    compose_path = compose_dir / compose_filename
    if not compose_path.exists():
        return
    subprocess.run(
        ["docker", "compose", "-f", compose_filename, "down", "-v"],
        cwd=compose_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    # subprocess.run(
    #     ["docker", "rm", "-f", "docker"],
    #     stdout=subprocess.DEVNULL,
    #     stderr=subprocess.DEVNULL,
    #     check=False,
    # )


def _fabric_artifacts_exist(paths: Dict[str, Path]) -> bool:
    """Check if Fabric artifacts already exist and are valid."""
    org_dir = paths["organizations_dir"]
    system_dir = paths["system_genesis_dir"]
    channel_dir = paths["channel_artifacts_dir"]
    genesis_block = system_dir / "genesis.block"
    channel_tx = channel_dir / "nebula-channel.tx"
    peer_org = org_dir / "peerOrganizations" / "org1.nebula.com"
    return (
        genesis_block.exists()
        and channel_tx.exists()
        and peer_org.exists()
        and any(peer_org.iterdir())
    )


def _ensure_fabric_artifacts(paths: Dict[str, Path], force: bool = False) -> None:
    org_dir = paths["organizations_dir"]
    system_dir = paths["system_genesis_dir"]
    channel_dir = paths["channel_artifacts_dir"]

    if not force and _fabric_artifacts_exist(paths):
        print("Fabric MSP artifacts already exist, skipping regeneration...")
        return

    print("Regenerating Fabric MSP artifacts...")
    for target in (org_dir, system_dir, channel_dir):
        if target.exists():
            shutil.rmtree(target)
    org_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)
    channel_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "cryptogen",
            "generate",
            f"--config={paths['crypto_config']}",
            f"--output={org_dir}",
        ],
        cwd=paths["api_gateway"],
        description="Generate MSP material",
    )
    env = os.environ.copy()
    env["FABRIC_CFG_PATH"] = str(paths["configtx_dir"])
    _run_command(
        [
            "configtxgen",
            "-profile",
            "NebulaGenesis",
            "-channelID",
            "system-channel",
            "-outputBlock",
            str(system_dir / "genesis.block"),
        ],
        cwd=paths["api_gateway"],
        env=env,
        description="Generate system genesis block",
    )
    _run_command(
        [
            "configtxgen",
            "-profile",
            "NebulaChannel",
            "-channelID",
            "nebulachannel",
            "-outputCreateChannelTx",
            str(channel_dir / "nebula-channel.tx"),
        ],
        cwd=paths["api_gateway"],
        env=env,
        description="Generate channel transaction",
    )
    _run_command(
        [
            "configtxgen",
            "-profile",
            "NebulaChannel",
            "-channelID",
            "nebulachannel",
            "-asOrg",
            "Org1MSP",
            "-outputAnchorPeersUpdate",
            str(channel_dir / "Org1MSPanchors.tx"),
        ],
        cwd=paths["api_gateway"],
        env=env,
        description="Generate anchor peers update",
    )


def _sync_admin_cacerts(paths: Dict[str, Path]) -> None:
    admin_home = paths["admin_home"]
    admin_msp = admin_home / "msp"
    source = paths["ca_cert"]
    if not source.exists():
        raise SystemExit(f"CA certificate missing at {source}")
    dest_dir = admin_msp / "cacerts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / source.name
    shutil.copy2(source, dest_path)
    template = paths["msp_template"]
    if template.exists():
        shutil.copy2(template, admin_msp / "config.yaml")
    tlsca_source = BLOCKCHAIN_ORG_DIR / "tlsca" / "tlsca.org1.nebula.com-cert.pem"
    if tlsca_source.exists():
        tls_dir = admin_msp / "tlscacerts"
        tls_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tlsca_source, tls_dir / tlsca_source.name)


def _ensure_ca_admin_identity(paths: Dict[str, Path]) -> None:
    ca_admin_home = paths["ca_admin_home"]
    msp_dir = ca_admin_home / "msp"
    if msp_dir.exists() and any(msp_dir.iterdir()):
        return
    print("Bootstrapping Fabric CA admin identity...")
    _start_fabric_ca_container(paths)
    try:
        env = os.environ.copy()
        env["FABRIC_CA_CLIENT_HOME"] = str(ca_admin_home)
        cmd = [
            "fabric-ca-client",
            "enroll",
            "-u",
            f"http://admin:adminpw@localhost:{CA_PORT}",
            "--caname",
            "ca-org1",
            "-M",
            str(msp_dir),
        ]
        _run_command(cmd, cwd=paths["api_gateway"], env=env, description="Bootstrap CA admin")
    finally:
        print("Stopping Fabric CA container after admin bootstrap...")
        _stop_fabric_ca_container()


def _resolve_auth_secret(paths: Dict[str, Path]) -> str:
    secret = os.getenv("AUTH_JWT_SECRET")
    if secret:
        return secret
    env_path = paths["api_gateway"] / ".env"
    env_vars = _load_env_file(env_path)
    secret = env_vars.get("AUTH_JWT_SECRET")
    if secret:
        return secret
    raise SystemExit(
        "AUTH_JWT_SECRET is required but not set. Export it or add it to "
        f"{env_path} before running this script.",
    )


def _generate_trainer_identities(paths: Dict[str, Path], auth_secret: str) -> None:
    _clear_directory(paths["unsigned_dir"])
    _clear_directory(paths["signed_dir"])
    _clear_directory(paths["tokens_dir"])
    if paths["keys_dir"].exists():
        for key_file in paths["keys_dir"].glob("trainer-node-*"):
            if key_file.is_file():
                key_file.unlink()
    print(f"Syncing node configs to blockchain repo -> {paths['nodes_dir']}")
    _copy_directory(NODES_DIR, paths["nodes_dir"])
    cmd = ["node", str(paths["generate_script"]), "--generate-jwt", "registration,runtime", "--auth-secret", auth_secret]
    print("Running blockchain identity generator...")
    _run_command(cmd, cwd=paths["api_gateway"], description="Trainer identity generation")
    if not paths["keys_dir"].exists():
        raise SystemExit(
            f"Trainer keys not found at {paths['keys_dir']} after generation.",
        )
    print(f"Copying blockchain keys into system config -> {KEYS_DIR}")
    _copy_directory(paths["keys_dir"], KEYS_DIR)


def _locate_admin_private_key() -> Path:
    candidates = [
        BLOCKCHAIN_SETUP_KEYS_DIR / "admin_ed25519_sk.pem",
        BLOCKCHAIN_API_GATEWAY_DIR / "admin_ed25519_sk.pem",
        BLOCKCHAIN_REPO_DIR / "admin_ed25519_sk.pem",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Admin Ed25519 private key (admin_ed25519_sk.pem) not found in nodes-setup/keys or "
        "the blockchain repository root. Generate it per the blockchain README before proceeding.",
    )


def _read_admin_public_key(paths: Dict[str, Path]) -> str:
    candidate = paths["admin_public_key_file"]
    if not candidate.exists():
        raise SystemExit(f"Admin public key file not found at {candidate}.")
    content = candidate.read_text().strip()
    if not content:
        raise SystemExit(f"Admin public key file {candidate} is empty.")
    return content


def _ensure_fabric_ca_client() -> None:
    result = subprocess.run(
        ["fabric-ca-client", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("fabric-ca-client binary not found in PATH. Install Hyperledger Fabric CA client tools.")


def _stop_fabric_ca_container() -> None:
    subprocess.run(["docker", "rm", "-f", CA_CONTAINER_NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def _start_fabric_ca_container(paths: Dict[str, Path]) -> None:
    print("Starting Fabric CA container for trainer enrollment...")
    _stop_fabric_ca_container()
    ca_volume = f"{paths['ca_dir']}:/etc/hyperledger/fabric-ca-server/ca"
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CA_CONTAINER_NAME,
        "-p",
        f"{CA_PORT}:{CA_PORT}",
        "-v",
        ca_volume,
        "-e",
        "FABRIC_CA_SERVER_CA_NAME=ca-org1",
        "-e",
        "FABRIC_CA_SERVER_CA_CERTFILE=/etc/hyperledger/fabric-ca-server/ca/ca.org1.nebula.com-cert.pem",
        "-e",
        "FABRIC_CA_SERVER_CA_KEYFILE=/etc/hyperledger/fabric-ca-server/ca/priv_sk",
        "-e",
        "FABRIC_CA_SERVER_TLS_ENABLED=false",
        CA_IMAGE,
        "sh",
        "-c",
        f"fabric-ca-server start -b admin:adminpw --port {CA_PORT}",
    ]
    _run_command(cmd, description="Start Fabric CA container")
    time.sleep(2)


def _ensure_org_admin_identity(paths: Dict[str, Path]) -> None:
    admin_home = paths["admin_home"]
    msp_dir = admin_home / "msp"
    signcerts = msp_dir / "signcerts"
    admincerts = msp_dir / "admincerts"
    if not signcerts.exists() or not any(signcerts.iterdir()):
        raise SystemExit(
            f"Admin MSP signcerts missing at {signcerts}. Ensure cryptogen has generated admin material.",
        )
    admincerts.mkdir(parents=True, exist_ok=True)
    for cert in signcerts.iterdir():
        if cert.is_file():
            shutil.copy2(cert, admincerts / cert.name)


def _ensure_ca_admin_identity(paths: Dict[str, Path]) -> None:
    ca_admin_home = paths["ca_admin_home"]
    msp_dir = ca_admin_home / "msp"
    if msp_dir.exists():
        shutil.rmtree(msp_dir)
    ca_admin_home.mkdir(parents=True, exist_ok=True)
    print("Bootstrapping Fabric CA admin identity...")
    env = os.environ.copy()
    env["FABRIC_CA_CLIENT_HOME"] = str(ca_admin_home)
    cmd = [
        "fabric-ca-client",
        "enroll",
        "-u",
        f"http://admin:adminpw@localhost:{CA_PORT}",
        "--caname",
        "ca-org1",
        "-M",
        str(msp_dir),
    ]
    _run_command(cmd, cwd=paths["api_gateway"], env=env, description="Bootstrap CA admin")


def _enroll_trainer_msps(paths: Dict[str, Path]) -> None:
    cmd = [
        "node",
        str(paths["enroll_script"]),
        "--ca-url",
        f"http://localhost:{CA_PORT}",
        "--ca-name",
        "ca-org1",
        "--tls-cert",
        str(paths["ca_cert"]),
        "--admin-home",
        str(paths["ca_admin_home"]),
    ]
    print("Registering and enrolling trainer MSP/TLS identities...")
    env = os.environ.copy()
    env["FABRIC_CA_CLIENT_MSPDIR"] = str(paths["ca_admin_home"] / "msp")
    _run_command(cmd, cwd=paths["api_gateway"], env=env, description="Trainer enrollment")


def _sign_trainer_vcs(paths: Dict[str, Path], admin_key: Path) -> None:
    cmd = ["node", str(paths["sign_script"]), "--key", str(admin_key)]
    print("Signing trainer verifiable credentials...")
    _run_command(cmd, cwd=paths["api_gateway"], description="Sign trainer VCs")


def _build_bulk_registration_payload(paths: Dict[str, Path]) -> None:
    cmd = [
        "node",
        str(paths["bulk_script"]),
        "--did-template",
        "did:nebula:trainer-node-{trainerSeq}",
        "--output",
        str(paths["bulk_output"]),
        "--force",
    ]
    print("Building bulk registration payload...")
    _run_command(cmd, cwd=paths["api_gateway"], description="Bulk registration payload")


def _generate_admin_jwt(paths: Dict[str, Path], auth_secret: str) -> None:
    env = os.environ.copy()
    env["AUTH_JWT_SECRET"] = auth_secret
    env["JWT_ALG"] = "HS256"
    env["JWT_ROLE"] = "admin"
    env["JWT_SUB"] = "admin"
    result = _run_command(
        ["node", "jwt.js"],
        cwd=paths["api_gateway"],
        env=env,
        description="Admin JWT generation",
        capture_output=True,
    )
    token = (result.stdout or "").strip()
    if not token:
        raise SystemExit("Admin JWT generation succeeded but produced empty output.")
    paths["admin_jwt_path"].write_text(f"{token}\n")
    print(f"Admin JWT written to {paths['admin_jwt_path']}")


def _parse_compose_ps(output: str) -> List[Dict[str, Any]]:
    text = (output or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        entries: List[Dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
        return entries
    else:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    return []


def _ensure_gateway_services_running(paths: Dict[str, Path], env: Dict[str, str]) -> None:
    required = {"api-gateway", "gateway-cli"}
    for attempt in range(1, 3):
        try:
            result = _run_command(
                ["docker", "compose", "ps", "--format", "json"],
                cwd=paths["api_gateway"],
                env=env,
                description="Inspect blockchain services",
                capture_output=True,
            )
        except SystemExit:
            return
        entries = _parse_compose_ps(result.stdout or "")
        states: Dict[str, str] = {}
        for entry in entries:
            service = entry.get("Service") or entry.get("service") or entry.get("Name") or entry.get("name")
            state = entry.get("State") or entry.get("state") or ""
            if service:
                states[service] = state
        restart = [
            svc
            for svc in required
            if not states.get(svc, "").lower().startswith("running")
        ]
        if not restart:
            return
        remaining = ", ".join(sorted(restart))
        print(f"[Attempt {attempt}] Restarting blockchain services for stability: {remaining}")
        for service in sorted(restart):
            result = subprocess.run(
                ["docker", "compose", "up", "-d", service],
                cwd=paths["api_gateway"],
                env=env,
                check=False,
            )
            if result.returncode != 0:
                print(f"Service {service} failed to start (exit code {result.returncode}).")
        time.sleep(5)
    raise SystemExit(
        "Blockchain services gateway-cli/api-gateway failed to remain running after multiple restart attempts.",
    )


def _wait_for_gateway_health(base_url: str, timeout: int = 240, interval: int = 5) -> None:
    url = f"{base_url.rstrip('/')}{GATEWAY_HEALTH_PATH}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib_request.urlopen(url, timeout=15) as response:
                if 200 <= response.status < 300:
                    print(f"Gateway is healthy at {url}")
                    return
        except (urllib_error.URLError, ConnectionResetError, OSError):
            pass
        time.sleep(interval)
    raise SystemExit(f"Gateway at {url} did not become healthy within {timeout} seconds.")


def _trainer_identifier(entry: Dict[str, Any]) -> str:
    """Resolve a consistent identifier for trainer payloads/results."""
    for key in (
        "jwt_sub",
        "jwtSub",
        "JWTSub",
        "nodeId",
        "NodeID",
        "node_id",
        "did",
        "DID",
        "trainerId",
        "trainer_id",
        "subject",
        "Subject",
    ):
        value = entry.get(key)
        if value:
            return str(value)
    return "unknown"


def _parse_bulk_errors(body: str) -> List[Dict[str, Any]]:
    """Return the failing entries from a bulk registration response body."""
    if not body.strip():
        return []
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    errors: List[Dict[str, Any]] = []
    for result in results:
        status_raw = result.get("status") or result.get("Status") or ""
        if str(status_raw).lower() == "ok":
            continue
        identifier = _trainer_identifier(result)
        errors.append(
            {
                "id": identifier,
                "error": result.get("error") or result.get("Error") or "unknown error",
                "http_status": result.get("status_code") or result.get("HTTPStatus"),
            },
        )
    return errors


def _bulk_register_trainers(paths: Dict[str, Path], base_url: str) -> None:
    if not paths["admin_jwt_path"].exists():
        raise SystemExit(f"Admin JWT not found at {paths['admin_jwt_path']}.")
    if not paths["bulk_output"].exists():
        raise SystemExit(f"Bulk registration payload not found at {paths['bulk_output']}.")
    token = paths["admin_jwt_path"].read_text().strip()
    if not token:
        raise SystemExit(f"Admin JWT file {paths['admin_jwt_path']} is empty.")
    entries = json.loads(paths["bulk_output"].read_text())
    url = f"{base_url.rstrip('/')}{GATEWAY_BULK_PATH}"
    batch_size = 5
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    total_batches = (len(entries) + batch_size - 1) // batch_size
    for offset in range(0, len(entries), batch_size):
        batch = entries[offset : offset + batch_size]
        batch_index = offset // batch_size + 1
        trainer_ids = [_trainer_identifier(entry) for entry in batch]
        print(
            f"Registering trainers {', '.join(trainer_ids)} (batch {batch_index}/{total_batches})",
        )
        pending = list(batch)
        attempt = 1
        last_errors: List[str] = []
        while pending and attempt <= 3:
            pending_ids = [_trainer_identifier(entry) for entry in pending]
            payload = json.dumps(pending).encode()
            request = urllib_request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib_request.urlopen(request, timeout=60) as response:
                    body = response.read().decode("utf-8", errors="replace")
                errors = _parse_bulk_errors(body)
                if not errors:
                    print(f"Registered trainers {', '.join(pending_ids)} successfully.")
                    pending = []
                    break
                last_errors = [
                    f"{err['id']}: {err['error']} (HTTP {err.get('http_status') or 'unknown'})"
                    for err in errors
                ]
                print(
                    f"Batch {batch_index} attempt {attempt} encountered {len(errors)} errors; "
                    "retrying the affected trainers.",
                )
                entry_map = {_trainer_identifier(entry): entry for entry in pending}
                pending = [
                    entry_map[err["id"]]
                    for err in errors
                    if err["id"] in entry_map
                ]
            except urllib_error.HTTPError as exc:
                if exc.code == 409:
                    print(
                        f"Trainers {', '.join(pending_ids)} were already registered (HTTP 409).",
                    )
                    pending = []
                    break
                body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
                last_errors = [f"HTTP {exc.code}: {body.strip()}"]
                if attempt >= 3:
                    raise SystemExit(
                        f"Bulk registration batch {batch_index} failed "
                        f"(HTTP {exc.code}): {body.strip()}",
                    )
                print(
                    f"Batch {batch_index} attempt {attempt} failed with HTTP {exc.code}; retrying...",
                )
            except (urllib_error.URLError, http_client.RemoteDisconnected) as err:
                last_errors = [str(err)]
                if attempt >= 3:
                    raise SystemExit(
                        f"Batch {batch_index} failed: {err}",
                    ) from err
                print(
                    f"Batch {batch_index} attempt {attempt} failed: {err}. Retrying...",
                )
            attempt += 1
            if pending and attempt <= 3:
                time.sleep(5)
        if pending:
            detail = "; ".join(last_errors) or "unknown error"
            failed_ids = ", ".join(_trainer_identifier(entry) for entry in pending)
            raise SystemExit(
                f"Bulk registration batch {batch_index} failed for trainers {failed_ids}: {detail}",
            )


def _start_blockchain_stack(paths: Dict[str, Path], auth_secret: str) -> str:
    admin_public_key = _read_admin_public_key(paths)
    env = os.environ.copy()
    env["AUTH_JWT_SECRET"] = auth_secret
    env["ADMIN_PUBLIC_KEY"] = admin_public_key
    env.setdefault("DOCKER_BUILDKIT", "1")
    print("Starting blockchain network via docker compose...")
    result = subprocess.run(
        ["docker", "compose", "up", "--build", "-d"],
        cwd=paths["api_gateway"],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        print(
            "Blockchain docker compose up exited with a non-zero status. "
            "Attempting to restart gateway services automatically...",
        )
    _ensure_gateway_services_running(paths, env)
    return env.get("BLOCKCHAIN_GATEWAY_URL", DEFAULT_GATEWAY_URL)


def _prepare_blockchain_artifacts(paths: Dict[str, Path], auth_secret: str) -> None:
    admin_public_key = _ensure_admin_keypair(paths)
    _ensure_env_file(paths, auth_secret, admin_public_key)
    _teardown_blockchain_stack(paths)
    _ensure_fabric_tools_available()
    _ensure_fabric_artifacts(paths)
    _sync_admin_cacerts(paths)
    _generate_trainer_identities(paths, auth_secret)
    _ensure_fabric_ca_client()
    _ensure_org_admin_identity(paths)
    try:
        _start_fabric_ca_container(paths)
        _ensure_ca_admin_identity(paths)
        _enroll_trainer_msps(paths)
    finally:
        print("Stopping Fabric CA container...")
        _stop_fabric_ca_container()
    admin_key = _locate_admin_private_key()
    _sign_trainer_vcs(paths, admin_key)
    _build_bulk_registration_payload(paths)
    _generate_admin_jwt(paths, auth_secret)


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


def _resolve_state_map_path(cli_path: Optional[Path]) -> Optional[Path]:
    if cli_path:
        return _resolve_repo_path(cli_path).resolve()
    default_path = STATE_MAP_FILE
    if default_path.exists():
        return default_path
    return None


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


def _load_state_assignments(state_map_path: Optional[Path]) -> List[Tuple[str, str, str]]:
    """
    Load state-to-node assignments from configuration file.

    The JSON schema is:
    {
        "states": [
            {"state_id": "state_a", "count": 20},
            {"state_id": "state_b", "count": 15}
        ]
    }
    The system always creates container names sequentially (node_0, node_1, ...),
    but each state may optionally define the logical node identifiers via the
    "nodes" list. When provided, those identifiers become the node_id and
    blockchain identity inside the generated config while the container keeps its
    predictable node_X name.
    """
    if state_map_path is None:
        return []
    if not state_map_path.exists():
        raise SystemExit(f"State map file not found: {state_map_path}")
    try:
        data = json.loads(state_map_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in state map {state_map_path}: {exc}") from exc

    states = data.get("states")
    if not isinstance(states, list) or not states:
        raise SystemExit(f"State map {state_map_path} must contain a non-empty 'states' list.")

    assignments: List[Tuple[str, str, str]] = []
    next_index = 0
    for entry in states:
        if not isinstance(entry, dict):
            raise SystemExit(f"Invalid state entry in {state_map_path}: {entry!r}")
        state_id = entry.get("state_id") or entry.get("id")
        if not state_id:
            raise SystemExit(f"State entry missing 'state_id': {entry!r}")
        explicit_nodes = entry.get("nodes")
        if explicit_nodes:
            if not isinstance(explicit_nodes, list):
                raise SystemExit(f"'nodes' for state {state_id} must be a list")
            for alias in explicit_nodes:
                alias_str = str(alias).strip()
                if not alias_str:
                    raise SystemExit(f"Invalid node identifier in state {state_id}: {alias!r}")
                node_name = f"node_{next_index}"
                trainer_id = _normalize_trainer_id(alias_str, next_index)
                assignments.append((node_name, str(state_id), trainer_id))
                next_index += 1
            count = entry.get("count")
            if count is not None and int(count) != len(explicit_nodes):
                raise SystemExit(
                    f"State {state_id} specifies count={count} but provided {len(explicit_nodes)} nodes."
                )
            continue
        count = entry.get("count")
        if count is None:
            raise SystemExit(
                f"State entry for {state_id} must include either 'nodes' or 'count'."
            )
        try:
            count_val = int(count)
        except (TypeError, ValueError):
            raise SystemExit(f"Invalid count for state {state_id}: {count!r}") from None
        if count_val <= 0:
            raise SystemExit(f"State {state_id} must have count >= 1 (found {count_val}).")
        for _ in range(count_val):
            node_name = f"node_{next_index}"
            trainer_id = _normalize_trainer_id(None, next_index)
            assignments.append((node_name, str(state_id), trainer_id))
            next_index += 1
    return assignments


def load_compose_template(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Docker compose template not found at {path}")
    return yaml.safe_load(path.read_text())


def _extract_ipfs_services(compose_data: Dict[str, Any]) -> List[str]:
    services = compose_data.get("services", {})
    ipfs_services = [name for name in services if name.startswith("ipfs-node")]
    return sorted(ipfs_services, key=_ipfs_sort_key)


def _apply_shared_node_image(service: Optional[Dict[str, Any]], image_tag: Optional[str]) -> None:
    """Ensure a service reuses the shared node image to avoid redundant builds."""
    if not service or not image_tag:
        return
    service["image"] = image_tag
    service.pop("build", None)


def _docker_image_exists(image_tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _build_shared_node_image(image_tag: str) -> None:
    print(f"Building shared training node image '{image_tag}' from {NODE_DOCKERFILE}...")
    cmd = [
        "docker",
        "build",
        "-t",
        image_tag,
        "-f",
        str(NODE_DOCKERFILE),
        str(ROOT_DIR),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"docker build failed with exit code {result.returncode}; "
            "see the logs above for details.",
        )


def _ensure_shared_node_image(image_tag: str, skip_build: bool) -> None:
    if skip_build:
        if not _docker_image_exists(image_tag):
            raise SystemExit(
                f"Shared node image '{image_tag}' not found. "
                "Run without --no-build (or build it manually) before skipping builds.",
            )
        print(f"Skipping rebuild of '{image_tag}' (already present, --no-build specified).")
        return
    _build_shared_node_image(image_tag)


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
    image_tag: Optional[str] = None,
) -> Dict[str, Any]:
    service = deepcopy(base_template)
    _apply_shared_node_image(service, image_tag)
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
    if image_tag is None:
        service.setdefault("build", DEFAULT_NODE_SERVICE["build"])
    service.setdefault("networks", DEFAULT_NODE_SERVICE["networks"])
    service.setdefault("env_file", DEFAULT_NODE_SERVICE["env_file"])
    service.setdefault("environment", DEFAULT_NODE_SERVICE["environment"])
    service.setdefault("volumes", DEFAULT_NODE_SERVICE["volumes"])
    return service


def _ensure_argument(args: List[str], arg_name: str, value: int) -> List[str]:
    """Ensure a command-line argument has the specified value."""
    str_value = str(value)
    for idx, token in enumerate(args):
        if token == arg_name:
            if idx + 1 < len(args):
                args[idx + 1] = str_value
            else:
                args.append(str_value)
            return args
        if token.startswith(f"{arg_name}="):
            args[idx] = f"{arg_name}={str_value}"
            return args
    args.extend([arg_name, str_value])
    return args


def _update_ttp_command(command: Any, num_nodes: int, clique_size: int) -> Any:
    """Update the TTP command definition with node count and clique size."""
    if isinstance(command, list):
        args = list(command)
        args = _ensure_argument(args, "--num-clients", num_nodes)
        args = _ensure_argument(args, "--clique-size", clique_size)
        return args
    if isinstance(command, str):
        tokens = shlex.split(command)
        tokens = _ensure_argument(tokens, "--num-clients", num_nodes)
        tokens = _ensure_argument(tokens, "--clique-size", clique_size)
        return shlex.join(tokens)
    return command


def _update_environment_section(section: Any, key: str, value: int) -> Any:
    str_value = str(value)
    if isinstance(section, dict):
        section[key] = str_value
        return section
    if isinstance(section, list):
        prefix = f"{key}="
        for idx, entry in enumerate(section):
            if isinstance(entry, str) and entry.startswith(prefix):
                section[idx] = f"{prefix}{str_value}"
                break
        else:
            section.append(f"{prefix}{str_value}")
        return section
    return section


def _update_ttp_service(
    services: Dict[str, Any],
    num_nodes: int,
    clique_size: int,
    image_tag: Optional[str] = None,
) -> None:
    """Propagate the generated node count and clique size to the TTP service config."""
    service = services.get("ttp")
    if not service:
        return
    _apply_shared_node_image(service, image_tag)
    if "command" in service:
        service["command"] = _update_ttp_command(service["command"], num_nodes, clique_size)
    if "environment" in service:
        service["environment"] = _update_environment_section(service["environment"], "NUM_CLIENTS", num_nodes)
        service["environment"] = _update_environment_section(service["environment"], "CLIQUE_SIZE", clique_size)


def update_compose_file(
    compose_data: Dict[str, Any],
    node_names: List[str],
    ipfs_services: List[str],
    clique_size: int = 3,
    node_image: Optional[str] = None,
) -> Dict[str, Any]:
    services = compose_data.setdefault("services", {})
    base_node_service = _locate_node_template(compose_data)
    # Remove any pre-existing node definitions.
    for key in list(services.keys()):
        if key.startswith("node_"):
            services.pop(key)
    if "node" in services:
        _apply_shared_node_image(services.get("node"), node_image)
    for idx, node_name in enumerate(node_names):
        ipfs_service = _select_ipfs_service(idx, ipfs_services)
        services[node_name] = _merge_node_service(node_name, base_node_service, ipfs_service, node_image)
    _update_ttp_service(services, len(node_names), clique_size, node_image)
    return compose_data


def write_compose_file(data: Dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def generate_prometheus_config(node_names: List[str]) -> None:
    """Generate prometheus.yml with targets for all nodes."""
    targets = [f"{name}:8000" for name in node_names]
    config = {
        "global": {
            "scrape_interval": "5s",
            "evaluation_interval": "5s",
        },
        "scrape_configs": [
            {
                "job_name": "fl_nodes",
                "static_configs": [
                    {
                        "targets": targets,
                        "labels": {"group": "training_nodes"},
                    }
                ],
            }
        ],
    }
    PROMETHEUS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PROMETHEUS_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Generated Prometheus config for {len(node_names)} nodes -> {PROMETHEUS_CONFIG}")


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
    if args.blockchain_only and args.generate_only:
        raise SystemExit("--blockchain-only cannot be combined with --generate-only.")

    compose_template_path = _resolve_repo_path(args.compose_template)
    compose_output_path = _resolve_repo_path(args.compose_output)
    system_config_path = resolve_system_config_path(args.system_config)
    state_map_path = _resolve_state_map_path(args.state_map)
    state_assignments = _load_state_assignments(state_map_path)
    if state_assignments:
        node_names = [name for name, _, _ in state_assignments]
        node_count = len(node_names)
        state_counts: Dict[str, int] = {}
        for _, state_id, _ in state_assignments:
            state_counts[state_id or "unassigned"] = state_counts.get(state_id or "unassigned", 0) + 1
        state_summary = ", ".join(f"{state}: {count}" for state, count in state_counts.items())
        print(f"Loaded state map {state_map_path} ({state_summary})")
    else:
        node_count = determine_node_count(args.nodes, system_config_path)
        node_names = [f"node_{i}" for i in range(node_count)]
        state_assignments = [(name, "", _normalize_trainer_id(None, idx)) for idx, name in enumerate(node_names)]
        print(
            f"Using node count {node_count} from "
            f"{'--nodes CLI' if args.nodes is not None else system_config_path}"
        )
    shared_node_image = NODE_IMAGE_TAG

    template_config = load_node_template()
    compose_template = load_compose_template(compose_template_path)
    ipfs_services = _extract_ipfs_services(compose_template)

    blockchain_paths = _require_blockchain_repo_paths()
    auth_secret = _resolve_auth_secret(blockchain_paths)
    _teardown_blockchain_stack(blockchain_paths)
    _teardown_system_stack(compose_output_path.parent, compose_output_path.name)
    _clear_runtime_state(blockchain_paths)
    _reset_nodes_dir()
    # Generate configs with the correct IPFS/blockchain layout.
    for idx, (container_name, state_id, trainer_id) in enumerate(state_assignments):
        config = deepcopy(template_config)
        config["node_id"] = trainer_id
        config["trainer_id"] = trainer_id
        if state_id:
            config["state_id"] = state_id
        ipfs_service = _select_ipfs_service(idx, ipfs_services)
        _apply_ipfs_distribution(ipfs_service, config, ipfs_services)
        _apply_blockchain_identity(idx, config, identity_override=trainer_id)
        config_path = NODES_DIR / f"{container_name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    updated_compose = update_compose_file(
        compose_template,
        node_names,
        ipfs_services,
        args.clique_size,
        shared_node_image,
    )
    compose_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_compose_file(updated_compose, compose_output_path)
    print(f"Generated docker compose file with {node_count} nodes, clique_size={args.clique_size} -> {compose_output_path}")
    if state_map_path:
        print(f"(node count source: {state_map_path})")
    else:
        print(f"(node count source: {'--nodes CLI' if args.nodes is not None else system_config_path})")

    generate_prometheus_config(node_names)

    topology_data = generate_preliminary_topology(node_count, args.clique_size)
    TOPOLOGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOPOLOGY_FILE.write_text(json.dumps(topology_data, indent=2) + "\n")
    print(f"Generated preliminary topology: {topology_data['num_cliques']} cliques -> {TOPOLOGY_FILE}")

    dashboard_script = ROOT_DIR / "scripts" / "generate_grafana_dashboard.py"
    if dashboard_script.exists():
        print("Generating Grafana dashboard...")
        subprocess.run([sys.executable, str(dashboard_script)], check=False)

    _prepare_blockchain_artifacts(blockchain_paths, auth_secret)

    if args.generate_only:
        print("Skipped docker compose up (generate-only mode).")
        print(f"Run: (cd docker && docker compose -f {compose_output_path.name} up --build)")
        print(
            f"Build the shared node image with: docker build -t {shared_node_image} "
            f"-f docker/node.Dockerfile {ROOT_DIR}",
        )
        return

    if not args.blockchain_only:
        _ensure_shared_node_image(shared_node_image, skip_build=args.no_build)

    gateway_base_url = _start_blockchain_stack(blockchain_paths, auth_secret)
    _wait_for_gateway_health(gateway_base_url)
    _bulk_register_trainers(blockchain_paths, gateway_base_url)

    if args.blockchain_only:
        print("Blockchain network and API gateway are running. Skipping IPFS/FL stack (--blockchain-only).")
        print(f"Gateway URL: {gateway_base_url}")
        return

    build = not args.no_build
    detach = args.detach and not args.no_detach
    return_code = run_docker_compose(compose_output_path, detach, build)
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
