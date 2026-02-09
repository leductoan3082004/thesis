"""Blockchain artifact preparation, trainer registration, and Fabric CA management (process-only)."""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from http import client as http_client
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

ROOT_DIR = Path(__file__).resolve().parents[2]
PARENT_DIR = ROOT_DIR.parent

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
BLOCKCHAIN_COMPOSE_FILE = BLOCKCHAIN_API_GATEWAY_DIR / "docker-compose.yaml"
BLOCKCHAIN_PROCESS_RUNNER = BLOCKCHAIN_API_GATEWAY_DIR / "process-runner" / "manage.sh"
BLOCKCHAIN_PROCESS_RUNTIME_DIR = BLOCKCHAIN_PROCESS_RUNNER.parent / "runtime"

CA_PORT = "7054"
DEFAULT_GATEWAY_URL = os.environ.get("BLOCKCHAIN_GATEWAY_URL", "http://localhost:9000")
GATEWAY_HEALTH_PATH = "/health"
GATEWAY_BULK_PATH = "/auth/register-trainers"
KEYS_DIR = ROOT_DIR / "config" / "keys"
NODES_DIR = ROOT_DIR / "config" / "nodes"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


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
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0:
        extra = ""
        if capture_output:
            extra = f"\nSTDOUT:\n{result.stdout or ''}\nSTDERR:\n{result.stderr or ''}"
        raise SystemExit(f"{desc} failed with exit code {result.returncode}.{extra}")
    return result


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


def _load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _set_env_value(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    found = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prefix = ""
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].strip()
            prefix = "export "
        if "=" not in stripped:
            continue
        current_key, _ = stripped.split("=", 1)
        if current_key.strip() == key:
            lines[idx] = f"{prefix}{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def require_blockchain_repo_paths() -> Dict[str, Path]:
    errors: List[str] = []
    checks = [
        (BLOCKCHAIN_REPO_DIR, "Blockchain repo"),
        (BLOCKCHAIN_API_GATEWAY_DIR, "api-gateway directory"),
        (BLOCKCHAIN_IDENTITY_SCRIPT, "Trainer identity generator"),
        (BLOCKCHAIN_ENROLL_SCRIPT, "Trainer enrollment script"),
        (BLOCKCHAIN_SIGN_VC_SCRIPT, "VC signing script"),
        (BLOCKCHAIN_BUILD_BULK_SCRIPT, "Bulk registration script"),
        (BLOCKCHAIN_SETUP_ROOT, "nodes-setup directory"),
        (BLOCKCHAIN_VCTOOL, "vctool binary"),
        (BLOCKCHAIN_API_JWT_SCRIPT, "JWT helper"),
        (BLOCKCHAIN_ENV_EXAMPLE, ".env.example"),
        (BLOCKCHAIN_CRYPTO_CONFIG, "crypto-config.yaml"),
        (BLOCKCHAIN_CONFIGTX_DIR, "configtx directory"),
    ]
    for path, label in checks:
        if not path.exists():
            errors.append(f"{label} not found at {path}.")
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


# ---------------------------------------------------------------------------
# Auth and key helpers
# ---------------------------------------------------------------------------


def resolve_auth_secret(paths: Dict[str, Path]) -> str:
    secret = os.getenv("AUTH_JWT_SECRET")
    if secret:
        return secret
    env_vars = _load_env_file(paths["api_gateway"] / ".env")
    secret = env_vars.get("AUTH_JWT_SECRET")
    if secret:
        return secret
    raise SystemExit(
        "AUTH_JWT_SECRET is required but not set. Export it or add it to "
        f"{paths['api_gateway'] / '.env'} before running this script.",
    )


def read_admin_public_key(paths: Dict[str, Path]) -> str:
    candidate = paths["admin_public_key_file"]
    if not candidate.exists():
        raise SystemExit(f"Admin public key file not found at {candidate}.")
    content = candidate.read_text().strip()
    if not content:
        raise SystemExit(f"Admin public key file {candidate} is empty.")
    return content


def _derive_admin_public_key(private_key_path: Path) -> str:
    result = subprocess.run(
        ["openssl", "pkey", "-in", str(private_key_path), "-pubout", "-outform", "DER"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(f"Failed to extract admin public key: {result.stderr.decode().strip()}")
    raw = result.stdout[-32:]
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
    shutil.copy2(private_key_path, paths["keys_dir"] / private_key_path.name)
    shutil.copy2(public_key_path, paths["keys_dir"] / public_key_path.name)
    return pub_text


# ---------------------------------------------------------------------------
# Fabric artifact generation
# ---------------------------------------------------------------------------


def _ensure_binary(name: str) -> None:
    result = subprocess.run(
        [name, "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Required binary '{name}' is not available in PATH.")


def ensure_fabric_artifacts(paths: Dict[str, Path], force: bool = False) -> None:
    org_dir = paths["organizations_dir"]
    system_dir = paths["system_genesis_dir"]
    channel_dir = paths["channel_artifacts_dir"]
    genesis = system_dir / "genesis.block"
    channel_tx = channel_dir / "nebula-channel.tx"
    peer_org = org_dir / "peerOrganizations" / "org1.nebula.com"

    if not force and genesis.exists() and channel_tx.exists() and peer_org.exists() and any(peer_org.iterdir()):
        print("Fabric MSP artifacts already exist, skipping regeneration...")
        return

    _ensure_binary("cryptogen")
    _ensure_binary("configtxgen")

    print("Regenerating Fabric MSP artifacts...")
    for target in (org_dir, system_dir, channel_dir):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    _run_command(
        ["cryptogen", "generate", f"--config={paths['crypto_config']}", f"--output={org_dir}"],
        cwd=paths["api_gateway"],
        description="Generate MSP material",
    )
    env = os.environ.copy()
    env["FABRIC_CFG_PATH"] = str(paths["configtx_dir"])
    for profile, channel_id, output_flag, output_path, desc in [
        ("NebulaGenesis", "system-channel", "-outputBlock", str(system_dir / "genesis.block"), "system genesis block"),
        ("NebulaChannel", "nebulachannel", "-outputCreateChannelTx", str(channel_dir / "nebula-channel.tx"), "channel transaction"),
    ]:
        _run_command(
            ["configtxgen", "-profile", profile, "-channelID", channel_id, output_flag, output_path],
            cwd=paths["api_gateway"],
            env=env,
            description=f"Generate {desc}",
        )
    _run_command(
        [
            "configtxgen", "-profile", "NebulaChannel", "-channelID", "nebulachannel",
            "-asOrg", "Org1MSP", "-outputAnchorPeersUpdate",
            str(channel_dir / "Org1MSPanchors.tx"),
        ],
        cwd=paths["api_gateway"],
        env=env,
        description="Generate anchor peers update",
    )


# ---------------------------------------------------------------------------
# Fabric CA process management (replaces Docker-based CA)
# ---------------------------------------------------------------------------


def start_fabric_ca_process(paths: Dict[str, Path]) -> int:
    """Launch ``fabric-ca-server`` as a host process and return its PID."""
    _ensure_binary("fabric-ca-server")
    ca_cert = paths["ca_dir"] / "ca.org1.nebula.com-cert.pem"
    ca_key = paths["ca_dir"] / "priv_sk"
    if not ca_cert.exists():
        raise SystemExit(f"CA certificate missing at {ca_cert}")
    if not ca_key.exists():
        raise SystemExit(f"CA private key missing at {ca_key}")

    ca_home = paths.get("ca_admin_home", paths["ca_dir"]).parent / "ca-server-home"
    ca_home.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["FABRIC_CA_SERVER_CA_NAME"] = "ca-org1"
    env["FABRIC_CA_SERVER_CA_CERTFILE"] = str(ca_cert)
    env["FABRIC_CA_SERVER_CA_KEYFILE"] = str(ca_key)
    env["FABRIC_CA_SERVER_TLS_ENABLED"] = "false"
    env["FABRIC_CA_HOME"] = str(ca_home)

    proc = subprocess.Popen(
        ["fabric-ca-server", "start", "-b", "admin:adminpw", "--port", CA_PORT],
        cwd=str(ca_home),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for the CA server to become ready.
    time.sleep(3)
    if proc.poll() is not None:
        raise SystemExit(f"fabric-ca-server exited immediately (code {proc.returncode}).")
    print(f"Started Fabric CA process (pid {proc.pid}) on port {CA_PORT}")
    return proc.pid


def stop_fabric_ca_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ---------------------------------------------------------------------------
# Identity, VC signing, and registration
# ---------------------------------------------------------------------------


def _sync_admin_cacerts(paths: Dict[str, Path]) -> None:
    admin_msp = paths["admin_home"] / "msp"
    source = paths["ca_cert"]
    if not source.exists():
        raise SystemExit(f"CA certificate missing at {source}")
    dest_dir = admin_msp / "cacerts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest_dir / source.name)
    template = paths["msp_template"]
    if template.exists():
        shutil.copy2(template, admin_msp / "config.yaml")
    tlsca = BLOCKCHAIN_ORG_DIR / "tlsca" / "tlsca.org1.nebula.com-cert.pem"
    if tlsca.exists():
        tls_dir = admin_msp / "tlscacerts"
        tls_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tlsca, tls_dir / tlsca.name)


def _ensure_org_admin_identity(paths: Dict[str, Path]) -> None:
    msp_dir = paths["admin_home"] / "msp"
    signcerts = msp_dir / "signcerts"
    admincerts = msp_dir / "admincerts"
    if not signcerts.exists() or not any(signcerts.iterdir()):
        raise SystemExit(f"Admin MSP signcerts missing at {signcerts}.")
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
    _run_command(
        [
            "fabric-ca-client", "enroll",
            "-u", f"http://admin:adminpw@localhost:{CA_PORT}",
            "--caname", "ca-org1",
            "-M", str(msp_dir),
        ],
        cwd=paths["api_gateway"],
        env=env,
        description="Bootstrap CA admin",
    )


def generate_trainer_identities(paths: Dict[str, Path], auth_secret: str) -> None:
    _clear_directory(paths["unsigned_dir"])
    _clear_directory(paths["signed_dir"])
    _clear_directory(paths["tokens_dir"])
    if paths["keys_dir"].exists():
        for key_file in paths["keys_dir"].glob("trainer-node-*"):
            if key_file.is_file():
                key_file.unlink()
    # Sync node configs to the blockchain repo.  Prefer the runtime dir
    # (generated by config_generator) over the legacy config/nodes/ dir
    # which may contain stale entries from a previous run.
    nodes_src = ROOT_DIR / "process-runtime" / "config" / "nodes"
    if not nodes_src.exists():
        nodes_src = ROOT_DIR / "config" / "nodes"
    if nodes_src.exists():
        _copy_directory(nodes_src, paths["nodes_dir"])
    _run_command(
        ["node", str(paths["generate_script"]), "--generate-jwt", "registration,runtime", "--auth-secret", auth_secret],
        cwd=paths["api_gateway"],
        description="Trainer identity generation",
    )
    if not paths["keys_dir"].exists():
        raise SystemExit(f"Trainer keys not found at {paths['keys_dir']} after generation.")
    _copy_directory(paths["keys_dir"], KEYS_DIR)


def _enroll_trainer_msps(paths: Dict[str, Path]) -> None:
    env = os.environ.copy()
    env["FABRIC_CA_CLIENT_MSPDIR"] = str(paths["ca_admin_home"] / "msp")
    _run_command(
        [
            "node", str(paths["enroll_script"]),
            "--ca-url", f"http://localhost:{CA_PORT}",
            "--ca-name", "ca-org1",
            "--tls-cert", str(paths["ca_cert"]),
            "--admin-home", str(paths["ca_admin_home"]),
        ],
        cwd=paths["api_gateway"],
        env=env,
        description="Trainer enrollment",
    )


def _sign_trainer_vcs(paths: Dict[str, Path], admin_key: Path) -> None:
    _run_command(
        ["node", str(paths["sign_script"]), "--key", str(admin_key)],
        cwd=paths["api_gateway"],
        description="Sign trainer VCs",
    )


def _build_bulk_registration_payload(paths: Dict[str, Path]) -> None:
    _run_command(
        [
            "node", str(paths["bulk_script"]),
            "--did-template", "did:nebula:trainer-node-{trainerSeq}",
            "--output", str(paths["bulk_output"]),
            "--force",
        ],
        cwd=paths["api_gateway"],
        description="Bulk registration payload",
    )


def _generate_admin_jwt(paths: Dict[str, Path], auth_secret: str) -> None:
    env = os.environ.copy()
    env.update(AUTH_JWT_SECRET=auth_secret, JWT_ALG="HS256", JWT_ROLE="admin", JWT_SUB="admin")
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


def _locate_admin_private_key() -> Path:
    for candidate in [
        BLOCKCHAIN_SETUP_KEYS_DIR / "admin_ed25519_sk.pem",
        BLOCKCHAIN_API_GATEWAY_DIR / "admin_ed25519_sk.pem",
        BLOCKCHAIN_REPO_DIR / "admin_ed25519_sk.pem",
    ]:
        if candidate.exists():
            return candidate
    raise SystemExit("Admin Ed25519 private key (admin_ed25519_sk.pem) not found.")


# ---------------------------------------------------------------------------
# Full artifact preparation pipeline (replaces Docker-based _prepare_blockchain_artifacts)
# ---------------------------------------------------------------------------


def prepare_blockchain_artifacts(paths: Dict[str, Path], auth_secret: str) -> None:
    """Prepare all Fabric artifacts using host-process CA instead of Docker."""
    admin_public_key = _ensure_admin_keypair(paths)
    _ensure_env_file(paths, auth_secret, admin_public_key)
    ensure_fabric_artifacts(paths)
    _sync_admin_cacerts(paths)
    generate_trainer_identities(paths, auth_secret)
    _ensure_binary("fabric-ca-client")
    _ensure_org_admin_identity(paths)
    ca_pid = start_fabric_ca_process(paths)
    try:
        _ensure_ca_admin_identity(paths)
        _enroll_trainer_msps(paths)
    finally:
        print("Stopping Fabric CA process...")
        stop_fabric_ca_process(ca_pid)
    admin_key = _locate_admin_private_key()
    _sign_trainer_vcs(paths, admin_key)
    _build_bulk_registration_payload(paths)
    _generate_admin_jwt(paths, auth_secret)


def _ensure_env_file(paths: Dict[str, Path], auth_secret: str, admin_public_key: str) -> None:
    env_path = paths["env_file"]
    if not env_path.exists():
        shutil.copy(paths["env_example"], env_path)
    _set_env_value(env_path, "AUTH_JWT_SECRET", auth_secret)
    _set_env_value(env_path, "ADMIN_PUBLIC_KEY", admin_public_key)


# ---------------------------------------------------------------------------
# Blockchain process lifecycle (orderer + peers + gateway via manage.sh)
# ---------------------------------------------------------------------------


def start_blockchain(paths: Dict[str, Path], auth_secret: str, gateway_url: str) -> None:
    admin_public_key = read_admin_public_key(paths)
    env = os.environ.copy()
    env["AUTH_JWT_SECRET"] = auth_secret
    env["ADMIN_PUBLIC_KEY"] = admin_public_key
    env.setdefault("BLOCKCHAIN_GATEWAY_URL", gateway_url)
    if not BLOCKCHAIN_PROCESS_RUNNER.exists():
        raise SystemExit(f"Blockchain process runner not found at {BLOCKCHAIN_PROCESS_RUNNER}.")
    print(f"Starting blockchain process stack (orderer + peers + API gateway)...")
    result = subprocess.run(
        ["./manage.sh", "start"],
        cwd=BLOCKCHAIN_PROCESS_RUNNER.parent,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Blockchain process runner 'start' failed (exit code {result.returncode}).")


def stop_blockchain() -> None:
    if not BLOCKCHAIN_PROCESS_RUNNER.exists():
        return
    subprocess.run(
        ["./manage.sh", "stop"],
        cwd=BLOCKCHAIN_PROCESS_RUNNER.parent,
        env=os.environ.copy(),
        check=False,
    )


def clear_blockchain_runtime(paths: Dict[str, Path]) -> None:
    if BLOCKCHAIN_PROCESS_RUNTIME_DIR.exists():
        shutil.rmtree(BLOCKCHAIN_PROCESS_RUNTIME_DIR)
    BLOCKCHAIN_PROCESS_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    trainer_db = paths.get("trainer_db")
    if trainer_db:
        trainer_db.parent.mkdir(parents=True, exist_ok=True)
        trainer_db.write_text("[\n]\n")


# ---------------------------------------------------------------------------
# Gateway health and trainer registration
# ---------------------------------------------------------------------------


def wait_for_gateway_health(base_url: str, timeout: int = 240, interval: int = 5) -> None:
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
    for key in ("jwt_sub", "jwtSub", "nodeId", "node_id", "did", "trainerId", "trainer_id", "subject"):
        value = entry.get(key)
        if value:
            return str(value)
    return "unknown"


def bulk_register_trainers(paths: Dict[str, Path], base_url: str) -> None:
    if not paths["admin_jwt_path"].exists():
        raise SystemExit(f"Admin JWT not found at {paths['admin_jwt_path']}.")
    if not paths["bulk_output"].exists():
        raise SystemExit(f"Bulk registration payload not found at {paths['bulk_output']}.")
    token = paths["admin_jwt_path"].read_text().strip()
    if not token:
        raise SystemExit(f"Admin JWT file {paths['admin_jwt_path']} is empty.")
    entries = json.loads(paths["bulk_output"].read_text())
    url = f"{base_url.rstrip('/')}{GATEWAY_BULK_PATH}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    batch_size = 5
    total_batches = (len(entries) + batch_size - 1) // batch_size

    for offset in range(0, len(entries), batch_size):
        batch = entries[offset:offset + batch_size]
        batch_idx = offset // batch_size + 1
        ids = [_trainer_identifier(e) for e in batch]
        print(f"Registering trainers {', '.join(ids)} (batch {batch_idx}/{total_batches})")
        pending = list(batch)
        for attempt in range(1, 4):
            payload = json.dumps(pending).encode()
            req = urllib_request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib_request.urlopen(req, timeout=60) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                errors = _parse_bulk_errors(body)
                if not errors:
                    pending = []
                    break
                entry_map = {_trainer_identifier(e): e for e in pending}
                pending = [entry_map[err["id"]] for err in errors if err["id"] in entry_map]
            except urllib_error.HTTPError as exc:
                if exc.code == 409:
                    pending = []
                    break
                if attempt >= 3:
                    body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
                    raise SystemExit(f"Bulk registration batch {batch_idx} failed (HTTP {exc.code}): {body.strip()}")
            except (urllib_error.URLError, http_client.RemoteDisconnected) as err:
                if attempt >= 3:
                    raise SystemExit(f"Batch {batch_idx} failed: {err}") from err
            if pending and attempt < 3:
                time.sleep(5)
        if pending:
            failed = ", ".join(_trainer_identifier(e) for e in pending)
            raise SystemExit(f"Bulk registration failed for trainers {failed}")


def _parse_bulk_errors(body: str) -> List[Dict[str, Any]]:
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
        status = str(result.get("status") or result.get("Status") or "").lower()
        if status == "ok":
            continue
        errors.append({
            "id": _trainer_identifier(result),
            "error": result.get("error") or result.get("Error") or "unknown",
        })
    return errors
