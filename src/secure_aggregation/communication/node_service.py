"""Node service for federated learning with secure aggregation."""

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import grpc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.communication.aggregator_service import AggregatorServicer, serve as serve_aggregator
from secure_aggregation.communication.bridge_service import BridgeClient, serve_bridge
from secure_aggregation.communication.inter_cluster_aggregator import InterClusterAggregator
from secure_aggregation.convergence import ConvergenceConfig, ConvergenceTracker
from secure_aggregation.convergence.central_broadcast import (
    CENTRAL_METADATA_CLUSTER_ID,
    fetch_central_metadata,
)
from secure_aggregation.convergence.central_checker import CentralChecker
from secure_aggregation.config.models import NodeRole
from secure_aggregation.config.system import load_system_config
from secure_aggregation.crypto.sign import SigningKeyPair
from secure_aggregation.data import dirichlet_partition
from secure_aggregation.node import ECMBuffer, NodeEngine, NodeRuntimeConfig, ReliabilityScore
from secure_aggregation.protocol import MergeConfig, SecureAggregationNode
from secure_aggregation.protocol.core import AdvertiseMessage, Round1Ciphertext, SHARE_BYTES, _int_to_bytes
from secure_aggregation.storage.model_store import (
    BlockchainInterface,
    GatewayBlockchain,
    IPFSInterface,
    KuboIPFS,
    MockBlockchain,
    MockIPFS,
    verify_model_hash,
)
from secure_aggregation.topology import elect_clique_aggregator, get_inter_clique_neighbors, is_bridge_node
from secure_aggregation.utils import configure_logging, get_logger

logger = get_logger("node_service")


class MnistLinear(nn.Module):
    """Simple linear classifier for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)


def flatten_params(model: nn.Module) -> List[float]:
    """Flatten model parameters to list of floats."""
    vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    return vec.tolist()


def load_params(model: nn.Module, flat: List[float]) -> None:
    """Load flattened parameters into model."""
    tensor = torch.tensor(flat, dtype=torch.float32)
    torch.nn.utils.vector_to_parameters(tensor, model.parameters())


def quantize_vector(vec: List[float], scale: float) -> List[int]:
    """Quantize float vector to integers."""
    return [int(round(v * scale)) for v in vec]


def dequantize_vector(ints: List[int], scale: float) -> List[float]:
    """Dequantize integer vector to floats."""
    return [float(i) / scale for i in ints]


def _encode_share(x: int, share: int) -> bytes:
    """Pack (x, share) tuple for transport."""
    return _int_to_bytes(x, 2) + _int_to_bytes(share, SHARE_BYTES)


class GlobalStopRequested(Exception):
    """Raised when global convergence has been confirmed and execution should halt."""


class AggregatorUnavailable(Exception):
    """Raised when the elected aggregator cannot be reached after repeated attempts."""


class NodeService:
    """Node service that coordinates training and secure aggregation."""

    def __init__(self, config_path: str) -> None:
        self.config = self._load_config(config_path)
        self.system_config, self.system_config_path = load_system_config(Path(config_path))
        self.node_id = self.config["node_id"]
        self.role = NodeRole(self.config["role"])
        self.ttp_address = self.config["ttp_address"]
        self.port = self.config["port"]
        self.dataset_config = self.config["dataset"]
        self.training_config = self.config["training"]
        self.secagg_config = self.config["secure_agg"]
        self.threshold = self.secagg_config["threshold"]
        self.scale = self.secagg_config["scale"]
        env_rounds = os.getenv("MAX_TRAINING_ROUNDS")
        default_round_cap = 50
        if env_rounds:
            try:
                self.max_training_rounds = max(1, int(env_rounds))
            except ValueError:
                logger.warning(
                    "Invalid MAX_TRAINING_ROUNDS=%s; falling back to default %d",
                    env_rounds,
                    default_round_cap,
                )
                self.max_training_rounds = default_round_cap
        else:
            self.max_training_rounds = default_round_cap

        # State
        self.signing_keypair: Optional[SigningKeyPair] = None
        self.participants: List[secureagg_pb2.NodeInfo] = []
        self.participant_map: Dict[str, str] = {}  # node_id -> address
        self.model: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.current_round = 0
        self.is_aggregator = False
        self.aggregator_id: Optional[str] = None
        self.aggregator_address: Optional[str] = None
        self.aggregator_server: Optional[grpc.Server] = None

        # Clique membership from TTP
        self.clique_id: int = -1
        self.clique_members: List[str] = []
        self.clique_threshold: int = 0
        self.assigned_data_indices: List[int] = []

        # Inter-cluster aggregation state
        self.inter_cluster_config = self.config.get("inter_cluster", {})
        self.inter_cluster_enabled = self.inter_cluster_config.get("enabled", False)
        self.inter_edges: List[Tuple[str, str]] = []
        self.is_bridge_node = False
        self.neighbor_bridge_addresses: List[str] = []
        self.neighbor_address_map: Dict[str, str] = {}
        self.central_neighbor_addresses: Dict[str, str] = {}
        self.ecm_buffer: Optional[ECMBuffer] = None
        self.bridge_server: Optional[grpc.Server] = None
        self.bridge_client: Optional[BridgeClient] = None
        self.inter_cluster_aggregator: Optional[InterClusterAggregator] = None
        self.ipfs: Optional[IPFSInterface] = None
        self.blockchain: Optional[BlockchainInterface] = None
        self.ecm_forward_wait = float(self.inter_cluster_config.get("ecm_forward_wait_seconds", 5.0))

        # Convergence state
        self.convergence_config = self._load_convergence_config()
        self.convergence_tracker: Optional[ConvergenceTracker] = None
        self._latest_cluster_converged: bool = False
        self._latest_delta_norm: float = 0.0
        self.central_metadata = None
        self.central_checker: Optional[CentralChecker] = None
        self.aggregator_servicer: Optional[AggregatorServicer] = None
        self._bootstrap_anchors: List[Tuple[str, int, str, Optional[str], Optional[str]]] = []
        self._logged_central_addresses = False
        self._clique_signal_addresses: Dict[str, str] = {}
        self._known_convergence_data_ids: Set[str] = set()
        self._pending_convergence_data_ids: Set[str] = set()
        self._acknowledged_convergence_data_ids: Set[str] = set()
        self._convergence_payload_cache: Dict[str, Dict[str, Any]] = {}
        self._confirmed_global_convergence_round: Optional[int] = None
        self._confirmed_global_convergence_reason: str = ""

        logger.info(f"Node {self.node_id} initialized (role={self.role}, port={self.port})")

    def _load_config(self, path: str) -> dict:
        """Load node configuration from JSON file."""
        with open(path) as f:
            return json.load(f)

    def _load_convergence_config(self) -> ConvergenceConfig:
        """Load convergence configuration preferring the shared system config."""
        system_convergence = (self.system_config or {}).get("convergence")
        if system_convergence is not None:
            return ConvergenceConfig.from_dict(system_convergence)
        node_convergence = self.config.get("convergence")
        if node_convergence is not None:
            logger.warning(
                "Using per-node convergence config for %s; please move it to %s",
                self.node_id,
                self.system_config_path,
            )
            return ConvergenceConfig.from_dict(node_convergence)
        return ConvergenceConfig()

    def _prime_convergence_tracker_state(self) -> None:
        """Advance convergence tracker state without computing deltas."""
        if not self.convergence_tracker or not self.convergence_config.enabled:
            return
        if self.model is None:
            return
        flat_params = np.array(flatten_params(self.model), dtype=np.float32)
        self.convergence_tracker.update(flat_params, track_diff=False)

    def _maybe_check_convergence_during_retry(
        self,
        attempt_idx: int,
        max_attempts: int,
        stage: str,
        first_check: int = 40,
        interval: int = 5,
    ) -> None:
        """
        Periodically poll for convergence while retrying aggregator RPCs.

        Args:
            attempt_idx: Current attempt number (1-based).
            max_attempts: Total attempts allowed.
            stage: Human-readable stage description.
            first_check: Attempt number to trigger the first convergence check.
            interval: Attempt interval for subsequent checks.
        """
        if attempt_idx < first_check:
            return
        if attempt_idx == first_check or ((attempt_idx - first_check) % interval == 0):
            logger.info(
                "Aggregator %s unavailable at stage %s (attempt %d/%d); re-checking convergence state",
                self.aggregator_id,
                stage,
                attempt_idx,
                max_attempts,
            )
            should_stop, stop_reason = self._refresh_convergence_state(False, "")
            if should_stop:
                logger.info(
                    "Halting retries because convergence was confirmed while waiting "
                    "(reason=%s)",
                    stop_reason,
                )
                raise GlobalStopRequested()

    def _raise_aggregator_unavailable(
        self,
        stage: str,
        attempts: int,
        last_error: Optional[Exception],
    ) -> None:
        """
        Raise AggregatorUnavailable with a detailed message.
        """
        reason = ""
        if isinstance(last_error, grpc.RpcError):
            code = last_error.code().name if hasattr(last_error, "code") else "UNKNOWN"
            details = last_error.details() if hasattr(last_error, "details") else ""
            reason = f" (last_error={code} {details})"
        elif last_error:
            reason = f" (last_error={last_error})"
        message = (
            f"Aggregator {self.aggregator_id} unreachable during {stage} after "
            f"{attempts} attempts{reason}"
        )
        raise AggregatorUnavailable(message)

    def _hydrate_anchor_bootstrap(self) -> None:
        """Persist bootstrap anchor references once blockchain client is ready."""
        if not self.blockchain or not self._bootstrap_anchors:
            return
        for cluster_id, round_num, data_id, cid, hash_val in self._bootstrap_anchors:
            self.blockchain.remember_anchor(cluster_id, round_num, data_id, cid, hash_val)
        self._bootstrap_anchors.clear()

    def register_with_ttp(self) -> None:
        """Register with TTP and receive signing keys and clique assignment."""
        logger.info(f"Registering with TTP at {self.ttp_address}")

        max_retries = 10
        for attempt in range(max_retries):
            try:
                channel = grpc.insecure_channel(self.ttp_address)
                stub = secureagg_pb2_grpc.TTPServiceStub(channel)

                request = secureagg_pb2.RegisterRequest(
                    node_id=self.node_id,
                    address=f"{self.node_id}:{self.port}"
                )
                response = stub.RegisterNode(request, timeout=5)

                if response.success:
                    self.signing_keypair = SigningKeyPair(
                        private_key=bytes(response.signing_private_key),
                        public_key=bytes(response.signing_public_key)
                    )

                    # Extract clique assignment from TTP response
                    self.clique_id = response.clique_id
                    self.clique_members = list(response.clique_members)
                    self.clique_threshold = response.clique_threshold
                    self.assigned_data_indices = list(response.data_indices)

                    # Use clique threshold if provided, otherwise fall back to config
                    if self.clique_threshold > 0:
                        self.threshold = self.clique_threshold

                logger.info(
                    f"Registered with TTP: clique={self.clique_id}, "
                    f"members={len(self.clique_members)}, threshold={self.threshold}, "
                    f"data_samples={len(self.assigned_data_indices)}"
                )
                metadata_data_id = getattr(response, "central_metadata_data_id", "")
                if metadata_data_id:
                    metadata_version = getattr(response, "central_metadata_version", 0)
                    self._bootstrap_anchors.append(
                        (
                            CENTRAL_METADATA_CLUSTER_ID,
                            metadata_version or 0,
                            metadata_data_id,
                            None,
                            None,
                        )
                    )
                else:
                    logger.info("No central metadata anchor provided by TTP; continuing without bootstrap.")

                # Get list of participants
                participants_response = stub.GetParticipants(secureagg_pb2.ParticipantsRequest())
                self.participants = list(participants_response.participants)
                self.participant_map = {p.node_id: p.address for p in self.participants}
                logger.info(f"Retrieved {len(self.participants)} total participants")

                channel.close()
                return

            except grpc.RpcError as e:
                logger.warning(f"TTP connection attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(2)

        raise RuntimeError("Failed to connect to TTP after max retries")

    def setup_data(self) -> None:
        """Setup MNIST dataset using indices assigned by TTP or local partition."""
        logger.info("Setting up MNIST dataset")
        tform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(root="/app/data", train=True, download=False, transform=tform)
        test_ds = datasets.MNIST(root="/app/data", train=False, download=False, transform=tform)

        # Use TTP-assigned indices if available, otherwise compute locally
        if self.assigned_data_indices:
            indices = self.assigned_data_indices
            logger.info(f"Using {len(indices)} TTP-assigned data samples")
        else:
            # Fallback: compute partition locally
            labels = {i: int(train_ds[i][1]) for i in range(len(train_ds))}
            num_clients = self.dataset_config["num_clients"]
            alpha = self.dataset_config["alpha"]
            seed = self.dataset_config.get("seed", 42)

            parts = dirichlet_partition(
                list(range(len(train_ds))), labels, num_clients=num_clients, alpha=alpha, seed=seed
            )

            node_index = int(self.node_id.split("_")[-1])
            client_key = f"client_{node_index}"
            indices = parts.get(client_key, [])
            logger.info(f"Using {len(indices)} locally-computed data samples")

        batch_size = self.training_config["batch_size"]
        self.train_loader = DataLoader(Subset(train_ds, indices), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    def setup_model(self) -> None:
        """Initialize model."""
        self.model = MnistLinear()
        logger.info("Model initialized")

    def train_local(self, epochs: int) -> None:
        """Train model locally for specified epochs."""
        if not self.model or not self.train_loader:
            raise RuntimeError("Model or data not initialized")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model.train()

        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                logits = model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                opt.step()

        self.model = model.cpu()
        logger.info(f"Local training completed for {epochs} epochs")

    def evaluate(self) -> float:
        """Evaluate model on test set."""
        if not self.model or not self.test_loader:
            return 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model).to(device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = correct / total if total else 0.0
        return accuracy

    def elect_aggregator(self, round_idx: int) -> str:
        """Elect aggregator within clique using round-robin."""
        # Use clique members if available, otherwise fall back to all participants
        if self.clique_members:
            aggregator_id = elect_clique_aggregator(self.clique_members, round_idx)
            logger.info(f"Elected clique aggregator for round {round_idx}: {aggregator_id} (clique {self.clique_id})")
        else:
            sorted_participants = sorted(self.participant_map.keys())
            aggregator_id = sorted_participants[round_idx % len(sorted_participants)]
            logger.info(f"Elected global aggregator for round {round_idx}: {aggregator_id}")
        return aggregator_id

    def start_aggregator_server(self) -> None:
        """Start aggregator gRPC server if this node is elected."""
        if self.aggregator_server is not None:
            logger.warning("Aggregator server already running")
            return

        # Use clique members if available, otherwise all participants
        participant_ids = self.clique_members if self.clique_members else list(self.participant_map.keys())
        signing_public_keys: Dict[str, bytes] = {}
        if self.participants:
            signing_public_keys = {p.node_id: bytes(p.signing_public_key) for p in self.participants}
        self.aggregator_server, self.aggregator_servicer = serve_aggregator(
            self.node_id,
            self.port + 1000,
            self.threshold,
            participant_ids,
            signing_public_keys=signing_public_keys or None,
            ecm_buffer=self.ecm_buffer if self.inter_cluster_enabled else None,
            convergence_signal_handler=self._handle_convergence_signal_from_peer,
        )
        logger.info(f"Started aggregator server on port {self.port + 1000} for {len(participant_ids)} clique members")

    def stop_aggregator_server(self) -> None:
        """Stop aggregator server."""
        if self.aggregator_server:
            self.aggregator_server.stop(0)
            self.aggregator_server = None
            self.aggregator_servicer = None
            logger.info("Stopped aggregator server")

    def setup_inter_cluster(self) -> None:
        """Setup inter-cluster aggregation components."""
        if not self.inter_cluster_enabled:
            logger.info(
                "Inter-cluster aggregation disabled (inter_cluster.enabled=false); "
                "bridge forwarding and ECM gossip phases will be skipped"
            )
            return

        ipfs_config = self.inter_cluster_config.get("ipfs", {})
        blockchain_config = self.inter_cluster_config.get("blockchain", {})

        use_mock = self.inter_cluster_config.get("use_mock", True)

        if use_mock:
            ipfs_path = ipfs_config.get("storage_path", "/app/data/ipfs")
            blockchain_path = blockchain_config.get("storage_path", "/app/data/blockchain.json")
            self.ipfs = MockIPFS(storage_path=ipfs_path)
            self.blockchain = MockBlockchain(storage_path=blockchain_path)
            logger.info(f"Using mock storage: ipfs={ipfs_path}, blockchain={blockchain_path}")
        else:
            ipfs_url = ipfs_config.get("api_url", "http://ipfs-node-1:5001")
            gateway_url = blockchain_config.get(
                "gateway_url",
                os.environ.get("BLOCKCHAIN_GATEWAY_URL", "http://localhost:9000"),
            )
            identity = blockchain_config.get("identity", self.node_id)
            private_key_path = blockchain_config.get(
                "private_key_path",
                f"config/keys/{identity}_sk.pem",
            )
            state_path = blockchain_config.get(
                "state_path",
                f"data/blockchain/{identity}.json",
            )
            jwt_role = blockchain_config.get("jwt_role", "trainer")
            jwt_state = blockchain_config.get("jwt_state", "system")
            jwt_ttl = blockchain_config.get("jwt_ttl_seconds", 24 * 3600)
            self.ipfs = KuboIPFS(api_url=ipfs_url)
            self.blockchain = GatewayBlockchain(
                base_url=gateway_url,
                identity=identity,
                private_key_path=private_key_path,
                state_path=state_path,
                jwt_role=jwt_role,
                jwt_state=jwt_state,
                jwt_ttl_seconds=jwt_ttl,
            )
            logger.info(f"Using real storage: ipfs={ipfs_url}, gateway={gateway_url}")

        if self.blockchain:
            self._hydrate_anchor_bootstrap()

        merge_config = MergeConfig(
            window_size=self.inter_cluster_config.get("window_size", 10),
            alpha=self.inter_cluster_config.get("alpha", 0.5),
            base_gamma=self.inter_cluster_config.get("base_gamma", 0.2),
            max_neighbors=self.inter_cluster_config.get("max_neighbors"),
        )

        self.ecm_buffer = ECMBuffer(
            freshness_window=self.inter_cluster_config.get("freshness_window", 300.0)
        )

        self.inter_cluster_aggregator = InterClusterAggregator(
            cluster_id=f"cluster_{self.clique_id}",
            ipfs=self.ipfs,
            blockchain=self.blockchain,
            merge_config=merge_config,
        )
        self._refresh_central_metadata()
        self._ensure_bridge_stack()

    def _ensure_bridge_stack(self) -> None:
        """Ensure the bridge server/client are available for convergence gossip."""
        if not self.inter_cluster_enabled:
            return
        if self.ecm_buffer is None:
            freshness = float(self.inter_cluster_config.get("freshness_window", 300.0))
            self.ecm_buffer = ECMBuffer(freshness_window=freshness)
        if self.bridge_server is None:
            try:
                self.bridge_server = serve_bridge(
                    self.node_id,
                    self.port + 2000,
                    self.ecm_buffer,
                )
                logger.info("Bridge server started on port %d", self.port + 2000)
            except Exception as exc:  # noqa: BLE001
                self.bridge_server = None
                logger.error("Failed to start bridge server on port %d: %s", self.port + 2000, exc, exc_info=True)
        if self.bridge_client is None:
            self.bridge_client = BridgeClient(self.node_id)

    def setup_bridge_node(self, inter_edges: List[Tuple[str, str]]) -> bool:
        """Setup bridge node if this node has inter-clique connections."""
        self.inter_edges = inter_edges
        self.is_bridge_node = is_bridge_node(self.node_id, inter_edges)

        if not self.is_bridge_node:
            logger.info(f"Node {self.node_id} is not a bridge node")
            return False

        neighbors = get_inter_clique_neighbors(self.node_id, inter_edges)
        self.neighbor_address_map = {}
        for neighbor in neighbors:
            base_address = self.participant_map.get(neighbor)
            attempts = 0
            while base_address is None and attempts < 5:
                logger.info(
                    f"Neighbor {neighbor} not registered yet; refreshing participant map"
                )
                time.sleep(2)
                self.register_with_ttp()
                base_address = self.participant_map.get(neighbor)
                attempts += 1
            if not base_address:
                logger.warning(
                    f"Could not resolve bridge address for neighbor {neighbor}; ECM gossip disabled for this edge"
                )
                continue
            host, port_str = base_address.split(":")
            neighbor_bridge_port = int(port_str) + 2000
            address = f"{host}:{neighbor_bridge_port}"
            self.neighbor_address_map[neighbor] = address
        self.neighbor_bridge_addresses = list(self.neighbor_address_map.values())

        logger.info(
            f"Node {self.node_id} is a bridge node with {len(self.neighbor_address_map)} "
            f"resolved inter-clique neighbors: {self.neighbor_address_map}"
        )

        if not self.inter_cluster_enabled:
            logger.warning(
                "Inter-cluster disabled while configuring bridge node %s; skipping bridge server",
                self.node_id,
            )
            return False
        self._ensure_bridge_stack()
        self._update_central_neighbor_addresses()
        return self.bridge_client is not None

    def stop_bridge_server(self) -> None:
        """Stop bridge server."""
        stopped = False
        if self.bridge_server:
            stop_future = self.bridge_server.stop(0)
            if stop_future:
                stop_future.wait()
            self.bridge_server = None
            stopped = True
        if self.bridge_client:
            self.bridge_client.close()
            self.bridge_client = None
            stopped = True
        if stopped:
            logger.info("Bridge server stopped")

    def _init_bridge_with_retries(
        self,
        inter_edges: List[Tuple[str, str]],
        max_attempts: int = 5,
        delay: float = 2.0,
        fatal: bool = False,
    ) -> bool:
        """Attempt to initialize bridge stack with retries."""
        if not inter_edges or not is_bridge_node(self.node_id, inter_edges):
            return False
        for attempt in range(max_attempts):
            try:
                if self.setup_bridge_node(inter_edges):
                    return True
            except OSError as exc:  # port bind/IO issues
                logger.error(
                    "Bridge initialization attempt %d/%d failed for %s due to OS error: %s",
                    attempt + 1,
                    max_attempts,
                    self.node_id,
                    exc,
                )
                # Port may still be bound; wait before retry.
                time.sleep(delay)
                continue
            logger.warning(
                "Bridge initialization attempt %d/%d failed for %s; retrying in %.1fs",
                attempt + 1,
                max_attempts,
                self.node_id,
                delay,
            )
            time.sleep(delay)
        if fatal and self.is_bridge_node:
            raise RuntimeError(
                f"Bridge node {self.node_id} could not initialize bridge client "
                f"after {max_attempts} attempts"
            )
        logger.warning(
            "Bridge initialization failed for %s after %d attempts; continuing without bridge server",
            self.node_id,
            max_attempts,
        )
        return False

    def _refresh_central_metadata(self) -> None:
        """Fetch central metadata from blockchain and update coordinator."""
        if not self.blockchain:
            return
        metadata = fetch_central_metadata(self.blockchain)
        if metadata:
            self.central_metadata = metadata
            logger.info(
                f"Fetched central metadata: central clique={metadata.central_clique_idx}, "
                f"central nodes={metadata.central_nodes}, checker candidates={metadata.checker_candidates}"
            )
            self._update_central_neighbor_addresses()
            if (
                self.convergence_tracker
                and metadata.checker_candidates
                and not self.convergence_config.central_checker_id
            ):
                self.convergence_config.central_checker_id = metadata.checker_candidates[0]
                self.convergence_tracker.config.central_checker_id = metadata.checker_candidates[0]
            self._maybe_init_central_checker()

    def _update_central_neighbor_addresses(self) -> None:
        """Build mapping to central neighbor bridge addresses when metadata is available."""
        self.central_neighbor_addresses = {}
        if not self.central_metadata or not self.participant_map:
            return
        for node_id in self.central_metadata.central_nodes:
            base_address = self.participant_map.get(node_id)
            if not base_address:
                continue
            try:
                host, port_str = base_address.split(":")
                bridge_port = int(port_str) + 2000
            except ValueError:
                logger.warning(
                    "Invalid address format for central node %s: %s",
                    node_id,
                    base_address,
                )
                continue
            self.central_neighbor_addresses[node_id] = f"{host}:{bridge_port}"

        if self.central_neighbor_addresses and not self._logged_central_addresses:
            details = ", ".join(f"{node}@{addr}" for node, addr in self.central_neighbor_addresses.items())
            logger.info(f"Central neighbor addresses: {details}")
            self._logged_central_addresses = True

    def _update_clique_signal_addresses(self) -> None:
        """Precompute bridge endpoints for clique peers to receive convergence broadcasts."""
        self._clique_signal_addresses = {}
        candidates = self.clique_members if self.clique_members else list(self.participant_map.keys())
        for member in candidates:
            base_address = self.participant_map.get(member)
            if not base_address:
                continue
            try:
                host, port_str = base_address.split(":")
                bridge_port = int(port_str) + 2000
            except ValueError:
                logger.warning("Invalid address format for clique member %s: %s", member, base_address)
                continue
            self._clique_signal_addresses[member] = f"{host}:{bridge_port}"

    def _maybe_init_central_checker(self) -> None:
        """Instantiate central checker if this node is a candidate."""
        if not self.central_metadata:
            self.central_checker = None
            return
        if self.node_id in self.central_metadata.checker_candidates:
            if self.central_checker is None:
                self.central_checker = CentralChecker(
                    self.blockchain,
                    total_cliques=self.central_metadata.total_cliques,
                    cluster_ids=self.central_metadata.cluster_ids,
                    on_announcement=self._handle_global_convergence_announcement,
                )
        else:
            self.central_checker = None

    def gossip_ecm(self, cid: str, model_hash: str, round_num: int) -> None:
        """Gossip ECM to neighbor cluster bridge nodes."""
        if not self.is_bridge_node:
            return
        if not self._ensure_bridge_client():
            logger.warning("Cannot gossip ECM: bridge client unavailable")
            return

        cluster_id = f"cluster_{self.clique_id}"
        accepted = self.bridge_client.broadcast_ecm(
            self.neighbor_bridge_addresses,
            cluster_id,
            round_num,
            cid,
            model_hash,
        )
        logger.info(
            f"Gossiped ECM to {accepted}/{len(self.neighbor_bridge_addresses)} neighbors"
        )

    def forward_ecms_to_aggregator(self) -> int:
        """Forward buffered ECMs from this bridge node to the current aggregator.

        Bridge nodes receive ECMs from neighbor clusters. If the bridge node is not
        the aggregator, it must forward these ECMs to the aggregator so that the
        aggregator can merge neighbor models with the intra-cluster model.

        Returns:
            Number of ECMs forwarded, or 0 if not applicable.
        """
        if not self.is_bridge_node:
            return 0
        if not self.ecm_buffer:
            return 0
        if not self._ensure_bridge_client():
            logger.warning("Bridge client unavailable; skipping ECM forward")
            return 0

        fresh_ecms = self.ecm_buffer.get_fresh_ecms()
        fresh_ecms = [ecm for ecm in fresh_ecms if not ecm.is_signal]
        if not fresh_ecms:
            logger.info(
                "No fresh ECMs to forward to aggregator (buffer_size=%d)",
                len(self.ecm_buffer),
            )
            return 0

        if self.is_aggregator:
            # Aggregator already holds these ECMs locally via its bridge server.
            logger.debug(
                "Aggregator is also a bridge node; %d ECMs already staged locally",
                len(fresh_ecms),
            )
            return len(fresh_ecms)

        if not self.aggregator_address:
            logger.warning("Cannot forward ECMs: aggregator address not set")
            return 0

        agg_port = int(self.aggregator_address.split(":")[-1]) + 1000
        agg_host = self.aggregator_address.split(":")[0]
        agg_addr = f"{agg_host}:{agg_port}"

        try:
            channel = grpc.insecure_channel(agg_addr)
            stub = secureagg_pb2_grpc.AggregatorServiceStub(channel)

            ecm_messages = [
                secureagg_pb2.ECMMessage(
                    cid=ecm.cid,
                    hash=ecm.hash,
                    source_cluster=ecm.source_cluster or "",
                )
                for ecm in fresh_ecms
            ]

            request = secureagg_pb2.ECMSubmitRequest(
                node_id=self.node_id,
                ecms=ecm_messages,
            )

            response = stub.SubmitECMs(request, timeout=10)
            if response.accepted:
                logger.info(
                    f"Forwarded {len(fresh_ecms)} ECMs to aggregator {self.aggregator_id}"
                )
                if self.ecm_buffer:
                    removed = self.ecm_buffer.remove_cids([ecm.cid for ecm in fresh_ecms])
                    logger.debug("Removed %d ECMs from buffer after forwarding", removed)
                return len(fresh_ecms)
            else:
                logger.warning(f"Aggregator rejected ECMs: {response.message}")
                return 0
        except grpc.RpcError as e:
            logger.warning(f"Failed to forward ECMs to aggregator: {e}")
            return 0

    def _wait_for_neighbor_ecms(self) -> None:
        """Poll for incoming ECMs before forwarding to the aggregator."""
        if (
            not self.ecm_buffer
            or self.ecm_forward_wait <= 0
            or not self.neighbor_bridge_addresses
        ):
            return
        self._ensure_bridge_client()

        deadline = time.time() + self.ecm_forward_wait
        poll_interval = 1.0
        while time.time() < deadline:
            fresh = [ecm for ecm in self.ecm_buffer.get_fresh_ecms() if not ecm.is_signal]
            if fresh:
                logger.debug(
                    "ECM buffer received %d entries from neighbors; proceeding to forward",
                    len(fresh),
                )
                return
            logger.debug(
                "Waiting for neighbor ECMs (%d neighbors)...",
                len(self.neighbor_bridge_addresses),
            )
            self._abort_if_global_stop()
            time.sleep(poll_interval)
            self._abort_if_global_stop()
        logger.info(
            "No ECMs received from %d neighbors after waiting %ds; continuing without them",
            len(self.neighbor_bridge_addresses),
            int(self.ecm_forward_wait),
        )

    def _ensure_bridge_client(self) -> bool:
        """Ensure bridge infrastructure is running; restart if necessary."""
        if not self.is_bridge_node or not self.inter_cluster_enabled:
            return False
        if self.bridge_client:
            return True
        if not self.inter_edges:
            logger.warning(
                "Bridge client unavailable for %s and no inter_edges configured",
                self.node_id,
            )
            return False
        logger.warning(
            "Bridge client missing for %s; restarting bridge server on port %d",
            self.node_id,
            self.port + 2000,
        )
        # Tear down any existing server before reconfiguring.
        self.stop_bridge_server()
        return self._init_bridge_with_retries(
            self.inter_edges,
            max_attempts=3,
            delay=1.0,
            fatal=False,
        )

    def run_secure_aggregation_round(self) -> List[float]:
        """Run one round of secure aggregation protocol."""
        self._abort_if_global_stop()
        logger.info(f"Starting secure aggregation round {self.current_round}")
        # Client-side secure aggregation state
        client = SecureAggregationNode(
            self.node_id,
            signing_private=self.signing_keypair.private_key if self.signing_keypair else None,
            signing_public=self.signing_keypair.public_key if self.signing_keypair else None,
        )

        # Get aggregator address
        agg_port = int(self.aggregator_address.split(":")[-1]) + 1000
        agg_host = self.aggregator_address.split(":")[0]
        agg_addr = f"{agg_host}:{agg_port}"

        channel = grpc.insecure_channel(agg_addr)
        stub = secureagg_pb2_grpc.AggregatorServiceStub(channel)

        # SAP Round 0: Advertise keys
        logger.info("SAP-Round 0: Advertising keys")
        advert_msg = client.advertise_keys()

        # Retry logic for initial aggregator connection.
        retry_delay = 1
        max_retries = 30
        if self.inter_cluster_enabled:
            retry_delay = 2
            max_retries = 75
        response = None
        last_rpc_error: Optional[grpc.RpcError] = None
        for attempt in range(max_retries):
            self._abort_if_global_stop()
            try:
                response = stub.Round0AdvertiseKeys(
                    secureagg_pb2.KeyAdvertisement(
                        node_id=self.node_id,
                        c_public_key=advert_msg.c_public,
                        s_public_key=advert_msg.s_public,
                        signature=advert_msg.signature,
                    ),
                    timeout=30
                )
                break
            except grpc.RpcError as e:
                last_rpc_error = e
                if attempt < max_retries - 1:
                    if attempt < 5 or (attempt + 1) % 5 == 0:
                        logger.warning(
                            "Aggregator %s at %s connection attempt %d/%d failed (%s); retrying in %ds...",
                            self.aggregator_id,
                            agg_addr,
                            attempt + 1,
                            max_retries,
                            e.code().name if hasattr(e, "code") else "unknown",
                            retry_delay,
                        )
                    attempt_idx = attempt + 1
                    self._maybe_check_convergence_during_retry(
                        attempt_idx, max_retries, "Round0AdvertiseKeys"
                    )
                    time.sleep(retry_delay)
                    self._abort_if_global_stop()
                else:
                    break

        if response is None:
            self._raise_aggregator_unavailable(
                "Round0AdvertiseKeys", max_retries, last_rpc_error
            )

        if not response.accepted:
            raise RuntimeError(f"Round 0 failed: {response.message}")

        # Wait for ALL clique members to advertise (not just threshold).
        expected_participants = len(self.clique_members)
        while len(response.all_keys) < expected_participants:
            self._abort_if_global_stop()
            time.sleep(1)
            self._abort_if_global_stop()
            response = stub.Round0AdvertiseKeys(
                secureagg_pb2.KeyAdvertisement(
                    node_id=self.node_id,
                    c_public_key=advert_msg.c_public,
                    s_public_key=advert_msg.s_public,
                    signature=advert_msg.signature,
                ),
                timeout=30
            )

            logger.info(f"SAP-Round 0 complete: received {len(response.all_keys)} participants")

        # Pass received advertisements to client
        ordered_participants = [p.node_id for p in response.all_keys]
        adverts = [
            AdvertiseMessage(
                node_id=p.node_id,
                c_public=bytes(p.c_public_key),
                s_public=bytes(p.s_public_key),
                signature=bytes(p.signature),
                signing_public=None,
            )
            for p in response.all_keys
        ]
        client.receive_advertisements(adverts)

        # SAP Round 1: Share keys (simplified - just send empty shares)
        logger.info("SAP-Round 1: Sharing keys")
        ct_list = client.create_round1_ciphertexts(ordered_participants, self.threshold)
        response1 = stub.Round1ShareKeys(
            secureagg_pb2.ShareKeysMessage(
                node_id=self.node_id,
                ciphertexts=[
                    secureagg_pb2.Round1Ciphertext(
                        sender_id=ct.sender_id,
                        recipient_id=ct.recipient_id,
                        iv=ct.iv,
                        ciphertext=ct.ciphertext,
                        tag=ct.tag,
                    )
                    for ct in ct_list
                ],
            ),
            timeout=30,
        )
        mailbox = [
            Round1Ciphertext(
                sender_id=ct.sender_id,
                recipient_id=ct.recipient_id,
                iv=bytes(ct.iv),
                ciphertext=bytes(ct.ciphertext),
                tag=bytes(ct.tag),
            )
            for ct in response1.mailbox
        ]
        # Poll until mailbox has entries from all n participants (each sends to all including self).
        expected_mail = len(ordered_participants)
        while len(mailbox) < expected_mail:
            self._abort_if_global_stop()
            time.sleep(1)
            self._abort_if_global_stop()
            response1 = stub.Round1ShareKeys(
                secureagg_pb2.ShareKeysMessage(node_id=self.node_id, ciphertexts=[]),
                timeout=30,
            )
            mailbox = [
                Round1Ciphertext(
                    sender_id=ct.sender_id,
                    recipient_id=ct.recipient_id,
                    iv=bytes(ct.iv),
                    ciphertext=bytes(ct.ciphertext),
                    tag=bytes(ct.tag),
                )
                for ct in response1.mailbox
            ]
        client.receive_round1_ciphertexts(mailbox)

        # SAP Round 2: Send masked input
        logger.info("SAP-Round 2: Sending masked model")
        model_vec = flatten_params(self.model)
        quantized = quantize_vector(model_vec, self.scale)

        masked = client.create_masked_input(quantized)
        masked_bytes = [_int_to_bytes(val, SHARE_BYTES) for val in masked.masked_vector]
        response2 = stub.Round2MaskedInput(
            secureagg_pb2.MaskedInputMessage(node_id=self.node_id, masked_vector=masked_bytes),
            timeout=30,
        )

        # Wait for survivors list
        while not response2.survivors:
            self._abort_if_global_stop()
            time.sleep(1)
            self._abort_if_global_stop()
            response2 = stub.Round2MaskedInput(
                secureagg_pb2.MaskedInputMessage(node_id=self.node_id, masked_vector=masked_bytes),
                timeout=30,
            )

        logger.info(f"SAP-Round 2 complete: {len(response2.survivors)} survivors")

        # SAP Round 3: Consistency check
        logger.info("SAP-Round 3: Consistency check")
        survivor_sig = client.sign_survivor_list(response2.survivors)
        response3 = stub.Round3ConsistencyCheck(
            secureagg_pb2.ConsistencySignature(node_id=self.node_id, signature=survivor_sig.signature),
            timeout=30,
        )

        # SAP Round 4: Unmask (simplified - send empty shares)
        logger.info("SAP-Round 4: Unmasking")
        dropouts = set(ordered_participants) - set(response2.survivors)
        unmask_payload = client.prepare_unmasking_payload(dropouts, response2.survivors)
        response4 = stub.Round4Unmask(
            secureagg_pb2.UnmaskShares(
                node_id=self.node_id,
                dropout_s_shares={k: _encode_share(x, s) for k, (x, s) in unmask_payload.s_shares_for_dropouts.items()},
                survivor_b_shares={k: _encode_share(x, b) for k, (x, b) in unmask_payload.b_shares_for_survivors.items()},
            ),
            timeout=30,
        )

        # Wait for aggregation to complete
        while not response4.aggregation_complete:
            self._abort_if_global_stop()
            time.sleep(1)
            self._abort_if_global_stop()
            response4 = stub.Round4Unmask(
                secureagg_pb2.UnmaskShares(
                    node_id=self.node_id,
                    dropout_s_shares={k: _encode_share(x, s) for k, (x, s) in unmask_payload.s_shares_for_dropouts.items()},
                    survivor_b_shares={k: _encode_share(x, b) for k, (x, b) in unmask_payload.b_shares_for_survivors.items()},
                ),
                timeout=30,
            )

        logger.info("SAP-Round 4 complete: aggregation done")

        # Get global model
        logger.info("Fetching global model")
        model_response = stub.GetGlobalModel(secureagg_pb2.ModelRequest(round=self.current_round), timeout=30)

        channel.close()

        aggregated = list(model_response.model_weights)
        logger.info(f"Received aggregated model ({len(aggregated)} parameters)")
        return aggregated

    def run_training_loop(self) -> None:
        """Convergence-driven training loop with secure aggregation.

        Training continues until either:
        1. Global convergence is achieved (model delta below tolerance for patience rounds,
           and all neighbor clusters have also converged)
        2. Maximum rounds limit is reached (safety cap)
        """
        local_epochs = self.training_config["local_epochs"]
        max_rounds = self.max_training_rounds
        convergence_warmup = max(0, self.convergence_config.warmup_rounds)

        # Initialize convergence tracker
        self.convergence_tracker = ConvergenceTracker(
            self.convergence_config, f"cluster_{self.clique_id}"
        )
        self._refresh_central_metadata()

        logger.info(
            f"Starting convergence-driven training (max_rounds={max_rounds}, "
            f"warmup_rounds={convergence_warmup}, "
            f"tol_abs={self.convergence_config.tol_abs}, patience={self.convergence_config.patience})"
        )

        should_stop = False
        stop_reason = ""

        while self.current_round < max_rounds and not should_stop:
            round_idx = self.current_round
            logger.info(f"\n{'='*60}")
            logger.info(f"Round {round_idx + 1}/{max_rounds}")
            logger.info(f"{'='*60}")
            should_stop, stop_reason = self._refresh_convergence_state(should_stop, stop_reason)
            if should_stop:
                logger.info(f"Stopping training before local update: {stop_reason}")
                break

            # Phase 1: Local training
            logger.info("Phase 1: Local training")
            self.train_local(local_epochs)

            acc_before = self.evaluate()
            logger.info(f"Accuracy before aggregation: {acc_before:.4f}")

            should_stop, stop_reason = self._refresh_convergence_state(should_stop, stop_reason)
            if should_stop:
                logger.info(f"Stopping training after local update: {stop_reason}")
                break

            # Aggregator election
            self.aggregator_id = self.elect_aggregator(round_idx)
            self.aggregator_address = self.participant_map[self.aggregator_id]
            self.is_aggregator = (self.aggregator_id == self.node_id)
            aggregator_is_bridge = is_bridge_node(self.aggregator_id, self.inter_edges)
            wait_for_aggregator = 5
            if self.inter_cluster_enabled and aggregator_is_bridge:
                wait_for_aggregator = max(wait_for_aggregator, 8)

            if self.is_aggregator:
                logger.info(f"*** This node is the AGGREGATOR for round {round_idx} ***")
                self.start_aggregator_server()

            logger.info(
                "Waiting for aggregator %s to be ready (sleeping %ds)...",
                self.aggregator_id,
                wait_for_aggregator,
            )
            wait_remaining = wait_for_aggregator
            while wait_remaining > 0 and not should_stop:
                interval = min(1.0, wait_remaining)
                time.sleep(interval)
                wait_remaining -= interval
                should_stop, stop_reason = self._refresh_convergence_state(should_stop, stop_reason)
                if should_stop:
                    logger.info(f"Stopping training while waiting for aggregator: {stop_reason}")
                    break

            if should_stop:
                if self.is_aggregator:
                    self.stop_aggregator_server()
                break

            cid: Optional[str] = None
            model_hash: Optional[str] = None
            model_data_id: Optional[str] = None
            try:
                round_failed = False
                # Phase 2: Secure aggregation
                logger.info("Phase 2: Secure aggregation")
                try:
                    aggregated_weights = self.run_secure_aggregation_round()
                except GlobalStopRequested:
                    should_stop = True
                    stop_reason = "global_convergence"
                    round_failed = True
                    logger.info("Aborting secure aggregation due to confirmed global convergence")
                    break

                # Phase 3: Bridge nodes forward ECMs (if inter-cluster is enabled)
                if self.is_bridge_node:
                    if self.inter_cluster_enabled:
                        if self.neighbor_bridge_addresses:
                            self._wait_for_neighbor_ecms()
                        logger.info("Phase 3: Forwarding ECMs to aggregator")
                        forwarded = self.forward_ecms_to_aggregator()
                        if forwarded > 0:
                            if self.is_aggregator:
                                logger.info(
                                    "Phase 3: Aggregator staging %d ECMs from neighbor clusters",
                                    forwarded,
                                )
                            else:
                                logger.info(f"Phase 3: Forwarded {forwarded} ECMs to aggregator")
                        else:
                            logger.info("Phase 3: No fresh ECMs to forward this round")
                    else:
                        logger.info(
                            "Phase 3: Skipped ECM forwarding because inter_cluster.enabled is false"
                        )

                if aggregated_weights:
                    logger.info("Phase 4: Updating model with aggregated weights")
                    dequantized = dequantize_vector([int(w) for w in aggregated_weights], self.scale)
                    load_params(self.model, dequantized)

                    # Phase 5: Inter-cluster merge (aggregator only)
                    if self.is_aggregator and self.inter_cluster_enabled and self.inter_cluster_aggregator:
                        # Wait briefly for ECMs from bridge nodes to arrive
                        time.sleep(2)
                        logger.info("Phase 5: Inter-cluster merge")
                        intra_model = np.array(dequantized, dtype=np.float32)

                        if self.ecm_buffer:
                            consumed_ecms = self.ecm_buffer.get_fresh_ecms()
                            for ecm in consumed_ecms:
                                self.inter_cluster_aggregator.receive_ecms(self.node_id, [ecm])
                                # Update neighbor convergence status from ECM
                                if hasattr(ecm, "cluster_converged"):
                                    self.convergence_tracker.receive_neighbor_convergence(
                                        ecm.source_cluster, ecm.cluster_converged
                                    )
                            if consumed_ecms:
                                removed = self.ecm_buffer.remove_cids([ecm.cid for ecm in consumed_ecms])
                                logger.debug("Aggregator consumed and removed %d ECMs from buffer", removed)

                        merged_data = self.inter_cluster_aggregator.process_round(
                            intra_model, round_idx
                        )
                        merged_model, cid, model_hash = merged_data
                        model_data_id = getattr(self.inter_cluster_aggregator, "last_data_id", None)

                        # Update convergence state with merged model
                        conv_state = self.convergence_tracker.update(merged_model)
                        should_stop = conv_state.should_stop
                        stop_reason = conv_state.stop_reason
                        self._latest_cluster_converged = conv_state.cluster_converged
                        self._latest_delta_norm = conv_state.delta_norm

                        # Store convergence state in aggregator servicer for distribution
                        if self.aggregator_servicer:
                            self.aggregator_servicer.set_convergence_state(
                                model_cid=cid,
                                model_hash=model_hash,
                                model_data_id=model_data_id,
                                should_stop=should_stop,
                                stop_reason=stop_reason,
                                delta_norm=conv_state.delta_norm,
                                cluster_converged=conv_state.cluster_converged,
                            )

                        dequantized = merged_model.tolist()
                        load_params(self.model, dequantized)
                        logger.info(f"Inter-cluster merge complete: cid={cid[:16] if cid else 'N/A'}...")

                    elif self.is_aggregator and self.convergence_config.enabled:
                        # Aggregator without inter-cluster: still track convergence
                        model_array = np.array(dequantized, dtype=np.float32)
                        conv_state = self.convergence_tracker.update(model_array)
                        should_stop = conv_state.should_stop
                        stop_reason = conv_state.stop_reason
                        self._latest_cluster_converged = conv_state.cluster_converged
                        self._latest_delta_norm = conv_state.delta_norm

                        if self.aggregator_servicer:
                            self.aggregator_servicer.set_convergence_state(
                                model_cid=None,
                                model_hash=None,
                                model_data_id=None,
                                should_stop=should_stop,
                                stop_reason=stop_reason,
                                delta_norm=conv_state.delta_norm,
                                cluster_converged=conv_state.cluster_converged,
                            )

                    elif not self.is_aggregator:
                        # Non-aggregator: fetch convergence decision from aggregator.
                        # Bridge nodes wait until the aggregator publishes IPFS/chain metadata
                        # so we can gossip ECM references reliably.
                        wait_for_model_ref = self.is_bridge_node and self.inter_cluster_enabled
                        model_response = self._fetch_convergence_status(wait_for_model_ref=wait_for_model_ref)
                        if model_response:
                            should_stop = model_response.should_stop
                            stop_reason = model_response.stop_reason
                            self._latest_cluster_converged = model_response.cluster_converged
                            self._latest_delta_norm = model_response.delta_norm

                            response_cid = model_response.model_cid or ""
                            response_hash = model_response.model_hash or ""
                            if self.is_bridge_node and response_cid and response_hash:
                                # Remember reference for ECM gossip even if fetch fails locally
                                cid = response_cid
                                model_hash = response_hash

                            cluster_anchor_id = f"cluster_{self.clique_id}"
                            response_data_id = getattr(model_response, "model_data_id", "")
                            if response_data_id and self.blockchain:
                                self.blockchain.remember_anchor(
                                    cluster_anchor_id,
                                    round_idx,
                                    response_data_id,
                                    model_response.model_cid or None,
                                    model_response.model_hash or None,
                                )

                            if model_response.model_cid and self.ipfs:
                                logger.info(
                                    f"Fetching merged model from IPFS fallback: {model_response.model_cid[:16]}..."
                                )
                                try:
                                    merged_from_ipfs = self.ipfs.get(model_response.model_cid)
                                except Exception as ipfs_err:  # noqa: BLE001
                                    merged_from_ipfs = None
                                    logger.warning(
                                        "IPFS fallback fetch failed for cid=%s: %s",
                                        model_response.model_cid[:16],
                                        ipfs_err,
                                    )
                                if merged_from_ipfs is not None:
                                    load_params(self.model, merged_from_ipfs.tolist())
                                    if self.is_bridge_node:
                                        cid = model_response.model_cid
                                        model_hash = model_response.model_hash
                                else:
                                    logger.warning(
                                        "Failed to fetch merged model from IPFS fallback; "
                                        "sharing ECM reference anyway"
                                    )

                    # Prime tracker state so future aggregator rounds have accurate baseline.
                    if not self.is_aggregator:
                        self._prime_convergence_tracker_state()

                acc_after = self.evaluate()
                logger.info(f"Accuracy after aggregation: {acc_after:.4f}")
                logger.info(f"Improvement: {acc_after - acc_before:+.4f}")

            except AggregatorUnavailable as exc:
                round_failed = True
                logger.warning(
                    "%s. Will retry after backoff unless convergence is confirmed.",
                    exc,
                )
            except Exception as e:
                round_failed = True
                logger.error("Secure aggregation failed: %s", e, exc_info=True)

            finally:
                if self.is_aggregator:
                    time.sleep(2)
                    self.stop_aggregator_server()

            if round_failed:
                retry_delay = 5
                logger.warning(
                    "Round %d failed. Retrying after %ds once aggregator %s is reachable.",
                    round_idx,
                    retry_delay,
                    self.aggregator_id,
                )
                time.sleep(retry_delay)
                continue

            # Phase 6: ECM gossip with convergence status (bridge nodes only)
            if cid and model_hash and self.is_bridge_node:
                logger.info("Phase 6: ECM gossip to neighbor clusters")
                cluster_converged = self._latest_cluster_converged
                delta_norm = self._latest_delta_norm
                self.gossip_ecm_with_convergence(cid, model_hash, round_idx, cluster_converged, delta_norm)
                self._broadcast_central_signal(round_idx, cluster_converged, delta_norm)

            should_stop, stop_reason = self._sync_stop_state_from_tracker(should_stop, stop_reason)
            if should_stop:
                logger.info(f"Stopping training: {stop_reason}")
                self._check_cached_convergence_data()
                break

            logger.info(f"Training Round {round_idx} complete. Waiting before next round...")
            time.sleep(5)
            self.current_round += 1

        logger.info("\n" + "="*60)
        logger.info(f"Training completed after {self.current_round + 1} rounds (reason: {stop_reason or 'max_rounds'})")
        logger.info("="*60)

    def _fetch_convergence_status(self, wait_for_model_ref: bool = False) -> Optional[secureagg_pb2.ModelResponse]:
        """
        Fetch convergence status from aggregator.

        Args:
            wait_for_model_ref: If True, poll until aggregator publishes IPFS metadata.
                Bridge nodes enable this so they always obtain CID/hash for ECM gossip.
        """
        channel = None
        try:
            agg_port = int(self.aggregator_address.split(":")[-1]) + 1000
            agg_host = self.aggregator_address.split(":")[0]
            agg_addr = f"{agg_host}:{agg_port}"

            channel = grpc.insecure_channel(agg_addr)
            stub = secureagg_pb2_grpc.AggregatorServiceStub(channel)

            attempts = 0
            max_attempts = 10 if (wait_for_model_ref and self.inter_cluster_enabled) else 1
            delay = 2
            response: Optional[secureagg_pb2.ModelResponse] = None

            while attempts < max_attempts:
                response = stub.GetGlobalModel(
                    secureagg_pb2.ModelRequest(round=self.current_round),
                    timeout=10
                )
                if (
                    not wait_for_model_ref
                    or not self.inter_cluster_enabled
                    or (response.model_cid and response.model_hash)
                ):
                    return response

                attempts += 1
                if attempts < max_attempts:
                    logger.debug(
                        "Aggregator metadata not ready (attempt %d/%d); waiting %ds",
                        attempts,
                        max_attempts,
                        delay,
                    )
                    time.sleep(delay)

            if wait_for_model_ref and self.inter_cluster_enabled:
                logger.warning(
                    "Aggregator metadata unavailable after %d attempts; proceeding without CID/hash",
                    max_attempts,
                )
            return response
        except Exception as e:
            logger.warning(f"Failed to fetch convergence status: {e}")
            return None
        finally:
            if channel:
                channel.close()

    def gossip_ecm_with_convergence(
        self, cid: str, model_hash: str, round_num: int,
        cluster_converged: bool, delta_norm: float
    ) -> None:
        """Gossip ECM with convergence status to neighbor cluster bridge nodes."""
        if not self.is_bridge_node:
            return
        if not self._ensure_bridge_client():
            logger.warning("Cannot gossip ECM (node %s): bridge client unavailable", self.node_id)
            return

        cluster_id = f"cluster_{self.clique_id}"
        accepted = self.bridge_client.broadcast_ecm_with_convergence(
            self.neighbor_bridge_addresses,
            cluster_id,
            round_num,
            cid,
            model_hash,
            cluster_converged,
            delta_norm,
        )
        logger.info(
            f"Gossiped ECM with convergence (converged={cluster_converged}) to "
            f"{accepted}/{len(self.neighbor_bridge_addresses)} neighbors"
        )

    def _broadcast_central_signal(
        self,
        round_idx: int,
        cluster_converged: bool,
        delta_norm: float,
    ) -> None:
        """Send convergence-only signal to central checkers."""
        if not self.central_neighbor_addresses:
            return

        delivered = 0
        total_targets = len(self.central_neighbor_addresses)

        if self.node_id in self.central_neighbor_addresses and self.central_checker:
            self.central_checker.record_signal(
                f"cluster_{self.clique_id}",
                round_idx,
                cluster_converged,
                delta_norm,
            )
            delivered += 1

        remote_addresses = [
            addr
            for checker_id, addr in self.central_neighbor_addresses.items()
            if checker_id != self.node_id
        ]
        if remote_addresses:
            if not self.is_bridge_node or not self.bridge_client:
                logger.warning(
                    "Cannot dispatch convergence signal to remote central nodes: bridge client unavailable"
                )
            else:
                cid = f"signal::cluster_{self.clique_id}::{round_idx}"
                accepted = self.bridge_client.broadcast_ecm_with_convergence(
                    remote_addresses,
                    f"cluster_{self.clique_id}",
                    round_idx,
                    cid,
                    "signal",
                    cluster_converged,
                    delta_norm,
                    convergence_data_id=None,
                )
                delivered += accepted

        if delivered > 0:
            logger.info(
                "Dispatched convergence signal (converged=%s) to %d/%d central targets",
                cluster_converged,
                delivered,
                total_targets,
            )

    def _broadcast_convergence_data_id_to_neighbors(
        self,
        data_id: str,
        round_idx: int,
        targets: Optional[List[str]] = None,
    ) -> None:
        """Broadcast convergence confirmation to bridge neighbors."""
        if not data_id or not self.bridge_client or not self.inter_cluster_enabled:
            return
        addresses = targets or self.neighbor_bridge_addresses
        if not addresses:
            logger.debug("No neighbor bridges available for convergence data broadcast")
            return
        cluster_id = f"cluster_{self.clique_id}"
        cid = f"signal::convergence::{data_id}"
        accepted = self.bridge_client.broadcast_ecm_with_convergence(
            addresses,
            cluster_id,
            round_idx,
            cid,
            "signal",
            True,
            0.0,
            convergence_data_id=data_id,
        )
        logger.info(
            "Broadcast convergence data_id=%s to %d/%d bridge targets",
            data_id,
            accepted,
            len(addresses),
        )

    def _broadcast_convergence_data_id_to_clique(self, data_id: str, round_idx: int) -> None:
        """Broadcast convergence confirmation to non-bridge clique members."""
        if not data_id or not self.bridge_client or not self.inter_cluster_enabled:
            return
        if not self._clique_signal_addresses:
            return
        targets: List[str] = []
        for member, address in self._clique_signal_addresses.items():
            if member == self.node_id:
                continue
            if self.inter_edges and is_bridge_node(member, self.inter_edges):
                continue
            targets.append(address)
        if not targets:
            return
        cluster_id = f"cluster_{self.clique_id}"
        cid = f"signal::convergence::{data_id}"
        accepted = self.bridge_client.broadcast_ecm_with_convergence(
            targets,
            cluster_id,
            round_idx,
            cid,
            "signal",
            True,
            0.0,
            convergence_data_id=data_id,
        )
        logger.info(
            "Propagated convergence data_id=%s to %d/%d clique peers",
            data_id,
            accepted,
            len(targets),
        )

    def _register_convergence_data_id(self, data_id: Optional[str], round_idx: Optional[int] = None) -> bool:
        """Cache convergence data_id for later verification."""
        if not data_id:
            return False
        if data_id in self._known_convergence_data_ids:
            return False
        self._known_convergence_data_ids.add(data_id)
        if data_id not in self._acknowledged_convergence_data_ids:
            self._pending_convergence_data_ids.add(data_id)
        logger.info(
            "Cached convergence confirmation data_id=%s%s",
            data_id,
            f" (round={round_idx})" if round_idx is not None else "",
        )
        self._send_convergence_signal_to_aggregator(data_id, round_idx)
        return True

    def _handle_convergence_signal_from_peer(self, data_id: str, round_idx: int) -> None:
        """Receive convergence notification from clique peers via aggregator RPC."""
        logger.info(
            "Aggregator %s received convergence signal via RPC data_id=%s round=%s",
            self.node_id,
            data_id,
            round_idx,
        )
        if self._register_convergence_data_id(data_id, round_idx):
            self._broadcast_convergence_data_id_to_neighbors(data_id, round_idx)
            self._broadcast_convergence_data_id_to_clique(data_id, round_idx)

    def _send_convergence_signal_to_aggregator(self, data_id: str, round_idx: Optional[int]) -> None:
        """Notify the current aggregator about convergence confirmation."""
        if (
            not data_id
            or self.aggregator_id is None
            or self.aggregator_id == self.node_id
            or not self.aggregator_address
        ):
            return
        try:
            host, port_str = self.aggregator_address.split(":")
            agg_port = int(port_str) + 1000
        except ValueError:
            logger.debug(
                "Cannot notify aggregator %s about convergence: invalid address %s",
                self.aggregator_id,
                self.aggregator_address,
            )
            return
        agg_addr = f"{host}:{agg_port}"
        channel = grpc.insecure_channel(agg_addr)
        stub = secureagg_pb2_grpc.AggregatorServiceStub(channel)
        try:
            stub.NotifyConvergenceSignal(
                secureagg_pb2.ConvergenceSignal(
                    data_id=data_id,
                    round=round_idx if round_idx is not None else self.current_round,
                ),
                timeout=5,
            )
        except grpc.RpcError as exc:
            logger.debug(
                "Failed to notify aggregator %s about convergence data_id=%s: %s",
                self.aggregator_id,
                data_id,
                exc,
            )
        finally:
            channel.close()

    def _handle_global_convergence_announcement(
        self,
        round_idx: int,
        data_id: Optional[str],
        payload: Optional[Dict],
    ) -> None:
        """Callback executed when central checker finalizes convergence."""
        if not data_id:
            logger.warning("Central checker declared convergence but no data_id was returned")
            return
        if payload:
            self._convergence_payload_cache[data_id] = payload
        if self._register_convergence_data_id(data_id, round_idx):
            self._broadcast_convergence_data_id_to_neighbors(data_id, round_idx)
            self._broadcast_convergence_data_id_to_clique(data_id, round_idx)

    def _process_incoming_signals(self) -> None:
        if not self.ecm_buffer:
            return
        signals = self.ecm_buffer.pop_signal_ecms()
        for ecm in signals:
            if ecm.convergence_data_id:
                is_new = self._register_convergence_data_id(ecm.convergence_data_id, ecm.round_idx)
                if is_new and self.is_bridge_node:
                    self._broadcast_convergence_data_id_to_clique(ecm.convergence_data_id, ecm.round_idx)
            if (
                self.central_checker
                and ecm.source_cluster
                and ecm.round_idx >= 0
                and ecm.cluster_delta_norm is not None
            ):
                self.central_checker.record_signal(
                    ecm.source_cluster,
                    ecm.round_idx,
                    ecm.cluster_converged,
                    ecm.cluster_delta_norm,
                )

    def _check_cached_convergence_data(self) -> None:
        if not self.blockchain or not self._pending_convergence_data_ids:
            return
        for data_id in list(self._pending_convergence_data_ids):
            record: Optional[Dict[str, Any]]
            try:
                record = self.blockchain.fetch_data(data_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch convergence payload data_id=%s: %s", data_id, exc)
                continue
            metadata = self._extract_convergence_metadata(record)
            if not metadata:
                continue
            round_idx = metadata.get("round")
            if round_idx is None:
                logger.warning("Convergence payload %s missing round field", data_id)
                continue
            self._pending_convergence_data_ids.discard(data_id)
            self._acknowledged_convergence_data_ids.add(data_id)
            self._convergence_payload_cache[data_id] = metadata
            if self.convergence_tracker:
                self.convergence_tracker.receive_global_convergence(int(round_idx))
            self._confirmed_global_convergence_round = int(round_idx)
            self._confirmed_global_convergence_reason = "global_convergence"
            logger.info(
                "Verified convergence payload data_id=%s (round=%s); stopping after confirmation",
                data_id,
                round_idx,
            )

    def _refresh_convergence_state(self, should_stop: bool, stop_reason: str) -> Tuple[bool, str]:
        """Process buffered signals, fetch payloads, and align stop flags."""
        self._process_incoming_signals()
        self._check_cached_convergence_data()
        return self._sync_stop_state_from_tracker(should_stop, stop_reason)

    def _extract_convergence_metadata(self, record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not record:
            return None
        payload = record.get("payload")
        if payload is None:
            return None
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("Failed to parse convergence payload string")
                return None
        metadata = payload.get("metadata")
        if metadata is None:
            metadata = payload
        if not isinstance(metadata, dict):
            return None
        if metadata.get("type") != "global_convergence":
            return None
        return metadata

    def _sync_stop_state_from_tracker(self, should_stop: bool, stop_reason: str) -> Tuple[bool, str]:
        """
        Align local loop control flags with the convergence tracker state.

        Returns:
            Tuple updated with the tracker decision if a central/global stop
            signal was received outside of the aggregator update flow.
        """
        if not should_stop and self._confirmed_global_convergence_round is not None:
            reason = self._confirmed_global_convergence_reason or stop_reason or "global_convergence"
            logger.info(
                "Global convergence confirmed on-chain at round %s; halting",
                self._confirmed_global_convergence_round,
            )
            return True, reason
        if self.convergence_tracker and not should_stop:
            tracker_state = self.convergence_tracker.state
            if tracker_state.should_stop:
                stop_reason = tracker_state.stop_reason or stop_reason or "global_convergence"
                logger.info(
                    "Convergence tracker requested stop (reason=%s, round=%s)",
                    stop_reason,
                    tracker_state.global_stop_round,
                )
                return True, stop_reason
        return should_stop, stop_reason

    def _abort_if_global_stop(self) -> None:
        """Raise if global convergence has already been confirmed."""
        should_stop, _ = self._refresh_convergence_state(False, "")
        if should_stop:
            raise GlobalStopRequested()

    def start(self) -> None:
        """Start the node service."""
        logger.info(f"Starting node {self.node_id} on port {self.port}")

        # Register with TTP
        self.register_with_ttp()

        # Setup data and model
        self.setup_data()
        self.setup_model()

        # Setup inter-cluster aggregation
        self.setup_inter_cluster()

        # Load inter_edges from topology file or config
        inter_edges: List[Tuple[str, str]] = []
        topology_file = self.inter_cluster_config.get("topology_file", "/app/config/topology.json")
        if Path(topology_file).exists():
            with open(topology_file) as f:
                topology_data = json.load(f)
            inter_edges = [(e[0], e[1]) for e in topology_data.get("inter_edges", [])]
            logger.info(f"Loaded {len(inter_edges)} inter-edges from {topology_file}")
        else:
            inter_edges_config = self.inter_cluster_config.get("inter_edges", [])
            inter_edges = [(e[0], e[1]) for e in inter_edges_config]

        # Wait for clique members or all nodes to be ready
        if self.clique_members:
            expected_count = len(self.clique_members)
            logger.info(f"Waiting for {expected_count} clique members to register...")
            registered_clique_members = set(self.clique_members) & set(self.participant_map.keys())
            while len(registered_clique_members) < expected_count:
                time.sleep(2)
                self.register_with_ttp()
                registered_clique_members = set(self.clique_members) & set(self.participant_map.keys())
            logger.info(f"All {len(registered_clique_members)} clique members ready. Starting training...")
        else:
            logger.info("Waiting for all nodes to register...")
            while len(self.participants) < self.dataset_config["num_clients"]:
                time.sleep(2)
                self.register_with_ttp()
            logger.info(f"All {len(self.participants)} nodes are ready. Starting training...")

        self._update_clique_signal_addresses()

        if inter_edges:
            bridge_ready = self._init_bridge_with_retries(inter_edges, fatal=False)
            if not bridge_ready and self.is_bridge_node:
                logger.warning(
                    "Bridge node %s could not initialize bridge server; continuing without inter-cluster gossip",
                    self.node_id,
                )

        time.sleep(2)

        try:
            # Run training loop
            self.run_training_loop()
        finally:
            # Cleanup bridge server
            self.stop_bridge_server()

        logger.info(f"Node {self.node_id} finished")


def main() -> None:
    """Entry point for node service."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Secure Aggregation Node Service")
    parser.add_argument("--config", required=True, help="Path to node configuration file")
    args = parser.parse_args()

    node = NodeService(args.config)
    node.start()


if __name__ == "__main__":
    main()
