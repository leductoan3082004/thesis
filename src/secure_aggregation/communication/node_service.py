"""Node service for federated learning with secure aggregation."""

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
from secure_aggregation.convergence import ConvergenceConfig, ConvergenceSignal, ConvergenceTracker
from secure_aggregation.convergence.central_broadcast import (
    CENTRAL_METADATA_CLUSTER_ID,
    fetch_central_metadata,
    fetch_global_convergence_round,
)
from secure_aggregation.convergence.central_checker import CentralChecker
from secure_aggregation.config.models import NodeRole
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


class NodeService:
    """Node service that coordinates training and secure aggregation."""

    def __init__(self, config_path: str) -> None:
        self.config = self._load_config(config_path)
        self.node_id = self.config["node_id"]
        self.role = NodeRole(self.config["role"])
        self.ttp_address = self.config["ttp_address"]
        self.port = self.config["port"]
        self.dataset_config = self.config["dataset"]
        self.training_config = self.config["training"]
        self.secagg_config = self.config["secure_agg"]
        self.threshold = self.secagg_config["threshold"]
        self.scale = self.secagg_config["scale"]

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

        # Convergence state
        self.convergence_config = ConvergenceConfig.from_dict(self.config.get("convergence"))
        self.convergence_tracker: Optional[ConvergenceTracker] = None
        self.central_metadata = None
        self._pending_convergence_signal: Optional[ConvergenceSignal] = None
        self.central_checker: Optional[CentralChecker] = None
        self.aggregator_servicer: Optional[AggregatorServicer] = None
        self._bootstrap_anchors: List[Tuple[str, int, str, Optional[str], Optional[str]]] = []
        self._logged_central_addresses = False

        logger.info(f"Node {self.node_id} initialized (role={self.role}, port={self.port})")

    def _load_config(self, path: str) -> dict:
        """Load node configuration from JSON file."""
        with open(path) as f:
            return json.load(f)

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
            logger.info("Inter-cluster aggregation disabled")
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

    def setup_bridge_node(self, inter_edges: List[Tuple[str, str]]) -> None:
        """Setup bridge node if this node has inter-clique connections."""
        self.inter_edges = inter_edges
        self.is_bridge_node = is_bridge_node(self.node_id, inter_edges)

        if not self.is_bridge_node:
            logger.info(f"Node {self.node_id} is not a bridge node")
            return

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

        if self.inter_cluster_enabled and self.ecm_buffer:
            self.bridge_server = serve_bridge(
                self.node_id,
                self.port + 2000,
                self.ecm_buffer,
            )
            self.bridge_client = BridgeClient(self.node_id)
            logger.info(f"Bridge server started on port {self.port + 2000}")
        self._update_central_neighbor_addresses()

    def stop_bridge_server(self) -> None:
        """Stop bridge server."""
        if self.bridge_server:
            self.bridge_server.stop(0)
            self.bridge_server = None
        if self.bridge_client:
            self.bridge_client.close()
            self.bridge_client = None
        logger.info("Bridge server stopped")

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
        """Build mapping to central neighbor addresses when metadata is available."""
        self.central_neighbor_addresses = {}
        if not self.central_metadata:
            return
        for node_id in self.central_metadata.central_nodes:
            address = self.neighbor_address_map.get(node_id)
            if address:
                self.central_neighbor_addresses[node_id] = address
        if self.central_neighbor_addresses and not self._logged_central_addresses:
            details = ", ".join(f"{node}@{addr}" for node, addr in self.central_neighbor_addresses.items())
            logger.info(f"Central neighbor addresses: {details}")
            self._logged_central_addresses = True

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
                )
        else:
            self.central_checker = None

    def gossip_ecm(self, cid: str, model_hash: str, round_num: int) -> None:
        """Gossip ECM to neighbor cluster bridge nodes."""
        if not self.is_bridge_node or not self.bridge_client:
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
        if self.is_aggregator:
            return 0
        if not self.ecm_buffer:
            return 0

        fresh_ecms = self.ecm_buffer.get_fresh_ecms()
        fresh_ecms = [ecm for ecm in fresh_ecms if not ecm.is_signal]
        if not fresh_ecms:
            logger.debug("No fresh ECMs to forward to aggregator")
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
                return len(fresh_ecms)
            else:
                logger.warning(f"Aggregator rejected ECMs: {response.message}")
                return 0
        except grpc.RpcError as e:
            logger.warning(f"Failed to forward ECMs to aggregator: {e}")
            return 0

    def run_secure_aggregation_round(self) -> List[float]:
        """Run one round of secure aggregation protocol."""
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

        # Round 0: Advertise keys
        logger.info("Round 0: Advertising keys")
        advert_msg = client.advertise_keys()

        # Retry logic for initial aggregator connection.
        max_retries = 30
        retry_delay = 1
        response = None
        for attempt in range(max_retries):
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
                if attempt < max_retries - 1:
                    logger.warning(f"Aggregator connection attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise

        if response is None:
            raise RuntimeError("Failed to connect to aggregator")

        if not response.accepted:
            raise RuntimeError(f"Round 0 failed: {response.message}")

        # Wait for ALL clique members to advertise (not just threshold).
        expected_participants = len(self.clique_members)
        while len(response.all_keys) < expected_participants:
            time.sleep(1)
            response = stub.Round0AdvertiseKeys(
                secureagg_pb2.KeyAdvertisement(
                    node_id=self.node_id,
                    c_public_key=advert_msg.c_public,
                    s_public_key=advert_msg.s_public,
                    signature=advert_msg.signature,
                ),
                timeout=30
            )

        logger.info(f"Round 0 complete: received {len(response.all_keys)} participants")

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

        # Round 1: Share keys (simplified - just send empty shares)
        logger.info("Round 1: Sharing keys")
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
            time.sleep(1)
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

        # Round 2: Send masked input
        logger.info("Round 2: Sending masked model")
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
            time.sleep(1)
            response2 = stub.Round2MaskedInput(
                secureagg_pb2.MaskedInputMessage(node_id=self.node_id, masked_vector=masked_bytes),
                timeout=30,
            )

        logger.info(f"Round 2 complete: {len(response2.survivors)} survivors")

        # Round 3: Consistency check
        logger.info("Round 3: Consistency check")
        survivor_sig = client.sign_survivor_list(response2.survivors)
        response3 = stub.Round3ConsistencyCheck(
            secureagg_pb2.ConsistencySignature(node_id=self.node_id, signature=survivor_sig.signature),
            timeout=30,
        )

        # Round 4: Unmask (simplified - send empty shares)
        logger.info("Round 4: Unmasking")
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
            time.sleep(1)
            response4 = stub.Round4Unmask(
                secureagg_pb2.UnmaskShares(
                    node_id=self.node_id,
                    dropout_s_shares={k: _encode_share(x, s) for k, (x, s) in unmask_payload.s_shares_for_dropouts.items()},
                    survivor_b_shares={k: _encode_share(x, b) for k, (x, b) in unmask_payload.b_shares_for_survivors.items()},
                ),
                timeout=30,
            )

        logger.info("Round 4 complete: aggregation done")

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
        max_rounds = self.convergence_config.max_rounds

        # Initialize convergence tracker
        self.convergence_tracker = ConvergenceTracker(
            self.convergence_config, f"cluster_{self.clique_id}"
        )
        self.convergence_tracker.set_signal_sender(self._handle_convergence_signal)
        self._refresh_central_metadata()

        logger.info(
            f"Starting convergence-driven training (max_rounds={max_rounds}, "
            f"tol_abs={self.convergence_config.tol_abs}, patience={self.convergence_config.patience})"
        )

        should_stop = False
        stop_reason = ""

        while self.current_round < max_rounds and not should_stop:
            round_idx = self.current_round
            logger.info(f"\n{'='*60}")
            logger.info(f"Round {round_idx + 1}/{max_rounds}")
            logger.info(f"{'='*60}")
            self._process_incoming_signals()
            self._check_global_convergence_signal()

            # Phase 1: Local training
            logger.info("Phase 1: Local training")
            self.train_local(local_epochs)

            acc_before = self.evaluate()
            logger.info(f"Accuracy before aggregation: {acc_before:.4f}")

            # Aggregator election
            self.aggregator_id = self.elect_aggregator(round_idx)
            self.aggregator_address = self.participant_map[self.aggregator_id]
            self.is_aggregator = (self.aggregator_id == self.node_id)

            if self.is_aggregator:
                logger.info(f"*** This node is the AGGREGATOR for round {round_idx} ***")
                self.start_aggregator_server()

            logger.info(f"Waiting for aggregator {self.aggregator_id} to be ready...")
            time.sleep(5)

            cid: Optional[str] = None
            model_hash: Optional[str] = None
            model_data_id: Optional[str] = None
            try:
                # Phase 2: Secure aggregation
                logger.info("Phase 2: Secure aggregation")
                aggregated_weights = self.run_secure_aggregation_round()

                # Phase 2.5: Bridge nodes forward ECMs to aggregator
                if self.inter_cluster_enabled and self.is_bridge_node and not self.is_aggregator:
                    forwarded = self.forward_ecms_to_aggregator()
                    if forwarded > 0:
                        logger.info(f"Phase 2.5: Forwarded {forwarded} ECMs to aggregator")

                if aggregated_weights:
                    logger.info("Phase 3: Updating model with aggregated weights")
                    dequantized = dequantize_vector([int(w) for w in aggregated_weights], self.scale)
                    load_params(self.model, dequantized)

                    # Phase 4: Inter-cluster merge (aggregator only)
                    if self.is_aggregator and self.inter_cluster_enabled and self.inter_cluster_aggregator:
                        # Wait briefly for ECMs from bridge nodes to arrive
                        time.sleep(2)
                        logger.info("Phase 4: Inter-cluster merge")
                        intra_model = np.array(dequantized, dtype=np.float32)

                        if self.ecm_buffer:
                            for ecm in self.ecm_buffer.get_fresh_ecms():
                                self.inter_cluster_aggregator.receive_ecms(self.node_id, [ecm])
                                # Update neighbor convergence status from ECM
                                if hasattr(ecm, "cluster_converged"):
                                    self.convergence_tracker.receive_neighbor_convergence(
                                        ecm.source_cluster, ecm.cluster_converged
                                    )

                        merged_data = self.inter_cluster_aggregator.process_round(
                            intra_model, round_idx
                        )
                        merged_model, cid, model_hash = merged_data
                        model_data_id = getattr(self.inter_cluster_aggregator, "last_data_id", None)

                        # Update convergence state with merged model
                        conv_state = self.convergence_tracker.update(merged_model)
                        should_stop = conv_state.should_stop
                        stop_reason = conv_state.stop_reason

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

                            response_cid = model_response.model_cid or ""
                            response_hash = model_response.model_hash or ""
                            if self.is_bridge_node and response_cid and response_hash:
                                # Remember reference for ECM gossip even if fetch fails locally
                                cid = response_cid
                                model_hash = response_hash

                            cluster_anchor_id = f"cluster_{self.clique_id}"
                            response_data_id = getattr(model_response, "model_data_id", "")
                            fetched_from_chain = False
                            if response_data_id and self.blockchain:
                                self.blockchain.remember_anchor(
                                    cluster_anchor_id,
                                    round_idx,
                                    response_data_id,
                                    model_response.model_cid or None,
                                    model_response.model_hash or None,
                                )
                                anchor = self.blockchain.get_anchor(cluster_anchor_id, round_idx)
                                if anchor and self.ipfs:
                                    cid_from_chain, expected_hash = anchor
                                    if self.is_bridge_node:
                                        cid = cid_from_chain
                                        model_hash = expected_hash
                                    logger.info(
                                        "Fetching merged model via blockchain anchor: %s",
                                        cid_from_chain[:16],
                                    )
                                    merged_from_ipfs = self.ipfs.get(cid_from_chain)
                                    if merged_from_ipfs is not None and verify_model_hash(
                                        merged_from_ipfs, expected_hash
                                    ):
                                        load_params(self.model, merged_from_ipfs.tolist())
                                        fetched_from_chain = True
                                    else:
                                        logger.warning(
                                            "Failed to verify model fetched from IPFS via blockchain; "
                                            "will still advertise anchor reference"
                                        )

                            if not fetched_from_chain and model_response.model_cid and self.ipfs:
                                logger.info(
                                    f"Fetching merged model from IPFS fallback: {model_response.model_cid[:16]}..."
                                )
                                merged_from_ipfs = self.ipfs.get(model_response.model_cid)
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

                acc_after = self.evaluate()
                logger.info(f"Accuracy after aggregation: {acc_after:.4f}")
                logger.info(f"Improvement: {acc_after - acc_before:+.4f}")

            except Exception as e:
                logger.error(f"Secure aggregation failed: {e}", exc_info=True)

            finally:
                if self.is_aggregator:
                    time.sleep(2)
                    self.stop_aggregator_server()

            # Phase 5: ECM gossip with convergence status (bridge nodes only)
            if cid and model_hash and self.is_bridge_node:
                logger.info("Phase 5: ECM gossip to neighbor clusters")
                cluster_converged = self.convergence_tracker.state.cluster_converged if self.convergence_tracker else False
                delta_norm = self.convergence_tracker.state.delta_norm if self.convergence_tracker else 0.0
                self.gossip_ecm_with_convergence(cid, model_hash, round_idx, cluster_converged, delta_norm)
                self._dispatch_pending_convergence_signal(round_idx)

            if should_stop:
                logger.info(f"Stopping training: {stop_reason}")
                self._check_global_convergence_signal()
                break

            logger.info(f"Round {round_idx} complete. Waiting before next round...")
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
                    logger.info(
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
        if not self.is_bridge_node or not self.bridge_client:
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

    def _handle_convergence_signal(self, signal: ConvergenceSignal) -> None:
        """Callback from ConvergenceTracker when local convergence state changes."""
        self._pending_convergence_signal = signal
        if signal.converged and self.convergence_config.signal_timeout > 0:
            time.sleep(self.convergence_config.signal_timeout)
        self._dispatch_pending_convergence_signal(signal.round_idx)

    def _dispatch_pending_convergence_signal(self, round_idx: int) -> None:
        if (
            not self._pending_convergence_signal
            or not self.is_bridge_node
            or not self.bridge_client
        ):
            return
        if not self.central_neighbor_addresses:
            return
        success = False
        for checker_id in self.central_neighbor_addresses.keys():
            if self._send_signal_to_checker(self._pending_convergence_signal, checker_id):
                success = True
        if success:
            logger.info(
                f"Sent convergence signal (converged={self._pending_convergence_signal.converged}) "
                f"to {len(self.central_neighbor_addresses)} central neighbors"
            )
            self._pending_convergence_signal = None

    def _send_signal_to_checker(self, signal: ConvergenceSignal, checker_id: str) -> bool:
        address = self.central_neighbor_addresses.get(checker_id)
        if not address:
            return False
        cluster_converged = signal.converged
        delta_norm = self.convergence_tracker.state.delta_norm if self.convergence_tracker else 0.0
        cid = f"signal::{signal.cluster_id}::{signal.round_idx}"
        try:
            return self.bridge_client.send_ecm_with_convergence(
                address,
                f"cluster_{self.clique_id}",
                signal.round_idx,
                cid=cid,
                model_hash="signal",
                cluster_converged=cluster_converged,
                cluster_delta_norm=delta_norm,
            )
        except Exception as exc:
            logger.warning(f"Failed to dispatch convergence signal to {checker_id}: {exc}")
            return False

    def _process_incoming_signals(self) -> None:
        if not self.central_checker or not self.ecm_buffer:
            return
        signals = self.ecm_buffer.pop_signal_ecms()
        for ecm in signals:
            if not ecm.source_cluster or ecm.round_idx < 0:
                continue
            self.central_checker.record_signal(ecm.source_cluster, ecm.round_idx, ecm.cluster_converged)

    def _check_global_convergence_signal(self) -> None:
        if not self.convergence_tracker:
            return
        round_idx = fetch_global_convergence_round(self.blockchain)
        if round_idx is not None:
            self.convergence_tracker.receive_global_convergence(round_idx)

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

        if inter_edges:
            self.setup_bridge_node(inter_edges)

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
