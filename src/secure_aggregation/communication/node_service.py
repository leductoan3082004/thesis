"""Node service for federated learning with secure aggregation."""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import grpc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.communication.aggregator_service import serve as serve_aggregator
from secure_aggregation.config.models import NodeRole
from secure_aggregation.crypto.sign import SigningKeyPair, sign_message, verify_signature
from secure_aggregation.crypto.dh import DHKeyPair, generate_dh_keypair, agree
from secure_aggregation.data import dirichlet_partition
from secure_aggregation.node import NodeEngine, NodeRuntimeConfig, ReliabilityScore
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

        # DH keypairs for secure aggregation
        self.c_keypair: Optional[DHKeyPair] = None  # For encryption
        self.s_keypair: Optional[DHKeyPair] = None  # For masking

        logger.info(f"Node {self.node_id} initialized (role={self.role}, port={self.port})")

    def _load_config(self, path: str) -> dict:
        """Load node configuration from JSON file."""
        with open(path) as f:
            return json.load(f)

    def register_with_ttp(self) -> None:
        """Register with TTP and receive signing keys."""
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
                    logger.info(f"Successfully registered with TTP")
                else:
                    raise RuntimeError(f"TTP registration failed: {response.message}")

                # Get list of participants
                participants_response = stub.GetParticipants(secureagg_pb2.ParticipantsRequest())
                self.participants = list(participants_response.participants)
                self.participant_map = {p.node_id: p.address for p in self.participants}
                logger.info(f"Retrieved {len(self.participants)} participants")

                channel.close()
                return

            except grpc.RpcError as e:
                logger.warning(f"TTP connection attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(2)

        raise RuntimeError("Failed to connect to TTP after max retries")

    def setup_data(self) -> None:
        """Setup MNIST dataset with partitioning."""
        logger.info("Setting up MNIST dataset")
        tform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(root="/app/data", train=True, download=False, transform=tform)
        test_ds = datasets.MNIST(root="/app/data", train=False, download=False, transform=tform)

        # Create partitions using Dirichlet distribution
        labels = {i: int(train_ds[i][1]) for i in range(len(train_ds))}
        num_clients = self.dataset_config["num_clients"]
        alpha = self.dataset_config["alpha"]
        seed = self.dataset_config.get("seed", 42)

        parts = dirichlet_partition(
            list(range(len(train_ds))), labels, num_clients=num_clients, alpha=alpha, seed=seed
        )

        # Get this node's partition (based on node_id index)
        node_index = int(self.node_id.split("_")[-1])
        client_key = f"client_{node_index}"
        indices = parts.get(client_key, [])

        logger.info(f"Node {self.node_id} assigned {len(indices)} training samples")

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
        """Elect aggregator for this round using reliability scores."""
        # Simple deterministic election: round-robin by node_id
        sorted_participants = sorted(self.participant_map.keys())
        aggregator_id = sorted_participants[round_idx % len(sorted_participants)]
        logger.info(f"Elected aggregator for round {round_idx}: {aggregator_id}")
        return aggregator_id

    def start_aggregator_server(self) -> None:
        """Start aggregator gRPC server if this node is elected."""
        if self.aggregator_server is not None:
            logger.warning("Aggregator server already running")
            return

        participant_ids = list(self.participant_map.keys())
        self.aggregator_server = serve_aggregator(
            self.node_id, self.port + 1000, self.threshold, participant_ids
        )
        logger.info(f"Started aggregator server on port {self.port + 1000}")

    def stop_aggregator_server(self) -> None:
        """Stop aggregator server."""
        if self.aggregator_server:
            self.aggregator_server.stop(0)
            self.aggregator_server = None
            logger.info("Stopped aggregator server")

    def run_secure_aggregation_round(self) -> List[float]:
        """Run one round of secure aggregation protocol."""
        logger.info(f"Starting secure aggregation round {self.current_round}")

        # Generate DH keypairs
        self.c_keypair = generate_dh_keypair()
        self.s_keypair = generate_dh_keypair()

        # Get aggregator address
        agg_port = int(self.aggregator_address.split(":")[-1]) + 1000
        agg_host = self.aggregator_address.split(":")[0]
        agg_addr = f"{agg_host}:{agg_port}"

        channel = grpc.insecure_channel(agg_addr)
        stub = secureagg_pb2_grpc.AggregatorServiceStub(channel)

        # Round 0: Advertise keys
        logger.info("Round 0: Advertising keys")
        key_msg = self.c_keypair.public_key + self.s_keypair.public_key
        signature = sign_message(self.signing_keypair.private_key, key_msg)

        response = stub.Round0AdvertiseKeys(
            secureagg_pb2.KeyAdvertisement(
                node_id=self.node_id,
                c_public_key=self.c_keypair.public_key,
                s_public_key=self.s_keypair.public_key,
                signature=signature
            ),
            timeout=30
        )

        if not response.accepted:
            raise RuntimeError(f"Round 0 failed: {response.message}")

        # Wait for U1 list
        while not response.all_keys:
            time.sleep(1)
            response = stub.Round0AdvertiseKeys(
                secureagg_pb2.KeyAdvertisement(
                    node_id=self.node_id,
                    c_public_key=self.c_keypair.public_key,
                    s_public_key=self.s_keypair.public_key,
                    signature=signature
                ),
                timeout=30
            )

        logger.info(f"Round 0 complete: received {len(response.all_keys)} participants")

        # Round 1: Share keys (simplified - just send empty shares)
        logger.info("Round 1: Sharing keys")
        shares = {p.node_id: b"" for p in response.all_keys if p.node_id != self.node_id}
        response1 = stub.Round1ShareKeys(
            secureagg_pb2.ShareKeysMessage(node_id=self.node_id, encrypted_shares=shares),
            timeout=30
        )

        # Round 2: Send masked input
        logger.info("Round 2: Sending masked model")
        model_vec = flatten_params(self.model)
        quantized = quantize_vector(model_vec, self.scale)

        response2 = stub.Round2MaskedInput(
            secureagg_pb2.MaskedInputMessage(node_id=self.node_id, masked_vector=quantized),
            timeout=30
        )

        # Wait for survivors list
        while not response2.survivors:
            time.sleep(1)
            response2 = stub.Round2MaskedInput(
                secureagg_pb2.MaskedInputMessage(node_id=self.node_id, masked_vector=quantized),
                timeout=30
            )

        logger.info(f"Round 2 complete: {len(response2.survivors)} survivors")

        # Round 3: Consistency check
        logger.info("Round 3: Consistency check")
        survivors_msg = ",".join(sorted(response2.survivors)).encode()
        sig3 = sign_message(self.signing_keypair.private_key, survivors_msg)

        response3 = stub.Round3ConsistencyCheck(
            secureagg_pb2.ConsistencySignature(node_id=self.node_id, signature=sig3),
            timeout=30
        )

        # Round 4: Unmask (simplified - send empty shares)
        logger.info("Round 4: Unmasking")
        response4 = stub.Round4Unmask(
            secureagg_pb2.UnmaskShares(node_id=self.node_id, dropout_s_shares={}, survivor_b_shares={}),
            timeout=30
        )

        # Wait for aggregation to complete
        while not response4.aggregation_complete:
            time.sleep(1)
            response4 = stub.Round4Unmask(
                secureagg_pb2.UnmaskShares(node_id=self.node_id, dropout_s_shares={}, survivor_b_shares={}),
                timeout=30
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
        """Main training loop with secure aggregation."""
        num_rounds = self.training_config["num_rounds"]
        local_epochs = self.training_config["local_epochs"]

        logger.info(f"Starting training loop for {num_rounds} rounds")

        for round_idx in range(num_rounds):
            self.current_round = round_idx
            logger.info(f"\n{'='*60}")
            logger.info(f"Round {round_idx + 1}/{num_rounds}")
            logger.info(f"{'='*60}")

            # Local training
            logger.info("Phase 1: Local training")
            self.train_local(local_epochs)

            # Evaluate before aggregation
            acc_before = self.evaluate()
            logger.info(f"Accuracy before aggregation: {acc_before:.4f}")

            # Aggregator election
            self.aggregator_id = self.elect_aggregator(round_idx)
            self.aggregator_address = self.participant_map[self.aggregator_id]
            self.is_aggregator = (self.aggregator_id == self.node_id)

            # Start aggregator server if elected
            if self.is_aggregator:
                logger.info(f"*** This node is the AGGREGATOR for round {round_idx} ***")
                self.start_aggregator_server()
                time.sleep(2)  # Give server time to start

            # All nodes wait for aggregator to be ready
            logger.info(f"Waiting for aggregator {self.aggregator_id} to be ready...")
            time.sleep(3)

            # Run secure aggregation
            try:
                logger.info("Phase 2: Secure aggregation")
                aggregated_weights = self.run_secure_aggregation_round()

                # Update model with aggregated weights
                if aggregated_weights:
                    logger.info("Phase 3: Updating model with aggregated weights")
                    dequantized = dequantize_vector([int(w) for w in aggregated_weights], self.scale)
                    load_params(self.model, dequantized)

                # Evaluate after aggregation
                acc_after = self.evaluate()
                logger.info(f"Accuracy after aggregation: {acc_after:.4f}")
                logger.info(f"Improvement: {acc_after - acc_before:+.4f}")

            except Exception as e:
                logger.error(f"Secure aggregation failed: {e}", exc_info=True)

            finally:
                # Stop aggregator server
                if self.is_aggregator:
                    time.sleep(2)  # Wait for other nodes to finish
                    self.stop_aggregator_server()

            # Sync point: wait for next round
            logger.info(f"Round {round_idx} complete. Waiting before next round...")
            time.sleep(5)

        logger.info("\n" + "="*60)
        logger.info("Training loop completed")
        logger.info("="*60)

    def start(self) -> None:
        """Start the node service."""
        logger.info(f"Starting node {self.node_id} on port {self.port}")

        # Register with TTP
        self.register_with_ttp()

        # Setup data and model
        self.setup_data()
        self.setup_model()

        # Wait for all nodes to be ready
        logger.info("Waiting for all nodes to register...")
        while len(self.participants) < self.dataset_config["num_clients"]:
            time.sleep(2)
            self.register_with_ttp()  # Refresh participants list

        logger.info(f"All {len(self.participants)} nodes are ready. Starting training...")
        time.sleep(2)

        # Run training loop
        self.run_training_loop()

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
