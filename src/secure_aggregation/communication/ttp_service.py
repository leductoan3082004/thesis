"""Trusted Third Party (TTP) service for key distribution and topology management."""

import logging
from concurrent import futures
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set

import grpc
from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.crypto.sign import SigningKeyPair, generate_signing_keypair
from secure_aggregation.data import dirichlet_partition
from secure_aggregation.topology import (
    build_full_topology,
    compute_clique_threshold,
    compute_node_labels_from_partition,
    find_node_clique,
)
from secure_aggregation.utils import get_logger

logger = get_logger("ttp_service")


@dataclass
class TopologyConfig:
    """Configuration for D-Cliques topology construction."""

    num_clients: int
    clique_size: int
    alpha: float = 0.5
    seed: int = 42
    inter_clique_edges: str = "small_world"
    topology_iterations: int = 1000
    small_world_c: int = 2


@dataclass
class TopologyState:
    """Stores computed topology and data assignments."""

    cliques: List[Set[str]] = field(default_factory=list)
    node_to_clique: Dict[str, int] = field(default_factory=dict)
    partition: Dict[str, List[int]] = field(default_factory=dict)
    thresholds: Dict[int, int] = field(default_factory=dict)


class TTPServicer(secureagg_pb2_grpc.TTPServiceServicer):
    """TTP service implementation for distributing signing keys and topology info."""

    def __init__(
        self,
        topology_config: Optional[TopologyConfig] = None,
        labels: Optional[Mapping[int, int]] = None,
    ) -> None:
        self.registered_nodes: Dict[str, tuple[bytes, bytes, str]] = {}  # node_id -> (sk, pk, address)
        self.topology: TopologyState = TopologyState()

        if topology_config and labels:
            self._build_topology(topology_config, labels)
        logger.info("TTP service initialized")

    def _build_topology(self, config: TopologyConfig, labels: Mapping[int, int]) -> None:
        """Build D-Cliques topology from data labels at startup."""
        dataset_indices = list(labels.keys())

        partition = dirichlet_partition(
            dataset=dataset_indices,
            labels=labels,
            num_clients=config.num_clients,
            alpha=config.alpha,
            seed=config.seed,
        )

        node_labels = compute_node_labels_from_partition(partition, labels)

        cliques, intra_edges, inter_edges, edge_counts = build_full_topology(
            node_labels=node_labels,
            clique_size=config.clique_size,
            iterations=config.topology_iterations,
            edge_mode=config.inter_clique_edges,
            small_world_c=config.small_world_c,
            seed=config.seed,
        )

        node_to_clique: Dict[str, int] = {}
        for clique_idx, clique_members in enumerate(cliques):
            for node_id in clique_members:
                node_to_clique[node_id] = clique_idx

        thresholds: Dict[int, int] = {}
        for clique_idx, clique_members in enumerate(cliques):
            thresholds[clique_idx] = compute_clique_threshold(len(clique_members))

        self.topology = TopologyState(
            cliques=cliques,
            node_to_clique=node_to_clique,
            partition=partition,
            thresholds=thresholds,
        )

        logger.info(
            f"Topology built: {len(cliques)} cliques, "
            f"clique sizes: {[len(c) for c in cliques]}, "
            f"thresholds: {list(thresholds.values())}"
        )

    def RegisterNode(self, request: secureagg_pb2.RegisterRequest, context) -> secureagg_pb2.RegisterResponse:
        """Register a node and distribute signing keypair with clique assignment."""
        node_id = request.node_id
        address = request.address

        clique_id = -1
        clique_members: List[str] = []
        clique_threshold = 0
        data_indices: List[int] = []

        if self.topology.cliques:
            if node_id in self.topology.node_to_clique:
                clique_id = self.topology.node_to_clique[node_id]
                clique_members = sorted(self.topology.cliques[clique_id])
                clique_threshold = self.topology.thresholds.get(clique_id, 0)
            if node_id in self.topology.partition:
                data_indices = self.topology.partition[node_id]

        if node_id in self.registered_nodes:
            sk, pk, _ = self.registered_nodes[node_id]
            logger.info(f"Node {node_id} already registered, returning existing keys")
            return secureagg_pb2.RegisterResponse(
                signing_private_key=sk,
                signing_public_key=pk,
                success=True,
                message="Already registered",
                clique_id=clique_id,
                clique_members=clique_members,
                clique_threshold=clique_threshold,
                data_indices=data_indices,
            )

        keypair = generate_signing_keypair()
        self.registered_nodes[node_id] = (keypair.private_key, keypair.public_key, address)

        logger.info(f"Registered node {node_id} at {address}, clique={clique_id}, threshold={clique_threshold}")
        return secureagg_pb2.RegisterResponse(
            signing_private_key=keypair.private_key,
            signing_public_key=keypair.public_key,
            success=True,
            message="Registration successful",
            clique_id=clique_id,
            clique_members=clique_members,
            clique_threshold=clique_threshold,
            data_indices=data_indices,
        )

    def GetParticipants(self, request: secureagg_pb2.ParticipantsRequest, context) -> secureagg_pb2.ParticipantsResponse:
        """Return list of all registered participants."""
        participants = [
            secureagg_pb2.NodeInfo(node_id=node_id, address=addr, signing_public_key=pk)
            for node_id, (_, pk, addr) in self.registered_nodes.items()
        ]
        logger.info(f"Returning {len(participants)} registered participants")
        return secureagg_pb2.ParticipantsResponse(participants=participants)


def serve(
    port: int = 50051,
    topology_config: Optional[TopologyConfig] = None,
    labels: Optional[Mapping[int, int]] = None,
) -> None:
    """Start the TTP gRPC server with optional topology configuration."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = TTPServicer(topology_config=topology_config, labels=labels)
    secureagg_pb2_grpc.add_TTPServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"TTP server started on port {port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("TTP server shutting down")
        server.stop(0)


if __name__ == "__main__":
    from secure_aggregation.utils import configure_logging

    configure_logging()
    serve()
