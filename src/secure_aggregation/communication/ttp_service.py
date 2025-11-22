"""Trusted Third Party (TTP) service for key distribution."""

import logging
from concurrent import futures
from typing import Dict

import grpc
from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.crypto.sign import SigningKeyPair, generate_signing_keypair
from secure_aggregation.utils import get_logger

logger = get_logger("ttp_service")


class TTPServicer(secureagg_pb2_grpc.TTPServiceServicer):
    """TTP service implementation for distributing signing keys."""

    def __init__(self) -> None:
        self.registered_nodes: Dict[str, tuple[bytes, bytes, str]] = {}  # node_id -> (sk, pk, address)
        logger.info("TTP service initialized")

    def RegisterNode(self, request: secureagg_pb2.RegisterRequest, context) -> secureagg_pb2.RegisterResponse:
        """Register a node and distribute signing keypair."""
        node_id = request.node_id
        address = request.address

        if node_id in self.registered_nodes:
            sk, pk, _ = self.registered_nodes[node_id]
            logger.info(f"Node {node_id} already registered, returning existing keys")
            return secureagg_pb2.RegisterResponse(
                signing_private_key=sk, signing_public_key=pk, success=True, message="Already registered"
            )

        keypair = generate_signing_keypair()
        self.registered_nodes[node_id] = (keypair.private_key, keypair.public_key, address)

        logger.info(f"Registered node {node_id} at {address}")
        return secureagg_pb2.RegisterResponse(
            signing_private_key=keypair.private_key,
            signing_public_key=keypair.public_key,
            success=True,
            message="Registration successful",
        )

    def GetParticipants(self, request: secureagg_pb2.ParticipantsRequest, context) -> secureagg_pb2.ParticipantsResponse:
        """Return list of all registered participants."""
        participants = [
            secureagg_pb2.NodeInfo(node_id=node_id, address=addr, signing_public_key=pk)
            for node_id, (_, pk, addr) in self.registered_nodes.items()
        ]
        logger.info(f"Returning {len(participants)} registered participants")
        return secureagg_pb2.ParticipantsResponse(participants=participants)


def serve(port: int = 50051) -> None:
    """Start the TTP gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    secureagg_pb2_grpc.add_TTPServiceServicer_to_server(TTPServicer(), server)
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
