"""
Bridge service for inter-cluster ECM gossip.

Bridge nodes use this service to:
1. Send ECMs to neighbor cluster bridge nodes
2. Receive ECMs from neighbor clusters
"""

from concurrent import futures
from typing import Dict, List, Optional

import grpc

from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.node import ECM, ECMBuffer
from secure_aggregation.utils import get_logger

logger = get_logger("bridge_service")


class BridgeServicer(secureagg_pb2_grpc.BridgeServiceServicer):
    """gRPC servicer for receiving ECMs from neighbor clusters."""

    def __init__(self, node_id: str, ecm_buffer: ECMBuffer) -> None:
        self.node_id = node_id
        self.ecm_buffer = ecm_buffer
        logger.info(f"BridgeServicer initialized for node {node_id}")

    def SubmitECMs(
        self,
        request: secureagg_pb2.ECMSubmitRequest,
        context,
    ) -> secureagg_pb2.ECMSubmitResponse:
        """Receive ECMs submitted by this node to be forwarded to aggregator."""
        for ecm_msg in request.ecms:
            ecm = ECM(
                cid=ecm_msg.cid,
                hash=ecm_msg.hash,
                source_cluster=ecm_msg.source_cluster,
            )
            self.ecm_buffer.add(ecm)
            logger.debug(f"Received ECM submission: cid={ecm.cid[:8]}...")

        return secureagg_pb2.ECMSubmitResponse(
            accepted=True,
            message=f"Received {len(request.ecms)} ECMs",
        )

    def ReceiveECM(
        self,
        request: secureagg_pb2.ECMBroadcast,
        context,
    ) -> secureagg_pb2.ECMSubmitResponse:
        """Receive ECM broadcast from neighbor cluster with convergence status."""
        cid = request.cid or f"signal::{request.cluster_id}::{request.round}"
        if request.convergence_data_id and not request.cid:
            cid = f"signal::convergence::{request.convergence_data_id}"
        is_signal = cid.startswith("signal::")
        ecm = ECM(
            cid=cid,
            hash=request.hash,
            source_cluster=request.cluster_id,
            cluster_converged=request.cluster_converged,
            cluster_delta_norm=request.cluster_delta_norm,
            round_idx=request.round,
            is_signal=is_signal,
            convergence_data_id=request.convergence_data_id or None,
        )
        self.ecm_buffer.add(ecm)
        logger.info(
            f"Received ECM from cluster {request.cluster_id} "
            f"round {request.round}: cid={request.cid[:8]}... "
            f"(converged={request.cluster_converged}, data_id={request.convergence_data_id or 'N/A'})"
        )

        return secureagg_pb2.ECMSubmitResponse(
            accepted=True,
            message=f"Received ECM from cluster {request.cluster_id}",
        )


class BridgeClient:
    """Client for sending ECMs to neighbor cluster bridge nodes."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._channels: Dict[str, grpc.Channel] = {}
        self._stubs: Dict[str, secureagg_pb2_grpc.BridgeServiceStub] = {}

    def _get_stub(self, address: str) -> secureagg_pb2_grpc.BridgeServiceStub:
        """Get or create stub for address."""
        if address not in self._stubs:
            channel = grpc.insecure_channel(address)
            self._channels[address] = channel
            self._stubs[address] = secureagg_pb2_grpc.BridgeServiceStub(channel)
        return self._stubs[address]

    def send_ecm(
        self,
        neighbor_address: str,
        cluster_id: str,
        round_num: int,
        cid: str,
        model_hash: str,
    ) -> bool:
        """
        Send ECM to a neighbor cluster bridge node.

        Args:
            neighbor_address: Address of neighbor bridge node.
            cluster_id: This cluster's ID.
            round_num: Current training round.
            cid: IPFS CID of published model.
            model_hash: SHA256 hash for verification.

        Returns:
            True if ECM was accepted.
        """
        try:
            stub = self._get_stub(neighbor_address)
            request = secureagg_pb2.ECMBroadcast(
                cluster_id=cluster_id,
                round=round_num,
                cid=cid,
                hash=model_hash,
            )
            response = stub.ReceiveECM(request, timeout=10)
            if response.accepted:
                logger.debug(f"ECM sent to {neighbor_address}")
            return response.accepted
        except grpc.RpcError as e:
            logger.warning(f"Failed to send ECM to {neighbor_address}: {e}")
            return False

    def broadcast_ecm(
        self,
        neighbor_addresses: List[str],
        cluster_id: str,
        round_num: int,
        cid: str,
        model_hash: str,
    ) -> int:
        """
        Broadcast ECM to all neighbor cluster bridge nodes.

        Returns:
            Number of neighbors that accepted the ECM.
        """
        accepted = 0
        for addr in neighbor_addresses:
            if self.send_ecm(addr, cluster_id, round_num, cid, model_hash):
                accepted += 1

        logger.info(
            f"Broadcast ECM to {accepted}/{len(neighbor_addresses)} neighbors "
            f"(cluster={cluster_id}, round={round_num})"
        )
        return accepted

    def send_ecm_with_convergence(
        self,
        neighbor_address: str,
        cluster_id: str,
        round_num: int,
        cid: str,
        model_hash: str,
        cluster_converged: bool,
        cluster_delta_norm: float,
        convergence_data_id: Optional[str] = None,
    ) -> bool:
        """Send ECM with convergence status to a neighbor cluster bridge node."""
        try:
            stub = self._get_stub(neighbor_address)
            request = secureagg_pb2.ECMBroadcast(
                cluster_id=cluster_id,
                round=round_num,
                cid=cid,
                hash=model_hash,
                cluster_converged=cluster_converged,
                cluster_delta_norm=cluster_delta_norm,
                convergence_data_id=convergence_data_id or "",
            )
            response = stub.ReceiveECM(request, timeout=10)
            if response.accepted:
                logger.debug(f"ECM with convergence sent to {neighbor_address}")
            return response.accepted
        except grpc.RpcError as e:
            logger.warning(f"Failed to send ECM to {neighbor_address}: {e}")
            return False

    def broadcast_ecm_with_convergence(
        self,
        neighbor_addresses: List[str],
        cluster_id: str,
        round_num: int,
        cid: str,
        model_hash: str,
        cluster_converged: bool,
        cluster_delta_norm: float,
        convergence_data_id: Optional[str] = None,
    ) -> int:
        """
        Broadcast ECM with convergence status to all neighbor cluster bridge nodes.

        Returns:
            Number of neighbors that accepted the ECM.
        """
        accepted = 0
        for addr in neighbor_addresses:
            if self.send_ecm_with_convergence(
                addr,
                cluster_id,
                round_num,
                cid,
                model_hash,
                cluster_converged,
                cluster_delta_norm,
                convergence_data_id=convergence_data_id,
            ):
                accepted += 1

        logger.info(
            f"Broadcast ECM with convergence to {accepted}/{len(neighbor_addresses)} neighbors "
            f"(cluster={cluster_id}, round={round_num}, converged={cluster_converged})"
        )
        return accepted

    def close(self) -> None:
        """Close all channels."""
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()
        self._stubs.clear()


def serve_bridge(
    node_id: str,
    port: int,
    ecm_buffer: ECMBuffer,
) -> grpc.Server:
    """Start the bridge gRPC server."""
    servicer = BridgeServicer(node_id, ecm_buffer)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    secureagg_pb2_grpc.add_BridgeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Bridge server started on port {port}")
    return server
