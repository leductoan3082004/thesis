"""
Bridge service for inter-cluster ECM gossip.

Bridge nodes use this service to:
1. Send ECMs to neighbor cluster bridge nodes
2. Receive ECMs from neighbor clusters
"""

from concurrent import futures
from typing import Any, Callable, Dict, Iterable, List, Optional

import grpc

from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.node import ECM, ECMBuffer
from secure_aggregation.utils import get_logger
from secure_aggregation.utils.traffic_recorder import TrafficRecorder

logger = get_logger("bridge_service")
STATE_CHANNEL_PREFIX = "state::"


class BridgeServicer(secureagg_pb2_grpc.BridgeServiceServicer):
    """gRPC servicer for receiving ECMs from neighbor clusters."""

    def __init__(
        self,
        node_id: str,
        ecm_buffer: ECMBuffer,
        ecm_hooks: Optional[Iterable[Callable[[ECM], None]]] = None,
    ) -> None:
        self.node_id = node_id
        self.ecm_buffer = ecm_buffer
        self._ecm_hooks = list(ecm_hooks or [])
        logger.info(f"BridgeServicer initialized for node {node_id}")

    def _emit_hooks(self, ecm: ECM) -> None:
        for hook in self._ecm_hooks:
            try:
                hook(ecm)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Bridge ECM hook raised error: %s", exc)

    def _record_exchange(
        self,
        *,
        cmd: str,
        package_type: str,
        round_idx: Optional[int],
        source: str,
        request: Optional[Any] = None,
        response: Optional[Any] = None,
        additional_info: str = "",
    ) -> None:
        recorder = TrafficRecorder.get_instance()
        if not recorder:
            return
        req_size = request.ByteSize() if request is not None else 0
        resp_size = response.ByteSize() if response is not None else 0
        info = additional_info
        if source:
            info = f"{additional_info},peer={source}" if additional_info else f"peer={source}"
        recorder.record_bytes_exchange(
            cmd=cmd,
            package_type=package_type,
            round_idx=round_idx,
            source=self.node_id,
            destination=source,
            request_size=req_size,
            response_size=resp_size,
            additional_info=info,
        )

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
            self._emit_hooks(ecm)
            logger.debug(f"Received ECM submission: cid={ecm.cid[:8]}...")

        received_count = len(request.ecms)
        response = secureagg_pb2.ECMSubmitResponse(
            accepted=True,
            message=f"Received {received_count} ECMs",
        )
        self._record_exchange(
            cmd="Bridge.SubmitECMs",
            package_type="ecm_submission",
            round_idx=None,
            source=request.node_id or context.peer(),
            request=request,
            response=response,
            additional_info=f"ecms={received_count}",
        )
        return response

    def ReceiveECM(
        self,
        request: secureagg_pb2.ECMBroadcast,
        context,
    ) -> secureagg_pb2.ECMSubmitResponse:
        """Receive ECM broadcast from neighbor cluster."""
        cid = request.cid or f"signal::{request.cluster_id}::{request.round}"
        is_state_channel = request.cluster_id.startswith(STATE_CHANNEL_PREFIX)
        if request.convergence_data_id and not request.cid and not is_state_channel:
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
        self._emit_hooks(ecm)
        if not is_signal:
            logger.info(
                f"Received ECM from cluster {request.cluster_id} "
                f"round {request.round}: cid={request.cid[:8]}..."
            )

        response = secureagg_pb2.ECMSubmitResponse(
            accepted=True,
            message=f"Received ECM from cluster {request.cluster_id}",
        )
        package = "ecm_gossip" if not is_state_channel else "scope_fanout"
        self._record_exchange(
            cmd="Bridge.ReceiveECM",
            package_type=package,
            round_idx=request.round,
            source=request.cluster_id or context.peer(),
            request=request,
            response=response,
            additional_info=f"cid={cid}",
        )
        return response


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
        *,
        destination_node_id: Optional[str] = None,
        package_type: str = "ecm_gossip",
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
            response = None
            rpc_error: Optional[grpc.RpcError] = None
            info = f"cid={cid},channel={cluster_id}"
            try:
                response = stub.ReceiveECM(request, timeout=10)
            except grpc.RpcError as exc:
                rpc_error = exc
            finally:
                recorder = TrafficRecorder.get_instance()
                if recorder:
                    extra = info
                    if rpc_error is not None:
                        status = rpc_error.code().name if hasattr(rpc_error, "code") else "error"
                        extra = f"{info},error={status}"
                    elif response is not None:
                        extra = f"{info},accepted={response.accepted}"
                    recorder.record_bytes_exchange(
                        cmd="Bridge.ReceiveECM",
                        package_type=package_type,
                        round_idx=round_num,
                        source=self.node_id,
                        destination=destination_node_id or neighbor_address,
                        request_size=request.ByteSize(),
                        response_size=response.ByteSize() if response else 0,
                        additional_info=extra,
                    )
            if rpc_error is not None:
                raise rpc_error
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
        *,
        neighbor_ids: Optional[List[str]] = None,
        package_type: str = "ecm_gossip",
    ) -> int:
        """
        Broadcast ECM to all neighbor cluster bridge nodes.

        Returns:
            Number of neighbors that accepted the ECM.
        """
        accepted = 0
        for idx, addr in enumerate(neighbor_addresses):
            dest_id = None
            if neighbor_ids and idx < len(neighbor_ids):
                dest_id = neighbor_ids[idx]
            if self.send_ecm(
                addr,
                cluster_id,
                round_num,
                cid,
                model_hash,
                destination_node_id=dest_id,
                package_type=package_type,
            ):
                accepted += 1

        if cluster_id.startswith(STATE_CHANNEL_PREFIX):
            logger.info(
                f"Broadcast state artifact to {accepted}/{len(neighbor_addresses)} neighbors "
                f"(state={cluster_id}, round={round_num})"
            )
        else:
            logger.info(
                f"Broadcast ECM to {accepted}/{len(neighbor_addresses)} neighbors "
                f"(cluster={cluster_id}, round={round_num})"
            )
        return accepted

    def send_ecm_with_metadata(
        self,
        neighbor_address: str,
        cluster_id: str,
        round_num: int,
        cid: str,
        model_hash: str,
        metadata: Optional[str] = None,
        *,
        destination_node_id: Optional[str] = None,
        package_type: str = "scope_fanout",
    ) -> bool:
        """Send ECM with auxiliary metadata (used for state artifacts)."""
        try:
            stub = self._get_stub(neighbor_address)
            request = secureagg_pb2.ECMBroadcast(
                cluster_id=cluster_id,
                round=round_num,
                cid=cid,
                hash=model_hash,
                convergence_data_id=metadata or "",
            )
            response = None
            rpc_error: Optional[grpc.RpcError] = None
            info = f"cid={cid},channel={cluster_id},metadata={metadata or ''}"
            try:
                response = stub.ReceiveECM(request, timeout=10)
            except grpc.RpcError as exc:
                rpc_error = exc
            finally:
                recorder = TrafficRecorder.get_instance()
                if recorder:
                    extra = info
                    if rpc_error is not None:
                        status = rpc_error.code().name if hasattr(rpc_error, "code") else "error"
                        extra = f"{info},error={status}"
                    elif response is not None:
                        extra = f"{info},accepted={response.accepted}"
                    recorder.record_bytes_exchange(
                        cmd="Bridge.ReceiveECM",
                        package_type=package_type,
                        round_idx=round_num,
                        source=self.node_id,
                        destination=destination_node_id or neighbor_address,
                        request_size=request.ByteSize(),
                        response_size=response.ByteSize() if response else 0,
                        additional_info=extra,
                    )
            if rpc_error is not None:
                raise rpc_error
            if response.accepted:
                logger.debug(f"ECM with metadata sent to {neighbor_address}")
            return response.accepted
        except grpc.RpcError as e:
            logger.warning(f"Failed to send ECM to {neighbor_address}: {e}")
            return False

    def broadcast_ecm_with_metadata(
        self,
        neighbor_addresses: List[str],
        cluster_id: str,
        round_num: int,
        cid: str,
        model_hash: str,
        metadata: Optional[str] = None,
        *,
        neighbor_ids: Optional[List[str]] = None,
        package_type: str = "scope_fanout",
    ) -> int:
        """Broadcast ECM with optional metadata to all neighbor bridge nodes."""
        accepted = 0
        for idx, addr in enumerate(neighbor_addresses):
            dest_id = None
            if neighbor_ids and idx < len(neighbor_ids):
                dest_id = neighbor_ids[idx]
            if self.send_ecm_with_metadata(
                addr,
                cluster_id,
                round_num,
                cid,
                model_hash,
                metadata=metadata,
                destination_node_id=dest_id,
                package_type=package_type,
            ):
                accepted += 1
        return accepted

    def wait_for_ready(self, address: str, timeout: float = 2.0) -> bool:
        """Return True if the target bridge endpoint responds within timeout."""
        try:
            stub = self._get_stub(address)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create bridge stub for %s: %s", address, exc)
            return False
        channel = self._channels.get(address)
        if channel is None:
            return False
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
            return True
        except grpc.FutureTimeoutError:
            logger.warning("Bridge peer %s not ready after %.1fs", address, timeout)
            return False

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
    ecm_hooks: Optional[Iterable[Callable[[ECM], None]]] = None,
) -> grpc.Server:
    """Start the bridge gRPC server."""
    servicer = BridgeServicer(node_id, ecm_buffer, ecm_hooks=ecm_hooks)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    secureagg_pb2_grpc.add_BridgeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Bridge server started on port {port}")
    return server
