"""Aggregator service that coordinates secure aggregation protocol."""

import logging
from concurrent import futures
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import grpc
from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.protocol import (
    AdvertiseMessage,
    MaskedInput,
    Round1Ciphertext as Round1CiphertextModel,
    SecureAggregationAggregator,
    SecureAggregationConfig,
    SurvivorSignature,
    UnmaskingShares as UnmaskingSharesModel,
)
from secure_aggregation.protocol.core import _bytes_to_int
from secure_aggregation.protocol.core import _int_to_bytes as int_to_bytes
from secure_aggregation.protocol.core import DH_PRIV_BYTES, SHARE_BYTES
from secure_aggregation.node import ECM, ECMBuffer
from secure_aggregation.utils import get_logger

logger = get_logger("aggregator_service")


def _decode_round1_ciphertexts(requests: Sequence[secureagg_pb2.Round1Ciphertext]) -> List[Round1CiphertextModel]:
    return [
        Round1CiphertextModel(
            sender_id=ct.sender_id,
            recipient_id=ct.recipient_id,
            iv=bytes(ct.iv),
            ciphertext=bytes(ct.ciphertext),
            tag=bytes(ct.tag),
        )
        for ct in requests
    ]


def _encode_round1_ciphertexts(ciphertexts: Sequence[Round1CiphertextModel]) -> List[secureagg_pb2.Round1Ciphertext]:
    return [
        secureagg_pb2.Round1Ciphertext(
            sender_id=ct.sender_id,
            recipient_id=ct.recipient_id,
            iv=ct.iv,
            ciphertext=ct.ciphertext,
            tag=ct.tag,
        )
        for ct in ciphertexts
    ]


def _encode_unmask_share(x: int, share: int) -> bytes:
    """Pack (x, share) into bytes for transport."""
    return int_to_bytes(x, 2) + int_to_bytes(share, SHARE_BYTES)


def _decode_unmask_share(data: bytes) -> Tuple[int, int]:
    """Unpack bytes into (x, share) tuple."""
    x = int.from_bytes(data[:2], "big")
    share = _bytes_to_int(data[2:])
    return x, share


class AggregatorServicer(secureagg_pb2_grpc.AggregatorServiceServicer):
    """Aggregator that coordinates the full 4-round secure aggregation protocol."""

    def __init__(
        self,
        node_id: str,
        threshold: int,
        participant_ids: List[str],
        signing_public_keys: Optional[Mapping[str, bytes]] = None,
        ecm_buffer: Optional[ECMBuffer] = None,
    ) -> None:
        self.node_id = node_id
        self.threshold = threshold
        self.participant_ids = participant_ids
        config = SecureAggregationConfig(participants=participant_ids, threshold=threshold)
        self.aggregator = SecureAggregationAggregator(config=config, signing_public_keys=signing_public_keys)
        self.aggregated_result: Optional[List[float]] = None
        self.current_round = 0
        self._adverts: Dict[str, AdvertiseMessage] = {}
        self._adverts_committed = False
        self._round3_signatures: Dict[str, bytes] = {}
        self._round4_payloads: List[UnmaskingSharesModel] = []

        # ECM buffer for receiving ECMs from bridge nodes
        self.ecm_buffer = ecm_buffer

        # Convergence state for global coordination
        self.merged_model_cid: Optional[str] = None
        self.merged_model_hash: Optional[str] = None
        self.merged_model_data_id: Optional[str] = None
        self.should_stop: bool = False
        self.stop_reason: str = ""
        self.delta_norm: float = 0.0
        self.cluster_converged: bool = False

        logger.info(
            f"Aggregator {node_id} initialized with threshold={threshold}, participants={len(participant_ids)}"
        )

    def set_convergence_state(
        self,
        model_cid: Optional[str],
        model_hash: Optional[str],
        model_data_id: Optional[str],
        should_stop: bool,
        stop_reason: str,
        delta_norm: float,
        cluster_converged: bool,
    ) -> None:
        """Store IPFS reference and convergence info for distribution to all nodes."""
        self.merged_model_cid = model_cid
        self.merged_model_hash = model_hash
        self.merged_model_data_id = model_data_id
        self.should_stop = should_stop
        self.stop_reason = stop_reason
        self.delta_norm = delta_norm
        self.cluster_converged = cluster_converged

    def _validate_participant(self, node_id: str) -> bool:
        return node_id in self.participant_ids

    def Round0AdvertiseKeys(self, request: secureagg_pb2.KeyAdvertisement, context) -> secureagg_pb2.KeyAdvertisementAck:
        """Collect DH public keys from participants (SAP-Round 0)."""
        node_id = request.node_id

        if not self._validate_participant(node_id):
            logger.warning(f"Rejected key advertisement from {node_id}: not a clique member")
            return secureagg_pb2.KeyAdvertisementAck(accepted=False, message="Node not in clique")

        try:
            advert = AdvertiseMessage(
                node_id=node_id,
                c_public=bytes(request.c_public_key),
                s_public=bytes(request.s_public_key),
                signature=bytes(request.signature),
                signing_public=None,  # Expected from signing_public_keys provided at init
            )
            if node_id not in self._adverts:
                self._adverts[node_id] = advert
            # Once we have threshold, commit into aggregator once.
            if not self._adverts_committed and len(self._adverts) >= self.threshold:
                self.aggregator.receive_advertisements(list(self._adverts.values()))
                self._adverts_committed = True
            elif self._adverts_committed and node_id not in self.aggregator.advertisements:
                # Add late adverts after initial commit.
                self.aggregator.receive_advertisements([advert])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SAP-Round0 advert rejected from {node_id}: {exc}")
            return secureagg_pb2.KeyAdvertisementAck(accepted=False, message=str(exc))

        # Once we have threshold, return full list.
        all_keys = self.aggregator.broadcast_advertisements() if self._adverts_committed else []
        ack_keys = [
            secureagg_pb2.KeyAdvertisement(
                node_id=adv.node_id,
                c_public_key=adv.c_public,
                s_public_key=adv.s_public,
                signature=adv.signature,
            )
            for adv in all_keys
        ]
        return secureagg_pb2.KeyAdvertisementAck(
            accepted=True,
            message="SAP-Round 0 OK" if len(all_keys) >= self.threshold else "Waiting for more participants",
            all_keys=ack_keys if len(all_keys) >= self.threshold else [],
        )

    def Round1ShareKeys(self, request: secureagg_pb2.ShareKeysMessage, context) -> secureagg_pb2.ShareKeysAck:
        """Collect encrypted secret shares (SAP-Round 1) and deliver mailbox."""
        node_id = request.node_id
        if not self._validate_participant(node_id):
            logger.warning(f"Rejected shares from {node_id}: not a clique member")
            return secureagg_pb2.ShareKeysAck(accepted=False, message="Node not in clique")

        try:
            ciphertexts = _decode_round1_ciphertexts(request.ciphertexts)
            self.aggregator.receive_round1_ciphertexts(ciphertexts)
            mailbox = self.aggregator.deliver_round1_ciphertexts(node_id)
            return secureagg_pb2.ShareKeysAck(
                accepted=True,
                message="SAP-Round 1 OK",
                mailbox=_encode_round1_ciphertexts(mailbox),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SAP-Round1 processing failed for {node_id}: {exc}")
            mailbox = self.aggregator.deliver_round1_ciphertexts(node_id)
            return secureagg_pb2.ShareKeysAck(
                accepted=False,
                message=str(exc),
                mailbox=_encode_round1_ciphertexts(mailbox),
            )

    def Round2MaskedInput(self, request: secureagg_pb2.MaskedInputMessage, context) -> secureagg_pb2.MaskedInputAck:
        """Collect masked model updates (SAP-Round 2)."""
        node_id = request.node_id
        if not self._validate_participant(node_id):
            logger.warning(f"Rejected masked input from {node_id}: not a clique member")
            return secureagg_pb2.MaskedInputAck(accepted=False, message="Node not in clique")
        try:
            # Check if this node has already submitted (polling case).
            if node_id not in self.aggregator.masked_inputs:
                masked = MaskedInput(
                    node_id=node_id,
                    masked_vector=[int.from_bytes(v, byteorder="big") for v in request.masked_vector],
                )
                self.aggregator.receive_masked_input(masked)
            # Wait for ALL participants before returning survivors (not just threshold).
            if len(self.aggregator.masked_inputs) >= len(self.participant_ids):
                survivors = self.aggregator.broadcast_survivors()
                return secureagg_pb2.MaskedInputAck(accepted=True, message="SAP-Round 2 OK", survivors=survivors)
            return secureagg_pb2.MaskedInputAck(accepted=True, message="Waiting for all participants", survivors=[])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SAP-Round2 processing failed for {node_id}: {exc}")
            survivors = self.aggregator.survivors or []
            return secureagg_pb2.MaskedInputAck(accepted=False, message=str(exc), survivors=survivors)

    def Round3ConsistencyCheck(self, request: secureagg_pb2.ConsistencySignature, context) -> secureagg_pb2.ConsistencyAck:
        """Collect consistency signatures (SAP-Round 3)."""
        node_id = request.node_id
        if not self._validate_participant(node_id):
            logger.warning(f"Rejected signature from {node_id}: not a clique member")
            return secureagg_pb2.ConsistencyAck(accepted=False, message="Node not in clique")
        try:
            sig = SurvivorSignature(node_id=node_id, signature=bytes(request.signature))
            self._round3_signatures[node_id] = sig.signature
            if len(self._round3_signatures) >= len(self.aggregator.survivors):
                sigs = [SurvivorSignature(node_id=n, signature=s) for n, s in self._round3_signatures.items()]
                self.aggregator.verify_survivor_signatures(sigs)
                return secureagg_pb2.ConsistencyAck(accepted=True, message="SAP-Round 3 OK")
            return secureagg_pb2.ConsistencyAck(accepted=True, message="Waiting for more signatures")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SAP-Round3 processing failed for {node_id}: {exc}")
            return secureagg_pb2.ConsistencyAck(accepted=False, message=str(exc))

    def Round4Unmask(self, request: secureagg_pb2.UnmaskShares, context) -> secureagg_pb2.UnmaskAck:
        """Collect unmasking shares and compute aggregate (SAP-Round 4)."""
        node_id = request.node_id
        if not self._validate_participant(node_id):
            logger.warning(f"Rejected unmask shares from {node_id}: not a clique member")
            return secureagg_pb2.UnmaskAck(accepted=False, message="Node not in clique", aggregation_complete=False)
        try:
            # Only accept first submission from each node to prevent duplicates during polling.
            already_submitted = any(p.node_id == node_id for p in self._round4_payloads)
            if not already_submitted:
                drop_shares = {k: _decode_unmask_share(v) for k, v in request.dropout_s_shares.items()}
                surv_shares = {k: _decode_unmask_share(v) for k, v in request.survivor_b_shares.items()}
                payload = UnmaskingSharesModel(
                    node_id=node_id,
                    s_shares_for_dropouts=drop_shares,
                    b_shares_for_survivors=surv_shares,
                )
                self._round4_payloads.append(payload)
            if len(self._round4_payloads) >= self.threshold:
                result = self.aggregator.receive_unmasking_shares(self._round4_payloads)
                self.aggregated_result = result.aggregate_mean
                return secureagg_pb2.UnmaskAck(
                    accepted=True,
                    message="Aggregation complete",
                    aggregation_complete=True,
                )
            return secureagg_pb2.UnmaskAck(
                accepted=True,
                message="Waiting for more unmask shares",
                aggregation_complete=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SAP-Round4 processing failed for {node_id}: {exc}")
            return secureagg_pb2.UnmaskAck(accepted=False, message=str(exc), aggregation_complete=False)

    def GetGlobalModel(self, request: secureagg_pb2.ModelRequest, context) -> secureagg_pb2.ModelResponse:
        """Return the global model with convergence signals.

        If inter-cluster merge was performed, returns IPFS reference (cid, hash)
        so nodes can fetch the merged model from IPFS. Otherwise returns the
        intra-cluster aggregated weights directly.
        """
        if self.aggregated_result is None:
            return secureagg_pb2.ModelResponse(
                model_weights=[],
                round=self.current_round,
                aggregator_id=self.node_id,
                should_stop=False,
                stop_reason="",
                delta_norm=0.0,
                cluster_converged=False,
                model_cid="",
                model_hash="",
            )

        logger.info(
            f"Serving global model (round {self.current_round}, "
            f"should_stop={self.should_stop}, delta={self.delta_norm:.2e}, "
            f"cid={self.merged_model_cid[:16] if self.merged_model_cid else 'N/A'}...)"
        )
        return secureagg_pb2.ModelResponse(
            model_weights=self.aggregated_result,
            round=self.current_round,
            aggregator_id=self.node_id,
            should_stop=self.should_stop,
            stop_reason=self.stop_reason,
            delta_norm=self.delta_norm,
            cluster_converged=self.cluster_converged,
            model_cid=self.merged_model_cid or "",
            model_hash=self.merged_model_hash or "",
            model_data_id=self.merged_model_data_id or "",
        )

    def SubmitECMs(
        self,
        request: secureagg_pb2.ECMSubmitRequest,
        context,
    ) -> secureagg_pb2.ECMSubmitResponse:
        """Receive ECMs forwarded by bridge nodes for inter-cluster merge."""
        if self.ecm_buffer is None:
            logger.warning("Received ECMs but no ECM buffer configured")
            return secureagg_pb2.ECMSubmitResponse(
                accepted=False,
                message="ECM buffer not configured on aggregator",
            )

        received_count = 0
        for ecm_msg in request.ecms:
            ecm = ECM(
                cid=ecm_msg.cid,
                hash=ecm_msg.hash,
                source_cluster=ecm_msg.source_cluster,
            )
            self.ecm_buffer.add(ecm)
            received_count += 1
            logger.debug(
                f"Received ECM from bridge {request.node_id}: "
                f"cid={ecm.cid[:8]}... cluster={ecm.source_cluster}"
            )

        logger.info(
            f"Aggregator received {received_count} ECMs from bridge node {request.node_id}"
        )
        if self.ecm_buffer:
            unique_ecms = self.ecm_buffer.get_unique_cids()
            if unique_ecms:
                formatted = ", ".join(
                    f"{cid[:8]}...:{hash_val[:8]}..."
                    for cid, hash_val in unique_ecms.items()
                )
                logger.info(
                    "Aggregator ECM buffer now has %d unique models: %s",
                    len(unique_ecms),
                    formatted,
                )
            else:
                logger.info("Aggregator ECM buffer is empty after update")
        return secureagg_pb2.ECMSubmitResponse(
            accepted=True,
            message=f"Received {received_count} ECMs",
        )

    def reset_for_next_round(self) -> None:
        """Reset state for next aggregation round."""
        self.aggregator = SecureAggregationAggregator(
            config=SecureAggregationConfig(participants=self.participant_ids, threshold=self.threshold),
            signing_public_keys=self.aggregator.signing_public_keys,
        )
        self.aggregated_result = None
        self.merged_model_cid = None
        self.merged_model_hash = None
        self.merged_model_data_id = None
        self._adverts.clear()
        self._adverts_committed = False
        self._round3_signatures.clear()
        self._round4_payloads.clear()
        self.current_round += 1
        logger.info(f"Reset for round {self.current_round}")


def serve(
    node_id: str,
    port: int,
    threshold: int,
    participant_ids: List[str],
    signing_public_keys: Optional[Mapping[str, bytes]] = None,
    ecm_buffer: Optional[ECMBuffer] = None,
) -> Tuple[grpc.Server, AggregatorServicer]:
    """Start the aggregator gRPC server.

    Returns:
        Tuple of (server, servicer) so caller can access servicer for convergence state.
    """
    servicer = AggregatorServicer(
        node_id,
        threshold,
        participant_ids,
        signing_public_keys=signing_public_keys,
        ecm_buffer=ecm_buffer,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    secureagg_pb2_grpc.add_AggregatorServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Aggregator server started on port {port}")
    return server, servicer
