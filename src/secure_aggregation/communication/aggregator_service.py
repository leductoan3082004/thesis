"""Aggregator service that coordinates secure aggregation protocol."""

import logging
import os
import threading
import time
from concurrent import futures
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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


class PortBindingError(RuntimeError):
    """Raised when the aggregator server cannot bind to the requested port."""

    def __init__(self, port: int, message: str = "") -> None:
        prefix = f"Failed to bind aggregator server to port {port}"
        full_message = f"{prefix}: {message}" if message else prefix
        super().__init__(full_message)
        self.port = port

DEFAULT_GRPC_MAX_MESSAGE_MB = 200
_grpc_max_env = os.environ.get("GRPC_MAX_MESSAGE_MB")
try:
    _grpc_max_mb = int(_grpc_max_env) if _grpc_max_env else DEFAULT_GRPC_MAX_MESSAGE_MB
except ValueError:
    _grpc_max_mb = DEFAULT_GRPC_MAX_MESSAGE_MB
GRPC_MAX_MESSAGE_LENGTH_BYTES = max(1, _grpc_max_mb) * 1024 * 1024


def grpc_message_options(max_length: int = GRPC_MAX_MESSAGE_LENGTH_BYTES) -> List[Tuple[str, int]]:
    """Return gRPC channel/server options enforcing a higher message size limit."""
    return [
        ("grpc.max_send_message_length", max_length),
        ("grpc.max_receive_message_length", max_length),
    ]


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
        convergence_signal_handler: Optional[Callable[[str, int], None]] = None,
        round0_timeout_seconds: float = 0.0,
    ) -> None:
        self.node_id = node_id
        self.threshold = threshold
        self.all_participant_ids = list(participant_ids)
        self.participant_ids = list(participant_ids)
        config = SecureAggregationConfig(participants=participant_ids, threshold=threshold)
        self._signing_public_keys = signing_public_keys
        self.aggregator = SecureAggregationAggregator(config=config, signing_public_keys=signing_public_keys)
        self.aggregated_result: Optional[List[float]] = None
        self.current_round = 0
        self._adverts: Dict[str, AdvertiseMessage] = {}
        self._adverts_committed = False
        self._round3_signatures: Dict[str, bytes] = {}
        self._round4_payloads: List[UnmaskingSharesModel] = []
        self._round_snapshots: Dict[int, secureagg_pb2.ModelResponse] = {}
        self._committed_adverts: List[AdvertiseMessage] = []
        self._round0_finalized: bool = False
        self._round2_finalized: bool = False
        self._round2_survivors: List[str] = []
        self._active_threshold = max(1, min(self.threshold, len(self.participant_ids)))
        self._round0_timeout_seconds = max(0.0, round0_timeout_seconds)
        self._round0_opened_at = time.monotonic()
        self._round0_lock = threading.Lock()

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
        self.convergence_streak: int = 0
        self.metadata_ready: bool = False
        self._convergence_signal_handler = convergence_signal_handler

        # Ensure a clean baseline state.
        self.prepare_round(0)

        logger.info(
            f"Aggregator {node_id} initialized with threshold={threshold}, participants={len(participant_ids)}"
        )

    def set_convergence_signal_handler(self, handler: Callable[[str, int], None]) -> None:
        """Register callback that receives convergence data_id notifications."""
        self._convergence_signal_handler = handler

    def set_convergence_state(
        self,
        model_cid: Optional[str],
        model_hash: Optional[str],
        model_data_id: Optional[str],
        should_stop: bool,
        stop_reason: str,
        delta_norm: float,
        cluster_converged: bool,
        convergence_streak: int,
    ) -> None:
        """Store IPFS reference and convergence info for distribution to all nodes."""
        self.merged_model_cid = model_cid
        self.merged_model_hash = model_hash
        self.merged_model_data_id = model_data_id
        self.should_stop = should_stop
        self.stop_reason = stop_reason
        self.delta_norm = delta_norm
        self.cluster_converged = cluster_converged
        self.convergence_streak = convergence_streak
        self.metadata_ready = True

    def _validate_participant(self, node_id: str) -> bool:
        return node_id in self.participant_ids

    def _round0_timeout_elapsed_locked(self) -> bool:
        if self._round0_timeout_seconds <= 0:
            return False
        return (time.monotonic() - self._round0_opened_at) >= self._round0_timeout_seconds

    def _finalize_round0_if_ready_locked(self) -> None:
        if self._round0_finalized:
            return
        committed = len(self._committed_adverts)
        total = len(self.all_participant_ids)
        if committed >= total and total > 0:
            self._round0_finalized = True
            logger.info("SAP-Round 0 finalized after accepting all %d participants", total)
            self._activate_committed_participants_locked()
            return
        if committed >= self.threshold and self._round0_timeout_elapsed_locked():
            self._round0_finalized = True
            elapsed = time.monotonic() - self._round0_opened_at
            logger.info(
                "SAP-Round 0 finalized after timeout (participants=%d, threshold=%d, timeout=%.1fs, elapsed=%.1fs)",
                committed,
                self.threshold,
                self._round0_timeout_seconds,
                elapsed,
            )
            self._activate_committed_participants_locked()

    def _activate_committed_participants_locked(self) -> None:
        committed_ids = [adv.node_id for adv in self._committed_adverts]
        if not committed_ids:
            return
        self.participant_ids = committed_ids
        self._active_threshold = max(1, min(self.threshold, len(self.participant_ids)))
        self.aggregator = SecureAggregationAggregator(
            config=SecureAggregationConfig(participants=self.participant_ids, threshold=self._active_threshold),
            signing_public_keys=self._signing_public_keys,
        )
        try:
            self.aggregator.receive_advertisements(self._committed_adverts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to reapply committed adverts after Round 0 finalization: %s", exc)
        logger.info(
            "Activated %d Round 0 participants: %s (threshold=%d)",
            len(self.participant_ids),
            ", ".join(sorted(self.participant_ids)),
            self._active_threshold,
        )
    def _encoded_committed_adverts(self) -> List[secureagg_pb2.KeyAdvertisement]:
        return [
            secureagg_pb2.KeyAdvertisement(
                node_id=adv.node_id,
                c_public_key=adv.c_public,
                s_public_key=adv.s_public,
                signature=adv.signature,
            )
            for adv in self._committed_adverts
        ]

    def Round0AdvertiseKeys(self, request: secureagg_pb2.KeyAdvertisement, context) -> secureagg_pb2.KeyAdvertisementAck:
        """Collect DH public keys from participants (SAP-Round 0)."""
        node_id = request.node_id

        if not self._validate_participant(node_id):
            logger.warning(f"Rejected key advertisement from {node_id}: not a clique member")
            return secureagg_pb2.KeyAdvertisementAck(accepted=False, message="Node not in clique")

        response_keys: List[secureagg_pb2.KeyAdvertisement] = []
        response_message = "Waiting for more participants"
        accepted = True
        with self._round0_lock:
            if self._round0_finalized and node_id not in self._adverts:
                logger.warning("Rejected key advertisement from %s: Round 0 already finalized", node_id)
                return secureagg_pb2.KeyAdvertisementAck(
                    accepted=False,
                    message="Round 0 finalized",
                    all_keys=self._encoded_committed_adverts(),
                )

            try:
                advert = AdvertiseMessage(
                    node_id=node_id,
                    c_public=bytes(request.c_public_key),
                    s_public=bytes(request.s_public_key),
                    signature=bytes(request.signature),
                    signing_public=None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"SAP-Round0 advert rejected from {node_id}: {exc}")
                return secureagg_pb2.KeyAdvertisementAck(accepted=False, message=str(exc))

            is_new_advert = node_id not in self._adverts
            if is_new_advert:
                self._adverts[node_id] = advert

            try:
                if not self._adverts_committed and len(self._adverts) >= self.threshold:
                    self.aggregator.receive_advertisements(list(self._adverts.values()))
                    self._adverts_committed = True
                elif self._adverts_committed and is_new_advert:
                    self.aggregator.receive_advertisements([advert])
            except Exception as exc:  # noqa: BLE001
                if is_new_advert:
                    self._adverts.pop(node_id, None)
                logger.warning(f"SAP-Round0 advert rejected from {node_id}: {exc}")
                return secureagg_pb2.KeyAdvertisementAck(accepted=False, message=str(exc))

            if self._adverts_committed:
                self._committed_adverts = self.aggregator.broadcast_advertisements()
            else:
                self._committed_adverts = []

            self._finalize_round0_if_ready_locked()
            response_keys = self._encoded_committed_adverts()
            if response_keys:
                response_message = "SAP-Round 0 OK"
            if self._round0_finalized and node_id not in self._adverts:
                return secureagg_pb2.KeyAdvertisementAck(
                    accepted=False,
                    message="Round 0 finalized",
                    all_keys=response_keys,
                )

        return secureagg_pb2.KeyAdvertisementAck(
            accepted=True,
            message=response_message,
            all_keys=response_keys,
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
            if self._round2_finalized:
                survivors = list(self._round2_survivors)
                if node_id in self.aggregator.masked_inputs:
                    return secureagg_pb2.MaskedInputAck(
                        accepted=True,
                        message="SAP-Round 2 OK",
                        survivors=survivors,
                    )
                logger.warning("Masked input from %s rejected: Round 2 already finalized", node_id)
                return secureagg_pb2.MaskedInputAck(
                    accepted=False,
                    message="Round 2 finalized",
                    survivors=survivors,
                )
            # Check if this node has already submitted (polling case).
            if node_id not in self.aggregator.masked_inputs:
                masked = MaskedInput(
                    node_id=node_id,
                    masked_vector=[int.from_bytes(v, byteorder="big") for v in request.masked_vector],
                )
                self.aggregator.receive_masked_input(masked)
            if len(self.aggregator.masked_inputs) >= self._active_threshold:
                survivors = self.aggregator.broadcast_survivors()
                self._round2_survivors = survivors
                self._round2_finalized = True
                logger.info(
                    "SAP-Round 2 finalized with %d survivors (threshold=%d)",
                    len(survivors),
                    self._active_threshold,
                )
                return secureagg_pb2.MaskedInputAck(accepted=True, message="SAP-Round 2 OK", survivors=survivors)
            return secureagg_pb2.MaskedInputAck(accepted=True, message="Waiting for more participants", survivors=[])
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
        if self._round2_finalized and node_id not in self.aggregator.survivors:
            logger.warning("Rejected signature from %s: not a survivor", node_id)
            return secureagg_pb2.ConsistencyAck(accepted=False, message="Node not a survivor")
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
        if self._round2_finalized and node_id not in self.aggregator.survivors:
            logger.warning("Rejected unmask shares from %s: not a survivor", node_id)
            return secureagg_pb2.UnmaskAck(
                accepted=False,
                message="Node not a survivor",
                aggregation_complete=False,
            )
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
            if len(self._round4_payloads) >= self._active_threshold:
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

    def _default_model_response(self, _: int) -> secureagg_pb2.ModelResponse:
        """Return an empty placeholder response for callers still waiting on aggregation."""
        return secureagg_pb2.ModelResponse(
            model_weights=[],
            round=self.current_round,
            aggregator_id=self.node_id,
            should_stop=self.should_stop,
            stop_reason=self.stop_reason,
            delta_norm=self.delta_norm,
            cluster_converged=self.cluster_converged,
            convergence_streak=self.convergence_streak,
            metadata_ready=self.metadata_ready,
            model_cid=self.merged_model_cid or "",
            model_hash=self.merged_model_hash or "",
            model_data_id=self.merged_model_data_id or "",
        )

    def _build_model_response(self) -> secureagg_pb2.ModelResponse:
        """Compose the response for the current round using the latest metadata."""
        if self.aggregated_result is None:
            return self._default_model_response(self.current_round)
        return secureagg_pb2.ModelResponse(
            model_weights=self.aggregated_result,
            round=self.current_round,
            aggregator_id=self.node_id,
            should_stop=self.should_stop,
            stop_reason=self.stop_reason,
            delta_norm=self.delta_norm,
            cluster_converged=self.cluster_converged,
            convergence_streak=self.convergence_streak,
            metadata_ready=self.metadata_ready,
            model_cid=self.merged_model_cid or "",
            model_hash=self.merged_model_hash or "",
            model_data_id=self.merged_model_data_id or "",
        )

    def _cache_round_snapshot(self, round_idx: int, response: secureagg_pb2.ModelResponse) -> None:
        """Store the completed round result so stragglers can still fetch it later."""
        self._round_snapshots[round_idx] = response
        # Retain only the most recent round and its predecessor to bound memory usage.
        obsolete = [idx for idx in self._round_snapshots if idx < round_idx - 1]
        for idx in obsolete:
            self._round_snapshots.pop(idx, None)

    def GetGlobalModel(self, request: secureagg_pb2.ModelRequest, context) -> secureagg_pb2.ModelResponse:
        """Return the global model with convergence signals.

        If inter-cluster merge was performed, returns IPFS reference (cid, hash)
        so nodes can fetch the merged model from IPFS. Otherwise returns the
        intra-cluster aggregated weights directly.
        """
        requested_round = request.round

        if requested_round != self.current_round or self.aggregated_result is None:
            cached = self._round_snapshots.get(requested_round)
            if cached:
                logger.debug(
                    "Serving cached global model for round %d to requester %s",
                    requested_round,
                    getattr(request, "node_id", "unknown"),
                )
                return cached
            return self._default_model_response(requested_round)

        logger.info(
            f"Serving global model (round {self.current_round}, "
            f"should_stop={self.should_stop}, delta={self.delta_norm:.2e}, "
            f"cid={self.merged_model_cid[:16] if self.merged_model_cid else 'N/A'}...)"
        )
        response = self._build_model_response()
        self._cache_round_snapshot(self.current_round, response)
        return response

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

    def NotifyConvergenceSignal(
        self,
        request: secureagg_pb2.ConvergenceSignal,
        context,
    ) -> secureagg_pb2.ConvergenceAck:
        """Allow clique members to push convergence confirmations to the aggregator."""
        if not request.data_id:
            return secureagg_pb2.ConvergenceAck(accepted=False, message="Missing data_id")
        if self._convergence_signal_handler is None:
            logger.debug(
                "Aggregator %s received convergence data_id=%s but no handler configured",
                self.node_id,
                request.data_id,
            )
            return secureagg_pb2.ConvergenceAck(accepted=False, message="No handler configured")
        try:
            self._convergence_signal_handler(request.data_id, request.round)
            return secureagg_pb2.ConvergenceAck(accepted=True, message="Convergence acknowledged")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Convergence signal handler failed for data_id=%s: %s", request.data_id, exc)
            return secureagg_pb2.ConvergenceAck(accepted=False, message=str(exc))

    def prepare_round(self, round_idx: int) -> None:
        """Reset state in preparation for the supplied aggregation round."""
        self.participant_ids = list(self.all_participant_ids)
        self._active_threshold = max(1, min(self.threshold, len(self.participant_ids)))
        self.aggregator = SecureAggregationAggregator(
            config=SecureAggregationConfig(participants=self.participant_ids, threshold=self._active_threshold),
            signing_public_keys=self._signing_public_keys,
        )
        self.aggregated_result = None
        self.merged_model_cid = None
        self.merged_model_hash = None
        self.merged_model_data_id = None
        self.should_stop = False
        self.stop_reason = ""
        self.delta_norm = 0.0
        self.cluster_converged = False
        self.convergence_streak = 0
        self.metadata_ready = False
        self._adverts.clear()
        self._adverts_committed = False
        self._committed_adverts = []
        self._round0_finalized = False
        self._round3_signatures.clear()
        self._round4_payloads.clear()
        self._round2_finalized = False
        self._round2_survivors = []
        self._round0_opened_at = time.monotonic()
        self.current_round = round_idx
        logger.info("Aggregator %s prepared for round %d", self.node_id, round_idx)

    def reset_for_next_round(self) -> None:
        """Maintain backward compatibility with previous API."""
        self.prepare_round(self.current_round + 1)


def serve(
    node_id: str,
    port: int,
    threshold: int,
    participant_ids: List[str],
    signing_public_keys: Optional[Mapping[str, bytes]] = None,
    ecm_buffer: Optional[ECMBuffer] = None,
    convergence_signal_handler: Optional[Callable[[str, int], None]] = None,
    round0_timeout_seconds: float = 0.0,
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
        convergence_signal_handler=convergence_signal_handler,
        round0_timeout_seconds=round0_timeout_seconds,
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=grpc_message_options(),
    )
    secureagg_pb2_grpc.add_AggregatorServiceServicer_to_server(servicer, server)
    bound_port = 0
    try:
        bound_port = server.add_insecure_port(f"[::]:{port}")
    except Exception as exc:  # noqa: BLE001
        server.stop(0)
        raise PortBindingError(port, str(exc)) from exc
    if not bound_port:
        server.stop(0)
        raise PortBindingError(port, "gRPC returned 0 (port already in use?)")
    server.start()
    logger.info(f"Aggregator server started on port {port}")
    return server, servicer
