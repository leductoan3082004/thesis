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
        timeout_seconds: float = 0.0,
        plaintext_mode: bool = False,
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
        self._round2_waiting_on_aggregator: bool = False
        self._round2_timed_out: bool = False
        self._active_threshold = max(1, min(self.threshold, len(self.participant_ids)))
        self._timeout_seconds = max(0.0, timeout_seconds)
        self._round0_opened_at = time.monotonic()
        self._round2_opened_at = time.monotonic()
        self._round0_lock = threading.Lock()
        self._round2_lock = threading.Lock()
        self._round4_lock = threading.Lock()
        self._round4_computing: bool = False

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
        self.plaintext_mode = plaintext_mode
        self._plaintext_updates: Dict[str, List[float]] = {}
        self._plaintext_lock = threading.Lock()
        self._plaintext_vector_length: Optional[int] = None
        self._plaintext_round_opened_at = time.monotonic()

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

    @staticmethod
    def _check_round_sync(request_round: int, server_round: int) -> int:
        """Compare request round to the already-captured server round.

        Accepts server_round explicitly to avoid re-reading self.current_round,
        which could change between the caller's snapshot and this call.
        """
        if request_round < server_round:
            return secureagg_pb2.ROUND_SYNC_STALE
        if request_round > server_round:
            return secureagg_pb2.ROUND_SYNC_AHEAD
        return secureagg_pb2.ROUND_SYNC_OK

    def _round0_timeout_elapsed_locked(self) -> bool:
        if self._timeout_seconds <= 0:
            return False
        return (time.monotonic() - self._round0_opened_at) >= self._timeout_seconds

    def _round2_timeout_elapsed(self) -> bool:
        if self._timeout_seconds <= 0:
            return False
        return (time.monotonic() - self._round2_opened_at) >= self._timeout_seconds

    def _finalize_round0_if_ready_locked(self) -> None:
        if self._round0_finalized:
            return
        committed = len(self.aggregator.advertisements)
        total = len(self.all_participant_ids)
        if committed >= total and total > 0:
            self._round0_finalized = True
            self._committed_adverts = self.aggregator.broadcast_advertisements()
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
                self._timeout_seconds,
                elapsed,
            )
            self._committed_adverts = self.aggregator.broadcast_advertisements()
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
        server_round = self.current_round

        if node_id not in self.all_participant_ids:
            logger.warning(f"Rejected key advertisement from {node_id}: not a clique member")
            return secureagg_pb2.KeyAdvertisementAck(
                accepted=False,
                message="Node not in clique",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_NOT_MEMBER,
            )

        if request.round != server_round:
            sync_code = self._check_round_sync(request.round, server_round)
            logger.warning(
                "Round 0 mismatch from %s: request.round=%d, server_round=%d",
                node_id, request.round, server_round,
            )
            return secureagg_pb2.KeyAdvertisementAck(
                accepted=False,
                message=f"Round mismatch: request={request.round}, server={server_round}",
                server_round=server_round,
                sync_code=sync_code,
            )

        if not self._validate_participant(node_id):
            logger.warning(f"Rejected key advertisement from {node_id}: not active in current round")
            return secureagg_pb2.KeyAdvertisementAck(
                accepted=False,
                message="Node not active in current round",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
            )

        response_keys: List[secureagg_pb2.KeyAdvertisement] = []
        response_message = "Waiting for more participants"
        with self._round0_lock:
            if self._round0_finalized and node_id not in self._adverts:
                logger.warning("Rejected key advertisement from %s: Round 0 already finalized", node_id)
                return secureagg_pb2.KeyAdvertisementAck(
                    accepted=False,
                    message="Round 0 finalized",
                    all_keys=self._encoded_committed_adverts(),
                    server_round=server_round,
                    sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
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
                return secureagg_pb2.KeyAdvertisementAck(
                    accepted=False,
                    message=str(exc),
                    server_round=server_round,
                    sync_code=secureagg_pb2.ROUND_SYNC_OK,
                )

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
                return secureagg_pb2.KeyAdvertisementAck(
                    accepted=False,
                    message=str(exc),
                    server_round=server_round,
                    sync_code=secureagg_pb2.ROUND_SYNC_OK,
                )

            self._finalize_round0_if_ready_locked()
            if self._round0_finalized:
                self._committed_adverts = self.aggregator.broadcast_advertisements()
                response_keys = self._encoded_committed_adverts()
                response_message = "SAP-Round 0 OK"
            else:
                response_keys = []
                response_message = "Waiting for more participants"
            if self._round0_finalized and node_id not in self._adverts:
                return secureagg_pb2.KeyAdvertisementAck(
                    accepted=False,
                    message="Round 0 finalized",
                    all_keys=response_keys,
                    server_round=server_round,
                    sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
                )

        return secureagg_pb2.KeyAdvertisementAck(
            accepted=True,
            message=response_message,
            all_keys=response_keys,
            server_round=server_round,
            sync_code=secureagg_pb2.ROUND_SYNC_OK,
        )

    def Round1ShareKeys(self, request: secureagg_pb2.ShareKeysMessage, context) -> secureagg_pb2.ShareKeysAck:
        """Collect encrypted secret shares (SAP-Round 1) and deliver mailbox."""
        node_id = request.node_id
        server_round = self.current_round

        if node_id not in self.all_participant_ids:
            logger.warning(f"Rejected shares from {node_id}: not a clique member")
            return secureagg_pb2.ShareKeysAck(
                accepted=False,
                message="Node not in clique",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_NOT_MEMBER,
            )

        if request.round != server_round:
            sync_code = self._check_round_sync(request.round, server_round)
            return secureagg_pb2.ShareKeysAck(
                accepted=False,
                message=f"Round mismatch: request={request.round}, server={server_round}",
                server_round=server_round,
                sync_code=sync_code,
            )

        if not self._validate_participant(node_id):
            return secureagg_pb2.ShareKeysAck(
                accepted=False,
                message="Node not active in current round",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
            )

        try:
            ciphertexts = _decode_round1_ciphertexts(request.ciphertexts)
            self.aggregator.receive_round1_ciphertexts(ciphertexts)
            mailbox = self.aggregator.deliver_round1_ciphertexts(node_id)
            return secureagg_pb2.ShareKeysAck(
                accepted=True,
                message="SAP-Round 1 OK",
                mailbox=_encode_round1_ciphertexts(mailbox),
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SAP-Round1 processing failed for {node_id}: {exc}")
            mailbox = self.aggregator.deliver_round1_ciphertexts(node_id)
            return secureagg_pb2.ShareKeysAck(
                accepted=False,
                message=str(exc),
                mailbox=_encode_round1_ciphertexts(mailbox),
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )

    def Round2MaskedInput(self, request: secureagg_pb2.MaskedInputMessage, context) -> secureagg_pb2.MaskedInputAck:
        """Collect masked model updates (SAP-Round 2)."""
        node_id = request.node_id
        server_round = self.current_round

        if node_id not in self.all_participant_ids:
            logger.warning(f"Rejected masked input from {node_id}: not a clique member")
            return secureagg_pb2.MaskedInputAck(
                accepted=False,
                message="Node not in clique",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_NOT_MEMBER,
            )

        if request.round != server_round:
            sync_code = self._check_round_sync(request.round, server_round)
            return secureagg_pb2.MaskedInputAck(
                accepted=False,
                message=f"Round mismatch: request={request.round}, server={server_round}",
                survivors=list(self._round2_survivors) if self._round2_finalized else [],
                timed_out=self._round2_timed_out,
                server_round=server_round,
                sync_code=sync_code,
            )

        if not self._validate_participant(node_id):
            logger.warning(f"Rejected masked input from {node_id}: not active in current round")
            return secureagg_pb2.MaskedInputAck(
                accepted=False,
                message="Node not active in current round",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
            )

        with self._round2_lock:
            try:
                if self._round2_finalized:
                    survivors = list(self._round2_survivors)
                    if node_id in self.aggregator.masked_inputs:
                        return secureagg_pb2.MaskedInputAck(
                            accepted=True,
                            message="SAP-Round 2 OK",
                            survivors=survivors,
                            timed_out=self._round2_timed_out,
                            server_round=server_round,
                            sync_code=secureagg_pb2.ROUND_SYNC_OK,
                        )
                    logger.warning("Masked input from %s rejected: Round 2 already finalized", node_id)
                    return secureagg_pb2.MaskedInputAck(
                        accepted=False,
                        message="Round 2 finalized",
                        survivors=survivors,
                        timed_out=self._round2_timed_out,
                        server_round=server_round,
                        sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
                    )
                if node_id not in self.aggregator.masked_inputs:
                    masked = MaskedInput(
                        node_id=node_id,
                        masked_vector=[int.from_bytes(v, byteorder="big") for v in request.masked_vector],
                    )
                    self.aggregator.receive_masked_input(masked)
                    logger.info(
                        "SAP-Round 2 recorded masked input from %s (%d total)",
                        node_id,
                        len(self.aggregator.masked_inputs),
                    )
                masked_count = len(self.aggregator.masked_inputs)
                threshold_met = masked_count >= self._active_threshold
                aggregator_submitted = self.node_id in self.aggregator.masked_inputs
                total_expected = len(self.participant_ids)
                if threshold_met and not aggregator_submitted:
                    if not self._round2_waiting_on_aggregator:
                        logger.info(
                            "SAP-Round 2 threshold met (%d masked inputs) but aggregator %s has not submitted yet; waiting",
                            masked_count,
                            self.node_id,
                        )
                        self._round2_waiting_on_aggregator = True
                    return secureagg_pb2.MaskedInputAck(
                        accepted=True,
                        message="Waiting for aggregator masked input",
                        survivors=[],
                        timed_out=False,
                        server_round=server_round,
                        sync_code=secureagg_pb2.ROUND_SYNC_OK,
                    )
                finalize_reason = ""
                if total_expected > 0 and masked_count >= total_expected:
                    logger.info("SAP-Round 2 received masked input from %s (%d/%d)", node_id, masked_count, total_expected)
                    finalize_reason = "all_participants"
                elif threshold_met and self._round2_timeout_elapsed():
                    finalize_reason = "timeout"
                    self._round2_timed_out = True
                if finalize_reason:
                    self._round2_waiting_on_aggregator = False
                    survivors = self.aggregator.broadcast_survivors()
                    self._round2_survivors = survivors
                    self._round2_finalized = True
                    if finalize_reason == "timeout":
                        elapsed = time.monotonic() - self._round2_opened_at
                        logger.warning(
                            "SAP-Round 2 timeout after %.1fs; survivors=%s",
                            elapsed,
                            ", ".join(sorted(survivors)),
                        )
                        message = "SAP-Round 2 finalized after timeout"
                    else:
                        logger.info(
                            "SAP-Round 2 finalized with %d survivors (threshold=%d)",
                            len(survivors),
                            self._active_threshold,
                        )
                        message = "SAP-Round 2 OK"
                    return secureagg_pb2.MaskedInputAck(
                        accepted=True,
                        message=message,
                        survivors=survivors,
                        timed_out=self._round2_timed_out,
                        server_round=server_round,
                        sync_code=secureagg_pb2.ROUND_SYNC_OK,
                    )
                return secureagg_pb2.MaskedInputAck(
                    accepted=True,
                    message="Waiting for more participants",
                    survivors=[],
                    timed_out=False,
                    server_round=server_round,
                    sync_code=secureagg_pb2.ROUND_SYNC_OK,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"SAP-Round2 processing failed for {node_id}: {exc}")
                survivors = self.aggregator.survivors or []
                return secureagg_pb2.MaskedInputAck(
                    accepted=False,
                    message=str(exc),
                    survivors=survivors,
                    timed_out=self._round2_timed_out,
                    server_round=server_round,
                    sync_code=secureagg_pb2.ROUND_SYNC_OK,
                )

    def Round3ConsistencyCheck(self, request: secureagg_pb2.ConsistencySignature, context) -> secureagg_pb2.ConsistencyAck:
        """Collect consistency signatures (SAP-Round 3)."""
        node_id = request.node_id
        server_round = self.current_round

        if node_id not in self.all_participant_ids:
            logger.warning(f"Rejected signature from {node_id}: not a clique member")
            return secureagg_pb2.ConsistencyAck(
                accepted=False,
                message="Node not in clique",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_NOT_MEMBER,
            )

        if request.round != server_round:
            sync_code = self._check_round_sync(request.round, server_round)
            return secureagg_pb2.ConsistencyAck(
                accepted=False,
                message=f"Round mismatch: request={request.round}, server={server_round}",
                server_round=server_round,
                sync_code=sync_code,
            )

        if not self._validate_participant(node_id):
            return secureagg_pb2.ConsistencyAck(
                accepted=False,
                message="Node not active in current round",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
            )

        if self._round2_finalized and node_id not in self.aggregator.survivors:
            logger.warning("Rejected signature from %s: not a survivor", node_id)
            return secureagg_pb2.ConsistencyAck(
                accepted=False,
                message="Node not a survivor",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
            )
        try:
            sig = SurvivorSignature(node_id=node_id, signature=bytes(request.signature))
            self._round3_signatures[node_id] = sig.signature
            if len(self._round3_signatures) >= len(self.aggregator.survivors):
                sigs = [SurvivorSignature(node_id=n, signature=s) for n, s in self._round3_signatures.items()]
                self.aggregator.verify_survivor_signatures(sigs)
                return secureagg_pb2.ConsistencyAck(
                    accepted=True,
                    message="SAP-Round 3 OK",
                    server_round=server_round,
                    sync_code=secureagg_pb2.ROUND_SYNC_OK,
                )
            return secureagg_pb2.ConsistencyAck(
                accepted=True,
                message="Waiting for more signatures",
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SAP-Round3 processing failed for {node_id}: {exc}")
            return secureagg_pb2.ConsistencyAck(
                accepted=False,
                message=str(exc),
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )

    def Round4Unmask(self, request: secureagg_pb2.UnmaskShares, context) -> secureagg_pb2.UnmaskAck:
        """Collect unmasking shares and compute aggregate (SAP-Round 4).

        The aggregation (mask removal + mean) is dispatched to a background thread once the
        threshold is reached so that the gRPC handler never blocks.  Subsequent poll calls
        return aggregation_complete=True as soon as the result is available.
        """
        node_id = request.node_id
        server_round = self.current_round

        if node_id not in self.all_participant_ids:
            logger.warning(f"Rejected unmask shares from {node_id}: not a clique member")
            return secureagg_pb2.UnmaskAck(
                accepted=False,
                message="Node not in clique",
                aggregation_complete=False,
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_NOT_MEMBER,
            )

        if request.round != server_round:
            sync_code = self._check_round_sync(request.round, server_round)
            return secureagg_pb2.UnmaskAck(
                accepted=False,
                message=f"Round mismatch: request={request.round}, server={server_round}",
                aggregation_complete=self.aggregated_result is not None,
                server_round=server_round,
                sync_code=sync_code,
            )

        if not self._validate_participant(node_id):
            return secureagg_pb2.UnmaskAck(
                accepted=False,
                message="Node not active in current round",
                aggregation_complete=False,
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
            )

        if self._round2_finalized and node_id not in self.aggregator.survivors:
            logger.warning("Rejected unmask shares from %s: not a survivor", node_id)
            return secureagg_pb2.UnmaskAck(
                accepted=False,
                message="Node not a survivor",
                aggregation_complete=False,
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_FINALIZED,
            )

        # Fast-path for polling nodes: background computation already finished.
        if self.aggregated_result is not None:
            return secureagg_pb2.UnmaskAck(
                accepted=True,
                message="Aggregation complete",
                aggregation_complete=True,
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )

        try:
            should_start_computing = False
            payloads_snapshot: List[UnmaskingSharesModel] = []
            with self._round4_lock:
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

                # Dispatch computation exactly once when threshold is reached.
                if len(self._round4_payloads) >= self._active_threshold and not self._round4_computing:
                    self._round4_computing = True
                    should_start_computing = True
                    payloads_snapshot = list(self._round4_payloads)

            if should_start_computing:
                threading.Thread(
                    target=self._run_unmasking,
                    args=(payloads_snapshot,),
                    daemon=True,
                    name=f"round4-unmask-{self.current_round}",
                ).start()

            return secureagg_pb2.UnmaskAck(
                accepted=True,
                message="Waiting for more unmask shares",
                aggregation_complete=False,
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"SAP-Round4 processing failed for {node_id}: {exc}")
            return secureagg_pb2.UnmaskAck(
                accepted=False,
                message=str(exc),
                aggregation_complete=False,
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )

    def _run_unmasking(self, payloads: List[UnmaskingSharesModel]) -> None:
        """Background thread: reconstruct masks and compute aggregate mean.

        Runs off the gRPC thread pool so that mask removal (expensive Python bigint arithmetic
        over the full model vector) never blocks handler threads or triggers client timeouts.
        """
        try:
            result = self.aggregator.receive_unmasking_shares(payloads)
            self.aggregated_result = result.aggregate_mean
            logger.info("SAP-Round 4 aggregation complete (%d survivors)", len(result.survivors))
        except Exception as exc:  # noqa: BLE001
            logger.error("SAP-Round 4 aggregation failed: %s", exc, exc_info=True)
            # Reset flag so a retry poll can re-attempt if something went wrong.
            self._round4_computing = False

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

    def _plaintext_timeout_elapsed_locked(self) -> bool:
        if self._timeout_seconds <= 0:
            return False
        return (time.monotonic() - self._plaintext_round_opened_at) >= self._timeout_seconds

    def _finalize_plaintext_aggregation_locked(self) -> bool:
        if self.aggregated_result is not None:
            return True
        contributor_count = len(self._plaintext_updates)
        if contributor_count == 0:
            return False
        if contributor_count < self._active_threshold and not self._plaintext_timeout_elapsed_locked():
            return False

        vectors = list(self._plaintext_updates.values())
        vector_length = self._plaintext_vector_length or len(vectors[0])
        if vector_length == 0:
            return False

        aggregate = [0.0] * vector_length
        for vec in vectors:
            if len(vec) != vector_length:
                raise ValueError("Plaintext vector length mismatch")
            for idx, value in enumerate(vec):
                aggregate[idx] += float(value)

        self.aggregated_result = [val / contributor_count for val in aggregate]
        logger.info(
            "Plaintext aggregation complete on %s with %d contributors (threshold=%d)",
            self.node_id,
            contributor_count,
            self._active_threshold,
        )
        return True

    def SubmitPlaintextUpdate(
        self,
        request: secureagg_pb2.PlaintextUpdate,
        context,
    ) -> secureagg_pb2.PlaintextAck:
        """Accept unmasked model vectors and compute their mean when SAP is disabled."""
        node_id = request.node_id
        server_round = self.current_round
        sync_code = self._check_round_sync(request.round, server_round)
        if sync_code != secureagg_pb2.ROUND_SYNC_OK:
            return secureagg_pb2.PlaintextAck(
                accepted=False,
                message="Round synchronization mismatch",
                aggregation_complete=False,
                submissions=len(self._plaintext_updates),
                server_round=server_round,
                sync_code=sync_code,
            )
        if not self.plaintext_mode:
            logger.debug("Plaintext submission received while mode disabled")
            return secureagg_pb2.PlaintextAck(
                accepted=False,
                message="Plaintext aggregation disabled",
                aggregation_complete=False,
                submissions=len(self._plaintext_updates),
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )
        if not self._validate_participant(node_id):
            logger.warning("Plaintext submission from non-member %s", node_id)
            return secureagg_pb2.PlaintextAck(
                accepted=False,
                message="Node not part of clique",
                aggregation_complete=False,
                submissions=len(self._plaintext_updates),
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_NOT_MEMBER,
            )
        if not request.model_weights:
            return secureagg_pb2.PlaintextAck(
                accepted=False,
                message="Missing model weights",
                aggregation_complete=False,
                submissions=len(self._plaintext_updates),
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )

        vector = [float(v) for v in request.model_weights]
        aggregation_complete = False
        submissions = 0
        try:
            with self._plaintext_lock:
                if self._plaintext_vector_length is None:
                    self._plaintext_vector_length = len(vector)
                elif len(vector) != self._plaintext_vector_length:
                    raise ValueError(
                        f"Vector length mismatch: expected {self._plaintext_vector_length}, got {len(vector)}"
                    )
                self._plaintext_updates[node_id] = vector
                submissions = len(self._plaintext_updates)
                logger.info(
                    "Received plaintext model from %s (%d/%d)",
                    node_id,
                    submissions,
                    self._active_threshold,
                )
                aggregation_complete = self._finalize_plaintext_aggregation_locked()
        except ValueError as exc:
            logger.warning("Plaintext submission invalid for %s: %s", node_id, exc)
            return secureagg_pb2.PlaintextAck(
                accepted=False,
                message=str(exc),
                aggregation_complete=False,
                submissions=submissions,
                server_round=server_round,
                sync_code=secureagg_pb2.ROUND_SYNC_OK,
            )

        message = "Update accepted"
        if aggregation_complete:
            message = "Aggregation complete"

        return secureagg_pb2.PlaintextAck(
            accepted=True,
            message=message,
            aggregation_complete=aggregation_complete,
            submissions=submissions,
            server_round=server_round,
            sync_code=secureagg_pb2.ROUND_SYNC_OK,
        )

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
        self._round2_waiting_on_aggregator = False
        self._round2_timed_out = False
        self._round2_opened_at = time.monotonic()
        self._round0_opened_at = time.monotonic()
        self._round4_computing = False
        self.current_round = round_idx
        self._plaintext_updates.clear()
        self._plaintext_vector_length = None
        self._plaintext_round_opened_at = time.monotonic()
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
    timeout_seconds: float = 0.0,
    plaintext_mode: bool = False,
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
        timeout_seconds=timeout_seconds,
        plaintext_mode=plaintext_mode,
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
