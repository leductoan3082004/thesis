"""Aggregator service that coordinates secure aggregation protocol."""

import logging
from concurrent import futures
from typing import Dict, List, Optional

import grpc
from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.protocol import SecureAggregationAggregator, SecureAggregationConfig
from secure_aggregation.utils import get_logger

logger = get_logger("aggregator_service")


class AggregatorServicer(secureagg_pb2_grpc.AggregatorServiceServicer):
    """Aggregator that coordinates the 4-round secure aggregation protocol."""

    def __init__(self, node_id: str, threshold: int, participant_ids: List[str]) -> None:
        self.node_id = node_id
        self.threshold = threshold
        self.participant_ids = participant_ids

        # Protocol state storage
        self.round0_keys: Dict[str, tuple] = {}  # node_id -> (c_pk, s_pk, sig)
        self.round1_shares: Dict[str, Dict[str, bytes]] = {}  # node_id -> {recipient -> ciphertext}
        self.round2_masked: Dict[str, List[int]] = {}  # node_id -> masked_vector
        self.round3_signatures: Dict[str, bytes] = {}  # node_id -> signature
        self.round4_unmasking: Dict[str, tuple] = {}  # node_id -> (dropout_shares, survivor_shares)

        self.u1_list: List[tuple] = []  # List of (node_id, c_pk, s_pk, sig)
        self.u3_survivors: List[str] = []
        self.aggregated_result: Optional[List[float]] = None
        self.current_round = 0

        logger.info(f"Aggregator {node_id} initialized with threshold={threshold}, participants={len(participant_ids)}")

    def Round0AdvertiseKeys(self, request: secureagg_pb2.KeyAdvertisement, context) -> secureagg_pb2.KeyAdvertisementAck:
        """Collect DH public keys from participants (Round 0)."""
        node_id = request.node_id

        # If this is a duplicate but we already have all keys, return the full list
        if node_id in self.round0_keys:
            logger.warning(f"Duplicate key advertisement from {node_id}")
            # If we already have threshold participants, return the U1 list
            if len(self.round0_keys) >= self.threshold and self.u1_list:
                all_keys = [
                    secureagg_pb2.KeyAdvertisement(
                        node_id=nid, c_public_key=c_pk, s_public_key=s_pk, signature=sig
                    )
                    for nid, c_pk, s_pk, sig in self.u1_list
                ]
                logger.info(f"Round 0 complete: {len(self.u1_list)} participants")
                return secureagg_pb2.KeyAdvertisementAck(
                    accepted=True,
                    all_keys=all_keys,
                    message=f"Round 0 complete with {len(self.u1_list)} participants"
                )
            # Otherwise, still waiting for more participants
            return secureagg_pb2.KeyAdvertisementAck(accepted=True, message="Waiting for more participants")

        self.round0_keys[node_id] = (
            bytes(request.c_public_key),
            bytes(request.s_public_key),
            bytes(request.signature)
        )

        logger.info(f"Received keys from {node_id} ({len(self.round0_keys)}/{len(self.participant_ids)})")

        # Wait until we have threshold participants
        if len(self.round0_keys) >= self.threshold:
            # Build U1 list
            self.u1_list = [
                (nid, c_pk, s_pk, sig)
                for nid, (c_pk, s_pk, sig) in self.round0_keys.items()
            ]

            # Broadcast U1 to all
            all_keys = [
                secureagg_pb2.KeyAdvertisement(
                    node_id=nid, c_public_key=c_pk, s_public_key=s_pk, signature=sig
                )
                for nid, c_pk, s_pk, sig in self.u1_list
            ]

            logger.info(f"Round 0 complete: {len(self.u1_list)} participants")
            return secureagg_pb2.KeyAdvertisementAck(accepted=True, message="Round 0 complete", all_keys=all_keys)

        return secureagg_pb2.KeyAdvertisementAck(accepted=True, message="Waiting for more participants")

    def Round1ShareKeys(self, request: secureagg_pb2.ShareKeysMessage, context) -> secureagg_pb2.ShareKeysAck:
        """Collect encrypted secret shares (Round 1)."""
        node_id = request.node_id

        if node_id in self.round1_shares:
            logger.warning(f"Duplicate shares from {node_id}")
            return secureagg_pb2.ShareKeysAck(accepted=False, message="Duplicate shares")

        self.round1_shares[node_id] = dict(request.encrypted_shares)
        logger.info(f"Received shares from {node_id} ({len(self.round1_shares)}/{len(self.u1_list)})")

        if len(self.round1_shares) >= self.threshold:
            logger.info("Round 1 complete")
            return secureagg_pb2.ShareKeysAck(accepted=True, message="Round 1 complete")

        return secureagg_pb2.ShareKeysAck(accepted=True, message="Waiting for more shares")

    def Round2MaskedInput(self, request: secureagg_pb2.MaskedInputMessage, context) -> secureagg_pb2.MaskedInputAck:
        """Collect masked model updates (Round 2)."""
        node_id = request.node_id

        if node_id in self.round2_masked:
            logger.warning(f"Duplicate masked input from {node_id}")
            # If we already have threshold participants, return the survivors list
            if len(self.round2_masked) >= self.threshold and self.u3_survivors:
                logger.info(f"Round 2 complete: {len(self.u3_survivors)} survivors")
                return secureagg_pb2.MaskedInputAck(
                    accepted=True, message=f"Round 2 complete with {len(self.u3_survivors)} survivors", survivors=self.u3_survivors
                )
            # Otherwise, still waiting for more participants
            return secureagg_pb2.MaskedInputAck(accepted=True, message="Waiting for more inputs")

        self.round2_masked[node_id] = list(request.masked_vector)
        logger.info(f"Received masked input from {node_id} ({len(self.round2_masked)}/{len(self.u1_list)})")

        if len(self.round2_masked) >= self.threshold:
            # Determine survivors (U3)
            self.u3_survivors = list(self.round2_masked.keys())
            logger.info(f"Round 2 complete: {len(self.u3_survivors)} survivors")
            return secureagg_pb2.MaskedInputAck(
                accepted=True, message="Round 2 complete", survivors=self.u3_survivors
            )

        return secureagg_pb2.MaskedInputAck(accepted=True, message="Waiting for more inputs")

    def Round3ConsistencyCheck(self, request: secureagg_pb2.ConsistencySignature, context) -> secureagg_pb2.ConsistencyAck:
        """Collect consistency signatures (Round 3)."""
        node_id = request.node_id

        if node_id in self.round3_signatures:
            logger.warning(f"Duplicate signature from {node_id}")
            return secureagg_pb2.ConsistencyAck(accepted=False, message="Duplicate signature")

        self.round3_signatures[node_id] = bytes(request.signature)
        logger.info(f"Received signature from {node_id} ({len(self.round3_signatures)}/{len(self.u3_survivors)})")

        if len(self.round3_signatures) >= len(self.u3_survivors):
            logger.info("Round 3 complete")
            return secureagg_pb2.ConsistencyAck(accepted=True, message="Round 3 complete")

        return secureagg_pb2.ConsistencyAck(accepted=True, message="Waiting for more signatures")

    def Round4Unmask(self, request: secureagg_pb2.UnmaskShares, context) -> secureagg_pb2.UnmaskAck:
        """Collect unmasking shares and compute aggregate (Round 4)."""
        node_id = request.node_id

        if node_id in self.round4_unmasking:
            logger.warning(f"Duplicate unmask shares from {node_id}")
            # If aggregation is already complete, return success
            if len(self.round4_unmasking) >= self.threshold and self.aggregated_result is not None:
                logger.info("Round 4 complete - aggregation already done")
                return secureagg_pb2.UnmaskAck(
                    accepted=True, message="Aggregation complete", aggregation_complete=True
                )
            # Otherwise, still waiting for more participants
            return secureagg_pb2.UnmaskAck(accepted=True, message="Waiting for unmask shares", aggregation_complete=False)

        self.round4_unmasking[node_id] = (
            dict(request.dropout_s_shares),
            dict(request.survivor_b_shares)
        )
        logger.info(f"Received unmask shares from {node_id} ({len(self.round4_unmasking)}/{len(self.u3_survivors)})")

        if len(self.round4_unmasking) >= self.threshold:
            # Compute aggregate (simplified - just average the masked inputs)
            logger.info("Round 4 complete - computing aggregate")
            self.aggregated_result = self._compute_aggregate()
            return secureagg_pb2.UnmaskAck(
                accepted=True, message="Aggregation complete", aggregation_complete=True
            )

        return secureagg_pb2.UnmaskAck(accepted=True, message="Waiting for unmask shares", aggregation_complete=False)

    def GetGlobalModel(self, request: secureagg_pb2.ModelRequest, context) -> secureagg_pb2.ModelResponse:
        """Return the aggregated global model."""
        if self.aggregated_result is None:
            return secureagg_pb2.ModelResponse(model_weights=[], round=self.current_round, aggregator_id=self.node_id)

        logger.info(f"Serving global model (round {self.current_round})")
        return secureagg_pb2.ModelResponse(
            model_weights=self.aggregated_result,
            round=self.current_round,
            aggregator_id=self.node_id
        )

    def _compute_aggregate(self) -> List[float]:
        """Compute the aggregated model by averaging masked inputs."""
        if not self.round2_masked:
            return []

        # Get vector dimension
        first_vec = next(iter(self.round2_masked.values()))
        dim = len(first_vec)

        # Simple averaging (in reality, we'd unmask first)
        aggregate = [0.0] * dim
        for masked_vec in self.round2_masked.values():
            for i, val in enumerate(masked_vec):
                aggregate[i] += float(val)

        # Average
        num_participants = len(self.round2_masked)
        aggregate = [x / num_participants for x in aggregate]

        logger.info(f"Computed aggregate from {num_participants} participants")
        return aggregate

    def reset_for_next_round(self) -> None:
        """Reset state for next aggregation round."""
        self.round0_keys.clear()
        self.round1_shares.clear()
        self.round2_masked.clear()
        self.round3_signatures.clear()
        self.round4_unmasking.clear()
        self.u1_list.clear()
        self.u3_survivors.clear()
        self.aggregated_result = None
        self.current_round += 1
        logger.info(f"Reset for round {self.current_round}")


def serve(node_id: str, port: int, threshold: int, participant_ids: List[str]) -> grpc.Server:
    """Start the aggregator gRPC server."""
    servicer = AggregatorServicer(node_id, threshold, participant_ids)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    secureagg_pb2_grpc.add_AggregatorServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Aggregator server started on port {port}")
    return server
