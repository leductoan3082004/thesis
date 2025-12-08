"""End-to-end happy-path test for AggregatorServicer wrapping SecureAggregationAggregator."""

from __future__ import annotations

from typing import Dict, List

import pytest

from secure_aggregation.communication import secureagg_pb2
from secure_aggregation.communication.aggregator_service import AggregatorServicer
from secure_aggregation.crypto.sign import generate_signing_keypair
from secure_aggregation.protocol import Round1Ciphertext, SecureAggregationConfig, SecureAggregationNode
from secure_aggregation.protocol.core import SHARE_BYTES, AdvertiseMessage, _int_to_bytes


def _encode_share(x: int, share: int) -> bytes:
    return _int_to_bytes(x, 2) + _int_to_bytes(share, SHARE_BYTES)


def _build_advert(node_id: str, advert: AdvertiseMessage) -> secureagg_pb2.KeyAdvertisement:
    return secureagg_pb2.KeyAdvertisement(
        node_id=node_id,
        c_public_key=advert.c_public,
        s_public_key=advert.s_public,
        signature=advert.signature,
    )


def _round_trip_servicer(num_clients: int = 3, threshold: int = 2) -> List[float]:
    participant_ids = [f"u{i}" for i in range(1, num_clients + 1)]
    nodes: Dict[str, SecureAggregationNode] = {}
    for pid in participant_ids:
        pair = generate_signing_keypair()
        nodes[pid] = SecureAggregationNode(pid, signing_private=pair.private_key, signing_public=pair.public_key)
    signing_keys = {pid: node.signing_public for pid, node in nodes.items()}
    servicer = AggregatorServicer(
        node_id="agg",
        threshold=threshold,
        participant_ids=participant_ids,
        signing_public_keys=signing_keys,
    )

    # Round 0
    adverts = {pid: node.advertise_keys() for pid, node in nodes.items()}
    for pid, advert in adverts.items():
        servicer.Round0AdvertiseKeys(_build_advert(pid, advert), None)
    ordered = list(adverts.keys())
    ack = servicer.Round0AdvertiseKeys(_build_advert(ordered[0], adverts[ordered[0]]), None)
    # Retry until broadcast is available (once threshold reached and committed)
    while not ack.all_keys:
        ack = servicer.Round0AdvertiseKeys(_build_advert(ordered[0], adverts[ordered[0]]), None)
    broadcast = [
        AdvertiseMessage(
            node_id=a.node_id,
            c_public=bytes(a.c_public_key),
            s_public=bytes(a.s_public_key),
            signature=bytes(a.signature),
            signing_public=None,
        )
        for a in ack.all_keys
    ]
    for node in nodes.values():
        node.receive_advertisements(broadcast)

    # Round 1
    for pid, node in nodes.items():
        cts = node.create_round1_ciphertexts(ordered, threshold)
        req = secureagg_pb2.ShareKeysMessage(
            node_id=pid,
            ciphertexts=[
                secureagg_pb2.Round1Ciphertext(
                    sender_id=ct.sender_id, recipient_id=ct.recipient_id, iv=ct.iv, ciphertext=ct.ciphertext, tag=ct.tag
                )
                for ct in cts
            ],
        )
        resp = servicer.Round1ShareKeys(req, None)
        mailbox = [
            Round1Ciphertext(
                sender_id=m.sender_id,
                recipient_id=m.recipient_id,
                iv=bytes(m.iv),
                ciphertext=bytes(m.ciphertext),
                tag=bytes(m.tag),
            )
            for m in resp.mailbox
        ]
        if mailbox:
            node.receive_round1_ciphertexts(mailbox)

    # Deliver remaining mailboxes so every node gets peers' shares
    for pid, node in nodes.items():
        resp = servicer.Round1ShareKeys(secureagg_pb2.ShareKeysMessage(node_id=pid), None)
        inbox = [
            Round1Ciphertext(
                sender_id=m.sender_id,
                recipient_id=m.recipient_id,
                iv=bytes(m.iv),
                ciphertext=bytes(m.ciphertext),
                tag=bytes(m.tag),
            )
            for m in resp.mailbox
        ]
        if inbox:
            node.receive_round1_ciphertexts(inbox)

    # Round 2
    model_vectors = {pid: [i + 1, i + 2] for i, pid in enumerate(ordered)}
    for pid, node in nodes.items():
        masked = node.create_masked_input(model_vectors[pid])
        servicer.Round2MaskedInput(
            secureagg_pb2.MaskedInputMessage(
                node_id=pid,
                masked_vector=[_int_to_bytes(v, SHARE_BYTES) for v in masked.masked_vector],
            ),
            None,
        )
    survivors = servicer.aggregator.broadcast_survivors()

    # Round 3
    for pid in survivors:
        sig = nodes[pid].sign_survivor_list(survivors)
        servicer.Round3ConsistencyCheck(
            secureagg_pb2.ConsistencySignature(node_id=pid, signature=sig.signature),
            None,
        )

    # Round 4
    dropouts = set(participant_ids) - set(survivors)
    for pid in survivors:
        unmask = nodes[pid].prepare_unmasking_payload(dropouts, survivors)
        resp = servicer.Round4Unmask(
            secureagg_pb2.UnmaskShares(
                node_id=pid,
                dropout_s_shares={k: _encode_share(x, s) for k, (x, s) in unmask.s_shares_for_dropouts.items()},
                survivor_b_shares={k: _encode_share(x, b) for k, (x, b) in unmask.b_shares_for_survivors.items()},
            ),
            None,
        )
        if resp.aggregation_complete:
            break

    assert servicer.aggregated_result is not None
    return servicer.aggregated_result


def test_servicer_happy_path_end_to_end() -> None:
    mean = _round_trip_servicer(num_clients=3, threshold=2)
    # model vectors were [1,2], [2,3], [3,4] -> mean = [2, 3]
    assert pytest.approx(mean[0]) == 2.0
    assert pytest.approx(mean[1]) == 3.0
