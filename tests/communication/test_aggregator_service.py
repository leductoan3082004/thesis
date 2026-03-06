"""End-to-end happy-path test for AggregatorServicer wrapping SecureAggregationAggregator."""

from __future__ import annotations

import time
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


def _prepare_servicer_ready_for_round2(
    num_clients: int = 3, threshold: int = 3, timeout: float = 0.1
) -> tuple[AggregatorServicer, Dict[str, SecureAggregationNode], List[str]]:
    participant_ids = [f"u{i}" for i in range(1, num_clients + 1)]
    nodes: Dict[str, SecureAggregationNode] = {}
    signing_keys: Dict[str, bytes] = {}
    for pid in participant_ids:
        pair = generate_signing_keypair()
        node = SecureAggregationNode(pid, signing_private=pair.private_key, signing_public=pair.public_key)
        nodes[pid] = node
        signing_keys[pid] = node.signing_public
    aggregator_id = participant_ids[0]
    servicer = AggregatorServicer(
        node_id=aggregator_id,
        threshold=threshold,
        participant_ids=participant_ids,
        signing_public_keys=signing_keys,
        timeout_seconds=timeout,
    )
    adverts = {pid: node.advertise_keys() for pid, node in nodes.items()}
    for pid, advert in adverts.items():
        servicer.Round0AdvertiseKeys(_build_advert(pid, advert), None)
    ack = servicer.Round0AdvertiseKeys(_build_advert(aggregator_id, adverts[aggregator_id]), None)
    while not ack.all_keys:
        ack = servicer.Round0AdvertiseKeys(_build_advert(aggregator_id, adverts[aggregator_id]), None)
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
    ordered = participant_ids
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
    # Deliver remaining mailboxes
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
    return servicer, nodes, ordered


# ── Round-sync contract tests ──────────────────────────────────────────────────


def _make_servicer(participant_ids: List[str], threshold: int = 2) -> AggregatorServicer:
    return AggregatorServicer(
        node_id="agg",
        threshold=threshold,
        participant_ids=participant_ids,
        signing_public_keys=None,
    )


def test_round0_stale_request_returns_sync_stale() -> None:
    servicer = _make_servicer(["u1", "u2", "u3"])
    servicer.prepare_round(5)
    pair = generate_signing_keypair()
    node = SecureAggregationNode("u1", signing_private=pair.private_key, signing_public=pair.public_key)
    advert = node.advertise_keys()
    req = secureagg_pb2.KeyAdvertisement(
        node_id="u1",
        c_public_key=advert.c_public,
        s_public_key=advert.s_public,
        signature=advert.signature,
        round=0,  # stale: server is on round 5
    )
    ack = servicer.Round0AdvertiseKeys(req, None)
    assert ack.accepted is False
    assert ack.sync_code == secureagg_pb2.ROUND_SYNC_STALE
    assert ack.server_round == 5


def test_round0_ahead_request_returns_sync_ahead() -> None:
    servicer = _make_servicer(["u1", "u2", "u3"])
    # server is at round 0 by default
    pair = generate_signing_keypair()
    node = SecureAggregationNode("u1", signing_private=pair.private_key, signing_public=pair.public_key)
    advert = node.advertise_keys()
    req = secureagg_pb2.KeyAdvertisement(
        node_id="u1",
        c_public_key=advert.c_public,
        s_public_key=advert.s_public,
        signature=advert.signature,
        round=10,  # ahead of server
    )
    ack = servicer.Round0AdvertiseKeys(req, None)
    assert ack.accepted is False
    assert ack.sync_code == secureagg_pb2.ROUND_SYNC_AHEAD


def test_round0_broadcast_identical_for_all_nodes() -> None:
    participants = ["u1", "u2", "u3"]
    signing_keys = {}
    nodes = {}
    for pid in participants:
        pair = generate_signing_keypair()
        nodes[pid] = SecureAggregationNode(pid, signing_private=pair.private_key, signing_public=pair.public_key)
        signing_keys[pid] = pair.public_key
    servicer = AggregatorServicer(
        node_id="agg",
        threshold=2,
        participant_ids=participants,
        signing_public_keys=signing_keys,
    )
    adverts = {pid: nodes[pid].advertise_keys() for pid in nodes}
    for pid, advert in adverts.items():
        servicer.Round0AdvertiseKeys(
            secureagg_pb2.KeyAdvertisement(
                node_id=pid,
                c_public_key=advert.c_public,
                s_public_key=advert.s_public,
                signature=advert.signature,
                round=0,
            ),
            None,
        )
    ack1 = servicer.Round0AdvertiseKeys(
        secureagg_pb2.KeyAdvertisement(
            node_id="u1",
            c_public_key=adverts["u1"].c_public,
            s_public_key=adverts["u1"].s_public,
            signature=adverts["u1"].signature,
            round=0,
        ),
        None,
    )
    ack2 = servicer.Round0AdvertiseKeys(
        secureagg_pb2.KeyAdvertisement(
            node_id="u2",
            c_public_key=adverts["u2"].c_public,
            s_public_key=adverts["u2"].s_public,
            signature=adverts["u2"].signature,
            round=0,
        ),
        None,
    )
    assert ack1.all_keys == ack2.all_keys


def test_round2_shortfall_triggers_abort_and_retry_signal() -> None:
    servicer, nodes, participants = _prepare_servicer_ready_for_round2(threshold=3, timeout=0.1)
    model_vectors = {pid: [i + 1, i + 2] for i, pid in enumerate(participants)}
    masked_msgs: Dict[str, secureagg_pb2.MaskedInputMessage] = {}
    for pid in participants[:2]:
        masked = nodes[pid].create_masked_input(model_vectors[pid])
        msg = secureagg_pb2.MaskedInputMessage(
            node_id=pid, masked_vector=[_int_to_bytes(v, SHARE_BYTES) for v in masked.masked_vector]
        )
        masked_msgs[pid] = msg
        servicer.Round2MaskedInput(msg, None)
    # Force timeout expiration
    servicer._round2_opened_at = time.monotonic() - servicer._timeout_seconds - 1.0
    resp = servicer.Round2MaskedInput(masked_msgs[participants[0]], None)
    assert resp.accepted is False
    assert resp.sync_code == secureagg_pb2.ROUND_SYNC_AHEAD
    assert servicer._round2_failed is True
    resp3 = servicer.Round3ConsistencyCheck(
        secureagg_pb2.ConsistencySignature(node_id=participants[0], signature=b"sig"),
        None,
    )
    assert resp3.sync_code == secureagg_pb2.ROUND_SYNC_AHEAD


def test_round0_not_member_returns_not_member() -> None:
    servicer = _make_servicer(["u1", "u2"])
    pair = generate_signing_keypair()
    node = SecureAggregationNode("stranger", signing_private=pair.private_key, signing_public=pair.public_key)
    advert = node.advertise_keys()
    req = secureagg_pb2.KeyAdvertisement(
        node_id="stranger",
        c_public_key=advert.c_public,
        s_public_key=advert.s_public,
        signature=advert.signature,
        round=0,
    )
    ack = servicer.Round0AdvertiseKeys(req, None)
    assert ack.accepted is False
    assert ack.sync_code == secureagg_pb2.ROUND_SYNC_NOT_MEMBER


def test_round0_happy_path_sync_ok_and_server_round() -> None:
    participant_ids = ["u1", "u2", "u3"]
    nodes: Dict[str, SecureAggregationNode] = {}
    for pid in participant_ids:
        pair = generate_signing_keypair()
        nodes[pid] = SecureAggregationNode(pid, signing_private=pair.private_key, signing_public=pair.public_key)
    servicer = AggregatorServicer(
        node_id="agg",
        threshold=2,
        participant_ids=participant_ids,
        signing_public_keys={pid: node.signing_public for pid, node in nodes.items()},
    )
    advert = nodes["u1"].advertise_keys()
    req = secureagg_pb2.KeyAdvertisement(
        node_id="u1",
        c_public_key=advert.c_public,
        s_public_key=advert.s_public,
        signature=advert.signature,
        round=0,
    )
    ack = servicer.Round0AdvertiseKeys(req, None)
    assert ack.sync_code in (secureagg_pb2.ROUND_SYNC_OK, secureagg_pb2.ROUND_SYNC_UNSPECIFIED)
    assert ack.server_round == 0


def test_round1_stale_returns_sync_stale() -> None:
    servicer = _make_servicer(["u1", "u2", "u3"])
    servicer.prepare_round(3)
    req = secureagg_pb2.ShareKeysMessage(node_id="u1", ciphertexts=[], round=1)
    ack = servicer.Round1ShareKeys(req, None)
    assert ack.sync_code == secureagg_pb2.ROUND_SYNC_STALE
    assert ack.server_round == 3


def test_round2_stale_returns_sync_stale() -> None:
    servicer = _make_servicer(["u1", "u2", "u3"])
    servicer.prepare_round(4)
    req = secureagg_pb2.MaskedInputMessage(node_id="u1", masked_vector=[], round=2)
    ack = servicer.Round2MaskedInput(req, None)
    assert ack.sync_code == secureagg_pb2.ROUND_SYNC_STALE
    assert ack.server_round == 4


def test_round3_non_survivor_gets_finalized() -> None:
    servicer = _make_servicer(["u1", "u2", "u3"])
    # Simulate Round 2 finalization with u1 and u2 as survivors (u3 dropped)
    servicer._round2_finalized = True
    servicer._round2_survivors = ["u1", "u2"]
    servicer.aggregator._survivors = ["u1", "u2"]
    req = secureagg_pb2.ConsistencySignature(node_id="u3", signature=b"sig", round=0)
    ack = servicer.Round3ConsistencyCheck(req, None)
    assert ack.accepted is False
    assert ack.sync_code == secureagg_pb2.ROUND_SYNC_FINALIZED


def test_round4_non_survivor_gets_finalized() -> None:
    servicer = _make_servicer(["u1", "u2", "u3"])
    servicer._round2_finalized = True
    servicer._round2_survivors = ["u1", "u2"]
    servicer.aggregator._survivors = ["u1", "u2"]
    req = secureagg_pb2.UnmaskShares(node_id="u3", round=0)
    ack = servicer.Round4Unmask(req, None)
    assert ack.accepted is False
    assert ack.sync_code == secureagg_pb2.ROUND_SYNC_FINALIZED


def test_happy_path_responses_have_sync_ok() -> None:
    """Existing happy-path round trip produces ROUND_SYNC_OK in acks."""
    participant_ids = ["u1", "u2", "u3"]
    nodes: Dict[str, SecureAggregationNode] = {}
    for pid in participant_ids:
        pair = generate_signing_keypair()
        nodes[pid] = SecureAggregationNode(pid, signing_private=pair.private_key, signing_public=pair.public_key)
    servicer = AggregatorServicer(
        node_id="agg",
        threshold=2,
        participant_ids=participant_ids,
        signing_public_keys={pid: node.signing_public for pid, node in nodes.items()},
    )
    # Round 0 with round=0
    for pid, node in nodes.items():
        advert = node.advertise_keys()
        ack = servicer.Round0AdvertiseKeys(
            secureagg_pb2.KeyAdvertisement(
                node_id=pid,
                c_public_key=advert.c_public,
                s_public_key=advert.s_public,
                signature=advert.signature,
                round=0,
            ),
            None,
        )
        assert ack.server_round == 0
        if ack.accepted:
            assert ack.sync_code == secureagg_pb2.ROUND_SYNC_OK
