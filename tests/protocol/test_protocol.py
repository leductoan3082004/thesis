import copy

import pytest

from secure_aggregation.protocol import (
    Round1Ciphertext,
    SecureAggregationAggregator,
    SecureAggregationConfig,
    SecureAggregationNode,
)


def _setup_session(num_clients: int = 4, threshold: int = 3):
    participant_ids = [f"u{i+1}" for i in range(num_clients)]
    nodes = {pid: SecureAggregationNode(pid) for pid in participant_ids}
    signing_keys = {pid: node.signing_public for pid, node in nodes.items()}
    agg = SecureAggregationAggregator(SecureAggregationConfig(participant_ids, threshold), signing_keys)
    # Round 0
    adverts = [node.advertise_keys() for node in nodes.values()]
    agg.receive_advertisements(adverts)
    broadcast = agg.broadcast_advertisements()
    for node in nodes.values():
        node.receive_advertisements(broadcast)
    # Round 1
    for node in nodes.values():
        cts = node.create_round1_ciphertexts(participant_ids, threshold)
        agg.receive_round1_ciphertexts(cts)
    for node in nodes.values():
        inbox = agg.deliver_round1_ciphertexts(node.node_id)
        node.receive_round1_ciphertexts(inbox)
    return agg, nodes


def _finish_protocol(agg, nodes, model_vectors, senders=None):
    if senders is None:
        senders = list(nodes.keys())
    # Round 2: masked inputs
    for pid in senders:
        masked = nodes[pid].create_masked_input(model_vectors[pid])
        agg.receive_masked_input(masked)
    survivors = agg.broadcast_survivors()
    # Round 3: survivor signatures
    signatures = [nodes[pid].sign_survivor_list(survivors) for pid in survivors]
    agg.verify_survivor_signatures(signatures)
    dropouts = [p for p in agg.participants if p not in survivors]
    # Round 4: unmasking shares (survivors respond)
    unmask_payloads = [nodes[pid].prepare_unmasking_payload(dropouts, survivors) for pid in survivors]
    return agg.receive_unmasking_shares(unmask_payloads)


def test_happy_path_end_to_end() -> None:
    agg, nodes = _setup_session(num_clients=4, threshold=3)
    model_vectors = {
        "u1": [1, 2, 3],
        "u2": [4, 5, 6],
        "u3": [7, 8, 9],
        "u4": [10, 11, 12],
    }
    result = _finish_protocol(agg, nodes, model_vectors)
    assert result.survivors == ["u1", "u2", "u3", "u4"]
    assert result.aggregate_sum == [sum(v[i] for v in model_vectors.values()) for i in range(3)]
    assert result.aggregate_mean == [s / 4 for s in result.aggregate_sum]


def test_dropout_handled_with_mask_removal() -> None:
    agg, nodes = _setup_session(num_clients=4, threshold=3)
    model_vectors = {
        "u1": [1, 1, 1],
        "u2": [2, 2, 2],
        "u3": [3, 3, 3],
        "u4": [100, 100, 100],  # dropout (masked input not sent)
    }
    senders = ["u1", "u2", "u3"]  # u4 drops after Round 1
    result = _finish_protocol(agg, nodes, model_vectors, senders=senders)
    assert result.survivors == senders
    assert result.aggregate_sum == [6, 6, 6]


def test_invalid_survivor_signature_aborts() -> None:
    agg, nodes = _setup_session(num_clients=4, threshold=3)
    model_vectors = {pid: [i + 1] for i, pid in enumerate(nodes)}
    # Round 2
    for pid in nodes:
        agg.receive_masked_input(nodes[pid].create_masked_input(model_vectors[pid]))
    survivors = agg.broadcast_survivors()
    signatures = [nodes[pid].sign_survivor_list(survivors) for pid in survivors]
    # Tamper one signature
    bad = copy.copy(signatures[0])
    bad.signature = b"\x00" * len(bad.signature)
    signatures[0] = bad
    with pytest.raises(ValueError):
        agg.verify_survivor_signatures(signatures)


def test_tampered_ciphertext_detected() -> None:
    participant_ids = ["u1", "u2", "u3"]
    nodes = {pid: SecureAggregationNode(pid) for pid in participant_ids}
    signing_keys = {pid: node.signing_public for pid, node in nodes.items()}
    agg = SecureAggregationAggregator(SecureAggregationConfig(participant_ids, 2), signing_keys)
    adverts = [node.advertise_keys() for node in nodes.values()]
    agg.receive_advertisements(adverts)
    broadcast = agg.broadcast_advertisements()
    for node in nodes.values():
        node.receive_advertisements(broadcast)
    for node in nodes.values():
        cts = node.create_round1_ciphertexts(participant_ids, threshold=2)
        agg.receive_round1_ciphertexts(cts)
    inbox = agg.deliver_round1_ciphertexts("u2")
    tampered_msg = Round1Ciphertext(
        sender_id=inbox[0].sender_id,
        recipient_id=inbox[0].recipient_id,
        iv=inbox[0].iv,
        ciphertext=inbox[0].ciphertext,
        tag=b"\x00" * len(inbox[0].tag),
    )
    with pytest.raises(Exception):
        nodes["u2"].receive_round1_ciphertexts([tampered_msg])
