import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from secure_aggregation.config.models import NodeRole
from secure_aggregation.protocol import (
    Round1Ciphertext,
    SecureAggregationAggregator,
    SecureAggregationConfig,
    SecureAggregationNode,
)


@dataclass(frozen=True)
class ReliabilityScore:
    uptime: float
    bandwidth: float
    latency: float

    def score(self) -> float:
        # Higher uptime/bandwidth increase score; higher latency penalizes.
        return self.uptime + self.bandwidth - self.latency


@dataclass(frozen=True)
class NodeRuntimeConfig:
    node_id: str
    role: NodeRole
    reliability: ReliabilityScore
    freshness_window: float = 10.0
    decay_tau: float = 5.0
    gossip_degree: int = 2
    cache_size: int = 3


@dataclass
class ModelSnapshot:
    source_id: str
    vector: List[float]
    timestamp: float


class GossipCache:
    """Stores freshest remote snapshots up to capacity."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._snapshots: List[ModelSnapshot] = []

    def add_snapshot(self, snapshot: ModelSnapshot) -> None:
        self._snapshots.append(snapshot)
        self._snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        self._snapshots = self._snapshots[: self.capacity]

    def fresh_snapshots(self, now_ts: float, freshness_window: float) -> List[ModelSnapshot]:
        return [s for s in self._snapshots if now_ts - s.timestamp <= freshness_window]


class NodeEngine:
    def __init__(self, config: NodeRuntimeConfig) -> None:
        self.config = config
        self.gossip_cache = GossipCache(config.cache_size)

    @property
    def reliability_score(self) -> float:
        return self.config.reliability.score()

    def add_remote_snapshot(self, vector: List[float], timestamp: Optional[float] = None, source_id: Optional[str] = None) -> None:
        ts = time.time() if timestamp is None else timestamp
        src = source_id or self.config.node_id
        self.gossip_cache.add_snapshot(ModelSnapshot(src, vector, ts))

    def merge_with_remote(self, local_mean: List[float], now_ts: Optional[float] = None) -> List[float]:
        now = time.time() if now_ts is None else now_ts
        fresh = self.gossip_cache.fresh_snapshots(now, self.config.freshness_window)
        if not fresh:
            return list(local_mean)
        weights = [math.exp(-(now - s.timestamp) / self.config.decay_tau) for s in fresh]
        weight_sum = 1.0 + sum(weights)  # include local_mean weight
        merged: List[float] = []
        for i in range(len(local_mean)):
            accumulator = local_mean[i]
            for w, snap in zip(weights, fresh):
                accumulator += w * snap.vector[i]
            merged.append(accumulator / weight_sum)
        return merged

    @staticmethod
    def select_aggregator(engines: Sequence["NodeEngine"], window_index: int) -> str:
        eligible = [e for e in engines if e.config.role in {NodeRole.TRAINER, NodeRole.AGGREGATOR, NodeRole.HYBRID}]
        if not eligible:
            raise ValueError("No eligible nodes for aggregation")
        ordered = sorted(eligible, key=lambda e: (-e.reliability_score, e.config.node_id))
        return ordered[window_index % len(ordered)].config.node_id

    @staticmethod
    def orchestrate_window(
        engines: Sequence["NodeEngine"],
        model_vectors: Mapping[str, List[int]],
        threshold: int,
        window_index: int,
        dropouts: Optional[Iterable[str]] = None,
    ):
        dropouts = set(dropouts or [])
        participants = list(model_vectors.keys())
        nodes = {pid: SecureAggregationNode(pid) for pid in participants}
        signing_keys = {pid: node.signing_public for pid, node in nodes.items()}
        aggregator = SecureAggregationAggregator(SecureAggregationConfig(participants, threshold), signing_keys)
        aggregator_id = NodeEngine.select_aggregator(engines, window_index)
        if aggregator_id not in participants:
            raise ValueError("Selected aggregator is not part of participants for this window")
        # Round 0
        adverts = [node.advertise_keys() for node in nodes.values()]
        aggregator.receive_advertisements(adverts)
        broadcast = aggregator.broadcast_advertisements()
        for node in nodes.values():
            node.receive_advertisements(broadcast)
        # Round 1
        for node in nodes.values():
            cts = node.create_round1_ciphertexts(participants, threshold)
            aggregator.receive_round1_ciphertexts(cts)
        for node in nodes.values():
            inbox = aggregator.deliver_round1_ciphertexts(node.node_id)
            node.receive_round1_ciphertexts(inbox)
        # Round 2
        for pid, node in nodes.items():
            if pid in dropouts:
                continue
            masked = node.create_masked_input(model_vectors[pid])
            aggregator.receive_masked_input(masked)
        survivors = aggregator.broadcast_survivors()
        # Round 3
        signatures = [nodes[pid].sign_survivor_list(survivors) for pid in survivors]
        aggregator.verify_survivor_signatures(signatures)
        # Round 4
        unmask_payloads = [nodes[pid].prepare_unmasking_payload(dropouts, survivors) for pid in survivors]
        result = aggregator.receive_unmasking_shares(unmask_payloads)
        return aggregator_id, result
