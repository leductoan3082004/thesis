import pytest
import numpy as np

from collections import OrderedDict

from secure_aggregation.communication.hierarchy_mixin import HierarchyMixin, ScopeRoundHandler
from secure_aggregation.node.ecm_buffer import ECM, ECMBuffer
from secure_aggregation.state import HierarchyLevelConfig
from secure_aggregation.state.aggregation import StateAggregator
from secure_aggregation.storage.model_store import ModelAnchor, compute_model_hash


class FakeIPFS:
    def __init__(self) -> None:
        self._storage: dict[str, np.ndarray] = {}

    def add(self, model: np.ndarray) -> str:
        cid = compute_model_hash(model)
        self._storage[cid] = model.copy()
        return cid

    def get(self, cid: str) -> np.ndarray:
        return self._storage[cid].copy()


class FakeBlockchain:
    def __init__(self) -> None:
        self._anchors: dict[str, dict[str, dict[int, ModelAnchor]]] = {}

    def anchor(self, scope_id: str, round_num: int, cid: str, hash_val: str, *, scope: str) -> str:
        scope_store = self._anchors.setdefault(scope, {}).setdefault(scope_id, {})
        anchor = ModelAnchor(cluster_id=scope_id, round_num=round_num, cid=cid, hash=hash_val)
        scope_store[round_num] = anchor
        return f"{scope}::{scope_id}::{round_num}"

    def get_anchor(self, scope_id: str, round_num: int, *, scope: str, suppress_not_found_log: bool = False):
        store = self._anchors.get(scope, {}).get(scope_id, {})
        anchor = store.get(round_num)
        if anchor:
            return (anchor.cid, anchor.hash)
        return None

    def get_latest_scope_model(self, scope_name: str, scope_id: str) -> ModelAnchor | None:
        store = self._anchors.get(scope_name.lower(), {}).get(scope_id, {})
        if not store:
            return None
        latest_round = max(store.keys())
        return store[latest_round]


class DummyBridgeClient:
    def wait_for_ready(self, address: str, timeout: float = 2.0) -> bool:
        return True

    def broadcast_ecm(self, *args, **kwargs) -> int:  # pragma: no cover - broadcast path mocked
        return 1


class TestHierarchyNode(HierarchyMixin):
    def __init__(self) -> None:
        self.node_id = "trainer-node-001"
        self.scope_name = "state"
        self.scope_config = HierarchyLevelConfig(
            enabled=True,
            scope_index=1,
            scope_name="state",
            scope_id="state_alpha",
            interval_seconds=10,
            wait_seconds=5,
        )
        self.higher_scope_name = "nation"
        self.higher_scope_config = HierarchyLevelConfig(
            enabled=True,
            scope_index=2,
            scope_name="nation",
            scope_id="nation_0",
            interval_seconds=30,
            wait_seconds=10,
        )
        self.scope_configs = OrderedDict(
            (
                ("state", self.scope_config),
                ("nation", self.higher_scope_config),
            )
        )
        self.inter_cluster_config = {"freshness_window": 30.0}
        self.training_config = {}
        self.system_config = {}
        self.participant_map = {
            "trainer-node-001": "127.0.0.1:5000",
            "trainer-node-002": "127.0.0.1:5001",
        }
        self.central_metadata = None
        self.bridge_client = DummyBridgeClient()
        self.ipfs = FakeIPFS()
        self.blockchain = FakeBlockchain()
        self.model = object()
        self._applied_tensors: list[np.ndarray] = []
        self._scope_runtimes = {}

    # Override helpers to keep the test deterministic.
    def _scope_member_roster(self, scope_name: str, scope_id: str | None) -> list[str]:
        if scope_name.lower() == "state":
            return ["trainer-node-001", "trainer-node-002"]
        return []

    def _node_scope_membership_map(self):
        return {"state": "state_alpha", "nation": "nation_0", "cluster": "cluster_0"}

    def _child_scope_ids_for_runtime(self, runtime):
        if runtime.scope_name.lower() == "state":
            return ["state::cluster_alpha", "state::cluster_beta"]
        return []

    def _apply_model_tensor(self, tensor: np.ndarray) -> None:
        self._applied_tensors.append(tensor.copy())

    def _prime_convergence_tracker_state(self) -> None:
        """No-op for tests."""
        return None


@pytest.mark.usefixtures("tmp_path")
def test_state_hierarchy_full_flow() -> None:
    """End-to-end sanity test of the hierarchy flow using deterministic stubs."""

    node = TestHierarchyNode()
    runtime = node._ensure_scope_runtime(node.scope_name, node.scope_config)
    runtime.scope_id = node.scope_config.scope_id
    runtime.candidates = ["trainer-node-001", "trainer-node-002"]
    runtime.is_candidate = True
    runtime.ecm_buffer = ECMBuffer(freshness_window=60.0)
    runtime.aggregator = StateAggregator(node.scope_config, node.ipfs, node.blockchain)

    # Step 1: Seed cluster ECMs so the state aggregator has fresh inputs to merge.
    cluster_a = np.array([1.0, 2.0], dtype=np.float32)
    cluster_b = np.array([3.0, 4.0], dtype=np.float32)
    cid_a = node.ipfs.add(cluster_a)
    cid_b = node.ipfs.add(cluster_b)
    runtime.ecm_buffer.add(ECM(cid=cid_a, hash=compute_model_hash(cluster_a), source_cluster="state::cluster_alpha"))
    runtime.ecm_buffer.add(ECM(cid=cid_b, hash=compute_model_hash(cluster_b), source_cluster="state::cluster_beta"))

    # Step 2: Execute a state round and ensure the elected aggregator publishes the merged model.
    handler = ScopeRoundHandler(
        scope_name=node.scope_name,
        config=node.scope_config,
        trigger_label="cluster",
        round_queue=runtime.round_queue,
        rounds_logged=runtime.rounds_logged,
        round_cache=runtime.round_cache,
        round_hashes=runtime.round_hashes,
        committed_rounds=runtime.committed_rounds,
        is_candidate_fn=lambda: runtime.is_candidate,
        dispatch_fn=lambda scope_round, parent_round: None,
        execute_fn=lambda scope_round, parent_round: node._execute_scope_round(scope_round, parent_round, runtime),
    )
    node._register_scope_round_handler(handler)
    runtime.round_queue.append((1, 0))
    result = node._run_next_scope_round()
    assert result == ("state", 1, 0)
    assert 1 in runtime.committed_rounds, "Aggregator should mark the round committed after publishing."

    # Step 3: Pull the latest state model via the blockchain endpoint and apply it locally.
    node._apply_scope_model_from_anchor("state", node.scope_config, is_local_scope=True)
    assert node._applied_tensors, "Applying the anchored model should record the tensor payload."
    applied = node._applied_tensors[-1]
    expected = np.vstack([cluster_a, cluster_b]).mean(axis=0)
    np.testing.assert_allclose(applied, expected, err_msg="State model should equal the average of cluster contributions.")
