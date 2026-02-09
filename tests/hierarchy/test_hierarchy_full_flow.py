import pytest
import numpy as np

from collections import OrderedDict, deque

from secure_aggregation.communication.hierarchy_mixin import HierarchyMixin, ScopeRoundHandler
from secure_aggregation.communication.node_service import NodeService
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

    # Step 3: Aggregator should apply the merged state model immediately after committing.
    assert node._applied_tensors, "Aggregator should apply its own merged state model without refetching."
    applied = node._applied_tensors[0]
    expected = np.vstack([cluster_a, cluster_b]).mean(axis=0)
    np.testing.assert_allclose(applied, expected, err_msg="State model should equal the average of cluster contributions.")
    ready_set = getattr(node, "_ready_scope_fetches", None) or set()
    assert "state" not in ready_set, "Aggregator should not schedule an extra fetch after committing."
    queue = getattr(node, "_pending_scope_waits", None) or deque()
    assert not queue, "Aggregator wait queue should be cleared after committing the state round."


def test_state_hierarchy_follower_waits_for_anchor(caplog) -> None:
    """Ensure non-leader nodes skip aggregation work and explicitly wait for anchors."""

    node = TestHierarchyNode()
    node.node_id = "trainer-node-002"
    runtime = node._ensure_scope_runtime(node.scope_name, node.scope_config)
    runtime.scope_id = node.scope_config.scope_id
    runtime.candidates = ["trainer-node-001", "trainer-node-002"]
    runtime.is_candidate = True
    runtime.ecm_buffer = ECMBuffer(freshness_window=60.0)
    runtime.aggregator = StateAggregator(node.scope_config, node.ipfs, node.blockchain)
    runtime.config.wait_seconds = 0.05

    caplog.set_level("INFO", logger="hierarchy")
    result = node._execute_scope_round(1, 0, runtime)
    assert result, "Follower nodes should proceed after waiting for the anchor."

    messages = [record.getMessage() for record in caplog.records]
    wait_logs = [msg for msg in messages if "aggregator=trainer-node-001" in msg and "waiting up to" in msg.lower()]
    assert wait_logs, "Follower should announce that it is waiting for the elected aggregator."
    assert all("collected" not in msg for msg in messages), "Follower must not log aggregation progress."


def test_drain_scope_rounds_executes_one_scope_per_call() -> None:
    """Ensure draining scope rounds processes scopes sequentially (state before nation)."""

    class DummyNode:
        def __init__(self) -> None:
            self.scope_name = "state"
            self._scope_execution_order = ["state", "nation"]
            self.state_results = [("state", 1, 0), None]
            self.nation_results = [("nation", 1, 0), None]
            self.completed: list[tuple[str, int, int]] = []

        def _run_next_scope_round(self, scope_name=None):
            if scope_name is None:
                return self.state_results.pop(0)
            return self.nation_results.pop(0)

        def _handle_completed_scope_round(self, scope_name, scope_round, source_round):
            self.completed.append((scope_name, scope_round, source_round))

    dummy = DummyNode()
    assert NodeService._drain_scope_rounds(dummy) is True, "State round should execute first."
    assert not dummy.completed, "State rounds do not invoke higher-scope completion handler."

    assert NodeService._drain_scope_rounds(dummy) is True, "Nation round should run on the next tick."
    assert dummy.completed == [("nation", 1, 0)], "Nation completion should be recorded once."

    assert NodeService._drain_scope_rounds(dummy) is False, "No additional scope rounds should be pending."


def test_state_model_applied_before_nation_round() -> None:
    """Verify that applying ready state models happens before the next nation round executes."""

    class ScopeFlowHarness:
        def __init__(self) -> None:
            self.scope_name = "state"
            self._scope_execution_order = ["state", "nation"]
            self.state_queue = [("state", 1, 0), None]
            self.nation_queue = [("nation", 1, 0), None]
            self.ready_fetch = False
            self.state_applied = False
            self.nation_started = False

        def _drain_scope_rounds(self):
            return NodeService._drain_scope_rounds(self)

        def _run_next_scope_round(self, scope_name=None):
            if scope_name is None:
                return self.state_queue.pop(0)
            assert scope_name == "nation"
            if not self.state_applied:
                raise AssertionError("Nation round started before state model was applied.")
            self.nation_started = True
            return self.nation_queue.pop(0)

        def _handle_completed_scope_round(self, scope_name, scope_round, source_round):
            """No-op in harness."""
            return None

        def _pause_for_scope_waits(self):
            if not self.state_applied and not self.ready_fetch:
                self.ready_fetch = True
                return True
            return False

        def _apply_ready_scope_models(self):
            if self.ready_fetch:
                self.ready_fetch = False
                self.state_applied = True
                return True
            return False

    harness = ScopeFlowHarness()
    assert NodeService._process_high_level_rounds(harness) is True, "State round + wait/apply should report work done."
    assert harness.state_applied, "State model should be applied after the wait window."

    assert NodeService._process_high_level_rounds(harness) is True, "Nation round should execute on the second invocation."
    assert harness.nation_started, "Nation round must run after the state model was applied."


def test_wait_for_scope_anchor_uses_latest_endpoint() -> None:
    """Ensure anchor wait loop polls the latest endpoint rather than round-specific queries."""

    node = TestHierarchyNode()
    runtime = node._ensure_scope_runtime(node.scope_name, node.scope_config)
    runtime.scope_id = node.scope_config.scope_id

    latest_calls = {"count": 0}

    def fake_latest(scope_name: str, scope_id: str):
        latest_calls["count"] += 1
        return None

    def fail_get_anchor(*args, **kwargs):
        raise AssertionError("Round-specific get_anchor should not be invoked for scope waits.")

    node.blockchain.get_latest_scope_model = fake_latest  # type: ignore[method-assign]
    node.blockchain.get_anchor = fail_get_anchor  # type: ignore[method-assign]

    result = node._wait_for_scope_anchor(1, timeout=0.01, runtime=runtime)
    assert result is None
    assert latest_calls["count"] >= 1, "Latest endpoint should be queried at least once during the wait loop."


def test_anchor_wait_expedites_ready_fetch() -> None:
    """Anchors observed during wait windows should unlock immediate fetch cycles."""

    node = TestHierarchyNode()
    runtime = node._ensure_scope_runtime(node.scope_name, node.scope_config)
    runtime.scope_id = node.scope_config.scope_id
    node._queue_scope_wait(node.scope_name, node.scope_config)
    cid = "cid-new"
    hash_val = "hash-new"
    node.blockchain.anchor(runtime.scope_id, 1, cid, hash_val, scope="state")

    result = node._wait_for_scope_anchor(1, timeout=0.1, runtime=runtime, expedite_fetch=True)
    assert result == (cid, hash_val)

    ready = getattr(node, "_ready_scope_fetches", None) or set()
    assert "state" in ready, "State scope should be ready to fetch immediately after observing anchor."
    queue = getattr(node, "_pending_scope_waits", None)
    if queue:
        assert any(entry[0] == "state" for entry in queue), "Wait window should stay queued until the state model is applied."


def test_scope_waits_only_progress_when_lower_scope_applied(monkeypatch) -> None:
    """Ensure nodes do not begin higher-scope waits before applying lower-scope models."""

    node = TestHierarchyNode()
    runtime_state = node._ensure_scope_runtime(node.scope_name, node.scope_config)
    runtime_state.scope_id = node.scope_config.scope_id

    tensor = np.ones((2, 2), dtype=np.float32)
    cid = node.ipfs.add(tensor)
    hash_val = compute_model_hash(tensor)
    node.blockchain.anchor(runtime_state.scope_id, 1, cid, hash_val, scope="state")

    node._queue_scope_wait("state", node.scope_config)
    node._queue_scope_wait("nation", node.higher_scope_config)

    anchor = node.blockchain.get_latest_scope_model("state", runtime_state.scope_id)
    node._mark_scope_fetch_ready(runtime_state, anchor.cid if anchor else None, anchor)

    ready = getattr(node, "_ready_scope_fetches", None)
    assert ready is not None and "state" in ready

    monkeypatch.setattr("secure_aggregation.communication.hierarchy_mixin.time.sleep", lambda seconds: None)

    assert node._pause_for_scope_waits() is False, "Wait processing should pause until ready scopes are applied."
    queue = getattr(node, "_pending_scope_waits", None)
    assert queue and any(entry[0] == "state" for entry in queue), "State wait window must remain queued before apply."

    assert node._apply_ready_scope_models() is True
    queue_after_state = getattr(node, "_pending_scope_waits", None)
    assert queue_after_state and all(entry[0] != "state" for entry in queue_after_state)

    assert node._pause_for_scope_waits() is True, "Next scope wait should run only after state application."
    ready_after_pause = getattr(node, "_ready_scope_fetches", None)
    assert ready_after_pause is not None and "nation" in ready_after_pause


def test_pending_anchor_skips_latest_query() -> None:
    """Applying a cached anchor should not re-query the blockchain."""

    node = TestHierarchyNode()
    runtime = node._ensure_scope_runtime(node.scope_name, node.scope_config)
    runtime.scope_id = node.scope_config.scope_id
    tensor = np.array([1.0, 2.0], dtype=np.float32)
    cid = node.ipfs.add(tensor)
    runtime.pending_anchor = ModelAnchor(cluster_id=runtime.scope_id, round_num=1, cid=cid, hash=compute_model_hash(tensor))

    def fail_latest(*args, **kwargs):
        raise AssertionError("Blockchain should not be queried when pending anchor is available")

    node.blockchain.get_latest_scope_model = fail_latest  # type: ignore[method-assign]
    node._apply_scope_model_from_anchor("state", node.scope_config, is_local_scope=True)
    assert node._applied_tensors, "Cached anchor should allow immediate model application."
