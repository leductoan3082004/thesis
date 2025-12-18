"""Tests for convergence tracking logic."""

import os

import pytest

np = pytest.importorskip("numpy")

from secure_aggregation.convergence import ConvergenceConfig, ConvergenceState, ConvergenceTracker

WARMUP_ENV = "CONVERGENCE_WARMUP_ROUNDS"


def clear_warmup_env() -> None:
    """Ensure environment override does not leak between tests."""
    os.environ.pop(WARMUP_ENV, None)


class TestConvergenceConfig:
    """Tests for ConvergenceConfig dataclass."""

    def setup_method(self) -> None:
        clear_warmup_env()

    def test_default_values(self) -> None:
        config = ConvergenceConfig()
        assert config.enabled is True
        assert config.convergence_warmup_rounds == 5
        assert config.tol_abs == 1e-5
        assert config.tol_rel == 0.001
        assert config.patience == 3
        assert config.require_neighbor_convergence is True
        assert config.central_checker_id is None
        assert config.signal_timeout == 30.0

    def test_from_dict_with_values(self) -> None:
        data = {
            "enabled": False,
            "convergence_warmup_rounds": 50,
            "tol_abs": 1e-6,
            "tol_rel": 0.01,
            "patience": 5,
            "require_neighbor_convergence": False,
            "central_checker_id": "central_0",
            "signal_timeout": 12.5,
        }
        config = ConvergenceConfig.from_dict(data)
        assert config.enabled is False
        assert config.convergence_warmup_rounds == 50
        assert config.tol_abs == 1e-6
        assert config.tol_rel == 0.01
        assert config.patience == 5
        assert config.require_neighbor_convergence is False
        assert config.central_checker_id == "central_0"
        assert config.signal_timeout == 12.5

    def test_from_dict_with_none(self) -> None:
        config = ConvergenceConfig.from_dict(None)
        assert config.enabled is True
        assert config.convergence_warmup_rounds == 5

    def test_from_dict_with_partial_values(self) -> None:
        data = {"convergence_warmup_rounds": 200}
        config = ConvergenceConfig.from_dict(data)
        assert config.convergence_warmup_rounds == 200
        assert config.tol_abs == 1e-5  # default

    def test_env_override_takes_precedence(self) -> None:
        os.environ[WARMUP_ENV] = "7"
        config = ConvergenceConfig.from_dict({"max_rounds": 2})
        assert config.max_rounds == 7


class TestConvergenceTracker:
    """Tests for ConvergenceTracker class."""

    def setup_method(self) -> None:
        os.environ[WARMUP_ENV] = "0"

    def teardown_method(self) -> None:
        clear_warmup_env()

    def test_initial_state(self) -> None:
        config = ConvergenceConfig()
        tracker = ConvergenceTracker(config, "cluster_0")

        assert tracker.state.round_idx == 0
        assert tracker.state.prev_model is None
        assert tracker.state.convergence_streak == 0
        assert tracker.state.cluster_converged is False
        assert tracker.state.should_stop is False

    def test_first_update_no_convergence(self) -> None:
        config = ConvergenceConfig()
        tracker = ConvergenceTracker(config, "cluster_0")

        model = np.array([1.0, 2.0, 3.0])
        state = tracker.update(model)

        # First update can't converge (no previous model to compare)
        assert state.convergence_streak == 0
        assert state.cluster_converged is False
        assert state.should_stop is False
        assert state.round_idx == 1

    def test_convergence_detection_with_stable_model(self) -> None:
        config = ConvergenceConfig(tol_abs=0.1, patience=2)
        tracker = ConvergenceTracker(config, "cluster_0")

        # First model
        model1 = np.array([1.0, 2.0, 3.0])
        tracker.update(model1)

        # Second model - very small change (below tolerance)
        model2 = np.array([1.0, 2.0, 3.00001])
        state = tracker.update(model2)
        assert state.convergence_streak == 1
        assert state.cluster_converged is False

        # Third model - another small change
        model3 = np.array([1.0, 2.0, 3.00002])
        state = tracker.update(model3)
        assert state.convergence_streak == 2
        assert state.cluster_converged is True
        assert state.should_stop is True
        assert state.stop_reason == "global_convergence"

    def test_convergence_streak_reset_on_large_change(self) -> None:
        config = ConvergenceConfig(tol_abs=0.1, patience=3)
        tracker = ConvergenceTracker(config, "cluster_0")

        # Build up streak
        model1 = np.array([1.0, 2.0, 3.0])
        tracker.update(model1)

        model2 = np.array([1.0, 2.0, 3.00001])
        state = tracker.update(model2)
        assert state.convergence_streak == 1

        # Large change - streak should reset
        model3 = np.array([1.0, 2.0, 5.0])
        state = tracker.update(model3)
        assert state.convergence_streak == 0
        assert state.cluster_converged is False

    def test_max_rounds_stop(self) -> None:
        config = ConvergenceConfig(convergence_warmup_rounds=3, tol_abs=1e-10)
        tracker = ConvergenceTracker(config, "cluster_0")

        # Run until warmup rounds have elapsed; tracker should not force-stop just
        # because the warmup cap was reached.
        for i in range(4):
            model = np.array([float(i), float(i + 1), float(i + 2)])
            state = tracker.update(model)

        assert state.should_stop is False

    def test_neighbor_convergence_required(self) -> None:
        config = ConvergenceConfig(
            tol_abs=0.1, patience=1, require_neighbor_convergence=True
        )
        tracker = ConvergenceTracker(config, "cluster_0")

        # Set up neighbor that hasn't converged
        tracker.receive_neighbor_convergence("cluster_1", False)

        # Make local model converge
        model1 = np.array([1.0, 2.0, 3.0])
        tracker.update(model1)

        model2 = np.array([1.0, 2.0, 3.00001])
        state = tracker.update(model2)

        # Should be locally converged but not globally (neighbor not ready)
        assert state.cluster_converged is True
        assert state.should_stop is False

        # Now neighbor converges
        tracker.receive_neighbor_convergence("cluster_1", True)

        model3 = np.array([1.0, 2.0, 3.00002])
        state = tracker.update(model3)

        # Now should stop
        assert state.should_stop is True
        assert state.stop_reason == "global_convergence"

    def test_neighbor_convergence_not_required(self) -> None:
        config = ConvergenceConfig(
            tol_abs=0.1, patience=1, require_neighbor_convergence=False
        )
        tracker = ConvergenceTracker(config, "cluster_0")

        # Set up neighbor that hasn't converged
        tracker.receive_neighbor_convergence("cluster_1", False)

        # Make local model converge
        model1 = np.array([1.0, 2.0, 3.0])
        tracker.update(model1)

        model2 = np.array([1.0, 2.0, 3.00001])
        state = tracker.update(model2)

        # Should stop even though neighbor hasn't converged
        assert state.cluster_converged is True
        assert state.should_stop is True

    def test_disabled_convergence(self) -> None:
        config = ConvergenceConfig(enabled=False)
        tracker = ConvergenceTracker(config, "cluster_0")

        model1 = np.array([1.0, 2.0, 3.0])
        tracker.update(model1)

        model2 = np.array([1.0, 2.0, 3.00001])
        state = tracker.update(model2)

        # Convergence tracking disabled - should not converge
        assert state.convergence_streak == 0
        assert state.cluster_converged is False
        assert state.should_stop is False

    def test_reset(self) -> None:
        config = ConvergenceConfig(tol_abs=0.1, patience=1)
        tracker = ConvergenceTracker(config, "cluster_0")

        model1 = np.array([1.0, 2.0, 3.0])
        tracker.update(model1)

        model2 = np.array([1.0, 2.0, 3.00001])
        state = tracker.update(model2)
        assert state.cluster_converged is True

        # Reset
        tracker.reset()

        assert tracker.state.round_idx == 0
        assert tracker.state.prev_model is None
        assert tracker.state.convergence_streak == 0
        assert tracker.state.cluster_converged is False

    def test_delta_norm_calculation(self) -> None:
        config = ConvergenceConfig(tol_abs=0.1, patience=3)
        tracker = ConvergenceTracker(config, "cluster_0")

        model1 = np.array([0.0, 0.0, 0.0])
        tracker.update(model1)

        model2 = np.array([3.0, 4.0, 0.0])  # L2 norm of delta = 5.0
        state = tracker.update(model2)

        assert abs(state.delta_norm - 5.0) < 1e-6

    def test_relative_tolerance(self) -> None:
        config = ConvergenceConfig(tol_abs=1e-10, tol_rel=0.01, patience=1)
        tracker = ConvergenceTracker(config, "cluster_0")

        # Large model where 1% change is significant
        model1 = np.array([100.0, 200.0, 300.0])
        tracker.update(model1)

        # 0.5% change - below relative tolerance
        model2 = np.array([100.5, 200.0, 300.0])
        state = tracker.update(model2)

        assert state.convergence_streak == 1
        assert state.cluster_converged is True

    def test_central_signal_flow(self) -> None:
        signals = []

        def sender(signal) -> None:
            signals.append(signal)

        config = ConvergenceConfig(tol_abs=0.1, patience=1, central_checker_id="central_0")
        tracker = ConvergenceTracker(config, "cluster_0", signal_sender=sender)

        model1 = np.array([1.0, 2.0, 3.0])
        tracker.update(model1)
        assert len(signals) == 1
        assert signals[-1].converged is False
        assert signals[-1].destination == "central_0"

        model2 = np.array([1.0, 2.0, 3.00001])
        state = tracker.update(model2)
        assert len(signals) == 2
        assert signals[-1].converged is True
        assert state.cluster_converged is True
        assert state.should_stop is False  # waiting for central confirmation

        tracker.receive_global_convergence(round_idx=state.round_idx)
        state = tracker.update(np.array([1.0, 2.0, 3.00002]))
        assert state.should_stop is True
        assert state.stop_reason == "global_convergence"
