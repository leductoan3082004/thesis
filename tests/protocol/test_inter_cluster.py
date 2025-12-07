"""Tests for inter-cluster merge algorithm."""

import numpy as np
import pytest

from secure_aggregation.protocol.inter_cluster import (
    AdaptiveClipper,
    InterClusterMerger,
    MergeConfig,
    clip_delta,
    compute_adaptive_gamma,
)


class TestClipDelta:
    """Tests for clip_delta function."""

    def test_no_clipping_when_below_threshold(self) -> None:
        delta = np.array([3.0, 4.0])  # Norm = 5.0
        clipped = clip_delta(delta, c_cluster=10.0)
        np.testing.assert_array_almost_equal(clipped, delta)

    def test_clips_when_above_threshold(self) -> None:
        delta = np.array([3.0, 4.0])  # Norm = 5.0
        clipped = clip_delta(delta, c_cluster=2.0)
        assert np.linalg.norm(clipped) == pytest.approx(2.0)
        # Direction should be preserved.
        expected_direction = delta / np.linalg.norm(delta)
        actual_direction = clipped / np.linalg.norm(clipped)
        np.testing.assert_array_almost_equal(actual_direction, expected_direction)

    def test_handles_zero_vector(self) -> None:
        delta = np.array([0.0, 0.0, 0.0])
        clipped = clip_delta(delta, c_cluster=1.0)
        np.testing.assert_array_equal(clipped, delta)

    def test_exact_threshold(self) -> None:
        delta = np.array([3.0, 4.0])  # Norm = 5.0
        clipped = clip_delta(delta, c_cluster=5.0)
        np.testing.assert_array_almost_equal(clipped, delta)


class TestAdaptiveClipper:
    """Tests for AdaptiveClipper class."""

    def test_initialization_requires_positive_window(self) -> None:
        with pytest.raises(ValueError, match="window_size must be positive"):
            AdaptiveClipper(window_size=0)

    def test_returns_initial_threshold_when_empty(self) -> None:
        clipper = AdaptiveClipper(window_size=5, initial_threshold=2.5)
        assert clipper.get_threshold() == 2.5

    def test_uses_90th_percentile(self) -> None:
        clipper = AdaptiveClipper(window_size=10)
        norms = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        clipper.update(norms)
        threshold = clipper.get_threshold()
        expected = np.percentile(norms, 90)
        assert threshold == pytest.approx(expected)

    def test_sliding_window_limits_history(self) -> None:
        clipper = AdaptiveClipper(window_size=2)
        clipper.update([1.0, 2.0])
        clipper.update([3.0, 4.0])
        clipper.update([100.0, 200.0])  # These should push out older values.
        # With window_size=2 and 2 values per update, max_history = 4.
        assert len(clipper.norm_history) <= 4

    def test_update_and_get_threshold(self) -> None:
        clipper = AdaptiveClipper(window_size=5)
        threshold = clipper.update_and_get_threshold([1.0, 2.0, 3.0])
        assert threshold > 0


class TestComputeAdaptiveGamma:
    """Tests for compute_adaptive_gamma function."""

    def test_returns_base_gamma_when_no_deltas(self) -> None:
        gamma = compute_adaptive_gamma([], alpha=0.5, base_gamma=0.2)
        assert gamma == 0.2

    def test_reduces_gamma_with_disagreement(self) -> None:
        deltas = [np.array([10.0, 0.0]), np.array([0.0, 10.0])]
        gamma = compute_adaptive_gamma(deltas, alpha=0.5, base_gamma=0.2)
        assert gamma < 0.2

    def test_higher_alpha_more_sensitive(self) -> None:
        deltas = [np.array([5.0, 0.0])]
        gamma_low_alpha = compute_adaptive_gamma(deltas, alpha=0.1, base_gamma=0.2)
        gamma_high_alpha = compute_adaptive_gamma(deltas, alpha=1.0, base_gamma=0.2)
        assert gamma_high_alpha < gamma_low_alpha

    def test_zero_disagreement_returns_base_gamma(self) -> None:
        deltas = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
        gamma = compute_adaptive_gamma(deltas, alpha=0.5, base_gamma=0.2)
        assert gamma == pytest.approx(0.2)


class TestInterClusterMerger:
    """Tests for InterClusterMerger class."""

    def test_returns_local_when_no_neighbors(self) -> None:
        merger = InterClusterMerger()
        theta_local = np.array([1.0, 2.0, 3.0])
        result = merger.merge(theta_local, [])
        np.testing.assert_array_equal(result, theta_local)

    def test_merges_with_single_neighbor(self) -> None:
        config = MergeConfig(base_gamma=0.5, alpha=0.0)
        merger = InterClusterMerger(config)
        theta_local = np.array([1.0, 1.0])
        neighbor = np.array([3.0, 3.0])
        result = merger.merge(theta_local, [neighbor])
        # With gamma=0.5 and uniform weights, result should be midpoint.
        expected = (1 - 0.5) * theta_local + 0.5 * neighbor
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_clips_extreme_neighbor(self) -> None:
        config = MergeConfig(initial_c_cluster=1.0, base_gamma=0.5, alpha=0.0)
        merger = InterClusterMerger(config)
        theta_local = np.array([0.0, 0.0])

        # First, establish history with normal deltas so threshold stays low.
        normal_neighbors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        for _ in range(5):
            merger.merge(theta_local, normal_neighbors)

        # Now the extreme neighbor should be clipped.
        extreme_neighbor = np.array([100.0, 0.0])
        result = merger.merge(theta_local, [extreme_neighbor])
        # Result should be much smaller than 50 (which is what we'd get without clipping).
        # The 90th percentile threshold adapts, but clipping should still reduce the impact.
        assert np.linalg.norm(result) < 10.0

    def test_multiple_neighbors_averaged(self) -> None:
        config = MergeConfig(base_gamma=1.0, alpha=0.0, initial_c_cluster=100.0)
        merger = InterClusterMerger(config)
        theta_local = np.array([0.0, 0.0])
        neighbors = [np.array([2.0, 0.0]), np.array([0.0, 2.0])]
        result = merger.merge(theta_local, neighbors)
        # With gamma=1.0, result is pure average of clipped neighbors.
        expected = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_adaptive_clipping_threshold_updates(self) -> None:
        merger = InterClusterMerger()
        theta_local = np.array([0.0, 0.0])
        initial_threshold = merger.get_current_threshold()

        neighbors = [np.array([10.0, 0.0])]
        merger.merge(theta_local, neighbors)

        # Threshold should have updated based on the delta norm.
        new_threshold = merger.get_current_threshold()
        assert new_threshold != initial_threshold or new_threshold == 10.0

    def test_preserves_model_dimension(self) -> None:
        merger = InterClusterMerger()
        theta_local = np.random.randn(100)
        neighbors = [np.random.randn(100) for _ in range(3)]
        result = merger.merge(theta_local, neighbors)
        assert result.shape == theta_local.shape


class TestMergeConfig:
    """Tests for MergeConfig dataclass."""

    def test_default_values(self) -> None:
        config = MergeConfig()
        assert config.window_size == 10
        assert config.alpha == 0.5
        assert config.base_gamma == 0.2
        assert config.initial_c_cluster == 1.0

    def test_custom_values(self) -> None:
        config = MergeConfig(window_size=5, alpha=0.3, base_gamma=0.1)
        assert config.window_size == 5
        assert config.alpha == 0.3
        assert config.base_gamma == 0.1
