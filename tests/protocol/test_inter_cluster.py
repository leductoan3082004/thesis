"""Tests for inter-cluster merge algorithm."""

import numpy as np

from secure_aggregation.protocol.inter_cluster import InterClusterMerger, MergeConfig


class TestInterClusterMerger:
    """Tests for InterClusterMerger class."""

    def test_returns_local_when_no_neighbors(self) -> None:
        merger = InterClusterMerger()
        theta_local = np.array([1.0, 2.0, 3.0])
        result = merger.merge(theta_local, [])
        np.testing.assert_array_equal(result, theta_local)

    def test_merges_with_single_neighbor(self) -> None:
        merger = InterClusterMerger()
        theta_local = np.array([1.0, 1.0])
        neighbor = np.array([3.0, 3.0])
        result = merger.merge(theta_local, [neighbor])
        expected = (theta_local + neighbor) / 2
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_multiple_neighbors_averaged(self) -> None:
        merger = InterClusterMerger()
        theta_local = np.array([0.0, 0.0])
        neighbors = [np.array([2.0, 0.0]), np.array([0.0, 2.0])]
        result = merger.merge(theta_local, neighbors)
        expected = np.array([2.0 / 3.0, 2.0 / 3.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

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
        assert config.max_neighbors == 4

    def test_custom_values(self) -> None:
        config = MergeConfig(max_neighbors=2)
        assert config.max_neighbors == 2
