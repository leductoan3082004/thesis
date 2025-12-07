"""Tests for bridge node helper functions."""

import pytest

from secure_aggregation.topology import (
    get_bridge_nodes,
    get_clique_bridge_nodes,
    get_inter_clique_neighbors,
    get_neighbor_clique_indices,
    is_bridge_node,
)


class TestGetBridgeNodes:
    """Tests for get_bridge_nodes function."""

    def test_empty_edges(self) -> None:
        result = get_bridge_nodes([])
        assert result == set()

    def test_single_edge(self) -> None:
        inter_edges = [("n0", "n3")]
        result = get_bridge_nodes(inter_edges)
        assert result == {"n0", "n3"}

    def test_multiple_edges(self) -> None:
        inter_edges = [("n0", "n3"), ("n1", "n4"), ("n0", "n5")]
        result = get_bridge_nodes(inter_edges)
        assert result == {"n0", "n1", "n3", "n4", "n5"}

    def test_no_duplicates(self) -> None:
        inter_edges = [("n0", "n3"), ("n0", "n4"), ("n0", "n5")]
        result = get_bridge_nodes(inter_edges)
        assert result == {"n0", "n3", "n4", "n5"}


class TestGetInterCliqueNeighbors:
    """Tests for get_inter_clique_neighbors function."""

    def test_no_neighbors(self) -> None:
        inter_edges = [("n0", "n3")]
        result = get_inter_clique_neighbors("n1", inter_edges)
        assert result == []

    def test_single_neighbor_first_position(self) -> None:
        inter_edges = [("n0", "n3")]
        result = get_inter_clique_neighbors("n0", inter_edges)
        assert result == ["n3"]

    def test_single_neighbor_second_position(self) -> None:
        inter_edges = [("n0", "n3")]
        result = get_inter_clique_neighbors("n3", inter_edges)
        assert result == ["n0"]

    def test_multiple_neighbors(self) -> None:
        inter_edges = [("n0", "n3"), ("n0", "n4"), ("n0", "n5")]
        result = get_inter_clique_neighbors("n0", inter_edges)
        assert set(result) == {"n3", "n4", "n5"}


class TestIsBridgeNode:
    """Tests for is_bridge_node function."""

    def test_is_bridge_first_position(self) -> None:
        inter_edges = [("n0", "n3")]
        assert is_bridge_node("n0", inter_edges) is True

    def test_is_bridge_second_position(self) -> None:
        inter_edges = [("n0", "n3")]
        assert is_bridge_node("n3", inter_edges) is True

    def test_not_bridge(self) -> None:
        inter_edges = [("n0", "n3")]
        assert is_bridge_node("n1", inter_edges) is False

    def test_empty_edges(self) -> None:
        assert is_bridge_node("n0", []) is False


class TestGetCliqueBridgeNodes:
    """Tests for get_clique_bridge_nodes function."""

    def test_clique_with_bridge_nodes(self) -> None:
        cliques = [{"n0", "n1", "n2"}, {"n3", "n4", "n5"}]
        inter_edges = [("n0", "n3"), ("n1", "n4")]
        result = get_clique_bridge_nodes(0, cliques, inter_edges)
        assert result == {"n0", "n1"}

    def test_clique_without_bridge_nodes(self) -> None:
        cliques = [{"n0", "n1", "n2"}, {"n3", "n4", "n5"}, {"n6", "n7", "n8"}]
        inter_edges = [("n0", "n3")]
        result = get_clique_bridge_nodes(2, cliques, inter_edges)
        assert result == set()

    def test_all_nodes_are_bridges(self) -> None:
        cliques = [{"n0", "n1"}, {"n2", "n3"}]
        inter_edges = [("n0", "n2"), ("n0", "n3"), ("n1", "n2"), ("n1", "n3")]
        result = get_clique_bridge_nodes(0, cliques, inter_edges)
        assert result == {"n0", "n1"}


class TestGetNeighborCliqueIndices:
    """Tests for get_neighbor_clique_indices function."""

    def test_single_neighbor(self) -> None:
        cliques = [{"n0", "n1"}, {"n2", "n3"}, {"n4", "n5"}]
        inter_edges = [("n0", "n2")]
        result = get_neighbor_clique_indices(0, cliques, inter_edges)
        assert result == {1}

    def test_multiple_neighbors(self) -> None:
        cliques = [{"n0", "n1"}, {"n2", "n3"}, {"n4", "n5"}]
        inter_edges = [("n0", "n2"), ("n1", "n4")]
        result = get_neighbor_clique_indices(0, cliques, inter_edges)
        assert result == {1, 2}

    def test_no_neighbors(self) -> None:
        cliques = [{"n0", "n1"}, {"n2", "n3"}, {"n4", "n5"}]
        inter_edges = [("n2", "n4")]
        result = get_neighbor_clique_indices(0, cliques, inter_edges)
        assert result == set()

    def test_bidirectional_edge(self) -> None:
        cliques = [{"n0", "n1"}, {"n2", "n3"}]
        inter_edges = [("n0", "n2")]
        # From clique 0's perspective.
        result0 = get_neighbor_clique_indices(0, cliques, inter_edges)
        assert result0 == {1}
        # From clique 1's perspective.
        result1 = get_neighbor_clique_indices(1, cliques, inter_edges)
        assert result1 == {0}


class TestIntegration:
    """Integration tests for bridge node functions."""

    def test_full_topology_analysis(self) -> None:
        cliques = [
            {"n0", "n1", "n2"},
            {"n3", "n4", "n5"},
            {"n6", "n7", "n8"},
        ]
        inter_edges = [
            ("n0", "n3"),
            ("n1", "n6"),
            ("n4", "n7"),
        ]
        all_bridges = get_bridge_nodes(inter_edges)
        assert all_bridges == {"n0", "n1", "n3", "n4", "n6", "n7"}
        c0_bridges = get_clique_bridge_nodes(0, cliques, inter_edges)
        assert c0_bridges == {"n0", "n1"}
        c0_neighbors = get_neighbor_clique_indices(0, cliques, inter_edges)
        assert c0_neighbors == {1, 2}
        n0_neighbors = get_inter_clique_neighbors("n0", inter_edges)
        assert n0_neighbors == ["n3"]
        assert is_bridge_node("n2", inter_edges) is False
