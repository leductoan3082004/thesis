"""
End-to-end integration test for inter-cluster aggregation flow.

This test verifies the complete inter-cluster aggregation pipeline:
1. ECM buffer receives and stores references from neighbor clusters
2. InterClusterAggregator fetches and verifies models from IPFS
3. InterClusterMerger performs adaptive clipping and weighted merge
4. Merged model is published to IPFS and anchored on blockchain
5. Bridge nodes can gossip ECMs to neighbor clusters
"""

import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from secure_aggregation.communication.bridge_service import BridgeClient
from secure_aggregation.communication.inter_cluster_aggregator import InterClusterAggregator
from secure_aggregation.node import ECM, ECMBuffer
from secure_aggregation.protocol import InterClusterMerger, MergeConfig
from secure_aggregation.storage.model_store import (
    MockBlockchain,
    MockIPFS,
    compute_model_hash,
    verify_model_hash,
)
from secure_aggregation.topology import (
    build_d_cliques,
    build_interclique_edges,
    assign_node_edges,
    get_bridge_nodes,
    get_inter_clique_neighbors,
    is_bridge_node,
)


class TestInterClusterFlowIntegration:
    """Integration tests for the complete inter-cluster aggregation flow."""

    def test_full_inter_cluster_round(self) -> None:
        """
        Test complete inter-cluster round:
        1. Cluster 0 and Cluster 1 each produce intra-cluster models
        2. Both publish to shared IPFS and anchor on blockchain
        3. ECMs are exchanged via buffers
        4. Each cluster merges with neighbor models
        5. Verify merged models are different from originals
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs_path = str(Path(tmpdir) / "ipfs")
            blockchain_path = str(Path(tmpdir) / "blockchain.json")

            ipfs = MockIPFS(storage_path=ipfs_path)
            blockchain = MockBlockchain(storage_path=blockchain_path)

            # Cluster 0's intra-cluster model (from SAP)
            model_c0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            # Cluster 1's intra-cluster model (from SAP)
            model_c1 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)

            # Both clusters publish their models
            cid_c0 = ipfs.add(model_c0)
            hash_c0 = compute_model_hash(model_c0)
            blockchain.anchor("cluster_0", round_num=1, cid=cid_c0, hash_val=hash_c0)

            cid_c1 = ipfs.add(model_c1)
            hash_c1 = compute_model_hash(model_c1)
            blockchain.anchor("cluster_1", round_num=1, cid=cid_c1, hash_val=hash_c1)

            # Create inter-cluster aggregators for each cluster
            agg_c0 = InterClusterAggregator(
                cluster_id="cluster_0",
                ipfs=ipfs,
                blockchain=blockchain,
                merge_config=MergeConfig(window_size=10, alpha=0.5, base_gamma=0.2),
            )
            agg_c1 = InterClusterAggregator(
                cluster_id="cluster_1",
                ipfs=ipfs,
                blockchain=blockchain,
                merge_config=MergeConfig(window_size=10, alpha=0.5, base_gamma=0.2),
            )

            # Cluster 0 receives ECM from Cluster 1
            ecm_from_c1 = ECM(cid=cid_c1, hash=hash_c1, source_cluster="cluster_1")
            agg_c0.receive_ecms("bridge_node_0", [ecm_from_c1])

            # Cluster 1 receives ECM from Cluster 0
            ecm_from_c0 = ECM(cid=cid_c0, hash=hash_c0, source_cluster="cluster_0")
            agg_c1.receive_ecms("bridge_node_1", [ecm_from_c0])

            # Cluster 0 performs inter-cluster merge
            merged_c0, merged_cids_c0 = agg_c0.merge_with_neighbors(model_c0)
            assert len(merged_cids_c0) == 1
            assert merged_cids_c0[0] == cid_c1

            # Cluster 1 performs inter-cluster merge
            merged_c1, merged_cids_c1 = agg_c1.merge_with_neighbors(model_c1)
            assert len(merged_cids_c1) == 1
            assert merged_cids_c1[0] == cid_c0

            # Verify merged models are different from originals (mixing occurred)
            assert not np.array_equal(merged_c0, model_c0)
            assert not np.array_equal(merged_c1, model_c1)

            # Verify merged models are closer to each other than originals
            original_diff = np.linalg.norm(model_c0 - model_c1)
            merged_diff = np.linalg.norm(merged_c0 - merged_c1)
            assert merged_diff < original_diff

    def test_multi_cluster_topology(self) -> None:
        """
        Test with 3 clusters in a ring topology:
        Cluster 0 <-> Cluster 1 <-> Cluster 2 <-> Cluster 0
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs_path = str(Path(tmpdir) / "ipfs")
            blockchain_path = str(Path(tmpdir) / "blockchain.json")

            ipfs = MockIPFS(storage_path=ipfs_path)
            blockchain = MockBlockchain(storage_path=blockchain_path)

            # Three cluster models
            models = {
                "cluster_0": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "cluster_1": np.array([4.0, 5.0, 6.0], dtype=np.float32),
                "cluster_2": np.array([7.0, 8.0, 9.0], dtype=np.float32),
            }

            # Publish all models
            cids = {}
            hashes = {}
            for cluster_id, model in models.items():
                cid = ipfs.add(model)
                h = compute_model_hash(model)
                blockchain.anchor(cluster_id, round_num=1, cid=cid, hash_val=h)
                cids[cluster_id] = cid
                hashes[cluster_id] = h

            # Create aggregators
            aggregators = {}
            for cluster_id in models:
                aggregators[cluster_id] = InterClusterAggregator(
                    cluster_id=cluster_id,
                    ipfs=ipfs,
                    blockchain=blockchain,
                )

            # Ring topology: each cluster gets ECMs from both neighbors
            # Cluster 0 neighbors: 1, 2
            aggregators["cluster_0"].receive_ecms("bridge", [
                ECM(cid=cids["cluster_1"], hash=hashes["cluster_1"], source_cluster="cluster_1"),
                ECM(cid=cids["cluster_2"], hash=hashes["cluster_2"], source_cluster="cluster_2"),
            ])
            # Cluster 1 neighbors: 0, 2
            aggregators["cluster_1"].receive_ecms("bridge", [
                ECM(cid=cids["cluster_0"], hash=hashes["cluster_0"], source_cluster="cluster_0"),
                ECM(cid=cids["cluster_2"], hash=hashes["cluster_2"], source_cluster="cluster_2"),
            ])
            # Cluster 2 neighbors: 0, 1
            aggregators["cluster_2"].receive_ecms("bridge", [
                ECM(cid=cids["cluster_0"], hash=hashes["cluster_0"], source_cluster="cluster_0"),
                ECM(cid=cids["cluster_1"], hash=hashes["cluster_1"], source_cluster="cluster_1"),
            ])

            # All clusters merge
            merged_models = {}
            for cluster_id, agg in aggregators.items():
                merged, _ = agg.merge_with_neighbors(models[cluster_id])
                merged_models[cluster_id] = merged

            # Verify all models moved toward each other
            original_variance = np.var([m.mean() for m in models.values()])
            merged_variance = np.var([m.mean() for m in merged_models.values()])
            assert merged_variance < original_variance

    def test_adaptive_clipping_across_rounds(self) -> None:
        """
        Test that adaptive clipping threshold adjusts across multiple rounds.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs_path = str(Path(tmpdir) / "ipfs")
            blockchain_path = str(Path(tmpdir) / "blockchain.json")

            ipfs = MockIPFS(storage_path=ipfs_path)
            blockchain = MockBlockchain(storage_path=blockchain_path)

            agg = InterClusterAggregator(
                cluster_id="cluster_0",
                ipfs=ipfs,
                blockchain=blockchain,
                merge_config=MergeConfig(window_size=5, alpha=0.5, base_gamma=0.2),
            )

            local_model = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            thresholds = []

            for round_num in range(5):
                # Neighbor model gets progressively more different
                neighbor_model = np.array([1.0, 1.0, 1.0], dtype=np.float32) * (round_num + 1)
                cid = ipfs.add(neighbor_model)
                h = compute_model_hash(neighbor_model)
                blockchain.anchor("cluster_1", round_num=round_num, cid=cid, hash_val=h)

                agg.ecm_buffer.clear()
                ecm = ECM(cid=cid, hash=h, source_cluster="cluster_1")
                agg.receive_ecms("bridge", [ecm])

                merged, _, _ = agg.process_round(local_model, round_num)
                thresholds.append(agg.inter_cluster_aggregator.get_current_threshold()
                                  if hasattr(agg, 'inter_cluster_aggregator') else
                                  agg.merger.get_current_threshold())

            # Threshold should increase as norms increase
            assert thresholds[-1] > thresholds[0]

    def test_publish_and_retrieve_model(self) -> None:
        """Test model publishing to IPFS and blockchain anchoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs_path = str(Path(tmpdir) / "ipfs")
            blockchain_path = str(Path(tmpdir) / "blockchain.json")

            ipfs = MockIPFS(storage_path=ipfs_path)
            blockchain = MockBlockchain(storage_path=blockchain_path)

            agg = InterClusterAggregator(
                cluster_id="cluster_0",
                ipfs=ipfs,
                blockchain=blockchain,
            )

            model = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
            cid, model_hash = agg.publish_model(model, round_num=1)

            assert cid is not None
            assert model_hash is not None

            # Verify can retrieve from blockchain
            anchor = blockchain.get_anchor("cluster_0", round_num=1)
            assert anchor is not None
            retrieved_cid, retrieved_hash = anchor
            assert retrieved_cid == cid
            assert retrieved_hash == model_hash

            # Verify can retrieve from IPFS
            retrieved_model = ipfs.get(cid)
            assert retrieved_model is not None
            np.testing.assert_array_equal(retrieved_model, model)

    def test_ecm_deduplication(self) -> None:
        """Test that duplicate ECMs are deduplicated by CID."""
        buffer = ECMBuffer(freshness_window=300.0)

        cid = "QmTestCID123"
        hash_val = "abc123"

        # Add same ECM multiple times
        for _ in range(5):
            buffer.add_from_message(cid, hash_val, source_cluster="cluster_1")

        # Should only have one unique entry
        unique = buffer.get_unique_cids()
        assert len(unique) == 1
        assert cid in unique
        assert unique[cid] == hash_val

    def test_hash_verification_rejects_tampered_model(self) -> None:
        """Test that hash verification rejects tampered models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs_path = str(Path(tmpdir) / "ipfs")
            blockchain_path = str(Path(tmpdir) / "blockchain.json")

            ipfs = MockIPFS(storage_path=ipfs_path)
            blockchain = MockBlockchain(storage_path=blockchain_path)

            # Publish original model
            original_model = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            cid = ipfs.add(original_model)
            original_hash = compute_model_hash(original_model)
            blockchain.anchor("cluster_1", round_num=1, cid=cid, hash_val=original_hash)

            # Create aggregator
            agg = InterClusterAggregator(
                cluster_id="cluster_0",
                ipfs=ipfs,
                blockchain=blockchain,
            )

            # Tamper with model in IPFS by replacing it
            tampered_model = np.array([999.0, 999.0, 999.0], dtype=np.float32)
            # Directly overwrite the file
            import pickle
            with open(Path(ipfs_path) / cid, "wb") as f:
                f.write(pickle.dumps(tampered_model))

            # Add ECM with original hash (attacker doesn't know new hash)
            ecm = ECM(cid=cid, hash=original_hash, source_cluster="cluster_1")
            agg.receive_ecms("bridge", [ecm])

            # Merge should reject tampered model (hash mismatch)
            local_model = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            merged, merged_cids = agg.merge_with_neighbors(local_model)

            # No models should be merged due to hash verification failure
            assert len(merged_cids) == 0
            np.testing.assert_array_equal(merged, local_model)


class TestBridgeNodeTopology:
    """Tests for bridge node identification and topology helpers."""

    def test_bridge_node_identification(self) -> None:
        """Test correct identification of bridge nodes from inter-edges."""
        # Create a simple 2-clique topology
        inter_edges = [("node_0", "node_3"), ("node_1", "node_4")]

        bridge_nodes = get_bridge_nodes(inter_edges)
        assert bridge_nodes == {"node_0", "node_1", "node_3", "node_4"}

        assert is_bridge_node("node_0", inter_edges)
        assert is_bridge_node("node_3", inter_edges)
        assert not is_bridge_node("node_2", inter_edges)

    def test_inter_clique_neighbors(self) -> None:
        """Test getting inter-clique neighbors for a node."""
        inter_edges = [("node_0", "node_3"), ("node_0", "node_4")]

        neighbors = get_inter_clique_neighbors("node_0", inter_edges)
        assert set(neighbors) == {"node_3", "node_4"}

        neighbors = get_inter_clique_neighbors("node_3", inter_edges)
        assert neighbors == ["node_0"]

        neighbors = get_inter_clique_neighbors("node_2", inter_edges)
        assert neighbors == []

    def test_d_clique_topology_with_inter_edges(self) -> None:
        """Test building complete D-Clique topology with inter-clique edges."""
        # 6 nodes, clique size 3 -> 2 cliques
        node_labels = {f"node_{i}": {"0": 1.0} for i in range(6)}

        cliques = build_d_cliques(node_labels, clique_size=3, seed=42)
        assert len(cliques) == 2

        interclique_edges = build_interclique_edges(cliques, mode="ring")
        assert len(interclique_edges) == 1  # Ring with 2 cliques has 1 unique edge

        inter_edges, _ = assign_node_edges(cliques, interclique_edges)
        assert len(inter_edges) == 1

        # Verify bridge nodes span both cliques
        bridge_nodes = get_bridge_nodes(inter_edges)
        assert len(bridge_nodes) == 2

        clique_0_bridges = bridge_nodes & cliques[0]
        clique_1_bridges = bridge_nodes & cliques[1]
        assert len(clique_0_bridges) == 1
        assert len(clique_1_bridges) == 1
