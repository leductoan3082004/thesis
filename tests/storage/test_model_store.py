"""Tests for IPFS and Blockchain mock implementations."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from secure_aggregation.storage.model_store import (
    MockBlockchain,
    MockIPFS,
    ModelAnchor,
    compute_model_hash,
    verify_model_hash,
)


class TestHashFunctions:
    """Tests for hash utility functions."""

    def test_compute_model_hash_deterministic(self) -> None:
        model = np.array([1.0, 2.0, 3.0])
        hash1 = compute_model_hash(model)
        hash2 = compute_model_hash(model)
        assert hash1 == hash2

    def test_compute_model_hash_different_for_different_models(self) -> None:
        model1 = np.array([1.0, 2.0, 3.0])
        model2 = np.array([1.0, 2.0, 3.1])
        assert compute_model_hash(model1) != compute_model_hash(model2)

    def test_verify_model_hash_success(self) -> None:
        model = np.array([1.0, 2.0, 3.0])
        hash_val = compute_model_hash(model)
        assert verify_model_hash(model, hash_val) is True

    def test_verify_model_hash_failure(self) -> None:
        model = np.array([1.0, 2.0, 3.0])
        assert verify_model_hash(model, "wrong_hash") is False


class TestMockIPFSInMemory:
    """Tests for MockIPFS with in-memory storage."""

    def test_add_and_get(self) -> None:
        ipfs = MockIPFS()
        model = np.array([1.0, 2.0, 3.0])
        cid = ipfs.add(model)
        retrieved = ipfs.get(cid)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, model)

    def test_cid_is_hash(self) -> None:
        ipfs = MockIPFS()
        model = np.array([1.0, 2.0, 3.0])
        cid = ipfs.add(model)
        expected_hash = compute_model_hash(model)
        assert cid == expected_hash

    def test_exists(self) -> None:
        ipfs = MockIPFS()
        model = np.array([1.0, 2.0, 3.0])
        cid = ipfs.add(model)
        assert ipfs.exists(cid) is True
        assert ipfs.exists("nonexistent") is False

    def test_get_nonexistent(self) -> None:
        ipfs = MockIPFS()
        assert ipfs.get("nonexistent") is None

    def test_clear(self) -> None:
        ipfs = MockIPFS()
        cid = ipfs.add(np.array([1.0, 2.0]))
        ipfs.clear()
        assert ipfs.exists(cid) is False


class TestMockIPFSDirectory:
    """Tests for MockIPFS with directory storage."""

    def test_add_and_get_with_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs = MockIPFS(storage_path=tmpdir)
            model = np.array([1.0, 2.0, 3.0])
            cid = ipfs.add(model)
            retrieved = ipfs.get(cid)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, model)

    def test_file_created_on_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs = MockIPFS(storage_path=tmpdir)
            model = np.array([1.0, 2.0, 3.0])
            cid = ipfs.add(model)
            assert (Path(tmpdir) / cid).exists()

    def test_persistence_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = np.array([1.0, 2.0, 3.0])
            ipfs1 = MockIPFS(storage_path=tmpdir)
            cid = ipfs1.add(model)
            ipfs2 = MockIPFS(storage_path=tmpdir)
            retrieved = ipfs2.get(cid)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, model)



class TestMockBlockchainInMemory:
    """Tests for MockBlockchain with in-memory storage."""

    def test_anchor_and_get(self) -> None:
        blockchain = MockBlockchain()
        blockchain.anchor("cluster_0", round_num=5, cid="abc123", hash_val="def456")
        result = blockchain.get_anchor("cluster_0", round_num=5)
        assert result == ("abc123", "def456")

    def test_get_nonexistent(self) -> None:
        blockchain = MockBlockchain()
        assert blockchain.get_anchor("cluster_0", round_num=0) is None

    def test_get_latest_anchor(self) -> None:
        blockchain = MockBlockchain()
        blockchain.anchor("cluster_0", round_num=1, cid="cid1", hash_val="h1")
        blockchain.anchor("cluster_0", round_num=5, cid="cid5", hash_val="h5")
        blockchain.anchor("cluster_0", round_num=3, cid="cid3", hash_val="h3")
        latest = blockchain.get_latest_anchor("cluster_0")
        assert latest is not None
        assert latest.round_num == 5
        assert latest.cid == "cid5"

    def test_get_latest_anchor_nonexistent_cluster(self) -> None:
        blockchain = MockBlockchain()
        assert blockchain.get_latest_anchor("nonexistent") is None

    def test_multiple_clusters(self) -> None:
        blockchain = MockBlockchain()
        blockchain.anchor("cluster_0", round_num=1, cid="c0_cid", hash_val="c0_h")
        blockchain.anchor("cluster_1", round_num=1, cid="c1_cid", hash_val="c1_h")
        assert blockchain.get_anchor("cluster_0", 1) == ("c0_cid", "c0_h")
        assert blockchain.get_anchor("cluster_1", 1) == ("c1_cid", "c1_h")

    def test_clear(self) -> None:
        blockchain = MockBlockchain()
        blockchain.anchor("cluster_0", round_num=1, cid="cid", hash_val="h")
        blockchain.clear()
        assert blockchain.get_anchor("cluster_0", 1) is None


class TestMockBlockchainFile:
    """Tests for MockBlockchain with file storage."""

    def test_anchor_and_get_with_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "blockchain.json")
            blockchain = MockBlockchain(storage_path=path)
            blockchain.anchor("cluster_0", round_num=5, cid="abc", hash_val="def")
            result = blockchain.get_anchor("cluster_0", round_num=5)
            assert result == ("abc", "def")

    def test_persistence_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "blockchain.json")
            bc1 = MockBlockchain(storage_path=path)
            bc1.anchor("cluster_0", round_num=1, cid="cid1", hash_val="h1")
            bc2 = MockBlockchain(storage_path=path)
            result = bc2.get_anchor("cluster_0", round_num=1)
            assert result == ("cid1", "h1")



class TestModelAnchor:
    """Tests for ModelAnchor dataclass."""

    def test_creation(self) -> None:
        anchor = ModelAnchor(cluster_id="c0", round_num=5, cid="abc", hash="def")
        assert anchor.cluster_id == "c0"
        assert anchor.round_num == 5
        assert anchor.cid == "abc"
        assert anchor.hash == "def"


class TestIntegration:
    """Integration tests for IPFS and Blockchain working together."""

    def test_full_flow_in_memory(self) -> None:
        ipfs = MockIPFS()
        blockchain = MockBlockchain()
        model = np.random.randn(100)
        cid = ipfs.add(model)
        hash_val = compute_model_hash(model)
        blockchain.anchor("cluster_0", round_num=1, cid=cid, hash_val=hash_val)
        retrieved_cid, retrieved_hash = blockchain.get_anchor("cluster_0", 1)
        retrieved_model = ipfs.get(retrieved_cid)
        assert retrieved_model is not None
        assert verify_model_hash(retrieved_model, retrieved_hash)
        np.testing.assert_array_equal(retrieved_model, model)

    def test_full_flow_with_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs_path = str(Path(tmpdir) / "ipfs")
            bc_path = str(Path(tmpdir) / "blockchain.json")
            ipfs = MockIPFS(storage_path=ipfs_path)
            blockchain = MockBlockchain(storage_path=bc_path)
            model = np.random.randn(100)
            cid = ipfs.add(model)
            hash_val = compute_model_hash(model)
            blockchain.anchor("cluster_0", round_num=1, cid=cid, hash_val=hash_val)
            ipfs2 = MockIPFS(storage_path=ipfs_path)
            blockchain2 = MockBlockchain(storage_path=bc_path)
            retrieved_cid, retrieved_hash = blockchain2.get_anchor("cluster_0", 1)
            retrieved_model = ipfs2.get(retrieved_cid)
            assert retrieved_model is not None
            assert verify_model_hash(retrieved_model, retrieved_hash)
            np.testing.assert_array_equal(retrieved_model, model)

    def test_multiple_clusters_sharing_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ipfs_path = str(Path(tmpdir) / "ipfs")
            bc_path = str(Path(tmpdir) / "blockchain.json")
            ipfs = MockIPFS(storage_path=ipfs_path)
            blockchain = MockBlockchain(storage_path=bc_path)
            model_c0 = np.array([1.0, 0.0])
            model_c1 = np.array([0.0, 1.0])
            cid_c0 = ipfs.add(model_c0)
            cid_c1 = ipfs.add(model_c1)
            blockchain.anchor("cluster_0", 1, cid_c0, compute_model_hash(model_c0))
            blockchain.anchor("cluster_1", 1, cid_c1, compute_model_hash(model_c1))
            # Cluster 0 fetches cluster 1's model.
            c1_cid, c1_hash = blockchain.get_anchor("cluster_1", 1)
            c1_model = ipfs.get(c1_cid)
            assert verify_model_hash(c1_model, c1_hash)
            np.testing.assert_array_equal(c1_model, model_c1)
