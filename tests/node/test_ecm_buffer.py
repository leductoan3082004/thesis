"""Tests for ECM buffer functionality."""

import time

import pytest

from secure_aggregation.node.ecm_buffer import ECM, ECMBuffer


class TestECM:
    """Tests for ECM dataclass."""

    def test_creates_with_required_fields(self) -> None:
        ecm = ECM(cid="abc123", hash="def456")
        assert ecm.cid == "abc123"
        assert ecm.hash == "def456"
        assert ecm.source_cluster is None

    def test_creates_with_optional_fields(self) -> None:
        ecm = ECM(cid="abc", hash="def", source_cluster="cluster_0", received_at=100.0)
        assert ecm.source_cluster == "cluster_0"
        assert ecm.received_at == 100.0

    def test_auto_sets_timestamp(self) -> None:
        before = time.time()
        ecm = ECM(cid="abc", hash="def")
        after = time.time()
        assert before <= ecm.received_at <= after


class TestECMBuffer:
    """Tests for ECMBuffer class."""

    def test_initialization_requires_positive_window(self) -> None:
        with pytest.raises(ValueError, match="freshness_window must be positive"):
            ECMBuffer(freshness_window=0)

    def test_add_ecm(self) -> None:
        buffer = ECMBuffer(freshness_window=300.0)
        ecm = ECM(cid="abc", hash="def")
        buffer.add(ecm)
        assert len(buffer) == 1

    def test_add_from_message(self) -> None:
        buffer = ECMBuffer(freshness_window=300.0)
        buffer.add_from_message(cid="abc", hash_val="def", source_cluster="c0")
        assert len(buffer) == 1
        ecms = buffer.get_all()
        assert ecms[0].cid == "abc"
        assert ecms[0].source_cluster == "c0"

    def test_deduplicates_by_cid(self) -> None:
        buffer = ECMBuffer(freshness_window=300.0)
        ecm1 = ECM(cid="same_cid", hash="hash1", received_at=100.0)
        ecm2 = ECM(cid="same_cid", hash="hash2", received_at=200.0)
        buffer.add(ecm1)
        buffer.add(ecm2)
        assert len(buffer) == 1
        # Should keep the newer one.
        ecms = buffer.get_all()
        assert ecms[0].hash == "hash2"

    def test_keeps_older_if_newer_not_provided(self) -> None:
        buffer = ECMBuffer(freshness_window=300.0)
        ecm1 = ECM(cid="same_cid", hash="hash1", received_at=200.0)
        ecm2 = ECM(cid="same_cid", hash="hash2", received_at=100.0)
        buffer.add(ecm1)
        buffer.add(ecm2)
        ecms = buffer.get_all()
        assert ecms[0].hash == "hash1"

    def test_get_fresh_ecms_filters_by_window(self) -> None:
        buffer = ECMBuffer(freshness_window=100.0)
        now = time.time()
        fresh_ecm = ECM(cid="fresh", hash="h1", received_at=now - 50)
        stale_ecm = ECM(cid="stale", hash="h2", received_at=now - 150)
        buffer.add(fresh_ecm)
        buffer.add(stale_ecm)
        fresh = buffer.get_fresh_ecms(now=now)
        assert len(fresh) == 1
        assert fresh[0].cid == "fresh"

    def test_get_unique_cids(self) -> None:
        buffer = ECMBuffer(freshness_window=300.0)
        buffer.add_from_message("cid1", "hash1")
        buffer.add_from_message("cid2", "hash2")
        buffer.add_from_message("cid1", "hash1_updated")  # Duplicate CID.
        unique = buffer.get_unique_cids()
        assert len(unique) == 2
        assert "cid1" in unique
        assert "cid2" in unique

    def test_clear(self) -> None:
        buffer = ECMBuffer(freshness_window=300.0)
        buffer.add_from_message("cid1", "hash1")
        buffer.add_from_message("cid2", "hash2")
        assert len(buffer) == 2
        buffer.clear()
        assert len(buffer) == 0

    def test_remove_stale(self) -> None:
        buffer = ECMBuffer(freshness_window=100.0)
        now = time.time()
        buffer.add(ECM(cid="fresh", hash="h1", received_at=now - 50))
        buffer.add(ECM(cid="stale1", hash="h2", received_at=now - 150))
        buffer.add(ECM(cid="stale2", hash="h3", received_at=now - 200))
        removed = buffer.remove_stale(now=now)
        assert removed == 2
        assert len(buffer) == 1

    def test_thread_safety(self) -> None:
        """Basic thread safety test with concurrent adds."""
        import threading

        buffer = ECMBuffer(freshness_window=300.0)
        errors = []

        def add_ecms(prefix: str) -> None:
            try:
                for i in range(100):
                    buffer.add_from_message(f"{prefix}_{i}", f"hash_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_ecms, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Should have 500 unique CIDs.
        assert len(buffer) == 500
