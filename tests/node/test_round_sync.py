"""Tests for the round-sync decision engine and SapResult."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from secure_aggregation.communication import secureagg_pb2
from secure_aggregation.communication.node_service import (
    FatalSyncError,
    NodeService,
    SapResult,
)


class TestEvaluateSyncDecision:
    def test_sync_ok_returns_proceed(self):
        result = NodeService._evaluate_sync_decision(
            secureagg_pb2.ROUND_SYNC_OK, 5, 5, "round0"
        )
        assert result == "PROCEED"

    def test_sync_stale_returns_passive_wait(self):
        result = NodeService._evaluate_sync_decision(
            secureagg_pb2.ROUND_SYNC_STALE, 3, 5, "round0"
        )
        assert result == "PASSIVE_WAIT"

    def test_sync_finalized_returns_passive_wait(self):
        result = NodeService._evaluate_sync_decision(
            secureagg_pb2.ROUND_SYNC_FINALIZED, 5, 5, "round1"
        )
        assert result == "PASSIVE_WAIT"

    def test_sync_ahead_returns_retry(self):
        result = NodeService._evaluate_sync_decision(
            secureagg_pb2.ROUND_SYNC_AHEAD, 5, 3, "round0"
        )
        assert result == "RETRY_SAME_PHASE"

    def test_sync_not_member_returns_fatal(self):
        result = NodeService._evaluate_sync_decision(
            secureagg_pb2.ROUND_SYNC_NOT_MEMBER, 0, 0, "round0"
        )
        assert result == "FATAL"

    def test_sync_unspecified_returns_proceed(self):
        result = NodeService._evaluate_sync_decision(
            secureagg_pb2.ROUND_SYNC_UNSPECIFIED, 0, 0, "round0"
        )
        assert result == "PROCEED"


class TestSapResult:
    def test_defaults(self):
        r = SapResult()
        assert r.passive_wait is False
        assert r.target_round == 0
        assert r.aggregated_weights == []

    def test_passive_result(self):
        r = SapResult(passive_wait=True, target_round=7)
        assert r.passive_wait is True
        assert r.target_round == 7
        assert r.aggregated_weights == []

    def test_normal_result(self):
        weights = [1.0, 2.0, 3.0]
        r = SapResult(aggregated_weights=weights)
        assert r.passive_wait is False
        assert r.aggregated_weights == weights


class TestSyncHandling:
    def test_check_sync_or_abort_raises_fatal_sync_error_for_not_member(self):
        node = NodeService.__new__(NodeService)
        node.node_id = "node_a"
        node.current_round = 5
        node.prom_metrics = None

        response = SimpleNamespace(
            sync_code=secureagg_pb2.ROUND_SYNC_NOT_MEMBER,
            server_round=5,
        )
        with pytest.raises(FatalSyncError):
            node._check_sync_or_abort(response, "round1")


class TestMergeCommStats:
    def test_merge_comm_stats_combines_totals_and_phase_breakdown(self):
        first = {
            "bytes_sent": 10,
            "bytes_received": 20,
            "messages_sent": 2,
            "messages_received": 2,
            "avg_latency_ms": 100.0,
            "by_phase": {"sap_round0": {"bytes_sent": 10, "bytes_received": 20}},
        }
        second = {
            "bytes_sent": 5,
            "bytes_received": 7,
            "messages_sent": 1,
            "messages_received": 1,
            "avg_latency_ms": 40.0,
            "by_phase": {"model_fetch": {"bytes_sent": 5, "bytes_received": 7}},
        }

        merged = NodeService._merge_comm_stats(first, second)
        assert merged["bytes_sent"] == 15
        assert merged["bytes_received"] == 27
        assert merged["messages_sent"] == 3
        assert merged["messages_received"] == 3
        assert merged["avg_latency_ms"] == pytest.approx((100.0 * 2 + 40.0 * 1) / 3)
        assert merged["by_phase"]["sap_round0"]["bytes_sent"] == 10
        assert merged["by_phase"]["sap_round0"]["bytes_received"] == 20
        assert merged["by_phase"]["model_fetch"]["bytes_sent"] == 5
        assert merged["by_phase"]["model_fetch"]["bytes_received"] == 7
