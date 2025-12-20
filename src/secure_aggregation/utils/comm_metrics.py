"""Communication metrics utilities for gRPC message tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from google.protobuf.message import Message


@dataclass
class MessageStats:
    """Statistics for a single gRPC message exchange."""

    method: str
    request_size: int
    response_size: int
    latency_ms: float
    timestamp: float
    phase: str = ""


class CommunicationTracker:
    """Tracks gRPC communication metrics per round."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._messages: List[MessageStats] = []
        self._current_round: int = 0
        self._current_phase: str = ""
        self._round_messages: Dict[int, List[MessageStats]] = {}

    def set_round(self, round_idx: int) -> None:
        """Set the current round for tracking."""
        self._current_round = round_idx
        if round_idx not in self._round_messages:
            self._round_messages[round_idx] = []

    def set_phase(self, phase: str) -> None:
        """Set the current phase for categorization."""
        self._current_phase = phase

    def record_message(
        self,
        method: str,
        request_size: int,
        response_size: int,
        latency_ms: float,
    ) -> None:
        """Record a single message exchange."""
        stats = MessageStats(
            method=method,
            request_size=request_size,
            response_size=response_size,
            latency_ms=latency_ms,
            timestamp=time.time(),
            phase=self._current_phase,
        )
        self._messages.append(stats)
        if self._current_round in self._round_messages:
            self._round_messages[self._current_round].append(stats)

    def get_round_stats(self, round_idx: int) -> Dict[str, Any]:
        """Get aggregated stats for a specific round."""
        messages = self._round_messages.get(round_idx, [])
        if not messages:
            return {
                "bytes_sent": 0,
                "bytes_received": 0,
                "messages_sent": 0,
                "messages_received": 0,
                "avg_latency_ms": 0.0,
                "by_phase": {},
            }

        bytes_sent = sum(m.request_size for m in messages)
        bytes_received = sum(m.response_size for m in messages)
        avg_latency = sum(m.latency_ms for m in messages) / len(messages)

        # Group by phase
        by_phase: Dict[str, Dict[str, int]] = {}
        for m in messages:
            phase = m.phase or "unknown"
            if phase not in by_phase:
                by_phase[phase] = {"bytes_sent": 0, "bytes_received": 0}
            by_phase[phase]["bytes_sent"] += m.request_size
            by_phase[phase]["bytes_received"] += m.response_size

        return {
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "messages_sent": len(messages),
            "messages_received": len(messages),
            "avg_latency_ms": avg_latency,
            "by_phase": by_phase,
        }

    def get_total_stats(self) -> Dict[str, Any]:
        """Get total stats across all rounds."""
        all_messages = self._messages
        if not all_messages:
            return {
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
                "total_messages": 0,
            }

        return {
            "total_bytes_sent": sum(m.request_size for m in all_messages),
            "total_bytes_received": sum(m.response_size for m in all_messages),
            "total_messages": len(all_messages),
        }

    def reset(self) -> None:
        """Clear all tracked messages."""
        self._messages.clear()
        self._round_messages.clear()
        self._current_round = 0
        self._current_phase = ""


def get_message_size(msg: "Message") -> int:
    """Get serialized size of a protobuf message."""
    return msg.ByteSize()


def track_rpc_call(
    request: "Message",
    response: "Message",
    method_name: str,
    latency_ms: float,
    tracker: Optional[CommunicationTracker] = None,
) -> Tuple[int, int]:
    """
    Track a single RPC call's communication cost.

    Returns:
        Tuple of (request_bytes, response_bytes).
    """
    req_size = get_message_size(request)
    resp_size = get_message_size(response)

    if tracker:
        tracker.record_message(method_name, req_size, resp_size, latency_ms)

    return req_size, resp_size
