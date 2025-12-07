"""gRPC communication layer for secure aggregation."""

from . import secureagg_pb2, secureagg_pb2_grpc
from .bridge_service import BridgeClient, BridgeServicer, serve_bridge
from .inter_cluster_aggregator import InterClusterAggregator

__all__ = [
    "secureagg_pb2",
    "secureagg_pb2_grpc",
    "BridgeClient",
    "BridgeServicer",
    "serve_bridge",
    "InterClusterAggregator",
]
