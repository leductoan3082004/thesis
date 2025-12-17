"""gRPC communication layer for secure aggregation."""

import sys

from . import secureagg_pb2 as _secureagg_pb2

# Ensure protoc modules are importable via their legacy top-level names before other imports.
sys.modules.setdefault("secureagg_pb2", _secureagg_pb2)

from . import secureagg_pb2_grpc as _secureagg_pb2_grpc

sys.modules.setdefault("secureagg_pb2_grpc", _secureagg_pb2_grpc)

secureagg_pb2 = _secureagg_pb2
secureagg_pb2_grpc = _secureagg_pb2_grpc
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
