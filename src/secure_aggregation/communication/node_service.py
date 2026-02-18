"""Node service for federated learning with secure aggregation."""

import argparse
import copy
import json
import math
import os
import re
import socket
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import grpc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from secure_aggregation.communication import secureagg_pb2, secureagg_pb2_grpc
from secure_aggregation.communication.aggregator_service import (
    AggregatorServicer,
    PortBindingError,
    grpc_message_options,
    serve as serve_aggregator,
)
from secure_aggregation.communication.bridge_service import (
    BridgeClient,
    serve_bridge,
)
from secure_aggregation.communication.inter_cluster_aggregator import InterClusterAggregator
from secure_aggregation.communication.hierarchy_mixin import (
    HierarchyMixin,
    ScopeRoundHandler,
    ScopeRuntime,
)
from secure_aggregation.convergence import ConvergenceConfig, ConvergenceState, ConvergenceTracker
from secure_aggregation.convergence.central_broadcast import (
    CENTRAL_METADATA_CLUSTER_ID,
    fetch_central_metadata,
)
from secure_aggregation.config.models import NodeRole
from secure_aggregation.config.system import load_system_config
from secure_aggregation.crypto.sign import SigningKeyPair
from secure_aggregation.data import dirichlet_partition, get_labels, load_dataset
from secure_aggregation.node import ECM, ECMBuffer, NodeEngine, NodeRuntimeConfig, ReliabilityScore
from secure_aggregation.protocol import MergeConfig, SecureAggregationNode
from secure_aggregation.protocol.core import AdvertiseMessage, Round1Ciphertext, SHARE_BYTES, _int_to_bytes
from secure_aggregation.state import HierarchyLevelConfig, StateAggregator
from secure_aggregation.storage.model_store import (
    AnchorScope,
    BlockchainInterface,
    GatewayBlockchain,
    IPFSInterface,
    KuboIPFS,
    MockBlockchain,
    MockIPFS,
    compute_model_hash,
    verify_model_hash,
)
from secure_aggregation.topology import (
    compute_average_degree,
    compute_max_degree,
    compute_node_degrees,
    elect_clique_aggregator,
    get_inter_clique_neighbors,
    is_bridge_node,
)
from secure_aggregation.utils import (
    configure_logging,
    get_logger,
    CommunicationTracker,
    track_rpc_call,
)
from secure_aggregation.utils.prometheus_metrics import PrometheusMetrics

logger = get_logger("node_service")


class MnistLinear(nn.Module):
    """Simple linear classifier for vectorized image inputs (e.g., MNIST)."""

    def __init__(self, input_shape: Tuple[int, ...] = (1, 28, 28), num_classes: int = 10) -> None:
        super().__init__()
        features = 1
        for dim in input_shape:
            features *= dim
        self.fc = nn.Linear(features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CifarConvNet(nn.Module):
    """Compact convolutional network for CIFAR-sized RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def flatten_params(model: nn.Module) -> List[float]:
    """Flatten model parameters to list of floats."""
    vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    return vec.tolist()


def load_params(model: nn.Module, flat: List[float]) -> None:
    """Load flattened parameters into model."""
    tensor = torch.tensor(flat, dtype=torch.float32)
    torch.nn.utils.vector_to_parameters(tensor, model.parameters())


def quantize_vector(vec: List[float], scale: float) -> List[int]:
    """Quantize float vector to integers."""
    return [int(round(v * scale)) for v in vec]


def dequantize_vector(ints: List[int], scale: float) -> List[float]:
    """Dequantize integer vector to floats."""
    return [float(i) / scale for i in ints]


def _encode_share(x: int, share: int) -> bytes:
    """Pack (x, share) tuple for transport."""
    return _int_to_bytes(x, 2) + _int_to_bytes(share, SHARE_BYTES)


class GlobalStopRequested(Exception):
    """Raised when global convergence has been confirmed and execution should halt."""


class AggregatorUnavailable(Exception):
    """Raised when the elected aggregator cannot be reached after repeated attempts."""


class NodeService(HierarchyMixin):
    """Node service that coordinates training and secure aggregation."""

    def __init__(self, config_path: str) -> None:
        self._config_path = config_path
        self.config = self._load_config(config_path)
        self.system_config, self.system_config_path = load_system_config(Path(config_path))
        self.node_id = self.config["node_id"]
        self.state_id = self.config.get("state_id")
        self.role = NodeRole(self.config["role"])
        self.ttp_address = self.config["ttp_address"]
        self.port = self.config["port"]
        self.aggregator_port_offset = self._resolve_port_offset("aggregator_port_offset", 1000)
        self.bridge_port_offset = self._resolve_port_offset("bridge_port_offset", 2000)
        self._aggregator_port_guard: Optional[socket.socket] = None
        self._bridge_port_guard: Optional[socket.socket] = None
        self._port_guard_notices: Set[str] = set()
        self._initialize_port_guards()
        self.network_host = (
            self.config.get("network_host")
            or os.environ.get("NODE_HOSTNAME")
            or os.environ.get("HOSTNAME")
            or self.node_id
        )
        self.dataset_config = self.config["dataset"]
        self.dataset_name: str = self.dataset_config.get("name", "mnist")
        self.dataset_input_shape: Optional[Tuple[int, ...]] = None
        self.dataset_num_classes: Optional[int] = None
        self.training_config = self.config["training"]
        self.secagg_config = self.config["secure_agg"]
        self.threshold = self.secagg_config["threshold"]
        self.scale = self.secagg_config["scale"]
        env_rounds = os.getenv("MAX_TRAINING_ROUNDS")
        default_round_cap = 200
        system_training_cfg = (self.system_config or {}).get("training") if self.system_config else None
        system_rounds = None
        if isinstance(system_training_cfg, dict):
            system_rounds = system_training_cfg.get("max_rounds")
        if env_rounds:
            try:
                self.max_training_rounds = max(1, int(env_rounds))
            except ValueError:
                logger.warning(
                    "Invalid MAX_TRAINING_ROUNDS=%s; falling back to configured/default value",
                    env_rounds,
                )
                self.max_training_rounds = (
                    max(1, int(system_rounds)) if system_rounds else default_round_cap
                )
        elif system_rounds:
            try:
                self.max_training_rounds = max(1, int(system_rounds))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid system training.rounds=%s; falling back to default %d",
                    system_rounds,
                    default_round_cap,
                )
                self.max_training_rounds = default_round_cap
        else:
            self.max_training_rounds = default_round_cap

        # State
        self.signing_keypair: Optional[SigningKeyPair] = None
        self.participants: List[secureagg_pb2.NodeInfo] = []
        self.participant_map: Dict[str, str] = {}  # node_id -> address
        self.model: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.current_round = 0
        self.is_aggregator = False
        self.aggregator_id: Optional[str] = None
        self.aggregator_address: Optional[str] = None
        self.aggregator_server: Optional[grpc.Server] = None

        # Clique membership from TTP
        self.clique_id: int = -1
        self.clique_members: List[str] = []
        self.clique_threshold: int = 0
        self.assigned_data_indices: List[int] = []

        # Inter-cluster aggregation state
        self.inter_cluster_config = self.config.get("inter_cluster", {})
        self.inter_cluster_enabled = self.inter_cluster_config.get("enabled", False)
        self.inter_edges: List[Tuple[str, str]] = []
        self.is_bridge_node = False
        self.neighbor_bridge_addresses: List[str] = []
        self.neighbor_address_map: Dict[str, str] = {}
        self.central_neighbor_addresses: Dict[str, str] = {}
        self.scope_configs = self._load_scope_config()
        self._init_scope_role_pools()
        self._init_scope_timers()
        self._state_rosters, self._state_cluster_map = self._load_state_metadata()
        self.ecm_buffer: Optional[ECMBuffer] = None
        self.bridge_server: Optional[grpc.Server] = None
        self.bridge_client: Optional[BridgeClient] = None
        self.inter_cluster_aggregator: Optional[InterClusterAggregator] = None
        self.ipfs: Optional[IPFSInterface] = None
        self.blockchain: Optional[BlockchainInterface] = None
        self.ecm_forward_wait = float(self.inter_cluster_config.get("ecm_forward_wait_seconds", 5.0))

        # State-level aggregation (hierarchy) state
        (
            self.scope_name,
            self.scope_config,
            self.higher_scope_name,
            higher_scope_config,
        ) = self._select_scope_roles(self.scope_configs)
        self.higher_scope_config = higher_scope_config or HierarchyLevelConfig(scope_index=self.scope_config.scope_index + 1)
        scope_interval = self._scope_interval_seconds(self.scope_config)
        higher_interval = self._scope_interval_seconds(self.higher_scope_config)
        logger.info(
            "%s aggregation config: enabled=%s, rounds_per_scope=%s, interval_seconds=%.1f, scope_id=%s",
            self._scope_label_upper(),
            self.scope_config.enabled,
            self.scope_config.rounds_per_scope,
            scope_interval,
            self.scope_config.scope_id,
        )
        logger.info(
            "%s scheduling config: enabled=%s, rounds_per_scope=%s, interval_seconds=%.1f, apply_policy=%s, apply_alpha=%.3f",
            self._higher_scope_label_upper(),
            self.higher_scope_config.enabled,
            self.higher_scope_config.rounds_per_scope,
            higher_interval,
            getattr(self.higher_scope_config, "apply_policy", "replace"),
            float(getattr(self.higher_scope_config, "apply_alpha", 0.0) or 0.0),
        )
        self._scope_runtimes: Dict[str, ScopeRuntime] = {}
        for runtime_name, runtime_config in self.scope_configs.items():
            self._ensure_scope_runtime(runtime_name, runtime_config)
        self._bridge_ecm_hooks: List[Callable[[ECM], None]] = []
        self._scope_last_applied_rounds: Dict[str, int] = defaultdict(int)
        self._scope_last_applied_cids: Dict[str, str] = {}
        self._pending_scope_waits: Deque[Tuple[str, float, float]] = deque()
        self._ready_scope_fetches: Set[str] = set()
        self._scope_execution_order = [
            name for name, _ in sorted(self.scope_configs.items(), key=lambda item: item[1].scope_index)
        ]
        self._build_scope_handlers()
        self._prime_scope_fetches()
        self._configure_scope_layer()

        # Convergence state
        self.convergence_config = self._load_convergence_config()
        self._convergence_runtime_enabled = self._should_enable_convergence_runtime()
        if not self._convergence_runtime_enabled:
            self.convergence_config.enabled = False
        self.convergence_tracker: Optional[ConvergenceTracker] = None
        self._latest_cluster_converged: bool = False
        self._latest_delta_norm: float = 0.0
        self._latest_convergence_streak: int = 0
        self._last_model_cid: Optional[str] = None
        self._last_model_hash: Optional[str] = None
        self._last_model_data_id: Optional[str] = None
        self.central_metadata = None
        self.aggregator_servicer: Optional[AggregatorServicer] = None
        self._bootstrap_anchors: List[
            Tuple[str, int, str, Optional[str], Optional[str], AnchorScope]
        ] = []
        self._logged_central_addresses = False
        max_failures = self.secagg_config.get("max_aggregator_failure_rounds", 2)
        try:
            self._max_aggregator_failure_rounds = max(1, int(max_failures))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid max_aggregator_failure_rounds=%s; defaulting to 2", max_failures
            )
            self._max_aggregator_failure_rounds = 2
        self._consecutive_aggregator_failures = 0

        # Metrics tracking state
        self.prom_metrics: Optional[PrometheusMetrics] = None
        self.comm_tracker: Optional[CommunicationTracker] = None
        self.val_loader: Optional[DataLoader] = None
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []

        logger.info(f"Node {self.node_id} initialized (role={self.role}, port={self.port})")

    def _load_config(self, path: str) -> dict:
        """Load node configuration from JSON file."""
        with open(path) as f:
            return json.load(f)

    def _resolve_port_offset(self, key: str, default: int) -> int:
        """Parse a configurable port offset, falling back to a sane default."""
        value = self.config.get(key, default)
        try:
            offset = int(value)
        except (TypeError, ValueError):
            logger.warning("%s=%s is invalid; defaulting to %d", key, value, default)
            return default
        if offset < 0:
            logger.warning("%s=%s cannot be negative; defaulting to %d", key, value, default)
            return default
        return offset

    def _initialize_port_guards(self) -> None:
        """Reserve aggregator/bridge ports so the OS never assigns them to ephemeral sockets."""
        self._ensure_port_guard("_aggregator_port_guard", self._aggregator_listen_port(), "aggregator")
        self._ensure_port_guard("_bridge_port_guard", self._bridge_listen_port(), "bridge")

    def _acquire_port_guard(self, port: int, label: str) -> socket.socket:
        """Bind a dummy socket to keep a port reserved until the real server starts."""
        last_error: Optional[Exception] = None
        for family, bind_addr in (
            (socket.AF_INET6, ("::", port)),
            (socket.AF_INET, ("0.0.0.0", port)),
        ):
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
                    # Mirror gRPC's default dual-stack behavior.
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                sock.bind(bind_addr)
                sock.listen(1)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
        raise RuntimeError(
            f"Port guard for {label} failed on port {port}: {last_error}"
        )

    def _ensure_port_guard(self, attr: str, port: int, label: str) -> None:
        """Ensure we hold a guard socket for the given label."""
        guard = getattr(self, attr, None)
        if guard:
            return
        guard = self._acquire_port_guard(port, label)
        setattr(self, attr, guard)
        if label not in self._port_guard_notices:
            logger.info("Reserved %s port %d for node %s", label, port, self.node_id)
            self._port_guard_notices.add(label)

    def _release_port_guard(self, attr: str) -> None:
        """Close and clear the guard socket for the supplied attribute."""
        guard = getattr(self, attr, None)
        if guard:
            try:
                guard.close()
            finally:
                setattr(self, attr, None)

    def _load_convergence_config(self) -> ConvergenceConfig:
        """Load convergence configuration preferring the shared system config."""
        system_convergence = (self.system_config or {}).get("convergence")
        if system_convergence is not None:
            return ConvergenceConfig.from_dict(system_convergence)
        node_convergence = self.config.get("convergence")
        if node_convergence is not None:
            logger.warning(
                "Using per-node convergence config for %s; please move it to %s",
                self.node_id,
                self.system_config_path,
            )
            return ConvergenceConfig.from_dict(node_convergence)
        return ConvergenceConfig()

    def _should_enable_convergence_runtime(self) -> bool:
        """Determine if convergence detection runtime should be active."""
        env_toggle = os.getenv("ENABLE_CLUSTER_CONVERGENCE")
        if env_toggle is not None:
            return env_toggle.lower() in {"1", "true", "yes", "on"}

        def _resolve_runtime_flag(source: Optional[Dict[str, Any]]) -> Optional[bool]:
            if not source or not isinstance(source, dict):
                return None
            if "runtime_enabled" in source:
                return bool(source["runtime_enabled"])
            return None

        system_flag = _resolve_runtime_flag((self.system_config or {}).get("convergence"))
        if system_flag is not None:
            return system_flag
        node_flag = _resolve_runtime_flag(self.config.get("convergence"))
        if node_flag is not None:
            return node_flag
        return False

    def _is_convergence_runtime_enabled(self) -> bool:
        """Helper to check if convergence runtime is active."""
        return bool(self._convergence_runtime_enabled)

    def _prime_convergence_tracker_state(self) -> None:
        """Advance convergence tracker state without computing deltas."""
        if not self.convergence_tracker or not self.convergence_config.enabled:
            return
        if self.model is None:
            return
        flat_params = np.array(flatten_params(self.model), dtype=np.float32)
        self.convergence_tracker.update(flat_params, track_diff=False)
        self.convergence_tracker.state.convergence_streak = self._latest_convergence_streak

    def _update_convergence_state_from_model(
        self,
        model_array: np.ndarray,
        *,
        model_cid: Optional[str] = None,
        model_hash: Optional[str] = None,
        model_data_id: Optional[str] = None,
    ) -> Optional[ConvergenceState]:
        """Run tracker update and propagate state to the local aggregator server."""
        if self.convergence_tracker and self.convergence_config.enabled:
            conv_state = self.convergence_tracker.update(model_array)
            self._latest_cluster_converged = conv_state.cluster_converged
            self._latest_delta_norm = conv_state.delta_norm
            self._latest_convergence_streak = conv_state.convergence_streak
        else:
            conv_state = ConvergenceState()
            conv_state.delta_norm = 0.0
            conv_state.cluster_converged = False
            conv_state.convergence_streak = self._latest_convergence_streak
            conv_state.should_stop = False
            conv_state.stop_reason = ""
            self._latest_cluster_converged = False
            self._latest_delta_norm = 0.0

        if self.aggregator_servicer:
            self.aggregator_servicer.set_convergence_state(
                model_cid=model_cid,
                model_hash=model_hash,
                model_data_id=model_data_id,
                should_stop=conv_state.should_stop,
                stop_reason=conv_state.stop_reason,
                delta_norm=conv_state.delta_norm,
                cluster_converged=conv_state.cluster_converged,
                convergence_streak=conv_state.convergence_streak,
            )
        return conv_state

    def _apply_model_tensor(self, tensor: np.ndarray) -> None:
        """Load flattened tensor data into the local model."""
        if not self.model:
            return
        load_params(self.model, tensor.flatten().tolist())
        self._prime_convergence_tracker_state()

    def _export_local_model_vector(self) -> Optional[np.ndarray]:
        """Return flattened local model parameters as numpy array."""
        if not self.model:
            return None
        return np.array(flatten_params(self.model), dtype=np.float32)

    def _log_final_model_status(self) -> None:
        """Log the final model accuracy and identifiers before stopping."""
        accuracy = self.evaluate()
        cid = self._last_model_cid or "N/A"
        data_id = self._last_model_data_id or "N/A"
        logger.info(
            "Final model checkpoint: accuracy=%.4f, cid=%s, data_id=%s",
            accuracy,
            cid,
            data_id,
        )

    def _maybe_check_convergence_during_retry(
        self,
        attempt_idx: int,
        max_attempts: int,
        stage: str,
        first_check: int = 40,
        interval: int = 5,
    ) -> None:
        """
        Periodically poll for convergence while retrying aggregator RPCs.

        Args:
            attempt_idx: Current attempt number (1-based).
            max_attempts: Total attempts allowed.
            stage: Human-readable stage description.
            first_check: Attempt number to trigger the first convergence check.
            interval: Attempt interval for subsequent checks.
        """
        if not self._is_convergence_runtime_enabled():
            return
        if attempt_idx < first_check:
            return
        if attempt_idx == first_check or ((attempt_idx - first_check) % interval == 0):
            logger.info(
                "Aggregator %s unavailable at stage %s (attempt %d/%d); re-checking convergence state",
                self.aggregator_id,
                stage,
                attempt_idx,
                max_attempts,
            )
            should_stop, stop_reason = self._refresh_convergence_state(False, "")
            if should_stop:
                logger.info(
                    "Halting retries because convergence was confirmed while waiting "
                    "(reason=%s)",
                    stop_reason,
                )
                raise GlobalStopRequested()

    def _raise_aggregator_unavailable(
        self,
        stage: str,
        attempts: int,
        last_error: Optional[Exception],
    ) -> None:
        """
        Raise AggregatorUnavailable with a detailed message.
        """
        reason = ""
        if isinstance(last_error, grpc.RpcError):
            code = last_error.code().name if hasattr(last_error, "code") else "UNKNOWN"
            details = last_error.details() if hasattr(last_error, "details") else ""
            reason = f" (last_error={code} {details})"
        elif last_error:
            reason = f" (last_error={last_error})"
        message = (
            f"Aggregator {self.aggregator_id} unreachable during {stage} after "
            f"{attempts} attempts{reason}"
        )
        raise AggregatorUnavailable(message)

    def _hydrate_anchor_bootstrap(self) -> None:
        """Persist bootstrap anchor references once blockchain client is ready."""
        if not self.blockchain or not self._bootstrap_anchors:
            return
        for cluster_id, round_num, data_id, cid, hash_val, scope in self._bootstrap_anchors:
            self.blockchain.remember_anchor(
                cluster_id,
                round_num,
                data_id,
                cid,
                hash_val,
                scope=scope or AnchorScope.CLUSTER,
            )
        self._bootstrap_anchors.clear()

    def register_with_ttp(self) -> None:
        """Register with TTP and receive signing keys and clique assignment."""
        logger.info(f"Registering with TTP at {self.ttp_address}")

        max_retries = 10
        for attempt in range(max_retries):
            try:
                channel = grpc.insecure_channel(self.ttp_address)
                stub = secureagg_pb2_grpc.TTPServiceStub(channel)

                request = secureagg_pb2.RegisterRequest(
                    node_id=self.node_id,
                    address=f"{self.network_host}:{self.port}"
                )
                response = stub.RegisterNode(request, timeout=5)

                if response.success:
                    self.signing_keypair = SigningKeyPair(
                        private_key=bytes(response.signing_private_key),
                        public_key=bytes(response.signing_public_key)
                    )

                    # Extract clique assignment from TTP response
                    self.clique_id = response.clique_id
                    self.clique_members = list(response.clique_members)
                    self.clique_threshold = response.clique_threshold
                    self.assigned_data_indices = list(response.data_indices)

                    # Use clique threshold if provided, otherwise fall back to config
                    if self.clique_threshold > 0:
                        self.threshold = self.clique_threshold

                logger.info(
                    f"Registered with TTP: clique={self.clique_id}, "
                    f"members={len(self.clique_members)}, threshold={self.threshold}, "
                    f"data_samples={len(self.assigned_data_indices)}"
                )
                metadata_data_id = getattr(response, "central_metadata_data_id", "")
                if metadata_data_id:
                    metadata_version = getattr(response, "central_metadata_version", 0)
                    self._bootstrap_anchors.append(
                        (
                            CENTRAL_METADATA_CLUSTER_ID,
                            metadata_version or 0,
                            metadata_data_id,
                            None,
                            None,
                            AnchorScope.CONTROL,
                        )
                    )
                else:
                    logger.info("No central metadata anchor provided by TTP; continuing without bootstrap.")

                # Get list of participants
                participants_response = stub.GetParticipants(secureagg_pb2.ParticipantsRequest())
                self.participants = list(participants_response.participants)
                self.participant_map = {p.node_id: p.address for p in self.participants}
                logger.info(f"Retrieved {len(self.participants)} total participants")

                channel.close()
                return

            except grpc.RpcError as e:
                logger.warning(f"TTP connection attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(2)

        raise RuntimeError("Failed to connect to TTP after max retries")

    def setup_data(self) -> None:
        """Setup dataset using config-driven loader with indices assigned by TTP or local partition."""
        dataset_name = self.dataset_config.get("name", "mnist")
        self.dataset_name = dataset_name
        # Resolve datasets config: explicit config value, then relative to node config file, then Docker fallback.
        _default_datasets = str(Path(self._config_path).resolve().parent.parent / "datasets.json") if hasattr(self, "_config_path") else "/app/config/datasets.json"
        datasets_config_path = self.dataset_config.get("config_path", _default_datasets)
        logger.info(f"Setting up dataset: {dataset_name}")

        train_ds = load_dataset(dataset_name, datasets_config_path, train=True)
        test_ds = load_dataset(dataset_name, datasets_config_path, train=False)
        self._capture_dataset_metadata(train_ds)

        node_index = self._extract_node_index()
        num_clients = self.dataset_config["num_clients"]
        num_clients = max(num_clients, node_index + 1)
        if self.participants:
            num_clients = max(num_clients, len(self.participants))

        # Use TTP-assigned indices if available, otherwise compute locally.
        dataset_size = len(train_ds)
        indices_source = "TTP-assigned"

        if self.assigned_data_indices:
            indices = self._sanitize_indices(self.assigned_data_indices, dataset_size)
            logger.info(f"Using {len(indices)} {indices_source} data samples")
        else:
            labels = get_labels(train_ds)
            alpha = self.dataset_config["alpha"]
            seed = self.dataset_config.get("seed", 42)

            parts = dirichlet_partition(
                list(range(len(train_ds))), labels, num_clients=num_clients, alpha=alpha, seed=seed
            )

            client_key = f"client_{node_index}"
            indices = parts.get(client_key, [])
            if not indices:
                logger.warning(
                    "Dirichlet partition returned no samples for %s (client_%s); falling back to deterministic split",
                    self.node_id,
                    node_index,
                )
                indices = self._deterministic_partition(len(train_ds), num_clients, node_index)
            indices = self._sanitize_indices(indices, dataset_size)
            indices_source = "locally-computed"
            logger.info(f"Using {len(indices)} {indices_source} data samples")

        if not indices:
            fallback = self._deterministic_partition(dataset_size, num_clients, node_index)
            logger.warning(
                "No valid %s data indices for %s; falling back to deterministic partition of %d samples",
                indices_source,
                self.node_id,
                len(fallback),
            )
            indices = fallback

        # Split indices into train (80%) and validation (20%) for metrics tracking
        import random
        indices_copy = list(indices)
        random.seed(42)
        random.shuffle(indices_copy)
        split_point = int(len(indices_copy) * 0.8)
        self.train_indices = indices_copy[:split_point]
        self.val_indices = indices_copy[split_point:]

        batch_size = self.training_config["batch_size"]
        self.train_loader = DataLoader(Subset(train_ds, self.train_indices), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(Subset(train_ds, self.val_indices), batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        logger.info(f"Data split: {len(self.train_indices)} train, {len(self.val_indices)} validation samples")

    def _extract_node_index(self) -> int:
        """Extract trailing numeric component from node_id, regardless of delimiters."""
        suffix = self.node_id.split("_")[-1]
        if suffix.isdigit():
            return int(suffix)
        match = re.search(r"(\d+)$", self.node_id)
        if match:
            return int(match.group(1))
        raise ValueError(f"Node ID '{self.node_id}' does not end with a numeric index")

    @staticmethod
    def _deterministic_partition(dataset_size: int, num_clients: int, node_index: int) -> List[int]:
        """Split dataset evenly when probabilistic partitioning yields zero samples."""
        base = dataset_size // num_clients
        remainder = dataset_size % num_clients
        extra = 1 if node_index < remainder else 0
        start = node_index * base + min(node_index, remainder)
        end = min(dataset_size, start + base + extra)
        if start >= dataset_size:
            return [dataset_size - 1]
        return list(range(start, end))

    @staticmethod
    def _sanitize_indices(indices: Sequence[int], dataset_size: int) -> List[int]:
        """Drop invalid/duplicate indices to avoid dataset bounds issues."""
        valid: List[int] = []
        seen: Set[int] = set()
        dropped = 0
        for idx in indices:
            if not isinstance(idx, int) or idx < 0 or idx >= dataset_size:
                dropped += 1
                continue
            if idx in seen:
                continue
            seen.add(idx)
            valid.append(idx)
        if dropped:
            logger.warning(
                "Dropped %d invalid data indices outside dataset bounds (size=%d)",
                dropped,
                dataset_size,
            )
        if not valid and dataset_size:
            logger.error("No valid data indices remain after sanitization (dataset size=%d)", dataset_size)
        return valid

    def _capture_dataset_metadata(self, dataset: Any) -> None:
        """Capture dataset shape/num_classes for downstream model selection."""
        if self.dataset_input_shape is None:
            try:
                first_item = dataset[0]
                sample = first_item[0] if isinstance(first_item, (tuple, list)) else first_item
                if hasattr(sample, "shape"):
                    shape = tuple(int(dim) for dim in sample.shape)  # type: ignore[attr-defined]
                    self.dataset_input_shape = shape
            except Exception as exc:  # noqa: BLE001
                logger.debug("Unable to capture dataset input shape: %s", exc)

        if self.dataset_num_classes is None:
            num_classes: Optional[int] = None
            if hasattr(dataset, "classes"):
                classes = getattr(dataset, "classes")
                if isinstance(classes, Sequence):
                    num_classes = len(classes)
            elif hasattr(dataset, "class_to_idx"):
                mapping = getattr(dataset, "class_to_idx")
                if isinstance(mapping, Mapping):
                    num_classes = len(mapping)
            elif hasattr(dataset, "targets"):
                targets = getattr(dataset, "targets")
                if isinstance(targets, Sequence) and targets:
                    try:
                        num_classes = len(set(int(t) for t in targets))
                    except Exception:  # noqa: BLE001
                        num_classes = None
            if num_classes:
                self.dataset_num_classes = num_classes

    def setup_model(self) -> None:
        """Initialize model matching dataset characteristics."""
        dataset_name = (self.dataset_name or "mnist").lower()
        num_classes = self.dataset_num_classes or 10
        input_shape = self.dataset_input_shape or (1, 28, 28)

        if dataset_name.startswith("cifar"):
            self.model = CifarConvNet(num_classes=num_classes)
            model_name = "CifarConvNet"
        else:
            self.model = MnistLinear(input_shape=input_shape, num_classes=num_classes)
            model_name = "MnistLinear"

        logger.info(
            "Model initialized (%s, num_classes=%d, input_shape=%s)",
            model_name,
            num_classes,
            input_shape,
        )

    def train_local(self, epochs: int) -> Tuple[int, int]:
        """Train model locally for specified epochs.

        Returns:
            Tuple of (total_samples, total_batches) for metrics tracking.
        """
        if not self.model or not self.train_loader:
            raise RuntimeError("Model or data not initialized")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model.train()

        total_samples = 0
        total_batches = 0

        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                logits = model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                opt.step()
                total_samples += data.size(0)
                total_batches += 1

        self.model = model.cpu()
        logger.info(f"Local training completed for {epochs} epochs ({total_samples} samples, {total_batches} batches)")
        return total_samples, total_batches

    def evaluate(self) -> float:
        """Evaluate model on test set."""
        if not self.model or not self.test_loader:
            return 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model).to(device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = correct / total if total else 0.0
        return accuracy

    def evaluate_on_train(self) -> float:
        """Evaluate model accuracy on training set."""
        if not self.model or not self.train_loader:
            return 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model).to(device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return correct / total if total else 0.0

    def evaluate_on_val(self) -> float:
        """Evaluate model accuracy on validation set."""
        if not self.model or not self.val_loader:
            return 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model).to(device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return correct / total if total else 0.0


    def elect_aggregator(self, round_idx: int) -> str:
        """Elect aggregator within clique using round-robin."""
        # Use clique members if available, otherwise fall back to all participants
        if self.clique_members:
            aggregator_id = elect_clique_aggregator(self.clique_members, round_idx)
            logger.info(
                f"Elected clique aggregator for round {round_idx + 1}: {aggregator_id} "
                f"(clique {self.clique_id})"
            )
        else:
            sorted_participants = sorted(self.participant_map.keys())
            aggregator_id = sorted_participants[round_idx % len(sorted_participants)]
            logger.info(f"Elected global aggregator for round {round_idx + 1}: {aggregator_id}")
        return aggregator_id

    def _handle_completed_scope_round(self, scope_name: str, scope_round: int, source_round: int) -> None:
        logger.debug(
            "Completed %s round %d (scheduled after %s round %d)",
            scope_name.upper(),
            scope_round,
            self.scope_name,
            source_round,
        )
        if scope_name != self.scope_name:
            config = self._get_scope_config_entry(scope_name)
            wait_seconds = float(getattr(config, "wait_seconds", 0.0) or 0.0) if config else 0.0
            if wait_seconds > 0:
                scope_key = str(scope_name or "").lower()
                queue = getattr(self, "_pending_scope_waits", None)
                ready = getattr(self, "_ready_scope_fetches", None)
                scope_ready = bool(ready and scope_key in ready)
                has_pending_wait = bool(queue and any(entry[0] == scope_key for entry in queue))
                if has_pending_wait and not scope_ready:
                    logger.debug(
                        "%s round %d finished; waiting %.0fs before pulling latest %s model",
                        scope_name.upper(),
                        scope_round,
                        wait_seconds,
                        scope_name.upper(),
                    )
                else:
                    logger.debug(
                        "%s round %d finished; wait window already satisfied; pulling latest model without delay",
                        scope_name.upper(),
                        scope_round,
                    )
            else:
                logger.debug(
                    "%s round %d finished; no wait window configured before pulling latest model",
                    scope_name.upper(),
                    scope_round,
                )

    def _process_high_level_rounds(self) -> bool:
        """Drain scheduled rounds, enforce wait windows, and pull latest scope models."""
        work_done = False
        if self._drain_scope_rounds():
            work_done = True
        if self._pause_for_scope_waits():
            work_done = True
        if self._apply_ready_scope_models():
            work_done = True
        return work_done

    def _drain_scope_rounds(self) -> bool:
        """Execute at most one pending scope round (state, nation, etc.) per invocation."""
        execution_order = getattr(self, "_scope_execution_order", [self.scope_name])
        for scope in execution_order:
            handler_name = None if scope == self.scope_name else scope
            result = self._run_next_scope_round(handler_name)
            if result is None:
                continue
            scope_name, scope_round, source_round = result
            if scope_name != self.scope_name:
                self._handle_completed_scope_round(scope_name, scope_round, source_round)
            return True
        return False

    def _build_scope_handlers(self) -> None:
        """Register scope round handlers for every configured hierarchy level."""
        self._scope_round_handlers = {}
        scope_order = self._scope_execution_order or []
        for idx, scope_name in enumerate(scope_order):
            config = self.scope_configs[scope_name]
            runtime = self._ensure_scope_runtime(scope_name, config)
            trigger_label = "cluster" if idx == 0 else scope_order[idx - 1]
            handler = ScopeRoundHandler(
                scope_name=scope_name,
                config=config,
                trigger_label=trigger_label,
                round_queue=runtime.round_queue,
                rounds_logged=runtime.rounds_logged,
                round_cache=runtime.round_cache,
                round_hashes=runtime.round_hashes,
                committed_rounds=runtime.committed_rounds,
                is_candidate_fn=lambda rt=runtime: rt.is_candidate,
                dispatch_fn=lambda scope_round, parent_round, rt=runtime: self._dispatch_scope_artifacts(scope_round, parent_round, rt),
                execute_fn=lambda scope_round, parent_round, rt=runtime: self._execute_scope_round(scope_round, parent_round, rt),
                budget_fn=lambda rt=runtime: self._scope_round_budget_for(rt),
            )
            self._register_scope_round_handler(handler)

    def _scope_round_budget_for(self, runtime: ScopeRuntime) -> Optional[int]:
        """Return a budget only for the primary scope runtime."""
        if runtime.scope_name.lower() != self.scope_name.lower():
            return None
        return self._scope_round_budget()

    def _aggregator_listen_port(self) -> int:
        """Local gRPC port used when this node is the clique aggregator."""
        return int(self.port) + self.aggregator_port_offset

    def _bridge_listen_port(self) -> int:
        """Local gRPC port used for bridge gossip."""
        return int(self.port) + self.bridge_port_offset

    def _address_with_offset(self, address: Optional[str], offset: int, label: str) -> Optional[str]:
        """Apply a port offset to an address of the form host:port."""
        if not address:
            return None
        try:
            host, port_str = address.rsplit(":", 1)
        except ValueError:
            logger.error("Invalid %s address '%s'", label, address)
            return None
        try:
            base_port = int(port_str)
        except ValueError:
            logger.error("Invalid %s port in address '%s'", label, address)
            return None
        return f"{host}:{base_port + offset}"

    def _aggregator_rpc_address(self) -> Optional[str]:
        """Remote address that points to the elected aggregator's gRPC endpoint."""
        return self._address_with_offset(
            self.aggregator_address,
            self.aggregator_port_offset,
            "aggregator",
        )

    def start_aggregator_server(self) -> None:
        """Start aggregator gRPC server if this node is elected."""
        if self.aggregator_server is not None:
            logger.warning("Aggregator server already running")
            return

        # Use clique members if available, otherwise all participants
        participant_ids = self.clique_members if self.clique_members else list(self.participant_map.keys())
        signing_public_keys: Dict[str, bytes] = {}
        if self.participants:
            signing_public_keys = {p.node_id: bytes(p.signing_public_key) for p in self.participants}
        convergence_handler = None
        agg_port = self._aggregator_listen_port()
        self._release_port_guard("_aggregator_port_guard")
        try:
            self.aggregator_server, self.aggregator_servicer = serve_aggregator(
                self.node_id,
                agg_port,
                self.threshold,
                participant_ids,
                signing_public_keys=signing_public_keys or None,
                ecm_buffer=self.ecm_buffer if self.inter_cluster_enabled else None,
                convergence_signal_handler=convergence_handler,
            )
        except PortBindingError:
            logger.error(
                "Failed to bind aggregator server for %s on port %d; another process may be using it",
                self.node_id,
                agg_port,
            )
            self._ensure_port_guard("_aggregator_port_guard", agg_port, "aggregator")
            raise
        except Exception:
            self._ensure_port_guard("_aggregator_port_guard", agg_port, "aggregator")
            raise
        logger.info(f"Started aggregator server on port {agg_port} for {len(participant_ids)} clique members")

    def stop_aggregator_server(self) -> None:
        """Stop aggregator server."""
        try:
            if self.aggregator_server:
                stop_future = self.aggregator_server.stop(0)
                if stop_future:
                    stop_future.wait()
                self.aggregator_server = None
                self.aggregator_servicer = None
                logger.info("Stopped aggregator server")
        finally:
            self._ensure_port_guard("_aggregator_port_guard", self._aggregator_listen_port(), "aggregator")

    def setup_inter_cluster(self) -> None:
        """Setup inter-cluster aggregation components."""
        if not self.inter_cluster_enabled:
            logger.info(
                "Inter-cluster aggregation disabled (inter_cluster.enabled=false); "
                "bridge forwarding and ECM gossip phases will be skipped"
            )
            return

        ipfs_config = self.inter_cluster_config.get("ipfs", {})
        blockchain_config = self.inter_cluster_config.get("blockchain", {})

        use_mock = self.inter_cluster_config.get("use_mock", True)

        if use_mock:
            ipfs_path = ipfs_config.get("storage_path", "/app/data/ipfs")
            blockchain_path = blockchain_config.get("storage_path", "/app/data/blockchain.json")
            self.ipfs = MockIPFS(storage_path=ipfs_path)
            self.blockchain = MockBlockchain(storage_path=blockchain_path)
            logger.info(f"Using mock storage: ipfs={ipfs_path}, blockchain={blockchain_path}")
        else:
            ipfs_url = ipfs_config.get("api_url", "http://ipfs-node-1:5001")
            ipfs_timeout = ipfs_config.get("timeout", 30.0)
            ipfs_max_retries = ipfs_config.get("max_retries", 5)
            ipfs_retry_delay = ipfs_config.get("retry_delay", 2.0)
            replica_urls = ipfs_config.get("replica_api_urls", [])
            gateway_url = blockchain_config.get(
                "gateway_url",
                os.environ.get("BLOCKCHAIN_GATEWAY_URL", "http://localhost:9000"),
            )
            identity = blockchain_config.get("identity", self.node_id)
            private_key_path = blockchain_config.get(
                "private_key_path",
                f"config/keys/{identity}_sk.pem",
            )
            state_path = blockchain_config.get(
                "state_path",
                f"data/blockchain/{identity}.json",
            )
            jwt_role = blockchain_config.get("jwt_role", "trainer")
            jwt_state = blockchain_config.get("jwt_state", "system")
            jwt_ttl = blockchain_config.get("jwt_ttl_seconds", 24 * 3600)
            self.ipfs = KuboIPFS(
                api_url=ipfs_url,
                timeout=ipfs_timeout,
                max_retries=ipfs_max_retries,
                retry_delay=ipfs_retry_delay,
                replica_api_urls=replica_urls,
            )
            self.blockchain = GatewayBlockchain(
                base_url=gateway_url,
                identity=identity,
                private_key_path=private_key_path,
                state_path=state_path,
                jwt_role=jwt_role,
                jwt_state=jwt_state,
                jwt_ttl_seconds=jwt_ttl,
            )
            logger.info(f"Using real storage: ipfs={ipfs_url}, gateway={gateway_url}")

        if self.blockchain:
            self._hydrate_anchor_bootstrap()

        merge_config = MergeConfig(
            window_size=self.inter_cluster_config.get("window_size", 10),
            alpha=self.inter_cluster_config.get("alpha", 0.5),
            base_gamma=self.inter_cluster_config.get("base_gamma", 0.2),
            max_neighbors=self.inter_cluster_config.get("max_neighbors"),
        )

        self.ecm_buffer = ECMBuffer(
            freshness_window=self.inter_cluster_config.get("freshness_window", 300.0)
        )

        self.inter_cluster_aggregator = InterClusterAggregator(
            cluster_id=f"cluster_{self.clique_id}",
            ipfs=self.ipfs,
            blockchain=self.blockchain,
            merge_config=merge_config,
        )
        self._refresh_central_metadata()
        self._ensure_bridge_stack()
        # Reconfigure hierarchy runtimes now that storage backends exist.
        self._configure_scope_layer()

    def _ensure_bridge_stack(self, allow_state_layer: bool = False) -> None:
        """Ensure the bridge server/client are available for convergence gossip."""
        state_layer_requested = allow_state_layer and self._scope_layer_enabled()
        if not self.inter_cluster_enabled and not state_layer_requested:
            return
        if self.ecm_buffer is None:
            freshness = float(self.inter_cluster_config.get("freshness_window", 300.0))
            self.ecm_buffer = ECMBuffer(freshness_window=freshness)
        if self.bridge_server is None:
            bridge_port = self._bridge_listen_port()
            try:
                self._release_port_guard("_bridge_port_guard")
                self.bridge_server = serve_bridge(
                    self.node_id,
                    bridge_port,
                    self.ecm_buffer,
                    ecm_hooks=self._bridge_ecm_hooks or None,
                )
                logger.info(
                    "Bridge server started on port %d (state_layer=%s)",
                    bridge_port,
                    state_layer_requested,
                )
            except Exception as exc:  # noqa: BLE001
                self.bridge_server = None
                logger.error("Failed to start bridge server on port %d: %s", bridge_port, exc, exc_info=True)
                self._ensure_port_guard("_bridge_port_guard", bridge_port, "bridge")
        if self.bridge_client is None and (self.inter_cluster_enabled or state_layer_requested):
            self.bridge_client = BridgeClient(self.node_id)

    def _update_bridge_hooks(self) -> None:
        """Rebuild the list of ECM hooks and restart the bridge server if needed."""
        hooks: List[Callable[[ECM], None]] = [self._route_scope_ecm]
        if hooks == self._bridge_ecm_hooks:
            return
        self._bridge_ecm_hooks = hooks
        if self.bridge_server:
            self.stop_bridge_server()
            self.bridge_server = None

    def setup_bridge_node(self, inter_edges: List[Tuple[str, str]]) -> bool:
        """Setup bridge node if this node has inter-clique connections."""
        self.inter_edges = inter_edges
        self.is_bridge_node = is_bridge_node(self.node_id, inter_edges)

        if not self.is_bridge_node:
            logger.info(f"Node {self.node_id} is not a bridge node")
            return False

        neighbors = get_inter_clique_neighbors(self.node_id, inter_edges)
        self.neighbor_address_map = {}
        for neighbor in neighbors:
            base_address = self.participant_map.get(neighbor)
            attempts = 0
            while base_address is None and attempts < 5:
                logger.info(
                    f"Neighbor {neighbor} not registered yet; refreshing participant map"
                )
                time.sleep(2)
                self.register_with_ttp()
                base_address = self.participant_map.get(neighbor)
                attempts += 1
            if not base_address:
                logger.warning(
                    f"Could not resolve bridge address for neighbor {neighbor}; ECM gossip disabled for this edge"
                )
                continue
            neighbor_bridge_addr = self._address_with_offset(
                base_address,
                self.bridge_port_offset,
                f"neighbor {neighbor}",
            )
            if not neighbor_bridge_addr:
                continue
            self.neighbor_address_map[neighbor] = neighbor_bridge_addr
        self.neighbor_bridge_addresses = list(self.neighbor_address_map.values())

        logger.info(
            f"Node {self.node_id} is a bridge node with {len(self.neighbor_address_map)} "
            f"resolved inter-clique neighbors: {self.neighbor_address_map}"
        )

        if not self.inter_cluster_enabled:
            logger.warning(
                "Inter-cluster disabled while configuring bridge node %s; skipping bridge server",
                self.node_id,
            )
            return False
        self._ensure_bridge_stack()
        self._update_central_neighbor_addresses()
        return self.bridge_client is not None

    def _route_scope_ecm(self, ecm: ECM) -> None:
        """Route incoming ECMs to the correct per-scope buffer."""
        source = ecm.source_cluster or ""
        runtime: Optional[ScopeRuntime] = None
        if "::" in source:
            scope_label, _ = source.split("::", 1)
            try:
                runtime = self._runtime_for_scope(scope_label)
            except KeyError:
                runtime = None
        else:
            runtime = self.scope_runtime
        if runtime and runtime.ecm_buffer is not None:
            runtime.ecm_buffer.add(ecm)

    def stop_bridge_server(self) -> None:
        """Stop bridge server."""
        stopped = False
        try:
            if self.bridge_server:
                stop_future = self.bridge_server.stop(0)
                if stop_future:
                    stop_future.wait()
                self.bridge_server = None
                stopped = True
            if self.bridge_client:
                self.bridge_client.close()
                self.bridge_client = None
                stopped = True
        finally:
            self._ensure_port_guard("_bridge_port_guard", self._bridge_listen_port(), "bridge")
        if stopped:
            logger.info("Bridge server stopped")

    def _init_bridge_with_retries(
        self,
        inter_edges: List[Tuple[str, str]],
        max_attempts: int = 5,
        delay: float = 2.0,
        fatal: bool = False,
    ) -> bool:
        """Attempt to initialize bridge stack with retries."""
        if not inter_edges or not is_bridge_node(self.node_id, inter_edges):
            return False
        for attempt in range(max_attempts):
            try:
                if self.setup_bridge_node(inter_edges):
                    return True
            except OSError as exc:  # port bind/IO issues
                logger.error(
                    "Bridge initialization attempt %d/%d failed for %s due to OS error: %s",
                    attempt + 1,
                    max_attempts,
                    self.node_id,
                    exc,
                )
                # Port may still be bound; wait before retry.
                time.sleep(delay)
                continue
            logger.warning(
                "Bridge initialization attempt %d/%d failed for %s; retrying in %.1fs",
                attempt + 1,
                max_attempts,
                self.node_id,
                delay,
            )
            time.sleep(delay)
        if fatal and self.is_bridge_node:
            raise RuntimeError(
                f"Bridge node {self.node_id} could not initialize bridge client "
                f"after {max_attempts} attempts"
            )
        logger.warning(
            "Bridge initialization failed for %s after %d attempts; continuing without bridge server",
            self.node_id,
            max_attempts,
        )
        return False

    def _refresh_central_metadata(self, *, skip_if_cached: bool = False) -> None:
        """Fetch central metadata from blockchain and update coordinator."""
        if skip_if_cached and self.central_metadata is not None:
            return
        if not self.blockchain:
            return
        metadata = fetch_central_metadata(self.blockchain)
        if metadata:
            self.central_metadata = metadata
            if metadata.central_nodes:
                logger.info(
                    "Fetched central metadata: central clique=%s, central nodes=%s",
                    metadata.central_clique_idx,
                    metadata.central_nodes,
                )
            else:
                logger.info("Fetched central metadata: no dedicated central clique")
            if metadata.scope_central_nodes:
                for scope, nodes in sorted(metadata.scope_central_nodes.items()):
                    if nodes:
                        logger.info("Central clique for %s: %s", scope, nodes)
            self._update_central_neighbor_addresses()

    def _update_central_neighbor_addresses(self) -> None:
        """Build mapping to central neighbor bridge addresses when metadata is available."""
        self.central_neighbor_addresses = {}
        if not self.central_metadata or not self.participant_map:
            return
        scope_label = self._scope_label_upper()
        candidate_nodes = self._preferred_scope_candidates()
        for node_id in candidate_nodes:
            base_address = self.participant_map.get(node_id)
            attempts = 0
            while not base_address and attempts < 5:
                logger.info(
                    "%s aggregator candidate %s not registered in map yet; refreshing participant map",
                    scope_label,
                    node_id or "unknown",
                )
                time.sleep(1)
                self.register_with_ttp()
                base_address = self.participant_map.get(node_id)
                attempts += 1
            if not base_address:
                logger.warning(
                    "Could not resolve address for %s aggregator candidate %s; excluding from state routing",
                    scope_label,
                    node_id or "unknown",
                )
                continue
            candidate_addr = self._address_with_offset(
                base_address,
                self.bridge_port_offset,
                f"{scope_label} aggregator candidate {node_id or 'unknown'}",
            )
            if not candidate_addr:
                continue
            self.central_neighbor_addresses[node_id] = candidate_addr

        if self.central_neighbor_addresses and not self._logged_central_addresses:
            details = ", ".join(f"{node}@{addr}" for node, addr in self.central_neighbor_addresses.items())
            logger.info(f"{scope_label} aggregator candidate addresses: {details}")
            self._logged_central_addresses = True

        if self.scope_config.enabled:
            self._configure_scope_layer()

    def _log_scope_aggregator_candidates(self) -> None:
        """Log all configured hierarchy-level aggregator candidate rosters."""
        scope_configs = getattr(self, "scope_configs", None) or {}
        if not scope_configs:
            return
        ordered = sorted(scope_configs.items(), key=lambda item: (item[1].scope_index, item[0]))
        for scope_key, config in ordered:
            if not getattr(config, "enabled", True):
                continue
            scope_name = getattr(config, "scope_name", scope_key) or scope_key
            scope_label = scope_name.upper()
            scope_id = self._node_scope_identifier_for(scope_name, config) or getattr(config, "scope_id", None)
            roster = self._scope_member_roster(scope_name, scope_id) if scope_id else []
            candidate_text = ", ".join(roster) if roster else "(dynamic)"
            logger.info(
                "%s aggregator candidates for %s: %s",
                scope_label,
                scope_id or "(dynamic)",
                candidate_text,
            )

    def gossip_ecm(self, cid: str, model_hash: str, round_num: int) -> None:
        """Gossip ECM to neighbor cluster bridge nodes."""
        if not self.is_bridge_node:
            return
        if not self._ensure_bridge_client():
            logger.warning("Cannot gossip ECM: bridge client unavailable")
            return

        cluster_id = f"cluster_{self.clique_id}"
        accepted = self.bridge_client.broadcast_ecm(
            self.neighbor_bridge_addresses,
            cluster_id,
            round_num,
            cid,
            model_hash,
        )
        logger.info(
            f"Gossiped ECM to {accepted}/{len(self.neighbor_bridge_addresses)} neighbors"
        )

    def forward_ecms_to_aggregator(self) -> int:
        """Forward buffered ECMs from this bridge node to the current aggregator.

        Bridge nodes receive ECMs from neighbor clusters. If the bridge node is not
        the aggregator, it must forward these ECMs to the aggregator so that the
        aggregator can merge neighbor models with the intra-cluster model.

        Returns:
            Number of ECMs forwarded, or 0 if not applicable.
        """
        if not self.is_bridge_node:
            return 0
        if not self.ecm_buffer:
            return 0
        if not self._ensure_bridge_client():
            logger.warning("Bridge client unavailable; skipping ECM forward")
            return 0

        fresh_ecms = self.ecm_buffer.get_fresh_ecms()
        fresh_ecms = [ecm for ecm in fresh_ecms if not ecm.is_signal]
        if not fresh_ecms:
            logger.info(
                "No fresh ECMs to forward to aggregator (buffer_size=%d)",
                len(self.ecm_buffer),
            )
            return 0

        if self.is_aggregator:
            # Aggregator already holds these ECMs locally via its bridge server.
            logger.debug(
                "Aggregator is also a bridge node; %d ECMs already staged locally",
                len(fresh_ecms),
            )
            return len(fresh_ecms)

        agg_addr = self._aggregator_rpc_address()
        if not agg_addr:
            logger.warning("Cannot forward ECMs: aggregator RPC address unresolved")
            return 0

        try:
            channel = self._create_aggregator_channel(agg_addr)
            stub = secureagg_pb2_grpc.AggregatorServiceStub(channel)

            ecm_messages = [
                secureagg_pb2.ECMMessage(
                    cid=ecm.cid,
                    hash=ecm.hash,
                    source_cluster=ecm.source_cluster or "",
                )
                for ecm in fresh_ecms
            ]

            request = secureagg_pb2.ECMSubmitRequest(
                node_id=self.node_id,
                ecms=ecm_messages,
            )

            response = stub.SubmitECMs(request, timeout=10)
            if response.accepted:
                logger.info(
                    f"Forwarded {len(fresh_ecms)} ECMs to aggregator {self.aggregator_id}"
                )
                if self.ecm_buffer:
                    removed = self.ecm_buffer.remove_cids([ecm.cid for ecm in fresh_ecms])
                    logger.debug("Removed %d ECMs from buffer after forwarding", removed)
                return len(fresh_ecms)
            else:
                logger.warning(f"Aggregator rejected ECMs: {response.message}")
                return 0
        except grpc.RpcError as e:
            logger.warning(f"Failed to forward ECMs to aggregator: {e}")
            return 0

    def _wait_for_neighbor_ecms(self) -> None:
        """Poll for incoming ECMs before forwarding to the aggregator."""
        if (
            not self.ecm_buffer
            or self.ecm_forward_wait <= 0
            or not self.neighbor_bridge_addresses
        ):
            return
        self._ensure_bridge_client()

        deadline = time.time() + self.ecm_forward_wait
        poll_interval = 1.0
        while time.time() < deadline:
            fresh = [ecm for ecm in self.ecm_buffer.get_fresh_ecms() if not ecm.is_signal]
            if fresh:
                logger.debug(
                    "ECM buffer received %d entries from neighbors; proceeding to forward",
                    len(fresh),
                )
                return
            logger.debug(
                "Waiting for neighbor ECMs (%d neighbors)...",
                len(self.neighbor_bridge_addresses),
            )
            self._abort_if_global_stop()
            time.sleep(poll_interval)
            self._abort_if_global_stop()
        logger.info(
            "No ECMs received from %d neighbors after waiting %ds; continuing without them",
            len(self.neighbor_bridge_addresses),
            int(self.ecm_forward_wait),
        )

    def _ensure_bridge_client(self, allow_state_layer: bool = False) -> bool:
        """
        Ensure bridge infrastructure is running; restart if necessary.

        Args:
            allow_state_layer: When True, permit a lightweight client to be created
                for state-level dispatch even if this node is not configured as a bridge.
        """
        if self.bridge_client:
            return True
        state_layer_requested = allow_state_layer and self._scope_layer_enabled()
        if self.is_bridge_node and self.inter_cluster_enabled:
            if not self.inter_edges:
                logger.warning(
                    "Bridge client unavailable for %s and no inter_edges configured",
                    self.node_id,
                )
                return False
            logger.warning(
                "Bridge client missing for %s; restarting bridge server on port %d",
                self.node_id,
                self._bridge_listen_port(),
            )
            # Tear down any existing server before reconfiguring.
            self.stop_bridge_server()
            return self._init_bridge_with_retries(
                self.inter_edges,
                max_attempts=3,
                delay=1.0,
                fatal=False,
            )
        if state_layer_requested:
            logger.debug(
                "State layer requested bridge client for %s; creating lightweight client",
                self.node_id,
            )
            self.bridge_client = BridgeClient(self.node_id)
            return True
        return False

    def _create_aggregator_channel(self, address: str) -> grpc.Channel:
        """Create an aggregator gRPC channel with increased message limits."""
        return grpc.insecure_channel(address, options=grpc_message_options())

    def _invoke_tracked_rpc(
        self,
        rpc_name: str,
        rpc_fn: Callable[..., Any],
        request: Any,
        timeout: int = 30,
    ) -> Any:
        """Invoke an RPC and record request/response sizes for comm metrics."""
        start = time.monotonic()
        response = rpc_fn(request, timeout=timeout)
        if self.comm_tracker is not None:
            track_rpc_call(
                request=request,
                response=response,
                method_name=rpc_name,
                latency_ms=(time.monotonic() - start) * 1000.0,
                tracker=self.comm_tracker,
            )
        return response

    def run_secure_aggregation_round(self) -> List[float]:
        """Run one round of secure aggregation protocol."""
        self._abort_if_global_stop()
        logger.info(f"Starting secure aggregation round {self.current_round}")
        # Client-side secure aggregation state
        client = SecureAggregationNode(
            self.node_id,
            signing_private=self.signing_keypair.private_key if self.signing_keypair else None,
            signing_public=self.signing_keypair.public_key if self.signing_keypair else None,
        )

        # Get aggregator address
        agg_addr = self._aggregator_rpc_address()
        if not agg_addr:
            raise AggregatorUnavailable("Aggregator address unavailable")

        channel = self._create_aggregator_channel(agg_addr)
        stub = secureagg_pb2_grpc.AggregatorServiceStub(channel)

        # SAP Round 0: Advertise keys
        logger.info("SAP-Round 0: Advertising keys")
        sap_r0_start = time.monotonic()
        self.comm_tracker.set_phase("sap_round0")
        advert_msg = client.advertise_keys()

        # Retry logic for initial aggregator connection.
        retry_delay = 1
        max_retries = 30
        if self.inter_cluster_enabled:
            retry_delay = 2
            max_retries = 75
        response = None
        last_rpc_error: Optional[grpc.RpcError] = None
        for attempt in range(max_retries):
            self._abort_if_global_stop()
            try:
                round0_request = secureagg_pb2.KeyAdvertisement(
                    node_id=self.node_id,
                    c_public_key=advert_msg.c_public,
                    s_public_key=advert_msg.s_public,
                    signature=advert_msg.signature,
                )
                response = self._invoke_tracked_rpc(
                    "Round0AdvertiseKeys",
                    stub.Round0AdvertiseKeys,
                    round0_request,
                    timeout=30,
                )
                break
            except grpc.RpcError as e:
                last_rpc_error = e
                if attempt < max_retries - 1:
                    if attempt < 5 or (attempt + 1) % 5 == 0:
                        logger.warning(
                            "Aggregator %s at %s connection attempt %d/%d failed (%s); retrying in %ds...",
                            self.aggregator_id,
                            agg_addr,
                            attempt + 1,
                            max_retries,
                            e.code().name if hasattr(e, "code") else "unknown",
                            retry_delay,
                        )
                    attempt_idx = attempt + 1
                    self._maybe_check_convergence_during_retry(
                        attempt_idx, max_retries, "Round0AdvertiseKeys"
                    )
                    time.sleep(retry_delay)
                    self._abort_if_global_stop()
                else:
                    break

        if response is None:
            self._raise_aggregator_unavailable(
                "Round0AdvertiseKeys", max_retries, last_rpc_error
            )

        if not response.accepted:
            raise RuntimeError(f"Round 0 failed: {response.message}")

        # Wait for ALL clique members to advertise (not just threshold).
        expected_participants = len(self.clique_members)
        while len(response.all_keys) < expected_participants:
            self._abort_if_global_stop()
            time.sleep(1)
            self._abort_if_global_stop()
            round0_request = secureagg_pb2.KeyAdvertisement(
                node_id=self.node_id,
                c_public_key=advert_msg.c_public,
                s_public_key=advert_msg.s_public,
                signature=advert_msg.signature,
            )
            response = self._invoke_tracked_rpc(
                "Round0AdvertiseKeys",
                stub.Round0AdvertiseKeys,
                round0_request,
                timeout=30,
            )

            logger.info(f"SAP-Round 0 complete: received {len(response.all_keys)} participants")

        # Record Round 0 timing
        if self.prom_metrics:
            self.prom_metrics.observe_sap_phase("round0", time.monotonic() - sap_r0_start)

        # Pass received advertisements to client
        ordered_participants = [p.node_id for p in response.all_keys]
        adverts = [
            AdvertiseMessage(
                node_id=p.node_id,
                c_public=bytes(p.c_public_key),
                s_public=bytes(p.s_public_key),
                signature=bytes(p.signature),
                signing_public=None,
            )
            for p in response.all_keys
        ]
        client.receive_advertisements(adverts)

        # SAP Round 1: Share keys (simplified - just send empty shares)
        logger.info("SAP-Round 1: Sharing keys")
        sap_r1_start = time.monotonic()
        self.comm_tracker.set_phase("sap_round1")
        ct_list = client.create_round1_ciphertexts(ordered_participants, self.threshold)
        round1_request = secureagg_pb2.ShareKeysMessage(
            node_id=self.node_id,
            ciphertexts=[
                secureagg_pb2.Round1Ciphertext(
                    sender_id=ct.sender_id,
                    recipient_id=ct.recipient_id,
                    iv=ct.iv,
                    ciphertext=ct.ciphertext,
                    tag=ct.tag,
                )
                for ct in ct_list
            ],
        )
        response1 = self._invoke_tracked_rpc(
            "Round1ShareKeys",
            stub.Round1ShareKeys,
            round1_request,
            timeout=30,
        )
        mailbox = [
            Round1Ciphertext(
                sender_id=ct.sender_id,
                recipient_id=ct.recipient_id,
                iv=bytes(ct.iv),
                ciphertext=bytes(ct.ciphertext),
                tag=bytes(ct.tag),
            )
            for ct in response1.mailbox
        ]
        # Poll until mailbox has entries from all n participants (each sends to all including self).
        expected_mail = len(ordered_participants)
        while len(mailbox) < expected_mail:
            self._abort_if_global_stop()
            time.sleep(1)
            self._abort_if_global_stop()
            round1_poll_request = secureagg_pb2.ShareKeysMessage(node_id=self.node_id, ciphertexts=[])
            response1 = self._invoke_tracked_rpc(
                "Round1ShareKeys",
                stub.Round1ShareKeys,
                round1_poll_request,
                timeout=30,
            )
            mailbox = [
                Round1Ciphertext(
                    sender_id=ct.sender_id,
                    recipient_id=ct.recipient_id,
                    iv=bytes(ct.iv),
                    ciphertext=bytes(ct.ciphertext),
                    tag=bytes(ct.tag),
                )
                for ct in response1.mailbox
            ]
        client.receive_round1_ciphertexts(mailbox)

        # Record Round 1 timing
        if self.prom_metrics:
            self.prom_metrics.observe_sap_phase("round1", time.monotonic() - sap_r1_start)

        # SAP Round 2: Send masked input
        logger.info("SAP-Round 2: Sending masked model")
        sap_r2_start = time.monotonic()
        self.comm_tracker.set_phase("sap_round2")
        model_vec = flatten_params(self.model)
        quantized = quantize_vector(model_vec, self.scale)

        masked = client.create_masked_input(quantized)
        masked_bytes = [_int_to_bytes(val, SHARE_BYTES) for val in masked.masked_vector]
        round2_request = secureagg_pb2.MaskedInputMessage(node_id=self.node_id, masked_vector=masked_bytes)
        response2 = self._invoke_tracked_rpc(
            "Round2MaskedInput",
            stub.Round2MaskedInput,
            round2_request,
            timeout=30,
        )

        # Wait for survivors list
        while not response2.survivors:
            self._abort_if_global_stop()
            time.sleep(1)
            self._abort_if_global_stop()
            round2_poll_request = secureagg_pb2.MaskedInputMessage(
                node_id=self.node_id,
                masked_vector=masked_bytes,
            )
            response2 = self._invoke_tracked_rpc(
                "Round2MaskedInput",
                stub.Round2MaskedInput,
                round2_poll_request,
                timeout=30,
            )

        logger.info(f"SAP-Round 2 complete: {len(response2.survivors)} survivors")

        # Record Round 2 timing
        if self.prom_metrics:
            self.prom_metrics.observe_sap_phase("round2", time.monotonic() - sap_r2_start)

        # SAP Round 3: Consistency check
        logger.info("SAP-Round 3: Consistency check")
        sap_r3_start = time.monotonic()
        self.comm_tracker.set_phase("sap_round3")
        survivor_sig = client.sign_survivor_list(response2.survivors)
        round3_request = secureagg_pb2.ConsistencySignature(
            node_id=self.node_id,
            signature=survivor_sig.signature,
        )
        response3 = self._invoke_tracked_rpc(
            "Round3ConsistencyCheck",
            stub.Round3ConsistencyCheck,
            round3_request,
            timeout=30,
        )

        # Record Round 3 timing
        if self.prom_metrics:
            self.prom_metrics.observe_sap_phase("round3", time.monotonic() - sap_r3_start)

        # SAP Round 4: Unmask (simplified - send empty shares)
        logger.info("SAP-Round 4: Unmasking")
        sap_r4_start = time.monotonic()
        self.comm_tracker.set_phase("sap_round4")
        dropouts = set(ordered_participants) - set(response2.survivors)
        unmask_payload = client.prepare_unmasking_payload(dropouts, response2.survivors)
        round4_request = secureagg_pb2.UnmaskShares(
            node_id=self.node_id,
            dropout_s_shares={k: _encode_share(x, s) for k, (x, s) in unmask_payload.s_shares_for_dropouts.items()},
            survivor_b_shares={k: _encode_share(x, b) for k, (x, b) in unmask_payload.b_shares_for_survivors.items()},
        )
        response4 = self._invoke_tracked_rpc(
            "Round4Unmask",
            stub.Round4Unmask,
            round4_request,
            timeout=30,
        )

        # Wait for aggregation to complete
        while not response4.aggregation_complete:
            self._abort_if_global_stop()
            time.sleep(1)
            self._abort_if_global_stop()
            round4_poll_request = secureagg_pb2.UnmaskShares(
                node_id=self.node_id,
                dropout_s_shares={k: _encode_share(x, s) for k, (x, s) in unmask_payload.s_shares_for_dropouts.items()},
                survivor_b_shares={k: _encode_share(x, b) for k, (x, b) in unmask_payload.b_shares_for_survivors.items()},
            )
            response4 = self._invoke_tracked_rpc(
                "Round4Unmask",
                stub.Round4Unmask,
                round4_poll_request,
                timeout=30,
            )

        logger.info("SAP-Round 4 complete: aggregation done")

        # Record Round 4 timing
        if self.prom_metrics:
            self.prom_metrics.observe_sap_phase("round4", time.monotonic() - sap_r4_start)

        # Get global model
        aggregated: List[float] = []
        if self.is_aggregator:
            logger.info("Fetching global model")
            self.comm_tracker.set_phase("model_fetch")
            model_request = secureagg_pb2.ModelRequest(round=self.current_round)
            model_response = self._invoke_tracked_rpc(
                "GetGlobalModel",
                stub.GetGlobalModel,
                model_request,
                timeout=30,
            )
            aggregated = list(model_response.model_weights)
            logger.info(f"Received aggregated model ({len(aggregated)} parameters)")

        channel.close()

        return aggregated

    def run_training_loop(self) -> None:
        """Convergence-driven training loop with secure aggregation.

        Training continues until either:
        1. Global convergence is achieved (model delta below tolerance for patience rounds,
           and all neighbor clusters have also converged)
        2. Maximum rounds limit is reached (safety cap)
        """
        local_epochs = self.training_config["local_epochs"]
        max_rounds = self.max_training_rounds
        convergence_warmup = max(0, self.convergence_config.warmup_rounds)

        # Initialize convergence tracker
        if self._is_convergence_runtime_enabled():
            self.convergence_tracker = ConvergenceTracker(
                self.convergence_config, f"cluster_{self.clique_id}"
            )
        else:
            self.convergence_tracker = None
            logger.info(
                "Cluster convergence runtime disabled; training will run fixed %d rounds",
                max_rounds,
            )
        self._refresh_central_metadata(skip_if_cached=True)

        # Initialize Prometheus metrics and start HTTP server for scraping
        self.prom_metrics = PrometheusMetrics.get_instance(self.node_id, self.clique_id)
        metrics_port = self.config.get("metrics_port", 8000)
        self.prom_metrics.start_server(port=metrics_port)
        self.prom_metrics.set_training_samples(len(self.train_indices))
        self.prom_metrics.set_model_parameters(sum(p.numel() for p in self.model.parameters()))
        pending = getattr(self, "_pending_topology_metrics", None)
        if pending:
            self.prom_metrics.set_topology_max_degree(pending["max_degree"])
            self.prom_metrics.set_topology_average_degree(pending["avg_degree"])
            self.prom_metrics.set_node_connections(pending["node_connections"])
        pending_type = getattr(self, "_pending_topology_type", None)
        if pending_type:
            self.prom_metrics.set_topology_type(pending_type)
        self.comm_tracker = CommunicationTracker(self.node_id)

        logger.info(
            f"Starting convergence-driven training (max_rounds={max_rounds}, "
            f"warmup_rounds={convergence_warmup}, "
            f"tol_abs={self.convergence_config.tol_abs}, patience={self.convergence_config.patience})"
        )
        self._apply_ready_scope_models()

        should_stop = False
        stop_reason = ""

        while self.current_round < max_rounds and not should_stop:
            if self._process_high_level_rounds():
                continue
            round_idx = self.current_round
            round_start_time = time.monotonic()
            self.comm_tracker.set_round(round_idx)
            self.prom_metrics.set_round(round_idx)

            logger.info(f"\n{'='*60}")
            logger.info(f"Cluster Round {round_idx + 1}/{max_rounds}")
            logger.info(f"{'='*60}")
            should_stop, stop_reason = self._refresh_convergence_state(should_stop, stop_reason)
            if should_stop:
                logger.info(f"Stopping training before local update: {stop_reason}")
                break

            # Phase 1: Local training
            logger.info("Phase 1: Local training")
            train_start = time.monotonic()
            train_samples, train_batches = self.train_local(local_epochs)
            local_training_time = time.monotonic() - train_start
            self.prom_metrics.observe_local_training(local_training_time)

            # Evaluate on train/val/test sets for metrics
            train_acc = self.evaluate_on_train()
            val_acc = self.evaluate_on_val()

            acc_before = self.evaluate()
            logger.info(f"Accuracy before aggregation: {acc_before:.4f}")

            should_stop, stop_reason = self._refresh_convergence_state(should_stop, stop_reason)
            if should_stop:
                logger.info(f"Stopping training after local update: {stop_reason}")
                break

            # Aggregator election
            self.aggregator_id = self.elect_aggregator(round_idx)
            self.aggregator_address = self.participant_map[self.aggregator_id]
            self.is_aggregator = (self.aggregator_id == self.node_id)
            aggregator_is_bridge = is_bridge_node(self.aggregator_id, self.inter_edges)
            wait_for_aggregator = 5
            if self.inter_cluster_enabled and aggregator_is_bridge:
                wait_for_aggregator = max(wait_for_aggregator, 8)

            cid: Optional[str] = None
            model_hash: Optional[str] = None
            model_data_id: Optional[str] = None
            aggregator_unreachable_this_round = False
            round_failed = False
            try:
                if self.is_aggregator:
                    logger.info(f"*** This node is the AGGREGATOR for round {round_idx + 1} ***")
                    self.start_aggregator_server()

                logger.info(
                    "Waiting for aggregator %s to be ready (sleeping %ds)...",
                    self.aggregator_id,
                    wait_for_aggregator,
                )
                wait_remaining = wait_for_aggregator
                while wait_remaining > 0 and not should_stop:
                    interval = min(1.0, wait_remaining)
                    time.sleep(interval)
                    wait_remaining -= interval
                    should_stop, stop_reason = self._refresh_convergence_state(should_stop, stop_reason)
                    if should_stop:
                        logger.info(f"Stopping training while waiting for aggregator: {stop_reason}")
                        break

                if should_stop:
                    break

                # Phase 2: Secure aggregation
                logger.info("Phase 2: Secure aggregation")
                sap_start = time.monotonic()
                try:
                    aggregated_weights = self.run_secure_aggregation_round()
                except GlobalStopRequested:
                    should_stop = True
                    stop_reason = "global_convergence"
                    round_failed = True
                    logger.info("Aborting secure aggregation due to confirmed global convergence")
                    break
                finally:
                    agg_time = time.monotonic() - sap_start
                    self.prom_metrics.observe_aggregation(agg_time)

                # Phase 3: Bridge nodes forward ECMs (if inter-cluster is enabled)
                if self.is_bridge_node:
                    if self.inter_cluster_enabled:
                        if self.neighbor_bridge_addresses:
                            self._wait_for_neighbor_ecms()
                        logger.info("Phase 3: Forwarding ECMs to aggregator")
                        forwarded = self.forward_ecms_to_aggregator()
                        if forwarded > 0:
                            if self.is_aggregator:
                                logger.info(
                                    "Phase 3: Aggregator staging %d ECMs from neighbor clusters",
                                    forwarded,
                                )
                            else:
                                logger.info(f"Phase 3: Forwarded {forwarded} ECMs to aggregator")
                        else:
                            logger.info("Phase 3: No fresh ECMs to forward this round")
                    else:
                        logger.info(
                            "Phase 3: Skipped ECM forwarding because inter_cluster.enabled is false"
                        )

                final_model_array: Optional[np.ndarray] = None
                final_cid: Optional[str] = None
                final_hash: Optional[str] = None
                final_data_id: Optional[str] = None

                if self.is_aggregator and aggregated_weights:
                    logger.info("Phase 4: Updating model with aggregated weights")
                    dequantized = dequantize_vector([int(w) for w in aggregated_weights], self.scale)
                    load_params(self.model, dequantized)
                    model_array = np.array(dequantized, dtype=np.float32)
                    final_model_array = model_array

                    # Phase 5: Inter-cluster merge (aggregator only)
                    if self.inter_cluster_enabled and self.inter_cluster_aggregator:
                        # Wait briefly for ECMs from bridge nodes to arrive
                        time.sleep(2)
                        logger.info("Phase 5: Inter-cluster merge")
                        intra_model = np.array(dequantized, dtype=np.float32)

                        if self.ecm_buffer:
                            consumed_ecms = self.ecm_buffer.get_fresh_ecms()
                            for ecm in consumed_ecms:
                                self.inter_cluster_aggregator.receive_ecms(self.node_id, [ecm])
                                # Update neighbor convergence status from ECM
                                if (
                                    hasattr(ecm, "cluster_converged")
                                    and self.convergence_tracker is not None
                                    and self._is_convergence_runtime_enabled()
                                ):
                                    self.convergence_tracker.receive_neighbor_convergence(
                                        ecm.source_cluster, ecm.cluster_converged
                                    )
                            if consumed_ecms:
                                removed = self.ecm_buffer.remove_cids([ecm.cid for ecm in consumed_ecms])
                                logger.debug("Aggregator consumed and removed %d ECMs from buffer", removed)

                        merged_data = self.inter_cluster_aggregator.process_round(
                            intra_model, round_idx
                        )
                        merged_model, cid, model_hash = merged_data
                        model_data_id = getattr(self.inter_cluster_aggregator, "last_data_id", None)

                        final_model_array = merged_model
                        final_cid = cid
                        final_hash = model_hash
                        final_data_id = model_data_id

                        load_params(self.model, merged_model.tolist())
                        logger.info(f"Inter-cluster merge complete: cid={cid[:16] if cid else 'N/A'}...")

                    if final_model_array is not None and self.aggregator_servicer:
                        quantized_final = quantize_vector(final_model_array.tolist(), self.scale)
                        self.aggregator_servicer.aggregated_result = [float(v) for v in quantized_final]

                    if final_model_array is not None:
                        conv_state = self._update_convergence_state_from_model(
                            final_model_array,
                            model_cid=final_cid,
                            model_hash=final_hash,
                            model_data_id=final_data_id,
                        )
                        if conv_state:
                            should_stop = conv_state.should_stop
                            stop_reason = conv_state.stop_reason

                    self._last_model_cid = final_cid
                    self._last_model_hash = final_hash
                    self._last_model_data_id = final_data_id

                if not self.is_aggregator:
                    wait_for_model_ref = self.is_bridge_node and self.inter_cluster_enabled
                    model_response = self._await_aggregated_model(
                        wait_for_model_ref=wait_for_model_ref,
                        require_ready=True,
                    )
                    if not model_response or not model_response.model_weights:
                        raise AggregatorUnavailable(
                            f"Aggregator {self.aggregator_id} did not publish merged model"
                        )
                    should_stop = model_response.should_stop
                    stop_reason = model_response.stop_reason
                    self._latest_cluster_converged = model_response.cluster_converged
                    self._latest_delta_norm = model_response.delta_norm
                    self._latest_convergence_streak = getattr(
                        model_response,
                        "convergence_streak",
                        self._latest_convergence_streak,
                    )
                    response_cid = model_response.model_cid or ""
                    response_hash = model_response.model_hash or ""
                    response_data_id = getattr(model_response, "model_data_id", "") or ""
                    self._last_model_cid = response_cid or None
                    self._last_model_hash = response_hash or None
                    self._last_model_data_id = response_data_id or None

                    quantized_final = [int(w) for w in model_response.model_weights]
                    dequantized = dequantize_vector(quantized_final, self.scale)
                    load_params(self.model, dequantized)

                    if self.is_bridge_node and response_cid and response_hash:
                        cid = response_cid
                        model_hash = response_hash

                    cluster_anchor_id = f"cluster_{self.clique_id}"
                    if response_data_id and self.blockchain:
                        self.blockchain.remember_anchor(
                            cluster_id=cluster_anchor_id,
                            round_num=round_idx,
                            data_id=response_data_id,
                            cid=response_cid or None,
                            hash_val=response_hash or None,
                        )

                    # Prime tracker state so future aggregator rounds have accurate baseline.
                    self._prime_convergence_tracker_state()

                acc_after = self.evaluate()
                # Update Prometheus metrics
                self.prom_metrics.set_accuracy(train_acc, val_acc, acc_after)
                self.prom_metrics.set_convergence(
                    self._latest_delta_norm,
                    self.convergence_tracker.state.convergence_streak if self.convergence_tracker else 0,
                    self._latest_cluster_converged,
                )
                self.prom_metrics.set_aggregator_status(self.is_aggregator)
                logger.info(f"Accuracy after aggregation: {acc_after:.4f}")
                logger.info(f"Improvement: {acc_after - acc_before:+.4f}")

            except PortBindingError as exc:
                round_failed = True
                aggregator_unreachable_this_round = True
                logger.error("Aggregator %s could not start gRPC server: %s", self.aggregator_id, exc)
            except AggregatorUnavailable as exc:
                round_failed = True
                aggregator_unreachable_this_round = True
                logger.warning(
                    "%s. Will retry after backoff unless convergence is confirmed.",
                    exc,
                )
            except Exception as e:
                round_failed = True
                logger.error("Secure aggregation failed: %s", e, exc_info=True)

            finally:
                if self.is_aggregator:
                    time.sleep(2)
                    self.stop_aggregator_server()

            if round_failed:
                if aggregator_unreachable_this_round:
                    self._consecutive_aggregator_failures += 1
                    if self._consecutive_aggregator_failures >= self._max_aggregator_failure_rounds:
                        should_stop = True
                        stop_reason = "aggregator_unreachable"
                        logger.error(
                            "Stopping training: failed to reach aggregator %s for %d consecutive rounds",
                            self.aggregator_id,
                            self._consecutive_aggregator_failures,
                        )
                        break
                else:
                    self._consecutive_aggregator_failures = 0
                retry_delay = 5
                logger.warning(
                    "Round %d failed. Retrying after %ds once aggregator %s is reachable.",
                    round_idx + 1,
                    retry_delay,
                    self.aggregator_id,
                )
                time.sleep(retry_delay)
                continue
            else:
                self._consecutive_aggregator_failures = 0

            # Phase 6: ECM gossip with convergence status (bridge nodes only)
            if cid and model_hash and self.is_bridge_node:
                logger.info("Phase 6: ECM gossip to neighbor clusters")
                self.gossip_ecm(cid, model_hash, round_idx)

            should_stop, stop_reason = self._sync_stop_state_from_tracker(should_stop, stop_reason)
            if should_stop:
                logger.info(f"Stopping training: {stop_reason}")
                break

            # Record round total time
            round_total_time = time.monotonic() - round_start_time
            self.prom_metrics.observe_round_total(round_total_time)

            # Get communication stats from tracker and record to Prometheus
            comm_stats = self.comm_tracker.get_round_stats(round_idx)
            self.prom_metrics.add_bytes_sent(comm_stats["bytes_sent"])
            self.prom_metrics.add_bytes_received(comm_stats["bytes_received"])
            self.prom_metrics.add_messages_sent(comm_stats["messages_sent"])
            self.prom_metrics.add_messages_received(comm_stats["messages_received"])

            total_bytes = comm_stats["bytes_sent"] + comm_stats["bytes_received"]
            self.prom_metrics.set_total_bytes_per_round(total_bytes)

            logger.info(f"Training Cluster Round {round_idx + 1} complete.")
            logger.info("Waiting before next cluster round...")

            self._maybe_schedule_scope_round(round_idx)
            self._process_high_level_rounds()

            time.sleep(5)
            self.current_round += 1

        self._log_final_model_status()
        self._process_high_level_rounds()
        logger.info("\n" + "="*60)
        logger.info(f"Training completed after {self.current_round + 1} rounds (reason: {stop_reason or 'max_rounds'})")
        logger.info("="*60)

    def _await_aggregated_model(
        self,
        wait_for_model_ref: bool = False,
        require_ready: bool = False,
    ) -> Optional[secureagg_pb2.ModelResponse]:
        """
        Fetch convergence status from aggregator.

        Args:
            wait_for_model_ref: If True, poll until aggregator publishes IPFS metadata.
                Bridge nodes enable this so they always obtain CID/hash for ECM gossip.
        """
        channel = None
        agg_addr = self._aggregator_rpc_address()
        if not agg_addr:
            raise AggregatorUnavailable("Aggregator address unavailable")
        try:
            channel = self._create_aggregator_channel(agg_addr)
            stub = secureagg_pb2_grpc.AggregatorServiceStub(channel)

            delay = 2
            attempts = 0

            while True:
                attempts += 1
                try:
                    response = self._invoke_tracked_rpc(
                        "GetGlobalModel",
                        stub.GetGlobalModel,
                        secureagg_pb2.ModelRequest(round=self.current_round),
                        timeout=10,
                    )
                except grpc.RpcError as exc:
                    code = exc.code().name if hasattr(exc, "code") else "UNKNOWN"
                    raise AggregatorUnavailable(
                        f"Aggregator {self.aggregator_id} unreachable during GetGlobalModel "
                        f"(attempt {attempts}, error={code})"
                    ) from exc

                has_metadata = (
                    (not wait_for_model_ref)
                    or not self.inter_cluster_enabled
                    or (response.model_cid and response.model_hash)
                )
                ready_flag = bool(getattr(response, "metadata_ready", getattr(response, "convergence_ready", False)))
                weights_ready = bool(response.model_weights)
                if has_metadata and (not require_ready or ready_flag) and weights_ready:
                    return response

                if attempts % 5 == 0:
                    logger.info(
                        "Aggregator %s still finalizing merged model (attempt %d); waiting %ds",
                        self.aggregator_id,
                        attempts,
                        delay,
                    )
                time.sleep(delay)
        except AggregatorUnavailable:
            raise
        except Exception as e:
            raise AggregatorUnavailable(f"Failed to fetch convergence status: {e}") from e
        finally:
            if channel:
                channel.close()

    def _refresh_convergence_state(self, should_stop: bool, stop_reason: str) -> Tuple[bool, str]:
        """Process buffered signals and align stop flags."""
        return self._sync_stop_state_from_tracker(should_stop, stop_reason)

    def _sync_stop_state_from_tracker(self, should_stop: bool, stop_reason: str) -> Tuple[bool, str]:
        """
        Align local loop control flags with the convergence tracker state.

        Returns:
            Tuple updated with the tracker decision if a central/global stop
            signal was received outside of the aggregator update flow.
        """
        if not self._is_convergence_runtime_enabled():
            return should_stop, stop_reason
        if self.convergence_tracker and not should_stop:
            tracker_state = self.convergence_tracker.state
            if tracker_state.should_stop:
                stop_reason = tracker_state.stop_reason or stop_reason or "global_convergence"
                logger.info(
                    "Convergence tracker requested stop (reason=%s, round=%s)",
                    stop_reason,
                    tracker_state.global_stop_round,
                )
                return True, stop_reason
        return should_stop, stop_reason

    def _abort_if_global_stop(self) -> None:
        """Raise if global convergence has already been confirmed."""
        if not self._is_convergence_runtime_enabled():
            return
        should_stop, _ = self._refresh_convergence_state(False, "")
        if should_stop:
            raise GlobalStopRequested()

    def start(self) -> None:
        """Start the node service."""
        logger.info(f"Starting node {self.node_id} on port {self.port}")

        # Register with TTP
        self.register_with_ttp()

        # Setup data and model
        self.setup_data()
        self.setup_model()

        # Setup inter-cluster aggregation
        self.setup_inter_cluster()

        # Load inter_edges from topology file or config
        inter_edges: List[Tuple[str, str]] = []
        topology_file = self.inter_cluster_config.get("topology_file", "/app/config/topology.json")
        if Path(topology_file).exists():
            with open(topology_file) as f:
                topology_data = json.load(f)
            inter_edges = [(e[0], e[1]) for e in topology_data.get("inter_edges", [])]
            logger.info(f"Loaded {len(inter_edges)} inter-edges from {topology_file}")

            cliques = topology_data.get("cliques", [])
            if cliques and inter_edges is not None:
                max_degree = compute_max_degree(cliques, inter_edges)
                avg_degree = compute_average_degree(cliques, inter_edges)
                node_degrees = compute_node_degrees(cliques, inter_edges)
                node_connections = int(node_degrees.get(self.node_id, 0))
                logger.info(f"Topology metrics: max_degree={max_degree}, avg_degree={avg_degree:.2f}")
                # Store for deferred export once prom_metrics is initialized.
                self._pending_topology_metrics = {
                    "max_degree": max_degree,
                    "avg_degree": avg_degree,
                    "node_connections": node_connections,
                }

            topology_type = topology_data.get("topology_type", "d_cliques")
            self._pending_topology_type = topology_type
        else:
            inter_edges_config = self.inter_cluster_config.get("inter_edges", [])
            inter_edges = [(e[0], e[1]) for e in inter_edges_config]
            intra_connections = max(0, len(self.clique_members) - 1)
            inter_connections = sum(1 for a, b in inter_edges if self.node_id in (a, b))
            self._pending_topology_metrics = {
                "max_degree": 0,
                "avg_degree": 0.0,
                "node_connections": intra_connections + inter_connections,
            }

        # Wait for clique members or all nodes to be ready
        if self.clique_members:
            expected_count = len(self.clique_members)
            logger.info(f"Waiting for {expected_count} clique members to register...")
            registered_clique_members = set(self.clique_members) & set(self.participant_map.keys())
            while len(registered_clique_members) < expected_count:
                time.sleep(2)
                self.register_with_ttp()
                registered_clique_members = set(self.clique_members) & set(self.participant_map.keys())
            logger.info(f"All {len(registered_clique_members)} clique members ready. Starting training...")
        else:
            logger.info("Waiting for all nodes to register...")
            while len(self.participants) < self.dataset_config["num_clients"]:
                time.sleep(2)
                self.register_with_ttp()
            logger.info(f"All {len(self.participants)} nodes are ready. Starting training...")

        self._log_scope_aggregator_candidates()

        if inter_edges:
            bridge_ready = self._init_bridge_with_retries(inter_edges, fatal=False)
            if not bridge_ready and self.is_bridge_node:
                logger.warning(
                    "Bridge node %s could not initialize bridge server; continuing without inter-cluster gossip",
                    self.node_id,
                )

        time.sleep(2)

        try:
            # Run training loop
            self.run_training_loop()
        finally:
            # Cleanup bridge server
            self.stop_bridge_server()

        logger.info(f"Node {self.node_id} finished")


def main() -> None:
    """Entry point for node service."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Secure Aggregation Node Service")
    parser.add_argument("--config", required=True, help="Path to node configuration file")
    args = parser.parse_args()

    node = NodeService(args.config)
    node.start()


if __name__ == "__main__":
    main()
