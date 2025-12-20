"""Prometheus metrics for federated learning nodes."""

from __future__ import annotations

import threading
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class PrometheusMetrics:
    """Singleton class for exposing Prometheus metrics from FL nodes."""

    _instance: Optional["PrometheusMetrics"] = None
    _lock = threading.Lock()

    def __init__(self, node_id: str, clique_id: int) -> None:
        self.node_id = node_id
        self.clique_id = clique_id
        self._server_started = False
        self._current_round = 0

        if not PROMETHEUS_AVAILABLE:
            return

        labels = ["node_id", "clique_id"]
        round_labels = ["node_id", "clique_id", "round"]

        self.current_round = Gauge(
            "fl_current_round",
            "Current federated learning round",
            labels,
        )
        self.training_samples = Gauge(
            "fl_training_samples",
            "Number of local training samples",
            labels,
        )
        self.model_parameters = Gauge(
            "fl_model_parameters",
            "Number of model parameters",
            labels,
        )
        self.train_accuracy = Gauge(
            "fl_train_accuracy",
            "Training accuracy",
            labels,
        )
        self.val_accuracy = Gauge(
            "fl_val_accuracy",
            "Validation accuracy",
            labels,
        )
        self.test_accuracy = Gauge(
            "fl_test_accuracy",
            "Test accuracy after aggregation",
            labels,
        )
        self.accuracy = Gauge(
            "fl_accuracy",
            "Model accuracy with dataset label",
            ["node_id", "clique_id", "dataset"],
        )
        self.accuracy_by_round = Gauge(
            "fl_accuracy_by_round",
            "Model accuracy indexed by round",
            ["node_id", "clique_id", "dataset", "round"],
        )
        self.convergence_signal = Gauge(
            "fl_convergence_signal",
            "Convergence signal (1 if converged)",
            labels,
        )
        self.convergence_metric = Gauge(
            "fl_convergence_metric",
            "Current convergence metric value",
            labels,
        )
        self.cluster_converged = Gauge(
            "fl_cluster_converged",
            "Cluster convergence status (1 if converged)",
            labels,
        )
        self.convergence_streak = Gauge(
            "fl_convergence_streak",
            "Current convergence streak count",
            labels,
        )
        self.delta_norm = Gauge(
            "fl_delta_norm",
            "Model delta norm between rounds",
            labels,
        )
        self.delta_by_round = Gauge(
            "fl_delta_by_round",
            "Model delta norm indexed by round",
            round_labels,
        )
        self.streak_by_round = Gauge(
            "fl_streak_by_round",
            "Convergence streak indexed by round",
            round_labels,
        )
        self.converged_by_round = Gauge(
            "fl_converged_by_round",
            "Cluster converged status indexed by round",
            round_labels,
        )
        self.is_aggregator = Gauge(
            "fl_is_aggregator",
            "Whether this node is the aggregator (1 if true)",
            labels,
        )

        self.local_training_time = Histogram(
            "fl_local_training_seconds",
            "Time spent on local training",
            labels,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        )
        self.sap_phase_time = Histogram(
            "fl_sap_phase_seconds",
            "Time spent on secure aggregation phases",
            ["node_id", "clique_id", "phase"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )
        self.aggregation_time = Histogram(
            "fl_aggregation_seconds",
            "Time spent on aggregation",
            labels,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        self.aggregation_time_by_round = Gauge(
            "fl_aggregation_time_by_round",
            "Aggregation time indexed by round",
            round_labels,
        )
        self.round_total_time = Histogram(
            "fl_round_total_seconds",
            "Total time per round",
            labels,
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
        )
        self.round_time_by_round = Gauge(
            "fl_round_time_by_round",
            "Total round time indexed by round",
            round_labels,
        )
        self.training_time_by_round = Gauge(
            "fl_training_time_by_round",
            "Local training time indexed by round",
            round_labels,
        )

        self.bytes_sent = Counter(
            "fl_bytes_sent_total",
            "Total bytes sent",
            labels,
        )
        self.bytes_received = Counter(
            "fl_bytes_received_total",
            "Total bytes received",
            labels,
        )
        self.messages_sent = Counter(
            "fl_messages_sent_total",
            "Total messages sent",
            labels,
        )
        self.messages_received = Counter(
            "fl_messages_received_total",
            "Total messages received",
            labels,
        )

    @classmethod
    def get_instance(cls, node_id: str, clique_id: int) -> "PrometheusMetrics":
        """Get or create the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(node_id, clique_id)
            return cls._instance

    def start_server(self, port: int = 8000) -> None:
        """Start the Prometheus HTTP server."""
        if not PROMETHEUS_AVAILABLE or self._server_started:
            return
        try:
            start_http_server(port)
            self._server_started = True
        except Exception:
            pass

    def _labels(self) -> dict:
        return {"node_id": self.node_id, "clique_id": str(self.clique_id)}

    def _round_labels(self) -> dict:
        return {"node_id": self.node_id, "clique_id": str(self.clique_id), "round": str(self._current_round)}

    def set_round(self, round_idx: int) -> None:
        if PROMETHEUS_AVAILABLE:
            self._current_round = round_idx
            self.current_round.labels(**self._labels()).set(round_idx)

    def set_training_samples(self, count: int) -> None:
        if PROMETHEUS_AVAILABLE:
            self.training_samples.labels(**self._labels()).set(count)

    def set_model_parameters(self, count: int) -> None:
        if PROMETHEUS_AVAILABLE:
            self.model_parameters.labels(**self._labels()).set(count)

    def set_accuracy(self, train_acc: float, val_acc: float, test_acc: float) -> None:
        if PROMETHEUS_AVAILABLE:
            labels = self._labels()
            self.train_accuracy.labels(**labels).set(train_acc)
            self.val_accuracy.labels(**labels).set(val_acc)
            self.test_accuracy.labels(**labels).set(test_acc)
            self.accuracy.labels(**labels, dataset="train").set(train_acc)
            self.accuracy.labels(**labels, dataset="validation").set(val_acc)
            self.accuracy.labels(**labels, dataset="test").set(test_acc)
            round_labels = self._round_labels()
            self.accuracy_by_round.labels(**round_labels, dataset="train").set(train_acc)
            self.accuracy_by_round.labels(**round_labels, dataset="validation").set(val_acc)
            self.accuracy_by_round.labels(**round_labels, dataset="test").set(test_acc)

    def set_convergence(self, delta_norm: float, streak: int, converged: bool) -> None:
        if PROMETHEUS_AVAILABLE:
            labels = self._labels()
            self.convergence_signal.labels(**labels).set(1 if converged else 0)
            self.convergence_metric.labels(**labels).set(delta_norm)
            self.cluster_converged.labels(**labels).set(1 if converged else 0)
            self.convergence_streak.labels(**labels).set(streak)
            self.delta_norm.labels(**labels).set(delta_norm)
            round_labels = self._round_labels()
            self.delta_by_round.labels(**round_labels).set(delta_norm)
            self.streak_by_round.labels(**round_labels).set(streak)
            self.converged_by_round.labels(**round_labels).set(1 if converged else 0)

    def set_aggregator_status(self, is_agg: bool) -> None:
        if PROMETHEUS_AVAILABLE:
            self.is_aggregator.labels(**self._labels()).set(1 if is_agg else 0)

    def observe_local_training(self, duration: float) -> None:
        if PROMETHEUS_AVAILABLE:
            self.local_training_time.labels(**self._labels()).observe(duration)
            self.training_time_by_round.labels(**self._round_labels()).set(duration)

    def observe_sap_phase(self, phase: str, duration: float) -> None:
        if PROMETHEUS_AVAILABLE:
            self.sap_phase_time.labels(
                node_id=self.node_id,
                clique_id=str(self.clique_id),
                phase=phase,
            ).observe(duration)

    def observe_aggregation(self, duration: float) -> None:
        if PROMETHEUS_AVAILABLE:
            self.aggregation_time.labels(**self._labels()).observe(duration)
            self.aggregation_time_by_round.labels(**self._round_labels()).set(duration)

    def observe_round_total(self, duration: float) -> None:
        if PROMETHEUS_AVAILABLE:
            self.round_total_time.labels(**self._labels()).observe(duration)
            self.round_time_by_round.labels(**self._round_labels()).set(duration)

    def add_bytes_sent(self, count: int) -> None:
        if PROMETHEUS_AVAILABLE:
            self.bytes_sent.labels(**self._labels()).inc(count)

    def add_bytes_received(self, count: int) -> None:
        if PROMETHEUS_AVAILABLE:
            self.bytes_received.labels(**self._labels()).inc(count)

    def add_messages_sent(self, count: int) -> None:
        if PROMETHEUS_AVAILABLE:
            self.messages_sent.labels(**self._labels()).inc(count)

    def add_messages_received(self, count: int) -> None:
        if PROMETHEUS_AVAILABLE:
            self.messages_received.labels(**self._labels()).inc(count)
