"""Tests for scripts.runtime.port_allocator — deterministic port allocation and conflict detection."""

import socket
import threading

import pytest

from scripts.runtime.port_allocator import (
    PortMap,
    allocate_ports,
    check_conflicts,
    verify_released,
    _port_in_use,
)


class TestPortMap:
    def test_all_ports_includes_infrastructure(self):
        pm = PortMap(ttp=50051, prometheus=9090, grafana=3000, loki=3100, nodes=[])
        labels = [label for label, _ in pm.all_ports()]
        assert "ttp" in labels
        assert "prometheus" in labels
        assert "grafana" in labels
        assert "loki" in labels

    def test_all_ports_includes_node_entries(self):
        nodes = [{"service": 51000, "aggregator": 52000, "bridge": 53000, "metrics": 61000}]
        pm = PortMap(ttp=50051, prometheus=9090, grafana=3000, loki=3100, nodes=nodes)
        all_ports = pm.all_ports()
        node_labels = [label for label, _ in all_ports if label.startswith("node_")]
        assert len(node_labels) == 4

    def test_all_ports_no_duplicates(self):
        pm = allocate_ports(3)
        ports = [port for _, port in pm.all_ports()]
        assert len(ports) == len(set(ports)), "Port map contains duplicate port numbers"


class TestAllocatePorts:
    def test_default_infrastructure_ports(self):
        pm = allocate_ports(1)
        assert pm.ttp == 50051
        assert pm.prometheus == 9090
        assert pm.grafana == 3000
        assert pm.loki == 3100

    def test_custom_infrastructure_ports(self):
        pm = allocate_ports(1, ttp_port=60000, prometheus_port=7090, grafana_port=4000, loki_port=4100)
        assert pm.ttp == 60000
        assert pm.prometheus == 7090
        assert pm.grafana == 4000
        assert pm.loki == 4100

    def test_node_count_matches(self):
        for n in (1, 4, 10):
            pm = allocate_ports(n)
            assert len(pm.nodes) == n

    def test_node_port_scheme(self):
        pm = allocate_ports(3, base_node=51000, base_metrics=61000)
        for i, node in enumerate(pm.nodes):
            assert node["service"] == 51000 + i
            assert node["aggregator"] == 52000 + i
            assert node["bridge"] == 53000 + i
            assert node["metrics"] == 61000 + i

    def test_zero_nodes(self):
        pm = allocate_ports(0)
        assert pm.nodes == []
        assert len(pm.all_ports()) == 4


class TestPortInUse:
    def test_unused_port_returns_false(self):
        assert _port_in_use(59999) is False

    def test_listening_port_returns_true(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            assert _port_in_use(port) is True
        finally:
            server.close()


class TestCheckConflicts:
    def test_no_conflicts_on_free_ports(self):
        pm = allocate_ports(2, ttp_port=59800, base_node=59900, base_metrics=59950,
                            prometheus_port=59810, grafana_port=59820, loki_port=59830)
        assert check_conflicts(pm) == []

    def test_detects_occupied_port(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            pm = PortMap(ttp=port, prometheus=59811, grafana=59821, loki=59831, nodes=[])
            conflicts = check_conflicts(pm)
            assert len(conflicts) == 1
            assert conflicts[0] == ("ttp", port)
        finally:
            server.close()


class TestVerifyReleased:
    def test_delegates_to_check_conflicts(self):
        pm = allocate_ports(1, ttp_port=59850, base_node=59860, base_metrics=59870,
                            prometheus_port=59880, grafana_port=59890, loki_port=59895)
        assert verify_released(pm) == []
