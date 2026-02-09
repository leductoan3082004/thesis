"""Deterministic port allocation and conflict detection for process-mode runtime."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class PortMap:
    """Complete port mapping for the entire stack."""

    ttp: int
    prometheus: int
    grafana: int
    loki: int
    nodes: List[Dict[str, int]] = field(default_factory=list)

    def all_ports(self) -> List[Tuple[str, int]]:
        result: List[Tuple[str, int]] = [
            ("ttp", self.ttp),
            ("prometheus", self.prometheus),
            ("grafana", self.grafana),
            ("loki", self.loki),
        ]
        for i, node in enumerate(self.nodes):
            for role, port in node.items():
                result.append((f"node_{i}_{role}", port))
        return result


def allocate_ports(
    num_nodes: int,
    *,
    ttp_port: int = 50051,
    base_node: int = 51000,
    base_metrics: int = 61000,
    prometheus_port: int = 9090,
    grafana_port: int = 3000,
    loki_port: int = 3100,
) -> PortMap:
    """Build a deterministic port map for *num_nodes* FL nodes plus infrastructure."""
    nodes: List[Dict[str, int]] = []
    for i in range(num_nodes):
        service_port = base_node + i
        nodes.append(
            {
                "service": service_port,
                "aggregator": service_port + 1000,
                "bridge": service_port + 2000,
                "metrics": base_metrics + i,
            }
        )
    return PortMap(
        ttp=ttp_port,
        prometheus=prometheus_port,
        grafana=grafana_port,
        loki=loki_port,
        nodes=nodes,
    )


def check_conflicts(port_map: PortMap) -> List[Tuple[str, int]]:
    """Return a list of (label, port) pairs that are already in use."""
    conflicts: List[Tuple[str, int]] = []
    for label, port in port_map.all_ports():
        if _port_in_use(port):
            conflicts.append((label, port))
    return conflicts


def verify_released(port_map: PortMap) -> List[Tuple[str, int]]:
    """Return ports from *port_map* that are still in use after shutdown."""
    return check_conflicts(port_map)


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.3)
        result = sock.connect_ex((host, port))
        return result == 0
    finally:
        sock.close()
