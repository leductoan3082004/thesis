import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class NodeRole(str, Enum):
    TRAINER = "trainer"
    AGGREGATOR = "aggregator"
    HYBRID = "hybrid"
    TTP = "ttp"


@dataclass
class Timeouts:
    advertise_keys: float = 5.0
    share_keys: float = 5.0
    masked_input: float = 5.0
    consistency: float = 5.0
    unmasking: float = 5.0

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, float]]) -> "Timeouts":
        base = cls()
        if not data:
            return base
        for key, value in data.items():
            if not hasattr(base, key):
                raise ValueError(f"Unknown timeout key '{key}'")
            if value <= 0:
                raise ValueError(f"Timeout '{key}' must be positive")
            setattr(base, key, float(value))
        return base


@dataclass
class MountConfig:
    config_dir: str = "config"
    data_dir: str = "data"
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, str]]) -> "MountConfig":
        if not data:
            return cls()
        kwargs = {}
        for key in ("config_dir", "data_dir", "logs_dir", "checkpoints_dir"):
            if key in data:
                value = str(data[key]).strip()
                if not value:
                    raise ValueError(f"Mount path '{key}' cannot be empty")
                kwargs[key] = value
        return cls(**kwargs)


@dataclass
class ScenarioConfig:
    name: str
    participants: List[str]
    threshold: int
    clique_size: int
    inter_clique_edges: str = "small_world"
    topology_iterations: int = 1000
    small_world_c: int = 2
    service_hostnames: Dict[str, str] = field(default_factory=dict)
    timeouts: Timeouts = field(default_factory=Timeouts)
    mounts: MountConfig = field(default_factory=MountConfig)

    @classmethod
    def from_file(cls, path: Path) -> "ScenarioConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: Dict) -> "ScenarioConfig":
        try:
            name = str(data["name"])
            participants = list(data["participants"])
            threshold = int(data["threshold"])
            clique_size = int(data.get("clique_size", 10))
        except KeyError as exc:
            raise ValueError(f"Scenario missing required field {exc}") from exc
        if not participants:
            raise ValueError("Scenario must include at least one participant")
        if len(set(participants)) != len(participants):
            raise ValueError("Scenario participants must be unique")
        if threshold <= 0 or threshold > len(participants):
            raise ValueError("Scenario threshold must satisfy 0 < t <= |participants|")
        inter_clique_edges = data.get("inter_clique_edges", "small_world")
        if inter_clique_edges not in {"ring", "fractal", "small_world", "fully_connected"}:
            raise ValueError(f"Unknown inter-clique edge mode '{inter_clique_edges}'")
        topology_iterations = int(data.get("topology_iterations", 1000))
        if topology_iterations <= 0:
            raise ValueError("topology_iterations must be positive")
        small_world_c = int(data.get("small_world_c", 2))
        if small_world_c <= 0:
            raise ValueError("small_world_c must be positive")
        service_hostnames = {
            str(k): str(v) for k, v in (data.get("service_hostnames") or {}).items()
        }
        timeouts = Timeouts.from_mapping(data.get("timeouts"))
        mounts = MountConfig.from_mapping(data.get("mounts"))
        cfg = cls(
            name=name,
            participants=participants,
            threshold=threshold,
            clique_size=clique_size,
            inter_clique_edges=inter_clique_edges,
            topology_iterations=topology_iterations,
            small_world_c=small_world_c,
            service_hostnames=service_hostnames,
            timeouts=timeouts,
            mounts=mounts,
        )
        return cfg

    def ensure_services_match(self, nodes: Iterable["NodeConfig"]) -> None:
        """Validate that node hostnames align with the scenario mapping."""
        if not self.service_hostnames:
            return
        for node in nodes:
            expected = self.service_hostnames.get(node.node_id)
            if expected and node.host != expected:
                raise ValueError(
                    f"Node '{node.node_id}' host '{node.host}' "
                    f"does not match scenario hostname '{expected}'"
                )


@dataclass
class NodeConfig:
    node_id: str
    role: NodeRole
    host: str
    port: int
    log_level: str = "INFO"
    tls_enabled: bool = False
    timeouts: Timeouts = field(default_factory=Timeouts)

    @classmethod
    def from_file(cls, path: Path, scenario: Optional[ScenarioConfig] = None) -> "NodeConfig":
        return cls.from_dict(json.loads(Path(path).read_text()), scenario)

    @classmethod
    def from_dict(cls, data: Dict, scenario: Optional[ScenarioConfig] = None) -> "NodeConfig":
        try:
            node_id = str(data["node_id"])
            role = NodeRole(str(data["role"]))
            host = str(data["host"])
            port = int(data["port"])
        except KeyError as exc:
            raise ValueError(f"Node config missing required field {exc}") from exc
        if port <= 0 or port > 65535:
            raise ValueError("Node port must be within 1-65535")
        log_level = str(data.get("log_level", "INFO")).upper()
        tls_enabled = bool(data.get("tls_enabled", False))
        timeouts = Timeouts.from_mapping(data.get("timeouts"))
        cfg = cls(
            node_id=node_id,
            role=role,
            host=host,
            port=port,
            log_level=log_level,
            tls_enabled=tls_enabled,
            timeouts=timeouts,
        )
        if scenario:
            cfg.validate_against_scenario(scenario)
        return cfg

    def validate_against_scenario(self, scenario: ScenarioConfig) -> None:
        if self.node_id not in scenario.participants and self.role != NodeRole.TTP:
            raise ValueError(
                f"Node '{self.node_id}' is not present in scenario participants"
            )
        scenario_host = scenario.service_hostnames.get(self.node_id)
        if scenario_host and scenario_host != self.host:
            raise ValueError(
                f"Node '{self.node_id}' host '{self.host}' does not match "
                f"scenario service hostname '{scenario_host}'"
            )


def validate_mounts(base_path: Path, mounts: MountConfig, create: bool = False) -> Dict[str, Path]:
    """
    Ensure required directories are present under base_path (e.g., for Compose mounts).
    Returns resolved paths keyed by mount name.
    """
    resolved: Dict[str, Path] = {}
    for attr in ("config_dir", "data_dir", "logs_dir", "checkpoints_dir"):
        rel = getattr(mounts, attr)
        path = (base_path / rel).resolve()
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Mount path '{rel}' exists but is not a directory")
        else:
            if create:
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Mount path '{rel}' does not exist at {path}")
        resolved[attr] = path
    return resolved
