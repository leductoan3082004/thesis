"""Utilities for loading global system-level configuration."""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

SYSTEM_CONFIG_FILENAME = "system-config.json"
SYSTEM_CONFIG_ENV_VAR = "SYSTEM_CONFIG_PATH"


def _default_system_config_dir(node_config_path: Path) -> Path:
    """
    Determine the config directory that should contain the system-level config.

    Node configs live inside config/nodes/, so the system config defaults to
    config/system-config.json. If the node config is elsewhere, fall back to
    the same directory as the node config file.
    """
    config_dir = node_config_path.parent
    if config_dir.name == "nodes":
        config_dir = config_dir.parent
    return config_dir


def resolve_system_config_path(node_config_path: Path) -> Path:
    """Resolve the path to the system-level configuration file."""
    node_config_path = Path(node_config_path).resolve()
    env_value = os.getenv(SYSTEM_CONFIG_ENV_VAR)
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate
    return (_default_system_config_dir(node_config_path) / SYSTEM_CONFIG_FILENAME).resolve()


def load_system_config(node_config_path: Path) -> Tuple[Dict, Path]:
    """
    Load the system configuration shared across nodes.

    Returns:
        (config_dict, resolved_path)

    Raises:
        ValueError: if the JSON is invalid.
    """
    path = resolve_system_config_path(node_config_path)
    if not path.exists():
        return {}, path
    try:
        return json.loads(path.read_text()), path
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid system config JSON at {path}: {exc}") from exc
