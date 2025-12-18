import pytest

from secure_aggregation.config.system import (
    SYSTEM_CONFIG_ENV_VAR,
    load_system_config,
    resolve_system_config_path,
)


def test_load_system_config_from_default_location(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    nodes_dir = config_dir / "nodes"
    nodes_dir.mkdir(parents=True)
    node_config = nodes_dir / "node_0.json"
    node_config.write_text("{}")
    system_config = config_dir / "system-config.json"
    system_config.write_text('{"convergence": {"patience": 5}}')

    data, path = load_system_config(node_config)

    assert path == system_config.resolve()
    assert data["convergence"]["patience"] == 5


def test_load_system_config_from_env_override(tmp_path, monkeypatch):
    nodes_dir = tmp_path / "configs"
    nodes_dir.mkdir()
    node_config = nodes_dir / "node.json"
    node_config.write_text("{}")
    override_path = tmp_path / "custom.json"
    override_path.write_text('{"convergence": {"enabled": false}}')
    monkeypatch.setenv(SYSTEM_CONFIG_ENV_VAR, str(override_path))

    data, path = load_system_config(node_config)

    assert path == override_path
    assert data["convergence"]["enabled"] is False


def test_resolve_system_config_path_handles_non_standard_layout(tmp_path, monkeypatch):
    node_config = tmp_path / "node.json"
    node_config.write_text("{}")

    path = resolve_system_config_path(node_config)

    assert path == (node_config.parent / "system-config.json").resolve()


def test_load_system_config_invalid_json(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    nodes_dir = config_dir / "nodes"
    nodes_dir.mkdir(parents=True)
    node_config = nodes_dir / "node_0.json"
    node_config.write_text("{}")
    system_config = config_dir / "system-config.json"
    system_config.write_text("{invalid json")

    with pytest.raises(ValueError):
        load_system_config(node_config)
