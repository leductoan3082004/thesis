"""Tests for scripts.runtime.config_generator — layout generation and config helpers."""

import json
import os
import shutil
from pathlib import Path

import pytest

from scripts.runtime.config_generator import (
    IPFSTarget,
    _apply_blockchain_identity,
    _apply_ipfs_distribution,
    _normalize_trainer_id,
    _select_ipfs_target,
    determine_node_count,
    extract_scope_names,
    load_process_ipfs_targets,
    resolve_system_config_path,
)


class TestNormalizeTrainerId:
    def test_numeric_suffix(self):
        assert _normalize_trainer_id("trainer-123", 0) == "trainer-node-123"

    def test_node_prefix(self):
        assert _normalize_trainer_id("node-5", 0) == "trainer-node-005"

    def test_none_falls_back_to_index(self):
        assert _normalize_trainer_id(None, 0) == "trainer-node-001"

    def test_empty_falls_back_to_index(self):
        assert _normalize_trainer_id("", 2) == "trainer-node-003"

    def test_no_digits_sanitizes(self):
        assert _normalize_trainer_id("my-node", 0) == "my-node"

    def test_whitespace_stripped(self):
        assert _normalize_trainer_id("  trainer-42  ", 0) == "trainer-node-042"


class TestExtractScopeNames:
    def test_empty_config_returns_state(self):
        assert extract_scope_names({}) == ["state"]

    def test_no_hierarchy_levels(self):
        assert extract_scope_names({"other_key": True}) == ["state"]

    def test_ordered_by_scope_index(self):
        config = {
            "hierarchy_levels": [
                {"scope_name": "nation", "scope_index": 3},
                {"scope_name": "state", "scope_index": 1},
                {"scope_name": "cluster", "scope_index": 2},
            ]
        }
        result = extract_scope_names(config)
        assert result == ["state", "cluster", "nation"]

    def test_deduplication(self):
        config = {
            "hierarchy_levels": [
                {"scope_name": "state", "scope_index": 0},
                {"scope_name": "state", "scope_index": 1},
            ]
        }
        assert extract_scope_names(config) == ["state"]

    def test_missing_scope_name_skipped(self):
        config = {
            "hierarchy_levels": [
                {"scope_index": 0},
                {"scope_name": "state", "scope_index": 1},
            ]
        }
        assert extract_scope_names(config) == ["state"]


class TestDetermineNodeCount:
    def test_cli_overrides_config(self, tmp_path):
        cfg = tmp_path / "sys.json"
        cfg.write_text('{"number_of_nodes": 10}')
        assert determine_node_count(4, cfg) == 4

    def test_reads_from_config(self, tmp_path):
        cfg = tmp_path / "sys.json"
        cfg.write_text('{"number_of_nodes": 8}')
        assert determine_node_count(None, cfg) == 8

    def test_reads_from_deployment_key(self, tmp_path):
        cfg = tmp_path / "sys.json"
        cfg.write_text('{"deployment": {"number_of_nodes": 6}}')
        assert determine_node_count(None, cfg) == 6

    def test_missing_field_raises(self, tmp_path):
        cfg = tmp_path / "sys.json"
        cfg.write_text('{"other": 1}')
        with pytest.raises(SystemExit, match="number_of_nodes not found"):
            determine_node_count(None, cfg)

    def test_zero_cli_raises(self, tmp_path):
        cfg = tmp_path / "sys.json"
        cfg.write_text("{}")
        with pytest.raises(SystemExit, match="must be >= 1"):
            determine_node_count(0, cfg)

    def test_invalid_value_raises(self, tmp_path):
        cfg = tmp_path / "sys.json"
        cfg.write_text('{"number_of_nodes": "abc"}')
        with pytest.raises(SystemExit, match="Invalid number_of_nodes"):
            determine_node_count(None, cfg)

    def test_preloaded_config_data(self, tmp_path):
        cfg = tmp_path / "sys.json"
        cfg.write_text("{}")
        data = {"number_of_nodes": 5}
        assert determine_node_count(None, cfg, config_data=data) == 5


class TestResolveSystemConfigPath:
    def test_cli_path_absolute(self, tmp_path):
        p = tmp_path / "custom.json"
        assert resolve_system_config_path(p) == p.resolve()

    def test_env_var_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "env-config.json"
        monkeypatch.setenv("SYSTEM_CONFIG_PATH", str(custom))
        result = resolve_system_config_path(None)
        assert result == custom.resolve()

    def test_default_when_no_cli_or_env(self, monkeypatch):
        monkeypatch.delenv("SYSTEM_CONFIG_PATH", raising=False)
        result = resolve_system_config_path(None)
        assert result.name == "system-config.json"


class TestLoadProcessIpfsTargets:
    def test_valid_config(self, tmp_path):
        config = {
            "nodes": [
                {"name": "ipfs-1", "api_port": 5001, "client_host": "127.0.0.1"},
                {"name": "ipfs-2", "api_port": 5002},
            ]
        }
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text(json.dumps(config))
        targets = load_process_ipfs_targets(cfg_file)
        assert len(targets) == 2
        assert targets[0].name == "ipfs-1"
        assert targets[0].api_url == "http://127.0.0.1:5001"
        assert targets[1].api_url == "http://127.0.0.1:5002"

    def test_missing_api_port_raises(self, tmp_path):
        config = {"nodes": [{"name": "bad-node"}]}
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text(json.dumps(config))
        with pytest.raises(SystemExit, match="api_port"):
            load_process_ipfs_targets(cfg_file)

    def test_empty_nodes_raises(self, tmp_path):
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text('{"nodes": []}')
        with pytest.raises(SystemExit, match="at least one node"):
            load_process_ipfs_targets(cfg_file)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="not found"):
            load_process_ipfs_targets(tmp_path / "nonexistent.json")

    def test_custom_api_url_takes_precedence(self, tmp_path):
        config = {"nodes": [{"name": "n", "api_port": 5001, "api_url": "http://custom:9999"}]}
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text(json.dumps(config))
        targets = load_process_ipfs_targets(cfg_file)
        assert targets[0].api_url == "http://custom:9999"


class TestSelectIpfsTarget:
    def test_round_robin(self):
        targets = [IPFSTarget("a", "http://a"), IPFSTarget("b", "http://b"), IPFSTarget("c", "http://c")]
        assert _select_ipfs_target(0, targets).name == "a"
        assert _select_ipfs_target(1, targets).name == "b"
        assert _select_ipfs_target(3, targets).name == "a"

    def test_empty_raises(self):
        with pytest.raises(SystemExit, match="No IPFS"):
            _select_ipfs_target(0, [])


class TestApplyIpfsDistribution:
    def test_sets_api_url_and_replicas(self):
        targets = [IPFSTarget("a", "http://a"), IPFSTarget("b", "http://b")]
        config: dict = {}
        _apply_ipfs_distribution(targets[0], config, targets)
        assert config["inter_cluster"]["ipfs"]["api_url"] == "http://a"
        assert config["inter_cluster"]["ipfs"]["replica_api_urls"] == ["http://b"]

    def test_single_target_no_replicas(self):
        targets = [IPFSTarget("a", "http://a")]
        config: dict = {}
        _apply_ipfs_distribution(targets[0], config, targets)
        assert "replica_api_urls" not in config["inter_cluster"]["ipfs"]


class TestApplyBlockchainIdentity:
    def test_default_identity(self):
        config: dict = {}
        _apply_blockchain_identity(0, config)
        bc = config["inter_cluster"]["blockchain"]
        assert bc["identity"] == "trainer-node-001"
        assert "001" in bc["private_key_path"]

    def test_identity_override(self):
        config: dict = {}
        _apply_blockchain_identity(0, config, identity_override="custom-node")
        assert config["inter_cluster"]["blockchain"]["identity"] == "custom-node"

    def test_index_numbering(self):
        config: dict = {}
        _apply_blockchain_identity(9, config)
        assert config["inter_cluster"]["blockchain"]["identity"] == "trainer-node-010"


class TestDataLinkCleanup:
    """Verify that the data-link cleanup handles symlinks, directories, and files."""

    def _simulate_data_link_cleanup(self, data_link: Path) -> None:
        """Replicate the cleanup logic from generate_process_layout."""
        if data_link.is_symlink():
            data_link.unlink()
        elif data_link.is_dir():
            shutil.rmtree(data_link)
        elif data_link.exists():
            data_link.unlink()

    def test_cleanup_symlink(self, tmp_path):
        target = tmp_path / "real_data"
        target.mkdir()
        link = tmp_path / "data"
        link.symlink_to(target)
        self._simulate_data_link_cleanup(link)
        assert not link.exists()
        assert target.exists()

    def test_cleanup_directory(self, tmp_path):
        """A real directory from a previous --copy-data run must be removed."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file.txt").write_text("content")
        self._simulate_data_link_cleanup(data_dir)
        assert not data_dir.exists()

    def test_cleanup_file(self, tmp_path):
        data_file = tmp_path / "data"
        data_file.write_text("stale")
        self._simulate_data_link_cleanup(data_file)
        assert not data_file.exists()

    def test_cleanup_nonexistent_is_noop(self, tmp_path):
        self._simulate_data_link_cleanup(tmp_path / "data")
