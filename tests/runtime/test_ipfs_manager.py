"""Tests for scripts.runtime.ipfs_manager — IPFS layout loading and node configuration."""

import json

import pytest

from scripts.runtime.ipfs_manager import (
    IPFSLayout,
    IPFSNode,
    _resolve_path,
    load_layout,
)


class TestResolvePath:
    def test_absolute_path_unchanged(self):
        p = _resolve_path("/absolute/path")
        assert str(p) == "/absolute/path"

    def test_relative_path_resolved_to_root(self):
        p = _resolve_path("data/ipfs/node-1")
        assert p.is_absolute()
        assert str(p).endswith("data/ipfs/node-1")


class TestLoadLayout:
    def test_valid_config(self, tmp_path):
        config = {
            "peer_registry": str(tmp_path / "peers"),
            "nodes": [
                {
                    "name": "ipfs-1",
                    "repo": str(tmp_path / "repo1"),
                    "api_host": "0.0.0.0",
                    "api_port": 5001,
                    "gateway_host": "0.0.0.0",
                    "gateway_port": 8080,
                    "swarm_host": "0.0.0.0",
                    "swarm_port": 4001,
                    "advertise_host": "127.0.0.1",
                    "advertise_proto": "ip4",
                },
            ],
        }
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text(json.dumps(config))
        layout = load_layout(cfg_file)
        assert len(layout.nodes) == 1
        assert layout.nodes[0].name == "ipfs-1"
        assert layout.nodes[0].api_port == 5001

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="not found"):
            load_layout(tmp_path / "nonexistent.json")

    def test_empty_nodes_raises(self, tmp_path):
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text('{"nodes": []}')
        with pytest.raises(SystemExit, match="does not define"):
            load_layout(cfg_file)

    def test_missing_required_ports_raises(self, tmp_path):
        config = {
            "nodes": [
                {
                    "name": "bad",
                    "repo": str(tmp_path / "repo"),
                    "api_port": 5001,
                    # Missing gateway_port and swarm_port.
                }
            ]
        }
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text(json.dumps(config))
        with pytest.raises(SystemExit, match="missing"):
            load_layout(cfg_file)

    def test_multiple_nodes(self, tmp_path):
        nodes = []
        for i in range(3):
            nodes.append({
                "name": f"ipfs-{i}",
                "repo": str(tmp_path / f"repo-{i}"),
                "api_port": 5001 + i,
                "gateway_port": 8080 + i,
                "swarm_port": 4001 + i,
            })
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text(json.dumps({"nodes": nodes}))
        layout = load_layout(cfg_file)
        assert len(layout.nodes) == 3

    def test_default_host_values(self, tmp_path):
        config = {
            "nodes": [
                {
                    "repo": str(tmp_path / "repo"),
                    "api_port": 5001,
                    "gateway_port": 8080,
                    "swarm_port": 4001,
                }
            ]
        }
        cfg_file = tmp_path / "ipfs.json"
        cfg_file.write_text(json.dumps(config))
        layout = load_layout(cfg_file)
        node = layout.nodes[0]
        assert node.api_host == "127.0.0.1"
        assert node.gateway_host == "127.0.0.1"
        assert node.swarm_host == "0.0.0.0"
        assert node.advertise_host == "127.0.0.1"
        assert node.advertise_proto == "dns4"
