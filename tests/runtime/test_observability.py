"""Tests for scripts.runtime.observability — config generation for Loki, Promtail, Prometheus, Grafana."""

import json

import pytest

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from scripts.runtime.observability import (
    generate_grafana_config,
    generate_loki_config,
    generate_prometheus_config,
    generate_promtail_config,
)


class TestGenerateLokiConfig:
    def test_creates_config_file(self, tmp_path):
        config_path = generate_loki_config(tmp_path)
        assert config_path.exists()

    def test_config_is_parseable(self, tmp_path):
        config_path = generate_loki_config(tmp_path)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        assert data["server"]["http_listen_port"] == 3100
        assert data["auth_enabled"] is False

    def test_creates_data_directories(self, tmp_path):
        generate_loki_config(tmp_path)
        loki_dir = tmp_path / "observability" / "loki"
        assert loki_dir.exists()


class TestGeneratePromtailConfig:
    def test_creates_config_file(self, tmp_path):
        config_path = generate_promtail_config(tmp_path, node_count=3)
        assert config_path.exists()

    def test_scrape_configs_include_all_services(self, tmp_path):
        config_path = generate_promtail_config(tmp_path, node_count=3)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        job_names = [sc["job_name"] for sc in data["scrape_configs"]]
        assert "fl_nodes" in job_names
        assert "ttp" in job_names
        assert "ipfs" in job_names

    def test_pushes_to_loki(self, tmp_path):
        config_path = generate_promtail_config(tmp_path, node_count=1)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        assert data["clients"][0]["url"] == "http://localhost:3100/loki/api/v1/push"

    def test_positions_file_inside_runtime(self, tmp_path):
        config_path = generate_promtail_config(tmp_path, node_count=1)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        positions = data["positions"]["filename"]
        assert str(tmp_path) in positions

    def test_node_ids_label_attached(self, tmp_path):
        """When node_ids are provided, each FL node stream gets a node_id label."""
        node_ids = ["trainer-node-001", "trainer-node-002"]
        config_path = generate_promtail_config(tmp_path, node_count=2, node_ids=node_ids)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        fl_job = [sc for sc in data["scrape_configs"] if sc["job_name"] == "fl_nodes"][0]
        static_cfgs = fl_job["static_configs"]
        assert len(static_cfgs) == 2
        assert static_cfgs[0]["labels"]["node_id"] == "trainer-node-001"
        assert static_cfgs[1]["labels"]["node_id"] == "trainer-node-002"

    def test_per_node_static_configs(self, tmp_path):
        """Each node gets its own static_config entry with node_index label."""
        config_path = generate_promtail_config(tmp_path, node_count=3)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        fl_job = [sc for sc in data["scrape_configs"] if sc["job_name"] == "fl_nodes"][0]
        assert len(fl_job["static_configs"]) == 3
        indices = [sc["labels"]["node_index"] for sc in fl_job["static_configs"]]
        assert indices == ["0", "1", "2"]


class TestGeneratePrometheusConfig:
    def test_creates_config_file(self, tmp_path):
        config_path = generate_prometheus_config(tmp_path, node_count=4)
        assert config_path.exists()

    def test_targets_match_node_count(self, tmp_path):
        config_path = generate_prometheus_config(tmp_path, node_count=4, base_metrics_port=61000)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        targets = data["scrape_configs"][0]["static_configs"][0]["targets"]
        assert len(targets) == 4
        assert "localhost:61000" in targets
        assert "localhost:61003" in targets

    def test_custom_base_port(self, tmp_path):
        config_path = generate_prometheus_config(tmp_path, node_count=2, base_metrics_port=70000)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        targets = data["scrape_configs"][0]["static_configs"][0]["targets"]
        assert "localhost:70000" in targets
        assert "localhost:70001" in targets


class TestGenerateGrafanaConfig:
    def test_creates_directory_structure(self, tmp_path):
        grafana_dir = generate_grafana_config(tmp_path)
        assert (grafana_dir / "provisioning" / "datasources" / "datasources.yml").exists()
        assert (grafana_dir / "provisioning" / "dashboards" / "dashboards.yml").exists()
        assert (grafana_dir / "dashboards").is_dir()

    def test_datasources_include_prometheus_and_loki(self, tmp_path):
        grafana_dir = generate_grafana_config(tmp_path)
        ds_path = grafana_dir / "provisioning" / "datasources" / "datasources.yml"
        content = ds_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        names = [ds["name"] for ds in data["datasources"]]
        assert "Prometheus" in names
        assert "Loki" in names

    def test_prometheus_is_default_datasource(self, tmp_path):
        grafana_dir = generate_grafana_config(tmp_path)
        ds_path = grafana_dir / "provisioning" / "datasources" / "datasources.yml"
        content = ds_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        for ds in data["datasources"]:
            if ds["name"] == "Prometheus":
                assert ds["isDefault"] is True

    def test_loki_datasource_url(self, tmp_path):
        grafana_dir = generate_grafana_config(tmp_path)
        ds_path = grafana_dir / "provisioning" / "datasources" / "datasources.yml"
        content = ds_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        loki_ds = [ds for ds in data["datasources"] if ds["name"] == "Loki"][0]
        assert loki_ds["url"] == "http://localhost:3100"
