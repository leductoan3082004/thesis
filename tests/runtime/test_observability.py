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
    start_promtail,
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
        """When node_ids are provided, node_id labels are derived from node_index."""
        node_ids = ["trainer-node-001", "trainer-node-002"]
        config_path = generate_promtail_config(tmp_path, node_count=2, node_ids=node_ids)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        fl_job = [sc for sc in data["scrape_configs"] if sc["job_name"] == "fl_nodes"][0]
        matches = [s["match"] for s in fl_job["pipeline_stages"] if "match" in s]
        selectors = [m["selector"] for m in matches]
        assert '{service="fl_node", node_index="0"}' in selectors
        assert '{service="fl_node", node_index="1"}' in selectors
        static_labels = [m["stages"][0]["static_labels"]["node_id"] for m in matches]
        assert "trainer-node-001" in static_labels
        assert "trainer-node-002" in static_labels

    def test_single_static_config_for_fl_nodes(self, tmp_path):
        """FL node scraping uses one wildcard target to reduce target-manager load."""
        config_path = generate_promtail_config(tmp_path, node_count=3)
        content = config_path.read_text()
        if HAS_YAML:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        fl_job = [sc for sc in data["scrape_configs"] if sc["job_name"] == "fl_nodes"][0]
        assert len(fl_job["static_configs"]) == 1
        assert "node_*/logs/*.log" in fl_job["static_configs"][0]["labels"]["__path__"]

    def test_start_promtail_fails_fast_when_process_exits(self, tmp_path, monkeypatch):
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "pids").mkdir(parents=True, exist_ok=True)

        class _DeadProc:
            pid = 12345
            returncode = 1

            @staticmethod
            def poll():
                return 1

        monkeypatch.setattr("scripts.runtime.observability.shutil.which", lambda _name: "/usr/bin/promtail")
        monkeypatch.setattr("scripts.runtime.observability.subprocess.Popen", lambda *args, **kwargs: _DeadProc())
        monkeypatch.setattr("scripts.runtime.observability.time.sleep", lambda _secs: None)

        with pytest.raises(SystemExit, match="Promtail exited immediately"):
            start_promtail(tmp_path, node_count=2, node_ids=["trainer-node-001", "trainer-node-002"])

    def test_start_promtail_skips_when_env_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SKIP_PROMTAIL", "1")
        result = start_promtail(tmp_path, node_count=1)
        assert result is None


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
