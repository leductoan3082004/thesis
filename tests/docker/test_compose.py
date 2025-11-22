import yaml


def test_compose_services_and_mounts_present(tmp_path):
    content = (tmp_path / "docker-compose.yml").read_text() if (tmp_path / "docker-compose.yml").exists() else None
    # fallback to repo file
    if content is None:
        from pathlib import Path

        content = (Path(__file__).resolve().parents[2] / "docker" / "docker-compose.yml").read_text()
    compose = yaml.safe_load(content)
    services = compose.get("services", {})
    assert "ttp" in services and "node" in services
    node_vols = services["node"]["volumes"]
    for mount in ("../config:/app/config", "../data:/app/data", "../logs:/app/logs", "../checkpoints:/app/checkpoints"):
        assert mount in node_vols
    assert compose.get("networks") and "secureagg" in compose["networks"]
