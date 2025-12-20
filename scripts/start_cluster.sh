#!/bin/bash
# Start the federated learning cluster with auto-generated Grafana dashboards.
#
# Usage:
#   ./scripts/start_cluster.sh [docker-compose-file]
#
# Default: docker/docker-compose.6nodes.yml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

COMPOSE_FILE="${1:-$PROJECT_ROOT/docker/docker-compose.6nodes.yml}"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: Docker Compose file not found: $COMPOSE_FILE"
    exit 1
fi

echo "=== Generating Grafana Dashboard ==="
python3 "$SCRIPT_DIR/generate_grafana_dashboard.py"

echo ""
echo "=== Starting Docker Compose ==="
docker compose -f "$COMPOSE_FILE" up -d --build

echo ""
echo "=== Services Started ==="
echo "Grafana:    http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo ""
echo "Use 'docker compose -f $COMPOSE_FILE logs -f' to view logs"
