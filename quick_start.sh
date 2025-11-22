#!/bin/bash
# Quick start script - runs docker compose without rebuilding

set -e

echo "🚀 Starting Secure Aggregation FL System..."

# Navigate to docker directory
cd "$(dirname "$0")/docker"

# Check if images exist
if ! docker images | grep -q "docker-node_0"; then
    echo "📦 First run detected - building images..."
    docker compose build
fi

# Start containers
echo "▶️  Starting containers..."
docker compose up -d

echo ""
echo "✅ System started in background!"
echo ""
echo "📊 View logs:"
echo "   docker compose -f docker/docker-compose.yml logs -f"
echo ""
echo "📈 Monitor specific node:"
echo "   docker compose -f docker/docker-compose.yml logs -f node_0"
echo ""
echo "🛑 Stop system:"
echo "   docker compose -f docker/docker-compose.yml down"
echo ""
