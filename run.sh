#!/bin/bash
# Quick start script for secure aggregation FL system

set -e

echo "=================================================="
echo "  Secure Aggregation FL - Quick Start"
echo "=================================================="
echo ""

# Check if running in project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Step 1: Checking prerequisites..."
if ! command_exists python; then
    echo "Error: Python is not installed"
    exit 1
fi

if ! command_exists docker; then
    echo "Error: Docker is not installed"
    exit 1
fi

echo "✓ All prerequisites found"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Step 2: Creating virtual environment..."
    python -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "Step 2: Virtual environment already exists"
fi
echo ""

# Activate venv and install dependencies
echo "Step 3: Installing dependencies..."
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[mnist]"
pip install -q grpcio grpcio-tools
echo "✓ Dependencies installed"
echo ""

# Generate gRPC code
echo "Step 4: Generating gRPC code..."
python -m grpc_tools.protoc \
    -I=protos \
    --python_out=src/secure_aggregation/communication \
    --grpc_python_out=src/secure_aggregation/communication \
    protos/secureagg.proto
echo "✓ gRPC code generated"
echo ""

# Download MNIST if needed
if [ ! -d "data/MNIST" ]; then
    echo "Step 5: Downloading MNIST dataset..."
    python scripts/prepare_data.py
    echo "✓ MNIST downloaded"
else
    echo "Step 5: MNIST dataset already exists"
fi
echo ""

# Ask user what to do
echo "=================================================="
echo "Setup complete! What would you like to do?"
echo "=================================================="
echo ""
echo "1) Run with Docker Compose (recommended)"
echo "2) Run tests"
echo "3) Exit"
echo ""
read -p "Enter your choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Starting Docker Compose..."
        echo "=================================================="
        echo "TIP: Press Ctrl+C to stop all containers"
        echo "TIP: Use 'docker compose logs -f node_0' to view specific logs"
        echo "=================================================="
        echo ""
        sleep 2
        cd docker
        docker compose up --build
        ;;
    2)
        echo ""
        echo "Running tests..."
        pytest tests/ -v
        ;;
    3)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
