# Makefile for Secure Aggregation Federated Learning System (Process-Only Runtime)
#
# Usage:
#   make setup          - Install dependencies and generate gRPC code
#   make start          - Start the system (infrastructure + training)
#   make stop           - Stop all services
#   make status         - Show status of all managed processes
#   make logs           - View logs (via Loki or file fallback)
#   make clean          - Stop processes and remove generated files
#   make test           - Run tests
#
# Configuration:
#   NODES=6             - Number of training nodes (default: 6)
#   CLIQUE_SIZE=3       - Size of each clique in D-Cliques topology (default: 3)

.PHONY: setup start stop status logs clean test help install-ipfs install-fabric build-vctool
.PHONY: setup-venv setup-deps setup-grpc setup-data setup-blockchain setup-monitoring
.PHONY: logs-node logs-errors clean-all

SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)
PROJECT_ROOT := $(shell pwd)
BLOCKCHAIN_DIR := $(PROJECT_ROOT)/../thesis-blockchain/api-gateway
BLOCKCHAIN_REPO_URL ?= https://github.com/letienthanh364/thesis-blockchain.git
VENV := $(PROJECT_ROOT)/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

NODES ?= 6
CLIQUE_SIZE ?= 3
NODES_MAP ?=
STATE_MAP ?=

MAP_PATH := $(strip $(if $(NODES_MAP),$(NODES_MAP),$(STATE_MAP)))

ifeq ($(strip $(MAP_PATH)),)
STATE_ARG := --nodes $(NODES)
else
STATE_ARG := --nodes-map $(MAP_PATH)
override NODES := $(shell $(PYTHON) $(PROJECT_ROOT)/scripts/nodes_map_count.py $(MAP_PATH))
endif

# Default target
help:
	@echo "Secure Aggregation FL System (Process-Only Runtime)"
	@echo ""
	@echo "Usage:"
	@echo "  make setup              Install dependencies and generate gRPC code"
	@echo "  make start              Start full system (blockchain + monitoring + training)"
	@echo "  make stop               Stop all managed processes"
	@echo "  make status             Show status of all managed processes"
	@echo "  make logs               View aggregated logs (Loki or file fallback)"
	@echo "  make logs-node NODE=X   View logs for a specific node"
	@echo "  make logs-errors        View error-level logs only"
	@echo "  make clean              Stop processes and remove generated files"
	@echo "  make test               Run unit tests"
	@echo ""
	@echo "Options:"
	@echo "  NODES=N                 Number of training nodes (default: 6)"
	@echo "  CLIQUE_SIZE=N           Size of each clique (default: 3)"
	@echo "  NODES_MAP=path          Hierarchical node roster (overrides NODES)"
	@echo "  STATE_MAP=path          Legacy alias for NODES_MAP"
	@echo "  BLOCKCHAIN_REPO_URL=... Override thesis-blockchain git URL"
	@echo ""
	@echo "Examples:"
	@echo "  make start NODES=10 CLIQUE_SIZE=5"
	@echo "  make start NODES_MAP=config/nodes-map.json CLIQUE_SIZE=4"
	@echo "  make logs-node NODE=trainer-node-001"
	@echo "  make logs-errors"

install-ipfs:
	@echo "Installing IPFS (Kubo)..."
	@bash $(PROJECT_ROOT)/scripts/install_ipfs.sh

install-fabric:
	@echo "Installing Hyperledger Fabric CLI binaries..."
	@bash $(PROJECT_ROOT)/scripts/install_fabric_binaries.sh

build-vctool:
	@echo "Building vctool from blockchain repo..."
	@bash $(PROJECT_ROOT)/scripts/build_vctool.sh


# ------------------------------------------------------------------------------
# Setup targets
# ------------------------------------------------------------------------------

setup: setup-venv setup-deps setup-grpc setup-data setup-blockchain setup-monitoring
	@echo ""
	@echo "Setup complete. Run 'make start' to launch the system."

setup-venv:
	@echo "[1/5] Setting up virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
		echo "      Virtual environment created"; \
	else \
		echo "      Virtual environment already exists"; \
	fi

setup-deps: setup-venv
	@echo "[2/5] Installing dependencies..."
	@$(PIP) install -q --upgrade pip
	@$(PIP) install -q -e "$(PROJECT_ROOT)[mnist]" 2>/dev/null || $(PIP) install -q -e "$(PROJECT_ROOT)"
	@$(PIP) install -q grpcio grpcio-tools PyYAML
	@echo "      Dependencies installed"

setup-grpc: setup-deps
	@echo "[3/5] Generating gRPC code..."
	@$(PYTHON) -m grpc_tools.protoc \
		-I=$(PROJECT_ROOT)/protos \
		--python_out=$(PROJECT_ROOT)/src/secure_aggregation/communication \
		--grpc_python_out=$(PROJECT_ROOT)/src/secure_aggregation/communication \
		$(PROJECT_ROOT)/protos/secureagg.proto
	@echo "      gRPC code generated"

setup-data: setup-deps
	@echo "[4/5] Preparing MNIST dataset..."
	@if [ ! -d "$(PROJECT_ROOT)/data/MNIST" ]; then \
		$(PYTHON) $(PROJECT_ROOT)/scripts/prepare_data.py; \
		echo "      MNIST downloaded"; \
	else \
		echo "      MNIST dataset already exists"; \
	fi

setup-blockchain:
	@echo "[5/5] Setting up blockchain environment..."
	@if [ ! -d "$(BLOCKCHAIN_DIR)" ]; then \
		echo "      Blockchain repository not found. Cloning to ../thesis-blockchain..."; \
		BLOCKCHAIN_PARENT="$$(dirname "$(BLOCKCHAIN_DIR)")"; \
		mkdir -p "$$BLOCKCHAIN_PARENT"; \
		git clone "$(BLOCKCHAIN_REPO_URL)" "$$BLOCKCHAIN_PARENT"; \
	fi
	@if [ ! -f "$(BLOCKCHAIN_DIR)/.env.example" ]; then \
		echo "ERROR: Missing $(BLOCKCHAIN_DIR)/.env.example"; \
		echo "       Verify thesis-blockchain was cloned correctly."; \
		exit 1; \
	fi
	@if [ ! -f "$(BLOCKCHAIN_DIR)/.env" ]; then \
		cp "$(BLOCKCHAIN_DIR)/.env.example" "$(BLOCKCHAIN_DIR)/.env"; \
		sed -i '' 's/AUTH_JWT_SECRET=change-me/AUTH_JWT_SECRET=secure-agg-dev-$$(date +%s)/' "$(BLOCKCHAIN_DIR)/.env" 2>/dev/null || \
		sed -i 's/AUTH_JWT_SECRET=change-me/AUTH_JWT_SECRET=secure-agg-dev-secret/' "$(BLOCKCHAIN_DIR)/.env"; \
		echo "      Created blockchain .env file"; \
	else \
		echo "      Blockchain .env already exists"; \
	fi

setup-monitoring:
	@echo "[6/6] Installing monitoring tools (Loki, Promtail, Prometheus, Grafana)..."
	@bash $(PROJECT_ROOT)/scripts/install_monitoring.sh
	@echo "      Monitoring tools installed"


# ------------------------------------------------------------------------------
# Runtime targets (process-only via secureagg_ctl.py)
# ------------------------------------------------------------------------------

start: setup
	@echo ""
	@echo "Starting full system with $(NODES) nodes (clique_size=$(CLIQUE_SIZE))..."
	@$(PYTHON) $(PROJECT_ROOT)/scripts/secureagg_ctl.py start \
		$(STATE_ARG) \
		--clique-size $(CLIQUE_SIZE)

stop:
	@$(PYTHON) $(PROJECT_ROOT)/scripts/secureagg_ctl.py stop

status:
	@$(PYTHON) $(PROJECT_ROOT)/scripts/secureagg_ctl.py status

logs:
	@$(PYTHON) $(PROJECT_ROOT)/scripts/secureagg_ctl.py logs

logs-node:
	@if [ -z "$(NODE)" ]; then \
		echo "Usage: make logs-node NODE=trainer-node-001"; \
	else \
		$(PYTHON) $(PROJECT_ROOT)/scripts/secureagg_ctl.py logs --node $(NODE); \
	fi

logs-errors:
	@$(PYTHON) $(PROJECT_ROOT)/scripts/secureagg_ctl.py logs --errors


# ------------------------------------------------------------------------------
# Cleanup targets
# ------------------------------------------------------------------------------

clean:
	@echo "Stopping processes and cleaning generated files..."
	@$(PYTHON) $(PROJECT_ROOT)/scripts/secureagg_ctl.py cleanup --purge-logs 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/process-runtime 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/config/nodes/*.json 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/logs/* 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/checkpoints/* 2>/dev/null || true
	@rm -f $(PROJECT_ROOT)/config/topology.json 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/data/blockchain/* 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/data/ipfs/* 2>/dev/null || true
	@echo "Clean complete"

clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "Full clean complete"


# ------------------------------------------------------------------------------
# Test targets
# ------------------------------------------------------------------------------

test: setup-deps
	@echo "Running tests..."
	@PYTHONPATH=$(PROJECT_ROOT)/src $(PYTHON) -m pytest $(PROJECT_ROOT)/tests/ -v

test-coverage: setup-deps
	@echo "Running tests with coverage..."
	@PYTHONPATH=$(PROJECT_ROOT)/src $(PYTHON) -m pytest $(PROJECT_ROOT)/tests/ --cov=src/secure_aggregation -v
