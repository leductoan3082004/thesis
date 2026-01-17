# Makefile for Secure Aggregation Federated Learning System
#
# Usage:
#   make setup          - Install dependencies and generate gRPC code
#   make start          - Start the system (infrastructure + training)
#   make start-training - Restart only training nodes (keeps infrastructure)
#   make stop           - Stop all services
#   make logs           - View logs from all containers
#   make clean          - Remove generated files and containers
#   make test           - Run tests
#
# Configuration:
#   NODES=6             - Number of training nodes (default: 6)
#   CLIQUE_SIZE=3       - Size of each clique in D-Cliques topology (default: 3)
#   DETACH=1            - Run in background (default: foreground)

.PHONY: setup start start-training stop logs clean test help
.PHONY: setup-venv setup-deps setup-grpc setup-data setup-blockchain
.PHONY: generate-configs generate-dashboard
.PHONY: start-blockchain start-monitoring start-storage
.PHONY: stop-training clean-state

SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)
PROJECT_ROOT := $(shell pwd)
BLOCKCHAIN_DIR := $(PROJECT_ROOT)/../thesis-blockchain/api-gateway
VENV := $(PROJECT_ROOT)/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
COMPOSE_FILE := $(PROJECT_ROOT)/docker/docker-compose.auto.yml
COMPOSE_TEMPLATE := $(PROJECT_ROOT)/docker/docker-compose.yml

NODES ?= 6
CLIQUE_SIZE ?= 3
FOREGROUND ?= 0
STATE_MAP ?=
NO_BUILD ?= 0

ifeq ($(strip $(STATE_MAP)),)
STATE_ARG := --nodes $(NODES)
else
STATE_ARG := --state-map $(STATE_MAP)
override NODES := $(shell $(PYTHON) $(PROJECT_ROOT)/scripts/state_map_count.py $(STATE_MAP))
endif

BUILD_ARG := $(if $(filter 1,$(NO_BUILD)),--no-build,)

# Default target
help:
	@echo "Secure Aggregation FL System"
	@echo ""
	@echo "Usage:"
	@echo "  make setup              Install dependencies and generate gRPC code"
	@echo "  make start              Start full system (blockchain + monitoring + training)"
	@echo "  make start-training     Restart training nodes only (keeps infrastructure)"
	@echo "  make stop               Stop all services"
	@echo "  make logs               View container logs"
	@echo "  make clean              Remove generated files and stop containers"
	@echo "  make test               Run unit tests"
	@echo ""
	@echo "Options:"
	@echo "  NODES=N                 Number of training nodes (default: 6)"
	@echo "  CLIQUE_SIZE=N           Size of each clique (default: 3)"
	@echo "  STATE_MAP=path          JSON file describing states -> node identities"
	@echo "  NO_BUILD=1              Skip rebuilding the shared node image"
	@echo "  FOREGROUND=1            Run containers in foreground (default: background)"
	@echo ""
	@echo "Examples:"
	@echo "  make start NODES=10 CLIQUE_SIZE=5    Start with 10 nodes in cliques of 5"
	@echo "  make start STATE_MAP=config/state-map.json CLIQUE_SIZE=4"
	@echo "  make start NO_BUILD=1                Reuse previously built images"
	@echo "  make start FOREGROUND=1              Start in foreground (watch logs)"
	@echo "  make start-training NODES=8          Restart training with 8 nodes"


# ------------------------------------------------------------------------------
# Setup targets
# ------------------------------------------------------------------------------

setup: setup-venv setup-deps setup-grpc setup-data setup-blockchain
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
		echo "ERROR: Blockchain repository not found at $(BLOCKCHAIN_DIR)"; \
		echo "       Clone thesis-blockchain as a sibling directory."; \
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


# ------------------------------------------------------------------------------
# Generate targets
# ------------------------------------------------------------------------------

generate-configs: setup-deps setup-blockchain
	@echo "Generating node configurations for $(NODES) nodes (clique_size=$(CLIQUE_SIZE))..."
	@$(PYTHON) $(PROJECT_ROOT)/scripts/run_docker_with_nodes.py \
		$(STATE_ARG) \
		--clique-size $(CLIQUE_SIZE) \
		$(BUILD_ARG) \
		--generate-only

generate-dashboard: setup-deps
	@echo "Generating Grafana dashboard..."
	@$(PYTHON) $(PROJECT_ROOT)/scripts/generate_grafana_dashboard.py


# ------------------------------------------------------------------------------
# Start targets
# ------------------------------------------------------------------------------

start: setup
	@echo ""
	@echo "Starting full system with $(NODES) nodes (clique_size=$(CLIQUE_SIZE))..."
	@$(PYTHON) $(PROJECT_ROOT)/scripts/run_docker_with_nodes.py \
		$(STATE_ARG) \
		--clique-size $(CLIQUE_SIZE) \
		$(BUILD_ARG) \
		$(if $(filter 1,$(FOREGROUND)),--no-detach,)

start-training: setup generate-configs stop-training clean-state
	@echo ""
	@echo "Starting training services with $(NODES) nodes..."
	@docker compose -f $(COMPOSE_FILE) up --build -d \
		ttp $(shell for i in $$(seq 0 $$(($(NODES)-1))); do echo "node_$$i"; done)

start-blockchain: setup
	@echo "Starting blockchain infrastructure..."
	@$(PYTHON) $(PROJECT_ROOT)/scripts/run_docker_with_nodes.py \
		$(STATE_ARG) \
		--clique-size $(CLIQUE_SIZE) \
		$(BUILD_ARG) \
		--generate-only
	@cd $(PROJECT_ROOT)/../thesis-blockchain/api-gateway && \
		docker compose up -d --build

start-monitoring: generate-dashboard
	@echo "Starting monitoring services..."
	@docker compose -f $(COMPOSE_FILE) up -d prometheus grafana
	@echo ""
	@echo "Grafana:    http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

start-storage:
	@echo "Starting storage services..."
	@docker compose -f $(COMPOSE_FILE) up -d --build ipfs-node-1 registry


# ------------------------------------------------------------------------------
# Stop targets
# ------------------------------------------------------------------------------

stop:
	@echo "Stopping all services..."
	@if [ -f "$(COMPOSE_FILE)" ]; then \
		docker compose -f $(COMPOSE_FILE) down -v; \
	fi
	@if [ -f "$(PROJECT_ROOT)/../thesis-blockchain/api-gateway/docker-compose.yaml" ]; then \
		cd $(PROJECT_ROOT)/../thesis-blockchain/api-gateway && docker compose down -v 2>/dev/null || true; \
	fi
	@echo "All services stopped"

stop-training:
	@echo "Stopping training services..."
	@docker rm -f ttp 2>/dev/null || true
	@for i in $$(seq 0 $$(($(NODES)-1))); do \
		docker rm -f "node_$$i" 2>/dev/null || true; \
	done


# ------------------------------------------------------------------------------
# Utility targets
# ------------------------------------------------------------------------------

logs:
	@if [ -f "$(COMPOSE_FILE)" ]; then \
		docker compose -f $(COMPOSE_FILE) logs -f; \
	else \
		echo "No compose file found. Run 'make start' first."; \
	fi

logs-node:
	@if [ -z "$(NODE)" ]; then \
		echo "Usage: make logs-node NODE=0"; \
	else \
		docker compose -f $(COMPOSE_FILE) logs -f node_$(NODE); \
	fi

clean-state:
	@echo "Clearing training state..."
	@rm -rf $(PROJECT_ROOT)/logs/* 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/checkpoints/* 2>/dev/null || true
	@rm -f $(PROJECT_ROOT)/config/topology.json 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/data/blockchain/* 2>/dev/null || true

clean: stop
	@echo "Cleaning generated files..."
	@rm -rf $(PROJECT_ROOT)/config/nodes/*.json 2>/dev/null || true
	@rm -f $(COMPOSE_FILE) 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/logs/* 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/checkpoints/* 2>/dev/null || true
	@rm -f $(PROJECT_ROOT)/config/topology.json 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/data/blockchain/* 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/data/ipfs/* 2>/dev/null || true
	@echo "Clean complete"

clean-all: clean
	@echo "Removing virtual environment and Docker resources..."
	@rm -rf $(VENV)
	@docker system prune -f 2>/dev/null || true
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
