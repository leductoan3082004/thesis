# Secure Aggregation for Federated Learning

## Project Overview

Privacy-preserving federated learning implementation using the **Bonawitz et al. (CCS 2017)** secure aggregation protocol. Multiple parties collaboratively train ML models while keeping data private — the server learns only the aggregate model, never individual updates.

**Key capabilities:** Process-only runtime (no Docker required), hierarchical D-Cliques topology, Hyperledger Fabric blockchain integration, IPFS decentralized storage, full observability (Prometheus, Grafana, Loki).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Federated Learning System                │
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │ Node 1  │   │ Node 2  │   │ Node 3  │   │ Node N  │    │
│  │(Trainer)│   │(Trainer)│   │(Trainer)│   │(Trainer)│    │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘    │
│       │              │              │              │         │
│       └──────────────┴──────┬───────┴──────────────┘         │
│                             │                                │
│                    ┌────────v────────┐                       │
│                    │   Aggregator    │ (round-robin elected) │
│                    │ (4-Round SAP)   │                       │
│                    └────────┬────────┘                       │
│                             │                                │
│              ┌──────────────┴──────────────┐                 │
│              │                             │                 │
│     ┌────────v───────┐          ┌─────────v──────┐          │
│     │  IPFS Storage  │          │  Blockchain    │          │
│     │  (Model CIDs)  │          │  (Anchoring)   │          │
│     └────────────────┘          └────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

## Key Source Directories

| Directory | Purpose |
|-----------|---------|
| `src/secure_aggregation/protocol/` | 4-round SAP protocol (`core.py`) + inter-cluster merge (`inter_cluster.py`) |
| `src/secure_aggregation/crypto/` | AEAD (`aead.py`), ECDH (`dh.py`), Ed25519 (`sign.py`), PRG (`prg.py`), Shamir (`shamir.py`) |
| `src/secure_aggregation/communication/` | gRPC services: TTP (`ttp_service.py`), Aggregator (`aggregator_service.py`), Node (`node_service.py`), Bridge (`bridge_service.py`), InterClusterAggregator (`inter_cluster_aggregator.py`), HierarchyMixin (`hierarchy_mixin.py`) |
| `src/secure_aggregation/data/` | Config-driven dataset loading with torchvision + CSV support (`datasets.py`), simpler torchvision-only loader (`dataset.py`), Dirichlet partitioning (`partition.py`) |
| `src/secure_aggregation/training/` | MNIST training flow (`mnist_flow.py`) |
| `src/secure_aggregation/topology/` | D-Cliques construction with label-skew mitigation (`graph.py`) |
| `src/secure_aggregation/node/` | NodeEngine (`engine.py`) for reliability scoring/gossip, ECM buffering (`ecm_buffer.py`) |
| `src/secure_aggregation/storage/` | IPFS/Blockchain abstractions (`model_store.py`), FastAPI blockchain gateway with JWT auth (`blockchain_gateway.py`), HTTP model anchor registry (`registry_service.py`) |
| `src/secure_aggregation/convergence/` | Convergence tracker (`tracker.py`), central checker (`central_checker.py`), central metadata broadcast (`central_broadcast.py`) |
| `src/secure_aggregation/state/` | Hierarchical state aggregation (`aggregation.py`), scope config (`config.py`), nodes-map parser (`nodes_map.py`) |
| `src/secure_aggregation/config/` | ScenarioConfig, NodeConfig, NodeRole, Timeouts, MountConfig (`models.py`), system config loader (`system.py`) |
| `src/secure_aggregation/models/` | VectorModel (`vector.py`), ModelRegistry (`registry.py`) |
| `src/secure_aggregation/utils/` | Logging (`logging.py`), Prometheus metrics (`prometheus_metrics.py`), comm metrics (`comm_metrics.py`), retry logic (`retry.py`) |

## Training Flow (Per-Round)

1. **Local Training** — Each node trains on its Dirichlet-partitioned data (SGD, configurable epochs).
2. **Aggregator Election** — Round-robin selection within the clique.
3. **SAP Round 0** — Nodes generate & broadcast ECDH keypairs (c_keypair for encryption, s_keypair for masking), signed with Ed25519.
4. **SAP Round 1** — Nodes create Shamir shares of secrets, AEAD-encrypt per recipient, and distribute via aggregator mailbox.
5. **SAP Round 2** — Nodes compute pairwise + self masks, apply to quantized model vector (mod PRIME), submit masked input. Aggregator determines survivors.
6. **SAP Round 3** — Nodes sign the survivor list for consistency verification.
7. **SAP Round 4** — Nodes send unmask shares. Aggregator reconstructs masks via Lagrange interpolation, removes masks, computes mean.
8. **Model Update** — All nodes fetch the aggregated global model.
9. **ECM Forwarding** — Bridge nodes forward encrypted cluster models to neighbor clusters via IPFS.
10. **Inter-Cluster Merge** — Adaptive clipping + weighted averaging of neighbor models.
11. **Convergence Check** — Track accuracy delta, patience mechanism, global convergence signal.

## Hierarchical Aggregation

### Hierarchy Structure

```
Nation (scope_index=2, interval=300s)
  └── State (scope_index=1, interval=120s)
       └── Cluster (D-Clique, every round)
            └── Nodes (trainer/aggregator/hybrid)
```

Node assignment defined in `config/nodes-map.json`. Each node resolves its membership path: `node_id` -> `cluster_N` -> `state_X` -> `nation_Y`.

### Aggregation Levels

| Level | Trigger | Merge Method | Apply Policy | Key Files |
|-------|---------|-------------|--------------|-----------|
| Cluster | Every training round | 4-round SAP (secure avg) | Direct replacement | `protocol/core.py`, `communication/aggregator_service.py` |
| Inter-Cluster | After each SAP round | Adaptive clipping + weighted avg | Convex combination | `protocol/inter_cluster.py`, `communication/bridge_service.py` |
| State | Every `interval_seconds` (120s) | Mean of cluster models | `replace` or `interpolate` | `state/aggregation.py`, `communication/hierarchy_mixin.py` |
| Nation | Every `interval_seconds` (300s) | Mean of state models | `interpolate` (alpha=0.2) | `state/aggregation.py`, `communication/hierarchy_mixin.py` |

### ECM (Encrypted Cluster Model) Flow

```
Cluster SAP -> Publish model to IPFS -> Bridge nodes gossip ECMs to neighbors
    -> State aggregator collects child ECMs (with timeout)
    -> Fetches models from IPFS, verifies hashes
    -> Merges (mean), publishes state model to IPFS + blockchain
    -> Nation aggregator repeats for state-level models
    -> All nodes fetch upstream model, apply per policy
```

### Key Components

- **`HierarchyLevelConfig`** (`state/config.py`): Per-scope settings — interval, timeout, apply_policy, apply_alpha, fanout_count, max_aggregators.
- **`ScopeRuntime`** (`communication/hierarchy_mixin.py`): Mutable runtime state per scope — round queue, model cache, ECM buffer, aggregator instance.
- **`StateAggregator`** (`state/aggregation.py`): Builds snapshot of child ECMs, fetches from IPFS, merges, publishes.
- **`InterClusterAggregator`** (`communication/inter_cluster_aggregator.py`): Extends SAP with cross-cluster model merging — collects ECMs from bridge nodes, fetches/verifies neighbor models from IPFS, applies adaptive clipping + weighted merge, publishes merged model to IPFS and anchors on blockchain.
- **`ECMBuffer`** (`node/ecm_buffer.py`): Thread-safe buffer for incoming ECMs with freshness window (300s) and deduplication.
- **`BridgeServicer`** (`communication/bridge_service.py`): gRPC service for inter-clique ECM gossip. Channel prefixes: `"cluster_0"` (cluster), `"state::state_alpha"` (state), `"signal::"` (convergence).
- **`HierarchyMixin`** (`communication/hierarchy_mixin.py`): Main orchestrator (~1833 lines) — scope scheduling, round execution, model application, fanout selection.
- **`NodesMapMetadata`** (`state/nodes_map.py`): Parses hierarchical nodes-map.json — builds rosters, child_map, and per-node membership paths.
- **`CentralMetadata` / `CentralBroadcast`** (`convergence/central_broadcast.py`): Publishes/fetches central clique topology metadata and checker health via blockchain.

### Scope Scheduling

- After each cluster round, check all scope handlers against `interval_seconds`.
- When due, add `(scope_round, cluster_round)` to the scope's round queue.
- Elect leader via round-robin within scope candidates.
- Dispatch: bridge nodes (fanout_count) send child artifacts to elected aggregator.
- Execute: aggregator collects ECMs (up to `collection_timeout_seconds`), merges, publishes.
- Each hierarchy level can specify an `approach` field (e.g., `"ring_star"`) for the aggregation topology pattern.

### Model Application Policies

- **`replace`** (default for state): `model = upstream_model` — full replacement.
- **`interpolate`** (for nation): `model = alpha * upstream + (1 - alpha) * local` — gradual blend (e.g., 20% new, 80% existing).

### Convergence

- Each cluster sends convergence signal via ECM (`is_signal=True`, `cid` starts with `"signal::"`).
- `CentralChecker` (`convergence/central_checker.py`) aggregates signals from all clusters.
- When all clusters report converged -> publish global convergence to blockchain.
- All nodes fetch convergence status and stop training.

## Cryptographic Primitives

- **PRIME field:** `2^521 - 1` (for Shamir sharing and modular arithmetic)
- **ECDH:** P-256 (SECP256R1) for pairwise key agreement
- **AEAD:** AES-256-GCM (32-byte key, 96-bit nonce) for share encryption
- **Signing:** Ed25519 for message authentication and consistency checks
- **PRG:** AES-CTR based (HKDF-SHA256 key derivation) for deterministic mask generation
- **Shamir:** (t, n) threshold secret sharing over modular arithmetic

## Configuration

| File | Purpose |
|------|---------|
| `config/system-config.json` | Global: hierarchy levels, convergence thresholds, timeouts |
| `config/system-config.sample.json` | Sample system config for reference |
| `config/node.config.template.json` | Per-node: role, ports, dataset, training params, secure_agg, inter_cluster settings |
| `config/nodes-map.json` | Hierarchy roster: nation -> state -> cluster -> nodes |
| `config/nodes-map.sample.json` | Sample nodes map for reference |
| `config/datasets.json` | Available datasets: MNIST, Fashion-MNIST, CIFAR-10 (torchvision-based) |
| `config/topology.json` | Generated D-Cliques topology (output of topology builder, not hand-edited) |
| `config/scenario.sample.json` | Sample scenario with 3 nodes, label distributions, reliability scores |
| `config/scenario.single.json` | Single-node demo scenario |
| `config/ipfs-process.json` | IPFS Kubo process-mode config: 3 local nodes with ports 15101-15103 |

### ScenarioConfig Fields

`ScenarioConfig` (`config/models.py`) drives topology construction:
- `participants`: List of node IDs.
- `threshold`: Minimum survivors for SAP (t+1 of n).
- `clique_size`: Target clique size for D-Cliques.
- `inter_clique_edges`: Edge strategy — `ring`, `ring_extra`, `ring_star`, `fractal`, `small_world`, `fully_connected`.
- `topology_iterations`: Optimization iterations for D-Cliques builder.
- `small_world_c`: Parameter for small-world edge strategy.
- `node_labels`: Per-node label distributions (for label-skew mitigation).
- `reliability`: Per-node uptime, bandwidth, latency scores (for aggregator election).
- `service_hostnames`: Node-to-hostname mapping.
- `timeouts`: Per-phase SAP timeouts (advertise_keys, share_keys, masked_input, consistency, unmasking).
- `mounts`: Directory paths for config, data, logs, checkpoints.

### NodeConfig Fields

`NodeConfig` (`config/models.py`):
- `node_id`, `role` (trainer/aggregator/hybrid/ttp), `host`, `port`.
- `log_level`: Defaults to INFO.
- `tls_enabled`: Boolean, defaults to false.
- `timeouts`: Per-node SAP timeout overrides.

## Port Allocation

- TTP: `50051`
- Node i service: `51000 + i`
- Node i aggregator: `52000 + i` (service + 1000)
- Node i bridge: `53000 + i` (service + 2000)
- Node i metrics: `61000 + i`
- Grafana: `3000`, Loki: `3100`, Prometheus: `9090`
- Blockchain: `7050` (orderer), `7051/8051/9051` (peers), `9000` (gateway)

## CLI & Makefile

```bash
make setup              # One-time: venv, deps, gRPC codegen, datasets, blockchain, monitoring
make start NODES=10 CLIQUE_SIZE=5          # Launch full system
make start NODES_MAP=config/nodes-map.json CLIQUE_SIZE=4  # Launch with hierarchy roster
make stop               # Graceful shutdown
make status             # Process health check
make logs               # Aggregated logs (Loki or file fallback)
make logs-node NODE=trainer-node-001       # Logs for specific node
make logs-errors        # Error-level logs only
make test               # Run unit tests
make test-coverage      # Tests with coverage report
make clean              # Stop processes, remove generated files (keeps venv)
make clean-all          # Full cleanup including venv
make install-ipfs       # Install IPFS Kubo
make install-fabric     # Install Hyperledger Fabric CLI binaries
make build-vctool       # Build vctool from blockchain repo
```

**Direct launch:**
```bash
python scripts/secureagg_ctl.py start --nodes 10 --clique-size 5
python scripts/run_ttp_with_topology.py --topology config/topology.json --port 50051
python -m secure_aggregation.communication.node_service --config config/node_0.json
```

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/secureagg_ctl.py` | Main CLI: start/stop/status/logs/cleanup for the full system |
| `scripts/run_ttp_with_topology.py` | Launch TTP server with topology file |
| `scripts/run_mnist_secure_agg.py` | Run MNIST secure aggregation training |
| `scripts/prepare_data.py` | Download and prepare MNIST dataset |
| `scripts/generate_keys.py` | Generate Ed25519 key pairs for nodes |
| `scripts/generate_grafana_dashboard.py` | Auto-generate Grafana dashboard JSON |
| `scripts/nodes_map_count.py` | Count nodes from a nodes-map JSON file |
| `scripts/install_ipfs.sh` | Install IPFS Kubo binary |
| `scripts/install_fabric_binaries.sh` | Install Hyperledger Fabric CLI |
| `scripts/install_monitoring.sh` | Install Loki, Promtail, Prometheus, Grafana |
| `scripts/build_vctool.sh` | Build vctool from thesis-blockchain repo |

## Dependencies

Requires Python >= 3.10 (`pyproject.toml`). Package name: `secure-aggregation-fl`.

- **Core:** cryptography>=42.0,<44.0, fastapi>=0.110, uvicorn>=0.23, httpx>=0.27, pyyaml>=6.0, protobuf>=4.25, prometheus_client>=0.19, numpy>=1.23
- **ML (optional `mnist` extra):** torch>=2.2, torchvision>=0.17, pandas>=2.0
- **Test (optional `test` extra):** pytest>=7.4, pyyaml>=6.0
- **Infrastructure:** gRPC (grpcio, grpcio-tools), IPFS Kubo, Hyperledger Fabric, Loki/Promtail/Prometheus/Grafana

## Testing

16 test modules across `tests/`: crypto, protocol, data, models, node, communication, storage, topology, integration, hierarchy, runtime, config, docker, performance, utils, plus root-level convergence and system config tests.

```bash
make test                    # All tests
make test-coverage           # Tests with coverage
pytest tests/protocol/       # Protocol-specific tests
pytest tests/integration/    # Full system integration tests
pytest tests/runtime/        # Runtime infrastructure tests (port allocator, IPFS, blockchain, Loki, config generator)
```

pytest config: `testpaths = ["tests"]`, `pythonpath = ["src", "."]`.

## Development Conventions

- Python >= 3.10, package at `src/secure_aggregation/`
- gRPC definitions in `protos/secureagg.proto`, generated code in `communication/secureagg_pb2.py` and `secureagg_pb2_grpc.py`
- Config-driven: all behavior controlled via JSON config files
- Process-only runtime: all components run as managed host processes (no Docker required)
- Quantization scale factor: `1e6` (float -> int conversion for secure aggregation)
- Storage layer uses abstract interfaces (`IPFSInterface`, `BlockchainInterface`) with mock and real implementations (`KuboIPFS`, `RegistryBlockchain`)
- `AnchorScope` enum (`CLUSTER`, `STATE`, `CONTROL`) namespaces blockchain records
- Two FastAPI services: `registry_service.py` (port 8000, model anchor HTTP registry) and `blockchain_gateway.py` (port 9000, JWT-authenticated gateway simulating Hyperledger Fabric)
- IPFS process-mode: 3 local Kubo nodes with API ports 15101-15103, swarm ports 14101-14103, gateway ports 18101-18103
- `data/` module has two dataset loaders: `datasets.py` (full-featured with CSV support, normalization) and `dataset.py` (simpler torchvision-only, used by node/TTP services)

