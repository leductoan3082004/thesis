# Secure Aggregation for Federated Learning

Privacy-preserving federated learning using the secure aggregation protocol from Bonawitz et al. (CCS 2017). Multiple parties collaboratively train a machine learning model while keeping their data private — the server learns only the aggregate model, never individual updates.

## Features

- **4-Round Secure Aggregation Protocol** with key exchange, masking, and reconstruction
- **D-Cliques Topology** for scalable hierarchical aggregation
- **Dropout Tolerance** via threshold-based aggregation (survives up to n-t failures)
- **Non-IID Data** using Dirichlet partitioning for realistic heterogeneous settings
- **Blockchain Integration** with Hyperledger Fabric for trainer identity and model registry
- **IPFS Storage** for decentralized model distribution
- **Full Monitoring** with Prometheus metrics and Grafana dashboards
- **Centralized Logging** via Loki + Promtail with CLI querying
- **Process-Only Runtime** — all components run as managed host processes, no Docker required

## Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core runtime |
| Hyperledger Fabric binaries | 2.x | `cryptogen`, `configtxgen`, `fabric-ca-server`, `fabric-ca-client` |
| Node.js | 18+ | Blockchain gateway scripts |
| IPFS Kubo | Latest | `ipfs` binary for decentralized storage |
| 4GB+ free disk | | Datasets, models, blockchain state |

The monitoring tools (Loki, Promtail, Prometheus, Grafana) are installed automatically by `make setup`.

### Repository Layout

The blockchain repository must be cloned as a sibling directory:

```
parent/
├── secure_aggregation/       # this repo
└── thesis-blockchain/        # blockchain helper repo
```

```bash
# From the parent directory
git clone https://github.com/letienthanh364/thesis-blockchain.git
```

## End-to-End Setup Guide

### Step 1: Clone and Enter the Repository

```bash
git clone <this-repo-url> secure_aggregation
cd secure_aggregation
```

### Step 2: One-Command Setup

```bash
make setup
```

This runs six steps automatically:
1. Creates Python virtual environment (`.venv/`)
2. Installs Python dependencies (`pip install -e ".[mnist]"`, gRPC tools, PyYAML)
3. Generates gRPC protobuf code from `protos/secureagg.proto`
4. Downloads MNIST dataset to `data/MNIST/`
5. Prepares blockchain `.env` config
6. Installs monitoring tools (Loki, Promtail, Prometheus, Grafana) to `~/.local/bin/`

<details>
<summary>Manual setup (if you prefer not to use Make)</summary>

```bash
# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Python dependencies
pip install -e ".[mnist]"
pip install grpcio grpcio-tools PyYAML prometheus_client

# gRPC code generation
python -m grpc_tools.protoc -I=protos \
    --python_out=src/secure_aggregation/communication \
    --grpc_python_out=src/secure_aggregation/communication \
    protos/secureagg.proto

# MNIST dataset
python scripts/prepare_data.py

# Monitoring tools
bash scripts/install_monitoring.sh

# Blockchain .env
cp ../thesis-blockchain/api-gateway/.env.example ../thesis-blockchain/api-gateway/.env
```
</details>

### Step 3: Start the Full System

```bash
# Default: 6 nodes, clique size 3
make start

# Custom: 10 nodes, clique size 5
make start NODES=10 CLIQUE_SIZE=5

# With hierarchical node roster
make start NODES_MAP=config/nodes-map.json CLIQUE_SIZE=4
```

Or use the CLI directly:

```bash
.venv/bin/python scripts/secureagg_ctl.py start --nodes 10 --clique-size 5
```

The startup sequence:
1. IPFS daemons for decentralized model storage
2. Blockchain stack (Hyperledger Fabric orderer, peers, gateway)
3. Trainer identity registration
4. TTP service (Ed25519 key distribution)
5. N FL training nodes with Dirichlet-partitioned MNIST data
6. Monitoring stack (Loki, Promtail, Prometheus, Grafana)

You will see:
```
Starting IPFS processes...
Starting blockchain stack (orderer + peers + gateway)...
Waiting for gateway health at http://localhost:9000...
Registering trainers...
Starting TTP service...
Starting 10 FL nodes...
  [trainer-node-001] pid=12345 service=51000 metrics=61000
  [trainer-node-002] pid=12346 service=51001 metrics=61001
  ...
Starting monitoring stack...
  Loki      -> http://localhost:3100
  Promtail  -> http://localhost:9080
  Prometheus-> http://localhost:9090
  Grafana   -> http://localhost:3000 (admin/admin)
```

#### Skip Optional Components

```bash
# Skip monitoring (faster startup for development)
.venv/bin/python scripts/secureagg_ctl.py start --nodes 4 --skip-monitoring

# Skip blockchain (no model anchoring)
.venv/bin/python scripts/secureagg_ctl.py start --nodes 4 --skip-blockchain

# Skip IPFS (no decentralized storage)
.venv/bin/python scripts/secureagg_ctl.py start --nodes 4 --skip-ipfs
```

### Step 4: Monitor Training

#### Check Process Status

```bash
make status
# or
.venv/bin/python scripts/secureagg_ctl.py status
```

Output:
```
NAME                        PID STATUS     TYPE            PORTS
-------------------------------------------------------------------------
ipfs_0                    12340 running    infrastructure  15101,18180
blockchain                   -1 running    infrastructure  7050,7051,8051,9051,9000
ttp                       12350 running    ttp             50051
node_0                    12360 running    training        51000,52000,53000,61000
node_1                    12361 running    training        51001,52001,53001,61001
...
loki                      12380 running    monitoring      3100
promtail                  12381 running    monitoring      9080
prometheus                12382 running    monitoring      9090
grafana                   12383 running    monitoring      3000
```

#### View Logs via CLI

```bash
# All logs (uses Loki when available, falls back to files)
make logs

# Follow logs in real-time
.venv/bin/python scripts/secureagg_ctl.py logs --follow

# Filter by node
make logs-node NODE=trainer-node-001

# Filter by service (fl_node, ttp, ipfs)
.venv/bin/python scripts/secureagg_ctl.py logs --service ttp

# Error-level only
make logs-errors

# Advanced querying
.venv/bin/python scripts/secureagg_ctl.py logs --level ERROR --since 30m --json
.venv/bin/python scripts/secureagg_ctl.py logs --contains "accuracy" --limit 50
```

#### Grafana Dashboards

Open http://localhost:3000 (login: admin / admin).

Two dashboards are auto-provisioned on startup:

**Federated Learning - Cluster Metrics** (Prometheus)
- Overview: total clusters, nodes, global round, test accuracy, convergence status
- Accuracy over time per node and per clique
- Accuracy by round (train / validation / test)
- Delta norm and convergence streak
- Timing: local training, aggregation, round total
- Network: messages and bytes sent/received, SAP phase durations
- Topology: nodes per clique, model parameters, training samples
- Template variables: filter by clique and node

**Federated Learning - Logs** (Loki)
- All node logs with free-text filter
- Accuracy and convergence signals
- SAP protocol phase progression
- IPFS and blockchain activity
- Errors and warnings
- Per-node individual log panels (auto-generated based on node count)
- Infrastructure: TTP and IPFS logs
- Log volume and error rate over time

### Step 5: Stop the System

```bash
make stop
# or
.venv/bin/python scripts/secureagg_ctl.py stop
```

### Cleanup

```bash
# Kill stale processes and verify ports are free
.venv/bin/python scripts/secureagg_ctl.py cleanup

# Also purge logs and observability data
.venv/bin/python scripts/secureagg_ctl.py cleanup --purge-logs

# Full cleanup: stop processes, remove all generated files
make clean

# Full cleanup including virtual environment
make clean-all
```

## Expected Training Output

```
[trainer-node-001] Round 1/10
[trainer-node-001] Phase 1: Local training
[trainer-node-001] Accuracy before aggregation: 0.7465
[trainer-node-001] *** This node is the AGGREGATOR for round 0 ***
[trainer-node-001] Phase 2: Secure aggregation
[trainer-node-001] Round 0: Advertising keys
[trainer-node-001] Accepted by aggregator, received all 4 keys
[trainer-node-001] Round 2: Sending masked model
[trainer-node-001] Masked input accepted, 4 survivors
[trainer-node-001] Round 4: Sending unmask shares
[trainer-node-001] Round 4 complete: aggregation done
[trainer-node-001] Phase 3: Updating model with aggregated weights
[trainer-node-001] Accuracy after aggregation: 0.8234
[trainer-node-001] Improvement: +0.0769
```

**Accuracy progression (MNIST, 10 nodes):**
- Round 1: 74-78% (local models with non-IID data)
- Round 5: 85-88% (after collaboration)
- Round 10: ~91% (all nodes converge)

## Architecture

```
+-----------------------------------------------------+
|                   TTP Service                        |
|         (Ed25519 Key Distribution)                   |
+-----------------------+-----------------------------+
                        | Register & Get Keys
    +-------------------+-------------------+-------------------+
    |                   |                   |                   |
+---v----+        +-----v-----+       +-----v-----+       +-----v-----+
| Node 0 |        |  Node 1   |       |  Node 2   |       |  Node N   |
+--------+        +-----------+       +-----------+       +-----------+
     |                  |                   |                   |
     +------------------+-------------------+-------------------+
                    Secure Aggregation (D-Cliques)
                            |
          +-----------------+-----------------+
          |                                   |
     +----v-----+                       +-----v----+
     |  Clique 0 |                       | Clique 1 |
     | (intra-   |<--- inter-cluster --->| (intra-  |
     | aggregate)|     via IPFS +        | aggregate|
     +-----------+     Blockchain        +----------+
```

## Port Allocation

| Component | Port |
|---|---|
| TTP | 50051 |
| Node i service | 51000 + i |
| Node i aggregator | 52000 + i |
| Node i bridge | 53000 + i |
| Node i metrics | 61000 + i |
| Loki | 3100 |
| Promtail | 9080 |
| Prometheus | 9090 |
| Grafana | 3000 |
| Blockchain Gateway | 9000 |
| Blockchain Orderer | 7050 |
| Blockchain Peers | 7051, 8051, 9051 |

Override base ports: `--ttp-port`, `--base-node-port`, `--base-metrics-port`.

## Configuration

### Node Config Template

Node configs are generated from `config/node.config.template.json`. The orchestrator substitutes process-friendly values (localhost addresses, sequential ports). Key training parameters:

| Parameter | Default | Description |
|---|---|---|
| `training.rounds` | 3 | Federated training rounds |
| `training.local_epochs` | 1 | Local training epochs per round |
| `training.batch_size` | 64 | Mini-batch size |
| `dataset.alpha` | 0.5 | Dirichlet non-IID parameter (lower = more non-IID) |
| `secure_agg.threshold` | 3 | Minimum nodes for secure aggregation |

### System Config

Copy `config/system-config.sample.json` to `config/system-config.json` to configure:
- **Convergence detection**: warmup rounds, tolerance, patience
- **Fleet size**: `number_of_nodes` (used when `--nodes` is omitted)
- **Hierarchy levels**: state/nation scope identifiers, timer intervals, merge policies

### Hierarchy Rosters

`config/nodes-map.json` defines which trainers belong to each scope:

```json
{
  "nation": [{
    "nation_id": "nation_0",
    "states": [{
      "state_id": "state_alpha",
      "clusters": [{
        "cluster_id": "cluster_0",
        "nodes": ["trainer-node-001", "trainer-node-002", "trainer-node-003"]
      }]
    }]
  }]
}
```

### Dataset Configuration

```bash
.venv/bin/python scripts/prepare_data.py --list           # List available datasets
.venv/bin/python scripts/prepare_data.py --dataset mnist
.venv/bin/python scripts/prepare_data.py --dataset fashion_mnist
```

## Process Runtime Layout

All runtime state lives under `process-runtime/` (gitignored):

```
process-runtime/
├── registry.json               # Managed process tracking
├── datasets.json               # Process-mode dataset paths
├── topology.json               # D-Cliques topology
├── config/nodes/               # Per-node JSON configs
├── nodes/
│   └── node_<i>/
│       ├── data/               # Symlink to shared dataset
│       ├── logs/node.log       # Node stdout/stderr
│       ├── checkpoints/        # Model checkpoints
│       └── pids/               # PID files
├── logs/                       # TTP, IPFS, monitoring logs
├── pids/                       # Infrastructure PID files
└── observability/
    ├── loki.yml                # Loki config (auto-generated)
    ├── promtail.yml            # Promtail config (auto-generated)
    ├── prometheus.yml          # Prometheus config (auto-generated)
    └── grafana/
        ├── provisioning/       # Datasources + dashboard providers
        └── dashboards/         # Auto-provisioned dashboard JSON
```

## Make Targets

| Target | Description |
|---|---|
| `make setup` | Full setup: venv, deps, gRPC, data, blockchain, monitoring |
| `make start` | Start full system (runs setup first) |
| `make start NODES=10 CLIQUE_SIZE=5` | Start with custom topology |
| `make stop` | Stop all managed processes |
| `make status` | Show status of all processes |
| `make logs` | View aggregated logs |
| `make logs-node NODE=trainer-node-001` | View logs for a specific node |
| `make logs-errors` | View error-level logs only |
| `make clean` | Stop processes, remove generated files |
| `make clean-all` | Full cleanup including virtual environment |
| `make test` | Run unit tests |
| `make test-coverage` | Run tests with coverage |

## Troubleshooting

### Port Conflicts
```bash
.venv/bin/python scripts/secureagg_ctl.py cleanup
make status
```

### SSL Certificate Errors (MNIST Download)
```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
.venv/bin/python scripts/prepare_data.py
```

### gRPC Import Error
If `ModuleNotFoundError: No module named 'secureagg_pb2'`:
```bash
sed -i '' 's/^import secureagg_pb2/from . import secureagg_pb2/' \
    src/secure_aggregation/communication/secureagg_pb2_grpc.py
```

### Blockchain Setup Failures
```bash
ls ../thesis-blockchain/api-gateway/
which cryptogen configtxgen fabric-ca-server fabric-ca-client
```

### Nodes Not Progressing
```bash
make logs-errors
make stop && make start
```

### Grafana Shows "No Data"
Verify Prometheus targets are up:
```bash
curl -s 'http://localhost:9090/api/v1/targets' | python3 -c "
import sys, json
targets = json.load(sys.stdin)['data']['activeTargets']
for t in targets:
    print(f\"{t['scrapeUrl']:40s} {t['health']}  {t.get('lastError', '')}\")
"
```

If all targets show `down`, ensure `prometheus_client` is installed:
```bash
.venv/bin/pip install prometheus_client
```
Then restart the stack.

## Security Properties

1. **Privacy**: Server learns only aggregate model, never individual updates
2. **Authentication**: Ed25519 signatures prevent impersonation
3. **Consistency**: Round 3 signatures ensure all parties agree on participants
4. **Dropout Tolerance**: System continues if threshold nodes survive
5. **No Central Trust**: After TTP setup, no single party controls the system

## Performance

| Metric | Value |
|---|---|
| Per round | ~30-60 seconds (CPU) |
| Full training (10 rounds) | ~5-10 minutes |
| Final accuracy (MNIST) | ~91% |
| Communication | ~5MB per node per round |
| Tested scale | 4-10 nodes |

## Project Structure

```
secure_aggregation/
├── Makefile                          # Build and run commands
├── scripts/
│   ├── secureagg_ctl.py              # Unified CLI orchestrator
│   ├── install_monitoring.sh         # Monitoring tools installer
│   ├── runtime/
│   │   ├── port_allocator.py         # Deterministic port allocation
│   │   ├── process_registry.py       # PID tracking and lifecycle
│   │   ├── config_generator.py       # Per-node config generation
│   │   ├── ipfs_manager.py           # IPFS daemon management
│   │   ├── blockchain_helpers.py     # Blockchain artifact prep
│   │   ├── observability.py          # Loki/Promtail/Prometheus/Grafana
│   │   └── loki_client.py            # Loki HTTP API client
│   ├── run_ttp_with_topology.py      # TTP service runner
│   └── prepare_data.py               # Download datasets
├── src/secure_aggregation/
│   ├── communication/     # gRPC services (node, aggregator, TTP)
│   ├── protocol/          # Secure aggregation protocol (core.py)
│   ├── crypto/            # Primitives (DH, signatures, AEAD, PRG, Shamir)
│   ├── data/              # Dirichlet partitioning
│   ├── node/              # Node engine
│   ├── training/          # MNIST training flow
│   ├── topology/          # D-Cliques topology
│   └── config/            # Configuration models
├── config/
│   ├── datasets.json               # Dataset configurations
│   ├── node.config.template.json   # Node config template
│   ├── ipfs-process.json           # IPFS daemon config
│   └── system-config.sample.json   # System config sample
├── docker/grafana/dashboards/      # Grafana dashboard templates
├── protos/                         # gRPC protocol definitions
└── tests/                          # Unit and integration tests
```

## References

> Bonawitz, Keith, et al. "Practical secure aggregation for privacy-preserving machine learning." ACM CCS 2017.

## License

See [LICENSE](LICENSE) file for details.
