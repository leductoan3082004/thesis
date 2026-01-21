# Secure Aggregation for Federated Learning

A complete implementation of privacy-preserving federated learning using the secure aggregation protocol from Bonawitz et al. (CCS 2017). This system enables multiple parties to collaboratively train a machine learning model while keeping their data private. The server learns only the aggregate model, never individual updates.

## Features

- **4-Round Secure Aggregation Protocol**: Fully implemented with key exchange, masking, and reconstruction
- **Automatic Coordination**: Nodes self-organize without manual intervention
- **Dropout Tolerance**: Threshold-based aggregation (survives up to n-t failures)
- **Aggregator Rotation**: Round-robin election distributes load across nodes
- **Non-IID Data**: Dirichlet partitioning simulates realistic heterogeneous federated settings
- **Blockchain Integration**: Hyperledger Fabric for trainer identity and model registry
- **Docker Deployment**: One-command launch of entire federated network
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **gRPC Communication**: Efficient, type-safe distributed protocol with 200MB message budgets (override via `GRPC_MAX_MESSAGE_MB`) so large model updates flow without truncation
- **MNIST Demonstration**: Complete end-to-end training example

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Hyperledger Fabric binaries (cryptogen, configtxgen, fabric-ca-client)
- Node.js 18+ (for blockchain scripts)
- 4GB+ free disk space

The blockchain repository (`thesis-blockchain`) must be cloned as a sibling directory.

## Quick Start

### Using Makefile (Recommended)

```bash
# First time setup: install dependencies, generate gRPC code, download MNIST
make setup

# Start the full system (blockchain + monitoring + training nodes)
make start

# Start with custom number of nodes
make start NODES=10

# Start in background (detached mode)
make start DETACH=1

# View logs
make logs

# Stop all services
make stop
```

### Available Make Targets

| Target | Description |
|--------|-------------|
| `make setup` | Install dependencies, generate gRPC code, download MNIST |
| `make start` | Start full system (blockchain + monitoring + training) |
| `make start NODES=N` | Start with N training nodes (default: 6) |
| `make start DETACH=1` | Start in background mode |
| `make start-training` | Restart only training nodes (keeps infrastructure running) |
| `make start-blockchain` | Start only blockchain infrastructure |
| `make start-monitoring` | Start only Prometheus and Grafana |
| `make stop` | Stop all services |
| `make stop-training` | Stop only training nodes |
| `make logs` | View logs from all containers |
| `make logs-node NODE=0` | View logs from specific node |
| `make clean` | Remove generated files and stop containers |
| `make clean-all` | Full cleanup including virtual environment |
| `make test` | Run unit tests |

### Manual Setup

If you prefer not to use the Makefile:

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[mnist]"
pip install grpcio grpcio-tools PyYAML

# 2. Generate gRPC protobuf code
python -m grpc_tools.protoc -I=protos \
    --python_out=src/secure_aggregation/communication \
    --grpc_python_out=src/secure_aggregation/communication \
    protos/secureagg.proto

# 3. Download MNIST dataset
python scripts/prepare_data.py

# 4. Run with Docker Compose
python scripts/run_docker_with_nodes.py --nodes 6
```

## System Behavior

The system automatically:
1. Starts blockchain infrastructure (Hyperledger Fabric network)
2. Registers trainer identities with verifiable credentials
3. Starts IPFS for decentralized model storage
4. Launches a TTP (Trusted Third Party) for key distribution
5. Spawns N federated nodes with partitioned MNIST data
6. Runs federated training with secure aggregation
7. Logs accuracy improvements after each round

## Expected Output

```
[node_0] Round 1/10
[node_0] Phase 1: Local training
[node_0] Accuracy before aggregation: 0.7465
[node_0] *** This node is the AGGREGATOR for round 0 ***
[node_0] Phase 2: Secure aggregation
[node_0] Round 0: Advertising keys
[node_0] Accepted by aggregator, received all 4 keys
[node_0] Round 2: Sending masked model
[node_0] Masked input accepted, 4 survivors
[node_0] Round 4: Sending unmask shares
[node_0] Round 4 complete: aggregation done
[node_0] Phase 3: Updating model with aggregated weights
[node_0] Accuracy after aggregation: 0.8234
[node_0] Improvement: +0.0769
```

**Accuracy progression (verified):**
- Round 1: 74-78% (local models with non-IID data)
- Round 5: 85-88% (after collaboration)
- Round 10: 91.81% (all nodes converge to same accuracy)

## Architecture

```
+-----------------------------------------------------+
|                   TTP Service                       |
|         (Ed25519 Key Distribution)                  |
+-----------------------+-----------------------------+
                        | Register & Get Keys
    +-------------------+-------------------+-------------------+
    |                   |                   |                   |
+---v----+        +-----v-----+       +-----v-----+       +-----v-----+
| Node 0 |        |  Node 1   |       |  Node 2   |       |  Node 3   |
| 11.6K  |        |  11.7K    |       |  19.2K    |       |  17.4K    |
|samples |        |  samples  |       |  samples  |       |  samples  |
|(19.4%) |        |  (19.5%)  |       |  (32.0%)  |       |  (29.1%)  |
+--------+        +-----------+       +-----------+       +-----------+
     |                  |                   |                   |
     +------------------+-------------------+-------------------+
                    Secure Aggregation
               (Rounds 0 -> 2 -> 4, simplified)
                            |
                      +-----v------+
                      |   Global   |
                      |   Model    |
                      +------------+
```

**Data Distribution:**
- Non-IID partitioning using Dirichlet(alpha=0.5)
- Each sample assigned to exactly ONE node (no overlap)
- Nodes have different amounts and label distributions

**Per Training Round:**
1. **Local Training**: Each node trains on its partition (2 epochs)
2. **Aggregator Election**: Round-robin selection (node_0 -> node_1 -> ...)
3. **Secure Aggregation**: 3-round protocol (Rounds 0, 2, 4)
   - Round 0: Advertise ECDH + Ed25519 keys
   - Round 2: Send masked model (quantized weights + PRG masks)
   - Round 4: Unmask shares for dropped nodes
4. **Model Update**: All nodes receive aggregated weights from aggregator
5. **Evaluation**: Accuracy measured on global test set

## Documentation

- [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md): Detailed usage guide with troubleshooting
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md): Complete technical overview
- [TOPOLOGY_IMPLEMENTATION.md](TOPOLOGY_IMPLEMENTATION.md): D-Cliques topology design and implementation
- [INTER_CLUSTER_AGGREGATION.md](INTER_CLUSTER_AGGREGATION.md): Inter-cluster flow with IPFS/Blockchain

## Security Properties

1. **Privacy**: Server learns only aggregate model (individual updates remain private)
2. **Authentication**: Ed25519 signatures prevent impersonation
3. **Consistency**: Round 3 signatures ensure all parties agree on participants
4. **Dropout Tolerance**: System continues if threshold nodes survive
5. **No Central Trust**: After TTP setup, no single party controls the system

## Configuration

Node configs are generated into `config/nodes/node_X.json` by `scripts/run_docker_with_nodes.py` using the defaults in `config/node.config.template.json`. Update the template to change the baseline settings before launching.

Set the target fleet size via `number_of_nodes` in `config/system-config.json`. The script reads that value automatically when `--nodes` is omitted.

```json
{
  "dataset": {
    "alpha": 0.5,
    "num_clients": 4
  },
  "training": {
    "num_rounds": 10,
    "local_epochs": 2,
    "batch_size": 64
  },
  "secure_agg": {
    "threshold": 3,
    "scale": 1000000.0
  }
}
```

### System Config

Copy `config/system-config.sample.json` to `config/system-config.json` to configure:
- Convergence detection: `enabled`, `warmup_rounds`, `tol_abs`, `tol_rel`, `patience`
- Fleet size: `number_of_nodes` for Docker launches

### Dataset Configuration

Datasets are configured in `config/datasets.json`. To switch datasets:

1. Download the dataset:
```bash
# List available datasets
python scripts/prepare_data.py --list

# Download a specific dataset
python scripts/prepare_data.py --dataset mnist
python scripts/prepare_data.py --dataset fashion_mnist
python scripts/prepare_data.py --dataset cifar10
```

2. Update `config/node.config.template.json`:
```json
{
  "dataset": {
    "name": "fashion_mnist",
    "num_clients": 12,
    "alpha": 0.5,
    "seed": 42
  }
}
```

To add a custom dataset (e.g., from Kaggle), add an entry to `config/datasets.json`:

```json
{
  "my_kaggle_dataset": {
    "type": "csv",
    "train_path": "/app/data/kaggle/train.csv",
    "test_path": "/app/data/kaggle/test.csv",
    "num_classes": 10,
    "input_shape": [1, 784],
    "label_column": "label"
  }
}
```

Supported dataset types:
- `torchvision`: Built-in datasets (MNIST, FashionMNIST, CIFAR10, etc.)
- `csv`: CSV files with features and a label column

## Project Structure

```
secure_aggregation/
├── Makefile                 # Build and run commands
├── src/secure_aggregation/
│   ├── communication/       # gRPC services (node_service, aggregator_service, ttp_service)
│   ├── protocol/            # Secure aggregation protocol (core.py)
│   ├── crypto/              # Primitives (dh.py, sign.py, aead.py, prg.py, shamir.py)
│   ├── data/                # Dirichlet partitioning (partition.py)
│   ├── node/                # Node engine (engine.py)
│   ├── training/            # MNIST training flow (mnist_flow.py)
│   ├── topology/            # Topology utilities (graph.py)
│   └── config/              # Configuration models (models.py)
├── config/
│   ├── nodes/               # Generated node configurations
│   ├── keys/                # Trainer keys (generated)
│   ├── datasets.json        # Dataset configurations
│   └── node.config.template.json
├── docker/
│   ├── docker-compose.yml   # Base compose template
│   ├── docker-compose.auto.yml  # Generated compose file
│   └── node.Dockerfile
├── protos/                  # gRPC protocol definitions
├── scripts/
│   ├── run_docker_with_nodes.py  # Main orchestrator
│   ├── prepare_data.py           # Download datasets
│   ├── run_ttp_with_topology.py  # TTP service runner
│   ├── generate_grafana_dashboard.py  # Dashboard generator
│   └── generate_keys.py          # Key generation utility
└── tests/                   # Unit and integration tests
```

## Performance

- **Training Time**: 3-5 minutes for 10 rounds on CPU
- **Communication**: ~5MB per node per round (quantized weights)
- **Final Accuracy**: 91.81% on MNIST test set (verified)
- **Scalability**: Tested with 4-10 nodes

## Monitoring

```bash
# View all logs in real-time
make logs

# View specific node logs
make logs-node NODE=0

# Save all logs to file
docker compose -f docker/docker-compose.auto.yml logs > training.log

# Check container status
docker compose -f docker/docker-compose.auto.yml ps
```

Access the monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test suite
PYTHONPATH=src .venv/bin/python -m pytest tests/test_protocol.py -v
```

## Troubleshooting

### SSL Certificate Errors (MNIST Download)

The `prepare_data.py` script handles SSL issues automatically. If errors persist:
```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
python scripts/prepare_data.py
```

### gRPC Import Error

If you see `ModuleNotFoundError: No module named 'secureagg_pb2'`:
```bash
sed -i '' 's/^import secureagg_pb2/from . import secureagg_pb2/' \
    src/secure_aggregation/communication/secureagg_pb2_grpc.py
```

### gRPC Message Size Configuration

Aggregator RPC servers and clients now set `grpc.max_send_message_length` and `grpc.max_receive_message_length` to 200 MB so CIFAR-scale models fit inside SAP Round 2 payloads. Export `GRPC_MAX_MESSAGE_MB=<megabytes>` before starting services to customize this ceiling.

### Port Conflicts

If port 50051 is in use, edit `docker/docker-compose.yml` to change the TTP port mapping.

### Docker Build Issues

```bash
make clean-all
make start
```

### Nodes Stuck or Not Progressing

```bash
make logs
make stop
make start
```

### Blockchain Setup Failures

Ensure the `thesis-blockchain` repository is cloned as a sibling directory and Hyperledger Fabric binaries are installed:
```bash
ls ../thesis-blockchain/api-gateway/
which cryptogen configtxgen fabric-ca-client
```

## References

This implementation follows the paper:
> Bonawitz, Keith, et al. "Practical secure aggregation for privacy-preserving machine learning." ACM CCS 2017.

## License

See [LICENSE](LICENSE) file for details.

---

**Status**: Fully Functional | **Last Updated**: 2025-12-20 | **Version**: 1.1.0
