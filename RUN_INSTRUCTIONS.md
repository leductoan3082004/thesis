# Running Secure Aggregation Federated Learning

Complete guide to running end-to-end federated learning with secure aggregation on MNIST.

## Architecture

- **TTP Service**: Distributes Ed25519 signing keys to all nodes
- **4 Nodes**: Train on Dirichlet-partitioned MNIST, perform secure aggregation
- **Aggregator Election**: Round-robin election (node_0 → node_1 → node_2 → node_3 → ...)
- **4-Round Protocol**: Advertise Keys → Share Keys → Masked Input → Consistency Check → Unmask
- **Model Synchronization**: All nodes update with aggregated weights after each round

## Quick Start (Recommended)

Before running any scripts, clone the blockchain helper repo so automation can locate shared assets. The expected layout keeps this repo under `full-system/system` with the blockchain repo alongside it as `full-system/thesis-blockchain`:

```bash
# From the full-system directory that already contains the system/ folder
git clone https://github.com/letienthanh364/thesis-blockchain.git
```

As a result, your tree should look like:
```
full-system/
├── system/            # this repo
└── thesis-blockchain/ # cloned helper repo
```
Keep both directories next to each other under `full-system/` so automation in `system/` can reference blockchain artifacts without extra configuration.

### Step 1: Install Dependencies

Using a Python virtual environment keeps Homebrew’s Python clean and avoids the “externally-managed-environment” error:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

```bash
# Install Python dependencies
pip install -e ".[mnist]"
pip install grpcio grpcio-tools

# Generate gRPC code
python -m grpc_tools.protoc -I=protos --python_out=src/secure_aggregation/communication --grpc_python_out=src/secure_aggregation/communication protos/secureagg.proto
```

### Step 2: Prepare MNIST Data

```bash
# Download MNIST dataset (needed before Docker run)
python scripts/prepare_data.py
```

This downloads MNIST to the `data/` directory which is mounted in Docker containers.

### Step 3: Run with Docker Compose

```bash
# End-to-end automation: regenerates Fabric crypto, creates node configs,
# generates blockchain identities, starts the Fabric stack, bulk-registers
# trainers, and launches federated nodes. Omit --nodes to read
# config/system-config.json:number_of_nodes.
AUTH_JWT_SECRET="super-secret" python scripts/run_docker_with_nodes.py --nodes 10

# View logs from all federated containers
cd docker
docker compose -f docker-compose.auto.yml logs -f

# View logs from specific node
docker compose -f docker-compose.auto.yml logs -f node_0

# Stop services
docker compose -f docker-compose.auto.yml down -v
```

## What You'll See

### Phase 1: Initialization
```
✓ TTP server starts on port 50051
✓ Each node registers and receives signing keys
✓ All nodes wait for full participant list (4 nodes)
```

### Phase 2: Training Loop (10 rounds)
For each round:
```
1. Local Training
   - Each node trains on its MNIST partition for 2 epochs
   - Accuracy evaluated before aggregation

2. Aggregator Election
   - Round 0: node_0 is aggregator
   - Round 1: node_1 is aggregator
   - ... (round-robin)

3. Secure Aggregation (4 rounds)
   - Round 0: Nodes advertise DH public keys
   - Round 1: Nodes share encrypted secrets
   - Round 2: Nodes send masked model updates
   - Round 3: Consistency check with signatures
   - Round 4: Unmask and compute aggregate

4. Model Update
   - All nodes fetch aggregated model
   - Accuracy evaluated after aggregation
   - Improvement logged

5. Next Round
   - Aggregator rotates to next node
   - Process repeats
```

### Expected Output (Per Node)

```
[node_0] Round 1/10
[node_0] Phase 1: Local training
[node_0] Local training completed for 2 epochs
[node_0] Accuracy before aggregation: 0.4523
[node_0] *** This node is the AGGREGATOR for round 0 ***
[node_0] Phase 2: Secure aggregation
[node_0] Round 0: Advertising keys
[node_0] Round 0 complete: received 4 participants
[node_0] Round 1: Sharing keys
[node_0] Round 2: Sending masked model
[node_0] Round 2 complete: 4 survivors
[node_0] Round 3: Consistency check
[node_0] Round 4: Unmasking
[node_0] Round 4 complete: aggregation done
[node_0] Phase 3: Updating model with aggregated weights
[node_0] Accuracy after aggregation: 0.6891
[node_0] Improvement: +0.2368
```

## Configuration

Node configs live under `config/nodes/` and are generated automatically from `config/node.config.template.json` every time you run `scripts/run_docker_with_nodes.py`. Update the template to change defaults before launching, or tweak individual node files after generation if specific overrides are needed. The helper rotates IPFS endpoints and blockchain identities (`trainer-node-XXX`) automatically so you only need to supply the template once, and it reads the target fleet size from `number_of_nodes` in `config/system-config.json` whenever `--nodes` is omitted.

### Dataset Partitioning (Dirichlet)
- `alpha=0.5`: Moderate non-IID (realistic federated setting)
- Lower alpha = more non-IID, higher alpha = more IID
- Edit in [config/nodes/node_X.json](config/nodes/node_0.json)

### Training Parameters
- `num_rounds`: 10 federated rounds
- `local_epochs`: 2 epochs per round
- `batch_size`: 64
- `threshold`: 3 (minimum nodes for secure aggregation)

### Convergence Warmup (system-wide)
- `warmup_rounds` inside `config/system-config.json` controls how many rounds each node waits before emitting convergence signals (default `5` in the sample file).
- Lower it to `0` to start convergence checks immediately or raise it to defer signals; this replaces the deprecated `CONVERGENCE_WARMUP_ROUNDS` environment override.
- This is distinct from `MAX_TRAINING_ROUNDS`, which caps the total number of federated rounds.
- Set `number_of_nodes` in the same file once so Docker launches know how many node configs/services to generate when you omit `--nodes`.

### Global Convergence Settings
- Copy `config/system-config.sample.json` to `config/system-config.json` and edit it to change `enabled`, `tol_abs`, `tol_rel`, or `patience` without touching every node file. The resolved file is gitignored so you can keep environment-specific thresholds private.
- Nodes automatically consume this file; override the location with `SYSTEM_CONFIG_PATH=/path/to/system-config.json` when launching services if you need a custom mount.

### Aggregator Rotation
Automatic round-robin election:
- Round 0 → node_0
- Round 1 → node_1
- Round 2 → node_2
- Round 3 → node_3
- Round 4 → node_0 (cycles)

## Local Testing (Without Docker)

### Terminal 1: TTP
```bash
python -m secure_aggregation.communication.ttp_service
```

### Terminal 2-5: Nodes
```bash
# Prepare data first
python scripts/prepare_data.py

# Start nodes
python -m secure_aggregation.communication.node_service --config config/nodes/node_0.json
python -m secure_aggregation.communication.node_service --config config/nodes/node_1.json
python -m secure_aggregation.communication.node_service --config config/nodes/node_2.json
python -m secure_aggregation.communication.node_service --config config/nodes/node_3.json
```

## Implementation Details

### Secure Aggregation Protocol
- **Round 0**: Generate DH keypairs (c, s), sign and advertise
- **Round 1**: Shamir-share secrets, AEAD-encrypt per-peer
- **Round 2**: Apply pairwise+self masks, send quantized model
- **Round 3**: Sign survivor list for consistency
- **Round 4**: Send unmask shares, reconstruct aggregate

### Model Synchronization
- Models quantized to integers (scale=1e6) before masking
- Aggregator computes average of masked models
- All nodes dequantize and load aggregated weights
- Process repeats for convergence

### Security Properties
- Server learns only aggregate (privacy)
- Dropout tolerance (up to n-t nodes can fail)
- Signature verification (authenticity)
- No single point of trust after TTP setup

## Troubleshooting

### SSL Certificate Errors
Run `python scripts/prepare_data.py` to download MNIST with SSL workaround.

### Port Conflicts
Change ports in node configs if 50051-50055 are in use.

### Docker Issues
```bash
# Clean everything
docker compose down -v
docker system prune -af

# Rebuild
docker compose build --no-cache
docker compose up
```

### Logs
```bash
# All logs
docker compose logs -f

# Specific service
docker compose logs -f node_0
docker compose logs -f ttp

# Save logs
docker compose logs > training.log
```

### Node Hangs at "Waiting for all nodes"
Ensure all 4 nodes are starting. Check:
```bash
docker compose ps
```

## Performance

- **Per Round**: ~30-60 seconds (depends on hardware)
- **Full Training (10 rounds)**: ~5-10 minutes
- **Accuracy**: Should reach ~85-90% after 10 rounds
- **Communication**: ~10MB per round per node (model size + overhead)

## Next Steps

1. **Tune hyperparameters**: Adjust alpha, epochs, learning rate
2. **Add convergence detection**: Stop when target accuracy reached
3. **Scale to more nodes**: Add node_4, node_5, etc.
4. **Try different datasets**: CIFAR-10, Fashion-MNIST
5. **Enable topology**: Implement D-cliques from context_codex.md

## Files Structure

```
secure_aggregation/
├── config/nodes/          # Node configurations
├── data/                  # MNIST dataset (gitignored)
├── docker/                # Docker Compose + Dockerfiles
├── logs/                  # Training logs (gitignored)
├── protos/                # gRPC protocol definitions
├── scripts/               # Helper scripts
└── src/secure_aggregation/
    ├── communication/     # TTP, Aggregator, Node services
    ├── crypto/            # Cryptographic primitives
    ├── data/              # Data partitioning
    ├── protocol/          # Secure aggregation protocol
    └── ...
```
