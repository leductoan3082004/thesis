# Secure Aggregation for Federated Learning — Agent Guide

## Project Overview

Privacy-preserving federated learning implementation using the **Bonawitz et al. (CCS 2017)** secure aggregation protocol. Multiple parties collaboratively train ML models while keeping data private — the server learns only the aggregate model, never individual updates.

**Key capabilities:** Process-only runtime (no Docker required), hierarchical D-Cliques topology, Hyperledger Fabric blockchain integration, IPFS decentralized storage, full observability (Prometheus, Grafana, Loki).

---

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

---

## Hierarchical Aggregation (Detailed)

### Hierarchy Structure

```
Nation (scope_index=2, interval=300s, policy=interpolate alpha=0.2)
  └── State (scope_index=1, interval=120s, policy=replace)
       └── Cluster (D-Clique, every training round, 4-round SAP)
            └── Nodes (trainer/aggregator/hybrid, 5-10 per clique)
```

Node assignment defined in `config/nodes-map.json`. Each node resolves: `node_id` -> `cluster_N` -> `state_X` -> `nation_Y`.

### Nodes Map Structure (`config/nodes-map.json`)

```json
{
  "nation": [{
    "nation_id": "nation_0",
    "states": [{
      "state_id": "state_alpha",
      "nodes": ["trainer-node-001", ..., "trainer-node-010"]
    }, {
      "state_id": "state_beta",
      "nodes": ["trainer-node-011", ..., "trainer-node-020"]
    }]
  }]
}
```

Parsed by `NodesMapMetadata` (`state/nodes_map.py`) into:
- `rosters`: `{scope_name: {scope_id: [node_list]}}` — who belongs where.
- `child_map`: `{(parent_scope, parent_id): {child_scope: [child_ids]}}` — parent-child relationships.
- `memberships`: `{node_id: {scope_name: scope_id}}` — each node's full membership path.

### Aggregation Levels

| Level | Trigger | Merge Method | Apply Policy | Key Files |
|-------|---------|-------------|--------------|-----------|
| Cluster | Every training round | 4-round SAP (secure avg) | Direct replacement | `protocol/core.py`, `communication/aggregator_service.py` |
| Inter-Cluster | After each SAP round | Adaptive clipping + weighted avg | Convex combination | `protocol/inter_cluster.py`, `communication/bridge_service.py` |
| State | Every `interval_seconds` (default 120s) | Mean of cluster models | `replace` or `interpolate` | `state/aggregation.py`, `communication/hierarchy_mixin.py` |
| Nation | Every `interval_seconds` (default 300s) | Mean of state models | `interpolate` (alpha=0.2) | `state/aggregation.py`, `communication/hierarchy_mixin.py` |

### HierarchyLevelConfig (`state/config.py`)

Each scope level has independent configuration:

```python
@dataclass
class HierarchyLevelConfig:
    enabled: bool                       # Level participates in aggregation?
    scope_index: int                    # Order (1=state, 2=nation, etc.)
    scope_name: str                     # "state", "nation"
    scope_id: str                       # "state_0", "nation_0"
    interval_seconds: float             # Wall-clock interval between scope rounds
    wait_seconds: float                 # Delay before applying upstream model
    collection_timeout_seconds: float   # Max wait for child ECMs (default 15s)
    commit_timeout_seconds: float       # Per-candidate blockchain commit deadline (10s)
    apply_policy: str                   # "replace" or "interpolate"
    apply_alpha: float                  # Blend factor for interpolation (0-0.49)
    apply_layer_mask: list[str]        # Selective layer application
    approach: str                       # "ring_star" or "custom"
    max_aggregators: Optional[int]      # Concurrent aggregators allowed
    fanout_count: Optional[int]         # Bridge nodes used per round
```

### System Config Example (`config/system-config.json`)

```json
{
  "hierarchy_defaults": {
    "collection_timeout_seconds": 30.0,
    "wait_seconds": 10
  },
  "hierarchy_levels": [
    {
      "scope_index": 1, "scope_name": "state", "scope_id": "state_0",
      "enabled": true, "interval_seconds": 120, "wait_seconds": 20,
      "max_aggregators": 2, "fanout_count": 2, "apply_policy": "replace"
    },
    {
      "scope_index": 2, "scope_name": "nation", "scope_id": "nation_0",
      "enabled": true, "interval_seconds": 300, "wait_seconds": 30,
      "apply_policy": "interpolate", "apply_alpha": 0.2
    }
  ]
}
```

### ECM (Encrypted Cluster Model) Data Structure

```python
@dataclass
class ECM:
    cid: str                      # IPFS content identifier
    hash: str                     # SHA256 hash for verification
    source_cluster: Optional[str] # "cluster_0", "cluster_1", etc.
    cluster_converged: bool       # Convergence signal flag
    cluster_delta_norm: float     # Model update magnitude
    round_idx: int               # Which training round
    is_signal: bool              # True = convergence only, False = real model
```

### ECM Buffer (`node/ecm_buffer.py`)

Thread-safe buffer for incoming ECMs:

- `freshness_window=300.0` — keep ECMs for 5 minutes.
- `_buffer: Dict[str, ECM]` — deduplicated by CID.
- `get_fresh_ecms()` — return ECMs within freshness window.
- `pop_signal_ecms()` — extract convergence signals (non-destructive for model ECMs).

### Bridge Service (`communication/bridge_service.py`)

gRPC service for inter-clique ECM gossip:

- `BridgeServicer.ReceiveECM()` — receives ECMs, stores in buffer.
- `BridgeClient.broadcast_ecm()` — sends ECM to all neighbor bridge nodes.
- **Channel prefixes**: `"cluster_0"` (cluster-level), `"state::state_alpha"` (state-level), `"signal::"` (convergence signal).

### Complete ECM Data Flow

```
┌─────────────────────────────────────────────────────────┐
│         CLUSTER TRAINING (Every training round)         │
│  Each clique: run SAP independently                     │
│  Output: cluster model on IPFS + blockchain anchor      │
└─────────────────────┬───────────────────────────────────┘
                      v
┌─────────────────────────────────────────────────────────┐
│           ECM GOSSIP (Bridge nodes, ongoing)            │
│  Bridge nodes receive CID from SAP aggregator           │
│  Create ECM {cid, hash, source_cluster}                │
│  Send to neighbor clusters via BridgeClient             │
│  Neighbor bridges store in ECMBuffer                    │
└─────────────────────┬───────────────────────────────────┘
                      v
┌─────────────────────────────────────────────────────────┐
│        STATE AGGREGATION (Every 120 seconds)            │
│  Elected aggregator (round-robin within state):         │
│    1. Collect ECMs from buffer (all child clusters)     │
│    2. Fetch models from IPFS + verify hashes            │
│    3. Merge: state_model = mean(cluster_models)        │
│    4. Publish to IPFS + anchor on blockchain           │
│    5. Broadcast state ECM to other state nodes          │
└─────────────────────┬───────────────────────────────────┘
                      v
┌─────────────────────────────────────────────────────────┐
│       NATION AGGREGATION (Every 300 seconds)            │
│  Elected aggregator (round-robin within nation):        │
│    1. Collect state ECMs from all states               │
│    2. Fetch state models from IPFS                     │
│    3. Merge: nation_model = mean(state_models)         │
│    4. Publish to IPFS + anchor on blockchain           │
└─────────────────────┬───────────────────────────────────┘
                      v
┌─────────────────────────────────────────────────────────┐
│         DOWNSTREAM MODEL APPLICATION                    │
│  All nodes:                                              │
│    1. Query blockchain for latest upstream model        │
│    2. Apply per policy (replace or interpolate)         │
│    3. Resume training with updated baseline             │
└─────────────────────────────────────────────────────────┘
```

### HierarchyMixin — Main Orchestrator (`communication/hierarchy_mixin.py`)

~1833 lines. Manages all hierarchy logic:

**ScopeRuntime** — mutable runtime state per scope:
```python
@dataclass
class ScopeRuntime:
    scope_name: str                      # "state", "nation"
    config: HierarchyLevelConfig
    scope_id: Optional[str]              # This node's scope ID
    round_queue: Deque[(int, int)]       # (scope_round, source_round)
    round_cache: Dict[int, np.ndarray]   # Merged models per round
    committed_rounds: Set[int]            # Rounds published to blockchain
    candidates: List[str]                # All nodes in this scope
    is_candidate: bool                   # Can this node aggregate?
    ecm_buffer: Optional[ECMBuffer]      # Incoming child ECMs
    aggregator: Optional[StateAggregator]# Merge logic
    last_model_cid: Optional[str]        # Latest published CID
```

**Scope Scheduling:**
1. After each cluster round, check all scope handlers against `interval_seconds`.
2. When due, add `(scope_round, cluster_round)` to the scope's round queue.
3. Elect leader via round-robin: `members[scope_round % len(members)]`.
4. Dispatch: fanout bridge nodes send child artifacts to elected aggregator.
5. Execute: aggregator collects ECMs (up to `collection_timeout_seconds`), merges, publishes.

**Model Application Policies:**
- **`replace`** (default for state): `model = upstream_model` — full replacement.
- **`interpolate`** (default for nation): `model = alpha * upstream + (1 - alpha) * local` — gradual blend.

### StateAggregator (`state/aggregation.py`)

```python
class StateAggregator:
    build_snapshot(ecms, required_clusters, target_round)
        # Deduplicate per cluster, filter by round, find missing clusters
    fetch_models(snapshot)
        # Fetch from IPFS, verify hash for each cluster model
    merge_models(models)
        # Simple average: np.mean(stacked, axis=0)
    publish_state_model(state_model, state_round)
        # Publish to IPFS + anchor on blockchain with scope=STATE
```

### Convergence in Hierarchy

- Each cluster sends convergence signal via ECM (`is_signal=True`, `cid` starts with `"signal::"`).
- `CentralChecker` (`convergence/central_checker.py`) aggregates signals from all clusters.
- When all clusters report converged -> publish global convergence to blockchain.
- `CentralMetadata` (`convergence/central_broadcast.py`) tracks central clique, checker candidates, cluster IDs.
- All nodes query blockchain for convergence status and stop training.

### Edge Cases

- **Missing child artifacts**: Aggregator continues with available models after `collection_timeout_seconds`.
- **Disabled scope**: `enabled=False` or `interval_seconds=0` -> scope round skipped, but downstream models still applied if ancestor scope enabled.
- **Fanout selection**: `fanout_count` bridge nodes selected per round via round-robin within scope.

---

## Key Source Directories

| Directory | Purpose |
|-----------|---------|
| `src/secure_aggregation/protocol/` | 4-round SAP protocol (`core.py`) + inter-cluster merge (`inter_cluster.py`) |
| `src/secure_aggregation/crypto/` | AEAD (AES-256-GCM), ECDH (P-256), Ed25519, ChaCha20 PRG, Shamir sharing |
| `src/secure_aggregation/communication/` | gRPC services: TTP, Aggregator, Node, Bridge, hierarchy mixin |
| `src/secure_aggregation/data/` | Config-driven dataset loading (`datasets.py`), Dirichlet partitioning (`partition.py`) |
| `src/secure_aggregation/training/` | MNIST training flow (`mnist_flow.py`) |
| `src/secure_aggregation/topology/` | D-Cliques construction with label-skew mitigation (`graph.py`) |
| `src/secure_aggregation/node/` | NodeEngine (reliability scoring, gossip cache), ECM buffering |
| `src/secure_aggregation/storage/` | IPFS interface, blockchain gateway (FastAPI), model registry |
| `src/secure_aggregation/convergence/` | Convergence tracker, central broadcast/checker |
| `src/secure_aggregation/state/` | Hierarchical state aggregation (cluster/state/nation scopes) |
| `src/secure_aggregation/config/` | ScenarioConfig, NodeRole, Timeouts, system config models |
| `src/secure_aggregation/models/` | VectorModel (flatten/unflatten), ModelRegistry (versioning) |
| `src/secure_aggregation/utils/` | Logging, Prometheus metrics, comm metrics, retry logic |

### Key Files by Importance

| File | Purpose |
|------|---------|
| `communication/node_service.py` | Main node orchestrator: training loop, SAP execution, model fetching (~2718 lines) |
| `protocol/core.py` | 4-round SAP: `SecureAggregationNode` + `SecureAggregationAggregator` classes |
| `communication/aggregator_service.py` | gRPC server handling all 4 protocol rounds |
| `crypto/shamir.py` | Shamir (t,n) threshold secret sharing over PRIME field |
| `crypto/dh.py` | ECDH key exchange (P-256), shared key derivation (HKDF-SHA256) |
| `data/partition.py` | Dirichlet non-IID partitioning |
| `data/datasets.py` | Config-driven dataset loading (torchvision, CSV) |
| `topology/graph.py` | D-Cliques with label-skew mitigation (greedy swapping) |
| `protocol/inter_cluster.py` | Adaptive clipping + weighted merge for inter-cluster |
| `convergence/tracker.py` | Accuracy/loss delta monitoring, warmup, patience |
| `storage/model_store.py` | IPFS + Blockchain abstractions (real + mock implementations) |

---

## Training Flow — Complete Per-Round Pipeline

### Phase 1: Local Training

Each node trains on its Dirichlet-partitioned data subset.

- **Models supported:**
  - `MnistLinear`: Simple linear classifier (784 inputs -> 10 classes)
  - `CifarConvNet`: Compact CNN with batch norm, max pooling, dropout
- **Optimizer:** SGD (lr=0.1, momentum=0.9)
- **Configurable:** epochs, batch_size, learning rate via per-node JSON config
- **Output:** Updated model parameters flattened to a float vector

### Phase 2: Aggregator Election

- Round-robin within the clique based on reliability score.
- `score = uptime + bandwidth - latency`
- `elected = sorted_members[round_index % len(sorted_members)]`
- Elected node starts gRPC aggregator server on its aggregator port.

### Phase 3: Secure Aggregation Protocol (4 Rounds)

#### SAP Round 0 — Key Advertisement

**Client-side:** `client.advertise_keys()`
- Generate two ECDH keypairs:
  - `c_keypair`: For encryption of shares (pairwise key derivation)
  - `s_keypair`: For masking vector generation (pairwise mask derivation)
- Sign both public keys with Ed25519 signing key.
- Send `AdvertiseMessage(node_id, c_public, s_public, signature, signing_public)`.

**Aggregator-side:**
- Collect advertisements from all participants.
- Verify Ed25519 signatures.
- Wait for threshold (or all) participants.
- Broadcast all advertisements to every node.

#### SAP Round 1 — Secret Share Distribution

**Client-side:** `client.create_round1_ciphertexts(ordered_participants, threshold)`
- Pick random seeds for two secrets:
  - `s_secret_int = c_private.private_value` (ECDH private key as integer)
  - `b_seed_bytes = os.urandom(32)` (PRG seed for self-mask)
- **Shamir split** both secrets into (t, n) shares over PRIME field.
- **Encrypt** each share per recipient:
  - Derive shared key: `ECDH(c_private, peer.c_public)` -> HKDF-SHA256 with `info=b"secure-agg/cipher"` -> 32-byte AES key.
  - Package: `x_index || s_share || b_share`
  - Encrypt with AES-256-GCM.
- Return `Round1Ciphertext(sender_id, recipient_id, iv, ciphertext, tag)`.

**Aggregator-side:**
- Collect ciphertexts from all participants.
- Route to recipient mailboxes: `encrypted_shares[recipient_id]`.
- Participants poll for their mailbox.

#### SAP Round 2 — Masked Input Submission

**Client-side:** `client.create_masked_input(model_vector)`

1. **Quantize model:** `quantized = [int(round(w * scale)) for w in flatten_params(model)]` (scale = 1e6).
2. **Pairwise masks:** For each peer:
   - `shared_key = ECDH(s_private, peer.s_public)` -> HKDF-SHA256 -> PRG seed.
   - `mask = PRG(shared_key, vector_length)` mod PRIME.
   - Direction-dependent: if `node_id > peer_id`, negate mask. Ensures `mask_ij = -mask_ji` (antisymmetric cancellation).
3. **Self-mask:** `self_mask = PRG(b_seed, vector_length)` mod PRIME.
4. **Masking:** `masked = quantized + self_mask + sum(pairwise_masks)` all mod PRIME.
5. Submit masked vector to aggregator.

**Aggregator-side:**
- Collect masked inputs from participants.
- Once threshold met, determine survivors: `survivors = sorted(masked_inputs.keys())`.
- Broadcast survivor list to all nodes.

#### SAP Round 3 — Consistency Check

**Client-side:** `client.sign_survivor_list(survivors)`
- Prepare message: `",".join(sorted(survivors)).encode()`.
- Sign with Ed25519: `signature = sign_message(signing_private, message)`.
- Return `SurvivorSignature(node_id, signature)`.

**Aggregator-side:**
- Collect signatures from all survivors.
- Verify all signatures match the survivor list.
- Ensures consensus on who participated.

#### SAP Round 4 — Unmasking & Aggregation

**Client-side:** `client.prepare_unmasking_payload(dropouts, survivors)`
- For each dropout: send stored `s_shares[dropout_id]`.
- For each survivor: send stored `b_shares[survivor_id]`.

**Aggregator-side:**
1. Collect shares from threshold survivors.
2. **Reconstruct dropout secrets** via Lagrange interpolation:
   - `s_private = combine_shares(s_shares_for_dropout)` -> recover ECDH private key.
   - Recompute pairwise masks between dropout and each survivor.
   - Remove those masks from aggregate.
3. **Reconstruct survivor b_seeds:**
   - `b_seed = combine_shares(b_shares_for_survivor)`.
   - Recompute self-mask and remove from aggregate.
4. **Compute mean:**
   - `aggregate = sum(masked_inputs)` mod PRIME.
   - Remove all masks.
   - Convert from modular to signed: if `val > PRIME/2`: `val -= PRIME`.
   - `mean = aggregate / num_survivors`.
5. **Dequantize:** `weights = mean / scale`.

### Phase 4: Model Update

- All nodes fetch aggregated model from aggregator via `GetGlobalModel` gRPC call.
- Load dequantized weights into local model.

### Phase 5: ECM Forwarding (Bridge Nodes Only)

- Bridge nodes forward **Encrypted Cluster Models (ECMs)** to neighbor clusters via IPFS.
- ECM structure: `{cid, hash, source_cluster, cluster_converged}`.
- Stored in ECM buffer with deduplication per source.

### Phase 6: Inter-Cluster Merge (Aggregator Only)

- Collect neighbor ECMs from bridge buffers.
- **Adaptive clipping:**
  - Compute delta: `delta = neighbor_model - local_model`.
  - Maintain sliding window of delta norms (last 10 rounds).
  - Set threshold to 90th percentile.
  - Clip: `clipped_delta = delta * min(1, threshold / ||delta||)`.
- **Weighted average** of clipped models.
- **Convex combination:** `theta_final = (1 - gamma) * theta_local + gamma * theta_robust`.
- Store merged model CID on IPFS, register on blockchain.

### Phase 7: Convergence Check

- Track test accuracy delta and loss over rounds.
- **Warmup:** Skip convergence check for first N rounds (default: 3).
- **Tolerance:** Absolute (`tol_abs=1e-5`) and relative (`tol_rel=0.001`) thresholds.
- **Patience:** N consecutive rounds below threshold before declaring convergence (default: 3).
- **Global convergence:** Requires quorum of clusters to agree.
- Once triggered: broadcast stop signal to all nodes.

### Termination Conditions

- `max_rounds` reached (configured in `system-config.json`).
- Global convergence signal (all clusters converged).
- Manual stop via `make stop`.

---

## Cryptographic Primitives

| Primitive | Algorithm | Details | File |
|-----------|-----------|---------|------|
| Key Exchange | ECDH (P-256) | SECP256R1, X9.62 uncompressed format (65 bytes) | `crypto/dh.py` |
| AEAD | AES-256-GCM | 32-byte key, 96-bit nonce, 128-bit auth tag | `crypto/aead.py` |
| Signing | Ed25519 (EdDSA) | 256-bit keys, message authentication | `crypto/sign.py` |
| PRG | AES-CTR | HKDF-SHA256 key derivation, deterministic mask generation | `crypto/prg.py` |
| Secret Sharing | Shamir over Z_p | p = 2^521-1 (Mersenne prime), 66-byte shares | `crypto/shamir.py` |

**PRIME field:** `2^521 - 1` — All modular arithmetic in SAP uses this field.

**Key derivation chains:**
- Share encryption: `ECDH(c_private, peer.c_public)` -> `HKDF-SHA256(info=b"secure-agg/cipher")` -> 32-byte AES key.
- Mask generation: `ECDH(s_private, peer.s_public)` -> `HKDF-SHA256(info=b"secure-agg/dh")` -> AES-CTR PRG seed.

---

## Data Partitioning

### Dirichlet Non-IID Distribution
**File:** `src/secure_aggregation/data/partition.py`

- `dirichlet_partition(dataset_indices, labels, num_clients, alpha, seed)`
- **alpha parameter** controls heterogeneity:
  - `alpha=0.5` -> high non-IID (realistic federated setting)
  - `alpha -> infinity` -> uniform IID partitioning
- Per-label class: draw Dirichlet proportions for each client, distribute samples accordingly.

### Supported Datasets
**File:** `src/secure_aggregation/data/datasets.py` + `config/datasets.json`

| Dataset | Type | Input Shape | Classes |
|---------|------|-------------|---------|
| MNIST | torchvision | [1, 28, 28] | 10 |
| CIFAR-10 | torchvision | [3, 32, 32] | 10 |
| CIFAR-100 | torchvision | [3, 32, 32] | 100 |
| Fashion-MNIST | torchvision | [1, 28, 28] | 10 |
| Custom CSV | config-driven | variable | variable |

---

## Topology: D-Cliques

**File:** `src/secure_aggregation/topology/graph.py`

### Clique Formation
- `build_d_cliques(node_labels, clique_size, iterations=1000, seed)`
- Greedy swapping algorithm (1000 iterations) to minimize intra-clique label variance (L1 distance to global distribution).
- Merge singleton cliques to avoid isolated nodes.

### Inter-Clique Edge Modes
- `ring`: Cliques in ring topology.
- `ring_extra`: Ring + extra edges for robustness.
- `ring_star`: Ring with star connections.
- `small_world`: Watts-Strogatz small-world topology.
- `fully_connected`: All cliques connected.

### Aggregator Election
- Sort members by reliability score: `score = uptime + bandwidth - latency`.
- Round-robin: `elected = sorted_members[round_index % len(sorted_members)]`.

---

## Communication Layer (gRPC)

### Protobuf Definition
**File:** `protos/secureagg.proto`

### Services

| Service | Port | Purpose |
|---------|------|---------|
| AggregatorService | base + 1000 | 4-round SAP (Round0-4), GetGlobalModel, SubmitECMs, ConvergenceSignal |
| NodeService | base (51000+i) | Receives calls from TTP and other nodes |
| BridgeService | base + 2000 | Inter-cluster ECM exchange |
| TTPService | 50051 | Key distribution, node registration |

### Message Size Limits
- Default: 200 MB max gRPC message.
- Configurable via `GRPC_MAX_MESSAGE_MB` environment variable.

---

## Configuration

| File | Purpose |
|------|---------|
| `config/system-config.json` | Global: hierarchy levels, convergence thresholds, timeouts |
| `config/node.config.template.json` | Per-node: role, ports, dataset, training params, secure_agg settings |
| `config/nodes-map.json` | Hierarchy roster: nation -> state -> cluster -> nodes |
| `config/datasets.json` | Available datasets (MNIST, CIFAR-10, Fashion-MNIST, custom CSV) |
| `config/topology.json` | Generated D-Cliques topology |

### Key Config Parameters

```json
{
  "dataset": { "name": "cifar10", "num_clients": 12, "alpha": 0.5, "seed": 42 },
  "training": { "rounds": 3, "local_epochs": 1, "batch_size": 64 },
  "secure_agg": { "threshold": 3, "scale": 1000000.0 },
  "convergence": { "warmup_rounds": 3, "tol_abs": 1e-05, "patience": 3 }
}
```

---

## Port Allocation

| Component | Port |
|-----------|------|
| TTP | 50051 |
| Node i service | 51000 + i |
| Node i aggregator | 52000 + i |
| Node i bridge | 53000 + i |
| Node i metrics | 61000 + i |
| Grafana | 3000 |
| Loki | 3100 |
| Prometheus | 9090 |
| Blockchain orderer | 7050 |
| Blockchain peers | 7051, 8051, 9051 |
| Blockchain gateway | 9000 |

---

## CLI & Makefile

```bash
make setup                          # One-time: venv, deps, gRPC codegen, datasets, blockchain
make start NODES=10 CLIQUE_SIZE=5   # Launch full system
make stop                           # Graceful shutdown
make status                         # Process health check
make logs                           # Aggregated logs (Loki or file fallback)
make test                           # Run unit tests
make clean                          # Full cleanup (keeps venv)
make clean-all                      # Remove venv too
```

**Direct launch:**
```bash
python scripts/secureagg_ctl.py start --nodes 10 --clique-size 5
python scripts/run_ttp_with_topology.py --topology config/topology.json --port 50051
python -m secure_aggregation.communication.node_service --config config/node_0.json
uvicorn secure_aggregation.storage.blockchain_gateway:app --host 0.0.0.0 --port 9000
```

---

## Dependencies

- **Core:** cryptography>=42.0, fastapi>=0.110, uvicorn, httpx, pyyaml, protobuf, prometheus_client, numpy
- **ML:** torch>=2.2, torchvision>=0.17, pandas>=2.0 (optional `mnist` extra)
- **Infrastructure:** gRPC (grpcio, grpcio-tools), IPFS Kubo, Hyperledger Fabric, Loki/Promtail/Prometheus/Grafana

---

## Testing

25 test directories under `tests/`:

| Category | Directory | Tests |
|----------|-----------|-------|
| Crypto | `tests/crypto/` | AEAD, ECDH, Ed25519, PRG, Shamir |
| Protocol | `tests/protocol/` | 4-round execution, arithmetic, dropout, message format |
| Data | `tests/data/` | Dirichlet partitioning, label skew, dataset loading |
| Integration | `tests/integration/` | Full MNIST flow, topology integration, inter-cluster flow |
| Communication | `tests/communication/` | gRPC service tests |
| Topology | `tests/topology/` | D-Cliques construction |
| Runtime | `tests/runtime/` | Process registry, port allocation, config generation |
| Hierarchy | `tests/hierarchy/` | State-level aggregation, convergence consensus |

```bash
make test                    # All tests
pytest tests/protocol/       # Protocol-specific
pytest tests/integration/    # Full system integration
```

---

## Typical Performance

| Metric | Value |
|--------|-------|
| Per-round wall time | 30-60 seconds (CPU) |
| Local training (1-2 epochs) | 5-10 sec/node |
| 4-round protocol | 10-20 sec/aggregator |
| Model size (MNIST) | ~20 KB (quantized) |
| Communication/round | ~5 MB per node |
| Full training (10 rounds) | 5-10 minutes |
| Final accuracy (MNIST) | ~91% |
| Tested scale | 4-10 nodes (demo), supports 50+ |

---

## Security Properties

1. **Privacy:** Aggregator learns only average model, never individual updates (semantic security).
2. **Dropout Tolerance:** System survives if >= threshold nodes remain (at least t+1 of n).
3. **Authentication:** Ed25519 signatures prevent impersonation.
4. **Consistency:** Round 3 signatures ensure agreement on participants.
5. **No Central Trust:** After TTP setup, no single party controls system.
6. **Quantization:** Float->int conversion prevents information leakage via floating-point precision.

---

## Development Conventions

- Python package at `src/secure_aggregation/`.
- gRPC definitions in `protos/secureagg.proto`, generated code in `communication/`.
- Config-driven: all behavior controlled via JSON config files.
- Process-only runtime: all components run as managed host processes (no Docker required).
- Quantization scale factor: `1e6` (float -> int conversion for secure aggregation).
- Node roles: TRAINER, AGGREGATOR, HYBRID, TTP (defined in `config/models.py`).
