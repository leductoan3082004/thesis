# D-Cliques Topology - Design & Implementation

This document is the **single source of truth** for the D-Cliques topology system and its integration with secure aggregation.

---

## 1. Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STARTUP PHASE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  1. TTP loads scenario.json (num_clients, clique_size, alpha, seed)     │
│  2. TTP runs dirichlet_partition() → data assignment per node           │
│  3. TTP computes node_labels from partition (label counts per node)     │
│  4. TTP runs build_full_topology() → cliques, edges                     │
│  5. TTP stores topology and distributes to nodes on registration        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PHASE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  For each training round:                                               │
│    1. Each node trains locally on assigned data                         │
│    2. Each CLIQUE runs independent secure aggregation:                  │
│       - Elect aggregator within clique (round-robin)                    │
│       - Run 4-round protocol with clique members only                   │
│       - threshold = ceil(0.6667 * clique_size)                          │
│    3. Update model with clique aggregate                                │
│    4. [FUTURE] Gossip to inter-clique neighbors                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Topology computation | TTP computes once at startup | Deterministic, no runtime overhead |
| node_labels source | Computed from data partition | No redundant configuration |
| Threshold per clique | `ceil(0.6667 * clique_size)` | 2/3 majority for fault tolerance |
| Inter-clique communication | Phase 2 (future) | Start simple, add gossip later |
| Aggregator election | Round-robin within clique | Fair distribution of load |

---

## 2. Configuration

### scenario.json

```json
{
  "num_clients": 50,
  "clique_size": 10,
  "alpha": 0.5,
  "seed": 42,
  "inter_clique_edges": "ring_star",
  "topology_iterations": 1000,
  "small_world_c": 2,
  "max_neighbors": null,
  "ring_star_extra": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `num_clients` | int | Total number of nodes in federation |
| `clique_size` | int | Maximum nodes per clique |
| `alpha` | float | Dirichlet parameter (lower = more non-IID) |
| `seed` | int | Random seed for reproducibility |
| `inter_clique_edges` | string | Edge mode: "ring", "ring_star" (default), "small_world", "fractal", "fully_connected" |
| `topology_iterations` | int | Greedy swap iterations for clique optimization |
| `small_world_c` | int | Number of power-of-2 offsets for small-world mode |
| `max_neighbors` | int/null | Optional cap on neighbor models merged per round (per aggregator) |
| `ring_star_extra` | int | Additional random edges per clique when using ring_star |

### Derived Values

```python
num_cliques = ceil(num_clients / clique_size)
threshold_per_clique = ceil(0.6667 * clique_size)

# Examples:
# 50 clients, clique_size=10 → 5 cliques, threshold=7 per clique
# 30 clients, clique_size=6  → 5 cliques, threshold=4 per clique
```

---

## 3. Topology Construction Flow

### Step 1: Data Partition

```python
from secure_aggregation.data.partition import dirichlet_partition

# Partition MNIST indices among clients
partition = dirichlet_partition(
    dataset=list(range(60000)),  # MNIST indices
    labels=mnist_labels,          # {idx: label}
    num_clients=50,
    alpha=0.5,
    seed=42
)
# Returns: {"client_0": [idx1, idx2, ...], "client_1": [...], ...}
```

### Step 2: Compute Label Distributions

```python
from collections import Counter

def compute_node_labels(partition, labels):
    """Convert partition to label distributions per node."""
    node_labels = {}
    for client, indices in partition.items():
        counts = Counter(str(labels[idx]) for idx in indices)
        node_labels[client] = dict(counts)
    return node_labels

# Returns: {"client_0": {"0": 100, "1": 50, ...}, ...}
```

### Step 3: Build Topology

```python
from secure_aggregation.topology import build_full_topology

cliques, intra_edges, inter_edges, edge_counts = build_full_topology(
    node_labels=node_labels,
    clique_size=10,
    iterations=1000,
    edge_mode="small_world",
    small_world_c=2,
    ring_star_extra=0,
    seed=42
)
```

### Step 4: Compute Per-Clique Threshold

```python
from math import ceil

def compute_clique_threshold(clique_size: int) -> int:
    """Compute threshold as 2/3 majority of clique size."""
    return ceil(0.6667 * clique_size)

# clique_size=10 → threshold=7
# clique_size=6  → threshold=4
# clique_size=3  → threshold=2
```

---

## 4. D-Cliques Algorithm

### Phase 1: Label Distribution Collection

Each node's local data distribution is computed from its partition:
```
p_i(y) = count of label y in node i's data / total samples in node i
```

Global distribution (assuming equal sample sizes):
```
p(y) = (1/n) × Σ p_i(y)
```

### Phase 2: Greedy Swap Optimization

```
Input: nodes with label distributions, clique_size M, iterations K
Output: List of cliques with minimized label skew

1. Randomly partition nodes into cliques of size M
2. For k = 1 to K:
   a. Pick two random cliques C₁, C₂
   b. For each pair (i ∈ C₁, j ∈ C₂):
      - Compute Δskew if we swap i and j
   c. If any swap reduces total skew:
      - Apply one random improving swap
3. Return cliques
```

### Phase 3: Inter-Clique Edge Assignment

**Clique-level edges**:

- `ring_star` (default): ensure ring connectivity, then select the largest clique (ties → lowest index) as the central hub and connect it to every other clique, forming a star overlay on the ring. Each designated hub node is wired directly to every outer clique so all data paths are one hop.
- `ring_star_extra`: optional integer (`ring_star_extra`) adds that many additional pseudo-random edges per clique (beyond the ring + star). This densifies the overlay so the hub nodes can maintain high degree without starving other cliques of redundancy.
- `RING_STAR_NUMBER_OF_CENTRAL_NODES` (env var, default `2`): controls how many hub members become “central” bridge nodes. These nodes handle inter-clique edges for the ring-star hub.
- `small_world`: after the base ring, add offsets `2^k` for `k = 0..small_world_c-1`, connecting clique `i` to `(i + offset) mod L`.
- Other modes keep the existing behavior: `ring` (ring only), `fractal` (ring + stride jump), `fully_connected` (clique-complete graph).

**Node-level edges** (load-balanced):
```
For each clique-edge (C_a, C_b):
    best_a = node in C_a with lowest edge count
    best_b = node in C_b with lowest edge count
    Create edge (best_a, best_b)
    Increment edge counts
```

For `ring_star`, the node-level assignment guarantees at least **RING_STAR_NUMBER_OF_CENTRAL_NODES** high-connectivity bridge nodes inside the central clique (or all members if the clique is smaller). Edges touching the hub always pick from that set, and the algorithm adds direct edges so each hub node connects to every outer clique, keeping paths to the central checker to a single hop. Extra random edges reuse the same nodes but never require multi-hop routing.

### Key Formulas

| Formula | Description |
|---------|-------------|
| `skew(C) = Σ \|p_C(y) - p(y)\|` | L1 distance between clique and global distribution |
| `w_ij = 1 / (1 + max(deg(i), deg(j)))` | Metropolis-Hastings edge weight |

---

## 5. Secure Aggregation Integration

### Per-Clique Aggregation (Phase 1 - Current Goal)

Each clique runs **independent** secure aggregation:

```
Clique 0: {node_0, node_1, ..., node_9}
  └─ Aggregator: elected via round-robin within clique
  └─ Threshold: 7 (for clique_size=10)
  └─ Protocol: 4-round secure aggregation
  └─ Output: Aggregated model for clique 0

Clique 1: {node_10, node_11, ..., node_19}
  └─ (same as above, independent execution)

...
```

### Aggregator Election (Per-Clique)

```python
def elect_clique_aggregator(clique_members: List[str], round_idx: int) -> str:
    """Elect aggregator via round-robin within clique."""
    sorted_members = sorted(clique_members)
    return sorted_members[round_idx % len(sorted_members)]
```

### Node Registration Response

When a node registers with TTP, it receives:

```python
{
    "node_id": "client_5",
    "clique_id": 0,
    "clique_members": ["client_0", "client_1", ..., "client_9"],
    "threshold": 7,
    "data_indices": [1234, 5678, ...]  # assigned data
}
```

### Inter-Clique Communication (Phase 2 - Future)

After clique aggregation, bridge nodes gossip to neighbors:

```
1. Identify inter-clique neighbors from inter_edges
2. Send aggregated model snapshot to neighbors
3. Before next round, merge fresh snapshots:
   merged = weighted_average(local_model, remote_snapshots)
```

This uses the existing `GossipCache` and `merge_with_remote()` in NodeEngine.

---

## 6. Implementation Status

### Topology Module (Complete)

| Component | File | Status |
|-----------|------|--------|
| `compute_label_distribution()` | [graph.py](src/secure_aggregation/topology/graph.py) | ✅ |
| `compute_skew()` | [graph.py](src/secure_aggregation/topology/graph.py) | ✅ |
| `build_d_cliques()` | [graph.py](src/secure_aggregation/topology/graph.py) | ✅ |
| `build_interclique_edges()` | [graph.py](src/secure_aggregation/topology/graph.py) | ✅ |
| `assign_node_edges()` | [graph.py](src/secure_aggregation/topology/graph.py) | ✅ |
| `build_full_topology()` | [graph.py](src/secure_aggregation/topology/graph.py) | ✅ |
| `metropolis_hastings_weights()` | [graph.py](src/secure_aggregation/topology/graph.py) | ✅ |

### Integration (TODO)

| Component | File | Status |
|-----------|------|--------|
| TTP topology computation | [ttp_service.py](src/secure_aggregation/communication/ttp_service.py) | ❌ TODO |
| Node clique membership | [node_service.py](src/secure_aggregation/communication/node_service.py) | ❌ TODO |
| Per-clique aggregator election | [engine.py](src/secure_aggregation/node/engine.py) | ❌ TODO |
| Clique boundary validation | [aggregator_service.py](src/secure_aggregation/communication/aggregator_service.py) | ❌ TODO |
| Threshold calculation | [config/models.py](src/secure_aggregation/config/models.py) | ❌ TODO |

---

## 7. Tests

### Unit Tests

**File**: [tests/topology/test_topology.py](tests/topology/test_topology.py)

| Test | Purpose |
|------|---------|
| `test_greedy_swaps_reduce_average_skew` | Verify swaps reduce label skew |
| `test_interclique_edges_are_connected_and_bounded` | Verify graph connectivity |
| `test_metropolis_hastings_weights_are_row_stochastic` | Verify weight matrix validity |
| `test_assign_node_edges_produces_correct_count` | Verify edge assignment correctness |
| `test_assign_node_edges_load_balances` | Verify load balancing (max diff ≤ 1) |
| `test_build_full_topology_returns_all_components` | Verify complete output |
| `test_build_full_topology_edges_connect_correct_nodes` | Verify intra/inter edge separation |

### MNIST Integration Tests

**File**: [tests/topology/test_topology_mnist.py](tests/topology/test_topology_mnist.py)

| Test | Purpose |
|------|---------|
| `test_greedy_swaps_reduce_skew_on_mnist` | Real MNIST data skew reduction |
| `test_clique_distributions_approach_global` | Verify distributions balance |
| `test_load_balancing_with_mnist_topology` | Load balancing with real data |
| `test_full_topology_with_mnist` | Complete pipeline test |
| `test_distribution_balance_across_cliques` | Verify skew variance is low |

### Run Tests

```bash
# All topology tests
.venv/bin/python -m pytest tests/topology/ -v

# MNIST tests only
.venv/bin/python -m pytest tests/topology/test_topology_mnist.py -v -s
```

### Test Results (MNIST with 50 clients, clique_size=10)

```
Number of cliques: 5
Average skew: 0.0705    (very low - close to global)
Min skew: 0.0559
Max skew: 0.0809
All 12 tests passed
```

---

## 8. File Structure

```
secure_aggregation/
├── src/secure_aggregation/
│   ├── topology/
│   │   ├── __init__.py              # Exports all topology functions
│   │   └── graph.py                 # Core topology implementation
│   ├── data/
│   │   └── partition.py             # Dirichlet partitioning
│   ├── protocol/
│   │   └── core.py                  # 4-round secure aggregation (unchanged)
│   ├── communication/
│   │   ├── ttp_service.py           # TTP service (TODO: add topology)
│   │   ├── node_service.py          # Node service (TODO: add clique membership)
│   │   └── aggregator_service.py    # Aggregator (TODO: add clique validation)
│   ├── node/
│   │   └── engine.py                # Node engine (TODO: clique-aware election)
│   └── config/
│       └── models.py                # Configuration models
├── tests/topology/
│   ├── test_topology.py             # Unit tests
│   └── test_topology_mnist.py       # MNIST integration tests
├── config/
│   └── scenario.sample.json         # Sample configuration
└── TOPOLOGY_IMPLEMENTATION.md       # This document (source of truth)
```

---

## 9. Quick Reference

### Topology API

```python
from secure_aggregation.topology import (
    build_full_topology,           # Complete pipeline
    build_d_cliques,               # Just clustering
    build_interclique_edges,       # Just inter-clique edges
    assign_node_edges,             # Just load-balanced assignment
    compute_skew,                  # Measure clique quality
    metropolis_hastings_weights,   # Aggregation weights
)

# One-liner for complete topology
cliques, intra, inter, counts = build_full_topology(node_labels, clique_size=10, seed=42)
```

### Data Partition API

```python
from secure_aggregation.data.partition import dirichlet_partition

partition = dirichlet_partition(indices, labels, num_clients=50, alpha=0.5, seed=42)
```

### Threshold Calculation

```python
threshold = ceil(0.6667 * clique_size)
```

---

## 10. Corrections from Original Documentation

| Issue | Original (Notion) | Corrected |
|-------|-------------------|-----------|
| Terminology | "SBM topology" | **D-Cliques** (SBM is a different concept) |
| Small-world | "2-step jumps (1-3, 1-5, 1-7...)" | **Powers of 2** (1, 2, 4, 8...) |
| Weight matrix | "doubly stochastic" | **Row-stochastic** (rows sum to 1) |
| Load balancing | Not specified | **Greedy min-edge-count** assignment |

---

## 11. Glossary

| Term | Definition |
|------|------------|
| **D-Cliques** | Distribution-aware cliques - groups of nodes with balanced label distributions |
| **Clique** | A group of nodes that run secure aggregation together |
| **Intra-clique edges** | Fully connected edges within a clique |
| **Inter-clique edges** | Edges between cliques (for future gossip) |
| **Skew** | L1 distance between a clique's label distribution and the global distribution |
| **Threshold** | Minimum survivors needed to complete secure aggregation |
| **Small-world** | Topology with ring + power-of-2 shortcuts for short path lengths |
