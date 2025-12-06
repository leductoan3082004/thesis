# D-Cliques Topology Implementation Summary

This document summarizes all work done on the D-Cliques topology for the secure aggregation system.

---

## 1. What Was Built

### Core Implementation

**File**: [src/secure_aggregation/topology/graph.py](src/secure_aggregation/topology/graph.py)

| Function | Purpose | Status |
|----------|---------|--------|
| `compute_label_distribution()` | Normalize node label counts to probability distributions | ✅ Existing |
| `compute_skew()` | Calculate L1 distance between clique and global distribution | ✅ Existing |
| `build_d_cliques()` | Build cliques using greedy swap algorithm to minimize skew | ✅ Existing |
| `build_interclique_edges()` | Construct inter-clique edges (ring, small_world, fractal, fully_connected) | ✅ Existing |
| `assign_node_edges()` | **NEW** - Assign node-to-node edges with load balancing | ✅ Implemented |
| `build_full_topology()` | **NEW** - Complete pipeline combining all steps | ✅ Implemented |
| `metropolis_hastings_weights()` | Compute row-stochastic aggregation weights | ✅ Existing |

### Exports

**File**: [src/secure_aggregation/topology/__init__.py](src/secure_aggregation/topology/__init__.py)

```python
from .graph import (
    assign_node_edges,
    build_d_cliques,
    build_full_topology,
    build_interclique_edges,
    compute_label_distribution,
    compute_skew,
    metropolis_hastings_weights,
)
```

---

## 2. Algorithm Overview

### D-Cliques Construction (3 Phases)

```
Phase 1: Data Distribution Collection
├── Each node reports label distribution p_i(y)
└── Compute global distribution: p(y) = (1/n) * Σ p_i(y)

Phase 2: D-Cliques Construction (Greedy Swap)
├── Randomly partition nodes into cliques of size M
├── For K iterations:
│   ├── Pick two random cliques C₁, C₂
│   ├── Find swap (i,j) that reduces skew(C₁) + skew(C₂)
│   └── Apply one random improving swap
└── Output: List of cliques with minimized label skew

Phase 3: Inter-Clique Edge Assignment
├── Build clique-level edges (ring + small-world shortcuts)
├── For each clique-edge (C_a, C_b):
│   ├── Pick node with lowest edge count from C_a
│   ├── Pick node with lowest edge count from C_b
│   └── Create node-to-node edge (load balanced)
└── Output: Intra-clique edges + Inter-clique edges
```

### Key Formulas

```
Skew:           skew(C) = Σ |p_C(y) - p(y)|  (L1 distance)
Small-world:    Connect clique i to (i + 2^k) mod L for k = 0,1,2,...
MH Weights:     w_ij = 1 / (1 + max(degree(i), degree(j)))
```

---

## 3. Usage Example

```python
from secure_aggregation.topology import build_full_topology

# Node label distributions (from MNIST partition, etc.)
node_labels = {
    "client_0": {"0": 100, "1": 50, "2": 30},
    "client_1": {"0": 20, "1": 150, "2": 10},
    # ... more clients
}

# Build complete topology
cliques, intra_edges, inter_edges, edge_counts = build_full_topology(
    node_labels=node_labels,
    clique_size=10,           # Max nodes per clique
    iterations=500,           # Greedy swap iterations
    edge_mode="small_world",  # Inter-clique topology
    small_world_c=2,          # Power-of-2 offsets (1, 2, 4)
    seed=42                   # Reproducibility
)

# Results:
# - cliques: List[Set[str]] - node groupings
# - intra_edges: List[Tuple[str, str]] - edges within cliques (fully connected)
# - inter_edges: List[Tuple[str, str]] - edges between cliques (load balanced)
# - edge_counts: Dict[str, int] - final edge count per node
```

---

## 4. Tests

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

# MNIST tests only (downloads dataset if needed)
.venv/bin/python -m pytest tests/topology/test_topology_mnist.py -v -s
```

### Test Results (MNIST)

```
Number of cliques: 5
Average skew: 0.0705    (very low - close to global)
Min skew: 0.0559
Max skew: 0.0809
All 12 tests passed
```

---

## 5. Documentation

**File**: [sbm_topology.md](sbm_topology.md)

Complete documentation covering:
- Data distribution collection
- D-Cliques construction algorithm
- Inter-clique edge assignment with load balancing
- Aggregation weights (Metropolis-Hastings)
- Implementation reference
- Corrections from original Notion documentation

---

## 6. Key Corrections Made

| Issue | Original (Notion) | Corrected |
|-------|-------------------|-----------|
| Terminology | "SBM topology" | **D-Cliques** (SBM is a different concept) |
| Small-world | "2-step jumps (1-3, 1-5, 1-7...)" | **Powers of 2** (1, 2, 4, 8...) |
| Weight matrix | "doubly stochastic" | **Row-stochastic** (rows sum to 1) |
| Load balancing | Not implemented | **Now implemented** via `assign_node_edges()` |

---

## 7. File Structure

```
secure_aggregation/
├── src/secure_aggregation/
│   ├── topology/
│   │   ├── __init__.py          # Exports all topology functions
│   │   └── graph.py             # Core implementation
│   └── data/
│       └── partition.py         # Dirichlet partitioning for non-IID
├── tests/topology/
│   ├── test_topology.py         # Unit tests
│   └── test_topology_mnist.py   # MNIST integration tests
├── sbm_topology.md              # Full documentation
└── TOPOLOGY_IMPLEMENTATION.md   # This summary file
```

---

## 8. Next Steps (If Needed)

1. **Integration with Aggregator**: Use `build_full_topology()` output to configure secure aggregation communication graph.

2. **Topology Matrix to IPFS**: The Notion doc mentions committing topology to IPFS - this would require serializing the edge lists.

3. **Dynamic Topology**: Current implementation is static. Could add support for node join/leave.

4. **Weighted Global Distribution**: Current implementation assumes equal sample sizes. Could weight by actual dataset sizes.

---

## 9. Quick Reference

```python
# Import
from secure_aggregation.topology import (
    build_full_topology,      # Complete pipeline
    build_d_cliques,          # Just clustering
    build_interclique_edges,  # Just inter-clique edges
    assign_node_edges,        # Just load-balanced assignment
    compute_skew,             # Measure clique quality
    metropolis_hastings_weights,  # Aggregation weights
)

# One-liner for complete topology
cliques, intra, inter, counts = build_full_topology(node_labels, clique_size=10, seed=42)
```
