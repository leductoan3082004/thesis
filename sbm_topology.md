# D-Cliques Topology for Secure Aggregation

This document describes the **D-Cliques** topology used to distribute nodes into clusters for decentralized federated learning with secure aggregation.

> **Note**: Despite the common reference to "SBM topology", this system uses **D-Cliques** (Distribution-aware Cliques), not the Stochastic Block Model. D-Cliques is a clustering algorithm that optimizes for balanced data distributions.

---

## Overview

The topology construction has three phases:

1. **Data Distribution Collection** - Nodes report their label distributions
2. **D-Cliques Construction** - Nodes are clustered to minimize label skew
3. **Inter-Clique Edge Assignment** - Cliques are connected via small-world topology

---

## 1. Data Distribution

Each node `i` has a local label distribution:

```
p_i(y) = (samples of class y at node i) / (total samples at node i)
```

The **global distribution** is computed by averaging across all nodes:

```
p(y) = (1/n) * Σ p_i(y)  for all nodes i ∈ N
```

Distributions are stored as vectors with a predefined label-to-index mapping (e.g., `[0.2, 0.8]` means 20% label A, 80% label B).

---

## 2. D-Cliques Construction (Greedy Swap Algorithm)

### Parameters

| Parameter | Description |
|-----------|-------------|
| `M` | Maximum clique size (controls intra-clique density) |
| `K` | Number of swap iterations |
| `N` | Set of all nodes `{1, ..., n}` |

### Phase 1: Initial Random Cliques

1. Start with an empty clique list `DC = []`
2. Shuffle nodes randomly
3. Partition into groups of size `M`:
   ```
   DC = [{node_1, ..., node_M}, {node_M+1, ..., node_2M}, ...]
   ```

**Example**: 100 nodes with `M=20` produces ~5 cliques.

### Phase 2: Greedy Swap Iterations

For each iteration `k = 1..K`:

1. **Select** two random cliques `C₁` and `C₂`

2. **Compute current skew**:
   ```
   skew(C) = Σ |p_C(y) - p(y)|  for all labels y
   ```
   Where `p_C(y)` is the aggregated distribution of clique `C`.

3. **Find improving swaps**: For each pair `(i, j)` where `i ∈ C₁` and `j ∈ C₂`:
   - Create candidate cliques: `C₁' = C₁ - {i} + {j}`, `C₂' = C₂ - {j} + {i}`
   - If `skew(C₁') + skew(C₂') < skew(C₁) + skew(C₂)`, mark as improving

4. **Apply one random improving swap** (if any exist)

### Choosing K (Iteration Count)

A commonly used heuristic:

```
L = n / M              (number of cliques)
K = α * L * log(L)     (where α ∈ [1, 10])
```

**Example**: 100 nodes, M=20 → L=5 → K ≈ 10 × 5 × log(5) ≈ 35 iterations.

---

## 3. Inter-Clique Edge Assignment

After clustering, cliques are connected to form a communication graph.

### Topology Modes

| Mode | Description |
|------|-------------|
| `ring` | Each clique connects to its neighbors: 1→2→3→4→1 |
| `small_world` | Ring + power-of-2 shortcuts (1→3, 1→5, 2→4, 2→6, ...) |
| `fractal` | Ring + half-stride shortcuts |
| `fully_connected` | All cliques connected to all others |

### Small-World Construction (Default)

1. **Base ring**: Connect clique `i` to clique `(i+1) mod L`
2. **Add shortcuts**: For each power `k = 0, 1, 2, ...` up to `small_world_c`:
   - Connect clique `i` to clique `(i + 2^k) mod L`

This creates O(log L) connections per clique while maintaining short path lengths.

### Edge Assignment to Nodes (Load Balancing)

For each inter-clique edge `(Clique_A, Clique_B)`, assign the actual node-to-node connection using a greedy load-balancing heuristic:

1. **Initialize**: Each node starts with `M-1` edges (fully connected within its clique of size `M`)
2. **For each inter-clique edge**:
   - Pick the node in `Clique_A` with the **lowest current edge count**
   - Pick the node in `Clique_B` with the **lowest current edge count**
   - Create an edge between those two nodes
   - Increment both nodes' edge counts

```
node_edge_count[n] = clique_size - 1   (initial: intra-clique edges)

for each (clique_a, clique_b) in interclique_edges:
    best_a = argmin(node_edge_count[n] for n in clique_a)
    best_b = argmin(node_edge_count[n] for n in clique_b)
    create_edge(best_a, best_b)
    node_edge_count[best_a] += 1
    node_edge_count[best_b] += 1
```

This ensures edges are distributed evenly across nodes, preventing "hub" nodes with disproportionately many connections.

---

## 4. Aggregation Weights (Metropolis-Hastings)

For decentralized averaging, edge weights are computed using the Metropolis-Hastings rule:

```
w_ij = 1 / (1 + max(degree(i), degree(j)))    for neighbors i, j
w_ii = 1 - Σ w_ij                              (self-weight)
```

This ensures:
- Row-stochastic weights (each row sums to 1)
- Symmetric influence between neighbors
- Convergence to global average

---

## Implementation Reference

The topology is implemented in [topology/graph.py](src/secure_aggregation/topology/graph.py):

| Function | Purpose |
|----------|---------|
| `compute_label_distribution()` | Normalize node label counts to distributions |
| `compute_skew()` | Calculate L1 distance between clique and global distribution |
| `build_d_cliques()` | Main clustering algorithm with greedy swaps |
| `build_interclique_edges()` | Construct inter-clique connectivity (clique-level) |
| `assign_node_edges()` | Assign node-to-node edges with load balancing |
| `build_full_topology()` | Complete pipeline: cliques + intra + inter edges |
| `metropolis_hastings_weights()` | Compute row-stochastic aggregation weights |

---

## Key Properties

| Property | Value |
|----------|-------|
| Intra-clique | Fully connected (all nodes in a clique communicate) |
| Inter-clique | Small-world (O(log L) edges per clique) |
| Skew minimization | Greedy local search (randomized) |
| Weight matrix | Row-stochastic (for convergence) |
| Complexity | O(K × M²) for clustering, O(L log L) for edges |

---

## Corrections from Original Documentation

The following corrections were made from the original Notion documentation:

1. **Terminology**: The system uses **D-Cliques**, not SBM (Stochastic Block Model). SBM is a different concept (a generative model for random graphs with community structure).

2. **Small-world shortcuts**: The original doc described "2-step jumps (1-3, 1-5, 1-7...)" but the implementation uses **powers of 2** (offsets: 1, 2, 4, 8...), which is the standard small-world construction.

3. **Weight matrix**: Described as "doubly stochastic" but the Metropolis-Hastings implementation produces **row-stochastic** weights (rows sum to 1, but columns may not).

4. **Global distribution assumption**: The formula `p(y) = (1/n) * Σ p_i(y)` assumes equal sample sizes per node. If nodes have different dataset sizes, a weighted average would be more accurate.
