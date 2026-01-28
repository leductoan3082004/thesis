# Secure Aggregation Federated Learning – Technical Flow

## 1. Purpose and Scope
- Deliver privacy-preserving federated training where the coordinator never observes individual model updates.
- Support heterogeneous deployments that span local cliques, state-level overlays, and eventual nation-tier rollups.
- Provide production-ready automation (Docker, blockchain bootstrap, monitoring) backed by reproducible configuration.

## 2. High-Level Architecture
```
┌───────────────┐      ┌──────────────────┐      ┌────────────────────┐
│    Dataset    │      │ Trusted Third    │      │ Blockchain + IPFS  │
│ (e.g. MNIST)  │─Dirichlet partition─►│ Party (TTP)     │◄────┬─anchors models────┐
└───────────────┘      │ · key registry  │      │ · Fabric registry  │
                       │ · clique planner│      │ · Gateway/Mock     │
                       └────────┬────────┘      └─────────┬──────────┘
                                │                          │
                      provisioning│gRPC                    │CID/hash storage
                                ▼                          │
                    ┌────────────────────┐                 │
                    │ Federated Nodes    │◄────ECM gossip──┘
                    │ (node_service.py)  │
                    │ · local trainers   │
                    │ · rotate aggregators
                    │ · bridge neighbors │
                    └────────────────────┘
                                │
                                ▼
                    ┌────────────────────┐
                    │ Aggregator Service │
                    │ · Runs SAP rounds  │
                    │ · Publishes model  │
                    └────────────────────┘
```

Key components:
1. **TTP Service (`communication/ttp_service.py`)** – issues Ed25519 signing keys, builds D-Cliques topology, and publishes metadata to Fabric/IPFS when configured.
2. **Node Service (`communication/node_service.py`)** – loads data shards, trains, speaks gRPC to the aggregator, mirrors ECMs via bridge hooks, and exposes Prometheus metrics.
3. **Aggregator Service (`communication/aggregator_service.py`)** – enforces the 4-round secure aggregation protocol from Bonawitz et al., validates signatures, and broadcasts survivor-approved aggregates.
4. **Bridge Service (`communication/bridge_service.py`)** – ferries External Cluster Models (ECMs) plus “state::” signals between neighboring cliques to enable hierarchical consensus.
5. **State Aggregator (`state/aggregation.py`)** – gathers per-clique CIDs, re-downloads models from IPFS, averages them at the state layer, and anchors the result.
6. **Storage Stack (`storage/model_store.py`)** – abstract IPFS client (Mock or Kubo) plus blockchain adapters (Mock, Registry REST, Hyperledger Fabric Gateway) for tamper-evident anchoring.

## 3. Data, Identity, and Topology Preparation Flow
1. **Dataset ingestion** – `scripts/prepare_data.py` retrieves torchvision datasets or CSV inputs and materializes them under `data/`.
2. **Dirichlet partitioning** – The TTP (or standalone tooling) calls `secure_aggregation.data.dirichlet_partition()` to assign indices per client, ensuring configurable non-IID skew (controlled by `alpha`).
3. **Topology computation** – Using the node label distributions, `secure_aggregation.topology.build_full_topology()` creates balanced D-Cliques, derives clique-level thresholds (`ceil(2/3 * |clique|)`), and picks inter-clique edges (ring-star, small-world, etc.).
4. **Key distribution** – Each node registers with the TTP, receives per-node signing material (`SigningKeyPair`), and learns its clique membership plus participants list (see `NodeConfig` and `ScenarioConfig` in `config/models.py`).
5. **System configuration** – `config/system-config.json` aggregates convergence, dataset, and hierarchy settings so every container consumes identical values via the `SYSTEM_CONFIG_PATH` mount.

## 4. Runtime Responsibilities Per Component
- **Node Service**
  - Loads the configured dataset slice, instantiates either `MnistLinear` or `CifarConvNet`, and keeps flattened parameter buffers for quantization.
  - Manages a `NodeEngine` that tracks reliability-weighted gossip caches, selects the clique aggregator, and stages secure aggregation rounds.
  - Stitches together local training, secure aggregation RPC calls, bridge client interactions, and convergence tracking.
- **Aggregator Service**
  - Constructs `SecureAggregationAggregator` objects with participant signing keys, enforces message-size ceilings (`GRPC_MAX_MESSAGE_MB`), and writes aggregated vectors to shared checkpoints.
  - Emits convergence metadata (CID, hash, delta norms) so bridge nodes can propagate “cluster_converged” signals.
- **Bridge Service + ECM Buffer**
  - Provides a gRPC ingress for neighbor cliques to push ECMs.
  - Deduplicates entries by CID, respects freshness windows, and exposes signal poppers so higher layers (state/nation) can react without starving clique-level buffers.
- **State Aggregation Helpers**
  - Poll ECM buffers looking for coverage across the required clique roster (state roster derived from `state-map.json`).
  - Re-hydrate tensors from IPFS, verify SHA256 hashes, average contributions, and publish the consolidated state checkpoint while coordinating digest consensus among state candidates.
- **Storage Interfaces**
  - `IPFSInterface` allows a mock in-memory store for dev or a true Kubo HTTP API in production.
  - `BlockchainInterface` abstracts Hyperledger Fabric gateway, registry service, or local mock files so any layer can anchor models and convergence metadata.

## 5. Training and Secure Aggregation Lifecycle
1. **Local Epochs** – Each round starts with `node_service` cloning the latest global model, training on its Dirichlet partition (default SGD, lr=0.1, momentum=0.9, configurable epochs/batch size), and flattening weights.
2. **Quantization** – Parameters are scaled (default `scale=1e6`) and cast to integers so they can be masked modulo the `PRIME` used by the cryptosystem.
3. **Aggregator Election** – `NodeEngine.select_aggregator()` sorts clique members by reliability score (uptime + bandwidth − latency) and uses the round index modulus to pick the window leader, ensuring fairness even when nodes drop or roles differ (TRAINER/AGGREGATOR/HYBRID).
4. **Secure Aggregation Protocol (Rounds 0–4)** – Implemented in `protocol/core.py`:
   - **Round 0 (Advertise)** – Nodes generate DH key pairs (`c`, `s`), sign payloads, and the aggregator verifies Ed25519 signatures using keys minted by the TTP.
   - **Round 1 (Share Keys)** – Nodes split their secrets and self-mask seeds via Shamir, encrypt them with pairwise ECDH-derived AEAD keys, and the aggregator relays ciphertext mailboxes.
   - **Round 2 (Masked Inputs)** – Survivors add pairwise masks plus their self-mask to the quantized vector and submit to the aggregator; dropouts simply stop after Round 1, preserving privacy.
   - **Round 3 (Consistency)** – Survivors sign the canonical survivor list; the aggregator requires a signature from every survivor to prevent a malicious coordinator from excluding honest parties.
   - **Round 4 (Unmask)** – Survivors send Shamir shares that allow the aggregator to reconstruct dropout masks and remove everyone’s self mask, yielding the aggregate sum and mean.
5. **Model Update** – Aggregator broadcasts the aggregate, nodes dequantize (`vector / scale`), load parameters, and log accuracy improvements. `ConvergenceTracker` compares delta norms against absolute/relative tolerances, requiring `patience` consecutive satisfied rounds before declaring local convergence.
6. **Convergence and stop logic** – Once a clique converges, it emits signals through the bridge, optionally waits for neighbor consensus, or honors directives from a central checker defined in `ConvergenceConfig`.

## 6. Hierarchical Flow Across Levels
- **Clique Level (“current level”)**
  - Data locality: Each clique runs a self-contained secure aggregation window using only members assigned by the TTP.
  - Reliability-sensitive rotation: The `NodeEngine` reliability score discourages unstable nodes from becoming aggregators too often while still keeping deterministic behavior.
  - ECM Production: Aggregator nodes publish their merged model to IPFS and push ECM announcements to bridge peers, tagging convergence metrics for downstream layers.
- **State Level**
  - Bridge services flag CIDs originating from state digests using the `STATE_SIGNAL_PREFIX`, enabling quick filtering inside `StateAggregator`.
  - State leaders re-fetch clique outputs (or fall back to anchored data) and average them; digest consensus ensures all state candidates agree on the merged hash before any blockchain commit.
  - Once consensus forms, leaders anchor the state-level CID/hash, and optional higher-tier `HierarchyLevelConfig` hooks can trigger additional rollups.
- **Nation/Future Levels**
  - Config placeholders orchestrate every N state rounds; while nation aggregation is currently a stub, the scheduling primitives and digest plumbing are already in place.

## 7. Deployment and Operational Flow
1. **Automation entry point** – `scripts/run_docker_with_nodes.py` reads `system-config.json`, optional `state-map.json`, and `node.config.template.json`, then:
   - Generates per-node configs and Compose overrides.
   - Bootstraps Hyperledger Fabric artifacts via the sibling `thesis-blockchain` repo (key enrollment, VC signing, JWT issuance).
   - Starts IPFS, monitoring stack, the TTP, and N node containers (set via `--nodes` or config).
2. **Docker services** – Each node container mounts config/data/logs/checkpoints and spawns:
   - `node_service` (training + aggregator client)
   - `aggregator_service` (activated only when elected)
   - optional bridge server/client threads
3. **Monitoring** – Prometheus and Grafana containers (pre-wired dashboards via `scripts/generate_grafana_dashboard.py`) scrape node metrics and expose cluster convergence traces. Logs route to the shared `logs/` volume and can be tailed with `make logs` or `docker compose logs`.
4. **Local testing** – Developers can start the TTP plus a handful of node services manually without Docker (documented in `RUN_INSTRUCTIONS.md`) for lightweight debugging.

## 8. Reliability, Security, and Observability Features
- **Dropout tolerance** – Each clique threshold enforces at least `t` survivors; Round 4 refuses to proceed until ≥ threshold unmasking payloads are present.
- **Authentication** – Every advertisement and survivor signature is validated against the TTP-issued Ed25519 keys, preventing spoofing of masking material.
- **Transport hardening** – All gRPC servers/clients set explicit message size ceilings (default 200 MB) so large CNN updates do not fail mid-round.
- **Convergence governance** – Central broadcast utilities can halt training when a separate convergence checker decides the federation is done, avoiding unbounded rounds.
- **Observability** – `PrometheusMetrics` exports round durations, message counts, bridge gossip stats, and convergence streaks; structured logs enumerate each secure aggregation phase and bridge signal.
- **Resilience against partial failures** – Bridge buffers only expire ECMs after configurable freshness windows, aggregator rotation spreads heavy lifting, and reliability scores keep flaky nodes at the periphery of leadership.

## 9. Configuration Entry Points
- `config/node.config.template.json` – authoritative defaults for dataset selection, model hyperparameters, thresholds, and networking.
- `config/system-config.json` – runtime toggles for convergence, hierarchy cadence, dataset overrides, and monitoring IDs.
- `config/state-map.json` – maps logical states to sequential node IDs so the generator can size each state roster and feed the level-1 `HierarchyLevelConfig`.
- Environment variables (e.g., `GRPC_MAX_MESSAGE_MB`, `CENTRAL_METADATA_GATEWAY_URL`, `SYSTEM_CONFIG_PATH`) empower operators to swap infrastructure back ends without code changes.

## 10. Flow Recap
1. Prepare data → 2. Start TTP & issue credentials → 3. Launch nodes/bridges via Docker → 4. Nodes locally train and enter SAP Round 0 → 5. Aggregator completes Rounds 1–4 → 6. Aggregated model published to IPFS/blockchain → 7. Bridge disseminates ECMs and convergence signals → 8. State aggregators reconcile digests and commit a higher-level model → 9. Repeat until convergence criteria or configured rounds are met.

This document captures the current flow so future extensions (e.g., nation-level aggregation or new datasets) can align with the established orchestration pipeline without rediscovering the underlying mechanics.
