# Implementation Plan (aligned with `context_codex`)

Goal: Ship a stable, configurable secure-aggregation FL system that supports arbitrary datasets, prioritizes Docker Compose deployment, and includes thorough unit tests per phase. Each phase yields a runnable MVP with a complete test suite for its scope.

## Folder Structure (planned)
- `src/`
  - `node/` (engine, roles, lifecycle, aggregator rotation)
  - `protocol/` (secure aggregation state machine)
  - `crypto/` (ECDH, Shamir, AES-GCM, PRG, signatures)
  - `topology/` (D-cliques construction, inter-clique edges, weights)
  - `data/` (partitioner, loaders, dataset adapters)
  - `models/` (pluggable model definitions)
  - `communication/` (gRPC stubs, clients, servers, TLS helpers)
  - `config/` (load/validate scenario + per-node configs)
  - `utils/` (logging, metrics, retries, resource cleanup)
- `protos/` (messages.proto)
- `docker/` (docker-compose.yml, Dockerfiles, entrypoints)
- `scripts/` (orchestration, health checks, cleanup)
- `tests/` (unit + integration per module; dockerized integration)
- `config/scenarios/`, `config/node_configs/` (templates + examples)
- `data/` (sample partitions), `logs/`, `checkpoints/` (mounted in Compose)

## Phase Breakdown (each phase = feature-complete slice + tests)
1) **Foundation + Config — DONE**  
   - Implemented config loading/validation (scenario + node + TTP auth keys), mount checks, service hostname alignment.  
   - Logging/metrics scaffolding in place.  
   - Tests: schema validation, defaulting, bad-input rejection, mounts creation.

2) **Crypto Primitives — DONE**  
   - ECDH P-256 KDF, Shamir t-of-n, AES-GCM, AES-CTR PRG, Ed25519 signing.  
   - Tests: vectors per primitive, share combine/failure paths, AEAD tag failure detection, PRG length/repro.

3) **Topology (D-Cliques) — DONE**  
   - Greedy-swap clique builder, inter-clique edge options (ring/fractal/small-world/fully-connected), Metropolis-Hastings weights.  
   - Tests: skew reduction property, edge count bounds, connectivity, weight row-stochasticity.

4) **Protocol Core (Secure Aggregation FSM) — DONE**  
   - Implemented per-round flows (Rounds 0–4) with Shamir sharing, pairwise/self masks, AEAD-protected shares, survivor signatures, dropout reconstruction, uniform mean aggregation.  
   - Tests: full end-to-end happy path, dropout handling with mask removal, tampered survivor signature rejection, AEAD tamper detection; threshold enforcement on signatures/shares.  
   - Planned (retained for traceability):  
     - Setup/TTP: distribute signing keys before Round 0.  
     - Round 0: generate two DH keypairs (c, s); sign/broadcast; verify signatures and uniqueness.  
     - Round 1: verify peers, derive KA(c) for AEAD, Shamir-share `s_SK` and `b_u`, AEAD-encrypt per-peer shares, forward/store.  
     - Round 2: decrypt shares, derive pairwise masks via KA(s) with sign rules, add self-mask, send masked update; server thresholds to U3.  
     - Round 3 (active): sign survivor list, verify all; abort on failure.  
     - Round 4: decrypt dropout shares, send s-shares for dropouts and b-shares for survivors, server reconstructs, removes masks, outputs Σx_u and uniform mean.  
     - Tests: happy-path end-to-end (small n), dropout handling, aborts on malformed signatures/AEAD failures/threshold violations, double-masking invariants, threshold checks (HBC t>n/2; active 2t>n+nC).

5) **Node Engine + Roles — DONE (logic + tests; transport TBD)**  
   - Trainer/hybrid role hooks with per-window aggregator rotation using deterministic score (uptime + bandwidth - latency).  
   - Gossip cache storing freshest K snapshots; freshness window Δt + recency weighting `w=exp(-age/τ)`; merge falls back to local mean if no fresh peers.  
   - Secure-agg orchestration helper for one window, supporting dropouts.  
   - Tests: aggregator rotation fairness/determinism, gossip capacity + freshness gating, recency-weighted merge math + fallback, orchestration end-to-end with dropout.  
   - Pending: wire to actual gRPC transport with TLS, retries, and health timeouts.

6) **Data + Models + Adaptation Layer — DONE**  
   - Dataset partitioners for IID (round-robin) and Dirichlet non-IID splits; label-aware assignment with deterministic seeding.  
   - Minimal vector model + registry for pluggable extensions; uniform averaging preserved (no data-ratio weighting).  
   - Tests: partition shapes/balance, Dirichlet alpha validation, model roundtrip/apply/compute-update, registry register/create semantics.

7) **Orchestration + Docker Compose — DONE (files + validation test; runtime wiring later)**  
   - Dockerfiles for node and TTP; compose file with shared network and volume mounts (`config`, `data`, `logs`, `checkpoints`), service names aligned to configs.  
   - Tests: compose structure validation (services, mounts, network).  
   - Pending: full dockerized integration run, health checks, cleanup script.

8) **Observability & Metrics — DONE (core plumbing; exporters later)**  
   - In-memory metrics + composite fan-out; convergence detector for “rounds-to-convergence”; Timer helper.  
   - Tests: composite fan-out, convergence detection.

9) **Reliability + Hardening — DONE (helpers; transport/backpressure later)**
   - Retry helper with exponential backoff; cleanup manager for resource teardown.
   - Tests: retry success/failure, cleanup order.
   - Security checks already enforced in protocol rounds (sigs/AEAD/thresholds).

10) **Performance + Scale Smoke — DONE (small-scale smoke)**
   - Small synthetic multi-window orchestration to ensure no failures under 10 nodes.
   - Tests: small-scale secure-agg cycles with dropouts.

11) **Networked Nodes + Docker Topology — TODO**
   - Add real node service process (HTTP/gRPC) that exposes round endpoints and orchestrates secure-agg with peer/aggregator election; derive clique membership + neighbors from topology builder.
   - Scenario-driven startup: node reads scenario + node config, builds D-cliques + inter-clique edges, selects per-clique aggregator for the window, and gossips over defined neighbors.
   - Docker Compose: update entrypoints to run the node service (not sleep), allow scaling `node=N`, inject scenario/node ids via env, and keep mounts (`config`, `data`, `logs`, `checkpoints`).
   - Health/coordination: bootstrapping to fetch TTP keys, advertise keys, run rounds over the network; add minimal healthcheck for readiness.
   - Integration: add a small end-to-end dockerized test that spins a few nodes on the Docker network and verifies topology-driven aggregation completes.

## Architecture & Scalability Notes
- Modular layering: data/model agnostic; swapping datasets/models via config only.  
- Topology minimizes edges while mitigating label skew.  
- Aggregator rotates per aggregation window to distribute load/fault tolerance; cross-cluster gossip + freshness-gated merge accelerates convergence.  
- Uniform averaging avoids data-ratio leakage; no weighted FedAvg.  
- Dual DH keypairs (c for share encryption, s for masking) per round; TTP-provisioned signatures for authentication.  
- gRPC over TLS; timeouts/retries configurable.  
- Compose-first: service names used in configs; resource limits set per service.

## Configurability
- Scenario JSON: participants, thresholds `t`, topology choice, dataset/model hyperparams, timeouts, aggregation-window length (one secure-agg cycle), freshness window `Δt`, gossip degree, cache size `K`, recency decay `τ`.  
- Node JSON: role, addresses, TLS, per-round timeouts, logging level; references to TTP signing keys.  
- Environment overrides for ports/paths in Compose.

## Docker Compose Run (preferred)
1. Build images: `docker compose build`.  
2. Launch: `docker compose up --scale node=N` (ensure TTP starts first to distribute signing keys before Round 0).  
3. Mount `config/`, `data/`, `logs/`, `checkpoints/`; ensure service hostnames match node configs.  
4. Health checks gate protocol start; logs streamed.  
5. Cleanup: `docker compose down -v` to remove networks/volumes; cleanup script verifies no dangling containers.

## Testing Strategy
- Unit tests per module (crypto, topology, protocol, data, models, config, metrics).  
- Integration tests: in-process gRPC nodes; dockerized mini-fleet.  
- Fault-injection: dropped peers, bad signatures, threshold shortfalls.  
- CI: lint/type-check, unit, integration (non-Compose), then optional dockerized smoke.

## Stability & Safety
- Strict input validation, signature/uniqueness checks, threshold enforcement.  
- Deterministic seeds for tests; graceful teardown on aborts.  
- Resource hygiene: close files/sockets, delete temp dirs, Compose down with volumes.  
- Logging/metrics for observability; alerts on protocol aborts/timeouts.
