# Process-Only Runtime + Loki CLI Logging Plan (No Docker, Low Friction)

## Summary
1. Migrate the stack to process-only runtime on a single host, with zero Docker dependency in the supported path.
2. Keep one-command operator workflow through Makefile + Python CLI.
3. Add Loki-based centralized logging so troubleshooting does not require shell tail/grep per node.
4. Minimize friction: auto-generate runtime configs and per-node artifacts; keep manual config edits optional.
5. Guarantee graceful process shutdown and no leftover runtime processes/files that waste host resources.

## Verified Current State
1. IPFS already supports process execution via `scripts/run_ipfs_processes.py`.
2. Blockchain/Fabric already supports process execution via `../thesis-blockchain/api-gateway/process-runner/manage.sh`.
3. The current full stack is not Docker-free yet because:
   - process mode still launches FL services through Docker compose in `scripts/run_process_mode.py`.
   - blockchain artifact prep still launches Fabric CA in Docker in `scripts/run_docker_with_nodes.py`.

## Final Target (Locked)
1. Topology: single machine, multiple processes.
2. Scope: TTP + trainer nodes + IPFS + Fabric/API gateway + monitoring all as processes.
3. Isolation policy: unique ports and per-process isolated runtime directories; no shared mutable runtime resources.
4. Dataset policy: full per-node dataset copies (no shared dataset root between nodes).
5. Memory policy: monitoring and alerts (no hard OS-level memory caps in v1).
6. Cutover policy: hard cutover to process-only supported runtime.
7. Runtime entrypoint: Makefile wrapping a single Python control CLI.
8. Registry runtime policy: remove legacy registry service from required runtime.
9. Lifecycle policy: all managed processes must exit gracefully on stop/failure and leave no orphan PIDs, locked ports, or stale runtime handles.

## Public Interface Changes
1. Add unified control CLI: `scripts/secureagg_ctl.py`.
2. Standardize commands:
   - `secureagg_ctl start`
   - `secureagg_ctl stop`
   - `secureagg_ctl status`
   - `secureagg_ctl logs`
   - `secureagg_ctl logs --node trainer-node-001 --since 30m`
   - `secureagg_ctl logs --follow --level error`
3. Update Makefile targets to wrap the new CLI:
   - `make start`, `make stop`, `make status`, `make logs`, `make logs-node NODE=...`, `make logs-errors`.
4. Add optional runtime spec file: `config/process-runtime.json`.
5. Keep existing advanced files optional; normal runs require no manual per-node edits.

## Architecture and Runtime Layout
1. Runtime root: `process-runtime/`.
2. Per-node isolated layout:
   - `process-runtime/nodes/node_<i>/config/`
   - `process-runtime/nodes/node_<i>/data/`
   - `process-runtime/nodes/node_<i>/logs/`
   - `process-runtime/nodes/node_<i>/checkpoints/`
   - `process-runtime/nodes/node_<i>/pids/`
3. Shared immutable code is allowed; mutable state is per-process.
4. Port allocation defaults:
   - TTP: `50051`
   - Node i service: `51000 + i`
   - Node i aggregator: `node_port + 1000`
   - Node i bridge: `node_port + 2000`
   - Node i metrics: `61000 + i`
   - Prometheus: `9090`
   - Grafana: `3000`
   - Loki: `3100`
5. Enforce port collision checks before startup.

## Implementation Plan

### 1) Unified Process Orchestrator
1. Create `scripts/secureagg_ctl.py` as the only supported operator entrypoint.
2. Move lifecycle management from `scripts/run_process_mode.py` into `secureagg_ctl.py` (or make `run_process_mode.py` a thin compatibility wrapper).
3. Implement `start/stop/status/logs` subcommands with PID tracking and graceful shutdown.
4. Enforce shutdown sequence with signal escalation (`SIGTERM` -> timeout -> `SIGKILL`), PID-file cleanup, and post-stop verification that managed ports are released.
5. Add failure-trap cleanup so partial startup does not leave background processes on the machine.
6. Remove Docker runtime calls from supported execution path.

### 2) Docker-Free Artifact and Config Generation
1. Add generator module: `scripts/generate_process_layout.py`.
2. Inputs:
   - `--nodes` or nodes-map/system-config sources.
   - optional `config/process-runtime.json` overrides.
3. Outputs:
   - per-node configs in isolated runtime dirs.
   - topology artifacts with absolute local paths.
   - process-mode Prometheus targets using localhost metrics ports.
   - process-mode Grafana datasource configs using localhost URLs.
4. Ensure generated node configs set process-friendly addresses:
   - `network_host=127.0.0.1`
   - `ttp_address=127.0.0.1:50051`
   - no `/app/...` defaults in generated runtime files.

### 3) Blockchain Prep Without Docker
1. Extract reusable blockchain helper logic from `scripts/run_docker_with_nodes.py` into `scripts/runtime/blockchain_helpers.py`.
2. Replace Fabric CA Docker bootstrap with local process-based CA bootstrap (`fabric-ca-server` + `fabric-ca-client`).
3. Keep identity generation, VC signing, bulk registration, and gateway health flow unchanged functionally.
4. Remove process-path hard requirement on blockchain Docker compose files.

### 4) Node/TTP Runtime Hardening for Process Mode
1. Update `src/secure_aggregation/communication/node_service.py`:
   - use configurable `metrics_port` (remove hardcoded `8000`).
   - ensure path resolution works cleanly with runtime absolute host paths.
2. Update `scripts/run_ttp_with_topology.py` defaults to host process paths where needed.
3. Update `src/secure_aggregation/communication/hierarchy_mixin.py` to avoid strict container-name assumptions, supporting trainer IDs and canonical node IDs.

### 5) Loki + Promtail Integration (CLI-first)
1. Add Loki process management to orchestrator.
2. Add Promtail process management to orchestrator.
3. Auto-generate runtime observability configs:
   - `process-runtime/observability/loki.yml`
   - `process-runtime/observability/promtail.yml`
4. Promtail labels per log stream:
   - `service` (node, ttp, ipfs, peer, gateway, prometheus, grafana, etc.)
   - `node_id`
   - `state_id` (when available)
5. Ensure all managed processes write log files under `process-runtime/.../logs/` for ingestion.

### 6) CLI Log Query UX (No Manual Tail/Grep)
1. Implement `secureagg_ctl logs` backed by Loki HTTP API.
2. Support filters:
   - `--node`, `--service`, `--level`, `--contains`, `--since`, `--until`, `--limit`, `--follow`, `--json`.
3. Add convenient presets:
   - `secureagg_ctl logs --errors`
   - `secureagg_ctl logs --node trainer-node-003 --follow`
4. Keep readable default output for fast troubleshooting.

### 7) Monitoring Process Integration
1. Run Prometheus and Grafana as processes managed by orchestrator.
2. Reuse dashboard generation via `scripts/generate_grafana_dashboard.py`.
3. Generate process-mode Grafana datasource provisioning for:
   - Prometheus at `http://localhost:9090`
   - Loki at `http://localhost:3100`

### 8) Process Lifecycle and Cleanup Guarantees
1. Define a managed-process registry with component -> pidfile -> port list mapping.
2. On `stop`, verify all managed PIDs are gone and all reserved ports are no longer listening.
3. On `start`, refuse to continue if non-managed conflicting listeners exist; clean up only known managed leftovers.
4. Add `secureagg_ctl cleanup` to remove stale pidfiles/runtime temp artifacts after abnormal termination.
5. Ensure runtime state under `process-runtime/` is pruned safely (preserve logs by default, remove stale lock/pid/temp files).

### 9) Makefile and Docs Hard Cutover
1. Update Makefile to process-only supported runtime commands.
2. Keep Docker commands only as deprecated references (not supported runtime path).
3. Rewrite `README.md` and `RUN_INSTRUCTIONS.md` to CLI-first process workflow.
4. Document one-path operations:
   - start stack
   - check status
   - query logs
   - stop stack

## Test Plan

### Unit Tests
1. Port allocator: uniqueness, reserved-port conflict detection, deterministic mapping.
2. Layout generator: per-node isolated dirs and dataset copy paths.
3. Config generation: no Docker hostnames (`host.docker.internal`) and no container-only paths in process runtime outputs.
4. Loki query builder: correct query construction for all CLI filter combinations.

### Integration Tests
1. `secureagg_ctl start` launches full stack with healthy status.
2. `secureagg_ctl status` reports component health and pids accurately.
3. Node logs are ingested into Loki and queryable via `secureagg_ctl logs --node ...`.
4. `--follow` streams near-real-time logs from Loki.
5. End-to-end FL smoke with 4 nodes completes initialization and training rounds.
6. `secureagg_ctl stop` shuts down all managed processes cleanly.
7. Abrupt interruption test (`SIGINT`/startup failure) triggers cleanup and leaves no managed process running.
8. Post-stop verification confirms reserved ports are free and PID files are removed.

### Regression Tests
1. Hierarchy/state mapping remains correct after trainer-id/container-id mapping changes.
2. Blockchain registration flow remains successful in process-only path.
3. No supported runtime command invokes Docker.

## Acceptance Criteria
1. Entire system runs via `make start` without Docker installed.
2. Operators can troubleshoot by CLI logs only (no manual file tail/grep required).
3. Each node has isolated mutable runtime resources.
4. Default run path requires zero manual per-node config edits.
5. Full stack components (TTP, nodes, IPFS, Fabric/API, Prometheus, Grafana, Loki, Promtail) are process-managed.
6. After `make stop` (or failure rollback), there are no leftover managed processes, stale pidfiles, or occupied managed ports.

## Assumptions and Defaults
1. Single-host deployment only in v1.
2. Dataset duplication per node is mandatory per requirement.
3. Memory control is monitoring/alerts, not hard kernel-enforced limits in v1.
4. Host binaries are installed: `ipfs`, `orderer`, `peer`, `fabric-ca-client`, `fabric-ca-server`, `prometheus`, `grafana-server`, and Loki/Promtail binaries.
5. Runtime command surface is Makefile + `secureagg_ctl.py`; advanced overrides are optional.
