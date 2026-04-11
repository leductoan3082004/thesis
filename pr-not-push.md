## Summary

- Added full multi-scope hierarchy support (cluster → state → nation) so federation can scale beyond single cliques. State and nation rounds now run on wall-clock intervals, collect lower-scope artifacts via fan-out buffers, and commit a single anchor per round.
- Introduced `HierarchyMixin`, state aggregation engines, nodes-map parsing, and per-scope configuration that let operators describe arbitrary hierarchies through `config/system-config.json` and `config/nodes-map.json`.
- Revamped logging, blockchain interactions, and warm-start flow: nodes publish scope-aware aggregator candidate rosters, poll gateways during wait windows, skip redundant anchors, and emit friendly “No STATE model available yet…” messages instead of raw 404s.
- Updated documentation, scripts, topology helpers, and tests to capture the hierarchy architecture, deployment process, and monitoring expectations.

## Changes

### Architecture & Runtime
- Refactored `node_service` to mix in `HierarchyMixin`, which manages scope runtimes, timers, wait windows, fan-out queues, bridge gossip, and the new `_process_high_level_rounds()` loop.
- Implemented state/nation round handling: ECM collection, averaging, commit (`StateAggregator`), and per-scope aggregator rotation pulled from `nodes-map.json`. Followers and aggregators unify around the same fetch/merge path, and the committing node now clears its own wait window by handing the CID/hash to `_mark_scope_fetch_ready`.
- Added per-scope candidate discovery logs (“STATE/NATION aggregator candidates …”, “… addresses …”) and ensured bridge services can gossip ECMs or fan-out payloads to higher scopes.

### Blockchain/Gateway & Storage
- Extended `ModelStore`/gateway client to expose scope-aware latest-model endpoints, metadata commits, and improved 404 handling (`No <SCOPE> model available yet` while polling, `No new <SCOPE> model available` when CIDs repeat). Removed noisy control-cluster anchor logs.
- Warm-start logic now applies uniformly at startup and during hierarchy rounds: all nodes fetch via the gateway/IPFS path, verify hashes, and apply the configured merge policy (replace or interpolate) as soon as a new CID appears.

### Configuration & Tooling
- Added `config/system-config.json` (with `hierarchy_levels`, convergence, fleet size) and `config/nodes-map.json` plus sample files. `scripts/run_docker_with_nodes.py` reads these to generate node configs, derive scope rosters, and emit hierarchy-aware compose stacks.
- Fixed topology parsing, bridge routing, and aggregator identity handling so states map correctly to clusters and nation rounds see the expected fan-out reports.
- Added `scripts/nodes_map_count.py` and other helpers for validating nodes-map contents.

### Documentation & Tests
- Rewrote README, RUN_INSTRUCTIONS, IMPLEMENTATION_SUMMARY, TOPOLOGY docs, and `technical-docs/hierarchy-techincal-document.md` to describe the hierarchy flow, polling semantics, configuration knobs, and operational procedures.
- Added Grafana/Prometheus dashboard updates for hierarchy metrics and a full-flow hierarchy test plus supporting unit tests to exercise the scheduling and configuration logic.

## Testing

- Manual hierarchy runs (6+ trainers) exercising:
  - State/nation timers firing on schedule, collecting ECMs, and anchoring exactly once per interval
  - Startup warm-start logs showing “No STATE model available yet …” when expected
  - Aggregators skipping remaining wait windows after commit and re-fetching their own models via the gateway
  - Followers polling every 5 s until new CIDs arrive or wait windows expire, logging “No new STATE/NATION model available …” for redundant anchors
  - Nation scope mirroring the same behavior with its longer intervals
  - Documentation rendering cleanly (checked via `markdownlint`)

---

## Summary

- Added deterministic SAP dropout simulation (env-controlled) so we can reproduce partial participation scenarios and ensure retry logic behaves as expected.
- Hardened SAP orchestration: nodes now interpret aggregator sync hints, back off/retry cleanly on failures, and respect `SapResult` passive-wait signals instead of desyncing when the leader is ahead.
- Prevented stale cluster artifacts from poisoning higher scopes by detecting stalled cliques (round gap/seconds/retries) and publishing local fallback snapshots to IPFS/blockchain for fan-out.

## Changes

### SAP Dropout & Simulation
- Makefile exports `NO_SAP`, `DROP_OUT_NODES`, `DROP_OUT_SEED`; `NodeService` consumes them to initialize a `DropoutManager` (`run_training_loop` calls `_planned_dropout_stage` and `_dropout_decision_for_round`).
- Nodes honor preplanned `DropoutStage` values (before Round0, before masked input, etc.) unless the node is the aggregator or the round is already retrying.
- Added CLI/env plumbing so operators can disable the entire SAP pipeline (`NO_SAP=1`) and fall back to plaintext aggregation without editing configs.

### SAP Failure Handling
- Introduced `SapResult` to capture passive waits vs. successful aggregates, and taught `_check_sync_or_abort` to translate aggregator sync codes into retry/passive behaviors.
- `_await_aggregator_round` waits for the elected leader to reach the target round; the training loop updates `current_round` when the server is already ahead, clearing stale state.
- `_sap_retry_counts` guard dropouts after a failure, aggregator servers auto-restart after errors, and retry messages now include context (method, gRPC code, etc.).
- Non-aggregators poll `GetGlobalModel` until metadata+weights are ready, ensuring they always load what the aggregator committed.

### Stale Cluster Mitigation
- System config gains `cluster_stale_detection` (round gap=5, seconds=300, retries=5). `NodeService` tracks the last cluster CID, publish timestamp, and neighbor ECM rounds.
- Fan-out logic now calls `_cluster_payload_is_stale`; when stale, `_publish_local_cluster_model` snapshots local weights to IPFS/blockchain and uses that CID for fan-out instead of the old aggregate.
- Bridge hooks record neighbor ECM rounds so stale detection knows how far behind the clique is relative to peers.
- Documentation was updated (`README.md`, `technical-docs/hierarchy-techincal-document.md`) to explain the new knobs.

## Testing

- SAP dropout simulation: ran clusters with `DROP_OUT_NODES=2 DROP_OUT_SEED=42`, confirming planned dropouts skip the intended stages and the round restarts cleanly.
- Plaintext fallback: `NO_SAP=1 make start` completes Round 1 without entering the secure protocol.
- Stale detection: forced permanent SAP failures (by dropping the aggregator) and observed fan-out publishing local fallback CIDs, with state aggregators pulling those fresh snapshots instead of the stale CID.
