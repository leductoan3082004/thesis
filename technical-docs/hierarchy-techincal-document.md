# Hierarchical Aggregation Logic (State → Nation)

## 1. Scope
This document describes the orchestration logic for aggregation layers that sit above the clique/cluster level. It focuses on two tiers:
- **State level** – currently implemented end-to-end (collection, digest consensus, anchoring, and downstream replay).
- **Nation level** – scheduling hooks that watch the state layer and announce higher-tier rounds.

All mechanics live primarily in `secure_aggregation.communication.node_service.NodeService` and supporting helpers under `state/` and `convergence/`.

## 2. Inputs and Topology Metadata
1. **State Map (`config/state-map.json`)**  
   - Defines roster per state (`state_id`, desired node count, optional explicit trainer IDs).  
   - `scripts/run_docker_with_nodes.py` consumes this file to generate node configs and to label containers with logical trainer IDs that align with blockchain identities.
2. **Central Metadata (`central_broadcast.py`)**  
   - TTP populates `central_nodes` (ring-star “hub” clique) and `cluster_ids`.  
   - All nodes can fetch the latest version from the blockchain registry; state aggregators use it to know which clusters must contribute ECMs.
3. **System Configuration (`system-config.json`)**  
   - Carries `state_aggregation` and `nation_aggregation` blocks.  
   - Fields include `enabled`, scheduling interval (`rounds_per_state`/`rounds_per_nation`), consensus/collection timeouts, approach (`ring_star`), and identifiers used when anchoring models.

## 3. State-Level Flow

### 3.1 Candidate Election
1. Nodes parse `state_aggregation` at startup (`NodeService._load_state_config`).  
2. When central metadata arrives, `_configure_state_layer()`:
   - Derives the candidate pool. For `ring_star`, candidates are `central_nodes`.  
   - Creates a dedicated `ECMBuffer` for candidates so state digests do not evict clique-level ECMs.  
   - Instantiates a `StateAggregator` (needs IPFS + blockchain handles).
3. Each candidate advertises its bridge address offset (+2000) so central peers can exchange digests.

### 3.2 Scheduling
1. `_state_layer_enabled()` requires `enabled` and `rounds_per_state > 0`.  
2. At the end of each cluster round, `_maybe_start_state_round(round_idx)` checks whether `(round_idx + 1) % rounds_per_state == 0`.  
3. If due, it enqueues `(state_round_id, cluster_round_idx)` so execution can happen after local persistence tasks finish.

### 3.3 ECM Collection
1. `_run_next_state_round()` drains the queue and calls `_dispatch_state_ecm_to_central()` to ensure bridge nodes forward their latest ECM (CID/hash) to all central targets.  
2. State candidates read their dedicated `state_ecm_buffer`; `_execute_state_round()` repeatedly calls `StateAggregator.build_snapshot()` until every required cluster has contributed, or a collection deadline (default `collection_timeout_seconds`) hits.
3. Missing clusters trigger warning logs; if coverage is incomplete at the deadline, the round aborts and cluster-level training continues without anchoring.

### 3.4 Model Retrieval and Merge
1. For each snapshot entry, `StateAggregator.fetch_models()` pulls tensors from IPFS using the recorded CID or, if missing, a fallback anchor lookup (`_lookup_cluster_anchor`).  
2. `verify_model_hash()` guarantees integrity.  
3. `StateAggregator.merge_models()` averages the numpy arrays in deterministic order (cluster ID sorted) and caches the merged tensor plus its SHA256 hash inside `NodeService._state_round_cache`.

### 3.5 Digest Consensus
1. Each candidate broadcasts a lightweight digest via bridge gRPC using the `state::STATE_ID` channel. Payload contains `cluster_round`, `node_id`, and the hash.  
2. `_await_state_digest_consensus()` waits up to `digest_timeout_seconds` for digests from every candidate.  
3. `_record_state_digest()` stores both local and remote digests; `_verify_state_digest_consensus()` checks that every hash matches. If hashes differ, the round is rejected (no anchoring).  
4. Once all digests agree, `_maybe_finalize_state_round()` invokes `_try_state_commit()`.

### 3.6 Commit and Anchoring
1. `_try_state_commit()` rotates the commit leader each round (`leader_index = state_round % len(candidates)`) to distribute blockchain usage.  
2. The active leader:
   - Calls `StateAggregator.publish_state_model()` to add the merged tensor to IPFS (`cid`) and anchor the CID/hash on-chain under `AnchorScope.STATE`.  
   - If anchoring fails, the next candidate waits for the public anchor; if none appears before `commit_timeout_seconds`, the function iterates to the next candidate.
3. Non-leaders poll `_wait_for_state_anchor()` to observe the blockchain commit and mark the round as completed once seen.
4. After anchoring, the merged model stays cached for downstream application, and the digest/ECM caches for that round are cleared.

### 3.7 Consumption by Cluster Nodes
1. `_maybe_apply_state_model(round_idx)` runs before each cluster round.  
2. When a new state anchor exists for `state_id`, nodes fetch it from blockchain/IPFS, verify the hash, and overwrite their local PyTorch model parameters.  
3. Convergence trackers are “primed” so subsequent delta comparisons measure drift from the state baseline instead of the pre-state global model.  
4. ECM dispatch for subsequent state rounds always references the most recently applied state model (`_last_model_cid/hash`).

### 3.8 Failure and Retry Handling
- **Incomplete ECM coverage** – The round is skipped; scheduler re-attempts on the next interval.  
- **Digest mismatch** – Logs warning, no anchor is attempted; the round must rerun later.  
- **Anchor publish failure** – Leader logs error and next candidate takes over (round remains scheduled until any anchor is observed).  
- **Observer fallback** – Non-candidate nodes use `_wait_for_state_anchor_observer()` to poll for a committed anchor before resuming cluster rounds, preventing divergence when they rely on state checkpoints.

## 4. Generalized Multi-Tier Pattern

To add layers above state without rewriting orchestration code, we treat every tier as an instance of the same state-machine with pluggable inputs/outputs and on-demand connectivity. Aggregators no longer need to keep permanent channels to every lower-tier member; instead, they reach out to a configurable subset when a tier round starts.

### 4.1 Tier-Agnostic Phases

| Phase | Tier-Agnostic Behavior | Tier-Specific Inputs | Tier Outputs |
|-------|-----------------------|----------------------|--------------|
| Scope aggregator election | Pick the most robust nodes for the tier (uptime, stake, reliability, or even random for experimental tiers). | `candidate_selector` strategy, max aggregator count, minimum reputation. | Ordered list of aggregators plus fallback list. |
| On-demand fan-out | When a tier round begins, each aggregator dials up to `fanout_per_group` nodes per lower-level group (configurable). | Lower-tier roster, connection policy, timeouts. | Set of `SourceContribution` records (group id, node id, cid/hash). |
| Scheduling | Determine when a tier-round fires (`rounds_per_<tier>`). | Interval + pointer to driving rounds. | Queue entry `(tier_round_id, source_round_idx)`. |
| Collection & dedupe | Merge contributions from multiple nodes per group, filter duplicates, ensure quorum per group. | Dedup key (cid/hash), per-group quorum. | Snapshot map `{group_id -> [unique references]}`. |
| Retrieval & merge | Fetch tensors, verify hashes, average/merge. | Tensor shape + merge weights. | Merged ndarray + hash cached per tier round. |
| Digest consensus | Aggregators exchange hashes via `tier::<tier_id>` channel only after forming ad-hoc connections. | Tier name + aggregator contact info. | Consensus digest ledger. |
| Commit | Leader (rotating) anchors merged tensor to blockchain/IPFS. | Anchor scope + identifier. | Persistent CID/hash for downstream tiers. |
| Replay | Lower tiers poll for new anchors and overwrite their baseline. | Scope ID + application policy. | Updated model parameters + convergence reset. |

### 4.2 Implementation Strategy
1. **Tier descriptors** – Define `TierConfig` objects with `tier_name`, `anchor_scope`, `rounds_per_tier`, `digest_channel_prefix`, `source_scope`, `candidate_selector`, `max_aggregators`, and `fanout_per_group`.
2. **Aggregator selection** – Provide pluggable strategies:
   - Reliability-weighted selection (state tier default).
   - Random sampling for experimental tiers (e.g., nation).
   - Stake-based or reputation-based as future options.
3. **Ephemeral connections** – When `_maybe_execute_tier_round(tier_cfg)` runs, selected aggregators:
   - Query the topology service for the roster of lower-level groups.
   - Randomly choose up to `fanout_per_group` nodes inside each group and establish temporary channels (gRPC, HTTP, etc.).
   - Request the latest anchored CID/hash references.
4. **Snapshot assembly** – Deduplicate references per group so even if both aggregators fetched overlapping nodes the merged snapshot lists each CID once.
5. **Digest + commit helpers** – Reuse the existing functions but parameterize by `tier_cfg`. Aggregators exchange digests only with other aggregators of the same tier; no broad gossip is needed.
6. **Recursive replay** – After anchoring, call `_apply_tier_model(tier_cfg, tier_round)` to push the update downstream. Lower tiers keep no knowledge of the connection fan-out—they simply subscribe to their designated scope on the blockchain.

With this pattern, adding “nation”, “federation”, or deeper layers is purely configuration: define how to elect aggregators, how many lower-tier nodes to interrogate per group, and how often the tier round should run. The scheduler, digest, and anchoring pipelines stay identical.

## 5. Nation-Level Flow (Example)

### 5.1 Configuration and Scheduling
- `NationAggregationConfig` adds `max_aggregators`, `fanout_per_state`, and `rounds_per_nation`.  
- The tier scheduler watches completed state rounds; whenever `(state_round % rounds_per_nation) == 0`, it enqueues a nation round with a pointer to the triggering state round.  
- The active config defines the `nation::NATION_ID` digest channel and the anchor scope (either `STATE` or a dedicated `NATION` namespace).  

### 5.2 Aggregator Selection and Fan-out
- **Aggregator count** – `max_aggregators = 2`. Two nodes are randomly selected from the entire fleet (configurable to prefer reputational scores later).  
- **Connection pattern** – For each state group, a nation aggregator chooses up to `fanout_per_group` nodes (e.g., 3) and establishes temporary connections only while collecting references. The nodes need not be bridge nodes; any participant holding the latest state anchor reference suffices.  
- **Reference gathering** – Each contacted node returns `(cid, hash, state_round)` for its state. Aggregators deduplicate references (if both pulled from the same state node) so each state contributes once.

### 5.3 Nation Round Lifecycle
1. **Scheduling** – After every `rounds_per_nation` state rounds, `_maybe_schedule_tier_round(nation_cfg)` enqueues a nation round with pointer to the triggering state round.  
2. **Collection** – Aggregators fetch references from lower-tier nodes using the fan-out rules. They verify that every state group is represented, respecting per-group quorum thresholds.  
3. **Merge & digest** – Aggregators download tensors from IPFS, merge into the nation model, and exchange digests on the `nation::NATION_ID` channel. Multiple aggregators cross-validate hash equality before proceeding.  
4. **Commit** – A rotating leader anchors the nation model under `AnchorScope.STATE` (or a new `NATION` scope) and uploads to IPFS. Other aggregators observe the anchor and mark the round complete.  
5. **Distribution** – Using the same temporary connections, aggregators notify at least one node per state that a new nation model is available. States then replay the anchor and push it down to their respective cliques.

This example illustrates how higher tiers inherit the same primitives—only the selection policy, fan-out, and scope identifiers change.

## 6. Sequence Summary
1. Cluster rounds emit ECMs + metadata.  
2. State round scheduling fires every `rounds_per_state`.  
3. State candidates collect ECMs → fetch models → merge → broadcast digests → anchor.  
4. Cluster nodes pull anchored state models before continuing training.  
5. After each successful state round, nation round counters increment; once the configured ratio is met, nation hooks notify downstream automation so the upper tier can repeat the same pattern with aggregated state checkpoints.

## 7. Top-Down Model Application Logic

To avoid destabilizing local models when higher tiers publish new checkpoints, application must proceed scope-by-scope (nation → state → cluster → node) using a configurable assimilation algorithm rather than naive replacement. This mirrors best practices from large-scale FL systems (e.g., model interpolation and dampened updates used in production fleets).

### 7.1 General Algorithm
For each tier `T` (nation, federation, state, etc.) and its immediate downstream tier `T-1`:
1. Wait until the downstream tier confirms the upstream anchor exists (poll blockchain/IPFS).
2. Fetch the upstream tensor `M_T` and verify its hash.
3. Run an `ApplyPolicy` that blends `M_T` with the downstream baseline `M_{T-1}` before handing it off.

Common `ApplyPolicy` strategies (configurable per tier):
- **Full Replace** (state → cluster legacy mode): `M_{T-1}' = M_T`.  
- **Model interpolation / partial averaging** (default for higher tiers):  
  \[
  W_{t+1}^{(T-1)} = \alpha \, W_{t+1}^{(T)} + (1 - \alpha) \, W_{t}^{(T-1)}, \quad 0 < \alpha < 0.5
  \]
  where \(W_{t}^{(T-1)}\) is the downstream baseline before the update, \(W_{t+1}^{(T)}\) is the freshly anchored upstream model, and \(\alpha\) is supplied via configuration (e.g., `state_aggregation.apply_alpha = 0.2`).  
- **Adaptive Trust** (inspired by FedProx/Elastic Averaging): dynamically set \(\alpha\) based on divergence or group reliability.  
- **Layer-wise Injection**: only replace specific layers (e.g., classifier head) while leaving feature backbone untouched.

### 7.2 Top-Down Propagation Steps
Propagation is triggered only when the system reaches a tier that has no higher-level rounds queued. Example: if a nation round finishes but a continent round is scheduled to run immediately after, the nation layer caches its result and waits for the continent tier to finish (and for confirmation that no higher tier is pending). Once the topmost active tier completes, cascading proceeds top-down, one tier at a time, ensuring no higher layer will override the update mid-flight.

1. **Nation → State**  
   - Nation aggregator publishes anchor.  
   - If a higher tier (e.g., continent) is scheduled or running, nation nodes defer propagation until that tier completes. Otherwise, each state aggregator polls `AnchorScope.NATION`, retrieves `M_nation`, and applies the configured policy (default: interpolated merge) to produce `M_state'`.  
   - `M_state'` becomes the new baseline for state-level rounds and is cached as the reference when dispatching ECMs to nation aggregators during the next cycle.
2. **State → Cluster**  
   - State anchors already exist today; extend `_maybe_apply_state_model()` to support both full replace or interpolation (controlled via config `state_aggregation.apply_policy`). If the state tier still has pending state rounds for the same macro-cycle, propagation is postponed until the state round queue drains.  
   - After applying, call `_prime_convergence_tracker_state()` so delta metrics reset relative to the updated baseline.
3. **Cluster → Node**  
   - When clusters finish secure aggregation and no more cluster-level windows remain before the next higher-tier trigger, nodes optionally perform a final blend with their local cache (useful for devices with personalization layers). This is already supported via `NodeEngine.merge_with_remote()`; exposing the same policy knobs keeps all tiers consistent.

### 7.3 Implementation Hooks
- Introduce a shared utility `apply_tier_update(tier_cfg, upstream_tensor, local_tensor)` returning the blended tensor and metadata (`alpha_used`, `source_round`).  
- Extend `StateAggregationConfig`/`NationAggregationConfig` with policy fields (including the interpolation weight \(\alpha\)):
  ```json
  {
    "apply_policy": "interpolate",
    "apply_alpha": 0.3,
    "apply_layer_mask": ["classifier.*"]
  }
  ```
- Ensure each tier calls `apply_tier_update` only after its direct upstream has settled; no tier jumps directly to node level. This guarantees deterministic sequencing: Nation anchor → states assimilate → clusters assimilate → individual nodes refresh.

By enforcing step-wise application with configurable blending—and waiting until the highest pending tier completes before cascading—we prevent abrupt shifts, preserve personalization, and make it trivial to add yet another tier. Once the topmost round settles, every scope inherits the new model in order: highest tier → … → nation → state → cluster → node, never skipping intermediate checkpoints.
