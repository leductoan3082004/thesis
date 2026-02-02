# Hierarchical Aggregation Logic (Cluster → State → Nation)

## 1. Scope and Terminology
- **Level 0** is synonymous with the cluster layer. This naming is fixed.  
- A **high-level** round is any round where `level > 0` (state, nation, future tiers).  
- The document captures the clarified intent of the hierarchy architecture so implementation teams can evolve existing code (`secure_aggregation.communication.node_service.NodeService` and friends) toward the new behavior.

## 2. Inputs and Configuration
1. **Hierarchical Nodes Map (`config/nodes-map.json`)** lists every scope and its members. The schema mirrors `hierarchy_levels`: each object uses the `scope_name` from the config as its key and nests children using the next level’s `scope_name`. Example (scope levels: nation → state → cluster → node):
   ```json
   {
     "nation": [
       {
         "nation_id": "VNM",
         "states": [
           {
             "state_id": "state_alpha",
             "clusters": [
               {
                 "cluster_id": "cluster_1",
                 "nodes": [
                   "trainer-node-001",
                   "trainer-node-002"
                 ]
               }
             ]
           }
         ]
       }
     ]
   }
   ```
   Administrators can add new levels by editing both `hierarchy_levels` and this map; the runtime resolves scope IDs dynamically at startup and treats the resulting structure as immutable for the session.  
2. **Topology Metadata (`central_broadcast.py`)** distributes the latest cluster roster and ring-neighbor hints. Nodes use it to learn which clusters participate in each scope and how to reach peers when establishing temporary fan-out links to higher-level aggregators.  
3. **System Configuration (`system-config.json`)** declares an array of `hierarchy_levels`. Each entry now includes:
   - `scope_index`, `scope_name`, `scope_id`  
   - `interval_seconds` (replaces `rounds_per_scope` for high-levels)  
   - Waiting policy knobs (`wait_seconds`, retry count) that nodes honor when pausing for high-level models  
   - Merge policy fields (`apply_policy`, `apply_alpha`, etc.) so nodes know how to assimilate each scope’s model when fetched directly.

## 3. Level Inventory
Current deployments use three scopes:
1. **Level 0 – Cluster**: baseline secure aggregation with secure aggregation protocol → blockchain/IPFS commit after every cluster round.  
2. **Level 1 – State**: high-level scope aggregating cluster representatives.  
3. **Level 2 – Nation**: high-level scope aggregating state representatives.  
The system remains extensible; to add more tiers, extend the config and the time-based scheduler automatically activates them.

Every scope reuses the ring topology for peer discovery; extra edges are opened only for the duration of a high-level round so fan-out nodes can talk directly to the single elected aggregator. No central clique is required.

## 4. Time-Based Scheduling of High-Level Rounds
- Level 0 still runs each time the cluster secure aggregation completes.  
- Level ≥ 1 rounds are no longer triggered by counting lower-level rounds.  
- Instead, each high-level scope defines an `interval_seconds`:
  - Every `t1` seconds → trigger the state round.  
  - Every `t2` seconds → trigger the nation round.  
- Administrators ensure `t1 < t2` (or any other hierarchy-consistent constraints) through configuration.  
- Scheduling daemons fire based on wall-clock timers; if a timer fires while a prior high-level round is still running, the new request is queued until the scope is idle.

## 5. Node Runtime Behavior (Non-Blocking Execution)
### 5.1 Waiting Logic
- Nodes are never globally blocked by a running high-level round.  
- When a scheduler announces that scope `S` started, every node checks whether it belongs to `S`.  
- If yes, it waits for `scope.wait_seconds`. During this window the node pauses local training to give aggregators time to publish models.  
- After the timer elapses, the node proceeds regardless of aggregator status.

### 5.2 Model Fetch Endpoint
- Once the wait period ends, nodes call the blockchain/IPFS endpoint for scope `S` with retries as configured.  
- Endpoints are level-aware: the node must specify `(scope_id, desired_level)` so it receives the latest CID for that scope.  
- If the endpoint returns `404 / empty`, the node resumes cluster rounds. There is no further blocking; nodes will check again the next time the waiting logic triggers.

### 5.3 Direct Pull and Merge
- High-level models are no longer pushed downward tier by tier. Nodes directly pull the scope that was executing.  
- Nodes cache merge metadata per `(scope_id, cid)`. If the CID matches the last applied model, the node skips the merge and continues training.  
- When a new CID is observed, the node fetches the model from IPFS and merges it into its local model using the level-specific policy declared in `hierarchy_levels`:
  - e.g., State level → `apply_policy = "replace"`.  
  - Nation level → `apply_policy = "interpolate"` with `alpha = 0.2`.  
- These policies are enforced in `NodeEngine.merge_with_scope_model()` (or equivalent) to ensure deterministic application.

## 6. High-Level Aggregation Pipeline
### 6.1 Collection with Relaxed Input Freshness
- Every high-level aggregator still requires a submission from each immediate lower-level representative.  
- The submission does **not** need to correspond to the most recent round; the representative simply uploads its latest available model.  
- Aggregators proceed as soon as all representatives have provided *something*, even if those models represent different lower-level rounds. Newer contributions will eventually surface during later intervals.

### 6.2 Merge and Commit
1. **Merge** – The elected aggregator fetches input tensors (cluster models for state, state models for nation) from IPFS, verifies hashes, and runs the scope’s merge algorithm.  
2. **Commit** – That same aggregator publishes the merged tensor to IPFS and anchors the CID/hash on-chain under `/scope/{scope_name}/models`. Peers simply watch the blockchain to observe completion.  
3. **Notification** – Once anchored, nodes requesting the latest `(scope_id, level)` automatically receive the just-committed CID.

### 6.3 Fan-Out Responsibilities
- Every immediate lower scope must designate a fan-out set of nodes responsible for relaying its latest model metadata (CID + hash) to the next higher scope.  
- `system-config.json` adds `fanout_count` (per scope) so administrators can tune redundancy. Example: a state with four clusters and `fanout_count = 2` means each cluster assigns two nodes to report its cluster model to the state aggregator.  
- During each high-level round:
  1. Fan-out nodes fetch the latest CID/hash for their *own* scope (e.g., cluster nodes read the current cluster model anchor).  
  2. Before pushing, each fan-out node pings the elected aggregator’s bridge endpoint. If the node is unreachable, they advance to the next candidate in the round-robin order and repeat the health probe. Only when a candidate responds do they establish the temporary gRPC link and send the metadata. No hub or central clique is involved; the overlay is a ring with these short-lived extra edges. Duplicates remain acceptable and are deduplicated by `(scope_id, cid)` on the aggregator.  
  3. After deduplication, the aggregator expects exactly one unique CID per lower-level scope. Missing scopes trigger the usual timeout logic, but duplicated submissions are normal and provide resilience.  
- Example: a nation round covers four states. Each state has `fanout_count = 2`, so eight fan-out nodes send their state model CIDs/hashes to the nation aggregator. The aggregator reduces these to four unique CIDs, verifies each hash against blockchain, pulls the corresponding models, and merges them into the new nation model.

## 7. Asynchronous Model Retrieval via Blockchain
### 7.1 Level Identification
- Blockchain exposes `GET /model/{scope_id}` (illustrative) so nodes can retrieve the latest CID per level without needing to know round indices.  
- Each node must store the IDs of every high-level scope it belongs to (cluster, state, nation, etc.) and use those IDs when querying.

### 7.2 Merge Tracking
- Nodes store `(scope_id, last_merged_cid)` locally.  
- After the waiting period, nodes call the endpoint; if the returned CID differs from the stored CID, they perform the merge and update the tracker. Otherwise they skip.

### 7.3 Async / Skippable Behavior
- Because retrieval is CID-based instead of round-based, nodes can merge a delayed round later or skip a stale round entirely.  
- Example scenarios now supported:
  - A node merges the nation model from round 3 during the execution window of round 4.  
  - If round 4’s model never completes (timeout), the node simply waits for round 5’s CID and merges that.  
- Eventually every node incorporates high-level information even if they miss intermediate rounds.

## 8. Aggregator Election (Single Aggregator per Round)
- For each high-level round, only one aggregator performs the final commit.  
- Aggregators are assigned in round-robin across the candidate list.  
- If the chosen aggregator fails the fan-out health probes or cannot meet deadlines, the system automatically advances to the next candidate for that same round.  
- There is no inter-aggregator consensus stage; once the active aggregator publishes the model, others accept it.

## 9. Sequence Summary Under the Updated Architecture
1. Cluster rounds proceed continuously; after each round, secure aggregation publishes to blockchain/IPFS as before.  
2. Independent timers trigger state and nation rounds at intervals `t1` and `t2`.  
3. When a high-level round begins, nodes belonging to that scope wait for `wait_seconds`, then query the blockchain endpoint for `(scope_id, level)` to fetch the latest CID.  
4. Aggregators collect the latest available submissions from lower levels, even if representing different round numbers, merge, and commit using the single assigned aggregator.  
5. Nodes retry fetches until they discover a CID they have not yet merged and then apply the configured policy (replace/interpolate/etc.) directly from that scope.  
6. Because merges are logged per-CID, nodes can safely operate asynchronously: they may ingest round `n-1` during round `n`, or skip entirely if a newer CID supersedes it.  
7. The process repeats independently for each level, and new levels can be added by editing `hierarchy_levels` with appropriate IDs, intervals, and merge rules.

This document now serves as the canonical specification for implementing the clarified hierarchy approach: high-level rounds are scheduled by time, nodes remain non-blocking, model dissemination happens via direct pulls keyed by scope IDs, aggregators accept asynchronous inputs, and exactly one aggregator anchors each round using round-robin selection.
