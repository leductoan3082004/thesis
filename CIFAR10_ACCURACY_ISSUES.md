# CIFAR-10 Low Accuracy Analysis

**Date:** 2026-03-01
**Setup:** 24 nodes, 4 cliques, Dirichlet alpha=0.5, 49 rounds completed
**Observed:** 42.08% global test accuracy, 65.83% train, 57.71% val
**Expected range (literature):** 55-65% for decentralized FL with this config

---

## Issue 1: BatchNorm Statistics Not Aggregated (CRITICAL)

**Symptom:** 23.75% gap between train accuracy (65.83%) and test accuracy (42.08%).

**Root cause:** `CifarConvNet` uses `nn.BatchNorm2d` (3 layers), but the aggregation
pipeline only transfers learnable parameters via `parameters_to_vector()`, which excludes
BatchNorm running statistics (`running_mean`, `running_var`). After each SAP round, nodes
receive aggregated weights but retain their own local BN stats, which are biased toward
each node's non-IID data distribution.

**Files:**
- Model definition: `src/secure_aggregation/communication/node_service.py:101-112`
- Flatten (params only): `src/secure_aggregation/communication/node_service.py:127-130`
- Reload (params only): `src/secure_aggregation/communication/node_service.py:133-136`

**Solution:** Replace `nn.BatchNorm2d` with `nn.GroupNorm` in `CifarConvNet`.
GroupNorm normalizes per-sample (no running statistics), making it FL-compatible.

```python
# Before
nn.BatchNorm2d(32)
nn.BatchNorm2d(64)
nn.BatchNorm2d(128)

# After (8 groups is a common choice)
nn.GroupNorm(8, 32)
nn.GroupNorm(8, 64)
nn.GroupNorm(16, 128)
```

---

## Issue 2: Only 1 Local Epoch Per Round (HIGH)

**Symptom:** Slow convergence, curve still climbing at round 49.

**Root cause:** `local_epochs: 1` means each node processes ~1,666 samples in ~26 batches
per round. This is minimal local computation — the model barely updates before aggregation.
FL benchmarks (NIID-Bench, FedAvg papers) use 5 local epochs as the standard.

**Files:**
- Config: `config/node.config.template.json:16` — `"local_epochs": 1`
- Training loop: `src/secure_aggregation/communication/node_service.py:910-940`

**Solution:** Change `local_epochs` from 1 to 5 in `node.config.template.json`.

```json
"training": {
    "rounds": 3,
    "local_epochs": 5,
    "batch_size": 64
}
```

**Note:** Increasing beyond 10 can worsen client drift with non-IID data. 5 is the
recommended sweet spot from literature.

---

## Issue 3: Learning Rate Too High, No Scheduler (HIGH)

**Symptom:** Sawtooth oscillation visible in per-node accuracy graph.

**Root cause:** `lr=0.1` is hardcoded for SGD. This is standard for centralized training
but too aggressive for FL where each round trains on only ~1,666 non-IID samples. High LR
on skewed data causes large, inconsistent parameter updates across nodes, amplifying client
drift. No learning rate decay means the model never settles.

**Files:**
- Optimizer: `src/secure_aggregation/communication/node_service.py:921`

**Solution:** Reduce LR to 0.01 and add cosine annealing or step decay.

```python
# Before
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# After — make LR configurable and add scheduler
lr = self.training_config.get("lr", 0.01)
opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

for epoch in range(epochs):
    for data, target in self.train_loader:
        # ... existing training loop ...
    scheduler.step()
```

Also add `"lr": 0.01` to the training section of `node.config.template.json`.

---

## Issue 4: State "replace" Policy Discards Local Progress (MEDIUM)

**Symptom:** Periodic accuracy dips when state aggregation fires.

**Root cause:** `apply_policy: "replace"` at state scope completely overwrites each node's
model with the mean of cluster models. While the early-return guard in
`hierarchy_mixin.py:1426` prevents this from firing when child artifacts are missing,
when it does fire, all cluster-level training progress since the last state round is lost.

**Files:**
- Config: `config/system-config.json:33` — `"apply_policy": "replace"`
- Apply logic: `src/secure_aggregation/communication/hierarchy_mixin.py:1213`

**Solution:** Change state apply_policy from `"replace"` to `"interpolate"` with a
moderate alpha (e.g., 0.3).

```json
{
    "scope_name": "state",
    "apply_policy": "interpolate",
    "apply_alpha": 0.3
}
```

This blends 30% state model with 70% local, preserving cluster training momentum.

---

## Issue 5: Over-Conservative Inter-Cluster Merging (MEDIUM)

**Symptom:** Cliques converge to different accuracy levels (34%, 39%, 46%).

**Root cause:** Adaptive gamma formula `gamma = base_gamma / (1 + alpha * avg_disagreement)`
with `base_gamma=0.2` and `alpha=0.5` yields very small mixing weights when disagreement
is high (common early in training). Effective gamma can drop to 0.05-0.10, meaning only
5-10% weight is given to neighbor clusters.

**Files:**
- Formula: `src/secure_aggregation/protocol/inter_cluster.py:82`
- Config: `config/node.config.template.json:43-44`

**Solution:** Increase `base_gamma` from 0.2 to 0.4 to allow more cross-clique learning.

```json
"inter_cluster": {
    "alpha": 0.5,
    "base_gamma": 0.4
}
```

---

## Issue 6: No Data Augmentation (LOW-MEDIUM)

**Symptom:** Overfitting on small, skewed local datasets.

**Root cause:** CIFAR-10 transforms are only `ToTensor()` + `Normalize(0.5, 0.5)`. With
~1,666 train samples per node (some classes severely underrepresented due to Dir 0.5),
the model overfits to local data distribution. Standard CIFAR-10 augmentation adds ~5%
accuracy even in centralized settings.

**Files:**
- Transforms: `src/secure_aggregation/data/datasets.py:98`

**Solution:** Add standard CIFAR-10 augmentation to the training transform pipeline.

```python
# For CIFAR training transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Keep test transforms unchanged (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

**Note:** This requires separating train vs test transforms in the dataset loading path.

---

## Issue 7: Insufficient Rounds (LOW)

**Symptom:** Accuracy curve still climbing at round 49, no plateau.

**Root cause:** 49 rounds with 1 local epoch = only 49 passes over each node's ~1,666
samples. The system has not converged. The round cap is MAX_TRAINING_ROUNDS (default 200),
so the system can run longer.

**Files:**
- Round cap: `src/secure_aggregation/communication/node_service.py:197`

**Solution:** Let the system run to at least 150-200 rounds. After applying fixes 1-4,
convergence should be faster and plateau around 100 rounds.

---

## Summary Table

| # | Issue | Severity | Fix | Expected Impact |
|---|-------|----------|-----|----------------|
| 1 | BatchNorm not aggregated | CRITICAL | Replace with GroupNorm | +5-10% accuracy |
| 2 | 1 local epoch | HIGH | Increase to 5 | +10-15% accuracy |
| 3 | LR=0.1, no scheduler | HIGH | LR=0.01 + cosine decay | +3-5% accuracy |
| 4 | State replace policy | MEDIUM | Switch to interpolate(0.3) | +3-5% accuracy |
| 5 | Conservative inter-cluster gamma | MEDIUM | base_gamma 0.2 -> 0.4 | +2-3% accuracy |
| 6 | No data augmentation | LOW-MEDIUM | Add RandomCrop + HFlip | +2-5% accuracy |
| 7 | Only 49 rounds | LOW | Run to 150-200 rounds | Full convergence |

**Realistic target after all fixes:** 55-65% test accuracy on CIFAR-10 with 24 nodes,
4 cliques, Dir(0.5), D-Cliques topology.

## Implementation Order

Recommended order for incremental testing:

1. **BatchNorm -> GroupNorm** (biggest bang, code change in one file)
2. **local_epochs 1 -> 5** (config change only)
3. **LR 0.1 -> 0.01** (one-line code change, optionally add to config)
4. **Run 150+ rounds** to verify convergence
5. **State replace -> interpolate** (config change)
6. **Add augmentation** (requires train/test transform separation)
7. **Tune base_gamma** (config change, test last)
