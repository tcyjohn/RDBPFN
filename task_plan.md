# Task Plan: RDB Generation Performance Optimization

## Goal

Reduce per-RDB generation time by optimizing two independently-verified bottlenecks without changing output correctness.

Target: ~8s → ~2-3s per RDB (from profiling baseline of 64 RDBs / 8m29s).

**STATUS: COMPLETE (2026-05-20). Actual result: 8m29s → 1m18s (6.5x speedup, 1.22s/RDB).**

## Profiling Baseline (64 RDBs, single process)

```
Total: 8m 29s (per RDB: 7.95s)
  Task generation:     6m 4s  (71.5%)  ← Phase A
    InstanceGraph.gen: 4m51s  (57.1%)
  Data generation:     2m 1s  (23.8%)  ← Phase B
    HSBM FK sampling:  1m13s  (14.3%)
    MLPSCM forward:      47s  ( 9.2%)
```

---

## Phase A — Eliminate Per-Row InstanceGraph Construction (57% → ~0%)

### A.1 Root Cause

`generate_instance_graphs_and_compute_labels()` at `task_generation.py:736` sets `num_samples = key_table.num_rows`, constructing one `InstanceGraph` per row. Each `ig.generate()` (at `task_generation_utils.py:221`) copies pandas DataFrames and does FK filtering. For a 2000-row table: 2000 × (3 DataFrames copied + FK filtered) = 6000 pandas operations per task × 133 tasks = 233,333 total calls.

### A.2 Optimization: Bulk Pandas Joins

**File: `data_generation/RDB/src/table_def/task_generation.py`**

#### Step 1 — Split `combine_features_and_labels()` by task type (line 854-868)

Current code blindly calls `generate_instance_graphs_and_compute_labels` for both `DIRECT_ATTRIBUTE_PREDICTION` and `RELATIONAL_AGGREGATION_PREDICTION`. Change to:

- **DirectAttribute**: Extract label directly from `key_table.dataframe[target_column]`. No IG needed. The `compute_label()` for DirectAttribute just returns `records.iloc[0][column_name]` — same as reading the column directly.

  ```python
  # Pseudocode for DirectAttribute
  unified_df = key_table.dataframe[feature_columns].copy()
  unified_df[task.real_name_for_target_column] = key_table.dataframe[task.target_column]
  unified_df = self._join_related_features(unified_df, key_table_name, task.schema_graph)
  ```

- **RelationalAggregation**: Replace per-row IG with a bulk pandas merge/groupby pipeline. The key insight: InstanceGraph FK traversal = SQL LEFT JOIN along schema graph edges.

#### Step 2 — New method: `_compute_aggregation_labels_bulk(task, key_table, ...)` 

For RelationalAggregation target, the current per-row flow is:
```
for each row r in focal_table:
    ig = build_instance_graph(r, schema)      # FK traversal
    records = ig.get_records(target_table)     # get matched rows
    value = agg_func(records[agg_column])     # aggregate
    label = predicate(value)                  # threshold
```

This is equivalent to the following bulk pipeline (single pass, all rows):

```python
def _compute_aggregation_labels_bulk(self, task, key_table_name, key_table):
    """
    1. Traverse FK path from focal_table → target_table via pandas merges
    2. Group by focal PK, compute aggregate
    3. Compute median-based threshold, apply predicate
    """
    focal_df = key_table.dataframe.copy()
    focal_pk = key_table.column_names[0]  # PK is always column 0
    
    # Traverse schema graph to find path from focal → target
    target_table_name = task.target_computation.target_node_set.table_name
    path = task.schema_graph.find_path(key_table_name, target_table_name)
    
    if not path:
        # Single table aggregation (target_table == focal_table)
        merged = focal_df
    else:
        # Walk FK edges and merge
        merged = self._walk_and_merge(focal_df, key_table_name, path, task.schema_graph)
    
    # Group by focal PK, compute aggregate
    agg_col = task.target_computation.aggregation_column
    agg_func = task.target_computation.aggregation_func
    grouped = merged.groupby(focal_pk)[agg_col].agg(agg_func.name.lower())
    
    # Predicate: use median as threshold (same as current code)
    threshold = float(np.median(grouped.values))
    labels = (grouped > threshold).astype(int)
    
    # Build result: focal features + labels
    result = focal_df.set_index(focal_pk)
    result[task.real_name_for_target_column] = labels.reindex(result.index)
    return result.reset_index(drop=True)
```

#### Step 3 — New method: `_walk_and_merge(df, start_table, path, schema_graph)`

Traverse the FK edges in `path` sequentially, doing pandas merges:

```python
def _walk_and_merge(self, df, start_table, path, schema_graph):
    """Walk FK edges from start_table through path, merging at each step."""
    current_df = df
    current_table = start_table
    for next_table in path:
        edge = schema_graph.get_edge(current_table, next_table)
        if edge is None:
            edge = schema_graph.get_edge(next_table, current_table)
        if edge.direction == PK_TO_FK:
            # current PK → next FK: merge current.PK = next.FK
            pk_col = self.rdb.tables[current_table].column_names[0]
            fk_col = self.rdb.tables[next_table].column_names[edge.to_column]
            next_df = self.rdb.tables[next_table].dataframe
            current_df = current_df.merge(next_df, left_on=pk_col, right_on=fk_col)
        else:  # FK_TO_PK
            # current FK → next PK: merge current.FK = next.PK
            fk_col = self.rdb.tables[current_table].column_names[edge.from_column]
            pk_col = self.rdb.tables[next_table].column_names[0]
            next_df = self.rdb.tables[next_table].dataframe
            current_df = current_df.merge(next_df, left_on=fk_col, right_on=pk_col)
        current_table = next_table
    return current_df
```

#### Step 4 — Handle retry logic (lines 756-818)

Current code retries up to `max_retries` times if labels collapse to a single class. The bulk approach computes all labels in one pass, so retry just means: re-sample aggregation column/function/predicate, re-run `_compute_aggregation_labels_bulk`, check label diversity.

#### Step 5 — Add `find_path()` to SchemaGraph (in `task_generation_utils.py`)

Simple BFS shortest path between two table nodes.

### A.3 Correctness Proof

| Current behavior | New behavior | Why equivalent |
|---|---|---|
| `ig.generate()`: for row r, copy DataFrame T, filter `T[FK_col].isin({r.PK})` | `merge(focal_df, T_df, left_on=PK, right_on=FK)` | Both match rows where FK = PK. Merge matches ALL rows at once. |
| `compute_aggregated_value(ig)`: `agg_func(records[agg_col])` | `groupby(focal_pk)[agg_col].agg(agg_func)` | Same aggregation, applied per-group (one group per focal row). |
| `predicate_func.apply(val)` with median threshold | `grouped > median(grouped)` | Same threshold computation, same application. |
| DirectAttribute: `records.iloc[0][target_column]` | `df[target_column]` | Focal record of InstanceGraph IS the row from the DataFrame. Bit-identical. |

**Key invariant**: The current code constructs InstanceGraph by FK-filtering DataFrames. Pandas merge with equal FK/PK values produces the exact same row matching. The aggregation function (mean/std/sum/max) is deterministic given the same input rows.

### A.4 Risk / Edge Cases

- **Multi-hop bridge joins** (lines 616-696): The current `_join_related_features` has a complex bridge-join path for 2-hop schemas. If this path is needed for label computation (not just feature joining), `_walk_and_merge` must handle it. But the label computation only aggregates over `TargetNodeSet.table_name` — which in practice is always the focal table or a direct neighbor. The bridge logic is only for **feature** joining, not label computation. Verify this.

- **Empty groups**: Some focal rows may have no matching target records. Current code returns 0 (via `AggregationProcessor`). Bulk approach: `groupby.agg()` produces NaN for empty groups → fillna(0).

- **Remove `generate_instance_graphs_and_compute_labels` after migration**: If no other caller uses it, delete. If there are other callers, assess whether they also benefit from the bulk approach.

---

## Phase B — Vectorize HSBM Per-Row Sampling Loop (14% → ~1%)

### B.1 Root Cause

`_sample_fk_per_parent()` at `hsbm.py:129` loops over every child row in Python:

```python
for b_idx in range(size_b):            # ~3000 iterations
    probs = np.ones(size_a, dtype=np.float64)  # allocate ~3000-length array
    for l_idx in range(len(probs_at_levels)):
        probs *= probs_at_levels[l_idx][cluster_a[:, l_idx], cluster_b[b_idx, l_idx]]
    fk_ids[b_idx] = rng.choice(size_a, p=probs)  # single sample
```

O(child_rows × parent_rows × num_levels), with Python-level loop overhead.

### B.2 Optimization: Cluster-Grouped Sampling

**File: `data_generation/RDB/src/prior/hsbm.py`**

#### Key insight

Child rows in the same cluster path (at all hierarchy levels) share the **identical** probability distribution over parent rows:

```
probs_at_levels[l][cluster_a[:, l], cluster_b[b_idx, l]]
```

The probability depends only on `cluster_b[b_idx, l]` — which is the same for all rows sharing the same cluster assignment. Number of unique cluster paths = `prod(hierarchy_child)`, which is a small constant (typically 4–27).

#### Step 1 — Rewrite `_sample_fk_per_parent()`

```python
def _sample_fk_per_parent(
    size_a: int,
    size_b: int,
    cluster_a: np.ndarray,    # (size_a, num_levels)
    cluster_b: np.ndarray,    # (size_b, num_levels)
    probs_at_levels: list,
    rng: np.random.RandomState,
) -> np.ndarray:
    num_levels = len(probs_at_levels)
    fk_ids = np.empty(size_b, dtype=np.int64)
    
    # Group child rows by cluster path (tuple of cluster indices across levels)
    cluster_groups: dict[tuple, np.ndarray] = {}
    for b_idx in range(size_b):
        key = tuple(cluster_b[b_idx, l] for l in range(num_levels))
        cluster_groups.setdefault(key, []).append(b_idx)
    
    # For each cluster group, compute probs once, sample all at once
    for cluster_path, indices in cluster_groups.items():
        indices_arr = np.array(indices)
        n_in_cluster = len(indices_arr)
        
        # Compute probability vector (same for all rows in this cluster)
        probs = np.ones(size_a, dtype=np.float64)
        for l in range(num_levels):
            probs *= probs_at_levels[l][cluster_a[:, l], cluster_path[l]]
        p_sum = probs.sum()
        if p_sum > 0:
            probs /= p_sum
        else:
            probs = None
        
        # Sample all rows in cluster at once
        sampled = rng.choice(size_a, size=n_in_cluster, p=probs)
        fk_ids[indices_arr] = sampled
    
    return fk_ids
```

#### Step 2 — Multi-parent case (`compute_hsbm_fk_ids_multi`)

For the multi-parent case, `cluster_b` is randomly sampled per child row (line 221-222 of hsbm.py). The same optimization applies: group by cluster path, sample per group. Each parent reads its own prefix of the shared cluster_b.

No structural change needed — `_sample_fk_per_parent` is called once per parent, and the optimization handles it naturally.

### B.3 Correctness Proof

| Property | Current | New | Why equivalent |
|---|---|---|---|
| Distribution per sample | `rng.choice(size_a, p=probs)` | `rng.choice(size_a, p=probs, size=N)` | Both draw independent samples from categorical(probs) |
| RNG advancement | N calls consume N RNG values | 1 call with size=N consumes N RNG values | Same number of RNG draws, same distribution |
| Cluster probs | Computed per-row (redundantly) | Computed once per cluster | Same probs vector (mathematically identical) |
| Multi-parent RNG sharing | Parent 1 draws, then Parent 2 draws from same RNG state | Parent 1 draws differently (contiguous per cluster), Parent 2 state differs | FK values are random by design. RNG state after generation is irrelevant. |

**Critical detail on RNG state**: In multi-parent mode, after the optimization, parent 1's RNG draws are contiguous within each cluster instead of interleaved with different clusters. This changes the RNG state at which parent 2 starts. However, the FK values themselves are **random by design** — the HSBM process is a random graph model, and any particular FK assignment is just one valid realization. Changing the RNG order is equivalent to using a different random seed. The statistical distribution of the bipartite graph (block structure, cluster probabilities) is preserved.

### B.4 Alternative: Also remove the Python grouping loop

The grouping loop (`for b_idx in range(size_b)`) to form `cluster_groups` is O(size_b) which is acceptable (3000 iterations of a dict lookup, very fast). The real savings come from reducing `rng.choice` calls from ~3000 to ~8.

Alternative approach for the grouping step (even faster):
```python
# Use numpy to encode cluster paths as integers
multipliers = np.cumprod([1] + hierarchy_child[:-1])
cluster_keys = (cluster_b * multipliers).sum(axis=1)  # (size_b,) unique cluster ID
for key in np.unique(cluster_keys):
    indices = np.where(cluster_keys == key)[0]
    ...
```
This avoids the Python loop entirely.

---

## Phase C — Verification

### C.1 Correctness Test

1. Generate 64 RDBs with `run_pipeline.sh` **before** changes → save as baseline
2. Apply both optimizations
3. Generate same 64 RDBs with same seeds → compare:
   - **For Phase A**: labels should be identical (bit-exact match)
   - **For Phase B**: FK distributions should have same statistical properties (cluster sizes, per-cluster FK diversity). Exact FK values may differ due to RNG reordering, but block structure must match.
4. Run full pipeline (64 RDBs, preprocess, merge, train 1 epoch) → training loss should be in the same range

### C.2 Performance Test

Run `scripts/profile_generation.py` with 64 RDBs after changes:
- Expect task generation (Phase A): 6m4s → ~30s (12x speedup)
- Expect HSBM (Phase B): 1m13s → ~5s (15x speedup)
- Total: 8m29s → ~2m (4x overall speedup)

### C.3 Regression Test

- `use_complex_tasks=False` path (simple tasks): ensure still works
- `skip_hsbm` path: ensure still works
- Multi-parent FKs: ensure joint sampling still produces valid FK references

---

## Implementation Order

1. **Phase B first** — smaller, isolated change in a single file (`hsbm.py`), easy to test
2. **Phase A second** — larger change across `task_generation.py` + `task_generation_utils.py`, more complex logic
3. **Phase C** — run verification on both together
