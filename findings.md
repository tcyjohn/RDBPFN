# Findings: RDB Generation Performance Optimization

## Profiling Results (64 RDBs, single process, 2026-05-20)

Full breakdown with task-level instrumentation:

```
Total: 8m 29s (per RDB: 7.95s)

initialize_tasks_with_complex_tasks           6m 4s  (71.5%)
  └─ generate_task_data                       6m 4s
       └─ combine_features_and_labels          6m 4s
            └─ generate_instance_graphs_and_compute_labels  6m 4s
                 └─ InstanceGraph.generate()   4m51s  (57.1%)  ← #1 BOTTLENECK
                 └─ _join_related_features     1.52s  ( 0.3%)
            └─ split_task_data                 0.33s
  └─ save_all_task_data                        2.12s

generate_all_data_from_SCM                    2m 1s  (23.8%)
  └─ _sample_fk_per_parent (HSBM inner)       1m13s  (14.3%)  ← #2 BOTTLENECK
  └─ compute_hsbm_fk_ids_multi                 57.8s  (11.3%)
  └─ forward_with_input (MLPSCM)               27.9s  ( 5.5%)
  └─ compute_hsbm_fk_ids                       15.9s  ( 3.1%)
  └─ forward_without_input                     14.8s  ( 2.9%)
  └─ init_table_SCMs                            5.6s
  └─ _materialize_tables_from_pending           3.3s
```

## Root Cause A: InstanceGraph.generate() Called Per Row

**File:** `data_generation/RDB/src/table_def/task_generation.py`

**Call chain:**
```
combine_features_and_labels (line 866)
  → generate_instance_graphs_and_compute_labels (line 736)
    num_samples = key_table.num_rows   ← ALL rows, not a sample
    for each row i:
      ig = InstanceGraph(FocalEntity(table, i), schema_graph)
      ig.generate(rdb)   ← copies DataFrames, does FK filtering
```

**What InstanceGraph.generate() does** (`task_generation_utils.py:221`):
1. Gets focal record from table dataframe
2. For each non-focal table in topological order:
   - Copies the full table DataFrame
   - Filters by FK constraints from parent records

For 3-table schema with 2000 focal rows: 2000 × 2 = 4000 DataFrame copies + filters per task. Across 133 tasks: 233,333 total `ig.generate()` calls.

**What InstanceGraph is used for:**
- `DirectAttributeTarget.compute_label(ig)`: returns `records.iloc[0][column_name]` — just one column of the focal row. Equivalent to `df[column_name]`.
- `RelationalAggregationTarget.compute_aggregated_value(ig)`: gets FK-matched records from target table, applies aggregation. Equivalent to a pandas merge + groupby.

**Key insight:** The entire per-row InstanceGraph traversal is equivalent to a SQL LEFT JOIN along FK edges, followed by GROUP BY + aggregation. Pandas can do this in one pass for all rows simultaneously.

## Root Cause B: HSBM Python Loop Over Every Child Row

**File:** `data_generation/RDB/src/prior/hsbm.py`

`_sample_fk_per_parent()` (line 138):
```python
for b_idx in range(size_b):       # Python loop, ~3000x
    probs = np.ones(size_a)       # allocate, ~3000 elements
    for l in range(num_levels):
        probs *= probs_at_levels[l][cluster_a[:, l], cluster_b[b_idx, l]]
    fk_ids[b_idx] = rng.choice(size_a, p=probs)
```

**Key insight:** Probability vector depends only on `cluster_b[b_idx, :]` — the cluster assignment of child row `b_idx`. All rows in the same cluster path share the identical `probs` vector. Number of unique cluster paths = `prod(hierarchy)` is a small constant (typical: 2³=8).

Instead of computing probs and sampling once per row (3000×), compute probs once per cluster path (~8×) and sample all rows in that cluster in a single `rng.choice(size_a, p=probs, size=N)` call.

## Correctness Analysis

### Phase A (InstanceGraph → Bulk Joins)

| Aspect | Why unchanged |
|--------|--------------|
| Label values | Both approaches match FK values → same rows → same aggregation → same label |
| Random sampling | Same seed, same aggregation functions, same predicate thresholds |
| Feature joining | `_join_related_features` called identically, separate from label computation |
| Edge cases | Empty groups → NaN → fillna(0), same as current `return 0` behavior |

### Phase B (HSBM Cluster Grouping)

| Aspect | Why unchanged |
|--------|--------------|
| Statistical distribution | `rng.choice(N, p=probs)` draws from same categorical(probs) whether N=1 ×3000 or N=3000 ×1 |
| Block structure | Same hierarchy, same probs_at_levels, same cluster assignments → same probability matrices |
| RNG state | Multi-parent: state advances differently but FK values are random draws — distribution preserved |
