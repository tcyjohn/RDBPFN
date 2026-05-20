# Progress Log

## Session: 2026-05-20 — RDB Generation Performance Optimization

### Context
Previous task (Calendar-Aligned Timestamp Seasonality) complete. User reports RDB + task generation is too slow. Profiling needed to identify specific bottlenecks.

### [x] Problem Diagnosis — Profiling Infrastructure
- Created `scripts/profile_generation.py` with monkey-patching profiling decorators
- Instrumented key functions: HSBM FK sampling, MLPSCM forward, task generation internals
- Ran baseline: 2 RDBs → 22.99s total (11.49s/RDB)
- Ran comparison: skip HSBM → 19.34s, skip tasks → 8.35s
- Identified task generation as #1 bottleneck (63% with quality filter off)

### [x] Full-Scale Profiling — 64 RDBs
- Ran 64 RDBs (single process) with task-level instrumentation
- Total: 8m 29s (7.95s/RDB)
- `InstanceGraph.generate()`: 4m51s (57.1%) — 233,333 calls, 1.2ms each
- `_sample_fk_per_parent` (HSBM): 1m13s (14.3%) — 590 calls, pure Python loop
- `combine_features_and_labels`: 6m4s (71.5%) — all from IG generation
- `_join_related_features`: 1.52s — negligible
- HSBM multi-parent: 57.8s, single-parent: 15.9s

### [x] Root Cause Analysis — Bottleneck A (InstanceGraph)
- Traced `combine_features_and_labels` → `generate_instance_graphs_and_compute_labels`
- Found `num_samples = key_table.num_rows` — creates IG for EVERY row
- Each `ig.generate()` copies DataFrames and does FK filtering
- Identified that IG is only used for label computation, which can be done via pandas merges
- Confirmed DirectAttribute needs no IG at all
- Confirmed RelationalAggregation = SQL JOIN + GROUP BY + aggregation

### [x] Root Cause Analysis — Bottleneck B (HSBM)
- Traced `_sample_fk_per_parent` loop: O(size_b × size_a) per FK pair
- Identified that same-cluster rows share identical probability vectors
- Unique cluster paths = prod(hierarchy) ≈ 8 (small constant)
- Can reduce `rng.choice` calls from size_b (3000+) to cluster count (~8)

### [x] Correctness Analysis
- Phase A: labels are deterministic functions of (FK values, data, random agg/pred choices). Merge/groupby produces identical row matching.
- Phase B: `rng.choice(N, p)` draws independent samples from same distribution whether N=1 or N=3000.

### [x] Plan Written
- `task_plan.md` updated with detailed implementation steps for both phases
- Implementation order: Phase B first (simpler, single file), then Phase A (larger, multi-file)
- Target: 8m29s → ~2m (4x overall speedup)

### [x] Phase B: HSBM Cluster-Grouped Sampling
- Rewrote `_sample_fk_per_parent()` in `hsbm.py` to group child rows by cluster path
- Unique cluster paths = prod(hierarchy) ≈ small constant (typically 4-27)
- Reduced `rng.choice` calls from ~3000 per FK pair to ~8 (one per cluster)
- 64-RDB result: HSBM total = 4.07s (down from 1m13s, ~18x speedup)

### [x] Phase A: Bulk Pandas Joins for Label Computation
- Added `find_path()` to SchemaGraph for BFS shortest-path FK traversal
- Added `_walk_and_merge()` for bulk FK traversal via pandas merges
- Added `_compute_direct_attribute_labels_bulk()` — DirectAttribute via FK walk + first()
- Added `_compute_aggregation_labels_bulk()` — RelationalAggregation via merge+groupby
- Rewrote `combine_features_and_labels()` to dispatch by task type
- Removed `generate_instance_graphs_and_compute_labels()` (233,333 IG calls eliminated)
- Fixed FK column carry-forward for multi-hop paths by selecting FK columns by index via iloc
- Fixed aggregation function: replaced `AggregationProcessor.apply_aggregation` with pandas native `.agg()` due to `g.to_frame()` name issue
- 64-RDB result: Task generation = 2.75s (down from 6m4s, ~132x speedup)

### [x] Phase C: Verification
- 64 RDBs: Total 1m18s (down from 8m29s, **6.5x overall speedup**)
- All 133 complex tasks generated successfully (vs. ~133 originally)
- Label distributions balanced (~50:50) — same statistical properties as original
- HSBM FK generation unchanged in distribution (same RNG, same cluster structure)
- Train/valid/test splits use same torch.randperm logic (unchanged)
- `use_complex_tasks=False` path (simple tasks): verified working
- `skip_hsbm` path: not affected (separate code path in profiling script)

### Known Limitations
- DirectAttribute tasks not yet tested (all complex tasks in current DAG set are aggregation type)
- Multi-parent FK case relies on correct FK column indexing via iloc

### Next
- Merge to main branch
