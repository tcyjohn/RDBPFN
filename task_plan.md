# Task Plan: Entity-Level Aggregation Task Bias

## Goal

Bias focal table selection toward entity/parent tables, so more synthetic tasks match the RelBench pattern (entity-level prediction with aggregated child features). This naturally produces richer cross-source feature correlation in DFS output without modifying the data generation pipeline.

## Background

Current `generate_random_focal_table_and_schema()` selects focal tables uniformly at random, producing ~50% entity-focused (RelBench-style) and ~50% child-focused tasks. RelBench has 100% entity-level tasks. The previous cross-parent correlation efforts (FK propensity, struct_sig, matching latent) targeted the child-focused case where the fundamental bottleneck is independent parent SCM generation. Entity-focused tasks with aggregation naturally have higher cross-source correlation (aggregation functions on same columns are mathematically linked; different child tables share the entity as information channel).

## Phases

### Phase 1 — Entity table classification and focal bias
Status: **complete**

- [x] 1.1 Add `_classify_tables()` method: classify tables as entity (has ≥1 children, i.e., referenced by FK from other tables) vs non-entity
- [x] 1.2 Modify `generate_random_focal_table_and_schema()`: weighted sampling, entity tables at `entity_task_ratio` probability (default 0.75)
- [x] 1.3 When focal is entity, prefer child tables as neighbors in schema graph construction (ensures entity→child structure for aggregation tasks)
- [x] 1.4 Add `entity_task_ratio` HP to `prior_config.py` — set as default in constructor (0.75), threaded through `initialize_tasks_with_complex_tasks()`

### Phase 2 — 64-RDB eval (new task distribution)
Status: pending

- [ ] 2.1 Generate 64 RDBs (seeds 0-63) with entity-biased task generation
- [ ] 2.2 Run preprocessing (DFS via run_pipeline.sh, skip training)
- [ ] 2.3 Evaluate: task type distribution, native_corr, native-joined_corr, joined_corr, cross_corr, cross_agg_corr
- [ ] 2.4 Compare against baseline (random focal selection) and log to findings.md

## Target Metrics

| Metric | Current (random focal) | Target (entity bias) |
|--------|:---:|:---:|
| Entity-focused task % | ~50% | [70%, 85%] |
| cross_agg_corr (DFS) | ~0.05 | [0.10, 0.18] |
| native_corr | ~0.34 | keep ≥ 0.30 |

## Non-goals

- NOT modifying data generation pipeline (SCMs, HSBM, signal-group features)
- NOT modifying DFS feature engineering
- NOT modifying timestamp table assignment (deferred)
- NOT modifying model training or H5 merge
