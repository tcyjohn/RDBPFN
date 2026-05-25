# Progress Log

## Session: 2026-05-25 — FK Matching Latent (COMPLETE)

### Design
- Delete struct_sig (failed post-hoc injection)
- Single-parent: Restore FK Propensity (Section 12)
- Multi-parent: Shared matching latent via cross-parent W projection

### Steps
1. ✅ Delete struct_sig code (4 files cleaned)
2. ✅ 64-RDB baseline (evaluated, results in findings.md)
3. ✅ Restore propensity (hsbm.py + prior_config.py + table_generation.py)
4. ✅ Implement matching latent (hsbm.py + prior_config.py + table_generation.py)
5. ✅ 64-RDB final eval (results in findings.md)

---

## Session: 2026-05-25 — Entity-Level Aggregation Task Bias (CURRENT)

### Context
Research concluded cross-parent correlation gap (~0.03 vs 0.21) is ~70% measurement artifact (aggregation vs join), ~30% genuine (collider bias). Real RelBench tasks are all entity-level aggregation. Biasing focal table selection toward entity tables naturally produces richer cross-source correlation in DFS output.

### Changes (Phase 1 complete)
- `task_generation.py`: Added `_classify_tables()`, `_has_children()`, `_get_children()` helpers
- `task_generation.py`: Added `entity_task_ratio` param to `TaskGenerator.__init__` (default 0.75)
- `task_generation.py`: Modified `generate_random_focal_table_and_schema()` with weighted entity-table selection
- `table_generation.py`: Added `create_multi_hop_schema_graph_biased()` with child-preferring first-hop neighbor selection
- `table_generation.py`: Threaded `entity_task_ratio` through `initialize_tasks_with_complex_tasks()`
- `task_plan.md`, `findings.md`: Updated with new plan and research findings

### Files modified
- `data_generation/RDB/src/table_def/task_generation.py` (+60 lines)
- `data_generation/RDB/src/table_def/table_generation.py` (+90 lines)
- `task_plan.md`, `findings.md`, `progress.md`
