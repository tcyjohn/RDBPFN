# Findings

## Commit 3732d95 Analysis

**Commit:** `3732d95 feat: add task quality gate, parallel pipeline, and eval CSV generation`
**Branch:** `feature/hsbm-fk-generation`
**Date:** 2026-05-16

### 1. Quality Gate (`task_quality.py` — NEW)

Two-stage task quality checker based on TabLeak paper (arxiv.org/html/2602.11139v1).

**Stage A (rule-based, instant):**
- a1: label_nunique == 1
- a2: pos_ratio < 0.05 or > 0.95
- a3: n < 64 or min(n_pos, n_neg) < 8
- a4: zero feature_ columns or all constant
- a5: structural independence (schema-based)
- a6: extreme child/parent size ratio (> 100:1)

**Stage B (model-based):**
- ExtraTrees OOF (3-fold, 64 trees, max_depth=12)
- Three-zone AUC: reject ≤0.52, bootstrap (0.52-0.58), accept ≥0.58
- Bootstrap CI on OOF predictions (no retraining)
- Leakage check on FK/ID/timestamp columns

**Integration:** `_generate_tasks_with_quality_gate()` in `dag_to_rdb_generator.py`
- Retries up to `quality_max_retries` times with different seeds
- Keeps best attempt (most passed tasks)
- Failed task schemas pruned from `rdb.task_generation_schemas`

### 2. Complex Task Feature Join (`task_generation.py`)

Added `_join_related_features()` method to `TaskDataGenerator`:
- Traverses schema graph from focal table outward (BFS-like)
- Joins feature columns from related tables via FK relationships
- FK_TO_PK direction: direct merge (feature columns prefixed with table name)
- PK_TO_FK direction: groupby aggregation (mean+std prefixed with table name)
- Column metadata patched after joins via `_get_joined_feature_columns()`

### 3. Training Bug Fix (`eval_utils.py`)

Corrupted `.npz` cache files now trigger regeneration instead of crashing:
```python
try:
    data = np.load(npz_path, allow_pickle=True)
    ...
except Exception as exc:
    print(f"Cached {npz_path} corrupted ({exc}), regenerating...")
    npz_path.unlink(missing_ok=True)
```

### 4. Eval CSV Generation (`convert_rdb_to_csv.py` — NEW)

Converts RDB datasets (npz or parquet) to CSV files for training-time eval `csv_dirs`:
- Handles nested task groups in 4DBInfer metadata
- Combines train/test/validation splits
- Only classification tasks

### 5. Pipeline Script (`run_pipeline.sh`)

Major update:
- 9 positional parameters (was 5)
- Parallel preprocessing with semaphore (NUM_PROCESSES concurrent jobs)
- Quality gate flags passed to generation
- Step 3: H5 merge (with skip option)
- Step 4: Auto-convert first 10 RDBs to eval CSVs
- Step 5: Training

### Branch State

| Branch | State |
|--------|-------|
| `feature/hsbm-fk-generation` (current) | Has all 7 files from 3732d95 |
| `main` | At 0720d1d (Init Commit), very stale |
| `feature/hsbm-fk-no-ts-input` | Old branch, no quality gate |
| `HSBM_time` | Unknown state |
