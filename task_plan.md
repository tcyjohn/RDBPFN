# Task Plan: Apply Quality Gate + Training Fixes to Target Branch

## Goal

Apply the changes from commit `3732d95` (currently on `feature/hsbm-fk-generation`) to a target branch. Changes include:
1. Task quality gate (`task_quality.py`) integrated into data generation pipeline
2. Complex task fixes (`_join_related_features` for cross-table feature joining)
3. Training eval bug fix (corrupted npz cache recovery)
4. Eval CSV generation script (`convert_rdb_to_csv.py`)
5. Parallel preprocessing in `run_pipeline.sh`

## Source

- Source branch: `feature/hsbm-fk-generation`
- Source commit: `3732d95 feat: add task quality gate, parallel pipeline, and eval CSV generation`
- 7 files changed (+1153/-72)

## Files Changed (3732d95)

| File | Change Type | Lines |
|------|------------|-------|
| `data_generation/RDB/dag_to_rdb_generator.py` | Quality gate integration | +126 |
| `data_generation/RDB/src/table_def/task_generation.py` | Cross-table feature join | +239 |
| `data_generation/RDB/src/table_def/task_quality.py` | NEW: Two-stage quality checker | +420 |
| `model_pretrain/src/eval_utils.py` | Corrupted npz recovery | +24 |
| `scripts/convert_rdb_to_csv.py` | NEW: RDB to CSV converter | +119 |
| `scripts/diagnose_complex_tasks.py` | NEW: CLI quality checker | +127 |
| `scripts/run_pipeline.sh` | Parallel preprocess, QC flags | +170 |

## Phases

### Phase 1: Confirm target branch

- [ ] Clarify: which branch should receive these changes?
  - `main`? `feature/hsbm-fk-no-ts-input`? Other?
  - Changes are already on `feature/hsbm-fk-generation`

### Phase 2: Identify delta between target and source

- [ ] Diff target branch vs `feature/hsbm-fk-generation` for the 7 files
- [ ] Determine if cherry-pick, merge, or manual apply is needed

### Phase 3: Apply changes

- [ ] Apply each file's changes to target branch
- [ ] Ensure imports and dependencies are consistent
- [ ] Verify no regressions in existing functionality

### Phase 4: Validation

- [ ] Verify target branch code parses correctly
- [ ] Run a quick generation test if applicable
