# parent_entity_ids Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `parent_entity_ids` to H5 merge and thread through model so FK bias matches at entity level instead of row level.

**Architecture:** Compute `parent_entity_ids` lazily in H5 merge step (no data generation changes). FKAttentionBias accepts optional `parent_entity_ids`; when provided, uses it for matching. Dataloader loads from H5, training/eval thread through batch.

**Tech Stack:** Python, PyTorch, h5py, numpy

## Global Constraints

- No changes to RDB data generation or serialization format
- No changes to table schema (PK/FK semantics unchanged)
- FK bias falls back to raw FK matching when parent_entity_ids unavailable
- Entity bias design unchanged

---

### Task 1: Compute parent_entity_ids in H5 merge

**Files:**
- Modify: `data_preprocessing/merge_dbinfer_to_h5.py:_prepare_task_sample`

**Interfaces:**
- Consumes: `task` (DBBRDBTask) with train_set/test_set containing FK columns and parent table data
- Produces: `parent_entity_ids_sampled: np.ndarray | None` shape `(total_rows, num_fk_cols)`, int64, −1 pad

- [ ] **Step 1: Add helper to compute parent_entity_ids**

In `merge_dbinfer_to_h5.py`, add a module-level helper. The key insight: DFS-preprocessed parent table parquet files are in the same directory as the task table data. Load them to extract `entity_id` column for FK→entity mapping.

```python
def _compute_parent_entity_ids(
    task: DBBRDBTask,
    fk_column_name: str,
    dataset: DBBRDBDataset,
) -> np.ndarray | None:
    """Map FK row-index values to parent entity_id values.

    Loads the parent table's DFS-preprocessed parquet file from the same
    dataset directory to extract the entity_id column, then maps::

        parent_entity_ids[i] = parent.entity_ids[fk_value[i]]

    Returns None when parent table has no entity_id (non-entity or not found).
    """
    # Resolve parent table name from FK column metadata
    task_table_name = task.metadata.target_table
    task_table_meta = task.metadata.tables[task_table_name]
    parent_table_name = None
    for col_meta in task_table_meta.columns:
        if col_meta.name == fk_column_name and getattr(col_meta, "link_to", None):
            parent_table_name = col_meta.link_to.split(".")[0]
            break

    if parent_table_name is None:
        return None

    # DBBRDBDataset already loaded all table data
    parent_table_data = dataset.tables.get(parent_table_name)
    if parent_table_data is None:
        return None
    if "entity_id" not in parent_table_data:
        return None

    parent_eids = parent_table_data["entity_id"].astype(np.int64)

    # Map train FK values
    fk_train = task.train_set[fk_column_name].astype(np.float64)
    fk_train = np.nan_to_num(fk_train, nan=-1).astype(np.int64)
    peids_train = np.full(len(fk_train), -1, dtype=np.int64)
    valid_train = (fk_train >= 0) & (fk_train < len(parent_eids))
    peids_train[valid_train] = parent_eids[fk_train[valid_train]]

    # Map test FK values
    fk_test = task.test_set[fk_column_name].astype(np.float64)
    fk_test = np.nan_to_num(fk_test, nan=-1).astype(np.int64)
    peids_test = np.full(len(fk_test), -1, dtype=np.int64)
    valid_test = (fk_test >= 0) & (fk_test < len(parent_eids))
    peids_test[valid_test] = parent_eids[fk_test[valid_test]]

    return np.concatenate([peids_train, peids_test], axis=0)
```

- [ ] **Step 2: Call helper in _prepare_task_sample**

After FK extraction (line ~323) and before sampling (line ~339), add:

```python
# --- Compute parent_entity_ids (entity-level FK for attention bias) ---
parent_entity_ids_list = []
for fk_name in fk_column_names:
    peids = _compute_parent_entity_ids(task, fk_name, dataset)
    if peids is not None:
        parent_entity_ids_list.append(peids)
```

Pass the `dataset` (DBBRDBDataset) object to `_prepare_task_sample` — it already has `dataset.tables[parent_table_name]` with all table data loaded.

- [ ] **Step 3: Sample parent_entity_ids with same indices**

After sampling indices (line ~340-355):

```python
parent_entity_ids_sampled = None
if parent_entity_ids_list:
    peids_combined = np.stack(parent_entity_ids_list, axis=1)
    parent_entity_ids_sampled = peids_combined[indices]
```

- [ ] **Step 4: Return parent_entity_ids in result dict**

Add to the result dict at end of `_prepare_task_sample`:

```python
"parent_entity_ids": parent_entity_ids_sampled,
```

- [ ] **Step 5: Write to H5 in _write_hdf5**

After sampling indices (line ~340-355):

```python
parent_entity_ids_sampled = None
if parent_entity_ids_list:
    peids_combined = np.stack(parent_entity_ids_list, axis=1)
    parent_entity_ids_sampled = peids_combined[indices]
```

- [ ] **Step 5: Write to H5 in _write_hdf5**

Following the same pattern as `fk_values`, add `parent_entity_ids` as an H5 dataset:

```python
# In _write_hdf5, alongside fk_values writing:
all_peids = []
for sample in sorted_keys:
    peids = all_results[sample].get("parent_entity_ids")
    if peids is not None:
        all_peids.append(peids)

if all_peids:
    max_fk_cols = max(p.shape[1] for p in all_peids)
    peids_arr = np.full((len(all_peids), total_rows, max_fk_cols), -1, dtype=np.int64)
    for i, p in enumerate(all_peids):
        peids_arr[i, :, :p.shape[1]] = p
    h5f.create_dataset("parent_entity_ids", data=peids_arr)
```

- [ ] **Step 6: Verify with manual merge test**

Run merge on a few v6.2 RDBs and verify:
- `parent_entity_ids` values correctly map to parent entity_id
- FK pair ratio > 1% (vs 0.015% in v6.2)

---

### Task 2: Modify FKAttentionBias to accept parent_entity_ids

**Files:**
- Modify: `model_pretrain/src/models.py:FKAttentionBias.forward`

**Interfaces:**
- Consumes: `parent_entity_ids: torch.Tensor | None` — shape `(B, total_rows, K)`, int64, −1 pad
- Produces: `(bias_left, bias_right)` — same as current forward, but uses parent_entity_ids for matching when provided

- [ ] **Step 1: Add parent_entity_ids parameter to forward**

```python
def forward(
    self,
    fk_values: torch.Tensor,
    train_rows: int,
    parent_entity_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
```

- [ ] **Step 2: Use parent_entity_ids for matching when provided**

```python
if fk_values is None or fk_values.shape[-1] == 0:
    return None, None

B, total_rows, K = fk_values.shape
test_rows = total_rows - train_rows
lambdas = self._lambdas_for(K)

# Use entity-level keys when available, raw FK otherwise
match_keys = parent_entity_ids if parent_entity_ids is not None else fk_values

keys_train = match_keys[:, :train_rows, :]  # (B, tr, K)
keys_test = match_keys[:, train_rows:, :]    # (B, te, K)

valid_train = (keys_train >= 0)
valid_test = (keys_test >= 0)

# bias_left: (B, tr, tr)
match_left = (keys_train.unsqueeze(2) == keys_train.unsqueeze(1)).float()
both_valid_left = (valid_train.unsqueeze(2) & valid_train.unsqueeze(1)).float()
bias_left = ((match_left * both_valid_left) * lambdas).sum(dim=-1)

# bias_right: (B, te, tr)
match_right = (keys_test.unsqueeze(2) == keys_train.unsqueeze(1)).float()
both_valid_right = (valid_test.unsqueeze(2) & valid_train.unsqueeze(1)).float()
bias_right = ((match_right * both_valid_right) * lambdas).sum(dim=-1)

return bias_left, bias_right
```

- [ ] **Step 3: Thread parent_entity_ids through TransformerEncoderLayer.forward**

Add `parent_entity_ids: torch.Tensor | None = None` parameter. Pass to `self.fk_bias(..., parent_entity_ids=parent_entity_ids)`.

- [ ] **Step 4: Thread through NanoTabPFNModel.forward**

Add `parent_entity_ids: torch.Tensor | None = None` parameter. Pass to each layer's forward call.

- [ ] **Step 5: Verify FK bias can accept parent_entity_ids without breaking existing callers**

The new parameter has default `None` → falls back to existing raw FK matching. All existing callers work unchanged.

---

### Task 3: Load parent_entity_ids in dataloader

**Files:**
- Modify: `model_pretrain/src/inmemory_dataloader.py`

**Interfaces:**
- Consumes: H5 file with optional `parent_entity_ids` dataset
- Produces: `parent_entity_ids` tensor in each sample dict

- [ ] **Step 1: Load parent_entity_ids from H5 in InMemoryDataset.__init__**

```python
self.parent_entity_ids_ds = (
    self._file["parent_entity_ids"] if "parent_entity_ids" in self._file else None
)
```

- [ ] **Step 2: Return parent_entity_ids in __getitem__**

```python
if self.parent_entity_ids_ds is not None:
    peids_np = self.parent_entity_ids_ds[idx, :num_rows, :]
    parent_entity_ids = torch.from_numpy(peids_np.astype(np.int64))
else:
    parent_entity_ids = torch.zeros(num_rows, 0, dtype=torch.long)

# Add to sample dict:
sample["parent_entity_ids"] = parent_entity_ids
```

- [ ] **Step 3: Pad/stack in collate_batch**

Follow the same pattern as `fk_values` (lines 148-167): detect if any sample has parent_entity_ids, stack with padding (−1 fill).

```python
has_peids = any(
    s.get("parent_entity_ids") is not None and s["parent_entity_ids"].shape[-1] > 0
    for s in samples
)
if has_peids:
    max_fk = max(s.get("parent_entity_ids", torch.zeros(0, 0)).shape[-1] for s in samples)
    peids_parts = []
    for s in samples:
        p = s.get("parent_entity_ids")
        if p is None or p.numel() == 0:
            p = torch.full((num_rows, max_fk), -1, dtype=torch.long)
        elif p.shape[-1] < max_fk:
            pad = torch.full((p.shape[0], max_fk - p.shape[-1]), -1, dtype=torch.long)
            p = torch.cat([p, pad], dim=-1)
        peids_parts.append(p)
    collated["parent_entity_ids"] = torch.stack(peids_parts, dim=0)
```

---

### Task 4: Thread parent_entity_ids through training

**Files:**
- Modify: `model_pretrain/src/training.py`

**Interfaces:**
- Consumes: `parent_entity_ids` from batch dict
- Produces: passed to `model.forward()`

- [ ] **Step 1: Extract from batch in _compute_batch_loss / training loop**

In `_compute_batch_loss` (or equivalent function), extract parent_entity_ids from batch dict:

```python
parent_entity_ids = full_data.get("parent_entity_ids")
```

- [ ] **Step 2: Pass to model forward**

Thread through the existing call chain:
```python
model(X, y, ..., fk_values=fk_values, entity_ids=entity_ids, parent_entity_ids=parent_entity_ids)
```

- [ ] **Step 3: Thread through _compute_losses_with_augmentations**

If augmentations are used, parent_entity_ids must be passed through the same slicing as fk_values.

---

### Task 5: Thread parent_entity_ids through eval

**Files:**
- Modify: `model_pretrain/src/eval_utils.py`
- Modify: `model_pretrain/src/eval.py`

**Interfaces:**
- `load_task_split` returns 5th element: `parent_entity_ids: np.ndarray | None`

- [ ] **Step 1: Update load_task_split return signature**

```python
def load_task_split(task, split) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    # ... existing X, y, fk_values, entity_ids extraction ...

    # Extract parent_entity_ids
    parent_entity_ids = None
    peids_cols = [c.name for c in task.table.columns if ...]  # find parent_entity columns
    # ... extract and return

    return X, y, fk_values, entity_ids, parent_entity_ids
```

- [ ] **Step 2: Update eval.py callers to unpack 5th element**

- [ ] **Step 3: Pass parent_entity_ids to classifier.fit/predict_proba**

---

### Task 6: Enable both biases in training config

**Files:**
- Modify: `model_pretrain/conf_train/RDBPFN_hsbm.yaml`

- [ ] **Step 1: Enable FK and entity bias in config**

```yaml
model:
  use_fk_bias: true
  use_entity_bias: true
```

---

### Task 7: Smoke test — generate 32 RDBs + H5 merge + verify

**Files:**
- None (verify only)

- [ ] **Step 1: Run pipeline (gen only)**

```bash
bash scripts/run_pipeline.sh 32 0 v6.3_parenteid false true true false 3 16 true true "0" true
```

- [ ] **Step 2: Verify parent_entity_ids in H5**

```python
import h5py
f = h5py.File("model_pretrain/pretrain_datasets/v6.3_parenteid.h5", "r")
peids = f["parent_entity_ids"][:]
# Check: same parent entity → same parent_entity_ids value
# Check: FK pair ratio > 1%
```

- [ ] **Step 3: Verify FK pair ratio improvement**

Expected: FK pair ratio > 1% (vs 0.015% in v6.2).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add parent_entity_ids for entity-level FK bias matching"
```
