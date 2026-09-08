# parent_entity_ids: Entity-Level FK Bias Signal

## Problem

FK bias λ stuck at init (0.1) because FK column stores parent **row index**.
For entity parents with snapshots S, different child rows connecting to the same
parent entity but different snapshots get **different FK values**. FK bias formula
`λ × I[fk_i == fk_j]` never fires — pair ratio is 0.015% in v6.2.

Root cause: `process_data()` (table_generation.py:888) stores `FK_id` as-is.
`FK_id` is the parent row index from HSBM, ranging [0, parent_rows−1]. An entity
parent with N entities × S snapshots has N×S rows, but children should logically
connect at the **entity** level (N unique identifiers).

## Solution

Add `parent_entity_ids` column — maps each FK value from parent row index to
parent entity identifier.

```
parent_entity_ids[i] = parent_table.entity_ids[fk_value[i]]   (entity parent)
parent_entity_ids[i] = fk_value[i]                           (non-entity, fallback)
```

Effect: all children connecting to ANY snapshot of parent entity E get the same
`parent_entity_ids` value. FK pair ratio scales ~S×.

## Data Flow

```
RDB generation                    H5 merge                      Model
─────────────────                 ─────────                     ─────
_materialize_tables               _prepare_task_sample          FKAttentionBias
  ↓                                 ↓                             ↓
compute parent_entity_ids  →  extract from RDB  →  use parent_entity_ids
per FK column                  store in H5           for I[id_i == id_j] match
store as table column          separate from X        instead of raw FK values
```

## Changes

### 1. Data generation (`table_generation.py`)

**`_materialize_tables_from_pending`**: After FK values are set in `process_data`,
for each FK column j:

```python
parent_name = parent_tables[j]
parent_table = self.tables[parent_name]
if parent_table.entity_column is not None:
    fk_vals = self.tables[child_name].data[:, fk_col_idx]
    parent_eids = parent_table.data[:, parent_table.entity_column]
    parent_entity_ids[:, j] = parent_eids[fk_vals.long()]
else:
    parent_entity_ids[:, j] = fk_vals  # no snapshots, fallback
```

### 2. Table metadata (`table_generation.py`)

- New column type or marker: `parent_entity_indices: List[int]` (one per FK column)
- Stored at end of `data` tensor, after entity_id column if present
- Exposed through RDB serialization metadata

### 3. H5 merge (`merge_dbinfer_to_h5.py`)

- Read `parent_entity_ids` from RDB metadata (separate from X features)
- Store as H5 dataset `parent_entity_ids`: shape (num_tasks, 600, max_fk_cols), int64, −1 pad
- Non-FK tasks: all −1

### 4. FK bias module (`models.py`)

`FKAttentionBias.forward()` accepts optional `parent_entity_ids` parameter.
When provided, computes:

```python
match = (parent_entity_ids[i] == parent_entity_ids[j]) & (parent_entity_ids[i] != -1)
bias = match.float() * softplus(self.raw_lambdas[j])
```

Fallback to raw FK matching when `parent_entity_ids` not provided.

### 5. Dataloader (`inmemory_dataloader.py`, `dataloaders.py`)

- Load `parent_entity_ids` from separate H5 dataset
- `collate_batch`: pad/stack, fill with −1
- Thread through to model.forward()

### 6. Training/Eval (`training.py`, `eval.py`, `eval_utils.py`)

- Thread `parent_entity_ids` through batch dict and eval pipeline
- `load_task_split()` returns `parent_entity_ids` from real benchmark data (if available)

## Validation

1. Generate 32 RDBs → verify `parent_entity_ids` maps correctly to parent entity_id values
2. H5 merge → verify FK pair ratio > 1% (vs 0.015% in v6.2)
3. Smoke training → verify FK bias λ moves from init (−2.252) after a few thousand steps

## Non-goals

- Changing PK/FK semantics in RDB schema
- Modifying HSBM hierarchy or propensity parameters
- Modifying entity bias design
