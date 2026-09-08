# Task Plan

## Completed: AAAI manuscript — RQ2 two-by-two sensitivity-grid revision

### Goal

Integrate the low-temporal-history-density + grouped-sources-disabled result into
the paper without presenting the four independently materialized corpora as a
factorial causal experiment. Keep aggregate, task-level, and mechanism evidence
non-overlapping, and make the seven-page main body visually full before
references begin on page 8.

### Work plan

1. **Analysis pipeline** — add the joint-removal checkpoint, compute all aligned
   task/seed contrasts, and regenerate Figure 2 with R3-vs-R2 and
   R3-vs-joint-removal panels.
2. **Main manuscript** — replace RQ2 and Table 3 with the four-cell sensitivity
   grid; update Figure 2 caption, setup, discussion, and limitations.
3. **Technical supplement** — add aggregate provenance, the 10-by-4 per-seed
   table, conditional intervals, and the descriptive interaction with an
   explicit non-causal boundary.
4. **Verification and layout** — compile both PDFs, audit numbers and
   terminology, render pages, and adjust floats/text so main content fills seven
   pages and references start on page 8.

### Locked decisions

- SA-RDB-PFN means R3.
- Public prose uses temporal-history density, never snapshot language.
- Main Table 3 owns aggregate AUROCs and scale; Figure 2 owns per-task
  heterogeneity; Figures 3–4 retain mechanism diagnostics.
- The fourth setting is “Low density, no grouped sources” in the table and the
  “joint-removal operating point” in prose/captions.
- Conditional gains are sensitivity evidence, not causal main effects or a
  factorial interaction.

## Active: `parent_entity_ids` — Entity-Level FK Bias

### Context

FK bias λ stuck at init (0.1) because FK column stores parent ROW INDEX.
For entity tables with snapshots (S rows per entity), different child rows
connecting to the same parent entity but different snapshots get different
FK values → FK bias sees zero signal (0.015% pair ratio in v6.2).

Root cause: `process_data()` line 888 stores `FK_id` (parent row index) as-is.

### Design

Add `parent_entity_ids` column (shape: `num_rows × num_fk_cols`, int64):
- For entity parents: `parent_entity_ids[i] = parent.entity_ids[fk_value[i]]`
- For non-entity parents: `parent_entity_ids[i] = fk_value[i]` (fallback)
- Stored alongside FK columns, not in feature X
- FK bias uses `parent_entity_ids` for matching instead of raw FK values

Effect: FK pair ratio increases from ~0.015% to ~S× (where S = avg snapshots).

### Files to change

1. **Data generation** (`table_generation.py`):
   - `_materialize_tables_from_pending`: compute `parent_entity_ids` per FK col
   - Store as new column type `PARENT_ENTITY_ID` at end of table data
   - Save metadata for downstream consumption

2. **Metadata** (`table_generation.py`, `dag_to_rdb_generator.py`):
   - Track `parent_entity_cols: List[int]` per table (column indices)
   - Expose through RDB metadata/serialization

3. **H5 merge** (`merge_dbinfer_to_h5.py`):
   - Read `parent_entity_ids` from RDB data (separate from X features)
   - Store in H5 as `parent_entity_ids` dataset (shape: tasks × 600 × max_fk_cols)

4. **Model** (`models.py`):
   - `FKAttentionBias`: optionally use `parent_entity_ids` instead of raw FK
   - Or: rename/refactor to accept entity-level FK grouping keys

5. **Dataloader** (`inmemory_dataloader.py`, `dataloaders.py`):
   - Load `parent_entity_ids` from H5
   - Thread through collate_batch → model

6. **Training/Eval** (`training.py`, `eval.py`, `eval_utils.py`):
   - Thread `parent_entity_ids` through batch and eval

### Validation

1. Generate 32 RDBs, verify `parent_entity_ids` correctness
2. H5 merge, verify FK pair ratio > 5%
3. Train smoke run, verify FK bias λ moves from init

---

## Completed

### v6.2: Child Entity Preference + FK Bias Attempt
- `task_generation.py`: `_has_parent()` + child entity 70% preference in relbench mode
- FK bias still frozen at init (0.015% pair ratio)
- Root cause identified: FK = parent row index, not entity ID

### v6.1: Entity ID Corruption Fix
- `task_generation.py`: exclude entity_id from homophily + task targets
- Entity pair ratio dropped to 0.06% after fix
- Entity bias λ learned in layer 5 only

### Entity Temporal Snapshots
- Entity tables: `num_rows = num_entities × snapshots_per_entity`
- entity_id column: `[0,0,...,0, 1,1,...,1, ...]` per entity
- Same-entity attention bias (EntityAttentionBias)

### Homophily-Controlled Labels
- `homophily.py`: HomophilyLabelGenerator from OPENRFM Appendix G
- Block assignments via HSBM block paths → parent blocks → pseudo-blocks
