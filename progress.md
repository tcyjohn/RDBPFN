# Progress Log

## 2026-07-25 — RQ2 joint-removal revision in progress

- Completed a grill-with-docs decision pass for the new low-density,
  no-grouped-sources result.
- Locked a four-cell sensitivity-grid interpretation and non-overlapping
  evidence roles for Table 3 and Figures 2–4.
- Verified the joint-removal operating point: 947 completed RDBs, 2,667 stored
  tasks, 128,403,345 raw cells, mean AUROC 0.6672127449.
- Verified all four conditional differences are positive in each of ten aligned
  support samplings; exact intervals and the descriptive interaction will go to
  the technical supplement.
- Updated the shared domain glossary and added ADRs 0032–0033.
- Integrated the joint-removal checkpoint into the reproducible performance
  analysis and generated separate core, sensitivity, and entity-aware figures.
- Rewrote RQ2 and Table 3 as a four-cell sensitivity grid; the main figure now
  shows R3 versus RDB-PFN-Small and versus joint removal.
- Added per-seed values, four conditional intervals, task-level conditional
  contrasts, and the non-causal descriptive interaction to the technical
  supplement.
- Enlarged and reorganized Figures 3–4 for readability. The main body and
  figures now occupy seven pages, with References beginning on page 8.
- Compiled and visually inspected the main paper and both supplement entry
  points; numerical assertions and terminology audits passed.

## 2026-07-07

### Reverted post-v6.2 code paths while preserving v6.2

- Read `试验记录.csv` and identified v6.2 as the keep point.
- Removed v6.3 HSBM Zipf FK concentration (`hsbm_zipf_alpha`, Zipf weighting, parent entity IDs passed into HSBM sampling).
- Restored v6.2-style `FKAttentionBias` with per-column `raw_lambdas`.
- Removed the same-entity density gate from `EntityAttentionBias`.
- Reverted v6.2.x eval activation changes: no primary-key fallback for `entity_ids`; eval config defaults FK/entity bias to false.
- Preserved v6.2 features: `parent_entity_ids` data/model path, child-entity relbench preference, root target mode, propensity/matching HSBM, entity snapshots, same-entity bias support.

## 2026-06-19

### 1. Thorough pipeline audit → identified FK bias root cause

- Traced v6.1→v6.2 FK bias signal: 0% → 0.015% pair ratio, still frozen at init
- Root cause: `process_data()` stores parent row index as FK value
- Entity snapshots fragment FK values: row k vs k+1 of same entity → different FK values
- HSBM hierarchy parameters don't affect density (ratio invariant within clusters)

### 2. Design decision: add `parent_entity_ids` column

- New column at data generation: maps FK row index → parent entity_id
- FK bias uses parent_entity_ids for entity-level matching
- Expected FK pair ratio: ~S× improvement (0.015% → 1-2%+)

### 3. Child entity preference hack (v6.2)

- Added `_has_parent()` + 70% child entity preference in relbench mode
- Confirmed: 30.8% tasks have FK values, but density still too low for FK bias
- Code kept for potential future use

### 4. Brainstorming complete: parent_entity_ids design approved

- User chose option A (parent_entity_ids) over B (increase child density)
- Name: `parent_entity_ids` (user's choice)
- Computed at data generation time, threaded through H5 → dataloader → model

## 2026-06-15 Session

### FK Sibling Ratio Root Cause Analysis

- FK shared-pair ratio: 0.002% (median=0) in both v6 and v6_20000
- FK sparsity: 80% of H5 samples have zero valid FK values
- HSBM already uses with-replacement sampling (correctly ported from plurel)

### Entity ID corruption fix (v6.1)

- `_maybe_apply_homophily_label` disabled for entity_id columns
- entity_id excluded from task target candidates
- Verified: 0/20 corrupted entity_ids after fix

## 2026-06-08 Session

### Entity temporal snapshots design

- Entity tables: num_rows = num_entities × snapshots_per_entity
- entity_id column: deterministic per-entity across all snapshots

### Homophily label generation (OPENRFM Appendix G)

- Implemented HomophilyLabelGenerator in homophily.py
- get_block_assignments with priority: HSBM block_paths → parent_blocks → pseudo-blocks

### FK bias + entity bias architecture

- FKAttentionBias: λ_j × I[fk_i == fk_j] per FK column
- EntityAttentionBias: λ_se × I[entity_id_i == entity_id_j]
- Integrated into TransformerEncoderLayer as additive attn_mask

### H5 merge stratified entity sampling

- Entity-grouped sampling: K entities → all snapshots → ~2-5% same-entity pair ratio
