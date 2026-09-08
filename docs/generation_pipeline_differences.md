# Generation Pipeline Differences: Modified Repo vs Original RDBPFN Main

Comparison baseline:

- Modified repo: current working tree on `feature/hsbm-temporal-refactor`.
- Original repo: `origin/main` fetched on 2026-07-03, commit `0720d1d167256d9219cd993ac7b2c6c07f3b639d`, read from `/data/caijunyu/RDBPFN-main`.
- Scope: generation pipeline only. Training, model architecture, and evaluation code are excluded except where a generation artifact exists specifically to support them.

## Executive Summary

The modified repository changes the synthetic data generator from an MLP-centric row sampler into a structural relational generator. The largest changes are:

1. Explicit entity tables with temporal snapshots and `entity_id`.
2. HSBM-based FK generation outside the MLP, including multi-parent shared latent paths.
3. Propensity and matching-latent FK reweighting.
4. Signal-group feature generation using time, parent, HSBM path, and intrinsic signals.
5. Calendar-aligned timestamp generation with shared event calendars.
6. RelBench-style entity-focal task generation.
7. Optional homophily labels and quality filtering.
8. New diagnostics and full pipeline scripts.

## 1. File-Level Additions And Major Edits

### Added In Modified Repo

- `data_generation/RDB/src/prior/hsbm.py`
  - New HSBM FK generation module.
  - Provides single-parent, multi-parent, propensity, and matching-latent FK sampling.

- `data_generation/RDB/src/table_def/task_quality.py`
  - New task quality checker and feature-quality diagnostics.
  - Used by the generation quality gate.

- `docs/data_generation_pipeline.md`
  - Existing generated documentation for the modified pipeline.

- `docs/signal-group-feature-gen-journey.md`
  - Development record for signal-group feature generation.

- `scripts/convert_dag_rdb.py`
  - Converts generated 4DBInfer-style data to RelBench layout.

- `scripts/generateRDB_hsbm_v2.sh`
  - Raw generation plus RelBench conversion/preprocessing helper.

- `scripts/run_pipeline.sh`
  - End-to-end orchestration for generation, preprocessing, H5 merge, and optional training.

### Major Edited Generation Files

- `dag_to_rdb_generator.py`
- `src/table_def/table_generation.py`
- `src/table_def/task_generation.py`
- `src/table_def/task_generation_utils.py`
- `src/prior/mlp_scm.py`
- `src/prior/prior_config.py`
- `src/prior/temporal_vocab.py`
- `src/prior/hp_sampling.py`
- `src/prior/utils.py`
- `dag_to_rdb_config_small.yaml`

## 2. Input And CLI Differences

Original main:

- CLI supports core generation settings: number of RDBs, start index, output directory, config file, process count, complex tasks, row GNN.
- Original `RDB_generate.sh` directly launches large generation batches.

Modified repo:

- Adds CLI flags for generation behavior:
  - `--relbench_mode`
  - `--use_homophily_labels`
  - `--no_path_signal`
  - `--snapshots_per_entity_min`
  - `--snapshots_per_entity_max`
  - `--entity_timestamp_prob`
  - `--no-quality-filter`
  - `--quality-max-retries`
  - `--snr-threshold`
- Adds script-level control over preprocessing, H5 merge, FK/entity-bias experiments, snapshot range, and RelBench conversion.

Impact:

- Original main has one relatively fixed data-generation mode.
- Modified repo exposes generation regimes as experimental controls, especially entity density, task distribution, path signal, and label homophily.

## 3. Dimension Config Differences

Original main:

- `dag_to_rdb_config_small.yaml` only configures row and column scaling.
- Default config has no timestamp or entity-table section.

Modified repo:

- `dag_to_rdb_config_small.yaml` adds:
  - `timestamp.activity_prob`
  - legacy `timestamp.entity_prob`
- `DAGToRDBGenerator.DEFAULT_DIMENSION_CONFIG` adds:
  - `timestamp.prob`
  - `entity_table.snapshots_per_entity_min`
  - `entity_table.snapshots_per_entity_max`
  - `entity_table.timestamp_prob`

Impact:

- Timestamp/entity behavior can now be tuned without code changes.
- Entity snapshot density becomes a first-class generation parameter.

## 4. Table-Type Semantics

Original main:

- Tables are only source/child by in-degree.
- No explicit "entity table" type.
- No root entity / child entity distinction.

Modified repo:

- A table with out-degree >= 1 is an entity table.
- Entity tables are split into:
  - root entity: no FK parents, has children
  - child entity: has FK parents and has children
- Non-entity leaf/activity tables have no children.

Impact:

- The generator can align synthetic tasks with RelBench-style entity prediction.
- Entity-specific columns and snapshot behavior can be applied consistently.

## 5. Row Count Semantics

Original main:

- Every table row count is sampled from the mapped/fluctuated DAG row count.
- One row corresponds to one table record.

Modified repo:

- Non-entity tables still use mapped/fluctuated row counts.
- Entity tables reinterpret the mapped row count as `num_entities`.
- Entity table row count becomes:

```text
num_rows = num_entities * snapshots_per_entity
```

Impact:

- Entity tables contain repeated entities over time.
- Same-entity row pairs exist within the generated table.
- This directly supports same-entity structural priors and temporal entity tasks.

## 6. Column Layout Differences

Original main:

```text
PK, FK..., optional timestamp, feature_0...
```

Modified repo:

```text
PK, FK..., optional timestamp, optional entity_id, feature_0...
```

Impact:

- `entity_id` is a generated metadata column, not a prediction target.
- Downstream task generation excludes `entity_id`.
- Entity ids are also used by downstream preprocessing/model code for same-entity and parent-entity matching signals.

## 7. Timestamp Assignment Differences

Original main:

- Only tables with exactly two parents can become timestamp tables.
- Such tables randomly choose whether to include a timestamp.
- Source tables and one-parent child tables generally have no timestamp.

Modified repo:

- Entity tables can receive timestamps with configurable probability, default `1.0`.
- Non-entity source tables never receive timestamps.
- Leaf/activity tables receive timestamps with configurable activity probability, default `1.0` in the current small config.

Impact:

- Timestamps are no longer restricted to two-parent tables.
- Entity snapshots usually carry time, matching real RelBench entity histories more closely.

## 8. Relationship Construction Differences

Original main:

- DAG edges become FK relationships.
- FK values are sampled later inside `MLPSCM`.

Modified repo:

- DAG edges still become FK relationships.
- FK values are sampled by `TableGenerator._compute_hsbm_fk_ids()` before child SCM forward.

Impact:

- Relationship schema is mostly unchanged.
- The semantics of how FK values are generated are completely different.

## 9. SCM Hyperparameter Differences

Original main:

- Samples only general MLP SCM hyperparameters from `DEFAULT_SAMPLED_HP`.
- Parent sampling parameters are `parent_sampling_dist` and `parent_sampling_alpha`.

Modified repo:

- Keeps general MLP SCM hyperparameters.
- Adds signal-group feature hyperparameters:
  - `use_signal_group_features`
  - `archetype_perturb_std`
  - `max_groups_per_feature`
  - `loading_sigma`
  - `loading_log_mean`
  - `basis_group_divisor`
  - `coupling_rank`
  - `coupling_lambda`
  - `residual_sigma`
  - `basis_perturb_eta`
  - `group_scale_time`
  - `group_scale_parent`
  - `group_scale_path`
  - `group_scale_intrinsic`
- Adds temporal hyperparameters:
  - `trend_active`
  - `seasonal_active`
  - `spike_active`
  - `gamma_tier`
  - `p_sort`
- Adds `DEFAULT_HSBM_HP`:
  - HSBM depth and cluster counts
  - propensity
  - matching latent
  - FK sparsity placeholders
- Adds `DEFAULT_HOMOPHILY_HP`.

Impact:

- The modified generator samples structural priors at the FK, feature, timestamp, and label levels.
- The original generator is mostly an MLP SCM with internal parent-row sampling.

## 10. FK Generation Location

Original main:

- FK sampling happens inside `MLPSCM.forward_with_input()`.
- The MLP receives sampled parent rows and then generates an `EDGE_PROB` output.
- `EDGE_PROB` filters the candidate rows down to the final `seq_len`.

Modified repo:

- FK sampling happens before `MLPSCM.forward_with_input()`.
- `MLPSCM.forward_with_input()` receives fixed `fk_ids`.
- No `EDGE_PROB` mask is used in the current HSBM path.

Impact:

- Original FK structure is learned/selected indirectly by MLP output filtering.
- Modified FK structure is explicitly controlled by HSBM and its reweighting mechanisms.

## 11. FK Sampling Distribution

Original main:

- For each parent independently:
  - uniform sampling, or
  - row-level Zipf sampling.
- Multi-parent FK tuples are just independent per-parent draws before the MLP/EDGE_PROB filter.

Modified repo:

- Single-parent:
  - HSBM block product probabilities
  - optional propensity reweighting
- Multi-parent:
  - shared child latent cluster path
  - parent-specific prefix alignment
  - optional matching-latent reweighting

Impact:

- Modified multi-parent FK tuples are correlated by construction.
- Optional propensity and matching-latent paths can bias FK choices by parent causal state.

## 12. HSBM Hierarchy Handling

Original main:

- No HSBM hierarchy.

Modified repo:

- Samples `hsbm_num_levels` and `hsbm_clusters_per_level`.
- Clips hierarchy depth/branching so leaf block count fits both parent and child row counts.
- Stores child-side and parent-side block paths.

Impact:

- FK topology creates reusable latent structural labels.
- The same block paths feed path signal and optional homophily labels.

## 13. Parent Feature Injection Differences

Original main:

- Parent causal outputs are concatenated to the child MLP input.
- Feature columns are masked from flattened MLP hidden states after the child MLP.

Modified repo:

- Parent causal outputs are still concatenated to child MLP input.
- Additionally, signal-group feature generation projects parent `CAUSAL_OUTPUT` through per-edge linear layers into a parent signal.
- The parent signal directly contributes to generated feature columns.

Impact:

- Parent influence is explicit in feature construction, not only implicit through the child MLP hidden state.

## 14. Path Signal Differences

Original main:

- No path signal and no HSBM block path.

Modified repo:

- HSBM block paths are encoded by per-edge `PathEncoder`s.
- Path embeddings are mean-pooled across edges.
- The path signal contributes directly to feature columns when enabled.
- CLI flag `--no_path_signal` can disable the path signal.

Impact:

- Feature values can reveal relational block structure directly.
- The draft's current generation description is incomplete without this signal group.

## 15. Intrinsic Signal Differences

Original main:

- Source features are raw masked MLP hidden states.

Modified repo:

- Source tables use the `intrinsic` signal group.
- Intrinsic signal is a learned projection from the table's own `CAUSAL_OUTPUT`.

Impact:

- Source feature columns share a controlled latent basis rather than independent hidden slices.

## 16. Signal-Group Feature Construction

Original main:

- No signal groups.
- Features are selected from MLP hidden states by masks.

Modified repo:

- Table archetype determines active groups:
  - source: intrinsic
  - timestamp child: time, parent, path
  - non-timestamp child: parent, path
- Per-feature groups are sampled by Dirichlet and top-k selection.
- Bases are subspace-split and assigned round-robin.
- Signs are shared per group/basis.
- Loadings are log-normal.
- Residual Gaussian noise is added.
- Cross-feature coupling mixes columns.

Impact:

- Modified synthetic tables have stronger, intentionally structured feature covariance and variance.
- This is one of the most important missing approach sections in `paper_draft.md`.

## 17. Timestamp Value Generation

Original main:

- Temporal intensity over an abstract range, default `(0,10)`.
- Components: trend, seasonality, spikes, noise.
- Enhanced temporal sampling uses temporal reinforcement for two-parent timestamp tables.

Modified repo:

- Day-index range is 1970-01-01 to 2020-12-31.
- Components:
  - trend
  - week/month/year Fourier seasonality
  - day-of-week factor from Dirichlet templates
  - shared event calendar
  - table-specific event sensitivity
  - noise
- Child timestamps can use parent timestamp lower bounds.
- `p_sort` can sort timestamps within FK groups.

Impact:

- Modified timestamps are calendar-like and relationally constrained.
- Original timestamps are synthetic intensities on a generic numeric time axis.

## 18. Enhanced Temporal Sampling Status

Original main:

- Timestamp child tables use `forward_with_enhanced_temporal_sampling()`.
- This path assumes exactly two parents.

Modified repo:

- That old method remains in the file as deprecated comparison code.
- Production path uses time-as-input with `time_dim > 0` and HSBM FK ids.

Impact:

- Timestamp generation is now integrated into normal MLP/HSBM flow and is not restricted to two-parent tables.

## 19. Materialization Differences

Original main:

- Materializes PK, FK, timestamp, and feature columns.
- FK values are parent row indices sampled by MLPSCM.

Modified repo:

- Materializes PK, FK, timestamp, feature columns, plus `entity_id`.
- FK values are parent row indices sampled by HSBM.
- Entity ids are deterministic per entity snapshot group.

Impact:

- Generated tables contain stable within-entity grouping keys.
- FK semantics are structurally generated rather than MLP-filtered.

## 20. HSBM Blocks For Labels

Original main:

- No HSBM blocks.
- No block metadata for labels.

Modified repo:

- Child `cluster_b` and parent `cluster_a` are retained.
- Optional homophily labels use block assignments with priority:
  1. target table child-side HSBM blocks
  2. target table parent-side HSBM blocks
  3. pseudo-blocks from feature clustering

Impact:

- Label generation can be tied to relational structure.
- Root entities can use parent-side blocks when they act as parents in child FK generation.

## 21. Task Focal Table Selection

Original main:

- Randomly chooses any eligible focal table.

Modified repo:

- Classifies tables into entity/non-entity.
- In `relbench_mode`, focal candidates are entity tables.
- Child entities with FK parents are preferred with probability `0.7`.
- Outside relbench mode, tables with FK parents are preferred to support FK-bias experiments.

Impact:

- Modified task distribution is intentionally shaped toward entity prediction and/or FK-signal coverage.
- Original task distribution is more generic and random.

## 22. Target Table Selection

Original main:

- Complex tasks choose a random leaf target table.

Modified repo:

- `SchemaGraph.generate_target_table_name(root_p=0.0)` preserves leaf-target behavior by default.
- In relbench mode, it is called with `root_p=1.0`, so the target is the focal/root table.

Impact:

- Modified relbench-mode tasks are `DIRECT_ATTRIBUTE_PREDICTION` on entity tables.
- Original complex tasks usually target leaf tables and often become aggregation tasks.

## 23. Schema Graph Size Requirement

Original main:

- Complex tasks require exactly three nodes.
- Any schema graph with node count not equal to three is skipped.

Modified repo:

- Allows two-node graphs for direct attribute prediction.
- Still requires enough graph structure for relational aggregation tasks.
- Uses biased first-hop child selection for entity focal tables.

Impact:

- Modified relbench-mode can generate more entity direct-attribute tasks.
- Original generator discards simpler but useful two-table structures.

## 24. Target Column Selection

Original main:

- Direct attribute prediction selects a categorical feature column.
- Relational aggregation selects a float feature column.

Modified repo:

- Same broad rule, but explicitly excludes `entity_id` from candidate target columns.
- Homophily replacement can overwrite the selected classification target.

Impact:

- Metadata columns used for structural bias are not leaked as labels.

## 25. Homophily Label Generation

Original main:

- No homophily-controlled labels.

Modified repo:

- Optional OPENRFM-style label generator.
- Per-RDB target homophily controls mixture between cluster-driven and feature-driven labels.
- Supports homophily and heterophily.

Impact:

- Modified synthetic tasks can vary how much labels depend on relational block structure.

## 26. Task Quality Filtering

Original main:

- No explicit quality gate in generation.

Modified repo:

- `task_quality.py` checks degenerate labels, class imbalance, sample size, constant features, structural independence, child/parent size ratios, lightweight model signal, and leakage.
- Generation retries and keeps the best passed tasks.

Impact:

- Modified pipeline may discard or retry low-quality generated tasks.
- Output task count and task composition can differ from requested count.

## 27. SNR And Feature Diagnostics

Original main:

- No signal-group diagnostics.

Modified repo:

- Dumps `_feature_diagnostics.json`.
- Computes SNR proxy from group scales and residual noise.
- Can discard an RDB below `--snr-threshold`.

Impact:

- Modified generation has a post-hoc audit trail for feature construction.
- The generator can filter low-signal RDBs before downstream preprocessing.

## 28. Saved Generation Schemas

Original main:

- Saves table/task generation schemas.
- Table schema includes SCM parameters and masks.

Modified repo:

- Saves table/task schemas plus RDB-level event calendar.
- Table schema includes HSBM parameters and signal-group-related SCM params.

Impact:

- Modified outputs are more reproducible/auditable for structural generation choices.

## 29. RelBench Conversion

Original main:

- No dedicated `convert_dag_rdb.py` in the repo snapshot.

Modified repo:

- Adds conversion from generated DBB/4DBInfer-style directories to RelBench `Database` and task directory layout.

Impact:

- Synthetic generation is integrated with RelBench-style preprocessing and evaluation workflows.

## 30. Parent Entity IDs

Downstream/model-support side:

- Separate preprocessing/model code may compute or load `parent_entity_ids` for FK attention matching. This maps parent row indices back to parent entity ids when parent tables have temporal snapshots.

Original main:

- No entity ids and therefore no parent-entity-level FK semantics.

Impact:

- The modified pipeline can represent parent-entity equality separately from raw parent-row equality for model-side FK attention.

## 31. Row GNN

Original main:

- Optional row-level GNN refinement exists.

Modified repo:

- Row GNN remains optional and is still run before materialization.

Impact:

- This is not a major difference in the current generation design, though generated row embeddings now include signal-group/HSBM-conditioned outputs before optional refinement.

## 32. What Did Not Fundamentally Change

The following remain broadly shared:

- DAG nodes still become tables.
- DAG edges still become FK relationships.
- Tables still save PK/FK/feature/timestamp columns in 4DBInfer-style metadata.
- Each table still has an MLP-based SCM.
- Parent `CAUSAL_OUTPUT` still conditions child generation.
- Optional row GNN refinement remains available.
- Simple and complex task abstractions still exist.

## 33. Paper-Relevant Missing Approach Content

The current `paper_draft.md` already discusses EntityAttentionBias, `parent_entity_ids`, and `relbench_mode`. It should additionally cover these generation-side changes:

1. Entity-table temporal snapshots and `entity_id` column generation.
2. HSBM FK generation replacing internal MLPSCM parent sampling.
3. Multi-parent shared latent HSBM path.
4. FK propensity and matching-latent reweighting.
5. Signal-group feature generation:
   - table archetype alpha
   - time/parent/path/intrinsic groups
   - PathEncoder over HSBM block paths
   - parent projectors
   - intrinsic projector
   - loadings, residuals, coupling
6. Calendar-aligned timestamp generation and parent-aware timestamp lower bounds.
7. Optional homophily label generation.
8. Task quality gate and SNR filtering, if the reported dataset used them.

Items 1-6 are central method changes. Items 7-8 should be described as optional or experimental unless the exact reported run used them.
