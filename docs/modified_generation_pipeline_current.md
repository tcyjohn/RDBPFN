# Modified RDBPFN Generation Pipeline

Scope: current working tree on `feature/hsbm-temporal-refactor`. This document covers the generation-side pipeline only: DAG input to generated RDB directories with tasks and generation schemas. Training/model changes are intentionally out of scope.

Primary code paths:
- `data_generation/RDB/dag_to_rdb_generator.py`
- `data_generation/RDB/src/table_def/table_generation.py`
- `data_generation/RDB/src/prior/mlp_scm.py`
- `data_generation/RDB/src/prior/hsbm.py`
- `data_generation/RDB/src/prior/temporal_vocab.py`
- `data_generation/RDB/src/table_def/task_generation.py`
- `data_generation/RDB/src/table_def/task_quality.py`

## Pipeline Overview

```text
DAG structures (`rdb_v1.pth`)
  -> parse DAG nodes/edges and table dimensions
  -> create table configs:
       PK/FK layout, optional timestamp, optional entity_id,
       entity temporal snapshots
  -> create `RDB`, `Table`, `Relationship`, and `TableGenerator` objects
  -> initialize one `MLPSCM` per table in topological order:
       SCM HPs + HSBM HPs + temporal HPs + signal-group HPs
  -> generate rows in topological order:
       source tables: MLP SCM + intrinsic signal-group features
       child tables: HSBM FK ids -> parent-conditioned MLP SCM
                    -> time/parent/path/intrinsic signal-group features
  -> optional row-level GNN refinement
  -> materialize tensors into typed table columns
  -> generate tasks:
       simple tasks or complex RelBench-style tasks
       optional homophily labels and quality gate
  -> save 4DBInfer-style dataset:
       parquet tables, task splits, `metadata.yaml`, `generation_schemas.yaml`,
       optional `_feature_diagnostics.json`
```

## 1. Inputs And Entry Points

The raw structural prior is a PyTorch dictionary loaded from `data_generation/RDB/datasets/rdb_v1.pth` with:

- `src_list`: source node for each DAG edge.
- `dst_list`: destination node for each DAG edge.
- `x_n_list`: per-node raw `[num_rows, num_cols]`.
- `y_list`: labels retained from the source DAG corpus.

The main entry point is `DAGToRDBGenerator` in `dag_to_rdb_generator.py`. The current CLI adds generation controls beyond the original repo:

- `--relbench_mode`
- `--use_homophily_labels`
- `--no_path_signal`
- `--snapshots_per_entity_min`
- `--snapshots_per_entity_max`
- `--entity_timestamp_prob`
- `--no-quality-filter`
- `--quality-max-retries`
- `--snr-threshold`

`scripts/run_pipeline.sh` wraps generation, preprocessing, H5 merge, and optional training. For this document, only its generation step is in scope. `scripts/generateRDB_hsbm_v2.sh` is a narrower raw-generation plus RelBench-conversion helper.

## 2. DAG To Table Schema

`parse_dag_structure()` maps each DAG node to a table and each edge `src -> dst` to a FK relationship from child `dst` to parent `src`.

Row and feature counts are derived from `dimension_config`:

- Raw row counts are mapped into roughly `[1000, 5000]` with piecewise linear rules and fluctuation.
- Raw column counts are mapped into feature counts, usually `8-12`.
- Table columns are then assembled as:

```text
column 0: primary key
columns 1..num_parents: foreign keys
optional: timestamp
optional: entity_id
remaining: feature_0, feature_1, ...
```

## 3. Entity Tables And Temporal Snapshots

The modified generator introduces an explicit entity-table concept:

- A table is an entity table when it has at least one outgoing FK edge, i.e. some child table references it.
- Entity rows are expanded as:

```text
num_rows = num_entities * snapshots_per_entity
snapshots_per_entity ~ randint(min, max), default [5, 20]
```

For entity tables with more than one snapshot, an `entity_id` column is inserted after FK/timestamp columns and before feature columns. During materialization, `RDB._materialize_tables_from_pending()` fills it as:

```text
0, 0, ..., 0, 1, 1, ..., 1, ...
```

where each entity id repeats `snapshots_per_entity` times. This column is generation metadata for same-entity structure and is excluded from task target candidates.

Timestamp assignment is also changed:

- Entity tables use `entity_table.timestamp_prob`, default `1.0` unless overridden.
- Non-entity source tables never get timestamps.
- Activity/leaf tables use `timestamp.activity_prob`, default `1.0` in the small config.

## 4. RDB Object Construction

`create_rdb_from_config()` creates:

- `Table` objects with typed `DataTypeConfig`s.
- `Relationship` objects for every DAG edge.
- `TableGenerator` objects attached to each table.

The `Table` stores `time_column` and optionally `entity_column`. The matching `TableGenerator` stores:

- `is_entity_table`
- `num_entities`
- `snapshots_per_entity`

These fields drive entity id materialization and parent-entity-aware FK sampling.

## 5. SCM And HSBM Initialization

`RDB.init_table_SCMs()` runs in DAG topological order. For each table it samples and wires:

- General SCM hyperparameters from `DEFAULT_SAMPLED_HP`.
- Per-parent HSBM depth and cluster count from `DEFAULT_HSBM_HP`.
- FK propensity parameters for single-parent child tables.
- Matching-latent parameters for multi-parent child tables.
- Zipf concentration exponent `hsbm_zipf_alpha` for child tables.
- Temporal lifecycle parameters `gamma_tier` and intra-FK timestamp sort probability `p_sort`.
- Signal-group feature generation hyperparameters.
- Per-edge `RelationKey`s, parent causal dimensions, and HSBM levels for parent/path signal modules.

An RDB-level `EventCalendar` is sampled once and attached to timestamp-table `TemporalVocab`s with table-specific sensitivity.

## 6. HSBM FK Generation

FK ids are now generated outside the MLP in `TableGenerator._compute_hsbm_fk_ids()`, before child rows are generated. The implementation lives in `src/prior/hsbm.py`.

### Single-Parent FK Sampling

For one parent, HSBM assigns parent rows (`cluster_a`) and child rows (`cluster_b`) to hierarchical block paths. Per-level block probability matrices use:

- within-block diagonal probability `0.9`
- cross-block probability sampled from `Uniform(0.001, 0.002)`

The connection probability is the product across hierarchy levels. Each child row samples one parent row.

If `propensity_rho > 0`, the single-parent path uses `compute_hsbm_fk_ids_with_propensity()`. It ranks parent rows from parent `CAUSAL_OUTPUT`, samples child-specific targets, and reweights HSBM probabilities by:

```text
exp(-propensity_beta * |rank_score(parent) - target(child)|)
```

### Multi-Parent FK Sampling

For multiple parents, child rows share one latent child cluster path. Each parent reads the prefix matching its own HSBM depth. This induces tuple-level FK correlation because all parent choices for a child are aligned to the same latent path.

If `matching_latent_dim > 0`, `compute_hsbm_fk_ids_multi_with_matching()` projects parent `CAUSAL_OUTPUT`s into a shared latent space and samples parent rows by proximity to a child-specific target latent.

### Entity-Level Zipf Concentration

When `hsbm_zipf_alpha` is set, HSBM probabilities are multiplied by Zipf weights:

```text
weight = 1 / rank^alpha, alpha in [1.1, 2.0]
```

If the parent table has `entity_id`, the Zipf distribution is applied at entity level rather than row level. All snapshots of the same parent entity share one popularity weight, divided across that entity's rows.

This concentration path is wired through:

- plain single-parent HSBM
- single-parent propensity HSBM
- plain multi-parent HSBM
- multi-parent matching-latent HSBM

### HSBM Block Outputs

The child-side block paths are stored on the child `TableGenerator` as `hsbm_block_paths`. Parent-side blocks (`cluster_a`) are stored back on parent generators as `hsbm_parent_blocks`. These are used by optional homophily label generation.

## 7. MLPSCM Row And Feature Generation

Each table has an `MLPSCM`.

For source tables:

```text
causes_raw -> MLP layers -> CAUSAL_OUTPUT + raw X
```

For child tables:

```text
HSBM fk_ids are already fixed
causes_raw + optional time features + parent CAUSAL_OUTPUT[fk_ids]
  -> MLP layers -> CAUSAL_OUTPUT + raw X
```

The child MLP no longer samples parent rows internally. It receives FK ids and conditions on the selected parent causal outputs.

The MLP output still provides:

- `MASK_TYPE.CAUSAL_OUTPUT`: propagated to downstream children.
- `MASK_TYPE.FULL`: full hidden representation.
- `MASK_TYPE.X`: initial raw feature slice, then usually overwritten by signal-group features.
- `MASK_TYPE.TIMESTAMP`: timestamp values for timestamp tables.

## 8. Signal-Group Feature Generation

The current default sets `use_signal_group_features=True`. This replaces the original "sample feature columns from flattened MLP hidden states" path for `MASK_TYPE.X`.

The feature generator constructs each feature column as a mixture of latent signal groups:

- `time`: calendar basis and gate vector from timestamp generation.
- `parent`: per-edge linear projection of parent `CAUSAL_OUTPUT`, mean-pooled across parents.
- `path`: per-edge `PathEncoder` embedding of HSBM block paths, mean-pooled across edges.
- `intrinsic`: projection of the table's own `CAUSAL_OUTPUT`.

The group mixture is table-archetype dependent:

- Source tables: `intrinsic` only.
- Timestamp child tables: `time + parent + path`.
- Non-timestamp child tables: `parent + path`.
- Multi-parent tables raise the parent share up to a cap.

For each table:

1. A base 4-vector alpha over `[time, parent, path, intrinsic]` is derived from table metadata.
2. Active groups receive log-normal perturbation and are renormalized.
3. Each feature samples group weights from a Dirichlet distribution and keeps the top groups.
4. Each selected group uses round-robin basis assignment, shared signs per basis, log-normal loadings, and small basis perturbations.
5. Signals are RMS-normalized and scaled by group-specific `group_scale_*`.
6. Feature values are summed from projected signals plus residual Gaussian noise.
7. Uniform off-diagonal cross-feature coupling is applied:

```text
X_out = X_struct + coupling_lambda * (X_struct @ W)
```

This is the main missing approach in the previous draft: feature columns are not merely raw MLP hidden slices. They are explicitly structured by temporal, relational-parent, HSBM-path, and intrinsic latent groups.

## 9. Timestamp Generation

The temporal generator is calendar-aligned.

`TemporalVocab` samples timestamps in day-index space over 1970-01-01 to 2020-12-31. The intensity combines:

- linear trend
- week/month/year Fourier seasonality
- day-of-week multiplicative factors from a Dirichlet-template mixture
- RDB-level shared event calendar with per-table sensitivity
- noise

For child timestamp tables, `TableGenerator._compute_t_min_for_child()` computes a per-row lower bound from parent timestamps when available. If parent timestamps are unavailable, it falls back to a DAG-position-dependent time range.

`p_sort` optionally sorts timestamps inside FK groups, making child activity times more coherent within parent-linked groups.

## 10. Materialization

After all tables are generated, optional row-level GNN refinement may update row embeddings. Then `RDB._materialize_tables_from_pending()` converts pending SCM outputs into table tensors:

- PK is `0..num_rows-1`.
- FK columns are HSBM parent row indices.
- Timestamp columns are integer days since epoch and saved as datetime parquet columns.
- `entity_id` is deterministic per entity snapshot group.
- Feature columns are processed through `ColumnDataProcessor` for float/categorical conversion and outlier handling.

Each table can then be saved both as CSV and as parquet.

## 11. Task Generation

The generator supports simple single-table tasks and complex relational tasks. The modified path mainly uses complex tasks.

### relbench_mode

`relbench_mode=True` changes focal/target selection:

- `entity_task_ratio` is forced to `1.0`.
- Focal tables are entity tables.
- Among entity tables, child entities with FK parents are preferred with probability `0.7` when available.
- `SchemaGraph.generate_target_table_name(root_p=1.0)` makes the target the focal/root table.
- The task becomes `DIRECT_ATTRIBUTE_PREDICTION`.
- `entity_id` is excluded from target candidates.

The generated task still includes neighbor tables through schema graph construction, so child features can enter through downstream DFS preprocessing even when the target is the focal entity table.

### Optional Homophily Labels

When `use_homophily_labels=True`, classification targets can be replaced with OPENRFM-style homophily-controlled labels:

1. Prefer child-side HSBM blocks for the target table.
2. Fall back to parent-side HSBM blocks.
3. Fall back to feature-derived pseudo-blocks.

For entity tables with snapshots, block assignments are converted to entity-level blocks and broadcast back to snapshots.

### Quality Gate

When quality filtering is enabled, task generation is retried. `task_quality.py` checks rule-based failures and model-based quality diagnostics. The best attempt is kept and failing task schemas are pruned before saving.

Separately, `_feature_diagnostics.json` records signal-group assignments and scales. An SNR proxy,

```text
mean(active group scales) / mean(residual sigmas)
```

can discard low-signal RDBs.

## 12. Outputs

Each generated RDB directory contains:

- table parquet files
- `metadata.yaml`
- task split data
- `generation_schemas.yaml`
- optional `_feature_diagnostics.json`
- optional CSV exports under `csv_data/`

The directory is a 4DBInfer-style synthetic relational dataset and can be converted to RelBench layout by `scripts/convert_dag_rdb.py` or fed into downstream preprocessing/H5 merge scripts.
