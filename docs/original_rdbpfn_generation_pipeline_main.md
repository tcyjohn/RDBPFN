# Original RDBPFN Main Generation Pipeline

Scope: original upstream code from `origin/main` after `git fetch origin main`, commit `0720d1d167256d9219cd993ac7b2c6c07f3b639d`. This document covers the generation-side pipeline only.

Primary code paths in the original main branch:

- `/data/caijunyu/RDBPFN-main/data_generation/RDB/dag_to_rdb_generator.py`
- `/data/caijunyu/RDBPFN-main/data_generation/RDB/src/table_def/table_generation.py`
- `/data/caijunyu/RDBPFN-main/data_generation/RDB/src/prior/mlp_scm.py`
- `/data/caijunyu/RDBPFN-main/data_generation/RDB/src/prior/temporal_vocab.py`
- `/data/caijunyu/RDBPFN-main/data_generation/RDB/src/table_def/task_generation.py`

## Pipeline Overview

```text
DAG structures (`rdb_v1.pth`)
  -> parse DAG nodes/edges and table dimensions
  -> create table configs:
       PK/FK layout, optional timestamp only for 2-parent children
  -> create `RDB`, `Table`, `Relationship`, and `TableGenerator` objects
  -> initialize one `MLPSCM` per table in topological order
  -> generate rows in topological order:
       source tables: MLP SCM without parents
       child tables: MLPSCM samples parent rows internally,
                     generates EDGE_PROB,
                     samples final rows by EDGE_PROB
       timestamp child tables: enhanced temporal sampling path
  -> optional row-level GNN refinement
  -> materialize tensors into typed table columns
  -> generate simple or complex tasks
  -> save 4DBInfer-style dataset:
       parquet tables, task splits, `metadata.yaml`, `generation_schemas.yaml`
```

## 1. Inputs And Entry Points

The original generator reads the same DAG data format:

- `src_list`
- `dst_list`
- `x_n_list`
- `y_list`

The main CLI is `dag_to_rdb_generator.py`. The original script supports:

- number of RDBs
- start index
- output directory
- config file
- number of processes
- `--use_complex_tasks`
- optional `--use_row_gnn`

The original `RDB_generate.sh` launches large generation batches directly, producing small/large datasets with and without row GNN.

## 2. DAG To Table Schema

Each DAG node becomes one table. Each DAG edge `src -> dst` becomes one FK from child table `dst` to parent table `src`.

Rows and feature columns are derived from the dimension config:

- row count: piecewise mapped and fluctuated, usually within `[1000, 5000]`
- feature count: mapped from DAG column count

Original column layout:

```text
column 0: primary key
columns 1..num_parents: foreign keys
optional timestamp
feature_0, feature_1, ...
```

The original code has no entity-table classification, no temporal snapshots per entity, and no `entity_id` column.

## 3. Timestamp Table Selection

The original timestamp logic is narrow:

- A table can become a timestamp table only if it has exactly two parents.
- For such 2-parent child tables, timestamp is chosen randomly.
- Other tables do not receive timestamps.

If timestamp is present, the timestamp column is inserted after PK/FK columns and before features.

## 4. RDB Object Construction

`create_rdb_from_config()` creates:

- `Table` objects with `DataTypeConfig`s.
- `Relationship` objects from child FK to parent PK.
- `TableGenerator` objects for all tables.

Table metadata includes `time_column` for timestamp tables. There is no entity metadata on either `Table` or `TableGenerator`.

## 5. SCM Initialization

`RDB.init_table_SCMs()` runs in topological order.

For each table:

1. Parent tables are identified by FK relationships.
2. `MASK_TYPE.X` is assigned to the table's feature count.
3. For non-timestamp child tables, `MASK_TYPE.EDGE_PROB` is added.
4. For timestamp child tables, `MASK_TYPE.TIMESTAMP` is added.
5. General SCM hyperparameters are sampled from `DEFAULT_SAMPLED_HP`.
6. A `TableGenerationSchema` is recorded.
7. `TableGenerator.init_table_SCM()` creates an `MLPSCM`.

The original code does not sample HSBM hyperparameters and does not create per-edge path encoders, parent projectors, or signal-group feature assignments.

## 6. Source Table Generation

For source tables, `MLPSCM.forward_without_input()` samples root causes from `XSampler`, forwards them through MLP layers, flattens hidden states, and extracts feature columns through masks.

The source table output contains:

- `MASK_TYPE.X`
- `MASK_TYPE.CAUSAL_OUTPUT`
- `MASK_TYPE.FULL`

These outputs are cached for downstream child tables.

## 7. Child Table Generation

For child tables without timestamp-based enhanced sampling, `MLPSCM.forward_with_input(parent_data_list)` does all FK sampling internally:

1. Sample `seq_len * sampling_ratio` root causes.
2. For each parent table, sample parent row indices independently:
   - `uniform` via `random.choices`, or
   - row-level `zipf` via `sample_zipf_indices()`.
3. Gather parent `CAUSAL_OUTPUT` rows at those sampled indices.
4. Concatenate parent outputs to root causes.
5. Forward through the MLP.
6. Extract masked outputs, including `EDGE_PROB`.
7. Normalize `EDGE_PROB` to `[0,1]`.
8. Sample exactly `seq_len` final rows without replacement according to `EDGE_PROB`.
9. Apply the same final-row selection to `X` and `parent_idxes`.

The resulting `parent_idxes` become FK values during table materialization.

Key implication: FK selection is not a structural graph prior. It is a per-parent independent sampling path followed by a learned edge-probability filter from the child MLP output.

## 8. Enhanced Temporal Sampling

For timestamp child tables, original `TableGenerator.generate_data()` calls `MLPSCM.forward_with_enhanced_temporal_sampling()`.

This path is designed for exactly two parent tables:

1. Sample timestamps from `TemporalVocab`.
2. Generate parent augmentation embeddings.
3. Initialize parent mass vectors and an edge kernel.
4. In batches, sample a row from parent P.
5. Sample a row from parent Q conditioned on parent P and the edge kernel.
6. Generate child features from root causes and the selected parent outputs.
7. Update parent mass vectors with temporal reinforcement.
8. Return child features, two parent FK indices, and normalized timestamps.

The original `TemporalVocab` generates an abstract intensity over a small numeric time range. It combines trend, seasonality, spikes, and noise components from pre-defined configuration families.

## 9. Feature Generation

Original feature columns come from flattened MLP hidden states selected by masks:

```text
outputs_flat = concat(MLP hidden layers)
X[mask] = outputs_flat[:, selected_indices]
```

For child tables, parent information affects features because sampled parent `CAUSAL_OUTPUT` rows are concatenated into the MLP input. There is no explicit signal-group feature construction, no HSBM path signal, no intrinsic signal projection, and no cross-feature coupling layer.

## 10. Materialization

After every table is generated, optional row-level GNN refinement may update row embeddings. Then `_materialize_tables_from_pending()` converts generated tensors into typed table data:

- PK is `0..num_rows-1`.
- FK columns use the sampled parent row indices.
- Timestamp columns are processed into days-since-epoch and saved as datetime columns.
- Feature columns are processed by `ColumnDataProcessor` into float or categorical values.

## 11. Task Generation

The original generator supports simple tasks and complex tasks.

### Simple Tasks

`generate_tasks_for_rdb_per_table()` creates one single-table prediction task per eligible table by selecting non-PK/FK/timestamp feature columns.

### Complex Tasks

Original `generate_tasks_for_rdb_with_complex_tasks()`:

1. Randomly selects any eligible focal table.
2. Builds a two-hop, one-neighbor-per-hop schema graph.
3. Requires the schema graph to contain exactly three nodes.
4. Selects a target table by choosing a random leaf node.
5. Determines whether the task is direct attribute prediction or relational aggregation.
6. Chooses a categorical target for direct attribute prediction or a float target for relational aggregation.
7. Generates task data and writes metadata.

There is no RelBench-style entity-focal mode, no root-target override, no child-entity preference, no homophily label replacement, and no task quality gate.

## 12. Outputs

Each generated RDB directory contains:

- parquet table files
- task split files
- `metadata.yaml`
- `generation_schemas.yaml`
- optional CSV files under `csv_data/`

The original pipeline saves raw synthetic RDBs directly in 4DBInfer-style format. Large original training runs are orchestrated by `RDB_generate.sh` with multiple 20k-80k dataset batches.
