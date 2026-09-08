# Data Generation

This directory contains the data generation stage of RDB_PFN. It has two subprojects:

1. `single_table/` for synthetic single-table priors.
2. `RDB/` for synthetic relational database generation.

The outputs of both subprojects feed into the preprocessing stage documented in [../data_preprocessing/README.md](../data_preprocessing/README.md).

## Directory Layout

`single_table/`
Python package for generating synthetic single-table tasks.

`RDB/`
Python package for generating synthetic RDBs.

`single_table_datasets/`
Default output location for generated single-table batches.

`RDB_datasets/`
Default output location for generated synthetic relational databases.

## Subproject 1: Single-Table Generation

### Purpose

This subproject generates large synthetic single-table batches that are later merged into `.h5` priors for model pretraining.

### Installation

From the repository root:

```bash
cd data_generation/single_table
pip install -e .
```

The single-table generation code is adapted from the [tabicl](https://github.com/tabicl/tabicl) project with slight modifications to the prior. You can also refer to that repository for additional usage details. We gratefully acknowledge their work.


### Main Entry Points

- [single_table/single_table_generate.sh](single_table/single_table_generate.sh): launches the default generation runs.
- `src/tabicl/prior/genload.py`: lower-level generator invoked by the shell script.

### Default Usage

Run the provided generation script:

```bash
cd data_generation/single_table
bash single_table_generate.sh
```

The current script generates:

- `single_table_datasets/single_table_stage1`
- `single_table_datasets/single_table_stage2`

These directories are consumed later by [../data_preprocessing/single_table_processing.sh](../data_preprocessing/single_table_processing.sh).

### Notes

- The provided script is configured for large-scale generation, which may take tens of hours to complete.
- Generation parameters such as `--num_batches`, feature count, class count, and sequence length are currently hard-coded in the shell script. You can modify them to generate smaller datasets for testing.

## Subproject 2: RDB Generation

### Current Generator

The main entry point is [RDB/dag_to_rdb_generator.py](RDB/dag_to_rdb_generator.py). Install the shared environment with `pixi install` from the repository root. [RDB/RDB_generate.sh](RDB/RDB_generate.sh) retains the original large-scale schedule; the current combined pipeline lives in [../scripts/run_pipeline.sh](../scripts/run_pipeline.sh).

Generate a small raw corpus from the repository root:

```bash
pixi run python data_generation/RDB/dag_to_rdb_generator.py \
  --dag_data_path data_generation/RDB/datasets/rdb_v1.pth \
  --config_file data_generation/RDB/dag_to_rdb_config_small.yaml \
  --num_rdbs 4 --start_index 0 --num_processes 1 \
  --output_base_dir data_generation/RDB_datasets/my_run \
  --use_complex_tasks true
```

Each accepted database is saved in 4DBInfer format with parquet tables, task splits, and `metadata.yaml`. Quality filtering can reject databases, so the requested index range does not guarantee that every output exists.

### Changes in This Fork

- [HSBM](RDB/src/prior/hsbm.py) samples FK connectivity, including correlated multi-parent assignments.
- [MLP SCM](RDB/src/prior/mlp_scm.py) and [table generation](RDB/src/table_def/table_generation.py) implement signal-group features and entity temporal snapshots.
- [Task generation](RDB/src/table_def/task_generation.py) supports complex tasks and RelBench-style entity prediction.
- [Task quality](RDB/src/table_def/task_quality.py) filters tasks; [homophily](RDB/src/prior/homophily.py) provides optional label generation.

| Option | Purpose |
| --- | --- |
| `--relbench_mode` | Select entity-focused RelBench-style tasks. |
| `--snapshots_per_entity_min`, `--snapshots_per_entity_max` | Override the entity snapshot-count range. |
| `--entity_timestamp_prob` | Override the probability of timestamps on entity tables. |
| `--no-quality-filter`, `--quality-max-retries` | Disable the quality gate or set its retry budget. |
| `--snr-threshold` | Set the SNR acceptance threshold; `0` disables it. |
| `--use_homophily_labels` | Enable optional homophily-controlled labels. |
| `--no_path_signal` | Disable the path-signal component. |
| `--use_row_gnn`, `--gnn_device` | Enable optional row GNN refinement and select its device. |

Sizing is configured in [RDB/dag_to_rdb_config_small.yaml](RDB/dag_to_rdb_config_small.yaml); prior hyperparameters are in [RDB/src/prior/prior_config.py](RDB/src/prior/prior_config.py). See [../docs/data_generation_pipeline.md](../docs/data_generation_pipeline.md) for implementation details.

## Handoff to Preprocessing

After generation, use [../data_preprocessing/README.md](../data_preprocessing/README.md) for converting single-table batches into `.h5` priors and raw RDBs into processed task datasets and pretraining files.
