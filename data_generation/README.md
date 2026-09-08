# Data Generation

This stage generates relational databases in 4DBInfer format and single-table prior batches for [preprocessing](../data_preprocessing/README.md).

## Relational Databases

Run `pixi install` at the repository root, then generate a small corpus:

```bash
pixi run python data_generation/RDB/dag_to_rdb_generator.py \
  --dag_data_path data_generation/RDB/datasets/rdb_v1.pth \
  --config_file data_generation/RDB/dag_to_rdb_config_small.yaml \
  --num_rdbs 4 --start_index 0 --num_processes 1 \
  --output_base_dir data_generation/RDB_datasets/my_run \
  --use_complex_tasks true
```

The DAG input must exist before generation. Accepted databases are saved as `dag_rdb_<index>/` directories containing parquet tables, task splits, and `metadata.yaml`. Quality filtering can reject candidates, so not every requested index necessarily produces a database.

The generator builds table schemas from DAGs, samples FK connections, generates column values and temporal data, and creates prediction tasks. Sizing rules are in [dag_to_rdb_config_small.yaml](RDB/dag_to_rdb_config_small.yaml); sampled prior settings are in [prior_config.py](RDB/src/prior/prior_config.py).

| Option | Purpose |
| --- | --- |
| `--num_processes` | Number of generation workers. |
| `--use_complex_tasks true` | Generate complex tasks; the CLI default is false. |
| `--relbench_mode` | Select RelBench-style entity prediction tasks. |
| `--snapshots_per_entity_min`, `--snapshots_per_entity_max` | Override the entity snapshot-count range. |
| `--entity_timestamp_prob` | Override the probability of timestamps on entity tables. |
| `--no-quality-filter`, `--quality-max-retries` | Disable the task quality gate or set its retry budget. |
| `--snr-threshold` | Set the database SNR acceptance threshold; `0` disables this threshold. |
| `--use_homophily_labels` | Enable optional homophily-controlled labels. |
| `--no_path_signal` | Disable the path-signal component. |
| `--use_row_gnn`, `--gnn_device` | Enable optional row GNN refinement and select its device. |

Use `--help` for the full argument list. The main implementations are [HSBM sampling](RDB/src/prior/hsbm.py), [MLP SCM](RDB/src/prior/mlp_scm.py), [table generation](RDB/src/table_def/table_generation.py), and [task generation](RDB/src/table_def/task_generation.py).

## Single-Table Priors

See [single_table/README.md](single_table/README.md) for installation and a small generation example. [single_table_generate.sh](single_table/single_table_generate.sh) provides the large-scale schedule: 600,000 batches each for stage 1 (18 features) and stage 2 (30 features), with 600 rows per task.

## Continue to Preprocessing

Follow [data_preprocessing/README.md](../data_preprocessing/README.md) to convert raw databases and prior batches into HDF5 files. [scripts/run_pipeline.sh](../scripts/run_pipeline.sh) combines relational generation, preprocessing, and training. [RDB_generate.sh](RDB/RDB_generate.sh) retains the original large-scale generation schedule with fixed output names.
