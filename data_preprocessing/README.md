# Data Preprocessing

Convert generated relational databases into DFS task tables and HDF5 priors, or merge single-table prior batches into HDF5. Run `pixi install` at the repository root for the shared environment.

## Relational Workflow

[run_preprocess.py](run_preprocess.py) runs one preprocessing stage at a time. The current pipeline uses a pre-DFS transform, two-hop DFS, and a post-DFS transform. The following commands start from the repository root and process `dag_rdb_0` from the generation example:

```bash
pixi run python data_preprocessing/run_preprocess.py \
  data_generation/RDB_datasets/my_run/dag_rdb_0 transform \
  data_generation/RDB_datasets/my_run-tmp/dag_rdb_0-pre \
  data_preprocessing/configs/transform/pre-dfs.yaml 2

pixi run python data_preprocessing/run_preprocess.py \
  data_generation/RDB_datasets/my_run-tmp/dag_rdb_0-pre dfs \
  data_generation/RDB_datasets/my_run-tmp/dag_rdb_0-dfs \
  data_preprocessing/configs/dfs/dfs-2-ft.yaml 2

pixi run python data_preprocessing/run_preprocess.py \
  data_generation/RDB_datasets/my_run-tmp/dag_rdb_0-dfs transform \
  data_generation/RDB_datasets/my_run-processed/dag_rdb_0-dfs-2 \
  data_preprocessing/configs/transform/post-dfs.yaml 2

pixi run python data_preprocessing/merge_dbinfer_to_h5.py \
  --dataset-root data_generation/RDB_datasets/my_run-processed \
  --output model_pretrain/pretrain_datasets/my_run.h5 \
  --total-rows 600 --max-columns 90
```

Repeat the three preprocessing stages for other generated databases before merging. DFS depth is configured by the YAML's `max_depth`; the final positional argument is accepted by the wrapper but does not override that setting. [scripts/run_pipeline.sh](../scripts/run_pipeline.sh) automates the multi-database workflow and also launches training.

The merger samples task rows and columns. Its optional structural arrays include `fk_values`, `entity_ids`, and `parent_entity_ids`. Parent entity IDs map FK targets to entities across temporal snapshots. Missing identity values use `-1`; features, targets, and structural arrays must keep the same row order. Older corpora may lack this metadata.

Use repeated `--dataset-root` arguments to merge multiple processed corpora. Run the merger with `--help` for row counts, train-split ratios, column limits, and feature-importance options.

## Single-Table Workflow

After generating `single_table_stage1/` and `single_table_stage2/`, run from the repository root:

```bash
cd data_preprocessing
pixi run bash single_table_processing.sh
```

[single_table_processing.sh](single_table_processing.sh) calls [merge_icl_batches_to_h5.py](merge_icl_batches_to_h5.py) and writes:

- `model_pretrain/pretrain_datasets/single_table_stage1.h5`: 600 rows, 18 features.
- `model_pretrain/pretrain_datasets/single_table_stage2.h5`: 600 rows, 30 features.

These paths are relative to the repository root; the script itself runs from `data_preprocessing/`.

## Other Entry Points

- [RDB_processing.sh](RDB_processing.sh) retains the original fixed corpus schedule; its dataset names differ from the `my_run` example.
- [filter_h5_sampling_columns.py](filter_h5_sampling_columns.py) downsamples columns in an existing HDF5 corpus.
- DFS preprocessing uses code adapted from DBInfer.

Continue with [model_pretrain/README.md](../model_pretrain/README.md). Training corpora and evaluation datasets are separate inputs; creating a training HDF5 file does not populate the benchmark directories.
