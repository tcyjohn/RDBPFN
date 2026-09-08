# Single-Table Prior Generation

This subproject adapts TabICL's synthetic prior generator to produce the single-table initialization corpora used by RDB-PFN.

## Install

The local package has additional dependencies beyond the root Pixi environment. Install it in a separate environment using Python 3.10, or another Python version supported by [pyproject.toml](pyproject.toml). From the repository root:

```bash
python -m pip install -e data_generation/single_table
```

## Generate a Small Batch

From the repository root, using the environment where the local package is installed:

```bash
cd data_generation/single_table
python src/tabicl/prior/genload.py \
  --save_dir ../single_table_datasets/demo \
  --np_seed 42 --torch_seed 42 \
  --num_batches 4 --batch_size 1 --batch_size_per_gp 1 \
  --prior_type mix_scm --min_features 18 --max_features 18 \
  --max_classes 2 --max_seq_len 600 \
  --min_train_size 0.5 --max_train_size 0.9 \
  --n_jobs 1 --num_threads_per_generate 1 --device cpu
```

[genload.py](src/tabicl/prior/genload.py) saves batches containing features, labels, feature counts, sequence lengths, and training split sizes.

## Full Schedule

From `data_generation/single_table/`:

```bash
bash single_table_generate.sh
```

The script runs two sequential CPU generation jobs:

| Output under `data_generation/single_table_datasets/` | Batches | Features | Rows per task |
| --- | --- | --- | --- |
| `single_table_stage1/` | 600,000 | 18 | 600 |
| `single_table_stage2/` | 600,000 | 30 | 600 |

Counts and paths are fixed in [single_table_generate.sh](single_table_generate.sh). Adjust them before a smaller run. Convert the outputs with [single_table_processing.sh](../../data_preprocessing/single_table_processing.sh), as described in the [preprocessing README](../../data_preprocessing/README.md).
