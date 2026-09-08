# RDB_PFN

This is a modified research fork of the code for the paper [Relational In-Context Learning via Synthetic Pre-training with Structural Prior](https://arxiv.org/abs/2603.03805). It presents a synthetic pre-training framework for relational-database foundation models.

The repository is organized as a staged pipeline:

1. Generate synthetic single-table and relational data.
2. Preprocess generated datasets into training formats.
3. Pretrain the foundation model and evaluate it on downstream benchmark datasets.
4. Provide a simple inference interface for applying the model to arbitrary user datasets when needed.

## Setup and Current Workflow

The shared environment is defined in [pixi.toml](pixi.toml) and pinned by `pixi.lock` (Linux x86-64, Python 3.10):

```bash
pixi install
# Optional for Hugging Face access from mainland China:
export HF_ENDPOINT=https://hf-mirror.com
```

This fork adds hierarchical stochastic block model (HSBM) foreign-key sampling, signal-group feature generation, entity temporal snapshots, task quality filtering, and optional FK/entity attention biases. Preprocessing preserves structural metadata through HDF5 loading, training, and evaluation.

From the repository root:

```bash
bash scripts/run_pipeline.sh 4 0 my_run
```

This runs generation, pre-DFS/DFS/post-DFS preprocessing, HDF5 merging, evaluation CSV preparation, and training. It needs `data_generation/RDB/datasets/rdb_v1.pth` and the initialization checkpoint configured in `model_pretrain/conf_train/RDBPFN_hsbm.yaml`. Read the positional arguments in [scripts/run_pipeline.sh](scripts/run_pipeline.sh) before launching: its CSV preparation stage replaces `model_pretrain/datasets/clf/` contents. Set GPU visibility explicitly for your machine.

See the stage READMEs below for generation-only commands, HDF5 metadata, and row-aligned evaluation. [Generation details](docs/data_generation_pipeline.md) and [evaluation notes](docs/evaluation.md) describe the implementation and evaluation workflow; current code/configs define defaults. Scripts for individual experiments may contain machine-specific paths and checkpoint names; inspect them before reuse.

Generated datasets, checkpoints, and local experiment outputs are not installed by `pixi install`.

## Project Structure

`data_generation/`
Generates pretraining corpora. It contains two subprojects:

- `data_generation/single_table/`: synthetic single-table task generation.
- `data_generation/RDB/`: synthetic relational database generation.

`data_preprocessing/`
Processes generated data into the formats used by pretraining. It supports both single-table and relational workflows.

`model_pretrain/`
Contains model configs, training code, evaluation code, baseline model configs, and local paths for datasets/checkpoints.

`inference/`
Provides a standalone lightweight inference package for quick use on flat data.

## Recommended Reading Order

If you are new to the repository, read the documentation in this order:

0. If you only want a quick trial of the released model on your own data, start with [inference/README.md](inference/README.md). This is the lightweight standalone path and does not require understanding the full generation, preprocessing, or pretraining pipeline.
1. This README for the overall pipeline.
2. [data_generation/README.md](data_generation/README.md) to generate raw synthetic data.
3. [data_preprocessing/README.md](data_preprocessing/README.md) to convert raw data into pretraining and evaluation datasets.
4. [model_pretrain/README.md](model_pretrain/README.md) to evaluate checkpoints or pretrain a model.

We also provide well-processed pretraining datasets and benchmark datasets formatted for our model at [Huggingface](https://huggingface.co/datasets/yamboo/RDB_PFN). You can download them and use them directly for pretraining and evaluation.

## End-to-End Pipeline

### Stage 1: Data Generation

Use `data_generation/single_table/` to build synthetic single-table priors and `data_generation/RDB/` to build synthetic relational databases.

Outputs from this stage include:

- raw single-table batches under `data_generation/single_table_datasets/`
- raw synthetic RDBs under `data_generation/RDB_datasets/`

### Stage 2: Data Preprocessing

Use `data_preprocessing/` to convert generation outputs or benchmark datasets into `.h5` or benchmark-specific task directories.

Outputs from this stage include:

- pretraining `.h5` files under `model_pretrain/pretrain_datasets/`
- optional intermediate `.h5` files under `data_preprocessing/RDB_datasets/`

### Stage 3: Model Pretraining and Evaluation

Use `model_pretrain/` to:

- pretrain a single-table initialization model
- continue to pretrain the final RDB foundation model from the single-table initialization model
- evaluate RDB_PFN or baseline models on benchmark datasets

## Repository Status

Currently available:

- synthetic single-table generation
- synthetic RDB generation
- single-table preprocessing
- RDB preprocessing
- model pretraining
- model evaluation
- standalone inference
