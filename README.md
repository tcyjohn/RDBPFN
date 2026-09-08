# SA_RDB_PFN

A research fork of RDB-PFN for synthetic pretraining on relational databases, based on [Relational In-Context Learning via Synthetic Pre-training with Structural Prior](https://arxiv.org/abs/2603.03805).

The relational generator adds HSBM foreign-key sampling, signal-group features, entity temporal snapshots, and task quality filtering. Training and evaluation support optional foreign-key and entity attention biases.

## Project Layout

| Directory | Purpose |
| --- | --- |
| [data_generation](data_generation/README.md) | Generate relational databases and single-table priors. |
| [data_preprocessing](data_preprocessing/README.md) | Apply Deep Feature Synthesis (DFS) and build HDF5 training corpora. |
| [model_pretrain](model_pretrain/README.md) | Train models and evaluate checkpoints or baselines. |
| [inference](inference/README.md) | Apply the standalone RDBPFN classifier to flat tables. |
| [scripts](scripts) | Pipeline launchers and experiment utilities. |

## Environment

From the repository root:

```bash
pixi install
```

[pixi.toml](pixi.toml) and `pixi.lock` define the shared Linux x86-64 / Python 3.10 environment, including a CUDA 12.4 PyTorch package source. Run commands through `pixi run` to use it. The single-table generator and optional baseline models have additional dependencies described in their READMEs.

For Hugging Face access through a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Environment installation does not download training corpora or create experiment checkpoints. Upstream data are published separately at [yamboo/RDB_PFN](https://huggingface.co/datasets/yamboo/RDB_PFN); fork-specific experiment paths in configs refer to locally generated corpora.

## Run the Pipeline

Start with the small generation command in [data_generation/README.md](data_generation/README.md), then follow preprocessing and training in order.

The combined launcher [scripts/run_pipeline.sh](scripts/run_pipeline.sh) generates database candidates, runs pre-DFS/DFS/post-DFS processing, merges an HDF5 prior, prepares evaluation CSVs, and starts training. It requires the DAG file `data_generation/RDB/datasets/rdb_v1.pth`, the initialization checkpoint in [RDBPFN_hsbm.yaml](model_pretrain/conf_train/RDBPFN_hsbm.yaml), and enough GPU memory for the configured run. Its evaluation preparation stage replaces the contents of `model_pretrain/datasets/clf/`. Inspect its positional arguments before use. Supply `+train.run_final_eval=true` (or `false`) through its `extra_train_args` argument: the training entry point reads that field, but the HSBM YAML preset does not define it. The stage READMEs provide explicit commands with the required overrides.

| Artifact | Location |
| --- | --- |
| Generated relational databases | `data_generation/RDB_datasets/<run_name>/` |
| DFS outputs | `data_generation/RDB_datasets/<run_name>-processed/` |
| Training corpus | `model_pretrain/pretrain_datasets/<run_name>.h5` |
| Training checkpoints | `model_pretrain/checkpoints/<run_name>/` |
| Evaluation results | `model_pretrain/results/` |

Individual ablation and queue scripts contain experiment-specific paths and settings; adapt those before reuse. For prediction on your own flat data, use the independent [inference package](inference/README.md).
