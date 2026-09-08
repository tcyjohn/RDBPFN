# Model Pretraining

This directory contains the training and evaluation stage for RDB_PFN. It includes:

- Hydra configs for training and evaluation
- model implementation and training loops
- evaluation code for RDB_PFN and baseline models
- local directories for checkpoints and datasets

## Directory Highlights

- [src/train.py](src/train.py): main pretraining entry point.
- [src/eval.py](src/eval.py): main evaluation entry point.
- [conf_train/RDBPFN_single.yaml](conf_train/RDBPFN_single.yaml): single-table pretraining config.
- [conf_train/RDBPFN.yaml](conf_train/RDBPFN.yaml): final RDB foundation model pretraining config.
- [conf_eval/dataset](conf_eval/dataset): dataset presets for evaluation.
- [conf_eval/model](conf_eval/model): model presets for RDB_PFN and baselines.

## Installation

This stage has a dependency manifest at [pyproject.toml](pyproject.toml).

Recommended installation order:

1. Install a `torch` build that matches your machine.
2. Install the default dependencies in `model_pretrain/`.
3. Optionally install the extra baseline dependencies.

The default dependency set includes:

- PyTorch
- Hydra
- Accelerate
- `schedulefree`
- NumPy and scikit-learn
- AutoGluon

The optional extra `all-baselines` additionally installs:

- TabPFN
- TabICL

Example commands after installing the correct Torch build:

```bash
pip install -e model_pretrain
pip install -e model_pretrain[all-baselines]
```

Torch note:

- `pyproject.toml` intentionally does not install PyTorch automatically.
- Users should install Torch manually so they can choose the correct CPU or CUDA build for their platform.
- This avoids mismatches between the installed Torch wheel and the user GPU environment.

LimiX note:

- LimiX is intentionally not included in `pyproject.toml`.
- If you want to evaluate the LimiX baselines, create a separate environment and follow the guidelines in [LimiX](https://github.com/stableai-org/LimiX).

## Current Fork Entry Points

For the shared Linux environment, run `pixi install` at the repository root. [run_train.py](run_train.py) and [run_eval.py](run_eval.py) are launch wrappers. Additional training presets are:

- [RDBPFN_hsbm.yaml](conf_train/RDBPFN_hsbm.yaml): relational HDF5 training with optional FK/entity attention biases (both enabled in this preset).
- [RDBPFN_mix_v62_r3.yaml](conf_train/RDBPFN_mix_v62_r3.yaml): source-weighted training from two separate HDF5 corpora, with entity bias enabled and FK bias disabled.

Edit corpus/checkpoint paths before training. For an existing `pretrain_datasets/my_run.h5`, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash scripts/run_train.sh my_run RDBPFN_hsbm
```

Training now counts `train.num_steps` in optimizer updates. Gradient accumulation and GPU count must be accounted for when comparing against historical runs. Optional structural arrays (`fk_values`, `entity_ids`, `parent_entity_ids`) are passed through the loader and model; enable the bias settings appropriate to the checkpoint and corpus.

### Row-Aligned Evaluation

Use [src/eval_aligned.py](src/eval_aligned.py) when evaluating structural biases: subsampling applies the same row indices to features, labels, and structural metadata. From the repository root:

```bash
cd model_pretrain
CUDA_VISIBLE_DEVICES=0 pixi run python -m src.eval_aligned \
  dataset=full-512 model=RDBPFN \
  model.checkpoint_path=checkpoints/RDBPFN/model_eval00528.pt \
  'dataset.seeds=[0,1,2]' \
  model.nanopfn.use_fk_bias=false model.nanopfn.use_entity_bias=false \
  output_path=results/rdbpfn_full512.csv
```

The example disables the added biases for the original checkpoint. Set them to match a newly trained checkpoint; entity identity also depends on the evaluation dataset metadata. Evaluation writes aggregate results and a companion `*_per_seed.csv`. `model.eval_chunk_size_override` controls prediction chunk size. The [shell wrapper](../scripts/run_eval_aligned.sh) is also available but currently contains a local interpreter path; use the module command above on other machines.

The examples below document the original training and baseline workflows.

## Data Layout

### Pretraining Datasets

Expected under `model_pretrain/pretrain_datasets/`.

Current training configs reference:

- synthetic single-table `.h5` priors
- synthetic RDB-derived `.h5` priors

### Evaluation Datasets

Expected under `model_pretrain/datasets/` for single-table evaluation or `model_pretrain/rdb_datasets/` for RDB evaluation.


### Checkpoints

Checkpoints are expected under `model_pretrain/checkpoints/`.

Two repository-known checkpoint paths are already referenced by configs:

- `checkpoints/RDBPFN_single/`
- `checkpoints/RDBPFN/`

## Dataset Download Guide

All pretrain required datasets, and evaluation required datasets are provided at [Huggingface](https://huggingface.co/datasets/yamboo/RDB_PFN). You can download them and use them directly for pretraining and evaluation.

## Evaluate a Model

Evaluation is Hydra-based and starts from [src/eval.py](src/eval.py) for RDB and [src/eval_csv.py](src/eval_csv.py) for single-table.

### Example: Evaluate RDBPFN on all RDBs under a given shot number preset

From the repository root:

```bash
cd model_pretrain
python -m src.eval dataset=full-1024 model=RDBPFN
```

This uses:

- the existing dataset preset `conf_eval/dataset/full-1024.yaml`
- the model preset `conf_eval/model/RDBPFN.yaml`

### Example: Switch Evaluation Shots or Model Presets

```bash
python -m src.eval dataset=full-512 model=RDBPFN_single
```

You can swap `model=` to other provided presets such as:

- `RDBPFN`
- `RDBPFN_single`

- `xgboost`
- `random_forest`
- `autogluon-medium`

- `tabpfnv25`
- `tabpfnv25_lite`
- `tabpfnv2`
- `tabiclv11`
- `tabiclv11_lite`
- `tabiclv1`
- `autogluon-mitra`
- `limix16m`
- `limix16m_lite`
- `limix2m`


### Example: Evaluation on Single-Table Datasets

```bash
cd model_pretrain
python -m src.eval_csv model=RDBPFN dataset=clf_npz
```

## Pretrain the Model

Training is Hydra-based and starts from [src/train.py](src/train.py).

### Step 1: Pretrain the Single-Table Initialization Model

The single-table config is [conf_train/RDBPFN_single.yaml](conf_train/RDBPFN_single.yaml).

Example:

```bash
cd model_pretrain
python -m accelerate.commands.launch --num_processes 1 -m src.train --config-name RDBPFN_single
```

This stage trains from single-table priors and saves into `checkpoints/RDBPFN_single/`.

### Step 2: Pretrain the Final RDB Foundation Model

The full RDB config is [conf_train/RDBPFN.yaml](conf_train/RDBPFN.yaml).

Example:

```bash
python -m accelerate.commands.launch --multi_gpu --num_processes 8 -m src.train --config-name RDBPFN
```

This config currently mixes multiple RDB-derived and single-table-derived `.h5` datasets and initializes from `checkpoints/RDBPFN_single/`.

### Training Notes

- The training is parallelized across multiple GPUs using `accelerate`.
- We found that training results can vary slightly across machines, so we also provide the final trained model checkpoints.
- If `wandb.enabled=true`, `WANDB_API_KEY` must be set in the environment.
