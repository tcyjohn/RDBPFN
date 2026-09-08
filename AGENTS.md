# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

Modified repository for [Relational In-Context Learning via Synthetic Pre-training with Structural Prior](https://arxiv.org/abs/2603.03805). A staged pipeline for pretraining a relational-database foundation model (RDB-PFN, LimiX-based architecture).

Pipeline: `data_generation/` → `data_preprocessing/` → `model_pretrain/` → `inference/`

## Build System

[pixi](https://pixi.sh) (Conda + PyPI hybrid). All dependencies declared in root `pixi.toml`.

```bash
pixi install          # install all deps
```

Subprojects (`data_generation/RDB/`, `data_generation/single_table/`, `data_preprocessing/`, `model_pretrain/`) each have a `pyproject.toml` and can be `pip install -e .` independently.

## Tasks

```bash
pixi run peek-parquet <file.parquet> [--head N] [--schema-only] [--json-schema]
```

## Data Paths

- Raw synthetic RDBs: `data_generation/RDB_datasets/` (active: `plurel_scale_complex_hsbm_v2`)
- DAG structure data: `data_generation/RDB/datasets/rdb_v1.pth`
- Preprocessed data: `~/scratch/relbench/`
- Pretraining `.h5` files: `model_pretrain/pretrain_datasets/`
- Checkpoints: `model_pretrain/checkpoints/RDBPFN/`, `inference/checkpoints/RDBPFN/`

## Architecture

### data_generation/RDB/ (primary active code)

Generates synthetic relational databases from DAG structures using Structural Causal Models (SCMs).

**Key source files under `src/`:**

- `src/table_def/table_generation.py` — Core: `Table`, `Relationship`, `RDB`, `TableGenerator` classes. `TableGenerator.generate_data()` drives the FK generation (HSBM) + column data generation (MLPSCM) pipeline.
- `src/prior/mlp_scm.py` — `MLPSCM`: MLP-based SCM that generates synthetic column values. Supports causal and non-causal modes, temporal sampling, multi-parent FK injection.
- `src/prior/hsbm.py` — HSBM-based bipartite FK generation (see HSBM section below).
- `src/prior/prior_config.py` — Default hyperparameter configs for SCM sampling (`DEFAULT_SAMPLED_HP`) and HSBM FK generation (`DEFAULT_HSBM_HP`).
- `src/prior/hp_sampling.py` — `HpSamplerList`: samples hyperparameters from distributions defined in config.
- `src/prior/row_gnn.py` — `RowGraphBuilder`, `RowGNNRunner`: optional row-level GNN refinement of generated embeddings.
- `src/prior/utils.py` — `GaussianNoise`, `XSampler`, `MASK_TYPE`, `SCM_OUTPUT` types.
- `src/prior/temporal_vocab.py` — Temporal vocabulary for timestamp generation.
- `src/prior/activations.py` — Activation function utilities (`get_activations`).
- `src/table_def/reg2cls.py` — Converts regression targets to classification targets.
- `src/table_def/task_generation.py` — `TaskGenerator`: generates ML tasks (predict a column) from generated RDBs.
- `src/table_def/task_generation_utils.py` — `SchemaGraph`, `InstanceGraph`, etc. for task generation.
- `src/table_def/dataset_meta.py`, `yaml_utils.py` — 4DBInfer metadata serialization.

**Entry point:** `dag_to_rdb_generator.py` — `DAGToRDBGenerator` loads DAG structures, creates table configs, instantiates `RDB` objects, initializes SCMs, generates data, and saves in 4DBInfer format.

### Generation Flow

```
DAG data (rdb_v1.pth)
  → DAGToRDBGenerator parses DAG → creates TableConfigs (rows, cols, PK/FK layout)
  → RDB.init_table_SCMs() samples HPs per table → creates MLPSCM per TableGenerator
  → RDB.generate_all_data_from_SCM() iterates tables in topological order:
      For each table:
        1. TableGenerator._compute_hsbm_fk_ids() → HSBM samples FK connections to parents
        2. MLPSCM.forward_with_input(parent_data, fk_ids) → generates X, Y, latent embeddings
        3. TableGenerator.cache_pending_outputs() → stores intermediate state
    Then (optional): RowGNNRunner refines row embeddings
    Then: _materialize_tables_from_pending() → Table.process_data() converts to final dtypes
  → RDB.initialize_tasks() → generates prediction tasks
  → RDB.save_to_4dbinfer_dataset_with_tasks() → outputs parquet + metadata.yaml
```

### HSBM FK Generation (current branch: `feature/hsbm-fk-generation`)

Replaces selective SCM FK generation with Hierarchical Stochastic Block Model (HSBM) bipartite sampling:

- **Single parent:** `hsbm.compute_hsbm_fk_ids()` — each child row samples one parent row via product of per-level block probabilities (diagonal=0.9 within-block, off-diagonal=Uniform(0.001,0.002) cross-block).
- **Multi-parent joint sampling:** `hsbm.compute_hsbm_fk_ids_multi()` — child samples a shared latent cluster path across hierarchy levels, then each parent FK is drawn within the aligned parent cluster. Tuple-level FK correlation comes from the shared latent variable.
- **Clipping:** `TableGenerator._clip_hierarchy()` ensures leaf block count ≤ min(parent_rows, child_rows).
- HP sampling: `DEFAULT_HSBM_HP` in `prior_config.py` — per parent, `hsbm_num_levels ∈ [1,5]`, `hsbm_clusters_per_level ∈ [1,3]`.

### Other pipeline stages

- `data_preprocessing/` — Converts raw RDBs to `.h5` pretraining datasets and benchmark task directories. Uses `dbinfer-bench` for on-disk dataset format + `relbench` for benchmarking.
- `model_pretrain/` — LimiX-based model training/eval. Configs in `conf_train/` and `conf_eval/`. Supports RDB-PFN and baselines (XGBoost, TabPFN, TabICL, AutoGluon).
- `inference/` — Standalone lightweight inference on flat CSV data.

## Main Pipeline Scripts

```bash
# Full pipeline (data gen → relbench convert → rustler pre → embed):
bash scripts/generateRDB_hsbm_v2.sh

# Just generate raw RDB data:
pixi run python ./data_generation/RDB/dag_to_rdb_generator.py \
  --dag_data_path data_generation/RDB/datasets/rdb_v1.pth \
  --config_file data_generation/RDB/dag_to_rdb_config_small.yaml \
  --num_rdbs 128 --start_index 0 \
  --output_base_dir data_generation/RDB_datasets/my_output \
  --use_complex_tasks True

# Convert to relbench format:
pixi run python ./scripts/convert_dag_rdb.py \
  --range 0 128 --src-root data_generation/RDB_datasets/my_output \
  --dst-prefix "dag_rdb_complex_hsbm_v2_"
```

## Environment

- 2× RTX 4090 GPUs — beware OOM
- Mainland China network — use HF mirrors (`export HF_ENDPOINT=https://hf-mirror.com`)
- Default device in code is `"cpu"` — change to `"cuda"` for GPU
