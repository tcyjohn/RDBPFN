# Model Training and Evaluation

Train on HDF5 priors and evaluate checkpoints on relational task directories or flat classification datasets. Commands below start from the repository root unless a different working directory is specified.

## Environment and Data

```bash
pixi install
```

The root Pixi environment supplies the core training dependencies, including PyTorch, Accelerate, Hydra, and `schedulefree`. Optional baselines may require separate dependencies. [pyproject.toml](pyproject.toml) is an alternative pip dependency manifest: it includes AutoGluon but does **not** install PyTorch. In a separate environment, install a suitable PyTorch build first, then use:

```bash
python -m pip install -e model_pretrain
# Additionally install the pinned TabPFN and TabICL baseline packages:
python -m pip install -e 'model_pretrain[all-baselines]'
```

| Input/output | Path relative to `model_pretrain/` |
| --- | --- |
| Training priors | `pretrain_datasets/*.h5` |
| Relational evaluation tasks | `rdb_datasets/<dataset>/` |
| Flat evaluation datasets | `datasets/clf_real/`, `datasets/clf_rel/` |
| Checkpoints | `checkpoints/<run>/` |
| Evaluation results | `results/` |

Generate fork-specific priors using [preprocessing](../data_preprocessing/README.md). Config paths must match your local files. The upstream data collection is [yamboo/RDB_PFN](https://huggingface.co/datasets/yamboo/RDB_PFN); it is not a download source for every fork-specific experiment named in these configs.

## Training

[run_train.py](run_train.py) launches [src/train.py](src/train.py). Presets in [conf_train](conf_train) specify corpus paths, checkpoint initialization, model settings, and evaluation cadence:

| Preset | Use |
| --- | --- |
| `RDBPFN_single` | Single-table initialization from `single_table_stage1.h5`. |
| `RDBPFN` | Original multi-corpus relational and single-table mixture. |
| `RDBPFN_hsbm` | One relational corpus; FK and entity biases enabled. |
| `RDBPFN_mix_v62_r3` | Two separate source corpora; entity bias enabled and FK bias disabled. |

Single-GPU initialization:

```bash
cd model_pretrain
CUDA_VISIBLE_DEVICES=0 pixi run torchrun --standalone --nproc_per_node=1 \
  run_train.py --config-name=RDBPFN_single \
  +train.full_eval_steps=0 +train.full_eval_dataset_dir=rdb_datasets \
  '+train.full_eval_seeds=[0]' +train.run_final_eval=true
```

For an existing `pretrain_datasets/my_run.h5`, run from `model_pretrain/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run torchrun --standalone --nproc_per_node=2 \
  run_train.py --config-name=RDBPFN_hsbm \
  train.datasets.0.path=pretrain_datasets/my_run.h5 \
  train.save_model_path=checkpoints/my_run/model.pt \
  wandb.enabled=false +train.run_final_eval=true
```

The explicit `+train.*` arguments supply fields read by the training entry point but absent from these YAML presets. A Python dataclass default does not automatically add a missing Hydra YAML key.

The relational preset initializes from `checkpoints/RDBPFN_single/model_eval00360.pt`; change `train.load_model_path` to your own compatible checkpoint when needed. Set the launcher process count and `train.num_gpus` consistently for your hardware. `train.num_steps` counts optimizer updates; gradient accumulation changes the number of batches consumed per update. Periodic and final evaluation need the datasets selected by the training config. If W&B is enabled, configure its credentials.

## Relational Evaluation

Use [src/eval_aligned.py](src/eval_aligned.py) to keep sampled features, labels, FK values, entity IDs, and parent entity IDs aligned. From the repository root:

```bash
cd model_pretrain
CUDA_VISIBLE_DEVICES=0 pixi run python -m src.eval_aligned \
  dataset=full-512 model=RDBPFN \
  model.checkpoint_path=checkpoints/RDBPFN/model_eval00528.pt \
  'dataset.seeds=[0,1,2]' \
  +output_path=results/rdbpfn_full512.csv
```

[full-512.yaml](conf_eval/dataset/full-512.yaml) lists the relational benchmark directories and limits the training context to 512 rows. Other `full-*` presets change that limit. The default `RDBPFN` model config disables FK/entity biases for the original checkpoint. For a checkpoint trained with them, set `model.nanopfn.use_fk_bias` and `model.nanopfn.use_entity_bias` to match the training configuration and provide the required structural metadata.

Evaluation saves an aggregate CSV and a companion `*_per_seed.csv`. Hydra requires `+output_path` because that key is not present in the base YAML. To override chunk size, add `+model.eval_chunk_size_override=500` to the command. Chunk size changes which test rows are processed together and should be kept consistent when comparing runs.

## Flat-Table Evaluation

From `model_pretrain/`:

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python -m src.eval_aligned \
  --config-name=eval_csv dataset=clf_npz model=RDBPFN \
  +output_path=results/rdbpfn_clf.csv
```

`clf_npz` selects `datasets/clf_real/`; `clf_rel_npz` selects `datasets/clf_rel/`. Inspect their [dataset configs](conf_eval/dataset) for split, sampling, and caching settings. Flat inputs do not supply relational identity metadata.

Baseline presets are in [conf_eval/model](conf_eval/model). Choose a preset with `model=<name>` after installing its dependencies. The standalone LimiX baselines require their own environment. [scripts/run_eval_aligned.sh](../scripts/run_eval_aligned.sh) is an experiment wrapper with a machine-specific interpreter path; the module commands above avoid that path.
