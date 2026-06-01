# RDBPFN Evaluation Guide

## 1. Eval 对象

### 1.1 Tabular 分类数据集 (CSV/npz)

预处理好的 `.npz` 文件已缓存，直接加载无需重新生成。

| Config 名称 | 数据源目录 | 缓存目录 | 数据集数量 | 说明 |
|---|---|---|---|---|
| `clf_rel_npz` | `datasets/clf_rel/` | `datasets/clf_rel_subsamples/` | 19 | Relbench 关系型分类任务 |
| `clf_npz` | `datasets/clf_real/` | `datasets/clf_real_subsamples/` | 19 | Real-world tabular 分类数据集 |

**clf_rel_npz 数据集列表** (Relbench):

| 数据集 | 类别数 | imbalance_ratio |
|---|---|---|
| Amazon-dfs-2__churn | 2 | 0.87 |
| avs-dfs-2__repeater | 2 | 0.37 |
| diginetica-dfs-2__ctr | 2 | 0.008 |
| outbrain-small-dfs-2__ctr | 2 | 0.20 |
| rel-amazon-dfs-2__item-churn | 2 | 0.78 |
| rel-amazon-dfs-2__user-churn | 2 | 0.66 |
| rel-avito-dfs-2__user-clicks | 2 | 0.022 |
| rel-avito-dfs-2__user-visits | 2 | 0.10 |
| rel-event-dfs-2__user-ignore | 2 | 0.18 |
| rel-event-dfs-2__user-repeat | 2 | 0.93 |
| rel-f1-dfs-2__driver-dnf | 2 | 0.14 |
| rel-f1-dfs-2__driver-top3 | 2 | 0.21 |
| rel-hm-dfs-2__user-churn | 2 | 0.20 |
| rel-stack-dfs-2__user-badge | 2 | 0.048 |
| rel-stack-dfs-2__user-engagement | 2 | 0.049 |
| rel-trial-dfs-2__study-outcome | 2 | 0.57 |
| retailrocket-dfs-2__cvr | 2 | 0.035 |
| stackexchange-dfs-2__churn | 2 | 0.12 |
| stackexchange-dfs-2__upvote | 2 | 0.42 |

每个 `.npz` 文件包含: `X_train`, `X_test`, `y_train`, `y_test`，均已截断到最多 1000 行。

### 1.2 合成 RDB 数据集 (relbench/dbinfer-bench 格式)

| Config 名称 | 说明 |
|---|---|
| `full-512` | 13 个合成 RDB，max_train_samples=512 (默认) |
| `full-1024` | 同上，max_train_samples=1024 |
| `full-256` / `full-128` / `full-64` / `full-32` | 更小的训练集变体 |
| `full-debug` | 小型调试集 |

数据路径: `rdb_datasets/` 下的 `amazon-dfs-2`, `avs-dfs-2`, `diginetica-dfs-2` 等 13 个目录。

### 1.3 可用模型 Config

| Config | 说明 |
|---|---|
| `RDBPFN` | 默认 RDBPFN 模型 (embedding_size=96, 6 layers) |
| `RDBPFN_single` | RDBPFN 单变体 |
| `tabpfnv2` / `tabpfnv25` / `tabpfnv25_lite` | TabPFN 系列 |
| `tabiclv1` / `tabiclv11` / `tabiclv11_lite` | TabICL 系列 |
| `limix2m` / `limix16m` / `limix16m_lite` | LimiX 系列 |
| `autogluon-medium` / `autogluon-mitra` | AutoGluon |
| `xgboost` / `random_forest` | 传统 ML baseline |

---

## 2. 运行方式

### 2.1 使用 eval wrapper 脚本 (推荐)

```bash
bash scripts/run_eval.sh <checkpoint_path> [dataset_config] [gpu_id]
```

参数:
- `checkpoint_path`: `.pt` checkpoint 文件路径 (必需)
- `dataset_config`: `conf_eval/dataset/` 下的 config 名称 (默认: `full-512`)
- `gpu_id`: GPU 编号 (默认: 0)

示例:
```bash
# 在 clf_rel 上评估 RDBPFN checkpoint
bash scripts/run_eval.sh checkpoints/RDBPFN/model_eval00528.pt clf_rel_npz 0

# 在 clf_real 上评估 v5.1_relmode
bash scripts/run_eval.sh checkpoints/v5.1_relmode/model.pt clf_npz 1

# 在合成 RDB 上评估
bash scripts/run_eval.sh checkpoints/RDBPFN_single/model_eval00360.pt full-512 0
```

### 2.2 直接调用 Python

```bash
cd model_pretrain
pixi run python -m src.eval \
    --config-name=eval_csv \
    dataset=clf_rel_npz \
    model.checkpoint_path=checkpoints/RDBPFN/model_eval00528.pt
```

- `--config-name=eval_csv`: 使用 CSV/npz 评估模式
- `dataset=clf_rel_npz` 或 `dataset=clf_npz`: 选择数据集
- 合成 RDB 评估不使用 `eval_csv`，直接用默认 config + `paths` 配置

---

## 3. 评估流程

### 3.1 CSV/npz 模式 (`_evaluate_csv_datasets`)

```
cfg.dataset.dirs (e.g. datasets/clf_rel)
    │
    ▼
prepare_eval_splits() → _load_local_csv_datasets()
    │  扫描目录下 *.csv 文件
    │  检查 {dir}_subsamples/{name}_split.npz 缓存
    │  有缓存 → 直接加载 (X_train, X_test, y_train, y_test)
    │  无缓存 → 从 CSV 生成 (stratified 70/30 split, 截断到 1000 行, 保存缓存)
    │
    ▼
对每个 dataset、每个 seed:
    1. downsample_split (均匀随机, 非 stratified)
    2. fill_nans
    3. classifier.fit(X_train, y_train)
    4. predict_proba_in_chunks(X_test)
    5. 计算 roc_auc, accuracy, balanced_acc
    │
    ▼
聚合: 每个 dataset 的 10-seed 平均 AUROC → 总体平均
```

### 3.2 RDB 模式 (`_evaluate_datasets`)

```
cfg.dataset.paths (e.g. rdb_datasets/amazon-dfs-2)
    │
    ▼
DBBRDBDataset(path) → 遍历 tasks
    │  过滤: 仅 classification tasks
    │  load_task_split(task, "train"/"test")
    │
    ▼
对每个 dataset、每个 seed:
    1. downsample_split
    2. fill_nans
    3. classifier.fit → predict_proba
    4. compute_metric (使用 task.metadata.evaluation_metric)
    │
    ▼
聚合: 每个 task 的多 seed 平均 → 总体平均
```

### 3.3 输出

- 逐 dataset 平均 AUROC (10 seeds)
- 总体平均 AUROC
- CSV 结果保存到 `model_pretrain/results/`

---

## 4. 注意事项

- **GPU 内存**: 2x RTX 4090，多个 checkpoint 评估需串行执行，避免 OOM
- **Class imbalance**: `downsample_split` 使用均匀随机采样 (非 stratified)，极端不平衡数据集 (如 diginetica ctr, ratio=0.008) 的 AUROC 可能不稳定
- **缓存机制**: `.npz` 缓存存在即直接加载；如需重新生成，删除 `*_subsamples/` 下对应文件
- **Hydra override**: 未在 yaml 中声明的字段需用 `+field=value` 语法
