# v6.x Experiment Handoff

每个版本的 **数据生成差异** + **模型/训练差异** + **关键实验结果**。

---

## 公共基础

以下配置在所有 v6.x 版本中保持不变：

- **训练 config**: `conf_train/RDBPFN_hsbm.yaml`
  - `embedding_size=96, num_attention_heads=4, mlp_hidden_size=192, num_layers=6`
  - `batch_size=1, gradient_accumulation_steps=2, num_gpus=2 → effective batch=4`
  - 初始 checkpoint: `checkpoints/RDBPFN_single/model_eval00360.pt`（单表预训练权重）
  - `full_eval_steps=16000, full_eval_dataset_dir=rdb_datasets` — full128 评测使用 19 个 held-out synthetic tasks
- **数据生成 config**: `dag_to_rdb_config_small.yaml`
  - `use_signal_group_features=True`（信号组特征生成）
  - `snapshots_per_entity ∈ [5, 20], entity_timestamp_prob=1.0`
  - `num_processes=16`
- **预处理**: `pre-dfs → dfs-2 → post-dfs`, `max-columns=90`

---

## 关键定义

- **entity table**（实体表）：out-degree ≥ 1，即有至少一个 child 表通过 FK 引用它的表
  - **root entity**: in-degree = 0（没有 FK 指向任何 parent），位于 DAG 顶层
  - **child entity**: in-degree ≥ 1（有 FK 指向 parent），位于 DAG 中间层
- **entity coverage**：训练/评测数据中有 `entity_id` 列（≥ 0）的 sample/task 比例。v6.2 relbench_mode 下 ≈ 100%（所有 task 的 focal 都是 entity table）
- **entity pair ratio**：同 entity_id 的 row pairs / 全部 row pairs。在 600-row sample 中，理论上限 ≈ 1/E（E = 该 sample 中 unique entity 数量）
- **FK pair ratio**：FK sibling pairs（同 FK 值） / 全部 row pairs。root entity 表无 FK 列 → FK pair ratio = 0
- **relbench_mode**：`root_p=1.0, entity_task_ratio=1.0` — focal 从 entity table 中选（70% bias toward child entity，即有 FK parent 的 entity），target = focal（DIRECT_ATTR_PREDICTION），child features 通过 DFS join 聚合。实际观测（由于三层dag图相对更少） ~80% 任务 target 落在 root entity（无 FK parent），~20% 落在 child entity（有 FK 列）
- **parent_entity_ids**：entity-level FK key。将 raw FK（parent row index）映射为 parent entity 的 PK 值，使 FK matching 为 entity-level 而非 row-level（一个 parent entity 的多个快照 → 同一 key）

---

## v6（基线，2026-06-09 前）

**数据生成差异**:
- 1024 RDBs, 10 seeds
- `relbench_mode=True` (root_p=1.0, entity_task_ratio=1.0)
- SG (signal group) 基线: intrinsic from MLP CAUSAL_OUTPUT
- FK sparsity 启用, complex tasks 启用, quality gate 启用
- homophily labels=off, FK bias=off, entity bias=off

**模型/训练差异**:
- 无 FK bias, 无 entity bias

**结果**: full128 avg_auroc 0.6759 (v6/model_eval00304.pt)

---

## v6.1（entity_id 修复 + entity bias 上线，2026-06-15~17）

**数据生成差异**:
- 相对 v6: 修复 entity_id 被 homophily 覆盖的 bug
- `homophily_labels=False`（关闭 homophily 以验证 entity_id 修复）
- `relbench_mode=True`（保持）
- 1024 RDBs, 1 seed

**模型/训练差异**:
- `use_entity_bias=True`（首次启用 EntityAttentionBias）
- `use_fk_bias=False`
- **代码新增**: `EntityAttentionBias` 类（`model_pretrain/src/models.py`）
  - 单标量 λ_se，softplus 约束，初始值 0.1
  - `I[entity_id[i] == entity_id[j]] × λ_se` 加到 attention scores

**关键发现**:
- Entity bias 学到 λ_se = 0.48（Layer 5），验证 entity_id 修复成功
- 但 same-entity pair ratio 从伪 42%（bug 时期）暴跌到 1.35%（真实值），entity 覆盖率 ~100%（relmode 下所有任务来自 entity 表）
- 训练 loss 日志 bug：`original_loss_sum` / `total_loss_sum` 初始化后从未累加，log 中始终显示 `loss=0.0000`（`model_pretrain/src/training.py` 行 993-1098）

**结果**: full128 peak 0.6871 (v6.1/model_eval00112.pt)

---

## v6.2（FK bias + parent_entity_ids 上线，2026-06-18）

**数据生成差异**:
- 相对 v6.1: 新增 `parent_entity_ids` 计算（entity-level FK key，用于跨快照和跨 child 行共享同一 parent entity 标识）
- `relbench_mode=True` (root_p=1.0, entity_task_ratio=1.0)
- 1024 RDBs, 1 seed
- **注意**: root_p=1.0 → target 总是 root entity → ~80% 任务 target root entity, ~20% target child entity

**模型/训练差异**:
- `use_fk_bias=True`（首次启用 FKAttentionBias）
- `use_entity_bias=True`
- **代码新增** (`model_pretrain/src/models.py`):
  - `FKAttentionBias`: per-column λ_j（vector），`bias = Σ_j λ_j × I[FK_i == FK_j in column j]`
  - 动态扩展: `_lambdas_for(K)` 按需扩展 `raw_lambdas` 向量
- **代码新增** (`data_preprocessing/merge_dbinfer_to_h5.py`):
  - `parent_entity_ids` 计算: 将 FK 值映射为 parent entity 的 PK 值（替代 raw FK row index），使 FK matching 为 entity-level
- **代码新增** (`model_pretrain/src/dataloaders.py`):
  - `InMemoryDataset` 支持从 H5 加载 `parent_entity_ids`
- **代码新增** (`model_pretrain/src/training.py`):
  - `parent_entity_ids` 从 dataloader → model forward 的完整链路

**关键发现**:
- FK pair ratio = 0.006%（几乎为零）— root entity 无 FK columns（全部 -1），只有 ~31% 的 child entity 任务有 FK 信号
- Parent entity pair ratio = 0.163%（使用 entity-level matching 略有提升）
- Entity pair ratio = 1.35%, entity coverage = 100%
- FK bias: **完全冻结**在初始值（所有 6 层 × 5 列 = 30 个 λ 全是 -2.2522 → softplus = 0.1）
- Entity bias: **正常学习**，λ_se (L5) = 1.57
- FK dim = 5（因 root entity 任务 FK 列全部为空，max 由 31% child 任务决定）

**结果**: full128 peak **0.7016** (v6.2/model_eval00112.pt, step 112000), relbench avg 0.6667

**分析**: FK bias 完全不学但性能反而最好。核心不是 FK bias 有害，而是 relmode 下 entity coverage=100% + DIRECT_ATTR 任务偏简单，entity bias 单独工作已经足够。

---

## v6.2.1（PK → entity_id 映射 + eval bias 修复，2026-06-22）

**目标**: 修复 eval 时 entity bias 和 FK bias 完全不生效的问题。

**根因分析**:
- 真实 RelBench eval 数据（`rdb_datasets/*/train.npz`）没有 `entity_id` 列 → `load_task_split` 返回 `entity_ids=None` → Entity bias 短路（`models.py:316`）
- 8/13 的 eval dataset（所有 `rel-*`）target 是 root entity 表 → 无 FK 列 → FK bias 短路
- 但 eval 数据中 **PK 列**（如 `driverId`、`customer_id`）天然是 entity grouping：rel-f1 driver-top3 的 `driverId` 有 2.05% pair ratio（128-row 采样），比 v6.3 FK pair ratio（1.26%）还强
- Eval config `conf_eval/model/RDBPFN.yaml` 中 `use_fk_bias: false, use_entity_bias: false` → 即使有 bias 数据，模型也没有 bias 模块

**代码改动**:
1. **`model_pretrain/src/eval_utils.py:260-268`** — PK → entity_id fallback:
   - `load_task_split` 在无 `entity_id` 列时，fallback 到 task metadata 中第一个 `primary_key` 列
   - 使 eval 时 entity bias 能拿到 entity grouping 信号
2. **`model_pretrain/conf_eval/model/RDBPFN.yaml`** — `use_fk_bias: true, use_entity_bias: true`
   - Standalone eval 构建模型时包含完整的 FK/entity bias 模块
   - 旧 checkpoint（无 bias 参数）通过 `strict=False` 加载，bias 从 init（λ=0.1）开始

**效果**:
- v6.2 checkpoint 重新 eval：entity bias 首次在 eval 中激活，λ_se 从训练中学会的 1.57 被实际使用
- 对所有 eval dataset 生效：PK 列普遍存在于 entity table 任务中

**待验证**:
- v6.2 checkpoint + 新 eval code 的 full-512 结果
- 训练时 entity bias 学到的 λ 是否在真实 PK 分组上有效（训练 entity 密度 ≠ eval entity 密度）

---

## v6.3（relmode OFF + Zipf FK + 单标量 FK λ，2026-06-19~21）

**目标**: 让 FK bias 有 100% 覆盖率的信号可供学习，同时提高 FK sibling density。

**数据生成差异**:
- `relbench_mode=False`: root_p=0.0 → target 是 leaf table
  - entity_task_ratio=0.75（75% 任务有 entity focal）
  - 所有 leaf 任务都有 FK columns（FK bias 全量激活）
- **entity-level Zipf FK sampling** (`data_generation/RDB/src/prior/hsbm.py`):
  - 新增 `_compute_zipf_weights()` 函数: `weight = 1/rank^alpha`，以 parent entity 为单位（非 per-row）
  - `hsbm_zipf_alpha ∈ Uniform(1.1, 2.0)` — 加到 `DEFAULT_HSBM_HP`
  - 注入到全部 4 条 HSBM 路径: `_sample_fk_per_parent`, `compute_hsbm_fk_ids`, `_with_propensity`, `_with_matching`
  - 最终 FK 概率: `final_prob = HSBM_block × propensity(feature) × Zipf(concentration)`
- `propensity_rho ∈ [0.05, 0.20], propensity_beta ∈ [2.0, 8.0]`（保持）
- `matching_latent_dim ∈ [2, 3, 4], matching_temperature ∈ [0.10, 0.50]`（保持）
- 1024 RDBs, 1 seed

**模型/训练差异**:
- **FKAttentionBias 重构** (`model_pretrain/src/models.py`):
  - **旧**: per-column `raw_lambdas` (vector, 30D) — `bias = Σ_j λ_j × match_j`
  - **新**: single scalar `raw_lambda` — `bias = λ_fk × Σ_j match_j`（多列匹配可叠加）
  - 删除 `_lambdas_for()` 方法
  - 动机: 单列 FK match 覆盖率 9.8%（中位数 0%），任一列匹配覆盖 39.6%。单标量将 30 列信号聚合到一个参数上，信号密度 ×4。

**实验过程（按时间顺序）**:
1. `v6.3_norelmode` — 关闭 relmode，无 Zipf → FK pair ratio 0.015%（和 v6.2 一样低）
2. `v6.3_zipf` — 加 row-level Zipf → 无效果（entity snapshots 稀释了 row-level 集中）
3. `v6.3_zipf_ent` — entity-level Zipf → FK pair ratio ~0.70%（×47 提升），但 propensity 绕过 `_sample_fk_per_parent` → 未生效
4. `v6.3_zipf_v2` — 修复 propensity + matching 路径中的 Zipf 注入 → FK pair ratio ~1.2%
5. `v6.3_zipf_v3` — 微调参数，保持全配置 → FK pair ratio mean=1.26%, median=0.20%
6. `v6.3` (最终) — 基于 zipf_v3 的 1024 RDB 全量生成 + 单标量 FK λ 训练

**关键发现**:
- FK 覆盖率 100% 但 entity 覆盖率暴跌到 30%（leaf entity 需要 ≥3 层 DAG 才有 non-root entity 表，58% DAG 只有 2 层）
- FK pair ratio 从 0.006% 提升到 1.26%（×210 倍），median 0.20%
- FK dim = 30（leaf 表更深，FK 列更多）
- FK bias **开始学习**: λ_fk (L5) = 0.50 (@16K) → 0.82 (@144K) → 0.87 (@176K)
- Entity bias **爆炸**: λ_se (L5) = 0.49 (@16K) → 2.80 (@144K) — 比 v6.2 的 1.57 高出 78%
- 浅层 bias 全部坍缩: layers 0-4 的 λ_fk 和 λ_se 都衰减到 0.001-0.02
  - "最后一层接管一切"模式在两个版本都存在，但 v6.3 更极端

**结果**: full128 peak **0.6892** (v6.3/model_eval00144.pt, step 144000) vs v6.2 的 0.7016（-0.0124）

**根因分析**:
- 性能间隙从 step 16K（FK bias 刚开始学）就已存在: v6.3 0.6579 vs v6.2 0.6696（-0.012）
- 两者从各自 baseline 的提升幅度相同（~+0.031）→ 天花板由数据决定，非 bias 学习动力学决定
- **核心原因: entity coverage 差距**。v6.2 entity 覆盖率 100%，v6.3 仅 30%。Entity bias 是确定性正确（同 entity = 同 label for DIRECT_ATTR），70% 任务缺失 entity 信号导致模型被迫过度依赖内容特征 + 弱 FK 信号，无法弥补

---

## v6.4（混合 v6.2 + v6.3 H5，2026-06-21）

**数据差异**:
- 合并 `v6.2.h5` (1219 samples) + `v6.3.h5` (1155 samples) → `v6.4.h5` (2374 samples)
- v6.2 fk_values: 从 (1219, 600, 5) pad 到 (1219, 600, 30)（-1 填充）
- v6.2 parent_entity_ids: 用 padded fk_values 作为 fallback（v6.2 生成时未计算 parent_entity_ids）
- `num_fk_columns=30`

**混合后覆盖率**:
| | v6.2 部分 (1219) | v6.3 部分 (1155) | 合并 (2374) |
|---|---|---|---|
| Entity 覆盖 | 100% | 30% | **66%** |
| FK 覆盖 | 31% | 100% | **65%** |

**模型/训练差异**:
- 同 v6.3（单标量 FK λ + entity λ）
- 使用 `JointPriorLoader` / `JointDataset` 的加权随机采样（每 batch 从一个 H5 来源采样）
- FK dim 动态适配（每 batch 可以是 5 或 30，取决于来源——但合并后已统一为 30）
- `max_num_classes` 取 max（两个来源都是 1）

**预期**:
- Entity bias 梯度密度从 30% → 66%（翻倍），应学到更合理的 λ_se
- FK bias 保持 65% 梯度密度，两种任务都在训练中贡献梯度
- 同时存在 root entity DIRECT_ATTR (v6.2 部分) 和 leaf entity (v6.3 部分) 任务

**状态**: 已合并 H5，等待训练验证

---

## 代码改动清单（按文件）

### `model_pretrain/src/eval_utils.py`
- `load_task_split` 新增 PK → entity_id fallback（v6.2.1）：无 `entity_id` 列时，取 `primary_key` 列作为 entity_id

### `model_pretrain/conf_eval/model/RDBPFN.yaml`
- `use_fk_bias/use_entity_bias: false → true`（v6.2.1）

### `model_pretrain/src/models.py`
- 新增 `FKAttentionBias` 类 — 动态可扩展 per-column FK bias（v6.2 引入，v6.3 改为单标量）
- 新增 `EntityAttentionBias` 类 — 单标量同 entity attention bias（v6.1 引入）
- `TransformerEncoderLayer.__init__` 新增 `use_fk_bias`, `use_entity_bias` 参数
- `TransformerEncoderLayer.forward` 新增 bias 计算 + 注入 attention mask 逻辑

### `data_preprocessing/merge_dbinfer_to_h5.py`
- 新增 `parent_entity_ids` 计算: FK value → parent entity PK mapping（v6.2）
- 新增 stratified entity sampling（v6.1 后，提升 same-entity pair ratio）

### `model_pretrain/src/dataloaders.py`
- `PriorDumpDataLoader` 新增 `has_fk_values`, `has_entity_ids` 检测
- 新增 `fk_values` 和 `entity_ids` 的 batch 加载
- `InMemoryDataset` 新增 `has_parent_entity_ids` + 加载逻辑

### `model_pretrain/src/training.py`
- `parent_entity_ids` 从 dataloader → model.forward 的完整链路（v6.2）
- Loss 日志 bug: `original_loss_sum` / `total_loss_sum` 初始化后从未累加（待修复）

### `data_generation/RDB/src/prior/hsbm.py`
- 新增 `_compute_zipf_weights()` — entity-level Zipf power-law FK 权重（v6.3）
- 4 条 FK 采样路径全部注入 Zipf: `_sample_fk_per_parent`, `compute_hsbm_fk_ids`, `_with_propensity`, `_with_matching`
- `parent_entity_ids` 参数贯穿全部 FK 采样函数

### `data_generation/RDB/src/prior/prior_config.py`
- `DEFAULT_HSBM_HP` 新增 `hsbm_zipf_alpha: Uniform(1.1, 2.0)`（v6.3）
- `TEMPORAL_FOURIER_HP` calendar-aligned Fourier seasonality + EventCalendar 参数（各版本通用）

### `data_generation/RDB/src/table_def/table_generation.py`
- `TableGenerator` 新增 `hsbm_zipf_alpha` 属性
- `_compute_hsbm_fk_ids` 新增 `parent_entity_ids_list` 参数，传递给 HSBM 函数
- `generate_one_table_data_from_SCM` 提取 parent entity_ids 并传递给 FK 采样
- `init_table_SCMs` 采样 `hsbm_zipf_alpha`

### `model_pretrain/conf_train/RDBPFN_hsbm.yaml`
- 新增 `use_fk_bias: true`, `use_entity_bias: true`（v6.1/v6.2）

### `scripts/run_pipeline.sh`
- 新增参数: `relbench_mode`, `use_fk_bias`, `use_entity_bias`, `use_homophily_labels`, `no_path_signal`, `snapshots_per_entity_min/max`, `entity_timestamp_prob`, `load_ckpt`, `extra_train_args`
- `EXTRA_TRAIN_ARGS` 支持空字符串占位符 `""` 跳过

---

## 已知问题

1. **Loss 日志**: `training.py` 中 loss 累加变量未更新，log 显示 `loss=0.0000`（不影响实际训练和 backprop，仅日志遗漏）
2. **parent_entity_ids for v6.2**: 合并时 v6.2 部分用 padded fk_values 作为 fallback（非真正的 entity-level key），但 v6.2 的 FK coverage 仅 31% 且 FK bias 在 v6.2 数据上本就不活跃，影响有限
3. **浅层 bias 坍缩**: 所有版本呈现 layers 0-4 bias → 0, layer 5 独占的模式，可能是深层语义更适合 structure bias，也可能有优化问题
