# RDBPFN 数据生成流水线 (Data Generation Pipeline)

## 目录

1. [概述](#1-概述)
2. [整体架构](#2-整体架构)
3. [Stage 1: DAG 到 RDB 结构生成](#3-stage-1-dag-到-rdb-结构生成)
4. [Stage 2: SCM 超参数采样与初始化](#4-stage-2-scm-超参数采样与初始化)
5. [Stage 3: 按拓扑序生成表数据](#5-stage-3-按拓扑序生成表数据)
6. [Stage 4: Task 生成](#6-stage-4-task-生成)
7. [Stage 5: 预处理与 H5 合并](#7-stage-5-预处理与-h5-合并)
8. [核心机制详解: Signal-Group Feature Generation](#8-核心机制详解-signal-group-feature-generation)
9. [核心机制详解: HSBM FK 生成](#9-核心机制详解-hsbm-fk-生成)
10. [核心机制详解: 时间戳生成](#10-核心机制详解-时间戳生成)
11. [超参数配置参考](#11-超参数配置参考)
12. [关键文件索引](#12-关键文件索引)

---

## 1. 概述

整个流水线的目标是**合成关系型数据库 (RDB)**，用于预训练一个关系型表格基础模型 (RDB-PFN)。每个合成 RDB 包含多张通过外键关联的表，每张表的特征列由 Structural Causal Model (SCM) 生成，外键连接由 Hierarchical Stochastic Block Model (HSBM) 控制。

流水线全景：

```
DAG structures (rdb_v1.pth)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Stage 1: DAG → Table/Relationship 结构定义  │
│  Stage 2: 超参数采样 → SCM 初始化            │
│  Stage 3: 按拓扑序逐表生成列数据 + FK 连接   │
│  Stage 4: 生成预测 Task                      │
└─────────────────────────────────────────────┘
    │  输出: Parquet + metadata.yaml (每个 RDB 一个目录)
    ▼
┌─────────────────────────────────────────────┐
│  Stage 5a: Pre-DFS Transform                │
│  Stage 5b: DFS (Deep Feature Synthesis)     │
│  Stage 5c: Post-DFS Transform               │
│  Stage 5d: Merge to H5                      │
└─────────────────────────────────────────────┘
    │  输出: .h5 文件 (N_tasks × 600 × 60)
    ▼
   Model Training
```

**入口脚本**: `data_generation/RDB/dag_to_rdb_generator.py`

**一键运行**: `bash scripts/run_pipeline.sh <num_rdbs> <start_index> <run_name>`

---

## 2. 整体架构

### 2.1 核心类与职责

| 类 | 文件 | 职责 |
|---|---|---|
| `DAGToRDBGenerator` | `dag_to_rdb_generator.py` | 总控: 加载 DAG 数据, 解析结构, 创建 RDB, 并行生成 |
| `RDB` | `src/table_def/table_generation.py` | 关系型数据库容器: 管理 Table/Relationship/TableGenerator |
| `Table` | `src/table_def/table_generation.py` | 单表定义: schema, 列名, 数据类型, DataFrame 生成 |
| `TableGenerator` | `src/table_def/table_generation.py` | 单表数据生成器: 持有 MLPSCM 实例, 协调 FK 生成与列生成 |
| `MLPSCM` | `src/prior/mlp_scm.py` | 核心 SCM: MLP 生成列值 + Signal-Group Feature Generation |
| `Relationship` | `src/table_def/table_generation.py` | 表间关系: PK→FK 或 FK→PK |
| `HpSamplerList` | `src/prior/hp_sampling.py` | 超参数采样器: 从配置分布中采样 |
| `TemporalVocab` | `src/prior/temporal_vocab.py` | 时间戳生成: 日历对齐的 Fourier 季节性 + 事件日历 |
| `EventCalendar` | `src/prior/temporal_vocab.py` | RDB 级共享事件日历: 可被多表的不同敏感度读取 |
| `TaskGenerator` | `src/table_def/task_generation.py` | Task 生成: 单表预测 / 复杂预测 (DFS 聚合目标) |
| `ColumnDataProcessor` | `src/table_def/table_generation.py` | 后处理: 异常值去除, 连续值→分类转换, 时间戳格式化 |

### 2.2 DAG 数据结构

DAG 数据来自 `datasets/rdb_v1.pth`，是一个 PyTorch 保存的字典:

```python
{
    "src_list": [...],   # 每条边的源节点列表
    "dst_list": [...],   # 每条边的目标节点列表
    "x_n_list": [...],   # 每个节点的 [num_rows, num_cols]
    "y_list": [...],     # 标签
}
```

每个 DAG 被解析为:
- **节点** → 一张表 (Table)
- **边** `(src→dst)` → 外键关系 (dst 表的 FK 引用 src 表的 PK)
- **节点的 `x_n_list`** → 表的原始行列数, 经 `dimension_config` 映射为实际行列数

### 2.3 表结构约定

每张表的结构 (column layout):

```
列 0:                        primary key   (table_X_id)
列 1 ~ num_parents:          foreign keys  (table_P1_id, table_P2_id, ...)
列 1+num_parents (可选):     timestamp     (timestamp)
剩余列:                      features      (feature_0, feature_1, ..., feature_{K-1})
```

- `num_cols = 1 (PK) + num_parents (FKs) + (1 if timestamp else 0) + num_features`
- `num_features` = 来自 DAG 的原始列数 (通常 8~12)
- 特征列交替为 float / categorical

---

## 3. Stage 1: DAG 到 RDB 结构生成

### 3.1 解析 DAG 结构

`DAGToRDBGenerator.parse_dag_structure(dag_idx)`:

1. 读取 `src_list[i]`, `dst_list[i]`, `x_n_list[i]`
2. 构建邻接关系: `in_degree` (每节点有几个父节点), `out_degree` (每节点有几个子节点)
3. 将原始行列数映射为实际行列数:
   - `num_rows`: 分段线性映射, 区间 [1000, 5000], ±20% 波动
   - `num_cols`: min/max threshold (8→12, >8→12), ±100% 取整波动

### 3.2 创建表配置

`DAGToRDBGenerator.create_all_table_configs(dag_structure)`:

对每个节点:
- 确定 `num_rows` (含 ±20% 波动)
- 确定 `num_features` = 原始 num_cols
- 确定 `num_cols` = feature 数 + 1 (PK) + num_parents (FKs) + (可选 timestamp)
- Timestamp 分配策略 (v2):
  - Source 表 (`num_parents == 0`): 永不 timestamp
  - Entity 表 (`out_degree >= 1`): `entity_prob` (当前 0.0, 永不 timestamp)
  - Activity/Leaf 表 (`out_degree == 0`): `activity_prob` (当前 1.0, 必定 timestamp)

### 3.3 创建 RDB 对象

`DAGToRDBGenerator.create_rdb_from_config(table_configs, relationships, rdb_name)`:

1. 创建 `RDB(rdb_name)` 实例
2. 为每个配置创建 `Table`:
   - 自动生成列名: `[table_X_id, table_P1_id, ..., timestamp?, feature_0, feature_1, ...]`
   - 自动生成 `DataTypeConfig`: PK → FK(s) → timestamp? → float/categorical 交替的 feature
3. 添加 `Relationship` (子表 FK → 父表 PK)
4. 记录 `timestamp_config` 用于 SCM 初始化

### 3.4 并行生成

`generate_rdbs_from_dags()`:
- 随机选一个 DAG 结构
- 每个 worker 独立: `seed = rdb_index`
- 支持 multiprocessing `Pool.imap_unordered` 并行

---

## 4. Stage 2: SCM 超参数采样与初始化

### 4.1 采样流程

`RDB.init_table_SCMs(seed)` 按**拓扑序**为每张表初始化 SCM:

```
for table_name in topological_sort(graph):
    1. HpSamplerList(DEFAULT_SAMPLED_HP)   → 采样 SCM 超参数
    2. HpSamplerList(DEFAULT_HSBM_HP)      → 采样 HSBM 超参数 (每个 parent 独立)
    3. 组装 combined_params                → 传给 MLPSCM.__init__()
    4. TableGenerator.init_table_SCM()     → 创建 MLPSCM 实例
```

### 4.2 关键超参数类别

**MLPSCM 结构参数** (每个表独立采样):
- `num_causes`: [3, 12] (log-normal)
- `num_outputs`: [3, 12] (log-normal)
- `num_layers`: [3, 12] (log-normal)
- `hidden_dim`: [6, 48] (log-normal)
- `mlp_dropout_prob`: [0.01, 0.05]
- `init_std`, `noise_std`: 采样的初始化和噪声标准差

**HSBM FK 参数** (每个父表独立采样):
- `hsbm_num_levels`: {1, 2, 3, 4, 5} (均匀选择)
- `hsbm_clusters_per_level`: {1, 2, 3} (均匀选择)

**Signal-Group Feature Generation 参数** (详见第 8 节):
- `group_scale_{time,parent,path,intrinsic}`: log-normal, max_mean=15, min_mean=3
- `intrinsic_dim`: 8 (固定, 计划可配置化)
- `loading_log_mean`: [0.35, 1.7]
- `loading_sigma`: [0.2, 0.6]
- `max_groups_per_feature`: {2, 3}
- `archetype_perturb_std`: [0.05, 0.15]
- `coupling_lambda`: [0.02, 0.06]
- `coupling_rank`: {2, 3}
- `residual_sigma`: [0.1, 0.4]
- `basis_perturb_eta`: {0.10} (固定)

**Temporal 参数**:
- `trend_active`, `seasonal_active`, `spike_active`: 各布尔采样
- `gamma_tier`: {0.0, 0.5, 1.5, 3.0}
- `p_sort`: [0.3, 0.8]

---

## 5. Stage 3: 按拓扑序生成表数据

### 5.1 总体流程

`RDB.generate_all_data_from_SCM()`:

```
for table_name in topological_sort(graph):
    parent_tables = [FK 引用的父表名]
    generate_one_table_data_from_SCM(table_name, parent_tables)

    if 没有父表 (源表):
        MLPSCM.forward_without_input()
            ├── XSampler 采样 root causes (seq_len, num_causes)
            ├── MLP layers 前向传播 → CAUSAL_OUTPUT, Y
            ├── [可选] _prepare_time_features() (time_dim=0 → 跳过)
            ├── _build_intrinsic_signal(): CAUSAL_OUTPUT → Linear(num_outputs, 8) → (seq_len, 8)
            └── [Signal-Group]: _construct_features() 覆盖 X (α=[0,0,0,1], 100% intrinsic)

    else (有父表):
        _compute_hsbm_fk_ids(): HSBM 采样 FK 连接
        MLPSCM.forward_with_input(parent_data, fk_ids)
            ├── XSampler 采样 root causes
            ├── [可选] _prepare_time_features() + t_min (仅 timestamp child)
            ├── 拼接 parent_data[CAUSAL_OUTPUT][fk_ids] 作为额外输入
            ├── MLP layers 前向传播 → CAUSAL_OUTPUT, Y
            └── [Signal-Group]: _construct_features() 覆盖 X

    cache_pending_outputs(X, fk_ids) → row_embeddings

[可选] Row GNN 精炼行嵌入
_materialize_tables_from_pending() → Table.process_data()
```

### 5.2 源表 vs 子表的区别

| | 源表 (no parents) | Timestamp child (leaf 表) | Non-ts child (entity 表) |
|---|---|---|---|
| MLP 输入 | `causes_raw` | `causes_raw + time_features + parent_causal_outputs` | `causes_raw + parent_causal_outputs` |
| FK 生成 | 无 | HSBM 采样 | HSBM 采样 |
| time_dim | 0 | >0 | 0 |
| Signal-Group α (4维) | `[0, 0, 0, 1]` (仅 intrinsic) | `[~0.5, ~0.25, ~0.25, 0]` (time+parent+path) | `[0, ~0.6, ~0.4, 0]` (parent+path only) |

**关键架构**: MLP 负责生成 Y (预测目标) 和 CAUSAL_OUTPUT (传给子表的隐层表示)；Signal-Group 负责构造 X (特征列)。Source 表通过 `CAUSAL_OUTPUT → intrinsic_projector → intrinsic_sig` 使 X 和 Y 共享 MLP 中间表示。Non-ts child 通过 parent CAUSAL_OUTPUT → parent_sig 间接获得 MLP 输出，形成级联耦合效应。

### 5.3 数据后处理: ColumnDataProcessor

`Table.process_data()`:

1. 逐列处理 SCM 输出的原始连续值
2. **异常值去除**: 两阶段 MAD/z-score 裁剪 (threshold=4σ for float, 3σ for categorical)
3. **类型转换**:
   - FLOAT: 可选 normalization
   - CATEGORICAL: quantile/rank/value-based binning → 整数标签
   - TIMESTAMP: day index → days-since-epoch (datetime64)
   - PK: 0, 1, 2, ..., n-1
   - FK: HSBM 生成的 parent row indices

### 5.4 输出格式

每个 RDB 保存为一个目录:
```
dag_rdb_0/
├── metadata.yaml              # 4DBInfer 格式: table schemas + task metas
├── generation_schemas.yaml    # 完整生成记录: SCM 参数、HSBM 配置等
├── _feature_diagnostics.json  # 每表特征分配诊断 (K, α, group assignments)
├── table_0.parquet
├── table_1.parquet
└── tasks/
    ├── task_0/
    │   ├── train.parquet
    │   ├── val.parquet
    │   └── test.parquet
    └── ...
```

---

## 6. Stage 4: Task 生成

### 6.1 Simple Tasks (单表预测)

`RDB.initialize_tasks()` → `TaskGenerator.generate_tasks_for_rdb_per_table()`:
- 每表生成一个预测任务: 随机选一列 feature 作为 target
- `train_ratio=0.75`, `valid_ratio=0.05`
- 不涉及跨表聚合

### 6.2 Complex Tasks (复杂预测)

`RDB.initialize_tasks_with_complex_tasks()` → `TaskGenerator.generate_tasks_for_rdb_with_complex_tasks()`:

**Task 类型:**
- `DIRECT_ATTRIBUTE_PREDICTION`: 直接预测目标表的一个 feature 列 (target 在 schema graph 中的 `row_num == "1"`)
- `RELATIONAL_AGGREGATION_PREDICTION`: 预测目标表的聚合值, 如 SUM/AVG/COUNT/MAX/MIN (target 的 `row_num != "1"`)

**Focal table 选择:**
- `entity_task_ratio` 控制选 entity 表 (被 FK 引用的表) 作为 focal 的概率
- 默认 `entity_task_ratio=0.75`: 75% 选 entity, 25% 选 non-entity
- 选 entity 时, BFS 第一步 80% 偏向选 child; 选 non-entity 时均匀选所有邻居

**Target table 选择:**
- `root_p` 参数控制选 root (schema graph 拓扑序第一节点) 作为 target 的概率
- 默认 `root_p=0.0`: 总是选 leaf node → 产生 `RELATIONAL_AGGREGATION_PREDICTION`
- Focal=entity 且 target=leaf (子表) 时: edge 方向 `FK_TO_PK` → leaf 的 `row_num="1"` → 仍为 `DIRECT_ATTRIBUTE_PREDICTION`

**RelBench Mode** (`relbench_mode=True`):
- `entity_task_ratio` 强制为 1.0: focal 永远是 entity 表
- `root_p` 强制为 1.0: target 永远是 root (= entity 表)
- 生成 RelBench 风格的 **entity-level prediction** 任务: 预测 entity 自身的一个 categorical feature 列
- Target column 限制为 `only_categorical=True` (质量门的 `np.bincount` 只接受整数 label)

### 6.3 Task 类型约束 (Schema Graph 节点数)

Schema graph 由 focal 表出发 BFS (2 hops, 每跳 1 个邻居) 构建:

| Task 类型 | 最少节点数 | 原因 |
|---|---|---|
| `DIRECT_ATTRIBUTE_PREDICTION` | ≥2 | 只需 focal→neighbor 的单跳关系即可生成特征 |
| `RELATIONAL_AGGREGATION_PREDICTION` | ≥3 | 需要两跳关系进行聚合计算 |

- `<2 节点`: 始终跳过 (孤立表, 无 FK)
- 2 节点 + `DIRECT_ATTRIBUTE_PREDICTION`: 接受 (focal=entity, BFS 选 child; 或 focal=non-entity, BFS 选 parent 且 target=root)
- 2 节点 + `RELATIONAL_AGGREGATION_PREDICTION`: 跳过 (需要 ≥3 节点)
- ≥3 节点: 两种 task 类型都接受

在 `relbench_mode` 下, focal 固定为 entity (有子表), BFS 第一步 80% 选 child → 大部分 schema graph 为 2 节点 Entity→Child (`DIRECT_ATTRIBUTE_PREDICTION`)。

### 6.4 Quality Gate

`DAGToRDBGenerator._generate_tasks_with_quality_gate()`:
- **Stage A** (规则): 标签单一值、极端不平衡 (`pos_ratio < 0.05` 或 `> 0.95`)、样本过少 (`n < 64` 或 min_class < 8)、无有效特征、结构独立性问题、父子表大小比极端
- **Stage B** (模型): ExtraTrees OOF AUC 检查 (AUC≤0.52 拒绝, AUC≥0.58 接受, 中间灰区做 bootstrap CI, CI 上界需 ≥0.55)
- 最多重试 `quality_max_retries` 次, 每次使用不同 seed (`base_seed + attempt × 1000`), 保留通过最多的那次
- `TaskGenerator` 和 `TaskDataGenerator` 的 `random_seed` 参数设为 `None` 时不覆盖全局 random state, 确保 retry seed 差异生效
- **注意**: Stage B 使用 `np.bincount(y)` 计算类别分布, 只支持整数 label。若 target 列是 float, `_etoof()` 会 crash → 任务被丢弃。因此 entity target 限制为 `only_categorical=True`

---

## 7. Stage 5: 预处理与 H5 合并

### 7.1 三步预处理

`run_pipeline.sh` 中每个 RDB 独立执行:

```
Step 5a: Pre-DFS Transform
  输入: 原始 4DBInfer 格式 (Parquet + metadata.yaml)
  处理: 标准化列名、格式转换、时间列处理
  输出: 临时目录 tmp_pre/

Step 5b: DFS (Deep Feature Synthesis)
  配置: dfs-2-ft.yaml → max_depth=2, engine=featuretools
  处理: 从 __task__ 表出发, 沿 FK 关系构造特征。depth=0 为自身列,
        depth=1 为一步关系可达的表, depth=2 为两步关系可达的表。
        - forward (子→父, many-to-one): depth=1 直接拿到父表列值 (IdentityFeature)
        - backward (父→子, one-to-many): depth=1 即可产生聚合特征 (MEAN/SUM/COUNT 等)
        - depth=2 产生穿透两步的递归聚合特征 (e.g. MEAN(child.MEAN(grandchild.col)))
  聚合函数: MEAN, STD, COUNT, MAX, MIN, MODE, SUM
  filter_features 仅过滤根 IdentityFeature 是 PK/FK key column 的特征
  (如 COUNT(child) 基于 child.PK), 非 key 的 IdentityFeature 和 AggregationFeature 均保留
  → 实际有效特征从 depth≥1 开始 (backward 聚合) 和 depth≥2 (递归聚合)
  输出: 临时目录 tmp_post/

Step 5c: Post-DFS Transform
  处理: 列筛选、规范化、标签提取
  输出: PROCESSED_DIR/dag_rdb_X-dfs-2/
```

### 7.2 H5 合并

`merge_dbinfer_to_h5.py`:
- 读入所有处理后的 RDB 的 task train split
- 每个 task **截断或填充**到 `total_rows=600` 行, `max_columns=60` 列
- 输出 shape: `(N_tasks, 600, 60)`
- 同时保存 `num_features` (每 task 的实际有效特征数) 和 `labels`
- DS 存储为 HDF5 datasets: `X`, `y`, `num_features`, `task_ids`

### 7.3 训练时的子采样

训练时从每个 task 的 60 列中随机子采样 30 列做为一个 "batch" 的输入维度。这是模拟真实场景中每次查询只涉及一部分列的情况。

---

## 8. 核心机制详解: Signal-Group Feature Generation

这是整个系统最核心的设计。当 `use_signal_group_features=True` 时, SCM 不再直接使用 MLP 中间层输出作为特征, 而是通过四组信号源 (time, parent, path, intrinsic) 的基向量投影来构造特征。

### 8.1 四组信号源

| 信号组 | 维度 | 含义 | 来源 |
|---|---|---|---|
| **time** | 11 | 时间信号 | basis(8) + gates(3), 来自 TemporalVocab |
| **parent** | 12 | 父表特征信号 | 每个 FK 边: parent CAUSAL_OUTPUT → Linear → mean-pool |
| **path** | 8 | HSBM 块路径信号 | 每个 FK 边: Block indices → Embedding → PathEncoder → mean-pool |
| **intrinsic** | 8 | 内在信号 | `CAUSAL_OUTPUT → Linear(num_outputs, 8)` — 本表 MLP 中间输出经 Linear 投影 |

信号产生流程:
```
time_sig:
  _prepare_time_features() → timestamps
    → evaluate_basis(t): [trend_level, trend_slope, 4 Fourier harmonics, dow_factor] (8维)
    + build_gate_vector(): [1.0, 1.0, events_active?1:0] (3维)
    → cat → (seq_len, 11)

parent_sig:
  parent_data[CAUSAL_OUTPUT][fk_ids[:, i]] → Linear(CAUSAL_DIM → 12) → mean over edges
    → (seq_len, 12)

path_sig:
  block_paths[:, :n_levels] → Embedding per level → concat → Linear → mean over edges
    → (seq_len, 8)

intrinsic_sig:
  CAUSAL_OUTPUT (MLP 中间层输出) → Linear(num_outputs → 8) → (seq_len, 8)
```

### 8.2 原型 (Archetype) 驱动的 α 分配

`MLPSCM._compute_base_alpha(archetype)`:

每张表被分类为一个原型 (archetype), 决定四个信号组的相对权重 α:

| 原型 | 条件 | α = [time, parent, path, intrinsic] |
|---|---|---|
| Source table | `num_parents == 0` | `[0, 0, 0, 1.0]` |
| Timestamp child | `is_timestamp=True` | `[~0.5, ~0.25, ~0.25, 0]` |
| Dependent non-ts | 其他子表 (无时间戳, 有父表) | `[0, ~0.6, ~0.4, 0]` (仅 parent+path, 无 intrinsic) |

多父表情况下, parent 份额上调 (最多 0.6)。不可用组 (如 source 表的 parent/path, non-ts 表的 time) 被 mask 为零后重新归一化。

然后 `_perturb_alpha()` 施加乘性对数正态扰动 (`archetype_perturb_std ∈ [0.05, 0.15]`), 并重新归一化。

### 8.2.1 X-Y 耦合分析

MLP 和 Signal-Group 是两个解耦的机制:
- **MLP 路径**: `causes → MLP → Y, CAUSAL_OUTPUT` — 生成预测目标和子表隐层表示
- **Signal-Group 路径**: `signals → basis projection → X` — 构造特征列

**各 archetype 的 X-Y 耦合程度**:

| Archetype | X 信号来源 | MLP 输入共享 | 耦合强度 |
|-----------|-----------|-------------|---------|
| Source | intrinsic = Linear(CAUSAL_OUTPUT) | CAUSAL_OUTPUT (MLP 中间层) | 中强 |
| Timestamp child | time + parent + path | time_features + parent_causes | 强 |
| Non-ts child | parent + path | parent_causes | 中 |

Source 表的 intrinsic signal 和 Y 都来自同一 MLP 中间层 (CAUSAL_OUTPUT)，共享了 MLP 的非线性变换。Non-ts child 表通过 parent_sig (parent CAUSAL_OUTPUT → projector) 间接获得 MLP 输出，形成级联耦合。

### 8.3 特征分配

`_sample_feature_groups(α, n_features)`:

对每个特征 j:
1. 从 Dirichlet(α) 采样 group weights
2. 选 top-K 活跃组 (K = `max_groups_per_feature` ∈ {2, 3})
3. Round-robin 分配每个组的 basis index (保证特征间均匀共享基)
4. 分配共享 sign (同一 `(group, basis_idx)` 的所有特征共用相同符号)
5. 采样 magnitude ∼ LogNormal(`loading_log_mean`, `loading_sigma`)

**K 计算 (每个组的基向量数量)**:
- `K_time = min(3, max(1, ceil(8 / divisor)))` — 当前 divisor=3 → K=3
- `K_parent = max(1, ceil(12 / 3))` = 4
- `K_path = max(1, ceil(8 / 2))` = 4

### 8.4 子空间正交基

`_init_group_bases()`:

每个组的信号空间被等分为 K 个子空间。基 k 只在第 k 个子空间上有非零值（其他维度为零）。这保证了**纯信号投影下, 不同基向量间的相关性为零** (信号级零相关)。

```
time  (8维):  subspace_0=[0:2], subspace_1=[2:5], subspace_2=[5:8]
parent(12维): subspace_0=[0:3], subspace_1=[3:6], subspace_2=[6:9], subspace_3=[9:12]
path  (8维):  subspace_0=[0:2], subspace_1=[2:4], subspace_2=[4:6], subspace_3=[6:8]
```

### 8.5 特征构造

`_construct_features(signals, residual_sigma)`:

对每个特征 j 的每个分配的组:
```
X_j += sign × magnitude × (signal @ basis_k)    # 信号投影
```

然后特征间耦合:
```
X_j += ε_j           # ε_j ~ N(0, σ²_res)
X_out = X + λ · X @ W    # W = (1 - I) / (n - 1), uniform off-diagonal coupling
```

**Basis perturbation** (`basis_perturb_eta=0.10`): 每个 (feature, group) 对被施加一个固定的随机扰动向量 (η-scaled, unit-norm), 加到基向量上。这打破了 rank-1 的块内完全相关结构, 引入了一定的块内特征差异。

**信号归一化** (`_normalize_signals()`):
- time_basis: RMS-norm × `group_scale_time`
- time_gates: 不做 RMS (直接透传)
- parent: RMS-norm × `group_scale_parent`
- path: RMS-norm × `group_scale_path`
- intrinsic: RMS-norm × `group_scale_intrinsic`

### 8.6 完整特征构造流程

```
输入: signals = {time_basis(8), time_gates(3), parent(12), path(8), intrinsic(8)}

对每个特征 j:
  ├── 从 Dirichlet(α) 采样 group weights  (α 为 4 维向量)
  ├── 选 top-K 活跃组
  ├── 对每个活跃组 g:
  │   ├── 取 round-robin 分配的 basis index k
  │   ├── 加载 subspace-orthogonal basis B_g[k]
  │   ├── 施加 perturbation: B_eff = normalize(B_g[k] + pert_vec × η)
  │   └── X_j += sign × magnitude × (signal_g @ B_eff)
  │
  ├── 加残差噪声: X_j += N(0, σ²_res)
  └── 跨特征耦合: X_out = X + λ × X @ W

输出: X_out ∈ R^{seq_len × n_features}
```

---

## 9. 核心机制详解: HSBM FK 生成

### 9.1 分层随机块模型

HSBM 将父表和子表的行分别分配到层次化的块结构中, FK 连接概率为各层块间概率的乘积。

**单父表** (`compute_hsbm_fk_ids`):
```
对 child row j:
  child 的 cluster path 来自确定性分配 (均分)
  对 parent row a:
    P(a → j) ∝ ∏_{l} Probs_l[cluster_a(a,l), cluster_b(j,l)]
  从 P 中采样一个 parent row

其中:
  Probs_l[diag] = 0.9      (块内高连接概率)
  Probs_l[off-diag] ~ Uniform(0.001, 0.002)  (跨块低连接概率)
```

**多父表** (`compute_hsbm_fk_ids_multi`):
- 所有父表共享同一个 child-side latent cluster path (随机采样, 非确定)
- 每个父表读取其层级数的前缀
- 父表间 FK 的 tuple 级相关性来自共享的 latent variable

### 9.2 Hierarchy 裁剪

`TableGenerator._clip_hierarchy()`:

如果叶子块数 `clusters_per_level ^ num_levels > min(parent_rows, child_rows)`, 则缩小 `num_levels`, 保证每个块至少有一条记录。

### 9.3 HSBM Locality

`compute_hsbm_locality(hierarchies)`:
- 计算每个 FK 边的层级结构 "集中度" (log-product of cluster counts, normalized)
- 被用作 archetype params 的 `hsbm_locality` 特征
- 影响 PathEncoder 输出信号的强度

### 9.4 FK Propensity (单父表)

`compute_hsbm_fk_ids_with_propensity()`:

在 HSBM 块约束内, FK 采样偏向于父表行中 feature-derived rank score 与 child-specific target 值接近的行。

```
对 parent:
  rank_score[a] = percentile_rank(random_linear_combo(CAUSAL_OUTPUT[a, 2-3 cols]))

对 child row j:
  t(j) = ρ·z(j) + √(1-ρ²)·ε(j)    # z(j)~Uniform, ε(j)~Uniform

在 HSBM 块内:
  P(a → j) ∝ exp(-β·|rank_score[a] - t(j)|)  ×  base_probs[a]
```

直觉: 同一个 child row 的各特征共享 parent 端的 rank score → 特征值相近的 parent 行倾向于被同一 child row 引用。

HPs: `propensity_rho` ∈ Uniform(0.05, 0.20), `propensity_beta` ∈ Uniform(2.0, 8.0)

### 9.5 Shared Matching Latent (多父表)

`compute_hsbm_fk_ids_multi_with_matching()`:

所有父表的 CAUSAL_OUTPUT 做 per-column z-score 归一化后, 通过**同一个随机投影矩阵 W** 映射到共享低维空间。每个 child row 采样一个 target latent, 在 HSBM 块内 FK 采样偏向于 matching latent 接近 target 的父表行。

```
对所有 parent p:
  X_norm[p] = zscore(parent_p.CAUSAL_OUTPUT)    # per-column
  D_common = min(D_1, ..., D_P)
  W ~ N(0, 1/D_common)                           # (D_common, D_latent), 所有 parent 共享
  matching_latent[p] = X_norm[p][:, :D_common] @ W

对 child row j:
  target[j] ~ N(0, I)                             # D_latent 维标准正态

在 HSBM 块内:
  P(a → j) ∝ softmax(-||matching_latent[p][a] - target[j]||² / temperature)
```

直觉: 共享 W 使所有 parent 的特征映射到可比较的 latent 空间。如果 parent_A 的某行和 parent_B 的某行在此空间中接近同一个 target, 它们倾向于被同一个 child row 配对引用。

与 FK Propensity 的区别:

| | Propensity (单父表) | Matching Latent (多父表) |
|---|---|---|
| 投影 | 1D random combo per parent | 共享 W: 所有 parent → 相同 D_latent 空间 |
| 可比性 | Rank per-parent, 跨 parent 不可比 | Latent 在同一空间, 跨 parent 可直接比较 |
| Target | 1D scalar via ρ-mixing | D_latent 维标准正态 |
| Sampling | exp(-β·|rank - t|) | softmax(-||latent - target||² / temp) |

HPs: `matching_latent_dim` ∈ {2, 3, 4}, `matching_temperature` ∈ {0.10, 0.20, 0.35, 0.50}

---

## 10. 核心机制详解: 时间戳生成

### 10.0 Timestamp 分配策略 (v2)

当前策略 (config: `entity_prob=0.0, activity_prob=1.0`):

| 表类型 | 条件 | timestamp 概率 | 说明 |
|--------|------|---------------|------|
| Source | `num_parents == 0` | 0% | 源表无时间戳，靠 intrinsic signal |
| Entity | `out_degree >= 1` | `entity_prob` (0%) | 非 leaf 表均无时间戳 |
| Activity/Leaf | `out_degree == 0` | `activity_prob` (100%) | 所有 leaf 表必定有时间戳 |

注: `num_parents == 0` 且 `out_degree == 0` 的表（孤立节点，极少见）按 source 分支处理，但也可能被 activity 分支捕获。

### 10.1 TemporalVocab: 日历对齐的强度函数

`TemporalVocab.generate(time_range)`:

```
raw(t) = trend(t_norm)      # m_lin × t_norm + c_lin
       + seasonality(t)      # 3-period Fourier series (week/month/year)
       + events(t)           # RDB 级 EventCalendar × 表敏感度
       + noise(t)            # N(0, noise_std²)

base_intensity(t) = softplus(raw(t))
intensity(t) = base_intensity(t) × p_dow[dow(t)] × 7
```

### 10.2 Fourier Seasonality

三个周期各含多个谐波:
- Week (7天): 3 harmonics, 系数 ∼ N(0, 1/f)
- Month (30.4375天): 6 harmonics, 系数 ∼ N(0, 1/f)
- Year (365.25天): 6 harmonics, 系数 ∼ N(0, 1/f)

系数被 rescale 到 unit norm。各频率的调制权重 `m_week`, `m_month`, `m_year` 采样自不同分布。

### 10.3 DOW 模式

`_sample_dow_pvec()`: 从 11 种 Dirichlet 模板的混合分布中采样一周七天概率向量:
- 28% 接近均匀
- 20% 周中高周末低
- 13% 周末高周中低
- 其他: 不同峰值日 (周一峰/周三峰/周六峰/周日峰 等)
- 每个模板施加乘性浓度噪声 (LogNormal(0, 0.7))

### 10.4 Event Calendar

`EventCalendar`: RDB 级共享, H ∈ [3, 8] 个随机事件:
- 每个事件: 随机 base_doy (U(0,365)), sigma (U(1,3)), importance (clipped LogNormal)
- 每个表独立采样 `table_sens` (H 维, Beta(1,4) × Bernoulli(0.4))
- 事件贡献 = Gaussian kernel at t_center (含年度 jitter)

### 10.5 子表时间约束

`_compute_t_min_for_child()`:
- **t_min** = parent 时间戳中最新者 (保证子表时间 ≥ 父表)
- 无时间戳父表: 按 DAG 位置 (source/intermediate/leaf) 设置 fallback t_min
- `gamma_tier` 控制生命周期衰减强度 (距 t_min 越远的行越不可能)

### 10.6 分箱排序

`_apply_intra_group_sort()`:
- 以概率 `p_sort` 对同一个 FK group 内的子表行按时间戳排序
- 引入时间有序性, 模仿真实数据中的批次效应

---

## 11. 超参数配置参考

### 11.1 当前最佳参数 (raw-table corr ≈ 0.089)

| 参数 | 值 | 说明 |
|---|---|---|
| `group_scale_{time,parent,path,intrinsic}` | log-normal, max_mean=15, min_mean=3 | 四组信号能量缩放 |
| `basis_perturb_eta` | 0.10 (固定) | 打破 rank-1 块结构 |
| `loading_log_mean` | Uniform(0.35, 1.7) | LogNormal 加载幅度的均值 |
| `loading_sigma` | Uniform(0.2, 0.6) | LogNormal 加载幅度的标准差 |
| `max_groups_per_feature` | {2, 3} | 每个特征最多几个信号组 |
| `archetype_perturb_std` | Uniform(0.05, 0.15) | α 乘性扰动的标准差 |
| `coupling_lambda` | Uniform(0.02, 0.06) | 跨特征 uniform 耦合强度 |
| `coupling_rank` | {2, 3} | 耦合矩阵的秩 (当前 uniform 耦合下等价于 lambda) |
| `residual_sigma` | Uniform(0.1, 0.4) | 每特征独立噪声标准差 |
| `basis_group_divisor` | {3} | 用于计算 `K_time` (parent/path 用固定公式) |

### 11.2 K 计算总结

| 组 | 维度 | K | 公式 |
|---|---|---|---|
| time | 8 | 3 | `min(3, max(1, ceil(8/divisor)))` |
| parent | 12 | 4 | `max(1, ceil(12/3))` |
| path | 8 | 4 | `max(1, ceil(8/2))` |
| intrinsic | 8 | 4 | `max(1, ceil(intrinsic_dim/2))` |

---

## 12. 关键文件索引

```
data_generation/RDB/
├── dag_to_rdb_generator.py          # 主入口: DAG 解析, RDB 创建, 并行生成
├── dag_to_rdb_config_small.yaml     # 行列数映射规则
├── datasets/rdb_v1.pth              # DAG 结构数据
│
└── src/
    ├── prior/
    │   ├── mlp_scm.py               # MLPSCM: SCM 核心 + Signal-Group Feature Gen
    │   ├── hsbm.py                  # HSBM FK 采样 (单父表 + 多父表联合)
    │   ├── prior_config.py          # DEFAULT_SAMPLED_HP, DEFAULT_HSBM_HP, TEMPORAL_FOURIER_HP
    │   ├── hp_sampling.py           # HpSamplerList, HpSampler (超参数采样)
    │   ├── temporal_vocab.py        # TemporalVocab, EventCalendar, SeasonalityVocab
    │   ├── activations.py           # get_activations()
    │   ├── utils.py                 # GaussianNoise, XSampler, MASK_TYPE, SCM_OUTPUT
    │   └── row_gnn.py              # RowGraphBuilder, RowGNNRunner (可选行级 GNN)
    │
    └── table_def/
        ├── table_generation.py      # Table, Relationship, RDB, TableGenerator
        ├── task_generation.py       # Task, TaskGenerator, TaskDataGenerator
        ├── task_generation_utils.py # SchemaGraph, TargetNodeSet, AggregationFunction
        ├── task_quality.py          # Stage A/B Quality Gate, eigen spectrum, Wasserstein
        ├── reg2cls.py               # Regression → Classification 转换
        ├── dataset_meta.py          # DBBRDBDatasetMeta, DBBTableSchema, DBBColumnSchema
        └── yaml_utils.py            # load_pyd, save_pyd

data_preprocessing/
├── run_preprocess.py                # Pre-DFS → DFS → Post-DFS 三步预处理
├── merge_dbinfer_to_h5.py           # 合并处理后的数据为 H5
└── configs/
    ├── transform/pre-dfs.yaml       # Pre-DFS 转换配置
    ├── dfs/dfs-2-ft.yaml            # DFS 配置 (depth=2, featuretools)
    └── transform/post-dfs.yaml      # Post-DFS 转换配置

scripts/
└── run_pipeline.sh                  # 一键运行全流水线 (支持 --relbench_mode)
```
