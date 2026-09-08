# Findings

## 2026-07-25 — Two-by-two operating-point sensitivity grid

Four independently materialized R3 corpora form a descriptive sensitivity grid:

| Temporal-history density | Grouped sources | Stored tasks | Raw cells | AUROC |
|---|---:|---:|---:|---:|
| Dense | On | 1,600 | 1.010B | 0.7075542321 |
| Dense | Off | 1,626 | 1.010B | 0.7005588277 |
| Low | On | 2,656 | 128.40M | 0.6931528116 |
| Low | Off | 2,667 | 128.40M | 0.6672127449 |

Aligned-support conditional differences:

- Grouped sources at dense history: +0.0069954, 95% paired t interval
  [0.0034739, 0.0105169].
- Grouped sources at low history: +0.0259401, interval
  [0.0196857, 0.0321944].
- Dense history with grouped sources: +0.0144014, interval
  [0.0121490, 0.0166539].
- Dense history without grouped sources: +0.0333461, interval
  [0.0272207, 0.0394714].
- Complete SA-RDB-PFN versus the joint-removal operating point: +0.0403415,
  interval [0.0328286, 0.0478544].

The descriptive difference-in-differences is -0.0189447, but the four corpora
were independently generated and contain unequal task counts. It belongs only
in the supplement and must not be framed as a causal interaction. The larger
conditional gain from either factor when the other is absent is consistent with
partial substitution or diminishing marginal returns.

## 2026-06-19 — FK Bias Signal Deficit Root Cause: Row-Level FK Values

### FK column stores parent row index, not entity ID

In `process_data()` (table_generation.py:888), FK values are parent row indices from HSBM.
For entity parents with S snapshots per entity, rows [k, k+S−1] belong to entity E.
Child A connecting to parent row k gets FK=k, child B to parent row k+1 gets FK=k+1.
Both→"same entity" but different FK values → FK bias sees zero match.

### HSBM within-cluster FK density is invariant to hierarchy

Expected children/parent within cluster = child_rows/parent_rows (multinomial allocation).
Reducing clusters_per_level or num_levels does NOT change this ratio.
Propensity (ρ ∈ [0.05, 0.20]) already enabled in defaults.

### FK pair ratio data

| Dataset | FK pair ratio | Entity pair ratio | FK bias learning |
|---------|--------------|-------------------|-----------------|
| v6.1 | 0% (no FK cols) | 0.06% | N/A (disabled) |
| v6.2 | 0.015% | 1.35% | Frozen at init |

### v6.2 FK bias check: all layers frozen

All 6 layers: raw_lambdas = −2.252168 (softplus⁻¹(0.1)), same as init.
Entity bias: layer 5 softplus(1.33) ≈ 1.57, others near zero.

## 2026-06-19 — RelBench Benchmark Task Distribution

Real benchmark metadata (all rel-* datasets): 100% tasks on entity tables as root.
Root entity = has children (out_degree ≥ 1) but no FK parents → flat table has no FK cols.
Entity tables in real benchmarks are typically roots of FK graph.

## 2026-06-19 — OpenRFM vs RDBPFN Architecture Comparison

### RT (OpenRFM backbone) FK mechanism
- BFS walk follows FK edges → cell token sequence
- M_fk attention MASK: structural constraint, not soft bias
- FK operates at context construction + attention routing
- Root entities still get child cells via BFS walk

### RDBPFN FK mechanism
- DFS precomputes aggregations (featuretools) → flat table
- FK bias: λ × I[fk_i == fk_j] as additive soft hint
- Only works when flat table has valid FK columns (task must have FK parents)
- entity table as root → flat FK cols = −1 → FK bias useless

### Dual-stage ICL (OpenRFM)
- Relational block + batch-level TabICL cross-attention
- Second ICL channel independent of FK neighborhood
- PFN also provides batch-level ICL naturally

## 2026-06-19 — Design: parent_entity_ids for Entity-Level FK Bias

### Problem
FK = parent row index → entity snapshots create FK value fragmentation.
S = snapshots per entity → effective FK target space = N×S instead of N.

### Solution
Compute `parent_entity_ids` column: maps FK row index → parent entity_id.
- Entity parents: `parent_entity_ids[i] = parent.entity_ids[fk_value[i]]`
- Non-entity parents: `parent_entity_ids[i] = fk_value[i]` (no snapshots)
- FK bias uses parent_entity_ids for matching → all children of same entity match
- Expected FK pair ratio: ~S× current → from 0.015% to 1-2%+

### Previous findings below...

## 2026-06-07 — fk_block_id Experiment: PFN Cannot Leverage Block Structure

### Setup
- 1024 RDBs, `--no_path_signal` + `--use_homophily_labels`
- fk_block_id as explicit categorical column (hash of HSBM block path)
- Path signal removed from SG

### Result
- **Corruption probe: no improvement** — model could not leverage explicit block_id
- **Conclusion**: PFN's dense between-datapoints attention fundamentally cannot learn relational structure from features alone, even when given perfectly clean block membership signals
- Bottleneck is in ARCHITECTURE, not signal strength
- Need FK-aware attention bias to provide inductive bias

---

## 2026-06-07 — Homophily Label Diversity: No Improvement

### Corruption probe comparison

| Metric | v5.3 | hp2_homophily | Δ |
|--------|------|---------------|---|
| Clean acc | 0.8175 | 0.8247 | +0.7% |
| Δ(shuffle_labels) | +3.01% | +3.52% | +0.5% |
| Δ(shuffle_features) | +2.80% | +3.52% | +0.7% |

### full-128 downstream

| Model | Avg AUROC |
|-------|-----------|
| v5.3 | 0.6688 |
| hp2_homophily | 0.6630 |

### Root cause: PFN architecture can't leverage FK-aware label diversity

- RT has explicit FK-neighbor attention masks (`M_fk`) → model naturally knows which FK-related rows
- PFN has dense all-to-all attention → FK relations must be learned from feature similarity
- Path signal is too weak: 8-dim embedding → projection → diluted by time/parent/intrinsic
- Homophily diversity needs FK-aware attention to work; PFN lacks this inductive bias

### Path signal weakness (confirmed)

Path signal flow: `HSBM block_paths → PathEncoder(embedding) → 8-dim → project → dilute → feature`

- 8-dim compression of multi-level block hierarchy
- Learned embedding (not deterministic) — different blocks may get similar embeddings
- Diluted: each feature = weighted sum of time + parent + path + intrinsic projections
- Cross-feature coupling further mixes signals

### Decision: explicit FK block ID column + remove path signal

Add `fk_block_id` as categorical column during data generation. PFN sees FK structure directly as a feature value, no need to "discover" it from weak embeddings.

---

## 2026-06-06 — Corruption Probe: v5.3 Borderline Lazy/Feature-Learning Regime

### Method

Context corruption probe (OPENRFM Table 2 methodology) on v5.3 (1024 RDBs, linear SG, relmode+complex+quality gate).

### Results (19 clf_rel tasks)

| Condition | Mean Accuracy | Δ |
|-----------|--------------|---|
| Clean | 0.8175 | — |
| Shuffle labels | 0.7874 | +3.01% |
| Shuffle features | 0.7895 | +2.80% |

- 12/19 tasks: lazy (Δ < 1%)
- 7/19 tasks: partial feature-learning (Δ 1-16%)

### Comparison with OPENRFM

| Model | Δ(shuffle_labels) | Regime |
|-------|-------------------|--------|
| RT-synthetic (PluRel) | ~0.3% | Pure lazy |
| RT-synth-diverse (+homophily) | ~3-4% | Partial FL |
| RT-cotrain (real data) | ~5-15% | Strong FL |
| **RDBPFN v5.3** | **+3.0%** | **Borderline** |

---

## 2026-06-06 — relbench_mode and Entity Table FK Structure

### relbench_mode constraints
- `entity_task_ratio=1.0`: focal is always entity table (has children FK-referencing it)
- `root_p=1.0`: target = subgraph root = entity table itself
- Entity tables are roots of subgraph but may have FK parents in full RDB
- HSBM block_paths exist for entity tables if they have FK parents in full RDB
- For ultimate root entities: pseudo-blocks from feature clustering

---

## 2026-06-08 — Entity Table Temporal Snapshots + Same-Entity Bias Design

### Real benchmark entity table structure

Analyzed `model_pretrain/rdb_datasets/` (rel-amazon-dfs-2, rel-stack-dfs-2, amazon-dfs-2):

- **Entity tables have MULTIPLE rows per entity**: user-churn 4.7M rows with `customer_id`, each customer appears at many timestamps
- **All task tables have time_column**: 100% of tasks across all 3 datasets
- **Non-root entity tasks exist**: post-votes (rel-stack) targets `posts` which has FK parents; rating/purchase (amazon-dfs-2) target `Review` (child table)
- Current generation assumption ("entity=1 row, no timestamp") contradicts all real benchmarks

### plurel reference

- Entity tables: multi-row (num_rows range [500, 1000])
- Entity tables: NO timestamps (min/max_timestamp = None for Entity type)
- Activity tables: multi-row (num_rows range [2000, 5000]), HAVE timestamps

### HSBM cluster_a vs cluster_b

- `cluster_a`: parent-side block assignments (shape: parent_rows × n_levels)
- `cluster_b`: child-side block assignments (shape: child_rows × n_levels)
- Currently `cluster_a` is COMPUTED in `compute_hsbm_fk_ids()` but DISCARDED
- Only `cluster_b` is returned as `block_paths` (stored on child table)
- Root entities have no FK parents → never get HSBM block_paths → homophily falls to pseudo-blocks
- Fix: store `cluster_a` when entity acts as parent → entity gets genuine HSBM blocks → replaces pseudo-blocks

### OPENRFM label determination

- Label = `b_parent % 2` (block-based, NOT FK adjacency based)
- For non-root entities: blocks from HSBM child-side `cluster_b` (genuine structural signal)
- For root entities: blocks from pseudo-feature-clustering (NOT HSBM-structured — gap)
- FK-neighbor attention in RT is parent↔child (bidirectional), works for ALL entities
- PFN FK bias is sibling-only (child↔child sharing same FK parent) — different mechanism

### PK bias decision: DROPPED

- No OPENRFM precedent, mechanism too complex
- FK bias + same-entity bias provide sufficient coverage:
  - FK bias: child sibling clustering (cross-table structure)
  - Same-entity bias: entity temporal self-connection (within-table temporal)
- `cluster_a` is still stored for homophily label quality (replaces pseudo-blocks for root entities)

### Three biases (now two)

| Bias | Formula | Purpose | Works on root entity |
|---|---|---|---|
| FK bias | I[fk[i]==fk[j]]·λ_fk | Child siblings share attention | NO |
| Same-entity bias | I[eid[i]==eid[j]]·λ_se | Same entity across time | YES |
| ~~PK bias~~ | ~~I[cluster_a[i]==cluster_a[j]]~~ | DROPPED | — |

### Relbench_mode and root entities

- relbench_mode: entity_task_ratio=1.0, root_p=1.0 → always targets root entity
- With same-entity bias, root entities ARE covered (temporal self-connection)
- No need to change relbench_mode to exclude root entities
