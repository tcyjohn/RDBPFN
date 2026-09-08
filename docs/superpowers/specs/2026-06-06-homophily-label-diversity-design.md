# Homophily-Controlled Label Diversity Design

**Date**: 2026-06-06
**Branch**: `feature/hsbm-temporal-refactor`
**Status**: Spec — awaiting user review

## Motivation

### Problem

Corruption probe on v5.3 (1024 RDBs, linear SG) reveals a borderline lazy/feature-learning regime:

| Condition | Accuracy | Δ |
|-----------|----------|---|
| Clean | 0.8175 | — |
| Shuffle labels | 0.7874 | +3.0% |
| Shuffle features | 0.7895 | +2.8% |

- 12/19 tasks: lazy (Δ < 1%)
- 7/19 tasks: partial feature-learning (Δ 1-16%)
- Aggregate Δ ≈ 3% — better than PluRel (Δ ≈ 0.3%) but far from real-data cotraining (Δ ≈ 5-15%)

### Root Cause (from OPENRFM, arXiv:2606.04320)

PluRel/RDBPFN synthetic pre-training leads to a lazy regime because the label-generation process lacks a **support-identifiable relational latent**. Existing priors (schema, SCM, HSBM) contain relational components but none is an explicit, recurring axis that controls label-FK dependence across episodes.

### Hypothesis

Injecting a per-RDB **homophily target h** into label generation will push the model from borderline lazy into a stronger feature-learning regime, increasing downstream performance by making the model genuinely use relational structure.

## Design

### Architecture

```
Existing pipeline (UNCHANGED):
  SCM → X (features) → SG → X_sg (enhanced features)

New (task generation only):
  Per-RDB: h ~ Uniform({h₁, ..., h₂₀})  (K=20 grid on [-1, +1])

  For task table rows:
    ├── y_cluster = f(block_hierarchy, h)    # FK-structure-driven
    ├── y_feature = random_projection(X_cols) # Feature-driven
    └── y = mix(y_cluster, y_feature, |h|)   # Replace target column
```

### Core Component: `HomophilyLabelGenerator`

```
data_generation/RDB/src/prior/homophily.py

class HomophilyLabelGenerator:
    """
    h_target: float ∈ [-1,+1]       — per-RDB homophily target
    block_assignments: (n_rows, 2)   — (b_parent, b_child) per row
    features: (n_rows, n_feats)      — selected feature columns

    y_cluster (h > 0, homophily):
        b_parent % 2                 → same parent block = same label

    y_cluster (h < 0, heterophily):
        (b_parent % 2) XOR (b_child % 2)  → cross-block alternation

    y_feature:
        ϕ = random projection (n_feats → 1)
        y_feature = (σ(ϕ · features) > 0.5).int()

    y = y_cluster  with prob |h|
        y_feature  with prob 1-|h|

    Returns: y (n_rows,), meta {actual_homophily, cluster_ratio, ...}
    """
```

### Block Hierarchy Strategy

**Two-tier approach**: try HSBM blocks first, fall back to pseudo-blocks.

```
get_block_assignments(target_table, rdb):
  1. Check if target_table has FK parents (is child in any Relationship):
     → YES: use HSBM block paths from FK generation
       - block_paths already stored during FK generation phase
       - b_parent = block_path[:, 0], b_child = block_path[:, 1]
     → NO (ultimate root entity, no FK parents at all):
       - Use pseudo-blocks from feature clustering:
         a. Select 3-5 random feature columns → (n, d)
         b. Random projection → (n, 1)
         c. Quantile: 2 parent blocks, 4 child blocks hierarchically
```

**Why this works**:

- In relbench mode, the target is always the subgraph root (entity table), but the entity may have FK parents in the full RDB that are outside the eval subgraph. If so, HSBM block paths exist.
- For the ultimate root entity (no FK parents anywhere), pseudo-blocks provide structural variation.
- For `relbench_mode=False` with non-root targets, HSBM blocks are always available.

### Integration Point

In `task_generation.py:generate_tasks_for_rdb_with_complex_tasks()`:

```
After target_column_name selection:
  1. Check: is target table categorical? (binary classification task)
  2. Check: prob_homophily ≈ 0.5 (half of RDBs use homophily labels)
  3. If yes:
     a. Build block hierarchy:
        - relbench_mode: pseudo-blocks from feature clustering
        - otherwise: HSBM block paths from FK hierarchy
     b. HomophilyLabelGenerator(h_target, blocks, features) → y_new
     c. Replace target column data in table with y_new
     d. Record h_target in task metadata
  4. If no:
     - Use existing target column (baseline behavior)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| h grid | K=20, uniform on [-1,+1] | Matches OPENRFM; enough for support-identifiability |
| Mix ratio | prob_homophily = 0.5 | Half diversity/half baseline for safe A/B comparison |
| Label type | Binary classification only | Simplest case; multi-class/regression as follow-up |
| Block source | HSBM first, pseudo-block fallback | Entity tables may have FK parents in full RDB (outside eval subgraph); ultimate roots use feature clustering |
| FK sparsity | Disabled (FK=-1 → no orphan rows) | v5.3 baseline; orphans would complicate block hierarchy |
| Feature-driven channel | Random projection of 3-5 feature cols | Simple, keeps causal structure |
| Feature pipeline | UNCHANGED | SCM + SG runs as before; only label column is replaced |

### File Changes

| File | Change | Scope |
|------|--------|-------|
| `src/prior/homophily.py` | NEW: `HomophilyLabelGenerator` class | ~80 lines |
| `src/prior/prior_config.py` | NEW: `DEFAULT_HOMOPHILY_HP` config | ~15 lines |
| `src/table_def/task_generation.py` | MODIFY: label substitution in `generate_tasks_for_rdb_with_complex_tasks` | ~30 lines |
| `dag_to_rdb_generator.py` | MODIFY: add `use_homophily_labels` flag, pass to TaskGenerator | ~10 lines |

### Validation Plan

1. **Unit test**: `HomophilyLabelGenerator` produces labels with correct homophily properties
2. **64-RDB generation**: 32 homophily + 32 baseline (prob_homophily=0.5)
3. **Preprocess + train**: Small-scale (~20000 steps) on 64 RDBs
4. **Corruption probe**: Compare Δ(shuffle_labels) — expect +8% or higher
5. **Downstream eval**: Compare clf_rel_npz AUC against v5.3 baseline

### Risks

1. **Pseudo-blocks too weak**: Feature-clustering blocks may not create enough label-FK structure for the model to latch onto. Mitigation: try different clustering methods (k-means, random projection, actual degree-based blocks)
2. **Distribution mismatch**: Homophily labels come from a different distribution than normal features. Mitigation: the mix ratio (|h|) creates a smooth interpolation
3. **Model doesn't benefit**: Corruption probe Δ doesn't improve → homophily diversity not the bottleneck for PFN architecture. Mitigation: fall back to other diversity axes (schema, FK sparsity, task complexity)
