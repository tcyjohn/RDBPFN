# FK Matching Latent: Cross-Table Feature Correlation via Shared Projection

## Summary

Replace post-hoc injection (struct_sig) with endogenous FK sampling that couples parent feature values to FK connection decisions. Single-parent tables use restored FK Propensity (Section 12); multi-parent tables use a new shared-matching-latent mechanism where parent features are projected through a shared W matrix into a comparable low-D space, then FK sampling biases toward row tuples with matching latent values.

## Deletion: struct_sig

- Remove `data_generation/RDB/src/prior/struct_sig.py`
- Remove pipeline hook in `table_generation.py` (`generate_all_data_from_SCM()`)
- Remove `struct_sig_*` params from `table_generation.py` (`init_table_SCM()`)
- Remove `struct_sig_*` HPs from `prior_config.py`

## Single-Parent: Restored FK Propensity

Restore `compute_hsbm_fk_ids_with_propensity()` from Section 12 (previously removed).

Flow:
```
per-parent: rank_score[a] = percentile_rank(random_linear_combo(feature[a, 2-3 cols]))
per-child:  t(j) = rho * z(j) + sqrt(1-rho^2) * eps(j)  where z(j)~Uniform
sampling:   within HSBM block, weight by exp(-beta * |rank_score[a] - t(j)|)
```

HPs: `propensity_rho` [0.05, 0.20], `propensity_beta` [2.0, 8.0]

## Multi-Parent: Shared Matching Latent

### Core Idea

All parent tables share the same projection matrix W, so parent_A's latent and parent_B's latent live in the same space and are directly comparable. FK sampling biases toward tuples where all parents' matching latents are close to a shared target.

### Per-Table Normalization

Each parent p's CAUSAL_OUTPUT is z-score normalized per-column BEFORE projection, ensuring different-scale parent features map to comparable latent values.

### Dimension Handling

D_common = min(D_1, D_2, ..., D_P) — truncate to smallest common dimension across parents for that child table.

### Algorithm

```
For each multi-parent child table C with P parents:

  # Once per parent table
  For each parent p:
    X_norm[p] = zscore(parent_p.CAUSAL_OUTPUT)  # per-column
    D_common = min(D_1, ..., D_P)
    W_shared ~ N(0, 1/D_common)  # shape: (D_common, D_latent), same for all parents

  # Once per child row j
  For each child row j:
    target[j] ~ N(0, I)  # D_latent-dim standard normal
    cluster_path[j] ~ random (existing HSBM logic, unchanged)

  For each parent p:
    matching_latent[p] = X_norm[p][:, :D_common] @ W_shared  # (N_p, D_latent)

    For each child row j:
      candidates = rows in parent p assigned to cluster_path[j][:n_levels[p]]
      scores[a] = -||matching_latent[p][a] - target[j]||^2
      probs = softmax(scores / temperature)
      FK_p[j] = sample(probs)
```

### Key Differences from FK Propensity

| | Propensity (single) | Matching Latent (multi) |
|---|---|---|
| Projection | 1D random combo per parent | Shared W: all parents → same D_latent space |
| Comparability | Ranks per-parent, not comparable | Latents in shared space, directly comparable |
| Target | 1D scalar via rho-mixing | D_latent-dim standard normal |
| Sampling | exp(-beta * |rank - t|) | softmax(-||latent - target||^2 / temp) |

### HPs

| HP | Distribution | Description |
|---|---|---|
| matching_latent_dim | {2, 3, 4} | Shared latent space dimension |
| matching_temperature | log-uniform [0.1, 0.5] | Softmax temperature (smaller = stronger bias) |

## Evaluation

### Metrics

Separate by task type (single-parent vs multi-parent tasks in complex tasks):

| Metric | Definition |
|---|---|
| native_corr | Correlation among a table's own feature columns |
| native-joined_corr | Correlation between native features and joined parent features |
| joined_corr | Correlation among joined parent feature columns (from same parent) |
| cross_corr | Correlation between joined features from different parents (multi-parent only) |

### Pipeline

1. Generate 64 RDBs via `dag_to_rdb_generator.py`
2. Preprocess via `run_pipeline.sh` (skip training)
3. Evaluate at H5 level (after DFS, 30-col subsample)

### Baseline → Final comparison

Same 64 seeds (0-63), compare pre-change vs post-change on all metrics.
