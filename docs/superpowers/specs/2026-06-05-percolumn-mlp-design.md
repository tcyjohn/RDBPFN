# Per-Column Nonlinear MLP for Feature Diversity

**Date**: 2026-06-05
**Status**: Design approved, ready for implementation

## Motivation

Signal Group (SG) features suffer from low effective rank because all feature columns are linear combinations of the same 4-5 low-dimensional signal vectors:

```
feature_j = Σ sign × magnitude × (signal @ basis_vector) + ε
```

Since `signal` has at most ~40-50 independent dimensions across all sources (time_basis:8, time_gates:3, parent:12, path:8, intrinsic:12), the feature matrix rank is bounded by dim(signal), causing all RDBs to exhibit `eff_rank_ratio ≈ 0.53`. The model learns this as a shortcut.

Additionally, `v5.x eval` does not improve with increasing SG complexity (v5:0.699 → v5.3:0.684), confirming that feature diversity—not signal complexity—is the bottleneck.

## Core Idea

Replace the shared linear basis projection in `_construct_features` with **per-column random nonlinear MLPs**, while preserving SG's two proven mechanisms:

1. **Multi-source signal combination**: archetype-dependent signal source selection (time/parent/path/intrinsic)
2. **Per-group differentiated scale**: RMS-norm × group_scale for each signal type

## Design

### Architecture

```
Old (linear SG):
  signals → RMS-norm × scale → feature_j = Σ(signal @ basis_vector)

New (per-column MLP):
  signals → RMS-norm × scale → MLP_j(concat(scaled_signals)) + residual_noise
```

### Signal Flow

```
For table with archetype ∈ {source, non_ts_child, ts_child}:

1. time_sig, parent_sig, path_sig, intrinsic_sig  ← built as before
2. RMS-norm × group_scale per signal              ← preserved from SG
3. For each feature j:
     inputs = concat(selected_signals)             ← archetype-dependent
     feature_j = MLP_j(inputs) + ε_j              ← per-column, random, fixed
```

### Per-Column MLP Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_layers` | `∈ [2,4]` per-table sampled | Plurel `mlp_num_layers_choices` |
| `hidden_dim` | 32 (fixed) | Plurel `mlp_emb_dim` |
| `activation` | per-layer sampled (ReLU/GELU/Tanh/LeakyReLU) | Plurel pattern |
| `init_std` | `DEFAULT_SAMPLED_HP["init_std"]` | Existing config |
| `input_dim` | adaptive (12/32/43 by archetype) | Natural concat |
| `output_dim` | 1 (scalar per feature) | — |
| `seed` | `hash(rdb_id, table_name, col_idx, layer_idx, 0x7FFFFFFF)` | Per-RDB diversity |

### Archetype Input Dimensions

| Archetype | Signals | Total dim |
|-----------|---------|-----------|
| source | intrinsic_sig | 12 |
| non_ts_child | parent_sig + path_sig + intrinsic_sig | 32 (12+8+12) |
| ts_child | time_basis + time_gates + parent_sig + path_sig + intrinsic_sig | 43 (8+3+12+8+12) |

### Removed vs Preserved from SG

**Preserved:**
- Signal source selection logic (archetype → signal sources)
- `_normalize_signals`: RMS-norm + group_scale per signal type
- `residual_sigma`: per-feature residual noise
- `_build_time_signal`, `_build_parent_signal`, `_build_path_signal`, `_build_intrinsic_signal`

**Removed:**
- Shared basis vectors (`group_bases`, `self.group_bases`)
- `_perturb_basis` — basis perturbation no longer applicable
- `coupling_lambda`, `cross_feature_coupling` — MLP nonlinearity breaks linear dependency
- `basis_perturb_eta`, `archetype_perturb_std` — basis concept removed
- `feature_assignments` — replaced by per-column MLP dispatch
- `_construct_features` — replaced by `_construct_features_mlp`
- `_apply_cross_feature_coupling` — removed

**Deprecated HP config keys (kept for backward compat, ignored):**
- `max_groups_per_feature`, `loading_sigma`, `loading_log_mean`
- `basis_group_divisor`, `coupling_rank`, `coupling_lambda`
- `basis_perturb_eta`, `archetype_perturb_std`, `num_basis_families`
- `basis_family_rho`
- `group_scale_time`, `group_scale_parent`, `group_scale_path`, `group_scale_intrinsic`
  — these remain ACTIVE (used in `_normalize_signals`)

### New HP Config

```python
PERCOL_MLP_HP = {
    "percol_mlp_num_layers": {
        "distribution": "meta_choice",
        "choice_values": [2, 3, 4],
    },
    "percol_mlp_hidden_dim": 32,  # fixed
    "percol_mlp_activation": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
}
```

### Feature Gate

Gated behind `use_percol_mlp: bool` (default `True`). When `False`, falls back to current SG path. The old SG basis machinery remains in code but unused when per-column MLP is active.

### Expected Effects

- **eff_rank**: from ~0.53 → ~min(N, F)/F (each feature independently nonlinear)
- **feature-label coupling**: maintained (intrinsic_sig ← CAUSAL_OUTPUT causal chain preserved)
- **Cross-RDB diversity**: maximized (independent random MLP per RDB/table/column)
- **Aggregation features**: benefit from per-table feature diversity increase

## Files Changed

| File | Change |
|------|--------|
| `prior_config.py` | Add `PERCOL_MLP_HP`, add `use_percol_mlp` to `DEFAULT_SAMPLED_HP` |
| `mlp_scm.py` | Add `PerColumnMLP` class, `_construct_features_mlp` method, gate logic in `forward_with_input`/`forward_without_input` |
| `table_generation.py` | Pass percol MLP HP through HP sampling pipeline |

## Implementation Phases

### Phase 1: PerColumnMLP module
- `PerColumnMLP(in_dim, hidden_dim, num_layers, activations, init_std)` class
- Forward: Linear → Act → ... → Linear → scalar output
- Weight init: `N(0, init_std)` per layer

### Phase 2: Integration into MLPSCM
- `_construct_features_mlp(signals, n_features)` — replaces `_construct_features`
- Per-column MLP dispatch with pre-computed seeds
- Gate behind `use_percol_mlp`

### Phase 3: HP plumbing + verification
- Add to `DEFAULT_SAMPLED_HP`
- Pass through `table_generation.py` HP sampling
- Generate 64-RDB test, run feature analysis (eff_rank, corr, fl_corr)
- Compare vs v53 baseline

### Phase 4: Full experiment
- 1024-RDB training + eval
