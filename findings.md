# Findings: Temporal Generation Refactor

## Current Time Generation Architecture

### Three Problems Identified

1. **虚假相关性 (Spurious correlation)**: `TemporalVocab.sample_time()` samples timestamps from intensity distribution, then passes the SAME raw time values to MLP as input. MLP learns "high-intensity periods → special feature values" as a shortcut, which does not exist in real data.

2. **跨表时序不一致 (Cross-table temporal inconsistency)**: Parent and child tables sample timestamps independently. Child row can get a timestamp earlier than its parent row (e.g., order before user registration).

3. **FK 组内时间戳无结构 (No intra-group structure)**: All child rows sample timestamps independently from the same global intensity. No differentiation between different FK groups (e.g., different users' order sequences).

### Root Cause

All three problems stem from a single architectural flaw: **time generation conflates "time as event occurrence distribution" with "time as feature influence factor," and lacks cross-row/cross-table temporal structure constraints.**

---

## Code Trace

### Timestamp Generation Flow (current)

```
TemporalVocab.generate()
  → trend + seasonal + spikes + noise → intensity(t) on [0, 10] grid
  → TemporalVocab.sample_time() → samples timestamps from intensity via multinomial
  → MLPSCM._sample_time_embedding():
      - normalizes to [0, 1]
      - optional Fourier encoding (time_embed_mode="fourier")
      - concat to MLP causes
      - store in X[MASK_TYPE.TIMESTAMP]
```

### Key Files

| File | Key Content |
|------|------------|
| `data_generation/RDB/src/prior/temporal_vocab.py` | `TemporalVocab`, `TrendVocab`, `SeasonalityVocab`, `SpikesVocab`, `NoiseVocab` |
| `data_generation/RDB/src/prior/mlp_scm.py` | `MLPSCM._sample_time_embedding()` (L415-446), `forward_without_input()` (L448-479), `forward_with_input()` (L481-533) |
| `data_generation/RDB/src/table_def/table_generation.py` | `TableGenerator._compute_hsbm_fk_ids()` (L1213-1267), `_convert_to_datetime64()` (L714-804), `init_table_SCMs()` (L1539-1651) |
| `data_generation/RDB/src/prior/prior_config.py` | HP config (no time-specific params currently) |
| `data_generation/RDB/dag_to_rdb_generator.py` | Default timestamp config (L60-64): prob=1.0, time_dim=8, time_embed_mode="fourier" |

### HSBM FK Allocation Confirmed

- `child_rows = self.num_rows` (table_generation.py:1229)
- HSBM determines FK mapping `fk_ids[j] = parent_idx` for each child row
- After HSBM, each parent row p has N_p child rows (implicit from FK assignment)
- Problem 3 is: arrange N_p known timestamps within `[t_parent_p, T]`

### Timestamp Config

- Default `prob: 1.0` — all tables get timestamp column
- `time_dim: 8` (default), `time_embed_mode: "fourier"` (default)
- Timestamps stored as `np.datetime64[D]` in range [1970-01-01, 2020-12-31]
- Each timestamp column gets random sub-range within bounds

---

## Design Decisions (from brainstorming)

### Problem 1: Basis Vector Input

- `time_dim = 11`: trend(2) + seasonal(4) + spike(2) + gates(3)
- Trend: level + slope (2D)
- Seasonal: low-freq sin/cos + high-freq sin/cos (4D)
- Spike: amplitude + decay (2D)
- Gate: [trend_active, seasonal_active, spike_active] (3D) — 0-masking for inactive components
- Noise stays in sampler only (NOT passed to MLP)
- No fallback to raw time if all gates=0 (design inconsistency)
- Remove Fourier encoding and `time_embed_mode` config

### Problem 2: Cross-Table Constraints

- Child t_min = max(timestamps of all parent rows referenced by FK)
- Fallback based on DAG topology: source→Uniform(0, 0.1T), intermediate→Uniform(0.2T, 0.4T), leaf→Uniform(0.4T, 0.6T)
- Gamma decay: discrete tiers {0.0, 0.5, 1.5, 3.0}
- Decay uses normalized relative time: tau = (t - t_min) / (T - t_min)
- Only collect timestamps from parents that have timestamp column (no force-requirement)

### Problem 3: Intra-Group Sort

- Independent sampling within `[t_min_j, T]` per row (window constraint already provides inter-group differentiation)
- Probabilistic per-FK-group sort: p_sort ~ Uniform(0.3, 0.8) per table
- Not all tables should have sorted groups (analytical tables shouldn't)
- No Hawkes process (too e-commerce specific for general database generation)

### NaN / Zero Defense Mechanisms

**No unified defense mechanism exists.** Each calculation site handles edge cases independently:

| File | Line | Calculation | Defense |
|------|------|-------------|---------|
| `analyze_rdb_stats.py` | 592 | Gini coefficient | `if total_refs > 0 and n > 1` else `gini=0.0` |
| `analyze_rdb_stats.py` | 692 | Shannon entropy | `np.errstate(divide="ignore")` + `isfinite` check → fallback 0/1 |
| `analyze_rdb_stats.py` | 582 | In-degree distribution | `len(child_fk_vals)==0` → all-zero in-degree |
| `analyze_rdb_stats.py` | 682 | Coverage ratio | `n_parent_pks==0` → skip edge |
| `task_quality.py` | 206 | Label uniqueness | `n_unique < 2` → reject task (a1) |
| `task_quality.py` | 222 | Sample count | `n < 64` or `min(n_pos,n_neg) < 8` → reject (a3) |
| `task_quality.py` | 244 | Child/parent ratio | `ratio > 100` → reject (a6) |
| `temporal_vocab.py` | 258 | Intensity NaN | `torch.where(isnan, 1.0, intensity)` → fill 1 |

**Risk:** When FK data is extremely sparse (child table has 0-1 rows), Gini is hard-coded to 0 ("perfectly uniform"), which is misleading but won't crash. For our new `t_min = max(parent_ts)` logic, the fallback path (DAG topology) already handles the empty `T_parents` case.

### Rejected Approaches

- Full Z(t) latent state with NHPP (too complex, deferred)
- Hawkes self-excitation (too e-commerce specific)
- Forcing all parent tables to have timestamps (too restrictive)
- Noise component in MLP input (would become pseudo-signal)
- Raw time fallback when all gates=0 (design contradiction)

---

## Implementation Notes

1. `TemporalVocab.evaluate_basis(t)` must evaluate at arbitrary continuous t, not just grid points. Current `generate()` discretizes on 1000-point grid. Need analytical evaluation of each basis component.
2. Per-row `t_min` sampling via batched multinomial: construct `(N, K)` mask, zero out intensities below t_min, renormalize per row, categorical draw.
3. Sort is applied AFTER sampling — does not affect marginal distribution.
4. Gamma decay with normalized tau: `exp(-gamma * tau_j)` where `tau_j = (t - t_min_j) / (T - t_min_j)`. If `t_min_j = T`, tau is undefined → fallback to tau = 0 (single-point window).
