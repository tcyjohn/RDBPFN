# Temporal Generation Refactor Design

## Motivation

当前时间生成有三个问题：

1. **虚假相关性**：MLP 读取 raw time value 来决定 feature，而 raw time 本身来自 intensity-based 采样——高 intensity 时段的 feature 值特殊，模型学到"密集=特殊"的捷径，但这在真实数据中不存在。
2. **跨表时序不一致**：子表时间戳独立于父表采样，可能出现"订单时间早于用户注册时间"。
3. **FK 组内时间戳无结构**：同一父行下的 N 个子行时间戳独立采样，未区分不同 FK 组。

## Core Principle

**时间戳采样、时间→特征的映射、跨表时序约束三者解耦，但通过共享的 basis functions 保持因果一致性。**

```
                    +---------------------------+
                    |   Basis Functions B(t)    |
                    |  trend + seasonal + spike |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
              lambda(t)       B(t_i)      window [t_min, T]
              (intensity)   (MLP input)  (per-FK-group constraint)
                    |             |             |
                    v             v             v
            timestamp sampling  features    intra-group sort
            (density control)  (MLPSCM)    (optional, probabilistic)
```

---

## Design

### 1. Decouple Time Input from Sampling Intensity

**Problem:** MLP sees raw normalized time `t_i`, which correlates artificially with sampling density.

**Solution:** MLP reads basis function vectors evaluated at `t_i`, not `t_i` itself.

#### 1a. Fixed `time_dim = 11`

| Slot | Dims | Content |
|------|------|---------|
| trend | 2 | level + slope |
| seasonal | 4 | low-freq sin/cos + high-freq sin/cos |
| spike | 2 | amplitude + decay |
| gate indicators | 3 | [trend_active, seasonal_active, spike_active] |

- Component inactive → corresponding slot filled with 0, gate bit = 0.
- All three gates = 0 → MLP receives all-zeros and learns "this table has no temporal structure."
- `time_dim = 11` is a **compile-time constant**; MLP input dimensionality is stable across all tables.

#### 1b. New method: `TemporalVocab.evaluate_basis(t)`

```python
def evaluate_basis(self, t: torch.Tensor) -> torch.Tensor:
    """
    Args:
        t: (N,) tensor, time values in [0, T]
    Returns:
        (N, 8) tensor — [trend_level, trend_slope, season_low_sin,
         season_low_cos, season_high_sin, season_high_cos,
         spike_amplitude, spike_decay]
        Plus (N, 3) gate tensor appended by caller.
    """
```

Each active component evaluates its basis at t. Inactive components output zeros.

#### 1c. Rename `_sample_time_embedding` → `_prepare_time_features`

Returns `(N, 11)` tensor: basis values (8) + gate indicators (3).

#### 1d. Remove Fourier encoding

Basis vector already carries temporal structure; Fourier positional encoding of raw `t` is no longer needed. Remove `time_embed_mode` config parameter.

#### 1e. Noise component stays in sampler only

`TemporalVocab` noise component (white/colored/OU/etc.) continues to perturb `lambda(t)` for sampling density randomness. It is NOT passed to the MLP. This prevents the MLP from latching onto local random jitter as a pseudo-signal.

---

### 2. Multi-Table Temporal Constraints

**Problem:** Child row timestamps can precede parent row timestamps.

**Solution:** Child row `t_min_j = max(timestamps of all parent rows referenced by this child row's FKs)`.

#### 2a. Window constraint (hard)

```
For child row j with FK mapping fk_ids[j] = [p1_idx, p2_idx, ...]:
  T_parents = { parent_table_k[pk_idx].timestamp | only if parent_table_k has timestamp column }
  t_min_j = max(T_parents)  if T_parents non-empty
          = fallback(topo_position)  if T_parents empty
  t_j sampled from [t_min_j, T]
```

#### 2b. Fallback when no parent has timestamp

Classify table by DAG topology position:

| Position | Condition | t_min distribution |
|----------|-----------|-------------------|
| Source | `num_parents == 0` | Uniform(0, 0.15 * T) |
| Intermediate | `num_parents > 0 and num_children > 0` | Uniform(0.15*T, 0.45*T) |
| Leaf | `num_parents > 0 and num_children == 0` | Uniform(0.45*T, 0.85*T) |

Intermediate and leaf nodes default to later time windows, matching their semantic position in the DAG. Values are configurable.

#### 2c. Lifecycle decay (soft, optional)

Discrete gamma tiers, per-table uniformly chosen:

```python
GAMMA_TIERS = [0.0, 0.5, 1.5, 3.0]
```

Decay uses normalized relative time within window:

```python
tau_j = (t - t_min_j) / (T - t_min_j)   # in [0, 1]
lambda_j(t) = lambda(t) * exp(-gamma * tau_j) * I[t >= t_min_j]
```

Normalization ensures decay strength is invariant to absolute window size.

#### 2d. Sampling implementation

`TemporalVocab.sample_time()` accepts `t_min: torch.Tensor` of shape `(N,)`:

```
For each row j:
  mask_j[t] = 1 if grid_point[t] >= t_min_j else 0
  lambda_j(t) = lambda(t) * mask_j[t] * decay_j[t]
  renormalize: lambda_j /= sum(lambda_j)
  t_j ~ Categorical(lambda_j)
```

Vectorized via `(N, K)` mask matrix + batched multinomial.

---

### 3. Intra-Group Timestamp Arrangement

**Problem:** After HSBM, each parent row p has N_p child rows. Their timestamps should have meaningful intra-group structure.

**Solution:** Independent sampling within window + probabilistic per-group sort.

#### 3a. Independent sampling per row

Each child row j independently samples `t_j` from `lambda_j(t)` in `[t_min_j, T]`. Section 2's window constraints already provide inter-group differentiation (different t_min per group).

#### 3b. Probabilistic per-FK-group sort

Per table, sample `p_sort ~ Uniform(0.3, 0.8)` (configurable). For each FK group:

```python
if random() < p_sort:
    group_timestamps.sort(ascending)
else:
    # keep sampled order as-is
```

- High `p_sort` → OLTP-like tables (rows naturally ordered by insertion time)
- Low `p_sort` → analytical / dimension tables (row order has no temporal meaning)
- Training time: model cannot rely on "row order = time order" as a universal invariant.
- Sort is applied **after** sampling, so the marginal distribution of timestamps is unchanged.

---

### Configuration Changes

**New per-table HP (sampled):**

| Parameter | Distribution | Purpose |
|-----------|-------------|---------|
| `trend_active` | Bernoulli(0.8) | Whether trend basis is used |
| `seasonal_active` | Bernoulli(0.6) | Whether seasonal basis is used |
| `spike_active` | Bernoulli(0.3) | Whether spike basis is used |
| `gamma_tier` | Categorical([0.0, 0.5, 1.5, 3.0]) | Lifecycle decay strength |
| `p_sort` | Uniform(0.3, 0.8) | Intra-group sort probability |

**Removed parameters:**
- `time_embed_mode` (no longer needed — Fourier encoding replaced by basis vector)

**Unchanged:**
- `timestamp.prob` (still controls whether table has timestamp column)
- `time_dim` (now fixed constant 11)

---

### Files Affected

| File | Changes | Est. lines |
|------|---------|-----------|
| `temporal_vocab.py` | Add `evaluate_basis()`, expose component dimensions, per-component eval methods | +80 |
| `mlp_scm.py` | Rename `_sample_time_embedding` → `_prepare_time_features`, new basis-vector logic, t_min window, removal of Fourier encoding | ~100 changed |
| `table_generation.py` | Pass `t_min` through FK chain, fallback based on DAG topology, gate indicator construction | +40 |
| `prior_config.py` | Remove `time_embed_mode`, add new HP params (trend_active, seasonal_active, spike_active, gamma_tier, p_sort) | ~20 changed |
| `dag_to_rdb_generator.py` | Remove `time_embed_mode` from timestamp_config, pass DAG topology info for fallback | ~10 changed |
| `hsbm.py` | No changes (FK allocation unchanged) | 0 |

---

### Rollout Strategy

**Phase 1:** Problem 1 — basis vector input to MLP (lowest risk, highest gain)
- `TemporalVocab.evaluate_basis()`
- `MLPSCM._prepare_time_features()` rewrite
- Remove Fourier encoding
- Fixed `time_dim = 11` with gate indicators

**Phase 2:** Problem 2 — cross-table temporal constraints
- `t_min` propagation via FK chain
- DAG topology fallback
- Lifecycle decay with discrete gamma tiers

**Phase 3:** Problem 3 — intra-group sort
- Probabilistic per-FK-group sort
- `p_sort` HP sampling

Each phase is independently testable. Phase 2 depends on Phase 1 (needs the new `sample_time(t_min=...)` interface). Phase 3 is standalone.
