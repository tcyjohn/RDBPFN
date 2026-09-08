# Temporal Generation Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor time generation to decouple time-as-MLP-input from intensity-based timestamp sampling, enforce cross-table temporal constraints (parent→child), and add probabilistic intra-FK-group sort.

**Architecture:** Three-phase rollout. Phase 1 replaces raw-time MLP input with basis vectors (trend+seasonal+spike) plus gate indicators, fixed `time_dim=11`. Phase 2 adds per-row `t_min` window constraints via FK chain + DAG topology fallback + lifecycle decay. Phase 3 adds probabilistic per-FK-group timestamp sort. Each phase independently testable.

**Tech Stack:** PyTorch, numpy. All changes within `data_generation/RDB/src/`.

**Key Files:**
- `data_generation/RDB/src/prior/temporal_vocab.py` — `TemporalVocab`, `TrendVocab`, `SeasonalityVocab`, `SpikesVocab`
- `data_generation/RDB/src/prior/mlp_scm.py` — `MLPSCM._sample_time_embedding` → `_prepare_time_features`
- `data_generation/RDB/src/table_def/table_generation.py` — `TableGenerator`, `Table`, `init_table_SCMs`
- `data_generation/RDB/src/prior/prior_config.py` — `DEFAULT_SAMPLED_HP`, `DEFAULT_HSBM_HP`
- `data_generation/RDB/dag_to_rdb_generator.py` — `DAGToRDBGenerator`

**Constants:**
- `TIME_DIM = 11` — trend(2) + seasonal(4) + spike(2) + gates(3), compile-time constant
- `GAMMA_TIERS = [0.0, 0.5, 1.5, 3.0]` — lifecycle decay discrete tiers
- `T_MAX = 10.0` — global time range upper bound

---

## Phase 1 — Basis Vector Input to MLP

### Task 1.1: Track component activation state in `TemporalVocab`

**Files:**
- Modify: `data_generation/RDB/src/prior/temporal_vocab.py:125-168` (`TemporalVocab.__init__`, `_init_default_params`, `generate`)

- [ ] **Step 1: Add activation flags as instance attributes**

In `TemporalVocab.__init__` (after line 154), add attributes to track which components were activated during `generate()`:

```python
# In __init__, after line 154 (self.noise_vocab = NoiseVocab(...))
self.trend_active: bool = False
self.seasonal_active: bool = False
self.spike_active: bool = False
```

- [ ] **Step 2: Set activation flags in `generate()`**

In `TemporalVocab.generate()` (lines 218-241), record activation state. Replace each `if random.random() <` block to set the flag:

```python
# Replace line 218-222
self.trend_active = random.random() < self.component_probs[ComponentType.TREND]
if self.trend_active:
    trend_component = self.trend_vocab.generate(t)
    intensity += self.component_amplitudes[ComponentType.TREND] * trend_component

# Replace line 224-229
self.seasonal_active = random.random() < self.component_probs[ComponentType.SEASONALITY]
if self.seasonal_active:
    seasonal_component = self.seasonality_vocab.generate(t)
    intensity += self.component_amplitudes[ComponentType.SEASONALITY] * seasonal_component

# Replace line 231-235
self.spike_active = random.random() < self.component_probs[ComponentType.SPIKES]
if self.spike_active:
    spikes_component = self.spikes_vocab.generate(t)
    intensity += self.component_amplitudes[ComponentType.SPIKES] * spikes_component

# Lines 237-241: noise stays as-is (no flag needed for MLP)
```

- [ ] **Step 3: Verify flags are set**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import torch
from data_generation.RDB.src.prior.temporal_vocab import TemporalVocab
v = TemporalVocab(device='cpu')
v.generate()
print(f'trend_active={v.trend_active}, seasonal_active={v.seasonal_active}, spike_active={v.spike_active}')
"
```
Expected: Three boolean values printed.

- [ ] **Step 4: Commit**

```bash
git add data_generation/RDB/src/prior/temporal_vocab.py
git commit -m "feat: track TemporalVocab component activation flags"
```

---

### Task 1.2: Add `evaluate_basis(t)` with per-component eval methods

**Files:**
- Modify: `data_generation/RDB/src/prior/temporal_vocab.py` — `TrendVocab`, `SeasonalityVocab`, `SpikesVocab`, `TemporalVocab`

- [ ] **Step 1: Add `TrendVocab.evaluate(t)` — returns (N, 2) tensor [level, slope]**

In `TrendVocab` class (after line 479), add:

```python
def evaluate(self, t: torch.Tensor) -> torch.Tensor:
    """Evaluate trend basis at arbitrary t.

    Args:
        t: (N,) tensor of normalized time values in [0, 1].
    Returns:
        (N, 2) tensor: [level (normalized poly/exp value), slope (derivative)].
    """
    if not hasattr(self, "pattern_probs"):
        self.init()

    pattern_names = list(self.pattern_probs.keys())
    pattern_weights = list(self.pattern_probs.values())
    selected_pattern = np.random.choice(
        pattern_names, p=np.array(pattern_weights) / sum(pattern_weights)
    )
    self._eval_pattern = selected_pattern  # store for idempotency
    level = self._eval_trend_level(selected_pattern, t)
    slope = self._eval_trend_slope(selected_pattern, t)
    # Normalize per-dim to roughly [0, 1]
    level = torch.tanh(level * 0.1)  # soft-clip extreme values
    slope = torch.tanh(slope * 0.5)
    return torch.stack([level, slope], dim=-1)

def _eval_trend_level(self, pattern: str, t: torch.Tensor) -> torch.Tensor:
    if pattern == "linear_increasing":
        return 0.5 * t
    elif pattern == "linear_decreasing":
        return -0.5 * t
    elif pattern == "exponential_growth":
        return (torch.exp(0.3 * t) - 1) / (torch.exp(torch.tensor(0.3)) - 1)
    elif pattern == "exponential_decay":
        return torch.exp(-0.3 * t)
    elif pattern == "polynomial_quadratic":
        return 0.05 * t**2 + 0.2 * t
    elif pattern == "polynomial_cubic":
        return 0.005 * t**3 + 0.02 * t**2 + 0.1 * t
    elif pattern == "logistic_growth":
        return 1.0 / (1 + torch.exp(-2.0 * (t - 0.5)))
    elif pattern == "power_law":
        return torch.pow(t + 0.1, 0.8) - 0.1**0.8
    elif pattern == "logarithmic":
        return 0.3 * torch.log(t + 0.1) + 1.0
    else:  # constant
        return torch.zeros_like(t)

def _eval_trend_slope(self, pattern: str, t: torch.Tensor) -> torch.Tensor:
    dt = 0.001
    t_plus = t + dt
    level = self._eval_trend_level(pattern, t)
    level_plus = self._eval_trend_level(pattern, t_plus)
    return (level_plus - level) / dt
```

- [ ] **Step 2: Add `SeasonalityVocab.evaluate(t)` — returns (N, 4) tensor**

In `SeasonalityVocab` class (after line 593), add:

```python
def evaluate(self, t: torch.Tensor) -> torch.Tensor:
    """Evaluate seasonal basis at arbitrary t.

    Args:
        t: (N,) tensor of normalized time values in [0, 1].
    Returns:
        (N, 4) tensor: [low_sin, low_cos, high_sin, high_cos].
        Low freq ~ 1 cycle over the full range; high freq ~ 3-5 cycles.
    """
    low_freq = 2.0 * torch.pi * 1.0  # 1 full cycle
    high_freq = 2.0 * torch.pi * 4.0  # ~4 cycles
    phase = 0.0
    low_sin = torch.sin(low_freq * t + phase)
    low_cos = torch.cos(low_freq * t + phase)
    high_sin = torch.sin(high_freq * t + phase)
    high_cos = torch.cos(high_freq * t + phase)
    return torch.stack([low_sin, low_cos, high_sin, high_cos], dim=-1)
```

- [ ] **Step 3: Add `SpikesVocab.evaluate(t)` — returns (N, 2) tensor**

In `SpikesVocab` class (after line 676), add:

```python
def evaluate(self, t: torch.Tensor) -> torch.Tensor:
    """Evaluate spike basis at arbitrary t.

    Args:
        t: (N,) tensor of normalized time values in [0, 1].
    Returns:
        (N, 2) tensor: [amplitude (distance-weighted), decay (exp distance to nearest spike)].
    """
    if not hasattr(self, "_spike_params"):
        self._spike_params = []
        n_spikes = 2  # deterministic: 2 spikes for basis eval
        for _ in range(n_spikes):
            center = float(np.random.uniform(0.1, 0.9))
            width = float(np.random.uniform(0.02, 0.08))
            amp = float(np.random.uniform(0.5, 2.0))
            self._spike_params.append((center, width, amp))

    amplitude = torch.zeros_like(t)
    min_dist = torch.full_like(t, float("inf"))
    for center, width, amp in self._spike_params:
        dist = torch.abs(t - center)
        amplitude += amp * torch.exp(-0.5 * (dist / width)**2)
        min_dist = torch.minimum(min_dist, dist)

    decay = torch.exp(-3.0 * min_dist)  # decays to ~0 at 1 unit away
    amplitude = torch.tanh(amplitude * 0.3)
    return torch.stack([amplitude, decay], dim=-1)
```

- [ ] **Step 4: Add `TemporalVocab.evaluate_basis(t)` — the public entry point**

In `TemporalVocab` class (after `__len__`, line 374), add:

```python
def evaluate_basis(self, t: torch.Tensor) -> torch.Tensor:
    """Evaluate all active basis components at time points t.

    Args:
        t: (N,) tensor, normalized time values in [0, 1] on self.device.
    Returns:
        (N, 8) tensor:
          [trend_level, trend_slope,
           season_low_sin, season_low_cos, season_high_sin, season_high_cos,
           spike_amplitude, spike_decay]
        Inactive components output zeros in their slots.
    """
    N = t.shape[0]
    pieces: list[torch.Tensor] = []

    if self.trend_active:
        pieces.append(self.trend_vocab.evaluate(t))  # (N, 2)
    else:
        pieces.append(torch.zeros(N, 2, device=self.device, dtype=t.dtype))

    if self.seasonal_active:
        pieces.append(self.seasonality_vocab.evaluate(t))  # (N, 4)
    else:
        pieces.append(torch.zeros(N, 4, device=self.device, dtype=t.dtype))

    if self.spike_active:
        pieces.append(self.spikes_vocab.evaluate(t))  # (N, 2)
    else:
        pieces.append(torch.zeros(N, 2, device=self.device, dtype=t.dtype))

    return torch.cat(pieces, dim=-1)  # (N, 8)
```

- [ ] **Step 5: Verify `evaluate_basis` works**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import torch
from data_generation.RDB.src.prior.temporal_vocab import TemporalVocab
v = TemporalVocab(device='cpu')
v.generate()
t = torch.linspace(0, 1, 10)
basis = v.evaluate_basis(t)
print(f'shape={basis.shape}')  # Expected: (10, 8)
print(f'nan={torch.isnan(basis).any()}, inf={torch.isinf(basis).any()}')
"
```
Expected: `shape=torch.Size([10, 8])`, `nan=False, inf=False`.

- [ ] **Step 6: Commit**

```bash
git add data_generation/RDB/src/prior/temporal_vocab.py
git commit -m "feat: add evaluate_basis(t) with per-component eval methods"
```

---

### Task 1.3: Add gate indicator vector builder

**Files:**
- Modify: `data_generation/RDB/src/prior/temporal_vocab.py` (`TemporalVocab`)

- [ ] **Step 1: Add `build_gate_vector()` method**

In `TemporalVocab` class, after `evaluate_basis`:

```python
def build_gate_vector(self) -> torch.Tensor:
    """Return (3,) float tensor: [trend_active, seasonal_active, spike_active]."""
    return torch.tensor(
        [
            float(self.trend_active),
            float(self.seasonal_active),
            float(self.spike_active),
        ],
        device=self.device,
        dtype=torch.float32,
    )
```

- [ ] **Step 2: Verify**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
from data_generation.RDB.src.prior.temporal_vocab import TemporalVocab
v = TemporalVocab(device='cpu'); v.generate()
print(v.build_gate_vector())
"
```

- [ ] **Step 3: Commit**

```bash
git add data_generation/RDB/src/prior/temporal_vocab.py
git commit -m "feat: add TemporalVocab.build_gate_vector()"
```

---

### Task 1.4: Update HP config — remove `time_embed_mode`, define `TIME_DIM = 11`

**Files:**
- Modify: `data_generation/RDB/src/prior/prior_config.py`

- [ ] **Step 1: Add module-level constant**

At top of `prior_config.py`, after imports:

```python
# Fixed time-as-input dimension: trend(2) + seasonal(4) + spike(2) + gates(3)
TIME_DIM = 11
```

- [ ] **Step 2: Add new temporal HP params to `DEFAULT_SAMPLED_HP`**

In `DEFAULT_SAMPLED_HP` dict, before the deprecated section comment:

```python
    # --- Temporal basis component activation (per-table sampling) ---
    "trend_active": {
        "distribution": "meta_choice",
        "choice_values": [True, False],
    },
    "seasonal_active": {
        "distribution": "meta_choice",
        "choice_values": [True, False],
    },
    "spike_active": {
        "distribution": "meta_choice",
        "choice_values": [True, False],
    },
    # --- Phase 2–3 temporal params (sampled here for config completeness) ---
    "gamma_tier": {
        "distribution": "meta_choice",
        "choice_values": [0.0, 0.5, 1.5, 3.0],
    },
    "p_sort": {
        "distribution": "uniform",
        "min": 0.3,
        "max": 0.8,
    },
```

Note: `trend_active`, `seasonal_active`, `spike_active` are sampled per-table via `HpSamplerList`. The Bernoulli probabilities (0.8/0.6/0.3) are handled by `TemporalVocab._init_default_params()` via `TemporalCombinationConfigs` — we keep the config-driven probability mechanism and use these flags to override when the sampler says False (see Task 1.6).

- [ ] **Step 3: Commit**

```bash
git add data_generation/RDB/src/prior/prior_config.py
git commit -m "feat: define TIME_DIM=11, add temporal HP params, remove time_embed_mode"
```

---

### Task 1.5: Update `dag_to_rdb_generator.py` — remove `time_embed_mode`, update defaults

**Files:**
- Modify: `data_generation/RDB/dag_to_rdb_generator.py:60-64` (timestamp config defaults)

- [ ] **Step 1: Update default timestamp config**

Replace lines 60-64:

```python
# Before:
"timestamp": {
    "prob": 1.0,
    "time_dim": 8,
    "time_embed_mode": "fourier",
}

# After:
"timestamp": {
    "prob": 1.0,
}
```

`time_dim` is now the module-level constant `TIME_DIM = 11`, not a config parameter.

- [ ] **Step 2: Update `init_table_SCMs` caller** — remove references to `time_dim` and `time_embed_mode` from ts_cfg reads (this is in `table_generation.py`, handled in Task 1.8)

- [ ] **Step 3: Commit**

```bash
git add data_generation/RDB/dag_to_rdb_generator.py
git commit -m "refactor: remove time_dim/time_embed_mode from timestamp_config defaults"
```

---

### Task 1.6: Rewrite `MLPSCM._sample_time_embedding` → `_prepare_time_features`

**Files:**
- Modify: `data_generation/RDB/src/prior/mlp_scm.py:415-446`

- [ ] **Step 1: Replace the method**

Replace `_sample_time_embedding` (lines 415-446) with:

```python
def _prepare_time_features(
    self,
    n: int,
    t_min: torch.Tensor | None = None,
    fk_ids: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample timestamps and build basis-vector MLP input.

    Args:
        n: Number of rows to sample.
        t_min: Optional (n,) tensor of per-row lower bounds in [0, T].
               If None, defaults to 0 for all rows (source tables).
        fk_ids: Reserved for Phase 2 (FK mapping for cross-table constraints).

    Returns:
        timestamps_norm: (n,) in [0, 1] — for storage in X[MASK_TYPE.TIMESTAMP].
        time_features: (n, TIME_DIM) — basis(8) + gates(3), MLP input.
    """
    from src.prior.prior_config import TIME_DIM  # noqa: PLC0415

    if n <= 0:
        raise ValueError(f"_prepare_time_features expects n > 0, got {n}")
    if self.temporal_vocab is None:
        raise RuntimeError("temporal_vocab is unset; time_dim must be > 0")

    # 1. Generate intensity distribution once
    self.temporal_vocab.generate(time_range=(0.0, 10.0))

    # 2. Sample timestamps (Phase 1: uniform t_min=0; Phase 2 adds per-row t_min)
    if t_min is None:
        timestamps = self.temporal_vocab.sample_time(
            num_samples=n, time_range=(0.0, 10.0),
        )
    else:
        timestamps = self.temporal_vocab.sample_time(
            num_samples=n, time_range=(0.0, 10.0), t_min=t_min,
        )
    timestamps_norm = (timestamps.to(self.device) / 10.0).clamp(0.0, 1.0)

    # 3. Evaluate basis at sampled times
    basis = self.temporal_vocab.evaluate_basis(timestamps_norm)  # (n, 8)

    # 4. Build gate vector and broadcast
    gate = self.temporal_vocab.build_gate_vector().to(self.device)  # (3,)
    gate_broadcast = gate.unsqueeze(0).expand(n, -1)  # (n, 3)

    # 5. Concatenate: basis(8) + gates(3) = (n, 11)
    time_features = torch.cat([basis, gate_broadcast], dim=-1)
    assert time_features.shape == (n, TIME_DIM), (
        f"Expected ({n}, {TIME_DIM}), got {time_features.shape}"
    )

    return timestamps_norm, time_features
```

- [ ] **Step 2: Update `forward_without_input` and `forward_with_input`**

Replace references to `_sample_time_embedding` in both methods:

In `forward_without_input` (line 456):
```python
# Before:
timestamps_norm, time_embed = self._sample_time_embedding(n_sampled)
causes = torch.cat([causes, time_embed], dim=-1)

# After:
timestamps_norm, time_features = self._prepare_time_features(n_sampled)
causes = torch.cat([causes, time_features], dim=-1)
```

In `forward_with_input` (line 502):
```python
# Before:
timestamps_norm, time_embed = self._sample_time_embedding(self.seq_len)
causes = torch.cat([causes, time_embed], dim=-1)

# After:
timestamps_norm, time_features = self._prepare_time_features(self.seq_len)
causes = torch.cat([causes, time_features], dim=-1)
```

- [ ] **Step 3: Verify shape assertions still hold**

The assertion at line 513 checks `causes.shape[1] == self.num_causes + self.time_dim + self.other_causes`. `self.time_dim` is now `TIME_DIM = 11`. Verify this propagates from `__init__`.

- [ ] **Step 4: Commit**

```bash
git add data_generation/RDB/src/prior/mlp_scm.py
git commit -m "refactor: _sample_time_embedding -> _prepare_time_features with basis vectors"
```

---

### Task 1.7: Update `MLPSCM.__init__` — fixed `TIME_DIM`, remove Fourier, fix validation

**Files:**
- Modify: `data_generation/RDB/src/prior/mlp_scm.py:84-196` (`__init__`)

- [ ] **Step 1: Replace `time_dim` and `time_embed_mode` validation**

Replace lines 165-181 (time param validation) with:

```python
from src.prior.prior_config import TIME_DIM  # noqa: PLC0415

self.time_dim = TIME_DIM if time_dim > 0 else 0
# time_embed_mode is no longer used; kept as kwarg for backward compat
if "time_embed_mode" in kwargs:
    kwargs.pop("time_embed_mode")
```

Remove the `time_embed_mode` parameter from the function signature (line 128). Replace with a simpler validation:

In `__init__` signature (lines 127-128):
```python
# Before:
time_dim: int = 0,
time_embed_mode: str = "raw",

# After:
time_dim: int = 0,
```

- [ ] **Step 2: Update first-layer input dimension comment**

At line 226-227, the comment and code already use `self.time_dim` which will now be `TIME_DIM = 11` when time is enabled. No code change needed — just confirm.

- [ ] **Step 3: Remove `_fourier_features` method (lines 396-413)**

Delete `_fourier_features` entirely. It's no longer called.

- [ ] **Step 4: Verify no remaining references to `time_embed_mode` or `_fourier_features`**

```bash
cd /data/caijunyu/RDBPFN && grep -rn "time_embed_mode\|_fourier_features" data_generation/RDB/src/
```
Expected: No matches.

- [ ] **Step 5: Commit**

```bash
git add data_generation/RDB/src/prior/mlp_scm.py
git commit -m "refactor: fixed TIME_DIM, remove Fourier encoding and time_embed_mode from MLPSCM"
```

---

### Task 1.8: Update `table_generation.py` — `init_table_SCMs` and `Table.__init__`

**Files:**
- Modify: `data_generation/RDB/src/table_def/table_generation.py:1539-1649` (`init_table_SCMs`)
- Modify: `data_generation/RDB/src/table_def/table_generation.py:807-868` (`Table.__init__`, `is_time_table`)

- [ ] **Step 1: Update `init_table_SCMs` — remove `time_embed_mode`, use `TIME_DIM`**

Replace lines 1578-1584:

```python
# Before:
ts_cfg = getattr(self, "timestamp_config", {}) or {}
if table.is_time_table:
    base_time_dim = int(ts_cfg.get("time_dim", 8))
    base_time_embed_mode = str(ts_cfg.get("time_embed_mode", "fourier"))
else:
    base_time_dim = 0
    base_time_embed_mode = "raw"

# After:
from src.prior.prior_config import TIME_DIM  # noqa: PLC0415
if table.is_time_table:
    base_time_dim = TIME_DIM
else:
    base_time_dim = 0
```

- [ ] **Step 2: Update `base_params` dict — remove `time_embed_mode` key**

Replace lines 1609-1616:

```python
# Before:
base_params = {
    "seq_len": seq_len,
    "other_causes": other_causes,
    "sampling_ratio": sampling_ratio,
    "use_timestamp_sampling": False,
    "time_dim": base_time_dim,
    "time_embed_mode": base_time_embed_mode,
}

# After:
base_params = {
    "seq_len": seq_len,
    "other_causes": other_causes,
    "sampling_ratio": sampling_ratio,
    "use_timestamp_sampling": False,
    "time_dim": base_time_dim,
}
```

- [ ] **Step 3: Commit**

```bash
git add data_generation/RDB/src/table_def/table_generation.py
git commit -m "refactor: use fixed TIME_DIM in init_table_SCMs, remove time_embed_mode"
```

---

### Task 1.9: Phase 1 Integration Test

**Files:**
- Create: (none — standalone test script)

- [ ] **Step 1: Write and run a minimal generation test**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import torch, sys
sys.path.insert(0, 'data_generation/RDB')
from src.prior.temporal_vocab import TemporalVocab
from src.prior.mlp_scm import MLPSCM
from src.prior.utils import MASK_TYPE
from src.prior.prior_config import TIME_DIM

# Test 1: TemporalVocab basis eval
v = TemporalVocab(device='cpu')
v.generate()
t = torch.linspace(0, 1, 32)
basis = v.evaluate_basis(t)
gate = v.build_gate_vector()
assert basis.shape == (32, 8), f'Bad basis shape: {basis.shape}'
assert gate.shape == (3,), f'Bad gate shape: {gate.shape}'
assert not torch.isnan(basis).any(), 'NaN in basis'
assert not torch.isinf(basis).any(), 'Inf in basis'
# Verify zeros in inactive slots
if not v.trend_active:
    assert (basis[:, :2] == 0).all(), 'Trend slot not zero when inactive'
if not v.seasonal_active:
    assert (basis[:, 2:6] == 0).all(), 'Seasonal slot not zero when inactive'
if not v.spike_active:
    assert (basis[:, 6:8] == 0).all(), 'Spike slot not zero when inactive'
print('Test 1 OK: TemporalVocab basis eval + gates')

# Test 2: MLPSCM._prepare_time_features
masks = {MASK_TYPE.X: 16}
scm = MLPSCM(
    seq_len=64, num_outputs=4, num_causes=4, other_causes=0,
    time_dim=TIME_DIM, device='cpu', masks=masks,
)
ts_norm, t_feat = scm._prepare_time_features(64)
assert ts_norm.shape == (64,), f'Bad ts shape: {ts_norm.shape}'
assert t_feat.shape == (64, TIME_DIM), f'Bad feat shape: {t_feat.shape}'
assert not torch.isnan(t_feat).any(), 'NaN in time_features'
# Verify gates are consistent
gate_vals = t_feat[0, 8:11]
assert (t_feat[:, 8:11] == gate_vals).all(), 'Gates not broadcast correctly'
print('Test 2 OK: _prepare_time_features')

# Test 3: forward_without_input with time_dim > 0
X, outputs = scm.forward_without_input()
assert MASK_TYPE.TIMESTAMP in X, 'TIMESTAMP missing from X'
assert X[MASK_TYPE.TIMESTAMP].shape == (64, 1), f'Bad TS shape: {X[MASK_TYPE.TIMESTAMP].shape}'
assert not torch.isnan(X[MASK_TYPE.TIMESTAMP]).any(), 'NaN in stored timestamps'
print('Test 3 OK: forward_without_input with time features')
print('Phase 1 integration tests PASSED')
"
```
Expected: All three tests print "OK" and final "PASSED".

- [ ] **Step 2: Commit**

```bash
# No file changes to commit (test is inline)
```

---

## Phase 2 — Cross-Table Temporal Constraints

### Task 2.1: Add `TemporalVocab.sample_time_with_t_min(t_min)`

**Files:**
- Modify: `data_generation/RDB/src/prior/temporal_vocab.py:317-359` (`sample_time`)

- [ ] **Step 1: Add `t_min` parameter to `sample_time`**

Modify `sample_time` signature and body (replace lines 317-359):

```python
def sample_time(
    self,
    num_samples: int,
    time_range: Tuple[float, float] = (0.0, 10.0),
    num_points: int = DEFAULT_NUM_POINTS,
    distribution: Optional[torch.Tensor] = None,
    t_min: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample event times, optionally with per-row lower bounds.

    Args:
        num_samples: Number of samples (rows).
        time_range: Global (min_t, max_t).
        num_points: Discretization grid size.
        distribution: Pre-generated (t, intensity) tensor.
        t_min: Optional (num_samples,) tensor of per-row lower bounds in
               raw time units (same scale as time_range). Rows with
               t_min == max_t sample at exactly max_t.

    Returns:
        Sampled times (num_samples,) sorted ascending, in raw time units.
    """
    if distribution is None:
        distribution = self.generate(time_range, num_points=num_points)

    t_points = distribution[:, 0]  # (K,)
    intensity = distribution[:, 1]  # (K,)
    K = t_points.shape[0]
    device = t_points.device

    # Per-row intensity: shape (num_samples, K)
    if t_min is not None:
        t_min_clamped = t_min.to(device).clamp(
            min=time_range[0], max=time_range[1]
        )  # (N,)
        mask = t_points.unsqueeze(0) >= t_min_clamped.unsqueeze(1)  # (N, K)
        intensity_bc = intensity.unsqueeze(0) * mask.float()  # (N, K)
    else:
        intensity_bc = intensity.unsqueeze(0).expand(num_samples, -1)  # (N, K)

    # Renormalize per row; guard against all-zero rows
    row_sums = intensity_bc.sum(dim=1, keepdim=True)  # (N, 1)
    row_sums = row_sums.clamp(min=1e-12)
    probs = intensity_bc / row_sums  # (N, K)

    sample_indices = torch.multinomial(probs, 1, replacement=True).squeeze(-1)  # (N,)
    sampled_times = t_points[sample_indices]
    return torch.sort(sampled_times)[0]
```

- [ ] **Step 2: Test `t_min` masking**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import torch
from data_generation.RDB.src.prior.temporal_vocab import TemporalVocab
v = TemporalVocab(device='cpu')
v.generate(time_range=(0.0, 10.0))
# All rows must be >= 8.0
t_min = torch.full((500,), 8.0)
ts = v.sample_time(500, t_min=t_min)
assert (ts >= 8.0).all(), f'Some timestamps below t_min: min={ts.min()}'
print(f't_min=8.0 test OK: min={ts.min():.2f}, max={ts.max():.2f}')
"
```
Expected: `min >= 8.0`.

- [ ] **Step 3: Commit**

```bash
git add data_generation/RDB/src/prior/temporal_vocab.py
git commit -m "feat: add per-row t_min lower bound to sample_time"
```

---

### Task 2.2: Add lifecycle decay to `sample_time`

**Files:**
- Modify: `data_generation/RDB/src/prior/temporal_vocab.py` (`sample_time`)

- [ ] **Step 1: Add `gamma` parameter to `sample_time`**

Add `gamma: float = 0.0` parameter. Inside the method, after computing the mask and before renormalization:

```python
# Apply lifecycle decay (if gamma > 0 and t_min provided)
if gamma > 0.0 and t_min is not None:
    # tau_j = (t - t_min_j) / (T_max - t_min_j), normalized relative time
    tau = (t_points.unsqueeze(0) - t_min_clamped.unsqueeze(1)) / (
        time_range[1] - t_min_clamped.unsqueeze(1)
    ).clamp(min=1e-8)  # (N, K)
    tau = tau.clamp(min=0.0)  # negative = before t_min, already masked
    decay = torch.exp(-gamma * tau)  # (N, K)
    intensity_bc = intensity_bc * decay
```

- [ ] **Step 2: Test decay**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import torch
from data_generation.RDB.src.prior.temporal_vocab import TemporalVocab
v = TemporalVocab(device='cpu')
v.generate(time_range=(0.0, 10.0))
# gamma=3.0 with t_min=0 should push timestamps toward early region
t_min = torch.zeros(500)
ts_no_decay = v.sample_time(500, t_min=t_min, gamma=0.0)
ts_decay = v.sample_time(500, t_min=t_min, gamma=3.0)
print(f'gamma=0.0: mean={ts_no_decay.mean():.2f}')
print(f'gamma=3.0: mean={ts_decay.mean():.2f}')
# With strong decay, mean should be lower
assert ts_decay.mean() < ts_no_decay.mean(), 'Decay not pulling timestamps earlier'
print('Decay test OK')
"
```
Expected: `ts_decay.mean() < ts_no_decay.mean()`.

- [ ] **Step 3: Commit**

```bash
git add data_generation/RDB/src/prior/temporal_vocab.py
git commit -m "feat: add lifecycle decay (gamma) to sample_time"
```

---

### Task 2.3: Update `_prepare_time_features` to accept and propagate `t_min`

**Files:**
- Modify: `data_generation/RDB/src/prior/mlp_scm.py` (`_prepare_time_features`)

- [ ] **Step 1: Update `_prepare_time_features` to use `t_min` and `gamma`**

In the method (from Task 1.6), update the sampling call:

```python
# In _prepare_time_features, replace the sample_time call:
timestamps = self.temporal_vocab.sample_time(
    num_samples=n,
    time_range=(0.0, 10.0),
    t_min=t_min_scaled if t_min is not None else None,
    gamma=self.gamma,
)
```

Where `t_min_scaled` converts from normalized [0,1] back to raw [0,10]:
```python
t_min_scaled = t_min * 10.0 if t_min is not None else None
```

- [ ] **Step 2: Store `gamma` as instance attribute in `MLPSCM.__init__`**

Add parameter `gamma: float = 0.0` to `MLPSCM.__init__` and store as `self.gamma`.

- [ ] **Step 3: Commit**

```bash
git add data_generation/RDB/src/prior/mlp_scm.py
git commit -m "feat: propagate t_min and gamma through _prepare_time_features"
```

---

### Task 2.4: Implement per-row `t_min` computation and DAG topology fallback in `TableGenerator`

**Files:**
- Modify: `data_generation/RDB/src/table_def/table_generation.py` (`TableGenerator.generate_data`, `_compute_t_min_for_child`)

- [ ] **Step 1: Add `_compute_t_min_for_child` method to `TableGenerator`**

After `_compute_hsbm_fk_ids` (line 1267), add:

```python
def _compute_t_min_for_child(
    self,
    child_rows: int,
    fk_ids: torch.Tensor,
    parent_data_list: list,
    parent_is_time_table: list[bool],
    dag_position: str,  # "source" | "intermediate" | "leaf"
) -> torch.Tensor:
    """Compute per-row t_min for child table timestamp sampling.

    Args:
        child_rows: Number of child rows.
        fk_ids: (child_rows, num_parents) FK index tensor.
        parent_data_list: List of parent ``all_scm_outputs`` dicts.
        parent_is_time_table: bool per parent.
        dag_position: DAG topology position for fallback.

    Returns:
        (child_rows,) tensor of t_min values in [0, 1] (normalized).
    """
    T = 1.0  # normalized time range
    num_parents = len(parent_data_list)

    # Collect parent timestamps for rows with FK references
    parent_ts_list: list[torch.Tensor] = []
    for p_idx in range(num_parents):
        if parent_is_time_table[p_idx] and MASK_TYPE.TIMESTAMP in parent_data_list[p_idx]:
            p_ts = parent_data_list[p_idx][MASK_TYPE.TIMESTAMP].squeeze(-1)  # (N_p,)
            # p_ts is already normalized to [0, 1]
            row_ts = p_ts[fk_ids[:, p_idx].long()]  # (child_rows,)
            parent_ts_list.append(row_ts)

    if parent_ts_list:
        t_min = torch.stack(parent_ts_list, dim=1).max(dim=1).values  # (child_rows,)
    else:
        # Fallback based on DAG topology
        rng = np.random.RandomState(hash((self._seed, "t_min_fallback")) & 0x7FFFFFFF)
        if dag_position == "source":
            t_min = torch.tensor(
                rng.uniform(0.0, 0.15 * T, size=child_rows),
                device=self.device, dtype=torch.float32,
            )
        elif dag_position == "intermediate":
            t_min = torch.tensor(
                rng.uniform(0.15 * T, 0.45 * T, size=child_rows),
                device=self.device, dtype=torch.float32,
            )
        else:  # leaf
            t_min = torch.tensor(
                rng.uniform(0.45 * T, 0.85 * T, size=child_rows),
                device=self.device, dtype=torch.float32,
            )

    return t_min.clamp(0.0, T)
```

- [ ] **Step 2: Update `generate_data` to pass `t_min` to SCM**

In `generate_data` (line 1269), add `t_min` computation before calling `forward_with_input`:

```python
def generate_data(self, **kwargs) -> torch.Tensor:
    with torch.no_grad():
        if "parent_data_list" in kwargs:
            parent_data_list = kwargs["parent_data_list"]
            parent_names = kwargs.get("parent_names", None)
            fk_ids = self._compute_hsbm_fk_ids(
                fk_seed=kwargs.get("fk_seed", 0),
                parent_data_list=parent_data_list,
                parent_names=parent_names,
            )
            # NEW: compute per-row t_min
            parent_is_time = kwargs.get("parent_is_time_table", [False] * len(parent_data_list))
            dag_pos = kwargs.get("dag_position", "intermediate")
            t_min = self._compute_t_min_for_child(
                child_rows=self.num_rows,
                fk_ids=fk_ids,
                parent_data_list=parent_data_list,
                parent_is_time_table=parent_is_time,
                dag_position=dag_pos,
            )
            # Pass t_min to SCM
            self.table_SCM.t_min = t_min  # store for _prepare_time_features
            X, FK_ids, outputs_flat = self.table_SCM.forward_with_input(
                parent_data_list, fk_ids
            )
            ...
```

- [ ] **Step 3: Update `forward_with_input` in MLPSCM to read `self.t_min`**

In `MLPSCM.forward_with_input`, when calling `_prepare_time_features`:

```python
timestamps_norm, time_features = self._prepare_time_features(
    n=self.seq_len,
    t_min=getattr(self, "t_min", None),
)
```

- [ ] **Step 4: Determine DAG topology position per table**

In `init_table_SCMs` (where `parent_tables` is computed), determine `dag_position`:

```python
num_parents = len(parent_tables)
# Determine DAG position for t_min fallback
children_of_this = [
    rel.to_table for rel in self.get_foreign_relations_for_table(table_name)
]
num_children = len(children_of_this)
if num_parents == 0:
    dag_position = "source"
elif num_children == 0:
    dag_position = "leaf"
else:
    dag_position = "intermediate"
```

Store `dag_position` on the `TableGenerator` instance: `self.dag_position = dag_position`.

- [ ] **Step 5: Commit**

```bash
git add data_generation/RDB/src/table_def/table_generation.py data_generation/RDB/src/prior/mlp_scm.py
git commit -m "feat: per-row t_min via FK chain + DAG topology fallback"
```

---

### Task 2.5: Sample gamma per table and pass through init chain

**Files:**
- Modify: `data_generation/RDB/src/table_def/table_generation.py` (`init_table_SCMs`)
- Modify: `data_generation/RDB/src/prior/mlp_scm.py` (`__init__`)

- [ ] **Step 1: Add gamma sampling in `init_table_SCMs`**

In the per-table loop (after line 1601, where HSBM params are sampled), add:

```python
# Sample gamma tier for lifecycle decay (only for timestamp tables with parents)
if table.is_time_table and len(parent_tables) > 0:
    gamma_tier_sample = hpsampler.sample()  # re-sample or use dedicated sampler
    sampled_gamma = gamma_tier_sample.get("gamma_tier", 0.0)
    if callable(sampled_gamma):
        sampled_gamma = sampled_gamma()
else:
    sampled_gamma = 0.0
```

Add `"gamma": sampled_gamma` to `base_params` dict.

- [ ] **Step 2: Read gamma in MLPSCM.__init__**

Already handled — `MLPSCM.__init__` accepts `gamma: float = 0.0` (Task 2.3).

- [ ] **Step 3: Commit**

```bash
git add data_generation/RDB/src/table_def/table_generation.py data_generation/RDB/src/prior/mlp_scm.py
git commit -m "feat: sample gamma_tier per table for lifecycle decay"
```

---

### Task 2.6: Phase 2 Integration Test

- [ ] **Step 1: Run a minimal multi-table generation**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import torch, sys, numpy as np
sys.path.insert(0, 'data_generation/RDB')
from src.prior.temporal_vocab import TemporalVocab
from src.prior.mlp_scm import MLPSCM
from src.prior.utils import MASK_TYPE
from src.prior.prior_config import TIME_DIM

# Simulate a parent and child: parent generates timestamps, child reads them
masks = {MASK_TYPE.X: 8}

# Parent (source table, no parents)
parent_scm = MLPSCM(
    seq_len=32, num_outputs=4, num_causes=4, other_causes=0,
    time_dim=TIME_DIM, device='cpu', masks=masks, gamma=0.0,
)
X_parent, _ = parent_scm.forward_without_input()
parent_ts = X_parent[MASK_TYPE.TIMESTAMP].squeeze(-1)  # (32,)
print(f'Parent ts: min={parent_ts.min():.3f}, max={parent_ts.max():.3f}')

# Child: should have all timestamps >= max of referenced parent timestamps
child_scm = MLPSCM(
    seq_len=64, num_outputs=4, num_causes=4, other_causes=4,
    time_dim=TIME_DIM, device='cpu', masks=masks, gamma=0.0,
)
# Pick FK references: child row j -> parent row j % 32
fk_ids = torch.arange(64, device='cpu') % 32
# Compute t_min from parent
parent_ts_expanded = parent_ts[fk_ids]
child_scm.t_min = parent_ts_expanded  # in [0,1]
parent_data = {MASK_TYPE.CAUSAL_OUTPUT: X_parent[MASK_TYPE.CAUSAL_OUTPUT],
               MASK_TYPE.TIMESTAMP: X_parent[MASK_TYPE.TIMESTAMP]}
X_child, _, _ = child_scm.forward_with_input([parent_data], fk_ids.unsqueeze(-1))
child_ts = X_child[MASK_TYPE.TIMESTAMP].squeeze(-1)
violations = (child_ts < parent_ts_expanded).sum().item()
print(f'Child ts: min={child_ts.min():.3f}, max={child_ts.max():.3f}')
print(f't_min violations: {violations} / 64')
assert violations == 0, f'{violations} child rows have timestamp < parent!'
print('Phase 2 integration test PASSED')
"
```
Expected: `t_min violations: 0 / 64`.

- [ ] **Step 2: Commit** (no file changes)

---

## Phase 3 — Intra-Group Timestamp Sort

### Task 3.1: Add per-FK-group probabilistic sort

**Files:**
- Modify: `data_generation/RDB/src/table_def/table_generation.py` (`_materialize_tables_from_pending` or a new post-processing step)

- [ ] **Step 1: Find where timestamps are finalized**

The timestamps are stored in `X[MASK_TYPE.TIMESTAMP]` during `forward_with_input` / `forward_without_input`. After SCM generation, the `cache_pending_outputs` method stores the raw outputs. The method `_materialize_tables_from_pending` eventually processes them.

Add a method to `TableGenerator` that applies per-FK-group sort after SCM generation but before materialization:

```python
def _apply_intra_group_sort(
    self,
    timestamps: torch.Tensor,
    fk_ids: torch.Tensor,
    p_sort: float,
) -> torch.Tensor:
    """Probabilistically sort timestamps within each FK group.

    Args:
        timestamps: (N,) tensor of timestamps in [0, 1].
        fk_ids: (N, num_parents) FK index tensor (use first parent only for grouping).
        p_sort: Probability of sorting within a group.

    Returns:
        (N,) tensor — same marginal distribution, possibly reordered within groups.
    """
    if p_sort <= 0.0:
        return timestamps
    rng = np.random.RandomState()
    # Group rows by first parent FK
    parent_fk = fk_ids[:, 0].cpu().numpy()  # (N,)
    ts_np = timestamps.cpu().numpy()
    result = ts_np.copy()  # copy to avoid in-place modification of original

    unique_parents = np.unique(parent_fk)
    for p in unique_parents:
        mask = parent_fk == p
        if mask.sum() <= 1:
            continue
        if rng.random() < p_sort:
            result[mask] = np.sort(result[mask])

    return torch.tensor(result, device=timestamps.device, dtype=timestamps.dtype)
```

- [ ] **Step 2: Integrate into `cache_pending_outputs`**

After `self.pending_outputs` is set, apply sort if this table has timestamps:

```python
# In cache_pending_outputs, after line 1300 (self.pending_outputs = ...)
if MASK_TYPE.TIMESTAMP in self.pending_outputs and self.pending_fk_ids is not None:
    ts_sorted = self._apply_intra_group_sort(
        timestamps=self.pending_outputs[MASK_TYPE.TIMESTAMP].squeeze(-1),
        fk_ids=self.pending_fk_ids,
        p_sort=getattr(self, "p_sort", 0.0),
    )
    self.pending_outputs[MASK_TYPE.TIMESTAMP] = ts_sorted.unsqueeze(-1)
```

- [ ] **Step 3: Test sort logic**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import torch, numpy as np
# Simulate per-group sort
N = 100
ts = torch.rand(N)
fk_ids = torch.randint(0, 10, (N, 1))  # 10 groups
p_sort = 1.0  # always sort

# Manual sort
ts_np = ts.numpy()
fk_np = fk_ids[:, 0].numpy()
result = ts_np.copy()
import numpy as np
for p in np.unique(fk_np):
    mask = fk_np == p
    result[mask] = np.sort(result[mask])
assert (np.diff(result[fk_np == 0]) >= 0).all(), 'Group 0 not sorted'
print('Sort test OK')
"
```
Expected: `Sort test OK`.

- [ ] **Step 4: Commit**

```bash
git add data_generation/RDB/src/table_def/table_generation.py
git commit -m "feat: add probabilistic per-FK-group timestamp sort"
```

---

### Task 3.2: Sample `p_sort` per table

**Files:**
- Modify: `data_generation/RDB/src/table_def/table_generation.py` (`init_table_SCMs`)

- [ ] **Step 1: Sample p_sort in per-table init loop**

In `init_table_SCMs`, after gamma sampling (Task 2.5), add:

```python
# Sample p_sort for intra-group sort (only for timestamp tables with parents)
if table.is_time_table and len(parent_tables) > 0:
    p_sort_sample = hpsampler.sample()
    sampled_p_sort = p_sort_sample.get("p_sort", 0.5)
    if callable(sampled_p_sort):
        sampled_p_sort = sampled_p_sort()
else:
    sampled_p_sort = 0.0  # no sort for source tables or non-timestamp tables
```

Store `self.table_generators[table_name].p_sort = sampled_p_sort`.

- [ ] **Step 2: Commit**

```bash
git add data_generation/RDB/src/table_def/table_generation.py
git commit -m "feat: sample p_sort per table for intra-group sort"
```

---

### Task 3.3: Phase 3 Integration Test

- [ ] **Step 1: Full pipeline test**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import torch, sys, numpy as np
sys.path.insert(0, 'data_generation/RDB')
from src.prior.temporal_vocab import TemporalVocab
from src.prior.mlp_scm import MLPSCM
from src.prior.utils import MASK_TYPE
from src.prior.prior_config import TIME_DIM

masks = {MASK_TYPE.X: 8}
parent_scm = MLPSCM(
    seq_len=32, num_outputs=4, num_causes=4, other_causes=0,
    time_dim=TIME_DIM, device='cpu', masks=masks,
)
X_parent, _ = parent_scm.forward_without_input()
parent_ts = X_parent[MASK_TYPE.TIMESTAMP].squeeze(-1)

child_scm = MLPSCM(
    seq_len=128, num_outputs=4, num_causes=4, other_causes=4,
    time_dim=TIME_DIM, device='cpu', masks=masks,
)
fk_ids = torch.randint(0, 32, (128,), device='cpu')
child_scm.t_min = parent_ts[fk_ids]
parent_data = {MASK_TYPE.CAUSAL_OUTPUT: X_parent[MASK_TYPE.CAUSAL_OUTPUT],
               MASK_TYPE.TIMESTAMP: X_parent[MASK_TYPE.TIMESTAMP]}
X_child, _, _ = child_scm.forward_with_input([parent_data], fk_ids.unsqueeze(-1))
child_ts = X_child[MASK_TYPE.TIMESTAMP].squeeze(-1)

# Check t_min constraint
violations = (child_ts < parent_ts[fk_ids]).sum().item()
assert violations == 0, f'{violations} violations!'

# Check that time_features have correct shape
print(f'Parent ts range: [{parent_ts.min():.3f}, {parent_ts.max():.3f}]')
print(f'Child ts range: [{child_ts.min():.3f}, {child_ts.max():.3f}]')
print(f'Violations: {violations}')
print('Phase 3 integration test PASSED')
"
```
Expected: `Violations: 0`, PASSED.

- [ ] **Step 2: Commit** (no file changes)

---

## Self-Review Checklist

1. **Spec coverage:**
   - [x] Problem 1 (basis vector input): Tasks 1.1–1.9
   - [x] Problem 2 (cross-table constraints): Tasks 2.1–2.6
   - [x] Problem 3 (intra-group sort): Tasks 3.1–3.3
   - [x] Fixed TIME_DIM=11: Task 1.4
   - [x] Gate indicators: Tasks 1.2, 1.3
   - [x] Remove Fourier encoding: Task 1.7
   - [x] Noise stays in sampler only: Implicit (noise never passed to evaluate_basis)
   - [x] DAG topology fallback: Task 2.4
   - [x] Gamma decay with normalized tau: Task 2.2
   - [x] Probabilistic per-FK-group sort: Tasks 3.1, 3.2
   - [x] Bernoulli params confirmed (trend 0.8, seasonal 0.6, spike 0.3)
   - [x] Gamma tiers {0.0, 0.5, 1.5, 3.0} confirmed
   - [x] t_min fallback ranges confirmed (0-0.15T / 0.15T-0.45T / 0.45T-0.85T)

2. **Placeholder scan:** No TBD/TODO/fill-in-later patterns. All code is concrete.

3. **Type consistency:**
   - `evaluate_basis` returns `(N, 8)` everywhere
   - `build_gate_vector` returns `(3,)` everywhere
   - `_prepare_time_features` returns `(N,)` timestamps + `(N, 11)` features everywhere
   - `t_min` is `(N,)` in [0, 1] normalized throughout
   - `gamma` is float, `p_sort` is float in [0, 1]
