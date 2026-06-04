# Per-Column Nonlinear MLP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SG's shared linear basis projection with per-(rdb,table,col) random nonlinear MLPs to break the low-rank feature constraint.

**Architecture:** New `PerColumnMLP` class (2-4 layer MLP, fixed hidden_dim=32, per-layer sampled activation) replaces `_construct_features`. SG's signal normalization + group_scale preserved. Gated behind `use_percol_mlp` flag.

**Tech Stack:** PyTorch, existing MLPSCM framework

**Spec:** `docs/superpowers/specs/2026-06-05-percolumn-mlp-design.md`

---

### File Map

| File | Purpose |
|------|---------|
| `data_generation/RDB/src/prior/prior_config.py` | Add `PERCOL_MLP_HP`, add `use_percol_mlp` gate to `DEFAULT_SAMPLED_HP` |
| `data_generation/RDB/src/prior/mlp_scm.py` | Add `PerColumnMLP` class, `_construct_features_mlp`, gate logic |
| `data_generation/RDB/src/table_def/table_generation.py` | Pass `use_percol_mlp` + `percol_mlp_hp` through HP sampling |

---

### Task 1: Add `PerColumnMLP` class to `mlp_scm.py`

**Files:**
- Modify: `data_generation/RDB/src/prior/mlp_scm.py`

- [ ] **Step 1: Add `PerColumnMLP` class before `MLPSCM` class**

Insert after line ~88 (after `MLPOutput` namedtuple), before `class MLPSCM`:

```python
class PerColumnMLP(nn.Module):
    """Per-column random nonlinear MLP for feature diversity.

    Takes concatenated normalized signals → produces a scalar feature value.
    Weights are randomly initialized and FIXED (not trained).

    Parameters
    ----------
    in_dim : int
        Input dimension (varies by archetype: 12/32/43).
    hidden_dim : int
        Hidden layer dimension (fixed at 32, per Plurel).
    num_layers : int
        Total layers including input and output (2-4).
    activations : list[nn.Module]
        Per-hidden-layer activation functions.
    init_std : float
        Weight init std for all Linear layers.
    seed : int
        RNG seed for deterministic weight init.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        activations: list,
        init_std: float,
        seed: int,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"
        assert len(activations) == num_layers - 1, (
            f"Need {num_layers - 1} activations, got {len(activations)}"
        )

        g = torch.Generator()
        g.manual_seed(seed)

        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [1]
        for i in range(num_layers):
            linear = nn.Linear(dims[i], dims[i + 1], bias=True)
            nn.init.normal_(linear.weight, std=init_std)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)
            if i < num_layers - 1:
                self.layers.append(activations[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (seq_len, in_dim) → (seq_len, 1)"""
        for layer in self.layers:
            x = layer(x)
        return x
```

- [ ] **Step 2: Verify syntax**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import ast; ast.parse(open('data_generation/RDB/src/prior/mlp_scm.py').read()); print('OK')
"
```

---

### Task 2: Add per-column MLP init logic to `MLPSCM.__init__`

**Files:**
- Modify: `data_generation/RDB/src/prior/mlp_scm.py`

- [ ] **Step 1: Add `use_percol_mlp` and `percol_mlp_hp` params to `__init__` signature**

In `MLPSCM.__init__`, after line 257 (`_exp_skip_sg: bool = False,`), add:

```python
        # Per-column nonlinear MLP (replaces linear SG basis when active)
        use_percol_mlp: bool = False,
        percol_mlp_hp: dict | None = None,
```

- [ ] **Step 2: Store new params**

After line 330 (`self.coupling_lambda = coupling_lambda`), add:

```python
        self.use_percol_mlp = use_percol_mlp
        self.percol_mlp_hp = percol_mlp_hp or {}
```

- [ ] **Step 3: Build per-column MLPs when `use_percol_mlp` is active**

In the `if self.use_signal_group_features:` block (line 365), add an inner branch. After `if self.use_signal_group_features:` (line 365), modify the block so that when `use_percol_mlp` is True, we skip `_sample_feature_groups`/`_init_group_bases`/`intrinsic_projector` init and instead build MLPs.

Replace lines 365-385:
```python
        if self.use_signal_group_features:
            # --- Core hyperparameters ---
            # ...
```
with:

```python
        if self.use_signal_group_features:
            if self.use_percol_mlp:
                self._init_percol_mlps(masks)
            else:
                # existing SG init...
                self.feature_assignments = self._sample_feature_groups(...)
                # ... etc (existing code, unchanged)
```

Actually, to minimize diff risk, add the percol MLP init in a separate block AFTER the existing SG init block. Find the end of the `if self.use_signal_group_features:` block (after `self._cross_feature_mix` init, around line 397). After the block ends, add:

```python
        # --- Per-column MLP initialization (replaces SG basis when active) ---
        if self.use_signal_group_features and self.use_percol_mlp:
            self._init_percol_mlps(masks)
```

- [ ] **Step 4: Add `_init_percol_mlps` method**

After `_init_group_bases` method (after line 1349 `return group_bases`), add:

```python
    def _init_percol_mlps(self, masks: Dict[MASK_TYPE, int]) -> None:
        """Initialize per-column random nonlinear MLPs for feature generation.

        Each feature column j gets its own MLP_j with independent random weights.
        Input signals are selected by archetype and concatenated.
        """
        from .activations import get_activations

        hp = self.percol_mlp_hp
        num_layers = hp.get("percol_mlp_num_layers", 3)
        hidden_dim = hp.get("percol_mlp_hidden_dim", 32)
        init_std = self.init_std
        n_features = masks.get(MASK_TYPE.X, 12)

        # Sample per-layer activations (list of callables)
        act_choices = hp.get("percol_mlp_activation", [nn.Tanh])
        if callable(act_choices):
            act_choices = act_choices()
        if not isinstance(act_choices, list):
            act_choices = [act_choices]
        activations = [
            act_choices[i % len(act_choices)]() if callable(act_choices[i % len(act_choices)])
            else act_choices[i % len(act_choices)]
            for i in range(num_layers - 1)
        ]

        self.percol_mlps = nn.ModuleList()
        for j in range(n_features):
            seed = hash((id(self), j)) & 0x7FFFFFFF
            mlp_j = PerColumnMLP(
                in_dim=self._percol_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activations=activations,
                init_std=init_std,
                seed=seed,
            )
            self.percol_mlps.append(mlp_j)
```

- [ ] **Step 5: Compute `_percol_input_dim` from archetype**

In `_init_percol_mlps`, before the MLP creation loop, compute input dim from archetype. Read the existing archetype params from `self.archetype_params`:

```python
        ap = self.archetype_params or {}
        is_source = ap.get("is_source", False)
        is_timestamp = ap.get("is_timestamp", False)

        # Compute input dimension from active signal sources
        dim = 0
        if is_timestamp:
            dim += 11  # time_basis(8) + time_gates(3)
        if not is_source:
            dim += 12 + 8  # parent_sig(12) + path_sig(8)
        dim += self.intrinsic_dim  # intrinsic always present
        self._percol_input_dim = dim
```

Wait — the signals are already normalized (RMS-norm × scale) before entering the MLP. We need to know which signals to concat. Let's compute it from the normalized signals we build in the forward path.

Better approach: compute input_dim lazily on first forward call, or compute from `_normalize_signals` output. Actually, the simplest: just compute it from archetype_params as above, which matches the design doc's table (12/32/43).

- [ ] **Step 6: Verify syntax**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import ast; ast.parse(open('data_generation/RDB/src/prior/mlp_scm.py').read()); print('OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add data_generation/RDB/src/prior/mlp_scm.py
git commit -m "feat: add PerColumnMLP class and init logic to MLPSCM"
```

---

### Task 3: Add `_construct_features_mlp` and gate logic

**Files:**
- Modify: `data_generation/RDB/src/prior/mlp_scm.py`

- [ ] **Step 1: Add `_construct_features_mlp` method**

After `_init_percol_mlps`, add:

```python
    def _construct_features_mlp(
        self,
        signals: dict,
        residual_sigma: float,
    ) -> torch.Tensor:
        """Per-column MLP feature construction (replaces linear SG basis).

        Each feature_j = MLP_j(concat(selected_signals)) + ε_j.

        Parameters
        ----------
        signals : dict
            Normalized signals: "time_basis", "time_gates", "parent", "path", "intrinsic".
        residual_sigma : float
            Per-feature residual noise std.

        Returns
        -------
        X : (seq_len, n_features)
        """
        # Build concatenated input per archetype
        ap = self.archetype_params or {}
        is_source = ap.get("is_source", False)
        is_timestamp = ap.get("is_timestamp", False)

        parts = []
        if is_timestamp:
            parts.append(signals["time_basis"])       # (N, 8)
            parts.append(signals.get("time_gates",    # (N, 3)
                torch.zeros(signals["time_basis"].shape[0], 3, device=self.device)))
        if not is_source:
            parts.append(signals["parent"])            # (N, 12)
            parts.append(signals["path"])              # (N, 8)
        parts.append(signals["intrinsic"])             # (N, intrinsic_dim)

        x_input = torch.cat(parts, dim=1)  # (N, input_dim)

        n_features = len(self.percol_mlps)
        X = torch.zeros(self.seq_len, n_features, device=self.device)

        for j, mlp_j in enumerate(self.percol_mlps):
            if j >= X.shape[1]:
                break
            X[:, j] = mlp_j(x_input).squeeze(-1)
            if residual_sigma > 0:
                X[:, j] += torch.randn(self.seq_len, device=self.device) * residual_sigma

        return X
```

- [ ] **Step 2: Gate in `forward_without_input` (source tables)**

In `forward_without_input` (around line 672), change:
```python
        if self.use_signal_group_features and not self._exp_skip_sg:
            ...
            X_sg = self._construct_features(signals, self.residual_sigma)
            X_sg = self._apply_cross_feature_coupling(X_sg)
            X[MASK_TYPE.X] = X_sg
```
to:
```python
        if self.use_signal_group_features and not self._exp_skip_sg:
            ...
            if self.use_percol_mlp:
                X_sg = self._construct_features_mlp(signals, self.residual_sigma)
            else:
                X_sg = self._construct_features(signals, self.residual_sigma)
                X_sg = self._apply_cross_feature_coupling(X_sg)
            X[MASK_TYPE.X] = X_sg
```

- [ ] **Step 3: Gate in `forward_with_input` (child tables)**

Same pattern. Around line 774-805, change the same block identically.

- [ ] **Step 4: Handle `_exp_skip_sg` compatibility**

When `_exp_skip_sg` is True, skip both SG and percol MLP (use raw MLP X). No change needed — the existing guard `if self.use_signal_group_features and not self._exp_skip_sg:` already handles this.

- [ ] **Step 5: Verify syntax**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import ast; ast.parse(open('data_generation/RDB/src/prior/mlp_scm.py').read()); print('OK')
"
```

- [ ] **Step 6: Commit**

```bash
git add data_generation/RDB/src/prior/mlp_scm.py
git commit -m "feat: add _construct_features_mlp and gate logic for percol MLP"
```

---

### Task 4: Add `PERCOL_MLP_HP` and `use_percol_mlp` gate to config

**Files:**
- Modify: `data_generation/RDB/src/prior/prior_config.py`

- [ ] **Step 1: Add `PERCOL_MLP_HP` config dict**

After `DEFAULT_HSBM_HP` (after line 393), add:

```python
# Per-column nonlinear MLP hyperparameters (replaces SG linear basis when active).
PERCOL_MLP_HP = {
    "percol_mlp_num_layers": {
        "distribution": "meta_choice",
        "choice_values": [2, 3, 4],
    },
    "percol_mlp_hidden_dim": 32,  # fixed, not sampled
}
```

Note: `percol_mlp_activation` is handled separately via `get_activations` in MLPSCM init (same pattern as `mlp_activations` in `DEFAULT_MLP_SCM_CONFIG`), not through the HP sampler.

- [ ] **Step 2: Add gate flag to `DEFAULT_SAMPLED_HP`**

After the `use_signal_group_features` entry (line 224), add:

```python
    "use_percol_mlp": {
        "distribution": "meta_choice",
        "choice_values": [True],
    },
```

- [ ] **Step 3: Verify config imports**

```bash
cd /data/caijunyu/RDBPFN/data_generation/RDB && pixi run python -c "
from src.prior.prior_config import PERCOL_MLP_HP
print('percol_mlp_num_layers:', PERCOL_MLP_HP['percol_mlp_num_layers'])
print('percol_mlp_hidden_dim:', PERCOL_MLP_HP['percol_mlp_hidden_dim'])
"
```

- [ ] **Step 4: Commit**

```bash
git add data_generation/RDB/src/prior/prior_config.py
git commit -m "feat: add PERCOL_MLP_HP and use_percol_mlp gate config"
```

---

### Task 5: HP plumbing in `table_generation.py`

**Files:**
- Modify: `data_generation/RDB/src/table_def/table_generation.py`

- [ ] **Step 1: Sample `use_percol_mlp` and `percol_mlp_num_layers`**

In the HP sampling block (around line 1747, where `sampled_scm = hpsampler.sample()`), add percol MLP HP extraction. After the existing SCM HP sampling, add:

```python
            # Per-column MLP HP (only relevant when use_signal_group_features=True)
            percol_sample = hpsampler.sample()
            use_percol_mlp_val = percol_sample.get("use_percol_mlp", False)
            use_percol_mlp = use_percol_mlp_val() if callable(use_percol_mlp_val) else use_percol_mlp_val
            percol_mlp_nl_val = percol_sample.get("percol_mlp_num_layers", 3)
            percol_mlp_num_layers = percol_mlp_nl_val() if callable(percol_mlp_nl_val) else percol_mlp_nl_val
            percol_mlp_hp = {
                "percol_mlp_num_layers": percol_mlp_num_layers,
                "percol_mlp_hidden_dim": 32,
                "percol_mlp_activation": None,  # resolved in MLPSCM._init_percol_mlps
            }
```

- [ ] **Step 2: Pass to `combined_params`**

In the `combined_params` assembly (around line 1812-1817), add:

```python
            combined_params["use_percol_mlp"] = use_percol_mlp
            combined_params["percol_mlp_hp"] = percol_mlp_hp
```

- [ ] **Step 3: No external seed plumbing needed**

Per-column MLP seeds are derived from `id(self)` in `_init_percol_mlps` (see Task 2 Step 4). Each MLPSCM instance has a unique `id()`, so per-table uniqueness is guaranteed without external seed injection. No changes to `TableGenerator` seed infrastructure needed.

- [ ] **Step 4: Import `PERCOL_MLP_HP` in HP sampler init**

In `table_generation.py`, find where HSBM sampler is created (search for `hsbm_sampler =`). Near it, ensure the percol HP keys are in the sampler's config. If the sampler reads from `DEFAULT_SAMPLED_HP` directly (which includes both SCM and HSBM keys), the `use_percol_mlp` and `percol_mlp_num_layers` will already be available since we added them there.

- [ ] **Step 5: Verify syntax**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import ast; ast.parse(open('data_generation/RDB/src/table_def/table_generation.py').read()); print('OK')
"
```

- [ ] **Step 6: Commit**

```bash
git add data_generation/RDB/src/table_def/table_generation.py
git commit -m "feat: plumb percol MLP HP through table generation pipeline"
```

---

### Task 6: End-to-end smoke test (4 RDBs)

**Files:**
- None (test only)

- [ ] **Step 1: Generate 4 RDBs with percol MLP enabled**

```bash
cd /data/caijunyu/RDBPFN && bash scripts/run_pipeline.sh 4 0 percol_mlp_test false true true false 3 4 true true "0" true 2>&1 | head -40
```

Expected: data generation completes without errors, 4 RDBs in `data_generation/RDB_datasets/percol_mlp_test/`.

- [ ] **Step 2: Verify generated data has features**

```bash
cd /data/caijunyu/RDBPFN && pixi run python -c "
import pandas as pd
from pathlib import Path
rdb0 = Path('data_generation/RDB_datasets/percol_mlp_test/dag_rdb_0')
for pq in sorted(rdb0.glob('table_*.parquet'))[:2]:
    df = pd.read_parquet(pq)
    print(f'{pq.stem}: {df.shape}')
    print(df.head(2))
"
```

Expected: tables with rows × columns, numeric feature values (not all zeros).

- [ ] **Step 3: Commit**

```bash
echo "Smoke test passed: 4 RDBs generated with percol MLP" >> progress.md
git add progress.md
git commit -m "chore: note percol MLP smoke test passed"
```

---

### Task 7: Feature analysis on 64-RDB run

**Files:**
- Modify: `scripts/fk_sparsity_analyze.py` (extend for percol MLP comparison)

- [ ] **Step 1: Generate 64 RDBs with percol MLP**

```bash
cd /data/caijunyu/RDBPFN && bash scripts/run_pipeline.sh 64 0 percol_mlp_v1 false false false false 3 16 true true "0" true
```

- [ ] **Step 2: Run h5 feature analysis**

```bash
cd /data/caijunyu/RDBPFN && pixi run python scripts/fk_sparsity_analyze.py \
  --h5-path model_pretrain/pretrain_datasets/percol_mlp_v1.h5 \
  --baseline-h5 model_pretrain/pretrain_datasets/v53.h5 \
  --skip-raw
```

- [ ] **Step 3: Compare against design expectations**

Check:
- eff_rank_ratio: should be substantially higher than 0.01 (previously both were ~0.01 due to analysis issue)
- mean|corr|: should be lower than v53's 0.1404 (less linear redundancy)
- feat-label |corr|: should be comparable to v53's 0.1211

- [ ] **Step 4: Document results in findings.md**

Run the analysis, capture the output, and append to findings.md:

```bash
cd /data/caijunyu/RDBPFN && pixi run python scripts/fk_sparsity_analyze.py \
  --h5-path model_pretrain/pretrain_datasets/percol_mlp_v1.h5 \
  --baseline-h5 model_pretrain/pretrain_datasets/v53.h5 \
  --skip-raw 2>&1 | tee -a /tmp/percol_mlp_analysis.txt
```

Then append a summary table to findings.md with actual numbers from the run output.

---

### Task 8: Full 1024-RDB training (optional, gated on Task 7 results)

Only proceed if Task 7 shows meaningful improvement in eff_rank or mean|corr|.

- [ ] **Step 1: Generate 1024 RDBs**

```bash
cd /data/caijunyu/RDBPFN && bash scripts/run_pipeline.sh 1024 0 percol_mlp_full false false false false 3 16 false true "0,1" true
```

- [ ] **Step 2: Compare eval AUC vs v5.3 baseline**
