from __future__ import annotations

import math
import os
import random
import warnings
from collections import namedtuple
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .utils import GaussianNoise, XSampler, MASK_TYPE, SCM_OUTPUT
from .prior_config import DEFAULT_MLP_SCM_CONFIG
from .temporal_vocab import TemporalVocab

# ---------------------------------------------------------------------------
# Contract 0.1 — RelationKey
# ---------------------------------------------------------------------------

RelationKey = namedtuple("RelationKey", ["child_table", "fk_col_idx", "parent_table"])


def make_relation_key(child_table: str, fk_col_idx: int, parent_table: str) -> RelationKey:
    """Single factory for all RelationKey creation. One place to validate."""
    return RelationKey(child_table=child_table, fk_col_idx=fk_col_idx, parent_table=parent_table)


# ---------------------------------------------------------------------------
# Contract 0.2 — Time signal split hard boundaries
# ---------------------------------------------------------------------------

TIME_BASIS_DIMS = (0, 8)   # [0:8] — Fourier basis components
TIME_GATE_DIMS = (8, 11)   # [8:11] — Dirichlet gate vector


# ---------------------------------------------------------------------------
# Contract 0.4 — FeatureGenMode gating table
# ---------------------------------------------------------------------------

FEATURE_GEN_MODE_ACTIVE_HPS = {
    True: {
        "archetype_perturb_std", "max_groups_per_feature", "loading_sigma",
        "loading_log_mean", "residual_sigma", "basis_group_divisor",
        "group_scale_time", "group_scale_parent", "group_scale_path",
        "group_scale_intrinsic",
    },
    False: {"parent_injection_scale"},
}


# ---------------------------------------------------------------------------
# Contract 0.2 helper — PathEncoder
# ---------------------------------------------------------------------------

def compute_hsbm_locality(hierarchies: list[list[int]]) -> float:
    """Deterministic structural proxy for HSBM connection concentration.

    Higher values → tighter block-local connections → stronger feature correlation.
    """
    if not hierarchies:
        return 0.0
    per_edge = []
    for h in hierarchies:
        val = sum(math.log(max(h[l], 1)) for l in range(len(h)))
        per_edge.append(val)
    max_possible = max(per_edge) if per_edge else 1.0
    if max_possible <= 0:
        return 0.0
    return float(np.mean([v / max_possible for v in per_edge]))


class PathEncoder(nn.Module):
    """Per-edge path encoder: embed each hierarchy level then project to fixed dim.

    Parameters
    ----------
    num_levels : int
        Number of hierarchy levels (L_p) for this FK edge.
    max_blocks_per_level : int
        Maximum number of blocks at any single level (vocabulary size per embedding).
    out_dim : int, default=8
        Output dimensionality.
    embed_dim : int, default=4
        Embedding dimensionality per hierarchy level.
    """

    def __init__(
        self,
        num_levels: int,
        max_blocks_per_level: int,
        out_dim: int = 8,
        embed_dim: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(max_blocks_per_level, embed_dim) for _ in range(num_levels)
        ])
        self.projection = nn.Linear(num_levels * embed_dim, out_dim)

    def forward(self, block_paths: torch.Tensor) -> torch.Tensor:
        """Encode a block-path prefix into a fixed-dim signal vector.

        Parameters
        ----------
        block_paths : torch.Tensor
            LongTensor of shape ``(seq_len, num_levels)``. Each entry is a
            block index at the corresponding hierarchy level.

        Returns
        -------
        torch.Tensor of shape ``(seq_len, out_dim)``.
        """
        embeds = []
        for l in range(self.num_levels):
            embeds.append(self.embeddings[l](block_paths[:, l]))
        concat = torch.cat(embeds, dim=-1)
        return self.projection(concat)


class PerColumnMLP(nn.Module):
    """Per-column random nonlinear MLP for feature diversity.

    Takes concatenated normalized signals and produces a scalar feature value.
    Weights are randomly initialized and FIXED (not trained).

    Parameters
    ----------
    in_dim : int
        Input dimension (varies by archetype: 12/32/43).
    hidden_dim : int
        Hidden layer dimension (fixed at 32).
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

    def forward(self, x):
        """x: (seq_len, in_dim) -> (seq_len, 1)"""
        for layer in self.layers:
            x = layer(x)
        return x


class MLPSCM(nn.Module):
    """Generates synthetic tabular datasets using a Multi-Layer Perceptron (MLP) based Structural Causal Model (SCM).

    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    num_outputs : int, default=1
        The number of outputs.

    is_causal : bool, default=True
        - If `True`, simulates a causal graph: `X` and `y` are sampled from the
          intermediate hidden states of the MLP transformation applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the MLP becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    other_causes : int, default=0
        The number of additional features from parent tables.

    sampling_ratio : float, default=1.0
        If parent tables are used, we sample `sampling_ratio` * `seq_len` samples,
        and only accept `seq_len` samples with high existence probability.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final MLP layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate MLP outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate MLP outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=10
        The total number of layers in the MLP transformation network. Must be >= 2.
        Includes the initial linear layer and subsequent blocks of
        (Activation -> Linear -> Noise).

    hidden_dim : int, default=20
        The dimensionality of the hidden representations within the MLP layers.
        If `is_causal=True`, this is automatically increased if it's smaller than
        `num_outputs + 2 * num_features` to ensure enough intermediate variables
        are generated for sampling `X` and `y`.

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    time_dim : int, default=0
        If > 0, timestamps are sampled and basis-vector features are concatenated
        to the MLP input (time-as-input). ``0`` means no time column.
        When enabled, the value is the compile-time constant ``TIME_DIM = 11``.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        # num_features: int = 100,  # Deprecated, now merge to masks
        num_outputs: int = 10,
        num_causes: int = 10,  # Meaning changed to additional noise
        other_causes: int = 0,
        sampling_ratio: float = 1.0,
        is_causal: bool = True,  # Always True now
        in_clique: bool = True,  # Used now
        sort_features: bool = True,  # No need to sort features now
        num_layers: int = 10,
        hidden_dim: int = 20,
        # config: Dict[str, Any] = DEFAULT_MLP_SCM_CONFIG,
        mlp_activations: nn.Module = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        device: str = "cpu",
        masks: Dict[MASK_TYPE, int] = {},
        # New parameters for timestamp-based sampling
        use_timestamp_sampling: bool = False,
        embedding_dim: int = 10,
        batch_size: int = 32,
        parent_sampling_dist: str = "uniform",  # "uniform" or "zipf"
        parent_sampling_alpha: float = 1.0,
        time_dim: int = 0,
        parent_injection_scale: float = 0.15,
        # Signal-group feature generation parameters
        use_signal_group_features: bool = False,
        archetype_params: dict | None = None,
        relation_keys: list[RelationKey] | None = None,
        parent_causal_dims: list[int] | None = None,
        levels_per_edge: list[int] | None = None,
        group_scale_time: float = 1.0,
        group_scale_parent: float = 1.0,
        group_scale_path: float = 1.0,
        group_scale_intrinsic: float = 1.0,
        intrinsic_dim: int = 8,
        residual_sigma: float = 0.1,
        loading_sigma: float = 0.5,
        loading_log_mean: float = -0.5,
        max_groups_per_feature: int = 1,
        archetype_perturb_std: float = 0.2,
        basis_group_divisor: int = 4,
        coupling_rank: int = 2,
        coupling_lambda: float = 0.15,
        basis_perturb_eta: float = 0.0,
        num_basis_families: int = 2,
        basis_family_rho: float = 0.7,
        # Experiment flag: skip signal-group, use raw MLP X directly
        _exp_skip_sg: bool = False,
        # Per-column nonlinear MLP (replaces linear SG basis when active)
        use_percol_mlp: bool = False,
        percol_mlp_hp: dict | None = None,
        **kwargs: Dict[str, Any],
    ):
        super(MLPSCM, self).__init__()
        self.seq_len = seq_len
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.other_causes = other_causes
        self.sampling_ratio = sampling_ratio
        self.in_clique = in_clique
        self.sort_features = sort_features

        assert is_causal, "is_causal is not implemented"
        assert num_layers >= 2, "Number of layers must be at least 2."
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std

        self.device = device

        # New parameters for timestamp-based sampling
        self.use_timestamp_sampling = use_timestamp_sampling
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.parent_sampling_dist = parent_sampling_dist
        self.parent_sampling_alpha = parent_sampling_alpha

        self.time_dim = time_dim  # Will be TIME_DIM if timestamp table, 0 otherwise
        if self.time_dim > 0:
            # time_embed_mode is deprecated; basis-vector input replaces Fourier
            pass

        if self.use_timestamp_sampling:
            warnings.warn(
                "use_timestamp_sampling is deprecated and ignored: timestamp tables "
                "now use time-as-input (time_dim > 0) instead of "
                "forward_with_enhanced_temporal_sampling.",
                DeprecationWarning,
                stacklevel=2,
            )

        # TemporalVocab for real-valued timestamps fed into the MLP (not mask slices).
        self.temporal_vocab: TemporalVocab | None = None
        if self.time_dim > 0:
            self.temporal_vocab = TemporalVocab(device=self.device)

        self.eta = kwargs.get("eta", 1.0)  # Controls influence of embedding affinity
        self.parent_injection_scale = parent_injection_scale

        # --- Signal-group feature generation init ---
        self.use_signal_group_features = use_signal_group_features
        self.group_scale_time = group_scale_time
        self.group_scale_parent = group_scale_parent
        self.group_scale_path = group_scale_path
        self.group_scale_intrinsic = group_scale_intrinsic
        self.intrinsic_dim = intrinsic_dim
        self.residual_sigma = residual_sigma
        self.loading_sigma = loading_sigma
        self.loading_log_mean = loading_log_mean
        self.max_groups_per_feature = max_groups_per_feature
        self.archetype_perturb_std = archetype_perturb_std
        self.basis_group_divisor = basis_group_divisor
        self.coupling_rank = coupling_rank
        self.coupling_lambda = coupling_lambda
        self.basis_perturb_eta = basis_perturb_eta
        self.num_basis_families = num_basis_families
        self.basis_family_rho = basis_family_rho
        self._exp_skip_sg = _exp_skip_sg or (
            os.environ.get("EXP_SKIP_SG", "0") == "1"
        )
        self.use_percol_mlp = use_percol_mlp
        self.percol_mlp_hp = percol_mlp_hp or {}

        # --- Experiment: parameter ablation overrides (env-var controlled) ---
        # These must run BEFORE the signal-group init block below,
        # because basis_perturb_eta is used in _sample_feature_groups(),
        # and coupling_lambda/max_groups_per_feature/archetype_perturb_std
        # affect init-time ops (_init_cross_feature_coupling, etc.).
        if os.environ.get("EXP_ZERO_BASIS_PERTURB", "0") == "1":
            self.basis_perturb_eta = 0.0
        if os.environ.get("EXP_ZERO_COUPLING", "0") == "1":
            self.coupling_lambda = 0.0
        if os.environ.get("EXP_ZERO_RESIDUAL", "0") == "1":
            self.residual_sigma = 0.0
        if os.environ.get("EXP_UNIFORM_SCALE", "0") == "1":
            self.group_scale_time = 1.0
            self.group_scale_parent = 1.0
            self.group_scale_path = 1.0
            self.group_scale_intrinsic = 1.0
        if os.environ.get("EXP_NO_ALPHA_PERTURB", "0") == "1":
            self.archetype_perturb_std = 0.0
        if os.environ.get("EXP_MAX_GROUPS_1", "0") == "1":
            self.max_groups_per_feature = 1

        self.alpha_final: torch.Tensor | None = None
        self.feature_assignments: list[dict] = []
        self.group_bases: dict[str, dict[str, torch.Tensor]] = {}
        self.parent_projectors = nn.ModuleDict()
        self.path_encoders = nn.ModuleDict()

        if self.use_signal_group_features:
            _archetype_params = archetype_params or {}
            _relation_keys = relation_keys or []
            _parent_causal_dims = parent_causal_dims or []
            _levels_per_edge = levels_per_edge or []

            archetype = self._classify_archetype(_archetype_params)
            alpha_base = self._compute_base_alpha(archetype)
            self.alpha_final = self._perturb_alpha(alpha_base)

            n_feat = masks.get(MASK_TYPE.X, 12)
            self.feature_assignments = self._sample_feature_groups(
                self.alpha_final, n_feat,
            )
            self.group_bases = self._init_group_bases()
            self.intrinsic_projector = nn.Linear(
                self.num_outputs, self.intrinsic_dim, device=self.device,
            )
            self._init_parent_projectors(_relation_keys, _parent_causal_dims)
            self._init_path_encoders(_relation_keys, _levels_per_edge)
            self._init_cross_feature_coupling(n_feat)

            # --- Per-column MLP init (replaces SG basis when active) ---
            if self.use_percol_mlp:
                self._init_percol_mlps(masks)

        if self.is_causal:
            total_features = masks[MASK_TYPE.X]
            # Ensure enough intermediate variables for sampling X and y
            self.hidden_dim = max(
                self.hidden_dim,
                int(
                    np.ceil(
                        (self.num_outputs + total_features + 1) / (self.num_layers - 1)
                    )
                ),
            )
        else:
            # In non-causal mode, features are the causes
            raise ValueError("Non-causal mode is not implemented")
            # total_features = sum(masks.values()) if masks else 100
            # self.num_causes = total_features

        # Define the input sampler
        self.xsampler = XSampler(
            int(self.seq_len * self.sampling_ratio),
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # Build layers: optional time embedding is concatenated after root causes,
        # before parent causal outputs (same total width as num_causes + time_dim + other_causes).
        first_in_dim = self.num_causes + self.other_causes + self.time_dim
        layers = [nn.Linear(first_in_dim, self.hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(self.generate_layer_modules())
        if not self.is_causal:
            layers.append(self.generate_layer_modules(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)

        # Initialize layers
        self.initialize_parameters()

        # Generate the masks, defining what position are used
        self.generate_masks(masks)

    def generate_layer_modules(self, is_output_layer=False):
        """Generates a layer module with activation, linear transformation, and noise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim
        activation = self.mlp_activations()
        linear_layer = nn.Linear(self.hidden_dim, out_dim)

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(
                    torch.zeros(size=(1, out_dim), device=self.device),
                    float(self.noise_std),
                )
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(activation, linear_layer, noise_layer)

    def initialize_parameters(self):
        """Initializes parameters using block-wise dropout or normal initialization."""
        with torch.no_grad():
            for i, (_, param) in enumerate(self.layers.named_parameters()):
                if self.block_wise_dropout and param.dim() == 2:
                    self.initialize_with_block_dropout(param, i)
                else:
                    self.initialize_normally(param, i)

    def initialize_with_block_dropout(self, param, index):
        """Initializes parameters using block-wise dropout."""
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [dim // n_blocks for dim in param.shape]
        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        for block in range(n_blocks):
            block_slice = tuple(
                slice(dim * block, dim * (block + 1)) for dim in block_size
            )
            nn.init.normal_(
                param[block_slice],
                std=self.init_std
                / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1),
            )

    def initialize_normally(self, param, index):
        """Initializes parameters using normal distribution."""
        if param.dim() == 2:  # Applies only to weights, not biases
            dropout_prob = (
                self.mlp_dropout_prob if index > 0 else 0
            )  # No dropout for the first layer's weights
            dropout_prob = min(dropout_prob, 0.99)
            std = self.init_std / (
                (1 - dropout_prob) ** 0.5 if self.scale_init_std_by_dropout else 1
            )
            nn.init.normal_(param, std=std)
            param *= torch.bernoulli(torch.full_like(param, 1 - dropout_prob))

    def generate_masks(self, masks: Dict[MASK_TYPE, int]):
        """Generates the masks, defining what position are used"""
        self.masks = {}
        total_features = (self.num_layers - 1) * self.hidden_dim
        for key, value in masks.items():
            # each key is a string, each value is an integer representing the number of features to use
            if self.in_clique:
                # sample a consecutive block of features
                idx = torch.arange(value, device=self.device)
                idx += random.randint(0, total_features - value)
            else:
                # sample random features
                idx = torch.tensor(
                    random.sample(range(total_features), value),
                    device=self.device,
                )
            self.masks[key] = idx

        if self.in_clique:
            # sample a consecutive block of features
            idx = torch.arange(self.num_outputs, device=self.device)
            idx += random.randint(0, total_features - self.num_outputs)
        else:
            # sample random features
            idx = torch.tensor(
                random.sample(
                    range(total_features - self.num_outputs, total_features),
                    self.num_outputs,
                ),
                device=self.device,
            )
        self.masks[MASK_TYPE.CAUSAL_OUTPUT] = idx
        self.masks[MASK_TYPE.FULL] = torch.arange(total_features, device=self.device)

        return

    def sample_zipf_indices(
        self, n_parent_samples: int, num_samples: int, alpha: float = 2.0
    ):
        """
        Sample indices from a Zipf distribution with random remapping.

        This method uses np.random.zipf to generate samples following a Zipf
        distribution, then remaps them through a random permutation. This ensures
        the long-tail property is preserved, but the "popular" indices are randomly
        distributed rather than always being the first few indices.

        For example, instead of index 0 being most common, a random index like 47
        might be most common, preserving the Zipf distribution shape.

        Parameters
        ----------
        n_parent_samples : int
            Number of samples available in the parent table (max index)
        num_samples : int
            Number of indices to sample
        alpha : float
            Zipf distribution parameter (must be > 1). Higher values create
            heavier tails (more concentration on small indices).

        Returns
        -------
        list
            List of sampled indices in range [0, n_parent_samples) following
            a Zipf distribution remapped to random positions
        """
        # Edge case: empty parent table
        if n_parent_samples <= 0:
            raise ValueError(
                f"Parent table has {n_parent_samples} samples, cannot sample from empty table"
            )

        # Edge case: single row in parent table
        if n_parent_samples == 1:
            return [0] * num_samples

        # Ensure alpha is valid for Zipf distribution (must be > 1)
        alpha = max(alpha, 1.01)

        # Create a random permutation for remapping
        # This determines which actual indices will be "popular"
        index_permutation = np.random.permutation(n_parent_samples)

        # Sample from Zipf distribution
        # np.random.zipf generates values >= 1, so we subtract 1 to get 0-indexed
        zipf_samples = np.random.zipf(alpha, size=num_samples) - 1

        # Clip samples to valid range [0, n_parent_samples)
        # This handles the case where zipf samples exceed parent table size
        clipped_indices = np.clip(zipf_samples, 0, n_parent_samples - 1)

        # Remap through the permutation
        # This makes the long-tail distribution apply to random indices
        # rather than always favoring index 0
        remapped_indices = index_permutation[clipped_indices]

        return remapped_indices.tolist()


    def _prepare_time_features(
        self,
        n: int,
        t_min: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample timestamps and build basis-vector MLP input.

        Args:
            n: Number of rows to sample.
            t_min: Optional (n,) tensor of per-row lower bounds **in days**
                   (same scale as NUM_DAYS). If None, defaults to 0.

        Returns:
            timestamps_days: (n,) day indices — stored in X[MASK_TYPE.TIMESTAMP].
            time_features: (n, self.time_dim) — basis(8) + gates(3), MLP input.
        """
        if n <= 0:
            raise ValueError(f"_prepare_time_features expects n > 0, got {n}")
        if self.temporal_vocab is None:
            raise RuntimeError("temporal_vocab is unset; time_dim must be > 0")

        from .temporal_vocab import NUM_DAYS  # noqa: PLC0415

        num_days = float(NUM_DAYS)

        # 1. Generate intensity distribution once (day-index domain)
        self.temporal_vocab.generate(time_range=(0.0, num_days))

        # 2. Sample timestamps with optional per-row t_min (already in days) and decay
        t_min_raw = None
        if t_min is not None:
            t_min_raw = t_min.to(self.device).clamp(0.0, num_days)
        timestamps = self.temporal_vocab.sample_time(
            num_samples=n,
            time_range=(0.0, num_days),
            t_min=t_min_raw,
            gamma=getattr(self, "gamma", 0.0),
        )
        timestamps_days = timestamps.to(self.device).clamp(0.0, num_days)

        # 3. Evaluate basis at sampled day indices
        basis = self.temporal_vocab.evaluate_basis(timestamps_days)  # (n, 8)

        # 4. Build gate vector and broadcast to all rows
        gate = self.temporal_vocab.build_gate_vector().to(self.device)  # (3,)
        gate_broadcast = gate.unsqueeze(0).expand(n, -1)  # (n, 3)

        # 5. Concatenate: basis(8) + gates(3) = (n, time_dim)
        time_features = torch.cat([basis, gate_broadcast], dim=-1)
        assert time_features.shape == (n, self.time_dim), (
            f"Expected ({n}, {self.time_dim}), got {time_features.shape}"
        )

        return timestamps_days, time_features

    def forward_without_input(self):
        """
        This case is for generate tables without parent tables.
        Therefore, we do not need to sample the parent tables.
        """
        causes_raw = self.xsampler.sample()  # (seq_len, num_causes)
        n_sampled = int(self.seq_len * self.sampling_ratio)
        if self.time_dim > 0:
            timestamps_norm, time_features = self._prepare_time_features(n_sampled)
            causes = torch.cat([causes_raw, time_features], dim=-1)
        else:
            causes = causes_raw

        # Generate outputs through MLP layers
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[
            2:
        ]  # Start from 2 because the first layer is only linear without activation

        # Handle outputs based on causality
        X, outputs_flat = self.handle_outputs(outputs, self.masks)

        if self.time_dim > 0:
            X[MASK_TYPE.TIMESTAMP] = timestamps_norm.unsqueeze(-1)

        # Check for NaNs and handle them by setting to default values
        for _, value in X.items():
            if torch.any(torch.isnan(value)):
                value[:] = 0.0

        if self.use_signal_group_features and not self._exp_skip_sg:
            # Save MLP original X before overwrite, for diagnostic comparison
            X_mlp = X[MASK_TYPE.X].clone()
            self._x_causal_corr_mlp = self._compute_x_causal_corr(
                X_mlp, X[MASK_TYPE.CAUSAL_OUTPUT],
            )
            X[MASK_TYPE.X_MLP] = X_mlp

            self._cached_time_features = time_features if self.time_dim > 0 else None

            time_sig_raw = self._build_time_signal()
            intrinsic_sig_raw = self._build_intrinsic_signal(X[MASK_TYPE.CAUSAL_OUTPUT])
            parent_sig_raw = torch.zeros(self.seq_len, 12, device=self.device)
            path_sig_raw = torch.zeros(self.seq_len, 8, device=self.device)

            time_sig, parent_sig, path_sig, intrinsic_sig = self._normalize_signals(
                time_sig_raw, parent_sig_raw, path_sig_raw, intrinsic_sig_raw,
            )

            signals = {
                "time_basis": time_sig[:, TIME_BASIS_DIMS[0]:TIME_BASIS_DIMS[1]],
                "time_gates": time_sig[:, TIME_GATE_DIMS[0]:TIME_GATE_DIMS[1]],
                "parent": parent_sig,
                "path": path_sig,
                "intrinsic": intrinsic_sig,
            }

            X_sg = self._construct_features(signals, self.residual_sigma)
            X_sg = self._apply_cross_feature_coupling(X_sg)
            X[MASK_TYPE.X] = X_sg

        # Diagnostic: per-feature max |pearsonr| with CAUSAL_OUTPUT
        self._x_causal_corr = self._compute_x_causal_corr(
            X[MASK_TYPE.X], X[MASK_TYPE.CAUSAL_OUTPUT])

        # Return both masked outputs and full outputs for TableGenerator
        return X, outputs_flat

    def forward_with_input(
        self,
        parent_data_list: list,
        fk_ids: torch.Tensor,
        block_paths: torch.Tensor | None = None,
    ):
        """Generate child-table rows conditioned on pre-determined FK connections.

        FK connections are determined externally (e.g., via HSBM) and passed in
        as ``fk_ids``.  This method looks up parent CAUSAL_OUTPUT embeddings
        using those FK indices and runs the MLP forward pass to produce features.

        Parameters
        ----------
        parent_data_list : list of dict
            Each element is the ``all_scm_outputs`` dict of a parent table.
        fk_ids : torch.Tensor
            Shape ``(seq_len, num_parents)``.  ``fk_ids[:, i]`` are the parent
            row indices (0-based) to use for parent ``i``.
        block_paths : torch.Tensor or None
            Shape ``(seq_len, max_levels)``. Hierarchical block path per child
            row from HSBM. Only used when ``use_signal_group_features=True``.
        """
        causes_raw = self.xsampler.sample()  # (seq_len, num_causes)
        if self.time_dim > 0:
            timestamps_norm, time_features = self._prepare_time_features(
                n=self.seq_len,
                t_min=getattr(self, "t_min", None),
            )
            causes = torch.cat([causes_raw, time_features], dim=-1)
        else:
            causes = causes_raw

        for i, parent_table_data in enumerate(parent_data_list):
            parent_idx = fk_ids[:, i]
            null_mask = (parent_idx < 0).unsqueeze(-1)
            safe_idx = parent_idx.clamp(min=0)
            parent_causes = parent_table_data[MASK_TYPE.CAUSAL_OUTPUT][safe_idx]
            parent_causes = parent_causes * (~null_mask).float()  # FK=-1 → zero
            causes = torch.cat([causes, parent_causes], dim=-1)

        assert causes.shape[0] == self.seq_len, (
            f"Expected {self.seq_len} rows, got {causes.shape[0]}"
        )
        assert causes.shape[1] == self.num_causes + self.time_dim + self.other_causes, (
            f"Expected {self.num_causes + self.time_dim + self.other_causes} input dims, "
            f"got {causes.shape[1]}"
        )

        # MLP forward pass
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]

        X, outputs_flat = self.handle_outputs(outputs, self.masks)

        if self.time_dim > 0:
            X[MASK_TYPE.TIMESTAMP] = timestamps_norm.unsqueeze(-1)

        for _, value in X.items():
            if torch.any(torch.isnan(value)):
                value[:] = 0.0

        if self.use_signal_group_features and not self._exp_skip_sg:
            # Save MLP original X before overwrite, for diagnostic comparison
            X_mlp = X[MASK_TYPE.X].clone()
            self._x_causal_corr_mlp = self._compute_x_causal_corr(
                X_mlp, X[MASK_TYPE.CAUSAL_OUTPUT],
            )
            X[MASK_TYPE.X_MLP] = X_mlp

            # --- Signal-group feature construction ---
            # Cache time features for _build_time_signal
            self._cached_time_features = time_features if self.time_dim > 0 else None

            time_sig_raw = self._build_time_signal()
            parent_sig_raw = self._build_parent_signal(parent_data_list, fk_ids)
            path_sig_raw = self._build_path_signal(block_paths)
            intrinsic_sig_raw = self._build_intrinsic_signal(X[MASK_TYPE.CAUSAL_OUTPUT])

            time_sig, parent_sig, path_sig, intrinsic_sig = self._normalize_signals(
                time_sig_raw, parent_sig_raw, path_sig_raw, intrinsic_sig_raw,
            )

            # Split time signal for feature construction
            signals = {
                "time_basis": time_sig[:, TIME_BASIS_DIMS[0]:TIME_BASIS_DIMS[1]],
                "time_gates": time_sig[:, TIME_GATE_DIMS[0]:TIME_GATE_DIMS[1]],
                "parent": parent_sig,
                "path": path_sig,
                "intrinsic": intrinsic_sig,
            }

            X_sg = self._construct_features(signals, self.residual_sigma)
            X_sg = self._apply_cross_feature_coupling(X_sg)
            X[MASK_TYPE.X] = X_sg

        elif not self.use_signal_group_features:
            # Old path: parent feature explicit injection
            if self.parent_injection_scale > 0:
                parent_parts = []
                for i, parent_data in enumerate(parent_data_list):
                    if MASK_TYPE.FULL in parent_data:
                        parent_full = parent_data[MASK_TYPE.FULL]
                        parent_idx = fk_ids[:, i]
                        null_mask = (parent_idx < 0).unsqueeze(-1)
                        safe_idx = parent_idx.clamp(min=0)
                        parent_feat = parent_full[safe_idx]
                        parent_feat = parent_feat * (~null_mask).float()
                        parent_parts.append(parent_feat)
                if parent_parts:
                    parent_flat = torch.cat(parent_parts, dim=-1)
                    n_feat = X[MASK_TYPE.X].shape[1]
                    D = parent_flat.shape[1]
                    for j in range(n_feat):
                        w = torch.randn(D, device=self.device)
                        w = w * (self.parent_injection_scale / (D**0.5))
                        X[MASK_TYPE.X][:, j] += parent_flat @ w

        # Diagnostic: per-feature max |pearsonr| with CAUSAL_OUTPUT
        self._x_causal_corr = self._compute_x_causal_corr(
            X[MASK_TYPE.X], X[MASK_TYPE.CAUSAL_OUTPUT])

        return X, fk_ids, outputs_flat

    def forward_with_enhanced_temporal_sampling(
        self, parent_data_list: List[Dict[str, torch.Tensor]]
    ):
        """Deprecated baseline (Pólya urn + mass reinforcer); kept for comparison only.

        The production path uses time-as-input via ``time_dim`` and
        ``forward_with_input`` / ``forward_without_input`` instead.

        Enhanced temporal sampling method implementing the 5-step process:
        1. Draw Timestamps from temporal function Λ(t)
        2. Initialize Objects (mass vectors and edge kernel)
        3. Nested Sampling Loop (for each timestamp)
        4. Reinforcer Update (update masses and edge kernel)
        5. Return completed child table

        Parameters
        ----------
        parent_data_list : List[torch.Tensor]
            List of exactly 2 parent tables.
            Each parent table is a tensor of shape (seq_len_i, num_features_i).

        Returns
        -------
        X : Dict[str, torch.Tensor]
            Generated features for the child table
        parent_idxes : torch.Tensor
            Indices of selected parent samples
        outputs_flat : torch.Tensor
            Full flattened outputs from all MLP layers
        """
        if self.temporal_vocab is None:
            self.temporal_vocab = TemporalVocab(device=self.device)

        assert (
            len(parent_data_list) == 2
        ), "This method requires exactly 2 parent tables"

        parent_P_data, parent_Q_data = parent_data_list

        # Step 1: Draw Timestamps from temporal function Λ(t)
        timestamps = self.temporal_vocab.sample(
            num_samples=self.seq_len,
            time_range=(0, 10),
        )

        # Step 2: Initialize Objects
        aug_emb_P, aug_emb_Q = self._generate_parent_embeddings(
            parent_P_data, parent_Q_data
        )
        self.temporal_vocab_p = TemporalVocab()
        self.temporal_vocab_q = TemporalVocab()
        self.temporal_vocab_p.generate(time_range=(0, 10))
        self.temporal_vocab_q.generate(time_range=(0, 10))
        self.temporal_vocab_p.norm_intensity()
        self.temporal_vocab_q.norm_intensity()

        # Initialize edge kernel E_ij = <e^P_i, e^Q_j>
        E = torch.matmul(aug_emb_P, aug_emb_Q.T)  # (n_P, n_Q)
        E = E.clamp(min=-10000.0, max=10000.0)

        if E.max() - E.min() != 0:
            E = (E - E.min()) / (E.max() - E.min())
        else:
            E = torch.ones_like(E)

        # Initialize mass vectors based on the degree of E.
        # Norm to mean 1, std 0.2. Clip to be non-negative and less than 10.
        m_P = E.sum(dim=1)
        m_Q = E.sum(dim=0)
        if m_P.max() - m_P.min() != 0:
            m_P = torch.clamp((m_P - m_P.mean()) / m_P.std() * 0.2 + 1, min=0.1, max=10)
        else:
            m_P = torch.ones_like(m_P)
        if m_Q.max() - m_Q.min() != 0:
            m_Q = torch.clamp((m_Q - m_Q.mean()) / m_Q.std() * 0.2 + 1, min=0.1, max=10)
        else:
            m_Q = torch.ones_like(m_Q)

        # Storage for results
        selected_P_indices = []
        selected_Q_indices = []
        all_child_features = []

        # Step 3: Nested Sampling Loop. Each with a batch of samples
        for batch_idx in range(0, self.seq_len, self.batch_size):
            if batch_idx + self.batch_size > self.seq_len:
                batch_timestamps = timestamps[batch_idx:]
            else:
                batch_timestamps = timestamps[batch_idx : batch_idx + self.batch_size]
            batch_size = batch_timestamps.shape[0]

            # 3.1: Pick a row from parent P: i ~ Cat(m^P)
            if m_P.sum() <= 0:
                P_probs = torch.full_like(m_P, 1.0 / m_P.numel())
            else:
                P_probs = m_P / m_P.sum()
            i = torch.multinomial(P_probs, num_samples=batch_size, replacement=True)

            # 3.2: Pick a row from parent Q conditioned on i: j ~ Cat(m^Q + η * E_{i•})
            if m_Q.max() - m_Q.min() != 0:
                norm_m_Q = (m_Q - m_Q.min()) / (m_Q.max() - m_Q.min())
            else:
                norm_m_Q = torch.full_like(m_Q, 1.0 / m_Q.numel())
            Q_weights = norm_m_Q + self.eta * E[i, :]
            row_sums = Q_weights.sum(dim=1, keepdim=True)
            zero_rows = row_sums.squeeze(1) <= 0
            safe_row_sums = torch.where(
                row_sums <= 0, torch.ones_like(row_sums), row_sums
            )
            Q_probs = Q_weights / safe_row_sums
            if zero_rows.any():
                Q_probs[zero_rows] = 1.0 / Q_weights.size(1)
            j = torch.multinomial(Q_probs, num_samples=1, replacement=True).squeeze(1)

            # 3.3: Generate child features via the child-table SCM
            # Create input causes for this sample
            cause_sample = self.xsampler.sample_batch(batch_size)
            parent_P_sample = parent_P_data[MASK_TYPE.CAUSAL_OUTPUT][i]
            parent_Q_sample = parent_Q_data[MASK_TYPE.CAUSAL_OUTPUT][j]

            # Concatenate all inputs
            combined_input = torch.cat(
                [cause_sample, parent_P_sample, parent_Q_sample], dim=-1
            )

            # Generate child features through SCM
            child_features = self._generate_batch_child_features(combined_input)

            # 3.4: Append (i, j, t_k, features) as a new child row
            selected_P_indices.extend(i.tolist())
            selected_Q_indices.extend(j.tolist())
            all_child_features.append(child_features)

            # Step 4: Reinforcer Update
            # Update masses with temporal decay factors
            beta_P_t = self._get_temporal_reinforcement(
                batch_timestamps, self.temporal_vocab_p.get_intensity
            )
            beta_Q_t = self._get_temporal_reinforcement(
                batch_timestamps, self.temporal_vocab_q.get_intensity
            )
            # beta_pair_t = self._get_temporal_reinforcement(t_k, self.beta_pair)

            if self.eta > 1:
                m_P[i] += (beta_P_t / self.eta).to(self.device)
                m_Q[j] += (beta_Q_t / self.eta).to(self.device)
            else:
                m_P[i] += (beta_P_t * (-4 * self.eta + 5)).to(self.device)
                m_Q[j] += (beta_Q_t * (-4 * self.eta + 5)).to(self.device)
            # E[i, j] += beta_pair_t

        # Step 5: Return completed child table
        # Combine all results
        all_child_features = torch.cat(all_child_features, dim=0)
        final_P_indices = torch.tensor(selected_P_indices, device=self.device)
        final_Q_indices = torch.tensor(selected_Q_indices, device=self.device)
        final_parent_idxes = torch.stack([final_P_indices, final_Q_indices], dim=1)

        # Create output dictionary
        X, outputs_flat = self.handle_outputs(
            all_child_features, self.masks, skip_concat=True
        )

        # Store the sampled timestamps if TIMESTAMP mask exists
        if MASK_TYPE.TIMESTAMP in self.masks:
            # Normalize timestamps to [0, 1] range for consistency with other features
            if timestamps.max() - timestamps.min() > 0:
                normalized_timestamps = (timestamps - timestamps.min()) / (
                    timestamps.max() - timestamps.min()
                )
            else:
                normalized_timestamps = torch.zeros_like(timestamps)

            # Store as 2D tensor (seq_len, 1)
            X[MASK_TYPE.TIMESTAMP] = normalized_timestamps.unsqueeze(1).to(self.device)

        # Check for NaNs and handle them by setting to default values
        for _, value in X.items():
            if torch.any(torch.isnan(value)):
                value[:] = 0.0

        return X, final_parent_idxes, outputs_flat

    # ``sample_final_output`` is removed — FK connections are now determined
    # externally via HSBM and passed as ``fk_ids`` to ``forward_with_input``.

    def handle_outputs(self, outputs, masks, skip_concat=False):
        """
        Handles outputs from the MLP layers.

        Parameters
        ----------
        outputs : list of torch.Tensor
            List of output tensors from MLP layers

        masks : dict of str -> list of int
            Dictionary of masks, each key is a string, each value is a list of integers representing the features to use

        Returns
        -------
        X : Dict of str -> torch.Tensor
            Input features (seq_len, num_features)
        outputs_flat : torch.Tensor
            Full flattened outputs from all MLP layers
        """
        X = {}
        if skip_concat:
            outputs_flat = outputs
        else:
            outputs_flat = torch.cat(outputs, dim=-1)
        for key, value in masks.items():
            X[key] = outputs_flat[:, value]

        return X, outputs_flat

    def _generate_parent_embeddings(self, *args):
        """
        Generate mass and augmentation embeddings for a parent table.

        Parameters
        ----------
        parent_data : torch.Tensor
            Parent table data of shape (n_samples, n_features)

        Returns
        -------
        mass_emb : torch.Tensor
            Mass embeddings of shape (n_samples, embedding_dim)
        aug_emb : torch.Tensor
            Augmentation embeddings of shape (n_samples, embedding_dim)
        """
        # Simple linear transformation to generate embeddings
        aug_emb = []
        for parent_data in args:
            # random pick number of self.embedding_dim idx from the parent_data[FULL]
            aug_idx = torch.randperm(parent_data[MASK_TYPE.FULL].shape[1])[
                : self.embedding_dim
            ]
            aug_emb.append(parent_data[MASK_TYPE.FULL][:, aug_idx])

        return aug_emb

    def _get_temporal_reinforcement(self, t: torch.Tensor, intensity: torch.Tensor):
        """
        Get temporal reinforcement factor β(t).

        Parameters
        ----------
        t : torch.Tensor
            Current timestamp
        intensity : torch.Tensor
            Intensity of the temporal distribution

        Returns
        -------
        beta_t : torch.Tensor
            Time-dependent reinforcement factor
        """
        beta_t = intensity[t]
        return beta_t

    def _generate_batch_child_features(self, combined_input: torch.Tensor):
        """
        Generate child features for a batch of samples using the SCM.

        Parameters
        ----------
        combined_input : torch.Tensor
            Combined input tensor with causes and parent data

        Returns
        -------
        features : torch.Tensor
            Generated child features for one batch of samples
        """
        # Generate outputs through MLP layers
        outputs = [combined_input]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]  # Skip first two layers

        # Flatten and extract features
        outputs_flat = torch.cat(outputs, dim=-1)

        return outputs_flat

    # ------------------------------------------------------------------
    # Phase 0 stubs — Signal-Group Feature Generation (Contract 0.4)
    # Implemented in Phase 1–4.
    # ------------------------------------------------------------------

    def _classify_archetype(self, params: dict) -> dict:
        """Extract archetype signals from table metadata."""
        return {
            "is_source": params.get("is_source", False),
            "is_timestamp": params.get("is_timestamp", False),
            "num_parents": params.get("num_parents", 0),
            "hsbm_locality": params.get("hsbm_locality", 0.0),
        }

    def _compute_base_alpha(self, archetype: dict) -> torch.Tensor:
        """Map archetype signals to base Dirichlet α (4-vector: time, parent, path, intrinsic).

        Source tables use only the intrinsic group (latent causes);
        timestamp children use time+parent+path;
        dependent non-ts tables use only parent+path (no time, no intrinsic).
        """
        is_source = archetype["is_source"]
        is_timestamp = archetype["is_timestamp"]
        num_parents = archetype["num_parents"]

        if is_source:
            # Source tables: 100% intrinsic from MLP CAUSAL_OUTPUT
            alpha = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        elif is_timestamp:
            # Timestamp child: time + parent + path
            alpha = torch.tensor([0.5, 0.25, 0.25, 0.0], device=self.device)
            if num_parents > 1:
                parent_share = min(0.6, max(0.25, num_parents * 0.10))
                alpha[1] = parent_share
                remain = 1.0 - alpha[1]
                alpha[0] = remain * 0.5
                alpha[2] = remain * 0.5
        else:
            # Dependent non-timestamp: parent + path only, no time weight
            alpha = torch.tensor([0.0, 0.6, 0.4, 0.0], device=self.device)
            if num_parents > 1:
                parent_share = min(0.6, max(0.6, num_parents * 0.10))
                alpha[1] = parent_share
                alpha[2] = 1.0 - parent_share

        # Mask unavailable groups
        if num_parents == 0:
            alpha[1] = 0.0  # parent group unavailable
            alpha[2] = 0.0  # path group unavailable (no HSBM)
        if not is_timestamp and self.time_dim == 0:
            alpha[0] = 0.0  # time group unavailable

        # Re-normalize among active groups
        total = alpha.sum()
        if total > 0:
            alpha = alpha / total
        else:
            alpha = torch.zeros(4, device=self.device)

        return alpha

    def _perturb_alpha(self, alpha_base: torch.Tensor) -> torch.Tensor:
        """Apply multiplicative log-normal perturbation to base α, re-normalize.

        Only active groups (α > 0) get perturbed; masked groups stay at 0.
        """
        active_mask = alpha_base > 0
        if active_mask.sum() == 0:
            return alpha_base.clone()

        n_groups = len(alpha_base)
        eta = torch.randn(n_groups, device=self.device) * self.archetype_perturb_std
        alpha_perturbed = alpha_base.clone()
        alpha_perturbed[active_mask] = alpha_base[active_mask] * torch.exp(
            eta[active_mask],
        )
        alpha_perturbed = alpha_perturbed / alpha_perturbed.sum()
        return alpha_perturbed

    def _sample_feature_groups(
        self, alpha_final: torch.Tensor, n_features: int,
    ) -> list[dict]:
        """Sample per-feature group assignments from Dirichlet(α_final).

        Returns list of dicts, one per feature, with keys: groups, basis_indices,
        signs, magnitudes.
        """
        GROUP_NAMES = ["time", "parent", "path", "intrinsic"]
        # K_g = min(3, max(1, ceil(S_g / divisor))). With divisor=8→K=2.
        # When divisor ≥ S_g → K=1, forcing all features in a group
        # to share the same basis vector (max within-group correlation).
        DIV = self.basis_group_divisor
        TIME_S = 8; PARENT_S = 12; PATH_S = 8; INT_S = self.intrinsic_dim
        K_PER_GROUP = {
            "time": min(3, max(1, int(np.ceil(TIME_S / DIV)))),
            "parent": max(1, int(np.ceil(PARENT_S / 3))),
            "path": max(1, int(np.ceil(PATH_S / 2))),
            "intrinsic": max(1, int(np.ceil(INT_S / 2))),
        }
        self._K_PER_GROUP = K_PER_GROUP.copy()  # save for diagnostic dump

        alpha_np = alpha_final.detach().cpu().numpy()
        feature_assignments = []

        # Per-group basis-index counters for round-robin allocation
        basis_counters = {g: 0 for g in GROUP_NAMES}
        # Per-basis-index sign, fixed per (group, basis_idx) across features
        basis_signs: dict[str, list[float]] = {}
        for g in GROUP_NAMES:
            K_g = K_PER_GROUP[g]
            basis_signs[g] = [1.0 if np.random.random() > 0.5 else -1.0 for _ in range(K_g)]

        for ft_idx in range(n_features):
            # Sample group weights from Dirichlet
            group_weights = np.random.dirichlet(alpha_np + 1e-6)

            # Select top-K groups among active ones
            active_indices = np.where(alpha_np > 0)[0]
            if len(active_indices) == 0:
                feature_assignments.append({
                    "groups": [],
                    "basis_indices": {},
                    "signs": {},
                    "magnitudes": {},
                })
                continue

            active_weights = group_weights[active_indices]
            k = min(self.max_groups_per_feature, len(active_indices))
            top_k_local = np.argsort(active_weights)[-k:]
            top_k_global = active_indices[top_k_local]

            fa = {
                "groups": [],
                "basis_indices": {},
                "signs": {},
                "magnitudes": {},
            }

            for g_idx in top_k_global:
                g_name = GROUP_NAMES[g_idx]
                K_g = K_PER_GROUP[g_name]

                # Round-robin basis index — ensures uniform sharing across features
                n_basis = min(K_g, 1)
                basis_idx = basis_counters[g_name] % K_g
                basis_counters[g_name] += 1

                # Shared sign per (group, basis_idx) — features with same basis get SAME sign
                sign_val = basis_signs[g_name][basis_idx]

                # Magnitude ~ LogNormal with independent per-feature noise
                mag_val = float(np.random.lognormal(
                    mean=self.loading_log_mean, sigma=self.loading_sigma,
                ))

                # Per-feature-basis fixed perturbation vector (unit-norm, scaled by eta)
                signal_dim_map = {"time": 8, "parent": 12, "path": 8}
                sig_dim = signal_dim_map.get(g_name, 8)
                pert_vec = np.random.randn(sig_dim).astype(np.float32)
                pert_vec = pert_vec / (np.linalg.norm(pert_vec) + 1e-8) * self.basis_perturb_eta

                fa["groups"].append(g_name)
                fa["basis_indices"][g_name] = [basis_idx]
                fa["signs"][g_name] = [sign_val]
                fa["magnitudes"][g_name] = [mag_val]
                fa.setdefault("perturbations", {})[g_name] = [pert_vec]

            feature_assignments.append(fa)

        return feature_assignments

    def _init_group_bases(self) -> dict[str, dict[str, torch.Tensor]]:
        """Initialize subspace-orthogonal bases.

        Each group's signal is split into K equal subspaces. Basis k only
        projects from its own subspace, guaranteeing zero cross-basis
        correlation from the signal. Cross-basis coupling is then
        controlled solely by the coupling layer.
        """
        DIV = self.basis_group_divisor
        TIME_S = 8; PARENT_S = 12; PATH_S = 8; INT_S = self.intrinsic_dim
        K_time = min(3, max(1, int(np.ceil(TIME_S / DIV))))
        K_parent = max(1, int(np.ceil(PARENT_S / 3)))
        K_path = max(1, int(np.ceil(PATH_S / 2)))
        K_intrinsic = max(1, int(np.ceil(INT_S / 2)))

        group_bases: dict[str, dict[str, torch.Tensor]] = {}
        self._basis_families: dict[str, list[int]] = {}
        # Map: group → per-basis subspace slice (start, end)
        self._basis_subspaces: dict[str, list[tuple[int, int]]] = {}

        def _make_subspace_bases(K: int, D: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
            """Generate K unit-norm basis vectors from disjoint subspaces."""
            bases = torch.zeros(K, D, device=self.device)
            subspaces: list[tuple[int, int]] = []
            for k in range(K):
                start = k * D // K
                end = (k + 1) * D // K
                subspaces.append((start, end))
                sub_D = end - start
                b_sub = torch.randn(sub_D, device=self.device)
                bases[k, start:end] = b_sub / b_sub.norm()
            return bases, subspaces

        B_time_basis, subspaces_time = _make_subspace_bases(K_time, TIME_S)
        B_time_gate, subspaces_time_gate = _make_subspace_bases(K_time, 3)
        group_bases["time"] = {"basis": B_time_basis, "gate": B_time_gate}
        self._basis_families["time"] = list(range(K_time))
        self._basis_subspaces["time_basis"] = subspaces_time
        self._basis_subspaces["time_gate"] = subspaces_time_gate

        B_parent, subspaces_parent = _make_subspace_bases(K_parent, PARENT_S)
        group_bases["parent"] = {"basis": B_parent}
        self._basis_families["parent"] = list(range(K_parent))
        self._basis_subspaces["parent"] = subspaces_parent

        B_path, subspaces_path = _make_subspace_bases(K_path, PATH_S)
        group_bases["path"] = {"basis": B_path}
        self._basis_families["path"] = list(range(K_path))
        self._basis_subspaces["path"] = subspaces_path

        B_intrinsic, subspaces_intrinsic = _make_subspace_bases(K_intrinsic, INT_S)
        group_bases["intrinsic"] = {"basis": B_intrinsic}
        self._basis_families["intrinsic"] = list(range(K_intrinsic))
        self._basis_subspaces["intrinsic"] = subspaces_intrinsic

        return group_bases

    def _init_percol_mlps(self, masks: dict) -> None:
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

        # Resolve activation functions
        act_choices = hp.get("percol_mlp_activation", get_activations(
            random=True, scale=True, diverse=True,
        ))
        if isinstance(act_choices, list):
            activations = [act() if callable(act) else act for act in act_choices]
        elif callable(act_choices):
            activations = [act_choices() for _ in range(num_layers - 1)]
        else:
            activations = [act_choices] * (num_layers - 1)

        # Pad/trim activations to match num_layers - 1
        while len(activations) < num_layers - 1:
            activations.append(activations[-1])
        activations = activations[:num_layers - 1]

        # Compute input dimension from archetype
        ap = self.archetype_params or {}
        is_source = ap.get("is_source", False)
        is_timestamp = ap.get("is_timestamp", False)
        dim = 0
        if is_timestamp:
            dim += 11  # time_basis(8) + time_gates(3)
        if not is_source:
            dim += 12 + 8  # parent_sig(12) + path_sig(8)
        dim += self.intrinsic_dim  # intrinsic always present
        self._percol_input_dim = dim

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

    def _init_parent_projectors(
        self,
        relation_keys: list[RelationKey],
        parent_causal_dims: list[int],
    ) -> None:
        """Create per-edge Linear projectors: parent CAUSAL_OUTPUT → 12-dim.

        Stored in ``self.parent_projectors`` (nn.ModuleDict, keyed by RelationKey).
        """
        for rk, parent_dim in zip(relation_keys, parent_causal_dims):
            self.parent_projectors[str(rk)] = nn.Linear(parent_dim, 12).to(
                self.device,
            )

    def _init_path_encoders(
        self,
        relation_keys: list[RelationKey],
        levels_per_edge: list[int],
    ) -> None:
        """Create per-edge PathEncoder instances.

        Stored in ``self.path_encoders`` (nn.ModuleDict, keyed by RelationKey).
        """
        MAX_BLOCKS = 4  # max_blocks_per_level from DEFAULT_HSBM_HP max
        for rk, n_lv in zip(relation_keys, levels_per_edge):
            self.path_encoders[str(rk)] = PathEncoder(
                num_levels=n_lv,
                max_blocks_per_level=MAX_BLOCKS,
                out_dim=8,
                embed_dim=4,
            ).to(self.device)

    def _build_time_signal(self) -> torch.Tensor:
        """Return cached time features from the most recent forward call.

        Returns ``(seq_len, 11)`` tensor, or zeros if no time_dim.
        """
        if hasattr(self, "_cached_time_features") and self._cached_time_features is not None:
            return self._cached_time_features
        return torch.zeros(self.seq_len, 11, device=self.device)

    def _build_parent_signal(
        self,
        parent_data_list: list,
        fk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-edge projected parent signal, mean-pool → (seq_len, 12).

        Returns zeros if no parents or no projectors registered.
        """
        num_parents = fk_ids.shape[1] if fk_ids.dim() > 1 and fk_ids.shape[1] > 0 else 0
        if num_parents == 0 or len(self.parent_projectors) == 0:
            return torch.zeros(self.seq_len, 12, device=self.device)

        edge_outputs = []
        for i in range(num_parents):
            parent_data = parent_data_list[i]
            parent_idx = fk_ids[:, i]
            null_mask = (parent_idx < 0).unsqueeze(-1)
            safe_idx = parent_idx.clamp(min=0)
            parent_rows = parent_data[MASK_TYPE.CAUSAL_OUTPUT][safe_idx]
            parent_rows = parent_rows * (~null_mask).float()  # FK=-1 → zero
            projector = list(self.parent_projectors.values())[i]
            edge_outputs.append(projector(parent_rows))

        if not edge_outputs:
            return torch.zeros(self.seq_len, 12, device=self.device)

        return torch.stack(edge_outputs, dim=0).mean(dim=0)

    def _build_path_signal(
        self,
        block_paths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build per-edge path embedding signal, mean-pool → (seq_len, 8).

        Returns zeros if block_paths is None or no path_encoders registered.
        """
        if block_paths is None or len(self.path_encoders) == 0:
            return torch.zeros(self.seq_len, 8, device=self.device)

        if not isinstance(block_paths, torch.Tensor):
            block_paths = torch.tensor(block_paths, dtype=torch.long, device=self.device)
        else:
            block_paths = block_paths.to(device=self.device, dtype=torch.long)

        edge_outputs = []
        for key, encoder in self.path_encoders.items():
            n_lv = encoder.num_levels
            edge_path = block_paths[:, :n_lv]  # prefix up to this edge's depth
            edge_outputs.append(encoder(edge_path))  # (seq_len, 8)

        if not edge_outputs:
            return torch.zeros(self.seq_len, 8, device=self.device)

        # Mean-pool across edges (single edge → identity)
        return torch.stack(edge_outputs, dim=0).mean(dim=0)

    def _build_intrinsic_signal(self, causes: torch.Tensor) -> torch.Tensor:
        """Project CAUSAL_OUTPUT (MLP intermediate) to intrinsic signal → (seq_len, intrinsic_dim).

        The intrinsic signal captures the table's own latent variation,
        independent of time, parent, or path signals.
        """
        return self.intrinsic_projector(causes)

    def _normalize_signals(
        self,
        time_sig: torch.Tensor,
        parent_sig: torch.Tensor,
        path_sig: torch.Tensor,
        intrinsic_sig: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply group-specific RMS normalization + scaling.

        - time_basis[:, 0:8]: RMS-norm × group_scale_time
        - time_gates[:, 8:11]: AS-IS (no RMS, no scale)
        - parent: RMS-norm × group_scale_parent
        - path: RMS-norm × group_scale_path
        - intrinsic: RMS-norm × group_scale_intrinsic
        """
        # Time: split at hard boundary
        time_basis = time_sig[:, TIME_BASIS_DIMS[0]:TIME_BASIS_DIMS[1]]
        time_gates = time_sig[:, TIME_GATE_DIMS[0]:TIME_GATE_DIMS[1]]

        rms_basis = torch.sqrt(torch.mean(time_basis ** 2))
        if rms_basis > 1e-8:
            time_basis_norm = time_basis / rms_basis * self.group_scale_time
        else:
            time_basis_norm = time_basis * 0.0
        time_out = torch.cat([time_basis_norm, time_gates], dim=-1)

        # Parent: RMS norm
        rms_parent = torch.sqrt(torch.mean(parent_sig ** 2))
        if rms_parent > 1e-8:
            parent_out = parent_sig / rms_parent * self.group_scale_parent
        else:
            parent_out = parent_sig * 0.0

        # Path: RMS norm
        rms_path = torch.sqrt(torch.mean(path_sig ** 2))
        if rms_path > 1e-8:
            path_out = path_sig / rms_path * self.group_scale_path
        else:
            path_out = path_sig * 0.0

        # Intrinsic: RMS norm
        if intrinsic_sig is None:
            intrinsic_out = torch.zeros(
                time_sig.shape[0], self.intrinsic_dim, device=self.device,
            )
        else:
            rms_int = torch.sqrt(torch.mean(intrinsic_sig ** 2))
            if rms_int > 1e-8:
                intrinsic_out = intrinsic_sig / rms_int * self.group_scale_intrinsic
            else:
                intrinsic_out = intrinsic_sig * 0.0

        return time_out, parent_out, path_out, intrinsic_out

    def _perturb_basis(self, B_k: torch.Tensor, pert_vec) -> torch.Tensor:
        """Return re-normalized B_k + pert_vec if pert_vec is non-zero, else B_k."""
        if pert_vec is None:
            return B_k
        if isinstance(pert_vec, np.ndarray):
            if not np.any(pert_vec != 0):
                return B_k
            pert_t = torch.from_numpy(pert_vec).to(self.device)
        elif isinstance(pert_vec, torch.Tensor):
            if not pert_vec.any():
                return B_k
            pert_t = pert_vec.to(self.device)
        else:
            return B_k
        B_eff = B_k + pert_t
        return B_eff / (B_eff.norm() + 1e-8)

    def _construct_features(
        self,
        signals: dict,
        residual_sigma: float,
    ) -> torch.Tensor:
        """Construct feature matrix X from normalized signals + assignments.

        Returns (seq_len, n_features) tensor where n_features = len(self.feature_assignments).
        """
        n_features = len(self.feature_assignments)
        n_feat = n_features
        if n_feat == 0:
            n_feat = 12  # fallback default
        X = torch.zeros(self.seq_len, n_feat, device=self.device)

        for j, fa in enumerate(self.feature_assignments):
            if j >= X.shape[1]:
                break
            for g_name in fa["groups"]:
                basis_indices = fa["basis_indices"].get(g_name, [])
                signs = fa["signs"].get(g_name, [])
                magnitudes = fa["magnitudes"].get(g_name, [])
                perts = fa.get("perturbations", {}).get(g_name, [])

                if g_name == "time":
                    basis_sig = signals["time_basis"]  # (seq_len, 8)
                    gate_sig = signals.get("time_gates")  # (seq_len, 3)
                    B_basis = self.group_bases["time"]["basis"]  # (K, 8)
                    B_gate = self.group_bases["time"]["gate"]  # (K, 3)
                    for idx_in_list, k in enumerate(basis_indices):
                        B_eff = self._perturb_basis(
                            B_basis[k], perts[idx_in_list] if idx_in_list < len(perts) else None)
                        proj = basis_sig @ B_eff
                        if gate_sig is not None:
                            B_gate_eff = self._perturb_basis(
                                B_gate[k], None)  # gates not perturbed
                            proj += gate_sig @ B_gate_eff
                        X[:, j] += signs[idx_in_list] * magnitudes[idx_in_list] * proj

                elif g_name == "parent":
                    parent_sig = signals["parent"]
                    B_parent = self.group_bases["parent"]["basis"]
                    for idx_in_list, k in enumerate(basis_indices):
                        B_eff = self._perturb_basis(
                            B_parent[k], perts[idx_in_list] if idx_in_list < len(perts) else None)
                        proj = parent_sig @ B_eff
                        X[:, j] += signs[idx_in_list] * magnitudes[idx_in_list] * proj

                elif g_name == "path":
                    path_sig = signals["path"]
                    B_path = self.group_bases["path"]["basis"]
                    for idx_in_list, k in enumerate(basis_indices):
                        B_eff = self._perturb_basis(
                            B_path[k], perts[idx_in_list] if idx_in_list < len(perts) else None)
                        proj = path_sig @ B_eff
                        X[:, j] += signs[idx_in_list] * magnitudes[idx_in_list] * proj

                elif g_name == "intrinsic":
                    int_sig = signals["intrinsic"]
                    B_int = self.group_bases["intrinsic"]["basis"]
                    for idx_in_list, k in enumerate(basis_indices):
                        B_eff = self._perturb_basis(
                            B_int[k], perts[idx_in_list] if idx_in_list < len(perts) else None)
                        proj = int_sig @ B_eff
                        X[:, j] += signs[idx_in_list] * magnitudes[idx_in_list] * proj

            # Add per-feature residual noise (ε_j ~ N(0, σ²_res))
            if residual_sigma > 0:
                X[:, j] += torch.randn(self.seq_len, device=self.device) * residual_sigma

        return X

    def _init_cross_feature_coupling(self, n_features: int) -> None:
        """Initialize uniform off-diagonal coupling.

        W = (1 − I) / (n−1). Each column gets λ · mean(all other columns).
        This is the simplest controllable mixing: λ directly sets the
        fraction of other-feature signal added to each feature.
        """
        if self.coupling_lambda <= 0 or self.coupling_rank <= 0:
            self._coupling_W: torch.Tensor | None = None
            return

        nf = n_features
        W = torch.ones(nf, nf, device=self.device) / (nf - 1)
        W = W - torch.diag(torch.diag(W))
        self._coupling_W = W

    def _apply_cross_feature_coupling(
        self, X_struct: torch.Tensor,
    ) -> torch.Tensor:
        """Apply: X_out = X_struct + λ · (X_struct @ W)."""
        if (not hasattr(self, "_coupling_W") or self._coupling_W is None
                or self.coupling_lambda <= 0):
            return X_struct
        return X_struct + self.coupling_lambda * (X_struct @ self._coupling_W)

    def _compute_x_causal_corr(
        self, x: torch.Tensor, causal: torch.Tensor,
    ) -> dict:
        """Compute per-feature max |pearsonr| with any CAUSAL_OUTPUT dimension.

        Returns a dict with per-feature stats for diagnostic comparison.
        """
        x_np = x.detach().cpu().numpy()  # (seq_len, n_features)
        causal_np = causal.detach().cpu().numpy()  # (seq_len, num_outputs)

        n_feat = x_np.shape[1]
        per_feat_corr = []
        for j in range(n_feat):
            xj = x_np[:, j]
            # Max |pearsonr| across all causal output dims
            max_corr = 0.0
            for c in range(causal_np.shape[1]):
                xc = causal_np[:, c]
                # Skip constant columns
                if xj.std() < 1e-8 or xc.std() < 1e-8:
                    continue
                corr = abs(float(np.corrcoef(xj, xc)[0, 1]))
                if corr > max_corr:
                    max_corr = corr
            per_feat_corr.append(max_corr)

        arr = np.array(per_feat_corr)
        return {
            "per_feature": arr.tolist(),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "mode": "skip_sg" if self._exp_skip_sg else "signal_group",
        }

    def get_feature_diagnostics(self) -> dict:
        """Return feature assignment metadata for post-hoc analysis."""
        K = getattr(self, "_K_PER_GROUP", None)
        if K is None:
            DIV = self.basis_group_divisor
            TIME_S = 8; PARENT_S = 12; PATH_S = 8; INT_S = self.intrinsic_dim
            K = {
                "time": min(3, max(1, int(np.ceil(TIME_S / DIV)))),
                "parent": max(1, int(np.ceil(PARENT_S / 3))),
                "path": max(1, int(np.ceil(PATH_S / 2))),
                "intrinsic": max(1, int(np.ceil(INT_S / 2))),
            }
        return {
            "n_features": len(self.feature_assignments),
            "K_per_group": K,
            "alpha_final": self.alpha_final.tolist() if self.alpha_final is not None else None,
            "feature_assignments": [
                {
                    "groups": fa["groups"],
                    "basis_indices": {g: fa["basis_indices"].get(g, []) for g in fa["groups"]},
                }
                for fa in self.feature_assignments
            ],
            "coupling_lambda": self.coupling_lambda,
            "basis_perturb_eta": self.basis_perturb_eta,
            "group_scales": {
                "time": self.group_scale_time,
                "parent": self.group_scale_parent,
                "path": self.group_scale_path,
                "intrinsic": self.group_scale_intrinsic,
            },
            "residual_sigma": self.residual_sigma,
            "x_causal_corr": getattr(self, "_x_causal_corr", None),
            "x_causal_corr_mlp": getattr(self, "_x_causal_corr_mlp", None),
            "mode": "skip_sg" if self._exp_skip_sg else "signal_group",
        }


if __name__ == "__main__":
    # Example with uniform sampling (default behavior)
    model_uniform = MLPSCM(
        seq_len=16,
        num_outputs=10,
        is_causal=True,
        num_causes=10,
        other_causes=0,
        sampling_ratio=1.0,
        masks={
            MASK_TYPE.X: 10,
            MASK_TYPE.EDGE_PROB: 1,  # Add edge_prob for sampling
        },
        parent_sampling_dist="uniform",  # Uniform distribution
    )

    # Example with Zipf distribution (long-tail distribution)
    # Higher alpha values create heavier tails (more concentration on small indices)
    model_zipf = MLPSCM(
        seq_len=16,
        num_outputs=10,
        is_causal=True,
        num_causes=10,
        other_causes=0,
        sampling_ratio=1.0,
        masks={
            MASK_TYPE.X: 10,
            MASK_TYPE.EDGE_PROB: 1,
        },
        parent_sampling_dist="zipf",  # Zipf distribution for long-tail sampling
        parent_sampling_alpha=2.0,  # Alpha > 1 required; typical range: 1.5-4.0
    )
