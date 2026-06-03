import torch.nn as nn
from src.prior.activations import get_activations

# Fixed time-as-input dimension: trend(2) + seasonal(4) + spike(2) + gates(3) = 11.
# Must match the temporal pipeline in mlp_scm.py and table_generation.py.
TIME_DIM = 11

"""
    MLP SCM Config
    --------------
    mlp_activations : default=nn.Tanh
        The activation function to be used after each linear transformation
        in the MLP layers (except the first).

    init_std : float, default=1.0
        The standard deviation of the normal distribution used for initializing
        the weights of the MLP's linear layers.

    block_wise_dropout : bool, default=True
        Specifies the weight initialization strategy.
        - If `True`, uses a 'block-wise dropout' initialization where only random
          blocks within the weight matrix are initialized with values drawn from
          a normal distribution (scaled by `init_std` and potentially dropout),
          while the rest are zero. This encourages sparsity.
        - If `False`, uses standard normal initialization for all weights, followed
          by applying dropout mask based on `mlp_dropout_prob`.

    mlp_dropout_prob : float, default=0.1
        The dropout probability applied to weights during *standard* initialization
        (i.e., when `block_wise_dropout=False`). Ignored if
        `block_wise_dropout=True`. The probability is clamped between 0 and 0.99.

    scale_init_std_by_dropout : bool, default=True
        Whether to scale the `init_std` during weight initialization to compensate
        for the variance reduction caused by dropout. If `True`, `init_std` is
        divided by `sqrt(1 - dropout_prob)` or `sqrt(keep_prob)` depending on the
        initialization method.

    sampling : str, default="normal"
        The method used by `XSampler` to generate the initial 'cause' variables.
        Options:
        - "normal": Standard normal distribution (potentially with pre-sampled stats).
        - "uniform": Uniform distribution between 0 and 1.
        - "mixed": A random combination of normal, multinomial (categorical),
          Zipf (power-law), and uniform distributions across different cause variables.

    pre_sample_cause_stats : bool, default=False
        If `True` and `sampling="normal"`, the mean and standard deviation for
        each initial cause variable are pre-sampled. Passed to `XSampler`.

    noise_std : float, default=0.01
        The base standard deviation for the Gaussian noise added after each MLP
        layer's linear transformation (except the first layer).

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.

"""

DEFAULT_MLP_SCM_CONFIG = {
    "mlp_activations": nn.Tanh,
    "init_std": 1.0,
    "block_wise_dropout": False,
    "mlp_dropout_prob": 0.03,
    "scale_init_std_by_dropout": False,
    "sampling": "normal",
    "pre_sample_cause_stats": False,
    "noise_std": 0.01,
    "pre_sample_noise_std": False,
}


DEFAULT_FIXED_HP = {
    # SCMPrior
    "mix_probs": (0.7, 0.3),
    # TreeSCM
    "tree_model": "xgboost",
    "tree_depth_lambda": 0.5,
    "tree_n_estimators_lambda": 0.5,
    # Reg2Cls
    "balanced": False,
    "multiclass_ordered_prob": 0.0,
    "cat_prob": 0.2,
    "max_categories": float("inf"),
    "scale_by_max_features": False,
    "permute_features": True,
    "permute_labels": True,
}

DEFAULT_SAMPLED_HP = {
    # # Reg2Cls
    # "multiclass_type": {
    #     "distribution": "meta_choice",
    #     "choice_values": ["value", "rank"],
    # },
    # MLPSCM
    "mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
    "block_wise_dropout": {
        "distribution": "meta_choice",
        "choice_values": [False],
    },
    "mlp_dropout_prob": {
        "distribution": "uniform",
        "min": 0.01,
        "max": 0.05,
    },
    # MLPSCM and TreeSCM
    # "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
    "is_causal": {
        "distribution": "meta_choice",
        "choice_values": [True],
    },  # Always True now
    "num_causes": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 3,
        "round": True,
        "lower_bound": 1,
    },
    "num_outputs": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 3,
        "round": True,
        "lower_bound": 1,
    },
    # "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "y_is_effect": {
        "distribution": "meta_choice",
        "choice_values": [True],
    },  # Always True now
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
    # "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sort_features": {
        "distribution": "meta_choice",
        "choice_values": [True],
    },  # Always True now
    "num_layers": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 3,
        "round": True,
        "lower_bound": 2,
    },
    "hidden_dim": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 48,
        "min_mean": 6,
        "round": True,
        "lower_bound": 4,
    },
    "init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 0.0001,
        "round": False,
        "lower_bound": 0.0,
    },
    "sampling": {
        "distribution": "meta_choice",
        "choice_values": ["normal", "mixed", "uniform"],
        # "choice_values": ["normal", "uniform"],
        # "choice_values": ["normal"],
    },
    "pre_sample_cause_stats": {
        "distribution": "meta_choice",
        "choice_values": [False],  # Always False now
    },
    "pre_sample_noise_std": {
        "distribution": "meta_choice",
        "choice_values": [False],  # Always False now
    },
    "eta": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 5.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
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
    # --- Phase 2-3 temporal params ---
    "gamma_tier": {
        "distribution": "meta_choice",
        "choice_values": [0.0, 0.5, 1.5, 3.0],
    },
    "p_sort": {
        "distribution": "uniform",
        "min": 0.3,
        "max": 0.8,
    },
    # --- Parent feature injection (方案 B, Task 2.3) ---
    # Guarded behind use_signal_group_features=False (old path).
    "parent_injection_scale": {
        "distribution": "uniform",
        "min": 0.05,
        "max": 0.25,
    },
    # --- Signal-group feature generation (replaces parent_injection_scale when active) ---
    "use_signal_group_features": {
        "distribution": "meta_choice",
        "choice_values": [True],
    },
    "archetype_perturb_std": {
        "distribution": "uniform",
        "min": 0.05,
        "max": 0.15,
    },
    "max_groups_per_feature": {
        "distribution": "meta_choice",
        "choice_values": [2, 3],
    },
    "loading_sigma": {
        "distribution": "uniform",
        "min": 0.2,
        "max": 0.6,
    },
    "loading_log_mean": {
        "distribution": "uniform",
        "min": 0.35,
        "max": 1.7,
    },
    "basis_group_divisor": {
        "distribution": "meta_choice",
        "choice_values": [3],
    },
    # Cross-feature coupling (low-rank)
    "coupling_rank": {
        "distribution": "meta_choice",
        "choice_values": [2, 3],
    },
    "coupling_lambda": {
        "distribution": "uniform",
        "min": 0.02,
        "max": 0.06,
    },
    "residual_sigma": {
        "distribution": "uniform",
        "min": 0.1,
        "max": 0.4,
    },
    "basis_perturb_eta": {
        "distribution": "meta_choice",
        "choice_values": [0.10],
    },
    "num_basis_families": {
        "distribution": "meta_choice",
        "choice_values": [2, 3],
    },
    "basis_family_rho": {
        "distribution": "uniform",
        "min": 0.5,
        "max": 0.85,
    },
    "group_scale_time": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 15.0,
        "min_mean": 3.0,
        "round": False,
        "lower_bound": 0.5,
    },
    "group_scale_parent": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 15.0,
        "min_mean": 3.0,
        "round": False,
        "lower_bound": 0.5,
    },
    "group_scale_path": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 15.0,
        "min_mean": 3.0,
        "round": False,
        "lower_bound": 0.5,
    },
    "group_scale_intrinsic": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 15.0,
        "min_mean": 3.0,
        "round": False,
        "lower_bound": 0.5,
    },
    # --- Deprecated: kept for backward compat, ignored by HSBM path ---
    "parent_sampling_dist": {
        "distribution": "meta_choice",
        "choice_values": ["uniform", "zipf"],
    },
    "parent_sampling_alpha": {
        "distribution": "uniform",
        "max": 2.0,
        "min": 1.1,
    },
}

# Calendar-aligned Fourier seasonality + EventCalendar hyperparameters.
# Sampled per-table (modulation) and per-RDB (event calendar).
TEMPORAL_FOURIER_HP = {
    # Per-table modulation weights
    "m_week": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
    "m_month": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.2,
    },
    "m_year": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.3,
    },
    # Trend params
    "m_lin": {
        "distribution": "normal",
        "mean": 0.0,
        "std": 0.3,
    },
    "c_lin": {
        "distribution": "normal",
        "mean": 0.0,
        "std": 0.1,
    },
    # Event calendar (per-RDB)
    "event_H_min": 3,
    "event_H_max": 8,
    "event_importance_log_mu": -0.2,
    "event_importance_log_sigma": 0.5,
    "event_importance_clip_min": 0.2,
    "event_importance_clip_max": 2.5,
    "event_sigma_min": 1.0,
    "event_sigma_max": 3.0,
    # Per-table event sensitivity
    "table_event_p": 0.4,
    "table_sens_beta_alpha": 1.0,
    "table_sens_beta_beta": 4.0,
    # Noise
    "noise_std_log_min": -3.0,
    "noise_std_log_max": -1.0,
}


# HSBM FK generation hyperparameters — sampled per parent relation independently.
DEFAULT_HSBM_HP = {
    "hsbm_num_levels": {
        "distribution": "meta_choice",
        "choice_values": [1, 2, 3, 4, 5],
    },
    "hsbm_clusters_per_level": {
        "distribution": "meta_choice",
        "choice_values": [1, 2, 3],
    },
    "propensity_rho": {
        "distribution": "uniform",
        "min": 0.05,
        "max": 0.20,
    },
    "propensity_beta": {
        "distribution": "uniform",
        "min": 2.0,
        "max": 8.0,
    },
    "matching_latent_dim": {
        "distribution": "meta_choice",
        "choice_values": [2, 3, 4],
    },
    "matching_temperature": {
        "distribution": "meta_choice",
        "choice_values": [0.10, 0.20, 0.35, 0.50],
    },
}
