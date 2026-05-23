"""Task quality checker for complex tasks.

Stage A (ultra-cheap, rules-only, no sklearn):
  a1. label_nunique == 1
  a2. pos_ratio < 0.05 or > 0.95
  a3. n < 64 or min(n_pos, n_neg) < 8
  a4. zero feature_ columns or all constant
  a5. structural independence (schema-based, best-effort)
  a6. extreme child/parent size ratio (> 100:1 or < 1:100)

Stage B (model-based):
  b1. Lightweight ExtraTrees OOF AUC
  b2. Bootstrap CI for gray-zone tasks (0.50 < OOF AUC < 0.60)
  b3. Leakage check on FK/ID/timestamp columns

Reference: TabLeak (https://arxiv.org/html/2602.11139v1)

Usage from pipeline (in-memory, no disk I/O):
    from .task_quality import diagnose_dataframe
    result = diagnose_dataframe(df, schema_data=...)

Usage from CLI:
    from .task_quality import diagnose_parquet_path
    result = diagnose_parquet_path("/path/to/train.parquet")
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── feature quality metrics ────────────────────────────────────────────────

def compute_feature_stats(df: pd.DataFrame) -> dict:
    """Compute per-feature and pairwise-feature statistics.

    Returns dict with keys: avg_abs_corr_p50, feature_var_p50, feature_var_mean,
    n_features, avg_abs_corr_mean.
    """
    feature_cols = _get_feature_cols(df)
    if len(feature_cols) < 2:
        return {
            "avg_abs_corr_p50": float("nan"),
            "avg_abs_corr_mean": float("nan"),
            "feature_var_p50": float("nan"),
            "feature_var_mean": float("nan"),
            "n_features": len(feature_cols),
        }

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Per-feature variance
    feats_var = X.var(ddof=1).dropna()
    var_p50 = float(np.median(feats_var.values)) if len(feats_var) > 0 else float("nan")
    var_mean = float(np.mean(feats_var.values)) if len(feats_var) > 0 else float("nan")

    # Pairwise absolute correlation (use at most 50 columns to stay cheap)
    cols = feature_cols[:50]
    X_sub = X[cols].values.astype(np.float64)
    X_sub = np.where(np.isfinite(X_sub), X_sub, 0.0)

    # Remove constant columns
    stds = np.std(X_sub, axis=0)
    var_mask = stds > 1e-10
    if var_mask.sum() < 2:
        return {
            "avg_abs_corr_p50": float("nan"),
            "avg_abs_corr_mean": float("nan"),
            "feature_var_p50": var_p50,
            "feature_var_mean": var_mean,
            "n_features": len(feature_cols),
        }

    X_sub = X_sub[:, var_mask]
    corr = np.corrcoef(X_sub.T)
    triu_idx = np.triu_indices_from(corr, k=1)
    abs_corr_vals = np.abs(corr[triu_idx])
    abs_corr_vals = abs_corr_vals[np.isfinite(abs_corr_vals)]

    return {
        "avg_abs_corr_p50": float(np.median(abs_corr_vals)) if len(abs_corr_vals) > 0 else float("nan"),
        "avg_abs_corr_mean": float(np.mean(abs_corr_vals)) if len(abs_corr_vals) > 0 else float("nan"),
        "feature_var_p50": var_p50,
        "feature_var_mean": var_mean,
        "n_features": len(feature_cols),
    }


def compute_feature_stats_npz(path: str) -> dict:
    """Compute feature stats from an NPZ file (X_train only)."""
    data = np.load(path)
    X = data.get("X_train", data.get("X_meta", data.get("X", None)))
    if X is None:
        keys = [k for k in data.keys() if k.startswith("X")]
        X = data[keys[0]] if keys else None
    if X is None or X.shape[0] < 2 or X.shape[1] < 2:
        return {
            "avg_abs_corr_p50": float("nan"),
            "avg_abs_corr_mean": float("nan"),
            "feature_var_p50": float("nan"),
            "feature_var_mean": float("nan"),
            "n_features": 0,
            "n_samples": X.shape[0] if X is not None else 0,
        }

    X = X.astype(np.float64)
    X = np.where(np.isfinite(X), X, 0.0)

    # Per-feature variance
    feats_var = np.var(X, axis=0, ddof=1)
    feats_var = feats_var[np.isfinite(feats_var)]
    var_p50 = float(np.median(feats_var)) if len(feats_var) > 0 else float("nan")
    var_mean = float(np.mean(feats_var)) if len(feats_var) > 0 else float("nan")

    # Pairwise corr (limit to 50 cols)
    cols = X.shape[1]
    if cols > 50:
        rng = np.random.RandomState(42)
        idx = rng.choice(cols, 50, replace=False)
        X = X[:, idx]

    stds = np.std(X, axis=0)
    var_mask = stds > 1e-10
    if var_mask.sum() < 2:
        return {
            "avg_abs_corr_p50": float("nan"),
            "avg_abs_corr_mean": float("nan"),
            "feature_var_p50": var_p50,
            "feature_var_mean": var_mean,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }

    X = X[:, var_mask]
    corr = np.corrcoef(X.T)
    triu_idx = np.triu_indices_from(corr, k=1)
    abs_corr_vals = np.abs(corr[triu_idx])
    abs_corr_vals = abs_corr_vals[np.isfinite(abs_corr_vals)]

    return {
        "avg_abs_corr_p50": float(np.median(abs_corr_vals)) if len(abs_corr_vals) > 0 else float("nan"),
        "avg_abs_corr_mean": float(np.mean(abs_corr_vals)) if len(abs_corr_vals) > 0 else float("nan"),
        "feature_var_p50": var_p50,
        "feature_var_mean": var_mean,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
    }


# ── structural metrics ──────────────────────────────────────────────────────

def _robust_corr(X: np.ndarray) -> np.ndarray:
    """Correlation matrix with NaN/Inf handling."""
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, 0.0)
    stds = np.std(X, axis=0, ddof=1)
    mask = stds > 1e-10
    if mask.sum() < 2:
        return np.eye(X.shape[1])
    X = X[:, mask]
    return np.corrcoef(X.T)


def compute_eigen_spectrum(X: np.ndarray, max_cols: int = 50) -> dict:
    """Eigenvalue spectrum of the feature correlation matrix.

    Returns: eigenvalues (sorted descending), top5_ratio (top5/total),
    entropy (normalized Shannon entropy of normalized eigenvalues).
    """
    if X.ndim != 2 or X.shape[1] < 2:
        return {"eigvals": [], "top5_ratio": float("nan"), "entropy": float("nan")}

    if X.shape[1] > max_cols:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[1], max_cols, replace=False)
        X = X[:, idx]

    corr = _robust_corr(X)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.sort(eigvals)[::-1]  # descending
    eigvals = eigvals[eigvals > 1e-10]  # clip tiny negatives

    if len(eigvals) == 0:
        return {"eigvals": [], "top5_ratio": float("nan"), "entropy": float("nan")}

    total = eigvals.sum()
    normalized = eigvals / total
    top5_sum = eigvals[:min(5, len(eigvals))].sum()
    top5_ratio = top5_sum / total

    # Normalized Shannon entropy: 0 = all mass in one eigenvalue, 1 = uniform
    log_vals = np.log(normalized + 1e-12)
    entropy = -np.sum(normalized * log_vals) / np.log(len(eigvals))

    return {
        "eigvals": eigvals.tolist(),
        "top5_ratio": float(top5_ratio),
        "entropy": float(entropy),
        "n_features": int(X.shape[1]),
    }


def compute_wasserstein_1d(X_synth: np.ndarray, X_real: np.ndarray) -> dict:
    """Per-feature 1D Wasserstein-1 (earth mover) distance.

    Projection: for each column, sort both distributions and compute mean |diff|.
    Also returns the KS statistic as a simpler baseline.
    """
    X_synth = np.asarray(X_synth, dtype=np.float64)
    X_real = np.asarray(X_real, dtype=np.float64)
    X_synth = np.where(np.isfinite(X_synth), X_synth, 0.0)
    X_real = np.where(np.isfinite(X_real), X_real, 0.0)

    # Align feature dimension to the smaller
    n_feat = min(X_synth.shape[1], X_real.shape[1])
    if n_feat < 2:
        return {"w1_p50": float("nan"), "ks_p50": float("nan"), "n_feat": n_feat}

    # Standardize each side independently for scale-invariant comparison
    X_sr = X_synth[:, :n_feat]
    X_rr = X_real[:, :n_feat]
    s_mean = X_sr.mean(axis=0, keepdims=True)
    s_std = X_sr.std(axis=0, ddof=1, keepdims=True).clip(min=1e-8)
    r_mean = X_rr.mean(axis=0, keepdims=True)
    r_std = X_rr.std(axis=0, ddof=1, keepdims=True).clip(min=1e-8)

    X_s = (X_sr - s_mean) / s_std
    X_r = (X_rr - r_mean) / r_std

    w1_vals = []
    ks_vals = []
    from scipy.stats import ks_2samp  # noqa: PLC0415

    # Subsample both sides to the smaller size for W1 comparison
    n_common = min(X_s.shape[0], X_r.shape[0], 2000)
    rng = np.random.RandomState(42)
    idx_s = rng.choice(X_s.shape[0], n_common, replace=False)
    idx_r = rng.choice(X_r.shape[0], n_common, replace=False)

    for j in range(n_feat):
        s_col = X_s[idx_s, j]
        r_col = X_r[idx_r, j]
        s_sorted = np.sort(s_col)
        r_sorted = np.sort(r_col)
        # W1 = mean(|s_i - r_i|) on sorted values (exact for 1D equal-length)
        w1 = np.mean(np.abs(s_sorted - r_sorted))
        w1_vals.append(w1)
        # KS statistic (uses full samples)
        ks_stat, _ = ks_2samp(X_s[:, j], X_r[:, j])
        ks_vals.append(ks_stat)

    return {
        "w1_p50": float(np.median(w1_vals)),
        "w1_mean": float(np.mean(w1_vals)),
        "ks_p50": float(np.median(ks_vals)),
        "ks_mean": float(np.mean(ks_vals)),
        "n_feat": n_feat,
    }


def compute_energy_distance(X_synth: np.ndarray, X_real: np.ndarray,
                             max_samples: int = 500) -> dict:
    """Energy distance (squared) between two multivariate distributions.

    E² = 2·E||X-Y|| − E||X-X'|| − E||Y-Y'||.
    Subsamples to max_samples for O(n²) cost.
    """
    n_s = min(X_synth.shape[0], max_samples)
    n_r = min(X_real.shape[0], max_samples)

    # Align feature dimension to the smaller
    n_feat = min(X_synth.shape[1], X_real.shape[1])
    if n_feat < 2:
        return {"energy_sq": float("nan"), "energy_sqrt": float("nan"),
                "n_synth": n_s, "n_real": n_r}

    rng = np.random.RandomState(42)
    Xs = X_synth[rng.choice(X_synth.shape[0], n_s, replace=False), :n_feat]
    Xr = X_real[rng.choice(X_real.shape[0], n_r, replace=False), :n_feat]
    Xs = np.where(np.isfinite(Xs), Xs, 0.0)
    Xr = np.where(np.isfinite(Xr), Xr, 0.0)

    # Standardize jointly (use pooled mean/std for meaningful distance)
    pooled = np.vstack([Xs, Xr])
    mean = pooled.mean(axis=0, keepdims=True)
    std = pooled.std(axis=0, ddof=1, keepdims=True).clip(min=1e-8)
    Xs = (Xs - mean) / std
    Xr = (Xr - mean) / std

    from scipy.spatial.distance import cdist  # noqa: PLC0415

    # Cross-term
    cross = cdist(Xs, Xr, metric="euclidean").mean()
    # Within-synthetic
    within_s = cdist(Xs, Xs, metric="euclidean").mean()
    # Within-real
    within_r = cdist(Xr, Xr, metric="euclidean").mean()

    e_sq = 2.0 * cross - within_s - within_r
    return {
        "energy_sq": float(e_sq),
        "energy_sqrt": float(np.sqrt(max(e_sq, 0.0))),
        "n_synth": n_s,
        "n_real": n_r,
    }


def compute_raw_table_metrics(
    synth_tables: list[str],  # list of parquet file paths
    real_tables: list[str],   # list of parquet file paths
) -> dict:
    """Aggregate structural metrics comparing synthetic vs real raw tables.

    For each real-synth pair, computes:
    - eigval_entropy: Shannon entropy of normalized eigenvalue spectrum (0=concentrated, 1=uniform)
    - top5_ratio: fraction of eigenvalue mass in top-5 components
    - w1_p50: median per-feature Wasserstein-1 distance (standardized)
    - energy_sqrt: sqrt of energy distance between distributions

    Uses all synthetic tables and pairs with available real tables.
    """
    # Load all real numeric features
    real_Xs = []
    for path in real_tables:
        try:
            df = pd.read_parquet(path)
            cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if not c.endswith("_id")]
            if len(cols) < 2:
                continue
            X = df[cols].values.astype(np.float64)
            X = np.where(np.isfinite(X), X, 0.0)
            X = X[:, np.std(X, axis=0) > 1e-10]
            if X.shape[1] >= 4:
                real_Xs.append(X)
        except Exception:
            continue

    # Load all synthetic feature columns
    synth_Xs = []
    for path in synth_tables:
        try:
            df = pd.read_parquet(path)
            cols = [c for c in df.columns if "feature_" in c]
            if len(cols) < 2:
                continue
            X = df[cols].values.astype(np.float64)
            X = np.where(np.isfinite(X), X, 0.0)
            X = X[:, np.std(X, axis=0) > 1e-10]
            if X.shape[1] >= 4:
                synth_Xs.append(X)
        except Exception:
            continue

    if not synth_Xs or not real_Xs:
        return {"error": "No valid tables found", "n_synth": len(synth_Xs), "n_real": len(real_Xs)}

    # Eigen-spectrum stats
    synth_ent = []; synth_top5 = []
    for X in synth_Xs:
        spec = compute_eigen_spectrum(X)
        if not np.isnan(spec["entropy"]):
            synth_ent.append(spec["entropy"])
            synth_top5.append(spec["top5_ratio"])

    real_ent = []; real_top5 = []
    for X in real_Xs:
        spec = compute_eigen_spectrum(X)
        if not np.isnan(spec["entropy"]):
            real_ent.append(spec["entropy"])
            real_top5.append(spec["top5_ratio"])

    # Wasserstein: pair each real table with a randomly chosen synthetic table
    w1_all = []; ks_all = []
    for Xr in real_Xs[:20]:  # limit to 20 pairs
        Xs = synth_Xs[np.random.RandomState(42).choice(len(synth_Xs))]
        w1 = compute_wasserstein_1d(Xs, Xr)
        if not np.isnan(w1.get("w1_p50", float("nan"))):
            w1_all.append(w1["w1_p50"])
            ks_all.append(w1["ks_p50"])

    # Energy distance: sample a few pairs (expensive O(n²))
    en_all = []
    for _ in range(min(10, len(real_Xs), len(synth_Xs))):
        Xs = synth_Xs[np.random.RandomState(42 + _).choice(len(synth_Xs))]
        Xr = real_Xs[np.random.RandomState(42 + _).choice(len(real_Xs))]
        en = compute_energy_distance(Xs, Xr, max_samples=300)
        if not np.isnan(en.get("energy_sqrt", float("nan"))):
            en_all.append(en["energy_sqrt"])

    return {
        "n_synth_tables": len(synth_Xs),
        "n_real_tables": len(real_Xs),
        "eigen": {
            "synth_entropy_p50": float(np.median(synth_ent)) if synth_ent else float("nan"),
            "real_entropy_p50": float(np.median(real_ent)) if real_ent else float("nan"),
            "synth_top5_ratio_p50": float(np.median(synth_top5)) if synth_top5 else float("nan"),
            "real_top5_ratio_p50": float(np.median(real_top5)) if real_top5 else float("nan"),
        },
        "wasserstein": {
            "w1_p50": float(np.median(w1_all)) if w1_all else float("nan"),
            "ks_p50": float(np.median(ks_all)) if ks_all else float("nan"),
        },
        "energy_distance": {
            "energy_p50": float(np.median(en_all)) if en_all else float("nan"),
        },
    }


# ── constants ──────────────────────────────────────────────────────────────
AUC_HARD_REJECT = 0.52       # OOF AUC <= this -> reject directly
AUC_HARD_ACCEPT = 0.58       # OOF AUC >= this -> accept, no bootstrap
AUC_BASELINE = 0.55           # bootstrap CI upper bound must reach this
LEAKAGE_SINGLE_AUC = 0.90     # single-col AUC threshold (suspicious)
LEAKAGE_DROP = 0.10           # full-model AUC must drop by this much
CHILD_PARENT_RATIO_MAX = 100  # reject if child/parent row ratio exceeds this


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature columns (focal + joined from related tables), non-constant."""
    cols = []
    for c in df.columns:
        if c == "labels" or c == "timestamp":
            continue
        if c.endswith("_id"):
            continue
        if "feature_" in c:
            cols.append(c)
    if not cols:
        return []
    valid = df[cols].nunique() > 1
    return [c for c, ok in zip(cols, valid) if ok]


def _count_fk_cols(df: pd.DataFrame) -> int:
    """Count columns that look like foreign keys (name ends with _id)."""
    return sum(1 for c in df.columns
               if c.endswith("_id") and c != "labels" and "feature_" not in c)


def _etoof(X: np.ndarray, y: np.ndarray, n_splits: int = 3,
           n_estimators: int = 64, max_depth: int = 12,
           random_state: int = 42) -> Tuple[np.ndarray, float]:
    """ExtraTrees OOF predictions + AUC. Returns (oof_preds, auc)."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    n = len(y)
    if n < 2 * n_splits:
        n_splits = max(2, n // 2)
    min_class = int(min(np.bincount(y)))
    if min_class < n_splits:
        n_splits = max(2, min_class)
    oof = np.full(n, np.nan, dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in cv.split(X, y):
        m = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=random_state, n_jobs=1)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    valid = ~np.isnan(oof)
    if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
        return oof, 0.5
    return oof, float(roc_auc_score(y[valid], oof[valid]))


def _bootstrap_auc_from_oof(oof_preds: np.ndarray, y: np.ndarray,
                             n_bootstrap: int = 100,
                             random_state: int = 42) -> Tuple[float, float, float]:
    """Bootstrap AUC CI by resampling OOF predictions (no retraining)."""
    from sklearn.metrics import roc_auc_score

    valid = ~np.isnan(oof_preds)
    p = oof_preds[valid]
    t = y[valid]
    if len(np.unique(t)) < 2:
        return 0.5, 0.5, 0.5

    rng = np.random.RandomState(random_state)
    n = len(p)
    aucs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(t[idx])) < 2:
            aucs[i] = 0.5
        else:
            aucs[i] = float(roc_auc_score(t[idx], p[idx]))
    return (float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def _check_leakage(df: pd.DataFrame, y: np.ndarray, feature_cols: List[str],
                   full_auc: float) -> Tuple[bool, str]:
    """Check FK/ID/timestamp columns for label leakage."""
    id_cols = [c for c in df.columns
               if (c.endswith("_id") and c != "labels" and "feature_" not in c)
               or c == "timestamp"]
    if not id_cols:
        return False, ""

    for col in id_cols:
        if col not in df.columns:
            continue
        ser = df[col].copy()
        if pd.api.types.is_datetime64_any_dtype(ser):
            ser = ser.astype("int64")
        ser = pd.to_numeric(ser, errors="coerce")
        if ser.nunique() < 2:
            continue
        sX = ser.fillna(ser.median()).values.reshape(-1, 1).astype(np.float64)
        _, single_auc = _etoof(sX, y, n_splits=5, n_estimators=30, max_depth=6)
        if single_auc < LEAKAGE_SINGLE_AUC:
            continue

        wo_cols = [c for c in feature_cols if c != col]
        if len(wo_cols) < 1:
            continue
        wo_X = df[wo_cols].copy()
        wo_X = wo_X.replace([np.inf, -np.inf], np.nan)
        wo_X = wo_X.fillna(wo_X.median())
        wo_val = wo_X.values.astype(np.float64)
        _, wo_auc = _etoof(wo_val, y, n_splits=5, n_estimators=30, max_depth=6)
        drop = full_auc - wo_auc
        if drop > LEAKAGE_DROP:
            return True, (f"b3_leakage:{col} single={single_auc:.3f} "
                          f"full={full_auc:.3f} wo={wo_auc:.3f} drop={drop:.3f}")

    return False, ""


# ═════════════════════════════════════════════════════════════════════════════
# Stage A: rule-based pre-screening
# ═════════════════════════════════════════════════════════════════════════════

def stage_a_check_df(df: pd.DataFrame,
                     schema_data: Optional[Dict] = None,
                     focal_table: Optional[str] = None,
                     target_table: Optional[str] = None,
                     task_name: Optional[str] = None) -> dict:
    """Run all Stage A checks on a DataFrame.

    Args:
        df: Task dataframe (must have ``labels`` column).
        schema_data: Optional schema dict with ``table_generation_schemas`` and
                     ``task_generation_schemas`` keys (for a5/a6 checks).
        focal_table: Focal table name (for a5/a6 context).
        target_table: Target table name (for a5/a6 context).
        task_name: Task name (for a5/a6 context).

    Returns:
        dict with keys: a_pass, a_failures, n_samples, n_pos, n_neg,
        pos_ratio, n_features, n_fk_cols.
    """
    result = {
        "a_pass": False,
        "a_failures": [],
        "n_samples": 0, "n_pos": 0, "n_neg": 0, "pos_ratio": float("nan"),
        "n_features": 0, "n_fk_cols": 0,
    }

    if df is None or df.empty or "labels" not in df.columns:
        result["a_failures"].append("load_error")
        return result

    y = df["labels"].values
    n = len(df)
    n_unique = len(np.unique(y))
    result["n_samples"] = n

    # a1: label all-same
    if n_unique < 2:
        result["a_failures"].append("a1_label_unique_1")
        result["pos_ratio"] = float(np.mean(y))
        return result

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_ratio = n_pos / n
    result["n_pos"] = n_pos
    result["n_neg"] = n_neg
    result["pos_ratio"] = pos_ratio

    # a2: extreme label imbalance
    if pos_ratio < 0.05 or pos_ratio > 0.95:
        result["a_failures"].append(f"a2_pos_ratio_{pos_ratio:.3f}")

    # a3: too few samples
    if n < 64:
        result["a_failures"].append(f"a3_n_{n}_lt_64")
    if min(n_pos, n_neg) < 8:
        result["a_failures"].append(f"a3_min_class_{min(n_pos, n_neg)}_lt_8")

    # a4: no valid features
    raw_features = [c for c in df.columns if "feature_" in c]
    valid_features = _get_feature_cols(df)
    result["n_features"] = len(valid_features)
    if len(raw_features) == 0:
        result["a_failures"].append("a4_zero_feature_cols")
    elif len(valid_features) == 0:
        result["a_failures"].append("a4_all_constant")

    # a5: structural independence (schema-based, best-effort)
    if schema_data is not None and focal_table and target_table:
        tables = schema_data.get("table_generation_schemas", {})
        if isinstance(tables, dict):
            if target_table not in tables or focal_table not in tables:
                result["a_failures"].append("a5_table_not_in_schema")

    # a6: extreme child/parent size ratio
    result["n_fk_cols"] = _count_fk_cols(df)
    if schema_data is not None and result["n_fk_cols"] > 0:
        tables_dict = schema_data.get("table_generation_schemas", {})
        task_schemas = schema_data.get("task_generation_schemas", [])
        if isinstance(tables_dict, dict) and task_schemas:
            row_counts = {}
            for tname, info in tables_dict.items():
                scm = info.get("scm_params", {})
                if "seq_len" in scm:
                    row_counts[tname] = scm["seq_len"]
            child_rows = n
            for ts in task_schemas:
                if ts.get("task_name") != task_name:
                    continue
                parent_tables_list = ts.get("primary_table_generation_schema", {}).get(
                    "parent_tables", [])
                for pt in parent_tables_list:
                    parent_rows = row_counts.get(pt)
                    if parent_rows and child_rows > 0:
                        ratio = child_rows / parent_rows
                        if ratio > CHILD_PARENT_RATIO_MAX or ratio < (1.0 / CHILD_PARENT_RATIO_MAX):
                            result["a_failures"].append(f"a6_size_ratio_{ratio:.1f}")
                            break

    result["a_pass"] = len(result["a_failures"]) == 0
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Stage B: model-based checks
# ═════════════════════════════════════════════════════════════════════════════

def stage_b_check_df(df: pd.DataFrame) -> dict:
    """Stage B: shared OOF fit -> bootstrap CI on preds, leakage check."""
    result = {
        "b_pass": False,
        "b_failures": [],
        "et_auc": float("nan"),
        "et_ci_lower": float("nan"),
        "et_ci_upper": float("nan"),
        "was_gray_zone": False,
    }

    if df is None:
        result["b_failures"].append("b_load_error")
        return result

    y = df["labels"].values
    feature_cols = _get_feature_cols(df)
    if not feature_cols:
        result["b_failures"].append("b_no_features")
        return result

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X_val = X.values.astype(np.float64)

    # shared OOF fit (once per task)
    try:
        oof_preds, oof_auc = _etoof(X_val, y, n_splits=3, n_estimators=64, max_depth=12)
    except Exception as e:
        result["b_failures"].append(f"b1_oof_failed:{e}")
        return result

    # b1+b2: three-zone AUC check
    if oof_auc <= AUC_HARD_REJECT:
        result["et_auc"] = oof_auc
        result["b_failures"].append(f"b2_unlearnable: AUC={oof_auc:.4f} <= {AUC_HARD_REJECT}")
    elif oof_auc >= AUC_HARD_ACCEPT:
        result["et_auc"] = oof_auc  # clear pass
    else:
        result["was_gray_zone"] = True
        try:
            mean_auc, ci_l, ci_u = _bootstrap_auc_from_oof(
                oof_preds, y, n_bootstrap=100)
            result["et_auc"] = mean_auc
            result["et_ci_lower"] = ci_l
            result["et_ci_upper"] = ci_u
            if ci_u < AUC_BASELINE:
                result["b_failures"].append(
                    f"b2_unlearnable: AUC={mean_auc:.4f} CI=[{ci_l:.4f},{ci_u:.4f}]")
        except Exception as e:
            result["b_failures"].append(f"b2_bootstrap_failed:{e}")

    # b3: leakage check
    is_leak, leak_reason = _check_leakage(df, y, feature_cols, oof_auc)
    if is_leak:
        result["b_failures"].append(leak_reason)

    result["b_pass"] = len(result["b_failures"]) == 0
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Main entry points
# ═════════════════════════════════════════════════════════════════════════════

def diagnose_dataframe(df: pd.DataFrame,
                       schema_data: Optional[Dict] = None,
                       focal_table: Optional[str] = None,
                       target_table: Optional[str] = None,
                       task_name: Optional[str] = None) -> dict:
    """Two-stage diagnosis on an in-memory DataFrame (no disk I/O).

    Args:
        df: Task dataframe with ``labels`` column.
        schema_data: Optional schema dict for a5/a6 checks.
        focal_table, target_table, task_name: Context for a5/a6 checks.

    Returns:
        dict with keys: passed, stage, et_auc, et_ci_lower, et_ci_upper,
        all_failures, n_samples, n_pos, n_neg, pos_ratio, n_features.
    """
    a = stage_a_check_df(df, schema_data, focal_table, target_table, task_name)
    if not a["a_pass"]:
        return {
            **a,
            "passed": False,
            "stage": "A",
            "et_auc": float("nan"),
            "et_ci_lower": float("nan"),
            "et_ci_upper": float("nan"),
            "all_failures": a["a_failures"],
        }
    b = stage_b_check_df(df)
    return {
        **a,
        "passed": b["b_pass"],
        "stage": "B",
        "et_auc": b["et_auc"],
        "et_ci_lower": b["et_ci_lower"],
        "et_ci_upper": b["et_ci_upper"],
        "all_failures": a["a_failures"] + b["b_failures"],
    }


def diagnose_parquet_path(path: str) -> dict:
    """Two-stage diagnosis on a train.parquet file path.

    Loads the parquet, extracts schema from the parent directory if available.
    """
    df = pd.read_parquet(path)
    task_dir = os.path.dirname(path)
    task_name = os.path.basename(task_dir)

    # Best-effort schema loading
    schema_data = None
    rdb_dir = os.path.dirname(task_dir)
    for schema_name in ("generation_schemas.yaml", "metadata.yaml"):
        schema_path = os.path.join(rdb_dir, schema_name)
        if os.path.exists(schema_path):
            try:
                import yaml
                with open(schema_path) as f:
                    schema_data = yaml.safe_load(f)
                break
            except Exception:
                pass

    # Extract focal/target table from schema
    focal_table: Optional[str] = None
    target_table: Optional[str] = None
    if schema_data is not None:
        task_schemas = schema_data.get("task_generation_schemas", [])
        for ts in task_schemas:
            if ts.get("task_name") == task_name:
                focal_table = ts.get("primary_table_name")
                tns = ts.get("target_node_set", None)
                if tns and isinstance(tns, dict):
                    target_table = tns.get("table_name")
                break

    result = diagnose_dataframe(df, schema_data, focal_table, target_table, task_name)
    result["path"] = path
    return result
