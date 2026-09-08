"""Evaluate synthetic data learnability and realism.

Three metric categories beyond eff_rank and mean|corr|:

1. INTER-RDB DIVERGENCE — are different RDBs structurally diverse?
   - eff_rank, mean|corr|, eigen entropy distributions across RDBs
   - Higher variance = more inter-RDB diversity (good)

2. NONLINEAR FEATURE-LABEL COUPLING — can the label be recovered nonlinearly?
   - Mutual information per feature (sklearn mutual_info_regression)
   - Compares percol MLP (nonlinear features) vs SG (linear features)

3. LINEAR PROBE AUC — is the label linearly learnable from features?
   - LogisticRegression OOF AUC per RDB
   - Directly measures training signal quality

Usage:
  pixi run python scripts/learnability_metrics.py \
    --h5-syn model_pretrain/pretrain_datasets/percol_mlp_v1.h5 \
    --h5-baseline model_pretrain/pretrain_datasets/v53.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean_features(X: np.ndarray) -> np.ndarray:
    """Replace inf/nan, remove constant columns."""
    X = np.where(np.isfinite(X), X, 0.0)
    stds = np.std(X, axis=0)
    valid = stds > 1e-8
    if valid.sum() < 2:
        return X[:, :2]
    return X[:, valid]


def _robust_corr(X: np.ndarray) -> np.ndarray:
    """Correlation of cleaned data."""
    Xc = _clean_features(X)
    return np.corrcoef(Xc, rowvar=False)


def _eff_rank_ratio(X: np.ndarray) -> float:
    """95%-variance effective rank / n_features."""
    Xc = _clean_features(X)
    if Xc.shape[1] < 2:
        return float("nan")
    U, S, Vh = np.linalg.svd(Xc - Xc.mean(axis=0), full_matrices=False)
    total_var = (S ** 2).sum()
    if total_var < 1e-8:
        return float("nan")
    cumsum = np.cumsum(S ** 2) / total_var
    return (int(np.searchsorted(cumsum, 0.95)) + 1) / len(S)


def _eigen_entropy(X: np.ndarray) -> float:
    """Normalized Shannon entropy of correlation eigenvalues. 0=rank-1, 1=uniform."""
    Xc = _clean_features(X)
    if Xc.shape[1] < 2:
        return float("nan")
    corr = _robust_corr(Xc)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return float("nan")
    total = eigvals.sum()
    normalized = eigvals / total
    log_vals = np.log(normalized + 1e-12)
    return float(-np.sum(normalized * log_vals) / np.log(len(eigvals)))


def _mean_abs_corr(X: np.ndarray) -> float:
    """Mean absolute pairwise correlation."""
    Xc = _clean_features(X)
    if Xc.shape[1] < 3:
        return float("nan")
    corr = _robust_corr(Xc)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(corr[mask]).mean())


# ── Metric 1: Inter-RDB Divergence ──────────────────────────────────────────

def compute_inter_rdb_divergence(h5_path: str, label: str) -> dict:
    """Per-dataset eff_rank, corr, entropy → measure inter-RDB variance."""
    print(f"\n{'='*60}")
    print(f"Metric 1: Inter-RDB Divergence — {label}")
    print(f"{'='*60}")

    with h5py.File(h5_path, "r") as f:
        X_all = f["X"][:].astype(np.float64)
        y_all = f["y"][:]
        nf_all = f["num_features"][:]

    eff_ranks = []
    mean_corrs = []
    eigen_entropies = []

    for ds_idx in range(X_all.shape[0]):
        nf = int(nf_all[ds_idx])
        X = X_all[ds_idx, :, :nf]
        y = y_all[ds_idx]

        # Remove rows with NaN labels
        valid_y = np.isfinite(y)
        if valid_y.sum() < 10:
            continue
        X = X[valid_y, :]
        y = y[valid_y]

        er = _eff_rank_ratio(X)
        mc = _mean_abs_corr(X)
        ent = _eigen_entropy(X)

        if not np.isnan(er):
            eff_ranks.append(er)
        if not np.isnan(mc):
            mean_corrs.append(mc)
        if not np.isnan(ent):
            eigen_entropies.append(ent)

    er_a = np.array(eff_ranks)
    mc_a = np.array(mean_corrs)
    ent_a = np.array(eigen_entropies)

    print(f"  n_datasets:          {len(er_a)}")
    print(f"  eff_rank_ratio:      {er_a.mean():.4f} ± {er_a.std():.4f}  (P10={np.percentile(er_a,10):.4f} P90={np.percentile(er_a,90):.4f})")
    print(f"  mean|corr|:          {mc_a.mean():.4f} ± {mc_a.std():.4f}  (P10={np.percentile(mc_a,10):.4f} P90={np.percentile(mc_a,90):.4f})")
    print(f"  eigen_entropy:       {ent_a.mean():.4f} ± {ent_a.std():.4f}  (P10={np.percentile(ent_a,10):.4f} P90={np.percentile(ent_a,90):.4f})")

    # Key: coefficient of variation (std/mean) = inter-RDB diversity
    er_cv = float(er_a.std() / er_a.mean()) if er_a.mean() > 0 else float("nan")
    mc_cv = float(mc_a.std() / mc_a.mean()) if mc_a.mean() > 0 else float("nan")
    ent_cv = float(ent_a.std() / ent_a.mean()) if ent_a.mean() > 0 else float("nan")
    print(f"  CV eff_rank (diversity):  {er_cv:.4f}")
    print(f"  CV mean|corr| (diversity):{mc_cv:.4f}")
    print(f"  CV entropy (diversity):   {ent_cv:.4f}")

    return {
        "eff_rank_mean": float(er_a.mean()), "eff_rank_std": float(er_a.std()),
        "eff_rank_cv": er_cv,
        "mean_corr_mean": float(mc_a.mean()), "mean_corr_std": float(mc_a.std()),
        "mean_corr_cv": mc_cv,
        "entropy_mean": float(ent_a.mean()), "entropy_std": float(ent_a.std()),
        "entropy_cv": ent_cv,
        "eff_ranks": er_a.tolist(),
        "mean_corrs": mc_a.tolist(),
        "entropies": ent_a.tolist(),
    }


# ── Metric 2: Nonlinear Feature-Label MI ─────────────────────────────────────

def compute_feature_label_mi(h5_path: str, label: str) -> dict:
    """Mutual information between each feature and label."""
    print(f"\n{'='*60}")
    print(f"Metric 2: Feature-Label Mutual Information — {label}")
    print(f"{'='*60}")

    with h5py.File(h5_path, "r") as f:
        X_all = f["X"][:].astype(np.float64)
        y_all = f["y"][:]
        nf_all = f["num_features"][:]

    all_mi_means = []   # per-dataset mean MI
    all_mi_medians = []
    all_mi_maxes = []   # top feature MI

    for ds_idx in range(min(X_all.shape[0], 200)):  # MI is expensive, sample 200
        nf = int(nf_all[ds_idx])
        X = X_all[ds_idx, :, :nf]
        y = y_all[ds_idx]

        valid_y = np.isfinite(y)
        if valid_y.sum() < 20:
            continue
        X = X[valid_y, :]
        y = y[valid_y]
        Xc = _clean_features(X)
        if Xc.shape[1] < 3:
            continue

        # MI with discrete y (binary classification)
        y_binary = (y > np.median(y)).astype(int)
        if len(np.unique(y_binary)) < 2:
            continue

        try:
            mi = mutual_info_regression(Xc, y_binary, random_state=42, n_neighbors=3)
            mi = mi[np.isfinite(mi)]
            if len(mi) > 0:
                all_mi_means.append(float(np.mean(mi)))
                all_mi_medians.append(float(np.median(mi)))
                all_mi_maxes.append(float(np.max(mi)))
        except Exception:
            continue

    mi_mean = np.array(all_mi_means)
    mi_med = np.array(all_mi_medians)
    mi_max = np.array(all_mi_maxes)

    print(f"  n_datasets:          {len(mi_mean)}")
    print(f"  MI mean per ds:      {mi_mean.mean():.4f} ± {mi_mean.std():.4f}")
    print(f"  MI median per ds:    {mi_med.mean():.4f} ± {mi_med.std():.4f}")
    print(f"  MI max (top feat):   {mi_max.mean():.4f} ± {mi_max.std():.4f}")

    return {
        "mi_mean": float(mi_mean.mean()), "mi_mean_std": float(mi_mean.std()),
        "mi_median": float(mi_med.mean()), "mi_median_std": float(mi_med.std()),
        "mi_max": float(mi_max.mean()), "mi_max_std": float(mi_max.std()),
    }


# ── Metric 3: Linear Probe AUC per RDB ───────────────────────────────────────

def compute_linear_probe_auc(h5_path: str, label: str) -> dict:
    """Per-RDB linear probe AUC using 3-fold CV logistic regression."""
    print(f"\n{'='*60}")
    print(f"Metric 3: Linear Probe AUC per RDB — {label}")
    print(f"{'='*60}")

    with h5py.File(h5_path, "r") as f:
        X_all = f["X"][:].astype(np.float64)
        y_all = f["y"][:]
        nf_all = f["num_features"][:]

    aucs = []

    for ds_idx in range(min(X_all.shape[0], 500)):  # limit to 500 for speed
        nf = int(nf_all[ds_idx])
        X = X_all[ds_idx, :, :nf]
        y = y_all[ds_idx]

        valid_y = np.isfinite(y)
        if valid_y.sum() < 30:
            continue
        X = X[valid_y, :]
        y = y[valid_y]
        Xc = _clean_features(X)
        if Xc.shape[1] < 3:
            continue

        # Binarize label for classification AUC
        y_median = np.median(y)
        if np.isnan(y_median):
            continue
        y_bin = (y > y_median).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue

        Xs = StandardScaler().fit_transform(Xc)

        # 3-fold CV
        try:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fold_aucs = []
            for train_idx, test_idx in skf.split(Xs, y_bin):
                clf = LogisticRegression(max_iter=500, random_state=42, n_jobs=1)
                clf.fit(Xs[train_idx], y_bin[train_idx])
                y_pred = clf.predict_proba(Xs[test_idx])[:, 1]
                fold_aucs.append(roc_auc_score(y_bin[test_idx], y_pred))
            aucs.append(np.mean(fold_aucs))
        except Exception:
            continue

    auc_a = np.array(aucs)
    print(f"  n_datasets:          {len(auc_a)}")
    print(f"  Linear probe AUC:    {auc_a.mean():.4f} ± {auc_a.std():.4f}")
    print(f"  AUC P10: {np.percentile(auc_a, 10):.4f}  P25: {np.percentile(auc_a, 25):.4f}")
    print(f"  AUC P50: {np.percentile(auc_a, 50):.4f}  P75: {np.percentile(auc_a, 75):.4f}")
    print(f"  AUC P90: {np.percentile(auc_a, 90):.4f}")

    # Fraction of "hard" datasets (AUC < 0.6) vs "easy" (AUC > 0.8)
    hard = (auc_a < 0.6).mean()
    easy = (auc_a > 0.8).mean()
    print(f"  Hard (AUC<0.6): {hard:.1%}   Easy (AUC>0.8): {easy:.1%}")

    return {
        "auc_mean": float(auc_a.mean()), "auc_std": float(auc_a.std()),
        "auc_p10": float(np.percentile(auc_a, 10)),
        "auc_p50": float(np.percentile(auc_a, 50)),
        "auc_p90": float(np.percentile(auc_a, 90)),
        "hard_frac": float(hard), "easy_frac": float(easy),
        "aucs": auc_a.tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-syn", default="model_pretrain/pretrain_datasets/percol_mlp_v1.h5")
    parser.add_argument("--h5-baseline", default="model_pretrain/pretrain_datasets/v53.h5")
    args = parser.parse_args()

    print("=" * 60)
    print("SYNTHETIC DATA LEARNABILITY METRICS")
    print("=" * 60)

    results = {}

    for path, lbl in [(args.h5_baseline, "v53 (SG linear)"), (args.h5_syn, "Percol MLP")]:
        if not Path(path).exists():
            print(f"\n  SKIP: {path} not found")
            continue

        results[lbl] = {}
        results[lbl]["divergence"] = compute_inter_rdb_divergence(path, lbl)
        results[lbl]["mi"] = compute_feature_label_mi(path, lbl)
        results[lbl]["linear_probe"] = compute_linear_probe_auc(path, lbl)

    # ── Summary Table ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    rows = list(results.keys())
    if len(rows) < 2:
        print("  Need both datasets for comparison")
        return

    print(f"\n  {'Metric':<40} {'v53 (SG)':>15} {'Percol MLP':>15}")
    print(f"  {'-'*40} {'-'*15} {'-'*15}")

    compare = [
        ("eff_rank CV (inter-RDB diversity ↑)", "divergence", "eff_rank_cv"),
        ("mean|corr| CV (inter-RDB diversity ↑)", "divergence", "mean_corr_cv"),
        ("entropy CV (inter-RDB diversity ↑)", "divergence", "entropy_cv"),
        ("eigen_entropy mean", "divergence", "entropy_mean"),
        ("Feature-label MI mean", "mi", "mi_mean"),
        ("Feature-label MI max (top feat)", "mi", "mi_max"),
        ("Linear probe AUC mean", "linear_probe", "auc_mean"),
        ("Linear probe AUC std", "linear_probe", "auc_std"),
        ("Easy datasets (AUC>0.8)", "linear_probe", "easy_frac"),
        ("Hard datasets (AUC<0.6)", "linear_probe", "hard_frac"),
    ]

    for metric_name, cat, key in compare:
        v0 = results[rows[0]][cat][key]
        v1 = results[rows[1]][cat][key]
        if isinstance(v0, float):
            arrow = "↑" if v1 > v0 else ("↓" if v1 < v0 else "=")
            print(f"  {metric_name:<40} {v0:>15.4f} {v1:>15.4f}  {arrow}")
        else:
            print(f"  {metric_name:<40} {str(v0):>15} {str(v1):>15}")


if __name__ == "__main__":
    main()
