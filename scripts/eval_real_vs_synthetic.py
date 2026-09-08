#!/usr/bin/env python3
"""Compare synthetic data correlations against real data benchmarks."""

import numpy as np
import sys
from pathlib import Path


def compute_overall_corr(X: np.ndarray, subsample: int = 30, rng: np.random.RandomState | None = None) -> float:
    """Mean absolute pairwise correlation, subsampled to `subsample` columns."""
    if rng is None:
        rng = np.random.RandomState(42)
    n_cols = X.shape[1]
    if n_cols <= subsample:
        cols = np.arange(n_cols)
    else:
        cols = rng.choice(n_cols, size=subsample, replace=False)
    X_sub = X[:, cols].astype(np.float64)

    # Remove constant columns
    stds = np.std(X_sub, axis=0)
    valid_cols = np.nonzero(stds > 1e-10)[0]
    if len(valid_cols) < 2:
        return float("nan")
    X_sub = X_sub[:, valid_cols]

    corr_sum = 0.0
    count = 0
    for i in range(X_sub.shape[1]):
        for j in range(i + 1, X_sub.shape[1]):
            c = np.corrcoef(X_sub[:, i], X_sub[:, j])[0, 1]
            if not np.isnan(c):
                corr_sum += abs(c)
                count += 1
    return corr_sum / count if count > 0 else float("nan")


def eval_real_dir(dir_path: str, label: str, subsample: int = 30):
    """Evaluate all .npz files in a directory."""
    results = []
    rng = np.random.RandomState(42)
    for fpath in sorted(Path(dir_path).glob("*.npz")):
        data = np.load(fpath)
        X = data["X_train"]
        data.close()
        corr = compute_overall_corr(X, subsample=subsample, rng=rng)
        results.append((fpath.stem, X.shape[1], corr))

    print(f"\n=== {label} (n={len(results)}, subsample={subsample}) ===")
    if results:
        corrs = np.array([r[2] for r in results])
        print(f"  ncols: min={min(r[1] for r in results)}, max={max(r[1] for r in results)}, median={np.median([r[1] for r in results]):.0f}")
        print(f"  overall_corr: mean={np.nanmean(corrs):.4f}, std={np.nanstd(corrs):.4f}, "
              f"p50={np.nanmedian(corrs):.4f}, min={np.nanmin(corrs):.4f}, max={np.nanmax(corrs):.4f}")
    return results


def eval_synthetic_overall(processed_dir: str, num_rdbs: int, subsample: int, label: str):
    """Compute overall pairwise correlation on synthetic task data."""
    results = []
    rng = np.random.RandomState(42)
    for idx in range(num_rdbs):
        task_dir = Path(processed_dir) / f"dag_rdb_{idx}-dfs-1"
        if not task_dir.exists():
            continue
        for td in sorted(task_dir.iterdir()):
            if not td.is_dir():
                continue
            train_file = td / "train.npz"
            if not train_file.exists():
                continue
            data = np.load(train_file, allow_pickle=True)
            # Build float feature matrix
            float_cols = []
            for col_name in data.files:
                if col_name == "labels":
                    continue
                arr = data[col_name]
                if np.issubdtype(arr.dtype, np.floating):
                    float_cols.append(arr)
            data.close()
            if len(float_cols) < 2:
                continue
            X = np.column_stack(float_cols)
            corr = compute_overall_corr(X, subsample=subsample, rng=rng)
            results.append(corr)

    if results:
        arr = np.array(results)
        finite = arr[np.isfinite(arr)]
        print(f"\n=== {label} (n={len(results)}, subsample={subsample}, n_finite={len(finite)}) ===")
        print(f"  overall_corr: mean={np.nanmean(arr):.4f}, std={np.nanstd(arr):.4f}, "
              f"p50={np.nanmedian(arr):.4f}, min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}")
    return results


if __name__ == "__main__":
    base = "/data/caijunyu/RDBPFN-hsbm-fk-generation/model_pretrain/datasets"
    subsample = 30

    # Real non-relational data (all native columns)
    eval_real_dir(f"{base}/clf_real_subsamples", "Real (non-relational, all-native)", subsample)

    # Real relational DFS-2 data (mixed native + joined + aggregated)
    eval_real_dir(f"{base}/clf_rel_subsamples", "Real (relational DFS-2, mixed)", subsample)

    # Synthetic baseline
    eval_synthetic_overall(
        "/data/caijunyu/RDBPFN-hsbm-fk-generation/data_generation/RDB_datasets/baseline_no_structsig-processed",
        64, subsample, "Synthetic Baseline (no propensity/matching)",
    )

    # Synthetic final
    eval_synthetic_overall(
        "/data/caijunyu/RDBPFN-hsbm-fk-generation/data_generation/RDB_datasets/propensity_matching_64-processed",
        64, subsample, "Synthetic Final (propensity + matching)",
    )
