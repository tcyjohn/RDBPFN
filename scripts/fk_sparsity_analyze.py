"""FK sparsity analysis: compare fk_sparse_v1 vs baseline on FK coverage and feature diversity.

Three metrics:
1. FK coverage per relation — % of child rows with FK=-1
2. Feature mean|corr| distribution from h5
3. Feature eff_rank_ratio distribution from h5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def table_archetype(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fk_cols = [c for c in cols if c.endswith("_id") and c != cols[0]]
    has_ts = "timestamp" in cols
    has_fk = len(fk_cols) > 0
    if not has_fk:
        return "source"
    elif has_ts:
        return "ts_child"
    else:
        return "non_ts_child"


def analyze_raw_fk_coverage(raw_dir: str, num_rdbs: int) -> dict:
    """Per-child-table FK null rate from raw parquet data."""
    raw_path = Path(raw_dir)
    fk_stats = []  # list of (rdb_idx, table_name, parent_col, null_rate, total_rows)

    for idx in range(num_rdbs):
        rdb_dir = raw_path / f"dag_rdb_{idx}"
        if not rdb_dir.is_dir():
            continue
        for pq_file in sorted(rdb_dir.glob("table_*.parquet")):
            table_name = pq_file.stem
            df = pd.read_parquet(pq_file)
            cols = list(df.columns)
            pk_col = cols[0]
            fk_cols = [c for c in cols if c.endswith("_id") and c != pk_col]

            if not fk_cols:
                continue  # source table, no FK

            for fk_col in fk_cols:
                total = len(df)
                null_count = (df[fk_col] == -1).sum()
                null_rate = null_count / total
                fk_stats.append({
                    "rdb": idx,
                    "table": table_name,
                    "parent_col": fk_col,
                    "null_rate": null_rate,
                    "null_count": int(null_count),
                    "total_rows": total,
                })

    return fk_stats


def print_fk_summary(fk_stats: list[dict], label: str):
    """Print FK coverage summary."""
    if not fk_stats:
        print(f"  {label}: No FK stats found")
        return

    rates = [s["null_rate"] for s in fk_stats]
    rates_arr = np.array(rates)
    print(f"  {label} ({len(fk_stats)} FK relations):")
    print(f"    Mean null rate:  {rates_arr.mean():.4f}")
    print(f"    Median:          {np.median(rates_arr):.4f}")
    print(f"    Min / Max:       {rates_arr.min():.4f} / {rates_arr.max():.4f}")
    print(f"    Std:             {rates_arr.std():.4f}")

    # Distribution buckets
    buckets = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    print(f"    Rate distribution:")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = ((rates_arr >= lo) & (rates_arr < hi)).sum()
        pct = count / len(rates_arr) * 100
        print(f"      [{lo:.2f}, {hi:.2f}): {count:3d} ({pct:5.1f}%)")

    # Show top 10 most sparse relations
    print(f"    Top 10 sparsest relations:")
    top10 = sorted(fk_stats, key=lambda x: -x["null_rate"])[:10]
    for s in top10:
        print(f"      RDB {s['rdb']:3d}  {s['table']:20s} → {s['parent_col']:20s}  "
              f"null_rate={s['null_rate']:.4f}  ({s['null_count']}/{s['total_rows']})")
    print()


def analyze_h5_features(h5_path: str):
    """Load h5 and compute per-dataset feature statistics.

    h5 format: X=(n_datasets, 600, 90) pre-stacked tensor,
    with num_features per dataset (some cols may be padding).
    """
    import h5py

    print(f"  Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        print(f"    h5 keys: {list(f.keys())}")
        X_full = f["X"][:]
        n_feat_vec = f["num_features"][:] if "num_features" in f else None
        y_full = f["y"][:] if "y" in f else None

    n_datasets = X_full.shape[0]
    print(f"    X shape: {X_full.shape}, y shape: {y_full.shape if y_full is not None else 'N/A'}")
    print(f"    n_datasets: {n_datasets}")

    all_mean_corrs = []
    all_eff_ranks = []
    all_feat_label_corrs = []
    per_ds_stats = []

    for ds_idx in range(n_datasets):
        n_feat = int(n_feat_vec[ds_idx]) if n_feat_vec is not None else X_full.shape[2]
        if n_feat <= 0 or n_feat > X_full.shape[2]:
            n_feat = X_full.shape[2]

        X = X_full[ds_idx, :, :n_feat].astype(np.float64)

        # Remove constant columns
        stds = np.std(X, axis=0)
        valid_cols = np.where(stds > 1e-8)[0]
        if len(valid_cols) < 2:
            continue
        X_valid = X[:, valid_cols]

        # Mean absolute correlation
        corr = np.corrcoef(X_valid, rowvar=False)
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        mean_corr = np.abs(corr[mask]).mean()

        # Effective rank (95% variance)
        U, S, Vh = np.linalg.svd(X_valid, full_matrices=False)
        total_var = (S ** 2).sum()
        if total_var < 1e-8:
            continue
        cumsum = np.cumsum(S ** 2) / total_var
        eff_rank = int(np.searchsorted(cumsum, 0.95)) + 1
        eff_rank_ratio = eff_rank / len(S)

        # Feature-label |corr| (if y available)
        feat_label_corr = 0.0
        if y_full is not None:
            y = y_full[ds_idx]
            if y.ndim > 1:
                y = y.ravel()
            valid_y = np.isfinite(y)
            if valid_y.sum() > 10 and np.std(y[valid_y]) > 1e-8:
                fl_corrs = []
                for j in range(X_valid.shape[1]):
                    xj = X_valid[valid_y, j]
                    if np.std(xj) > 1e-8:
                        fl_corrs.append(abs(np.corrcoef(xj, y[valid_y])[0, 1]))
                if fl_corrs:
                    feat_label_corr = np.mean(fl_corrs)

        all_mean_corrs.append(mean_corr)
        all_eff_ranks.append(eff_rank_ratio)
        all_feat_label_corrs.append(feat_label_corr)
        per_ds_stats.append({
            "ds": ds_idx,
            "n_features": n_feat,
            "n_valid": len(valid_cols),
            "eff_rank_ratio": eff_rank_ratio,
            "mean_abs_corr": mean_corr,
            "feat_label_corr": feat_label_corr,
        })

    if all_mean_corrs:
        mc = np.array(all_mean_corrs)
        er = np.array(all_eff_ranks)
        fl = np.array(all_feat_label_corrs)
        print(f"    n_datasets:       {len(all_mean_corrs)}")
        print(f"    mean|corr|:       {mc.mean():.4f} ± {mc.std():.4f}  (median={np.median(mc):.4f})")
        print(f"    eff_rank_ratio:   {er.mean():.4f} ± {er.std():.4f}  (median={np.median(er):.4f})")
        print(f"    feat-label |corr|:{fl.mean():.4f} ± {fl.std():.4f}  (median={np.median(fl):.4f})")
        print(f"    mean|corr| range: [{mc.min():.4f}, {mc.max():.4f}]")
        print(f"    eff_rank range:   [{er.min():.4f}, {er.max():.4f}]")
        print(f"    fl_corr range:    [{fl.min():.4f}, {fl.max():.4f}]")

        # Print bottom/top 5 eff_rank
        print(f"    Bottom 5 eff_rank_ratio (most redundant):")
        bottom = sorted(per_ds_stats, key=lambda x: x["eff_rank_ratio"])[:5]
        for s in bottom:
            print(f"      ds={s['ds']:3d} eff_rank={s['eff_rank_ratio']:.4f}  "
                  f"corr={s['mean_abs_corr']:.4f}  fl_corr={s['feat_label_corr']:.4f}  "
                  f"n_feat={s['n_features']}/{s['n_valid']}")
        print(f"    Top 5 eff_rank_ratio (most diverse):")
        top = sorted(per_ds_stats, key=lambda x: -x["eff_rank_ratio"])[:5]
        for s in top:
            print(f"      ds={s['ds']:3d} eff_rank={s['eff_rank_ratio']:.4f}  "
                  f"corr={s['mean_abs_corr']:.4f}  fl_corr={s['feat_label_corr']:.4f}  "
                  f"n_feat={s['n_features']}/{s['n_valid']}")

    return per_ds_stats


def main():
    parser = argparse.ArgumentParser(description="FK sparsity analysis")
    parser.add_argument("--raw-dir", type=str,
                        default="data_generation/RDB_datasets/fk_sparse_v1",
                        help="Raw RDB output directory")
    parser.add_argument("--num-rdbs", type=int, default=64)
    parser.add_argument("--h5-path", type=str,
                        default="model_pretrain/pretrain_datasets/fk_sparse_v1.h5",
                        help="Merged h5 file path")
    parser.add_argument("--baseline-h5", type=str,
                        default="model_pretrain/pretrain_datasets/v53.h5",
                        help="Baseline h5 for comparison")
    parser.add_argument("--skip-h5", action="store_true",
                        help="Skip h5 analysis (if h5 not yet merged)")
    args = parser.parse_args()

    # 1. FK coverage analysis
    print("=" * 70)
    print("FK Coverage Analysis")
    print("=" * 70)
    fk_stats = analyze_raw_fk_coverage(args.raw_dir, args.num_rdbs)
    print_fk_summary(fk_stats, args.raw_dir)

    if args.skip_h5:
        print("Skipping h5 analysis (--skip-h5)")
        return

    # 2. h5 feature analysis
    print("=" * 70)
    print("h5 Feature Statistics")
    print("=" * 70)
    print(f"\n  [{args.h5_path}]")
    stats = analyze_h5_features(args.h5_path)

    if Path(args.baseline_h5).exists():
        print(f"\n  [Baseline: {args.baseline_h5}]")
        analyze_h5_features(args.baseline_h5)
    else:
        print(f"\n  [Baseline {args.baseline_h5} not found, skipping comparison]")


if __name__ == "__main__":
    main()
