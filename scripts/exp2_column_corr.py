"""Exp 2: Feature column-wise correlation structure analysis.

Compares Signal-Group (SG) vs MLP raw features on:
- Pairwise feature correlations (mean absolute off-diagonal)
- Effective rank (SVD, 95% variance threshold)
- Per-archetype breakdown
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def is_sg_feature(col_name: str) -> bool:
    """Signal-Group feature: feature_N (not ending in _mlp, not PK/FK/timestamp)."""
    if col_name.endswith("_id") or col_name == "timestamp":
        return False
    if col_name.endswith("_mlp"):
        return False
    return True


def is_mlp_feature(col_name: str) -> bool:
    """MLP raw feature: feature_N_mlp."""
    return col_name.endswith("_mlp")


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


def compute_stats(X: np.ndarray, n_raw: int) -> dict:
    """Compute correlation and rank stats for a feature matrix."""
    if X.shape[1] < 2:
        return {"n_features": n_raw, "n_valid": X.shape[1],
                "mean_abs_corr": 0.0, "eff_rank": max(1, X.shape[1]),
                "eff_rank_ratio": 1.0}

    corr = np.corrcoef(X.T)
    mask = ~np.eye(X.shape[1], dtype=bool)
    mean_abs_corr = float(np.abs(corr[mask]).mean())

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_norm = S / S.sum()
    cum_var = np.cumsum(S_norm)
    eff_rank = int(np.searchsorted(cum_var, 0.95) + 1)

    return {
        "n_features": n_raw,
        "n_valid": X.shape[1],
        "mean_abs_corr": mean_abs_corr,
        "eff_rank": eff_rank,
        "eff_rank_ratio": eff_rank / max(X.shape[1], 1),
    }


def analyze_table(df: pd.DataFrame) -> dict:
    """Compute stats separately for SG features and MLP features."""
    sg_cols = [c for c in df.columns if is_sg_feature(c)]
    mlp_cols = [c for c in df.columns if is_mlp_feature(c)]

    sg_X = df[sg_cols].to_numpy(dtype=np.float64)
    mlp_X = df[mlp_cols].to_numpy(dtype=np.float64)

    # Remove constant columns
    for label, X in [("sg", sg_X), ("mlp", mlp_X)]:
        stds = X.std(axis=0)
        valid = stds > 1e-8
        if label == "sg":
            sg_X = X[:, valid]
        else:
            mlp_X = X[:, valid]

    return {
        "sg": compute_stats(sg_X, len(sg_cols)),
        "mlp": compute_stats(mlp_X, len(mlp_cols)),
    }


def print_group(arch: str, key: str, results: list):
    """Print stats for one archetype and one feature type."""
    corrs = [r[key]["mean_abs_corr"] for r in results]
    ranks = [r[key]["eff_rank"] for r in results]
    ratios = [r[key]["eff_rank_ratio"] for r in results]
    n_feats = [r[key]["n_features"] for r in results]

    print(f"  {key}:")
    print(f"    n_features:     {np.mean(n_feats):.1f}")
    print(f"    mean_abs_corr:  {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")
    print(f"    eff_rank:       {np.mean(ranks):.1f} ± {np.std(ranks):.1f}")
    print(f"    eff_rank_ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--num_rdbs", type=int, default=64)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    all_results = []
    for idx in range(args.num_rdbs):
        rdb_dir = Path(args.dir) / f"dag_rdb_{idx}"
        if not rdb_dir.exists():
            continue
        for pq_file in sorted(rdb_dir.glob("table_*.parquet")):
            df = pd.read_parquet(pq_file)
            arch = table_archetype(df)
            stats = analyze_table(df)
            stats["rdb"] = idx
            stats["table"] = pq_file.stem
            stats["archetype"] = arch
            all_results.append(stats)

    if not all_results:
        print("No tables found")
        return

    # Per-archetype comparison
    for arch in ["source", "ts_child", "non_ts_child"]:
        group = [r for r in all_results if r["archetype"] == arch]
        if not group:
            continue
        print(f"\n{'='*60}")
        print(f"{arch} ({len(group)} tables)")
        print(f"{'='*60}")
        print_group(arch, "sg", group)
        print_group(arch, "mlp", group)

        # Delta summary
        sg_corrs = np.mean([r["sg"]["mean_abs_corr"] for r in group])
        mlp_corrs = np.mean([r["mlp"]["mean_abs_corr"] for r in group])
        sg_rank = np.mean([r["sg"]["eff_rank_ratio"] for r in group])
        mlp_rank = np.mean([r["mlp"]["eff_rank_ratio"] for r in group])
        print(f"  Δ mean_abs_corr:  {sg_corrs - mlp_corrs:+.4f}")
        print(f"  Δ eff_rank_ratio: {sg_rank - mlp_rank:+.3f}")

    # Overall
    print(f"\n{'='*60}")
    print(f"Overall ({len(all_results)} tables)")
    print(f"{'='*60}")
    print_group("overall", "sg", all_results)
    print_group("overall", "mlp", all_results)

    sg_corrs = np.mean([r["sg"]["mean_abs_corr"] for r in all_results])
    mlp_corrs = np.mean([r["mlp"]["mean_abs_corr"] for r in all_results])
    sg_rank = np.mean([r["sg"]["eff_rank_ratio"] for r in all_results])
    mlp_rank = np.mean([r["mlp"]["eff_rank_ratio"] for r in all_results])
    print(f"  Δ mean_abs_corr:  {sg_corrs - mlp_corrs:+.4f}")
    print(f"  Δ eff_rank_ratio: {sg_rank - mlp_rank:+.3f}")

    # Sanity check: correlation BETWEEN SG and MLP feature sets
    print(f"\n{'='*60}")
    print("Cross-space correlation (SG vs MLP)")
    print(f"{'='*60}")
    cross_corrs = []
    for r in all_results:
        rdb_dir = Path(args.dir) / f"dag_rdb_{r['rdb']}"
        pq_file = rdb_dir / f"{r['table']}.parquet"
        df = pd.read_parquet(pq_file)
        sg_cols = [c for c in df.columns if is_sg_feature(c)]
        mlp_cols = [c for c in df.columns if is_mlp_feature(c)]
        sg_X = df[sg_cols].to_numpy(dtype=np.float64)
        mlp_X = df[mlp_cols].to_numpy(dtype=np.float64)
        # Drop constant columns
        sg_X = sg_X[:, sg_X.std(axis=0) > 1e-8]
        mlp_X = mlp_X[:, mlp_X.std(axis=0) > 1e-8]
        if sg_X.shape[1] < 1 or mlp_X.shape[1] < 1:
            continue
        # Cross-correlation: corr between each SG col and each MLP col
        cross = np.corrcoef(sg_X.T, mlp_X.T)[:sg_X.shape[1], sg_X.shape[1]:]
        cross_corrs.append(float(np.abs(cross).mean()))
    print(f"  mean|corr(SG, MLP)|: {np.mean(cross_corrs):.4f} ± {np.std(cross_corrs):.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
