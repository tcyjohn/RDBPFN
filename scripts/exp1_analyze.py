"""Exp 1 analysis: SG-only vs SCM-only feature-label correlation.

Reads two separate output directories (sg_only and scm_only), computes
per-feature |pearsonr(feature, labels)| from complex task parquet files,
and compares distributions (overall + per-archetype).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def build_archetype_map(rdb_dir: Path) -> dict[str, str]:
    """Classify each table as source/ts_child/non_ts_child from its parquet."""
    arch_map = {}
    for pq_file in sorted(rdb_dir.glob("table_*.parquet")):
        table_name = pq_file.stem
        df = pd.read_parquet(pq_file)
        cols = list(df.columns)
        id_cols = [c for c in cols if c.endswith("_id")]
        pk_col = cols[0]
        fk_cols = [c for c in id_cols if c != pk_col]
        has_ts = "timestamp" in cols
        if not fk_cols:
            arch_map[table_name] = "source"
        elif has_ts:
            arch_map[table_name] = "ts_child"
        else:
            arch_map[table_name] = "non_ts_child"
    return arch_map


def collect_feature_label_corrs(
    base_dir: str, num_rdbs: int
) -> tuple[list[float], dict[str, list[float]], list[dict]]:
    """Collect per-feature |pearsonr(feature, labels)| from task parquets."""
    all_corrs: list[float] = []
    arch_corrs: dict[str, list[float]] = {
        "source": [], "ts_child": [], "non_ts_child": [],
    }
    table_stats: list[dict] = []
    n_tasks_processed = 0
    n_tables_skipped = 0

    for idx in range(num_rdbs):
        rdb_dir = Path(base_dir) / f"dag_rdb_{idx}"
        if not rdb_dir.exists():
            continue

        meta_file = rdb_dir / "metadata.yaml"
        if not meta_file.exists():
            continue

        # Build archetype map from main table parquets
        archetype_map = build_archetype_map(rdb_dir)

        with open(meta_file) as f:
            meta = yaml.safe_load(f)

        tasks = meta.get("tasks", [])
        for task in tasks:
            task_name = task.get("name")
            target_table = task.get("target_table")
            if not task_name or not target_table:
                continue

            task_dir = rdb_dir / task_name
            train_file = task_dir / "train.parquet"
            if not train_file.exists():
                continue

            arch = archetype_map.get(target_table, "unknown")
            if arch == "unknown":
                n_tables_skipped += 1
                continue

            df = pd.read_parquet(train_file)
            if "labels" not in df.columns:
                continue

            y = df["labels"].values.astype(np.float64)
            if np.std(y) < 1e-8:
                continue

            feat_cols = [
                c for c in df.columns
                if c.startswith("feature_") and not c.endswith("_mlp")
            ]

            table_corrs: list[float] = []
            for ft in feat_cols:
                x = df[ft].values.astype(np.float64)
                if np.std(x) < 1e-8:
                    continue
                corr = abs(float(np.corrcoef(x, y)[0, 1]))
                if not np.isnan(corr):
                    all_corrs.append(corr)
                    arch_corrs[arch].append(corr)
                    table_corrs.append(corr)

            if table_corrs:
                table_stats.append({
                    "rdb": idx, "task": task_name, "table": target_table,
                    "archetype": arch,
                    "n_feats": len(feat_cols),
                    "mean_corr": float(np.mean(table_corrs)),
                })
            n_tasks_processed += 1

    print(f"  ({n_tasks_processed} tasks processed, {n_tables_skipped} skipped — "
          f"no archetype match)")
    return all_corrs, arch_corrs, table_stats


def print_mode_stats(label: str, all_corrs: list[float],
                     arch_corrs: dict[str, list[float]]):
    """Print statistics for one mode."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    n_tasks = len(all_corrs)
    if n_tasks == 0:
        print("  (no data)")
        return
    c = np.array(all_corrs)
    print(f"  N feature-label pairs: {len(c)}")
    print(f"  Mean  |corr|: {np.mean(c):.5f}")
    print(f"  Median:        {np.median(c):.5f}")
    print(f"  P10: {np.percentile(c, 10):.5f}  P90: {np.percentile(c, 90):.5f}")

    print(f"\n  Per-archetype:")
    print(f"  {'Archetype':<16s} {'Mean':>10s} {'Median':>10s} {'N':>7s}")
    print(f"  {'-'*43}")
    for arch in ["source", "ts_child", "non_ts_child"]:
        ac = arch_corrs.get(arch, [])
        if ac:
            a = np.array(ac)
            print(f"  {arch:<16s} {np.mean(a):10.5f} {np.median(a):10.5f} {len(a):7d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_sg", required=True,
                        help="SG-only output directory")
    parser.add_argument("--dir_scm", required=True,
                        help="SCM-only (EXP_SKIP_SG=1) output directory")
    parser.add_argument("--num_rdbs", type=int, default=64)
    args = parser.parse_args()

    sg_corrs, sg_arch, sg_tables = collect_feature_label_corrs(
        args.dir_sg, args.num_rdbs)
    scm_corrs, scm_arch, scm_tables = collect_feature_label_corrs(
        args.dir_scm, args.num_rdbs)

    print_mode_stats("SG-only  (X = signal-group output)", sg_corrs, sg_arch)
    print_mode_stats("SCM-only (X = MLP raw slice)", scm_corrs, scm_arch)

    if sg_corrs and scm_corrs:
        sg = np.array(sg_corrs)
        scm = np.array(scm_corrs)
        print(f"\n{'='*60}")
        print(f"  SG vs SCM Comparison")
        print(f"{'='*60}")
        print(f"  SG  mean |corr|: {np.mean(sg):.5f}  (N={len(sg)})")
        print(f"  SCM mean |corr|: {np.mean(scm):.5f}  (N={len(scm)})")
        delta = np.mean(sg) - np.mean(scm)
        ratio = np.mean(sg) / np.mean(scm) if np.mean(scm) > 0 else float("inf")
        print(f"  Δ (SG - SCM):    {delta:+.5f}")
        print(f"  SG / SCM ratio:  {ratio:.2f}x")

        print(f"\n  Per-archetype comparison:")
        print(f"  {'Archetype':<16s} {'SG':>10s} {'SCM':>10s} {'Δ':>10s} {'Ratio':>8s}")
        print(f"  {'-'*58}")
        for arch in ["source", "ts_child", "non_ts_child"]:
            sg_a = np.array(sg_arch.get(arch, []))
            scm_a = np.array(scm_arch.get(arch, []))
            if len(sg_a) and len(scm_a):
                d = np.mean(sg_a) - np.mean(scm_a)
                r = np.mean(sg_a) / np.mean(scm_a) if np.mean(scm_a) > 0 else float("inf")
                print(f"  {arch:<16s} {np.mean(sg_a):10.5f} {np.mean(scm_a):10.5f} "
                      f"{d:+10.5f} {r:7.2f}x")

    print()


if __name__ == "__main__":
    main()
