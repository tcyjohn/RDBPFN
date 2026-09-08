"""Two-stage task quality checker (CLI).

Usage:
    pixi run python scripts/diagnose_complex_tasks.py \
        --dirs data_generation/RDB_datasets/hsbm_v2.5 --all
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import core quality logic from the RDB module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_generation", "RDB"))
from src.table_def.task_quality import (  # noqa: E402
    diagnose_parquet_path,
    compute_feature_stats,
    compute_feature_stats_npz,
)


def _diagnose_one(path: str) -> dict:
    """Wrapper so ProcessPoolExecutor can pickle the function."""
    return diagnose_parquet_path(path)


def _collect_feature_stats(tasks: List[str]) -> List[dict]:
    """Collect per-task feature quality (pairwise corr + variance).

    Handles both .parquet (raw RDB tasks) and .npz (preprocessed datasets).
    """
    stats_list = []
    for i, path in enumerate(tasks):
        try:
            if path.endswith(".npz"):
                stats = compute_feature_stats_npz(path)
                stats["task"] = os.path.basename(path).replace("_split.npz", "")
            else:
                df = pd.read_parquet(path)
                stats = compute_feature_stats(df)
                stats["task"] = os.path.basename(os.path.dirname(path))
            stats_list.append(stats)
        except Exception as e:
            stats_list.append({
                "task": os.path.basename(path), "error": str(e),
            })
        if (i + 1) % 50 == 0:
            print(f"  feature stats ... {i + 1}/{len(tasks)}")
    return stats_list


def _collect_real_stats(npz_dir: str) -> List[dict]:
    """Collect per-dataset feature stats from real NPZ files."""
    stats_list = []
    npy_files = sorted(glob.glob(os.path.join(npz_dir, "*_split.npz")))
    for path in npy_files:
        try:
            stats = compute_feature_stats_npz(path)
            stats["dataset"] = os.path.basename(path).replace("_split.npz", "")
            stats_list.append(stats)
        except Exception as e:
            stats_list.append({"dataset": os.path.basename(path), "error": str(e)})
    return stats_list


def _print_feature_stats_table(stats_list: List[dict], label: str):
    """Print aggregate feature statistics from per-task stats."""
    corrs = [s["avg_abs_corr_p50"] for s in stats_list
             if not np.isnan(s.get("avg_abs_corr_p50", float("nan")))]
    vars_ = [s["feature_var_p50"] for s in stats_list
             if not np.isnan(s.get("feature_var_p50", float("nan")))]
    errs = [s for s in stats_list if "error" in s]

    print(f"\n{'─'*60}")
    print(f"  Feature quality: {label} ({len(stats_list)} datasets)")
    print(f"{'─'*60}")
    if errs:
        print(f"  Errors: {len(errs)}")

    for name, vals in [("avg |corr| p50", corrs), ("feature var p50", vars_)]:
        if not vals:
            print(f"  {name}: (no data)")
            continue
        arr = np.array(vals)
        print(f"  {name}:")
        print(f"    mean={np.mean(arr):.4f}  p25={np.percentile(arr,25):.4f}  "
              f"p50={np.median(arr):.4f}  p75={np.percentile(arr,75):.4f}")
        print(f"    min={np.min(arr):.4f}  max={np.max(arr):.4f}")
    return corrs, vars_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+")
    parser.add_argument("--tasks", nargs="+", help="Direct paths to train.parquet files")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--feature-stats", action="store_true",
                        help="Collect feature correlation/variance stats")
    parser.add_argument("--compare-real", nargs="+", default=None,
                        help="NPZ directories with real data for comparison")
    args = parser.parse_args()

    if not args.dirs and not args.tasks:
        parser.error("One of --dirs or --tasks is required")

    # Collect tasks
    all_tasks = []
    if args.dirs:
        for base_dir in args.dirs:
            pattern = os.path.join(base_dir, "dag_rdb_*", "complex_task_*", "train.parquet")
            paths = sorted(glob.glob(pattern))
            all_tasks.extend(paths)
    if args.tasks:
        all_tasks.extend(args.tasks)

    if not all_tasks:
        print("No tasks found")
        return

    if args.all or args.max_tasks is None or len(all_tasks) <= args.max_tasks:
        sample = all_tasks
    else:
        sample = np.random.RandomState(42).choice(
            all_tasks, args.max_tasks, replace=False).tolist()

    # --- Diagnostic mode ---
    if not args.feature_stats and not args.compare_real:
        for base_dir in (args.dirs or []):
            pattern = os.path.join(base_dir, "dag_rdb_*", "complex_task_*", "train.parquet")
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"No tasks found in {base_dir}")
                continue

            if args.all or args.max_tasks is None or len(paths) <= args.max_tasks:
                sample = paths
            else:
                sample = np.random.RandomState(42).choice(
                    paths, args.max_tasks, replace=False).tolist()

            print(f"\n{'='*70}")
            print(f"Directory: {base_dir}")
            print(f"Total tasks: {len(paths)}  |  Checked: {len(sample)}")
            print(f"{'='*70}")

            n_workers = min(cpu_count(), len(sample), 16)
            print(f"  Processing {len(sample)} tasks with {n_workers} workers...")
            results = []
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_diagnose_one, p): i for i, p in enumerate(sample)}
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    if len(results) % 50 == 0 or len(results) == len(sample):
                        print(f"  ... {len(results)}/{len(sample)}")

            passed = [r for r in results if r["passed"]]
            failed = [r for r in results if not r["passed"]]
            a_rejected = [r for r in failed if r["stage"] == "A"]
            b_rejected = [r for r in failed if r["stage"] == "B"]

            print(f"\n  Stage A rejected: {len(a_rejected)}")
            print(f"  Stage B rejected: {len(b_rejected)}")
            print(f"  FINAL PASSED:     {len(passed)} / {len(results)} "
                  f"({100*len(passed)/max(1,len(results)):.1f}%)")

            if a_rejected:
                a_counts: Dict[str, int] = {}
                for r in a_rejected:
                    for f in r.get("a_failures", r.get("all_failures", [])):
                        key = "_".join(f.split("_")[:2])
                        a_counts[key] = a_counts.get(key, 0) + 1
                print(f"\n  Stage A failure breakdown:")
                for reason, cnt in sorted(a_counts.items(), key=lambda x: -x[1]):
                    print(f"    {reason}: {cnt}")

            if b_rejected:
                b_counts: Dict[str, int] = {}
                for r in b_rejected:
                    for f in r.get("all_failures", []):
                        key = f.split(":")[0].strip()[:30]
                        b_counts[key] = b_counts.get(key, 0) + 1
                print(f"\n  Stage B failure breakdown:")
                for reason, cnt in sorted(b_counts.items(), key=lambda x: -x[1]):
                    print(f"    {reason}: {cnt}")

            et_aucs = [r["et_auc"] for r in results
                       if not np.isnan(r.get("et_auc", float("nan")))]
            if et_aucs:
                arr = np.array(et_aucs)
                print(f"\n  ExtraTrees OOF AUC ({len(arr)} tasks):")
                print(f"    mean={np.mean(arr):.4f}  median={np.median(arr):.4f}")
                print(f"    min={np.min(arr):.4f}  max={np.max(arr):.4f}")
                for th in [0.55, 0.60, 0.65, 0.70]:
                    print(f"    frac > {th}: {np.mean(arr > th):.1%}")

            if a_rejected:
                print(f"\n  Sample A-rejected (first 5):")
                for r in a_rejected[:5]:
                    name = os.path.basename(os.path.dirname(r.get("path", "?")))
                    print(f"    {name}: {r.get('a_failures', r.get('all_failures', []))}")
            if b_rejected:
                print(f"\n  Sample B-rejected (first 5):")
                for r in b_rejected[:5]:
                    name = os.path.basename(os.path.dirname(r.get("path", "?")))
                    print(f"    {name}: AUC={r.get('et_auc', float('nan')):.4f}  "
                          f"{r.get('all_failures',[])}")

        return

    # --- Feature-stats mode ---
    if args.feature_stats:
        print(f"\nCollecting feature stats from {len(sample)} tasks...")
        synth_stats = _collect_feature_stats(sample)
        synth_corrs, synth_vars = _print_feature_stats_table(
            synth_stats, "Synthetic (signal-group features)")

    # --- Compare with real data ---
    if args.compare_real:
        for real_dir in args.compare_real:
            if os.path.isdir(real_dir):
                real_stats = _collect_real_stats(real_dir)
                real_corrs, real_vars = _print_feature_stats_table(
                    real_stats, f"Real — {os.path.basename(real_dir)}")

                # Side-by-side summary
                if args.feature_stats and synth_corrs and real_corrs:
                    print(f"\n  {'='*60}")
                    print(f"  Comparison Summary: Synthetic vs {os.path.basename(real_dir)}")
                    print(f"  {'='*60}")
                    for name, s_vals, r_vals in [
                        ("avg |corr| p50", synth_corrs, real_corrs),
                        ("feature var p50", synth_vars, real_vars),
                    ]:
                        s_arr = np.array(s_vals)
                        r_arr = np.array(r_vals)
                        print(f"  {name}:")
                        print(f"    synthetic → p50={np.median(s_arr):.4f}  "
                              f"mean={np.mean(s_arr):.4f}")
                        print(f"    real      → p50={np.median(r_arr):.4f}  "
                              f"mean={np.mean(r_arr):.4f}")
                        print(f"    Δ (p50)   = {np.median(s_arr) - np.median(r_arr):.4f}")
            else:
                print(f"Directory not found: {real_dir}")


if __name__ == "__main__":
    main()
