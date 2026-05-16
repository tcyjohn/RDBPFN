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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import core quality logic from the RDB module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_generation", "RDB"))
from src.table_def.task_quality import diagnose_parquet_path  # noqa: E402


def _diagnose_one(path: str) -> dict:
    """Wrapper so ProcessPoolExecutor can pickle the function."""
    return diagnose_parquet_path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    for base_dir in args.dirs:
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

        # A-stage failure breakdown
        if a_rejected:
            a_counts: Dict[str, int] = {}
            for r in a_rejected:
                for f in r.get("a_failures", r.get("all_failures", [])):
                    key = "_".join(f.split("_")[:2])
                    a_counts[key] = a_counts.get(key, 0) + 1
            print(f"\n  Stage A failure breakdown:")
            for reason, cnt in sorted(a_counts.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {cnt}")

        # B-stage failure breakdown
        if b_rejected:
            b_counts: Dict[str, int] = {}
            for r in b_rejected:
                for f in r.get("all_failures", []):
                    key = f.split(":")[0].strip()[:30]
                    b_counts[key] = b_counts.get(key, 0) + 1
            print(f"\n  Stage B failure breakdown:")
            for reason, cnt in sorted(b_counts.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {cnt}")

        # AUC distribution
        et_aucs = [r["et_auc"] for r in results
                   if not np.isnan(r.get("et_auc", float("nan")))]
        if et_aucs:
            arr = np.array(et_aucs)
            print(f"\n  ExtraTrees OOF AUC ({len(arr)} tasks):")
            print(f"    mean={np.mean(arr):.4f}  median={np.median(arr):.4f}")
            print(f"    min={np.min(arr):.4f}  max={np.max(arr):.4f}")
            for th in [0.55, 0.60, 0.65, 0.70]:
                print(f"    frac > {th}: {np.mean(arr > th):.1%}")

        # Sample failures
        if a_rejected:
            print(f"\n  Sample A-rejected (first 5):")
            for r in a_rejected[:5]:
                name = os.path.basename(os.path.dirname(r.get("path", "?")))
                print(f"    {name}: {r.get('a_failures', r.get('all_failures', []))}")
        if b_rejected:
            print(f"\n  Sample B-rejected (first 5):")
            for r in b_rejected[:5]:
                name = os.path.basename(os.path.dirname(r.get("path", "?")))
                print(f"    {name}: AUC={r.get('et_auc', float('nan')):.4f}  {r.get('all_failures',[])}")


if __name__ == "__main__":
    main()
