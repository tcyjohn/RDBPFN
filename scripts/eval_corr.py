#!/usr/bin/env python3
"""
Evaluate cross-table feature correlation on preprocessed (post-DFS) data.

Metrics:
  - native_corr: correlation among a table's own feature columns
  - native-joined_corr: correlation between native features and joined parent features
  - joined_corr: correlation among joined parent feature columns (from same parent)
  - cross_corr: correlation between joined features from different parents (multi-parent only)

Results are split by single-parent vs multi-parent tasks.

Usage:
    pixi run python scripts/eval_corr.py \
        --raw_dir data_generation/RDB_datasets/baseline_no_structsig \
        --processed_dir data_generation/RDB_datasets/baseline_no_structsig-processed \
        --num_rdbs 64 --start_index 0 \
        --subsample 30
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_fk_relations(raw_dir: Path, rdb_idx: int) -> dict[str, list[str]]:
    """Parse metadata.yaml to map table_name -> [parent_table_names]."""
    meta_path = raw_dir / f"dag_rdb_{rdb_idx}" / "metadata.yaml"
    if not meta_path.exists():
        logger.warning("Metadata not found: %s", meta_path)
        return {}

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    fk_map: dict[str, list[str]] = {}
    for table in meta.get("tables", []):
        table_name = table["name"]
        parents = []
        for col in table.get("columns", []):
            if col.get("dtype") == "foreign_key":
                link_to = col.get("link_to", "")
                parent_name = link_to.split(".")[0]
                if parent_name:
                    parents.append(parent_name)
        fk_map[table_name] = parents
    return fk_map


def parse_task_target(task_dir_name: str) -> str | None:
    """Extract target table name from task dir like 'complex_task_table_8_relational_aggregation_prediction_4'."""
    m = re.match(r"(?:simple_|complex_)?task_table_(\d+)_", task_dir_name)
    if m:
        return f"table_{m.group(1)}"
    return None


def is_native_feature(col_name: str) -> bool:
    """Native feature: no table prefix, not a timestamp derivative, not a PK/FK/ID column."""
    if "." in col_name:
        return False
    if "_" in col_name:
        # Check if it's an aggregated feature like table_X_feature_Y_mean
        parts = col_name.split("_")
        if len(parts) >= 3 and parts[-1] in ("mean", "std", "min", "max", "sum", "count"):
            return False
    # Exclude PK, FK, timestamp derivatives, labels
    if col_name.endswith("_id"):
        return False
    if col_name in ("labels", "timestamp") or col_name.startswith("TIMESTAMP_"):
        return False
    if col_name in ("YEAR_timestamp", "MONTH_timestamp", "DAY_timestamp", "DAYOFWEEK_timestamp"):
        return False
    # Must be a feature column
    if not col_name.startswith("feature_"):
        return False
    return True


def is_joined_parent_feature(col_name: str) -> bool:
    """Joined parent feature: 'parent_table.feature_N' format."""
    if "." not in col_name:
        return False
    parts = col_name.split(".")
    if len(parts) != 2:
        return False
    return parts[1].startswith("feature_")


def get_joined_parent(col_name: str) -> str | None:
    """Extract parent table name from joined column like 'table_6.feature_0'."""
    if "." not in col_name:
        return None
    return col_name.split(".")[0]


def is_float_column(arr: np.ndarray) -> bool:
    """Check if array contains float values."""
    return np.issubdtype(arr.dtype, np.floating)


def compute_mean_abs_corr(cols: dict[str, np.ndarray]) -> float:
    """Compute mean absolute pairwise Pearson correlation among a set of columns."""
    if len(cols) < 2:
        return float("nan")
    names = list(cols.keys())
    n = len(names)
    corr_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = np.corrcoef(cols[names[i]], cols[names[j]])[0, 1]
            if not np.isnan(c):
                corr_sum += abs(c)
                count += 1
    return corr_sum / count if count > 0 else float("nan")


def compute_cross_corr(
    joined_cols: dict[str, dict[str, np.ndarray]], num_parents: int
) -> float:
    """Mean absolute correlation between joined features from DIFFERENT parents."""
    if num_parents < 2:
        return float("nan")
    parent_names = list(joined_cols.keys())
    if len(parent_names) < 2:
        return float("nan")
    corr_sum = 0.0
    count = 0
    for pi in range(len(parent_names)):
        for pj in range(pi + 1, len(parent_names)):
            p_i_cols = joined_cols[parent_names[pi]]
            p_j_cols = joined_cols[parent_names[pj]]
            for ci_name, ci_arr in p_i_cols.items():
                for cj_name, cj_arr in p_j_cols.items():
                    c = np.corrcoef(ci_arr, cj_arr)[0, 1]
                    if not np.isnan(c):
                        corr_sum += abs(c)
                        count += 1
    return corr_sum / count if count > 0 else float("nan")


def subsample_columns(
    col_dict: dict[str, np.ndarray], max_cols: int, rng: np.random.RandomState
) -> dict[str, np.ndarray]:
    """Randomly subsample to at most max_cols columns."""
    if len(col_dict) <= max_cols:
        return col_dict
    selected = rng.choice(sorted(col_dict.keys()), size=max_cols, replace=False)
    return {k: col_dict[k] for k in selected}


def evaluate_one_rdb(
    raw_dir: Path,
    processed_dir: Path,
    rdb_idx: int,
    subsample: int,
    rng: np.random.RandomState,
    dfs_suffix: str = "-dfs-1",
) -> dict[str, list[float]]:
    """Evaluate all tasks in one preprocessed RDB.

    Returns dict with keys like 'single_native_corr', 'multi_native_corr', etc.
    """
    fk_map = load_fk_relations(raw_dir, rdb_idx)
    proc_rdb_dir = processed_dir / f"dag_rdb_{rdb_idx}-{dfs_suffix}"
    if not proc_rdb_dir.exists():
        logger.warning("Preprocessed dir not found: %s", proc_rdb_dir)
        return {}

    results: dict[str, list[float]] = defaultdict(list)

    for task_dir in sorted(proc_rdb_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        target_table = parse_task_target(task_name)
        if target_table is None:
            continue

        train_file = task_dir / "train.npz"
        if not train_file.exists():
            continue

        parents = fk_map.get(target_table, [])
        num_parents = len(parents)
        if num_parents == 0:
            continue
        task_type = "single" if num_parents == 1 else "multi"

        data = np.load(train_file, allow_pickle=True)

        # Classify columns
        native_float_cols: dict[str, np.ndarray] = {}
        joined_by_parent: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

        for col_name in data.files:
            if col_name == "labels":
                continue
            arr = data[col_name]
            if not is_float_column(arr):
                continue

            if is_native_feature(col_name):
                native_float_cols[col_name] = arr
            elif is_joined_parent_feature(col_name):
                parent = get_joined_parent(col_name)
                if parent:
                    joined_by_parent[parent][col_name] = arr

        data.close()

        # Subsample to control dimensionality
        native_float_cols = subsample_columns(native_float_cols, subsample, rng)
        for parent in joined_by_parent:
            joined_by_parent[parent] = subsample_columns(joined_by_parent[parent], subsample, rng)

        # All joined columns (across all parents)
        all_joined_cols: dict[str, np.ndarray] = {}
        for p_cols in joined_by_parent.values():
            all_joined_cols.update(p_cols)

        # 1. native_corr
        nc = compute_mean_abs_corr(native_float_cols)
        if not np.isnan(nc):
            results[f"{task_type}_native_corr"].append(nc)

        # 2. native-joined_corr: between native and joined
        nj_corrs = []
        if native_float_cols and all_joined_cols:
            for nname, narr in native_float_cols.items():
                for jname, jarr in all_joined_cols.items():
                    c = np.corrcoef(narr, jarr)[0, 1]
                    if not np.isnan(c):
                        nj_corrs.append(abs(c))
        if nj_corrs:
            results[f"{task_type}_native_joined_corr"].append(float(np.mean(nj_corrs)))

        # 3. joined_corr: within each parent
        for parent, p_cols in joined_by_parent.items():
            jc = compute_mean_abs_corr(p_cols)
            if not np.isnan(jc):
                results[f"{task_type}_joined_corr"].append(jc)

        # 4. cross_corr: between different parents (multi-parent only)
        if num_parents >= 2:
            cc = compute_cross_corr(joined_by_parent, num_parents)
            if not np.isnan(cc):
                results[f"multi_cross_corr"].append(cc)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-table correlations")
    parser.add_argument("--raw_dir", type=str, required=True, help="Raw RDB output dir")
    parser.add_argument("--processed_dir", type=str, required=True, help="Preprocessed RDB dir")
    parser.add_argument("--num_rdbs", type=int, default=64)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=30, help="Max cols to subsample")
    parser.add_argument("--dfs_suffix", type=str, default="dfs-1", help="DFS suffix (e.g. dfs-1 or dfs-2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    dfs_suffix = args.dfs_suffix
    rng = np.random.RandomState(args.seed)

    all_results: dict[str, list[float]] = defaultdict(list)

    for idx in range(args.start_index, args.start_index + args.num_rdbs):
        rdb_results = evaluate_one_rdb(raw_dir, processed_dir, idx, args.subsample, rng, dfs_suffix)
        for key, vals in rdb_results.items():
            all_results[key].extend(vals)

    # Print summary
    metric_order = [
        ("single_native_corr", "native_corr (single-parent)"),
        ("multi_native_corr", "native_corr (multi-parent)"),
        ("single_native_joined_corr", "native-joined_corr (single-parent)"),
        ("multi_native_joined_corr", "native-joined_corr (multi-parent)"),
        ("single_joined_corr", "joined_corr (single-parent)"),
        ("multi_joined_corr", "joined_corr (multi-parent)"),
        ("multi_cross_corr", "cross_corr (multi-parent)"),
    ]

    print("=" * 70)
    print(f"Correlation Evaluation (subsample={args.subsample})")
    print(f"RDBs: {args.start_index}-{args.start_index + args.num_rdbs - 1}")
    print(f"Raw: {raw_dir}")
    print(f"Processed: {processed_dir}")
    print("=" * 70)
    print(f"{'Metric':<38} {'Mean':>8} {'Std':>8} {'Count':>6}")
    print("-" * 70)

    for key, label in metric_order:
        vals = all_results.get(key, [])
        if vals:
            print(f"{label:<38} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} {len(vals):>6}")
        else:
            print(f"{label:<38} {'N/A':>8} {'N/A':>8} {0:>6}")


if __name__ == "__main__":
    main()
