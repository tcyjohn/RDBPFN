#!/usr/bin/env python3
"""Detailed per-dataset correlation breakdown: real vs synthetic.

Uses original CSV headers (column names) to classify native/aggregated/joined columns,
then reads subsampled .npz files for the actual data.
"""

import csv
import re
import numpy as np
from collections import defaultdict
from pathlib import Path


AGG_FUNCTIONS = {
    "COUNT", "MAX", "MEAN", "MIN", "SUM", "STD", "MODE",
    "SKEW", "NUM_UNIQUE", "FIRST", "LAST", "ANY", "TREND",
    "N_MOST_COMMON", "PERCENT_TRUE", "N_UNIQUE",
}


def _get_agg_child(col_name: str) -> str | None:
    m = re.search(r"\.?[A-Z]+\(([^)]+)\)", col_name)
    if m:
        inner = m.group(1)
        child = inner.split(".")[0]
        child = re.sub(r"\[.*\]", "", child)
        return child
    return None


def _has_agg(col_name: str) -> bool:
    for agg in AGG_FUNCTIONS:
        if f".{agg}(" in col_name or col_name.startswith(f"{agg}("):
            return True
    return False


def _is_time_like(col_name: str) -> bool:
    prefixes = ("TIMESTAMP_", "YEAR_", "MONTH_", "DAY_", "DAYOFWEEK_",
                "QUARTER_", "WEEK_", "HOUR_", "MINUTE_", "SECOND_")
    for tp in prefixes:
        if col_name.startswith(tp) or f".{tp}" in col_name:
            return True
    return False


def classify_columns(col_names: list[str]) -> dict:
    """Classify column indices by type using DFS naming conventions."""
    # Identify target entity (most common prefix among non-agg, non-time cols)
    entity_counts: dict[str, int] = defaultdict(int)
    for c in col_names:
        if _has_agg(c) or _is_time_like(c):
            continue
        if "." in c:
            entity = c.split(".")[0]
            if not entity[0].isdigit():  # skip numeric-prefixed
                entity_counts[entity] += 1
    target_entity = max(entity_counts, key=entity_counts.get) if entity_counts else None

    native = []
    agg_by_child: dict[str, list[int]] = defaultdict(list)
    joined_by_parent: dict[str, list[int]] = defaultdict(list)
    other = []

    for i, c in enumerate(col_names):
        if _is_time_like(c):
            other.append(i)
            continue
        low = c.lower()
        if low in ("index", "target", "label") or low.endswith("_id") or low.endswith("id"):
            other.append(i)
            continue

        if _has_agg(c):
            child = _get_agg_child(c) or "__unknown__"
            agg_by_child[child].append(i)
        elif "." in c:
            entity = c.split(".")[0]
            if target_entity and entity != target_entity:
                joined_by_parent[entity].append(i)
            else:
                native.append(i)
        else:
            native.append(i)

    return {
        "native": native,
        "agg_by_child": dict(agg_by_child),
        "joined_by_parent": dict(joined_by_parent),
        "other": other,
        "target_entity": target_entity,
    }


def compute_mean_abs_corr(data: np.ndarray, indices: list[int]) -> float:
    if len(indices) < 2:
        return float("nan")
    cols = data[:, indices].astype(np.float64)
    stds = np.std(cols, axis=0)
    valid = np.nonzero(stds > 1e-10)[0]
    if len(valid) < 2:
        return float("nan")
    cols = cols[:, valid]
    corr_sum = 0.0
    count = 0
    for i in range(cols.shape[1]):
        for j in range(i + 1, cols.shape[1]):
            c = np.corrcoef(cols[:, i], cols[:, j])[0, 1]
            if not np.isnan(c):
                corr_sum += abs(c)
                count += 1
    return corr_sum / count if count > 0 else float("nan")


def compute_cross_source_corr(data: np.ndarray, groups: dict[str, list[int]]) -> float:
    group_names = [g for g in groups if len(groups[g]) >= 1]
    if len(group_names) < 2:
        return float("nan")
    corr_sum = 0.0
    count = 0
    for gi in range(len(group_names)):
        for gj in range(gi + 1, len(group_names)):
            idx_i = groups[group_names[gi]]
            idx_j = groups[group_names[gj]]
            for ii in idx_i:
                for jj in idx_j:
                    if np.std(data[:, ii]) < 1e-10 or np.std(data[:, jj]) < 1e-10:
                        continue
                    c = np.corrcoef(
                        data[:, ii].astype(np.float64),
                        data[:, jj].astype(np.float64),
                    )[0, 1]
                    if not np.isnan(c):
                        corr_sum += abs(c)
                        count += 1
    return corr_sum / count if count > 0 else float("nan")


def eval_one_dataset(csv_path: str, npz_path: str) -> dict:
    """Classify columns from CSV header, compute correlations from NPZ data."""
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))

    label_col = header[-1]
    numeric_cols = []
    numeric_header = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        row1 = next(reader)
        for i, v in enumerate(row1[:-1]):  # exclude label
            try:
                float(v)
                numeric_cols.append(i)
                numeric_header.append(header[i])
            except ValueError:
                pass

    cls = classify_columns(numeric_header)
    native = cls["native"]
    agg_by_child = cls["agg_by_child"]
    joined_by_parent = cls["joined_by_parent"]

    data = np.load(npz_path)
    X = data["X_train"]
    data.close()

    # Build result
    result = {
        "name": Path(npz_path).stem.replace("_split", ""),
        "n_rows": X.shape[0],
        "n_cols": X.shape[1],
        "n_native": len(native),
        "n_agg_groups": len(agg_by_child),
        "n_agg_cols": sum(len(v) for v in agg_by_child.values()),
        "n_joined_groups": len(joined_by_parent),
        "n_joined_cols": sum(len(v) for v in joined_by_parent.values()),
        "target_entity": cls["target_entity"],
    }

    # Overall: all columns
    all_indices = list(range(X.shape[1]))
    result["overall_corr"] = compute_mean_abs_corr(X, all_indices)

    result["native_corr"] = compute_mean_abs_corr(X, native)

    # Aggregated: within each child source
    agg_within = []
    for child, indices in agg_by_child.items():
        c = compute_mean_abs_corr(X, indices)
        if not np.isnan(c):
            agg_within.append(c)
    result["agg_within_corr"] = float(np.nanmean(agg_within)) if agg_within else float("nan")

    # Cross-child aggregated
    result["cross_agg_corr"] = compute_cross_source_corr(X, agg_by_child)

    # Joined from parents (rare, mostly absent in real relational data)
    if joined_by_parent:
        joined_within = []
        for parent, indices in joined_by_parent.items():
            c = compute_mean_abs_corr(X, indices)
            if not np.isnan(c):
                joined_within.append(c)
        result["joined_corr"] = float(np.nanmean(joined_within)) if joined_within else float("nan")
        result["cross_joined_corr"] = compute_cross_source_corr(X, joined_by_parent)
    else:
        result["joined_corr"] = float("nan")
        result["cross_joined_corr"] = float("nan")

    return result


# ============================================================
base = Path("/data/caijunyu/RDBPFN-hsbm-fk-generation/model_pretrain/datasets")

# ---- Real non-relational (clf_real) ----
print("=" * 110)
print("REAL NON-RELATIONAL (clf_real) — all columns native")
print("=" * 110)
print(f"{'Dataset':<45} {'rows':>7} {'ncols':>5} {'overall':>8} {'native_corr':>10}")
print("-" * 78)
all_real = []
for csv_path in sorted(base.joinpath("clf_real").glob("*.csv")):
    npz_path = base / "clf_real_subsamples" / (csv_path.stem + "_split.npz")
    if not npz_path.exists():
        continue
    r = eval_one_dataset(str(csv_path), str(npz_path))
    all_real.append(r)
    print(f"{r['name']:<45} {r['n_rows']:>7} {r['n_cols']:>5} {r['overall_corr']:>8.4f} {r['native_corr']:>10.4f}")

def _print_summary(label, vals_list, keys):
    print(f"\n--- {label} SUMMARY ---")
    for k in keys:
        arr = np.array([x.get(k, float("nan")) for x in vals_list])
        fin = arr[np.isfinite(arr)]
        if len(fin) > 1:
            print(f"  {k:<22}: mean={np.nanmean(arr):.4f}  median={np.nanmedian(arr):.4f}  "
                  f"std={np.nanstd(arr):.4f}  n={len(fin)}")
        elif len(fin) == 1:
            print(f"  {k:<22}: {fin[0]:.4f}  n=1")
        else:
            print(f"  {k:<22}: N/A")

_print_summary("Real Non-Relational", all_real, ["overall_corr", "native_corr"])

# ---- Real relational DFS-2 (clf_rel) ----
print()
print("=" * 130)
print("REAL RELATIONAL DFS-2 (clf_rel) — native + aggregated from children (+ rarely joined from parents)")
print("=" * 130)
hdr = (f"{'Dataset':<40} {'rows':>7} {'ncol':>4} {'nat':>4} {'aggG':>4} {'jG':>3} "
       f"{'overall':>8} {'native':>8} {'agg_wi':>7} {'crs_agg':>7}")
print(hdr)
print("-" * len(hdr))
all_rel = []
for csv_path in sorted(base.joinpath("clf_rel").glob("*.csv")):
    npz_path = base / "clf_rel_subsamples" / (csv_path.stem + "_split.npz")
    if not npz_path.exists():
        continue
    r = eval_one_dataset(str(csv_path), str(npz_path))
    all_rel.append(r)
    print(f"{r['name']:<40} {r['n_rows']:>7} {r['n_cols']:>4} "
          f"{r['n_native']:>4} {r['n_agg_groups']:>4} {r['n_joined_groups']:>3} "
          f"{r['overall_corr']:>8.4f} {r['native_corr']:>8.4f} "
          f"{r['agg_within_corr']:>7.4f} {r['cross_agg_corr']:>7.4f}")

_print_summary("Real Relational DFS-2", all_rel,
               ["overall_corr", "native_corr", "agg_within_corr", "cross_agg_corr"])

# ---- Synthetic overall correlation ----
def compute_synthetic_overall(processed_dir: str, num_rdbs: int, subsample: int = 30) -> list[float]:
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
            float_cols = [data[k] for k in data.files
                          if k != "labels" and np.issubdtype(data[k].dtype, np.floating)]
            data.close()
            if len(float_cols) < 2:
                continue
            X = np.column_stack(float_cols)
            all_idx = list(range(X.shape[1]))
            c = compute_mean_abs_corr(X, all_idx)
            if not np.isnan(c):
                results.append(c)
    return results

print()
print("=" * 70)
print("SYNTHETIC OVERALL CORRELATION (task-level, all float cols)")
print("=" * 70)
for label, d in [
    ("Baseline", "/data/caijunyu/RDBPFN-hsbm-fk-generation/data_generation/RDB_datasets/baseline_no_structsig-processed"),
    ("Final", "/data/caijunyu/RDBPFN-hsbm-fk-generation/data_generation/RDB_datasets/propensity_matching_64-processed"),
]:
    vals = compute_synthetic_overall(d, 64)
    if vals:
        arr = np.array(vals)
        fin = arr[np.isfinite(arr)]
        print(f"  {label}: overall_corr: mean={np.nanmean(arr):.4f}  median={np.nanmedian(arr):.4f}  "
              f"std={np.nanstd(arr):.4f}  n={len(fin)}")

print()
print("=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)
print(f"{'':<35} {'overall':>8} {'native':>8} {'within_src':>10} {'cross_src':>9}")
print("-" * 72)
print(f"{'Real non-rel (n=19)':<35} {0.1862:>8.4f} {0.1862:>8.4f} {'N/A':>10} {'N/A':>9}")
print(f"{'Real rel DFS-2 (n=19)':<35} {np.nanmean([r['overall_corr'] for r in all_rel]):>8.4f} "
      f"{np.nanmean([r['native_corr'] for r in all_rel]):>8.4f} "
      f"{np.nanmean([r['agg_within_corr'] for r in all_rel]):>10.4f} "
      f"{np.nanmean([r['cross_agg_corr'] for r in all_rel]):>9.4f}")
print(f"{'Synthetic Baseline (n=126)':<35} {'—':>8} {0.3413:>8.4f} {0.2866:>10.4f} {0.0231:>9.4f}")
print(f"{'Synthetic Final (n=129)':<35} {'—':>8} {0.3651:>8.4f} {0.3380:>10.4f} {0.0289:>9.4f}")
print()
print("Notes:")
print("  - Real 'within_src' = agg_within_corr (same-child aggregation)")
print("  - Synthetic 'within_src' = joined_corr single-parent (same-parent join)")
print("  - Real 'cross_src' = cross_agg_corr (different-child aggregation)")
print("  - Synthetic 'cross_src' = cross_corr multi-parent (different-parent join)")
print("  - Real rel data has NO joined-from-parent features; cross-source signal is from child aggregations")
print("  - Synthetic native_corr is inflated by signal-group feature generation (0.34 vs 0.19 real)")
