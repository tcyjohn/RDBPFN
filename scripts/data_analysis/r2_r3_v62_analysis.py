#!/usr/bin/env python3
"""Run the preregistered R2--R3--v6.2 data analyses.

The script deliberately separates three identifiable units:

* raw RDB metrics are paired and resampled by ``dag_rdb_i``;
* H5/real-task feature summaries are descriptive corpus statistics;
* performance results are paired by evaluation task and seed.

No model training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "model_pretrain/results/r2_r3_v62_data_analysis"
DEFAULT_R2_RAW = Path(
    "/tmp/RDBPFN-ablation-r2/data_generation/RDB_datasets/"
    "ablation_r2_original_style_1024_raw"
)
DEFAULT_R3_RAW = Path(
    "/tmp/RDBPFN-ablation-r3/data_generation/RDB_datasets/"
    "ablation_r3_data_prior_1024_raw"
)
DEFAULT_H5 = {
    "R2 filtered": Path(
        "/tmp/RDBPFN-ablation-r2/model_pretrain/pretrain_datasets/"
        "ablation_r2_original_style_1024.h5"
    ),
    "R3 direct": Path(
        "/tmp/RDBPFN-ablation-r3/model_pretrain/pretrain_datasets/"
        "ablation_r3_data_prior_1024_unsampled.h5"
    ),
}
DEFAULT_PERFORMANCE_FILES = {
    "R2": REPO_ROOT
    / "model_pretrain/results/ablation_r2_original_filter_1024_00112_full512_aligned_per_seed.csv",
    "R3": REPO_ROOT
    / "model_pretrain/results/ablation_r3_data_prior_1024_direct_00112_full512_aligned_per_seed.csv",
    "No signal groups": REPO_ROOT
    / "model_pretrain/results/ablation_r25_no_signal_groups_00112_full512_aligned_per_seed.csv",
    "Low density": REPO_ROOT
    / "model_pretrain/results/ablation_r3_data_prior_1024_snapshot1_00112_full512_aligned_per_seed.csv",
    "Joint removal": REPO_ROOT
    / "model_pretrain/results/ablation_r25_no_signal_groups_1024_snapshot1_00112_full512_aligned_per_seed.csv",
    "v6.2": REPO_ROOT
    / "model_pretrain/results/v6.2_00112_full512_entity_aligned_per_seed.csv",
    "Legacy Mix": REPO_ROOT
    / "model_pretrain/results/mix_v62_r3filtered_entitybias_00112_full512_aligned_per_seed.csv",
    "Corrected Mix": REPO_ROOT
    / "model_pretrain/results/mix_v62_r3direct_entitybias_00112_full512_aligned_per_seed.csv",
}
RAW_METRICS = (
    "temporal_prevalence",
    "temporal_table_fraction",
    "calendar_activity_synchrony",
    "effective_parent_coverage",
    "multi_parent_fk_nmi",
    "normalized_effective_rank",
)


def stable_seed(text: str) -> int:
    """Return a process-independent uint32 seed derived from text."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def normalized_effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    """Entropy effective rank of a correlation matrix, normalized to [0, 1].

    Non-finite values are median-imputed per column. Constant columns are not
    eligible and are removed before the denominator is determined.
    """

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return float("nan")

    cleaned = np.empty_like(X, dtype=np.float64)
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        finite = np.isfinite(col)
        if not finite.any():
            cleaned[:, col_idx] = 0.0
            continue
        fill = float(np.median(col[finite]))
        cleaned[:, col_idx] = np.where(finite, col, fill)

    varying = np.std(cleaned, axis=0) > eps
    cleaned = cleaned[:, varying]
    n_features = cleaned.shape[1]
    if n_features < 2:
        return float("nan")

    corr = np.corrcoef(cleaned, rowvar=False)
    if corr.ndim != 2 or not np.isfinite(corr).all():
        return float("nan")
    eigenvalues = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= eps:
        return float("nan")
    weights = eigenvalues[eigenvalues > eps] / total
    entropy = float(-np.sum(weights * np.log(weights)))
    return float(np.exp(entropy) / n_features)


def strong_correlation_rate(
    X: np.ndarray, threshold: float = 0.9, eps: float = 1e-12
) -> float:
    """Fraction of eligible feature pairs with absolute Pearson r >= threshold."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return float("nan")
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        finite = np.isfinite(col)
        fill = float(np.median(col[finite])) if finite.any() else 0.0
        X[:, col_idx] = np.where(finite, col, fill)
    X = X[:, np.std(X, axis=0) > eps]
    if X.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(X, rowvar=False)
    values = np.abs(corr[np.triu_indices_from(corr, k=1)])
    values = values[np.isfinite(values)]
    return float(np.mean(values >= threshold)) if len(values) else float("nan")


def effective_parent_coverage(values: np.ndarray, n_parent_rows: int) -> float:
    """Compute exp(entropy(FK assignment)) / available parent rows."""

    values = np.asarray(values)
    valid = values[np.isfinite(values)].astype(np.int64, copy=False)
    valid = valid[(valid >= 0) & (valid < n_parent_rows)]
    if len(valid) == 0 or n_parent_rows <= 0:
        return float("nan")
    counts = np.unique(valid, return_counts=True)[1].astype(np.float64)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy) / n_parent_rows)


def multi_parent_nmi(left: np.ndarray, right: np.ndarray) -> float:
    """NMI for two FK assignments; undefined when either side is constant."""

    left = np.asarray(left)
    right = np.asarray(right)
    valid = np.isfinite(left) & np.isfinite(right) & (left >= 0) & (right >= 0)
    left = left[valid].astype(np.int64, copy=False)
    right = right[valid].astype(np.int64, copy=False)
    if len(left) < 2 or len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return float("nan")
    return float(normalized_mutual_info_score(left, right, average_method="arithmetic"))


def _load_metadata(rdb_dir: Path) -> dict[str, Any]:
    with (rdb_dir / "metadata.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def complete_rdb_ids(root: Path) -> list[str]:
    """List numeric-sorted RDB IDs that contain metadata and table parquet."""

    complete = []
    for rdb_dir in root.glob("dag_rdb_*"):
        if (rdb_dir / "metadata.yaml").is_file() and any(
            rdb_dir.glob("table_*.parquet")
        ):
            complete.append(rdb_dir.name)
    return sorted(complete, key=lambda name: int(name.rsplit("_", 1)[1]))


def _sample_rows(frame: pd.DataFrame, cap: int, key: str) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    rng = np.random.default_rng(stable_seed(key))
    indices = np.sort(rng.choice(len(frame), size=cap, replace=False))
    return frame.iloc[indices]


def monthly_activity_curve(values: pd.Series) -> np.ndarray | None:
    """Return a normalized 1970--2020 monthly activity histogram.

    The shared calendar prior is expressed on concrete day indices. Monthly
    aggregation retains calendar-scale variation while avoiding the extreme
    sparsity of daily histograms for the smaller R2 tables.
    """

    timestamps = pd.to_datetime(values, errors="coerce").dropna()
    if len(timestamps) < 2:
        return None
    month_indices = (
        (timestamps.dt.year.to_numpy(dtype=np.int64) - 1970) * 12
        + timestamps.dt.month.to_numpy(dtype=np.int64)
        - 1
    )
    month_indices = month_indices[(month_indices >= 0) & (month_indices < 612)]
    if len(month_indices) < 2:
        return None
    histogram = np.bincount(month_indices, minlength=612).astype(np.float64)
    total = float(histogram.sum())
    if total <= 0.0 or float(histogram.std()) <= 1e-12:
        return None
    return histogram / total


def analyze_one_raw_rdb(
    root: str, rdb_id: str, source: str, row_cap: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return RDB-level metrics and their table/relationship-level components."""

    rdb_dir = Path(root) / rdb_id
    metadata = _load_metadata(rdb_dir)
    tables = {table["name"]: table for table in metadata.get("tables", [])}
    unit_rows: list[dict[str, Any]] = []

    temporal_tables = 0
    temporal_curves: list[tuple[str, np.ndarray]] = []
    for table_name, table in tables.items():
        source_path = rdb_dir / table["source"]
        time_column = table.get("time_column")
        if time_column and time_column in pq.read_schema(source_path).names:
            temporal_tables += 1
            timestamp_frame = pq.read_table(
                source_path, columns=[time_column]
            ).to_pandas()
            curve = monthly_activity_curve(timestamp_frame[time_column])
            if curve is not None:
                temporal_curves.append((table_name, curve))
    temporal_value = float(temporal_tables > 0)
    temporal_table_fraction = (
        float(temporal_tables / len(tables)) if tables else float("nan")
    )
    temporal_pair_correlations = []
    for (left_name, left), (right_name, right) in itertools.combinations(
        temporal_curves, 2
    ):
        correlation = float(np.corrcoef(left, right)[0, 1])
        if not np.isfinite(correlation):
            continue
        temporal_pair_correlations.append(correlation)
        unit_rows.append(
            {
                "source": source,
                "rdb_id": rdb_id,
                "metric": "calendar_activity_synchrony",
                "unit": f"{left_name}::{right_name}",
                "value": correlation,
                "eligible_rows": 612,
            }
        )
    unit_rows.append(
        {
            "source": source,
            "rdb_id": rdb_id,
            "metric": "temporal_prevalence",
            "unit": rdb_id,
            "value": temporal_value,
            "eligible_rows": len(tables),
        }
    )
    unit_rows.append(
        {
            "source": source,
            "rdb_id": rdb_id,
            "metric": "temporal_table_fraction",
            "unit": rdb_id,
            "value": temporal_table_fraction,
            "eligible_rows": len(tables),
        }
    )

    coverage_values: list[float] = []
    nmi_values: list[float] = []
    rank_values: list[float] = []

    for table_name, table in tables.items():
        source_path = rdb_dir / table["source"]
        columns = table.get("columns", [])
        fk_columns = [col for col in columns if col.get("dtype") == "foreign_key"]
        feature_columns = [
            col["name"]
            for col in columns
            if col.get("dtype") in {"float", "category"}
            and col["name"].startswith("feature_")
        ]

        required = [col["name"] for col in fk_columns] + feature_columns
        if not required:
            continue
        frame = pq.read_table(source_path, columns=required).to_pandas()

        for fk_column in fk_columns:
            parent_name = fk_column["link_to"].split(".", 1)[0]
            parent = tables.get(parent_name)
            if parent is None:
                continue
            n_parent_rows = pq.ParquetFile(rdb_dir / parent["source"]).metadata.num_rows
            value = effective_parent_coverage(frame[fk_column["name"]].to_numpy(), n_parent_rows)
            if np.isfinite(value):
                coverage_values.append(value)
                unit_rows.append(
                    {
                        "source": source,
                        "rdb_id": rdb_id,
                        "metric": "effective_parent_coverage",
                        "unit": f"{table_name}.{fk_column['name']}",
                        "value": value,
                        "eligible_rows": int(frame[fk_column["name"]].notna().sum()),
                    }
                )

        for left, right in itertools.combinations(fk_columns, 2):
            value = multi_parent_nmi(
                frame[left["name"]].to_numpy(), frame[right["name"]].to_numpy()
            )
            if np.isfinite(value):
                nmi_values.append(value)
                unit_rows.append(
                    {
                        "source": source,
                        "rdb_id": rdb_id,
                        "metric": "multi_parent_fk_nmi",
                        "unit": f"{table_name}.{left['name']}::{right['name']}",
                        "value": value,
                        "eligible_rows": int(
                            (
                                frame[left["name"]].notna()
                                & frame[right["name"]].notna()
                            ).sum()
                        ),
                    }
                )

        if len(feature_columns) >= 2:
            feature_frame = _sample_rows(
                frame[feature_columns], row_cap, f"{rdb_id}:{table_name}:feature-rank"
            )
            value = normalized_effective_rank(feature_frame.to_numpy(dtype=np.float64))
            if np.isfinite(value):
                rank_values.append(value)
                unit_rows.append(
                    {
                        "source": source,
                        "rdb_id": rdb_id,
                        "metric": "normalized_effective_rank",
                        "unit": table_name,
                        "value": value,
                        "eligible_rows": len(feature_frame),
                    }
                )

    aggregates = {
        "temporal_prevalence": [temporal_value],
        "temporal_table_fraction": [temporal_table_fraction],
        "calendar_activity_synchrony": temporal_pair_correlations,
        "effective_parent_coverage": coverage_values,
        "multi_parent_fk_nmi": nmi_values,
        "normalized_effective_rank": rank_values,
    }
    rdb_rows = []
    for metric, values in aggregates.items():
        rdb_rows.append(
            {
                "source": source,
                "rdb_id": rdb_id,
                "metric": metric,
                "value": float(np.median(values)) if values else float("nan"),
                "n_units": len(values),
            }
        )
    return rdb_rows, unit_rows


def paired_bootstrap_summary(
    rdb_metrics: pd.DataFrame, draws: int = 10_000, seed: int = 20270720
) -> pd.DataFrame:
    """Summarize paired R3-minus-R2 effects by resampling RDB identifiers."""

    rows = []
    for metric in RAW_METRICS:
        subset = rdb_metrics[rdb_metrics["metric"] == metric]
        wide = subset.pivot(index="rdb_id", columns="source", values="value")
        wide = wide.dropna(subset=["R2", "R3"])
        r2 = wide["R2"].to_numpy(dtype=np.float64)
        r3 = wide["R3"].to_numpy(dtype=np.float64)
        differences = r3 - r2
        n_pairs = len(differences)
        if n_pairs == 0:
            continue
        rng = np.random.default_rng(seed + stable_seed(metric))
        boot = np.empty(draws, dtype=np.float64)
        for start in range(0, draws, 500):
            size = min(500, draws - start)
            indices = rng.integers(0, n_pairs, size=(size, n_pairs))
            boot[start : start + size] = np.median(differences[indices], axis=1)
        rows.append(
            {
                "metric": metric,
                "n_pairs": n_pairs,
                "r2_median": float(np.median(r2)),
                "r2_q1": float(np.quantile(r2, 0.25)),
                "r2_q3": float(np.quantile(r2, 0.75)),
                "r3_median": float(np.median(r3)),
                "r3_q1": float(np.quantile(r3, 0.25)),
                "r3_q3": float(np.quantile(r3, 0.75)),
                "median_paired_difference": float(np.median(differences)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "r3_higher_fraction": float(np.mean(differences > 1e-12)),
                "tied_fraction": float(np.mean(np.abs(differences) <= 1e-12)),
                "r3_lower_fraction": float(np.mean(differences < -1e-12)),
            }
        )
    return pd.DataFrame(rows)


def run_raw_analysis(args: argparse.Namespace) -> None:
    r2_root = Path(args.r2_raw)
    r3_root = Path(args.r3_raw)
    common = sorted(
        set(complete_rdb_ids(r2_root)) & set(complete_rdb_ids(r3_root)),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if args.limit is not None:
        common = common[: args.limit]
    if not common:
        raise RuntimeError("No complete paired RDB identifiers were found")

    jobs = [
        (str(root), rdb_id, source, args.row_cap)
        for rdb_id in common
        for source, root in (("R2", r2_root), ("R3", r3_root))
    ]
    if args.workers == 1:
        results = [analyze_one_raw_rdb(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_raw_job, jobs, chunksize=4))

    rdb_rows = [row for result in results for row in result[0]]
    unit_rows = [row for result in results for row in result[1]]
    rdb_frame = pd.DataFrame(rdb_rows)
    unit_frame = pd.DataFrame(unit_rows)
    summary = paired_bootstrap_summary(rdb_frame, draws=args.bootstrap_draws)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rdb_frame.to_csv(output / "raw_rdb_metrics.csv", index=False)
    unit_frame.to_csv(output / "raw_unit_metrics.csv", index=False)
    summary.to_csv(output / "raw_paired_summary.csv", index=False)
    (output / "paired_rdb_ids.txt").write_text("\n".join(common) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote raw analysis to {output}")


def _raw_job(job: tuple[str, str, str, int]):
    return analyze_one_raw_rdb(*job)


def _schema_topology(metadata: dict[str, Any]) -> tuple[int, int, int]:
    tables = metadata.get("tables", [])
    names = [table["name"] for table in tables]
    parents: dict[str, set[str]] = {name: set() for name in names}
    n_edges = 0
    for table in tables:
        for column in table.get("columns", []):
            if column.get("dtype") != "foreign_key":
                continue
            parent = column.get("link_to", "").split(".", 1)[0]
            if parent and parent not in parents[table["name"]]:
                parents[table["name"]].add(parent)
                n_edges += 1

    depths: dict[str, int] = {}

    def depth(name: str, active: set[str]) -> int:
        if name in depths:
            return depths[name]
        if name in active:
            raise ValueError("Schema metadata contains a cycle")
        if not parents[name]:
            depths[name] = 0
        else:
            depths[name] = 1 + max(depth(parent, active | {name}) for parent in parents[name])
        return depths[name]

    max_depth = max((depth(name, set()) for name in names), default=0)
    return len(names), n_edges, max_depth


def run_attrition_analysis(args: argparse.Namespace) -> None:
    r2_root = Path(args.r2_raw)
    r3_complete = set(complete_rdb_ids(Path(args.r3_raw)))
    rows = []
    for rdb_id in complete_rdb_ids(r2_root):
        n_tables, n_edges, depth = _schema_topology(_load_metadata(r2_root / rdb_id))
        rows.append(
            {
                "rdb_id": rdb_id,
                "r3_status": "complete" if rdb_id in r3_complete else "missing",
                "n_tables": n_tables,
                "n_edges": n_edges,
                "dag_depth": depth,
            }
        )
    frame = pd.DataFrame(rows)
    summaries = []
    for status, group in frame.groupby("r3_status"):
        for metric in ["n_tables", "n_edges", "dag_depth"]:
            summaries.append(
                {
                    "r3_status": status,
                    "metric": metric,
                    "n_rdbs": len(group),
                    "median": float(group[metric].median()),
                    "q1": float(group[metric].quantile(0.25)),
                    "q3": float(group[metric].quantile(0.75)),
                    "mean": float(group[metric].mean()),
                }
            )
    summary = pd.DataFrame(summaries)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "attrition_schema_metrics.csv", index=False)
    summary.to_csv(output / "attrition_schema_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote attrition audit to {output}")


def _prepare_feature_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        finite = np.isfinite(col)
        fill = float(np.median(col[finite])) if finite.any() else 0.0
        X[:, col_idx] = np.where(finite, col, fill)
    return X


def analyze_h5_corpus(path: Path, corpus: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for task_idx in range(handle["X"].shape[0]):
            n_rows = int(handle["num_datapoints"][task_idx])
            n_features = int(handle["num_features"][task_idx])
            X = handle["X"][task_idx, :n_rows, :n_features]
            X = _prepare_feature_matrix(X)
            rows.append(
                {
                    "corpus": corpus,
                    "task_id": str(task_idx),
                    "n_rows": n_rows,
                    "n_features": n_features,
                    "normalized_effective_rank": normalized_effective_rank(X),
                    "strong_corr_rate_080": strong_correlation_rate(X.copy(), 0.80),
                    "strong_corr_rate_090": strong_correlation_rate(X.copy(), 0.90),
                    "strong_corr_rate_095": strong_correlation_rate(X.copy(), 0.95),
                }
            )
    return rows


def analyze_real_tasks(row_cap: int = 600) -> list[dict[str, Any]]:
    from model_pretrain.src.dbinfer_bench_simplified.dataset_meta import DBBTaskType
    from model_pretrain.src.dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
    from model_pretrain.src.eval_utils import load_task_split

    config_path = REPO_ROOT / "model_pretrain/conf_eval/dataset/full-512.yaml"
    config = yaml.safe_load(config_path.read_text())
    rows: list[dict[str, Any]] = []
    for relative in config["paths"]:
        dataset = DBBRDBDataset(REPO_ROOT / "model_pretrain" / relative)
        for task in dataset.tasks:
            if task.metadata.task_type != DBBTaskType.classification:
                continue
            X_train, *_ = load_task_split(task, "train")
            X_test, *_ = load_task_split(task, "test")
            X = np.concatenate([X_train, X_test], axis=0)
            if len(X) > row_cap:
                rng = np.random.default_rng(
                    stable_seed(f"{dataset.dataset_name}:{task.metadata.name}:rank")
                )
                indices = np.sort(rng.choice(len(X), size=row_cap, replace=False))
                X = X[indices]
            X = _prepare_feature_matrix(X)
            rows.append(
                {
                    "corpus": "Real DFS",
                    "task_id": f"{dataset.dataset_name}/{task.metadata.name}",
                    "n_rows": len(X),
                    "n_features": X.shape[1],
                    "normalized_effective_rank": normalized_effective_rank(X),
                    "strong_corr_rate_080": strong_correlation_rate(X.copy(), 0.80),
                    "strong_corr_rate_090": strong_correlation_rate(X.copy(), 0.90),
                    "strong_corr_rate_095": strong_correlation_rate(X.copy(), 0.95),
                }
            )
    return rows


def _median_exemplar(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame.dropna(subset=["normalized_effective_rank"]).copy()
    median = float(eligible["normalized_effective_rank"].median())
    eligible["distance"] = (eligible["normalized_effective_rank"] - median).abs()
    chosen = eligible.sort_values(["distance", "task_id"]).iloc[0]
    return {
        "task_id": str(chosen["task_id"]),
        "normalized_effective_rank": float(chosen["normalized_effective_rank"]),
        "corpus_median": median,
    }


def run_h5_analysis(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    h5_paths = {
        "R2 filtered": Path(args.r2_h5),
        "R3 direct": Path(args.r3_h5),
    }
    for corpus, path in h5_paths.items():
        rows.extend(analyze_h5_corpus(path, corpus))
    if not args.skip_real:
        rows.extend(analyze_real_tasks(row_cap=args.real_row_cap))
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("corpus", sort=False)
        .agg(
            n_tasks=("task_id", "count"),
            effective_rank_median=("normalized_effective_rank", "median"),
            effective_rank_q1=("normalized_effective_rank", lambda x: x.quantile(0.25)),
            effective_rank_q3=("normalized_effective_rank", lambda x: x.quantile(0.75)),
            strong_corr_090_median=("strong_corr_rate_090", "median"),
        )
        .reset_index()
    )
    exemplars = {
        corpus: _median_exemplar(group)
        for corpus, group in frame.groupby("corpus", sort=False)
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "h5_task_metrics.csv", index=False)
    summary.to_csv(output / "h5_corpus_summary.csv", index=False)
    (output / "h5_median_exemplars.json").write_text(
        json.dumps(exemplars, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(exemplars, indent=2))
    print(f"Wrote H5/real-task analysis to {output}")


def _entity_overlap(support: np.ndarray, query: np.ndarray) -> dict[str, float]:
    support = np.asarray(support, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    support_valid = support[support >= 0]
    query_valid = query[query >= 0]
    if len(query_valid) == 0:
        return {
            "valid_query_fraction": 0.0,
            "query_support_overlap": float("nan"),
            "matched_support_count_median": float("nan"),
            "matched_support_count_p90": float("nan"),
            "n_query_valid": 0,
            "n_query_matched": 0,
        }
    values, counts = np.unique(support_valid, return_counts=True)
    lookup = dict(zip(values.tolist(), counts.tolist()))
    matched = np.asarray([lookup.get(int(value), 0) for value in query_valid])
    positive = matched[matched > 0]
    return {
        "valid_query_fraction": float(len(query_valid) / len(query)) if len(query) else float("nan"),
        "query_support_overlap": float(np.mean(matched > 0)),
        "matched_support_count_median": float(np.median(positive)) if len(positive) else 0.0,
        "matched_support_count_p90": float(np.quantile(positive, 0.9)) if len(positive) else 0.0,
        "n_query_valid": int(len(query_valid)),
        "n_query_matched": int(np.sum(matched > 0)),
    }


def analyze_training_entity_exposure(path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(path, "r") as handle:
        if "entity_ids" not in handle:
            raise ValueError(f"{path} has no entity_ids dataset")
        for task_idx in range(handle["entity_ids"].shape[0]):
            n_rows = int(handle["num_datapoints"][task_idx])
            split = int(handle["single_eval_pos"][task_idx])
            entity_ids = handle["entity_ids"][task_idx, :n_rows]
            exposure = _entity_overlap(entity_ids[:split], entity_ids[split:])
            rows.append(
                {
                    "scope": "training",
                    "dataset": "v6.2 H5",
                    "task": str(task_idx),
                    "seed": -1,
                    "valid_all_row_fraction": float(np.mean(entity_ids >= 0)),
                    **exposure,
                }
            )
    return pd.DataFrame(rows)


def analyze_evaluation_entity_exposure() -> pd.DataFrame:
    from model_pretrain.src.dbinfer_bench_simplified.dataset_meta import DBBTaskType
    from model_pretrain.src.dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
    from model_pretrain.src.eval_utils import _stable_random_state, load_task_split

    config_path = REPO_ROOT / "model_pretrain/conf_eval/dataset/full-512.yaml"
    config = yaml.safe_load(config_path.read_text())
    rows = []
    for relative in config["paths"]:
        dataset = DBBRDBDataset(REPO_ROOT / "model_pretrain" / relative)
        for task in dataset.tasks:
            if task.metadata.task_type != DBBTaskType.classification:
                continue
            _, _, _, train_ids, _ = load_task_split(
                task, "train", use_primary_key_as_entity_id=True
            )
            _, _, _, test_ids, _ = load_task_split(
                task, "test", use_primary_key_as_entity_id=True
            )
            for seed in config["seeds"]:
                if train_ids is None or test_ids is None:
                    rows.append(
                        {
                            "scope": "evaluation",
                            "dataset": dataset.dataset_name,
                            "task": task.metadata.name,
                            "seed": seed,
                            "valid_all_row_fraction": float("nan"),
                            "valid_query_fraction": 0.0,
                            "query_support_overlap": float("nan"),
                            "matched_support_count_median": float("nan"),
                            "matched_support_count_p90": float("nan"),
                            "n_query_valid": 0,
                            "n_query_matched": 0,
                        }
                    )
                    continue
                indices = np.arange(len(train_ids))
                if len(indices) > config["max_train_samples"]:
                    key = f"{dataset.dataset_name}:{task.metadata.name}:{seed}"
                    rng = np.random.default_rng(_stable_random_state(key))
                    indices = np.sort(
                        rng.choice(
                            indices, size=config["max_train_samples"], replace=False
                        )
                    )
                query = test_ids
                if len(query) > 50_000:
                    key = f"{dataset.dataset_name}:{task.metadata.name}:{seed}:test"
                    rng = np.random.default_rng(_stable_random_state(key))
                    selected = np.sort(rng.choice(len(query), size=50_000, replace=False))
                    query = query[selected]
                exposure = _entity_overlap(train_ids[indices], query)
                rows.append(
                    {
                        "scope": "evaluation",
                        "dataset": dataset.dataset_name,
                        "task": task.metadata.name,
                        "seed": seed,
                        "valid_all_row_fraction": float(
                            np.mean(np.concatenate([train_ids[indices], query]) >= 0)
                        ),
                        **exposure,
                    }
                )
    return pd.DataFrame(rows)


def learned_entity_biases(checkpoint: Path) -> pd.DataFrame:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    rows = []
    for key, value in state.items():
        if key.endswith("entity_bias.raw_lambda"):
            raw = float(value.detach().cpu())
            strength = float(np.logaddexp(0.0, raw))
            block = int(key.split(".")[1])
            rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "transformer_block": block,
                    "raw_lambda": raw,
                    "bias_strength": strength,
                    "initial_strength": 0.1,
                    "change_from_initial": strength - 0.1,
                }
            )
    return pd.DataFrame(rows).sort_values("transformer_block")


def run_entity_analysis(args: argparse.Namespace) -> None:
    training = analyze_training_entity_exposure(Path(args.v62_h5))
    evaluation = analyze_evaluation_entity_exposure()
    exposure = pd.concat([training, evaluation], ignore_index=True)
    evaluation_task = (
        evaluation.groupby(["dataset", "task"], dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            valid_query_fraction=("valid_query_fraction", "mean"),
            query_support_overlap=("query_support_overlap", "mean"),
            matched_support_count_median=("matched_support_count_median", "median"),
            matched_support_count_p90=("matched_support_count_p90", "median"),
        )
        .reset_index()
    )
    biases = learned_entity_biases(Path(args.checkpoint))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    exposure.to_csv(output / "entity_exposure_per_seed.csv", index=False)
    evaluation_task.to_csv(output / "entity_exposure_evaluation_tasks.csv", index=False)
    biases.to_csv(output / "learned_entity_biases.csv", index=False)

    training_valid = training["query_support_overlap"].dropna()
    eval_valid = evaluation_task["query_support_overlap"].dropna()
    summary = {
        "training_tasks": len(training),
        "training_valid_id_fraction": float(training["valid_all_row_fraction"].mean()),
        "training_overlap_mean": float(training_valid.mean()),
        "training_overlap_median": float(training_valid.median()),
        "training_overlap_row_weighted": float(
            training["n_query_matched"].sum() / training["n_query_valid"].sum()
        ),
        "evaluation_tasks": len(evaluation_task),
        "evaluation_tasks_with_ids": len(eval_valid),
        "evaluation_overlap_mean": float(eval_valid.mean()),
        "evaluation_overlap_median": float(eval_valid.median()),
        "bias_strengths": biases["bias_strength"].tolist(),
    }
    (output / "entity_exposure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote entity analysis to {output}")


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Two-sided exact randomization p-value for paired differences."""

    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(differences)))
    if n <= 20:
        means = []
        for bits in range(1 << n):
            signs = np.fromiter(
                (1.0 if (bits >> idx) & 1 else -1.0 for idx in range(n)),
                dtype=np.float64,
                count=n,
            )
            means.append(abs(float(np.mean(signs * differences))))
        return float(np.mean(np.asarray(means) >= observed - 1e-15))
    raise ValueError("Exact sign-flip enumeration is limited to 20 pairs")


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(len(pvalues), np.nan)
    valid_indices = np.flatnonzero(np.isfinite(pvalues))
    ordered = valid_indices[np.argsort(pvalues[valid_indices])]
    running = 0.0
    m = len(ordered)
    for rank, original_idx in enumerate(ordered):
        candidate = (m - rank) * pvalues[original_idx]
        running = max(running, candidate)
        adjusted[original_idx] = min(1.0, running)
    return adjusted


def paired_t_interval(differences: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return float("nan"), float("nan")
    interval = stats.t.interval(
        confidence,
        len(differences) - 1,
        loc=float(np.mean(differences)),
        scale=float(stats.sem(differences)),
    )
    return float(interval[0]), float(interval[1])


def _load_performance(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["metric"].astype(str).str.lower().eq("auroc")].copy()
    frame["variant"] = label
    return frame[["variant", "dataset", "task", "seed", "metric_value"]]


def compare_performance(
    left: pd.DataFrame, right: pd.DataFrame, comparison: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["dataset", "task", "seed"]
    joined = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    joined["difference"] = joined["metric_value_left"] - joined["metric_value_right"]
    task_rows = []
    for (dataset, task), group in joined.groupby(["dataset", "task"], sort=True):
        differences = group["difference"].to_numpy()
        ci_low, ci_high = paired_t_interval(differences)
        task_rows.append(
            {
                "comparison": comparison,
                "dataset": dataset,
                "task": task,
                "n_seeds": len(differences),
                "left_mean": float(group["metric_value_left"].mean()),
                "right_mean": float(group["metric_value_right"].mean()),
                "mean_difference": float(np.mean(differences)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "exact_p": exact_sign_flip_pvalue(differences),
            }
        )
    task_frame = pd.DataFrame(task_rows)
    task_frame["holm_p"] = holm_adjust(task_frame["exact_p"])
    task_frame["holm_significant"] = task_frame["holm_p"] < 0.05

    per_seed = (
        joined.groupby("seed")[["metric_value_left", "metric_value_right"]]
        .mean()
        .reset_index()
    )
    suite_differences = (
        per_seed["metric_value_left"] - per_seed["metric_value_right"]
    ).to_numpy()
    ci_low, ci_high = paired_t_interval(suite_differences)
    suite = {
        "comparison": comparison,
        "n_tasks": len(task_frame),
        "n_seeds": len(per_seed),
        "left_mean": float(per_seed["metric_value_left"].mean()),
        "right_mean": float(per_seed["metric_value_right"].mean()),
        "mean_difference": float(np.mean(suite_differences)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "exact_p": exact_sign_flip_pvalue(suite_differences),
        "left_task_wins": int((task_frame["mean_difference"] > 0).sum()),
        "right_task_wins": int((task_frame["mean_difference"] < 0).sum()),
        "holm_significant_tasks": int(task_frame["holm_significant"].sum()),
    }
    return task_frame, suite


def run_performance_analysis(args: argparse.Namespace) -> None:
    frames = {
        label: _load_performance(path, label)
        for label, path in DEFAULT_PERFORMANCE_FILES.items()
    }
    comparisons = [
        ("R3", "R2"),
        ("R3", "No signal groups"),
        ("Low density", "Joint removal"),
        ("R3", "Low density"),
        ("No signal groups", "Joint removal"),
        ("R3", "Joint removal"),
        ("v6.2", "R3"),
        ("Legacy Mix", "R3"),
        ("Corrected Mix", "R3"),
        ("Corrected Mix", "v6.2"),
    ]
    task_frames = []
    suite_rows = []
    for left, right in comparisons:
        label = f"{left} - {right}"
        task_frame, suite = compare_performance(frames[left], frames[right], label)
        task_frames.append(task_frame)
        suite_rows.append(suite)
    tasks = pd.concat(task_frames, ignore_index=True)
    suites = pd.DataFrame(suite_rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(output / "performance_task_deltas.csv", index=False)
    suites.to_csv(output / "performance_suite_summary.csv", index=False)

    sensitivity_variants = [
        "R3",
        "No signal groups",
        "Low density",
        "Joint removal",
    ]
    seed_means = pd.concat(
        [
            frames[label]
            .groupby("seed", as_index=False)["metric_value"]
            .mean()
            .assign(variant=label)
            for label in sensitivity_variants
        ],
        ignore_index=True,
    )
    seed_grid = seed_means.pivot(
        index="seed", columns="variant", values="metric_value"
    ).reset_index()
    seed_grid["descriptive_interaction"] = (
        seed_grid["R3"]
        - seed_grid["No signal groups"]
        - seed_grid["Low density"]
        + seed_grid["Joint removal"]
    )
    seed_grid.to_csv(output / "performance_sensitivity_per_seed.csv", index=False)

    interaction = seed_grid["descriptive_interaction"].to_numpy()
    interaction_low, interaction_high = paired_t_interval(interaction)
    pd.DataFrame(
        [
            {
                "estimand": "descriptive_difference_in_differences",
                "n_support_samplings": len(interaction),
                "mean": float(np.mean(interaction)),
                "ci_low": interaction_low,
                "ci_high": interaction_high,
                "exact_p": exact_sign_flip_pvalue(interaction),
            }
        ]
    ).to_csv(output / "performance_sensitivity_interaction.csv", index=False)
    print(suites.to_string(index=False))
    print("\nHolm-significant task deltas:")
    print(
        tasks[tasks["holm_significant"]][
            ["comparison", "dataset", "task", "mean_difference", "holm_p"]
        ].to_string(index=False)
    )
    print(f"Wrote performance analysis to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw = subparsers.add_parser("raw", help="Analyze paired raw RDBs")
    raw.add_argument("--r2-raw", type=Path, default=DEFAULT_R2_RAW)
    raw.add_argument("--r3-raw", type=Path, default=DEFAULT_R3_RAW)
    raw.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    raw.add_argument("--limit", type=int)
    raw.add_argument("--workers", type=int, default=8)
    raw.add_argument("--row-cap", type=int, default=4096)
    raw.add_argument("--bootstrap-draws", type=int, default=10_000)
    raw.set_defaults(func=run_raw_analysis)

    attrition = subparsers.add_parser(
        "attrition", help="Audit coarse schema topology for R3 completion"
    )
    attrition.add_argument("--r2-raw", type=Path, default=DEFAULT_R2_RAW)
    attrition.add_argument("--r3-raw", type=Path, default=DEFAULT_R3_RAW)
    attrition.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    attrition.set_defaults(func=run_attrition_analysis)

    h5 = subparsers.add_parser("h5", help="Analyze H5 and real DFS correlations")
    h5.add_argument("--r2-h5", type=Path, default=DEFAULT_H5["R2 filtered"])
    h5.add_argument("--r3-h5", type=Path, default=DEFAULT_H5["R3 direct"])
    h5.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    h5.add_argument("--real-row-cap", type=int, default=600)
    h5.add_argument("--skip-real", action="store_true")
    h5.set_defaults(func=run_h5_analysis)

    entity = subparsers.add_parser("entity", help="Analyze v6.2 entity exposure")
    entity.add_argument(
        "--v62-h5", type=Path, default=REPO_ROOT / "model_pretrain/pretrain_datasets/v6.2.h5"
    )
    entity.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "model_pretrain/checkpoints/v6.2/model_eval00112.pt",
    )
    entity.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    entity.set_defaults(func=run_entity_analysis)

    performance = subparsers.add_parser(
        "performance", help="Analyze aligned per-seed performance deltas"
    )
    performance.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    performance.set_defaults(func=run_performance_analysis)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
