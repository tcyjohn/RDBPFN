from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import openml
except ImportError:
    openml = None

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .dbinfer_bench_simplified.rdb_dataset import DBBRDBTask
from .dbinfer_bench_simplified.dataset_meta import DBBColumnDType, DBBTaskEvalMetric


logger = logging.getLogger(__name__)

DEFAULT_EVAL_DIRS = [Path("datasets/clf_cat")]
SUBSAMPLE_SUFFIX = "_subsamples"


def _stable_random_state(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


TARGET_COLUMN_CANDIDATES = [
    "target",
    "class",
    "label",
    "labels",
    "y",
    "response",
    "outcome",
]


def _infer_target_column(df: pd.DataFrame) -> str:
    lookup = {col.lower(): col for col in df.columns}
    for candidate in TARGET_COLUMN_CANDIDATES:
        lowered = candidate.lower()
        if lowered in lookup:
            return lookup[lowered]
    return df.columns[-1]


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    data = df.copy()
    for column in data.columns:
        series = data[column]
        if not pd.api.types.is_numeric_dtype(series):
            codes, _ = pd.factorize(series.astype(str), sort=True)
            series = pd.Series(codes, index=series.index)
        else:
            series = pd.to_numeric(series, errors="coerce")
        if series.isna().any():
            fill_value = float(series.median()) if not series.dropna().empty else 0.0
            series = series.fillna(fill_value)
        data[column] = series.astype(np.float32)
    return data.to_numpy(dtype=np.float32)


def _prepare_target(series: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(series)
    if (codes < 0).any():
        raise ValueError("Target column contains invalid values after encoding.")
    return codes.astype(np.int64)


def _load_csv_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)
    target_column = _infer_target_column(df)
    df = df.dropna(subset=[target_column])
    y = _prepare_target(df[target_column])
    X = _prepare_features(df.drop(columns=[target_column]))
    return X, y


def _subsample_dir_for(csv_dir: Path) -> Path:
    return csv_dir.parent / f"{csv_dir.name}{SUBSAMPLE_SUFFIX}"


def _load_local_csv_datasets(data_dirs: list[Path]):
    grouped_splits: dict[str, list] = {}
    grouped_names: dict[str, list] = {}
    for directory in data_dirs:
        if not directory.exists():
            continue
        dir_key = directory.name
        dir_splits = []
        dir_names = []
        subsample_dir = _subsample_dir_for(directory)
        subsample_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(directory.glob("*.csv")):
            npz_path = subsample_dir / f"{csv_path.stem}_split.npz"
            if npz_path.exists():
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    X_train, X_test, y_train, y_test = (
                        data["X_train"],
                        data["X_test"],
                        data["y_train"],
                        data["y_test"],
                    )
                    dir_splits.append((X_train, X_test, y_train, y_test))
                    dir_names.append(csv_path.stem)
                    continue
                except Exception as exc:
                    print(f"Cached {npz_path} corrupted ({exc}), regenerating...")
                    npz_path.unlink(missing_ok=True)
            try:
                X, y = _load_csv_dataset(csv_path)
            except Exception as exc:
                print(f"Skipping {csv_path}: {exc}")
                continue
            if len(np.unique(y)) < 2:
                print(f"Skipping {csv_path}: target has only one class.")
                continue
            random_state = _stable_random_state(csv_path.name)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=random_state,
                stratify=y,
            )
            X_train, y_train = downsample_split(
                X_train, y_train, max_samples=1000, seed=random_state
            )
            X_test, y_test = downsample_split(
                X_test, y_test, max_samples=1000, seed=random_state
            )
            np.savez_compressed(
                npz_path,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
            dir_splits.append((X_train, X_test, y_train, y_test))
            dir_names.append(csv_path.stem)
        grouped_splits[dir_key] = dir_splits
        grouped_names[dir_key] = dir_names
    return grouped_splits, grouped_names


def prepare_eval_splits(
    data_dirs: list[Path] | None = None,
):
    dirs = data_dirs if data_dirs else DEFAULT_EVAL_DIRS
    return _load_local_csv_datasets(dirs)


def evaluate_classifier(classifier, splits_by_dir: dict[str, list]):
    metric_names = ["roc_auc", "acc", "balanced_acc"]
    per_dir_scores: dict[str, float] = {}
    total_splits = 0
    overall_sums = {k: 0.0 for k in metric_names}

    for dir_key, dir_splits in splits_by_dir.items():
        dir_sums = {k: 0.0 for k in metric_names}
        for X_train, X_test, y_train, y_test in dir_splits:
            classifier.fit(X_train, y_train)
            prob = classifier.predict_proba(X_test)
            pred = prob.argmax(axis=1)
            if prob.shape[1] == 2:
                roc = roc_auc_score(y_test, prob[:, 1])
            else:
                roc = roc_auc_score(y_test, prob, multi_class="ovr")
            roc_val = float(roc)
            acc_val = float(accuracy_score(y_test, pred))
            bal_val = float(balanced_accuracy_score(y_test, pred))
            dir_sums["roc_auc"] += roc_val
            dir_sums["acc"] += acc_val
            dir_sums["balanced_acc"] += bal_val
            overall_sums["roc_auc"] += roc_val
            overall_sums["acc"] += acc_val
            overall_sums["balanced_acc"] += bal_val

        n = len(dir_splits)
        total_splits += n
        if n > 0:
            for k in metric_names:
                per_dir_scores[f"{dir_key}/{k}"] = dir_sums[k] / n

    scores: dict[str, float] = {}
    for k in metric_names:
        scores[k] = overall_sums[k] / total_splits if total_splits > 0 else 0.0
    scores.update(per_dir_scores)
    return scores


def load_task_split(
    task: DBBRDBTask,
    split: str,
    *,
    use_primary_key_as_entity_id: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    if split == "train":
        source = task.train_set
    elif split in {"val", "validation"}:
        source = task.validation_set
    elif split == "test":
        source = task.test_set
    else:
        raise ValueError(f"Unknown split {split}")
    if source is None:
        raise ValueError(f"Task {task.metadata.name} has no {split} data")
    target_col = task.metadata.target_column
    fk_cols = [
        col.name
        for col in task.metadata.columns
        if col.dtype == DBBColumnDType.foreign_key and col.name != target_col
    ]
    feature_cols = [
        col.name
        for col in task.metadata.columns
        if (
            col.dtype == DBBColumnDType.float_t
            or col.dtype == DBBColumnDType.category_t
        )
        and col.name != target_col
    ]
    if not feature_cols:
        raise ValueError(f"Task {task.metadata.name} has no feature columns")
    feature_cols.sort()
    X = np.column_stack([source[col] for col in feature_cols]).astype(np.float32)
    y = np.asarray(source[target_col])

    # Extract raw FK values for attention bias (not in X)
    fk_values = None
    if fk_cols:
        fk_parts = []
        for fk_name in fk_cols:
            fk_raw = source[fk_name].astype(np.float64)
            fk_raw = np.nan_to_num(fk_raw, nan=-1).astype(np.int64)
            fk_parts.append(fk_raw)
        fk_values = np.column_stack(fk_parts)

    # Extract entity_ids for same-entity attention bias (not in X)
    entity_ids = None
    if "entity_id" in source:
        entity_raw = source["entity_id"].astype(np.float64)
        entity_ids = np.nan_to_num(entity_raw, nan=-1).astype(np.int64)
    elif use_primary_key_as_entity_id:
        primary_key_cols = [
            col.name
            for col in task.metadata.columns
            if col.dtype == DBBColumnDType.primary_key and col.name in source
        ]
        if primary_key_cols:
            entity_raw = source[primary_key_cols[0]].astype(np.float64)
            entity_ids = np.nan_to_num(entity_raw, nan=-1).astype(np.int64)

    # Extract parent_entity_ids for entity-level FK matching (not in X)
    # Shape: (total_rows, K) where K is number of FK relations, -1 for null.
    # These may not exist in eval datasets (only in H5 training data).
    parent_entity_ids = None
    if "parent_entity_ids" in source:
        peids_raw = source["parent_entity_ids"].astype(np.float64)
        parent_entity_ids = np.nan_to_num(peids_raw, nan=-1).astype(np.int64)

    return X, y, fk_values, entity_ids, parent_entity_ids


def downsample_split(
    X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(X) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=max_samples, replace=False)
    return X[indices], y[indices]


def predict_proba_in_chunks(
    classifier,
    X: np.ndarray,
    chunk_size: int | None,
    fk_values_test: np.ndarray | None = None,
    entity_ids_test: np.ndarray | None = None,
    parent_entity_ids_test: np.ndarray | None = None,
) -> np.ndarray:
    if chunk_size is None or len(X) <= chunk_size:
        return classifier.predict_proba(
            X,
            fk_values_test=fk_values_test,
            entity_ids_test=entity_ids_test,
            parent_entity_ids_test=parent_entity_ids_test,
        )
    probs = []
    for start in range(0, len(X), chunk_size):
        end = start + chunk_size
        fk_chunk = fk_values_test[start:end] if fk_values_test is not None else None
        eid_chunk = entity_ids_test[start:end] if entity_ids_test is not None else None
        peid_chunk = parent_entity_ids_test[start:end] if parent_entity_ids_test is not None else None
        probs.append(classifier.predict_proba(
            X[start:end],
            fk_values_test=fk_chunk,
            entity_ids_test=eid_chunk,
            parent_entity_ids_test=peid_chunk,
        ))
    return np.concatenate(probs, axis=0)


def fill_nans(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col_idx in range(X_train.shape[1]):
        train_col = X_train[:, col_idx]
        test_col = X_test[:, col_idx]

        train_has_nan = np.isnan(train_col).any()
        test_has_nan = np.isnan(test_col).any()

        if train_has_nan or test_has_nan:
            # Compute fill value from training data only
            median_val = np.nanmedian(train_col)
            fill_value = median_val if not np.isnan(median_val) else 0.0

            if train_has_nan:
                X_train[np.isnan(X_train[:, col_idx]), col_idx] = fill_value
            if test_has_nan:
                X_test[np.isnan(X_test[:, col_idx]), col_idx] = fill_value

    return X_train, X_test


# =============================================================================
# Metric Computation
# =============================================================================


def compute_metric(
    metric: DBBTaskEvalMetric,
    y_true: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
) -> float:
    """Compute a classification metric.

    Args:
        metric: The metric type to compute
        y_true: Ground truth labels
        prob: Predicted probabilities (shape: [n_samples, n_classes])
        pred: Predicted class labels

    Returns:
        The computed metric value, or NaN if computation fails.
    """
    try:
        if metric == DBBTaskEvalMetric.accuracy:
            return float(accuracy_score(y_true, pred))
        if metric == DBBTaskEvalMetric.f1:
            return float(f1_score(y_true, pred, average="weighted"))
        if metric == DBBTaskEvalMetric.recall:
            return float(recall_score(y_true, pred, average="weighted"))
        if metric == DBBTaskEvalMetric.ap:
            if prob.shape[1] == 2:
                return float(average_precision_score(y_true, prob[:, 1]))
            return float(average_precision_score(y_true, prob, average="macro"))
        if metric == DBBTaskEvalMetric.auroc:
            if prob.shape[1] == 2:
                return float(roc_auc_score(y_true, prob[:, 1]))
            return float(roc_auc_score(y_true, prob, multi_class="ovr"))
        if metric == DBBTaskEvalMetric.balanced_acc:
            return float(balanced_accuracy_score(y_true, pred))
        if metric == DBBTaskEvalMetric.logloss:
            return float(log_loss(y_true, prob))
    except ValueError as exc:
        logger.warning("Failed to compute metric %s: %s", metric, exc)
        return float("nan")
    logger.warning("Metric %s not implemented; returning NaN", metric)
    return float("nan")


# =============================================================================
# Path Utilities
# =============================================================================


def derive_output_name(checkpoint_path_str: str | None) -> str:
    """Derive a CSV output name from the checkpoint path.

    Formats:
        checkpoints/xxx.pt            -> xxx
        checkpoints/xxx/              -> xxx
        checkpoints/xxx/model.pt      -> xxx
        checkpoints/xxx/model_evalyyy.pt -> xxx_yyy
    """
    if not checkpoint_path_str:
        return "results"
    if "checkpoints" not in checkpoint_path_str:
        return checkpoint_path_str
    path = Path(checkpoint_path_str)

    # If it's a directory (or ends with /), use directory name
    if checkpoint_path_str.endswith("/") or path.suffix == "":
        return path.name

    # It's a .pt file
    filename = path.stem  # filename without .pt

    if filename == "model":
        # checkpoints/xxx/model.pt -> xxx
        return path.parent.name
    elif filename.startswith("model_eval"):
        # checkpoints/xxx/model_evalyyy.pt -> xxx_yyy
        suffix = filename[len("model_eval") :]  # extract yyy
        return f"{path.parent.name}_{suffix}"
    else:
        # checkpoints/xxx.pt -> xxx
        return filename


def simplify_model_label(label: str) -> str:
    """Extract relative path under checkpoints/ from full path."""
    # Find "checkpoints/" in the path and keep everything after it
    marker = "checkpoints/"
    idx = label.find(marker)
    if idx != -1:
        return label[idx + len(marker) :]
    # Fallback: just return the basename
    return Path(label).name


# =============================================================================
# Results I/O
# =============================================================================


def save_results_to_csv(
    all_model_results: dict[str, list[dict]],
    output_path: Path,
    metric_key: str = "metric_value",
) -> None:
    """Save results to CSV with tasks as columns and models as rows.

    Args:
        all_model_results: Dict mapping model_label -> list of aggregated task results
        output_path: Path to save the CSV file
        metric_key: Which metric to use as the cell value (default: "metric_value")
    """
    df = build_results_dataframe(all_model_results, metric_key=metric_key)
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)


PER_SEED_RESULT_COLUMNS = [
    "model",
    "dataset",
    "task",
    "seed",
    "metric",
    "metric_value",
    "accuracy",
    "balanced_acc",
]


def derive_per_seed_output_path(output_path: Path) -> Path:
    """Return the detail CSV path alongside an aggregate result CSV."""
    return output_path.with_name(f"{output_path.stem}_per_seed{output_path.suffix}")


def build_per_seed_results_dataframe(
    all_model_results: dict[str, list[dict]],
) -> pd.DataFrame:
    """Build one row per model, task, and evaluation seed."""
    rows = []
    for model_label, results in all_model_results.items():
        for result in results:
            rows.append(
                {
                    "model": simplify_model_label(model_label),
                    "dataset": result["dataset"],
                    "task": result["task"],
                    "seed": result["seed"],
                    "metric": result["metric"],
                    "metric_value": result["metric_value"],
                    "accuracy": result["accuracy"],
                    "balanced_acc": result["balanced_acc"],
                }
            )
    dataframe = pd.DataFrame(rows, columns=PER_SEED_RESULT_COLUMNS)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(
            ["model", "dataset", "task", "seed"], kind="stable"
        ).reset_index(drop=True)
    return dataframe


def save_per_seed_results_to_csv(
    all_model_results: dict[str, list[dict]], output_path: Path
) -> None:
    dataframe = build_per_seed_results_dataframe(all_model_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    logger.info("Per-seed results saved to %s", output_path)


def append_per_seed_results_to_csv(
    all_model_results: dict[str, list[dict]], output_path: Path
) -> None:
    new_dataframe = build_per_seed_results_dataframe(all_model_results)
    if output_path.exists():
        existing_dataframe = pd.read_csv(output_path)
        combined = pd.concat([existing_dataframe, new_dataframe], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["model", "dataset", "task", "seed"], keep="last"
        )
        combined = combined.sort_values(
            ["model", "dataset", "task", "seed"], kind="stable"
        ).reset_index(drop=True)
        combined.to_csv(output_path, index=False)
        logger.info("Appended per-seed results to %s", output_path)
    else:
        save_per_seed_results_to_csv(all_model_results, output_path)


def build_results_dataframe(
    all_model_results: dict[str, list[dict]],
    metric_key: str = "metric_value",
) -> pd.DataFrame:
    """Build a dataframe with tasks as columns and models as rows."""
    # Collect all unique task names (dataset/task) across all models
    all_tasks: set[str] = set()
    for results in all_model_results.values():
        for r in results:
            task_col = f"{r['dataset']}/{r['task']}"
            all_tasks.add(task_col)

    # Sort tasks for consistent column order
    task_columns = sorted(all_tasks)

    # Build rows: each row is a model, columns are tasks
    rows = []
    for model_label, results in all_model_results.items():
        # Simplify model label to relative path under checkpoints/
        simplified_label = simplify_model_label(model_label)
        row = {"model": simplified_label}
        task_to_metric = {f"{r['dataset']}/{r['task']}": r[metric_key] for r in results}
        for task_col in task_columns:
            value = task_to_metric.get(task_col, float("nan"))
            # Round to 4 decimal places
            row[task_col] = round(value, 4) if not np.isnan(value) else value
        # Add average across all tasks
        values = [row[t] for t in task_columns if not np.isnan(row[t])]
        row["average"] = round(float(np.mean(values)), 4) if values else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns: model first, then tasks, then average
    column_order = ["model"] + task_columns + ["average"]
    df = df[column_order]
    return df


def append_results_to_csv(
    all_model_results: dict[str, list[dict]],
    output_path: Path,
    metric_key: str = "metric_value",
) -> None:
    """Append new model rows into an existing results CSV, unioning columns."""
    new_df = build_results_dataframe(all_model_results, metric_key=metric_key)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        # Union columns, keeping order: model, sorted tasks, average (if present)
        all_cols = list(existing_df.columns)
        for col in new_df.columns:
            if col not in all_cols:
                all_cols.insert(-1 if "average" in all_cols else len(all_cols), col)
        for col in all_cols:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
            if col not in new_df.columns:
                new_df[col] = np.nan
        combined = pd.concat([existing_df[all_cols], new_df[all_cols]], ignore_index=True)
        # Drop duplicate models keeping the last occurrence
        combined = combined.drop_duplicates(subset=["model"], keep="last")
        combined.to_csv(output_path, index=False)
        logger.info("Appended results to %s", output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(output_path, index=False)
        logger.info("Results saved to %s", output_path)
