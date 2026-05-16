"""Extract held-out tasks from a pretraining H5 file as CSV eval datasets.

Each extracted task becomes one CSV file in the output directory, with the
original train/test split preserved (rows before single_eval_pos = train,
rows after = test). The CSV format matches what eval_utils.py expects.

Usage:
    pixi run python scripts/extract_eval_from_h5.py \
        --h5-path model_pretrain/pretrain_datasets/hsbm_v2_d2.h5 \
        --output-dir model_pretrain/datasets/clf \
        --num-tasks 15
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_task_to_csv(
    X: np.ndarray,
    y: np.ndarray,
    split_pos: int,
    num_features: int,
    num_datapoints: int,
    feature_is_categorical: np.ndarray | None,
    task_idx: int,
    output_dir: Path,
):
    # Use only the actual features and rows (not padding)
    X_valid = X[:num_datapoints, :num_features].copy()
    y_valid = y[:num_datapoints].copy()

    # Build column names
    col_names = []
    if feature_is_categorical is not None:
        cat_mask = feature_is_categorical[:num_features]
    else:
        cat_mask = np.zeros(num_features, dtype=bool)
    for j in range(num_features):
        prefix = "cat" if cat_mask[j] else "feat"
        col_names.append(f"{prefix}{j}")

    df = pd.DataFrame(X_valid, columns=col_names)
    df["target"] = y_valid.astype(int)

    csv_name = f"hsbm_task_{task_idx:04d}.csv"
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)
    logger.info(
        "Saved %s: rows=%d, features=%d, train=%d, test=%d, classes=%d",
        csv_name,
        num_datapoints,
        num_features,
        split_pos,
        num_datapoints - split_pos,
        len(np.unique(y_valid)),
    )


def main():
    parser = argparse.ArgumentParser(description="Extract eval tasks from H5 file")
    parser.add_argument("--h5-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-tasks", type=int, default=15)
    parser.add_argument("--task-indices", type=int, nargs="*", default=None)
    parser.add_argument("--min-test-rows", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5_path, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
        split_pos = f["single_eval_pos"][:]
        num_features = f["num_features"][:]
        num_datapoints = f["num_datapoints"][:]
        cat_mask = f.get("feature_is_categorical")
        cat_mask = cat_mask[:] if cat_mask is not None else None

    total_tasks = X.shape[0]
    logger.info("H5 file: %d total tasks", total_tasks)

    # Remove old CSV files
    for old_csv in args.output_dir.glob("hsbm_task_*.csv"):
        old_csv.unlink()
        logger.info("Removed old %s", old_csv.name)

    if args.task_indices:
        indices = args.task_indices
    else:
        # Select tasks spaced evenly across the dataset
        step = max(1, total_tasks // args.num_tasks)
        indices = list(range(0, total_tasks, step))[: args.num_tasks]

    extracted = 0
    for i in indices:
        test_rows = int(num_datapoints[i]) - int(split_pos[i])
        if test_rows < args.min_test_rows:
            logger.warning(
                "Skipping task %d: only %d test rows (min %d)",
                i, test_rows, args.min_test_rows,
            )
            continue
        extract_task_to_csv(
            X[i], y[i],
            split_pos=int(split_pos[i]),
            num_features=int(num_features[i]),
            num_datapoints=int(num_datapoints[i]),
            feature_is_categorical=cat_mask[i] if cat_mask is not None else None,
            task_idx=i,
            output_dir=args.output_dir,
        )
        extracted += 1

    logger.info("Extracted %d tasks to %s", extracted, args.output_dir)


if __name__ == "__main__":
    main()
