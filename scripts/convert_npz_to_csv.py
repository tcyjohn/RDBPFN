"""Convert relational benchmark .npz files (from HuggingFace) to CSV for training eval.

Each task has train.npz, validation.npz, test.npz in a subdirectory.
Each .npz contains per-column arrays + a target column.
We merge all splits into one CSV with the target column renamed to "target".
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RDB_DIR = ROOT / "model_pretrain/rdb_datasets"
OUT_DIR = ROOT / "model_pretrain/datasets/clf_rel"


def convert_task(task_dir: Path, target_col: str, out_dir: Path) -> Path | None:
    """Merge train/val/test .npz into a single CSV."""
    dfs = []
    for split in ["train", "validation", "test"]:
        npz_path = task_dir / f"{split}.npz"
        if not npz_path.exists():
            print(f"  WARN: {npz_path} not found, skipping split")
            continue
        data = dict(np.load(npz_path, allow_pickle=True))
        # Build DataFrame from per-column arrays
        col_data = {}
        for col_name, arr in data.items():
            col_data[col_name] = arr
        if col_data:
            dfs.append(pd.DataFrame(col_data))

    if not dfs:
        print(f"  FAIL: no data loaded")
        return None

    df = pd.concat(dfs, ignore_index=True)

    if target_col not in df.columns:
        print(f"  FAIL: target column '{target_col}' not found in {list(df.columns)}")
        return None

    # Rename target to "target" (what eval_utils.py auto-detects)
    df["target"] = df[target_col]
    df = df.drop(columns=[target_col])

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = task_dir.parent.name  # e.g., rel-f1-dfs-2
    task_name = task_dir.name            # e.g., driver-dnf
    csv_name = f"{dataset_name}__{task_name}.csv"
    csv_path = out_dir / csv_name
    df.to_csv(csv_path, index=False)
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  OK   {csv_name} ({df.shape[0]} rows, {df.shape[1]} cols, {size_mb:.1f}MB)")
    return csv_path


def main():
    import yaml

    tasks_to_convert = [
        ("rel-f1-dfs-2", "driver-dnf", "did_not_finish"),
        ("rel-f1-dfs-2", "driver-top3", "qualifying"),
        ("rel-event-dfs-2", "user-ignore", "user-ignore"),
        ("rel-event-dfs-2", "user-repeat", "user-repeat"),
    ]

    converted = 0
    for dataset, task, target in tasks_to_convert:
        task_dir = RDB_DIR / dataset / task
        if not task_dir.exists():
            print(f"SKIP {dataset}/{task} (not found)")
            continue
        print(f"\n=== {dataset}/{task} (target={target}) ===")
        result = convert_task(task_dir, target, OUT_DIR)
        if result:
            converted += 1

    print(f"\n=== Done: {converted} tasks converted -> {OUT_DIR} ===")


if __name__ == "__main__":
    main()
