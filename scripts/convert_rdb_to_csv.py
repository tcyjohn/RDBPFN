"""Convert RDB datasets (npz or parquet) to CSV files for training-time eval csv_dirs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLITS = ["train", "test", "validation"]


def _detect_format(task_dir: Path) -> str | None:
    for ext in (".npz", ".parquet"):
        if (task_dir / f"train{ext}").exists():
            return ext
    return None


def _load_split(task_dir: Path, split: str, ext: str) -> pd.DataFrame:
    path = task_dir / f"{split}{ext}"
    if not path.exists():
        return None
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        return pd.DataFrame({k: data[k] for k in data.files})
    else:
        return pd.read_parquet(path)


def _load_tasks_flat(meta: dict) -> list[dict]:
    """Return flat list of task entries. Handles nested groups (4DBInfer raw format)."""
    if isinstance(meta, list):
        return meta
    # raw 4DBInfer: tasks = [{"task_group": ..., "tasks": [...]}, ...]
    tasks = meta.get("tasks", [])
    flat = []
    for entry in tasks:
        if "tasks" in entry:
            flat.extend(entry["tasks"])
        else:
            flat.append(entry)
    return flat


def convert_dataset(dataset_path: Path, output_dir: Path):
    meta = yaml.safe_load((dataset_path / "metadata.yaml").read_text())
    dataset_name = meta["dataset_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks_flat(meta)
    for task_meta in tasks:
        if task_meta["task_type"] != "classification":
            continue

        task_name = task_meta["name"]
        target_col = task_meta["target_column"]
        task_dir = dataset_path / task_name

        ext = _detect_format(task_dir)
        if ext is None:
            print(f"  SKIP {task_name}: no train.{npz,parquet} found")
            continue

        frames = []
        for split in SPLITS:
            df = _load_split(task_dir, split, ext)
            if df is not None:
                frames.append(df)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna(subset=[target_col])

        cols = [c for c in combined.columns if c != target_col] + [target_col]
        combined = combined[cols]

        out_path = output_dir / f"{dataset_name}__{task_name}.csv"
        combined.to_csv(out_path, index=False)
        print(f"  {out_path.name}  rows={len(combined)}  cols={len(cols)}")


def main():
    parser = argparse.ArgumentParser(description="Convert RDB datasets to CSV for eval")
    parser.add_argument("--src-dir", type=Path, required=True,
                        help="Source dir: individual dataset dir (with metadata.yaml), "
                             "or parent dir of such dirs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for CSV files")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset names to convert (default: all)")
    args = parser.parse_args()

    src_dir = args.src_dir
    # Detect: if src_dir itself has metadata.yaml, treat as single dataset
    if (src_dir / "metadata.yaml").exists():
        dataset_dirs = [src_dir]
    elif args.datasets:
        dataset_dirs = [src_dir / d for d in args.datasets]
    else:
        dataset_dirs = sorted(
            d for d in src_dir.iterdir()
            if d.is_dir() and (d / "metadata.yaml").exists()
        )

    for ds_path in dataset_dirs:
        if not ds_path.exists():
            print(f"SKIP {ds_path.name}: not found")
            continue
        print(f"Converting {ds_path.name}...")
        convert_dataset(ds_path, args.output_dir)

    print(f"\nDone. CSVs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
