#!/usr/bin/env python3
"""Apply the R3 task-quality criteria to an existing raw 4DBInfer corpus.

This is a post-hoc, no-retry screen: it evaluates each saved task's
``train.parquet`` with the same ``diagnose_dataframe`` entry point used by the
R3 generator.  Passing tasks are materialized into an isolated hard-link copy
of the raw corpus; the source corpus is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RDB_SOURCE_ROOT = REPO_ROOT / "data_generation" / "RDB"
sys.path.insert(0, str(RDB_SOURCE_ROOT))

from src.table_def.task_quality import diagnose_dataframe  # noqa: E402


def _audit_one(train_path_text: str) -> dict[str, Any]:
    train_path = Path(train_path_text)
    task_dir = train_path.parent
    rdb_dir = task_dir.parent
    try:
        frame = pd.read_parquet(train_path)
        result = diagnose_dataframe(frame)
        return {
            "rdb": rdb_dir.name,
            "task": task_dir.name,
            "train_path": str(train_path),
            "passed": bool(result["passed"]),
            "stage": result.get("stage"),
            "n_samples": result.get("n_samples"),
            "n_pos": result.get("n_pos"),
            "n_neg": result.get("n_neg"),
            "pos_ratio": result.get("pos_ratio"),
            "n_features": result.get("n_features"),
            "et_auc": result.get("et_auc"),
            "et_ci_lower": result.get("et_ci_lower"),
            "et_ci_upper": result.get("et_ci_upper"),
            "failures": result.get("all_failures", []),
            "error": "",
        }
    except Exception as exc:  # fail closed and preserve the exception in audit
        return {
            "rdb": rdb_dir.name,
            "task": task_dir.name,
            "train_path": str(train_path),
            "passed": False,
            "stage": "ERROR",
            "n_samples": None,
            "n_pos": None,
            "n_neg": None,
            "pos_ratio": None,
            "n_features": None,
            "et_auc": None,
            "et_ci_lower": None,
            "et_ci_upper": None,
            "failures": ["audit_error"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _atomic_yaml_write(path: Path, payload: Any, *, safe: bool = True) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            dumper = yaml.safe_dump if safe else yaml.dump
            dumper(payload, handle, sort_keys=False)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rdb",
        "task",
        "train_path",
        "passed",
        "stage",
        "n_samples",
        "n_pos",
        "n_neg",
        "pos_ratio",
        "n_features",
        "et_auc",
        "et_ci_lower",
        "et_ci_upper",
        "failures",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["failures"] = json.dumps(
                encoded["failures"], ensure_ascii=False, separators=(",", ":")
            )
            writer.writerow(encoded)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["passed"] = row["passed"].lower() == "true"
            row["failures"] = json.loads(row["failures"])
            rows.append(row)
    return rows


def _hardlink_copy(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    shutil.copytree(source_root, output_root, copy_function=os.link, symlinks=True)


def _materialize_filtered_view(
    source_root: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
) -> tuple[int, int]:
    _hardlink_copy(source_root, output_root)
    passed_by_rdb: dict[str, set[str]] = {}
    all_by_rdb: dict[str, set[str]] = {}
    for row in rows:
        all_by_rdb.setdefault(row["rdb"], set()).add(row["task"])
        if row["passed"]:
            passed_by_rdb.setdefault(row["rdb"], set()).add(row["task"])

    # The old R2 corpus also contains generated RDBs with zero task directories;
    # they never appear in the task manifest and must not survive in the
    # effective filtered corpus.
    for output_rdb in output_root.glob("dag_rdb_*"):
        if output_rdb.is_dir() and output_rdb.name not in passed_by_rdb:
            shutil.rmtree(output_rdb)

    retained_rdbs = 0
    retained_tasks = 0
    for rdb_name, all_tasks in all_by_rdb.items():
        output_rdb = output_root / rdb_name
        passed_tasks = passed_by_rdb.get(rdb_name, set())
        if not passed_tasks:
            continue

        for task_name in all_tasks - passed_tasks:
            task_dir = output_rdb / task_name
            if task_dir.exists():
                shutil.rmtree(task_dir)

        metadata_path = output_rdb / "metadata.yaml"
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle)
        metadata["tasks"] = [
            task
            for task in metadata.get("tasks", [])
            if task.get("name") in passed_tasks
        ]
        _atomic_yaml_write(metadata_path, metadata)

        # generation_schemas.yaml is provenance for what the old generator
        # attempted, not the effective 4DBInfer task registry.  Historical R2
        # files also embed Python objects that current code cannot round-trip.
        # Keep it byte-identical; metadata.yaml plus task directories define
        # the filtered corpus consumed by preprocessing.

        retained_rdbs += 1
        retained_tasks += len(passed_tasks)

    return retained_rdbs, retained_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="Skip diagnostics and materialize from an existing manifest.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write the manifest without creating a filtered raw view.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not args.audit_only and args.output_root is None:
        raise ValueError("--output-root is required unless --audit-only is set")

    if args.reuse_manifest:
        rows = _read_manifest(args.manifest)
    else:
        train_paths = sorted(source_root.glob("dag_rdb_*/*/train.parquet"))
        if not train_paths:
            raise RuntimeError(f"No raw task train.parquet files under {source_root}")
        rows = []
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_audit_one, str(path)): path for path in train_paths
            }
            for future in as_completed(futures):
                rows.append(future.result())
                completed += 1
                if completed % 50 == 0 or completed == len(train_paths):
                    passed = sum(bool(row["passed"]) for row in rows)
                    print(
                        f"Audited {completed}/{len(train_paths)} tasks "
                        f"({passed} passing)",
                        flush=True,
                    )
        rows.sort(key=lambda row: (row["rdb"], row["task"]))
        _write_manifest(args.manifest, rows)

    passed = sum(bool(row["passed"]) for row in rows)
    errors = sum(bool(row.get("error")) for row in rows)
    print(
        f"Quality audit: {passed}/{len(rows)} tasks pass; "
        f"{len(rows) - passed} fail; {errors} audit errors."
    )

    if args.audit_only:
        return 0

    output_root = args.output_root.resolve()
    retained_rdbs, retained_tasks = _materialize_filtered_view(
        source_root, output_root, rows
    )
    if retained_tasks != passed:
        raise RuntimeError(
            f"Materialization mismatch: manifest={passed}, view={retained_tasks}"
        )
    print(
        f"Filtered raw view: {retained_rdbs} RDBs, {retained_tasks} tasks "
        f"at {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
