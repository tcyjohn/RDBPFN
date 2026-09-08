#!/usr/bin/env python3
"""Build auditable assets for the SA-RDB-PFN Technical Supplement.

This script adds the generation-gated RDB-PFN-Small control to the frozen
main-study artifacts. It does not train or evaluate a model.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import yaml

from scripts.data_analysis.r2_r3_v62_analysis import (
    _load_performance,
    compare_performance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "aaai2027_submission/technical_supplement/generated"
FIGURES = REPO_ROOT / "aaai2027_submission/AuthorKit27/Figures"
FROZEN = (
    REPO_ROOT
    / "aaai2027_submission/anonymous_reproducibility/sa_rdbpfn_repro/results/per_seed"
)

PER_SEED = {
    "Published-scale RDB-PFN": FROZEN / "rdbpfn_scale_reference.csv",
    "RDB-PFN-Small (original filter)": FROZEN / "rdbpfn_small.csv",
    "RDB-PFN-Small (generation gate)": REPO_ROOT
    / "model_pretrain/results/"
    "ablation_r2_quality_native_1024_opt112k_full512_aligned_per_seed.csv",
    "SA-RDB-PFN": FROZEN / "sa_rdbpfn_high_on.csv",
}

RAW_ROOTS = {
    "RDB-PFN-Small (original filter)": Path(
        "/tmp/RDBPFN-ablation-r2/data_generation/RDB_datasets/"
        "ablation_r2_original_style_1024_raw"
    ),
    "RDB-PFN-Small (generation gate)": Path(
        "/tmp/RDBPFN-ablation-r2/data_generation/RDB_datasets/"
        "ablation_r2_quality_native_1024_raw"
    ),
    "SA-RDB-PFN": Path(
        "/tmp/RDBPFN-ablation-r3/data_generation/RDB_datasets/"
        "ablation_r3_data_prior_1024_raw"
    ),
}

H5_PATHS = {
    "RDB-PFN-Small (original filter)": Path(
        "/tmp/RDBPFN-ablation-r2/model_pretrain/pretrain_datasets/"
        "ablation_r2_original_style_1024.h5"
    ),
    "RDB-PFN-Small (generation gate)": Path(
        "/tmp/RDBPFN-ablation-r2/model_pretrain/pretrain_datasets/"
        "ablation_r2_quality_native_1024_direct.h5"
    ),
    "SA-RDB-PFN": Path(
        "/tmp/RDBPFN-ablation-r3/model_pretrain/pretrain_datasets/"
        "ablation_r3_data_prior_1024_unsampled.h5"
    ),
}

POLICIES = {
    "RDB-PFN-Small (original filter)": (
        "Original post-materialization repetitive-column filter"
    ),
    "RDB-PFN-Small (generation gate)": (
        "Task-level generation gate, retry/best attempt, direct merge"
    ),
    "SA-RDB-PFN": (
        "Task-level generation gate, retry/best attempt, RDB-level SNR, direct merge"
    ),
}

DISPLAY_NAMES = {
    ("Amazon-dfs-2", "churn"): "Amazon churn",
    ("avs-dfs-2", "repeater"): "AVS repeater",
    ("diginetica-dfs-2", "ctr"): "Diginetica CTR",
    ("outbrain-small-dfs-2", "ctr"): "Outbrain CTR",
    ("rel-amazon-dfs-2", "item-churn"): "Rel-Amazon item churn",
    ("rel-amazon-dfs-2", "user-churn"): "Rel-Amazon user churn",
    ("rel-avito-dfs-2", "user-clicks"): "Rel-Avito user clicks",
    ("rel-avito-dfs-2", "user-visits"): "Rel-Avito user visits",
    ("rel-event-dfs-2", "user-ignore"): "Rel-Event user ignore",
    ("rel-event-dfs-2", "user-repeat"): "Rel-Event user repeat",
    ("rel-f1-dfs-2", "driver-dnf"): "Rel-F1 driver DNF",
    ("rel-f1-dfs-2", "driver-top3"): "Rel-F1 driver top-3",
    ("rel-hm-dfs-2", "user-churn"): "Rel-H&M user churn",
    ("rel-stack-dfs-2", "user-badge"): "Rel-Stack user badge",
    ("rel-stack-dfs-2", "user-engagement"): "Rel-Stack user engagement",
    ("rel-trial-dfs-2", "study-outcome"): "Rel-Trial study outcome",
    ("retailrocket-dfs-2", "cvr"): "Retailrocket CVR",
    ("stackexchange-dfs-2", "churn"): "StackExchange churn",
    ("stackexchange-dfs-2", "upvote"): "StackExchange upvote",
}


def _short_task(dataset: str, task: str) -> str:
    dataset = dataset.replace("-dfs-2", "").replace("rel-", "r-")
    return f"{dataset}/{task}"


def build_performance_assets() -> None:
    frames = {
        label: _load_performance(path, label) for label, path in PER_SEED.items()
    }
    comparisons = [
        (
            "RDB-PFN-Small (generation gate)",
            "RDB-PFN-Small (original filter)",
            "Generation gate - original filter",
        ),
        (
            "SA-RDB-PFN",
            "RDB-PFN-Small (generation gate)",
            "SA-RDB-PFN - generation gate",
        ),
    ]
    task_frames = []
    suite_rows = []
    for left, right, label in comparisons:
        tasks, suite = compare_performance(frames[left], frames[right], label)
        task_frames.append(tasks)
        suite_rows.append(suite)
    task_deltas = pd.concat(task_frames, ignore_index=True)
    task_deltas.to_csv(OUTPUT / "rq1_generation_gate_task_deltas.csv", index=False)
    pd.DataFrame(suite_rows).to_csv(
        OUTPUT / "rq1_generation_gate_suite_summary.csv", index=False
    )

    task_means = []
    for label, frame in frames.items():
        grouped = (
            frame.groupby(["dataset", "task"], as_index=False)["metric_value"]
            .mean()
            .rename(columns={"metric_value": "mean_auroc"})
        )
        grouped["variant"] = label
        task_means.append(grouped)
    means = pd.concat(task_means, ignore_index=True)
    means["display_name"] = [
        DISPLAY_NAMES[(dataset, task)]
        for dataset, task in zip(means["dataset"], means["task"])
    ]
    means.to_csv(OUTPUT / "rq1_generation_gate_task_means.csv", index=False)

    seed_means = pd.concat(
        [
            frame.groupby("seed", as_index=False)["metric_value"]
            .mean()
            .assign(variant=label)
            for label, frame in frames.items()
        ],
        ignore_index=True,
    )
    seed_means.to_csv(OUTPUT / "rq1_generation_gate_seed_means.csv", index=False)
    plot_rq1_forest(task_deltas)


def _raw_accounting(root: Path) -> dict[str, int]:
    requested_dirs = sorted(path for path in root.glob("dag_rdb_*") if path.is_dir())
    completed = [
        path
        for path in requested_dirs
        if (path / "metadata.yaml").is_file() and any(path.glob("table_*.parquet"))
    ]
    raw_tasks = 0
    taskful_rdbs = 0
    table_count = 0
    raw_rows = 0
    raw_cells = 0
    for rdb_dir in completed:
        task_dirs = [
            path
            for path in rdb_dir.iterdir()
            if path.is_dir() and "task" in path.name
        ]
        raw_tasks += len(task_dirs)
        taskful_rdbs += int(bool(task_dirs))
        for table_path in rdb_dir.glob("table_*.parquet"):
            metadata = pq.ParquetFile(table_path).metadata
            table_count += 1
            raw_rows += metadata.num_rows
            raw_cells += metadata.num_rows * metadata.num_columns
    return {
        "requested_rdbs": len(requested_dirs),
        "completed_rdbs": len(completed),
        "taskful_rdbs": taskful_rdbs,
        "pre_h5_tasks": raw_tasks,
        "raw_tables": table_count,
        "raw_rows": raw_rows,
        "raw_cells": raw_cells,
    }


def build_corpus_accounting() -> None:
    rows = []
    for label, raw_root in RAW_ROOTS.items():
        accounting = _raw_accounting(raw_root)
        with h5py.File(H5_PATHS[label], "r") as handle:
            stored_tasks = int(handle["X"].shape[0])
            rows_per_task = int(handle["X"].shape[1])
            max_features = int(handle["X"].shape[2])
        rows.append(
            {
                "variant": label,
                **accounting,
                "stored_tasks": stored_tasks,
                "rows_per_task": rows_per_task,
                "h5_max_features": max_features,
                "policy": POLICIES[label],
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT / "rq1_corpus_accounting.csv", index=False)


def build_evaluation_manifest() -> None:
    from model_pretrain.src.dbinfer_bench_simplified.dataset_meta import DBBTaskType
    from model_pretrain.src.dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
    from model_pretrain.src.eval_utils import load_task_split

    config_path = REPO_ROOT / "model_pretrain/conf_eval/dataset/full-512.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = []
    for relative in config["paths"]:
        dataset_path = REPO_ROOT / "model_pretrain" / relative
        dataset = DBBRDBDataset(dataset_path)
        for task in dataset.tasks:
            if task.metadata.task_type != DBBTaskType.classification:
                continue
            X_train, y_train, *_ = load_task_split(task, "train")
            X_test, y_test, *_ = load_task_split(task, "test")
            key = (dataset.dataset_name, task.metadata.name)
            rows.append(
                {
                    "dataset": dataset.dataset_name,
                    "task": task.metadata.name,
                    "display_name": DISPLAY_NAMES[key],
                    "metric": str(task.metadata.evaluation_metric),
                    "train_pool_rows": len(y_train),
                    "test_pool_rows": len(y_test),
                    "dfs_features": X_train.shape[1],
                    "support_cap": int(config["max_train_samples"]),
                    "test_cap": 50_000,
                    "evaluation_seeds": "0,1,2,3,4,5,6,7,8,9",
                    "release_path": relative,
                }
            )
    manifest = pd.DataFrame(rows)
    if len(manifest) != 19:
        raise RuntimeError(f"Expected 19 classification tasks, found {len(manifest)}")
    manifest.to_csv(OUTPUT / "evaluation_task_manifest.csv", index=False)


def plot_rq1_forest(task_deltas: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_style("whitegrid", {"grid.color": "#E6E6E6", "grid.linewidth": 0.7})
    comparisons = [
        "Generation gate - original filter",
        "SA-RDB-PFN - generation gate",
    ]
    titles = [
        "Generation gate minus\noriginal filter",
        "SA-RDB-PFN minus\ngeneration-gated control",
    ]
    colors = ["#0072B2", "#D55E00"]
    first = task_deltas[task_deltas["comparison"] == comparisons[0]].copy()
    first["label"] = [
        _short_task(dataset, task)
        for dataset, task in zip(first["dataset"], first["task"])
    ]
    task_order = first.sort_values("mean_difference")["label"].tolist()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 4.8), sharey=True, constrained_layout=True
    )
    for axis, comparison, title, color in zip(
        axes, comparisons, titles, colors
    ):
        subset = task_deltas[task_deltas["comparison"] == comparison].copy()
        subset["label"] = [
            _short_task(dataset, task)
            for dataset, task in zip(subset["dataset"], subset["task"])
        ]
        subset = subset.set_index("label").loc[task_order].reset_index()
        y = np.arange(len(subset))
        x = subset["mean_difference"].to_numpy()
        xerr = np.vstack([x - subset["ci_low"], subset["ci_high"] - x])
        axis.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            alpha=0.9,
        )
        for idx, row in subset.iterrows():
            axis.scatter(
                row["mean_difference"],
                idx,
                s=23,
                marker="o",
                facecolor=color if row["holm_significant"] else "white",
                edgecolor=color,
                linewidth=0.9,
                zorder=3,
            )
        axis.axvline(0, color="black", linewidth=0.7, linestyle="--")
        axis.set_title(title, pad=4)
        axis.set_yticks(y, task_order)
        axis.grid(axis="y", visible=False)
        sns.despine(ax=axis)
    axes[0].text(
        0.02,
        0.99,
        "filled = Holm-significant",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )
    fig.supxlabel("Paired AUROC difference (95% aligned-support interval)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "performance_delta_generation_gate.pdf")
    fig.savefig(FIGURES / "performance_delta_generation_gate.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_performance_assets()
    build_corpus_accounting()
    build_evaluation_manifest()
    print(f"Wrote Technical Supplement assets to {OUTPUT}")


if __name__ == "__main__":
    main()
