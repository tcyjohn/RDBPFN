#!/usr/bin/env python3
"""Generate manuscript-ready figures from frozen R2--R3--v6.2 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from scripts.data_analysis.r2_r3_v62_analysis import (
    DEFAULT_H5,
    REPO_ROOT,
    _prepare_feature_matrix,
    stable_seed,
)


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Preserve the exact 7-inch manuscript canvas.  Tight bounding
            # boxes change the physical PDF width and therefore the final font
            # size after LaTeX rescales the figure to \textwidth.
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
        }
    )
    sns.set_style("whitegrid", {"grid.color": "#E6E6E6", "grid.linewidth": 0.7})


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_mechanism_survival(
    results: Path, figures: Path, signal_results: Path
) -> None:
    rdb = pd.read_csv(results / "raw_rdb_metrics.csv")
    signal_rdb = pd.read_csv(signal_results / "raw_rdb_metrics.csv")
    summary = pd.read_csv(results / "raw_paired_summary.csv").set_index("metric")
    signal_summary = pd.read_csv(
        signal_results / "raw_paired_summary.csv"
    ).set_index("metric")
    metrics = [
        ("temporal_table_fraction", "Temporal coverage", "Table fraction"),
        (
            "calendar_activity_synchrony",
            "Cross-table time synchrony",
            "Activity correlation",
        ),
        ("effective_parent_coverage", "Effective parent coverage", "Normalized coverage"),
        ("multi_parent_fk_nmi", "Multi-parent FK NMI", "Normalized MI"),
        ("normalized_effective_rank", "Raw feature effective rank", "Normalized rank"),
    ]
    # Group the two temporal-availability diagnostics above the three
    # connectivity/content diagnostics.  Independent row grids let all five
    # 9-pt panels fill the manuscript width without a dummy sixth axis.
    fig = plt.figure(figsize=(7.0, 3.3), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, hspace=0.32)
    temporal_grid = outer[0].subgridspec(1, 2)
    structure_grid = outer[1].subgridspec(1, 3)
    axes = [
        fig.add_subplot(temporal_grid[0, 0]),
        fig.add_subplot(temporal_grid[0, 1]),
        fig.add_subplot(structure_grid[0, 0]),
        fig.add_subplot(structure_grid[0, 1]),
        fig.add_subplot(structure_grid[0, 2]),
    ]
    palette = {"R2": BLUE, "R3": ORANGE}
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        data = rdb[rdb["metric"] == metric].dropna(subset=["value"])
        order = ["R2", "R3"]
        metric_palette = palette
        display_labels = ["RDB-PFN-\nSmall", "SA-RDB-\nPFN"]
        if metric == "normalized_effective_rank":
            no_signal = signal_rdb[
                (signal_rdb["metric"] == metric) & (signal_rdb["source"] == "R2")
            ].dropna(subset=["value"]).copy()
            no_signal["source"] = "No signal groups"
            data = pd.concat([data, no_signal], ignore_index=True)
            order = ["R2", "No signal groups", "R3"]
            metric_palette = {
                "R2": BLUE,
                "No signal groups": PURPLE,
                "R3": ORANGE,
            }
            display_labels = ["RDB-PFN\nSmall", "No SG", "SA-RDB-\nPFN"]
        if metric == "temporal_table_fraction":
            medians = data.groupby("source")["value"].median().reindex(order)
            axis.bar(
                range(len(order)),
                medians,
                color=[metric_palette[key] for key in order],
                edgecolor="black",
                linewidth=0.7,
            )
            axis.set_ylim(0.0, 1.0)
        elif metric != "calendar_activity_synchrony":
            # All three metrics are normalized by definition.  Clipping removes
            # floating-point spillover before KDE estimation; cut=0 then keeps
            # the violin support within the observed feasible range.
            data = data.copy()
            data["value"] = data["value"].clip(0.0, 1.0)
            if not data["value"].between(0.0, 1.0).all():
                raise ValueError(f"{metric} contains values outside [0, 1]")
            sns.violinplot(
                data=data,
                x="source",
                y="value",
                hue="source",
                order=order,
                palette=metric_palette,
                legend=False,
                inner=None,
                cut=0,
                linewidth=0.7,
                ax=axis,
            )
            sns.boxplot(
                data=data,
                x="source",
                y="value",
                order=order,
                width=0.22,
                showfliers=False,
                color="white",
                linecolor="black",
                linewidth=0.7,
                ax=axis,
            )
            axis.set_ylim(0.0, 1.0)
        else:
            sns.violinplot(
                data=data,
                x="source",
                y="value",
                hue="source",
                order=order,
                palette=metric_palette,
                legend=False,
                inner=None,
                cut=0,
                linewidth=0.7,
                ax=axis,
            )
            sns.boxplot(
                data=data,
                x="source",
                y="value",
                order=order,
                width=0.22,
                showfliers=False,
                color="white",
                linecolor="black",
                linewidth=0.7,
                ax=axis,
            )
            axis.axhline(0.0, color=GRAY, linewidth=0.7, linestyle=":")
            axis.set_ylim(-0.25, 1.0)
        axis.set_xticks(range(len(order)), display_labels)
        row = (
            signal_summary.loc[metric]
            if metric == "normalized_effective_rank"
            else summary.loc[metric]
        )
        annotation = (
            f"paired $\Delta$={row['median_paired_difference']:+.3f}\n"
            f"95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
        )
        axis.set_title(f"{title}\n{annotation}", pad=4, fontsize=9)
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.tick_params(
            axis="x",
            labelsize=9,
            pad=3,
        )
        axis.grid(axis="x", visible=False)
        sns.despine(ax=axis)
    save_figure(fig, figures / "mechanism_survival")


def _h5_matrix(path: Path, task_idx: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        n_rows = int(handle["num_datapoints"][task_idx])
        n_features = int(handle["num_features"][task_idx])
        return _prepare_feature_matrix(handle["X"][task_idx, :n_rows, :n_features])


def _real_matrix(task_id: str, row_cap: int = 600) -> np.ndarray:
    from model_pretrain.src.dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
    from model_pretrain.src.eval_utils import load_task_split

    dataset_name, task_name = task_id.split("/", 1)
    dataset = DBBRDBDataset(REPO_ROOT / "model_pretrain/rdb_datasets" / dataset_name)
    task = dataset.get_task(task_name)
    X_train, *_ = load_task_split(task, "train")
    X_test, *_ = load_task_split(task, "test")
    X = np.concatenate([X_train, X_test], axis=0)
    if len(X) > row_cap:
        rng = np.random.default_rng(stable_seed(f"{dataset_name}:{task_name}:rank"))
        X = X[np.sort(rng.choice(len(X), size=row_cap, replace=False))]
    return _prepare_feature_matrix(X)


def _ordered_correlation(X: np.ndarray) -> np.ndarray:
    varying = np.std(X, axis=0) > 1e-12
    corr = np.corrcoef(X[:, varying], rowvar=False)
    if corr.shape[0] <= 2:
        return corr
    distance = np.clip(1.0 - np.abs(corr), 0.0, 2.0)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    return corr[np.ix_(order, order)]


def plot_dfs_correlation(
    results: Path,
    figures: Path,
    signal_results: Path,
    r2_h5: Path = DEFAULT_H5["R2 filtered"],
    r25_h5: Path = Path(
        "/tmp/RDBPFN-ablation-r25/model_pretrain/pretrain_datasets/"
        "ablation_r25_no_signal_groups_1024.h5"
    ),
    r3_h5: Path = DEFAULT_H5["R3 direct"],
) -> None:
    metrics = pd.read_csv(results / "h5_task_metrics.csv")
    signal_metrics = pd.read_csv(signal_results / "h5_task_metrics.csv")
    no_signal_metrics = signal_metrics[signal_metrics["corpus"] == "R2 filtered"].copy()
    no_signal_metrics["corpus"] = "No signal groups"
    metrics = pd.concat([metrics, no_signal_metrics], ignore_index=True)
    exemplars = json.loads((results / "h5_median_exemplars.json").read_text())
    signal_exemplars = json.loads(
        (signal_results / "h5_median_exemplars.json").read_text()
    )
    matrices = {
        "RDB-PFN-Small": _h5_matrix(
            r2_h5, int(exemplars["R2 filtered"]["task_id"])
        ),
        "No signal groups": _h5_matrix(
            r25_h5, int(signal_exemplars["R2 filtered"]["task_id"])
        ),
        "SA-RDB-PFN": _h5_matrix(
            r3_h5, int(exemplars["R3 direct"]["task_id"])
        ),
        "Real DFS": _real_matrix(exemplars["Real DFS"]["task_id"]),
    }
    # Emit the four correlation exemplars as a 2x2 one-column figure while
    # preserving the physical size of each heatmap.  The effective-rank
    # summary remains a separate one-column figure.
    fig = plt.figure(figsize=(3.28, 2.65), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.10])
    heat_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    for axis, (label, X) in zip(heat_axes, matrices.items()):
        corr = _ordered_correlation(X)
        # Use a QuadMesh rather than imshow so each matrix cell remains a vector
        # object in the PDF instead of becoming an embedded low-resolution image.
        image = axis.pcolormesh(
            corr,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            shading="nearest",
            edgecolors="none",
            antialiased=False,
            rasterized=False,
        )
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_title(label, pad=2, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel(f"{corr.shape[0]} features", fontsize=9)
    colorbar_axis = fig.add_subplot(grid[2, :])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("Pearson correlation", fontsize=9)
    colorbar.ax.tick_params(labelsize=9)
    save_figure(fig, figures / "dfs_correlation_heatmaps")

    fig, distribution_axis = plt.subplots(
        figsize=(3.28, 1.65), constrained_layout=True
    )
    order = ["R2 filtered", "No signal groups", "R3 direct", "Real DFS"]
    sns.boxplot(
        data=metrics,
        y="corpus",
        x="normalized_effective_rank",
        hue="corpus",
        order=order,
        palette={
            "R2 filtered": BLUE,
            "No signal groups": PURPLE,
            "R3 direct": ORANGE,
            "Real DFS": GREEN,
        },
        legend=False,
        showfliers=False,
        linewidth=0.7,
        ax=distribution_axis,
    )
    sns.stripplot(
        data=metrics[metrics["corpus"] == "Real DFS"],
        y="corpus",
        x="normalized_effective_rank",
        order=order,
        color="black",
        marker="o",
        size=2.5,
        alpha=0.65,
        ax=distribution_axis,
    )
    distribution_axis.set_xlim(0, 1)
    distribution_axis.set_xlabel("Normalized effective rank")
    distribution_axis.set_ylabel("")
    distribution_axis.set_yticks(
        [0, 1, 2, 3],
        ["RDB-PFN-Small", "No signal groups", "SA-RDB-PFN", "Real DFS"],
    )
    distribution_axis.grid(axis="y", visible=False)
    sns.despine(ax=distribution_axis)
    save_figure(fig, figures / "dfs_effective_rank")


def _short_task(dataset: str, task: str) -> str:
    dataset = dataset.replace("-dfs-2", "").replace("rel-", "r-")
    return f"{dataset}/{task}"


def plot_entity_exposure(results: Path, figures: Path) -> None:
    exposure = pd.read_csv(results / "entity_exposure_evaluation_tasks.csv")
    exposure["label"] = [
        _short_task(dataset, task)
        for dataset, task in zip(exposure["dataset"], exposure["task"])
    ]
    valid = exposure.dropna(subset=["query_support_overlap"]).sort_values(
        "query_support_overlap", ascending=False
    )
    top = valid.head(5).copy()
    remainder = valid.iloc[5:]
    compact = pd.concat(
        [
            top[["label", "query_support_overlap"]],
            pd.DataFrame(
                {
                    "label": [f"Other valid ({len(remainder)})", "No usable ID (4)"],
                    "query_support_overlap": [
                        remainder["query_support_overlap"].median(),
                        0.0,
                    ],
                }
            ),
        ],
        ignore_index=True,
    )
    compact = compact.iloc[::-1]
    fig, axis = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)
    y = np.arange(len(compact))
    axis.hlines(y, 0, compact["query_support_overlap"] * 100, color=LIGHT_GRAY, linewidth=1.3)
    for idx, row in compact.reset_index(drop=True).iterrows():
        no_id = row["label"].startswith("No usable")
        axis.scatter(
            row["query_support_overlap"] * 100,
            idx,
            color=GRAY if no_id else ORANGE,
            edgecolor="black" if not no_id else None,
            linewidth=0.7,
            s=24,
            marker="x" if no_id else "o",
            zorder=3,
        )
    axis.set_yticks(y, compact["label"])
    axis.set_xlabel("Same-entity query overlap (%)")
    axis.set_xlim(0, 38)
    axis.grid(axis="y", visible=False)
    sns.despine(ax=axis)
    save_figure(fig, figures / "entity_exposure")

    full = exposure.sort_values("query_support_overlap", na_position="first")
    fig, axis = plt.subplots(figsize=(7.0, 3.7), constrained_layout=True)
    y = np.arange(len(full))
    values = full["query_support_overlap"].fillna(0) * 100
    colors = np.where(full["query_support_overlap"].isna(), GRAY, ORANGE)
    markers = ["x" if missing else "o" for missing in full["query_support_overlap"].isna()]
    axis.hlines(y, 0, values, color=LIGHT_GRAY, linewidth=1.1)
    for idx, (value, color, marker) in enumerate(zip(values, colors, markers)):
        axis.scatter(value, idx, color=color, marker=marker, s=25, linewidth=0.8, zorder=3)
    axis.set_yticks(
        y,
        [_short_task(d, t) for d, t in zip(full["dataset"], full["task"])],
    )
    axis.set_xlabel("Query--support same-entity overlap (%)")
    axis.set_xlim(0, 36)
    axis.grid(axis="y", visible=False)
    sns.despine(ax=axis)
    save_figure(fig, figures / "entity_exposure_all_tasks")


def plot_performance_forest(results: Path, figures: Path) -> None:
    data = pd.read_csv(results / "performance_task_deltas.csv")
    display_titles = {
        "R3 - R2": "SA-RDB-PFN minus\nRDB-PFN-Small",
        "R3 - No signal groups": "Dense history:\ngrouped sources on minus off",
        "Low density - Joint removal": "Low history:\ngrouped sources on minus off",
        "R3 - Low density": "Grouped sources on:\ndense minus low history",
        "No signal groups - Joint removal": "Grouped sources off:\ndense minus low history",
        "R3 - Joint removal": (
            "SA-RDB-PFN minus\nlow-cardinality/grouped-off"
        ),
        "v6.2 - R3": "Entity-aware minus\nSA-RDB-PFN",
    }
    main_comparisons = ["R3 - R2", "R3 - Joint removal"]
    sensitivity_comparisons = [
        "R3 - No signal groups",
        "Low density - Joint removal",
        "R3 - Low density",
        "No signal groups - Joint removal",
    ]
    entity_comparisons = ["v6.2 - R3"]
    first = data[data["comparison"] == main_comparisons[0]].copy()
    first["label"] = [
        _short_task(dataset, task)
        for dataset, task in zip(first["dataset"], first["task"])
    ]
    task_order = first.sort_values("mean_difference")["label"].tolist()

    figure_specs = [
        (
            main_comparisons,
            [BLUE, PURPLE],
            (7.0, 4.65),
            "performance_delta_core",
            1,
        ),
        (
            sensitivity_comparisons,
            [PURPLE, ORANGE, BLUE, GREEN],
            (7.0, 8.2),
            "performance_delta_sensitivity",
            2,
        ),
        (
            entity_comparisons,
            [ORANGE],
            (4.2, 4.75),
            "performance_delta_entity",
            1,
        ),
    ]
    for comparisons, colors, size, output_name, nrows in figure_specs:
        ncols = int(np.ceil(len(comparisons) / nrows))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=size,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for axis, comparison, color in zip(axes, comparisons, colors):
            subset = data[data["comparison"] == comparison].copy()
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
            axis.set_title(display_titles[comparison], fontsize=9, pad=4)
            axis.set_xlabel("")
            axis.set_yticks(y, task_order)
            axis.grid(axis="y", visible=False)
            sns.despine(ax=axis)
        for axis in axes[len(comparisons):]:
            axis.set_visible(False)
        axes[0].text(
            0.02,
            0.99,
            "filled = Holm-significant",
            transform=axes[0].transAxes,
            va="top",
            fontsize=9,
        )
        fig.supxlabel("Paired AUROC difference (95% CI)", fontsize=9)
        save_figure(fig, figures / output_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "model_pretrain/results/r2_r3_v62_data_analysis",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=REPO_ROOT / "aaai2027_submission/AuthorKit27/Figures",
    )
    parser.add_argument(
        "--signal-results",
        type=Path,
        default=REPO_ROOT
        / "model_pretrain/results/r2_r3_v62_data_analysis/signal_group_ablation",
    )
    parser.add_argument("--r2-h5", type=Path, default=DEFAULT_H5["R2 filtered"])
    parser.add_argument(
        "--r25-h5",
        type=Path,
        default=Path(
            "/tmp/RDBPFN-ablation-r25/model_pretrain/pretrain_datasets/"
            "ablation_r25_no_signal_groups_1024.h5"
        ),
    )
    parser.add_argument("--r3-h5", type=Path, default=DEFAULT_H5["R3 direct"])
    args = parser.parse_args()
    set_paper_style()
    plot_mechanism_survival(args.results, args.figures, args.signal_results)
    plot_dfs_correlation(
        args.results,
        args.figures,
        args.signal_results,
        args.r2_h5,
        args.r25_h5,
        args.r3_h5,
    )
    plot_entity_exposure(args.results, args.figures)
    plot_performance_forest(args.results, args.figures)
    print(f"Wrote figures to {args.figures}")


if __name__ == "__main__":
    main()
