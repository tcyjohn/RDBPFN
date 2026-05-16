"""LR baseline diagnostic on single-table PriorDataset tasks.

Compares against the complex-task results to determine whether the unlearnable
label problem is specific to complex multi-table tasks or universal.

Usage:
    LD_LIBRARY_PATH=... pixi run python scripts/diagnose_single_table.py --n-tasks 50
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def gen_and_diagnose_single(n_tasks: int = 50, seed: int = 42):
    """Generate n_tasks single-table datasets and run LR baseline on each."""

    from tabicl.prior.dataset import PriorDataset

    ds = PriorDataset(
        batch_size=n_tasks,
        batch_size_per_gp=1,
        min_features=18,
        max_features=18,
        max_classes=2,
        max_seq_len=600,
        min_train_size=0.5,
        max_train_size=0.9,
        prior_type="mix_scm",
        device="cpu",
    )

    X_batch, y_batch, d_batch, seq_lens_batch, train_sizes_batch = ds.get_batch()

    results = []
    for i in range(n_tasks):
        seq_len = int(seq_lens_batch[i].item())
        train_size = int(train_sizes_batch[i].item())
        n_features = int(d_batch[i].item())

        X = X_batch[i, :seq_len, :n_features].cpu().numpy()
        y = y_batch[i, :seq_len].cpu().numpy()

        # Avoid constant columns
        col_std = X.std(axis=0)
        valid_cols = col_std > 1e-8
        X = X[:, valid_cols]
        if X.shape[1] < 2:
            results.append({"error": f"too few valid columns ({X.shape[1]})"})
            continue

        # NaN guard
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        train_X, train_y = X[:train_size], y[:train_size]
        test_X, test_y = X[train_size:], y[train_size:]

        n_classes = len(np.unique(train_y))
        pos_ratio = float(np.mean(train_y == 1)) if n_classes <= 2 else float("nan")

        # LR CV AUC
        lr_auc = float("nan")
        try:
            if n_classes >= 2 and len(train_y) >= 10:
                min_class = min(np.bincount(train_y.astype(int)))
                n_splits = min(5, min_class) if min_class >= 2 else 2
                pipe = make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=500, n_jobs=1)
                )
                cv = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=42
                )
                scores = cross_val_score(
                    pipe, train_X, train_y, cv=cv, scoring="roc_auc"
                )
                lr_auc = float(np.mean(scores))
        except Exception:
            pass

        # Best single-feature AUC
        best_sf_auc = float("nan")
        for j in range(min(X.shape[1], 20)):
            col = train_X[:, j:j + 1]
            try:
                s = cross_val_score(
                    make_pipeline(
                        StandardScaler(), LogisticRegression(max_iter=500)
                    ),
                    col,
                    train_y,
                    cv=min(3, min_class),
                    scoring="roc_auc",
                )
                auc = float(np.mean(s))
                if np.isnan(best_sf_auc) or auc > best_sf_auc:
                    best_sf_auc = auc
            except Exception:
                pass

        results.append({
            "i": i,
            "n_features": int(n_features),
            "seq_len": seq_len,
            "train_size": train_size,
            "n_classes": n_classes,
            "pos_ratio": pos_ratio,
            "lr_auc": lr_auc,
            "best_single_feat_auc": best_sf_auc,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=50)
    args = parser.parse_args()

    print(f"Generating {args.n_tasks} single-table datasets and running LR baseline...")
    results = gen_and_diagnose_single(args.n_tasks)

    lr_aucs = [
        r["lr_auc"]
        for r in results
        if not np.isnan(r.get("lr_auc", float("nan")))
    ]
    best_feats = [
        r["best_single_feat_auc"]
        for r in results
        if not np.isnan(r.get("best_single_feat_auc", float("nan")))
    ]
    errors = [r for r in results if "error" in r]

    print(f"\n{'='*60}")
    print("Single-Table LR Baseline Results")
    print(f"{'='*60}")
    print(f"  Tasks: {len(results)}, Errors: {len(errors)}, Valid AUCs: {len(lr_aucs)}")

    print(f"\n  Classes per task:")
    n_class_counts = {}
    for r in results:
        nc = r.get("n_classes", "?")
        n_class_counts[nc] = n_class_counts.get(nc, 0) + 1
    for nc, cnt in sorted(n_class_counts.items()):
        print(f"    {nc} classes: {cnt} tasks")

    print(f"\n  LR (full features) CV AUC:")
    if lr_aucs:
        arr = np.array(lr_aucs)
        print(f"    mean={np.mean(arr):.4f}  median={np.median(arr):.4f}")
        print(f"    min={np.min(arr):.4f}  max={np.max(arr):.4f}")
        print(f"    std={np.std(arr):.4f}")
        print(f"    frac > 0.55: {np.mean(arr > 0.55):.3f}")
        print(f"    frac > 0.60: {np.mean(arr > 0.60):.3f}")
        print(f"    frac > 0.70: {np.mean(arr > 0.70):.3f}")

    print(f"\n  Best single-feature CV AUC:")
    if best_feats:
        arr = np.array(best_feats)
        print(f"    mean={np.mean(arr):.4f}  median={np.median(arr):.4f}")

    print(f"\n{'='*60}")
    print("COMPARISON WITH COMPLEX TASKS")
    print(f"{'='*60}")
    print(f"  Complex (128 RDB):  LR AUC mean=0.5069  median=0.5037  >0.55=10%")
    print(f"  Complex (1024 RDB): LR AUC mean=0.4984  median=0.5008  >0.55=5%")
    if lr_aucs:
        arr = np.array(lr_aucs)
        print(f"  Single-table:       LR AUC mean={np.mean(arr):.4f}  median={np.median(arr):.4f}  >0.55={np.mean(arr>0.55):.0%}")


if __name__ == "__main__":
    main()
