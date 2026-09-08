"""Context corruption probe — diagnose lazy vs feature-learning regime.

Based on OPENRFM (arXiv:2606.04320) Table 2 methodology.

Usage:
    python scripts/corruption_probe.py \
        --checkpoint model_pretrain/checkpoints/v5.3_20000/model_eval00080.pt \
        --data_dir model_pretrain/datasets/clf_rel \
        --gpu 0
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PRETRAIN = _REPO_ROOT / "model_pretrain"

sys.path.insert(0, str(_MODEL_PRETRAIN))

from src.models import ModelConfig, build_model, load_checkpoint, NanoTabPFNClassifier
from src.eval_utils import (
    _stable_random_state,
    _load_csv_dataset,
    downsample_split,
    fill_nans,
)

MAX_TRAIN_SAMPLES = 512
MAX_TEST_SAMPLES = 512
TEST_SIZE = 0.3


def corrupt_shuffle_labels(y: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.permutation(y)


def corrupt_shuffle_features(X: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()
    X_c = X.copy()
    for j in range(X_c.shape[1]):
        rng.shuffle(X_c[:, j])
    return X_c


def load_model(ckpt_path: str, device: str) -> NanoTabPFNClassifier:
    # v5.3 uses num_layers=6 (matches conf_eval/model/RDBPFN.yaml)
    model_cfg = ModelConfig(num_layers=6)
    model = build_model(model_cfg)
    load_checkpoint(model, Path(ckpt_path), device)
    model.to(device)
    model.eval()
    return NanoTabPFNClassifier(model, torch.device(device))


def run_probe(
    classifier: NanoTabPFNClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    results = {}

    # --- Condition 1: None (clean) ---
    classifier.fit(X_train.copy(), y_train.copy())
    prob_clean = classifier.predict_proba(X_test)
    results["clean"] = float(np.mean(prob_clean.argmax(axis=1) == y_test))

    # --- Condition 2: Shuffle labels ---
    classifier.fit(X_train.copy(), y_train.copy())
    classifier.y_train = corrupt_shuffle_labels(classifier.y_train)
    prob_shuf = classifier.predict_proba(X_test)
    results["shuffle_labels"] = float(np.mean(prob_shuf.argmax(axis=1) == y_test))

    # --- Condition 3: Shuffle features (per-column) ---
    classifier.fit(X_train.copy(), y_train.copy())
    classifier.X_train = corrupt_shuffle_features(classifier.X_train)
    prob_rf = classifier.predict_proba(X_test)
    results["shuffle_features"] = float(np.mean(prob_rf.argmax(axis=1) == y_test))

    return results


def main():
    parser = argparse.ArgumentParser(description="Context corruption probe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")

    classifier = load_model(args.checkpoint, device)

    data_dir = Path(args.data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"\nFound {len(csv_files)} CSV files")
    if args.max_tasks:
        csv_files = csv_files[: args.max_tasks]
        print(f"Limiting to first {args.max_tasks}")

    all_results = []
    for csv_path in csv_files:
        try:
            X, y = _load_csv_dataset(csv_path)
        except Exception as exc:
            print(f"  SKIP {csv_path.name}: {exc}")
            continue

        if len(np.unique(y)) < 2:
            print(f"  SKIP {csv_path.name}: single class")
            continue

        seed = _stable_random_state(csv_path.name)
        rng = np.random.default_rng(seed)
        n = len(X)
        n_test = max(1, int(n * TEST_SIZE))
        idx = rng.permutation(n)
        X_train, X_test = X[idx[:-n_test]], X[idx[-n_test:]]
        y_train, y_test = y[idx[:-n_test]], y[idx[-n_test:]]

        X_train, y_train = downsample_split(X_train, y_train, MAX_TRAIN_SAMPLES, seed)
        X_test, y_test = downsample_split(X_test, y_test, MAX_TEST_SAMPLES, seed)
        X_train, X_test = fill_nans(X_train, X_test)

        res = run_probe(classifier, X_train, y_train, X_test, y_test)
        res["task"] = csv_path.stem
        res["n_train"] = len(X_train)
        res["n_test"] = len(X_test)
        res["n_classes"] = len(np.unique(y))
        all_results.append(res)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gap_sl = res["clean"] - res["shuffle_labels"]
        gap_sf = res["clean"] - res["shuffle_features"]
        regime = (
            "FEATURE_LEARNING" if (gap_sl > 0.01 or gap_sf > 0.01) else "LAZY"
        )
        print(
            f"  {csv_path.stem:40s} "
            f"clean={res['clean']:.4f}  shuf_y={res['shuffle_labels']:.4f}  "
            f"shuf_x={res['shuffle_features']:.4f}  "
            f"Δy={gap_sl:+.4f}  Δx={gap_sf:+.4f}  [{regime}]"
        )

    if not all_results:
        print("No valid tasks found")
        return

    # Aggregate
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("AGGREGATE (mean accuracy):")
    for col in ["clean", "shuffle_labels", "shuffle_features"]:
        print(f"  {col:20s}: {df[col].mean():.4f}")
    gap_sl_avg = (df["clean"] - df["shuffle_labels"]).mean()
    gap_sf_avg = (df["clean"] - df["shuffle_features"]).mean()
    print(f"\n  Δ(shuffle_labels):  {gap_sl_avg:+.4f}")
    print(f"  Δ(shuffle_features): {gap_sf_avg:+.4f}")

    n_lazy = ((df["clean"] - df["shuffle_labels"]).abs() < 0.01).sum()
    n_fl = len(df) - n_lazy
    print(f"\n  Lazy tasks: {n_lazy}/{len(df)}")
    print(f"  Feature-learning tasks: {n_fl}/{len(df)}")

    if gap_sl_avg < 0.01 and gap_sf_avg < 0.01:
        diagnosis = (
            "LAZY REGIME CONFIRMED — model does not meaningfully use support labels "
            "or features. Label diversity (homophily control) is likely a bottleneck."
        )
    else:
        diagnosis = (
            "FEATURE-LEARNING REGIME — model uses support labels and/or features. "
            "Label diversity may not be the primary bottleneck."
        )
    print(f"\n  DIAGNOSIS: {diagnosis}")


if __name__ == "__main__":
    main()
