"""
Corruption probe: measures whether model relies on structural bias (entity/FK)
vs. feature-content mapping.

Three conditions per task:
  - clean:       normal eval
  - shuffle_y:   randomly permute y_train (entity/FK structure intact, labels wrong)
  - shuffle_x:   randomly permute X_train rows (entity/FK structure intact, features wrong)

Interpretation for PFN with entity/FK bias:
  - Small Δ(shuffle_y):   model does NOT use entity/FK bias for label propagation
                           (labels don't matter → model predicts from features alone)
  - Large Δ(shuffle_y):   model DOES use entity/FK bias (wrong labels propagated)
  - Small Δ(shuffle_x):   model relies on entity/FK bias, not feature content
  - Large Δ(shuffle_x):   model relies on feature→label mapping

Usage:
  python -m scripts.corruption_probe \
    --checkpoint checkpoints/v6.2/model_eval00112.pt \
    --dataset-dir rdb_datasets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("corruption_probe")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
from src.dbinfer_bench_simplified.dataset_meta import DBBTaskType
from src.models import ModelConfig, build_model, build_classifier, load_checkpoint
from src.eval_utils import (
    load_task_split,
    downsample_split,
    _stable_random_state,
    fill_nans,
)
from src.utils import get_default_device

DEFAULT_MAX_TRAIN = 512
DEFAULT_MAX_TEST = 50000


def run_condition(classifier_factory, X_train, y_train, X_test, y_test,
                  fk_train, eid_train, peid_train, fk_test, eid_test, peid_test,
                  condition: str, seed: int):
    """Run one corruption condition. Returns accuracy."""
    rng = np.random.default_rng(seed)

    if condition == "shuffle_y":
        y_train = rng.permutation(y_train)
    elif condition == "shuffle_x":
        perm = rng.permutation(len(X_train))
        X_train = X_train[perm]
        if fk_train is not None:
            fk_train = fk_train[perm]
        if eid_train is not None:
            eid_train = eid_train[perm]
        if peid_train is not None:
            peid_train = peid_train[perm]

    clf = classifier_factory()
    clf.fit(X_train, y_train,
            fk_values=fk_train,
            entity_ids=eid_train,
            parent_entity_ids=peid_train)

    from src.eval_utils import predict_proba_in_chunks
    prob = predict_proba_in_chunks(
        clf, X_test, chunk_size=2000,
        fk_values_test=fk_test,
        entity_ids_test=eid_test,
        parent_entity_ids_test=peid_test,
    )
    pred = prob.argmax(axis=1)
    return float((pred == y_test).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="rdb_datasets")
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lambda-se", type=float, default=None,
                        help="Override entity bias lambda (softplus(raw) = value)")
    args = parser.parse_args()

    device_str = args.device or get_default_device()
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    # Build model
    model_cfg = ModelConfig(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=6,
        use_fk_bias=True,
        use_entity_bias=True,
    )
    model = build_model(model_cfg)
    ckpt_path = Path(args.checkpoint)
    load_checkpoint(model, ckpt_path, str(device))
    if args.lambda_se is not None:
        import torch.nn as nn
        import numpy as np
        target = args.lambda_se
        raw_val = float(np.log(np.exp(target) - 1)) if target > 0 else -10.0
        overridden = 0
        for name, module in model.named_modules():
            # Only EntityAttentionBias has raw_lambda without raw_lambdas
            if hasattr(module, 'raw_lambda') and not hasattr(module, 'raw_lambdas'):
                module.raw_lambda.data.fill_(raw_val)
                overridden += 1
        logger.info("Overrode %d EntityAttentionBias modules: λ_se = %.4f (raw=%.4f)", overridden, target, raw_val)
    model.to(device)
    model.eval()

    def classifier_factory():
        return build_classifier(model, device, model_cfg)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error("Dataset dir %s not found", dataset_dir)
        sys.exit(1)

    # Collect all classification tasks
    all_tasks = []
    for dpath in sorted(dataset_dir.iterdir()):
        if not dpath.is_dir():
            continue
        try:
            ds = DBBRDBDataset(dpath)
        except Exception:
            continue
        for task in ds.tasks:
            if task.metadata.task_type != DBBTaskType.classification:
                continue
            try:
                X_train, y_train, fk_train, eid_train, peid_train = load_task_split(task, "train")
                X_test, y_test, fk_test, eid_test, peid_test = load_task_split(task, "test")
            except Exception:
                continue
            all_tasks.append({
                "dataset": ds.dataset_name,
                "task": task.metadata.name,
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test,
                "fk_train": fk_train, "eid_train": eid_train, "peid_train": peid_train,
                "fk_test": fk_test, "eid_test": eid_test, "peid_test": peid_test,
            })

    logger.info("Found %d classification tasks across %d datasets",
                len(all_tasks), len(set(t["dataset"] for t in all_tasks)))

    CONDITIONS = ["clean", "shuffle_y", "shuffle_x"]
    results_by_cond = defaultdict(list)

    for tinfo in all_tasks:
        for seed in args.seeds:
            seed_key = f"{tinfo['dataset']}:{tinfo['task']}:{seed}"
            seed_offset = _stable_random_state(seed_key)

            # Downsample train
            X_train, y_train = tinfo["X_train"], tinfo["y_train"]
            if args.max_train and len(X_train) > args.max_train:
                rng = np.random.default_rng(seed_offset)
                idx = rng.choice(len(X_train), size=args.max_train, replace=False)
                idx.sort()
            else:
                idx = np.arange(len(X_train))
            X_tr = X_train[idx]
            y_tr = y_train[idx]
            fk_tr = tinfo["fk_train"][idx] if tinfo["fk_train"] is not None else None
            eid_tr = tinfo["eid_train"][idx] if tinfo["eid_train"] is not None else None
            peid_tr = tinfo["peid_train"][idx] if tinfo["peid_train"] is not None else None

            # Downsample test
            test_key = f"{tinfo['dataset']}:{tinfo['task']}:{seed}:test"
            test_offset = _stable_random_state(test_key)
            X_te, y_te = downsample_split(
                tinfo["X_test"], tinfo["y_test"], args.max_test, test_offset
            )
            # Also downsample test auxiliary arrays
            if args.max_test and len(tinfo["X_test"]) > args.max_test:
                test_rng = np.random.default_rng(test_offset)
                test_idx = test_rng.choice(len(tinfo["X_test"]), size=args.max_test, replace=False)
                fk_te = tinfo["fk_test"][test_idx] if tinfo["fk_test"] is not None else None
                eid_te = tinfo["eid_test"][test_idx] if tinfo["eid_test"] is not None else None
                peid_te = tinfo["peid_test"][test_idx] if tinfo["peid_test"] is not None else None
            else:
                fk_te = tinfo["fk_test"]
                eid_te = tinfo["eid_test"]
                peid_te = tinfo["peid_test"]
            X_tr, X_te = fill_nans(X_tr, X_te)

            if len(np.unique(y_tr)) < 2:
                continue

            for cond in CONDITIONS:
                cond_seed = _stable_random_state(f"{seed_key}:{cond}")
                acc = run_condition(
                    classifier_factory, X_tr, y_tr, X_te, y_te,
                    fk_tr, eid_tr, peid_tr,
                    fk_te, eid_te, peid_te,
                    cond, cond_seed,
                )
                results_by_cond[cond].append({
                    "dataset": tinfo["dataset"],
                    "task": tinfo["task"],
                    "seed": seed,
                    "acc": acc,
                })

    # Print summary
    print("\n" + "=" * 70)
    print("  Corruption Probe Results")
    print("=" * 70)
    clean_mean = np.mean([r["acc"] for r in results_by_cond["clean"]])
    print(f"  {'Condition':<20s}  {'Mean Acc':>10s}  {'Δ from clean':>14s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*14}")
    for cond in CONDITIONS:
        accs = [r["acc"] for r in results_by_cond[cond]]
        mean_acc = np.mean(accs)
        delta = (clean_mean - mean_acc) * 100  # in percentage points
        print(f"  {cond:<20s}  {mean_acc:>10.4f}  {delta:>+13.2f} pp")
    print("=" * 70)

    # Per-task breakdown
    print("\nPer-task Δ (percentage points drop from clean):")
    tasks = sorted(set((r["dataset"], r["task"]) for r in results_by_cond["clean"]))
    clean_by_task = {}
    for ds, tk in tasks:
        clean_accs = [r["acc"] for r in results_by_cond["clean"]
                      if r["dataset"] == ds and r["task"] == tk]
        clean_by_task[(ds, tk)] = np.mean(clean_accs)

    print(f"  {'Dataset':<30s} {'Task':<30s} {'Clean':>8s} {'Δy':>8s} {'Δx':>8s}")
    for ds, tk in tasks:
        c = clean_by_task[(ds, tk)]
        dy_accs = [r["acc"] for r in results_by_cond["shuffle_y"]
                    if r["dataset"] == ds and r["task"] == tk]
        dx_accs = [r["acc"] for r in results_by_cond["shuffle_x"]
                    if r["dataset"] == ds and r["task"] == tk]
        dy = (c - np.mean(dy_accs)) * 100
        dx = (c - np.mean(dx_accs)) * 100
        print(f"  {ds:<30s} {tk:<30s} {c:>8.4f} {dy:>+7.2f} {dx:>+7.2f}")


if __name__ == "__main__":
    main()
