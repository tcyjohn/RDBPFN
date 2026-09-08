"""
Attention pattern diagnostic for specific eval tasks.
Hooks into the entity bias computation rather than replacing attention.
"""
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("attn_diag")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
from src.dbinfer_bench_simplified.dataset_meta import DBBTaskType, DBBColumnDType
from src.models import ModelConfig, build_model, load_checkpoint, build_classifier
from src.eval_utils import load_task_split, fill_nans, _stable_random_state
from src.utils import get_default_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--max-train", type=int, default=512)
    parser.add_argument("--max-test", type=int, default=256)
    parser.add_argument("--lambda-se", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device_str = args.device or get_default_device()
    device = torch.device(device_str)

    model_cfg = ModelConfig(
        embedding_size=96, num_attention_heads=4,
        mlp_hidden_size=192, num_layers=6,
        use_fk_bias=True, use_entity_bias=True,
    )
    model = build_model(model_cfg)
    load_checkpoint(model, Path(args.checkpoint), str(device))

    if args.lambda_se is not None:
        target = args.lambda_se
        raw_val = float(np.log(np.exp(target) - 1)) if target > 0 else -10.0
        overridden = 0
        for name, module in model.named_modules():
            if hasattr(module, 'raw_lambda') and not hasattr(module, 'raw_lambdas'):
                module.raw_lambda.data.fill_(raw_val)
                overridden += 1
        logger.info("λ_se override: %.4f (%d modules)", target, overridden)

    model.to(device)
    model.eval()
    clf = build_classifier(model, device, model_cfg)

    task_targets = [t.strip() for t in args.tasks.split(",")]

    for target in task_targets:
        ds_name, task_name = target.split("/")
        ds_path = Path("rdb_datasets") / ds_name
        if not ds_path.exists():
            continue

        dataset = DBBRDBDataset(ds_path)
        task = None
        for t in dataset.tasks:
            if t.metadata.task_type == DBBTaskType.classification and t.metadata.name == task_name:
                task = t
                break
        if task is None:
            continue

        X_train, y_train, fk_train, eid_train, peid_train = load_task_split(task, "train")
        X_test, y_test, fk_test, eid_test, peid_test = load_task_split(task, "test")

        seed_key = f"{ds_name}:{task_name}:attn_diag"
        seed_offset = _stable_random_state(seed_key)

        rng = np.random.default_rng(seed_offset)
        if args.max_train and len(X_train) > args.max_train:
            idx = rng.choice(len(X_train), args.max_train, replace=False)
            idx.sort()
            X_tr = X_train[idx]; y_tr = y_train[idx]
            eid_tr = eid_train[idx] if eid_train is not None else None
        else:
            X_tr = X_train; y_tr = y_train; eid_tr = eid_train

        if args.max_test and len(X_test) > args.max_test:
            t_idx = rng.choice(len(X_test), args.max_test, replace=False)
            X_te = X_test[t_idx]; y_te = y_test[t_idx]
            eid_te = eid_test[t_idx] if eid_test is not None else None
        else:
            X_te = X_test; y_te = y_test; eid_te = eid_test

        X_tr, X_te = fill_nans(X_tr, X_te)
        n_train, n_test = len(X_tr), len(X_te)
        n_features = X_tr.shape[1]

        # ── Entity stats ──
        print(f"\n{'='*70}")
        print(f"  {ds_name}/{task_name}")
        print(f"  train={n_train}, test={n_test}, features={n_features}")

        if eid_tr is not None:
            valid = eid_tr >= 0
            n_valid = valid.sum()
            n_unique = len(np.unique(eid_tr[valid]))
            # Count rows per entity in context
            if n_unique > 0:
                _, cnts = np.unique(eid_tr[valid], return_counts=True)
            else:
                cnts = []
            print(f"  entities_in_context={n_unique}, valid_eid_rows={n_valid}, "
                  f"mean_snap={cnts.mean() if len(cnts) > 0 else 0:.1f}, "
                  f"max_snap={cnts.max() if len(cnts) > 0 else 0}")

            # Entity pair ratio in context
            eid_t = torch.from_numpy(eid_tr).long()
            same_mask = (eid_t.unsqueeze(0) == eid_t.unsqueeze(1)) & (eid_t.unsqueeze(0) >= 0)
            n_same = same_mask.sum().item()
            n_total = n_train * n_train
            pair_ratio = n_same / n_total if n_total > 0 else 0
            print(f"  entity_pair_ratio_in_context={pair_ratio:.6f}")

            # Label constancy in context
            el = defaultdict(set)
            for i in range(n_train):
                if eid_tr[i] >= 0:
                    el[int(eid_tr[i])].add(int(y_tr[i]))
            n_const = sum(1 for v in el.values() if len(v) == 1)
            print(f"  const_label_in_context={n_const}/{len(el)} ({n_const/len(el) if el else 0:.3f})")

        # ── Compute entity bias contribution directly ──
        print(f"\n  Entity bias per layer (λ_se, bias statistics):")
        eid_t_full = None
        if eid_tr is not None:
            if eid_te is not None:
                eid_full = np.concatenate([eid_tr, eid_te])
            else:
                eid_full = np.concatenate([eid_tr, np.full(n_test, -1, dtype=np.int64)])
            eid_t_full = torch.from_numpy(eid_full).unsqueeze(0).long().to(device)

        # Run inference and capture entity bias values
        clf.fit(X_tr, y_tr, fk_values=None, entity_ids=eid_tr)

        # Get entity bias contribution for each layer
        with torch.no_grad():
            for li, layer in enumerate(model.transformer_blocks):
                if layer.entity_bias is None:
                    continue
                lam = torch.nn.functional.softplus(layer.entity_bias.raw_lambda).item()

                # Compute: for the context, what fraction of entity pairs get boosted?
                if eid_t_full is not None:
                    eid_ctx = eid_t_full[0, :n_train]
                    same_entity = (eid_ctx.unsqueeze(0) == eid_ctx.unsqueeze(1)) & (eid_ctx.unsqueeze(0) >= 0)
                    n_same_pairs = same_entity.sum().item()
                    avg_bias = lam * (n_same_pairs / (n_train * n_train))
                    # The bias added per same-entity pair is λ_se
                    print(f"  Layer {li}: λ_se={lam:.4f}, "
                          f"same_entity_pairs={n_same_pairs}/{n_train*n_train}, "
                          f"avg_additive_bias={avg_bias:.6f}, "
                          f"max_additive_bias={lam:.4f}")

        # ── Run clean and shuffle_y ──
        acc_clean = None; acc_shuf = None
        for condition, y_use in [("clean", y_tr), ("shuffle_y", rng.permutation(y_tr))]:
            clf.fit(X_tr, y_use, fk_values=None, entity_ids=eid_tr)

            x = np.concatenate([X_tr, X_te])
            x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
            y_t = torch.from_numpy(y_use).unsqueeze(0).float().to(device)

            with torch.no_grad():
                out = model((x_t, y_t), train_test_split_index=n_train,
                           fk_values=None, entity_ids=eid_t_full, parent_entity_ids=None)
                logits = out.squeeze(0)[n_train:, :clf.num_classes]
                if logits.shape[0] == 0 or logits.shape[1] < 2:
                    acc = 0.5
                else:
                    pred = logits.softmax(-1).argmax(-1)
                    acc = (pred.cpu().numpy() == y_te).mean()
            if condition == "clean":
                acc_clean = acc
            else:
                acc_shuf = acc
            print(f"  {condition}: acc={acc:.4f}")

        # ── Compare with: λ_se forced to 0 (disable entity bias) ──
        # Temporarily set raw_lambda to very negative
        raw_saved = {}
        for name, module in model.named_modules():
            if hasattr(module, 'raw_lambda') and not hasattr(module, 'raw_lambdas'):
                raw_saved[name] = module.raw_lambda.data.clone()
                module.raw_lambda.data.fill_(-10.0)  # softplus(-10) ≈ 0

        clf.fit(X_tr, y_tr, fk_values=None, entity_ids=eid_tr)
        x = np.concatenate([X_tr, X_te])
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
        y_t = torch.from_numpy(y_tr).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out_nobias = model((x_t, y_t), train_test_split_index=n_train,
                              fk_values=None, entity_ids=eid_t_full, parent_entity_ids=None)
            logits_nb = out_nobias.squeeze(0)[n_train:, :clf.num_classes]
            acc_nobias = (logits_nb.softmax(-1).argmax(-1).cpu().numpy() == y_te).mean() if logits_nb.shape[0] > 0 else 0.5
        print(f"  entity_bias_disabled (λ_se→0): acc={acc_nobias:.4f}")

        # Restore
        for name, module in model.named_modules():
            if name in raw_saved:
                module.raw_lambda.data.copy_(raw_saved[name])

        # ── Summary ──
        print(f"\n  Summary for {task_name}:")
        print(f"    Δ(clean - no_entity_bias) = {(acc_clean - acc_nobias)*100:+.2f} pp")
        print(f"    Δ(clean - shuffle_y)      = {(acc_clean - acc_shuf)*100:+.2f} pp "
              f"  [acc_shuf={acc_shuf:.4f}]")
        print(f"    acc_no_bias               = {acc_nobias:.4f}")


if __name__ == "__main__":
    main()
