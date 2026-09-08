#!/usr/bin/env bash
# Retrain R2.5 for a true 112K optimizer updates, validate, then full-512 eval.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R25_ROOT="/tmp/RDBPFN-ablation-r25"
RUN_NAME="ablation_r25_no_signal_groups_1024_opt112k"
CKPT="${R25_ROOT}/model_pretrain/checkpoints/${RUN_NAME}/model.pt"
EXPECTED_STEP=112000

cd "${R25_ROOT}"
bash scripts/run_ablation_r25_train.sh \
    "${RUN_NAME}" \
    pretrain_datasets/ablation_r25_no_signal_groups_1024.h5 \
    "${EXPECTED_STEP}" \
    0,1 \
    checkpoints/RDBPFN_single/model_eval00360.pt \
    2>&1 | tee "${RUN_NAME}.log"

"${ROOT}/.pixi/envs/default/bin/python" -c '
import sys, torch
path = sys.argv[1]
expected = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
actual = payload.get("step") if isinstance(payload, dict) else None
if actual != expected:
    raise SystemExit(f"checkpoint step mismatch: expected {expected}, got {actual}")
print(f"Verified endpoint step={actual}: {path}")
' "${CKPT}" "${EXPECTED_STEP}"

cd "${ROOT}"
bash scripts/run_eval_aligned.sh \
    "${CKPT}" full-512 0 \
    '+output_path=results/ablation_r25_no_signal_groups_00112_full512_aligned.csv'

echo "R2.5 true-112K training and aligned full-512 evaluation complete."
