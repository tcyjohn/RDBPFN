#!/usr/bin/env bash
# Train true 112K-optimizer-step R3 and R2 serially, then evaluate both.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R3_ROOT="/tmp/RDBPFN-ablation-r3"
R2_ROOT="/tmp/RDBPFN-ablation-r2"
R3_RUN="ablation_r3_data_prior_1024_direct_opt112k"
R2_RUN="ablation_r2_original_filter_1024_opt112k"
R3_CKPT="${R3_ROOT}/model_pretrain/checkpoints/${R3_RUN}/model.pt"
R2_CKPT="${R2_ROOT}/model_pretrain/checkpoints/${R2_RUN}/model.pt"
EXPECTED_STEP=112000

validate_step() {
    "${ROOT}/.pixi/envs/default/bin/python" -c '
import sys, torch
path = sys.argv[1]
expected = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
actual = payload.get("step") if isinstance(payload, dict) else None
if actual != expected:
    raise SystemExit(f"checkpoint step mismatch: expected {expected}, got {actual}")
print(f"Verified endpoint step={actual}: {path}")
' "$1" "${EXPECTED_STEP}"
}

echo "Training R3 for ${EXPECTED_STEP} optimizer steps..."
cd "${R3_ROOT}"
bash scripts/run_ablation_r3_direct_train.sh \
    "${R3_RUN}" \
    pretrain_datasets/ablation_r3_data_prior_1024_unsampled.h5 \
    "${EXPECTED_STEP}" \
    0,1 \
    checkpoints/RDBPFN_single/model_eval00360.pt \
    2>&1 | tee "${R3_RUN}.log"
validate_step "${R3_CKPT}"

echo "R3 complete. Training original-filter R2 for ${EXPECTED_STEP} optimizer steps..."
cd "${R2_ROOT}"
bash scripts/run_ablation_r2_train.sh \
    "${R2_RUN}" \
    pretrain_datasets/ablation_r2_original_style_1024.h5 \
    "${EXPECTED_STEP}" \
    0,1 \
    checkpoints/RDBPFN_single/model_eval00360.pt \
    2>&1 | tee "${R2_RUN}.log"
validate_step "${R2_CKPT}"

echo "Both trainings complete. Evaluating R3 direct 112K on full-512..."
cd "${ROOT}"
bash scripts/run_eval_aligned.sh \
    "${R3_CKPT}" full-512 0 \
    '+output_path=results/ablation_r3_data_prior_1024_direct_00112_full512_aligned.csv'

echo "Evaluating R2 original-filter 112K on full-512..."
bash scripts/run_eval_aligned.sh \
    "${R2_CKPT}" full-512 0 \
    '+output_path=results/ablation_r2_original_filter_1024_00112_full512_aligned.csv'

echo "R3/R2 true-112K full-512 evaluation queue complete."
