#!/usr/bin/env bash
# Train corrected v6.2 + R3-direct mix for 112K optimizer updates, then evaluate.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_NAME="mix_v62_r3direct_entitybias_opt112k"
CKPT="${ROOT}/model_pretrain/checkpoints/${RUN_NAME}/model.pt"
EXPECTED_STEP=112000

cd "${ROOT}"

# Exercise the exact corrected data/model path before committing to the long run.
bash scripts/run_mix_v62_r3.sh \
    0,1 \
    mix_v62_r3direct_entitybias_smoke \
    1 \
    wandb.enabled=false \
    2>&1 | tee model_pretrain/logs/mix_v62_r3direct_entitybias_smoke.log

"${ROOT}/.pixi/envs/default/bin/python" -c '
import torch
path = "model_pretrain/checkpoints/mix_v62_r3direct_entitybias_smoke/model.pt"
actual = torch.load(path, map_location="cpu", weights_only=False).get("step")
if actual != 1:
    raise SystemExit(f"mix smoke checkpoint step mismatch: expected 1, got {actual}")
print(f"Verified mix smoke endpoint step={actual}: {path}")
'

bash scripts/run_mix_v62_r3.sh \
    0,1 \
    "${RUN_NAME}" \
    "${EXPECTED_STEP}" \
    2>&1 | tee "model_pretrain/logs/${RUN_NAME}.log"

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

bash scripts/run_eval_aligned.sh \
    "${CKPT}" full-512 0 \
    'model.nanopfn.use_entity_bias=true model.nanopfn.use_fk_bias=false dataset.eval_chunk_size=256 +dataset.use_primary_key_as_entity_id=true +output_path=results/mix_v62_r3direct_entitybias_00112_full512_aligned.csv'

echo "Corrected mix true-112K training and aligned full-512 evaluation complete."
