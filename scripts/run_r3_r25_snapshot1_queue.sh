#!/usr/bin/env bash
# Regenerate, retrain, and full-512 evaluate R3 then R2.5 with one entity
# snapshot. All artifact names are isolated from the earlier 5-20 snapshot runs.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R3_ROOT="/tmp/RDBPFN-ablation-r3"
R25_ROOT="/tmp/RDBPFN-ablation-r25"
R3_RUN="ablation_r3_data_prior_1024_snapshot1_direct_opt112k"
R25_RUN="ablation_r25_no_signal_groups_1024_snapshot1_opt112k"
R3_CKPT="${R3_ROOT}/model_pretrain/checkpoints/${R3_RUN}/model.pt"
R25_CKPT="${R25_ROOT}/model_pretrain/checkpoints/${R25_RUN}/model.pt"
EXPECTED_STEP=112000
NUM_RDBS=1024
NUM_PROCESSES=16
CUDA_DEVICES="0,1"

export SNAPSHOTS_PER_ENTITY_MIN=1
export SNAPSHOTS_PER_ENTITY_MAX=1

validate_rows() {
    local raw_dir="$1"
    "${ROOT}/.pixi/envs/default/bin/python" -c '
import glob
import os
import sys

import pyarrow.parquet as pq

raw_dir = sys.argv[1]
files = [
    path
    for path in glob.glob(os.path.join(raw_dir, "dag_rdb_*", "table_*.parquet"))
    if os.path.isfile(path)
]
if not files:
    raise SystemExit(f"no generated table parquet files found under {raw_dir}")
counts = [pq.ParquetFile(path).metadata.num_rows for path in files]
if min(counts) < 1000 or max(counts) > 5000:
    raise SystemExit(
        f"snapshot1 row-count validation failed: min={min(counts)}, max={max(counts)}"
    )
print(
    f"Verified snapshot1 rows: tables={len(counts)}, "
    f"min={min(counts)}, max={max(counts)}"
)
' "${raw_dir}"
}

validate_checkpoint() {
    local checkpoint="$1"
    "${ROOT}/.pixi/envs/default/bin/python" -c '
import sys

import torch

path = sys.argv[1]
expected = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
actual = payload.get("step") if isinstance(payload, dict) else None
if actual != expected:
    raise SystemExit(f"checkpoint step mismatch: expected {expected}, got {actual}")
print(f"Verified endpoint step={actual}: {path}")
' "${checkpoint}" "${EXPECTED_STEP}"
}

echo "=== Phase 1: R3 snapshot1 generation/preprocessing/merge ==="
cd "${R3_ROOT}"
bash scripts/run_ablation_r3_pipeline.sh \
    "${NUM_RDBS}" 0 "${R3_RUN}" \
    false false false true \
    "${CUDA_DEVICES}" "${EXPECTED_STEP}" "${NUM_PROCESSES}" ""
validate_rows "${R3_ROOT}/data_generation/RDB_datasets/${R3_RUN}_raw"

echo "=== Phase 2: R3 snapshot1 true-112K training ==="
bash scripts/run_ablation_r3_direct_train.sh \
    "${R3_RUN}" \
    "pretrain_datasets/${R3_RUN}_unsampled.h5" \
    "${EXPECTED_STEP}" \
    "${CUDA_DEVICES}" \
    "checkpoints/RDBPFN_single/model_eval00360.pt"
validate_checkpoint "${R3_CKPT}"

echo "=== Phase 3: R3 snapshot1 full-512 evaluation ==="
cd "${ROOT}"
bash scripts/run_eval_aligned.sh \
    "${R3_CKPT}" full-512 0 \
    '+output_path=results/ablation_r3_data_prior_1024_snapshot1_00112_full512_aligned.csv'

echo "=== Phase 4: R2.5 snapshot1 generation/preprocessing/merge ==="
cd "${R25_ROOT}"
bash scripts/run_ablation_r25_pipeline.sh \
    "${NUM_RDBS}" 0 "${R25_RUN}" \
    false false false true \
    "${CUDA_DEVICES}" "${EXPECTED_STEP}" "${NUM_PROCESSES}" ""
validate_rows "${R25_ROOT}/data_generation/RDB_datasets/${R25_RUN}_raw"

echo "=== Phase 5: R2.5 snapshot1 true-112K training ==="
bash scripts/run_ablation_r25_train.sh \
    "${R25_RUN}" \
    "pretrain_datasets/${R25_RUN}.h5" \
    "${EXPECTED_STEP}" \
    "${CUDA_DEVICES}" \
    "checkpoints/RDBPFN_single/model_eval00360.pt"
validate_checkpoint "${R25_CKPT}"

echo "=== Phase 6: R2.5 snapshot1 full-512 evaluation ==="
cd "${ROOT}"
bash scripts/run_eval_aligned.sh \
    "${R25_CKPT}" full-512 0 \
    '+output_path=results/ablation_r25_no_signal_groups_1024_snapshot1_00112_full512_aligned.csv'

echo "R3-first snapshot1 retraining and full-512 evaluation queue complete."
