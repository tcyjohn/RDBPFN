#!/bin/bash
# Train a source-balanced v6.2 + R3 mixture without regenerating data.
# Usage: bash scripts/run_mix_v62_r3.sh [cuda_devices] [run_name] [num_steps] [extra_hydra_args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CUDA_DEVICES_ARG="${1:-0,1}"
RUN_NAME="${2:-mix_v62_r3_entitybias}"
OPTIMIZER_STEPS="${3:-2200000}"
shift $(( $# >= 3 ? 3 : $# ))

V62_H5="${ROOT}/model_pretrain/pretrain_datasets/v6.2.h5"
R3_H5="${ROOT}/model_pretrain/pretrain_datasets/ablation_r3_data_prior_1024_direct.h5"
for required in "${V62_H5}" "${R3_H5}"; do
    if [ ! -f "${required}" ]; then
        echo "Missing required dataset: ${required}" >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_ARG}"
GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
export HF_ENDPOINT="https://hf-mirror.com"

if [ -n "${http_proxy:-}${HTTP_PROXY:-}" ]; then
    _proxy="${http_proxy:-$HTTP_PROXY}"
    export http_proxy="${_proxy}"
    export https_proxy="${https_proxy:-${_proxy}}"
    export HTTP_PROXY="${_proxy}"
    export HTTPS_PROXY="${https_proxy:-${_proxy}}"
    export no_proxy="${no_proxy:-localhost,127.0.0.1},api.wandb.ai,wandb.ai"
    export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1},api.wandb.ai,wandb.ai"
fi

echo "Run: ${RUN_NAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${GPU_COUNT} processes)"
echo "Optimizer steps: ${OPTIMIZER_STEPS}"
echo "Sources: v6.2=0.5, R3=0.5"
echo "Biases: entity=on, FK=off"

cd "${ROOT}/model_pretrain"
exec pixi run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPU_COUNT}" \
    run_train.py \
    --config-name=RDBPFN_mix_v62_r3 \
    "train.num_steps=${OPTIMIZER_STEPS}" \
    "train.num_gpus=${GPU_COUNT}" \
    "train.save_model_path=checkpoints/${RUN_NAME}/model.pt" \
    "wandb.run_name=${RUN_NAME}" \
    "$@"
