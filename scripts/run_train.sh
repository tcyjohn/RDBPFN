#!/bin/bash
# Training-only script
# Usage: bash scripts/run_train.sh [run_name] [config_name]
#   run_name: matches pretrain_datasets/<run_name>.h5 and checkpoints/<run_name>/model.pt
#   config_name: hydra config under conf_train/ (default: RDBPFN_hsbm)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

RUN_NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${2:-RDBPFN_hsbm}"

# Build LD_LIBRARY_PATH from pixi env's nvidia libs
PIXI_ENV="${ROOT}/.pixi/envs/default/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "$PIXI_ENV/nvidia" -name "*.so*" -path "*/lib/*" 2>/dev/null | sed 's|/[^/]*$||' | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_ENDPOINT="https://hf-mirror.com"

echo "╔══════════════════════════════════════════════════╗"
echo "║     RDBPFN Training                             ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║ Run    : ${RUN_NAME}"
echo "║ Config : ${CONFIG}"
echo "║ H5     : pretrain_datasets/${RUN_NAME}.h5"
echo "║ Ckpt   : checkpoints/${RUN_NAME}/model.pt"
echo "╚══════════════════════════════════════════════════╝"
echo ""

cd "${ROOT}/model_pretrain"

pixi run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC:-2}" \
    run_train.py \
    --config-name="${CONFIG}" \
    "train.datasets.0.path=pretrain_datasets/${RUN_NAME}.h5" \
    "train.save_model_path=checkpoints/${RUN_NAME}/model.pt" \
    "wandb.run_name=${RUN_NAME}"

echo ""
echo "Training complete."
