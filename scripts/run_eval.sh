#!/bin/bash
# Eval script
# Usage: bash scripts/run_eval.sh <checkpoint_path> [dataset_config] [gpu_id]
#   checkpoint_path: path to .pt checkpoint file (required)
#   dataset_config: hydra dataset config under conf_eval/dataset/ (default: full-512)
#   gpu_id: which GPU to use (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CKPT="${1:?Error: checkpoint path required}"
DATASET="${2:-full-512}"
GPU="${3:-1}"

# Build LD_LIBRARY_PATH from pixi env's nvidia libs
PIXI_ENV="${ROOT}/.pixi/envs/default/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "$PIXI_ENV/nvidia" -name "*.so*" -path "*/lib/*" 2>/dev/null | sed 's|/[^/]*$||' | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_ENDPOINT="https://hf-mirror.com"

echo "=============================================="
echo "  RDBPFN Eval"
echo "=============================================="
echo "  Checkpoint : ${CKPT}"
echo "  Dataset    : ${DATASET}"
echo "  GPU        : ${GPU}"
echo "=============================================="
echo ""

cd "${ROOT}/model_pretrain"

/data/caijunyu/RDBPFN/.pixi/envs/default/bin/python3 -m src.eval \
    "dataset=${DATASET}" \
    "model.checkpoint_path=${CKPT}"

echo ""
echo "Eval complete."
