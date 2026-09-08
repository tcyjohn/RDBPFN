#!/bin/bash
# Eval wrapper with row-aligned structural-array downsampling
#
# Usage: bash scripts/run_eval_aligned.sh <checkpoint_path> [dataset_config] [gpu_id] [extra_args]
#
# Arguments:
#   checkpoint_path  Path to .pt checkpoint file (required)
#   dataset_config   Hydra dataset config name under conf_eval/dataset/ (default: full-512)
#   gpu_id           Which GPU to use (default: 1)
#   extra_args       Extra Hydra CLI overrides (e.g. "model.nanopfn.use_fk_bias=true")
#
# Available dataset configs:
#   clf_rel_npz      Relbench classification tasks (19 datasets, uses clf_rel_subsamples/ cache)
#   clf_npz          Real-world tabular classification tasks (19 datasets, uses clf_real_subsamples/ cache)
#   full-512         Synthetic RDB datasets (default, 512-row variant)
#   full-1024/full-256/full-128/full-64/full-32  Other synthetic RDB size variants
#   full-debug       Small debug set
#
# Examples:
#   bash scripts/run_eval_aligned.sh checkpoints/RDBPFN/model_eval00528.pt clf_rel_npz 0
#   bash scripts/run_eval_aligned.sh checkpoints/v5.1_relmode/model.pt clf_npz 1
#   bash scripts/run_eval_aligned.sh checkpoints/RDBPFN_single/model_eval00360.pt full-512 0
#   bash scripts/run_eval_aligned.sh checkpoints/fk_bias_run/model.pt clf_rel_npz 0 "model.nanopfn.use_fk_bias=true"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CKPT="${1:?Error: checkpoint path required}"
DATASET="${2:-full-512}"
GPU="${3:-${CUDA_VISIBLE_DEVICES:-2}}"
EXTRA_EVAL_ARGS="${4:-}"  # optional extra Hydra overrides

# Choose config name: CSV datasets (clf_*) use eval_csv, RDB datasets (full-*) use eval
if [[ "${DATASET}" == clf_* ]]; then
    CONFIG_NAME="eval_csv"
else
    CONFIG_NAME="eval"
fi

# Build LD_LIBRARY_PATH from pixi env's nvidia libs
PIXI_ENV="${ROOT}/.pixi/envs/default/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "$PIXI_ENV/nvidia" -name "*.so*" -path "*/lib/*" 2>/dev/null | sed 's|/[^/]*$||' | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_ENDPOINT="https://hf-mirror.com"

echo "=============================================="
echo "  RDBPFN Eval (aligned downsampling)"
echo "=============================================="
echo "  Checkpoint : ${CKPT}"
echo "  Dataset    : ${DATASET}"
echo "  GPU        : ${GPU}"
echo "=============================================="
echo ""

cd "${ROOT}/model_pretrain"

EVAL_ARGS=(
    --config-name="${CONFIG_NAME}"
    "dataset=${DATASET}"
    "model.checkpoint_path=${CKPT}"
)
if [ -n "${EXTRA_EVAL_ARGS}" ]; then
    for arg in ${EXTRA_EVAL_ARGS}; do
        EVAL_ARGS+=("${arg}")
    done
fi

/data/caijunyu/RDBPFN/.pixi/envs/default/bin/python3 -m src.eval_aligned "${EVAL_ARGS[@]}"

echo ""
echo "Eval complete."
