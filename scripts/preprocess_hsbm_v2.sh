#!/bin/bash
# Batch preprocessing script for plurel_scale_complex_hsbm_v2 datasets.
# Runs tab2graph pre-dfs → DFS depth=2 → post-dfs on all 128 datasets.
#
# Usage: bash scripts/preprocess_hsbm_v2.sh [START] [END] [MAX_JOBS]
#   START: first dataset index (default: 0)
#   END: last dataset index (default: 128)
#   MAX_JOBS: parallel workers (default: 4)

set -e

START="${1:-0}"
END="${2:-128}"
MAX_JOBS="${3:-4}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RAW_DIR="${PROJECT_ROOT}/data_generation/RDB_datasets/plurel_scale_complex_hsbm_v2"
TMP_DIR="${RAW_DIR}-tmp"
PROCESSED_DIR="${RAW_DIR}-processed"

# Ensure tab2graph and dbinfer_bench_simplified on path
export PYTHONPATH="${PROJECT_ROOT}/data_preprocessing:${PROJECT_ROOT}/model_pretrain/src:${PYTHONPATH}"

PREPROCESS_SCRIPT="${PROJECT_ROOT}/data_preprocessing/run_preprocess.py"
PRE_DFS_CONFIG="${PROJECT_ROOT}/data_preprocessing/configs/transform/pre-dfs.yaml"
DFS_CONFIG="${PROJECT_ROOT}/data_preprocessing/configs/dfs/dfs-2-ft.yaml"
POST_DFS_CONFIG="${PROJECT_ROOT}/data_preprocessing/configs/transform/post-dfs.yaml"

echo "=== Batch preprocessing datasets ${START}-$((END-1)) (max ${MAX_JOBS} jobs, depth=2) ==="
echo "RAW: ${RAW_DIR}"
echo "TMP: ${TMP_DIR}"
echo "OUT: ${PROCESSED_DIR}"
echo ""

mkdir -p "${TMP_DIR}"
mkdir -p "${PROCESSED_DIR}"

process_one() {
    local idx="$1"
    local ds_name="dag_rdb_${idx}"
    local src="${RAW_DIR}/${ds_name}"
    local tmp_pre="${TMP_DIR}/${ds_name}-pre-dfs"
    local tmp_post="${TMP_DIR}/${ds_name}-post-dfs"
    local out="${PROCESSED_DIR}/${ds_name}-dfs-2"

    if [ -d "${out}" ]; then
        echo "[SKIP ${idx}/${ds_name}] Already processed"
        return 0
    fi
    if [ ! -d "${src}" ]; then
        echo "[SKIP ${idx}/${ds_name}] Source not found"
        return 0
    fi

    echo "[START ${idx}/${ds_name}]"

    pixi run python "${PREPROCESS_SCRIPT}" \
        "${src}" transform "${tmp_pre}" "${PRE_DFS_CONFIG}" 2 \
        && pixi run python "${PREPROCESS_SCRIPT}" \
            "${tmp_pre}" dfs "${tmp_post}" "${DFS_CONFIG}" 2 \
        && pixi run python "${PREPROCESS_SCRIPT}" \
            "${tmp_post}" transform "${out}" "${POST_DFS_CONFIG}" 2 \
        || { echo "[FAIL ${idx}/${ds_name}]" >&2; return 1; }

    # Clean up intermediate files to save space
    rm -rf "${tmp_pre}" "${tmp_post}"
    echo "[OK ${idx}/${ds_name}]"
}

export -f process_one
export PREPROCESS_SCRIPT PRE_DFS_CONFIG DFS_CONFIG POST_DFS_CONFIG TMP_DIR PROCESSED_DIR RAW_DIR

echo "Processing datasets ${START} to $((END-1))..."
for idx in $(seq "${START}" $((END-1))); do
    echo "${idx}"
done | xargs -P "${MAX_JOBS}" -I {} bash -c 'process_one "$@"' _ {}

echo ""
echo "=== All done! Output in: ${PROCESSED_DIR} ==="
