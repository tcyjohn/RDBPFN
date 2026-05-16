#!/bin/bash
# Full pipeline: data generation → preprocessing → merge to h5 → training
# Usage: bash scripts/run_pipeline.sh [num_rdbs] [start_index] [run_name] [skip_gen] [skip_preprocess] [skip_merge] [no_quality_filter] [quality_max_retries] [num_processes]
#   num_rdbs: number of RDBs to generate (default: 4)
#   start_index: starting index (default: 0)
#   run_name: output subdirectory name (default: auto-generated with timestamp, e.g. "run_20260515_003000")
#   skip_gen: "true" to skip data generation (default: false)
#   skip_preprocess: "true" to skip preprocessing (default: false)
#   skip_merge: "true" to skip H5 merge (default: false)
#   no_quality_filter: "true" to disable quality gate (default: false; enabled by default)
#   quality_max_retries: max retries for quality gate (default: 3)
#   num_processes: number of parallel worker processes for generation (default: $(nproc))
#
# Output layout:
#   data_generation/RDB_datasets/<run_name>/          raw 4DBInfer data
#   data_generation/RDB_datasets/<run_name>-processed/  DFS-preprocessed data
#   model_pretrain/pretrain_datasets/<run_name>.h5       merged training data
#
# Examples:
#   bash scripts/run_pipeline.sh                        # quick test (4 RDBs, timestamped name)
#   bash scripts/run_pipeline.sh 128 0 my_experiment    # production run
#   bash scripts/run_pipeline.sh 4 0 my_run false true  # gen only, skip preprocess

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# --- Config ---
NUM_RDBS="${1:-4}"
START_INDEX="${2:-0}"
RUN_NAME="${3:-run_$(date +%Y%m%d_%H%M%S)}"
SKIP_GEN="${4:-false}"
SKIP_PREPROCESS="${5:-false}"
SKIP_MERGE="${6:-false}"
NO_QUALITY_FILTER="${7:-false}"
QUALITY_MAX_RETRIES="${8:-3}"
NUM_PROCESSES="${9:-$(nproc 2>/dev/null || echo 4)}"
END_INDEX=$((START_INDEX + NUM_RDBS))

RDB_GEN_DIR="${ROOT}/data_generation/RDB"
RAW_OUTPUT_DIR="${ROOT}/data_generation/RDB_datasets/${RUN_NAME}"
TMP_DIR="${RAW_OUTPUT_DIR}-tmp"
PROCESSED_DIR="${RAW_OUTPUT_DIR}-processed"
H5_OUTPUT="${ROOT}/model_pretrain/pretrain_datasets/${RUN_NAME}.h5"

PREPROCESS_SCRIPT="${ROOT}/data_preprocessing/run_preprocess.py"
MERGE_SCRIPT="${ROOT}/data_preprocessing/merge_dbinfer_to_h5.py"
PRE_DFS_CONFIG="${ROOT}/data_preprocessing/configs/transform/pre-dfs.yaml"
DFS_CONFIG="${ROOT}/data_preprocessing/configs/dfs/dfs-1-ft.yaml"
POST_DFS_CONFIG="${ROOT}/data_preprocessing/configs/transform/post-dfs.yaml"

# Build LD_LIBRARY_PATH from pixi env's nvidia libs
PIXI_ENV="${ROOT}/.pixi/envs/default/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "$PIXI_ENV/nvidia" -name "*.so*" -path "*/lib/*" 2>/dev/null | sed 's|/[^/]*$||' | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_ENDPOINT="https://hf-mirror.com"

echo "╔══════════════════════════════════════════════════╗"
echo "║     RDBPFN Full Pipeline                         ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║ Run : ${RUN_NAME}"
echo "║ RDBs: ${NUM_RDBS} (indices ${START_INDEX}-$((END_INDEX - 1)))"
echo "║ Raw : ${RAW_OUTPUT_DIR}"
echo "║ H5  : ${H5_OUTPUT}"
echo "║ QC  : enabled=$([ "${NO_QUALITY_FILTER}" != "true" ] && echo "yes" || echo "no")  max_retries=${QUALITY_MAX_RETRIES}"
echo "║ Proc: ${NUM_PROCESSES}"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ============================================================
# Step 1: Data Generation
# ============================================================
if [ "${SKIP_GEN}" != "true" ]; then
    echo "=== [1/5] Generating ${NUM_RDBS} RDB datasets ==="
    mkdir -p "${RAW_OUTPUT_DIR}"

    cd "${ROOT}"

    QUALITY_FLAGS=()
    if [ "${NO_QUALITY_FILTER}" = "true" ]; then
        QUALITY_FLAGS+=(--no-quality-filter)
    fi
    QUALITY_FLAGS+=(--quality-max-retries "${QUALITY_MAX_RETRIES}")

    pixi run python ./data_generation/RDB/dag_to_rdb_generator.py \
        --dag_data_path "${RDB_GEN_DIR}/datasets/rdb_v1.pth" \
        --config_file "${RDB_GEN_DIR}/dag_to_rdb_config_small.yaml" \
        --num_rdbs "${NUM_RDBS}" \
        --start_index "${START_INDEX}" \
        --output_base_dir "${RAW_OUTPUT_DIR}" \
        --use_complex_tasks True \
        --num_processes "${NUM_PROCESSES}" \
        "${QUALITY_FLAGS[@]}"

    echo "Data generation complete."
else
    echo "=== [1/5] SKIP data generation ==="
fi

# ============================================================
# Step 2: Preprocessing (3-step: pre-dfs → dfs → post-dfs)
# ============================================================
if [ "${SKIP_PREPROCESS}" != "true" ]; then
    echo ""
    echo "=== [2/5] Preprocessing (pre-dfs → dfs → post-dfs, ${NUM_PROCESSES} workers) ==="
    mkdir -p "${TMP_DIR}"
    mkdir -p "${PROCESSED_DIR}"

    FAILFILE=$(mktemp)
    echo 0 > "${FAILFILE}"
    running=0

    for idx in $(seq "${START_INDEX}" $((END_INDEX - 1))); do
        (
            ds_name="dag_rdb_${idx}"
            src="${RAW_OUTPUT_DIR}/${ds_name}"
            tmp_pre="${TMP_DIR}/${ds_name}-pre-dfs"
            tmp_post="${TMP_DIR}/${ds_name}-post-dfs"
            out="${PROCESSED_DIR}/${ds_name}-dfs-1"

            if [ -d "${out}" ]; then
                echo "[SKIP ${idx}/${ds_name}] Already processed"
                exit 0
            fi
            if [ ! -d "${src}" ]; then
                echo "[SKIP ${idx}/${ds_name}] Source not found at ${src}" >&2
                exit 1
            fi

            echo "[${idx}/${ds_name}] Preprocessing..."
            if pixi run python "${PREPROCESS_SCRIPT}" \
                "${src}" transform "${tmp_pre}" "${PRE_DFS_CONFIG}" 1 \
            && pixi run python "${PREPROCESS_SCRIPT}" \
                "${tmp_pre}" dfs "${tmp_post}" "${DFS_CONFIG}" 1 \
            && pixi run python "${PREPROCESS_SCRIPT}" \
                "${tmp_post}" transform "${out}" "${POST_DFS_CONFIG}" 1; then
                rm -rf "${tmp_pre}" "${tmp_post}"
                echo "[OK ${idx}/${ds_name}]"
            else
                echo "[FAIL ${idx}/${ds_name}]" >&2
                echo 1 >> "${FAILFILE}"
            fi
        ) &
        running=$((running + 1))
        if [ "${running}" -ge "${NUM_PROCESSES}" ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait

    FAILED=$(($(wc -l < "${FAILFILE}") - 1))
    rm -f "${FAILFILE}"

    if [ "${FAILED}" -gt 0 ]; then
        echo "WARNING: ${FAILED} dataset(s) failed preprocessing."
    fi
    echo "Preprocessing complete."
else
    echo "=== [2/4] SKIP preprocessing ==="
fi

# ============================================================
# Step 3: Merge to H5
# ============================================================
if [ "${SKIP_MERGE}" != "true" ]; then
    echo ""
    echo "=== [3/5] Merging preprocessed datasets to H5 ==="

    # Determine max-columns from DFS depth
    DFS_DEPTH=$(grep -Po 'max_depth:\s*\K\d+' "${DFS_CONFIG}" || echo "1")
    case "${DFS_DEPTH}" in
        1) MAX_COLS=60 ;;
        2) MAX_COLS=90 ;;
        *) MAX_COLS=60 ;;
    esac
    echo "DFS depth=${DFS_DEPTH} → max-columns=${MAX_COLS}"

    cd "${ROOT}"
    pixi run python "${MERGE_SCRIPT}" \
        --dataset-root "${PROCESSED_DIR}" \
        --output "${H5_OUTPUT}" \
        --total-rows 600 \
        --max-columns "${MAX_COLS}"

    echo "H5 merge complete: ${H5_OUTPUT}"
else
    echo ""
    echo "=== [3/5] SKIP H5 merge ==="
fi

# ============================================================
# Step 4: Convert first 20 synthetic RDBs to eval CSVs
# ============================================================
echo ""
echo "=== [4/5] Converting synthetic tasks to eval CSVs ==="

CLF_DIR="${ROOT}/model_pretrain/datasets/clf"
rm -rf "${CLF_DIR:?}"/*
mkdir -p "${CLF_DIR}"

N_EVAL="${NUM_RDBS}"
if [ "${N_EVAL}" -gt 10 ]; then N_EVAL=10; fi

EVAL_DATASETS=""
for i in $(seq "${START_INDEX}" $((START_INDEX + N_EVAL - 1))); do
    EVAL_DATASETS="${EVAL_DATASETS} dag_rdb_${i}"
done

pixi run python "${ROOT}/scripts/convert_rdb_to_csv.py" \
    --src-dir "${RAW_OUTPUT_DIR}" \
    --output-dir "${CLF_DIR}" \
    --datasets ${EVAL_DATASETS}

echo "Eval CSVs ready in ${CLF_DIR} ($(ls "${CLF_DIR}"/*.csv 2>/dev/null | wc -l) files)"

# ============================================================
# Step 5: Training
# ============================================================
echo ""
echo "=== [5/5] Launching training ==="

cd "${ROOT}/model_pretrain"

pixi run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC:-2}" \
    run_train.py \
    --config-name=RDBPFN_hsbm \
    "train.datasets.0.path=pretrain_datasets/${RUN_NAME}.h5" \
    "train.save_model_path=checkpoints/${RUN_NAME}/model.pt" \
    "wandb.run_name=${RUN_NAME}"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Pipeline Complete!                            ║"
echo "╚══════════════════════════════════════════════════╝"
