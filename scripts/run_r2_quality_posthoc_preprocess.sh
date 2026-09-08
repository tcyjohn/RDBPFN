#!/usr/bin/env bash
# Preprocess a post-hoc quality-filtered R2 raw corpus and merge it directly
# into H5.  Intentionally does not run filter_h5_sampling_columns.py.

set -euo pipefail

ROOT="${ROOT:-/tmp/RDBPFN-ablation-r2}"
RAW_DIR="${RAW_DIR:-${ROOT}/data_generation/RDB_datasets/ablation_r2_quality_posthoc_1024_raw}"
TMP_DIR="${TMP_DIR:-${RAW_DIR}-tmp}"
PROCESSED_DIR="${PROCESSED_DIR:-${RAW_DIR}-processed}"
H5_OUTPUT="${H5_OUTPUT:-${ROOT}/model_pretrain/pretrain_datasets/ablation_r2_quality_posthoc_1024_direct.h5}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
PYTHON_BIN="${PYTHON_BIN:-/data/caijunyu/RDBPFN/.pixi/envs/default/bin/python}"

export PYTHONPATH="${ROOT}/data_generation/RDB:${ROOT}/data_preprocessing:${ROOT}/model_pretrain:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r2-quality-posthoc}"

mkdir -p "${TMP_DIR}" "${PROCESSED_DIR}" "$(dirname "${H5_OUTPUT}")"
failfile="$(mktemp)"
echo 0 > "${failfile}"

cd "${ROOT}/data_preprocessing"
running=0
for dataset_path in "${RAW_DIR}"/*; do
    [ -d "${dataset_path}" ] || continue
    dataset_name="$(basename "${dataset_path}")"
    out_dir="${PROCESSED_DIR}/${dataset_name}-dfs-2"
    if [ -d "${out_dir}" ]; then
        echo "Skipping existing ${dataset_name}"
        continue
    fi
    (
        echo "Preprocessing ${dataset_name}"
        if "${PYTHON_BIN}" -m tab2graph.main preprocess \
                "${dataset_path}" transform \
                "${TMP_DIR}/${dataset_name}-pre-dfs" \
                -c configs/transform/pre-dfs.yaml \
            && "${PYTHON_BIN}" -m tab2graph.main preprocess \
                "${TMP_DIR}/${dataset_name}-pre-dfs" dfs \
                "${TMP_DIR}/${dataset_name}-post-dfs" \
                -c configs/dfs/dfs-2.yaml \
            && "${PYTHON_BIN}" -m tab2graph.main preprocess \
                "${TMP_DIR}/${dataset_name}-post-dfs" transform \
                "${out_dir}" \
                -c configs/transform/post-dfs.yaml; then
            echo "Finished ${dataset_name}"
        else
            echo "FAILED ${dataset_name}" >&2
            echo 1 >> "${failfile}"
        fi
    ) &
    running=$((running + 1))
    if [ "${running}" -ge "${NUM_PROCESSES}" ]; then
        wait -n || true
        running=$((running - 1))
    fi
done
wait || true

failed=$(($(wc -l < "${failfile}") - 1))
rm -f "${failfile}"
if [ "${failed}" -gt 0 ]; then
    echo "ERROR: ${failed} dataset(s) failed preprocessing." >&2
    exit 1
fi

cd "${ROOT}"
"${PYTHON_BIN}" data_preprocessing/merge_dbinfer_to_h5.py \
    --dataset-root "${PROCESSED_DIR}" \
    --output "${H5_OUTPUT}" \
    --total-rows 600 \
    --max-columns 90 \
    --min-train-ratio 0.5 \
    --max-train-ratio 0.9

echo "Direct H5 ready: ${H5_OUTPUT}"
