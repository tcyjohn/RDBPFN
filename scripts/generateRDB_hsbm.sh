#!/bin/bash

set -e

HSBM_START=10000
HSBM_COUNT=1024
HSBM_END=$((HSBM_START + HSBM_COUNT))

RDB_GEN_DIR=/data/caijunyu/RDBPFN/data_generation/RDB
RAW_OUTPUT_DIR=/data/caijunyu/RDBPFN/data_generation/RDB_datasets/plurel_scale_complex_hsbm
DST_PREFIX="dag_rdb_complex_hsbm_"

echo "=== 1. Generate raw RDB data (${HSBM_COUNT} datasets, indices ${HSBM_START}-$((HSBM_END - 1))) ==="
cd /data/caijunyu/RDBPFN
pixi run python ./data_generation/RDB/dag_to_rdb_generator.py \
  --dag_data_path "${RDB_GEN_DIR}/datasets/rdb_v1.pth" \
  --config_file "${RDB_GEN_DIR}/dag_to_rdb_config_small.yaml" \
  --num_rdbs ${HSBM_COUNT} \
  --start_index ${HSBM_START} \
  --output_base_dir "${RAW_OUTPUT_DIR}" \
  --use_complex_tasks True

echo "=== 2. Convert to RelBench format ==="
pixi run python ./scripts/convert_dag_rdb.py \
  --range ${HSBM_START} ${HSBM_END} \
  --src-root "${RAW_OUTPUT_DIR}" \
  --dst-prefix "${DST_PREFIX}"

echo "=== 3. Preprocess (rustler pre: ${DST_PREFIX}*) ==="
cd /data/caijunyu/relational-transformer/rustler
shopt -s nullglob
for db_path in "${HOME}/scratch/relbench/${DST_PREFIX}"*/; do
  db=$(basename "$db_path")
  pixi run cargo run --release -- pre "$db"
done
shopt -u nullglob

cd /data/caijunyu/relational-transformer

echo "=== 4. Generate text embeddings ==="
export HF_ENDPOINT=https://hf-mirror.com
pixi run python -u -m rt.embed --bulk_complex=true

echo "=== 5. GPU resource check ==="
if command -v nvidia-smi &>/dev/null; then
    echo "Current GPU usage:"
    nvidia-smi
else
    echo "nvidia-smi not found, skip GPU monitoring"
fi

echo "Sleep 60s to let GPU/system release resources..."
sleep 60

echo ""
echo "=== 6. Pretrain ==="
export PYTHONUNBUFFERED=1
pixi run torchrun --standalone --nproc_per_node=2 scripts/example_pretrain_dag_rdb.py \
  --db-prefix "${DST_PREFIX}" \
  --num-datasets ${HSBM_COUNT}

echo "=== All done! ==="
