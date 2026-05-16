#! /bin/bash

set -e 

echo "=== 1. 开始生成 RDB ==="
cd /data/caijunyu/RDBPFN
# RDB_GEN_DIR=/data/caijunyu/RDBPFN/data_generation/RDB
# pixi run python ./data_generation/RDB/dag_to_rdb_generator.py \
#   --dag_data_path "${RDB_GEN_DIR}/datasets/rdb_v1.pth" \
#  --config_file "${RDB_GEN_DIR}/dag_to_rdb_config_small.yaml" \
#  --num_rdbs 1024 \
#  --output_base_dir /data/caijunyu/RDBPFN/data_generation/RDB_datasets/plurel_scale_complex_raw \
#  --use_complex_tasks True

echo "=== 2. 开始转换格式 ==="
pixi run python ./scripts/convert_dag_rdb.py --all


echo "=== 3. 开始预处理（relbench: dag_rdb_complex_* → ~/scratch/pre/dag_rdb_complex_*）==="
cd /data/caijunyu/relational-transformer/rustler
shopt -s nullglob
for db_path in "${HOME}/scratch/relbench/dag_rdb_complex_"*/; do
  db=$(basename "$db_path")
  pixi run cargo run --release -- pre "$db"
done
shopt -u nullglob

cd /data/caijunyu/relational-transformer # 回到原目录

echo "=== 4. 开始生成 Text embedding（~/scratch/pre/dag_rdb_complex_*）==="
export HF_ENDPOINT=https://hf-mirror.com
pixi run python -u -m rt.embed --bulk_complex=true

echo "=== 5. 预训练前休眠并清理资源（尝试释放GPU显存）==="

# 可选资源清理（如需）
if command -v nvidia-smi &>/dev/null; then
    echo "当前GPU占用情况："
    nvidia-smi
else
    echo "未找到 nvidia-smi，跳过GPU监控"
fi

# 休眠一段时间给系统调度和垃圾回收留缓冲
echo "休眠60秒，让GPU和系统有时间释放未清理资源..."
sleep 300

echo ""
echo "=== 5. 开始pretrain ==="
export PYTHONUNBUFFERED=1
pixi run torchrun --standalone --nproc_per_node=2 scripts/example_pretrain_dag_rdb.py --complex

echo "=== 全部任务完成！ ==="

