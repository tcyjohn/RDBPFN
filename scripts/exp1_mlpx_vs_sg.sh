#!/bin/bash
# Experiment 1: SG-only vs SCM-only feature-label correlation
# Two independent runs, both with relbench_mode + complex tasks.
#
# Usage: bash scripts/exp1_mlpx_vs_sg.sh

set -e
cd "$(dirname "$0")/.."

NUM_RDBS=64
START=0
CONFIG="data_generation/RDB/dag_to_rdb_config_small.yaml"
DAG="data_generation/RDB/datasets/rdb_v1.pth"
GEN_SCRIPT="data_generation/RDB/dag_to_rdb_generator.py"
BASE_DIR="data_generation/RDB_datasets"

echo "=============================================="
echo "  Exp 1: SG-only vs SCM-only"
echo "  ${NUM_RDBS} RDBs each, relbench_mode + complex_tasks"
echo "=============================================="

# Run 1: SG-only (default: signal-group constructs features)
echo ""
echo "=== [1/2] SG-only: X = X_sg ==="
pixi run python "$GEN_SCRIPT" \
    --dag_data_path "$DAG" \
    --config_file "$CONFIG" \
    --num_rdbs "$NUM_RDBS" \
    --start_index "$START" \
    --output_base_dir "${BASE_DIR}/exp1_sg_only" \
    --use_complex_tasks true \
    --relbench_mode \
    --num_processes 4 \
    --no-quality-filter

# Run 2: SCM-only (EXP_SKIP_SG=1: MLP raw contiguous slice as features)
echo ""
echo "=== [2/2] SCM-only: X = MLP raw slice ==="
EXP_SKIP_SG=1 pixi run python "$GEN_SCRIPT" \
    --dag_data_path "$DAG" \
    --config_file "$CONFIG" \
    --num_rdbs "$NUM_RDBS" \
    --start_index "$START" \
    --output_base_dir "${BASE_DIR}/exp1_scm_only" \
    --use_complex_tasks true \
    --relbench_mode \
    --num_processes 4 \
    --no-quality-filter

echo ""
echo "=== Analysis ==="
pixi run python scripts/exp1_analyze.py \
    --dir_sg "${BASE_DIR}/exp1_sg_only" \
    --dir_scm "${BASE_DIR}/exp1_scm_only" \
    --num_rdbs "$NUM_RDBS"
