#!/bin/bash
# Experiment 3: Parameter ablation study
# Runs each ablation (env var) on 32 RDBs, then compares with baseline.
#
# Usage: bash scripts/exp3_ablation.sh

set -e
cd "$(dirname "$0")/.."

NUM_RDBS=32
START=0
BASE_DIR="data_generation/RDB_datasets"
CONFIG="data_generation/RDB/dag_to_rdb_config_small.yaml"
DAG="data_generation/RDB/datasets/rdb_v1.pth"
GEN_SCRIPT="data_generation/RDB/dag_to_rdb_generator.py"
ANALYSIS_SCRIPT="scripts/exp1_analyze.py"

COMMON_FLAGS="--dag_data_path $DAG --config_file $CONFIG --num_rdbs $NUM_RDBS --start_index $START --use_complex_tasks false --relbench_mode --num_processes 4 --no-quality-filter"

run_experiment() {
    local label="$1"
    local env_var="$2"
    local out_dir="${BASE_DIR}/exp3_${label}"

    echo ""
    echo "=== Ablation: ${label} ==="
    echo "    Env: ${env_var}"

    env ${env_var} pixi run python "$GEN_SCRIPT" \
        --output_base_dir "$out_dir" \
        $COMMON_FLAGS 2>&1 | tail -5
}

# Baseline (SG mode, from Exp 1)
BASELINE_DIR="${BASE_DIR}/exp1_mlpx_vs_sg"
if [ ! -d "$BASELINE_DIR" ]; then
    echo "ERROR: Baseline dir $BASELINE_DIR not found."
    echo "       Run scripts/exp1_mlpx_vs_sg.sh first."
    exit 1
fi
echo "Using SG baseline: ${BASELINE_DIR}"

# Run ablations
run_experiment "skip_sg"        "EXP_SKIP_SG=1"
run_experiment "no_basis_perturb" "EXP_ZERO_BASIS_PERTURB=1"
run_experiment "no_coupling"     "EXP_ZERO_COUPLING=1"
run_experiment "no_residual"     "EXP_ZERO_RESIDUAL=1"
run_experiment "uniform_scale"   "EXP_UNIFORM_SCALE=1"
run_experiment "no_alpha_perturb" "EXP_NO_ALPHA_PERTURB=1"
run_experiment "max_groups_1"    "EXP_MAX_GROUPS_1=1"

echo ""
echo "============================================================"
echo "  Exp 3 Ablation Results (${NUM_RDBS} RDBs each)"
echo "============================================================"

for label in skip_sg no_basis_perturb no_coupling no_residual uniform_scale no_alpha_perturb max_groups_1; do
    echo ""
    echo "--- ${label} vs baseline ---"
    pixi run python "$ANALYSIS_SCRIPT" \
        --dir "${BASE_DIR}/exp3_${label}" \
        --num_rdbs "$NUM_RDBS" 2>&1 | grep -E "Per-table|Mean|P10|P50|P90|Δ|distribution"
done
