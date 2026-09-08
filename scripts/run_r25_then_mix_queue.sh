#!/usr/bin/env bash
# Wait for the active R2.5 train/eval session, then run corrected mix train/eval.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R25_SESSION="ablation_r25_true112k"

echo "Waiting for ${R25_SESSION} to finish..."
while tmux has-session -t "${R25_SESSION}" 2>/dev/null; do
    sleep 60
done

R25_RESULT="${ROOT}/model_pretrain/results/ablation_r25_no_signal_groups_00112_full512_aligned.csv"
if [ ! -f "${R25_RESULT}" ]; then
    echo "ERROR: R2.5 queue exited without expected evaluation result: ${R25_RESULT}" >&2
    exit 1
fi

echo "R2.5 complete. Starting corrected mix train/eval..."
cd "${ROOT}"
bash scripts/run_mix_corrected_true112k_train_eval.sh
