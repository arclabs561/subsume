#!/usr/bin/env bash
# Quick local sweep to find the best hyperparams before GPU training.
# Runs short WN18RR trainings on CPU and reports MRR.
#
# Usage: bash scripts/local_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_FILE="scripts/sweep_results.txt"
echo "=== Local WN18RR Sweep ($(date)) ===" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

run_config() {
    local name="$1"
    shift
    echo "--- Running: $name ---"
    local output
    output=$(env "$@" cargo run --features candle-backend --example wn18rr_candle --release 2>&1)
    local mrr=$(echo "$output" | grep "MRR:" | awk '{print $2}')
    local loss=$(echo "$output" | grep "Loss:" | awk -F'-> ' '{print $2}' | awk '{print $1}')
    local time=$(echo "$output" | grep "Training:" | awk '{print $2}')
    echo "  $name: MRR=$mrr loss=$loss time=$time"
    printf "%-40s MRR=%-8s loss=%-8s time=%s\n" "$name" "$mrr" "$loss" "$time" >> "$RESULTS_FILE"
}

# Baseline
run_config "baseline(m=6,lr=0.01)" \
    DIM=32 EPOCHS=30 LR=0.01 NEG=16 BATCH=256 MARGIN=6.0 ADV_TEMP=1.0

# Higher margin
run_config "margin=12" \
    DIM=32 EPOCHS=30 LR=0.01 NEG=16 BATCH=256 MARGIN=12.0 ADV_TEMP=1.0

# Higher margin + more negs
run_config "margin=12,neg=64" \
    DIM=32 EPOCHS=30 LR=0.01 NEG=64 BATCH=256 MARGIN=12.0 ADV_TEMP=1.0

# Lower LR
run_config "lr=0.001,margin=9" \
    DIM=32 EPOCHS=30 LR=0.001 NEG=16 BATCH=256 MARGIN=9.0 ADV_TEMP=1.0

# Higher LR
run_config "lr=0.05,margin=9" \
    DIM=32 EPOCHS=30 LR=0.05 NEG=16 BATCH=256 MARGIN=9.0 ADV_TEMP=1.0

# No self-adversarial
run_config "no_adv,margin=9" \
    DIM=32 EPOCHS=30 LR=0.01 NEG=16 BATCH=256 MARGIN=9.0 ADV_TEMP=0.0

echo ""
echo "=== Results ==="
cat "$RESULTS_FILE"
