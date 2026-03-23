#!/usr/bin/env bash
# GPU benchmark for WN18RR with CandleBoxTrainer.
#
# Run on the GPU instance after: cargo build --features cuda --example wn18rr_candle --release
#
# Usage: bash scripts/gpu_benchmark.sh [quick|full]
#   quick: 3 configs, ~30 min on A10G
#   full:  12 configs, ~3 hours on A10G

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-quick}"
RESULTS="scripts/gpu_benchmark_results.txt"
BINARY="target/release/examples/wn18rr_candle"

if [ ! -f "$BINARY" ]; then
    echo "Building..."
    cargo build --features cuda --example wn18rr_candle --release
fi

echo "=== WN18RR GPU Benchmark ($MODE) -- $(date) ===" | tee "$RESULTS"
echo "" | tee -a "$RESULTS"

run() {
    local name="$1"
    shift
    echo "--- $name ---"
    local out
    out=$(env "$@" "$BINARY" 2>&1)
    local mrr h1 h3 h10 mr loss time
    mrr=$(echo "$out" | grep "MRR:" | awk '{print $2}')
    h1=$(echo "$out" | grep "MRR:" | awk '{print $4}')
    h3=$(echo "$out" | grep "MRR:" | awk '{print $6}')
    h10=$(echo "$out" | grep "MRR:" | awk '{print $8}')
    mr=$(echo "$out" | grep "MRR:" | awk '{print $10}')
    loss=$(echo "$out" | grep "Loss:" | awk -F'-> ' '{print $2}' | awk '{print $1}')
    time=$(echo "$out" | grep "Training:" | awk '{print $2}')
    printf "%-45s MRR=%-7s H@1=%-7s H@10=%-7s MR=%-9s loss=%-8s time=%s\n" \
        "$name" "$mrr" "$h1" "$h10" "$mr" "$loss" "$time" | tee -a "$RESULTS"
}

if [ "$MODE" = "quick" ]; then
    # Quick sweep: 3 configs to find the right ballpark
    run "dim200_ep200_m9_lr001" \
        DIM=200 EPOCHS=200 LR=0.001 NEG=64 BATCH=512 MARGIN=9.0 ADV_TEMP=1.0 INSIDE_W=0.0 BOUNDS_EVERY=0

    run "dim200_ep200_m12_lr0005" \
        DIM=200 EPOCHS=200 LR=0.0005 NEG=128 BATCH=512 MARGIN=12.0 ADV_TEMP=1.0 INSIDE_W=0.0 BOUNDS_EVERY=0

    run "dim200_ep200_m9_inside" \
        DIM=200 EPOCHS=200 LR=0.001 NEG=64 BATCH=512 MARGIN=9.0 ADV_TEMP=1.0 INSIDE_W=0.05 BOUNDS_EVERY=50

elif [ "$MODE" = "full" ]; then
    # Full sweep: systematically vary key hyperparams

    # Margin sweep (most impactful for box embeddings)
    for M in 3 6 9 12 18; do
        run "margin=$M" \
            DIM=200 EPOCHS=300 LR=0.001 NEG=128 BATCH=512 MARGIN=$M ADV_TEMP=1.0 INSIDE_W=0.0 BOUNDS_EVERY=0
    done

    # LR sweep
    for LR in 0.0001 0.0005 0.001 0.005; do
        run "lr=$LR" \
            DIM=200 EPOCHS=300 LR=$LR NEG=128 BATCH=512 MARGIN=9.0 ADV_TEMP=1.0 INSIDE_W=0.0 BOUNDS_EVERY=0
    done

    # Inside weight
    run "inside_w=0.05" \
        DIM=200 EPOCHS=300 LR=0.001 NEG=128 BATCH=512 MARGIN=9.0 ADV_TEMP=1.0 INSIDE_W=0.05 BOUNDS_EVERY=50

    # Bounds freshness
    run "bounds_every=20" \
        DIM=200 EPOCHS=300 LR=0.001 NEG=128 BATCH=512 MARGIN=9.0 ADV_TEMP=1.0 INSIDE_W=0.0 BOUNDS_EVERY=20

    # Best config extended
    run "extended_500ep" \
        DIM=200 EPOCHS=500 LR=0.001 NEG=128 BATCH=512 MARGIN=9.0 ADV_TEMP=1.0 INSIDE_W=0.02 BOUNDS_EVERY=50
fi

echo ""
echo "=== Summary ==="
sort -t= -k2 -rn "$RESULTS" | head -5
echo ""
echo "Full results: $RESULTS"
