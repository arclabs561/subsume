#!/usr/bin/env bash
# Hyperparameter sweep on GALEN EL++ benchmark.
# Tests different LR, margin, and dim combinations.
# Each run takes ~30-60s (1000 epochs).
set -euo pipefail

echo "=== GALEN Hyperparameter Sweep ==="
echo "Each run: 1000 epochs, dim varies"
echo

for dim in 50 100 200; do
  for lr in 0.005 0.01 0.02; do
    for margin in 0.05 0.15 0.3; do
      echo "--- dim=$dim lr=$lr margin=$margin ---"
      BACKEND=candle DIM=$dim EPOCHS=1000 LR=$lr MARGIN=$margin \
        cargo run --features candle-backend --example el_benchmark --release -- data/GALEN 2>&1 \
        | grep -E "NF[1-4]|Restoring"
      echo
    done
  done
done
