#!/usr/bin/env bash
set -euo pipefail

for dataset in GALEN GO ANATOMY; do
    echo "=== $dataset ==="
    EPOCHS=5000 cargo run --example el_benchmark --release -- "data/$dataset" 2>&1 | grep -E "NF|Training|Final"
    echo
done
