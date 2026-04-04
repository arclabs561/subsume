#!/usr/bin/env bash
# Benchmark box embeddings on WN18RR.
#
# Usage:
#   # Local (quick test):
#   ./scripts/benchmark_wn18rr.sh
#
#   # Remote (full benchmark):
#   runctl aws create g4dn.xlarge --spot
#   # scp repo to instance, then:
#   DIM=200 EPOCHS=500 NEG=10 SELF_ADV=1 ./scripts/benchmark_wn18rr.sh
#
# Environment variables:
#   DIM       Embedding dimension (default: 200)
#   EPOCHS    Training epochs (default: 500)
#   LR        Learning rate (default: 0.0005)
#   NEG       Negative samples per triple (default: 10)
#   SELF_ADV  Self-adversarial sampling: 1=on, 0=off (default: 1)

set -euo pipefail

cd "$(dirname "$0")/.."

export DIM="${DIM:-200}"
export EPOCHS="${EPOCHS:-500}"
export LR="${LR:-0.0005}"
export NEG="${NEG:-10}"
export SELF_ADV="${SELF_ADV:-1}"

# Install Rust if not present
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

echo "=== WN18RR Box Embedding Benchmark ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname)"
echo "Config: dim=$DIM epochs=$EPOCHS lr=$LR neg=$NEG self_adv=$SELF_ADV"
echo ""

if [ ! -f data/WN18RR/train.txt ]; then
    echo "WN18RR data not found, downloading..."
    python3 scripts/download_wn18rr.py
fi

cargo build --release --example wn18rr_training 2>&1

echo "Training..."
cargo run --release --example wn18rr_training 2>&1 | tee "data/WN18RR/benchmark_dim${DIM}_ep${EPOCHS}.txt"

echo ""
echo "Done. Results: data/WN18RR/benchmark_dim${DIM}_ep${EPOCHS}.txt"
