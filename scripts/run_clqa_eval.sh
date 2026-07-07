#!/usr/bin/env bash
# Run the data-gated GALEN CLQA evaluation with stable metrics output.

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
    printf 'Usage: %s [box|full|symbolic|learned]\n' "${0##*/}"
    printf '\n'
    printf 'Environment:\n'
    printf '  DATASET       Box2EL dataset name under data/ (default: GALEN)\n'
    printf '  BACKEND       wgpu or ndarray (default: wgpu)\n'
    printf '  METRICS_CSV   output CSV path (default: target/clqa-eval/<dataset>-<mode>-<backend>.csv)\n'
    printf '  DIM           embedding dimension passed to the example (default: example default)\n'
    printf '  EPOCHS        training epochs passed to the example (default: example default)\n'
    printf '  QUERIES       query count passed to the example (default: example default)\n'
    printf '  LEARNED_*     learned-ranker knobs forwarded to the example\n'
}

mode="${1:-box}"
case "$mode" in
    -h|--help)
        usage
        exit 0
        ;;
    box|full|symbolic|learned)
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac

dataset="${DATASET:-GALEN}"
backend="${BACKEND:-wgpu}"
case "$backend" in
    wgpu)
        features='burn-wgpu,kge'
        ;;
    ndarray)
        features='burn-ndarray,kge'
        ;;
    *)
        printf 'unknown BACKEND=%s (expected wgpu or ndarray)\n' "$backend" >&2
        exit 1
        ;;
esac

metrics_csv="${METRICS_CSV:-target/clqa-eval/${dataset}-${mode}-${backend}.csv}"
mkdir -p "$(dirname "$metrics_csv")"

printf '==> CLQA eval: dataset=%s mode=%s backend=%s metrics=%s\n' \
    "$dataset" "$mode" "$backend" "$metrics_csv"

env_args=(
    "DATASET=$dataset"
    "METRICS_CSV=$metrics_csv"
)

case "$mode" in
    box)
        env_args+=("SKIP_TRANSE=1")
        ;;
    full)
        ;;
    symbolic)
        env_args+=("SYMBOLIC_ONLY=1")
        ;;
    learned)
        env_args+=("SYMBOLIC_ONLY=1" "LEARNED_RETRIEVAL=1")
        ;;
esac

for var in DIM EPOCHS QUERIES BATCH LR OFFSET_CLAMP TIGHTNESS LEARNED_EXTRA_HOPS LEARNED_EPOCHS LEARNED_LR LEARNED_L2 LEARNED_REPEATS LEARNED_SPLIT_SEED LEARNED_CASE_LIMIT; do
    if [[ ${!var+x} ]]; then
        env_args+=("$var=${!var}")
    fi
done

env "${env_args[@]}" cargo run --release --no-default-features \
    --features "$features" \
    --example el_clqa_galen

if [[ -f "$metrics_csv" ]]; then
    printf '==> metrics: %s\n' "$metrics_csv"
else
    printf '==> no metrics written: %s\n' "$metrics_csv"
fi
