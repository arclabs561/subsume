default:
    @just --list

# Prefer nextest if installed; fall back to cargo test.
test:
    if command -v cargo-nextest >/dev/null 2>&1; then cargo nextest run; else cargo test; fi

check:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    just test

docs:
    RUSTDOCFLAGS="--html-in-header docs/katex-header.html" cargo doc --no-deps
    @echo "Docs generated at: target/doc/subsume_core/index.html"

docs-open: docs
    if command -v open >/dev/null 2>&1; then open target/doc/subsume_core/index.html; \
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open target/doc/subsume_core/index.html; \
    elif command -v start >/dev/null 2>&1; then start target/doc/subsume_core/index.html; \
    else echo "Open: target/doc/subsume_core/index.html"; fi

docs-watch:
    cargo watch -x "doc --no-deps --html-in-header docs/katex-header.html"

typst-docs:
    ./docs/typst/build.sh

typst-preview FILE:
    if [[ -z "{{FILE}}" ]]; then echo "Usage: just typst-preview <filename>"; exit 1; fi
    ./docs/typst/preview.sh "{{FILE}}"

