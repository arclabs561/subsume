#!/bin/bash
# Generate rustdoc with KaTeX and open in browser
# Usage: ./rustdoc-preview.sh [--open]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HEADER_FILE="$SCRIPT_DIR/katex-header.html"

if [ ! -f "$HEADER_FILE" ]; then
    echo "Error: KaTeX header file not found at $HEADER_FILE"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "📚 Generating rustdoc with KaTeX support..."
echo "   Header file: $HEADER_FILE"

# Generate docs with KaTeX header
RUSTDOCFLAGS="--html-in-header $HEADER_FILE" cargo doc --no-deps

if [ "$1" = "--open" ] || [ "$1" = "-o" ]; then
    echo "🌐 Opening in browser..."
    # Try to open the main index
    if command -v open >/dev/null 2>&1; then
        # macOS
        open target/doc/subsume_core/index.html
    elif command -v xdg-open >/dev/null 2>&1; then
        # Linux
        xdg-open target/doc/subsume_core/index.html
    elif command -v start >/dev/null 2>&1; then
        # Windows
        start target/doc/subsume_core/index.html
    else
        echo "📖 Docs generated at: target/doc/subsume_core/index.html"
        echo "   Open this file in your browser"
    fi
else
    echo "📖 Docs generated at: target/doc/subsume_core/index.html"
    echo "   Run with --open to open automatically"
fi

echo ""
echo "💡 Tip: Edit Rust source files and run this script again to regenerate"
echo "   Or use: cargo doc --no-deps --open (with RUSTDOCFLAGS set)"

