#!/bin/bash
# Preview Typst documents with auto-reload
# Uses typst watch to automatically rebuild on changes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$DOCS_DIR/typst-output"
PDF_DIR="$OUTPUT_DIR/pdf"

# Check if typst is installed
if ! command -v typst &> /dev/null; then
    echo "Error: typst not found. Install with: cargo install --git https://github.com/typst/typst typst-cli"
    exit 1
fi

# Create output directory
mkdir -p "$PDF_DIR"

# Get file to preview (default: first file)
FILE="${1:-gumbel-box-volume}"

if [ ! -f "$SCRIPT_DIR/$FILE.typ" ]; then
    echo "Error: File $FILE.typ not found in $SCRIPT_DIR"
    echo "Available files:"
    ls -1 "$SCRIPT_DIR"/*.typ | xargs -n1 basename | sed 's/\.typ$//' | grep -v template
    exit 1
fi

echo "Watching $FILE.typ for changes..."
echo "Output: $PDF_DIR/$FILE.pdf"
echo ""
echo "Press Ctrl+C to stop"

# Watch and compile
typst watch "$SCRIPT_DIR/$FILE.typ" "$PDF_DIR/$FILE.pdf"

