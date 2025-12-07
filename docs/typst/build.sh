#!/bin/bash
# Build script for Typst math documentation
# Generates PDFs and HTML from Typst source files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$DOCS_DIR/typst-output"
PDF_DIR="$OUTPUT_DIR/pdf"
HTML_DIR="$OUTPUT_DIR/html"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building Typst documentation...${NC}"

# Create output directories
mkdir -p "$PDF_DIR" "$HTML_DIR"

# Check if typst is installed
if ! command -v typst &> /dev/null; then
    echo "Error: typst not found. Install with: cargo install --git https://github.com/typst/typst typst-cli"
    exit 1
fi

# List of Typst files to build (in narrative order)
FILES=(
    "00-introduction"
    "subsumption"
    "gumbel-box-volume"
    "containment-probability"
    "gumbel-max-stability"
    "log-sum-exp-intersection"
    "local-identifiability"
    "07-applications"
    "08-future"
)

# Build PDFs
echo -e "${GREEN}Generating PDFs...${NC}"
for file in "${FILES[@]}"; do
    echo "  Building $file.pdf..."
    typst compile "$SCRIPT_DIR/$file.typ" "$PDF_DIR/$file.pdf" 2>&1 | grep -v "^warning:" || true
done

# Build HTML (using typst's HTML export)
echo -e "${GREEN}Generating HTML...${NC}"
for file in "${FILES[@]}"; do
    echo "  Building $file.html..."
    # Typst doesn't have native HTML export, so we'll use a workaround
    # For now, create a simple HTML wrapper that links to PDF
    cat > "$HTML_DIR/$file.html" <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$(echo $file | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }
        .pdf-link {
            display: inline-block;
            padding: 10px 20px;
            background: #0366d6;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin: 20px 0;
        }
        .pdf-link:hover {
            background: #0256c2;
        }
    </style>
</head>
<body>
    <h1>$(echo $file | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')</h1>
    <p>This document is available as a PDF. Click below to view or download.</p>
    <a href="../pdf/$file.pdf" class="pdf-link">View PDF</a>
    <p><em>Note: HTML rendering from Typst is coming soon. For now, please use the PDF version.</em></p>
</body>
</html>
EOF
done

echo -e "${GREEN}Build complete!${NC}"
echo "  PDFs: $PDF_DIR"
echo "  HTML: $HTML_DIR"

