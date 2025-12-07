# Typst Math Documentation

This directory contains the Typst source files for the mathematical one-pagers. These documents are pre-rendered to PDF for optimal math typesetting.

## Quick Start

### Build All Documents

```bash
./docs/typst/build.sh
```

This generates PDFs in `docs/typst-output/pdf/` and HTML placeholders in `docs/typst-output/html/`.

### Preview a Single Document (with auto-reload)

```bash
./docs/typst/preview.sh gumbel-box-volume
```

This watches the Typst file and automatically rebuilds the PDF when you make changes.

### Preview All Documents

```bash
# Build all PDFs
./docs/typst/build.sh

# Open output directory
open docs/typst-output/pdf  # macOS
# or
xdg-open docs/typst-output/pdf  # Linux
```

## Documents

The following documents are available as Typst files, organized as a narrative story:

### Introduction
1. **00-introduction.typ** - Historical context and motivation: The embedding evolution from points to regions

### Core Mathematical Concepts
2. **subsumption.typ** - Geometric containment as logical subsumption
3. **gumbel-box-volume.typ** - Expected volume for Gumbel boxes
4. **containment-probability.typ** - First-order Taylor approximation
5. **gumbel-max-stability.typ** - Max-stability and algebraic closure
6. **log-sum-exp-intersection.typ** - Log-sum-exp and Gumbel intersection
7. **local-identifiability.typ** - The learning problem and solution

### Modern Applications and Future
8. **07-applications.typ** - Modern applications (2023-2025): RegD, TransBox, Concept2Box
9. **08-future.typ** - Future directions and open questions

The documents tell a complete story: from historical motivation through mathematical foundations to modern applications and future research directions.

## Workflow

1. **Edit Typst files** in `docs/typst/`
2. **Preview with watch mode**: `./docs/typst/preview.sh <filename>`
3. **Build all**: `./docs/typst/build.sh`
4. **View PDFs** in `docs/typst-output/pdf/`

## Typst Installation

If Typst is not installed:

```bash
# Using Cargo (recommended)
cargo install --git https://github.com/typst/typst typst-cli

# Or using Homebrew (macOS)
brew install typst
```

## Template

The `template.typ` file defines:
- Page layout and margins
- Font settings (Linux Libertine)
- Theorem, definition, proof, and example styling
- Math equation numbering

All documents import this template for consistent formatting.

## Integration with Main Documentation

The Markdown versions in `docs/` are kept for:
- GitHub rendering (README and overview docs)
- Quick web viewing
- Search and indexing

The Typst versions provide:
- Professional PDF output
- Superior math typesetting
- Print-ready formatting
- Consistent typography

Both formats are maintained in parallel, with Typst as the authoritative source for the mathematical one-pagers.

