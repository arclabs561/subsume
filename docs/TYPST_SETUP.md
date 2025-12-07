# Typst Setup for Math Documentation

The mathematical one-pagers are written in Typst for superior math typesetting and PDF output, while maintaining Markdown versions for web viewing.

## Quick Start

### Build All PDFs

```bash
make typst-docs
# or
./docs/typst/build.sh
```

### Preview with Auto-Reload

```bash
make typst-preview FILE=gumbel-box-volume
# or
./docs/typst/preview.sh gumbel-box-volume
```

## Installation

If Typst is not installed:

```bash
# Using Cargo (recommended)
cargo install --git https://github.com/typst/typst typst-cli

# Or using Homebrew (macOS)
brew install typst
```

## Structure

- **Source files**: `docs/typst/*.typ` - Typst source files
- **Output PDFs**: `docs/typst-output/pdf/*.pdf` - Generated PDFs
- **Markdown versions**: `docs/*.md` - Web-friendly versions (kept for GitHub)

## Workflow

1. **Edit Typst files** in `docs/typst/`
2. **Preview**: `make typst-preview FILE=<name>` (auto-reloads on changes)
3. **Build all**: `make typst-docs`
4. **View PDFs**: Open `docs/typst-output/pdf/`

## Available Documents

- `gumbel-box-volume` - Expected volume for Gumbel boxes
- `containment-probability` - First-order Taylor approximation
- `subsumption` - Geometric containment as logical subsumption
- `gumbel-max-stability` - Max-stability and algebraic closure
- `log-sum-exp-intersection` - Log-sum-exp and Gumbel intersection
- `local-identifiability` - The learning problem and solution

## Why Typst?

- **Superior math typesetting**: Native support for mathematical notation
- **Professional PDFs**: Print-ready output with consistent typography
- **Modern typesetting**: Clean syntax, fast compilation
- **Maintainable**: Source files are readable and version-controllable

The Markdown versions remain for:
- GitHub rendering
- Quick web viewing
- Search and indexing

Both formats are maintained in parallel.

