# Rust Documentation with KaTeX Math Rendering

This guide explains how to generate and view Rust documentation with proper KaTeX math rendering.

## Quick Start

### Option 1: Using Make (Recommended)

```bash
# Generate docs with KaTeX and open in browser
make docs-open
```

### Option 2: Using the Preview Script

```bash
# Generate docs with KaTeX and open in browser
./docs/rustdoc-preview.sh --open
```

### Option 3: Using Cargo Directly

```bash
# Set environment variable and generate docs
RUSTDOCFLAGS="--html-in-header docs/katex-header.html" cargo doc --no-deps --open
```

### Option 4: Using .cargo/config.toml (Automatic)

The `.cargo/config.toml` file is configured to automatically include KaTeX. Just run:

```bash
cargo doc --no-deps --open
```

## How It Works

1. **KaTeX Header**: `docs/katex-header.html` includes KaTeX CSS and JavaScript
2. **Auto-render**: KaTeX automatically finds and renders all LaTeX math in doc comments
3. **Dynamic Content**: Watches for dynamically loaded content (rustdoc uses JS to load pages)

## Math Syntax in Rust Docs

Use LaTeX syntax in your Rust doc comments:

```rust
/// Compute the expected volume using the Bessel approximation.
///
/// For Gumbel boxes with `X ~ MinGumbel(μₓ, β)` and `Y ~ MaxGumbel(μᵧ, β)`:
///
/// ```text
/// E[max(Y - X, 0)] = 2β K₀(2e^(-(μᵧ - μₓ)/(2β)))
/// ```
///
/// Or use LaTeX for better rendering:
///
/// \[
/// \mathbb{E}[\max(Y-X, 0)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
/// \]
pub fn volume() -> f32 {
    // ...
}
```

## Supported Math Delimiters

- `\[...\]` - Display math (block, centered)
- `\(...\)` - Inline math
- `$$...$$` - Display math (alternative)
- `$...$` - Inline math (alternative)

## Live Preview Workflow

1. **Edit Rust source files** with math in doc comments
2. **Regenerate docs:**
   ```bash
   make docs-open
   # or
   ./docs/rustdoc-preview.sh --open
   ```
3. **View in browser** - Math renders automatically with KaTeX
4. **Refresh browser** (Cmd+R / Ctrl+R) after regenerating to see changes

### Watch Mode (Auto-regenerate)

Install `cargo-watch` for automatic regeneration:
```bash
cargo install cargo-watch
make docs-watch
```

This will automatically regenerate docs when you save Rust files.

## Configuration

### Custom Header File

Edit `docs/katex-header.html` to customize:
- KaTeX options (delimiters, error handling)
- Custom CSS styling
- Additional JavaScript

### Environment Variable

You can also set `RUSTDOCFLAGS` in your shell:

```bash
export RUSTDOCFLAGS="--html-in-header docs/katex-header.html"
cargo doc --no-deps --open
```

## Troubleshooting

### Math not rendering?

1. **Check browser console** for JavaScript errors
2. **Verify KaTeX is loading** - Check Network tab for CDN resources
3. **Check LaTeX syntax** - Use `\[...\]` for display math
4. **Clear browser cache** - Sometimes cached docs don't include new header

### Docs not updating?

1. **Regenerate docs** - Run `cargo doc` again
2. **Hard refresh browser** - Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows/Linux)
3. **Check header file** - Ensure `docs/katex-header.html` exists

### Port conflicts?

If `cargo doc --open` fails to open:
```bash
# Generate without opening
cargo doc --no-deps

# Then manually open
open target/doc/subsume_core/index.html  # macOS
xdg-open target/doc/subsume_core/index.html  # Linux
```

## Comparison: Rust Docs vs Markdown

| Feature | Rust Docs (rustdoc) | Markdown Docs |
|---------|---------------------|---------------|
| Math rendering | ✅ KaTeX (with setup) | ✅ MathJax (GitHub) |
| Live preview | ✅ Browser refresh | ✅ Auto-reload tools |
| Code examples | ✅ Tested examples | ❌ Not tested |
| API reference | ✅ Automatic | ❌ Manual |
| Cross-references | ✅ Automatic links | ⚠️ Manual links |
| Best for | ✅ API documentation | ✅ Conceptual docs |

**Recommendation:**
- **Rust docs**: For API reference, function docs, code examples
- **Markdown docs**: For conceptual overviews, tutorials, research papers

## Advanced: Custom KaTeX Options

Edit `docs/katex-header.html` to customize KaTeX:

```javascript
renderMathInElement(document.body, {
    delimiters: [
        {left: "\\[", right: "\\]", display: true},
        {left: "\\(", right: "\\)", display: false}
    ],
    throwOnError: false,
    strict: false,
    trust: true,
    // Add custom macros
    macros: {
        "\\RR": "\\mathbb{R}",
        "\\EE": "\\mathbb{E}",
        "\\Vol": "\\text{Vol}"
    }
});
```

## See Also

- [`README_PREVIEW.md`](README_PREVIEW.md) - Markdown preview tools
- [`README_KATEX.md`](README_KATEX.md) - KaTeX setup details
- [Rustdoc Book](https://doc.rust-lang.org/rustdoc/) - Official rustdoc documentation

