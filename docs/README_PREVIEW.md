# Live Markdown Preview

Preview your markdown documentation with proper math rendering and **live reload** as you edit.

## Quick Start: All Docs Preview (Recommended)

**Preview all markdown files with GitHub-style rendering:**

```bash
./docs/preview-all.sh
```

Then open `http://localhost:3333` in your browser.

**Preview a specific file:**
```bash
./docs/preview-all.sh --file MATHEMATICAL_FOUNDATIONS.md
```

**Custom port:**
```bash
./docs/preview-all.sh 8000
```

**Options:**
```bash
./docs/preview-all.sh --help
```

### Features

- ✅ **GitHub-style rendering** - Uses GitHub API if `gh` CLI is available
- ✅ **KaTeX math rendering** - Falls back to local rendering with KaTeX
- ✅ **Auto-reload** - Refreshes every 2 seconds as you edit
- ✅ **File selector** - Switch between documents easily
- ✅ **Auto-open browser** - Opens automatically (use `--no-open` to disable)

## Alternative: npx Preview

**For Node.js-based preview:**

```bash
cd docs
npx --yes -p markdown-it -p markdown-it-katex node preview.mjs
```

## Alternative: Rust Docs Preview

**For Rust documentation with KaTeX:**

```bash
make docs-open
# or
./docs/rustdoc-preview.sh --open
```

See [`README_RUSTDOC.md`](README_RUSTDOC.md) for details.

## Comparison

| Tool | Best For | Math Rendering | Live Reload |
|------|----------|----------------|-------------|
| `preview-all.sh` | All markdown docs | ✅ KaTeX/GitHub | ✅ Yes |
| `preview.mjs` | Node.js users | ✅ KaTeX | ✅ Yes |
| `make docs-open` | Rust API docs | ✅ KaTeX | ⚠️ Manual refresh |
| GitHub | Final viewing | ✅ MathJax | ❌ No |

## Workflow

1. **Start preview:**
   ```bash
   ./docs/preview-all.sh
   ```

2. **Edit markdown files** in your editor

3. **View in browser** - Auto-reloads every 2 seconds

4. **Math renders automatically** with KaTeX

## Troubleshooting

### GitHub API not working?

The script falls back to local rendering with KaTeX. For GitHub-style rendering:
```bash
gh auth login
```

### Port already in use?

```bash
./docs/preview-all.sh 3000  # Use different port
```

### Math not rendering?

1. Check browser console for errors
2. Ensure CDN resources are loading
3. Check LaTeX syntax (use `\[...\]` for display math)

## See Also

- [`README_RUSTDOC.md`](README_RUSTDOC.md) - Rust documentation preview
- [`README_KATEX.md`](README_KATEX.md) - KaTeX setup details
- [`MATH_RENDERING_GUIDE.md`](MATH_RENDERING_GUIDE.md) - Complete rendering guide
