# KaTeX Setup for Documentation

This directory contains tools for previewing markdown documentation with proper KaTeX math rendering.

## Quick Start: npx Preview (Recommended)

**Best option - no installation needed!**

```bash
cd docs
npx --yes -p markdown-it -p markdown-it-katex node preview.mjs
```

Then open `http://localhost:8000` in your browser. The preview auto-reloads every 2 seconds as you edit!

**Preview a specific file:**
```bash
npx --yes -p markdown-it -p markdown-it-katex node preview.mjs MATHEMATICAL_FOUNDATIONS.md
```

See [`README_PREVIEW.md`](README_PREVIEW.md) for complete preview guide.

## Alternative: Python Server

If you prefer Python:

```bash
cd docs
python3 preview-server.py
```

## Alternative: Static Preview

If you prefer a static preview without auto-reload:

1. Start a local web server:
   ```bash
   cd docs
   python3 -m http.server 8000
   ```

2. Open `http://localhost:8000/preview.html` in your browser

3. Select a document from the dropdown

## KaTeX Configuration

The KaTeX setup uses the following delimiters:

- `\[...\]` - Display math (block)
- `\(...\)` - Inline math
- `$$...$$` - Display math (alternative)
- `$...$` - Inline math (alternative)

### Features

- ✅ Fast rendering (KaTeX is faster than MathJax)
- ✅ Server-side safe (no eval)
- ✅ Automatic rendering of all math in document
- ✅ Proper spacing and alignment
- ✅ Error handling (won't break on invalid LaTeX)

## Comparison: KaTeX vs MathJax

| Feature | KaTeX | MathJax |
|---------|-------|---------|
| Speed | ⚡ Very fast | 🐢 Slower |
| Size | 📦 ~100KB | 📦 ~200KB+ |
| GitHub | ❌ Uses MathJax | ✅ Native |
| Custom sites | ✅ Recommended | ✅ Works |
| Error handling | ✅ Graceful | ✅ Graceful |

**Recommendation:** Use KaTeX for custom documentation sites, GitHub's MathJax is fine for GitHub viewing.

## Integration with Documentation

### For GitHub

GitHub automatically renders LaTeX math using MathJax. No setup needed - just use standard LaTeX syntax:

```markdown
\[
\mathbb{E}[\text{Vol}(B)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
\]
```

### For Local Preview

Use `preview.html` or convert to HTML with `convert_to_html.sh` for KaTeX rendering.

### For Custom Documentation Sites

Include KaTeX in your HTML template:

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "\\[", right: "\\]", display: true},
        {left: "\\(", right: "\\)", display: false}
      ]
    });
  });
</script>
```

## Troubleshooting

### Math not rendering?

1. **Check delimiters**: Make sure you're using `\[...\]` or `\(...\)`
2. **Check browser console**: Look for JavaScript errors
3. **Check network**: Ensure CDN resources are loading
4. **Try preview.html**: Use the interactive preview to test

### CORS errors?

The preview requires a local web server. Use:
```bash
python3 -m http.server 8000
# or
npx serve
```

### KaTeX errors in console?

KaTeX is configured with `throwOnError: false`, so it won't break the page. Check the console for specific LaTeX syntax issues.

## Advanced Usage

### Custom KaTeX Options

Edit `preview.html` or `convert_to_html.sh` to customize:

```javascript
const katexOptions = {
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
        "\\EE": "\\mathbb{E}"
    }
};
```

### Batch Conversion

Convert all markdown files:

```bash
for file in *.md; do
    if [ "$file" != "README_KATEX.md" ]; then
        ./convert_to_html.sh "$file" "${file%.md}.html"
    fi
done
```

## See Also

- [KaTeX Documentation](https://katex.org/docs/options.html)
- [KaTeX Supported Functions](https://katex.org/docs/supported.html)
- [`MATH_RENDERING_GUIDE.md`](MATH_RENDERING_GUIDE.md) - Complete rendering guide

