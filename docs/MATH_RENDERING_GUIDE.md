# Mathematical Notation Rendering Guide

This guide explains how mathematical formulas are rendered in different contexts and the best practices for each.

## Current Setup

### Markdown Documentation Files

**Format:** LaTeX-style math using `\[...\]` for display and `\(...\)` for inline

**Example:**
```markdown
\[
\mathbb{E}[\text{Vol}(B)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
\]
```

**Rendering:**
- ✅ **GitHub**: Renders natively using MathJax (automatic)
- ✅ **VS Code**: Renders with Markdown Preview Enhanced or similar extensions
- ✅ **Documentation sites**: Can use KaTeX or MathJax
- ❌ **Plain text**: Shows as raw LaTeX

### Rust Documentation (rustdoc)

**Format:** Code blocks with Unicode symbols

**Example:**
```rust
/// ```text
/// E[Vol(B)] = 2β K₀(2e^(-(μᵧ - μₓ)/(2β)))
/// ```
```

**Rendering:**
- ✅ **rustdoc HTML**: Renders as monospace code (readable)
- ✅ **All contexts**: Always readable, no dependencies
- ❌ **No math rendering**: Plain text only

---

## Comparison: KaTeX vs Typst vs Current Approach

### KaTeX

**What it is:** JavaScript library for rendering LaTeX math in web browsers

**Pros:**
- ✅ Fast rendering (faster than MathJax)
- ✅ Lightweight (~100KB)
- ✅ Works in browsers
- ✅ Good for web-based documentation sites
- ✅ GitHub uses MathJax (similar, but KaTeX is faster)

**Cons:**
- ❌ Requires JavaScript
- ❌ Doesn't work in rustdoc
- ❌ Doesn't work in plain markdown viewers
- ❌ Need to set up for custom docs sites

**Best for:**
- Custom documentation websites (docs.rs-style)
- Web-based documentation viewers
- Interactive documentation

**Not needed for:**
- GitHub markdown (already has MathJax)
- Rust documentation (rustdoc doesn't support it)

### Typst

**What it is:** Modern typesetting system (alternative to LaTeX)

**Pros:**
- ✅ Modern, clean syntax
- ✅ Fast compilation
- ✅ Great for PDFs and full documents
- ✅ Better than LaTeX for some use cases

**Cons:**
- ❌ Not for inline math in markdown
- ❌ Requires separate compilation step
- ❌ Doesn't integrate with markdown/rustdoc
- ❌ Overkill for documentation

**Best for:**
- Writing papers
- Creating PDF documentation
- Full document typesetting

**Not suitable for:**
- Inline documentation
- GitHub markdown
- Rust documentation

### Current Approach (LaTeX in Markdown + Code Blocks in Rust)

**Pros:**
- ✅ Works everywhere (GitHub, VS Code, etc.)
- ✅ Standard LaTeX syntax (familiar to researchers)
- ✅ Rust docs are readable without dependencies
- ✅ No setup required
- ✅ Works offline

**Cons:**
- ❌ Rust docs don't render math (but code blocks are readable)
- ❌ Plain text viewers show raw LaTeX

---

## Recommendations

### For Markdown Documentation (Current - Keep It!)

**✅ Keep LaTeX syntax** - It works perfectly on GitHub and most markdown viewers.

**Why:**
- GitHub automatically renders `\[...\]` and `\(...\)` with MathJax
- VS Code and most editors support it
- Standard format familiar to researchers
- No additional setup needed

**Example (current, works great):**
```markdown
\[
P(B \subseteq A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(B)}
\]
```

### For Rust Documentation (Current - Optimal!)

**✅ Keep code blocks with Unicode** - This is the best approach for rustdoc.

**Why:**
- rustdoc doesn't support math rendering
- Code blocks render as readable monospace
- No dependencies or setup
- Works in all contexts

**Example (current, optimal):**
```rust
/// ```text
/// P(other ⊆ self) = Vol(self ∩ other) / Vol(other)
/// ```
```

### If Building a Custom Documentation Site

**✅ KaTeX is now set up!** See [`README_KATEX.md`](README_KATEX.md) for complete setup.

**Quick start:**
1. Use `preview.html` for interactive preview
2. Use `convert_to_html.sh` to convert markdown to HTML with KaTeX
3. See [`README_KATEX.md`](README_KATEX.md) for full instructions

**Why KaTeX:**
- ⚡ Faster rendering than MathJax
- 📦 Lighter weight (~100KB vs ~200KB+)
- ✅ Better error handling
- ✅ Server-side safe

**Setup:**
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

**But:** GitHub already uses MathJax automatically, so KaTeX is mainly for custom sites and local preview.

---

## Best Practice Summary

### ✅ Do This

1. **Markdown files**: Use LaTeX `\[...\]` syntax
   - Renders on GitHub automatically
   - Works in most editors
   - Standard format

2. **Rust docs**: Use code blocks with Unicode
   - Always readable
   - No dependencies
   - Works everywhere

3. **For PDFs**: Consider Typst if creating standalone PDF documentation
   - Better than LaTeX for modern documents
   - But not needed for inline docs

### ❌ Don't Do This

1. **Don't use KaTeX in markdown** - GitHub already has MathJax
2. **Don't use Typst for inline docs** - Wrong tool for the job
3. **Don't use LaTeX in rustdoc** - Won't render, use code blocks instead

---

## Current Status: Optimal! ✅

Your current setup is **already optimal**:

- ✅ Markdown uses LaTeX (renders on GitHub)
- ✅ Rust docs use code blocks (readable everywhere)
- ✅ No additional dependencies needed
- ✅ Works in all contexts

**No changes needed!** The current approach balances:
- Mathematical correctness (LaTeX in markdown)
- Universal readability (code blocks in Rust)
- Zero setup overhead
- Works everywhere

---

## If You Want to Enhance

### Option 1: Add KaTeX to Custom Docs Site (Optional)

Only if you build a custom documentation website (like docs.rs):

```bash
# Add to HTML template
npm install katex
```

But GitHub already handles this, so only for custom sites.

### Option 2: Generate PDF with Typst (For Papers)

If creating standalone PDF documentation:

```typst
#set page(margin: 2cm)
#set text(font: "Linux Libertine", size: 11pt)

= Mathematical Foundations

The expected volume is:
$ E[max(Y-X, 0)] = 2 beta K_0(2 e^(-(mu_y - mu_x)/(2 beta))) $
```

But this is for **separate PDF generation**, not inline docs.

---

## Conclusion

**Current approach is best!** 

- LaTeX in markdown → Renders on GitHub ✅
- Code blocks in Rust → Readable everywhere ✅
- No setup needed ✅
- Works offline ✅

**KaTeX**: Only needed for custom documentation sites (GitHub already has MathJax)

**Typst**: Only for standalone PDF generation (not for inline docs)

**Recommendation**: Keep current setup, it's optimal for your use case! 🎯

