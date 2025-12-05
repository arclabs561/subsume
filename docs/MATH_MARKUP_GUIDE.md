# Math Markup Guide for Rust Documentation

This guide explains how to write mathematical notation in Rust documentation that renders correctly with KaTeX while remaining readable in plain text.

## Best Practices (2024-2025)

Based on current rustdoc capabilities and KaTeX integration:

### ✅ Recommended: LaTeX with KaTeX

**Display math** (centered, larger):
```rust
/// \[
/// \mathbb{E}[\text{Vol}(B)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
/// \]
```

**Inline math** (within text):
```rust
/// For temperature \(\tau\), the probability is computed using...
```

### Delimiters Supported

Our KaTeX setup (via `katex-header.html`) supports:
- **Display**: `\[...\]` or `$$...$$`
- **Inline**: `\(...\)` or `$...$`

**Recommendation**: Use `\[...\]` and `\(...\)` for consistency with LaTeX conventions.

### Graceful Degradation

Since rustdoc doesn't natively render math, write formulas that are readable even without rendering:

**Good** (readable in plain text):
```rust
/// The volume is computed as: Vol(B) = ∏ᵢ max(Z_i - z_i, 0)
/// 
/// Or in LaTeX:
/// \[
/// \text{Vol}(B) = \prod_{i=1}^{d} \max(Z_i - z_i, 0)
/// \]
```

**Avoid** (unreadable without rendering):
```rust
/// The formula is: \[ \prod_{i=1}^{d} \max(Z_i - z_i, 0) \]
/// // Without KaTeX, this shows raw LaTeX which is hard to read
```

## Gardner-Style Explanations

Martin Gardner's approach to mathematical explanation emphasizes:

1. **Intuitive explanations first** - Build understanding before formal math
2. **Concrete examples** - Use real-world analogies
3. **Step-by-step reasoning** - Show the "why" not just the "what"
4. **Friendly tone** - Approachable, not intimidating

### Example Structure

```rust
/// # Intuitive Explanation
///
/// Think of this as [concrete analogy]. Imagine [specific example].
/// This works because [intuitive reason].
///
/// # Mathematical Formulation
///
/// \[
/// \text{formal formula here}
/// \]
///
/// where:
/// - \(x\) is [explanation]
/// - \(y\) is [explanation]
```

### Real Example from Codebase

```rust
/// # Intuitive Explanation
///
/// Traditional Euclidean distance treats all boxes equally, regardless of their size.
/// But in hierarchical structures (like taxonomies), larger boxes (more general concepts)
/// should be "farther" from smaller boxes (more specific concepts), even if their centers
/// are close. This is similar to how hyperbolic space naturally models hierarchies.
///
/// **The crowding problem**: In Euclidean space, as you add more children to a parent node,
/// they all cluster near the parent. Depth distance solves this by incorporating volume:
/// boxes with very different sizes are pushed apart, even if their centers are close.
///
/// # Mathematical Formulation
///
/// \[
/// d_{\text{depth}}(A, B) = d_{\text{Euclidean}}(A, B) + \alpha \cdot |\log(\text{Vol}(A)) - \log(\text{Vol}(B))|
/// \]
```

## Coverage Status

### ✅ Files with Gardner-Style Math Explanations

- `box_trait.rs` - Core box operations with intuitive explanations
- `gumbel.rs` - Gumbel distributions with step-by-step derivations
- `utils.rs` - Numerical stability with intuitive explanations
- `distance.rs` - Distance metrics with concrete examples
- `boxe.rs` - BoxE model with intuitive geometric explanations
- `center_offset.rs` - Representation conversion with clear reasoning

### 📝 Files Needing More Math Explanations

- `embedding.rs` - Collection operations (mostly data structures, less math)
- `trainer.rs` - Training utilities (could add more intuitive loss explanations)
- `training.rs` - Training metrics (evaluation formulas could use more explanation)
- `dataset.rs` - Data loading (minimal math, mostly I/O)

## Common Patterns

### Probability Notation

```rust
/// The containment probability \(P(B \subseteq A)\) is computed as:
/// \[
/// P(B \subseteq A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(B)}
/// \]
```

### Expectation Values

```rust
/// The expected volume \(\mathbb{E}[\text{Vol}(B)]\) for Gumbel boxes is:
/// \[
/// \mathbb{E}[\text{Vol}(B)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
/// \]
```

### Sets and Containment

```rust
/// Box \(A\) contains box \(B\) if \(B \subseteq A\), meaning:
/// \[
/// \forall i: \min_i^A \leq \min_i^B \text{ and } \max_i^B \leq \max_i^A
/// \]
```

### Limits and Asymptotics

```rust
/// As temperature decreases (\(\tau \to 0\)), the probability becomes sharper
/// (more like hard membership). As temperature increases (\(\tau \to \infty\)), the
/// probability becomes smoother (approaches uniform).
```

## Testing Math Rendering

1. **Generate docs**: `RUSTDOCFLAGS="--html-in-header docs/katex-header.html" cargo doc --no-deps`
2. **Open in browser**: `open target/doc/subsume_core/index.html`
3. **Check rendering**: Math should render with proper formatting
4. **Check fallback**: View source to ensure LaTeX is readable even if KaTeX fails

## References

- [KaTeX Supported Functions](https://katex.org/docs/supported.html)
- [LaTeX Math Symbols](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)
- [Martin Gardner's Writing Style](https://www.scientificamerican.com/article/martin-gardner-hofstadter/) - Inspiration for intuitive explanations

