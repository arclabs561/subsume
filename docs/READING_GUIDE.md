# Documentation Reading Guide

This guide provides recommended reading orders for different audiences and use cases.

## Quick Start (New Users)

**Goal:** Get up and running quickly with box embeddings.

1. **[`README.md`](../README.md)** - Project overview, installation, basic usage
2. **[`CONCEPTUAL_OVERVIEW.md`](CONCEPTUAL_OVERVIEW.md)** - Why box embeddings? High-level intuition
3. **[`PRACTICAL_GUIDE.md`](PRACTICAL_GUIDE.md)** - How to use the library in practice

**Time:** ~15-20 minutes

---https://github.com/arclabs561/mcp-axum/settingshttps://github.com/arclabs561/mcp-axum/settingshttps://github.com/arclabs561/mcp-axum/settings

## Understanding the Math (Researchers/Students)

**Goal:** Deep understanding of mathematical foundations.

### Beginner Path (No prior knowledge of box embeddings)

1. **[`CONCEPTUAL_OVERVIEW.md`](CONCEPTUAL_OVERVIEW.md)** - Intuitive explanation of the approach
2. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Guide to mathematical foundations
3. Read mathematical foundations in order (prefer PDFs for best typesetting):
   - **[PDF: Introduction](typst-output/pdf/00-introduction.pdf)** - Historical context and motivation
   - **[PDF: Subsumption](typst-output/pdf/subsumption.pdf)** | [Markdown: Subsumption](SUBSUMPTION.md) - Geometric containment as logical subsumption
   - **[PDF: Gumbel Box Volume](typst-output/pdf/gumbel-box-volume.pdf)** | [Markdown: Gumbel Box Volume](GUMBEL_BOX_VOLUME.md) - Expected volume for Gumbel boxes
   - **[PDF: Containment Probability](typst-output/pdf/containment-probability.pdf)** | [Markdown: Containment Probability](CONTAINMENT_PROBABILITY.md) - Containment probability approximation
   - **[PDF: Gumbel Max-Stability](typst-output/pdf/gumbel-max-stability.pdf)** | [Markdown: Gumbel Max-Stability](GUMBEL_MAX_STABILITY.md) - Max-stability and algebraic closure
   - **[PDF: Log-Sum-Exp](typst-output/pdf/log-sum-exp-intersection.pdf)** | [Markdown: Log-Sum-Exp](LOG_SUM_EXP_INTERSECTION.md) - Log-sum-exp and intersection
   - **[PDF: Local Identifiability](typst-output/pdf/local-identifiability.pdf)** | [Markdown: Local Identifiability](LOCAL_IDENTIFIABILITY.md) - The learning problem and solution
   - **[PDF: Applications](typst-output/pdf/07-applications.pdf)** - Modern applications (2023-2025)
   - **[PDF: Future Directions](typst-output/pdf/08-future.pdf)** - Open questions and future work
4. **[`MATH_QUICK_REFERENCE.md`](MATH_QUICK_REFERENCE.md)** - Quick lookup for formulas

**Time:** ~1-2 hours

### Advanced Path (Familiar with embeddings, want deep theory)

1. All one-pagers from beginner path (for completeness)
2. **[`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md)** - See how theory maps to code
3. Original papers referenced in docs

**Time:** ~2-3 hours

---

## Implementation Focus (Developers)

**Goal:** Understand how to implement or extend the codebase.

1. **[`CONCEPTUAL_OVERVIEW.md`](CONCEPTUAL_OVERVIEW.md)** - Understand the problem domain
2. **[`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md)** - Theory-to-code mappings
3. **Rust Documentation** (`cargo doc --open`):
   - `subsume_core::Box` trait
   - `subsume_core::GumbelBox` trait
   - `subsume_core::utils` module
4. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Reference as needed for formulas
5. **[`REMAINING_WORK.md`](REMAINING_WORK.md)** - See what's planned/needed

**Time:** ~1-2 hours

---

## Research/Paper Writing (Academics)

**Goal:** Understand state-of-the-art and recent developments.

1. **[`CONCEPTUAL_OVERVIEW.md`](CONCEPTUAL_OVERVIEW.md)** - Quick refresher
2. **[`RECENT_RESEARCH.md`](RECENT_RESEARCH.md)** - Latest papers (2023-2025)
3. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Theoretical foundations (one-pagers)
4. **[`REMAINING_WORK.md`](REMAINING_WORK.md)** - Open questions and future work

**Time:** ~2-3 hours

---

## Troubleshooting/Deep Dive (Debugging Issues)

**Goal:** Understand specific concepts or fix problems.

1. **[`MATH_QUICK_REFERENCE.md`](MATH_QUICK_REFERENCE.md)** - Find the relevant formula
2. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Navigate to specific one-pager
3. **[`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md)** - See implementation details
5. **Rust Documentation** - Check function docs for numerical stability notes

**Time:** ~30 minutes per concept

---

## Complete Understanding (Comprehensive)

**Goal:** Master all aspects of the codebase and theory.

### Phase 1: Concepts (Week 1)
1. [`README.md`](../README.md)
2. [`CONCEPTUAL_OVERVIEW.md`](CONCEPTUAL_OVERVIEW.md)
3. [`PRACTICAL_GUIDE.md`](PRACTICAL_GUIDE.md)

### Phase 2: Mathematics (Week 2)
1. [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md) - Guide to one-pagers
2. All one-pagers (read in order from foundations guide)
3. [`MATH_QUICK_REFERENCE.md`](MATH_QUICK_REFERENCE.md) - Memorize key formulas

### Phase 3: Implementation (Week 3)
1. [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md)
2. **Rust Documentation** - All modules
3. Code examples in `examples/` directories

### Phase 4: Research Context (Week 4)
1. [`RECENT_RESEARCH.md`](RECENT_RESEARCH.md)
2. [`REMAINING_WORK.md`](REMAINING_WORK.md)
3. Original papers referenced in docs

**Time:** ~4 weeks (part-time)

---

## By Topic

### Understanding Subsumption
1. **[PDF](typst-output/pdf/subsumption.pdf)** | [Markdown](SUBSUMPTION.md) - Geometric containment as logical subsumption
2. [`CONCEPTUAL_OVERVIEW.md`](CONCEPTUAL_OVERVIEW.md) - "The Subsumption Connection"

### Understanding Gumbel Distributions
1. **[PDF](typst-output/pdf/gumbel-max-stability.pdf)** | [Markdown](GUMBEL_MAX_STABILITY.md) - Max-stability and min-stability
2. [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md) - "Gumbel Distribution and Max-Stability"

### Understanding Volume Calculations
1. **[PDF](typst-output/pdf/gumbel-box-volume.pdf)** | [Markdown](GUMBEL_BOX_VOLUME.md) - Expected volume with Bessel function
2. [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md) - "Bessel Function and Volume Calculation"

### Understanding Numerical Stability
1. [`LOG_SUM_EXP_INTERSECTION.md`](LOG_SUM_EXP_INTERSECTION.md) - Stable log-sum-exp computation
2. [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md) - "Numerical Stability Patterns"
3. **Rust Documentation** - `subsume_core::utils` module

---

## Document Dependencies

```
README.md
  └─> CONCEPTUAL_OVERVIEW.md
       └─> MATHEMATICAL_FOUNDATIONS.md
            ├─> MATH_QUICK_REFERENCE.md (reference)
            ├─> MATH_TO_CODE_CONNECTIONS.md
            └─> Typst PDFs in typst-output/pdf/ (detailed derivations)
```

**Legend:**
- **→** Read before
- **↗** Reference as needed
- **↓** Read after

---

## Quick Reference

**Just need a formula?** → [`MATH_QUICK_REFERENCE.md`](MATH_QUICK_REFERENCE.md)

**Want to understand why?** → [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)

**Need to implement?** → [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md)

**Looking for a proof?** → See the one-pagers in [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)

**Troubleshooting?** → Rust docs (`cargo doc`) + [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md)

---

## Tips

1. **Start with concepts** - Understanding the "why" makes the "how" easier
2. **Use quick reference** - Don't memorize, reference as needed
3. **Read code alongside docs** - Theory + implementation = understanding
4. **Follow links** - Documentation is cross-referenced for a reason
5. **Work through examples** - See [`REAL_TRAINING_EXAMPLES.md`](REAL_TRAINING_EXAMPLES.md) for practical usage

---

## Feedback

If you find a better reading order for your use case, consider updating this guide!

