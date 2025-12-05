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
2. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Read sections in order:
   - Subsumption: The Core Logical Concept
   - Volume Calculation Methods (Hard, Soft, Gumbel-Box)
   - Containment Probability
   - Intersection Operations
   - Probabilistic Interpretation
3. **[`MATH_QUICK_REFERENCE.md`](MATH_QUICK_REFERENCE.md)** - Quick lookup for formulas
4. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Advanced sections:
   - Local Identifiability Problem
   - Theoretical Guarantees
   - Gumbel-Softmax Framework
   - Training Dynamics

**Time:** ~2-3 hours

### Advanced Path (Familiar with embeddings, want deep theory)

1. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Complete read
2. **[`MATH_EXPLANATION_GAPS.md`](MATH_EXPLANATION_GAPS.md)** - Detailed analysis of concepts
3. **[`MATH_DERIVATION_INDEX.md`](MATH_DERIVATION_INDEX.md)** - Navigate to specific derivations
4. **[`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md)** - See how theory maps to code

**Time:** ~3-4 hours

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
3. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Theoretical foundations
4. **[`MATH_IMPROVEMENTS_SUMMARY.md`](MATH_IMPROVEMENTS_SUMMARY.md)** - What's been enhanced
5. **[`REMAINING_WORK.md`](REMAINING_WORK.md)** - Open questions and future work

**Time:** ~2-3 hours

---

## Troubleshooting/Deep Dive (Debugging Issues)

**Goal:** Understand specific concepts or fix problems.

1. **[`MATH_QUICK_REFERENCE.md`](MATH_QUICK_REFERENCE.md)** - Find the relevant formula
2. **[`MATH_DERIVATION_INDEX.md`](MATH_DERIVATION_INDEX.md)** - Navigate to specific derivation
3. **[`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md)** - Read the relevant section
4. **[`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md)** - See implementation details
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
1. [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md) - Full read
2. [`MATH_QUICK_REFERENCE.md`](MATH_QUICK_REFERENCE.md) - Memorize key formulas
3. [`MATH_EXPLANATION_GAPS.md`](MATH_EXPLANATION_GAPS.md) - Deep analysis
4. [`MATH_DERIVATION_INDEX.md`](MATH_DERIVATION_INDEX.md) - Work through derivations

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
1. [`CONCEPTUAL_OVERVIEW.md`](CONCEPTUAL_OVERVIEW.md) - "The Subsumption Connection"
2. [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md) - "Subsumption: The Core Logical Concept"

### Understanding Gumbel Distributions
1. [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md) - "Gumbel-Softmax Framework"
2. [`MATH_EXPLANATION_GAPS.md`](MATH_EXPLANATION_GAPS.md) - "Gumbel Distribution Max-Stability"
3. [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md) - "Gumbel Distribution and Max-Stability"

### Understanding Volume Calculations
1. [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md) - "Volume Calculation Methods"
2. [`MATH_EXPLANATION_GAPS.md`](MATH_EXPLANATION_GAPS.md) - "Modified Bessel Function K₀"
3. [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md) - "Bessel Function and Volume Calculation"

### Understanding Numerical Stability
1. [`MATH_TO_CODE_CONNECTIONS.md`](MATH_TO_CODE_CONNECTIONS.md) - "Numerical Stability Patterns"
2. **Rust Documentation** - `subsume_core::utils` module
3. [`MATHEMATICAL_FOUNDATIONS.md`](MATHEMATICAL_FOUNDATIONS.md) - "Log-Sum-Exp Function" section

---

## Document Dependencies

```
README.md
  └─> CONCEPTUAL_OVERVIEW.md
       └─> MATHEMATICAL_FOUNDATIONS.md
            ├─> MATH_QUICK_REFERENCE.md (reference)
            ├─> MATH_TO_CODE_CONNECTIONS.md
            ├─> MATH_EXPLANATION_GAPS.md (deep dive)
            └─> MATH_DERIVATION_INDEX.md (navigation)
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

**Looking for a proof?** → [`MATH_DERIVATION_INDEX.md`](MATH_DERIVATION_INDEX.md)

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

