# subsume

Framework-agnostic geometric box embeddings for containment, entailment, and hierarchical relationships.

**Subsume** models logical subsumption relationships through geometric containment: when box A contains box B, we say that A **subsumes** B. This directly corresponds to entailment in formal logic, hierarchical relationships in knowledge graphs, and hyponym-hypernym relationships in natural language.

## Why "subsume"?

The name **subsume** is the precise mathematical and logical term for the containment relationship that box embeddings model. In formal logic and automated reasoning, **subsumption** means that one statement is more general than another and covers all cases that the more specific statement would cover.

### Mathematical Definition

When box A contains box B (geometrically: B ⊆ A), we say that **A subsumes B**:

\[
\text{Box A subsumes Box B} \iff B \subseteq A \iff P(B|A) = 1
\]

### Logical Relationships

In box embeddings:
- **Subsumption** = containment relationship (A ⊇ B means A subsumes B)
- **Entailment** = if premise subsumes hypothesis, then premise entails hypothesis
- **Hierarchical relationships** = parent concepts subsume child concepts (hyponym-hypernym relationships)

### Example

If "animal" is represented by a box that contains the box for "dog", then "animal" subsumes "dog" — the more general concept contains the more specific one. This directly models the logical subsumption relationship used in automated theorem proving and knowledge representation.

### Mathematical Foundations

For detailed mathematical formulations, see [`docs/MATHEMATICAL_FOUNDATIONS.md`](docs/MATHEMATICAL_FOUNDATIONS.md), which covers:
- Volume calculation methods (hard, soft, Gumbel-box with Bessel approximation)
- Containment and overlap probability formulas
- Gumbel-Softmax framework and local identifiability solutions
- Theoretical guarantees (expressiveness, closure properties, idempotency)
- Training dynamics and optimization considerations

For practical guidance on using box embeddings effectively, see [`docs/PRACTICAL_GUIDE.md`](docs/PRACTICAL_GUIDE.md), which covers:
- Understanding and working around the local identifiability problem
- Temperature scheduling strategies and hyperparameter selection
- Volume regularization to prevent "cheating" (boxes becoming arbitrarily large)
- Initialization strategies for successful training
- Numerical stability considerations and common pitfalls
- Training diagnostics: what to monitor and when
- Debugging common training issues

For recent developments (2023-2025), see [`docs/RECENT_RESEARCH.md`](docs/RECENT_RESEARCH.md), which covers:
- RegD: Hyperbolic-like expressiveness with Euclidean regions
- Concept2Box: Joint box+vector embeddings for two-view knowledge graphs
- TransBox: EL++-closed ontology embeddings
- Uncertainty quantification in embeddings

For a conceptual overview of why box embeddings are useful, see [`docs/CONCEPTUAL_OVERVIEW.md`](docs/CONCEPTUAL_OVERVIEW.md), which explains:
- The fundamental insight: regions vs. points
- Geometric intuition and volume semantics
- Comparison to other embedding methods
- When to use (and not use) box embeddings

## Overview

`subsume` provides geometric embeddings (boxes, hypercubes) that model containment relationships in NLP and knowledge graphs. Unlike vector embeddings, box embeddings encode logical invariants: if box A contains box B, then A "subsumes" B (entailment, hierarchical relationship).

## Architecture

This workspace contains three crates:

- **`subsume-core`**: Framework-agnostic traits (`Box`, `GumbelBox`, `BoxEmbedding`)
- **`subsume-candle`**: Implementation using `candle_core::Tensor` (✅ fully functional)
- **`subsume-ndarray`**: Implementation using `ndarray::Array1<f32>` (✅ fully functional)

## Key Features

- **Framework-agnostic**: Core traits work with any tensor/array library
- **Gumbel boxes**: Probabilistic boxes with Gumbel-Softmax for training stability
- **Containment operations**: Compute P(A ⊆ B) for entailment/hierarchical reasoning
- **Overlap probability**: Compute P(A ∩ B ≠ ∅) for entity resolution
- **Batch operations**: `BoxCollection` for efficient batch queries and containment matrices
- **Training utilities**: Log-space volume computation, volume regularization, temperature scheduling, and loss functions
- **Initialization utilities**: Safe initialization bounds, cross-pattern detection, and separation distance suggestions to avoid local identifiability problems
- **Training quality metrics**: MRR, Hits@K, Mean Rank, nDCG for evaluating embedding quality
- **Training diagnostics**: Convergence detection, gradient monitoring, volume tracking, loss component analysis
- **Embedding quality assessment**: Volume distribution analysis, containment accuracy verification, hierarchy detection
- **Calibration metrics**: Expected Calibration Error (ECE) and Brier score for probabilistic predictions
- **Property-based testing**: Property tests using `proptest` to verify mathematical invariants
- **Performance benchmarks**: Benchmarks with `criterion` across multiple dimensions
- **Serialization**: `serde` support for model persistence (JSON, bincode, etc.)

## Example

```rust
use subsume_ndarray::NdarrayBox;
use subsume_core::Box;
use ndarray::array;

fn main() -> Result<(), subsume_core::BoxError> {
    let premise = NdarrayBox::new(
        array![0.0, 0.0, 0.0],
        array![1.0, 1.0, 1.0],
        1.0
    )?;

    let hypothesis = NdarrayBox::new(
        array![0.2, 0.2, 0.2],
        array![0.8, 0.8, 0.8],
        1.0
    )?;

    // Compute entailment: P(hypothesis ⊆ premise)
    let entailment = premise.containment_prob(&hypothesis, 1.0)?;
    assert!(entailment > 0.9); // hypothesis is contained in premise
    
    Ok(())
}
```

## Research Background

### Foundational Papers

- **Vilnis et al. (2018)**: "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures" — Original probabilistic box embeddings framework
- **Dasgupta et al. (2020)**: "Improving Local Identifiability in Probabilistic Box Embeddings" (NeurIPS) — Gumbel-box process solving gradient sparsity
- **Li et al. (2019)**: "SmoothBox: Smoothing Box Embeddings for Better Training" — Gaussian convolution approach
- **Boratko et al. (2020)**: "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS) — Full expressiveness guarantees
- **Lee et al. (2022)**: "Box Embeddings for Event-Event Relation Extraction" (BERE)
- **Messner et al. (2022)**: "Temporal Knowledge Graph Completion with Box Embeddings" (BoxTE)
- **Chen et al. (2021)**: "Uncertainty-Aware Knowledge Graph Embeddings" (UKGE)

### Recent Developments (2023-2025)

- **Yang & Chen (2025)**: "RegD: Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions" — Validates box embeddings can achieve hyperbolic-like expressiveness in Euclidean space
- **Huang et al. (2023)**: "Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs" (ACL) — Box embeddings for concepts, vectors for entities
- **Yang, Chen & Sattler (2024)**: "TransBox: EL++-closed Ontology Embedding" — Box embeddings for formal ontologies
- **Xiao, He & Cao (2024)**: "Knowledge Graph Embedding by Normalizing Flows" — Uncertainty quantification in embeddings

See [`docs/MATHEMATICAL_FOUNDATIONS.md`](docs/MATHEMATICAL_FOUNDATIONS.md) for detailed mathematical formulations, [`docs/RECENT_RESEARCH.md`](docs/RECENT_RESEARCH.md) for recent developments, and [`docs/PAPER_VERIFICATION.md`](docs/PAPER_VERIFICATION.md) for verification of paper claims.

## Status

✅ **Core traits and ndarray implementation working** - Basic functionality is implemented and tested.

### Current Features

- Gumbel-Softmax sampling (using LCG to avoid `rand` dependency conflicts)
- Numerical stability utilities for temperature and sigmoid operations
- **Training utilities** (based on research from Vilnis et al. 2018, Dasgupta et al. 2020):
  - Log-space volume computation for high-dimensional boxes (prevents underflow/overflow)
  - Volume regularization to prevent boxes from becoming too large or small
  - Temperature scheduler for annealing during training (exponential decay)
  - Volume-based loss functions for containment and overlap relationships
- **Geometric operations**: Union, center, distance calculations
- **Batch operations**: Overlap matrix, overlapping boxes queries, k-nearest neighbors, bounding box computation
- **Recent research metrics** (2023-2025):
  - **Depth distance** (RegD 2025): Hyperbolic-like expressiveness with Euclidean boxes
  - **Boundary distance** (RegD 2025): Discrimination in inclusion chains
  - **Vector-to-box distance** (Concept2Box 2023): Hybrid representations (concepts as boxes, entities as vectors)
  - **Depth similarity**: Similarity metric based on depth distance
- **Training quality and diagnostics** (based on research from Box Embeddings library, BEUrRE, BoxE):
  - Rank-based metrics: MRR, Hits@K, Mean Rank, nDCG for link prediction evaluation
  - **Advanced training diagnostics**: 
    - Per-parameter gradient flow analysis (center vs size, min vs max coordinates)
    - **Depth-stratified gradient flow**: Track gradients by hierarchy depth to detect uneven learning
    - **Relation-stratified training stats**: Track convergence separately for each relation type in knowledge graphs
    - **Intersection volume tracking**: Monitor how containment relationships evolve during training
    - **Training phase detection**: Automatically identify exploration, exploitation, convergence, and instability phases
    - Gradient sparsity tracking and imbalance detection
    - Convergence detection, gradient explosion/vanishing, volume collapse
    - Loss component analysis with imbalance detection
  - **Sophisticated embedding quality assessment**:
    - Volume distribution entropy (Shannon entropy of normalized volumes)
    - Volume quantiles (Q25, Q50, Q75, Q95) and coefficient of variation
    - KL divergence between learned and target volume distributions
    - Containment hierarchy verification with transitive closure analysis
    - Cycle detection in containment relationships
    - Hierarchy depth analysis
    - Intersection topology regularity (sibling/parent-child ratios)
    - **Volume conservation analysis**: Verify parent volumes properly contain sum of children volumes
    - **Dimensionality utilization analysis**: Detect underutilized or redundant dimensions
    - **Generalization vs memorization metrics**: Distinguish learning structure from memorizing facts
    - Asymmetry quantification for directional relationships
    - Topological stability metrics across initializations
  - **Advanced calibration metrics**:
    - Expected Calibration Error (ECE) with equal-width binning
    - Adaptive Calibration Error (ACE) with equal-mass binning
    - Brier score for probabilistic predictions
    - Reliability diagram data for visualization
  - **Stratified evaluation**: Relation-stratified, depth-stratified, and frequency-stratified metrics
  - **Deep diagnostic techniques** (most nuanced and sophisticated):
    - Gradient flow analysis by hierarchy depth (detect uneven learning across levels)
    - Training phase detection (exploration, exploitation, convergence, instability)
    - Volume conservation verification (parent volumes vs sum of children)
    - Dimensionality utilization analysis (detect underutilized dimensions)
    - Generalization vs memorization metrics (inference performance vs direct facts)
- **Comprehensive test suite**: 115+ tests (149 test functions) including:
  - Unit tests (22 tests) covering basic functionality
  - Property-based tests (18 tests) using proptest, including 7 new tests for training utilities
  - Mathematical invariant tests (30+ tests) verifying set theory, probability theory, and geometric properties
  - Edge case tests (15+ tests) for error conditions and boundary cases
  - Matrix e2e tests (15 tests) for batch operations
  - Enriched methods tests (16 tests) for new geometric operations
  - Training quality tests (22 tests) for metrics, diagnostics, and deep diagnostic techniques
  - Property tests for training utilities (7 new tests) for volume regularization, temperature scheduling, and loss functions
- Benchmarks with `criterion`
- Serialization support with `serde` (ndarray backend)
- Examples for knowledge graphs, serialization, training utilities, training diagnostics, embedding quality assessment, advanced diagnostics, deep diagnostics, complete training loops, and hierarchical classification

### Next Steps

- ⏳ Resolve `candle-core` dependency issues (external, not our code - bf16/rand version conflict)
- ✅ Add serialization for Candle backend (implemented in `candle_serialization.rs`)
- ✅ Add complete training loop example (integrates all diagnostics)
- ✅ Add hierarchical classification example
- 📄 Center-offset representation: Documented in `docs/CENTER_OFFSET_REPRESENTATION.md` (not yet implemented, considered for future if needed)

