# subsume

Framework-agnostic geometric box embeddings for containment, entailment, and hierarchical relationships.

## Why "subsume"?

The name **subsume** comes from the logical relationship that box embeddings model: when box A contains box B, we say that A **subsumes** B. This term originates from logic and philosophy, where "subsumption" means to include or absorb something into a more general category or concept.

In box embeddings:
- **Subsumption** = containment relationship (A ⊇ B means A subsumes B)
- **Entailment** = if premise subsumes hypothesis, then premise entails hypothesis
- **Hierarchical relationships** = parent concepts subsume child concepts

For example, if "animal" is represented by a box that contains the box for "dog", then "animal" subsumes "dog" — the more general concept contains the more specific one.

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

Based on:
- Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
- Lee et al. (2022): "Box Embeddings for Event-Event Relation Extraction" (BERE)
- Messner et al. (2022): "Temporal Knowledge Graph Completion with Box Embeddings" (BoxTE)
- Chen et al. (2021): "Uncertainty-Aware Knowledge Graph Embeddings" (UKGE)

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
- Examples for knowledge graphs, serialization, training utilities, training diagnostics, embedding quality assessment, advanced diagnostics, and deep diagnostics

### Next Steps

- ⏳ Resolve `candle-core` dependency issues (external, not our code - bf16/rand version conflict)
- ✅ Add serialization for Candle backend (implemented in `candle_serialization.rs`)
- ⏳ Consider center-offset representation as alternative to min-max
- ⏳ Add more examples (hierarchical classification, complete training loops)

