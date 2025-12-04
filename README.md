# subsume

Framework-agnostic geometric box embeddings for containment, entailment, and hierarchical relationships.

## Overview

`subsume` provides geometric embeddings (boxes, hypercubes) that model containment relationships in NLP and knowledge graphs. Unlike vector embeddings, box embeddings encode logical invariants: if box A contains box B, then A "subsumes" B (entailment, hierarchical relationship).

## Architecture

This workspace contains three crates:

- **`subsume-core`**: Framework-agnostic traits (`Box`, `GumbelBox`, `BoxEmbedding`)
- **`subsume-candle`**: Implementation using `candle_core::Tensor`
- **`subsume-ndarray`**: Implementation using `ndarray::Array1<f32>`

## Key Features

- **Framework-agnostic**: Core traits work with any tensor/array library
- **Gumbel boxes**: Probabilistic boxes with Gumbel-Softmax for training stability
- **Containment operations**: Compute P(A ⊆ B) for entailment/hierarchical reasoning
- **Overlap probability**: Compute P(A ∩ B ≠ ∅) for entity resolution
- **Batch operations**: `BoxCollection` for efficient batch queries and containment matrices
- **Property-based testing**: Property tests using `proptest` to verify mathematical invariants
- **Performance benchmarks**: Benchmarks with `criterion` across multiple dimensions
- **Serialization**: `serde` support for model persistence (JSON, bincode, etc.)

## Example

```rust
use subsume_ndarray::NdarrayBox;
use subsume_core::Box;
use ndarray::array;

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
- **Comprehensive test suite**: 77+ tests including:
  - Unit tests (22 tests) covering basic functionality
  - Property-based tests (10+ tests) using proptest
  - Mathematical invariant tests (30+ tests) verifying set theory, probability theory, and geometric properties
  - Edge case tests (15+ tests) for error conditions and boundary cases
- Benchmarks with `criterion`
- Serialization support with `serde` (ndarray backend)
- Examples for knowledge graphs and serialization

### Next Steps

- ⏳ Resolve `candle-core` dependency issues (external, not our code - bf16/rand version conflict)
- ⏳ Add serialization for Candle backend
- ⏳ Consider center-offset representation as alternative to min-max
- ⏳ Add more examples (hierarchical classification, training loops)

