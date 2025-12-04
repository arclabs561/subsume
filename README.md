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
- **Property-based testing**: Comprehensive property tests using `proptest` to verify mathematical invariants
- **Performance benchmarks**: Detailed benchmarks with `criterion` across multiple dimensions
- **Serialization**: Full `serde` support for model persistence (JSON, bincode, etc.)

## Example

```rust
use subsume_ndarray::NdarrayBox;
use subsume_core::Box;

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

✅ **Core functionality complete** - Traits and implementations are working.

### Recent Updates

- ✅ Implemented Gumbel-Softmax sampling using LCG (Linear Congruential Generator)
- ✅ Non-deterministic sampling without external `rand` dependency (removed from ndarray)
- ✅ Gumbel-Softmax membership probability calculation with numerical stability
- ✅ Fixed `overlap_prob` implementation (proper inclusion-exclusion formula)
- ✅ Added error conversion for Candle backend
- ✅ Removed redundant device field from CandleBox
- ✅ Added comprehensive unit tests (32 tests covering edge cases)
- ✅ Added runnable examples for both backends
- ✅ Implemented `BoxEmbedding` trait with `BoxCollection` for batch operations
- ✅ Documented all error variant fields
- ✅ Fixed intersection handling for disjoint boxes
- ✅ Added numerical stability utilities for temperature handling and Gumbel-Softmax
- ✅ **Property-based testing** with `proptest` (10+ property tests covering mathematical invariants)
- ✅ **Performance benchmarks** with `criterion` (6 benchmark suites across multiple dimensions)
- ✅ **Serialization support** with `serde` (JSON and other formats for model persistence)

### Next Steps

- ⏳ Resolve `candle-core` dependency issues (external, not our code - bf16/rand version conflict)
- ⏳ Add serialization for Candle backend
- ⏳ Consider center-offset representation as alternative to min-max
- ⏳ Add more comprehensive examples (knowledge graphs, hierarchical classification)

