# subsume

Geometric box embeddings for containment, entailment, and hierarchical relationships.

## Overview

Box embeddings represent concepts as axis-aligned hyperrectangles. Containment models subsumption: if box $A$ contains box $B$, then $A$ subsumes $B$:

$$B \subseteq A \iff \text{A subsumes B}$$

This directly models logical subsumption from formal logic, where one statement is more general than another.

## Quick Start

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

// Compute containment probability: P(hypothesis ⊆ premise)
let entailment = premise.containment_prob(&hypothesis, 1.0)?;
println!("Entailment: {:.2}", entailment);
```

## Core Operations

- **Containment probability**: $P(A \subseteq B)$ — probability that box $A$ contains box $B$
- **Overlap probability**: $P(A \cap B \neq \emptyset)$ — probability that boxes $A$ and $B$ intersect

## Structure

- `subsume-core`: Traits (`Box`, `GumbelBox`, `BoxEmbedding`)
- `subsume-candle`: Implementation using `candle_core::Tensor`
- `subsume-ndarray`: Implementation using `ndarray::Array1<f32>`

## Features

- Gumbel boxes: Probabilistic boxes with Gumbel-Softmax for training stability
- Batch operations: `BoxCollection` for containment matrices
- Training utilities: Volume regularization, temperature scheduling, loss functions
- Training metrics: MRR, Hits@K, Mean Rank, nDCG
- Training diagnostics: Gradient flow, volume tracking, phase detection
- Quality assessment: Volume distribution, hierarchy verification, calibration
- Automated evaluation: Benchmark infrastructure with plotting and metrics collection
- Property-based tests: 149+ tests (property tests, unit tests, integration tests)
- Serialization: `serde` support for all box types
- Distance metrics: Boundary distance, depth distance, vector-to-box distance
- Optimizers: Adam, AdamW, SGD with configurable hyperparameters

## Documentation

**New to box embeddings?** Start with [`docs/READING_GUIDE.md`](docs/READING_GUIDE.md) for recommended reading order.

**Preview Docs:** 
- **All markdown docs**: `./docs/preview-all.sh` - GitHub-style preview with KaTeX (see [`docs/README_PREVIEW.md`](docs/README_PREVIEW.md))
- **Rust docs**: `make docs-open` - API docs with KaTeX rendering (see [`docs/README_RUSTDOC.md`](docs/README_RUSTDOC.md))

- [`docs/READING_GUIDE.md`](docs/READING_GUIDE.md) — **Start here!** Recommended reading order
- [`docs/CONCEPTUAL_OVERVIEW.md`](docs/CONCEPTUAL_OVERVIEW.md) — Why box embeddings
- [`docs/MATHEMATICAL_FOUNDATIONS.md`](docs/MATHEMATICAL_FOUNDATIONS.md) — Mathematical details with derivations
- [`docs/MATH_TO_CODE_CONNECTIONS.md`](docs/MATH_TO_CODE_CONNECTIONS.md) — How theory maps to implementation
- [`docs/MATH_QUICK_REFERENCE.md`](docs/MATH_QUICK_REFERENCE.md) — Quick reference for key formulas
- [`docs/PRACTICAL_GUIDE.md`](docs/PRACTICAL_GUIDE.md) — Usage guidance
- [`docs/RECENT_RESEARCH.md`](docs/RECENT_RESEARCH.md) — Recent papers (2023-2025)
- [`docs/REAL_TRAINING_EXAMPLES.md`](docs/REAL_TRAINING_EXAMPLES.md) — Real training examples with datasets

## Research

**Foundational:**
- Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
- Dasgupta et al. (2020): "Improving Local Identifiability in Probabilistic Box Embeddings" (NeurIPS)
- Boratko et al. (2020): "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS)

**Recent:**
- Yang & Chen (2025): "RegD: Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions"
- Huang et al. (2023): "Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs" (ACL)
- Yang, Chen & Sattler (2024): "TransBox: EL++-closed Ontology Embedding"

## Status

- ✅ Core traits and implementations working
- ✅ 149 tests passing (property tests, unit tests, integration tests, regression tests)
- ✅ Dataset loading utilities for WN18RR, FB15k-237, YAGO3-10
- ✅ Training infrastructure: Adam/AdamW/SGD optimizers, negative sampling, evaluation
- ✅ BoxE implementation: translational bumps, scoring, margin-based loss
- ✅ Center-offset representation: conversion utilities implemented
- ✅ Candle backend: serialization and all examples implemented
- ✅ Distance metrics: Boundary, depth, vector-to-box distances
- ✅ Training examples: Real training loops for WN18RR, FB15k-237, BoxE
- ✅ Error handling: Proper Result types, no panics in user-facing code
- ✅ Paper configurations: Dasgupta 2020, Boratko 2020, Vilnis 2018 presets
