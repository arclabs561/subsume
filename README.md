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
- Property-based tests: 137+ tests
- Serialization: `serde` support

## Documentation

- [`docs/CONCEPTUAL_OVERVIEW.md`](docs/CONCEPTUAL_OVERVIEW.md) — Why box embeddings
- [`docs/MATHEMATICAL_FOUNDATIONS.md`](docs/MATHEMATICAL_FOUNDATIONS.md) — Mathematical details
- [`docs/PRACTICAL_GUIDE.md`](docs/PRACTICAL_GUIDE.md) — Usage guidance
- [`docs/RECENT_RESEARCH.md`](docs/RECENT_RESEARCH.md) — Recent papers (2023-2025)

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

- Core traits and implementations working
- 137+ tests passing
- Candle backend: serialization implemented
- Known issue: `candle-core` dependency conflict (external, bf16/rand version mismatch)
