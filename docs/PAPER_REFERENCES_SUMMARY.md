# Paper References in Rust Documentation

This document summarizes all research paper references integrated into the Rust API documentation, providing grounding for mathematical explanations and paths for deeper exploration.

## Summary

**36 paper references** have been integrated across 7 core modules, connecting code implementations to their research foundations.

## References by Module

### `box_trait.rs` - Core Box Operations

- **Vilnis et al. (2018)**: "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
  - Foundational work establishing box volumes as probability measures
  - Containment probability formula: \(P(B|A) = \text{Vol}(A \cap B) / \text{Vol}(A)\)
  - Reference: [arXiv:1805.06627](https://arxiv.org/abs/1805.06627)

### `gumbel.rs` - Gumbel Box Embeddings

- **Dasgupta et al. (2020)**: "Improving Local Identifiability in Probabilistic Box Embeddings" (NeurIPS)
  - Introduces Gumbel-box process to solve local identifiability problem
  - Expected volume formula using Bessel function \(K_0\)
  - Reference: [arXiv:2004.13131](https://arxiv.org/abs/2004.13131) | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/2cfa8a9da6cdae0c7ac0b94a2c3f4c0d-Abstract.html)
  
- **Jang et al. (2016)**: "Categorical Reparameterization with Gumbel-Softmax"
  - Foundation for Gumbel-Softmax trick used in probabilistic embeddings

- **Vilnis et al. (2018)**: Foundational probabilistic box embeddings

### `boxe.rs` - BoxE Model

- **Boratko et al. (2020)**: "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS)
  - Translational bump model for relation-specific transformations
  - Margin-based ranking loss for training
  - Reference: [arXiv:2007.06267](https://arxiv.org/abs/2007.06267) | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/6dbbe6abe5f14af882ff977fc3f35501-Abstract.html)

### `distance.rs` - Advanced Distance Metrics

- **Yang & Chen (2025)**: "RegD: Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions"
  - Depth-based distance incorporating region size
  - Boundary distance for containment relationships
  - Reference: arXiv:2501.17518

- **Huang et al. (2023)**: "Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs" (ACL)
  - Vector-to-box distance for hybrid representations
  - Reference: ACL 2023

### `trainer.rs` - Training Infrastructure

- **Bordes et al. (2013)**: "Translating Embeddings for Modeling Multi-relational Data" (TransE)
  - Introduces negative sampling for knowledge graphs
  - Margin-based ranking loss
  - Standard link prediction evaluation protocol

- **Boratko et al. (2020)**: BoxE training approach

- **Vilnis et al. (2018)**: Foundational box embedding training

### `training.rs` - Evaluation Metrics

- **Bordes et al. (2013)**: Standard link prediction metrics (MRR, Hits@K, Mean Rank)

### `utils.rs` - Loss Functions

- **Vilnis et al. (2018)**: Probabilistic box embedding loss functions
- **Boratko et al. (2020)**: BoxE margin-based ranking loss

## Integration Pattern

All paper references follow a consistent pattern:

1. **Research Background section**: Provides context and key papers
2. **Specific references**: Links to papers with arXiv/venue links where available
3. **Key insights**: Highlights what the paper contributes to understanding the code
4. **Section references**: Points to specific sections when relevant

## Benefits

1. **Grounded explanations**: Mathematical concepts are tied to their research origins
2. **Exploration paths**: Readers can dive deeper into specific papers
3. **Credibility**: Shows the implementation is based on peer-reviewed research
4. **Context**: Helps readers understand why certain design choices were made

## Usage in Documentation

Paper references appear in:
- Module-level documentation (`//!`)
- Trait documentation
- Function documentation
- Mathematical formulation sections

All references use proper LaTeX formatting for mathematical notation and include links to papers when available.

