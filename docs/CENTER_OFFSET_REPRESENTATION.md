# Center-Offset vs Min-Max Representation for Box Embeddings

## Overview

Box embeddings can be parameterized using two fundamental representations: **center-offset** and **min-max**. This document analyzes the trade-offs and provides guidance on when to use each approach.

## Representations

### Min-Max Representation (Current Implementation)

**Definition**: A box is characterized by two points `(x_m, x_M)`, where `x_m, x_M ∈ R^d` are the minimum and maximum corners with the constraint `x_{m,i} ≤ x_{M,i}` for each dimension `i`.

**Current Implementation**: Our `NdarrayBox` and `CandleBox` use min-max representation directly.

**Advantages**:
- **Direct geometric operations**: Computing volume, checking containment, or determining intersection is straightforward without transformation overhead
- **Memory efficiency**: Stores exactly what's needed—the actual boundaries—without redundant parameters
- **Inference speed**: Once trained, provides faster inference for ranking/retrieval applications
- **Geometric intuition**: Directly represents the box boundaries, making it easy to visualize and reason about

**Disadvantages**:
- **Constraint satisfaction during learning**: Enforcing `x_{m,i} ≤ x_{M,i}` during gradient descent is challenging. If parameters violate constraints, gradients can become degenerate
- **Difficult parameterization for networks**: Neural networks struggle to naturally produce valid box representations. Requires explicit constraint layers or post-hoc clipping
- **Asymmetric learning**: Treats minimum and maximum corners asymmetrically, requiring the network to learn that these are related

### Center-Offset Representation (Not Yet Implemented)

**Definition**: A box is encoded using a center point `c_x ∈ R^d` and an offset vector `o_x ∈ R^d`. The conversion to min-max follows:

```
x_m = σ(c_x - SOFTPLUS(o_x))
x_M = σ(c_x + SOFTPLUS(o_x))
```

where `σ` is the element-wise sigmoid function and `SOFTPLUS` ensures non-negative offsets.

**Advantages**:
- **Neural network compatibility**: More suitable for gradient-based learning. Decouples center position from box dimensions, allowing semi-independent learning
- **Automatic constraint satisfaction**: Using activation functions (SOFTPLUS + sigmoid) ensures box validity constraints are automatically satisfied through the forward pass
- **Improved gradient flow**: Enables smoother gradient flow through the network. No need to enforce inequality constraints during backpropagation
- **Geometric interpretability**: Explicitly separates semantic meaning: center indicates spatial position, offset controls box extent

**Disadvantages**:
- **Computational overhead**: Each forward pass requires multiple function evaluations (SOFTPLUS and sigmoid) to convert to min-max coordinates
- **Normalization requirements**: SOFTPLUS can output values exceeding 1, requiring explicit post-hoc normalization
- **Indirection in geometric operations**: Must convert to min-max coordinates first for geometric operations, adding layers of indirection

## When to Use Each Representation

### Use Center-Offset When:
- **Training with gradient-based optimization**: SGD, Adam, or other gradient methods benefit from smoother loss landscapes
- **End-to-end differentiable models**: Any architecture requiring backpropagation through box operations
- **Multi-task learning**: When jointly learning box embeddings with other tasks, prevents conflicting gradient directions
- **Probabilistic box embeddings**: When modeling uncertainty or using distributions over boxes (e.g., Gumbel distributions)

### Use Min-Max When:
- **Inference-only applications**: Once trained, min-max provides faster geometric operations
- **Memory-constrained environments**: Direct storage of boundaries without transformation overhead
- **High-dimensional embeddings**: For very high dimensions (768+), center-offset computational cost scales linearly per dimension
- **Simple geometric queries**: When primarily doing containment checks or volume computations without parameter updates

## Implementation Considerations

### Hybrid Approach
Some implementations store embeddings in center-offset form during training for gradient stability, then convert to min-max for efficient inference. This trades memory and conversion cost during training for downstream speed.

### Activation Function Selection
The choice between SOFTPLUS and other functions affects gradient flow and expressiveness. SOFTPLUS smoothly transitions from linear to constant behavior, but alternatives like ReLU or ELU may offer different speed-stability trade-offs.

### Normalization Strategy
Decide whether to normalize during forward pass or in loss computation. During-forward normalization maintains bounded values throughout the network but adds computation. Loss-based normalization is cheaper but risks numerical instability.

### Initialization Strategy
Center-offset representation requires careful initialization:
- **Small offsets** (near zero): Initializes boxes as points, which can slow learning
- **Larger offsets**: Initializes overlapping boxes, which may hinder type hierarchy learning
- The initialization choice significantly impacts convergence speed

## Current Status

**Min-Max Representation**: ✅ Fully implemented and tested in `subsume-ndarray` and `subsume-candle`

**Center-Offset Representation**: ⏳ Not yet implemented. Considered for future enhancement if training stability becomes an issue or if users request it for neural network integration.

## References

1. Dasgupta et al. (2021). "Modeling Fine-Grained Entity Types with Box Embeddings." ACL 2021.
2. Vilnis et al. (2018). "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures." ACL 2018.
3. Various implementations in Box Embeddings library, BEUrRE, and BoxE

