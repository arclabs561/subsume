# Box Embeddings Research Summary

## Key Papers Reviewed

### Foundational Papers
1. **Vilnis et al. (2018)**: "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
   - Introduced box embeddings as hyperrectangles
   - Volume represents probability
   - Can model both positive and negative correlations (unlike cone-based models)
   - Soft volume calculations to avoid gradient sparsity
   - Volume regularization is critical for training stability

2. **Dasgupta et al. (2020)**: "Improving Local Identifiability in Probabilistic Box Embeddings"
   - Gumbel distributions for box edges
   - Expected volume calculations using Bessel functions
   - Better gradient flow than soft volume approach
   - Addresses local identifiability problems

3. **Recent Advances (2020-2024)**:
   - Volume regularization methods
   - Temperature annealing for Gumbel-Softmax
   - Joint geometric and probabilistic loss functions
   - Integration with GNNs and transformers
   - Applications in taxonomy expansion, knowledge graphs, task modeling

## Key Mathematical Invariants

### Containment Properties
- **Reflexivity**: A ⊆ A (always true)
- **Transitivity**: If A ⊆ B and B ⊆ C, then A ⊆ C
- **Antisymmetry**: If A ⊆ B and B ⊆ A, then A = B
- **Containment implies overlap**: If A ⊆ B, then P(A ∩ B ≠ ∅) should be high

### Volume Properties
- **Non-negativity**: Volume ≥ 0
- **Monotonicity**: If A ⊆ B geometrically, then vol(A) ≤ vol(B)
- **Subadditivity**: vol(A ∪ B) ≤ vol(A) + vol(B)
- **Scaling**: vol(kA) = k^d * vol(A) where d is dimension

### Intersection Properties
- **Commutativity**: A ∩ B = B ∩ A
- **Associativity**: (A ∩ B) ∩ C = A ∩ (B ∩ C)
- **Idempotence**: A ∩ A = A
- **Volume bound**: vol(A ∩ B) ≤ min(vol(A), vol(B))

### Overlap Properties
- **Reflexivity**: P(A ∩ A ≠ ∅) = 1.0
- **Symmetry**: P(A ∩ B ≠ ∅) = P(B ∩ A ≠ ∅)
- **Bounds**: 0 ≤ P(A ∩ B ≠ ∅) ≤ 1

### Gumbel Box Properties
- **Membership bounds**: 0 ≤ P(x ∈ box) ≤ 1
- **Sample bounds**: Samples should be within [min, max] bounds
- **Temperature effects**: Lower temperature → harder bounds, higher temperature → softer bounds

## Numerical Stability Issues

### Critical Issues Identified
1. **Volume underflow/overflow**: Products over many dimensions can underflow to 0 or overflow to inf
   - **Solution**: Compute volumes in log-space
   - **Current status**: Not implemented - direct multiplication used

2. **Temperature extremes**: Very low temperatures cause vanishing gradients, very high temperatures lose correspondence
   - **Solution**: Clamp temperature to safe range [1e-3, 10.0]
   - **Current status**: ✅ Implemented in `utils.rs`

3. **Division by zero**: When computing containment_prob = intersection_volume / volume
   - **Solution**: Add epsilon guards
   - **Current status**: Partially handled - need to verify all cases

4. **Gumbel sampling**: ln(0) errors when uniform sample is exactly 0 or 1
   - **Solution**: Clamp uniform samples to [epsilon, 1-epsilon]
   - **Current status**: ✅ Implemented with 1e-7 epsilon

5. **Sigmoid overflow**: exp(-x) can overflow for large negative x
   - **Solution**: Stable sigmoid implementation
   - **Current status**: ✅ Implemented in `utils.rs`

## Training Techniques from Research

### Volume Regularization
- **Purpose**: Prevent boxes from becoming arbitrarily large or small
- **Method**: Add penalty term: λ * max(0, vol(box) - threshold)
- **Current status**: ✅ Implemented in `utils.rs` as `volume_regularization()`
- **Usage**: `volume_regularization(volume, threshold_max, threshold_min, lambda)`

### Temperature Annealing
- **Purpose**: Start with high temperature (exploration) and decrease to low temperature (exploitation)
- **Method**: Exponential decay: T(t) = T₀ * decay^t
- **Current status**: ✅ Implemented in `utils.rs` as `temperature_scheduler()`
- **Usage**: `temperature_scheduler(initial_temp, decay_rate, step, min_temp)`

### Log-Space Volume Computation
- **Purpose**: Avoid numerical underflow/overflow in high dimensions
- **Method**: log(vol) = Σ log(max[i] - min[i])
- **Current status**: ✅ Implemented in `utils.rs` as `log_space_volume()` and automatically used for boxes with dim > 10
- **Usage**: `log_space_volume(side_lengths)` or automatic in `NdarrayBox::volume()` for high-dimensional boxes

### Gumbel Intersection with Bessel Approximation
- **Purpose**: Better gradient flow for Gumbel boxes
- **Method**: Use softplus approximation of Bessel function
- **Current status**: ⚠️ Partially implemented - using tanh mapping
- **Recommendation**: Consider implementing Bessel-based intersection

## Edge Cases to Test

### Zero-Volume Boxes
- Boxes where min[i] == max[i] for some dimension
- Should have volume = 0
- Intersection with zero-volume box should handle gracefully

### Disjoint Boxes
- Boxes that don't overlap
- Intersection should return zero-volume box (not error)
- Containment_prob should be 0
- Overlap_prob should be 0

### Very Small Boxes
- Boxes with very small side-lengths (near machine epsilon)
- Volume calculations may underflow
- Need epsilon guards

### Very Large Boxes
- Boxes spanning large coordinate ranges
- Volume calculations may overflow
- Need log-space computation

### High-Dimensional Boxes
- Boxes with many dimensions (d > 10)
- Volume products become very small
- Log-space computation essential

### Boundary Cases
- Boxes touching at boundaries (min_a == max_b)
- Should be considered as overlapping (or not, depending on definition)
- Need consistent handling

## Recommendations for Implementation

### High Priority
1. ✅ **Log-space volume computation**: Critical for high-dimensional boxes - **IMPLEMENTED**
2. ✅ **Volume regularization utilities**: Important for training stability - **IMPLEMENTED**
3. ✅ **Temperature scheduler**: Useful for training workflows - **IMPLEMENTED**
4. ✅ **Volume-based loss functions**: For training containment/overlap relationships - **IMPLEMENTED**

### Medium Priority
4. **Bessel-based Gumbel intersection**: Better gradient flow
5. **Additional property tests**: Test volume subadditivity, scaling properties
6. **Edge case handling**: More comprehensive tests for boundary conditions

### Low Priority
7. **Center-offset representation**: Alternative parameterization
8. **Soft volume options**: Multiple volume computation methods
9. **Training utilities**: Loss functions, optimizers specific to box embeddings

## References

- Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
- Dasgupta et al. (2020): "Improving Local Identifiability in Probabilistic Box Embeddings"
- Box Embeddings Library (2021): "Box Embeddings: An open-source library for representation learning with box embeddings"
- Task2Box (2024): "Box Embeddings for Modeling Asymmetric Task Relationships"
- ProtoBox (2024): "Contextualized Box Embeddings for Word Sense Disambiguation"
- BoxTM (2024): "Self-supervised Topic Taxonomy Discovery in the Wild"
- BOXTAXO (2024): "Taxonomy Expansion via Box Embeddings"
- TAXBOX (2024): "Taxonomy Completion via Box Embedding"

