# Reproducing Paper Results

This document provides methodology for reproducing quantitative claims from papers.

## Dasgupta et al. (2020): "6 F1 Score Improvement"

### Claim
GumbelBox achieves ~6 F1 score improvement over SmoothBox on knowledge graph completion tasks.

### Methodology

**Dataset**: WordNet (WN18RR) or similar knowledge graph dataset

**Evaluation Protocol**:
1. Train two models:
   - Model A: Using SmoothBox (soft volume with Gaussian convolution)
   - Model B: Using GumbelBox (Gumbel-box process)
2. Use identical hyperparameters except for box type
3. Evaluate on link prediction task
4. Measure F1 score on test set

**Expected Results**:
- SmoothBox F1: ~0.XX (baseline)
- GumbelBox F1: ~0.XX + 0.06 (improvement)

**Implementation Notes**:
- SmoothBox: Use `log_space_volume()` with temperature smoothing
- GumbelBox: Use `NdarrayGumbelBox` or `CandleGumbelBox`
- Training: Use `volume_containment_loss()` for both
- Evaluation: Use `ContainmentAccuracy::f1()` metric

**Verification**:
```rust
// See subsume-ndarray/src/quantitative_verification_tests.rs
// test_dasgupta_2020_gradient_density_quantitative()
```

---

## RegD (2025): "Hyperbolic-Like Expressiveness"

### Claim
Euclidean boxes with depth distance achieve hyperbolic-like expressiveness.

### Methodology

**Test Setup**:
1. Create hierarchical dataset (tree structure)
2. Embed using:
   - Method A: Euclidean distance
   - Method B: Depth distance (RegD)
3. Measure:
   - Crowding effect (distance variance in hierarchies)
   - Hierarchy preservation (distance increases with depth)
   - Expressiveness (ability to represent complex hierarchies)

**Expected Results**:
- Depth distance should show:
  - Higher coefficient of variation (less crowding)
  - Monotonic increase with hierarchy depth
  - Better separation of siblings

**Verification**:
```rust
// See subsume-ndarray/src/quantitative_verification_tests.rs
// test_regd_2025_crowding_effect_mitigation()
// test_regd_2025_hierarchy_depth_property()
```

---

## Concept2Box (2023): "Joint Learning Improves Performance"

### Claim
Joint learning of concept boxes and entity vectors improves performance on two-view KGs.

### Methodology

**Dataset**: DBpedia KG or similar two-view knowledge graph

**Evaluation Protocol**:
1. Train three models:
   - Model A: Concepts as boxes only
   - Model B: Entities as vectors only
   - Model C: Joint learning (concepts as boxes, entities as vectors)
2. Use `vector_to_box_distance()` for Model C
3. Evaluate on:
   - Link prediction (entity-entity)
   - Concept classification (entity-concept)
   - Concept hierarchy (concept-concept)

**Expected Results**:
- Model C should outperform Models A and B
- Joint learning should improve both views

**Verification**:
- See `subsume-ndarray/examples/concept2box_joint_learning.rs`
- Requires full training implementation

---

## Boratko et al. (2020): "State-of-the-Art Performance"

### Claim
BoxE achieves state-of-the-art performance on FB15k-237, WN18RR, YAGO3-10.

### Methodology

**Datasets**:
- FB15k-237: Freebase knowledge graph
- WN18RR: WordNet knowledge graph
- YAGO3-10: YAGO knowledge graph

**Evaluation Metrics**:
- MRR (Mean Reciprocal Rank)
- Hits@1, Hits@3, Hits@10

**Expected Results** (from paper):
- FB15k-237: MRR ~0.XX, Hits@10 ~0.XX
- WN18RR: MRR ~0.XX, Hits@10 ~0.XX
- YAGO3-10: MRR ~0.XX, Hits@10 ~0.XX

**Implementation Notes**:
- Requires full BoxE implementation (translational bumps)
- Current implementation provides foundation (Box trait, containment operations)
- Full BoxE requires additional components (bumps, training loop)

**Verification**:
- See `subsume-ndarray/src/invariant_tests.rs`
- `test_expressiveness_properties()` verifies theoretical expressiveness
- Full performance verification requires complete BoxE implementation

---

## Verification Status

| Paper | Quantitative Claim | Verification Method | Status |
|-------|-------------------|---------------------|--------|
| Dasgupta 2020 | 6 F1 improvement | Training simulation | ⏳ Requires full training loop |
| RegD 2025 | Hyperbolic-like expressiveness | Crowding effect test | ✅ Verified (test_regd_2025_crowding_effect_mitigation) |
| RegD 2025 | Addresses crowding | Hierarchy depth test | ✅ Verified (test_regd_2025_hierarchy_depth_property) |
| Concept2Box 2023 | Joint learning improves | Joint training example | ⏳ Requires full training implementation |
| Boratko 2020 | SOTA performance | Expressiveness tests | ✅ Verified (theoretical), ⏳ Performance requires full BoxE |

---

## Running Verification

```bash
# Run quantitative verification tests
cargo test --workspace --lib quantitative_verification

# Run comparative benchmarks
cargo bench --workspace --bench comparative_benchmarks

# Run paper verification tests
cargo test --workspace --lib paper_verification
```

---

## Next Steps

1. **Full Training Loops**: Implement complete training examples that reproduce paper results
2. **Dataset Integration**: Add support for standard KG datasets (WN18RR, FB15k-237)
3. **Performance Baselines**: Establish baseline performance metrics for comparison
4. **Automated Verification**: Create CI tests that verify performance doesn't regress

