# Gaps Analysis: What's Missing

This document identifies what we've missed in justification, explanation, exemplification, testing, and evaluation.

## Not Justified

### 1. Generic `depth_distance` Volume Approximation

**Issue**: The generic implementation in `subsume-core/src/distance.rs` uses volume directly instead of `log(volume)`, which is a significant deviation from the RegD (2025) paper formula.

**Current Implementation**:
```rust
// Uses volume directly, not log(volume)
let log_vol_a = if vol_a > B::Scalar::from(1e-10) {
    vol_a // Simplified: use volume directly
} else {
    B::Scalar::from(-10.0)
};
```

**Paper Formula**: `d_depth(A, B) = d_Euclidean(A, B) + α·|log(Vol(A)) - log(Vol(B))|`

**Gap**: No justification for why this approximation is acceptable. The paper explicitly uses log volumes to address the crowding effect.

**Fix Needed**: Document why this approximation exists (generic trait limitations) and when backends should override it.

---

### 2. Dual Implementation Strategy

**Issue**: We have both generic (`subsume-core`) and backend-specific (`subsume-ndarray`, `subsume-candle`) implementations of depth distance.

**Gap**: Not explained why this dual approach is necessary or when to use which.

**Fix Needed**: Document the design decision: generic for framework-agnostic code, backend-specific for optimized implementations with actual log operations.

---

## Not Explained Well Enough

### 3. Performance Claims Without Verification Methodology

**Issue**: Paper verification doc mentions "GumbelBox achieves ~6 F1 score improvement over SmoothBox" but doesn't explain:
- What dataset was used?
- What evaluation protocol?
- How to reproduce this result?
- What are the conditions (hyperparameters, initialization, etc.)?

**Gap**: Claims are stated but not actionable.

**Fix Needed**: Add a "Reproducing Paper Results" section with:
- Dataset requirements
- Hyperparameter settings
- Evaluation protocol
- Expected results ranges

---

### 4. Volume vs Log(Volume) Approximation

**Issue**: The generic `depth_distance` uses volumes instead of logs, but this isn't clearly documented as a limitation.

**Gap**: Users might not realize the generic implementation is approximate.

**Fix Needed**: Add prominent warning in doc comments that generic implementation is approximate and backends provide exact implementations.

---

### 5. Expressiveness Claims Without Empirical Validation

**Issue**: Papers claim "hyperbolic-like expressiveness" and "addresses crowding effect" but we only verify mathematical properties, not empirical claims.

**Gap**: We verify the formula is correct, but not that it achieves the claimed benefits.

**Fix Needed**: Add tests/benchmarks that demonstrate:
- Crowding effect in hierarchies
- How depth distance addresses it
- Comparison to Euclidean distance on hierarchical data

---

## Not Exemplified

### 6. New Metrics Not Integrated into Training Examples

**Issue**: `complete_training_loop.rs` doesn't use the new distance metrics (depth_distance, boundary_distance, vector_to_box_distance).

**Gap**: Users don't see how to use new metrics in realistic training scenarios.

**Fix Needed**: Integrate new metrics into `complete_training_loop.rs`:
- Use depth distance for similarity search
- Use boundary distance for hierarchy validation
- Use vector-to-box distance for hybrid representations

---

### 7. No Concept2Box Joint Learning Example

**Issue**: Concept2Box paper emphasizes joint learning of concept boxes and entity vectors, but we only have a standalone distance metric example.

**Gap**: No example showing the full Concept2Box workflow.

**Fix Needed**: Create `concept2box_joint_learning.rs` example showing:
- Concepts as boxes
- Entities as vectors
- Joint training objective
- Evaluation on two-view KG

---

### 8. No RegD Training Example

**Issue**: RegD paper shows training with depth distance for hierarchy learning, but we only have a standalone metric example.

**Gap**: No example showing RegD-style training.

**Fix Needed**: Create `regd_training.rs` example showing:
- Training with depth distance loss
- Comparison to Euclidean distance
- Hierarchy preservation

---

### 9. No Candle Examples for New Metrics

**Issue**: All new metric examples use `ndarray` backend only.

**Gap**: Candle users don't have examples.

**Fix Needed**: Create `candle_recent_research_metrics.rs` example.

---

## Not Tested

### 10. No Quantitative Performance Verification

**Issue**: Paper verification tests only check mathematical properties, not quantitative performance claims.

**Gap**: We don't verify "6 F1 improvement" or "SOTA performance" claims.

**Fix Needed**: Add tests that:
- Simulate training with GumbelBox vs SmoothBox
- Measure F1 scores
- Verify improvement is in expected range

---

### 11. No Crowding Effect Tests

**Issue**: RegD paper claims depth distance addresses crowding effect, but we don't test this.

**Gap**: No empirical verification of the key claim.

**Fix Needed**: Add test that:
- Creates hierarchy with many children
- Measures distance distribution with Euclidean vs depth distance
- Verifies depth distance provides better separation

---

### 12. No Hyperbolic-Like Expressiveness Tests

**Issue**: RegD claims "hyperbolic-like expressiveness" but we don't test this empirically.

**Gap**: No verification of expressiveness claim.

**Fix Needed**: Add test that:
- Measures embedding quality on hierarchical data
- Compares to hyperbolic embeddings (if available) or Euclidean baseline
- Verifies depth distance achieves similar expressiveness

---

### 13. Missing Edge Case Tests

**Issue**: New metrics lack comprehensive edge case coverage.

**Gap**: Missing tests for:
- Zero-volume boxes in depth distance
- Identical boxes in boundary distance
- Extreme hierarchies (very deep, very wide)
- Boundary cases in vector-to-box distance

**Fix Needed**: Add edge case tests for all new metrics.

---

### 14. No Integration Tests

**Issue**: No tests showing all new metrics work together.

**Gap**: Don't verify metrics are compatible in realistic scenarios.

**Fix Needed**: Add integration test that:
- Uses all new metrics together
- Verifies they produce consistent results
- Tests in realistic KG scenario

---

## Not Evaluated

### 15. Benchmarks Don't Compare to Baselines

**Issue**: Benchmarks measure our implementation speed, but don't compare to:
- Euclidean distance baseline
- Paper-reported performance
- Alternative implementations

**Gap**: Can't validate performance claims from papers.

**Fix Needed**: Add comparative benchmarks:
- Depth distance vs Euclidean on hierarchies
- Boundary distance vs containment probability
- Vector-to-box vs point-in-box checks

---

### 16. No Precision Issue Validation

**Issue**: RegD claims "eliminates precision issues of hyperbolic methods" but we don't benchmark this.

**Gap**: No validation of precision claim.

**Fix Needed**: Add benchmark that:
- Measures numerical precision/stability
- Compares to hyperbolic operations (if available)
- Verifies depth distance is more stable

---

### 17. No Joint Learning Performance Benchmarks

**Issue**: Concept2Box claims "joint learning improves performance" but we don't benchmark this.

**Gap**: No validation of performance improvement claim.

**Fix Needed**: Add benchmark that:
- Measures performance with boxes only
- Measures performance with vectors only
- Measures performance with joint learning
- Verifies joint learning improves results

---

### 18. No Regression Tests

**Issue**: No tests that ensure performance doesn't degrade over time.

**Gap**: Can't catch performance regressions.

**Fix Needed**: Add regression tests that:
- Track benchmark results over time
- Fail if performance degrades significantly
- Store baseline performance metrics

---

## Summary Table

| Category | Issue | Priority | Status |
|----------|-------|----------|--------|
| **Justified** | Generic depth_distance approximation | High | ⏳ Pending |
| **Justified** | Dual implementation strategy | Medium | ⏳ Pending |
| **Explained** | Performance claims methodology | High | ⏳ Pending |
| **Explained** | Volume vs log approximation | High | ⏳ Pending |
| **Explained** | Expressiveness validation | Medium | ⏳ Pending |
| **Exemplified** | Integrate into training loop | High | ⏳ Pending |
| **Exemplified** | Concept2Box joint learning | High | ⏳ Pending |
| **Exemplified** | RegD training example | Medium | ⏳ Pending |
| **Exemplified** | Candle examples | Medium | ⏳ Pending |
| **Tested** | Quantitative performance | High | ⏳ Pending |
| **Tested** | Crowding effect | High | ⏳ Pending |
| **Tested** | Expressiveness | Medium | ⏳ Pending |
| **Tested** | Edge cases | Medium | ⏳ Pending |
| **Tested** | Integration | Medium | ⏳ Pending |
| **Evaluated** | Baseline comparisons | High | ⏳ Pending |
| **Evaluated** | Precision validation | Medium | ⏳ Pending |
| **Evaluated** | Joint learning benchmarks | Medium | ⏳ Pending |
| **Evaluated** | Regression tests | Low | ⏳ Pending |

---

## Next Steps

1. **High Priority**: Fix justification and explanation gaps (documentation)
2. **High Priority**: Add integration examples (complete_training_loop, Concept2Box, RegD)
3. **High Priority**: Add quantitative verification tests (F1 improvement, crowding effect)
4. **Medium Priority**: Add comparative benchmarks (baselines, precision)
5. **Medium Priority**: Add edge case and integration tests

