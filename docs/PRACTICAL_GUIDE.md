# Practical Guide to Box Embeddings

This guide provides practical insights, common pitfalls, and nuanced considerations when working with box embeddings that go beyond the mathematical foundations.

## Understanding the Local Identifiability Problem

The local identifiability problem is the most critical theoretical issue that affects practical training. Understanding it deeply helps explain many training behaviors.

### What It Really Means

**Local identifiability** means that for any parameter configuration, there exists a neighborhood where all parameters produce different loss values. When this fails, you get "flat regions" in the loss landscape where many parameter configurations yield identical loss.

### Why This Matters Practically

1. **Gradient Descent Stalls**: When you're in a flat region, gradients are zero or near-zero, so the optimizer has no direction to move. The model appears to converge, but it's actually stuck in a degenerate configuration.

2. **Multiple Solutions**: The same training data can be satisfied by many different box configurations. For example, if box A should contain box B, you could:
   - Make A very large and B small
   - Make A slightly larger than B
   - Make A and B the same size but A shifted to contain B
   
   All of these satisfy the constraint, but they're not equally good for generalization.

3. **Volume Slackness**: This is a specific manifestation where boxes can grow arbitrarily large while still satisfying containment constraints. The loss doesn't penalize unnecessary volume, so boxes "cheat" by becoming huge.

### How Gumbel Boxes Solve This

Gumbel boxes don't eliminate the problem entirely, but they transform it:

- **Before**: Hard boxes have discrete "on/off" containment. Two boxes are either disjoint (zero gradient) or overlapping (gradient exists). This creates sharp boundaries.

- **After**: Gumbel boxes make containment probabilistic and continuous. Even when boxes are far apart, there's a small probability of overlap, which means there's always a non-zero gradient (though it may be very small).

The key insight: **Gumbel boxes ensure that every parameter affects the expected loss**, even if the effect is small. This restores local identifiability in the probabilistic sense.

## Temperature: The Critical Hyperparameter

Temperature controls the "hardness" of Gumbel boxes and is one of the most important hyperparameters.

### What Temperature Actually Does

Temperature `τ` controls the variance of the Gumbel distribution:
- **Low temperature** (τ → 0): Boxes become "hard" with crisp boundaries. Approaches deterministic boxes.
- **High temperature** (τ → ∞): Boxes become "soft" with fuzzy boundaries. Approaches uniform distribution.

### The Trade-off

**Low Temperature**:
- ✅ Better correspondence to discrete containment relationships
- ✅ More interpretable (boxes have clear boundaries)
- ❌ Can reintroduce gradient sparsity (approaches hard box behavior)
- ❌ More sensitive to initialization

**High Temperature**:
- ✅ Smoother gradients throughout training
- ✅ More robust to initialization
- ❌ Less interpretable (fuzzy boundaries)
- ❌ May not converge to desired containment relationships

### Temperature Annealing Strategy

The standard approach is to start with higher temperature and anneal down:

1. **Early Training** (high τ): Smooth loss landscape, easier optimization
2. **Mid Training** (medium τ): Balance between smoothness and precision
3. **Late Training** (low τ): Sharp boundaries, precise containment

This is implemented in `temperature_scheduler()` which uses exponential decay:
```
τ(t) = max(τ₀ × decay^t, τ_min)
```

### Practical Recommendations

- **Start with τ = 1.0-2.0** for most tasks
- **Decay rate of 0.95-0.99** per epoch/step
- **Minimum temperature of 0.01-0.1** to maintain some smoothness
- **Monitor gradient norms**: If they drop to near-zero, temperature might be too low

## Volume Regularization: Preventing the "Cheating" Problem

Volume regularization is essential for preventing boxes from becoming arbitrarily large.

### Why Boxes "Cheat"

Consider a containment constraint: box A should contain box B. The loss only cares about containment, not box size. So the model can satisfy this by:
- Making A huge (covers everything)
- Making B tiny (fits anywhere)

This satisfies the constraint but loses all semantic meaning.

### Regularization Strategies

**Upper Bound Regularization**:
```rust
L_reg = λ × max(0, Vol(B) - V_max)
```
Penalizes boxes larger than a threshold. This prevents boxes from becoming too large.

**Lower Bound Regularization**:
```rust
L_reg = λ × max(0, V_min - Vol(B))
```
Penalizes boxes smaller than a threshold. This prevents boxes from collapsing to points.

**Combined** (as in `volume_regularization()`):
```rust
L_reg = λ × [max(0, Vol - V_max) + max(0, V_min - Vol)]
```

### Choosing Thresholds

- **V_max**: Should be based on your embedding space. If boxes are in [0,1]^d, a reasonable V_max might be 0.5^d (half the space).
- **V_min**: Should prevent collapse. A reasonable V_min might be 0.01^d (1% of the space).
- **λ**: Start with 0.01-0.1 and adjust based on whether boxes are too large or too small.

### Monitoring Volume Distribution

Use `VolumeDistribution` to track:
- **Entropy**: High entropy = diverse box sizes (good). Low entropy = all boxes similar size (might indicate collapse or uniform growth).
- **Coefficient of Variation (CV)**: CV > 1 suggests good diversity. CV < 0.5 suggests boxes are too uniform.
- **Quantiles**: Check if Q95 is much larger than Q5, indicating some boxes are much larger than others.

## Initialization: Setting Up for Success

Initialization is critical because box embeddings are sensitive to starting configurations.

### The Problem with Random Initialization

Random initialization can create problematic configurations:
1. **All boxes overlapping**: Everything is contained in everything, no structure
2. **All boxes disjoint**: Nothing is contained in anything, can't learn relationships
3. **Perfect nesting**: Some boxes perfectly nested, creating local identifiability issues

### Better Initialization Strategies

**Separation-Based Initialization**:
- Initialize boxes with minimum separation distance
- Ensures initial structure without perfect nesting
- Use `suggest_separation_distance()` to find appropriate separation

**Cross-Pattern Detection**:
- Detect if initialization creates problematic patterns (perfect nesting, complete overlap)
- Use `detect_cross_patterns()` to identify issues
- Re-initialize if patterns detected

**Volume-Aware Initialization**:
- Initialize boxes with diverse volumes
- Prevents all boxes from starting at the same size
- Helps establish hierarchy early

### Practical Initialization Code

```rust
use subsume_core::utils::{suggest_separation_distance, detect_cross_patterns};

// 1. Initialize boxes with separation
let separation = suggest_separation_distance(dimension, num_boxes);
for i in 0..num_boxes {
    let min = random_point_with_separation(separation);
    let max = min + random_size(min_vol, max_vol);
    boxes.push(NdarrayBox::new(min, max, 1.0)?);
}

// 2. Check for problematic patterns
if detect_cross_patterns(&boxes, 0.9) {
    // Re-initialize or adjust
}
```

## Numerical Stability: Hidden Pitfalls

Box embeddings involve many numerical operations that can fail silently.

### Log-Space Volume Computation

In high dimensions, volume computation can underflow:
```
Vol = ∏(max[i] - min[i])
```

If you have 20 dimensions with side length 0.5:
```
0.5^20 ≈ 9.5 × 10^-7
```

This can underflow to 0.0 in f32! Use `log_space_volume()` which computes:
```
log(Vol) = Σ log(max[i] - min[i])
```

### Stable Sigmoid

The standard sigmoid `1/(1+exp(-x))` can overflow when x is large:
- For x > 88, exp(-x) underflows to 0, sigmoid → 1.0
- For x < -88, exp(-x) overflows to inf, sigmoid → 0.0

Use `stable_sigmoid()` which handles these cases:
- For x ≥ 0: `1/(1+exp(-x))` (avoids exp overflow)
- For x < 0: `exp(x)/(1+exp(x))` (avoids exp(-x) overflow)

### Gumbel Sampling

Gumbel sampling involves `-ln(-ln(U))` which can fail:
- If U ≈ 0, ln(U) → -∞, then -ln(-∞) is undefined
- If U ≈ 1, ln(U) ≈ 0, then -ln(0) → ∞

Always clamp U to [ε, 1-ε] before sampling, as done in `sample_gumbel()`.

### Temperature Clamping

Very low temperatures can cause:
- Division by zero in `(x - min) / τ`
- Exponential overflow in sigmoid calculations
- Vanishing gradients

Always use `clamp_temperature()` to keep temperature in safe range [1e-3, 10.0].

## Training Diagnostics: What to Monitor

The training diagnostics provide rich information, but knowing what to look for is key.

### Early Training (First 10-20% of epochs)

**What to Check**:
- **Gradient Flow**: Are gradients flowing? Check `GradientFlowAnalysis` for sparsity < 50%
- **Phase Detection**: Should be in `Exploration` phase
- **Volume Distribution**: Entropy should be increasing (boxes diversifying)

**Red Flags**:
- Gradient sparsity > 80%: Model isn't learning
- All volumes collapsing to near-zero: Need lower bound regularization
- All volumes growing unbounded: Need upper bound regularization

### Mid Training (20-80% of epochs)

**What to Check**:
- **Convergence**: `TrainingStats::is_converged()` should eventually return true
- **Loss Components**: Are they balanced? Use `LossComponents::is_imbalanced()`
- **Depth-Stratified Gradients**: Are all hierarchy levels learning? Check `DepthStratifiedGradientFlow`

**Red Flags**:
- Loss components imbalanced: One loss term dominating (e.g., regularization >> containment)
- Depth imbalance: Only root level learning, deeper levels stuck
- Phase stuck in `Exploration`: Model not finding structure

### Late Training (Last 20% of epochs)

**What to Check**:
- **Phase**: Should be in `Convergence` or `Exploitation`
- **Volume Conservation**: Parent volumes should contain children (ratio < 1.0)
- **Generalization Gap**: Inference performance vs direct facts should be similar

**Red Flags**:
- Phase is `Instability`: Loss/gradients oscillating, might need to stop early
- Volume conservation violations: Hierarchy structure breaking down
- Large generalization gap: Model memorizing, not learning structure

## Common Mistakes and How to Avoid Them

### Mistake 1: Ignoring Volume Regularization

**Symptom**: Boxes become huge or collapse to points
**Solution**: Always use volume regularization with appropriate thresholds

### Mistake 2: Temperature Too Low Too Early

**Symptom**: Training stalls, gradients vanish
**Solution**: Start with higher temperature, anneal gradually

### Mistake 3: Poor Initialization

**Symptom**: Model converges to degenerate solution
**Solution**: Use separation-based initialization, check for cross-patterns

### Mistake 4: Not Monitoring Intersection Volumes

**Symptom**: Containment relationships don't improve
**Solution**: Track `intersection_volume_stats()` to see if relationships are actually evolving

### Mistake 5: Ignoring Calibration

**Symptom**: High containment probabilities but poor actual containment
**Solution**: Monitor `expected_calibration_error()` - probabilities should match reality

### Mistake 6: Single Metric Focus

**Symptom**: MRR looks good but model fails on other tasks
**Solution**: Use stratified metrics to check performance across relation types, hierarchy depths, frequency bands

## Advanced Techniques

### Joint Training with Multiple Objectives

Box embeddings often need to satisfy multiple constraints:
- Containment relationships (hierarchical)
- Overlap relationships (related but distinct)
- Disjoint relationships (mutually exclusive)

Use weighted combination:
```rust
L_total = λ₁×L_containment + λ₂×L_overlap + λ₃×L_disjoint + λ₄×L_regularization
```

Monitor `LossComponents` to ensure balance.

### Hierarchical Constraints

For knowledge graphs with hierarchies, enforce volume conservation:
```rust
// Parent volume should contain sum of children
if parent_vol < sum(children_vols) {
    L += λ × (sum(children_vols) - parent_vol)
}
```

Use `VolumeConservation` to track and enforce this.

### Relation-Specific Learning Rates

Different relation types may need different learning rates:
- `is_a` (hierarchical): May need slower learning to maintain structure
- `has_part` (meronymic): May need faster learning

Use `RelationStratifiedTrainingStats` to monitor convergence per relation type, then adjust learning rates accordingly.

## Debugging Training Issues

### Problem: Model Not Learning

**Check**:
1. Gradient sparsity: Should be < 50%
2. Temperature: Should be > 0.1
3. Initialization: Check for cross-patterns
4. Loss components: Are they non-zero?

**Fix**: Increase temperature, re-initialize with separation, check data

### Problem: Boxes Collapsing

**Check**:
1. Volume distribution: Is min volume → 0?
2. Lower bound regularization: Is it too weak?
3. Temperature: Is it too high? (soft boxes can collapse)

**Fix**: Increase lower bound regularization, decrease temperature

### Problem: Boxes Growing Unbounded

**Check**:
1. Volume distribution: Is max volume → ∞?
2. Upper bound regularization: Is it too weak?
3. Volume slackness: Are boxes "cheating"?

**Fix**: Increase upper bound regularization, add volume penalty to loss

### Problem: Poor Generalization

**Check**:
1. Generalization gap: Is inference performance much worse than direct?
2. Calibration: Are probabilities calibrated?
3. Stratified metrics: Are some relation types failing?

**Fix**: Increase regularization, check for overfitting, use more diverse training data

## Best Practices Summary

1. **Always use volume regularization** with both upper and lower bounds
2. **Start with higher temperature** and anneal down gradually
3. **Initialize with separation** to avoid problematic patterns
4. **Monitor multiple diagnostics** throughout training, not just loss
5. **Use log-space volume** for high-dimensional boxes
6. **Track intersection volumes** to see if relationships are actually improving
7. **Check calibration** - probabilities should match reality
8. **Use stratified evaluation** to catch performance issues in specific subsets
9. **Enforce volume conservation** for hierarchical structures
10. **Balance loss components** - no single term should dominate

## References

- Dasgupta et al. (2020): "Improving Local Identifiability in Probabilistic Box Embeddings" - Deep dive into the identifiability problem
- Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures" - Original framework
- See `docs/MATHEMATICAL_FOUNDATIONS.md` for theoretical details

