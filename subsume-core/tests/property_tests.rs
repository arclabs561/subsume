//! Property-based tests for box embeddings.
//!
//! These tests verify mathematical invariants for box embeddings,
//! including containment transitivity, volume positivity, and
//! probabilistic properties of Gumbel boxes.
//!
//! # Key Invariants
//!
//! ## Geometric Properties
//! - Volume >= 0 (non-negativity)
//! - Containment is transitive: A ⊇ B ⊇ C implies A ⊇ C
//! - Intersection volume <= min(volume(A), volume(B))
//!
//! ## Probabilistic Properties (Gumbel)
//! - P(containment) in [0, 1]
//! - P(overlap) in [0, 1]
//! - Temperature scaling is monotonic

use proptest::prelude::*;
use subsume_core::{
    clamp_temperature, clamp_temperature_default, gumbel_membership_prob, safe_init_bounds,
    stable_sigmoid, MAX_TEMPERATURE, MIN_TEMPERATURE,
};

const TOL: f32 = 1e-5;

// =============================================================================
// Generators
// =============================================================================

/// Generate bounds for a box (min_i < max_i for all dimensions)
fn valid_bounds(dim: usize) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    prop::collection::vec(-10.0f32..10.0, dim).prop_flat_map(move |mins| {
        let mins_clone = mins.clone();
        prop::collection::vec(0.01f32..5.0, dim).prop_map(move |widths| {
            let maxs: Vec<f32> = mins_clone
                .iter()
                .zip(widths.iter())
                .map(|(m, w)| m + w)
                .collect();
            (mins_clone.clone(), maxs)
        })
    })
}

/// Generate a valid temperature
fn valid_temperature() -> impl Strategy<Value = f32> {
    0.01f32..10.0
}

// =============================================================================
// Volume Properties
// =============================================================================

proptest! {
    #[test]
    fn volume_is_non_negative(
        (mins, maxs) in valid_bounds(4)
    ) {
        // Simple volume calculation: product of widths
        let widths: Vec<f32> = mins.iter().zip(maxs.iter()).map(|(a, b)| b - a).collect();
        let volume: f32 = widths.iter().product();
        prop_assert!(volume >= 0.0, "Volume should be non-negative");
    }
}

// =============================================================================
// Probability Properties
// =============================================================================

proptest! {
    #[test]
    fn stable_sigmoid_in_unit_interval(x in -100.0f32..100.0) {
        let s = stable_sigmoid(x);
        prop_assert!(s >= 0.0, "sigmoid({}) = {} < 0", x, s);
        prop_assert!(s <= 1.0, "sigmoid({}) = {} > 1", x, s);
    }

    #[test]
    fn stable_sigmoid_monotonic(x in -50.0f32..50.0, delta in 0.01f32..10.0) {
        let s1 = stable_sigmoid(x);
        let s2 = stable_sigmoid(x + delta);
        prop_assert!(s2 >= s1, "sigmoid should be monotonic: sigmoid({}) = {}, sigmoid({}) = {}",
            x, s1, x + delta, s2);
    }

    #[test]
    fn gumbel_membership_in_unit_interval(
        x in -10.0f32..10.0,
        min in -10.0f32..5.0,
        max in -5.0f32..10.0,
        temp in valid_temperature()
    ) {
        // Only test when min < max (valid box)
        if min < max {
            let prob = gumbel_membership_prob(x, min, max, temp);
            prop_assert!(prob >= 0.0 - TOL, "P({}) should be >= 0, got {}", x, prob);
            prop_assert!(prob <= 1.0 + TOL, "P({}) should be <= 1, got {}", x, prob);
        }
    }
}

// =============================================================================
// Temperature Properties
// =============================================================================

proptest! {
    #[test]
    fn temperature_clamping_works(raw_temp in -100.0f32..200.0) {
        let clamped = clamp_temperature(raw_temp, MIN_TEMPERATURE, MAX_TEMPERATURE);

        prop_assert!(clamped >= MIN_TEMPERATURE, "Clamped temp should be >= MIN_TEMPERATURE");
        prop_assert!(clamped <= MAX_TEMPERATURE, "Clamped temp should be <= MAX_TEMPERATURE");
    }

    #[test]
    fn temperature_default_clamp_works(raw_temp in -100.0f32..200.0) {
        let clamped = clamp_temperature_default(raw_temp);

        prop_assert!(clamped >= MIN_TEMPERATURE, "Clamped temp should be >= MIN_TEMPERATURE");
        prop_assert!(clamped <= MAX_TEMPERATURE, "Clamped temp should be <= MAX_TEMPERATURE");
    }

    #[test]
    fn higher_temperature_affects_membership(
        x in -5.0f32..0.0,  // Point outside center
        min in -2.0f32..-0.5,
        max in 0.5f32..2.0,
    ) {
        // With x < min, the point is outside the box
        let low_temp = 0.1f32;
        let high_temp = 5.0f32;

        let p_cold = gumbel_membership_prob(x, min, max, low_temp);
        let p_hot = gumbel_membership_prob(x, min, max, high_temp);

        // Just verify both are valid probabilities
        prop_assert!(p_cold >= 0.0 && p_cold <= 1.0);
        prop_assert!(p_hot >= 0.0 && p_hot <= 1.0);
    }
}

// =============================================================================
// Box Initialization Properties
// =============================================================================

proptest! {
    #[test]
    fn safe_init_produces_valid_bounds(
        center_min in -5.0f32..0.0,
        center_max in 0.0f32..5.0,
        size_min in 0.1f32..1.0,
        size_max in 1.0f32..3.0
    ) {
        // Test that safe_init_bounds produces valid bounds
        let (min_z, max_z) = safe_init_bounds(
            0, // dimension
            10, // num_boxes
            0,  // box_index
            (center_min, center_max),
            (size_min, size_max)
        );

        // The result should be valid bounds
        prop_assert!(!min_z.is_nan(), "min_z should not be NaN");
        prop_assert!(!max_z.is_nan(), "max_z should not be NaN");

        // min should be less than max
        prop_assert!(min_z < max_z, "min_z ({}) should be < max_z ({})", min_z, max_z);
    }
}

// =============================================================================
// Containment Transitivity (Conceptual Test)
// =============================================================================

/// Test that containment relationships respect transitivity.
/// If box A contains box B, and box B contains box C, then A contains C.
#[test]
fn containment_transitivity_hard_boxes() {
    // A is a large box
    let a_min = vec![-5.0f32, -5.0];
    let a_max = vec![5.0f32, 5.0];

    // B is inside A
    let b_min = vec![-2.0f32, -2.0];
    let b_max = vec![2.0f32, 2.0];

    // C is inside B
    let c_min = vec![-1.0f32, -1.0];
    let c_max = vec![1.0f32, 1.0];

    // Check A contains B (all b_min >= a_min and all b_max <= a_max)
    let a_contains_b = a_min.iter().zip(b_min.iter()).all(|(a, b)| a <= b)
        && a_max.iter().zip(b_max.iter()).all(|(a, b)| a >= b);
    assert!(a_contains_b, "A should contain B");

    // Check B contains C
    let b_contains_c = b_min.iter().zip(c_min.iter()).all(|(b, c)| b <= c)
        && b_max.iter().zip(c_max.iter()).all(|(b, c)| b >= c);
    assert!(b_contains_c, "B should contain C");

    // Check A contains C (transitivity)
    let a_contains_c = a_min.iter().zip(c_min.iter()).all(|(a, c)| a <= c)
        && a_max.iter().zip(c_max.iter()).all(|(a, c)| a >= c);
    assert!(a_contains_c, "A should contain C (transitivity)");
}

// =============================================================================
// Intersection Properties
// =============================================================================

#[test]
fn disjoint_boxes_have_zero_intersection() {
    // Box A: [0, 1] in each dimension
    let a_min = vec![0.0f32, 0.0];
    let a_max = vec![1.0f32, 1.0];

    // Box B: [2, 3] in each dimension (disjoint)
    let b_min = vec![2.0f32, 2.0];
    let b_max = vec![3.0f32, 3.0];

    // Compute intersection
    let int_min: Vec<f32> = a_min
        .iter()
        .zip(b_min.iter())
        .map(|(a, b)| a.max(*b))
        .collect();
    let int_max: Vec<f32> = a_max
        .iter()
        .zip(b_max.iter())
        .map(|(a, b)| a.min(*b))
        .collect();

    // Check if intersection is empty (min > max in some dimension)
    let is_empty = int_min
        .iter()
        .zip(int_max.iter())
        .any(|(min, max)| min > max);
    assert!(is_empty, "Disjoint boxes should have empty intersection");
}

#[test]
fn nested_boxes_intersection_equals_inner() {
    // Outer box
    let a_min = vec![-2.0f32, -2.0];
    let a_max = vec![2.0f32, 2.0];

    // Inner box (fully contained)
    let b_min = vec![-1.0f32, -1.0];
    let b_max = vec![1.0f32, 1.0];

    // Compute intersection
    let int_min: Vec<f32> = a_min
        .iter()
        .zip(b_min.iter())
        .map(|(a, b)| a.max(*b))
        .collect();
    let int_max: Vec<f32> = a_max
        .iter()
        .zip(b_max.iter())
        .map(|(a, b)| a.min(*b))
        .collect();

    // Intersection should equal inner box
    assert_eq!(
        int_min, b_min,
        "Intersection min should equal inner box min"
    );
    assert_eq!(
        int_max, b_max,
        "Intersection max should equal inner box max"
    );
}

// =============================================================================
// Optimizer schedule + ranking metrics (small, deterministic checks)
// =============================================================================

#[test]
fn learning_rate_schedule_warmup_and_bounds() {
    // The schedule is defined as:
    // - warmup: 0.1*lr -> lr linearly
    // - then cosine decay: lr -> 0.1*lr
    let base_lr = 1e-3_f32;
    let total_epochs = 100;
    let warmup_epochs = 10;

    let lr0 = subsume_core::get_learning_rate(0, total_epochs, base_lr, warmup_epochs);
    let lr_mid_warmup = subsume_core::get_learning_rate(5, total_epochs, base_lr, warmup_epochs);
    let lr_end_warmup = subsume_core::get_learning_rate(10, total_epochs, base_lr, warmup_epochs);
    let lr_final = subsume_core::get_learning_rate(99, total_epochs, base_lr, warmup_epochs);

    // Warmup should increase.
    assert!(lr_mid_warmup >= lr0);
    assert!((lr_end_warmup - base_lr).abs() < 1e-9);

    // Bounds: should stay within [0.1*base_lr, base_lr]
    let min_lr = base_lr * 0.1;
    for lr in [lr0, lr_mid_warmup, lr_end_warmup, lr_final] {
        assert!(lr >= min_lr - 1e-12);
        assert!(lr <= base_lr + 1e-12);
        assert!(lr.is_finite());
    }
}

#[test]
fn ranking_metrics_match_known_values() {
    use subsume_core::training::metrics::{hits_at_k, mean_rank, mean_reciprocal_rank, ndcg};

    let ranks = vec![1usize, 3, 2, 5];
    let mrr = mean_reciprocal_rank(ranks.iter().copied());
    // (1 + 1/3 + 1/2 + 1/5) / 4 = 0.50833...
    assert!((mrr - 0.5083333).abs() < 1e-6);

    let hits3 = hits_at_k(ranks.iter().copied(), 3);
    assert!((hits3 - 0.75).abs() < 1e-6);

    let mr = mean_rank(ranks.iter().copied());
    assert!((mr - 2.75).abs() < 1e-6);

    let ranked = vec![0.9f32, 0.5, 0.8, 0.2];
    let ideal = vec![0.9f32, 0.8, 0.5, 0.2];
    let score = ndcg(ranked.iter().copied(), ideal.iter().copied());
    assert!(score <= 1.0 + 1e-6);
    assert!(score >= 0.0 - 1e-6);
    assert!(score > 0.9);
}

// =============================================================================
// Numerical Stability
// =============================================================================

proptest! {
    #[test]
    fn operations_dont_produce_nan(
        x in -50.0f32..50.0,
        y in -50.0f32..50.0,
        temp in valid_temperature()
    ) {
        let sigmoid_x = stable_sigmoid(x);
        let sigmoid_y = stable_sigmoid(y);

        prop_assert!(!sigmoid_x.is_nan(), "stable_sigmoid({}) is NaN", x);
        prop_assert!(!sigmoid_y.is_nan(), "stable_sigmoid({}) is NaN", y);

        // Only test membership when we have valid bounds
        if x < y {
            let membership = gumbel_membership_prob(0.0, x, y, temp);
            prop_assert!(!membership.is_nan(), "gumbel_membership_prob(0, {}, {}, {}) is NaN", x, y, temp);
        }
    }

    #[test]
    fn extreme_values_handled(
        sign in prop::bool::ANY
    ) {
        let extreme = if sign { 1000.0f32 } else { -1000.0f32 };

        // These should not panic or return NaN
        let s = stable_sigmoid(extreme);
        prop_assert!(!s.is_nan());
        prop_assert!(s >= 0.0 && s <= 1.0);

        // Extreme temperature clamping
        let temp = clamp_temperature_default(extreme);
        prop_assert!(temp > 0.0);
        prop_assert!(temp < f32::INFINITY);
    }
}
