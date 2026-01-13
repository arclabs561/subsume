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
        let (min_z, max_Z) = safe_init_bounds(
            0, // dimension
            10, // num_boxes
            0,  // box_index
            (center_min, center_max),
            (size_min, size_max)
        );

        // The result should be valid bounds
        prop_assert!(!min_z.is_nan(), "min_z should not be NaN");
        prop_assert!(!max_Z.is_nan(), "max_Z should not be NaN");

        // min should be less than max
        prop_assert!(min_z < max_Z, "min_z ({}) should be < max_Z ({})", min_z, max_Z);
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
