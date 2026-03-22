//! Property-based tests for Gumbel box invariants.
//!
//! Tests mathematical properties of NdarrayGumbelBox: volume non-negativity,
//! intersection sandwich bounds, containment reflexivity/monotonicity,
//! softplus continuity, and degenerate-box safety.

#![cfg(feature = "ndarray-backend")]

use ndarray::Array1;
use proptest::prelude::*;
use subsume::ndarray_backend::NdarrayGumbelBox;
use subsume::utils::{gumbel_lse_max, gumbel_lse_min, softplus};
use subsume::Box as BoxTrait;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a GumbelBox from raw vecs (convenience for proptest strategies).
fn make_gumbel(mins: Vec<f32>, maxs: Vec<f32>, temp: f32) -> NdarrayGumbelBox {
    NdarrayGumbelBox::new(Array1::from(mins), Array1::from(maxs), temp).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Volume non-negativity
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn volume_non_negative(
        dim in 2usize..=64,
        temp in 0.1f32..50.0f32,
        seed in 0u64..10000,
    ) {
        // Generate random box coordinates deterministically from seed
        let mut rng = seed;
        let lcg = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*s >> 33) as f32) / (u32::MAX as f32) * 200.0 - 100.0
        };

        let mut mins = Vec::with_capacity(dim);
        let mut maxs = Vec::with_capacity(dim);
        for _ in 0..dim {
            let a = lcg(&mut rng);
            let b = lcg(&mut rng);
            mins.push(a.min(b));
            maxs.push(a.max(b));
        }

        let bx = make_gumbel(mins, maxs, temp);
        let vol = bx.volume().unwrap();
        prop_assert!(vol >= 0.0, "volume must be >= 0, got {vol}");
        // Volume can be +inf for high-dim boxes with low temperature (Bessel
        // side lengths sum to large log-volume that overflows f32::exp).
        // The invariant is: non-negative and not NaN.
        prop_assert!(!vol.is_nan(), "volume must not be NaN");
    }
}

// ---------------------------------------------------------------------------
// 2. Intersection sandwich invariant
// ---------------------------------------------------------------------------
//
// For any two boxes A, B with temperature T:
//   gumbel_lse_min(z_a, z_b, T) >= max(z_a, z_b)   (soft min-coord >= hard min-coord)
//   gumbel_lse_max(Z_a, Z_b, T) <= min(Z_a, Z_b)   (soft max-coord <= hard max-coord)
//
// The Gumbel intersection is always contained within the hard intersection.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn intersection_sandwich(
        dim in 2usize..=32,
        temp in 0.1f32..10.0f32,
        seed in 0u64..10000,
    ) {
        let mut rng = seed;
        let lcg = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*s >> 33) as f32) / (u32::MAX as f32) * 200.0 - 100.0
        };

        let mut mins_a = Vec::with_capacity(dim);
        let mut maxs_a = Vec::with_capacity(dim);
        let mut mins_b = Vec::with_capacity(dim);
        let mut maxs_b = Vec::with_capacity(dim);
        for _ in 0..dim {
            let a1 = lcg(&mut rng);
            let a2 = lcg(&mut rng);
            mins_a.push(a1.min(a2));
            maxs_a.push(a1.max(a2));

            let b1 = lcg(&mut rng);
            let b2 = lcg(&mut rng);
            mins_b.push(b1.min(b2));
            maxs_b.push(b1.max(b2));
        }

        let a = make_gumbel(mins_a.clone(), maxs_a.clone(), temp);
        let b = make_gumbel(mins_b.clone(), maxs_b.clone(), temp);
        let inter: NdarrayGumbelBox = a.intersection(&b).unwrap();

        for d in 0..dim {
            let hard_min = mins_a[d].max(mins_b[d]);
            let hard_max = maxs_a[d].min(maxs_b[d]);
            let soft_min = inter.min()[d];
            let soft_max = inter.max()[d];

            // Soft min >= hard min (LSE is a smooth upper bound on max)
            prop_assert!(
                soft_min >= hard_min - 1e-4,
                "dim {d}: soft_min {soft_min} < hard_min {hard_min}"
            );

            // Soft max <= hard max (negative LSE is a smooth lower bound on min)
            prop_assert!(
                soft_max <= hard_max + 1e-4,
                "dim {d}: soft_max {soft_max} > hard_max {hard_max}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Containment reflexivity
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn containment_reflexivity(
        dim in 2usize..=16,
        seed in 0u64..10000,
    ) {
        let mut rng = seed;
        let lcg = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*s >> 33) as f32) / (u32::MAX as f32) * 100.0 - 50.0
        };

        let mut mins = Vec::with_capacity(dim);
        let mut maxs = Vec::with_capacity(dim);
        for _ in 0..dim {
            let a = lcg(&mut rng);
            let width = ((lcg(&mut rng)).abs() % 50.0) + 5.0;
            mins.push(a);
            maxs.push(a + width);
        }

        // Core invariant: self-containment at low T should be near 1.0.
        // The Bessel approximation applies a 2*gamma*T offset per dimension,
        // so vol(A cap A) < vol(A) when T > 0. This shrinkage compounds
        // across dimensions. Use low T to verify the reflexivity property.
        let temp_lo = 0.01f32;
        let bx_lo = make_gumbel(mins.clone(), maxs.clone(), temp_lo);
        let sc_lo: f32 = bx_lo.containment_prob(&bx_lo).unwrap();
        prop_assert!(
            sc_lo > 0.5,
            "self-containment at T=0.01 should be > 0.5, got {sc_lo} (dim={dim})"
        );

        // Monotonicity: lowering temperature should increase self-containment.
        let temp_hi = 0.5f32;
        let bx_hi = make_gumbel(mins, maxs, temp_hi);
        let sc_hi: f32 = bx_hi.containment_prob(&bx_hi).unwrap();
        prop_assert!(
            sc_lo >= sc_hi - 1e-5,
            "self-containment should increase as T decreases: sc(T=0.01)={sc_lo} < sc(T=0.5)={sc_hi}"
        );

        // Both must be non-NaN and in [0, 1]
        prop_assert!(!sc_lo.is_nan());
        prop_assert!(!sc_hi.is_nan());
        prop_assert!((0.0..=1.0).contains(&sc_lo));
        prop_assert!((0.0..=1.0).contains(&sc_hi));
    }
}

// ---------------------------------------------------------------------------
// 4. Containment monotonicity
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn containment_monotonicity(
        dim in 2usize..=8,
        temp in 0.1f32..0.5f32,
        seed in 0u64..10000,
    ) {
        let mut rng = seed;
        let lcg_pos = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*s >> 33) as f32) / (u32::MAX as f32) * 10.0 + 1.0
        };
        let lcg_base = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*s >> 33) as f32) / (u32::MAX as f32) * 40.0 - 20.0
        };

        let mut outer_min = Vec::with_capacity(dim);
        let mut outer_max = Vec::with_capacity(dim);
        let mut inner_min = Vec::with_capacity(dim);
        let mut inner_max = Vec::with_capacity(dim);

        for _ in 0..dim {
            let om = lcg_base(&mut rng);
            let margin_lo = lcg_pos(&mut rng);
            let inner_width = lcg_pos(&mut rng);
            let margin_hi = lcg_pos(&mut rng);

            let im = om + margin_lo;
            let ix = im + inner_width;
            let ox = ix + margin_hi;

            outer_min.push(om);
            inner_min.push(im);
            inner_max.push(ix);
            outer_max.push(ox);
        }

        let outer = make_gumbel(outer_min, outer_max, temp);
        let inner = make_gumbel(inner_min, inner_max, temp);

        // outer strictly contains inner => P(outer contains inner) should be high
        let prob: f32 = outer.containment_prob(&inner).unwrap();
        prop_assert!(
            prob > 0.5,
            "P(outer contains inner) should be > 0.5, got {prob} (dim={dim}, temp={temp})"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. softplus branch continuity
// ---------------------------------------------------------------------------

#[test]
fn softplus_branch_boundaries() {
    let beta = 1.0f32;

    // Transition at bx = 20.0 => x = 20.0 for beta=1
    let at_boundary = softplus(20.0, beta);
    let just_below = softplus(19.99, beta);
    let just_above = softplus(20.01, beta);

    // Continuity: values near the boundary should be close
    assert!(
        (at_boundary - just_below).abs() < 0.05,
        "discontinuity at upper branch: {at_boundary} vs {just_below}"
    );
    assert!(
        (at_boundary - just_above).abs() < 0.05,
        "discontinuity at upper branch: {at_boundary} vs {just_above}"
    );

    // Lower branch at bx = -20.0 => x = -20.0 for beta=1
    let at_lower = softplus(-20.0, beta);
    let just_below_lower = softplus(-20.01, beta);
    let just_above_lower = softplus(-19.99, beta);

    assert!(
        (at_lower - just_below_lower).abs() < 0.05,
        "discontinuity at lower branch: {at_lower} vs {just_below_lower}"
    );
    assert!(
        (at_lower - just_above_lower).abs() < 0.05,
        "discontinuity at lower branch: {at_lower} vs {just_above_lower}"
    );

    // Extreme inputs: must not panic or produce NaN/Inf
    let extreme_neg = softplus(-1000.0, 1.0);
    assert!(extreme_neg.is_finite(), "softplus(-1000) = {extreme_neg}");
    assert!(extreme_neg >= 0.0);

    let extreme_pos = softplus(1000.0, 1.0);
    assert!(extreme_pos.is_finite(), "softplus(1000) = {extreme_pos}");
    assert!((extreme_pos - 1000.0).abs() < 0.01);

    let at_zero = softplus(0.0, 1.0);
    assert!(
        (at_zero - std::f32::consts::LN_2).abs() < 0.01,
        "softplus(0) should be ln(2) ~ 0.693, got {at_zero}"
    );
}

#[test]
fn softplus_extreme_beta() {
    // High beta: sharp transition
    let high_beta = softplus(0.5, 100.0);
    assert!(high_beta.is_finite());
    assert!(high_beta >= 0.0);

    // Low beta: gentle slope
    let low_beta = softplus(0.5, 0.01);
    assert!(low_beta.is_finite());
    assert!(low_beta >= 0.0);
}

// ---------------------------------------------------------------------------
// 6. Zero-volume degenerate box (min == max)
// ---------------------------------------------------------------------------

#[test]
fn degenerate_box_no_panic() {
    // min == max in all dimensions
    let mins = Array1::from(vec![1.0, 2.0, 3.0]);
    let maxs = Array1::from(vec![1.0, 2.0, 3.0]);
    let bx = NdarrayGumbelBox::new(mins, maxs, 1.0).unwrap();

    // Volume should not be NaN, and should be near zero (softplus gives
    // a small positive value due to the Bessel approximation offset)
    let vol = bx.volume().unwrap();
    assert!(!vol.is_nan(), "degenerate box volume is NaN");
    assert!(
        vol.is_finite(),
        "degenerate box volume is not finite: {vol}"
    );
    assert!(vol >= 0.0, "degenerate box volume is negative: {vol}");

    // Intersection with self should not panic
    let inter: NdarrayGumbelBox = bx.intersection(&bx).unwrap();
    let inter_vol = inter.volume().unwrap();
    assert!(!inter_vol.is_nan(), "degenerate intersection volume is NaN");
    assert!(inter_vol.is_finite());

    // Containment prob should not panic or produce NaN
    let cp: f32 = bx.containment_prob(&bx).unwrap();
    assert!(!cp.is_nan(), "degenerate self-containment is NaN");
}

#[test]
fn degenerate_box_high_dim() {
    // 64-dimensional degenerate box
    let val: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let mins = Array1::from(val.clone());
    let maxs = Array1::from(val);
    let bx = NdarrayGumbelBox::new(mins, maxs, 0.5).unwrap();

    let vol = bx.volume().unwrap();
    assert!(!vol.is_nan(), "high-dim degenerate volume is NaN");
    assert!(
        vol.is_finite(),
        "high-dim degenerate volume is not finite: {vol}"
    );
}

// ---------------------------------------------------------------------------
// Supplementary: gumbel_lse_min/max element-wise sandwich (standalone)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// gumbel_lse_min(a, b, T) >= max(a, b) for all finite a, b, T > 0.
    #[test]
    fn lse_min_ge_hard_max(
        a in -100.0f32..100.0f32,
        b in -100.0f32..100.0f32,
        t in 0.01f32..10.0f32,
    ) {
        let soft = gumbel_lse_min(a, b, t);
        let hard = a.max(b);
        prop_assert!(
            soft >= hard - 1e-5,
            "gumbel_lse_min({a}, {b}, {t}) = {soft} < max = {hard}"
        );
    }

    /// gumbel_lse_max(a, b, T) <= min(a, b) for all finite a, b, T > 0.
    #[test]
    fn lse_max_le_hard_min(
        a in -100.0f32..100.0f32,
        b in -100.0f32..100.0f32,
        t in 0.01f32..10.0f32,
    ) {
        let soft = gumbel_lse_max(a, b, t);
        let hard = a.min(b);
        prop_assert!(
            soft <= hard + 1e-5,
            "gumbel_lse_max({a}, {b}, {t}) = {soft} > min = {hard}"
        );
    }
}
