//! Regression tests for fixed numerical bugs. Each test states the correct
//! behavior and the reference it comes from.

use std::f32::consts::PI;
use subsume::spherical_cap::SphericalCap;

/// P(B ⊆ B) is 1 for any box with positive volume. An 8-d box with side
/// 1e-6 has volume 1e-48, below the smallest positive f32, so the ratio must
/// not be formed from the two volumes directly.
#[cfg(feature = "ndarray-backend")]
#[test]
fn containment_prob_of_tiny_box_in_itself_is_one() {
    use subsume::ndarray_backend::NdarrayBox;
    use subsume::Box as _;
    for d in [2usize, 8, 10] {
        let b = NdarrayBox::new(
            subsume::ndarray::Array1::zeros(d),
            subsume::ndarray::Array1::from_elem(d, 1e-6),
            1.0,
        )
        .unwrap();
        let p = b.containment_prob(&b);
        assert!(
            matches!(p, Ok(v) if (v - 1.0).abs() < 1e-6),
            "d = {d}: P(b ⊆ b) = {p:?}"
        );
        let mut out = [0.0f32];
        b.containment_prob_many(std::slice::from_ref(&b), &mut out)
            .unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6, "d = {d}: many = {}", out[0]);
    }
}

/// Cap area fraction on S^{d-1}: (1/2) I_{sin²θ}((d-1)/2, 1/2) for θ <= π/2
/// and one minus that for θ > π/2. Reference values from
/// `scipy.special.betainc`.
#[test]
fn spherical_cap_area_fraction_matches_incomplete_beta() {
    let cases: [(usize, f32, f32); 10] = [
        (4, PI / 2.0, 0.5),
        (5, PI, 1.0),
        (4, 1.4, 0.392_318_82),
        (4, 2.4, 0.922_488_3),
        (4, 0.3, 0.005_627_325),
        (10, 1.0, 0.043_092_596),
        (10, 2.0, 0.898_501_7),
        (64, 1.2, 0.001_507_249),
        (3, 1.0, 0.229_848_85),
        (2, 1.0, std::f32::consts::FRAC_1_PI),
    ];
    for (d, theta, expected) in cases {
        let mut center = vec![0.0f32; d];
        center[0] = 1.0;
        let f = SphericalCap::new(center, theta).unwrap().area_fraction();
        assert!(
            (f - expected).abs() < 1e-5,
            "d = {d}, θ = {theta}: area_fraction = {f}, expected {expected}"
        );
    }
}

/// A larger cap covers more of the sphere in every dimension.
#[test]
fn spherical_cap_area_fraction_is_monotone_in_radius() {
    for d in 2..=12usize {
        let mut center = vec![0.0f32; d];
        center[0] = 1.0;
        let mut prev = 0.0f32;
        for k in 1..=31 {
            let theta = PI * k as f32 / 32.0;
            let f = SphericalCap::new(center.clone(), theta)
                .unwrap()
                .area_fraction();
            assert!(
                f >= prev,
                "d = {d}: area_fraction fell from {prev} to {f} at θ = {theta}"
            );
            prev = f;
        }
    }
}
