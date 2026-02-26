//! Ndarray implementation of GumbelBox trait.

use crate::ndarray_backend::ndarray_box::NdarrayBox;
use crate::{
    gumbel_membership_prob, map_gumbel_to_bounds, sample_gumbel, Box, BoxError, GumbelBox,
};
use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A Gumbel box embedding implemented using `ndarray::Array1<f32>`.
#[derive(Debug, Clone)]
pub struct NdarrayGumbelBox {
    inner: NdarrayBox,
}

impl Serialize for NdarrayGumbelBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NdarrayGumbelBox {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = NdarrayBox::deserialize(deserializer)?;
        Ok(Self { inner })
    }
}

impl NdarrayGumbelBox {
    /// Create a new NdarrayGumbelBox.
    pub fn new(min: Array1<f32>, max: Array1<f32>, temperature: f32) -> Result<Self, BoxError> {
        Ok(Self {
            inner: NdarrayBox::new(min, max, temperature)?,
        })
    }
}

impl Box for NdarrayGumbelBox {
    type Scalar = f32;
    type Vector = Array1<f32>;

    fn min(&self) -> &Self::Vector {
        self.inner.min()
    }

    fn max(&self) -> &Self::Vector {
        self.inner.max()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn volume(&self, temperature: Self::Scalar) -> Result<Self::Scalar, BoxError> {
        self.inner.volume(temperature)
    }

    fn intersection(&self, other: &Self) -> Result<Self, BoxError> {
        Ok(Self {
            inner: self.inner.intersection(&other.inner)?,
        })
    }

    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        self.inner.containment_prob(&other.inner, temperature)
    }

    fn overlap_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        self.inner.overlap_prob(&other.inner, temperature)
    }

    fn union(&self, other: &Self) -> Result<Self, BoxError> {
        Ok(Self {
            inner: self.inner.union(&other.inner)?,
        })
    }

    fn center(&self) -> Result<Self::Vector, BoxError> {
        self.inner.center()
    }

    fn distance(&self, other: &Self) -> Result<Self::Scalar, BoxError> {
        self.inner.distance(&other.inner)
    }

    fn truncate(&self, k: usize) -> Result<Self, BoxError> {
        Ok(Self {
            inner: self.inner.truncate(k)?,
        })
    }
}

impl GumbelBox for NdarrayGumbelBox {
    fn temperature(&self) -> Self::Scalar {
        self.inner.temperature
    }

    fn membership_probability(&self, point: &Self::Vector) -> Result<Self::Scalar, BoxError> {
        if point.len() != self.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: point.len(),
            });
        }

        // P(point ∈ box) = ∏ P(min\[i\] <= point\[i\] <= max\[i\])
        // Using numerically stable Gumbel-Softmax probability calculation
        let temp = self.temperature();
        let mut prob = 1.0;

        for (i, &coord) in point.iter().enumerate() {
            let dim_prob = gumbel_membership_prob(coord, self.min()[i], self.max()[i], temp);
            prob *= dim_prob;
        }

        Ok(prob)
    }

    fn sample(&self) -> Self::Vector {
        use ndarray::Array1;

        // Use LCG for pseudo-random sampling to avoid rand dependency conflicts.
        // Note: LCG has known limitations (correlation, period) but is sufficient
        // for non-cryptographic sampling in embeddings.
        let temp = self.temperature();
        let dim = self.dim();

        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: u64 = 1u64 << 32;

        let mut sampled = Vec::with_capacity(dim);
        for i in 0..dim {
            seed = (A.wrapping_mul(seed).wrapping_add(C)) % M;
            let u = (seed as f32) / (M as f32);
            let gumbel = sample_gumbel(u, 1e-7);
            let value = map_gumbel_to_bounds(gumbel, self.min()[i], self.max()[i], temp);
            sampled.push(value);
        }

        Array1::from(sampled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Box as BoxTrait;
    use crate::GumbelBox as GumbelBoxTrait;
    use ndarray::array;

    // ---- Membership probability ----

    #[test]
    fn membership_prob_inside_point_is_high() {
        let gb = NdarrayGumbelBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let point = array![1.0, 1.0]; // center of the box
        let p = gb.membership_probability(&point).unwrap();
        // For a 2D box [0,2]^2 at temp=1.0, center membership is
        // prod_d [sigmoid(1)*sigmoid(1)] = (0.731*0.731)^2 ~ 0.285.
        // "High" relative to outside points; just verify it is the maximum.
        assert!(
            p > 0.2,
            "Center point should have non-trivial membership, got {}",
            p
        );
        assert!(p <= 1.0, "Membership must be <= 1.0");
    }

    #[test]
    fn membership_prob_far_outside_is_low() {
        let gb = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let point = array![10.0, 10.0];
        let p = gb.membership_probability(&point).unwrap();
        assert!(
            p < 0.01,
            "Far-outside point should have near-zero membership, got {}",
            p
        );
        assert!(p >= 0.0, "Membership must be >= 0.0");
    }

    #[test]
    fn membership_prob_always_in_unit_interval() {
        let gb = NdarrayGumbelBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 0.5).unwrap();
        let test_points = vec![
            array![0.5, 0.5, 0.5],   // inside
            array![-5.0, -5.0, -5.0], // far below
            array![10.0, 10.0, 10.0], // far above
            array![0.0, 0.0, 0.0],    // on boundary (min)
            array![1.0, 1.0, 1.0],    // on boundary (max)
        ];
        for pt in &test_points {
            let p = gb.membership_probability(pt).unwrap();
            assert!(
                (0.0..=1.0).contains(&p),
                "Membership {} out of [0,1] for point {:?}",
                p,
                pt
            );
        }
    }

    // ---- Containment monotonicity ----

    #[test]
    fn containment_monotonicity_nested_gumbel_boxes() {
        // If C inside B inside A, then P(C|A) should be >= P(B|A) is not
        // guaranteed, but P(A contains B) should be high and P(A contains C) should be high.
        // More precisely: P(B inside A) >= P(C inside A) is not guaranteed because
        // containment_prob = Vol(intersection) / Vol(other). Instead we check
        // that both are near 1.0 when boxes are strictly nested.
        let a =
            NdarrayGumbelBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let b =
            NdarrayGumbelBox::new(array![1.0, 1.0], array![9.0, 9.0], 1.0).unwrap();
        let c =
            NdarrayGumbelBox::new(array![2.0, 2.0], array![8.0, 8.0], 1.0).unwrap();

        let p_b_in_a = a.containment_prob(&b, 1.0).unwrap();
        let p_c_in_a = a.containment_prob(&c, 1.0).unwrap();
        let p_c_in_b = b.containment_prob(&c, 1.0).unwrap();

        assert!(p_b_in_a > 0.99, "B should be inside A, got {}", p_b_in_a);
        assert!(p_c_in_a > 0.99, "C should be inside A, got {}", p_c_in_a);
        assert!(p_c_in_b > 0.99, "C should be inside B, got {}", p_c_in_b);
    }

    // ---- Temperature effects on Gumbel membership ----

    #[test]
    fn low_temperature_sharpens_membership() {
        let gb_sharp = NdarrayGumbelBox::new(array![0.0, 0.0], array![2.0, 2.0], 0.01).unwrap();
        let gb_soft = NdarrayGumbelBox::new(array![0.0, 0.0], array![2.0, 2.0], 100.0).unwrap();

        // Point well inside
        let inside = array![1.0, 1.0];
        let p_sharp = gb_sharp.membership_probability(&inside).unwrap();
        let p_soft = gb_soft.membership_probability(&inside).unwrap();

        // At low temperature, interior point should have membership closer to 1.0
        // than at high temperature (where probability spreads out).
        assert!(
            p_sharp > p_soft || (p_sharp - p_soft).abs() < 0.05,
            "Low temp should give sharper membership for interior: sharp={}, soft={}",
            p_sharp,
            p_soft
        );

        // Point outside
        let outside = array![5.0, 5.0];
        let p_sharp_out = gb_sharp.membership_probability(&outside).unwrap();
        let p_soft_out = gb_soft.membership_probability(&outside).unwrap();

        // At low temperature, outside point should have membership closer to 0.0
        assert!(
            p_sharp_out < p_soft_out || (p_sharp_out - p_soft_out).abs() < 0.05,
            "Low temp should give lower membership for exterior: sharp={}, soft={}",
            p_sharp_out,
            p_soft_out
        );
    }

    // ---- Serialization round-trip ----

    #[test]
    fn gumbel_box_serde_round_trip() {
        let original =
            NdarrayGumbelBox::new(array![0.1, 0.2], array![0.8, 0.9], 0.5).unwrap();
        let json = serde_json::to_string(&original).expect("serialize");
        let deserialized: NdarrayGumbelBox =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(original.dim(), deserialized.dim());
        assert_eq!(original.temperature(), deserialized.temperature());
        for i in 0..original.dim() {
            assert!(
                (original.min()[i] - deserialized.min()[i]).abs() < 1e-7,
                "min mismatch at dim {}",
                i
            );
            assert!(
                (original.max()[i] - deserialized.max()[i]).abs() < 1e-7,
                "max mismatch at dim {}",
                i
            );
        }
    }

    // ---- Sample stays finite ----

    #[test]
    fn sample_produces_finite_values() {
        let gb = NdarrayGumbelBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0).unwrap();
        let s = gb.sample();
        assert_eq!(s.len(), 3);
        for &v in s.iter() {
            assert!(v.is_finite(), "Sampled value must be finite, got {}", v);
        }
    }

    // ---- Construction error paths ----

    #[test]
    fn gumbel_box_dim_mismatch_error() {
        let result = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn gumbel_box_invalid_bounds_error() {
        let result = NdarrayGumbelBox::new(array![5.0], array![1.0], 1.0);
        assert!(result.is_err());
    }

    // ---- Membership dimension mismatch ----

    #[test]
    fn membership_prob_dimension_mismatch() {
        let gb = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let point = array![0.5]; // wrong dimension
        let result = gb.membership_probability(&point);
        assert!(result.is_err());
    }

    // ---- Temperature accessor ----

    #[test]
    fn temperature_accessor_returns_construction_value() {
        let gb = NdarrayGumbelBox::new(array![0.0], array![1.0], 0.42).unwrap();
        assert!((gb.temperature() - 0.42).abs() < 1e-7);
    }

    // ---- Soft containment: membership near boundary is ~0.5 per dim ----

    #[test]
    fn membership_at_boundary_is_near_half_per_dim() {
        // At temperature 1.0, membership at the min or max boundary should be
        // sigmoid(0) = 0.5 in that dimension. For a 1D box, P = sigmoid(0)*sigmoid(width/temp).
        let gb = NdarrayGumbelBox::new(array![0.0], array![10.0], 1.0).unwrap();
        let p_at_min = gb.membership_probability(&array![0.0]).unwrap();
        let p_at_max = gb.membership_probability(&array![10.0]).unwrap();
        // At min: sigmoid(0) * sigmoid(10) ~ 0.5 * 0.99995 ~ 0.5
        // At max: sigmoid(10) * sigmoid(0) ~ 0.99995 * 0.5 ~ 0.5
        assert!(
            (p_at_min - 0.5).abs() < 0.05,
            "At min boundary expected ~0.5, got {}",
            p_at_min
        );
        assert!(
            (p_at_max - 0.5).abs() < 0.05,
            "At max boundary expected ~0.5, got {}",
            p_at_max
        );
    }

    // ---- Temperature extremes ----

    #[test]
    fn very_low_temperature_membership_approaches_hard() {
        // At near-zero temperature, inside points should have membership near 1.0
        let gb = NdarrayGumbelBox::new(array![0.0, 0.0], array![4.0, 4.0], 0.001).unwrap();
        let center = array![2.0, 2.0];
        let p = gb.membership_probability(&center).unwrap();
        assert!(
            p > 0.99,
            "At very low temp, center should have membership ~1.0, got {}",
            p
        );
    }

    #[test]
    fn very_low_temperature_outside_membership_approaches_zero() {
        let gb = NdarrayGumbelBox::new(array![0.0, 0.0], array![4.0, 4.0], 0.001).unwrap();
        let outside = array![-1.0, -1.0];
        let p = gb.membership_probability(&outside).unwrap();
        assert!(
            p < 0.01,
            "At very low temp, outside should have membership ~0.0, got {}",
            p
        );
    }

    // ---- Delegated Box trait operations ----

    #[test]
    fn gumbel_box_intersection_delegates() {
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let b = NdarrayGumbelBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let inter = a.intersection(&b).unwrap();
        let vol = inter.volume(1.0).unwrap();
        assert!((vol - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gumbel_box_union_delegates() {
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let u = a.union(&a).unwrap();
        let vol_a = a.volume(1.0).unwrap();
        let vol_u = u.volume(1.0).unwrap();
        assert!((vol_a - vol_u).abs() < 1e-6);
    }

    #[test]
    fn gumbel_box_center_delegates() {
        let gb = NdarrayGumbelBox::new(array![0.0, 4.0], array![2.0, 8.0], 1.0).unwrap();
        let c = gb.center().unwrap();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn gumbel_box_distance_delegates() {
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayGumbelBox::new(array![3.0, 0.0], array![4.0, 1.0], 1.0).unwrap();
        let d = a.distance(&b).unwrap();
        assert!((d - 2.0).abs() < 1e-5, "Gap of 2 in x only, expected 2.0, got {}", d);
    }

    #[test]
    fn gumbel_box_truncate_delegates() {
        let gb = NdarrayGumbelBox::new(array![0.0, 1.0, 2.0], array![3.0, 4.0, 5.0], 0.5).unwrap();
        let t = gb.truncate(2).unwrap();
        assert_eq!(t.dim(), 2);
        assert!((t.min()[0] - 0.0).abs() < 1e-7);
        assert!((t.max()[1] - 4.0).abs() < 1e-7);
    }

    // ---- Membership monotonicity in distance from center ----

    #[test]
    fn membership_decreases_moving_away_from_center() {
        let gb = NdarrayGumbelBox::new(array![0.0, 0.0], array![4.0, 4.0], 1.0).unwrap();
        let center = array![2.0, 2.0];
        let near_edge = array![3.5, 3.5];
        let outside = array![6.0, 6.0];

        let p_center = gb.membership_probability(&center).unwrap();
        let p_edge = gb.membership_probability(&near_edge).unwrap();
        let p_out = gb.membership_probability(&outside).unwrap();

        assert!(
            p_center > p_edge,
            "Center ({}) should have higher membership than near-edge ({})",
            p_center, p_edge
        );
        assert!(
            p_edge > p_out,
            "Near-edge ({}) should have higher membership than outside ({})",
            p_edge, p_out
        );
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use crate::GumbelBox as GumbelBoxTrait;
    use ndarray::Array1;
    use proptest::prelude::*;

    // ---- Property 1: membership_probability is in [0, 1] ----

    proptest! {
        #[test]
        fn proptest_membership_in_unit_interval(
            box_pairs in proptest::collection::vec((-20.0f32..20.0f32, 1.0f32..10.0f32), 1..=5),
            temp in 0.01f32..10.0f32,
            point_coords in proptest::collection::vec(-30.0f32..30.0f32, 1..=5),
        ) {
            let dim = box_pairs.len();
            // Skip if point dimension doesn't match box dimension
            prop_assume!(point_coords.len() >= dim);

            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (lo, width) in &box_pairs {
                mins.push(*lo);
                maxs.push(*lo + *width);
            }
            let gb = NdarrayGumbelBox::new(
                Array1::from(mins),
                Array1::from(maxs),
                temp,
            ).unwrap();

            let point = Array1::from(point_coords[..dim].to_vec());
            let p = gb.membership_probability(&point).unwrap();
            prop_assert!(
                (0.0..=1.0).contains(&p),
                "membership_probability must be in [0,1], got {} (temp={}, point={:?})",
                p, temp, point
            );
            prop_assert!(p.is_finite(), "membership must be finite, got {}", p);
        }
    }

    // ---- Property 2: Center membership > boundary membership ----

    proptest! {
        #[test]
        fn proptest_center_membership_gt_boundary(
            box_pairs in proptest::collection::vec((-10.0f32..10.0f32, 2.0f32..10.0f32), 1..=4),
            temp in 0.1f32..5.0f32,
        ) {
            let dim = box_pairs.len();
            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (lo, width) in &box_pairs {
                mins.push(*lo);
                maxs.push(*lo + *width);
            }
            let gb = NdarrayGumbelBox::new(
                Array1::from(mins.clone()),
                Array1::from(maxs.clone()),
                temp,
            ).unwrap();

            // Center point
            let center: Vec<f32> = mins.iter().zip(maxs.iter())
                .map(|(lo, hi)| (lo + hi) / 2.0)
                .collect();
            // Boundary point (at min in every dimension)
            let boundary: Vec<f32> = mins.clone();

            let p_center = gb.membership_probability(&Array1::from(center)).unwrap();
            let p_boundary = gb.membership_probability(&Array1::from(boundary)).unwrap();

            prop_assert!(
                p_center >= p_boundary - 1e-6,
                "center membership ({}) should be >= boundary membership ({}), temp={}",
                p_center, p_boundary, temp
            );
        }
    }

    // ---- Property 3: Temperature monotonicity for outside points ----
    // Higher temperature -> membership at a point *outside* the box increases,
    // because the soft sigmoid "leaks" more probability mass beyond the hard boundary.
    //
    // At the boundary itself, behavior is non-monotone: the hard-box path (very low temp)
    // returns 1.0 for boundary points (they satisfy min <= x <= max), but the soft
    // sigmoid path gives sigmoid(0) ~ 0.5 per boundary dimension. So we test with
    // points strictly outside the box, where the monotonicity is clean.

    proptest! {
        #[test]
        fn proptest_temperature_monotonicity_outside(
            box_pairs in proptest::collection::vec((-10.0f32..10.0f32, 2.0f32..10.0f32), 1..=3),
            temp_lo in 0.1f32..1.0f32,
            temp_delta in 1.0f32..50.0f32,
            offset in 1.0f32..5.0f32,
        ) {
            let dim = box_pairs.len();
            let temp_hi = temp_lo + temp_delta;

            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (lo, width) in &box_pairs {
                mins.push(*lo);
                maxs.push(*lo + *width);
            }

            let gb_lo = NdarrayGumbelBox::new(
                Array1::from(mins.clone()),
                Array1::from(maxs.clone()),
                temp_lo,
            ).unwrap();
            let gb_hi = NdarrayGumbelBox::new(
                Array1::from(mins.clone()),
                Array1::from(maxs.clone()),
                temp_hi,
            ).unwrap();

            // Point strictly outside: shift below min by offset in every dimension
            let outside: Vec<f32> = mins.iter().map(|lo| lo - offset).collect();
            let outside_pt = Array1::from(outside);
            let p_lo = gb_lo.membership_probability(&outside_pt).unwrap();
            let p_hi = gb_hi.membership_probability(&outside_pt).unwrap();

            // Higher temperature should give higher membership for outside points
            // (more probability leaks beyond the hard boundary).
            prop_assert!(
                p_hi >= p_lo - 1e-5,
                "higher temp ({}) should give >= membership for outside point than lower temp ({}): p_hi={}, p_lo={}",
                temp_hi, temp_lo, p_hi, p_lo
            );
        }
    }
}
