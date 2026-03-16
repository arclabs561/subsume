//! Ndarray Gumbel box embedding with temperature-controlled probabilistic bounds.

use crate::ndarray_backend::ndarray_box::NdarrayBox;
use crate::utils::{bessel_log_volume, gumbel_lse_max, gumbel_lse_min};
use crate::utils::{gumbel_membership_prob, map_gumbel_to_bounds, sample_gumbel};
use crate::{Box, BoxError};
use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A Gumbel box embedding implemented using `ndarray::Array1<f32>`.
///
/// # Temperature semantics
///
/// This type stores a single temperature used for all operations (intersection,
/// volume, containment). When two GumbelBoxes with different temperatures are
/// intersected, `self`'s temperature is used. This makes intersection
/// argument-order-dependent when temperatures differ.
///
/// The reference Python implementation (Boratko et al., EMNLP 2021) passes
/// temperature as a function argument instead. Per-relation temperatures
/// (Chen et al., ACL 2021 / BEUrRE) make the asymmetry semantically
/// motivated: P(B|A) naturally uses A's uncertainty scale.
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

    /// Gumbel box volume using the Bessel/softplus approximation (Dasgupta et al., 2020).
    ///
    /// Uses the stored temperature from construction; the `temperature` parameter is ignored.
    ///
    /// Per dimension: `softplus(Z - z - 2*gamma*T, beta=1/T)`
    /// where `gamma` is the Euler-Mascheroni constant and `T` is the temperature.
    ///
    /// This provides smooth gradients even for near-empty boxes, solving the local
    /// identifiability problem that motivates Gumbel boxes in the first place.
    fn volume(&self, _temperature: Self::Scalar) -> Result<Self::Scalar, BoxError> {
        let t = self.inner.temperature;
        let mins = self.min().as_slice().unwrap();
        let maxs = self.max().as_slice().unwrap();
        let (_, vol) = bessel_log_volume(mins, maxs, t, t);
        Ok(vol)
    }

    /// Gumbel box intersection using log-sum-exp (Dasgupta et al., 2020).
    ///
    /// ```text
    /// z_cap[d] =  T * logsumexp(z_a[d]/T, z_b[d]/T)    (smooth max)
    /// Z_cap[d] = -T * logsumexp(-Z_a[d]/T, -Z_b[d]/T)  (smooth min)
    /// ```
    ///
    /// The LSE intersection provides smooth gradients at box boundaries,
    /// approaching the hard intersection as T -> 0.
    fn intersection(&self, other: &Self) -> Result<Self, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let t = self.inner.temperature;
        let n = self.dim();

        let mut new_min = Vec::with_capacity(n);
        let mut new_max = Vec::with_capacity(n);

        for d in 0..n {
            new_min.push(gumbel_lse_min(self.min()[d], other.min()[d], t));
            new_max.push(gumbel_lse_max(self.max()[d], other.max()[d], t));
        }

        // No hard clipping: softplus volume handles "flipped" boxes (z > Z)
        // gracefully by returning near-zero volume.
        Ok(Self {
            inner: NdarrayBox::new_unchecked(Array1::from(new_min), Array1::from(new_max), t),
        })
    }

    /// Containment probability using Gumbel volume and LSE intersection.
    ///
    /// Uses the stored temperature from construction; the `temperature` parameter is ignored.
    ///
    /// `P(other inside self) = Vol(self cap other) / Vol(other)`
    fn containment_prob(
        &self,
        other: &Self,
        _temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        let inter = self.intersection(other)?;
        let inter_vol = inter.volume(0.0)?;
        let other_vol = other.volume(0.0)?;
        if other_vol <= 1e-30 {
            return Ok(0.0);
        }
        Ok((inter_vol / other_vol).clamp(0.0, 1.0))
    }

    /// Overlap probability using Gumbel volume and LSE intersection.
    ///
    /// `P(self cap other != empty) = Vol(self cap other) / Vol(self cup other)`
    fn overlap_prob(
        &self,
        other: &Self,
        _temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        let inter = self.intersection(other)?;
        let inter_vol = inter.volume(0.0)?;
        let self_vol = self.volume(0.0)?;
        let other_vol = other.volume(0.0)?;
        let union_vol = self_vol + other_vol - inter_vol;
        if union_vol <= 1e-30 {
            return Ok(0.0);
        }
        Ok((inter_vol / union_vol).clamp(0.0, 1.0))
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

impl NdarrayGumbelBox {
    /// Get the temperature parameter (controls softness of bounds).
    pub fn temperature(&self) -> f32 {
        self.inner.temperature
    }

    /// Compute membership probability for a point using Gumbel-Softmax.
    ///
    /// Returns P(point in self) as the product of per-dimension sigmoid probabilities.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if point dimension doesn't match box dimension.
    pub fn membership_probability(&self, point: &Array1<f32>) -> Result<f32, BoxError> {
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

    /// Sample a point from the box distribution using Gumbel-Softmax.
    pub fn sample(&self) -> Array1<f32> {
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
            array![0.5, 0.5, 0.5],    // inside
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
        // With Bessel volume (softplus with 2*gamma*T offset), the effective side
        // lengths are smaller than the hard box. At T=1.0, the offset is ~1.154
        // per dimension. Containment probabilities are lower than hard-box but
        // should still show monotonicity: more nested = higher containment.
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let b = NdarrayGumbelBox::new(array![1.0, 1.0], array![9.0, 9.0], 1.0).unwrap();
        let c = NdarrayGumbelBox::new(array![2.0, 2.0], array![8.0, 8.0], 1.0).unwrap();

        let p_b_in_a = a.containment_prob(&b, 1.0).unwrap();
        let p_c_in_a = a.containment_prob(&c, 1.0).unwrap();
        let p_c_in_b = b.containment_prob(&c, 1.0).unwrap();

        // All should be well above 0.5 for clearly nested boxes
        assert!(p_b_in_a > 0.7, "B should be inside A, got {}", p_b_in_a);
        assert!(p_c_in_a > 0.7, "C should be inside A, got {}", p_c_in_a);
        assert!(p_c_in_b > 0.7, "C should be inside B, got {}", p_c_in_b);

        // At low temperature, should approach hard-box behavior
        let a_sharp = NdarrayGumbelBox::new(array![0.0, 0.0], array![10.0, 10.0], 0.01).unwrap();
        let b_sharp = NdarrayGumbelBox::new(array![1.0, 1.0], array![9.0, 9.0], 0.01).unwrap();
        let p_sharp = a_sharp.containment_prob(&b_sharp, 0.01).unwrap();
        assert!(
            p_sharp > 0.99,
            "At low T, containment should be ~1.0, got {}",
            p_sharp
        );
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
        let original = NdarrayGumbelBox::new(array![0.1, 0.2], array![0.8, 0.9], 0.5).unwrap();
        let json = serde_json::to_string(&original).expect("serialize");
        let deserialized: NdarrayGumbelBox = serde_json::from_str(&json).expect("deserialize");

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
    fn gumbel_box_intersection_uses_lse() {
        // LSE intersection produces smoother bounds than hard intersection.
        // For boxes [0,2]^2 and [1,3]^2 at T=1.0:
        //   z_cap = T * lse(0/T, 1/T) = lse(0, 1) = ln(1 + e) ~ 1.313
        //   Z_cap = -T * lse(-2/T, -3/T) = -lse(-2, -3) ~ -(-1.687) = 1.687 (wrong sign calc)
        // Actually: gumbel_lse_max(2, 3, 1) = -lse(-2, -3, 1) which is smooth min(2,3) ~ 1.687
        // So intersection box is approximately [1.313, 1.687]^2, vol ~ 0.14
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let b = NdarrayGumbelBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let inter = a.intersection(&b).unwrap();
        let vol = inter.volume(1.0).unwrap();
        // Volume should be positive but less than the hard intersection volume (1.0)
        assert!(
            vol > 0.0,
            "intersection volume should be positive, got {}",
            vol
        );
        assert!(
            vol < 1.0,
            "LSE intersection volume should be < hard vol (1.0), got {}",
            vol
        );

        // At low temperature, should approach hard intersection
        let a_sharp = NdarrayGumbelBox::new(array![0.0, 0.0], array![2.0, 2.0], 0.01).unwrap();
        let b_sharp = NdarrayGumbelBox::new(array![1.0, 1.0], array![3.0, 3.0], 0.01).unwrap();
        let inter_sharp = a_sharp.intersection(&b_sharp).unwrap();
        let vol_sharp = inter_sharp.volume(0.01).unwrap();
        // Hard intersection is [1,2]^2 = 1.0; Bessel vol at T=0.01 should be close
        assert!(
            (vol_sharp - 1.0).abs() < 0.1,
            "At low T, intersection volume should be ~1.0, got {}",
            vol_sharp
        );
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
        assert!(
            (d - 2.0).abs() < 1e-5,
            "Gap of 2 in x only, expected 2.0, got {}",
            d
        );
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
            p_center,
            p_edge
        );
        assert!(
            p_edge > p_out,
            "Near-edge ({}) should have higher membership than outside ({})",
            p_edge,
            p_out
        );
    }

    // ---- Bessel volume properties ----

    #[test]
    fn bessel_volume_monotone_in_side_length() {
        // Larger side length => larger Bessel volume
        let small = NdarrayGumbelBox::new(array![0.0], array![2.0], 1.0).unwrap();
        let large = NdarrayGumbelBox::new(array![0.0], array![5.0], 1.0).unwrap();
        let v_small = small.volume(1.0).unwrap();
        let v_large = large.volume(1.0).unwrap();
        assert!(
            v_large > v_small,
            "larger box should have larger vol: {v_large} vs {v_small}"
        );
    }

    #[test]
    fn bessel_volume_positive_for_nonempty_box() {
        let b = NdarrayGumbelBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0).unwrap();
        let v = b.volume(1.0).unwrap();
        assert!(
            v > 0.0,
            "non-empty box should have positive volume, got {v}"
        );
        assert!(v.is_finite(), "volume should be finite");
    }

    #[test]
    fn bessel_volume_approaches_hard_at_low_temperature() {
        // At T=0.01, Bessel volume should be close to hard volume
        let b = NdarrayGumbelBox::new(array![0.0, 0.0], array![3.0, 4.0], 0.01).unwrap();
        let v = b.volume(0.01).unwrap();
        let hard_vol = 3.0 * 4.0;
        assert!(
            (v - hard_vol).abs() / hard_vol < 0.05,
            "At low T, Bessel vol ({v}) should be close to hard vol ({hard_vol})"
        );
    }

    #[test]
    fn bessel_volume_smaller_than_hard_at_high_temperature() {
        // The 2*gamma*T offset reduces effective side lengths
        let b = NdarrayGumbelBox::new(array![0.0, 0.0], array![5.0, 5.0], 1.0).unwrap();
        let bessel_vol = b.volume(1.0).unwrap();
        let hard_vol = 5.0 * 5.0;
        assert!(
            bessel_vol < hard_vol,
            "Bessel vol ({bessel_vol}) should be < hard vol ({hard_vol}) due to gamma offset"
        );
    }

    // ---- LSE intersection properties ----

    #[test]
    fn lse_intersection_symmetric() {
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![3.0, 3.0], 1.0).unwrap();
        let b = NdarrayGumbelBox::new(array![1.0, 1.0], array![4.0, 4.0], 1.0).unwrap();
        let ab = a.intersection(&b).unwrap();
        let ba = b.intersection(&a).unwrap();
        for d in 0..2 {
            assert!(
                (ab.min()[d] - ba.min()[d]).abs() < 1e-6,
                "intersection min should be symmetric at dim {d}"
            );
            assert!(
                (ab.max()[d] - ba.max()[d]).abs() < 1e-6,
                "intersection max should be symmetric at dim {d}"
            );
        }
    }

    #[test]
    fn lse_intersection_approaches_hard_at_low_temperature() {
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![3.0, 3.0], 0.01).unwrap();
        let b = NdarrayGumbelBox::new(array![1.0, 1.0], array![4.0, 4.0], 0.01).unwrap();
        let inter = a.intersection(&b).unwrap();
        // Hard intersection would be [1,3]^2
        assert!(
            (inter.min()[0] - 1.0).abs() < 0.05,
            "LSE min should be ~1.0 at low T, got {}",
            inter.min()[0]
        );
        assert!(
            (inter.max()[0] - 3.0).abs() < 0.05,
            "LSE max should be ~3.0 at low T, got {}",
            inter.max()[0]
        );
    }

    #[test]
    fn lse_intersection_bounds_are_inside_parents() {
        // LSE intersection min >= max(a.min, b.min) (smooth max is an upper bound on hard max)
        // LSE intersection max <= min(a.max, b.max) (smooth min is a lower bound on hard min)
        // In fact the opposite: LSE min >= hard min (smooth max >= hard max) and
        // LSE max <= hard max (smooth min <= hard min). So the LSE box is contained
        // within the hard intersection box... no, the other way:
        // lse(a, b) >= max(a, b), so z_cap_lse >= z_cap_hard
        // -lse(-a, -b) <= min(a, b), so Z_cap_lse <= Z_cap_hard
        // So the LSE intersection is INSIDE the hard intersection (tighter).
        let a = NdarrayGumbelBox::new(array![0.0, 1.0], array![5.0, 6.0], 1.0).unwrap();
        let b = NdarrayGumbelBox::new(array![2.0, 0.0], array![4.0, 7.0], 1.0).unwrap();
        let inter = a.intersection(&b).unwrap();
        // LSE min >= hard min: lse(0,2) >= max(0,2) = 2 and lse(1,0) >= max(1,0) = 1
        assert!(
            inter.min()[0] >= 2.0 - 1e-6,
            "lse min[0] ({}) should be >= 2.0",
            inter.min()[0]
        );
        assert!(
            inter.min()[1] >= 1.0 - 1e-6,
            "lse min[1] ({}) should be >= 1.0",
            inter.min()[1]
        );
        // LSE max <= hard max: -lse(-5,-4) <= min(5,4) = 4 and -lse(-6,-7) <= min(6,7) = 6
        assert!(
            inter.max()[0] <= 4.0 + 1e-6,
            "lse max[0] ({}) should be <= 4.0",
            inter.max()[0]
        );
        assert!(
            inter.max()[1] <= 6.0 + 1e-6,
            "lse max[1] ({}) should be <= 6.0",
            inter.max()[1]
        );
    }

    #[test]
    fn disjoint_boxes_have_near_zero_gumbel_containment() {
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayGumbelBox::new(array![10.0, 10.0], array![11.0, 11.0], 1.0).unwrap();
        let p = a.containment_prob(&b, 1.0).unwrap();
        assert!(
            p < 0.01,
            "disjoint boxes should have near-zero containment, got {p}"
        );
    }

    #[test]
    fn gumbel_overlap_prob_reasonable() {
        // Overlapping boxes
        let a = NdarrayGumbelBox::new(array![0.0, 0.0], array![3.0, 3.0], 1.0).unwrap();
        let b = NdarrayGumbelBox::new(array![1.0, 1.0], array![4.0, 4.0], 1.0).unwrap();
        let p = a.overlap_prob(&b, 1.0).unwrap();
        assert!(
            p > 0.0,
            "overlapping boxes should have positive overlap, got {p}"
        );
        assert!(p <= 1.0, "overlap should be <= 1.0");

        // Disjoint boxes
        let c = NdarrayGumbelBox::new(array![10.0, 10.0], array![11.0, 11.0], 1.0).unwrap();
        let p_disjoint = a.overlap_prob(&c, 1.0).unwrap();
        assert!(
            p_disjoint < 0.01,
            "disjoint boxes should have near-zero overlap, got {p_disjoint}"
        );
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
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

    // ---- Property 4: Bessel volume is non-negative and finite ----

    proptest! {
        #[test]
        fn proptest_bessel_volume_non_negative(
            box_pairs in proptest::collection::vec((-10.0f32..10.0f32, 0.5f32..10.0f32), 1..=5),
            temp in 0.01f32..5.0f32,
        ) {
            let dim = box_pairs.len();
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

            let vol = gb.volume(temp).unwrap();
            prop_assert!(vol >= 0.0, "Bessel volume must be >= 0, got {vol}");
            prop_assert!(vol.is_finite(), "Bessel volume must be finite, got {vol}");
        }
    }

    // ---- Property 5: Containment probability in [0, 1] ----

    proptest! {
        #[test]
        fn proptest_gumbel_containment_in_unit_interval(
            a_pairs in proptest::collection::vec((-5.0f32..5.0f32, 1.0f32..5.0f32), 1..=3),
            b_pairs in proptest::collection::vec((-5.0f32..5.0f32, 1.0f32..5.0f32), 1..=3),
            temp in 0.1f32..3.0f32,
        ) {
            let dim = a_pairs.len().min(b_pairs.len());
            prop_assume!(dim > 0);

            let a = NdarrayGumbelBox::new(
                Array1::from(a_pairs[..dim].iter().map(|(lo, _)| *lo).collect::<Vec<_>>()),
                Array1::from(a_pairs[..dim].iter().map(|(lo, w)| lo + w).collect::<Vec<_>>()),
                temp,
            ).unwrap();
            let b = NdarrayGumbelBox::new(
                Array1::from(b_pairs[..dim].iter().map(|(lo, _)| *lo).collect::<Vec<_>>()),
                Array1::from(b_pairs[..dim].iter().map(|(lo, w)| lo + w).collect::<Vec<_>>()),
                temp,
            ).unwrap();

            let p = a.containment_prob(&b, temp).unwrap();
            prop_assert!(p >= -1e-6, "containment prob must be >= 0, got {p}");
            prop_assert!(p <= 1.0 + 1e-6, "containment prob must be <= 1, got {p}");
            prop_assert!(p.is_finite(), "containment prob must be finite, got {p}");
        }
    }

    // ---- Property 6: LSE intersection is symmetric ----

    proptest! {
        #[test]
        fn proptest_lse_intersection_symmetric(
            a_pairs in proptest::collection::vec((-5.0f32..5.0f32, 1.0f32..5.0f32), 1..=3),
            b_pairs in proptest::collection::vec((-5.0f32..5.0f32, 1.0f32..5.0f32), 1..=3),
            temp in 0.1f32..3.0f32,
        ) {
            let dim = a_pairs.len().min(b_pairs.len());
            prop_assume!(dim > 0);

            let a = NdarrayGumbelBox::new(
                Array1::from(a_pairs[..dim].iter().map(|(lo, _)| *lo).collect::<Vec<_>>()),
                Array1::from(a_pairs[..dim].iter().map(|(lo, w)| lo + w).collect::<Vec<_>>()),
                temp,
            ).unwrap();
            let b = NdarrayGumbelBox::new(
                Array1::from(b_pairs[..dim].iter().map(|(lo, _)| *lo).collect::<Vec<_>>()),
                Array1::from(b_pairs[..dim].iter().map(|(lo, w)| lo + w).collect::<Vec<_>>()),
                temp,
            ).unwrap();

            let ab = a.intersection(&b).unwrap();
            let ba = b.intersection(&a).unwrap();
            let vol_ab = ab.volume(temp).unwrap();
            let vol_ba = ba.volume(temp).unwrap();
            prop_assert!(
                (vol_ab - vol_ba).abs() < 1e-4,
                "intersection volume should be symmetric: {vol_ab} vs {vol_ba}"
            );
        }
    }

    // ---- Property 7: LSE intersection is contained within both parents ----
    // For any two boxes A, B and their LSE intersection I:
    //   I.min[d] >= max(A.min[d], B.min[d])  (gumbel_lse_min >= hard max)
    //   I.max[d] <= min(A.max[d], B.max[d])  (gumbel_lse_max <= hard min)
    // The LSE bounds are tighter (or equal) than the hard intersection bounds.

    proptest! {
        #[test]
        fn proptest_lse_intersection_within_parents(
            a_pairs in proptest::collection::vec((-5.0f32..5.0f32, 0.5f32..5.0f32), 1..=4),
            b_pairs in proptest::collection::vec((-5.0f32..5.0f32, 0.5f32..5.0f32), 1..=4),
            temp in 0.01f32..3.0f32,
        ) {
            let dim = a_pairs.len().min(b_pairs.len());
            prop_assume!(dim > 0);

            let a_min: Vec<f32> = a_pairs[..dim].iter().map(|(lo, _)| *lo).collect();
            let a_max: Vec<f32> = a_pairs[..dim].iter().map(|(lo, w)| lo + w).collect();
            let b_min: Vec<f32> = b_pairs[..dim].iter().map(|(lo, _)| *lo).collect();
            let b_max: Vec<f32> = b_pairs[..dim].iter().map(|(lo, w)| lo + w).collect();

            let a = NdarrayGumbelBox::new(
                Array1::from(a_min.clone()),
                Array1::from(a_max.clone()),
                temp,
            ).unwrap();
            let b = NdarrayGumbelBox::new(
                Array1::from(b_min.clone()),
                Array1::from(b_max.clone()),
                temp,
            ).unwrap();

            let inter = a.intersection(&b).unwrap();

            for d in 0..dim {
                let hard_min = a_min[d].max(b_min[d]);
                let hard_max = a_max[d].min(b_max[d]);

                // LSE min (smooth max) >= hard max of mins
                prop_assert!(
                    inter.min()[d] >= hard_min - 1e-5,
                    "dim {d}: LSE intersection min {} < hard min {hard_min}",
                    inter.min()[d]
                );

                // LSE max (smooth min) <= hard min of maxes
                prop_assert!(
                    inter.max()[d] <= hard_max + 1e-5,
                    "dim {d}: LSE intersection max {} > hard max {hard_max}",
                    inter.max()[d]
                );
            }
        }
    }

    // ---- Property 8: Bessel volume <= hard volume ----
    // The Bessel volume subtracts 2*gamma*T from each dimension's side length
    // before applying softplus, so for boxes with side >> 2*gamma*T, the
    // Bessel volume should be less than the hard volume.
    // For any box, the per-dim Bessel side = softplus(Z - z - 2*gamma*T, 1/T).
    // Since softplus(x, beta) >= 0 and approaches x for large x, but always
    // softplus(x) <= x + (1/beta)*ln(2), the total Bessel volume should be
    // bounded relative to the hard volume.
    //
    // Concrete invariant: at low temperature (T <= 0.1), for boxes with side > 1.0,
    // Bessel volume should be close to (but not exceed) hard volume.

    proptest! {
        #[test]
        fn proptest_bessel_volume_bounded_by_hard_volume_at_low_t(
            pairs in proptest::collection::vec((-5.0f32..5.0f32, 2.0f32..10.0f32), 1..=4),
            temp in 0.001f32..0.1f32,
        ) {
            let mins: Vec<f32> = pairs.iter().map(|(lo, _)| *lo).collect();
            let maxs: Vec<f32> = pairs.iter().map(|(lo, w)| lo + w).collect();

            let hard_vol: f32 = pairs.iter().map(|(_, w)| w).product();

            let gb = NdarrayGumbelBox::new(
                Array1::from(mins),
                Array1::from(maxs),
                temp,
            ).unwrap();
            let bessel_vol = gb.volume(temp).unwrap();

            // At low T, Bessel volume should be close to hard volume but slightly less
            // (the 2*gamma*T offset reduces each side by ~2*0.577*T per dim).
            prop_assert!(
                bessel_vol <= hard_vol + 1e-3,
                "Bessel vol ({bessel_vol}) should be <= hard vol ({hard_vol}) at T={temp}"
            );
            prop_assert!(
                bessel_vol > 0.0,
                "Bessel vol should be positive, got {bessel_vol}"
            );
        }
    }

    // ---- Property 9: Gumbel containment approaches hard containment at low T ----
    // For boxes where A fully contains B (hard containment = 1.0),
    // Gumbel containment should approach 1.0 as T -> 0.

    proptest! {
        #[test]
        fn proptest_gumbel_containment_approaches_hard_at_low_t(
            b_pairs in proptest::collection::vec((-3.0f32..3.0f32, 0.5f32..2.0f32), 1..=3),
            margin in 0.5f32..3.0f32,
        ) {
            // B is the inner box; A is B expanded by `margin` on each side.
            let b_min: Vec<f32> = b_pairs.iter().map(|(lo, _)| *lo).collect();
            let b_max: Vec<f32> = b_pairs.iter().map(|(lo, w)| lo + w).collect();
            let a_min: Vec<f32> = b_min.iter().map(|lo| lo - margin).collect();
            let a_max: Vec<f32> = b_max.iter().map(|hi| hi + margin).collect();

            // At a low temperature, containment should be close to 1.0.
            let t_low = 0.01f32;
            let a = NdarrayGumbelBox::new(
                Array1::from(a_min),
                Array1::from(a_max),
                t_low,
            ).unwrap();
            let b = NdarrayGumbelBox::new(
                Array1::from(b_min),
                Array1::from(b_max),
                t_low,
            ).unwrap();

            let containment = a.containment_prob(&b, t_low).unwrap();
            prop_assert!(
                containment > 0.9,
                "hard containment=1.0 but Gumbel containment at T={t_low} is only {containment}"
            );
        }
    }

    // ---- Property 10: Volume non-negative for arbitrary temperature ----
    // NdarrayGumbelBox volume must be >= 0 for any finite positive temperature,
    // including values well above the box's stored temperature.

    proptest! {
        #[test]
        fn proptest_volume_non_negative_any_temperature(
            box_pairs in proptest::collection::vec((-10.0f32..10.0f32, 0.1f32..10.0f32), 1..=8),
            stored_temp in 0.1f32..10.0f32,
            query_temp in 0.1f32..10.0f32,
        ) {
            let dim = box_pairs.len();
            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (lo, width) in &box_pairs {
                mins.push(*lo);
                maxs.push(*lo + *width);
            }
            let gb = NdarrayGumbelBox::new(
                Array1::from(mins),
                Array1::from(maxs),
                stored_temp,
            ).unwrap();

            let vol = gb.volume(query_temp).unwrap();
            prop_assert!(vol >= 0.0,
                "Gumbel volume must be >= 0 for any temp, got {vol} (stored_t={stored_temp}, query_t={query_temp})");
            prop_assert!(vol.is_finite(),
                "Gumbel volume must be finite, got {vol}");
        }
    }
}
