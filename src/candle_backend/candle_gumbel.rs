//! Candle Gumbel box embedding with temperature-controlled probabilistic bounds.

use crate::candle_backend::candle_box::CandleBox;
use crate::utils::{
    bessel_log_volume, gumbel_lse_max, gumbel_lse_min, gumbel_membership_prob,
    map_gumbel_to_bounds, sample_gumbel,
};
use crate::{Box, BoxError};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// A Gumbel box embedding implemented using `candle_core::Tensor`.
///
/// # Temperature semantics
///
/// This type stores a single temperature used for all operations (intersection,
/// volume, containment). When two GumbelBoxes with different temperatures are
/// intersected, `self`'s temperature is used. This makes intersection
/// argument-order-dependent when temperatures differ.
///
/// The [`Box`] trait methods use Gumbel-specific formulas:
/// - `volume`: Bessel/softplus approximation (Dasgupta et al., 2020)
/// - `intersection`: log-sum-exp smooth min/max
/// - `containment_prob`: `Vol(self cap other) / Vol(other)` using Gumbel volumes
/// - `overlap_prob`: `Vol(self cap other) / Vol(self cup other)` using Gumbel volumes
#[derive(Debug, Clone)]
pub struct CandleGumbelBox {
    inner: CandleBox,
}

impl CandleGumbelBox {
    /// Create a new CandleGumbelBox.
    pub fn new(min: Tensor, max: Tensor, temperature: f32) -> std::result::Result<Self, BoxError> {
        Ok(Self {
            inner: CandleBox::new(min, max, temperature)?,
        })
    }
}

impl Box for CandleGumbelBox {
    type Scalar = f32;
    type Vector = Tensor;

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
    fn volume(&self, _temperature: Self::Scalar) -> std::result::Result<Self::Scalar, BoxError> {
        let t = self.inner.temperature;
        let mins = self
            .min()
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let maxs = self
            .max()
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let (_, vol) = bessel_log_volume(&mins, &maxs, t, t);
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
    fn intersection(&self, other: &Self) -> std::result::Result<Self, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let t = self.inner.temperature;
        let n = self.dim();

        let self_min = self
            .min()
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let self_max = self
            .max()
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let other_min = other
            .min()
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let other_max = other
            .max()
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;

        let mut new_min = Vec::with_capacity(n);
        let mut new_max = Vec::with_capacity(n);

        for d in 0..n {
            new_min.push(gumbel_lse_min(self_min[d], other_min[d], t));
            new_max.push(gumbel_lse_max(self_max[d], other_max[d], t));
        }

        let device = self.min().device();
        let min_tensor =
            Tensor::new(&new_min[..], device).map_err(|e| BoxError::Internal(e.to_string()))?;
        let max_tensor =
            Tensor::new(&new_max[..], device).map_err(|e| BoxError::Internal(e.to_string()))?;

        // No hard clipping: softplus volume handles "flipped" boxes (z > Z)
        // gracefully by returning near-zero volume.
        Ok(Self {
            inner: CandleBox::new_unchecked(min_tensor, max_tensor, t),
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
    ) -> std::result::Result<Self::Scalar, BoxError> {
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
    ) -> std::result::Result<Self::Scalar, BoxError> {
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

    fn union(&self, other: &Self) -> std::result::Result<Self, BoxError> {
        Ok(Self {
            inner: self.inner.union(&other.inner)?,
        })
    }

    fn center(&self) -> std::result::Result<Self::Vector, BoxError> {
        self.inner.center()
    }

    fn distance(&self, other: &Self) -> std::result::Result<Self::Scalar, BoxError> {
        self.inner.distance(&other.inner)
    }

    fn truncate(&self, k: usize) -> std::result::Result<Self, BoxError> {
        Ok(Self {
            inner: self.inner.truncate(k)?,
        })
    }
}

impl CandleGumbelBox {
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
    pub fn membership_probability(&self, point: &Tensor) -> std::result::Result<f32, BoxError> {
        if point.dims() != [self.dim()] {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: point.dims().len(),
            });
        }

        // P(point ∈ box) = ∏ P(min\[i\] <= point\[i\] <= max\[i\])
        // Using numerically stable Gumbel-Softmax probability calculation
        let temp = self.temperature();

        // For each dimension: P(min\[i\] <= point\[i\] <= max\[i\])
        let point_data = point
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let min_data = self
            .min()
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let max_data = self
            .max()
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;

        let mut prob = 1.0;
        for (i, &coord) in point_data.iter().enumerate() {
            // Use numerically stable Gumbel-Softmax probability calculation
            let dim_prob = gumbel_membership_prob(coord, min_data[i], max_data[i], temp);
            prob *= dim_prob;
        }

        Ok(prob)
    }

    /// Sample a point from the box distribution using Gumbel-Softmax.
    pub fn sample(&self) -> Tensor {
        use candle_core::Tensor;

        // Use LCG for pseudo-random sampling to avoid rand dependency conflicts.
        // Note: LCG has known limitations but is sufficient for non-cryptographic use.
        let device = self.min().device();
        let dim = self.dim();

        let min_data = self.min().to_vec1::<f32>().unwrap_or_default();
        let max_data = self.max().to_vec1::<f32>().unwrap_or_default();

        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: u64 = 1u64 << 32;

        let temp = self.temperature();
        let mut samples: Vec<f32> = Vec::with_capacity(dim);
        for i in 0..dim {
            seed = (A.wrapping_mul(seed).wrapping_add(C)) % M;
            let u = (seed as f32) / (M as f32);
            let gumbel = sample_gumbel(u, 1e-7);
            let value = map_gumbel_to_bounds(gumbel, min_data[i], max_data[i], temp);
            samples.push(value);
        }

        Tensor::new(&*samples, device).unwrap_or_else(|e| {
            // Fallback: return center point if tensor creation fails
            // This should be rare - only occurs if device allocation fails
            let center: Vec<f32> = (0..dim)
                .map(|i| (min_data[i] + max_data[i]) / 2.0)
                .collect();
            Tensor::new(&*center, device).unwrap_or_else(|_| {
                // If fallback also fails, try creating zero tensor on same device
                // This is better than panicking, though indicates serious device issues
                eprintln!(
                    "WARNING: Failed to create sample tensor (original error: {}). Using zero tensor as fallback.",
                    e
                );
                // Return a zero tensor - this is a degraded state but better than panic
                // If this also fails, we have no choice but to panic (indicates system failure)
                Tensor::zeros(&[dim], candle_core::DType::F32, device)
                    .unwrap_or_else(|_| {
                        panic!(
                            "CRITICAL: Cannot create any tensor on device {:?}. System may be out of memory or device unavailable. Original error: {}",
                            device, e
                        )
                    })
            })
        })
    }
}

impl Serialize for CandleGumbelBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Delegate to inner CandleBox serialization
        self.inner.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CandleGumbelBox {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = CandleBox::deserialize(deserializer)?;
        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gumbel_box_membership_uses_gumbel_math() {
        // CandleGumbelBox.membership_probability uses Gumbel-Softmax, not hard indicator.
        // A point exactly at min should have probability < 1.0 (Gumbel smoothing).
        let device = candle_core::Device::Cpu;
        let min = Tensor::new(&[0.0f32, 0.0], &device).unwrap();
        let max = Tensor::new(&[1.0f32, 1.0], &device).unwrap();
        let gb = CandleGumbelBox::new(min, max, 1.0).unwrap();

        let boundary_point = Tensor::new(&[0.0f32, 0.0], &device).unwrap();
        let prob = gb.membership_probability(&boundary_point).unwrap();
        // Gumbel smoothing: boundary point has ~0.25 probability, not 1.0
        assert!(
            prob > 0.0 && prob < 1.0,
            "boundary prob should be smooth, got {}",
            prob
        );

        let center_point = Tensor::new(&[0.5f32, 0.5], &device).unwrap();
        let center_prob = gb.membership_probability(&center_point).unwrap();
        assert!(
            center_prob > prob,
            "center should have higher prob than boundary"
        );
    }

    #[test]
    fn gumbel_box_volume_uses_bessel() {
        // CandleGumbelBox.volume uses Bessel/softplus, not hard-box product.
        let device = candle_core::Device::Cpu;
        let min = Tensor::new(&[0.0f32, 0.0], &device).unwrap();
        let max = Tensor::new(&[2.0f32, 3.0], &device).unwrap();
        let gb = CandleGumbelBox::new(min.clone(), max.clone(), 1.0).unwrap();
        let hb = CandleBox::new(min, max, 1.0).unwrap();

        let gumbel_vol = gb.volume(1.0).unwrap();
        let hard_vol = hb.volume(1.0).unwrap();
        // Bessel volume is smaller than hard volume due to 2*gamma*T offset
        assert!(
            gumbel_vol < hard_vol,
            "Bessel vol ({}) should be < hard vol ({})",
            gumbel_vol,
            hard_vol
        );
        assert!(
            gumbel_vol > 0.0,
            "Bessel vol should be positive, got {}",
            gumbel_vol
        );
    }

    #[test]
    fn gumbel_box_volume_approaches_hard_at_low_temperature() {
        let device = candle_core::Device::Cpu;
        let min = Tensor::new(&[0.0f32, 0.0], &device).unwrap();
        let max = Tensor::new(&[3.0f32, 4.0], &device).unwrap();
        let gb = CandleGumbelBox::new(min, max, 0.01).unwrap();
        let vol = gb.volume(0.01).unwrap();
        let hard_vol = 3.0 * 4.0;
        assert!(
            (vol - hard_vol).abs() / hard_vol < 0.05,
            "At low T, Bessel vol ({}) should be close to hard vol ({})",
            vol,
            hard_vol
        );
    }

    #[test]
    fn gumbel_box_containment_monotonicity() {
        let device = candle_core::Device::Cpu;
        let a = CandleGumbelBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).unwrap(),
            Tensor::new(&[10.0f32, 10.0], &device).unwrap(),
            1.0,
        )
        .unwrap();
        let b = CandleGumbelBox::new(
            Tensor::new(&[1.0f32, 1.0], &device).unwrap(),
            Tensor::new(&[9.0f32, 9.0], &device).unwrap(),
            1.0,
        )
        .unwrap();
        let c = CandleGumbelBox::new(
            Tensor::new(&[2.0f32, 2.0], &device).unwrap(),
            Tensor::new(&[8.0f32, 8.0], &device).unwrap(),
            1.0,
        )
        .unwrap();

        let p_b_in_a = a.containment_prob(&b, 1.0).unwrap();
        let p_c_in_a = a.containment_prob(&c, 1.0).unwrap();
        let p_c_in_b = b.containment_prob(&c, 1.0).unwrap();

        assert!(p_b_in_a > 0.7, "B should be inside A, got {}", p_b_in_a);
        assert!(p_c_in_a > 0.7, "C should be inside A, got {}", p_c_in_a);
        assert!(p_c_in_b > 0.7, "C should be inside B, got {}", p_c_in_b);
    }

    #[test]
    fn gumbel_box_disjoint_near_zero_containment() {
        let device = candle_core::Device::Cpu;
        let a = CandleGumbelBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).unwrap(),
            Tensor::new(&[1.0f32, 1.0], &device).unwrap(),
            1.0,
        )
        .unwrap();
        let b = CandleGumbelBox::new(
            Tensor::new(&[10.0f32, 10.0], &device).unwrap(),
            Tensor::new(&[11.0f32, 11.0], &device).unwrap(),
            1.0,
        )
        .unwrap();
        let p = a.containment_prob(&b, 1.0).unwrap();
        assert!(
            p < 0.01,
            "disjoint boxes should have near-zero containment, got {}",
            p
        );
    }

    #[test]
    fn gumbel_box_overlap_reasonable() {
        let device = candle_core::Device::Cpu;
        let a = CandleGumbelBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).unwrap(),
            Tensor::new(&[3.0f32, 3.0], &device).unwrap(),
            1.0,
        )
        .unwrap();
        let b = CandleGumbelBox::new(
            Tensor::new(&[1.0f32, 1.0], &device).unwrap(),
            Tensor::new(&[4.0f32, 4.0], &device).unwrap(),
            1.0,
        )
        .unwrap();
        let p = a.overlap_prob(&b, 1.0).unwrap();
        assert!(p > 0.0, "overlapping boxes should have positive overlap");
        assert!(p <= 1.0, "overlap should be <= 1.0");
    }

    #[test]
    fn gumbel_box_intersection_uses_lse() {
        let device = candle_core::Device::Cpu;
        let a = CandleGumbelBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).unwrap(),
            Tensor::new(&[2.0f32, 2.0], &device).unwrap(),
            1.0,
        )
        .unwrap();
        let b = CandleGumbelBox::new(
            Tensor::new(&[1.0f32, 1.0], &device).unwrap(),
            Tensor::new(&[3.0f32, 3.0], &device).unwrap(),
            1.0,
        )
        .unwrap();
        let inter = a.intersection(&b).unwrap();
        let vol = inter.volume(1.0).unwrap();
        // LSE intersection volume should be positive but less than hard intersection (1.0)
        assert!(vol > 0.0, "LSE intersection should have positive vol");
        assert!(
            vol < 1.0,
            "LSE intersection vol ({}) should be < hard vol (1.0)",
            vol
        );
    }
}
