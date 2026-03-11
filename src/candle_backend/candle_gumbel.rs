//! Candle implementation of GumbelBox trait.

use crate::candle_backend::candle_box::CandleBox;
use crate::{
    gumbel_membership_prob, map_gumbel_to_bounds, sample_gumbel, Box, BoxError, GumbelBox,
};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// A Gumbel box embedding implemented using `candle_core::Tensor`.
///
/// # Approximation note
///
/// The [`GumbelBox`] trait methods ([`membership_probability`](GumbelBox::membership_probability),
/// [`sample`](GumbelBox::sample)) use proper Gumbel math. However, the [`Box`] trait methods
/// (`intersection`, `volume`, `containment_prob`, `overlap_prob`) delegate to [`CandleBox`]
/// (hard-box operations) rather than the Gumbel-specific formulas used by
/// [`NdarrayGumbelBox`](crate::ndarray_backend::NdarrayGumbelBox) (Bessel volume, LSE intersection).
///
/// For training workflows that require exact Gumbel volume and intersection, use the ndarray backend.
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

    fn volume(&self, temperature: Self::Scalar) -> std::result::Result<Self::Scalar, BoxError> {
        self.inner.volume(temperature)
    }

    fn intersection(&self, other: &Self) -> std::result::Result<Self, BoxError> {
        Ok(Self {
            inner: self.inner.intersection(&other.inner)?,
        })
    }

    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> std::result::Result<Self::Scalar, BoxError> {
        self.inner.containment_prob(&other.inner, temperature)
    }

    fn overlap_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> std::result::Result<Self::Scalar, BoxError> {
        self.inner.overlap_prob(&other.inner, temperature)
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

impl GumbelBox for CandleGumbelBox {
    fn temperature(&self) -> Self::Scalar {
        self.inner.temperature
    }

    fn membership_probability(
        &self,
        point: &Self::Vector,
    ) -> std::result::Result<Self::Scalar, BoxError> {
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

    fn sample(&self) -> Self::Vector {
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
    fn gumbel_box_volume_delegates_to_hard_box() {
        // Documents the approximation: volume uses hard-box math, not Bessel volume.
        let device = candle_core::Device::Cpu;
        let min = Tensor::new(&[0.0f32, 0.0], &device).unwrap();
        let max = Tensor::new(&[2.0f32, 3.0], &device).unwrap();
        let gb = CandleGumbelBox::new(min.clone(), max.clone(), 1.0).unwrap();
        let hb = CandleBox::new(min, max, 1.0).unwrap();

        let gumbel_vol = gb.volume(1.0).unwrap();
        let hard_vol = hb.volume(1.0).unwrap();
        assert!(
            (gumbel_vol - hard_vol).abs() < 1e-6,
            "CandleGumbelBox.volume delegates to CandleBox (hard-box approximation)"
        );
    }
}
