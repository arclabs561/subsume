//! Candle implementation of GumbelBox trait.

use candle_core::Tensor;
use subsume_core::{Box, BoxError, GumbelBox, gumbel_membership_prob, sample_gumbel, map_gumbel_to_bounds};
use crate::candle_box::CandleBox;

/// A Gumbel box embedding implemented using `candle_core::Tensor`.
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
}

impl GumbelBox for CandleGumbelBox {
    fn temperature(&self) -> Self::Scalar {
        self.inner.temperature
    }

    fn membership_probability(&self, point: &Self::Vector) -> std::result::Result<Self::Scalar, BoxError> {
        if point.dims() != &[self.dim()] {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: point.dims().len(),
            });
        }

        // P(point ∈ box) = ∏ P(min[i] <= point[i] <= max[i])
        // Using numerically stable Gumbel-Softmax probability calculation
        let temp = self.temperature();

        // For each dimension: P(min[i] <= point[i] <= max[i])
        let point_data = point.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        let min_data = self.min().to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        let max_data = self.max().to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;

        let mut prob = 1.0;
        for (i, &coord) in point_data.iter().enumerate() {
            // Use numerically stable Gumbel-Softmax probability calculation
            let dim_prob = gumbel_membership_prob(
                coord,
                min_data[i],
                max_data[i],
                temp,
            );
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
            let value = map_gumbel_to_bounds(
                gumbel,
                min_data[i],
                max_data[i],
                temp,
            );
            samples.push(value);
        }
        
        Tensor::new(&*samples, device).unwrap_or_else(|e| {
            // Fallback: return center point if tensor creation fails
            // This should be rare - only occurs if device allocation fails
            let center: Vec<f32> = (0..dim)
                .map(|i| (min_data[i] + max_data[i]) / 2.0)
                .collect();
            Tensor::new(&*center, device).unwrap_or_else(|_| {
                panic!("Failed to create sample tensor: original error: {}, fallback also failed", e)
            })
        })
    }
}

