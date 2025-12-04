//! Candle implementation of Box trait.

use candle_core::{Device, Result, Tensor};
use subsume_core::{Box, BoxError};

impl From<candle_core::Error> for BoxError {
    fn from(err: candle_core::Error) -> Self {
        BoxError::Internal(err.to_string())
    }
}

/// A box embedding implemented using `candle_core::Tensor`.
#[derive(Debug, Clone)]
pub struct CandleBox {
    /// Minimum bounds [d]
    min: Tensor,
    /// Maximum bounds [d]
    max: Tensor,
    /// Temperature for Gumbel-Softmax (1.0 = standard box)
    pub(crate) temperature: f32,
}

impl CandleBox {
    /// Create a new CandleBox.
    ///
    /// # Errors
    ///
    /// Returns `BoxError` if min/max have different shapes or if any min[i] > max[i].
    pub fn new(min: Tensor, max: Tensor, temperature: f32) -> Result<Self, BoxError> {
        if min.shape() != max.shape() {
            return Err(BoxError::DimensionMismatch {
                expected: min.dims().len(),
                actual: max.dims().len(),
            });
        }

        // Validate bounds
        let min_data = min.to_vec1::<f32>()?;
        let max_data = max.to_vec1::<f32>()?;
        for (i, (&m, &max_val)) in min_data.iter().zip(max_data.iter()).enumerate() {
            if m > max_val {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: m as f64,
                    max: max_val as f64,
                });
            }
        }

        Ok(Self {
            min,
            max,
            temperature,
        })
    }
}

impl Box for CandleBox {
    type Scalar = f32;
    type Vector = Tensor;

    fn min(&self) -> &Self::Vector {
        &self.min
    }

    fn max(&self) -> &Self::Vector {
        &self.max
    }

    fn dim(&self) -> usize {
        self.min.dims().iter().product()
    }

    fn volume(&self, temperature: Self::Scalar) -> Result<Self::Scalar, BoxError> {
        // Volume = ∏(max[i] - min[i])
        let diff = self.max.sub(&self.min)?;
        let volume = diff.prod_all::<f32>()?;
        Ok(volume)
    }

    fn intersection(&self, other: &Self) -> Result<Self, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let intersection_min = self.min.maximum(&other.min)?;
        let intersection_max = self.max.minimum(&other.max)?;

        // Check if intersection is valid (not disjoint)
        // If min > max in any dimension, boxes are disjoint
        let min_data = intersection_min.to_vec1::<f32>()?;
        let max_data = intersection_max.to_vec1::<f32>()?;
        for (min_val, max_val) in min_data.iter().zip(max_data.iter()) {
            if min_val > max_val {
                // Boxes are disjoint - return a zero-volume box
                return Self::new(intersection_min.clone(), intersection_min.clone(), self.temperature);
            }
        }

        Self::new(intersection_min, intersection_max, self.temperature)
    }

    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        // P(other ⊆ self) = intersection_volume(other, self) / other.volume()
        let intersection = self.intersection(other)?;
        let intersection_vol = intersection.volume(temperature)?;
        let other_vol = other.volume(temperature)?;

        if other_vol <= 0.0 {
            return Err(BoxError::ZeroVolume);
        }

        Ok((intersection_vol / other_vol).clamp(0.0, 1.0))
    }

    fn overlap_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        // P(self ∩ other ≠ ∅) = intersection_volume / union_volume
        // Using inclusion-exclusion: P(A ∩ B ≠ ∅) = intersection_vol(A, B) / union_vol(A, B)
        let intersection = self.intersection(other)?;
        let intersection_vol = intersection.volume(temperature)?;
        
        // Union volume = vol(A) + vol(B) - intersection_vol(A, B)
        let vol_a = self.volume(temperature)?;
        let vol_b = other.volume(temperature)?;
        let union_vol = vol_a + vol_b - intersection_vol;
        
        if union_vol <= 0.0 {
            return Ok(0.0);
        }
        
        Ok((intersection_vol / union_vol).clamp(0.0, 1.0))
    }
}

