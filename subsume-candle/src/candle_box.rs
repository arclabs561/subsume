//! Candle implementation of Box trait.

use candle_core::{Device, Result, Tensor};
use subsume_core::{Box, BoxError};

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
        let min_data = min.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        let max_data = max.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
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
        let diff = self.max.sub(&self.min).map_err(|e| BoxError::Internal(e.to_string()))?;
        let volume = diff.prod_all::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        Ok(volume)
    }

    fn intersection(&self, other: &Self) -> Result<Self, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let intersection_min = self.min.maximum(&other.min).map_err(|e| BoxError::Internal(e.to_string()))?;
        let intersection_max = self.max.minimum(&other.max).map_err(|e| BoxError::Internal(e.to_string()))?;

        // Check if intersection is valid (not disjoint)
        // If min > max in any dimension, boxes are disjoint
        let min_data = intersection_min.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        let max_data = intersection_max.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
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

    fn union(&self, other: &Self) -> Result<Self, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let union_min = self.min.minimum(&other.min).map_err(|e| BoxError::Internal(e.to_string()))?;
        let union_max = self.max.maximum(&other.max).map_err(|e| BoxError::Internal(e.to_string()))?;

        Self::new(union_min, union_max, self.temperature)
    }

    fn center(&self) -> Result<Self::Vector, BoxError> {
        // Center = (min + max) / 2
        let sum = self.min.add(&self.max).map_err(|e| BoxError::Internal(e.to_string()))?;
        let two = Tensor::new(&[2.0f32], self.min.device()).map_err(|e| BoxError::Internal(e.to_string()))?;
        let center = sum.broadcast_div(&two).map_err(|e| BoxError::Internal(e.to_string()))?;
        Ok(center)
    }

    fn distance(&self, other: &Self) -> Result<Self::Scalar, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        // Check if boxes overlap (distance = 0)
        let intersection = self.intersection(other)?;
        let intersection_vol = intersection.volume(1.0)?;
        if intersection_vol > 0.0 {
            return Ok(0.0);
        }

        // Compute minimum distance between two axis-aligned boxes
        // For each dimension, compute the gap (or 0 if overlapping)
        // Use element-wise operations on tensors
        let self_min_data = self.min.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        let self_max_data = self.max.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        let other_min_data = other.min.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        let other_max_data = other.max.to_vec1::<f32>().map_err(|e| BoxError::Internal(e.to_string()))?;
        
        let mut dist_sq = 0.0;
        for i in 0..self.dim() {
            let gap = if self_max_data[i] < other_min_data[i] {
                other_min_data[i] - self_max_data[i]
            } else if other_max_data[i] < self_min_data[i] {
                self_min_data[i] - other_max_data[i]
            } else {
                0.0
            };
            dist_sq += gap * gap;
        }
        
        Ok(dist_sq.sqrt())
    }
}

