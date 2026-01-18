//! Candle implementation of Box trait.

use candle_core::Tensor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
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
    /// Returns `BoxError` if min/max have different shapes or if any min\[i\] > max\[i\].
    pub fn new(min: Tensor, max: Tensor, temperature: f32) -> std::result::Result<Self, BoxError> {
        if min.shape() != max.shape() {
            return Err(BoxError::DimensionMismatch {
                expected: min.dims().len(),
                actual: max.dims().len(),
            });
        }

        // Validate bounds
        let min_data = min
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let max_data = max
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
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

    fn volume(&self, _temperature: Self::Scalar) -> std::result::Result<Self::Scalar, BoxError> {
        // Volume = ∏(max\[i\] - min\[i\])
        let diff = self
            .max
            .sub(&self.min)
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        // Compute product by iterating over elements
        let diff_data = diff
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let volume = diff_data.iter().product::<f32>();
        Ok(volume)
    }

    fn intersection(&self, other: &Self) -> std::result::Result<Self, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let intersection_min = self
            .min
            .maximum(&other.min)
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let intersection_max = self
            .max
            .minimum(&other.max)
            .map_err(|e| BoxError::Internal(e.to_string()))?;

        // Check if intersection is valid (not disjoint)
        // If min > max in any dimension, boxes are disjoint
        let min_data = intersection_min
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let max_data = intersection_max
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        for (min_val, max_val) in min_data.iter().zip(max_data.iter()) {
            if min_val > max_val {
                // Boxes are disjoint - return a zero-volume box
                return Self::new(
                    intersection_min.clone(),
                    intersection_min.clone(),
                    self.temperature,
                );
            }
        }

        Self::new(intersection_min, intersection_max, self.temperature)
    }

    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> std::result::Result<Self::Scalar, BoxError> {
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
    ) -> std::result::Result<Self::Scalar, BoxError> {
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

    fn union(&self, other: &Self) -> std::result::Result<Self, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let union_min = self
            .min
            .minimum(&other.min)
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let union_max = self
            .max
            .maximum(&other.max)
            .map_err(|e| BoxError::Internal(e.to_string()))?;

        Self::new(union_min, union_max, self.temperature)
    }

    fn center(&self) -> std::result::Result<Self::Vector, BoxError> {
        // Center = (min + max) / 2
        let sum = self
            .min
            .add(&self.max)
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let two = Tensor::new(&[2.0f32], self.min.device())
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let center = sum
            .broadcast_div(&two)
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        Ok(center)
    }

    fn distance(&self, other: &Self) -> std::result::Result<Self::Scalar, BoxError> {
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
        let self_min_data = self
            .min
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let self_max_data = self
            .max
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let other_min_data = other
            .min
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let other_max_data = other
            .max
            .to_vec1::<f32>()
            .map_err(|e| BoxError::Internal(e.to_string()))?;

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

    fn truncate(&self, k: usize) -> std::result::Result<Self, BoxError> {
        let d = self.dim();
        if k > d {
            return Err(BoxError::DimensionMismatch {
                expected: d,
                actual: k,
            });
        }
        if k == d {
            return Ok(self.clone());
        }

        // Expect 1-D tensors of shape [d].
        let min_t = self
            .min
            .narrow(0, 0, k)
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        let max_t = self
            .max
            .narrow(0, 0, k)
            .map_err(|e| BoxError::Internal(e.to_string()))?;
        Self::new(min_t, max_t, self.temperature)
    }
}

impl Serialize for CandleBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("CandleBox", 3)?;

        // Convert tensors to Vec<f32> for serialization
        let min_vec = self.min.to_vec1::<f32>().map_err(|e| {
            serde::ser::Error::custom(format!("Failed to serialize min tensor: {}", e))
        })?;
        let max_vec = self.max.to_vec1::<f32>().map_err(|e| {
            serde::ser::Error::custom(format!("Failed to serialize max tensor: {}", e))
        })?;

        state.serialize_field("min", &min_vec)?;
        state.serialize_field("max", &max_vec)?;
        state.serialize_field("temperature", &self.temperature)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for CandleBox {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Min,
            Max,
            Temperature,
        }

        struct CandleBoxVisitor;

        impl<'de> Visitor<'de> for CandleBoxVisitor {
            type Value = CandleBox;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CandleBox")
            }

            fn visit_map<V>(self, mut map: V) -> Result<CandleBox, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut min_vec: Option<Vec<f32>> = None;
                let mut max_vec: Option<Vec<f32>> = None;
                let mut temperature: Option<f32> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Min => {
                            if min_vec.is_some() {
                                return Err(de::Error::duplicate_field("min"));
                            }
                            min_vec = Some(map.next_value()?);
                        }
                        Field::Max => {
                            if max_vec.is_some() {
                                return Err(de::Error::duplicate_field("max"));
                            }
                            max_vec = Some(map.next_value()?);
                        }
                        Field::Temperature => {
                            if temperature.is_some() {
                                return Err(de::Error::duplicate_field("temperature"));
                            }
                            temperature = Some(map.next_value()?);
                        }
                    }
                }

                let min_vec = min_vec.ok_or_else(|| de::Error::missing_field("min"))?;
                let max_vec = max_vec.ok_or_else(|| de::Error::missing_field("max"))?;
                let temperature = temperature.unwrap_or(1.0);

                // Reconstruct tensors from vectors
                // Note: We use CPU device by default during deserialization
                // Users can move tensors to GPU after deserialization if needed
                let device = candle_core::Device::Cpu;
                let min_tensor = Tensor::new(&min_vec[..], &device).map_err(|e| {
                    de::Error::custom(format!("Failed to create min tensor: {}", e))
                })?;
                let max_tensor = Tensor::new(&max_vec[..], &device).map_err(|e| {
                    de::Error::custom(format!("Failed to create max tensor: {}", e))
                })?;

                CandleBox::new(min_tensor, max_tensor, temperature)
                    .map_err(|e| de::Error::custom(format!("Failed to create CandleBox: {}", e)))
            }
        }

        const FIELDS: &[&str] = &["min", "max", "temperature"];
        deserializer.deserialize_struct("CandleBox", FIELDS, CandleBoxVisitor)
    }
}
