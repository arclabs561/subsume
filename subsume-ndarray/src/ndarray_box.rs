//! Ndarray implementation of Box trait.

use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use subsume_core::{Box, BoxError};

/// A box embedding implemented using `ndarray::Array1<f32>`.
#[derive(Debug, Clone)]
pub struct NdarrayBox {
    /// Minimum bounds [d]
    min: Array1<f32>,
    /// Maximum bounds [d]
    max: Array1<f32>,
    /// Temperature for Gumbel-Softmax (1.0 = standard box)
    pub(crate) temperature: f32,
}

impl Serialize for NdarrayBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NdarrayBox", 3)?;
        state.serialize_field("min", &self.min.to_vec())?;
        state.serialize_field("max", &self.max.to_vec())?;
        state.serialize_field("temperature", &self.temperature)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for NdarrayBox {
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

        struct NdarrayBoxVisitor;

        impl<'de> Visitor<'de> for NdarrayBoxVisitor {
            type Value = NdarrayBox;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct NdarrayBox")
            }

            fn visit_map<V>(self, mut map: V) -> Result<NdarrayBox, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut min = None;
                let mut max = None;
                let mut temperature = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Min => {
                            if min.is_some() {
                                return Err(de::Error::duplicate_field("min"));
                            }
                            min = Some(map.next_value::<Vec<f32>>()?);
                        }
                        Field::Max => {
                            if max.is_some() {
                                return Err(de::Error::duplicate_field("max"));
                            }
                            max = Some(map.next_value::<Vec<f32>>()?);
                        }
                        Field::Temperature => {
                            if temperature.is_some() {
                                return Err(de::Error::duplicate_field("temperature"));
                            }
                            temperature = Some(map.next_value()?);
                        }
                    }
                }
                let min = min.ok_or_else(|| de::Error::missing_field("min"))?;
                let max = max.ok_or_else(|| de::Error::missing_field("max"))?;
                let temperature =
                    temperature.ok_or_else(|| de::Error::missing_field("temperature"))?;
                NdarrayBox::new(Array1::from(min), Array1::from(max), temperature)
                    .map_err(|e| de::Error::custom(format!("Invalid box: {}", e)))
            }
        }

        const FIELDS: &[&str] = &["min", "max", "temperature"];
        deserializer.deserialize_struct("NdarrayBox", FIELDS, NdarrayBoxVisitor)
    }
}

impl NdarrayBox {
    /// Create a new NdarrayBox.
    ///
    /// # Errors
    ///
    /// Returns `BoxError` if min/max have different shapes or if any min\[i\] > max\[i\].
    pub fn new(min: Array1<f32>, max: Array1<f32>, temperature: f32) -> Result<Self, BoxError> {
        if min.len() != max.len() {
            return Err(BoxError::DimensionMismatch {
                expected: min.len(),
                actual: max.len(),
            });
        }

        // Validate bounds
        for (i, (&m, &max_val)) in min.iter().zip(max.iter()).enumerate() {
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

impl Box for NdarrayBox {
    type Scalar = f32;
    type Vector = Array1<f32>;

    fn min(&self) -> &Self::Vector {
        &self.min
    }

    fn max(&self) -> &Self::Vector {
        &self.max
    }

    fn dim(&self) -> usize {
        self.min.len()
    }

    fn volume(&self, _temperature: Self::Scalar) -> Result<Self::Scalar, BoxError> {
        use subsume_core::log_space_volume;

        // Volume calculation: standard geometric volume (product of side lengths).
        //
        // For hard boxes, this is the exact volume. For Gumbel boxes, the theoretical
        // foundation uses the expected volume with Bessel function K_0:
        //   E[Vol(B)] = 2β K_0(2e^(-(μ_y - μ_x)/(2β)))
        //
        // However, in practice, we use the standard volume formula for computational
        // efficiency. The Bessel function provides the theoretical foundation for why
        // Gumbel boxes work (smooth gradients, local identifiability), but the implementation
        // uses simpler calculations that are mathematically equivalent in the regimes
        // where they're used.
        //
        // See [`docs/typst-output/pdf/gumbel-box-volume.pdf`](../../../docs/typst-output/pdf/gumbel-box-volume.pdf)
        // for the complete derivation from Gumbel distributions to Bessel functions.

        // For high-dimensional boxes, use log-space computation to avoid underflow/overflow
        let diff = &self.max - &self.min;
        let dim = self.dim();

        if dim > 10 {
            // High-dimensional: use log-space
            let (_, volume) = log_space_volume(diff.iter().copied());
            Ok(volume.max(0.0))
        } else {
            // Low-dimensional: direct multiplication is fine
            let volume = diff.iter().product::<f32>();
            Ok(volume.max(0.0))
        }
    }

    fn intersection(&self, other: &Self) -> Result<Self, BoxError> {
        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let intersection_min: Vec<f32> = self
            .min
            .iter()
            .zip(other.min.iter())
            .map(|(a, b)| a.max(*b))
            .collect();
        let intersection_max: Vec<f32> = self
            .max
            .iter()
            .zip(other.max.iter())
            .map(|(a, b)| a.min(*b))
            .collect();

        // Check if intersection is valid (not disjoint)
        for (min_val, max_val) in intersection_min.iter().zip(intersection_max.iter()) {
            if min_val > max_val {
                // Boxes are disjoint - return a zero-volume box
                let zero_min = Array1::from(intersection_min.clone());
                let zero_max = Array1::from(intersection_min.clone()); // min == max gives zero volume
                return Self::new(zero_min, zero_max, self.temperature);
            }
        }

        Self::new(
            Array1::from(intersection_min),
            Array1::from(intersection_max),
            self.temperature,
        )
    }

    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        // Containment probability: P(other ⊆ self) = Vol(intersection) / Vol(other)
        //
        // This implements the first-order Taylor approximation for containment probability:
        //   E[Vol(A ∩ B) / Vol(B)] ≈ E[Vol(A ∩ B)] / E[Vol(B)]
        //
        // For hard boxes, this is exact (deterministic volumes). For Gumbel boxes, this
        // approximation is accurate when the coefficient of variation is small (typically
        // when β < 0.2). The approximation breaks down when volumes are highly variable.
        //
        // See [`docs/typst-output/pdf/containment-probability.pdf`](../../../docs/typst-output/pdf/containment-probability.pdf)
        // for the complete derivation, error analysis, and validity conditions.

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

        let union_min: Array1<f32> = self
            .min
            .iter()
            .zip(other.min.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        let union_max: Array1<f32> = self
            .max
            .iter()
            .zip(other.max.iter())
            .map(|(a, b)| a.max(*b))
            .collect();

        Self::new(union_min, union_max, self.temperature)
    }

    fn center(&self) -> Result<Self::Vector, BoxError> {
        let center: Array1<f32> = self
            .min
            .iter()
            .zip(self.max.iter())
            .map(|(min_val, max_val)| (min_val + max_val) / 2.0)
            .collect();
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
        let mut dist_sq = 0.0;
        for i in 0..self.dim() {
            let gap = if self.max[i] < other.min[i] {
                other.min[i] - self.max[i]
            } else if other.max[i] < self.min[i] {
                self.min[i] - other.max[i]
            } else {
                0.0
            };
            dist_sq += gap * gap;
        }

        Ok(dist_sq.sqrt())
    }
}
