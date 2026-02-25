//! Ndarray implementation of Box trait.

use crate::{Box, BoxError};
use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
        use crate::log_space_volume;

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
        // See [`docs/typst-output/pdf/gumbel-box-volume.pdf`](../../docs/typst-output/pdf/gumbel-box-volume.pdf)
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
        _temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        // Containment probability: P(other ⊆ self) = Vol(self ∩ other) / Vol(other).
        //
        // IMPORTANT: this is a hot path (training + evaluation). Constructing an explicit
        // intersection box allocates and is avoidable. We compute the intersection volume
        // directly in one pass over dimensions, with the same numerical strategy as `volume()`:
        // use log-space for higher dimensions to avoid underflow/overflow.

        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        use crate::log_space_volume;

        let dim = self.dim();

        if dim > 10 {
            let mut i = 0usize;
            let (_log_other, other_vol) = log_space_volume(std::iter::from_fn(|| {
                if i >= dim {
                    return None;
                }
                let len = (other.max[i] - other.min[i]).max(0.0);
                i += 1;
                Some(len)
            }));

            if other_vol <= 0.0 {
                return Err(BoxError::ZeroVolume);
            }

            let mut j = 0usize;
            let (_log_intersection, intersection_vol) =
                log_space_volume(std::iter::from_fn(|| {
                    if j >= dim {
                        return None;
                    }
                    let lo = self.min[j].max(other.min[j]);
                    let hi = self.max[j].min(other.max[j]);
                    let len = (hi - lo).max(0.0);
                    j += 1;
                    Some(len)
                }));

            Ok((intersection_vol / other_vol).clamp(0.0, 1.0))
        } else {
            let mut intersection_vol = 1.0f32;
            let mut other_vol = 1.0f32;

            for k in 0..dim {
                let other_len = other.max[k] - other.min[k];
                other_vol *= other_len.max(0.0);

                let lo = self.min[k].max(other.min[k]);
                let hi = self.max[k].min(other.max[k]);
                intersection_vol *= (hi - lo).max(0.0);
            }

            if other_vol <= 0.0 {
                return Err(BoxError::ZeroVolume);
            }

            Ok((intersection_vol / other_vol).clamp(0.0, 1.0))
        }
    }

    fn containment_prob_many(
        &self,
        others: &[Self],
        _temperature: Self::Scalar,
        out: &mut [Self::Scalar],
    ) -> Result<(), BoxError> {
        if out.len() < others.len() {
            return Err(BoxError::Internal(format!(
                "output buffer too small: need {}, got {}",
                others.len(),
                out.len()
            )));
        }

        use crate::log_space_volume;

        let dim = self.dim();

        for (idx, other) in others.iter().enumerate() {
            if dim != other.dim() {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: other.dim(),
                });
            }

            let p = if dim > 10 {
                let mut i = 0usize;
                let (_log_other, other_vol) = log_space_volume(std::iter::from_fn(|| {
                    if i >= dim {
                        return None;
                    }
                    let len = (other.max[i] - other.min[i]).max(0.0);
                    i += 1;
                    Some(len)
                }));

                if other_vol <= 0.0 {
                    return Err(BoxError::ZeroVolume);
                }

                let mut j = 0usize;
                let (_log_intersection, intersection_vol) =
                    log_space_volume(std::iter::from_fn(|| {
                        if j >= dim {
                            return None;
                        }
                        let lo = self.min[j].max(other.min[j]);
                        let hi = self.max[j].min(other.max[j]);
                        let len = (hi - lo).max(0.0);
                        j += 1;
                        Some(len)
                    }));

                (intersection_vol / other_vol).clamp(0.0, 1.0)
            } else {
                let mut intersection_vol = 1.0f32;
                let mut other_vol = 1.0f32;

                for k in 0..dim {
                    let other_len = other.max[k] - other.min[k];
                    other_vol *= other_len.max(0.0);

                    let lo = self.min[k].max(other.min[k]);
                    let hi = self.max[k].min(other.max[k]);
                    intersection_vol *= (hi - lo).max(0.0);
                }

                if other_vol <= 0.0 {
                    return Err(BoxError::ZeroVolume);
                }

                (intersection_vol / other_vol).clamp(0.0, 1.0)
            };

            out[idx] = p;
        }

        Ok(())
    }

    fn overlap_prob(
        &self,
        other: &Self,
        _temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError> {
        // Overlap probability: Vol(self ∩ other) / Vol(self ∪ other).
        //
        // Same optimization as `containment_prob`: avoid allocating an intersection box.
        use crate::log_space_volume;

        if self.dim() != other.dim() {
            return Err(BoxError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let dim = self.dim();

        let (intersection_vol, vol_a, vol_b) = if dim > 10 {
            let mut i = 0usize;
            let (_, vol_a) = log_space_volume(std::iter::from_fn(|| {
                if i >= dim {
                    return None;
                }
                let len = (self.max[i] - self.min[i]).max(0.0);
                i += 1;
                Some(len)
            }));

            let mut j = 0usize;
            let (_, vol_b) = log_space_volume(std::iter::from_fn(|| {
                if j >= dim {
                    return None;
                }
                let len = (other.max[j] - other.min[j]).max(0.0);
                j += 1;
                Some(len)
            }));

            let mut k = 0usize;
            let (_, intersection_vol) = log_space_volume(std::iter::from_fn(|| {
                if k >= dim {
                    return None;
                }
                let lo = self.min[k].max(other.min[k]);
                let hi = self.max[k].min(other.max[k]);
                let len = (hi - lo).max(0.0);
                k += 1;
                Some(len)
            }));

            (intersection_vol, vol_a, vol_b)
        } else {
            let mut intersection_vol = 1.0f32;
            let mut vol_a = 1.0f32;
            let mut vol_b = 1.0f32;

            for k in 0..dim {
                vol_a *= (self.max[k] - self.min[k]).max(0.0);
                vol_b *= (other.max[k] - other.min[k]).max(0.0);

                let lo = self.min[k].max(other.min[k]);
                let hi = self.max[k].min(other.max[k]);
                intersection_vol *= (hi - lo).max(0.0);
            }

            (intersection_vol, vol_a, vol_b)
        };

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

    fn truncate(&self, k: usize) -> Result<Self, BoxError> {
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
        Self::new(
            self.min.slice(ndarray::s![..k]).to_owned(),
            self.max.slice(ndarray::s![..k]).to_owned(),
            self.temperature,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Box as BoxTrait;
    use ndarray::array;

    // ---- Intersection edge cases ----

    #[test]
    fn intersection_disjoint_boxes_has_zero_volume() {
        // Two boxes that do not overlap in any dimension.
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![2.0, 2.0], array![3.0, 3.0], 1.0).unwrap();
        let inter = a.intersection(&b).unwrap();
        let vol = inter.volume(1.0).unwrap();
        assert_eq!(vol, 0.0, "Disjoint boxes must have zero intersection volume");
    }

    #[test]
    fn intersection_partial_overlap() {
        // Boxes overlap in a sub-region.
        let a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let inter = a.intersection(&b).unwrap();
        // Intersection should be [1,1] to [2,2], volume = 1.0
        let vol = inter.volume(1.0).unwrap();
        assert!(
            (vol - 1.0).abs() < 1e-6,
            "Partial intersection volume should be 1.0, got {}",
            vol
        );
    }

    #[test]
    fn intersection_full_containment() {
        // B is fully inside A; intersection = B.
        let a = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let inter = a.intersection(&b).unwrap();
        let vol_inter = inter.volume(1.0).unwrap();
        let vol_b = b.volume(1.0).unwrap();
        assert!(
            (vol_inter - vol_b).abs() < 1e-6,
            "When B is inside A, intersection volume should equal B's volume"
        );
    }

    #[test]
    fn intersection_disjoint_in_one_dimension() {
        // Overlap in x but disjoint in y.
        let a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 2.0], array![3.0, 3.0], 1.0).unwrap();
        let inter = a.intersection(&b).unwrap();
        let vol = inter.volume(1.0).unwrap();
        assert_eq!(
            vol, 0.0,
            "Disjoint in one dimension means zero intersection volume"
        );
    }

    // ---- Union edge cases ----

    #[test]
    fn union_volume_at_least_max_of_parts() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let u = a.union(&b).unwrap();
        let vol_u = u.volume(1.0).unwrap();
        let vol_a = a.volume(1.0).unwrap();
        let vol_b = b.volume(1.0).unwrap();
        assert!(
            vol_u >= vol_a - 1e-6,
            "Union volume {} must be >= volume of A {}",
            vol_u,
            vol_a
        );
        assert!(
            vol_u >= vol_b - 1e-6,
            "Union volume {} must be >= volume of B {}",
            vol_u,
            vol_b
        );
    }

    #[test]
    fn union_of_identical_boxes_equals_self() {
        let a = NdarrayBox::new(array![1.0, 2.0], array![3.0, 4.0], 1.0).unwrap();
        let u = a.union(&a).unwrap();
        let vol_a = a.volume(1.0).unwrap();
        let vol_u = u.volume(1.0).unwrap();
        assert!(
            (vol_a - vol_u).abs() < 1e-6,
            "Union of a box with itself should have the same volume"
        );
    }

    // ---- Truncation / Matryoshka ----

    #[test]
    fn truncation_preserves_containment() {
        // If B is inside A in full dimensions, truncating both to fewer
        // dimensions should preserve containment.
        let a = NdarrayBox::new(array![0.0, 0.0, 0.0], array![4.0, 4.0, 4.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0, 1.0], array![3.0, 3.0, 3.0], 1.0).unwrap();

        let full_prob = a.containment_prob(&b, 1.0).unwrap();
        assert!(full_prob > 0.99, "B should be inside A in full dims");

        for k in 1..=3 {
            let a_trunc = a.truncate(k).unwrap();
            let b_trunc = b.truncate(k).unwrap();
            let trunc_prob = a_trunc.containment_prob(&b_trunc, 1.0).unwrap();
            assert!(
                trunc_prob > 0.99,
                "Containment should be preserved when truncating to {} dims, got {}",
                k,
                trunc_prob
            );
        }
    }

    #[test]
    fn truncation_reduces_dim() {
        let a = NdarrayBox::new(array![0.0, 0.0, 0.0, 0.0], array![1.0, 1.0, 1.0, 1.0], 1.0)
            .unwrap();
        let t = a.truncate(2).unwrap();
        assert_eq!(t.dim(), 2);
    }

    #[test]
    fn truncation_to_full_dim_is_identity() {
        let a = NdarrayBox::new(array![0.0, 1.0], array![2.0, 3.0], 1.0).unwrap();
        let t = a.truncate(2).unwrap();
        assert_eq!(t.min(), a.min());
        assert_eq!(t.max(), a.max());
    }

    // ---- Temperature edge cases ----

    #[test]
    fn containment_prob_very_small_temperature() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        // NdarrayBox containment_prob ignores temperature for hard boxes,
        // but the call should not panic or produce NaN.
        let p = a.containment_prob(&b, 0.01).unwrap();
        assert!(p.is_finite(), "Containment prob must be finite at low temp");
        assert!(p >= 0.0 && p <= 1.0, "Containment prob must be in [0,1]");
    }

    #[test]
    fn containment_prob_very_large_temperature() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let p = a.containment_prob(&b, 100.0).unwrap();
        assert!(p.is_finite(), "Containment prob must be finite at high temp");
        assert!(p >= 0.0 && p <= 1.0, "Containment prob must be in [0,1]");
    }

    // ---- Serialization round-trip ----

    #[test]
    fn serde_json_round_trip() {
        let original =
            NdarrayBox::new(array![0.1, 0.2, 0.3], array![0.4, 0.5, 0.6], 0.75).unwrap();
        let json = serde_json::to_string(&original).expect("serialize");
        let deserialized: NdarrayBox = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original.dim(), deserialized.dim());
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

    // ---- Distance edge cases ----

    #[test]
    fn distance_overlapping_boxes_is_zero() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let d = a.distance(&b).unwrap();
        assert_eq!(d, 0.0, "Overlapping boxes should have zero distance");
    }

    #[test]
    fn distance_disjoint_boxes_is_positive() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![3.0, 3.0], array![4.0, 4.0], 1.0).unwrap();
        let d = a.distance(&b).unwrap();
        // Gap is (2, 2), distance = sqrt(8) ≈ 2.828
        assert!(d > 0.0, "Disjoint boxes should have positive distance");
        assert!(
            (d - 8.0_f32.sqrt()).abs() < 1e-5,
            "Expected sqrt(8), got {}",
            d
        );
    }

    #[test]
    fn distance_identical_boxes_is_zero() {
        let a = NdarrayBox::new(array![1.0, 2.0], array![3.0, 4.0], 1.0).unwrap();
        let d = a.distance(&a).unwrap();
        assert_eq!(d, 0.0, "Distance to self should be zero");
    }

    // ---- Containment semantics ----

    #[test]
    fn containment_prob_disjoint_is_zero() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![5.0, 5.0], array![6.0, 6.0], 1.0).unwrap();
        let p = a.containment_prob(&b, 1.0).unwrap();
        assert_eq!(p, 0.0, "Disjoint boxes should have zero containment");
    }

    #[test]
    fn containment_prob_full_containment_is_one() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let p = a.containment_prob(&b, 1.0).unwrap();
        assert!(
            (p - 1.0).abs() < 1e-6,
            "Full containment should give prob ~1.0, got {}",
            p
        );
    }

    #[test]
    fn overlap_prob_identical_boxes_is_one() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let p = a.overlap_prob(&a, 1.0).unwrap();
        assert!(
            (p - 1.0).abs() < 1e-6,
            "Identical boxes should have overlap prob 1.0, got {}",
            p
        );
    }

    #[test]
    fn overlap_prob_disjoint_is_zero() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![5.0, 5.0], array![6.0, 6.0], 1.0).unwrap();
        let p = a.overlap_prob(&b, 1.0).unwrap();
        assert_eq!(p, 0.0, "Disjoint boxes should have overlap prob 0.0");
    }
}
