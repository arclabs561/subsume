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
            if !m.is_finite() || !max_val.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: m as f64,
                    max: max_val as f64,
                });
            }
            if m > max_val {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: m as f64,
                    max: max_val as f64,
                });
            }
        }

        // Validate temperature: must be finite and positive
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(BoxError::InvalidTemperature {
                value: temperature as f64,
                reason: "temperature must be finite and positive",
            });
        }

        Ok(Self {
            min,
            max,
            temperature,
        })
    }

    /// Create an NdarrayBox without bounds validation.
    ///
    /// Used internally by Gumbel LSE intersection, which can produce boxes
    /// where `min[d] > max[d]` ("flipped" boxes). The Bessel softplus volume
    /// handles these gracefully by returning near-zero volume.
    pub(crate) fn new_unchecked(min: Array1<f32>, max: Array1<f32>, temperature: f32) -> Self {
        debug_assert_eq!(min.len(), max.len());
        Self {
            min,
            max,
            temperature,
        }
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

        // Hard box volume: product of side lengths (Vilnis et al., 2018).
        // NdarrayGumbelBox overrides this with the Bessel/softplus volume
        // from Dasgupta et al. (2020).

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

    // Hard intersection (Vilnis et al., 2018). NdarrayGumbelBox overrides this
    // with log-sum-exp intersection from Dasgupta et al. (2020).
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
        assert_eq!(
            vol, 0.0,
            "Disjoint boxes must have zero intersection volume"
        );
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
        let a =
            NdarrayBox::new(array![0.0, 0.0, 0.0, 0.0], array![1.0, 1.0, 1.0, 1.0], 1.0).unwrap();
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
        assert!(
            (0.0..=1.0).contains(&p),
            "Containment prob must be in [0,1]"
        );
    }

    #[test]
    fn containment_prob_very_large_temperature() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let p = a.containment_prob(&b, 100.0).unwrap();
        assert!(
            p.is_finite(),
            "Containment prob must be finite at high temp"
        );
        assert!(
            (0.0..=1.0).contains(&p),
            "Containment prob must be in [0,1]"
        );
    }

    // ---- Serialization round-trip ----

    #[test]
    fn serde_json_round_trip() {
        let original = NdarrayBox::new(array![0.1, 0.2, 0.3], array![0.4, 0.5, 0.6], 0.75).unwrap();
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

    // ---- Construction error paths ----

    #[test]
    fn new_dimension_mismatch_returns_error() {
        let result = NdarrayBox::new(array![0.0, 0.0], array![1.0], 1.0);
        assert!(result.is_err());
        match result.unwrap_err() {
            BoxError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("Expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn new_invalid_bounds_returns_error() {
        // min[1] > max[1]
        let result = NdarrayBox::new(array![0.0, 5.0], array![1.0, 3.0], 1.0);
        assert!(result.is_err());
        match result.unwrap_err() {
            BoxError::InvalidBounds { dim, .. } => {
                assert_eq!(dim, 1);
            }
            other => panic!("Expected InvalidBounds, got {:?}", other),
        }
    }

    #[test]
    fn new_zero_width_box_is_valid() {
        // min == max is allowed (zero-volume box)
        let b = NdarrayBox::new(array![1.0, 2.0], array![1.0, 2.0], 1.0).unwrap();
        let vol = b.volume(1.0).unwrap();
        assert_eq!(vol, 0.0);
    }

    #[test]
    fn new_1d_box() {
        let b = NdarrayBox::new(array![0.0], array![5.0], 1.0).unwrap();
        assert_eq!(b.dim(), 1);
        let vol = b.volume(1.0).unwrap();
        assert!((vol - 5.0).abs() < 1e-6);
    }

    // ---- Center ----

    #[test]
    fn center_is_midpoint() {
        let b = NdarrayBox::new(array![0.0, 2.0, 4.0], array![4.0, 6.0, 8.0], 1.0).unwrap();
        let c = b.center().unwrap();
        assert!((c[0] - 2.0).abs() < 1e-6);
        assert!((c[1] - 4.0).abs() < 1e-6);
        assert!((c[2] - 6.0).abs() < 1e-6);
    }

    // ---- containment_prob_many ----

    #[test]
    fn containment_prob_many_matches_individual() {
        let parent = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let children = vec![
            NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap(),
            NdarrayBox::new(array![5.0, 5.0], array![6.0, 6.0], 1.0).unwrap(),
            NdarrayBox::new(array![20.0, 20.0], array![30.0, 30.0], 1.0).unwrap(),
        ];
        let mut out = vec![0.0f32; 3];
        parent
            .containment_prob_many(&children, 1.0, &mut out)
            .unwrap();

        for (i, child) in children.iter().enumerate() {
            let expected = parent.containment_prob(child, 1.0).unwrap();
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "Mismatch at index {}: batch={} individual={}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn containment_prob_many_buffer_too_small() {
        let parent = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let children = vec![
            NdarrayBox::new(array![0.0, 0.0], array![0.5, 0.5], 1.0).unwrap(),
            NdarrayBox::new(array![0.0, 0.0], array![0.5, 0.5], 1.0).unwrap(),
        ];
        let mut out = vec![0.0f32; 1]; // too small
        let result = parent.containment_prob_many(&children, 1.0, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn containment_prob_many_dimension_mismatch() {
        let parent = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let children =
            vec![NdarrayBox::new(array![0.0, 0.0, 0.0], array![0.5, 0.5, 0.5], 1.0).unwrap()];
        let mut out = vec![0.0f32; 1];
        let result = parent.containment_prob_many(&children, 1.0, &mut out);
        assert!(result.is_err());
    }

    // ---- High-dimensional (>10) paths ----

    #[test]
    fn volume_high_dim_uses_log_space() {
        // 12 dimensions, all side length 2.0 => volume = 2^12 = 4096
        let min_vals = Array1::from(vec![0.0; 12]);
        let max_vals = Array1::from(vec![2.0; 12]);
        let b = NdarrayBox::new(min_vals, max_vals, 1.0).unwrap();
        let vol = b.volume(1.0).unwrap();
        assert!((vol - 4096.0).abs() < 1.0, "Expected ~4096, got {}", vol);
    }

    #[test]
    fn containment_prob_high_dim_nested() {
        let dim = 15;
        let parent = NdarrayBox::new(
            Array1::from(vec![0.0; dim]),
            Array1::from(vec![10.0; dim]),
            1.0,
        )
        .unwrap();
        let child = NdarrayBox::new(
            Array1::from(vec![2.0; dim]),
            Array1::from(vec![8.0; dim]),
            1.0,
        )
        .unwrap();
        let p = parent.containment_prob(&child, 1.0).unwrap();
        assert!(
            (p - 1.0).abs() < 1e-4,
            "Fully nested in high dim should give ~1.0, got {}",
            p
        );
    }

    #[test]
    fn containment_prob_high_dim_disjoint() {
        let dim = 15;
        let a = NdarrayBox::new(
            Array1::from(vec![0.0; dim]),
            Array1::from(vec![1.0; dim]),
            1.0,
        )
        .unwrap();
        let b = NdarrayBox::new(
            Array1::from(vec![5.0; dim]),
            Array1::from(vec![6.0; dim]),
            1.0,
        )
        .unwrap();
        let p = a.containment_prob(&b, 1.0).unwrap();
        assert_eq!(p, 0.0);
    }

    #[test]
    fn overlap_prob_high_dim_identical() {
        let dim = 15;
        let a = NdarrayBox::new(
            Array1::from(vec![0.0; dim]),
            Array1::from(vec![1.0; dim]),
            1.0,
        )
        .unwrap();
        let p = a.overlap_prob(&a, 1.0).unwrap();
        assert!(
            (p - 1.0).abs() < 1e-4,
            "Identical high-dim boxes should have overlap ~1.0, got {}",
            p
        );
    }

    #[test]
    fn containment_prob_many_high_dim() {
        let dim = 15;
        let parent = NdarrayBox::new(
            Array1::from(vec![0.0; dim]),
            Array1::from(vec![10.0; dim]),
            1.0,
        )
        .unwrap();
        let children = vec![
            NdarrayBox::new(
                Array1::from(vec![1.0; dim]),
                Array1::from(vec![9.0; dim]),
                1.0,
            )
            .unwrap(),
            NdarrayBox::new(
                Array1::from(vec![20.0; dim]),
                Array1::from(vec![30.0; dim]),
                1.0,
            )
            .unwrap(),
        ];
        let mut out = vec![0.0f32; 2];
        parent
            .containment_prob_many(&children, 1.0, &mut out)
            .unwrap();
        assert!(
            out[0] > 0.99,
            "Nested child should have ~1.0, got {}",
            out[0]
        );
        assert_eq!(out[1], 0.0, "Disjoint child should have 0.0");
    }

    // ---- Cross-operation dimension mismatch ----

    #[test]
    fn intersection_dimension_mismatch() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        assert!(a.intersection(&b).is_err());
    }

    #[test]
    fn union_dimension_mismatch() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        assert!(a.union(&b).is_err());
    }

    #[test]
    fn distance_dimension_mismatch() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        assert!(a.distance(&b).is_err());
    }

    #[test]
    fn containment_prob_dimension_mismatch() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        assert!(a.containment_prob(&b, 1.0).is_err());
    }

    #[test]
    fn overlap_prob_dimension_mismatch() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        assert!(a.overlap_prob(&b, 1.0).is_err());
    }

    #[test]
    fn truncation_beyond_dim_is_error() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        assert!(a.truncate(5).is_err());
    }

    // ---- Containment with zero-volume child ----

    #[test]
    fn containment_prob_zero_volume_child_is_error() {
        let parent = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], 1.0).unwrap();
        let child = NdarrayBox::new(array![1.0, 1.0], array![1.0, 1.0], 1.0).unwrap();
        // Zero-volume "other" causes ZeroVolume error
        let result = parent.containment_prob(&child, 1.0);
        assert!(result.is_err());
    }

    // ---- Serialization edge cases ----

    #[test]
    fn serde_json_rejects_invalid_bounds() {
        let json = r#"{"min":[5.0, 0.0],"max":[1.0, 1.0],"temperature":1.0}"#;
        let result: Result<NdarrayBox, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Deserialization should reject min > max");
    }

    #[test]
    fn serde_json_rejects_missing_field() {
        let json = r#"{"min":[0.0],"max":[1.0]}"#;
        let result: Result<NdarrayBox, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Should reject missing temperature");
    }

    #[test]
    fn serde_json_rejects_dimension_mismatch() {
        let json = r#"{"min":[0.0, 0.0],"max":[1.0],"temperature":1.0}"#;
        let result: Result<NdarrayBox, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Should reject mismatched dims");
    }

    // ---- Overlap partial ----

    #[test]
    fn overlap_prob_partial_overlap_in_unit_interval() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let p = a.overlap_prob(&b, 1.0).unwrap();
        // Intersection = [1,1]-[2,2] vol=1, union = [0,0]-[3,3] vol=9
        // But overlap = vol_intersection / vol_union = 1/(4+4-1) = 1/7
        assert!(
            p > 0.0 && p < 1.0,
            "Partial overlap should be in (0,1), got {}",
            p
        );
        assert!((p - 1.0 / 7.0).abs() < 1e-5);
    }

    // ---- NaN rejection ----

    #[test]
    fn nan_min_returns_err() {
        let result = NdarrayBox::new(array![f32::NAN], array![1.0], 1.0);
        assert!(result.is_err(), "NaN min should be rejected");
    }

    #[test]
    fn nan_max_returns_err() {
        let result = NdarrayBox::new(array![0.0], array![f32::NAN], 1.0);
        assert!(result.is_err(), "NaN max should be rejected");
    }

    // ---- Temperature validation ----

    #[test]
    fn temperature_zero_returns_err() {
        let result = NdarrayBox::new(array![0.0], array![1.0], 0.0);
        assert!(result.is_err(), "temperature=0 should be rejected");
    }

    #[test]
    fn temperature_negative_returns_err() {
        let result = NdarrayBox::new(array![0.0], array![1.0], -1.0);
        assert!(result.is_err(), "temperature=-1 should be rejected");
    }

    #[test]
    fn temperature_nan_returns_err() {
        let result = NdarrayBox::new(array![0.0], array![1.0], f32::NAN);
        assert!(result.is_err(), "temperature=NaN should be rejected");
    }

    #[test]
    fn temperature_inf_returns_err() {
        let result = NdarrayBox::new(array![0.0], array![1.0], f32::INFINITY);
        assert!(result.is_err(), "temperature=inf should be rejected");
    }

    #[test]
    fn invalid_temperature_variant_and_display() {
        let err = NdarrayBox::new(array![0.0], array![1.0], -2.0).unwrap_err();
        match &err {
            BoxError::InvalidTemperature { value, reason } => {
                assert!((*value - (-2.0)).abs() < f64::EPSILON);
                assert!(reason.contains("finite and positive"));
            }
            other => panic!("expected InvalidTemperature, got {other:?}"),
        }
        let display = err.to_string();
        assert!(
            display.contains("Invalid temperature") && display.contains("-2"),
            "unexpected Display output: {display}"
        );
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use crate::Box as BoxTrait;
    use ndarray::Array1;
    use proptest::prelude::*;

    /// Strategy: generate a valid NdarrayBox with `dim` dimensions.
    /// Coordinates are bounded to [-50, 50] with min <= max enforced by sorting.
    fn arb_box(dim: usize) -> impl Strategy<Value = NdarrayBox> {
        proptest::collection::vec((-50.0f32..50.0f32, -50.0f32..50.0f32), dim).prop_map(
            move |pairs| {
                let mut mins = Vec::with_capacity(dim);
                let mut maxs = Vec::with_capacity(dim);
                for (a, b) in pairs {
                    let lo = a.min(b);
                    let hi = a.max(b);
                    mins.push(lo);
                    maxs.push(hi);
                }
                NdarrayBox::new(Array1::from(mins), Array1::from(maxs), 1.0).unwrap()
            },
        )
    }

    // ---- Property 1: Volume non-negative ----

    proptest! {
        #[test]
        fn proptest_volume_non_negative(dim in 1usize..=5) {
            let strat = arb_box(dim);
            proptest::test_runner::TestRunner::default()
                .run(&strat, |b| {
                    let v = b.volume(1.0).unwrap();
                    prop_assert!(v >= 0.0, "volume must be >= 0, got {}", v);
                    prop_assert!(v.is_finite(), "volume must be finite, got {}", v);
                    Ok(())
                })
                .unwrap();
        }
    }

    proptest! {
        #[test]
        fn proptest_volume_non_negative_flat(
            pairs in proptest::collection::vec((-50.0f32..50.0f32, -50.0f32..50.0f32), 1..=5)
        ) {
            let dim = pairs.len();
            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (a, b) in &pairs {
                mins.push(a.min(*b));
                maxs.push(a.max(*b));
            }
            let bx = NdarrayBox::new(Array1::from(mins), Array1::from(maxs), 1.0).unwrap();
            let v = bx.volume(1.0).unwrap();
            prop_assert!(v >= 0.0, "volume must be >= 0, got {}", v);
            prop_assert!(v.is_finite(), "volume must be finite");
        }
    }

    // ---- Property 2: Self-containment == 1.0 ----

    proptest! {
        #[test]
        fn proptest_self_containment(
            pairs in proptest::collection::vec((-50.0f32..50.0f32, 0.01f32..20.0f32), 1..=5)
        ) {
            let dim = pairs.len();
            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (lo, width) in &pairs {
                mins.push(*lo);
                maxs.push(*lo + *width);
            }
            let bx = NdarrayBox::new(Array1::from(mins), Array1::from(maxs), 1.0).unwrap();
            let p = bx.containment_prob(&bx, 1.0).unwrap();
            prop_assert!(
                (p - 1.0).abs() < 1e-5,
                "containment_prob(b, b) should be 1.0, got {}",
                p
            );
        }
    }

    // ---- Property 3: Self-overlap == 1.0 ----

    proptest! {
        #[test]
        fn proptest_self_overlap(
            pairs in proptest::collection::vec((-50.0f32..50.0f32, 0.01f32..20.0f32), 1..=5)
        ) {
            let dim = pairs.len();
            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (lo, width) in &pairs {
                mins.push(*lo);
                maxs.push(*lo + *width);
            }
            let bx = NdarrayBox::new(Array1::from(mins), Array1::from(maxs), 1.0).unwrap();
            let p = bx.overlap_prob(&bx, 1.0).unwrap();
            prop_assert!(
                (p - 1.0).abs() < 1e-5,
                "overlap_prob(b, b) should be 1.0, got {}",
                p
            );
        }
    }

    // ---- Property 4: Intersection idempotent ----

    proptest! {
        #[test]
        fn proptest_intersection_idempotent(
            pairs in proptest::collection::vec((-50.0f32..50.0f32, -50.0f32..50.0f32), 1..=5)
        ) {
            let dim = pairs.len();
            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (a, b) in &pairs {
                mins.push(a.min(*b));
                maxs.push(a.max(*b));
            }
            let bx = NdarrayBox::new(Array1::from(mins), Array1::from(maxs), 1.0).unwrap();
            let inter = bx.intersection(&bx).unwrap();
            for i in 0..dim {
                prop_assert!(
                    (inter.min()[i] - bx.min()[i]).abs() < 1e-6,
                    "intersect(b,b).min[{}] = {}, expected {}",
                    i, inter.min()[i], bx.min()[i]
                );
                prop_assert!(
                    (inter.max()[i] - bx.max()[i]).abs() < 1e-6,
                    "intersect(b,b).max[{}] = {}, expected {}",
                    i, inter.max()[i], bx.max()[i]
                );
            }
        }
    }

    // ---- Property 5: Union idempotent ----

    proptest! {
        #[test]
        fn proptest_union_idempotent(
            pairs in proptest::collection::vec((-50.0f32..50.0f32, -50.0f32..50.0f32), 1..=5)
        ) {
            let dim = pairs.len();
            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (a, b) in &pairs {
                mins.push(a.min(*b));
                maxs.push(a.max(*b));
            }
            let bx = NdarrayBox::new(Array1::from(mins), Array1::from(maxs), 1.0).unwrap();
            let u = bx.union(&bx).unwrap();
            for i in 0..dim {
                prop_assert!(
                    (u.min()[i] - bx.min()[i]).abs() < 1e-6,
                    "union(b,b).min[{}] = {}, expected {}",
                    i, u.min()[i], bx.min()[i]
                );
                prop_assert!(
                    (u.max()[i] - bx.max()[i]).abs() < 1e-6,
                    "union(b,b).max[{}] = {}, expected {}",
                    i, u.max()[i], bx.max()[i]
                );
            }
        }
    }

    // ---- Property 6: Containment implies overlap ----

    proptest! {
        #[test]
        fn proptest_containment_implies_overlap(
            dim in 1usize..=3,
            // Generate outer box, then inner box strictly inside it
            outer_lo in -20.0f32..0.0f32,
            outer_hi in 5.0f32..20.0f32,
            shrink_lo in 0.0f32..2.0f32,
            shrink_hi in 0.0f32..2.0f32,
        ) {
            // Ensure outer box is valid
            prop_assume!(outer_lo < outer_hi);
            let inner_lo = outer_lo + shrink_lo;
            let inner_hi = outer_hi - shrink_hi;
            prop_assume!(inner_lo < inner_hi);

            let outer_mins = Array1::from(vec![outer_lo; dim]);
            let outer_maxs = Array1::from(vec![outer_hi; dim]);
            let inner_mins = Array1::from(vec![inner_lo; dim]);
            let inner_maxs = Array1::from(vec![inner_hi; dim]);

            let a = NdarrayBox::new(outer_mins, outer_maxs, 1.0).unwrap();
            let b = NdarrayBox::new(inner_mins, inner_maxs, 1.0).unwrap();

            let cp = a.containment_prob(&b, 1.0).unwrap();
            if cp > 0.9 {
                let op = a.overlap_prob(&b, 1.0).unwrap();
                prop_assert!(
                    op > 0.0,
                    "containment_prob={} > 0.9 but overlap_prob={} is not > 0",
                    cp, op
                );
            }
        }
    }

    // ---- Property 7: Truncation preserves containment ----
    // truncate(b, k) should be "contained" in b when projected to k dims,
    // i.e. truncate(b, k).min >= b.min[..k] and truncate(b, k).max <= b.max[..k]

    proptest! {
        #[test]
        fn proptest_truncation_preserves_bounds(
            pairs in proptest::collection::vec((-50.0f32..50.0f32, -50.0f32..50.0f32), 2..=5)
        ) {
            let dim = pairs.len();
            let mut mins = Vec::with_capacity(dim);
            let mut maxs = Vec::with_capacity(dim);
            for (a, b) in &pairs {
                mins.push(a.min(*b));
                maxs.push(a.max(*b));
            }
            let bx = NdarrayBox::new(Array1::from(mins.clone()), Array1::from(maxs.clone()), 1.0).unwrap();

            // Truncate to every valid sub-dimension
            for k in 1..dim {
                let trunc = bx.truncate(k).unwrap();
                prop_assert_eq!(trunc.dim(), k);
                for i in 0..k {
                    prop_assert!(
                        (trunc.min()[i] - mins[i]).abs() < 1e-6,
                        "truncate({}).min[{}] = {}, expected {}",
                        k, i, trunc.min()[i], mins[i]
                    );
                    prop_assert!(
                        (trunc.max()[i] - maxs[i]).abs() < 1e-6,
                        "truncate({}).max[{}] = {}, expected {}",
                        k, i, trunc.max()[i], maxs[i]
                    );
                }
            }
        }
    }

    // ---- Property 8: Intersection volume <= min(input volumes) ----

    /// Strategy for a non-degenerate box (positive width per dimension).
    fn arb_nondegenerate_box(dim: usize) -> impl Strategy<Value = NdarrayBox> {
        proptest::collection::vec((-20.0f32..20.0f32, 0.1f32..10.0f32), dim).prop_map(
            move |pairs| {
                let mut mins = Vec::with_capacity(dim);
                let mut maxs = Vec::with_capacity(dim);
                for (lo, width) in pairs {
                    mins.push(lo);
                    maxs.push(lo + width);
                }
                NdarrayBox::new(Array1::from(mins), Array1::from(maxs), 1.0).unwrap()
            },
        )
    }

    proptest! {
        #[test]
        fn proptest_intersection_volume_le_min_input(
            (a, b) in (arb_nondegenerate_box(3), arb_nondegenerate_box(3))
        ) {
            let inter = a.intersection(&b).unwrap();
            let vol_inter = inter.volume(1.0).unwrap();
            let vol_a = a.volume(1.0).unwrap();
            let vol_b = b.volume(1.0).unwrap();
            let min_vol = vol_a.min(vol_b);
            prop_assert!(
                vol_inter <= min_vol + 1e-4,
                "intersection volume ({vol_inter}) must be <= min(vol_a={vol_a}, vol_b={vol_b})"
            );
        }
    }

    // ---- Property 9: Containment transitivity ----
    // If A contains B and B contains C, then A contains C.

    proptest! {
        #[test]
        fn proptest_containment_transitivity(
            base in -10.0f32..10.0,
            w1 in 1.0f32..10.0,
            shrink1 in 0.1f32..0.4,
            shrink2 in 0.1f32..0.4,
        ) {
            let dim = 3;
            // A = [base, base+w1]^d
            let a_lo = base;
            let a_hi = base + w1;
            // B = A shrunk by shrink1 on each side
            let b_lo = a_lo + shrink1;
            let b_hi = a_hi - shrink1;
            prop_assume!(b_lo < b_hi);
            // C = B shrunk by shrink2 on each side
            let c_lo = b_lo + shrink2;
            let c_hi = b_hi - shrink2;
            prop_assume!(c_lo < c_hi);

            let a = NdarrayBox::new(
                Array1::from(vec![a_lo; dim]),
                Array1::from(vec![a_hi; dim]),
                1.0,
            ).unwrap();
            let b = NdarrayBox::new(
                Array1::from(vec![b_lo; dim]),
                Array1::from(vec![b_hi; dim]),
                1.0,
            ).unwrap();
            let c = NdarrayBox::new(
                Array1::from(vec![c_lo; dim]),
                Array1::from(vec![c_hi; dim]),
                1.0,
            ).unwrap();

            let p_ab = a.containment_prob(&b, 1.0).unwrap();
            let p_bc = b.containment_prob(&c, 1.0).unwrap();
            let p_ac = a.containment_prob(&c, 1.0).unwrap();

            // If A contains B and B contains C, then A must contain C.
            if p_ab > 0.99 && p_bc > 0.99 {
                prop_assert!(
                    p_ac > 0.99,
                    "transitivity: A contains B ({p_ab}) and B contains C ({p_bc}), \
                     but A contains C only {p_ac}"
                );
            }
        }
    }
}

#[cfg(test)]
mod degenerate_tests {
    use super::*;
    use crate::Box as BoxTrait;
    use ndarray::Array1;
    use proptest::prelude::*;

    fn assert_finite_prob(v: f32, label: &str) {
        assert!(v.is_finite(), "{label} not finite: {v}");
        assert!((0.0..=1.0).contains(&v), "{label} out of [0,1]: {v}");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Zero-volume boxes (min == max), partial zero-volume, and single-dim.
        #[test]
        fn degenerate_zero_and_partial_volume(
            val in proptest::collection::vec(-50.0f32..50.0f32, 2..=8),
            k in 1usize..=4,
        ) {
            let d = val.len();
            // Fully zero-volume: min == max everywhere.
            let zb = NdarrayBox::new(Array1::from(val.clone()), Array1::from(val.clone()), 1.0).unwrap();
            let vol = zb.volume(1.0).unwrap();
            assert!(vol == 0.0, "zero-volume box has vol {vol}");
            // containment_prob may return Err(ZeroVolume) -- that's valid.
            let cp = zb.containment_prob(&zb, 1.0);
            if let Ok(v) = cp { assert!(v.is_finite(), "zero-vol containment not finite: {v}"); }

            // Partial zero-volume: first k dims collapsed, rest have width 1.
            let k = k.min(d);
            let mut maxs = val.clone();
            for m in maxs.iter_mut().skip(k) {
                *m += 1.0;
            }
            let pb = NdarrayBox::new(Array1::from(val), Array1::from(maxs), 1.0).unwrap();
            let pv = pb.volume(1.0).unwrap();
            assert!(pv.is_finite() && pv >= 0.0, "partial zero-vol: {pv}");
        }

        /// Near-zero-volume: min very close to max.
        #[test]
        fn degenerate_near_zero_volume(
            base in proptest::collection::vec(-50.0f32..50.0f32, 2..=8),
        ) {
            let eps = 1e-7_f32;
            let maxs: Vec<f32> = base.iter().map(|v| v + eps).collect();
            let b = NdarrayBox::new(Array1::from(base), Array1::from(maxs), 1.0).unwrap();
            let vol = b.volume(1.0).unwrap();
            assert!(vol.is_finite() && vol >= 0.0, "near-zero vol: {vol}");
            // May return Err(ZeroVolume) if width underflows to zero.
            if let Ok(cp) = b.containment_prob(&b, 1.0) {
                assert!(cp.is_finite(), "near-zero containment: {cp}");
            }
        }

        /// Large coordinates: near f32::MAX / 1000, no overflow.
        #[test]
        fn degenerate_large_coords(dim in 1usize..=5) {
            let big = f32::MAX / 1000.0;
            let b = NdarrayBox::new(
                Array1::from(vec![big - 1.0; dim]),
                Array1::from(vec![big; dim]),
                1.0,
            ).unwrap();
            let vol = b.volume(1.0).unwrap();
            // Volume may overflow to inf in high dims; just must not NaN or panic.
            assert!(!vol.is_nan(), "large-coord volume is NaN");
        }

        /// High dimensionality (d up to 200).
        #[test]
        fn degenerate_high_dim(dim in prop::sample::select(vec![50, 100, 200])) {
            let b = NdarrayBox::new(
                Array1::from(vec![0.0f32; dim]),
                Array1::from(vec![1.0f32; dim]),
                1.0,
            ).unwrap();
            let vol = b.volume(1.0).unwrap();
            assert!(!vol.is_nan(), "high-dim volume is NaN (d={dim})");
            let cp = b.containment_prob(&b, 1.0).unwrap();
            assert!(cp.is_finite(), "high-dim containment not finite (d={dim})");
        }

        /// Extreme temperature values.
        #[test]
        fn degenerate_extreme_temperature(
            temp in prop::sample::select(vec![1e-3_f32, 0.01, 0.1, 5.0, 10.0]),
        ) {
            let b = NdarrayBox::new(
                Array1::from(vec![0.0, -1.0, 2.0]),
                Array1::from(vec![1.0, 0.0, 3.0]),
                temp,
            ).unwrap();
            let vol = b.volume(temp).unwrap();
            assert!(vol.is_finite() && vol >= 0.0, "temp={temp} vol={vol}");
            let cp = b.containment_prob(&b, temp).unwrap();
            assert_finite_prob(cp, &format!("temp={temp} self-containment"));
        }

        /// Single dimension (d=1).
        #[test]
        fn degenerate_single_dim(lo in -100.0f32..100.0f32, width in 0.0f32..50.0f32) {
            let b = NdarrayBox::new(
                Array1::from(vec![lo]),
                Array1::from(vec![lo + width]),
                1.0,
            ).unwrap();
            let vol = b.volume(1.0).unwrap();
            assert!(vol.is_finite() && vol >= 0.0, "1d vol: {vol}");
            assert!((vol - width).abs() < 1e-4, "1d vol should equal width: {vol} vs {width}");
        }
    }
}
