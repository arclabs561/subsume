//! Ndarray implementation of the Cone trait.

use crate::cone::{Cone, ConeError};
use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A cone embedding implemented using `ndarray::Array1<f32>`.
///
/// Each cone is defined by:
/// - **apex**: a point in d-dimensional space (the cone's origin)
/// - **axis**: a unit direction vector (the cone's central direction)
/// - **aperture**: half-angle in radians, in (0, pi)
///
/// Larger aperture means a more general concept. A cone with aperture near pi
/// covers almost the entire angular space; a cone with aperture near 0 is very
/// specific.
#[derive(Debug, Clone)]
pub struct NdarrayCone {
    /// Apex point [d].
    apex: Array1<f32>,
    /// Unit axis direction [d].
    axis: Array1<f32>,
    /// Half-angle aperture in radians, in (0, pi).
    aperture: f32,
}

impl Serialize for NdarrayCone {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NdarrayCone", 3)?;
        state.serialize_field("apex", &self.apex.to_vec())?;
        state.serialize_field("axis", &self.axis.to_vec())?;
        state.serialize_field("aperture", &self.aperture)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for NdarrayCone {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Apex,
            Axis,
            Aperture,
        }

        struct NdarrayConeVisitor;

        impl<'de> Visitor<'de> for NdarrayConeVisitor {
            type Value = NdarrayCone;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct NdarrayCone")
            }

            fn visit_map<V>(self, mut map: V) -> Result<NdarrayCone, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut apex = None;
                let mut axis = None;
                let mut aperture = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Apex => {
                            if apex.is_some() {
                                return Err(de::Error::duplicate_field("apex"));
                            }
                            apex = Some(map.next_value::<Vec<f32>>()?);
                        }
                        Field::Axis => {
                            if axis.is_some() {
                                return Err(de::Error::duplicate_field("axis"));
                            }
                            axis = Some(map.next_value::<Vec<f32>>()?);
                        }
                        Field::Aperture => {
                            if aperture.is_some() {
                                return Err(de::Error::duplicate_field("aperture"));
                            }
                            aperture = Some(map.next_value()?);
                        }
                    }
                }
                let apex = apex.ok_or_else(|| de::Error::missing_field("apex"))?;
                let axis = axis.ok_or_else(|| de::Error::missing_field("axis"))?;
                let aperture =
                    aperture.ok_or_else(|| de::Error::missing_field("aperture"))?;
                NdarrayCone::new(Array1::from(apex), Array1::from(axis), aperture)
                    .map_err(|e| de::Error::custom(format!("Invalid cone: {}", e)))
            }
        }

        const FIELDS: &[&str] = &["apex", "axis", "aperture"];
        deserializer.deserialize_struct("NdarrayCone", FIELDS, NdarrayConeVisitor)
    }
}

impl NdarrayCone {
    /// Create a new `NdarrayCone`.
    ///
    /// The axis vector is normalized to unit length. Aperture must be in (0, pi).
    ///
    /// # Errors
    ///
    /// - [`ConeError::DimensionMismatch`] if apex and axis have different lengths.
    /// - [`ConeError::ZeroAxis`] if the axis vector has zero norm.
    /// - [`ConeError::InvalidAperture`] if aperture is not in (0, pi).
    pub fn new(apex: Array1<f32>, axis: Array1<f32>, aperture: f32) -> Result<Self, ConeError> {
        if apex.len() != axis.len() {
            return Err(ConeError::DimensionMismatch {
                expected: apex.len(),
                actual: axis.len(),
            });
        }

        if aperture <= 0.0 || aperture >= std::f32::consts::PI {
            return Err(ConeError::InvalidAperture {
                value: aperture as f64,
            });
        }

        // Normalize axis to unit length.
        let norm = axis.dot(&axis).sqrt();
        if norm < 1e-12 {
            return Err(ConeError::ZeroAxis);
        }
        let axis = &axis / norm;

        Ok(Self {
            apex,
            axis,
            aperture,
        })
    }

    /// Create a `NdarrayCone` from a pre-normalized axis (no normalization check).
    ///
    /// This is an internal fast path. The caller must guarantee that `axis` is unit-length.
    fn from_raw(apex: Array1<f32>, axis: Array1<f32>, aperture: f32) -> Self {
        Self {
            apex,
            axis,
            aperture,
        }
    }
}

/// Numerically stable sigmoid: 1 / (1 + exp(-x)), clamped to avoid overflow.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Compute the angular distance between two unit vectors.
///
/// Returns arccos(dot(a, b)), clamped to [0, pi].
fn angular_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos()
}

impl Cone for NdarrayCone {
    type Scalar = f32;
    type Vector = Array1<f32>;

    fn apex(&self) -> &Self::Vector {
        &self.apex
    }

    fn axis(&self) -> &Self::Vector {
        &self.axis
    }

    fn aperture(&self) -> Self::Scalar {
        self.aperture
    }

    fn dim(&self) -> usize {
        self.apex.len()
    }

    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, ConeError> {
        if self.dim() != other.dim() {
            return Err(ConeError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        // P(other inside self) = sigmoid((self.aperture - angular_dist - other.aperture) / temp)
        let ang_dist = angular_distance(&self.axis, &other.axis);
        let margin = self.aperture - ang_dist - other.aperture;
        Ok(sigmoid(margin / temperature))
    }

    fn intersection(&self, other: &Self) -> Result<Self, ConeError> {
        if self.dim() != other.dim() {
            return Err(ConeError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let ang_dist = angular_distance(&self.axis, &other.axis);

        // Check if cones overlap angularly.
        // Overlap exists when the sum of apertures exceeds the angular distance.
        if ang_dist >= self.aperture + other.aperture {
            // No overlap: return a degenerate cone (minimal aperture) along the midpoint axis.
            let mid_axis = (&self.axis + &other.axis) / 2.0;
            let norm = mid_axis.dot(&mid_axis).sqrt();
            let axis = if norm < 1e-12 {
                self.axis.clone()
            } else {
                &mid_axis / norm
            };
            // Use a very small aperture to represent "empty" intersection.
            let eps = 1e-6_f32;
            return Ok(NdarrayCone::from_raw(self.apex.clone(), axis, eps));
        }

        // Compute the intersection cone.
        // The intersection axis bisects the overlap region.
        // Weighted average of axes, biased toward the narrower cone.
        let w_self = other.aperture;
        let w_other = self.aperture;
        let weighted_axis = &self.axis * w_self + &other.axis * w_other;
        let norm = weighted_axis.dot(&weighted_axis).sqrt();
        let axis = if norm < 1e-12 {
            self.axis.clone()
        } else {
            &weighted_axis / norm
        };

        // The intersection aperture is the angular extent of the overlap region.
        // Each cone spans [center - aperture, center + aperture] on the angular axis.
        // Overlap half-width = (aperture_A + aperture_B - angular_dist) / 2
        let intersection_aperture =
            ((self.aperture + other.aperture - ang_dist) / 2.0).clamp(1e-6, std::f32::consts::PI - 1e-6);

        // Use the midpoint of the two apexes as the intersection apex.
        let apex = (&self.apex + &other.apex) / 2.0;

        Ok(NdarrayCone::from_raw(apex, axis, intersection_aperture))
    }

    fn complement(&self) -> Result<Self, ConeError> {
        // Complement: negate the axis, aperture becomes pi - aperture.
        let new_aperture = std::f32::consts::PI - self.aperture;

        if new_aperture <= 0.0 || new_aperture >= std::f32::consts::PI {
            return Err(ConeError::InvalidAperture {
                value: new_aperture as f64,
            });
        }

        let neg_axis = &self.axis * -1.0;
        Ok(NdarrayCone::from_raw(self.apex.clone(), neg_axis, new_aperture))
    }

    fn overlap_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, ConeError> {
        if self.dim() != other.dim() {
            return Err(ConeError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        // Overlap exists when the sum of apertures exceeds the angular distance.
        // P(overlap) = sigmoid((aperture_A + aperture_B - angular_dist) / temperature)
        let ang_dist = angular_distance(&self.axis, &other.axis);
        let margin = self.aperture + other.aperture - ang_dist;
        Ok(sigmoid(margin / temperature))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f32::consts::PI;

    // ---- Construction ----

    #[test]
    fn new_normalizes_axis() {
        let cone = NdarrayCone::new(array![0.0, 0.0], array![3.0, 4.0], 0.5).unwrap();
        let norm: f32 = cone.axis().dot(cone.axis()).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Axis should be unit-length, got norm {}",
            norm
        );
    }

    #[test]
    fn new_rejects_zero_axis() {
        let result = NdarrayCone::new(array![0.0, 0.0], array![0.0, 0.0], 0.5);
        assert!(matches!(result, Err(ConeError::ZeroAxis)));
    }

    #[test]
    fn new_rejects_invalid_aperture_zero() {
        let result = NdarrayCone::new(array![0.0], array![1.0], 0.0);
        assert!(matches!(result, Err(ConeError::InvalidAperture { .. })));
    }

    #[test]
    fn new_rejects_invalid_aperture_pi() {
        let result = NdarrayCone::new(array![0.0], array![1.0], PI);
        assert!(matches!(result, Err(ConeError::InvalidAperture { .. })));
    }

    #[test]
    fn new_rejects_dimension_mismatch() {
        let result = NdarrayCone::new(array![0.0, 0.0], array![1.0], 0.5);
        assert!(matches!(result, Err(ConeError::DimensionMismatch { .. })));
    }

    // ---- Containment ----

    #[test]
    fn containment_same_axis_wider_contains_narrower() {
        // Wide cone (aperture 1.5) should contain narrow cone (aperture 0.3)
        // when they share the same axis.
        let wide = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 1.5).unwrap();
        let narrow = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.3).unwrap();
        let p = wide.containment_prob(&narrow, 0.1).unwrap();
        assert!(
            p > 0.99,
            "Wide cone should contain narrow cone with same axis, got {}",
            p
        );
    }

    #[test]
    fn containment_narrower_does_not_contain_wider() {
        let wide = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 1.5).unwrap();
        let narrow = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.3).unwrap();
        let p = narrow.containment_prob(&wide, 0.1).unwrap();
        assert!(
            p < 0.01,
            "Narrow cone should not contain wide cone, got {}",
            p
        );
    }

    #[test]
    fn containment_distant_axes_low_prob() {
        // Cones pointing in opposite directions should have very low containment.
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.5).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0], array![-1.0, 0.0], 0.5).unwrap();
        let p = a.containment_prob(&b, 0.1).unwrap();
        assert!(
            p < 0.01,
            "Opposite-direction cones should have near-zero containment, got {}",
            p
        );
    }

    #[test]
    fn containment_self_is_boundary() {
        // Self-containment: margin = aperture - 0 - aperture = 0, so sigmoid(0) = 0.5.
        // This is the expected boundary behavior with the sigmoid relaxation: a cone
        // exactly contains itself at probability 0.5 (the decision boundary).
        let a = NdarrayCone::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 0.0], 1.0).unwrap();
        let p = a.containment_prob(&a, 0.1).unwrap();
        assert!(
            (p - 0.5).abs() < 0.01,
            "Self-containment should be ~0.5 (sigmoid boundary), got {}",
            p
        );
    }

    #[test]
    fn containment_strictly_wider_is_high() {
        // A cone with larger aperture and same axis should have high containment prob
        // for a narrower cone, even at the sigmoid boundary.
        let wide = NdarrayCone::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 0.0], 1.5).unwrap();
        let narrow = NdarrayCone::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 0.0], 0.5).unwrap();
        let p = wide.containment_prob(&narrow, 0.1).unwrap();
        assert!(
            p > 0.99,
            "Strictly wider cone should have high containment, got {}",
            p
        );
    }

    #[test]
    fn containment_dimension_mismatch() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.5).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0], 0.5).unwrap();
        let result = a.containment_prob(&b, 1.0);
        assert!(matches!(result, Err(ConeError::DimensionMismatch { .. })));
    }

    // ---- Complement / Negation ----

    #[test]
    fn complement_aperture_is_pi_minus_original() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.8).unwrap();
        let comp = a.complement().unwrap();
        let expected_aperture = PI - 0.8;
        assert!(
            (comp.aperture() - expected_aperture).abs() < 1e-6,
            "Complement aperture should be pi - 0.8 = {}, got {}",
            expected_aperture,
            comp.aperture()
        );
    }

    #[test]
    fn complement_axis_is_negated() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.8).unwrap();
        let comp = a.complement().unwrap();
        for (orig, neg) in a.axis().iter().zip(comp.axis().iter()) {
            assert!(
                (orig + neg).abs() < 1e-6,
                "Complement axis should be negated"
            );
        }
    }

    #[test]
    fn double_complement_is_identity() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.8).unwrap();
        let double_comp = a.complement().unwrap().complement().unwrap();
        assert!(
            (a.aperture() - double_comp.aperture()).abs() < 1e-5,
            "Double complement aperture should equal original"
        );
        for (orig, dc) in a.axis().iter().zip(double_comp.axis().iter()) {
            assert!(
                (orig - dc).abs() < 1e-5,
                "Double complement axis should equal original"
            );
        }
    }

    // ---- Overlap ----

    #[test]
    fn overlap_identical_cones_is_high() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 1.0).unwrap();
        let p = a.overlap_prob(&a, 0.1).unwrap();
        assert!(
            p > 0.99,
            "Identical cones should have high overlap, got {}",
            p
        );
    }

    #[test]
    fn overlap_opposite_narrow_cones_is_low() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.3).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0], array![-1.0, 0.0], 0.3).unwrap();
        let p = a.overlap_prob(&b, 0.1).unwrap();
        assert!(
            p < 0.01,
            "Opposite narrow cones should have low overlap, got {}",
            p
        );
    }

    #[test]
    fn overlap_nearby_axes_moderate() {
        // Two cones at ~45 degrees apart, each with aperture 0.5 rad (~28 deg).
        // Angular distance ~0.785 rad, sum of apertures = 1.0 > 0.785, so they overlap.
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.5).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0], array![1.0, 1.0], 0.5).unwrap();
        let p = a.overlap_prob(&b, 0.5).unwrap();
        assert!(
            p > 0.1 && p < 1.0,
            "Nearby cones should have moderate overlap, got {}",
            p
        );
    }

    // ---- Intersection ----

    #[test]
    fn intersection_of_identical_cones_has_same_aperture() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 1.0).unwrap();
        let inter = a.intersection(&a).unwrap();
        // Intersection aperture = (1.0 + 1.0 - 0.0) / 2 = 1.0
        assert!(
            (inter.aperture() - 1.0).abs() < 1e-5,
            "Intersection of identical cones should have same aperture, got {}",
            inter.aperture()
        );
    }

    #[test]
    fn intersection_of_disjoint_cones_is_degenerate() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.3).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0], array![-1.0, 0.0], 0.3).unwrap();
        let inter = a.intersection(&b).unwrap();
        assert!(
            inter.aperture() < 1e-4,
            "Disjoint cones should produce degenerate intersection, got aperture {}",
            inter.aperture()
        );
    }

    #[test]
    fn intersection_dimension_mismatch() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 0.5).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0], 0.5).unwrap();
        let result = a.intersection(&b);
        assert!(matches!(result, Err(ConeError::DimensionMismatch { .. })));
    }

    // ---- Numerical stability ----

    #[test]
    fn containment_prob_finite_at_extreme_temperatures() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 1.0).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0], array![1.0, 1.0], 0.5).unwrap();

        let p_cold = a.containment_prob(&b, 0.001).unwrap();
        assert!(p_cold.is_finite(), "Containment prob must be finite at low temp");
        assert!((0.0..=1.0).contains(&p_cold), "Containment prob must be in [0,1]");

        let p_hot = a.containment_prob(&b, 100.0).unwrap();
        assert!(p_hot.is_finite(), "Containment prob must be finite at high temp");
        assert!((0.0..=1.0).contains(&p_hot), "Containment prob must be in [0,1]");
    }

    #[test]
    fn overlap_prob_finite_at_extreme_temperatures() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.0], 1.0).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0], array![1.0, 1.0], 0.5).unwrap();

        let p_cold = a.overlap_prob(&b, 0.001).unwrap();
        assert!(p_cold.is_finite(), "Overlap prob must be finite at low temp");

        let p_hot = a.overlap_prob(&b, 100.0).unwrap();
        assert!(p_hot.is_finite(), "Overlap prob must be finite at high temp");
    }

    // ---- Higher dimensions ----

    #[test]
    fn works_in_high_dimensions() {
        let d = 128;
        let mut apex = vec![0.0f32; d];
        let mut axis_a = vec![0.0f32; d];
        let mut axis_b = vec![0.0f32; d];
        apex[0] = 1.0;
        axis_a[0] = 1.0;
        axis_b[1] = 1.0;

        let a = NdarrayCone::new(
            Array1::from(apex.clone()),
            Array1::from(axis_a),
            1.2,
        )
        .unwrap();
        let b = NdarrayCone::new(
            Array1::from(apex),
            Array1::from(axis_b),
            0.8,
        )
        .unwrap();

        let p_cont = a.containment_prob(&b, 0.5).unwrap();
        assert!(p_cont.is_finite());
        assert!((0.0..=1.0).contains(&p_cont));

        let p_over = a.overlap_prob(&b, 0.5).unwrap();
        assert!(p_over.is_finite());
        assert!((0.0..=1.0).contains(&p_over));

        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter.dim(), d);

        let comp = a.complement().unwrap();
        assert_eq!(comp.dim(), d);
    }
}
