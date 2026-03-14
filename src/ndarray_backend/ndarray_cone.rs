//! Ndarray cone embedding.
//!
//! Implements the ConE model (Zhang & Wang, NeurIPS 2021) using Cartesian products
//! of 2D angular sectors. Each dimension has an axis angle in \[-pi, pi\] and an
//! aperture (half-width) in \[0, pi\].

use crate::cone::ConeError;
use ndarray::Array1;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::f32::consts::PI;

/// Normalize an angle to \[-pi, pi\].
#[inline]
fn normalize_angle(mut a: f32) -> f32 {
    // Use fmod-style wrapping.
    a %= 2.0 * PI;
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Clamp a value to \[0, pi\].
#[inline]
fn clamp_aperture(a: f32) -> f32 {
    a.clamp(0.0, PI)
}

/// A cone embedding as a Cartesian product of `d` independent 2D angular sectors.
///
/// Each dimension has an axis angle in \[-pi, pi\] and an aperture (half-width)
/// in \[0, pi\]. Subsumption is measured by per-dimension angular containment,
/// and the scores are summed across dimensions.
///
/// Cones support negation: the complement of a cone is a cone (per-dimension axis
/// shift by pi, aperture becomes pi - aperture). This closure under complementation
/// enables modeling FOL operations including conjunction, disjunction, and negation.
///
/// Each cone is a Cartesian product of `d` independent 2D angular sectors:
/// - **axes**: per-dimension center angles, each in \[-pi, pi\]
/// - **apertures**: per-dimension half-widths, each in \[0, pi\]
///
/// Wider apertures mean more general concepts. An aperture of pi covers the
/// entire circle in that dimension; an aperture of 0 is a single point.
///
/// Reference: Zhang & Wang (2021), "ConE: Cone Embeddings for Multi-Hop Reasoning
/// over Knowledge Graphs" (NeurIPS 2021).
#[derive(Debug, Clone)]
pub struct NdarrayCone {
    /// Per-dimension axis angles \[d\], each in \[-pi, pi\].
    axes: Array1<f32>,
    /// Per-dimension apertures (half-widths) \[d\], each in \[0, pi\].
    apertures: Array1<f32>,
}

impl Serialize for NdarrayCone {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NdarrayCone", 2)?;
        state.serialize_field("axes", &self.axes.to_vec())?;
        state.serialize_field("apertures", &self.apertures.to_vec())?;
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
            Axes,
            Apertures,
        }

        struct NdarrayConeVisitor;

        impl<'de> Visitor<'de> for NdarrayConeVisitor {
            type Value = NdarrayCone;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct NdarrayCone with fields axes, apertures")
            }

            fn visit_map<V>(self, mut map: V) -> Result<NdarrayCone, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut axes = None;
                let mut apertures = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Axes => {
                            if axes.is_some() {
                                return Err(de::Error::duplicate_field("axes"));
                            }
                            axes = Some(map.next_value::<Vec<f32>>()?);
                        }
                        Field::Apertures => {
                            if apertures.is_some() {
                                return Err(de::Error::duplicate_field("apertures"));
                            }
                            apertures = Some(map.next_value::<Vec<f32>>()?);
                        }
                    }
                }
                let axes = axes.ok_or_else(|| de::Error::missing_field("axes"))?;
                let apertures = apertures.ok_or_else(|| de::Error::missing_field("apertures"))?;
                NdarrayCone::new(Array1::from(axes), Array1::from(apertures))
                    .map_err(|e| de::Error::custom(format!("Invalid cone: {e}")))
            }
        }

        const FIELDS: &[&str] = &["axes", "apertures"];
        deserializer.deserialize_struct("NdarrayCone", FIELDS, NdarrayConeVisitor)
    }
}

impl NdarrayCone {
    /// Create a new `NdarrayCone`.
    ///
    /// Axes are normalized to \[-pi, pi\]; apertures are clamped to \[0, pi\].
    ///
    /// # Errors
    ///
    /// - [`ConeError::DimensionMismatch`] if axes and apertures have different lengths.
    /// - [`ConeError::InvalidBounds`] if any value is NaN.
    pub fn new(axes: Array1<f32>, apertures: Array1<f32>) -> Result<Self, ConeError> {
        if axes.len() != apertures.len() {
            return Err(ConeError::DimensionMismatch {
                expected: axes.len(),
                actual: apertures.len(),
            });
        }

        if axes.iter().any(|v| !v.is_finite()) || apertures.iter().any(|v| !v.is_finite()) {
            return Err(ConeError::InvalidBounds {
                reason: "non-finite values (NaN/Inf) are not allowed in axes or apertures".into(),
            });
        }

        let axes = axes.mapv(normalize_angle);
        let apertures = apertures.mapv(clamp_aperture);

        Ok(Self { axes, apertures })
    }

    /// Create a `NdarrayCone` from pre-validated arrays (no clamping).
    ///
    /// Internal fast path. The caller must guarantee axes are in \[-pi, pi\] and
    /// apertures are in \[0, pi\].
    fn from_raw(axes: Array1<f32>, apertures: Array1<f32>) -> Self {
        Self { axes, apertures }
    }

    /// Create a uniform cone covering the entire angular space in each dimension.
    ///
    /// This is the "top" element: it contains every other cone.
    #[must_use]
    pub fn full(dim: usize) -> Self {
        Self {
            axes: Array1::zeros(dim),
            apertures: Array1::from_elem(dim, PI),
        }
    }

    /// Create a point cone (zero aperture in every dimension).
    ///
    /// This is the "bottom" element: contained by every non-degenerate cone
    /// whose axis matches.
    #[must_use]
    pub fn point(axes: Array1<f32>) -> Self {
        let d = axes.len();
        let axes = axes.mapv(normalize_angle);
        Self {
            axes,
            apertures: Array1::zeros(d),
        }
    }

    /// Get the per-dimension axis angles.
    /// Each element is in \[-pi, pi\].
    pub fn axes(&self) -> &Array1<f32> {
        &self.axes
    }

    /// Get the per-dimension apertures (half-widths).
    /// Each element is in \[0, pi\].
    pub fn apertures(&self) -> &Array1<f32> {
        &self.apertures
    }

    /// Get the number of dimensions.
    pub fn dim(&self) -> usize {
        self.axes.len()
    }

    /// Compute the ConE distance score between an entity cone and this query cone.
    ///
    /// Uses the per-dimension scoring from ConE (Zhang & Wang, 2021):
    ///
    /// ```text
    /// distance_to_axis[i] = |sin((entity_axis[i] - query_axis[i]) / 2)|
    /// distance_base[i]    = |sin(query_aperture[i] / 2)|
    /// ```
    ///
    /// Points inside the sector contribute `cen * distance_in`; points outside
    /// contribute `distance_out`. The total is summed across dimensions.
    ///
    /// Lower distance = better containment. The `cen` parameter (typically 0.02)
    /// weights the inside distance relative to outside distance.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if cones have different dimensions.
    pub fn cone_distance(&self, entity: &Self, cen: f32) -> Result<f32, ConeError> {
        if self.dim() != entity.dim() {
            return Err(ConeError::DimensionMismatch {
                expected: self.dim(),
                actual: entity.dim(),
            });
        }

        // Per-dimension ConE scoring (Zhang & Wang, 2021).
        // self = query, entity = entity being scored.
        let mut dist_out_sum = 0.0_f32;
        let mut dist_in_sum = 0.0_f32;

        for i in 0..self.dim() {
            let e = entity.axes[i];
            let q_axis = self.axes[i];
            let q_aper = self.apertures[i];

            let distance_to_axis = ((e - q_axis) / 2.0).sin().abs();
            let distance_base = (q_aper / 2.0).sin().abs();

            if distance_to_axis < distance_base {
                // Entity is inside the cone in this dimension.
                let dist_in = distance_to_axis.min(distance_base);
                dist_in_sum += dist_in;
            } else {
                // Entity is outside the cone in this dimension.
                // Distance to nearest boundary.
                let delta1 = e - (q_axis - q_aper);
                let delta2 = e - (q_axis + q_aper);
                let d1 = (delta1 / 2.0).sin().abs();
                let d2 = (delta2 / 2.0).sin().abs();
                dist_out_sum += d1.min(d2);
            }
        }

        Ok(dist_out_sum + cen * dist_in_sum)
    }

    /// Compute the complement (negation) of this cone.
    ///
    /// Per-dimension:
    /// - axis\[i\] shifts by pi (positive axes subtract pi, negative axes add pi),
    ///   keeping the result in \[-pi, pi\].
    /// - aperture\[i\] becomes pi - aperture\[i\].
    ///
    /// This closure under complementation is the key advantage over boxes.
    #[must_use]
    pub fn complement(&self) -> Self {
        // Per-dimension negation (ConE paper):
        // - axis[i] shifts by pi (positive -> subtract pi, negative -> add pi)
        // - aperture[i] = pi - aperture[i]
        let new_axes = self.axes.mapv(|a| if a >= 0.0 { a - PI } else { a + PI });
        let new_apertures = self.apertures.mapv(|a| PI - a);

        NdarrayCone::from_raw(new_axes, new_apertures)
    }

    /// Compute the intersection of two cones.
    ///
    /// Uses the closed-form circular mean for axes (attention-weighted average in
    /// Cartesian coordinates, then atan2 back to angle) and per-dimension minimum
    /// for apertures.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if cones have different dimensions.
    pub fn intersection(&self, other: &Self) -> Result<Self, ConeError> {
        if self.dim() != other.dim() {
            return Err(ConeError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let d = self.dim();
        let mut new_axes = Array1::zeros(d);
        let mut new_apertures = Array1::zeros(d);

        for i in 0..d {
            // Circular mean: convert to (cos, sin), average, then atan2.
            let x = self.axes[i].cos() + other.axes[i].cos();
            let y = self.axes[i].sin() + other.axes[i].sin();

            // Guard against zero-length resultant (opposite angles).
            if x.abs() < 1e-8 && y.abs() < 1e-8 {
                new_axes[i] = self.axes[i]; // Arbitrary choice when opposite.
            } else {
                new_axes[i] = y.atan2(x);
            }

            // Per-dimension minimum aperture.
            new_apertures[i] = self.apertures[i].min(other.apertures[i]);
        }

        Ok(NdarrayCone::from_raw(new_axes, new_apertures))
    }

    /// Apply a relation projection to this cone.
    ///
    /// Per-dimension:
    /// - axis\[i\] += relation_axis\[i\] (modular addition, wrapped to \[-pi, pi\])
    /// - aperture\[i\] = clamp(aperture\[i\] + relation_aperture\[i\], 0, pi)
    ///
    /// The relation transforms the cone's position and width in each angular sector.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if dimensions don't match.
    pub fn project(
        &self,
        relation_axes: &Array1<f32>,
        relation_apertures: &Array1<f32>,
    ) -> Result<Self, ConeError> {
        if self.dim() != relation_axes.len() || self.dim() != relation_apertures.len() {
            return Err(ConeError::DimensionMismatch {
                expected: self.dim(),
                actual: relation_axes.len(),
            });
        }

        // Per-dimension rotation + aperture adjustment.
        let new_axes = (&self.axes + relation_axes).mapv(normalize_angle);
        let new_apertures = (&self.apertures + relation_apertures).mapv(clamp_aperture);

        Ok(NdarrayCone::from_raw(new_axes, new_apertures))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    // ---- Construction ----

    #[test]
    fn new_normalizes_axes_and_clamps_apertures() {
        let cone = NdarrayCone::new(
            array![4.0, -4.0], // Outside [-pi, pi]
            array![-0.5, 4.0], // Outside [0, pi]
        )
        .unwrap();

        for &a in cone.axes().iter() {
            assert!(
                (-PI..=PI).contains(&a),
                "axis should be in [-pi, pi], got {a}"
            );
        }
        for &a in cone.apertures().iter() {
            assert!(
                (0.0..=PI).contains(&a),
                "aperture should be in [0, pi], got {a}"
            );
        }
    }

    #[test]
    fn new_rejects_dimension_mismatch() {
        let result = NdarrayCone::new(array![0.0, 0.0], array![0.5]);
        assert!(matches!(result, Err(ConeError::DimensionMismatch { .. })));
    }

    #[test]
    fn new_rejects_nan_axes() {
        let result = NdarrayCone::new(array![f32::NAN, 0.0], array![0.5, 0.5]);
        assert!(matches!(result, Err(ConeError::InvalidBounds { .. })));
    }

    #[test]
    fn new_rejects_nan_apertures() {
        let result = NdarrayCone::new(array![0.0, 0.0], array![0.5, f32::NAN]);
        assert!(matches!(result, Err(ConeError::InvalidBounds { .. })));
    }

    // ---- Containment / Distance ----

    #[test]
    fn distance_self_is_small() {
        // A cone's distance to itself should be small (only the cen * inside component).
        let cone = NdarrayCone::new(array![0.5, -0.3, 1.0], array![0.8, 0.5, 1.2]).unwrap();
        let d = cone.cone_distance(&cone, 0.02).unwrap();
        // Each dimension contributes cen * min(0, distance_base) for the inside part.
        // Since entity == query, distance_to_axis == 0 < distance_base, so dist_in = 0.
        assert!(d < 0.01, "Self-distance should be near zero, got {d}");
    }

    #[test]
    fn wider_cone_has_lower_distance_to_entity() {
        // A wider cone should have lower distance to an entity inside it.
        let entity = NdarrayCone::new(array![0.3, -0.2], array![0.1, 0.1]).unwrap();
        let wide = NdarrayCone::new(array![0.3, -0.2], array![1.5, 1.5]).unwrap();
        let narrow = NdarrayCone::new(array![0.3, -0.2], array![0.2, 0.2]).unwrap();

        let d_wide = wide.cone_distance(&entity, 0.02).unwrap();
        let d_narrow = narrow.cone_distance(&entity, 0.02).unwrap();

        assert!(
            d_wide <= d_narrow + 1e-6,
            "Wider cone should have <= distance: wide={d_wide}, narrow={d_narrow}"
        );
    }

    #[test]
    fn distant_entity_has_high_distance() {
        let query = NdarrayCone::new(array![0.0, 0.0], array![0.3, 0.3]).unwrap();
        let near = NdarrayCone::new(array![0.1, 0.1], array![0.1, 0.1]).unwrap();
        let far = NdarrayCone::new(array![PI, PI], array![0.1, 0.1]).unwrap();

        let d_near = query.cone_distance(&near, 0.02).unwrap();
        let d_far = query.cone_distance(&far, 0.02).unwrap();

        assert!(
            d_far > d_near,
            "Far entity should have higher distance: near={d_near}, far={d_far}"
        );
    }

    #[test]
    fn distance_dimension_mismatch() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![0.5, 0.5]).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0, 0.0], array![0.5, 0.5, 0.5]).unwrap();
        let result = a.cone_distance(&b, 0.02);
        assert!(matches!(result, Err(ConeError::DimensionMismatch { .. })));
    }

    // ---- Complement / Negation ----

    #[test]
    fn complement_aperture_is_pi_minus_original() {
        let cone = NdarrayCone::new(array![0.5, -0.3], array![0.8, 1.2]).unwrap();
        let comp = cone.complement();
        for (i, (&orig, &neg)) in cone
            .apertures()
            .iter()
            .zip(comp.apertures().iter())
            .enumerate()
        {
            let expected = PI - orig;
            assert!(
                (neg - expected).abs() < 1e-6,
                "Complement aperture[{i}] should be pi - {orig} = {expected}, got {neg}"
            );
        }
    }

    #[test]
    fn complement_axis_shifts_by_pi() {
        let cone = NdarrayCone::new(array![0.5, -0.3], array![0.8, 1.2]).unwrap();
        let comp = cone.complement();
        // axis[0] = 0.5 (positive) -> 0.5 - pi
        assert!(
            (comp.axes()[0] - (0.5 - PI)).abs() < 1e-6,
            "Complement axes[0] should be 0.5 - pi, got {}",
            comp.axes()[0]
        );
        // axis[1] = -0.3 (negative) -> -0.3 + pi
        assert!(
            (comp.axes()[1] - (-0.3 + PI)).abs() < 1e-6,
            "Complement axes[1] should be -0.3 + pi, got {}",
            comp.axes()[1]
        );
    }

    #[test]
    fn double_complement_is_identity() {
        let cone = NdarrayCone::new(array![0.5, -0.3, 2.0], array![0.8, 1.2, 0.5]).unwrap();
        let double = cone.complement().complement();

        for (i, (&orig, &dc)) in cone.axes().iter().zip(double.axes().iter()).enumerate() {
            assert!(
                (orig - dc).abs() < 1e-5,
                "Double complement axes[{i}]: {orig} vs {dc}"
            );
        }
        for (i, (&orig, &dc)) in cone
            .apertures()
            .iter()
            .zip(double.apertures().iter())
            .enumerate()
        {
            assert!(
                (orig - dc).abs() < 1e-5,
                "Double complement apertures[{i}]: {orig} vs {dc}"
            );
        }
    }

    // ---- Intersection ----

    #[test]
    fn intersection_of_identical_cones_preserves_apertures() {
        let cone = NdarrayCone::new(array![0.5, -0.3], array![0.8, 1.2]).unwrap();
        let inter = cone.intersection(&cone).unwrap();

        for (i, (&orig, &intr)) in cone
            .apertures()
            .iter()
            .zip(inter.apertures().iter())
            .enumerate()
        {
            assert!(
                (orig - intr).abs() < 1e-5,
                "Intersection aperture[{i}] should match: {orig} vs {intr}"
            );
        }
    }

    #[test]
    fn intersection_takes_min_aperture() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![1.0, 0.5]).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0], array![0.3, 0.8]).unwrap();
        let inter = a.intersection(&b).unwrap();

        assert!((inter.apertures()[0] - 0.3).abs() < 1e-6);
        assert!((inter.apertures()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn intersection_dimension_mismatch() {
        let a = NdarrayCone::new(array![0.0, 0.0], array![0.5, 0.5]).unwrap();
        let b = NdarrayCone::new(array![0.0, 0.0, 0.0], array![0.5, 0.5, 0.5]).unwrap();
        let result = a.intersection(&b);
        assert!(matches!(result, Err(ConeError::DimensionMismatch { .. })));
    }

    // ---- Projection ----

    #[test]
    fn projection_rotates_axes() {
        let cone = NdarrayCone::new(array![0.0, 0.0], array![0.5, 0.5]).unwrap();
        let projected = cone.project(&array![0.5, -0.3], &array![0.0, 0.0]).unwrap();
        assert!((projected.axes()[0] - 0.5).abs() < 1e-6);
        assert!((projected.axes()[1] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn projection_adjusts_apertures() {
        let cone = NdarrayCone::new(array![0.0, 0.0], array![0.5, 0.5]).unwrap();
        let projected = cone.project(&array![0.0, 0.0], &array![0.3, -0.2]).unwrap();
        assert!((projected.apertures()[0] - 0.8).abs() < 1e-6);
        assert!((projected.apertures()[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn projection_clamps_apertures() {
        let cone = NdarrayCone::new(array![0.0], array![2.5]).unwrap();
        let projected = cone.project(&array![0.0], &array![2.0]).unwrap();
        // 2.5 + 2.0 = 4.5 > pi, should clamp to pi.
        assert!((projected.apertures()[0] - PI).abs() < 1e-6);
    }

    #[test]
    fn projection_wraps_axes() {
        let cone = NdarrayCone::new(array![2.5], array![0.5]).unwrap();
        let projected = cone.project(&array![2.0], &array![0.0]).unwrap();
        // 2.5 + 2.0 = 4.5, wraps to 4.5 - 2*pi ~ -1.783
        let expected = normalize_angle(4.5);
        assert!(
            (projected.axes()[0] - expected).abs() < 1e-5,
            "Expected {expected}, got {}",
            projected.axes()[0]
        );
    }

    // ---- Higher dimensions ----

    #[test]
    fn works_in_high_dimensions() {
        let d = 128;
        let axes_a = Array1::from_vec(vec![0.5; d]);
        let aper_a = Array1::from_vec(vec![1.0; d]);
        let axes_b = Array1::from_vec(vec![0.6; d]);
        let aper_b = Array1::from_vec(vec![0.3; d]);

        let a = NdarrayCone::new(axes_a, aper_a).unwrap();
        let b = NdarrayCone::new(axes_b, aper_b).unwrap();

        let dist = a.cone_distance(&b, 0.02).unwrap();
        assert!(dist.is_finite());

        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter.dim(), d);

        let comp = a.complement();
        assert_eq!(comp.dim(), d);

        let rel_axes = Array1::from_vec(vec![0.1; d]);
        let rel_aper = Array1::from_vec(vec![0.05; d]);
        let proj = a.project(&rel_axes, &rel_aper).unwrap();
        assert_eq!(proj.dim(), d);
    }

    // ---- Numerical stability ----

    #[test]
    fn distance_finite_at_boundary_values() {
        // Aperture at 0 (point) and pi (full circle).
        let point = NdarrayCone::new(array![0.0], array![0.0]).unwrap();
        let full = NdarrayCone::new(array![0.0], array![PI]).unwrap();

        let d1 = full.cone_distance(&point, 0.02).unwrap();
        assert!(d1.is_finite(), "Distance must be finite, got {d1}");

        let d2 = point.cone_distance(&full, 0.02).unwrap();
        assert!(d2.is_finite(), "Distance must be finite, got {d2}");
    }

    // ---- Property tests ----

    fn arb_axes(dim: usize) -> impl Strategy<Value = Array1<f32>> {
        proptest::collection::vec(-PI..PI, dim).prop_map(Array1::from)
    }

    fn arb_apertures(dim: usize) -> impl Strategy<Value = Array1<f32>> {
        proptest::collection::vec(0.01f32..PI - 0.01, dim).prop_map(Array1::from)
    }

    fn arb_cone(dim: usize) -> impl Strategy<Value = NdarrayCone> {
        (arb_axes(dim), arb_apertures(dim))
            .prop_map(|(axes, apertures)| NdarrayCone::from_raw(axes, apertures))
    }

    proptest! {
        /// complement(complement(cone)) = cone (involution).
        #[test]
        fn prop_complement_of_complement_is_identity(cone in arb_cone(4)) {
            let double = cone.complement().complement();
            for (i, (&orig, &dc)) in cone.axes().iter().zip(double.axes().iter()).enumerate() {
                prop_assert!(
                    (orig - dc).abs() < 1e-4,
                    "double complement axes[{i}]: {orig} vs {dc}"
                );
            }
            for (i, (&orig, &dc)) in cone.apertures().iter().zip(double.apertures().iter()).enumerate() {
                prop_assert!(
                    (orig - dc).abs() < 1e-4,
                    "double complement apertures[{i}]: {orig} vs {dc}"
                );
            }
        }

        /// aperture + complement.aperture = pi (per dimension).
        #[test]
        fn prop_complement_aperture_sum_is_pi(cone in arb_cone(4)) {
            let comp = cone.complement();
            for (i, (&orig, &neg)) in cone.apertures().iter().zip(comp.apertures().iter()).enumerate() {
                let sum = orig + neg;
                prop_assert!(
                    (sum - PI).abs() < 1e-5,
                    "aperture[{i}] sum should be pi: {orig} + {neg} = {sum}"
                );
            }
        }

        /// Any cone's distance to itself should be very small (reflexive containment).
        #[test]
        fn prop_containment_is_reflexive(cone in arb_cone(4)) {
            let d = cone.cone_distance(&cone, 0.02).unwrap();
            // Self-distance: distance_to_axis = 0 in every dim, so all dims are "inside"
            // with dist_in = 0. Total = cen * 0 = 0.
            prop_assert!(
                d < 1e-5,
                "Self-distance should be near zero, got {d}"
            );
        }

        /// If all apertures of A >= B and axes match, A "contains" B (lower distance).
        #[test]
        fn prop_wider_cone_contains_narrower(
            axes in arb_axes(4),
            aper_wide in proptest::collection::vec(0.5f32..PI - 0.01, 4usize).prop_map(Array1::from),
        ) {
            // Make narrow apertures strictly smaller.
            let aper_narrow = aper_wide.mapv(|a| (a - 0.3).max(0.01));
            let entity = NdarrayCone::from_raw(axes.clone(), aper_narrow.clone());
            let wide = NdarrayCone::from_raw(axes.clone(), aper_wide.clone());
            let narrow = NdarrayCone::from_raw(axes.clone(), aper_narrow);

            let d_wide = wide.cone_distance(&entity, 0.02).unwrap();
            let d_narrow = narrow.cone_distance(&entity, 0.02).unwrap();

            prop_assert!(
                d_wide <= d_narrow + 1e-5,
                "Wider cone should have <= distance to entity: wide={d_wide}, narrow={d_narrow}"
            );
        }

        /// Modifying dimension i does not affect the scoring contribution from dimension j.
        #[test]
        fn prop_per_dimension_independence(
            cone in arb_cone(4),
            entity in arb_cone(4),
        ) {
            // Compute full distance, then modify dim 0 of entity and check that
            // the change in distance comes only from dim 0's contribution.
            let d_orig = cone.cone_distance(&entity, 0.0).unwrap(); // cen=0 so only outside

            // Create modified entity with dim 0 shifted.
            let mut modified_axes = entity.axes().clone();
            modified_axes[0] = normalize_angle(modified_axes[0] + 0.5);
            let modified = NdarrayCone::from_raw(modified_axes, entity.apertures().clone());
            let d_mod = cone.cone_distance(&modified, 0.0).unwrap();

            // The difference should be explainable by dim 0 alone.
            // Compute dim-0 only contributions.
            let dim0_orig = {
                let c = NdarrayCone::from_raw(
                    array![cone.axes()[0]],
                    array![cone.apertures()[0]],
                );
                let e = NdarrayCone::from_raw(
                    array![entity.axes()[0]],
                    array![entity.apertures()[0]],
                );
                c.cone_distance(&e, 0.0).unwrap()
            };
            let dim0_mod = {
                let c = NdarrayCone::from_raw(
                    array![cone.axes()[0]],
                    array![cone.apertures()[0]],
                );
                let e = NdarrayCone::from_raw(
                    array![modified.axes()[0]],
                    array![modified.apertures()[0]],
                );
                c.cone_distance(&e, 0.0).unwrap()
            };

            let full_delta = (d_mod - d_orig).abs();
            let dim0_delta = (dim0_mod - dim0_orig).abs();

            prop_assert!(
                (full_delta - dim0_delta).abs() < 1e-4,
                "Modifying dim 0 should only affect dim 0: full_delta={full_delta}, dim0_delta={dim0_delta}"
            );
        }

        /// After projection, result is still a valid cone (apertures in [0, pi], axes in [-pi, pi]).
        #[test]
        fn prop_projection_preserves_cone_structure(
            cone in arb_cone(4),
            rel_axes in arb_axes(4),
            rel_aper in proptest::collection::vec(-1.0f32..1.0, 4usize).prop_map(Array1::from),
        ) {
            let projected = cone.project(&rel_axes, &rel_aper).unwrap();

            for (i, &a) in projected.axes().iter().enumerate() {
                prop_assert!(
                    (-PI - 1e-6..=PI + 1e-6).contains(&a),
                    "Projected axes[{i}] should be in [-pi, pi], got {a}"
                );
            }
            for (i, &a) in projected.apertures().iter().enumerate() {
                prop_assert!(
                    (-1e-6..=PI + 1e-6).contains(&a),
                    "Projected apertures[{i}] should be in [0, pi], got {a}"
                );
            }
        }
    }
}
