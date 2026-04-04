//! Spherical cap embeddings for subsumption on the unit sphere.
//!
//! Each concept is represented as a spherical cap: a region on S^{d-1}
//! defined by a center (unit vector) and an angular radius. Subsumption
//! is modeled via angular containment: cap A is contained in cap B iff
//! `angle(c_A, c_B) + theta_A <= theta_B`.
//!
//! # Motivation
//!
//! The existing `spherical` module represents entities as points on S^{d-1}
//! with rotation-based relations — no subsumption. Spherical caps extend
//! this by adding an angular radius, turning points into regions that can
//! contain one another. This is the spherical analog of ball containment
//! in Euclidean space.
//!
//! No published paper has done spherical cap subsumption. This module
//! implements the natural extension:
//!
//! - Points become caps with angular radius (degenerate cap: theta = 0)
//! - Relations combine rotation (existing) with angular scaling
//! - Containment uses angular distance, not Euclidean distance
//!
//! # Containment formula
//!
//! ```text
//! cap_A ⊆ cap_B  ⟺  angle(c_A, c_B) + theta_A <= theta_B
//! ```
//!
//! where `angle(c_A, c_B) = arccos(dot(c_A, c_B))` is the geodesic distance
//! between centers on the sphere.
//!
//! For soft containment (differentiable scoring):
//!
//! ```text
//! P(A ⊆ B) = sigmoid(k * (theta_B - angle(c_A, c_B) - theta_A))
//! ```
//!
//! # Angular distance
//!
//! The geodesic (great-circle) distance between two points on S^{d-1}:
//!
//! ```text
//! angle(a, b) = arccos(clamp(dot(a, b), -1, 1))
//! ```
//!
//! Returns a value in [0, pi].

use crate::BoxError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A spherical cap on S^{d-1}.
///
/// Defined by a center (unit vector) and an angular radius in (0, pi].
/// The cap contains all points on the sphere within angular distance
/// `theta` of the center.
///
/// # Containment
///
/// Cap A is contained in cap B iff `angle(c_A, c_B) + theta_A <= theta_B`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SphericalCap {
    center: Vec<f32>,
    angular_radius: f32,
}

/// A relation transform for spherical caps.
///
/// Combines rotation of the center with scaling of the angular radius:
///
/// ```text
/// transform(cap, relation) = SphericalCap(
///     center = rotate(cap.center, relation.axis, relation.angle),
///     angular_radius = cap.angular_radius * relation.angle_scale,
/// )
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SphericalCapRelation {
    axis: Vec<f32>,
    angle: f32,
    angle_scale: f32,
}

/// Collection of spherical cap embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphericalCapEmbedding {
    entities: Vec<SphericalCap>,
    relations: Vec<SphericalCapRelation>,
    dim: usize,
}

// ---------------------------------------------------------------------------
// SphericalCap construction and accessors
// ---------------------------------------------------------------------------

impl SphericalCap {
    /// Create a new spherical cap.
    ///
    /// The center is automatically normalized to a unit vector.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if angular_radius is not in (0, pi],
    /// if the center is a zero vector, or if any coordinate is non-finite.
    pub fn new(center: Vec<f32>, angular_radius: f32) -> Result<Self, BoxError> {
        if !angular_radius.is_finite()
            || angular_radius <= 0.0
            || angular_radius > std::f32::consts::PI
        {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: angular_radius as f64,
            });
        }
        for (i, &c) in center.iter().enumerate() {
            if !c.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: c as f64,
                    max: c as f64,
                });
            }
        }
        let norm = center.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-12 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: 0.0,
            });
        }
        let center: Vec<f32> = center.iter().map(|x| x / norm).collect();
        Ok(Self {
            center,
            angular_radius,
        })
    }

    /// Create a cap from center and log-tan(theta/2) parameterization.
    ///
    /// This maps R -> (0, pi) smoothly, useful for training:
    /// `theta = 2 * atan(exp(log_tan_half))`
    pub fn from_log_tan_half(center: Vec<f32>, log_tan_half: f32) -> Result<Self, BoxError> {
        if !log_tan_half.is_finite() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: log_tan_half as f64,
                max: log_tan_half as f64,
            });
        }
        let theta = 2.0 * log_tan_half.exp().atan();
        Self::new(center, theta)
    }

    /// Dimensionality of the ambient space.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.center.len()
    }

    /// Returns a reference to the center (unit vector).
    pub fn center(&self) -> &[f32] {
        &self.center
    }

    /// Angular radius in radians.
    #[must_use]
    pub fn angular_radius(&self) -> f32 {
        self.angular_radius
    }

    /// Log-tan(theta/2) parameter (for training stability).
    #[must_use]
    pub fn log_tan_half(&self) -> f32 {
        (self.angular_radius / 2.0).tan().ln()
    }

    /// Surface area of the cap as a fraction of the full sphere.
    ///
    /// For d=2 (circle): fraction = theta / pi
    /// For d=3 (sphere): fraction = (1 - cos(theta)) / 2
    /// For general d: uses regularized incomplete beta function approximation.
    #[must_use]
    pub fn area_fraction(&self) -> f32 {
        let d = self.center.len();
        match d {
            2 => self.angular_radius / std::f32::consts::PI,
            3 => (1.0 - self.angular_radius.cos()) / 2.0,
            _ => {
                // Approximation using the dominant term for high dimensions
                // Area fraction ~ (sin(theta))^(d-1) / B((d-1)/2, 1/2)
                let sin_theta = self.angular_radius.sin();
                if sin_theta < 1e-12 {
                    return 0.0;
                }
                let log_sin = (d as f32 - 1.0) * sin_theta.ln();
                // Clamp to avoid overflow
                if log_sin < -50.0 {
                    0.0
                } else {
                    log_sin.exp().min(1.0)
                }
            }
        }
    }

    /// Mutable access to center (for training).
    pub fn center_mut(&mut self) -> &mut [f32] {
        &mut self.center
    }

    /// Set angular radius from log-tan(theta/2) (for training).
    pub fn set_log_tan_half(&mut self, log_tan_half: f32) {
        let clamped = log_tan_half.clamp(-10.0, 10.0);
        self.angular_radius =
            (2.0 * clamped.exp().atan()).clamp(0.001, std::f32::consts::PI - 0.001);
    }
}

// ---------------------------------------------------------------------------
// SphericalCapRelation
// ---------------------------------------------------------------------------

impl SphericalCapRelation {
    /// Create a new spherical cap relation.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if angle_scale is non-positive,
    /// or if any coordinate is non-finite.
    pub fn new(axis: Vec<f32>, angle: f32, angle_scale: f32) -> Result<Self, BoxError> {
        if !angle_scale.is_finite() || angle_scale <= 0.0 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: angle_scale as f64,
            });
        }
        for (i, &a) in axis.iter().enumerate() {
            if !a.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: a as f64,
                    max: a as f64,
                });
            }
        }
        if !angle.is_finite() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: angle as f64,
                max: angle as f64,
            });
        }
        let norm = axis.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-12 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: 0.0,
            });
        }
        let axis: Vec<f32> = axis.iter().map(|x| x / norm).collect();
        Ok(Self {
            axis,
            angle,
            angle_scale,
        })
    }

    /// Identity relation (zero rotation, unit scale).
    #[must_use]
    pub fn identity(dim: usize) -> Self {
        let mut axis = vec![0.0; dim];
        if dim > 0 {
            axis[0] = 1.0;
        }
        Self {
            axis,
            angle: 0.0,
            angle_scale: 1.0,
        }
    }

    /// Apply this relation to a cap.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if dimensions disagree.
    pub fn apply(&self, cap: &SphericalCap) -> Result<SphericalCap, BoxError> {
        if self.axis.len() != cap.center.len() {
            return Err(BoxError::DimensionMismatch {
                expected: cap.center.len(),
                actual: self.axis.len(),
            });
        }
        let new_center = rotate_vector(&cap.center, &self.axis, self.angle);
        let new_radius = cap.angular_radius * self.angle_scale;
        // Clamp angular radius to valid range
        let new_radius = new_radius.clamp(0.001, std::f32::consts::PI);
        SphericalCap::new(new_center, new_radius)
    }
}

// ---------------------------------------------------------------------------
// SphericalCapEmbedding
// ---------------------------------------------------------------------------

impl SphericalCapEmbedding {
    /// Create a new spherical cap embedding model.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any component has wrong dimension.
    pub fn new(
        entities: Vec<SphericalCap>,
        relations: Vec<SphericalCapRelation>,
        dim: usize,
    ) -> Result<Self, BoxError> {
        for e in &entities {
            if e.dim() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: e.dim(),
                });
            }
        }
        for r in &relations {
            if r.axis.len() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: r.axis.len(),
                });
            }
        }
        Ok(Self {
            entities,
            relations,
            dim,
        })
    }

    /// Embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of entity caps.
    #[must_use]
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    /// Number of relation transforms.
    #[must_use]
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }

    /// Entity caps.
    pub fn entities(&self) -> &[SphericalCap] {
        &self.entities
    }

    /// Relation transforms.
    pub fn relations(&self) -> &[SphericalCapRelation] {
        &self.relations
    }
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// Geodesic (great-circle) distance between two unit vectors.
///
/// ```text
/// angle(a, b) = arccos(clamp(dot(a, b), -1, 1))
/// ```
///
/// Returns a value in [0, pi].
pub fn geodesic_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    (dot.clamp(-1.0, 1.0)).acos()
}

/// Soft containment probability: P(inner ⊆ outer).
///
/// ```text
/// P = sigmoid(k * (theta_outer - angle(c_inner, c_outer) - theta_inner))
/// ```
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the caps differ in dimension.
pub fn containment_prob(
    inner: &SphericalCap,
    outer: &SphericalCap,
    k: f32,
) -> Result<f32, BoxError> {
    if inner.dim() != outer.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: inner.dim(),
            actual: outer.dim(),
        });
    }
    let dist = geodesic_distance(&inner.center, &outer.center);
    let margin = outer.angular_radius - dist - inner.angular_radius;
    Ok(sigmoid(k * margin))
}

/// Angular surface distance between two caps.
///
/// ```text
/// d = max(0, angle(c_A, c_B) - theta_A - theta_B)
/// ```
///
/// Returns 0 when caps overlap, positive gap when disjoint.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the caps differ in dimension.
pub fn surface_distance(a: &SphericalCap, b: &SphericalCap) -> Result<f32, BoxError> {
    if a.dim() != b.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim(),
            actual: b.dim(),
        });
    }
    let dist = geodesic_distance(&a.center, &b.center);
    Ok((dist - a.angular_radius - b.angular_radius).max(0.0))
}

/// Overlap probability (angular proxy).
///
/// ```text
/// overlap = max(0, theta_A + theta_B - angle(c_A, c_B)) / (theta_A + theta_B)
/// ```
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the caps differ in dimension.
pub fn overlap_prob(a: &SphericalCap, b: &SphericalCap) -> Result<f32, BoxError> {
    if a.dim() != b.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim(),
            actual: b.dim(),
        });
    }
    let dist = geodesic_distance(&a.center, &b.center);
    let sum_theta = a.angular_radius + b.angular_radius;
    if sum_theta < 1e-12 {
        return Ok(0.0);
    }
    let overlap_depth = (sum_theta - dist).max(0.0);
    Ok(overlap_depth / sum_theta)
}

/// Score a triple (head, relation, tail) using spherical cap embeddings.
///
/// Transforms the head cap by the relation, then computes angular surface
/// distance to the tail cap. Lower scores indicate better matches.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if any component has mismatched dimensions.
pub fn score_triple(
    head: &SphericalCap,
    relation: &SphericalCapRelation,
    tail: &SphericalCap,
) -> Result<f32, BoxError> {
    let transformed = relation.apply(head)?;
    surface_distance(&transformed, tail)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Rotate a vector by angle `theta` in the plane spanned by the vector
/// and a unit axis vector.
///
/// For a vector v and unit axis a:
/// - Decompose v into parallel and perpendicular components: v = v_par + v_perp
/// - v_par = (v . a) * a
/// - v_perp = v - v_par
/// - rotated = v_par + cos(theta) * v_perp + sin(theta) * ||v_perp|| * a
fn rotate_vector(v: &[f32], axis: &[f32], angle: f32) -> Vec<f32> {
    debug_assert_eq!(v.len(), axis.len());
    let d = v.len();

    let dot: f32 = v.iter().zip(axis.iter()).map(|(&x, &y)| x * y).sum();
    let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();

    if v_norm < 1e-12 {
        return v.to_vec();
    }

    // Parallel component
    let v_par: Vec<f32> = axis.iter().map(|&a| a * dot).collect();
    // Perpendicular component
    let v_perp: Vec<f32> = v.iter().zip(v_par.iter()).map(|(&v, &p)| v - p).collect();
    let perp_norm = v_perp.iter().map(|x| x * x).sum::<f32>().sqrt();

    if perp_norm < 1e-12 {
        // v is parallel to axis, no rotation possible
        return v.to_vec();
    }

    let cos_a = angle.cos();
    let sin_a = angle.sin();

    if d == 3 {
        // Rodrigues' formula: v_rot = v_par + cos(a)*v_perp + sin(a)*(axis × v_perp)
        let cross = [
            axis[1] * v_perp[2] - axis[2] * v_perp[1],
            axis[2] * v_perp[0] - axis[0] * v_perp[2],
            axis[0] * v_perp[1] - axis[1] * v_perp[0],
        ];
        v_par
            .iter()
            .zip(v_perp.iter())
            .zip(cross.iter())
            .map(|((&vp, &vpe), &cr)| vp + cos_a * vpe + sin_a * cr)
            .collect()
    } else {
        // For d != 3: rotate in the plane spanned by v_perp and a perpendicular
        // direction. We use the Gram-Schmidt approach: find a unit vector
        // orthogonal to both axis and v_perp by rotating v_perp in the
        // (v_perp, axis)-plane by 90 degrees.
        // This is a generalized rotation that preserves the angle with the axis.
        let w: Vec<f32> = axis
            .iter()
            .zip(v_perp.iter())
            .map(|(&a, &vp)| (a * perp_norm - vp * dot) / perp_norm)
            .collect();
        v_par
            .iter()
            .zip(v_perp.iter())
            .zip(w.iter())
            .map(|((&vp, &vpe), &we)| vp + cos_a * vpe + sin_a * we)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn cap_new_valid() {
        let c = SphericalCap::new(vec![1.0, 0.0, 0.0], PI / 4.0).unwrap();
        assert_eq!(c.dim(), 3);
        assert!((c.angular_radius() - PI / 4.0).abs() < 1e-6);
        // center should be normalized
        let norm: f32 = c.center().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cap_normalizes_center() {
        let c = SphericalCap::new(vec![3.0, 4.0, 0.0], 0.5).unwrap();
        let norm: f32 = c.center().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cap_from_log_tan_half() {
        let c = SphericalCap::from_log_tan_half(vec![1.0, 0.0], 0.0).unwrap();
        // theta = 2 * atan(exp(0)) = 2 * atan(1) = 2 * pi/4 = pi/2
        assert!((c.angular_radius() - PI / 2.0).abs() < 1e-5);
    }

    #[test]
    fn cap_rejects_zero_radius() {
        assert!(SphericalCap::new(vec![1.0, 0.0], 0.0).is_err());
    }

    #[test]
    fn cap_rejects_too_large_radius() {
        assert!(SphericalCap::new(vec![1.0, 0.0], PI + 0.1).is_err());
    }

    #[test]
    fn cap_rejects_zero_vector() {
        assert!(SphericalCap::new(vec![0.0, 0.0], 0.5).is_err());
    }

    #[test]
    fn cap_rejects_non_finite() {
        assert!(SphericalCap::new(vec![f32::NAN, 0.0], 0.5).is_err());
        assert!(SphericalCap::new(vec![1.0, 0.0], f32::NAN).is_err());
    }

    // --- Containment ---

    #[test]
    fn containment_identical_is_half() {
        let a = SphericalCap::new(vec![1.0, 0.0, 0.0], PI / 4.0).unwrap();
        let p = containment_prob(&a, &a, 10.0).unwrap();
        // margin = 0, sigmoid(0) = 0.5
        assert!((p - 0.5).abs() < 1e-4);
    }

    #[test]
    fn containment_nested_is_near_one() {
        let inner = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.1).unwrap();
        let outer = SphericalCap::new(vec![1.0, 0.0, 0.0], 1.0).unwrap();
        let p = containment_prob(&inner, &outer, 10.0).unwrap();
        // margin = 1.0 - 0 - 0.1 = 0.9, sigmoid(9) ≈ 1
        assert!(p > 0.99, "nested containment = {p}, expected > 0.99");
    }

    #[test]
    fn containment_disjoint_is_near_zero() {
        let a = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.1).unwrap();
        let b = SphericalCap::new(vec![-1.0, 0.0, 0.0], 0.1).unwrap();
        let p = containment_prob(&a, &b, 10.0).unwrap();
        // angle = pi, margin = 0.1 - pi - 0.1 = -pi, sigmoid(-31) ≈ 0
        assert!(p < 1e-4, "disjoint containment = {p}, expected ~0");
    }

    #[test]
    fn containment_dimension_mismatch() {
        let a = SphericalCap::new(vec![1.0, 0.0], 0.5).unwrap();
        let b = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap();
        assert!(containment_prob(&a, &b, 1.0).is_err());
    }

    // --- Surface distance ---

    #[test]
    fn surface_distance_overlapping_is_zero() {
        let a = SphericalCap::new(vec![1.0, 0.0, 0.0], 1.0).unwrap();
        let b = SphericalCap::new(vec![0.0, 1.0, 0.0], 1.0).unwrap();
        let d = surface_distance(&a, &b).unwrap();
        // angle = pi/2 ≈ 1.57, sum_theta = 2.0, gap = max(0, 1.57 - 2.0) = 0
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn surface_distance_disjoint_is_positive() {
        let a = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.1).unwrap();
        let b = SphericalCap::new(vec![-1.0, 0.0, 0.0], 0.1).unwrap();
        let d = surface_distance(&a, &b).unwrap();
        // angle = pi, sum_theta = 0.2, gap = pi - 0.2 ≈ 2.94
        assert!((d - (PI - 0.2)).abs() < 1e-4);
    }

    #[test]
    fn surface_distance_identical_is_zero() {
        let a = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap();
        let d = surface_distance(&a, &a).unwrap();
        assert!(d.abs() < 1e-6);
    }

    // --- Overlap ---

    #[test]
    fn overlap_identical_is_one() {
        let a = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap();
        let p = overlap_prob(&a, &a).unwrap();
        assert!((p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn overlap_disjoint_is_zero() {
        let a = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.1).unwrap();
        let b = SphericalCap::new(vec![-1.0, 0.0, 0.0], 0.1).unwrap();
        let p = overlap_prob(&a, &b).unwrap();
        assert!(p.abs() < 1e-6);
    }

    #[test]
    fn overlap_symmetric() {
        let a = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap();
        let b = SphericalCap::new(vec![0.0, 1.0, 0.0], 0.8).unwrap();
        let p_ab = overlap_prob(&a, &b).unwrap();
        let p_ba = overlap_prob(&b, &a).unwrap();
        assert!((p_ab - p_ba).abs() < 1e-6);
    }

    // --- Area fraction ---

    #[test]
    fn area_fraction_2d() {
        let c = SphericalCap::new(vec![1.0, 0.0], PI / 2.0).unwrap();
        let f = c.area_fraction();
        assert!((f - 0.5).abs() < 1e-6, "half circle = 0.5, got {f}");
    }

    #[test]
    fn area_fraction_3d_hemisphere() {
        let c = SphericalCap::new(vec![1.0, 0.0, 0.0], PI / 2.0).unwrap();
        let f = c.area_fraction();
        assert!((f - 0.5).abs() < 1e-6, "hemisphere = 0.5, got {f}");
    }

    // --- Relation ---

    #[test]
    fn relation_identity_preserves_cap() {
        let c = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap();
        let r = SphericalCapRelation::identity(3);
        let t = r.apply(&c).unwrap();
        for (i, (&a, &b)) in c.center().iter().zip(t.center().iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "center[{i}] changed: {a} -> {b}");
        }
        assert!((t.angular_radius() - c.angular_radius()).abs() < 1e-6);
    }

    #[test]
    fn relation_rotation() {
        // 90-degree rotation around z-axis in 3D
        let c = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap();
        let r = SphericalCapRelation::new(vec![0.0, 0.0, 1.0], PI / 2.0, 1.0).unwrap();
        let t = r.apply(&c).unwrap();
        // [1,0,0] rotated 90° around z → [0,1,0]
        assert!(t.center()[0].abs() < 1e-5);
        assert!((t.center()[1] - 1.0).abs() < 1e-5);
        assert!((t.angular_radius() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn relation_scaling() {
        let c = SphericalCap::new(vec![1.0, 0.0], 0.5).unwrap();
        let r = SphericalCapRelation::new(vec![0.0, 1.0], 0.0, 2.0).unwrap();
        let t = r.apply(&c).unwrap();
        assert!((t.angular_radius() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn relation_rejects_zero_scale() {
        assert!(SphericalCapRelation::new(vec![1.0, 0.0], 0.0, 0.0).is_err());
    }

    #[test]
    fn relation_dimension_mismatch() {
        let c = SphericalCap::new(vec![1.0, 0.0], 0.5).unwrap();
        let r = SphericalCapRelation::new(vec![0.0, 0.0, 1.0], 0.0, 1.0).unwrap();
        assert!(r.apply(&c).is_err());
    }

    // --- Triple scoring ---

    #[test]
    fn score_triple_perfect_match() {
        let h = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap();
        let r = SphericalCapRelation::identity(3);
        let t = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap();
        let s = score_triple(&h, &r, &t).unwrap();
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn score_triple_mismatch() {
        let h = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.1).unwrap();
        let r = SphericalCapRelation::identity(3);
        let t = SphericalCap::new(vec![-1.0, 0.0, 0.0], 0.1).unwrap();
        let s = score_triple(&h, &r, &t).unwrap();
        assert!(s > 2.0, "mismatch score = {s}, expected > 2");
    }

    // --- Embedding model ---

    #[test]
    fn embedding_model_construction() {
        let entities = vec![
            SphericalCap::new(vec![1.0, 0.0, 0.0], 0.5).unwrap(),
            SphericalCap::new(vec![0.0, 1.0, 0.0], 0.3).unwrap(),
        ];
        let relations = vec![SphericalCapRelation::identity(3)];
        let model = SphericalCapEmbedding::new(entities, relations, 3).unwrap();
        assert_eq!(model.num_entities(), 2);
        assert_eq!(model.num_relations(), 1);
        assert_eq!(model.dim(), 3);
    }

    #[test]
    fn embedding_model_rejects_dim_mismatch() {
        let entities = vec![SphericalCap::new(vec![1.0, 0.0], 0.5).unwrap()];
        let relations = vec![SphericalCapRelation::identity(3)];
        assert!(SphericalCapEmbedding::new(entities, relations, 2).is_err());
    }

    // --- Geodesic distance ---

    #[test]
    fn geodesic_identical_is_zero() {
        let a = [1.0, 0.0, 0.0];
        let d = geodesic_distance(&a, &a);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn geodesic_orthogonal_is_pi_half() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let d = geodesic_distance(&a, &b);
        assert!((d - PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn geodesic_opposite_is_pi() {
        let a = [1.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0];
        let d = geodesic_distance(&a, &b);
        assert!((d - PI).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_large_positive() {
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sigmoid_large_negative() {
        assert!(sigmoid(-100.0).abs() < 1e-4);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::f32::consts::PI;

    fn arb_cap(dim: usize) -> impl Strategy<Value = SphericalCap> {
        (
            prop::collection::vec(-10.0f32..10.0, dim),
            0.01f32..(PI - 0.01),
        )
            .prop_filter_map("valid cap", move |(center, theta)| {
                SphericalCap::new(center, theta).ok()
            })
    }

    fn arb_cap_pair(dim: usize) -> impl Strategy<Value = (SphericalCap, SphericalCap)> {
        (arb_cap(dim), arb_cap(dim))
    }

    fn arb_relation(dim: usize) -> impl Strategy<Value = SphericalCapRelation> {
        (
            prop::collection::vec(-5.0f32..5.0, dim),
            -PI..PI,
            0.1f32..5.0,
        )
            .prop_filter_map("valid relation", move |(axis, angle, scale)| {
                SphericalCapRelation::new(axis, angle, scale).ok()
            })
    }

    proptest! {
        #[test]
        fn prop_containment_in_unit_interval(
            (a, b) in arb_cap_pair(4)
        ) {
            let p = containment_prob(&a, &b, 10.0).unwrap();
            prop_assert!(p >= -1e-6, "containment_prob < 0: {p}");
            prop_assert!(p <= 1.0 + 1e-6, "containment_prob > 1: {p}");
        }

        #[test]
        fn prop_surface_distance_nonneg(
            (a, b) in arb_cap_pair(4)
        ) {
            let d = surface_distance(&a, &b).unwrap();
            prop_assert!(d >= -1e-6, "surface_distance < 0: {d}");
        }

        #[test]
        fn prop_overlap_in_unit_interval(
            (a, b) in arb_cap_pair(4)
        ) {
            let p = overlap_prob(&a, &b).unwrap();
            prop_assert!(p >= -1e-6, "overlap_prob < 0: {p}");
            prop_assert!(p <= 1.0 + 1e-6, "overlap_prob > 1: {p}");
        }

        #[test]
        fn prop_overlap_symmetric(
            (a, b) in arb_cap_pair(4)
        ) {
            let p_ab = overlap_prob(&a, &b).unwrap();
            let p_ba = overlap_prob(&b, &a).unwrap();
            prop_assert!(
                (p_ab - p_ba).abs() < 1e-5,
                "overlap should be symmetric: {p_ab} != {p_ba}"
            );
        }

        #[test]
        fn prop_surface_distance_symmetric(
            (a, b) in arb_cap_pair(4)
        ) {
            let d_ab = surface_distance(&a, &b).unwrap();
            let d_ba = surface_distance(&b, &a).unwrap();
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-3,
                "surface_distance should be symmetric: {d_ab} != {d_ba}"
            );
        }

        #[test]
        fn prop_relation_apply_preserves_dim(
            c in arb_cap(4),
            r in arb_relation(4)
        ) {
            let t = r.apply(&c).unwrap();
            prop_assert_eq!(t.dim(), c.dim());
            prop_assert!(t.angular_radius() > 0.0, "radius should be positive");
            prop_assert!(t.angular_radius() <= PI, "radius should be <= pi");
        }

        #[test]
        fn prop_score_triple_nonneg(
            h in arb_cap(4),
            r in arb_relation(4),
            t in arb_cap(4)
        ) {
            let s = score_triple(&h, &r, &t).unwrap();
            prop_assert!(s >= -1e-6, "score_triple < 0: {s}");
        }

        #[test]
        fn prop_area_fraction_in_unit_interval(
            c in arb_cap(4)
        ) {
            let f = c.area_fraction();
            prop_assert!(f >= 0.0, "area_fraction < 0: {f}");
            prop_assert!(f <= 1.0 + 1e-5, "area_fraction > 1: {f}");
        }

        #[test]
        fn prop_center_is_unit_vector(
            c in arb_cap(4)
        ) {
            let norm: f32 = c.center().iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((norm - 1.0).abs() < 1e-5, "center not unit: {norm}");
        }

        #[test]
        fn prop_geodesic_in_range(
            (a, b) in arb_cap_pair(4)
        ) {
            let d = geodesic_distance(a.center(), b.center());
            prop_assert!(d >= 0.0, "geodesic < 0: {d}");
            prop_assert!(d <= PI + 1e-5, "geodesic > pi: {d}");
        }
    }
}
