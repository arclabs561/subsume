//! Ball embeddings for knowledge graph completion and taxonomy expansion.
//!
//! Each concept is represented as a solid ball in Euclidean space, parameterized
//! by a center vector and a radius. Subsumption is modeled via geometric containment:
//! ball A is contained in ball B iff `||c_A - c_B|| + r_A <= r_B`.
//!
//! # Motivation
//!
//! Balls are the simplest region beyond boxes: they require only `d + 1` parameters
//! (center + single radius) versus `2d` for boxes. This makes them more parameter-efficient
//! while still encoding hierarchy through containment.
//!
//! Two key papers use ball embeddings:
//!
//! - **SpherE** (SIGIR 2024, [arXiv:2404.19130](https://arxiv.org/abs/2404.19130)):
//!   Balls with relation transforms (translate + scale) for set retrieval.
//! - **RegD** (Jan 2025, rev Oct 2025, [arXiv:2501.17518](https://arxiv.org/abs/2501.17518)):
//!   Proves that balls in R^d with depth dissimilarity are bijectively isometric
//!   to points in hyperbolic space H^{d+1}.
//!
//! # Containment formula
//!
//! ```text
//! ball_A ⊆ ball_B  ⟺  ||c_A - c_B|| + r_A <= r_B
//! ```
//!
//! For soft containment (differentiable scoring):
//!
//! ```text
//! P(A ⊆ B) = sigmoid(k * (r_B - ||c_A - c_B|| - r_A))
//! ```
//!
//! where `k` is a sharpness parameter (higher = harder boundary).
//!
//! # Distance formula
//!
//! The minimum distance between two ball surfaces:
//!
//! ```text
//! d(ball_A, ball_B) = max(0, ||c_A - c_B|| - r_A - r_B)
//! ```
//!
//! Returns 0 when balls overlap, positive gap when they are disjoint.
//!
//! # Volume
//!
//! The volume of a d-dimensional ball of radius r:
//!
//! ```text
//! V_d(r) = (pi^(d/2) / Gamma(d/2 + 1)) * r^d
//! ```
//!
//! Log-volume (numerically stable):
//!
//! ```text
//! ln(V_d(r)) = (d/2) * ln(pi) - ln(Gamma(d/2 + 1)) + d * ln(r)
//! ```

use crate::BoxError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A solid ball in d-dimensional Euclidean space.
///
/// Parameterized by center `c ∈ R^d` and radius `r > 0`.
///
/// # Containment
///
/// Ball A is contained in ball B iff `||c_A - c_B|| + r_A <= r_B`.
/// This directly models subsumption: if concept A's ball is inside concept B's
/// ball, then B subsumes A (B is more general).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ball {
    center: Vec<f32>,
    radius: f32,
}

/// A relation transform for balls (SpherE-style).
///
/// Transforms a ball by translating its center and scaling its radius:
///
/// ```text
/// transform(ball, relation) = Ball(
///     center = ball.center + relation.translation,
///     radius = ball.radius * relation.scale,
/// )
/// ```
///
/// This enables modeling many-to-many relations where the same entity
/// participates with different granularities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BallRelation {
    translation: Vec<f32>,
    scale: f32,
}

/// Collection of ball embeddings with optional relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BallEmbedding {
    entities: Vec<Ball>,
    relations: Vec<BallRelation>,
    dim: usize,
}

// ---------------------------------------------------------------------------
// Ball construction and accessors
// ---------------------------------------------------------------------------

impl Ball {
    /// Create a new ball from center coordinates and radius.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if radius is non-positive or non-finite,
    /// or if any center coordinate is non-finite.
    pub fn new(center: Vec<f32>, radius: f32) -> Result<Self, BoxError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: radius as f64,
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
        Ok(Self { center, radius })
    }

    /// Create a ball from center and log-radius (for training).
    ///
    /// The radius is computed as `exp(log_radius)`, ensuring positivity.
    pub fn from_log_radius(center: Vec<f32>, log_radius: f32) -> Result<Self, BoxError> {
        if !log_radius.is_finite() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: log_radius as f64,
                max: log_radius as f64,
            });
        }
        Self::new(center, log_radius.exp())
    }

    /// Dimensionality of the ambient space.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.center.len()
    }

    /// Returns a reference to the center coordinates.
    pub fn center(&self) -> &[f32] {
        &self.center
    }

    /// Radius of the ball.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Log-radius (for training stability).
    #[must_use]
    pub fn log_radius(&self) -> f32 {
        self.radius.ln()
    }

    /// Log-volume of the ball in d dimensions.
    ///
    /// ```text
    /// ln(V_d) = (d/2) * ln(pi) - ln(Gamma(d/2 + 1)) + d * ln(r)
    /// ```
    ///
    /// Uses `lgamma` for numerical stability in high dimensions.
    #[must_use]
    pub fn log_volume(&self) -> f32 {
        let d = self.center.len() as f32;
        let log_pi_term = 0.5 * d * std::f32::consts::PI.ln();
        let log_gamma = lgamma(d / 2.0 + 1.0);
        let log_r_term = d * self.radius.ln();
        log_pi_term - log_gamma + log_r_term
    }

    /// Mutable access to center coordinates (for training).
    pub fn center_mut(&mut self) -> &mut [f32] {
        &mut self.center
    }

    /// Set the radius from a log-radius value (for training).
    pub fn set_log_radius(&mut self, log_radius: f32) {
        self.radius = log_radius.clamp(-10.0, 5.0).exp();
    }
}

// ---------------------------------------------------------------------------
// BallRelation
// ---------------------------------------------------------------------------

impl BallRelation {
    /// Create a new ball relation.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if scale is non-positive or non-finite,
    /// or if any translation component is non-finite.
    pub fn new(translation: Vec<f32>, scale: f32) -> Result<Self, BoxError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: scale as f64,
            });
        }
        for (i, &t) in translation.iter().enumerate() {
            if !t.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: t as f64,
                    max: t as f64,
                });
            }
        }
        Ok(Self { translation, scale })
    }

    /// Create the identity relation (zero translation, unit scale).
    #[must_use]
    pub fn identity(dim: usize) -> Self {
        Self {
            translation: vec![0.0; dim],
            scale: 1.0,
        }
    }

    /// Apply this relation to a ball, returning the transformed ball.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if dimensions disagree.
    pub fn apply(&self, ball: &Ball) -> Result<Ball, BoxError> {
        if self.translation.len() != ball.center.len() {
            return Err(BoxError::DimensionMismatch {
                expected: ball.center.len(),
                actual: self.translation.len(),
            });
        }
        let new_center: Vec<f32> = ball
            .center
            .iter()
            .zip(self.translation.iter())
            .map(|(&c, &t)| c + t)
            .collect();
        let new_radius = ball.radius * self.scale;
        Ball::new(new_center, new_radius)
    }

    /// Mutable access to translation (for training).
    pub fn translation_mut(&mut self) -> &mut [f32] {
        &mut self.translation
    }

    /// Returns a reference to the translation vector.
    pub fn translation(&self) -> &[f32] {
        &self.translation
    }

    /// Returns the scale factor.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Log-scale (for training stability).
    #[must_use]
    pub fn log_scale(&self) -> f32 {
        self.scale.ln()
    }

    /// Set the scale from a log-scale value (for training).
    pub fn set_log_scale(&mut self, log_scale: f32) {
        self.scale = log_scale.clamp(-5.0, 5.0).exp();
    }
}

// ---------------------------------------------------------------------------
// BallEmbedding
// ---------------------------------------------------------------------------

impl BallEmbedding {
    /// Create a new ball embedding model.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any entity or relation
    /// has a dimension different from `dim`.
    pub fn new(
        entities: Vec<Ball>,
        relations: Vec<BallRelation>,
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
            if r.translation.len() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: r.translation.len(),
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

    /// Number of entities.
    #[must_use]
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    /// Number of relations.
    #[must_use]
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }

    /// Returns a reference to entity embeddings.
    pub fn entities(&self) -> &[Ball] {
        &self.entities
    }

    /// Returns a reference to relation embeddings.
    pub fn relations(&self) -> &[BallRelation] {
        &self.relations
    }
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// Euclidean distance between two center vectors.
fn center_distance(a: &Ball, b: &Ball) -> f32 {
    debug_assert_eq!(a.center.len(), b.center.len());
    a.center
        .iter()
        .zip(b.center.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Soft containment probability: P(inner ⊆ outer).
///
/// ```text
/// P = sigmoid(k * (r_outer - ||c_inner - c_outer|| - r_inner))
/// ```
///
/// Returns a value in (0, 1). Values near 1.0 indicate strong containment;
/// values near 0.0 indicate the inner ball is mostly outside the outer ball.
///
/// The sharpness parameter `k` controls how hard the boundary is:
/// - `k = 1`: soft boundary, gradual transition
/// - `k = 10`: moderate boundary
/// - `k = 100`: near-hard boundary (approaches step function)
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the balls differ in dimension.
pub fn containment_prob(inner: &Ball, outer: &Ball, k: f32) -> Result<f32, BoxError> {
    if inner.dim() != outer.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: inner.dim(),
            actual: outer.dim(),
        });
    }
    let dist = center_distance(inner, outer);
    let margin = outer.radius - dist - inner.radius;
    Ok(sigmoid(k * margin))
}

/// Surface distance between two balls.
///
/// ```text
/// d = max(0, ||c_A - c_B|| - r_A - r_B)
/// ```
///
/// Returns 0 when balls overlap, positive gap when disjoint.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the balls differ in dimension.
pub fn surface_distance(a: &Ball, b: &Ball) -> Result<f32, BoxError> {
    if a.dim() != b.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim(),
            actual: b.dim(),
        });
    }
    let dist = center_distance(a, b);
    Ok((dist - a.radius - b.radius).max(0.0))
}

/// Overlap probability (Jaccard-style): Vol(intersection) / Vol(union).
///
/// For balls, the intersection volume has a closed form in any dimension
/// using the regularized incomplete beta function. For computational
/// efficiency, we use a proxy based on the overlap depth:
///
/// ```text
/// overlap = max(0, r_A + r_B - ||c_A - c_B||) / (r_A + r_B)
/// ```
///
/// This is 1 when one ball contains the other, 0 when they are disjoint,
/// and linearly interpolates in between.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the balls differ in dimension.
pub fn overlap_prob(a: &Ball, b: &Ball) -> Result<f32, BoxError> {
    if a.dim() != b.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim(),
            actual: b.dim(),
        });
    }
    let dist = center_distance(a, b);
    let sum_r = a.radius + b.radius;
    if sum_r < 1e-12 {
        return Ok(0.0);
    }
    let overlap_depth = (sum_r - dist).max(0.0);
    Ok(overlap_depth / sum_r)
}

/// Score a triple (head, relation, tail) using ball embeddings (SpherE-style).
///
/// Transforms the head ball by the relation, then computes the surface
/// distance to the tail ball. Lower scores indicate better matches.
///
/// ```text
/// score = surface_distance(transform(head, relation), tail)
/// ```
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if any component has mismatched dimensions.
pub fn score_triple(head: &Ball, relation: &BallRelation, tail: &Ball) -> Result<f32, BoxError> {
    let transformed = relation.apply(head)?;
    surface_distance(&transformed, tail)
}

// ---------------------------------------------------------------------------
// RegD scoring (Yang & Chen, 2025, arXiv:2501.17518)
// ---------------------------------------------------------------------------

/// RegD depth dissimilarity for balls.
///
/// ```text
/// d_dep(A, B) = (||c_A - c_B||^p + |r_A - r_B|^p) / (r_A * r_B)
/// ```
///
/// With `p = 2` and `g(x) = arcosh(x + 1)`, this is a bijective isometry
/// to hyperbolic distance in H^{n+1} (curvature -1). With `g(x) = x`,
/// it preserves the order of all hyperbolic distances.
///
/// As a region shrinks toward the empty set (r → 0), the dissimilarity
/// diverges to infinity — mimicking hyperbolic boundary divergence.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the balls differ in dimension.
pub fn regd_depth_dissimilarity(a: &Ball, b: &Ball, p: f32) -> Result<f32, BoxError> {
    if a.dim() != b.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim(),
            actual: b.dim(),
        });
    }
    let center_dist = center_distance(a, b).powf(p);
    let radius_diff = (a.radius - b.radius).abs().powf(p);
    Ok((center_dist + radius_diff) / (a.radius * b.radius))
}

/// RegD boundary dissimilarity for balls (cone-based, RegD Eq. 6).
///
/// ```text
/// d_bd(inner, outer) = arcsinh((||c_inner - c_outer|| - r_inner) / r_outer) + arcsinh(1)
/// ```
///
/// Measures the minimal translation cost to move the inner ball out of
/// the outer ball. Zero when the inner ball is tangent to the outer
/// boundary, positive when there is a gap, negative when the inner
/// ball is strictly inside.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the balls differ in dimension.
pub fn regd_boundary_dissimilarity(inner: &Ball, outer: &Ball) -> Result<f32, BoxError> {
    if inner.dim() != outer.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: inner.dim(),
            actual: outer.dim(),
        });
    }
    let dist = center_distance(inner, outer);
    let gap = dist - inner.radius;
    Ok((gap / outer.radius).asinh() + 1.0_f32.asinh())
}

/// Combined RegD score for containment ranking.
///
/// ```text
/// score = alpha * d_dep(outer, inner) + (1 - alpha) * d_bd(inner, outer)
/// ```
///
/// Lower scores indicate stronger containment. The `alpha` parameter
/// weights depth dissimilarity (hierarchical separation) vs boundary
/// dissimilarity (containment tightness).
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the balls differ in dimension.
pub fn regd_score(outer: &Ball, inner: &Ball, alpha: f32, p: f32) -> Result<f32, BoxError> {
    let depth = regd_depth_dissimilarity(outer, inner, p)?;
    let boundary = regd_boundary_dissimilarity(inner, outer)?;
    Ok(alpha * depth + (1.0 - alpha) * boundary)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Numerically stable sigmoid function.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Natural log of the Gamma function.
///
/// Uses the Lanczos approximation for accuracy across the full range.
fn lgamma(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NAN;
    }
    const G: f32 = 7.0;
    const COEFF: [f32; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        std::f32::consts::PI.ln() - (std::f32::consts::PI * x).sin().ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEFF[0];
        for i in 1..COEFF.len() {
            a += COEFF[i] / (x + i as f32);
        }
        let tmp = x + G + 0.5;
        (2.0 * std::f32::consts::PI).sqrt().ln() + (x + 0.5) * tmp.ln() - tmp + a.ln()
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
    fn ball_new_valid() {
        let b = Ball::new(vec![0.0, 0.0, 0.0], 1.0).unwrap();
        assert_eq!(b.dim(), 3);
        assert_eq!(b.radius(), 1.0);
    }

    #[test]
    fn ball_from_log_radius() {
        let b = Ball::from_log_radius(vec![0.0, 0.0], 0.0).unwrap();
        assert!((b.radius() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ball_rejects_zero_radius() {
        assert!(Ball::new(vec![0.0], 0.0).is_err());
    }

    #[test]
    fn ball_rejects_negative_radius() {
        assert!(Ball::new(vec![0.0], -1.0).is_err());
    }

    #[test]
    fn ball_rejects_non_finite_center() {
        assert!(Ball::new(vec![f32::NAN], 1.0).is_err());
        assert!(Ball::new(vec![f32::INFINITY], 1.0).is_err());
    }

    #[test]
    fn ball_rejects_non_finite_radius() {
        assert!(Ball::new(vec![0.0], f32::NAN).is_err());
        assert!(Ball::new(vec![0.0], f32::INFINITY).is_err());
    }

    #[test]
    fn containment_identical_is_half() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let p = containment_prob(&a, &a, 10.0).unwrap();
        assert!((p - 0.5).abs() < 1e-4);
    }

    #[test]
    fn containment_nested_is_near_one() {
        let inner = Ball::new(vec![0.0, 0.0], 0.5).unwrap();
        let outer = Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let p = containment_prob(&inner, &outer, 10.0).unwrap();
        assert!(p > 0.99, "nested containment = {p}, expected > 0.99");
    }

    #[test]
    fn containment_disjoint_is_near_zero() {
        let a = Ball::new(vec![0.0, 0.0], 0.5).unwrap();
        let b = Ball::new(vec![10.0, 0.0], 0.5).unwrap();
        let p = containment_prob(&a, &b, 10.0).unwrap();
        assert!(p < 1e-4, "disjoint containment = {p}, expected ~0");
    }

    #[test]
    fn containment_tangent() {
        let inner = Ball::new(vec![1.0, 0.0], 1.0).unwrap();
        let outer = Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let p = containment_prob(&inner, &outer, 10.0).unwrap();
        assert!((p - 0.5).abs() < 1e-4);
    }

    #[test]
    fn containment_dimension_mismatch() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let b = Ball::new(vec![0.0], 1.0).unwrap();
        assert!(containment_prob(&a, &b, 1.0).is_err());
    }

    #[test]
    fn surface_distance_overlapping_is_zero() {
        let a = Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let b = Ball::new(vec![1.0, 0.0], 2.0).unwrap();
        let d = surface_distance(&a, &b).unwrap();
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn surface_distance_disjoint_is_positive() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let b = Ball::new(vec![5.0, 0.0], 1.0).unwrap();
        let d = surface_distance(&a, &b).unwrap();
        assert!((d - 3.0).abs() < 1e-5);
    }

    #[test]
    fn surface_distance_tangent_is_zero() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let b = Ball::new(vec![2.0, 0.0], 1.0).unwrap();
        let d = surface_distance(&a, &b).unwrap();
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn surface_distance_identical_is_zero() {
        let a = Ball::new(vec![1.0, 2.0], 3.0).unwrap();
        let d = surface_distance(&a, &a).unwrap();
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn overlap_identical_is_one() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let p = overlap_prob(&a, &a).unwrap();
        assert!((p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn overlap_disjoint_is_zero() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let b = Ball::new(vec![10.0, 0.0], 1.0).unwrap();
        let p = overlap_prob(&a, &b).unwrap();
        assert!(p.abs() < 1e-6);
    }

    #[test]
    fn overlap_symmetric() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let b = Ball::new(vec![1.5, 0.0], 2.0).unwrap();
        let p_ab = overlap_prob(&a, &b).unwrap();
        let p_ba = overlap_prob(&b, &a).unwrap();
        assert!((p_ab - p_ba).abs() < 1e-6);
    }

    #[test]
    fn log_volume_2d_unit_circle() {
        let b = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let lv = b.log_volume();
        assert!((lv - PI.ln()).abs() < 1e-4);
    }

    #[test]
    fn log_volume_3d_unit_sphere() {
        let b = Ball::new(vec![0.0, 0.0, 0.0], 1.0).unwrap();
        let lv = b.log_volume();
        let expected = ((4.0 / 3.0) * PI).ln();
        assert!((lv - expected).abs() < 1e-4);
    }

    #[test]
    fn log_volume_scales_with_radius() {
        let d = 5;
        let center = vec![0.0; d];
        let b1 = Ball::new(center.clone(), 1.0).unwrap();
        let b2 = Ball::new(center, 2.0).unwrap();
        let diff = b2.log_volume() - b1.log_volume();
        let expected = d as f32 * 2.0f32.ln();
        assert!((diff - expected).abs() < 1e-4);
    }

    #[test]
    fn relation_identity_preserves_ball() {
        let b = Ball::new(vec![1.0, 2.0, 3.0], 0.5).unwrap();
        let r = BallRelation::identity(3);
        let t = r.apply(&b).unwrap();
        for (i, (&a, &tr)) in b.center().iter().zip(t.center().iter()).enumerate() {
            assert!((a - tr).abs() < 1e-6, "center[{i}] changed: {a} -> {tr}");
        }
        assert!((t.radius() - b.radius()).abs() < 1e-6);
    }

    #[test]
    fn relation_translation() {
        let b = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let r = BallRelation::new(vec![3.0, 4.0], 1.0).unwrap();
        let t = r.apply(&b).unwrap();
        assert!((t.center()[0] - 3.0).abs() < 1e-6);
        assert!((t.center()[1] - 4.0).abs() < 1e-6);
        assert!((t.radius() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn relation_scaling() {
        let b = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let r = BallRelation::new(vec![0.0, 0.0], 2.5).unwrap();
        let t = r.apply(&b).unwrap();
        assert!((t.radius() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn relation_rejects_zero_scale() {
        assert!(BallRelation::new(vec![0.0], 0.0).is_err());
        assert!(BallRelation::new(vec![0.0], -1.0).is_err());
    }

    #[test]
    fn relation_dimension_mismatch() {
        let b = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let r = BallRelation::new(vec![0.0], 1.0).unwrap();
        assert!(r.apply(&b).is_err());
    }

    #[test]
    fn score_triple_perfect_match() {
        let h = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let r = BallRelation::identity(2);
        let t = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let s = score_triple(&h, &r, &t).unwrap();
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn score_triple_mismatch() {
        let h = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let r = BallRelation::identity(2);
        let t = Ball::new(vec![5.0, 0.0], 1.0).unwrap();
        let s = score_triple(&h, &r, &t).unwrap();
        assert!((s - 3.0).abs() < 1e-5);
    }

    #[test]
    fn embedding_model_construction() {
        let entities = vec![
            Ball::new(vec![0.0, 0.0], 1.0).unwrap(),
            Ball::new(vec![1.0, 1.0], 0.5).unwrap(),
        ];
        let relations = vec![BallRelation::identity(2)];
        let model = BallEmbedding::new(entities, relations, 2).unwrap();
        assert_eq!(model.num_entities(), 2);
        assert_eq!(model.num_relations(), 1);
        assert_eq!(model.dim(), 2);
    }

    #[test]
    fn embedding_model_rejects_dim_mismatch() {
        let entities = vec![Ball::new(vec![0.0, 0.0], 1.0).unwrap()];
        let relations = vec![BallRelation::identity(3)];
        assert!(BallEmbedding::new(entities, relations, 2).is_err());
    }

    #[test]
    fn sigmoid_large_positive() {
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sigmoid_large_negative() {
        assert!(sigmoid(-100.0).abs() < 1e-4);
    }

    #[test]
    fn sigmoid_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    // --- RegD tests ---

    #[test]
    fn regd_depth_identical() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let d = regd_depth_dissimilarity(&a, &a, 2.0).unwrap();
        // center_dist=0, radius_diff=0, so d=0
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn regd_depth_same_center_different_radius() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let b = Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let d = regd_depth_dissimilarity(&a, &b, 2.0).unwrap();
        // center_dist=0, |1-2|^2=1, denominator=2, so d=0.5
        assert!((d - 0.5).abs() < 1e-5);
    }

    #[test]
    fn regd_depth_diverges_as_radius_shrinks() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let b = Ball::new(vec![0.0, 0.0], 0.001).unwrap();
        let d = regd_depth_dissimilarity(&a, &b, 2.0).unwrap();
        // denominator = 0.001, so d should be large
        assert!(d > 100.0, "depth dissimilarity should diverge, got {d}");
    }

    #[test]
    fn regd_boundary_tangent_is_asinh_one() {
        // inner tangent to outer: ||c_i - c_o|| = r_o - r_i
        let inner = Ball::new(vec![1.0, 0.0], 1.0).unwrap();
        let outer = Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let d = regd_boundary_dissimilarity(&inner, &outer).unwrap();
        // gap = 1 - 1 = 0, so d = asinh(0/2) + asinh(1) = asinh(1)
        assert!((d - 1.0_f32.asinh()).abs() < 1e-5);
    }

    #[test]
    fn regd_boundary_nested_is_negative() {
        // inner strictly inside outer: ||c_i - c_o|| < r_o - r_i
        let inner = Ball::new(vec![0.0, 0.0], 0.5).unwrap();
        let outer = Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let d = regd_boundary_dissimilarity(&inner, &outer).unwrap();
        // gap = 0 - 0.5 = -0.5, so d = asinh(-0.5/2) + asinh(1)
        let expected = (-0.25_f32).asinh() + 1.0_f32.asinh();
        assert!((d - expected).abs() < 1e-5);
    }

    #[test]
    fn regd_boundary_disjoint_is_large() {
        let inner = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let outer = Ball::new(vec![10.0, 0.0], 1.0).unwrap();
        let d = regd_boundary_dissimilarity(&inner, &outer).unwrap();
        // gap = 10 - 1 = 9, so d = asinh(9/1) + asinh(1) ≈ 2.89 + 0.88 ≈ 3.77
        assert!(
            d > 3.0,
            "disjoint boundary dissimilarity = {d}, expected > 3"
        );
    }

    #[test]
    fn regd_combined_dimension_mismatch() {
        let a = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let b = Ball::new(vec![0.0], 1.0).unwrap();
        assert!(regd_depth_dissimilarity(&a, &b, 2.0).is_err());
        assert!(regd_boundary_dissimilarity(&a, &b).is_err());
        assert!(regd_score(&a, &b, 0.5, 2.0).is_err());
    }

    #[test]
    fn regd_score_nested_is_low() {
        let outer = Ball::new(vec![0.0, 0.0], 5.0).unwrap();
        let inner = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let s = regd_score(&outer, &inner, 0.5, 2.0).unwrap();
        // both depth and boundary should be small for nested balls
        assert!(s < 5.0, "nested regd_score = {s}, expected small");
    }

    #[test]
    fn regd_score_disjoint_is_high() {
        let outer = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let inner = Ball::new(vec![20.0, 0.0], 1.0).unwrap();
        let s = regd_score(&outer, &inner, 0.5, 2.0).unwrap();
        // both depth and boundary should be large for disjoint balls
        assert!(s > 100.0, "disjoint regd_score = {s}, expected large");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_ball(dim: usize) -> impl Strategy<Value = Ball> {
        (prop::collection::vec(-100.0f32..100.0, dim), -5.0f32..5.0)
            .prop_filter_map("valid ball", move |(center, log_r)| {
                Ball::from_log_radius(center, log_r).ok()
            })
    }

    fn arb_ball_pair(dim: usize) -> impl Strategy<Value = (Ball, Ball)> {
        (arb_ball(dim), arb_ball(dim))
    }

    fn arb_relation(dim: usize) -> impl Strategy<Value = BallRelation> {
        (prop::collection::vec(-10.0f32..10.0, dim), -2.0f32..2.0)
            .prop_filter_map("valid relation", move |(t, log_s)| {
                BallRelation::new(t, log_s.exp()).ok()
            })
    }

    proptest! {
        #[test]
        fn prop_containment_in_unit_interval(
            (a, b) in arb_ball_pair(4)
        ) {
            let p = containment_prob(&a, &b, 10.0).unwrap();
            prop_assert!(p >= -1e-6, "containment_prob < 0: {p}");
            prop_assert!(p <= 1.0 + 1e-6, "containment_prob > 1: {p}");
        }

        #[test]
        fn prop_surface_distance_nonneg(
            (a, b) in arb_ball_pair(4)
        ) {
            let d = surface_distance(&a, &b).unwrap();
            prop_assert!(d >= -1e-6, "surface_distance < 0: {d}");
        }

        #[test]
        fn prop_overlap_in_unit_interval(
            (a, b) in arb_ball_pair(4)
        ) {
            let p = overlap_prob(&a, &b).unwrap();
            prop_assert!(p >= -1e-6, "overlap_prob < 0: {p}");
            prop_assert!(p <= 1.0 + 1e-6, "overlap_prob > 1: {p}");
        }

        #[test]
        fn prop_overlap_symmetric(
            (a, b) in arb_ball_pair(4)
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
            (a, b) in arb_ball_pair(4)
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
            b in arb_ball(4),
            r in arb_relation(4)
        ) {
            let t = r.apply(&b).unwrap();
            prop_assert_eq!(t.dim(), b.dim());
            prop_assert!(t.radius() > 0.0, "radius should be positive: {}", t.radius());
        }

        #[test]
        fn prop_score_triple_nonneg(
            h in arb_ball(4),
            r in arb_relation(4),
            t in arb_ball(4)
        ) {
            let s = score_triple(&h, &r, &t).unwrap();
            prop_assert!(s >= -1e-6, "score_triple < 0: {s}");
        }

        #[test]
        fn prop_log_volume_finite(
            b in arb_ball(8)
        ) {
            let lv = b.log_volume();
            prop_assert!(lv.is_finite(), "log_volume not finite: {lv}");
        }

        #[test]
        fn prop_containment_sharper_boundary(
            inner in arb_ball(4),
            outer in arb_ball(4)
        ) {
            let p_soft = containment_prob(&inner, &outer, 1.0).unwrap();
            let p_hard = containment_prob(&inner, &outer, 100.0).unwrap();
            let margin = outer.radius() - center_distance(&inner, &outer) - inner.radius();
            if margin.abs() > 0.1 {
                let dist_soft = if margin > 0.0 { 1.0 - p_soft } else { p_soft };
                let dist_hard = if margin > 0.0 { 1.0 - p_hard } else { p_hard };
                prop_assert!(
                    dist_hard <= dist_soft + 1e-4,
                    "sharper k should push closer to boundary: soft={p_soft}, hard={p_hard}"
                );
            }
        }

        // --- RegD proptests ---

        #[test]
        fn prop_regd_depth_nonneg(
            (a, b) in arb_ball_pair(4)
        ) {
            let d = regd_depth_dissimilarity(&a, &b, 2.0).unwrap();
            prop_assert!(d >= -1e-5, "regd_depth < 0: {d}");
        }

        #[test]
        fn prop_regd_depth_symmetric(
            (a, b) in arb_ball_pair(4)
        ) {
            let d_ab = regd_depth_dissimilarity(&a, &b, 2.0).unwrap();
            let d_ba = regd_depth_dissimilarity(&b, &a, 2.0).unwrap();
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-3,
                "regd_depth should be symmetric: {d_ab} != {d_ba}"
            );
        }

        #[test]
        fn prop_regd_depth_diverges_for_small_radius(
            a in arb_ball(4),
            log_r_small in -20.0f32..-10.0,
        ) {
            let center: Vec<f32> = (0..4).map(|i| a.center()[i]).collect();
            if let Ok(b) = Ball::from_log_radius(center, log_r_small) {
                let d = regd_depth_dissimilarity(&a, &b, 2.0).unwrap();
                prop_assert!(d > 100.0, "depth should diverge for small radius: {d}");
            }
        }

        #[test]
        fn prop_regd_boundary_finite(
            (a, b) in arb_ball_pair(4)
        ) {
            let d = regd_boundary_dissimilarity(&a, &b).unwrap();
            prop_assert!(d.is_finite(), "regd_boundary not finite: {d}");
        }

        #[test]
        fn prop_regd_score_finite(
            (a, b) in arb_ball_pair(4)
        ) {
            let s = regd_score(&a, &b, 0.5, 2.0).unwrap();
            prop_assert!(s.is_finite(), "regd_score not finite: {s}");
        }
    }
}
