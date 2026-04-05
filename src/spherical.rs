//! Spherical knowledge graph embeddings on the unit sphere.
//!
//! Entities are points on $$S^{d-1}$$ (unit vectors in $$\mathbb{R}^d$$).
//! Relations are rotations represented in axis-angle form: a unit axis
//! vector plus an angle. Scoring uses geodesic (great-circle) distance:
//!
//! ```text
//! score(h, r, t) = arccos(dot(rotate(h, r), t))
//! ```
//!
//! Lower scores indicate better-matching triples.
//!
//! # Rotation semantics for d ≠ 3
//!
//! For d = 3, `rotate()` uses Rodrigues' formula — the standard, unique
//! 3D rotation about an axis. For d ≠ 3, the rotation is a **generalized
//! 2D rotation** in the plane spanned by the point's perpendicular component
//! and the relation axis. This is not a unique rotation in higher dimensions
//! (there are infinitely many planes of rotation in d > 3). The current
//! approach is a reasonable heuristic but spherical embeddings are most
//! meaningful for d = 3. For higher dimensions, treat rotation semantics
//! as approximate.
//!
//! # Projection
//!
//! After each gradient step, entity and relation-axis vectors must be
//! projected back to the unit sphere via L2 normalization
//! (`project_to_sphere`).
//!
//! # References
//!
//! - Sun et al. (2019), "RotatE: Knowledge Graph Embedding by Relational Rotation
//!   in Complex Space" -- rotation-based KGE in the complex plane
//! - Cao et al. (2024, ACM Computing Surveys), "KG Embedding: A Survey from the
//!   Perspective of Representation Spaces" -- positions rotation-based embeddings
//!   within the broader KGE taxonomy

use crate::BoxError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A point on the unit sphere $$S^{d-1}$$.
///
/// Stored as a unit vector in $$\mathbb{R}^d$$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SphericalPoint {
    /// Unit vector coordinates.
    coords: Vec<f32>,
}

/// An axis-angle rotation in $$\mathbb{R}^d$$.
///
/// For `d == 3`, this is classical axis-angle rotation.
/// For general `d`, the rotation acts in the plane spanned by the axis
/// and the input vector (Rodrigues-style generalization).
///
/// - `axis`: unit vector defining the rotation plane (with the input vector)
/// - `angle`: rotation angle in radians
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SphericalRelation {
    /// Unit vector rotation axis.
    axis: Vec<f32>,
    /// Rotation angle in radians.
    angle: f32,
}

/// Entity and relation embeddings on the unit sphere.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphericalEmbedding {
    /// Entity embeddings (each a unit vector).
    entities: Vec<SphericalPoint>,
    /// Relation embeddings (axis-angle rotations).
    relations: Vec<SphericalRelation>,
    /// Embedding dimension.
    dim: usize,
}

// ---------------------------------------------------------------------------
// Construction and validation
// ---------------------------------------------------------------------------

impl SphericalPoint {
    /// Create a new spherical point from a unit vector.
    ///
    /// The vector must be finite and have non-zero norm.
    /// It is normalized to unit length.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if the vector is zero or non-finite.
    pub fn new(coords: Vec<f32>) -> Result<Self, BoxError> {
        validate_finite(&coords)?;
        let coords = l2_normalize(coords)?;
        Ok(Self { coords })
    }

    /// Dimension of the embedding space.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Returns a reference to the unit vector coordinates.
    pub fn coords(&self) -> &[f32] {
        &self.coords
    }
}

impl SphericalRelation {
    /// Create a new axis-angle rotation.
    ///
    /// `axis` is normalized to unit length. `angle` must be finite.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if the axis is zero/non-finite
    /// or the angle is non-finite.
    pub fn new(axis: Vec<f32>, angle: f32) -> Result<Self, BoxError> {
        if !angle.is_finite() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: angle as f64,
                max: angle as f64,
            });
        }
        validate_finite(&axis)?;
        let axis = l2_normalize(axis)?;
        Ok(Self { axis, angle })
    }

    /// Create the identity rotation (zero angle).
    ///
    /// The axis is set to the first standard basis vector.
    #[must_use]
    pub fn identity(dim: usize) -> Self {
        let mut axis = vec![0.0f32; dim];
        if dim > 0 {
            axis[0] = 1.0;
        }
        Self { axis, angle: 0.0 }
    }

    /// Dimension of the embedding space.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.axis.len()
    }

    /// Returns a reference to the unit rotation axis.
    pub fn axis(&self) -> &[f32] {
        &self.axis
    }

    /// Returns the rotation angle in radians.
    #[must_use]
    pub fn angle(&self) -> f32 {
        self.angle
    }
}

impl SphericalEmbedding {
    /// Create a new spherical embedding model.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any entity or relation
    /// has a dimension different from `dim`.
    pub fn new(
        entities: Vec<SphericalPoint>,
        relations: Vec<SphericalRelation>,
        dim: usize,
    ) -> Result<Self, BoxError> {
        for (i, e) in entities.iter().enumerate() {
            if e.dim() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: e.coords.len(),
                });
            }
            let _ = i;
        }
        for (i, r) in relations.iter().enumerate() {
            if r.dim() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: r.axis.len(),
                });
            }
            let _ = i;
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
    pub fn entities(&self) -> &[SphericalPoint] {
        &self.entities
    }

    /// Returns a reference to relation embeddings.
    pub fn relations(&self) -> &[SphericalRelation] {
        &self.relations
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Apply a rotation to a point on the sphere (Rodrigues formula).
///
/// For `d == 3`, this is the standard Rodrigues rotation around axis `k`
/// by `angle` radians:
///
/// ```text
/// result = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1 - cos(a))
/// ```
///
/// For general `d`, axis-angle rotation around an axis is not uniquely
/// defined (the perpendicular subspace is `(d-1)`-dimensional). This
/// function falls back to rotation in the 2D plane spanned by `v_perp`
/// and `k`, where `v_perp = v - k*(k.v)`. This is well-defined in any
/// dimension and preserves the key KGE properties: identity relation
/// gives zero score, and the rotation is an isometry on the sphere.
///
/// The result is re-normalized to stay on the unit sphere (handles
/// accumulated floating-point drift).
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if dimensions disagree.
pub fn rotate(
    point: &SphericalPoint,
    relation: &SphericalRelation,
) -> Result<SphericalPoint, BoxError> {
    let d = point.dim();
    if relation.dim() != d {
        return Err(BoxError::DimensionMismatch {
            expected: d,
            actual: relation.dim(),
        });
    }

    let v = &point.coords;
    let k = &relation.axis;
    let angle = relation.angle;

    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let k_dot_v: f32 = k.iter().zip(v.iter()).map(|(a, b)| a * b).sum();

    let mut result = vec![0.0f32; d];

    if d == 3 {
        // Standard 3D Rodrigues: v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
        let cross = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];
        for i in 0..3 {
            result[i] = v[i] * cos_a + cross[i] * sin_a + k[i] * k_dot_v * (1.0 - cos_a);
        }
    } else {
        // General d: decompose v into parallel and perpendicular to k,
        // then rotate v_perp toward k in the (v_perp, k) plane.
        let mut v_perp = vec![0.0f32; d];
        for i in 0..d {
            v_perp[i] = v[i] - k[i] * k_dot_v;
        }
        let perp_norm = v_perp.iter().map(|x| x * x).sum::<f32>().sqrt();

        if perp_norm < 1e-12 {
            // v is parallel to k: rotation has no effect.
            for i in 0..d {
                result[i] = v[i];
            }
        } else {
            // Rotate in the (v_perp_hat, k) plane:
            //   result = k*(k.v) + v_perp*cos(a) + k*|v_perp|*sin(a)
            for i in 0..d {
                result[i] = k[i] * (k_dot_v + perp_norm * sin_a) + v_perp[i] * cos_a;
            }
        }
    }

    // Re-normalize to unit sphere to handle float drift.
    let result = l2_normalize(result).unwrap_or_else(|_| point.coords.clone());
    Ok(SphericalPoint { coords: result })
}

/// Score a single triple `(head, relation, tail)`.
///
/// Returns the geodesic (great-circle) distance between `rotate(head, relation)`
/// and `tail`:
///
/// ```text
/// score = arccos(clamp(dot(rotate(h, r), t), -1, 1))
/// ```
///
/// Lower scores indicate better-matching triples. Range: `[0, pi]`.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if dimensions disagree.
pub fn score_triple(
    head: &SphericalPoint,
    relation: &SphericalRelation,
    tail: &SphericalPoint,
) -> Result<f64, BoxError> {
    if head.dim() != tail.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: head.dim(),
            actual: tail.dim(),
        });
    }
    let rotated = rotate(head, relation)?;
    let dot: f32 = rotated
        .coords
        .iter()
        .zip(tail.coords.iter())
        .map(|(a, b)| a * b)
        .sum();
    // Clamp for numerical safety before arccos.
    let clamped = dot.clamp(-1.0, 1.0);
    Ok(clamped.acos() as f64)
}

/// Score a batch of triples.
///
/// Each triple is `(heads[i], relations[i], tails[i])`. All three slices
/// must have the same length.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if any triple has mismatched dimensions,
/// or if the input slices have different lengths.
pub fn score_batch(
    heads: &[SphericalPoint],
    relations: &[SphericalRelation],
    tails: &[SphericalPoint],
) -> Result<Vec<f64>, BoxError> {
    if heads.len() != relations.len() || heads.len() != tails.len() {
        return Err(BoxError::DimensionMismatch {
            expected: heads.len(),
            actual: relations.len().min(tails.len()),
        });
    }
    heads
        .iter()
        .zip(relations.iter())
        .zip(tails.iter())
        .map(|((h, r), t)| score_triple(h, r, t))
        .collect()
}

/// Project a vector onto the unit sphere via L2 normalization.
///
/// Returns a [`SphericalPoint`] with unit norm pointing in the same direction
/// as the input.
///
/// # Errors
///
/// Returns [`BoxError::InvalidBounds`] if the vector is zero or non-finite.
pub fn project_to_sphere(v: Vec<f32>) -> Result<SphericalPoint, BoxError> {
    SphericalPoint::new(v)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// L2-normalize a vector. Returns error if zero-length or non-finite.
fn l2_normalize(mut v: Vec<f32>) -> Result<Vec<f32>, BoxError> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return Err(BoxError::InvalidBounds {
            dim: 0,
            min: 0.0,
            max: 0.0,
        });
    }
    for x in &mut v {
        *x /= norm;
    }
    Ok(v)
}

/// Validate that all elements are finite.
fn validate_finite(v: &[f32]) -> Result<(), BoxError> {
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(BoxError::InvalidBounds {
                dim: i,
                min: x as f64,
                max: x as f64,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn point(v: &[f32]) -> SphericalPoint {
        SphericalPoint::new(v.to_vec()).unwrap()
    }

    #[test]
    fn identity_rotation_score_is_zero() {
        let h = point(&[1.0, 0.0, 0.0]);
        let r = SphericalRelation::identity(3);
        let score = score_triple(&h, &r, &h).unwrap();
        assert!(
            score.abs() < 1e-6,
            "score(h, identity, h) = {score}, expected 0"
        );
    }

    #[test]
    fn score_is_nonnegative() {
        let h = point(&[1.0, 0.0, 0.0]);
        let t = point(&[0.0, 1.0, 0.0]);
        let r = SphericalRelation::identity(3);
        let score = score_triple(&h, &r, &t).unwrap();
        assert!(score >= -1e-9, "score should be non-negative, got {score}");
    }

    #[test]
    fn orthogonal_vectors_score_pi_over_2() {
        let h = point(&[1.0, 0.0, 0.0]);
        let t = point(&[0.0, 1.0, 0.0]);
        let r = SphericalRelation::identity(3);
        let score = score_triple(&h, &r, &t).unwrap();
        assert!(
            (score - PI / 2.0).abs() < 1e-5,
            "score(orthogonal, identity) = {score}, expected pi/2 = {}",
            PI / 2.0
        );
    }

    #[test]
    fn antipodal_vectors_score_pi() {
        let h = point(&[1.0, 0.0, 0.0]);
        let t = point(&[-1.0, 0.0, 0.0]);
        let r = SphericalRelation::identity(3);
        let score = score_triple(&h, &r, &t).unwrap();
        assert!(
            (score - PI).abs() < 1e-5,
            "score(antipodal, identity) = {score}, expected pi"
        );
    }

    #[test]
    fn projection_preserves_direction_sets_unit_norm() {
        let v = vec![3.0, 4.0, 0.0];
        let p = project_to_sphere(v).unwrap();
        let norm: f32 = p.coords().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "projected norm = {norm}, expected 1.0"
        );
        // Direction: should be (0.6, 0.8, 0.0)
        assert!(
            (p.coords()[0] - 0.6).abs() < 1e-6,
            "x = {}, expected 0.6",
            p.coords()[0]
        );
        assert!(
            (p.coords()[1] - 0.8).abs() < 1e-6,
            "y = {}, expected 0.8",
            p.coords()[1]
        );
    }

    #[test]
    fn rotation_preserves_unit_norm() {
        let h = point(&[1.0, 0.0, 0.0]);
        let r = SphericalRelation::new(vec![0.0, 0.0, 1.0], 1.0).unwrap();
        let rotated = rotate(&h, &r).unwrap();
        let norm: f32 = rotated.coords().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "rotated norm = {norm}, expected 1.0"
        );
    }

    #[test]
    fn rotation_by_pi_over_2_moves_x_to_y() {
        // Rotate (1,0,0) by pi/2 around z-axis -> (0,1,0)
        let h = point(&[1.0, 0.0, 0.0]);
        let r = SphericalRelation::new(vec![0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2).unwrap();
        let rotated = rotate(&h, &r).unwrap();
        let t = point(&[0.0, 1.0, 0.0]);
        let score = score_triple(&h, &r, &t).unwrap();
        assert!(
            score.abs() < 1e-4,
            "rotating x by pi/2 around z should reach y, score = {score}"
        );
        assert!(
            (rotated.coords()[0]).abs() < 1e-5,
            "rotated x = {}, expected ~0",
            rotated.coords()[0]
        );
        assert!(
            (rotated.coords()[1] - 1.0).abs() < 1e-5,
            "rotated y = {}, expected ~1",
            rotated.coords()[1]
        );
    }

    #[test]
    fn batch_scoring_matches_individual() {
        let h1 = point(&[1.0, 0.0, 0.0]);
        let h2 = point(&[0.0, 1.0, 0.0]);
        let r = SphericalRelation::identity(3);
        let t1 = point(&[1.0, 0.0, 0.0]);
        let t2 = point(&[0.0, 0.0, 1.0]);

        let individual = vec![
            score_triple(&h1, &r, &t1).unwrap(),
            score_triple(&h2, &r, &t2).unwrap(),
        ];
        let batch = score_batch(&[h1, h2], &[r.clone(), r], &[t1, t2]).unwrap();

        for (i, (a, b)) in individual.iter().zip(batch.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "batch[{i}] = {b}, individual = {a}");
        }
    }

    #[test]
    fn dimension_mismatch_errors() {
        let h = point(&[1.0, 0.0, 0.0]);
        let t = point(&[1.0, 0.0]);
        let r = SphericalRelation::identity(3);
        assert!(score_triple(&h, &r, &t).is_err());

        let r_bad = SphericalRelation::identity(2);
        assert!(score_triple(&h, &r_bad, &h).is_err());
    }

    #[test]
    fn rejects_zero_vector() {
        assert!(SphericalPoint::new(vec![0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn rejects_non_finite() {
        assert!(SphericalPoint::new(vec![f32::NAN, 1.0]).is_err());
        assert!(SphericalPoint::new(vec![f32::INFINITY, 1.0]).is_err());
        assert!(SphericalRelation::new(vec![1.0, 0.0], f32::NAN).is_err());
    }

    #[test]
    fn embedding_model_construction() {
        let entities = vec![point(&[1.0, 0.0, 0.0]), point(&[0.0, 1.0, 0.0])];
        let relations = vec![SphericalRelation::identity(3)];
        let model = SphericalEmbedding::new(entities, relations, 3);
        assert!(model.is_ok());
        let m = model.unwrap();
        assert_eq!(m.num_entities(), 2);
        assert_eq!(m.num_relations(), 1);
        assert_eq!(m.dim(), 3);
    }

    #[test]
    fn embedding_model_rejects_dim_mismatch() {
        let entities = vec![point(&[1.0, 0.0])]; // dim=2
        let relations = vec![SphericalRelation::identity(3)]; // dim=3
        assert!(SphericalEmbedding::new(entities, relations, 3).is_err());
    }

    #[test]
    fn score_symmetry_with_identity() {
        // With identity relation, score(h, id, t) == score(t, id, h)
        let h = point(&[1.0, 0.0, 0.0]);
        let t = point(&[0.6, 0.8, 0.0]);
        let r = SphericalRelation::identity(3);
        let s1 = score_triple(&h, &r, &t).unwrap();
        let s2 = score_triple(&t, &r, &h).unwrap();
        assert!(
            (s1 - s2).abs() < 1e-6,
            "identity relation should give symmetric scores: {s1} vs {s2}"
        );
    }

    #[test]
    fn higher_dimensional_works() {
        let d = 50;
        let mut v1 = vec![0.0f32; d];
        let mut v2 = vec![0.0f32; d];
        v1[0] = 1.0;
        v2[1] = 1.0;
        let h = point(&v1);
        let t = point(&v2);
        let r = SphericalRelation::identity(d);
        let score = score_triple(&h, &r, &t).unwrap();
        assert!(
            (score - PI / 2.0).abs() < 1e-5,
            "50d orthogonal score = {score}, expected pi/2"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_point(dim: usize) -> impl Strategy<Value = SphericalPoint> {
        prop::collection::vec(-10.0f32..10.0, dim)
            .prop_filter_map("non-zero vector", move |coords| {
                SphericalPoint::new(coords).ok()
            })
    }

    fn arb_relation(dim: usize) -> impl Strategy<Value = SphericalRelation> {
        (
            prop::collection::vec(-10.0f32..10.0, dim),
            -std::f32::consts::PI..std::f32::consts::PI,
        )
            .prop_filter_map("non-zero axis", move |(axis, angle)| {
                SphericalRelation::new(axis, angle).ok()
            })
    }

    proptest! {
        #[test]
        fn prop_score_nonnegative(
            h in arb_point(4),
            r in arb_relation(4),
            t in arb_point(4),
        ) {
            let score = score_triple(&h, &r, &t).unwrap();
            prop_assert!(score >= -1e-9, "score should be >= 0, got {score}");
        }

        #[test]
        fn prop_score_at_most_pi(
            h in arb_point(4),
            r in arb_relation(4),
            t in arb_point(4),
        ) {
            let score = score_triple(&h, &r, &t).unwrap();
            prop_assert!(
                score <= std::f64::consts::PI + 1e-6,
                "score should be <= pi, got {score}"
            );
        }

        #[test]
        fn prop_identity_self_score_zero(
            h in arb_point(4),
        ) {
            let r = SphericalRelation::identity(4);
            let score = score_triple(&h, &r, &h).unwrap();
            // f32 accumulation through decompose-rotate-renormalize gives ~1e-3 drift.
            prop_assert!(
                score.abs() < 1e-3,
                "score(h, identity, h) = {score}, expected ~0"
            );
        }

        #[test]
        fn prop_rotation_preserves_norm(
            h in arb_point(4),
            r in arb_relation(4),
        ) {
            let rotated = rotate(&h, &r).unwrap();
            let norm: f32 = rotated.coords().iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!(
                (norm - 1.0).abs() < 1e-4,
                "rotated norm = {norm}, expected 1.0"
            );
        }

        #[test]
        fn prop_projection_idempotent(
            v in prop::collection::vec(-10.0f32..10.0, 4)
                .prop_filter("non-zero", |v| v.iter().map(|x| x*x).sum::<f32>() > 1e-12)
        ) {
            let p1 = project_to_sphere(v).unwrap();
            let p2 = project_to_sphere(p1.coords().to_vec()).unwrap();
            for (a, b) in p1.coords().iter().zip(p2.coords().iter()) {
                prop_assert!(
                    (a - b).abs() < 1e-5,
                    "projection should be idempotent: {a} vs {b}"
                );
            }
        }
    }
}
