//! # Hyperbolic Embeddings
//!
//! Embeddings in hyperbolic space for representing hierarchical structures.
//!
//! Hyperbolic space has the remarkable property that it can embed trees with
//! arbitrarily low distortion, unlike Euclidean space. This makes it ideal for:
//! - Entity type hierarchies (PERSON → POLITICIAN → PRESIDENT)
//! - Ontology embeddings (is-a relationships)
//! - Taxonomic structures
//!
//! # Why Hyperbolic for Knowledge Graphs?
//!
//! Many KG structures are inherently hierarchical:
//! - WordNet synsets
//! - Named entity type ontologies
//! - Subsumption hierarchies
//!
//! Hyperbolic space grows exponentially with radius, allowing deep hierarchies
//! to be embedded compactly near the boundary of the Poincaré ball.
//!
//! # The Poincaré Ball Model
//!
//! We use the Poincaré ball model B^n = {x ∈ R^n : ||x|| < 1}.
//!
//! Key operations:
//! - **Möbius addition**: x ⊕ y (generalized addition in hyperbolic space)
//! - **Exponential map**: exp_x(v) (move from x in direction v)
//! - **Logarithmic map**: log_x(y) (direction from x to y)
//! - **Distance**: d(x, y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
//!
//! # Relationship to Box Embeddings
//!
//! Recent work (RegD 2025) shows that Euclidean box embeddings with depth-based
//! distance can achieve "hyperbolic-like expressiveness" for hierarchies.
//! This suggests boxes + depth distance may be sufficient for many NLP tasks.
//!
//! However, hyperbolic embeddings remain useful for:
//! - Direct ontology embedding
//! - When hierarchical structure is the primary concern
//! - Integration with Riemannian optimization frameworks (geoopt)
//!
//! # Implementation Notes
//!
//! Training hyperbolic embeddings requires Riemannian optimization:
//! - Riemannian SGD (RSGD)
//! - Riemannian Adam
//!
//! These are complex to implement in pure Rust. For training, consider:
//! - Python: `geoopt` library with PyTorch
//! - Export trained embeddings to Rust for inference
//!
//! This module provides inference-time operations and the trait definitions.
//! Training implementations may be added to the `candle_backend` module with autograd support.
//!
//! # References
//!
//! - Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
//! - Nickel & Kiela (2018): "Learning Continuous Hierarchies in the Lorentz Model"
//! - Chami et al. (2019): "Hyperbolic Graph Convolutional Neural Networks"
//! - Gulcehre et al. (2019): "Hyperbolic Attention Networks"

/// Error type for hyperbolic operations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HyperbolicError {
    /// Point is outside the Poincaré ball (||x|| >= 1).
    #[error("Point outside Poincaré ball: ||x|| = {norm} >= 1")]
    OutsideBall {
        /// The norm of the point.
        norm: f64,
    },
    /// Dimension mismatch between points.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Invalid curvature (must be negative).
    #[error("Invalid curvature: {0} (must be negative)")]
    InvalidCurvature(f64),
}

/// Curvature of hyperbolic space.
///
/// Standard Poincaré ball has curvature -1.
/// Lower curvature (more negative) = more hyperbolic = more "space" near boundary.
#[derive(Debug, Clone, Copy)]
pub struct Curvature(f64);

impl Curvature {
    /// Standard hyperbolic curvature.
    pub const STANDARD: Self = Self(-1.0);

    /// Create new curvature (must be negative and finite).
    pub fn new(c: f64) -> Result<Self, HyperbolicError> {
        if c.is_nan() || c >= 0.0 || c.is_infinite() {
            return Err(HyperbolicError::InvalidCurvature(c));
        }
        Ok(Self(c))
    }

    /// Get the raw curvature value (negative).
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Get the absolute value |c|.
    pub fn abs(&self) -> f64 {
        self.0.abs()
    }
}

impl Default for Curvature {
    fn default() -> Self {
        Self::STANDARD
    }
}

// =============================================================================
// Simple Implementation (f64, Vec<f64>) - Backed by `hyp`
// =============================================================================

/// A point in the Poincare ball using `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct PoincareBallPoint {
    /// Coordinates in the ball.
    coords: Vec<f64>,
    /// Curvature (negative).
    curvature: Curvature,
}

impl PoincareBallPoint {
    /// Create a new point in the Poincaré ball.
    ///
    /// Returns error if the point is outside the ball.
    pub fn new(coords: Vec<f64>, curvature: Curvature) -> Result<Self, HyperbolicError> {
        let point = Self { coords, curvature };
        if !point.is_valid() {
            let norm = point.norm_squared().sqrt();
            return Err(HyperbolicError::OutsideBall { norm });
        }
        Ok(point)
    }

    /// Create a new point, projecting if necessary.
    ///
    /// Returns error if any coordinate is NaN.
    pub fn new_projected(coords: Vec<f64>, curvature: Curvature) -> Result<Self, HyperbolicError> {
        if coords.iter().any(|c| c.is_nan()) {
            return Err(HyperbolicError::OutsideBall { norm: f64::NAN });
        }
        let mut point = Self { coords, curvature };
        point = point.project(1e-5);
        Ok(point)
    }

    /// Create the origin point.
    pub fn origin(dim: usize, curvature: Curvature) -> Self {
        Self {
            coords: vec![0.0; dim],
            curvature,
        }
    }

    /// Get curvature.
    pub fn curvature(&self) -> Curvature {
        self.curvature
    }

    /// Helper to get the `hyp` manifold instance.
    fn hyp_manifold(&self) -> hyperball::PoincareBall<f64> {
        // hyp uses c > 0 where curvature K = -c.
        // subsume uses K < 0.
        // So c = -K.
        hyperball::PoincareBall::new(-self.curvature.value())
    }

    /// Dimension of the embedding space.
    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Get the coordinates in the Poincare ball.
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }

    /// Squared norm ||x||^2.
    pub fn norm_squared(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum()
    }

    /// Check if the point is inside the ball (||x|| < 1).
    pub fn is_valid(&self) -> bool {
        // hyperball::PoincareBall::is_in_ball checks ||x|| < 1/sqrt(c).
        // Our c = -K.
        // If K = -1, c = 1, limit = 1.
        use ndarray::ArrayView1;
        let m = self.hyp_manifold();
        // unwrap is safe because we constructed from Vec
        let v = ArrayView1::from_shape((self.coords.len(),), &self.coords).unwrap();
        m.is_in_ball(&v)
    }

    /// Conformal factor: 2 / (1 - c * ||x||^2).
    pub fn conformal_factor(&self) -> Result<f64, HyperbolicError> {
        let norm_sq = self.norm_squared();
        // hyp doesn't expose conformal factor directly in public API (it's internal).
        // We calculate it manually to match existing API, or add it to hyp.
        // Formula: 2 / (1 - c * ||x||^2)
        // c = -K.
        let c = -self.curvature.value();
        let denom = 1.0 - c * norm_sq;
        if denom <= 1e-15 {
            return Err(HyperbolicError::OutsideBall {
                norm: norm_sq.sqrt(),
            });
        }
        Ok(2.0 / denom)
    }

    /// Hyperbolic distance to another point.
    pub fn distance(&self, other: &Self) -> Result<f64, HyperbolicError> {
        if self.coords.len() != other.coords.len() {
            return Err(HyperbolicError::DimensionMismatch {
                expected: self.coords.len(),
                actual: other.coords.len(),
            });
        }
        use ndarray::ArrayView1;
        let m = self.hyp_manifold();
        let v1 = ArrayView1::from_shape((self.coords.len(),), &self.coords).unwrap();
        let v2 = ArrayView1::from_shape((other.coords.len(),), &other.coords).unwrap();

        Ok(m.distance(&v1, &v2))
    }

    /// Mobius addition: x (+) y.
    pub fn mobius_add(&self, other: &Self) -> Result<Self, HyperbolicError> {
        if self.coords.len() != other.coords.len() {
            return Err(HyperbolicError::DimensionMismatch {
                expected: self.coords.len(),
                actual: other.coords.len(),
            });
        }
        use ndarray::ArrayView1;
        let m = self.hyp_manifold();
        let v1 = ArrayView1::from_shape((self.coords.len(),), &self.coords).unwrap();
        let v2 = ArrayView1::from_shape((other.coords.len(),), &other.coords).unwrap();

        let res = m.mobius_add(&v1, &v2);
        Ok(Self {
            coords: res.to_vec(),
            curvature: self.curvature,
        })
    }

    /// Mobius scalar multiplication: r (x) x.
    pub fn mobius_scalar_mul(&self, r: f64) -> Result<Self, HyperbolicError> {
        // hyp doesn't have scalar mul yet?
        // Let's keep the local implementation or add it to hyp.
        // Local impl is fine for now as it's just a formula.
        let norm = self.norm_squared().sqrt();

        if norm < 1e-15 {
            return Ok(self.clone());
        }

        // tanh(r * arctanh(sqrt(c) * ||x||)) / sqrt(c) * x / ||x||
        // For c=1: tanh(r * arctanh(||x||)) * x / ||x||

        let c = -self.curvature.value();
        let c_sqrt = c.sqrt();

        let arg = c_sqrt * norm;
        if arg >= 1.0 {
            return Err(HyperbolicError::OutsideBall { norm });
        }

        let arctanh_arg = arg.atanh(); // standard atanh
        let new_norm_scaled = (r * arctanh_arg).tanh();
        let new_norm = new_norm_scaled / c_sqrt;

        let scale = new_norm / norm;
        let result: Vec<f64> = self.coords.iter().map(|x| x * scale).collect();

        Self::new(result, self.curvature)
    }

    /// Exponential map: exp_x(v). Maps a tangent vector to the manifold.
    pub fn exp_map(&self, tangent: &[f64]) -> Result<Self, HyperbolicError> {
        if self.coords.len() != tangent.len() {
            return Err(HyperbolicError::DimensionMismatch {
                expected: self.coords.len(),
                actual: tangent.len(),
            });
        }
        use ndarray::ArrayView1;
        use skel::Manifold; // Trait from skel

        let m = self.hyp_manifold();
        let x = ArrayView1::from_shape((self.coords.len(),), &self.coords).unwrap();
        let v = ArrayView1::from_shape((tangent.len(),), tangent).unwrap();

        let res = m.exp_map(&x, &v);
        Ok(Self {
            coords: res.to_vec(),
            curvature: self.curvature,
        })
    }

    /// Logarithmic map: log_x(y). Maps a point to a tangent vector at x.
    pub fn log_map(&self, other: &Self) -> Result<Vec<f64>, HyperbolicError> {
        if self.coords.len() != other.coords.len() {
            return Err(HyperbolicError::DimensionMismatch {
                expected: self.coords.len(),
                actual: other.coords.len(),
            });
        }
        use ndarray::ArrayView1;
        use skel::Manifold;

        let m = self.hyp_manifold();
        let x = ArrayView1::from_shape((self.coords.len(),), &self.coords).unwrap();
        let y = ArrayView1::from_shape((other.coords.len(),), &other.coords).unwrap();

        let res = m.log_map(&x, &y);
        Ok(res.to_vec())
    }

    /// Project to the Poincare ball if outside. Clips norm to < 1 - epsilon.
    pub fn project(&self, epsilon: f64) -> Self {
        let norm_sq = self.norm_squared();
        let c = -self.curvature.value();
        let max_norm = (1.0 / c).sqrt() - epsilon;

        if norm_sq < max_norm * max_norm {
            return self.clone();
        }

        let norm = norm_sq.sqrt();
        let scale = max_norm / norm;
        let coords: Vec<f64> = self.coords.iter().map(|x| x * scale).collect();

        Self {
            coords,
            curvature: self.curvature,
        }
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Compute pairwise distances for a set of points.
pub fn pairwise_distances(points: &[PoincareBallPoint]) -> Result<Vec<Vec<f64>>, HyperbolicError> {
    let n = points.len();
    let mut distances = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = points[i].distance(&points[j])?;
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }

    Ok(distances)
}

/// Check if a hierarchy is preserved in the embedding.
///
/// For each (parent, child) pair, verifies that the parent is closer to
/// the origin than the child (hierarchical property of Poincaré embeddings).
pub fn hierarchy_preserved(
    points: &[PoincareBallPoint],
    parent_child_pairs: &[(usize, usize)],
) -> Result<f64, HyperbolicError> {
    let mut correct = 0;
    let total = parent_child_pairs.len();

    for (parent_idx, child_idx) in parent_child_pairs {
        let parent_norm = points[*parent_idx].norm_squared();
        let child_norm = points[*child_idx].norm_squared();

        // Parent should be closer to origin (smaller norm)
        if parent_norm < child_norm {
            correct += 1;
        }
    }

    Ok(correct as f64 / total.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_origin() {
        let origin = PoincareBallPoint::origin(3, Curvature::STANDARD);
        assert!(origin.is_valid());
        assert!(origin.norm_squared() < 1e-10);
    }

    #[test]
    fn test_conformal_factor() {
        let origin = PoincareBallPoint::origin(3, Curvature::STANDARD);
        let lambda = origin.conformal_factor().unwrap();
        assert!((lambda - 2.0).abs() < 1e-10); // At origin: 2 / (1 - 0) = 2
    }

    #[test]
    fn test_distance_symmetric() {
        let p1 = PoincareBallPoint::new(vec![0.1, 0.2], Curvature::STANDARD).unwrap();
        let p2 = PoincareBallPoint::new(vec![0.3, 0.1], Curvature::STANDARD).unwrap();

        let d1 = p1.distance(&p2).unwrap();
        let d2 = p2.distance(&p1).unwrap();

        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_self() {
        let p = PoincareBallPoint::new(vec![0.5, 0.3], Curvature::STANDARD).unwrap();
        let d = p.distance(&p).unwrap();
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_mobius_add_origin() {
        let origin = PoincareBallPoint::origin(2, Curvature::STANDARD);
        let p = PoincareBallPoint::new(vec![0.3, 0.4], Curvature::STANDARD).unwrap();

        // origin ⊕ p = p
        let result = origin.mobius_add(&p).unwrap();
        assert!((result.coords[0] - 0.3).abs() < 1e-10);
        assert!((result.coords[1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_project() {
        // Point outside the ball
        let coords = vec![0.9, 0.9]; // norm ≈ 1.27
        let point = PoincareBallPoint::new_projected(coords, Curvature::STANDARD).unwrap();

        assert!(point.is_valid());
        assert!(point.norm_squared() < 1.0);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let base = PoincareBallPoint::new(vec![0.1, 0.1], Curvature::STANDARD).unwrap();
        let target = PoincareBallPoint::new(vec![0.3, 0.2], Curvature::STANDARD).unwrap();

        // log then exp should return to target
        let tangent = base.log_map(&target).unwrap();
        let result = base.exp_map(&tangent).unwrap();

        assert!((result.coords[0] - target.coords[0]).abs() < 1e-6);
        assert!((result.coords[1] - target.coords[1]).abs() < 1e-6);
    }

    #[test]
    fn test_curvature_validation() {
        // Positive curvature should fail.
        assert!(Curvature::new(1.0).is_err());
        assert!(Curvature::new(0.0).is_err());
        // Negative curvature should succeed.
        let c = Curvature::new(-2.0).unwrap();
        assert!((c.abs() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_outside_ball_rejected() {
        // Norm > 1 should fail for standard curvature.
        let result = PoincareBallPoint::new(vec![0.8, 0.8], Curvature::STANDARD);
        assert!(
            result.is_err(),
            "point with norm > 1 should be rejected, norm = {}",
            (0.8_f64 * 0.8 + 0.8 * 0.8).sqrt()
        );
    }

    #[test]
    fn test_dimension_mismatch_distance() {
        let p1 = PoincareBallPoint::new(vec![0.1, 0.2], Curvature::STANDARD).unwrap();
        let p2 = PoincareBallPoint::new(vec![0.1, 0.2, 0.3], Curvature::STANDARD).unwrap();
        assert!(p1.distance(&p2).is_err());
    }

    #[test]
    fn test_pairwise_distances_symmetric() {
        let points = vec![
            PoincareBallPoint::new(vec![0.1, 0.0], Curvature::STANDARD).unwrap(),
            PoincareBallPoint::new(vec![0.0, 0.2], Curvature::STANDARD).unwrap(),
            PoincareBallPoint::origin(2, Curvature::STANDARD),
        ];
        let dists = pairwise_distances(&points).unwrap();
        assert_eq!(dists.len(), 3);
        for (i, row) in dists.iter().enumerate() {
            assert!(row[i].abs() < 1e-10, "diagonal should be 0");
            for (j, &d_ij) in row.iter().enumerate() {
                assert!(
                    (d_ij - dists[j][i]).abs() < 1e-10,
                    "distance matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_hierarchy_preserved_basic() {
        // Parent closer to origin (smaller norm), child farther.
        let parent = PoincareBallPoint::new(vec![0.1, 0.0], Curvature::STANDARD).unwrap();
        let child = PoincareBallPoint::new(vec![0.5, 0.0], Curvature::STANDARD).unwrap();
        let points = vec![parent, child];

        let accuracy = hierarchy_preserved(&points, &[(0, 1)]).unwrap();
        assert!(
            (accuracy - 1.0).abs() < 1e-10,
            "parent (norm 0.1) should be closer to origin than child (norm 0.5), accuracy = {accuracy}"
        );

        // Reversed: child closer to origin => accuracy = 0
        let accuracy_rev = hierarchy_preserved(&points, &[(1, 0)]).unwrap();
        assert!(
            accuracy_rev.abs() < 1e-10,
            "reversed hierarchy should have accuracy 0, got {accuracy_rev}"
        );
    }

    // ---- Property tests ----

    use proptest::prelude::*;

    /// Generate a valid Poincare ball point with norm < 0.95.
    fn arb_poincare_point(dim: usize) -> impl Strategy<Value = PoincareBallPoint> {
        prop::collection::vec(-1.0f64..1.0, dim).prop_filter_map(
            "point must be inside Poincare ball",
            move |coords| {
                let norm_sq: f64 = coords.iter().map(|x| x * x).sum();
                if norm_sq >= 0.95 * 0.95 {
                    None
                } else {
                    PoincareBallPoint::new(coords, Curvature::STANDARD).ok()
                }
            },
        )
    }

    proptest! {
        /// d(a, b) == d(b, a) for any two valid Poincare points.
        #[test]
        fn prop_poincare_distance_symmetric(
            a in arb_poincare_point(3),
            b in arb_poincare_point(3),
        ) {
            let d_ab = a.distance(&b).unwrap();
            let d_ba = b.distance(&a).unwrap();
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-10,
                "distance should be symmetric: d(a,b)={d_ab}, d(b,a)={d_ba}"
            );
        }

        /// d(a, b) >= 0 for any two valid Poincare points.
        #[test]
        fn prop_poincare_distance_non_negative(
            a in arb_poincare_point(4),
            b in arb_poincare_point(4),
        ) {
            let d = a.distance(&b).unwrap();
            prop_assert!(d >= -1e-10, "distance should be non-negative, got {d}");
        }
    }

    // ---- NaN rejection ----

    #[test]
    fn curvature_nan_returns_err() {
        assert!(
            Curvature::new(f64::NAN).is_err(),
            "NaN curvature should be rejected"
        );
    }

    #[test]
    fn curvature_field_is_private() {
        // Curvature can only be constructed via new() or STANDARD.
        let c = Curvature::STANDARD;
        assert!((c.value() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn new_projected_nan_coords_returns_err() {
        let result = PoincareBallPoint::new_projected(vec![f64::NAN, 0.0], Curvature::STANDARD);
        assert!(result.is_err(), "NaN coords should be rejected");
    }

    #[test]
    fn new_projected_all_nan_returns_err() {
        let result =
            PoincareBallPoint::new_projected(vec![f64::NAN, f64::NAN], Curvature::STANDARD);
        assert!(result.is_err(), "All-NaN coords should be rejected");
    }

    #[test]
    fn curvature_neg_infinity_returns_err() {
        assert!(
            Curvature::new(-f64::INFINITY).is_err(),
            "-Inf curvature should be rejected"
        );
    }

    #[test]
    fn curvature_pos_infinity_returns_err() {
        assert!(
            Curvature::new(f64::INFINITY).is_err(),
            "+Inf curvature should be rejected (positive and infinite)"
        );
    }
}
