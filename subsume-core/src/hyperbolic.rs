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
//! # Why Hyperbolic for NLP?
//!
//! Many NLP structures are inherently hierarchical:
//! - WordNet synsets
//! - Named entity type ontologies
//! - Coreference type constraints
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
//! Training implementations may be added to subsume-candle with autograd support.
//!
//! # References
//!
//! - Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
//! - Nickel & Kiela (2018): "Learning Continuous Hierarchies in the Lorentz Model"
//! - Chami et al. (2019): "Hyperbolic Graph Convolutional Neural Networks"
//! - Gulcehre et al. (2019): "Hyperbolic Attention Networks"

use std::fmt::Debug;

/// Error type for hyperbolic operations.
#[derive(Debug, Clone, PartialEq)]
pub enum HyperbolicError {
    /// Point is outside the Poincaré ball (||x|| >= 1).
    OutsideBall {
        /// The norm of the point.
        norm: f64,
    },
    /// Dimension mismatch between points.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Numerical instability (e.g., division by near-zero).
    NumericalInstability(String),
    /// Invalid curvature (must be negative).
    InvalidCurvature(f64),
}

impl std::fmt::Display for HyperbolicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutsideBall { norm } => {
                write!(f, "Point outside Poincaré ball: ||x|| = {} >= 1", norm)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
            Self::InvalidCurvature(c) => write!(f, "Invalid curvature: {} (must be negative)", c),
        }
    }
}

impl std::error::Error for HyperbolicError {}

/// A point in hyperbolic space (Poincaré ball model).
pub trait HyperbolicPoint: Clone + Debug {
    /// Scalar type.
    type Scalar: Clone + Debug;
    /// Vector type for coordinates.
    type Vector: Clone + Debug;

    /// Dimension of the embedding space.
    fn dim(&self) -> usize;

    /// Get the coordinates in the Poincaré ball.
    fn coords(&self) -> &Self::Vector;

    /// Squared norm ||x||².
    fn norm_squared(&self) -> Self::Scalar;

    /// Check if the point is inside the ball (||x|| < 1).
    fn is_valid(&self) -> bool;

    /// Conformal factor λ_x = 2 / (1 - ||x||²).
    fn conformal_factor(&self) -> Result<Self::Scalar, HyperbolicError>;

    /// Hyperbolic distance to another point.
    ///
    /// d(x, y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
    fn distance(&self, other: &Self) -> Result<Self::Scalar, HyperbolicError>;

    /// Möbius addition: x ⊕ y.
    ///
    /// The generalized addition operation in hyperbolic space.
    fn mobius_add(&self, other: &Self) -> Result<Self, HyperbolicError>;

    /// Möbius scalar multiplication: r ⊗ x.
    fn mobius_scalar_mul(&self, r: Self::Scalar) -> Result<Self, HyperbolicError>;

    /// Exponential map: exp_x(v).
    ///
    /// Maps a tangent vector v at point x to a point on the manifold.
    fn exp_map(&self, tangent: &Self::Vector) -> Result<Self, HyperbolicError>;

    /// Logarithmic map: log_x(y).
    ///
    /// Maps a point y to a tangent vector at x.
    fn log_map(&self, other: &Self) -> Result<Self::Vector, HyperbolicError>;

    /// Project to the Poincaré ball if outside.
    ///
    /// Clips the norm to be < 1 - epsilon.
    fn project(&self, epsilon: Self::Scalar) -> Self;
}

/// Curvature of hyperbolic space.
///
/// Standard Poincaré ball has curvature -1.
/// Lower curvature (more negative) = more hyperbolic = more "space" near boundary.
#[derive(Debug, Clone, Copy)]
pub struct Curvature(pub f64);

impl Curvature {
    /// Standard hyperbolic curvature.
    pub const STANDARD: Self = Self(-1.0);

    /// Create new curvature (must be negative).
    pub fn new(c: f64) -> Result<Self, HyperbolicError> {
        if c >= 0.0 {
            return Err(HyperbolicError::InvalidCurvature(c));
        }
        Ok(Self(c))
    }

    /// Get the absolute value |c|.
    pub fn abs(&self) -> f64 {
        self.0.abs()
    }

    /// Get the square root of |c|.
    pub fn sqrt_abs(&self) -> f64 {
        self.0.abs().sqrt()
    }
}

impl Default for Curvature {
    fn default() -> Self {
        Self::STANDARD
    }
}

// =============================================================================
// Simple Implementation (f64, Vec<f64>)
// =============================================================================

/// A point in the Poincaré ball using Vec<f64>.
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
    pub fn new_projected(coords: Vec<f64>, curvature: Curvature) -> Self {
        let mut point = Self { coords, curvature };
        point = point.project(1e-5);
        point
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
}

impl HyperbolicPoint for PoincareBallPoint {
    type Scalar = f64;
    type Vector = Vec<f64>;

    fn dim(&self) -> usize {
        self.coords.len()
    }

    fn coords(&self) -> &Self::Vector {
        &self.coords
    }

    fn norm_squared(&self) -> Self::Scalar {
        self.coords.iter().map(|x| x * x).sum()
    }

    fn is_valid(&self) -> bool {
        self.norm_squared() < 1.0
    }

    fn conformal_factor(&self) -> Result<Self::Scalar, HyperbolicError> {
        let norm_sq = self.norm_squared();
        if norm_sq >= 1.0 {
            return Err(HyperbolicError::OutsideBall {
                norm: norm_sq.sqrt(),
            });
        }
        Ok(2.0 / (1.0 - norm_sq))
    }

    fn distance(&self, other: &Self) -> Result<Self::Scalar, HyperbolicError> {
        if self.coords.len() != other.coords.len() {
            return Err(HyperbolicError::DimensionMismatch {
                expected: self.coords.len(),
                actual: other.coords.len(),
            });
        }

        let c = self.curvature.sqrt_abs();

        let x_norm_sq = self.norm_squared();
        let y_norm_sq = other.norm_squared();

        if x_norm_sq >= 1.0 {
            return Err(HyperbolicError::OutsideBall {
                norm: x_norm_sq.sqrt(),
            });
        }
        if y_norm_sq >= 1.0 {
            return Err(HyperbolicError::OutsideBall {
                norm: y_norm_sq.sqrt(),
            });
        }

        // ||x - y||²
        let diff_sq: f64 = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        let numerator = 2.0 * diff_sq;
        let denominator = (1.0 - x_norm_sq) * (1.0 - y_norm_sq);

        if denominator < 1e-15 {
            return Err(HyperbolicError::NumericalInstability(
                "Denominator too small in distance calculation".into(),
            ));
        }

        let ratio = numerator / denominator;

        // d(x,y) = (1/c) * arcosh(1 + c² * ratio)
        // For c = 1: d(x,y) = arcosh(1 + ratio)
        let arg = 1.0 + c * c * ratio;

        // arcosh(x) = ln(x + sqrt(x² - 1)) for x >= 1
        let distance = if arg >= 1.0 {
            (arg + (arg * arg - 1.0).sqrt()).ln() / c
        } else {
            0.0 // Numerically, points are the same
        };

        Ok(distance)
    }

    fn mobius_add(&self, other: &Self) -> Result<Self, HyperbolicError> {
        if self.coords.len() != other.coords.len() {
            return Err(HyperbolicError::DimensionMismatch {
                expected: self.coords.len(),
                actual: other.coords.len(),
            });
        }

        let x = &self.coords;
        let y = &other.coords;

        let x_norm_sq = self.norm_squared();
        let y_norm_sq = other.norm_squared();

        // <x, y>
        let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        // Numerator: (1 + 2<x,y> + ||y||²)x + (1 - ||x||²)y
        let coef_x = 1.0 + 2.0 * dot + y_norm_sq;
        let coef_y = 1.0 - x_norm_sq;

        // Denominator: 1 + 2<x,y> + ||x||²||y||²
        let denom = 1.0 + 2.0 * dot + x_norm_sq * y_norm_sq;

        if denom.abs() < 1e-15 {
            return Err(HyperbolicError::NumericalInstability(
                "Denominator too small in Möbius addition".into(),
            ));
        }

        let result: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (coef_x * xi + coef_y * yi) / denom)
            .collect();

        Self::new(result, self.curvature)
    }

    fn mobius_scalar_mul(&self, r: Self::Scalar) -> Result<Self, HyperbolicError> {
        let norm = self.norm_squared().sqrt();

        if norm < 1e-15 {
            // At origin, scalar multiplication is trivial
            return Ok(self.clone());
        }

        // tanh(r * arctanh(||x||)) * x / ||x||
        let arctanh_norm = ((1.0 + norm) / (1.0 - norm)).ln() / 2.0;
        let new_norm = (r * arctanh_norm).tanh();

        let scale = new_norm / norm;
        let result: Vec<f64> = self.coords.iter().map(|x| x * scale).collect();

        Self::new(result, self.curvature)
    }

    fn exp_map(&self, tangent: &Self::Vector) -> Result<Self, HyperbolicError> {
        if self.coords.len() != tangent.len() {
            return Err(HyperbolicError::DimensionMismatch {
                expected: self.coords.len(),
                actual: tangent.len(),
            });
        }

        let lambda = self.conformal_factor()?;
        let v_norm: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();

        if v_norm < 1e-15 {
            return Ok(self.clone());
        }

        // exp_x(v) = x ⊕ (tanh(λ_x ||v|| / 2) * v / ||v||)
        let coef = (lambda * v_norm / 2.0).tanh() / v_norm;
        let direction: Vec<f64> = tangent.iter().map(|x| x * coef).collect();

        let direction_point = Self::new_projected(direction, self.curvature);
        self.mobius_add(&direction_point)
    }

    fn log_map(&self, other: &Self) -> Result<Self::Vector, HyperbolicError> {
        if self.coords.len() != other.coords.len() {
            return Err(HyperbolicError::DimensionMismatch {
                expected: self.coords.len(),
                actual: other.coords.len(),
            });
        }

        let lambda = self.conformal_factor()?;

        // -x ⊕ y
        let neg_x: Vec<f64> = self.coords.iter().map(|x| -x).collect();
        let neg_x_point = Self::new_projected(neg_x, self.curvature);
        let diff = neg_x_point.mobius_add(other)?;

        let diff_norm = diff.norm_squared().sqrt();
        if diff_norm < 1e-15 {
            return Ok(vec![0.0; self.coords.len()]);
        }

        // (2 / λ_x) * arctanh(||−x ⊕ y||) * (−x ⊕ y) / ||−x ⊕ y||
        let arctanh_norm = ((1.0 + diff_norm) / (1.0 - diff_norm + 1e-15)).ln() / 2.0;
        let coef = (2.0 / lambda) * arctanh_norm / diff_norm;

        Ok(diff.coords.iter().map(|x| x * coef).collect())
    }

    fn project(&self, epsilon: Self::Scalar) -> Self {
        let norm_sq = self.norm_squared();
        let max_norm = 1.0 - epsilon;

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
        let point = PoincareBallPoint::new_projected(coords, Curvature::STANDARD);

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
}
