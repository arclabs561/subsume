//! Generic `Region` trait for geometric shapes (boxes, balls, cones).
//!
//! This module defines a trait that abstracts over geometric regions in d-dimensional
//! Euclidean space, supporting point-in-region tests, signed distance queries, and
//! RegD dissimilarity metrics.
//!
//! # Research Background
//!
//! The `depth_dissimilarity` and `boundary_dissimilarity` metrics are based on the
//! RegD framework from **Yang & Chen (2025)**, "Achieving Hyperbolic-Like Expressiveness
//! with Arbitrary Euclidean Regions" (arXiv:2501.17518). RegD shows that incorporating
//! region size into distance calculations achieves hyperbolic-like expressiveness using
//! arbitrary Euclidean regions (boxes, balls, cones), without the computational overhead
//! of hyperbolic geometry.
//!
//! The `Region` trait itself is informed by two complementary lines of work:
//!
//! - **Xiong (2024)**, "Geometric Relational Embeddings" (PhD thesis, UC Santa Cruz),
//!   validates that abstracting over region shapes (boxes, balls, cones) with a shared
//!   interface is a principled design: different geometric primitives offer distinct
//!   inductive biases, and a unified trait lets downstream code stay shape-agnostic.
//!
//! - **Song et al. (2024)**, "Expressiveness Analysis and Enhancing Framework for
//!   Geometric KGE Models" (IEEE TKDE), provides a systematic analysis of which region
//!   shapes can capture which relation patterns (symmetry, antisymmetry, composition,
//!   inversion). This informs the choice of concrete `Region` implementors: boxes
//!   handle per-dimension independence well, cones add negation closure, and octagons
//!   add composition closure.

/// Errors from region operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RegionError {
    /// Dimension mismatch between a point and a region, or two regions.
    #[error("Region dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Actual dimensionality.
        actual: usize,
    },

    /// Region has zero volume (degenerate).
    #[error("Region has zero volume")]
    ZeroVolume,

    /// Generic internal error.
    #[error("Region error: {0}")]
    Internal(String),
}

/// A geometric region in d-dimensional Euclidean space.
///
/// Implementations include axis-aligned boxes (`NdarrayBox`), balls (`Ball`), and
/// potentially cones. The trait provides a uniform interface for point membership,
/// signed boundary distance, volume, and the RegD dissimilarity metrics.
///
/// # RegD Dissimilarity Metrics
///
/// The `depth_dissimilarity` and `boundary_dissimilarity` methods implement the RegD
/// framework's key insight: by incorporating region size (volume) into distance
/// calculations through polynomial functions, Euclidean regions can emulate the
/// exponential growth of hyperbolic distance, solving the crowding problem in
/// hierarchical embeddings.
pub trait Region {
    /// Test whether a point lies inside or on the boundary of this region.
    ///
    /// # Errors
    ///
    /// Returns `RegionError::DimensionMismatch` if `point.len() != self.dim()`.
    fn contains(&self, point: &[f32]) -> Result<bool, RegionError>;

    /// Signed distance from a point to the region boundary.
    ///
    /// - Positive: point is outside the region (distance to nearest boundary point).
    /// - Negative: point is inside the region (distance to nearest boundary point, negated).
    /// - Zero: point is on the boundary.
    ///
    /// # Errors
    ///
    /// Returns `RegionError::DimensionMismatch` if `point.len() != self.dim()`.
    fn boundary_distance(&self, point: &[f32]) -> Result<f32, RegionError>;

    /// Volume of the region.
    fn volume(&self) -> f32;

    /// Centroid (center of mass) of the region.
    fn center(&self) -> Vec<f32>;

    /// Dimensionality of the ambient space.
    fn dim(&self) -> usize;

    /// RegD depth dissimilarity between two regions.
    ///
    /// Incorporates region size through a length-scale normalization of volume,
    /// emulating hyperbolic distance's exponential growth with depth in a hierarchy.
    ///
    /// ## Formula
    ///
    /// ```text
    /// depth_d(r1, r2) = d(center(r1), center(r2)) / (vol(r1)^(1/d) + vol(r2)^(1/d))
    /// ```
    ///
    /// where `d` is the Euclidean distance between centers and `vol^(1/d)` normalizes
    /// volume to a characteristic length scale. Smaller regions (deeper in a hierarchy)
    /// produce larger dissimilarity for the same center distance.
    ///
    /// # Errors
    ///
    /// Returns `RegionError::DimensionMismatch` if regions have different dimensionality.
    fn depth_dissimilarity(&self, other: &dyn Region) -> Result<f32, RegionError> {
        if self.dim() != other.dim() {
            return Err(RegionError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let c1 = self.center();
        let c2 = other.center();
        let d = self.dim() as f32;

        // Euclidean distance between centers.
        let center_dist = c1
            .iter()
            .zip(c2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        // Length-scale normalization: vol^(1/d).
        let scale1 = self.volume().max(0.0).powf(1.0 / d);
        let scale2 = other.volume().max(0.0).powf(1.0 / d);

        let denom = scale1 + scale2;
        if denom < 1e-12 {
            // Both regions are degenerate; fall back to raw center distance.
            return Ok(center_dist);
        }

        Ok(center_dist / denom)
    }

    /// RegD boundary dissimilarity between two regions.
    ///
    /// Measures the asymmetric "protrusion" of one region beyond the other's boundary,
    /// capturing set-inclusion relationships. When r1 is fully inside r2, the
    /// dissimilarity is zero from the inclusion direction.
    ///
    /// ## Formula
    ///
    /// ```text
    /// boundary_d(r1, r2) = max(0,
    ///     boundary_distance(r1, closest_point(r2, center(r1)))
    ///   - boundary_distance(r2, closest_point(r1, center(r2)))
    /// )
    /// ```
    ///
    /// The default implementation uses the centers directly (a practical approximation
    /// when closest-point projection is not available for arbitrary region pairs).
    ///
    /// # Errors
    ///
    /// Returns `RegionError::DimensionMismatch` if regions have different dimensionality.
    fn boundary_dissimilarity(&self, other: &dyn Region) -> Result<f32, RegionError> {
        if self.dim() != other.dim() {
            return Err(RegionError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }

        let c1 = self.center();
        let c2 = other.center();

        // Default approximation: use the other region's center as the query point.
        // signed distance of other's center w.r.t. self's boundary.
        let bd_self_at_c2 = self.boundary_distance(&c2)?;
        // signed distance of self's center w.r.t. other's boundary.
        let bd_other_at_c1 = other.boundary_distance(&c1)?;

        // Positive bd means point is outside. The asymmetric protrusion is
        // how much more r1 "sticks out" of r2 than r2 sticks out of r1.
        Ok((bd_self_at_c2 - bd_other_at_c1).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal region for unit-testing the default trait methods.
    struct UnitBall {
        center: Vec<f32>,
        radius: f32,
    }

    impl Region for UnitBall {
        fn contains(&self, point: &[f32]) -> Result<bool, RegionError> {
            if point.len() != self.center.len() {
                return Err(RegionError::DimensionMismatch {
                    expected: self.center.len(),
                    actual: point.len(),
                });
            }
            let dist_sq: f32 = self
                .center
                .iter()
                .zip(point)
                .map(|(c, p)| (c - p) * (c - p))
                .sum();
            Ok(dist_sq.sqrt() <= self.radius)
        }

        fn boundary_distance(&self, point: &[f32]) -> Result<f32, RegionError> {
            if point.len() != self.center.len() {
                return Err(RegionError::DimensionMismatch {
                    expected: self.center.len(),
                    actual: point.len(),
                });
            }
            let dist: f32 = self
                .center
                .iter()
                .zip(point)
                .map(|(c, p)| (c - p) * (c - p))
                .sum::<f32>()
                .sqrt();
            Ok(dist - self.radius)
        }

        fn volume(&self) -> f32 {
            // 2D circle area for simplicity in tests.
            std::f32::consts::PI * self.radius * self.radius
        }

        fn center(&self) -> Vec<f32> {
            self.center.clone()
        }

        fn dim(&self) -> usize {
            self.center.len()
        }
    }

    #[test]
    fn depth_dissimilarity_identical_regions_is_zero() {
        let a = UnitBall {
            center: vec![0.0, 0.0],
            radius: 1.0,
        };
        let d = a.depth_dissimilarity(&a).unwrap();
        assert!(d.abs() < 1e-6, "Identical regions should have zero depth dissimilarity, got {}", d);
    }

    #[test]
    fn depth_dissimilarity_increases_for_smaller_regions() {
        // Two pairs of regions at the same center distance, but one pair is smaller.
        let big_a = UnitBall { center: vec![0.0, 0.0], radius: 2.0 };
        let big_b = UnitBall { center: vec![1.0, 0.0], radius: 2.0 };
        let small_a = UnitBall { center: vec![0.0, 0.0], radius: 0.1 };
        let small_b = UnitBall { center: vec![1.0, 0.0], radius: 0.1 };

        let d_big = big_a.depth_dissimilarity(&big_b).unwrap();
        let d_small = small_a.depth_dissimilarity(&small_b).unwrap();

        assert!(
            d_small > d_big,
            "Smaller regions should have larger depth dissimilarity: small={}, big={}",
            d_small, d_big
        );
    }

    #[test]
    fn boundary_dissimilarity_contained_is_zero() {
        // Small ball inside big ball: the small ball doesn't protrude.
        let big = UnitBall { center: vec![0.0, 0.0], radius: 5.0 };
        let small = UnitBall { center: vec![0.0, 0.0], radius: 1.0 };

        let d = small.boundary_dissimilarity(&big).unwrap();
        assert!(d < 1e-6, "Contained region should have ~zero boundary dissimilarity, got {}", d);
    }

    #[test]
    fn boundary_dissimilarity_protruding_is_positive() {
        // Two balls of same size, offset so they partially overlap.
        let a = UnitBall { center: vec![0.0, 0.0], radius: 1.0 };
        let b = UnitBall { center: vec![3.0, 0.0], radius: 1.0 };

        // At least one direction should show positive dissimilarity.
        let d_ab = a.boundary_dissimilarity(&b).unwrap();
        let d_ba = b.boundary_dissimilarity(&a).unwrap();
        assert!(
            d_ab > 0.0 || d_ba > 0.0,
            "Non-contained regions should have positive boundary dissimilarity in at least one direction"
        );
    }

    #[test]
    fn dimension_mismatch_errors() {
        let a = UnitBall { center: vec![0.0, 0.0], radius: 1.0 };
        let b = UnitBall { center: vec![0.0, 0.0, 0.0], radius: 1.0 };
        assert!(a.depth_dissimilarity(&b).is_err());
        assert!(a.boundary_dissimilarity(&b).is_err());
    }
}
