//! Annular sector embeddings for knowledge graph completion.
//!
//! Each concept is represented as an annular sector in the complex plane:
//! a ring-shaped region bounded by inner/outer radii and an angular span.
//! Subsumption is modeled via radial and angular containment.
//!
//! # Motivation
//!
//! Annular sectors combine rotation-based relations (like RotatE) with
//! region uncertainty. The annular (ring) structure naturally models
//! exclusion zones — entities that are "too close" or "too far" from
//! the relation center. This handles 1-N, N-1, N-N relations that
//! boxes and balls struggle with.
//!
//! # Containment
//!
//! Sector A is contained in sector B iff:
//! - `r_B_inner <= r_A_inner` (B's hole is smaller)
//! - `r_A_outer <= r_B_outer` (A fits inside B's outer boundary)
//! - Angular interval of A is contained in B's angular interval
//!
//! # References
//!
//! - Zhu & Zeng (2025), "Annular Sector Embeddings for Knowledge Graph
//!   Completion" (arXiv:2506.11099)

use crate::BoxError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An annular sector in the complex plane.
///
/// Bounded by inner radius `r_inner`, outer radius `r_outer`,
/// and angular span `[theta_start, theta_end]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnularSector {
    center_re: f32,
    center_im: f32,
    r_inner: f32,
    r_outer: f32,
    theta_start: f32,
    theta_end: f32,
}

/// A relation transform for annular sectors.
///
/// Applies rotation and radial scaling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnularRelation {
    rotation: f32,
    radial_scale: f32,
    angular_scale: f32,
}

/// Collection of annular sector embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnularEmbedding {
    entities: Vec<AnnularSector>,
    relations: Vec<AnnularRelation>,
}

// ---------------------------------------------------------------------------
// AnnularSector
// ---------------------------------------------------------------------------

impl AnnularSector {
    /// Create a new annular sector.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if radii are invalid, angular span
    /// is invalid, or any value is non-finite.
    pub fn new(
        center_re: f32,
        center_im: f32,
        r_inner: f32,
        r_outer: f32,
        theta_start: f32,
        theta_end: f32,
    ) -> Result<Self, BoxError> {
        if !center_re.is_finite() || !center_im.is_finite() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: center_re as f64,
            });
        }
        if !r_inner.is_finite() || r_inner < 0.0 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: r_inner as f64,
            });
        }
        if !r_outer.is_finite() || r_outer <= r_inner {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: r_inner as f64,
                max: r_outer as f64,
            });
        }
        if !theta_start.is_finite() || !theta_end.is_finite() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: theta_start as f64,
            });
        }
        // Normalize angles to [0, 2*pi)
        let two_pi = 2.0 * std::f32::consts::PI;
        let theta_start = theta_start.rem_euclid(two_pi);
        let theta_end = theta_end.rem_euclid(two_pi);

        Ok(Self {
            center_re,
            center_im,
            r_inner,
            r_outer,
            theta_start,
            theta_end,
        })
    }

    /// Create from polar center representation.
    pub fn from_polar(
        center_r: f32,
        center_theta: f32,
        r_inner: f32,
        r_outer: f32,
        theta_start: f32,
        theta_end: f32,
    ) -> Result<Self, BoxError> {
        let center_re = center_r * center_theta.cos();
        let center_im = center_r * center_theta.sin();
        Self::new(
            center_re,
            center_im,
            r_inner,
            r_outer,
            theta_start,
            theta_end,
        )
    }

    #[must_use]
    pub fn center_re(&self) -> f32 {
        self.center_re
    }

    #[must_use]
    pub fn center_im(&self) -> f32 {
        self.center_im
    }

    #[must_use]
    pub fn r_inner(&self) -> f32 {
        self.r_inner
    }

    #[must_use]
    pub fn r_outer(&self) -> f32 {
        self.r_outer
    }

    #[must_use]
    pub fn theta_start(&self) -> f32 {
        self.theta_start
    }

    #[must_use]
    pub fn theta_end(&self) -> f32 {
        self.theta_end
    }

    /// Angular span in radians.
    #[must_use]
    pub fn angular_span(&self) -> f32 {
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut span = self.theta_end - self.theta_start;
        if span < 0.0 {
            span += two_pi;
        }
        span
    }

    /// Radial width.
    #[must_use]
    pub fn radial_width(&self) -> f32 {
        self.r_outer - self.r_inner
    }

    /// Area proxy: angular_span * (r_outer^2 - r_inner^2).
    #[must_use]
    pub fn area_proxy(&self) -> f32 {
        self.angular_span() * (self.r_outer * self.r_outer - self.r_inner * self.r_inner)
    }
}

// ---------------------------------------------------------------------------
// AnnularRelation
// ---------------------------------------------------------------------------

impl AnnularRelation {
    /// Create a new annular relation.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if scales are non-positive or non-finite.
    pub fn new(rotation: f32, radial_scale: f32, angular_scale: f32) -> Result<Self, BoxError> {
        if !rotation.is_finite() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: rotation as f64,
            });
        }
        if !radial_scale.is_finite() || radial_scale <= 0.0 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: radial_scale as f64,
            });
        }
        if !angular_scale.is_finite() || angular_scale <= 0.0 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: angular_scale as f64,
            });
        }
        Ok(Self {
            rotation,
            radial_scale,
            angular_scale,
        })
    }

    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: 0.0,
            radial_scale: 1.0,
            angular_scale: 1.0,
        }
    }

    /// Apply this relation to a sector.
    pub fn apply(&self, sector: &AnnularSector) -> AnnularSector {
        let two_pi = 2.0 * std::f32::consts::PI;

        // Rotate center
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        let new_center_re = sector.center_re * cos_r - sector.center_im * sin_r;
        let new_center_im = sector.center_re * sin_r + sector.center_im * cos_r;

        // Scale radii
        let new_r_inner = sector.r_inner * self.radial_scale;
        let new_r_outer = sector.r_outer * self.radial_scale;

        // Scale angular span around the midpoint
        let mid = (sector.theta_start + sector.theta_end) / 2.0;
        let half_span = sector.angular_span() * self.angular_scale / 2.0;
        let new_theta_start = (mid - half_span).rem_euclid(two_pi);
        let new_theta_end = (mid + half_span).rem_euclid(two_pi);

        AnnularSector {
            center_re: new_center_re,
            center_im: new_center_im,
            r_inner: new_r_inner,
            r_outer: new_r_outer,
            theta_start: new_theta_start,
            theta_end: new_theta_end,
        }
    }
}

// ---------------------------------------------------------------------------
// AnnularEmbedding
// ---------------------------------------------------------------------------

impl AnnularEmbedding {
    /// Create a new annular embedding model.
    pub fn new(entities: Vec<AnnularSector>, relations: Vec<AnnularRelation>) -> Self {
        Self {
            entities,
            relations,
        }
    }

    #[must_use]
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    #[must_use]
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }

    pub fn entities(&self) -> &[AnnularSector] {
        &self.entities
    }

    pub fn relations(&self) -> &[AnnularRelation] {
        &self.relations
    }
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// Check if sector A is contained in sector B.
///
/// Returns a containment score in [0, 1]: 1 means A ⊆ B.
///
/// Containment requires:
/// - `r_B_inner <= r_A_inner` (B's hole is smaller or equal)
/// - `r_A_outer <= r_B_outer` (A fits inside B's outer boundary)
/// - Angular interval of A is contained in B's
pub fn containment_score(inner: &AnnularSector, outer: &AnnularSector) -> f32 {
    // Radial containment
    let radial_inner = if outer.r_inner <= inner.r_inner + 1e-6 {
        1.0
    } else {
        (inner.r_inner / outer.r_inner).max(0.0)
    };
    let radial_outer = if inner.r_outer <= outer.r_outer + 1e-6 {
        1.0
    } else {
        (outer.r_outer / inner.r_outer).max(0.0)
    };
    let radial_score = radial_inner.min(radial_outer);

    // Angular containment
    let angular_score = angular_containment(
        inner.theta_start,
        inner.theta_end,
        outer.theta_start,
        outer.theta_end,
    );

    radial_score.min(angular_score)
}

/// Angular containment: is interval A contained in interval B (on the circle)?
fn angular_containment(a_start: f32, a_end: f32, b_start: f32, b_end: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;

    // Normalize all angles
    let a_start = a_start.rem_euclid(two_pi);
    let a_end = a_end.rem_euclid(two_pi);
    let b_start = b_start.rem_euclid(two_pi);
    let b_end = b_end.rem_euclid(two_pi);

    let a_span = if a_end >= a_start {
        a_end - a_start
    } else {
        a_end + two_pi - a_start
    };
    let b_span = if b_end >= b_start {
        b_end - b_start
    } else {
        b_end + two_pi - b_start
    };

    if a_span <= b_span + 1e-6 {
        // A could fit inside B angularly
        // Check if A's start is within B's interval
        let a_in_b = angle_in_interval(a_start, b_start, b_end);
        if a_in_b {
            // A's end should also be in B (since span fits)
            let a_end_in_b = angle_in_interval(a_end, b_start, b_end);
            if a_end_in_b {
                return 1.0;
            }
        }
    }

    // Partial containment: ratio of overlap to A's span
    let overlap = angular_overlap(a_start, a_end, b_start, b_end);
    if a_span < 1e-6 {
        return 1.0;
    }
    (overlap / a_span).min(1.0)
}

/// Check if angle `a` is in the interval [start, end] (circular).
fn angle_in_interval(a: f32, start: f32, end: f32) -> bool {
    if end >= start {
        a >= start - 1e-6 && a <= end + 1e-6
    } else {
        // Wraps around
        a >= start - 1e-6 || a <= end + 1e-6
    }
}

/// Compute angular overlap between two intervals.
fn angular_overlap(a_start: f32, a_end: f32, b_start: f32, b_end: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let a_span = if a_end >= a_start {
        a_end - a_start
    } else {
        a_end + two_pi - a_start
    };
    let b_span = if b_end >= b_start {
        b_end - b_start
    } else {
        b_end + two_pi - b_start
    };

    // Simple approximation: min of spans if they could overlap
    // For exact computation, we'd need to handle wraparound cases
    if a_span < 1e-6 || b_span < 1e-6 {
        return 0.0;
    }

    // Check if intervals overlap at all
    let a_mid = (a_start + a_end) / 2.0;
    let b_mid = (b_start + b_end) / 2.0;
    let dist = (a_mid - b_mid).abs().min(two_pi - (a_mid - b_mid).abs());

    if dist > (a_span + b_span) / 2.0 {
        return 0.0;
    }

    // Approximate overlap
    ((a_span + b_span) / 2.0 - dist)
        .max(0.0)
        .min(a_span)
        .min(b_span)
}

/// Surface distance between two annular sectors.
///
/// Combines radial gap and angular gap.
pub fn surface_distance(a: &AnnularSector, b: &AnnularSector) -> f32 {
    // Radial distance
    let radial_dist = if a.r_outer < b.r_inner {
        b.r_inner - a.r_outer
    } else if b.r_outer < a.r_inner {
        a.r_inner - b.r_outer
    } else {
        0.0
    };

    // Angular distance
    let two_pi = 2.0 * std::f32::consts::PI;
    let a_mid = (a.theta_start + a.theta_end) / 2.0;
    let b_mid = (b.theta_start + b.theta_end) / 2.0;
    let angular_dist = (a_mid - b_mid).abs().min(two_pi - (a_mid - b_mid).abs());

    // Center distance
    let center_dist =
        ((a.center_re - b.center_re).powi(2) + (a.center_im - b.center_im).powi(2)).sqrt();

    radial_dist + angular_dist + center_dist
}

/// Score a triple (head, relation, tail) using annular sector embeddings.
///
/// Transforms the head sector by the relation, then computes surface distance.
pub fn score_triple(head: &AnnularSector, relation: &AnnularRelation, tail: &AnnularSector) -> f32 {
    let transformed = relation.apply(head);
    surface_distance(&transformed, tail)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn sector_new_valid() {
        let s = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        assert!((s.r_inner() - 0.5).abs() < 1e-6);
        assert!((s.r_outer() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn sector_from_polar() {
        let s = AnnularSector::from_polar(1.0, PI / 4.0, 0.5, 1.5, 0.0, PI).unwrap();
        assert!((s.center_re() - 1.0 * (PI / 4.0).cos()).abs() < 1e-5);
        assert!((s.center_im() - 1.0 * (PI / 4.0).sin()).abs() < 1e-5);
    }

    #[test]
    fn sector_rejects_invalid_radii() {
        assert!(AnnularSector::new(0.0, 0.0, 1.0, 0.5, 0.0, PI).is_err()); // r_inner > r_outer
        assert!(AnnularSector::new(0.0, 0.0, -0.1, 1.0, 0.0, PI).is_err()); // negative r_inner
    }

    #[test]
    fn sector_rejects_non_finite() {
        assert!(AnnularSector::new(f32::NAN, 0.0, 0.5, 1.5, 0.0, PI).is_err());
        assert!(AnnularSector::new(0.0, 0.0, f32::INFINITY, 1.5, 0.0, PI).is_err());
    }

    // --- Angular span ---

    #[test]
    fn angular_span_simple() {
        let s = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        assert!((s.angular_span() - PI).abs() < 1e-6);
    }

    #[test]
    fn angular_span_wrapping() {
        let s = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 1.5 * PI, 0.5 * PI).unwrap();
        assert!((s.angular_span() - PI).abs() < 1e-5);
    }

    // --- Radial width ---

    #[test]
    fn radial_width() {
        let s = AnnularSector::new(0.0, 0.0, 0.5, 2.0, 0.0, PI).unwrap();
        assert!((s.radial_width() - 1.5).abs() < 1e-6);
    }

    // --- Containment ---

    #[test]
    fn containment_identical_is_one() {
        let s = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let score = containment_score(&s, &s);
        assert!(score > 0.99, "identical containment = {score}, expected ~1");
    }

    #[test]
    fn containment_nested_is_one() {
        let inner = AnnularSector::new(0.0, 0.0, 0.7, 1.3, 0.2, PI - 0.2).unwrap();
        let outer = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let score = containment_score(&inner, &outer);
        assert!(score > 0.99, "nested containment = {score}, expected ~1");
    }

    #[test]
    fn containment_disjoint_is_zero() {
        let a = AnnularSector::new(0.0, 0.0, 0.5, 1.0, 0.0, 0.5).unwrap();
        let b = AnnularSector::new(0.0, 0.0, 2.0, 3.0, PI, PI + 0.5).unwrap();
        let score = containment_score(&a, &b);
        assert!(score < 0.01, "disjoint containment = {score}, expected ~0");
    }

    // --- Surface distance ---

    #[test]
    fn surface_distance_identical_is_zero() {
        let s = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let d = surface_distance(&s, &s);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn surface_distance_different_centers() {
        let a = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let b = AnnularSector::new(5.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let d = surface_distance(&a, &b);
        assert!(d > 4.0, "different center distance = {d}, expected > 4");
    }

    // --- Relation ---

    #[test]
    fn relation_identity_preserves_sector() {
        let s = AnnularSector::new(1.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let r = AnnularRelation::identity();
        let t = r.apply(&s);
        assert!((t.center_re() - 1.0).abs() < 1e-6);
        assert!((t.center_im()).abs() < 1e-6);
        assert!((t.r_inner() - 0.5).abs() < 1e-6);
        assert!((t.r_outer() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn relation_rotation() {
        let s = AnnularSector::new(1.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let r = AnnularRelation::new(PI / 2.0, 1.0, 1.0).unwrap();
        let t = r.apply(&s);
        // [1,0] rotated 90° → [0,1]
        assert!(t.center_re().abs() < 1e-5);
        assert!((t.center_im() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn relation_radial_scaling() {
        let s = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let r = AnnularRelation::new(0.0, 2.0, 1.0).unwrap();
        let t = r.apply(&s);
        assert!((t.r_inner() - 1.0).abs() < 1e-5);
        assert!((t.r_outer() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn relation_rejects_zero_scale() {
        assert!(AnnularRelation::new(0.0, 0.0, 1.0).is_err());
        assert!(AnnularRelation::new(0.0, 1.0, 0.0).is_err());
    }

    // --- Triple scoring ---

    #[test]
    fn score_triple_perfect_match() {
        let h = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let r = AnnularRelation::identity();
        let t = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let s = score_triple(&h, &r, &t);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn score_triple_mismatch() {
        let h = AnnularSector::new(0.0, 0.0, 0.5, 1.0, 0.0, 0.5).unwrap();
        let r = AnnularRelation::identity();
        let t = AnnularSector::new(10.0, 0.0, 2.0, 3.0, PI, PI + 0.5).unwrap();
        let s = score_triple(&h, &r, &t);
        assert!(s > 5.0, "mismatch score = {s}, expected > 5");
    }

    // --- Area proxy ---

    #[test]
    fn area_proxy_positive() {
        let s = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap();
        let a = s.area_proxy();
        assert!(a > 0.0, "area proxy = {a}, expected > 0");
    }

    // --- Model ---

    #[test]
    fn model_construction() {
        let entities = vec![
            AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, PI).unwrap(),
            AnnularSector::new(1.0, 0.0, 0.3, 1.0, 0.0, PI / 2.0).unwrap(),
        ];
        let relations = vec![AnnularRelation::identity()];
        let model = AnnularEmbedding::new(entities, relations);
        assert_eq!(model.num_entities(), 2);
        assert_eq!(model.num_relations(), 1);
    }

    #[test]
    fn angular_containment_full() {
        let score = angular_containment(0.2, 0.8, 0.0, PI);
        assert!(score > 0.99, "full containment = {score}");
    }

    #[test]
    fn angular_containment_partial() {
        let score = angular_containment(0.0, PI, PI / 2.0, 1.5 * PI);
        assert!(score > 0.0 && score < 1.0, "partial containment = {score}");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::f32::consts::PI;

    fn arb_sector() -> impl Strategy<Value = AnnularSector> {
        (
            -10.0f32..10.0,
            -10.0f32..10.0,
            0.01f32..5.0,
            0.01f32..5.0,
            0.0f32..(2.0 * PI),
            0.0f32..(2.0 * PI),
        )
            .prop_filter_map("valid sector", |(cre, cim, ri, ro, ts, te)| {
                let r_inner = ri.min(ro);
                let r_outer = ri.max(ro);
                if (r_outer - r_inner) < 0.01 {
                    return None;
                }
                AnnularSector::new(cre, cim, r_inner, r_outer, ts, te).ok()
            })
    }

    fn arb_sector_pair() -> impl Strategy<Value = (AnnularSector, AnnularSector)> {
        (arb_sector(), arb_sector())
    }

    fn arb_relation() -> impl Strategy<Value = AnnularRelation> {
        (-PI..PI, 0.1f32..5.0, 0.1f32..3.0)
            .prop_map(|(rot, rs, as_)| AnnularRelation::new(rot, rs, as_).unwrap())
    }

    proptest! {
        #[test]
        fn prop_containment_in_unit_interval(
            (a, b) in arb_sector_pair()
        ) {
            let s = containment_score(&a, &b);
            prop_assert!(s >= -1e-6, "containment_score < 0: {s}");
            prop_assert!(s <= 1.0 + 1e-6, "containment_score > 1: {s}");
        }

        #[test]
        fn prop_surface_distance_nonneg(
            (a, b) in arb_sector_pair()
        ) {
            let d = surface_distance(&a, &b);
            prop_assert!(d >= -1e-6, "surface_distance < 0: {d}");
        }

        #[test]
        fn prop_surface_distance_symmetric(
            (a, b) in arb_sector_pair()
        ) {
            let d_ab = surface_distance(&a, &b);
            let d_ba = surface_distance(&b, &a);
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-3,
                "surface_distance should be symmetric: {d_ab} != {d_ba}"
            );
        }

        #[test]
        fn prop_area_proxy_nonneg(
            s in arb_sector()
        ) {
            let a = s.area_proxy();
            prop_assert!(a >= -1e-6, "area_proxy < 0: {a}");
        }

        #[test]
        fn prop_score_triple_nonneg(
            h in arb_sector(),
            r in arb_relation(),
            t in arb_sector()
        ) {
            let s = score_triple(&h, &r, &t);
            prop_assert!(s >= -1e-6, "score_triple < 0: {s}");
        }

        #[test]
        fn prop_angular_span_positive(
            s in arb_sector()
        ) {
            let span = s.angular_span();
            prop_assert!(span >= 0.0, "angular_span < 0: {span}");
            prop_assert!(span <= 2.0 * PI + 1e-5, "angular_span > 2pi: {span}");
        }
    }
}
