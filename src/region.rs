//! A unified [`Region`] trait over subsume's geometries.
//!
//! Every geometry in this crate encodes subsumption the same way at the
//! conceptual level: a region `outer` subsumes `inner` to the degree that
//! `outer` geometrically CONTAINS `inner`. `Region` names that shared contract
//! so callers can write code generic over the geometry.
//!
//! Following the geometric-embedding literature, "region" is the umbrella term
//! (e.g. Yang & Chen 2025, "Hierarchical Embeddings via Dissimilarity between
//! Arbitrary Euclidean Regions"); the box geometry is [`crate::HyperBox`], one
//! implementor among balls, ellipsoids, and subspaces.
//!
//! # The score is per-geometry, not cross-geometry calibrated
//!
//! Each geometry's containment closed form lives on its own scale. A box uses a
//! volume-ratio probability (a box fully contains itself, score 1.0); a ball
//! uses a sigmoid of the containment margin (a ball "contains itself" at the
//! decision boundary, score 0.5); an ellipsoid uses a sigmoid of `-k * KL`. So
//! [`Region::subsumption_score`] is **monotone within a single geometry**
//! (more-contained scores higher) but the raw values are **not comparable
//! across geometries**. Use it to rank or threshold candidates of one geometry,
//! not to compare a ball's score against a box's. This reflects the literature:
//! these are distinct models that share the containment *concept*, not a common
//! calibrated measure.
//!
//! The trait is deliberately minimal: it captures only the subsumption contract
//! all geometries genuinely share. Geometry-specific operations (volume,
//! intersection, bounds) stay on the concrete types and on [`crate::HyperBox`].
//! Geometries whose score takes a softness temperature `k` (balls, ellipsoids)
//! use [`Region::DEFAULT_K`] through this trait; call their free functions
//! directly for a custom temperature.

use crate::box_trait::BoxError;
use crate::HyperBox;

/// A geometric region that scores subsumption via containment.
///
/// `self` is the (more general) container; the argument is the (more specific)
/// contained region. See the [module docs](self) for the crucial caveat that
/// the score is monotone within a geometry but not calibrated across geometries.
pub trait Region: Sized {
    /// Default softness temperature for geometries that take one as a parameter.
    const DEFAULT_K: f32 = 1.0;

    /// Number of ambient dimensions.
    fn dim(&self) -> usize;

    /// Monotone subsumption score for `inner ⊆ self`: higher means `self`
    /// (the more general region) subsumes `inner` more strongly. In `[0, 1]`.
    ///
    /// Comparable WITHIN one geometry, not across geometries (see module docs).
    fn subsumption_score(&self, inner: &Self) -> Result<f32, BoxError>;
}

// Box geometries: delegate to the richer `HyperBox` trait. A blanket
// `impl<T: HyperBox> Region for T` is rejected by coherence (the compiler cannot
// prove other concrete `Region` types are not `HyperBox`), so each box type
// implements `Region` explicitly via this macro. Using a distinct method name
// (`subsumption_score`) also avoids colliding with `HyperBox::containment_prob`.
macro_rules! impl_region_via_hyperbox {
    ($ty:ty) => {
        impl Region for $ty {
            fn dim(&self) -> usize {
                HyperBox::dim(self)
            }
            fn subsumption_score(&self, inner: &Self) -> Result<f32, BoxError> {
                HyperBox::containment_prob(self, inner)
            }
        }
    };
}

impl_region_via_hyperbox!(crate::ndarray_backend::NdarrayBox);
impl_region_via_hyperbox!(crate::ndarray_backend::NdarrayGumbelBox);
#[cfg(feature = "candle-backend")]
impl_region_via_hyperbox!(crate::candle_backend::CandleBox);
#[cfg(feature = "candle-backend")]
impl_region_via_hyperbox!(crate::candle_backend::CandleGumbelBox);

impl Region for crate::Ball {
    fn dim(&self) -> usize {
        self.dim()
    }
    fn subsumption_score(&self, inner: &Self) -> Result<f32, BoxError> {
        // ball::containment_prob(inner, outer, k): inner is the contained.
        crate::ball::containment_prob(inner, self, Self::DEFAULT_K)
    }
}

impl Region for crate::Ellipsoid {
    fn dim(&self) -> usize {
        self.dim()
    }
    fn subsumption_score(&self, inner: &Self) -> Result<f32, BoxError> {
        // ellipsoid::containment_prob(child, parent, k): child is the contained.
        crate::ellipsoid::containment_prob(inner, self, Self::DEFAULT_K)
    }
}

impl Region for crate::Subspace {
    fn dim(&self) -> usize {
        self.dim()
    }
    fn subsumption_score(&self, inner: &Self) -> Result<f32, BoxError> {
        // subspace::containment_score(a, b) scores a ⊆ b, so a is the contained.
        crate::subspace::containment_score(inner, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A function generic over Region: the whole point of the trait. Ranks a set
    // of candidate inner regions by how strongly `outer` subsumes each.
    fn most_subsumed<'a, R: Region>(outer: &R, candidates: &'a [R]) -> &'a R {
        candidates
            .iter()
            .max_by(|a, b| {
                outer
                    .subsumption_score(a)
                    .unwrap()
                    .total_cmp(&outer.subsumption_score(b).unwrap())
            })
            .unwrap()
    }

    #[test]
    fn ball_subsumption_is_monotone() {
        let outer = crate::Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let inside = crate::Ball::new(vec![0.0, 0.0], 0.3).unwrap(); // deep inside
        let straddling = crate::Ball::new(vec![1.9, 0.0], 1.0).unwrap(); // pokes out
        assert_eq!(Region::dim(&outer), 2);
        let s_in = outer.subsumption_score(&inside).unwrap();
        let s_out = outer.subsumption_score(&straddling).unwrap();
        assert!(
            s_in > s_out,
            "a deeply-contained ball must score higher than a straddling one: {s_in} vs {s_out}"
        );
        // Generic ranking picks the contained one.
        let cands = vec![straddling, inside];
        let best = most_subsumed(&outer, &cands);
        assert!(
            (best.radius() - 0.3).abs() < 1e-6,
            "ranking should pick the inside ball"
        );
    }

    #[test]
    fn ball_region_matches_free_function() {
        let outer = crate::Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let inner = crate::Ball::new(vec![0.0, 0.0], 0.5).unwrap();
        let via_trait = outer.subsumption_score(&inner).unwrap();
        let via_free =
            crate::ball::containment_prob(&inner, &outer, <crate::Ball as Region>::DEFAULT_K)
                .unwrap();
        assert_eq!(via_trait, via_free);
    }

    #[test]
    fn ndarray_box_is_a_region() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;
        let outer = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let inner = NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap();
        let disjoint = NdarrayBox::new(array![5.0, 5.0], array![6.0, 6.0], 1.0).unwrap();
        // Reached purely through the generic Region interface (no ambiguity:
        // only Region is in scope as the bound, and the method name is distinct
        // from HyperBox::containment_prob).
        fn score<R: Region>(outer: &R, inner: &R) -> (usize, f32) {
            (outer.dim(), outer.subsumption_score(inner).unwrap())
        }
        let (d, p_in) = score(&outer, &inner);
        let (_, p_out) = score(&outer, &disjoint);
        assert_eq!(d, 2);
        assert!(
            p_in > p_out,
            "contained box should outscore disjoint: {p_in} vs {p_out}"
        );
    }
}
