//! BoxE-inspired box-to-box scoring for knowledge base completion.
//!
//! Inspired by the BoxE model from **Abboud et al. (NeurIPS 2020)**:
//! "BoxE: A Box Embedding Model for Knowledge Base Completion"
//!
//! **Paper**: [arXiv:2007.06267](https://arxiv.org/abs/2007.06267) | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/6dbbe6abe5f14af882ff977fc3f35501-Abstract.html)
//!
//! **Note**: The original BoxE paper represents entities as **points** and relations as
//! **boxes**, with a polynomial inside/outside distance function. This implementation
//! departs from the paper: entities are represented as **boxes** and scoring uses
//! volume-ratio containment (`intersection_volume / tail_volume`). This is a box-to-box
//! containment model with translational bumps, not a faithful reproduction of BoxE.
//!
//! # Intuitive Explanation
//!
//! ## Paradigm Problem: Modeling Knowledge Graph Triples
//!
//! Consider the knowledge graph triple (Paris, located_in, France). We want to check whether
//! this relationship is true. This module's approach:
//!
//! 1. **Start with head entity**: "Paris" is represented as a box
//! 2. **Apply relation transformation**: The "located_in" relation has a "bump" vector that
//!    translates the Paris box to a new position
//! 3. **Check containment**: If the translated Paris box contains the "France" box, then
//!    the triple is likely true
//!
//! **Visual analogy**: Think of the relation as a "magnet" that pulls the head entity box
//! toward the tail entity. If the magnet is strong enough (the bump is right), the head box
//! will contain the tail box after translation.
//!
//! **Why different relations need different bumps**: The relation "capital_of" might require
//! a different translation than "located_in". Paris is the capital of France, but many cities
//! are located in France without being the capital. The model learns relation-specific
//! geometric patterns through these translational bumps.
//!
//! **Research foundation**: This approach is introduced in **Abboud et al. (2020)**, Section 3.1,
//! as an adaptation of TransE's translational approach (Bordes et al., 2013) to box embeddings.
//!
//! # Mathematical Formulation
//!
//! For a triple \((h, r, t)\):
//!
//! 1. **Apply relation bump**: \(h' = h + \text{bump}_r\)
//! 2. **Score**: \(P(t \subseteq h')\)
//!
//! The score is the containment probability: how much of the tail box fits inside the
//! translated head box. Higher scores mean the triple is more likely to be true.
//!
//! BoxE uses "translational bumps" - relation-specific translation vectors
//! that transform boxes for scoring knowledge graph triples.

use crate::BoxError;

/// Translational bump for a relation.
///
/// In BoxE, each relation has a "bump" vector that translates boxes
/// to model relation-specific transformations.
#[derive(Debug, Clone)]
pub struct Bump {
    /// Translation vector for this relation
    pub translation: Vec<f32>,
}

impl Bump {
    /// Create a new bump with the given translation vector.
    pub fn new(translation: Vec<f32>) -> Self {
        Self { translation }
    }

    /// Get the dimension of the bump.
    pub fn dim(&self) -> usize {
        self.translation.len()
    }
}

/// BoxE scoring function.
///
/// Computes the score for a triple (head, relation, tail) using BoxE's
/// translational bump model.
///
/// **Reference**: Abboud et al. (2020), Section 3.1 - Translational Bumps
///
/// # Intuitive Explanation
///
/// Think of this as a geometric matching problem: we translate the head box by the
/// relation's "bump" vector, then check how well the tail box fits inside the translated
/// head box. If they align well (high containment), the triple is likely true.
///
/// **Example**: For (Paris, located_in, France):
/// - Start with Paris box at some position
/// - Apply "located_in" bump to translate it
/// - Check if France box is contained in the translated Paris box
/// - High containment → triple is likely true
///
/// # Mathematical Formulation
///
/// For a triple \((h, r, t)\):
///
/// 1. **Apply relation bump**: \(h' = h + \text{bump}_r\)
/// 2. **Score**: \(P(t \subseteq h') = \frac{\text{Vol}(h' \cap t)}{\text{Vol}(t)}\)
///
/// The score is the containment probability: the fraction of the tail box that overlaps
/// with the translated head box. This ranges from 0 (disjoint) to 1 (complete containment).
///
/// # Parameters
///
/// * `head_min`: Minimum bounds of head entity box
/// * `head_max`: Maximum bounds of head entity box
/// * `tail_min`: Minimum bounds of tail entity box
/// * `tail_max`: Maximum bounds of tail entity box
/// * `bump`: Translation vector for the relation
/// * `temperature`: Temperature for probabilistic operations
///
/// # Returns
///
/// Score in [0, 1], where higher means more likely the triple is true.
pub fn boxe_score(
    head_min: &[f32],
    head_max: &[f32],
    tail_min: &[f32],
    tail_max: &[f32],
    bump: &[f32],
    _temperature: f32,
) -> Result<f32, BoxError> {
    if head_min.len() != head_max.len()
        || tail_min.len() != tail_max.len()
        || head_min.len() != bump.len()
        || tail_min.len() != bump.len()
    {
        return Err(BoxError::DimensionMismatch {
            expected: head_min.len(),
            actual: bump.len().max(tail_min.len()),
        });
    }

    // Apply bump to head box: h' = h + bump
    let mut bumped_head_min = Vec::with_capacity(head_min.len());
    let mut bumped_head_max = Vec::with_capacity(head_max.len());

    for i in 0..head_min.len() {
        bumped_head_min.push(head_min[i] + bump[i]);
        bumped_head_max.push(head_max[i] + bump[i]);
    }

    // Compute containment probability: P(tail ⊆ bumped_head)
    // This is the volume of intersection divided by volume of tail
    let mut intersection_volume = 1.0;
    let mut tail_volume = 1.0;

    for i in 0..head_min.len() {
        // Intersection bounds
        let int_min = bumped_head_min[i].max(tail_min[i]);
        let int_max = bumped_head_max[i].min(tail_max[i]);

        // Check if intersection is valid
        if int_min >= int_max {
            // Boxes are disjoint
            return Ok(0.0);
        }

        let int_side = int_max - int_min;
        let tail_side = tail_max[i] - tail_min[i];

        intersection_volume *= int_side;
        tail_volume *= tail_side;
    }

    // Containment probability: Vol(intersection) / Vol(tail)
    if tail_volume > 0.0 {
        Ok((intersection_volume / tail_volume).clamp(0.0, 1.0))
    } else {
        Ok(0.0)
    }
}

/// BoxE point-entity scoring (faithful to Abboud et al., 2020).
///
/// The original BoxE paper embeds entities as **points** and relations as **boxes**.
/// For a triple (h, r, t):
///
/// 1. Compute bumped positions: `h' = h + bump_t`, `t' = t + bump_h`
/// 2. Score each against the relation's head/tail boxes using the piecewise distance:
///    - Inside box: `|p - center| / (width/2 + 1)` (slow growth)
///    - Outside box: `|p - nearest_boundary|` (fast growth)
/// 3. Total distance: sum over dimensions of per-dim distances for both head and tail
///
/// Lower distance = higher compatibility. The score is negated distance.
///
/// # Parameters
///
/// * `head_point`: Head entity point embedding
/// * `tail_point`: Tail entity point embedding
/// * `rel_head_min`: Relation's head box minimum
/// * `rel_head_max`: Relation's head box maximum
/// * `rel_tail_min`: Relation's tail box minimum
/// * `rel_tail_max`: Relation's tail box maximum
/// * `bump_h`: Head entity's translational bump
/// * `bump_t`: Tail entity's translational bump
///
/// # Returns
///
/// Negative distance (higher = more compatible). Score is in (-inf, 0].
pub fn boxe_point_score(
    head_point: &[f32],
    tail_point: &[f32],
    rel_head_min: &[f32],
    rel_head_max: &[f32],
    rel_tail_min: &[f32],
    rel_tail_max: &[f32],
    bump_h: &[f32],
    bump_t: &[f32],
) -> Result<f32, BoxError> {
    let d = head_point.len();
    if tail_point.len() != d
        || rel_head_min.len() != d
        || rel_head_max.len() != d
        || rel_tail_min.len() != d
        || rel_tail_max.len() != d
        || bump_h.len() != d
        || bump_t.len() != d
    {
        return Err(BoxError::DimensionMismatch {
            expected: d,
            actual: 0, // multiple mismatches possible
        });
    }

    let mut total_dist = 0.0_f32;

    for i in 0..d {
        // Bumped head position: h' = h + bump_t
        let h_prime = head_point[i] + bump_t[i];
        // Bumped tail position: t' = t + bump_h
        let t_prime = tail_point[i] + bump_h[i];

        // Distance of h' to relation's head box
        total_dist += boxe_dim_distance(h_prime, rel_head_min[i], rel_head_max[i]);
        // Distance of t' to relation's tail box
        total_dist += boxe_dim_distance(t_prime, rel_tail_min[i], rel_tail_max[i]);
    }

    Ok(-total_dist)
}

/// Per-dimension BoxE distance function (Abboud et al., 2020, Section 3.1).
///
/// - Inside [lo, hi]: `|p - center| / (width/2 + 1)` (depth, normalized)
/// - Outside [lo, hi]: `|p - clamp(p, lo, hi)|` (gap to nearest boundary)
fn boxe_dim_distance(p: f32, lo: f32, hi: f32) -> f32 {
    if p >= lo && p <= hi {
        // Inside: normalized distance to center
        let center = (lo + hi) / 2.0;
        let half_width = (hi - lo) / 2.0;
        (p - center).abs() / (half_width + 1.0)
    } else {
        // Outside: distance to nearest boundary
        (p - p.clamp(lo, hi)).abs()
    }
}

/// BoxE loss function for training.
///
/// Implements margin-based ranking loss as used in the BoxE paper (Abboud et al., 2020).
///
/// **Reference**: Abboud et al. (2020), Section 3.2 - Training Objective
///
/// The margin-based ranking loss is a standard approach in knowledge graph embedding,
/// originally popularized by Bordes et al. (2013) for TransE and adapted for box embeddings.
///
/// # Intuitive Explanation
///
/// This loss encourages the model to separate positive triples (true facts) from negative
/// triples (corrupted/false facts) by at least a margin. Think of it as creating a "safety
/// zone" between true and false triples.
///
/// **How it works**:
/// - If positive score is much higher than negative score (by at least the margin), loss = 0
/// - Otherwise, loss = margin - (positive_score - negative_score)
/// - This pushes the model to make positive scores higher and negative scores lower
///
/// # Mathematical Formulation
///
/// \[
/// L = \max(0, \text{margin} - \text{score}_{\text{positive}} + \text{score}_{\text{negative}})
/// \]
///
/// The loss is zero when \(\text{score}_{\text{positive}} - \text{score}_{\text{negative}} \geq \text{margin}\),
/// meaning the positive triple is at least `margin` better than the negative triple.
///
/// # Parameters
///
/// * `positive_score`: Score for positive triple
/// * `negative_score`: Score for negative (corrupted) triple
/// * `margin`: Margin parameter (default: 1.0)
///
/// # Returns
///
/// Loss value (non-negative)
pub fn boxe_loss(positive_score: f32, negative_score: f32, margin: f32) -> f32 {
    (margin - positive_score + negative_score).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxe_score_containment() {
        // Head box: [0.0, 0.0] to [1.0, 1.0]
        // Tail box: [0.2, 0.2] to [0.8, 0.8] (contained in head)
        // Bump: [0.0, 0.0] (no translation)
        let head_min = [0.0, 0.0];
        let head_max = [1.0, 1.0];
        let tail_min = [0.2, 0.2];
        let tail_max = [0.8, 0.8];
        let bump = [0.0, 0.0];

        let score = boxe_score(&head_min, &head_max, &tail_min, &tail_max, &bump, 1.0).unwrap();
        assert!(score > 0.9); // High score for containment
    }

    #[test]
    fn test_boxe_score_with_bump() {
        // Head box: [0.0, 0.0] to [1.0, 1.0]
        // Tail box: [0.2, 0.2] to [0.8, 0.8]
        // Bump: [0.1, 0.1] (translates head to [0.1, 0.1] to [1.1, 1.1])
        let head_min = [0.0, 0.0];
        let head_max = [1.0, 1.0];
        let tail_min = [0.2, 0.2];
        let tail_max = [0.8, 0.8];
        let bump = [0.1, 0.1];

        let score = boxe_score(&head_min, &head_max, &tail_min, &tail_max, &bump, 1.0).unwrap();
        assert!(score > 0.0); // Should still have some overlap
    }

    #[test]
    fn test_boxe_score_disjoint() {
        // Head box: [0.0, 0.0] to [1.0, 1.0]
        // Tail box: [2.0, 2.0] to [3.0, 3.0] (disjoint)
        // Bump: [0.0, 0.0]
        let head_min = [0.0, 0.0];
        let head_max = [1.0, 1.0];
        let tail_min = [2.0, 2.0];
        let tail_max = [3.0, 3.0];
        let bump = [0.0, 0.0];

        let score = boxe_score(&head_min, &head_max, &tail_min, &tail_max, &bump, 1.0).unwrap();
        assert_eq!(score, 0.0); // Disjoint boxes should have score 0
    }

    #[test]
    fn test_boxe_loss() {
        // Positive score higher than negative, but not enough to make loss 0
        // margin - positive + negative = 1.0 - 0.9 + 0.1 = 0.2
        let loss = boxe_loss(0.9, 0.1, 1.0);
        assert!((loss - 0.2).abs() < 1e-5); // margin - 0.9 + 0.1 = 0.2

        // When positive - negative >= margin, loss should be 0
        // For loss to be 0: margin - positive + negative <= 0
        // i.e., positive - negative >= margin
        // Example: positive = 1.5, negative = 0.0, margin = 1.0
        // loss = max(0, 1.0 - 1.5 + 0.0) = max(0, -0.5) = 0.0
        let loss_zero = boxe_loss(1.5, 0.0, 1.0);
        assert_eq!(loss_zero, 0.0);

        // Negative score higher (bad case)
        let loss = boxe_loss(0.1, 0.9, 1.0);
        assert!((loss - 1.8).abs() < 1e-5); // margin - 0.1 + 0.9 = 1.8
    }

    // ---- boxe_point_score tests (faithful to Abboud et al. 2020) ----

    #[test]
    fn test_boxe_point_inside_box() {
        // Head point at center of relation's head box => zero distance
        let h = [0.5, 0.5];
        let t = [0.5, 0.5];
        let rh_min = [0.0, 0.0];
        let rh_max = [1.0, 1.0];
        let rt_min = [0.0, 0.0];
        let rt_max = [1.0, 1.0];
        let bh = [0.0, 0.0];
        let bt = [0.0, 0.0];

        let score = boxe_point_score(&h, &t, &rh_min, &rh_max, &rt_min, &rt_max, &bh, &bt).unwrap();
        assert!((score - 0.0).abs() < 1e-6, "center point should have score 0, got {score}");
    }

    #[test]
    fn test_boxe_point_outside_box() {
        let h = [5.0, 5.0]; // far outside box [0,1]^2
        let t = [0.5, 0.5]; // inside
        let rh_min = [0.0, 0.0];
        let rh_max = [1.0, 1.0];
        let rt_min = [0.0, 0.0];
        let rt_max = [1.0, 1.0];
        let bh = [0.0, 0.0];
        let bt = [0.0, 0.0];

        let score = boxe_point_score(&h, &t, &rh_min, &rh_max, &rt_min, &rt_max, &bh, &bt).unwrap();
        assert!(score < -5.0, "outside point should have very negative score, got {score}");
    }

    #[test]
    fn test_boxe_point_bump_moves_into_box() {
        let h = [5.0, 5.0];
        let t = [0.5, 0.5];
        let rh_min = [0.0, 0.0];
        let rh_max = [1.0, 1.0];
        let rt_min = [0.0, 0.0];
        let rt_max = [1.0, 1.0];
        let bh = [0.0, 0.0];
        let bt = [-4.5, -4.5]; // bump_t moves h' = 5 + (-4.5) = 0.5, inside

        let score = boxe_point_score(&h, &t, &rh_min, &rh_max, &rt_min, &rt_max, &bh, &bt).unwrap();
        assert!((score - 0.0).abs() < 1e-6, "bumped-in point should have score ~0, got {score}");
    }

    #[test]
    fn test_boxe_point_score_is_non_positive() {
        let h = [0.3, 0.7];
        let t = [0.5, 0.5];
        let rh_min = [0.0, 0.0];
        let rh_max = [1.0, 1.0];
        let rt_min = [0.0, 0.0];
        let rt_max = [1.0, 1.0];
        let bh = [0.0, 0.0];
        let bt = [0.0, 0.0];

        let score = boxe_point_score(&h, &t, &rh_min, &rh_max, &rt_min, &rt_max, &bh, &bt).unwrap();
        assert!(score <= 0.0 + 1e-6, "score should be <= 0, got {score}");
    }

    #[test]
    fn test_boxe_dim_distance_fn() {
        // Inside at center => 0
        assert!((boxe_dim_distance(0.5, 0.0, 1.0) - 0.0).abs() < 1e-6);
        // Inside at boundary => |0.5|/1.5 ~ 0.333
        assert!((boxe_dim_distance(1.0, 0.0, 1.0) - 1.0 / 3.0).abs() < 0.01);
        // Outside: 1 unit above
        assert!((boxe_dim_distance(2.0, 0.0, 1.0) - 1.0).abs() < 1e-6);
        // Outside: below
        assert!((boxe_dim_distance(-0.5, 0.0, 1.0) - 0.5).abs() < 1e-6);
    }

    // =========================================================================
    // Property tests
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Strategy: generate valid box bounds (lo < hi) and a point in any range.
        fn arb_box_and_point(dim: usize) -> impl Strategy<Value = (Vec<f32>, Vec<f32>, Vec<f32>)> {
            proptest::collection::vec(
                (-10.0f32..10.0f32, 0.01f32..5.0f32),
                dim,
            )
            .prop_flat_map(move |pairs| {
                let lo: Vec<f32> = pairs.iter().map(|(l, _)| *l).collect();
                let hi: Vec<f32> = pairs.iter().map(|(l, w)| l + w).collect();
                let point_strat = proptest::collection::vec(-15.0f32..15.0f32, dim);
                (Just(lo), Just(hi), point_strat)
            })
        }

        // ---- boxe_point_score is non-positive for any inputs ----
        proptest! {
            #[test]
            fn prop_boxe_point_score_non_positive(
                dim in 1usize..=4,
                h_coords in proptest::collection::vec(-10.0f32..10.0f32, 1..=4),
                t_coords in proptest::collection::vec(-10.0f32..10.0f32, 1..=4),
                rh_lo_w in proptest::collection::vec((-10.0f32..10.0f32, 0.01f32..5.0f32), 1..=4),
                rt_lo_w in proptest::collection::vec((-10.0f32..10.0f32, 0.01f32..5.0f32), 1..=4),
                bh_coords in proptest::collection::vec(-5.0f32..5.0f32, 1..=4),
                bt_coords in proptest::collection::vec(-5.0f32..5.0f32, 1..=4),
            ) {
                let d = dim.min(h_coords.len())
                    .min(t_coords.len())
                    .min(rh_lo_w.len())
                    .min(rt_lo_w.len())
                    .min(bh_coords.len())
                    .min(bt_coords.len());
                prop_assume!(d > 0);

                let h = &h_coords[..d];
                let t = &t_coords[..d];
                let rh_min: Vec<f32> = rh_lo_w[..d].iter().map(|(l, _)| *l).collect();
                let rh_max: Vec<f32> = rh_lo_w[..d].iter().map(|(l, w)| l + w).collect();
                let rt_min: Vec<f32> = rt_lo_w[..d].iter().map(|(l, _)| *l).collect();
                let rt_max: Vec<f32> = rt_lo_w[..d].iter().map(|(l, w)| l + w).collect();
                let bh = &bh_coords[..d];
                let bt = &bt_coords[..d];

                let score = boxe_point_score(h, t, &rh_min, &rh_max, &rt_min, &rt_max, bh, bt).unwrap();
                prop_assert!(
                    score <= 1e-6,
                    "boxe_point_score should be <= 0, got {score}"
                );
            }
        }

        // ---- boxe_point_score is zero when points are at box centers with no bump ----
        proptest! {
            #[test]
            fn prop_boxe_point_score_zero_at_centers(
                lo_w in proptest::collection::vec((-10.0f32..10.0f32, 0.1f32..5.0f32), 1..=4),
            ) {
                let d = lo_w.len();
                let rh_min: Vec<f32> = lo_w.iter().map(|(l, _)| *l).collect();
                let rh_max: Vec<f32> = lo_w.iter().map(|(l, w)| l + w).collect();
                let center: Vec<f32> = rh_min.iter().zip(rh_max.iter())
                    .map(|(lo, hi)| (lo + hi) / 2.0)
                    .collect();
                let zero_bump = vec![0.0f32; d];

                // h at center of rh box, t at center of rt box, zero bumps
                // => h' = h + bt = center + 0 = center, t' = t + bh = center + 0 = center
                let score = boxe_point_score(
                    &center, &center,
                    &rh_min, &rh_max,
                    &rh_min, &rh_max,
                    &zero_bump, &zero_bump,
                ).unwrap();
                prop_assert!(
                    score.abs() < 1e-5,
                    "boxe_point_score at box centers with zero bump should be ~0, got {score}"
                );
            }
        }
    }
}
