//! BoxE: Box Embedding Model for Knowledge Base Completion.
//!
//! Implements the BoxE model from **Boratko et al. (NeurIPS 2020)**:
//! "BoxE: A Box Embedding Model for Knowledge Base Completion"
//!
//! **Paper**: [arXiv:2007.06267](https://arxiv.org/abs/2007.06267) | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/6dbbe6abe5f14af882ff977fc3f35501-Abstract.html)
//!
//! # Intuitive Explanation
//!
//! ## Paradigm Problem: Modeling Knowledge Graph Triples
//!
//! Consider the knowledge graph triple (Paris, located_in, France). We want to check whether
//! this relationship is true. BoxE's approach:
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
//! **Research foundation**: This approach is introduced in **Boratko et al. (2020)**, Section 3.1,
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
/// **Reference**: Boratko et al. (2020), Section 3.1 - Translational Bumps
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

/// BoxE loss function for training.
///
/// Implements margin-based ranking loss as used in the BoxE paper (Boratko et al., 2020).
///
/// **Reference**: Boratko et al. (2020), Section 3.2 - Training Objective
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
}
