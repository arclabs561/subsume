//! TransBox: EL++-closed box embeddings with translational composition.
//!
//! TransBox extends box embeddings with role-specific transformations that
//! preserve EL++ semantics. Unlike standard box embeddings where roles are
//! simple translations, TransBox uses box-to-box transformations that handle
//! many-to-many relations and complex role compositions.
//!
//! # Key idea
//!
//! For a triple (h, r, t), instead of checking `box_h + translation_r ⊆ box_t`,
//! TransBox transforms the head box through the role box:
//!
//! ```text
//! transformed_center = center_h + center_r
//! transformed_offset = offset_h + offset_r  (additive, not subtractive)
//! score = inclusion_loss(transformed, box_t)
//! ```
//!
//! The additive offset formula (vs Box2EL's subtractive) reflects TransBox's
//! compositional semantics: roles widen the reachable region rather than
//! restricting it.
//!
//! # EL++ closure
//!
//! TransBox ensures that box operations preserve EL++ semantics:
//! - `C ⊑ D` (subsumption): `box_C ⊆ box_D`
//! - `C ⊓ D ⊑ E` (intersection): `box_C ∩ box_D ⊆ box_E`
//! - `∃r.C ⊑ D` (existential): `transform(box_C, r) ⊆ box_D`
//!
//! # References
//!
//! - Yang, Chen, Sattler (2024), "TransBox: EL++-closed Ontology Embedding"
//!   (arXiv:2410.14571)

use crate::BoxError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A TransBox concept: a box with center/offset representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransBoxConcept {
    center: Vec<f32>,
    offset: Vec<f32>,
}

/// A TransBox role: a box that transforms concepts via translation + offset addition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransBoxRole {
    center: Vec<f32>,
    offset: Vec<f32>,
}

/// A complete TransBox model with concepts and roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransBoxModel {
    concepts: Vec<TransBoxConcept>,
    roles: Vec<TransBoxRole>,
    dim: usize,
}

// ---------------------------------------------------------------------------
// TransBoxConcept
// ---------------------------------------------------------------------------

impl TransBoxConcept {
    /// Create a new concept box.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if center and offset differ in length.
    /// Returns [`BoxError::InvalidBounds`] if any offset is negative or any value non-finite.
    pub fn new(center: Vec<f32>, offset: Vec<f32>) -> Result<Self, BoxError> {
        if center.len() != offset.len() {
            return Err(BoxError::DimensionMismatch {
                expected: center.len(),
                actual: offset.len(),
            });
        }
        for (i, (&c, &o)) in center.iter().zip(offset.iter()).enumerate() {
            if !c.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: c as f64,
                    max: c as f64,
                });
            }
            if !o.is_finite() || o < 0.0 {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: 0.0,
                    max: o as f64,
                });
            }
        }
        Ok(Self { center, offset })
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.center.len()
    }

    pub fn center(&self) -> &[f32] {
        &self.center
    }

    pub fn offset(&self) -> &[f32] {
        &self.offset
    }

    /// Compute the min/max bounds of this box.
    #[must_use]
    pub fn bounds(&self) -> (Vec<f32>, Vec<f32>) {
        let min: Vec<f32> = self
            .center
            .iter()
            .zip(self.offset.iter())
            .map(|(&c, &o)| c - o)
            .collect();
        let max: Vec<f32> = self
            .center
            .iter()
            .zip(self.offset.iter())
            .map(|(&c, &o)| c + o)
            .collect();
        (min, max)
    }
}

// ---------------------------------------------------------------------------
// TransBoxRole
// ---------------------------------------------------------------------------

impl TransBoxRole {
    /// Create a new role box.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if center and offset differ in length.
    /// Returns [`BoxError::InvalidBounds`] if any offset is negative or any value non-finite.
    pub fn new(center: Vec<f32>, offset: Vec<f32>) -> Result<Self, BoxError> {
        if center.len() != offset.len() {
            return Err(BoxError::DimensionMismatch {
                expected: center.len(),
                actual: offset.len(),
            });
        }
        for (i, (&c, &o)) in center.iter().zip(offset.iter()).enumerate() {
            if !c.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: c as f64,
                    max: c as f64,
                });
            }
            if !o.is_finite() || o < 0.0 {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: 0.0,
                    max: o as f64,
                });
            }
        }
        Ok(Self { center, offset })
    }

    /// Apply this role to a concept, returning the transformed concept.
    ///
    /// TransBox uses additive composition:
    /// - `center' = center_concept + center_role`
    /// - `offset' = offset_concept + offset_role`
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if dimensions disagree.
    pub fn apply(&self, concept: &TransBoxConcept) -> Result<TransBoxConcept, BoxError> {
        if self.center.len() != concept.center.len() {
            return Err(BoxError::DimensionMismatch {
                expected: concept.center.len(),
                actual: self.center.len(),
            });
        }
        let new_center: Vec<f32> = concept
            .center
            .iter()
            .zip(self.center.iter())
            .map(|(&c, &r)| c + r)
            .collect();
        let new_offset: Vec<f32> = concept
            .offset
            .iter()
            .zip(self.offset.iter())
            .map(|(&c, &r)| c + r)
            .collect();
        TransBoxConcept::new(new_center, new_offset)
    }

    /// Compose two roles: `r ∘ s`.
    ///
    /// For `r ∘ s ⊑ t` (RI7 normal form):
    /// - `center = center_r + center_s`
    /// - `offset = offset_r + offset_s`
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if dimensions disagree.
    pub fn compose(&self, other: &TransBoxRole) -> Result<TransBoxRole, BoxError> {
        if self.center.len() != other.center.len() {
            return Err(BoxError::DimensionMismatch {
                expected: other.center.len(),
                actual: self.center.len(),
            });
        }
        let new_center: Vec<f32> = self
            .center
            .iter()
            .zip(other.center.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let new_offset: Vec<f32> = self
            .offset
            .iter()
            .zip(other.offset.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        TransBoxRole::new(new_center, new_offset)
    }
}

// ---------------------------------------------------------------------------
// TransBoxModel
// ---------------------------------------------------------------------------

impl TransBoxModel {
    /// Create a new TransBox model.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any component has wrong dimension.
    pub fn new(
        concepts: Vec<TransBoxConcept>,
        roles: Vec<TransBoxRole>,
        dim: usize,
    ) -> Result<Self, BoxError> {
        for c in &concepts {
            if c.dim() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: c.dim(),
                });
            }
        }
        for r in &roles {
            if r.center.len() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: r.center.len(),
                });
            }
        }
        Ok(Self {
            concepts,
            roles,
            dim,
        })
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[must_use]
    pub fn num_concepts(&self) -> usize {
        self.concepts.len()
    }

    #[must_use]
    pub fn num_roles(&self) -> usize {
        self.roles.len()
    }

    pub fn concepts(&self) -> &[TransBoxConcept] {
        &self.concepts
    }

    pub fn roles(&self) -> &[TransBoxRole] {
        &self.roles
    }
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// TransBox inclusion loss: `||relu(|c_a - c_b| + o_a - o_b - margin)||_2`.
///
/// Measures how much box A fails to be contained in box B. Zero when A ⊆ B.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if vectors differ in length.
pub fn inclusion_loss(
    center_a: &[f32],
    offset_a: &[f32],
    center_b: &[f32],
    offset_b: &[f32],
    margin: f32,
) -> Result<f32, BoxError> {
    let dim = center_a.len();
    if offset_a.len() != dim || center_b.len() != dim || offset_b.len() != dim {
        return Err(BoxError::DimensionMismatch {
            expected: dim,
            actual: offset_a.len().max(center_b.len()).max(offset_b.len()),
        });
    }

    let mut sum_sq = 0.0f32;
    for i in 0..dim {
        let v = (center_a[i] - center_b[i]).abs() + offset_a[i] - offset_b[i] - margin;
        let relu_v = v.max(0.0);
        sum_sq += relu_v * relu_v;
    }
    Ok(sum_sq.sqrt())
}

/// Score a triple (head_concept, role, tail_concept) using TransBox.
///
/// Transforms the head concept through the role, then computes inclusion
/// loss with the tail concept. Lower scores indicate better matches.
///
/// ```text
/// score = inclusion_loss(transform(head, role), tail)
/// ```
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if any component has mismatched dimensions.
pub fn score_triple(
    head: &TransBoxConcept,
    role: &TransBoxRole,
    tail: &TransBoxConcept,
    margin: f32,
) -> Result<f32, BoxError> {
    let transformed = role.apply(head)?;
    inclusion_loss(
        transformed.center(),
        transformed.offset(),
        tail.center(),
        tail.offset(),
        margin,
    )
}

/// Compute the existential restriction `∃r.C` (TransBox additive formula).
///
/// ```text
/// center = center_role + center_filler
/// offset = offset_role + offset_filler
/// ```
///
/// This is the TransBox formula (additive), which grows the box.
/// Contrast with Box2EL's subtractive formula: `offset = max(0, offset_filler - offset_role)`.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if vectors differ in length.
pub fn existential_transbox(
    role: &TransBoxRole,
    filler: &TransBoxConcept,
) -> Result<TransBoxConcept, BoxError> {
    role.apply(filler)
}

/// Check if concept A subsumes concept B (A ⊇ B).
///
/// Returns the inclusion loss: 0 means B ⊆ A (A subsumes B).
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if dimensions disagree.
pub fn subsumption_loss(
    subsumer: &TransBoxConcept,
    subsumed: &TransBoxConcept,
    margin: f32,
) -> Result<f32, BoxError> {
    inclusion_loss(
        subsumed.center(),
        subsumed.offset(),
        subsumer.center(),
        subsumer.offset(),
        margin,
    )
}

/// Intersection of two concepts: `C ⊓ D`.
///
/// Returns the smallest box containing the intersection:
/// - `center = (center_C + center_D) / 2`
/// - `offset = (offset_C + offset_D) / 2 - |center_C - center_D| / 2`
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if dimensions disagree.
pub fn intersection(a: &TransBoxConcept, b: &TransBoxConcept) -> Result<TransBoxConcept, BoxError> {
    if a.center.len() != b.center.len() {
        return Err(BoxError::DimensionMismatch {
            expected: a.center.len(),
            actual: b.center.len(),
        });
    }
    let center: Vec<f32> = a
        .center
        .iter()
        .zip(b.center.iter())
        .map(|(&ca, &cb)| (ca + cb) / 2.0)
        .collect();
    let offset: Vec<f32> = a
        .offset
        .iter()
        .zip(b.offset.iter())
        .zip(a.center.iter())
        .zip(b.center.iter())
        .map(|(((&oa, &ob), &ca), &cb)| ((oa + ob) / 2.0 - (ca - cb).abs() / 2.0).max(0.0))
        .collect();
    TransBoxConcept::new(center, offset)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concept_new_valid() {
        let c = TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        assert_eq!(c.dim(), 2);
    }

    #[test]
    fn concept_rejects_negative_offset() {
        assert!(TransBoxConcept::new(vec![0.0], vec![-1.0]).is_err());
    }

    #[test]
    fn concept_rejects_non_finite() {
        assert!(TransBoxConcept::new(vec![f32::NAN], vec![1.0]).is_err());
        assert!(TransBoxConcept::new(vec![0.0], vec![f32::INFINITY]).is_err());
    }

    #[test]
    fn concept_rejects_dim_mismatch() {
        assert!(TransBoxConcept::new(vec![0.0, 0.0], vec![1.0]).is_err());
    }

    #[test]
    fn role_new_valid() {
        let r = TransBoxRole::new(vec![1.0, 0.0], vec![0.5, 0.5]).unwrap();
        assert_eq!(r.center.len(), 2);
    }

    #[test]
    fn role_rejects_negative_offset() {
        assert!(TransBoxRole::new(vec![0.0], vec![-0.1]).is_err());
    }

    // --- Role application ---

    #[test]
    fn role_apply_additive() {
        let c = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let r = TransBoxRole::new(vec![1.0, 2.0], vec![0.3, 0.4]).unwrap();
        let t = r.apply(&c).unwrap();
        assert!((t.center()[0] - 1.0).abs() < 1e-6);
        assert!((t.center()[1] - 2.0).abs() < 1e-6);
        assert!((t.offset()[0] - 0.8).abs() < 1e-6);
        assert!((t.offset()[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn role_apply_dimension_mismatch() {
        let c = TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let r = TransBoxRole::new(vec![0.0], vec![1.0]).unwrap();
        assert!(r.apply(&c).is_err());
    }

    // --- Role composition ---

    #[test]
    fn role_compose() {
        let r1 = TransBoxRole::new(vec![1.0, 0.0], vec![0.5, 0.5]).unwrap();
        let r2 = TransBoxRole::new(vec![0.0, 1.0], vec![0.3, 0.3]).unwrap();
        let composed = r1.compose(&r2).unwrap();
        assert!((composed.center()[0] - 1.0).abs() < 1e-6);
        assert!((composed.center()[1] - 1.0).abs() < 1e-6);
        assert!((composed.offset()[0] - 0.8).abs() < 1e-6);
        assert!((composed.offset()[1] - 0.8).abs() < 1e-6);
    }

    // --- Inclusion loss ---

    #[test]
    fn inclusion_loss_contained_is_zero() {
        // A is inside B
        let ca = vec![0.0, 0.0];
        let oa = vec![0.5, 0.5];
        let cb = vec![0.0, 0.0];
        let ob = vec![1.0, 1.0];
        let loss = inclusion_loss(&ca, &oa, &cb, &ob, 0.0).unwrap();
        assert!(loss.abs() < 1e-6, "contained loss = {loss}, expected 0");
    }

    #[test]
    fn inclusion_loss_not_contained_is_positive() {
        // A extends beyond B
        let ca = vec![0.0, 0.0];
        let oa = vec![1.0, 1.0];
        let cb = vec![0.0, 0.0];
        let ob = vec![0.5, 0.5];
        let loss = inclusion_loss(&ca, &oa, &cb, &ob, 0.0).unwrap();
        assert!(loss > 0.0, "non-contained loss = {loss}, expected > 0");
    }

    #[test]
    fn inclusion_loss_with_margin() {
        let ca = vec![0.0, 0.0];
        let oa = vec![0.4, 0.4];
        let cb = vec![0.0, 0.0];
        let ob = vec![0.5, 0.5];
        // With margin=0.1, A fits inside B: 0.4 <= 0.5 - 0.1 = 0.4
        let loss = inclusion_loss(&ca, &oa, &cb, &ob, 0.1).unwrap();
        assert!(loss.abs() < 1e-6, "with margin loss = {loss}, expected 0");
    }

    #[test]
    fn inclusion_loss_dimension_mismatch() {
        assert!(inclusion_loss(&[0.0], &[1.0], &[0.0, 0.0], &[1.0, 1.0], 0.0).is_err());
    }

    // --- Triple scoring ---

    #[test]
    fn score_triple_perfect_match() {
        let h = TransBoxConcept::new(vec![0.0, 0.0], vec![0.3, 0.3]).unwrap();
        let r = TransBoxRole::new(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let t = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let s = score_triple(&h, &r, &t, 0.0).unwrap();
        assert!(s.abs() < 1e-6, "perfect match score = {s}, expected 0");
    }

    #[test]
    fn score_triple_mismatch_is_positive() {
        let h = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let r = TransBoxRole::new(vec![5.0, 5.0], vec![0.0, 0.0]).unwrap();
        let t = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let s = score_triple(&h, &r, &t, 0.0).unwrap();
        assert!(s > 0.0, "mismatch score = {s}, expected > 0");
    }

    // --- Existential ---

    #[test]
    fn existential_transbox_additive() {
        let role = TransBoxRole::new(vec![1.0, 0.0], vec![0.5, 0.5]).unwrap();
        let filler = TransBoxConcept::new(vec![0.0, 1.0], vec![0.3, 0.3]).unwrap();
        let ex = existential_transbox(&role, &filler).unwrap();
        assert!((ex.center()[0] - 1.0).abs() < 1e-6);
        assert!((ex.center()[1] - 1.0).abs() < 1e-6);
        assert!((ex.offset()[0] - 0.8).abs() < 1e-6);
        assert!((ex.offset()[1] - 0.8).abs() < 1e-6);
    }

    // --- Subsumption ---

    #[test]
    fn subsumption_loss_valid_is_zero() {
        let parent = TransBoxConcept::new(vec![0.0, 0.0], vec![2.0, 2.0]).unwrap();
        let child = TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let loss = subsumption_loss(&parent, &child, 0.0).unwrap();
        assert!(loss.abs() < 1e-6, "valid subsumption loss = {loss}");
    }

    #[test]
    fn subsumption_loss_invalid_is_positive() {
        let parent = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let child = TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let loss = subsumption_loss(&parent, &child, 0.0).unwrap();
        assert!(
            loss > 0.0,
            "invalid subsumption loss = {loss}, expected > 0"
        );
    }

    // --- Intersection ---

    #[test]
    fn intersection_of_overlapping() {
        let a = TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let b = TransBoxConcept::new(vec![1.0, 1.0], vec![1.0, 1.0]).unwrap();
        let inter = intersection(&a, &b).unwrap();
        // center = (0.5, 0.5), offset = (1.0 - 0.5, 1.0 - 0.5) = (0.5, 0.5)
        assert!((inter.center()[0] - 0.5).abs() < 1e-6);
        assert!((inter.offset()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn intersection_of_identical() {
        let a = TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let inter = intersection(&a, &a).unwrap();
        assert!((inter.center()[0]).abs() < 1e-6);
        assert!((inter.offset()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn intersection_disjoint_has_zero_offset() {
        let a = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let b = TransBoxConcept::new(vec![10.0, 10.0], vec![0.5, 0.5]).unwrap();
        let inter = intersection(&a, &b).unwrap();
        // center = (5, 5), offset = (1.0 - 5.0, 1.0 - 5.0) clamped to 0
        assert!(inter.offset()[0].abs() < 1e-6);
    }

    // --- Model ---

    #[test]
    fn model_construction() {
        let concepts = vec![
            TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap(),
            TransBoxConcept::new(vec![1.0, 1.0], vec![0.5, 0.5]).unwrap(),
        ];
        let roles = vec![TransBoxRole::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap()];
        let model = TransBoxModel::new(concepts, roles, 2).unwrap();
        assert_eq!(model.num_concepts(), 2);
        assert_eq!(model.num_roles(), 1);
        assert_eq!(model.dim(), 2);
    }

    #[test]
    fn model_rejects_dim_mismatch() {
        let concepts = vec![TransBoxConcept::new(vec![0.0], vec![1.0]).unwrap()];
        let roles = vec![TransBoxRole::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap()];
        assert!(TransBoxModel::new(concepts, roles, 1).is_err());
    }

    // --- Bounds ---

    #[test]
    fn concept_bounds() {
        let c = TransBoxConcept::new(vec![1.0, 2.0], vec![0.5, 1.0]).unwrap();
        let (min, max) = c.bounds();
        assert!((min[0] - 0.5).abs() < 1e-6);
        assert!((min[1] - 1.0).abs() < 1e-6);
        assert!((max[0] - 1.5).abs() < 1e-6);
        assert!((max[1] - 3.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_concept(dim: usize) -> impl Strategy<Value = TransBoxConcept> {
        (
            prop::collection::vec(-10.0f32..10.0, dim),
            prop::collection::vec(0.01f32..10.0, dim),
        )
            .prop_map(|(c, o)| TransBoxConcept::new(c, o).unwrap())
    }

    fn arb_role(dim: usize) -> impl Strategy<Value = TransBoxRole> {
        (
            prop::collection::vec(-5.0f32..5.0, dim),
            prop::collection::vec(0.01f32..5.0, dim),
        )
            .prop_map(|(c, o)| TransBoxRole::new(c, o).unwrap())
    }

    fn arb_concept_pair(dim: usize) -> impl Strategy<Value = (TransBoxConcept, TransBoxConcept)> {
        (arb_concept(dim), arb_concept(dim))
    }

    proptests! {
        #[test]
        fn prop_inclusion_loss_nonneg(
            (a, b) in arb_concept_pair(4)
        ) {
            let loss = inclusion_loss(a.center(), a.offset(), b.center(), b.offset(), 0.0).unwrap();
            prop_assert!(loss >= -1e-6, "inclusion_loss < 0: {loss}");
        }

        #[test]
        fn prop_inclusion_loss_zero_when_contained(
            parent in arb_concept(4),
            shrink_factor in 0.01f32..1.0,
        ) {
            // Create a child that is strictly inside the parent
            let child_center: Vec<f32> = parent.center().to_vec();
            let child_offset: Vec<f32> = parent.offset().iter().map(|&o| o * shrink_factor).collect();
            let child = TransBoxConcept::new(child_center, child_offset).unwrap();
            let loss = inclusion_loss(
                child.center(), child.offset(),
                parent.center(), parent.offset(),
                0.0
            ).unwrap();
            prop_assert!(loss < 1e-5, "contained inclusion_loss = {loss}, expected ~0");
        }

        #[test]
        fn prop_score_triple_nonneg(
            h in arb_concept(4),
            r in arb_role(4),
            t in arb_concept(4)
        ) {
            let s = score_triple(&h, &r, &t, 0.0).unwrap();
            prop_assert!(s >= -1e-6, "score_triple < 0: {s}");
        }

        #[test]
        fn prop_role_apply_preserves_dim(
            c in arb_concept(4),
            r in arb_role(4)
        ) {
            let t = r.apply(&c).unwrap();
            prop_assert_eq!(t.dim(), c.dim());
        }

        #[test]
        fn prop_role_compose_preserves_dim(
            r1 in arb_role(4),
            r2 in arb_role(4)
        ) {
            let composed = r1.compose(&r2).unwrap();
            prop_assert_eq!(composed.center.len(), r1.center.len());
        }

        #[test]
        fn prop_intersection_offset_nonneg(
            (a, b) in arb_concept_pair(4)
        ) {
            let inter = intersection(&a, &b).unwrap();
            for &o in inter.offset() {
                prop_assert!(o >= -1e-6, "intersection offset < 0: {o}");
            }
        }

        #[test]
        fn prop_subsumption_loss_symmetric_for_identical(
            c in arb_concept(4)
        ) {
            let loss = subsumption_loss(&c, &c, 0.0).unwrap();
            prop_assert!(loss < 1e-5, "self subsumption loss = {loss}, expected ~0");
        }
    }
}
