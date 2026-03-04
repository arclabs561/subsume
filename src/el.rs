//! EL++ ontology embedding primitives.
//!
//! This module provides geometric operations for embedding OWL EL++ description
//! logic ontologies using boxes. The two primary reference methods are:
//!
//! - **Box2EL** (WWW 2024, [arXiv:2301.11118](https://arxiv.org/abs/2301.11118)):
//!   dual head/tail role boxes with bump vectors
//! - **TransBox** (Oct 2024, [arXiv:2410.14571](https://arxiv.org/abs/2410.14571)):
//!   roles as single boxes with translational composition
//!
//! # EL++ Normal Forms
//!
//! EL++ ontologies are normalized into these forms:
//!
//! | Form | Pattern | Example |
//! |------|---------|---------|
//! | NF1 | C1 ⊓ C2 ⊑ D | HeartDisease ⊓ Genetic ⊑ InheritedHeartDisease |
//! | NF2 | C ⊑ D | Dog ⊑ Animal |
//! | NF3 | C ⊑ ∃r.D | Heart ⊑ ∃partOf.Body |
//! | NF4 | ∃r.C ⊑ D | ∃hasParent.Human ⊑ Human |
//! | RI6 | r ⊑ s | hasChild ⊑ hasDescendant |
//! | RI7 | r ∘ s ⊑ t | hasParent ∘ hasSibling ⊑ hasUncle |
//!
//! # Losses
//!
//! The `el_inclusion_loss` function implements the element-wise ReLU L2 loss
//! for `C ⊑ D` (concept C is subsumed by concept D):
//!
//! ```text
//! loss = ||relu(|c_C - c_D| + o_C - o_D - margin)||_2
//! ```
//!
//! This loss is zero when box C fits inside box D (with margin).

use crate::BoxError;

/// Compute EL++ inclusion loss: `||relu(|c_a - c_b| + o_a - o_b - margin)||_2`.
///
/// Measures how much box A fails to be contained in box B. The loss is zero
/// when A is fully inside B (with margin). Uses center/offset representation.
///
/// # Arguments
///
/// * `center_a`, `offset_a` - Center and offset (half-width) of box A (the subsumed concept)
/// * `center_b`, `offset_b` - Center and offset (half-width) of box B (the subsuming concept)
/// * `margin` - Margin for containment (larger = stricter)
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if vectors differ in length.
pub fn el_inclusion_loss(
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

/// Translate a box by a vector (shift center, keep offset).
///
/// This is the core operation in TransBox: roles act as translations
/// on concept boxes. `translated_center[i] = center[i] + translation[i]`.
///
/// # Arguments
///
/// * `center` - Box center to translate
/// * `translation` - Translation vector (role embedding)
/// * `out` - Output buffer for translated center
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if vectors differ in length.
pub fn translate(center: &[f32], translation: &[f32], out: &mut [f32]) -> Result<(), BoxError> {
    let dim = center.len();
    if translation.len() != dim || out.len() != dim {
        return Err(BoxError::DimensionMismatch {
            expected: dim,
            actual: translation.len().max(out.len()),
        });
    }
    for i in 0..dim {
        out[i] = center[i] + translation[i];
    }
    Ok(())
}

/// Compose two role translations (TransBox: r o s).
///
/// For `r ∘ s ⊑ t` (RI7 normal form), the composed role is:
/// - center: `center_r + center_s`
/// - offset: `offset_r + offset_s`
///
/// # Arguments
///
/// * `center_r`, `offset_r` - First role (center/offset)
/// * `center_s`, `offset_s` - Second role (center/offset)
/// * `center_out`, `offset_out` - Output buffers
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if vectors differ in length.
pub fn compose_roles(
    center_r: &[f32],
    offset_r: &[f32],
    center_s: &[f32],
    offset_s: &[f32],
    center_out: &mut [f32],
    offset_out: &mut [f32],
) -> Result<(), BoxError> {
    let dim = center_r.len();
    if offset_r.len() != dim
        || center_s.len() != dim
        || offset_s.len() != dim
        || center_out.len() != dim
        || offset_out.len() != dim
    {
        return Err(BoxError::DimensionMismatch {
            expected: dim,
            actual: offset_r
                .len()
                .max(center_s.len())
                .max(offset_s.len())
                .max(center_out.len())
                .max(offset_out.len()),
        });
    }
    for i in 0..dim {
        center_out[i] = center_r[i] + center_s[i];
        offset_out[i] = offset_r[i] + offset_s[i];
    }
    Ok(())
}

/// Compute the existential restriction box `∃r.C`.
///
/// For NF3/NF4 normal forms, the existential box combines a role and a filler:
/// - center: `center_role + center_filler`
/// - offset: `max(0, offset_filler - offset_role)`
///
/// # Design note (Box2EL vs TransBox)
///
/// This uses the **Box2EL** offset formula, which shrinks the filler box by
/// subtracting the role offset. The intuition: the existential restricts the
/// concept to only those instances that have an r-successor in C, producing a
/// narrower box.
///
/// **TransBox** (Proposition 4.1) instead uses an additive formula:
/// `offset = offset_role + offset_filler`, which grows the box. That reflects
/// a different compositional semantics where roles widen the reachable region.
///
/// We chose the Box2EL (subtractive) formula because this module's primary use
/// case is EL++ ontology embeddings where existentials should restrict, not
/// expand, the concept space. If additive semantics are needed, compose via
/// [`compose_roles`] (which uses addition) followed by a containment check.
///
/// # Arguments
///
/// * `role_center`, `role_offset` - Role box (center/offset)
/// * `filler_center`, `filler_offset` - Filler concept box (center/offset)
/// * `center_out`, `offset_out` - Output buffers
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if vectors differ in length.
pub fn existential_box(
    role_center: &[f32],
    role_offset: &[f32],
    filler_center: &[f32],
    filler_offset: &[f32],
    center_out: &mut [f32],
    offset_out: &mut [f32],
) -> Result<(), BoxError> {
    let dim = role_center.len();
    if role_offset.len() != dim
        || filler_center.len() != dim
        || filler_offset.len() != dim
        || center_out.len() != dim
        || offset_out.len() != dim
    {
        return Err(BoxError::DimensionMismatch {
            expected: dim,
            actual: role_offset
                .len()
                .max(filler_center.len())
                .max(filler_offset.len())
                .max(center_out.len())
                .max(offset_out.len()),
        });
    }
    for i in 0..dim {
        center_out[i] = role_center[i] + filler_center[i];
        offset_out[i] = (filler_offset[i] - role_offset[i]).max(0.0);
    }
    Ok(())
}

/// Disjointness loss for two boxes that should not overlap (Box2EL formula).
///
/// Computes per-dimension overlap, then takes the L2 norm:
///
/// ```text
/// loss = ||relu(-|c_A - c_B| + o_A + o_B - margin)||_2
/// ```
///
/// Each dimension contributes `relu(-|c_A_i - c_B_i| + o_A_i + o_B_i - margin)`,
/// which is positive when the boxes overlap in that dimension (center distance
/// is less than the sum of offsets minus margin). The L2 norm of this vector
/// penalizes overlap proportionally across dimensions.
///
/// This scales correctly in high dimensions: both the per-dimension gaps and
/// the norm grow consistently, unlike a scalar hinge on L2 distance vs offset
/// sum (which would mix O(sqrt(d)) with O(d)).
///
/// # Arguments
///
/// * `center_a`, `offset_a` - First box
/// * `center_b`, `offset_b` - Second box
/// * `margin` - Per-dimension margin for required separation
pub fn disjointness_loss(
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
        let overlap =
            (-(center_a[i] - center_b[i]).abs() + offset_a[i] + offset_b[i] - margin).max(0.0);
        sum_sq += overlap * overlap;
    }

    Ok(sum_sq.sqrt())
}

/// Intersection non-emptiness loss: penalizes when two boxes that should
/// intersect are disjoint.
///
/// Returns `||relu(|c_a - c_b| - o_a - o_b)||_2`.
/// Zero when the boxes overlap in all dimensions.
pub fn intersection_nonempty_loss(
    center_a: &[f32],
    offset_a: &[f32],
    center_b: &[f32],
    offset_b: &[f32],
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
        let gap = (center_a[i] - center_b[i]).abs() - offset_a[i] - offset_b[i];
        let relu_gap = gap.max(0.0);
        sum_sq += relu_gap * relu_gap;
    }

    Ok(sum_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ---- helpers for proptest ----

    fn vec_f32(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-10.0f32..10.0, dim)
    }

    fn vec_f32_nonneg(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(0.0f32..10.0, dim)
    }

    // ---- existing unit tests ----

    #[test]
    fn test_inclusion_loss_contained() {
        // Box A (small) inside box B (large): center same, offset_a < offset_b
        let loss =
            el_inclusion_loss(&[0.0, 0.0], &[0.5, 0.5], &[0.0, 0.0], &[2.0, 2.0], 0.0).unwrap();
        assert!(loss < 1e-6, "contained box should have loss ~0, got {loss}");
    }

    #[test]
    fn test_inclusion_loss_not_contained() {
        // Box A extends beyond box B
        let loss = el_inclusion_loss(
            &[0.0, 0.0],
            &[3.0, 3.0], // wider than B
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.0,
        )
        .unwrap();
        assert!(
            loss > 0.0,
            "non-contained box should have loss > 0, got {loss}"
        );
    }

    #[test]
    fn test_inclusion_loss_with_margin() {
        // Contained without margin, but the margin makes offset_a "effectively" larger
        // loss = relu(|0-0| + 0.9 - 1.0 - margin)
        // margin=0: relu(-0.1) = 0
        // margin=-0.5: relu(0.4) = 0.4 (negative margin = tighter)
        let loss_no_margin = el_inclusion_loss(&[0.0], &[0.9], &[0.0], &[1.0], 0.0).unwrap();
        // Just barely not contained: offset_a > offset_b
        let loss_not_contained = el_inclusion_loss(&[0.0], &[1.5], &[0.0], &[1.0], 0.0).unwrap();
        assert!(
            loss_no_margin < 1e-6,
            "contained box, loss should be 0, got {loss_no_margin}"
        );
        assert!(
            loss_not_contained > 0.0,
            "non-contained should have loss > 0"
        );
    }

    #[test]
    fn test_translate() {
        let mut out = [0.0f32; 3];
        translate(&[1.0, 2.0, 3.0], &[0.5, -1.0, 0.0], &mut out).unwrap();
        assert_eq!(out, [1.5, 1.0, 3.0]);
    }

    #[test]
    fn test_compose_roles() {
        let mut c_out = [0.0f32; 2];
        let mut o_out = [0.0f32; 2];
        compose_roles(
            &[1.0, 2.0],
            &[0.5, 0.5],
            &[0.3, -0.1],
            &[0.2, 0.3],
            &mut c_out,
            &mut o_out,
        )
        .unwrap();
        assert!((c_out[0] - 1.3).abs() < 1e-6);
        assert!((c_out[1] - 1.9).abs() < 1e-6);
        assert!((o_out[0] - 0.7).abs() < 1e-6);
        assert!((o_out[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_existential_box() {
        let mut c_out = [0.0f32; 2];
        let mut o_out = [0.0f32; 2];
        existential_box(
            &[1.0, 0.0],
            &[0.3, 0.5],
            &[0.5, 1.0],
            &[1.0, 0.2],
            &mut c_out,
            &mut o_out,
        )
        .unwrap();
        // center = role + filler
        assert!((c_out[0] - 1.5).abs() < 1e-6);
        assert!((c_out[1] - 1.0).abs() < 1e-6);
        // offset = max(0, filler_offset - role_offset)
        assert!((o_out[0] - 0.7).abs() < 1e-6); // 1.0 - 0.3
        assert!(o_out[1] < 1e-6); // max(0, 0.2 - 0.5) = 0
    }

    #[test]
    fn test_disjointness_loss_far_apart() {
        // centers 100 apart, offsets 0.1 each, margin 1.0
        // per-dim: relu(-100 + 0.2 - 1.0) = relu(-100.8) = 0
        let loss = disjointness_loss(&[0.0], &[0.1], &[100.0], &[0.1], 1.0).unwrap();
        assert!(
            loss < 1e-6,
            "far-apart boxes should have disjointness loss ~0, got {loss}"
        );
    }

    #[test]
    fn test_disjointness_loss_overlapping() {
        // centers 0.5 apart, offsets 1.0 each, margin 0.0
        // per-dim: relu(-0.5 + 2.0 - 0.0) = 1.5
        let loss = disjointness_loss(&[0.0], &[1.0], &[0.5], &[1.0], 0.0).unwrap();
        assert!(
            (loss - 1.5).abs() < 1e-6,
            "overlapping boxes: expected 1.5, got {loss}"
        );
    }

    #[test]
    fn test_disjointness_loss_high_dim_scaling() {
        // In high-d, the per-dimension formula scales correctly.
        // 32-d boxes at same center, offset 1.0, margin 0.0:
        // each dim contributes relu(-0 + 2.0) = 2.0
        // L2 norm = sqrt(32 * 4.0) = sqrt(128)
        let d = 32;
        let loss = disjointness_loss(
            &vec![0.0; d],
            &vec![1.0; d],
            &vec![0.0; d],
            &vec![1.0; d],
            0.0,
        )
        .unwrap();
        let expected = (d as f32 * 4.0).sqrt();
        assert!(
            (loss - expected).abs() < 1e-4,
            "high-d disjointness: expected {expected}, got {loss}"
        );
    }

    #[test]
    fn test_intersection_nonempty_overlapping() {
        let loss =
            intersection_nonempty_loss(&[0.0, 0.0], &[2.0, 2.0], &[1.0, 1.0], &[2.0, 2.0]).unwrap();
        assert!(
            loss < 1e-6,
            "overlapping boxes should have loss ~0, got {loss}"
        );
    }

    #[test]
    fn test_intersection_nonempty_disjoint() {
        let loss = intersection_nonempty_loss(&[0.0], &[0.5], &[5.0], &[0.5]).unwrap();
        assert!(
            loss > 0.0,
            "disjoint boxes should have loss > 0, got {loss}"
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        assert!(el_inclusion_loss(&[0.0], &[1.0], &[0.0, 0.0], &[1.0, 1.0], 0.0).is_err());
        assert!(translate(&[0.0], &[1.0, 2.0], &mut [0.0]).is_err());
    }

    // ---- audit-driven regression tests ----

    /// Hand-computed 3D disjointness loss.
    ///
    /// A: center=[1, 2, 3], offset=[1.0, 0.5, 2.0]
    /// B: center=[2, 3, 1], offset=[0.5, 1.0, 0.5]
    /// margin = 0.0
    ///
    /// dim 0: relu(-|1-2| + 1.0 + 0.5) = relu(-1 + 1.5) = 0.5
    /// dim 1: relu(-|2-3| + 0.5 + 1.0) = relu(-1 + 1.5) = 0.5
    /// dim 2: relu(-|3-1| + 2.0 + 0.5) = relu(-2 + 2.5) = 0.5
    /// L2 = sqrt(0.25 + 0.25 + 0.25) = sqrt(0.75)
    #[test]
    fn test_disjointness_loss_per_dimension() {
        let loss = disjointness_loss(
            &[1.0, 2.0, 3.0],
            &[1.0, 0.5, 2.0],
            &[2.0, 3.0, 1.0],
            &[0.5, 1.0, 0.5],
            0.0,
        )
        .unwrap();
        let expected = 0.75_f32.sqrt();
        assert!(
            (loss - expected).abs() < 1e-6,
            "per-dimension disjointness: expected {expected}, got {loss}"
        );
    }

    /// Verify disjointness loss scales correctly in 32D and 128D.
    /// Per-dimension formula with identical co-located boxes:
    /// Each dim contributes relu(-0 + o_a + o_b) = 2*offset.
    /// L2 = sqrt(d * (2*offset)^2) = 2*offset*sqrt(d).
    /// This is O(sqrt(d)), NOT O(d) -- confirming no scaling mismatch.
    #[test]
    fn test_disjointness_loss_high_dim_scaling_128d() {
        for d in [32, 128] {
            let offset = 1.0_f32;
            let loss = disjointness_loss(
                &vec![0.0; d],
                &vec![offset; d],
                &vec![0.0; d],
                &vec![offset; d],
                0.0,
            )
            .unwrap();
            let expected = 2.0 * offset * (d as f32).sqrt();
            assert!(
                (loss - expected).abs() < 1e-3,
                "d={d}: expected {expected}, got {loss}"
            );
        }
    }

    // Existential box offset is always >= 0 (proptest, 8D).
    proptest! {
        #[test]
        fn test_existential_box_offset_nonnegative(
            role_c in vec_f32(8),
            role_o in vec_f32_nonneg(8),
            filler_c in vec_f32(8),
            filler_o in vec_f32_nonneg(8),
        ) {
            let dim = 8;
            let mut c_out = vec![0.0f32; dim];
            let mut o_out = vec![0.0f32; dim];
            existential_box(&role_c, &role_o, &filler_c, &filler_o, &mut c_out, &mut o_out).unwrap();
            for i in 0..dim {
                prop_assert!(o_out[i] >= 0.0,
                    "existential offset[{i}] must be >= 0, got {}", o_out[i]);
            }
        }
    }

    /// Box2EL subtractive offset produces smaller-or-equal box vs TransBox additive.
    /// subtractive: max(0, filler_offset - role_offset) <= filler_offset + role_offset
    #[test]
    fn test_existential_box_documentation_choice() {
        let dim = 4;
        let role_o = vec![0.3, 0.5, 0.1, 0.8];
        let filler_o = vec![1.0, 0.2, 0.5, 0.3];

        let mut c_out = vec![0.0f32; dim];
        let mut o_sub = vec![0.0f32; dim];
        existential_box(
            &vec![0.0; dim],
            &role_o,
            &vec![0.0; dim],
            &filler_o,
            &mut c_out,
            &mut o_sub,
        )
        .unwrap();

        // TransBox additive: offset = filler + role
        let o_add: Vec<f32> = filler_o
            .iter()
            .zip(role_o.iter())
            .map(|(f, r)| f + r)
            .collect();

        for i in 0..dim {
            assert!(
                o_sub[i] <= o_add[i] + 1e-7,
                "dim {i}: subtractive offset {:.4} should be <= additive {:.4}",
                o_sub[i],
                o_add[i]
            );
        }
    }

    /// Verify inclusion loss formula matches Box2EL for a known case.
    /// Box A: center=[1, 0], offset=[2, 1]  => interval [-1, 3] x [-1, 1]
    /// Box B: center=[0, 0], offset=[5, 5]  => interval [-5, 5] x [-5, 5]
    /// A is fully inside B, so loss should be 0.
    #[test]
    fn test_inclusion_loss_matches_box2el() {
        let loss =
            el_inclusion_loss(&[1.0, 0.0], &[2.0, 1.0], &[0.0, 0.0], &[5.0, 5.0], 0.0).unwrap();
        assert!(loss < 1e-6, "A inside B => loss should be 0, got {loss}");

        // Now a case where A is NOT inside B:
        // A: center=[4, 0], offset=[2, 1] => [2, 6] x [-1, 1]
        // B: center=[0, 0], offset=[3, 3] => [-3, 3] x [-3, 3]
        // dim 0: relu(|4-0| + 2 - 3) = relu(3) = 3
        // dim 1: relu(|0-0| + 1 - 3) = relu(-2) = 0
        // L2 = 3.0
        let loss2 =
            el_inclusion_loss(&[4.0, 0.0], &[2.0, 1.0], &[0.0, 0.0], &[3.0, 3.0], 0.0).unwrap();
        assert!((loss2 - 3.0).abs() < 1e-6, "expected 3.0, got {loss2}");
    }

    /// Compose roles is associative for centers: (a o b) o c == a o (b o c).
    /// Already tested in proptest above with 3D, this adds a deterministic 4D check.
    #[test]
    fn test_compose_roles_associative_deterministic() {
        let dim = 4;
        let ca = [1.0, 2.0, -1.0, 0.5];
        let oa = [0.5, 0.3, 0.2, 0.1];
        let cb = [0.0, -1.0, 3.0, 2.0];
        let ob = [0.1, 0.4, 0.5, 0.3];
        let cc = [-0.5, 1.0, 0.0, -2.0];
        let oc = [0.3, 0.2, 0.1, 0.6];

        // (a o b) o c
        let (mut ab_c, mut ab_o) = (vec![0.0f32; dim], vec![0.0f32; dim]);
        compose_roles(&ca, &oa, &cb, &ob, &mut ab_c, &mut ab_o).unwrap();
        let (mut left_c, mut left_o) = (vec![0.0f32; dim], vec![0.0f32; dim]);
        compose_roles(&ab_c, &ab_o, &cc, &oc, &mut left_c, &mut left_o).unwrap();

        // a o (b o c)
        let (mut bc_c, mut bc_o) = (vec![0.0f32; dim], vec![0.0f32; dim]);
        compose_roles(&cb, &ob, &cc, &oc, &mut bc_c, &mut bc_o).unwrap();
        let (mut right_c, mut right_o) = (vec![0.0f32; dim], vec![0.0f32; dim]);
        compose_roles(&ca, &oa, &bc_c, &bc_o, &mut right_c, &mut right_o).unwrap();

        for i in 0..dim {
            assert!(
                (left_c[i] - right_c[i]).abs() < 1e-6,
                "center[{i}] not associative: {} vs {}",
                left_c[i],
                right_c[i]
            );
            assert!(
                (left_o[i] - right_o[i]).abs() < 1e-6,
                "offset[{i}] not associative: {} vs {}",
                left_o[i],
                right_o[i]
            );
        }
    }

    /// intersection_nonempty_loss == 0 iff disjointness_loss > 0 for well-separated boxes.
    /// When boxes are clearly separated, intersection_nonempty should be positive
    /// and disjointness should be zero.
    #[test]
    fn test_intersection_nonempty_vs_disjointness_complement() {
        // Clearly separated boxes
        let ca = [0.0];
        let oa = [1.0];
        let cb = [10.0];
        let ob = [1.0];

        let disj = disjointness_loss(&ca, &oa, &cb, &ob, 0.0).unwrap();
        let inter = intersection_nonempty_loss(&ca, &oa, &cb, &ob).unwrap();

        assert!(
            disj < 1e-6,
            "separated boxes: disjointness should be 0, got {disj}"
        );
        assert!(
            inter > 0.0,
            "separated boxes: intersection_nonempty should be > 0, got {inter}"
        );

        // Fully overlapping boxes
        let ca2 = [0.0];
        let oa2 = [2.0];
        let cb2 = [0.5];
        let ob2 = [2.0];

        let disj2 = disjointness_loss(&ca2, &oa2, &cb2, &ob2, 0.0).unwrap();
        let inter2 = intersection_nonempty_loss(&ca2, &oa2, &cb2, &ob2).unwrap();

        assert!(
            disj2 > 0.0,
            "overlapping boxes: disjointness should be > 0, got {disj2}"
        );
        assert!(
            inter2 < 1e-6,
            "overlapping boxes: intersection_nonempty should be 0, got {inter2}"
        );
    }

    // ---- cross-validation with known geometry ----

    #[test]
    fn test_unit_box_inclusion() {
        // Box A centered at 0 with offset 0.5 (i.e., [-0.5, 0.5]^d)
        // Box B covering [-1, 2]^d => center 0.5, offset 1.5
        let d = 4;
        let center_a = vec![0.0f32; d];
        let offset_a = vec![0.5f32; d];
        let center_b = vec![0.5f32; d];
        let offset_b = vec![1.5f32; d];
        let loss = el_inclusion_loss(&center_a, &offset_a, &center_b, &offset_b, 0.0).unwrap();
        assert!(
            loss < 1e-6,
            "unit box inside [-1,2]^d should have loss 0, got {loss}"
        );
    }

    #[test]
    fn test_identity_translation_composition() {
        // Composing zero translations yields zero.
        let d = 3;
        let zero = vec![0.0f32; d];
        let off = vec![1.0f32; d];
        let mut c_out = vec![0.0f32; d];
        let mut o_out = vec![0.0f32; d];
        compose_roles(&zero, &off, &zero, &off, &mut c_out, &mut o_out).unwrap();
        for i in 0..d {
            assert!((c_out[i]).abs() < 1e-6, "identity centers should sum to 0");
            assert!((o_out[i] - 2.0).abs() < 1e-6, "identity offsets should sum");
        }
    }

    #[test]
    fn test_existential_zero_offset_role_preserves_filler() {
        // Role with zero offset: existential center = role_center + filler_center,
        // offset = max(0, filler_offset - 0) = filler_offset.
        let d = 3;
        let role_c = vec![1.0f32; d];
        let role_o = vec![0.0f32; d];
        let filler_c = vec![2.0f32; d];
        let filler_o = vec![0.5f32; d];
        let mut c_out = vec![0.0f32; d];
        let mut o_out = vec![0.0f32; d];
        existential_box(
            &role_c, &role_o, &filler_c, &filler_o, &mut c_out, &mut o_out,
        )
        .unwrap();
        for i in 0..d {
            assert!((c_out[i] - 3.0).abs() < 1e-6);
            assert!(
                (o_out[i] - 0.5).abs() < 1e-6,
                "filler offset preserved when role offset=0"
            );
        }
    }

    // ---- property tests ----

    proptest! {
        #[test]
        fn prop_inclusion_loss_nonneg(
            center_a in vec_f32(4),
            offset_a in vec_f32_nonneg(4),
            center_b in vec_f32(4),
            offset_b in vec_f32_nonneg(4),
            margin in -5.0f32..5.0,
        ) {
            let loss = el_inclusion_loss(&center_a, &offset_a, &center_b, &offset_b, margin).unwrap();
            prop_assert!(loss >= 0.0, "inclusion loss must be non-negative, got {loss}");
        }

        #[test]
        fn prop_inclusion_loss_zero_when_contained(
            center in vec_f32(4),
            offset_inner in vec_f32_nonneg(4),
            extra in proptest::collection::vec(0.0f32..5.0, 4usize),
        ) {
            // Build outer box with same center but larger offset.
            let offset_outer: Vec<f32> = offset_inner.iter().zip(extra.iter())
                .map(|(o, e)| o + e)
                .collect();
            let loss = el_inclusion_loss(&center, &offset_inner, &center, &offset_outer, 0.0).unwrap();
            prop_assert!(loss < 1e-6, "contained box should have ~0 loss, got {loss}");
        }

        #[test]
        fn prop_inclusion_loss_decreases_with_margin(
            center_a in vec_f32(4),
            offset_a in vec_f32_nonneg(4),
            center_b in vec_f32(4),
            offset_b in vec_f32_nonneg(4),
        ) {
            // Margin is subtracted in the relu: relu(|c_a-c_b| + o_a - o_b - margin).
            // Larger margin => smaller relu argument => less loss.
            let loss_0 = el_inclusion_loss(&center_a, &offset_a, &center_b, &offset_b, 0.0).unwrap();
            let loss_1 = el_inclusion_loss(&center_a, &offset_a, &center_b, &offset_b, 1.0).unwrap();
            prop_assert!(loss_0 >= loss_1 - 1e-5,
                "loss with margin=0 ({loss_0}) should be >= loss with margin=1 ({loss_1})");
        }

        #[test]
        fn prop_translate_additive(
            center in vec_f32(4),
            t1 in vec_f32(4),
            t2 in vec_f32(4),
        ) {
            let dim = 4;
            // translate(translate(c, t1), t2)
            let mut mid = vec![0.0f32; dim];
            let mut result_seq = vec![0.0f32; dim];
            translate(&center, &t1, &mut mid).unwrap();
            translate(&mid, &t2, &mut result_seq).unwrap();

            // translate(c, t1 + t2)
            let t_sum: Vec<f32> = t1.iter().zip(t2.iter()).map(|(a, b)| a + b).collect();
            let mut result_sum = vec![0.0f32; dim];
            translate(&center, &t_sum, &mut result_sum).unwrap();

            for i in 0..dim {
                prop_assert!((result_seq[i] - result_sum[i]).abs() < 1e-4,
                    "translate should be additive: dim {i}: {} vs {}", result_seq[i], result_sum[i]);
            }
        }

        #[test]
        fn prop_compose_roles_center_associative(
            ca in vec_f32(3),
            cb in vec_f32(3),
            cc in vec_f32(3),
            oa in vec_f32_nonneg(3),
            ob in vec_f32_nonneg(3),
            oc in vec_f32_nonneg(3),
        ) {
            let dim = 3;
            // (a o b) o c
            let mut cab_c = vec![0.0f32; dim];
            let mut cab_o = vec![0.0f32; dim];
            compose_roles(&ca, &oa, &cb, &ob, &mut cab_c, &mut cab_o).unwrap();
            let mut left_c = vec![0.0f32; dim];
            let mut left_o = vec![0.0f32; dim];
            compose_roles(&cab_c, &cab_o, &cc, &oc, &mut left_c, &mut left_o).unwrap();

            // a o (b o c)
            let mut cbc_c = vec![0.0f32; dim];
            let mut cbc_o = vec![0.0f32; dim];
            compose_roles(&cb, &ob, &cc, &oc, &mut cbc_c, &mut cbc_o).unwrap();
            let mut right_c = vec![0.0f32; dim];
            let mut right_o = vec![0.0f32; dim];
            compose_roles(&ca, &oa, &cbc_c, &cbc_o, &mut right_c, &mut right_o).unwrap();

            for i in 0..dim {
                prop_assert!((left_c[i] - right_c[i]).abs() < 1e-4,
                    "compose_roles centers should be associative");
                prop_assert!((left_o[i] - right_o[i]).abs() < 1e-4,
                    "compose_roles offsets should be associative");
            }
        }

        #[test]
        fn prop_existential_box_offset_nonneg(
            role_c in vec_f32(4),
            role_o in vec_f32_nonneg(4),
            filler_c in vec_f32(4),
            filler_o in vec_f32_nonneg(4),
        ) {
            let dim = 4;
            let mut c_out = vec![0.0f32; dim];
            let mut o_out = vec![0.0f32; dim];
            existential_box(&role_c, &role_o, &filler_c, &filler_o, &mut c_out, &mut o_out).unwrap();
            for i in 0..dim {
                prop_assert!(o_out[i] >= 0.0,
                    "existential offset must be non-negative, dim {i}: {}", o_out[i]);
            }
        }

        #[test]
        fn prop_disjointness_loss_symmetric(
            center_a in vec_f32(4),
            offset_a in vec_f32_nonneg(4),
            center_b in vec_f32(4),
            offset_b in vec_f32_nonneg(4),
            margin in 0.0f32..5.0,
        ) {
            let loss_ab = disjointness_loss(&center_a, &offset_a, &center_b, &offset_b, margin).unwrap();
            let loss_ba = disjointness_loss(&center_b, &offset_b, &center_a, &offset_a, margin).unwrap();
            prop_assert!((loss_ab - loss_ba).abs() < 1e-5,
                "disjointness_loss should be symmetric: {loss_ab} vs {loss_ba}");
        }

        #[test]
        fn prop_intersection_nonempty_loss_symmetric(
            center_a in vec_f32(4),
            offset_a in vec_f32_nonneg(4),
            center_b in vec_f32(4),
            offset_b in vec_f32_nonneg(4),
        ) {
            let loss_ab = intersection_nonempty_loss(&center_a, &offset_a, &center_b, &offset_b).unwrap();
            let loss_ba = intersection_nonempty_loss(&center_b, &offset_b, &center_a, &offset_a).unwrap();
            prop_assert!((loss_ab - loss_ba).abs() < 1e-5,
                "intersection_nonempty_loss should be symmetric: {loss_ab} vs {loss_ba}");
        }

        #[test]
        fn prop_dimension_mismatch_errors(
            a in vec_f32(3),
            b in vec_f32(4),
        ) {
            prop_assert!(el_inclusion_loss(&a, &a, &b, &b, 0.0).is_err());
            prop_assert!(translate(&a, &b, &mut vec![0.0; 3]).is_err());
            prop_assert!(disjointness_loss(&a, &a, &b, &b, 0.0).is_err());
            prop_assert!(intersection_nonempty_loss(&a, &a, &b, &b).is_err());
        }

        // -- disjointness_loss >= 0 --

        #[test]
        fn prop_disjointness_loss_nonneg(
            center_a in vec_f32(4),
            offset_a in vec_f32_nonneg(4),
            center_b in vec_f32(4),
            offset_b in vec_f32_nonneg(4),
            margin in 0.0f32..5.0,
        ) {
            let loss = disjointness_loss(&center_a, &offset_a, &center_b, &offset_b, margin).unwrap();
            prop_assert!(loss >= 0.0, "disjointness_loss must be >= 0, got {loss}");
        }

        // -- el_inclusion_loss >= 0 --

        #[test]
        fn prop_el_inclusion_loss_nonneg_explicit(
            center_a in vec_f32(4),
            offset_a in vec_f32_nonneg(4),
            center_b in vec_f32(4),
            offset_b in vec_f32_nonneg(4),
        ) {
            let loss = el_inclusion_loss(&center_a, &offset_a, &center_b, &offset_b, 0.0).unwrap();
            prop_assert!(loss >= 0.0, "el_inclusion_loss must be >= 0, got {loss}");
        }

        // -- intersection_nonempty_loss >= 0 --

        #[test]
        fn prop_intersection_nonempty_loss_nonneg(
            center_a in vec_f32(4),
            offset_a in vec_f32_nonneg(4),
            center_b in vec_f32(4),
            offset_b in vec_f32_nonneg(4),
        ) {
            let loss = intersection_nonempty_loss(&center_a, &offset_a, &center_b, &offset_b).unwrap();
            prop_assert!(loss >= 0.0, "intersection_nonempty_loss must be >= 0, got {loss}");
        }

        // -- compose_roles associativity (proptest variant) --

        #[test]
        fn prop_compose_roles_associative(
            ca in vec_f32(4),
            oa in vec_f32_nonneg(4),
            cb in vec_f32(4),
            ob in vec_f32_nonneg(4),
            cc in vec_f32(4),
            oc in vec_f32_nonneg(4),
        ) {
            let dim = 4;
            // (a o b) o c
            let (mut ab_c, mut ab_o) = (vec![0.0f32; dim], vec![0.0f32; dim]);
            compose_roles(&ca, &oa, &cb, &ob, &mut ab_c, &mut ab_o).unwrap();
            let (mut left_c, mut left_o) = (vec![0.0f32; dim], vec![0.0f32; dim]);
            compose_roles(&ab_c, &ab_o, &cc, &oc, &mut left_c, &mut left_o).unwrap();

            // a o (b o c)
            let (mut bc_c, mut bc_o) = (vec![0.0f32; dim], vec![0.0f32; dim]);
            compose_roles(&cb, &ob, &cc, &oc, &mut bc_c, &mut bc_o).unwrap();
            let (mut right_c, mut right_o) = (vec![0.0f32; dim], vec![0.0f32; dim]);
            compose_roles(&ca, &oa, &bc_c, &bc_o, &mut right_c, &mut right_o).unwrap();

            for i in 0..dim {
                prop_assert!(
                    (left_c[i] - right_c[i]).abs() < 1e-4,
                    "compose_roles center not associative at dim {i}"
                );
                prop_assert!(
                    (left_o[i] - right_o[i]).abs() < 1e-4,
                    "compose_roles offset not associative at dim {i}"
                );
            }
        }
    }
}
