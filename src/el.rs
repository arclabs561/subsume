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

/// Compute NF1 intersection loss: `C1 ⊓ C2 ⊑ D`.
///
/// The intersection of boxes C1 and C2 should be contained in box D.
/// Box intersection is computed element-wise: `max(min)` for lower bounds,
/// `min(max)` for upper bounds (using center/offset representation).
///
/// If C1 and C2 don't overlap (empty intersection), returns a center-attraction
/// surrogate `0.1 * ||center_c1 - center_c2||` to provide gradient signal that
/// pulls the conjuncts together until they overlap.
///
/// # Arguments
///
/// * `center_c1`, `offset_c1` - First conjunct box (center/offset)
/// * `center_c2`, `offset_c2` - Second conjunct box (center/offset)
/// * `center_d`, `offset_d` - Target box that should contain the intersection
/// * `margin` - Margin for containment
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if vectors differ in length.
pub fn el_intersection_loss(
    center_c1: &[f32],
    offset_c1: &[f32],
    center_c2: &[f32],
    offset_c2: &[f32],
    center_d: &[f32],
    offset_d: &[f32],
    margin: f32,
) -> Result<f32, BoxError> {
    let dim = center_c1.len();
    if offset_c1.len() != dim
        || center_c2.len() != dim
        || offset_c2.len() != dim
        || center_d.len() != dim
        || offset_d.len() != dim
    {
        return Err(BoxError::DimensionMismatch {
            expected: dim,
            actual: offset_c1
                .len()
                .max(center_c2.len())
                .max(offset_c2.len())
                .max(center_d.len())
                .max(offset_d.len()),
        });
    }

    // Compute intersection of C1 and C2 in center/offset form.
    // min/max of C1: [center - offset, center + offset]
    // Intersection lower = max(lower_c1, lower_c2), upper = min(upper_c1, upper_c2)
    let mut inter_center = vec![0.0f32; dim];
    let mut inter_offset = vec![0.0f32; dim];
    let mut empty = false;

    for i in 0..dim {
        let lo_c1 = center_c1[i] - offset_c1[i];
        let hi_c1 = center_c1[i] + offset_c1[i];
        let lo_c2 = center_c2[i] - offset_c2[i];
        let hi_c2 = center_c2[i] + offset_c2[i];

        let lo = lo_c1.max(lo_c2);
        let hi = hi_c1.min(hi_c2);

        if lo > hi {
            empty = true;
            break;
        }

        inter_center[i] = (lo + hi) / 2.0;
        inter_offset[i] = (hi - lo) / 2.0;
    }

    // Empty intersection: conjuncts are disjoint.
    // Return a center-attraction surrogate to provide gradient signal
    // that pulls C1 and C2 together until they overlap. Without this,
    // NF1 axioms contribute zero gradient at random init (dead zone).
    // Scale factor 0.1 prevents this surrogate from dominating the
    // inclusion loss once boxes do overlap.
    if empty {
        let mut center_dist_sq = 0.0f32;
        for i in 0..dim {
            let d = center_c1[i] - center_c2[i];
            center_dist_sq += d * d;
        }
        return Ok(0.1 * center_dist_sq.sqrt());
    }

    el_inclusion_loss(&inter_center, &inter_offset, center_d, offset_d, margin)
}

/// Compute role inclusion loss: `r ⊑ s` (RI6).
///
/// The role box for r should be contained in the role box for s.
/// This delegates to [`el_inclusion_loss`] on the role boxes.
pub fn el_role_inclusion_loss(
    center_r: &[f32],
    offset_r: &[f32],
    center_s: &[f32],
    offset_s: &[f32],
    margin: f32,
) -> Result<f32, BoxError> {
    el_inclusion_loss(center_r, offset_r, center_s, offset_s, margin)
}

/// Compute role chain loss: `r ∘ s ⊑ t` (RI7).
///
/// The composed role (r ∘ s) should be contained in role t.
pub fn el_role_chain_loss(
    center_r: &[f32],
    offset_r: &[f32],
    center_s: &[f32],
    offset_s: &[f32],
    center_t: &[f32],
    offset_t: &[f32],
    margin: f32,
) -> Result<f32, BoxError> {
    let dim = center_r.len();
    let mut composed_center = vec![0.0f32; dim];
    let mut composed_offset = vec![0.0f32; dim];
    compose_roles(
        center_r,
        offset_r,
        center_s,
        offset_s,
        &mut composed_center,
        &mut composed_offset,
    )?;
    el_inclusion_loss(
        &composed_center,
        &composed_offset,
        center_t,
        offset_t,
        margin,
    )
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
    fn test_dimension_mismatch() {
        assert!(el_inclusion_loss(&[0.0], &[1.0], &[0.0, 0.0], &[1.0, 1.0], 0.0).is_err());
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
            for (i, &val) in o_out.iter().enumerate() {
                prop_assert!(val >= 0.0,
                    "existential offset[{i}] must be >= 0, got {}", val);
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
            for (i, &val) in o_out.iter().enumerate() {
                prop_assert!(val >= 0.0,
                    "existential offset must be non-negative, dim {i}: {}", val);
            }
        }

        #[test]
        fn prop_dimension_mismatch_errors(
            a in vec_f32(3),
            b in vec_f32(4),
        ) {
            prop_assert!(el_inclusion_loss(&a, &a, &b, &b, 0.0).is_err());
        }

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

        // -- NF1 intersection tests --

        #[test]
        fn prop_intersection_loss_nonneg(
            c1 in vec_f32(4),
            o1 in vec_f32_nonneg(4),
            c2 in vec_f32(4),
            o2 in vec_f32_nonneg(4),
            cd in vec_f32(4),
            od in vec_f32_nonneg(4),
        ) {
            let loss = el_intersection_loss(&c1, &o1, &c2, &o2, &cd, &od, 0.0).unwrap();
            prop_assert!(loss >= 0.0, "intersection loss must be >= 0, got {loss}");
        }

        #[test]
        fn prop_intersection_loss_zero_when_d_is_universe_and_overlapping(
            c1 in vec_f32(4),
            o1 in vec_f32_nonneg(4),
        ) {
            // If C1 == C2 (guaranteed overlap) and D covers [-100, 100]^d,
            // the intersection fits inside D => loss = 0.
            // (Disjoint conjuncts get a center-attraction surrogate > 0.)
            let cd = vec![0.0f32; 4];
            let od = vec![100.0f32; 4];
            let loss = el_intersection_loss(&c1, &o1, &c1, &o1, &cd, &od, 0.0).unwrap();
            prop_assert!(loss < 1e-4, "overlapping C1==C2 with universal D should have loss~0, got {loss}");
        }

        #[test]
        fn prop_role_chain_loss_nonneg(
            cr in vec_f32(3),
            or_ in vec_f32_nonneg(3),
            cs in vec_f32(3),
            os in vec_f32_nonneg(3),
            ct in vec_f32(3),
            ot in vec_f32_nonneg(3),
        ) {
            let loss = el_role_chain_loss(&cr, &or_, &cs, &os, &ct, &ot, 0.0).unwrap();
            prop_assert!(loss >= 0.0, "role chain loss must be >= 0, got {loss}");
        }
    }

    #[test]
    fn test_intersection_loss_overlapping() {
        // C1: center=[0], offset=[2] => [-2, 2]
        // C2: center=[1], offset=[2] => [-1, 3]
        // Intersection: [-1, 2] => center=0.5, offset=1.5
        // D: center=[0.5], offset=[2] => [-1.5, 2.5]
        // Intersection fits inside D => loss = 0
        let loss =
            el_intersection_loss(&[0.0], &[2.0], &[1.0], &[2.0], &[0.5], &[2.0], 0.0).unwrap();
        assert!(
            loss < 1e-6,
            "intersection inside D should have ~0 loss, got {loss}"
        );
    }

    #[test]
    fn test_intersection_loss_disjoint() {
        // C1: center=[0], offset=[1] => [-1, 1]
        // C2: center=[5], offset=[1] => [4, 6]
        // Disjoint => center-attraction surrogate: 0.1 * |0 - 5| = 0.5
        let loss =
            el_intersection_loss(&[0.0], &[1.0], &[5.0], &[1.0], &[0.0], &[0.1], 0.0).unwrap();
        let expected = 0.1 * 5.0; // 0.1 * L2 center distance
        assert!(
            (loss - expected).abs() < 1e-6,
            "disjoint boxes should get center-attraction surrogate: expected {expected}, got {loss}"
        );
    }

    #[test]
    fn test_intersection_loss_not_contained() {
        // C1: center=[0], offset=[3] => [-3, 3]
        // C2: center=[0], offset=[3] => [-3, 3]
        // Intersection: [-3, 3] => center=0, offset=3
        // D: center=[0], offset=[1] => [-1, 1]
        // Intersection does NOT fit => loss > 0
        let loss =
            el_intersection_loss(&[0.0], &[3.0], &[0.0], &[3.0], &[0.0], &[1.0], 0.0).unwrap();
        assert!(
            loss > 0.0,
            "intersection overflows D => loss > 0, got {loss}"
        );
    }

    #[test]
    fn test_intersection_loss_disjoint_surrogate() {
        // Disjoint conjuncts: C1=[0,2], C2=[10,12]. No overlap.
        // Previously returned 0.0 (dead zone). Now returns center-attraction surrogate.
        let loss_disjoint =
            el_intersection_loss(&[1.0], &[1.0], &[11.0], &[1.0], &[0.0], &[20.0], 0.0).unwrap();
        assert!(
            loss_disjoint > 0.0,
            "disjoint conjuncts should produce non-zero surrogate loss, got {loss_disjoint}"
        );

        // Closer disjoint conjuncts should have lower surrogate loss.
        let loss_closer =
            el_intersection_loss(&[1.0], &[1.0], &[5.0], &[1.0], &[0.0], &[20.0], 0.0).unwrap();
        assert!(
            loss_closer < loss_disjoint,
            "closer conjuncts ({loss_closer}) should have lower loss than farther ({loss_disjoint})"
        );
    }

    #[test]
    fn test_intersection_loss_overlap_still_works() {
        // Overlapping conjuncts: C1=[0,4], C2=[2,6]. Intersection=[2,4].
        // D=[0,10] contains intersection => loss should be 0.
        let loss =
            el_intersection_loss(&[2.0], &[2.0], &[4.0], &[2.0], &[5.0], &[5.0], 0.0).unwrap();
        assert!(
            loss < 1e-6,
            "overlapping conjuncts with D containing intersection should have ~0 loss, got {loss}"
        );
    }

    #[test]
    fn test_role_chain_loss() {
        // r: center=[1], offset=[0.5] => [0.5, 1.5]
        // s: center=[0.5], offset=[0.3] => [0.2, 0.8]
        // composed: center=[1.5], offset=[0.8] => [0.7, 2.3]
        // t: center=[1.5], offset=[1.0] => [0.5, 2.5] -- contains composed
        let loss = el_role_chain_loss(&[1.0], &[0.5], &[0.5], &[0.3], &[1.5], &[1.0], 0.0).unwrap();
        assert!(loss < 1e-6, "composed fits in t => loss 0, got {loss}");
    }
}
