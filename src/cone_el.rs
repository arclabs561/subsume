//! EL++ ontology embedding primitives for cones.
//!
//! Parallel to [`el`](crate::el) (which uses boxes), this module provides
//! loss functions and operations for embedding OWL EL++ ontologies using
//! cone geometry. The angular containment model has one key advantage over
//! boxes: **closure under complement**, enabling negation in logical queries.
//!
//! # EL++ Normal Forms (cone adaptation)
//!
//! | Form | Pattern | Cone geometric meaning |
//! |------|---------|------------------------|
//! | NF2 | C ⊑ D | Cone C fits inside cone D (per-dim angular containment) |
//! | NF3 | C ⊑ ∃r.D | Project C through relation r, result overlaps D |
//! | NF4 | ∃r.C ⊑ D | Inverse projection of C through r fits inside D |
//! | NF5 | C ⊓ D ⊑ ⊥ | Intersection of C and D has minimal aperture |
//! | RI6 | r ⊑ s | Role inclusion (axis/aperture containment) |
//! | RI7 | r ∘ s ⊑ t | Role composition (additive, like TransBox) |
//!
//! # References
//!
//! - Zhang et al. (NeurIPS 2021), "ConE: Cone Embeddings for Multi-Hop Reasoning"
//! - Jackermeier & Chen (WWW 2024), "Box2EL: Dual Box Embeddings for EL++"
//! - Chen et al. (2024 survey), "Ontology Embedding: Methods, Applications, Resources"

use crate::cone::ConeError;

/// Compute EL++ inclusion loss for cones: C ⊑ D.
///
/// Measures how much cone C fails to be contained in cone D, per dimension.
/// The loss is zero when C's angular sector fits inside D's sector in every
/// dimension (with margin).
///
/// Per-dimension loss:
/// ```text
/// violation[i] = max(0, |sin((axis_c[i] - axis_d[i]) / 2)| + |sin(aper_c[i] / 2)| - |sin(aper_d[i] / 2)| - margin)
/// loss = sqrt(sum(violation[i]^2))
/// ```
///
/// # Errors
///
/// Returns [`ConeError::DimensionMismatch`] if vectors differ in length.
pub fn cone_inclusion_loss(
    axes_c: &[f32],
    apertures_c: &[f32],
    axes_d: &[f32],
    apertures_d: &[f32],
    margin: f32,
) -> Result<f32, ConeError> {
    let dim = axes_c.len();
    if apertures_c.len() != dim || axes_d.len() != dim || apertures_d.len() != dim {
        return Err(ConeError::DimensionMismatch {
            expected: dim,
            actual: apertures_c.len().max(axes_d.len()).max(apertures_d.len()),
        });
    }

    let mut sum_sq = 0.0f32;
    for i in 0..dim {
        let axis_dist = ((axes_c[i] - axes_d[i]) / 2.0).sin().abs();
        let c_half = (apertures_c[i] / 2.0).sin().abs();
        let d_half = (apertures_d[i] / 2.0).sin().abs();
        let violation = (axis_dist + c_half - d_half - margin).max(0.0);
        sum_sq += violation * violation;
    }

    Ok(sum_sq.sqrt())
}

/// Compute the existential restriction cone ∃r.C (NF3/NF4).
///
/// - axis: `axis_role[i] + axis_filler[i]` (modular, wrapped to \[-pi, pi\])
/// - aperture: `max(0, aperture_filler[i] - aperture_role[i])` (restrictive, Box2EL style)
///
/// # Errors
///
/// Returns [`ConeError::DimensionMismatch`] if vectors differ in length.
pub fn cone_existential(
    role_axes: &[f32],
    role_apertures: &[f32],
    filler_axes: &[f32],
    filler_apertures: &[f32],
    axes_out: &mut [f32],
    apertures_out: &mut [f32],
) -> Result<(), ConeError> {
    let dim = role_axes.len();
    if role_apertures.len() != dim
        || filler_axes.len() != dim
        || filler_apertures.len() != dim
        || axes_out.len() != dim
        || apertures_out.len() != dim
    {
        return Err(ConeError::DimensionMismatch {
            expected: dim,
            actual: role_apertures
                .len()
                .max(filler_axes.len())
                .max(filler_apertures.len())
                .max(axes_out.len())
                .max(apertures_out.len()),
        });
    }

    for i in 0..dim {
        let mut axis = role_axes[i] + filler_axes[i];
        axis %= 2.0 * std::f32::consts::PI;
        if axis > std::f32::consts::PI {
            axis -= 2.0 * std::f32::consts::PI;
        } else if axis < -std::f32::consts::PI {
            axis += 2.0 * std::f32::consts::PI;
        }
        axes_out[i] = axis;
        apertures_out[i] = (filler_apertures[i] - role_apertures[i]).max(0.0);
    }

    Ok(())
}

/// Compose two relation cones (RI7: r ∘ s ⊑ t).
///
/// - axis: additive (modular)
/// - aperture: additive (clamped to \[0, pi\])
///
/// # Errors
///
/// Returns [`ConeError::DimensionMismatch`] if vectors differ in length.
pub fn compose_cone_roles(
    axes_r: &[f32],
    apertures_r: &[f32],
    axes_s: &[f32],
    apertures_s: &[f32],
    axes_out: &mut [f32],
    apertures_out: &mut [f32],
) -> Result<(), ConeError> {
    let dim = axes_r.len();
    if apertures_r.len() != dim
        || axes_s.len() != dim
        || apertures_s.len() != dim
        || axes_out.len() != dim
        || apertures_out.len() != dim
    {
        return Err(ConeError::DimensionMismatch {
            expected: dim,
            actual: apertures_r
                .len()
                .max(axes_s.len())
                .max(apertures_s.len())
                .max(axes_out.len())
                .max(apertures_out.len()),
        });
    }

    for i in 0..dim {
        let mut axis = axes_r[i] + axes_s[i];
        axis %= 2.0 * std::f32::consts::PI;
        if axis > std::f32::consts::PI {
            axis -= 2.0 * std::f32::consts::PI;
        } else if axis < -std::f32::consts::PI {
            axis += 2.0 * std::f32::consts::PI;
        }
        axes_out[i] = axis;
        apertures_out[i] = (apertures_r[i] + apertures_s[i]).clamp(0.0, std::f32::consts::PI);
    }

    Ok(())
}

/// Compute disjointness loss for cones: C ⊓ D ⊑ ⊥ (NF5).
///
/// Encourages the intersection of C and D to be empty (minimal overlap).
/// Loss is zero when the angular distance between axes exceeds the sum of half-widths.
///
/// # Errors
///
/// Returns [`ConeError::DimensionMismatch`] if vectors differ in length.
pub fn cone_disjointness_loss(
    axes_c: &[f32],
    apertures_c: &[f32],
    axes_d: &[f32],
    apertures_d: &[f32],
) -> Result<f32, ConeError> {
    let dim = axes_c.len();
    if apertures_c.len() != dim || axes_d.len() != dim || apertures_d.len() != dim {
        return Err(ConeError::DimensionMismatch {
            expected: dim,
            actual: apertures_c.len().max(axes_d.len()).max(apertures_d.len()),
        });
    }

    let mut total = 0.0f32;
    for i in 0..dim {
        let axis_dist = ((axes_c[i] - axes_d[i]) / 2.0).sin().abs();
        let c_half = (apertures_c[i] / 2.0).sin().abs();
        let d_half = (apertures_d[i] / 2.0).sin().abs();
        let overlap = (c_half + d_half - axis_dist).max(0.0);
        total += overlap;
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::f32::consts::PI;

    fn vec_angle(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-PI..PI, dim)
    }

    fn vec_aperture(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(0.01f32..PI - 0.01, dim)
    }

    #[test]
    fn inclusion_loss_zero_when_contained() {
        let loss =
            cone_inclusion_loss(&[0.0, 0.0], &[0.3, 0.3], &[0.0, 0.0], &[1.5, 1.5], 0.0).unwrap();
        assert!(
            loss < 1e-5,
            "contained cone should have ~0 loss, got {loss}"
        );
    }

    #[test]
    fn inclusion_loss_positive_when_not_contained() {
        let loss =
            cone_inclusion_loss(&[0.0, 0.0], &[2.0, 2.0], &[0.0, 0.0], &[0.5, 0.5], 0.0).unwrap();
        assert!(loss > 0.0, "non-contained cone should have loss > 0");
    }

    #[test]
    fn inclusion_loss_dimension_mismatch() {
        assert!(cone_inclusion_loss(&[0.0], &[0.5], &[0.0, 0.0], &[0.5, 0.5], 0.0).is_err());
    }

    #[test]
    fn disjointness_loss_zero_when_separated() {
        let loss =
            cone_disjointness_loss(&[0.0, 0.0], &[0.2, 0.2], &[PI, PI], &[0.2, 0.2]).unwrap();
        assert!(
            loss < 1e-3,
            "separated cones should have ~0 loss, got {loss}"
        );
    }

    #[test]
    fn disjointness_loss_positive_when_overlapping() {
        let loss =
            cone_disjointness_loss(&[0.0, 0.0], &[1.0, 1.0], &[0.0, 0.0], &[1.0, 1.0]).unwrap();
        assert!(loss > 0.0, "overlapping cones should have loss > 0");
    }

    #[test]
    fn existential_zero_role_preserves_filler() {
        let dim = 3;
        let mut axes_out = vec![0.0f32; dim];
        let mut aper_out = vec![0.0f32; dim];
        cone_existential(
            &[0.0; 3],
            &[0.0; 3],
            &[0.5, -0.3, 1.0],
            &[0.8, 0.5, 1.2],
            &mut axes_out,
            &mut aper_out,
        )
        .unwrap();
        assert!((axes_out[0] - 0.5).abs() < 1e-6);
        assert!((aper_out[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn compose_roles_adds_axes_and_apertures() {
        let dim = 2;
        let mut axes_out = vec![0.0f32; dim];
        let mut aper_out = vec![0.0f32; dim];
        compose_cone_roles(
            &[0.3, -0.1],
            &[0.5, 0.3],
            &[0.2, 0.4],
            &[0.2, 0.1],
            &mut axes_out,
            &mut aper_out,
        )
        .unwrap();
        assert!((axes_out[0] - 0.5).abs() < 1e-6);
        assert!((aper_out[0] - 0.7).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn prop_inclusion_loss_nonneg(
            axes_c in vec_angle(4),
            aper_c in vec_aperture(4),
            axes_d in vec_angle(4),
            aper_d in vec_aperture(4),
        ) {
            let loss = cone_inclusion_loss(&axes_c, &aper_c, &axes_d, &aper_d, 0.0).unwrap();
            prop_assert!(loss >= 0.0, "inclusion loss must be >= 0, got {loss}");
        }

        #[test]
        fn prop_inclusion_loss_zero_when_same_axis_wider_parent(
            axes in vec_angle(4),
            aper_inner in vec_aperture(4),
            extra in proptest::collection::vec(0.1f32..1.0, 4usize),
        ) {
            let aper_outer: Vec<f32> = aper_inner.iter().zip(extra.iter())
                .map(|(a, e)| (a + e).min(PI - 0.01))
                .collect();
            let loss = cone_inclusion_loss(&axes, &aper_inner, &axes, &aper_outer, 0.0).unwrap();
            prop_assert!(loss < 1e-4, "same-axis wider parent should have ~0 loss, got {loss}");
        }

        #[test]
        fn prop_disjointness_loss_nonneg(
            axes_c in vec_angle(4),
            aper_c in vec_aperture(4),
            axes_d in vec_angle(4),
            aper_d in vec_aperture(4),
        ) {
            let loss = cone_disjointness_loss(&axes_c, &aper_c, &axes_d, &aper_d).unwrap();
            prop_assert!(loss >= 0.0, "disjointness loss must be >= 0, got {loss}");
        }

        #[test]
        fn prop_existential_aperture_nonneg(
            role_ax in vec_angle(4),
            role_ap in vec_aperture(4),
            fill_ax in vec_angle(4),
            fill_ap in vec_aperture(4),
        ) {
            let dim = 4;
            let mut axes_out = vec![0.0f32; dim];
            let mut aper_out = vec![0.0f32; dim];
            cone_existential(&role_ax, &role_ap, &fill_ax, &fill_ap, &mut axes_out, &mut aper_out).unwrap();
            for (i, &val) in aper_out.iter().enumerate() {
                prop_assert!(val >= 0.0, "existential aperture[{i}] must be >= 0, got {}", val);
            }
        }
    }
}
