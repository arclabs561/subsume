//! Density matrix region embeddings for ontology concepts.
//!
//! Each concept is represented as a density matrix (positive semidefinite, trace 1)
//! in a `d`-dimensional Hilbert space. Subsumption is modeled via the Loewner order:
//! concept A is subsumed by concept B iff `rho_B - rho_A` is positive semidefinite.
//!
//! # Parametrization
//!
//! For efficiency, each concept stores a complex vector `v` (as interleaved
//! `[re_0, im_0, re_1, im_1, ...]`). The density matrix is the rank-1 projector:
//!
//! ```text
//! rho = |v><v| / <v|v>  =  v v^H / ||v||^2
//! ```
//!
//! This guarantees PSD and trace 1 by construction, with `2d` real parameters
//! per concept.
//!
//! # Scoring functions
//!
//! - **Fidelity**: for pure states, `F(rho_A, rho_B) = |<a|b>|^2 / (||a||^2 ||b||^2)`.
//!   Equals 1 for identical states, 0 for orthogonal states.
//! - **Subsumption loss**: penalizes when A is not subsumed by B.
//!   For pure states, `rho_B - rho_A` is PSD iff `rho_A = rho_B`, so the
//!   loss relaxes this to `max(0, 1 - fidelity)^2` (hinge on fidelity).
//! - **Disjointness loss**: `tr(rho_A rho_B) = fidelity(A, B)` for pure states.
//!   Penalizes non-zero overlap: `fidelity(A, B)^2`.
//!
//! # References
//!
//! - Garg et al. (2019), "Quantum Embedding of Knowledge for Reasoning" (NeurIPS)
//!
//! # Related crates
//! - [`qig`]: Full quantum information geometry (eigendecomposition-based fidelity, Bures distance). This module uses simplified pure-state formulas.

use crate::BoxError;
use serde::{Deserialize, Serialize};

/// A pure-state density matrix embedding: `rho = |v><v| / <v|v>`.
///
/// The complex vector is stored as interleaved real/imaginary parts:
/// `[re_0, im_0, re_1, im_1, ...]`, so `params.len() == 2 * dim`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DensityRegion {
    /// Interleaved complex vector `[re_0, im_0, re_1, im_1, ...]`.
    params: Vec<f32>,
    /// Hilbert space dimension (number of complex components).
    dim: usize,
}

impl DensityRegion {
    /// Create a new density region embedding.
    ///
    /// `params` must have length `2 * dim` (interleaved real/imaginary).
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if `params.len() != 2 * dim`.
    /// Returns [`BoxError::InvalidBounds`] if the vector is zero or contains non-finite values.
    pub fn new(params: Vec<f32>, dim: usize) -> Result<Self, BoxError> {
        if params.len() != 2 * dim {
            return Err(BoxError::DimensionMismatch {
                expected: 2 * dim,
                actual: params.len(),
            });
        }
        for (i, &p) in params.iter().enumerate() {
            if !p.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i / 2,
                    min: p as f64,
                    max: p as f64,
                });
            }
        }
        let norm_sq = squared_norm(&params);
        if norm_sq < 1e-12 {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: 0.0,
            });
        }
        Ok(Self { params, dim })
    }

    /// Hilbert space dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns a reference to the raw parameter vector.
    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// Squared norm `<v|v> = sum(re_i^2 + im_i^2)`.
    #[must_use]
    pub fn squared_norm(&self) -> f32 {
        squared_norm(&self.params)
    }

    /// Trace of the density matrix (always 1.0 for a valid pure state).
    ///
    /// This is a consistency check: `tr(rho) = <v|v> / <v|v> = 1`.
    #[must_use]
    pub fn trace(&self) -> f32 {
        1.0
    }

    /// Log-volume proxy: `ln(1 / d)` for a rank-1 projector.
    ///
    /// A pure state occupies a single ray in d-dimensional space,
    /// giving an effective volume of `1/d` (fraction of the full Hilbert space).
    #[must_use]
    pub fn log_volume(&self) -> f32 {
        -(self.dim as f32).ln()
    }
}

/// Squared norm of an interleaved complex vector.
fn squared_norm(params: &[f32]) -> f32 {
    params.iter().map(|x| x * x).sum()
}

/// Complex inner product `<a|b> = sum_i (conj(a_i) * b_i)`.
///
/// Returns `(real_part, imag_part)`.
fn complex_inner_product(a: &[f32], b: &[f32]) -> (f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    for i in (0..a.len()).step_by(2) {
        let a_re = a[i];
        let a_im = a[i + 1];
        let b_re = b[i];
        let b_im = b[i + 1];
        // conj(a) * b = (a_re - j*a_im) * (b_re + j*b_im)
        //             = (a_re*b_re + a_im*b_im) + j*(a_re*b_im - a_im*b_re)
        re += a_re * b_re + a_im * b_im;
        im += a_re * b_im - a_im * b_re;
    }
    (re, im)
}

/// Fidelity between two pure-state density matrices.
///
/// For pure states `rho_A = |a><a|/||a||^2` and `rho_B = |b><b|/||b||^2`:
///
/// ```text
/// F(rho_A, rho_B) = tr(rho_A rho_B) = |<a|b>|^2 / (||a||^2 * ||b||^2)
/// ```
///
/// Returns a value in `[0, 1]`: 1 for identical states, 0 for orthogonal states.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the two regions differ in dimension.
pub fn fidelity(a: &DensityRegion, b: &DensityRegion) -> Result<f32, BoxError> {
    if a.dim != b.dim {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim,
            actual: b.dim,
        });
    }
    let (re, im) = complex_inner_product(&a.params, &b.params);
    let inner_sq = re * re + im * im;
    let norm_a = a.squared_norm();
    let norm_b = b.squared_norm();
    Ok(inner_sq / (norm_a * norm_b))
}

/// Subsumption loss: penalizes when `child` is not subsumed by `parent`.
///
/// For pure states, strict Loewner order `rho_parent - rho_child >= 0` only
/// holds when the states are equal. We relax this to a soft loss using fidelity:
///
/// ```text
/// L_sub = max(0, 1 - F(child, parent))^2
/// ```
///
/// This is zero when `child == parent` (fidelity = 1) and increases as the
/// states diverge. The squared hinge provides a smooth gradient near the boundary.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the two regions differ in dimension.
pub fn subsumption_loss(child: &DensityRegion, parent: &DensityRegion) -> Result<f32, BoxError> {
    let f = fidelity(child, parent)?;
    let deficit = (1.0 - f).max(0.0);
    Ok(deficit * deficit)
}

/// Disjointness loss: penalizes overlap between two density matrices.
///
/// For pure states, `tr(rho_A rho_B) = F(A, B)`. Disjoint concepts should
/// have orthogonal state vectors, giving zero fidelity. The loss is:
///
/// ```text
/// L_disj = F(A, B)^2
/// ```
///
/// Zero for orthogonal states, increases as states align.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the two regions differ in dimension.
pub fn disjointness_loss(a: &DensityRegion, b: &DensityRegion) -> Result<f32, BoxError> {
    let f = fidelity(a, b)?;
    Ok(f * f)
}

/// Bures distance between two pure-state density matrices.
///
/// For pure states:
///
/// ```text
/// d_B(rho_A, rho_B)^2 = 2 * (1 - sqrt(F(rho_A, rho_B)))
/// ```
///
/// Returns the squared Bures distance in `[0, 2]`.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the two regions differ in dimension.
pub fn bures_distance_sq(a: &DensityRegion, b: &DensityRegion) -> Result<f32, BoxError> {
    let f = fidelity(a, b)?;
    Ok(2.0 * (1.0 - f.sqrt()))
}

/// Trace distance between two pure-state density matrices.
///
/// For pure states:
///
/// ```text
/// T(rho_A, rho_B) = sqrt(1 - F(rho_A, rho_B))
/// ```
///
/// Returns a value in `[0, 1]`: 0 for identical states, 1 for orthogonal states.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the two regions differ in dimension.
pub fn trace_distance(a: &DensityRegion, b: &DensityRegion) -> Result<f32, BoxError> {
    let f = fidelity(a, b)?;
    Ok((1.0 - f).max(0.0).sqrt())
}

/// Von Neumann entropy of a pure-state density matrix.
///
/// Pure states have zero entropy (they are maximally certain).
/// This is consistent: `S(|v><v|/||v||^2) = -tr(rho ln rho) = 0`
/// since eigenvalues are `{1, 0, 0, ...}` and `0 ln 0 = 0`.
#[must_use]
pub fn von_neumann_entropy(_state: &DensityRegion) -> f32 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_real(v: &[f32]) -> DensityRegion {
        let mut params = Vec::with_capacity(v.len() * 2);
        for &x in v {
            params.push(x);
            params.push(0.0);
        }
        DensityRegion::new(params, v.len()).unwrap()
    }

    fn make_complex(pairs: &[(f32, f32)]) -> DensityRegion {
        let mut params = Vec::with_capacity(pairs.len() * 2);
        for &(re, im) in pairs {
            params.push(re);
            params.push(im);
        }
        DensityRegion::new(params, pairs.len()).unwrap()
    }

    #[test]
    fn fidelity_identical() {
        let a = make_real(&[1.0, 0.0, 0.0]);
        let f = fidelity(&a, &a).unwrap();
        assert!((f - 1.0).abs() < 1e-6, "fidelity(a, a) = {f}, expected 1.0");
    }

    #[test]
    fn fidelity_orthogonal() {
        let a = make_real(&[1.0, 0.0, 0.0]);
        let b = make_real(&[0.0, 1.0, 0.0]);
        let f = fidelity(&a, &b).unwrap();
        assert!(f.abs() < 1e-6, "fidelity(orthogonal) = {f}, expected 0.0");
    }

    #[test]
    fn fidelity_scale_invariant() {
        let a = make_real(&[1.0, 0.0]);
        let b = make_real(&[3.0, 0.0]);
        let f = fidelity(&a, &b).unwrap();
        assert!(
            (f - 1.0).abs() < 1e-6,
            "fidelity should be scale-invariant: {f}"
        );
    }

    #[test]
    fn fidelity_complex() {
        // |a> = (1, i) / sqrt(2),  |b> = (1, -i) / sqrt(2)
        // <a|b> = 1*1 + (-i)(-i) = 1 + (-1) = 0  -> orthogonal
        let a = make_complex(&[(1.0, 0.0), (0.0, 1.0)]);
        let b = make_complex(&[(1.0, 0.0), (0.0, -1.0)]);
        let f = fidelity(&a, &b).unwrap();
        assert!(f.abs() < 1e-6, "fidelity((1,i),(1,-i)) = {f}, expected 0.0");
    }

    #[test]
    fn fidelity_partial_overlap() {
        // |a> = (1, 0), |b> = (1, 1)/sqrt(2)
        // <a|b> = 1, ||a||^2 = 1, ||b||^2 = 2
        // F = 1 / (1 * 2) = 0.5
        let a = make_real(&[1.0, 0.0]);
        let b = make_real(&[1.0, 1.0]);
        let f = fidelity(&a, &b).unwrap();
        assert!(
            (f - 0.5).abs() < 1e-6,
            "fidelity partial overlap = {f}, expected 0.5"
        );
    }

    #[test]
    fn subsumption_loss_identical_is_zero() {
        let a = make_real(&[1.0, 2.0, 3.0]);
        let loss = subsumption_loss(&a, &a).unwrap();
        assert!(loss.abs() < 1e-6, "subsumption_loss(a, a) = {loss}");
    }

    #[test]
    fn subsumption_loss_orthogonal_is_one() {
        let a = make_real(&[1.0, 0.0]);
        let b = make_real(&[0.0, 1.0]);
        let loss = subsumption_loss(&a, &b).unwrap();
        assert!(
            (loss - 1.0).abs() < 1e-6,
            "subsumption_loss(orthogonal) = {loss}, expected 1.0"
        );
    }

    #[test]
    fn disjointness_loss_orthogonal_is_zero() {
        let a = make_real(&[1.0, 0.0, 0.0]);
        let b = make_real(&[0.0, 1.0, 0.0]);
        let loss = disjointness_loss(&a, &b).unwrap();
        assert!(loss.abs() < 1e-6, "disjointness(orthogonal) = {loss}");
    }

    #[test]
    fn disjointness_loss_identical_is_one() {
        let a = make_real(&[1.0, 0.0]);
        let loss = disjointness_loss(&a, &a).unwrap();
        assert!(
            (loss - 1.0).abs() < 1e-6,
            "disjointness(a, a) = {loss}, expected 1.0"
        );
    }

    #[test]
    fn bures_distance_identical_is_zero() {
        let a = make_real(&[1.0, 2.0]);
        let d = bures_distance_sq(&a, &a).unwrap();
        assert!(d.abs() < 1e-6, "bures_distance_sq(a, a) = {d}");
    }

    #[test]
    fn bures_distance_orthogonal_is_two() {
        let a = make_real(&[1.0, 0.0]);
        let b = make_real(&[0.0, 1.0]);
        let d = bures_distance_sq(&a, &b).unwrap();
        assert!(
            (d - 2.0).abs() < 1e-6,
            "bures_distance_sq(orthogonal) = {d}, expected 2.0"
        );
    }

    #[test]
    fn trace_distance_bounds() {
        let a = make_real(&[1.0, 0.0]);
        let b = make_real(&[0.0, 1.0]);
        let t = trace_distance(&a, &b).unwrap();
        assert!(
            (t - 1.0).abs() < 1e-6,
            "trace_distance(orthogonal) = {t}, expected 1.0"
        );

        let t_self = trace_distance(&a, &a).unwrap();
        assert!(t_self.abs() < 1e-6, "trace_distance(a, a) = {t_self}");
    }

    #[test]
    fn von_neumann_entropy_is_zero() {
        let a = make_real(&[1.0, 2.0, 3.0]);
        assert_eq!(von_neumann_entropy(&a), 0.0);
    }

    #[test]
    fn rejects_zero_vector() {
        let result = DensityRegion::new(vec![0.0; 6], 3);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let result = DensityRegion::new(vec![1.0, 0.0, 0.0], 3);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_non_finite() {
        let result = DensityRegion::new(vec![f32::NAN, 0.0, 1.0, 0.0], 2);
        assert!(result.is_err());

        let result = DensityRegion::new(vec![f32::INFINITY, 0.0, 1.0, 0.0], 2);
        assert!(result.is_err());
    }

    #[test]
    fn trace_is_one() {
        let a = make_real(&[3.0, 4.0, 5.0]);
        assert_eq!(a.trace(), 1.0);
    }

    #[test]
    fn dimension_mismatch_error() {
        let a = make_real(&[1.0, 0.0]);
        let b = make_real(&[1.0, 0.0, 0.0]);
        assert!(fidelity(&a, &b).is_err());
        assert!(subsumption_loss(&a, &b).is_err());
        assert!(disjointness_loss(&a, &b).is_err());
        assert!(bures_distance_sq(&a, &b).is_err());
        assert!(trace_distance(&a, &b).is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_density(dim: usize) -> impl Strategy<Value = DensityRegion> {
        prop::collection::vec(-10.0f32..10.0, 2 * dim)
            .prop_filter_map("non-zero vector", move |params| {
                DensityRegion::new(params, dim).ok()
            })
    }

    fn arb_density_pair(dim: usize) -> impl Strategy<Value = (DensityRegion, DensityRegion)> {
        (arb_density(dim), arb_density(dim))
    }

    proptest! {
        #[test]
        fn prop_fidelity_in_unit_interval(
            (a, b) in arb_density_pair(4)
        ) {
            let f = fidelity(&a, &b).unwrap();
            prop_assert!(f >= -1e-6, "fidelity should be >= 0, got {f}");
            prop_assert!(f <= 1.0 + 1e-6, "fidelity should be <= 1, got {f}");
        }

        #[test]
        fn prop_self_fidelity_is_one(
            a in arb_density(4)
        ) {
            let f = fidelity(&a, &a).unwrap();
            prop_assert!((f - 1.0).abs() < 1e-5, "fidelity(a, a) = {f}, expected 1.0");
        }

        #[test]
        fn prop_fidelity_symmetric(
            (a, b) in arb_density_pair(4)
        ) {
            let f_ab = fidelity(&a, &b).unwrap();
            let f_ba = fidelity(&b, &a).unwrap();
            prop_assert!(
                (f_ab - f_ba).abs() < 1e-5,
                "fidelity should be symmetric: {f_ab} != {f_ba}"
            );
        }

        #[test]
        fn prop_subsumption_loss_nonneg(
            (a, b) in arb_density_pair(4)
        ) {
            let loss = subsumption_loss(&a, &b).unwrap();
            prop_assert!(loss >= -1e-6, "subsumption_loss should be >= 0, got {loss}");
        }

        #[test]
        fn prop_disjointness_loss_nonneg(
            (a, b) in arb_density_pair(4)
        ) {
            let loss = disjointness_loss(&a, &b).unwrap();
            prop_assert!(loss >= -1e-6, "disjointness_loss should be >= 0, got {loss}");
        }

        #[test]
        fn prop_bures_distance_nonneg(
            (a, b) in arb_density_pair(4)
        ) {
            let d = bures_distance_sq(&a, &b).unwrap();
            prop_assert!(d >= -1e-5, "bures_distance_sq should be >= 0, got {d}");
        }

        #[test]
        fn prop_trace_distance_in_unit_interval(
            (a, b) in arb_density_pair(4)
        ) {
            let t = trace_distance(&a, &b).unwrap();
            prop_assert!(t >= -1e-6, "trace_distance should be >= 0, got {t}");
            prop_assert!(t <= 1.0 + 1e-6, "trace_distance should be <= 1, got {t}");
        }

        #[test]
        fn prop_bures_distance_symmetric(
            (a, b) in arb_density_pair(4)
        ) {
            let d_ab = bures_distance_sq(&a, &b).unwrap();
            let d_ba = bures_distance_sq(&b, &a).unwrap();
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-5,
                "bures_distance should be symmetric: {d_ab} != {d_ba}"
            );
        }
    }
}
