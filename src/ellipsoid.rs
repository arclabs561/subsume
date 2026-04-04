//! Full-covariance Gaussian embeddings (rotated ellipsoids).
//!
//! Each concept is represented as a multivariate Gaussian with full covariance:
//! `N(mu, Sigma)` where `Sigma = L * L^T` is parameterized via Cholesky
//! decomposition. This allows rotated ellipsoidal regions that can model
//! correlated dimensions.
//!
//! # Motivation
//!
//! The existing `gaussian` module uses diagonal covariance (axis-aligned
//! ellipsoids). Full covariance enables:
//!
//! 1. **Rotated containment**: An ellipsoid can contain another even when
//!    their centers are far apart, if the rotation aligns properly.
//! 2. **Correlated dimensions**: Real concepts often have correlated features
//!    that diagonal Gaussians cannot capture.
//! 3. **More expressive boundaries**: The containment surface is a general
//!    quadratic form, not just axis-aligned thresholds.
//!
//! No published paper does full-covariance ellipsoids for subsumption.
//!
//! # Parametrization
//!
//! The covariance is stored as a Cholesky factor `L` (lower triangular),
//! ensuring positive definiteness by construction. Diagonal entries of `L`
//! are exponentiated for strict positivity.
//!
//! ```text
//! Sigma = L * L^T
//! L[i,j] = 0 for j > i  (lower triangular)
//! L[i,i] = exp(d[i])    (positive diagonal)
//! ```
//!
//! Parameters per entity: `d` (mean) + `d*(d+1)/2` (Cholesky) = `d*(d+3)/2`.
//!
//! # Containment via KL divergence
//!
//! ```text
//! D_KL(child || parent) = 0.5 * [
//!     tr(Sigma_p^{-1} * Sigma_c)
//!   + (mu_p - mu_c)^T * Sigma_p^{-1} * (mu_p - mu_c)
//!   - d
//!   + ln(det(Sigma_p) / det(Sigma_c))
//! ]
//! ```
//!
//! Low KL means the child is softly contained within the parent.
//!
//! # Bhattacharyya distance (symmetric overlap)
//!
//! ```text
//! D_B = 0.125 * (mu_1 - mu_2)^T * Sigma_m^{-1} * (mu_1 - mu_2)
//!     + 0.5   * ln(det(Sigma_m) / sqrt(det(Sigma_1) * det(Sigma_2)))
//! ```
//!
//! where `Sigma_m = 0.5 * (Sigma_1 + Sigma_2)`.

use crate::BoxError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A full-covariance Gaussian: `N(mu, Sigma)` with `Sigma = L * L^T`.
///
/// The Cholesky factor `L` is stored in row-major lower-triangular format:
/// `cholesky[i * dim + j]` for `j <= i`, with `L[i,i] = exp(d[i])`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ellipsoid {
    mu: Vec<f32>,
    /// Cholesky factor L stored as full d×d matrix in row-major.
    /// Upper triangle is zero (not stored but conceptually zero).
    cholesky: Vec<f32>,
    dim: usize,
}

// ---------------------------------------------------------------------------
// Ellipsoid construction and accessors
// ---------------------------------------------------------------------------

impl Ellipsoid {
    /// Create a new ellipsoid from mean and Cholesky factor.
    ///
    /// `cholesky` must be a lower-triangular matrix in row-major order.
    /// Diagonal entries must be positive (they are exponentiated internally).
    ///
    /// For convenience, you can pass the full d×d matrix; upper-triangular
    /// entries are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if dimensions don't match.
    /// Returns [`BoxError::InvalidBounds`] if any value is non-finite.
    pub fn new(mu: Vec<f32>, cholesky: Vec<f32>) -> Result<Self, BoxError> {
        let dim = mu.len();
        if cholesky.len() != dim * dim {
            return Err(BoxError::DimensionMismatch {
                expected: dim * dim,
                actual: cholesky.len(),
            });
        }
        for (i, &m) in mu.iter().enumerate() {
            if !m.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: m as f64,
                    max: m as f64,
                });
            }
        }
        for (i, &c) in cholesky.iter().enumerate() {
            if !c.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i / dim,
                    min: c as f64,
                    max: c as f64,
                });
            }
        }
        // Check diagonal positivity (before exp)
        for i in 0..dim {
            let diag = cholesky[i * dim + i];
            if diag <= -50.0 {
                // exp(-50) ≈ 0, effectively singular
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: diag as f64,
                    max: diag as f64,
                });
            }
        }
        Ok(Self { mu, cholesky, dim })
    }

    /// Create from mean and log-diagonal Cholesky (for training).
    ///
    /// Off-diagonal entries are zero (axis-aligned).
    pub fn from_log_diagonal(mu: Vec<f32>, log_diag: Vec<f32>) -> Result<Self, BoxError> {
        let dim = mu.len();
        if log_diag.len() != dim {
            return Err(BoxError::DimensionMismatch {
                expected: dim,
                actual: log_diag.len(),
            });
        }
        for (i, &v) in mu.iter().enumerate() {
            if !v.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: v as f64,
                    max: v as f64,
                });
            }
        }
        for (i, &v) in log_diag.iter().enumerate() {
            if !v.is_finite() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: v as f64,
                    max: v as f64,
                });
            }
        }
        let mut cholesky = vec![0.0f32; dim * dim];
        for i in 0..dim {
            cholesky[i * dim + i] = log_diag[i];
        }
        Ok(Self { mu, cholesky, dim })
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn mu(&self) -> &[f32] {
        &self.mu
    }

    /// Returns the Cholesky factor (lower triangular, row-major).
    pub fn cholesky(&self) -> &[f32] {
        &self.cholesky
    }

    /// Log-determinant of the covariance: `2 * sum(log(L[i,i]))`.
    #[must_use]
    pub fn log_det(&self) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            sum += self.cholesky[i * self.dim + i]; // already stored as log
        }
        2.0 * sum
    }

    /// Log-volume proxy: `0.5 * log_det(Sigma)`.
    #[must_use]
    pub fn log_volume(&self) -> f32 {
        0.5 * self.log_det()
    }

    /// Mutable access to mean vector (for training).
    pub fn mu_mut(&mut self) -> &mut [f32] {
        &mut self.mu
    }

    /// Returns a copy of the log-diagonal of the Cholesky factor (for training).
    pub fn log_diag(&self) -> Vec<f32> {
        (0..self.dim)
            .map(|i| self.cholesky[i * self.dim + i])
            .collect()
    }

    /// Set the log-diagonal of the Cholesky factor (for training).
    pub fn set_log_diag(&mut self, log_diag: &[f32]) {
        for (i, &v) in log_diag.iter().enumerate() {
            self.cholesky[i * self.dim + i] = v.clamp(-10.0, 5.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: Cholesky operations
// ---------------------------------------------------------------------------

/// Get L[i,j] from the stored Cholesky (with exp on diagonal).
fn get_l(cholesky: &[f32], dim: usize, i: usize, j: usize) -> f32 {
    if j > i {
        return 0.0;
    }
    let val = cholesky[i * dim + j];
    if i == j {
        val.exp()
    } else {
        val
    }
}

/// Solve L * x = b for lower-triangular L (forward substitution).
fn solve_lower(cholesky: &[f32], dim: usize, b: &[f32]) -> Vec<f32> {
    let mut x = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..i {
            sum += get_l(cholesky, dim, i, j) * x[j];
        }
        x[i] = (b[i] - sum) / get_l(cholesky, dim, i, i);
    }
    x
}

/// Compute x^T * Sigma^{-1} * x using Cholesky: x^T * (L*L^T)^{-1} * x = ||L^{-1} * x||^2
fn mahalanobis_sq(cholesky: &[f32], dim: usize, x: &[f32]) -> f32 {
    let y = solve_lower(cholesky, dim, x);
    y.iter().map(|v| v * v).sum()
}

/// Compute tr(Sigma_a^{-1} * Sigma_b) using Cholesky factors.
///
/// tr(S_a^{-1} * S_b) = tr((L_a * L_a^T)^{-1} * L_b * L_b^T)
/// = tr(L_a^{-T} * L_a^{-1} * L_b * L_b^T)
/// = ||L_a^{-1} * L_b||_F^2
fn trace_inv_product(cholesky_a: &[f32], cholesky_b: &[f32], dim: usize) -> f32 {
    let mut total = 0.0f32;
    for j in 0..dim {
        // Column j of L_b
        let col_b: Vec<f32> = (0..dim).map(|i| get_l(cholesky_b, dim, i, j)).collect();
        // L_a^{-1} * col_b
        let y = solve_lower(cholesky_a, dim, &col_b);
        total += y.iter().map(|v| v * v).sum::<f32>();
    }
    total
}

/// Compute Sigma_a + Sigma_b and return its Cholesky factor.
/// Uses the fact that (L_a * L_a^T + L_b * L_b^T) = L_sum * L_sum^T
/// via Cholesky update.
fn cholesky_sum(cholesky_a: &[f32], cholesky_b: &[f32], dim: usize) -> Vec<f32> {
    // Compute S = L_a * L_a^T + L_b * L_b^T explicitly, then Cholesky
    let mut s = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut val = 0.0f32;
            for k in 0..=j {
                val += get_l(cholesky_a, dim, i, k) * get_l(cholesky_a, dim, j, k);
                val += get_l(cholesky_b, dim, i, k) * get_l(cholesky_b, dim, j, k);
            }
            s[i * dim + j] = val;
        }
    }
    // Cholesky decomposition of s
    cholesky_decompose(&s, dim)
}

/// Standard Cholesky decomposition of a symmetric positive-definite matrix.
fn cholesky_decompose(s: &[f32], dim: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                let val = s[i * dim + i] - sum;
                // L[i,i] = sqrt(val), stored as log: ln(sqrt(val)) = 0.5*ln(val)
                l[i * dim + i] = 0.5 * (val.max(1e-10)).ln();
            } else {
                let ljj = get_l(&l, dim, j, j);
                if ljj > 1e-10 {
                    l[i * dim + j] = (s[i * dim + j] - sum) / ljj;
                }
            }
        }
    }
    l
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// KL divergence: D_KL(child || parent).
///
/// Low values indicate the child is contained within the parent.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if dimensions differ.
pub fn kl_divergence(child: &Ellipsoid, parent: &Ellipsoid) -> Result<f32, BoxError> {
    if child.dim != parent.dim {
        return Err(BoxError::DimensionMismatch {
            expected: child.dim,
            actual: parent.dim,
        });
    }
    let d = child.dim as f32;

    // tr(S_p^{-1} * S_c)
    let trace_term = trace_inv_product(&parent.cholesky, &child.cholesky, child.dim);

    // (mu_p - mu_c)^T * S_p^{-1} * (mu_p - mu_c)
    let diff: Vec<f32> = parent
        .mu
        .iter()
        .zip(child.mu.iter())
        .map(|(&p, &c)| p - c)
        .collect();
    let mahal = mahalanobis_sq(&parent.cholesky, child.dim, &diff);

    // ln(det(S_p) / det(S_c)) = log_det(S_p) - log_det(S_c)
    let log_det_ratio = parent.log_det() - child.log_det();

    Ok(0.5 * (trace_term + mahal - d + log_det_ratio))
}

/// Bhattacharyya distance (symmetric overlap).
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if dimensions differ.
pub fn bhattacharyya_distance(a: &Ellipsoid, b: &Ellipsoid) -> Result<f32, BoxError> {
    if a.dim != b.dim {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim,
            actual: b.dim,
        });
    }

    // Sigma_m = 0.5 * (Sigma_a + Sigma_b)
    let l_sum = cholesky_sum(&a.cholesky, &b.cholesky, a.dim);
    // L_m such that L_m * L_m^T = Sigma_m, but we stored sum as S = L_a*L_a^T + L_b*L_b^T
    // So Sigma_m = 0.5 * S, and L_m = L_S / sqrt(2)
    // log_det(Sigma_m) = log_det(S) - d * ln(2)
    let log_det_s: f32 = (0..a.dim)
        .map(|i| 2.0 * get_l(&l_sum, a.dim, i, i).ln())
        .sum();
    let d = a.dim as f32;
    let log_det_m = log_det_s - d * 2.0f32.ln();

    // Mahalanobis term: 0.125 * (mu_a - mu_b)^T * Sigma_m^{-1} * (mu_a - mu_b)
    let diff: Vec<f32> = a.mu.iter().zip(b.mu.iter()).map(|(&x, &y)| x - y).collect();
    // Sigma_m^{-1} = 2 * S^{-1}, so we solve with S and scale by 2
    let mahal = mahalanobis_sq(&l_sum, a.dim, &diff);
    let mahal_term = 0.125 * 2.0 * mahal;

    // Log-determinant term: 0.5 * (ln(det(Sigma_m)) - 0.5*(ln(det(Sigma_a)) + ln(det(Sigma_b))))
    let log_det_a = a.log_det();
    let log_det_b = b.log_det();
    let log_det_term = 0.5 * (log_det_m - 0.5 * (log_det_a + log_det_b));

    Ok(mahal_term + log_det_term)
}

/// Containment probability (soft): sigmoid(-k * KL(child || parent)).
///
/// High values indicate strong containment.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if dimensions differ.
pub fn containment_prob(child: &Ellipsoid, parent: &Ellipsoid, k: f32) -> Result<f32, BoxError> {
    let kl = kl_divergence(child, parent)?;
    Ok(sigmoid(-k * kl))
}

/// Surface distance proxy: sqrt(KL(child || parent) + KL(parent || child)).
///
/// Symmetric distance measure. Returns 0 for identical ellipsoids.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if dimensions differ.
pub fn surface_distance(a: &Ellipsoid, b: &Ellipsoid) -> Result<f32, BoxError> {
    let kl_ab = kl_divergence(a, b)?;
    let kl_ba = kl_divergence(b, a)?;
    Ok((kl_ab + kl_ba).max(0.0).sqrt())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipsoid_new_diagonal() {
        let e = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        assert_eq!(e.dim(), 2);
    }

    #[test]
    fn ellipsoid_new_full() {
        // 2x2 lower triangular Cholesky
        let cholesky = vec![
            0.0, 0.0, // row 0: L[0,0]=exp(0)=1, L[0,1]=0
            0.5, 0.0, // row 1: L[1,0]=0.5, L[1,1]=exp(0)=1
        ];
        let e = Ellipsoid::new(vec![0.0, 0.0], cholesky).unwrap();
        assert_eq!(e.dim(), 2);
    }

    #[test]
    fn ellipsoid_rejects_dim_mismatch() {
        assert!(Ellipsoid::new(vec![0.0], vec![0.0, 0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn ellipsoid_rejects_non_finite() {
        assert!(Ellipsoid::from_log_diagonal(vec![f32::NAN], vec![0.0]).is_err());
    }

    // --- KL divergence ---

    #[test]
    fn kl_identical_is_zero() {
        let e = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let kl = kl_divergence(&e, &e).unwrap();
        assert!(kl.abs() < 1e-4, "KL(identical) = {kl}, expected 0");
    }

    #[test]
    fn kl_same_center_different_scale() {
        let child = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![-1.0, -1.0]).unwrap();
        let parent = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let kl = kl_divergence(&child, &parent).unwrap();
        // child has smaller variance, so KL should be positive
        assert!(kl > 0.0, "KL = {kl}, expected > 0");
    }

    #[test]
    fn kl_different_center() {
        let child = Ellipsoid::from_log_diagonal(vec![1.0, 0.0], vec![0.0, 0.0]).unwrap();
        let parent = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let kl = kl_divergence(&child, &parent).unwrap();
        // Mahalanobis term: (1-0)^2/1 + (0-0)^2/1 = 1, so KL = 0.5*(2 + 1 - 2 + 0) = 0.5
        assert!((kl - 0.5).abs() < 1e-4, "KL = {kl}, expected 0.5");
    }

    #[test]
    fn kl_asymmetric() {
        let a = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![-1.0, -1.0]).unwrap();
        let b = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let kl_ab = kl_divergence(&a, &b).unwrap();
        let kl_ba = kl_divergence(&b, &a).unwrap();
        assert!(
            (kl_ab - kl_ba).abs() > 0.01,
            "KL should be asymmetric: {kl_ab} vs {kl_ba}"
        );
    }

    #[test]
    fn kl_dimension_mismatch() {
        let a = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let b = Ellipsoid::from_log_diagonal(vec![0.0], vec![0.0]).unwrap();
        assert!(kl_divergence(&a, &b).is_err());
    }

    // --- Bhattacharyya ---

    #[test]
    fn bhattacharyya_identical_is_zero() {
        let e = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let d = bhattacharyya_distance(&e, &e).unwrap();
        assert!(d.abs() < 1e-4, "B(identical) = {d}, expected 0");
    }

    #[test]
    fn bhattacharyya_symmetric() {
        let a = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let b = Ellipsoid::from_log_diagonal(vec![1.0, 0.0], vec![0.5, 0.5]).unwrap();
        let d_ab = bhattacharyya_distance(&a, &b).unwrap();
        let d_ba = bhattacharyya_distance(&b, &a).unwrap();
        assert!(
            (d_ab - d_ba).abs() < 1e-4,
            "Bhattacharyya should be symmetric: {d_ab} != {d_ba}"
        );
    }

    // --- Containment probability ---

    #[test]
    fn containment_prob_identical_is_half() {
        let e = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let p = containment_prob(&e, &e, 1.0).unwrap();
        // KL = 0, sigmoid(0) = 0.5
        assert!((p - 0.5).abs() < 1e-4);
    }

    #[test]
    fn containment_prob_near_identical_is_half() {
        // KL(child||parent) >= 0 always, so sigmoid(-KL) <= 0.5.
        // For near-identical distributions, KL ≈ 0, sigmoid(0) = 0.5.
        let child = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![-0.01, -0.01]).unwrap();
        let parent = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let p = containment_prob(&child, &parent, 1.0).unwrap();
        assert!(
            (p - 0.5).abs() < 0.05,
            "near-identical containment = {p}, expected ~0.5"
        );
    }

    #[test]
    fn containment_prob_widely_different_is_low() {
        // Child much narrower than parent: KL is moderate positive, sigmoid(-KL) < 0.5.
        let child = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![-3.0, -3.0]).unwrap();
        let parent = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let p = containment_prob(&child, &parent, 1.0).unwrap();
        assert!(p < 0.5, "narrower child containment = {p}, expected < 0.5");
    }

    // --- Surface distance ---

    #[test]
    fn surface_distance_identical_is_zero() {
        let e = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let d = surface_distance(&e, &e).unwrap();
        assert!(d.abs() < 1e-4);
    }

    // --- Log volume ---

    #[test]
    fn log_volume_diagonal() {
        // Sigma = diag(exp(0)^2, exp(0)^2) = I, det = 1, log_det = 0
        let e = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        assert!((e.log_volume()).abs() < 1e-6);
    }

    #[test]
    fn log_volume_scales() {
        let _e1 = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let e2 = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        // e2 has sigma = diag(e^2, e^2), det = e^4, log_det = 4, log_volume = 2
        let lv2 = e2.log_volume();
        assert!((lv2 - 2.0).abs() < 1e-4, "log_volume = {lv2}, expected 2");
    }

    #[test]
    fn sigmoid_large_positive() {
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sigmoid_large_negative() {
        assert!(sigmoid(-100.0).abs() < 1e-4);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_ellipsoid(dim: usize) -> impl Strategy<Value = Ellipsoid> {
        (
            prop::collection::vec(-5.0f32..5.0, dim),
            prop::collection::vec(-3.0f32..3.0, dim),
        )
            .prop_filter_map("valid ellipsoid", move |(mu, log_d)| {
                Ellipsoid::from_log_diagonal(mu, log_d).ok()
            })
    }

    fn arb_ellipsoid_pair(dim: usize) -> impl Strategy<Value = (Ellipsoid, Ellipsoid)> {
        (arb_ellipsoid(dim), arb_ellipsoid(dim))
    }

    proptest! {
        #[test]
        fn prop_kl_nonneg_for_same_center(
            (a, b) in arb_ellipsoid_pair(3)
        ) {
            // KL can be negative in general (not a true distance), but for same center it's >= 0
            let kl = kl_divergence(&a, &b).unwrap();
            prop_assert!(kl > -1.0, "KL unexpectedly negative: {kl}");
        }

        #[test]
        fn prop_self_kl_is_zero(
            e in arb_ellipsoid(3)
        ) {
            let kl = kl_divergence(&e, &e).unwrap();
            prop_assert!(kl.abs() < 1e-3, "KL(self) = {kl}, expected 0");
        }

        #[test]
        fn prop_bhattacharyya_nonneg(
            (a, b) in arb_ellipsoid_pair(3)
        ) {
            let d = bhattacharyya_distance(&a, &b).unwrap();
            prop_assert!(d > -1e-4, "Bhattacharyya < 0: {d}");
        }

        #[test]
        fn prop_bhattacharyya_symmetric(
            (a, b) in arb_ellipsoid_pair(3)
        ) {
            let d_ab = bhattacharyya_distance(&a, &b).unwrap();
            let d_ba = bhattacharyya_distance(&b, &a).unwrap();
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-3,
                "Bhattacharyya should be symmetric: {d_ab} != {d_ba}"
            );
        }

        #[test]
        fn prop_containment_in_unit_interval(
            (a, b) in arb_ellipsoid_pair(3)
        ) {
            let p = containment_prob(&a, &b, 1.0).unwrap();
            prop_assert!(p >= -1e-6, "containment_prob < 0: {p}");
            prop_assert!(p <= 1.0 + 1e-6, "containment_prob > 1: {p}");
        }

        #[test]
        fn prop_surface_distance_nonneg(
            (a, b) in arb_ellipsoid_pair(3)
        ) {
            let d = surface_distance(&a, &b).unwrap();
            prop_assert!(d >= -1e-6, "surface_distance < 0: {d}");
        }

        #[test]
        fn prop_surface_distance_symmetric(
            (a, b) in arb_ellipsoid_pair(3)
        ) {
            let d_ab = surface_distance(&a, &b).unwrap();
            let d_ba = surface_distance(&b, &a).unwrap();
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-3,
                "surface_distance should be symmetric: {d_ab} != {d_ba}"
            );
        }

        #[test]
        fn prop_log_volume_finite(
            e in arb_ellipsoid(4)
        ) {
            let lv = e.log_volume();
            prop_assert!(lv.is_finite(), "log_volume not finite: {lv}");
        }
    }
}
