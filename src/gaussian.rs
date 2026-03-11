//! Diagonal Gaussian box embeddings for taxonomy expansion.
//!
//! This module implements the Gaussian representation from
//! **TaxoBell** (WWW 2026, [arXiv:2601.09633](https://arxiv.org/abs/2601.09633)),
//! which derives diagonal Gaussians from box center/offset parameters and uses
//! KL divergence (asymmetric containment) and Bhattacharyya coefficient (symmetric
//! overlap) as scoring functions.
//!
//! # Motivation
//!
//! Standard box embeddings use hard containment (min/max coordinates). Gaussian
//! boxes replace the uniform distribution inside a box with a diagonal Gaussian,
//! gaining two advantages:
//!
//! 1. **Smooth asymmetric scores**: KL divergence provides dense gradients for
//!    parent-child containment, unlike hard containment which has flat regions.
//! 2. **Symmetric overlap**: The Bhattacharyya coefficient measures concept
//!    relatedness without assuming a hierarchy direction.
//!
//! # Formulas
//!
//! For two diagonal Gaussians `p = N(mu_p, diag(sigma_p^2))` and
//! `q = N(mu_q, diag(sigma_q^2))`:
//!
//! ## KL Divergence (child -> parent containment)
//!
//! ```text
//! D_KL(child || parent) = 0.5 * sum_i [
//!     (sigma_c_i / sigma_p_i)^2
//!   + (mu_p_i - mu_c_i)^2 / sigma_p_i^2
//!   - 1
//!   + 2 * ln(sigma_p_i / sigma_c_i)
//! ]
//! ```
//!
//! ## Bhattacharyya Coefficient (symmetric overlap)
//!
//! ```text
//! BC(p, q) = exp(-D_B)
//! D_B = 0.125 * sum_i (mu_1_i - mu_2_i)^2 / sigma_m_i
//!     + 0.5   * sum_i [ln(sigma_m_i) - 0.5 * (ln(s1_i^2) + ln(s2_i^2))]
//! ```
//!
//! where `sigma_m_i = (s1_i^2 + s2_i^2) / 2`.
//!
//! # References
//!
//! - TaxoBell (WWW 2026): taxonomy expansion with Gaussian boxes
//! - Li et al. (2019): "Smoothing the Geometry of Probabilistic Box Embeddings" (ICLR 2019)

use crate::BoxError;
use serde::Serialize;

/// A diagonal Gaussian embedding: `N(mu, diag(sigma^2))`.
///
/// Each dimension is independent, with its own mean and standard deviation.
/// This is the representation used by TaxoBell for taxonomy expansion.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GaussianBox {
    /// Mean vector (center of the Gaussian).
    mu: Vec<f32>,
    /// Standard deviation vector (must be positive).
    sigma: Vec<f32>,
}

impl GaussianBox {
    /// Create a new Gaussian box.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if `mu` and `sigma` differ in length.
    /// Returns [`BoxError::InvalidBounds`] if any sigma value is non-positive.
    pub fn new(mu: Vec<f32>, sigma: Vec<f32>) -> Result<Self, BoxError> {
        if mu.len() != sigma.len() {
            return Err(BoxError::DimensionMismatch {
                expected: mu.len(),
                actual: sigma.len(),
            });
        }
        for (i, &s) in sigma.iter().enumerate() {
            if s.is_nan() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: 0.0,
                    max: s as f64,
                });
            }
            if s <= 0.0 {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: 0.0,
                    max: s as f64,
                });
            }
        }
        for (i, &m) in mu.iter().enumerate() {
            if m.is_nan() {
                return Err(BoxError::InvalidBounds {
                    dim: i,
                    min: m as f64,
                    max: m as f64,
                });
            }
        }
        Ok(Self { mu, sigma })
    }

    /// Create a unit Gaussian (mu=0, sigma=1) of the given dimensionality.
    #[must_use]
    pub fn unit(dim: usize) -> Self {
        Self {
            mu: vec![0.0; dim],
            sigma: vec![1.0; dim],
        }
    }

    /// Dimensionality of this Gaussian.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.mu.len()
    }

    /// Returns a reference to the mean vector.
    pub fn mu(&self) -> &[f32] {
        &self.mu
    }

    /// Returns a reference to the standard deviation vector.
    pub fn sigma(&self) -> &[f32] {
        &self.sigma
    }

    /// Log-volume (log-determinant of covariance = sum of log-sigmas).
    ///
    /// For a diagonal Gaussian, `log det(Sigma) = 2 * sum(ln(sigma_i))`,
    /// so `log sqrt(det(Sigma)) = sum(ln(sigma_i))`.
    #[must_use]
    pub fn log_volume(&self) -> f32 {
        self.sigma.iter().map(|s| s.ln()).sum()
    }

    /// Create from center/offset box parameters (TaxoBell parameterization).
    ///
    /// The TaxoBell encoder outputs `(center, offset)` via two MLPs, then:
    /// - `mu = center`
    /// - `sigma = softplus(offset)` (ensures positivity)
    pub fn from_center_offset(center: Vec<f32>, offset: Vec<f32>) -> Result<Self, BoxError> {
        if center.len() != offset.len() {
            return Err(BoxError::DimensionMismatch {
                expected: center.len(),
                actual: offset.len(),
            });
        }
        let sigma: Vec<f32> = offset
            .iter()
            .map(|&o| {
                // softplus: ln(1 + exp(o))
                if o > 20.0 {
                    o
                } else if o < -20.0 {
                    1e-7
                } else {
                    o.exp().ln_1p()
                }
            })
            .collect();
        Ok(Self { mu: center, sigma })
    }
}

impl<'de> serde::Deserialize<'de> for GaussianBox {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Raw {
            mu: Vec<f32>,
            sigma: Vec<f32>,
        }
        let raw = Raw::deserialize(deserializer)?;
        GaussianBox::new(raw.mu, raw.sigma).map_err(serde::de::Error::custom)
    }
}

/// KL divergence between two diagonal Gaussians: `D_KL(child || parent)`.
///
/// This measures how much the child distribution diverges from the parent.
/// In taxonomy expansion, a small `D_KL(child || parent)` means the child
/// concept "fits inside" the parent concept.
///
/// # Formula
///
/// ```text
/// D_KL(c || p) = 0.5 * sum_i [
///     (sigma_c / sigma_p)^2
///   + (mu_p - mu_c)^2 / sigma_p^2
///   - 1
///   + 2 * ln(sigma_p / sigma_c)
/// ]
/// ```
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the two Gaussians differ in dimensionality.
pub fn kl_divergence(child: &GaussianBox, parent: &GaussianBox) -> Result<f32, BoxError> {
    if child.dim() != parent.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: child.dim(),
            actual: parent.dim(),
        });
    }

    const EPS: f32 = 1e-7;

    let mut sum = 0.0f32;
    for i in 0..child.dim() {
        let sc = child.sigma[i].max(EPS);
        let sp = parent.sigma[i].max(EPS);
        let dm = parent.mu[i] - child.mu[i];

        let ratio_sq = (sc / sp).powi(2);
        let mean_sq = dm * dm / (sp * sp);
        let log_ratio = 2.0 * (sp / sc).ln();

        sum += ratio_sq + mean_sq - 1.0 + log_ratio;
    }

    Ok(0.5 * sum)
}

/// Bhattacharyya distance between two diagonal Gaussians.
///
/// # Formula
///
/// ```text
/// D_B = 0.125 * sum_i (mu1 - mu2)^2 / sigma_m_i
///     + 0.5   * sum_i [ln(sigma_m_i) - 0.5 * (ln(s1^2) + ln(s2^2))]
/// ```
///
/// where `sigma_m_i = (s1_i^2 + s2_i^2) / 2`.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the two Gaussians differ in dimensionality.
pub fn bhattacharyya_distance(a: &GaussianBox, b: &GaussianBox) -> Result<f32, BoxError> {
    if a.dim() != b.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim(),
            actual: b.dim(),
        });
    }

    const EPS: f32 = 1e-7;

    let mut term1 = 0.0f32;
    let mut term2 = 0.0f32;

    for i in 0..a.dim() {
        let s1 = a.sigma[i].max(EPS);
        let s2 = b.sigma[i].max(EPS);
        let s1_sq = s1 * s1;
        let s2_sq = s2 * s2;
        let sigma_m = (s1_sq + s2_sq) / 2.0;
        let dm = a.mu[i] - b.mu[i];

        term1 += dm * dm / sigma_m;
        term2 += sigma_m.ln() - 0.5 * (s1_sq.ln() + s2_sq.ln());
    }

    Ok(0.125 * term1 + 0.5 * term2)
}

/// Bhattacharyya coefficient (symmetric overlap measure).
///
/// `BC(a, b) = exp(-D_B(a, b))`, where `D_B` is the Bhattacharyya distance.
/// Values range from 0 (no overlap) to 1 (identical distributions).
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the two Gaussians differ in dimensionality.
pub fn bhattacharyya_coefficient(a: &GaussianBox, b: &GaussianBox) -> Result<f32, BoxError> {
    Ok((-bhattacharyya_distance(a, b)?).exp())
}

/// Volume regularization loss for a Gaussian box.
///
/// Variance floor loss (paper's L_reg, Eq. 13): prevents variance collapse.
///
/// Per-dimension squared hinge on variance falling below a threshold:
///
/// ```text
/// L_reg = (1/d) * sum_i max(0, min_var - sigma_i^2)^2
/// ```
///
/// This matches TaxoBell Eq. 13: `(1/d) * ||(delta_var * I - Sigma)_+||_F^2`
/// for diagonal Sigma.
///
/// # Arguments
///
/// * `g` - The Gaussian box to regularize
/// * `min_var` - Minimum variance threshold (delta_var in the paper; reference code uses 0.25)
#[must_use]
pub fn volume_regularization(g: &GaussianBox, min_var: f32) -> f32 {
    let d = g.sigma.len();
    if d == 0 {
        return 0.0;
    }
    let sum: f32 = g
        .sigma
        .iter()
        .map(|&s| {
            let deficit = (min_var - s * s).max(0.0);
            deficit * deficit
        })
        .sum();
    sum / d as f32
}

/// Sigma floor loss: prevents sigma from collapsing to near-zero.
///
/// Returns `sum_i max(0, min_sigma - sigma_i)` over all dimensions.
/// This is a simpler linear-hinge variant of [`volume_regularization`].
///
/// For the paper-faithful squared-hinge on variance (Eq. 13), use
/// [`volume_regularization`] instead. This function is retained for
/// backward compatibility and for cases where a linear penalty on
/// the standard deviation (not variance) is preferred.
#[must_use]
pub fn sigma_clipping_loss(g: &GaussianBox, min_sigma: f32) -> f32 {
    g.sigma.iter().map(|&s| (min_sigma - s).max(0.0)).sum()
}

/// Variance ceiling loss (paper's L_clip, Eq. 14): prevents variance explosion.
///
/// Linear hinge on variance exceeding a maximum threshold:
///
/// ```text
/// L_clip = (1/d) * sum_i max(0, sigma_i^2 - max_var)
/// ```
///
/// This matches TaxoBell Eq. 14: `(1/d) * tr([Sigma - M * I]_+)` for
/// diagonal Sigma.
///
/// Use together with [`volume_regularization`] (floor) to bound variance
/// in both directions.
#[must_use]
pub fn sigma_ceiling_loss(g: &GaussianBox, max_var: f32) -> f32 {
    let d = g.sigma.len();
    if d == 0 {
        return 0.0;
    }
    let sum: f32 = g.sigma.iter().map(|&s| (s * s - max_var).max(0.0)).sum();
    sum / d as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // --- Proptest strategies ---

    /// Generate a valid GaussianBox with the given dimension.
    fn arb_gaussian(dim: usize) -> impl Strategy<Value = GaussianBox> {
        let mus = prop::collection::vec(-100.0f32..100.0, dim);
        let sigmas = prop::collection::vec(0.01f32..100.0, dim);
        (mus, sigmas).prop_map(|(mu, sigma)| GaussianBox::new(mu, sigma).unwrap())
    }

    /// Generate a pair of valid GaussianBoxes with the same dimension.
    fn arb_gaussian_pair(dim: usize) -> impl Strategy<Value = (GaussianBox, GaussianBox)> {
        (arb_gaussian(dim), arb_gaussian(dim))
    }

    // --- Property tests ---

    proptest! {
        #[test]
        fn prop_kl_nonnegative(
            (a, b) in arb_gaussian_pair(8)
        ) {
            let kl = kl_divergence(&a, &b).unwrap();
            prop_assert!(kl >= -1e-5, "KL divergence should be non-negative, got {}", kl);
        }

        #[test]
        fn prop_kl_identical_is_zero(
            g in arb_gaussian(8)
        ) {
            let kl = kl_divergence(&g, &g).unwrap();
            prop_assert!(kl.abs() < 1e-4, "KL(g, g) should be 0, got {}", kl);
        }

        #[test]
        fn prop_kl_asymmetric(
            (a, b) in arb_gaussian_pair(4)
        ) {
            let kl_ab = kl_divergence(&a, &b).unwrap();
            let kl_ba = kl_divergence(&b, &a).unwrap();
            // Not always different (equal when a == b), but generally asymmetric.
            // We just check it doesn't crash and values are non-negative.
            prop_assert!(kl_ab >= -1e-5);
            prop_assert!(kl_ba >= -1e-5);
        }

        #[test]
        fn prop_bc_symmetric(
            (a, b) in arb_gaussian_pair(8)
        ) {
            let bc_ab = bhattacharyya_coefficient(&a, &b).unwrap();
            let bc_ba = bhattacharyya_coefficient(&b, &a).unwrap();
            prop_assert!(
                (bc_ab - bc_ba).abs() < 1e-5,
                "BC should be symmetric: {} != {}", bc_ab, bc_ba
            );
        }

        #[test]
        fn prop_bc_in_unit_interval(
            (a, b) in arb_gaussian_pair(8)
        ) {
            let bc = bhattacharyya_coefficient(&a, &b).unwrap();
            prop_assert!((-1e-6..=1.0 + 1e-6).contains(&bc),
                "BC should be in [0, 1], got {}", bc);
        }

        #[test]
        fn prop_bc_identical_is_one(
            g in arb_gaussian(8)
        ) {
            let bc = bhattacharyya_coefficient(&g, &g).unwrap();
            prop_assert!((bc - 1.0).abs() < 1e-4,
                "BC(g, g) should be 1.0, got {}", bc);
        }

        #[test]
        fn prop_volume_regularization_nonneg(
            g in arb_gaussian(8),
            target in -10.0f32..10.0,
        ) {
            let loss = volume_regularization(&g, target);
            prop_assert!(loss >= 0.0, "Volume regularization should be non-negative, got {}", loss);
        }

        #[test]
        fn prop_from_center_offset_positive_sigma(
            center in prop::collection::vec(-100.0f32..100.0, 8),
            offset in prop::collection::vec(-50.0f32..50.0, 8),
        ) {
            let g = GaussianBox::from_center_offset(center, offset).unwrap();
            for (i, &s) in g.sigma().iter().enumerate() {
                prop_assert!(s > 0.0, "sigma[{}] should be positive, got {}", i, s);
            }
        }

        #[test]
        fn prop_new_rejects_nonpositive_sigma(
            mu in prop::collection::vec(-10.0f32..10.0, 1..=8usize),
        ) {
            let dim = mu.len();
            // All zeros -> should fail
            let sigma = vec![0.0f32; dim];
            prop_assert!(GaussianBox::new(mu.clone(), sigma).is_err());
            // Negative -> should fail
            let sigma_neg = vec![-1.0f32; dim];
            prop_assert!(GaussianBox::new(mu, sigma_neg).is_err());
        }

        #[test]
        fn prop_sigma_clipping_nonneg(
            g in arb_gaussian(8),
            min_sigma in 0.0f32..10.0,
        ) {
            let loss = sigma_clipping_loss(&g, min_sigma);
            prop_assert!(loss >= 0.0, "Sigma clipping loss should be non-negative, got {}", loss);
        }

        // -- Bhattacharyya distance >= 0 --

        #[test]
        fn prop_bhattacharyya_distance_nonneg(
            (a, b) in arb_gaussian_pair(8)
        ) {
            let bd = bhattacharyya_distance(&a, &b).unwrap();
            prop_assert!(bd >= -1e-5, "BD should be non-negative, got {bd}");
        }

        // -- sigma_ceiling_loss >= 0 --

        #[test]
        fn prop_sigma_ceiling_nonneg(
            g in arb_gaussian(8),
            max_var in 0.01f32..100.0,
        ) {
            let loss = sigma_ceiling_loss(&g, max_var);
            prop_assert!(loss >= 0.0, "sigma_ceiling_loss should be non-negative, got {loss}");
        }

        /// sigma_ceiling_loss uses a linear hinge: scaling sigma excess by k
        /// should scale the loss linearly, not quadratically.
        #[test]
        fn sigma_ceiling_is_linear_hinge(
            base_sigma in 1.1f32..10.0,
            max_var in 0.01f32..1.0,
        ) {
            // Single dimension: sigma^2 > max_var guaranteed since base_sigma > 1.1 and max_var < 1.0.
            let g1 = GaussianBox::new(vec![0.0], vec![base_sigma]).unwrap();
            // Double the excess: sigma2^2 - max_var = 2 * (sigma1^2 - max_var)
            // sigma2 = sqrt(2 * sigma1^2 - max_var)
            let doubled_var = 2.0 * base_sigma * base_sigma - max_var;
            if doubled_var <= 0.0 {
                return Ok(());
            }
            let g2 = GaussianBox::new(vec![0.0], vec![doubled_var.sqrt()]).unwrap();

            let loss1 = sigma_ceiling_loss(&g1, max_var);
            let loss2 = sigma_ceiling_loss(&g2, max_var);

            // With linear hinge: loss2 / loss1 should be ~2.0 (not 4.0 as squared would give).
            let ratio = loss2 / loss1;
            prop_assert!(
                (ratio - 2.0).abs() < 0.01,
                "linear hinge ratio should be ~2.0, got {ratio} (loss1={loss1}, loss2={loss2})"
            );
        }

        // -- volume_regularization >= 0 (already tested, but explicit) --

        /// volume_regularization is a per-dim squared hinge on variance below min_var.
        #[test]
        fn prop_volume_regularization_is_per_dim_squared_hinge(
            g in arb_gaussian(8),
            min_var in 0.01f32..10.0,
        ) {
            let loss = volume_regularization(&g, min_var);
            prop_assert!(loss >= 0.0, "volume_regularization should be non-negative, got {loss}");
            // Verify formula: (1/d) * sum max(0, min_var - sigma_i^2)^2
            let d = g.sigma.len() as f32;
            let expected: f32 = g.sigma.iter()
                .map(|&s| { let deficit = (min_var - s * s).max(0.0); deficit * deficit })
                .sum::<f32>() / d;
            prop_assert!(
                (loss - expected).abs() < 1e-3,
                "volume_regularization mismatch: {loss} vs expected {expected}"
            );
        }
    }

    // --- Edge case tests ---

    #[test]
    fn test_high_dim_256() {
        let a = GaussianBox::unit(256);
        let b = GaussianBox::unit(256);
        let kl = kl_divergence(&a, &b).unwrap();
        assert!(
            kl.abs() < 1e-4,
            "KL of identical 256-d unit Gaussians: {kl}"
        );
        let bc = bhattacharyya_coefficient(&a, &b).unwrap();
        assert!((bc - 1.0).abs() < 1e-4, "BC of identical 256-d: {bc}");
    }

    #[test]
    fn test_high_dim_1024() {
        let a = GaussianBox::unit(1024);
        let b = GaussianBox::unit(1024);
        let kl = kl_divergence(&a, &b).unwrap();
        assert!(
            kl.abs() < 1e-3,
            "KL of identical 1024-d unit Gaussians: {kl}"
        );
    }

    #[test]
    fn test_single_dim() {
        let a = GaussianBox::new(vec![3.0], vec![0.5]).unwrap();
        let b = GaussianBox::new(vec![5.0], vec![1.0]).unwrap();
        let kl = kl_divergence(&a, &b).unwrap();
        assert!(kl > 0.0);
        let bc = bhattacharyya_coefficient(&a, &b).unwrap();
        assert!(bc > 0.0 && bc < 1.0);
    }

    #[test]
    fn test_very_small_sigma_stability() {
        let a = GaussianBox::new(vec![0.0], vec![1e-6]).unwrap();
        let b = GaussianBox::new(vec![0.0], vec![1.0]).unwrap();
        let kl = kl_divergence(&a, &b).unwrap();
        assert!(
            kl.is_finite(),
            "KL should be finite with small sigma, got {kl}"
        );
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_very_large_sigma_stability() {
        let a = GaussianBox::new(vec![0.0], vec![1e6]).unwrap();
        let b = GaussianBox::new(vec![0.0], vec![1.0]).unwrap();
        let kl = kl_divergence(&a, &b).unwrap();
        assert!(
            kl.is_finite(),
            "KL should be finite with large sigma, got {kl}"
        );
    }

    #[test]
    fn test_large_mu_difference_bc_near_zero() {
        let a = GaussianBox::new(vec![0.0; 8], vec![1.0; 8]).unwrap();
        let mu_far: Vec<f32> = vec![1000.0; 8];
        let b = GaussianBox::new(mu_far, vec![1.0; 8]).unwrap();
        let bc = bhattacharyya_coefficient(&a, &b).unwrap();
        assert!(
            bc < 1e-10,
            "BC for very distant Gaussians should be ~0, got {bc}"
        );
    }

    #[test]
    fn test_sigma_clipping_all_above_threshold() {
        let g = GaussianBox::new(vec![0.0, 0.0], vec![1.0, 2.0]).unwrap();
        let loss = sigma_clipping_loss(&g, 0.5);
        assert!(
            loss.abs() < 1e-10,
            "All sigmas above threshold: loss should be 0, got {loss}"
        );
    }

    // --- Audit-driven regression tests ---

    /// Hand-computed 2D KL divergence to verify the formula matches the standard.
    ///
    /// For child = N([1, 2], diag([0.5, 1.5]^2)) and parent = N([3, 0], diag([2, 1]^2)):
    ///
    /// Per-dimension formula: 0.5 * [(sc/sp)^2 + (mp-mc)^2/sp^2 - 1 + 2*ln(sp/sc)]
    ///
    /// dim 0: 0.5 * [(0.5/2)^2 + (3-1)^2/4 - 1 + 2*ln(2/0.5)]
    ///       = 0.5 * [0.0625 + 1.0 - 1.0 + 2*ln(4)]
    ///       = 0.5 * [0.0625 + 2*1.3863]
    ///       = 0.5 * [0.0625 + 2.7726]
    ///       = 0.5 * 2.8351 = 1.41755
    ///
    /// dim 1: 0.5 * [(1.5/1)^2 + (0-2)^2/1 - 1 + 2*ln(1/1.5)]
    ///       = 0.5 * [2.25 + 4.0 - 1.0 + 2*(-0.4055)]
    ///       = 0.5 * [2.25 + 4.0 - 1.0 - 0.8109]
    ///       = 0.5 * 4.4391 = 2.21955
    ///
    /// total = 1.41755 + 2.21955 = 3.6371
    #[test]
    fn test_kl_divergence_formula_matches_standard() {
        let child = GaussianBox::new(vec![1.0, 2.0], vec![0.5, 1.5]).unwrap();
        let parent = GaussianBox::new(vec![3.0, 0.0], vec![2.0, 1.0]).unwrap();
        let kl = kl_divergence(&child, &parent).unwrap();

        // Hand-computed expected value
        let dim0 = 0.5
            * ((0.5_f32 / 2.0).powi(2) + (3.0 - 1.0_f32).powi(2) / 4.0 - 1.0
                + 2.0 * (2.0_f32 / 0.5).ln());
        let dim1 = 0.5
            * ((1.5_f32 / 1.0).powi(2) + (0.0 - 2.0_f32).powi(2) / 1.0 - 1.0
                + 2.0 * (1.0_f32 / 1.5).ln());
        let expected = dim0 + dim1;

        assert!(
            (kl - expected).abs() < 1e-4,
            "KL formula mismatch: got {kl}, expected {expected}"
        );
    }

    /// BC(N, N) must equal 1.0 exactly (within float tolerance).
    #[test]
    fn test_bhattacharyya_coefficient_identical_distributions() {
        let g = GaussianBox::new(vec![1.5, -2.3, 0.7], vec![0.3, 2.1, 1.0]).unwrap();
        let bc = bhattacharyya_coefficient(&g, &g).unwrap();
        assert!((bc - 1.0).abs() < 1e-6, "BC(N,N) should be 1.0, got {bc}");
    }

    // BC must always be in [0, 1] -- proptest with wider ranges than the existing test.
    proptest! {
        #[test]
        fn test_bhattacharyya_coefficient_range_invariant(
            (a, b) in arb_gaussian_pair(16)
        ) {
            let bc = bhattacharyya_coefficient(&a, &b).unwrap();
            prop_assert!(bc >= -1e-7, "BC below 0: {bc}");
            prop_assert!(bc <= 1.0 + 1e-6, "BC above 1: {bc}");
        }
    }

    /// sigma_clipping_loss penalizes below floor; sigma_ceiling_loss penalizes above ceiling.
    #[test]
    fn test_sigma_floor_vs_ceiling() {
        // Sigma = [0.05, 0.3, 5.0]
        let g = GaussianBox::new(vec![0.0; 3], vec![0.05, 0.3, 5.0]).unwrap();

        // Floor = 0.1: only dim 0 violates (0.05 < 0.1), penalty = 0.05
        let floor_loss = sigma_clipping_loss(&g, 0.1);
        assert!(
            (floor_loss - 0.05).abs() < 1e-6,
            "floor loss: expected 0.05, got {floor_loss}"
        );

        // Ceiling max_var = 1.0: sigma^2 = [0.0025, 0.09, 25.0]
        // Only dim 2 violates: excess = 25.0 - 1.0 = 24.0 (linear hinge)
        // Result = 24.0 / 3 = 8.0
        let ceil_loss = sigma_ceiling_loss(&g, 1.0);
        let expected_ceil = 24.0 / 3.0;
        assert!(
            (ceil_loss - expected_ceil).abs() < 1e-3,
            "ceiling loss: expected {expected_ceil}, got {ceil_loss}"
        );
    }

    /// from_center_offset with extreme negative offsets produces sigma >= 1e-7.
    #[test]
    fn test_softplus_floor_not_too_small() {
        let g = GaussianBox::from_center_offset(vec![0.0; 4], vec![-100.0, -50.0, -25.0, -21.0])
            .unwrap();
        for (i, &s) in g.sigma().iter().enumerate() {
            assert!(
                s >= 1e-7,
                "sigma[{i}] = {s} is below 1e-7 floor for extreme negative offset"
            );
        }
    }

    /// KL with sigma ratios of 1e6 should not produce NaN or Inf.
    #[test]
    fn test_kl_extreme_sigma_ratio() {
        let a = GaussianBox::new(vec![0.0, 0.0], vec![1e-3, 1e-3]).unwrap();
        let b = GaussianBox::new(vec![0.0, 0.0], vec![1e3, 1e3]).unwrap();

        let kl_ab = kl_divergence(&a, &b).unwrap();
        let kl_ba = kl_divergence(&b, &a).unwrap();

        assert!(kl_ab.is_finite(), "KL(small||large) is not finite: {kl_ab}");
        assert!(kl_ba.is_finite(), "KL(large||small) is not finite: {kl_ba}");
        assert!(kl_ab >= 0.0);
        assert!(kl_ba >= 0.0);
    }

    /// Volume regularization (Eq. 13) is 0 when all variances >= min_var.
    #[test]
    fn test_volume_regularization_zero_above_threshold() {
        // sigma = [2.0, 3.0, 1.5] => variance = [4.0, 9.0, 2.25]
        let g = GaussianBox::new(vec![0.0; 3], vec![2.0, 3.0, 1.5]).unwrap();
        // min_var = 1.0: all variances exceed it, so loss = 0
        let loss = volume_regularization(&g, 1.0);
        assert!(
            loss.abs() < 1e-10,
            "loss should be 0 when all variances >= min_var, got {loss}"
        );

        // min_var = 5.0: dims 0 and 2 violate (4.0 < 5.0, 2.25 < 5.0)
        // deficit_0 = 5.0 - 4.0 = 1.0, sq = 1.0
        // deficit_1 = 0 (9.0 >= 5.0)
        // deficit_2 = 5.0 - 2.25 = 2.75, sq = 7.5625
        // mean = (1.0 + 0 + 7.5625) / 3 = 2.854167
        let loss_nonzero = volume_regularization(&g, 5.0);
        let expected = (1.0 + 0.0 + 2.75 * 2.75) / 3.0;
        assert!(
            (loss_nonzero - expected).abs() < 1e-4,
            "loss should be {expected}, got {loss_nonzero}"
        );
    }

    // --- Existing tests ---

    #[test]
    fn test_gaussian_new_valid() {
        let g = GaussianBox::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        assert_eq!(g.dim(), 2);
    }

    #[test]
    fn test_gaussian_new_dim_mismatch() {
        let err = GaussianBox::new(vec![0.0], vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(err, BoxError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_gaussian_new_negative_sigma() {
        let err = GaussianBox::new(vec![0.0], vec![-1.0]).unwrap_err();
        assert!(matches!(err, BoxError::InvalidBounds { .. }));
    }

    #[test]
    fn test_kl_identical() {
        let g = GaussianBox::unit(4);
        let kl = kl_divergence(&g, &g).unwrap();
        assert!(
            (kl).abs() < 1e-6,
            "KL of identical Gaussians should be 0, got {kl}"
        );
    }

    #[test]
    fn test_kl_asymmetric() {
        let child = GaussianBox::new(vec![0.0], vec![0.5]).unwrap();
        let parent = GaussianBox::new(vec![0.0], vec![2.0]).unwrap();

        let kl_cp = kl_divergence(&child, &parent).unwrap();
        let kl_pc = kl_divergence(&parent, &child).unwrap();

        // child -> parent should be small (child fits in parent)
        // parent -> child should be large (parent doesn't fit in child)
        assert!(kl_cp < kl_pc, "D_KL(narrow||wide) < D_KL(wide||narrow)");
    }

    #[test]
    fn test_kl_known_value() {
        // KL(N(0,1) || N(1,2)): analytical = 0.5 * (0.25 + 0.25 - 1 + 2*ln(2))
        //                                   = 0.5 * (-0.5 + 1.3863) = 0.4431
        let child = GaussianBox::new(vec![0.0], vec![1.0]).unwrap();
        let parent = GaussianBox::new(vec![1.0], vec![2.0]).unwrap();
        let kl = kl_divergence(&child, &parent).unwrap();
        let expected = 0.5 * (0.25 + 0.25 - 1.0 + 2.0 * 2.0_f32.ln());
        assert!(
            (kl - expected).abs() < 1e-5,
            "expected {expected}, got {kl}"
        );
    }

    #[test]
    fn test_bhattacharyya_identical() {
        let g = GaussianBox::new(vec![1.0, 2.0], vec![0.5, 1.5]).unwrap();
        let bc = bhattacharyya_coefficient(&g, &g).unwrap();
        assert!(
            (bc - 1.0).abs() < 1e-6,
            "BC of identical Gaussians should be 1.0, got {bc}"
        );
    }

    #[test]
    fn test_bhattacharyya_symmetric() {
        let a = GaussianBox::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let b = GaussianBox::new(vec![2.0, 1.0], vec![0.5, 2.0]).unwrap();

        let bc_ab = bhattacharyya_coefficient(&a, &b).unwrap();
        let bc_ba = bhattacharyya_coefficient(&b, &a).unwrap();
        assert!(
            (bc_ab - bc_ba).abs() < 1e-6,
            "BC should be symmetric: {bc_ab} != {bc_ba}"
        );
    }

    #[test]
    fn test_bhattacharyya_distant() {
        let a = GaussianBox::new(vec![0.0], vec![0.1]).unwrap();
        let b = GaussianBox::new(vec![100.0], vec![0.1]).unwrap();
        let bc = bhattacharyya_coefficient(&a, &b).unwrap();
        assert!(bc < 1e-10, "distant Gaussians should have BC ~0, got {bc}");
    }

    #[test]
    fn test_from_center_offset() {
        let g = GaussianBox::from_center_offset(vec![1.0, -1.0], vec![0.0, 0.5]).unwrap();
        assert_eq!(g.mu(), [1.0, -1.0]);
        // softplus(0) = ln(2)
        assert!((g.sigma()[0] - std::f32::consts::LN_2).abs() < 0.01);
        // softplus(0.5) ≈ 0.974
        let expected_sp_half = (0.5_f32.exp() + 1.0).ln();
        assert!((g.sigma()[1] - expected_sp_half).abs() < 0.01);
    }

    #[test]
    fn test_volume_regularization() {
        // unit Gaussian: sigma = [1.0; 4], variance = [1.0; 4]
        let g = GaussianBox::unit(4);
        // min_var = 0.5: all variances (1.0) exceed it, so loss = 0
        let loss = volume_regularization(&g, 0.5);
        assert!(
            loss.abs() < 1e-6,
            "unit Gaussian with min_var=0.5 should have loss=0, got {loss}"
        );
        // min_var = 2.0: all variances (1.0) are below, deficit=1.0 per dim
        // loss = (1/4) * 4 * 1.0^2 = 1.0
        let loss2 = volume_regularization(&g, 2.0);
        assert!((loss2 - 1.0).abs() < 1e-6, "expected 1.0, got {loss2}");
    }

    #[test]
    fn test_sigma_clipping() {
        let g = GaussianBox::new(vec![0.0, 0.0], vec![0.01, 1.0]).unwrap();
        let loss = sigma_clipping_loss(&g, 0.1);
        // sigma[0]=0.01 < 0.1, penalty = 0.09; sigma[1]=1.0 >= 0.1, penalty = 0
        assert!((loss - 0.09).abs() < 1e-6, "expected 0.09, got {loss}");
    }

    #[test]
    fn test_sigma_ceiling_loss_below_threshold() {
        let g = GaussianBox::new(vec![0.0, 0.0], vec![0.5, 0.8]).unwrap();
        // max_var = 1.0; sigma^2 = [0.25, 0.64], both below 1.0
        let loss = sigma_ceiling_loss(&g, 1.0);
        assert!(
            loss.abs() < 1e-10,
            "all below threshold: expected 0, got {loss}"
        );
    }

    #[test]
    fn test_sigma_ceiling_loss_above_threshold() {
        let g = GaussianBox::new(vec![0.0, 0.0], vec![2.0, 0.5]).unwrap();
        // max_var = 1.0; sigma^2 = [4.0, 0.25]
        // dim 0: excess = (4.0 - 1.0) = 3.0 (linear hinge)
        // dim 1: excess = 0
        // result = 3.0 / 2 = 1.5
        let loss = sigma_ceiling_loss(&g, 1.0);
        assert!((loss - 1.5).abs() < 1e-5, "expected 1.5, got {loss}");
    }

    // ---- NaN rejection ----

    #[test]
    fn nan_sigma_returns_err() {
        let result = GaussianBox::new(vec![0.0], vec![f32::NAN]);
        assert!(result.is_err(), "NaN sigma should be rejected");
    }

    #[test]
    fn nan_mu_returns_err() {
        let result = GaussianBox::new(vec![f32::NAN], vec![1.0]);
        assert!(result.is_err(), "NaN mu should be rejected");
    }
}
