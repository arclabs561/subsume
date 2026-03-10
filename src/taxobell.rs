//! TaxoBell combined training loss for taxonomy expansion.
//!
//! Implements the four-component loss from **TaxoBell** (WWW 2026,
//! [arXiv:2601.09633](https://arxiv.org/abs/2601.09633)):
//!
//! ```text
//! L = alpha * L_sym + beta * L_asym + gamma * L_reg + delta * L_clip
//! ```
//!
//! | Component | Purpose | Scoring function |
//! |-----------|---------|-----------------|
//! | `L_sym`   | Sibling triplet loss (symmetric) | Bhattacharyya coefficient |
//! | `L_asym`  | Parent-child containment (asymmetric) | KL divergence |
//! | `L_reg`   | Volume regularization (Eq. 13) | Per-dim squared hinge floor on variance |
//! | `L_clip`  | Variance ceiling (Eq. 14) | Per-dim linear hinge ceiling on variance |
//!
//! # Usage
//!
//! ```rust
//! use subsume::gaussian::GaussianBox;
//! use subsume::taxobell::{TaxoBellConfig, TaxoBellLoss};
//!
//! let config = TaxoBellConfig::default();
//! let loss_fn = TaxoBellLoss::new(config);
//!
//! let anchor = GaussianBox::new(vec![0.0; 8], vec![1.0; 8]).unwrap();
//! let positive = GaussianBox::new(vec![0.1; 8], vec![0.9; 8]).unwrap();
//! let negative = GaussianBox::new(vec![5.0; 8], vec![1.0; 8]).unwrap();
//!
//! let sym = loss_fn.symmetric_loss(&anchor, &positive, &negative).unwrap();
//! ```

use crate::gaussian::{
    bhattacharyya_coefficient, kl_divergence, sigma_ceiling_loss, volume_regularization,
    GaussianBox,
};
use crate::BoxError;
use serde::{Deserialize, Serialize};

/// Configuration for TaxoBell combined loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxoBellConfig {
    /// Weight for symmetric (triplet) loss component.
    pub alpha: f32,
    /// Weight for asymmetric (KL containment) loss component.
    pub beta: f32,
    /// Weight for volume regularization loss.
    pub gamma: f32,
    /// Weight for sigma clipping loss.
    pub delta: f32,

    /// Margin for the symmetric triplet loss (used in triplet variant).
    pub symmetric_margin: f32,
    /// Margin for the asymmetric alignment triplet loss (delta in paper Eq. 10).
    pub asymmetric_margin: f32,
    /// Scale factor `C` for the diverge component (paper Eq. 11).
    /// `L_diverge = max(0, C * D_rep - KL(parent || child))` where
    /// D_rep = logVol(parent) - logVol(child) is computed dynamically.
    /// Default 1.5 per paper. Set to 0.0 to disable L_diverge.
    pub asymmetric_diverge_c: f32,
    /// Lambda weight for L_diverge in the asymmetric loss (paper Eq. 12).
    /// `L_asym = L_align + lambda * L_diverge`. Default 0.3 per paper.
    pub diverge_lambda: f32,

    /// Minimum variance threshold for L_reg (paper Eq. 13, delta_var).
    /// Variances below this are penalized with a squared hinge.
    /// Reference code uses 0.25 (= 0.5^2, hinge on std).
    pub min_var: f32,
    /// Maximum variance threshold for L_clip (paper Eq. 14, M_var).
    /// Variances above this are penalized with a linear hinge.
    /// Reference code uses 10.0.
    pub max_var: f32,
}

impl Default for TaxoBellConfig {
    fn default() -> Self {
        Self {
            alpha: 0.45,
            beta: 0.45,
            gamma: 0.10,
            delta: 0.10,
            symmetric_margin: 0.1,
            asymmetric_margin: 0.5,
            asymmetric_diverge_c: 1.5,
            diverge_lambda: 0.3,
            min_var: 0.25,
            max_var: 10.0,
        }
    }
}

/// TaxoBell combined loss for taxonomy expansion training.
///
/// Wraps the four loss components (symmetric, asymmetric, regularization,
/// clipping) with configurable weights and margins.
#[derive(Debug, Clone)]
pub struct TaxoBellLoss {
    /// Loss configuration (weights, margins, thresholds).
    pub config: TaxoBellConfig,
}

impl TaxoBellLoss {
    /// Create a new TaxoBell loss with the given configuration.
    #[must_use]
    pub fn new(config: TaxoBellConfig) -> Self {
        Self { config }
    }

    /// Symmetric loss using Bhattacharyya coefficient (paper Eq. 9, BCE form).
    ///
    /// ```text
    /// L_sym = -log(BC(anchor, positive)) - log(1 - BC(anchor, negative))
    /// ```
    ///
    /// This is the binary cross-entropy formulation from the TaxoBell paper:
    /// the first term pulls siblings together (maximize BC with positive),
    /// and the second term pushes non-siblings apart (minimize BC with negative).
    ///
    /// BC values are clamped to `[eps, 1-eps]` for numerical stability.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any pair differs in dimensionality.
    pub fn symmetric_loss(
        &self,
        anchor: &GaussianBox,
        positive: &GaussianBox,
        negative: &GaussianBox,
    ) -> Result<f32, BoxError> {
        let bc_pos = bhattacharyya_coefficient(anchor, positive)?;
        let bc_neg = bhattacharyya_coefficient(anchor, negative)?;
        let eps = 1e-7_f32;
        let loss = -bc_pos.clamp(eps, 1.0 - eps).ln() - (1.0 - bc_neg.clamp(eps, 1.0 - eps)).ln();
        Ok(loss.max(0.0))
    }

    /// Alternative symmetric loss: triplet margin form.
    ///
    /// ```text
    /// L_sym_triplet = max(0, margin + BC(anchor, negative) - BC(anchor, positive))
    /// ```
    ///
    /// This is a simpler triplet-margin variant (not the paper's Eq. 9).
    /// Use [`symmetric_loss`](Self::symmetric_loss) for the paper-faithful BCE version.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any pair differs in dimensionality.
    pub fn symmetric_loss_triplet(
        &self,
        anchor: &GaussianBox,
        positive: &GaussianBox,
        negative: &GaussianBox,
    ) -> Result<f32, BoxError> {
        let bc_pos = bhattacharyya_coefficient(anchor, positive)?;
        let bc_neg = bhattacharyya_coefficient(anchor, negative)?;
        Ok((self.config.symmetric_margin + bc_neg - bc_pos).max(0.0))
    }

    /// Asymmetric containment loss using KL divergence (paper Eq. 10-12).
    ///
    /// ```text
    /// L_align   = max(0, KL(child || parent) - delta)
    /// D_rep     = logVol(parent) - logVol(child)          [dynamic]
    /// L_diverge = max(0, C * D_rep - KL(parent || child))
    /// L_asym    = L_align + lambda * L_diverge
    /// ```
    ///
    /// - **L_align**: hinge ensuring child is "contained" (small forward KL).
    /// - **L_diverge** (Eq. 11): ensures the parent is sufficiently "larger"
    ///   than the child. D_rep is the log-volume gap, computed dynamically.
    ///
    /// When called without a negative (this 2-argument form), L_align uses a
    /// simple hinge. For the triplet form, use [`asymmetric_loss_triplet`](Self::asymmetric_loss_triplet).
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if boxes differ in dimensionality.
    pub fn asymmetric_loss(
        &self,
        child: &GaussianBox,
        parent: &GaussianBox,
    ) -> Result<f32, BoxError> {
        let kl = kl_divergence(child, parent)?;
        let l_align = (kl - self.config.asymmetric_margin).max(0.0);

        let l_diverge = if self.config.asymmetric_diverge_c > 0.0 {
            let kl_reverse = kl_divergence(parent, child)?;
            let d_rep = parent.log_volume() - child.log_volume();
            (self.config.asymmetric_diverge_c * d_rep - kl_reverse).max(0.0)
        } else {
            0.0
        };

        Ok(l_align + self.config.diverge_lambda * l_diverge)
    }

    /// Asymmetric containment loss with negative sample (paper Eq. 10-12).
    ///
    /// ```text
    /// L_align   = max(0, KL(child || parent) - KL(child || negative) + delta)
    /// D_rep     = logVol(parent) - logVol(child)          [dynamic]
    /// L_diverge = max(0, C * D_rep - KL(parent || child))
    /// ```
    ///
    /// This is the full triplet form from the paper. Use this when negative
    /// samples are available for contrastive training.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if boxes differ in dimensionality.
    pub fn asymmetric_loss_triplet(
        &self,
        child: &GaussianBox,
        parent: &GaussianBox,
        negative: &GaussianBox,
    ) -> Result<f32, BoxError> {
        let kl_pos = kl_divergence(child, parent)?;
        let kl_neg = kl_divergence(child, negative)?;
        let l_align = (kl_pos - kl_neg + self.config.asymmetric_margin).max(0.0);

        let l_diverge = if self.config.asymmetric_diverge_c > 0.0 {
            let kl_reverse = kl_divergence(parent, child)?;
            let d_rep = parent.log_volume() - child.log_volume();
            (self.config.asymmetric_diverge_c * d_rep - kl_reverse).max(0.0)
        } else {
            0.0
        };

        Ok(l_align + self.config.diverge_lambda * l_diverge)
    }

    /// Combined loss over a batch of taxonomy relationships.
    ///
    /// Sums the four components:
    /// - `L_sym`: symmetric triplet loss over `(anchor, positive, negative)` triples
    /// - `L_asym`: asymmetric KL loss over `(child, parent)` pairs
    /// - `L_reg`: volume regularization over all unique boxes
    /// - `L_clip`: sigma clipping over all unique boxes
    ///
    /// # Arguments
    ///
    /// * `positives` - `(child, parent)` pairs from the taxonomy
    /// * `negatives` - `(anchor, positive, negative)` triples for sibling discrimination
    /// * `all_boxes` - all unique boxes in the batch (for regularization/clipping)
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any pair of boxes differs in dimensionality.
    pub fn combined_loss(
        &self,
        positives: &[(&GaussianBox, &GaussianBox)],
        negatives: &[(&GaussianBox, &GaussianBox, &GaussianBox)],
        all_boxes: &[&GaussianBox],
    ) -> Result<CombinedLossResult, BoxError> {
        self.combined_loss_with_negative(positives, negatives, all_boxes, None)
    }

    /// Combined loss with an optional negative for the asymmetric component.
    ///
    /// When `asym_negative` is `Some`, each `(child, parent)` pair in `positives`
    /// uses [`asymmetric_loss_triplet`](Self::asymmetric_loss_triplet) with the
    /// provided negative. When `None`, falls back to the 2-argument
    /// [`asymmetric_loss`](Self::asymmetric_loss).
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any pair of boxes differs in dimensionality.
    pub fn combined_loss_with_negative(
        &self,
        positives: &[(&GaussianBox, &GaussianBox)],
        negatives: &[(&GaussianBox, &GaussianBox, &GaussianBox)],
        all_boxes: &[&GaussianBox],
        asym_negative: Option<&GaussianBox>,
    ) -> Result<CombinedLossResult, BoxError> {
        // L_sym: symmetric triplet losses
        let mut l_sym = 0.0f32;
        for &(anchor, positive, negative) in negatives {
            l_sym += self.symmetric_loss(anchor, positive, negative)?;
        }
        if !negatives.is_empty() {
            l_sym /= negatives.len() as f32;
        }

        // L_asym: asymmetric KL containment losses
        let mut l_asym = 0.0f32;
        for &(child, parent) in positives {
            l_asym += match asym_negative {
                Some(neg) => self.asymmetric_loss_triplet(child, parent, neg)?,
                None => self.asymmetric_loss(child, parent)?,
            };
        }
        if !positives.is_empty() {
            l_asym /= positives.len() as f32;
        }

        // L_reg (Eq. 13) + L_clip (Eq. 14): regularization over all boxes
        let mut l_reg = 0.0f32;
        let mut l_clip = 0.0f32;
        for &g in all_boxes {
            l_reg += volume_regularization(g, self.config.min_var);
            l_clip += sigma_ceiling_loss(g, self.config.max_var);
        }
        if !all_boxes.is_empty() {
            let n = all_boxes.len() as f32;
            l_reg /= n;
            l_clip /= n;
        }

        let total = self.config.alpha * l_sym
            + self.config.beta * l_asym
            + self.config.gamma * l_reg
            + self.config.delta * l_clip;

        Ok(CombinedLossResult {
            total,
            l_sym,
            l_asym,
            l_reg,
            l_clip,
        })
    }
}

/// Breakdown of the combined TaxoBell loss into its four components.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CombinedLossResult {
    /// Weighted total: `alpha * l_sym + beta * l_asym + gamma * l_reg + delta * l_clip`.
    pub total: f32,
    /// Mean symmetric (triplet) loss before weighting.
    pub l_sym: f32,
    /// Mean asymmetric (KL containment) loss before weighting.
    pub l_asym: f32,
    /// Mean volume regularization loss before weighting.
    pub l_reg: f32,
    /// Mean sigma clipping loss before weighting.
    pub l_clip: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box(mu: &[f32], sigma: &[f32]) -> GaussianBox {
        GaussianBox::new(mu.to_vec(), sigma.to_vec()).unwrap()
    }

    #[test]
    fn test_symmetric_loss_small_when_positive_closer() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
        // anchor and positive are nearby; negative is far away
        let anchor = make_box(&[0.0, 0.0], &[1.0, 1.0]);
        let positive = make_box(&[0.1, 0.1], &[1.0, 1.0]);
        let negative = make_box(&[10.0, 10.0], &[1.0, 1.0]);

        let l = loss_fn
            .symmetric_loss(&anchor, &positive, &negative)
            .unwrap();
        // BCE: -log(BC_pos ~1) - log(1 - BC_neg ~0) should be small
        assert!(
            l < 1.0,
            "loss should be small when positive is much closer than negative, got {l}"
        );
    }

    #[test]
    fn test_symmetric_loss_large_when_negative_closer() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
        // anchor and negative are nearby; positive is far away
        let anchor = make_box(&[0.0, 0.0], &[1.0, 1.0]);
        let positive = make_box(&[10.0, 10.0], &[1.0, 1.0]);
        let negative = make_box(&[0.1, 0.1], &[1.0, 1.0]);

        let l = loss_fn
            .symmetric_loss(&anchor, &positive, &negative)
            .unwrap();
        assert!(
            l > 1.0,
            "loss should be large when negative is closer than positive, got {l}"
        );
    }

    #[test]
    fn test_symmetric_loss_triplet_variant() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
        let anchor = make_box(&[0.0, 0.0], &[1.0, 1.0]);
        let positive = make_box(&[0.1, 0.1], &[1.0, 1.0]);
        let negative = make_box(&[10.0, 10.0], &[1.0, 1.0]);

        let l = loss_fn
            .symmetric_loss_triplet(&anchor, &positive, &negative)
            .unwrap();
        assert!(
            l < 1e-6,
            "triplet loss should be ~0 when positive is much closer, got {l}"
        );
    }

    #[test]
    fn test_asymmetric_loss_small_for_contained_child() {
        let config = TaxoBellConfig {
            asymmetric_margin: 0.0,
            ..TaxoBellConfig::default()
        };
        let loss_fn = TaxoBellLoss::new(config);
        // narrow child inside wide parent
        let child = make_box(&[0.0, 0.0], &[0.5, 0.5]);
        let parent = make_box(&[0.0, 0.0], &[2.0, 2.0]);

        let l_good = loss_fn.asymmetric_loss(&child, &parent).unwrap();

        // Reverse direction: wide "child" in narrow "parent" should have much larger KL
        let l_bad = loss_fn.asymmetric_loss(&parent, &child).unwrap();
        assert!(
            l_good < l_bad,
            "KL(narrow||wide) should be less than KL(wide||narrow): {l_good} vs {l_bad}"
        );
    }

    #[test]
    fn test_asymmetric_loss_large_for_non_contained() {
        let config = TaxoBellConfig {
            asymmetric_margin: 0.0,
            ..TaxoBellConfig::default()
        };
        let loss_fn = TaxoBellLoss::new(config);
        // wide "child" vs narrow "parent" -- bad containment
        let child = make_box(&[0.0, 0.0], &[5.0, 5.0]);
        let parent = make_box(&[0.0, 0.0], &[0.1, 0.1]);

        let l = loss_fn.asymmetric_loss(&child, &parent).unwrap();
        assert!(
            l > 5.0,
            "containment loss should be large when child doesn't fit, got {l}"
        );
    }

    #[test]
    fn test_asymmetric_loss_with_diverge() {
        let config = TaxoBellConfig {
            asymmetric_margin: 0.0,
            asymmetric_diverge_c: 1.0,
            ..TaxoBellConfig::default()
        };
        let loss_fn = TaxoBellLoss::new(config);
        // narrow child inside wide parent: KL(parent||child) should be large
        let child = make_box(&[0.0, 0.0], &[0.5, 0.5]);
        let parent = make_box(&[0.0, 0.0], &[5.0, 5.0]);

        let l = loss_fn.asymmetric_loss(&child, &parent).unwrap();
        // KL(parent||child) is large (wide->narrow), so L_diverge should be small/zero
        assert!(l.is_finite(), "loss should be finite, got {l}");
    }

    #[test]
    fn test_asymmetric_loss_triplet() {
        let config = TaxoBellConfig {
            asymmetric_margin: 1.0,
            ..TaxoBellConfig::default()
        };
        let loss_fn = TaxoBellLoss::new(config);
        let child = make_box(&[0.0, 0.0], &[0.5, 0.5]);
        let parent = make_box(&[0.0, 0.0], &[2.0, 2.0]);
        let negative = make_box(&[10.0, 10.0], &[0.5, 0.5]);

        let l = loss_fn
            .asymmetric_loss_triplet(&child, &parent, &negative)
            .unwrap();
        // KL(child||parent) should be much less than KL(child||negative)
        // so triplet loss should be ~0
        assert!(
            l < 1e-2,
            "triplet loss should be small when parent is closer than negative, got {l}"
        );
    }

    #[test]
    fn test_combined_loss_all_components() {
        let config = TaxoBellConfig {
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.1,
            delta: 0.1,
            symmetric_margin: 0.1,
            asymmetric_margin: 0.0,
            asymmetric_diverge_c: 0.0,
            diverge_lambda: 0.3,
            min_var: 0.01,
            max_var: 100.0,
        };
        let loss_fn = TaxoBellLoss::new(config);

        let a = make_box(&[0.0, 0.0], &[1.0, 1.0]);
        let b = make_box(&[0.1, 0.1], &[0.8, 0.8]);
        let c = make_box(&[5.0, 5.0], &[1.0, 1.0]);

        let positives = vec![(&b, &a)]; // b is child of a
        let negatives = vec![(&a, &b, &c)]; // a-b similar, c dissimilar
        let all_boxes = vec![&a, &b, &c];

        let result = loss_fn
            .combined_loss(&positives, &negatives, &all_boxes)
            .unwrap();

        assert!(result.total >= 0.0, "total loss must be non-negative");
        assert!(result.l_reg >= 0.0, "l_reg must be non-negative");
        assert!(result.l_clip >= 0.0, "l_clip must be non-negative");
        // total = alpha*l_sym + beta*l_asym + gamma*l_reg + delta*l_clip
        let expected =
            1.0 * result.l_sym + 1.0 * result.l_asym + 0.1 * result.l_reg + 0.1 * result.l_clip;
        assert!(
            (result.total - expected).abs() < 1e-6,
            "total={} != expected={expected}",
            result.total
        );
    }

    #[test]
    fn test_combined_loss_empty_batch() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
        let result = loss_fn.combined_loss(&[], &[], &[]).unwrap();
        assert!(
            (result.total).abs() < 1e-6,
            "empty batch should yield zero loss"
        );
    }

    #[test]
    fn test_dimension_mismatch_propagates() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
        let a = make_box(&[0.0, 0.0], &[1.0, 1.0]);
        let b = make_box(&[0.0], &[1.0]);

        assert!(loss_fn.symmetric_loss(&a, &a, &b).is_err());
        assert!(loss_fn.asymmetric_loss(&a, &b).is_err());
    }

    #[test]
    fn test_default_config() {
        let cfg = TaxoBellConfig::default();
        assert!((cfg.alpha - 0.45).abs() < f32::EPSILON);
        assert!((cfg.beta - 0.45).abs() < f32::EPSILON);
        assert!((cfg.gamma - 0.10).abs() < f32::EPSILON);
        assert!((cfg.delta - 0.10).abs() < f32::EPSILON);
        assert!(cfg.symmetric_margin > 0.0);
        assert!(cfg.min_var > 0.0);
        assert!(cfg.max_var > cfg.min_var);
    }

    // -- Audit-driven regression tests --

    /// BCE loss is 0 when BC_pos=1 and BC_neg=0.
    /// symmetric_loss = -log(BC_pos) - log(1 - BC_neg)
    /// When BC_pos=1 and BC_neg=0: -log(1) - log(1) = 0.
    #[test]
    fn test_symmetric_loss_bce_properties() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());

        // BC_pos ~ 1: anchor == positive (identical distributions)
        // BC_neg ~ 0: negative very far away
        let anchor = make_box(&[0.0, 0.0, 0.0, 0.0], &[1.0, 1.0, 1.0, 1.0]);
        let negative = make_box(&[1000.0; 4], &[0.01; 4]);

        let l = loss_fn.symmetric_loss(&anchor, &anchor, &negative).unwrap();
        // -log(1-eps) - log(1 - ~0) ~ 0
        assert!(
            l < 1e-4,
            "BCE loss should be ~0 when BC_pos=1 and BC_neg=0, got {l}"
        );
    }

    /// Compare BCE and triplet symmetric loss on the same inputs.
    /// They should agree on direction (when one is high, the other should be too).
    #[test]
    fn test_symmetric_loss_bce_vs_triplet() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig {
            symmetric_margin: 0.1,
            ..TaxoBellConfig::default()
        });

        let anchor = make_box(&[0.0; 4], &[1.0; 4]);
        let positive = make_box(&[0.1; 4], &[1.0; 4]);
        let negative = make_box(&[10.0; 4], &[1.0; 4]);

        let bce = loss_fn
            .symmetric_loss(&anchor, &positive, &negative)
            .unwrap();
        let triplet = loss_fn
            .symmetric_loss_triplet(&anchor, &positive, &negative)
            .unwrap();

        // Both should be small when positive is much closer than negative
        assert!(bce < 1.0, "BCE loss should be small, got {bce}");
        assert!(triplet < 0.2, "Triplet loss should be small, got {triplet}");

        // Now swap: negative closer than positive
        let bce_bad = loss_fn
            .symmetric_loss(&anchor, &negative, &positive)
            .unwrap();
        let triplet_bad = loss_fn
            .symmetric_loss_triplet(&anchor, &negative, &positive)
            .unwrap();

        // Both should be larger in the bad case
        assert!(
            bce_bad > bce,
            "BCE loss should increase when positive/negative swapped"
        );
        assert!(
            triplet_bad > triplet,
            "Triplet loss should increase when positive/negative swapped"
        );
    }

    /// Test L_diverge component: when c > 0, L_diverge = max(0, c*D_rep - KL(parent||child))
    /// where D_rep = logVol(parent) - logVol(child) (dynamic).
    /// When parent is moderately wider than child, D_rep > 0 but KL(parent||child) is small,
    /// so L_diverge should be positive.
    #[test]
    fn test_asymmetric_loss_with_diverge_active() {
        let config = TaxoBellConfig {
            asymmetric_margin: 100.0, // large margin so l_align = 0
            asymmetric_diverge_c: 2.0,
            ..TaxoBellConfig::default()
        };
        let loss_fn = TaxoBellLoss::new(config);

        // Parent moderately wider than child => D_rep > 0, KL(parent||child) moderate
        // With sigma_c=1.0, sigma_p=2.0: D_rep=4*ln(2)=2.77, KL(p||c)=3.23, c*D_rep=5.55>3.23
        let child = make_box(&[0.0; 4], &[1.0; 4]);
        let parent = make_box(&[0.0; 4], &[2.0; 4]);

        let l = loss_fn.asymmetric_loss(&child, &parent).unwrap();
        // l_align should be 0 (large margin).
        // D_rep = logVol(parent) - logVol(child) > 0, c*D_rep should exceed KL
        assert!(
            l > 0.0,
            "L_diverge should be active when parent is moderately larger, got {l}"
        );

        // Now with parent much wider: KL(parent||child) grows faster than D_rep
        let parent_wide = make_box(&[0.0; 4], &[100.0; 4]);
        let l2 = loss_fn.asymmetric_loss(&child, &parent_wide).unwrap();
        assert!(
            l2 < l,
            "L_diverge should decrease when parent is much wider: {l2} vs {l}"
        );
    }

    /// Compare asymmetric_loss (2-arg) vs asymmetric_loss_triplet (3-arg).
    #[test]
    fn test_asymmetric_loss_triplet_vs_simple() {
        let config = TaxoBellConfig {
            asymmetric_margin: 1.0,
            asymmetric_diverge_c: 0.0,
            ..TaxoBellConfig::default()
        };
        let loss_fn = TaxoBellLoss::new(config);

        let child = make_box(&[0.0; 4], &[0.5; 4]);
        let parent = make_box(&[0.0; 4], &[2.0; 4]);
        let negative = make_box(&[50.0; 4], &[0.5; 4]);

        let l_simple = loss_fn.asymmetric_loss(&child, &parent).unwrap();
        let l_triplet = loss_fn
            .asymmetric_loss_triplet(&child, &parent, &negative)
            .unwrap();

        // Simple: max(0, KL(child||parent) - margin)
        // Triplet: max(0, KL(child||parent) - KL(child||negative) + margin)
        // KL(child||negative) should be large, so triplet should be smaller
        assert!(
            l_triplet <= l_simple + 1e-6,
            "triplet with distant negative should be <= simple: triplet={l_triplet}, simple={l_simple}"
        );
    }

    /// Test with the paper's recommended weight split: 0.45/0.45/0.10.
    #[test]
    fn test_combined_loss_paper_weights() {
        let config = TaxoBellConfig {
            alpha: 0.45,
            beta: 0.45,
            gamma: 0.05,
            delta: 0.05,
            symmetric_margin: 0.1,
            asymmetric_margin: 1.0,
            asymmetric_diverge_c: 0.0,
            diverge_lambda: 0.3,
            min_var: 0.01,
            max_var: 100.0,
        };
        let loss_fn = TaxoBellLoss::new(config);

        let a = make_box(&[0.0; 8], &[2.0; 8]);
        let b = make_box(&[0.1; 8], &[1.0; 8]);
        let c = make_box(&[5.0; 8], &[1.0; 8]);

        let positives = vec![(&b, &a)];
        let negatives = vec![(&a, &b, &c)];
        let all = vec![&a, &b, &c];

        let r = loss_fn.combined_loss(&positives, &negatives, &all).unwrap();

        // Verify weighted sum
        let expected = 0.45 * r.l_sym + 0.45 * r.l_asym + 0.05 * r.l_reg + 0.05 * r.l_clip;
        assert!(
            (r.total - expected).abs() < 1e-5,
            "total={} != weighted sum={expected}",
            r.total
        );
        assert!(r.total.is_finite());
    }

    // -- Integration test with taxonomy module --

    #[test]
    fn test_loss_decreases_when_child_moved_inside_parent() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());

        // Parent: wide box centered at origin.
        let parent = make_box(&[0.0, 0.0, 0.0, 0.0], &[3.0, 3.0, 3.0, 3.0]);

        // Child far outside parent.
        let child_outside = make_box(&[10.0, 10.0, 10.0, 10.0], &[0.5, 0.5, 0.5, 0.5]);
        // Child inside parent (same center, narrower).
        let child_inside = make_box(&[0.0, 0.0, 0.0, 0.0], &[0.5, 0.5, 0.5, 0.5]);

        let l_outside = loss_fn.asymmetric_loss(&child_outside, &parent).unwrap();
        let l_inside = loss_fn.asymmetric_loss(&child_inside, &parent).unwrap();

        assert!(
            l_inside < l_outside,
            "moving child inside parent should decrease loss: {l_inside} vs {l_outside}"
        );
    }

    #[test]
    fn test_loss_increases_when_child_moved_outside_parent() {
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());

        let parent = make_box(&[0.0, 0.0, 0.0], &[2.0, 2.0, 2.0]);
        let child_near = make_box(&[0.1, 0.1, 0.1], &[0.5, 0.5, 0.5]);
        let child_far = make_box(&[20.0, 20.0, 20.0], &[0.5, 0.5, 0.5]);

        let l_near = loss_fn.asymmetric_loss(&child_near, &parent).unwrap();
        let l_far = loss_fn.asymmetric_loss(&child_far, &parent).unwrap();

        assert!(
            l_far > l_near,
            "moving child outside parent should increase loss: {l_far} vs {l_near}"
        );
    }

    #[test]
    fn test_combined_loss_mini_taxonomy() {
        // Inline 5-node taxonomy: root -> {A, B}, A -> {C, D}
        let loss_fn = TaxoBellLoss::new(TaxoBellConfig {
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.01,
            delta: 0.01,
            symmetric_margin: 0.1,
            asymmetric_margin: 0.0,
            asymmetric_diverge_c: 0.0,
            diverge_lambda: 0.3,
            min_var: 0.01,
            max_var: 100.0,
        });

        let root = make_box(&[0.0; 8], &[4.0; 8]);
        let a = make_box(&[-1.0; 8], &[2.0; 8]);
        let b = make_box(&[1.0; 8], &[2.0; 8]);
        let c = make_box(&[-1.5; 8], &[0.5; 8]);
        let d = make_box(&[-0.5; 8], &[0.5; 8]);

        // Parent-child pairs: (child, parent)
        let positives: Vec<(&GaussianBox, &GaussianBox)> =
            vec![(&a, &root), (&b, &root), (&c, &a), (&d, &a)];

        // Sibling triples: siblings share parent, non-sibling is from different subtree
        // (anchor=A, positive=B (siblings under root), negative=C (child of A, not sibling))
        let negatives: Vec<(&GaussianBox, &GaussianBox, &GaussianBox)> =
            vec![(&a, &b, &c), (&c, &d, &b)];

        let all: Vec<&GaussianBox> = vec![&root, &a, &b, &c, &d];
        let result = loss_fn.combined_loss(&positives, &negatives, &all).unwrap();

        assert!(result.total.is_finite(), "total loss must be finite");
        assert!(result.l_sym >= 0.0, "l_sym must be non-negative");
        assert!(result.l_asym >= 0.0, "l_asym must be non-negative");
        assert!(result.l_reg >= 0.0, "l_reg must be non-negative");
        assert!(result.l_clip >= 0.0, "l_clip must be non-negative");

        // Verify weighted sum.
        let expected =
            1.0 * result.l_sym + 1.0 * result.l_asym + 0.01 * result.l_reg + 0.01 * result.l_clip;
        assert!(
            (result.total - expected).abs() < 1e-5,
            "total={} != expected={expected}",
            result.total
        );
    }

    // -- Property tests --

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Strategy for a valid GaussianBox of a given dimension.
        fn arb_gaussian_box(dim: usize) -> impl Strategy<Value = GaussianBox> {
            let mu_strat = prop::collection::vec(-10.0f32..10.0f32, dim);
            let sigma_strat = prop::collection::vec(0.01f32..10.0f32, dim);
            (mu_strat, sigma_strat).prop_map(|(mu, sigma)| GaussianBox::new(mu, sigma).unwrap())
        }

        proptest! {
            #[test]
            fn symmetric_loss_is_non_negative(
                anchor in arb_gaussian_box(4),
                positive in arb_gaussian_box(4),
                negative in arb_gaussian_box(4),
            ) {
                let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
                let l = loss_fn.symmetric_loss(&anchor, &positive, &negative).unwrap();
                prop_assert!(l >= 0.0, "symmetric_loss must be >= 0, got {l}");
                prop_assert!(l.is_finite(), "symmetric_loss must be finite, got {l}");
            }

            #[test]
            fn symmetric_loss_zero_when_well_separated(
                anchor in arb_gaussian_box(4),
            ) {
                // Construct a negative that is very far from anchor.
                let far_mu: Vec<f32> = anchor.mu().iter().map(|&m| m + 100.0).collect();
                let negative = GaussianBox::new(far_mu, vec![0.1; 4]).unwrap();

                let config = TaxoBellConfig {
                    symmetric_margin: 0.1,
                    ..TaxoBellConfig::default()
                };
                let loss_fn = TaxoBellLoss::new(config);

                // BC(anchor, negative) should be ~0.
                // When positive=anchor, BC_pos=1 >> margin, so loss=0.
                let l = loss_fn.symmetric_loss(&anchor, &anchor, &negative).unwrap();
                prop_assert!(
                    l < 1e-4,
                    "loss should be ~0 when positive=anchor and negative is far: {l}"
                );
            }

            #[test]
            fn asymmetric_loss_is_non_negative(
                child in arb_gaussian_box(4),
                parent in arb_gaussian_box(4),
            ) {
                let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
                let l = loss_fn.asymmetric_loss(&child, &parent).unwrap();
                prop_assert!(l >= 0.0, "asymmetric_loss must be >= 0, got {l}");
                prop_assert!(l.is_finite(), "asymmetric_loss must be finite, got {l}");
            }

            #[test]
            fn combined_loss_components_sum_correctly(
                a in arb_gaussian_box(4),
                b in arb_gaussian_box(4),
                c in arb_gaussian_box(4),
            ) {
                let config = TaxoBellConfig {
                    alpha: 2.0,
                    beta: 3.0,
                    gamma: 0.5,
                    delta: 0.25,
                    ..TaxoBellConfig::default()
                };
                let loss_fn = TaxoBellLoss::new(config);

                let positives = vec![(&b, &a)];
                let negatives = vec![(&a, &b, &c)];
                let all = vec![&a, &b, &c];

                let r = loss_fn.combined_loss(&positives, &negatives, &all).unwrap();
                let expected = 2.0 * r.l_sym + 3.0 * r.l_asym + 0.5 * r.l_reg + 0.25 * r.l_clip;
                prop_assert!(
                    (r.total - expected).abs() < 1e-4,
                    "total={} != weighted sum={expected}",
                    r.total
                );
            }

            #[test]
            fn all_loss_components_finite(
                a in arb_gaussian_box(8),
                b in arb_gaussian_box(8),
                c in arb_gaussian_box(8),
            ) {
                let loss_fn = TaxoBellLoss::new(TaxoBellConfig::default());
                let positives = vec![(&b, &a)];
                let negatives = vec![(&a, &b, &c)];
                let all = vec![&a, &b, &c];

                let r = loss_fn.combined_loss(&positives, &negatives, &all).unwrap();
                prop_assert!(r.total.is_finite(), "total not finite: {}", r.total);
                prop_assert!(r.l_sym.is_finite(), "l_sym not finite: {}", r.l_sym);
                prop_assert!(r.l_asym.is_finite(), "l_asym not finite: {}", r.l_asym);
                prop_assert!(r.l_reg.is_finite(), "l_reg not finite: {}", r.l_reg);
                prop_assert!(r.l_clip.is_finite(), "l_clip not finite: {}", r.l_clip);
            }

            /// diverge_lambda scales L_diverge linearly in the asymmetric loss.
            #[test]
            fn diverge_lambda_scales_loss(
                child in arb_gaussian_box(4),
                parent in arb_gaussian_box(4),
            ) {
                let config_zero = TaxoBellConfig {
                    asymmetric_margin: 10.0,
                    asymmetric_diverge_c: 1.0,
                    diverge_lambda: 0.0,
                    ..TaxoBellConfig::default()
                };
                let config_one = TaxoBellConfig {
                    diverge_lambda: 1.0,
                    ..config_zero.clone()
                };
                let config_half = TaxoBellConfig {
                    diverge_lambda: 0.5,
                    ..config_zero.clone()
                };

                let loss_zero = TaxoBellLoss::new(config_zero).asymmetric_loss(&child, &parent).unwrap();
                let loss_one = TaxoBellLoss::new(config_one).asymmetric_loss(&child, &parent).unwrap();
                let loss_half = TaxoBellLoss::new(config_half).asymmetric_loss(&child, &parent).unwrap();

                // L_align is the same regardless of lambda. The difference
                // between lambda=1 and lambda=0 is exactly the diverge term.
                let diverge_contribution = loss_one - loss_zero;
                // lambda=0.5 should contribute half the diverge term.
                let expected_half = loss_zero + 0.5 * diverge_contribution;
                // Use relative tolerance since loss magnitudes vary widely.
                let tol = 1e-4 * (1.0 + expected_half.abs());
                prop_assert!(
                    (loss_half - expected_half).abs() < tol,
                    "lambda=0.5 should give half diverge: got {loss_half}, expected {expected_half} \
                     (zero={loss_zero}, one={loss_one}, tol={tol})"
                );
                // diverge contribution should be non-negative.
                prop_assert!(
                    diverge_contribution >= -1e-5,
                    "diverge contribution should be non-negative: {diverge_contribution}"
                );
            }

            /// When a negative is provided, combined_loss_with_negative uses triplet scoring
            /// for the asymmetric component.
            #[test]
            fn combined_loss_with_negative_uses_triplet(
                a in arb_gaussian_box(4),
                b in arb_gaussian_box(4),
                c in arb_gaussian_box(4),
            ) {
                let config = TaxoBellConfig {
                    asymmetric_diverge_c: 0.0,
                    diverge_lambda: 0.3,
                    ..TaxoBellConfig::default()
                };
                let loss_fn = TaxoBellLoss::new(config);

                let positives = vec![(&b, &a)];
                let negatives_sym = vec![(&a, &b, &c)];
                let all = vec![&a, &b, &c];

                let r_no_neg = loss_fn.combined_loss(&positives, &negatives_sym, &all).unwrap();
                let r_with_neg = loss_fn.combined_loss_with_negative(
                    &positives, &negatives_sym, &all, Some(&c)
                ).unwrap();

                // Symmetric, reg, clip components should be identical.
                prop_assert!(
                    (r_no_neg.l_sym - r_with_neg.l_sym).abs() < 1e-5,
                    "l_sym should be the same: {} vs {}", r_no_neg.l_sym, r_with_neg.l_sym
                );
                prop_assert!(
                    (r_no_neg.l_reg - r_with_neg.l_reg).abs() < 1e-5,
                    "l_reg should be the same: {} vs {}", r_no_neg.l_reg, r_with_neg.l_reg
                );
                prop_assert!(
                    (r_no_neg.l_clip - r_with_neg.l_clip).abs() < 1e-5,
                    "l_clip should be the same: {} vs {}", r_no_neg.l_reg, r_with_neg.l_clip
                );

                // The asymmetric components may differ (2-arg vs triplet).
                // Both must be finite and non-negative.
                prop_assert!(r_with_neg.l_asym >= 0.0, "l_asym with neg must be >= 0");
                prop_assert!(r_with_neg.l_asym.is_finite(), "l_asym with neg must be finite");
                prop_assert!(r_with_neg.total.is_finite(), "total with neg must be finite");
            }
        }
    }
}
