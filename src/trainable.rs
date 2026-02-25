//! Trainable geometric representations with learnable parameters.
//!
//! Contains both [`TrainableBox`] (axis-aligned hyperrectangles) and
//! [`TrainableCone`] (angular cones on the sphere).

use crate::optimizer::AMSGradState;
use serde::{Deserialize, Serialize};

/// A simple hard box used by the trainer implementation.
///
/// This is intentionally local to `subsume` (no tensor deps).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DenseBox {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

impl DenseBox {
    pub fn new(min: Vec<f32>, max: Vec<f32>) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn volume(&self) -> f32 {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(&a, &b)| (b - a).max(0.0))
            .product::<f32>()
    }

    #[inline]
    pub fn intersection_volume(&self, other: &Self) -> f32 {
        let mut v = 1.0f32;
        for i in 0..self.min.len().min(other.min.len()) {
            let lo = self.min[i].max(other.min[i]);
            let hi = self.max[i].min(other.max[i]);
            let side = (hi - lo).max(0.0);
            v *= side;
            if v == 0.0 {
                break;
            }
        }
        v
    }

    /// Containment-style probability: \(P(\text{other} \subseteq \text{self})\).
    ///
    /// For hard boxes: Vol(self ∩ other) / Vol(other), with 0 when other has 0 volume.
    #[inline]
    pub fn conditional_probability(&self, other: &Self) -> f32 {
        let denom = other.volume();
        if denom <= 0.0 {
            return 0.0;
        }
        (self.intersection_volume(other) / denom).clamp(0.0, 1.0)
    }
}

/// A trainable box embedding with learnable parameters.
///
/// Uses reparameterization to ensure min <= max:
/// - min = mu - exp(delta)/2
/// - max = mu + exp(delta)/2
///
/// This ensures boxes are always valid (min <= max).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableBox {
    /// Mean position in each dimension (d-dimensional vector).
    pub mu: Vec<f32>,
    /// Log-width in each dimension (width = exp(delta)).
    pub delta: Vec<f32>,
    /// Dimension
    pub dim: usize,
}

impl TrainableBox {
    /// Create a new trainable box.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean position (center of box)
    /// * `delta` - Log-width (width = exp(delta))
    ///
    /// The box will have:
    /// - min = mu - exp(delta) / 2
    /// - max = mu + exp(delta) / 2
    #[must_use]
    pub fn new(mu: Vec<f32>, delta: Vec<f32>) -> Self {
        assert_eq!(
            mu.len(),
            delta.len(),
            "mu and delta must have same dimension"
        );
        let dim = mu.len();
        Self { mu, delta, dim }
    }

    /// Initialize from a vector embedding.
    ///
    /// Creates a small box around the vector with initial width `init_width`.
    #[must_use]
    pub fn from_vector(vector: &[f32], init_width: f32) -> Self {
        let mu = vector.to_vec();
        let delta: Vec<f32> = vec![init_width.ln(); mu.len()];
        Self::new(mu, delta)
    }

    /// Convert to a BoxEmbedding (for inference).
    #[must_use]
    pub(crate) fn to_box(&self) -> DenseBox {
        let min: Vec<f32> = self
            .mu
            .iter()
            .zip(self.delta.iter())
            .map(|(&m, &d)| m - (d.exp() / 2.0))
            .collect();
        let max: Vec<f32> = self
            .mu
            .iter()
            .zip(self.delta.iter())
            .map(|(&m, &d)| m + (d.exp() / 2.0))
            .collect();
        DenseBox::new(min, max)
    }

    /// Update box parameters using AMSGrad optimizer.
    pub fn update_amsgrad(
        &mut self,
        grad_mu: &[f32],
        grad_delta: &[f32],
        state: &mut AMSGradState,
    ) {
        state.t += 1;
        let t = state.t as f32;

        // Update first moment (m)
        for (i, &grad) in grad_mu.iter().enumerate().take(self.dim) {
            state.m[i] = state.beta1 * state.m[i] + (1.0 - state.beta1) * grad;
        }

        // Update second moment (v) and max (v_hat)
        for (i, &grad) in grad_mu.iter().enumerate().take(self.dim) {
            let v_new = state.beta2 * state.v[i] + (1.0 - state.beta2) * grad * grad;
            state.v[i] = v_new;
            state.v_hat[i] = state.v_hat[i].max(v_new);
        }

        // Bias correction for first moment
        let m_hat: Vec<f32> = state
            .m
            .iter()
            .map(|&m| m / (1.0 - state.beta1.powf(t)))
            .collect();

        // Update mu
        for (i, &m_hat_val) in m_hat.iter().enumerate().take(self.dim) {
            let update = state.lr * m_hat_val / (state.v_hat[i].sqrt() + state.epsilon);
            self.mu[i] -= update;

            // Ensure finite
            if !self.mu[i].is_finite() {
                self.mu[i] = 0.0;
            }
        }

        // Similar for delta
        let mut m_delta = vec![0.0_f32; self.dim];
        let mut v_delta = vec![0.0_f32; self.dim];
        let mut v_hat_delta = vec![0.0_f32; self.dim];

        for i in 0..self.dim {
            m_delta[i] = state.beta1 * m_delta[i] + (1.0 - state.beta1) * grad_delta[i];
            let v_new: f32 =
                state.beta2 * v_delta[i] + (1.0 - state.beta2) * grad_delta[i] * grad_delta[i];
            v_delta[i] = v_new;
            v_hat_delta[i] = v_hat_delta[i].max(v_new);
        }

        let m_hat_delta: Vec<f32> = m_delta
            .iter()
            .map(|&m| m / (1.0 - state.beta1.powf(t)))
            .collect();

        for i in 0..self.dim {
            let update = state.lr * m_hat_delta[i] / (v_hat_delta[i].sqrt() + state.epsilon);
            self.delta[i] -= update;

            // Clamp delta to reasonable range (width between 0.01 and 10.0)
            self.delta[i] = self.delta[i].clamp(0.01_f32.ln(), 10.0_f32.ln());

            // Ensure finite
            if !self.delta[i].is_finite() {
                self.delta[i] = 0.5_f32.ln();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DenseCone: lightweight cone for the trainer (no tensor deps)
// ---------------------------------------------------------------------------

/// A simple cone used by the trainer implementation.
///
/// Intentionally local to `subsume` (no tensor deps), analogous to [`DenseBox`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DenseCone {
    /// Apex point (origin of the cone).
    pub apex: Vec<f32>,
    /// Unit axis direction.
    pub axis: Vec<f32>,
    /// Half-angle aperture in radians, in (0, pi).
    pub aperture: f32,
}

impl DenseCone {
    pub fn new(apex: Vec<f32>, axis: Vec<f32>, aperture: f32) -> Self {
        Self {
            apex,
            axis,
            aperture,
        }
    }

    /// Angular distance between this cone's axis and another's.
    #[inline]
    fn angular_distance(&self, other: &Self) -> f32 {
        let dot: f32 = self
            .axis
            .iter()
            .zip(other.axis.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        dot.clamp(-1.0, 1.0).acos()
    }

    /// Numerically stable sigmoid.
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    /// Containment probability: P(other inside self).
    ///
    /// ```text
    /// P(other inside self) = sigmoid((self.aperture - angular_dist - other.aperture) / temperature)
    /// ```
    #[inline]
    pub fn containment_prob(&self, other: &Self, temperature: f32) -> f32 {
        let ang_dist = self.angular_distance(other);
        let margin = self.aperture - ang_dist - other.aperture;
        Self::sigmoid(margin / temperature)
    }
}

// ---------------------------------------------------------------------------
// TrainableCone
// ---------------------------------------------------------------------------

/// A trainable cone embedding with learnable parameters.
///
/// Parameterization (ensures validity by construction):
/// - **apex**: unconstrained d-dimensional vector (cone origin)
/// - **raw_axis**: unconstrained d-dimensional vector; normalized to unit length during
///   the forward pass (so gradients flow through all components)
/// - **log_aperture**: unconstrained scalar; aperture = sigmoid(log_aperture) * pi,
///   which maps R -> (0, pi)
///
/// This mirrors the reparameterization strategy used by [`TrainableBox`] (mu/delta),
/// ensuring every parameter receives gradients regardless of the current cone geometry.
///
/// Reference: Zhang & Wang (2021), "ConE: Cone Embeddings for Multi-Hop Reasoning
/// over Knowledge Graphs" (NeurIPS 2021).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableCone {
    /// Apex position in d-dimensional space.
    pub apex: Vec<f32>,
    /// Raw (un-normalized) axis direction. Normalized to unit length in `to_cone()`.
    pub raw_axis: Vec<f32>,
    /// Logit of the aperture. Actual aperture = sigmoid(log_aperture) * pi.
    pub log_aperture: f32,
    /// Dimension.
    pub dim: usize,
}

impl TrainableCone {
    /// Create a new trainable cone from raw parameters.
    ///
    /// # Arguments
    ///
    /// * `apex` - Origin point of the cone
    /// * `raw_axis` - Un-normalized axis direction (will be normalized in forward pass)
    /// * `log_aperture` - Logit-space aperture (aperture = sigmoid(log_aperture) * pi)
    #[must_use]
    pub fn new(apex: Vec<f32>, raw_axis: Vec<f32>, log_aperture: f32) -> Self {
        assert_eq!(
            apex.len(),
            raw_axis.len(),
            "apex and raw_axis must have same dimension"
        );
        let dim = apex.len();
        Self {
            apex,
            raw_axis,
            log_aperture,
            dim,
        }
    }

    /// Initialize from a vector embedding with a given initial aperture.
    ///
    /// The apex is set to the origin, the axis to the normalized input vector,
    /// and the aperture to `init_aperture` (in radians).
    #[must_use]
    pub fn from_vector(vector: &[f32], init_aperture: f32) -> Self {
        let apex = vec![0.0; vector.len()];
        let raw_axis = vector.to_vec();
        // Invert: aperture = sigmoid(log_aperture) * pi  =>  log_aperture = logit(aperture / pi)
        let ratio = (init_aperture / std::f32::consts::PI).clamp(1e-6, 1.0 - 1e-6);
        let log_aperture = (ratio / (1.0 - ratio)).ln();
        Self::new(apex, raw_axis, log_aperture)
    }

    /// Compute the actual aperture in radians, in (0, pi).
    #[inline]
    #[must_use]
    pub fn aperture(&self) -> f32 {
        sigmoid_f32(self.log_aperture) * std::f32::consts::PI
    }

    /// Compute the normalized axis direction (unit vector).
    #[must_use]
    pub fn normalized_axis(&self) -> Vec<f32> {
        let norm: f32 = self
            .raw_axis
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        self.raw_axis.iter().map(|&x| x / norm).collect()
    }

    /// Convert to a [`DenseCone`] (for loss computation / inference).
    #[must_use]
    pub(crate) fn to_cone(&self) -> DenseCone {
        DenseCone::new(self.apex.clone(), self.normalized_axis(), self.aperture())
    }

    /// Number of learnable scalar parameters: dim (apex) + dim (axis) + 1 (aperture).
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        2 * self.dim + 1
    }

    /// Compute containment probability: P(other inside self).
    ///
    /// This is the public inference API. Internally materializes a [`DenseCone`]
    /// and delegates to the sigmoid-based containment formula.
    pub fn containment_prob(&self, other: &Self, temperature: f32) -> f32 {
        let dense_self = self.to_cone();
        let dense_other = other.to_cone();
        dense_self.containment_prob(&dense_other, temperature)
    }

    /// Update cone parameters using AMSGrad optimizer.
    ///
    /// The optimizer state must have been created with dimension = `3 * dim + 1`
    /// (but we only use the first `2 * dim + 1` slots, matching [`num_parameters`]).
    ///
    /// Gradients are passed as three slices:
    /// - `grad_apex` (dim elements): gradient w.r.t. apex
    /// - `grad_axis` (dim elements): gradient w.r.t. raw_axis
    /// - `grad_aperture` (scalar): gradient w.r.t. log_aperture
    pub fn update_amsgrad(
        &mut self,
        grad_apex: &[f32],
        grad_axis: &[f32],
        grad_aperture: f32,
        state: &mut AMSGradState,
    ) {
        // Flatten all grads into one vector for the shared AMSGrad state.
        let n = self.num_parameters();
        let mut grads = Vec::with_capacity(n);
        grads.extend_from_slice(&grad_apex[..self.dim]);
        grads.extend_from_slice(&grad_axis[..self.dim]);
        grads.push(grad_aperture);

        state.t += 1;
        let t = state.t as f32;

        // Update moments
        for (i, &g) in grads.iter().enumerate().take(n) {
            state.m[i] = state.beta1 * state.m[i] + (1.0 - state.beta1) * g;
            let v_new = state.beta2 * state.v[i] + (1.0 - state.beta2) * g * g;
            state.v[i] = v_new;
            state.v_hat[i] = state.v_hat[i].max(v_new);
        }

        // Bias-corrected first moment
        let bias_correction = 1.0 - state.beta1.powf(t);

        // Apply updates
        for i in 0..self.dim {
            let m_hat = state.m[i] / bias_correction;
            let update = state.lr * m_hat / (state.v_hat[i].sqrt() + state.epsilon);
            self.apex[i] -= update;
            if !self.apex[i].is_finite() {
                self.apex[i] = 0.0;
            }
        }

        for i in 0..self.dim {
            let idx = self.dim + i;
            let m_hat = state.m[idx] / bias_correction;
            let update = state.lr * m_hat / (state.v_hat[idx].sqrt() + state.epsilon);
            self.raw_axis[i] -= update;
            if !self.raw_axis[i].is_finite() {
                self.raw_axis[i] = 0.01; // small nonzero to avoid zero-axis
            }
        }

        {
            let idx = 2 * self.dim;
            let m_hat = state.m[idx] / bias_correction;
            let update = state.lr * m_hat / (state.v_hat[idx].sqrt() + state.epsilon);
            self.log_aperture -= update;
            // Clamp log_aperture so aperture stays in a reasonable range.
            // sigmoid(-6) ~ 0.0025 -> aperture ~ 0.008 rad
            // sigmoid(6)  ~ 0.9975 -> aperture ~ 3.134 rad
            self.log_aperture = self.log_aperture.clamp(-6.0, 6.0);
            if !self.log_aperture.is_finite() {
                self.log_aperture = 0.0; // default to pi/2
            }
        }
    }
}

/// Stable sigmoid for f32.
#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- TrainableCone tests --

    #[test]
    fn trainable_cone_aperture_in_valid_range() {
        for log_a in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let cone = TrainableCone::new(vec![0.0, 0.0], vec![1.0, 0.0], log_a);
            let a = cone.aperture();
            assert!(
                a > 0.0 && a < std::f32::consts::PI,
                "aperture must be in (0, pi), got {} for log_aperture={}",
                a,
                log_a
            );
        }
    }

    #[test]
    fn trainable_cone_normalized_axis_is_unit() {
        let cone = TrainableCone::new(vec![0.0, 0.0, 0.0], vec![3.0, 4.0, 0.0], 0.0);
        let axis = cone.normalized_axis();
        let norm: f32 = axis.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "normalized axis should be unit, got norm {}",
            norm
        );
    }

    #[test]
    fn trainable_cone_from_vector_roundtrip() {
        let init_aperture = 1.0_f32; // ~57 degrees
        let cone = TrainableCone::from_vector(&[1.0, 0.0, 0.0], init_aperture);
        let actual = cone.aperture();
        assert!(
            (actual - init_aperture).abs() < 0.01,
            "aperture should roundtrip, expected {} got {}",
            init_aperture,
            actual
        );
    }

    #[test]
    fn trainable_cone_to_dense_cone() {
        let cone = TrainableCone::new(vec![1.0, 2.0], vec![3.0, 4.0], 0.0);
        let dense = cone.to_cone();
        // axis should be normalized
        let norm: f32 = dense.axis.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        // aperture should be sigmoid(0) * pi = 0.5 * pi
        assert!((dense.aperture - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn dense_cone_containment_wide_contains_narrow() {
        // Wide cone (aperture ~2.5) along +x should contain narrow cone (aperture ~0.5) along +x.
        let wide = DenseCone::new(vec![0.0, 0.0], vec![1.0, 0.0], 2.5);
        let narrow = DenseCone::new(vec![0.0, 0.0], vec![1.0, 0.0], 0.5);
        let p = wide.containment_prob(&narrow, 0.1);
        assert!(
            p > 0.99,
            "wide cone should contain narrow cone, got {}",
            p
        );
    }

    #[test]
    fn dense_cone_containment_narrow_does_not_contain_wide() {
        let wide = DenseCone::new(vec![0.0, 0.0], vec![1.0, 0.0], 2.5);
        let narrow = DenseCone::new(vec![0.0, 0.0], vec![1.0, 0.0], 0.5);
        let p = narrow.containment_prob(&wide, 0.1);
        assert!(
            p < 0.01,
            "narrow cone should not contain wide cone, got {}",
            p
        );
    }

    #[test]
    fn trainable_cone_update_amsgrad_does_not_panic() {
        let mut cone = TrainableCone::new(vec![0.0, 0.0], vec![1.0, 0.0], 0.0);
        let mut state = AMSGradState::new(cone.num_parameters(), 0.01);
        let grad_apex = vec![0.1, -0.1];
        let grad_axis = vec![0.05, 0.05];
        let grad_aperture = -0.01;
        cone.update_amsgrad(&grad_apex, &grad_axis, grad_aperture, &mut state);
        // Should not panic and values should remain finite
        assert!(cone.apex.iter().all(|x| x.is_finite()));
        assert!(cone.raw_axis.iter().all(|x| x.is_finite()));
        assert!(cone.log_aperture.is_finite());
    }
}
