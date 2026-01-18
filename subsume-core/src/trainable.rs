//! Trainable box representations with learnable parameters.

use crate::optimizer::AMSGradState;
use serde::{Deserialize, Serialize};

/// A simple hard box used by the trainer implementation.
///
/// This is intentionally local to `subsume-core` (no tensor deps).
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
