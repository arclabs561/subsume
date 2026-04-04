//! Shared utilities for embedding trainers.
//!
//! Centralises Adam optimizer state, apply-grad helpers, and Adam
//! bias-correction so each trainer module need not duplicate them.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Adam optimizer state
// ---------------------------------------------------------------------------

/// Persistent Adam optimizer state for a set of named scalar parameters.
///
/// Using `String`-keyed HashMaps keeps trainers generic: each parameter
/// (e.g. `"h3_c0"` for entity 3, center dim 0) gets its own `m` / `v`.
/// State accumulates across `train_epoch` calls so that momentum is not
/// lost between epochs.
#[derive(Debug, Default)]
pub struct AdamState {
    m: HashMap<String, f32>,
    v: HashMap<String, f32>,
    pub step: usize,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl AdamState {
    /// Create a new Adam state with the standard hyperparameters.
    pub fn new() -> Self {
        Self {
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Increment the step counter and return `(bias1, bias2)`.
    pub fn tick(&mut self) -> (f32, f32) {
        self.step += 1;
        let t = self.step as f32;
        (1.0 - self.beta1.powf(t), 1.0 - self.beta2.powf(t))
    }

    /// Apply one Adam update to `param`, returning the raw Adam step.
    pub fn apply(
        &mut self,
        key: &str,
        param: &mut f32,
        grad: f32,
        lr: f32,
        bias1: f32,
        bias2: f32,
    ) {
        let m = self.m.entry(key.to_string()).or_insert(0.0);
        let v = self.v.entry(key.to_string()).or_insert(0.0);
        *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
        *v = self.beta2 * *v + (1.0 - self.beta2) * grad * grad;
        let m_hat = *m / bias1;
        let v_hat = (*v / bias2).max(0.0);
        *param -= lr * m_hat / (v_hat.sqrt() + self.eps);
    }

    /// Apply Adam to a log-parameterized scalar (e.g. log_radius).
    ///
    /// Gets current value, applies Adam update, then calls `setter` with the
    /// new value so the caller can re-apply `exp()` / `clamp` as needed.
    pub fn apply_log<T, F>(
        &mut self,
        key: &str,
        current: f32,
        grad: f32,
        lr: f32,
        bias1: f32,
        bias2: f32,
        setter: F,
        target: &mut T,
    ) where
        F: Fn(&mut T, f32),
    {
        let m = self.m.entry(key.to_string()).or_insert(0.0);
        let v = self.v.entry(key.to_string()).or_insert(0.0);
        *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
        *v = self.beta2 * *v + (1.0 - self.beta2) * grad * grad;
        let m_hat = *m / bias1;
        let v_hat = (*v / bias2).max(0.0);
        let new_val = current - lr * m_hat / (v_hat.sqrt() + self.eps);
        setter(target, new_val);
    }
}

// ---------------------------------------------------------------------------
// Self-adversarial negative weighting (Sun et al., RotatE, ICLR 2019)
// ---------------------------------------------------------------------------

/// Compute self-adversarial weights for a set of negative scores.
///
/// Given the current model scores for `n` negatives, weights each negative
/// by its softmax probability scaled by `temperature`:
///
/// ```text
/// w_i = softmax(alpha * score_i)_i
/// ```
///
/// where `alpha` is the adversarial temperature. Higher alpha → harder
/// negatives receive more weight.
///
/// Returns a `Vec<f32>` of the same length as `scores` summing to 1.0.
pub fn self_adversarial_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    if scores.is_empty() || temperature <= 0.0 {
        let n = scores.len();
        return vec![1.0 / n.max(1) as f32; n];
    }
    // Scale scores, subtract max for numerical stability
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores
        .iter()
        .map(|&s| ((s - max_s) * temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-12 {
        let n = scores.len();
        return vec![1.0 / n as f32; n];
    }
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_state_step_increments() {
        let mut s = AdamState::new();
        let (b1, b2) = s.tick();
        assert_eq!(s.step, 1);
        assert!((b1 - (1.0 - 0.9_f32)).abs() < 1e-6);
        assert!((b2 - (1.0 - 0.999_f32)).abs() < 1e-6);
    }

    #[test]
    fn adam_state_moves_param() {
        let mut s = AdamState::new();
        let (b1, b2) = s.tick();
        let mut p = 0.5f32;
        s.apply("x", &mut p, 1.0, 0.01, b1, b2);
        // Positive gradient → param decreased
        assert!(p < 0.5, "param should decrease for positive gradient");
    }

    #[test]
    fn adam_state_persists() {
        let mut s = AdamState::new();
        let mut p = 0.0f32;
        for _ in 0..5 {
            let (b1, b2) = s.tick();
            s.apply("x", &mut p, 1.0, 0.01, b1, b2);
        }
        // After 5 steps with grad=1, param should have moved meaningfully
        assert!(
            p < -0.01,
            "param should have decreased after 5 gradient steps"
        );
    }

    #[test]
    fn self_adversarial_uniform_for_equal_scores() {
        let weights = self_adversarial_weights(&[1.0, 1.0, 1.0], 1.0);
        for w in &weights {
            assert!((w - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn self_adversarial_sums_to_one() {
        let weights = self_adversarial_weights(&[0.5, 1.5, 0.1, 2.0], 1.0);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn self_adversarial_higher_score_gets_more_weight() {
        let weights = self_adversarial_weights(&[1.0, 3.0], 1.0);
        assert!(
            weights[1] > weights[0],
            "higher score should get higher weight"
        );
    }

    #[test]
    fn self_adversarial_empty_is_empty() {
        let weights = self_adversarial_weights(&[], 1.0);
        assert!(weights.is_empty());
    }
}
