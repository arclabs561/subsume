//! Optimizer implementations for box embeddings.

/// AMSGrad optimizer state for a single box.
#[derive(Debug, Clone)]
pub struct AMSGradState {
    /// First moment estimate (m)
    pub m: Vec<f32>,
    /// Second moment estimate (v)
    pub v: Vec<f32>,
    /// Max second moment estimate (v_hat)
    pub v_hat: Vec<f32>,
    /// Iteration counter
    pub t: usize,
    /// Learning rate
    pub lr: f32,
    /// Beta1 (momentum)
    pub beta1: f32,
    /// Beta2 (RMSprop)
    pub beta2: f32,
    /// Epsilon (numerical stability)
    pub epsilon: f32,
}

impl AMSGradState {
    /// Create new AMSGrad state.
    pub fn new(dim: usize, learning_rate: f32) -> Self {
        Self {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            v_hat: vec![0.0; dim],
            t: 0,
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Get learning rate with warmup and cosine decay.
pub fn get_learning_rate(epoch: usize, total_epochs: usize, base_lr: f32, warmup_epochs: usize) -> f32 {
    if epoch < warmup_epochs {
        // Linear warmup: 0.1 * lr → lr
        let warmup_lr = base_lr * 0.1;
        warmup_lr + (base_lr - warmup_lr) * (epoch as f32 / warmup_epochs as f32)
    } else {
        // Cosine decay: lr → 0.1 * lr
        let progress =
            (epoch - warmup_epochs) as f32 / (total_epochs - warmup_epochs).max(1) as f32;
        let min_lr = base_lr * 0.1;
        min_lr + (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
    }
}
