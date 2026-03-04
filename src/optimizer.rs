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
pub fn get_learning_rate(
    epoch: usize,
    total_epochs: usize,
    base_lr: f32,
    warmup_epochs: usize,
) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // AMSGradState tests
    // =========================================================================

    #[test]
    fn amsgrad_new_initializes_zeros() {
        let state = AMSGradState::new(4, 0.01);
        assert_eq!(state.m.len(), 4);
        assert_eq!(state.v.len(), 4);
        assert_eq!(state.v_hat.len(), 4);
        assert!(state.m.iter().all(|&x| x == 0.0));
        assert!(state.v.iter().all(|&x| x == 0.0));
        assert!(state.v_hat.iter().all(|&x| x == 0.0));
        assert_eq!(state.t, 0);
        assert_eq!(state.lr, 0.01);
    }

    #[test]
    fn amsgrad_default_hyperparams() {
        let state = AMSGradState::new(2, 1e-3);
        assert_eq!(state.beta1, 0.9);
        assert_eq!(state.beta2, 0.999);
        assert_eq!(state.epsilon, 1e-8);
    }

    #[test]
    fn amsgrad_set_lr() {
        let mut state = AMSGradState::new(2, 0.01);
        assert_eq!(state.lr, 0.01);
        state.set_lr(0.001);
        assert_eq!(state.lr, 0.001);
    }

    #[test]
    fn amsgrad_zero_dim() {
        let state = AMSGradState::new(0, 0.01);
        assert_eq!(state.m.len(), 0);
        assert_eq!(state.v.len(), 0);
        assert_eq!(state.v_hat.len(), 0);
    }

    #[test]
    fn amsgrad_large_dim() {
        let state = AMSGradState::new(10_000, 0.01);
        assert_eq!(state.m.len(), 10_000);
    }

    // =========================================================================
    // get_learning_rate tests
    // =========================================================================

    #[test]
    fn lr_warmup_start_is_tenth_of_base() {
        // At epoch 0, lr should be 0.1 * base_lr
        let lr = get_learning_rate(0, 100, 1.0, 10);
        assert!((lr - 0.1).abs() < 1e-6, "epoch 0: expected 0.1, got {lr}");
    }

    #[test]
    fn lr_warmup_end_equals_base() {
        // At the end of warmup (epoch == warmup_epochs), we enter cosine phase.
        // Just before warmup ends (epoch = warmup - 1), lr should be close to base.
        let base = 1.0;
        let warmup = 10;
        let lr_last_warmup = get_learning_rate(warmup - 1, 100, base, warmup);
        // epoch 9 out of 10 warmup: 0.1 + 0.9 * (9/10) = 0.1 + 0.81 = 0.91
        assert!((lr_last_warmup - 0.91).abs() < 1e-6);
    }

    #[test]
    fn lr_at_warmup_boundary_enters_cosine() {
        // At epoch == warmup_epochs, cosine phase begins with progress = 0
        // cos(0) = 1.0, so lr = min_lr + (base - min_lr) * (1 + 1) / 2 = base
        let lr = get_learning_rate(10, 100, 1.0, 10);
        assert!((lr - 1.0).abs() < 1e-6, "cosine start should equal base_lr");
    }

    #[test]
    fn lr_cosine_end_is_tenth_of_base() {
        // At last epoch, progress = 1.0, cos(pi) = -1
        // lr = min_lr + (base - min_lr) * (1 + (-1)) / 2 = min_lr = 0.1 * base
        let lr = get_learning_rate(99, 100, 1.0, 10);
        assert!(
            (lr - 0.1).abs() < 1e-3,
            "cosine end should approach 0.1*base, got {lr}"
        );
    }

    #[test]
    fn lr_cosine_midpoint_is_halfway() {
        // Midpoint of cosine: progress = 0.5, cos(pi/2) = 0
        // lr = min_lr + (base - min_lr) * (1 + 0) / 2 = (base + min_lr) / 2
        let base = 1.0;
        let warmup = 0; // No warmup for simplicity
        let total = 100;
        let mid = total / 2;
        let lr = get_learning_rate(mid, total, base, warmup);
        let expected = (base + 0.1 * base) / 2.0; // 0.55
        assert!(
            (lr - expected).abs() < 1e-3,
            "expected ~{expected}, got {lr}"
        );
    }

    #[test]
    fn lr_monotone_decreasing_in_cosine_phase() {
        let base = 1.0;
        let warmup = 5;
        let total = 50;
        let mut prev = get_learning_rate(warmup, total, base, warmup);
        for epoch in (warmup + 1)..total {
            let lr = get_learning_rate(epoch, total, base, warmup);
            assert!(
                lr <= prev + 1e-6,
                "LR should be non-increasing in cosine phase: epoch {epoch}"
            );
            prev = lr;
        }
    }

    #[test]
    fn lr_monotone_increasing_in_warmup() {
        let base = 1.0;
        let warmup = 20;
        let total = 100;
        let mut prev = get_learning_rate(0, total, base, warmup);
        for epoch in 1..warmup {
            let lr = get_learning_rate(epoch, total, base, warmup);
            assert!(
                lr >= prev - 1e-6,
                "LR should be non-decreasing in warmup: epoch {epoch}"
            );
            prev = lr;
        }
    }

    #[test]
    fn lr_no_warmup() {
        // warmup_epochs = 0 means no warmup, cosine from epoch 0
        let lr0 = get_learning_rate(0, 100, 1.0, 0);
        assert!(
            (lr0 - 1.0).abs() < 1e-6,
            "no warmup: epoch 0 should be base_lr"
        );
    }

    #[test]
    fn lr_single_epoch() {
        // total_epochs = 1, warmup = 0
        let lr = get_learning_rate(0, 1, 0.5, 0);
        assert!(lr.is_finite());
        assert!(lr > 0.0);
    }

    #[test]
    fn lr_warmup_equals_total() {
        // Edge case: warmup_epochs == total_epochs
        // All epochs are warmup. The cosine branch uses .max(1) denominator.
        let lr = get_learning_rate(5, 10, 1.0, 10);
        // epoch 5 < warmup 10, so warmup formula: 0.1 + 0.9 * 5/10 = 0.55
        assert!((lr - 0.55).abs() < 1e-6);
    }

    #[test]
    fn lr_always_positive() {
        for epoch in 0..200 {
            let lr = get_learning_rate(epoch, 100, 0.01, 10);
            assert!(lr > 0.0, "LR must be positive at epoch {epoch}");
            assert!(lr.is_finite(), "LR must be finite at epoch {epoch}");
        }
    }
}
