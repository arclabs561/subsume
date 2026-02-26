//! Learning rate schedulers for training box embeddings.
//!
//! Provides common learning rate scheduling strategies that work with
//! the optimizers in this crate. Use `set_learning_rate()` on the optimizer
//! after each scheduler step.
//!
//! # Example
//!
//! ```rust
//! use subsume::ndarray_backend::{Adam, scheduler::{CosineAnnealingLr, LrScheduler}};
//!
//! let mut optimizer = Adam::new(1e-3);
//! let mut scheduler = CosineAnnealingLr::new(1e-3, 1e-5, 1000);
//!
//! for step in 0..1000 {
//!     // ... training step ...
//!     let lr = scheduler.step();
//!     optimizer.set_learning_rate(lr);
//! }
//! ```

use std::f32::consts::PI;

/// Learning rate scheduler trait.
pub trait LrScheduler {
    /// Advance scheduler by one step and return the new learning rate.
    fn step(&mut self) -> f32;

    /// Get current learning rate without advancing.
    fn get_lr(&self) -> f32;

    /// Get current step number.
    fn get_step(&self) -> usize;
}

/// Constant learning rate (no scheduling).
///
/// Useful as a baseline or when no scheduling is needed.
#[derive(Debug, Clone)]
pub struct ConstantLr {
    lr: f32,
    step: usize,
}

impl ConstantLr {
    /// Create constant learning rate scheduler.
    pub fn new(lr: f32) -> Self {
        Self { lr, step: 0 }
    }
}

impl LrScheduler for ConstantLr {
    fn step(&mut self) -> f32 {
        self.step += 1;
        self.lr
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn get_step(&self) -> usize {
        self.step
    }
}

/// Linear warmup followed by constant learning rate.
///
/// During warmup, LR increases linearly from `start_lr` to `target_lr`.
/// After warmup, LR stays at `target_lr`.
///
/// Common for transformer training to avoid early instability.
#[derive(Debug, Clone)]
pub struct WarmupLr {
    start_lr: f32,
    target_lr: f32,
    warmup_steps: usize,
    current_step: usize,
}

impl WarmupLr {
    /// Create warmup scheduler.
    ///
    /// # Arguments
    /// * `target_lr` - Learning rate after warmup
    /// * `warmup_steps` - Number of warmup steps
    /// * `start_lr` - Initial learning rate (default: 0.0)
    pub fn new(target_lr: f32, warmup_steps: usize) -> Self {
        Self {
            start_lr: 0.0,
            target_lr,
            warmup_steps,
            current_step: 0,
        }
    }

    /// Create warmup scheduler with custom start LR.
    pub fn with_start_lr(target_lr: f32, warmup_steps: usize, start_lr: f32) -> Self {
        Self {
            start_lr,
            target_lr,
            warmup_steps,
            current_step: 0,
        }
    }
}

impl LrScheduler for WarmupLr {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }

    fn get_lr(&self) -> f32 {
        if self.current_step >= self.warmup_steps {
            self.target_lr
        } else {
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            self.start_lr + (self.target_lr - self.start_lr) * progress
        }
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

/// Cosine annealing learning rate scheduler.
///
/// LR follows a cosine curve from `max_lr` to `min_lr` over `total_steps`.
///
/// Formula: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / total_steps))
///
/// Reference: Loshchilov & Hutter (2017) "SGDR: Stochastic Gradient Descent
/// with Warm Restarts"
#[derive(Debug, Clone)]
pub struct CosineAnnealingLr {
    max_lr: f32,
    min_lr: f32,
    total_steps: usize,
    current_step: usize,
}

impl CosineAnnealingLr {
    /// Create cosine annealing scheduler.
    ///
    /// # Arguments
    /// * `max_lr` - Maximum (starting) learning rate
    /// * `min_lr` - Minimum (ending) learning rate
    /// * `total_steps` - Total number of steps for one cycle
    pub fn new(max_lr: f32, min_lr: f32, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr,
            total_steps,
            current_step: 0,
        }
    }
}

impl LrScheduler for CosineAnnealingLr {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }

    fn get_lr(&self) -> f32 {
        if self.current_step >= self.total_steps {
            self.min_lr
        } else {
            let progress = self.current_step as f32 / self.total_steps as f32;
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
        }
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

/// Cosine annealing with linear warmup.
///
/// Combines linear warmup with cosine annealing:
/// 1. During warmup: LR increases linearly from 0 to `max_lr`
/// 2. After warmup: LR follows cosine from `max_lr` to `min_lr`
///
/// Common schedule for transformer training.
#[derive(Debug, Clone)]
pub struct WarmupCosineAnnealingLr {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl WarmupCosineAnnealingLr {
    /// Create warmup + cosine annealing scheduler.
    ///
    /// # Arguments
    /// * `max_lr` - Peak learning rate (reached at end of warmup)
    /// * `min_lr` - Minimum learning rate (reached at end of training)
    /// * `warmup_steps` - Number of warmup steps
    /// * `total_steps` - Total training steps (including warmup)
    pub fn new(max_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }
}

impl LrScheduler for WarmupCosineAnnealingLr {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }

    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            self.max_lr * progress
        } else if self.current_step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine annealing
            let decay_steps = self.total_steps - self.warmup_steps;
            let progress = (self.current_step - self.warmup_steps) as f32 / decay_steps as f32;
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
        }
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

/// Exponential decay learning rate scheduler.
///
/// LR decays by a constant factor each step:
/// lr = initial_lr * gamma^step
///
/// Simple and effective for many tasks.
#[derive(Debug, Clone)]
pub struct ExponentialLr {
    initial_lr: f32,
    gamma: f32,
    current_step: usize,
}

impl ExponentialLr {
    /// Create exponential decay scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Starting learning rate
    /// * `gamma` - Decay factor per step (e.g., 0.999 for slow decay)
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            current_step: 0,
        }
    }
}

impl LrScheduler for ExponentialLr {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }

    fn get_lr(&self) -> f32 {
        self.initial_lr * self.gamma.powi(self.current_step as i32)
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

/// Step decay learning rate scheduler.
///
/// LR drops by `gamma` every `step_size` steps:
/// lr = initial_lr * gamma^(step / step_size)
///
/// Simple milestone-based decay.
#[derive(Debug, Clone)]
pub struct StepLr {
    initial_lr: f32,
    gamma: f32,
    step_size: usize,
    current_step: usize,
}

impl StepLr {
    /// Create step decay scheduler.
    ///
    /// # Arguments
    /// * `initial_lr` - Starting learning rate
    /// * `gamma` - Factor to multiply LR by at each milestone (e.g., 0.1)
    /// * `step_size` - Steps between LR reductions
    pub fn new(initial_lr: f32, gamma: f32, step_size: usize) -> Self {
        Self {
            initial_lr,
            gamma,
            step_size,
            current_step: 0,
        }
    }
}

impl LrScheduler for StepLr {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }

    fn get_lr(&self) -> f32 {
        let num_decays = self.current_step / self.step_size;
        self.initial_lr * self.gamma.powi(num_decays as i32)
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let mut scheduler = ConstantLr::new(0.01);
        for _ in 0..100 {
            assert!((scheduler.step() - 0.01).abs() < 1e-6);
        }
    }

    #[test]
    fn test_warmup_lr() {
        let mut scheduler = WarmupLr::new(1.0, 10);

        // During warmup, LR should increase
        let lr1 = scheduler.step();
        let lr2 = scheduler.step();
        assert!(lr2 > lr1);

        // After warmup, should be at target
        for _ in 0..20 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing() {
        let mut scheduler = CosineAnnealingLr::new(1.0, 0.0, 100);

        // Start near max
        let lr_start = scheduler.get_lr();
        assert!((lr_start - 1.0).abs() < 1e-6);

        // Middle should be around 0.5
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.5).abs() < 0.1);

        // End should be near min
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_cosine() {
        let mut scheduler = WarmupCosineAnnealingLr::new(1.0, 0.0, 10, 100);

        // Start at 0
        assert!((scheduler.get_lr() - 0.0).abs() < 1e-6);

        // After warmup, at peak
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1.0).abs() < 0.1);

        // End near min
        for _ in 0..90 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_decay() {
        let mut scheduler = ExponentialLr::new(1.0, 0.9);

        let lr0 = scheduler.get_lr();
        scheduler.step();
        let lr1 = scheduler.get_lr();

        // Should decay by gamma
        assert!((lr1 - 0.9).abs() < 1e-6);
        assert!(lr1 < lr0);
    }

    #[test]
    fn test_step_lr() {
        let mut scheduler = StepLr::new(1.0, 0.1, 10);

        // First 9 steps at initial LR (steps 1-9 have step/10 = 0)
        for i in 1..=9 {
            let lr = scheduler.step();
            assert!(
                (lr - 1.0).abs() < 1e-6,
                "Step {}: expected 1.0, got {}",
                i,
                lr
            );
        }

        // Step 10 triggers first decay (step/10 = 1)
        let lr = scheduler.step();
        assert!((lr - 0.1).abs() < 1e-6, "Step 10: expected 0.1, got {}", lr);
    }

    // ---- Exponential LR: monotone decrease ----

    #[test]
    fn exponential_lr_monotone_decrease() {
        let mut scheduler = ExponentialLr::new(1.0, 0.95);
        let mut prev = scheduler.get_lr();
        for _ in 0..50 {
            let lr = scheduler.step();
            assert!(lr <= prev + 1e-7, "LR should be non-increasing");
            assert!(lr > 0.0, "LR should remain positive");
            prev = lr;
        }
    }

    #[test]
    fn exponential_lr_exact_values() {
        let mut scheduler = ExponentialLr::new(2.0, 0.5);
        // step 0 (before step): 2.0
        assert!((scheduler.get_lr() - 2.0).abs() < 1e-6);
        // step 1: 2.0 * 0.5 = 1.0
        scheduler.step();
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-6);
        // step 2: 2.0 * 0.25 = 0.5
        scheduler.step();
        assert!((scheduler.get_lr() - 0.5).abs() < 1e-6);
    }

    // ---- Step LR: multiple milestones ----

    #[test]
    fn step_lr_multiple_milestones() {
        let mut scheduler = StepLr::new(1.0, 0.5, 5);
        // Steps 1-4: 1.0
        for _ in 0..4 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-6);

        // Step 5: first decay -> 0.5
        scheduler.step();
        assert!((scheduler.get_lr() - 0.5).abs() < 1e-6);

        // Steps 6-9: still 0.5
        for _ in 0..4 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.5).abs() < 1e-6);

        // Step 10: second decay -> 0.25
        scheduler.step();
        assert!((scheduler.get_lr() - 0.25).abs() < 1e-6);
    }

    // ---- Warmup LR: custom start ----

    #[test]
    fn warmup_lr_custom_start() {
        let mut scheduler = WarmupLr::with_start_lr(1.0, 10, 0.5);
        // At step 0 (before step), lr should be start_lr = 0.5
        assert!((scheduler.get_lr() - 0.5).abs() < 1e-6);

        // Midpoint of warmup: 0.5 + (1.0-0.5)*5/10 = 0.75
        for _ in 0..5 {
            scheduler.step();
        }
        assert!(
            (scheduler.get_lr() - 0.75).abs() < 1e-6,
            "Expected 0.75, got {}",
            scheduler.get_lr()
        );
    }

    // ---- Beyond total steps ----

    #[test]
    fn cosine_annealing_beyond_total_stays_at_min() {
        let mut scheduler = CosineAnnealingLr::new(1.0, 0.01, 100);
        for _ in 0..200 {
            scheduler.step();
        }
        assert!(
            (scheduler.get_lr() - 0.01).abs() < 1e-6,
            "Beyond total steps should stay at min_lr"
        );
    }

    #[test]
    fn warmup_cosine_beyond_total_stays_at_min() {
        let mut scheduler = WarmupCosineAnnealingLr::new(1.0, 0.01, 10, 100);
        for _ in 0..200 {
            scheduler.step();
        }
        assert!(
            (scheduler.get_lr() - 0.01).abs() < 1e-6,
            "Beyond total steps should stay at min_lr"
        );
    }

    // ---- get_step ----

    #[test]
    fn get_step_tracks_calls() {
        let mut s1 = ConstantLr::new(0.1);
        assert_eq!(s1.get_step(), 0);
        s1.step();
        s1.step();
        s1.step();
        assert_eq!(s1.get_step(), 3);

        let mut s2 = ExponentialLr::new(1.0, 0.9);
        assert_eq!(s2.get_step(), 0);
        for _ in 0..7 {
            s2.step();
        }
        assert_eq!(s2.get_step(), 7);
    }

    // ---- All schedulers produce finite positive LR ----

    #[test]
    fn all_schedulers_produce_finite_positive_lr() {
        let schedulers: Vec<Box<dyn LrScheduler>> = vec![
            Box::new(ConstantLr::new(0.01)),
            Box::new(WarmupLr::new(0.01, 10)),
            Box::new(CosineAnnealingLr::new(0.01, 0.001, 100)),
            Box::new(WarmupCosineAnnealingLr::new(0.01, 0.001, 10, 100)),
            Box::new(ExponentialLr::new(0.01, 0.99)),
            Box::new(StepLr::new(0.01, 0.5, 10)),
        ];

        for mut sched in schedulers {
            for step_num in 0..150 {
                let lr = sched.step();
                assert!(
                    lr.is_finite() && lr >= 0.0,
                    "Scheduler produced invalid LR {} at step {}",
                    lr, step_num
                );
            }
        }
    }

    // ---- Cosine annealing LR is non-increasing ----

    #[test]
    fn cosine_annealing_monotone_decrease() {
        let mut scheduler = CosineAnnealingLr::new(1.0, 0.0, 200);
        let mut prev = scheduler.get_lr();
        for _ in 0..200 {
            let lr = scheduler.step();
            assert!(
                lr <= prev + 1e-6,
                "Cosine annealing should be non-increasing"
            );
            prev = lr;
        }
    }
}
