//! Optimizer implementations for ndarray backend.
//!
//! Provides Adam, AdamW, and SGD optimizers for training box embeddings.
//! Based on paper hyperparameters: learning rate 1e-3 to 5e-4, Adam optimizer.

use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;

/// Adam optimizer state for a single parameter.
#[derive(Debug, Clone)]
struct AdamState {
    /// First moment estimate (momentum)
    m: Array1<f32>,
    /// Second moment estimate (variance)
    v: Array1<f32>,
    /// Time step
    t: usize,
}

/// Adam optimizer for training box embeddings.
///
/// Implements Adam optimizer as used in box embedding papers:
/// - Dasgupta 2020: Adam with learning rate 1e-3 to 5e-4
/// - Boratko 2020: Adam with learning rate 1e-3
/// - Vilnis 2018: Adam with learning rate 1e-3 to 1e-4
///
/// # Hyperparameters
///
/// - `learning_rate`: Step size (default: 1e-3, typical range: 5e-4 to 1e-3)
/// - `beta1`: Exponential decay rate for first moment (default: 0.9)
/// - `beta2`: Exponential decay rate for second moment (default: 0.999)
/// - `epsilon`: Small constant for numerical stability (default: 1e-8)
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate
    learning_rate: f32,
    /// Beta1 (momentum decay)
    beta1: f32,
    /// Beta2 (variance decay)
    beta2: f32,
    /// Epsilon for numerical stability
    epsilon: f32,
    /// State for each parameter
    states: HashMap<String, AdamState>,
}

impl Adam {
    /// Create new Adam optimizer.
    ///
    /// # Parameters
    ///
    /// - `learning_rate`: Step size (default: 1e-3, typical: 5e-4 to 1e-3)
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            states: HashMap::new(),
        }
    }

    /// Create Adam with custom hyperparameters.
    pub fn with_params(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            states: HashMap::new(),
        }
    }

    /// Update parameters using gradients.
    ///
    /// # Parameters
    ///
    /// - `param_name`: Unique identifier for this parameter
    /// - `param`: Parameter values to update (mutated in place)
    /// - `grad`: Gradients for this parameter
    pub fn update(&mut self, param_name: &str, param: &mut Array1<f32>, grad: ArrayView1<f32>) {
        // Initialize state if needed
        let state = self.states.entry(param_name.to_string()).or_insert_with(|| {
            AdamState {
                m: Array1::zeros(param.len()),
                v: Array1::zeros(param.len()),
                t: 0,
            }
        });

        state.t += 1;

        // Update biased first moment estimate
        let grad_owned = grad.to_owned();
        state.m = &(&state.m * self.beta1) + &(&grad_owned * (1.0 - self.beta1));

        // Update biased second raw moment estimate
        let grad_sq = grad.mapv(|x| x * x);
        state.v = &(&state.v * self.beta2) + &(&grad_sq * (1.0 - self.beta2));

        // Compute bias-corrected estimates
        let beta1_t = 1.0 - self.beta1.powi(state.t as i32);
        let beta2_t = 1.0 - self.beta2.powi(state.t as i32);
        let m_hat = &state.m / beta1_t;
        let v_hat = &state.v / beta2_t;

        // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        let v_hat_sqrt = v_hat.mapv(|x| x.sqrt());
        let denominator = &v_hat_sqrt + self.epsilon;
        let step = &(&m_hat / &denominator) * self.learning_rate;
        let param_new = param.to_owned() - &step;
        *param = param_new;
    }

    /// Reset optimizer state (useful for restarting training).
    pub fn reset(&mut self) {
        self.states.clear();
    }

    /// Get learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Set learning rate (useful for learning rate scheduling).
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
///
/// AdamW improves upon Adam by decoupling weight decay from gradient-based updates.
/// This often leads to better generalization and is the preferred optimizer in many
/// modern deep learning frameworks.
///
/// Key difference from Adam: weight decay is applied directly to parameters,
/// not added to gradients.
///
/// # Hyperparameters
///
/// - `learning_rate`: Step size (default: 1e-3)
/// - `weight_decay`: Decoupled weight decay coefficient (default: 1e-2)
/// - `beta1`: Exponential decay rate for first moment (default: 0.9)
/// - `beta2`: Exponential decay rate for second moment (default: 0.999)
/// - `epsilon`: Small constant for numerical stability (default: 1e-8)
#[derive(Debug, Clone)]
pub struct AdamW {
    /// Learning rate
    learning_rate: f32,
    /// Weight decay (decoupled)
    weight_decay: f32,
    /// Beta1 (momentum decay)
    beta1: f32,
    /// Beta2 (variance decay)
    beta2: f32,
    /// Epsilon for numerical stability
    epsilon: f32,
    /// State for each parameter
    states: HashMap<String, AdamState>,
}

impl AdamW {
    /// Create new AdamW optimizer.
    ///
    /// # Parameters
    ///
    /// - `learning_rate`: Step size (default: 1e-3)
    /// - `weight_decay`: Decoupled weight decay (default: 1e-2)
    pub fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            states: HashMap::new(),
        }
    }

    /// Create AdamW with custom hyperparameters.
    pub fn with_params(
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        Self {
            learning_rate,
            weight_decay,
            beta1,
            beta2,
            epsilon,
            states: HashMap::new(),
        }
    }

    /// Update parameters using gradients.
    ///
    /// # Parameters
    ///
    /// - `param_name`: Unique identifier for this parameter
    /// - `param`: Parameter values to update (mutated in place)
    /// - `grad`: Gradients for this parameter
    pub fn update(&mut self, param_name: &str, param: &mut Array1<f32>, grad: ArrayView1<f32>) {
        // Initialize state if needed
        let state = self.states.entry(param_name.to_string()).or_insert_with(|| {
            AdamState {
                m: Array1::zeros(param.len()),
                v: Array1::zeros(param.len()),
                t: 0,
            }
        });

        state.t += 1;

        // Update biased first moment estimate
        let grad_owned = grad.to_owned();
        state.m = &(&state.m * self.beta1) + &(&grad_owned * (1.0 - self.beta1));

        // Update biased second raw moment estimate
        let grad_sq = grad.mapv(|x| x * x);
        state.v = &(&state.v * self.beta2) + &(&grad_sq * (1.0 - self.beta2));

        // Compute bias-corrected estimates
        let beta1_t = 1.0 - self.beta1.powi(state.t as i32);
        let beta2_t = 1.0 - self.beta2.powi(state.t as i32);
        let m_hat = &state.m / beta1_t;
        let v_hat = &state.v / beta2_t;

        // Update parameters: param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
        // Key difference from Adam: weight decay is applied directly, not added to gradient
        let v_hat_sqrt = v_hat.mapv(|x| x.sqrt());
        let denominator = &v_hat_sqrt + self.epsilon;
        let step = &(&m_hat / &denominator) * self.learning_rate;
        let param_clone = param.to_owned();
        let weight_decay_step = &param_clone * (self.learning_rate * self.weight_decay);
        let param_new = param_clone - &step - &weight_decay_step;
        *param = param_new;
    }

    /// Reset optimizer state (useful for restarting training).
    pub fn reset(&mut self) {
        self.states.clear();
    }

    /// Get learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Set learning rate (useful for learning rate scheduling).
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Get weight decay.
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set weight decay.
    pub fn set_weight_decay(&mut self, wd: f32) {
        self.weight_decay = wd;
    }
}

/// Simple SGD optimizer (for comparison).
#[derive(Debug, Clone)]
pub struct SGD {
    /// Learning rate
    learning_rate: f32,
    /// Momentum coefficient (0.0 = no momentum)
    momentum: f32,
    /// Momentum buffer
    velocity: HashMap<String, Array1<f32>>,
}

impl SGD {
    /// Create new SGD optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            velocity: HashMap::new(),
        }
    }

    /// Create SGD with momentum.
    pub fn with_momentum(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }

    /// Update parameters using gradients.
    pub fn update(&mut self, param_name: &str, param: &mut Array1<f32>, grad: ArrayView1<f32>) {
        if self.momentum > 0.0 {
            // Momentum SGD
            let v = self
                .velocity
                .entry(param_name.to_string())
                .or_insert_with(|| Array1::zeros(param.len()));

            let grad_owned = grad.to_owned();
            let v_new = &(&v.to_owned() * self.momentum) + &(&grad_owned * self.learning_rate);
            *v = v_new.clone();
            let param_new = param.to_owned() - &v_new;
            *param = param_new;
        } else {
            // Plain SGD
            let grad_owned = grad.to_owned();
            let step = &grad_owned * self.learning_rate;
            let param_new = param.to_owned() - &step;
            *param = param_new;
        }
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.velocity.clear();
    }

    /// Set learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adam_update() {
        let mut optimizer = Adam::new(0.01);
        let mut param = array![1.0, 2.0, 3.0];
        let grad_array = array![0.1, 0.2, 0.3];
        let grad = grad_array.view();

        // First update
        optimizer.update("param1", &mut param, grad);
        
        // Parameter should have changed
        assert_ne!(param, array![1.0, 2.0, 3.0]);
        
        // Should decrease (gradient is positive, so we subtract)
        assert!(param[0] < 1.0);
    }

    #[test]
    fn test_sgd_update() {
        let mut optimizer = SGD::new(0.01);
        let mut param = array![1.0, 2.0];
        let grad_array = array![0.1, 0.2];
        let grad = grad_array.view();

        optimizer.update("param1", &mut param, grad);
        
        // SGD: param = param - lr * grad
        // Expected: [1.0 - 0.01*0.1, 2.0 - 0.01*0.2] = [0.999, 1.998]
        assert!((param[0] - 0.999).abs() < 1e-5);
        assert!((param[1] - 1.998).abs() < 1e-5);
    }

    #[test]
    fn test_adamw_update() {
        let mut optimizer = AdamW::new(0.01, 0.001);
        let mut param = array![1.0, 2.0, 3.0];
        let grad_array = array![0.1, 0.2, 0.3];
        let grad = grad_array.view();

        // First update
        optimizer.update("param1", &mut param, grad);
        
        // Parameter should have changed
        assert_ne!(param, array![1.0, 2.0, 3.0]);
        
        // Should decrease (gradient is positive, so we subtract)
        assert!(param[0] < 1.0);
        
        // Weight decay should also be applied
        // After first step, weight decay effect should be visible
        let old_param = param.clone();
        let grad2_array = array![0.05, 0.15, 0.25];
        optimizer.update("param1", &mut param, grad2_array.view());
        assert_ne!(param[0], old_param[0]);
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut optimizer = AdamW::new(0.01, 0.1); // High weight decay
        let mut param = array![1.0];
        let grad_array = array![0.0]; // Zero gradient
        
        // With zero gradient but weight decay, param should shrink
        optimizer.update("param1", &mut param, grad_array.view());
        assert!(param[0] < 1.0, "Weight decay should shrink parameter even with zero gradient");
    }
}

