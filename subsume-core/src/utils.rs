//! Numerical stability utilities for box embeddings.

/// Clamp temperature to a safe range to avoid numerical instability.
///
/// Very low temperatures can cause vanishing gradients and exponential underflow,
/// while very high temperatures lose correspondence to discrete distributions.
///
/// # Parameters
///
/// - `temp`: Raw temperature value
/// - `min`: Minimum safe temperature (default: 1e-3)
/// - `max`: Maximum safe temperature (default: 10.0)
///
/// # Returns
///
/// Clamped temperature value
pub fn clamp_temperature(temp: f32, min: f32, max: f32) -> f32 {
    temp.clamp(min, max)
}

/// Default minimum safe temperature to avoid numerical underflow.
pub const MIN_TEMPERATURE: f32 = 1e-3;

/// Default maximum safe temperature to maintain correspondence.
pub const MAX_TEMPERATURE: f32 = 10.0;

/// Clamp temperature using default safe bounds.
pub fn clamp_temperature_default(temp: f32) -> f32 {
    clamp_temperature(temp, MIN_TEMPERATURE, MAX_TEMPERATURE)
}

/// Compute numerically stable sigmoid: 1 / (1 + exp(-x))
///
/// Uses the standard trick to avoid overflow:
/// - If x < 0: use 1 / (1 + exp(x))
/// - If x >= 0: use exp(-x) / (1 + exp(-x))
pub fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Compute numerically stable Gumbel-Softmax probability.
///
/// For a value x with bounds [min, max] and temperature τ:
/// P(min ≤ x ≤ max) = sigmoid((x - min) / τ) * sigmoid((max - x) / τ)
///
/// # Parameters
///
/// - `x`: Point coordinate
/// - `min`: Minimum bound
/// - `max`: Maximum bound
/// - `temp`: Temperature (will be clamped for stability)
pub fn gumbel_membership_prob(x: f32, min: f32, max: f32, temp: f32) -> f32 {
    let temp_safe = clamp_temperature_default(temp);
    
    // Avoid division by zero and handle edge cases
    if temp_safe < MIN_TEMPERATURE {
        // Hard bounds: return 1.0 if in bounds, 0.0 otherwise
        if x >= min && x <= max {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    
    // P(x > min) using stable sigmoid
    let min_prob = stable_sigmoid((x - min) / temp_safe);
    // P(x < max) using stable sigmoid
    let max_prob = stable_sigmoid((max - x) / temp_safe);
    
    min_prob * max_prob
}

/// Sample from Gumbel distribution with numerical stability.
///
/// G = -ln(-ln(U)) where U ~ Uniform(0, 1)
///
/// # Parameters
///
/// - `u`: Uniform random value in [0, 1)
/// - `epsilon`: Minimum value to avoid log(0) (default: 1e-7)
///
/// # Returns
///
/// Gumbel sample
pub fn sample_gumbel(u: f32, epsilon: f32) -> f32 {
    let u_clamped = u.clamp(epsilon, 1.0 - epsilon);
    -(-u_clamped.ln()).ln()
}

/// Map Gumbel sample to box bounds using temperature-scaled transformation.
///
/// This ensures samples are within bounds with probability proportional to volume.
///
/// # Parameters
///
/// - `gumbel`: Gumbel-distributed sample
/// - `min`: Minimum bound
/// - `max`: Maximum bound
/// - `temp`: Temperature parameter
pub fn map_gumbel_to_bounds(gumbel: f32, min: f32, max: f32, temp: f32) -> f32 {
    let temp_safe = clamp_temperature_default(temp);
    
    // Use tanh to map Gumbel to [-1, 1], then scale to [0, 1]
    let normalized = (gumbel / temp_safe).tanh();
    let t = (normalized + 1.0) / 2.0;
    
    min + (max - min) * t.clamp(0.0, 1.0)
}

/// Compute volume in log-space to avoid numerical underflow/overflow.
///
/// For high-dimensional boxes, direct multiplication can underflow to 0 or overflow to inf.
/// This function computes log(volume) = Σ log(max[i] - min[i]), which is numerically stable.
///
/// # Parameters
///
/// - `side_lengths`: Iterator over side lengths (max[i] - min[i]) for each dimension
///
/// # Returns
///
/// `(log_volume, volume)` where `volume = exp(log_volume)`.
/// If any side length is ≤ 0, returns `(f32::NEG_INFINITY, 0.0)`.
///
/// # Example
///
/// ```rust
/// use subsume_core::utils::log_space_volume;
///
/// let side_lengths = vec![1.0, 2.0, 0.5, 0.1];
/// let (log_vol, vol) = log_space_volume(side_lengths.iter().copied());
/// assert!((vol - 0.1).abs() < 1e-6); // 1.0 * 2.0 * 0.5 * 0.1 = 0.1
/// ```
pub fn log_space_volume<I>(side_lengths: I) -> (f32, f32)
where
    I: Iterator<Item = f32>,
{
    const EPSILON: f32 = 1e-10;
    
    let mut log_sum = 0.0;
    let mut has_zero = false;
    
    for side_len in side_lengths {
        if side_len <= EPSILON {
            has_zero = true;
            break;
        }
        log_sum += side_len.ln();
    }
    
    if has_zero {
        (f32::NEG_INFINITY, 0.0)
    } else {
        let volume = log_sum.exp();
        (log_sum, volume)
    }
}

/// Compute volume regularization penalty.
///
/// Prevents boxes from becoming arbitrarily large or small during training.
/// Returns a penalty term: λ * max(0, vol - threshold) for large boxes,
/// or λ * max(0, threshold_min - vol) for small boxes.
///
/// # Parameters
///
/// - `volume`: Current box volume
/// - `threshold_max`: Maximum allowed volume (penalty if exceeded)
/// - `threshold_min`: Minimum allowed volume (penalty if below)
/// - `lambda`: Regularization strength
///
/// # Returns
///
/// Regularization penalty (0.0 if volume is within [threshold_min, threshold_max])
///
/// # Example
///
/// ```rust
/// use subsume_core::utils::volume_regularization;
///
/// // Penalize boxes larger than 10.0
/// let penalty = volume_regularization(15.0, 10.0, 0.01, 0.1);
/// assert!(penalty > 0.0);
/// ```
pub fn volume_regularization(
    volume: f32,
    threshold_max: f32,
    threshold_min: f32,
    lambda: f32,
) -> f32 {
    let penalty_large = (volume - threshold_max).max(0.0);
    let penalty_small = (threshold_min - volume).max(0.0);
    lambda * (penalty_large + penalty_small)
}

/// Temperature scheduler for annealing during training.
///
/// Implements exponential decay: T(t) = T₀ * decay^t
/// This allows starting with high temperature (exploration) and decreasing
/// to low temperature (exploitation) over training steps.
///
/// # Parameters
///
/// - `initial_temp`: Starting temperature T₀
/// - `decay_rate`: Decay factor per step (typically 0.95-0.99)
/// - `step`: Current training step (0-indexed)
/// - `min_temp`: Minimum temperature to clamp to (default: MIN_TEMPERATURE)
///
/// # Returns
///
/// Annealed temperature value
///
/// # Example
///
/// ```rust
/// use subsume_core::utils::{temperature_scheduler, MIN_TEMPERATURE};
///
/// let temp_0 = temperature_scheduler(10.0, 0.95, 0, MIN_TEMPERATURE);
/// assert_eq!(temp_0, 10.0);
///
/// let temp_100 = temperature_scheduler(10.0, 0.95, 100, MIN_TEMPERATURE);
/// assert!(temp_100 < temp_0); // Temperature decreased
/// ```
pub fn temperature_scheduler(
    initial_temp: f32,
    decay_rate: f32,
    step: usize,
    min_temp: f32,
) -> f32 {
    let decayed = initial_temp * decay_rate.powi(step as i32);
    decayed.max(min_temp)
}

/// Volume-based containment loss for training.
///
/// Computes a loss that encourages high containment probability when boxes
/// should be in a containment relationship, and low probability otherwise.
///
/// # Parameters
///
/// - `containment_prob`: P(other ⊆ self) from box operations
/// - `target`: Target containment (1.0 for positive pairs, 0.0 for negative)
/// - `margin`: Margin for negative pairs (default: 0.1)
///
/// # Returns
///
/// Loss value (higher for incorrect predictions)
///
/// # Example
///
/// ```rust
/// use subsume_core::utils::volume_containment_loss;
///
/// // Positive pair: should have high containment
/// let loss_pos = volume_containment_loss(0.9, 1.0, 0.1);
/// assert!(loss_pos < 0.5); // Low loss for correct prediction
///
/// // Negative pair: should have low containment
/// let loss_neg = volume_containment_loss(0.8, 0.0, 0.1);
/// assert!(loss_neg > 0.5); // Higher loss for incorrect prediction
/// ```
pub fn volume_containment_loss(containment_prob: f32, target: f32, margin: f32) -> f32 {
    if target > 0.5 {
        // Positive pair: use negative log-likelihood
        // Clamp probability to avoid log(0)
        let prob_clamped = containment_prob.max(1e-10);
        -prob_clamped.ln()
    } else {
        // Negative pair: use margin-based loss
        (containment_prob - margin).max(0.0)
    }
}

/// Volume-based overlap loss for training.
///
/// Computes a loss that encourages appropriate overlap probabilities
/// based on whether boxes should overlap or be disjoint.
///
/// # Parameters
///
/// - `overlap_prob`: P(self ∩ other ≠ ∅) from box operations
/// - `target`: Target overlap (1.0 for overlapping pairs, 0.0 for disjoint)
/// - `margin`: Margin for negative pairs (default: 0.1)
///
/// # Returns
///
/// Loss value (higher for incorrect predictions)
pub fn volume_overlap_loss(overlap_prob: f32, target: f32, margin: f32) -> f32 {
    if target > 0.5 {
        // Should overlap: use negative log-likelihood
        // Clamp probability to avoid log(0)
        let prob_clamped = overlap_prob.max(1e-10);
        -prob_clamped.ln()
    } else {
        // Should be disjoint: use margin-based loss
        (overlap_prob - margin).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_temperature() {
        assert_eq!(clamp_temperature_default(0.0), MIN_TEMPERATURE);
        assert_eq!(clamp_temperature_default(100.0), MAX_TEMPERATURE);
        assert_eq!(clamp_temperature_default(1.0), 1.0);
    }

    #[test]
    fn test_stable_sigmoid() {
        // Test extreme values
        assert!(stable_sigmoid(-100.0) > 0.0);
        assert!(stable_sigmoid(100.0) <= 1.0); // Can be exactly 1.0 due to floating point
        assert!(stable_sigmoid(100.0) > 0.99); // Should be very close to 1.0
        assert!((stable_sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gumbel_membership_prob() {
        // Point inside bounds
        let prob = gumbel_membership_prob(0.5, 0.0, 1.0, 1.0);
        assert!(prob > 0.0 && prob <= 1.0);
        
        // Point outside bounds
        let prob_out = gumbel_membership_prob(2.0, 0.0, 1.0, 0.001);
        assert!(prob_out < 0.5);
        
        // Hard bounds (low temperature)
        let prob_hard = gumbel_membership_prob(0.5, 0.0, 1.0, 0.0001);
        assert!((prob_hard - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_sample_gumbel() {
        let g = sample_gumbel(0.5, 1e-7);
        assert!(g.is_finite());
        
        // Test edge cases
        let g0 = sample_gumbel(0.0, 1e-7);
        assert!(g0.is_finite());
        
        let g1 = sample_gumbel(1.0, 1e-7);
        assert!(g1.is_finite());
    }

    #[test]
    fn test_map_gumbel_to_bounds() {
        let value = map_gumbel_to_bounds(0.0, 0.0, 1.0, 1.0);
        assert!(value >= 0.0 && value <= 1.0);
    }

    #[test]
    fn test_log_space_volume() {
        // Simple case
        let side_lengths = vec![1.0, 2.0, 0.5];
        let (log_vol, vol) = log_space_volume(side_lengths.iter().copied());
        assert!((vol - 1.0).abs() < 1e-6); // 1.0 * 2.0 * 0.5 = 1.0
        assert!((log_vol - 0.0).abs() < 1e-6); // ln(1.0) = 0.0

        // High-dimensional case (would underflow with direct multiplication)
        let many_small = vec![0.1; 20];
        let (log_vol_hd, vol_hd) = log_space_volume(many_small.iter().copied());
        assert!(log_vol_hd.is_finite());
        assert!(vol_hd > 0.0);
        assert!(vol_hd < 1.0);

        // Zero volume case
        let with_zero = vec![1.0, 0.0, 2.0];
        let (log_vol_zero, vol_zero) = log_space_volume(with_zero.iter().copied());
        assert_eq!(log_vol_zero, f32::NEG_INFINITY);
        assert_eq!(vol_zero, 0.0);
    }

    #[test]
    fn test_volume_regularization() {
        // Volume within bounds
        let penalty = volume_regularization(5.0, 10.0, 0.01, 0.1);
        assert_eq!(penalty, 0.0);

        // Volume too large
        let penalty_large = volume_regularization(15.0, 10.0, 0.01, 0.1);
        assert!((penalty_large - 0.5).abs() < 1e-6); // 0.1 * (15.0 - 10.0) = 0.5

        // Volume too small
        let penalty_small = volume_regularization(0.001, 10.0, 0.01, 0.1);
        assert!((penalty_small - 0.0009).abs() < 1e-6); // 0.1 * (0.01 - 0.001) = 0.0009
    }

    #[test]
    fn test_temperature_scheduler() {
        // Initial step
        let temp_0 = temperature_scheduler(10.0, 0.95, 0, MIN_TEMPERATURE);
        assert_eq!(temp_0, 10.0);

        // After decay
        let temp_10 = temperature_scheduler(10.0, 0.95, 10, MIN_TEMPERATURE);
        assert!(temp_10 < temp_0);
        assert!(temp_10 >= MIN_TEMPERATURE);

        // Clamped to minimum
        let temp_1000 = temperature_scheduler(10.0, 0.95, 1000, MIN_TEMPERATURE);
        assert_eq!(temp_1000, MIN_TEMPERATURE);
    }

    #[test]
    fn test_volume_containment_loss() {
        // Positive pair with high containment (low loss)
        let loss_pos_good = volume_containment_loss(0.9, 1.0, 0.1);
        assert!(loss_pos_good < 0.2);

        // Positive pair with low containment (high loss)
        let loss_pos_bad = volume_containment_loss(0.1, 1.0, 0.1);
        assert!(loss_pos_bad > 1.0);

        // Negative pair with low containment (low loss)
        let loss_neg_good = volume_containment_loss(0.05, 0.0, 0.1);
        assert_eq!(loss_neg_good, 0.0);

        // Negative pair with high containment (high loss)
        let loss_neg_bad = volume_containment_loss(0.8, 0.0, 0.1);
        assert!((loss_neg_bad - 0.7).abs() < 1e-6); // 0.8 - 0.1 = 0.7
    }

    #[test]
    fn test_volume_overlap_loss() {
        // Overlapping pair with high overlap (low loss)
        let loss_overlap_good = volume_overlap_loss(0.9, 1.0, 0.1);
        assert!(loss_overlap_good < 0.2);

        // Overlapping pair with low overlap (high loss)
        let loss_overlap_bad = volume_overlap_loss(0.1, 1.0, 0.1);
        assert!(loss_overlap_bad > 1.0);

        // Disjoint pair with low overlap (low loss)
        let loss_disjoint_good = volume_overlap_loss(0.05, 0.0, 0.1);
        assert_eq!(loss_disjoint_good, 0.0);

        // Disjoint pair with high overlap (high loss)
        let loss_disjoint_bad = volume_overlap_loss(0.8, 0.0, 0.1);
        assert!((loss_disjoint_bad - 0.7).abs() < 1e-6); // 0.8 - 0.1 = 0.7
    }
}

