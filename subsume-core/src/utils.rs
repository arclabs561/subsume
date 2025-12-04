//! Numerical stability utilities for box embeddings.

/// Clamp temperature to a safe range to avoid numerical instability.
///
/// Research shows that very low temperatures cause vanishing gradients and
/// exponential underflow, while very high temperatures lose correspondence
/// to discrete distributions.
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
    
    // Clamp to ensure we stay within bounds (defensive programming)
    min + (max - min) * t.clamp(0.0, 1.0)
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
}

