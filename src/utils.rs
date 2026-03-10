//! Numerical stability utilities for box embeddings.
//!
//! # Mathematical Background
//!
//! This module provides numerically stable implementations of mathematical operations
//! used in box embeddings. Key concepts:
//!
//! ## Log-Space Computations
//!
//! For high-dimensional volumes, direct multiplication can underflow. The solution is
//! to compute in log-space:
//!
//! \[
//! \log(\text{Vol}) = \sum_{i=1}^{d} \log(\text{side}_i), \quad \text{Vol} = \exp\left(\sum_{i=1}^{d} \log(\text{side}_i)\right)
//! \]
//!
//! ## Stable Sigmoid
//!
//! The sigmoid function `σ(x) = 1/(1 + e^(-x))` can overflow when `x` is large
//! and negative. The stable form uses:
//!
//! - If `x < 0`: use `e^x / (1 + e^x)`
//! - If `x ≥ 0`: use `1 / (1 + e^(-x))`
//!
//! ## Log-Sum-Exp Pattern
//!
//! Many operations use the log-sum-exp pattern for numerical stability:
//!
//! \[
//! \text{lse}_\beta(x, y) = \max(x, y) + \beta \log(1 + e^{-|x-y|/\beta})
//! \]
//!
//! This pattern appears in Gumbel intersections and other operations.
//!
//! See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../docs/MATHEMATICAL_FOUNDATIONS.md)
//! for complete mathematical derivations and [`docs/MATH_TO_CODE_CONNECTIONS.md`](../../docs/MATH_TO_CODE_CONNECTIONS.md)
//! for how these patterns are used in the codebase.
//!
//! **For detailed study:** PDF versions with professional typesetting are available:
//! - [`docs/typst-output/pdf/log-sum-exp-intersection.pdf`](../../docs/typst-output/pdf/log-sum-exp-intersection.pdf) - Log-sum-exp function and numerical stability
//! - [`docs/typst-output/pdf/gumbel-box-volume.pdf`](../../docs/typst-output/pdf/gumbel-box-volume.pdf) - Volume calculations and numerical considerations

/// Euler-Mascheroni constant (gamma ~ 0.5772).
///
/// Appears in the Bessel approximation for Gumbel box expected volume
/// (Dasgupta et al., 2020): the offset `2 * gamma * beta` accounts for
/// the mean of the Gumbel distribution.
pub const EULER_GAMMA: f32 = 0.577_215_7;

/// Numerically stable softplus: `(1/beta) * log(1 + exp(beta * x))`.
///
/// For large `beta * x`, returns `x` directly to avoid overflow.
/// When `beta = 1.0`, this is the standard `log(1 + exp(x))`.
///
/// # Parameters
///
/// - `x`: Input value
/// - `beta`: Steepness parameter (reciprocal of the temperature)
pub fn softplus(x: f32, beta: f32) -> f32 {
    let bx = beta * x;
    if bx > 20.0 {
        x // linear regime: softplus(x) ~ x for large x
    } else if bx < -20.0 {
        0.0 // exponentially small
    } else {
        bx.exp().ln_1p() / beta
    }
}

/// Numerically stable log-sum-exp of two values: `log(exp(a) + exp(b))`.
///
/// Uses the identity `lse(a, b) = max(a, b) + log(1 + exp(-|a - b|))`
/// to avoid overflow.
pub fn stable_logsumexp(a: f32, b: f32) -> f32 {
    let m = a.max(b);
    if m == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Gumbel LSE intersection: smooth approximation to `max(z_a, z_b)`.
///
/// Computes `T * logsumexp(z_a / T, z_b / T)`, which is a smooth upper
/// bound on `max(z_a, z_b)` that approaches the hard max as `T -> 0`.
///
/// Used for the **minimum** coordinates of the Gumbel intersection box.
pub fn gumbel_lse_min(z_a: f32, z_b: f32, temperature: f32) -> f32 {
    temperature * stable_logsumexp(z_a / temperature, z_b / temperature)
}

/// Gumbel LSE intersection: smooth approximation to `min(Z_a, Z_b)`.
///
/// Computes `-T * logsumexp(-Z_a / T, -Z_b / T)`, which is a smooth lower
/// bound on `min(Z_a, Z_b)` that approaches the hard min as `T -> 0`.
///
/// Used for the **maximum** coordinates of the Gumbel intersection box.
pub fn gumbel_lse_max(z_a: f32, z_b: f32, temperature: f32) -> f32 {
    -gumbel_lse_min(-z_a, -z_b, temperature)
}

/// Bessel-approximation volume for one dimension of a Gumbel box.
///
/// Per Dasgupta et al. (2020), the expected side length of a Gumbel box
/// in one dimension is approximated by:
///
/// ```text
/// softplus(Z - z - 2*gamma*T_int, beta = 1/T_vol)
/// ```
///
/// where `gamma` is the Euler-Mascheroni constant, `T_int` is the
/// intersection temperature (Gumbel scale), and `T_vol` controls the
/// softplus steepness.
///
/// Returns the per-dimension side length (not log-scale).
pub fn bessel_side_length(z: f32, big_z: f32, t_int: f32, t_vol: f32) -> f32 {
    let arg = big_z - z - 2.0 * EULER_GAMMA * t_int;
    softplus(arg, 1.0 / t_vol)
}

/// Bessel-approximation log-volume for a Gumbel box (Dasgupta et al., 2020).
///
/// Computes `sum_d log(softplus(Z_d - z_d - 2*gamma*T_int, beta=1/T_vol) + eps)`
/// over all dimensions. The `eps` prevents `log(0)` for near-empty boxes.
///
/// Returns `(log_volume, volume)`.
pub fn bessel_log_volume(mins: &[f32], maxs: &[f32], t_int: f32, t_vol: f32) -> (f32, f32) {
    const EPS: f32 = 1e-13;
    let log_vol: f32 = mins
        .iter()
        .zip(maxs.iter())
        .map(|(&z, &big_z)| (bessel_side_length(z, big_z, t_int, t_vol) + EPS).ln())
        .sum();
    (log_vol, log_vol.exp())
}

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
/// This computes the probability that a point lies within box bounds using the
/// Gumbel-Softmax framework.
///
/// ## Mathematical Formulation
///
/// For a value \(x\) with bounds \([min, max]\) and temperature \(\tau\):
///
/// \[
/// P(min \leq x \leq max) = \sigma\left(\frac{x - min}{\tau}\right) \cdot \sigma\left(\frac{max - x}{\tau}\right)
/// \]
///
/// where \(\sigma\) is the sigmoid function. This is the product of:
/// - \(P(x > min)\): Probability that point is above minimum bound
/// - \(P(x < max)\): Probability that point is below maximum bound
///
/// **Derivation**: For Gumbel-distributed box boundaries, the probability that a point lies
/// within the box is the product of the probabilities that it's above the minimum and below
/// the maximum. Each probability is computed using the sigmoid function, which provides
/// smooth gradients for training.
///
/// ## Temperature Behavior
///
/// - **\(\tau \to 0\)**: Approaches hard bounds (0 or 1) - deterministic membership
/// - **\(\tau \to \infty\)**: Approaches uniform probability - smooth, continuous
///
/// The temperature parameter controls the trade-off between:
/// - **Low \(\tau\)**: Sharp boundaries, better correspondence to discrete logic
/// - **High \(\tau\)**: Smooth boundaries, better gradients for optimization
///
/// ## Numerical Stability
///
/// Uses [`stable_sigmoid`] to avoid overflow when
/// \(|x - min|/\tau\) or \(|max - x|/\tau\) is large.
///
/// See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../docs/MATHEMATICAL_FOUNDATIONS.md)
/// section "Gumbel-Softmax Framework" for more details.
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
/// The Gumbel distribution is used in the Gumbel-max trick for differentiable sampling.
/// For a uniform random variable \(U \sim \text{Uniform}(0, 1)\), the Gumbel sample is:
///
/// \[
/// G = -\ln(-\ln(U))
/// \]
///
/// This produces \(G \sim \text{Gumbel}(0, 1)\) (standard Gumbel distribution).
///
/// ## Max-Stability Property
///
/// **Why this matters**: The Gumbel distribution is **max-stable**, meaning the maximum
/// of independent Gumbel random variables is itself Gumbel-distributed:
///
/// If \(G_1, \ldots, G_k \sim \text{Gumbel}(\mu, \beta)\) are independent, then:
///
/// \[
/// \max\{G_1, \ldots, G_k\} \sim \text{Gumbel}(\mu + \beta \ln k, \beta)
/// \]
///
/// **Proof sketch**: The CDF of the maximum is the product of individual CDFs:
///
/// \[
/// P(\max\{G_1, \ldots, G_k\} \leq x) = [e^{-e^{-(x-\mu)/\beta}}]^k = e^{-k e^{-(x-\mu)/\beta}}
/// \]
///
/// After algebraic manipulation, this equals \(e^{-e^{-(x-(\mu+\beta\ln k))/\beta}}\), which is
/// the CDF of \(\text{Gumbel}(\mu + \beta \ln k, \beta)\).
///
/// This property is crucial for maintaining the algebraic structure of box embeddings
/// when computing intersections (max of minimums, min of maximums). It ensures that
/// intersection operations preserve the Gumbel distribution family, enabling analytical
/// volume calculations.
///
/// See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../docs/MATHEMATICAL_FOUNDATIONS.md)
/// section "Min-Max Stability" for the complete proof.
///
/// # Numerical Stability
///
/// This function clamps U to [ε, 1-ε] to avoid:
/// - `ln(0)` when U ≈ 0 (would give -∞)
/// - `ln(1)` when U ≈ 1 (would give 0, then -ln(0) = ∞)
///
/// # Parameters
///
/// - `u`: Uniform random value in [0, 1)
/// - `epsilon`: Minimum value to avoid log(0) (default: 1e-7)
///
/// # Returns
///
/// Gumbel sample G ~ Gumbel(0, 1)
///
/// # Mathematical Foundation
///
/// This implements the inverse CDF method for sampling from a Gumbel distribution:
/// G = -ln(-ln(U)) where U ~ Uniform(0,1). The double logarithm structure is
/// fundamental to Gumbel distributions and enables max-stability: the maximum of
/// independent Gumbel random variables is itself Gumbel-distributed.
///
/// See [`docs/typst-output/pdf/gumbel-max-stability.pdf`](../../docs/typst-output/pdf/gumbel-max-stability.pdf)
/// for the complete derivation of max-stability and why it's crucial for box embeddings.
pub fn sample_gumbel(u: f32, epsilon: f32) -> f32 {
    let u_clamped = u.clamp(epsilon, 1.0 - epsilon);
    -(-u_clamped.ln()).ln()
}

/// Map Gumbel sample to box bounds using temperature-scaled transformation.
///
/// This function transforms a Gumbel-distributed sample into a point within the box bounds.
/// The temperature parameter controls how "concentrated" the sampling is:
///
/// - **Low temperature**: Samples cluster near the center (more deterministic)
/// - **High temperature**: Samples spread throughout the box (more uniform)
///
/// **Why tanh?**: The tanh function maps the unbounded Gumbel distribution to [-1, 1],
/// which is then scaled to [0, 1] and mapped to [min, max]. This ensures:
/// 1. All samples are within bounds
/// 2. The distribution is smooth and differentiable
/// 3. Temperature controls the concentration
///
/// # Parameters
///
/// - `gumbel`: Gumbel-distributed sample (from `sample_gumbel()`)
/// - `min`: Minimum bound of the box
/// - `max`: Maximum bound of the box
/// - `temp`: Temperature parameter (controls concentration, clamped to safe range)
///
/// # Returns
///
/// A point in [min, max] sampled according to the Gumbel distribution scaled by temperature
pub fn map_gumbel_to_bounds(gumbel: f32, min: f32, max: f32, temp: f32) -> f32 {
    let temp_safe = clamp_temperature_default(temp);

    // Use tanh to map Gumbel to [-1, 1], then scale to [0, 1]
    let normalized = (gumbel / temp_safe).tanh();
    let t = (normalized + 1.0) / 2.0;

    min + (max - min) * t.clamp(0.0, 1.0)
}

/// Compute volume in log-space to avoid numerical underflow/overflow.
///
/// ## The Problem
///
/// For high-dimensional boxes, direct volume computation can fail:
///
/// ```rust,ignore
/// // This can underflow in high dimensions!
/// let side_lengths = vec![0.5; 20]; // 20 dimensions
/// let volume = side_lengths.iter().product::<f32>();
/// // For 20 dimensions with side length 0.5: 0.5^20 ≈ 9.5×10^-7
/// // This can underflow to 0.0 in f32!
/// ```
///
/// ## The Solution
///
/// Compute volume in log-space, then exponentiate:
///
/// ```text
/// log(Vol) = Σᵢ log(side[i])
/// ```
///
/// ```text
/// Vol = exp(Σᵢ log(side[i]))
/// ```
///
/// This is numerically stable because:
/// - Log of small numbers is a large negative number (no underflow)
/// - Sum of logs is stable (no intermediate products)
/// - Exp of the sum recovers the volume
///
/// ## When to Use
///
/// Use this function when:
/// - Dimension > 10 (high-dimensional boxes)
/// - Side lengths < 1.0 (small boxes in unit hypercube)
/// - Computing many volumes in a loop (accumulated numerical error)
///
/// # Parameters
///
/// - `side_lengths`: Iterator over side lengths (max\[i\] - min\[i\]) for each dimension
///
/// # Returns
///
/// `(log_volume, volume)` where `volume = exp(log_volume)`.
/// If any side length is ≤ 0, returns `(f32::NEG_INFINITY, 0.0)`.
///
/// # Example
///
/// ```rust
/// use subsume::utils::log_space_volume;
///
/// // Low-dimensional: both methods work
/// let side_lengths = vec![1.0, 2.0, 0.5, 0.1];
/// let (log_vol, vol) = log_space_volume(side_lengths.iter().copied());
/// assert!((vol - 0.1).abs() < 1e-6); // 1.0 * 2.0 * 0.5 * 0.1 = 0.1
///
/// // High-dimensional: log-space is essential
/// let many_small = vec![0.5; 20];
/// let (log_vol_hd, vol_hd) = log_space_volume(many_small.iter().copied());
/// assert!(vol_hd > 0.0); // Would underflow with direct multiplication!
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
/// use subsume::utils::volume_regularization;
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
/// use subsume::utils::{temperature_scheduler, MIN_TEMPERATURE};
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
/// # Paradigm Problem: Training on Containment Relationships
///
/// **The problem**: Given positive pairs (A should contain B) and negative pairs (A should NOT
/// contain B), design a loss function that encourages correct containment while discouraging
/// incorrect containment.
///
/// **Step-by-step reasoning**:
///
/// 1. **For positive pairs**: We want high containment probability P(B|A) → 1.0
///    - Use negative log-likelihood: `-log(P(B|A))`
///    - When P(B|A) = 1.0, loss = 0 (perfect)
///    - When P(B|A) = 0.5, loss = log(2) ≈ 0.69 (penalty)
///    - This provides strong gradient signal when containment is not perfect
///
/// 2. **For negative pairs**: We want low containment probability P(B|A) → 0.0
///    - Use margin-based loss: `max(0, margin - (-log(P(B|A))))`
///    - When P(B|A) is very small, loss = 0 (good enough)
///    - When P(B|A) exceeds threshold, loss > 0 (penalty)
///    - This creates a "safety zone"—as long as containment is below threshold, no penalty
///
/// 3. **Why different losses?**: Positive pairs need to be maximized (log-likelihood provides
///    strong gradient), while negative pairs just need to stay below a threshold (margin loss
///    is sufficient and computationally cheaper).
///
/// # Research Background
///
/// This loss function combines negative log-likelihood (for positive pairs) with margin-based
/// loss (for negative pairs), following practices from knowledge graph embedding literature.
///
/// **Research foundation**:
/// - **Vilnis et al. (2018)**: "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
///   - Establishes volume as probability measure, uses log-likelihood for positive pairs
/// - **Abboud et al. (2020)**: "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS)
///   - Uses margin-based ranking loss for both positive and negative pairs
/// - **Bordes et al. (2013)**: "Translating Embeddings for Modeling Multi-relational Data"
///   - Introduces margin-based ranking loss for knowledge graphs, establishes standard training protocol
///
/// Compute safe initialization bounds for box embeddings.
///
/// Returns suggested min and max bounds for initializing boxes to avoid
/// local identifiability problems. Boxes initialized with these bounds will
/// be well-separated and avoid problematic geometric configurations.
///
/// # Parameters
///
/// - `dimension`: Dimension index (0-indexed)
/// - `num_boxes`: Total number of boxes being initialized
/// - `box_index`: Index of the current box (0-indexed)
/// - `center_range`: Range for box centers (default: [-2.0, 2.0])
/// - `size_range`: Range for box sizes (default: [0.1, 1.0])
///
/// # Returns
///
/// `(min_bound, max_bound)` for the specified dimension
///
/// # Example
///
/// ```rust
/// use subsume::utils::safe_init_bounds;
///
/// // Initialize 10 boxes in 3D space
/// for box_idx in 0..10 {
///     for dim in 0..3 {
///         let (min, max) = safe_init_bounds(dim, 10, box_idx, (-2.0, 2.0), (0.1, 1.0));
///         // Use min and max to create box bounds
///     }
/// }
/// ```
pub fn safe_init_bounds(
    dimension: usize,
    num_boxes: usize,
    box_index: usize,
    center_range: (f32, f32),
    size_range: (f32, f32),
) -> (f32, f32) {
    let (center_min, center_max) = center_range;
    let (size_min, size_max) = size_range;

    // Use different strategies per dimension to avoid cross patterns
    let dim_offset = dimension as f32 * 0.5;
    let box_offset = (box_index as f32) / (num_boxes as f32);

    // Create well-separated centers using a combination of dimension and box index
    let center = center_min + (center_max - center_min) * ((box_offset + dim_offset * 0.3) % 1.0);

    // Vary box sizes to avoid perfect nesting
    let size = size_min + (size_max - size_min) * (0.5 + 0.3 * (box_index as f32 * 1.618).sin()); // Golden ratio for variety

    let half_size = size / 2.0;
    (center - half_size, center + half_size)
}

/// Check if two boxes form a problematic cross pattern.
///
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
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn test_log_space_volume() {
        // Simple case
        let side_lengths = [1.0, 2.0, 0.5];
        let (log_vol, vol) = log_space_volume(side_lengths.iter().copied());
        assert!((vol - 1.0).abs() < 1e-6); // 1.0 * 2.0 * 0.5 = 1.0
        assert!((log_vol - 0.0).abs() < 1e-6); // ln(1.0) = 0.0

        // High-dimensional case (would underflow with direct multiplication)
        let many_small = [0.1; 20];
        let (log_vol_hd, vol_hd) = log_space_volume(many_small.iter().copied());
        assert!(log_vol_hd.is_finite());
        assert!(vol_hd > 0.0);
        assert!(vol_hd < 1.0);

        // Zero volume case
        let with_zero = [1.0, 0.0, 2.0];
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
    fn test_safe_init_bounds() {
        // Test that bounds are valid (min < max)
        let (min, max) = safe_init_bounds(0, 10, 0, (-2.0, 2.0), (0.1, 1.0));
        assert!(min < max);
        assert!(max - min >= 0.1); // At least minimum size

        // Test that different boxes get different bounds
        let (min1, max1) = safe_init_bounds(0, 10, 0, (-2.0, 2.0), (0.1, 1.0));
        let (min2, max2) = safe_init_bounds(0, 10, 1, (-2.0, 2.0), (0.1, 1.0));
        // They should be different (not identical)
        assert!(min1 != min2 || max1 != max2);

        // Test that different dimensions get different bounds
        let (min_dim0, max_dim0) = safe_init_bounds(0, 10, 5, (-2.0, 2.0), (0.1, 1.0));
        let (min_dim1, max_dim1) = safe_init_bounds(1, 10, 5, (-2.0, 2.0), (0.1, 1.0));
        // They should be different
        assert!(min_dim0 != min_dim1 || max_dim0 != max_dim1);
    }

    // ---- softplus ----

    #[test]
    fn test_softplus_basic() {
        // softplus(0, 1) = ln(2) ~ 0.693
        assert!((softplus(0.0, 1.0) - 0.693).abs() < 0.01);
        // softplus(x, 1) ~ x for large x
        assert!((softplus(100.0, 1.0) - 100.0).abs() < 0.01);
        // softplus(x, 1) ~ 0 for very negative x
        assert!(softplus(-100.0, 1.0) < 1e-10);
        // softplus is always non-negative
        assert!(softplus(-5.0, 1.0) >= 0.0);
        assert!(softplus(0.0, 1.0) >= 0.0);
        assert!(softplus(5.0, 1.0) >= 0.0);
    }

    #[test]
    fn test_softplus_with_beta() {
        // Higher beta = sharper transition
        let sharp = softplus(0.5, 10.0);
        let soft = softplus(0.5, 0.1);
        // Both should be positive
        assert!(sharp > 0.0);
        assert!(soft > 0.0);
        // softplus(x, beta) = (1/beta) * ln(1 + exp(beta*x))
        let expected = (1.0_f32 / 2.0) * (1.0 + (2.0 * 3.0_f32).exp()).ln();
        assert!((softplus(3.0, 2.0) - expected).abs() < 0.01);
    }

    // ---- stable_logsumexp ----

    #[test]
    fn test_stable_logsumexp() {
        // lse(0, 0) = ln(2) ~ 0.693
        assert!((stable_logsumexp(0.0, 0.0) - 0.693).abs() < 0.01);
        // lse(a, -inf) = a
        assert!((stable_logsumexp(5.0, f32::NEG_INFINITY) - 5.0).abs() < 1e-6);
        // lse(a, b) >= max(a, b)
        assert!(stable_logsumexp(3.0, 5.0) >= 5.0);
        assert!(stable_logsumexp(-1.0, -3.0) >= -1.0);
        // symmetric
        assert!((stable_logsumexp(2.0, 7.0) - stable_logsumexp(7.0, 2.0)).abs() < 1e-6);
        // large values don't overflow
        assert!(stable_logsumexp(100.0, 100.0).is_finite());
        assert!(stable_logsumexp(-100.0, -100.0).is_finite());
    }

    // ---- gumbel_lse_min / gumbel_lse_max ----

    #[test]
    fn test_gumbel_lse_min_is_smooth_max() {
        // gumbel_lse_min is the smooth max: should be >= max(a, b)
        assert!(gumbel_lse_min(1.0, 3.0, 1.0) >= 3.0);
        assert!(gumbel_lse_min(5.0, 2.0, 1.0) >= 5.0);
        // Approaches hard max at low T
        let hard_approx = gumbel_lse_min(1.0, 3.0, 0.01);
        assert!((hard_approx - 3.0).abs() < 0.05, "got {hard_approx}");
    }

    #[test]
    fn test_gumbel_lse_max_is_smooth_min() {
        // gumbel_lse_max is the smooth min: should be <= min(a, b)
        assert!(gumbel_lse_max(5.0, 3.0, 1.0) <= 3.0);
        assert!(gumbel_lse_max(2.0, 7.0, 1.0) <= 2.0);
        // Approaches hard min at low T
        let hard_approx = gumbel_lse_max(5.0, 3.0, 0.01);
        assert!((hard_approx - 3.0).abs() < 0.05, "got {hard_approx}");
    }

    #[test]
    fn test_gumbel_lse_identity() {
        // gumbel_lse_max(a, b, T) = -gumbel_lse_min(-a, -b, T)
        let a = 3.0;
        let b = 7.0;
        let t = 1.5;
        let via_max = gumbel_lse_max(a, b, t);
        let via_min = -gumbel_lse_min(-a, -b, t);
        assert!((via_max - via_min).abs() < 1e-5, "{via_max} vs {via_min}");
    }

    // ---- bessel_side_length / bessel_log_volume ----

    #[test]
    fn test_bessel_side_length_basic() {
        // For large side length (Z - z >> 2*gamma*T), should be close to Z - z
        let sl = bessel_side_length(0.0, 10.0, 0.01, 0.01);
        assert!(
            (sl - 10.0).abs() < 0.1,
            "large box at low T should have sl ~ 10, got {sl}"
        );

        // For zero side length, softplus gives non-zero (smooth)
        let sl_zero = bessel_side_length(5.0, 5.0, 1.0, 1.0);
        assert!(
            sl_zero > 0.0,
            "zero hard side should have positive Bessel side, got {sl_zero}"
        );
    }

    #[test]
    fn test_bessel_log_volume_basic() {
        let (log_v, v) = bessel_log_volume(&[0.0, 0.0], &[5.0, 5.0], 0.01, 0.01);
        assert!(v > 0.0, "volume should be positive");
        assert!(log_v.is_finite(), "log volume should be finite");
        // At low T, should be close to 5*5 = 25
        assert!(
            (v - 25.0).abs() < 2.0,
            "at low T, vol should be ~25, got {v}"
        );
    }

    #[test]
    fn test_euler_gamma_value() {
        assert!((EULER_GAMMA - 0.5772).abs() < 0.001);
    }

    // =========================================================================
    // Property tests
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        // ---- softplus monotonicity: a <= b => softplus(a, beta) <= softplus(b, beta) ----
        proptest! {
            #[test]
            fn prop_softplus_monotone(
                a in -50.0f32..50.0f32,
                delta in 0.0f32..50.0f32,
                beta in 0.01f32..10.0f32,
            ) {
                let b = a + delta;
                let sa = softplus(a, beta);
                let sb = softplus(b, beta);
                prop_assert!(
                    sb >= sa - 1e-6,
                    "softplus({a}, {beta})={sa} > softplus({b}, {beta})={sb}, violates monotonicity"
                );
            }
        }

        // ---- stable_logsumexp >= max(a, b) ----
        proptest! {
            #[test]
            fn prop_logsumexp_ge_max(
                a in -100.0f32..100.0f32,
                b in -100.0f32..100.0f32,
            ) {
                let lse = stable_logsumexp(a, b);
                let m = a.max(b);
                prop_assert!(
                    lse >= m - 1e-6,
                    "stable_logsumexp({a}, {b})={lse} < max={m}"
                );
            }
        }
    }
}
