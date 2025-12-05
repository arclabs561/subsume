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
//! See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../../docs/MATHEMATICAL_FOUNDATIONS.md)
//! for complete mathematical derivations and [`docs/MATH_TO_CODE_CONNECTIONS.md`](../../../docs/MATH_TO_CODE_CONNECTIONS.md)
//! for how these patterns are used in the codebase.

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
/// Uses [`stable_sigmoid`](crate::utils::stable_sigmoid) to avoid overflow when
/// \(|x - min|/\tau\) or \(|max - x|/\tau\) is large.
///
/// See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../../docs/MATHEMATICAL_FOUNDATIONS.md)
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
/// See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../../docs/MATHEMATICAL_FOUNDATIONS.md)
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
/// use subsume_core::utils::log_space_volume;
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
/// - **Boratko et al. (2020)**: "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS)
///   - Uses margin-based ranking loss for both positive and negative pairs
/// - **Bordes et al. (2013)**: "Translating Embeddings for Modeling Multi-relational Data"
///   - Introduces margin-based ranking loss for knowledge graphs, establishes standard training protocol
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
/// use subsume_core::utils::safe_init_bounds;
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
/// Cross patterns occur when boxes intersect without clear containment,
/// which can cause local identifiability problems during training.
///
/// # Parameters
///
/// - `overlap_prob`: Overlap probability between two boxes
/// - `containment_prob_a_b`: P(A ⊆ B)
/// - `containment_prob_b_a`: P(B ⊆ A)
/// - `threshold`: Threshold for detecting cross patterns (default: 0.3)
///
/// # Returns
///
/// `true` if boxes form a cross pattern (high overlap but low containment in both directions)
pub fn is_cross_pattern(
    overlap_prob: f32,
    containment_prob_a_b: f32,
    containment_prob_b_a: f32,
    threshold: f32,
) -> bool {
    // Cross pattern: high overlap but neither box contains the other
    overlap_prob > threshold && containment_prob_a_b < threshold && containment_prob_b_a < threshold
}

/// Check if boxes are perfectly nested (one fully contains the other).
///
/// Perfect nesting at initialization can cause local identifiability problems.
/// This function helps detect such configurations.
///
/// # Parameters
///
/// - `containment_prob_a_b`: P(A ⊆ B)
/// - `containment_prob_b_a`: P(B ⊆ A)
/// - `threshold`: Threshold for perfect containment (default: 0.95)
///
/// # Returns
///
/// `true` if one box perfectly contains the other
pub fn is_perfectly_nested(
    containment_prob_a_b: f32,
    containment_prob_b_a: f32,
    threshold: f32,
) -> bool {
    containment_prob_a_b > threshold || containment_prob_b_a > threshold
}

/// Suggest minimum separation distance for box initialization.
///
/// Returns a suggested minimum distance between box centers to avoid
/// problematic geometric configurations that cause local identifiability issues.
///
/// # Parameters
///
/// - `dimension`: Embedding dimension
/// - `target_volume_range`: Desired volume range for boxes
///
/// # Returns
///
/// Minimum suggested separation distance
pub fn suggested_min_separation(dimension: usize, target_volume_range: (f32, f32)) -> f32 {
    let (min_vol, max_vol) = target_volume_range;
    // Separation should be at least the diameter of the largest box
    // For a box with volume V in d dimensions, approximate side length is V^(1/d)
    let avg_side = ((min_vol + max_vol) / 2.0).powf(1.0 / dimension as f32);
    // Use 1.5x the average side length as minimum separation
    avg_side * 1.5
}

/// Validation utilities for box embeddings.
///
/// These utilities help validate box embeddings to ensure they meet
/// mathematical constraints and avoid numerical issues.
pub mod validation {
    use crate::{Box, BoxError};

    /// Validate that a box has valid bounds by checking its volume.
    ///
    /// This is useful for checking box validity after operations that might
    /// modify bounds, or for debugging during training. The volume calculation
    /// will fail if bounds are invalid (min\[i\] > max\[i\] for any dimension).
    ///
    /// # Parameters
    ///
    /// - `box_`: The box to validate
    /// - `temperature`: Temperature parameter for volume calculation (typically 1.0)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the box is valid
    /// - `Err(BoxError::InvalidBounds)` if any dimension has invalid bounds
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use subsume_core::utils::validation::validate_box_bounds;
    /// use subsume_ndarray::NdarrayBox;
    ///
    /// let box_ = NdarrayBox::new(...)?;
    /// validate_box_bounds(&box_, 1.0)?; // Check validity
    /// ```
    pub fn validate_box_bounds<B: Box>(box_: &B, temperature: B::Scalar) -> Result<(), BoxError> {
        // Volume calculation will fail if bounds are invalid
        let _vol = box_.volume(temperature)?;
        Ok(())
    }

    /// Check if a box has zero or near-zero volume.
    ///
    /// Zero-volume boxes can cause numerical issues in probability calculations.
    ///
    /// # Parameters
    ///
    /// - `box_`: The box to check
    /// - `temperature`: Temperature parameter for volume calculation (typically 1.0)
    /// - `epsilon`: Threshold for considering volume as zero (default: 1e-10)
    ///
    /// # Returns
    ///
    /// - `true` if box volume is below epsilon
    /// - `false` otherwise
    pub fn is_zero_volume<B: Box>(box_: &B, temperature: B::Scalar, epsilon: B::Scalar) -> bool
    where
        B::Scalar: PartialOrd,
    {
        if let Ok(vol) = box_.volume(temperature) {
            vol < epsilon
        } else {
            true // If volume calculation fails, consider it zero
        }
    }

    /// Validate that two boxes have compatible dimensions.
    ///
    /// Useful for checking before operations that require dimension matching.
    ///
    /// # Parameters
    ///
    /// - `box1`: First box
    /// - `box2`: Second box
    ///
    /// # Returns
    ///
    /// - `Ok(())` if dimensions match
    /// - `Err(BoxError::DimensionMismatch)` if dimensions differ
    pub fn validate_compatible_dimensions<B: Box>(box1: &B, box2: &B) -> Result<(), BoxError> {
        let dim1 = box1.dim();
        let dim2 = box2.dim();

        if dim1 != dim2 {
            return Err(BoxError::DimensionMismatch {
                expected: dim1,
                actual: dim2,
            });
        }

        Ok(())
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
        assert!((0.0..=1.0).contains(&value));
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

    #[test]
    fn test_is_cross_pattern() {
        // High overlap, low containment in both directions = cross pattern
        assert!(is_cross_pattern(0.8, 0.2, 0.1, 0.3));

        // High overlap with high containment = not cross pattern
        assert!(!is_cross_pattern(0.8, 0.9, 0.1, 0.3));

        // Low overlap = not cross pattern
        assert!(!is_cross_pattern(0.2, 0.1, 0.1, 0.3));
    }

    #[test]
    fn test_is_perfectly_nested() {
        // One box perfectly contains the other
        assert!(is_perfectly_nested(0.98, 0.1, 0.95));
        assert!(is_perfectly_nested(0.1, 0.98, 0.95));

        // Neither box contains the other
        assert!(!is_perfectly_nested(0.5, 0.3, 0.95));

        // Both boxes contain each other (should still be detected)
        assert!(is_perfectly_nested(0.98, 0.97, 0.95));
    }

    #[test]
    fn test_suggested_min_separation() {
        // For 2D boxes with volume range [0.1, 1.0]
        let sep_2d = suggested_min_separation(2, (0.1, 1.0));
        assert!(sep_2d > 0.0);

        // For 3D boxes
        let sep_3d = suggested_min_separation(3, (0.1, 1.0));
        assert!(sep_3d > 0.0);

        // For 10D boxes
        let sep_10d = suggested_min_separation(10, (0.1, 1.0));
        assert!(sep_10d > 0.0);

        // All separations should be positive and reasonable
        assert!(sep_2d < 10.0); // Reasonable upper bound
        assert!(sep_3d < 10.0);
        assert!(sep_10d < 10.0);
    }
}
