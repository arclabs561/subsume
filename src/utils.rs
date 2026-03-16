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
//! See [`docs/MATHEMATICAL_FOUNDATIONS.md`](https://github.com/arclabs561/subsume/blob/main/docs/MATHEMATICAL_FOUNDATIONS.md)
//! for complete mathematical derivations and [`docs/MATH_TO_CODE_CONNECTIONS.md`](https://github.com/arclabs561/subsume/blob/main/docs/MATH_TO_CODE_CONNECTIONS.md)
//! for how these patterns are used in the codebase.
//!
//! **For detailed study:** PDF versions with professional typesetting are available:
//! - [`docs/typst-output/pdf/log-sum-exp-intersection.pdf`](https://github.com/arclabs561/subsume/blob/main/docs/typst-output/pdf/log-sum-exp-intersection.pdf) - Log-sum-exp function and numerical stability
//! - [`docs/typst-output/pdf/gumbel-box-volume.pdf`](https://github.com/arclabs561/subsume/blob/main/docs/typst-output/pdf/gumbel-box-volume.pdf) - Volume calculations and numerical considerations

/// Euler-Mascheroni constant (gamma ~ 0.5772).
///
/// Appears in the Bessel approximation for Gumbel box expected volume
/// (Dasgupta et al., 2020): the offset `2 * gamma * beta` accounts for
/// the mean of the Gumbel distribution.
pub const EULER_GAMMA: f32 = 0.577_215_7;

/// Floor for log-volume when volume is near zero (ln(1e-10) ≈ -23.0).
/// Used in depth and boundary distance to avoid -inf from log(0).
#[cfg(any(feature = "ndarray-backend", feature = "candle-backend"))]
pub(crate) const LOG_VOLUME_FLOOR: f32 = -23.0;

/// Minimum containment probability to consider entities "fully contained"
/// in boundary distance computation.
#[cfg(any(feature = "ndarray-backend", feature = "candle-backend"))]
pub(crate) const BOUNDARY_CONTAINMENT_THRESHOLD: f32 = 0.99;

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
pub(crate) fn clamp_temperature(temp: f32, min: f32, max: f32) -> f32 {
    temp.clamp(min, max)
}

/// Default minimum safe temperature to avoid numerical underflow.
pub const MIN_TEMPERATURE: f32 = 1e-3;

/// Default maximum safe temperature to maintain correspondence.
pub const MAX_TEMPERATURE: f32 = 10.0;

/// Clamp temperature using default safe bounds.
pub(crate) fn clamp_temperature_default(temp: f32) -> f32 {
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
/// See [`docs/MATHEMATICAL_FOUNDATIONS.md`](https://github.com/arclabs561/subsume/blob/main/docs/MATHEMATICAL_FOUNDATIONS.md)
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
/// See [`docs/MATHEMATICAL_FOUNDATIONS.md`](https://github.com/arclabs561/subsume/blob/main/docs/MATHEMATICAL_FOUNDATIONS.md)
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
/// See [`docs/typst-output/pdf/gumbel-max-stability.pdf`](https://github.com/arclabs561/subsume/blob/main/docs/typst-output/pdf/gumbel-max-stability.pdf)
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

        // ---- gumbel_membership_prob always in [0, 1] for finite inputs ----
        proptest! {
            #[test]
            fn prop_gumbel_membership_prob_bounds(
                x in -10.0f32..10.0f32,
                min_val in -10.0f32..10.0f32,
                width in 0.01f32..20.0f32,
                temp in 0.1f32..10.0f32,
            ) {
                let max_val = min_val + width;
                let p = gumbel_membership_prob(x, min_val, max_val, temp);
                prop_assert!(
                    (0.0..=1.0).contains(&p),
                    "gumbel_membership_prob({x}, {min_val}, {max_val}, {temp}) = {p} not in [0, 1]"
                );
                prop_assert!(p.is_finite(),
                    "gumbel_membership_prob must be finite, got {p}");
            }
        }
    }
}
