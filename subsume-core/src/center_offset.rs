//! Center-offset representation for box embeddings.
//!
//! This module provides utilities for converting between center-offset and min-max
//! representations. The center-offset representation is better suited for neural network
//! training due to improved gradient flow and automatic constraint satisfaction.
//!
//! # Intuitive Explanation
//!
//! Instead of directly learning min and max coordinates (which must satisfy min ≤ max),
//! we learn a **center point** and an **offset** (half-width). This automatically ensures
//! valid boxes because:
//!
//! - The center is the midpoint: \(c = \frac{\min + \max}{2}\)
//! - The offset is half the width: \(o = \frac{\max - \min}{2}\)
//!
//! Since offset is always non-negative (enforced by softplus), we automatically get
//! \(\min \leq \max\). This is like describing a box by saying "it's centered at point X
//! and extends Y units in each direction" rather than "it goes from point A to point B".
//!
//! **Why this helps training**: Neural networks can freely update center and offset
//! without worrying about constraint violations. The sigmoid ensures bounds stay in [0, 1].
//!
//! # Mathematical Formulation
//!
//! **Center-offset to min-max**:
//!
//! \[
//! \begin{aligned}
//! \min_i &= \sigma(c_i - \text{softplus}(o_i)) \\
//! \max_i &= \sigma(c_i + \text{softplus}(o_i))
//! \end{aligned}
//! \]
//!
//! where:
//! - \(\sigma\) is the sigmoid function: \(\sigma(x) = \frac{1}{1 + e^{-x}}\)
//! - \(\text{softplus}(x) = \ln(1 + e^x)\) ensures non-negative offsets
//!
//! **Min-max to center-offset** (inverse):
//!
//! \[
//! \begin{aligned}
//! c_i &= \frac{\text{logit}(\min_i) + \text{logit}(\max_i)}{2} \\
//! o_i &= \text{softplus}^{-1}\left(\frac{\text{logit}(\max_i) - \text{logit}(\min_i)}{2}\right)
//! \end{aligned}
//! \]
//!
//! where \(\text{logit}(x) = \ln\left(\frac{x}{1-x}\right)\) is the inverse sigmoid.
//!
//! See `docs/CENTER_OFFSET_REPRESENTATION.md` for detailed comparison and trade-offs.

use crate::BoxError;

/// Convert center-offset representation to min-max bounds.
///
/// # Intuitive Explanation
///
/// This function takes a center point and offset (half-width) and converts them to
/// min-max coordinates. The process:
///
/// 1. **Apply softplus to offset**: Ensures it's non-negative (softplus(x) ≥ 0 always)
/// 2. **Subtract/add to center**: Creates min and max coordinates
/// 3. **Apply sigmoid**: Clamps to [0, 1] range for bounded boxes
///
/// **Why sigmoid?** It ensures the final box stays within [0, 1]^d, which is useful for
/// normalized embeddings. The sigmoid acts like a "soft clamp" that provides smooth gradients.
///
/// # Mathematical Formulation
///
/// For each dimension \(i\):
///
/// \[
/// \begin{aligned}
/// \min_i &= \sigma(c_i - \text{softplus}(o_i)) \\
/// \max_i &= \sigma(c_i + \text{softplus}(o_i))
/// \end{aligned}
/// \]
///
/// where:
/// - \(\sigma(x) = \frac{1}{1 + e^{-x}}\) is the sigmoid function
/// - \(\text{softplus}(x) = \ln(1 + e^x)\) ensures \(o_i \geq 0\)
///
/// **Numerical stability**: For large \(x\), we use \(\text{softplus}(x) \approx x\) to avoid
/// overflow. For large \(|x|\) in sigmoid, we clamp to avoid numerical issues.
///
/// # Parameters
///
/// * `center` - Center point \(c \in \mathbb{R}^d\) of the box
/// * `offset` - Offset vector \(o \in \mathbb{R}^d\) (will be made non-negative via softplus)
/// * `min_out` - Output: minimum bounds \(\min \in [0, 1]^d\)
/// * `max_out` - Output: maximum bounds \(\max \in [0, 1]^d\)
pub fn center_offset_to_min_max(
    center: &[f32],
    offset: &[f32],
    min_out: &mut [f32],
    max_out: &mut [f32],
) -> Result<(), BoxError> {
    if center.len() != offset.len()
        || center.len() != min_out.len()
        || center.len() != max_out.len()
    {
        return Err(BoxError::DimensionMismatch {
            expected: center.len(),
            actual: offset.len().max(min_out.len()).max(max_out.len()),
        });
    }

    for i in 0..center.len() {
        // Softplus: log(1 + exp(x)), ensures non-negative offset
        // Using stable softplus: log(1 + exp(x)) = log1p(exp(x))
        let softplus_offset = if offset[i] > 10.0 {
            offset[i] // For large x, softplus(x) ≈ x
        } else {
            offset[i].exp().ln_1p()
        };

        // Apply sigmoid to center ± softplus(offset)
        let min_val = center[i] - softplus_offset;
        let max_val = center[i] + softplus_offset;

        // Sigmoid: 1 / (1 + exp(-x))
        // Use stable sigmoid to avoid overflow
        let sigmoid = |x: f32| {
            if x > 10.0 {
                1.0
            } else if x < -10.0 {
                0.0
            } else {
                1.0 / (1.0 + (-x).exp())
            }
        };

        min_out[i] = sigmoid(min_val);
        max_out[i] = sigmoid(max_val);
    }

    Ok(())
}

/// Convert min-max representation to center-offset.
///
/// # Parameters
///
/// * `min` - Minimum bounds
/// * `max` - Maximum bounds
/// * `center_out` - Output: center point
/// * `offset_out` - Output: offset vector
///
/// # Formula
///
/// ```
/// center = (inverse_sigmoid(min) + inverse_sigmoid(max)) / 2
/// offset = softplus_inverse((inverse_sigmoid(max) - inverse_sigmoid(min)) / 2)
/// ```
pub fn min_max_to_center_offset(
    min: &[f32],
    max: &[f32],
    center_out: &mut [f32],
    offset_out: &mut [f32],
) -> Result<(), BoxError> {
    if min.len() != max.len() || min.len() != center_out.len() || min.len() != offset_out.len() {
        return Err(BoxError::DimensionMismatch {
            expected: min.len(),
            actual: max.len().max(center_out.len()).max(offset_out.len()),
        });
    }

    for i in 0..min.len() {
        // Inverse sigmoid: logit(x) = ln(x / (1 - x))
        // Clamp to avoid division by zero or log of negative
        let min_clamped = min[i].clamp(1e-7, 1.0 - 1e-7);
        let max_clamped = max[i].clamp(1e-7, 1.0 - 1e-7);

        let min_logit = (min_clamped / (1.0 - min_clamped)).ln();
        let max_logit = (max_clamped / (1.0 - max_clamped)).ln();

        // Center is average of logits
        center_out[i] = (min_logit + max_logit) / 2.0;

        // Offset is half the difference, converted back via inverse softplus
        let offset_logit = (max_logit - min_logit) / 2.0;
        // Inverse softplus: ln(exp(x) - 1)
        // Clamp to avoid log of non-positive
        let exp_offset = offset_logit.exp();
        offset_out[i] = if exp_offset > 1.0 {
            (exp_offset - 1.0).ln()
        } else {
            0.0 // If exp(x) <= 1, softplus_inverse is 0
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_offset_round_trip() {
        let min = [0.2, 0.3];
        let max = [0.8, 0.9];
        let mut center = [0.0; 2];
        let mut offset = [0.0; 2];
        let mut min_out = [0.0; 2];
        let mut max_out = [0.0; 2];

        // Convert min-max to center-offset
        min_max_to_center_offset(&min, &max, &mut center, &mut offset).unwrap();

        // Convert back to min-max
        center_offset_to_min_max(&center, &offset, &mut min_out, &mut max_out).unwrap();

        // Should be approximately equal (within numerical precision)
        for i in 0..2 {
            assert!((min_out[i] - min[i]).abs() < 1e-5, "min[{}] mismatch", i);
            assert!((max_out[i] - max[i]).abs() < 1e-5, "max[{}] mismatch", i);
        }
    }

    #[test]
    fn test_center_offset_constraints() {
        let center = [0.5, 0.5];
        let offset = [0.2, 0.3];
        let mut min = [0.0; 2];
        let mut max = [0.0; 2];

        center_offset_to_min_max(&center, &offset, &mut min, &mut max).unwrap();

        // Verify constraints: min[i] <= max[i] for all i
        for i in 0..2 {
            assert!(
                min[i] <= max[i],
                "Constraint violation: min[{}] > max[{}]",
                i,
                i
            );
            assert!(min[i] >= 0.0 && min[i] <= 1.0, "min[{}] out of bounds", i);
            assert!(max[i] >= 0.0 && max[i] <= 1.0, "max[{}] out of bounds", i);
        }
    }
}
