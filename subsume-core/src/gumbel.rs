//! Gumbel box embeddings for probabilistic containment.

use crate::{Box, BoxError};

/// A Gumbel box embedding with temperature-controlled softness.
///
/// Gumbel boxes use the Gumbel-Softmax reparameterization trick to make
/// box bounds differentiable during training. The `temperature` parameter
/// controls the "hardness" of the bounds:
///
/// - `temperature → 0`: Hard bounds (standard boxes)
/// - `temperature → ∞`: Soft bounds (approaches uniform distribution)
///
/// # Example
///
/// ```rust,ignore
/// // This example requires a backend implementation (e.g., subsume-ndarray)
/// use subsume_core::GumbelBox;
/// use subsume_ndarray::NdarrayGumbelBox;
/// use ndarray::array;
///
/// let gumbel_box = NdarrayGumbelBox::new(
///     array![0.0, 0.0],
///     array![1.0, 1.0],
///     0.5, // temperature
/// ).unwrap();
///
/// // Sample a point from the box distribution
/// let sample = gumbel_box.sample();
///
/// // Compute membership probability for a point
/// let point = array![0.5, 0.5];
/// let prob = gumbel_box.membership_probability(&point).unwrap();
/// ```
///
/// # Research Background
///
/// Based on:
/// - Jang et al. (2016): "Categorical Reparameterization with Gumbel-Softmax"
/// - Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
pub trait GumbelBox: Box {
    /// Get the temperature parameter (controls softness of bounds).
    fn temperature(&self) -> Self::Scalar;

    /// Compute membership probability for a point.
    ///
    /// This computes P(point ∈ self) using the Gumbel-Softmax framework.
    ///
    /// ## Mathematical Formulation
    ///
    /// For a point x and box with bounds [min, max] at temperature τ:
    ///
    /// \[
    /// P(\text{point} \in \text{self}) = \sigma\left(\frac{x - \text{min}}{\tau}\right) \cdot \sigma\left(\frac{\text{max} - x}{\tau}\right)
    /// \]
    ///
    /// where σ is the sigmoid function. This is the product of:
    /// - P(x > min): Probability that point is above minimum bound
    /// - P(x < max): Probability that point is below maximum bound
    ///
    /// ## Interpretation
    ///
    /// - **1.0**: Point is definitely inside the box (high confidence)
    /// - **0.0**: Point is definitely outside the box (high confidence)
    /// - **0.5**: Point is on the boundary (uncertain)
    ///
    /// As temperature decreases, the probability becomes sharper (more like hard membership).
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if point dimension doesn't match box dimension.
    fn membership_probability(&self, point: &Self::Vector) -> Result<Self::Scalar, BoxError>;

    /// Sample a point from the box distribution.
    ///
    /// Uses Gumbel-Softmax to sample a point that lies within the box bounds.
    /// The sampling is uniform within the box (all points equally likely), but the
    /// Gumbel distribution ensures the sampling is differentiable with respect to
    /// the box parameters.
    ///
    /// ## Sampling Process
    ///
    /// 1. Sample Gumbel noise for each dimension
    /// 2. Transform Gumbel samples to [0, 1] using tanh and temperature scaling
    /// 3. Map to box bounds [min, max] in each dimension
    ///
    /// This ensures samples are always within bounds while maintaining differentiability.
    ///
    /// ## Use Cases
    ///
    /// - **Training**: Sampling enables stochastic gradient estimation
    /// - **Inference**: Can sample multiple points to estimate box properties
    /// - **Visualization**: Sample points to visualize box shape and location
    fn sample(&self) -> Self::Vector;
}
