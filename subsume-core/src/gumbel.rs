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
    /// P(point ∈ self) using Gumbel-Softmax.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if point dimension doesn't match box dimension.
    fn membership_probability(
        &self,
        point: &Self::Vector,
    ) -> Result<Self::Scalar, BoxError>;

    /// Sample a point from the box distribution.
    ///
    /// Uses Gumbel-Softmax to sample a point that lies within the box
    /// with probability proportional to the box's volume.
    fn sample(&self) -> Self::Vector;
}

