//! Core trait for box embeddings.

/// A geometric box (axis-aligned hyperrectangle) in d-dimensional space.
///
/// Boxes model containment relationships: if box A contains box B,
/// then A "subsumes" B (entailment, hierarchical relationship).
///
/// # Type Parameters
///
/// - `Scalar`: The scalar type (e.g., `f32`, `Tensor`)
/// - `Vector`: The vector type (e.g., `Vec<f32>`, `Tensor`)
///
/// # Invariants
///
/// - `min[i] <= max[i]` for all dimensions i
/// - `dim()` returns the number of dimensions
///
/// # Example
///
/// ```rust,ignore
/// // This example requires a backend implementation (e.g., subsume-ndarray)
/// use subsume_core::Box;
/// use subsume_ndarray::NdarrayBox;
/// use ndarray::array;
///
/// // Create two boxes
/// let premise = NdarrayBox::new(
///     array![0.0, 0.0],
///     array![1.0, 1.0],
///     1.0,
/// ).unwrap();
///
/// let hypothesis = NdarrayBox::new(
///     array![0.2, 0.2],
///     array![0.8, 0.8],
///     1.0,
/// ).unwrap();
///
/// // Compute containment probability: P(hypothesis ⊆ premise)
/// let prob = premise.containment_prob(&hypothesis, 1.0).unwrap();
/// assert!(prob > 0.9); // hypothesis is contained in premise
/// ```
pub trait Box: Sized {
    /// Scalar type for probabilities, volumes, etc.
    type Scalar: Clone + Copy + PartialOrd;

    /// Vector type for min/max bounds.
    type Vector: Clone;

    /// Get the minimum bound in each dimension.
    fn min(&self) -> &Self::Vector;

    /// Get the maximum bound in each dimension.
    fn max(&self) -> &Self::Vector;

    /// Get the number of dimensions.
    fn dim(&self) -> usize;

    /// Compute the volume of the box.
    ///
    /// Volume = ∏(max[i] - min[i])
    ///
    /// # Errors
    ///
    /// Returns `BoxError::InvalidBounds` if any min[i] > max[i].
    fn volume(&self, temperature: Self::Scalar) -> Result<Self::Scalar, BoxError>;

    /// Compute the intersection of two boxes.
    ///
    /// Returns a new box representing the overlap region.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    fn intersection(&self, other: &Self) -> Result<Self, BoxError>;

    /// Compute the probability that `self` contains `other`.
    ///
    /// This is the core "subsumption" operation: P(other ⊆ self).
    ///
    /// For standard boxes: `intersection_volume(other) / other.volume()`
    /// For Gumbel boxes: Uses Gumbel-Softmax reparameterization.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    fn containment_prob(&self, other: &Self, temperature: Self::Scalar) -> Result<Self::Scalar, BoxError>;

    /// Compute the probability that two boxes overlap (non-empty intersection).
    ///
    /// P(self ∩ other ≠ ∅) = intersection_volume / union_volume
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    fn overlap_prob(&self, other: &Self, temperature: Self::Scalar) -> Result<Self::Scalar, BoxError>;
}

/// Errors that can occur during box operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BoxError {
    /// Boxes have different dimensions.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions.
        actual: usize,
    },

    /// Invalid bounds: min[i] > max[i] for some dimension i.
    #[error("Invalid bounds: min[{dim}] = {min} > max[{dim}] = {max}")]
    InvalidBounds {
        /// Dimension index where the invalid bounds occur.
        dim: usize,
        /// Minimum bound value at this dimension.
        min: f64,
        /// Maximum bound value at this dimension.
        max: f64,
    },

    /// Box has zero or negative volume.
    #[error("Box has zero or negative volume")]
    ZeroVolume,

    /// Internal error from tensor/array operations.
    #[error("Internal error: {0}")]
    Internal(String),
}
