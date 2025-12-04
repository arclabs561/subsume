//! Core trait for box embeddings.
//!
//! # Mathematical Foundations
//!
//! Box embeddings model **subsumption** relationships geometrically. In formal logic,
//! subsumption means that one statement is more general than another. When box A contains
//! box B (B ⊆ A), we say that A **subsumes** B.
//!
//! ## Subsumption Formula
//!
//! \[
//! \text{Box A subsumes Box B} \iff B \subseteq A \iff P(B|A) = \frac{\text{Vol}(A \cap B)}{\text{Vol}(A)} = 1
//! \]
//!
//! ## Volume Calculation
//!
//! For a box with min coordinates \(z_i\) and max coordinates \(Z_i\):
//!
//! \[
//! \text{Vol}(B) = \prod_{i=1}^{d} \max(Z_i - z_i, 0)
//! \]
//!
//! For Gumbel boxes, the expected volume uses the Bessel approximation:
//!
//! \[
//! \mathbb{E}[\text{Vol}(B)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
//! \]
//!
//! where \(K_0\) is the modified Bessel function of the second kind.
//!
//! See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../../docs/MATHEMATICAL_FOUNDATIONS.md) for complete details.

/// A geometric box (axis-aligned hyperrectangle) in d-dimensional space.
///
/// Boxes model containment relationships: if box A contains box B,
/// then A **subsumes** B (entailment, hierarchical relationship).
///
/// The term "subsume" comes from formal logic, where subsumption means
/// one statement is more general than another. This geometric representation
/// directly models logical subsumption through containment.
///
/// # Type Parameters
///
/// - `Scalar`: The scalar type (e.g., `f32`, `Tensor`)
/// - `Vector`: The vector type (e.g., `Vec<f32>`, `Tensor`)
///
/// # Invariants
///
/// - `min\[i\] <= max\[i\]` for all dimensions i
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
    /// ## Mathematical Formulation
    ///
    /// For a box with min coordinates \(z_i\) and max coordinates \(Z_i\):
    ///
    /// \[
    /// \text{Vol}(B) = \prod_{i=1}^{d} \max(Z_i - z_i, 0)
    /// \]
    ///
    /// where \(d\) is the number of dimensions.
    ///
    /// For Gumbel boxes, this computes the expected volume using the Bessel approximation.
    /// See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../../docs/MATHEMATICAL_FOUNDATIONS.md) for details.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::InvalidBounds` if any min\[i\] > max\[i\].
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
    /// This is the core **subsumption** operation: P(other ⊆ self).
    ///
    /// ## Mathematical Formulation
    ///
    /// For standard boxes:
    /// \[
    /// P(\text{other} \subseteq \text{self}) = \frac{\text{Vol}(\text{self} \cap \text{other})}{\text{Vol}(\text{other})}
    /// \]
    ///
    /// For Gumbel boxes, this uses the expected volume with first-order Taylor approximation:
    /// \[
    /// P(\text{other} \subseteq \text{self}) \approx \frac{\mathbb{E}[\text{Vol}(\text{self} \cap \text{other})]}{\mathbb{E}[\text{Vol}(\text{other})]}
    /// \]
    ///
    /// This directly models logical subsumption: if the probability is 1.0, then
    /// `self` completely subsumes `other` (entailment relationship).
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError>;

    /// Compute the probability that two boxes overlap (non-empty intersection).
    ///
    /// This measures whether two boxes represent related but distinct entities (e.g.,
    /// "dog" and "cat" are both animals but distinct species).
    ///
    /// ## Mathematical Definition
    ///
    /// Using the inclusion-exclusion principle:
    ///
    /// \[
    /// P(\text{self} \cap \text{other} \neq \emptyset) = \frac{\text{Vol}(\text{self} \cap \text{other})}{\text{Vol}(\text{self} \cup \text{other})}
    /// \]
    ///
    /// Where the union volume is computed as:
    ///
    /// \[
    /// \text{Vol}(\text{self} \cup \text{other}) = \text{Vol}(\text{self}) + \text{Vol}(\text{other}) - \text{Vol}(\text{self} \cap \text{other})
    /// \]
    ///
    /// ## Interpretation
    ///
    /// - **1.0**: Complete overlap (boxes are identical or one contains the other)
    /// - **0.0**: Complete disjointness (boxes don't intersect)
    /// - **0.5**: Partial overlap (half the union is intersection)
    ///
    /// ## Use Cases
    ///
    /// - **Entity resolution**: High overlap probability suggests two boxes represent the same entity
    /// - **Relatedness**: Moderate overlap suggests related but distinct concepts
    /// - **Disjointness**: Low overlap suggests mutually exclusive concepts
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    /// Compute the probability that two boxes overlap (non-empty intersection).
    ///
    /// This measures whether two boxes represent related but distinct entities (e.g.,
    /// "dog" and "cat" are both animals but distinct species).
    ///
    /// ## Mathematical Definition
    ///
    /// Using the inclusion-exclusion principle:
    ///
    /// \[
    /// P(\text{self} \cap \text{other} \neq \emptyset) = \frac{\text{Vol}(\text{self} \cap \text{other})}{\text{Vol}(\text{self} \cup \text{other})}
    /// \]
    ///
    /// Where the union volume is computed as:
    ///
    /// \[
    /// \text{Vol}(\text{self} \cup \text{other}) = \text{Vol}(\text{self}) + \text{Vol}(\text{other}) - \text{Vol}(\text{self} \cap \text{other})
    /// \]
    ///
    /// ## Interpretation
    ///
    /// - **1.0**: Complete overlap (boxes are identical or one contains the other)
    /// - **0.0**: Complete disjointness (boxes don't intersect)
    /// - **0.5**: Partial overlap (half the union is intersection)
    ///
    /// ## Use Cases
    ///
    /// - **Entity resolution**: High overlap probability suggests two boxes represent the same entity
    /// - **Relatedness**: Moderate overlap suggests related but distinct concepts
    /// - **Disjointness**: Low overlap suggests mutually exclusive concepts
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    fn overlap_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, BoxError>;

    /// Compute the union of two boxes.
    ///
    /// Returns the smallest box that contains both `self` and `other`.
    /// Union box has min = min(self.min, other.min) and max = max(self.max, other.max).
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    fn union(&self, other: &Self) -> Result<Self, BoxError>;

    /// Get the center point of the box.
    ///
    /// Center = (min + max) / 2 for each dimension.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::Internal` if center computation fails.
    fn center(&self) -> Result<Self::Vector, BoxError>;

    /// Compute the minimum distance between two boxes.
    ///
    /// Returns 0.0 if boxes overlap, otherwise the Euclidean distance
    /// between the closest points on the two boxes.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    fn distance(&self, other: &Self) -> Result<Self::Scalar, BoxError>;
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

    /// Invalid bounds: min\[i\] > max\[i\] for some dimension i.
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
