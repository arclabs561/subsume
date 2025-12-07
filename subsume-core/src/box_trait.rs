//! Core trait for box embeddings.
//!
//! # Mathematical Foundations
//!
//! Box embeddings model **subsumption** relationships geometrically. In formal logic,
//! subsumption means that one statement is more general than another. When box A contains
//! box B (B ⊆ A), we say that A **subsumes** B.
//!
//! # Research Background
//!
//! The foundational work on box embeddings for knowledge graphs is **Vilnis et al. (2018)**,
//! "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures". This paper establishes:
//!
//! - Box volumes as probability measures under uniform base measure
//! - Containment probability \(P(B|A) = \text{Vol}(A \cap B) / \text{Vol}(A)\) models entailment
//! - Boxes are closed under intersection operations (algebraic closure)
//! - Probabilistic boxes can represent negative correlations (disjoint boxes)
//!
//! **Reference**: Vilnis et al. (2018), [arXiv:1805.06627](https://arxiv.org/abs/1805.06627)
//!
//! ## Subsumption Formula
//!
//! ```text
//! Box A subsumes Box B  ⟺  B ⊆ A  ⟺  P(B|A) = Vol(A ∩ B) / Vol(A) = 1
//! ```
//!
//! ## Volume Calculation
//!
//! For a box with min coordinates `z_i` and max coordinates `Z_i`:
//!
//! ```text
//! Vol(B) = ∏ᵢ max(Z_i - z_i, 0)
//! ```
//!
//! For Gumbel boxes, the expected volume uses the Bessel approximation:
//!
//! ```text
//! E[Vol(B)] = 2β K₀(2e^(-(μᵧ - μₓ)/(2β)))
//! ```
//!
//! Or in LaTeX (renders with KaTeX in browser):
//!
//! \[
//! \mathbb{E}[\text{Vol}(B)] = 2\beta K_0\left(2e^{-\frac{\mu_y - \mu_x}{2\beta}}\right)
//! \]
//!
//! where \(K_0\) is the modified Bessel function of the second kind. The derivation from
//! Gumbel distributions to Bessel functions, asymptotic behavior, and numerical approximations
//! are explained in detail in the [mathematical foundations](../../docs/MATHEMATICAL_FOUNDATIONS.md).
//!
//! See [`docs/MATHEMATICAL_FOUNDATIONS.md`](../../docs/MATHEMATICAL_FOUNDATIONS.md) for:
//! - Complete derivation from Gumbel PDFs to Bessel functions
//! - Gumbel max-stability and algebraic closure properties
//! - Log-sum-exp function and numerical stability
//! - First-order Taylor approximation for containment probability
//! - Measure-theoretic foundations
//!
//! **For detailed study:** PDF versions with professional typesetting are available in
//! [`docs/typst-output/pdf/`](../../docs/typst-output/pdf/), including complete step-by-step
//! derivations, proofs, and examples following textbook-style exposition.

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
    /// For a box with min coordinates `z_i` and max coordinates `Z_i`:
    ///
    /// ```text
    /// Vol(B) = ∏ᵢ max(Z_i - z_i, 0)
    /// ```
    ///
    /// where `d` is the number of dimensions.
    ///
    /// For Gumbel boxes, this computes the expected volume using the Bessel approximation.
    /// See the [mathematical foundations](../../docs/MATHEMATICAL_FOUNDATIONS.md) for details.
    ///
    ///
    /// # Errors
    ///
    /// Returns `BoxError::InvalidBounds` if any min\[i\] > max\[i\].
    fn volume(&self, temperature: Self::Scalar) -> Result<Self::Scalar, BoxError>;

    /// Compute the intersection of two boxes.
    ///
    /// Returns a new box representing the overlap region.
    ///
    /// ## Mathematical Formulation
    ///
    /// For hard boxes, intersection is computed coordinate-wise:
    ///
    /// ```text
    /// z_∩,i = max(z_i^A, z_i^B),  Z_∩,i = min(Z_i^A, Z_i^B)
    /// ```
    ///
    /// The intersection is valid (non-empty) if `z_∩,i ≤ Z_∩,i` for all dimensions `i`.
    ///
    /// For Gumbel boxes, intersection coordinates use log-sum-exp due to max-stability:
    ///
    /// ```text
    /// Z_∩,i ~ MaxGumbel(lse_β(μ_{z,i}^A, μ_{z,i}^B))
    /// ```
    ///
    /// where `lse_β(x, y) = max(x, y) + β log(1 + exp(-|x-y|/β))` is the
    /// numerically stable log-sum-exp function.
    ///
    /// See the [mathematical foundations](../docs/MATHEMATICAL_FOUNDATIONS.md)
    /// for details on Gumbel intersections and max-stability.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::DimensionMismatch` if boxes have different dimensions.
    fn intersection(&self, other: &Self) -> Result<Self, BoxError>;

    /// Compute the probability that `self` contains `other`.
    ///
    /// This is the core **subsumption** operation: P(other ⊆ self).
    ///
    /// ## Paradigm Problem: Modeling Hierarchical Relationships
    ///
    /// Consider the knowledge graph triple (dog, is_a, mammal). We want to check whether
    /// the "mammal" box contains the "dog" box. If it does, then the relationship is likely true.
    ///
    /// **Step-by-step reasoning**:
    /// 1. Compute the intersection: What region do "dog" and "mammal" boxes share?
    /// 2. Compare volumes: If the intersection volume equals the "dog" volume, then "dog" is
    ///    completely contained in "mammal"
    /// 3. Normalize: Divide by "dog" volume to get a probability between 0 and 1
    ///
    /// **Intuitive interpretation**: This is analogous to asking: "If I randomly pick a point
    /// from the 'dog' box, what's the probability it's also in the 'mammal' box?" The answer
    /// is the containment probability.
    ///
    /// **Research foundation**: This formulation comes from **Vilnis et al. (2018)**, who established
    /// that box volumes can be interpreted as probability measures under uniform base measure.
    ///
    /// ## Mathematical Formulation
    ///
    /// For standard boxes:
    /// ```text
    /// P(other ⊆ self) = Vol(self ∩ other) / Vol(other)
    /// ```
    ///
    /// Or in LaTeX:
    /// \[
    /// P(\text{other} \subseteq \text{self}) = \frac{\text{Vol}(\text{self} \cap \text{other})}{\text{Vol}(\text{other})}
    /// \]
    ///
    /// For Gumbel boxes, this uses the expected volume with first-order Taylor approximation:
    /// \[
    /// P(\text{other} \subseteq \text{self}) \approx \frac{\mathbb{E}[\text{Vol}(\text{self} \cap \text{other})]}{\mathbb{E}[\text{Vol}(\text{other})]}
    /// \]
    ///
    /// The approximation \(\mathbb{E}[\text{X}/\text{Y}] \approx \mathbb{E}[\text{X}]/\mathbb{E}[\text{Y}]\) is valid when:
    /// - The coefficient of variation \(\text{Var}(Y)/\mu_Y^2\) is small
    /// - \(\mu_Y\) is bounded away from zero
    ///
    /// The second-order error term is approximately:
    /// \[
    /// \text{Error} \approx -\frac{\text{Cov}(\text{X}, \text{Y})}{\mu_Y^2} + \frac{\mu_X \text{Var}(\text{Y})}{\mu_Y^3}
    /// \]
    ///
    /// This directly models logical subsumption: if the probability is 1.0, then
    /// `self` completely subsumes `other` (entailment relationship).
    ///
    /// See the [mathematical foundations](../docs/MATHEMATICAL_FOUNDATIONS.md)
    /// section "Containment Probability" for the complete derivation of the Taylor approximation.
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
    /// ## Paradigm Problem: Related but Distinct Entities
    ///
    /// Consider "dog" and "cat". They're both animals, both mammals, both pets—so they're
    /// related. But they're distinct species—a dog is not a cat. How do we measure this
    /// relationship?
    ///
    /// **The answer**: Overlap probability. If the "dog" box and "cat" box overlap significantly,
    /// they share many properties (high overlap = related). If they barely overlap, they're
    /// more distinct (low overlap = less related).
    ///
    /// **Visual intuition**: Imagine two boxes in space. If they overlap a lot, they share
    /// much of the same "conceptual space". If they're far apart or barely touch, they're
    /// more distinct. Overlap probability quantifies this geometric relationship.
    ///
    /// **Step-by-step computation**:
    /// 1. Compute intersection: What region do both boxes share?
    /// 2. Compute union: What region does either box cover?
    /// 3. Ratio: Intersection volume / Union volume gives overlap probability
    ///    - If boxes are identical: overlap = 1.0 (completely related)
    ///    - If boxes are disjoint: overlap = 0.0 (completely distinct)
    ///    - If boxes partially overlap: 0.0 < overlap < 1.0 (related but distinct)
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
