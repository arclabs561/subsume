//! Octagon embeddings for knowledge graph reasoning.
//!
//! # Overview
//!
//! Octagon embeddings represent entities as axis-aligned octagons (8-sided polytopes in 2D,
//! or octagonal prisms in higher dimensions). Each octagon is defined by:
//! - **Axis-aligned bounds**: standard min/max per dimension (like a box)
//! - **Diagonal bounds**: constraints on (x_i + x_j) and (x_i - x_j) for adjacent
//!   dimension pairs (i, i+1)
//!
//! # Octagons vs Boxes
//!
//! | Property | Boxes | Octagons |
//! |---|---|---|
//! | Geometry | Axis-aligned hyperrectangles | Axis-aligned hyperrectangles + diagonal cuts |
//! | Constraints per dim | 2 (min, max) | 2 axis + 4 diagonal per adjacent pair |
//! | Closure under intersection | Yes | Yes |
//! | Closure under composition | No (for mappings) | Yes |
//! | Expressiveness for rules | Cannot model transitivity | Can model transitivity, composition |
//!
//! Octagons are strictly more expressive than boxes: any box is an octagon with
//! vacuous diagonal constraints, but octagons can represent shapes that boxes cannot.
//!
//! # Storage Efficiency
//!
//! For d dimensions, we store:
//! - 2d axis-aligned bounds (same as boxes)
//! - 4(d-1) diagonal bounds for adjacent pairs (i, i+1)
//!
//! This gives O(d) total storage, not O(d^2). The adjacent-pair restriction is a
//! practical tradeoff: full pairwise diagonal constraints would need O(d^2) storage
//! but adjacent pairs capture most of the relational structure.
//!
//! # Mathematical Foundations
//!
//! In 2D, an octagon is the intersection of:
//! ```text
//! x_min <= x <= x_max          (axis-aligned on x)
//! y_min <= y <= y_max          (axis-aligned on y)
//! sum_min <= x + y <= sum_max  (diagonal, slope -1)
//! diff_min <= x - y <= diff_max (diagonal, slope +1)
//! ```
//!
//! This gives 8 half-plane constraints, producing an 8-sided polygon (octagon).
//!
//! In d dimensions with adjacent-pair diagonal constraints, for each pair (i, i+1):
//! ```text
//! sum_min[k] <= x_i + x_{i+1} <= sum_max[k]
//! diff_min[k] <= x_i - x_{i+1} <= diff_max[k]
//! ```
//! where k indexes the adjacent pair.
//!
//! # Key Properties
//!
//! 1. **Closed under intersection**: intersection of two octagons is an octagon
//!    (take componentwise max of lower bounds, min of upper bounds for each constraint).
//!
//! 2. **Closed under composition**: if R and S are octagon-to-octagon mappings,
//!    then R compose S is also an octagon-to-octagon mapping.
//!
//! 3. **Strictly more expressive than boxes**: octagons can model relational rules
//!    (transitivity, composition) that boxes cannot.
//!
//! # References
//!
//! - Charpenay & Schockaert (2024), "Capturing Knowledge Graphs and Rules with
//!   Octagon Embeddings" (IJCAI 2024, arXiv:2401.16270)
//! - Gomber & Singh (2025, ICLR Workshop), "Neural Abstract Interpretation" --
//!   connects octagon embeddings to differentiable abstract interpretation; the Octagon
//!   abstract domain from static analysis is the same geometric object used here

/// Errors that can occur during octagon operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum OctagonError {
    /// Octagons have different dimensions.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions.
        actual: usize,
    },

    /// Invalid axis-aligned bounds: min\[i\] > max\[i\] for some dimension.
    #[error("Invalid axis bounds: min[{dim}] = {min} > max[{dim}] = {max}")]
    InvalidAxisBounds {
        /// Dimension index where the invalid bounds occur.
        dim: usize,
        /// Minimum bound value.
        min: f64,
        /// Maximum bound value.
        max: f64,
    },

    /// Invalid diagonal bounds: sum_min > sum_max or diff_min > diff_max.
    #[error("Invalid diagonal bounds for pair ({dim_i}, {dim_j}): {kind}_min = {lo} > {kind}_max = {hi}")]
    InvalidDiagonalBounds {
        /// First dimension index.
        dim_i: usize,
        /// Second dimension index.
        dim_j: usize,
        /// Which diagonal constraint ("sum" or "diff").
        kind: String,
        /// Lower bound value.
        lo: f64,
        /// Upper bound value.
        hi: f64,
    },

    /// Octagon is empty (infeasible constraints).
    #[error("Octagon is empty (constraints are infeasible)")]
    Empty,

    /// Internal computation error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Diagonal bounds for a single adjacent dimension pair (i, i+1).
///
/// Stores bounds on (x_i + x_{i+1}) and (x_i - x_{i+1}).
pub trait DiagonalBounds {
    /// Scalar type.
    type Scalar: Clone + Copy + PartialOrd;

    /// Lower bound on x_i + x_{i+1}.
    fn sum_min(&self) -> Self::Scalar;

    /// Upper bound on x_i + x_{i+1}.
    fn sum_max(&self) -> Self::Scalar;

    /// Lower bound on x_i - x_{i+1}.
    fn diff_min(&self) -> Self::Scalar;

    /// Upper bound on x_i - x_{i+1}.
    fn diff_max(&self) -> Self::Scalar;
}

/// An octagon embedding in d-dimensional space.
///
/// Octagons are axis-aligned polytopes with additional diagonal constraints
/// on adjacent dimension pairs. They are strictly more expressive than boxes
/// for modeling relational rules while remaining closed under intersection
/// and composition.
///
/// # Type Parameters
///
/// - `Scalar`: The scalar type (e.g., `f32`)
/// - `Vector`: The vector type (e.g., `Array1<f32>`)
/// - `Diag`: The diagonal bounds type for adjacent pairs
pub trait Octagon: Sized {
    /// Scalar type for probabilities, volumes, etc.
    type Scalar: Clone + Copy + PartialOrd;

    /// Vector type for axis-aligned bounds.
    type Vector: Clone;

    /// Diagonal bounds type.
    type Diag: DiagonalBounds<Scalar = Self::Scalar>;

    /// Get the minimum axis-aligned bound in each dimension.
    fn axis_min(&self) -> &Self::Vector;

    /// Get the maximum axis-aligned bound in each dimension.
    fn axis_max(&self) -> &Self::Vector;

    /// Get the number of dimensions.
    fn dim(&self) -> usize;

    /// Get the diagonal bounds for the k-th adjacent pair (k, k+1).
    ///
    /// Returns `None` if k >= dim - 1.
    fn diagonal_bounds(&self, pair_index: usize) -> Option<&Self::Diag>;

    /// Number of adjacent dimension pairs = dim - 1.
    fn num_diagonal_pairs(&self) -> usize {
        let d = self.dim();
        if d == 0 { 0 } else { d - 1 }
    }

    /// Check whether a point lies inside this octagon.
    ///
    /// A point is inside if it satisfies all axis-aligned bounds AND all
    /// diagonal bounds for adjacent dimension pairs.
    fn contains(&self, point: &[Self::Scalar]) -> Result<bool, OctagonError>;

    /// Compute the intersection of two octagons.
    ///
    /// The intersection is computed by taking the componentwise tighter bound
    /// for each constraint (max of lower bounds, min of upper bounds).
    /// The result is always an octagon (closure under intersection).
    ///
    /// Returns `OctagonError::Empty` if the intersection is empty.
    fn intersection(&self, other: &Self) -> Result<Self, OctagonError>;

    /// Compute the volume of this octagon.
    ///
    /// For 2D, this can be computed exactly. For higher dimensions,
    /// this uses a Monte Carlo approximation bounded by the bounding box volume.
    fn volume(&self) -> Result<Self::Scalar, OctagonError>;

    /// Soft containment probability: P(other inside self), smoothed by temperature.
    ///
    /// Uses sigmoid smoothing on each constraint to produce a differentiable
    /// containment score. Temperature controls sharpness (lower = harder).
    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, OctagonError>;

    /// Soft overlap probability: Vol(self intersection other) / Vol(self union other).
    fn overlap_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, OctagonError>;

    /// Convert this octagon to its bounding box (drop diagonal constraints).
    ///
    /// The bounding box is the smallest axis-aligned hyperrectangle containing
    /// the octagon. Since an octagon with no diagonal constraints is a box,
    /// this is always a valid outer approximation.
    fn to_bounding_box_bounds(&self) -> (Self::Vector, Self::Vector);
}
