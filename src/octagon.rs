//! Error types for octagon embeddings.
//!
//! Octagon embedding implementations live in [`crate::ndarray_backend::ndarray_octagon`].

/// Errors that can occur during octagon operations.
#[non_exhaustive]
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
}
