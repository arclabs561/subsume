//! Error types for cone embeddings.
//!
//! Cone embedding implementations live in [`crate::ndarray_backend::ndarray_cone`].

/// Errors that can occur during cone operations.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConeError {
    /// Cones have different dimensions.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions.
        actual: usize,
    },

    /// Invalid bounds (e.g., NaN values).
    #[error("Invalid bounds: {reason}")]
    InvalidBounds {
        /// Description of the invalid bounds.
        reason: String,
    },
}
