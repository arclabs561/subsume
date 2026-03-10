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
//!
//! # Examples
//!
//! ```rust
//! use ndarray::array;
//! use subsume::ndarray_backend::ndarray_octagon::{NdarrayDiagBounds, NdarrayOctagon};
//!
//! // 2D octagon: box [0,4]x[0,4] with diagonal cuts
//! let oct = NdarrayOctagon::new(
//!     array![0.0, 0.0],
//!     array![4.0, 4.0],
//!     vec![NdarrayDiagBounds {
//!         sum_min: 2.0, sum_max: 6.0,
//!         diff_min: -2.0, diff_max: 2.0,
//!     }],
//! ).unwrap();
//!
//! // Center (2,2) is inside; corner (0.1, 0.1) violates x+y >= 2
//! assert!(oct.contains(&[2.0, 2.0]).unwrap());
//! assert!(!oct.contains(&[0.1, 0.1]).unwrap());
//!
//! // Volume is smaller than the bounding box
//! let box_only = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![4.0, 4.0]).unwrap();
//! assert!(oct.volume().unwrap() < box_only.volume().unwrap());
//! ```

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
