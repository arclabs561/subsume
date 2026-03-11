//! Cone embeddings for modeling subsumption with negation support.
//!
//! # Overview
//!
//! Cone embeddings represent entities/queries as **Cartesian products of 2D angular
//! sectors**. Each dimension `i` has two parameters:
//!
//! - **axis\[i\]**: the center angle of the sector, in \[-pi, pi\].
//! - **aperture\[i\]**: the half-width of the sector, in \[0, pi\].
//!
//! A cone with `d` dimensions therefore lives in `(S^1)^d` -- a `d`-torus of circles.
//! Subsumption is modeled through **per-dimension angular containment**: cone A
//! subsumes cone B when, in every dimension, B's angular position falls within A's
//! sector.
//!
//! # Cones vs Boxes
//!
//! | Property | Boxes | Cones |
//! |---|---|---|
//! | Geometry | Axis-aligned hyperrectangles | Cartesian products of 2D angular sectors |
//! | Negation | Complement is **not** a box | Complement **is** a cone |
//! | FOL support | Conjunction, disjunction | Conjunction, disjunction, **negation** |
//! | Containment | Volume ratio | Per-dimension angular distance vs aperture |
//!
//! Cones are preferred when the task requires first-order logic reasoning with negation
//! (e.g., multi-hop KG reasoning with NOT). Boxes are preferred for axis-aligned
//! containment hierarchies where each dimension has independent semantics.
//!
//! # Mathematical Foundations
//!
//! **Scoring** (per-dimension, then summed):
//!
//! ```text
//! distance_to_axis[i] = |sin((entity_axis[i] - query_axis[i]) / 2)|
//! distance_base[i]    = |sin(query_aperture[i] / 2)|
//! ```
//!
//! Points inside the cone (distance_to_axis < distance_base) contribute an "inside"
//! distance; points outside contribute an "outside" distance based on angular distance
//! to the nearest cone boundary. The total score sums across dimensions.
//!
//! **Complement** (per-dimension negation):
//! - axis\[i\] shifts by pi (wrapping to \[-pi, pi\]).
//! - aperture\[i\] becomes pi - aperture\[i\].
//!
//! **Intersection** (attention-weighted circular mean for axes, gated minimum for
//! apertures). For this library (no autodiff), we use the closed-form weighted
//! circular mean: convert axes to (cos, sin), take a weighted average, then recover
//! the angle via atan2. Apertures use the per-dimension minimum.
//!
//! **Projection** (relation application): per-dimension rotation of the axis (addition
//! mod 2*pi) and scaling of the aperture. This is the paper's mechanism for multi-hop
//! reasoning.
//!
//! # Reference
//!
//! Zhang & Wang (2021), "ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge
//! Graphs" (NeurIPS 2021). This implementation faithfully follows the ConE
//! parameterization: Cartesian products of 2D angular sectors with per-dimension
//! scoring, negation, intersection, and projection.
//!
//! # Related Work
//!
//! - Yu et al. (2023), "Shadow Cones" -- generalizes entailment cones with
//!   projection-based partial orders in the Poincare ball
//! - Ozcep, Leemhuis, Wolter (2023, JAIR), "Embedding Ontologies in ALC by
//!   Axis-Aligned Cones" -- proves axis-aligned cones are complete for ALC
//! - Kharbanda et al. (2025, IEEE ICDE), "RConE: Rough Cone Embeddings" -- rough-set
//!   cones with lower/upper approximations
//! - Nguyen et al. (2023, EACL), "CylE: Cylinder Embeddings" -- extends 2D cones to
//!   3D cylinders for multi-hop reasoning

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
