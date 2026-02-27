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

/// A cone embedding as a Cartesian product of `d` independent 2D angular sectors.
///
/// Each dimension has an axis angle in \[-pi, pi\] and an aperture (half-width)
/// in \[0, pi\]. Subsumption is measured by per-dimension angular containment,
/// and the scores are summed across dimensions.
///
/// Cones support negation: the complement of a cone is a cone (per-dimension axis
/// shift by pi, aperture becomes pi - aperture). This closure under complementation
/// enables modeling FOL operations including conjunction, disjunction, and negation.
///
/// Reference: Zhang & Wang (2021), "ConE: Cone Embeddings for Multi-Hop Reasoning
/// over Knowledge Graphs" (NeurIPS 2021).
pub trait Cone: Sized {
    /// Scalar type for angles, scores, etc.
    type Scalar: Clone + Copy + PartialOrd;

    /// Vector type for per-dimension axes and apertures.
    type Vector: Clone;

    /// Get the per-dimension axis angles.
    /// Each element is in \[-pi, pi\].
    fn axes(&self) -> &Self::Vector;

    /// Get the per-dimension apertures (half-widths).
    /// Each element is in \[0, pi\].
    fn apertures(&self) -> &Self::Vector;

    /// Get the number of dimensions.
    fn dim(&self) -> usize;

    /// Compute the ConE distance score between an entity cone and this query cone.
    ///
    /// Uses the per-dimension scoring from ConE (Zhang & Wang, 2021):
    ///
    /// ```text
    /// distance_to_axis[i] = |sin((entity_axis[i] - query_axis[i]) / 2)|
    /// distance_base[i]    = |sin(query_aperture[i] / 2)|
    /// ```
    ///
    /// Points inside the sector contribute `cen * distance_in`; points outside
    /// contribute `distance_out`. The total is summed across dimensions.
    ///
    /// Lower distance = better containment. The `cen` parameter (typically 0.02)
    /// weights the inside distance relative to outside distance.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if cones have different dimensions.
    fn cone_distance(
        &self,
        entity: &Self,
        cen: Self::Scalar,
    ) -> Result<Self::Scalar, ConeError>;

    /// Compute the complement (negation) of this cone.
    ///
    /// Per-dimension:
    /// - axis\[i\] shifts by pi (positive axes subtract pi, negative axes add pi),
    ///   keeping the result in \[-pi, pi\].
    /// - aperture\[i\] becomes pi - aperture\[i\].
    ///
    /// This closure under complementation is the key advantage over boxes.
    fn complement(&self) -> Self;

    /// Compute the intersection of two cones.
    ///
    /// Uses the closed-form circular mean for axes (attention-weighted average in
    /// Cartesian coordinates, then atan2 back to angle) and per-dimension minimum
    /// for apertures.
    ///
    /// When `weights` is `None`, equal weights are used.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if cones have different dimensions.
    fn intersection(&self, other: &Self) -> Result<Self, ConeError>;

    /// Apply a relation projection to this cone.
    ///
    /// Per-dimension:
    /// - axis\[i\] += relation_axis\[i\] (modular addition, wrapped to \[-pi, pi\])
    /// - aperture\[i\] = clamp(aperture\[i\] + relation_aperture\[i\], 0, pi)
    ///
    /// The relation transforms the cone's position and width in each angular sector.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if dimensions don't match.
    fn project(
        &self,
        relation_axes: &Self::Vector,
        relation_apertures: &Self::Vector,
    ) -> Result<Self, ConeError>;
}

/// Errors that can occur during cone operations.
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

    /// Invalid aperture: must be in [0, pi].
    #[error("Invalid aperture: {value} (must be in [0, pi])")]
    InvalidAperture {
        /// The invalid aperture value.
        value: f64,
    },

    /// Internal error from array/tensor operations.
    #[error("Internal error: {0}")]
    Internal(String),
}
