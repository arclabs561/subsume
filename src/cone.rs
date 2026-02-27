//! Cone embeddings for modeling subsumption with negation support.
//!
//! # Overview
//!
//! Cone embeddings represent entities as angular cones in d-dimensional space. Each cone
//! is defined by an **apex** (center point on the unit sphere), an **axis** (direction
//! vector), and an **aperture** (half-angle in radians). Subsumption is modeled through
//! angular containment: cone A subsumes cone B when B's angular extent falls entirely
//! within A's angular extent.
//!
//! # Cones vs Boxes
//!
//! | Property | Boxes | Cones |
//! |---|---|---|
//! | Geometry | Axis-aligned hyperrectangles | Angular sectors on the sphere |
//! | Negation | Complement is **not** a box | Complement **is** a cone |
//! | FOL support | Conjunction, disjunction | Conjunction, disjunction, **negation** |
//! | Containment | Volume ratio | Angular distance vs aperture |
//!
//! Cones are preferred when the task requires first-order logic reasoning with negation
//! (e.g., multi-hop KG reasoning with NOT). Boxes are preferred for axis-aligned
//! containment hierarchies where each dimension has independent semantics.
//!
//! # Mathematical Foundations
//!
//! **Containment**: Cone A contains cone B when the angular distance between their axes
//! plus B's aperture is less than A's aperture:
//!
//! ```text
//! angular_dist(A, B) = arccos(dot(axis_A, axis_B) / (|axis_A| * |axis_B|))
//! P(B inside A) = sigmoid((aperture_A - angular_dist(A, B) - aperture_B) / temperature)
//! ```
//!
//! **Complement**: The complement of a cone with aperture alpha has aperture pi - alpha,
//! with the same apex and negated axis. This closure under complementation is the key
//! advantage over boxes.
//!
//! **Intersection**: The intersection of two cones whose axes are coplanar can be
//! approximated by a cone whose axis bisects the angular region of overlap and whose
//! aperture covers that overlap region.
//!
//! # Reference
//!
//! Inspired by Zhang & Wang (2021), "ConE: Cone Embeddings for Multi-Hop Reasoning
//! over Knowledge Graphs" (NeurIPS 2021). Note: this implementation uses a single
//! angular cone in d-dimensional space (apex + axis + aperture), whereas ConE uses
//! Cartesian products of 2D angular sectors with per-dimension scoring and learned
//! projection operators. The negation-closure property is shared; the parameterization
//! and scoring differ.
//!
//! # Related Work and Future Directions
//!
//! - Yu et al. (2023), "Shadow Cones" -- generalizes entailment cones with
//!   projection-based partial orders in the Poincare ball; richer than nested containment
//! - Ozcep, Leemhuis, Wolter (2023, JAIR), "Embedding Ontologies in ALC by
//!   Axis-Aligned Cones" -- proves axis-aligned cones are complete for the ALC description
//!   logic fragment (conjunction, disjunction, full negation)
//! - Kharbanda et al. (2025, IEEE ICDE), "RConE: Rough Cone Embeddings" -- rough-set
//!   cones with lower/upper approximations for uncertain or multi-modal data
//! - Nguyen et al. (2023, EACL), "CylE: Cylinder Embeddings" -- extends 2D cones to
//!   3D cylinders for multi-hop reasoning with additional capacity

/// A cone embedding in d-dimensional space.
///
/// Cones model subsumption relationships through angular containment.
/// Unlike boxes, cones support negation: the complement of a cone is a cone.
/// This enables modeling FOL operations (conjunction, disjunction, negation).
///
/// Reference: Zhang & Wang (2021), "ConE: Cone Embeddings for Multi-Hop
/// Reasoning over Knowledge Graphs" (NeurIPS 2021)
pub trait Cone: Sized {
    /// Scalar type for probabilities, angles, etc.
    type Scalar: Clone + Copy + PartialOrd;

    /// Vector type for apex and axis.
    type Vector: Clone;

    /// Get the apex (origin point) of the cone.
    fn apex(&self) -> &Self::Vector;

    /// Get the axis direction (unit vector) of the cone.
    fn axis(&self) -> &Self::Vector;

    /// Get the aperture (half-angle) of the cone in radians.
    /// Larger aperture = more general concept.
    /// Valid range: (0, pi).
    fn aperture(&self) -> Self::Scalar;

    /// Get the number of dimensions.
    fn dim(&self) -> usize;

    /// Compute containment probability: P(other is inside self).
    ///
    /// Based on angular distance between axes relative to apertures:
    ///
    /// ```text
    /// P(other inside self) = sigmoid((self.aperture - angular_dist - other.aperture) / temperature)
    /// ```
    ///
    /// Returns a value in [0, 1]. A value near 1.0 means `self` subsumes `other`.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if cones have different dimensions.
    fn containment_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, ConeError>;

    /// Compute the intersection of two cones.
    ///
    /// The intersection cone has an axis that bisects the overlap region and an
    /// aperture covering that region. When the cones do not overlap, returns a
    /// degenerate cone with zero aperture.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if cones have different dimensions.
    fn intersection(&self, other: &Self) -> Result<Self, ConeError>;

    /// Compute the complement (negation) of this cone.
    ///
    /// The complement of a cone with aperture alpha is a cone with aperture pi - alpha
    /// and negated axis. This closure under complementation is what distinguishes cones
    /// from boxes for FOL reasoning.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::InvalidAperture`] if the resulting aperture is out of range.
    fn complement(&self) -> Result<Self, ConeError>;

    /// Compute overlap probability between two cones.
    ///
    /// Measures how much angular extent the cones share relative to their combined
    /// angular extent.
    ///
    /// # Errors
    ///
    /// Returns [`ConeError::DimensionMismatch`] if cones have different dimensions.
    fn overlap_prob(
        &self,
        other: &Self,
        temperature: Self::Scalar,
    ) -> Result<Self::Scalar, ConeError>;
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

    /// Invalid aperture: must be in (0, pi).
    #[error("Invalid aperture: {value} (must be in (0, pi))")]
    InvalidAperture {
        /// The invalid aperture value.
        value: f64,
    },

    /// Axis vector has zero norm (cannot determine direction).
    #[error("Zero-norm axis vector")]
    ZeroAxis,

    /// Internal error from array/tensor operations.
    #[error("Internal error: {0}")]
    Internal(String),
}
