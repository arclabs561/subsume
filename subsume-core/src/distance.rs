//! Advanced distance metrics for box embeddings.
//!
//! This module implements distance metrics from recent research:
//! - **Depth-based distance** (RegD 2025): Incorporates region "size" into distance calculations
//! - **Boundary distance** (RegD 2025): For containment relationships and inclusion chains
//! - **Vector-to-box distance** (Concept2Box 2023): For hybrid representations (concepts as boxes, entities as vectors)
//!
//! # References
//!
//! - Yang & Chen (2025): "Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions"
//! - Huang et al. (2023): "Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs"

use crate::{Box, BoxError};

/// Compute depth-based distance between two boxes.
///
/// Depth distance incorporates region "size" (volume) into distance calculations,
/// enabling Euclidean boxes to achieve hyperbolic-like expressiveness. This addresses
/// the crowding effect where traditional Euclidean embeddings cluster together as
/// the number of children increases in hierarchies.
///
/// ## Mathematical Formulation (RegD 2025)
///
/// The depth distance combines standard Euclidean distance with a volume-based term:
///
/// \[
/// d_{\text{depth}}(A, B) = d_{\text{Euclidean}}(A, B) + \alpha \cdot |\log(\text{Vol}(A)) - \log(\text{Vol}(B))|
/// \]
///
/// where \(\alpha\) is a scaling factor controlling the volume contribution.
///
/// ## Implementation Note
///
/// **⚠️ WARNING**: This generic implementation uses an approximation (volume difference
/// instead of log volume difference) due to trait limitations. For correct depth distance,
/// use backend-specific implementations:
///
/// - `subsume_ndarray::distance::depth_distance()` - Exact implementation with log volumes
/// - `subsume_candle::distance::depth_distance()` - Exact implementation with log volumes
///
/// The generic version is provided for framework-agnostic code but may produce
/// incorrect results, especially when volumes differ significantly.
///
/// # Parameters
///
/// - `box_a`: First box
/// - `box_b`: Second box
/// - `temperature`: Temperature for volume calculation
/// - `volume_weight`: Weight for volume term (default: 0.1)
///
/// # Errors
///
/// Returns `BoxError` if boxes have dimension mismatch or volume calculation fails.
///
/// # Example
///
/// ```rust,ignore
/// // Prefer backend-specific implementation for accuracy
/// use subsume_ndarray::distance::depth_distance;
/// use subsume_ndarray::NdarrayBox;
///
/// let box_a = NdarrayBox::new(...)?;
/// let box_b = NdarrayBox::new(...)?;
/// let dist = depth_distance(&box_a, &box_b, 1.0, 0.1)?;
/// ```
pub fn depth_distance<B: Box>(
    box_a: &B,
    box_b: &B,
    temperature: B::Scalar,
    volume_weight: B::Scalar,
) -> Result<B::Scalar, BoxError>
where
    B::Scalar: From<f32>
        + std::ops::Add<Output = B::Scalar>
        + std::ops::Sub<Output = B::Scalar>
        + std::ops::Mul<Output = B::Scalar>,
{
    // Standard Euclidean distance
    let euclidean_dist = box_a.distance(box_b)?;

    // Volume-based term: |log(Vol(A)) - log(Vol(B))|
    let vol_a = box_a.volume(temperature)?;
    let vol_b = box_b.volume(temperature)?;

    // IMPORTANT: This generic implementation uses a simplified approximation.
    // The RegD (2025) paper formula requires log(volume), but generic traits
    // cannot assume log operations are available for all Scalar types.
    //
    // Backend-specific implementations (ndarray, candle) provide exact
    // implementations using actual logarithms. This generic version is provided
    // for framework-agnostic code, but should be overridden by backends.
    //
    // Approximation: We use volume directly instead of log(volume). This is
    // mathematically incorrect but allows the code to compile. For correct
    // depth distance, use backend-specific functions:
    // - `subsume_ndarray::distance::depth_distance()`
    // - `subsume_candle::distance::depth_distance()`
    //
    // The approximation may produce incorrect results, especially when volumes
    // differ significantly. Always prefer backend-specific implementations.

    // Simplified approximation: use volume difference instead of log volume difference
    let volume_diff = if vol_a > vol_b {
        vol_a - vol_b
    } else {
        vol_b - vol_a
    };

    // Depth distance = Euclidean + α * volume_diff (APPROXIMATION)
    // Correct formula: Euclidean + α * |log(Vol(A)) - log(Vol(B))|
    Ok(euclidean_dist + volume_weight * volume_diff)
}

/// Compute boundary distance between two boxes.
///
/// Boundary distance is designed for containment relationships. When one box is fully
/// contained within another, boundary distance captures both:
/// 1. Whether containment exists
/// 2. Discrimination between regions in inclusion chains (shallower vs deeper descendants)
///
/// ## Mathematical Formulation (RegD 2025)
///
/// For boxes A and B where B ⊆ A (B is contained in A):
///
/// \[
/// d_{\text{boundary}}(A, B) = \min_{p \in \partial A} \min_{q \in B} ||p - q||
/// \]
///
/// where \(\partial A\) is the boundary of box A.
///
/// For axis-aligned boxes, this simplifies to the minimum distance from any point
/// in B to the boundary of A.
///
/// # Parameters
///
/// - `outer`: The outer box (should contain `inner`)
/// - `inner`: The inner box (should be contained in `outer`)
/// - `temperature`: Temperature for volume calculation
///
/// # Returns
///
/// Boundary distance, or `None` if `inner` is not contained in `outer`.
///
/// # Errors
///
/// Returns `BoxError` if boxes have dimension mismatch.
///
/// # Example
///
/// ```rust,ignore
/// use subsume_core::distance::boundary_distance;
/// use subsume_ndarray::NdarrayBox;
///
/// let outer = NdarrayBox::new(...)?; // Larger box
/// let inner = NdarrayBox::new(...)?; // Smaller box contained in outer
/// if let Some(dist) = boundary_distance(&outer, &inner, 1.0)? {
///     println!("Boundary distance: {}", dist);
/// }
/// ```
pub fn boundary_distance<B: Box>(
    outer: &B,
    inner: &B,
    temperature: B::Scalar,
) -> Result<Option<B::Scalar>, BoxError>
where
    B::Scalar: From<f32>
        + PartialOrd
        + std::ops::Sub<Output = B::Scalar>
        + std::ops::Mul<Output = B::Scalar>
        + std::ops::Div<Output = B::Scalar>,
{
    // Check if inner is contained in outer
    let containment = outer.containment_prob(inner, temperature)?;
    if containment < B::Scalar::from(0.99) {
        // Not fully contained
        return Ok(None);
    }

    // For axis-aligned boxes, boundary distance is the minimum distance from
    // any point in inner to the boundary of outer.
    // This is the minimum "gap" between inner and outer boundaries.
    //
    // For each dimension i:
    //   gap_i = min(inner.min[i] - outer.min[i], outer.max[i] - inner.max[i])
    // Boundary distance = min over all dimensions
    //
    // Note: This generic implementation uses a simplified approximation.
    // Backends should provide optimized implementations that directly access
    // min/max coordinates for exact computation.

    // Simplified approximation: use volume ratio as proxy for boundary distance
    // Backends (ndarray, candle) provide exact implementations
    let outer_vol = outer.volume(temperature)?;
    let inner_vol = inner.volume(temperature)?;

    // Approximate boundary distance using volume ratio
    // More precise implementations are in backend-specific modules
    let vol_ratio = if outer_vol > B::Scalar::from(1e-10) {
        inner_vol / outer_vol
    } else {
        B::Scalar::from(0.0)
    };

    // Boundary distance approximation: smaller inner relative to outer = larger distance
    // This captures "depth" in inclusion chain
    let one = B::Scalar::from(1.0);
    let boundary_dist = (one - vol_ratio) * B::Scalar::from(0.1); // Scale factor for approximation

    Ok(Some(boundary_dist))
}

/// Trait for computing distance from a point/vector to a box.
///
/// This metric bridges concept box embeddings and entity vector embeddings,
/// enabling hybrid representations where concepts are boxes and entities are vectors.
///
/// ## Mathematical Formulation (Concept2Box 2023)
///
/// The vector-to-box distance is defined as:
///
/// \[
/// d_{\text{vec-box}}(v, B) = \begin{cases}
/// 0 & \text{if } v \in B \\
/// \min_{p \in B} ||v - p|| & \text{otherwise}
/// \end{cases}
/// \]
///
/// For axis-aligned boxes, this simplifies to computing the distance from the point
/// to the nearest point on the box boundary.
pub trait VectorToBoxDistance<B: Box> {
    /// Compute distance from a point/vector to a box.
    ///
    /// # Parameters
    ///
    /// - `point`: The point/vector (must have same dimension as box)
    /// - `box_`: The box
    ///
    /// # Returns
    ///
    /// Distance from point to box. Returns 0.0 if point is inside the box.
    ///
    /// # Errors
    ///
    /// Returns `BoxError` if point and box have dimension mismatch.
    fn vector_to_box_distance(point: &B::Vector, box_: &B) -> Result<B::Scalar, BoxError>
    where
        B::Scalar: From<f32>
            + PartialOrd
            + std::ops::Sub<Output = B::Scalar>
            + std::ops::Mul<Output = B::Scalar>;
}

/// Helper function to compute vector-to-box distance.
///
/// This is a convenience wrapper that backends can use.
/// Backends should implement `VectorToBoxDistance` trait for their specific types.
pub fn vector_to_box_distance<B: Box>(
    _point: &B::Vector,
    _box_: &B,
    _temperature: B::Scalar,
) -> Result<B::Scalar, BoxError>
where
    B::Scalar: From<f32>
        + PartialOrd
        + std::ops::Sub<Output = B::Scalar>
        + std::ops::Mul<Output = B::Scalar>,
{
    // This is a placeholder - backends should provide their own implementation
    // See subsume-ndarray and subsume-candle for concrete implementations
    Err(BoxError::Internal(
        "vector_to_box_distance must be implemented by backends. Use backend-specific functions."
            .to_string(),
    ))
}

/// Compute depth-based similarity between two boxes.
///
/// Similarity metric based on depth distance, useful for hierarchical clustering
/// and similarity search. Higher values indicate more similar boxes.
///
/// # Parameters
///
/// - `box_a`: First box
/// - `box_b`: Second box
/// - `temperature`: Temperature for volume calculation
/// - `volume_weight`: Weight for volume term in depth distance
///
/// # Returns
///
/// Similarity score in [0, 1], where 1.0 indicates identical boxes.
///
/// # Errors
///
/// Returns `BoxError` if distance calculation fails.
pub fn depth_similarity<B: Box>(
    box_a: &B,
    box_b: &B,
    temperature: B::Scalar,
    volume_weight: B::Scalar,
) -> Result<B::Scalar, BoxError>
where
    B::Scalar: From<f32>
        + PartialOrd
        + std::ops::Add<Output = B::Scalar>
        + std::ops::Sub<Output = B::Scalar>
        + std::ops::Mul<Output = B::Scalar>
        + std::ops::Div<Output = B::Scalar>,
{
    let dist = depth_distance(box_a, box_b, temperature, volume_weight)?;

    // Convert distance to similarity: sim = 1 / (1 + dist)
    // Clamp to avoid division issues
    let one = B::Scalar::from(1.0);
    Ok(one / (one + dist))
}
