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
/// # Paradigm Problem: The Crowding Effect in Taxonomies
///
/// Consider a taxonomy where "Animal" has many children: "Dog", "Cat", "Bird", "Fish", etc.
/// In traditional Euclidean embeddings, all these children cluster near the "Animal" center,
/// making them hard to distinguish. This is the **crowding problem**.
///
/// **The problem**: If "Animal" is at center (0.5, 0.5) and "Dog" is at (0.51, 0.51), the
/// Euclidean distance is small. But "Animal" is a very general concept (large box) while
/// "Dog" is specific (small box). They should be "farther apart" conceptually, even if
/// their centers are close.
///
/// **The solution**: Incorporate box size into distance. A large box (general concept) should
/// be farther from a small box (specific concept) than two boxes of similar size, even if
/// their centers are equally distant.
///
/// **Visual intuition**: Imagine two boxes with overlapping centers. One is huge (covers
/// the whole space), one is tiny (a small region). They're conceptually very different,
/// even though their centers coincide. Depth distance captures this by adding a term
/// proportional to the difference in their volumes.
///
/// # Research Background
///
/// Depth-based distance is introduced in **Yang & Chen (2025)**, "RegD: Achieving Hyperbolic-Like
/// Expressiveness with Arbitrary Euclidean Regions" (arXiv:2501.17518, Section 3.2).
///
/// **Key insight from the paper**: Traditional hyperbolic embeddings solve the crowding problem
/// through hyperbolic geometry. RegD shows that incorporating log-volume differences into
/// Euclidean distance achieves similar expressiveness without the computational complexity
/// of hyperbolic operations.
///
/// **Theoretical connection**: Hyperbolic space naturally models hierarchies because distance
/// grows exponentially with depth. RegD approximates this by making distance grow with
/// the log-volume difference, which captures the "depth" difference in a hierarchy.
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
/// - `subsume::ndarray_backend::distance::depth_distance()` - Exact implementation with log volumes
/// - `subsume::candle_backend::distance::depth_distance()` - Exact implementation with log volumes
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
/// use subsume::ndarray_backend::distance::depth_distance;
/// use subsume::ndarray_backend::NdarrayBox;
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
    // - `subsume::ndarray_backend::distance::depth_distance()`
    // - `subsume::candle_backend::distance::depth_distance()`
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
/// use subsume::distance::boundary_distance;
/// use subsume::ndarray_backend::NdarrayBox;
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
    //   gap_i = min(inner.min\[i\] - outer.min\[i\], outer.max\[i\] - inner.max\[i\])
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
/// ## Paradigm Problem: Two-View Knowledge Graphs
///
/// **The problem**: Real-world knowledge graphs have two types of entities:
/// - **Concepts** (like "Animal", "Mammal", "Location"): Benefit from box embeddings because
///   they have hierarchical relationships (Animal contains Mammal, Mammal contains Dog)
/// - **Entities** (like "Paris", "Einstein", "The Eiffel Tower"): May be better as vectors
///   because they're specific instances without clear hierarchies
///
/// **The challenge**: How do we score a triple like (Paris, instance_of, City) where "Paris"
/// is a vector and "City" is a box? We need a distance metric that works across representations.
///
/// **The solution**: Vector-to-box distance. If the "Paris" vector is inside the "City" box,
/// then Paris is a city. The distance measures how far the vector is from the box boundary.
///
/// **Visual intuition**: Imagine a box representing "City" in space. A vector representing
/// "Paris" is a point. If the point is inside the box, distance = 0 (Paris is a city).
/// If the point is outside, distance = minimum distance to the box boundary (how "far" is
/// Paris from being a city?).
///
/// **Step-by-step computation**:
/// 1. Check if vector is inside box: If yes, distance = 0
/// 2. If outside, find nearest point on box boundary
/// 3. Compute Euclidean distance from vector to that boundary point
///
/// This metric bridges concept box embeddings and entity vector embeddings,
/// enabling hybrid representations where concepts are boxes and entities are vectors.
///
/// # Research Background
///
/// Vector-to-box distance is introduced in **Huang et al. (2023)**, "Concept2Box: Joint Geometric
/// Embeddings for Learning Two-View Knowledge Graphs" (ACL 2023, Section 3.2).
///
/// **Key insight from the paper**: Real-world knowledge graphs often have heterogeneous entity types.
/// Concepts (like "Animal", "Mammal") benefit from box embeddings for subsumption, while specific
/// entities (like "Paris", "Einstein") may be better as vectors. The vector-to-box distance enables
/// scoring triples that mix these representations, allowing hybrid models that use the best
/// representation for each entity type.
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
    // See subsume and subsume-candle for concrete implementations
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

/// Compute Query2Box alpha-weighted distance from an entity point to a query box.
///
/// ## Mathematical Formulation (Ren et al., NeurIPS 2020)
///
/// The distance from an entity point `v` to a query box `q` is:
///
/// ```text
/// d(q, v) = d_out(q, v) + alpha * d_in(q, v)
/// ```
///
/// where:
/// - `d_out(q, v)` = per-dimension distance from entity to nearest box boundary,
///   summed over dimensions. Zero if the entity is inside the box.
/// - `d_in(q, v)` = per-dimension distance from entity to box center,
///   summed over dimensions. Zero if the entity is outside the box.
/// - `alpha < 1` (typically 0.02) penalizes inside-center distance less than
///   outside distance, so entities inside the box are scored closer than entities
///   the same Euclidean distance away but outside.
///
/// Unlike the Concept2Box vector-to-box distance (which returns 0 for points inside
/// the box), Query2Box uses `d_in` to discriminate among points that are all inside
/// the answer box -- points closer to the center are scored as better answers.
///
/// ## Parameters
///
/// - `query_center`: Center of the query box
/// - `query_offset`: Half-widths of the query box (non-negative)
/// - `entity_point`: The entity point
/// - `alpha`: Weight for inside distance (typically 0.02)
///
/// ## Returns
///
/// The weighted distance score. Lower = entity is a better match.
///
/// ## Errors
///
/// Returns [`BoxError::DimensionMismatch`] if vectors differ in length.
///
/// ## Example
///
/// ```rust
/// use subsume::distance::query2box_distance;
///
/// // Query box with center [5, 5] and half-width [2, 2] => box [3,7] x [3,7]
/// let center = [5.0, 5.0];
/// let offset = [2.0, 2.0];
///
/// // Entity at center: d_out=0, d_in=0
/// let d = query2box_distance(&center, &offset, &[5.0, 5.0], 0.02).unwrap();
/// assert!(d < 1e-6);
///
/// // Entity inside but off-center: only d_in contributes (scaled by alpha)
/// let d = query2box_distance(&center, &offset, &[4.0, 4.0], 0.02).unwrap();
/// assert!(d > 0.0 && d < 0.1); // small due to alpha=0.02
///
/// // Entity outside: d_out dominates
/// let d = query2box_distance(&center, &offset, &[10.0, 10.0], 0.02).unwrap();
/// assert!(d > 5.0); // d_out = (10-7) + (10-7) = 6
/// ```
///
/// ## References
///
/// - Ren et al. (ICLR 2020), "Query2Box: Reasoning over Knowledge Graphs in
///   Vector Space Using Box Embeddings"
pub fn query2box_distance(
    query_center: &[f32],
    query_offset: &[f32],
    entity_point: &[f32],
    alpha: f32,
) -> Result<f32, BoxError> {
    let d = query_center.len();
    if query_offset.len() != d || entity_point.len() != d {
        return Err(BoxError::DimensionMismatch {
            expected: d,
            actual: query_offset.len().max(entity_point.len()),
        });
    }

    let mut d_out = 0.0f32;
    let mut d_in = 0.0f32;

    for i in 0..d {
        let lo = query_center[i] - query_offset[i];
        let hi = query_center[i] + query_offset[i];
        let v = entity_point[i];

        if v < lo {
            d_out += lo - v;
        } else if v > hi {
            d_out += v - hi;
        } else {
            // Inside: distance to center.
            d_in += (v - query_center[i]).abs();
        }
    }

    Ok(d_out + alpha * d_in)
}

#[cfg(test)]
#[cfg(feature = "ndarray-backend")]
mod tests {
    use super::*;
    use crate::ndarray_backend::NdarrayBox;
    use ndarray::array;

    const TEMP: f32 = 1.0;
    const VOL_W: f32 = 0.1;
    const EPS: f32 = 1e-5;

    // ---- depth_distance ----

    #[test]
    fn depth_distance_identical_boxes() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let d = depth_distance(&a, &b, TEMP, VOL_W).unwrap();
        assert!(
            d.abs() < EPS,
            "identical boxes should have distance ~0, got {d}"
        );
    }

    #[test]
    fn depth_distance_nested_boxes() {
        let outer = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], TEMP).unwrap();
        let inner = NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], TEMP).unwrap();
        let d = depth_distance(&outer, &inner, TEMP, VOL_W).unwrap();
        // Box::distance for overlapping boxes returns 0 (min surface distance).
        // Volumes differ: 4.0 vs 1.0, so volume component > 0.
        assert!(
            d > 0.0,
            "nested boxes with different volumes should have positive distance, got {d}"
        );
        // The generic implementation uses |vol_a - vol_b| (not log), so volume_diff = 3.0.
        // Expected: 0 + 0.1 * 3.0 = 0.3
        let expected = VOL_W * (4.0 - 1.0);
        assert!(
            (d - expected).abs() < EPS,
            "nested: expected {expected}, got {d}"
        );
    }

    #[test]
    fn depth_distance_disjoint_boxes() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![5.0, 5.0], array![6.0, 6.0], TEMP).unwrap();
        let d = depth_distance(&a, &b, TEMP, VOL_W).unwrap();
        // Same volume => volume term = 0. Min surface distance = sqrt((5-1)^2 + (5-1)^2) = 4*sqrt(2).
        let gap_dist = ((5.0 - 1.0_f32).powi(2) * 2.0).sqrt();
        assert!(
            (d - gap_dist).abs() < EPS,
            "disjoint same-size boxes: distance should be {gap_dist}, got {d}"
        );
    }

    #[test]
    fn depth_distance_overlapping_boxes() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], TEMP).unwrap();
        let d = depth_distance(&a, &b, TEMP, VOL_W).unwrap();
        // Overlapping => Box::distance = 0. Same volume (4.0 each) => volume term = 0.
        assert!(
            d.abs() < EPS,
            "overlapping same-volume boxes: distance should be ~0, got {d}"
        );
    }

    #[test]
    fn depth_distance_single_dimension() {
        let a = NdarrayBox::new(array![0.0], array![1.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![3.0], array![4.0], TEMP).unwrap();
        let d = depth_distance(&a, &b, TEMP, VOL_W).unwrap();
        // Same volume (1.0 each). Surface gap = 3.0 - 1.0 = 2.0.
        assert!(
            (d - 2.0).abs() < EPS,
            "1D disjoint same-size: distance should be 2.0, got {d}"
        );
    }

    #[test]
    fn depth_distance_volume_weight_scales() {
        let outer = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], TEMP).unwrap();
        let inner = NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], TEMP).unwrap();
        let d_low = depth_distance(&outer, &inner, TEMP, 0.01).unwrap();
        let d_high = depth_distance(&outer, &inner, TEMP, 1.0).unwrap();
        assert!(
            d_high > d_low,
            "higher volume_weight should yield larger distance"
        );
    }

    // ---- boundary_distance ----
    //
    // NOTE: The generic boundary_distance uses a volume-ratio approximation:
    //   result = (1 - inner_vol / outer_vol) * 0.1
    // The exact gap-based computation lives in ndarray_backend::distance.

    #[test]
    fn boundary_distance_identical_boxes() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let d = boundary_distance(&a, &b, TEMP).unwrap();
        // Identical boxes: vol_ratio = 1.0, so (1 - 1) * 0.1 = 0.
        assert!(
            d.is_some(),
            "identical boxes should be considered contained"
        );
        assert!(
            d.unwrap().abs() < EPS,
            "identical boxes should have boundary distance ~0, got {:?}",
            d
        );
    }

    #[test]
    fn boundary_distance_nested_centered() {
        let outer = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], TEMP).unwrap();
        let inner = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], TEMP).unwrap();
        let d = boundary_distance(&outer, &inner, TEMP).unwrap();
        assert!(d.is_some(), "centered inner should be contained");
        let val = d.unwrap();
        // inner vol = 4, outer vol = 16, ratio = 0.25, result = (1 - 0.25) * 0.1 = 0.075
        let expected = (1.0 - 4.0 / 16.0) * 0.1;
        assert!(
            (val - expected).abs() < EPS,
            "nested: boundary distance should be {expected}, got {val}"
        );
    }

    #[test]
    fn boundary_distance_nested_smaller_inner_larger_distance() {
        // A smaller inner box (relative to outer) should yield a larger boundary distance
        // than a bigger inner box, because the volume ratio is smaller.
        let outer = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], TEMP).unwrap();
        let big_inner = NdarrayBox::new(array![0.5, 0.5], array![3.5, 3.5], TEMP).unwrap();
        let small_inner = NdarrayBox::new(array![1.5, 1.5], array![2.5, 2.5], TEMP).unwrap();
        let d_big = boundary_distance(&outer, &big_inner, TEMP)
            .unwrap()
            .unwrap();
        let d_small = boundary_distance(&outer, &small_inner, TEMP)
            .unwrap()
            .unwrap();
        assert!(
            d_small > d_big,
            "smaller inner should have larger boundary distance: small={d_small}, big={d_big}"
        );
    }

    #[test]
    fn boundary_distance_disjoint_returns_none() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![5.0, 5.0], array![6.0, 6.0], TEMP).unwrap();
        let d = boundary_distance(&a, &b, TEMP).unwrap();
        assert!(d.is_none(), "disjoint boxes should return None");
    }

    #[test]
    fn boundary_distance_overlapping_not_contained_returns_none() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], TEMP).unwrap();
        let d = boundary_distance(&a, &b, TEMP).unwrap();
        assert!(
            d.is_none(),
            "overlapping but not contained should return None"
        );
    }

    #[test]
    fn boundary_distance_touching_at_boundary() {
        // Inner box shares both faces in dim 0 with outer => vol_ratio = inner/outer.
        let outer = NdarrayBox::new(array![0.0, 0.0], array![4.0, 4.0], TEMP).unwrap();
        let inner = NdarrayBox::new(array![0.0, 1.0], array![4.0, 3.0], TEMP).unwrap();
        let d = boundary_distance(&outer, &inner, TEMP).unwrap();
        assert!(d.is_some(), "face-touching inner should be contained");
        let val = d.unwrap();
        // inner vol = 4*2 = 8, outer vol = 16, ratio = 0.5, result = 0.5 * 0.1 = 0.05
        let expected = (1.0 - 8.0 / 16.0) * 0.1;
        assert!(
            (val - expected).abs() < EPS,
            "touching at boundary: distance should be {expected}, got {val}"
        );
    }

    #[test]
    fn boundary_distance_single_dimension() {
        let outer = NdarrayBox::new(array![0.0], array![10.0], TEMP).unwrap();
        let inner = NdarrayBox::new(array![3.0], array![7.0], TEMP).unwrap();
        let d = boundary_distance(&outer, &inner, TEMP).unwrap();
        assert!(d.is_some());
        let val = d.unwrap();
        // inner vol = 4, outer vol = 10, ratio = 0.4, result = 0.6 * 0.1 = 0.06
        let expected = (1.0 - 4.0 / 10.0) * 0.1;
        assert!(
            (val - expected).abs() < EPS,
            "1D nested: boundary distance should be {expected}, got {val}"
        );
    }

    // ---- depth_similarity ----

    #[test]
    fn depth_similarity_identical_boxes() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let s = depth_similarity(&a, &b, TEMP, VOL_W).unwrap();
        assert!(
            (s - 1.0).abs() < EPS,
            "identical boxes should have similarity ~1.0, got {s}"
        );
    }

    #[test]
    fn depth_similarity_range_zero_to_one() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![10.0, 10.0], array![20.0, 20.0], TEMP).unwrap();
        let s = depth_similarity(&a, &b, TEMP, VOL_W).unwrap();
        assert!(
            s > 0.0 && s <= 1.0,
            "similarity should be in (0, 1], got {s}"
        );
    }

    #[test]
    fn depth_similarity_closer_boxes_more_similar() {
        let a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], TEMP).unwrap();
        let near = NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], TEMP).unwrap();
        let far = NdarrayBox::new(array![10.0, 10.0], array![11.0, 11.0], TEMP).unwrap();
        let s_near = depth_similarity(&a, &near, TEMP, VOL_W).unwrap();
        let s_far = depth_similarity(&a, &far, TEMP, VOL_W).unwrap();
        assert!(
            s_near > s_far,
            "nearby box should be more similar than distant box: near={s_near}, far={s_far}"
        );
    }

    #[test]
    fn depth_similarity_nested_less_than_one() {
        // Nested boxes have same center but different volumes => distance > 0 => similarity < 1.
        let outer = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], TEMP).unwrap();
        let inner = NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], TEMP).unwrap();
        let s = depth_similarity(&outer, &inner, TEMP, VOL_W).unwrap();
        assert!(
            s < 1.0,
            "nested boxes with different volumes should have similarity < 1, got {s}"
        );
        assert!(
            s > 0.0,
            "nested boxes should still have positive similarity, got {s}"
        );
    }

    #[test]
    fn depth_similarity_single_dimension() {
        let a = NdarrayBox::new(array![0.0], array![1.0], TEMP).unwrap();
        let b = NdarrayBox::new(array![0.0], array![1.0], TEMP).unwrap();
        let s = depth_similarity(&a, &b, TEMP, VOL_W).unwrap();
        assert!(
            (s - 1.0).abs() < EPS,
            "identical 1D boxes should have similarity ~1.0, got {s}"
        );
    }

    // ---- query2box_distance ----

    #[test]
    fn query2box_entity_inside_box() {
        // Entity at center => d_out=0, d_in=0 => distance=0.
        let d = query2box_distance(&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], 0.02).unwrap();
        assert!(
            d.abs() < EPS,
            "entity at center: distance should be 0, got {d}"
        );

        // Entity inside but not at center: d_out=0, d_in > 0.
        let d2 = query2box_distance(&[1.0, 1.0], &[1.0, 1.0], &[0.5, 0.5], 0.02).unwrap();
        assert!(
            d2 > 0.0,
            "entity inside but off-center should have positive distance"
        );
        // d_in = |0.5-1| + |0.5-1| = 1.0, d_out = 0 => d = 0.02 * 1.0 = 0.02
        assert!((d2 - 0.02).abs() < EPS, "expected 0.02, got {d2}");
    }

    #[test]
    fn query2box_entity_outside_box() {
        // Entity at (5, 5), box [0,2]x[0,2].
        let d = query2box_distance(&[1.0, 1.0], &[1.0, 1.0], &[5.0, 5.0], 0.02).unwrap();
        // d_out = (5-2) + (5-2) = 6, d_in = 0
        assert!(
            (d - 6.0).abs() < EPS,
            "entity outside: expected 6.0, got {d}"
        );
    }

    #[test]
    fn query2box_entity_on_boundary() {
        // Entity exactly on the max boundary.
        let d = query2box_distance(&[1.0, 1.0], &[1.0, 1.0], &[2.0, 2.0], 0.02).unwrap();
        // On boundary: v=2 == hi=2, so it's "inside" (not outside).
        // d_in = |2-1| + |2-1| = 2.0, d_out = 0 => d = 0.02 * 2.0 = 0.04
        assert!(
            (d - 0.04).abs() < EPS,
            "on boundary: expected 0.04, got {d}"
        );
    }

    #[test]
    fn query2box_alpha_one_reduces_to_center_distance() {
        // alpha=1: d = d_out + d_in, which for inside points is just L1 to center.
        let d = query2box_distance(&[1.0, 1.0], &[1.0, 1.0], &[0.5, 0.5], 1.0).unwrap();
        // d_in = |0.5-1| + |0.5-1| = 1.0
        assert!(
            (d - 1.0).abs() < EPS,
            "alpha=1, inside: expected L1 to center = 1.0, got {d}"
        );
    }

    #[test]
    fn query2box_alpha_zero_reduces_to_boundary_only() {
        // alpha=0: d = d_out only; inside entities get distance 0.
        let d = query2box_distance(&[1.0, 1.0], &[1.0, 1.0], &[0.5, 0.5], 0.0).unwrap();
        assert!(d.abs() < EPS, "alpha=0, inside: expected 0, got {d}");

        let d2 = query2box_distance(&[1.0, 1.0], &[1.0, 1.0], &[5.0, 5.0], 0.0).unwrap();
        assert!(
            (d2 - 6.0).abs() < EPS,
            "alpha=0, outside: expected 6.0, got {d2}"
        );
    }

    #[test]
    fn query2box_dim_mismatch() {
        assert!(query2box_distance(&[1.0], &[1.0, 1.0], &[0.5], 0.02).is_err());
    }
}
