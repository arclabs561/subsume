//! Backend-specific distance implementations for ndarray.

use crate::NdarrayBox;
use ndarray::Array1;
use subsume_core::{Box, BoxError};

/// Compute distance from a point (vector) to a box.
///
/// This implements the Concept2Box (2023) vector-to-box distance metric
/// for hybrid representations where concepts are boxes and entities are vectors.
///
/// ## Mathematical Formulation
///
/// For axis-aligned boxes, the distance from point v to box B is:
///
/// \[
/// d(v, B) = \sqrt{\sum_{i=1}^{d} \max(0, \min(v_i, B_{\min,i}) - B_{\min,i})^2 + \max(0, B_{\max,i} - \max(v_i, B_{\min,i}))^2}
/// \]
///
/// Simplified: for each dimension i:
/// - If v\[i\] < min\[i\]: distance contribution = (min\[i\] - v\[i\])²
/// - If v\[i\] > max\[i\]: distance contribution = (v\[i\] - max\[i\])²
/// - Otherwise: distance contribution = 0 (point is inside box in this dimension)
///
/// # Parameters
///
/// - `point`: The point/vector as `Array1<f32>`
/// - `box_`: The box
///
/// # Returns
///
/// Distance from point to box. Returns 0.0 if point is inside the box.
///
/// # Errors
///
/// Returns `BoxError` if point and box have dimension mismatch.
///
/// # Example
///
/// ```rust
/// use subsume_ndarray::{NdarrayBox, distance::vector_to_box_distance};
/// use ndarray::array;
///
/// let box_ = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
/// let point = array![0.5, 0.5]; // Point inside box
/// let dist = vector_to_box_distance(&point, &box_).unwrap();
/// assert_eq!(dist, 0.0);
/// ```
pub fn vector_to_box_distance(point: &Array1<f32>, box_: &NdarrayBox) -> Result<f32, BoxError> {
    if point.len() != box_.dim() {
        return Err(BoxError::DimensionMismatch {
            expected: box_.dim(),
            actual: point.len(),
        });
    }

    // For axis-aligned boxes, compute distance to nearest point on box boundary
    let mut dist_sq = 0.0;
    for i in 0..box_.dim() {
        let point_val = point[i];
        let min_val = box_.min()[i];
        let max_val = box_.max()[i];

        if point_val < min_val {
            // Point is below box in this dimension
            let gap = min_val - point_val;
            dist_sq += gap * gap;
        } else if point_val > max_val {
            // Point is above box in this dimension
            let gap = point_val - max_val;
            dist_sq += gap * gap;
        }
        // Otherwise point is inside box in this dimension, no contribution
    }

    Ok(dist_sq.sqrt())
}

/// Compute depth-based distance between two boxes (RegD 2025).
///
/// This is an optimized implementation for ndarray that uses actual log volumes.
pub fn depth_distance(
    box_a: &NdarrayBox,
    box_b: &NdarrayBox,
    temperature: f32,
    volume_weight: f32,
) -> Result<f32, BoxError> {
    // Standard Euclidean distance
    let euclidean_dist = box_a.distance(box_b)?;

    // Volume-based term: |log(Vol(A)) - log(Vol(B))|
    let vol_a = box_a.volume(temperature)?;
    let vol_b = box_b.volume(temperature)?;

    // Use actual logarithm for proper depth distance
    let log_vol_a = if vol_a > 1e-10 {
        vol_a.ln()
    } else {
        -23.0 // ln(1e-10) ≈ -23
    };

    let log_vol_b = if vol_b > 1e-10 { vol_b.ln() } else { -23.0 };

    // Volume difference term
    let volume_diff = (log_vol_a - log_vol_b).abs();

    // Depth distance = Euclidean + α * volume_diff
    Ok(euclidean_dist + volume_weight * volume_diff)
}

/// Compute boundary distance between two boxes (RegD 2025).
///
/// Optimized implementation for ndarray that computes exact boundary distance.
pub fn boundary_distance(
    outer: &NdarrayBox,
    inner: &NdarrayBox,
    temperature: f32,
) -> Result<Option<f32>, BoxError> {
    // Check if inner is contained in outer
    let containment = outer.containment_prob(inner, temperature)?;
    if containment < 0.99 {
        // Not fully contained
        return Ok(None);
    }

    // For axis-aligned boxes, boundary distance is the minimum distance from
    // any point in inner to the boundary of outer
    // This is the minimum "gap" between inner and outer boundaries in any dimension

    let mut min_gap = f32::INFINITY;

    for i in 0..outer.dim() {
        // Gap on the min side: inner.min\[i\] - outer.min\[i\]
        let gap_min = inner.min()[i] - outer.min()[i];
        // Gap on the max side: outer.max\[i\] - inner.max\[i\]
        let gap_max = outer.max()[i] - inner.max()[i];
        // Minimum gap in this dimension
        let gap = gap_min.min(gap_max);
        min_gap = min_gap.min(gap);
    }

    // If min_gap is still infinity, something went wrong
    if min_gap == f32::INFINITY {
        return Ok(Some(0.0));
    }

    Ok(Some(min_gap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_vector_to_box_distance_inside() {
        let box_ = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let point = array![0.5, 0.5];
        let dist = vector_to_box_distance(&point, &box_).unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_vector_to_box_distance_outside() {
        let box_ = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let point = array![2.0, 2.0];
        let dist = vector_to_box_distance(&point, &box_).unwrap();
        // Distance should be sqrt((2-1)^2 + (2-1)^2) = sqrt(2) ≈ 1.414
        assert!((dist - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_vector_to_box_distance_partial() {
        let box_ = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let point = array![0.5, 2.0]; // Inside in x, outside in y
        let dist = vector_to_box_distance(&point, &box_).unwrap();
        // Distance should be 1.0 (gap in y dimension)
        assert!((dist - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_depth_distance() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap();
        let dist = depth_distance(&box_a, &box_b, 1.0, 0.1).unwrap();
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_boundary_distance_contained() {
        let outer = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let inner = NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap();
        let dist = boundary_distance(&outer, &inner, 1.0).unwrap();
        assert!(dist.is_some());
        let dist_val = dist.unwrap();
        assert!(dist_val >= 0.0);
        assert!(dist_val <= 0.2); // Should be at most 0.2 (the gap)
    }

    #[test]
    fn test_boundary_distance_not_contained() {
        let outer = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let inner = NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], 1.0).unwrap();
        let dist = boundary_distance(&outer, &inner, 1.0).unwrap();
        assert!(dist.is_none()); // Not contained
    }
}
