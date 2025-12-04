//! Backend-specific distance implementations for candle.

use crate::CandleBox;
use candle_core::Tensor;
use subsume_core::{Box, BoxError};

/// Compute distance from a point (vector) to a box.
///
/// This implements the Concept2Box (2023) vector-to-box distance metric
/// for hybrid representations where concepts are boxes and entities are vectors.
///
/// # Parameters
///
/// - `point`: The point/vector as `Tensor`
/// - `box_`: The box
///
/// # Returns
///
/// Distance from point to box. Returns 0.0 if point is inside the box.
///
/// # Errors
///
/// Returns `BoxError` if point and box have dimension mismatch.
pub fn vector_to_box_distance(point: &Tensor, box_: &CandleBox) -> Result<f32, BoxError> {
    if point.dims() != &[box_.dim()] {
        return Err(BoxError::DimensionMismatch {
            expected: box_.dim(),
            actual: point.dims().len(),
        });
    }

    // Convert tensors to vectors for computation
    let point_data = point
        .to_vec1::<f32>()
        .map_err(|e| BoxError::Internal(e.to_string()))?;
    let min_data = box_
        .min()
        .to_vec1::<f32>()
        .map_err(|e| BoxError::Internal(e.to_string()))?;
    let max_data = box_
        .max()
        .to_vec1::<f32>()
        .map_err(|e| BoxError::Internal(e.to_string()))?;

    // For axis-aligned boxes, compute distance to nearest point on box boundary
    let mut dist_sq = 0.0;
    for i in 0..box_.dim() {
        let point_val = point_data[i];
        let min_val = min_data[i];
        let max_val = max_data[i];

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
/// Optimized implementation for candle that uses actual log volumes.
pub fn depth_distance(
    box_a: &CandleBox,
    box_b: &CandleBox,
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
/// Optimized implementation for candle that computes exact boundary distance.
pub fn boundary_distance(
    outer: &CandleBox,
    inner: &CandleBox,
    temperature: f32,
) -> Result<Option<f32>, BoxError> {
    // Check if inner is contained in outer
    let containment = outer.containment_prob(inner, temperature)?;
    if containment < 0.99 {
        // Not fully contained
        return Ok(None);
    }

    // Convert tensors to vectors for computation
    let outer_min_data = outer
        .min()
        .to_vec1::<f32>()
        .map_err(|e| BoxError::Internal(e.to_string()))?;
    let outer_max_data = outer
        .max()
        .to_vec1::<f32>()
        .map_err(|e| BoxError::Internal(e.to_string()))?;
    let inner_min_data = inner
        .min()
        .to_vec1::<f32>()
        .map_err(|e| BoxError::Internal(e.to_string()))?;
    let inner_max_data = inner
        .max()
        .to_vec1::<f32>()
        .map_err(|e| BoxError::Internal(e.to_string()))?;

    // For axis-aligned boxes, boundary distance is the minimum distance from
    // any point in inner to the boundary of outer
    let mut min_gap = f32::INFINITY;

    for i in 0..outer.dim() {
        // Gap on the min side: inner.min[i] - outer.min[i]
        let gap_min = inner_min_data[i] - outer_min_data[i];
        // Gap on the max side: outer.max[i] - inner.max[i]
        let gap_max = outer_max_data[i] - inner_max_data[i];
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
    use candle_core::Device;

    #[test]
    fn test_vector_to_box_distance_inside() -> Result<(), BoxError> {
        let device = Device::Cpu;
        let box_ = CandleBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            Tensor::new(&[1.0f32, 1.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            1.0,
        )?;
        let point =
            Tensor::new(&[0.5f32, 0.5], &device).map_err(|e| BoxError::Internal(e.to_string()))?;
        let dist = vector_to_box_distance(&point, &box_)?;
        assert_eq!(dist, 0.0);
        Ok(())
    }

    #[test]
    fn test_vector_to_box_distance_outside() -> Result<(), BoxError> {
        let device = Device::Cpu;
        let box_ = CandleBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            Tensor::new(&[1.0f32, 1.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            1.0,
        )?;
        let point =
            Tensor::new(&[2.0f32, 2.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?;
        let dist = vector_to_box_distance(&point, &box_)?;
        // Distance should be sqrt((2-1)^2 + (2-1)^2) = sqrt(2) ≈ 1.414
        assert!((dist - 2.0_f32.sqrt()).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_depth_distance() -> Result<(), BoxError> {
        let device = Device::Cpu;
        let box_a = CandleBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            Tensor::new(&[1.0f32, 1.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            1.0,
        )?;
        let box_b = CandleBox::new(
            Tensor::new(&[0.2f32, 0.2], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            Tensor::new(&[0.8f32, 0.8], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            1.0,
        )?;
        let dist = depth_distance(&box_a, &box_b, 1.0, 0.1)?;
        assert!(dist >= 0.0);
        Ok(())
    }

    #[test]
    fn test_boundary_distance_contained() -> Result<(), BoxError> {
        let device = Device::Cpu;
        let outer = CandleBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            Tensor::new(&[1.0f32, 1.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            1.0,
        )?;
        let inner = CandleBox::new(
            Tensor::new(&[0.2f32, 0.2], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            Tensor::new(&[0.8f32, 0.8], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            1.0,
        )?;
        let dist = boundary_distance(&outer, &inner, 1.0)?;
        assert!(dist.is_some());
        let dist_val = dist.unwrap();
        assert!(dist_val >= 0.0);
        assert!(dist_val <= 0.2); // Should be at most 0.2 (the gap)
        Ok(())
    }

    #[test]
    fn test_boundary_distance_not_contained() -> Result<(), BoxError> {
        let device = Device::Cpu;
        let outer = CandleBox::new(
            Tensor::new(&[0.0f32, 0.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            Tensor::new(&[1.0f32, 1.0], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            1.0,
        )?;
        let inner = CandleBox::new(
            Tensor::new(&[0.5f32, 0.5], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            Tensor::new(&[1.5f32, 1.5], &device).map_err(|e| BoxError::Internal(e.to_string()))?,
            1.0,
        )?;
        let dist = boundary_distance(&outer, &inner, 1.0)?;
        assert!(dist.is_none()); // Not contained
        Ok(())
    }
}
