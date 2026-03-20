//! Distance metrics for box embeddings.
//!
//! This module implements:
//! - **Vector-to-box distance** (Query2Box, Ren et al. 2020): alpha-weighted L1 distance
//!   from a point to a query box, discriminating inside vs outside entities.
//!
//! For depth-based and boundary distances (RegD 2025), use the backend-specific modules
//! which provide correct log-volume formulations:
//! - `subsume::ndarray_backend::distance`
//! - `subsume::candle_backend::distance`
//!
//! # References
//!
//! - Ren et al. (ICLR 2020): "Query2Box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings"

use crate::BoxError;

/// Compute Query2Box alpha-weighted distance from an entity point to a query box.
///
/// ## Mathematical Formulation (Ren et al., ICLR 2020)
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
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

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
