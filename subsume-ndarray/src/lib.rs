//! # subsume-ndarray
//!
//! Ndarray implementation of `subsume-core` traits for box embeddings.
//!
//! This crate provides `NdarrayBox` and `NdarrayGumbelBox` types that implement
//! the `Box` and `GumbelBox` traits using `ndarray::Array1<f32>`.
//!
//! # Example
//!
//! ```rust,no_run
//! use subsume_ndarray::NdarrayBox;
//! use subsume_core::Box;
//! use ndarray::array;
//!
//! let min = array![0.0, 0.0, 0.0];
//! let max = array![1.0, 1.0, 1.0];
//!
//! let box_a = NdarrayBox::new(min, max, 1.0)?;
//! let volume = box_a.volume(1.0)?;
//! # Ok::<(), subsume_core::BoxError>(())
//! ```

#![warn(missing_docs)]
// Allow acceptable patterns in tests and examples
#![allow(clippy::useless_vec)] // vec! is often clearer than alternatives in tests
#![allow(clippy::needless_range_loop)] // Indexing is sometimes necessary and clear
#![allow(clippy::module_inception)] // Test modules often match parent name

pub mod benchmark;
pub mod distance;
pub mod evaluation;
mod ndarray_box;
mod ndarray_gumbel;
pub mod optimizer;

#[cfg(test)]
mod matrix_e2e_tests;

#[cfg(test)]
mod enriched_methods_tests;

#[cfg(test)]
mod paper_verification_tests;

#[cfg(test)]
mod quantitative_verification_tests;

pub use benchmark::{BenchmarkConfig, BenchmarkResult, BenchmarkSuite};
pub use distance::{boundary_distance, depth_distance, vector_to_box_distance};
#[cfg(feature = "plotting")]
pub use evaluation::plotting;
pub use evaluation::{EvaluationConfig, EvaluationMetrics, OptimizerComparison};
pub use ndarray_box::NdarrayBox;
pub use ndarray_gumbel::NdarrayGumbelBox;
pub use optimizer::{Adam, AdamW, SGD};

#[cfg(test)]
mod proptest_tests;

#[cfg(test)]
mod invariant_tests;

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod trainer_integration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use serde_json;
    use subsume_core::{Box, BoxCollection, BoxEmbedding, GumbelBox};

    #[test]
    fn test_box_creation() {
        let min = array![0.0, 0.0, 0.0];
        let max = array![1.0, 1.0, 1.0];
        let box_a = NdarrayBox::new(min.clone(), max.clone(), 1.0).unwrap();
        assert_eq!(box_a.dim(), 3);
    }

    #[test]
    fn test_box_creation_invalid_bounds() {
        let min = array![1.0, 0.0];
        let max = array![0.0, 1.0];
        assert!(NdarrayBox::new(min, max, 1.0).is_err());
    }

    #[test]
    fn test_volume() {
        let min = array![0.0, 0.0];
        let max = array![2.0, 3.0];
        let box_a = NdarrayBox::new(min, max, 1.0).unwrap();
        let volume = box_a.volume(1.0).unwrap();
        assert_eq!(volume, 6.0);
    }

    #[test]
    fn test_volume_zero() {
        let min = array![0.0, 0.0];
        let max = array![0.0, 1.0];
        let box_a = NdarrayBox::new(min, max, 1.0).unwrap();
        let volume = box_a.volume(1.0).unwrap();
        assert_eq!(volume, 0.0);
    }

    #[test]
    fn test_intersection() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let intersection = box_a.intersection(&box_b).unwrap();
        let volume = intersection.volume(1.0).unwrap();
        assert_eq!(volume, 1.0);
    }

    #[test]
    fn test_intersection_disjoint() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![2.0, 2.0], array![3.0, 3.0], 1.0).unwrap();
        let intersection = box_a.intersection(&box_b).unwrap();
        let volume = intersection.volume(1.0).unwrap();
        assert_eq!(volume, 0.0);
        // Verify it's a valid zero-volume box
        assert_eq!(intersection.min(), intersection.max());
    }

    #[test]
    fn test_containment_prob() {
        let premise = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let hypothesis = NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap();
        let prob = premise.containment_prob(&hypothesis, 1.0).unwrap();
        assert!(prob > 0.9);
    }

    #[test]
    fn test_containment_prob_disjoint() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![2.0, 2.0], array![3.0, 3.0], 1.0).unwrap();
        let prob = box_a.containment_prob(&box_b, 1.0).unwrap();
        assert_eq!(prob, 0.0);
        // Reverse should also be 0
        let prob_reverse = box_b.containment_prob(&box_a, 1.0).unwrap();
        assert_eq!(prob_reverse, 0.0);
    }

    #[test]
    fn test_overlap_prob() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], 1.0).unwrap();
        let prob = box_a.overlap_prob(&box_b, 1.0).unwrap();
        assert!(prob > 0.0);
        assert!(prob <= 1.0);
    }

    #[test]
    fn test_overlap_prob_identical() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let prob = box_a.overlap_prob(&box_a, 1.0).unwrap();
        assert_eq!(prob, 1.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        assert!(box_a.intersection(&box_b).is_err());
        assert!(box_a.containment_prob(&box_b, 1.0).is_err());
    }

    #[test]
    fn test_gumbel_box() {
        let min = array![0.0, 0.0];
        let max = array![1.0, 1.0];
        let gumbel_box = NdarrayGumbelBox::new(min, max, 1.0).unwrap();
        assert_eq!(gumbel_box.temperature(), 1.0);
        let sample = gumbel_box.sample();
        assert_eq!(sample.len(), 2);
        // Verify sample is within bounds
        for i in 0..2 {
            assert!(sample[i] >= gumbel_box.min()[i]);
            assert!(sample[i] <= gumbel_box.max()[i]);
        }
    }

    #[test]
    fn test_gumbel_membership() {
        let gumbel_box = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let point = array![0.5, 0.5];
        let prob = gumbel_box.membership_probability(&point).unwrap();
        assert!(prob >= 0.0);
        assert!(prob <= 1.0);
    }

    #[test]
    fn test_gumbel_membership_outside() {
        let gumbel_box = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 0.1).unwrap();
        let point = array![2.0, 2.0];
        let prob = gumbel_box.membership_probability(&point).unwrap();
        assert!(prob < 0.5); // Should be low for point outside box
    }

    #[test]
    fn test_box_collection() {
        let mut collection = BoxCollection::new();
        collection.push(NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap());
        collection.push(NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap());
        collection.push(NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], 1.0).unwrap());

        assert_eq!(collection.len(), 3);
        assert!(!collection.is_empty());
    }

    #[test]
    fn test_box_collection_containment_matrix() {
        let collection: BoxCollection<NdarrayBox> = vec![
            NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap(),
            NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap(),
        ]
        .into();

        let matrix = collection.containment_matrix(1.0).unwrap();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        // Box 0 contains box 1, so matrix[0][1] should be high
        assert!(matrix[0][1] > 0.9);
        // Box 1 doesn't contain box 0, so matrix[1][0] should be low
        assert!(matrix[1][0] < 0.5);
    }

    #[test]
    fn test_box_collection_containing_boxes() {
        let collection: BoxCollection<NdarrayBox> = vec![
            NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap(),
            NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap(),
            NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], 1.0).unwrap(),
        ]
        .into();

        let query = NdarrayBox::new(array![0.3, 0.3], array![0.7, 0.7], 1.0).unwrap();
        let containing = collection.containing_boxes(&query, 0.5, 1.0).unwrap();
        // Box 0 and 1 should contain the query
        assert!(containing.contains(&0));
        assert!(containing.contains(&1));
        // Box 2 overlaps but doesn't fully contain
        assert!(!containing.contains(&2));
    }

    #[test]
    fn test_box_collection_contained_boxes() {
        let collection: BoxCollection<NdarrayBox> = vec![
            NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap(),
            NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap(),
        ]
        .into();

        let query = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let contained = collection.contained_boxes(&query, 0.5, 1.0).unwrap();
        // Both boxes should be contained in the query
        assert!(contained.contains(&0));
        assert!(contained.contains(&1));
    }

    #[test]
    fn test_serialize_ndarray_box() {
        let box_ = NdarrayBox::new(array![0.0, 1.0, 2.0], array![1.0, 2.0, 3.0], 1.5).unwrap();

        let serialized = serde_json::to_string(&box_).unwrap();
        let deserialized: NdarrayBox = serde_json::from_str(&serialized).unwrap();

        assert_eq!(box_.dim(), deserialized.dim());
        assert_eq!(box_.temperature, deserialized.temperature);
        assert_eq!(box_.min(), deserialized.min());
        assert_eq!(box_.max(), deserialized.max());
    }

    #[test]
    fn test_serialize_ndarray_gumbel_box() {
        let gumbel_box = NdarrayGumbelBox::new(array![0.0, 1.0], array![1.0, 2.0], 0.5).unwrap();

        let serialized = serde_json::to_string(&gumbel_box).unwrap();
        let deserialized: NdarrayGumbelBox = serde_json::from_str(&serialized).unwrap();

        assert_eq!(gumbel_box.dim(), deserialized.dim());
        assert_eq!(gumbel_box.temperature(), deserialized.temperature());
        assert_eq!(gumbel_box.min(), deserialized.min());
        assert_eq!(gumbel_box.max(), deserialized.max());
    }

    #[test]
    fn test_serialize_round_trip_operations() {
        let original = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();

        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: NdarrayBox = serde_json::from_str(&serialized).unwrap();

        // Operations should work identically
        let vol_original = original.volume(1.0).unwrap();
        let vol_deserialized = deserialized.volume(1.0).unwrap();
        assert_eq!(vol_original, vol_deserialized);
    }
}
