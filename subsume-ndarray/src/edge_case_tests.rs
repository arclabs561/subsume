//! Edge case and error condition tests.

#[cfg(test)]
mod edge_case_tests {
    use crate::{NdarrayBox, NdarrayGumbelBox};
    use ndarray::{array, Array1};
    use subsume_core::{Box, BoxError, BoxEmbedding, BoxCollection, GumbelBox};
    use serde_json;

    #[test]
    fn test_empty_collection() {
        let collection: BoxCollection<NdarrayBox> = BoxCollection::new();
        assert_eq!(collection.len(), 0);
        assert!(collection.is_empty());
        assert!(collection.get(0).is_err());
    }

    #[test]
    fn test_collection_out_of_bounds() {
        let collection: BoxCollection<NdarrayBox> = vec![
            NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap(),
        ].into();
        
        assert!(collection.get(0).is_ok());
        assert!(collection.get(1).is_err());
        assert!(collection.get(100).is_err());
    }

    #[test]
    fn test_containment_matrix_empty() {
        let collection: BoxCollection<NdarrayBox> = BoxCollection::new();
        let matrix = collection.containment_matrix(1.0).unwrap();
        assert_eq!(matrix.len(), 0);
    }

    #[test]
    fn test_containment_matrix_single_box() {
        let collection: BoxCollection<NdarrayBox> = vec![
            NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap(),
        ].into();
        
        let matrix = collection.containment_matrix(1.0).unwrap();
        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0].len(), 1);
        assert!((matrix[0][0] - 1.0).abs() < 1e-5); // Box contains itself
    }

    #[test]
    fn test_containing_boxes_empty_collection() {
        let collection: BoxCollection<NdarrayBox> = BoxCollection::new();
        let query = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let containing = collection.containing_boxes(&query, 0.5, 1.0).unwrap();
        assert!(containing.is_empty());
    }

    #[test]
    fn test_contained_boxes_empty_collection() {
        let collection: BoxCollection<NdarrayBox> = BoxCollection::new();
        let query = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let contained = collection.contained_boxes(&query, 0.5, 1.0).unwrap();
        assert!(contained.is_empty());
    }

    #[test]
    fn test_dimension_mismatch_containment() {
        let box_2d = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_3d = NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0).unwrap();
        
        assert!(matches!(
            box_2d.containment_prob(&box_3d, 1.0),
            Err(BoxError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            box_2d.overlap_prob(&box_3d, 1.0),
            Err(BoxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_dimension_mismatch_intersection() {
        let box_2d = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_3d = NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0).unwrap();
        
        assert!(matches!(
            box_2d.intersection(&box_3d),
            Err(BoxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_zero_volume_containment_error() {
        let zero_box = NdarrayBox::new(array![0.0, 0.0], array![0.0, 0.0], 1.0).unwrap();
        let normal_box = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        
        // Trying to compute containment where other has zero volume should error
        assert!(matches!(
            normal_box.containment_prob(&zero_box, 1.0),
            Err(BoxError::ZeroVolume)
        ));
    }

    #[test]
    fn test_single_dimension_box() {
        let box_1d = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        assert_eq!(box_1d.dim(), 1);
        let volume = box_1d.volume(1.0).unwrap();
        assert_eq!(volume, 1.0);
    }

    #[test]
    fn test_high_dimensional_box() {
        let min = Array1::from(vec![0.0f32; 10]);
        let max = Array1::from(vec![1.0f32; 10]);
        let box_10d = NdarrayBox::new(min, max, 1.0).unwrap();
        assert_eq!(box_10d.dim(), 10);
        let volume = box_10d.volume(1.0).unwrap();
        assert_eq!(volume, 1.0); // 1.0^10 = 1.0
    }

    #[test]
    fn test_negative_bounds() {
        let box_neg = NdarrayBox::new(array![-1.0, -2.0], array![0.0, -1.0], 1.0).unwrap();
        let volume = box_neg.volume(1.0).unwrap();
        assert!(volume >= 0.0);
    }

    #[test]
    fn test_very_small_volume() {
        let small_box = NdarrayBox::new(
            array![0.0, 0.0],
            array![1e-6, 1e-6],
            1.0,
        ).unwrap();
        let volume = small_box.volume(1.0).unwrap();
        assert!(volume >= 0.0);
        assert!(volume < 1e-10);
    }

    #[test]
    fn test_identical_boxes() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        
        let containment = box_a.containment_prob(&box_b, 1.0).unwrap();
        assert!((containment - 1.0).abs() < 1e-5);
        
        let overlap = box_a.overlap_prob(&box_b, 1.0).unwrap();
        assert!((overlap - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_touching_boxes() {
        // Boxes that touch at boundaries
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![1.0, 0.0], array![2.0, 1.0], 1.0).unwrap();
        
        let intersection = box_a.intersection(&box_b).unwrap();
        let volume = intersection.volume(1.0).unwrap();
        assert_eq!(volume, 0.0); // Touching boxes have zero intersection volume
    }

    #[test]
    fn test_nested_boxes() {
        // Box B is completely inside Box A
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![2.0, 2.0], array![8.0, 8.0], 1.0).unwrap();
        
        let containment = box_a.containment_prob(&box_b, 1.0).unwrap();
        assert!(containment > 0.99);
        
        let reverse = box_b.containment_prob(&box_a, 1.0).unwrap();
        assert!(reverse < 0.5); // B does not contain A
    }

    #[test]
    fn test_partial_overlap() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();
        let box_b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        
        let overlap = box_a.overlap_prob(&box_b, 1.0).unwrap();
        assert!(overlap > 0.0 && overlap < 1.0);
        
        let intersection = box_a.intersection(&box_b).unwrap();
        let vol_intersection = intersection.volume(1.0).unwrap();
        assert!(vol_intersection > 0.0);
    }

    #[test]
    fn test_gumbel_temperature_effects() {
        let min = array![0.0, 0.0];
        let max = array![1.0, 1.0];
        
        let gumbel_low_temp = NdarrayGumbelBox::new(min.clone(), max.clone(), 0.1).unwrap();
        let gumbel_high_temp = NdarrayGumbelBox::new(min, max, 10.0).unwrap();
        
        let point = array![0.5, 0.5];
        
        let prob_low = gumbel_low_temp.membership_probability(&point).unwrap();
        let prob_high = gumbel_high_temp.membership_probability(&point).unwrap();
        
        // Both should be valid probabilities
        assert!(prob_low >= 0.0 && prob_low <= 1.0);
        assert!(prob_high >= 0.0 && prob_high <= 1.0);
    }

    #[test]
    fn test_gumbel_dimension_mismatch() {
        let gumbel_box = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();
        let point_3d = array![0.5, 0.5, 0.5];
        
        assert!(matches!(
            gumbel_box.membership_probability(&point_3d),
            Err(BoxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_collection_from_vec() {
        let boxes = vec![
            NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap(),
            NdarrayBox::new(array![0.0], array![2.0], 1.0).unwrap(),
        ];
        let collection = BoxCollection::from_vec(boxes);
        assert_eq!(collection.len(), 2);
    }

    #[test]
    fn test_collection_as_slice() {
        let collection: BoxCollection<NdarrayBox> = vec![
            NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap(),
        ].into();
        
        let slice = collection.as_slice();
        assert_eq!(slice.len(), 1);
    }

    #[test]
    fn test_collection_into_vec() {
        let collection: BoxCollection<NdarrayBox> = vec![
            NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap(),
        ].into();
        
        let vec = collection.into_vec();
        assert_eq!(vec.len(), 1);
    }

    #[test]
    fn test_serialization_invalid_json() {
        let invalid_json = "not valid json";
        assert!(serde_json::from_str::<NdarrayBox>(invalid_json).is_err());
    }

    #[test]
    fn test_serialization_missing_fields() {
        let incomplete_json = r#"{"min": [0.0, 0.0]}"#;
        assert!(serde_json::from_str::<NdarrayBox>(incomplete_json).is_err());
    }
}

