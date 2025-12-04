//! Tests for enriched methods (union, center, distance, overlap_matrix, etc.)

#[cfg(test)]
mod enriched_methods_tests {
    use crate::{NdarrayBox, NdarrayGumbelBox};
    use ndarray::array;
    use subsume_core::{Box, BoxEmbedding, BoxCollection, GumbelBox};

    #[test]
    fn test_union() {
        let box_a = NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap();
        
        let box_b = NdarrayBox::new(
            array![0.5, 0.5],
            array![1.5, 1.5],
            1.0,
        ).unwrap();
        
        let union = box_a.union(&box_b).unwrap();
        
        // Union should contain both boxes
        assert!(union.min()[0] <= box_a.min()[0]);
        assert!(union.min()[1] <= box_a.min()[1]);
        assert!(union.max()[0] >= box_b.max()[0]);
        assert!(union.max()[1] >= box_b.max()[1]);
        
        // Union should be the smallest box containing both
        assert_eq!(union.min()[0], 0.0);
        assert_eq!(union.min()[1], 0.0);
        assert_eq!(union.max()[0], 1.5);
        assert_eq!(union.max()[1], 1.5);
    }

    #[test]
    fn test_center() {
        let box_ = NdarrayBox::new(
            array![0.0, 2.0, 4.0],
            array![2.0, 4.0, 6.0],
            1.0,
        ).unwrap();
        
        let center = box_.center().unwrap();
        
        assert!((center[0] - 1.0).abs() < 1e-6); // (0.0 + 2.0) / 2 = 1.0
        assert!((center[1] - 3.0).abs() < 1e-6); // (2.0 + 4.0) / 2 = 3.0
        assert!((center[2] - 5.0).abs() < 1e-6); // (4.0 + 6.0) / 2 = 5.0
    }

    #[test]
    fn test_distance_overlapping() {
        let box_a = NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap();
        
        let box_b = NdarrayBox::new(
            array![0.5, 0.5],
            array![1.5, 1.5],
            1.0,
        ).unwrap();
        
        let dist = box_a.distance(&box_b).unwrap();
        assert_eq!(dist, 0.0); // Overlapping boxes have distance 0
    }

    #[test]
    fn test_distance_disjoint() {
        let box_a = NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap();
        
        let box_b = NdarrayBox::new(
            array![2.0, 2.0],
            array![3.0, 3.0],
            1.0,
        ).unwrap();
        
        let dist = box_a.distance(&box_b).unwrap();
        // Distance should be sqrt((2.0-1.0)^2 + (2.0-1.0)^2) = sqrt(2)
        assert!((dist - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_distance_partially_overlapping() {
        let box_a = NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap();
        
        let box_b = NdarrayBox::new(
            array![1.5, 0.0],
            array![2.5, 1.0],
            1.0,
        ).unwrap();
        
        let dist = box_a.distance(&box_b).unwrap();
        // Distance should be 0.5 (overlap in y, gap in x)
        assert!((dist - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_overlap_matrix() {
        let mut collection = BoxCollection::new();
        collection.push(NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![0.5, 0.5],
            array![1.5, 1.5],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![2.0, 2.0],
            array![3.0, 3.0],
            1.0,
        ).unwrap());
        
        let matrix = collection.overlap_matrix(1.0).unwrap();
        
        assert_eq!(matrix.len(), 3);
        assert!(matrix.iter().all(|row| row.len() == 3));
        
        // Diagonal should be 1.0 (each box overlaps with itself)
        for i in 0..3 {
            assert!((matrix[i][i] - 1.0).abs() < 1e-5);
        }
        
        // Box 0 and 1 overlap
        assert!(matrix[0][1] > 0.0);
        assert!(matrix[1][0] > 0.0);
        
        // Box 0 and 2 don't overlap
        assert!(matrix[0][2] < 0.5);
        assert!(matrix[2][0] < 0.5);
    }

    #[test]
    fn test_overlapping_boxes() {
        let mut collection = BoxCollection::new();
        collection.push(NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![0.5, 0.5],
            array![1.5, 1.5],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![2.0, 2.0],
            array![3.0, 3.0],
            1.0,
        ).unwrap());
        
        let query = NdarrayBox::new(
            array![0.2, 0.2],
            array![0.8, 0.8],
            1.0,
        ).unwrap();
        
        // Check actual overlap probabilities
        let overlap_0 = collection.get(0).unwrap().overlap_prob(&query, 1.0).unwrap();
        let overlap_1 = collection.get(1).unwrap().overlap_prob(&query, 1.0).unwrap();
        let overlap_2 = collection.get(2).unwrap().overlap_prob(&query, 1.0).unwrap();
        
        // Use a threshold that's lower than the actual overlap
        let threshold = (overlap_0.min(overlap_1) * 0.5).max(0.01);
        let overlapping = collection.overlapping_boxes(&query, threshold, 1.0).unwrap();
        
        // Query overlaps with box 0 (contained) and box 1 (partial overlap)
        assert!(overlapping.contains(&0), "Query should overlap with box 0 (overlap={})", overlap_0);
        assert!(overlapping.contains(&1), "Query should overlap with box 1 (overlap={})", overlap_1);
        assert!(!overlapping.contains(&2), "Query should not overlap with box 2 (overlap={})", overlap_2);
    }

    #[test]
    fn test_nearest_boxes() {
        let mut collection = BoxCollection::new();
        collection.push(NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![2.0, 2.0],
            array![3.0, 3.0],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![5.0, 5.0],
            array![6.0, 6.0],
            1.0,
        ).unwrap());
        
        let query = NdarrayBox::new(
            array![0.5, 0.5],
            array![1.5, 1.5],
            1.0,
        ).unwrap();
        
        let nearest = collection.nearest_boxes(&query, 2).unwrap();
        
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0], 0); // Closest (overlapping)
        assert_eq!(nearest[1], 1); // Second closest
    }

    #[test]
    fn test_bounding_box() {
        let mut collection = BoxCollection::new();
        collection.push(NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![2.0, 2.0],
            array![3.0, 3.0],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![1.5, 0.5],
            array![2.5, 1.5],
            1.0,
        ).unwrap());
        
        let bbox = collection.bounding_box().unwrap();
        
        // Bounding box should contain all boxes
        assert_eq!(bbox.min()[0], 0.0);
        assert_eq!(bbox.min()[1], 0.0);
        assert_eq!(bbox.max()[0], 3.0);
        assert_eq!(bbox.max()[1], 3.0);
        
        // Verify it contains all boxes
        for i in 0..collection.len() {
            let box_i = collection.get(i).unwrap();
            let containment = bbox.containment_prob(box_i, 1.0).unwrap();
            assert!(containment > 0.99, "Bounding box should contain box {i}");
        }
    }

    #[test]
    fn test_bounding_box_empty() {
        let collection: BoxCollection<NdarrayBox> = BoxCollection::new();
        let result = collection.bounding_box();
        assert!(result.is_err());
    }

    #[test]
    fn test_union_gumbel() {
        let box_a = NdarrayGumbelBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap();
        
        let box_b = NdarrayGumbelBox::new(
            array![0.5, 0.5],
            array![1.5, 1.5],
            0.5,
        ).unwrap();
        
        let union = box_a.union(&box_b).unwrap();
        assert_eq!(union.min()[0], 0.0);
        assert_eq!(union.max()[0], 1.5);
    }

    #[test]
    fn test_center_gumbel() {
        let box_ = NdarrayGumbelBox::new(
            array![1.0, 3.0],
            array![3.0, 5.0],
            1.0,
        ).unwrap();
        
        let center = box_.center().unwrap();
        assert!((center[0] - 2.0).abs() < 1e-6);
        assert!((center[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_gumbel() {
        let box_a = NdarrayGumbelBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap();
        
        let box_b = NdarrayGumbelBox::new(
            array![2.0, 2.0],
            array![3.0, 3.0],
            0.5,
        ).unwrap();
        
        let dist = box_a.distance(&box_b).unwrap();
        assert!((dist - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_overlap_matrix_consistency() {
        let mut collection = BoxCollection::new();
        for i in 0..5 {
            let offset = (i as f32) * 0.2;
            collection.push(NdarrayBox::new(
                array![offset, offset],
                array![offset + 0.5, offset + 0.5],
                1.0,
            ).unwrap());
        }
        
        let overlap_matrix = collection.overlap_matrix(1.0).unwrap();
        let containment_matrix = collection.containment_matrix(1.0).unwrap();
        
        // Overlap matrix should be symmetric
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (overlap_matrix[i][j] - overlap_matrix[j][i]).abs() < 1e-5,
                    "Overlap matrix should be symmetric: [{i}][{j}] = {}, [{j}][{i}] = {}",
                    overlap_matrix[i][j], overlap_matrix[j][i]
                );
            }
        }
        
        // If A contains B, then A overlaps with B (but not necessarily vice versa)
        for i in 0..5 {
            for j in 0..5 {
                if containment_matrix[i][j] > 0.9 {
                    assert!(
                        overlap_matrix[i][j] > 0.5,
                        "If box {i} contains box {j}, they should overlap"
                    );
                }
            }
        }
    }

    #[test]
    fn test_nearest_boxes_k_larger_than_collection() {
        let mut collection = BoxCollection::new();
        collection.push(NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap());
        collection.push(NdarrayBox::new(
            array![2.0, 2.0],
            array![3.0, 3.0],
            1.0,
        ).unwrap());
        
        let query = NdarrayBox::new(
            array![0.5, 0.5],
            array![1.5, 1.5],
            1.0,
        ).unwrap();
        
        let nearest = collection.nearest_boxes(&query, 10).unwrap();
        assert_eq!(nearest.len(), 2); // Should return all boxes, not panic
    }

    #[test]
    fn test_nearest_boxes_k_zero() {
        let mut collection = BoxCollection::new();
        collection.push(NdarrayBox::new(
            array![0.0, 0.0],
            array![1.0, 1.0],
            1.0,
        ).unwrap());
        
        let query = NdarrayBox::new(
            array![0.5, 0.5],
            array![1.5, 1.5],
            1.0,
        ).unwrap();
        
        let nearest = collection.nearest_boxes(&query, 0).unwrap();
        assert_eq!(nearest.len(), 0);
    }
}

