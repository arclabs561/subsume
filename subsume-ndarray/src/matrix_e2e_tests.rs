//! End-to-end tests for matrix operations and batch queries.

#[cfg(test)]
mod matrix_e2e_tests {
    use crate::{NdarrayBox, NdarrayGumbelBox};
    use ndarray::array;
    use subsume_core::{Box, BoxCollection, BoxEmbedding};

    /// Create a hierarchical knowledge graph structure for testing.
    fn create_hierarchy() -> BoxCollection<NdarrayBox> {
        let mut collection = BoxCollection::new();

        // Level 0: Root (Entity)
        collection
            .push(NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0).unwrap());

        // Level 1: Categories
        collection
            .push(NdarrayBox::new(array![0.0, 0.0, 0.0], array![0.5, 0.5, 0.5], 1.0).unwrap()); // Category A

        collection
            .push(NdarrayBox::new(array![0.5, 0.0, 0.0], array![1.0, 0.5, 0.5], 1.0).unwrap()); // Category B

        // Level 2: Subcategories
        collection
            .push(NdarrayBox::new(array![0.0, 0.0, 0.0], array![0.3, 0.3, 0.3], 1.0).unwrap()); // Subcategory A1

        collection
            .push(NdarrayBox::new(array![0.2, 0.0, 0.0], array![0.5, 0.3, 0.3], 1.0).unwrap()); // Subcategory A2

        collection
            .push(NdarrayBox::new(array![0.5, 0.0, 0.0], array![0.7, 0.3, 0.3], 1.0).unwrap()); // Subcategory B1

        // Level 3: Leaf nodes
        collection
            .push(NdarrayBox::new(array![0.0, 0.0, 0.0], array![0.2, 0.2, 0.2], 1.0).unwrap()); // Leaf A1a

        collection
            .push(NdarrayBox::new(array![0.1, 0.0, 0.0], array![0.3, 0.2, 0.2], 1.0).unwrap()); // Leaf A1b

        collection
    }

    #[test]
    fn test_containment_matrix_hierarchy() {
        let collection = create_hierarchy();
        let matrix = collection.containment_matrix(1.0).unwrap();

        // Matrix should be n x n where n = 8
        assert_eq!(matrix.len(), 8);
        assert!(matrix.iter().all(|row| row.len() == 8));

        // Diagonal should be 1.0 (each box contains itself)
        for i in 0..8 {
            assert!(
                (matrix[i][i] - 1.0).abs() < 1e-5,
                "Diagonal element {i} should be 1.0"
            );
        }

        // Root (index 0) should contain all others with high probability
        for j in 1..8 {
            assert!(matrix[0][j] > 0.8, "Root should contain box {j}");
        }

        // Category A (index 1) should contain its subcategories (3, 4) and leaves (6, 7)
        assert!(
            matrix[1][3] > 0.8,
            "Category A should contain Subcategory A1"
        );
        assert!(
            matrix[1][4] > 0.8,
            "Category A should contain Subcategory A2"
        );
        assert!(matrix[1][6] > 0.8, "Category A should contain Leaf A1a");
        assert!(matrix[1][7] > 0.8, "Category A should contain Leaf A1b");

        // Category A should NOT contain Category B (index 2)
        assert!(
            matrix[1][2] < 0.5,
            "Category A should not contain Category B"
        );
    }

    #[test]
    fn test_containment_matrix_symmetry_properties() {
        let collection = create_hierarchy();
        let matrix = collection.containment_matrix(1.0).unwrap();

        // Containment is not symmetric, but we can check some properties
        // If A contains B, then B does not necessarily contain A
        // But if A contains B and B contains C, then A should contain C (transitivity)

        // Check transitivity: if matrix[i][j] > 0.9 and matrix[j][k] > 0.9,
        // then matrix[i][k] should be reasonably high
        for i in 0..8 {
            for j in 0..8 {
                if matrix[i][j] > 0.9 {
                    for k in 0..8 {
                        if matrix[j][k] > 0.9 {
                            // Transitivity: i contains j, j contains k, so i should contain k
                            assert!(
                                matrix[i][k] > 0.5,
                                "Transitivity violated: {} contains {} ({:.3}), {} contains {} ({:.3}), but {} contains {} ({:.3})",
                                i, j, matrix[i][j], j, k, matrix[j][k], i, k, matrix[i][k]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_containing_boxes_hierarchy() {
        let collection = create_hierarchy();

        // Query for a leaf node (index 6: Leaf A1a)
        let leaf = collection.get(6).unwrap();
        let containing = collection.containing_boxes(leaf, 0.5, 1.0).unwrap();

        // Should be contained by: Root (0), Category A (1), Subcategory A1 (3)
        assert!(containing.contains(&0), "Root should contain leaf");
        assert!(containing.contains(&1), "Category A should contain leaf");
        assert!(
            containing.contains(&3),
            "Subcategory A1 should contain leaf"
        );
        assert!(
            !containing.contains(&2),
            "Category B should not contain leaf from Category A"
        );

        // Query for a subcategory (index 3: Subcategory A1)
        let subcat = collection.get(3).unwrap();
        let containing_subcat = collection.containing_boxes(subcat, 0.5, 1.0).unwrap();

        assert!(
            containing_subcat.contains(&0),
            "Root should contain subcategory"
        );
        assert!(
            containing_subcat.contains(&1),
            "Category A should contain subcategory"
        );
        // Note: subcategory contains itself (prob = 1.0), so it may be in results
    }

    #[test]
    fn test_contained_boxes_hierarchy() {
        let collection = create_hierarchy();

        // Query with root (index 0)
        let root = collection.get(0).unwrap();
        let contained = collection.contained_boxes(root, 0.5, 1.0).unwrap();

        // Root should contain all others (including itself, since prob = 1.0 > threshold)
        assert!(contained.len() >= 7, "Root should contain at least 7 boxes");
        for i in 1..8 {
            assert!(contained.contains(&i), "Root should contain box {i}");
        }

        // Query with Category A (index 1)
        let category_a = collection.get(1).unwrap();
        let contained_in_a = collection.contained_boxes(category_a, 0.5, 1.0).unwrap();

        // Should contain: Subcategory A1 (3), Subcategory A2 (4), Leaf A1a (6), Leaf A1b (7)
        assert!(
            contained_in_a.contains(&3),
            "Category A should contain Subcategory A1"
        );
        assert!(
            contained_in_a.contains(&4),
            "Category A should contain Subcategory A2"
        );
        assert!(
            contained_in_a.contains(&6),
            "Category A should contain Leaf A1a"
        );
        assert!(
            contained_in_a.contains(&7),
            "Category A should contain Leaf A1b"
        );
        assert!(
            !contained_in_a.contains(&2),
            "Category A should not contain Category B"
        );
    }

    #[test]
    fn test_containment_matrix_large_collection() {
        // Test with a larger collection (20 boxes)
        let mut collection = BoxCollection::new();

        for i in 0..20 {
            let offset = (i as f32) * 0.05;
            collection.push(
                NdarrayBox::new(
                    array![offset, offset, offset],
                    array![offset + 0.5, offset + 0.5, offset + 0.5],
                    1.0,
                )
                .unwrap(),
            );
        }

        let matrix = collection.containment_matrix(1.0).unwrap();

        assert_eq!(matrix.len(), 20);
        assert!(matrix.iter().all(|row| row.len() == 20));

        // Check diagonal
        for i in 0..20 {
            assert!((matrix[i][i] - 1.0).abs() < 1e-5);
        }

        // Check that earlier boxes (smaller offsets) tend to contain later ones
        // when they overlap significantly
        for i in 0..10 {
            for j in (i + 1)..(i + 5).min(20) {
                // If boxes overlap significantly, containment should be non-zero
                let prob = matrix[i][j];
                assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Probability should be in [0, 1]"
                );
            }
        }
    }

    #[test]
    fn test_containing_boxes_threshold_variation() {
        let collection = create_hierarchy();
        let leaf = collection.get(6).unwrap();

        // Test with different thresholds
        let containing_strict = collection.containing_boxes(leaf, 0.9, 1.0).unwrap();
        let containing_moderate = collection.containing_boxes(leaf, 0.5, 1.0).unwrap();
        let containing_loose = collection.containing_boxes(leaf, 0.1, 1.0).unwrap();

        // Stricter threshold should return fewer or equal results
        assert!(containing_strict.len() <= containing_moderate.len());
        assert!(containing_moderate.len() <= containing_loose.len());

        // All strict results should be in moderate, all moderate in loose
        for &idx in &containing_strict {
            assert!(containing_moderate.contains(&idx));
        }
        for &idx in &containing_moderate {
            assert!(containing_loose.contains(&idx));
        }
    }

    #[test]
    fn test_contained_boxes_threshold_variation() {
        let collection = create_hierarchy();
        let root = collection.get(0).unwrap();

        let contained_strict = collection.contained_boxes(root, 0.9, 1.0).unwrap();
        let contained_moderate = collection.contained_boxes(root, 0.5, 1.0).unwrap();
        let contained_loose = collection.contained_boxes(root, 0.1, 1.0).unwrap();

        assert!(contained_strict.len() <= contained_moderate.len());
        assert!(contained_moderate.len() <= contained_loose.len());
    }

    #[test]
    fn test_matrix_consistency_with_individual_queries() {
        let collection = create_hierarchy();
        let matrix = collection.containment_matrix(1.0).unwrap();

        // Verify that matrix values match individual containment_prob calls
        for i in 0..collection.len() {
            let box_i = collection.get(i).unwrap();
            for j in 0..collection.len() {
                let box_j = collection.get(j).unwrap();
                let individual_prob = box_i.containment_prob(box_j, 1.0).unwrap();
                let matrix_prob = matrix[i][j];

                assert!(
                    (individual_prob - matrix_prob).abs() < 1e-5,
                    "Matrix value [{i}][{j}] = {matrix_prob} should match individual call = {individual_prob}"
                );
            }
        }
    }

    #[test]
    fn test_containing_boxes_consistency_with_matrix() {
        let collection = create_hierarchy();
        let matrix = collection.containment_matrix(1.0).unwrap();

        // For each box, check that containing_boxes returns indices matching matrix
        for query_idx in 0..collection.len() {
            let query = collection.get(query_idx).unwrap();
            let containing = collection.containing_boxes(query, 0.5, 1.0).unwrap();

            // Verify that all returned indices have matrix[idx][query_idx] > threshold
            for &idx in &containing {
                assert!(
                    matrix[idx][query_idx] > 0.5,
                    "Box {idx} returned by containing_boxes but matrix[{idx}][{query_idx}] = {} <= 0.5",
                    matrix[idx][query_idx]
                );
            }

            // Verify that all boxes with matrix[idx][query_idx] > threshold are in results
            for idx in 0..collection.len() {
                if matrix[idx][query_idx] > 0.5 {
                    assert!(
                        containing.contains(&idx),
                        "Box {idx} has matrix[{idx}][{query_idx}] = {} > 0.5 but not in containing_boxes results",
                        matrix[idx][query_idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_contained_boxes_consistency_with_matrix() {
        let collection = create_hierarchy();
        let matrix = collection.containment_matrix(1.0).unwrap();

        // For each box, check that contained_boxes returns indices matching matrix
        for query_idx in 0..collection.len() {
            let query = collection.get(query_idx).unwrap();
            let contained = collection.contained_boxes(query, 0.5, 1.0).unwrap();

            // Verify that all returned indices have matrix[query_idx][idx] > threshold
            for &idx in &contained {
                assert!(
                    matrix[query_idx][idx] > 0.5,
                    "Box {idx} returned by contained_boxes but matrix[{query_idx}][{idx}] = {} <= 0.5",
                    matrix[query_idx][idx]
                );
            }

            // Verify that all boxes with matrix[query_idx][idx] > threshold are in results
            // (including query_idx itself if prob > threshold)
            for idx in 0..collection.len() {
                if matrix[query_idx][idx] > 0.5 {
                    assert!(
                        contained.contains(&idx),
                        "Box {idx} has matrix[{query_idx}][{idx}] = {} > 0.5 but not in contained_boxes results",
                        matrix[query_idx][idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_matrix_with_gumbel_boxes() {
        let mut collection: BoxCollection<NdarrayGumbelBox> = BoxCollection::new();

        // Create a small hierarchy with Gumbel boxes
        collection.push(NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap());

        collection.push(
            NdarrayGumbelBox::new(
                array![0.2, 0.2],
                array![0.8, 0.8],
                0.5, // Lower temperature = harder bounds
            )
            .unwrap(),
        );

        collection.push(
            NdarrayGumbelBox::new(
                array![0.3, 0.3],
                array![0.7, 0.7],
                2.0, // Higher temperature = softer bounds
            )
            .unwrap(),
        );

        let matrix = collection.containment_matrix(1.0).unwrap();

        assert_eq!(matrix.len(), 3);
        assert!(matrix.iter().all(|row| row.len() == 3));

        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((matrix[i][i] - 1.0).abs() < 1e-5);
        }

        // Box 0 (largest) should contain others
        assert!(matrix[0][1] > 0.5);
        assert!(matrix[0][2] > 0.5);
    }

    #[test]
    fn test_matrix_temperature_sensitivity() {
        let collection = create_hierarchy();

        // Test with different temperatures
        let matrix_low_temp = collection.containment_matrix(0.1).unwrap();
        let matrix_normal = collection.containment_matrix(1.0).unwrap();
        let matrix_high_temp = collection.containment_matrix(10.0).unwrap();

        // Lower temperature should give sharper (more extreme) probabilities
        // Higher temperature should give softer (closer to uniform) probabilities

        // For boxes that clearly contain others, low temp should give higher probabilities
        // For boxes that don't contain others, low temp should give lower probabilities

        // Check a clear containment relationship (root contains leaf)
        assert!(matrix_low_temp[0][6] >= matrix_normal[0][6] - 0.1);
        assert!(matrix_normal[0][6] >= matrix_high_temp[0][6] - 0.1);
    }

    #[test]
    fn test_matrix_empty_collection() {
        let collection: BoxCollection<NdarrayBox> = BoxCollection::new();
        let matrix = collection.containment_matrix(1.0).unwrap();

        assert_eq!(matrix.len(), 0);
    }

    #[test]
    fn test_matrix_single_box() {
        let mut collection = BoxCollection::new();
        collection.push(NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap());

        let matrix = collection.containment_matrix(1.0).unwrap();

        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0].len(), 1);
        assert!((matrix[0][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_disjoint_boxes() {
        let mut collection = BoxCollection::new();

        // Create disjoint boxes
        collection.push(NdarrayBox::new(array![0.0, 0.0], array![0.5, 0.5], 1.0).unwrap());

        collection.push(NdarrayBox::new(array![0.6, 0.6], array![1.0, 1.0], 1.0).unwrap());

        let matrix = collection.containment_matrix(1.0).unwrap();

        // Diagonal should still be 1.0
        assert!((matrix[0][0] - 1.0).abs() < 1e-5);
        assert!((matrix[1][1] - 1.0).abs() < 1e-5);

        // Disjoint boxes should have low containment probabilities
        assert!(
            matrix[0][1] < 0.5,
            "Disjoint boxes should have low containment"
        );
        assert!(
            matrix[1][0] < 0.5,
            "Disjoint boxes should have low containment"
        );
    }
}
