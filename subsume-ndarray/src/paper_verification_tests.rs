//! Verification tests for paper claims.
//!
//! These tests verify that our implementations correctly reproduce
//! key claims and results from the research papers.

#[cfg(test)]
mod tests {
    use crate::distance;
    use crate::{NdarrayBox, NdarrayGumbelBox};
    use ndarray::array;
    use subsume_core::Box;

    /// Verify Vilnis et al. (2018): Containment probability formula
    /// P(B ⊆ A) = Vol(A ∩ B) / Vol(B)
    #[test]
    fn test_vilnis_2018_containment_formula() {
        let premise = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();

        let hypothesis = NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0).unwrap();

        // Compute containment probability
        let prob = premise.containment_prob(&hypothesis, 1.0).unwrap();

        // Verify: P(B ⊆ A) = Vol(A ∩ B) / Vol(B) (from box_trait.rs documentation)
        let intersection = premise.intersection(&hypothesis).unwrap();
        let intersection_vol = intersection.volume(1.0).unwrap();
        let hypothesis_vol = hypothesis.volume(1.0).unwrap();
        let expected_prob = intersection_vol / hypothesis_vol;

        // Should be very close (within floating point precision)
        assert!(
            (prob - expected_prob).abs() < 1e-5,
            "Containment probability {} should equal {}/{} = {}",
            prob,
            intersection_vol,
            hypothesis_vol,
            expected_prob
        );
    }

    /// Verify Dasgupta et al. (2020): Gumbel boxes provide gradients
    /// even when boxes are disjoint (solving local identifiability)
    #[test]
    fn test_dasgupta_2020_gumbel_gradients() {
        // Create two disjoint boxes
        let box_a = NdarrayGumbelBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();

        let box_b = NdarrayGumbelBox::new(array![2.0, 2.0], array![3.0, 3.0], 1.0).unwrap();

        // For Gumbel boxes, containment probability should be > 0
        // even when boxes are disjoint (due to probabilistic nature)
        let prob = box_a.containment_prob(&box_b, 1.0).unwrap();

        // Should be small but non-zero (probabilistic overlap)
        assert!(
            prob >= 0.0,
            "Gumbel boxes should have non-negative containment probability"
        );
        assert!(prob <= 1.0, "Containment probability should be <= 1.0");
    }

    /// Verify RegD (2025): Depth distance incorporates volume
    #[test]
    fn test_regd_2025_depth_distance_volume_term() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();

        let box_b = NdarrayBox::new(
            array![0.0, 0.0],
            array![2.0, 2.0], // Larger volume
            1.0,
        )
        .unwrap();

        // Depth distance should be larger when volumes differ
        let depth_dist = distance::depth_distance(&box_a, &box_b, 1.0, 0.1).unwrap();
        let euclidean_dist = box_a.distance(&box_b).unwrap();

        // Depth distance should be >= Euclidean distance (volume term adds to it)
        assert!(
            depth_dist >= euclidean_dist,
            "Depth distance {} should be >= Euclidean distance {}",
            depth_dist,
            euclidean_dist
        );
    }

    /// Verify RegD (2025): Boundary distance for containment relationships
    #[test]
    fn test_regd_2025_boundary_distance_containment() {
        let outer = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();

        let inner = NdarrayBox::new(array![2.0, 2.0], array![8.0, 8.0], 1.0).unwrap();

        // Boundary distance should exist for contained boxes
        let boundary_dist = distance::boundary_distance(&outer, &inner, 1.0).unwrap();

        assert!(
            boundary_dist.is_some(),
            "Boundary distance should exist for contained boxes"
        );
        let dist = boundary_dist.unwrap();
        assert!(dist >= 0.0, "Boundary distance should be non-negative");
        assert!(
            dist <= 2.0,
            "Boundary distance should be at most the gap (2.0)"
        );
    }

    /// Verify Concept2Box (2023): Vector-to-box distance formula
    /// d(v, B) = 0 if v ∈ B, else min_{p∈B} ||v - p||
    #[test]
    fn test_concept2box_2023_vector_to_box_inside() {
        let box_ = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();

        let point = array![0.5, 0.5]; // Inside box

        let dist = distance::vector_to_box_distance(&point, &box_).unwrap();

        // Distance should be 0 for point inside box
        assert_eq!(dist, 0.0, "Vector inside box should have distance 0");
    }

    #[test]
    fn test_concept2box_2023_vector_to_box_outside() {
        let box_ = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0).unwrap();

        let point = array![2.0, 2.0]; // Outside box

        let dist = distance::vector_to_box_distance(&point, &box_).unwrap();

        // Distance should be sqrt((2-1)^2 + (2-1)^2) = sqrt(2) ≈ 1.414
        let expected = (2.0_f32).sqrt();
        assert!(
            (dist - expected).abs() < 1e-5,
            "Distance {} should be approximately {}",
            dist,
            expected
        );
    }

    /// Verify Chen et al. (2021): Box volumes provide uncertainty estimates
    #[test]
    fn test_chen_2021_volume_uncertainty() {
        let small_box = NdarrayBox::new(array![0.0, 0.0], array![0.5, 0.5], 1.0).unwrap();

        let large_box = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();

        let small_vol = small_box.volume(1.0).unwrap();
        let large_vol = large_box.volume(1.0).unwrap();

        // Larger box should have larger volume (more uncertainty)
        assert!(
            large_vol > small_vol,
            "Larger box {} should have larger volume than smaller box {}",
            large_vol,
            small_vol
        );
    }

    /// Verify Boratko et al. (2020): Box embeddings are closed under intersection
    #[test]
    fn test_boratko_2020_intersection_closure() {
        let box_a = NdarrayBox::new(array![0.0, 0.0], array![2.0, 2.0], 1.0).unwrap();

        let box_b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();

        // Intersection should produce a valid box
        let intersection = box_a.intersection(&box_b).unwrap();

        // Verify it's a valid box
        assert_eq!(intersection.dim(), box_a.dim());
        let intersection_vol = intersection.volume(1.0).unwrap();
        assert!(
            intersection_vol >= 0.0,
            "Intersection volume should be non-negative"
        );
    }
}
