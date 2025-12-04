//! Property-based tests for box embeddings using proptest.

#[cfg(test)]
mod proptest_tests {
    use crate::{NdarrayBox, NdarrayGumbelBox};
    use ndarray::Array1;
    use proptest::prelude::*;
    use subsume_core::{Box, GumbelBox};


    /// Strategy for generating valid box bounds (min <= max for all dimensions).
    fn valid_box_strategy(dim: usize, min_val: f32, max_val: f32) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
        prop::collection::vec(min_val..max_val, dim)
            .prop_flat_map(move |min_vec| {
                let min_vec_clone = min_vec.clone();
                prop::collection::vec(min_val..max_val, dim)
                    .prop_map(move |max_vec| {
                        // Ensure min[i] <= max[i] for all i
                        let adjusted_max: Vec<f32> = min_vec_clone
                            .iter()
                            .zip(max_vec.iter())
                            .map(|(&m, &mx)| m.max(mx).max(m)) // Ensure max >= min
                            .collect();
                        (min_vec.clone(), adjusted_max)
                    })
            })
    }

    /// Strategy for generating a valid NdarrayBox.
    fn ndarray_box_strategy(dim: usize) -> impl Strategy<Value = NdarrayBox> {
        valid_box_strategy(dim, -10.0, 10.0)
            .prop_map(|(min_vec, max_vec)| {
                NdarrayBox::new(
                    Array1::from(min_vec),
                    Array1::from(max_vec),
                    1.0,
                )
                .unwrap()
            })
    }

    proptest! {
        #[test]
        fn volume_is_non_negative(box_ in ndarray_box_strategy(3)) {
            let volume = box_.volume(1.0).unwrap();
            prop_assert!(volume >= 0.0, "Volume must be non-negative, got {}", volume);
        }

        #[test]
        fn volume_scales_with_size(
            (min_vec, max_vec) in valid_box_strategy(2, 0.0, 10.0)
        ) {
            let box1 = NdarrayBox::new(
                Array1::from(min_vec.clone()),
                Array1::from(max_vec.clone()),
                1.0,
            ).unwrap();
            
            // Create a larger box by scaling
            let scaled_max: Vec<f32> = max_vec.iter().map(|&x| x * 2.0).collect();
            let box2 = NdarrayBox::new(
                Array1::from(min_vec),
                Array1::from(scaled_max),
                1.0,
            ).unwrap();
            
            let vol1 = box1.volume(1.0).unwrap();
            let vol2 = box2.volume(1.0).unwrap();
            
            // Volume should increase (or stay same if zero volume)
            prop_assert!(vol2 >= vol1, "Larger box should have larger or equal volume");
        }

        #[test]
        fn containment_probability_bounds(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            if let Ok(prob) = box_a.containment_prob(&box_b, 1.0) {
                prop_assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Containment probability must be in [0, 1], got {}",
                    prob
                );
            }
        }

        #[test]
        fn containment_reflexive(box_ in ndarray_box_strategy(3)) {
            // A box always contains itself (if it has non-zero volume)
            let volume = box_.volume(1.0).unwrap();
            if volume > 1e-6 {
                let prob = box_.containment_prob(&box_, 1.0).unwrap();
                prop_assert!((prob - 1.0).abs() < 1e-5, "Box should contain itself, got {}", prob);
            }
        }

        #[test]
        fn containment_transitive(
            box_a in ndarray_box_strategy(2),
            box_b in ndarray_box_strategy(2),
            box_c in ndarray_box_strategy(2),
        ) {
            // If A contains B with high probability, and B contains C with high probability,
            // then A should contain C with at least moderate probability
            // Skip if any box has zero volume
            let vol_a = box_a.volume(1.0).unwrap();
            let vol_b = box_b.volume(1.0).unwrap();
            let vol_c = box_c.volume(1.0).unwrap();
            
            if vol_a > 1e-6 && vol_b > 1e-6 && vol_c > 1e-6 {
                if let (Ok(p_ab), Ok(p_bc), Ok(p_ac)) = (
                    box_a.containment_prob(&box_b, 1.0),
                    box_b.containment_prob(&box_c, 1.0),
                    box_a.containment_prob(&box_c, 1.0),
                ) {
                    // Transitivity: if p_ab and p_bc are both high, p_ac should be reasonably high
                    if p_ab > 0.9 && p_bc > 0.9 {
                        prop_assert!(
                            p_ac > 0.5,
                            "Transitivity violated: P(A⊇B)={}, P(B⊇C)={}, but P(A⊇C)={}",
                            p_ab, p_bc, p_ac
                        );
                    }
                }
            }
        }

        #[test]
        fn overlap_probability_bounds(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            if let Ok(prob) = box_a.overlap_prob(&box_b, 1.0) {
                prop_assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Overlap probability must be in [0, 1], got {}",
                    prob
                );
            }
        }

        #[test]
        fn intersection_volume_leq_original(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            let intersection = box_a.intersection(&box_b).unwrap();
            let vol_a = box_a.volume(1.0).unwrap();
            let vol_b = box_b.volume(1.0).unwrap();
            let vol_intersection = intersection.volume(1.0).unwrap();
            
            prop_assert!(
                vol_intersection <= vol_a && vol_intersection <= vol_b,
                "Intersection volume {} must be <= both volumes ({} and {})",
                vol_intersection, vol_a, vol_b
            );
        }

        #[test]
        fn gumbel_membership_probability_bounds(
            gumbel_box in ndarray_box_strategy(3).prop_map(|b| {
                NdarrayGumbelBox::new(b.min().clone(), b.max().clone(), 1.0).unwrap()
            }),
            point_coords in prop::collection::vec(-20.0f32..20.0, 3),
        ) {
            let point = Array1::from(point_coords);
            if let Ok(prob) = gumbel_box.membership_probability(&point) {
                prop_assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Membership probability must be in [0, 1], got {}",
                    prob
                );
            }
        }

        #[test]
        fn gumbel_sample_within_bounds(
            gumbel_box in ndarray_box_strategy(3).prop_map(|b| {
                NdarrayGumbelBox::new(b.min().clone(), b.max().clone(), 1.0).unwrap()
            }),
        ) {
            let sample = gumbel_box.sample();
            for i in 0..gumbel_box.dim() {
                prop_assert!(
                    sample[i] >= gumbel_box.min()[i] && sample[i] <= gumbel_box.max()[i],
                    "Sample[{}] = {} must be in [{}, {}]",
                    i, sample[i], gumbel_box.min()[i], gumbel_box.max()[i]
                );
            }
        }

        #[test]
        fn volume_zero_iff_any_dimension_zero(
            (min_vec, max_vec) in valid_box_strategy(2, 0.0, 10.0)
        ) {
            let box_ = NdarrayBox::new(
                Array1::from(min_vec.clone()),
                Array1::from(max_vec.clone()),
                1.0,
            ).unwrap();
            
            let volume = box_.volume(1.0).unwrap();
            let has_zero_dim = min_vec.iter().zip(max_vec.iter())
                .any(|(&m, &mx)| (mx - m).abs() < 1e-6);
            
            if has_zero_dim {
                prop_assert!(
                    volume.abs() < 1e-5,
                    "Box with zero dimension should have zero volume, got {}",
                    volume
                );
            }
        }

        #[test]
        fn serialization_round_trip(box_ in ndarray_box_strategy(3)) {
            let serialized = serde_json::to_string(&box_).unwrap();
            let deserialized: NdarrayBox = serde_json::from_str(&serialized).unwrap();
            
            prop_assert_eq!(box_.dim(), deserialized.dim());
            prop_assert_eq!(box_.temperature, deserialized.temperature);
            // Compare volumes to verify correctness
            let vol_orig = box_.volume(1.0).unwrap();
            let vol_deser = deserialized.volume(1.0).unwrap();
            prop_assert!(
                (vol_orig - vol_deser).abs() < 1e-5,
                "Volumes should match: {} vs {}",
                vol_orig, vol_deser
            );
        }
    }
}

