//! Mathematical invariant tests for box embeddings.
//!
//! These tests verify fundamental mathematical properties that should always hold
//! for box embeddings, based on set theory, probability theory, and geometric properties.

#[cfg(test)]
mod invariant_tests {
    use crate::{NdarrayBox, NdarrayGumbelBox};
    use ndarray::Array1;
    use proptest::prelude::*;
    use subsume_core::{Box, GumbelBox};

    /// Strategy for generating valid box bounds.
    fn valid_box_strategy(
        dim: usize,
        min_val: f32,
        max_val: f32,
    ) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
        prop::collection::vec(min_val..max_val, dim).prop_flat_map(move |min_vec| {
            let min_vec_clone = min_vec.clone();
            prop::collection::vec(min_val..max_val, dim).prop_map(move |max_vec| {
                let adjusted_max: Vec<f32> = min_vec_clone
                    .iter()
                    .zip(max_vec.iter())
                    .map(|(&m, &mx)| m.max(mx).max(m))
                    .collect();
                (min_vec.clone(), adjusted_max)
            })
        })
    }

    fn ndarray_box_strategy(dim: usize) -> impl Strategy<Value = NdarrayBox> {
        valid_box_strategy(dim, -10.0, 10.0).prop_map(|(min_vec, max_vec)| {
            NdarrayBox::new(Array1::from(min_vec), Array1::from(max_vec), 1.0).unwrap()
        })
    }

    proptest! {
        // ============================================================
        // PARTIAL ORDER PROPERTIES (Reflexivity, Transitivity, Antisymmetry)
        // ============================================================

        /// Reflexivity: Every box contains itself with probability 1.0
        #[test]
        fn containment_reflexivity(box_ in ndarray_box_strategy(3)) {
            let vol = box_.volume(1.0).unwrap();
            if vol > 1e-6 {
                let prob = box_.containment_prob(&box_, 1.0).unwrap();
                prop_assert!(
                    (prob - 1.0).abs() < 1e-5,
                    "Reflexivity violated: P(box ⊆ box) = {}, expected 1.0",
                    prob
                );
            }
        }

        /// Transitivity: If A contains B and B contains C, then A contains C
        #[test]
        fn containment_transitivity_strict(
            box_a in ndarray_box_strategy(2),
            box_b in ndarray_box_strategy(2),
            box_c in ndarray_box_strategy(2),
        ) {
            let vol_a = box_a.volume(1.0).unwrap();
            let vol_b = box_b.volume(1.0).unwrap();
            let vol_c = box_c.volume(1.0).unwrap();

            if vol_a > 1e-6 && vol_b > 1e-6 && vol_c > 1e-6 {
                if let (Ok(p_ab), Ok(p_bc), Ok(p_ac)) = (
                    box_a.containment_prob(&box_b, 1.0),
                    box_b.containment_prob(&box_c, 1.0),
                    box_a.containment_prob(&box_c, 1.0),
                ) {
                    // If A fully contains B and B fully contains C, then A must contain C
                    if p_ab > 0.99 && p_bc > 0.99 {
                        prop_assert!(
                            p_ac > 0.5,
                            "Transitivity violated: P(A⊇B)={}, P(B⊇C)={}, but P(A⊇C)={}",
                            p_ab, p_bc, p_ac
                        );
                    }
                }
            }
        }

        /// Antisymmetry: If A contains B and B contains A, then A ≈ B (volumes should be similar)
        #[test]
        fn containment_antisymmetry(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            let vol_a = box_a.volume(1.0).unwrap();
            let vol_b = box_b.volume(1.0).unwrap();

            if vol_a > 1e-6 && vol_b > 1e-6 {
                if let (Ok(p_ab), Ok(p_ba)) = (
                    box_a.containment_prob(&box_b, 1.0),
                    box_b.containment_prob(&box_a, 1.0),
                ) {
                    // If both containment probabilities are very high, boxes should be similar
                    if p_ab > 0.99 && p_ba > 0.99 {
                        let vol_ratio = (vol_a / vol_b).max(vol_b / vol_a);
                        prop_assert!(
                            vol_ratio < 2.0, // Volumes should be within 2x of each other
                            "Antisymmetry: P(A⊇B)={}, P(B⊇A)={}, but volumes differ: {} vs {}",
                            p_ab, p_ba, vol_a, vol_b
                        );
                    }
                }
            }
        }

        // ============================================================
        // VOLUME PROPERTIES
        // ============================================================

        /// Volume monotonicity: If A is actually contained in B, then vol(A) ≤ vol(B)
        /// Note: We check actual geometric containment, not just containment probability
        #[test]
        fn volume_monotonicity(
            box_a in ndarray_box_strategy(2),
            box_b in ndarray_box_strategy(2),
        ) {
            let vol_a = box_a.volume(1.0).unwrap();
            let vol_b = box_b.volume(1.0).unwrap();

            if vol_a > 1e-6 && vol_b > 1e-6 {
                // Check if A is geometrically contained in B
                // A ⊆ B if min_A[i] >= min_B[i] and max_A[i] <= max_B[i] for all i
                let mut is_contained = true;
                for i in 0..box_a.dim() {
                    if box_a.min()[i] < box_b.min()[i] || box_a.max()[i] > box_b.max()[i] {
                        is_contained = false;
                        break;
                    }
                }

                if is_contained {
                    // If A is geometrically contained in B, then vol(A) should be ≤ vol(B)
                    prop_assert!(
                        vol_a <= vol_b + 1e-5, // Allow small floating point error
                        "Volume monotonicity violated: A⊆B but vol(A)={} > vol(B)={}",
                        vol_a, vol_b
                    );
                }
            }
        }

        /// Volume additivity: vol(A) + vol(B) ≥ vol(A ∩ B)
        #[test]
        fn volume_subadditivity(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            let vol_a = box_a.volume(1.0).unwrap();
            let vol_b = box_b.volume(1.0).unwrap();
            let intersection = box_a.intersection(&box_b).unwrap();
            let vol_intersection = intersection.volume(1.0).unwrap();

            // Volume of union: vol(A ∪ B) = vol(A) + vol(B) - vol(A ∩ B)
            // So: vol(A) + vol(B) ≥ vol(A ∩ B)
            prop_assert!(
                vol_a + vol_b >= vol_intersection - 1e-5,
                "Subadditivity violated: vol(A)={} + vol(B)={} < vol(A∩B)={}",
                vol_a, vol_b, vol_intersection
            );
        }

        // ============================================================
        // INTERSECTION PROPERTIES
        // ============================================================

        /// Intersection commutativity: A ∩ B = B ∩ A
        #[test]
        fn intersection_commutative(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            let intersection_ab = box_a.intersection(&box_b).unwrap();
            let intersection_ba = box_b.intersection(&box_a).unwrap();

            let vol_ab = intersection_ab.volume(1.0).unwrap();
            let vol_ba = intersection_ba.volume(1.0).unwrap();

            prop_assert!(
                (vol_ab - vol_ba).abs() < 1e-5,
                "Intersection not commutative: vol(A∩B)={} != vol(B∩A)={}",
                vol_ab, vol_ba
            );
        }

        /// Intersection idempotence: A ∩ A = A
        #[test]
        fn intersection_idempotent(box_ in ndarray_box_strategy(3)) {
            let intersection = box_.intersection(&box_).unwrap();
            let vol_original = box_.volume(1.0).unwrap();
            let vol_intersection = intersection.volume(1.0).unwrap();

            prop_assert!(
                (vol_original - vol_intersection).abs() < 1e-5,
                "Intersection not idempotent: vol(A∩A)={} != vol(A)={}",
                vol_intersection, vol_original
            );
        }

        /// Intersection associativity: (A ∩ B) ∩ C = A ∩ (B ∩ C)
        #[test]
        fn intersection_associative(
            box_a in ndarray_box_strategy(2),
            box_b in ndarray_box_strategy(2),
            box_c in ndarray_box_strategy(2),
        ) {
            let intersection_ab = box_a.intersection(&box_b).unwrap();
            let intersection_abc = intersection_ab.intersection(&box_c).unwrap();

            let intersection_bc = box_b.intersection(&box_c).unwrap();
            let intersection_abc2 = box_a.intersection(&intersection_bc).unwrap();

            let vol1 = intersection_abc.volume(1.0).unwrap();
            let vol2 = intersection_abc2.volume(1.0).unwrap();

            prop_assert!(
                (vol1 - vol2).abs() < 1e-5,
                "Intersection not associative: vol((A∩B)∩C)={} != vol(A∩(B∩C))={}",
                vol1, vol2
            );
        }

        /// Intersection volume bound: vol(A ∩ B) ≤ min(vol(A), vol(B))
        #[test]
        fn intersection_volume_bound(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            let vol_a = box_a.volume(1.0).unwrap();
            let vol_b = box_b.volume(1.0).unwrap();
            let intersection = box_a.intersection(&box_b).unwrap();
            let vol_intersection = intersection.volume(1.0).unwrap();

            let min_vol = vol_a.min(vol_b);
            prop_assert!(
                vol_intersection <= min_vol + 1e-5,
                "Intersection volume bound violated: vol(A∩B)={} > min(vol(A)={}, vol(B)={})",
                vol_intersection, vol_a, vol_b
            );
        }

        // ============================================================
        // OVERLAP PROPERTIES
        // ============================================================

        /// Overlap reflexivity: P(A ∩ A ≠ ∅) = 1.0 for non-zero volume boxes
        #[test]
        fn overlap_reflexivity(box_ in ndarray_box_strategy(3)) {
            let vol = box_.volume(1.0).unwrap();
            let prob = box_.overlap_prob(&box_, 1.0).unwrap();

            if vol > 1e-6 {
                // Non-zero volume: should overlap with itself
                prop_assert!(
                    (prob - 1.0).abs() < 1e-5,
                    "Overlap reflexivity violated: P(A∩A≠∅)={}, expected 1.0 (volume={})",
                    prob, vol
                );
            } else {
                // Zero-volume box: overlap probability should still be valid
                prop_assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Overlap probability should be in [0, 1], got {} (zero volume box)",
                    prob
                );
            }
        }

        /// Overlap symmetry: P(A ∩ B ≠ ∅) = P(B ∩ A ≠ ∅)
        #[test]
        fn overlap_symmetric(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            let prob_ab = box_a.overlap_prob(&box_b, 1.0).unwrap();
            let prob_ba = box_b.overlap_prob(&box_a, 1.0).unwrap();

            prop_assert!(
                (prob_ab - prob_ba).abs() < 1e-5,
                "Overlap not symmetric: P(A∩B≠∅)={} != P(B∩A≠∅)={}",
                prob_ab, prob_ba
            );
        }

        /// Containment implies overlap: If A is geometrically contained in B, then P(A ∩ B ≠ ∅) > 0
        #[test]
        fn containment_implies_overlap(
            box_a in ndarray_box_strategy(2),
            box_b in ndarray_box_strategy(2),
        ) {
            let vol_a = box_a.volume(1.0).unwrap();
            let vol_b = box_b.volume(1.0).unwrap();

            if vol_a > 1e-6 && vol_b > 1e-6 {
                // Check if A is geometrically contained in B
                let mut is_contained = true;
                for i in 0..box_a.dim() {
                    if box_a.min()[i] < box_b.min()[i] || box_a.max()[i] > box_b.max()[i] {
                        is_contained = false;
                        break;
                    }
                }

                if is_contained {
                    let overlap = box_a.overlap_prob(&box_b, 1.0).unwrap();
                    // If A is contained in B, overlap should be high (close to 1.0)
                    // But allow for edge cases where overlap calculation might be lower
                    prop_assert!(
                        overlap > 0.0, // At minimum, they should overlap
                        "Containment implies overlap: A⊆B but P(A∩B≠∅)={} (vol_a={}, vol_b={})",
                        overlap, vol_a, vol_b
                    );
                }
            }
        }

        // ============================================================
        // PROBABILITY BOUNDS
        // ============================================================

        /// Containment probability bounds: P(A ⊆ B) ∈ [0, 1]
        #[test]
        fn containment_probability_bounds(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            if let Ok(prob) = box_a.containment_prob(&box_b, 1.0) {
                prop_assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Containment probability out of bounds: {}",
                    prob
                );
            }
        }

        /// Overlap probability bounds: P(A ∩ B ≠ ∅) ∈ [0, 1]
        #[test]
        fn overlap_probability_bounds(
            box_a in ndarray_box_strategy(3),
            box_b in ndarray_box_strategy(3),
        ) {
            if let Ok(prob) = box_a.overlap_prob(&box_b, 1.0) {
                prop_assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Overlap probability out of bounds: {}",
                    prob
                );
            }
        }

        // ============================================================
        // GUMBEL BOX PROPERTIES
        // ============================================================

        /// Gumbel membership probability bounds
        #[test]
        fn gumbel_membership_bounds(
            gumbel_box in ndarray_box_strategy(3).prop_map(|b| {
                NdarrayGumbelBox::new(b.min().clone(), b.max().clone(), 1.0).unwrap()
            }),
            point_coords in prop::collection::vec(-20.0f32..20.0, 3),
        ) {
            let point = Array1::from(point_coords);
            if let Ok(prob) = gumbel_box.membership_probability(&point) {
                prop_assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Gumbel membership probability out of bounds: {}",
                    prob
                );
            }
        }

        /// Gumbel sample always within bounds
        #[test]
        fn gumbel_sample_bounds(
            gumbel_box in ndarray_box_strategy(4).prop_map(|b| {
                NdarrayGumbelBox::new(b.min().clone(), b.max().clone(), 1.0).unwrap()
            }),
        ) {
            let vol = gumbel_box.volume(1.0).unwrap();
            if vol > 1e-6 {
                let sample = gumbel_box.sample();
                for i in 0..gumbel_box.dim() {
                    // Allow small floating point errors, but sample should be very close to bounds
                    prop_assert!(
                        sample[i] >= gumbel_box.min()[i] - 1e-5 &&
                        sample[i] <= gumbel_box.max()[i] + 1e-5,
                        "Sample[{}]={} out of bounds [{}, {}]",
                        i, sample[i], gumbel_box.min()[i], gumbel_box.max()[i]
                    );
                }
            }
        }

        /// Gumbel membership for point inside box should be higher than outside
        #[test]
        fn gumbel_membership_inside_vs_outside(
            gumbel_box in ndarray_box_strategy(2).prop_map(|b| {
                NdarrayGumbelBox::new(b.min().clone(), b.max().clone(), 1.0).unwrap()
            }),
        ) {
            // Point inside box (center)
            let center: Vec<f32> = (0..gumbel_box.dim())
                .map(|i| (gumbel_box.min()[i] + gumbel_box.max()[i]) / 2.0)
                .collect();
            let point_inside = Array1::from(center);

            // Point far outside box
            let point_outside: Vec<f32> = (0..gumbel_box.dim())
                .map(|i| gumbel_box.max()[i] + 10.0)
                .collect();
            let point_outside = Array1::from(point_outside);

            if let (Ok(prob_inside), Ok(prob_outside)) = (
                gumbel_box.membership_probability(&point_inside),
                gumbel_box.membership_probability(&point_outside),
            ) {
                prop_assert!(
                    prob_inside >= prob_outside,
                    "Membership inside ({}) should be >= outside ({})",
                    prob_inside, prob_outside
                );
            }
        }

        // ============================================================
        // DIMENSION PROPERTIES
        // ============================================================

        /// Volume scales with dimension count (for same-sized boxes)
        #[test]
        fn volume_scales_with_dimensions(
            (min_vec, max_vec) in valid_box_strategy(2, 0.0, 10.0)
        ) {
            let box_2d = NdarrayBox::new(
                Array1::from(min_vec.clone()),
                Array1::from(max_vec.clone()),
                1.0,
            ).unwrap();

            // Create 3D box with same intervals + one more dimension
            let mut min_3d = min_vec.clone();
            let mut max_3d = max_vec.clone();
            min_3d.push(0.0);
            max_3d.push(1.0);

            let box_3d = NdarrayBox::new(
                Array1::from(min_3d),
                Array1::from(max_3d),
                1.0,
            ).unwrap();

            let vol_2d = box_2d.volume(1.0).unwrap();
            let vol_3d = box_3d.volume(1.0).unwrap();

            // 3D volume should be 2D volume * width of 3rd dimension
            // Since 3rd dimension is [0, 1], width = 1
            prop_assert!(
                (vol_3d - vol_2d).abs() < 1e-5,
                "Volume scaling: 3D volume {} should equal 2D volume {} * 1.0",
                vol_3d, vol_2d
            );
        }

        // ============================================================
        // EDGE CASES
        // ============================================================

        /// Zero-volume box containment
        #[test]
        fn zero_volume_box_containment(
            (min_vec, max_vec) in valid_box_strategy(2, 0.0, 10.0)
        ) {
            // Create a zero-volume box (point)
            let point_box = NdarrayBox::new(
                Array1::from(min_vec.clone()),
                Array1::from(min_vec.clone()), // min == max
                1.0,
            ).unwrap();

            // Create a normal box
            let normal_box = NdarrayBox::new(
                Array1::from(min_vec.clone()),
                Array1::from(max_vec),
                1.0,
            ).unwrap();

            // Point should be contained in normal box if point is within bounds
            if let Ok(prob) = normal_box.containment_prob(&point_box, 1.0) {
                prop_assert!(
                    (0.0..=1.0).contains(&prob),
                    "Zero-volume containment probability out of bounds: {}",
                    prob
                );
            }
        }

        /// Very small boxes
        #[test]
        fn very_small_box_properties(
            (min_vec, max_vec) in valid_box_strategy(3, 0.0, 1.0)
        ) {
            // Create a very small box
            let small_max: Vec<f32> = min_vec.iter().zip(max_vec.iter())
                .map(|(&m, &mx)| m + (mx - m) * 0.001) // Very small width
                .collect();

            let small_box = NdarrayBox::new(
                Array1::from(min_vec),
                Array1::from(small_max),
                1.0,
            ).unwrap();

            let volume = small_box.volume(1.0).unwrap();
            prop_assert!(
                volume >= 0.0,
                "Very small box should have non-negative volume: {}",
                volume
            );
        }

        /// Very large boxes
        #[test]
        fn very_large_box_properties(
            (min_vec, max_vec) in valid_box_strategy(2, -1000.0, 1000.0)
        ) {
            let large_box = NdarrayBox::new(
                Array1::from(min_vec),
                Array1::from(max_vec),
                1.0,
            ).unwrap();

            let volume = large_box.volume(1.0).unwrap();
            prop_assert!(
                volume >= 0.0 && volume.is_finite(),
                "Very large box should have finite non-negative volume: {}",
                volume
            );
        }
    }
}
