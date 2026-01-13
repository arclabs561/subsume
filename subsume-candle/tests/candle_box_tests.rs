//! Tests for CandleBox implementation.
//!
//! These tests verify that the Candle backend correctly implements
//! the Box trait with the same semantics as the ndarray backend.

use candle_core::{Device, Tensor};
use subsume_candle::CandleBox;
use subsume_core::Box;

fn device() -> Device {
    Device::Cpu
}

#[test]
fn test_box_creation() {
    let min = Tensor::new(&[0.0f32, 0.0, 0.0], &device()).unwrap();
    let max = Tensor::new(&[1.0f32, 1.0, 1.0], &device()).unwrap();
    let box_a = CandleBox::new(min, max, 1.0).unwrap();
    assert_eq!(box_a.dim(), 3);
}

#[test]
fn test_box_creation_invalid_bounds() {
    let min = Tensor::new(&[1.0f32, 0.0], &device()).unwrap();
    let max = Tensor::new(&[0.0f32, 1.0], &device()).unwrap();
    assert!(CandleBox::new(min, max, 1.0).is_err());
}

#[test]
fn test_volume() {
    let min = Tensor::new(&[0.0f32, 0.0], &device()).unwrap();
    let max = Tensor::new(&[2.0f32, 3.0], &device()).unwrap();
    let box_a = CandleBox::new(min, max, 1.0).unwrap();
    let volume = box_a.volume(1.0).unwrap();
    assert!((volume - 6.0).abs() < 1e-6);
}

#[test]
fn test_volume_zero() {
    let min = Tensor::new(&[0.0f32, 0.0], &device()).unwrap();
    let max = Tensor::new(&[0.0f32, 1.0], &device()).unwrap();
    let box_a = CandleBox::new(min, max, 1.0).unwrap();
    let volume = box_a.volume(1.0).unwrap();
    assert!((volume - 0.0).abs() < 1e-6);
}

#[test]
fn test_intersection() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[2.0f32, 2.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let box_b = CandleBox::new(
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        Tensor::new(&[3.0f32, 3.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let intersection = box_a.intersection(&box_b).unwrap();
    let volume = intersection.volume(1.0).unwrap();
    assert!((volume - 1.0).abs() < 1e-6);
}

#[test]
fn test_intersection_disjoint() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let box_b = CandleBox::new(
        Tensor::new(&[2.0f32, 2.0], &device()).unwrap(),
        Tensor::new(&[3.0f32, 3.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let intersection = box_a.intersection(&box_b).unwrap();
    let volume = intersection.volume(1.0).unwrap();
    assert!((volume - 0.0).abs() < 1e-6);
}

#[test]
fn test_containment_prob() {
    let premise = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let hypothesis = CandleBox::new(
        Tensor::new(&[0.2f32, 0.2], &device()).unwrap(),
        Tensor::new(&[0.8f32, 0.8], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let prob = premise.containment_prob(&hypothesis, 1.0).unwrap();
    assert!(prob > 0.9);
}

#[test]
fn test_containment_prob_disjoint() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let box_b = CandleBox::new(
        Tensor::new(&[2.0f32, 2.0], &device()).unwrap(),
        Tensor::new(&[3.0f32, 3.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let prob = box_a.containment_prob(&box_b, 1.0).unwrap();
    assert!((prob - 0.0).abs() < 1e-6);
}

#[test]
fn test_overlap_prob() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let box_b = CandleBox::new(
        Tensor::new(&[0.5f32, 0.5], &device()).unwrap(),
        Tensor::new(&[1.5f32, 1.5], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let prob = box_a.overlap_prob(&box_b, 1.0).unwrap();
    assert!(prob > 0.0);
    assert!(prob <= 1.0);
}

#[test]
fn test_overlap_prob_identical() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let prob = box_a.overlap_prob(&box_a, 1.0).unwrap();
    assert!((prob - 1.0).abs() < 1e-6);
}

#[test]
fn test_union() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let box_b = CandleBox::new(
        Tensor::new(&[0.5f32, 0.5], &device()).unwrap(),
        Tensor::new(&[1.5f32, 1.5], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let union_box = box_a.union(&box_b).unwrap();
    // Union should span [0,0] to [1.5,1.5]
    let volume = union_box.volume(1.0).unwrap();
    assert!((volume - 2.25).abs() < 1e-6);
}

#[test]
fn test_center() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[2.0f32, 4.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let center = box_a.center().unwrap();
    let center_vec = center.to_vec1::<f32>().unwrap();
    assert!((center_vec[0] - 1.0).abs() < 1e-6);
    assert!((center_vec[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_distance_overlapping() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[2.0f32, 2.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let box_b = CandleBox::new(
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        Tensor::new(&[3.0f32, 3.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let dist = box_a.distance(&box_b).unwrap();
    assert!((dist - 0.0).abs() < 1e-6);
}

#[test]
fn test_distance_separated() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let box_b = CandleBox::new(
        Tensor::new(&[2.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[3.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let dist = box_a.distance(&box_b).unwrap();
    assert!((dist - 1.0).abs() < 1e-6);
}

#[test]
fn test_dimension_mismatch() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 1.0], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    let box_b = CandleBox::new(
        Tensor::new(&[0.0f32], &device()).unwrap(),
        Tensor::new(&[1.0f32], &device()).unwrap(),
        1.0,
    )
    .unwrap();
    assert!(box_a.intersection(&box_b).is_err());
    assert!(box_a.containment_prob(&box_b, 1.0).is_err());
}

#[test]
fn test_serialization_roundtrip() {
    let box_a = CandleBox::new(
        Tensor::new(&[0.0f32, 1.0, 2.0], &device()).unwrap(),
        Tensor::new(&[1.0f32, 2.0, 3.0], &device()).unwrap(),
        1.5,
    )
    .unwrap();

    let serialized = serde_json::to_string(&box_a).unwrap();
    let deserialized: CandleBox = serde_json::from_str(&serialized).unwrap();

    assert_eq!(box_a.dim(), deserialized.dim());

    // Verify operations work the same after round-trip
    let vol_orig = box_a.volume(1.0).unwrap();
    let vol_deser = deserialized.volume(1.0).unwrap();
    assert!((vol_orig - vol_deser).abs() < 1e-6);

    let center_orig = box_a.center().unwrap().to_vec1::<f32>().unwrap();
    let center_deser = deserialized.center().unwrap().to_vec1::<f32>().unwrap();
    for (a, b) in center_orig.iter().zip(center_deser.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}
