//! Backend parity tests: verify ndarray and candle backends produce identical results.

#![cfg(all(feature = "ndarray-backend", feature = "candle-backend"))]

use candle_core::{Device, Tensor};
use ndarray::Array1;
use subsume::candle_backend::CandleBox;
use subsume::ndarray_backend::NdarrayBox;
use subsume::Box as BoxTrait;

const EPS: f32 = 1e-5;

/// Build an NdarrayBox and CandleBox from the same raw coordinates.
fn make_pair(min: &[f32], max: &[f32], temperature: f32) -> (NdarrayBox, CandleBox) {
    let nb = NdarrayBox::new(
        Array1::from(min.to_vec()),
        Array1::from(max.to_vec()),
        temperature,
    )
    .expect("NdarrayBox::new");
    let cb = CandleBox::new(
        Tensor::new(min, &Device::Cpu).unwrap(),
        Tensor::new(max, &Device::Cpu).unwrap(),
        temperature,
    )
    .expect("CandleBox::new");
    (nb, cb)
}

fn assert_close(label: &str, a: f32, b: f32) {
    assert!(
        (a - b).abs() < EPS,
        "{}: ndarray={} candle={} diff={}",
        label,
        a,
        b,
        (a - b).abs()
    );
}

/// Run all parity checks for a given dimensionality and temperature.
fn check_parity(dim: usize, temperature: f32) {
    let tag = format!("dim={} temp={}", dim, temperature);

    // Box A: [0..2] in every dimension
    let min_a: Vec<f32> = vec![0.0; dim];
    let max_a: Vec<f32> = vec![2.0; dim];

    // Box B (contained in A): [0.5..1.5]
    let min_b: Vec<f32> = vec![0.5; dim];
    let max_b: Vec<f32> = vec![1.5; dim];

    // Box C (partially overlapping A): [1..3]
    let min_c: Vec<f32> = vec![1.0; dim];
    let max_c: Vec<f32> = vec![3.0; dim];

    // Box D (disjoint from A): [5..6]
    let min_d: Vec<f32> = vec![5.0; dim];
    let max_d: Vec<f32> = vec![6.0; dim];

    let (na, ca) = make_pair(&min_a, &max_a, temperature);
    let (nb, cb) = make_pair(&min_b, &max_b, temperature);
    let (nc, cc) = make_pair(&min_c, &max_c, temperature);
    let (nd, cd) = make_pair(&min_d, &max_d, temperature);

    // -- volume --
    assert_close(
        &format!("{tag} volume(A)"),
        na.volume(temperature).unwrap(),
        ca.volume(temperature).unwrap(),
    );

    // -- containment_prob: contained, overlapping, disjoint --
    assert_close(
        &format!("{tag} contain(A,B)"),
        na.containment_prob(&nb, temperature).unwrap(),
        ca.containment_prob(&cb, temperature).unwrap(),
    );
    assert_close(
        &format!("{tag} contain(A,C)"),
        na.containment_prob(&nc, temperature).unwrap(),
        ca.containment_prob(&cc, temperature).unwrap(),
    );
    assert_close(
        &format!("{tag} contain(A,D)"),
        na.containment_prob(&nd, temperature).unwrap(),
        ca.containment_prob(&cd, temperature).unwrap(),
    );

    // -- overlap_prob --
    assert_close(
        &format!("{tag} overlap(A,B)"),
        na.overlap_prob(&nb, temperature).unwrap(),
        ca.overlap_prob(&cb, temperature).unwrap(),
    );
    assert_close(
        &format!("{tag} overlap(A,C)"),
        na.overlap_prob(&nc, temperature).unwrap(),
        ca.overlap_prob(&cc, temperature).unwrap(),
    );
    assert_close(
        &format!("{tag} overlap(A,D)"),
        na.overlap_prob(&nd, temperature).unwrap(),
        ca.overlap_prob(&cd, temperature).unwrap(),
    );

    // -- intersection: compare resulting min/max coordinates --
    let ni = na.intersection(&nc).unwrap();
    let ci = ca.intersection(&cc).unwrap();
    let ni_min: Vec<f32> = ni.min().iter().copied().collect();
    let ni_max: Vec<f32> = ni.max().iter().copied().collect();
    let ci_min: Vec<f32> = ci.min().to_vec1::<f32>().unwrap();
    let ci_max: Vec<f32> = ci.max().to_vec1::<f32>().unwrap();
    for i in 0..dim {
        assert_close(&format!("{tag} isect_min[{i}]"), ni_min[i], ci_min[i]);
        assert_close(&format!("{tag} isect_max[{i}]"), ni_max[i], ci_max[i]);
    }

    // Disjoint intersection produces zero-volume box in both backends
    let ni_d = na.intersection(&nd).unwrap();
    let ci_d = ca.intersection(&cd).unwrap();
    assert_close(
        &format!("{tag} isect_disjoint_vol"),
        ni_d.volume(temperature).unwrap(),
        ci_d.volume(temperature).unwrap(),
    );
}

#[test]
fn parity_2d() {
    for &temp in &[0.1, 1.0, 5.0] {
        check_parity(2, temp);
    }
}

#[test]
fn parity_5d() {
    for &temp in &[0.1, 1.0, 5.0] {
        check_parity(5, temp);
    }
}

#[test]
fn parity_10d() {
    for &temp in &[0.1, 1.0, 5.0] {
        check_parity(10, temp);
    }
}
