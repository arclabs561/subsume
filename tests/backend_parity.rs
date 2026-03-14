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

// -- GumbelBox parity --
//
// CandleGumbelBox delegates Box trait methods (volume, intersection, containment_prob)
// to hard-box CandleBox, while NdarrayGumbelBox uses Bessel volume and LSE intersection.
// These are documented to differ (see CandleGumbelBox doc comment). We therefore test
// only the GumbelBox-specific method: membership_probability, which uses identical
// scalar math (gumbel_membership_prob) in both backends.

use subsume::candle_backend::CandleGumbelBox;
use subsume::ndarray_backend::NdarrayGumbelBox;
use subsume::GumbelBox;

fn make_gumbel_pair(
    min: &[f32],
    max: &[f32],
    temperature: f32,
) -> (NdarrayGumbelBox, CandleGumbelBox) {
    let ng = NdarrayGumbelBox::new(
        Array1::from(min.to_vec()),
        Array1::from(max.to_vec()),
        temperature,
    )
    .expect("NdarrayGumbelBox::new");
    let cg = CandleGumbelBox::new(
        Tensor::new(min, &Device::Cpu).unwrap(),
        Tensor::new(max, &Device::Cpu).unwrap(),
        temperature,
    )
    .expect("CandleGumbelBox::new");
    (ng, cg)
}

#[test]
fn gumbel_parity_membership_probability() {
    for &temp in &[0.5, 1.0, 2.0] {
        let (ng, cg) = make_gumbel_pair(&[0.0, 0.0, 0.0], &[2.0, 3.0, 1.0], temp);

        let points: Vec<Vec<f32>> = vec![
            vec![1.0, 1.5, 0.5],  // interior
            vec![0.0, 0.0, 0.0],  // boundary (min)
            vec![2.0, 3.0, 1.0],  // boundary (max)
            vec![-1.0, 1.0, 0.5], // outside
        ];

        for pt in &points {
            let nd_pt = Array1::from(pt.clone());
            let cd_pt = Tensor::new(pt.as_slice(), &Device::Cpu).unwrap();

            let np = ng.membership_probability(&nd_pt).unwrap();
            let cp = cg.membership_probability(&cd_pt).unwrap();
            assert_close(&format!("gumbel_membership temp={temp} pt={pt:?}"), np, cp);
        }
    }
}

// -- Extended parity: union, center, distance, truncate --

fn check_extended_parity(dim: usize, temperature: f32) {
    let tag = format!("dim={} temp={}", dim, temperature);

    let min_a: Vec<f32> = vec![0.0; dim];
    let max_a: Vec<f32> = vec![2.0; dim];
    let min_c: Vec<f32> = vec![1.0; dim];
    let max_c: Vec<f32> = vec![3.0; dim];
    let min_d: Vec<f32> = vec![5.0; dim];
    let max_d: Vec<f32> = vec![6.0; dim];

    let (na, ca) = make_pair(&min_a, &max_a, temperature);
    let (nc, cc) = make_pair(&min_c, &max_c, temperature);
    let (nd, cd) = make_pair(&min_d, &max_d, temperature);

    // -- union --
    let nu = na.union(&nc).unwrap();
    let cu = ca.union(&cc).unwrap();
    let nu_min: Vec<f32> = nu.min().iter().copied().collect();
    let nu_max: Vec<f32> = nu.max().iter().copied().collect();
    let cu_min: Vec<f32> = cu.min().to_vec1::<f32>().unwrap();
    let cu_max: Vec<f32> = cu.max().to_vec1::<f32>().unwrap();
    for i in 0..dim {
        assert_close(&format!("{tag} union_min[{i}]"), nu_min[i], cu_min[i]);
        assert_close(&format!("{tag} union_max[{i}]"), nu_max[i], cu_max[i]);
    }
    // Union volume parity.
    assert_close(
        &format!("{tag} union_vol"),
        nu.volume(temperature).unwrap(),
        cu.volume(temperature).unwrap(),
    );

    // -- center --
    let nc_center: Vec<f32> = na.center().unwrap().iter().copied().collect();
    let cc_center: Vec<f32> = ca.center().unwrap().to_vec1::<f32>().unwrap();
    for i in 0..dim {
        assert_close(&format!("{tag} center[{i}]"), nc_center[i], cc_center[i]);
    }

    // -- distance: overlapping, disjoint --
    assert_close(
        &format!("{tag} dist(A,C)"),
        na.distance(&nc).unwrap(),
        ca.distance(&cc).unwrap(),
    );
    assert_close(
        &format!("{tag} dist(A,D)"),
        na.distance(&nd).unwrap(),
        ca.distance(&cd).unwrap(),
    );

    // -- truncate --
    if dim >= 2 {
        let k = dim / 2;
        let nt = na.truncate(k).unwrap();
        let ct = ca.truncate(k).unwrap();
        assert_eq!(nt.dim(), ct.dim(), "{tag} truncate dim mismatch");
        assert_close(
            &format!("{tag} truncate_vol"),
            nt.volume(temperature).unwrap(),
            ct.volume(temperature).unwrap(),
        );
    }
}

#[test]
fn extended_parity_2d() {
    for &temp in &[0.1, 1.0, 5.0] {
        check_extended_parity(2, temp);
    }
}

#[test]
fn extended_parity_5d() {
    for &temp in &[0.1, 1.0, 5.0] {
        check_extended_parity(5, temp);
    }
}

#[test]
fn extended_parity_10d() {
    for &temp in &[0.1, 1.0, 5.0] {
        check_extended_parity(10, temp);
    }
}

#[test]
fn gumbel_parity_temperature() {
    for &temp in &[0.1, 1.0, 5.0] {
        let (ng, cg) = make_gumbel_pair(&[0.0, 0.0], &[1.0, 1.0], temp);
        assert_close(
            &format!("gumbel_temperature temp={temp}"),
            ng.temperature(),
            cg.temperature(),
        );
    }
}
