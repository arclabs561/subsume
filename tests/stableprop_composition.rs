//! Integration checks for propagating uncertainty through TaxoBell's center MLP.
#![cfg(all(feature = "burn-ndarray", feature = "kge"))]

use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_full, propagate_relu, propagate_relu_full, Moments,
    MomentsFull,
};
use subsume::trainer::burn_taxobell_trainer::BurnTaxoBellEncoder;

type Backend = NdArray<f32>;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn taxobell_center_weights_compose_with_both_stableprop_covariance_modes() {
    let device = Default::default();
    let encoder = BurnTaxoBellEncoder::<Backend>::new(3, 4, 2, &device);
    let mean = Tensor::from_data(
        TensorData::new(vec![0.2f32, -0.4, 0.7, -0.3, 0.8, 0.1], [2, 3]),
        &device,
    );
    let variance = Tensor::from_data(
        TensorData::new(vec![0.01f32, 0.04, 0.09, 0.02, 0.03, 0.05], [2, 3]),
        &device,
    );
    let (w1, b1, w2, b2) = encoder.center_weights();

    let diagonal = propagate_linear(
        &propagate_relu(&propagate_linear(
            &Moments::new(mean.clone(), variance.clone()),
            w1.clone(),
            b1.clone(),
        )),
        w2.clone(),
        b2.clone(),
    );
    let full = propagate_linear_full(
        &propagate_relu_full(&propagate_linear_full(
            &MomentsFull::from_diagonal(mean, variance),
            w1,
            b1,
        )),
        w2,
        b2,
    );

    assert_eq!(diagonal.mean.dims(), [2, 2]);
    assert_eq!(diagonal.var.dims(), [2, 2]);
    assert_eq!(full.mean.dims(), [2, 2]);
    assert_eq!(full.cov.dims(), [2, 2, 2]);
    assert_close(
        &diagonal.mean.into_data().to_vec::<f32>().unwrap(),
        &full.mean.into_data().to_vec::<f32>().unwrap(),
        1e-5,
    );

    let covariance = full.cov.into_data().to_vec::<f32>().unwrap();
    for matrix in covariance.chunks_exact(4) {
        assert_close(&[matrix[1]], &[matrix[2]], 1e-5);
        assert!(matrix[0].is_finite() && matrix[0] >= 0.0);
        assert!(matrix[3].is_finite() && matrix[3] >= 0.0);
    }
}
