//! Composition: analytic uncertainty on TaxoBell box centers via stableprop.
//!
//! subsume's `BurnTaxoBellEncoder` maps text embeddings to Gaussian boxes
//! `(mu, sigma)`. When the input embedding is itself uncertain (noisy retrieval,
//! a paraphrase, an approximate embedding), how uncertain is the resulting box
//! center? `stableprop` answers this analytically: it propagates the input
//! variance through the encoder's center path (`Linear -> ReLU -> Linear`) in
//! one forward pass, no sampling.
//!
//! This trains a small encoder, then compares two analytic modes (diagonal and
//! full-covariance) against a Monte Carlo estimate over many noisy inputs.
//! Full-covariance tracks the MC magnitude; diagonal preserves the ranking
//! across concepts but underestimates magnitude (it drops hidden-unit
//! covariance). Either way, stableprop gives subsume's box-center uncertainty
//! without sampling.
//!
//! Run: `cargo run --release --example taxobell_box_uncertainty --features burn-ndarray,kge`

use burn::backend::Autodiff;
use burn::tensor::{Distribution, Tensor, TensorData};
use burn_ndarray::NdArray;
use std::collections::HashMap;

use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_full, propagate_relu, propagate_relu_full, Moments,
    MomentsFull,
};
use subsume::trainer::burn_taxobell_trainer::{train_taxobell_burn, BurnTaxoBellTrainingConfig};

type Ad = Autodiff<NdArray<f32>>;

fn main() {
    let dev = Default::default();

    // Tiny taxonomy: 0 is the root; 1,2 are children of 0; 3 of 1; 4 of 2.
    let node_ids = vec![0usize, 1, 2, 3, 4];
    let edges = vec![(0, 1), (0, 2), (1, 3), (2, 4)]; // (parent, child)
    let embeddings: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.2, -0.1, 0.3],
        vec![0.6, 0.5, 0.0, 0.1, 0.2, 0.0],
        vec![0.6, 0.0, 0.5, -0.2, 0.1, 0.1],
        vec![0.3, 0.7, 0.0, 0.0, 0.4, -0.2],
        vec![0.3, 0.0, 0.7, 0.2, -0.3, 0.2],
    ];
    let node_index: HashMap<usize, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let config = BurnTaxoBellTrainingConfig {
        learning_rate: 5e-3,
        epochs: 200,
        num_negatives: 2,
        hidden_dim: 16,
        box_dim: 8,
        seed: 7,
        warmup_epochs: 10,
        ..Default::default()
    };

    let (enc, losses) =
        train_taxobell_burn::<Ad>(&embeddings, &edges, &node_ids, &node_index, &config, &dev)
            .expect("training failed");
    println!(
        "trained TaxoBell encoder: loss {:.3} -> {:.3}",
        losses[0],
        losses.last().unwrap()
    );

    let n = embeddings.len();
    let embed_dim = embeddings[0].len();
    let box_dim = enc.box_dim();
    let std = 0.1f64;

    let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
    let x = Tensor::<Ad, 2>::from_data(TensorData::new(flat, [n, embed_dim]), &dev);

    // --- Analytic: propagate input variance through the center path ---
    let var0 = Tensor::<Ad, 2>::full([n, embed_dim], std * std, &dev);
    let (w1, b1, w2, b2) = enc.center_weights();

    // Diagonal propagation: cheap, but drops hidden-unit covariance.
    let h_diag = propagate_relu(&propagate_linear(
        &Moments::new(x.clone(), var0.clone()),
        w1.clone(),
        b1.clone(),
    ));
    let diag_var: Vec<f32> = propagate_linear(&h_diag, w2.clone(), b2.clone())
        .var
        .into_data()
        .to_vec()
        .unwrap();

    // Full-covariance propagation: keeps the hidden covariance matrix.
    let h_full = propagate_relu_full(&propagate_linear_full(
        &MomentsFull::from_diagonal(x.clone(), var0),
        w1,
        b1,
    ));
    let full_var: Vec<f32> = propagate_linear_full(&h_full, w2, b2)
        .variance()
        .into_data()
        .to_vec()
        .unwrap();

    // --- Monte Carlo: empirical variance of mu over noisy inputs ---
    let draws = 4000usize;
    let mut sum = vec![0.0f64; n * box_dim];
    let mut sumsq = vec![0.0f64; n * box_dim];
    for _ in 0..draws {
        let noise = Tensor::<Ad, 2>::random([n, embed_dim], Distribution::Normal(0.0, std), &dev);
        let (mu, _sigma) = enc.encode(x.clone() + noise);
        let muv: Vec<f32> = mu.into_data().to_vec().unwrap();
        for (j, &v) in muv.iter().enumerate() {
            sum[j] += v as f64;
            sumsq[j] += (v as f64) * (v as f64);
        }
    }
    let mc_var: Vec<f32> = (0..n * box_dim)
        .map(|j| {
            let mean = sum[j] / draws as f64;
            (sumsq[j] / draws as f64 - mean * mean).max(0.0) as f32
        })
        .collect();

    // --- Report: per-node total box-center variance, analytic vs MC ---
    let total = |v: &[f32], i: usize| -> f32 { v[i * box_dim..(i + 1) * box_dim].iter().sum() };
    println!("\nbox-center uncertainty (sum of per-dim variance), input noise std={std}:");
    println!("  node   diagonal    full-cov   monte-carlo");
    for i in 0..n {
        println!(
            "  {i:>4}   {:>8.4}   {:>8.4}   {:>11.4}",
            total(&diag_var, i),
            total(&full_var, i),
            total(&mc_var, i)
        );
    }
    println!(
        "\nBoth analytic modes recover the box-center uncertainty in one forward pass, no sampling."
    );
    println!("Diagonal propagation preserves the ranking across concepts but underestimates");
    println!("magnitude (it drops hidden-unit covariance); full-covariance propagation tracks the");
    println!(
        "Monte Carlo magnitude. stableprop gives subsume's box-center uncertainty either way."
    );
}
