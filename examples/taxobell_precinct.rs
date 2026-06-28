//! End-to-end: train TaxoBell Gaussian boxes with Burn, then index + query them
//! in precinct (region-as-object nearest-neighbour search).
//!
//! subsume produces *region* embeddings (Gaussian boxes), so a region index
//! (precinct) is the natural serving layer (tranz's *point* embeddings go to
//! vicinity instead). This trains the Burn TaxoBell encoder on a small
//! taxonomy, turns each concept's `(mu, sigma)` into a precinct `AxisBox`
//! (`from_mu_delta`), builds a `RegionIndex`, and runs point-to-region queries.
//!
//! Run: `cargo run --release --example taxobell_precinct --features burn-ndarray,kge`
//!
//! Sample output (queries return the concept itself plus its taxonomic
//! relatives; distance 0 because the query center lies inside those boxes):
//! ```text
//! nearest concept-boxes to a query concept's center:
//!      dog: [dog(0.00), mammal(0.00), cat(0.00), animal(0.00)]
//!     bird: [bird(0.00), animal(0.00), thing(0.00), mammal(0.00)]
//!      oak: [oak(0.00), plant(0.00), thing(0.00), rose(0.00)]
//! ```

#![allow(missing_docs)]

use std::collections::HashMap;

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;

use precinct::{AxisBox, IndexParams, RegionIndex, SearchParams};
use subsume::trainer::burn_taxobell_trainer::{train_taxobell_burn, BurnTaxoBellTrainingConfig};

type Ad = Autodiff<NdArray<f32>>;

fn main() {
    let dev = Default::default();

    // Small animal/plant taxonomy. 0=thing; 1=animal,2=plant; 3=mammal,4=bird;
    // 5=dog,6=cat (mammals); 7=oak,8=rose (plants).
    let names = [
        "thing", "animal", "plant", "mammal", "bird", "dog", "cat", "oak", "rose",
    ];
    let node_ids: Vec<usize> = (0..names.len()).collect();
    let edges = vec![
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (3, 5),
        (3, 6),
        (2, 7),
        (2, 8),
    ];
    // Structured embeddings: an "animal vs plant" axis plus per-node signature.
    let embeddings: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // thing
        vec![1.0, 0.2, 0.0, 0.0, 0.0, 0.0],   // animal
        vec![-1.0, 0.0, 0.2, 0.0, 0.0, 0.0],  // plant
        vec![1.0, 0.8, 0.0, 0.3, 0.0, 0.0],   // mammal
        vec![1.0, 0.8, 0.0, -0.3, 0.0, 0.0],  // bird
        vec![1.0, 1.0, 0.0, 0.3, 0.5, 0.0],   // dog
        vec![1.0, 1.0, 0.0, 0.3, -0.5, 0.0],  // cat
        vec![-1.0, 0.0, 1.0, 0.0, 0.0, 0.5],  // oak
        vec![-1.0, 0.0, 1.0, 0.0, 0.0, -0.5], // rose
    ];
    let node_index: HashMap<usize, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let config = BurnTaxoBellTrainingConfig {
        learning_rate: 5e-3,
        epochs: 400,
        num_negatives: 3,
        hidden_dim: 24,
        box_dim: 8,
        seed: 11,
        warmup_epochs: 20,
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

    // Encode every concept to (mu, sigma).
    let n = embeddings.len();
    let embed_dim = embeddings[0].len();
    let box_dim = enc.box_dim();
    let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
    let x = Tensor::<Ad, 2>::from_data(TensorData::new(flat, [n, embed_dim]), &dev);
    let (mu_t, sigma_t) = enc.encode(x);
    let mu: Vec<f32> = mu_t.into_data().to_vec().unwrap();
    let sigma: Vec<f32> = sigma_t.into_data().to_vec().unwrap();

    // Build a precinct region index: each concept is an axis-aligned box
    // centered at mu with half-widths sigma.
    let mut idx = RegionIndex::<AxisBox>::new(box_dim, IndexParams::default()).expect("index");
    for i in 0..n {
        let m = mu[i * box_dim..(i + 1) * box_dim].to_vec();
        let s = sigma[i * box_dim..(i + 1) * box_dim].to_vec();
        idx.add(i as u32, AxisBox::from_mu_delta(m, s))
            .expect("add");
    }
    idx.build().expect("build");
    println!("indexed {n} concept boxes in precinct");

    // Query: nearest concept-boxes to a few concepts' centers. Related concepts
    // (siblings, parents) should surface near the query.
    let params = || SearchParams {
        ef: 32,
        overretrieve: 16,
    };
    println!("\nnearest concept-boxes to a query concept's center:");
    for &q in &[5usize /*dog*/, 4 /*bird*/, 7 /*oak*/] {
        let center = mu[q * box_dim..(q + 1) * box_dim].to_vec();
        let hits = idx.search(&center, 4, params()).expect("search");
        let got: Vec<String> = hits
            .iter()
            .map(|(id, d)| format!("{}({:.2})", names[*id as usize], d))
            .collect();
        println!("  {:>6}: [{}]", names[q], got.join(", "));
    }
    println!("\nsubsume Burn boxes index and query end-to-end in precinct.");
}
