//! Batched ball embedding training on WN18RR using the Burn backend.
//!
//! This is the production-quality version of the ball trainer — uses batched
//! tensor ops through Burn's autodiff backend instead of per-triple SGD.
//! Expected to be 10-100x faster than wn18rr_ball.rs on the same hardware.
//!
//! Run: cargo run --features burn-ndarray --example wn18rr_ball_burn --release
//!
//! Reference: SpherE (Li et al., SIGIR 2024): MRR 0.453 on WN18RR

use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn_ndarray::NdArray;
use std::collections::HashMap;
use std::path::Path;
use subsume::dataset::load_dataset;
use subsume::trainer::burn_ball_trainer::BurnBallTrainer;
use subsume::trainer::{CpuBoxTrainingConfig, FilteredTripleIndexIds};

type Backend = Autodiff<NdArray>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_path = Path::new("data/WN18RR");
    if !data_path.exists() {
        eprintln!("WN18RR data not found at data/WN18RR/");
        eprintln!("Download from: https://github.com/TimDettmers/ConvE");
        std::process::exit(1);
    }

    println!("=== WN18RR Ball Embedding Training (Burn / Batched) ===\n");

    let dataset = load_dataset(data_path)?;
    println!(
        "Dataset: {} train / {} valid / {} test triples",
        dataset.train.len(),
        dataset.valid.len(),
        dataset.test.len()
    );

    let interned = dataset.into_interned();
    let num_entities = interned.entities.len();
    let num_relations = interned.relations.len();
    println!("Entities: {num_entities}, Relations: {num_relations}");

    // Config
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let lr: f32 = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);
    let margin: f32 = std::env::var("MARGIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let neg: usize = std::env::var("NEG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let batch: usize = std::env::var("BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    println!(
        "\nConfig: dim={dim}, epochs={epochs}, lr={lr}, margin={margin}, neg={neg}, batch={batch}"
    );

    let config = CpuBoxTrainingConfig {
        learning_rate: lr,
        margin,
        epochs,
        negative_samples: neg,
        batch_size: batch,
        ..Default::default()
    };

    // Build entity/relation name → index maps
    let mut entity_to_idx: HashMap<String, usize> = HashMap::new();
    for i in 0..num_entities {
        if let Some(name) = interned.entities.get(i) {
            entity_to_idx.insert(name.to_string(), i);
        }
    }
    let mut relation_to_idx: HashMap<String, usize> = HashMap::new();
    for i in 0..num_relations {
        if let Some(name) = interned.relations.get(i) {
            relation_to_idx.insert(name.to_string(), i);
        }
    }

    // Convert interned triples to string format
    let train_triples: Vec<subsume::dataset::Triple> = interned
        .train
        .iter()
        .filter_map(|t| {
            Some(subsume::dataset::Triple {
                head: interned.entities.get(t.head)?.to_string(),
                relation: interned.relations.get(t.relation)?.to_string(),
                tail: interned.entities.get(t.tail)?.to_string(),
            })
        })
        .collect();

    let device = Default::default();
    let mut trainer = BurnBallTrainer::<Backend>::new();
    let mut model = trainer.init_model(num_entities, num_relations, dim, &device);
    let mut optim = AdamConfig::new().init::<Backend, _>();

    println!(
        "\nTraining {epochs} epochs ({} triples/epoch, batch={batch})...\n",
        train_triples.len()
    );

    let mut best_val_mrr = 0.0f32;
    let mut best_epoch = 0usize;

    for epoch in 0..epochs {
        // Cosine LR decay
        let lr_min = lr * 0.01;
        let t = epoch as f32 / epochs.max(1) as f32;
        let epoch_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f32::consts::PI * t).cos());

        let epoch_config = CpuBoxTrainingConfig {
            learning_rate: epoch_lr,
            ..config.clone()
        };

        // Increment epoch seed for different neg sampling each epoch
        trainer.epoch_seed = epoch as u64 * 7919;

        let loss = trainer.train_epoch(
            &mut model,
            &mut optim,
            &train_triples,
            &epoch_config,
            &entity_to_idx,
            &relation_to_idx,
            &device,
        );

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!("  epoch {epoch:>4}/{epochs}: loss = {loss:.6}, lr = {epoch_lr:.6}");

            if (epoch + 1) % 10 == 0 {
                let val_sample_size = 200.min(interned.valid.len());
                let val_sample = &interned.valid[..val_sample_size];
                let results = trainer.evaluate(&model, val_sample, None);
                println!(
                    "    val (sample {val_sample_size}): MRR={:.4}, H@10={:.4}, MR={:.1}",
                    results.mrr, results.hits_at_10, results.mean_rank
                );
                if results.mrr > best_val_mrr {
                    best_val_mrr = results.mrr;
                    best_epoch = epoch;
                }
            }
        }
    }

    println!("\n  Best val MRR: {best_val_mrr:.4} at epoch {best_epoch}");

    // Final test evaluation
    println!("\n--- Test Set Evaluation (filtered) ---\n");
    let filter = FilteredTripleIndexIds::from_dataset(&interned);
    let results = trainer.evaluate(&model, &interned.test, Some(&filter));
    println!("  MRR:       {:.4}", results.mrr);
    println!("  Hits@1:    {:.4}", results.hits_at_1);
    println!("  Hits@3:    {:.4}", results.hits_at_3);
    println!("  Hits@10:   {:.4}", results.hits_at_10);
    println!("  Mean Rank: {:.1}", results.mean_rank);
    println!(
        "\n  ({} test triples, {num_entities} entities, filtered)",
        interned.test.len()
    );

    Ok(())
}
