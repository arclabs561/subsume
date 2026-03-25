//! Train box embeddings on WN18RR (full benchmark).
//!
//! WN18RR is the standard WordNet benchmark for knowledge graph embeddings
//! (Dettmers et al., 2018). 40,943 entities, 11 relations, 86,835 training
//! triples. This example trains a BoxEmbeddingTrainer and evaluates with
//! filtered link prediction metrics (MRR, Hits@k).
//!
//! Expects WN18RR data at `data/WN18RR/{train,valid,test}.txt` in TSV format.
//! Download from: https://github.com/TimDettmers/ConvE
//!
//! Run: cargo run -p subsume --example wn18rr_training --release
//!
//! Reference metrics (box methods on WN18RR):
//!   BoxE (Abboud et al., 2020): MRR 0.451, Hits@10 0.541
//!   Query2Box (Ren et al., 2020): MRR ~0.40 (link prediction mode)

use std::path::Path;
use subsume::dataset::load_dataset;
use subsume::trainer::{BoxEmbeddingTrainer, FilteredTripleIndexIds};
use subsume::TrainingConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_path = Path::new("data/WN18RR");
    if !data_path.exists() {
        eprintln!("WN18RR data not found at data/WN18RR/");
        eprintln!("Download from: https://github.com/TimDettmers/ConvE");
        eprintln!("Expected files: train.txt, valid.txt, test.txt (tab-separated)");
        std::process::exit(1);
    }

    println!("=== WN18RR Box Embedding Training ===\n");

    let dataset = load_dataset(data_path)?;
    println!(
        "Dataset: {} train / {} valid / {} test triples",
        dataset.train.len(),
        dataset.valid.len(),
        dataset.test.len()
    );

    println!(
        "Entities: {}, Relations: {}",
        dataset.entities().len(),
        dataset.relations().len()
    );

    // Intern the dataset for efficient training.
    let interned = dataset.into_interned();
    let train_triples: Vec<(usize, usize, usize)> = interned
        .train
        .iter()
        .map(|t| (t.head, t.relation, t.tail))
        .collect();
    let entities = &interned.entities;

    // Configuration via environment variables for remote training.
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let lr: f32 = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5e-4);
    let neg: usize = std::env::var("NEG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let self_adv: bool = std::env::var("SELF_ADV")
        .ok()
        .map(|s| s == "1")
        .unwrap_or(false);

    let config = TrainingConfig {
        learning_rate: lr,
        epochs,
        batch_size: 512,
        negative_samples: neg,
        margin: 0.5,
        regularization: 1e-4,
        negative_weight: 1.0,
        softplus_beta: 5.0,
        softplus_beta_final: 20.0,
        max_grad_norm: 5.0,
        adversarial_temperature: 1.0,
        warmup_epochs: (epochs / 10).max(1),
        self_adversarial: self_adv,
        ..Default::default()
    };
    let mut trainer = BoxEmbeddingTrainer::new(config.clone(), dim);

    // Ensure all entities exist in the trainer.
    for id in 0..entities.len() {
        trainer.ensure_entity(id);
    }

    println!(
        "\nTraining {} epochs (dim={dim}, batch_size={})...\n",
        config.epochs, config.batch_size
    );

    // Train in batches.
    let _n_batches = train_triples.len().div_ceil(config.batch_size);
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for batch_start in (0..train_triples.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(train_triples.len());
            let batch = &train_triples[batch_start..batch_end];
            let loss = trainer.train_step(batch)?;
            epoch_loss += loss;
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count as f32;

        // Print loss every epoch for live progress.
        println!(
            "  epoch {epoch:>3}/{}: avg_loss = {avg_loss:.6}",
            config.epochs
        );

        // Quick validation on a small sample (first 50 triples) every 10 epochs.
        // Full evaluation is too slow for 40K entities at every checkpoint.
        if (epoch + 1) % 10 == 0 {
            let sample_size = 50.min(interned.valid.len());
            let val_sample = &interned.valid[..sample_size];
            let filter = FilteredTripleIndexIds::from_dataset(&interned);
            match trainer.evaluate(val_sample, entities, Some(&filter)) {
                Ok(results) => {
                    println!(
                        "    val (sample {sample_size}): MRR={:.4}, H@10={:.4}, MR={:.1}",
                        results.mrr, results.hits_at_10, results.mean_rank
                    );
                }
                Err(e) => println!("    val: error: {e}"),
            }
        }
    }

    // Final evaluation on test set.
    println!("\n--- Test Set Evaluation (filtered) ---\n");
    let test_triples = &interned.test;
    let filter = FilteredTripleIndexIds::from_dataset(&interned);
    let results = trainer.evaluate(test_triples, entities, Some(&filter))?;
    println!("  MRR:       {:.4}", results.mrr);
    println!("  Hits@1:    {:.4}", results.hits_at_1);
    println!("  Hits@3:    {:.4}", results.hits_at_3);
    println!("  Hits@10:   {:.4}", results.hits_at_10);
    println!("  Mean Rank: {:.1}", results.mean_rank);
    println!(
        "\n  ({} test triples, {} entities, filtered ranking)",
        interned.test.len(),
        entities.len()
    );

    // Save checkpoint.
    let checkpoint = serde_json::to_string(&trainer)?;
    let checkpoint_path = "data/WN18RR/checkpoint.json";
    std::fs::write(checkpoint_path, &checkpoint)?;
    println!(
        "\nCheckpoint saved: {} ({} bytes)",
        checkpoint_path,
        checkpoint.len()
    );

    Ok(())
}
