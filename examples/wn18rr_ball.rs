//! Train ball embeddings on WN18RR (full benchmark).
//!
//! Compares ball embeddings (SpherE + RegD) against box embeddings on the
//! standard WN18RR benchmark. 40,943 entities, 11 relations, 86,835 training
//! triples.
//!
//! Expects WN18RR data at `data/WN18RR/{train,valid,test}.txt` in TSV format.
//!
//! Run: cargo run -p subsume --example wn18rr_ball --release
//!
//! Reference metrics:
//!   SpherE (Li et al., 2024): MRR 0.453, Hits@10 0.537 on WN18RR
//!   BoxE (Abboud et al., 2020): MRR 0.451, Hits@10 0.541

use std::collections::HashMap;
use std::path::Path;
use subsume::dataset::load_dataset;
use subsume::trainer::ball_trainer::BallTrainer;
use subsume::trainer::{CpuBoxTrainingConfig, FilteredTripleIndexIds};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_path = Path::new("data/WN18RR");
    if !data_path.exists() {
        eprintln!("WN18RR data not found at data/WN18RR/");
        eprintln!("Download from: https://github.com/TimDettmers/ConvE");
        eprintln!("Expected files: train.txt, valid.txt, test.txt (tab-separated)");
        std::process::exit(1);
    }

    println!("=== WN18RR Ball Embedding Training ===\n");

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

    let interned = dataset.into_interned();

    // Configuration via environment variables.
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let lr: f32 = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);
    let margin: f32 = std::env::var("MARGIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let neg: usize = std::env::var("NEG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let adv_temp: f32 = std::env::var("ADV_TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let num_entities = interned.entities.len();
    let num_relations = interned.relations.len();

    println!("\nConfig: dim={dim}, epochs={epochs}, lr={lr}, margin={margin}, neg={neg}, adv_temp={adv_temp}");

    // Build entity/relation maps
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

    // Convert interned triples back to Triple (string) format for the trainer
    let train_triples: Vec<subsume::dataset::Triple> = interned
        .train
        .iter()
        .filter_map(|t| {
            let head = interned.entities.get(t.head)?.to_string();
            let relation = interned.relations.get(t.relation)?.to_string();
            let tail = interned.entities.get(t.tail)?.to_string();
            Some(subsume::dataset::Triple {
                head,
                relation,
                tail,
            })
        })
        .collect();

    let config = CpuBoxTrainingConfig {
        learning_rate: lr,
        margin,
        epochs,
        negative_samples: neg,
        adversarial_temperature: adv_temp,
        ..Default::default()
    };

    let mut trainer = BallTrainer::new(42);
    let (mut entities, mut relations) = trainer.init_embeddings(num_entities, num_relations, dim);

    println!(
        "\nTraining {} epochs (dim={dim}, {} entities, {} relations)...\n",
        epochs, num_entities, num_relations
    );

    let mut best_val_mrr = 0.0f32;
    let mut best_epoch = 0usize;

    for epoch in 0..epochs {
        // Cosine LR decay: lr_epoch = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * t/T))
        let lr_min = lr * 0.01;
        let t = epoch as f32 / epochs.max(1) as f32;
        let epoch_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f32::consts::PI * t).cos());

        let epoch_config = CpuBoxTrainingConfig {
            learning_rate: epoch_lr,
            ..config.clone()
        };

        let loss = trainer.train_epoch(
            &mut entities,
            &mut relations,
            &train_triples,
            &epoch_config,
            &entity_to_idx,
            &relation_to_idx,
        );

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!("  epoch {epoch:>3}/{epochs}: avg_loss = {loss:.6}, lr = {epoch_lr:.5}");

            // Quick validation
            if (epoch + 1) % 10 == 0 {
                let sample_size = 100.min(interned.valid.len());
                let val_sample = &interned.valid[..sample_size];
                let results = trainer.evaluate(&entities, &relations, val_sample, None);
                println!(
                    "    val (sample {sample_size}): MRR={:.4}, H@10={:.4}, MR={:.1}",
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
    // Final evaluation on test set.
    println!("\n--- Test Set Evaluation ---\n");
    let filter = FilteredTripleIndexIds::from_dataset(&interned);
    let results = trainer.evaluate(&entities, &relations, &interned.test, Some(&filter));
    println!("  MRR:       {:.4}", results.mrr);
    println!("  Hits@1:    {:.4}", results.hits_at_1);
    println!("  Hits@3:    {:.4}", results.hits_at_3);
    println!("  Hits@10:   {:.4}", results.hits_at_10);
    println!("  Mean Rank: {:.1}", results.mean_rank);
    println!(
        "\n  ({} test triples, {} entities, filtered ranking)",
        interned.test.len(),
        num_entities
    );

    Ok(())
}
