//! Real training example on WN18RR dataset.
//!
//! This example demonstrates a complete training pipeline for knowledge graph
//! link prediction using the WN18RR dataset.
//!
//! # Dataset Setup
//!
//! Download WN18RR from: https://github.com/kkteru/grail
//! Place in `data/wn18rr/` directory with files:
//! - train.txt
//! - valid.txt
//! - test.txt
//!
//! # Usage
//!
//! ```bash
//! # With dataset in default location
//! cargo run --example real_training_wn18rr
//!
//! # With custom dataset path
//! cargo run --example real_training_wn18rr -- data/custom_wn18rr
//! ```

use ndarray::Array1;
use subsume_core::dataset::{load_dataset, Dataset};
use subsume_core::trainer::{
    evaluate_link_prediction, generate_negative_samples, NegativeSamplingStrategy, TrainingConfig,
};
use subsume_core::training::{
    diagnostics::TrainingStats,
};
use subsume_core::Box as CoreBox;
use subsume_ndarray::{Adam, NdarrayBox};
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Real Training Example: WN18RR Link Prediction");
    println!("===============================================\n");

    // Get dataset path from command line or use default
    let dataset_path = env::args()
        .nth(1)
        .map(|p| Path::new(&p).to_path_buf())
        .unwrap_or_else(|| Path::new("data/wn18rr").to_path_buf());

    println!("Loading dataset from: {:?}", dataset_path);

    // Load dataset
    let dataset = match load_dataset(&dataset_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("\nError loading dataset: {}", e);
            eprintln!("\nTo use this example:");
            eprintln!("1. Download WN18RR from: https://github.com/kkteru/grail");
            eprintln!("2. Extract to: data/wn18rr/");
            eprintln!("3. Ensure files exist: train.txt, valid.txt, test.txt");
            eprintln!("\nFor now, using synthetic data...\n");
            
            // Fallback to synthetic data
            create_synthetic_dataset()
        }
    };

    let stats = dataset.stats();
    println!("Dataset Statistics:");
    println!("  Entities: {}", stats.num_entities);
    println!("  Relations: {}", stats.num_relations);
    println!("  Train triples: {}", stats.num_train);
    println!("  Valid triples: {}", stats.num_valid);
    println!("  Test triples: {}\n", stats.num_test);

    // Collect all entities
    let entities: HashSet<String> = dataset.entities();

    // Initialize box embeddings
    let embedding_dim = 50; // Standard dimension for box embeddings
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    
    println!("Initializing {} box embeddings (dim={})...", entities.len(), embedding_dim);
    
    // Initialize with small random boxes (centered around origin)
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for entity in &entities {
        // Random initialization: boxes near origin with small size
        let center: Vec<f32> = (0..embedding_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let size: Vec<f32> = (0..embedding_dim)
            .map(|_| rng.gen_range(0.1..0.3))
            .collect();
        
        let min = Array1::from_iter(
            center.iter().zip(size.iter())
                .map(|(c, s)| c - s / 2.0)
        );
        let max = Array1::from_iter(
            center.iter().zip(size.iter())
                .map(|(c, s)| c + s / 2.0)
        );
        
        let box_ = NdarrayBox::new(min, max, 1.0)?;
        entity_boxes.insert(entity.clone(), box_);
    }

    // Training configuration (paper defaults)
    let config = TrainingConfig {
        learning_rate: 1e-3, // Paper default
        epochs: 100,
        batch_size: 512, // Paper range: 512-4096
        negative_samples: 1,
        negative_strategy: NegativeSamplingStrategy::CorruptTail,
        regularization_weight: 1e-5,
        temperature: 1.0,
        weight_decay: 1e-5,
        margin: 1.0,
        early_stopping_patience: Some(10),
    };

    println!("Training Configuration:");
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Epochs: {}", config.epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Negative samples: {}", config.negative_samples);
    println!("  Embedding dim: {}\n", embedding_dim);

    // Create Adam optimizer
    let mut optimizer = Adam::new(config.learning_rate);

    // Training diagnostics
    let mut training_stats = TrainingStats::new(config.epochs);
    let mut best_valid_mrr = 0.0;
    let mut patience_counter = 0;

    println!("Starting training...\n");

    // Training loop
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        // Shuffle training data (simplified - in production, use proper shuffling)
        let mut train_triples = dataset.train.clone();
        use rand::seq::SliceRandom;
        train_triples.shuffle(&mut rng);

        for batch in train_triples.chunks(config.batch_size) {
            for triple in batch {
                if let (Some(head_box), Some(tail_box)) = (
                    entity_boxes.get(&triple.head),
                    entity_boxes.get(&triple.tail),
                ) {
                    // Positive sample: maximize containment
                    let pos_score = head_box.containment_prob(tail_box, config.temperature)?;
                    let mut batch_loss = 1.0 - pos_score;

                    // Generate negative samples
                    let negatives = generate_negative_samples(
                        triple,
                        &entities,
                        &config.negative_strategy,
                        config.negative_samples,
                    );

                    // Negative samples: minimize containment
                    for neg in negatives.iter().take(config.negative_samples) {
                        if let Some(neg_tail_box) = entity_boxes.get(&neg.tail) {
                            let neg_score = head_box.containment_prob(neg_tail_box, config.temperature)?;
                            // Margin-based loss: max(0, margin - pos_score + neg_score)
                            batch_loss += (config.margin - pos_score + neg_score).max(0.0);
                        }
                    }

                    // Update head box (simplified gradient)
                    // Clone tail values before mutable borrow
                    let tail_min = tail_box.min().to_owned();
                    let tail_max = tail_box.max().to_owned();
                    
                    let head_box_mut = entity_boxes.get_mut(&triple.head).unwrap();
                    let mut head_min = head_box_mut.min().to_owned();

                    // Approximate gradient: move head box to contain tail box
                    let grad_min_vec: Vec<f32> = head_min
                        .iter()
                        .zip(tail_min.iter())
                        .map(|(h, t)| batch_loss * 0.01 * (t - h))
                        .collect();
                    let grad_min = Array1::from_vec(grad_min_vec);
                    optimizer.update(&format!("{}_min", triple.head), &mut head_min, grad_min.view());

                    let mut head_max = head_box_mut.max().to_owned();
                    let grad_max_vec: Vec<f32> = head_max
                        .iter()
                        .zip(tail_max.iter())
                        .map(|(h, t)| batch_loss * 0.01 * (t - h))
                        .collect();
                    let grad_max = Array1::from_vec(grad_max_vec);
                    optimizer.update(&format!("{}_max", triple.head), &mut head_max, grad_max.view());

                    *head_box_mut = NdarrayBox::new(head_min, head_max, config.temperature)?;

                    epoch_loss += batch_loss;
                }
            }
            batch_count += 1;
        }

        let avg_loss = epoch_loss / (batch_count as f32).max(1.0);

        // Evaluate on validation set
        let valid_results = evaluate_link_prediction::<NdarrayBox>(&dataset.valid, &entity_boxes, None)?;
        let valid_mrr = valid_results.mrr;

        // Record training stats
        let avg_volume = entity_boxes
            .values()
            .map(|b| b.volume(config.temperature).unwrap_or(0.0))
            .sum::<f32>()
            / entity_boxes.len() as f32;

        training_stats.record(avg_loss, avg_volume, 0.1);

        // Early stopping
        if valid_mrr > best_valid_mrr {
            best_valid_mrr = valid_mrr;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }

        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            println!(
                "Epoch {}: Loss = {:.4}, Valid MRR = {:.4}, Hits@10 = {:.4}",
                epoch + 1,
                avg_loss,
                valid_mrr,
                valid_results.hits_at_10
            );
        }

        if config.early_stopping_patience.is_some()
            && patience_counter >= config.early_stopping_patience.unwrap()
        {
            println!("\nEarly stopping at epoch {} (no improvement for {} epochs)", 
                epoch + 1, patience_counter);
            break;
        }
    }

    println!("\nTraining complete!\n");

    // Final evaluation on test set
    println!("Evaluating on test set...");
    let test_results = evaluate_link_prediction::<NdarrayBox>(&dataset.test, &entity_boxes, None)?;

    println!("\n=== Final Test Results ===");
    println!("MRR:      {:.4}", test_results.mrr);
    println!("Hits@1:   {:.4}", test_results.hits_at_1);
    println!("Hits@3:   {:.4}", test_results.hits_at_3);
    println!("Hits@10:  {:.4}", test_results.hits_at_10);
    println!("Mean Rank: {:.2}", test_results.mean_rank);

    // Print training summary
    println!("\n=== Training Summary ===");
    if let Some((mean, min, max)) = training_stats.loss_stats() {
        println!("Final loss - Mean: {:.4}, Min: {:.4}, Max: {:.4}", mean, min, max);
    }
    if let Some((mean, min, max)) = training_stats.volume_stats() {
        println!("Volume - Mean: {:.4}, Min: {:.4}, Max: {:.4}", mean, min, max);
    }
    println!("Best valid MRR: {:.4}", best_valid_mrr);

    Ok(())
}

/// Create synthetic dataset for testing when real dataset is not available.
fn create_synthetic_dataset() -> Dataset {
    use subsume_core::dataset::Triple;
    
    // Create a small synthetic knowledge graph
    let train = vec![
        Triple { head: "animal".to_string(), relation: "hypernym".to_string(), tail: "entity".to_string() },
        Triple { head: "mammal".to_string(), relation: "hypernym".to_string(), tail: "animal".to_string() },
        Triple { head: "dog".to_string(), relation: "hypernym".to_string(), tail: "mammal".to_string() },
        Triple { head: "cat".to_string(), relation: "hypernym".to_string(), tail: "mammal".to_string() },
        Triple { head: "bird".to_string(), relation: "hypernym".to_string(), tail: "animal".to_string() },
        Triple { head: "sparrow".to_string(), relation: "hypernym".to_string(), tail: "bird".to_string() },
    ];
    
    let valid = vec![
        Triple { head: "fish".to_string(), relation: "hypernym".to_string(), tail: "animal".to_string() },
    ];
    
    let test = vec![
        Triple { head: "whale".to_string(), relation: "hypernym".to_string(), tail: "mammal".to_string() },
    ];
    
    Dataset::new(train, valid, test)
}

