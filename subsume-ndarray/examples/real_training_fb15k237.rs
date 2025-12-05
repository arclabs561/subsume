//! Real training example on FB15k-237 dataset.
//!
//! This example demonstrates training box embeddings for knowledge graph completion
//! on the FB15k-237 dataset (Freebase subset).
//!
//! # Dataset Setup
//!
//! Download FB15k-237 from: https://github.com/TimDettmers/ConvE
//! Place in `data/fb15k-237/` directory with files:
//! - train.txt
//! - valid.txt
//! - test.txt
//!
//! # Usage
//!
//! ```bash
//! cargo run --example real_training_fb15k237
//! ```

use ndarray::Array1;
use subsume_core::dataset::load_dataset;
use subsume_core::trainer::{
    evaluate_link_prediction, generate_negative_samples, NegativeSamplingStrategy, TrainingConfig,
};
use subsume_core::training::diagnostics::TrainingStats;
use subsume_core::Box as CoreBox;
use subsume_ndarray::{AdamW, NdarrayBox};
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Real Training Example: FB15k-237 Knowledge Graph Completion");
    println!("============================================================\n");

    let dataset_path = env::args()
        .nth(1)
        .map(|p| Path::new(&p).to_path_buf())
        .unwrap_or_else(|| Path::new("data/fb15k-237").to_path_buf());

    println!("Loading dataset from: {:?}", dataset_path);

    let dataset = match load_dataset(&dataset_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("\nError loading dataset: {}", e);
            eprintln!("\nTo use this example:");
            eprintln!("1. Download FB15k-237 from: https://github.com/TimDettmers/ConvE");
            eprintln!("2. Extract to: data/fb15k-237/");
            eprintln!("3. Ensure files exist: train.txt, valid.txt, test.txt");
            eprintln!("\nFor now, using synthetic data...\n");
            return Err(Box::new(e));
        }
    };

    let stats = dataset.stats();
    println!("Dataset Statistics:");
    println!("  Entities: {}", stats.num_entities);
    println!("  Relations: {}", stats.num_relations);
    println!("  Train triples: {}", stats.num_train);
    println!("  Valid triples: {}", stats.num_valid);
    println!("  Test triples: {}\n", stats.num_test);

    // FB15k-237 is large, so use smaller subset for demo
    let max_train = 10000.min(stats.num_train);
    let train_subset: Vec<_> = dataset.train.iter().take(max_train).cloned().collect();
    println!("Using {} training triples (subset for demo)", train_subset.len());

    let entities: HashSet<String> = dataset.entities();

    // Initialize embeddings
    let embedding_dim = 100; // Higher dimension for FB15k-237
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    
    println!("Initializing {} box embeddings (dim={})...", entities.len(), embedding_dim);
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for entity in &entities {
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

    // Training config optimized for FB15k-237
    let config = TrainingConfig {
        learning_rate: 5e-4, // Slightly lower for stability
        epochs: 50,
        batch_size: 1024, // Larger batch size for efficiency
        negative_samples: 5, // More negatives for better learning
        negative_strategy: NegativeSamplingStrategy::CorruptTail,
        regularization_weight: 1e-5,
        temperature: 1.0,
        weight_decay: 1e-4, // Higher weight decay for regularization
        margin: 1.0,
        early_stopping_patience: Some(5),
    };

    println!("Training Configuration:");
    println!("  Optimizer: AdamW (with weight decay)");
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Weight decay: {}", config.weight_decay);
    println!("  Epochs: {}", config.epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Negative samples: {}\n", config.negative_samples);

    // Use AdamW for better generalization
    let mut optimizer = AdamW::new(config.learning_rate, config.weight_decay);
    let mut training_stats = TrainingStats::new(config.epochs);
    let mut best_valid_mrr = 0.0;
    let mut patience_counter = 0;

    println!("Starting training...\n");

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        use rand::seq::SliceRandom;
        let mut shuffled = train_subset.clone();
        shuffled.shuffle(&mut rng);

        for batch in shuffled.chunks(config.batch_size) {
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

        // Validation
        let valid_subset: Vec<_> = dataset.valid.iter().take(1000).cloned().collect();
        let valid_results = evaluate_link_prediction::<NdarrayBox>(&valid_subset, &entity_boxes, None)?;
        let valid_mrr = valid_results.mrr;

        let avg_volume = entity_boxes
            .values()
            .map(|b| b.volume(config.temperature).unwrap_or(0.0))
            .sum::<f32>()
            / entity_boxes.len() as f32;

        training_stats.record(avg_loss, avg_volume, 0.1);

        if valid_mrr > best_valid_mrr {
            best_valid_mrr = valid_mrr;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            println!(
                "Epoch {}: Loss = {:.4}, Valid MRR = {:.4}, Hits@10 = {:.4}",
                epoch + 1, avg_loss, valid_mrr, valid_results.hits_at_10
            );
        }

        if patience_counter >= config.early_stopping_patience.unwrap_or(10) {
            println!("\nEarly stopping at epoch {}", epoch + 1);
            break;
        }
    }

    println!("\nTraining complete!\n");

    // Test evaluation
    let test_subset: Vec<_> = dataset.test.iter().take(1000).cloned().collect();
    let test_results = evaluate_link_prediction::<NdarrayBox>(&test_subset, &entity_boxes, None)?;

    println!("=== Final Test Results ===");
    println!("MRR:      {:.4}", test_results.mrr);
    println!("Hits@1:   {:.4}", test_results.hits_at_1);
    println!("Hits@10:  {:.4}", test_results.hits_at_10);
    println!("Mean Rank: {:.2}", test_results.mean_rank);

    Ok(())
}

