//! Complete training loop example with dataset integration.
//!
//! This example demonstrates:
//! 1. Loading a knowledge graph dataset
//! 2. Training box embeddings with negative sampling
//! 3. Evaluating on test set
//! 4. Using all diagnostic tools

use ndarray::Array1;
use subsume_core::dataset::Triple;
use subsume_core::trainer::{
    evaluate_link_prediction, generate_negative_samples, NegativeSamplingStrategy, TrainingConfig,
};
use subsume_core::training::diagnostics::{LossComponents, TrainingStats};
use subsume_core::Box as CoreBox;
use subsume_ndarray::NdarrayBox;
use std::collections::{HashMap, HashSet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Full Training Loop with Dataset Integration");
    println!("==========================================\n");

    // In production, you would load a real dataset:
    // let dataset = load_dataset("data/wn18rr")?;
    
    // For this example, create synthetic data
    println!("Creating synthetic knowledge graph...");
    let train_triples = vec![
        Triple { head: "animal".to_string(), relation: "is_a".to_string(), tail: "mammal".to_string() },
        Triple { head: "mammal".to_string(), relation: "is_a".to_string(), tail: "dog".to_string() },
        Triple { head: "mammal".to_string(), relation: "is_a".to_string(), tail: "cat".to_string() },
        Triple { head: "animal".to_string(), relation: "is_a".to_string(), tail: "bird".to_string() },
    ];
    
    let test_triples = vec![
        Triple { head: "bird".to_string(), relation: "is_a".to_string(), tail: "sparrow".to_string() },
    ];

    // Collect all entities
    let mut entities = HashSet::new();
    for triple in train_triples.iter().chain(test_triples.iter()) {
        entities.insert(triple.head.clone());
        entities.insert(triple.tail.clone());
    }

    println!("Training on {} triples", train_triples.len());
    println!("Testing on {} triples", test_triples.len());
    println!("Total entities: {}\n", entities.len());

    // Initialize box embeddings randomly
    println!("Initializing box embeddings...");
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    for (idx, entity) in entities.iter().enumerate() {
        // Simple initialization: boxes at different positions
        let offset = idx as f32 * 2.0;
        let box_ = NdarrayBox::new(
            Array1::from(vec![offset, offset]),
            Array1::from(vec![offset + 1.5, offset + 1.5]),
            1.0,
        )?;
        entity_boxes.insert(entity.clone(), box_);
    }

    // Training configuration
    let config = TrainingConfig {
        learning_rate: 0.01,
        epochs: 5,
        batch_size: 2,
        negative_samples: 1,
        negative_strategy: NegativeSamplingStrategy::CorruptTail,
        regularization_weight: 0.01,
        temperature: 1.0,
    };

    // Training diagnostics
    let mut training_stats = TrainingStats::new(10);
    let mut loss_components = LossComponents::new(0.0, 0.0, 0.0);

    println!("Training for {} epochs...\n", config.epochs);

    // Simplified training loop (no actual optimizer, just demonstrates structure)
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for batch in train_triples.chunks(config.batch_size) {
            // Generate negative samples
            let mut all_samples = Vec::new();
            for triple in batch {
                all_samples.push(triple.clone());
                let negatives = generate_negative_samples(
                    triple,
                    &entities,
                    &config.negative_strategy,
                    config.negative_samples,
                );
                all_samples.extend(negatives);
            }

            // Compute loss (simplified - in real training, would use optimizer)
            let mut batch_loss = 0.0;
            for triple in batch {
                if let (Some(head_box), Some(tail_box)) = (
                    entity_boxes.get(&triple.head),
                    entity_boxes.get(&triple.tail),
                ) {
                    // Positive sample: maximize containment
                    let pos_score = head_box.containment_prob(tail_box, config.temperature)?;
                    batch_loss += 1.0 - pos_score; // Simple loss: 1 - containment_prob
                }
            }

            epoch_loss += batch_loss;
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count as f32;
        
        // Record training stats
        let avg_volume = entity_boxes
            .values()
            .map(|b| b.volume(config.temperature).unwrap_or(0.0))
            .sum::<f32>()
            / entity_boxes.len() as f32;
        
        training_stats.record(avg_loss, avg_volume, 0.1); // Simplified gradient norm
        
        loss_components = LossComponents::new(avg_loss, 0.0, config.regularization_weight * avg_volume);

        println!("Epoch {}: Loss = {:.4}, Avg Volume = {:.4}", epoch + 1, avg_loss, avg_volume);
    }

    println!("\nTraining complete!\n");

    // Evaluate on test set
    println!("Evaluating on test set...");
    let results = evaluate_link_prediction::<NdarrayBox>(
        &test_triples,
        &entity_boxes,
        None,
    )?;

    println!("\nTest Results:");
    println!("  MRR:      {:.4}", results.mrr);
    println!("  Hits@1:   {:.4}", results.hits_at_1);
    println!("  Hits@3:   {:.4}", results.hits_at_3);
    println!("  Hits@10:  {:.4}", results.hits_at_10);
    println!("  Mean Rank: {:.2}", results.mean_rank);

    // Training diagnostics summary
    println!("\nTraining Diagnostics:");
    if training_stats.is_converged(0.01, 3) {
        println!("  ✓ Training converged");
    } else {
        println!("  ⏳ Training not yet converged");
    }

    if loss_components.is_imbalanced() {
        println!("  ⚠ Loss components are imbalanced");
    } else {
        println!("  ✓ Loss components are balanced");
    }

    println!("\nTo use with real datasets:");
    println!("  1. Download WN18RR, FB15k-237, or YAGO3-10");
    println!("  2. Extract to directory with train.txt, valid.txt, test.txt");
    println!("  3. Replace synthetic data with: let dataset = load_dataset(\"path\")?;");
    println!("  4. Use dataset.train, dataset.valid, dataset.test");

    Ok(())
}

