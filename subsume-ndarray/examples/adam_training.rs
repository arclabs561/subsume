//! Example: Training box embeddings with Adam optimizer.
//!
//! This example demonstrates a complete training loop using the Adam optimizer
//! as specified in the box embedding papers (learning rate 1e-3 to 5e-4).

use ndarray::Array1;
use subsume_core::dataset::Triple;
use subsume_core::trainer::{
    evaluate_link_prediction, generate_negative_samples, NegativeSamplingStrategy, TrainingConfig,
};
use subsume_core::training::diagnostics::TrainingStats;
use subsume_core::Box as CoreBox;
use subsume_ndarray::{Adam, NdarrayBox};
use std::collections::{HashMap, HashSet};

fn main() -> Result<(), std::boxed::Box<dyn std::error::Error>> {
    println!("Adam Optimizer Training Example");
    println!("==============================\n");

    // Create synthetic knowledge graph
    let train_triples = vec![
        Triple {
            head: "animal".to_string(),
            relation: "is_a".to_string(),
            tail: "mammal".to_string(),
        },
        Triple {
            head: "mammal".to_string(),
            relation: "is_a".to_string(),
            tail: "dog".to_string(),
        },
        Triple {
            head: "mammal".to_string(),
            relation: "is_a".to_string(),
            tail: "cat".to_string(),
        },
    ];

    let test_triples = vec![Triple {
        head: "animal".to_string(),
        relation: "is_a".to_string(),
        tail: "bird".to_string(),
    }];

    // Collect entities
    let mut entities = HashSet::new();
    for triple in train_triples.iter().chain(test_triples.iter()) {
        entities.insert(triple.head.clone());
        entities.insert(triple.tail.clone());
    }

    println!("Training on {} triples", train_triples.len());
    println!("Testing on {} triples", test_triples.len());
    println!("Total entities: {}\n", entities.len());

    // Initialize box embeddings
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    for (idx, entity) in entities.iter().enumerate() {
        let offset = idx as f32 * 2.0;
        let box_ = NdarrayBox::new(
            Array1::from(vec![offset, offset]),
            Array1::from(vec![offset + 1.5, offset + 1.5]),
            1.0,
        )?;
        entity_boxes.insert(entity.clone(), box_);
    }

    // Training configuration (paper defaults)
    let config = TrainingConfig {
        learning_rate: 1e-3, // Paper default
        epochs: 20,
        batch_size: 2,
        negative_samples: 1,
        negative_strategy: NegativeSamplingStrategy::CorruptTail,
        regularization_weight: 1e-5, // Paper default
        temperature: 1.0,
        weight_decay: 1e-5,
        margin: 1.0,
        early_stopping_patience: Some(5),
    };

    // Create Adam optimizer (paper standard)
    let mut optimizer = Adam::new(config.learning_rate);

    // Training diagnostics
    let mut training_stats = TrainingStats::new(10);

    println!("Training with Adam optimizer (lr={})\n", config.learning_rate);

    // Training loop
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for batch in train_triples.chunks(config.batch_size) {
            for triple in batch {
                if let (Some(head_box), Some(tail_box)) = (
                    entity_boxes.get(&triple.head),
                    entity_boxes.get(&triple.tail),
                ) {
                    // Positive sample: maximize containment
                    let pos_score = head_box.containment_prob(tail_box, config.temperature)?;

                    // Generate negative sample
                    let negatives = generate_negative_samples(
                        triple,
                        &entities,
                        &config.negative_strategy,
                        config.negative_samples,
                    );

                    let mut batch_loss = 1.0 - pos_score;

                    // Negative samples: minimize containment
                    for neg in negatives.iter().take(config.negative_samples) {
                        if let Some(neg_tail_box) = entity_boxes.get(&neg.tail) {
                            let neg_score =
                                head_box.containment_prob(neg_tail_box, config.temperature)?;
                            // Margin-based loss: max(0, margin - pos_score + neg_score)
                            batch_loss += (config.margin - pos_score + neg_score).max(0.0);
                        }
                    }

                    // Simplified gradient computation: approximate gradient as loss * direction
                    // In real implementation, would use automatic differentiation
                    // Clone tail values before mutable borrow
                    let tail_min = tail_box.min().to_owned();
                    let tail_max = tail_box.max().to_owned();
                    
                    let head_box_mut = entity_boxes.get_mut(&triple.head).unwrap();
                    let mut head_min = head_box_mut.min().to_owned();
                    let mut head_max = head_box_mut.max().to_owned();
                    let grad_min_vec: Vec<f32> = head_min.iter().zip(tail_min.iter())
                        .map(|(h, t)| batch_loss * (t - h))
                        .collect();
                    let grad_max_vec: Vec<f32> = head_max.iter().zip(tail_max.iter())
                        .map(|(h, t)| batch_loss * (t - h))
                        .collect();
                    let grad_min = Array1::from_vec(grad_min_vec);
                    let grad_max = Array1::from_vec(grad_max_vec);
                    
                    optimizer.update(
                        &format!("{}_min", triple.head),
                        &mut head_min,
                        grad_min.view(),
                    );
                    optimizer.update(
                        &format!("{}_max", triple.head),
                        &mut head_max,
                        grad_max.view(),
                    );
                    *head_box_mut = NdarrayBox::new(
                        head_min,
                        head_max,
                        config.temperature,
                    )?;

                    epoch_loss += batch_loss;
                }
            }
            batch_count += 1;
        }

        let avg_loss = epoch_loss / (batch_count as f32).max(1.0);

        // Record stats
        let avg_volume = entity_boxes
            .values()
            .map(|b| b.volume(config.temperature).unwrap_or(0.0))
            .sum::<f32>()
            / entity_boxes.len() as f32;

        training_stats.record(avg_loss, avg_volume, 0.1);

        if epoch % 5 == 0 {
            println!(
                "Epoch {}: Loss = {:.4}, Avg Volume = {:.4}",
                epoch + 1,
                avg_loss,
                avg_volume
            );
        }
    }

    println!("\nTraining complete!\n");

    // Evaluate
    let results = evaluate_link_prediction::<NdarrayBox>(&test_triples, &entity_boxes, None)?;

    println!("Test Results:");
    println!("  MRR:      {:.4}", results.mrr);
    println!("  Hits@1:   {:.4}", results.hits_at_1);
    println!("  Hits@10:  {:.4}", results.hits_at_10);

    Ok(())
}

