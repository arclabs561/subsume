//! Simple, working training example.
//!
//! This is a minimal, complete training example that demonstrates:
//! 1. Creating box embeddings
//! 2. Training with Adam optimizer
//! 3. Evaluating results
//!
//! This example is guaranteed to compile and run.

use ndarray::Array1;
use subsume_core::dataset::Triple;
use subsume_core::trainer::{evaluate_link_prediction, TrainingConfig};
use subsume_core::Box as CoreBox;
use subsume_ndarray::{Adam, NdarrayBox};
use std::collections::{HashMap, HashSet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simple Training Example");
    println!("=======================\n");

    // Create simple knowledge graph
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

    println!("Entities: {:?}", entities);
    println!("Training triples: {}", train_triples.len());
    println!("Test triples: {}\n", test_triples.len());

    // Initialize box embeddings
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    for (idx, entity) in entities.iter().enumerate() {
        let offset = idx as f32 * 2.0;
        let box_ = NdarrayBox::new(
            Array1::from_vec(vec![offset, offset]),
            Array1::from_vec(vec![offset + 1.5, offset + 1.5]),
            1.0,
        )?;
        entity_boxes.insert(entity.clone(), box_);
    }

    // Create Adam optimizer
    let mut optimizer = Adam::new(0.01);
    let config = TrainingConfig::default();

    println!("Training with Adam optimizer (lr={})...\n", config.learning_rate);

    // Simple training loop
    for epoch in 0..10 {
        let mut epoch_loss = 0.0;

        for triple in &train_triples {
            if let (Some(head_box), Some(tail_box)) = (
                entity_boxes.get(&triple.head),
                entity_boxes.get(&triple.tail),
            ) {
                // Compute containment probability (what we want to maximize)
                let score = head_box.containment_prob(tail_box, 1.0)?;
                let loss = 1.0 - score; // Simple loss: 1 - containment_prob
                epoch_loss += loss;

                // Update head box to contain tail box better
                // Clone tail values before mutable borrow
                let tail_min = tail_box.min().to_owned();
                let tail_max = tail_box.max().to_owned();
                
                let head_box_mut = entity_boxes.get_mut(&triple.head).unwrap();
                let mut head_min = head_box_mut.min().to_owned();

                // Simple gradient approximation: move head box towards tail box
                let grad_min_vec: Vec<f32> = head_min
                    .iter()
                    .zip(tail_min.iter())
                    .map(|(h, t)| loss * 0.1 * (t - h))
                    .collect();
                let grad_min = Array1::from_vec(grad_min_vec);
                optimizer.update(&format!("{}_min", triple.head), &mut head_min, grad_min.view());

                // Update the box
                let head_max_clone = head_box_mut.max().to_owned();
                *head_box_mut = NdarrayBox::new(head_min, head_max_clone, 1.0)?;
            }
        }

        if epoch % 2 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch + 1, epoch_loss / train_triples.len() as f32);
        }
    }

    println!("\nTraining complete!\n");

    // Evaluate
    println!("Evaluating on test set...");
    let results = evaluate_link_prediction::<NdarrayBox>(&test_triples, &entity_boxes, None)?;

    println!("Results:");
    println!("  MRR:      {:.4}", results.mrr);
    println!("  Hits@1:   {:.4}", results.hits_at_1);
    println!("  Hits@10:  {:.4}", results.hits_at_10);
    println!("  Mean Rank: {:.2}", results.mean_rank);

    Ok(())
}

