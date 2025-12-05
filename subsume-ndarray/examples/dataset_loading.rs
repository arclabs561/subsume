//! Example: Loading and working with knowledge graph datasets.
//!
//! This example demonstrates how to load standard KG datasets (WN18RR, FB15k-237, YAGO3-10)
//! and prepare them for training box embeddings.

use subsume_core::dataset::Triple;
use subsume_core::trainer::{evaluate_link_prediction, TrainingConfig};
use std::collections::HashMap;
use subsume_ndarray::NdarrayBox;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Dataset Loading Example");
    println!("======================\n");

    // Example: Load a dataset (replace with actual path)
    // let dataset = load_dataset("data/wn18rr")?;
    
    // For demonstration, create a small synthetic dataset
    println!("Creating synthetic knowledge graph dataset...\n");
    
    let synthetic_triples = vec![
        Triple { head: "animal".to_string(), relation: "is_a".to_string(), tail: "mammal".to_string() },
        Triple { head: "mammal".to_string(), relation: "is_a".to_string(), tail: "dog".to_string() },
        Triple { head: "mammal".to_string(), relation: "is_a".to_string(), tail: "cat".to_string() },
        Triple { head: "animal".to_string(), relation: "is_a".to_string(), tail: "bird".to_string() },
        Triple { head: "bird".to_string(), relation: "is_a".to_string(), tail: "sparrow".to_string() },
    ];

    println!("Loaded {} triples", synthetic_triples.len());
    
    // Collect entities and relations
    let mut entities = std::collections::HashSet::new();
    let mut relations = std::collections::HashSet::new();
    
    for triple in &synthetic_triples {
        entities.insert(triple.head.clone());
        entities.insert(triple.tail.clone());
        relations.insert(triple.relation.clone());
    }
    
    println!("Found {} unique entities", entities.len());
    println!("Found {} unique relations\n", relations.len());

    // Create box embeddings for entities
    println!("Creating box embeddings for entities...\n");
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();
    
    // Initialize boxes (in real training, these would be learned)
    for (idx, entity) in entities.iter().enumerate() {
        let min_val = idx as f32 * 0.5;
        let max_val = min_val + 1.0;
        let box_ = NdarrayBox::new(
            Array1::from(vec![min_val, min_val]),
            Array1::from(vec![max_val, max_val]),
            1.0,
        )?;
        entity_boxes.insert(entity.clone(), box_);
    }

    // Evaluate link prediction
    println!("Evaluating link prediction...\n");
    let results = evaluate_link_prediction::<NdarrayBox>(
        &synthetic_triples,
        &entity_boxes,
        None,
    )?;

    println!("Evaluation Results:");
    println!("  MRR:      {:.4}", results.mrr);
    println!("  Hits@1:   {:.4}", results.hits_at_1);
    println!("  Hits@3:   {:.4}", results.hits_at_3);
    println!("  Hits@10:  {:.4}", results.hits_at_10);
    println!("  Mean Rank: {:.2}", results.mean_rank);

    // Show training configuration
    println!("\nTraining Configuration:");
    let config = TrainingConfig::default();
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Epochs: {}", config.epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Negative samples: {}", config.negative_samples);

    println!("\nTo use with real datasets:");
    println!("  1. Download WN18RR, FB15k-237, or YAGO3-10");
    println!("  2. Extract to a directory with train.txt, valid.txt, test.txt");
    println!("  3. Use: let dataset = load_dataset(\"path/to/dataset\")?;");

    Ok(())
}

