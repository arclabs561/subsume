//! Real BoxE training example for knowledge graph completion.
//!
//! This example demonstrates training the BoxE model (Boratko et al. 2020)
//! with translational bumps for relation-specific transformations.
//!
//! # Dataset Setup
//!
//! Works with any standard KG dataset (WN18RR, FB15k-237, YAGO3-10).
//! Place dataset in `data/wn18rr/` or specify path as argument.
//!
//! # Mathematical Foundations
//!
//! BoxE extends box embeddings with relation-specific transformations (translational bumps).
//! The core idea remains geometric containment: entities are boxes, and relations transform
//! these boxes. The scoring function measures how well transformed boxes satisfy containment
//! relationships.
//!
//! For detailed mathematical foundations, see:
//! - [`docs/typst-output/pdf/subsumption.pdf`](../../../docs/typst-output/pdf/subsumption.pdf) - Geometric containment as logical subsumption
//! - [`docs/typst-output/pdf/local-identifiability.pdf`](../../../docs/typst-output/pdf/local-identifiability.pdf) - Why Gumbel boxes enable learning
//! - [`docs/typst-output/pdf/07-applications.pdf`](../../../docs/typst-output/pdf/07-applications.pdf) - Modern applications and extensions

use ndarray::Array1;
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;
use subsume_core::boxe::{boxe_loss, boxe_score, Bump};
use subsume_core::dataset::load_dataset;
use subsume_core::trainer::{evaluate_link_prediction_filtered, FilteredTripleIndex, TrainingConfig};
use subsume_core::Box as CoreBox;
use subsume_ndarray::{AdamW, NdarrayBox};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Real BoxE Training Example");
    println!("==========================\n");

    let dataset_path = env::args()
        .nth(1)
        .map(|p| Path::new(&p).to_path_buf())
        .unwrap_or_else(|| Path::new("data/wn18rr").to_path_buf());

    println!("Loading dataset from: {:?}", dataset_path);

    let dataset = match load_dataset(&dataset_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("\nError loading dataset: {}", e);
            eprintln!("\nPlease download a dataset and place it in data/wn18rr/");
            return Err(Box::new(e));
        }
    };

    let stats = dataset.stats();
    println!(
        "Dataset: {} entities, {} relations",
        stats.num_entities, stats.num_relations
    );
    println!(
        "Train: {}, Valid: {}, Test: {}\n",
        stats.num_train, stats.num_valid, stats.num_test
    );

    let entities: HashSet<String> = dataset.entities();
    let relations: HashSet<String> = dataset.relations();

    let filter = FilteredTripleIndex::from_triples(
        dataset
            .train
            .iter()
            .chain(dataset.valid.iter())
            .chain(dataset.test.iter()),
    );

    // Initialize entity boxes
    let embedding_dim = 50;
    let mut entity_boxes: HashMap<String, NdarrayBox> = HashMap::new();

    use rand::Rng;
    let mut rng = rand::thread_rng();

    println!("Initializing {} entity boxes...", entities.len());
    for entity in &entities {
        let center: Vec<f32> = (0..embedding_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let size: Vec<f32> = (0..embedding_dim)
            .map(|_| rng.gen_range(0.1..0.3))
            .collect();

        let min = Array1::from_iter(center.iter().zip(size.iter()).map(|(c, s)| c - s / 2.0));
        let max = Array1::from_iter(center.iter().zip(size.iter()).map(|(c, s)| c + s / 2.0));

        entity_boxes.insert(entity.clone(), NdarrayBox::new(min, max, 1.0)?);
    }

    // Initialize relation bumps (translational vectors)
    let mut relation_bumps: HashMap<String, Bump> = HashMap::new();

    println!("Initializing {} relation bumps...", relations.len());
    for relation in &relations {
        let translation: Vec<f32> = (0..embedding_dim)
            .map(|_| rng.gen_range(-0.05..0.05))
            .collect();
        relation_bumps.insert(relation.clone(), Bump::new(translation));
    }

    let config = TrainingConfig {
        learning_rate: 1e-3,
        epochs: 50,
        batch_size: 512,
        negative_samples: 1,
        negative_strategy: subsume_core::trainer::NegativeSamplingStrategy::CorruptTail,
        regularization_weight: 1e-5,
        temperature: 1.0,
        weight_decay: 1e-5,
        margin: 1.0,
        early_stopping_patience: Some(10),
        ..TrainingConfig::default()
    };

    println!("Training Configuration:");
    println!("  Model: BoxE (with translational bumps)");
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Epochs: {}", config.epochs);
    println!("  Margin: {}\n", config.margin);

    // Optimizers for entities and relations
    let mut entity_optimizer = AdamW::new(config.learning_rate, config.weight_decay);
    let mut relation_optimizer = AdamW::new(config.learning_rate, config.weight_decay);

    println!("Starting BoxE training...\n");

    // Training loop
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        use rand::seq::SliceRandom;
        let mut train_triples = dataset.train.clone();
        train_triples.shuffle(&mut rng);

        for batch in train_triples.chunks(config.batch_size) {
            for triple in batch {
                if let (Some(head_box), Some(tail_box), Some(bump)) = (
                    entity_boxes.get(&triple.head),
                    entity_boxes.get(&triple.tail),
                    relation_bumps.get(&triple.relation),
                ) {
                    // Compute BoxE score: P(tail ⊆ (head + bump))
                    let pos_score = boxe_score(
                        head_box.min().as_slice().unwrap(),
                        head_box.max().as_slice().unwrap(),
                        tail_box.min().as_slice().unwrap(),
                        tail_box.max().as_slice().unwrap(),
                        &bump.translation,
                        1.0,
                    )?;

                    // Generate negative sample
                    let mut neg_tail = triple.tail.clone();
                    while neg_tail == triple.tail {
                        let entity_vec: Vec<&String> = entities.iter().collect();
                        neg_tail = entity_vec[rng.gen_range(0..entity_vec.len())].clone();
                    }

                    if let Some(neg_tail_box) = entity_boxes.get(&neg_tail) {
                        let neg_score = boxe_score(
                            head_box.min().as_slice().unwrap(),
                            head_box.max().as_slice().unwrap(),
                            neg_tail_box.min().as_slice().unwrap(),
                            neg_tail_box.max().as_slice().unwrap(),
                            &bump.translation,
                            1.0,
                        )?;

                        // BoxE margin-based loss
                        let loss = boxe_loss(pos_score, neg_score, config.margin);
                        epoch_loss += loss;

                        // Update head box (simplified gradient)
                        // Clone tail values before mutable borrow
                        let tail_min = tail_box.min().to_owned();
                        let tail_max = tail_box.max().to_owned();

                        let head_box_mut = entity_boxes.get_mut(&triple.head).unwrap();
                        let mut head_min = head_box_mut.min().to_owned();

                        let grad_min_vec: Vec<f32> = head_min
                            .iter()
                            .zip(tail_min.iter())
                            .map(|(h, t)| loss * 0.01 * (t - h))
                            .collect();
                        let grad_min = Array1::from_vec(grad_min_vec);
                        entity_optimizer.update(
                            &format!("{}_min", triple.head),
                            &mut head_min,
                            grad_min.view(),
                        );

                        let mut head_max = head_box_mut.max().to_owned();
                        let grad_max_vec: Vec<f32> = head_max
                            .iter()
                            .zip(tail_max.iter())
                            .map(|(h, t)| loss * 0.01 * (t - h))
                            .collect();
                        let grad_max = Array1::from_vec(grad_max_vec);
                        entity_optimizer.update(
                            &format!("{}_max", triple.head),
                            &mut head_max,
                            grad_max.view(),
                        );

                        *head_box_mut = NdarrayBox::new(head_min, head_max, 1.0)?;

                        // Update relation bump (simplified)
                        let bump_mut = relation_bumps.get_mut(&triple.relation).unwrap();
                        let mut bump_translation =
                            Array1::from_iter(bump_mut.translation.iter().cloned());
                        let bump_grad_vec: Vec<f32> =
                            bump_translation.iter().map(|_| loss * 0.005).collect();
                        let bump_grad = Array1::from_vec(bump_grad_vec);
                        relation_optimizer.update(
                            &format!("{}_bump", triple.relation),
                            &mut bump_translation,
                            bump_grad.view(),
                        );
                        bump_mut.translation = bump_translation.to_vec();
                    }
                }
            }
            batch_count += 1;
        }

        let avg_loss = epoch_loss / (batch_count as f32).max(1.0);

        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            // Evaluate on validation set
            let valid_subset: Vec<_> = dataset.valid.iter().take(500).cloned().collect();
            let valid_results = evaluate_link_prediction_filtered::<NdarrayBox>(
                &valid_subset,
                &entity_boxes,
                None,
                &filter,
            )?;

            println!(
                "Epoch {}: Loss = {:.4}, Valid MRR = {:.4}, Hits@10 = {:.4}",
                epoch + 1,
                avg_loss,
                valid_results.mrr,
                valid_results.hits_at_10
            );
        }
    }

    println!("\nTraining complete!\n");

    // Final evaluation
    let test_subset: Vec<_> = dataset.test.iter().take(1000).cloned().collect();
    let test_results =
        evaluate_link_prediction_filtered::<NdarrayBox>(&test_subset, &entity_boxes, None, &filter)?;

    println!("=== Final Test Results ===");
    println!("MRR:      {:.4}", test_results.mrr);
    println!("Hits@1:   {:.4}", test_results.hits_at_1);
    println!("Hits@10:  {:.4}", test_results.hits_at_10);
    println!("Mean Rank: {:.2}", test_results.mean_rank);

    Ok(())
}
