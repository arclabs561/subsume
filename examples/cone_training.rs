//! Training cone embeddings on a taxonomy with 18 entities and 3+ levels.
//!
//! This example shows how to learn cone representations that capture
//! hierarchical relationships. Cones model subsumption through per-dimension
//! angular containment: a wider cone "subsumes" a narrower cone when the
//! narrower cone's axis falls within the wider cone's sector in each dimension.
//!
//! Taxonomy (4 levels, 18 entities):
//!   entity
//!     animal
//!       mammal
//!         dog, cat, whale
//!       bird
//!         eagle, sparrow
//!       fish
//!         salmon, tuna
//!     vehicle
//!       land_vehicle
//!         car, truck
//!       aircraft
//!         helicopter
//!
//! Key properties of the ConE model:
//! - Each dimension is an independent 2D angular sector (axis + aperture).
//! - Cones support negation (complement of a cone is a cone).
//! - Containment uses per-dimension |sin| distance (ConE scoring).
//! - Reparameterization: axes through tanh, apertures through tanh.
//!
//! Reference: Zhang & Wang (2021), "ConE: Cone Embeddings for Multi-Hop
//! Reasoning over Knowledge Graphs" (NeurIPS 2021).
//!
//! Run: cargo run -p subsume --example cone_training

use std::collections::HashMap;
use subsume::{ConeEmbeddingTrainer, TrainingConfig};

fn main() {
    println!("=== Cone Embeddings (ConE): Training on a Taxonomy (18 entities, 4 levels) ===\n");

    // --- Define the taxonomy ---
    let entity_names: HashMap<usize, &str> = [
        (0, "entity"),
        (1, "animal"),
        (2, "vehicle"),
        (3, "mammal"),
        (4, "bird"),
        (5, "fish"),
        (6, "land_vehicle"),
        (7, "aircraft"),
        (8, "dog"),
        (9, "cat"),
        (10, "whale"),
        (11, "eagle"),
        (12, "sparrow"),
        (13, "salmon"),
        (14, "tuna"),
        (15, "car"),
        (16, "truck"),
        (17, "helicopter"),
    ]
    .into_iter()
    .collect();

    // Positive pairs (head subsumes tail):
    let positive_pairs: Vec<(usize, usize)> = vec![
        (0, 1),  // entity > animal
        (0, 2),  // entity > vehicle
        (1, 3),  // animal > mammal
        (1, 4),  // animal > bird
        (1, 5),  // animal > fish
        (2, 6),  // vehicle > land_vehicle
        (2, 7),  // vehicle > aircraft
        (3, 8),  // mammal > dog
        (3, 9),  // mammal > cat
        (3, 10), // mammal > whale
        (4, 11), // bird > eagle
        (4, 12), // bird > sparrow
        (5, 13), // fish > salmon
        (5, 14), // fish > tuna
        (6, 15), // land_vehicle > car
        (6, 16), // land_vehicle > truck
        (7, 17), // aircraft > helicopter
    ];

    // Negative pairs: siblings should not subsume each other,
    // and leaf nodes should not subsume ancestors.
    let negative_pairs: Vec<(usize, usize)> = vec![
        (8, 0),   // dog does NOT subsume entity
        (15, 0),  // car does NOT subsume entity
        (11, 1),  // eagle does NOT subsume animal
        (13, 1),  // salmon does NOT subsume animal
        (1, 2),   // animal does NOT subsume vehicle
        (2, 1),   // vehicle does NOT subsume animal
        (3, 6),   // mammal does NOT subsume land_vehicle
        (4, 7),   // bird does NOT subsume aircraft
        (8, 9),   // dog does NOT subsume cat
        (9, 8),   // cat does NOT subsume dog
        (8, 10),  // dog does NOT subsume whale
        (11, 12), // eagle does NOT subsume sparrow
        (13, 14), // salmon does NOT subsume tuna
        (15, 16), // car does NOT subsume truck
        (3, 1),   // mammal does NOT subsume animal
        (5, 1),   // fish does NOT subsume animal
        (6, 2),   // land_vehicle does NOT subsume vehicle
    ];

    let n_entities = entity_names.len();
    let dim = 16;
    let epochs = 300;

    // --- Initialize trainer ---
    let config = TrainingConfig {
        learning_rate: 0.01,
        temperature: 1.0,
        margin: 0.3,
        regularization: 0.001,
        negative_weight: 2.0,
        ..Default::default()
    };

    let mut trainer = ConeEmbeddingTrainer::new(config.clone(), dim, None);

    // Pre-register all entities
    for &id in entity_names.keys() {
        trainer.ensure_entity(id);
    }

    // --- Training loop ---
    println!(
        "Training for {} epochs (dim={}, {} entities, {} pos + {} neg pairs)...\n",
        epochs,
        dim,
        n_entities,
        positive_pairs.len(),
        negative_pairs.len()
    );

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for &(head, tail) in &positive_pairs {
            let loss = trainer.train_step(head, tail, true);
            epoch_loss += loss;
        }

        for &(head, tail) in &negative_pairs {
            let loss = trainer.train_step(head, tail, false);
            epoch_loss += loss;
        }

        let n_pairs = (positive_pairs.len() + negative_pairs.len()) as f32;
        let avg_loss = epoch_loss / n_pairs;

        if epoch % 75 == 0 || epoch == epochs - 1 {
            println!("  Epoch {:>4}: avg_loss = {:.4}", epoch, avg_loss);
        }
    }

    // --- Evaluate learned cones ---
    println!("\n--- Learned Cone Properties ---\n");
    println!(
        "{:>14} {:>12} {:>12}",
        "entity", "mean_aper", "mean_deg"
    );
    println!("{}", "-".repeat(40));

    let mut entity_ids: Vec<usize> = entity_names.keys().copied().collect();
    entity_ids.sort();

    for &id in &entity_ids {
        let cone = &trainer.cones[&id];
        let mean_aper = cone.mean_aperture();
        let degrees = mean_aper.to_degrees();
        println!(
            "{:>14} {:>12.4} {:>12.1}",
            entity_names[&id], mean_aper, degrees
        );
    }

    // --- Containment checks (selected pairs) ---
    println!("\n--- Selected Containment Distances (lower = better containment) ---\n");

    let cen = 0.02;
    let selected_checks: Vec<(&str, usize, usize, bool)> = vec![
        // True positives (should have low distance)
        ("entity > animal", 0, 1, true),
        ("entity > vehicle", 0, 2, true),
        ("animal > mammal", 1, 3, true),
        ("animal > bird", 1, 4, true),
        ("mammal > dog", 3, 8, true),
        ("mammal > cat", 3, 9, true),
        ("bird > eagle", 4, 11, true),
        ("fish > salmon", 5, 13, true),
        ("land_vehicle > car", 6, 15, true),
        ("aircraft > helicopter", 7, 17, true),
        // True negatives (should have high distance)
        ("dog > entity (reverse)", 8, 0, false),
        ("dog > cat (sibling)", 8, 9, false),
        ("animal > vehicle (cross)", 1, 2, false),
        ("mammal > land_vehicle (cross)", 3, 6, false),
    ];

    // Use a threshold based on median distance to distinguish positive/negative.
    let mut pos_dists = Vec::new();
    let mut neg_dists = Vec::new();
    for (label, head, tail, expect_low) in &selected_checks {
        let d = trainer.cones[head].cone_distance(&trainer.cones[tail], cen);
        let status = if *expect_low { "POS" } else { "NEG" };
        println!("  [{:>3}] {:<30} dist = {:.4}", status, label, d);
        if *expect_low {
            pos_dists.push(d);
        } else {
            neg_dists.push(d);
        }
    }

    let avg_pos: f32 = pos_dists.iter().sum::<f32>() / pos_dists.len() as f32;
    let avg_neg: f32 = neg_dists.iter().sum::<f32>() / neg_dists.len() as f32;

    println!(
        "\nAvg positive distance: {:.4}, Avg negative distance: {:.4}",
        avg_pos, avg_neg
    );
    if avg_pos < avg_neg {
        println!("Positive pairs have lower distance than negatives (as expected).");
    } else {
        println!("Warning: separation not achieved. Consider more epochs or tuning.");
    }

    println!("\nKey takeaways:");
    println!("  - More general concepts (entity, animal) get wider mean apertures");
    println!("  - More specific concepts (dog, car) get narrower mean apertures");
    println!("  - Containment is directional: animal > mammal, but NOT mammal > animal");
    println!("  - Cross-branch distance (animal > vehicle) stays high");
    println!("  - Unlike boxes, cones support negation: complement of a cone is a cone");
}
