//! Training cone embeddings on a taxonomy with 18 entities and 3+ levels.
//!
//! This example shows how to learn cone representations that capture
//! hierarchical relationships. Cones model subsumption through angular
//! containment: a wider cone "subsumes" a narrower cone pointing in a
//! similar direction.
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
//! Key differences from box training:
//! - Cones support negation (the complement of a cone is a cone).
//! - Containment is based on angular distance + aperture, not volume ratios.
//! - Reparameterization: axis is normalized, aperture goes through sigmoid.
//!
//! Reference: Zhang & Wang (2021), "ConE: Cone Embeddings for Multi-Hop
//! Reasoning over Knowledge Graphs" (NeurIPS 2021).
//!
//! Run: cargo run -p subsume --example cone_training

use std::collections::HashMap;
use subsume::{ConeEmbeddingTrainer, TrainingConfig};

fn main() {
    println!("=== Cone Embeddings: Training on a Taxonomy (18 entities, 4 levels) ===\n");

    // --- Define the taxonomy ---
    //
    // 18 entities across 4 hierarchy levels.
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
        // Level 0 -> 1
        (0, 1),  // entity > animal
        (0, 2),  // entity > vehicle
        // Level 1 -> 2
        (1, 3),  // animal > mammal
        (1, 4),  // animal > bird
        (1, 5),  // animal > fish
        (2, 6),  // vehicle > land_vehicle
        (2, 7),  // vehicle > aircraft
        // Level 2 -> 3
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
        // Leaves do not subsume roots
        (8, 0),  // dog does NOT subsume entity
        (15, 0), // car does NOT subsume entity
        (11, 1), // eagle does NOT subsume animal
        (13, 1), // salmon does NOT subsume animal
        // Cross-branch: animal vs vehicle
        (1, 2),  // animal does NOT subsume vehicle
        (2, 1),  // vehicle does NOT subsume animal
        (3, 6),  // mammal does NOT subsume land_vehicle
        (4, 7),  // bird does NOT subsume aircraft
        // Siblings do not subsume each other
        (8, 9),  // dog does NOT subsume cat
        (9, 8),  // cat does NOT subsume dog
        (8, 10), // dog does NOT subsume whale
        (11, 12),// eagle does NOT subsume sparrow
        (13, 14),// salmon does NOT subsume tuna
        (15, 16),// car does NOT subsume truck
        // Reverse containment
        (3, 1),  // mammal does NOT subsume animal
        (5, 1),  // fish does NOT subsume animal
        (6, 2),  // land_vehicle does NOT subsume vehicle
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
        epochs, dim, n_entities, positive_pairs.len(), negative_pairs.len()
    );

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        // Positive pairs
        for &(head, tail) in &positive_pairs {
            let loss = trainer.train_step(head, tail, true);
            epoch_loss += loss;
        }

        // Negative pairs
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
        "entity", "aperture", "aperture_deg"
    );
    println!("{}", "-".repeat(40));

    let mut entity_ids: Vec<usize> = entity_names.keys().copied().collect();
    entity_ids.sort();

    for &id in &entity_ids {
        let cone = &trainer.cones[&id];
        let aperture = cone.aperture();
        let degrees = aperture.to_degrees();
        println!(
            "{:>14} {:>12.4} {:>12.1}",
            entity_names[&id], aperture, degrees
        );
    }

    // --- Containment matrix (selected pairs) ---
    println!("\n--- Selected Containment Probabilities ---\n");

    let temp = 1.0;
    let selected_checks: Vec<(&str, usize, usize, bool)> = vec![
        // True positives (should be high)
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
        // True negatives (should be low)
        ("dog > entity (reverse)", 8, 0, false),
        ("dog > cat (sibling)", 8, 9, false),
        ("animal > vehicle (cross)", 1, 2, false),
        ("mammal > land_vehicle (cross)", 3, 6, false),
    ];

    let mut correct = 0;
    let mut total = 0;
    for (label, head, tail, expect_high) in &selected_checks {
        let p = trainer.cones[head].containment_prob(&trainer.cones[tail], temp);
        let ok = if *expect_high { p > 0.5 } else { p < 0.5 };
        let status = if ok { "OK" } else { "FAIL" };
        println!("  [{:>4}] {:<30} P = {:.3}", status, label, p);
        if ok {
            correct += 1;
        }
        total += 1;
    }

    println!(
        "\nHierarchy accuracy: {}/{} ({:.0}%)",
        correct,
        total,
        100.0 * correct as f32 / total as f32
    );

    println!("\nKey takeaways:");
    println!("  - More general concepts (entity, animal) get wider apertures");
    println!("  - More specific concepts (dog, car) get narrower apertures");
    println!("  - Containment is directional: animal > mammal, but NOT mammal > animal");
    println!("  - Cross-branch containment (animal > vehicle) stays low");
    println!("  - Unlike boxes, cones support negation: complement of a cone is a cone");
}
