//! Training box embeddings on a 20+ entity taxonomy.
//!
//! This example trains axis-aligned box embeddings to represent hierarchical
//! relationships through geometric containment. A box A containing box B
//! means "B is-a A" (e.g., dog is-a mammal).
//!
//! Taxonomy (3 levels, 25 entities):
//!   entity
//!     animal
//!       mammal: dog, cat, whale, bat
//!       bird: eagle, sparrow, penguin
//!       fish: salmon, tuna
//!     plant
//!       tree: oak, pine
//!       flower: rose, tulip
//!     vehicle
//!       car, truck, bicycle
//!
//! The training uses direct coordinate updates: for each (head, tail) pair where
//! head should contain tail, push head_min below tail_min and head_max above
//! tail_max. This is a simplified approach; production systems would use
//! backpropagation through the containment probability.
//!
//! Reference: Vilnis et al. (2018), "Probabilistic Embedding of Knowledge
//! Graphs with Box Lattice Measures"
//!
//! Run: cargo run -p subsume --example box_training

use ndarray::Array1;
use std::collections::HashMap;
use subsume::ndarray_backend::NdarrayBox;
use subsume::Box as BoxTrait;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Box Embedding Training (25 entities, direct coordinate updates) ===\n");

    // --- Define taxonomy as (head, tail) containment pairs ---
    // head should contain tail (head is more general).
    let containment_pairs: Vec<(&str, &str)> = vec![
        // Level 0 -> 1
        ("entity", "animal"),
        ("entity", "plant"),
        ("entity", "vehicle"),
        // Level 1 -> 2
        ("animal", "mammal"),
        ("animal", "bird"),
        ("animal", "fish"),
        ("plant", "tree"),
        ("plant", "flower"),
        // Level 2 -> 3
        ("mammal", "dog"),
        ("mammal", "cat"),
        ("mammal", "whale"),
        ("mammal", "bat"),
        ("bird", "eagle"),
        ("bird", "sparrow"),
        ("bird", "penguin"),
        ("fish", "salmon"),
        ("fish", "tuna"),
        ("tree", "oak"),
        ("tree", "pine"),
        ("flower", "rose"),
        ("flower", "tulip"),
        ("vehicle", "car"),
        ("vehicle", "truck"),
        ("vehicle", "bicycle"),
    ];

    // Collect all entity names
    let mut entity_set = std::collections::HashSet::new();
    for (h, t) in &containment_pairs {
        entity_set.insert(*h);
        entity_set.insert(*t);
    }
    let entity_names: Vec<&str> = {
        let mut v: Vec<&str> = entity_set.iter().copied().collect();
        v.sort();
        v
    };
    let n_entities = entity_names.len();

    println!("Entities: {}", n_entities);
    println!("Containment pairs: {}\n", containment_pairs.len());

    // --- Initialize box embeddings ---
    //
    // Start each entity with a box of width 1.0 centered at a unique offset.
    // Training will expand parent boxes and shrink/shift child boxes.
    let dim = 8;
    let mut boxes: HashMap<&str, (Array1<f32>, Array1<f32>)> = HashMap::new();
    for (idx, &name) in entity_names.iter().enumerate() {
        let center = idx as f32 * 0.3;
        let half = 0.5;
        let min_arr = Array1::from_vec(vec![center - half; dim]);
        let max_arr = Array1::from_vec(vec![center + half; dim]);
        boxes.insert(name, (min_arr, max_arr));
    }

    let lr = 0.05;
    let epochs = 200;

    // --- Training loop ---
    //
    // For each containment pair (head contains tail), compute per-dimension
    // violations and push coordinates to fix them:
    //   - If head_min[d] > tail_min[d], decrease head_min[d]
    //   - If head_max[d] < tail_max[d], increase head_max[d]
    println!("Training for {} epochs (dim={}, lr={})...\n", epochs, dim, lr);

    for epoch in 0..epochs {
        let mut total_violation = 0.0f32;

        for &(head, tail) in &containment_pairs {
            let (tail_min, tail_max) = boxes[tail].clone();
            let (head_min, head_max) = boxes.get_mut(head).unwrap();

            for d in 0..dim {
                // head_min should be <= tail_min (with margin)
                let margin = 0.05;
                if head_min[d] > tail_min[d] - margin {
                    let violation = head_min[d] - (tail_min[d] - margin);
                    head_min[d] -= lr * violation;
                    total_violation += violation.abs();
                }
                // head_max should be >= tail_max (with margin)
                if head_max[d] < tail_max[d] + margin {
                    let violation = (tail_max[d] + margin) - head_max[d];
                    head_max[d] += lr * violation;
                    total_violation += violation.abs();
                }
            }
        }

        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!(
                "  Epoch {:>4}: total_violation = {:.4}",
                epoch, total_violation
            );
        }
    }

    // Build NdarrayBox instances for evaluation
    let entity_boxes: HashMap<&str, NdarrayBox> = boxes
        .iter()
        .map(|(&name, (min, max))| {
            let b = NdarrayBox::new(min.clone(), max.clone(), 1.0)
                .expect("box construction should succeed after training");
            (name, b)
        })
        .collect();

    // --- Evaluate learned boxes ---
    println!("\n--- Learned Box Volumes (larger = more general) ---\n");

    let mut vol_pairs: Vec<(&str, f32)> = entity_boxes
        .iter()
        .map(|(&name, b)| (name, b.volume(1.0).unwrap_or(0.0)))
        .collect();
    vol_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (name, vol) in &vol_pairs {
        println!("  {:>12}: volume = {:.4}", name, vol);
    }

    // --- Containment checks ---
    println!("\n--- Containment Checks ---\n");

    let checks: Vec<(&str, &str, &str, bool)> = vec![
        // Positives (should have high containment probability)
        ("entity > animal", "entity", "animal", true),
        ("entity > vehicle", "entity", "vehicle", true),
        ("animal > mammal", "animal", "mammal", true),
        ("animal > bird", "animal", "bird", true),
        ("mammal > dog", "mammal", "dog", true),
        ("mammal > cat", "mammal", "cat", true),
        ("bird > eagle", "bird", "eagle", true),
        ("fish > salmon", "fish", "salmon", true),
        ("plant > tree", "plant", "tree", true),
        ("tree > oak", "tree", "oak", true),
        ("flower > rose", "flower", "rose", true),
        ("vehicle > car", "vehicle", "car", true),
        // Negatives (should have low containment probability)
        ("dog > animal (reverse)", "dog", "animal", false),
        ("cat > dog (sibling)", "cat", "dog", false),
        ("animal > vehicle (cross)", "animal", "vehicle", false),
    ];

    let mut correct = 0;
    let total = checks.len();
    for (label, head, tail, expect_high) in &checks {
        let hb = &entity_boxes[head];
        let tb = &entity_boxes[tail];
        let p = hb.containment_prob(tb, 1.0)?;
        let ok = if *expect_high { p > 0.5 } else { p < 0.5 };
        let status = if ok { "OK" } else { "FAIL" };
        println!("  [{:>4}] {:<30} P = {:.3}", status, label, p);
        if ok {
            correct += 1;
        }
    }

    println!(
        "\nHierarchy accuracy: {}/{} ({:.0}%)",
        correct, total, 100.0 * correct as f32 / total as f32
    );

    println!("\nNotes:");
    println!("  - This uses direct coordinate updates, not backpropagation");
    println!("  - Negative checks (reverse, sibling, cross-branch) may not separate well");
    println!("    without explicit negative sampling and margin-based loss");
    println!("  - Volume ordering (general > specific) emerges from containment constraints");

    // See scripts/plot_box_concept.py for a visualization of box containment geometry.

    Ok(())
}
