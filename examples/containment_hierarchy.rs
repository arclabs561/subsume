//! Containment hierarchy: modeling taxonomic is-a relationships with box embeddings.
//!
//! Box embeddings represent concepts as axis-aligned hyperrectangles. Containment
//! (B inside A) models entailment: "B is-a A". Larger boxes represent more general
//! concepts; smaller boxes represent more specific ones.
//!
//! This example builds a small taxonomy (animal > mammal > {dog, cat}) and shows:
//! 1. Containment probabilities reflect the hierarchy
//! 2. Volume correlates with concept generality
//! 3. Overlap detects related-but-distinct concepts
//! 4. Temperature controls the sharpness of probabilistic scores
//!
//! Reference: Vilnis et al. (2018), "Probabilistic Embedding of Knowledge Graphs
//! with Box Lattice Measures"
//!
//! Run: cargo run -p subsume --example containment_hierarchy

use ndarray::array;
use subsume::ndarray_backend::NdarrayBox;
use subsume::Box as BoxTrait;

fn main() -> Result<(), subsume::BoxError> {
    println!("=== Box Embeddings: Containment Hierarchy ===\n");

    // Build a small taxonomy as nested boxes in 3D.
    // More general concepts get larger boxes.
    let animal = NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0)?;
    let mammal = NdarrayBox::new(array![0.05, 0.05, 0.05], array![0.95, 0.95, 0.95], 1.0)?;
    let dog = NdarrayBox::new(array![0.1, 0.1, 0.1], array![0.4, 0.4, 0.4], 1.0)?;
    let cat = NdarrayBox::new(array![0.5, 0.5, 0.1], array![0.9, 0.9, 0.4], 1.0)?;
    let fish = NdarrayBox::new(array![0.1, 0.1, 0.6], array![0.5, 0.5, 0.95], 1.0)?;

    let temp = 1.0;

    // --- Part 1: Volumes reflect generality ---
    println!("--- Volumes (larger = more general) ---\n");
    let entities = [
        ("animal", &animal),
        ("mammal", &mammal),
        ("dog", &dog),
        ("cat", &cat),
        ("fish", &fish),
    ];
    for (name, b) in &entities {
        println!("  {:>8}: volume = {:.4}", name, b.volume(temp)?);
    }

    // --- Part 2: Containment probabilities ---
    println!("\n--- Containment P(row inside col) ---\n");
    print!("{:>10}", "");
    for (name, _) in &entities {
        print!("{:>10}", name);
    }
    println!();
    for (rname, rb) in &entities {
        print!("{:>10}", rname);
        for (_cname, cb) in &entities {
            let p = cb.containment_prob(rb, temp)?;
            print!("{:>10.3}", p);
        }
        println!();
    }

    // --- Part 3: Overlap probabilities ---
    println!("\n--- Overlap P(row intersects col) ---\n");
    print!("{:>10}", "");
    for (name, _) in &entities {
        print!("{:>10}", name);
    }
    println!();
    for (rname, rb) in &entities {
        print!("{:>10}", rname);
        for (_cname, cb) in &entities {
            let p = rb.overlap_prob(cb, temp)?;
            print!("{:>10.3}", p);
        }
        println!();
    }

    // --- Part 4: Temperature effects ---
    println!("\n--- Temperature effect on P(dog inside animal) ---\n");
    println!("{:>12} {:>12}", "temperature", "P(dog|animal)");
    for &t in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let p = animal.containment_prob(&dog, t)?;
        println!("{:>12.1} {:>12.4}", t, p);
    }

    println!("\nKey observations:");
    println!("  - dog is inside mammal, which is inside animal (transitive containment)");
    println!("  - dog and cat overlap partially but neither contains the other");
    println!("  - fish overlaps with animal but not with mammal (different sub-hierarchy)");
    println!("  - lower temperature makes containment probabilities sharper (closer to 0/1)");

    // See scripts/plot_box_concept.py for a visualization of box containment, overlap,
    // and Gumbel soft boundaries.

    Ok(())
}
