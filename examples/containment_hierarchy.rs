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
//! 4. Volume ratios quantify how much more general one concept is than another
//!
//! Reference: Vilnis et al. (2018), "Probabilistic Embedding of Knowledge Graphs
//! with Box Lattice Measures"
//!
//! Run: cargo run -p subsume --example containment_hierarchy
//!
//! Related examples:
//! - `octagon_demo`: adds diagonal constraints to boxes for tighter geometry
//! - `query2box`: compositional query answering with box intersection
//! - `box_training`: learning box embeddings from data (vs hand-placed here)

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
    let fish = NdarrayBox::new(array![0.1, 0.1, 0.96], array![0.5, 0.5, 0.99], 1.0)?;

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
        println!("  {:>8}: volume = {:.4}", name, b.volume()?);
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
            let p = cb.containment_prob(rb)?;
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
            let p = rb.overlap_prob(cb)?;
            print!("{:>10.3}", p);
        }
        println!();
    }

    // --- Part 4: Volume ratios ---
    println!("\n--- Volume ratios (generality) ---\n");
    let vol_animal = animal.volume()?;
    let vol_mammal = mammal.volume()?;
    let vol_dog = dog.volume()?;
    println!("  animal / mammal = {:.1}x", vol_animal / vol_mammal);
    println!("  mammal / dog    = {:.1}x", vol_mammal / vol_dog);
    println!("  animal / dog    = {:.1}x", vol_animal / vol_dog);

    // --- Part 5: Verify key relationships with computed values ---
    println!("\n--- Key relationships (computed) ---\n");

    let p_dog_mammal = mammal.containment_prob(&dog)?;
    let p_mammal_animal = animal.containment_prob(&mammal)?;
    let p_dog_animal = animal.containment_prob(&dog)?;
    println!(
        "  P(dog in mammal) = {:.4}, P(mammal in animal) = {:.4}, P(dog in animal) = {:.4}",
        p_dog_mammal, p_mammal_animal, p_dog_animal
    );
    println!("  -> Transitive containment: dog inside mammal inside animal");

    let p_dog_cat = dog.overlap_prob(&cat)?;
    let p_cat_dog = cat.overlap_prob(&dog)?;
    println!(
        "\n  overlap(dog, cat) = {:.4}, overlap(cat, dog) = {:.4}",
        p_dog_cat, p_cat_dog
    );
    println!("  -> Low overlap: dog and cat occupy different regions");

    let p_fish_mammal = mammal.containment_prob(&fish)?;
    let p_fish_animal = animal.containment_prob(&fish)?;
    println!(
        "\n  P(fish in mammal) = {:.4}, P(fish in animal) = {:.4}",
        p_fish_mammal, p_fish_animal
    );
    println!("  -> Fish is inside animal but outside mammal (different sub-hierarchy)");

    // For hard boxes (NdarrayBox), containment is binary: 1.0 if fully inside, 0.0 otherwise.
    // Temperature smoothing only affects Gumbel boxes (see gumbel_box_exploration example).
    // Hard boxes give binary containment: P(dog|animal) = 1.0 when dog is inside animal
    // because hard containment is exact. For probabilistic soft boundaries, use GumbelBox.
    let v_animal = animal.volume()?;
    let v_dog = dog.volume()?;
    println!(
        "\n  volume(animal) = {:.4}, volume(dog) = {:.4}, ratio = {:.1}x",
        v_animal,
        v_dog,
        v_animal / v_dog
    );
    println!(
        "  -> More general concepts have larger volume (animal is {:.0}x bigger than dog)",
        v_animal / v_dog
    );

    // See scripts/plot_box_concept.py for a visualization of box containment, overlap,
    // and Gumbel soft boundaries.

    Ok(())
}
