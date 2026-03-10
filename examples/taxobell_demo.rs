//! TaxoBell: taxonomy expansion with Gaussian box embeddings.
//!
//! This example demonstrates the TaxoBell pipeline (WWW 2026,
//! arXiv:2601.09633) without any file I/O -- all data is inline.
//!
//! What it shows:
//! 1. Creating GaussianBox embeddings for taxonomy concepts
//! 2. KL divergence as asymmetric containment (parent-child)
//! 3. Bhattacharyya coefficient as symmetric similarity (siblings)
//! 4. TaxoBellLoss: combining all four loss components
//!
//! Run: cargo run -p subsume --example taxobell_demo

use subsume::gaussian::{bhattacharyya_coefficient, kl_divergence, GaussianBox};
use subsume::taxobell::{TaxoBellConfig, TaxoBellLoss};

fn main() -> Result<(), subsume::BoxError> {
    println!("=== TaxoBell: Gaussian Box Embeddings for Taxonomy Expansion ===\n");

    // --- Part 1: Build a small inline taxonomy ---
    //
    //              entity
    //             /      \
    //        animal      vehicle
    //       /     \      /     \
    //     dog     cat  car    truck
    //
    // Parents get wider sigmas (larger volume = more general concept).
    // Children are centered near their parent but with narrower sigmas.

    let dim = 8;

    let entity = GaussianBox::new(vec![0.0; dim], vec![4.0; dim])?;
    let animal = GaussianBox::new(vec![-1.0; dim], vec![2.0; dim])?;
    let vehicle = GaussianBox::new(vec![3.0; dim], vec![2.0; dim])?;
    let dog = GaussianBox::new(vec![-1.5; dim], vec![0.5; dim])?;
    let cat = GaussianBox::new(vec![-0.5; dim], vec![0.5; dim])?;
    let car = GaussianBox::new(vec![2.5; dim], vec![0.5; dim])?;
    let truck = GaussianBox::new(vec![3.5; dim], vec![0.5; dim])?;

    let nodes: Vec<(&str, &GaussianBox)> = vec![
        ("entity", &entity),
        ("animal", &animal),
        ("vehicle", &vehicle),
        ("dog", &dog),
        ("cat", &cat),
        ("car", &car),
        ("truck", &truck),
    ];

    println!("--- Part 1: Taxonomy nodes (dim={dim}) ---\n");
    println!("{:>10} {:>12} {:>12}", "concept", "log-volume", "sigma[0]");
    println!("{}", "-".repeat(36));
    for (name, g) in &nodes {
        println!(
            "{:>10} {:>12.4} {:>12.4}",
            name,
            g.log_volume(),
            g.sigma()[0],
        );
    }

    // --- Part 2: Asymmetric containment via KL divergence ---
    //
    // D_KL(child || parent) measures how well the child fits inside the parent.
    // Small KL -> good containment. Large KL -> child doesn't fit.

    println!("\n--- Part 2: KL divergence (child -> parent containment) ---\n");

    let parent_child_pairs: Vec<(&str, &GaussianBox, &str, &GaussianBox)> = vec![
        ("animal", &animal, "entity", &entity),
        ("vehicle", &vehicle, "entity", &entity),
        ("dog", &dog, "animal", &animal),
        ("cat", &cat, "animal", &animal),
        ("car", &car, "vehicle", &vehicle),
        ("truck", &truck, "vehicle", &vehicle),
    ];

    println!(
        "{:>10} {:>10} {:>12}  {}",
        "child", "parent", "D_KL", "interpretation"
    );
    println!("{}", "-".repeat(55));
    for (cname, child, pname, parent) in &parent_child_pairs {
        let kl = kl_divergence(child, parent)?;
        let interp = if kl < 1.0 {
            "good containment"
        } else if kl < 5.0 {
            "partial containment"
        } else {
            "poor containment"
        };
        println!("{:>10} {:>10} {:>12.4}  {}", cname, pname, kl, interp);
    }

    // Show that KL is asymmetric: D_KL(animal||dog) >> D_KL(dog||animal)
    let kl_dog_animal = kl_divergence(&dog, &animal)?;
    let kl_animal_dog = kl_divergence(&animal, &dog)?;
    println!(
        "\n  Asymmetry: D_KL(dog||animal) = {:.4}, D_KL(animal||dog) = {:.4}",
        kl_dog_animal, kl_animal_dog,
    );
    println!("  dog fits in animal (small KL), but animal does NOT fit in dog (large KL)");

    // Cross-domain: dog -> vehicle should be very large
    let kl_dog_vehicle = kl_divergence(&dog, &vehicle)?;
    println!(
        "  Cross-domain: D_KL(dog||vehicle) = {:.4} (no containment)",
        kl_dog_vehicle,
    );

    // --- Part 3: Symmetric similarity via Bhattacharyya coefficient ---
    //
    // BC(a, b) in [0, 1] measures distributional overlap.
    // BC = 1 means identical distributions; BC ~ 0 means no overlap.
    // Siblings (under the same parent) should have higher BC than non-siblings.

    println!("\n--- Part 3: Bhattacharyya coefficient (symmetric similarity) ---\n");

    let sibling_pairs: Vec<(&str, &GaussianBox, &str, &GaussianBox)> = vec![
        ("dog", &dog, "cat", &cat),               // siblings under animal
        ("car", &car, "truck", &truck),           // siblings under vehicle
        ("dog", &dog, "car", &car),               // cross-domain (not siblings)
        ("animal", &animal, "vehicle", &vehicle), // cross-domain top-level
    ];

    println!(
        "{:>10} {:>10} {:>10}  {}",
        "concept_a", "concept_b", "BC", "relationship"
    );
    println!("{}", "-".repeat(50));
    for (aname, a, bname, b) in &sibling_pairs {
        let bc = bhattacharyya_coefficient(a, b)?;
        let rel = if bc > 0.5 {
            "high overlap (siblings)"
        } else if bc > 0.01 {
            "some overlap"
        } else {
            "near-disjoint"
        };
        println!("{:>10} {:>10} {:>10.6}  {}", aname, bname, bc, rel);
    }

    // --- Part 4: TaxoBell combined loss ---
    //
    // L = alpha * L_sym + beta * L_asym + gamma * L_reg + delta * L_clip
    //
    // L_sym:  triplet loss over siblings (symmetric BC)
    // L_asym: KL containment loss over parent-child pairs
    // L_reg:  volume regularization (prevent unbounded growth)
    // L_clip: sigma clipping (prevent collapse to zero-width)

    println!("\n--- Part 4: TaxoBell combined loss ---\n");

    let config = TaxoBellConfig {
        alpha: 1.0,
        beta: 1.0,
        gamma: 0.01,
        delta: 0.01,
        symmetric_margin: 0.1,
        ..TaxoBellConfig::default()
    };
    let loss_fn = TaxoBellLoss::new(config);

    // Parent-child pairs: (child, parent)
    let positives: Vec<(&GaussianBox, &GaussianBox)> = vec![
        (&animal, &entity),
        (&vehicle, &entity),
        (&dog, &animal),
        (&cat, &animal),
        (&car, &vehicle),
        (&truck, &vehicle),
    ];

    // Sibling triples: (anchor, positive_sibling, negative_non_sibling)
    let negatives: Vec<(&GaussianBox, &GaussianBox, &GaussianBox)> = vec![
        (&dog, &cat, &car),        // dog-cat are siblings, car is not
        (&car, &truck, &dog),      // car-truck are siblings, dog is not
        (&animal, &vehicle, &dog), // animal-vehicle are siblings under entity, dog is not
    ];

    let all_boxes: Vec<&GaussianBox> = vec![&entity, &animal, &vehicle, &dog, &cat, &car, &truck];

    let result = loss_fn.combined_loss(&positives, &negatives, &all_boxes)?;

    println!("  L_sym  (symmetric triplet)  = {:.6}", result.l_sym);
    println!("  L_asym (KL containment)     = {:.6}", result.l_asym);
    println!("  L_reg  (volume regulation)  = {:.6}", result.l_reg);
    println!("  L_clip (sigma clipping)     = {:.6}", result.l_clip);
    println!();
    println!(
        "  Total = {:.2}*{:.4} + {:.2}*{:.4} + {:.2}*{:.4} + {:.2}*{:.4}",
        1.0, result.l_sym, 1.0, result.l_asym, 0.01, result.l_reg, 0.01, result.l_clip,
    );
    println!("        = {:.6}", result.total);

    // L_reg = 0: no volume regularization pressure (sigmas are hand-chosen, not learned)
    // L_clip > 0: N of 7 boxes have sigma below the clipping threshold (sigma_min)
    // In training, L_clip penalizes collapsed boxes to prevent degenerate solutions.

    println!("\nKey takeaways:");
    println!("  - KL divergence is asymmetric: it measures directed containment (child in parent)");
    println!("  - Bhattacharyya coefficient is symmetric: it measures distributional overlap");
    println!("  - TaxoBell combines both with regularization for end-to-end taxonomy training");
    println!("  - Wider sigmas = more general concepts; narrower sigmas = more specific");
    println!("  - Reference: TaxoBell (WWW 2026, arXiv:2601.09633)");

    Ok(())
}
