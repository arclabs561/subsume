//! Poincare ball embeddings for tree-like hierarchies.
//!
//! Hyperbolic space grows exponentially with radius, so deep trees embed with
//! low distortion -- something Euclidean space cannot do. In the Poincare ball
//! model, general concepts sit near the origin (small norm) and specific ones
//! sit near the boundary (large norm). Distance between sibling subtrees is
//! large even when they share a nearby parent, thanks to the exponential
//! expansion of volume near the boundary.
//!
//! This example builds a 15-entity biological taxonomy across 4 depth levels
//! (Life > Kingdom > Order > Species) and shows:
//! 1. Norm increases with depth (general near origin, specific near boundary)
//! 2. Pairwise hyperbolic distances respect the hierarchy
//! 3. `hierarchy_preserved` verifies parent-child norm ordering
//! 4. Distance grows exponentially with depth, demonstrating capacity advantage
//!
//! Reference: Nickel & Kiela (2017), "Poincare Embeddings for Learning
//! Hierarchical Representations"
//!
//! Run: cargo run -p subsume --example hyperbolic_demo
//!
//! Related examples:
//! - `containment_hierarchy`: Euclidean box alternative for hierarchies
//! - `dataset_training`: training pipeline for learned embeddings

use subsume::{hierarchy_preserved, pairwise_distances, Curvature, PoincareBallPoint};

/// Place a point along a direction at a given norm in 2D.
///
/// `angle` in radians, `norm` in (0, 1).
fn place(norm: f64, angle: f64, curvature: Curvature) -> PoincareBallPoint {
    let x = norm * angle.cos();
    let y = norm * angle.sin();
    PoincareBallPoint::new(vec![x, y], curvature).expect("coords inside ball")
}

fn main() -> Result<(), subsume::HyperbolicError> {
    let c = Curvature::STANDARD;
    let pi = std::f64::consts::PI;

    println!("=== Poincare Ball Embeddings: Biological Taxonomy ===\n");

    // ---------------------------------------------------------------
    // Build a 4-level taxonomy (15 entities).
    //
    // Depth 0: Life (origin)
    // Depth 1: Animalia, Plantae, Fungi
    // Depth 2: Carnivora, Primates, Rosales, Poales, Agaricales
    // Depth 3: Wolf, Lion, Human, Chimp, Rose, Wheat, Amanita
    //
    // Norms increase with depth; angular separation distinguishes
    // siblings within a subtree.
    // ---------------------------------------------------------------

    let names: Vec<&str> = vec![
        // depth 0
        "Life",
        // depth 1 (3)
        "Animalia",
        "Plantae",
        "Fungi",
        // depth 2 (5)
        "Carnivora",
        "Primates",
        "Rosales",
        "Poales",
        "Agaricales",
        // depth 3 (7 -- note: 6 would also work, using 7 for breadth)
        "Wolf",
        "Lion",
        "Human",
        "Chimp",
        "Rose",
        "Wheat",
        "Amanita",
    ];

    let depths: Vec<u32> = vec![
        0, // Life
        1, 1, 1, // kingdoms
        2, 2, 2, 2, 2, // orders
        3, 3, 3, 3, 3, 3, 3, // species
    ];

    // Norm bands per depth level.
    let norm_for_depth: [f64; 4] = [0.0, 0.30, 0.60, 0.88];

    // Angular layout: kingdoms get wide separation, sub-groups cluster.
    let angles: Vec<f64> = vec![
        0.0,                   // Life (origin, angle irrelevant)
        0.0,                   // Animalia
        2.0 * pi / 3.0,        // Plantae
        4.0 * pi / 3.0,        // Fungi
        -0.15,                 // Carnivora  (under Animalia)
        0.15,                  // Primates   (under Animalia)
        2.0 * pi / 3.0 - 0.15, // Rosales    (under Plantae)
        2.0 * pi / 3.0 + 0.15, // Poales     (under Plantae)
        4.0 * pi / 3.0,        // Agaricales (under Fungi)
        -0.22,                 // Wolf       (under Carnivora)
        -0.08,                 // Lion       (under Carnivora)
        0.08,                  // Human      (under Primates)
        0.22,                  // Chimp      (under Primates)
        2.0 * pi / 3.0 - 0.15, // Rose       (under Rosales)
        2.0 * pi / 3.0 + 0.15, // Wheat      (under Poales)
        4.0 * pi / 3.0,        // Amanita    (under Agaricales)
    ];

    let points: Vec<PoincareBallPoint> = names
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let d = depths[i] as usize;
            place(norm_for_depth[d], angles[i], c)
        })
        .collect();

    // ---------------------------------------------------------------
    // Part 1: Norm vs depth
    // ---------------------------------------------------------------
    println!("--- Norm by depth (general near origin, specific near boundary) ---\n");
    println!(
        "{:>14} {:>6} {:>8} {:>10}",
        "entity", "depth", "norm", "conformal"
    );
    for (i, name) in names.iter().enumerate() {
        let norm = points[i].norm_squared().sqrt();
        let cf = points[i].conformal_factor()?;
        println!("{:>14} {:>6} {:>8.4} {:>10.4}", name, depths[i], norm, cf);
    }
    println!();
    println!("  Conformal factor = 2/(1 - ||x||^2). Grows sharply near the boundary,");
    println!("  meaning a small coordinate shift near the boundary covers much more");
    println!("  hyperbolic distance than the same shift near the origin.\n");

    // ---------------------------------------------------------------
    // Part 2: Pairwise distances -- hierarchy respected
    // ---------------------------------------------------------------
    let dists = pairwise_distances(&points)?;

    println!("--- Selected pairwise hyperbolic distances ---\n");
    let pairs_to_show: Vec<(usize, usize, &str)> = vec![
        (0, 1, "Life -> Animalia (depth 0->1)"),
        (1, 4, "Animalia -> Carnivora (depth 1->2)"),
        (4, 9, "Carnivora -> Wolf (depth 2->3)"),
        (9, 10, "Wolf <-> Lion (siblings)"),
        (9, 11, "Wolf <-> Human (cross-order)"),
        (9, 14, "Wolf <-> Rose (cross-kingdom)"),
        (11, 12, "Human <-> Chimp (siblings)"),
        (0, 15, "Life -> Amanita (root to leaf)"),
    ];
    for (i, j, label) in &pairs_to_show {
        println!("  {:<42} d = {:.4}", label, dists[*i][*j]);
    }

    // ---------------------------------------------------------------
    // Part 3: hierarchy_preserved
    // ---------------------------------------------------------------
    let parent_child: Vec<(usize, usize)> = vec![
        // depth 0 -> 1
        (0, 1),
        (0, 2),
        (0, 3),
        // depth 1 -> 2
        (1, 4),
        (1, 5),
        (2, 6),
        (2, 7),
        (3, 8),
        // depth 2 -> 3
        (4, 9),
        (4, 10),
        (5, 11),
        (5, 12),
        (6, 13),
        (7, 14),
        (8, 15),
    ];

    let accuracy = hierarchy_preserved(&points, &parent_child)?;
    println!(
        "\n--- Hierarchy preservation: {:.1}% ({}/{} parent-child pairs correct) ---\n",
        accuracy * 100.0,
        (accuracy * parent_child.len() as f64).round() as usize,
        parent_child.len()
    );

    // ---------------------------------------------------------------
    // Part 4: Exponential capacity -- distance grows with depth
    // ---------------------------------------------------------------
    println!("--- Exponential capacity: distance from origin by depth ---\n");
    println!(
        "{:>6} {:>14} {:>14} {:>14}",
        "depth", "norm", "d(origin,p)", "ratio"
    );

    let origin = PoincareBallPoint::origin(2, c);
    let mut prev_dist = None;
    for d in 0..4u32 {
        // Pick the first entity at each depth.
        let idx = depths.iter().position(|&dd| dd == d).unwrap();
        let norm = points[idx].norm_squared().sqrt();
        let dist = origin.distance(&points[idx])?;
        let ratio = match prev_dist {
            Some(pd) if pd > 1e-10 => format!("{:.2}x", dist / pd),
            _ => String::from("-"),
        };
        println!("{:>6} {:>14.4} {:>14.4} {:>14}", d, norm, dist, ratio);
        prev_dist = Some(dist);
    }

    println!();
    println!("  Distance-to-origin grows faster than linearly with norm because the");
    println!("  hyperbolic metric inflates near the boundary: d = 2 * arctanh(||x||).");
    println!("  This is the exponential capacity that lets trees embed without distortion.\n");

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    println!("Key observations:");
    println!("  - All parent-child norm orderings are satisfied (hierarchy preserved)");
    println!(
        "  - Cross-kingdom distances (Wolf<->Rose) exceed within-order distances (Wolf<->Lion)"
    );
    println!("  - Hyperbolic distance from origin accelerates with depth, encoding exponential");
    println!("    branching capacity in bounded coordinates");

    Ok(())
}
