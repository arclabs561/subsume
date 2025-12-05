//! Example: BoxE training with translational bumps.
//!
//! Demonstrates the BoxE model (Boratko et al. 2020) with relation-specific
//! translation vectors (bumps) for knowledge graph completion.

use ndarray::array;
use subsume_core::boxe::{boxe_loss, boxe_score, Bump};
use subsume_core::Box as CoreBox;
use subsume_ndarray::NdarrayBox;

fn main() -> Result<(), subsume_core::BoxError> {
    println!("BoxE Training Example");
    println!("====================\n");

    // Create entity boxes
    let head_box = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0)?;
    let tail_box = NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0)?;

    // Create relation bump (translation vector)
    let bump = Bump::new(vec![0.1, 0.1]);

    println!("Head box: min=[0.0, 0.0], max=[1.0, 1.0]");
    println!("Tail box: min=[0.2, 0.2], max=[0.8, 0.8]");
    println!("Relation bump: [0.1, 0.1]\n");

    // Compute BoxE score
    let score = boxe_score(
        head_box.min().as_slice().unwrap(),
        head_box.max().as_slice().unwrap(),
        tail_box.min().as_slice().unwrap(),
        tail_box.max().as_slice().unwrap(),
        &bump.translation,
        1.0,
    )?;

    println!("BoxE score (head + bump, tail): {:.4}", score);

    // Simulate training: positive vs negative sample
    let positive_score = score;
    let negative_score = 0.1; // Lower score for corrupted triple

    let loss = boxe_loss(positive_score, negative_score, 1.0);
    println!("BoxE loss (margin=1.0): {:.4}", loss);

    println!("\nBoxE features:");
    println!("- Translational bumps: relation-specific transformations");
    println!("- Margin-based ranking loss");
    println!("- SOTA performance on FB15k-237, WN18RR, YAGO3-10");

    Ok(())
}

