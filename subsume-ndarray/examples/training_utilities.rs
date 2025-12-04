//! Example demonstrating training utilities: log-space volume, regularization, temperature scheduling, and loss functions.

use ndarray::Array1;
use subsume_core::{
    temperature_scheduler, volume_containment_loss, volume_overlap_loss, volume_regularization,
    Box, MIN_TEMPERATURE,
};
use subsume_ndarray::{NdarrayBox, NdarrayGumbelBox};

fn main() -> Result<(), subsume_core::BoxError> {
    println!("=== Training Utilities Example ===\n");

    // 1. Log-space volume computation (critical for high-dimensional boxes)
    println!("1. Log-space Volume Computation");
    println!("   For high-dimensional boxes, direct multiplication can underflow/overflow.");

    let high_dim_box = NdarrayBox::new(
        Array1::from(vec![0.0; 20]),
        Array1::from(vec![0.1; 20]),
        1.0,
    )?;

    let volume = high_dim_box.volume(1.0)?;
    println!(
        "   High-dimensional box (20D, side=0.1): volume = {:.6e}",
        volume
    );
    println!("   Using log-space prevents underflow!\n");

    // 2. Volume regularization
    println!("2. Volume Regularization");
    println!("   Prevents boxes from becoming too large or too small during training.");

    let large_box = NdarrayBox::new(
        Array1::from(vec![0.0, 0.0]),
        Array1::from(vec![10.0, 10.0]),
        1.0,
    )?;
    let vol = large_box.volume(1.0)?;
    let penalty = volume_regularization(vol, 5.0, 0.01, 0.1);
    println!(
        "   Box volume: {:.2}, threshold: [0.01, 5.0], lambda: 0.1",
        vol
    );
    println!("   Regularization penalty: {:.4}\n", penalty);

    // 3. Temperature scheduling (annealing)
    println!("3. Temperature Scheduling");
    println!("   Start with high temperature (exploration) and decrease (exploitation).");

    let initial_temp = 10.0;
    let decay_rate = 0.95;
    println!(
        "   Initial temperature: {}, decay rate: {}",
        initial_temp, decay_rate
    );

    for step in [0, 10, 50, 100] {
        let temp = temperature_scheduler(initial_temp, decay_rate, step, MIN_TEMPERATURE);
        println!("   Step {}: temperature = {:.4}", step, temp);
    }
    println!();

    // 4. Volume-based loss functions
    println!("4. Volume-based Loss Functions");
    println!("   For training box embeddings with containment/overlap relationships.\n");

    // Positive containment pair (should have high containment)
    let premise = NdarrayBox::new(
        Array1::from(vec![0.0, 0.0]),
        Array1::from(vec![1.0, 1.0]),
        1.0,
    )?;
    let hypothesis = NdarrayBox::new(
        Array1::from(vec![0.2, 0.2]),
        Array1::from(vec![0.8, 0.8]),
        1.0,
    )?;

    let containment_prob = premise.containment_prob(&hypothesis, 1.0)?;
    let loss_pos = volume_containment_loss(containment_prob, 1.0, 0.1);
    println!("   Positive pair (hypothesis ⊆ premise):");
    println!(
        "     Containment prob: {:.4}, Loss: {:.4}",
        containment_prob, loss_pos
    );

    // Negative containment pair (should have low containment)
    let disjoint = NdarrayBox::new(
        Array1::from(vec![2.0, 2.0]),
        Array1::from(vec![3.0, 3.0]),
        1.0,
    )?;
    let containment_prob_neg = premise.containment_prob(&disjoint, 1.0)?;
    let loss_neg = volume_containment_loss(containment_prob_neg, 0.0, 0.1);
    println!("   Negative pair (disjoint boxes):");
    println!(
        "     Containment prob: {:.4}, Loss: {:.4}",
        containment_prob_neg, loss_neg
    );

    // Overlap loss
    let overlap_prob = premise.overlap_prob(&hypothesis, 1.0)?;
    let loss_overlap = volume_overlap_loss(overlap_prob, 1.0, 0.1);
    println!("   Overlapping pair:");
    println!(
        "     Overlap prob: {:.4}, Loss: {:.4}\n",
        overlap_prob, loss_overlap
    );

    // 5. Training loop example
    println!("5. Example Training Loop");
    println!("   Combining all utilities in a training scenario:\n");

    let gumbel_box = NdarrayGumbelBox::new(
        Array1::from(vec![0.0, 0.0]),
        Array1::from(vec![1.0, 1.0]),
        10.0, // Start with high temperature
    )?;

    for step in 0..5 {
        let temp = temperature_scheduler(10.0, 0.9, step, MIN_TEMPERATURE);
        // In real training, you would update the box parameters here
        let vol = gumbel_box.volume(temp)?;
        let reg_penalty = volume_regularization(vol, 1.0, 0.01, 0.1);

        println!(
            "   Step {}: temp={:.4}, vol={:.4}, reg={:.4}",
            step, temp, vol, reg_penalty
        );
    }

    Ok(())
}
