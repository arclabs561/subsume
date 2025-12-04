//! Basic example demonstrating box embeddings with ndarray.

use ndarray::array;
use subsume_core::Box;
use subsume_ndarray::NdarrayBox;

fn main() -> Result<(), subsume_core::BoxError> {
    // Create a premise box (larger)
    let premise = NdarrayBox::new(
        array![0.0, 0.0, 0.0],
        array![1.0, 1.0, 1.0],
        1.0,
    )?;

    // Create a hypothesis box (contained within premise)
    let hypothesis = NdarrayBox::new(
        array![0.2, 0.2, 0.2],
        array![0.8, 0.8, 0.8],
        1.0,
    )?;

    // Compute entailment: P(hypothesis ⊆ premise)
    let entailment = premise.containment_prob(&hypothesis, 1.0)?;
    println!("Entailment probability: {:.4}", entailment);
    assert!(entailment > 0.9);

    // Compute overlap probability
    let overlap = premise.overlap_prob(&hypothesis, 1.0)?;
    println!("Overlap probability: {:.4}", overlap);

    // Compute volumes
    let premise_vol = premise.volume(1.0)?;
    let hypothesis_vol = hypothesis.volume(1.0)?;
    println!("Premise volume: {:.4}", premise_vol);
    println!("Hypothesis volume: {:.4}", hypothesis_vol);

    Ok(())
}

