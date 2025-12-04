//! Basic example demonstrating box embeddings with Candle.

use candle_core::{Device, Tensor};
use subsume_core::Box;
use subsume_candle::CandleBox;

fn main() -> Result<(), subsume_core::BoxError> {
    let device = Device::Cpu;

    // Create a premise box (larger)
    let premise = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0, 0.0], &device)?,
        Tensor::new(&[1.0f32, 1.0, 1.0], &device)?,
        1.0,
    )?;

    // Create a hypothesis box (contained within premise)
    let hypothesis = CandleBox::new(
        Tensor::new(&[0.2f32, 0.2, 0.2], &device)?,
        Tensor::new(&[0.8f32, 0.8, 0.8], &device)?,
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

