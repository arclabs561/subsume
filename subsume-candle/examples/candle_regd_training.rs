//! Example: RegD-style training with depth distance (Candle).
//!
//! Demonstrates RegD (2025) depth distance for hierarchy learning.

use candle_core::{Device, Tensor};
use subsume_candle::{distance, CandleBox};

fn main() -> Result<(), subsume_core::BoxError> {
    let device = Device::Cpu;
    println!("RegD Training with Depth Distance (Candle)");
    println!("===========================================\n");

    // Create hierarchy: Animal > Mammal > Dog
    let animal = CandleBox::new(
        Tensor::new(&[0.0f32, 0.0], &device)
            .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?,
        Tensor::new(&[10.0f32, 10.0], &device)
            .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?,
        1.0,
    )?;

    let mammal = CandleBox::new(
        Tensor::new(&[2.0f32, 2.0], &device)
            .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?,
        Tensor::new(&[8.0f32, 8.0], &device)
            .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?,
        1.0,
    )?;

    let dog = CandleBox::new(
        Tensor::new(&[3.0f32, 3.0], &device)
            .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?,
        Tensor::new(&[7.0f32, 7.0], &device)
            .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?,
        1.0,
    )?;

    // Depth distance (RegD 2025)
    let depth_animal_mammal = distance::depth_distance(&animal, &mammal, 1.0, 0.1)?;
    let depth_mammal_dog = distance::depth_distance(&mammal, &dog, 1.0, 0.1)?;

    println!("Depth distance (Animal, Mammal): {:.4}", depth_animal_mammal);
    println!("Depth distance (Mammal, Dog):    {:.4}", depth_mammal_dog);
    println!();

    // Boundary distance
    if let Some(boundary) = distance::boundary_distance(&animal, &mammal, 1.0)? {
        println!("Boundary distance (Animal ⊇ Mammal): {:.4}", boundary);
    }

    println!("\nRegD enables:");
    println!("- Hyperbolic-like expressiveness with Euclidean boxes");
    println!("- Depth distance captures hierarchy depth");
    println!("- Better discrimination in inclusion chains");

    Ok(())
}

