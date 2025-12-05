//! Example: Concept2Box joint learning with Candle.
//!
//! Demonstrates joint learning of concept boxes and entity vectors
//! as described in Concept2Box (ACL 2023).

use candle_core::{Device, Tensor};
use subsume_candle::{distance, CandleBox};
use subsume_core::Box;

fn main() -> Result<(), subsume_core::BoxError> {
    let device = Device::Cpu;
    println!("Concept2Box Joint Learning (Candle)");
    println!("====================================\n");

    // Concepts as boxes
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

    // Entities as vectors (points)
    let entity_fido = Tensor::new(&[4.0f32, 4.0], &device)
        .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?;

    // Compute vector-to-box distances
    let dist_fido_animal = distance::vector_to_box_distance(&entity_fido, &animal)?;
    let dist_fido_mammal = distance::vector_to_box_distance(&entity_fido, &mammal)?;

    println!("Distance from entity 'Fido' to Animal box: {:.4}", dist_fido_animal);
    println!("Distance from entity 'Fido' to Mammal box: {:.4}", dist_fido_mammal);
    println!();

    // Concept-concept relations (box-box)
    let containment = animal.containment_prob(&mammal, 1.0)?;
    println!("Animal contains Mammal: {:.4}", containment);

    println!("\nConcept2Box enables:");
    println!("- Concepts as boxes (hierarchical, containment)");
    println!("- Entities as vectors (instances, efficient)");
    println!("- Joint learning of both representations");

    Ok(())
}

