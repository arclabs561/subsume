//! Example: Demonstrating recent research distance metrics (2023-2025) with Candle.
//!
//! This example showcases the new distance metrics from recent papers:
//! - **RegD (2025)**: Depth distance and boundary distance
//! - **Concept2Box (2023)**: Vector-to-box distance for hybrid representations

use candle_core::{Device, Tensor};
use subsume_candle::{distance, CandleBox};
use subsume_core::Box;

fn main() -> Result<(), subsume_core::BoxError> {
    let device = Device::Cpu;
    println!("Recent Research Distance Metrics (Candle)");
    println!("==========================================\n");

    // Create boxes representing a hierarchy: Animal > Mammal > Dog
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

    // 1. Depth Distance (RegD 2025)
    println!("1. Depth Distance (RegD 2025)");
    println!("-------------------------------");
    let depth_animal_mammal = distance::depth_distance(&animal, &mammal, 1.0, 0.1)?;
    let depth_mammal_dog = distance::depth_distance(&mammal, &dog, 1.0, 0.1)?;
    let depth_animal_dog = distance::depth_distance(&animal, &dog, 1.0, 0.1)?;

    println!("Depth distance (Animal, Mammal): {:.4}", depth_animal_mammal);
    println!("Depth distance (Mammal, Dog):    {:.4}", depth_mammal_dog);
    println!("Depth distance (Animal, Dog):     {:.4}", depth_animal_dog);
    println!();

    // 2. Boundary Distance (RegD 2025)
    println!("2. Boundary Distance (RegD 2025)");
    println!("---------------------------------");
    if let Some(boundary_animal_mammal) = distance::boundary_distance(&animal, &mammal, 1.0)? {
        println!("Boundary distance (Animal ⊇ Mammal): {:.4}", boundary_animal_mammal);
    }
    if let Some(boundary_mammal_dog) = distance::boundary_distance(&mammal, &dog, 1.0)? {
        println!("Boundary distance (Mammal ⊇ Dog):    {:.4}", boundary_mammal_dog);
    }
    if let Some(boundary_animal_dog) = distance::boundary_distance(&animal, &dog, 1.0)? {
        println!("Boundary distance (Animal ⊇ Dog):     {:.4}", boundary_animal_dog);
    }
    println!();

    // 3. Vector-to-Box Distance (Concept2Box 2023)
    println!("3. Vector-to-Box Distance (Concept2Box 2023)");
    println!("---------------------------------------------");
    let entity_fido = Tensor::new(&[4.0f32, 4.0], &device)
        .map_err(|e| subsume_core::BoxError::Internal(e.to_string()))?;

    let dist_fido_animal = distance::vector_to_box_distance(&entity_fido, &animal)?;
    let dist_fido_mammal = distance::vector_to_box_distance(&entity_fido, &mammal)?;
    let dist_fido_dog = distance::vector_to_box_distance(&entity_fido, &dog)?;

    println!("Distance from entity 'Fido' to Animal box: {:.4}", dist_fido_animal);
    println!("Distance from entity 'Fido' to Mammal box: {:.4}", dist_fido_mammal);
    println!("Distance from entity 'Fido' to Dog box:    {:.4}", dist_fido_dog);
    println!();

    // 4. Depth Similarity
    println!("4. Depth Similarity");
    println!("-------------------");
    use subsume_core::distance::depth_similarity;
    let sim_animal_mammal = depth_similarity(&animal, &mammal, 1.0, 0.1)?;
    let sim_mammal_dog = depth_similarity(&mammal, &dog, 1.0, 0.1)?;

    println!("Depth similarity (Animal, Mammal): {:.4}", sim_animal_mammal);
    println!("Depth similarity (Mammal, Dog):    {:.4}", sim_mammal_dog);

    Ok(())
}

