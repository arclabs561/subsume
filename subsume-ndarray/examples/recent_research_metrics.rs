//! Example: Demonstrating recent research distance metrics (2023-2025).
//!
//! This example showcases the new distance metrics from recent papers:
//! - **RegD (2025)**: Depth distance and boundary distance
//! - **Concept2Box (2023)**: Vector-to-box distance for hybrid representations
//!
//! These metrics enable:
//! - Hyperbolic-like expressiveness with Euclidean boxes (RegD)
//! - Joint learning of concept boxes and entity vectors (Concept2Box)
//! - Better discrimination in inclusion chains (boundary distance)

use ndarray::array;
use subsume_core::Box;
use subsume_ndarray::{distance, NdarrayBox};

fn main() -> Result<(), subsume_core::BoxError> {
    println!("Recent Research Distance Metrics");
    println!("================================\n");

    // Create boxes representing a hierarchy: Animal > Mammal > Dog
    let animal = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0)?;

    let mammal = NdarrayBox::new(array![2.0, 2.0], array![8.0, 8.0], 1.0)?;

    let dog = NdarrayBox::new(array![3.0, 3.0], array![7.0, 7.0], 1.0)?;

    // 1. Depth Distance (RegD 2025)
    println!("1. Depth Distance (RegD 2025)");
    println!("-------------------------------");
    println!("Depth distance incorporates volume into distance calculations,");
    println!("enabling Euclidean boxes to achieve hyperbolic-like expressiveness.\n");

    let depth_animal_mammal = distance::depth_distance(&animal, &mammal, 1.0, 0.1)?;
    let depth_mammal_dog = distance::depth_distance(&mammal, &dog, 1.0, 0.1)?;
    let depth_animal_dog = distance::depth_distance(&animal, &dog, 1.0, 0.1)?;

    println!(
        "Depth distance (Animal, Mammal): {:.4}",
        depth_animal_mammal
    );
    println!("Depth distance (Mammal, Dog):    {:.4}", depth_mammal_dog);
    println!("Depth distance (Animal, Dog):     {:.4}", depth_animal_dog);
    println!();

    // Note: Depth distance should increase with hierarchy depth
    println!("Note: Depth distance increases with hierarchy depth,");
    println!("capturing the 'crowding effect' where many children cluster together.\n");

    // 2. Boundary Distance (RegD 2025)
    println!("2. Boundary Distance (RegD 2025)");
    println!("---------------------------------");
    println!("Boundary distance captures containment relationships and");
    println!("discriminates between regions in inclusion chains.\n");

    if let Some(boundary_animal_mammal) = distance::boundary_distance(&animal, &mammal, 1.0)? {
        println!(
            "Boundary distance (Animal ⊇ Mammal): {:.4}",
            boundary_animal_mammal
        );
    }

    if let Some(boundary_mammal_dog) = distance::boundary_distance(&mammal, &dog, 1.0)? {
        println!(
            "Boundary distance (Mammal ⊇ Dog):    {:.4}",
            boundary_mammal_dog
        );
    }

    if let Some(boundary_animal_dog) = distance::boundary_distance(&animal, &dog, 1.0)? {
        println!(
            "Boundary distance (Animal ⊇ Dog):     {:.4}",
            boundary_animal_dog
        );
    }
    println!();

    println!("Note: Boundary distance measures the minimum gap between");
    println!("inner and outer box boundaries, capturing 'depth' in inclusion chains.\n");

    // 3. Vector-to-Box Distance (Concept2Box 2023)
    println!("3. Vector-to-Box Distance (Concept2Box 2023)");
    println!("---------------------------------------------");
    println!("This metric bridges concept box embeddings and entity vector embeddings,");
    println!(
        "enabling hybrid representations where concepts are boxes and entities are vectors.\n"
    );

    // Create a point representing an entity (e.g., "Fido" the dog)
    let entity_fido = array![4.0, 4.0];

    // Compute distances from entity to concept boxes
    let dist_fido_animal = distance::vector_to_box_distance(&entity_fido, &animal)?;
    let dist_fido_mammal = distance::vector_to_box_distance(&entity_fido, &mammal)?;
    let dist_fido_dog = distance::vector_to_box_distance(&entity_fido, &dog)?;

    println!(
        "Distance from entity 'Fido' to Animal box: {:.4}",
        dist_fido_animal
    );
    println!(
        "Distance from entity 'Fido' to Mammal box: {:.4}",
        dist_fido_mammal
    );
    println!(
        "Distance from entity 'Fido' to Dog box:    {:.4}",
        dist_fido_dog
    );
    println!();

    // Since Fido is inside all boxes, distances should be 0
    println!("Note: Since 'Fido' is inside all concept boxes, distances are 0.");
    println!("This demonstrates that the entity belongs to all these concepts.\n");

    // Example with entity outside a box
    let entity_cat = array![15.0, 15.0]; // Outside all boxes
    let dist_cat_animal = distance::vector_to_box_distance(&entity_cat, &animal)?;
    println!(
        "Distance from entity 'Cat' (outside) to Animal box: {:.4}",
        dist_cat_animal
    );
    println!();

    // 4. Depth Similarity
    println!("4. Depth Similarity");
    println!("-------------------");
    println!("Depth similarity converts depth distance to a similarity score [0, 1].\n");

    use subsume_core::distance::depth_similarity;
    let sim_animal_mammal = depth_similarity(&animal, &mammal, 1.0, 0.1)?;
    let sim_mammal_dog = depth_similarity(&mammal, &dog, 1.0, 0.1)?;

    println!(
        "Depth similarity (Animal, Mammal): {:.4}",
        sim_animal_mammal
    );
    println!("Depth similarity (Mammal, Dog):    {:.4}", sim_mammal_dog);
    println!();

    // Summary
    println!("Summary");
    println!("=======");
    println!("These metrics enable:");
    println!("- Hyperbolic-like expressiveness with Euclidean boxes (RegD)");
    println!("- Joint learning of concepts (boxes) and entities (vectors) (Concept2Box)");
    println!("- Better discrimination in hierarchical structures");
    println!("- More expressive distance metrics for knowledge graphs");

    Ok(())
}
