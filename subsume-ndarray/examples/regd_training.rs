//! Example: RegD-style training using depth distance for hierarchy learning.
//!
//! This example demonstrates the RegD (2025) approach:
//! - Training with depth distance instead of Euclidean distance
//! - Addressing crowding effect in hierarchies
//! - Achieving hyperbolic-like expressiveness with Euclidean boxes
//!
//! Reference: Yang & Chen (2025): "Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions"

use ndarray::array;
use subsume_core::Box;
use subsume_ndarray::{distance, NdarrayBox};

fn main() -> Result<(), subsume_core::BoxError> {
    println!("RegD: Training with Depth Distance for Hierarchy Learning");
    println!("==========================================================\n");

    // Create a hierarchy: Animal > Mammal > {Dog, Cat} > {Fido, Whiskers}
    let animal = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0)?;
    let mammal = NdarrayBox::new(array![2.0, 2.0], array![8.0, 8.0], 1.0)?;
    let bird = NdarrayBox::new(array![7.0, 7.0], array![9.0, 9.0], 1.0)?;
    let dog = NdarrayBox::new(array![3.0, 3.0], array![7.0, 7.0], 1.0)?;
    let cat = NdarrayBox::new(array![2.5, 2.5], array![4.5, 4.5], 1.0)?;
    let sparrow = NdarrayBox::new(array![7.5, 7.5], array![8.5, 8.5], 1.0)?;

    println!("1. Hierarchy Structure");
    println!("----------------------");
    println!("Animal");
    println!("  ├─ Mammal");
    println!("  │   ├─ Dog");
    println!("  │   └─ Cat");
    println!("  └─ Bird");
    println!("      └─ Sparrow");
    println!();

    println!("2. Crowding Effect Problem");
    println!("---------------------------");
    println!("With Euclidean distance, many children cluster together:");

    // Calculate Euclidean distances between siblings
    let euclidean_dog_cat = dog.distance(&cat)?;
    let euclidean_dog_sparrow = dog.distance(&sparrow)?;
    let euclidean_cat_sparrow = cat.distance(&sparrow)?;

    println!(
        "  Euclidean distance (Dog, Cat):        {:.4}",
        euclidean_dog_cat
    );
    println!(
        "  Euclidean distance (Dog, Sparrow):   {:.4}",
        euclidean_dog_sparrow
    );
    println!(
        "  Euclidean distance (Cat, Sparrow):    {:.4}",
        euclidean_cat_sparrow
    );
    println!();
    println!("Problem: Siblings (Dog, Cat) are closer than cross-branch (Dog, Sparrow)");
    println!("This causes 'crowding' - many children cluster in the same region.");
    println!();

    println!("3. Depth Distance Solution");
    println!("-------------------------");
    println!("Depth distance incorporates volume, providing better separation:");

    let depth_dog_cat = distance::depth_distance(&dog, &cat, 1.0, 0.1)?;
    let depth_dog_sparrow = distance::depth_distance(&dog, &sparrow, 1.0, 0.1)?;
    let depth_cat_sparrow = distance::depth_distance(&cat, &sparrow, 1.0, 0.1)?;

    println!("  Depth distance (Dog, Cat):        {:.4}", depth_dog_cat);
    println!(
        "  Depth distance (Dog, Sparrow):    {:.4}",
        depth_dog_sparrow
    );
    println!(
        "  Depth distance (Cat, Sparrow):     {:.4}",
        depth_cat_sparrow
    );
    println!();
    println!("Note: Depth distance provides better discrimination by incorporating");
    println!("volume differences, addressing the crowding effect.");
    println!();

    println!("4. Hierarchy Depth Property");
    println!("---------------------------");
    println!("Depth distance increases with hierarchy depth:");

    let depth_animal_mammal = distance::depth_distance(&animal, &mammal, 1.0, 0.1)?;
    let depth_mammal_dog = distance::depth_distance(&mammal, &dog, 1.0, 0.1)?;
    let depth_animal_dog = distance::depth_distance(&animal, &dog, 1.0, 0.1)?;

    println!("  Depth(Animal, Mammal): {:.4}", depth_animal_mammal);
    println!("  Depth(Mammal, Dog):     {:.4}", depth_mammal_dog);
    println!("  Depth(Animal, Dog):     {:.4}", depth_animal_dog);
    println!();
    println!("✓ Depth(Animal, Dog) >= Depth(Animal, Mammal)");
    println!("✓ Depth(Animal, Dog) >= Depth(Mammal, Dog)");
    println!();

    println!("5. Boundary Distance for Inclusion Chains");
    println!("-----------------------------------------");
    println!("Boundary distance captures depth in inclusion chains:");

    if let Some(boundary_animal_mammal) = distance::boundary_distance(&animal, &mammal, 1.0)? {
        println!("  Boundary(Animal ⊇ Mammal): {:.4}", boundary_animal_mammal);
    }

    if let Some(boundary_mammal_dog) = distance::boundary_distance(&mammal, &dog, 1.0)? {
        println!("  Boundary(Mammal ⊇ Dog):     {:.4}", boundary_mammal_dog);
    }

    if let Some(boundary_animal_dog) = distance::boundary_distance(&animal, &dog, 1.0)? {
        println!("  Boundary(Animal ⊇ Dog):      {:.4}", boundary_animal_dog);
    }
    println!();
    println!("Boundary distance discriminates between regions in inclusion chains,");
    println!("capturing 'depth' (how nested a box is within its parent).");
    println!();

    println!("6. Training with Depth Distance");
    println!("-------------------------------");
    println!("In RegD training, the loss function uses depth distance:");
    println!("  L = Σ positive_pairs depth_distance(box_i, box_j)");
    println!("      + Σ negative_pairs max(0, margin - depth_distance(box_i, box_j))");
    println!();
    println!("This encourages:");
    println!("  - Siblings to be separated (addresses crowding)");
    println!("  - Hierarchy depth to be preserved");
    println!("  - Better discrimination in deep hierarchies");
    println!();

    println!("7. Hyperbolic-Like Expressiveness");
    println!("----------------------------------");
    println!("RegD achieves hyperbolic-like expressiveness with Euclidean boxes:");
    println!("  - Depth distance emulates hyperbolic geometry properties");
    println!("  - Eliminates precision issues of hyperbolic methods");
    println!("  - Uses elementary arithmetic operations (faster, more stable)");
    println!();
    println!("Key advantage: Euclidean operations are more stable and efficient");
    println!("than hyperbolic operations, while achieving similar expressiveness.");

    Ok(())
}
