//! Example: Using box embeddings for knowledge graph reasoning.
//!
//! This example demonstrates how to use box embeddings to model hierarchical
//! relationships in knowledge graphs, such as "Animal contains Dog" or
//! "Location contains City".

use ndarray::array;
use subsume_core::{Box, BoxCollection, BoxEmbedding};
use subsume_ndarray::NdarrayBox;

fn main() -> Result<(), subsume_core::BoxError> {
    // Define entities as boxes in a 3D embedding space
    // Higher-level concepts (like "Animal") should contain more specific ones (like "Dog")

    // Animal: a large box covering the animal concept space
    let animal = NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0)?;

    // Dog: a smaller box contained within Animal
    let dog = NdarrayBox::new(array![0.2, 0.3, 0.1], array![0.6, 0.7, 0.5], 1.0)?;

    // Cat: another box contained within Animal, but separate from Dog
    let cat = NdarrayBox::new(array![0.4, 0.1, 0.6], array![0.8, 0.5, 0.9], 1.0)?;

    // Location: a separate concept space
    let location = NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0)?;

    // City: contained within Location
    let city = NdarrayBox::new(array![0.1, 0.2, 0.1], array![0.5, 0.6, 0.4], 1.0)?;

    // Check containment relationships
    println!("Knowledge Graph Reasoning:");
    println!("==========================");

    // Animal contains Dog?
    let p_dog_in_animal = animal.containment_prob(&dog, 1.0)?;
    println!("P(Dog ⊆ Animal) = {:.3}", p_dog_in_animal);
    assert!(p_dog_in_animal > 0.9, "Dog should be contained in Animal");

    // Animal contains Cat?
    let p_cat_in_animal = animal.containment_prob(&cat, 1.0)?;
    println!("P(Cat ⊆ Animal) = {:.3}", p_cat_in_animal);
    assert!(p_cat_in_animal > 0.9, "Cat should be contained in Animal");

    // Dog and Cat overlap? (they're both animals but different species)
    let p_dog_cat_overlap = dog.overlap_prob(&cat, 1.0)?;
    println!("P(Dog ∩ Cat ≠ ∅) = {:.3}", p_dog_cat_overlap);
    // They might overlap slightly in the animal space, but not much

    // Location contains City?
    let p_city_in_location = location.containment_prob(&city, 1.0)?;
    println!("P(City ⊆ Location) = {:.3}", p_city_in_location);
    assert!(
        p_city_in_location > 0.9,
        "City should be contained in Location"
    );

    // Dog and City should not overlap (different concept spaces)
    let p_dog_city_overlap = dog.overlap_prob(&city, 1.0)?;
    println!("P(Dog ∩ City ≠ ∅) = {:.3}", p_dog_city_overlap);
    assert!(p_dog_city_overlap < 0.5, "Dog and City should not overlap");

    // Batch operations: create a collection and query it
    println!("\nBatch Operations:");
    println!("==================");

    // Create new boxes for the collection (since we moved ownership above)
    let animal2 = NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0)?;
    let dog2 = NdarrayBox::new(array![0.2, 0.3, 0.1], array![0.6, 0.7, 0.5], 1.0)?;
    let cat2 = NdarrayBox::new(array![0.4, 0.1, 0.6], array![0.8, 0.5, 0.9], 1.0)?;
    let location2 = NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0)?;
    let city2 = NdarrayBox::new(array![0.1, 0.2, 0.1], array![0.5, 0.6, 0.4], 1.0)?;

    let entities: BoxCollection<NdarrayBox> = vec![
        animal2.clone(),
        dog2.clone(),
        cat2.clone(),
        location2.clone(),
        city2.clone(),
    ]
    .into();

    // Find all entities that contain "Dog"
    let containing_dog = entities.containing_boxes(&dog2, 0.5, 1.0)?;
    println!("Entities containing Dog: {:?}", containing_dog);

    // Find all entities contained in "Animal"
    let animal_box = entities.get(0)?;
    let contained_in_animal = entities.contained_boxes(animal_box, 0.5, 1.0)?;
    println!("Entities contained in Animal: {:?}", contained_in_animal);

    // Compute full containment matrix
    let matrix = entities.containment_matrix(1.0)?;
    println!("\nContainment Matrix (row contains column):");
    println!("  Animal  Dog  Cat  Location  City");
    for (i, row) in matrix.iter().enumerate() {
        let names = ["Animal", "Dog", "Cat", "Location", "City"];
        print!("{}", names[i]);
        for prob in row {
            print!("  {:.2}", prob);
        }
        println!();
    }

    // Demonstrate new geometric operations
    println!("\nGeometric Operations:");
    println!("====================");

    // Union: combine Dog and Cat into a single box
    let dog_cat_union = dog2.union(&cat2)?;
    println!("Union of Dog and Cat boxes:");
    println!(
        "  Min: [{:.2}, {:.2}, {:.2}]",
        dog_cat_union.min()[0],
        dog_cat_union.min()[1],
        dog_cat_union.min()[2]
    );
    println!(
        "  Max: [{:.2}, {:.2}, {:.2}]",
        dog_cat_union.max()[0],
        dog_cat_union.max()[1],
        dog_cat_union.max()[2]
    );

    // Center: get center point of Animal box
    let animal_center = animal2.center()?;
    println!(
        "Center of Animal box: [{:.2}, {:.2}, {:.2}]",
        animal_center[0], animal_center[1], animal_center[2]
    );

    // Distance: measure distance between Dog and City (should be large)
    let dog_city_dist = dog2.distance(&city2)?;
    println!("Distance between Dog and City: {:.3}", dog_city_dist);

    // Overlap matrix: find which entities overlap
    let overlap_matrix = entities.overlap_matrix(1.0)?;
    println!("\nOverlap Matrix (row overlaps with column):");
    println!("  Animal  Dog  Cat  Location  City");
    for (i, row) in overlap_matrix.iter().enumerate() {
        let names = ["Animal", "Dog", "Cat", "Location", "City"];
        print!("{}", names[i]);
        for prob in row {
            print!("  {:.2}", prob);
        }
        println!();
    }

    // Find overlapping entities
    let overlapping_dog = entities.overlapping_boxes(&dog2, 0.1, 1.0)?;
    println!("\nEntities overlapping with Dog: {:?}", overlapping_dog);

    // Find nearest entities
    let nearest_to_dog = entities.nearest_boxes(&dog2, 3)?;
    println!("3 nearest entities to Dog: {:?}", nearest_to_dog);

    // Compute bounding box of all entities
    let bbox = entities.bounding_box()?;
    println!("\nBounding box of all entities:");
    println!(
        "  Min: [{:.2}, {:.2}, {:.2}]",
        bbox.min()[0],
        bbox.min()[1],
        bbox.min()[2]
    );
    println!(
        "  Max: [{:.2}, {:.2}, {:.2}]",
        bbox.max()[0],
        bbox.max()[1],
        bbox.max()[2]
    );
    let bbox_vol = bbox.volume(1.0)?;
    println!("  Volume: {:.3}", bbox_vol);

    Ok(())
}
