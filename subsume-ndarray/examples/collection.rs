//! Example demonstrating box embedding collections for batch operations.

use ndarray::array;
use subsume_core::{BoxCollection, BoxEmbedding};
use subsume_ndarray::NdarrayBox;

fn main() -> Result<(), subsume_core::BoxError> {
    // Create a collection of boxes representing a hierarchy
    let mut collection: BoxCollection<NdarrayBox> = BoxCollection::new();

    // Animal (top level)
    collection.push(NdarrayBox::new(
        array![0.0, 0.0, 0.0],
        array![1.0, 1.0, 1.0],
        1.0,
    )?);

    // Mammal (contained in Animal)
    collection.push(NdarrayBox::new(
        array![0.1, 0.1, 0.1],
        array![0.6, 0.6, 0.6],
        1.0,
    )?);

    // Dog (contained in Mammal)
    collection.push(NdarrayBox::new(
        array![0.2, 0.2, 0.2],
        array![0.5, 0.5, 0.5],
        1.0,
    )?);

    // Bird (contained in Animal, but not Mammal)
    collection.push(NdarrayBox::new(
        array![0.7, 0.1, 0.1],
        array![0.9, 0.6, 0.6],
        1.0,
    )?);

    println!("Collection size: {}", collection.len());

    // Compute pairwise containment matrix
    let matrix = collection.containment_matrix(1.0)?;
    println!("\nContainment matrix (row i contains column j):");
    for (i, row) in matrix.iter().enumerate() {
        print!("Box {}: ", i);
        for prob in row {
            print!("{:.2} ", prob);
        }
        println!();
    }

    // Find boxes that contain "Dog"
    let dog = collection.get(2)?;
    let containing = collection.containing_boxes(dog, 0.5, 1.0)?;
    println!("\nBoxes containing Dog (index 2): {:?}", containing);
    // Should include Animal (0) and Mammal (1)

    // Find boxes contained by "Animal"
    let animal = collection.get(0)?;
    let contained = collection.contained_boxes(animal, 0.5, 1.0)?;
    println!("Boxes contained by Animal (index 0): {:?}", contained);
    // Should include Mammal (1), Dog (2), and Bird (3)

    Ok(())
}
