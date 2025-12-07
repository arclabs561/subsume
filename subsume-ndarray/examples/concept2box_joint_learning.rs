//! Example: Concept2Box-style joint learning of concept boxes and entity vectors.
//!
//! This example demonstrates the Concept2Box (2023) approach:
//! - Concepts are represented as box embeddings
//! - Entities are represented as vector embeddings
//! - Joint learning improves performance on two-view knowledge graphs
//!
//! Reference: Huang et al. (2023): "Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs"
//!
//! # Mathematical Foundations
//!
//! Concept2Box addresses the two-view nature of knowledge graphs: concepts (ontology view)
//! and entities (instance view). Boxes naturally represent hierarchical concept relationships
//! through containment, while vectors efficiently represent entity similarity. The joint
//! learning objective connects these views through distance metrics.
//!
//! For detailed mathematical foundations and modern applications, see:
//! - [`docs/typst-output/pdf/07-applications.pdf`](../../../docs/typst-output/pdf/07-applications.pdf) - Modern applications including Concept2Box
//! - [`docs/typst-output/pdf/subsumption.pdf`](../../../docs/typst-output/pdf/subsumption.pdf) - Why boxes for concepts
//! - [`docs/typst-output/pdf/gumbel-box-volume.pdf`](../../../docs/typst-output/pdf/gumbel-box-volume.pdf) - Volume as concept granularity

use ndarray::array;
use subsume_core::Box;
use subsume_ndarray::{distance, NdarrayBox};

fn main() -> Result<(), subsume_core::BoxError> {
    println!("Concept2Box: Joint Learning of Concepts (Boxes) and Entities (Vectors)");
    println!("========================================================================\n");

    // Create concept boxes (ontology view)
    let animal = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0)?;
    let mammal = NdarrayBox::new(array![2.0, 2.0], array![8.0, 8.0], 1.0)?;
    let bird = NdarrayBox::new(array![7.0, 7.0], array![9.0, 9.0], 1.0)?;
    let dog = NdarrayBox::new(array![3.0, 3.0], array![7.0, 7.0], 1.0)?;
    let cat = NdarrayBox::new(array![2.5, 2.5], array![4.5, 4.5], 1.0)?;

    // Create entity vectors (instance view)
    // In practice, these would be learned embeddings
    let entity_fido = array![4.0, 4.0]; // Instance of "dog"
    let entity_whiskers = array![3.5, 3.5]; // Instance of "cat"
    let entity_tweety = array![8.0, 8.0]; // Instance of "bird"

    println!("1. Concept Boxes (Ontology View)");
    println!("--------------------------------");
    println!("Concepts are represented as boxes with volumes indicating granularity:");
    println!("  Animal: volume = {:.2}", animal.volume(1.0)?);
    println!("  Mammal: volume = {:.2}", mammal.volume(1.0)?);
    println!("  Dog:    volume = {:.2}", dog.volume(1.0)?);
    println!("  Cat:    volume = {:.2}", cat.volume(1.0)?);
    println!("  Bird:   volume = {:.2}", bird.volume(1.0)?);
    println!();

    println!("2. Entity Vectors (Instance View)");
    println!("----------------------------------");
    println!("Entities are represented as vectors:");
    println!(
        "  Fido (dog instance):     [{:.1}, {:.1}]",
        entity_fido[0], entity_fido[1]
    );
    println!(
        "  Whiskers (cat instance): [{:.1}, {:.1}]",
        entity_whiskers[0], entity_whiskers[1]
    );
    println!(
        "  Tweety (bird instance): [{:.1}, {:.1}]",
        entity_tweety[0], entity_tweety[1]
    );
    println!();

    println!("3. Vector-to-Box Distance (Bridging Concepts and Entities)");
    println!("----------------------------------------------------------");

    // Compute distances from entities to concept boxes
    let dist_fido_animal = distance::vector_to_box_distance(&entity_fido, &animal)?;
    let dist_fido_mammal = distance::vector_to_box_distance(&entity_fido, &mammal)?;
    let dist_fido_dog = distance::vector_to_box_distance(&entity_fido, &dog)?;
    let dist_fido_cat = distance::vector_to_box_distance(&entity_fido, &cat)?;
    let dist_fido_bird = distance::vector_to_box_distance(&entity_fido, &bird)?;

    println!("Distances from 'Fido' (dog entity) to concepts:");
    println!(
        "  Animal: {:.4} (should be 0 - Fido is an animal)",
        dist_fido_animal
    );
    println!(
        "  Mammal: {:.4} (should be 0 - Fido is a mammal)",
        dist_fido_mammal
    );
    println!(
        "  Dog:    {:.4} (should be 0 - Fido is a dog)",
        dist_fido_dog
    );
    println!(
        "  Cat:    {:.4} (should be > 0 - Fido is not a cat)",
        dist_fido_cat
    );
    println!(
        "  Bird:   {:.4} (should be > 0 - Fido is not a bird)",
        dist_fido_bird
    );
    println!();

    let dist_whiskers_animal = distance::vector_to_box_distance(&entity_whiskers, &animal)?;
    let dist_whiskers_mammal = distance::vector_to_box_distance(&entity_whiskers, &mammal)?;
    let dist_whiskers_cat = distance::vector_to_box_distance(&entity_whiskers, &cat)?;

    println!("Distances from 'Whiskers' (cat entity) to concepts:");
    println!("  Animal: {:.4}", dist_whiskers_animal);
    println!("  Mammal: {:.4}", dist_whiskers_mammal);
    println!("  Cat:    {:.4}", dist_whiskers_cat);
    println!();

    println!("4. Joint Learning Objective");
    println!("---------------------------");
    println!("In Concept2Box, the joint learning objective combines:");
    println!("  - Concept containment: P(concept_B ⊆ concept_A)");
    println!("  - Entity-concept alignment: d(entity, concept_box)");
    println!("  - Entity-entity relationships: d(entity_1, entity_2)");
    println!();
    println!("This enables:");
    println!("  - Concepts to capture hierarchical structure (via boxes)");
    println!("  - Entities to capture fine-grained relationships (via vectors)");
    println!("  - Joint optimization improves both views");
    println!();

    println!("5. Box Volumes as Concept Granularity");
    println!("--------------------------------------");
    println!("In Concept2Box, box volumes are interpreted as concept granularity:");
    println!("  - Larger volumes = more general concepts (Animal)");
    println!("  - Smaller volumes = more specific concepts (Dog, Cat)");
    println!("  - This provides probabilistic semantics for concepts");
    println!();

    // Verify that volumes decrease with specificity
    assert!(animal.volume(1.0)? > mammal.volume(1.0)?);
    assert!(mammal.volume(1.0)? > dog.volume(1.0)?);
    assert!(mammal.volume(1.0)? > cat.volume(1.0)?);
    println!("✓ Volume hierarchy verified: Animal > Mammal > {{Dog, Cat}}");
    println!();

    println!("6. Performance Benefits of Joint Learning");
    println!("----------------------------------------");
    println!("Concept2Box paper shows that joint learning improves:");
    println!("  - Link prediction accuracy on DBpedia KG");
    println!("  - Concept classification performance");
    println!("  - Entity clustering quality");
    println!();
    println!("Key insight: Two-view KGs require dual representations:");
    println!("  - Ontology view (concepts) → Box embeddings");
    println!("  - Instance view (entities) → Vector embeddings");
    println!("  - Vector-to-box distance bridges the two views");

    Ok(())
}
