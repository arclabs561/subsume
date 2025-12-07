//! Example: Demonstrating geometric operations on box embeddings.
//!
//! This example shows how to use the enriched geometric methods:
//! - union: Combine boxes
//! - center: Get center points
//! - distance: Measure distances between boxes
//! - overlap_matrix: Batch overlap queries
//! - nearest_boxes: Similarity search
//! - bounding_box: Aggregate operations
//!
//! # Mathematical Foundations
//!
//! Box embeddings represent concepts as geometric regions (axis-aligned hyperrectangles).
//! The fundamental operation is **containment**: if box A contains box B, then A subsumes B.
//! This geometric relationship directly models logical subsumption from formal logic.
//!
//! For detailed mathematical foundations, see:
//! - [`docs/typst-output/pdf/subsumption.pdf`](../../../docs/typst-output/pdf/subsumption.pdf) - Geometric containment as logical subsumption
//! - [`docs/typst-output/pdf/gumbel-box-volume.pdf`](../../../docs/typst-output/pdf/gumbel-box-volume.pdf) - Volume calculations and their meaning
//! - [`docs/typst-output/pdf/containment-probability.pdf`](../../../docs/typst-output/pdf/containment-probability.pdf) - Probabilistic containment

use ndarray::array;
use subsume_core::{Box, BoxCollection, BoxEmbedding};
use subsume_ndarray::NdarrayBox;

fn main() -> Result<(), subsume_core::BoxError> {
    println!("Geometric Operations on Box Embeddings");
    println!("======================================\n");

    // Create a collection of boxes representing different concepts
    let mut collection = BoxCollection::new();

    // Create boxes in a 2D space for visualization
    let box_a = NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0)?;
    collection.push(box_a.clone());

    let box_b = NdarrayBox::new(array![0.5, 0.5], array![1.5, 1.5], 1.0)?;
    collection.push(box_b.clone());

    let box_c = NdarrayBox::new(array![2.0, 2.0], array![3.0, 3.0], 1.0)?;
    collection.push(box_c.clone());

    let box_d = NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0)?;
    collection.push(box_d.clone());

    // 1. Union operation
    println!("1. Union Operation");
    println!("------------------");
    let union_ab = box_a.union(&box_b)?;
    println!("Union of Box A [0,0]→[1,1] and Box B [0.5,0.5]→[1.5,1.5]:");
    println!(
        "  Result: [{:.1},{:.1}]→[{:.1},{:.1}]",
        union_ab.min()[0],
        union_ab.min()[1],
        union_ab.max()[0],
        union_ab.max()[1]
    );
    println!("  Volume: {:.3}\n", union_ab.volume(1.0)?);

    // 2. Center operation
    println!("2. Center Operation");
    println!("--------------------");
    for (i, box_) in [&box_a, &box_b, &box_c].iter().enumerate() {
        let center = box_.center()?;
        println!(
            "Box {} center: [{:.2}, {:.2}]",
            ['A', 'B', 'C'][i],
            center[0],
            center[1]
        );
    }
    println!();

    // 3. Distance operation
    println!("3. Distance Operation");
    println!("---------------------");
    let dist_ab = box_a.distance(&box_b)?;
    let dist_ac = box_a.distance(&box_c)?;
    let dist_bc = box_b.distance(&box_c)?;

    println!(
        "Distance A→B: {:.3} (overlapping: {})",
        dist_ab,
        dist_ab == 0.0
    );
    println!("Distance A→C: {:.3}", dist_ac);
    println!("Distance B→C: {:.3}\n", dist_bc);

    // 4. Overlap matrix
    println!("4. Overlap Matrix");
    println!("-----------------");
    let overlap_matrix = collection.overlap_matrix(1.0)?;
    println!("Pairwise overlap probabilities:");
    for (i, row) in overlap_matrix.iter().enumerate() {
        print!("Box {}: ", i);
        for prob in row {
            print!("{:.2} ", prob);
        }
        println!();
    }
    println!();

    // 5. Overlapping boxes query
    println!("5. Overlapping Boxes Query");
    println!("--------------------------");
    let query = NdarrayBox::new(array![0.3, 0.3], array![0.7, 0.7], 1.0)?;

    let overlapping = collection.overlapping_boxes(&query, 0.1, 1.0)?;
    println!(
        "Boxes overlapping with query [0.3,0.3]→[0.7,0.7]: {:?}",
        overlapping
    );
    println!();

    // 6. Nearest boxes (k-NN search)
    println!("6. Nearest Boxes (k-NN)");
    println!("------------------------");
    let nearest = collection.nearest_boxes(&query, 3)?;
    println!("3 nearest boxes to query: {:?}", nearest);

    // Show distances
    for &idx in &nearest {
        let box_i = collection.get(idx)?;
        let dist = query.distance(box_i)?;
        println!("  Box {}: distance = {:.3}", idx, dist);
    }
    println!();

    // 7. Bounding box
    println!("7. Bounding Box");
    println!("---------------");
    let bbox = collection.bounding_box()?;
    println!("Bounding box of all {} boxes:", collection.len());
    println!("  Min: [{:.2}, {:.2}]", bbox.min()[0], bbox.min()[1]);
    println!("  Max: [{:.2}, {:.2}]", bbox.max()[0], bbox.max()[1]);
    println!("  Volume: {:.3}", bbox.volume(1.0)?);

    // Verify bounding box contains all boxes
    println!("\nVerification: Bounding box contains all boxes?");
    for i in 0..collection.len() {
        let box_i = collection.get(i)?;
        let containment = bbox.containment_prob(box_i, 1.0)?;
        println!(
            "  Box {}: {:.3} {}",
            i,
            containment,
            if containment > 0.99 { "✓" } else { "✗" }
        );
    }

    Ok(())
}
