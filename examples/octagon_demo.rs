//! Octagon embeddings: strictly more expressive than boxes.
//!
//! Octagons add diagonal constraints (x_i + x_{i+1}, x_i - x_{i+1}) to the
//! standard axis-aligned bounds of boxes. This lets them model relational
//! structures (transitivity, composition) that boxes cannot.
//!
//! This example demonstrates:
//! 1. Constructing octagons with diagonal constraints
//! 2. Point containment (axis + diagonal checks)
//! 3. Intersection (componentwise tightening, closed under intersection)
//! 4. Volume comparison: octagons are tighter than their bounding box
//! 5. Soft containment and overlap probabilities
//!
//! Reference: Charpenay & Schockaert (IJCAI 2024, arXiv:2401.16270),
//! "Capturing Knowledge Graphs and Rules with Octagon Embeddings"
//!
//! Run: cargo run -p subsume --example octagon_demo
//!
//! Related examples:
//! - `containment_hierarchy`: box containment and overlap (no diagonal constraints)
//! - `query2box`: compositional query answering with box intersection
//! - `fuzzy_query`: fuzzy operators for query answering (complementary approach)

use ndarray::array;
use subsume::ndarray_backend::ndarray_octagon::{NdarrayDiagBounds, NdarrayOctagon};
use subsume::Octagon;

fn main() -> Result<(), subsume::OctagonError> {
    println!("=== Octagon Embeddings Demo ===\n");

    // --- Part 1: Octagon vs Box ---
    //
    // A 2D octagon with axis bounds [0,4]x[0,4] and diagonal constraints:
    //   2 <= x + y <= 6   (cuts the lower-left and upper-right corners)
    //   -2 <= x - y <= 2  (cuts the upper-left and lower-right corners)
    //
    // This produces an 8-sided polygon inside the [0,4]x[0,4] square.

    let oct = NdarrayOctagon::new(
        array![0.0, 0.0],
        array![4.0, 4.0],
        vec![NdarrayDiagBounds {
            sum_min: 2.0,
            sum_max: 6.0,
            diff_min: -2.0,
            diff_max: 2.0,
        }],
    )?;

    let box_equiv = NdarrayOctagon::from_box_bounds(array![0.0, 0.0], array![4.0, 4.0])?;

    let oct_vol = oct.volume()?;
    let box_vol = box_equiv.volume()?;
    println!("Part 1: Octagon vs Box\n");
    println!("  Bounding box [0,4]x[0,4] volume: {box_vol:.2}");
    println!("  Octagon volume (with diagonal cuts): {oct_vol:.2}");
    println!(
        "  Ratio: {:.1}% -- the diagonal constraints remove {:.1}% of the box area\n",
        100.0 * oct_vol / box_vol,
        100.0 * (1.0 - oct_vol / box_vol)
    );

    // --- Part 2: Point containment ---
    //
    // The center (2,2) is inside. The corner (0.1, 0.1) is inside the box
    // but violates the diagonal constraint x+y >= 2.

    println!("Part 2: Point containment\n");
    let points = [
        ([2.0, 2.0], "center"),
        (
            [0.1, 0.1],
            "corner (0.1, 0.1) -- inside box, outside octagon",
        ),
        ([1.0, 1.5], "off-center (1.0, 1.5)"),
        ([3.5, 3.5], "near corner (3.5, 3.5) -- x+y=7 > 6"),
        ([0.0, 2.0], "edge (0.0, 2.0) -- x+y=2, x-y=-2"),
    ];

    for (pt, desc) in &points {
        let inside = oct.contains(pt)?;
        println!(
            "  ({:.1}, {:.1}) {}: {}",
            pt[0],
            pt[1],
            desc,
            if inside { "INSIDE" } else { "outside" }
        );
    }
    println!();

    // --- Part 3: Intersection ---
    //
    // Intersection of two octagons is always an octagon (closure under intersection).

    let oct_a = NdarrayOctagon::new(
        array![0.0, 0.0],
        array![3.0, 3.0],
        vec![NdarrayDiagBounds {
            sum_min: 1.0,
            sum_max: 5.0,
            diff_min: -2.0,
            diff_max: 2.0,
        }],
    )?;

    let oct_b = NdarrayOctagon::new(
        array![1.0, 1.0],
        array![4.0, 4.0],
        vec![NdarrayDiagBounds {
            sum_min: 3.0,
            sum_max: 7.0,
            diff_min: -1.0,
            diff_max: 1.0,
        }],
    )?;

    let inter = oct_a.intersection(&oct_b)?;
    let vol_a = oct_a.volume()?;
    let vol_b = oct_b.volume()?;
    let vol_inter = inter.volume()?;

    println!("Part 3: Intersection (closure property)\n");
    println!("  Octagon A: axis [0,3]x[0,3], volume={vol_a:.2}");
    println!("  Octagon B: axis [1,4]x[1,4], volume={vol_b:.2}");
    println!(
        "  Intersection: axis [{:.0},{:.0}]x[{:.0},{:.0}], volume={vol_inter:.2}",
        inter.axis_min()[0],
        inter.axis_max()[0],
        inter.axis_min()[1],
        inter.axis_max()[1]
    );
    println!(
        "  Volume ratio: {:.1}% of A, {:.1}% of B\n",
        100.0 * vol_inter / vol_a,
        100.0 * vol_inter / vol_b
    );

    // --- Part 4: Soft containment and overlap ---

    println!("Part 4: Soft containment and overlap probabilities\n");
    let wide = NdarrayOctagon::new(
        array![0.0, 0.0],
        array![6.0, 6.0],
        vec![NdarrayDiagBounds {
            sum_min: 1.0,
            sum_max: 11.0,
            diff_min: -5.0,
            diff_max: 5.0,
        }],
    )?;
    let narrow = NdarrayOctagon::new(
        array![1.0, 1.0],
        array![5.0, 5.0],
        vec![NdarrayDiagBounds {
            sum_min: 3.0,
            sum_max: 9.0,
            diff_min: -3.0,
            diff_max: 3.0,
        }],
    )?;

    let temp = 0.5;
    let cont_wn = wide.containment_prob(&narrow, temp)?;
    let cont_nw = narrow.containment_prob(&wide, temp)?;
    let overlap = wide.overlap_prob(&narrow, temp)?;

    println!("  P(wide contains narrow) = {cont_wn:.4}  (should be high)");
    println!("  P(narrow contains wide) = {cont_nw:.4}  (should be low)");
    println!("  P(overlap)              = {overlap:.4}  (should be high)\n");

    // --- Part 5: Bounding box conversion ---

    println!("Part 5: Bounding box (drop diagonal constraints)\n");
    let (bb_min, bb_max) = oct.to_bounding_box_bounds();
    println!(
        "  Original octagon: axis [{:.0},{:.0}]x[{:.0},{:.0}] with diagonal cuts",
        oct.axis_min()[0],
        oct.axis_max()[0],
        oct.axis_min()[1],
        oct.axis_max()[1]
    );
    println!(
        "  Bounding box:     [{:.0},{:.0}]x[{:.0},{:.0}] (outer approximation)\n",
        bb_min[0], bb_max[0], bb_min[1], bb_max[1]
    );

    println!("--- Summary ---\n");
    println!("  Octagons are strictly more expressive than boxes:");
    println!("  - Any box is an octagon with vacuous diagonal constraints");
    println!("  - Diagonal cuts remove unreachable corners, improving fit");
    println!("  - Closed under intersection: composing relations stays in the octagon domain");
    println!("  - O(d) storage: 2d axis bounds + 4(d-1) diagonal bounds");

    Ok(())
}
