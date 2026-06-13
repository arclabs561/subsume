//! Writing code generic over the geometry, via the [`subsume::Region`] trait.
//!
//! subsume offers many region geometries (boxes, balls, ellipsoids, subspaces).
//! They share one contract: a region subsumes another to the degree it contains
//! it. The `Region` trait names that contract, so a retrieval/ranking routine
//! can be written ONCE and reused across geometries.
//!
//! Caveat the example also demonstrates: the score is monotone WITHIN a geometry
//! (more-contained ranks higher) but is NOT calibrated across geometries -- a
//! box's 1.0 and a ball's 0.5 are different scales. So rank within one geometry;
//! don't compare a ball's score against a box's.
//!
//! Run: `cargo run --example region_generic`

use subsume::ndarray_backend::NdarrayBox;
use subsume::{Ball, Region};

/// Generic over ANY region geometry: return the candidate most strongly
/// subsumed by `query`, with its score. Written once, used for boxes and balls.
fn best_subsumed<'a, R: Region>(query: &R, candidates: &'a [(&'a str, R)]) -> (&'a str, f32) {
    candidates
        .iter()
        .map(|(name, r)| (*name, query.subsumption_score(r).unwrap()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

fn main() {
    println!("Region trait: one ranking routine, many geometries\n");

    // --- Boxes (axis-aligned hyperrectangles) ---
    let query_box =
        NdarrayBox::new(ndarray::array![0.0, 0.0], ndarray::array![4.0, 4.0], 1.0).unwrap();
    let box_candidates = vec![
        (
            "inside   ",
            NdarrayBox::new(ndarray::array![1.0, 1.0], ndarray::array![3.0, 3.0], 1.0).unwrap(),
        ),
        (
            "straddle ",
            NdarrayBox::new(ndarray::array![3.0, 3.0], ndarray::array![6.0, 6.0], 1.0).unwrap(),
        ),
        (
            "disjoint ",
            NdarrayBox::new(ndarray::array![9.0, 9.0], ndarray::array![10.0, 10.0], 1.0).unwrap(),
        ),
    ];
    println!("query: box [0,4]^2 (dim {})", Region::dim(&query_box));
    for (name, b) in &box_candidates {
        println!(
            "  subsumes {name}: {:.4}",
            query_box.subsumption_score(b).unwrap()
        );
    }
    let (best, score) = best_subsumed(&query_box, &box_candidates);
    println!("  -> most subsumed: {} ({score:.4})\n", best.trim());

    // --- Balls (Euclidean) --- same `best_subsumed`, different geometry ---
    let query_ball = Ball::new(vec![0.0, 0.0], 3.0).unwrap();
    let ball_candidates = vec![
        ("inside   ", Ball::new(vec![0.0, 0.0], 0.5).unwrap()),
        ("straddle ", Ball::new(vec![2.8, 0.0], 1.0).unwrap()),
        ("disjoint ", Ball::new(vec![10.0, 0.0], 0.5).unwrap()),
    ];
    println!(
        "query: ball center (0,0) r=3 (dim {})",
        Region::dim(&query_ball)
    );
    for (name, b) in &ball_candidates {
        println!(
            "  subsumes {name}: {:.4}",
            query_ball.subsumption_score(b).unwrap()
        );
    }
    let (best, score) = best_subsumed(&query_ball, &ball_candidates);
    println!("  -> most subsumed: {} ({score:.4})", best.trim());

    println!(
        "\nNote: the box scores (self-containment ~1.0) and ball scores (sigmoid\n\
         of margin, ~0.5 at the boundary) live on different scales. Rank within a\n\
         geometry; don't compare a ball's number to a box's."
    );
}
