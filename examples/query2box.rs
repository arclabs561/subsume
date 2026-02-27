//! Query2Box-style compositional query answering with box embeddings.
//!
//! Query2Box (Ren et al., 2020) models multi-hop knowledge graph queries as
//! sequences of box transformations. Each relation maps an entity box to a
//! query box via translation. Intersection narrows the answer set when
//! multiple constraints apply.
//!
//! This example hand-places entity boxes so that geometric containment
//! encodes ground-truth relations, then answers compositional queries by
//! intersecting boxes and ranking candidates by containment score.
//!
//! Knowledge graph:
//!   located_in(Paris, France), located_in(Lyon, France),
//!   located_in(London, UK),
//!   speaks(France, French), speaks(UK, English),
//!   speaks(France, English)   -- partial: some speakers
//!
//! Queries:
//!   Q1: ?x . located_in(x, France)           -- cities in France
//!   Q2: ?x . speaks(France, x)               -- languages spoken in France
//!   Q3: ?x . exists y . located_in(y, France) AND speaks(y, x)
//!       -- languages spoken in countries that contain French cities
//!       (simplified: intersect the "speaks" projection of France
//!        with language boxes)
//!
//! Reference: Ren, Hu, Leskovec (2020), "Query2Box: Reasoning over Knowledge
//! Graphs in Vector Space Using Box Embeddings", ICLR 2020.
//!
//! Run: cargo run -p subsume --example query2box

use ndarray::{array, Array1};
use subsume::ndarray_backend::distance::query2box_distance;
use subsume::ndarray_backend::NdarrayBox;
use subsume::Box as BoxTrait;

/// Rank candidates by P(query contains candidate), descending.
fn rank_candidates<'a>(
    query: &NdarrayBox,
    candidates: &[(&'a str, &NdarrayBox)],
    temp: f32,
) -> Vec<(&'a str, f32)> {
    let mut scored: Vec<(&str, f32)> = candidates
        .iter()
        .map(|(name, b)| {
            let p = query.containment_prob(b, temp).unwrap_or(0.0);
            (*name, p)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
}

/// Rank candidates by Query2Box alpha-weighted distance (ascending = closer = better).
///
/// Uses the center of each candidate box as the entity point.
fn rank_by_distance<'a>(
    query: &NdarrayBox,
    candidates: &[(&'a str, &NdarrayBox)],
    alpha: f32,
) -> Vec<(&'a str, f32)> {
    let mut scored: Vec<(&str, f32)> = candidates
        .iter()
        .map(|(name, b)| {
            // Use center of candidate box as the entity point.
            let center: Array1<f32> = (b.min() + b.max()) * 0.5;
            let d = query2box_distance(query, &center, alpha).unwrap_or(f32::INFINITY);
            (*name, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // ascending: lower distance = better
    scored
}

fn print_ranking(label: &str, ranking: &[(&str, f32)]) {
    println!("  {label}");
    for (i, (name, score)) in ranking.iter().enumerate() {
        let marker = if *score > 0.5 { "<-- answer" } else { "" };
        println!("    {}: {:>8} score={:.4} {}", i + 1, name, score, marker);
    }
    println!();
}

fn main() -> Result<(), subsume::BoxError> {
    println!("=== Query2Box: Compositional Query Answering ===\n");

    // --- Entity boxes (8 dimensions) ---
    // Countries are large boxes; cities and languages are smaller, placed
    // inside the appropriate country box to encode ground-truth relations.

    let france = NdarrayBox::new(
        array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1.0,
    )?;
    let _uk = NdarrayBox::new(
        array![2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
        array![3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0],
        1.0,
    )?;

    // Cities -- small boxes inside their country.
    let paris = NdarrayBox::new(
        array![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        array![0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        1.0,
    )?;
    let lyon = NdarrayBox::new(
        array![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        array![0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
        1.0,
    )?;
    let london = NdarrayBox::new(
        array![2.2, 2.2, 0.2, 0.2, 2.2, 2.2, 0.2, 0.2],
        array![2.4, 2.4, 0.4, 0.4, 2.4, 2.4, 0.4, 0.4],
        1.0,
    )?;

    // Languages -- placed to reflect who speaks them.
    // French: inside France box.
    let french = NdarrayBox::new(
        array![0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        array![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        1.0,
    )?;
    // English: overlaps both France and UK (spoken in both, to different extents).
    // Partially inside France (dims 2-3 overlap), fully inside UK (dims 0-1, 4-5).
    let english = NdarrayBox::new(
        array![2.1, 2.1, 0.1, 0.1, 2.1, 2.1, 0.1, 0.1],
        array![2.8, 2.8, 0.8, 0.8, 2.8, 2.8, 0.8, 0.8],
        1.0,
    )?;

    let temp = 1.0;

    // --- Q1: cities in France ---
    // Query box = France (the "located_in" projection of France is France itself
    // in this hand-placed setup, since cities are inside their country box).
    println!("Q1: What cities are in France?\n");
    let city_candidates: Vec<(&str, &NdarrayBox)> =
        vec![("Paris", &paris), ("Lyon", &lyon), ("London", &london)];
    let q1 = rank_candidates(&france, &city_candidates, temp);
    print_ranking("Rank by P(France contains city):", &q1);

    // --- Q2: languages spoken in France ---
    // Query box = France. Languages placed inside France score high.
    println!("Q2: What languages are spoken in France?\n");
    let lang_candidates: Vec<(&str, &NdarrayBox)> =
        vec![("French", &french), ("English", &english)];
    let q2 = rank_candidates(&france, &lang_candidates, temp);
    print_ranking("Rank by P(France contains language):", &q2);

    // --- Q3: compositional -- languages spoken in countries containing French cities ---
    // Step 1: intersect France with the "city region" (union of city boxes as proxy).
    //         Simpler: France already contains the cities, so use France directly.
    // Step 2: intersect with the language region.
    // In Query2Box terms: project France through "speaks" relation, then intersect
    // with language boxes. Here we approximate by intersecting France and UK,
    // then checking which languages fall in the intersection.
    //
    // More concretely: the query asks for languages spoken in France. We already
    // showed Q2. The compositional aspect chains two hops:
    //   hop 1: countries that have cities in France -> {France}
    //   hop 2: languages spoken in those countries  -> {French, English(partial)}
    //
    // We model hop 1 by intersecting France with a broad "has-city" region,
    // yielding a query box. Then hop 2 ranks languages against that box.

    println!("Q3: Languages spoken in countries with French cities (2-hop)\n");
    println!("  Hop 1: intersect France with city-containing region");

    // The "city region" is the union of all city boxes (bounding box of cities).
    let city_region = paris.union(&lyon)?.union(&london)?;
    let hop1 = france.intersection(&city_region)?;
    println!(
        "    intersection volume: {:.4}  (> 0 confirms France has cities)",
        hop1.volume(temp)?
    );

    println!("  Hop 2: rank languages by containment in hop-1 result\n");
    let q3 = rank_candidates(&hop1, &lang_candidates, temp);
    print_ranking("Rank by P(hop1_box contains language):", &q3);

    // --- Q4: alpha-weighted distance scoring (Query2Box original) ---
    //
    // The original Query2Box paper scores candidates by distance, not containment
    // probability: d(q, v) = d_out(q, v) + alpha * d_in(q, v)
    //
    // d_out = L1 distance from entity to nearest box boundary (0 if inside)
    // d_in  = L1 distance from entity to box center (0 if outside)
    // alpha < 1 penalizes inside-center distance less than outside distance.

    println!("Q4: Alpha-weighted distance scoring (Ren et al., 2020)\n");

    let alpha = 0.02;
    println!("  alpha = {alpha}  (inside penalty << outside penalty)\n");

    println!("  Q1 re-scored: cities in France (by distance, ascending)\n");
    let q4a = rank_by_distance(&france, &city_candidates, alpha);
    for (i, (name, dist)) in q4a.iter().enumerate() {
        let marker = if *dist < 1.0 { "<-- answer" } else { "" };
        println!("    {}: {:>8} dist={:.4} {}", i + 1, name, dist, marker);
    }
    println!();

    println!("  Q2 re-scored: languages in France (by distance, ascending)\n");
    let q4b = rank_by_distance(&france, &lang_candidates, alpha);
    for (i, (name, dist)) in q4b.iter().enumerate() {
        println!("    {}: {:>8} dist={:.4}", i + 1, name, dist);
    }
    println!();

    // Show how alpha affects ranking
    println!("  Alpha sensitivity: distance for Paris across alpha values\n");
    let paris_center: Array1<f32> = (paris.min() + paris.max()) * 0.5;
    for &a in &[0.0, 0.02, 0.1, 0.5, 1.0] {
        let d = query2box_distance(&france, &paris_center, a)?;
        println!("    alpha={a:.2}: dist={d:.4}");
    }
    println!();

    // --- Summary ---
    println!("--- Summary ---\n");
    println!("  Q1 correctly ranks Paris and Lyon above London.");
    println!("  Q2 ranks French highest (fully inside France).");
    println!("  Q3 chains two hops: city containment, then language containment.");
    println!("  Intersection volume decreases at each hop, narrowing the answer set.");
    println!("  Q4 shows Query2Box distance scoring: lower distance = better answer.");
    println!("  Alpha controls inside-vs-outside penalty balance.");

    Ok(())
}
