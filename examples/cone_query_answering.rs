//! Cone query answering: FOL queries with conjunction, disjunction, negation.
//!
//! Demonstrates the cone query algebra for multi-hop reasoning over a small
//! knowledge graph. Cones (unlike boxes) are closed under complement, enabling
//! negation queries like "animals that are NOT cats".
//!
//! This example shows:
//! 1. Building a toy entity hierarchy as cones
//! 2. Composing FOL queries (AND, OR, NOT, projection)
//! 3. Ranking entities against composed queries
//! 4. Fuzzy bridge: converting cone distances to [0,1] scores for t-norm composition
//! 5. Multi-hop projection queries
//!
//! References:
//! - Zhang et al. (NeurIPS 2021), "ConE: Cone Embeddings for Multi-Hop Reasoning"
//! - Chen et al. (AAAI 2022), "FuzzQE: Fuzzy Logic Based Logical Query Answering"
//!
//! Run: cargo run -p subsume --example cone_query_answering
//!
//! Related examples:
//! - `cone_training`: training cone embeddings on a taxonomy
//! - `fuzzy_query`: fuzzy query answering with t-norms (scalar domain)
//! - `query2box`: compositional query answering with box intersection

use ndarray::array;
use subsume::cone_query::{cone_containment_score, ConeQuery};
use subsume::fuzzy::TNorm;
use subsume::ndarray_backend::NdarrayCone;

/// Build a small hierarchy of cone embeddings.
/// Wider aperture = more general concept. Axes encode semantic position.
fn build_entities() -> Vec<(&'static str, NdarrayCone)> {
    vec![
        // Broad categories (wide apertures)
        (
            "entity",
            NdarrayCone::new(array![0.0, 0.0, 0.0, 0.0], array![2.5, 2.5, 2.5, 2.5]).unwrap(),
        ),
        (
            "animal",
            NdarrayCone::new(array![0.3, 0.2, 0.1, 0.0], array![1.8, 1.8, 1.8, 1.8]).unwrap(),
        ),
        (
            "vehicle",
            NdarrayCone::new(array![-1.5, -1.0, -0.5, 0.0], array![1.5, 1.5, 1.5, 1.5]).unwrap(),
        ),
        // Animals (narrower, positioned near "animal" axis)
        (
            "dog",
            NdarrayCone::new(array![0.5, 0.4, 0.3, 0.1], array![0.4, 0.4, 0.4, 0.4]).unwrap(),
        ),
        (
            "cat",
            NdarrayCone::new(array![0.2, 0.6, 0.1, -0.1], array![0.4, 0.4, 0.4, 0.4]).unwrap(),
        ),
        (
            "fish",
            NdarrayCone::new(array![0.1, -0.3, 0.5, 0.2], array![0.4, 0.4, 0.4, 0.4]).unwrap(),
        ),
        (
            "bird",
            NdarrayCone::new(array![0.4, 0.1, -0.2, 0.3], array![0.4, 0.4, 0.4, 0.4]).unwrap(),
        ),
        // Vehicles (positioned far from animals)
        (
            "car",
            NdarrayCone::new(array![-1.3, -0.8, -0.4, 0.1], array![0.3, 0.3, 0.3, 0.3]).unwrap(),
        ),
        (
            "truck",
            NdarrayCone::new(array![-1.7, -1.2, -0.6, -0.1], array![0.3, 0.3, 0.3, 0.3])
                .unwrap(),
        ),
    ]
}

fn main() {
    println!("=== Cone Query Answering ===\n");

    let entities = build_entities();
    let cen = 0.02; // ConE inside-distance weight
    let gamma = 10.0; // Sigmoid sharpness for fuzzy bridge

    // Helper: find entity cone by name
    let cone_of = |name: &str| -> NdarrayCone {
        entities
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap()
            .1
            .clone()
    };

    // Helper: entity cones as points (zero aperture) for scoring
    let entity_points: Vec<(&str, NdarrayCone)> = entities
        .iter()
        .map(|(name, c)| (*name, NdarrayCone::point(c.axes().clone())))
        .collect();

    // --- Query 1: simple atom (what is most like "animal"?) ---

    println!("Query 1: atom(animal) -- rank entities by distance to 'animal'\n");
    let q1 = ConeQuery::Atom(cone_of("animal"));
    print_rankings(&q1, &entity_points, cen, gamma);

    // --- Query 2: conjunction (animal AND NOT cat) ---

    println!("\nQuery 2: animal AND NOT cat -- animals excluding cats\n");
    let q2 = ConeQuery::Intersection(vec![
        ConeQuery::Atom(cone_of("animal")),
        ConeQuery::Complement(Box::new(ConeQuery::Atom(cone_of("cat")))),
    ]);
    print_rankings(&q2, &entity_points, cen, gamma);

    // --- Query 3: disjunction (dog OR car) ---

    println!("\nQuery 3: dog OR car -- entities matching either concept\n");
    let q3 = ConeQuery::Union(vec![
        ConeQuery::Atom(cone_of("dog")),
        ConeQuery::Atom(cone_of("car")),
    ]);
    print_rankings(&q3, &entity_points, cen, gamma);

    // --- Query 4: negation (NOT vehicle) ---

    println!("\nQuery 4: NOT vehicle -- everything except vehicles\n");
    let q4 = ConeQuery::Complement(Box::new(ConeQuery::Atom(cone_of("vehicle"))));
    print_rankings(&q4, &entity_points, cen, gamma);

    // --- Query 5: multi-hop projection ---
    // Simulate: start from "dog", project through "is_similar_to" relation

    println!("\nQuery 5: project(dog, is_similar_to) -- similar entities\n");
    let q5 = ConeQuery::Projection {
        query: Box::new(ConeQuery::Atom(cone_of("dog"))),
        // Relation: slight axis shift + widen apertures (more general)
        relation_axes: array![-0.2, 0.1, -0.1, 0.0],
        relation_apertures: array![0.3, 0.3, 0.3, 0.3],
    };
    let result_cone = q5.evaluate().unwrap();
    println!(
        "  Projected cone aperture (mean): {:.3} (wider = more general search)",
        result_cone.apertures().mean().unwrap()
    );
    print_rankings(&q5, &entity_points, cen, gamma);

    // --- Query 6: composed query (project through relation, then intersect) ---

    println!("\nQuery 6: project(entity, has_legs) AND NOT fish -- legged non-fish\n");
    let q6 = ConeQuery::Intersection(vec![
        ConeQuery::Projection {
            query: Box::new(ConeQuery::Atom(cone_of("entity"))),
            relation_axes: array![0.3, 0.2, 0.1, 0.0],
            relation_apertures: array![-0.5, -0.5, -0.5, -0.5], // Narrow: restrict
        },
        ConeQuery::Complement(Box::new(ConeQuery::Atom(cone_of("fish")))),
    ]);
    print_rankings(&q6, &entity_points, cen, gamma);

    // --- Fuzzy bridge: compose cone scores with t-norms ---

    println!("\n--- Fuzzy bridge: cone score + t-norm composition ---\n");
    println!("  Combining 'closeness to animal' AND 'closeness to dog' via Product t-norm:\n");
    let animal_cone = cone_of("animal");
    let dog_cone = cone_of("dog");

    println!("  {:>10} | {:>8} {:>8} | {:>8}", "entity", "animal", "dog", "Product");
    println!("  {:-<10}-+-{:-<8}-{:-<8}-+-{:-<8}", "", "", "", "");

    for (name, point) in &entity_points {
        let s_animal = cone_containment_score(&animal_cone, point, cen, gamma).unwrap();
        let s_dog = cone_containment_score(&dog_cone, point, cen, gamma).unwrap();
        let combined = TNorm::Product.apply(s_animal, s_dog);
        println!(
            "  {:>10} | {:>8.4} {:>8.4} | {:>8.4}",
            name, s_animal, s_dog, combined
        );
    }

    // --- Complement closure demonstration ---

    println!("\n--- Complement closure: cones are closed under negation ---\n");
    let c = cone_of("cat");
    let not_c = c.complement();
    let not_not_c = not_c.complement();
    println!("  cat apertures:          {:?}", c.apertures().as_slice().unwrap());
    println!("  NOT cat apertures:      {:?}", not_c.apertures().as_slice().unwrap());
    println!("  NOT NOT cat apertures:  {:?}", not_not_c.apertures().as_slice().unwrap());
    println!("  Double complement = identity: {}", {
        let diff: f32 = c.apertures().iter().zip(not_not_c.apertures().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        diff < 1e-4
    });

    println!("\n  This closure property is why cones can handle FOL negation");
    println!("  while boxes cannot (complement of a box is not a box).");
}

fn print_rankings(query: &ConeQuery, entities: &[(&str, NdarrayCone)], cen: f32, gamma: f32) {
    let cones: Vec<NdarrayCone> = entities.iter().map(|(_, c)| c.clone()).collect();
    let ranked = query.rank_entities(&cones, cen).unwrap();

    println!(
        "  {:>4} {:>10} {:>10} {:>10}",
        "rank", "entity", "distance", "score"
    );
    println!("  {:-<4} {:-<10} {:-<10} {:-<10}", "", "", "", "");

    for (rank, &idx) in ranked.iter().enumerate() {
        let name = entities[idx].0;
        let dist = query.score_entity(&cones[idx], cen).unwrap();
        let score = 1.0 / (1.0 + (gamma * dist).exp());
        println!(
            "  {:>4} {:>10} {:>10.4} {:>10.4}",
            rank + 1,
            name,
            dist,
            score
        );
    }
}
