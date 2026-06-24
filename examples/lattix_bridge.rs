//! Build a subsume `Dataset` from a `lattix` `KnowledgeGraph`.
//!
//! `subsume::lattix_bridge::kg_to_dataset` converts a lattix knowledge graph
//! (RDF-style subject/predicate/object triples) into the `Triple`/`Dataset`
//! format subsume trains box and cone embeddings on. Here the graph is built in
//! memory; in practice it comes from `KnowledgeGraph::from_ntriples_file` over
//! an ontology such as the Gene Ontology or SNOMED-CT (lattix also reads
//! Turtle, N-Quads, CSV, and JSON-LD).
//!
//! Run: `cargo run --example lattix_bridge --features kge`

use lattix::{KnowledgeGraph, Triple};
use std::collections::BTreeSet;
use subsume::lattix_bridge::kg_to_dataset;

fn main() {
    // A small is-a taxonomy as RDF triples.
    let mut kg = KnowledgeGraph::new();
    for (s, p, o) in [
        ("dog", "is_a", "mammal"),
        ("cat", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
        ("animal", "is_a", "living_thing"),
    ] {
        kg.add_triple(Triple::new(s, p, o));
    }

    let dataset = kg_to_dataset(&kg);
    let kg_triples = kg.triples().count();

    println!("lattix KnowledgeGraph -> subsume Dataset");
    println!("  kg triples: {kg_triples}");
    println!("  dataset triples: {}", dataset.train.len());
    for t in &dataset.train {
        println!("    {} --{}--> {}", t.head, t.relation, t.tail);
    }

    // Every KG triple becomes exactly one Dataset training triple.
    assert_eq!(
        dataset.train.len(),
        kg_triples,
        "every KG triple should become one Dataset triple"
    );
    // The specific is-a triple survives the bridge unchanged.
    assert!(
        dataset
            .train
            .iter()
            .any(|t| t.head == "dog" && t.relation == "is_a" && t.tail == "mammal"),
        "the `dog is_a mammal` triple should survive the bridge"
    );
    // All five taxonomy entities appear in the dataset.
    let entities: BTreeSet<&str> = dataset
        .train
        .iter()
        .flat_map(|t| [t.head.as_str(), t.tail.as_str()])
        .collect();
    assert_eq!(
        entities.len(),
        5,
        "the five taxonomy entities should all appear"
    );

    println!(
        "  [PASS] {kg_triples} KG triples -> {} dataset triples over {} entities",
        dataset.train.len(),
        entities.len()
    );
}
