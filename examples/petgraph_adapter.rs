//! Build a subsume `Dataset` from a `petgraph` `DiGraph`.
//!
//! `subsume::petgraph_adapter::from_graph` turns a directed graph whose nodes
//! are entity names and edges are relation names into the `Triple`/`Dataset`
//! format subsume trains box and cone embeddings on. This is the bridge from an
//! in-memory graph (or anything that builds one) to region-embedding training
//! data, with no intermediate file.
//!
//! Run: `cargo run --example petgraph_adapter --features petgraph`

use petgraph::graph::DiGraph;
use std::collections::BTreeSet;
use subsume::petgraph_adapter::from_graph;

fn main() {
    // A small is-a taxonomy as a directed graph: leaves point at their parents.
    let mut g = DiGraph::new();
    let dog = g.add_node("dog");
    let cat = g.add_node("cat");
    let mammal = g.add_node("mammal");
    let animal = g.add_node("animal");
    let living = g.add_node("living_thing");

    g.add_edge(dog, mammal, "is_a");
    g.add_edge(cat, mammal, "is_a");
    g.add_edge(mammal, animal, "is_a");
    g.add_edge(animal, living, "is_a");

    let dataset = from_graph(&g);

    println!("petgraph DiGraph -> subsume Dataset");
    println!("  nodes: {}, edges: {}", g.node_count(), g.edge_count());
    println!("  dataset triples: {}", dataset.train.len());
    for t in &dataset.train {
        println!("    {} --{}--> {}", t.head, t.relation, t.tail);
    }

    // Every directed edge becomes exactly one (head, relation, tail) triple.
    assert_eq!(
        dataset.train.len(),
        g.edge_count(),
        "each directed edge should yield one triple"
    );
    // The specific is-a edge round-trips through the adapter unchanged.
    assert!(
        dataset
            .train
            .iter()
            .any(|t| t.head == "dog" && t.relation == "is_a" && t.tail == "mammal"),
        "the `dog is_a mammal` edge should appear as a triple"
    );
    // Every node that participates in an edge appears as a Dataset entity.
    let entities: BTreeSet<&str> = dataset
        .train
        .iter()
        .flat_map(|t| [t.head.as_str(), t.tail.as_str()])
        .collect();
    assert_eq!(
        entities.len(),
        g.node_count(),
        "all graph nodes should appear as entities"
    );

    println!(
        "  [PASS] {} edges -> {} triples over {} entities",
        g.edge_count(),
        dataset.train.len(),
        entities.len()
    );
}
