//! Adapter for constructing [`Dataset`] from [`petgraph`] graphs.
//!
//! Converts a `petgraph::Graph` (or `DiGraph`) where nodes are entity names
//! and edges are relation names into subsume's [`Triple`] / [`Dataset`] format.

use crate::dataset::{Dataset, Triple};

/// Convert a directed petgraph into a [`Dataset`] with all triples in the training split.
///
/// Node weights become entity names (via `ToString`), edge weights become
/// relation names. All edges become training triples; validation and test
/// splits are empty (callers can split manually).
///
/// # Example
///
/// ```rust,ignore
/// use petgraph::graph::DiGraph;
/// use subsume::petgraph_adapter::from_graph;
///
/// let mut g = DiGraph::new();
/// let dog = g.add_node("dog");
/// let animal = g.add_node("animal");
/// g.add_edge(dog, animal, "is_a");
///
/// let dataset = from_graph(&g);
/// assert_eq!(dataset.train.len(), 1);
/// ```
pub fn from_graph<N, E>(graph: &petgraph::graph::DiGraph<N, E>) -> Dataset
where
    N: ToString,
    E: ToString,
{
    let triples: Vec<Triple> = graph
        .edge_indices()
        .filter_map(|e| {
            let (src, dst) = graph.edge_endpoints(e)?;
            let rel = graph.edge_weight(e)?;
            Some(Triple::new(
                graph[src].to_string(),
                rel.to_string(),
                graph[dst].to_string(),
            ))
        })
        .collect();

    Dataset::new(triples, Vec::new(), Vec::new())
}
