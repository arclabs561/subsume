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

/// Convert a directed petgraph into a [`Dataset`] with train/valid/test split.
///
/// Splits edges into train (80%), valid (10%), test (10%) by edge index order.
/// For reproducible random splits, shuffle edge indices before calling.
pub fn from_graph_with_split<N, E>(graph: &petgraph::graph::DiGraph<N, E>) -> Dataset
where
    N: ToString,
    E: ToString,
{
    let all_triples: Vec<Triple> = graph
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

    let n = all_triples.len();
    let train_end = (n * 8) / 10;
    let valid_end = train_end + n / 10;

    let mut iter = all_triples.into_iter();
    let train: Vec<Triple> = iter.by_ref().take(train_end).collect();
    let valid: Vec<Triple> = iter.by_ref().take(valid_end - train_end).collect();
    let test: Vec<Triple> = iter.collect();

    Dataset::new(train, valid, test)
}
