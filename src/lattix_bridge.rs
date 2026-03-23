//! Bridge between [`lattix`] knowledge graphs and subsume datasets.
//!
//! Converts a [`lattix::KnowledgeGraph`] into subsume's [`Dataset`](crate::dataset::Dataset)
//! for training box/cone embeddings. Supports loading from any format lattix handles
//! (N-Triples, Turtle, N-Quads, CSV, JSON-LD).
//!
//! # Example
//!
//! ```rust,no_run
//! use subsume::lattix_bridge::kg_to_dataset;
//!
//! // Load a KG from N-Triples (Gene Ontology, SNOMED-CT, etc.)
//! let kg = lattix::KnowledgeGraph::from_ntriples_file("go.nt").unwrap();
//! let dataset = kg_to_dataset(&kg);
//! println!("Triples: {}", dataset.train.len());
//! ```

use crate::dataset::{Dataset, Triple};

/// Convert a lattix [`KnowledgeGraph`](lattix::KnowledgeGraph) into a subsume [`Dataset`].
///
/// All triples become training triples (no validation/test split).
/// Use [`Dataset`] methods or manual splitting to create val/test sets.
pub fn kg_to_dataset(kg: &lattix::KnowledgeGraph) -> Dataset {
    let triples: Vec<Triple> = kg
        .triples()
        .map(|t| {
            Triple::new(
                t.subject().as_str(),
                t.predicate().as_str(),
                t.object().as_str(),
            )
        })
        .collect();

    Dataset {
        train: triples,
        valid: Vec::new(),
        test: Vec::new(),
    }
}

/// Extract only subsumption triples (rdfs:subClassOf) from a lattix KG.
///
/// Filters to triples where the predicate matches `rdfs:subClassOf` or
/// a custom predicate string. Returns a [`Dataset`] with these as training triples.
pub fn extract_subsumption_triples(kg: &lattix::KnowledgeGraph, predicate: &str) -> Dataset {
    let triples: Vec<Triple> = kg
        .triples()
        .filter(|t| t.predicate().as_str() == predicate)
        .map(|t| {
            Triple::new(
                t.subject().as_str(),
                t.predicate().as_str(),
                t.object().as_str(),
            )
        })
        .collect();

    Dataset {
        train: triples,
        valid: Vec::new(),
        test: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_kg() -> lattix::KnowledgeGraph {
        let mut kg = lattix::KnowledgeGraph::new();
        kg.add_triple(lattix::Triple::new("Dog", "subClassOf", "Animal"));
        kg.add_triple(lattix::Triple::new("Cat", "subClassOf", "Animal"));
        kg.add_triple(lattix::Triple::new("Dog", "hasColor", "Brown"));
        kg
    }

    #[test]
    fn test_kg_to_dataset() {
        let kg = make_test_kg();
        let ds = kg_to_dataset(&kg);
        assert_eq!(ds.train.len(), 3);
        assert!(ds.valid.is_empty());
        assert!(ds.test.is_empty());
    }

    #[test]
    fn test_extract_subsumption() {
        let kg = make_test_kg();
        let ds = extract_subsumption_triples(&kg, "subClassOf");
        assert_eq!(ds.train.len(), 2);
        assert_eq!(ds.train[0].relation, "subClassOf");
    }

    #[test]
    fn test_extract_empty() {
        let kg = make_test_kg();
        let ds = extract_subsumption_triples(&kg, "nonexistent");
        assert!(ds.train.is_empty());
    }
}
