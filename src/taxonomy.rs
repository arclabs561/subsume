//! Taxonomy dataset loading for the TaxoBell format.
//!
//! TaxoBell datasets represent concept taxonomies as directed graphs where edges
//! encode hypernym (parent-child) relationships. Each dataset consists of:
//!
//! - A **terms file** (`.terms`): tab-separated `id\tname` mapping concept IDs to names.
//! - A **taxonomy file** (`.taxo`): tab-separated `parent_id\tchild_id` hypernym edges.
//! - An optional **dictionary file** (`dic.json`): JSON object mapping concept names
//!   to natural-language definitions (used as encoder input for definition-aware models).
//!
//! Datasets include Science (429 nodes), Environment (475), Food (1486),
//! WordNet subsets, and MeSH (9710).
//!
//! # Example
//!
//! ```no_run
//! use subsume::taxonomy::TaxonomyDataset;
//! use std::path::Path;
//!
//! let taxo = TaxonomyDataset::load(
//!     Path::new("data/science.terms"),
//!     Path::new("data/science.taxo"),
//!     Some(Path::new("data/science.dic.json")),
//! )?;
//! println!("{} nodes, {} edges", taxo.nodes.len(), taxo.edges.len());
//!
//! // Convert to subsume Triple format for training
//! let triples = taxo.to_triples();
//! # Ok::<(), subsume::dataset::DatasetError>(())
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use crate::dataset::{DatasetError, Triple};

/// A single concept node in a taxonomy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaxonomyNode {
    /// Numeric ID from the terms file.
    pub id: usize,
    /// Human-readable concept name.
    pub name: String,
    /// Optional natural-language definition (from dic.json).
    pub definition: Option<String>,
}

/// A taxonomy dataset: nodes (concepts) and directed edges (hypernym relations).
///
/// Edges are stored as `(parent_id, child_id)` pairs where IDs refer to
/// `TaxonomyNode::id` values. Use `node_index` to look up a node's position
/// in the `nodes` vec by its ID.
#[derive(Debug, Clone)]
pub struct TaxonomyDataset {
    /// All concept nodes.
    pub nodes: Vec<TaxonomyNode>,
    /// Hypernym edges as `(parent_id, child_id)` pairs.
    pub edges: Vec<(usize, usize)>,
    /// Maps `TaxonomyNode::id` to its index in `nodes`.
    pub node_index: HashMap<usize, usize>,
}

impl TaxonomyDataset {
    /// Load a TaxoBell taxonomy from disk.
    ///
    /// - `terms_path`: tab-separated file with lines `id\tname`.
    /// - `taxo_path`: tab-separated file with lines `parent_id\tchild_id`.
    /// - `dict_path`: optional JSON file mapping concept names to definitions.
    pub fn load(
        terms_path: &Path,
        taxo_path: &Path,
        dict_path: Option<&Path>,
    ) -> Result<Self, DatasetError> {
        // Load definitions dict if provided.
        let definitions: HashMap<String, String> = match dict_path {
            Some(path) if path.exists() => load_definitions(path)?,
            _ => HashMap::new(),
        };

        // Parse terms file.
        let (nodes, node_index) = load_terms(terms_path, &definitions)?;

        // Parse taxonomy edges.
        let edges = load_edges(taxo_path, &node_index)?;

        Ok(Self {
            nodes,
            edges,
            node_index,
        })
    }

    /// Convert taxonomy edges to subsume `Triple`s.
    ///
    /// Each `(parent, child)` edge becomes a triple with:
    /// - `head`: child concept name
    /// - `relation`: `"hypernym"`
    /// - `tail`: parent concept name
    ///
    /// This follows the convention that the child *is-a* (hypernym of) the parent,
    /// matching TaxoBell's evaluation protocol.
    pub fn to_triples(&self) -> Vec<Triple> {
        self.edges
            .iter()
            .map(|&(parent_id, child_id)| {
                let parent_name = &self.nodes[self.node_index[&parent_id]].name;
                let child_name = &self.nodes[self.node_index[&child_id]].name;
                Triple {
                    head: child_name.clone(),
                    relation: "hypernym".to_string(),
                    tail: parent_name.clone(),
                }
            })
            .collect()
    }

    /// Split edges into train/validation/test sets.
    ///
    /// Uses a deterministic shuffle seeded by `seed`. Ratios should sum to at most 1.0;
    /// any remainder goes to the test set.
    ///
    /// Returns `(train_edges, val_edges, test_edges)` as vectors of `(parent_id, child_id)`.
    pub fn split(
        &self,
        train_ratio: f64,
        val_ratio: f64,
        seed: u64,
    ) -> (
        Vec<(usize, usize)>,
        Vec<(usize, usize)>,
        Vec<(usize, usize)>,
    ) {
        let mut edges = self.edges.clone();
        deterministic_shuffle(&mut edges, seed);

        let n = edges.len();
        let train_end = (n as f64 * train_ratio).round() as usize;
        let val_end = train_end + (n as f64 * val_ratio).round() as usize;
        let val_end = val_end.min(n);

        let test = edges.split_off(val_end);
        let val = edges.split_off(train_end);
        let train = edges;

        (train, val, test)
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Deterministic Fisher-Yates shuffle using a simple xorshift64 PRNG.
///
/// Avoids pulling in `rand` for a single shuffle operation.
fn deterministic_shuffle<T>(slice: &mut [T], seed: u64) {
    let mut state = seed.wrapping_add(1); // avoid zero state
    for i in (1..slice.len()).rev() {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        slice.swap(i, j);
    }
}

/// Parse a terms file (tab-separated `id\tname`).
fn load_terms(
    path: &Path,
    definitions: &HashMap<String, String>,
) -> Result<(Vec<TaxonomyNode>, HashMap<usize, usize>), DatasetError> {
    if !path.exists() {
        return Err(DatasetError::MissingFile(format!(
            "Terms file not found: {}",
            path.display()
        )));
    }

    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut nodes = Vec::new();
    let mut node_index = HashMap::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.splitn(2, '\t').collect();
        if parts.len() != 2 {
            return Err(DatasetError::InvalidFormat(format!(
                "{}:{}: expected tab-separated 'id\\tname', got '{}'",
                path.display(),
                line_num + 1,
                trimmed,
            )));
        }

        let id: usize = parts[0].parse().map_err(|_| {
            DatasetError::InvalidFormat(format!(
                "{}:{}: invalid node ID '{}'",
                path.display(),
                line_num + 1,
                parts[0],
            ))
        })?;

        let name = parts[1].to_string();
        let definition = definitions.get(&name).cloned();

        let idx = nodes.len();
        node_index.insert(id, idx);
        nodes.push(TaxonomyNode {
            id,
            name,
            definition,
        });
    }

    Ok((nodes, node_index))
}

/// Parse a taxonomy file (tab-separated `parent_id\tchild_id`).
fn load_edges(
    path: &Path,
    node_index: &HashMap<usize, usize>,
) -> Result<Vec<(usize, usize)>, DatasetError> {
    if !path.exists() {
        return Err(DatasetError::MissingFile(format!(
            "Taxonomy file not found: {}",
            path.display()
        )));
    }

    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut edges = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() != 2 {
            return Err(DatasetError::InvalidFormat(format!(
                "{}:{}: expected tab-separated 'parent_id\\tchild_id', got '{}'",
                path.display(),
                line_num + 1,
                trimmed,
            )));
        }

        let parent_id: usize = parts[0].parse().map_err(|_| {
            DatasetError::InvalidFormat(format!(
                "{}:{}: invalid parent ID '{}'",
                path.display(),
                line_num + 1,
                parts[0],
            ))
        })?;

        let child_id: usize = parts[1].parse().map_err(|_| {
            DatasetError::InvalidFormat(format!(
                "{}:{}: invalid child ID '{}'",
                path.display(),
                line_num + 1,
                parts[1],
            ))
        })?;

        // Validate that both IDs exist in the terms file.
        if !node_index.contains_key(&parent_id) {
            return Err(DatasetError::InvalidFormat(format!(
                "{}:{}: parent ID {} not found in terms file",
                path.display(),
                line_num + 1,
                parent_id,
            )));
        }
        if !node_index.contains_key(&child_id) {
            return Err(DatasetError::InvalidFormat(format!(
                "{}:{}: child ID {} not found in terms file",
                path.display(),
                line_num + 1,
                child_id,
            )));
        }

        edges.push((parent_id, child_id));
    }

    Ok(edges)
}

/// Load definitions from a JSON dictionary file.
///
/// When `serde_json` is available (via `ndarray-backend`), parses the file as
/// `{"name": "definition", ...}`. Without `serde_json`, returns an empty map
/// and logs a note.
#[cfg(feature = "ndarray-backend")]
fn load_definitions(path: &Path) -> Result<HashMap<String, String>, DatasetError> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let map: HashMap<String, String> = serde_json::from_reader(reader).map_err(|e| {
        DatasetError::InvalidFormat(format!("Failed to parse dictionary JSON: {e}"))
    })?;
    Ok(map)
}

#[cfg(not(feature = "ndarray-backend"))]
fn load_definitions(_path: &Path) -> Result<HashMap<String, String>, DatasetError> {
    // serde_json not available; definitions are optional so just skip.
    Ok(HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn load_small_taxonomy() {
        let dir = tempdir().unwrap();

        let terms = write_file(
            dir.path(),
            "test.terms",
            "0\tanimal\n1\tdog\n2\tcat\n3\tmammal\n",
        );
        let taxo = write_file(dir.path(), "test.taxo", "0\t3\n3\t1\n3\t2\n");

        let ds = TaxonomyDataset::load(&terms, &taxo, None).unwrap();
        assert_eq!(ds.num_nodes(), 4);
        assert_eq!(ds.num_edges(), 3);

        // Check node lookup.
        let animal_idx = ds.node_index[&0];
        assert_eq!(ds.nodes[animal_idx].name, "animal");
        assert!(ds.nodes[animal_idx].definition.is_none());
    }

    #[test]
    fn to_triples_produces_hypernym_relation() {
        let dir = tempdir().unwrap();

        let terms = write_file(dir.path(), "t.terms", "10\tparent\n20\tchild\n");
        let taxo = write_file(dir.path(), "t.taxo", "10\t20\n");

        let ds = TaxonomyDataset::load(&terms, &taxo, None).unwrap();
        let triples = ds.to_triples();

        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].head, "child");
        assert_eq!(triples[0].relation, "hypernym");
        assert_eq!(triples[0].tail, "parent");
    }

    #[test]
    fn split_covers_all_edges() {
        let dir = tempdir().unwrap();

        // 10 nodes, 9 edges (chain).
        let terms_content: String = (0..10).map(|i| format!("{i}\tn{i}\n")).collect();
        let taxo_content: String = (0..9).map(|i| format!("{i}\t{}\n", i + 1)).collect();

        let terms = write_file(dir.path(), "s.terms", &terms_content);
        let taxo = write_file(dir.path(), "s.taxo", &taxo_content);

        let ds = TaxonomyDataset::load(&terms, &taxo, None).unwrap();
        let (train, val, test) = ds.split(0.6, 0.2, 42);

        // All edges accounted for.
        assert_eq!(train.len() + val.len() + test.len(), 9);
        // Approximate split sizes (rounding may shift by 1).
        assert!(
            train.len() >= 4 && train.len() <= 6,
            "train len = {}",
            train.len()
        );
        assert!(val.len() >= 1 && val.len() <= 3, "val len = {}", val.len());
    }

    #[test]
    fn split_is_deterministic() {
        let dir = tempdir().unwrap();

        let terms_content: String = (0..20).map(|i| format!("{i}\tn{i}\n")).collect();
        let taxo_content: String = (0..19).map(|i| format!("{i}\t{}\n", i + 1)).collect();

        let terms = write_file(dir.path(), "d.terms", &terms_content);
        let taxo = write_file(dir.path(), "d.taxo", &taxo_content);

        let ds = TaxonomyDataset::load(&terms, &taxo, None).unwrap();
        let (t1, v1, e1) = ds.split(0.7, 0.15, 123);
        let (t2, v2, e2) = ds.split(0.7, 0.15, 123);

        assert_eq!(t1, t2);
        assert_eq!(v1, v2);
        assert_eq!(e1, e2);
    }

    #[test]
    fn missing_terms_file_errors() {
        let dir = tempdir().unwrap();
        let taxo = write_file(dir.path(), "x.taxo", "0\t1\n");
        let result = TaxonomyDataset::load(&dir.path().join("missing.terms"), &taxo, None);
        assert!(matches!(result, Err(DatasetError::MissingFile(_))));
    }

    #[test]
    fn invalid_id_in_taxo_errors() {
        let dir = tempdir().unwrap();
        let terms = write_file(dir.path(), "e.terms", "0\ta\n1\tb\n");
        let taxo = write_file(dir.path(), "e.taxo", "0\t99\n"); // 99 not in terms

        let result = TaxonomyDataset::load(&terms, &taxo, None);
        assert!(matches!(result, Err(DatasetError::InvalidFormat(_))));
    }

    // ---- audit-driven regression tests ----

    /// Bad TSV format (missing tab separator) should be rejected.
    #[test]
    fn test_load_rejects_malformed_terms() {
        let dir = tempdir().unwrap();

        // No tab separator -- just "0 animal" with a space
        let terms = write_file(dir.path(), "bad.terms", "0 animal\n1\tdog\n");
        let taxo = write_file(dir.path(), "bad.taxo", "0\t1\n");

        let result = TaxonomyDataset::load(&terms, &taxo, None);
        assert!(
            matches!(result, Err(DatasetError::InvalidFormat(_))),
            "should reject terms line without tab separator, got {result:?}"
        );
    }

    /// Same seed produces exactly the same split.
    #[test]
    fn test_split_deterministic() {
        let dir = tempdir().unwrap();

        let terms_content: String = (0..50).map(|i| format!("{i}\tn{i}\n")).collect();
        let taxo_content: String = (0..49).map(|i| format!("{i}\t{}\n", i + 1)).collect();

        let terms = write_file(dir.path(), "det.terms", &terms_content);
        let taxo = write_file(dir.path(), "det.taxo", &taxo_content);

        let ds = TaxonomyDataset::load(&terms, &taxo, None).unwrap();

        for seed in [0, 42, 12345, u64::MAX] {
            let (t1, v1, e1) = ds.split(0.6, 0.2, seed);
            let (t2, v2, e2) = ds.split(0.6, 0.2, seed);
            assert_eq!(t1, t2, "train differs for seed {seed}");
            assert_eq!(v1, v2, "val differs for seed {seed}");
            assert_eq!(e1, e2, "test differs for seed {seed}");
        }

        // Different seeds produce different splits (with high probability for 49 edges)
        let (t_a, _, _) = ds.split(0.6, 0.2, 1);
        let (t_b, _, _) = ds.split(0.6, 0.2, 2);
        assert_ne!(
            t_a, t_b,
            "different seeds should (almost surely) produce different splits"
        );
    }

    /// to_triples: parent is head? No -- child is head, parent is tail.
    /// This matches the convention: child *is-a* (hypernym of) parent.
    #[test]
    fn test_to_triples_parent_child_direction() {
        let dir = tempdir().unwrap();

        let terms = write_file(dir.path(), "dir.terms", "100\tanimal\n200\tdog\n");
        let taxo = write_file(dir.path(), "dir.taxo", "100\t200\n"); // parent=100(animal), child=200(dog)

        let ds = TaxonomyDataset::load(&terms, &taxo, None).unwrap();
        let triples = ds.to_triples();

        assert_eq!(triples.len(), 1);
        // Child is head (the one asserting the relation)
        assert_eq!(triples[0].head, "dog", "child should be head");
        assert_eq!(triples[0].tail, "animal", "parent should be tail");
        assert_eq!(triples[0].relation, "hypernym");
    }

    #[cfg(feature = "ndarray-backend")]
    #[test]
    fn load_with_definitions() {
        let dir = tempdir().unwrap();

        let terms = write_file(dir.path(), "def.terms", "0\tanimal\n1\tdog\n");
        let taxo = write_file(dir.path(), "def.taxo", "0\t1\n");
        let dict = write_file(
            dir.path(),
            "dic.json",
            r#"{"animal": "A living organism", "dog": "A domesticated canid"}"#,
        );

        let ds = TaxonomyDataset::load(&terms, &taxo, Some(&dict)).unwrap();
        assert_eq!(
            ds.nodes[ds.node_index[&0]].definition.as_deref(),
            Some("A living organism"),
        );
        assert_eq!(
            ds.nodes[ds.node_index[&1]].definition.as_deref(),
            Some("A domesticated canid"),
        );
    }
}
