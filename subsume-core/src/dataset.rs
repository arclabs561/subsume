//! Utilities for loading and parsing knowledge graph datasets.
//!
//! Supports common formats like WN18RR, FB15k-237, YAGO3-10.
//!
//! # Dataset Format
//!
//! Datasets are expected to be in the standard format:
//! - `train.txt`, `valid.txt`, `test.txt` files
//! - Each line: `head_entity\trelation\ttail_entity` or `head_entity relation tail_entity`
//! - Optional: `entities.dict` and `relations.dict` mapping files

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

/// Represents a single triple in a knowledge graph (head, relation, tail).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    /// Head entity ID or name.
    pub head: String,
    /// Relation ID or name.
    pub relation: String,
    /// Tail entity ID or name.
    pub tail: String,
}

/// A compact ID vocabulary for interning strings to integers.
///
/// Motivation: training/evaluation hot paths frequently do lookups and comparisons over
/// entity/relation IDs. Using `usize` IDs enables:
/// - cheaper hashing (or no hashing in vector-based storage)
/// - denser data structures (vectors instead of `HashMap<String, _>`)
/// - lower memory usage for repeated strings
#[derive(Debug, Clone, Default)]
pub struct Vocab {
    to_id: HashMap<String, usize>,
    from_id: Vec<String>,
}

impl Vocab {
    /// Number of interned items.
    pub fn len(&self) -> usize {
        self.from_id.len()
    }

    /// True iff no items are interned.
    pub fn is_empty(&self) -> bool {
        self.from_id.is_empty()
    }

    /// Intern a string, returning its stable ID.
    pub fn intern(&mut self, s: String) -> usize {
        if let Some(&id) = self.to_id.get(&s) {
            return id;
        }
        let id = self.from_id.len();
        self.from_id.push(s.clone());
        self.to_id.insert(s, id);
        id
    }

    /// Get the string for an ID.
    pub fn get(&self, id: usize) -> Option<&str> {
        self.from_id.get(id).map(|s| s.as_str())
    }

    /// Get the ID for a string (if already interned).
    pub fn id(&self, s: &str) -> Option<usize> {
        self.to_id.get(s).copied()
    }
}

/// A triple stored as interned integer IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TripleIds {
    /// Head entity ID (index into `InternedDataset::entities`).
    pub head: usize,
    /// Relation ID (index into `InternedDataset::relations`).
    pub relation: usize,
    /// Tail entity ID (index into `InternedDataset::entities`).
    pub tail: usize,
}

/// Dataset structure using interned IDs for entities/relations.
#[derive(Debug, Clone)]
pub struct InternedDataset {
    /// Training triples (interned).
    pub train: Vec<TripleIds>,
    /// Validation triples (interned).
    pub valid: Vec<TripleIds>,
    /// Test triples (interned).
    pub test: Vec<TripleIds>,
    /// Entity vocabulary (ID ↔ string).
    pub entities: Vocab,
    /// Relation vocabulary (ID ↔ string).
    pub relations: Vocab,
}

/// Errors that can occur during dataset operations.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid data format in the dataset file.
    #[error("Invalid data format: {0}")]
    InvalidFormat(String),
    /// A required dataset file is missing.
    #[error("Missing file: {0}")]
    MissingFile(String),
    /// Network error when downloading datasets.
    #[error("Network error: {0}")]
    Network(String),
}

/// Dataset structure containing train/valid/test splits.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Training triples
    pub train: Vec<Triple>,
    /// Validation triples
    pub valid: Vec<Triple>,
    /// Test triples
    pub test: Vec<Triple>,
    /// Entity ID to name mapping (if available)
    pub entity_map: Option<HashMap<String, String>>,
    /// Relation ID to name mapping (if available)
    pub relation_map: Option<HashMap<String, String>>,
}

impl Dataset {
    /// Create a new dataset from triples.
    pub fn new(train: Vec<Triple>, valid: Vec<Triple>, test: Vec<Triple>) -> Self {
        Self {
            train,
            valid,
            test,
            entity_map: None,
            relation_map: None,
        }
    }

    /// Get all unique entities from the dataset.
    pub fn entities(&self) -> std::collections::HashSet<String> {
        let mut entities = std::collections::HashSet::new();
        for triple in self
            .train
            .iter()
            .chain(self.valid.iter())
            .chain(self.test.iter())
        {
            entities.insert(triple.head.clone());
            entities.insert(triple.tail.clone());
        }
        entities
    }

    /// Get all unique relations from the dataset.
    pub fn relations(&self) -> std::collections::HashSet<String> {
        let mut relations = std::collections::HashSet::new();
        for triple in self
            .train
            .iter()
            .chain(self.valid.iter())
            .chain(self.test.iter())
        {
            relations.insert(triple.relation.clone());
        }
        relations
    }

    /// Get statistics about the dataset.
    pub fn stats(&self) -> DatasetStats {
        DatasetStats {
            num_entities: self.entities().len(),
            num_relations: self.relations().len(),
            num_train: self.train.len(),
            num_valid: self.valid.len(),
            num_test: self.test.len(),
        }
    }

    /// Convert this dataset into an interned representation (consumes `self`).
    ///
    /// This is the recommended form for high-performance training/evaluation loops.
    pub fn into_interned(self) -> InternedDataset {
        let mut entities = Vocab::default();
        let mut relations = Vocab::default();

        let mut intern_triple = |t: Triple| -> TripleIds {
            let head = entities.intern(t.head);
            let relation = relations.intern(t.relation);
            let tail = entities.intern(t.tail);
            TripleIds { head, relation, tail }
        };

        let train = self.train.into_iter().map(&mut intern_triple).collect();
        let valid = self.valid.into_iter().map(&mut intern_triple).collect();
        let test = self.test.into_iter().map(&mut intern_triple).collect();

        InternedDataset {
            train,
            valid,
            test,
            entities,
            relations,
        }
    }
}

/// Dataset statistics.
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Number of unique entities
    pub num_entities: usize,
    /// Number of unique relations
    pub num_relations: usize,
    /// Number of training triples
    pub num_train: usize,
    /// Number of validation triples
    pub num_valid: usize,
    /// Number of test triples
    pub num_test: usize,
}

/// Loads a knowledge graph dataset from a specified directory.
///
/// Expects `train.txt`, `valid.txt`, `test.txt` files in the directory,
/// each containing whitespace-separated or tab-separated triples (head relation tail).
///
/// # Arguments
///
/// * `path` - Path to the directory containing the dataset files.
///
/// # Returns
///
/// A `Dataset` containing train, validation, and test triples.
///
/// # Example
///
/// ```no_run
/// use subsume_core::dataset::load_dataset;
/// use std::path::Path;
///
/// let dataset = load_dataset(Path::new("data/wn18rr"))?;
/// println!("Loaded {} training triples", dataset.train.len());
/// # Ok::<(), subsume_core::dataset::DatasetError>(())
/// ```
pub fn load_dataset(path: &Path) -> Result<Dataset, DatasetError> {
    let train_path = path.join("train.txt");
    let valid_path = path.join("valid.txt");
    let test_path = path.join("test.txt");

    let train_triples = load_triples(&train_path)?;
    let valid_triples = load_triples(&valid_path)?;
    let test_triples = load_triples(&test_path)?;

    // Try to load entity and relation maps if available
    let entity_map = load_map(&path.join("entities.dict")).ok();
    let relation_map = load_map(&path.join("relations.dict")).ok();

    let mut dataset = Dataset::new(train_triples, valid_triples, test_triples);
    dataset.entity_map = entity_map;
    dataset.relation_map = relation_map;

    Ok(dataset)
}

/// Loads triples from a single file.
///
/// Supports both tab-separated and whitespace-separated formats.
fn load_triples(file_path: &Path) -> Result<Vec<Triple>, DatasetError> {
    if !file_path.exists() {
        return Err(DatasetError::MissingFile(format!(
            "Dataset file not found: {}",
            file_path.display()
        )));
    }
    let file = File::open(file_path)?;
    let reader = io::BufReader::new(file);
    let mut triples = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue; // Skip empty lines and comments
        }

        // Try tab-separated first, then whitespace
        let parts: Vec<&str> = if trimmed.contains('\t') {
            trimmed.split('\t').collect()
        } else {
            trimmed.split_whitespace().collect()
        };

        if parts.len() == 3 {
            triples.push(Triple {
                head: parts[0].to_string(),
                relation: parts[1].to_string(),
                tail: parts[2].to_string(),
            });
        } else if !trimmed.is_empty() {
            return Err(DatasetError::InvalidFormat(format!(
                "Line {} has invalid format: '{}'. Expected 3 parts (head, relation, tail).",
                line_num + 1,
                trimmed
            )));
        }
    }
    Ok(triples)
}

/// Loads an entity or relation mapping file.
///
/// Expects a file where each line is `ID\tName` or `ID Name`.
pub fn load_map(file_path: &Path) -> Result<HashMap<String, String>, DatasetError> {
    if !file_path.exists() {
        return Err(DatasetError::MissingFile(format!(
            "Mapping file not found: {}",
            file_path.display()
        )));
    }
    let file = File::open(file_path)?;
    let reader = io::BufReader::new(file);
    let mut map = HashMap::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Try tab-separated first, then whitespace
        let parts: Vec<&str> = if trimmed.contains('\t') {
            trimmed.split('\t').collect()
        } else {
            trimmed.split_whitespace().collect()
        };

        if parts.len() >= 2 {
            // ID is first part, name is the rest (in case name contains spaces)
            let id = parts[0].to_string();
            let name = parts[1..].join(" ");
            map.insert(id, name);
        } else if !trimmed.is_empty() {
            return Err(DatasetError::InvalidFormat(format!(
                "Line {} has invalid format: '{}'. Expected 2 parts (ID, Name).",
                line_num + 1,
                trimmed
            )));
        }
    }
    Ok(map)
}

#[cfg(test)]
mod intern_tests {
    use super::*;

    #[test]
    fn dataset_into_interned_roundtrips_ids() {
        let ds = Dataset::new(
            vec![
                Triple {
                    head: "a".to_string(),
                    relation: "r".to_string(),
                    tail: "b".to_string(),
                },
                Triple {
                    head: "b".to_string(),
                    relation: "r".to_string(),
                    tail: "c".to_string(),
                },
            ],
            vec![Triple {
                head: "a".to_string(),
                relation: "r".to_string(),
                tail: "c".to_string(),
            }],
            vec![],
        );

        let interned = ds.into_interned();
        assert_eq!(interned.relations.len(), 1);
        assert_eq!(interned.entities.len(), 3);
        assert_eq!(interned.train.len(), 2);
        assert_eq!(interned.valid.len(), 1);

        // Spot-check that the IDs refer back to the right strings.
        let t0 = interned.train[0];
        assert_eq!(interned.entities.get(t0.head), Some("a"));
        assert_eq!(interned.relations.get(t0.relation), Some("r"));
        assert_eq!(interned.entities.get(t0.tail), Some("b"));
    }
}

/// Download a standard dataset (placeholder - requires actual download implementation).
///
/// This is a placeholder function. In a real implementation, this would:
/// 1. Check if dataset already exists locally
/// 2. Download from standard URLs (e.g., https://github.com/...)
/// 3. Extract and verify the dataset
///
/// # Supported Datasets
///
/// - `wn18rr`: WordNet knowledge graph
/// - `fb15k-237`: Freebase knowledge graph
/// - `yago3-10`: YAGO knowledge graph
pub fn download_dataset(_name: &str, _output_dir: &Path) -> Result<(), DatasetError> {
    // Placeholder - would implement actual download logic
    Err(DatasetError::Network(
        "Dataset download not yet implemented. Please download datasets manually from:\n\
         - WN18RR: https://github.com/kkteru/grail\n\
         - FB15k-237: https://github.com/TimDettmers/ConvE\n\
         - YAGO3-10: https://github.com/TimDettmers/ConvE"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_load_triples_success() -> Result<(), DatasetError> {
        let dir = tempdir()?;
        let file_path = dir.path().join("test.txt");
        let mut file = File::create(&file_path)?;
        writeln!(file, "e1 r1 e2")?;
        writeln!(file, "e3 r2 e4")?;

        let triples = load_triples(&file_path)?;
        assert_eq!(triples.len(), 2);
        assert_eq!(
            triples[0],
            Triple {
                head: "e1".to_string(),
                relation: "r1".to_string(),
                tail: "e2".to_string()
            }
        );
        Ok(())
    }

    #[test]
    fn test_load_triples_tab_separated() -> Result<(), DatasetError> {
        let dir = tempdir()?;
        let file_path = dir.path().join("test.txt");
        let mut file = File::create(&file_path)?;
        writeln!(file, "e1\tr1\te2")?;

        let triples = load_triples(&file_path)?;
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].head, "e1");
        Ok(())
    }

    #[test]
    fn test_load_triples_invalid_format() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "e1 r1").unwrap();

        let err = load_triples(&file_path).unwrap_err();
        assert!(matches!(err, DatasetError::InvalidFormat(_)));
    }

    #[test]
    fn test_load_map_success() -> Result<(), DatasetError> {
        let dir = tempdir()?;
        let file_path = dir.path().join("map.txt");
        let mut file = File::create(&file_path)?;
        writeln!(file, "0 entity_0")?;
        writeln!(file, "1 entity_1")?;

        let map = load_map(&file_path)?;
        assert_eq!(map.len(), 2);
        assert_eq!(map["0"], "entity_0");
        Ok(())
    }

    #[test]
    fn test_load_dataset_success() -> Result<(), DatasetError> {
        let dir = tempdir()?;
        let train_path = dir.path().join("train.txt");
        let valid_path = dir.path().join("valid.txt");
        let test_path = dir.path().join("test.txt");

        File::create(&train_path)?.write_all(b"e1 r1 e2\n")?;
        File::create(&valid_path)?.write_all(b"e3 r2 e4\n")?;
        File::create(&test_path)?.write_all(b"e5 r3 e6\n")?;

        let dataset = load_dataset(dir.path())?;
        assert_eq!(dataset.train.len(), 1);
        assert_eq!(dataset.valid.len(), 1);
        assert_eq!(dataset.test.len(), 1);
        Ok(())
    }

    #[test]
    fn test_dataset_entities() {
        let dataset = Dataset::new(
            vec![
                Triple {
                    head: "e1".to_string(),
                    relation: "r1".to_string(),
                    tail: "e2".to_string(),
                },
                Triple {
                    head: "e2".to_string(),
                    relation: "r1".to_string(),
                    tail: "e3".to_string(),
                },
            ],
            vec![],
            vec![],
        );
        let entities = dataset.entities();
        assert_eq!(entities.len(), 3);
        assert!(entities.contains("e1"));
        assert!(entities.contains("e2"));
        assert!(entities.contains("e3"));
    }

    #[test]
    fn test_dataset_stats() {
        let dataset = Dataset::new(
            vec![Triple {
                head: "e1".to_string(),
                relation: "r1".to_string(),
                tail: "e2".to_string(),
            }],
            vec![Triple {
                head: "e2".to_string(),
                relation: "r1".to_string(),
                tail: "e3".to_string(),
            }],
            vec![Triple {
                head: "e3".to_string(),
                relation: "r1".to_string(),
                tail: "e4".to_string(),
            }],
        );
        let stats = dataset.stats();
        assert_eq!(stats.num_train, 1);
        assert_eq!(stats.num_valid, 1);
        assert_eq!(stats.num_test, 1);
        assert_eq!(stats.num_entities, 4);
        assert_eq!(stats.num_relations, 1);
    }
}
