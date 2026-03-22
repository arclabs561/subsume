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

impl Triple {
    /// Create a new triple from head, relation, and tail.
    pub fn new(
        head: impl Into<String>,
        relation: impl Into<String>,
        tail: impl Into<String>,
    ) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
        }
    }
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

impl InternedDataset {
    /// Create an interned dataset from pre-mapped integer triple arrays.
    ///
    /// For use with OGB or other pipelines that already have integer-mapped triples.
    /// Entity and relation names are auto-generated as "e0", "e1", ... and "r0", "r1", ...
    ///
    /// # Arguments
    ///
    /// * `train` - Training triples as `(head, relation, tail)` tuples
    /// * `valid` - Validation triples
    /// * `test` - Test triples
    /// * `num_entities` - Total entity count (for vocabulary size)
    /// * `num_relations` - Total relation count (for vocabulary size)
    pub fn from_arrays(
        train: &[(usize, usize, usize)],
        valid: &[(usize, usize, usize)],
        test: &[(usize, usize, usize)],
        num_entities: usize,
        num_relations: usize,
    ) -> Self {
        let to_ids = |triples: &[(usize, usize, usize)]| -> Vec<TripleIds> {
            triples
                .iter()
                .map(|&(h, r, t)| TripleIds {
                    head: h,
                    relation: r,
                    tail: t,
                })
                .collect()
        };

        let mut entities = Vocab::default();
        for i in 0..num_entities {
            entities.intern(format!("e{i}"));
        }
        let mut relations = Vocab::default();
        for i in 0..num_relations {
            relations.intern(format!("r{i}"));
        }

        Self {
            train: to_ids(train),
            valid: to_ids(valid),
            test: to_ids(test),
            entities,
            relations,
        }
    }
}

/// Errors that can occur during dataset operations.
#[non_exhaustive]
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
}

impl Dataset {
    /// Create a new dataset from triples.
    pub fn new(train: Vec<Triple>, valid: Vec<Triple>, test: Vec<Triple>) -> Self {
        Self { train, valid, test }
    }

    /// Create a dataset from a flat list of triples, splitting into train/valid/test.
    ///
    /// Shuffles the triples deterministically using the given seed, then splits
    /// by the given ratios. Ratios are normalized to sum to 1.0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume::dataset::{Dataset, Triple};
    ///
    /// let triples = vec![
    ///     Triple::new("Dog", "is_a", "Animal"),
    ///     Triple::new("Cat", "is_a", "Animal"),
    ///     Triple::new("Bird", "is_a", "Animal"),
    ///     Triple::new("Fish", "is_a", "Animal"),
    ///     Triple::new("Dog", "has", "Tail"),
    /// ];
    /// let ds = Dataset::from_all_triples(triples, 0.8, 0.1, 0.1, 42);
    /// assert_eq!(ds.train.len() + ds.valid.len() + ds.test.len(), 5);
    /// ```
    pub fn from_all_triples(
        mut triples: Vec<Triple>,
        train_ratio: f64,
        valid_ratio: f64,
        test_ratio: f64,
        seed: u64,
    ) -> Self {
        // Deterministic shuffle using a simple LCG.
        let n = triples.len();
        if n == 0 {
            return Self::new(Vec::new(), Vec::new(), Vec::new());
        }

        let mut rng = seed;
        let lcg = |s: &mut u64| -> usize {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as usize
        };

        // Fisher-Yates shuffle.
        for i in (1..n).rev() {
            let j = lcg(&mut rng) % (i + 1);
            triples.swap(i, j);
        }

        // Normalize ratios.
        let total = train_ratio + valid_ratio + test_ratio;
        let train_end = ((train_ratio / total) * n as f64).round() as usize;
        let valid_end = train_end + ((valid_ratio / total) * n as f64).round() as usize;

        let test = triples.split_off(valid_end.min(n));
        let valid = triples.split_off(train_end.min(triples.len()));
        let train = triples;

        Self::new(train, valid, test)
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
            TripleIds {
                head,
                relation,
                tail,
            }
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
/// use subsume::dataset::load_dataset;
/// use std::path::Path;
///
/// let dataset = load_dataset(Path::new("data/wn18rr"))?;
/// println!("Loaded {} training triples", dataset.train.len());
/// # Ok::<(), subsume::dataset::DatasetError>(())
/// ```
pub fn load_dataset(path: &Path) -> Result<Dataset, DatasetError> {
    let train_path = path.join("train.txt");
    let valid_path = path.join("valid.txt");
    let test_path = path.join("test.txt");

    let train_triples = load_triples(&train_path)?;
    let valid_triples = load_triples(&valid_path)?;
    let test_triples = load_triples(&test_path)?;

    Ok(Dataset::new(train_triples, valid_triples, test_triples))
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
    fn test_interned_from_arrays() {
        let train = vec![(0, 0, 1), (1, 0, 2)];
        let valid = vec![(0, 0, 2)];
        let test = vec![(2, 0, 0)];
        let ds = InternedDataset::from_arrays(&train, &valid, &test, 3, 1);

        assert_eq!(ds.train.len(), 2);
        assert_eq!(ds.valid.len(), 1);
        assert_eq!(ds.test.len(), 1);
        assert_eq!(ds.entities.len(), 3);
        assert_eq!(ds.relations.len(), 1);
        assert_eq!(ds.train[0].head, 0);
        assert_eq!(ds.train[0].tail, 1);
        assert_eq!(ds.entities.get(0), Some("e0"));
        assert_eq!(ds.relations.get(0), Some("r0"));
    }
}
