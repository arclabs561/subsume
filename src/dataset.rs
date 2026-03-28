//! Knowledge graph dataset loading and interning.
//!
//! Core types ([`Triple`], [`Dataset`], [`InternedDataset`], [`Vocab`], [`TripleIds`])
//! and loading functions ([`load_dataset`]) are provided by [`lattix::kge`] and
//! re-exported here. This module adds subsume-specific extensions.

// Re-export everything from lattix::kge
pub use lattix::kge::{
    load_dataset, load_triples, Dataset, FilterIndex, InternedDataset, Triple, TripleIds, Vocab,
};

// Re-export lattix error type for dataset operations
pub use lattix::Error as DatasetError;

/// Extension methods for [`Dataset`] specific to subsume.
pub trait DatasetExt {
    /// Create a dataset from a flat list of triples, splitting into train/valid/test.
    ///
    /// Shuffles the triples deterministically using the given seed, then splits
    /// by the given ratios. Ratios are normalized to sum to 1.0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use subsume::dataset::{Dataset, Triple, DatasetExt};
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
    fn from_all_triples(
        triples: Vec<Triple>,
        train_ratio: f64,
        valid_ratio: f64,
        test_ratio: f64,
        seed: u64,
    ) -> Dataset;
}

impl DatasetExt for Dataset {
    fn from_all_triples(
        mut triples: Vec<Triple>,
        train_ratio: f64,
        valid_ratio: f64,
        test_ratio: f64,
        seed: u64,
    ) -> Dataset {
        let n = triples.len();
        if n == 0 {
            return Dataset::new(Vec::new(), Vec::new(), Vec::new());
        }

        // Deterministic shuffle using a simple LCG.
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

        assert!(
            train_ratio >= 0.0 && valid_ratio >= 0.0 && test_ratio >= 0.0,
            "split ratios must be non-negative"
        );
        let total = train_ratio + valid_ratio + test_ratio;
        assert!(total > 0.0, "at least one split ratio must be positive");
        let train_end = ((train_ratio / total) * n as f64).round() as usize;
        let valid_end = train_end + ((valid_ratio / total) * n as f64).round() as usize;

        let test = triples.split_off(valid_end.min(n));
        let valid = triples.split_off(train_end.min(triples.len()));
        let train = triples;

        Dataset::new(train, valid, test)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_from_all_triples_splits_correctly() {
        let triples = vec![
            Triple::new("a", "r", "b"),
            Triple::new("b", "r", "c"),
            Triple::new("c", "r", "d"),
            Triple::new("d", "r", "e"),
            Triple::new("e", "r", "f"),
        ];
        let ds = Dataset::from_all_triples(triples, 0.6, 0.2, 0.2, 42);
        assert_eq!(ds.train.len() + ds.valid.len() + ds.test.len(), 5);
    }

    #[test]
    fn dataset_into_interned_roundtrips() {
        let ds = Dataset::new(
            vec![Triple::new("a", "r", "b"), Triple::new("b", "r", "c")],
            vec![Triple::new("a", "r", "c")],
            vec![],
        );
        let interned = ds.into_interned();
        assert_eq!(interned.entities.len(), 3);
        assert_eq!(interned.relations.len(), 1);
        assert_eq!(interned.train.len(), 2);
        assert_eq!(interned.valid.len(), 1);

        let t0 = interned.train[0];
        assert_eq!(interned.entities.get(t0.head), Some("a"));
        assert_eq!(interned.relations.get(t0.relation), Some("r"));
        assert_eq!(interned.entities.get(t0.tail), Some("b"));
    }

    #[test]
    fn from_arrays_roundtrips() {
        let train = vec![(0, 0, 1), (1, 0, 2)];
        let valid = vec![(0, 0, 2)];
        let test = vec![(2, 0, 0)];
        let ds = InternedDataset::from_arrays(&train, &valid, &test, 3, 1);
        assert_eq!(ds.num_entities(), 3);
        assert_eq!(ds.num_relations(), 1);
        assert_eq!(ds.entities.get(0), Some("e0"));
    }
}
