use std::collections::{HashMap, HashSet};

/// Per-relation cardinality statistics for Bernoulli negative sampling.
///
/// Bernoulli sampling (Wang et al., 2014) adjusts the probability of corrupting
/// head vs tail based on relation cardinality. For a 1-to-N relation like
/// `born_in`, there are many tails per head, so corrupting the tail is easier
/// (more candidates) and less informative. Bernoulli sampling compensates by
/// corrupting the head more often for such relations.
///
/// **Reference**: Wang et al. (2014), "Knowledge Graph Embedding by Translating
/// on Hyperplanes" (AAAI), Section 4.
#[derive(Debug, Clone)]
pub struct RelationCardinality {
    /// Average tails per head for this relation.
    pub tph: f32,
    /// Average heads per tail for this relation.
    pub hpt: f32,
}

/// Relation-specific entity pools derived from training data.
#[derive(Debug, Clone, Default)]
pub struct RelationEntityPools {
    /// Unique head entities observed with this relation.
    pub heads: Vec<usize>,
    /// Unique tail entities observed with this relation.
    pub tails: Vec<usize>,
}

impl RelationCardinality {
    /// Probability of corrupting the head entity.
    ///
    /// `P(corrupt_head) = tph / (tph + hpt)`.
    /// Relations with many tails per head get higher head-corruption probability.
    #[inline]
    pub fn head_corrupt_prob(&self) -> f32 {
        let denom = self.tph + self.hpt;
        if denom == 0.0 {
            0.5
        } else {
            self.tph / denom
        }
    }
}

/// Compute per-relation cardinality statistics from a set of interned triples.
///
/// For each relation `r`, counts the number of unique heads and unique tails,
/// then computes:
/// - `tph` = (number of triples with relation r) / (number of unique heads for r)
/// - `hpt` = (number of triples with relation r) / (number of unique tails for r)
///
/// The returned map is keyed by relation ID (usize).
pub fn compute_relation_cardinalities(
    triples: &[(usize, usize, usize)],
) -> HashMap<usize, RelationCardinality> {
    // Per-relation: (unique heads, unique tails, count)
    let mut stats: HashMap<usize, (HashSet<usize>, HashSet<usize>, usize)> = HashMap::new();

    for &(h, r, t) in triples {
        let entry = stats
            .entry(r)
            .or_insert_with(|| (HashSet::new(), HashSet::new(), 0));
        entry.0.insert(h);
        entry.1.insert(t);
        entry.2 += 1;
    }

    stats
        .into_iter()
        .map(|(r, (heads, tails, count))| {
            let tph = count as f32 / heads.len().max(1) as f32;
            let hpt = count as f32 / tails.len().max(1) as f32;
            (r, RelationCardinality { tph, hpt })
        })
        .collect()
}

/// Compute unique head/tail entity pools per relation from interned triples.
///
/// These pools are useful for type-constrained negative sampling: sample a
/// corrupted head from heads observed with the same relation, or a corrupted
/// tail from the corresponding tail pool. When a relation is too sparse, the
/// caller should fall back to the full entity set.
pub fn compute_relation_entity_pools(
    triples: &[(usize, usize, usize)],
) -> HashMap<usize, RelationEntityPools> {
    let mut stats: HashMap<usize, (HashSet<usize>, HashSet<usize>)> = HashMap::new();

    for &(h, r, t) in triples {
        let entry = stats
            .entry(r)
            .or_insert_with(|| (HashSet::new(), HashSet::new()));
        entry.0.insert(h);
        entry.1.insert(t);
    }

    stats
        .into_iter()
        .map(|(r, (heads, tails))| {
            let mut heads: Vec<usize> = heads.into_iter().collect();
            let mut tails: Vec<usize> = tails.into_iter().collect();
            heads.sort_unstable();
            tails.sort_unstable();
            (r, RelationEntityPools { heads, tails })
        })
        .collect()
}

/// Sample an index from `candidates`, excluding `exclude` if present.
///
/// `sample_index` should return an index in `0..candidates.len()`. The helper
/// makes a few random attempts before falling back to the first valid entry.
pub fn sample_excluding<F>(
    candidates: &[usize],
    exclude: usize,
    mut sample_index: F,
) -> Option<usize>
where
    F: FnMut(usize) -> usize,
{
    if candidates.is_empty() {
        return None;
    }

    for _ in 0..8 {
        let idx = sample_index(candidates.len());
        if let Some(&candidate) = candidates.get(idx) {
            if candidate != exclude {
                return Some(candidate);
            }
        }
    }

    candidates
        .iter()
        .copied()
        .find(|&candidate| candidate != exclude)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_relation_cardinalities_empty() {
        let triples: Vec<(usize, usize, usize)> = vec![];
        let cards = compute_relation_cardinalities(&triples);
        assert!(cards.is_empty());
    }

    #[test]
    fn compute_relation_cardinalities_one_to_many() {
        // Relation 0: one head (0) maps to three tails (1, 2, 3).
        // tph = 3/1 = 3.0, hpt = 3/3 = 1.0
        // P(corrupt_head) = 3 / (3 + 1) = 0.75
        let triples = vec![(0, 0, 1), (0, 0, 2), (0, 0, 3)];
        let cards = compute_relation_cardinalities(&triples);
        let c = cards.get(&0).expect("relation 0 should be present");
        assert!(
            (c.tph - 3.0).abs() < 1e-6,
            "tph should be 3.0, got {}",
            c.tph
        );
        assert!(
            (c.hpt - 1.0).abs() < 1e-6,
            "hpt should be 1.0, got {}",
            c.hpt
        );
        assert!(
            (c.head_corrupt_prob() - 0.75).abs() < 1e-6,
            "P(corrupt_head) should be 0.75, got {}",
            c.head_corrupt_prob()
        );
    }

    #[test]
    fn compute_relation_cardinalities_many_to_one() {
        // Relation 0: three heads (0, 1, 2) all map to one tail (3).
        // tph = 3/3 = 1.0, hpt = 3/1 = 3.0
        // P(corrupt_head) = 1 / (1 + 3) = 0.25
        let triples = vec![(0, 0, 3), (1, 0, 3), (2, 0, 3)];
        let cards = compute_relation_cardinalities(&triples);
        let c = cards.get(&0).unwrap();
        assert!((c.tph - 1.0).abs() < 1e-6);
        assert!((c.hpt - 3.0).abs() < 1e-6);
        assert!(
            (c.head_corrupt_prob() - 0.25).abs() < 1e-6,
            "P(corrupt_head) should be 0.25, got {}",
            c.head_corrupt_prob()
        );
    }

    #[test]
    fn compute_relation_cardinalities_symmetric() {
        // Relation 0: two heads, two tails, two triples.
        // tph = 2/2 = 1.0, hpt = 2/2 = 1.0
        // P(corrupt_head) = 0.5
        let triples = vec![(0, 0, 1), (1, 0, 0)];
        let cards = compute_relation_cardinalities(&triples);
        let c = cards.get(&0).unwrap();
        assert!((c.head_corrupt_prob() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn compute_relation_cardinalities_multiple_relations() {
        let triples = vec![
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3), // rel 0: 1 head, 3 tails
            (1, 1, 0),
            (2, 1, 0),
            (3, 1, 0), // rel 1: 3 heads, 1 tail
        ];
        let cards = compute_relation_cardinalities(&triples);
        assert_eq!(cards.len(), 2);
        let c0 = cards.get(&0).unwrap();
        let c1 = cards.get(&1).unwrap();
        assert!((c0.tph - 3.0).abs() < 1e-6);
        assert!((c1.hpt - 3.0).abs() < 1e-6);
    }

    #[test]
    fn compute_relation_entity_pools_deduplicates_and_sorts() {
        let triples = vec![(2, 0, 3), (1, 0, 3), (2, 0, 4), (1, 0, 4)];
        let pools = compute_relation_entity_pools(&triples);
        let p = pools.get(&0).expect("relation 0 should exist");
        assert_eq!(p.heads, vec![1, 2]);
        assert_eq!(p.tails, vec![3, 4]);
    }

    #[test]
    fn sample_excluding_skips_target_and_falls_back() {
        let candidates = vec![3, 4, 5];
        let picked = sample_excluding(&candidates, 4, |_| 1).expect("should pick valid candidate");
        assert_ne!(picked, 4);

        let singleton = vec![7];
        assert_eq!(sample_excluding(&singleton, 7, |_| 0), None);
    }

    #[test]
    fn relation_cardinality_head_corrupt_prob_zero_denom() {
        // Edge case: both tph and hpt are 0 (shouldn't happen with real data,
        // but guard against division by zero).
        let c = RelationCardinality { tph: 0.0, hpt: 0.0 };
        assert!((c.head_corrupt_prob() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sample_excluding_empty_returns_none() {
        let empty: Vec<usize> = vec![];
        assert_eq!(sample_excluding(&empty, 0, |_| 0), None);
    }

    #[test]
    fn sample_excluding_all_valid_returns_some() {
        let candidates = vec![1, 2, 3, 4, 5];
        let mut call_count = 0usize;
        let picked = sample_excluding(&candidates, 99, |n| {
            call_count += 1;
            0 % n // always pick first
        });
        assert_eq!(picked, Some(1));
        assert_eq!(call_count, 1); // first attempt succeeds
    }

    #[test]
    fn relation_entity_pools_multiple_relations() {
        let triples = vec![
            (0, 0, 1),
            (0, 0, 2), // rel 0: head={0}, tails={1,2}
            (3, 1, 4),
            (5, 1, 4), // rel 1: heads={3,5}, tail={4}
        ];
        let pools = compute_relation_entity_pools(&triples);
        assert_eq!(pools.len(), 2);
        let p0 = pools.get(&0).unwrap();
        assert_eq!(p0.heads, vec![0]);
        assert_eq!(p0.tails, vec![1, 2]);
        let p1 = pools.get(&1).unwrap();
        assert_eq!(p1.heads, vec![3, 5]);
        assert_eq!(p1.tails, vec![4]);
    }

    #[test]
    fn type_constrained_sampling_draws_from_pool() {
        // Simulate type-constrained sampling: for relation 0, tails are {10, 20, 30}.
        // Sampling with exclude=20 should only return 10 or 30.
        let pool = vec![10, 20, 30];
        let mut rng_idx = 0usize;
        let sequence = [1, 0, 2]; // indices into pool
        let picked = sample_excluding(&pool, 20, |_n| {
            let idx = sequence[rng_idx % sequence.len()];
            rng_idx += 1;
            idx
        });
        // First attempt: pool[1] = 20 (excluded), second: pool[0] = 10 (valid)
        assert_eq!(picked, Some(10));
    }
}
