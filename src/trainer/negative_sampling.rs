use std::collections::{HashMap, HashSet};

#[cfg(feature = "rand")]
use crate::dataset::Triple;
#[cfg(feature = "rand")]
use rand::Rng;

#[cfg(feature = "rand")]
use super::NegativeSamplingStrategy;

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
        let entry = stats.entry(r).or_insert_with(|| (HashSet::new(), HashSet::new(), 0));
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

/// Generate negative samples for a triple.
///
/// # Intuitive Explanation
///
/// Creates "false facts" by corrupting a true triple. For example, if the positive triple is
/// (Paris, located_in, France), negative samples might be:
/// - (Paris, located_in, Germany) - same head/relation, different tail
/// - (Tokyo, located_in, France) - same relation/tail, different head
///
/// **Why this works**: The model learns to assign high scores to true facts and low scores
/// to false facts. By seeing many negative examples, it learns to distinguish plausible from
/// implausible relationships.
///
/// **Example**: If we always corrupt the tail, the model learns "given Paris and 'located_in',
/// which countries make sense?" It learns that France is plausible but Germany is not (for
/// this specific fact).
///
/// # Arguments
///
/// * `triple` - The positive triple (true fact) to corrupt
/// * `entities` - Set of all entities (candidates for corruption)
/// * `strategy` - Which part of the triple to corrupt
/// * `n` - Number of negative samples to generate
///
/// # Returns
///
/// Vector of negative triples (corrupted versions of the positive triple)
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
pub fn generate_negative_samples(
    triple: &Triple,
    entities: &HashSet<String>,
    strategy: &NegativeSamplingStrategy,
    n: usize,
) -> Vec<Triple> {
    let mut rng = rand::rng();
    generate_negative_samples_with_rng(triple, entities, strategy, n, &mut rng)
}

/// Generate negative samples, using a caller-provided RNG.
///
/// This is the deterministic/testing-friendly variant: if you use a seeded RNG, you
/// get reproducible negatives.
///
/// Notes:
/// - Determinism requires a stable iteration order for the candidate pool; this function
///   sorts the `HashSet` into a stable `Vec` before sampling.
/// - The returned vec can be shorter than `n` because we skip samples equal to the
///   positive triple.
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
pub fn generate_negative_samples_with_rng<R: Rng>(
    triple: &Triple,
    entities: &HashSet<String>,
    strategy: &NegativeSamplingStrategy,
    n: usize,
    rng: &mut R,
) -> Vec<Triple> {
    // Determinism note: HashSet iteration order is not stable.
    // We sort to make the sampling pool stable for a fixed entity set.
    //
    // Optimization: avoid cloning the entire entity set per triple. We only allocate
    // the sorted view (Vec<&str>) and only allocate Strings for the sampled entities.
    let pool = SortedEntityPool::new(entities);
    generate_negative_samples_from_sorted_pool_with_rng(triple, &pool, strategy, n, rng)
}

/// A stable, sorted view of an entity set for negative sampling.
///
/// This exists for two reasons:
/// - **Determinism**: `HashSet` iteration order is not stable; sorting fixes that.
/// - **Performance**: avoids cloning every entity ID into an owned `Vec<String>` per triple.
///
/// If you're generating negatives in an inner loop, build this once and reuse it.
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
#[derive(Debug, Clone)]
pub struct SortedEntityPool<'a> {
    pub(crate) entities: Vec<&'a str>,
}

#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
impl<'a> SortedEntityPool<'a> {
    /// Build a stable, sorted pool view from an entity set.
    ///
    /// Determinism note: `HashSet` iteration order is not stable across runs/processes.
    /// Sorting makes the pool stable for a fixed set of entity IDs.
    pub fn new(entities: &'a HashSet<String>) -> Self {
        let mut pool: Vec<&'a str> = entities.iter().map(|s| s.as_str()).collect();
        pool.sort();
        Self { entities: pool }
    }

    #[inline]
    /// Pick a uniformly random entity ID (by index) from the pool.
    pub(crate) fn pick<R: Rng>(&self, rng: &mut R) -> &'a str {
        let idx = rng.random_range(0..self.entities.len());
        self.entities[idx]
    }
}

/// Generate negative samples from a precomputed, sorted pool.
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
pub fn generate_negative_samples_from_sorted_pool_with_rng<R: Rng>(
    triple: &Triple,
    entity_pool: &SortedEntityPool<'_>,
    strategy: &NegativeSamplingStrategy,
    n: usize,
    rng: &mut R,
) -> Vec<Triple> {
    let mut negatives = Vec::with_capacity(n);

    if entity_pool.entities.is_empty() {
        return negatives;
    }

    for _ in 0..n {
        let negative = match strategy {
            NegativeSamplingStrategy::Uniform => {
                if rng.random::<bool>() {
                    Triple {
                        head: entity_pool.pick(rng).to_string(),
                        relation: triple.relation.clone(),
                        tail: triple.tail.clone(),
                    }
                } else {
                    Triple {
                        head: triple.head.clone(),
                        relation: triple.relation.clone(),
                        tail: entity_pool.pick(rng).to_string(),
                    }
                }
            }
            NegativeSamplingStrategy::CorruptHead => Triple {
                head: entity_pool.pick(rng).to_string(),
                relation: triple.relation.clone(),
                tail: triple.tail.clone(),
            },
            NegativeSamplingStrategy::CorruptTail => Triple {
                head: triple.head.clone(),
                relation: triple.relation.clone(),
                tail: entity_pool.pick(rng).to_string(),
            },
            NegativeSamplingStrategy::CorruptBoth => Triple {
                head: entity_pool.pick(rng).to_string(),
                relation: triple.relation.clone(),
                tail: entity_pool.pick(rng).to_string(),
            },
        };

        if negative != *triple {
            negatives.push(negative);
        }
    }

    negatives
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "rand")]
    use proptest::prelude::*;
    #[cfg(feature = "rand")]
    use proptest::proptest;
    #[cfg(feature = "rand")]
    use std::collections::HashSet;

    #[test]
    #[cfg(feature = "rand")]
    fn test_generate_negative_samples() {
        let triple = Triple {
            head: "e1".to_string(),
            relation: "r1".to_string(),
            tail: "e2".to_string(),
        };

        let entities: HashSet<String> = ["e1", "e2", "e3", "e4"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let negatives = generate_negative_samples(
            &triple,
            &entities,
            &NegativeSamplingStrategy::CorruptTail,
            5,
        );

        // May generate fewer than 5 if some negatives match the positive
        // With only 4 entities and CorruptTail, at most 3 unique negatives (e1, e3, e4)
        assert!(
            !negatives.is_empty(),
            "Expected at least 1 negative, got {}",
            negatives.len()
        );
        for neg in negatives {
            assert_eq!(neg.head, "e1");
            assert_eq!(neg.relation, "r1");
            assert_ne!(neg.tail, "e2"); // Should be different from positive
        }
    }

    #[cfg(feature = "rand")]
    proptest! {
        #[test]
        fn prop_generate_negative_samples_with_rng_is_deterministic(seed in any::<u64>()) {
            use rand::SeedableRng;
            use rand::rngs::StdRng;

            let triple = Triple {
                head: "e1".to_string(),
                relation: "r1".to_string(),
                tail: "e2".to_string(),
            };

            let entities: HashSet<String> = ["e1", "e2", "e3", "e4"]
                .iter()
                .map(|s| s.to_string())
                .collect();

            let mut rng1 = StdRng::seed_from_u64(seed);
            let mut rng2 = StdRng::seed_from_u64(seed);

            let a = generate_negative_samples_with_rng(
                &triple,
                &entities,
                &NegativeSamplingStrategy::Uniform,
                25,
                &mut rng1,
            );
            let b = generate_negative_samples_with_rng(
                &triple,
                &entities,
                &NegativeSamplingStrategy::Uniform,
                25,
                &mut rng2,
            );

            prop_assert_eq!(a, b);
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_generate_negative_samples_all_strategies() {
        let triple = Triple {
            head: "e1".to_string(),
            relation: "r1".to_string(),
            tail: "e2".to_string(),
        };

        let entities: HashSet<String> = ["e1", "e2", "e3", "e4", "e5"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Test all strategies
        for strategy in [
            NegativeSamplingStrategy::Uniform,
            NegativeSamplingStrategy::CorruptHead,
            NegativeSamplingStrategy::CorruptTail,
            NegativeSamplingStrategy::CorruptBoth,
        ] {
            let negatives = generate_negative_samples(&triple, &entities, &strategy, 10);
            assert!(
                !negatives.is_empty(),
                "Strategy {:?} should generate negatives",
                strategy
            );

            for neg in &negatives {
                assert_ne!(neg, &triple, "Negative should differ from positive");
            }
        }
    }

    // -----------------------------------------------------------------------
    // SortedEntityPool tests
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "rand")]
    fn sorted_entity_pool_is_sorted_and_stable() {
        let entities: HashSet<String> = ["charlie", "alice", "bob", "delta"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let pool = SortedEntityPool::new(&entities);
        let sorted: Vec<&str> = pool.entities.clone();
        let mut expected = sorted.clone();
        expected.sort();
        assert_eq!(sorted, expected, "pool must be sorted lexicographically");
    }

    #[test]
    #[cfg(feature = "rand")]
    fn sorted_entity_pool_pick_returns_pool_member() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let entities: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let pool = SortedEntityPool::new(&entities);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..20 {
            let picked = pool.pick(&mut rng);
            assert!(
                entities.contains(picked),
                "picked entity '{picked}' not in original set"
            );
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn negative_samples_empty_pool_returns_empty() {
        let entities: HashSet<String> = HashSet::new();
        let triple = Triple {
            head: "h".into(),
            relation: "r".into(),
            tail: "t".into(),
        };
        let negatives = generate_negative_samples(
            &triple,
            &entities,
            &NegativeSamplingStrategy::CorruptTail,
            5,
        );
        assert!(
            negatives.is_empty(),
            "empty entity set should yield no negatives"
        );
    }

    #[test]
    #[cfg(feature = "rand")]
    fn negative_samples_single_entity_may_produce_none() {
        // If the only entity equals the triple's tail, all CorruptTail samples
        // will equal the positive and be filtered out.
        let entities: HashSet<String> = ["t"].iter().map(|s| s.to_string()).collect();
        let triple = Triple {
            head: "h".into(),
            relation: "r".into(),
            tail: "t".into(),
        };
        let negatives = generate_negative_samples(
            &triple,
            &entities,
            &NegativeSamplingStrategy::CorruptTail,
            10,
        );
        assert!(
            negatives.is_empty(),
            "single-entity pool matching the positive tail should yield no negatives"
        );
    }

    #[test]
    #[cfg(feature = "rand")]
    fn negative_samples_corrupt_head_preserves_tail_and_relation() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let entities: HashSet<String> = ["e1", "e2", "e3", "e4"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let triple = Triple {
            head: "e1".into(),
            relation: "r".into(),
            tail: "e2".into(),
        };
        let mut rng = StdRng::seed_from_u64(99);
        let negatives = generate_negative_samples_with_rng(
            &triple,
            &entities,
            &NegativeSamplingStrategy::CorruptHead,
            20,
            &mut rng,
        );
        for neg in &negatives {
            assert_eq!(neg.tail, "e2", "CorruptHead must preserve tail");
            assert_eq!(neg.relation, "r", "CorruptHead must preserve relation");
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn negative_samples_corrupt_both_may_change_head_and_tail() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let entities: HashSet<String> = (0..20).map(|i| format!("e{i}")).collect();
        let triple = Triple {
            head: "e0".into(),
            relation: "r".into(),
            tail: "e1".into(),
        };
        let mut rng = StdRng::seed_from_u64(123);
        let negatives = generate_negative_samples_with_rng(
            &triple,
            &entities,
            &NegativeSamplingStrategy::CorruptBoth,
            50,
            &mut rng,
        );
        let any_head_changed = negatives.iter().any(|n| n.head != "e0");
        let any_tail_changed = negatives.iter().any(|n| n.tail != "e1");
        assert!(
            any_head_changed,
            "CorruptBoth should change head at least sometimes"
        );
        assert!(
            any_tail_changed,
            "CorruptBoth should change tail at least sometimes"
        );
    }

    // -----------------------------------------------------------------------
    // negative sampling statistical uniformity
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "rand")]
    fn negative_sampling_uniformity() {
        use rand::SeedableRng;
        use std::collections::HashMap;

        // Generate 1000 negatives from a pool of 5 entities using CorruptTail.
        // The triple's tail is "e0", so valid replacement tails are e1..e4 (4 entities).
        // Each should appear at least 100 times out of 1000 (loose bound).
        let triple = Triple {
            head: "e0".to_string(),
            relation: "r".to_string(),
            tail: "e0".to_string(),
        };
        let entities: HashSet<String> = (0..5).map(|i| format!("e{i}")).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let negatives = generate_negative_samples_with_rng(
            &triple,
            &entities,
            &NegativeSamplingStrategy::CorruptTail,
            1000,
            &mut rng,
        );

        // Count how many times each tail entity appears.
        let mut counts: HashMap<String, usize> = HashMap::new();
        for neg in &negatives {
            *counts.entry(neg.tail.clone()).or_insert(0) += 1;
        }

        // The positive tail "e0" should never appear (filtered out).
        assert!(
            !counts.contains_key("e0"),
            "positive tail should be filtered: counts={counts:?}"
        );

        // Each of the 4 replacement entities should appear at least 100 times.
        for i in 1..5 {
            let key = format!("e{i}");
            let c = counts.get(&key).copied().unwrap_or(0);
            assert!(
                c >= 100,
                "entity {key} appeared only {c} times out of {} negatives (expected >= 100)",
                negatives.len()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Bernoulli / relation cardinality tests
    // -----------------------------------------------------------------------

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
        assert!((c.tph - 3.0).abs() < 1e-6, "tph should be 3.0, got {}", c.tph);
        assert!((c.hpt - 1.0).abs() < 1e-6, "hpt should be 1.0, got {}", c.hpt);
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
    fn relation_cardinality_head_corrupt_prob_zero_denom() {
        // Edge case: both tph and hpt are 0 (shouldn't happen with real data,
        // but guard against division by zero).
        let c = RelationCardinality { tph: 0.0, hpt: 0.0 };
        assert!((c.head_corrupt_prob() - 0.5).abs() < 1e-6);
    }
}
