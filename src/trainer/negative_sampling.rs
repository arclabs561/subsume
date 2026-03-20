use crate::dataset::Triple;
#[cfg(feature = "rand")]
use rand::Rng;
use std::collections::{HashMap, HashSet};

use super::NegativeSamplingStrategy;

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

/// Generate negative samples weighted by entity degree.
///
/// Entities with higher degree (more training triples) are more likely to
/// be selected as corruptions. Uses the `degree^0.75` distribution from
/// Mikolov et al. (word2vec), adapted for KG negative sampling.
///
/// # Parameters
/// - `positive_triples` - training triples to corrupt
/// - `entity_degrees` - degree (number of triples) per entity ID
/// - `num_negatives` - number of negatives per positive
/// - `rng` - random number generator
///
/// # Returns
///
/// One corrupted triple per positive per `num_negatives`. Corruptions that
/// equal the positive are skipped, so the output can be shorter than
/// `positive_triples.len() * num_negatives`.
///
/// # Reference
///
/// Yang et al. (KDD 2020), "Understanding Negative Sampling in Graph
/// Representation Learning" -- recommends the degree^0.75 smoothing for
/// knowledge graph negative sampling.
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
pub fn generate_degree_weighted_negatives(
    positive_triples: &[(usize, usize, usize)],
    entity_degrees: &HashMap<usize, usize>,
    num_negatives: usize,
    rng: &mut impl rand::Rng,
) -> Vec<(usize, usize, usize)> {
    use rand::distr::weighted::WeightedIndex;
    use rand::distr::Distribution;

    if entity_degrees.is_empty() || positive_triples.is_empty() || num_negatives == 0 {
        return Vec::new();
    }

    // Build sorted entity list for deterministic ordering.
    let mut entities: Vec<usize> = entity_degrees.keys().copied().collect();
    entities.sort_unstable();

    // Compute degree^0.75 weights.
    let weights: Vec<f64> = entities
        .iter()
        .map(|id| {
            let deg = *entity_degrees.get(id).unwrap_or(&1) as f64;
            deg.powf(DEGREE_SMOOTHING_EXPONENT)
        })
        .collect();

    let dist = match WeightedIndex::new(&weights) {
        Ok(d) => d,
        Err(_) => return Vec::new(), // all-zero weights
    };

    let mut negatives = Vec::with_capacity(positive_triples.len() * num_negatives);

    for &(h, r, t) in positive_triples {
        for _ in 0..num_negatives {
            // Corrupt head or tail with equal probability.
            let corrupt_head = rng.random::<bool>();
            let sampled = entities[dist.sample(rng)];

            let neg = if corrupt_head {
                (sampled, r, t)
            } else {
                (h, r, sampled)
            };

            if neg != (h, r, t) {
                negatives.push(neg);
            }
        }
    }

    negatives
}

/// Generate self-adversarial negative samples weighted by model scores.
///
/// For each positive triple `(h, r, t)`, generates `num_negatives` corruptions
/// where harder negatives (those the model currently scores highly) are sampled
/// with higher probability. The sampling weight for a candidate negative
/// `(h, r, t')` is `exp(score(h, r, t') / temperature)`.
///
/// This implements the self-adversarial negative sampling from:
/// Sun et al. (2019), "RotatE: Knowledge Graph Embedding by Relational
/// Rotation in Complex Space" (ICLR 2019), Section 3.2.
///
/// # Parameters
/// - `positive_triples` - training triples to corrupt
/// - `scores_fn` - function mapping `(head_id, tail_id) -> f32` score
///   (higher = model thinks this pair is more likely to be true)
/// - `entity_ids` - all candidate entity IDs for corruption
/// - `temperature` - controls sharpness of adversarial distribution
///   (lower = more focused on hard negatives, higher = more uniform)
/// - `num_negatives` - number of negatives per positive
/// - `rng` - random number generator
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
pub fn generate_self_adversarial_negatives<F>(
    positive_triples: &[(usize, usize, usize)],
    scores_fn: F,
    entity_ids: &[usize],
    temperature: f32,
    num_negatives: usize,
    rng: &mut impl rand::Rng,
) -> Vec<(usize, usize, usize)>
where
    F: Fn(usize, usize) -> f32,
{
    use rand::distr::weighted::WeightedIndex;
    use rand::distr::Distribution;

    if entity_ids.is_empty() || positive_triples.is_empty() || num_negatives == 0 {
        return Vec::new();
    }

    let mut negatives = Vec::with_capacity(positive_triples.len() * num_negatives);

    for &(h, r, t) in positive_triples {
        // Compute adversarial weights for tail corruption candidates.
        let weights: Vec<f64> = entity_ids
            .iter()
            .map(|&candidate| {
                if candidate == t {
                    0.0 // don't sample the true tail
                } else {
                    let score = scores_fn(h, candidate);
                    (score as f64 / temperature as f64).exp()
                }
            })
            .collect();

        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            // Fallback: uniform sampling if all weights are zero.
            for _ in 0..num_negatives {
                let idx = rng.random_range(0..entity_ids.len());
                let candidate = entity_ids[idx];
                if candidate != t {
                    negatives.push((h, r, candidate));
                }
            }
            continue;
        }

        let dist = match WeightedIndex::new(&weights) {
            Ok(d) => d,
            Err(_) => continue,
        };

        for _ in 0..num_negatives {
            let candidate = entity_ids[dist.sample(rng)];
            if candidate != t {
                negatives.push((h, r, candidate));
            }
        }
    }

    negatives
}

/// Degree smoothing exponent for negative sampling (Mikolov et al., 2013).
///
/// Used in [`generate_degree_weighted_negatives`]: each entity's sampling
/// probability is proportional to `degree^0.75`, which down-weights
/// high-degree hub entities relative to pure frequency-based sampling.
#[cfg(feature = "rand")]
const DEGREE_SMOOTHING_EXPONENT: f64 = 0.75;

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
}
