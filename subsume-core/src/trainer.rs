//! Training utilities for box embeddings: negative sampling, loss kernels, and evaluation.
//!
//! ## Research background (minimal but specific)
//!
//! - Bordes et al. (2013): TransE-style negative sampling + margin ranking losses for KGEs.
//! - Vilnis et al. (2018): box lattice measures for probabilistic box embeddings.
//! - Boratko et al. (2020): BoxE (training patterns for box-shaped representations).
//!
//! ## Implementation invariants (why certain choices exist)
//!
//! - **Negative sampling prevents the trivial solution**: without negatives, “everything contains everything”
//!   can score well on positives. Negatives force discrimination.
//! - **Evaluation is deterministic**: ranking metrics are sensitive to tie-handling; we use a deterministic
//!   tie-break so \( \text{same model + same data} \Rightarrow \text{same metrics} \).
//! - **NaNs are treated as hard errors** in evaluation: silently propagating NaNs yields meaningless metrics.

use crate::dataset::Triple;
use crate::optimizer::AMSGradState;
use crate::trainable::TrainableBox;
use crate::training::metrics::{hits_at_k, mean_rank, mean_reciprocal_rank};
use crate::BoxError;
#[cfg(feature = "rand")]
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Negative sampling strategy for training.
///
/// # Research Background
///
/// Negative sampling was introduced in knowledge graph embedding by **Bordes et al. (2013)**
/// for TransE and has become standard practice. The choice of corruption strategy significantly
/// affects what the model learns to distinguish.
///
/// **Reference**: Bordes et al. (2013), "Translating Embeddings for Modeling Multi-relational Data"
///
/// # Intuitive Explanation
///
/// Negative sampling creates "false facts" to contrast with true facts during training.
/// Different strategies work better for different types of knowledge graphs:
///
/// - **CorruptTail**: Replace the tail entity (e.g., (Paris, located_in, ?) → (Paris, located_in, Germany))
///   - Best for: Most knowledge graphs where relations are directional
///   - Why: Head entity often determines what tail makes sense
///
/// - **CorruptHead**: Replace the head entity (e.g., (?, located_in, France) → (Tokyo, located_in, France))
///   - Best for: Relations where tail constrains head (e.g., "part_of")
///   - Why: Some relations work better when we fix the "container" and vary the "contained"
///
/// - **CorruptBoth**: Replace both entities
///   - Best for: Symmetric relations (e.g., "sibling_of", "married_to")
///   - Why: These relations don't have clear head/tail directionality
///
/// - **Uniform**: Randomly corrupt either head or tail
///   - Best for: Balanced datasets or when relation directionality is unclear
///   - Why: Provides diverse negative examples
///
/// **The key insight**: The strategy affects what the model learns to distinguish. CorruptTail
/// teaches "given a head and relation, which tails are plausible?" while CorruptHead teaches
/// "given a tail and relation, which heads are plausible?"
#[derive(Debug, Clone)]
pub enum NegativeSamplingStrategy {
    /// Uniform random sampling (corrupts head or tail randomly)
    Uniform,
    /// Corrupt head entity (fix tail, vary head)
    CorruptHead,
    /// Corrupt tail entity (fix head, vary tail) - most common
    CorruptTail,
    /// Corrupt both (for symmetric relations)
    CorruptBoth,
}

/// Training configuration.
///
/// # Research Background
///
/// Hyperparameter ranges are based on empirical findings from:
/// - **BoxE paper** (Boratko et al., 2020): Learning rates, batch sizes, regularization
/// - **Gumbel-Box papers** (Dasgupta et al., 2020): Temperature scheduling for Gumbel boxes
/// - **Knowledge graph embedding literature**: Standard practices for negative sampling, margins
///
/// **Key References**:
/// - Boratko et al. (2020): "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS)
/// - Dasgupta et al. (2020): "Improving Local Identifiability in Probabilistic Box Embeddings" (NeurIPS)
///
/// # Intuitive Guide to Hyperparameters
///
/// These parameters control how the model learns box embeddings from knowledge graphs.
/// Understanding what each does helps you tune for your specific dataset.
///
/// ## Core Learning Parameters
///
/// - **`learning_rate`**: How big steps the optimizer takes
///   - Too high: Model overshoots optimal box positions (unstable training)
///   - Too low: Model learns very slowly (wastes compute)
///   - Sweet spot: Usually 1e-3 to 5e-4 for box embeddings
///
/// - **`batch_size`**: How many triples processed together
///   - Larger: More stable gradients, faster training, but needs more memory
///   - Smaller: More noisy gradients, slower training, but less memory
///   - Sweet spot: 512-2048 for most knowledge graphs
///
/// - **`epochs`**: How many times to see the training data
///   - Too few: Model doesn't learn enough (underfitting)
///   - Too many: Model memorizes training data (overfitting)
///   - Use early stopping to find the right number automatically
///
/// ## Negative Sampling Parameters
///
/// - **`negative_samples`**: How many false facts per true fact
///   - More negatives: Model learns better discrimination, but slower training
///   - Fewer negatives: Faster training, but may not learn fine distinctions
///   - Common: 1-5 negatives per positive
///
/// - **`negative_strategy`**: Which part of triple to corrupt (see [`NegativeSamplingStrategy`])
///
/// ## Regularization Parameters
///
/// - **`regularization_weight`**: Penalty for boxes being too large
///   - Prevents boxes from growing unbounded (which would make everything contain everything)
///   - Higher: Tighter, more specific boxes
///   - Lower: Larger, more general boxes
///   - Common: 0.01-0.1
///
/// - **`weight_decay`**: L2 regularization on box parameters
///   - Prevents overfitting by keeping box coordinates small
///   - Higher: Stronger regularization (simpler model)
///   - Lower: Less regularization (more complex model)
///
/// ## Box-Specific Parameters
///
/// - **`temperature`**: Controls "softness" of Gumbel box boundaries
///   - Lower (0.1-0.5): Sharp boundaries, more like hard boxes
///   - Higher (1.0-2.0): Soft boundaries, smoother gradients
///   - Can be scheduled: Start high, decrease during training
///
/// - **`margin`**: Minimum score difference between positive and negative triples
///   - Higher: Forces stronger separation (better discrimination)
///   - Lower: Allows closer scores (may be easier to optimize)
///   - Common: 0.5-2.0
///
/// ## Training Control
///
/// - **`early_stopping_patience`**: Stop if validation doesn't improve for N epochs
///   - Prevents overfitting automatically
///   - None: Train for all epochs (may overfit)
///   - Some(10): Stop if no improvement for 10 epochs
///
/// # Mathematical Relationships
///
/// The total loss combines multiple terms:
///
/// \[
/// L_{\text{total}} = L_{\text{ranking}} + \lambda_{\text{reg}} \cdot L_{\text{volume}} + \lambda_{\text{wd}} \cdot ||\theta||^2
/// \]
///
/// where:
/// - \(L_{\text{ranking}}\) is the margin-based ranking loss
/// - \(L_{\text{volume}}\) is volume regularization (penalizes large boxes)
/// - \(||\theta||^2\) is L2 regularization on parameters
/// - \(\lambda_{\text{reg}}\) is `regularization_weight`
/// - \(\lambda_{\text{wd}}\) is `weight_decay`
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate (default: 1e-3, paper range: 1e-3 to 5e-4)
    pub learning_rate: f32,
    /// Number of training epochs (default: 100)
    pub epochs: usize,
    /// Batch size (default: 512, paper range: 512-4096)
    pub batch_size: usize,
    /// Number of negative samples per positive (default: 1)
    pub negative_samples: usize,
    /// Negative sampling strategy (default: CorruptTail)
    pub negative_strategy: NegativeSamplingStrategy,
    /// Regularization weight (default: 0.01)
    pub regularization_weight: f32,
    /// Temperature for Gumbel boxes (default: 1.0)
    pub temperature: f32,
    /// Weight decay for AdamW (default: 1e-5, paper range: 1e-5 to 1e-3)
    pub weight_decay: f32,
    /// Margin for ranking loss (default: 1.0)
    pub margin: f32,
    /// Early stopping patience (default: Some(10))
    pub early_stopping_patience: Option<usize>,
    /// Minimum improvement for early stopping (relative)
    pub early_stopping_min_delta: f32,
    /// Use self-adversarial negative sampling
    pub use_self_adversarial: bool,
    /// Temperature for self-adversarial sampling
    pub adversarial_temperature: f32,
    /// Multi-stage training: focus on positives first (epochs), then negatives
    pub positive_focus_epochs: Option<usize>,
    /// L2 regularization weight
    pub regularization: f32,
    /// Warmup epochs
    pub warmup_epochs: usize,

    /// Weight on the "negative" loss term (internal trainer loop).
    ///
    /// This does not affect `evaluate_link_prediction`; it only scales the margin term in
    /// `compute_pair_loss` for negative examples.
    pub negative_weight: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3, // Paper default: 1e-3 to 5e-4
            epochs: 100,
            batch_size: 512, // Paper range: 512-4096
            negative_samples: 1,
            negative_strategy: NegativeSamplingStrategy::CorruptTail,
            regularization_weight: 0.01,
            temperature: 1.0,
            weight_decay: 1e-5,                // Paper range: 1e-5 to 1e-3
            margin: 1.0,                       // Margin for ranking loss
            early_stopping_patience: Some(10), // Early stopping after 10 epochs without improvement
            early_stopping_min_delta: 0.001,
            use_self_adversarial: true,
            adversarial_temperature: 1.0,
            positive_focus_epochs: None,
            regularization: 0.0001,
            warmup_epochs: 10,
            negative_weight: 1.0,
        }
    }
}

/// Evaluation results for link prediction.
#[derive(Debug, Clone)]
pub struct EvaluationResults {
    /// Mean Reciprocal Rank
    pub mrr: f32,
    /// Hits@1
    pub hits_at_1: f32,
    /// Hits@3
    pub hits_at_3: f32,
    /// Hits@10
    pub hits_at_10: f32,
    /// Mean Rank
    pub mean_rank: f32,
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
#[cfg(feature = "rand")]
#[allow(deprecated)]
pub fn generate_negative_samples(
    triple: &Triple,
    entities: &HashSet<String>,
    strategy: &NegativeSamplingStrategy,
    n: usize,
) -> Vec<Triple> {
    let mut rng = rand::thread_rng();
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
#[cfg(feature = "rand")]
#[derive(Debug, Clone)]
pub struct SortedEntityPool<'a> {
    entities: Vec<&'a str>,
}

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
    #[allow(deprecated)]
    fn pick<R: Rng>(&self, rng: &mut R) -> &'a str {
        let idx = rng.gen_range(0..self.entities.len());
        self.entities[idx]
    }
}

/// Generate negative samples from a precomputed, sorted pool.
#[cfg(feature = "rand")]
#[allow(deprecated)]
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
                if rng.gen::<bool>() {
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

/// Generate negative samples from an explicit candidate entity pool.
///
/// This is the “integration seam”: callers can restrict the pool to hard negatives
/// (e.g., graph neighborhood candidates), while `subsume-core` stays dependency-free.
#[cfg(feature = "rand")]
#[allow(deprecated)]
pub fn generate_negative_samples_from_pool_with_rng<R: Rng>(
    triple: &Triple,
    entity_pool: &[String],
    strategy: &NegativeSamplingStrategy,
    n: usize,
    rng: &mut R,
) -> Vec<Triple> {
    let mut negatives = Vec::with_capacity(n);

    if entity_pool.is_empty() {
        return negatives;
    }

    for _ in 0..n {
        let negative = match strategy {
            NegativeSamplingStrategy::Uniform => {
                if rng.gen::<bool>() {
                    let head = entity_pool[rng.gen_range(0..entity_pool.len())].clone();
                    Triple { head, relation: triple.relation.clone(), tail: triple.tail.clone() }
                } else {
                    let tail = entity_pool[rng.gen_range(0..entity_pool.len())].clone();
                    Triple { head: triple.head.clone(), relation: triple.relation.clone(), tail }
                }
            }
            NegativeSamplingStrategy::CorruptHead => {
                let head = entity_pool[rng.gen_range(0..entity_pool.len())].clone();
                Triple { head, relation: triple.relation.clone(), tail: triple.tail.clone() }
            }
            NegativeSamplingStrategy::CorruptTail => {
                let tail = entity_pool[rng.gen_range(0..entity_pool.len())].clone();
                Triple { head: triple.head.clone(), relation: triple.relation.clone(), tail }
            }
            NegativeSamplingStrategy::CorruptBoth => {
                let head = entity_pool[rng.gen_range(0..entity_pool.len())].clone();
                let tail = entity_pool[rng.gen_range(0..entity_pool.len())].clone();
                Triple { head, relation: triple.relation.clone(), tail }
            }
        };

        if negative != *triple {
            negatives.push(negative);
        }
    }

    negatives
}

/// An index of “known true” triples for filtered link prediction evaluation.
///
/// In standard KGE evaluation (e.g. FB15k-237, WN18RR), **filtered ranking** removes any
/// candidate that is already a true triple in train/valid/test, except for the test triple
/// being evaluated. This avoids penalizing the model for ranking other true answers above
/// the held-out one.
///
/// This index is intentionally shaped for the most common evaluation query we currently
/// support in `subsume-core`:
/// - tail prediction: \((h, r, ?)\)
///
/// Notes:
/// - Building this index **allocates**, but using it during evaluation does not.
/// - Memory can be large for big KGs; prefer using `FilteredTripleIndexIds` with interned IDs.
#[derive(Debug, Default, Clone)]
pub struct FilteredTripleIndex {
    tails_by_head_rel: HashMap<String, HashMap<String, HashSet<String>>>,
}

impl FilteredTripleIndex {
    /// Build a filtered-ranking index from an iterator of triples.
    pub fn from_triples<'a, I>(triples: I) -> Self
    where
        I: IntoIterator<Item = &'a Triple>,
    {
        let mut index = Self::default();
        index.extend(triples);
        index
    }

    /// Build a filtered-ranking index from an iterator of **owned** triples.
    ///
    /// This avoids cloning `(head, relation, tail)` strings, which can matter when you're
    /// building an index for a one-shot evaluation and you don't need to retain the original
    /// triple list.
    pub fn from_owned_triples<I>(triples: I) -> Self
    where
        I: IntoIterator<Item = Triple>,
    {
        let mut index = Self::default();
        index.extend_owned(triples);
        index
    }

    /// Extend the index with more known-true triples.
    pub fn extend<'a, I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = &'a Triple>,
    {
        for t in triples {
            self.tails_by_head_rel
                .entry(t.head.clone())
                .or_default()
                .entry(t.relation.clone())
                .or_default()
                .insert(t.tail.clone());
        }
    }

    /// Extend the index with more known-true triples (owned).
    pub fn extend_owned<I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = Triple>,
    {
        for t in triples {
            self.tails_by_head_rel
                .entry(t.head)
                .or_default()
                .entry(t.relation)
                .or_default()
                .insert(t.tail);
        }
    }

    /// True iff `(head, relation, tail)` is a known true triple.
    #[inline]
    pub fn is_known_tail(&self, head: &str, relation: &str, tail: &str) -> bool {
        self.tails_by_head_rel
            .get(head)
            .and_then(|by_rel| by_rel.get(relation))
            .is_some_and(|tails| tails.contains(tail))
    }

    /// Return all known-true tails for the query \((head, relation, ?)\).
    #[inline]
    pub fn known_tails(&self, head: &str, relation: &str) -> Option<&HashSet<String>> {
        self.tails_by_head_rel
            .get(head)
            .and_then(|by_rel| by_rel.get(relation))
    }
}

/// Like [`FilteredTripleIndex`], but for interned integer IDs.
///
/// This is the preferred form for performance-sensitive evaluation, because it avoids
/// hashing/cloning `String` IDs in the hot loop.
#[derive(Debug, Default, Clone)]
pub struct FilteredTripleIndexIds {
    tails_by_head_rel: HashMap<usize, HashMap<usize, HashSet<usize>>>,
}

impl FilteredTripleIndexIds {
    /// Build a filtered-ranking index from an iterator of ID triples.
    pub fn from_triples<'a, I>(triples: I) -> Self
    where
        I: IntoIterator<Item = &'a crate::dataset::TripleIds>,
    {
        let mut index = Self::default();
        index.extend(triples);
        index
    }

    /// Extend the index with more known-true triples.
    pub fn extend<'a, I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = &'a crate::dataset::TripleIds>,
    {
        for t in triples {
            self.tails_by_head_rel
                .entry(t.head)
                .or_default()
                .entry(t.relation)
                .or_default()
                .insert(t.tail);
        }
    }

    /// True iff `(head, relation, tail)` is a known true triple.
    #[inline]
    pub fn is_known_tail(&self, head: usize, relation: usize, tail: usize) -> bool {
        self.tails_by_head_rel
            .get(&head)
            .and_then(|by_rel| by_rel.get(&relation))
            .is_some_and(|tails| tails.contains(&tail))
    }

    /// Return all known-true tails for the query \((head, relation, ?)\).
    #[inline]
    pub fn known_tails(&self, head: usize, relation: usize) -> Option<&HashSet<usize>> {
        self.tails_by_head_rel
            .get(&head)
            .and_then(|by_rel| by_rel.get(&relation))
    }
}

fn evaluate_link_prediction_inner<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    _relation_boxes: Option<&HashMap<String, B>>,
    filter: Option<&FilteredTripleIndex>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    let mut ranks = Vec::new();

    for triple in test_triples {
        // Get head and relation boxes
        let head_box = entity_boxes
            .get(&triple.head)
            .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity: {}", triple.head)))?;

        // For simplicity, use head box directly (can be extended for relation-specific boxes)
        let query_box = head_box;

        // Optimization: compute the rank in O(|E|) without sorting the full candidate list.
        //
        // We compute:
        // - `tail_score` (if the tail exists in the candidate set)
        // - count how many candidates have strictly greater score
        // - deterministic tie-break: among equal scores, rank by lexicographic entity id
        //
        // This avoids O(|E| log |E|) sorting work and avoids cloning entity IDs.
        let tail_score = match entity_boxes.get(&triple.tail) {
            Some(tail_box) => {
                let s = query_box.containment_prob_fast(tail_box, 1.0)?;
                if s.is_nan() {
                    return Err(crate::BoxError::Internal(
                        "NaN containment score encountered (tail)".to_string(),
                    ));
                }
                Some(s)
            }
            None => None,
        };

        // If the target tail isn't in the candidate set, the rank is undefined for this query.
        // (We preserve the old behavior: use usize::MAX, which drives MRR → 0 and MR → large.)
        let Some(tail_score) = tail_score else {
            ranks.push(usize::MAX);
            continue;
        };

        let mut better = 0usize;
        let mut tie_before = 0usize;
        for (entity, box_) in entity_boxes {
            if entity == &triple.tail {
                continue;
            }

            let score = query_box.containment_prob_fast(box_, 1.0)?;
            if score.is_nan() {
                return Err(crate::BoxError::Internal(
                    "NaN containment score encountered".to_string(),
                ));
            }

            if score > tail_score {
                better += 1;
            } else if score == tail_score && entity.as_str() < triple.tail.as_str() {
                tie_before += 1;
            }
        }

        // Filtered ranking optimization:
        //
        // Naively, filtered ranking would do a `HashSet::contains` check per candidate, which
        // is often the dominant cost. Instead:
        // 1) compute the unfiltered counts (better/tie_before) in one linear pass
        // 2) subtract the contribution of the (usually small) set of filtered candidates
        //
        // This keeps the hot loop branch-free w.r.t. filtering.
        if let Some(filter) = filter {
            if let Some(known_tails) = filter.known_tails(&triple.head, &triple.relation) {
                let mut filtered_better = 0usize;
                let mut filtered_tie_before = 0usize;

                for known_tail in known_tails {
                    if known_tail == &triple.tail {
                        continue;
                    }

                    // Only subtract candidates that are actually in the candidate set.
                    let Some(box_) = entity_boxes.get(known_tail) else {
                        continue;
                    };

                    let score = query_box.containment_prob_fast(box_, 1.0)?;
                    if score.is_nan() {
                        return Err(crate::BoxError::Internal(
                            "NaN containment score encountered".to_string(),
                        ));
                    }

                    if score > tail_score {
                        filtered_better += 1;
                    } else if score == tail_score && known_tail.as_str() < triple.tail.as_str() {
                        filtered_tie_before += 1;
                    }
                }

                better = better.saturating_sub(filtered_better);
                tie_before = tie_before.saturating_sub(filtered_tie_before);
            }
        }

        let rank = better + tie_before + 1;

        ranks.push(rank);
    }

    let mrr = mean_reciprocal_rank(ranks.iter().copied());
    let hits_at_1 = hits_at_k(ranks.iter().copied(), 1);
    let hits_at_3 = hits_at_k(ranks.iter().copied(), 3);
    let hits_at_10 = hits_at_k(ranks.iter().copied(), 10);
    let mean_rank = mean_rank(ranks.iter().copied());

    Ok(EvaluationResults {
        mrr,
        hits_at_1,
        hits_at_3,
        hits_at_10,
        mean_rank,
    })
}

fn evaluate_link_prediction_interned_inner<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    _relation_boxes: Option<&[B]>,
    filter: Option<&FilteredTripleIndexIds>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    let mut ranks = Vec::new();

    for triple in test_triples {
        let head_box = entity_boxes.get(triple.head).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (head): {}", triple.head))
        })?;
        let query_box = head_box;

        let tail_box = match entity_boxes.get(triple.tail) {
            Some(t) => t,
            None => {
                ranks.push(usize::MAX);
                continue;
            }
        };

        let tail_name = entities.get(triple.tail).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity label (tail): {}", triple.tail))
        })?;

        let tail_score = query_box.containment_prob_fast(tail_box, 1.0)?;
        if tail_score.is_nan() {
            return Err(crate::BoxError::Internal(
                "NaN containment score encountered (tail)".to_string(),
            ));
        }

        let mut better = 0usize;
        let mut tie_before = 0usize;
        for (entity_id, box_) in entity_boxes.iter().enumerate() {
            if entity_id == triple.tail {
                continue;
            }
            let score = query_box.containment_prob_fast(box_, 1.0)?;
            if score.is_nan() {
                return Err(crate::BoxError::Internal(
                    "NaN containment score encountered".to_string(),
                ));
            }

            if score > tail_score {
                better += 1;
            } else if score == tail_score {
                let name = entities.get(entity_id).ok_or_else(|| {
                    crate::BoxError::Internal(format!(
                        "Missing entity label (candidate): {}",
                        entity_id
                    ))
                })?;
                if name < tail_name {
                    tie_before += 1;
                }
            }
        }

        if let Some(filter) = filter {
            if let Some(known_tails) = filter.known_tails(triple.head, triple.relation) {
                let mut filtered_better = 0usize;
                let mut filtered_tie_before = 0usize;

                for &known_tail in known_tails {
                    if known_tail == triple.tail {
                        continue;
                    }
                    let Some(box_) = entity_boxes.get(known_tail) else {
                        continue;
                    };

                    let score = query_box.containment_prob_fast(box_, 1.0)?;
                    if score.is_nan() {
                        return Err(crate::BoxError::Internal(
                            "NaN containment score encountered".to_string(),
                        ));
                    }

                    if score > tail_score {
                        filtered_better += 1;
                    } else if score == tail_score {
                        let name = entities.get(known_tail).ok_or_else(|| {
                            crate::BoxError::Internal(format!(
                                "Missing entity label (filtered): {}",
                                known_tail
                            ))
                        })?;
                        if name < tail_name {
                            filtered_tie_before += 1;
                        }
                    }
                }

                better = better.saturating_sub(filtered_better);
                tie_before = tie_before.saturating_sub(filtered_tie_before);
            }
        }

        ranks.push(better + tie_before + 1);
    }

    let mrr = mean_reciprocal_rank(ranks.iter().copied());
    let hits_at_1 = hits_at_k(ranks.iter().copied(), 1);
    let hits_at_3 = hits_at_k(ranks.iter().copied(), 3);
    let hits_at_10 = hits_at_k(ranks.iter().copied(), 10);
    let mean_rank = mean_rank(ranks.iter().copied());

    Ok(EvaluationResults {
        mrr,
        hits_at_1,
        hits_at_3,
        hits_at_10,
        mean_rank,
    })
}

/// Evaluate link prediction performance.
///
/// # Research Background
///
/// Link prediction evaluation follows the standard protocol established by **Bordes et al. (2013)**
/// for TransE and used consistently across knowledge graph embedding literature. The metrics
/// (MRR, Hits@K, Mean Rank) are standard benchmarks for knowledge graph completion.
///
/// **Reference**: Bordes et al. (2013), "Translating Embeddings for Modeling Multi-relational Data"
///
/// # Intuitive Explanation
///
/// Link prediction is the core task: given (head, relation, ?), predict which tail entity
/// completes the triple. This function evaluates how well the model does this.
///
/// **The process**:
/// 1. For each test triple (e.g., (Paris, located_in, ?))
/// 2. Score all possible tail entities using containment probability
/// 3. Rank them by score (highest = most likely)
/// 4. Check where the correct answer (France) appears in the ranking
///
/// **Metrics computed**:
/// - **MRR (Mean Reciprocal Rank)**: Average of 1/rank for correct answers
///   - If correct answer is rank 1 → 1/1 = 1.0 (perfect)
///   - If correct answer is rank 5 → 1/5 = 0.2
///   - Higher is better (range: 0 to 1)
///
/// - **Hits@K**: Fraction of queries where correct answer is in top K
///   - Hits@10 = 0.8 means 80% of queries have correct answer in top 10
///   - Higher is better (range: 0 to 1)
///
/// - **Mean Rank**: Average position of correct answers
///   - Lower is better (best = 1.0, worst = number of entities)
///
/// **Why this matters**: These metrics tell you if the model learned meaningful geometric
/// relationships. High MRR means boxes are arranged so containment probabilities match
/// knowledge graph structure.
///
/// # Arguments
///
/// * `test_triples` - Test set triples (held-out true facts)
/// * `entity_boxes` - Map from entity ID to box embedding
/// * `relation_boxes` - Map from relation ID to box embedding (optional, for relation-specific boxes)
///
/// # Returns
///
/// Evaluation results with MRR, Hits@K, Mean Rank
///
/// # Note
///
/// This function requires `B::Scalar = f32`. For other scalar types, use backend-specific evaluation functions.
pub fn evaluate_link_prediction<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    _relation_boxes: Option<&HashMap<String, B>>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_inner(test_triples, entity_boxes, _relation_boxes, None)
}

/// Evaluate link prediction in the **filtered** setting.
///
/// Filtered ranking excludes any candidate tail \(t'\) such that \((h, r, t')\) is known true
/// (in train/valid/test), except for the test triple’s own tail.
pub fn evaluate_link_prediction_filtered<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    _relation_boxes: Option<&HashMap<String, B>>,
    filter: &FilteredTripleIndex,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_inner(test_triples, entity_boxes, _relation_boxes, Some(filter))
}

/// Evaluate link prediction using interned IDs (`usize`) for entities/relations.
///
/// This avoids string hashing/cloning in the candidate loop, which is often the dominant
/// overhead once the scoring kernel itself is optimized.
pub fn evaluate_link_prediction_interned<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    _relation_boxes: Option<&[B]>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_interned_inner(test_triples, entity_boxes, entities, _relation_boxes, None)
}

/// Evaluate link prediction in the **filtered** setting, using interned IDs.
pub fn evaluate_link_prediction_interned_filtered<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    _relation_boxes: Option<&[B]>,
    filter: &FilteredTripleIndexIds,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_interned_inner(
        test_triples,
        entity_boxes,
        entities,
        _relation_boxes,
        Some(filter),
    )
}

/// Training result with metrics and history.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final evaluation results
    pub final_results: EvaluationResults,
    /// Training loss history
    pub loss_history: Vec<f32>,
    /// Validation MRR history
    pub validation_mrr_history: Vec<f32>,
    /// Best epoch (based on validation MRR)
    pub best_epoch: usize,
    /// Total training time (if tracked)
    pub training_time_seconds: Option<f64>,
}

/// Hyperparameter search configuration.
#[derive(Debug, Clone)]
pub struct HyperparameterSearch {
    /// Learning rates to try
    pub learning_rates: Vec<f32>,
    /// Batch sizes to try
    pub batch_sizes: Vec<usize>,
    /// Embedding dimensions to try
    pub dimensions: Vec<usize>,
    /// Regularization weights to try
    pub regularization_weights: Vec<f32>,
    /// Number of trials per combination
    pub trials_per_config: usize,
}

impl Default for HyperparameterSearch {
    fn default() -> Self {
        Self {
            learning_rates: vec![1e-3, 5e-4, 1e-4],
            batch_sizes: vec![512, 1024, 2048],
            dimensions: vec![50, 100, 200],
            regularization_weights: vec![1e-5, 1e-4, 1e-3],
            trials_per_config: 1,
        }
    }
}

/// Log training results to file or stdout.
pub fn log_training_result(result: &TrainingResult, path: Option<&str>) -> Result<(), BoxError> {
    let output = format!(
        "Training Results\n\
         ===============\n\
         Final MRR: {:.4}\n\
         Final Hits@1: {:.4}\n\
         Final Hits@3: {:.4}\n\
         Final Hits@10: {:.4}\n\
         Final Mean Rank: {:.2}\n\
         Best Epoch: {}\n\
         Training Time: {:.2}s\n",
        result.final_results.mrr,
        result.final_results.hits_at_1,
        result.final_results.hits_at_3,
        result.final_results.hits_at_10,
        result.final_results.mean_rank,
        result.best_epoch,
        result.training_time_seconds.unwrap_or(0.0)
    );

    if let Some(p) = path {
        std::fs::write(p, output).map_err(|e| BoxError::Internal(e.to_string()))?;
    } else {
        println!("{}", output);
    }

    Ok(())
}

/// Trainer for box embeddings.
pub struct BoxEmbeddingTrainer {
    /// Training configuration
    pub config: TrainingConfig,
    /// Entity ID → TrainableBox mapping
    pub boxes: HashMap<usize, TrainableBox>,
    /// Entity ID → AMSGradState mapping
    pub optimizer_states: HashMap<usize, AMSGradState>,
    /// Embedding dimension
    pub dim: usize,
}

impl BoxEmbeddingTrainer {
    /// Create a new trainer.
    pub fn new(
        config: TrainingConfig,
        dim: usize,
        initial_embeddings: Option<HashMap<usize, Vec<f32>>>,
    ) -> Self {
        let mut boxes = HashMap::new();
        let mut optimizer_states = HashMap::new();

        if let Some(embeddings) = initial_embeddings {
            for (entity_id, vector) in embeddings {
                assert_eq!(vector.len(), dim);
                let box_embedding = TrainableBox::from_vector(&vector, 0.1);
                boxes.insert(entity_id, box_embedding.clone());
                optimizer_states.insert(entity_id, AMSGradState::new(dim, config.learning_rate));
            }
        }

        Self {
            config,
            boxes,
            optimizer_states,
            dim,
        }
    }

    /// Update boxes using analytical gradients.
    pub fn train_step(&mut self, id_a: usize, id_b: usize, is_positive: bool) -> f32 {
        let box_a = self.boxes.get(&id_a).cloned();
        let box_b = self.boxes.get(&id_b).cloned();

        if let (Some(box_a_ref), Some(box_b_ref)) = (box_a.as_ref(), box_b.as_ref()) {
            let loss = compute_pair_loss(box_a_ref, box_b_ref, is_positive, &self.config);
            let (grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b) =
                compute_analytical_gradients(box_a_ref, box_b_ref, is_positive, &self.config);

            if let (Some(box_a_mut), Some(state_a)) = (self.boxes.get_mut(&id_a), self.optimizer_states.get_mut(&id_a)) {
                box_a_mut.update_amsgrad(&grad_mu_a, &grad_delta_a, state_a);
            }
            if let (Some(box_b_mut), Some(state_b)) = (self.boxes.get_mut(&id_b), self.optimizer_states.get_mut(&id_b)) {
                box_b_mut.update_amsgrad(&grad_mu_b, &grad_delta_b, state_b);
            }
            loss
        } else {
            0.0
        }
    }
}

/// Compute loss for a pair of boxes.
///
/// Design choice (important):
/// - For **positive** examples this loss uses a *symmetric* score by taking
///   \(\min(P(B \subseteq A),\; P(A \subseteq B))\). This encourages “near-equivalence”
///   more than directed entailment.
/// - For hierarchy-like relations, a more typical objective is *directed* containment,
///   e.g. minimize \(-\ln P(B \subseteq A)\) only.
pub fn compute_pair_loss(
    box_a: &TrainableBox,
    box_b: &TrainableBox,
    is_positive: bool,
    config: &TrainingConfig,
) -> f32 {
    let box_a_embed = box_a.to_box();
    let box_b_embed = box_b.to_box();

    if is_positive {
        let p_a_b = box_a_embed.conditional_probability(&box_b_embed).max(1e-8);
        let p_b_a = box_b_embed.conditional_probability(&box_a_embed).max(1e-8);
        let min_prob = p_a_b.min(p_b_a);
        let neg_log_prob = -min_prob.ln();

        let vol_a = box_a_embed.volume();
        let vol_b = box_b_embed.volume();
        let reg = config.regularization * (vol_a + vol_b);

        (neg_log_prob + reg).max(0.0)
    } else {
        let p_a_b = box_a_embed.conditional_probability(&box_b_embed);
        let p_b_a = box_b_embed.conditional_probability(&box_a_embed);
        let max_prob = p_a_b.max(p_b_a);

        let margin_loss = if max_prob > config.margin {
            (max_prob - config.margin).powi(2)
        } else {
            0.0
        };

        config.negative_weight * margin_loss
    }
}

/// Compute analytical gradients for a pair of boxes.
pub fn compute_analytical_gradients(
    box_a: &TrainableBox,
    box_b: &TrainableBox,
    is_positive: bool,
    _config: &TrainingConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let box_a_embed = box_a.to_box();
    let box_b_embed = box_b.to_box();
    let dim = box_a.dim;

    let mut grad_mu_a = vec![0.0; dim];
    let mut grad_delta_a = vec![0.0; dim];
    let mut grad_mu_b = vec![0.0; dim];
    let mut grad_delta_b = vec![0.0; dim];

    let _vol_a = box_a_embed.volume();
    let _vol_b = box_b_embed.volume();
    let vol_intersection = box_a_embed.intersection_volume(&box_b_embed);

    if is_positive {
        for i in 0..dim {
            if vol_intersection > 1e-10 {
                grad_delta_a[i] -= 0.1; 
                grad_delta_b[i] -= 0.1;
            } else {
                let diff = box_b_embed.min[i] + box_b_embed.max[i] - (box_a_embed.min[i] + box_a_embed.max[i]);
                grad_mu_a[i] -= 0.5 * diff;
                grad_mu_b[i] += 0.5 * diff;
            }
        }
    }

    (grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "rand")]
    use std::collections::HashSet;
    #[cfg(feature = "rand")]
    use proptest::prelude::*;
    #[cfg(feature = "rand")]
    use proptest::proptest;

    #[test]
    fn compute_pair_loss_positive_prefers_containment_over_disjoint() {
        let cfg = TrainingConfig::default();

        // A: large box around origin
        let a = TrainableBox::new(vec![0.0, 0.0], vec![2.0_f32.ln(), 2.0_f32.ln()]);
        // B_in: small box centered at origin (contained)
        let b_in = TrainableBox::new(vec![0.0, 0.0], vec![0.2_f32.ln(), 0.2_f32.ln()]);
        // B_out: same size but far away (disjoint-ish)
        let b_out = TrainableBox::new(vec![100.0, 100.0], vec![0.2_f32.ln(), 0.2_f32.ln()]);

        let l_in = compute_pair_loss(&a, &b_in, true, &cfg);
        let l_out = compute_pair_loss(&a, &b_out, true, &cfg);

        assert!(l_in.is_finite() && l_out.is_finite());
        assert!(
            l_in < l_out,
            "positive loss should be lower for contained boxes (got l_in={l_in}, l_out={l_out})"
        );
    }

    #[test]
    fn compute_pair_loss_negative_penalizes_overlap_above_margin() {
        let mut cfg = TrainingConfig::default();
        cfg.margin = 0.2;
        cfg.negative_weight = 1.0;

        // A fixed box; compare B disjoint vs B overlapping.
        let a = TrainableBox::new(vec![0.0, 0.0], vec![1.0_f32.ln(), 1.0_f32.ln()]);
        let b_disjoint = TrainableBox::new(vec![100.0, 100.0], vec![1.0_f32.ln(), 1.0_f32.ln()]);
        let b_overlap = TrainableBox::new(vec![0.0, 0.0], vec![1.0_f32.ln(), 1.0_f32.ln()]);

        let l_disjoint = compute_pair_loss(&a, &b_disjoint, false, &cfg);
        let l_overlap = compute_pair_loss(&a, &b_overlap, false, &cfg);

        assert!(l_disjoint.is_finite() && l_overlap.is_finite());
        assert!(
            l_overlap >= l_disjoint,
            "negative loss should not decrease when overlap increases (got disjoint={l_disjoint}, overlap={l_overlap})"
        );
    }

    #[test]
    fn filtered_triple_index_membership() {
        let triples = vec![
            Triple {
                head: "h".to_string(),
                relation: "r".to_string(),
                tail: "t1".to_string(),
            },
            Triple {
                head: "h".to_string(),
                relation: "r".to_string(),
                tail: "t2".to_string(),
            },
            Triple {
                head: "h".to_string(),
                relation: "r2".to_string(),
                tail: "t3".to_string(),
            },
        ];

        let idx = FilteredTripleIndex::from_triples(triples.iter());

        assert!(idx.is_known_tail("h", "r", "t1"));
        assert!(idx.is_known_tail("h", "r", "t2"));
        assert!(!idx.is_known_tail("h", "r", "t3"));
        assert!(idx.is_known_tail("h", "r2", "t3"));
        assert!(!idx.is_known_tail("missing", "r", "t1"));
    }

    #[test]
    fn filtered_triple_index_from_owned_triples_avoids_cloning() {
        let triples = vec![
            Triple {
                head: "h".to_string(),
                relation: "r".to_string(),
                tail: "t1".to_string(),
            },
            Triple {
                head: "h".to_string(),
                relation: "r".to_string(),
                tail: "t2".to_string(),
            },
        ];

        let idx = FilteredTripleIndex::from_owned_triples(triples);
        assert!(idx.is_known_tail("h", "r", "t1"));
        assert!(idx.is_known_tail("h", "r", "t2"));
        assert!(!idx.is_known_tail("h", "r", "t3"));
    }

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
            negatives.len() >= 1,
            "Expected at least 1 negative, got {}",
            negatives.len()
        );
        for neg in negatives {
            assert_eq!(neg.head, "e1");
            assert_eq!(neg.relation, "r1");
            assert_ne!(neg.tail, "e2"); // Should be different from positive
        }
    }

    #[test]
    fn link_prediction_rank_linear_matches_deterministic_sort() {
        // The ranking logic in `evaluate_link_prediction` is intentionally O(|E|)
        // and uses a deterministic tie-break on entity id. This test ensures that
        // the linear-time rank matches an explicit sort using the same ordering.
        //
        // Ordering:
        //   higher score first; among equal scores, lexicographically smaller id first.
        for n in [1usize, 2, 10, 100] {
            let ids: Vec<String> = (0..n).map(|i| format!("e{i:03}")).collect();

            // Create a score pattern with ties, and a deterministic non-sorted iteration order
            // to ensure tie-breaking does not depend on input order.
            //
            // Scores are bucketed into 7 bins to force collisions:
            //   score(i) = (i % 7) / 7
            let mut scores: Vec<(String, f32)> = Vec::with_capacity(n);
            for j in 0..n {
                // A simple permutation (invertible when n is odd, but we don't need that).
                let i = (j.wrapping_mul(17) + 3) % n;
                let s = (i % 7) as f32 / 7.0;
                scores.push((ids[i].clone(), s));
            }

            // Pick a deterministic target (middle id if present).
            let tail = ids[n / 2].clone();
            let tail_score = ((n / 2) % 7) as f32 / 7.0;

            // Linear-time rank (same as evaluate_link_prediction).
            let mut better = 0usize;
            let mut tie_before = 0usize;
            for (id, s) in &scores {
                if id == &tail {
                    continue;
                }
                if *s > tail_score {
                    better += 1;
                } else if *s == tail_score && id.as_str() < tail.as_str() {
                    tie_before += 1;
                }
            }
            let rank_linear = better + tie_before + 1;

            // Deterministic sort-based rank.
            scores.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .expect("no NaNs in test scores")
                    .then_with(|| a.0.cmp(&b.0))
            });
            let rank_sort = scores
                .iter()
                .position(|(id, _)| id == &tail)
                .map(|pos| pos + 1)
                .unwrap_or(usize::MAX);

            assert_eq!(rank_linear, rank_sort);
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

    #[test]
    fn test_log_training_result() {
        let result = TrainingResult {
            final_results: EvaluationResults {
                mrr: 0.5,
                hits_at_1: 0.3,
                hits_at_3: 0.4,
                hits_at_10: 0.6,
                mean_rank: 5.5,
            },
            loss_history: vec![1.0, 0.8, 0.6],
            validation_mrr_history: vec![0.3, 0.4, 0.5],
            best_epoch: 2,
            training_time_seconds: Some(10.5),
        };

        // Test stdout logging (should not panic)
        log_training_result(&result, None).unwrap();

        // Test file logging
        let temp_file = std::env::temp_dir().join("test_training_result.txt");
        log_training_result(&result, Some(temp_file.to_str().unwrap())).unwrap();

        // Verify file was created and contains expected content
        let content = std::fs::read_to_string(&temp_file).unwrap();
        assert!(content.contains("Training Results"));
        assert!(content.contains("0.5000")); // MRR
        assert!(content.contains("2")); // Best epoch

        // Cleanup
        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_hyperparameter_search_default() {
        let search = HyperparameterSearch::default();
        assert!(!search.learning_rates.is_empty());
        assert!(!search.batch_sizes.is_empty());
        assert!(!search.dimensions.is_empty());
        assert!(!search.regularization_weights.is_empty());
        assert!(search.trials_per_config > 0);
    }

    #[test]
    #[allow(unused_variables)] // empty_boxes documents the test structure
    fn test_evaluate_link_prediction_basic() {
        // This test requires a backend implementation
        // We'll test the logic with a mock, but full integration test should be in backend tests
        // For now, just verify the function signature and error handling

        // Test with empty triples
        let _empty_boxes: HashMap<String, ()> = HashMap::new();
        // Can't actually call evaluate_link_prediction without a Box implementation
        // This test documents the need for integration tests in backend modules
        // Full integration tests are in subsume-ndarray/src/trainer_integration_tests.rs
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_box(dim: usize) -> impl Strategy<Value = TrainableBox> {
        let mu_strat = prop::collection::vec(-10.0f32..10.0, dim);
        let delta_strat = prop::collection::vec(-5.0f32..2.0, dim);
        (mu_strat, delta_strat).prop_map(move |(mu, delta)| TrainableBox::new(mu, delta))
    }

    proptest! {
        #[test]
        fn test_loss_is_non_negative(
            box_a in arb_box(8),
            box_b in arb_box(8),
            is_positive in any::<bool>()
        ) {
            let config = TrainingConfig::default();
            let loss = compute_pair_loss(&box_a, &box_b, is_positive, &config);
            prop_assert!(loss >= 0.0);
        }

        #[test]
        fn test_gradients_are_finite(
            box_a in arb_box(8),
            box_b in arb_box(8),
            is_positive in any::<bool>()
        ) {
            let config = TrainingConfig::default();
            let (g_mu_a, g_delta_a, g_mu_b, g_delta_b) = 
                compute_analytical_gradients(&box_a, &box_b, is_positive, &config);
            
            for g in [g_mu_a, g_delta_a, g_mu_b, g_delta_b] {
                for val in g {
                    prop_assert!(val.is_finite());
                }
            }
        }

        #[test]
        fn test_amsgrad_update_stays_valid(
            mut box_a in arb_box(8),
            grad_mu in prop::collection::vec(-1.0f32..1.0, 8),
            grad_delta in prop::collection::vec(-1.0f32..1.0, 8)
        ) {
            let mut state = AMSGradState::new(8, 0.001);
            box_a.update_amsgrad(&grad_mu, &grad_delta, &mut state);
            
            for &m in &box_a.mu {
                prop_assert!(m.is_finite());
            }
            for &d in &box_a.delta {
                prop_assert!(d.is_finite());
                // Delta should be within reasonable bounds set in update_amsgrad
                prop_assert!(d >= 0.01_f32.ln() - 1e-5);
                prop_assert!(d <= 10.0_f32.ln() + 1e-5);
            }
        }
    }
}

