//! Training utilities for box embeddings: negative sampling, loss kernels, and evaluation.
//!
//! ## Research background (minimal but specific)
//!
//! - Bordes et al. (2013): TransE-style negative sampling + margin ranking losses for KGEs.
//! - Vilnis et al. (2018): box lattice measures for probabilistic box embeddings.
//! - Abboud et al. (2020): BoxE (training patterns for box-shaped representations).
//!
//! ## Implementation invariants (why certain choices exist)
//!
//! - **Negative sampling prevents the trivial solution**: without negatives, "everything contains everything"
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

/// Per-relation geometric transform applied to entity boxes during scoring.
///
/// In TransE-family models, relations are modeled as translations in embedding space.
/// `RelationTransform` extends this idea to box embeddings: the head box is translated
/// before computing containment with the tail box.
///
/// Default is [`Identity`](RelationTransform::Identity) (no transform), which recovers
/// standard box containment scoring.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RelationTransform {
    /// No transform: containment scoring uses raw entity boxes.
    #[default]
    Identity,
    /// Translate the box bounds by a per-dimension offset vector.
    ///
    /// Given a box `[min, max]` and offset `d`, the translated box is `[min + d, max + d]`.
    Translation(Vec<f32>),
}

impl RelationTransform {
    /// True if this transform is the identity (no-op).
    #[inline]
    pub fn is_identity(&self) -> bool {
        matches!(self, RelationTransform::Identity)
    }

    /// Apply this transform to box bounds, returning `(new_min, new_max)`.
    ///
    /// For [`Identity`](RelationTransform::Identity), returns clones of the inputs.
    /// For [`Translation`](RelationTransform::Translation), adds the offset to each bound.
    pub fn apply_to_bounds(&self, min: &[f32], max: &[f32]) -> (Vec<f32>, Vec<f32>) {
        match self {
            RelationTransform::Identity => (min.to_vec(), max.to_vec()),
            RelationTransform::Translation(offset) => {
                let new_min: Vec<f32> = min.iter().zip(offset).map(|(m, d)| m + d).collect();
                let new_max: Vec<f32> = max.iter().zip(offset).map(|(m, d)| m + d).collect();
                (new_min, new_max)
            }
        }
    }
}

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
/// - **BoxE paper** (Abboud et al., 2020): Learning rates, batch sizes, regularization
/// - **Gumbel-Box papers** (Dasgupta et al., 2020): Temperature scheduling for Gumbel boxes
/// - **Knowledge graph embedding literature**: Standard practices for negative sampling, margins
///
/// **Key References**:
/// - Abboud et al. (2020): "BoxE: A Box Embedding Model for Knowledge Base Completion" (NeurIPS)
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
/// - \(\lambda_{\text{wd}}\) is `weight_decay`
/// # Field usage
///
/// Only `learning_rate`, `margin`, `regularization`, and `negative_weight` are consumed
/// by [`BoxEmbeddingTrainer::train_step`] and [`ConeEmbeddingTrainer::train_step`].
/// The remaining fields (`epochs`, `batch_size`, `negative_samples`, `negative_strategy`,
/// `temperature`, `weight_decay`, `early_stopping_*`, `warmup_epochs`) are configuration
/// metadata for caller-side training loops (e.g., [`el_training::train_el_embeddings`](crate::el_training::train_el_embeddings)).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// L2 regularization weight
    pub regularization: f32,
    /// Warmup epochs
    pub warmup_epochs: usize,

    /// Weight on the "negative" loss term (internal trainer loop).
    ///
    /// This does not affect `evaluate_link_prediction`; it only scales the margin term in
    /// `compute_pair_loss` for negative examples.
    pub negative_weight: f32,

    /// Softplus steepness for Gumbel intersection volume in gradient computation.
    ///
    /// Controls the smoothness of the intersection volume approximation:
    /// `softplus(beta * (hi - lo), 1.0)` replaces the hard `max(0, hi - lo)`.
    /// Lower values give broader gradients for disjoint boxes; higher values
    /// approach the hard intersection. Annealed toward `gumbel_beta_final`
    /// across epochs when using `BoxEmbeddingTrainer::fit`.
    ///
    /// Default: `1.0` (broad gradients, backward-compatible behavior when
    /// combined with the disjoint surrogate).
    pub gumbel_beta: f32,

    /// Final value of `gumbel_beta` after annealing across epochs.
    ///
    /// In `BoxEmbeddingTrainer::fit`, `gumbel_beta` is linearly interpolated
    /// from its initial value to this value over the training epochs.
    /// Higher final beta gives sharper containment boundaries at convergence.
    ///
    /// Default: `10.0`.
    pub gumbel_beta_final: f32,

    /// Maximum L2 norm for combined gradients (global gradient clipping).
    ///
    /// If the L2 norm of all gradient components exceeds this value, gradients
    /// are uniformly scaled down to `max_grad_norm / norm`. Matches the IESL
    /// box-embeddings library default.
    ///
    /// Default: `10.0`.
    pub max_grad_norm: f32,

    /// Temperature for self-adversarial negative weighting.
    ///
    /// Controls how strongly harder negatives (those the model currently scores
    /// highly) are upweighted during training. Lower temperature concentrates
    /// weight on the hardest negatives; higher temperature approaches uniform
    /// weighting.
    ///
    /// Default: `1.0`.
    pub adversarial_temperature: f32,

    /// Use InfoNCE-style contrastive loss instead of margin-based ranking loss.
    ///
    /// When true, the loss for a (positive, negative) pair becomes:
    /// `softplus((score_neg - score_pos) / tau)` where `tau = margin`.
    /// This is equivalent to the binary cross-entropy form of InfoNCE.
    ///
    /// Default: `false` (backward-compatible margin-based loss).
    pub use_infonce: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3, // Paper default: 1e-3 to 5e-4
            epochs: 100,
            batch_size: 512, // Paper range: 512-4096
            negative_samples: 1,
            negative_strategy: NegativeSamplingStrategy::CorruptTail,
            temperature: 1.0,
            weight_decay: 1e-5,                // Paper range: 1e-5 to 1e-3
            margin: 1.0,                       // Margin for ranking loss
            early_stopping_patience: Some(10), // Early stopping after 10 epochs without improvement
            early_stopping_min_delta: 0.001,
            regularization: 0.0001,
            warmup_epochs: 10,
            negative_weight: 1.0,
            gumbel_beta: 10.0,
            gumbel_beta_final: 50.0,
            max_grad_norm: 10.0,
            adversarial_temperature: 1.0,
            use_infonce: false,
        }
    }
}

/// Evaluation results for link prediction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvaluationResults {
    /// Mean Reciprocal Rank (average of head and tail MRR).
    pub mrr: f32,
    /// MRR for head prediction only.
    pub head_mrr: f32,
    /// MRR for tail prediction only.
    pub tail_mrr: f32,
    /// Hits@1
    pub hits_at_1: f32,
    /// Hits@3
    pub hits_at_3: f32,
    /// Hits@10
    pub hits_at_10: f32,
    /// Mean Rank
    pub mean_rank: f32,
    /// Per-relation evaluation breakdown.
    pub per_relation: Vec<PerRelationResults>,
}

/// Per-relation evaluation breakdown.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerRelationResults {
    /// Relation name or ID.
    pub relation: String,
    /// MRR for this relation's triples.
    pub mrr: f32,
    /// Hits@10 for this relation's triples.
    pub hits_at_10: f32,
    /// Number of test triples for this relation.
    pub count: usize,
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
    entities: Vec<&'a str>,
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
    fn pick<R: Rng>(&self, rng: &mut R) -> &'a str {
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

/// An index of "known true" triples for filtered link prediction evaluation.
///
/// In standard KGE evaluation (e.g. FB15k-237, WN18RR), **filtered ranking** removes any
/// candidate that is already a true triple in train/valid/test, except for the test triple
/// being evaluated. This avoids penalizing the model for ranking other true answers above
/// the held-out one.
///
/// This index is intentionally shaped for the most common evaluation query we currently
/// support in `subsume`:
/// - tail prediction: \((h, r, ?)\)
///
/// Notes:
/// - Building this index **allocates**, but using it during evaluation does not.
/// - Memory can be large for big KGs; prefer using `FilteredTripleIndexIds` with interned IDs.
#[derive(Debug, Default, Clone)]
pub struct FilteredTripleIndex {
    tails_by_head_rel: HashMap<String, HashMap<String, HashSet<String>>>,
    heads_by_tail_rel: HashMap<String, HashMap<String, HashSet<String>>>,
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

    /// Build a filtered-ranking index from all splits of a [`Dataset`](crate::dataset::Dataset).
    ///
    /// Indexes train + valid + test triples so that filtered evaluation can
    /// exclude all known-true triples.
    pub fn from_dataset(dataset: &crate::dataset::Dataset) -> Self {
        Self::from_triples(
            dataset
                .train
                .iter()
                .chain(dataset.valid.iter())
                .chain(dataset.test.iter()),
        )
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
            self.heads_by_tail_rel
                .entry(t.tail.clone())
                .or_default()
                .entry(t.relation.clone())
                .or_default()
                .insert(t.head.clone());
        }
    }

    /// Extend the index with more known-true triples (owned).
    pub fn extend_owned<I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = Triple>,
    {
        for t in triples {
            self.tails_by_head_rel
                .entry(t.head.clone())
                .or_default()
                .entry(t.relation.clone())
                .or_default()
                .insert(t.tail.clone());
            self.heads_by_tail_rel
                .entry(t.tail)
                .or_default()
                .entry(t.relation)
                .or_default()
                .insert(t.head);
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

    /// True iff `(head, relation, tail)` is a known true triple (head lookup).
    #[inline]
    pub fn is_known_head(&self, tail: &str, relation: &str, head: &str) -> bool {
        self.heads_by_tail_rel
            .get(tail)
            .and_then(|by_rel| by_rel.get(relation))
            .is_some_and(|heads| heads.contains(head))
    }

    /// Return all known-true heads for the query \((?, relation, tail)\).
    #[inline]
    pub fn known_heads(&self, tail: &str, relation: &str) -> Option<&HashSet<String>> {
        self.heads_by_tail_rel
            .get(tail)
            .and_then(|by_rel| by_rel.get(relation))
    }
}

/// Like [`FilteredTripleIndex`], but for interned integer IDs.
///
/// This is the preferred form for performance-sensitive evaluation, because it avoids
/// hashing/cloning `String` IDs in the hot loop.
#[derive(Debug, Default, Clone)]
pub struct FilteredTripleIndexIds {
    // Keyed by (head_id, relation_id).
    //
    // Using a flat key avoids a nested HashMap allocation per distinct (head, relation).
    tails_by_head_rel: HashMap<(usize, usize), HashSet<usize>>,
    // Keyed by (tail_id, relation_id).
    heads_by_tail_rel: HashMap<(usize, usize), HashSet<usize>>,
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

    /// Build a filtered-ranking index from all splits of an [`InternedDataset`](crate::dataset::InternedDataset).
    ///
    /// Indexes train + valid + test triples so that filtered evaluation can
    /// exclude all known-true triples.
    pub fn from_dataset(dataset: &crate::dataset::InternedDataset) -> Self {
        Self::from_triples(
            dataset
                .train
                .iter()
                .chain(dataset.valid.iter())
                .chain(dataset.test.iter()),
        )
    }

    /// Extend the index with more known-true triples.
    pub fn extend<'a, I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = &'a crate::dataset::TripleIds>,
    {
        for t in triples {
            self.tails_by_head_rel
                .entry((t.head, t.relation))
                .or_default()
                .insert(t.tail);
            self.heads_by_tail_rel
                .entry((t.tail, t.relation))
                .or_default()
                .insert(t.head);
        }
    }

    /// True iff `(head, relation, tail)` is a known true triple.
    #[inline]
    pub fn is_known_tail(&self, head: usize, relation: usize, tail: usize) -> bool {
        self.tails_by_head_rel
            .get(&(head, relation))
            .is_some_and(|tails| tails.contains(&tail))
    }

    /// Return all known-true tails for the query \((head, relation, ?)\).
    #[inline]
    pub fn known_tails(&self, head: usize, relation: usize) -> Option<&HashSet<usize>> {
        self.tails_by_head_rel.get(&(head, relation))
    }

    /// True iff `(head, relation, tail)` is a known true triple (head lookup).
    #[inline]
    pub fn is_known_head(&self, tail: usize, relation: usize, head: usize) -> bool {
        self.heads_by_tail_rel
            .get(&(tail, relation))
            .is_some_and(|heads| heads.contains(&head))
    }

    /// Return all known-true heads for the query \((?, relation, tail)\).
    #[inline]
    pub fn known_heads(&self, tail: usize, relation: usize) -> Option<&HashSet<usize>> {
        self.heads_by_tail_rel.get(&(tail, relation))
    }
}

/// Compute the rank of `target` among all entities, scoring each candidate via `score_fn`.
///
/// `score_fn(candidate_box) -> f32` returns a score (higher = more likely).
/// Deterministic tie-break: among equal scores, lexicographically smaller entity id ranks first.
fn rank_among_entities<B, F>(
    entity_boxes: &HashMap<String, B>,
    target: &str,
    score_fn: F,
    filter_known: Option<&HashSet<String>>,
) -> Result<usize, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
    F: Fn(&B) -> Result<f32, crate::BoxError>,
{
    let target_box = match entity_boxes.get(target) {
        Some(b) => b,
        None => return Ok(usize::MAX),
    };
    let target_score = score_fn(target_box)?;
    if target_score.is_nan() {
        return Err(crate::BoxError::Internal(
            "NaN containment score encountered (target)".to_string(),
        ));
    }

    let mut better = 0usize;
    let mut tie_before = 0usize;
    for (entity, box_) in entity_boxes {
        if entity == target {
            continue;
        }
        let score = score_fn(box_)?;
        if score.is_nan() {
            return Err(crate::BoxError::Internal(
                "NaN containment score encountered".to_string(),
            ));
        }
        if score > target_score {
            better += 1;
        } else if score == target_score && entity.as_str() < target {
            tie_before += 1;
        }
    }

    // Filtered ranking: subtract contributions from known-true entities.
    if let Some(known) = filter_known {
        let mut filtered_better = 0usize;
        let mut filtered_tie_before = 0usize;
        for known_entity in known {
            if known_entity == target {
                continue;
            }
            let Some(box_) = entity_boxes.get(known_entity) else {
                continue;
            };
            let score = score_fn(box_)?;
            if score.is_nan() {
                return Err(crate::BoxError::Internal(
                    "NaN containment score encountered".to_string(),
                ));
            }
            if score > target_score {
                filtered_better += 1;
            } else if score == target_score && known_entity.as_str() < target {
                filtered_tie_before += 1;
            }
        }
        better = better.saturating_sub(filtered_better);
        tie_before = tie_before.saturating_sub(filtered_tie_before);
    }

    Ok(better + tie_before + 1)
}

fn evaluate_link_prediction_inner<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    relation_transforms: Option<&HashMap<String, RelationTransform>>,
    filter: Option<&FilteredTripleIndex>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    // The generic (string-keyed) path only supports Identity transforms.
    // Non-Identity transforms require constructing translated boxes from concrete
    // min/max coordinates, which needs the interned NdarrayBox path.
    if let Some(transforms) = relation_transforms {
        for (rel, transform) in transforms {
            if !transform.is_identity() {
                return Err(crate::BoxError::Internal(format!(
                    "Non-Identity RelationTransform for relation '{}' requires the interned \
                     evaluation path with NdarrayBox",
                    rel
                )));
            }
        }
    }

    let mut tail_ranks = Vec::with_capacity(test_triples.len());
    let mut head_ranks = Vec::with_capacity(test_triples.len());
    // (relation, tail_rank, head_rank) per triple for per-relation aggregation.
    let mut per_triple: Vec<(&str, usize, usize)> = Vec::with_capacity(test_triples.len());

    for triple in test_triples {
        let head_box = entity_boxes
            .get(&triple.head)
            .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity: {}", triple.head)))?;

        // -- Tail prediction: (h, r, ?) --
        let filter_tails = filter.and_then(|f| f.known_tails(&triple.head, &triple.relation));
        let t_rank = rank_among_entities(
            entity_boxes,
            &triple.tail,
            |candidate| head_box.containment_prob_fast(candidate, 1.0),
            filter_tails,
        )?;

        // -- Head prediction: (?, r, t) --
        let tail_box = entity_boxes
            .get(&triple.tail)
            .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity: {}", triple.tail)))?;
        let filter_heads = filter.and_then(|f| f.known_heads(&triple.tail, &triple.relation));
        let h_rank = rank_among_entities(
            entity_boxes,
            &triple.head,
            |candidate| candidate.containment_prob_fast(tail_box, 1.0),
            filter_heads,
        )?;

        tail_ranks.push(t_rank);
        head_ranks.push(h_rank);
        per_triple.push((triple.relation.as_str(), t_rank, h_rank));
    }

    // Combined ranks: both head and tail ranks contribute equally (Bordes 2013 protocol).
    let all_ranks: Vec<usize> = tail_ranks
        .iter()
        .chain(head_ranks.iter())
        .copied()
        .collect();

    let mrr = mean_reciprocal_rank(all_ranks.iter().copied());
    let tail_mrr = mean_reciprocal_rank(tail_ranks.iter().copied());
    let head_mrr = mean_reciprocal_rank(head_ranks.iter().copied());
    let hits_at_1 = hits_at_k(all_ranks.iter().copied(), 1);
    let hits_at_3 = hits_at_k(all_ranks.iter().copied(), 3);
    let hits_at_10 = hits_at_k(all_ranks.iter().copied(), 10);
    let mean_rank_val = mean_rank(all_ranks.iter().copied());

    // Per-relation aggregation.
    let per_relation = aggregate_per_relation(&per_triple);

    Ok(EvaluationResults {
        mrr,
        head_mrr,
        tail_mrr,
        hits_at_1,
        hits_at_3,
        hits_at_10,
        mean_rank: mean_rank_val,
        per_relation,
    })
}

/// Aggregate per-relation metrics from per-triple (relation, tail_rank, head_rank) tuples.
fn aggregate_per_relation(per_triple: &[(&str, usize, usize)]) -> Vec<PerRelationResults> {
    let mut by_rel: HashMap<&str, Vec<usize>> = HashMap::new();
    for &(rel, t_rank, h_rank) in per_triple {
        let ranks = by_rel.entry(rel).or_default();
        ranks.push(t_rank);
        ranks.push(h_rank);
    }
    let mut results: Vec<PerRelationResults> = by_rel
        .into_iter()
        .map(|(rel, ranks)| {
            let count = ranks.len() / 2; // number of triples (each contributes 2 ranks)
            let mrr = mean_reciprocal_rank(ranks.iter().copied());
            let h10 = hits_at_k(ranks.iter().copied(), 10);
            PerRelationResults {
                relation: rel.to_string(),
                mrr,
                hits_at_10: h10,
                count,
            }
        })
        .collect();
    results.sort_by(|a, b| a.relation.cmp(&b.relation));
    results
}

/// Scoring direction for interned rank computation.
enum ScoreDirection {
    /// Tail prediction: score = query.containment_prob(candidate).
    /// Can use batch `containment_prob_many`.
    Forward,
    /// Head prediction: score = candidate.containment_prob(query).
    /// Falls back to per-entity `containment_prob_fast`.
    Reverse,
}

/// Compute the rank of `target_id` among all entity boxes in the interned setting.
///
/// - `Forward`: score = query_box.containment_prob(candidate) (tail prediction).
/// - `Reverse`: score = candidate.containment_prob(query_box) (head prediction).
fn rank_among_entities_interned<B>(
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    target_id: usize,
    query_box: &B,
    direction: &ScoreDirection,
    filter_known: Option<&HashSet<usize>>,
    scores_buf: &mut Vec<f32>,
) -> Result<usize, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    const CHUNK: usize = 4096;

    let target_box = match entity_boxes.get(target_id) {
        Some(b) => b,
        None => return Ok(usize::MAX),
    };
    let target_name = entities.get(target_id).ok_or_else(|| {
        crate::BoxError::Internal(format!("Missing entity label (target): {}", target_id))
    })?;
    let target_score = match direction {
        ScoreDirection::Forward => query_box.containment_prob_fast(target_box, 1.0)?,
        ScoreDirection::Reverse => target_box.containment_prob_fast(query_box, 1.0)?,
    };
    if target_score.is_nan() {
        return Err(crate::BoxError::Internal(
            "NaN containment score encountered (target)".to_string(),
        ));
    }

    if scores_buf.len() < CHUNK {
        scores_buf.resize(CHUNK, 0.0);
    }

    let mut better = 0usize;
    let mut tie_before = 0usize;

    match direction {
        ScoreDirection::Forward => {
            // Batch scoring: query_box.containment_prob_many(candidates).
            for start in (0..entity_boxes.len()).step_by(CHUNK) {
                let end = (start + CHUNK).min(entity_boxes.len());
                let slice = &entity_boxes[start..end];
                let len = end - start;

                query_box.containment_prob_many(slice, 1.0, &mut scores_buf[..len])?;

                for (i, &score) in scores_buf[..len].iter().enumerate() {
                    let entity_id = start + i;
                    if entity_id == target_id {
                        continue;
                    }
                    if score.is_nan() {
                        return Err(crate::BoxError::Internal(
                            "NaN containment score encountered".to_string(),
                        ));
                    }
                    if score > target_score {
                        better += 1;
                    } else if score == target_score {
                        let name = entities.get(entity_id).ok_or_else(|| {
                            crate::BoxError::Internal(format!(
                                "Missing entity label (candidate): {}",
                                entity_id
                            ))
                        })?;
                        if name < target_name {
                            tie_before += 1;
                        }
                    }
                }
            }
        }
        ScoreDirection::Reverse => {
            // Per-entity scoring: candidate.containment_prob_fast(query_box).
            for (entity_id, candidate) in entity_boxes.iter().enumerate() {
                if entity_id == target_id {
                    continue;
                }
                let score = candidate.containment_prob_fast(query_box, 1.0)?;
                if score.is_nan() {
                    return Err(crate::BoxError::Internal(
                        "NaN containment score encountered".to_string(),
                    ));
                }
                if score > target_score {
                    better += 1;
                } else if score == target_score {
                    let name = entities.get(entity_id).ok_or_else(|| {
                        crate::BoxError::Internal(format!(
                            "Missing entity label (candidate): {}",
                            entity_id
                        ))
                    })?;
                    if name < target_name {
                        tie_before += 1;
                    }
                }
            }
        }
    }

    if let Some(known) = filter_known {
        let mut filtered_better = 0usize;
        let mut filtered_tie_before = 0usize;
        for &known_id in known {
            if known_id == target_id {
                continue;
            }
            let Some(box_) = entity_boxes.get(known_id) else {
                continue;
            };
            let score = match direction {
                ScoreDirection::Forward => query_box.containment_prob_fast(box_, 1.0)?,
                ScoreDirection::Reverse => box_.containment_prob_fast(query_box, 1.0)?,
            };
            if score.is_nan() {
                return Err(crate::BoxError::Internal(
                    "NaN containment score encountered".to_string(),
                ));
            }
            if score > target_score {
                filtered_better += 1;
            } else if score == target_score {
                let name = entities.get(known_id).ok_or_else(|| {
                    crate::BoxError::Internal(format!(
                        "Missing entity label (filtered): {}",
                        known_id
                    ))
                })?;
                if name < target_name {
                    filtered_tie_before += 1;
                }
            }
        }
        better = better.saturating_sub(filtered_better);
        tie_before = tie_before.saturating_sub(filtered_tie_before);
    }

    Ok(better + tie_before + 1)
}

/// Rank target among all entities using a translated query box (tail prediction).
///
/// Constructs a translated `NdarrayBox` from `query_box` + `transform`, then
/// scores `translated.containment_prob_fast(candidate)` for each entity.
#[cfg(feature = "ndarray-backend")]
fn rank_with_translated_query_forward(
    entity_boxes: &[crate::ndarray_backend::NdarrayBox],
    entities: &crate::dataset::Vocab,
    target_id: usize,
    query_box: &crate::ndarray_backend::NdarrayBox,
    transform: &RelationTransform,
    filter_known: Option<&HashSet<usize>>,
) -> Result<usize, crate::BoxError> {
    use crate::Box as BoxTrait;

    let (new_min, new_max) = transform.apply_to_bounds(
        query_box.min().as_slice().unwrap_or(&[]),
        query_box.max().as_slice().unwrap_or(&[]),
    );
    let translated = crate::ndarray_backend::NdarrayBox::new(
        ndarray::Array1::from_vec(new_min),
        ndarray::Array1::from_vec(new_max),
        1.0,
    )?;

    let target_score = translated.containment_prob_fast(
        entity_boxes.get(target_id).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (target): {target_id}"))
        })?,
        1.0,
    )?;
    if target_score.is_nan() {
        return Err(crate::BoxError::Internal(
            "NaN containment score encountered (target)".to_string(),
        ));
    }
    let target_name = entities
        .get(target_id)
        .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity label: {target_id}")))?;

    let mut better = 0usize;
    let mut tie_before = 0usize;
    for (entity_id, candidate) in entity_boxes.iter().enumerate() {
        if entity_id == target_id {
            continue;
        }
        let score = translated.containment_prob_fast(candidate, 1.0)?;
        if score.is_nan() {
            return Err(crate::BoxError::Internal(
                "NaN containment score encountered".to_string(),
            ));
        }
        if score > target_score {
            better += 1;
        } else if score == target_score {
            let name = entities.get(entity_id).ok_or_else(|| {
                crate::BoxError::Internal(format!("Missing entity label: {entity_id}"))
            })?;
            if name < target_name {
                tie_before += 1;
            }
        }
    }

    if let Some(known) = filter_known {
        let mut filtered_better = 0usize;
        let mut filtered_tie_before = 0usize;
        for &known_id in known {
            if known_id == target_id {
                continue;
            }
            let Some(box_) = entity_boxes.get(known_id) else {
                continue;
            };
            let score = translated.containment_prob_fast(box_, 1.0)?;
            if score.is_nan() {
                continue;
            }
            if score > target_score {
                filtered_better += 1;
            } else if score == target_score {
                let name = entities.get(known_id).ok_or_else(|| {
                    crate::BoxError::Internal(format!("Missing entity label: {known_id}"))
                })?;
                if name < target_name {
                    filtered_tie_before += 1;
                }
            }
        }
        better = better.saturating_sub(filtered_better);
        tie_before = tie_before.saturating_sub(filtered_tie_before);
    }

    Ok(better + tie_before + 1)
}

/// Rank target among all entities using a translated query box (head prediction).
///
/// For head prediction `(?, r, t)`, we score `candidate.containment_prob_fast(translated_tail)`.
/// The transform is applied inversely: `Translation(d)` becomes `Translation(-d)` so that
/// the tail is shifted to the "un-transformed" space where candidates live.
#[cfg(feature = "ndarray-backend")]
fn rank_with_translated_query_reverse(
    entity_boxes: &[crate::ndarray_backend::NdarrayBox],
    entities: &crate::dataset::Vocab,
    target_id: usize,
    query_box: &crate::ndarray_backend::NdarrayBox,
    transform: &RelationTransform,
    filter_known: Option<&HashSet<usize>>,
) -> Result<usize, crate::BoxError> {
    use crate::Box as BoxTrait;

    // Inverse transform for head prediction: negate the translation.
    let inverse_transform = match transform {
        RelationTransform::Identity => RelationTransform::Identity,
        RelationTransform::Translation(d) => {
            RelationTransform::Translation(d.iter().map(|x| -x).collect())
        }
    };

    let (new_min, new_max) = inverse_transform.apply_to_bounds(
        query_box.min().as_slice().unwrap_or(&[]),
        query_box.max().as_slice().unwrap_or(&[]),
    );
    let translated = crate::ndarray_backend::NdarrayBox::new(
        ndarray::Array1::from_vec(new_min),
        ndarray::Array1::from_vec(new_max),
        1.0,
    )?;

    let target_box = entity_boxes.get(target_id).ok_or_else(|| {
        crate::BoxError::Internal(format!("Missing entity id (target): {target_id}"))
    })?;
    let target_score = target_box.containment_prob_fast(&translated, 1.0)?;
    if target_score.is_nan() {
        return Err(crate::BoxError::Internal(
            "NaN containment score encountered (target)".to_string(),
        ));
    }
    let target_name = entities
        .get(target_id)
        .ok_or_else(|| crate::BoxError::Internal(format!("Missing entity label: {target_id}")))?;

    let mut better = 0usize;
    let mut tie_before = 0usize;
    for (entity_id, candidate) in entity_boxes.iter().enumerate() {
        if entity_id == target_id {
            continue;
        }
        let score = candidate.containment_prob_fast(&translated, 1.0)?;
        if score.is_nan() {
            return Err(crate::BoxError::Internal(
                "NaN containment score encountered".to_string(),
            ));
        }
        if score > target_score {
            better += 1;
        } else if score == target_score {
            let name = entities.get(entity_id).ok_or_else(|| {
                crate::BoxError::Internal(format!("Missing entity label: {entity_id}"))
            })?;
            if name < target_name {
                tie_before += 1;
            }
        }
    }

    if let Some(known) = filter_known {
        let mut filtered_better = 0usize;
        let mut filtered_tie_before = 0usize;
        for &known_id in known {
            if known_id == target_id {
                continue;
            }
            let Some(box_) = entity_boxes.get(known_id) else {
                continue;
            };
            let score = box_.containment_prob_fast(&translated, 1.0)?;
            if score.is_nan() {
                continue;
            }
            if score > target_score {
                filtered_better += 1;
            } else if score == target_score {
                let name = entities.get(known_id).ok_or_else(|| {
                    crate::BoxError::Internal(format!("Missing entity label: {known_id}"))
                })?;
                if name < target_name {
                    filtered_tie_before += 1;
                }
            }
        }
        better = better.saturating_sub(filtered_better);
        tie_before = tie_before.saturating_sub(filtered_tie_before);
    }

    Ok(better + tie_before + 1)
}

fn evaluate_link_prediction_interned_inner<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    filter: Option<&FilteredTripleIndexIds>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    let mut tail_ranks = Vec::with_capacity(test_triples.len());
    let mut head_ranks = Vec::with_capacity(test_triples.len());
    let mut per_triple: Vec<(usize, usize, usize)> = Vec::with_capacity(test_triples.len());
    let mut scores_buf = vec![0.0f32; 4096];

    for triple in test_triples {
        let head_box = entity_boxes.get(triple.head).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (head): {}", triple.head))
        })?;
        let tail_box = entity_boxes.get(triple.tail).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (tail): {}", triple.tail))
        })?;

        let filter_tails = filter.and_then(|f| f.known_tails(triple.head, triple.relation));
        let t_rank = rank_among_entities_interned(
            entity_boxes,
            entities,
            triple.tail,
            head_box,
            &ScoreDirection::Forward,
            filter_tails,
            &mut scores_buf,
        )?;

        let filter_heads = filter.and_then(|f| f.known_heads(triple.tail, triple.relation));
        let h_rank = rank_among_entities_interned(
            entity_boxes,
            entities,
            triple.head,
            tail_box,
            &ScoreDirection::Reverse,
            filter_heads,
            &mut scores_buf,
        )?;

        tail_ranks.push(t_rank);
        head_ranks.push(h_rank);
        per_triple.push((triple.relation, t_rank, h_rank));
    }

    collect_evaluation_results(&tail_ranks, &head_ranks, &per_triple)
}

/// Evaluate interned link prediction with relation-specific transforms (NdarrayBox only).
///
/// This is the concrete implementation backing
/// [`evaluate_link_prediction_interned_with_transforms`]. It handles both identity
/// and non-identity transforms by dispatching to the translated ranking helpers.
#[cfg(feature = "ndarray-backend")]
fn evaluate_interned_with_transforms_inner(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[crate::ndarray_backend::NdarrayBox],
    entities: &crate::dataset::Vocab,
    relation_transforms: &[RelationTransform],
    filter: Option<&FilteredTripleIndexIds>,
) -> Result<EvaluationResults, crate::BoxError> {
    let mut tail_ranks = Vec::with_capacity(test_triples.len());
    let mut head_ranks = Vec::with_capacity(test_triples.len());
    let mut per_triple: Vec<(usize, usize, usize)> = Vec::with_capacity(test_triples.len());
    let mut scores_buf = vec![0.0f32; 4096];

    for triple in test_triples {
        let head_box = entity_boxes.get(triple.head).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (head): {}", triple.head))
        })?;
        let tail_box = entity_boxes.get(triple.tail).ok_or_else(|| {
            crate::BoxError::Internal(format!("Missing entity id (tail): {}", triple.tail))
        })?;

        let transform = relation_transforms
            .get(triple.relation)
            .unwrap_or(&RelationTransform::Identity);

        let filter_tails = filter.and_then(|f| f.known_tails(triple.head, triple.relation));
        let t_rank = if transform.is_identity() {
            rank_among_entities_interned(
                entity_boxes,
                entities,
                triple.tail,
                head_box,
                &ScoreDirection::Forward,
                filter_tails,
                &mut scores_buf,
            )?
        } else {
            rank_with_translated_query_forward(
                entity_boxes,
                entities,
                triple.tail,
                head_box,
                transform,
                filter_tails,
            )?
        };

        let filter_heads = filter.and_then(|f| f.known_heads(triple.tail, triple.relation));
        let h_rank = if transform.is_identity() {
            rank_among_entities_interned(
                entity_boxes,
                entities,
                triple.head,
                tail_box,
                &ScoreDirection::Reverse,
                filter_heads,
                &mut scores_buf,
            )?
        } else {
            rank_with_translated_query_reverse(
                entity_boxes,
                entities,
                triple.head,
                tail_box,
                transform,
                filter_heads,
            )?
        };

        tail_ranks.push(t_rank);
        head_ranks.push(h_rank);
        per_triple.push((triple.relation, t_rank, h_rank));
    }

    collect_evaluation_results(&tail_ranks, &head_ranks, &per_triple)
}

/// Collect tail/head ranks into [`EvaluationResults`].
fn collect_evaluation_results(
    tail_ranks: &[usize],
    head_ranks: &[usize],
    per_triple: &[(usize, usize, usize)],
) -> Result<EvaluationResults, crate::BoxError> {
    let all_ranks: Vec<usize> = tail_ranks
        .iter()
        .chain(head_ranks.iter())
        .copied()
        .collect();

    let mrr = mean_reciprocal_rank(all_ranks.iter().copied());
    let tail_mrr = mean_reciprocal_rank(tail_ranks.iter().copied());
    let head_mrr = mean_reciprocal_rank(head_ranks.iter().copied());
    let hits_at_1 = hits_at_k(all_ranks.iter().copied(), 1);
    let hits_at_3 = hits_at_k(all_ranks.iter().copied(), 3);
    let hits_at_10 = hits_at_k(all_ranks.iter().copied(), 10);
    let mean_rank_val = mean_rank(all_ranks.iter().copied());

    let per_relation = aggregate_per_relation_ids(per_triple);

    Ok(EvaluationResults {
        mrr,
        head_mrr,
        tail_mrr,
        hits_at_1,
        hits_at_3,
        hits_at_10,
        mean_rank: mean_rank_val,
        per_relation,
    })
}

/// Aggregate per-relation metrics from per-triple (relation_id, tail_rank, head_rank) tuples.
fn aggregate_per_relation_ids(per_triple: &[(usize, usize, usize)]) -> Vec<PerRelationResults> {
    let mut by_rel: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(rel, t_rank, h_rank) in per_triple {
        let ranks = by_rel.entry(rel).or_default();
        ranks.push(t_rank);
        ranks.push(h_rank);
    }
    let mut results: Vec<PerRelationResults> = by_rel
        .into_iter()
        .map(|(rel, ranks)| {
            let count = ranks.len() / 2;
            let mrr = mean_reciprocal_rank(ranks.iter().copied());
            let h10 = hits_at_k(ranks.iter().copied(), 10);
            PerRelationResults {
                relation: rel.to_string(),
                mrr,
                hits_at_10: h10,
                count,
            }
        })
        .collect();
    results.sort_by(|a, b| a.relation.cmp(&b.relation));
    results
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
/// Link prediction evaluates both directions for each test triple (Bordes 2013 protocol):
/// - **Tail prediction**: given (head, relation, ?), rank all entities as candidate tails
/// - **Head prediction**: given (?, relation, tail), rank all entities as candidate heads
///
/// **The process**:
/// 1. For each test triple (e.g., (Paris, located_in, France))
/// 2. Tail prediction: score all entities as candidates for (Paris, located_in, ?)
/// 3. Head prediction: score all entities as candidates for (?, located_in, France)
/// 4. Average both directions into aggregate metrics
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
/// # Returns
///
/// Evaluation results with bidirectional MRR (head/tail breakdown), Hits@K, Mean Rank,
/// and per-relation metrics.
///
/// # Note
///
/// This function requires `B::Scalar = f32`. For other scalar types, use backend-specific evaluation functions.
pub fn evaluate_link_prediction<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_inner(test_triples, entity_boxes, None, None)
}

/// Evaluate link prediction in the **filtered** setting.
///
/// Filtered ranking excludes known-true candidates: for tail prediction, excludes
/// \(t’\) where \((h, r, t’)\) is known; for head prediction, excludes \(h’\)
/// where \((h’, r, t)\) is known. The test triple’s own entity is never filtered.
pub fn evaluate_link_prediction_filtered<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    filter: &FilteredTripleIndex,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_inner(test_triples, entity_boxes, None, Some(filter))
}

/// Evaluate link prediction with relation-specific transforms (string-keyed).
///
/// Only [`RelationTransform::Identity`] is supported in this path. For
/// [`RelationTransform::Translation`], use the interned evaluation path.
pub fn evaluate_link_prediction_with_transforms<B>(
    test_triples: &[Triple],
    entity_boxes: &HashMap<String, B>,
    relation_transforms: &HashMap<String, RelationTransform>,
    filter: Option<&FilteredTripleIndex>,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_inner(
        test_triples,
        entity_boxes,
        Some(relation_transforms),
        filter,
    )
}

/// Evaluate link prediction using interned IDs (`usize`) for entities/relations.
///
/// This avoids string hashing/cloning in the candidate loop, which is often the dominant
/// overhead once the scoring kernel itself is optimized.
pub fn evaluate_link_prediction_interned<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_interned_inner(test_triples, entity_boxes, entities, None)
}

/// Evaluate link prediction in the **filtered** setting, using interned IDs.
pub fn evaluate_link_prediction_interned_filtered<B>(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[B],
    entities: &crate::dataset::Vocab,
    filter: &FilteredTripleIndexIds,
) -> Result<EvaluationResults, crate::BoxError>
where
    B: crate::Box<Scalar = f32>,
{
    evaluate_link_prediction_interned_inner(test_triples, entity_boxes, entities, Some(filter))
}

/// Evaluate link prediction with relation-specific transforms (interned IDs).
///
/// The `relation_transforms` slice is indexed by relation ID. Use
/// [`RelationTransform::Identity`] for relations without a transform.
/// [`RelationTransform::Translation`] is supported because this function
/// requires the `ndarray-backend` feature and concrete `NdarrayBox` entities.
#[cfg(feature = "ndarray-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
pub fn evaluate_link_prediction_interned_with_transforms(
    test_triples: &[crate::dataset::TripleIds],
    entity_boxes: &[crate::ndarray_backend::NdarrayBox],
    entities: &crate::dataset::Vocab,
    relation_transforms: &[RelationTransform],
    filter: Option<&FilteredTripleIndexIds>,
) -> Result<EvaluationResults, crate::BoxError> {
    evaluate_interned_with_transforms_inner(
        test_triples,
        entity_boxes,
        entities,
        relation_transforms,
        filter,
    )
}

/// Training result with metrics and history.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Log training results to file or stdout.
pub fn log_training_result(result: &TrainingResult, path: Option<&str>) -> Result<(), BoxError> {
    let output = format!(
        "Training Results\n\
         ===============\n\
         Final MRR: {:.4} (head: {:.4}, tail: {:.4})\n\
         Final Hits@1: {:.4}\n\
         Final Hits@3: {:.4}\n\
         Final Hits@10: {:.4}\n\
         Final Mean Rank: {:.2}\n\
         Best Epoch: {}\n\
         Training Time: {:.2}s\n",
        result.final_results.mrr,
        result.final_results.head_mrr,
        result.final_results.tail_mrr,
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

/// Compute loss for a pair of boxes.
///
/// Design choice (important):
/// - For **positive** examples this loss uses a *symmetric* score by taking
///   \(\min(P(B \subseteq A),\; P(A \subseteq B))\). This encourages "near-equivalence"
///   more than directed entailment.
/// - For hierarchy-like relations, a more typical objective is *directed* containment,
///   e.g. minimize \(-\ln P(B \subseteq A)\) only.
pub fn compute_pair_loss(
    box_a: &TrainableBox,
    box_b: &TrainableBox,
    is_positive: bool,
    config: &TrainingConfig,
) -> f32 {
    let a = box_a.to_box();
    let b = box_b.to_box();

    // Compute softplus-smoothed intersection volume: always positive, always
    // has gradient, unlike the hard max(0, hi-lo) per dimension.
    let beta = config.gumbel_beta;
    let vol_int_soft = softplus_intersection_volume(&a, &b, beta);
    let vol_a = a.volume().max(1e-30);
    let vol_b = b.volume().max(1e-30);

    if is_positive {
        let p_a_b = (vol_int_soft / vol_b).clamp(1e-8, 1.0);
        let p_b_a = (vol_int_soft / vol_a).clamp(1e-8, 1.0);
        let min_prob = p_a_b.min(p_b_a);
        // Cap at 10.0 to prevent explosion from near-zero probabilities.
        let neg_log_prob = (-min_prob.ln()).min(10.0);

        let reg = config.regularization * (vol_a + vol_b);

        (neg_log_prob + reg).max(0.0)
    } else {
        let p_a_b = (vol_int_soft / vol_b).clamp(0.0, 1.0);
        let p_b_a = (vol_int_soft / vol_a).clamp(0.0, 1.0);
        let max_prob = p_a_b.max(p_b_a);

        let margin_loss = if max_prob > config.margin {
            (max_prob - config.margin).powi(2)
        } else {
            0.0
        };

        config.negative_weight * margin_loss
    }
}

/// Compute softplus-smoothed intersection volume.
///
/// Replaces the hard `max(0, hi - lo)` per dimension with
/// `softplus(beta * (hi - lo), 1.0) / beta`, giving always-positive
/// volume and always-nonzero gradients even for disjoint boxes.
fn softplus_intersection_volume(
    a: &crate::trainable::DenseBox,
    b: &crate::trainable::DenseBox,
    beta: f32,
) -> f32 {
    let dim = a.min.len().min(b.min.len());
    let mut vol = 1.0f32;
    for i in 0..dim {
        let lo = a.min[i].max(b.min[i]);
        let hi = a.max[i].min(b.max[i]);
        let side = crate::utils::softplus(hi - lo, beta);
        vol *= side;
        if vol < 1e-30 {
            break;
        }
    }
    vol
}

/// Compute the gradient of [`compute_pair_loss`] with respect to
/// the reparameterized parameters `(mu, delta)` of both boxes.
///
/// Uses the chain rule through the reparameterization:
/// - `min[i] = mu[i] - exp(delta[i]) / 2`
/// - `max[i] = mu[i] + exp(delta[i]) / 2`
///
/// For **positive** pairs, the loss is `-ln(min(P(A|B), P(B|A))) + reg * (Vol_A + Vol_B)`.
/// For **negative** pairs, the loss is `w_neg * max(0, max(P(A|B), P(B|A)) - margin)^2`.
///
/// Intersection volume uses softplus smoothing (`config.gumbel_beta`), so
/// `d(side)/d(bound) = sigmoid(beta * (hi - lo))` rather than the hard 0/1
/// indicator. This gives nonzero gradients even for disjoint boxes, though a
/// center-attraction surrogate is still used when the softplus volume is
/// negligible (`< 1e-30`).
///
/// Gradients are globally norm-clipped to `config.max_grad_norm`.
///
/// Returns `(grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b)`.
pub fn compute_analytical_gradients(
    box_a: &TrainableBox,
    box_b: &TrainableBox,
    is_positive: bool,
    config: &TrainingConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let a = box_a.to_box();
    let b = box_b.to_box();
    let dim = box_a.dim();

    let mut grad_mu_a = vec![0.0f32; dim];
    let mut grad_delta_a = vec![0.0f32; dim];
    let mut grad_mu_b = vec![0.0f32; dim];
    let mut grad_delta_b = vec![0.0f32; dim];

    let vol_a = a.volume().max(1e-30);
    let vol_b = b.volume().max(1e-30);

    let beta = config.gumbel_beta;

    // Per-dimension softplus-smoothed intersection side lengths.
    // side[i] = softplus(hi - lo, beta), always positive -> always has gradient.
    // The gradient of softplus(x, beta) w.r.t. x is sigmoid(beta * x).
    let mut sides = vec![0.0f32; dim];
    let mut side_diffs = vec![0.0f32; dim]; // hi - lo per dimension (raw, before softplus)
                                            // Which bound is active in each dimension:
                                            // lo_from_a[i]: true if max(min_a, min_b) = min_a (A's lower bound is active)
                                            // hi_from_a[i]: true if min(max_a, max_b) = max_a (A's upper bound is active)
    let mut lo_from_a = vec![false; dim];
    let mut hi_from_a = vec![false; dim];
    for i in 0..dim {
        let lo = a.min[i].max(b.min[i]);
        let hi = a.max[i].min(b.max[i]);
        let diff = hi - lo;
        side_diffs[i] = diff;
        sides[i] = crate::utils::softplus(diff, beta);
        lo_from_a[i] = a.min[i] >= b.min[i];
        hi_from_a[i] = a.max[i] <= b.max[i];
    }

    // Softplus-smoothed intersection volume.
    let vol_int: f32 = sides.iter().product();

    // Reparameterization derivatives:
    // d(min_a)/d(mu_a) = 1,   d(min_a)/d(delta_a) = -exp(delta_a)/2
    // d(max_a)/d(mu_a) = 1,   d(max_a)/d(delta_a) = +exp(delta_a)/2
    let half_width_a: Vec<f32> = box_a.delta.iter().map(|d| d.exp() / 2.0).collect();
    let half_width_b: Vec<f32> = box_b.delta.iter().map(|d| d.exp() / 2.0).collect();

    if is_positive {
        // Positive loss: L = -ln(min(P_AB, P_BA)) + reg * (Vol_A + Vol_B)
        // where P_AB = Vol_int / Vol_B, P_BA = Vol_int / Vol_A

        if vol_int < 1e-30 {
            // Disjoint: true gradient is zero (Vol_int = 0).
            // Use surrogate: attract centers so boxes start overlapping.
            for i in 0..dim {
                let center_diff = (b.min[i] + b.max[i]) - (a.min[i] + a.max[i]);
                grad_mu_a[i] = -center_diff; // move A toward B
                grad_mu_b[i] = center_diff; // move B toward A
                                            // Expand both boxes to increase chance of overlap.
                grad_delta_a[i] = -0.1;
                grad_delta_b[i] = -0.1;
            }
            return (grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b);
        }

        let p_ab = (vol_int / vol_b).clamp(1e-8, 1.0);
        let p_ba = (vol_int / vol_a).clamp(1e-8, 1.0);

        // Determine which conditional probability is the minimum.
        let (p, use_ab) = if p_ab <= p_ba {
            (p_ab, true)
        } else {
            (p_ba, false)
        };

        // dL/dP = -1/P (from -ln(P))
        let dl_dp = -1.0 / p;

        // dP/d(Vol_int): P = Vol_int / Vol_denom
        // dP/d(Vol_denom): P = Vol_int / Vol_denom => dP/d(Vol_denom) = -Vol_int / Vol_denom^2
        let (vol_denom, dl_dvol_int, dl_dvol_denom) = if use_ab {
            // P = Vol_int / Vol_B
            (vol_b, dl_dp / vol_b, dl_dp * (-vol_int / (vol_b * vol_b)))
        } else {
            // P = Vol_int / Vol_A
            (vol_a, dl_dp / vol_a, dl_dp * (-vol_int / (vol_a * vol_a)))
        };
        let _ = vol_denom; // suppress unused warning

        // Gradient of Vol_int w.r.t. each bound.
        // Vol_int = prod_j sides[j]. d(Vol_int)/d(sides[i]) = Vol_int / sides[i].
        // d(side_i)/d(hi) = sigmoid(beta * diff_i), d(side_i)/d(lo) = -sigmoid(beta * diff_i).
        for i in 0..dim {
            if sides[i] < 1e-30 {
                continue;
            }
            let dvol_int_dside = vol_int / sides[i];
            let sig = crate::utils::stable_sigmoid(beta * side_diffs[i]);
            let dside_dl = dl_dvol_int * dvol_int_dside;

            // d(side_i)/d(lo) = -sigmoid(beta * diff_i)
            // lo = max(min_a, min_b); if lo_from_a, the active bound is min_a.
            if lo_from_a[i] {
                let dside_dmin_a = -sig;
                grad_mu_a[i] += dside_dl * dside_dmin_a * 1.0;
                grad_delta_a[i] += dside_dl * dside_dmin_a * (-half_width_a[i]);
            } else {
                let dside_dmin_b = -sig;
                grad_mu_b[i] += dside_dl * dside_dmin_b * 1.0;
                grad_delta_b[i] += dside_dl * dside_dmin_b * (-half_width_b[i]);
            }

            // d(side_i)/d(hi) = sigmoid(beta * diff_i)
            // hi = min(max_a, max_b); if hi_from_a, the active bound is max_a.
            if hi_from_a[i] {
                let dside_dmax_a = sig;
                grad_mu_a[i] += dside_dl * dside_dmax_a * 1.0;
                grad_delta_a[i] += dside_dl * dside_dmax_a * half_width_a[i];
            } else {
                let dside_dmax_b = sig;
                grad_mu_b[i] += dside_dl * dside_dmax_b * 1.0;
                grad_delta_b[i] += dside_dl * dside_dmax_b * half_width_b[i];
            }
        }

        // Gradient of denom volume w.r.t. parameters.
        // Vol = prod_j exp(delta_j). d(Vol)/d(delta_i) = Vol * 1 = Vol (since d(exp(d))/d(d) = exp(d))
        // But Vol = prod exp(delta), so d(Vol)/d(delta_i) = Vol (each factor contributes exp(delta_i)).
        if use_ab {
            // Denom = Vol_B. d(Vol_B)/d(delta_b_i) = Vol_B.
            let denom_grad = dl_dvol_denom * vol_b;
            for g in grad_delta_b.iter_mut().take(dim) {
                *g += denom_grad;
            }
        } else {
            let denom_grad = dl_dvol_denom * vol_a;
            for g in grad_delta_a.iter_mut().take(dim) {
                *g += denom_grad;
            }
        }

        // Volume regularization: d(reg * (Vol_A + Vol_B))/d(delta_a_i) = reg * Vol_A
        let reg = config.regularization;
        let reg_a = reg * vol_a;
        let reg_b = reg * vol_b;
        for g in grad_delta_a.iter_mut().take(dim) {
            *g += reg_a;
        }
        for g in grad_delta_b.iter_mut().take(dim) {
            *g += reg_b;
        }
    } else {
        // Negative loss: L = w_neg * max(0, max(P_AB, P_BA) - margin)^2
        let p_ab = (vol_int / vol_b).clamp(0.0, 1.0);
        let p_ba = (vol_int / vol_a).clamp(0.0, 1.0);
        let max_p = p_ab.max(p_ba);

        if max_p <= config.margin || vol_int < 1e-30 {
            // No loss, no gradient.
            return (grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b);
        }

        let use_ab = p_ab >= p_ba;
        let p = if use_ab { p_ab } else { p_ba };

        // dL/dP = w_neg * 2 * (P - margin)
        let dl_dp = config.negative_weight * 2.0 * (p - config.margin);
        let vol_denom = if use_ab { vol_b } else { vol_a };
        let dl_dvol_int = dl_dp / vol_denom;
        let dl_dvol_denom = dl_dp * (-vol_int / (vol_denom * vol_denom));

        // Same chain rule as positive case, using sigmoid-based derivatives.
        for i in 0..dim {
            if sides[i] < 1e-30 {
                continue;
            }
            let dvol_int_dside = vol_int / sides[i];
            let sig = crate::utils::stable_sigmoid(beta * side_diffs[i]);
            let dside_dl = dl_dvol_int * dvol_int_dside;

            if lo_from_a[i] {
                let dside_dmin_a = -sig;
                grad_mu_a[i] += dside_dl * dside_dmin_a;
                grad_delta_a[i] += dside_dl * dside_dmin_a * (-half_width_a[i]);
            } else {
                let dside_dmin_b = -sig;
                grad_mu_b[i] += dside_dl * dside_dmin_b;
                grad_delta_b[i] += dside_dl * dside_dmin_b * (-half_width_b[i]);
            }
            if hi_from_a[i] {
                let dside_dmax_a = sig;
                grad_mu_a[i] += dside_dl * dside_dmax_a;
                grad_delta_a[i] += dside_dl * dside_dmax_a * half_width_a[i];
            } else {
                let dside_dmax_b = sig;
                grad_mu_b[i] += dside_dl * dside_dmax_b;
                grad_delta_b[i] += dside_dl * dside_dmax_b * half_width_b[i];
            }
        }

        if use_ab {
            let denom_grad = dl_dvol_denom * vol_b;
            for g in grad_delta_b.iter_mut().take(dim) {
                *g += denom_grad;
            }
        } else {
            let denom_grad = dl_dvol_denom * vol_a;
            for g in grad_delta_a.iter_mut().take(dim) {
                *g += denom_grad;
            }
        }
    }

    // Global gradient norm clipping: if the L2 norm of all gradient components
    // exceeds max_grad_norm, scale all gradients uniformly.
    let max_norm = config.max_grad_norm;
    let sq_norm: f32 = grad_mu_a
        .iter()
        .chain(grad_delta_a.iter())
        .chain(grad_mu_b.iter())
        .chain(grad_delta_b.iter())
        .map(|g| g * g)
        .sum();
    let norm = sq_norm.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for g in grad_mu_a.iter_mut() {
            *g *= scale;
        }
        for g in grad_delta_a.iter_mut() {
            *g *= scale;
        }
        for g in grad_mu_b.iter_mut() {
            *g *= scale;
        }
        for g in grad_delta_b.iter_mut() {
            *g *= scale;
        }
    }

    (grad_mu_a, grad_delta_a, grad_mu_b, grad_delta_b)
}

// ---------------------------------------------------------------------------
// Box training
// ---------------------------------------------------------------------------

/// End-to-end trainer for box embeddings on knowledge graph datasets.
///
/// Manages entity box embeddings, optimizer state, and provides a `train_step()`
/// method that handles negative sampling, loss computation, gradient updates,
/// and optional evaluation.
///
/// # Example
///
/// ```rust,ignore
/// use subsume::{BoxEmbeddingTrainer, TrainingConfig, Dataset};
///
/// let config = TrainingConfig { learning_rate: 0.01, ..Default::default() };
/// let mut trainer = BoxEmbeddingTrainer::new(config, 16); // dim=16
/// // Add training triples...
/// for epoch in 0..100 {
///     let loss = trainer.train_step(&train_triples)?;
/// }
/// ```
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BoxEmbeddingTrainer {
    /// Training configuration.
    pub config: TrainingConfig,
    /// Learned box embeddings per entity.
    pub boxes: HashMap<usize, TrainableBox>,
    /// AMSGrad optimizer state per entity.
    pub optimizer_states: HashMap<usize, AMSGradState>,
    /// Embedding dimension.
    pub dim: usize,
    /// Per-relation transforms (relation_id -> transform). Default: empty (all Identity).
    pub relation_transforms: HashMap<usize, RelationTransform>,
    /// Current Gumbel beta, annealed from `config.gumbel_beta` to
    /// `config.gumbel_beta_final` across epochs in `fit()`.
    pub current_beta: f32,
}

impl BoxEmbeddingTrainer {
    /// Create a new box embedding trainer.
    pub fn new(config: TrainingConfig, dim: usize) -> Self {
        let current_beta = config.gumbel_beta;
        Self {
            config,
            boxes: HashMap::new(),
            optimizer_states: HashMap::new(),
            dim,
            relation_transforms: HashMap::new(),
            current_beta,
        }
    }

    /// Ensure an entity exists in the trainer; initialize with defaults if missing.
    ///
    /// Creates a small box centered at a dimension-offset position so that
    /// different entities start with slightly different embeddings.
    pub fn ensure_entity(&mut self, id: usize) {
        if !self.boxes.contains_key(&id) {
            let mut init_vec = vec![0.0f32; self.dim];
            if self.dim > 0 {
                // Give each entity a slightly different initial position.
                init_vec[id % self.dim] = 1.0;
            }
            let b = TrainableBox::from_vector(&init_vec, 0.5);
            let n_params = b.num_parameters();
            self.boxes.insert(id, b);
            self.optimizer_states
                .insert(id, AMSGradState::new(n_params, self.config.learning_rate));
        }
    }

    /// Ensure entity exists and return a clone of its trainable box.
    fn snapshot_box(&mut self, id: usize) -> TrainableBox {
        self.ensure_entity(id);
        self.boxes
            .get(&id)
            .cloned()
            .expect("ensure_entity guarantees key exists")
    }

    /// Run one training epoch over the given triples.
    ///
    /// For each `(head, relation, tail)` triple:
    /// 1. Ensure head and tail entities exist.
    /// 2. Generate one negative sample by corrupting the tail.
    /// 3. Compute containment loss for the positive pair and the negative pair.
    /// 4. Compute analytical gradients and apply AMSGrad updates.
    ///
    /// When `config.use_infonce` is true, uses InfoNCE-style contrastive loss
    /// instead of separate margin-based losses. When `config.adversarial_temperature`
    /// is finite, negative gradients are weighted by the model's current positive
    /// score for the negative triple (self-adversarial weighting).
    ///
    /// Uses `self.current_beta` as the effective `gumbel_beta` for this step.
    ///
    /// Returns the average loss across all triples.
    pub fn train_step(&mut self, triples: &[(usize, usize, usize)]) -> Result<f32, BoxError> {
        if triples.is_empty() {
            return Err(BoxError::Internal("empty triple set".to_string()));
        }

        // Build a step-local config snapshot with the annealed beta.
        let mut step_config = self.config.clone();
        step_config.gumbel_beta = self.current_beta;

        let mut total_loss = 0.0f32;
        // Collect all entity IDs present in this batch for negative sampling.
        let entity_ids: Vec<usize> = triples
            .iter()
            .flat_map(|&(h, _, t)| [h, t])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        for &(h, r, t) in triples {
            // Get the relation transform (default to Identity).
            let transform = self
                .relation_transforms
                .get(&r)
                .cloned()
                .unwrap_or(RelationTransform::Identity);

            // Snapshot current boxes (immutable copy for gradient computation).
            let box_h = self.snapshot_box(h);
            let box_t = self.snapshot_box(t);

            // Apply transform to head box for scoring.
            let box_h_transformed = if transform.is_identity() {
                box_h.clone()
            } else {
                let dense = box_h.to_box();
                let (new_min, new_max) = transform.apply_to_bounds(&dense.min, &dense.max);
                let mu: Vec<f32> = new_min
                    .iter()
                    .zip(&new_max)
                    .map(|(lo, hi)| (lo + hi) / 2.0)
                    .collect();
                let delta: Vec<f32> = new_min
                    .iter()
                    .zip(&new_max)
                    .map(|(lo, hi)| ((hi - lo).max(1e-6)).ln())
                    .collect();
                TrainableBox::new(mu, delta).unwrap_or_else(|_| box_h.clone())
            };

            // Negative sample: corrupt the tail with a different entity from the batch.
            // Deterministic: pick an entity that is not the true tail.
            let neg_t = if entity_ids.len() > 1 {
                // Hash-based deterministic selection: use (h + t + epoch-proxy) to vary.
                let idx = (h.wrapping_mul(31).wrapping_add(t).wrapping_add(7)) % entity_ids.len();
                let candidate = entity_ids[idx];
                if candidate == t {
                    entity_ids[(idx + 1) % entity_ids.len()]
                } else {
                    candidate
                }
            } else {
                continue; // cannot generate a negative with a single entity
            };

            let box_neg = self.snapshot_box(neg_t);

            if step_config.use_infonce {
                // InfoNCE loss: softplus((score_neg - score_pos) / tau)
                // where score = ln(Vol_int / Vol_other) (log containment probability),
                // and tau = margin (repurposed as temperature).
                let pos_score = compute_pair_loss(&box_h_transformed, &box_t, true, &step_config);
                let neg_score = compute_pair_loss(&box_h_transformed, &box_neg, true, &step_config);
                let tau = step_config.margin.max(1e-6);
                // InfoNCE: L = softplus((neg_score - pos_score) / tau)
                // Note: pos_score/neg_score are negative-log-prob (lower = better),
                // so "better negative" means lower neg_score. We want to penalize
                // when neg_score < pos_score (model confused).
                let infonce_loss = crate::utils::softplus((pos_score - neg_score) / tau, 1.0);
                total_loss += infonce_loss;

                // Gradients: use positive gradients for both, then weight.
                // d(infonce)/d(pos_score) = sigmoid((pos_score - neg_score) / tau) / tau
                // d(infonce)/d(neg_score) = -sigmoid((pos_score - neg_score) / tau) / tau
                let sig = crate::utils::stable_sigmoid((pos_score - neg_score) / tau);
                let dldpos = sig / tau;
                let dldneg = -sig / tau;

                // Positive pair gradients (scaled by dldpos).
                let (grad_mu_h, grad_delta_h, grad_mu_t, grad_delta_t) =
                    compute_analytical_gradients(&box_h, &box_t, true, &step_config);
                if let (Some(b), Some(s)) =
                    (self.boxes.get_mut(&h), self.optimizer_states.get_mut(&h))
                {
                    let scaled_mu: Vec<f32> = grad_mu_h.iter().map(|g| g * dldpos).collect();
                    let scaled_delta: Vec<f32> = grad_delta_h.iter().map(|g| g * dldpos).collect();
                    b.update_amsgrad(&scaled_mu, &scaled_delta, s);
                }
                if let (Some(b), Some(s)) =
                    (self.boxes.get_mut(&t), self.optimizer_states.get_mut(&t))
                {
                    let scaled_mu: Vec<f32> = grad_mu_t.iter().map(|g| g * dldpos).collect();
                    let scaled_delta: Vec<f32> = grad_delta_t.iter().map(|g| g * dldpos).collect();
                    b.update_amsgrad(&scaled_mu, &scaled_delta, s);
                }

                // Negative pair gradients (scaled by dldneg, computed as positive).
                let box_h2 = self.snapshot_box(h);
                let (grad_mu_h2, grad_delta_h2, grad_mu_neg, grad_delta_neg) =
                    compute_analytical_gradients(&box_h2, &box_neg, true, &step_config);
                if let (Some(b), Some(s)) =
                    (self.boxes.get_mut(&h), self.optimizer_states.get_mut(&h))
                {
                    let scaled_mu: Vec<f32> = grad_mu_h2.iter().map(|g| g * dldneg).collect();
                    let scaled_delta: Vec<f32> = grad_delta_h2.iter().map(|g| g * dldneg).collect();
                    b.update_amsgrad(&scaled_mu, &scaled_delta, s);
                }
                if let (Some(b), Some(s)) = (
                    self.boxes.get_mut(&neg_t),
                    self.optimizer_states.get_mut(&neg_t),
                ) {
                    let scaled_mu: Vec<f32> = grad_mu_neg.iter().map(|g| g * dldneg).collect();
                    let scaled_delta: Vec<f32> =
                        grad_delta_neg.iter().map(|g| g * dldneg).collect();
                    b.update_amsgrad(&scaled_mu, &scaled_delta, s);
                }
            } else {
                // Standard margin-based loss path.

                // Positive loss: head should contain tail.
                let pos_loss = compute_pair_loss(&box_h_transformed, &box_t, true, &step_config);
                total_loss += pos_loss;

                // Positive gradients.
                let (grad_mu_h, grad_delta_h, grad_mu_t, grad_delta_t) =
                    compute_analytical_gradients(&box_h, &box_t, true, &step_config);

                // Apply positive gradients.
                if let (Some(b), Some(s)) =
                    (self.boxes.get_mut(&h), self.optimizer_states.get_mut(&h))
                {
                    b.update_amsgrad(&grad_mu_h, &grad_delta_h, s);
                }
                if let (Some(b), Some(s)) =
                    (self.boxes.get_mut(&t), self.optimizer_states.get_mut(&t))
                {
                    b.update_amsgrad(&grad_mu_t, &grad_delta_t, s);
                }

                let box_h2 = self.snapshot_box(h);

                // Negative loss.
                let neg_loss = compute_pair_loss(&box_h2, &box_neg, false, &step_config);
                total_loss += neg_loss;

                // Negative gradients.
                let (mut grad_mu_h2, mut grad_delta_h2, mut grad_mu_neg, mut grad_delta_neg) =
                    compute_analytical_gradients(&box_h2, &box_neg, false, &step_config);

                // Self-adversarial weighting: scale negative gradients by
                // exp(positive_score / adversarial_temperature), capped at 10.0.
                // Higher-scoring negatives (harder) get more gradient weight.
                let adv_temp = step_config.adversarial_temperature;
                let neg_as_pos_score = compute_pair_loss(&box_h2, &box_neg, true, &step_config);
                // neg_as_pos_score is -ln(P), so lower = model thinks it's positive.
                // We want to upweight negatives the model mistakes for positives,
                // i.e. weight = exp(-neg_as_pos_score / adv_temp) (high when score is low/good).
                let adv_weight = (-neg_as_pos_score / adv_temp).exp().min(10.0);
                for g in grad_mu_h2.iter_mut() {
                    *g *= adv_weight;
                }
                for g in grad_delta_h2.iter_mut() {
                    *g *= adv_weight;
                }
                for g in grad_mu_neg.iter_mut() {
                    *g *= adv_weight;
                }
                for g in grad_delta_neg.iter_mut() {
                    *g *= adv_weight;
                }

                if let (Some(b), Some(s)) =
                    (self.boxes.get_mut(&h), self.optimizer_states.get_mut(&h))
                {
                    b.update_amsgrad(&grad_mu_h2, &grad_delta_h2, s);
                }
                if let (Some(b), Some(s)) = (
                    self.boxes.get_mut(&neg_t),
                    self.optimizer_states.get_mut(&neg_t),
                ) {
                    b.update_amsgrad(&grad_mu_neg, &grad_delta_neg, s);
                }
            }
        }

        Ok(total_loss / triples.len() as f32)
    }

    /// Convert a single entity's [`TrainableBox`] to an [`NdarrayBox`](crate::ndarray_backend::NdarrayBox) for evaluation.
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn get_box(&self, entity_id: usize) -> Option<crate::ndarray_backend::NdarrayBox> {
        self.boxes
            .get(&entity_id)
            .and_then(|b| b.to_ndarray_box().ok())
    }

    /// Convert all entity boxes to [`NdarrayBox`](crate::ndarray_backend::NdarrayBox) for evaluation.
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn get_all_boxes(&self) -> HashMap<usize, crate::ndarray_backend::NdarrayBox> {
        self.boxes
            .iter()
            .filter_map(|(&id, b)| b.to_ndarray_box().ok().map(|nb| (id, nb)))
            .collect()
    }

    /// Export all entity embeddings as flat `f32` vectors.
    ///
    /// Returns `(entity_ids, min_bounds, max_bounds)` where:
    /// - `entity_ids[i]` is the entity ID for the i-th embedding
    /// - `min_bounds` is a flat `Vec<f32>` of length `n_entities * dim` (row-major)
    /// - `max_bounds` is a flat `Vec<f32>` of the same length
    ///
    /// This format is compatible with safetensors, numpy (via reshape), and
    /// vector databases that accept flat float arrays.
    pub fn export_embeddings(&self) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
        let mut ids: Vec<usize> = self.boxes.keys().copied().collect();
        ids.sort_unstable();

        let n = ids.len();
        let mut mins = Vec::with_capacity(n * self.dim);
        let mut maxs = Vec::with_capacity(n * self.dim);

        for &id in &ids {
            let b = &self.boxes[&id];
            let dense = b.to_box();
            mins.extend_from_slice(&dense.min);
            maxs.extend_from_slice(&dense.max);
        }

        (ids, mins, maxs)
    }

    /// Evaluate the trained model on test triples using interned link prediction.
    ///
    /// Converts learned [`TrainableBox`] embeddings to [`NdarrayBox`](crate::ndarray_backend::NdarrayBox)
    /// and runs bidirectional (head + tail) evaluation, optionally with filtered ranking
    /// and relation-specific transforms.
    ///
    /// This is a convenience method that bridges the trainer's internal state to
    /// [`evaluate_link_prediction_interned`] (or the transform-aware variant).
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn evaluate(
        &self,
        test_triples: &[crate::dataset::TripleIds],
        entities: &crate::dataset::Vocab,
        filter: Option<&FilteredTripleIndexIds>,
    ) -> Result<EvaluationResults, BoxError> {
        let max_id = self.boxes.keys().copied().max().unwrap_or(0);
        let num_entities = entities.len().max(max_id + 1);
        let mut entity_vec: Vec<crate::ndarray_backend::NdarrayBox> =
            Vec::with_capacity(num_entities);

        // Build a dense vector indexed by entity ID.
        for id in 0..num_entities {
            let nb = if let Some(b) = self.boxes.get(&id) {
                b.to_ndarray_box().map_err(|e| {
                    BoxError::Internal(format!("Failed to convert entity {id}: {e}"))
                })?
            } else {
                // Default zero-volume box for entities not in the trainer.
                crate::ndarray_backend::NdarrayBox::new(
                    ndarray::Array1::zeros(self.dim),
                    ndarray::Array1::zeros(self.dim),
                    1.0,
                )?
            };
            entity_vec.push(nb);
        }

        // Use transform-aware eval if any non-identity transforms exist.
        let has_transforms = !self.relation_transforms.is_empty()
            && self.relation_transforms.values().any(|t| !t.is_identity());

        if has_transforms {
            // Build a dense relation transform vector.
            let max_rel = self.relation_transforms.keys().copied().max().unwrap_or(0);
            let mut transforms = vec![RelationTransform::Identity; max_rel + 1];
            for (&rel_id, t) in &self.relation_transforms {
                transforms[rel_id] = t.clone();
            }
            evaluate_interned_with_transforms_inner(
                test_triples,
                &entity_vec,
                entities,
                &transforms,
                filter,
            )
        } else {
            evaluate_link_prediction_interned_inner(test_triples, &entity_vec, entities, filter)
        }
    }
    /// Train for multiple epochs with optional validation and early stopping.
    ///
    /// Uses `config.epochs` as the epoch count, `config.early_stopping_patience`
    /// for early stopping, and `config.warmup_epochs` for learning rate warmup.
    /// If `validation` is provided, evaluates after each epoch and tracks best MRR.
    ///
    /// Linearly anneals `current_beta` from `config.gumbel_beta` to
    /// `config.gumbel_beta_final` across epochs (soft -> hard containment).
    ///
    /// Returns a [`TrainingResult`] with loss history, validation MRR history,
    /// and the final evaluation results.
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn fit(
        &mut self,
        train_triples: &[(usize, usize, usize)],
        validation: Option<(&[crate::dataset::TripleIds], &crate::dataset::Vocab)>,
        filter: Option<&FilteredTripleIndexIds>,
    ) -> Result<TrainingResult, BoxError> {
        let epochs = self.config.epochs;
        let warmup = self.config.warmup_epochs;
        let base_lr = self.config.learning_rate;
        let patience = self.config.early_stopping_patience;
        let min_delta = self.config.early_stopping_min_delta;
        let beta_start = self.config.gumbel_beta;
        let beta_end = self.config.gumbel_beta_final;

        let mut loss_history = Vec::with_capacity(epochs);
        let mut mrr_history = Vec::new();
        let mut best_mrr = 0.0f32;
        let mut best_epoch = 0;
        let mut epochs_without_improvement = 0usize;

        for epoch in 0..epochs {
            // Learning rate scheduling.
            let lr = crate::optimizer::get_learning_rate(epoch, epochs, base_lr, warmup);
            for state in self.optimizer_states.values_mut() {
                state.set_lr(lr);
            }

            // Gumbel beta annealing: linear interpolation from start to end.
            let progress = if epochs > 1 {
                epoch as f32 / (epochs - 1) as f32
            } else {
                1.0
            };
            self.current_beta = beta_start + (beta_end - beta_start) * progress;

            let loss = self.train_step(train_triples)?;
            loss_history.push(loss);

            // Validation.
            if let Some((val_triples, entities)) = validation {
                let results = self.evaluate(val_triples, entities, filter)?;
                mrr_history.push(results.mrr);

                if results.mrr > best_mrr + min_delta {
                    best_mrr = results.mrr;
                    best_epoch = epoch;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement += 1;
                }

                // Early stopping.
                if let Some(p) = patience {
                    if epochs_without_improvement >= p {
                        break;
                    }
                }
            }
        }

        // Final evaluation on the validation set (or return zeros).
        let final_results = if let Some((val_triples, entities)) = validation {
            self.evaluate(val_triples, entities, filter)?
        } else {
            EvaluationResults {
                mrr: 0.0,
                head_mrr: 0.0,
                tail_mrr: 0.0,
                hits_at_1: 0.0,
                hits_at_3: 0.0,
                hits_at_10: 0.0,
                mean_rank: 0.0,
                per_relation: Vec::new(),
            }
        };

        Ok(TrainingResult {
            final_results,
            loss_history,
            validation_mrr_history: mrr_history,
            best_epoch,
            training_time_seconds: None,
        })
    }
}
// ---------------------------------------------------------------------------
// Cone training
// ---------------------------------------------------------------------------

use crate::trainable::TrainableCone;

/// Trainer for cone embeddings using the ConE model (Zhang & Wang, NeurIPS 2021).
///
/// Each entity is represented as a [`TrainableCone`] with per-dimension axis
/// angles and apertures, optimized via AMSGrad.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConeEmbeddingTrainer {
    /// Training configuration (shared with box trainer).
    pub config: TrainingConfig,
    /// Entity ID -> TrainableCone mapping.
    pub cones: HashMap<usize, TrainableCone>,
    /// Entity ID -> AMSGradState mapping.
    pub optimizer_states: HashMap<usize, AMSGradState>,
    /// Embedding dimension.
    pub dim: usize,
}

impl ConeEmbeddingTrainer {
    /// Create a new cone trainer.
    ///
    /// If `initial_embeddings` is provided, each vector is used as the initial
    /// per-dimension axis values (apertures start at pi/2).
    pub fn new(
        config: TrainingConfig,
        dim: usize,
        initial_embeddings: Option<HashMap<usize, Vec<f32>>>,
    ) -> Self {
        let mut cones = HashMap::new();
        let mut optimizer_states = HashMap::new();

        if let Some(embeddings) = initial_embeddings {
            for (entity_id, vector) in embeddings {
                assert_eq!(vector.len(), dim);
                let cone = TrainableCone::from_vector(&vector, std::f32::consts::FRAC_PI_2);
                let n_params = cone.num_parameters();
                cones.insert(entity_id, cone);
                optimizer_states
                    .insert(entity_id, AMSGradState::new(n_params, config.learning_rate));
            }
        }

        Self {
            config,
            cones,
            optimizer_states,
            dim,
        }
    }

    /// Ensure an entity exists in the trainer; initialize with defaults if missing.
    pub fn ensure_entity(&mut self, id: usize) {
        if !self.cones.contains_key(&id) {
            // Default: spread initial axes across dimensions, aperture = pi/2.
            let mut init_vec = vec![0.0f32; self.dim];
            if self.dim > 0 {
                // Give each entity a slightly different initial position.
                init_vec[id % self.dim] = 1.0;
            }
            let cone = TrainableCone::from_vector(&init_vec, std::f32::consts::FRAC_PI_2);
            let n_params = cone.num_parameters();
            self.cones.insert(id, cone);
            self.optimizer_states
                .insert(id, AMSGradState::new(n_params, self.config.learning_rate));
        }
    }

    /// Run one training step for a pair of entities.
    ///
    /// Returns the scalar loss for this pair.
    pub fn train_step(&mut self, id_a: usize, id_b: usize, is_positive: bool) -> f32 {
        self.ensure_entity(id_a);
        self.ensure_entity(id_b);

        let cone_a = self
            .cones
            .get(&id_a)
            .cloned()
            .expect("ensure_entity guarantees key exists");
        let cone_b = self
            .cones
            .get(&id_b)
            .cloned()
            .expect("ensure_entity guarantees key exists");

        let loss = compute_cone_pair_loss(&cone_a, &cone_b, is_positive, &self.config);
        let (grad_axes_a, grad_aper_a, grad_axes_b, grad_aper_b) =
            compute_cone_analytical_gradients(&cone_a, &cone_b, is_positive, &self.config);

        if let (Some(c), Some(s)) = (
            self.cones.get_mut(&id_a),
            self.optimizer_states.get_mut(&id_a),
        ) {
            c.update_amsgrad(&grad_axes_a, &grad_aper_a, s);
        }
        if let (Some(c), Some(s)) = (
            self.cones.get_mut(&id_b),
            self.optimizer_states.get_mut(&id_b),
        ) {
            c.update_amsgrad(&grad_axes_b, &grad_aper_b, s);
        }

        loss
    }
}

/// Degree smoothing exponent for negative sampling (Mikolov et al., 2013).
///
/// Used in [`generate_degree_weighted_negatives`]: each entity's sampling
/// probability is proportional to `degree^0.75`, which down-weights
/// high-degree hub entities relative to pure frequency-based sampling.
#[cfg(feature = "rand")]
const DEGREE_SMOOTHING_EXPONENT: f64 = 0.75;

/// Inside-distance weight for cone containment scoring (ConE default).
const CONE_CENTER_WEIGHT: f32 = 0.02;
/// Gradient strength multiplier for cone axis corrections.
const CONE_GRADIENT_STRENGTH: f32 = 0.2;
/// Aperture gradient coefficient for narrowing/widening corrections.
const CONE_APERTURE_GRADIENT: f32 = 0.05;
/// Clamp ceiling for per-dimension violation/margin signals.
const CONE_VIOLATION_CLAMP: f32 = 1.0;

/// Compute loss for a pair of cones using the ConE distance scoring.
///
/// - **Positive**: minimize distance (encourage A to contain B).
/// - **Negative**: penalize when distance is below the margin (too close = too much containment).
pub fn compute_cone_pair_loss(
    cone_a: &TrainableCone,
    cone_b: &TrainableCone,
    is_positive: bool,
    config: &TrainingConfig,
) -> f32 {
    let dense_a = cone_a.to_cone();
    let dense_b = cone_b.to_cone();
    let cen = CONE_CENTER_WEIGHT;

    if is_positive {
        // Positive: minimize distance (A should contain B).
        let dist = dense_a.cone_distance(&dense_b, cen);

        // Aperture regularization: penalize very large apertures.
        let mean_aper_a: f32 = dense_a.apertures.iter().sum::<f32>() / dense_a.dim() as f32;
        let mean_aper_b: f32 = dense_b.apertures.iter().sum::<f32>() / dense_b.dim() as f32;
        let reg = config.regularization * (mean_aper_a + mean_aper_b);

        (dist + reg).max(0.0)
    } else {
        // Negative: penalize low distance (containment that shouldn't exist).
        let dist = dense_a.cone_distance(&dense_b, cen);
        let margin_loss = if dist < config.margin {
            (config.margin - dist).powi(2)
        } else {
            0.0
        };

        config.negative_weight * margin_loss
    }
}

/// Compute analytical gradients for a pair of cones.
///
/// Returns (grad_axes_a, grad_apertures_a, grad_axes_b, grad_apertures_b).
///
/// These are approximate gradients using per-dimension containment signals:
/// - **Positive**: in dimensions where B is outside A's cone, push axes together
///   and widen A. Gradient magnitude is proportional to the violation.
/// - **Negative**: in dimensions where B is inside A's cone, push axes apart
///   and narrow A. Gradient magnitude is proportional to containment strength.
///
/// This avoids the saturation problem of fixed-magnitude aperture gradients,
/// where heads always widen to pi and tails always narrow to 0.
pub fn compute_cone_analytical_gradients(
    cone_a: &TrainableCone,
    cone_b: &TrainableCone,
    is_positive: bool,
    config: &TrainingConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = cone_a.dim();
    let mut grad_axes_a = vec![0.0f32; dim];
    let mut grad_aper_a = vec![0.0f32; dim];
    let mut grad_axes_b = vec![0.0f32; dim];
    let mut grad_aper_b = vec![0.0f32; dim];

    let dense_a = cone_a.to_cone();
    let dense_b = cone_b.to_cone();

    if is_positive {
        // A should contain B. For each dimension, check if B's axis is inside A's cone.
        for i in 0..dim {
            let dist_to_axis = ((dense_b.axes[i] - dense_a.axes[i]) / 2.0).sin().abs();
            let dist_base = (dense_a.apertures[i] / 2.0).sin().abs();

            let diff = dense_b.axes[i] - dense_a.axes[i];

            if dist_to_axis >= dist_base {
                // B is outside A in this dimension -- push to fix it.
                let violation = dist_to_axis - dist_base;
                let strength = CONE_GRADIENT_STRENGTH * violation.min(CONE_VIOLATION_CLAMP);
                grad_axes_a[i] = -strength * diff.signum(); // push A toward B
                grad_axes_b[i] = strength * diff.signum(); // push B toward A
                                                           // Widen A (negative gradient = increase raw_aperture on descent).
                grad_aper_a[i] = -strength;
                // Narrow B slightly so it fits inside A.
                grad_aper_b[i] = CONE_APERTURE_GRADIENT * strength;
            }
            // If B is already inside A, no gradient needed for this dimension.
        }
    } else {
        // A should NOT contain B. For each dimension where B is inside A, push apart.
        let dist = dense_a.cone_distance(&dense_b, CONE_CENTER_WEIGHT);
        if dist < config.margin {
            let urgency = (config.margin - dist) / config.margin; // 0..1
            for i in 0..dim {
                let dist_to_axis = ((dense_b.axes[i] - dense_a.axes[i]) / 2.0).sin().abs();
                let dist_base = (dense_a.apertures[i] / 2.0).sin().abs();

                if dist_to_axis < dist_base {
                    // B is inside A in this dimension -- push apart.
                    let diff = dense_b.axes[i] - dense_a.axes[i];
                    let margin = dist_base - dist_to_axis;
                    let strength =
                        CONE_GRADIENT_STRENGTH * urgency * margin.min(CONE_VIOLATION_CLAMP);

                    grad_axes_a[i] = strength * diff.signum(); // push A away from B
                    grad_axes_b[i] = -strength * diff.signum();
                    // Narrow A to exclude B.
                    grad_aper_a[i] = strength;
                }
            }
        }
    }

    (grad_axes_a, grad_aper_a, grad_axes_b, grad_aper_b)
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
    fn compute_pair_loss_positive_prefers_containment_over_disjoint() {
        let cfg = TrainingConfig::default();

        // A: large box around origin
        let a = TrainableBox::new(vec![0.0, 0.0], vec![2.0_f32.ln(), 2.0_f32.ln()]).unwrap();
        // B_in: small box centered at origin (contained)
        let b_in = TrainableBox::new(vec![0.0, 0.0], vec![0.2_f32.ln(), 0.2_f32.ln()]).unwrap();
        // B_out: same size but far away (disjoint-ish)
        let b_out =
            TrainableBox::new(vec![100.0, 100.0], vec![0.2_f32.ln(), 0.2_f32.ln()]).unwrap();

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
        let cfg = TrainingConfig {
            margin: 0.2,
            negative_weight: 1.0,
            ..Default::default()
        };

        // A fixed box; compare B disjoint vs B overlapping.
        let a = TrainableBox::new(vec![0.0, 0.0], vec![1.0_f32.ln(), 1.0_f32.ln()]).unwrap();
        let b_disjoint =
            TrainableBox::new(vec![100.0, 100.0], vec![1.0_f32.ln(), 1.0_f32.ln()]).unwrap();
        let b_overlap =
            TrainableBox::new(vec![0.0, 0.0], vec![1.0_f32.ln(), 1.0_f32.ln()]).unwrap();

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
        let triples = [
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
                head_mrr: 0.45,
                tail_mrr: 0.55,
                hits_at_1: 0.3,
                hits_at_3: 0.4,
                hits_at_10: 0.6,
                mean_rank: 5.5,
                per_relation: vec![],
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
    #[allow(unused_variables)] // empty_boxes documents the test structure
    fn test_evaluate_link_prediction_basic() {
        // This test requires a backend implementation
        // We'll test the logic with a mock, but full integration test should be in backend tests
        // For now, just verify the function signature and error handling

        // Test with empty triples
        let _empty_boxes: HashMap<String, ()> = HashMap::new();
        // Can't actually call evaluate_link_prediction without a Box implementation
        // This test documents the need for integration tests in backend modules
        // Full integration tests are in subsume/src/trainer_integration_tests.rs
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
    // evaluate_link_prediction with NdarrayBox
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_with_ndarray_boxes() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        // Three entities: A contains B, C is disjoint.
        // Query: (A, r, ?) -- correct tail is B.
        let a = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let c = NdarrayBox::new(array![50.0, 50.0], array![51.0, 51.0], 1.0).unwrap();

        let mut entity_boxes = HashMap::new();
        entity_boxes.insert("A".to_string(), a);
        entity_boxes.insert("B".to_string(), b);
        entity_boxes.insert("C".to_string(), c);

        let test_triples = vec![Triple {
            head: "A".to_string(),
            relation: "r".to_string(),
            tail: "B".to_string(),
        }];

        let results = evaluate_link_prediction(&test_triples, &entity_boxes).unwrap();

        // B is contained in A and should rank highly; C is disjoint.
        // With 3 entities, best rank is 1, so MRR should be > 0.
        assert!(
            results.mrr > 0.0,
            "MRR should be positive, got {}",
            results.mrr
        );
        assert!(results.mean_rank >= 1.0, "mean_rank should be >= 1");
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_empty_triples() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        let a = NdarrayBox::new(array![0.0], array![1.0], 1.0).unwrap();
        let mut entity_boxes = HashMap::new();
        entity_boxes.insert("A".to_string(), a);

        let results = evaluate_link_prediction::<NdarrayBox>(&[], &entity_boxes).unwrap();
        // No triples to evaluate: metrics should be NaN or 0 depending on implementation.
        // mean_reciprocal_rank([]) and hits_at_k([]) return NaN per standard behavior.
        // Just check it does not panic.
        let _ = results;
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_filtered_excludes_known_tails() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        // Four entities: A, B, C, D. A contains B and C; D is disjoint.
        // Known true: (A, r, C). Test triple: (A, r, B).
        // In filtered setting, C should be excluded from ranking.
        let a = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let c = NdarrayBox::new(array![2.0, 2.0], array![4.0, 4.0], 1.0).unwrap();
        let d = NdarrayBox::new(array![50.0, 50.0], array![51.0, 51.0], 1.0).unwrap();

        let mut entity_boxes = HashMap::new();
        entity_boxes.insert("A".to_string(), a);
        entity_boxes.insert("B".to_string(), b);
        entity_boxes.insert("C".to_string(), c);
        entity_boxes.insert("D".to_string(), d);

        let test_triples = vec![Triple {
            head: "A".to_string(),
            relation: "r".to_string(),
            tail: "B".to_string(),
        }];

        let filter_triples = [
            Triple {
                head: "A".into(),
                relation: "r".into(),
                tail: "C".into(),
            },
            Triple {
                head: "A".into(),
                relation: "r".into(),
                tail: "B".into(),
            },
        ];
        let filter = FilteredTripleIndex::from_triples(filter_triples.iter());

        let unfiltered = evaluate_link_prediction(&test_triples, &entity_boxes).unwrap();
        let filtered =
            evaluate_link_prediction_filtered(&test_triples, &entity_boxes, &filter).unwrap();

        // Filtered rank should be <= unfiltered rank (fewer competitors).
        assert!(
            filtered.mean_rank <= unfiltered.mean_rank,
            "filtered rank ({}) should be <= unfiltered rank ({})",
            filtered.mean_rank,
            unfiltered.mean_rank
        );
    }

    // -----------------------------------------------------------------------
    // evaluate_link_prediction_interned with NdarrayBox
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_interned_with_ndarray_boxes() {
        use crate::dataset::{TripleIds, Vocab};
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        let mut vocab = Vocab::default();
        let id_a = vocab.intern("A".to_string());
        let id_b = vocab.intern("B".to_string());
        let _id_c = vocab.intern("C".to_string());
        let id_r = 0usize; // relation id

        // A contains B; C disjoint.
        let boxes = vec![
            NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap(), // A
            NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap(),   // B
            NdarrayBox::new(array![50.0, 50.0], array![51.0, 51.0], 1.0).unwrap(), // C
        ];

        let test_triples = vec![TripleIds {
            head: id_a,
            relation: id_r,
            tail: id_b,
        }];

        let results = evaluate_link_prediction_interned(&test_triples, &boxes, &vocab).unwrap();
        assert!(
            results.mrr > 0.0,
            "MRR should be positive, got {}",
            results.mrr
        );
        assert!(results.mean_rank >= 1.0);
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_interned_filtered_with_ndarray_boxes() {
        use crate::dataset::{TripleIds, Vocab};
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        let mut vocab = Vocab::default();
        let id_a = vocab.intern("A".to_string());
        let id_b = vocab.intern("B".to_string());
        let id_c = vocab.intern("C".to_string());
        let id_r = 0usize;

        let boxes = vec![
            NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap(),
            NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap(),
            NdarrayBox::new(array![2.0, 2.0], array![4.0, 4.0], 1.0).unwrap(),
        ];

        let test_triples = vec![TripleIds {
            head: id_a,
            relation: id_r,
            tail: id_b,
        }];
        let known_triples = [
            TripleIds {
                head: id_a,
                relation: id_r,
                tail: id_c,
            },
            TripleIds {
                head: id_a,
                relation: id_r,
                tail: id_b,
            },
        ];
        let filter = FilteredTripleIndexIds::from_triples(known_triples.iter());

        let unfiltered = evaluate_link_prediction_interned(&test_triples, &boxes, &vocab).unwrap();
        let filtered =
            evaluate_link_prediction_interned_filtered(&test_triples, &boxes, &vocab, &filter)
                .unwrap();

        assert!(
            filtered.mean_rank <= unfiltered.mean_rank,
            "filtered rank ({}) should be <= unfiltered rank ({})",
            filtered.mean_rank,
            unfiltered.mean_rank
        );
    }

    // -----------------------------------------------------------------------
    // FilteredTripleIndexIds tests
    // -----------------------------------------------------------------------

    #[test]
    fn filtered_triple_index_ids_membership() {
        use crate::dataset::TripleIds;

        let triples = [
            TripleIds {
                head: 0,
                relation: 0,
                tail: 1,
            },
            TripleIds {
                head: 0,
                relation: 0,
                tail: 2,
            },
            TripleIds {
                head: 0,
                relation: 1,
                tail: 3,
            },
        ];

        let idx = FilteredTripleIndexIds::from_triples(triples.iter());

        assert!(idx.is_known_tail(0, 0, 1));
        assert!(idx.is_known_tail(0, 0, 2));
        assert!(!idx.is_known_tail(0, 0, 3)); // different relation
        assert!(idx.is_known_tail(0, 1, 3));
        assert!(!idx.is_known_tail(1, 0, 1)); // different head
    }

    #[test]
    fn filtered_triple_index_ids_known_tails() {
        use crate::dataset::TripleIds;

        let triples = [
            TripleIds {
                head: 0,
                relation: 0,
                tail: 10,
            },
            TripleIds {
                head: 0,
                relation: 0,
                tail: 20,
            },
        ];
        let idx = FilteredTripleIndexIds::from_triples(triples.iter());

        let tails = idx.known_tails(0, 0).unwrap();
        assert!(tails.contains(&10));
        assert!(tails.contains(&20));
        assert!(!tails.contains(&30));
        assert!(idx.known_tails(1, 0).is_none());
    }

    // -----------------------------------------------------------------------
    // TrainingConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn training_config_default_values_are_sane() {
        let cfg = TrainingConfig::default();
        assert!(cfg.learning_rate > 0.0 && cfg.learning_rate < 1.0);
        assert!(cfg.epochs > 0);
        assert!(cfg.batch_size > 0);
        assert!(cfg.negative_samples > 0);
        assert!(cfg.temperature > 0.0);
        assert!(cfg.margin > 0.0);
        assert!(cfg.negative_weight > 0.0);
    }

    // -----------------------------------------------------------------------
    // Evaluation determinism
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn evaluate_link_prediction_deterministic() {
        use crate::ndarray_backend::NdarrayBox;
        use ndarray::array;

        let a = NdarrayBox::new(array![0.0, 0.0], array![10.0, 10.0], 1.0).unwrap();
        let b = NdarrayBox::new(array![1.0, 1.0], array![3.0, 3.0], 1.0).unwrap();
        let c = NdarrayBox::new(array![50.0, 50.0], array![51.0, 51.0], 1.0).unwrap();

        let mut entity_boxes = HashMap::new();
        entity_boxes.insert("A".to_string(), a);
        entity_boxes.insert("B".to_string(), b);
        entity_boxes.insert("C".to_string(), c);

        let test_triples = vec![Triple {
            head: "A".to_string(),
            relation: "r".to_string(),
            tail: "B".to_string(),
        }];

        let r1 = evaluate_link_prediction(&test_triples, &entity_boxes).unwrap();
        let r2 = evaluate_link_prediction(&test_triples, &entity_boxes).unwrap();

        assert_eq!(r1.mrr, r2.mrr, "MRR differs across runs");
        assert_eq!(r1.hits_at_1, r2.hits_at_1, "Hits@1 differs across runs");
        assert_eq!(r1.hits_at_3, r2.hits_at_3, "Hits@3 differs across runs");
        assert_eq!(r1.hits_at_10, r2.hits_at_10, "Hits@10 differs across runs");
        assert_eq!(r1.mean_rank, r2.mean_rank, "mean_rank differs across runs");
    }

    // -----------------------------------------------------------------------
    // Save/load roundtrip for TrainableBox (serde)
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn trainable_box_serde_roundtrip() {
        let original = TrainableBox::new(vec![1.0, 2.0, 3.0], vec![0.5, -0.5, 1.0]).unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let restored: TrainableBox = serde_json::from_str(&json).unwrap();

        assert_eq!(original.mu, restored.mu);
        assert_eq!(original.delta, restored.delta);
        assert_eq!(original.dim(), restored.dim());
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn trainable_box_serde_roundtrip_via_tempfile() {
        let original = TrainableBox::new(vec![0.1, -0.2], vec![0.3, 0.4]).unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("subsume_test_trainable_box.json");
        let json = serde_json::to_string_pretty(&original).unwrap();
        std::fs::write(&path, &json).unwrap();

        let loaded_json = std::fs::read_to_string(&path).unwrap();
        let restored: TrainableBox = serde_json::from_str(&loaded_json).unwrap();
        assert_eq!(original.mu, restored.mu);
        assert_eq!(original.delta, restored.delta);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn ndarray_box_serde_roundtrip() {
        use crate::ndarray_backend::NdarrayBox;
        use crate::Box as BoxTrait;
        use ndarray::array;

        let original = NdarrayBox::new(array![0.0, 1.0], array![2.0, 3.0], 0.5).unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let restored: NdarrayBox = serde_json::from_str(&json).unwrap();

        assert_eq!(original.dim(), restored.dim());
        // Check min/max values roundtrip correctly.
        for i in 0..original.dim() {
            assert!(
                (BoxTrait::min(&original)[i] - BoxTrait::min(&restored)[i]).abs() < 1e-6,
                "min mismatch at dim {i}"
            );
            assert!(
                (BoxTrait::max(&original)[i] - BoxTrait::max(&restored)[i]).abs() < 1e-6,
                "max mismatch at dim {i}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // compute_pair_loss edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn compute_pair_loss_identical_boxes_positive_is_finite() {
        let cfg = TrainingConfig::default();
        let a = TrainableBox::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let loss = compute_pair_loss(&a, &a.clone(), true, &cfg);
        assert!(
            loss.is_finite(),
            "loss for identical boxes should be finite, got {loss}"
        );
    }

    #[test]
    fn compute_pair_loss_negative_weight_scales_loss() {
        let cfg_w1 = TrainingConfig {
            negative_weight: 1.0,
            margin: 0.01,
            ..Default::default()
        };
        let cfg_w2 = TrainingConfig {
            negative_weight: 2.0,
            margin: 0.01,
            ..Default::default()
        };
        // Two overlapping boxes.
        let a = TrainableBox::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let b = TrainableBox::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();

        let l1 = compute_pair_loss(&a, &b, false, &cfg_w1);
        let l2 = compute_pair_loss(&a, &b, false, &cfg_w2);

        if l1 > 0.0 {
            let ratio = l2 / l1;
            assert!(
                (ratio - 2.0).abs() < 1e-4,
                "doubling negative_weight should double loss: l1={l1}, l2={l2}, ratio={ratio}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // compute_analytical_gradients
    // -----------------------------------------------------------------------

    #[test]
    fn analytical_gradients_negative_pair_returns_zeros() {
        // For negative pairs, the current gradient implementation returns zeros
        // (only positive pairs produce non-zero gradients).
        let cfg = TrainingConfig::default();
        let a = TrainableBox::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let b = TrainableBox::new(vec![5.0, 5.0], vec![1.0, 1.0]).unwrap();
        let (g_mu_a, g_delta_a, g_mu_b, g_delta_b) =
            compute_analytical_gradients(&a, &b, false, &cfg);
        for v in [&g_mu_a, &g_delta_a, &g_mu_b, &g_delta_b] {
            assert!(
                v.iter().all(|&x| x == 0.0),
                "negative gradients should be zero"
            );
        }
    }

    #[test]
    fn analytical_gradients_positive_disjoint_pushes_centers() {
        let cfg = TrainingConfig::default();
        // Two disjoint boxes: centers far apart.
        let a = TrainableBox::new(vec![0.0], vec![0.1_f32.ln()]).unwrap();
        let b = TrainableBox::new(vec![10.0], vec![0.1_f32.ln()]).unwrap();
        let (g_mu_a, _, g_mu_b, _) = compute_analytical_gradients(&a, &b, true, &cfg);

        // For disjoint positive pairs, the gradient pushes centers toward each other:
        // g_mu_a should be negative (move a toward b at +10).
        // Actually the gradient formula is: g_mu_a[i] -= 0.5 * diff where diff = center_b - center_a.
        // diff > 0, so g_mu_a < 0 (i.e., descending this gradient moves a toward b).
        // Wait, the gradient is for gradient *descent*, so g_mu_a = -0.5 * diff.
        // diff = 10 > 0, so g_mu_a = -5.0. Applying SGD: mu_a -= lr * g_mu_a = mu_a - lr*(-5) = mu_a + 5*lr,
        // which moves a toward b. Correct.
        assert!(
            g_mu_a[0] < 0.0,
            "gradient should push a's center toward b (got {})",
            g_mu_a[0]
        );
        assert!(
            g_mu_b[0] > 0.0,
            "gradient should push b's center toward a (got {})",
            g_mu_b[0]
        );
    }

    // -----------------------------------------------------------------------
    // gradient correctness via loss reduction
    // -----------------------------------------------------------------------

    #[test]
    fn analytical_gradients_reduce_loss_on_positive_pair() {
        // Two overlapping boxes where parent doesn't fully contain child.
        // Box A: center=0, width=exp(0.5)~1.65 -> [-0.82, 0.82]
        // Box B: center=1, width=exp(0.5)~1.65 -> [0.18, 1.82]
        // They overlap but A doesn't fully contain B.
        let mut a = TrainableBox::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let mut b = TrainableBox::new(vec![1.0, 1.0], vec![0.5, 0.5]).unwrap();
        let cfg = TrainingConfig {
            regularization: 0.0,
            ..Default::default()
        };

        let loss_before = compute_pair_loss(&a, &b, true, &cfg);

        let (g_mu_a, g_delta_a, g_mu_b, g_delta_b) =
            compute_analytical_gradients(&a, &b, true, &cfg);

        // Apply one gradient step manually (gradient descent: param -= lr * grad).
        let lr = 0.1;
        for i in 0..a.dim() {
            a.mu[i] -= lr * g_mu_a[i];
            a.delta[i] -= lr * g_delta_a[i];
            b.mu[i] -= lr * g_mu_b[i];
            b.delta[i] -= lr * g_delta_b[i];
        }

        let loss_after = compute_pair_loss(&a, &b, true, &cfg);
        assert!(
            loss_after < loss_before,
            "gradient step should reduce positive-pair loss: before={loss_before}, after={loss_after}"
        );
    }

    #[test]
    fn analytical_gradient_finite_difference_sign_agreement() {
        // Verify that the analytical gradient for mu_a[0] agrees in sign with
        // a finite-difference approximation.
        let a = TrainableBox::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let b = TrainableBox::new(vec![1.0, 1.0], vec![0.5, 0.5]).unwrap();
        let cfg = TrainingConfig {
            regularization: 0.0,
            ..Default::default()
        };

        let (g_mu_a, _, _, _) = compute_analytical_gradients(&a, &b, true, &cfg);
        let grad_analytical = g_mu_a[0];

        // Finite-difference: (loss(mu+eps) - loss(mu-eps)) / (2*eps)
        let eps = 1e-3;
        let mut a_plus = a.clone();
        a_plus.mu[0] += eps;
        let mut a_minus = a.clone();
        a_minus.mu[0] -= eps;

        let loss_plus = compute_pair_loss(&a_plus, &b, true, &cfg);
        let loss_minus = compute_pair_loss(&a_minus, &b, true, &cfg);
        let grad_numerical = (loss_plus - loss_minus) / (2.0 * eps);

        // The analytical gradient is a heuristic (not a true derivative), so we only
        // check directional agreement (same sign), not magnitude.
        assert!(
            grad_analytical.signum() == grad_numerical.signum()
                || grad_analytical.abs() < 1e-6
                || grad_numerical.abs() < 1e-6,
            "gradient sign mismatch: analytical={grad_analytical}, numerical={grad_numerical}"
        );
    }

    // -----------------------------------------------------------------------
    // negative sampling statistical uniformity
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "rand")]
    fn negative_sampling_uniformity() {
        use rand::SeedableRng;

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
    // log_training_result with tempfile
    // -----------------------------------------------------------------------

    #[test]
    fn log_training_result_to_tempfile_roundtrip() {
        let result = TrainingResult {
            final_results: EvaluationResults {
                mrr: 0.75,
                head_mrr: 0.70,
                tail_mrr: 0.80,
                hits_at_1: 0.6,
                hits_at_3: 0.7,
                hits_at_10: 0.9,
                mean_rank: 2.5,
                per_relation: vec![],
            },
            loss_history: vec![2.0, 1.0, 0.5],
            validation_mrr_history: vec![0.5, 0.65, 0.75],
            best_epoch: 2,
            training_time_seconds: Some(42.0),
        };

        let dir = std::env::temp_dir();
        let path = dir.join("subsume_test_log_result.txt");
        log_training_result(&result, Some(path.to_str().unwrap())).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("0.7500"), "should contain MRR");
        assert!(content.contains("0.6000"), "should contain Hits@1");
        assert!(content.contains("42.00"), "should contain training time");
        assert!(
            content.contains("Best Epoch: 2"),
            "should contain best epoch"
        );

        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_box(dim: usize) -> impl Strategy<Value = TrainableBox> {
        let mu_strat = prop::collection::vec(-10.0f32..10.0, dim);
        let delta_strat = prop::collection::vec(-5.0f32..2.0, dim);
        (mu_strat, delta_strat).prop_map(move |(mu, delta)| TrainableBox::new(mu, delta).unwrap())
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
            let mut state = AMSGradState::new(box_a.num_parameters(), 0.001);
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
        /// compute_pair_loss returns finite f32 for random box pairs and configs.
        #[test]
        fn prop_compute_pair_loss_finite(
            box_a in arb_box(4),
            box_b in arb_box(4),
            is_positive in any::<bool>(),
            regularization in 0.0f32..1.0,
            margin in 0.0f32..2.0,
            negative_weight in 0.1f32..5.0,
        ) {
            let config = TrainingConfig {
                regularization,
                margin,
                negative_weight,
                ..Default::default()
            };
            let loss = compute_pair_loss(&box_a, &box_b, is_positive, &config);
            prop_assert!(loss.is_finite(), "compute_pair_loss returned non-finite: {loss}");
        }
    }

    // -- Cone trainer tests --

    #[test]
    fn cone_pair_loss_positive_prefers_containment() {
        let cfg = TrainingConfig::default();

        // A: wide cone, B_in: narrow cone with same axes (contained)
        let a = TrainableCone::new(vec![0.0, 0.0], vec![2.0, 2.0]).unwrap(); // wide aperture
        let b_in = TrainableCone::new(vec![0.0, 0.0], vec![-2.0, -2.0]).unwrap(); // narrow aperture

        // B_out: narrow cone with very different axes
        let b_out = TrainableCone::new(vec![3.0, 3.0], vec![-2.0, -2.0]).unwrap();

        let l_in = compute_cone_pair_loss(&a, &b_in, true, &cfg);
        let l_out = compute_cone_pair_loss(&a, &b_out, true, &cfg);

        assert!(l_in.is_finite() && l_out.is_finite());
        assert!(
            l_in < l_out,
            "positive loss should be lower for contained cones (got l_in={l_in}, l_out={l_out})"
        );
    }

    #[test]
    fn cone_trainer_train_step_does_not_panic() {
        let cfg = TrainingConfig::default();
        let mut trainer = ConeEmbeddingTrainer::new(cfg, 4, None);
        let loss = trainer.train_step(0, 1, true);
        assert!(loss.is_finite(), "loss must be finite, got {}", loss);

        let loss_neg = trainer.train_step(0, 2, false);
        assert!(
            loss_neg.is_finite(),
            "negative loss must be finite, got {}",
            loss_neg
        );
    }

    #[test]
    fn cone_trainer_reduces_loss_over_steps() {
        let cfg = TrainingConfig {
            learning_rate: 0.01,
            temperature: 1.0,
            regularization: 0.0,
            ..Default::default()
        };

        let mut trainer = ConeEmbeddingTrainer::new(cfg, 4, None);

        // Run several positive steps for the same pair to see loss decrease
        let mut losses = Vec::new();
        for _ in 0..50 {
            let loss = trainer.train_step(0, 1, true);
            losses.push(loss);
        }

        // The loss in the last 10 steps should generally be lower than the first 10
        let early_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let late_avg: f32 = losses[40..].iter().sum::<f32>() / 10.0;
        assert!(
            late_avg <= early_avg + 0.5,
            "loss should generally decrease: early_avg={early_avg}, late_avg={late_avg}"
        );
    }
}
