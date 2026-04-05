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
//!   tie-break so `same model + same data => same metrics`.
//! - **NaNs are treated as hard errors** in evaluation: silently propagating NaNs yields meaningless metrics.

/// Annular sector trainer with finite-difference gradients.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod annular_trainer;
/// Ball embedding trainer with analytical gradients.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod ball_trainer;
/// Box embedding trainer, loss computation, and analytical gradients.
pub mod box_trainer;
/// Burn-based annular sector trainer with autodiff (multi-backend: ndarray/wgpu/tch).
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub mod burn_annular_trainer;
/// Burn-based ball trainer with autodiff (multi-backend: ndarray/wgpu/tch).
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub mod burn_ball_trainer;
/// Burn-based spherical cap trainer with autodiff (multi-backend: ndarray/wgpu/tch).
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub mod burn_cap_trainer;
/// Burn-based diagonal-Gaussian ellipsoid trainer with autodiff (multi-backend: ndarray/wgpu/tch).
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub mod burn_ellipsoid_trainer;
/// Burn-based subspace trainer with autodiff (multi-backend: ndarray/wgpu/tch).
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub mod burn_subspace_trainer;
/// Burn-based TransBox trainer with autodiff (multi-backend: ndarray/wgpu/tch).
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub mod burn_transbox_trainer;
/// Cone embedding trainer, loss computation, and analytical gradients.
pub mod cone_trainer;
/// Ellipsoid embedding trainer with analytical gradients.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod ellipsoid_trainer;
/// Filtered indices and link prediction evaluation functions.
pub mod evaluation;
/// Negative sampling functions for knowledge graph training.
pub mod negative_sampling;
/// Spherical cap embedding trainer with analytical gradients.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod spherical_cap_trainer;
/// Subspace embedding trainer with analytical gradients.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod subspace_trainer;
/// Shared Adam state and self-adversarial weighting utilities.
pub mod trainer_utils;
/// TransBox trainer with analytical gradients.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod transbox_trainer;

/// Candle-based box trainer with autograd (GPU-accelerated).
#[cfg(feature = "candle-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "candle-backend")))]
pub mod candle_trainer;

/// Candle-based EL++ trainer with autograd (GPU-accelerated).
#[cfg(feature = "candle-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "candle-backend")))]
pub mod candle_el_trainer;

/// Candle-based cone trainer with autograd (GPU-accelerated).
#[cfg(feature = "candle-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "candle-backend")))]
pub mod candle_cone_trainer;

use crate::BoxError;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

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
    ///
    /// # Panics
    ///
    /// Debug-asserts that `min`, `max`, and the offset vector (if [`Translation`](RelationTransform::Translation))
    /// all have the same length. In release builds, mismatched lengths silently
    /// truncate via `zip`.
    pub fn apply_to_bounds(&self, min: &[f32], max: &[f32]) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(min.len(), max.len(), "min/max length mismatch");
        match self {
            RelationTransform::Identity => (min.to_vec(), max.to_vec()),
            RelationTransform::Translation(offset) => {
                debug_assert_eq!(
                    offset.len(),
                    min.len(),
                    "Translation offset length ({}) != bounds length ({})",
                    offset.len(),
                    min.len()
                );
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
/// ## Box-Specific Parameters
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
///
/// $$
/// L_{\text{total}} = L_{\text{ranking}} + \lambda_{\text{reg}} \cdot L_{\text{volume}}
/// $$
///
/// where:
/// - `L_ranking` is the margin-based ranking loss
/// - `L_volume` is volume regularization (penalizes large boxes)
/// # Field usage
///
/// Fields consumed by [`BoxEmbeddingTrainer::train_step`]: `learning_rate`, `margin`,
/// `regularization`, `negative_weight`, `negative_samples`, `negative_strategy`,
/// `softplus_beta`, `max_grad_norm`, `adversarial_temperature`, `self_adversarial`,
/// `use_infonce`.
///
/// The remaining fields (`epochs`, `batch_size`,
/// `early_stopping_*`, `warmup_epochs`, `softplus_beta_final`) are configuration
/// metadata for caller-side training loops (e.g., [`BoxEmbeddingTrainer::fit`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CpuBoxTrainingConfig {
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
    /// Softplus steepness for intersection volume approximation.
    ///
    /// `softplus(beta * (hi - lo), 1.0)` replaces the hard `max(0, hi - lo)`.
    /// Lower values give broader gradients for disjoint boxes; higher values
    /// approach the hard intersection. Annealed toward `softplus_beta_final`
    /// across epochs when using `BoxEmbeddingTrainer::fit`.
    ///
    /// Default: `10.0`.
    ///
    /// Previously named `gumbel_beta`; the serde alias preserves backward
    /// compatibility with existing checkpoints.
    #[serde(alias = "gumbel_beta")]
    pub softplus_beta: f32,

    /// Final value of `softplus_beta` after annealing across epochs.
    ///
    /// In `BoxEmbeddingTrainer::fit`, `softplus_beta` is linearly interpolated
    /// from its initial value to this value over the training epochs.
    /// Higher final beta gives sharper containment boundaries at convergence.
    ///
    /// Default: `50.0`.
    #[serde(alias = "gumbel_beta_final")]
    pub softplus_beta_final: f32,

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

    /// Use symmetric loss `min(P(A|B), P(B|A))` instead of directed `P(B|A)`.
    ///
    /// The default directed loss `-ln P(B ⊆ A)` matches the evaluation metric
    /// (`containment_prob_fast`), which is correct for hierarchical relations
    /// (hypernym, part-of, subclass). Set to `true` for datasets with mostly
    /// symmetric relations (similar-to, sibling-of).
    ///
    /// Default: `false` (directed loss).
    #[serde(default)]
    pub symmetric_loss: bool,

    /// Enable self-adversarial negative sampling (Sun et al., RotatE ICLR 2019).
    ///
    /// When true, negative samples are weighted by softmax of their current
    /// model score scaled by `adversarial_temperature`:
    ///
    /// $$
    /// w_i = \frac{\exp(\alpha \cdot s_i)}{\sum_j \exp(\alpha \cdot s_j)}
    /// $$
    ///
    /// where `s_i` is the model's containment score for negative `i` and
    /// `alpha` is `adversarial_temperature`. This focuses gradient signal
    /// on "hard" negatives that the model currently scores highly.
    ///
    /// When false, all negatives are weighted uniformly.
    ///
    /// Default: `false`.
    #[serde(default)]
    pub self_adversarial: bool,

    /// Use Bernoulli negative sampling (Wang et al., 2014).
    ///
    /// When enabled, the probability of corrupting head vs tail is adjusted
    /// per relation based on cardinality statistics computed from training data:
    /// - `P(corrupt_head) = tph / (tph + hpt)` where `tph` = avg tails per head,
    ///   `hpt` = avg heads per tail.
    /// - For 1-to-N relations (many tails per head), this increases the probability
    ///   of corrupting the head, producing harder negatives.
    ///
    /// Only affects sampling when `negative_strategy` is `Uniform`.
    /// Ignored for `CorruptHead`, `CorruptTail`, and `CorruptBoth`.
    ///
    /// Default: `false` (uniform 50/50 head/tail corruption).
    #[serde(default)]
    pub bernoulli_sampling: bool,

    /// Steepness of the sigmoid in the ball containment scoring function.
    ///
    /// `containment_prob = sigmoid(k * margin)` where `margin = r_outer - dist - r_inner`.
    /// Lower k gives broader gradients (better for random initialisation); higher k
    /// sharpens the decision boundary. `k=2` is recommended for WN18RR ball training;
    /// `k=10` matches the CPU ball trainer default.
    ///
    /// Default: `2.0`.
    #[serde(default = "default_sigmoid_k")]
    pub sigmoid_k: f32,
}

fn default_sigmoid_k() -> f32 {
    2.0
}

impl Default for CpuBoxTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3, // Paper default: 1e-3 to 5e-4
            epochs: 100,
            batch_size: 512, // Paper range: 512-4096
            negative_samples: 1,
            negative_strategy: NegativeSamplingStrategy::CorruptTail,
            margin: 1.0,                       // Margin for ranking loss
            early_stopping_patience: Some(10), // Early stopping after 10 epochs without improvement
            early_stopping_min_delta: 0.001,
            regularization: 0.0001,
            warmup_epochs: 10,
            negative_weight: 1.0,
            softplus_beta: 10.0,
            softplus_beta_final: 50.0,
            max_grad_norm: 10.0,
            adversarial_temperature: 1.0,
            use_infonce: false,
            symmetric_loss: false,
            self_adversarial: false,
            bernoulli_sampling: false,
            sigmoid_k: 2.0,
        }
    }
}

impl CpuBoxTrainingConfig {
    /// Validate that all fields have sensible values.
    ///
    /// Call after deserialization to catch invalid configs early.
    /// Returns the first validation error found, if any.
    pub fn validate(&self) -> Result<(), BoxError> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(BoxError::Internal(format!(
                "learning_rate must be positive and finite, got {}",
                self.learning_rate
            )));
        }
        if self.batch_size == 0 {
            return Err(BoxError::Internal("batch_size must be > 0".to_string()));
        }
        if self.negative_samples == 0 {
            return Err(BoxError::Internal(
                "negative_samples must be > 0".to_string(),
            ));
        }
        if !self.margin.is_finite() || self.margin < 0.0 {
            return Err(BoxError::Internal(format!(
                "margin must be non-negative and finite, got {}",
                self.margin
            )));
        }
        if !self.softplus_beta.is_finite() || self.softplus_beta <= 0.0 {
            return Err(BoxError::Internal(format!(
                "softplus_beta must be positive and finite, got {}",
                self.softplus_beta
            )));
        }
        if !self.max_grad_norm.is_finite() || self.max_grad_norm <= 0.0 {
            return Err(BoxError::Internal(format!(
                "max_grad_norm must be positive and finite, got {}",
                self.max_grad_norm
            )));
        }
        if !self.sigmoid_k.is_finite() || self.sigmoid_k <= 0.0 {
            return Err(BoxError::Internal(format!(
                "sigmoid_k must be positive and finite, got {}",
                self.sigmoid_k
            )));
        }
        Ok(())
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

// ---------------------------------------------------------------------------
// Re-exports: maintain the same public API as the old single-file module
// ---------------------------------------------------------------------------

// Negative sampling
pub use negative_sampling::{compute_relation_cardinalities, RelationCardinality};

// Evaluation
pub use evaluation::{
    evaluate_link_prediction, evaluate_link_prediction_filtered, evaluate_link_prediction_interned,
    evaluate_link_prediction_interned_filtered, FilteredTripleIndex, FilteredTripleIndexIds,
};

#[cfg(feature = "ndarray-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
pub use evaluation::evaluate_link_prediction_interned_with_transforms;

// Box trainer
pub use box_trainer::{compute_analytical_gradients, compute_pair_loss, BoxEmbeddingTrainer};

// Cone trainer
pub use cone_trainer::{
    compute_cone_analytical_gradients, compute_cone_pair_loss, ConeEmbeddingTrainer,
};

// Ball trainer
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub use ball_trainer::BallTrainer;

// Spherical cap trainer
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub use spherical_cap_trainer::SphericalCapTrainer;

// Subspace trainer
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub use subspace_trainer::SubspaceTrainer;

// Ellipsoid trainer
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub use ellipsoid_trainer::EllipsoidTrainer;

// TransBox trainer
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub use transbox_trainer::TransBoxTrainer;

// Annular trainer
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub use annular_trainer::AnnularTrainer;

// Burn ball trainer
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub use burn_ball_trainer::BurnBallTrainer;

// Burn cap trainer
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub use burn_cap_trainer::BurnCapTrainer;

// Burn ellipsoid trainer
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub use burn_ellipsoid_trainer::BurnEllipsoidTrainer;

// Burn subspace trainer
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub use burn_subspace_trainer::BurnSubspaceTrainer;

// Burn TransBox trainer
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub use burn_transbox_trainer::BurnTransBoxTrainer;

// Burn annular trainer
#[cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "burn-ndarray", feature = "burn-wgpu"))))]
pub use burn_annular_trainer::BurnAnnularTrainer;

/// Backward-compatible alias for [`CpuBoxTrainingConfig`].
///
/// This type is only used by CPU trainers (`BoxEmbeddingTrainer`,
/// `ConeEmbeddingTrainer`). Candle trainers use builder-style constructors.
pub type TrainingConfig = CpuBoxTrainingConfig;
