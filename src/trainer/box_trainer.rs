use crate::optimizer::AMSGradState;
use crate::trainable::TrainableBox;
use crate::BoxError;
use std::collections::{HashMap, HashSet};

#[cfg(feature = "rand")]
use rand::seq::SliceRandom;
#[cfg(feature = "rand")]
use rand::{Rng, SeedableRng};

use super::negative_sampling::RelationCardinality;
#[cfg(feature = "ndarray-backend")]
use super::{EvaluationResults, TrainingResult};
use super::{NegativeSamplingStrategy, RelationTransform, TrainingConfig};
#[cfg(feature = "ndarray-backend")]
use crate::trainer::evaluation::evaluate_interned_with_transforms_inner;
#[cfg(feature = "ndarray-backend")]
use crate::trainer::evaluation::{evaluate_link_prediction_interned_inner, FilteredTripleIndexIds};

/// Compute loss for a pair of boxes.
///
/// For **positive** examples (default directed mode): `-ln P(B ⊆ A)` where
/// `P(B ⊆ A) = Vol(A ∩ B) / Vol(B)`. This directly matches the evaluation
/// metric (`containment_prob_fast`).
///
/// When `config.symmetric_loss` is `true`, uses the symmetric formulation
/// `min(P(B ⊆ A), P(A ⊆ B))` instead. This is appropriate for datasets
/// with symmetric relations.
///
/// For **negative** examples: `w_neg * max(0, max(P(A|B), P(B|A)) - margin)^2`.
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
        let prob = if config.symmetric_loss {
            let p_a_b = (vol_int_soft / vol_b).clamp(1e-8, 1.0);
            let p_b_a = (vol_int_soft / vol_a).clamp(1e-8, 1.0);
            p_a_b.min(p_b_a)
        } else {
            // Directed: P(B ⊆ A) = Vol(A ∩ B) / Vol(B)
            (vol_int_soft / vol_b).clamp(1e-8, 1.0)
        };
        // Cap at 10.0 to prevent explosion from near-zero probabilities.
        let neg_log_prob = (-prob.ln()).min(10.0);

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
/// For **positive** pairs (directed, default): the loss is `-ln P(B ⊆ A) + reg * (Vol_A + Vol_B)`
/// where `P(B ⊆ A) = Vol(A ∩ B) / Vol(B)`. When `config.symmetric_loss` is true,
/// uses `min(P(A|B), P(B|A))` instead.
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
        // Positive loss (directed): L = -ln P(B ⊆ A) + reg * (Vol_A + Vol_B)
        // where P(B ⊆ A) = Vol_int / Vol_B.
        // When symmetric_loss: L = -ln(min(P_AB, P_BA)).

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

        // Directed: always use P(B ⊆ A) = Vol_int / Vol_B.
        // Symmetric: use min(P_AB, P_BA).
        let (p, use_ab) = if config.symmetric_loss {
            if p_ab <= p_ba {
                (p_ab, true)
            } else {
                (p_ba, false)
            }
        } else {
            (p_ab, true)
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
    /// Learned per-relation translation vectors (relation_id -> Vec<f32> of length `dim`).
    /// Applied to head box before containment scoring. Initialized to zeros.
    #[serde(default)]
    pub relation_translations: HashMap<usize, Vec<f32>>,
    /// AMSGrad optimizer state for per-relation translation vectors.
    #[serde(default)]
    pub relation_optimizer_states: HashMap<usize, AMSGradState>,
    /// Cached per-relation cardinality statistics for Bernoulli negative sampling.
    /// Computed from training triples when `config.bernoulli_sampling` is true.
    #[serde(skip)]
    pub(crate) relation_cardinalities: HashMap<usize, RelationCardinality>,
}

impl BoxEmbeddingTrainer {
    /// Create a new box embedding trainer.
    ///
    /// This constructor does not validate the config (it cannot return `Result`
    /// without a breaking API change). Call [`TrainingConfig::validate`] after
    /// deserializing a config from an untrusted source. [`fit`](Self::fit)
    /// validates automatically before training.
    pub fn new(config: TrainingConfig, dim: usize) -> Self {
        let current_beta = config.gumbel_beta;
        Self {
            config,
            boxes: HashMap::new(),
            optimizer_states: HashMap::new(),
            dim,
            relation_transforms: HashMap::new(),
            current_beta,
            relation_translations: HashMap::new(),
            relation_optimizer_states: HashMap::new(),
            relation_cardinalities: HashMap::new(),
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

    /// Ensure a relation translation vector exists; initialize to zeros if missing.
    fn ensure_relation(&mut self, rel_id: usize) {
        if !self.relation_translations.contains_key(&rel_id) {
            self.relation_translations
                .insert(rel_id, vec![0.0f32; self.dim]);
            self.relation_optimizer_states.insert(
                rel_id,
                AMSGradState::new(self.dim, self.config.learning_rate),
            );
        }
    }

    /// Get all known entity IDs (for full-pool negative sampling).
    fn all_entity_ids(&self) -> Vec<usize> {
        self.boxes.keys().copied().collect()
    }

    /// Sample a negative (head, tail) pair for a given positive triple.
    ///
    /// Respects `config.negative_strategy` and `config.bernoulli_sampling`.
    /// When Bernoulli sampling is enabled and strategy is `Uniform`, the
    /// head/tail corruption probability is adjusted by relation cardinality.
    #[cfg(feature = "rand")]
    fn sample_negative(
        &self,
        h: usize,
        r: usize,
        t: usize,
        all_entities: &[usize],
        config: &TrainingConfig,
        rng: &mut impl Rng,
    ) -> (usize, usize) {
        let corrupt_head = match &config.negative_strategy {
            NegativeSamplingStrategy::CorruptHead => true,
            NegativeSamplingStrategy::CorruptTail => false,
            NegativeSamplingStrategy::CorruptBoth => {
                // Corrupt both: pick two independent replacements.
                let nh = loop {
                    let c = all_entities[rng.random_range(0..all_entities.len())];
                    if c != h {
                        break c;
                    }
                };
                let nt = loop {
                    let c = all_entities[rng.random_range(0..all_entities.len())];
                    if c != t {
                        break c;
                    }
                };
                return (nh, nt);
            }
            NegativeSamplingStrategy::Uniform => {
                if config.bernoulli_sampling {
                    // Bernoulli: P(corrupt_head) = tph / (tph + hpt).
                    let p_head = self
                        .relation_cardinalities
                        .get(&r)
                        .map(|c| c.head_corrupt_prob())
                        .unwrap_or(0.5);
                    rng.random::<f32>() < p_head
                } else {
                    rng.random::<bool>()
                }
            }
        };

        if corrupt_head {
            let nh = loop {
                let c = all_entities[rng.random_range(0..all_entities.len())];
                if c != h {
                    break c;
                }
            };
            (nh, t)
        } else {
            let nt = loop {
                let c = all_entities[rng.random_range(0..all_entities.len())];
                if c != t {
                    break c;
                }
            };
            (h, nt)
        }
    }

    /// Train one mini-batch with gradient accumulation.
    ///
    /// Unlike `train_step` (which applies updates per-triple), this method
    /// accumulates gradients across all triples in the batch and applies
    /// a single averaged update per entity. This produces more stable
    /// learning dynamics.
    ///
    /// When `rng` is provided, negatives are sampled randomly from the
    /// full entity pool. Otherwise falls back to deterministic hash-based sampling.
    #[cfg(feature = "rand")]
    fn train_step_minibatch(
        &mut self,
        triples: &[(usize, usize, usize)],
        all_entities: &[usize],
        rng: &mut impl Rng,
    ) -> Result<f32, BoxError> {
        if triples.is_empty() {
            return Ok(0.0);
        }

        let mut step_config = self.config.clone();
        step_config.gumbel_beta = self.current_beta;

        let n_neg = step_config.negative_samples.max(1);
        let mut total_loss = 0.0f32;
        let mut n_pairs = 0usize;

        // Gradient accumulators: entity_id -> (grad_mu_sum, grad_delta_sum, count)
        let mut grad_accum: HashMap<usize, (Vec<f32>, Vec<f32>, usize)> = HashMap::new();
        // Relation translation gradient accumulators: rel_id -> (grad_sum, count)
        let mut rel_grad_accum: HashMap<usize, (Vec<f32>, usize)> = HashMap::new();

        // Ensure all entities and relations exist before the batch.
        for &(h, r, t) in triples {
            self.ensure_entity(h);
            self.ensure_entity(t);
            self.ensure_relation(r);
        }

        for &(h, r, t) in triples {
            let box_h = self.boxes.get(&h).cloned().unwrap();
            let box_t = self.boxes.get(&t).cloned().unwrap();

            // Apply learned relation translation to head.
            let translation = self.relation_translations.get(&r).cloned();
            let box_h_translated = if let Some(ref trans) = translation {
                let dense = box_h.to_box();
                let new_min: Vec<f32> = dense.min.iter().zip(trans).map(|(m, t)| m + t).collect();
                let new_max: Vec<f32> = dense.max.iter().zip(trans).map(|(m, t)| m + t).collect();
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
            } else {
                box_h.clone()
            };

            // Positive loss.
            let pos_loss = compute_pair_loss(&box_h_translated, &box_t, true, &step_config);
            total_loss += pos_loss;
            n_pairs += 1;

            // Positive gradients (w.r.t. untranslated head params -- translation is additive).
            let (grad_mu_h, grad_delta_h, grad_mu_t, grad_delta_t) =
                compute_analytical_gradients(&box_h_translated, &box_t, true, &step_config);

            // Accumulate head gradients.
            let entry = grad_accum
                .entry(h)
                .or_insert_with(|| (vec![0.0; self.dim], vec![0.0; self.dim], 0));
            for (acc, g) in entry.0.iter_mut().zip(&grad_mu_h) {
                *acc += g;
            }
            for (acc, g) in entry.1.iter_mut().zip(&grad_delta_h) {
                *acc += g;
            }
            entry.2 += 1;

            // Accumulate tail gradients.
            let entry = grad_accum
                .entry(t)
                .or_insert_with(|| (vec![0.0; self.dim], vec![0.0; self.dim], 0));
            for (acc, g) in entry.0.iter_mut().zip(&grad_mu_t) {
                *acc += g;
            }
            for (acc, g) in entry.1.iter_mut().zip(&grad_delta_t) {
                *acc += g;
            }
            entry.2 += 1;

            // Accumulate relation translation gradients.
            // d_loss/d_translation[i] = d_loss/d_min_h[i] + d_loss/d_max_h[i]
            // Since min = mu - exp(delta)/2 and max = mu + exp(delta)/2,
            // d_loss/d_min = d_loss/d_mu * d_mu/d_min + d_loss/d_delta * d_delta/d_min
            // But for translation: translated_min = min + t, so d_loss/d_t = d_loss/d_translated_min
            // The gradient w.r.t. translation is the same as w.r.t. the mu of the head (since translation shifts mu).
            let rel_entry = rel_grad_accum
                .entry(r)
                .or_insert_with(|| (vec![0.0; self.dim], 0));
            for (acc, g) in rel_entry.0.iter_mut().zip(&grad_mu_h) {
                *acc += g;
            }
            rel_entry.1 += 1;

            // Negative samples: random from full entity pool.
            // Collect all negatives first (needed for softmax weighting).
            let mut neg_data: Vec<(
                usize,
                usize,
                TrainableBox,
                TrainableBox,
                f32,
                (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
            )> = Vec::with_capacity(n_neg);

            for _ in 0..n_neg {
                let (neg_h, neg_t) = self.sample_negative(h, r, t, all_entities, &step_config, rng);
                self.ensure_entity(neg_h);
                self.ensure_entity(neg_t);
                let box_neg_h = self.boxes.get(&neg_h).cloned().unwrap();
                let box_neg_t = self.boxes.get(&neg_t).cloned().unwrap();

                // Apply relation translation to negative head.
                let box_neg_h_translated = if let Some(ref trans) = translation {
                    let dense = box_neg_h.to_box();
                    let new_min: Vec<f32> =
                        dense.min.iter().zip(trans).map(|(m, t)| m + t).collect();
                    let new_max: Vec<f32> =
                        dense.max.iter().zip(trans).map(|(m, t)| m + t).collect();
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
                    TrainableBox::new(mu, delta).unwrap_or_else(|_| box_neg_h.clone())
                } else {
                    box_neg_h.clone()
                };

                let neg_loss =
                    compute_pair_loss(&box_neg_h_translated, &box_neg_t, false, &step_config);
                total_loss += neg_loss;
                n_pairs += 1;

                let grads = compute_analytical_gradients(
                    &box_neg_h_translated,
                    &box_neg_t,
                    false,
                    &step_config,
                );

                // Score for adversarial weighting: positive-side loss (lower = model thinks it's true).
                let score = if step_config.self_adversarial {
                    compute_pair_loss(&box_neg_h_translated, &box_neg_t, true, &step_config)
                } else {
                    0.0 // unused
                };

                neg_data.push((neg_h, neg_t, box_neg_h_translated, box_neg_t, score, grads));
            }

            // Compute per-negative weights.
            let weights: Vec<f32> = if step_config.self_adversarial && !neg_data.is_empty() {
                // Softmax over -score/temp (lower positive loss = higher containment = harder negative).
                let alpha = step_config.adversarial_temperature;
                let logits: Vec<f32> = neg_data
                    .iter()
                    .map(|(_, _, _, _, s, _)| -s * alpha)
                    .collect();
                let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let n = neg_data.len() as f32;
                // Scale by N so the total gradient magnitude is comparable to uniform weighting.
                exps.iter().map(|e| e / sum * n).collect()
            } else {
                vec![1.0; neg_data.len()]
            };

            // Accumulate weighted negative gradients.
            for (i, (neg_h, neg_t, _, _, _, (gmh, gdh, gmt, gdt))) in
                neg_data.into_iter().enumerate()
            {
                let w = weights[i];

                let entry = grad_accum
                    .entry(neg_h)
                    .or_insert_with(|| (vec![0.0; self.dim], vec![0.0; self.dim], 0));
                for (acc, g) in entry.0.iter_mut().zip(&gmh) {
                    *acc += g * w;
                }
                for (acc, g) in entry.1.iter_mut().zip(&gdh) {
                    *acc += g * w;
                }
                entry.2 += 1;

                let entry = grad_accum
                    .entry(neg_t)
                    .or_insert_with(|| (vec![0.0; self.dim], vec![0.0; self.dim], 0));
                for (acc, g) in entry.0.iter_mut().zip(&gmt) {
                    *acc += g * w;
                }
                for (acc, g) in entry.1.iter_mut().zip(&gdt) {
                    *acc += g * w;
                }
                entry.2 += 1;
            }
        }

        // Apply accumulated gradients (averaged per entity).
        for (entity_id, (grad_mu_sum, grad_delta_sum, count)) in &grad_accum {
            let scale = 1.0 / (*count as f32);
            let avg_mu: Vec<f32> = grad_mu_sum.iter().map(|g| g * scale).collect();
            let avg_delta: Vec<f32> = grad_delta_sum.iter().map(|g| g * scale).collect();
            if let (Some(b), Some(s)) = (
                self.boxes.get_mut(entity_id),
                self.optimizer_states.get_mut(entity_id),
            ) {
                b.update_amsgrad(&avg_mu, &avg_delta, s);
            }
        }

        // Apply accumulated relation translation gradients.
        for (rel_id, (grad_sum, count)) in &rel_grad_accum {
            let scale = 1.0 / (*count as f32);
            if let (Some(trans), Some(state)) = (
                self.relation_translations.get_mut(rel_id),
                self.relation_optimizer_states.get_mut(rel_id),
            ) {
                // Simple SGD with learning rate from optimizer state.
                let lr = state.lr;
                for (t, g) in trans.iter_mut().zip(grad_sum) {
                    *t -= lr * g * scale;
                }
            }
        }

        Ok(total_loss / n_pairs.max(1) as f32)
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
    /// instead of separate margin-based losses. When `config.self_adversarial`
    /// is true, negative gradients are weighted by softmax of the model's current
    /// containment score scaled by `adversarial_temperature` (Sun et al., RotatE
    /// ICLR 2019).
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

            // Generate negative samples using the configured strategy and count.
            if entity_ids.len() <= 1 {
                continue; // cannot generate negatives with a single entity
            }
            let n_neg = step_config.negative_samples.max(1);

            // Generate all negative samples for this triple first.
            // Needed for softmax weighting when self_adversarial is enabled.
            let mut neg_pairs: Vec<(usize, usize)> = Vec::with_capacity(n_neg);
            for neg_idx in 0..n_neg {
                let idx = (h.wrapping_mul(31).wrapping_add(t).wrapping_add(7 + neg_idx))
                    % entity_ids.len();
                let pair = match &step_config.negative_strategy {
                    NegativeSamplingStrategy::CorruptTail => {
                        let candidate = entity_ids[idx];
                        let nt = if candidate == t {
                            entity_ids[(idx + 1) % entity_ids.len()]
                        } else {
                            candidate
                        };
                        (h, nt)
                    }
                    NegativeSamplingStrategy::CorruptHead => {
                        let candidate = entity_ids[idx];
                        let nh = if candidate == h {
                            entity_ids[(idx + 1) % entity_ids.len()]
                        } else {
                            candidate
                        };
                        (nh, t)
                    }
                    NegativeSamplingStrategy::CorruptBoth => {
                        let nh_idx = idx;
                        let nt_idx = (idx.wrapping_add(3)) % entity_ids.len();
                        let nh = {
                            let c = entity_ids[nh_idx];
                            if c == h {
                                entity_ids[(nh_idx + 1) % entity_ids.len()]
                            } else {
                                c
                            }
                        };
                        let nt = {
                            let c = entity_ids[nt_idx];
                            if c == t {
                                entity_ids[(nt_idx + 1) % entity_ids.len()]
                            } else {
                                c
                            }
                        };
                        (nh, nt)
                    }
                    NegativeSamplingStrategy::Uniform => {
                        if neg_idx % 2 == 0 {
                            let candidate = entity_ids[idx];
                            let nt = if candidate == t {
                                entity_ids[(idx + 1) % entity_ids.len()]
                            } else {
                                candidate
                            };
                            (h, nt)
                        } else {
                            let candidate = entity_ids[idx];
                            let nh = if candidate == h {
                                entity_ids[(idx + 1) % entity_ids.len()]
                            } else {
                                candidate
                            };
                            (nh, t)
                        }
                    }
                };
                neg_pairs.push(pair);
            }

            // Snapshot negative boxes.
            let neg_boxes: Vec<(TrainableBox, TrainableBox)> = neg_pairs
                .iter()
                .map(|&(nh, nt)| (self.snapshot_box(nh), self.snapshot_box(nt)))
                .collect();

            if step_config.use_infonce {
                // InfoNCE path: process each negative independently.
                for (i, &(neg_h, neg_t)) in neg_pairs.iter().enumerate() {
                    let (ref box_neg_h, ref box_neg) = neg_boxes[i];

                    let pos_score =
                        compute_pair_loss(&box_h_transformed, &box_t, true, &step_config);
                    let neg_score = compute_pair_loss(box_neg_h, box_neg, true, &step_config);
                    let tau = step_config.margin.max(1e-6);
                    let infonce_loss = crate::utils::softplus((pos_score - neg_score) / tau, 1.0);
                    total_loss += infonce_loss;

                    let sig = crate::utils::stable_sigmoid((pos_score - neg_score) / tau);
                    let dldpos = sig / tau;
                    let dldneg = -sig / tau;

                    let (grad_mu_h, grad_delta_h, grad_mu_t, grad_delta_t) =
                        compute_analytical_gradients(&box_h, &box_t, true, &step_config);
                    if let (Some(b), Some(s)) =
                        (self.boxes.get_mut(&h), self.optimizer_states.get_mut(&h))
                    {
                        let scaled_mu: Vec<f32> = grad_mu_h.iter().map(|g| g * dldpos).collect();
                        let scaled_delta: Vec<f32> =
                            grad_delta_h.iter().map(|g| g * dldpos).collect();
                        b.update_amsgrad(&scaled_mu, &scaled_delta, s);
                    }
                    if let (Some(b), Some(s)) =
                        (self.boxes.get_mut(&t), self.optimizer_states.get_mut(&t))
                    {
                        let scaled_mu: Vec<f32> = grad_mu_t.iter().map(|g| g * dldpos).collect();
                        let scaled_delta: Vec<f32> =
                            grad_delta_t.iter().map(|g| g * dldpos).collect();
                        b.update_amsgrad(&scaled_mu, &scaled_delta, s);
                    }

                    let bnh = self.snapshot_box(neg_h);
                    let (gmnh, gdnh, gmnt, gdnt) =
                        compute_analytical_gradients(&bnh, box_neg, true, &step_config);
                    if let (Some(b), Some(s)) = (
                        self.boxes.get_mut(&neg_h),
                        self.optimizer_states.get_mut(&neg_h),
                    ) {
                        let sm: Vec<f32> = gmnh.iter().map(|g| g * dldneg).collect();
                        let sd: Vec<f32> = gdnh.iter().map(|g| g * dldneg).collect();
                        b.update_amsgrad(&sm, &sd, s);
                    }
                    if let (Some(b), Some(s)) = (
                        self.boxes.get_mut(&neg_t),
                        self.optimizer_states.get_mut(&neg_t),
                    ) {
                        let sm: Vec<f32> = gmnt.iter().map(|g| g * dldneg).collect();
                        let sd: Vec<f32> = gdnt.iter().map(|g| g * dldneg).collect();
                        b.update_amsgrad(&sm, &sd, s);
                    }
                }
            } else {
                // Standard margin-based loss path.

                // Positive loss (computed once per positive triple).
                let pos_loss = compute_pair_loss(&box_h_transformed, &box_t, true, &step_config);
                total_loss += pos_loss;

                let (grad_mu_h, grad_delta_h, grad_mu_t, grad_delta_t) =
                    compute_analytical_gradients(&box_h, &box_t, true, &step_config);

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

                // Compute negative losses and gradients, collecting scores for adversarial weighting.
                let mut neg_grads: Vec<(
                    usize,
                    usize,
                    f32,
                    Vec<f32>,
                    Vec<f32>,
                    Vec<f32>,
                    Vec<f32>,
                )> = Vec::with_capacity(neg_pairs.len());

                for (i, &(neg_h, neg_t)) in neg_pairs.iter().enumerate() {
                    let (ref bnh, ref bnt) = neg_boxes[i];
                    let neg_loss = compute_pair_loss(bnh, bnt, false, &step_config);
                    total_loss += neg_loss;

                    let (gmnh, gdnh, gmnt, gdnt) =
                        compute_analytical_gradients(bnh, bnt, false, &step_config);

                    let score = if step_config.self_adversarial {
                        compute_pair_loss(bnh, bnt, true, &step_config)
                    } else {
                        0.0
                    };

                    neg_grads.push((neg_h, neg_t, score, gmnh, gdnh, gmnt, gdnt));
                }

                // Compute weights: softmax when self_adversarial, uniform otherwise.
                let weights: Vec<f32> = if step_config.self_adversarial && !neg_grads.is_empty() {
                    let alpha = step_config.adversarial_temperature;
                    let logits: Vec<f32> = neg_grads
                        .iter()
                        .map(|(_, _, s, _, _, _, _)| -s * alpha)
                        .collect();
                    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    let n = neg_grads.len() as f32;
                    // Scale by N so total gradient magnitude matches uniform weighting.
                    exps.iter().map(|e| e / sum * n).collect()
                } else {
                    vec![1.0; neg_grads.len()]
                };

                // Apply weighted negative gradients.
                for (i, (neg_h, neg_t, _, gmnh, gdnh, gmnt, gdnt)) in
                    neg_grads.into_iter().enumerate()
                {
                    let w = weights[i];
                    let scaled_gmnh: Vec<f32> = gmnh.iter().map(|g| g * w).collect();
                    let scaled_gdnh: Vec<f32> = gdnh.iter().map(|g| g * w).collect();
                    let scaled_gmnt: Vec<f32> = gmnt.iter().map(|g| g * w).collect();
                    let scaled_gdnt: Vec<f32> = gdnt.iter().map(|g| g * w).collect();

                    if let (Some(b), Some(s)) = (
                        self.boxes.get_mut(&neg_h),
                        self.optimizer_states.get_mut(&neg_h),
                    ) {
                        b.update_amsgrad(&scaled_gmnh, &scaled_gdnh, s);
                    }
                    if let (Some(b), Some(s)) = (
                        self.boxes.get_mut(&neg_t),
                        self.optimizer_states.get_mut(&neg_t),
                    ) {
                        b.update_amsgrad(&scaled_gmnt, &scaled_gdnt, s);
                    }
                }
            }
        }

        // Average over triples and negatives per triple.
        let n_pairs = triples.len() as f32 * step_config.negative_samples.max(1) as f32;
        Ok(total_loss / n_pairs)
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
    /// [`evaluate_link_prediction_interned`](super::evaluate_link_prediction_interned) (or the transform-aware variant).
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

        // Build transforms from learned translations + explicit transforms.
        let mut combined_transforms: HashMap<usize, RelationTransform> =
            self.relation_transforms.clone();
        for (&rel_id, trans) in &self.relation_translations {
            // Only add if not already an explicit transform and translation is non-zero.
            if !combined_transforms.contains_key(&rel_id) && trans.iter().any(|&v| v.abs() > 1e-8) {
                combined_transforms.insert(rel_id, RelationTransform::Translation(trans.clone()));
            }
        }

        let has_transforms = !combined_transforms.is_empty()
            && combined_transforms.values().any(|t| !t.is_identity());

        if has_transforms {
            let max_rel = combined_transforms.keys().copied().max().unwrap_or(0);
            let mut transforms = vec![RelationTransform::Identity; max_rel + 1];
            for (&rel_id, t) in &combined_transforms {
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
        self.config.validate()?;
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

        // Pre-ensure all entities so all_entity_ids() is complete.
        for &(h, _r, t) in train_triples {
            self.ensure_entity(h);
            self.ensure_entity(t);
        }

        // Compute relation cardinalities for Bernoulli negative sampling.
        if self.config.bernoulli_sampling {
            self.relation_cardinalities =
                super::negative_sampling::compute_relation_cardinalities(train_triples);
        }

        // Shuffled copy of triples for mini-batch training.
        #[cfg(feature = "rand")]
        let mut shuffled_triples: Vec<(usize, usize, usize)> = train_triples.to_vec();
        #[cfg(feature = "rand")]
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for epoch in 0..epochs {
            // Learning rate scheduling.
            let lr = crate::optimizer::get_learning_rate(epoch, epochs, base_lr, warmup);
            for state in self.optimizer_states.values_mut() {
                state.set_lr(lr);
            }
            for state in self.relation_optimizer_states.values_mut() {
                state.set_lr(lr);
            }

            // Gumbel beta annealing: linear interpolation from start to end.
            let progress = if epochs > 1 {
                epoch as f32 / (epochs - 1) as f32
            } else {
                1.0
            };
            self.current_beta = beta_start + (beta_end - beta_start) * progress;

            // Mini-batch training with shuffling and gradient accumulation.
            #[cfg(feature = "rand")]
            let loss = {
                shuffled_triples.shuffle(&mut rng);
                let all_entities = self.all_entity_ids();
                let batch_size = self.config.batch_size.max(1);
                let mut epoch_loss = 0.0f32;
                let mut n_batches = 0usize;
                for chunk in shuffled_triples.chunks(batch_size) {
                    let batch_loss = self.train_step_minibatch(chunk, &all_entities, &mut rng)?;
                    epoch_loss += batch_loss;
                    n_batches += 1;
                }
                epoch_loss / n_batches.max(1) as f32
            };
            #[cfg(not(feature = "rand"))]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::AMSGradState;
    use proptest::prelude::*;

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
        assert!(cfg.margin > 0.0);
        assert!(cfg.negative_weight > 0.0);
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

    /// Centered finite-difference gradient check for `compute_analytical_gradients`.
    ///
    /// For each parameter component (mu_h, delta_h, mu_t, delta_t), perturbs by
    /// +/- epsilon and compares (loss_plus - loss_minus) / (2 * epsilon) against
    /// the analytical gradient. Uses relative tolerance of 1e-2.
    ///
    /// Runs for both positive and negative pair losses.
    #[test]
    fn gradcheck_analytical_vs_finite_difference() {
        // Disable gradient norm clipping so analytical gradients are unmodified.
        // Use non-zero regularization to exercise that path too.
        let cfg = TrainingConfig {
            regularization: 0.001,
            max_grad_norm: f32::MAX,
            gumbel_beta: 10.0,
            margin: 0.2,
            negative_weight: 1.0,
            ..Default::default()
        };

        let dim = 4;

        // Fixed, deterministic parameters. Boxes partially overlap with
        // different widths per dimension so both contribute active bounds
        // to the intersection. Head is overall larger (higher delta) so
        // P(B|A) < P(A|B), placing us on one side of the min() and
        // avoiding the non-smooth kink where P(A|B) ~= P(B|A).
        let head = TrainableBox::new(vec![0.1, -0.2, 0.5, 0.3], vec![1.2, 0.7, 0.6, 0.7]).unwrap();
        let tail =
            TrainableBox::new(vec![1.16, -0.42, 0.41, 1.54], vec![0.5, 0.5, 0.5, 0.5]).unwrap();

        let eps = 1e-4_f32;
        let rel_tol = 2e-2;

        // For the negative case, use boxes where the tail is fully inside the
        // head, giving max_prob ~1.0 which is well above margin (0.2).
        // This avoids the hinge boundary where gradients are discontinuous.
        let head_neg =
            TrainableBox::new(vec![0.0, 0.0, 0.0, 0.0], vec![1.5, 1.5, 1.5, 1.5]).unwrap();
        let tail_neg =
            TrainableBox::new(vec![0.5, 0.3, 0.4, 0.6], vec![0.3, 0.4, 0.2, 0.3]).unwrap();

        let test_cases: Vec<(bool, &TrainableBox, &TrainableBox)> =
            vec![(true, &head, &tail), (false, &head_neg, &tail_neg)];

        let mut checked = 0usize; // count of non-trivial gradient comparisons

        for (is_positive, h_box, t_box) in &test_cases {
            let (g_mu_h, g_delta_h, g_mu_t, g_delta_t) =
                compute_analytical_gradients(h_box, t_box, *is_positive, &cfg);

            // Helper: perturb a single parameter, compute loss.
            let is_pos = *is_positive;
            let perturb_loss = |which: &str, idx: usize, sign: f32| -> f32 {
                let mut h = (*h_box).clone();
                let mut t = (*t_box).clone();
                match which {
                    "mu_h" => h.mu[idx] += sign * eps,
                    "delta_h" => h.delta[idx] += sign * eps,
                    "mu_t" => t.mu[idx] += sign * eps,
                    "delta_t" => t.delta[idx] += sign * eps,
                    _ => unreachable!(),
                }
                compute_pair_loss(&h, &t, is_pos, &cfg)
            };

            let cases: &[(&str, &[f32])] = &[
                ("mu_h", &g_mu_h),
                ("delta_h", &g_delta_h),
                ("mu_t", &g_mu_t),
                ("delta_t", &g_delta_t),
            ];

            for &(name, analytical) in cases {
                #[allow(clippy::needless_range_loop)]
                for i in 0..dim {
                    let loss_plus = perturb_loss(name, i, 1.0);
                    let loss_minus = perturb_loss(name, i, -1.0);
                    let numerical = (loss_plus - loss_minus) / (2.0 * eps);
                    let a = analytical[i];

                    // Skip comparison when both values are negligibly small.
                    let abs_tol = 1e-4;
                    if a.abs() < abs_tol && numerical.abs() < abs_tol {
                        continue;
                    }

                    checked += 1;

                    // Relative error: |a - n| / max(|a|, |n|)
                    let denom = a.abs().max(numerical.abs());
                    let rel_err = (a - numerical).abs() / denom;

                    assert!(
                        rel_err < rel_tol,
                        "gradcheck failed: is_positive={is_pos}, {name}[{i}]: \
                         analytical={a:.6}, numerical={numerical:.6}, rel_err={rel_err:.6}"
                    );
                }
            }
        }

        // Ensure the test actually verified a meaningful number of components.
        // With dim=4, 4 parameter groups, 2 cases => up to 32 components.
        // When one box contains the other, only the inner box's bounds are
        // active in the intersection, so ~half of gradients are near-zero.
        assert!(
            checked >= 8,
            "gradcheck only verified {checked} non-trivial components (expected >= 8)"
        );
    }

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

    // -----------------------------------------------------------------------
    // Self-adversarial negative sampling tests
    // -----------------------------------------------------------------------

    #[test]
    fn self_adversarial_config_default_is_off() {
        let cfg = TrainingConfig::default();
        assert!(!cfg.self_adversarial);
        assert!((cfg.adversarial_temperature - 1.0).abs() < 1e-6);
    }

    #[test]
    fn self_adversarial_serde_roundtrip() {
        let cfg = TrainingConfig {
            self_adversarial: true,
            adversarial_temperature: 0.5,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: TrainingConfig = serde_json::from_str(&json).unwrap();
        assert!(restored.self_adversarial);
        assert!((restored.adversarial_temperature - 0.5).abs() < 1e-6);
    }

    #[test]
    fn self_adversarial_serde_missing_field_defaults_to_false() {
        // Backward compat: old JSON without self_adversarial should deserialize fine.
        let json = r#"{"learning_rate":0.001,"epochs":100,"batch_size":512,"negative_samples":1,"negative_strategy":"CorruptTail","margin":1.0,"early_stopping_patience":10,"early_stopping_min_delta":0.001,"regularization":0.0001,"warmup_epochs":10,"negative_weight":1.0,"gumbel_beta":10.0,"gumbel_beta_final":50.0,"max_grad_norm":10.0,"adversarial_temperature":1.0,"use_infonce":false}"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        assert!(!cfg.self_adversarial);
    }

    #[test]
    fn train_step_self_adversarial_produces_different_loss() {
        // With self_adversarial on vs off, training should produce different
        // parameter updates (and therefore different final losses).
        let triples = vec![(0, 0, 1), (0, 0, 2), (1, 0, 3), (2, 0, 3)];

        let run = |self_adv: bool| -> f32 {
            let config = TrainingConfig {
                negative_samples: 3,
                self_adversarial: self_adv,
                adversarial_temperature: 2.0,
                ..Default::default()
            };
            let mut trainer = BoxEmbeddingTrainer::new(config, 4);
            let mut total_loss = 0.0;
            for _ in 0..5 {
                total_loss = trainer.train_step(&triples).unwrap();
            }
            total_loss
        };

        let loss_off = run(false);
        let loss_on = run(true);

        assert!(loss_off.is_finite(), "loss_off should be finite");
        assert!(loss_on.is_finite(), "loss_on should be finite");
        // The losses should differ because adversarial weighting changes gradient distribution.
        // (They could theoretically be equal if all negatives score identically, but with
        // 4 entities and 3 negatives per positive this is unlikely after 5 steps.)
        assert!(
            (loss_off - loss_on).abs() > 1e-8,
            "self_adversarial should change training dynamics: loss_off={loss_off}, loss_on={loss_on}"
        );
    }

    #[test]
    fn self_adversarial_softmax_weights_sum_to_n() {
        // Verify the softmax weighting logic directly: weights should sum to N
        // (so total gradient magnitude matches uniform weighting).
        let scores: Vec<f32> = vec![0.5, 1.0, 2.0, 0.1, 3.0];
        let alpha = 2.0_f32;
        let logits: Vec<f32> = scores.iter().map(|s| -s * alpha).collect();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let n = scores.len() as f32;
        let weights: Vec<f32> = exps.iter().map(|e| e / sum * n).collect();

        let weight_sum: f32 = weights.iter().sum();
        assert!(
            (weight_sum - n).abs() < 1e-5,
            "weights should sum to N={n}, got {weight_sum}"
        );

        // Lower scores (harder negatives) should get higher weights.
        // score[3]=0.1 is the lowest (hardest negative) -> highest weight.
        let max_weight_idx = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let min_score_idx = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            max_weight_idx, min_score_idx,
            "hardest negative (lowest positive-side loss) should get highest weight"
        );
    }
}
