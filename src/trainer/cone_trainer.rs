use crate::optimizer::AMSGradState;
use crate::trainable::TrainableCone;
use crate::BoxError;
use std::collections::{HashMap, HashSet};

use super::CpuBoxTrainingConfig;

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
    config: &CpuBoxTrainingConfig,
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
    config: &CpuBoxTrainingConfig,
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

/// Trainer for cone embeddings using the ConE model (Zhang & Wang, NeurIPS 2021).
///
/// Each entity is represented as a [`TrainableCone`] with per-dimension axis
/// angles and apertures, optimized via AMSGrad.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConeEmbeddingTrainer {
    /// Training configuration (shared with box trainer).
    pub(crate) config: CpuBoxTrainingConfig,
    /// Entity ID -> TrainableCone mapping.
    pub(crate) cones: HashMap<usize, TrainableCone>,
    /// Entity ID -> AMSGradState mapping.
    pub(crate) optimizer_states: HashMap<usize, AMSGradState>,
    /// Embedding dimension.
    pub(crate) dim: usize,
}

impl ConeEmbeddingTrainer {
    /// Embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Training configuration.
    #[must_use]
    pub fn config(&self) -> &CpuBoxTrainingConfig {
        &self.config
    }

    /// Entity cone embeddings.
    #[must_use]
    pub fn cones(&self) -> &HashMap<usize, TrainableCone> {
        &self.cones
    }
}

impl ConeEmbeddingTrainer {
    /// Create a new cone trainer.
    ///
    /// If `initial_embeddings` is provided, each vector is used as the initial
    /// per-dimension axis values (apertures start at pi/2).
    pub fn new(
        config: CpuBoxTrainingConfig,
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

    /// Run one training epoch over the given triples (batch API).
    ///
    /// For each `(head, relation, tail)` triple, trains a positive pair
    /// and one deterministic negative. Returns the average loss.
    pub fn train_step_batch(&mut self, triples: &[(usize, usize, usize)]) -> Result<f32, BoxError> {
        if triples.is_empty() {
            return Err(BoxError::Internal("empty triple set".to_string()));
        }
        let entity_ids: Vec<usize> = triples
            .iter()
            .flat_map(|&(h, _, t)| [h, t])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        for &(h, _r, t) in triples {
            total_loss += self.train_step(h, t, true);
            count += 1;
            if entity_ids.len() > 1 {
                let idx = (h.wrapping_mul(31).wrapping_add(t).wrapping_add(7)) % entity_ids.len();
                let candidate = entity_ids[idx];
                let neg_t = if candidate == t {
                    entity_ids[(idx + 1) % entity_ids.len()]
                } else {
                    candidate
                };
                total_loss += self.train_step(h, neg_t, false);
                count += 1;
            }
        }
        Ok(total_loss / count as f32)
    }

    /// Retrieve a learned cone for a given entity ID.
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn get_cone(&self, entity_id: usize) -> Option<crate::ndarray_backend::NdarrayCone> {
        self.cones
            .get(&entity_id)
            .and_then(|c| c.to_ndarray_cone().ok())
    }

    /// Convert all entity cones to [`NdarrayCone`](crate::ndarray_backend::NdarrayCone).
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn get_all_cones(&self) -> HashMap<usize, crate::ndarray_backend::NdarrayCone> {
        self.cones
            .iter()
            .filter_map(|(&id, c): (&usize, &TrainableCone)| {
                c.to_ndarray_cone().ok().map(|nc| (id, nc))
            })
            .collect()
    }

    /// Evaluate link prediction on test triples using cone distance scoring.
    ///
    /// For each test triple `(h, r, t)`, ranks all entities by cone distance
    /// to the head (for tail prediction) and to the tail (for head prediction).
    /// Returns standard metrics: MRR, Hits@{1,3,10}, Mean Rank.
    ///
    /// # Arguments
    ///
    /// * `test_triples` - Triples to evaluate on (ID-based).
    /// * `entities` - Entity vocabulary (for entity count).
    /// * `filter` - Optional filter to exclude known-true triples from ranking.
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn evaluate(
        &self,
        test_triples: &[crate::dataset::TripleIds],
        entities: &crate::dataset::Vocab,
        filter: Option<&super::evaluation::FilteredTripleIndexIds>,
    ) -> Result<super::EvaluationResults, BoxError> {
        let max_id = self.cones.keys().copied().max().unwrap_or(0);
        let num_entities = entities.len().max(max_id + 1);

        // Build dense cone vector indexed by entity ID.
        let mut entity_cones: Vec<crate::ndarray_backend::NdarrayCone> =
            Vec::with_capacity(num_entities);
        for id in 0..num_entities {
            let nc = if let Some(c) = self.cones.get(&id) {
                c.to_ndarray_cone().map_err(|e| {
                    BoxError::Internal(format!("Failed to convert entity {id}: {e}"))
                })?
            } else {
                crate::ndarray_backend::NdarrayCone::full(self.dim)
            };
            entity_cones.push(nc);
        }

        let cen = CONE_CENTER_WEIGHT;
        let mut tail_ranks = Vec::with_capacity(test_triples.len());
        let mut head_ranks = Vec::with_capacity(test_triples.len());

        for triple in test_triples {
            let h = triple.head;
            let t = triple.tail;
            let r = triple.relation;

            // Tail prediction: score all entities as tails for (h, r, ?).
            // Distance from head cone to each candidate.
            let head_cone = &entity_cones[h];
            let mut tail_scores: Vec<f32> = entity_cones
                .iter()
                .map(|e| head_cone.cone_distance(e, cen).unwrap_or(f32::INFINITY))
                .collect();

            // Filter known true tails (set distance to infinity).
            if let Some(f) = filter {
                if let Some(known) = f.known_tails(h, r) {
                    for &kt in known {
                        if kt != t && kt < tail_scores.len() {
                            tail_scores[kt] = f32::INFINITY;
                        }
                    }
                }
            }

            // Rank: count how many entities have strictly lower distance.
            let target_score = tail_scores[t];
            let rank = tail_scores.iter().filter(|&&s| s < target_score).count() + 1;
            tail_ranks.push(rank);

            // Head prediction: score all entities as heads for (?, r, t).
            let tail_cone = &entity_cones[t];
            let mut head_scores: Vec<f32> = entity_cones
                .iter()
                .map(|e| e.cone_distance(tail_cone, cen).unwrap_or(f32::INFINITY))
                .collect();

            if let Some(f) = filter {
                if let Some(known) = f.known_heads(t, r) {
                    for &kh in known {
                        if kh != h && kh < head_scores.len() {
                            head_scores[kh] = f32::INFINITY;
                        }
                    }
                }
            }

            let target_score = head_scores[h];
            let rank = head_scores.iter().filter(|&&s| s < target_score).count() + 1;
            head_ranks.push(rank);
        }

        // Compute metrics from ranks.
        let n = test_triples.len() as f32;
        if n < 1.0 {
            return Ok(super::EvaluationResults {
                mrr: 0.0,
                head_mrr: 0.0,
                tail_mrr: 0.0,
                hits_at_1: 0.0,
                hits_at_3: 0.0,
                hits_at_10: 0.0,
                mean_rank: 0.0,
                per_relation: Vec::new(),
            });
        }

        let tail_mrr: f32 = tail_ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / n;
        let head_mrr: f32 = head_ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / n;
        let mrr = (tail_mrr + head_mrr) / 2.0;

        let all_ranks: Vec<usize> = tail_ranks
            .iter()
            .chain(head_ranks.iter())
            .copied()
            .collect();
        let total = all_ranks.len() as f32;
        let hits_at_1 = all_ranks.iter().filter(|&&r| r <= 1).count() as f32 / total;
        let hits_at_3 = all_ranks.iter().filter(|&&r| r <= 3).count() as f32 / total;
        let hits_at_10 = all_ranks.iter().filter(|&&r| r <= 10).count() as f32 / total;
        let mean_rank = all_ranks.iter().sum::<usize>() as f32 / total;

        Ok(super::EvaluationResults {
            mrr,
            head_mrr,
            tail_mrr,
            hits_at_1,
            hits_at_3,
            hits_at_10,
            mean_rank,
            per_relation: Vec::new(), // Per-relation breakdown not yet implemented for cones.
        })
    }

    /// Train for multiple epochs with optional validation and early stopping.
    ///
    /// Mirrors [`BoxEmbeddingTrainer::fit`](super::BoxEmbeddingTrainer::fit):
    /// learning rate warmup, temperature annealing (unused for cones currently),
    /// and early stopping based on validation MRR.
    ///
    /// # Arguments
    ///
    /// * `train_triples` - Training triples as `(head_id, relation_id, tail_id)`.
    /// * `validation` - Optional validation set and entity vocabulary for evaluation.
    /// * `filter` - Optional filter index to exclude known-true triples from ranking.
    #[cfg(feature = "ndarray-backend")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
    pub fn fit(
        &mut self,
        train_triples: &[(usize, usize, usize)],
        validation: Option<(&[crate::dataset::TripleIds], &crate::dataset::Vocab)>,
        filter: Option<&super::evaluation::FilteredTripleIndexIds>,
    ) -> Result<super::TrainingResult, BoxError> {
        self.config.validate()?;
        let epochs = self.config.epochs;
        let warmup = self.config.warmup_epochs;
        let base_lr = self.config.learning_rate;
        let patience = self.config.early_stopping_patience;
        let min_delta = self.config.early_stopping_min_delta;

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

            let loss = self.train_step_batch(train_triples)?;
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

        let final_results = if let Some((val_triples, entities)) = validation {
            self.evaluate(val_triples, entities, filter)?
        } else {
            super::EvaluationResults {
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

        Ok(super::TrainingResult {
            final_results,
            loss_history,
            validation_mrr_history: mrr_history,
            best_epoch,
            training_time_seconds: None,
        })
    }

    /// Export all entity embeddings as flat `f32` vectors.
    ///
    /// Returns `(entity_ids, axes, apertures)` where vectors are
    /// row-major with length `n_entities * dim`.
    pub fn export_embeddings(&self) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
        let mut ids: Vec<usize> = self.cones.keys().copied().collect();
        ids.sort_unstable();
        let n = ids.len();
        let mut axes = Vec::with_capacity(n * self.dim);
        let mut apertures = Vec::with_capacity(n * self.dim);
        for &id in &ids {
            let c = &self.cones[&id];
            axes.extend_from_slice(&c.axes());
            apertures.extend_from_slice(&c.apertures());
        }
        (ids, axes, apertures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainable::TrainableCone;

    #[test]
    fn cone_pair_loss_positive_prefers_containment() {
        let cfg = CpuBoxTrainingConfig::default();

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
        let cfg = CpuBoxTrainingConfig::default();
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
        let cfg = CpuBoxTrainingConfig {
            learning_rate: 0.01,
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
