//! Ball embedding trainer with analytical gradients.
//!
//! Trains ball embeddings using margin-based ranking loss with negative sampling.
//! The loss encourages positive triples (head, relation, tail) to have high
//! containment probability and negative triples to have low containment.
#![allow(missing_docs)]

use crate::ball::{Ball, BallRelation};
use crate::dataset::Triple;
use crate::trainer::negative_sampling::{compute_relation_entity_pools, sample_excluding};
use crate::trainer::CpuBoxTrainingConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Trains ball embeddings using analytical gradients.
pub struct BallTrainer {
    rng: StdRng,
    step: usize,
    /// Persistent Adam momentum (first moment).
    adam_m: HashMap<String, f32>,
    /// Persistent Adam velocity (second moment).
    adam_v: HashMap<String, f32>,
}

impl BallTrainer {
    /// Create a new ball trainer with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            step: 0,
            adam_m: HashMap::new(),
            adam_v: HashMap::new(),
        }
    }

    /// Initialize entity and relation embeddings randomly.
    pub fn init_embeddings(
        &mut self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
    ) -> (Vec<Ball>, Vec<BallRelation>) {
        let entities: Vec<Ball> = (0..num_entities)
            .map(|_| {
                let center: Vec<f32> = (0..dim).map(|_| self.rng.random_range(-0.1..0.1)).collect();
                let log_radius = self.rng.random_range(-1.0..0.0);
                Ball::from_log_radius(center, log_radius).unwrap()
            })
            .collect();

        let relations: Vec<BallRelation> = (0..num_relations)
            .map(|_| {
                let translation: Vec<f32> = (0..dim)
                    .map(|_| self.rng.random_range(-0.01..0.01))
                    .collect();
                let log_scale: f32 = self.rng.random_range(-0.1..0.1);
                BallRelation::new(translation, log_scale.exp()).unwrap()
            })
            .collect();

        (entities, relations)
    }

    /// Compute the score for a triple (head, relation, tail).
    /// Lower is better.
    pub fn score_triple(head: &Ball, relation: &BallRelation, tail: &Ball, k: f32) -> f32 {
        let transformed = match relation.apply(head) {
            Ok(t) => t,
            Err(_) => return f32::INFINITY,
        };
        let prob = match crate::ball::containment_prob(&transformed, tail, k) {
            Ok(p) => p,
            Err(_) => return f32::INFINITY,
        };
        let prob = prob.clamp(1e-6, 1.0 - 1e-6);
        -prob.ln()
    }

    /// Compute ranking loss for a (positive, negative) pair.
    /// Compute per-triple margin-ranking loss and analytical gradients.
    ///
    /// # Derivation
    ///
    /// Given:
    /// ```text
    /// transformed.center   = head.center + relation.translation
    /// transformed.log_r    = head.log_r  + relation.log_scale
    /// transformed.radius   = exp(transformed.log_r)
    ///
    /// margin_pos = tail.radius − dist(transformed.center, tail.center) − transformed.radius
    /// margin_neg = neg_tail.radius − dist(transformed.center, neg_tail.center) − transformed.radius
    ///
    /// prob_pos = σ(k · margin_pos)
    /// prob_neg = σ(k · margin_neg)
    ///
    /// L = max(0,  margin − ln prob_pos + ln prob_neg)
    /// ```
    ///
    /// Chain rule (when L > 0):
    /// ```text
    /// ∂L/∂head.center_i       = d_pos · σ'_pos · (−diff_pos_i / dist_pos)
    ///                          + d_neg · σ'_neg · (−diff_neg_i / dist_neg)
    ///
    /// ∂L/∂relation.translation_i  = same as head.center_i  (enters identically)
    ///
    /// ∂L/∂tail.center_i       = d_pos · σ'_pos · (+diff_pos_i / dist_pos)
    ///
    /// ∂L/∂neg_tail.center_i   = d_neg · σ'_neg · (+diff_neg_i / dist_neg)
    ///
    /// ∂L/∂head.log_r          = (d_pos · σ'_pos + d_neg · σ'_neg) · (−transformed.radius)
    ///
    /// ∂L/∂relation.log_scale  = same as head.log_r  (enters identically via transformed.log_r)
    ///
    /// ∂L/∂tail.log_r          = d_pos · σ'_pos · (+tail.radius)
    ///
    /// ∂L/∂neg_tail.log_r      = d_neg · σ'_neg · (+neg_tail.radius)
    /// ```
    /// where d_pos = −1/prob_pos, d_neg = +1/prob_neg,
    ///       σ'(x) = k · σ(x) · (1 − σ(x)).
    fn compute_pair_gradients(
        head: &Ball,
        relation: &BallRelation,
        tail: &Ball,
        neg_tail: &Ball,
        margin: f32,
        k: f32,
    ) -> (f32, BallGradients) {
        let dim = head.dim();
        let mut grads = BallGradients::new(dim);

        // Both positive and negative share the same transformed head ball.
        let transformed = match relation.apply(head) {
            Ok(t) => t,
            Err(_) => return (0.0, grads),
        };
        let transformed_radius = transformed.radius();

        let pos_prob = match crate::ball::containment_prob(&transformed, tail, k) {
            Ok(p) => p.clamp(1e-10, 1.0 - 1e-10),
            Err(_) => return (0.0, grads),
        };
        let neg_prob = match crate::ball::containment_prob(&transformed, neg_tail, k) {
            Ok(p) => p.clamp(1e-10, 1.0 - 1e-10),
            Err(_) => return (0.0, grads),
        };

        let loss = (margin - pos_prob.ln() + neg_prob.ln()).max(0.0);
        if loss <= 1e-8 {
            return (0.0, grads);
        }

        let d_pos = -1.0 / pos_prob; // ∂L/∂pos_prob
        let d_neg = 1.0 / neg_prob; // ∂L/∂neg_prob
        let sp = k * pos_prob * (1.0 - pos_prob); // σ'(k·margin_pos)
        let sn = k * neg_prob * (1.0 - neg_prob); // σ'(k·margin_neg)

        let pos_dist = center_distance(&transformed, tail);
        let neg_dist = center_distance(&transformed, neg_tail);

        // ── Center gradients ────────────────────────────────────────────────
        if pos_dist > 1e-8 {
            for i in 0..dim {
                let d = (transformed.center()[i] - tail.center()[i]) / pos_dist;
                grads.head_center[i] += d_pos * sp * (-d);
                grads.tail_center[i] += d_pos * sp * d;
            }
        }
        if neg_dist > 1e-8 {
            for i in 0..dim {
                let d = (transformed.center()[i] - neg_tail.center()[i]) / neg_dist;
                grads.head_center[i] += d_neg * sn * (-d);
                grads.neg_tail_center[i] += d_neg * sn * d;
            }
        }
        // relation.translation enters identically to head.center
        grads
            .relation_translation
            .copy_from_slice(&grads.head_center);

        // ── Log-radius gradients ─────────────────────────────────────────────
        // ∂margin_pos/∂transformed.log_r = −transformed.radius  (chain through exp)
        // ∂margin_neg/∂transformed.log_r = −transformed.radius
        let dl_d_tr_lr = (d_pos * sp + d_neg * sn) * (-transformed_radius);
        grads.head_log_radius = dl_d_tr_lr; // head.log_r enters via transformed.log_r
        grads.relation_log_scale = dl_d_tr_lr; // rel.log_scale enters identically

        // ∂margin_pos/∂tail.log_r = +tail.radius
        grads.tail_log_radius = d_pos * sp * tail.radius();

        // ∂margin_neg/∂neg_tail.log_r = +neg_tail.radius
        grads.neg_tail_log_radius = d_neg * sn * neg_tail.radius();

        (loss, grads)
    }

    /// Train for one epoch.
    pub fn train_epoch(
        &mut self,
        entities: &mut [Ball],
        relations: &mut [BallRelation],
        triples: &[Triple],
        config: &CpuBoxTrainingConfig,
        entity_to_idx: &HashMap<String, usize>,
        relation_to_idx: &HashMap<String, usize>,
    ) -> f32 {
        let num_entities = entities.len();
        let k = config.sigmoid_k;
        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        let lr = config.learning_rate;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let indexed_triples: Vec<(usize, usize, usize)> = triples
            .iter()
            .filter_map(|triple| {
                let head_idx = *entity_to_idx.get(&triple.head)?;
                let rel_idx = *relation_to_idx.get(&triple.relation)?;
                let tail_idx = *entity_to_idx.get(&triple.tail)?;
                Some((head_idx, rel_idx, tail_idx))
            })
            .collect();
        let relation_pools = compute_relation_entity_pools(&indexed_triples);

        let mut indices: Vec<usize> = (0..indexed_triples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = self.rng.random_range(0..=i);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let (head_idx, rel_idx, tail_idx) = indexed_triples[idx];

            // Clone the triple's data so we can drop the immutable borrows before
            // the mutable update phase.  Balls are small (dim f32 + 1 radius).
            let head = entities[head_idx].clone();
            let relation = relations[rel_idx].clone();
            let tail = entities[tail_idx].clone();

            // Generate multiple negatives
            let n_neg = config.negative_samples.max(1);
            let adv_temp = config.adversarial_temperature;

            // Collect (neg_tail_idx, neg_score) pairs for self-adversarial weighting
            let neg_pairs: Vec<(usize, f32)> = (0..n_neg)
                .map(|_| {
                    let neg_tail_idx = if let Some(pool) = relation_pools.get(&rel_idx) {
                        sample_excluding(&pool.tails, tail_idx, |len| self.rng.random_range(0..len))
                            .unwrap_or_else(|| loop {
                                let neg = self.rng.random_range(0..num_entities);
                                if neg != tail_idx {
                                    break neg;
                                }
                            })
                    } else {
                        loop {
                            let neg = self.rng.random_range(0..num_entities);
                            if neg != tail_idx {
                                break neg;
                            }
                        }
                    };
                    let neg_score =
                        Self::score_triple(&head, &relation, &entities[neg_tail_idx], k);
                    (neg_tail_idx, neg_score)
                })
                .collect();

            // Self-adversarial weights: weight harder negatives more
            let neg_scores: Vec<f32> = neg_pairs.iter().map(|(_, s)| *s).collect();
            let weights =
                crate::trainer::trainer_utils::self_adversarial_weights(&neg_scores, adv_temp);

            // Accumulate weighted gradients across negatives.
            // neg_tail updates are applied per-negative (each negative is a different entity).
            let mut avg_grads = BallGradients::new(head.dim());
            let mut avg_loss = 0.0f32;
            // Collect per-negative (entity_idx, scaled_grads) for deferred application.
            let mut neg_updates: Vec<(usize, BallGradients)> = Vec::with_capacity(n_neg);

            for ((neg_tail_idx, _), weight) in neg_pairs.iter().zip(&weights) {
                let neg_tail = entities[*neg_tail_idx].clone();
                let (loss, grads) = Self::compute_pair_gradients(
                    &head,
                    &relation,
                    &tail,
                    &neg_tail,
                    config.margin,
                    k,
                );
                let w = *weight;
                avg_loss += w * loss;

                for i in 0..head.dim() {
                    avg_grads.head_center[i] += w * grads.head_center[i];
                    avg_grads.tail_center[i] += w * grads.tail_center[i];
                    avg_grads.relation_translation[i] += w * grads.relation_translation[i];
                }
                avg_grads.head_log_radius += w * grads.head_log_radius;
                avg_grads.tail_log_radius += w * grads.tail_log_radius;
                avg_grads.relation_log_scale += w * grads.relation_log_scale;

                // Keep per-negative entity gradient (scaled) for individual application below.
                let mut neg_g = BallGradients::new(head.dim());
                for i in 0..head.dim() {
                    neg_g.neg_tail_center[i] = w * grads.neg_tail_center[i];
                }
                neg_g.neg_tail_log_radius = w * grads.neg_tail_log_radius;
                neg_updates.push((*neg_tail_idx, neg_g));
            }

            total_loss += avg_loss;
            count += 1;

            let grads = avg_grads;

            self.step += 1;
            let t = self.step as f32;
            let bias1 = 1.0 - beta1.powf(t);
            let bias2 = 1.0 - beta2.powf(t);

            // Update head, tail, relation using entity-scoped Adam keys so that
            // an entity accumulates consistent momentum regardless of whether it
            // appears as head, tail, or negative in a given step.
            for i in 0..head.dim() {
                apply_adam(
                    &mut self.adam_m,
                    &mut self.adam_v,
                    &format!("e{head_idx}_c{i}"),
                    &mut entities[head_idx].center_mut()[i],
                    grads.head_center[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
                apply_adam(
                    &mut self.adam_m,
                    &mut self.adam_v,
                    &format!("e{tail_idx}_c{i}"),
                    &mut entities[tail_idx].center_mut()[i],
                    grads.tail_center[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
                apply_adam(
                    &mut self.adam_m,
                    &mut self.adam_v,
                    &format!("r{rel_idx}_t{i}"),
                    &mut relations[rel_idx].translation_mut()[i],
                    grads.relation_translation[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
            }

            update_log_param_adam(
                &mut self.adam_m,
                &mut self.adam_v,
                &format!("e{head_idx}_lr"),
                entities[head_idx].log_radius(),
                grads.head_log_radius,
                lr,
                beta1,
                beta2,
                eps,
                bias1,
                bias2,
                |e, v| e.set_log_radius(v),
                &mut entities[head_idx],
            );
            update_log_param_adam(
                &mut self.adam_m,
                &mut self.adam_v,
                &format!("e{tail_idx}_lr"),
                entities[tail_idx].log_radius(),
                grads.tail_log_radius,
                lr,
                beta1,
                beta2,
                eps,
                bias1,
                bias2,
                |e, v| e.set_log_radius(v),
                &mut entities[tail_idx],
            );
            update_log_param_adam(
                &mut self.adam_m,
                &mut self.adam_v,
                &format!("r{rel_idx}_ls"),
                relations[rel_idx].log_scale(),
                grads.relation_log_scale,
                lr,
                beta1,
                beta2,
                eps,
                bias1,
                bias2,
                |r, v| r.set_log_scale(v),
                &mut relations[rel_idx],
            );

            // Apply per-negative-entity gradients with the same entity-scoped keys.
            for (neg_idx, neg_g) in neg_updates {
                for i in 0..head.dim() {
                    apply_adam(
                        &mut self.adam_m,
                        &mut self.adam_v,
                        &format!("e{neg_idx}_c{i}"),
                        &mut entities[neg_idx].center_mut()[i],
                        neg_g.neg_tail_center[i],
                        lr,
                        beta1,
                        beta2,
                        eps,
                        bias1,
                        bias2,
                    );
                }
                update_log_param_adam(
                    &mut self.adam_m,
                    &mut self.adam_v,
                    &format!("e{neg_idx}_lr"),
                    entities[neg_idx].log_radius(),
                    neg_g.neg_tail_log_radius,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                    |e, v| e.set_log_radius(v),
                    &mut entities[neg_idx],
                );
            }
        }

        if count == 0 {
            0.0
        } else {
            total_loss / count as f32
        }
    }

    /// Evaluate the trained model on test triples using filtered link prediction.
    ///
    /// Uses the generic evaluation infrastructure with ball-specific scoring:
    /// - Tail prediction: score = containment_prob(transform(head, relation), candidate)
    /// - Head prediction: score = containment_prob(transform(candidate, relation), tail)
    ///
    /// Higher containment probability = better rank.
    pub fn evaluate(
        &self,
        entities: &[Ball],
        relations: &[BallRelation],
        test_triples: &[crate::dataset::TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let num_entities = entities.len();
        // k only affects probability magnitude, not ranking order (sigmoid is monotone).
        // Use the same default as CpuBoxTrainingConfig to keep values consistent.
        let k = 2.0f32;

        let score_tail = |head_idx: usize, rel_idx: usize, tail_idx: usize| -> f32 {
            let head = &entities[head_idx];
            let relation = &relations[rel_idx];
            let tail = &entities[tail_idx];
            let transformed = match relation.apply(head) {
                Ok(t) => t,
                Err(_) => return 0.0,
            };
            crate::ball::containment_prob(&transformed, tail, k).unwrap_or(0.0)
        };

        // score_head: called with (candidate_head_idx, rel_idx, known_tail_idx).
        // Rank candidate heads by how well transform(candidate) is contained in tail.
        let score_head = |head_idx: usize, rel_idx: usize, tail_idx: usize| -> f32 {
            let candidate_head = &entities[head_idx];
            let relation = &relations[rel_idx];
            let tail = &entities[tail_idx];
            let transformed = match relation.apply(candidate_head) {
                Ok(t) => t,
                Err(_) => return 0.0,
            };
            crate::ball::containment_prob(&transformed, tail, k).unwrap_or(0.0)
        };

        crate::trainer::evaluation::evaluate_link_prediction_generic(
            test_triples,
            num_entities,
            filter,
            score_head,
            score_tail,
        )
        .unwrap_or_else(|_| crate::trainer::EvaluationResults {
            mrr: 0.0,
            head_mrr: 0.0,
            tail_mrr: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
            mean_rank: f32::MAX,
            per_relation: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Adam helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn apply_adam(
    m: &mut HashMap<String, f32>,
    v: &mut HashMap<String, f32>,
    key: &str,
    param: &mut f32,
    grad: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias1: f32,
    bias2: f32,
) {
    let m_val = m.entry(key.to_string()).or_insert(0.0);
    let v_val = v.entry(key.to_string()).or_insert(0.0);
    *m_val = beta1 * *m_val + (1.0 - beta1) * grad;
    *v_val = beta2 * *v_val + (1.0 - beta2) * grad * grad;
    let m_hat = *m_val / bias1;
    let v_hat = (*v_val / bias2).max(0.0);
    *param -= lr * m_hat / (v_hat.sqrt() + eps);
}

#[allow(clippy::too_many_arguments)]
fn update_log_param_adam<T, F>(
    m: &mut HashMap<String, f32>,
    v: &mut HashMap<String, f32>,
    key: &str,
    current_val: f32,
    grad: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias1: f32,
    bias2: f32,
    setter: F,
    target: &mut T,
) where
    F: Fn(&mut T, f32),
{
    let m_val = m.entry(key.to_string()).or_insert(0.0);
    let v_val = v.entry(key.to_string()).or_insert(0.0);
    *m_val = beta1 * *m_val + (1.0 - beta1) * grad;
    *v_val = beta2 * *v_val + (1.0 - beta2) * grad * grad;
    let m_hat = *m_val / bias1;
    let v_hat = (*v_val / bias2).max(0.0);
    let new_val = current_val - lr * m_hat / (v_hat.sqrt() + eps);
    setter(target, new_val);
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

struct BallGradients {
    head_center: Vec<f32>,
    head_log_radius: f32,
    relation_translation: Vec<f32>,
    relation_log_scale: f32,
    tail_center: Vec<f32>,
    tail_log_radius: f32,
    neg_tail_center: Vec<f32>,
    neg_tail_log_radius: f32,
}

impl BallGradients {
    fn new(dim: usize) -> Self {
        Self {
            head_center: vec![0.0; dim],
            head_log_radius: 0.0,
            relation_translation: vec![0.0; dim],
            relation_log_scale: 0.0,
            tail_center: vec![0.0; dim],
            tail_log_radius: 0.0,
            neg_tail_center: vec![0.0; dim],
            neg_tail_log_radius: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn center_distance(a: &Ball, b: &Ball) -> f32 {
    a.center()
        .iter()
        .zip(b.center().iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball::Ball;

    #[test]
    fn trainer_init() {
        let mut trainer = BallTrainer::new(42);
        let (entities, relations) = trainer.init_embeddings(10, 3, 4);
        assert_eq!(entities.len(), 10);
        assert_eq!(relations.len(), 3);
        assert_eq!(entities[0].dim(), 4);
    }

    #[test]
    fn score_triple_identical_is_low() {
        let head = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let relation = BallRelation::identity(2);
        let tail = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let score = BallTrainer::score_triple(&head, &relation, &tail, 10.0);
        assert!(score < 1.0, "identical score = {score}, expected low");
    }

    #[test]
    fn score_triple_disjoint_is_high() {
        let head = Ball::new(vec![0.0, 0.0], 0.5).unwrap();
        let relation = BallRelation::identity(2);
        let tail = Ball::new(vec![10.0, 0.0], 0.5).unwrap();
        let score = BallTrainer::score_triple(&head, &relation, &tail, 10.0);
        assert!(score > 5.0, "disjoint score = {score}, expected high");
    }

    #[test]
    fn gradients_are_finite() {
        let head = Ball::new(vec![0.0, 0.0], 1.0).unwrap();
        let relation = BallRelation::new(vec![0.1, 0.1], 1.0).unwrap();
        let tail = Ball::new(vec![0.0, 0.0], 2.0).unwrap();
        let neg_tail = Ball::new(vec![5.0, 0.0], 0.5).unwrap();

        let (loss, grads) =
            BallTrainer::compute_pair_gradients(&head, &relation, &tail, &neg_tail, 1.0, 10.0);

        assert!(loss.is_finite(), "loss not finite: {loss}");
        for (i, &g) in grads.head_center.iter().enumerate() {
            assert!(g.is_finite(), "head_center[{i}] gradient not finite: {g}");
        }
        assert!(grads.head_log_radius.is_finite());
        assert!(grads.tail_log_radius.is_finite());
        assert!(grads.relation_log_scale.is_finite());
    }

    #[test]
    fn train_epoch_runs() {
        let mut trainer = BallTrainer::new(42);
        let (mut entities, mut relations) = trainer.init_embeddings(20, 3, 4);

        let triples = vec![Triple {
            head: "e0".to_string(),
            relation: "r0".to_string(),
            tail: "e1".to_string(),
        }];
        let entity_map: HashMap<String, usize> = [("e0".to_string(), 0), ("e1".to_string(), 1)]
            .into_iter()
            .collect();
        let relation_map: HashMap<String, usize> = [("r0".to_string(), 0)].into_iter().collect();

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.01,
            margin: 1.0,
            ..Default::default()
        };

        let loss = trainer.train_epoch(
            &mut entities,
            &mut relations,
            &triples,
            &config,
            &entity_map,
            &relation_map,
        );

        assert!(loss.is_finite(), "epoch loss not finite: {loss}");
    }

    #[test]
    fn train_and_evaluate_synthetic() {
        use crate::dataset::{TripleIds, Vocab};

        let mut vocab = Vocab::default();
        let e0 = vocab.intern("e0".to_string());
        let e1 = vocab.intern("e1".to_string());
        let e2 = vocab.intern("e2".to_string());
        let e3 = vocab.intern("e3".to_string());
        let r0 = 0usize;
        let r1 = 1usize;

        let triples = vec![
            Triple {
                head: "e0".to_string(),
                relation: "r0".to_string(),
                tail: "e1".to_string(),
            },
            Triple {
                head: "e2".to_string(),
                relation: "r0".to_string(),
                tail: "e3".to_string(),
            },
            Triple {
                head: "e0".to_string(),
                relation: "r1".to_string(),
                tail: "e2".to_string(),
            },
        ];

        let test_triples = vec![
            TripleIds {
                head: e0,
                relation: r0,
                tail: e1,
            },
            TripleIds {
                head: e2,
                relation: r0,
                tail: e3,
            },
            TripleIds {
                head: e0,
                relation: r1,
                tail: e2,
            },
        ];

        let entity_map: HashMap<String, usize> = [
            ("e0".to_string(), 0),
            ("e1".to_string(), 1),
            ("e2".to_string(), 2),
            ("e3".to_string(), 3),
        ]
        .into_iter()
        .collect();
        let relation_map: HashMap<String, usize> = [("r0".to_string(), 0), ("r1".to_string(), 1)]
            .into_iter()
            .collect();

        let mut trainer = BallTrainer::new(42);
        let (mut entities, mut relations) = trainer.init_embeddings(4, 2, 4);

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            epochs: 100,
            ..Default::default()
        };

        let mut last_loss = f32::MAX;
        for epoch in 0..50 {
            let loss = trainer.train_epoch(
                &mut entities,
                &mut relations,
                &triples,
                &config,
                &entity_map,
                &relation_map,
            );
            if epoch % 10 == 0 {
                eprintln!("Epoch {epoch}: loss={loss:.4}");
            }
            last_loss = loss;
        }
        eprintln!("Final loss: {last_loss:.4}");

        let results = trainer.evaluate(&entities, &relations, &test_triples, None);

        assert!(
            results.mrr > 0.3,
            "MRR after training = {}, expected > 0.3 (random ≈ 0.25)",
            results.mrr
        );
        assert!(
            results.mean_rank < 3.0,
            "Mean rank = {}, expected < 3.0",
            results.mean_rank
        );
    }

    #[test]
    fn train_improves_eval_metrics() {
        use crate::dataset::{TripleIds, Vocab};

        let mut vocab = Vocab::default();
        let e0 = vocab.intern("e0".to_string());
        let e1 = vocab.intern("e1".to_string());
        let e2 = vocab.intern("e2".to_string());

        let triples = vec![
            Triple {
                head: "e0".to_string(),
                relation: "r0".to_string(),
                tail: "e1".to_string(),
            },
            Triple {
                head: "e0".to_string(),
                relation: "r0".to_string(),
                tail: "e2".to_string(),
            },
        ];

        let test_triples = vec![
            TripleIds {
                head: e0,
                relation: 0,
                tail: e1,
            },
            TripleIds {
                head: e0,
                relation: 0,
                tail: e2,
            },
        ];

        let entity_map: HashMap<String, usize> = [
            ("e0".to_string(), 0),
            ("e1".to_string(), 1),
            ("e2".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let relation_map: HashMap<String, usize> = [("r0".to_string(), 0)].into_iter().collect();

        let mut trainer = BallTrainer::new(42);
        let (mut entities, mut relations) = trainer.init_embeddings(3, 1, 4);

        let results_before = trainer.evaluate(&entities, &relations, &test_triples, None);

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            ..Default::default()
        };

        for _epoch in 0..30 {
            let loss = trainer.train_epoch(
                &mut entities,
                &mut relations,
                &triples,
                &config,
                &entity_map,
                &relation_map,
            );
            assert!(
                loss.is_finite(),
                "train_epoch returned non-finite loss: {loss}"
            );
        }

        let results_after = trainer.evaluate(&entities, &relations, &test_triples, None);

        assert!(
            results_after.mean_rank <= results_before.mean_rank + 0.5,
            "Mean rank worsened: before={}, after={}",
            results_before.mean_rank,
            results_after.mean_rank
        );
    }

    /// Gradient check: compare analytical gradients against finite differences.
    /// This catches sign errors, missing chain rule terms, and scaling bugs.
    #[test]
    fn gradient_check_against_finite_differences() {
        let head = Ball::new(vec![0.3, -0.2, 0.5], 0.8).unwrap();
        let relation = BallRelation::new(vec![0.1, -0.1, 0.2], 1.2).unwrap();
        let tail = Ball::new(vec![0.5, 0.1, 0.3], 1.5).unwrap();
        let neg_tail = Ball::new(vec![-0.5, 0.8, -0.3], 0.6).unwrap();
        let margin = 1.0f32;
        let k = 10.0f32;
        let eps = 1e-4f32;

        let (base_loss, analytical) =
            BallTrainer::compute_pair_gradients(&head, &relation, &tail, &neg_tail, margin, k);

        if base_loss <= 1e-8 {
            return;
        }

        let compute_loss = |h: &Ball, r: &BallRelation, t: &Ball, nt: &Ball| -> f32 {
            let pos_t = r.apply(h).unwrap();
            let pos_p = crate::ball::containment_prob(&pos_t, t, k)
                .unwrap()
                .max(1e-10);
            let neg_t = r.apply(h).unwrap();
            let neg_p = crate::ball::containment_prob(&neg_t, nt, k)
                .unwrap()
                .max(1e-10);
            (margin - pos_p.ln() + neg_p.ln()).max(0.0)
        };

        let dim = head.dim();
        let rel_tol = 0.02; // 2% — well within finite-difference noise for eps=1e-4
        let mut checked = 0;

        // Head center
        for i in 0..dim {
            let mut c = head.center().to_vec();
            c[i] += eps;
            let h_plus = Ball::new(c.clone(), head.radius()).unwrap();
            c[i] -= 2.0 * eps;
            let h_minus = Ball::new(c, head.radius()).unwrap();
            let numerical = (compute_loss(&h_plus, &relation, &tail, &neg_tail)
                - compute_loss(&h_minus, &relation, &tail, &neg_tail))
                / (2.0 * eps);
            let a = analytical.head_center[i];
            if a.abs() > 1e-4 || numerical.abs() > 1e-4 {
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs());
                assert!(
                    rel_err < rel_tol,
                    "head_center[{i}]: a={a:.6}, n={numerical:.6}, err={rel_err:.4}"
                );
                checked += 1;
            }
        }

        // Tail center
        for i in 0..dim {
            let mut c = tail.center().to_vec();
            c[i] += eps;
            let t_plus = Ball::new(c.clone(), tail.radius()).unwrap();
            c[i] -= 2.0 * eps;
            let t_minus = Ball::new(c, tail.radius()).unwrap();
            let numerical = (compute_loss(&head, &relation, &t_plus, &neg_tail)
                - compute_loss(&head, &relation, &t_minus, &neg_tail))
                / (2.0 * eps);
            let a = analytical.tail_center[i];
            if a.abs() > 1e-4 || numerical.abs() > 1e-4 {
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs());
                assert!(
                    rel_err < rel_tol,
                    "tail_center[{i}]: a={a:.6}, n={numerical:.6}, err={rel_err:.4}"
                );
                checked += 1;
            }
        }

        // Head log_radius
        {
            let h_plus =
                Ball::from_log_radius(head.center().to_vec(), head.log_radius() + eps).unwrap();
            let h_minus =
                Ball::from_log_radius(head.center().to_vec(), head.log_radius() - eps).unwrap();
            let numerical = (compute_loss(&h_plus, &relation, &tail, &neg_tail)
                - compute_loss(&h_minus, &relation, &tail, &neg_tail))
                / (2.0 * eps);
            let a = analytical.head_log_radius;
            if a.abs() > 1e-4 || numerical.abs() > 1e-4 {
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs());
                assert!(
                    rel_err < rel_tol,
                    "head_log_radius: a={a:.6}, n={numerical:.6}, err={rel_err:.4}"
                );
                checked += 1;
            }
        }

        // Tail log_radius
        {
            let t_plus =
                Ball::from_log_radius(tail.center().to_vec(), tail.log_radius() + eps).unwrap();
            let t_minus =
                Ball::from_log_radius(tail.center().to_vec(), tail.log_radius() - eps).unwrap();
            let numerical = (compute_loss(&head, &relation, &t_plus, &neg_tail)
                - compute_loss(&head, &relation, &t_minus, &neg_tail))
                / (2.0 * eps);
            let a = analytical.tail_log_radius;
            if a.abs() > 1e-4 || numerical.abs() > 1e-4 {
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs());
                assert!(
                    rel_err < rel_tol,
                    "tail_log_radius: a={a:.6}, n={numerical:.6}, err={rel_err:.4}"
                );
                checked += 1;
            }
        }

        // Relation translation
        for i in 0..dim {
            let mut t = relation.translation().to_vec();
            t[i] += eps;
            let r_plus = BallRelation::new(t.clone(), relation.scale()).unwrap();
            t[i] -= 2.0 * eps;
            let r_minus = BallRelation::new(t, relation.scale()).unwrap();
            let numerical = (compute_loss(&head, &r_plus, &tail, &neg_tail)
                - compute_loss(&head, &r_minus, &tail, &neg_tail))
                / (2.0 * eps);
            let a = analytical.relation_translation[i];
            if a.abs() > 1e-4 || numerical.abs() > 1e-4 {
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs());
                assert!(
                    rel_err < rel_tol,
                    "rel_trans[{i}]: a={a:.6}, n={numerical:.6}, err={rel_err:.4}"
                );
                checked += 1;
            }
        }

        // Relation log_scale
        {
            let r_plus = BallRelation::new(
                relation.translation().to_vec(),
                (relation.log_scale() + eps).exp(),
            )
            .unwrap();
            let r_minus = BallRelation::new(
                relation.translation().to_vec(),
                (relation.log_scale() - eps).exp(),
            )
            .unwrap();
            let numerical = (compute_loss(&head, &r_plus, &tail, &neg_tail)
                - compute_loss(&head, &r_minus, &tail, &neg_tail))
                / (2.0 * eps);
            let a = analytical.relation_log_scale;
            if a.abs() > 1e-4 || numerical.abs() > 1e-4 {
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs());
                assert!(
                    rel_err < rel_tol,
                    "rel_log_scale: a={a:.6}, n={numerical:.6}, err={rel_err:.4}"
                );
                checked += 1;
            }
        }

        // Neg-tail center
        for i in 0..dim {
            let mut c = neg_tail.center().to_vec();
            c[i] += eps;
            let nt_plus = Ball::new(c.clone(), neg_tail.radius()).unwrap();
            c[i] -= 2.0 * eps;
            let nt_minus = Ball::new(c, neg_tail.radius()).unwrap();
            let numerical = (compute_loss(&head, &relation, &tail, &nt_plus)
                - compute_loss(&head, &relation, &tail, &nt_minus))
                / (2.0 * eps);
            let a = analytical.neg_tail_center[i];
            if a.abs() > 1e-4 || numerical.abs() > 1e-4 {
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs());
                assert!(
                    rel_err < rel_tol,
                    "neg_tail_center[{i}]: a={a:.6}, n={numerical:.6}, err={rel_err:.4}"
                );
                checked += 1;
            }
        }

        // Neg-tail log_radius
        {
            let nt_plus =
                Ball::from_log_radius(neg_tail.center().to_vec(), neg_tail.log_radius() + eps)
                    .unwrap();
            let nt_minus =
                Ball::from_log_radius(neg_tail.center().to_vec(), neg_tail.log_radius() - eps)
                    .unwrap();
            let numerical = (compute_loss(&head, &relation, &tail, &nt_plus)
                - compute_loss(&head, &relation, &tail, &nt_minus))
                / (2.0 * eps);
            let a = analytical.neg_tail_log_radius;
            if a.abs() > 1e-4 || numerical.abs() > 1e-4 {
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs());
                assert!(
                    rel_err < rel_tol,
                    "neg_tail_log_radius: a={a:.6}, n={numerical:.6}, err={rel_err:.4}"
                );
                checked += 1;
            }
        }

        assert!(
            checked >= 5,
            "gradient check only verified {checked} components (expected >= 5)"
        );
    }

    /// Verify that training loss decreases over epochs on a small hierarchy.
    /// This catches gradient sign errors that might not be visible in a single
    /// gradient check but would prevent convergence.
    #[test]
    fn loss_decreases_over_training() {
        let mut trainer = BallTrainer::new(42);
        let (mut entities, mut relations) = trainer.init_embeddings(6, 2, 4);

        let triples = vec![
            Triple {
                head: "a".into(),
                relation: "r".into(),
                tail: "b".into(),
            },
            Triple {
                head: "a".into(),
                relation: "r".into(),
                tail: "c".into(),
            },
            Triple {
                head: "d".into(),
                relation: "s".into(),
                tail: "e".into(),
            },
            Triple {
                head: "d".into(),
                relation: "s".into(),
                tail: "f".into(),
            },
        ];
        let entity_map: HashMap<String, usize> = ["a", "b", "c", "d", "e", "f"]
            .iter()
            .enumerate()
            .map(|(i, &s)| (s.to_string(), i))
            .collect();
        let relation_map: HashMap<String, usize> = [("r".to_string(), 0), ("s".to_string(), 1)]
            .into_iter()
            .collect();

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.02,
            margin: 1.0,
            negative_samples: 3,
            ..Default::default()
        };

        let loss_epoch1 = trainer.train_epoch(
            &mut entities,
            &mut relations,
            &triples,
            &config,
            &entity_map,
            &relation_map,
        );

        // Train 20 more epochs
        let mut loss_epoch20 = loss_epoch1;
        for _ in 1..20 {
            loss_epoch20 = trainer.train_epoch(
                &mut entities,
                &mut relations,
                &triples,
                &config,
                &entity_map,
                &relation_map,
            );
        }

        assert!(
            loss_epoch20 < loss_epoch1,
            "Loss should decrease: epoch 1 = {loss_epoch1:.4}, epoch 20 = {loss_epoch20:.4}"
        );
    }

    /// Verify that multi-negative sampling with n_neg > 1 actually affects
    /// multiple entities per step (not just the first negative).
    #[test]
    fn multi_neg_updates_negative_entities() {
        let mut trainer = BallTrainer::new(42);
        let (mut entities, mut relations) = trainer.init_embeddings(10, 1, 4);

        let triples = vec![Triple {
            head: "e0".into(),
            relation: "r0".into(),
            tail: "e1".into(),
        }];
        let entity_map: HashMap<String, usize> = (0..10).map(|i| (format!("e{i}"), i)).collect();
        let relation_map: HashMap<String, usize> = [("r0".to_string(), 0)].into_iter().collect();

        // Save initial centers for entities 2..10 (potential negatives)
        let initial_centers: Vec<Vec<f32>> = entities[2..10]
            .iter()
            .map(|e| e.center().to_vec())
            .collect();

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            negative_samples: 5,
            ..Default::default()
        };

        // Run one epoch: should update some negative entities
        trainer.train_epoch(
            &mut entities,
            &mut relations,
            &triples,
            &config,
            &entity_map,
            &relation_map,
        );

        // Count how many non-head/tail entities changed
        let mut changed = 0;
        for (i, initial) in initial_centers.iter().enumerate() {
            let current = entities[i + 2].center();
            if initial
                .iter()
                .zip(current)
                .any(|(a, b)| (a - b).abs() > 1e-10)
            {
                changed += 1;
            }
        }

        assert!(
            changed >= 2,
            "Expected at least 2 negative entities to be updated, got {changed}"
        );
    }

    /// Verify that the sigmoid_k config parameter affects scoring behavior.
    #[test]
    fn sigmoid_k_affects_containment_sharpness() {
        let head = Ball::new(vec![0.0, 0.0], 0.5).unwrap();
        let relation = BallRelation::new(vec![0.0, 0.0], 1.0).unwrap();
        // Tail barely contains transformed head
        let tail = Ball::new(vec![0.0, 0.0], 0.6).unwrap();

        let score_low_k = BallTrainer::score_triple(&head, &relation, &tail, 1.0);
        let score_high_k = BallTrainer::score_triple(&head, &relation, &tail, 20.0);

        // Higher k -> sharper sigmoid -> score should be more extreme (lower for contained)
        assert!(
            score_high_k < score_low_k,
            "Higher k should give lower score for contained pair: k=1 score={score_low_k:.4}, k=20 score={score_high_k:.4}"
        );
    }
}
