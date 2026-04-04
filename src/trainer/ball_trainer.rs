//! Ball embedding trainer with analytical gradients.
//!
//! Trains ball embeddings using margin-based ranking loss with negative sampling.
//! The loss encourages positive triples (head, relation, tail) to have high
//! containment probability and negative triples to have low containment.
#![allow(missing_docs)]

use crate::ball::{Ball, BallRelation};
use crate::dataset::Triple;
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
    /// Returns (loss, gradients).
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

        let pos_transformed = match relation.apply(head) {
            Ok(t) => t,
            Err(_) => return (0.0, grads),
        };
        let pos_prob = match crate::ball::containment_prob(&pos_transformed, tail, k) {
            Ok(p) => p.max(1e-10),
            Err(_) => return (0.0, grads),
        };

        let neg_transformed = match relation.apply(head) {
            Ok(t) => t,
            Err(_) => return (0.0, grads),
        };
        let neg_prob = match crate::ball::containment_prob(&neg_transformed, neg_tail, k) {
            Ok(p) => p.max(1e-10),
            Err(_) => return (0.0, grads),
        };

        let pos_score = -pos_prob.ln();
        let neg_score = -neg_prob.ln();
        // Loss = margin + pos_score - neg_score = margin - ln(pos_prob) + ln(neg_prob)
        // Minimized when pos_prob is large and neg_prob is small.
        let loss = (margin + pos_score - neg_score).max(0.0);

        if loss <= 1e-8 {
            return (0.0, grads);
        }

        let pos_center_dist = center_distance(&pos_transformed, tail);
        let neg_center_dist = center_distance(&neg_transformed, neg_tail);

        let pos_sigmoid_deriv = k * pos_prob * (1.0 - pos_prob);
        let neg_sigmoid_deriv = k * neg_prob * (1.0 - neg_prob);

        let d_pos = -1.0 / pos_prob;
        let d_neg = 1.0 / neg_prob;

        if pos_center_dist > 1e-8 {
            for i in 0..dim {
                let diff = pos_transformed.center()[i] - tail.center()[i];
                let d_dist = diff / pos_center_dist;
                grads.head_center[i] += d_pos * pos_sigmoid_deriv * (-d_dist);
                grads.tail_center[i] += d_pos * pos_sigmoid_deriv * d_dist;
            }
        }
        if neg_center_dist > 1e-8 {
            for i in 0..dim {
                let diff = neg_transformed.center()[i] - neg_tail.center()[i];
                let d_dist = diff / neg_center_dist;
                grads.head_center[i] += d_neg * neg_sigmoid_deriv * (-d_dist);
                grads.neg_tail_center[i] += d_neg * neg_sigmoid_deriv * d_dist;
            }
        }

        grads.head_log_radius +=
            d_pos * pos_sigmoid_deriv * (-1.0) + d_neg * neg_sigmoid_deriv * (-1.0);
        grads.tail_log_radius += d_pos * pos_sigmoid_deriv;
        grads.neg_tail_log_radius += d_neg * neg_sigmoid_deriv;

        for i in 0..dim {
            grads.relation_translation[i] = grads.head_center[i];
        }
        grads.relation_log_scale = grads.head_log_radius * head.radius();

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
        let k = 10.0;
        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        let lr = config.learning_rate;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let mut indices: Vec<usize> = (0..triples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = self.rng.random_range(0..=i);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let triple = &triples[idx];
            let head_idx = match entity_to_idx.get(&triple.head) {
                Some(&i) => i,
                None => continue,
            };
            let rel_idx = match relation_to_idx.get(&triple.relation) {
                Some(&i) => i,
                None => continue,
            };
            let tail_idx = match entity_to_idx.get(&triple.tail) {
                Some(&i) => i,
                None => continue,
            };

            let head = &entities[head_idx];
            let relation = &relations[rel_idx];
            let tail = &entities[tail_idx];

            // Generate multiple negatives
            let n_neg = config.negative_samples.max(1);
            let adv_temp = config.adversarial_temperature;

            // Collect (neg_tail_idx, neg_score) pairs for self-adversarial weighting
            let neg_pairs: Vec<(usize, f32)> = (0..n_neg)
                .map(|_| {
                    let neg_tail_idx = loop {
                        let neg = self.rng.random_range(0..num_entities);
                        if neg != tail_idx {
                            break neg;
                        }
                    };
                    let neg_score = Self::score_triple(head, relation, &entities[neg_tail_idx], k);
                    (neg_tail_idx, neg_score)
                })
                .collect();

            // Self-adversarial weights: weight harder negatives more
            let neg_scores: Vec<f32> = neg_pairs.iter().map(|(_, s)| *s).collect();
            let weights =
                crate::trainer::trainer_utils::self_adversarial_weights(&neg_scores, adv_temp);

            // Accumulate weighted gradients
            let mut avg_grads = BallGradients::new(head.dim());
            let mut avg_loss = 0.0f32;

            for (((neg_tail_idx, _), weight), _neg_score) in
                neg_pairs.iter().zip(&weights).zip(&neg_scores)
            {
                let neg_tail = &entities[*neg_tail_idx];
                let (loss, grads) =
                    Self::compute_pair_gradients(head, relation, tail, neg_tail, config.margin, k);
                let w = weight;
                avg_loss += w * loss;

                for i in 0..head.dim() {
                    avg_grads.head_center[i] += w * grads.head_center[i];
                    avg_grads.tail_center[i] += w * grads.tail_center[i];
                    avg_grads.neg_tail_center[i] += w * grads.neg_tail_center[i];
                    avg_grads.relation_translation[i] += w * grads.relation_translation[i];
                }
                avg_grads.head_log_radius += w * grads.head_log_radius;
                avg_grads.tail_log_radius += w * grads.tail_log_radius;
                avg_grads.neg_tail_log_radius += w * grads.neg_tail_log_radius;
                avg_grads.relation_log_scale += w * grads.relation_log_scale;
            }

            total_loss += avg_loss;
            count += 1;

            let grads = avg_grads;

            self.step += 1;
            let t = self.step as f32;
            let bias1 = 1.0 - beta1.powf(t);
            let bias2 = 1.0 - beta2.powf(t);

            // Update head center
            for i in 0..head.dim() {
                apply_adam(
                    &mut self.adam_m,
                    &mut self.adam_v,
                    &format!("h{head_idx}_c{i}"),
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
                    &format!("t{tail_idx}_c{i}"),
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

            // Update radii/scale
            update_log_param_adam(
                &mut self.adam_m,
                &mut self.adam_v,
                &format!("h{head_idx}_lr"),
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
                &format!("t{tail_idx}_lr"),
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
        let k = 10.0;

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

        let score_head = |head_idx: usize, rel_idx: usize, tail_idx: usize| -> f32 {
            let head = &entities[head_idx];
            let relation = &relations[rel_idx];
            let tail = &entities[tail_idx];
            let transformed = match relation.apply(head) {
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
        use crate::trainer::evaluation::FilteredTripleIndexIds;

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
            let _ = trainer.train_epoch(
                &mut entities,
                &mut relations,
                &triples,
                &config,
                &entity_map,
                &relation_map,
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
        let rel_tol = 0.15;
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

        assert!(
            checked >= 3,
            "gradient check only verified {checked} components (expected >= 3)"
        );
    }
}
