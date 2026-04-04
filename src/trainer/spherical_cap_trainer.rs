//! Spherical cap embedding trainer with analytical gradients.
//!
//! Trains spherical cap embeddings using margin-based ranking loss with negative sampling.
#![allow(missing_docs)]

use crate::dataset::Triple;
use crate::spherical_cap::{SphericalCap, SphericalCapRelation};
use crate::trainer::trainer_utils::AdamState;
use crate::trainer::CpuBoxTrainingConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Trains spherical cap embeddings using analytical gradients.
pub struct SphericalCapTrainer {
    rng: StdRng,
    step: usize,
    /// Persistent Adam optimizer state.
    adam: AdamState,
}

impl SphericalCapTrainer {
    /// Create a new trainer with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            step: 0,
            adam: AdamState::new(),
        }
    }

    /// Initialize entity and relation embeddings randomly.
    pub fn init_embeddings(
        &mut self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
    ) -> (Vec<SphericalCap>, Vec<SphericalCapRelation>) {
        let entities: Vec<SphericalCap> = (0..num_entities)
            .map(|_| {
                let center: Vec<f32> = (0..dim).map(|_| self.rng.gen_range(-1.0..1.0)).collect();
                let log_tan_half = self.rng.gen_range(-1.0..0.0);
                SphericalCap::from_log_tan_half(center, log_tan_half).unwrap()
            })
            .collect();

        let relations: Vec<SphericalCapRelation> = (0..num_relations)
            .map(|_| {
                let axis: Vec<f32> = (0..dim).map(|_| self.rng.gen_range(-1.0..1.0)).collect();
                let angle = self.rng.gen_range(-0.5..0.5);
                let log_scale: f32 = self.rng.gen_range(-0.2..0.2);
                SphericalCapRelation::new(axis, angle, log_scale.exp()).unwrap()
            })
            .collect();

        (entities, relations)
    }

    /// Compute the score for a triple. Lower is better.
    pub fn score_triple(
        head: &SphericalCap,
        relation: &SphericalCapRelation,
        tail: &SphericalCap,
        k: f32,
    ) -> f32 {
        let transformed = match relation.apply(head) {
            Ok(t) => t,
            Err(_) => return f32::INFINITY,
        };
        let prob = match crate::spherical_cap::containment_prob(&transformed, tail, k) {
            Ok(p) => p,
            Err(_) => return f32::INFINITY,
        };
        let prob = prob.clamp(1e-6, 1.0 - 1e-6);
        -prob.ln()
    }

    /// Compute ranking loss and gradients for a (positive, negative) pair.
    fn compute_pair_gradients(
        head: &SphericalCap,
        relation: &SphericalCapRelation,
        tail: &SphericalCap,
        neg_tail: &SphericalCap,
        margin: f32,
        k: f32,
    ) -> (f32, CapGradients) {
        let dim = head.dim();
        let mut grads = CapGradients::new(dim);

        let pos_transformed = match relation.apply(head) {
            Ok(t) => t,
            Err(_) => return (0.0, grads),
        };
        let pos_prob = match crate::spherical_cap::containment_prob(&pos_transformed, tail, k) {
            Ok(p) => p.max(1e-10),
            Err(_) => return (0.0, grads),
        };

        let neg_transformed = match relation.apply(head) {
            Ok(t) => t,
            Err(_) => return (0.0, grads),
        };
        let neg_prob = match crate::spherical_cap::containment_prob(&neg_transformed, neg_tail, k) {
            Ok(p) => p.max(1e-10),
            Err(_) => return (0.0, grads),
        };

        let pos_score = -pos_prob.ln();
        let neg_score = -neg_prob.ln();
        let loss = (margin + pos_score - neg_score).max(0.0);

        if loss <= 1e-8 {
            return (0.0, grads);
        }

        let pos_dist =
            crate::spherical_cap::geodesic_distance(pos_transformed.center(), tail.center());
        let neg_dist =
            crate::spherical_cap::geodesic_distance(neg_transformed.center(), neg_tail.center());

        let pos_deriv = k * pos_prob * (1.0 - pos_prob);
        let neg_deriv = k * neg_prob * (1.0 - neg_prob);

        let d_pos = -1.0 / pos_prob;
        let d_neg = 1.0 / neg_prob;

        // Gradients w.r.t. transformed head center
        if pos_dist > 1e-8 {
            for i in 0..dim {
                let diff = pos_transformed.center()[i] - tail.center()[i];
                let d_dist = diff / pos_dist;
                grads.head_center[i] += d_pos * pos_deriv * (-d_dist);
            }
        }
        if neg_dist > 1e-8 {
            for i in 0..dim {
                let diff = neg_transformed.center()[i] - neg_tail.center()[i];
                let d_dist = diff / neg_dist;
                grads.head_center[i] += d_neg * neg_deriv * (-d_dist);
            }
        }

        // Gradients w.r.t. angular radius
        grads.head_log_tan_half += d_pos * pos_deriv * (-1.0) + d_neg * neg_deriv * (-1.0);
        grads.tail_log_tan_half += d_pos * pos_deriv;
        grads.neg_tail_log_tan_half += d_neg * neg_deriv;

        // Gradients w.r.t. tail center
        if pos_dist > 1e-8 {
            for i in 0..dim {
                let diff = tail.center()[i] - pos_transformed.center()[i];
                let d_dist = diff / pos_dist;
                grads.tail_center[i] += d_pos * pos_deriv * (-d_dist);
            }
        }
        if neg_dist > 1e-8 {
            for i in 0..dim {
                let diff = neg_tail.center()[i] - neg_transformed.center()[i];
                let d_dist = diff / neg_dist;
                grads.neg_tail_center[i] += d_neg * neg_deriv * (-d_dist);
            }
        }

        // Through relation: translation -> center, scale -> angular_radius
        for i in 0..dim {
            grads.relation_axis[i] = grads.head_center[i];
        }
        grads.relation_log_scale = grads.head_log_tan_half * (head.angular_radius() / 2.0).tan();

        (loss, grads)
    }

    /// Train for one epoch.
    pub fn train_epoch(
        &mut self,
        entities: &mut [SphericalCap],
        relations: &mut [SphericalCapRelation],
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
        let _beta1 = self.adam.beta1;
        let _beta2 = self.adam.beta2;
        let _eps = self.adam.eps;

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

            // Multi-negative sampling with uniform weights
            let n_neg = config.negative_samples.max(1);
            let w = 1.0 / n_neg as f32;
            let mut avg_grads = CapGradients::new(head.dim());
            let mut avg_loss = 0.0f32;

            for _ in 0..n_neg {
                let neg_tail_idx = loop {
                    let neg = self.rng.random_range(0..num_entities);
                    if neg != tail_idx {
                        break neg;
                    }
                };
                let neg_tail = &entities[neg_tail_idx];
                let (loss, grads) =
                    Self::compute_pair_gradients(head, relation, tail, neg_tail, config.margin, k);
                avg_loss += w * loss;
                for i in 0..head.dim() {
                    avg_grads.head_center[i] += w * grads.head_center[i];
                    avg_grads.tail_center[i] += w * grads.tail_center[i];
                    avg_grads.neg_tail_center[i] += w * grads.neg_tail_center[i];
                    avg_grads.relation_axis[i] += w * grads.relation_axis[i];
                }
                avg_grads.head_log_tan_half += w * grads.head_log_tan_half;
                avg_grads.tail_log_tan_half += w * grads.tail_log_tan_half;
                avg_grads.neg_tail_log_tan_half += w * grads.neg_tail_log_tan_half;
                avg_grads.relation_log_scale += w * grads.relation_log_scale;
            }

            total_loss += avg_loss;
            count += 1;
            let grads = avg_grads;

            let (bias1, bias2) = self.adam.tick();

            // Update centers
            for i in 0..head.dim() {
                self.adam.apply(
                    &format!("h{head_idx}_c{i}"),
                    &mut entities[head_idx].center_mut()[i],
                    grads.head_center[i],
                    lr,
                    bias1,
                    bias2,
                );
                self.adam.apply(
                    &format!("t{tail_idx}_c{i}"),
                    &mut entities[tail_idx].center_mut()[i],
                    grads.tail_center[i],
                    lr,
                    bias1,
                    bias2,
                );
                self.adam.apply(
                    &format!("r{rel_idx}_a{i}"),
                    &mut relations[rel_idx].axis_mut()[i],
                    grads.relation_axis[i],
                    lr,
                    bias1,
                    bias2,
                );
            }

            // Update angular radii / scale (head, tail, relation only — neg_tail skipped for multi-neg)
            self.adam.apply_log(
                &format!("h{head_idx}_lt"),
                entities[head_idx].log_tan_half(),
                grads.head_log_tan_half,
                lr,
                bias1,
                bias2,
                |e, v| e.set_log_tan_half(v),
                &mut entities[head_idx],
            );
            self.adam.apply_log(
                &format!("t{tail_idx}_lt"),
                entities[tail_idx].log_tan_half(),
                grads.tail_log_tan_half,
                lr,
                bias1,
                bias2,
                |e, v| e.set_log_tan_half(v),
                &mut entities[tail_idx],
            );
            self.adam.apply_log(
                &format!("r{rel_idx}_ls"),
                relations[rel_idx].log_scale(),
                grads.relation_log_scale,
                lr,
                bias1,
                bias2,
                |r, v| r.set_log_scale(v),
                &mut relations[rel_idx],
            );

            // Re-normalize centers and axes after gradient update (head and tail only)
            for idx in [head_idx, tail_idx] {
                let norm: f32 = entities[idx]
                    .center()
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                if norm > 1e-12 {
                    for x in entities[idx].center_mut() {
                        *x /= norm;
                    }
                }
            }
            let axis_norm: f32 = relations[rel_idx]
                .axis()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            if axis_norm > 1e-12 {
                for x in relations[rel_idx].axis_mut() {
                    *x /= axis_norm;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            total_loss / count as f32
        }
    }

    /// Evaluate the trained model on test triples using filtered link prediction.
    pub fn evaluate(
        &self,
        entities: &[SphericalCap],
        relations: &[SphericalCapRelation],
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
            crate::spherical_cap::containment_prob(&transformed, tail, k).unwrap_or(0.0)
        };

        let score_head = |head_idx: usize, rel_idx: usize, tail_idx: usize| -> f32 {
            let head = &entities[head_idx];
            let relation = &relations[rel_idx];
            let tail = &entities[tail_idx];
            let transformed = match relation.apply(head) {
                Ok(t) => t,
                Err(_) => return 0.0,
            };
            crate::spherical_cap::containment_prob(&transformed, tail, k).unwrap_or(0.0)
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
// Gradients
// ---------------------------------------------------------------------------

struct CapGradients {
    head_center: Vec<f32>,
    head_log_tan_half: f32,
    relation_axis: Vec<f32>,
    relation_log_scale: f32,
    tail_center: Vec<f32>,
    tail_log_tan_half: f32,
    neg_tail_center: Vec<f32>,
    neg_tail_log_tan_half: f32,
}

impl CapGradients {
    fn new(dim: usize) -> Self {
        Self {
            head_center: vec![0.0; dim],
            head_log_tan_half: 0.0,
            relation_axis: vec![0.0; dim],
            relation_log_scale: 0.0,
            tail_center: vec![0.0; dim],
            tail_log_tan_half: 0.0,
            neg_tail_center: vec![0.0; dim],
            neg_tail_log_tan_half: 0.0,
        }
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

fn update_log_adam<T, F>(
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spherical_cap::SphericalCap;
    use std::f32::consts::PI;

    #[test]
    fn trainer_init() {
        let mut trainer = SphericalCapTrainer::new(42);
        let (entities, relations) = trainer.init_embeddings(10, 3, 4);
        assert_eq!(entities.len(), 10);
        assert_eq!(relations.len(), 3);
        assert_eq!(entities[0].dim(), 4);
    }

    #[test]
    fn score_triple_identical_is_low() {
        let head = SphericalCap::new(vec![1.0, 0.0, 0.0], PI / 4.0).unwrap();
        let relation = SphericalCapRelation::identity(3);
        let tail = SphericalCap::new(vec![1.0, 0.0, 0.0], PI / 4.0).unwrap();
        let score = SphericalCapTrainer::score_triple(&head, &relation, &tail, 10.0);
        assert!(score < 1.0, "identical score = {score}, expected low");
    }

    #[test]
    fn score_triple_disjoint_is_high() {
        let head = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.1).unwrap();
        let relation = SphericalCapRelation::identity(3);
        let tail = SphericalCap::new(vec![-1.0, 0.0, 0.0], 0.1).unwrap();
        let score = SphericalCapTrainer::score_triple(&head, &relation, &tail, 10.0);
        assert!(score > 5.0, "disjoint score = {score}, expected high");
    }

    #[test]
    fn gradients_are_finite() {
        let head = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.3).unwrap();
        let relation = SphericalCapRelation::new(vec![0.0, 1.0, 0.0], 0.1, 1.0).unwrap();
        let tail = SphericalCap::new(vec![1.0, 0.0, 0.0], 0.8).unwrap();
        let neg_tail = SphericalCap::new(vec![-1.0, 0.0, 0.0], 0.1).unwrap();

        let (loss, grads) = SphericalCapTrainer::compute_pair_gradients(
            &head, &relation, &tail, &neg_tail, 1.0, 10.0,
        );

        assert!(loss.is_finite(), "loss not finite: {loss}");
        for (i, &g) in grads.head_center.iter().enumerate() {
            assert!(g.is_finite(), "head_center[{i}] not finite: {g}");
        }
        assert!(grads.head_log_tan_half.is_finite());
        assert!(grads.relation_log_scale.is_finite());
    }

    #[test]
    fn train_epoch_runs() {
        let mut trainer = SphericalCapTrainer::new(42);
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
                relation: 0,
                tail: e1,
            },
            TripleIds {
                head: e2,
                relation: 0,
                tail: e3,
            },
            TripleIds {
                head: e0,
                relation: 1,
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

        let mut trainer = SphericalCapTrainer::new(42);
        let (mut entities, mut relations) = trainer.init_embeddings(4, 2, 4);

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
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
                eprintln!("Cap Epoch {epoch}: loss={loss:.4}");
            }
            last_loss = loss;
        }
        eprintln!("Cap Final loss: {last_loss:.4}");

        let results = trainer.evaluate(&entities, &relations, &test_triples, None);

        assert!(
            results.mrr > 0.3,
            "Cap MRR after training = {}, expected > 0.3",
            results.mrr
        );
        assert!(
            results.mean_rank < 3.0,
            "Cap Mean rank = {}, expected < 3.0",
            results.mean_rank
        );
    }
}
