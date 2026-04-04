//! Spherical cap embedding trainer with analytical gradients.
//!
//! Trains spherical cap embeddings using margin-based ranking loss with negative sampling.

use crate::dataset::Triple;
use crate::spherical_cap::{SphericalCap, SphericalCapRelation};
use crate::trainer::CpuBoxTrainingConfig;
use crate::BoxError;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Trains spherical cap embeddings using analytical gradients.
pub struct SphericalCapTrainer {
    rng: StdRng,
    step: usize,
}

impl SphericalCapTrainer {
    /// Create a new trainer with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            step: 0,
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
                let log_scale = self.rng.gen_range(-0.2..0.2);
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
        let loss = (margin - pos_score + neg_score).max(0.0);

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
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let mut indices: Vec<usize> = (0..triples.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = self.rng.random_range(0..=i);
            indices.swap(i, j);
        }

        let mut m: HashMap<String, f32> = HashMap::new();
        let mut v: HashMap<String, f32> = HashMap::new();

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

            let neg_tail_idx = loop {
                let neg = self.rng.random_range(0..num_entities);
                if neg != tail_idx {
                    break neg;
                }
            };

            let head = &entities[head_idx];
            let relation = &relations[rel_idx];
            let tail = &entities[tail_idx];
            let neg_tail = &entities[neg_tail_idx];

            let (loss, grads) =
                Self::compute_pair_gradients(head, relation, tail, neg_tail, config.margin, k);
            total_loss += loss;
            count += 1;

            self.step += 1;
            let t = self.step as f32;
            let bias1 = 1.0 - beta1.powf(t);
            let bias2 = 1.0 - beta2.powf(t);

            // Update centers
            for i in 0..head.dim() {
                apply_adam(
                    &mut m,
                    &mut v,
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
                    &mut m,
                    &mut v,
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
                    &mut m,
                    &mut v,
                    &format!("nt{neg_tail_idx}_c{i}"),
                    &mut entities[neg_tail_idx].center_mut()[i],
                    grads.neg_tail_center[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
                apply_adam(
                    &mut m,
                    &mut v,
                    &format!("r{rel_idx}_a{i}"),
                    &mut relations[rel_idx].axis_mut()[i],
                    grads.relation_axis[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
            }

            // Update angular radii / scale
            update_log_adam(
                &mut m,
                &mut v,
                &format!("h{head_idx}_lt"),
                entities[head_idx].log_tan_half(),
                grads.head_log_tan_half,
                lr,
                beta1,
                beta2,
                eps,
                bias1,
                bias2,
                |e, v| e.set_log_tan_half(v),
                &mut entities[head_idx],
            );
            update_log_adam(
                &mut m,
                &mut v,
                &format!("t{tail_idx}_lt"),
                entities[tail_idx].log_tan_half(),
                grads.tail_log_tan_half,
                lr,
                beta1,
                beta2,
                eps,
                bias1,
                bias2,
                |e, v| e.set_log_tan_half(v),
                &mut entities[tail_idx],
            );
            update_log_adam(
                &mut m,
                &mut v,
                &format!("nt{neg_tail_idx}_lt"),
                entities[neg_tail_idx].log_tan_half(),
                grads.neg_tail_log_tan_half,
                lr,
                beta1,
                beta2,
                eps,
                bias1,
                bias2,
                |e, v| e.set_log_tan_half(v),
                &mut entities[neg_tail_idx],
            );
            update_log_adam(
                &mut m,
                &mut v,
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

            // Re-normalize centers and axes after gradient update
            for e in [
                &mut entities[head_idx],
                &mut entities[tail_idx],
                &mut entities[neg_tail_idx],
            ] {
                let norm: f32 = e.center().iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    for x in e.center_mut() {
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
}
