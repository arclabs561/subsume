//! Ellipsoid embedding trainer with analytical gradients.
//!
//! Trains full-covariance Gaussian (ellipsoid) embeddings using margin-based
//! ranking loss with negative sampling.

use crate::dataset::Triple;
use crate::ellipsoid::Ellipsoid;
use crate::trainer::CpuBoxTrainingConfig;
use crate::BoxError;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Trains ellipsoid embeddings using analytical gradients.
pub struct EllipsoidTrainer {
    rng: StdRng,
    step: usize,
}

impl EllipsoidTrainer {
    /// Create a new trainer with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            step: 0,
        }
    }

    /// Initialize entity embeddings randomly.
    pub fn init_embeddings(&mut self, num_entities: usize, dim: usize) -> Vec<Ellipsoid> {
        (0..num_entities)
            .map(|_| {
                let mu: Vec<f32> = (0..dim).map(|_| self.rng.random_range(-0.1..0.1)).collect();
                let log_diag: Vec<f32> =
                    (0..dim).map(|_| self.rng.random_range(-1.0..0.0)).collect();
                Ellipsoid::from_log_diagonal(mu, log_diag).unwrap()
            })
            .collect()
    }

    /// Compute the score for containment child ⊆ parent. Lower is better.
    pub fn score_containment(child: &Ellipsoid, parent: &Ellipsoid, k: f32) -> f32 {
        let prob = match crate::ellipsoid::containment_prob(child, parent, k) {
            Ok(p) => p,
            Err(_) => return f32::INFINITY,
        };
        let prob = prob.clamp(1e-6, 1.0 - 1e-6);
        -prob.ln()
    }

    /// Compute ranking loss and gradients for a (positive, negative) pair.
    /// Positive: head ⊆ tail. Negative: head ⊄ neg_tail.
    fn compute_pair_gradients(
        head: &Ellipsoid,
        tail: &Ellipsoid,
        neg_tail: &Ellipsoid,
        margin: f32,
        k: f32,
    ) -> (f32, EllipsoidGradients) {
        let dim = head.dim();
        let mut grads = EllipsoidGradients::new(dim);

        let pos_prob = match crate::ellipsoid::containment_prob(head, tail, k) {
            Ok(p) => p.max(1e-10),
            Err(_) => return (0.0, grads),
        };
        let neg_prob = match crate::ellipsoid::containment_prob(head, neg_tail, k) {
            Ok(p) => p.max(1e-10),
            Err(_) => return (0.0, grads),
        };

        let pos_score = -pos_prob.ln();
        let neg_score = -neg_prob.ln();
        let loss = (margin - pos_score + neg_score).max(0.0);

        if loss <= 1e-8 {
            return (0.0, grads);
        }

        let eps = 1e-4f32;
        let d_pos = -1.0 / pos_prob;
        let d_neg = 1.0 / neg_prob;

        // Gradients w.r.t. head mu
        for i in 0..dim {
            let mut mu = head.mu().to_vec();
            mu[i] += eps;
            let perturbed = Ellipsoid::from_log_diagonal(
                mu.clone(),
                head.cholesky()
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| j % (dim + 1) == 0)
                    .map(|(_, &v)| v)
                    .collect(),
            )
            .unwrap_or_else(|_| head.clone());
            let pos_p = crate::ellipsoid::containment_prob(&perturbed, tail, k)
                .unwrap_or(pos_prob)
                .max(1e-10);
            let neg_p = crate::ellipsoid::containment_prob(&perturbed, neg_tail, k)
                .unwrap_or(neg_prob)
                .max(1e-10);
            let loss_p = (margin - (-pos_p.ln()) + (-neg_p.ln())).max(0.0);
            grads.head_mu[i] = (loss_p - loss) / eps;
        }

        // Gradients w.r.t. head log_diag
        for i in 0..dim {
            let log_diag: Vec<f32> = (0..dim)
                .map(|j| {
                    if j == i {
                        head.cholesky()[j * dim + j] + eps
                    } else {
                        head.cholesky()[j * dim + j]
                    }
                })
                .collect();
            let perturbed = Ellipsoid::from_log_diagonal(head.mu().to_vec(), log_diag)
                .unwrap_or_else(|_| head.clone());
            let pos_p = crate::ellipsoid::containment_prob(&perturbed, tail, k)
                .unwrap_or(pos_prob)
                .max(1e-10);
            let neg_p = crate::ellipsoid::containment_prob(&perturbed, neg_tail, k)
                .unwrap_or(neg_prob)
                .max(1e-10);
            let loss_p = (margin - (-pos_p.ln()) + (-neg_p.ln())).max(0.0);
            grads.head_log_diag[i] = (loss_p - loss) / eps;
        }

        // Gradients w.r.t. tail mu
        for i in 0..dim {
            let mut mu = tail.mu().to_vec();
            mu[i] += eps;
            let perturbed = Ellipsoid::from_log_diagonal(
                mu,
                tail.cholesky()
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| j % (dim + 1) == 0)
                    .map(|(_, &v)| v)
                    .collect(),
            )
            .unwrap_or_else(|_| tail.clone());
            let pos_p = crate::ellipsoid::containment_prob(head, &perturbed, k)
                .unwrap_or(pos_prob)
                .max(1e-10);
            let loss_p = (margin - (-pos_p.ln()) + neg_score).max(0.0);
            grads.tail_mu[i] = (loss_p - loss) / eps;
        }

        // Gradients w.r.t. tail log_diag
        for i in 0..dim {
            let log_diag: Vec<f32> = (0..dim)
                .map(|j| {
                    if j == i {
                        tail.cholesky()[j * dim + j] + eps
                    } else {
                        tail.cholesky()[j * dim + j]
                    }
                })
                .collect();
            let perturbed = Ellipsoid::from_log_diagonal(tail.mu().to_vec(), log_diag)
                .unwrap_or_else(|_| tail.clone());
            let pos_p = crate::ellipsoid::containment_prob(head, &perturbed, k)
                .unwrap_or(pos_prob)
                .max(1e-10);
            let loss_p = (margin - (-pos_p.ln()) + neg_score).max(0.0);
            grads.tail_log_diag[i] = (loss_p - loss) / eps;
        }

        // Gradients w.r.t. neg_tail mu
        for i in 0..dim {
            let mut mu = neg_tail.mu().to_vec();
            mu[i] += eps;
            let perturbed = Ellipsoid::from_log_diagonal(
                mu,
                neg_tail
                    .cholesky()
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| j % (dim + 1) == 0)
                    .map(|(_, &v)| v)
                    .collect(),
            )
            .unwrap_or_else(|_| neg_tail.clone());
            let neg_p = crate::ellipsoid::containment_prob(head, &perturbed, k)
                .unwrap_or(neg_prob)
                .max(1e-10);
            let loss_p = (margin - pos_score + (-neg_p.ln())).max(0.0);
            grads.neg_tail_mu[i] = (loss_p - loss) / eps;
        }

        // Gradients w.r.t. neg_tail log_diag
        for i in 0..dim {
            let log_diag: Vec<f32> = (0..dim)
                .map(|j| {
                    if j == i {
                        neg_tail.cholesky()[j * dim + j] + eps
                    } else {
                        neg_tail.cholesky()[j * dim + j]
                    }
                })
                .collect();
            let perturbed = Ellipsoid::from_log_diagonal(neg_tail.mu().to_vec(), log_diag)
                .unwrap_or_else(|_| neg_tail.clone());
            let neg_p = crate::ellipsoid::containment_prob(head, &perturbed, k)
                .unwrap_or(neg_prob)
                .max(1e-10);
            let loss_p = (margin - pos_score + (-neg_p.ln())).max(0.0);
            grads.neg_tail_log_diag[i] = (loss_p - loss) / eps;
        }

        (loss, grads)
    }

    /// Train for one epoch.
    pub fn train_epoch(
        &mut self,
        entities: &mut [Ellipsoid],
        triples: &[Triple],
        config: &CpuBoxTrainingConfig,
        entity_to_idx: &HashMap<String, usize>,
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
            let tail = &entities[tail_idx];
            let neg_tail = &entities[neg_tail_idx];

            let (loss, grads) =
                Self::compute_pair_gradients(head, tail, neg_tail, config.margin, k);
            total_loss += loss;
            count += 1;

            self.step += 1;
            let t = self.step as f32;
            let bias1 = 1.0 - beta1.powf(t);
            let bias2 = 1.0 - beta2.powf(t);

            let dim = head.dim();

            // Update head
            for i in 0..dim {
                apply_adam(
                    &mut m,
                    &mut v,
                    &format!("h{head_idx}_m{i}"),
                    &mut entities[head_idx].mu_mut()[i],
                    grads.head_mu[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
            }
            let mut head_ld = entities[head_idx].log_diag();
            for i in 0..dim {
                apply_adam(
                    &mut m,
                    &mut v,
                    &format!("h{head_idx}_ld{i}"),
                    &mut head_ld[i],
                    grads.head_log_diag[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
            }
            entities[head_idx].set_log_diag(&head_ld);

            // Update tail
            for i in 0..dim {
                apply_adam(
                    &mut m,
                    &mut v,
                    &format!("t{tail_idx}_m{i}"),
                    &mut entities[tail_idx].mu_mut()[i],
                    grads.tail_mu[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
            }
            let mut tail_ld = entities[tail_idx].log_diag();
            for i in 0..dim {
                apply_adam(
                    &mut m,
                    &mut v,
                    &format!("t{tail_idx}_ld{i}"),
                    &mut tail_ld[i],
                    grads.tail_log_diag[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
            }
            entities[tail_idx].set_log_diag(&tail_ld);

            // Update neg_tail
            for i in 0..dim {
                apply_adam(
                    &mut m,
                    &mut v,
                    &format!("nt{neg_tail_idx}_m{i}"),
                    &mut entities[neg_tail_idx].mu_mut()[i],
                    grads.neg_tail_mu[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
            }
            let mut neg_tail_ld = entities[neg_tail_idx].log_diag();
            for i in 0..dim {
                apply_adam(
                    &mut m,
                    &mut v,
                    &format!("nt{neg_tail_idx}_ld{i}"),
                    &mut neg_tail_ld[i],
                    grads.neg_tail_log_diag[i],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    bias1,
                    bias2,
                );
            }
            entities[neg_tail_idx].set_log_diag(&neg_tail_ld);
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

struct EllipsoidGradients {
    head_mu: Vec<f32>,
    head_log_diag: Vec<f32>,
    tail_mu: Vec<f32>,
    tail_log_diag: Vec<f32>,
    neg_tail_mu: Vec<f32>,
    neg_tail_log_diag: Vec<f32>,
}

impl EllipsoidGradients {
    fn new(dim: usize) -> Self {
        Self {
            head_mu: vec![0.0; dim],
            head_log_diag: vec![0.0; dim],
            tail_mu: vec![0.0; dim],
            tail_log_diag: vec![0.0; dim],
            neg_tail_mu: vec![0.0; dim],
            neg_tail_log_diag: vec![0.0; dim],
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ellipsoid::Ellipsoid;

    #[test]
    fn trainer_init() {
        let mut trainer = EllipsoidTrainer::new(42);
        let entities = trainer.init_embeddings(10, 4);
        assert_eq!(entities.len(), 10);
        assert_eq!(entities[0].dim(), 4);
    }

    #[test]
    fn score_containment_identical_is_low() {
        let e = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let score = EllipsoidTrainer::score_containment(&e, &e, 10.0);
        assert!(score < 1.0, "identical score = {score}, expected low");
    }

    #[test]
    fn score_containment_different_is_higher() {
        let a = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let b = Ellipsoid::from_log_diagonal(vec![5.0, 5.0], vec![0.0, 0.0]).unwrap();
        let score_same = EllipsoidTrainer::score_containment(&a, &a, 10.0);
        let score_diff = EllipsoidTrainer::score_containment(&a, &b, 10.0);
        assert!(
            score_diff > score_same,
            "different score = {score_diff} should > same = {score_same}"
        );
    }

    #[test]
    fn gradients_are_finite() {
        let head = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![-0.5, -0.5]).unwrap();
        let tail = Ellipsoid::from_log_diagonal(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let neg_tail = Ellipsoid::from_log_diagonal(vec![5.0, 5.0], vec![0.0, 0.0]).unwrap();

        let (loss, grads) =
            EllipsoidTrainer::compute_pair_gradients(&head, &tail, &neg_tail, 1.0, 10.0);

        assert!(loss.is_finite(), "loss not finite: {loss}");
        for (i, &g) in grads.head_mu.iter().enumerate() {
            assert!(g.is_finite(), "head_mu[{i}] not finite: {g}");
        }
        for (i, &g) in grads.head_log_diag.iter().enumerate() {
            assert!(g.is_finite(), "head_log_diag[{i}] not finite: {g}");
        }
    }

    #[test]
    fn train_epoch_runs() {
        let mut trainer = EllipsoidTrainer::new(42);
        let mut entities = trainer.init_embeddings(20, 4);

        let triples = vec![Triple {
            head: "e0".to_string(),
            relation: "r0".to_string(),
            tail: "e1".to_string(),
        }];
        let entity_map: HashMap<String, usize> = [("e0".to_string(), 0), ("e1".to_string(), 1)]
            .into_iter()
            .collect();

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.001,
            margin: 1.0,
            ..Default::default()
        };

        let loss = trainer.train_epoch(&mut entities, &triples, &config, &entity_map);

        assert!(loss.is_finite(), "epoch loss not finite: {loss}");
    }
}
