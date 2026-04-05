//! Subspace embedding trainer with analytical gradients.
//!
//! Trains subspace embeddings using margin-based ranking loss with negative sampling.
//! Subspaces are parameterized as unconstrained d×k matrices; projection uses
//! the general formula P = B(B^T B)^{-1} B^T which works for any full-rank basis.
#![allow(missing_docs)]

use crate::dataset::Triple;
use crate::subspace::Subspace;
use crate::trainer::negative_sampling::{compute_relation_entity_pools, sample_excluding};
use crate::trainer::trainer_utils::AdamState;
use crate::trainer::CpuBoxTrainingConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Trains subspace embeddings using analytical gradients.
pub struct SubspaceTrainer {
    rng: StdRng,
    /// Persistent Adam optimizer state.
    adam: AdamState,
}

impl SubspaceTrainer {
    /// Create a new trainer with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            adam: AdamState::new(),
        }
    }

    /// Initialize entity embeddings randomly.
    /// Each entity gets a subspace of random rank between 1 and max_rank.
    pub fn init_embeddings(
        &mut self,
        num_entities: usize,
        dim: usize,
        max_rank: usize,
    ) -> Vec<Subspace> {
        (0..num_entities)
            .map(|_| {
                let rank = self.rng.random_range(1..=max_rank);
                let vectors: Vec<Vec<f32>> = (0..rank)
                    .map(|_| (0..dim).map(|_| self.rng.random_range(-1.0..1.0)).collect())
                    .collect();
                Subspace::new(vectors).unwrap()
            })
            .collect()
    }

    /// Compute the score for containment A ⊆ B. Lower is better.
    pub fn score_containment(a: &Subspace, b: &Subspace, _k: f32) -> f32 {
        let score = match crate::subspace::containment_score(a, b) {
            Ok(s) => s,
            Err(_) => return f32::INFINITY,
        };
        let score = score.clamp(1e-6, 1.0 - 1e-6);
        -score.ln()
    }

    /// Compute ranking loss and gradients for a (positive, negative) pair.
    /// Positive: head ⊆ tail. Negative: head ⊄ neg_tail.
    fn compute_pair_gradients(
        head: &Subspace,
        tail: &Subspace,
        neg_tail: &Subspace,
        margin: f32,
        _k: f32,
    ) -> (f32, SubspaceGradients) {
        let dim = head.dim();
        let head_rank = head.rank();
        let tail_rank = tail.rank();
        let neg_tail_rank = neg_tail.rank();

        let mut grads = SubspaceGradients::new(dim, head_rank, tail_rank, neg_tail_rank);

        // Positive containment score
        let pos_score = match crate::subspace::containment_score(head, tail) {
            Ok(s) => s.max(1e-10),
            Err(_) => return (0.0, grads),
        };

        // Negative containment score
        let neg_score = match crate::subspace::containment_score(head, neg_tail) {
            Ok(s) => s.max(1e-10),
            Err(_) => return (0.0, grads),
        };

        let loss = (margin - pos_score.ln() + neg_score.ln()).max(0.0);

        if loss <= 1e-8 {
            return (0.0, grads);
        }

        // d_loss/d_pos_score = -1/pos_score
        // d_loss/d_neg_score = 1/neg_score
        let _d_pos = -1.0 / pos_score;
        let _d_neg = 1.0 / neg_score;

        // Compute gradients via numerical differentiation (finite differences)
        // This is simpler and more robust than analytical gradients through
        // the projection/orthonormalization pipeline.
        let eps = 1e-4f32;

        // Gradients w.r.t. head basis
        for j in 0..head_rank {
            for i in 0..dim {
                let mut v = head.basis_vector(j);
                v[i] += eps;
                let vectors: Vec<Vec<f32>> = (0..head_rank)
                    .map(|jj| {
                        if jj == j {
                            v.clone()
                        } else {
                            head.basis_vector(jj)
                        }
                    })
                    .collect();
                let perturbed = match Subspace::new(vectors) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let pos_p = crate::subspace::containment_score(&perturbed, tail)
                    .unwrap_or(pos_score)
                    .max(1e-10);
                let neg_p = crate::subspace::containment_score(&perturbed, neg_tail)
                    .unwrap_or(neg_score)
                    .max(1e-10);
                let loss_p = (margin - pos_p.ln() + neg_p.ln()).max(0.0);
                grads.head_basis[j][i] = (loss_p - loss) / eps;
            }
        }

        // Gradients w.r.t. tail basis
        for j in 0..tail_rank {
            for i in 0..dim {
                let mut v = tail.basis_vector(j);
                v[i] += eps;
                let vectors: Vec<Vec<f32>> = (0..tail_rank)
                    .map(|jj| {
                        if jj == j {
                            v.clone()
                        } else {
                            tail.basis_vector(jj)
                        }
                    })
                    .collect();
                let perturbed = match Subspace::new(vectors) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let pos_p = crate::subspace::containment_score(head, &perturbed)
                    .unwrap_or(pos_score)
                    .max(1e-10);
                let loss_p = (margin - pos_p.ln() + neg_score.ln()).max(0.0);
                grads.tail_basis[j][i] = (loss_p - loss) / eps;
            }
        }

        // Gradients w.r.t. neg_tail basis
        for j in 0..neg_tail_rank {
            for i in 0..dim {
                let mut v = neg_tail.basis_vector(j);
                v[i] += eps;
                let vectors: Vec<Vec<f32>> = (0..neg_tail_rank)
                    .map(|jj| {
                        if jj == j {
                            v.clone()
                        } else {
                            neg_tail.basis_vector(jj)
                        }
                    })
                    .collect();
                let perturbed = match Subspace::new(vectors) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let neg_p = crate::subspace::containment_score(head, &perturbed)
                    .unwrap_or(neg_score)
                    .max(1e-10);
                let loss_p = (margin - pos_score.ln() + neg_p.ln()).max(0.0);
                grads.neg_tail_basis[j][i] = (loss_p - loss) / eps;
            }
        }

        (loss, grads)
    }

    /// Train for one epoch.
    pub fn train_epoch(
        &mut self,
        entities: &mut [Subspace],
        triples: &[Triple],
        config: &CpuBoxTrainingConfig,
        entity_to_idx: &HashMap<String, usize>,
    ) -> f32 {
        let num_entities = entities.len();
        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        let lr = config.learning_rate;

        // Pre-index triples for type-constrained negative sampling.
        // Subspace has no relation; use rel_idx=0 for all triples to build a unified tail pool.
        let indexed_triples: Vec<(usize, usize, usize)> = triples
            .iter()
            .filter_map(|triple| {
                let head_idx = *entity_to_idx.get(&triple.head)?;
                let tail_idx = *entity_to_idx.get(&triple.tail)?;
                Some((head_idx, 0usize, tail_idx))
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

            // Multi-negative sampling with uniform weights
            let n_neg = config.negative_samples.max(1);
            let w = 1.0 / n_neg as f32;
            let head = &entities[head_idx];
            let tail = &entities[tail_idx];
            let mut avg_head_basis: Vec<Vec<f32>> =
                (0..head.rank()).map(|_| vec![0.0; head.dim()]).collect();
            let mut avg_tail_basis: Vec<Vec<f32>> =
                (0..tail.rank()).map(|_| vec![0.0; tail.dim()]).collect();
            let mut avg_loss = 0.0f32;

            let tail_pool = relation_pools.get(&rel_idx);

            for _ in 0..n_neg {
                let neg_tail_idx = tail_pool
                    .and_then(|p| {
                        sample_excluding(&p.tails, tail_idx, |n| self.rng.random_range(0..n))
                    })
                    .unwrap_or_else(|| loop {
                        let neg = self.rng.random_range(0..num_entities);
                        if neg != tail_idx {
                            break neg;
                        }
                    });
                let neg_tail = &entities[neg_tail_idx];
                let head = &entities[head_idx];
                let tail = &entities[tail_idx];
                let (loss, grads) =
                    Self::compute_pair_gradients(head, tail, neg_tail, config.margin, 10.0);
                avg_loss += w * loss;
                for (j, hb) in grads.head_basis.iter().enumerate() {
                    for (i, &g) in hb.iter().enumerate() {
                        avg_head_basis[j][i] += w * g;
                    }
                }
                for (j, tb) in grads.tail_basis.iter().enumerate() {
                    for (i, &g) in tb.iter().enumerate() {
                        avg_tail_basis[j][i] += w * g;
                    }
                }
            }
            total_loss += avg_loss;
            count += 1;

            // Fake grads struct with averaged values for update
            struct FakeGrads {
                head_basis: Vec<Vec<f32>>,
                tail_basis: Vec<Vec<f32>>,
            }
            let grads = FakeGrads {
                head_basis: avg_head_basis,
                tail_basis: avg_tail_basis,
            };
            let head = &entities[head_idx];
            let tail = &entities[tail_idx];

            let (bias1, bias2) = self.adam.tick();

            // Update head basis
            for j in 0..head.rank() {
                for i in 0..head.dim() {
                    self.adam.apply(
                        &format!("h{head_idx}_{j}_{i}"),
                        &mut entities[head_idx].basis_mut()[j][i],
                        grads.head_basis[j][i],
                        lr,
                        bias1,
                        bias2,
                    );
                }
            }

            // Update tail basis
            for j in 0..tail.rank() {
                for i in 0..tail.dim() {
                    self.adam.apply(
                        &format!("t{tail_idx}_{j}_{i}"),
                        &mut entities[tail_idx].basis_mut()[j][i],
                        grads.tail_basis[j][i],
                        lr,
                        bias1,
                        bias2,
                    );
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
        entities: &[Subspace],
        test_triples: &[crate::dataset::TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let num_entities = entities.len();

        let score_tail = |head_idx: usize, _rel_idx: usize, tail_idx: usize| -> f32 {
            let head = &entities[head_idx];
            let tail = &entities[tail_idx];
            crate::subspace::containment_score(head, tail).unwrap_or(0.0)
        };

        let score_head = |head_idx: usize, _rel_idx: usize, tail_idx: usize| -> f32 {
            let head = &entities[head_idx];
            let tail = &entities[tail_idx];
            crate::subspace::containment_score(head, tail).unwrap_or(0.0)
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

struct SubspaceGradients {
    head_basis: Vec<Vec<f32>>,
    tail_basis: Vec<Vec<f32>>,
    neg_tail_basis: Vec<Vec<f32>>,
}

impl SubspaceGradients {
    fn new(dim: usize, head_rank: usize, tail_rank: usize, neg_tail_rank: usize) -> Self {
        Self {
            head_basis: vec![vec![0.0; dim]; head_rank],
            tail_basis: vec![vec![0.0; dim]; tail_rank],
            neg_tail_basis: vec![vec![0.0; dim]; neg_tail_rank],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subspace::Subspace;

    #[test]
    fn trainer_init() {
        let mut trainer = SubspaceTrainer::new(42);
        let entities = trainer.init_embeddings(10, 4, 3);
        assert_eq!(entities.len(), 10);
        assert!(entities[0].rank() >= 1 && entities[0].rank() <= 3);
        assert_eq!(entities[0].dim(), 4);
    }

    #[test]
    fn score_containment_identical_is_low() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let score = SubspaceTrainer::score_containment(&a, &a, 10.0);
        assert!(score < 1.0, "identical score = {score}, expected low");
    }

    #[test]
    fn score_containment_orthogonal_is_high() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![0.0, 0.0, 1.0]]).unwrap();
        let score = SubspaceTrainer::score_containment(&a, &b, 10.0);
        assert!(score > 2.0, "orthogonal score = {score}, expected high");
    }

    #[test]
    fn gradients_are_finite() {
        let head = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let tail = Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap();
        let neg_tail = Subspace::new(vec![vec![0.0, 0.0, 1.0]]).unwrap();

        let (loss, grads) =
            SubspaceTrainer::compute_pair_gradients(&head, &tail, &neg_tail, 1.0, 10.0);

        assert!(loss.is_finite(), "loss not finite: {loss}");
        for (j, row) in grads.head_basis.iter().enumerate() {
            for (i, &g) in row.iter().enumerate() {
                assert!(g.is_finite(), "head_basis[{j}][{i}] not finite: {g}");
            }
        }
    }

    #[test]
    fn train_epoch_runs() {
        let mut trainer = SubspaceTrainer::new(42);
        let mut entities = trainer.init_embeddings(20, 4, 2);

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

        let mut trainer = SubspaceTrainer::new(42);
        let mut entities = trainer.init_embeddings(4, 4, 2);

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.01,
            margin: 1.0,
            ..Default::default()
        };

        let mut last_loss = f32::MAX;
        for epoch in 0..50 {
            let loss = trainer.train_epoch(&mut entities, &triples, &config, &entity_map);
            if epoch % 10 == 0 {
                eprintln!("Subspace Epoch {epoch}: loss={loss:.4}");
            }
            last_loss = loss;
        }
        eprintln!("Subspace Final loss: {last_loss:.4}");

        let results = trainer.evaluate(&entities, &test_triples, None);

        assert!(
            results.mrr > 0.3,
            "Subspace MRR after training = {}, expected > 0.3",
            results.mrr
        );
        assert!(
            results.mean_rank < 3.0,
            "Subspace Mean rank = {}, expected < 3.0",
            results.mean_rank
        );
    }
}
