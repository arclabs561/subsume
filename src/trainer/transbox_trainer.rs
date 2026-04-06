//! TransBox trainer with finite-difference gradients.
#![allow(missing_docs)]

use crate::dataset::Triple;
use crate::trainer::negative_sampling::{compute_relation_entity_pools, sample_excluding};
use crate::trainer::trainer_utils::AdamState;
use crate::trainer::CpuBoxTrainingConfig;
use crate::transbox::{TransBoxConcept, TransBoxRole};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

pub struct TransBoxTrainer {
    rng: StdRng,
    /// Persistent Adam optimizer state.
    adam: AdamState,
}

impl TransBoxTrainer {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            adam: AdamState::new(),
        }
    }

    pub fn init_embeddings(
        &mut self,
        num_concepts: usize,
        num_roles: usize,
        dim: usize,
    ) -> (Vec<TransBoxConcept>, Vec<TransBoxRole>) {
        let concepts: Vec<TransBoxConcept> = (0..num_concepts)
            .map(|_| {
                let center: Vec<f32> = (0..dim).map(|_| self.rng.random_range(-0.1..0.1)).collect();
                let offset: Vec<f32> = (0..dim).map(|_| self.rng.random_range(0.5..2.0)).collect();
                TransBoxConcept::new(center, offset).unwrap()
            })
            .collect();
        let roles: Vec<TransBoxRole> = (0..num_roles)
            .map(|_| {
                let center: Vec<f32> = (0..dim)
                    .map(|_| self.rng.random_range(-0.01..0.01))
                    .collect();
                let offset: Vec<f32> = (0..dim).map(|_| self.rng.random_range(0.01..0.1)).collect();
                TransBoxRole::new(center, offset).unwrap()
            })
            .collect();
        (concepts, roles)
    }

    pub fn score_triple(
        head: &TransBoxConcept,
        role: &TransBoxRole,
        tail: &TransBoxConcept,
        margin: f32,
    ) -> f32 {
        let transformed = match role.apply(head) {
            Ok(t) => t,
            Err(_) => return f32::INFINITY,
        };
        crate::transbox::inclusion_loss(
            transformed.center(),
            transformed.offset(),
            tail.center(),
            tail.offset(),
            margin,
        )
        .unwrap_or(f32::INFINITY)
    }

    fn compute_pair_gradients(
        head: &TransBoxConcept,
        role: &TransBoxRole,
        tail: &TransBoxConcept,
        neg_tail: &TransBoxConcept,
        margin: f32,
    ) -> (f32, TransBoxGradients) {
        let dim = head.dim();
        let mut grads = TransBoxGradients::new(dim);
        let pos_transformed = match role.apply(head) {
            Ok(t) => t,
            Err(_) => return (0.0, grads),
        };
        let pos_loss = crate::transbox::inclusion_loss(
            pos_transformed.center(),
            pos_transformed.offset(),
            tail.center(),
            tail.offset(),
            margin,
        )
        .unwrap_or(0.0);
        let neg_transformed = match role.apply(head) {
            Ok(t) => t,
            Err(_) => return (0.0, grads),
        };
        let neg_loss = crate::transbox::inclusion_loss(
            neg_transformed.center(),
            neg_transformed.offset(),
            neg_tail.center(),
            neg_tail.offset(),
            margin,
        )
        .unwrap_or(0.0);
        let loss = (margin + pos_loss - neg_loss).max(0.0);
        if loss <= 1e-8 {
            return (0.0, grads);
        }
        let eps = 1e-4f32;

        // Head center
        for i in 0..dim {
            let mut c = head.center().to_vec();
            c[i] += eps;
            let p = TransBoxConcept::new(c, head.offset().to_vec()).unwrap_or(head.clone());
            let pt = role.apply(&p).unwrap_or(p.clone());
            let pl = crate::transbox::inclusion_loss(
                pt.center(),
                pt.offset(),
                tail.center(),
                tail.offset(),
                margin,
            )
            .unwrap_or(pos_loss);
            let nt = role.apply(&p).unwrap_or(p.clone());
            let nl = crate::transbox::inclusion_loss(
                nt.center(),
                nt.offset(),
                neg_tail.center(),
                neg_tail.offset(),
                margin,
            )
            .unwrap_or(neg_loss);
            grads.head_center[i] = ((margin + pl - nl).max(0.0) - loss) / eps;
        }
        // Head offset
        for i in 0..dim {
            let mut o = head.offset().to_vec();
            o[i] += eps;
            let p = TransBoxConcept::new(head.center().to_vec(), o).unwrap_or(head.clone());
            let pt = role.apply(&p).unwrap_or(p.clone());
            let pl = crate::transbox::inclusion_loss(
                pt.center(),
                pt.offset(),
                tail.center(),
                tail.offset(),
                margin,
            )
            .unwrap_or(pos_loss);
            let nt = role.apply(&p).unwrap_or(p.clone());
            let nl = crate::transbox::inclusion_loss(
                nt.center(),
                nt.offset(),
                neg_tail.center(),
                neg_tail.offset(),
                margin,
            )
            .unwrap_or(neg_loss);
            grads.head_offset[i] = ((margin + pl - nl).max(0.0) - loss) / eps;
        }
        // Role center
        for i in 0..dim {
            let mut c = role.center().to_vec();
            c[i] += eps;
            let r = TransBoxRole::new(c, role.offset().to_vec()).unwrap_or(role.clone());
            let pt = r.apply(head).unwrap_or(head.clone());
            let pl = crate::transbox::inclusion_loss(
                pt.center(),
                pt.offset(),
                tail.center(),
                tail.offset(),
                margin,
            )
            .unwrap_or(pos_loss);
            let nt = r.apply(head).unwrap_or(head.clone());
            let nl = crate::transbox::inclusion_loss(
                nt.center(),
                nt.offset(),
                neg_tail.center(),
                neg_tail.offset(),
                margin,
            )
            .unwrap_or(neg_loss);
            grads.role_center[i] = ((margin + pl - nl).max(0.0) - loss) / eps;
        }
        // Role offset
        for i in 0..dim {
            let mut o = role.offset().to_vec();
            o[i] += eps;
            let r = TransBoxRole::new(role.center().to_vec(), o).unwrap_or(role.clone());
            let pt = r.apply(head).unwrap_or(head.clone());
            let pl = crate::transbox::inclusion_loss(
                pt.center(),
                pt.offset(),
                tail.center(),
                tail.offset(),
                margin,
            )
            .unwrap_or(pos_loss);
            let nt = r.apply(head).unwrap_or(head.clone());
            let nl = crate::transbox::inclusion_loss(
                nt.center(),
                nt.offset(),
                neg_tail.center(),
                neg_tail.offset(),
                margin,
            )
            .unwrap_or(neg_loss);
            grads.role_offset[i] = ((margin + pl - nl).max(0.0) - loss) / eps;
        }
        // Tail center
        for i in 0..dim {
            let mut c = tail.center().to_vec();
            c[i] += eps;
            let p = TransBoxConcept::new(c, tail.offset().to_vec()).unwrap_or(tail.clone());
            let pl = crate::transbox::inclusion_loss(
                pos_transformed.center(),
                pos_transformed.offset(),
                p.center(),
                p.offset(),
                margin,
            )
            .unwrap_or(pos_loss);
            grads.tail_center[i] = ((margin + pl - neg_loss).max(0.0) - loss) / eps;
        }
        // Tail offset
        for i in 0..dim {
            let mut o = tail.offset().to_vec();
            o[i] += eps;
            let p = TransBoxConcept::new(tail.center().to_vec(), o).unwrap_or(tail.clone());
            let pl = crate::transbox::inclusion_loss(
                pos_transformed.center(),
                pos_transformed.offset(),
                p.center(),
                p.offset(),
                margin,
            )
            .unwrap_or(pos_loss);
            grads.tail_offset[i] = ((margin + pl - neg_loss).max(0.0) - loss) / eps;
        }
        // Neg tail center
        for i in 0..dim {
            let mut c = neg_tail.center().to_vec();
            c[i] += eps;
            let p = TransBoxConcept::new(c, neg_tail.offset().to_vec()).unwrap_or(neg_tail.clone());
            let nl = crate::transbox::inclusion_loss(
                neg_transformed.center(),
                neg_transformed.offset(),
                p.center(),
                p.offset(),
                margin,
            )
            .unwrap_or(neg_loss);
            grads.neg_tail_center[i] = ((margin + pos_loss - nl).max(0.0) - loss) / eps;
        }
        // Neg tail offset
        for i in 0..dim {
            let mut o = neg_tail.offset().to_vec();
            o[i] += eps;
            let p = TransBoxConcept::new(neg_tail.center().to_vec(), o).unwrap_or(neg_tail.clone());
            let nl = crate::transbox::inclusion_loss(
                neg_transformed.center(),
                neg_transformed.offset(),
                p.center(),
                p.offset(),
                margin,
            )
            .unwrap_or(neg_loss);
            grads.neg_tail_offset[i] = ((margin + pos_loss - nl).max(0.0) - loss) / eps;
        }
        (loss, grads)
    }

    pub fn train_epoch(
        &mut self,
        concepts: &mut [TransBoxConcept],
        roles: &mut [TransBoxRole],
        triples: &[Triple],
        config: &CpuBoxTrainingConfig,
        entity_to_idx: &HashMap<String, usize>,
        relation_to_idx: &HashMap<String, usize>,
    ) -> f32 {
        let num_entities = concepts.len();
        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        let lr = config.learning_rate;

        // Pre-index triples for type-constrained negative sampling.
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
            // Multi-negative sampling with uniform weights
            let n_neg = config.negative_samples.max(1);
            let w = 1.0 / n_neg as f32;
            let dim = concepts[head_idx].dim();

            let mut avg_hc = vec![0.0f32; dim];
            let mut avg_ho = vec![0.0f32; dim];
            let mut avg_rc = vec![0.0f32; dim];
            let mut avg_ro = vec![0.0f32; dim];
            let mut avg_tc = vec![0.0f32; dim];
            let mut avg_to = vec![0.0f32; dim];
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
                let head = &concepts[head_idx];
                let role = &roles[rel_idx];
                let tail = &concepts[tail_idx];
                let neg_tail = &concepts[neg_tail_idx];
                let (loss, grads) =
                    Self::compute_pair_gradients(head, role, tail, neg_tail, config.margin);
                avg_loss += w * loss;
                for i in 0..dim {
                    avg_hc[i] += w * grads.head_center[i];
                    avg_ho[i] += w * grads.head_offset[i];
                    avg_rc[i] += w * grads.role_center[i];
                    avg_ro[i] += w * grads.role_offset[i];
                    avg_tc[i] += w * grads.tail_center[i];
                    avg_to[i] += w * grads.tail_offset[i];
                }
            }
            total_loss += avg_loss;
            count += 1;
            let (bias1, bias2) = self.adam.tick();

            for i in 0..dim {
                self.adam.apply(
                    &format!("h{head_idx}_c{i}"),
                    &mut concepts[head_idx].center_mut()[i],
                    avg_hc[i],
                    lr,
                    bias1,
                    bias2,
                );
                self.adam.apply(
                    &format!("h{head_idx}_o{i}"),
                    &mut concepts[head_idx].offset_mut()[i],
                    avg_ho[i],
                    lr,
                    bias1,
                    bias2,
                );
                self.adam.apply(
                    &format!("r{rel_idx}_c{i}"),
                    &mut roles[rel_idx].center_mut()[i],
                    avg_rc[i],
                    lr,
                    bias1,
                    bias2,
                );
                self.adam.apply(
                    &format!("r{rel_idx}_o{i}"),
                    &mut roles[rel_idx].offset_mut()[i],
                    avg_ro[i],
                    lr,
                    bias1,
                    bias2,
                );
                self.adam.apply(
                    &format!("t{tail_idx}_c{i}"),
                    &mut concepts[tail_idx].center_mut()[i],
                    avg_tc[i],
                    lr,
                    bias1,
                    bias2,
                );
                self.adam.apply(
                    &format!("t{tail_idx}_o{i}"),
                    &mut concepts[tail_idx].offset_mut()[i],
                    avg_to[i],
                    lr,
                    bias1,
                    bias2,
                );
            }
            for x in concepts[head_idx].offset_mut() {
                *x = x.max(0.0);
            }
            for x in roles[rel_idx].offset_mut() {
                *x = x.max(0.0);
            }
            for x in concepts[tail_idx].offset_mut() {
                *x = x.max(0.0);
            }
        }
        if count == 0 {
            0.0
        } else {
            total_loss / count as f32
        }
    }

    pub fn evaluate(
        &self,
        concepts: &[TransBoxConcept],
        roles: &[TransBoxRole],
        test_triples: &[crate::dataset::TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let num_entities = concepts.len();
        let margin = 0.0;
        let score = |h: usize, r: usize, t: usize| -> f32 {
            let head = &concepts[h];
            let role = &roles[r];
            let tail = &concepts[t];
            let transformed = match role.apply(head) {
                Ok(t) => t,
                Err(_) => return 0.0,
            };
            let loss = crate::transbox::inclusion_loss(
                transformed.center(),
                transformed.offset(),
                tail.center(),
                tail.offset(),
                margin,
            )
            .unwrap_or(f32::MAX);
            (-loss).exp()
        };
        crate::trainer::evaluation::evaluate_link_prediction_generic(
            test_triples,
            num_entities,
            filter,
            score,
            score,
        )
        .unwrap_or_else(|_| crate::trainer::EvaluationResults {
            mrr: 0.0,
            head_mrr: 0.0,
            tail_mrr: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
            mean_rank: f32::MAX,
            rank_variance: f32::NAN,
            rank_p25: f32::NAN,
            rank_p50: f32::NAN,
            rank_p75: f32::NAN,
            rank_p95: f32::NAN,
            per_relation: vec![],
        })
    }
}

struct TransBoxGradients {
    head_center: Vec<f32>,
    head_offset: Vec<f32>,
    role_center: Vec<f32>,
    role_offset: Vec<f32>,
    tail_center: Vec<f32>,
    tail_offset: Vec<f32>,
    neg_tail_center: Vec<f32>,
    neg_tail_offset: Vec<f32>,
}
impl TransBoxGradients {
    fn new(dim: usize) -> Self {
        Self {
            head_center: vec![0.0; dim],
            head_offset: vec![0.0; dim],
            role_center: vec![0.0; dim],
            role_offset: vec![0.0; dim],
            tail_center: vec![0.0; dim],
            tail_offset: vec![0.0; dim],
            neg_tail_center: vec![0.0; dim],
            neg_tail_offset: vec![0.0; dim],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{TripleIds, Vocab};

    #[test]
    fn trainer_init() {
        let mut t = TransBoxTrainer::new(42);
        let (c, r) = t.init_embeddings(10, 3, 4);
        assert_eq!(c.len(), 10);
        assert_eq!(r.len(), 3);
        assert_eq!(c[0].dim(), 4);
    }

    #[test]
    fn score_contained_is_low() {
        let h = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let r = TransBoxRole::new(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let t = TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        assert!(TransBoxTrainer::score_triple(&h, &r, &t, 0.0) < 0.1);
    }

    #[test]
    fn score_not_contained_is_high() {
        let h = TransBoxConcept::new(vec![0.0, 0.0], vec![2.0, 2.0]).unwrap();
        let r = TransBoxRole::new(vec![0.0, 0.0], vec![0.0, 0.0]).unwrap();
        let t = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        assert!(TransBoxTrainer::score_triple(&h, &r, &t, 0.0) > 1.0);
    }

    #[test]
    fn gradients_are_finite() {
        let h = TransBoxConcept::new(vec![0.0, 0.0], vec![0.5, 0.5]).unwrap();
        let r = TransBoxRole::new(vec![0.1, 0.1], vec![0.1, 0.1]).unwrap();
        let t = TransBoxConcept::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        let nt = TransBoxConcept::new(vec![5.0, 5.0], vec![0.5, 0.5]).unwrap();
        let (loss, grads) = TransBoxTrainer::compute_pair_gradients(&h, &r, &t, &nt, 1.0);
        assert!(loss.is_finite());
        for (i, &g) in grads.head_center.iter().enumerate() {
            assert!(g.is_finite(), "head_center[{i}]={g}");
        }
    }

    #[test]
    fn train_epoch_runs() {
        let mut t = TransBoxTrainer::new(42);
        let (mut c, mut r) = t.init_embeddings(20, 3, 4);
        let triples = vec![Triple {
            head: "e0".to_string(),
            relation: "r0".to_string(),
            tail: "e1".to_string(),
        }];
        let em: HashMap<String, usize> = [("e0".to_string(), 0), ("e1".to_string(), 1)]
            .into_iter()
            .collect();
        let rm: HashMap<String, usize> = [("r0".to_string(), 0)].into_iter().collect();
        let cfg = CpuBoxTrainingConfig {
            learning_rate: 0.01,
            margin: 1.0,
            ..Default::default()
        };
        assert!(t
            .train_epoch(&mut c, &mut r, &triples, &cfg, &em, &rm)
            .is_finite());
    }

    #[test]
    fn train_and_evaluate_synthetic() {
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
        let test = vec![
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
        let em: HashMap<String, usize> = [
            ("e0".into(), 0),
            ("e1".into(), 1),
            ("e2".into(), 2),
            ("e3".into(), 3),
        ]
        .into_iter()
        .collect();
        let rm: HashMap<String, usize> = [("r0".into(), 0), ("r1".into(), 1)].into_iter().collect();
        let mut t = TransBoxTrainer::new(42);
        let (mut c, mut r) = t.init_embeddings(4, 2, 4);
        let cfg = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.5,
            ..Default::default()
        };
        let mut last_loss = f32::MAX;
        for epoch in 0..50 {
            let loss = t.train_epoch(&mut c, &mut r, &triples, &cfg, &em, &rm);
            if epoch % 10 == 0 {
                eprintln!("TransBox Epoch {epoch}: loss={loss:.4}");
            }
            last_loss = loss;
        }
        eprintln!("TransBox Final loss: {last_loss:.4}");
        let results = t.evaluate(&c, &r, &test, None);
        assert!(results.mrr > 0.3, "TransBox MRR = {}", results.mrr);
        assert!(
            results.mean_rank < 3.0,
            "TransBox Mean rank = {}",
            results.mean_rank
        );
    }
}
