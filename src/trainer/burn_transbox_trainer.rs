//! Burn-based TransBox trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training
//!
//! # Scoring convention
//! TransBox inclusion loss is **lower = better** (zero means perfect containment).
//! For evaluation, scores are negated so `evaluate_link_prediction_generic` can
//! rank by descending score as usual.

use crate::dataset::TripleIds;
use crate::trainer::negative_sampling::{
    compute_relation_entity_pools, sample_excluding, RelationEntityPools,
};
use crate::trainer::trainer_utils::self_adversarial_weights;
use crate::trainer::CpuBoxTrainingConfig;
use crate::transbox::{TransBoxConcept, TransBoxRole};
use burn::module::{Param, ParamId};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Model structs
// ---------------------------------------------------------------------------

/// Entity embedding parameters for TransBox.
///
/// Each entity is a box with a center and a non-negative offset.  The offset
/// is stored as a raw (unconstrained) parameter and passed through `abs()`
/// before use so that gradient flow is unrestricted while the geometry remains
/// valid.
#[derive(Module, Debug)]
pub struct BurnTransBoxEntityParams<B: Backend> {
    /// Center vectors `[num_entities, dim]`.
    pub center: Param<Tensor<B, 2>>,
    /// Raw offset parameter (apply `.abs()` to obtain the actual offset) `[num_entities, dim]`.
    pub raw_offset: Param<Tensor<B, 2>>,
}

/// Relation embedding parameters for TransBox.
///
/// Each relation is a box-to-box transformation:
///   `transformed_center = head_center + role_center`
///   `transformed_offset = head_offset + role_offset`
#[derive(Module, Debug)]
pub struct BurnTransBoxRelationParams<B: Backend> {
    /// Per-relation center translation `[num_relations, dim]`.
    pub center: Param<Tensor<B, 2>>,
    /// Raw offset parameter (apply `.abs()` to obtain the actual offset) `[num_relations, dim]`.
    pub raw_offset: Param<Tensor<B, 2>>,
}

/// Combined TransBox model.
#[derive(Module, Debug)]
pub struct BurnTransBoxModel<B: Backend> {
    /// Entity embedding parameters.
    pub entities: BurnTransBoxEntityParams<B>,
    /// Relation embedding parameters.
    pub relations: BurnTransBoxRelationParams<B>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based TransBox trainer with autodiff.
pub struct BurnTransBoxTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
    cached_pools: Option<HashMap<usize, RelationEntityPools>>,
}

impl<B: AutodiffBackend> Default for BurnTransBoxTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
            cached_pools: None,
        }
    }
}

impl<B: AutodiffBackend> BurnTransBoxTrainer<B> {
    /// Create a new trainer.
    pub fn new() -> Self {
        Self::default()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch_seed = epoch.wrapping_mul(7919);
    }

    /// Initialize a randomly-weighted model.
    pub fn init_model(
        &self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnTransBoxModel<B> {
        let param = |shape: [usize; 2], lo: f64, hi: f64| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::random(shape, burn::tensor::Distribution::Uniform(lo, hi), device)
                    .require_grad(),
            )
        };
        BurnTransBoxModel {
            entities: BurnTransBoxEntityParams {
                center: param([num_entities, dim], -0.1, 0.1),
                // Initialise offsets positive so abs() is a near-identity at start.
                raw_offset: param([num_entities, dim], 0.5, 2.0),
            },
            relations: BurnTransBoxRelationParams {
                center: param([num_relations, dim], -0.01, 0.01),
                raw_offset: param([num_relations, dim], 0.01, 0.1),
            },
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train for one epoch, returning the mean batch loss.
    pub fn train_epoch(
        &mut self,
        model: &mut BurnTransBoxModel<B>,
        optim: &mut impl Optimizer<BurnTransBoxModel<B>, B>,
        triples: &[TripleIds],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.entities.center.val().dims()[0];
        let batch_size = config.batch_size.max(1);
        let n_neg = config.negative_samples.max(1);

        if self.cached_pools.is_none() {
            let indexed: Vec<(usize, usize, usize)> = triples
                .iter()
                .map(|t| (t.head, t.relation, t.tail))
                .collect();
            self.cached_pools = Some(compute_relation_entity_pools(&indexed));
        }
        let pools = self.cached_pools.as_ref().unwrap();

        let n = triples.len();
        if n == 0 {
            return 0.0;
        }

        let mut rng = fastrand::Rng::with_seed(self.epoch_seed.wrapping_add(1));
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.usize(0..=i));
        }

        let mut total_loss = 0.0f32;
        let mut batch_count = 0usize;

        for chunk in order.chunks(batch_size) {
            let bs = chunk.len();

            let batch_heads: Vec<i64> = chunk.iter().map(|&i| triples[i].head as i64).collect();
            let batch_rels: Vec<i64> = chunk.iter().map(|&i| triples[i].relation as i64).collect();
            let batch_tails: Vec<i64> = chunk.iter().map(|&i| triples[i].tail as i64).collect();

            let mut neg_tails: Vec<i64> = Vec::with_capacity(bs * n_neg);
            for (&ri, &ti) in batch_rels.iter().zip(&batch_tails) {
                let tail_pool = pools
                    .get(&(ri as usize))
                    .map(|p| p.tails.as_slice())
                    .unwrap_or(&[]);
                for _ in 0..n_neg {
                    let neg = sample_excluding(tail_pool, ti as usize, |len| rng.usize(0..len))
                        .map(|v| v as i64)
                        .unwrap_or_else(|| loop {
                            let v = rng.usize(0..num_entities) as i64;
                            if v != ti {
                                break v;
                            }
                        });
                    neg_tails.push(neg);
                }
            }

            let head_t = Tensor::<B, 1, Int>::from_data(batch_heads.as_slice(), device);
            let rel_t = Tensor::<B, 1, Int>::from_data(batch_rels.as_slice(), device);
            let tail_t = Tensor::<B, 1, Int>::from_data(batch_tails.as_slice(), device);

            let current_model = model.clone();
            let loss = self.batch_loss(
                &current_model,
                head_t.clone(),
                rel_t.clone(),
                tail_t.clone(),
                neg_tails,
                n_neg,
                config,
                device,
            );

            let loss_val = loss.clone().into_scalar().to_f32();
            if loss_val.is_finite() {
                total_loss += loss_val;
                batch_count += 1;
                let grads = GradientsParams::from_grads(loss.backward(), &current_model);
                *model = optim.step(config.learning_rate as f64, current_model, grads);
            } else {
                #[cfg(debug_assertions)]
                eprintln!("[burn_transbox] skipping non-finite batch loss: {loss_val}");
            }
        }

        if batch_count == 0 {
            0.0
        } else {
            total_loss / batch_count as f32
        }
    }

    // -----------------------------------------------------------------------
    // Loss
    // -----------------------------------------------------------------------

    /// TransBox inclusion loss for a batch.
    ///
    /// Positive pairs should have low inclusion loss (head ⊆ tail after transform).
    /// Negatives should have high inclusion loss.
    ///
    /// Ranking loss: `relu(margin + pos_loss - neg_loss).mean()`.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnTransBoxModel<B>,
        head_ids: Tensor<B, 1, Int>,
        rel_ids: Tensor<B, 1, Int>,
        tail_ids: Tensor<B, 1, Int>,
        neg_tail_ids_flat: Vec<i64>,
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = head_ids.dims()[0];
        let dim = model.entities.center.val().dims()[1];

        // Entity and relation embeddings for this batch.
        let hc = model.entities.center.val().select(0, head_ids.clone()); // [bs, dim]
        let ho = model.entities.raw_offset.val().select(0, head_ids).abs(); // [bs, dim]
        let tc = model.entities.center.val().select(0, tail_ids.clone()); // [bs, dim]
        let to = model.entities.raw_offset.val().select(0, tail_ids).abs(); // [bs, dim]
        let rc = model.relations.center.val().select(0, rel_ids.clone()); // [bs, dim]
        let ro = model.relations.raw_offset.val().select(0, rel_ids).abs(); // [bs, dim]

        // Transform head through role: center' = hc + rc, offset' = ho + ro.
        let trans_c = hc + rc; // [bs, dim]
        let trans_o = ho + ro; // [bs, dim]

        // Positive inclusion loss: relu(|trans_c - tc| + trans_o - to - margin)^2 summed, sqrt.
        let pos_loss =
            Self::inclusion_loss_batched(trans_c.clone(), trans_o.clone(), tc, to, config.margin); // [bs, 1]

        // Negative tails.
        let neg_ids = Tensor::<B, 1, Int>::from_data(neg_tail_ids_flat.as_slice(), device);
        let neg_c = model
            .entities
            .center
            .val()
            .select(0, neg_ids.clone())
            .reshape([bs, n_neg, dim]); // [bs, n_neg, dim]
        let neg_o = model
            .entities
            .raw_offset
            .val()
            .select(0, neg_ids)
            .abs()
            .reshape([bs, n_neg, dim]); // [bs, n_neg, dim]

        // Broadcast transformed head over n_neg negatives.
        let tc_rep = trans_c.reshape([bs, 1, dim]);
        let to_rep = trans_o.reshape([bs, 1, dim]);

        let neg_loss =
            Self::inclusion_loss_batched_neg(tc_rep, to_rep, neg_c, neg_o, config.margin); // [bs, n_neg]

        // Margin ranking loss: relu(margin + pos_loss - neg_loss).
        // pos_loss is LOW (good match), neg_loss is HIGH (bad match).
        // Loss is zero when neg_loss > pos_loss + margin (clear separation).
        let pos_2d = pos_loss.reshape([bs, 1]); // [bs, 1]
        let loss_per_neg = (pos_2d - neg_loss.clone() + config.margin).clamp_min(0.0); // [bs, n_neg]

        if config.self_adversarial && config.adversarial_temperature > 0.0 {
            let data = neg_loss.clone().into_data();
            let slice = data.as_slice::<f32>().expect("neg_loss f32");
            let mut weights: Vec<f32> = Vec::with_capacity(slice.len());
            // Self-adversarial weights: harder negatives (higher loss) get more weight.
            // Invert sign so high-loss negatives get high weight.
            for row in slice.chunks(n_neg) {
                // Negate because self_adversarial_weights expects scores (higher = harder).
                let neg_scores: Vec<f32> = row.iter().map(|&x| -x).collect();
                weights.extend(self_adversarial_weights(
                    &neg_scores,
                    config.adversarial_temperature,
                ));
            }
            let w = Tensor::<B, 1>::from_data(weights.as_slice(), device).reshape([bs, n_neg]);
            (loss_per_neg * w).sum_dim(1).mean()
        } else {
            loss_per_neg.mean_dim(1).mean()
        }
    }

    /// Per-dimension relu-L2 inclusion loss for positive pairs: `[bs, 1]`.
    ///
    /// `loss_i = sqrt(sum_d relu(|ca_d - cb_d| + oa_d - ob_d - margin)^2)`
    fn inclusion_loss_batched(
        center_a: Tensor<B, 2>, // [bs, dim]
        offset_a: Tensor<B, 2>, // [bs, dim]
        center_b: Tensor<B, 2>, // [bs, dim]
        offset_b: Tensor<B, 2>, // [bs, dim]
        margin: f32,
    ) -> Tensor<B, 2> {
        let violation = (center_a - center_b)
            .abs()
            .add(offset_a)
            .sub(offset_b)
            .sub_scalar(margin)
            .clamp_min(0.0); // [bs, dim]
        violation.powf_scalar(2.0).sum_dim(1).sqrt() // [bs, 1]
    }

    /// Per-dimension relu-L2 inclusion loss for negative pairs: `[bs, n_neg]`.
    fn inclusion_loss_batched_neg(
        center_a: Tensor<B, 3>, // [bs, 1, dim]
        offset_a: Tensor<B, 3>, // [bs, 1, dim]
        center_b: Tensor<B, 3>, // [bs, n_neg, dim]
        offset_b: Tensor<B, 3>, // [bs, n_neg, dim]
        margin: f32,
    ) -> Tensor<B, 2> {
        let bs = center_b.dims()[0];
        let n_neg = center_b.dims()[1];
        let violation = (center_a - center_b)
            .abs()
            .add(offset_a)
            .sub(offset_b)
            .sub_scalar(margin)
            .clamp_min(0.0); // [bs, n_neg, dim]
        violation
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .reshape([bs, n_neg]) // [bs, n_neg]
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Extract raw parameter vecs from the model.
    #[allow(clippy::type_complexity)]
    fn extract_params_raw(
        model: &BurnTransBoxModel<B>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize, usize, usize) {
        let n_e = model.entities.center.val().dims()[0];
        let n_r = model.relations.center.val().dims()[0];
        let dim = model.entities.center.val().dims()[1];

        let entity_centers: Vec<f32> = model.entities.center.val().into_data().to_vec().unwrap();
        let entity_offsets: Vec<f32> = model
            .entities
            .raw_offset
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v.abs())
            .collect();
        let rel_centers: Vec<f32> = model.relations.center.val().into_data().to_vec().unwrap();
        let rel_offsets: Vec<f32> = model
            .relations
            .raw_offset
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v.abs())
            .collect();

        (
            entity_centers,
            entity_offsets,
            rel_centers,
            rel_offsets,
            n_e,
            n_r,
            dim,
        )
    }

    /// Convert model parameters to `TransBoxConcept` and `TransBoxRole` vecs.
    pub fn to_transbox_embeddings(
        &self,
        model: &BurnTransBoxModel<B>,
    ) -> (Vec<TransBoxConcept>, Vec<TransBoxRole>) {
        let (ec, eo, rc, ro, n_e, n_r, dim) = Self::extract_params_raw(model);
        let concepts = (0..n_e)
            .map(|i| {
                TransBoxConcept::new(
                    ec[i * dim..(i + 1) * dim].to_vec(),
                    eo[i * dim..(i + 1) * dim].to_vec(),
                )
                .expect("valid concept")
            })
            .collect();
        let roles = (0..n_r)
            .map(|i| {
                TransBoxRole::new(
                    rc[i * dim..(i + 1) * dim].to_vec(),
                    ro[i * dim..(i + 1) * dim].to_vec(),
                )
                .expect("valid role")
            })
            .collect();
        (concepts, roles)
    }

    /// Evaluate link prediction on `test_triples`.
    ///
    /// Score: `exp(-inclusion_loss)` so higher = better containment, consistent
    /// with `evaluate_link_prediction_generic`'s descending-rank assumption.
    pub fn evaluate(
        &self,
        model: &BurnTransBoxModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let (concepts, roles) = self.to_transbox_embeddings(model);
        let n_e = concepts.len();

        let score = |h: usize, r: usize, t: usize| -> f32 {
            let loss = crate::transbox::score_triple(&concepts[h], &roles[r], &concepts[t], 0.0)
                .unwrap_or(f32::MAX);
            (-loss).exp()
        };

        crate::trainer::evaluation::evaluate_link_prediction_generic(
            test_triples,
            n_e,
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
            per_relation: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::optim::AdamConfig;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn model_init_shapes() {
        let device = Default::default();
        let model = BurnTransBoxTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        assert_eq!(model.entities.center.val().dims(), [10, 4]);
        assert_eq!(model.entities.raw_offset.val().dims(), [10, 4]);
        assert_eq!(model.relations.center.val().dims(), [3, 4]);
        assert_eq!(model.relations.raw_offset.val().dims(), [3, 4]);
    }

    #[test]
    fn loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnTransBoxTrainer::<TestBackend>::new();
        let model = trainer.init_model(20, 3, 4, &device);
        let config = CpuBoxTrainingConfig::default();
        let loss = trainer.batch_loss(
            &model,
            Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 2], &device),
            Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 0], &device),
            Tensor::<TestBackend, 1, Int>::from_data([1i64, 2, 0], &device),
            vec![3i64, 4, 5],
            1,
            &config,
            &device,
        );
        let v = loss.into_scalar().to_f32();
        assert!(v.is_finite(), "loss not finite: {v}");
        assert!(v >= 0.0, "loss negative: {v}");
    }

    #[test]
    fn to_transbox_embeddings_shapes() {
        let device = Default::default();
        let trainer = BurnTransBoxTrainer::<TestBackend>::new();
        let model = trainer.init_model(5, 2, 4, &device);
        let (concepts, roles) = trainer.to_transbox_embeddings(&model);
        assert_eq!(concepts.len(), 5);
        assert_eq!(roles.len(), 2);
        assert_eq!(concepts[0].dim(), 4);
        // Offsets should be non-negative (abs applied).
        for o in concepts[0].offset() {
            assert!(*o >= 0.0);
        }
    }

    #[test]
    fn train_and_evaluate_synthetic() {
        use crate::dataset::TripleIds;

        let triples = vec![
            TripleIds {
                head: 0,
                relation: 0,
                tail: 1,
            },
            TripleIds {
                head: 2,
                relation: 0,
                tail: 3,
            },
            TripleIds {
                head: 0,
                relation: 1,
                tail: 2,
            },
        ];

        let device = Default::default();
        let mut trainer = BurnTransBoxTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(4, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.5,
            negative_samples: 1,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnTransBoxModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..50 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        eprintln!("TransBox burn final loss={last_loss:.4}");

        let results = trainer.evaluate(&model, &triples, None);
        eprintln!(
            "TransBox burn MRR={:.3} mean_rank={:.1}",
            results.mrr, results.mean_rank
        );
        assert!(results.mrr > 0.3, "MRR={} expected >0.3", results.mrr);
        assert!(
            results.mean_rank <= 3.5,
            "mean_rank={} expected <=3.5",
            results.mean_rank
        );
    }

    #[test]
    fn loss_decreases_across_epochs() {
        use crate::dataset::TripleIds;

        let triples = vec![
            TripleIds {
                head: 0,
                relation: 0,
                tail: 1,
            },
            TripleIds {
                head: 2,
                relation: 1,
                tail: 3,
            },
        ];

        let device = Default::default();
        let mut trainer = BurnTransBoxTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.02,
            margin: 0.5,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnTransBoxModel<TestBackend>>();

        let loss_0 = trainer.train_epoch(&mut model, &mut optim, &triples, 0, &config, &device);
        let mut loss_last = loss_0;
        for epoch in 1..20 {
            loss_last =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        assert!(
            loss_last < loss_0,
            "loss should decrease: epoch 0 = {loss_0:.4}, epoch 19 = {loss_last:.4}"
        );
    }
}
