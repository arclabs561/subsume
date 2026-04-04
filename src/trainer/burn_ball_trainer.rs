//! Burn-based ball embedding trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training

use crate::ball::{Ball, BallRelation};
use crate::dataset::TripleIds;
use crate::trainer::negative_sampling::{compute_relation_entity_pools, sample_excluding};
use crate::trainer::trainer_utils::self_adversarial_weights;
use crate::trainer::CpuBoxTrainingConfig;
use burn::module::{Param, ParamId};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

// ---------------------------------------------------------------------------
// Model structs
// ---------------------------------------------------------------------------

/// Entity embedding parameters.
///
/// Must use `Param<Tensor<B, 2>>` — bare `Tensor` fields are treated as
/// constants by Burn's module system (`visit`/`map` are no-ops), so gradients
/// would never reach the optimizer.
///
/// Separate head and tail log-radii let the model simultaneously represent
/// an entity as a small, specific ball when used as a query head (e.g. "dog"
/// contains few things) and as a large, general ball when used as a tail
/// answer (e.g. "animal" contains many things).  The center is shared.
#[derive(Module, Debug)]
pub struct BurnBallEntityParams<B: Backend> {
    /// Center vectors `[num_entities, dim]`.
    pub center: Param<Tensor<B, 2>>,
    /// Log-radius when entity appears as the **head** (query) `[num_entities, 1]`.
    pub log_radius: Param<Tensor<B, 2>>,
    /// Log-radius when entity appears as the **tail** (answer) `[num_entities, 1]`.
    pub tail_log_radius: Param<Tensor<B, 2>>,
}

/// Relation embedding parameters.
///
/// Each relation has separate head and tail translations so the scoring
/// function can model asymmetric containment:
///   `score(h, r, t) = containment(h + r.head_trans, t + r.tail_trans)`
///
/// This mirrors BoxE's "bumping" approach: both head and tail are shifted
/// into a relation-specific sub-region before measuring containment.
#[derive(Module, Debug)]
pub struct BurnBallRelationParams<B: Backend> {
    /// Head translation per relation `[num_relations, dim]`.
    pub translation: Param<Tensor<B, 2>>,
    /// Tail translation per relation `[num_relations, dim]`.
    pub tail_translation: Param<Tensor<B, 2>>,
    /// Log-scale applied to the transformed head radius `[num_relations, 1]`.
    pub log_scale: Param<Tensor<B, 2>>,
}

/// Combined ball embedding model.
#[derive(Module, Debug)]
pub struct BurnBallModel<B: Backend> {
    /// Entity embedding parameters.
    pub entities: BurnBallEntityParams<B>,
    /// Relation embedding parameters.
    pub relations: BurnBallRelationParams<B>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based ball embedding trainer.
pub struct BurnBallTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
}

impl<B: AutodiffBackend> Default for BurnBallTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
        }
    }
}

impl<B: AutodiffBackend> BurnBallTrainer<B> {
    /// Create a new trainer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-epoch seed used for negative sampling.
    ///
    /// Call before each epoch (or let `train_epoch` derive it from the epoch
    /// counter via the `epoch` parameter).
    pub fn set_epoch(&mut self, epoch: u64) {
        // Multiply by a prime to spread seeds apart.
        self.epoch_seed = epoch.wrapping_mul(7919);
    }

    /// Initialize a randomly-weighted model.
    pub fn init_model(
        &self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnBallModel<B> {
        let param = |shape: [usize; 2], lo: f64, hi: f64| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::random(shape, burn::tensor::Distribution::Uniform(lo, hi), device)
                    .require_grad(),
            )
        };
        BurnBallModel {
            entities: BurnBallEntityParams {
                center: param([num_entities, dim], -0.1, 0.1),
                log_radius: param([num_entities, 1], -1.0, 0.0),
                tail_log_radius: param([num_entities, 1], -1.0, 0.0),
            },
            relations: BurnBallRelationParams {
                translation: param([num_relations, dim], -0.01, 0.01),
                tail_translation: param([num_relations, dim], -0.01, 0.01),
                log_scale: param([num_relations, 1], -0.1, 0.1),
            },
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train for one epoch.
    ///
    /// Accepts pre-indexed triples (`TripleIds`) so the caller does the
    /// string→index conversion once, not once per epoch.
    ///
    /// Pass the optimizer in so Adam momentum accumulates across epochs.
    pub fn train_epoch(
        &mut self,
        model: &mut BurnBallModel<B>,
        optim: &mut impl Optimizer<BurnBallModel<B>, B>,
        triples: &[TripleIds],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.entities.center.val().dims()[0];
        let batch_size = config.batch_size.max(1);
        let n_neg = config.negative_samples.max(1);
        let k = config.sigmoid_k;

        // Build per-relation type-constrained negative sampling pools.
        let indexed: Vec<(usize, usize, usize)> = triples
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect();
        let pools = compute_relation_entity_pools(&indexed);

        let n = triples.len();
        if n == 0 {
            return 0.0;
        }

        // Shuffle.
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

            // Corrupt tails (consistent with how we evaluate: rank all tails).
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
            let ranking_loss = self.batch_loss(
                &current_model,
                head_t.clone(),
                rel_t.clone(),
                tail_t.clone(),
                neg_tails,
                n_neg,
                config,
                k,
                device,
            );
            let reg_loss = Self::l2_reg(
                &current_model,
                &head_t,
                &rel_t,
                &tail_t,
                config.regularization,
            );
            let loss = ranking_loss + reg_loss;

            let loss_val = loss.clone().into_scalar().to_f32();
            if loss_val.is_finite() {
                total_loss += loss_val;
                batch_count += 1;
                let grads = GradientsParams::from_grads(loss.backward(), &current_model);
                *model = optim.step(config.learning_rate as f64, current_model, grads);
            } else {
                // Skipping non-finite batch; indicates a learning-rate or init problem.
                #[cfg(debug_assertions)]
                eprintln!("[burn_ball] skipping non-finite batch loss: {loss_val}");
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

    /// Batched ranking loss (margin or InfoNCE) for one batch.
    fn batch_loss(
        &self,
        model: &BurnBallModel<B>,
        head_ids: Tensor<B, 1, Int>,
        rel_ids: Tensor<B, 1, Int>,
        tail_ids: Tensor<B, 1, Int>,
        neg_tail_ids_flat: Vec<i64>, // length bs * n_neg
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        k: f32,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = head_ids.dims()[0];
        let dim = model.entities.center.val().dims()[1];

        // Positive pair embeddings.
        // Head entities use `log_radius` (head/query role).
        // Tail entities use `tail_log_radius` (tail/answer role).
        let hc = model.entities.center.val().select(0, head_ids.clone());
        let hlr = model.entities.log_radius.val().select(0, head_ids);
        let rt = model.relations.translation.val().select(0, rel_ids.clone());
        let rtt = model
            .relations
            .tail_translation
            .val()
            .select(0, rel_ids.clone());
        let rls = model.relations.log_scale.val().select(0, rel_ids);
        let tc = model.entities.center.val().select(0, tail_ids.clone());
        let tlr = model.entities.tail_log_radius.val().select(0, tail_ids);

        // Transform head (shift into relation-specific head region).
        let transformed_c = hc + rt; // [bs, dim]
        let transformed_lr = hlr + rls; // [bs, 1]

        // Transform tail (shift into relation-specific tail region).
        let shifted_tc = tc + rtt.clone(); // [bs, dim]

        // Positive containment margin.
        let pos_dist = (transformed_c.clone() - shifted_tc)
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt();
        let pos_marg = tlr.exp() - pos_dist - transformed_lr.clone().exp(); // [bs, 1]

        // Negative tail margins — negatives also use the tail translation.
        let neg_ids = Tensor::<B, 1, Int>::from_data(neg_tail_ids_flat.as_slice(), device);
        let neg_c = model
            .entities
            .center
            .val()
            .select(0, neg_ids.clone())
            .reshape([bs, n_neg, dim]);
        let neg_lr = model
            .entities
            .tail_log_radius
            .val()
            .select(0, neg_ids)
            .reshape([bs, n_neg, 1]);
        // Broadcast tail_translation [bs, dim] → [bs, n_neg, dim]
        let rtt_rep = rtt.reshape([bs, 1, dim]);
        let shifted_neg_c = neg_c + rtt_rep;
        let tc_rep = transformed_c.reshape([bs, 1, dim]);
        let tlr_rep = transformed_lr.reshape([bs, 1, 1]);
        let neg_dist = (tc_rep - shifted_neg_c)
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .reshape([bs, n_neg, 1]);
        let neg_marg = neg_lr.exp() - neg_dist - tlr_rep.exp(); // [bs, n_neg, 1]

        if config.use_infonce {
            // InfoNCE: cross-entropy over (1 + n_neg)-way softmax on margins.
            let neg_marg_2d = neg_marg.reshape([bs, n_neg]);
            let logits = Tensor::cat(vec![pos_marg.clone(), neg_marg_2d], 1) * k; // [bs, 1+n]
            let max_logit = logits.clone().max_dim(1);
            let lse = (logits - max_logit.clone()).exp().sum_dim(1).log() + max_logit;
            (lse - pos_marg * k).mean()
        } else {
            // Margin ranking loss.
            let pos_loss = sigmoid(pos_marg * k).clamp(1e-6, 1.0 - 1e-6).log().neg(); // [bs, 1]
            let neg_loss_2d = sigmoid(neg_marg.reshape([bs, n_neg]) * k)
                .clamp(1e-6, 1.0 - 1e-6)
                .log()
                .neg(); // [bs, n_neg]

            let neg_loss_avg = if config.self_adversarial {
                Self::apply_self_adv(neg_loss_2d, n_neg, config.adversarial_temperature, device)
            } else {
                neg_loss_2d.mean_dim(1).reshape([bs, 1])
            };

            (config.margin + pos_loss - neg_loss_avg)
                .clamp_min(0.0)
                .mean()
        }
    }

    /// Weighted negative loss using self-adversarial weights (stop-gradient on weights).
    fn apply_self_adv(
        neg_loss: Tensor<B, 2>, // [bs, n_neg]
        n_neg: usize,
        adv_temp: f32,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let bs = neg_loss.dims()[0];
        let data = neg_loss.clone().into_data();
        let slice = data.as_slice::<f32>().expect("neg_loss should be f32");
        let mut weights: Vec<f32> = Vec::with_capacity(slice.len());
        for row in slice.chunks(n_neg) {
            weights.extend(self_adversarial_weights(row, adv_temp));
        }
        let w = Tensor::<B, 1>::from_data(weights.as_slice(), device).reshape([bs, n_neg]);
        (neg_loss * w).sum_dim(1).reshape([bs, 1])
    }

    /// L2 penalty on the embeddings participating in this batch.
    fn l2_reg(
        model: &BurnBallModel<B>,
        head_ids: &Tensor<B, 1, Int>,
        rel_ids: &Tensor<B, 1, Int>,
        tail_ids: &Tensor<B, 1, Int>,
        reg: f32,
    ) -> Tensor<B, 1> {
        if reg == 0.0 {
            return Tensor::<B, 1>::zeros([1], &head_ids.device());
        }
        let hc = model.entities.center.val().select(0, head_ids.clone());
        let tc = model.entities.center.val().select(0, tail_ids.clone());
        let rt = model.relations.translation.val().select(0, rel_ids.clone());
        let rtt = model
            .relations
            .tail_translation
            .val()
            .select(0, rel_ids.clone());
        ((hc.powf_scalar(2.0).mean()
            + tc.powf_scalar(2.0).mean()
            + rt.powf_scalar(2.0).mean()
            + rtt.powf_scalar(2.0).mean())
            * reg)
            .reshape([1])
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Extract model parameters into raw vecs for CPU-side evaluation.
    ///
    /// Returns `(centers, head_log_r, tail_log_r, head_trans, tail_trans, log_scale, n_e, dim, n_r)`.
    fn extract_params_raw(
        model: &BurnBallModel<B>,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        usize,
        usize,
        usize,
    ) {
        let c = model.entities.center.val().into_data();
        let hlr = model.entities.log_radius.val().into_data();
        let tlr = model.entities.tail_log_radius.val().into_data();
        let ht = model.relations.translation.val().into_data();
        let tt = model.relations.tail_translation.val().into_data();
        let ls = model.relations.log_scale.val().into_data();

        let cs: Vec<f32> = c.to_vec().unwrap();
        let hlrs: Vec<f32> = hlr.to_vec().unwrap();
        let tlrs: Vec<f32> = tlr.to_vec().unwrap();
        let hts: Vec<f32> = ht.to_vec().unwrap();
        let tts: Vec<f32> = tt.to_vec().unwrap();
        let lss: Vec<f32> = ls.to_vec().unwrap();

        let n_e = hlrs.len();
        let dim = cs.len() / n_e;
        let n_r = lss.len();
        (cs, hlrs, tlrs, hts, tts, lss, n_e, dim, n_r)
    }

    /// Extract head/tail Ball vecs and BallRelation vec for external use.
    pub fn to_ball_embeddings(
        &self,
        model: &BurnBallModel<B>,
    ) -> (Vec<Ball>, Vec<Ball>, Vec<BallRelation>) {
        let (cs, hlrs, tlrs, hts, _, lss, n_e, dim, n_r) = Self::extract_params_raw(model);
        let head_balls = (0..n_e)
            .map(|i| Ball::from_log_radius(cs[i * dim..(i + 1) * dim].to_vec(), hlrs[i]).unwrap())
            .collect();
        let tail_balls = (0..n_e)
            .map(|i| Ball::from_log_radius(cs[i * dim..(i + 1) * dim].to_vec(), tlrs[i]).unwrap())
            .collect();
        let relations = (0..n_r)
            .map(|i| BallRelation::new(hts[i * dim..(i + 1) * dim].to_vec(), lss[i].exp()).unwrap())
            .collect();
        (head_balls, tail_balls, relations)
    }

    /// Evaluate link prediction on `test_triples`.
    ///
    /// Uses separate head/tail radii and separate head/tail translations per
    /// relation, matching the training objective exactly.
    pub fn evaluate(
        &self,
        model: &BurnBallModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let (cs, hlrs, tlrs, hts, tts, lss, n_e, dim, n_r) = Self::extract_params_raw(model);
        let k = 2.0f32;

        // Pre-build Ball/BallRelation slices.
        let head_balls: Vec<Ball> = (0..n_e)
            .map(|i| Ball::from_log_radius(cs[i * dim..(i + 1) * dim].to_vec(), hlrs[i]).unwrap())
            .collect();
        let tail_balls: Vec<Ball> = (0..n_e)
            .map(|i| Ball::from_log_radius(cs[i * dim..(i + 1) * dim].to_vec(), tlrs[i]).unwrap())
            .collect();
        let head_rels: Vec<BallRelation> = (0..n_r)
            .map(|i| BallRelation::new(hts[i * dim..(i + 1) * dim].to_vec(), lss[i].exp()).unwrap())
            .collect();
        let tail_rels: Vec<BallRelation> = (0..n_r)
            .map(|i| BallRelation::new(tts[i * dim..(i + 1) * dim].to_vec(), 1.0).unwrap())
            .collect();

        // score(h, r, t) = containment(transform_head(h), transform_tail(t))
        let score = |h: usize, r: usize, t: usize| -> f32 {
            let transformed_h = match head_rels[r].apply(&head_balls[h]) {
                Ok(x) => x,
                Err(_) => return 0.0,
            };
            // Shift the tail ball by the relation's tail translation.
            let shifted_t = match tail_rels[r].apply(&tail_balls[t]) {
                Ok(x) => x,
                Err(_) => return 0.0,
            };
            crate::ball::containment_prob(&transformed_h, &shifted_t, k).unwrap_or(0.0)
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

// Numerically stable sigmoid via tanh: σ(x) = (tanh(x/2) + 1) / 2.
fn sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.div_scalar(2.0).tanh().add_scalar(1.0).div_scalar(2.0)
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

    // ── compute_loss (scalar path kept for parity tests) ────────────────────

    /// Scalar margin-ranking loss for a single batch without multiple negatives.
    ///
    /// Used only in tests to verify parity with `batch_loss`.
    fn scalar_loss<B: AutodiffBackend>(
        model: BurnBallModel<B>,
        head_ids: Tensor<B, 1, Int>,
        rel_ids: Tensor<B, 1, Int>,
        tail_ids: Tensor<B, 1, Int>,
        neg_tail_ids: Tensor<B, 1, Int>,
        margin: f32,
        k: f32,
    ) -> Tensor<B, 1> {
        let hc = model.entities.center.val().select(0, head_ids.clone());
        let hlr = model.entities.log_radius.val().select(0, head_ids);
        let rt = model.relations.translation.val().select(0, rel_ids.clone());
        let rtt = model
            .relations
            .tail_translation
            .val()
            .select(0, rel_ids.clone());
        let rls = model.relations.log_scale.val().select(0, rel_ids);
        let tc = model.entities.center.val().select(0, tail_ids.clone());
        let tlr = model.entities.tail_log_radius.val().select(0, tail_ids);
        let neg_tc = model.entities.center.val().select(0, neg_tail_ids.clone());
        let neg_tlr = model.entities.tail_log_radius.val().select(0, neg_tail_ids);

        let transformed_c = hc + rt;
        let transformed_lr = hlr + rls;
        let shifted_tc = tc + rtt.clone();
        let shifted_neg_tc = neg_tc + rtt;

        let pos_marg = tlr.exp()
            - (transformed_c.clone() - shifted_tc)
                .powf_scalar(2.0)
                .sum_dim(1)
                .sqrt()
            - transformed_lr.clone().exp();
        let pos_loss = sigmoid(pos_marg * k).clamp(1e-6, 1.0 - 1e-6).log().neg();

        let neg_marg = neg_tlr.exp()
            - (transformed_c - shifted_neg_tc)
                .powf_scalar(2.0)
                .sum_dim(1)
                .sqrt()
            - transformed_lr.exp();
        let neg_loss = sigmoid(neg_marg * k).clamp(1e-6, 1.0 - 1e-6).log().neg();

        (margin + pos_loss - neg_loss).clamp_min(0.0).mean()
    }

    #[test]
    fn model_init_shapes() {
        let device = Default::default();
        let model = BurnBallTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        assert_eq!(model.entities.center.val().dims(), [10, 4]);
        assert_eq!(model.entities.log_radius.val().dims(), [10, 1]);
        assert_eq!(model.entities.tail_log_radius.val().dims(), [10, 1]);
        assert_eq!(model.relations.translation.val().dims(), [3, 4]);
        assert_eq!(model.relations.log_scale.val().dims(), [3, 1]);
    }

    #[test]
    fn sigmoid_values() {
        let d: <TestBackend as Backend>::Device = Default::default();
        let f = |v: f32| {
            sigmoid(Tensor::<TestBackend, 1>::from_data([v], &d))
                .into_scalar()
                .to_f32()
        };
        assert!((f(0.0) - 0.5).abs() < 1e-6);
        assert!(f(10.0) > 0.99);
        assert!(f(-10.0) < 0.01);
    }

    #[test]
    fn loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
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
            config.sigmoid_k,
            &device,
        );
        let v = loss.into_scalar().to_f32();
        assert!(v.is_finite(), "loss not finite: {v}");
        assert!(v >= 0.0, "loss negative: {v}");
    }

    #[test]
    fn gradient_step_does_not_increase_loss_much() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(10, 3, 4, &device);

        let h = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1], &device);
        let r = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1], &device);
        let t = Tensor::<TestBackend, 1, Int>::from_data([2i64, 3], &device);
        let nt = Tensor::<TestBackend, 1, Int>::from_data([4i64, 5], &device);

        let loss0 = scalar_loss(
            model.clone(),
            h.clone(),
            r.clone(),
            t.clone(),
            nt.clone(),
            1.0,
            2.0,
        );
        let v0 = loss0.clone().into_scalar().to_f32();
        let grads = GradientsParams::from_grads(loss0.backward(), &model);
        let mut optim = AdamConfig::new().init::<TestBackend, BurnBallModel<TestBackend>>();
        model = optim.step(1e-2, model, grads);

        let v1 = scalar_loss(model, h, r, t, nt, 1.0, 2.0)
            .into_scalar()
            .to_f32();
        assert!(v1 <= v0 + 0.5, "loss increased significantly: {v0} -> {v1}");
    }

    #[test]
    fn batch_loss_matches_scalar_loss_single_triple() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let model = trainer.init_model(10, 3, 4, &device);
        let config = CpuBoxTrainingConfig {
            sigmoid_k: 10.0,
            ..Default::default()
        };

        let h = Tensor::<TestBackend, 1, Int>::from_data([0i64], &device);
        let r = Tensor::<TestBackend, 1, Int>::from_data([1i64], &device);
        let t = Tensor::<TestBackend, 1, Int>::from_data([2i64], &device);
        let nt = Tensor::<TestBackend, 1, Int>::from_data([3i64], &device);

        let s = scalar_loss(
            model.clone(),
            h.clone(),
            r.clone(),
            t.clone(),
            nt,
            1.0,
            10.0,
        )
        .into_scalar()
        .to_f32();
        let b = trainer
            .batch_loss(&model, h, r, t, vec![3i64], 1, &config, 10.0, &device)
            .into_scalar()
            .to_f32();
        assert!((s - b).abs() < 1e-5, "scalar {s} != batch {b}");
    }

    #[test]
    fn param_ids_are_tracked_and_survive_clone() {
        use burn::module::list_param_ids;
        let device = Default::default();
        let model = BurnBallTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        let ids = list_param_ids(&model);
        assert_eq!(
            ids.len(),
            6,
            "expected 6 params (center, log_radius, tail_log_radius, translation, tail_translation, log_scale), got {}: {:?}",
            ids.len(),
            ids
        );
        assert_eq!(ids, list_param_ids(&model.clone()));
    }

    #[test]
    fn to_ball_embeddings_shapes() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let (heads, tails, rels) =
            trainer.to_ball_embeddings(&trainer.init_model(5, 2, 4, &device));
        assert_eq!(heads.len(), 5);
        assert_eq!(tails.len(), 5);
        assert_eq!(rels.len(), 2);
        assert_eq!(heads[0].dim(), 4);
        assert!(heads[0].radius() > 0.0);
        assert!(tails[0].radius() > 0.0);
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
        let test_triples = triples.clone();

        let device = Default::default();
        let mut trainer = BurnBallTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(4, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            use_infonce: true,
            negative_samples: 1,
            sigmoid_k: 2.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnBallModel<TestBackend>>();
        let mut last_loss = f32::MAX;
        for epoch in 0..50 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        eprintln!("final loss={last_loss:.4}");
        let results = trainer.evaluate(&model, &test_triples, None);
        eprintln!("MRR={:.3} mean_rank={:.1}", results.mrr, results.mean_rank);
        assert!(results.mrr > 0.3, "MRR={} expected >0.3", results.mrr);
        assert!(
            results.mean_rank <= 3.5,
            "mean_rank={} expected <=3.5",
            results.mean_rank
        );
    }
}
