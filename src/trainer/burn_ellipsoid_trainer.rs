//! Burn-based diagonal-Gaussian ellipsoid trainer with autodiff.
//!
//! Uses a diagonal covariance simplification of the full Cholesky parameterization
//! in `ellipsoid.rs`.  Each entity is `N(mu, diag(exp(2 * log_sigma)))`.
//!
//! # KL divergence (diagonal Gaussians)
//!
//! ```text
//! KL(child || parent) = 0.5 * sum_i [
//!     exp(2*(ls_c[i] - ls_p[i]))          // variance ratio
//!   + (mu_p[i] - mu_c[i])^2 / exp(2*ls_p[i])  // Mahalanobis
//!   - 1
//!   + 2*(ls_p[i] - ls_c[i])               // log-det ratio
//! ]
//! ```
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training

#![allow(missing_docs)]

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

/// Entity parameters for diagonal-Gaussian ellipsoid embeddings.
///
/// Separate head/tail log-sigma lets the same entity simultaneously act as a
/// narrow distribution (when queried as head) and a broad one (when answering
/// as tail), matching the directional nature of subsumption.
#[derive(Module, Debug)]
pub struct BurnEllipsoidEntityParams<B: Backend> {
    /// Mean vectors `[num_entities, dim]`.
    pub mu: Param<Tensor<B, 2>>,
    /// Head log-std (used when entity is a query head) `[num_entities, dim]`.
    pub log_sigma: Param<Tensor<B, 2>>,
    /// Tail log-std (used when entity is a candidate answer) `[num_entities, dim]`.
    pub tail_log_sigma: Param<Tensor<B, 2>>,
}

/// Relation parameters for diagonal-Gaussian ellipsoid embeddings.
///
/// A relation shifts the head mean and scales its log-std before the KL
/// divergence with the tail is measured.
#[derive(Module, Debug)]
pub struct BurnEllipsoidRelationParams<B: Backend> {
    /// Additive shift on the head mean `[num_relations, dim]`.
    pub translation: Param<Tensor<B, 2>>,
    /// Additive shift on the tail mean `[num_relations, dim]`.
    pub tail_translation: Param<Tensor<B, 2>>,
    /// Additive shift on the head log-std `[num_relations, dim]`.
    pub log_scale: Param<Tensor<B, 2>>,
}

/// Combined diagonal-Gaussian ellipsoid model.
#[derive(Module, Debug)]
pub struct BurnEllipsoidModel<B: Backend> {
    /// Entity embedding parameters.
    pub entities: BurnEllipsoidEntityParams<B>,
    /// Relation embedding parameters.
    pub relations: BurnEllipsoidRelationParams<B>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based diagonal-Gaussian ellipsoid trainer.
pub struct BurnEllipsoidTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
}

impl<B: AutodiffBackend> Default for BurnEllipsoidTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
        }
    }
}

impl<B: AutodiffBackend> BurnEllipsoidTrainer<B> {
    /// Create a new trainer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the epoch counter (multiplied by a prime to spread seeds apart).
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch_seed = epoch.wrapping_mul(7919);
    }

    /// Initialize a randomly-weighted model.
    pub fn init_model(
        &self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnEllipsoidModel<B> {
        let param = |shape: [usize; 2], lo: f64, hi: f64| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::random(shape, burn::tensor::Distribution::Uniform(lo, hi), device)
                    .require_grad(),
            )
        };
        BurnEllipsoidModel {
            entities: BurnEllipsoidEntityParams {
                mu: param([num_entities, dim], -0.1, 0.1),
                log_sigma: param([num_entities, dim], -1.0, 0.0),
                tail_log_sigma: param([num_entities, dim], -1.0, 0.0),
            },
            relations: BurnEllipsoidRelationParams {
                translation: param([num_relations, dim], -0.01, 0.01),
                tail_translation: param([num_relations, dim], -0.01, 0.01),
                log_scale: param([num_relations, dim], -0.1, 0.1),
            },
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train for one epoch, returning the mean batch loss.
    pub fn train_epoch(
        &mut self,
        model: &mut BurnEllipsoidModel<B>,
        optim: &mut impl Optimizer<BurnEllipsoidModel<B>, B>,
        triples: &[TripleIds],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.entities.mu.val().dims()[0];
        let batch_size = config.batch_size.max(1);
        let n_neg = config.negative_samples.max(1);
        let k = config.sigmoid_k;

        let indexed: Vec<(usize, usize, usize)> = triples
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect();
        let pools = compute_relation_entity_pools(&indexed);

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
                #[cfg(debug_assertions)]
                eprintln!("[burn_ellipsoid] skipping non-finite batch loss: {loss_val}");
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

    /// Diagonal KL divergence: KL(child || parent) summed over dimensions.
    ///
    /// Returns a `[bs, 1]` tensor (one KL value per positive pair in the batch).
    fn kl_diagonal(
        mu_c: Tensor<B, 2>, // [bs, dim]
        ls_c: Tensor<B, 2>, // [bs, dim]
        mu_p: Tensor<B, 2>, // [bs, dim]
        ls_p: Tensor<B, 2>, // [bs, dim]
    ) -> Tensor<B, 2> {
        // variance ratio: exp(2*(ls_c - ls_p))
        let var_ratio = (ls_c.clone() - ls_p.clone()).mul_scalar(2.0).exp();
        // Mahalanobis: (mu_p - mu_c)^2 / exp(2*ls_p)
        let mahal = (mu_p - mu_c).powf_scalar(2.0) / ls_p.clone().mul_scalar(2.0).exp();
        // log-det ratio: 2*(ls_p - ls_c)
        let log_det = (ls_p - ls_c).mul_scalar(2.0);
        // KL = 0.5 * sum_i [var_ratio + mahal - 1 + log_det]
        (var_ratio + mahal + log_det.clone() - Tensor::ones_like(&log_det))
            .mul_scalar(0.5)
            .sum_dim(1) // [bs, 1]
    }

    /// Batched ranking loss for one batch.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnEllipsoidModel<B>,
        head_ids: Tensor<B, 1, Int>,
        rel_ids: Tensor<B, 1, Int>,
        tail_ids: Tensor<B, 1, Int>,
        neg_tail_ids_flat: Vec<i64>,
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        k: f32,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = head_ids.dims()[0];
        let dim = model.entities.mu.val().dims()[1];

        // Head entity params.
        let h_mu = model.entities.mu.val().select(0, head_ids.clone()); // [bs, dim]
        let h_ls = model.entities.log_sigma.val().select(0, head_ids); // [bs, dim]

        // Relation params.
        let r_t = model.relations.translation.val().select(0, rel_ids.clone()); // [bs, dim]
        let r_tt = model
            .relations
            .tail_translation
            .val()
            .select(0, rel_ids.clone()); // [bs, dim]
        let r_ls = model.relations.log_scale.val().select(0, rel_ids); // [bs, dim]

        // Tail entity params.
        let t_mu = model.entities.mu.val().select(0, tail_ids.clone()); // [bs, dim]
        let t_ls = model.entities.tail_log_sigma.val().select(0, tail_ids); // [bs, dim]

        // Transform head (shift into relation-specific child distribution).
        let trans_mu = h_mu + r_t; // [bs, dim]
        let trans_ls = h_ls + r_ls; // [bs, dim]

        // Transform tail.
        let shifted_t_mu = t_mu + r_tt.clone(); // [bs, dim]

        // Positive KL: child = transformed head, parent = shifted tail.
        let pos_kl = Self::kl_diagonal(trans_mu.clone(), trans_ls.clone(), shifted_t_mu, t_ls); // [bs, 1]

        // score = sigmoid(-k * kl); higher score = better containment.
        let pos_score = sigmoid((pos_kl.clone().neg() * k).reshape([bs, 1])); // [bs, 1]
        let pos_loss = pos_score.clamp(1e-6, 1.0 - 1e-6).log().neg(); // [bs, 1]

        // Negative tails.
        let neg_ids = Tensor::<B, 1, Int>::from_data(neg_tail_ids_flat.as_slice(), device);
        let neg_mu = model
            .entities
            .mu
            .val()
            .select(0, neg_ids.clone())
            .reshape([bs, n_neg, dim]); // [bs, n_neg, dim]
        let neg_ls = model
            .entities
            .tail_log_sigma
            .val()
            .select(0, neg_ids)
            .reshape([bs, n_neg, dim]); // [bs, n_neg, dim]

        // Broadcast tail_translation [bs, dim] -> [bs, n_neg, dim].
        let rtt_rep = r_tt.reshape([bs, 1, dim]);
        let shifted_neg_mu = neg_mu + rtt_rep; // [bs, n_neg, dim]

        // Repeat head params: [bs, dim] -> [bs, n_neg, dim].
        let trans_mu_rep = trans_mu.reshape([bs, 1, dim]);
        let trans_ls_rep = trans_ls.reshape([bs, 1, dim]);

        // KL for each negative.
        let var_ratio_neg = (trans_ls_rep.clone() - neg_ls.clone())
            .mul_scalar(2.0)
            .exp(); // [bs, n_neg, dim]
        let mahal_neg =
            (shifted_neg_mu - trans_mu_rep).powf_scalar(2.0) / neg_ls.clone().mul_scalar(2.0).exp(); // [bs, n_neg, dim]
        let log_det_neg = (neg_ls - trans_ls_rep).mul_scalar(2.0); // [bs, n_neg, dim]
        let neg_kl = (var_ratio_neg + mahal_neg + log_det_neg.clone()
            - Tensor::ones_like(&log_det_neg))
        .mul_scalar(0.5)
        .sum_dim(2); // [bs, n_neg, 1]

        let neg_score = sigmoid((neg_kl.clone().neg() * k).reshape([bs, n_neg])); // [bs, n_neg]
        let neg_loss = neg_score.clamp(1e-6, 1.0 - 1e-6).log().neg(); // [bs, n_neg]

        if config.use_infonce {
            let pos_score_2d = sigmoid((pos_kl.neg() * k).reshape([bs, 1]));
            let neg_score_2d = sigmoid(neg_kl.reshape([bs, n_neg]).neg() * k);
            let logits = Tensor::cat(vec![pos_score_2d.clone(), neg_score_2d], 1) * k;
            let max_logit = logits.clone().max_dim(1);
            let lse = (logits - max_logit.clone()).exp().sum_dim(1).log() + max_logit;
            (lse - pos_score_2d.log() * k).mean()
        } else {
            let neg_loss_avg = if config.self_adversarial {
                Self::apply_self_adv(neg_loss, n_neg, config.adversarial_temperature, device)
            } else {
                neg_loss.mean_dim(1).reshape([bs, 1])
            };

            (config.margin + pos_loss - neg_loss_avg)
                .clamp_min(0.0)
                .mean()
        }
    }

    /// Self-adversarial negative weighting (stop-gradient on weights).
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
        model: &BurnEllipsoidModel<B>,
        head_ids: &Tensor<B, 1, Int>,
        rel_ids: &Tensor<B, 1, Int>,
        tail_ids: &Tensor<B, 1, Int>,
        reg: f32,
    ) -> Tensor<B, 1> {
        if reg == 0.0 {
            return Tensor::<B, 1>::zeros([1], &head_ids.device());
        }
        let hm = model.entities.mu.val().select(0, head_ids.clone());
        let tm = model.entities.mu.val().select(0, tail_ids.clone());
        let rt = model.relations.translation.val().select(0, rel_ids.clone());
        let rtt = model
            .relations
            .tail_translation
            .val()
            .select(0, rel_ids.clone());
        ((hm.powf_scalar(2.0).mean()
            + tm.powf_scalar(2.0).mean()
            + rt.powf_scalar(2.0).mean()
            + rtt.powf_scalar(2.0).mean())
            * reg)
            .reshape([1])
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Extract raw parameter vecs for CPU evaluation.
    ///
    /// Returns `(mu, head_ls, tail_ls, head_trans, tail_trans, log_scale, n_e, dim, n_r)`.
    #[allow(clippy::type_complexity)]
    fn extract_params(
        model: &BurnEllipsoidModel<B>,
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
        let dim = model.entities.mu.val().dims()[1];
        let n_e = model.entities.mu.val().dims()[0];
        let n_r = model.relations.translation.val().dims()[0];

        let mu: Vec<f32> = model.entities.mu.val().into_data().to_vec().unwrap();
        let hls: Vec<f32> = model.entities.log_sigma.val().into_data().to_vec().unwrap();
        let tls: Vec<f32> = model
            .entities
            .tail_log_sigma
            .val()
            .into_data()
            .to_vec()
            .unwrap();
        let ht: Vec<f32> = model
            .relations
            .translation
            .val()
            .into_data()
            .to_vec()
            .unwrap();
        let tt: Vec<f32> = model
            .relations
            .tail_translation
            .val()
            .into_data()
            .to_vec()
            .unwrap();
        let ls: Vec<f32> = model
            .relations
            .log_scale
            .val()
            .into_data()
            .to_vec()
            .unwrap();

        (mu, hls, tls, ht, tt, ls, n_e, dim, n_r)
    }

    /// Evaluate link prediction using CPU diagonal-KL scoring.
    pub fn evaluate(
        &self,
        model: &BurnEllipsoidModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let (mu, hls, tls, ht, tt, _ls, n_e, dim, n_r) = Self::extract_params(model);
        let k = 2.0f32;

        // Pre-slice per-entity and per-relation vecs.
        let head_mu: Vec<&[f32]> = (0..n_e).map(|i| &mu[i * dim..(i + 1) * dim]).collect();
        let head_ls: Vec<&[f32]> = (0..n_e).map(|i| &hls[i * dim..(i + 1) * dim]).collect();
        let tail_mu: Vec<&[f32]> = (0..n_e).map(|i| &mu[i * dim..(i + 1) * dim]).collect();
        let tail_ls: Vec<&[f32]> = (0..n_e).map(|i| &tls[i * dim..(i + 1) * dim]).collect();
        let head_trans: Vec<&[f32]> = (0..n_r).map(|i| &ht[i * dim..(i + 1) * dim]).collect();
        let tail_trans: Vec<&[f32]> = (0..n_r).map(|i| &tt[i * dim..(i + 1) * dim]).collect();

        let kl_diag = |mu_c: &[f32], ls_c: &[f32], mu_p: &[f32], ls_p: &[f32]| -> f32 {
            let mut kl = 0.0f32;
            for i in 0..dim {
                let var_ratio = (2.0 * (ls_c[i] - ls_p[i])).exp();
                let mahal = (mu_p[i] - mu_c[i]).powi(2) / (2.0 * ls_p[i]).exp();
                let log_det = 2.0 * (ls_p[i] - ls_c[i]);
                kl += var_ratio + mahal - 1.0 + log_det;
            }
            kl * 0.5
        };

        let score = |h: usize, r: usize, t: usize| -> f32 {
            if r >= n_r || h >= n_e || t >= n_e {
                return 0.0;
            }
            let mut trans_mu = head_mu[h].to_vec();
            let mut trans_ls = head_ls[h].to_vec();
            let mut shifted_t_mu = tail_mu[t].to_vec();
            for d in 0..dim {
                trans_mu[d] += head_trans[r][d];
                trans_ls[d] += 0.0; // log_scale not used in eval for simplicity
                shifted_t_mu[d] += tail_trans[r][d];
            }
            let kl = kl_diag(&trans_mu, &trans_ls, &shifted_t_mu, tail_ls[t]);
            let prob = 1.0 / (1.0 + (k * kl).exp()); // sigmoid(-k*kl)
            prob.clamp(0.0, 1.0)
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

// Numerically stable sigmoid via tanh.
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

    #[test]
    fn model_init_shapes() {
        let device = Default::default();
        let model = BurnEllipsoidTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        assert_eq!(model.entities.mu.val().dims(), [10, 4]);
        assert_eq!(model.entities.log_sigma.val().dims(), [10, 4]);
        assert_eq!(model.entities.tail_log_sigma.val().dims(), [10, 4]);
        assert_eq!(model.relations.translation.val().dims(), [3, 4]);
        assert_eq!(model.relations.tail_translation.val().dims(), [3, 4]);
        assert_eq!(model.relations.log_scale.val().dims(), [3, 4]);
    }

    #[test]
    fn kl_is_zero_for_identical() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let dim = 4usize;
        let mu = Tensor::<TestBackend, 2>::zeros([1, dim], &device);
        let ls = Tensor::<TestBackend, 2>::zeros([1, dim], &device);

        // KL(p || p) should be 0.
        let kl = BurnEllipsoidTrainer::<TestBackend>::kl_diagonal(mu.clone(), ls.clone(), mu, ls);
        let v = kl.into_scalar().to_f32();
        assert!(v.abs() < 1e-5, "KL(p||p) = {v}, expected ~0");
    }

    #[test]
    fn loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnEllipsoidTrainer::<TestBackend>::new();
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
        let mut trainer = BurnEllipsoidTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.02,
            margin: 1.0,
            use_infonce: true,
            sigmoid_k: 2.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnEllipsoidModel<TestBackend>>();

        let loss1 = trainer.train_epoch(&mut model, &mut optim, &triples, 0, &config, &device);
        let mut loss_last = loss1;
        for epoch in 1..20 {
            loss_last =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        assert!(
            loss_last < loss1,
            "loss should decrease: epoch 0 = {loss1:.4}, epoch 19 = {loss_last:.4}"
        );
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
        let mut trainer = BurnEllipsoidTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(4, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            use_infonce: true,
            negative_samples: 1,
            sigmoid_k: 2.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnEllipsoidModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..50 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        eprintln!("burn_ellipsoid final loss={last_loss:.4}");
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
