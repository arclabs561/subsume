//! Burn-based **relation-aware** hard-box KGE trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training
//!
//! # Task shape
//! Triple-based subsumption KGE over `(head, relation, tail)`: for a triple the
//! head **contains** the tail after each is shifted into a relation-specific
//! region. This mirrors the CPU [`crate::trainer::box_trainer::BoxEmbeddingTrainer`]
//! (the Python-facing flagship) and the relation-aware Burn ball trainer
//! [`crate::trainer::burn_ball_trainer`], so the flagship relation-aware box task
//! now trains on Burn. (An earlier version of this file was relation-free /
//! pair-based; relations subsume that case — a single zero translation recovers
//! plain containment.)
//!
//! # Scoring
//! Each entity is a box stored as a lower corner `min` and a raw width parameter
//! `raw_delta`; the width is `softplus(raw_delta)` (always positive) and
//! `max = min + width`. Each relation carries **dual** translations
//! (`translation` for the head, `tail_translation` for the tail), following the
//! Burn ball trainer's asymmetric-containment scheme:
//!
//! ```text
//! score(h, r, t) = P( transform_tail(t, r) ⊆ transform_head(h, r) )
//! transform_head(h, r) = box(min_h + translation_r,      width_h)
//! transform_tail(t, r) = box(min_t + tail_translation_r, width_t)
//! ```
//!
//! Containment reuses the CPU trainer's convention: `P(child ⊆ parent) =
//! Vol(parent ∩ child) / Vol(child)`. Intersection side lengths use softplus
//! smoothing `side_d = softplus(min(a_max,b_max) − max(a_min,b_min), beta)` with
//! `beta = config.softplus_beta`, and every volume is computed in log space to
//! avoid high-dimensional underflow. The ranking loss (InfoNCE or margin) pushes
//! the positive log-containment above the corrupted-tail negatives; evaluation
//! ranks by containment probability via
//! [`crate::trainer::evaluation::evaluate_link_prediction_generic`], so higher
//! score = better containment.

use crate::dataset::TripleIds;
use crate::trainer::negative_sampling::{
    compute_relation_entity_pools, sample_excluding, RelationEntityPools,
};
use crate::trainer::trainer_utils::self_adversarial_weights;
use crate::trainer::CpuBoxTrainingConfig;
use burn::module::{Param, ParamId};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

/// `ln(1e-8)` — lower clamp on `ln P` (matches the CPU trainer's `P.clamp(1e-8, 1)`).
const LN_MIN_PROB: f64 = -18.420_681;

// ---------------------------------------------------------------------------
// Numerically stable softplus
// ---------------------------------------------------------------------------

/// Softplus with steepness `beta`: `(1/beta)·ln(1 + exp(beta·x))`.
///
/// Stable form: `(relu(beta·x) + ln(1 + exp(−|beta·x|))) / beta`. Matches
/// [`crate::utils::softplus`] exactly so the Burn intersection volume agrees
/// with the CPU trainer's.
fn softplus_beta<B: Backend, const D: usize>(x: Tensor<B, D>, beta: f64) -> Tensor<B, D> {
    let bx = x.mul_scalar(beta);
    let stable = bx.clone().clamp_min(0.0) + bx.abs().neg().exp().add_scalar(1.0).log();
    stable.div_scalar(beta)
}

/// Numerically stable sigmoid via tanh: `σ(x) = (tanh(x/2) + 1) / 2`.
fn sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.div_scalar(2.0).tanh().add_scalar(1.0).div_scalar(2.0)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Hard-box entity parameters.
///
/// Each entity is a box stored as a lower corner `min` and a raw width parameter
/// `raw_delta`; the actual width is `softplus(raw_delta)`, so the upper corner is
/// `max = min + softplus(raw_delta)`. One box per entity (Box2EL-style); the head
/// vs tail asymmetry comes from the relation's dual translations, not separate
/// per-entity extents.
///
/// Must use `Param<Tensor<B, 2>>` — bare `Tensor` fields are treated as constants
/// by Burn's module system, so gradients would never reach the optimizer.
#[derive(Module, Debug)]
pub struct BurnBoxEntityParams<B: Backend> {
    /// Lower corners `[num_entities, dim]`.
    pub min: Param<Tensor<B, 2>>,
    /// Raw width parameter `[num_entities, dim]` (`width = softplus(raw_delta)`).
    pub raw_delta: Param<Tensor<B, 2>>,
}

/// Relation parameters: dual (head + tail) translations.
///
/// A translation shifts a box (both corners) into a relation-specific region:
/// `transform(box, trans) = (min + trans, max + trans)`, width unchanged. Head
/// and tail get separate translations so the scorer can model asymmetric
/// containment, mirroring [`crate::trainer::burn_ball_trainer`].
#[derive(Module, Debug)]
pub struct BurnBoxRelationParams<B: Backend> {
    /// Head translation per relation `[num_relations, dim]`.
    pub translation: Param<Tensor<B, 2>>,
    /// Tail translation per relation `[num_relations, dim]`.
    pub tail_translation: Param<Tensor<B, 2>>,
}

/// Combined relation-aware hard-box model.
#[derive(Module, Debug)]
pub struct BurnBoxModel<B: Backend> {
    /// Entity box parameters.
    pub entities: BurnBoxEntityParams<B>,
    /// Relation translation parameters.
    pub relations: BurnBoxRelationParams<B>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based relation-aware hard-box trainer with autodiff.
pub struct BurnBoxTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
    cached_pools: Option<HashMap<usize, RelationEntityPools>>,
}

impl<B: AutodiffBackend> Default for BurnBoxTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
            cached_pools: None,
        }
    }
}

impl<B: AutodiffBackend> BurnBoxTrainer<B> {
    /// Create a new trainer.
    pub fn new() -> Self {
        Self::default()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch_seed = epoch.wrapping_mul(7919);
    }

    /// Initialize a randomly-weighted model.
    ///
    /// Lower corners start near the origin; raw widths are initialized so that
    /// `softplus(raw_delta)` starts around 1–2, giving parents room to grow to
    /// contain their children. Relation translations start near zero.
    pub fn init_model(
        &self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnBoxModel<B> {
        let param = |shape: [usize; 2], lo: f64, hi: f64| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::random(shape, burn::tensor::Distribution::Uniform(lo, hi), device)
                    .require_grad(),
            )
        };
        let n_rel = num_relations.max(1);
        BurnBoxModel {
            entities: BurnBoxEntityParams {
                min: param([num_entities, dim], -0.1, 0.1),
                raw_delta: param([num_entities, dim], 0.5, 2.0),
            },
            relations: BurnBoxRelationParams {
                translation: param([n_rel, dim], -0.01, 0.01),
                tail_translation: param([n_rel, dim], -0.01, 0.01),
            },
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train for one epoch over `triples`, returning the mean batch loss.
    ///
    /// Negatives corrupt the tail using type-constrained per-relation pools
    /// (falling back to uniform entities). Pass the optimizer in so Adam momentum
    /// accumulates across epochs.
    pub fn train_epoch(
        &mut self,
        model: &mut BurnBoxModel<B>,
        optim: &mut impl Optimizer<BurnBoxModel<B>, B>,
        triples: &[TripleIds],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.entities.min.val().dims()[0];
        let batch_size = config.batch_size.max(1);
        let n_neg = config.negative_samples.max(1);

        let n = triples.len();
        if n == 0 || num_entities <= 1 {
            return 0.0;
        }

        if self.cached_pools.is_none() {
            let indexed: Vec<(usize, usize, usize)> = triples
                .iter()
                .map(|t| (t.head, t.relation, t.tail))
                .collect();
            self.cached_pools = Some(compute_relation_entity_pools(&indexed));
        }
        let pools = self.cached_pools.as_ref().unwrap();

        let mut rng = fastrand::Rng::with_seed(self.epoch_seed.wrapping_add(1));
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.usize(0..=i));
        }

        let mut total_loss = 0.0f32;
        let mut batch_count = 0usize;

        for chunk in order.chunks(batch_size) {
            let head_vec: Vec<i64> = chunk.iter().map(|&i| triples[i].head as i64).collect();
            let rel_vec: Vec<i64> = chunk.iter().map(|&i| triples[i].relation as i64).collect();
            let tail_vec: Vec<i64> = chunk.iter().map(|&i| triples[i].tail as i64).collect();

            // Type-constrained tail-corruption negatives.
            let mut neg_flat: Vec<i64> = Vec::with_capacity(chunk.len() * n_neg);
            for (&ri, &ti) in rel_vec.iter().zip(&tail_vec) {
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
                    neg_flat.push(neg);
                }
            }

            let current_model = model.clone();
            let loss = self.batch_loss(
                &current_model,
                &head_vec,
                &rel_vec,
                &tail_vec,
                &neg_flat,
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
                eprintln!("[burn_box] skipping non-finite batch loss: {loss_val}");
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

    /// Ranking loss (InfoNCE or margin) over the relation-transformed containment
    /// log-probability for a batch of triples plus corrupted-tail negatives.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnBoxModel<B>,
        head_vec: &[i64],
        rel_vec: &[i64],
        tail_vec: &[i64],
        neg_tail_flat: &[i64],
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = head_vec.len();
        let beta = config.softplus_beta as f64;
        let k = config.sigmoid_k as f64;

        let head_ids = Tensor::<B, 1, Int>::from_data(head_vec, device);
        let rel_ids = Tensor::<B, 1, Int>::from_data(rel_vec, device);
        let tail_ids = Tensor::<B, 1, Int>::from_data(tail_vec, device);

        // ---- Positive log-containment ln P(tail ⊆ head) ----
        let (h_min0, h_width) = gather_box(&model.entities, head_ids.clone());
        let (t_min0, t_width) = gather_box(&model.entities, tail_ids.clone());
        let h_trans = model.relations.translation.val().select(0, rel_ids.clone());
        let t_trans = model
            .relations
            .tail_translation
            .val()
            .select(0, rel_ids.clone());
        let h_min = h_min0.clone() + h_trans.clone();
        let t_min = t_min0.clone() + t_trans.clone();
        let (lvi, _lv_parent, lv_child) =
            box_logvols(h_min, h_width.clone(), t_min, t_width.clone(), beta);
        let pos_lnp = (lvi - lv_child).clamp(LN_MIN_PROB, 0.0); // [bs, 1]

        // ---- Negative log-containment for corrupted tails ----
        let head_rep: Vec<i64> = head_vec
            .iter()
            .flat_map(|&h| std::iter::repeat_n(h, n_neg))
            .collect();
        let rel_rep: Vec<i64> = rel_vec
            .iter()
            .flat_map(|&r| std::iter::repeat_n(r, n_neg))
            .collect();
        let hr_ids = Tensor::<B, 1, Int>::from_data(head_rep.as_slice(), device);
        let rr_ids = Tensor::<B, 1, Int>::from_data(rel_rep.as_slice(), device);
        let nt_ids = Tensor::<B, 1, Int>::from_data(neg_tail_flat, device);

        let (hr_min0, hr_width) = gather_box(&model.entities, hr_ids);
        let (nt_min0, nt_width) = gather_box(&model.entities, nt_ids);
        let hr_trans = model.relations.translation.val().select(0, rr_ids.clone());
        let nt_trans = model.relations.tail_translation.val().select(0, rr_ids);
        let hr_min = hr_min0 + hr_trans;
        let nt_min = nt_min0 + nt_trans;
        let (nlvi, _nlv_parent, nlv_child) = box_logvols(hr_min, hr_width, nt_min, nt_width, beta);
        let neg_lnp = (nlvi - nlv_child)
            .clamp(LN_MIN_PROB, 0.0)
            .reshape([bs, n_neg]); // [bs, n_neg]

        let ranking = if config.use_infonce {
            // InfoNCE: cross-entropy over the (1 + n_neg)-way softmax on scores.
            let logits = Tensor::cat(vec![pos_lnp.clone(), neg_lnp], 1).mul_scalar(k); // [bs, 1+n]
            let max_logit = logits.clone().max_dim(1);
            let lse = (logits - max_logit.clone()).exp().sum_dim(1).log() + max_logit;
            (lse - pos_lnp.mul_scalar(k)).mean()
        } else {
            // Margin ranking loss on the log-containment scores.
            let pos_loss = sigmoid(pos_lnp.mul_scalar(k))
                .clamp(1e-6, 1.0 - 1e-6)
                .log()
                .neg(); // [bs,1]
            let neg_loss_2d = sigmoid(neg_lnp.mul_scalar(k))
                .clamp(1e-6, 1.0 - 1e-6)
                .log()
                .neg(); // [bs, n_neg]
            let neg_avg = if config.self_adversarial && config.adversarial_temperature > 0.0 {
                Self::apply_self_adv(neg_loss_2d, n_neg, config.adversarial_temperature, device)
            } else {
                neg_loss_2d.mean_dim(1)
            }; // [bs, 1]
            (pos_loss.sub(neg_avg).add_scalar(config.margin as f64))
                .clamp_min(0.0)
                .mean()
        };

        // ---- L2 regularization on participating embeddings + translations ----
        let reg = config.regularization as f64;
        if reg == 0.0 {
            ranking
        } else {
            let reg_term = (h_min0.powf_scalar(2.0).mean()
                + t_min0.powf_scalar(2.0).mean()
                + h_trans.powf_scalar(2.0).mean()
                + t_trans.powf_scalar(2.0).mean())
            .mul_scalar(reg);
            ranking + reg_term
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
        (neg_loss * w).sum_dim(1)
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Extract `(mins, widths, head_trans, tail_trans, n_e, n_r, dim)` as flat
    /// row-major `Vec<f32>` (`width = softplus(raw_delta)`).
    #[allow(clippy::type_complexity)]
    fn extract_params(
        model: &BurnBoxModel<B>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize, usize, usize) {
        let n_e = model.entities.min.val().dims()[0];
        let dim = model.entities.min.val().dims()[1];
        let n_r = model.relations.translation.val().dims()[0];
        let mins: Vec<f32> = model
            .entities
            .min
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let raw: Vec<f32> = model
            .entities
            .raw_delta
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let widths: Vec<f32> = raw
            .into_iter()
            .map(|r| crate::utils::softplus(r, 1.0))
            .collect();
        let head_trans: Vec<f32> = model
            .relations
            .translation
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let tail_trans: Vec<f32> = model
            .relations
            .tail_translation
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        (mins, widths, head_trans, tail_trans, n_e, n_r, dim)
    }

    /// Evaluate link prediction on `test_triples` via the shared ranking harness.
    ///
    /// `score(h, r, t) = P(transform_tail(t, r) ⊆ transform_head(h, r))`, matching
    /// the training objective exactly.
    pub fn evaluate(
        &self,
        model: &BurnBoxModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let (mins, widths, head_trans, tail_trans, n_e, _n_r, dim) = Self::extract_params(model);
        let beta = 1.0f32;
        let score = |h: usize, r: usize, t: usize| -> f32 {
            box_containment_prob_rel(&mins, &widths, &head_trans, &tail_trans, h, r, t, dim, beta)
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
            rank_variance: f32::NAN,
            rank_p25: f32::NAN,
            rank_p50: f32::NAN,
            rank_p75: f32::NAN,
            rank_p95: f32::NAN,
            per_relation: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Gather `(min, width)` boxes for the given entity IDs. `width = softplus(raw_delta)`.
fn gather_box<B: Backend>(
    entities: &BurnBoxEntityParams<B>,
    ids: Tensor<B, 1, Int>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let min = entities.min.val().select(0, ids.clone());
    let raw = entities.raw_delta.val().select(0, ids);
    let width = softplus_beta(raw, 1.0);
    (min, width)
}

/// Log-space volumes for a batch of box pairs `(A = parent, B = child)`.
///
/// Returns `(log_vol_int, log_vol_A, log_vol_B)`, each `[N, 1]`, where the
/// intersection side lengths use softplus smoothing with steepness `beta`.
fn box_logvols<B: Backend>(
    min_a: Tensor<B, 2>,
    width_a: Tensor<B, 2>,
    min_b: Tensor<B, 2>,
    width_b: Tensor<B, 2>,
    beta: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let max_a = min_a.clone() + width_a.clone();
    let max_b = min_b.clone() + width_b.clone();
    let lo = min_a.max_pair(min_b);
    let hi = max_a.min_pair(max_b);
    let side = softplus_beta(hi - lo, beta);
    let log_vol_int = side.add_scalar(1e-30).log().sum_dim(1);
    let log_vol_a = width_a.add_scalar(1e-30).log().sum_dim(1);
    let log_vol_b = width_b.add_scalar(1e-30).log().sum_dim(1);
    (log_vol_int, log_vol_a, log_vol_b)
}

/// CPU relation-aware containment probability
/// `P(transform_tail(t,r) ⊆ transform_head(h,r)) = Vol(parent ∩ child) / Vol(child)`.
///
/// Mirrors [`box_logvols`] on flat parameter arrays for evaluation scoring, with
/// the head shifted by `head_trans[r]` and the tail by `tail_trans[r]`.
#[allow(clippy::too_many_arguments)]
fn box_containment_prob_rel(
    mins: &[f32],
    widths: &[f32],
    head_trans: &[f32],
    tail_trans: &[f32],
    h: usize,
    r: usize,
    t: usize,
    dim: usize,
    beta: f32,
) -> f32 {
    let ho = h * dim;
    let to = t * dim;
    let ro = r * dim;
    let mut log_vol_int = 0.0f32;
    let mut log_vol_child = 0.0f32;
    for i in 0..dim {
        let p_min = mins[ho + i] + head_trans[ro + i];
        let p_max = p_min + widths[ho + i];
        let c_min = mins[to + i] + tail_trans[ro + i];
        let c_max = c_min + widths[to + i];
        let lo = p_min.max(c_min);
        let hi = p_max.min(c_max);
        let side = crate::utils::softplus(hi - lo, beta);
        log_vol_int += (side + 1e-30).ln();
        log_vol_child += (widths[to + i] + 1e-30).ln();
    }
    (log_vol_int - log_vol_child).exp().clamp(0.0, 1.0)
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

    /// Containment hierarchy under relation 0 (hypernym): animal ⊇ {mammal, bird},
    /// mammal ⊇ {dog, cat}, plus the transitive closure animal ⊇ {dog, cat}.
    /// 0=animal 1=mammal 2=bird 3=dog 4=cat.
    fn hierarchy_triples() -> Vec<TripleIds> {
        [(0, 1), (0, 2), (1, 3), (1, 4), (0, 3), (0, 4)]
            .into_iter()
            .map(|(h, t)| TripleIds {
                head: h,
                relation: 0,
                tail: t,
            })
            .collect()
    }

    #[test]
    fn model_init_shapes() {
        let device = Default::default();
        let model = BurnBoxTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        assert_eq!(model.entities.min.val().dims(), [10, 4]);
        assert_eq!(model.entities.raw_delta.val().dims(), [10, 4]);
        assert_eq!(model.relations.translation.val().dims(), [3, 4]);
        assert_eq!(model.relations.tail_translation.val().dims(), [3, 4]);
    }

    #[test]
    fn batch_loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnBoxTrainer::<TestBackend>::new();
        let model = trainer.init_model(6, 2, 4, &device);
        let config = CpuBoxTrainingConfig::default();
        let loss = trainer.batch_loss(
            &model,
            &[0i64, 1],
            &[0i64, 1],
            &[1i64, 3],
            &[2i64, 4],
            1,
            &config,
            &device,
        );
        let v = loss.into_scalar().to_f32();
        assert!(v.is_finite(), "loss not finite: {v}");
        assert!(v >= 0.0, "loss negative: {v}");
    }

    #[test]
    fn loss_decreases_across_epochs() {
        let device = Default::default();
        let triples = hierarchy_triples();
        let mut trainer = BurnBoxTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 1, 8, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.5,
            negative_samples: 3,
            batch_size: 8,
            ..Default::default()
        };
        let mut optim = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<TestBackend, BurnBoxModel<TestBackend>>();

        let loss_0 = trainer.train_epoch(&mut model, &mut optim, &triples, 0, &config, &device);
        let mut loss_last = loss_0;
        for epoch in 1..40 {
            loss_last =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        assert!(
            loss_last < loss_0,
            "loss should decrease: epoch 0 = {loss_0:.4}, epoch 39 = {loss_last:.4}"
        );
    }

    /// Both relations' head and tail translations receive gradients.
    #[test]
    fn dual_translations_receive_gradients() {
        let device = Default::default();
        // Two relations so both translation rows are exercised.
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
        let mut trainer = BurnBoxTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.5,
            negative_samples: 2,
            ..Default::default()
        };

        let init_head = model.relations.translation.val().to_data();
        let init_tail = model.relations.tail_translation.val().to_data();

        let mut optim = AdamConfig::new().init::<TestBackend, BurnBoxModel<TestBackend>>();
        for epoch in 0..5 {
            trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }

        let final_head = model.relations.translation.val().to_data();
        let final_tail = model.relations.tail_translation.val().to_data();

        let head_changed = init_head
            .iter::<f32>()
            .zip(final_head.iter::<f32>())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        let tail_changed = init_tail
            .iter::<f32>()
            .zip(final_tail.iter::<f32>())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(
            head_changed,
            "head translation should change during training"
        );
        assert!(
            tail_changed,
            "tail translation should change during training"
        );
    }

    /// End-to-end: train on a relation-aware containment hierarchy and verify the
    /// model ranks true containment well above random under the shared filtered
    /// ranking harness (`evaluate_link_prediction_generic`).
    #[test]
    fn train_and_evaluate_synthetic() {
        let device = Default::default();
        let triples = hierarchy_triples();
        let mut trainer = BurnBoxTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 1, 8, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.3,
            negative_samples: 4,
            batch_size: 8,
            use_infonce: true,
            sigmoid_k: 2.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<TestBackend, BurnBoxModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..200 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }

        let results = trainer.evaluate(&model, &triples, None);
        eprintln!(
            "BurnBox (relation-aware) synthetic: final_loss={last_loss:.4} MRR={:.3} H@1={:.3} mean_rank={:.2}",
            results.mrr, results.hits_at_1, results.mean_rank
        );
        assert!(results.mrr > 0.4, "MRR={} expected >0.4", results.mrr);
        assert!(
            results.mean_rank <= 3.5,
            "mean_rank={} expected <=3.5",
            results.mean_rank
        );
    }

    #[test]
    fn param_ids_are_tracked_and_survive_clone() {
        use burn::module::list_param_ids;
        let device = Default::default();
        let model = BurnBoxTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        let ids = list_param_ids(&model);
        assert_eq!(
            ids.len(),
            4,
            "expected 4 params (min, raw_delta, translation, tail_translation), got {}: {:?}",
            ids.len(),
            ids
        );
        assert_eq!(ids, list_param_ids(&model.clone()));
    }
}
