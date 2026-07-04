//! Burn-based **relation-aware** octagon embedding trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training
//!
//! # Task shape
//! Triple-based subsumption KGE over `(head, relation, tail)`, entity-as-region:
//! the head octagon **contains** the tail octagon after each is shifted into a
//! relation-specific region. This is subsume's entity-as-region use of octagons
//! (mirroring [`crate::ndarray_backend::NdarrayOctagon::containment_prob`]), not
//! the relation-as-region rule-capturing model of Charpenay & Schockaert
//! (IJCAI 2024, arXiv:2401.16270), from which the octagon shape is taken.
//!
//! # Geometry
//! An octagon is an axis-aligned box plus diagonal constraints on adjacent
//! dimension pairs `(i, i+1)`: bounds on `x_i + x_{i+1}` (sum) and `x_i - x_{i+1}`
//! (diff), i.e. 45-degree corner cuts. Each entity carries:
//! - box: `min` and `width = softplus(raw_delta)` per dim (max = min + width);
//! - diagonals: sum and diff bounds per adjacent pair, stored as a center and a
//!   positive half-width (`bound = center +/- softplus(raw_hw)`).
//!
//! # Scoring (margin-based, avoids the volume degeneracy)
//! `P(child subset parent)` is a product of per-constraint sigmoids of the
//! containment margins (parent's bounds are looser than child's), so the log
//! score is a sum of `log-sigmoid(margin / tau)`. Unlike box's softplus-VOLUME
//! ratio, whose log-gradient vanishes when boxes are disjoint (fatal in high
//! dim), each margin term keeps a bounded gradient even for disjoint regions, so
//! no center-attraction bootstrap is needed. Evaluation ranks by this score via
//! [`crate::trainer::evaluation::evaluate_link_prediction_generic`].

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

/// Temperature for the margin sigmoids (smaller = sharper containment boundary).
const OCTAGON_TAU: f64 = 1.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Numerically stable softplus (beta = 1): `ln(1 + exp(x))`.
fn softplus1<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().clamp_min(0.0) + x.abs().neg().exp().add_scalar(1.0).log()
}

/// Numerically stable log-sigmoid: `log(sigmoid(x)) = -softplus(-x)`.
fn log_sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    softplus1(x.neg()).neg()
}

/// Numerically stable sigmoid via tanh.
fn sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.div_scalar(2.0).tanh().add_scalar(1.0).div_scalar(2.0)
}

/// Adjacent-pair sums `t[.., i] + t[.., i+1]`, shape `[bs, dim-1]`.
fn pair_sum<B: Backend>(t: Tensor<B, 2>) -> Tensor<B, 2> {
    let [bs, dim] = t.dims();
    t.clone().slice([0..bs, 0..dim - 1]) + t.slice([0..bs, 1..dim])
}

/// Adjacent-pair diffs `t[.., i] - t[.., i+1]`, shape `[bs, dim-1]`.
fn pair_diff<B: Backend>(t: Tensor<B, 2>) -> Tensor<B, 2> {
    let [bs, dim] = t.dims();
    t.clone().slice([0..bs, 0..dim - 1]) - t.slice([0..bs, 1..dim])
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Octagon entity parameters: box bounds + diagonal (sum/diff) bounds.
#[derive(Module, Debug)]
pub struct BurnOctagonEntityParams<B: Backend> {
    /// Box lower corners `[num_entities, dim]`.
    pub min: Param<Tensor<B, 2>>,
    /// Raw box width `[num_entities, dim]` (`width = softplus(raw_delta)`).
    pub raw_delta: Param<Tensor<B, 2>>,
    /// Sum-constraint centers `[num_entities, dim-1]` (bounds on `x_i + x_{i+1}`).
    pub sum_center: Param<Tensor<B, 2>>,
    /// Raw sum half-width `[num_entities, dim-1]` (`hw = softplus(raw)`).
    pub sum_raw_hw: Param<Tensor<B, 2>>,
    /// Diff-constraint centers `[num_entities, dim-1]` (bounds on `x_i - x_{i+1}`).
    pub diff_center: Param<Tensor<B, 2>>,
    /// Raw diff half-width `[num_entities, dim-1]`.
    pub diff_raw_hw: Param<Tensor<B, 2>>,
}

/// Relation parameters: dual (head + tail) translations. A translation shifts the
/// box corners and, consistently, the diagonal centers by the induced pair
/// sums/diffs of the translation vector.
#[derive(Module, Debug)]
pub struct BurnOctagonRelationParams<B: Backend> {
    /// Head translation per relation `[num_relations, dim]`.
    pub translation: Param<Tensor<B, 2>>,
    /// Tail translation per relation `[num_relations, dim]`.
    pub tail_translation: Param<Tensor<B, 2>>,
}

/// Combined relation-aware octagon model.
#[derive(Module, Debug)]
pub struct BurnOctagonModel<B: Backend> {
    /// Entity octagon parameters.
    pub entities: BurnOctagonEntityParams<B>,
    /// Relation translation parameters.
    pub relations: BurnOctagonRelationParams<B>,
}

/// A transformed octagon (relation-shifted), as tensors for one batch.
struct Octagon<B: Backend> {
    min: Tensor<B, 2>,        // [N, dim]
    width: Tensor<B, 2>,      // [N, dim]
    sum_center: Tensor<B, 2>, // [N, dim-1]
    sum_hw: Tensor<B, 2>,     // [N, dim-1]
    diff_center: Tensor<B, 2>,
    diff_hw: Tensor<B, 2>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based relation-aware octagon trainer with autodiff.
pub struct BurnOctagonTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
    cached_pools: Option<HashMap<usize, RelationEntityPools>>,
}

impl<B: AutodiffBackend> Default for BurnOctagonTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
            cached_pools: None,
        }
    }
}

impl<B: AutodiffBackend> BurnOctagonTrainer<B> {
    /// Create a new trainer.
    pub fn new() -> Self {
        Self::default()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch_seed = epoch.wrapping_mul(7919);
    }

    /// Initialize a randomly-weighted model. Diagonal half-widths start wide so
    /// the octagon begins close to its bounding box (non-cutting diagonals),
    /// tightening during training.
    pub fn init_model(
        &self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnOctagonModel<B> {
        let param = |shape: [usize; 2], lo: f64, hi: f64| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::random(shape, burn::tensor::Distribution::Uniform(lo, hi), device)
                    .require_grad(),
            )
        };
        let n_rel = num_relations.max(1);
        let pairs = dim.saturating_sub(1).max(1);
        BurnOctagonModel {
            entities: BurnOctagonEntityParams {
                min: param([num_entities, dim], -0.1, 0.1),
                raw_delta: param([num_entities, dim], 0.5, 2.0),
                sum_center: param([num_entities, pairs], -0.1, 0.1),
                sum_raw_hw: param([num_entities, pairs], 1.5, 2.5),
                diff_center: param([num_entities, pairs], -0.1, 0.1),
                diff_raw_hw: param([num_entities, pairs], 1.5, 2.5),
            },
            relations: BurnOctagonRelationParams {
                translation: param([n_rel, dim], -0.01, 0.01),
                tail_translation: param([n_rel, dim], -0.01, 0.01),
            },
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train for one epoch over `triples`, returning the mean batch loss.
    pub fn train_epoch(
        &mut self,
        model: &mut BurnOctagonModel<B>,
        optim: &mut impl Optimizer<BurnOctagonModel<B>, B>,
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
                eprintln!("[burn_octagon] skipping non-finite batch loss: {loss_val}");
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

    /// Gather a relation-transformed octagon for the given entity + relation ids.
    /// `head=true` uses the head translation, `head=false` the tail translation.
    fn gather_transformed(
        model: &BurnOctagonModel<B>,
        ent_ids: Tensor<B, 1, Int>,
        rel_ids: Tensor<B, 1, Int>,
        head: bool,
    ) -> Octagon<B> {
        let e = &model.entities;
        let min = e.min.val().select(0, ent_ids.clone());
        let width = softplus1(e.raw_delta.val().select(0, ent_ids.clone()));
        let sum_center = e.sum_center.val().select(0, ent_ids.clone());
        let sum_hw = softplus1(e.sum_raw_hw.val().select(0, ent_ids.clone()));
        let diff_center = e.diff_center.val().select(0, ent_ids.clone());
        let diff_hw = softplus1(e.diff_raw_hw.val().select(0, ent_ids));

        let trans = if head {
            model.relations.translation.val().select(0, rel_ids)
        } else {
            model.relations.tail_translation.val().select(0, rel_ids)
        };
        Octagon {
            min: min + trans.clone(),
            width,
            sum_center: sum_center + pair_sum(trans.clone()),
            sum_hw,
            diff_center: diff_center + pair_diff(trans),
            diff_hw,
        }
    }

    /// Margin-based log-containment score `log P(child subset parent)`, shape `[N, 1]`.
    fn logscore(parent: &Octagon<B>, child: &Octagon<B>, tau: f64) -> Tensor<B, 2> {
        let inv = 1.0 / tau;
        // Box: parent contains child => parent.min <= child.min, child.max <= parent.max.
        let p_max = parent.min.clone() + parent.width.clone();
        let c_max = child.min.clone() + child.width.clone();
        let box_lo = (child.min.clone() - parent.min.clone()).mul_scalar(inv);
        let box_hi = (p_max - c_max).mul_scalar(inv);
        let box_score = log_sigmoid(box_lo).sum_dim(1) + log_sigmoid(box_hi).sum_dim(1);

        // Diagonals: parent bounds looser => parent.lo <= child.lo, child.hi <= parent.hi.
        let p_sum_lo = parent.sum_center.clone() - parent.sum_hw.clone();
        let p_sum_hi = parent.sum_center.clone() + parent.sum_hw.clone();
        let c_sum_lo = child.sum_center.clone() - child.sum_hw.clone();
        let c_sum_hi = child.sum_center.clone() + child.sum_hw.clone();
        let p_diff_lo = parent.diff_center.clone() - parent.diff_hw.clone();
        let p_diff_hi = parent.diff_center.clone() + parent.diff_hw.clone();
        let c_diff_lo = child.diff_center.clone() - child.diff_hw.clone();
        let c_diff_hi = child.diff_center.clone() + child.diff_hw.clone();

        let ds_lo = (c_sum_lo - p_sum_lo).mul_scalar(inv);
        let ds_hi = (p_sum_hi - c_sum_hi).mul_scalar(inv);
        let dd_lo = (c_diff_lo - p_diff_lo).mul_scalar(inv);
        let dd_hi = (p_diff_hi - c_diff_hi).mul_scalar(inv);
        let diag_score =
            (log_sigmoid(ds_lo) + log_sigmoid(ds_hi) + log_sigmoid(dd_lo) + log_sigmoid(dd_hi))
                .sum_dim(1);

        box_score + diag_score
    }

    /// Ranking loss (InfoNCE or margin) over the containment score for a batch of
    /// triples plus corrupted-tail negatives.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnOctagonModel<B>,
        head_vec: &[i64],
        rel_vec: &[i64],
        tail_vec: &[i64],
        neg_tail_flat: &[i64],
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = head_vec.len();
        let k = config.sigmoid_k as f64;

        let head_ids = Tensor::<B, 1, Int>::from_data(head_vec, device);
        let rel_ids = Tensor::<B, 1, Int>::from_data(rel_vec, device);
        let tail_ids = Tensor::<B, 1, Int>::from_data(tail_vec, device);

        // Positive: parent = transformed head, child = transformed tail.
        let parent = Self::gather_transformed(model, head_ids.clone(), rel_ids.clone(), true);
        let child = Self::gather_transformed(model, tail_ids, rel_ids.clone(), false);
        let pos_score = Self::logscore(&parent, &child, OCTAGON_TAU); // [bs, 1]

        // Negatives: corrupted tails, same head/relation repeated.
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

        let parent_rep = Self::gather_transformed(model, hr_ids, rr_ids.clone(), true);
        let neg_child = Self::gather_transformed(model, nt_ids, rr_ids, false);
        let neg_score = Self::logscore(&parent_rep, &neg_child, OCTAGON_TAU).reshape([bs, n_neg]);

        if config.use_infonce {
            let logits = Tensor::cat(vec![pos_score.clone(), neg_score], 1).mul_scalar(k); // [bs, 1+n]
            let max_logit = logits.clone().max_dim(1);
            let lse = (logits - max_logit.clone()).exp().sum_dim(1).log() + max_logit;
            (lse - pos_score.mul_scalar(k)).mean()
        } else {
            let pos_loss = sigmoid(pos_score.mul_scalar(k))
                .clamp(1e-6, 1.0 - 1e-6)
                .log()
                .neg();
            let neg_loss_2d = sigmoid(neg_score.mul_scalar(k))
                .clamp(1e-6, 1.0 - 1e-6)
                .log()
                .neg();
            let neg_avg = if config.self_adversarial && config.adversarial_temperature > 0.0 {
                Self::apply_self_adv(neg_loss_2d, n_neg, config.adversarial_temperature, device)
            } else {
                neg_loss_2d.mean_dim(1)
            };
            (pos_loss.sub(neg_avg).add_scalar(config.margin as f64))
                .clamp_min(0.0)
                .mean()
        }
    }

    fn apply_self_adv(
        neg_loss: Tensor<B, 2>,
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

    #[allow(clippy::type_complexity)]
    fn extract(model: &BurnOctagonModel<B>) -> OctagonParams {
        let e = &model.entities;
        let to_vec =
            |p: &Param<Tensor<B, 2>>| -> Vec<f32> { p.val().into_data().to_vec::<f32>().unwrap() };
        let sp = |raw: Vec<f32>| -> Vec<f32> {
            raw.into_iter()
                .map(|r| crate::utils::softplus(r, 1.0))
                .collect()
        };
        OctagonParams {
            n_e: e.min.val().dims()[0],
            dim: e.min.val().dims()[1],
            n_r: model.relations.translation.val().dims()[0],
            pairs: e.sum_center.val().dims()[1],
            min: to_vec(&e.min),
            width: sp(to_vec(&e.raw_delta)),
            sum_center: to_vec(&e.sum_center),
            sum_hw: sp(to_vec(&e.sum_raw_hw)),
            diff_center: to_vec(&e.diff_center),
            diff_hw: sp(to_vec(&e.diff_raw_hw)),
            head_trans: to_vec(&model.relations.translation),
            tail_trans: to_vec(&model.relations.tail_translation),
        }
    }

    /// Evaluate link prediction via the shared ranking harness.
    pub fn evaluate(
        &self,
        model: &BurnOctagonModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let p = Self::extract(model);
        let score = |h: usize, r: usize, t: usize| -> f32 { p.logscore_cpu(h, r, t) };
        crate::trainer::evaluation::evaluate_link_prediction_generic(
            test_triples,
            p.n_e,
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

/// Flat CPU-side octagon parameters for evaluation scoring.
struct OctagonParams {
    n_e: usize,
    dim: usize,
    n_r: usize,
    pairs: usize,
    min: Vec<f32>,
    width: Vec<f32>,
    sum_center: Vec<f32>,
    sum_hw: Vec<f32>,
    diff_center: Vec<f32>,
    diff_hw: Vec<f32>,
    head_trans: Vec<f32>,
    tail_trans: Vec<f32>,
}

impl OctagonParams {
    #[allow(clippy::needless_range_loop)]
    fn logscore_cpu(&self, h: usize, r: usize, t: usize) -> f32 {
        let _ = self.n_r;
        let dim = self.dim;
        let pairs = self.pairs;
        let tau = OCTAGON_TAU as f32;
        let logsig = |x: f32| -> f32 { -crate::utils::softplus(-x, 1.0) };
        let ho = h * dim;
        let to = t * dim;
        let hro = r * dim;
        // Box margins (parent = head, child = tail), with relation translations.
        let mut score = 0.0f32;
        for i in 0..dim {
            let p_min = self.min[ho + i] + self.head_trans[hro + i];
            let p_max = p_min + self.width[ho + i];
            let c_min = self.min[to + i] + self.tail_trans[hro + i];
            let c_max = c_min + self.width[to + i];
            score += logsig((c_min - p_min) / tau);
            score += logsig((p_max - c_max) / tau);
        }
        // Diagonal margins per adjacent pair.
        let hp = h * pairs;
        let tp = t * pairs;
        let rp = r * dim;
        for j in 0..pairs {
            // Induced translation shifts on the pair (i=j, i+1=j+1).
            let hsum = self.head_trans[rp + j] + self.head_trans[rp + j + 1];
            let hdiff = self.head_trans[rp + j] - self.head_trans[rp + j + 1];
            let tsum = self.tail_trans[rp + j] + self.tail_trans[rp + j + 1];
            let tdiff = self.tail_trans[rp + j] - self.tail_trans[rp + j + 1];

            let p_sum_lo = (self.sum_center[hp + j] + hsum) - self.sum_hw[hp + j];
            let p_sum_hi = (self.sum_center[hp + j] + hsum) + self.sum_hw[hp + j];
            let c_sum_lo = (self.sum_center[tp + j] + tsum) - self.sum_hw[tp + j];
            let c_sum_hi = (self.sum_center[tp + j] + tsum) + self.sum_hw[tp + j];
            let p_diff_lo = (self.diff_center[hp + j] + hdiff) - self.diff_hw[hp + j];
            let p_diff_hi = (self.diff_center[hp + j] + hdiff) + self.diff_hw[hp + j];
            let c_diff_lo = (self.diff_center[tp + j] + tdiff) - self.diff_hw[tp + j];
            let c_diff_hi = (self.diff_center[tp + j] + tdiff) + self.diff_hw[tp + j];

            score += logsig((c_sum_lo - p_sum_lo) / tau);
            score += logsig((p_sum_hi - c_sum_hi) / tau);
            score += logsig((c_diff_lo - p_diff_lo) / tau);
            score += logsig((p_diff_hi - c_diff_hi) / tau);
        }
        score
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
        let model = BurnOctagonTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        assert_eq!(model.entities.min.val().dims(), [10, 4]);
        assert_eq!(model.entities.sum_center.val().dims(), [10, 3]);
        assert_eq!(model.entities.diff_raw_hw.val().dims(), [10, 3]);
        assert_eq!(model.relations.translation.val().dims(), [3, 4]);
    }

    #[test]
    fn batch_loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnOctagonTrainer::<TestBackend>::new();
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
    fn dual_translations_receive_gradients() {
        let device = Default::default();
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
        let mut trainer = BurnOctagonTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.5,
            negative_samples: 2,
            use_infonce: true,
            sigmoid_k: 2.0,
            ..Default::default()
        };
        let init_head = model.relations.translation.val().to_data();
        let init_tail = model.relations.tail_translation.val().to_data();
        let mut optim = AdamConfig::new().init::<TestBackend, BurnOctagonModel<TestBackend>>();
        for epoch in 0..20 {
            trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        let head_changed = init_head
            .iter::<f32>()
            .zip(model.relations.translation.val().to_data().iter::<f32>())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        let tail_changed = init_tail
            .iter::<f32>()
            .zip(
                model
                    .relations
                    .tail_translation
                    .val()
                    .to_data()
                    .iter::<f32>(),
            )
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(head_changed, "head translation should change");
        assert!(tail_changed, "tail translation should change");
    }

    #[test]
    fn train_and_evaluate_synthetic() {
        let device = Default::default();
        let triples = hierarchy_triples();
        let mut trainer = BurnOctagonTrainer::<TestBackend>::new();
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
            .init::<TestBackend, BurnOctagonModel<TestBackend>>();
        let mut last_loss = f32::MAX;
        for epoch in 0..200 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        let results = trainer.evaluate(&model, &triples, None);
        eprintln!(
            "BurnOctagon synthetic: final_loss={last_loss:.4} MRR={:.3} H@1={:.3} mean_rank={:.2}",
            results.mrr, results.hits_at_1, results.mean_rank
        );
        assert!(results.mrr > 0.4, "MRR={} expected >0.4", results.mrr);
        assert!(
            results.mean_rank <= 3.5,
            "mean_rank={} expected <=3.5",
            results.mean_rank
        );
    }
}
