//! Burn-based **relation-aware** cone (ConE) KGE trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training
//!
//! # Task shape
//! Triple-based subsumption KGE over `(head, relation, tail)`: the head cone
//! contains the tail after each axis is shifted into a relation-specific region.
//! Mirrors the CPU [`crate::trainer::cone_trainer::ConeEmbeddingTrainer`] (the
//! Python-facing flagship) and the relation-aware Burn ball trainer
//! [`crate::trainer::burn_ball_trainer`]. (An earlier version of this file was
//! relation-free / pair-based; relations subsume that case.)
//!
//! # Scoring convention (ported from [`crate::ndarray_backend::NdarrayCone::cone_distance`])
//! Each entity is a Cartesian product of `d` angular sectors: a per-dimension
//! `axis` angle and an `aperture = pi·sigmoid(raw_aperture) ∈ (0, pi)`. Each
//! relation carries **dual** translations (`translation` for the head axis,
//! `tail_translation` for the tail axis); every axis use is through `|sin(·/2)|`,
//! which is `2pi`-periodic, so wrapping is implicit.
//!
//! Per-dimension ConE distance (query = transformed head, entity = transformed
//! tail; entity apertures are unused — the child is treated as a point at its axis):
//! - `distance_to_axis = |sin((child_axis − parent_axis)/2)|`
//! - `distance_base     = |sin(parent_aperture/2)|`
//! - inside (`distance_to_axis < distance_base`): contribute `cen · distance_to_axis`
//! - outside: contribute `min(|sin((child_axis − (parent_axis − parent_aper))/2)|,
//!   |sin((child_axis − (parent_axis + parent_aper))/2)|)`
//! - `distance = Σ_d contribution_d`, `cen = 0.02` (ConE default).
//!
//! where `parent_axis = axis_h + translation_r` and `child_axis = axis_t +
//! tail_translation_r`.
//!
//! Loss:
//! - **Positive**: `distance + reg·(mean_aper_head + mean_aper_tail)`.
//! - **Negative**: `w_neg · max(0, margin − distance)^2` (penalize a corrupted
//!   tail whose distance falls below `margin`, i.e. spurious containment).
//!
//! Evaluation ranks by `−distance` (higher = better containment) via
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
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Inside-distance weight for cone containment scoring (ConE default, matches
/// `cone_trainer::CONE_CENTER_WEIGHT`).
const CONE_CENTER_WEIGHT: f64 = 0.02;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Cone entity parameters.
///
/// Each entity has a per-dimension `axis` angle (unconstrained; used only through
/// `|sin(·/2)|`) and a raw aperture parameter; the actual aperture is
/// `pi·sigmoid(raw_aperture) ∈ (0, pi)`.
#[derive(Module, Debug)]
pub struct BurnConeEntityParams<B: Backend> {
    /// Per-dimension axis angles `[num_entities, dim]`.
    pub axes: Param<Tensor<B, 2>>,
    /// Raw aperture parameter `[num_entities, dim]` (`aperture = pi·sigmoid(raw)`).
    pub raw_aperture: Param<Tensor<B, 2>>,
}

/// Relation parameters: dual (head + tail) axis translations.
///
/// A translation rotates a cone's axis into a relation-specific region; the
/// aperture is unchanged. Head and tail get separate translations so the scorer
/// can model asymmetric containment, mirroring [`crate::trainer::burn_ball_trainer`].
#[derive(Module, Debug)]
pub struct BurnConeRelationParams<B: Backend> {
    /// Head axis translation per relation `[num_relations, dim]`.
    pub translation: Param<Tensor<B, 2>>,
    /// Tail axis translation per relation `[num_relations, dim]`.
    pub tail_translation: Param<Tensor<B, 2>>,
}

/// Combined relation-aware cone model.
#[derive(Module, Debug)]
pub struct BurnConeModel<B: Backend> {
    /// Entity cone parameters.
    pub entities: BurnConeEntityParams<B>,
    /// Relation translation parameters.
    pub relations: BurnConeRelationParams<B>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based relation-aware cone trainer with autodiff.
pub struct BurnConeTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
    cached_pools: Option<HashMap<usize, RelationEntityPools>>,
}

impl<B: AutodiffBackend> Default for BurnConeTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
            cached_pools: None,
        }
    }
}

impl<B: AutodiffBackend> BurnConeTrainer<B> {
    /// Create a new trainer.
    pub fn new() -> Self {
        Self::default()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch_seed = epoch.wrapping_mul(7919);
    }

    /// Initialize a randomly-weighted model.
    ///
    /// Axes start near zero; raw apertures start near zero so that
    /// `pi·sigmoid(raw) ≈ pi/2` (matching the CPU cone trainer's default aperture).
    /// Relation translations start near zero.
    pub fn init_model(
        &self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnConeModel<B> {
        let param = |shape: [usize; 2], lo: f64, hi: f64| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::random(shape, burn::tensor::Distribution::Uniform(lo, hi), device)
                    .require_grad(),
            )
        };
        let n_rel = num_relations.max(1);
        BurnConeModel {
            entities: BurnConeEntityParams {
                axes: param([num_entities, dim], -0.3, 0.3),
                raw_aperture: param([num_entities, dim], -0.3, 0.3),
            },
            relations: BurnConeRelationParams {
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
        model: &mut BurnConeModel<B>,
        optim: &mut impl Optimizer<BurnConeModel<B>, B>,
        triples: &[TripleIds],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.entities.axes.val().dims()[0];
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
                eprintln!("[burn_cone] skipping non-finite batch loss: {loss_val}");
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

    /// ConE containment loss for a batch of triples plus corrupted-tail negatives,
    /// with per-relation axis translations applied to head and tail.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnConeModel<B>,
        head_vec: &[i64],
        rel_vec: &[i64],
        tail_vec: &[i64],
        neg_tail_flat: &[i64],
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = head_vec.len();

        let head_ids = Tensor::<B, 1, Int>::from_data(head_vec, device);
        let rel_ids = Tensor::<B, 1, Int>::from_data(rel_vec, device);
        let tail_ids = Tensor::<B, 1, Int>::from_data(tail_vec, device);

        // ---- Positive: distance(transform_head, transform_tail) + reg ----
        let (h_axes0, h_aper) = gather_cone(&model.entities, head_ids);
        let (t_axes0, t_aper) = gather_cone(&model.entities, tail_ids);
        let h_trans = model.relations.translation.val().select(0, rel_ids.clone());
        let t_trans = model
            .relations
            .tail_translation
            .val()
            .select(0, rel_ids.clone());
        let h_axes = h_axes0 + h_trans;
        let t_axes = t_axes0 + t_trans;
        let dist = cone_distance_batched(h_axes, h_aper.clone(), t_axes, CONE_CENTER_WEIGHT); // [bs, 1]

        let mean_aper_head = h_aper.mean_dim(1); // [bs, 1]
        let mean_aper_tail = t_aper.mean_dim(1); // [bs, 1]
        let reg = (mean_aper_head + mean_aper_tail).mul_scalar(config.regularization as f64);
        let pos_loss = (dist + reg).clamp_min(0.0); // [bs, 1]
        let pos_mean = pos_loss.mean();

        // ---- Negative: w_neg · max(0, margin − distance)^2 for corrupted tails ----
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

        let (hr_axes0, hr_aper) = gather_cone(&model.entities, hr_ids);
        let (nt_axes0, _nt_aper) = gather_cone(&model.entities, nt_ids);
        let hr_trans = model.relations.translation.val().select(0, rr_ids.clone());
        let nt_trans = model.relations.tail_translation.val().select(0, rr_ids);
        let hr_axes = hr_axes0 + hr_trans;
        let nt_axes = nt_axes0 + nt_trans;
        let dist_neg = cone_distance_batched(hr_axes, hr_aper, nt_axes, CONE_CENTER_WEIGHT); // [bs*n_neg, 1]

        let margin_loss = (dist_neg.clone().neg().add_scalar(config.margin as f64))
            .clamp_min(0.0)
            .powf_scalar(2.0)
            .mul_scalar(config.negative_weight as f64); // [bs*n_neg, 1]
        let neg_2d = margin_loss.reshape([bs, n_neg]);

        let neg_mean = if config.self_adversarial && config.adversarial_temperature > 0.0 {
            // Harder negatives are CLOSER (smaller distance); score = −distance.
            let scores = dist_neg.neg().reshape([bs, n_neg]).into_data();
            let slice = scores.as_slice::<f32>().expect("neg score f32");
            let mut weights: Vec<f32> = Vec::with_capacity(slice.len());
            for row in slice.chunks(n_neg) {
                weights.extend(self_adversarial_weights(
                    row,
                    config.adversarial_temperature,
                ));
            }
            let w = Tensor::<B, 1>::from_data(weights.as_slice(), device).reshape([bs, n_neg]);
            (neg_2d * w).sum_dim(1).mean()
        } else {
            neg_2d.mean_dim(1).mean()
        };

        pos_mean + neg_mean
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Extract `(axes, apertures, head_trans, tail_trans, n_e, n_r, dim)` as flat
    /// row-major `Vec<f32>` (`aperture = pi·sigmoid(raw_aperture)`).
    #[allow(clippy::type_complexity)]
    fn extract_params(
        model: &BurnConeModel<B>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize, usize, usize) {
        let n_e = model.entities.axes.val().dims()[0];
        let dim = model.entities.axes.val().dims()[1];
        let n_r = model.relations.translation.val().dims()[0];
        let axes: Vec<f32> = model
            .entities
            .axes
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let raw: Vec<f32> = model
            .entities
            .raw_aperture
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let apers: Vec<f32> = raw
            .into_iter()
            .map(|r| PI * crate::utils::stable_sigmoid(r))
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
        (axes, apers, head_trans, tail_trans, n_e, n_r, dim)
    }

    /// Evaluate link prediction on `test_triples` via the shared ranking harness.
    ///
    /// Scores by `−cone_distance(transform_head(h,r), transform_tail(t,r))` (higher
    /// = better containment), matching the training objective.
    pub fn evaluate(
        &self,
        model: &BurnConeModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let (axes, apers, head_trans, tail_trans, n_e, _n_r, dim) = Self::extract_params(model);
        let cen = CONE_CENTER_WEIGHT as f32;
        let score = |h: usize, r: usize, t: usize| -> f32 {
            -cone_distance_cpu_rel(&axes, &apers, &head_trans, &tail_trans, h, r, t, dim, cen)
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

/// Gather `(axes, apertures)` for the given entity IDs. `aperture = pi·sigmoid(raw)`.
fn gather_cone<B: Backend>(
    entities: &BurnConeEntityParams<B>,
    ids: Tensor<B, 1, Int>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let axes = entities.axes.val().select(0, ids.clone());
    let raw = entities.raw_aperture.val().select(0, ids);
    let aper = sigmoid(raw).mul_scalar(PI as f64);
    (axes, aper)
}

/// Batched ConE distance (query = parent, entity = child). Returns `[N, 1]`.
///
/// The inside/outside split is expressed as a (non-differentiable) mask; gradient
/// flows through the selected branch, which is the correct piecewise subgradient.
fn cone_distance_batched<B: Backend>(
    q_axes: Tensor<B, 2>,
    q_aper: Tensor<B, 2>,
    e_axes: Tensor<B, 2>,
    cen: f64,
) -> Tensor<B, 2> {
    let d_to_axis = (e_axes.clone() - q_axes.clone())
        .mul_scalar(0.5)
        .sin()
        .abs(); // [N, d]
    let d_base = q_aper.clone().mul_scalar(0.5).sin().abs(); // [N, d]
    let inside = d_to_axis.clone().lower(d_base).float(); // 1.0 inside, 0.0 outside

    // Outside distance: nearest of the two sector boundaries.
    let lower_edge = q_axes.clone() - q_aper.clone();
    let upper_edge = q_axes + q_aper;
    let d1 = (e_axes.clone() - lower_edge).mul_scalar(0.5).sin().abs();
    let d2 = (e_axes - upper_edge).mul_scalar(0.5).sin().abs();
    let d_out = d1.min_pair(d2); // [N, d]

    let inside_term = d_to_axis.mul_scalar(cen) * inside.clone();
    let outside_term = d_out * inside.neg().add_scalar(1.0); // × (1 − inside)
    (inside_term + outside_term).sum_dim(1) // [N, 1]
}

/// CPU relation-aware ConE distance mirroring [`cone_distance_batched`], with the
/// head axis shifted by `head_trans[r]` and the child axis by `tail_trans[r]`.
#[allow(clippy::too_many_arguments)]
fn cone_distance_cpu_rel(
    axes: &[f32],
    apers: &[f32],
    head_trans: &[f32],
    tail_trans: &[f32],
    h: usize,
    r: usize,
    t: usize,
    dim: usize,
    cen: f32,
) -> f32 {
    let ho = h * dim;
    let to = t * dim;
    let ro = r * dim;
    let mut dist_out = 0.0f32;
    let mut dist_in = 0.0f32;
    for i in 0..dim {
        let q_axis = axes[ho + i] + head_trans[ro + i];
        let q_aper = apers[ho + i];
        let e = axes[to + i] + tail_trans[ro + i];
        let d_to_axis = ((e - q_axis) / 2.0).sin().abs();
        let d_base = (q_aper / 2.0).sin().abs();
        if d_to_axis < d_base {
            dist_in += d_to_axis;
        } else {
            let d1 = ((e - (q_axis - q_aper)) / 2.0).sin().abs();
            let d2 = ((e - (q_axis + q_aper)) / 2.0).sin().abs();
            dist_out += d1.min(d2);
        }
    }
    dist_out + cen * dist_in
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

    /// Containment hierarchy under relation 0: animal ⊇ {mammal, bird},
    /// mammal ⊇ {dog, cat}, plus the transitive closure animal ⊇ {dog, cat}.
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
        let model = BurnConeTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        assert_eq!(model.entities.axes.val().dims(), [10, 4]);
        assert_eq!(model.entities.raw_aperture.val().dims(), [10, 4]);
        assert_eq!(model.relations.translation.val().dims(), [3, 4]);
        assert_eq!(model.relations.tail_translation.val().dims(), [3, 4]);
    }

    #[test]
    fn batch_loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnConeTrainer::<TestBackend>::new();
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
        let mut trainer = BurnConeTrainer::<TestBackend>::new();
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
            .init::<TestBackend, BurnConeModel<TestBackend>>();

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
        let mut trainer = BurnConeTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.5,
            negative_samples: 2,
            ..Default::default()
        };

        let init_head = model.relations.translation.val().to_data();
        let init_tail = model.relations.tail_translation.val().to_data();

        let mut optim = AdamConfig::new().init::<TestBackend, BurnConeModel<TestBackend>>();
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
    /// model ranks true containment above random under the shared filtered ranking
    /// harness. Cone is the hardest region type, so the bar is lower than box.
    #[test]
    fn train_and_evaluate_synthetic() {
        let device = Default::default();
        let triples = hierarchy_triples();
        let mut trainer = BurnConeTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 1, 8, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.5,
            negative_samples: 4,
            batch_size: 8,
            ..Default::default()
        };
        let mut optim = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<TestBackend, BurnConeModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..300 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }

        let results = trainer.evaluate(&model, &triples, None);
        eprintln!(
            "BurnCone (relation-aware) synthetic: final_loss={last_loss:.4} MRR={:.3} H@1={:.3} mean_rank={:.2}",
            results.mrr, results.hits_at_1, results.mean_rank
        );
        assert!(results.mrr > 0.3, "MRR={} expected >0.3", results.mrr);
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
        let model = BurnConeTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        let ids = list_param_ids(&model);
        assert_eq!(
            ids.len(),
            4,
            "expected 4 params (axes, raw_aperture, translation, tail_translation), got {}: {:?}",
            ids.len(),
            ids
        );
        assert_eq!(ids, list_param_ids(&model.clone()));
    }
}
