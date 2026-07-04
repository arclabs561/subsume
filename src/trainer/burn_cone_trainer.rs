//! Burn-based cone (ConE) trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training
//!
//! # Task shape
//! Pair-based subsumption over cones (Zhang & Wang, ConE, NeurIPS 2021). Each
//! entity is a Cartesian product of `d` angular sectors: a per-dimension `axis`
//! angle and an `aperture` (half-width) in `[0, pi]`. `parent ⊇ child` when the
//! parent cone contains the child, measured by the ConE distance (lower = better).
//!
//! # Scoring convention (ported from [`crate::ndarray_backend::NdarrayCone::cone_distance`]
//! and [`crate::trainer::cone_trainer::compute_cone_pair_loss`])
//! Apertures are reparameterized as `aperture = pi·sigmoid(raw_aperture)`, keeping
//! them in `(0, pi)` with unrestricted gradient flow. Axes are unconstrained: every
//! use is through `|sin(·/2)|`, which is `2pi`-periodic, so wrapping is implicit.
//!
//! Per-dimension ConE distance (query = parent, entity = child; entity apertures
//! are unused, matching the reference):
//! - `distance_to_axis = |sin((child_axis − parent_axis)/2)|`
//! - `distance_base     = |sin(parent_aperture/2)|`
//! - inside (`distance_to_axis < distance_base`): contribute `cen · distance_to_axis`
//! - outside: contribute `min(|sin((child_axis − (parent_axis − parent_aper))/2)|,
//!   |sin((child_axis − (parent_axis + parent_aper))/2)|)`
//! - `distance = Σ_d contribution_d`, with `cen = 0.02` (ConE default).
//!
//! Losses:
//! - **Positive**: `distance + reg·(mean_aper_parent + mean_aper_child)`.
//! - **Negative**: `w_neg · max(0, margin − distance)^2` (penalize a negative pair
//!   whose distance falls below `margin`, i.e. spurious containment).
//!
//! Evaluation ranks by `−distance` so higher score = better containment.

use crate::trainer::trainer_utils::self_adversarial_weights;
use crate::trainer::CpuBoxTrainingConfig;
use burn::module::{Param, ParamId};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use std::f32::consts::PI;

/// Inside-distance weight for cone containment scoring (ConE default, matches
/// `cone_trainer::CONE_CENTER_WEIGHT`).
const CONE_CENTER_WEIGHT: f64 = 0.02;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Cone embedding parameters.
///
/// Each entity has a per-dimension `axis` angle (unconstrained; used only through
/// `|sin(·/2)|`) and a raw aperture parameter; the actual aperture is
/// `pi·sigmoid(raw_aperture) ∈ (0, pi)`.
#[derive(Module, Debug)]
pub struct BurnConeModel<B: Backend> {
    /// Per-dimension axis angles `[num_entities, dim]`.
    pub axes: Param<Tensor<B, 2>>,
    /// Raw aperture parameter `[num_entities, dim]` (`aperture = pi·sigmoid(raw)`).
    pub raw_aperture: Param<Tensor<B, 2>>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based cone trainer with autodiff.
pub struct BurnConeTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
}

impl<B: AutodiffBackend> Default for BurnConeTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
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
    pub fn init_model(
        &self,
        num_entities: usize,
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
        BurnConeModel {
            axes: param([num_entities, dim], -0.1, 0.1),
            raw_aperture: param([num_entities, dim], -0.3, 0.3),
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train for one epoch over `pairs`, returning the mean batch loss.
    ///
    /// `pairs` are `(parent, child)` containment edges. Negatives corrupt the
    /// child uniformly at random (excluding the true child).
    pub fn train_epoch(
        &mut self,
        model: &mut BurnConeModel<B>,
        optim: &mut impl Optimizer<BurnConeModel<B>, B>,
        pairs: &[(usize, usize)],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.axes.val().dims()[0];
        let batch_size = config.batch_size.max(1);
        let n_neg = config.negative_samples.max(1);

        let n = pairs.len();
        if n == 0 || num_entities <= 1 {
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
            let parent_vec: Vec<i64> = chunk.iter().map(|&i| pairs[i].0 as i64).collect();
            let child_vec: Vec<i64> = chunk.iter().map(|&i| pairs[i].1 as i64).collect();

            let mut neg_flat: Vec<i64> = Vec::with_capacity(chunk.len() * n_neg);
            for &c in &child_vec {
                for _ in 0..n_neg {
                    let neg = loop {
                        let v = rng.usize(0..num_entities) as i64;
                        if v != c {
                            break v;
                        }
                    };
                    neg_flat.push(neg);
                }
            }

            let current_model = model.clone();
            let loss = self.batch_loss(
                &current_model,
                &parent_vec,
                &child_vec,
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

    /// ConE containment loss for a batch of `(parent, child)` pairs plus
    /// corrupted-child negatives.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnConeModel<B>,
        parent_vec: &[i64],
        child_vec: &[i64],
        neg_child_flat: &[i64],
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = parent_vec.len();

        let parent_ids = Tensor::<B, 1, Int>::from_data(parent_vec, device);
        let child_ids = Tensor::<B, 1, Int>::from_data(child_vec, device);

        // ---- Positive: distance + reg·(mean_aper_parent + mean_aper_child) ----
        let (p_axes, p_aper) = gather_cone(model, parent_ids);
        let (c_axes, c_aper) = gather_cone(model, child_ids);
        let dist =
            cone_distance_batched(p_axes.clone(), p_aper.clone(), c_axes, CONE_CENTER_WEIGHT); // [bs, 1]

        let mean_aper_parent = p_aper.mean_dim(1); // [bs, 1]
        let mean_aper_child = c_aper.mean_dim(1); // [bs, 1]
        let reg = (mean_aper_parent + mean_aper_child).mul_scalar(config.regularization as f64);
        let pos_loss = (dist + reg).clamp_min(0.0); // [bs, 1]
        let pos_mean = pos_loss.mean();

        // ---- Negative: w_neg · max(0, margin − distance)^2 ----
        let parent_rep: Vec<i64> = parent_vec
            .iter()
            .flat_map(|&p| std::iter::repeat_n(p, n_neg))
            .collect();
        let parent_rep_ids = Tensor::<B, 1, Int>::from_data(parent_rep.as_slice(), device);
        let neg_ids = Tensor::<B, 1, Int>::from_data(neg_child_flat, device);

        let (pr_axes, pr_aper) = gather_cone(model, parent_rep_ids);
        let (ne_axes, _ne_aper) = gather_cone(model, neg_ids);
        let dist_neg = cone_distance_batched(pr_axes, pr_aper, ne_axes, CONE_CENTER_WEIGHT); // [bs*n_neg, 1]

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

    /// Extract `(axes, apertures)` as flat row-major `Vec<f32>`
    /// (`aperture = pi·sigmoid(raw_aperture)`).
    fn extract_params(model: &BurnConeModel<B>) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let n_e = model.axes.val().dims()[0];
        let dim = model.axes.val().dims()[1];
        let axes: Vec<f32> = model.axes.val().into_data().to_vec::<f32>().unwrap();
        let raw: Vec<f32> = model
            .raw_aperture
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let apers: Vec<f32> = raw
            .into_iter()
            .map(|r| PI * crate::utils::stable_sigmoid(r))
            .collect();
        (axes, apers, n_e, dim)
    }

    /// Evaluate containment-ranking quality on `pairs` treated as `(parent, child)`
    /// subsumption edges.
    ///
    /// Scores by `−cone_distance(parent, child)` (higher = better containment) and
    /// ranks the true child / true parent per pair with the same filtered,
    /// self-excluded protocol as the box trainer (see
    /// [`crate::trainer::burn_box_trainer::containment_ranking`]).
    pub fn evaluate(
        &self,
        model: &BurnConeModel<B>,
        pairs: &[(usize, usize)],
    ) -> crate::trainer::EvaluationResults {
        let (axes, apers, n_e, dim) = Self::extract_params(model);
        let score = |parent: usize, child: usize| -> f32 {
            -cone_distance_cpu(&axes, &apers, parent, child, dim, CONE_CENTER_WEIGHT as f32)
        };
        crate::trainer::burn_box_trainer::containment_ranking(pairs, n_e, score)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Gather `(axes, apertures)` for the given entity IDs. `aperture = pi·sigmoid(raw)`.
fn gather_cone<B: Backend>(
    model: &BurnConeModel<B>,
    ids: Tensor<B, 1, Int>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let axes = model.axes.val().select(0, ids.clone());
    let raw = model.raw_aperture.val().select(0, ids);
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

/// CPU ConE distance mirroring [`cone_distance_batched`] and
/// [`crate::ndarray_backend::NdarrayCone::cone_distance`] for evaluation scoring.
fn cone_distance_cpu(
    axes: &[f32],
    apers: &[f32],
    parent: usize,
    child: usize,
    dim: usize,
    cen: f32,
) -> f32 {
    let qo = parent * dim;
    let eo = child * dim;
    let mut dist_out = 0.0f32;
    let mut dist_in = 0.0f32;
    for i in 0..dim {
        let e = axes[eo + i];
        let q_axis = axes[qo + i];
        let q_aper = apers[qo + i];
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

    /// Containment hierarchy: animal ⊇ {mammal, bird}, mammal ⊇ {dog, cat},
    /// plus the transitive closure animal ⊇ {dog, cat}.
    fn hierarchy_pairs() -> Vec<(usize, usize)> {
        // 0=animal 1=mammal 2=bird 3=dog 4=cat
        vec![(0, 1), (0, 2), (1, 3), (1, 4), (0, 3), (0, 4)]
    }

    #[test]
    fn model_init_shapes() {
        let device = Default::default();
        let model = BurnConeTrainer::<TestBackend>::new().init_model(10, 4, &device);
        assert_eq!(model.axes.val().dims(), [10, 4]);
        assert_eq!(model.raw_aperture.val().dims(), [10, 4]);
    }

    #[test]
    fn batch_loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnConeTrainer::<TestBackend>::new();
        let model = trainer.init_model(6, 4, &device);
        let config = CpuBoxTrainingConfig::default();
        let loss = trainer.batch_loss(
            &model,
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
        let pairs = hierarchy_pairs();
        let mut trainer = BurnConeTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 8, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.02,
            margin: 1.0,
            negative_samples: 3,
            batch_size: 8,
            regularization: 0.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<TestBackend, BurnConeModel<TestBackend>>();

        let loss_0 = trainer.train_epoch(&mut model, &mut optim, &pairs, 0, &config, &device);
        let mut loss_last = loss_0;
        for epoch in 1..40 {
            loss_last =
                trainer.train_epoch(&mut model, &mut optim, &pairs, epoch, &config, &device);
        }
        assert!(
            loss_last < loss_0,
            "loss should decrease: epoch 0 = {loss_0:.4}, epoch 39 = {loss_last:.4}"
        );
    }

    /// End-to-end: train on a containment hierarchy and verify the model ranks
    /// true parent/child containment above random under filtered, self-excluded
    /// containment ranking. Cone geometry is the hardest of the region types
    /// (lowest MRR in the geometry comparison), so the bar (0.55) sits below box's
    /// (0.7) but still clearly above random-init behavior.
    #[test]
    fn train_and_evaluate_synthetic() {
        let device = Default::default();
        let pairs = hierarchy_pairs();
        let mut trainer = BurnConeTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 16, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.02,
            margin: 1.0,
            negative_samples: 4,
            batch_size: 8,
            regularization: 0.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<TestBackend, BurnConeModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..300 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &pairs, epoch, &config, &device);
        }

        let results = trainer.evaluate(&model, &pairs);
        eprintln!(
            "BurnCone synthetic: final_loss={last_loss:.4} MRR={:.3} H@1={:.3} mean_rank={:.2}",
            results.mrr, results.hits_at_1, results.mean_rank
        );
        assert!(
            results.mrr > 0.55,
            "MRR={} expected >0.55 (filtered, self-excluded containment ranking)",
            results.mrr
        );
    }
}
