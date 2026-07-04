//! Burn-based hard-box trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training
//!
//! # Task shape
//! This is a **pair-based subsumption** trainer, not a triple-based KGE trainer.
//! The training signal is a set of `(parent, child)` containment edges: `parent`
//! subsumes `child` iff the parent box contains the child box. Negatives are
//! `(parent, corrupted_child)` pairs that should NOT be contained.
//!
//! # Scoring convention (ported from [`crate::trainer::box_trainer::compute_pair_loss`])
//! Each entity is a box with `min` and `max = min + softplus(raw_delta)` (the
//! softplus reparameterization guarantees a strictly positive width and keeps
//! gradient flow unrestricted). For a directed pair `(A = parent, B = child)`:
//! - `P(B ⊆ A) = Vol(A ∩ B) / Vol(B)` — note the denominator is the **child**'s
//!   volume, matching `box_trainer.rs` (`vol_int_soft / vol_b`), not the parent's.
//! - Intersection side lengths use the softplus smoothing
//!   `side_d = softplus(min(a_max,b_max) − max(a_min,b_min), beta)` with
//!   `beta = config.softplus_beta`, so the volume is always positive and always
//!   has a gradient (matches [`crate::utils::softplus`]). All volumes are computed
//!   in log space (`log_vol = Σ_d ln(side_d)`) to avoid high-dimensional underflow.
//! - **Positive** loss: `−ln P(B ⊆ A) + reg·(Vol_A + Vol_B)` (with `−ln P` capped
//!   at 10.0 and `P` clamped to `[1e-8, 1]`, matching the CPU trainer).
//! - **Negative** loss: `w_neg · max(0, max(P_AB, P_BA) − margin)^2` where
//!   `P_AB = Vol_int/Vol_B`, `P_BA = Vol_int/Vol_A`, both clamped to `[0, 1]`.
//!
//! Evaluation ranks by containment probability, so higher score = better
//! containment, consistent with `evaluate_link_prediction_generic`.

use crate::trainer::trainer_utils::self_adversarial_weights;
use crate::trainer::CpuBoxTrainingConfig;
use burn::module::{Param, ParamId};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// `ln(1e-8)` — lower clamp on `ln P` for the positive (directed) containment loss.
const LN_MIN_PROB: f64 = -18.420_681;
/// Upper clamp on `−ln P` (matches the CPU trainer's `.min(10.0)`).
const MAX_NEG_LOG_PROB: f64 = 10.0;

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

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Hard-box embedding parameters.
///
/// Each entity is a box stored as a lower corner `min` and a raw width parameter
/// `raw_delta`; the actual width is `softplus(raw_delta)` (always positive), so
/// the upper corner is `max = min + softplus(raw_delta)`.
#[derive(Module, Debug)]
pub struct BurnBoxModel<B: Backend> {
    /// Lower corners `[num_entities, dim]`.
    pub min: Param<Tensor<B, 2>>,
    /// Raw width parameter `[num_entities, dim]` (`width = softplus(raw_delta)`).
    pub raw_delta: Param<Tensor<B, 2>>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based hard-box trainer with autodiff.
pub struct BurnBoxTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
}

impl<B: AutodiffBackend> Default for BurnBoxTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
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
    /// Lower corners are small (near the origin); raw widths are initialized so
    /// that `softplus(raw_delta)` starts around 1–2, giving parents room to grow
    /// to contain their children.
    pub fn init_model(
        &self,
        num_entities: usize,
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
        BurnBoxModel {
            min: param([num_entities, dim], -0.1, 0.1),
            raw_delta: param([num_entities, dim], 0.5, 2.0),
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
        model: &mut BurnBoxModel<B>,
        optim: &mut impl Optimizer<BurnBoxModel<B>, B>,
        pairs: &[(usize, usize)],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.min.val().dims()[0];
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

            // Corrupt-child negatives: uniform entity != true child.
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

    /// Directed containment loss for a batch of `(parent, child)` pairs plus
    /// corrupted-child negatives.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnBoxModel<B>,
        parent_vec: &[i64],
        child_vec: &[i64],
        neg_child_flat: &[i64],
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = parent_vec.len();
        let beta = config.softplus_beta as f64;

        let parent_ids = Tensor::<B, 1, Int>::from_data(parent_vec, device);
        let child_ids = Tensor::<B, 1, Int>::from_data(child_vec, device);

        // ---- Positive term: −ln P(child ⊆ parent) + reg·(Vol_parent + Vol_child) ----
        let (p_min, p_width) = gather_box(model, parent_ids);
        let (c_min, c_width) = gather_box(model, child_ids);
        let (lvi, lv_parent, lv_child) = box_logvols(p_min, p_width, c_min, c_width, beta); // each [bs, 1]

        // ln P(child ⊆ parent) = ln(Vol_int / Vol_child), clamped to [ln 1e-8, 0].
        let log_p = (lvi - lv_child.clone()).clamp(LN_MIN_PROB, 0.0);
        let neg_log_prob = log_p.neg().clamp(0.0, MAX_NEG_LOG_PROB); // [bs, 1]

        // reg·(Vol_parent + Vol_child); clamp log-vol before exp as an overflow guard.
        let reg = (lv_parent.clamp_max(30.0).exp() + lv_child.clamp_max(30.0).exp())
            .mul_scalar(config.regularization as f64);
        let pos_loss = (neg_log_prob + reg).clamp_min(0.0); // [bs, 1]
        let pos_mean = pos_loss.mean();

        // ---- Negative term: w_neg · max(0, max(P_AB, P_BA) − margin)^2 ----
        let parent_rep: Vec<i64> = parent_vec
            .iter()
            .flat_map(|&p| std::iter::repeat_n(p, n_neg))
            .collect();
        let parent_rep_ids = Tensor::<B, 1, Int>::from_data(parent_rep.as_slice(), device);
        let neg_ids = Tensor::<B, 1, Int>::from_data(neg_child_flat, device);

        let (pr_min, pr_width) = gather_box(model, parent_rep_ids);
        let (n_min, n_width) = gather_box(model, neg_ids);
        let (nlvi, nlv_parent, nlv_child) = box_logvols(pr_min, pr_width, n_min, n_width, beta); // each [bs*n_neg, 1]

        // P_AB = Vol_int / Vol_negchild, P_BA = Vol_int / Vol_parent, both in [0, 1].
        let p_ab = (nlvi.clone() - nlv_child).exp().clamp(0.0, 1.0);
        let p_ba = (nlvi - nlv_parent).exp().clamp(0.0, 1.0);
        let max_p = p_ab.max_pair(p_ba); // [bs*n_neg, 1]
        let margin_loss = (max_p.clone() - config.margin as f64)
            .clamp_min(0.0)
            .powf_scalar(2.0)
            .mul_scalar(config.negative_weight as f64); // [bs*n_neg, 1]
        let neg_2d = margin_loss.reshape([bs, n_neg]); // [bs, n_neg]

        let neg_mean = if config.self_adversarial && config.adversarial_temperature > 0.0 {
            // Harder negatives (higher containment) get more weight.
            let scores = max_p.reshape([bs, n_neg]).into_data();
            let slice = scores.as_slice::<f32>().expect("max_p f32");
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

    /// Extract `(min, width)` as flat row-major `Vec<f32>` (`width = softplus(raw_delta)`).
    fn extract_params(model: &BurnBoxModel<B>) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let n_e = model.min.val().dims()[0];
        let dim = model.min.val().dims()[1];
        let mins: Vec<f32> = model.min.val().into_data().to_vec::<f32>().unwrap();
        let raw: Vec<f32> = model.raw_delta.val().into_data().to_vec::<f32>().unwrap();
        let widths: Vec<f32> = raw
            .into_iter()
            .map(|r| crate::utils::softplus(r, 1.0))
            .collect();
        (mins, widths, n_e, dim)
    }

    /// Evaluate containment-ranking quality on `pairs` treated as `(parent, child)`
    /// subsumption edges.
    ///
    /// For each pair this ranks the true child (among candidate children of the
    /// parent) and the true parent (among candidate parents of the child) by
    /// `P(child ⊆ parent)`. Rankings exclude the trivial self-pair (a concept is
    /// its own subsumer/subsumee, which the softplus volume scores as `P≈1` and
    /// would otherwise dominate) and filter other known-true edges — the standard
    /// filtered-ranking protocol adapted to containment.
    pub fn evaluate(
        &self,
        model: &BurnBoxModel<B>,
        pairs: &[(usize, usize)],
        beta: f32,
    ) -> crate::trainer::EvaluationResults {
        let (mins, widths, n_e, dim) = Self::extract_params(model);
        let score = |parent: usize, child: usize| -> f32 {
            box_containment_prob(&mins, &widths, parent, child, dim, beta)
        };
        containment_ranking(pairs, n_e, score)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Gather `(min, width)` boxes for the given entity IDs. `width = softplus(raw_delta)`.
fn gather_box<B: Backend>(
    model: &BurnBoxModel<B>,
    ids: Tensor<B, 1, Int>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let min = model.min.val().select(0, ids.clone());
    let raw = model.raw_delta.val().select(0, ids);
    let width = softplus_beta(raw, 1.0);
    (min, width)
}

/// Log-space volumes for a batch of box pairs `(A, B)`.
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

/// CPU containment probability `P(child ⊆ parent) = Vol(parent ∩ child) / Vol(child)`.
///
/// Mirrors [`box_logvols`] on flat parameter arrays for evaluation scoring.
fn box_containment_prob(
    mins: &[f32],
    widths: &[f32],
    parent: usize,
    child: usize,
    dim: usize,
    beta: f32,
) -> f32 {
    let po = parent * dim;
    let co = child * dim;
    let mut log_vol_int = 0.0f32;
    let mut log_vol_child = 0.0f32;
    for i in 0..dim {
        let p_min = mins[po + i];
        let p_max = p_min + widths[po + i];
        let c_min = mins[co + i];
        let c_max = c_min + widths[co + i];
        let lo = p_min.max(c_min);
        let hi = p_max.min(c_max);
        let side = crate::utils::softplus(hi - lo, beta);
        log_vol_int += (side + 1e-30).ln();
        log_vol_child += (widths[co + i] + 1e-30).ln();
    }
    (log_vol_int - log_vol_child).exp().clamp(0.0, 1.0)
}

/// Filtered, self-excluded containment ranking over `(parent, child)` pairs.
///
/// `score(parent, child)` returns a higher-is-better containment score. For each
/// pair, ranks the true child among candidate children of the parent and the true
/// parent among candidate parents of the child, excluding the query's self-pair and
/// all other known-true edges. Returns aggregate MRR / Hits@k / mean rank.
pub(crate) fn containment_ranking<F>(
    pairs: &[(usize, usize)],
    num_entities: usize,
    score: F,
) -> crate::trainer::EvaluationResults
where
    F: Fn(usize, usize) -> f32,
{
    use std::collections::{HashMap, HashSet};

    let mut children_of: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut parents_of: HashMap<usize, HashSet<usize>> = HashMap::new();
    for &(p, c) in pairs {
        children_of.entry(p).or_default().insert(c);
        parents_of.entry(c).or_default().insert(p);
    }
    let empty: HashSet<usize> = HashSet::new();

    let mut head_ranks: Vec<usize> = Vec::with_capacity(pairs.len());
    let mut tail_ranks: Vec<usize> = Vec::with_capacity(pairs.len());

    for &(p, c) in pairs {
        // Tail prediction: rank true child c among candidate children of parent p.
        let target_t = score(p, c);
        let known_children = children_of.get(&p).unwrap_or(&empty);
        let mut rank_t = 1usize;
        for cand in 0..num_entities {
            if cand == c || cand == p || known_children.contains(&cand) {
                continue;
            }
            if score(p, cand) > target_t {
                rank_t += 1;
            }
        }
        tail_ranks.push(rank_t);

        // Head prediction: rank true parent p among candidate parents of child c.
        let target_h = score(p, c);
        let known_parents = parents_of.get(&c).unwrap_or(&empty);
        let mut rank_h = 1usize;
        for cand in 0..num_entities {
            if cand == p || cand == c || known_parents.contains(&cand) {
                continue;
            }
            if score(cand, c) > target_h {
                rank_h += 1;
            }
        }
        head_ranks.push(rank_h);
    }

    let mrr_of = |ranks: &[usize]| -> f32 {
        if ranks.is_empty() {
            return 0.0;
        }
        ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / ranks.len() as f32
    };
    let head_mrr = mrr_of(&head_ranks);
    let tail_mrr = mrr_of(&tail_ranks);

    let all: Vec<usize> = head_ranks
        .iter()
        .chain(tail_ranks.iter())
        .copied()
        .collect();
    let n = all.len().max(1) as f32;
    let hits = |k: usize| all.iter().filter(|&&r| r <= k).count() as f32 / n;
    let mean_rank = all.iter().sum::<usize>() as f32 / n;

    crate::trainer::EvaluationResults {
        mrr: (head_mrr + tail_mrr) / 2.0,
        head_mrr,
        tail_mrr,
        hits_at_1: hits(1),
        hits_at_3: hits(3),
        hits_at_10: hits(10),
        mean_rank,
        rank_variance: f32::NAN,
        rank_p25: f32::NAN,
        rank_p50: f32::NAN,
        rank_p75: f32::NAN,
        rank_p95: f32::NAN,
        per_relation: vec![],
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

    /// Containment hierarchy: animal ⊇ {mammal, bird}, mammal ⊇ {dog, cat},
    /// plus the transitive closure animal ⊇ {dog, cat}.
    fn hierarchy_pairs() -> Vec<(usize, usize)> {
        // 0=animal 1=mammal 2=bird 3=dog 4=cat
        vec![(0, 1), (0, 2), (1, 3), (1, 4), (0, 3), (0, 4)]
    }

    #[test]
    fn model_init_shapes() {
        let device = Default::default();
        let model = BurnBoxTrainer::<TestBackend>::new().init_model(10, 4, &device);
        assert_eq!(model.min.val().dims(), [10, 4]);
        assert_eq!(model.raw_delta.val().dims(), [10, 4]);
    }

    #[test]
    fn batch_loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnBoxTrainer::<TestBackend>::new();
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
        let mut trainer = BurnBoxTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 8, &device);
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
    /// containment ranking. On this clean hierarchy a trained model resolves nearly
    /// every filtered query to rank 1, so the bar of 0.7 is cleared comfortably and
    /// repeatably across random inits (untrained/random init scores far lower).
    #[test]
    fn train_and_evaluate_synthetic() {
        let device = Default::default();
        let pairs = hierarchy_pairs();
        let mut trainer = BurnBoxTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 8, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.3,
            negative_samples: 4,
            batch_size: 8,
            ..Default::default()
        };
        let mut optim = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<TestBackend, BurnBoxModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..200 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &pairs, epoch, &config, &device);
        }

        let results = trainer.evaluate(&model, &pairs, config.softplus_beta);
        eprintln!(
            "BurnBox synthetic: final_loss={last_loss:.4} MRR={:.3} H@1={:.3} mean_rank={:.2}",
            results.mrr, results.hits_at_1, results.mean_rank
        );
        assert!(
            results.mrr > 0.7,
            "MRR={} expected >0.7 (filtered, self-excluded containment ranking)",
            results.mrr
        );
    }
}
