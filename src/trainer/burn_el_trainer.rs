//! Burn-based EL++ ontology embedding trainer with autodiff.
//!
//! Ports [`CandleElTrainer`](super::candle_el_trainer::CandleElTrainer) to the
//! burn backend for multi-backend training (ndarray CPU, wgpu GPU, tch CUDA).
//!
//! # Architecture
//!
//! Concept embeddings: center `[nc, dim]` + offset `[nc, dim]` (abs of raw params).
//! Bump translations: `[nc, dim]` (Box2EL-style directional encoding).
//! Role embeddings: head `[nr, dim*2]` + tail `[nr, dim*2]` (center + offset packed).
//!
//! # Training features
//!
//! - All 4 concept NF types (NF1-NF4) with balanced per-type mini-batch sampling
//! - Gumbel soft intersection for NF1 (Dasgupta et al., 2020) with beta annealing
//! - Center attraction fallback for NF1 degenerate intersections
//! - Box2EL-style bump-based existential encoding for NF3/NF4
//! - Disjointness-target negative sampling for NF2/NF3
//! - Cosine LR schedule
//! - Validation-based checkpointing (best NF2+NF3 MRR)
//! - Role L2 regularization
//!
//! # Backend selection
//!
//! - `burn-ndarray` (+ rayon) -- multi-core CPU training
//! - `burn-wgpu`              -- Metal/Vulkan/WebGPU training

use crate::el_training::{Axiom, Ontology};
use crate::optimizer::cosine_lr;
use burn::module::{Param, ParamId};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// EL++ box embedding model parameters.
#[derive(Module, Debug)]
pub struct BurnElModel<B: Backend> {
    /// Concept centers `[num_concepts, dim]`.
    pub concept_centers: Param<Tensor<B, 2>>,
    /// Concept raw offsets `[num_concepts, dim]` (abs applied at use).
    pub concept_offsets: Param<Tensor<B, 2>>,
    /// Per-concept bump translations `[num_concepts, dim]`.
    pub bumps: Param<Tensor<B, 2>>,
    /// Role head boxes `[num_roles, dim*2]` (center + offset packed).
    pub role_heads: Param<Tensor<B, 2>>,
    /// Role tail boxes `[num_roles, dim*2]` (center + offset packed).
    pub role_tails: Param<Tensor<B, 2>>,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Training configuration for the burn EL++ trainer.
#[derive(Debug, Clone)]
pub struct BurnElConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Margin for inclusion loss.
    pub margin: f32,
    /// Target separation distance for negatives.
    pub neg_dist: f32,
    /// Number of negative samples per positive.
    pub negative_samples: usize,
    /// Mini-batch size per NF type.
    pub batch_size: usize,
    /// Learning rate.
    pub lr: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Regularization factor for bumps.
    pub reg_factor: f32,
    /// NF1 center attraction weight (0.5 recommended).
    pub nf1_center_weight: f32,
    /// Gumbel beta start (soft, 0.3 recommended).
    pub beta_start: f32,
    /// Gumbel beta end (sharp, 2.0 recommended).
    pub beta_end: f32,
    /// Cosine LR minimum fraction (0.1 recommended).
    pub lr_min_frac: f64,
    /// NF4 negative weight (0.0 = disabled, matching Box2EL).
    pub nf4_neg_weight: f32,
    /// Role regularization multiplier relative to reg_factor (0.1 recommended).
    pub role_reg_mult: f32,
}

impl Default for BurnElConfig {
    fn default() -> Self {
        Self {
            dim: 200,
            margin: 0.1,
            neg_dist: 2.0,
            negative_samples: 2,
            batch_size: 512,
            lr: 0.01,
            epochs: 1000,
            reg_factor: 0.5,
            nf1_center_weight: 0.5,
            beta_start: 0.3,
            beta_end: 2.0,
            lr_min_frac: 0.1,
            nf4_neg_weight: 0.0,
            role_reg_mult: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based EL++ trainer.
pub struct BurnElTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> Default for BurnElTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: AutodiffBackend> BurnElTrainer<B> {
    /// Create a new trainer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize a randomly-weighted model.
    ///
    /// Uses L2-normalized initialization (matching Box2EL / CandleElTrainer):
    /// each row is sampled from Uniform[-1, 1] then divided by its L2 norm,
    /// placing all initial embeddings on the unit sphere. This gives the bump
    /// mechanism a consistent initial scale for NF3/NF4 existential encoding.
    pub fn init_model(
        num_concepts: usize,
        num_roles: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnElModel<B> {
        let l2_param = |shape: [usize; 2]| {
            let raw = Tensor::<B, 2>::random(
                shape,
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                device,
            );
            // L2-normalize each row (matching Box2EL init_embeddings).
            let norm = raw
                .clone()
                .powf_scalar(2.0)
                .sum_dim(1)
                .clamp_min(1e-8)
                .sqrt();
            let normalized = raw / norm;
            Param::initialized(ParamId::new(), normalized.require_grad())
        };
        let nr = num_roles.max(1);
        BurnElModel {
            concept_centers: l2_param([num_concepts, dim]),
            concept_offsets: l2_param([num_concepts, dim]),
            bumps: l2_param([num_concepts, dim]),
            role_heads: l2_param([nr, dim * 2]),
            role_tails: l2_param([nr, dim * 2]),
        }
    }

    /// Train on an ontology.
    ///
    /// Returns per-epoch loss values.
    #[allow(clippy::too_many_lines)]
    pub fn fit(
        &self,
        model: &mut BurnElModel<B>,
        ontology: &Ontology,
        config: &BurnElConfig,
        device: &B::Device,
    ) -> Vec<f32> {
        let nc = ontology.concept_names.len();
        let dim = config.dim;

        // GCI0 deductive closure for negative filtering (DELE, Mashkova et al. 2024).
        let closure = ontology.subsumption_closure();
        if !closure.is_empty() {
            eprintln!(
                "  Deductive closure: {} entailed subsumption pairs (filtering negatives)",
                closure.len()
            );
        }

        // Group axioms by type.
        let mut nf2_axioms: Vec<(usize, usize)> = Vec::new();
        let mut nf1_axioms: Vec<(usize, usize, usize)> = Vec::new();
        let mut nf3_axioms: Vec<(usize, usize, usize)> = Vec::new();
        let mut nf4_axioms: Vec<(usize, usize, usize)> = Vec::new();
        let mut disj_axioms: Vec<(usize, usize)> = Vec::new();

        for ax in &ontology.axioms {
            match *ax {
                Axiom::SubClassOf { sub, sup } => nf2_axioms.push((sub, sup)),
                Axiom::Intersection { c1, c2, target } => nf1_axioms.push((c1, c2, target)),
                Axiom::ExistentialRight { sub, role, filler } => {
                    nf3_axioms.push((sub, role, filler));
                }
                Axiom::Existential {
                    role,
                    filler,
                    target,
                } => nf4_axioms.push((role, filler, target)),
                Axiom::Disjoint { a, b } => disj_axioms.push((a, b)),
                _ => {}
            }
        }

        // LCG RNG (matches CandleElTrainer for reproducibility).
        let mut master_rng: u64 = 42;
        let lcg = |s: &mut u64| -> usize {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as usize
        };

        // Adam optimizer. Match candle-nn's epsilon (1e-8, not burn's default 1e-5)
        // to avoid dampening sparse bump/role gradient updates.
        let optim_config = burn::optim::AdamConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(0.0)));
        let mut optim = optim_config.init::<B, BurnElModel<B>>();

        let mut epoch_losses = Vec::with_capacity(config.epochs);

        // Validation checkpoint state.
        let mut best_val_score = f32::NEG_INFINITY;
        #[allow(clippy::type_complexity)]
        let mut best_params: Option<Vec<(String, Vec<f32>, Vec<usize>)>> = None;
        let mut best_epoch = 0usize;

        for epoch in 0..config.epochs {
            let mut rng_nf2 = master_rng.wrapping_add(1);
            let mut rng_nf1 = master_rng.wrapping_add(2);
            let mut rng_nf3 = master_rng.wrapping_add(3);
            let mut rng_nf4 = master_rng.wrapping_add(4);
            lcg(&mut master_rng);

            let current_lr = cosine_lr(epoch, config.epochs, config.lr, config.lr_min_frac);

            // Clone model for this epoch's forward pass.
            let current_model = model.clone();
            let mut losses: Vec<Tensor<B, 1>> = Vec::new();

            // --- NF2: C ⊑ D ---
            if !nf2_axioms.is_empty() {
                let bs = config.batch_size.min(nf2_axioms.len());
                let mut sub_ids = Vec::with_capacity(bs);
                let mut sup_ids = Vec::with_capacity(bs);
                let mut sub_ids_raw = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_nf2) % nf2_axioms.len();
                    sub_ids_raw.push(nf2_axioms[idx].0);
                    sub_ids.push(nf2_axioms[idx].0 as i64);
                    sup_ids.push(nf2_axioms[idx].1 as i64);
                }

                let sub_t = Tensor::<B, 1, Int>::from_data(sub_ids.as_slice(), device);
                let sup_t = Tensor::<B, 1, Int>::from_data(sup_ids.as_slice(), device);

                let (c_sub, o_sub) = concept_boxes(&current_model, &sub_t);
                let (c_sup, o_sup) = concept_boxes(&current_model, &sup_t);

                let pos_loss =
                    inclusion_loss(c_sub.clone(), o_sub.clone(), c_sup, o_sup, config.margin)
                        .powf_scalar(2.0)
                        .mean();

                // Negatives: disjointness target (closure-filtered).
                let mut neg_losses: Vec<Tensor<B, 1>> = Vec::new();
                for _ in 0..config.negative_samples {
                    let neg_ids: Vec<i64> = (0..bs)
                        .map(|j| {
                            let sub_j = sub_ids_raw[j];
                            for _ in 0..10 {
                                let neg = lcg(&mut rng_nf2) % nc;
                                if !closure.contains(&(sub_j, neg)) {
                                    return neg as i64;
                                }
                            }
                            (lcg(&mut rng_nf2) % nc) as i64
                        })
                        .collect();
                    let neg_t = Tensor::<B, 1, Int>::from_data(neg_ids.as_slice(), device);
                    let (c_neg, o_neg) = concept_boxes(&current_model, &neg_t);
                    let disj =
                        neg_loss_fn(c_sub.clone(), o_sub.clone(), c_neg, o_neg, config.margin);
                    let target_val = Tensor::<B, 2>::full([bs, 1], config.neg_dist, device);
                    let gap = (target_val - disj).powf_scalar(2.0).mean();
                    neg_losses.push(gap);
                }
                let neg_sum = neg_losses
                    .into_iter()
                    .reduce(|a, b| a + b)
                    .unwrap_or_else(|| Tensor::zeros([1], device));
                losses.push(pos_loss + neg_sum);
            }

            // --- NF1: C1 ⊓ C2 ⊑ D (Gumbel soft intersection) ---
            if !nf1_axioms.is_empty() {
                let bs = config.batch_size.min(nf1_axioms.len());
                let mut c1_ids = Vec::with_capacity(bs);
                let mut c2_ids = Vec::with_capacity(bs);
                let mut d_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_nf1) % nf1_axioms.len();
                    c1_ids.push(nf1_axioms[idx].0 as i64);
                    c2_ids.push(nf1_axioms[idx].1 as i64);
                    d_ids.push(nf1_axioms[idx].2 as i64);
                }

                let c1_t = Tensor::<B, 1, Int>::from_data(c1_ids.as_slice(), device);
                let c2_t = Tensor::<B, 1, Int>::from_data(c2_ids.as_slice(), device);
                let d_t = Tensor::<B, 1, Int>::from_data(d_ids.as_slice(), device);

                let (cc1, oc1) = concept_boxes(&current_model, &c1_t);
                let (cc2, oc2) = concept_boxes(&current_model, &c2_t);
                let (cd, od) = concept_boxes(&current_model, &d_t);

                // Gumbel beta annealing.
                let progress = epoch as f32 / config.epochs.max(1) as f32;
                let beta = config.beta_start + (config.beta_end - config.beta_start) * progress;
                let inv_beta = 1.0 / beta as f64;

                let min1 = cc1.clone() - oc1.clone();
                let max1 = cc1.clone() + oc1;
                let min2 = cc2.clone() - oc2.clone();
                let max2 = cc2.clone() + oc2;

                // soft_max(min1, min2) via LSE: intersection lower bound.
                let inter_min = {
                    let a = min1 * inv_beta;
                    let b = min2 * inv_beta;
                    let m = a.clone().max_pair(b.clone());
                    let s = (a - m.clone()).exp() + (b - m.clone()).exp();
                    (m + s.log()) * beta as f64
                };

                // soft_min(max1, max2) via negative LSE.
                let inter_max = {
                    let a = max1 * (-inv_beta);
                    let b = max2 * (-inv_beta);
                    let m = a.clone().max_pair(b.clone());
                    let s = (a - m.clone()).exp() + (b - m.clone()).exp();
                    (m + s.log()) * (-(beta as f64))
                };

                let inter_center = (inter_min.clone() + inter_max.clone()) * 0.5;
                let inter_offset = (inter_max - inter_min).clamp_min(0.0) * 0.5;

                let nf1_incl =
                    inclusion_loss(inter_center, inter_offset, cd.clone(), od, config.margin)
                        .mean();

                // Center attraction fallback.
                let midpoint = (cc1 + cc2) * 0.5;
                let center_dist = (midpoint - cd)
                    .powf_scalar(2.0)
                    .sum_dim(1)
                    .clamp_min(1e-8)
                    .sqrt()
                    .mean();

                let nf1_loss = nf1_incl + center_dist * config.nf1_center_weight as f64;
                losses.push(nf1_loss);
            }

            // --- NF3: C ⊑ ∃r.D (bump-based existential) ---
            if !nf3_axioms.is_empty() {
                let bs = config.batch_size.min(nf3_axioms.len());
                let mut sub_ids = Vec::with_capacity(bs);
                let mut role_ids = Vec::with_capacity(bs);
                let mut filler_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_nf3) % nf3_axioms.len();
                    sub_ids.push(nf3_axioms[idx].0 as i64);
                    role_ids.push(nf3_axioms[idx].1 as i64);
                    filler_ids.push(nf3_axioms[idx].2 as i64);
                }

                let sub_t = Tensor::<B, 1, Int>::from_data(sub_ids.as_slice(), device);
                let role_t = Tensor::<B, 1, Int>::from_data(role_ids.as_slice(), device);
                let filler_t = Tensor::<B, 1, Int>::from_data(filler_ids.as_slice(), device);

                let (c_sub, o_sub) = concept_boxes(&current_model, &sub_t);
                let (c_filler, o_filler) = concept_boxes(&current_model, &filler_t);
                let bump_sub = current_model.bumps.val().select(0, sub_t.clone());
                let bump_filler = current_model.bumps.val().select(0, filler_t.clone());
                let (c_head, o_head) = role_box(&current_model, &role_t, true, dim);
                let (c_tail, o_tail) = role_box(&current_model, &role_t, false, dim);

                // C + bump_D -> head box.
                let dist1 = inclusion_loss(
                    c_sub.clone() + bump_filler.clone(),
                    o_sub.clone(),
                    c_head.clone(),
                    o_head.clone(),
                    config.margin,
                );

                // D + bump_C -> tail box.
                let dist2 = inclusion_loss(
                    c_filler.clone() + bump_sub.clone(),
                    o_filler.clone(),
                    c_tail.clone(),
                    o_tail.clone(),
                    config.margin,
                );

                let nf3_pos = (dist1 + dist2).mean() * 0.5;

                // NF3 negatives.
                let mut nf3_neg_losses: Vec<Tensor<B, 1>> = Vec::new();
                for _ in 0..config.negative_samples {
                    // Corrupt tail (D).
                    let neg_tail_ids: Vec<i64> =
                        (0..bs).map(|_| (lcg(&mut rng_nf3) % nc) as i64).collect();
                    let neg_tail_t =
                        Tensor::<B, 1, Int>::from_data(neg_tail_ids.as_slice(), device);
                    let bump_neg_tail = current_model.bumps.val().select(0, neg_tail_t);
                    let nl1 = neg_loss_fn(
                        c_sub.clone() + bump_neg_tail,
                        o_sub.clone(),
                        c_head.clone(),
                        o_head.clone(),
                        config.margin,
                    );
                    let t1 = Tensor::<B, 2>::full([bs, 1], config.neg_dist, device);
                    let neg1 = (t1 - nl1).powf_scalar(2.0).mean();

                    // Corrupt head (C).
                    let neg_head_ids: Vec<i64> =
                        (0..bs).map(|_| (lcg(&mut rng_nf3) % nc) as i64).collect();
                    let neg_head_t =
                        Tensor::<B, 1, Int>::from_data(neg_head_ids.as_slice(), device);
                    let (c_neg_head, o_neg_head) = concept_boxes(&current_model, &neg_head_t);
                    let nl2 = neg_loss_fn(
                        c_neg_head + bump_sub.clone(),
                        o_neg_head,
                        c_tail.clone(),
                        o_tail.clone(),
                        config.margin,
                    );
                    let t2 = Tensor::<B, 2>::full([bs, 1], config.neg_dist, device);
                    let neg2 = (t2 - nl2).powf_scalar(2.0).mean();

                    nf3_neg_losses.push(neg1 + neg2);
                }
                let nf3_neg_sum = nf3_neg_losses
                    .into_iter()
                    .reduce(|a, b| a + b)
                    .unwrap_or_else(|| Tensor::zeros([1], device));
                losses.push(nf3_pos + nf3_neg_sum);
            }

            // --- NF4: ∃r.C ⊑ D ---
            if !nf4_axioms.is_empty() {
                let bs = config.batch_size.min(nf4_axioms.len());
                let mut role_ids = Vec::with_capacity(bs);
                let mut filler_ids = Vec::with_capacity(bs);
                let mut target_ids = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_nf4) % nf4_axioms.len();
                    role_ids.push(nf4_axioms[idx].0 as i64);
                    filler_ids.push(nf4_axioms[idx].1 as i64);
                    target_ids.push(nf4_axioms[idx].2 as i64);
                }

                let role_t = Tensor::<B, 1, Int>::from_data(role_ids.as_slice(), device);
                let filler_t = Tensor::<B, 1, Int>::from_data(filler_ids.as_slice(), device);
                let target_t = Tensor::<B, 1, Int>::from_data(target_ids.as_slice(), device);

                let (c_target, o_target) = concept_boxes(&current_model, &target_t);
                let bump_filler = current_model.bumps.val().select(0, filler_t);
                let (c_head, o_head) = role_box(&current_model, &role_t, true, dim);

                let c_head_shifted = c_head.clone() - bump_filler.clone();
                let nf4_loss = inclusion_loss(
                    c_head_shifted,
                    o_head.clone(),
                    c_target.clone(),
                    o_target.clone(),
                    config.margin,
                )
                .powf_scalar(2.0)
                .mean();

                let nf4_total = if config.nf4_neg_weight > 0.0 {
                    let mut nf4_neg_losses: Vec<Tensor<B, 1>> = Vec::new();
                    for _ in 0..config.negative_samples {
                        let neg_target_ids: Vec<i64> =
                            (0..bs).map(|_| (lcg(&mut rng_nf4) % nc) as i64).collect();
                        let neg_target_t =
                            Tensor::<B, 1, Int>::from_data(neg_target_ids.as_slice(), device);
                        let (c_neg, o_neg) = concept_boxes(&current_model, &neg_target_t);
                        let nl = neg_loss_fn(
                            c_head.clone() - bump_filler.clone(),
                            o_head.clone(),
                            c_neg,
                            o_neg,
                            config.margin,
                        );
                        let t = Tensor::<B, 2>::full([bs, 1], config.neg_dist, device);
                        nf4_neg_losses.push((t - nl).powf_scalar(2.0).mean());
                    }
                    let neg_sum = nf4_neg_losses
                        .into_iter()
                        .reduce(|a, b| a + b)
                        .unwrap_or_else(|| Tensor::zeros([1], device));
                    nf4_loss + neg_sum * config.nf4_neg_weight as f64
                } else {
                    nf4_loss
                };
                losses.push(nf4_total);
            }

            // --- DISJ: C ⊓ D ⊑ ⊥ (disjointness -- boxes should not overlap) ---
            if !disj_axioms.is_empty() {
                let bs = config.batch_size.min(disj_axioms.len());
                let mut a_ids = Vec::with_capacity(bs);
                let mut b_ids = Vec::with_capacity(bs);
                let mut rng_disj = master_rng.wrapping_add(5);
                for _ in 0..bs {
                    let idx = lcg(&mut rng_disj) % disj_axioms.len();
                    a_ids.push(disj_axioms[idx].0 as i64);
                    b_ids.push(disj_axioms[idx].1 as i64);
                }

                let a_t = Tensor::<B, 1, Int>::from_data(a_ids.as_slice(), device);
                let b_t = Tensor::<B, 1, Int>::from_data(b_ids.as_slice(), device);

                let (c_a, o_a) = concept_boxes(&current_model, &a_t);
                let (c_b, o_b) = concept_boxes(&current_model, &b_t);

                // Positive: penalize overlap. neg_loss_fn returns distance-to-separation;
                // we want boxes separated by at least margin, so use neg_loss_fn with
                // a zero-target (minimize overlap).
                let separation = neg_loss_fn(
                    c_a.clone(),
                    o_a.clone(),
                    c_b.clone(),
                    o_b.clone(),
                    config.margin,
                );
                // Loss = relu(neg_dist - separation)^2: push separation toward neg_dist.
                let disj_target = Tensor::<B, 2>::full([bs, 1], config.neg_dist, device);
                let disj_pos_loss = (disj_target - separation).powf_scalar(2.0).mean();

                // Negative: random non-disjoint pairs should overlap.
                let mut disj_neg_losses: Vec<Tensor<B, 1>> = Vec::new();
                for _ in 0..config.negative_samples {
                    let neg_ids: Vec<i64> =
                        (0..bs).map(|_| (lcg(&mut rng_disj) % nc) as i64).collect();
                    let neg_t = Tensor::<B, 1, Int>::from_data(neg_ids.as_slice(), device);
                    let (c_neg, o_neg) = concept_boxes(&current_model, &neg_t);
                    // Non-disjoint pairs should have small inclusion loss (overlap).
                    let overlap =
                        inclusion_loss(c_a.clone(), o_a.clone(), c_neg, o_neg, config.margin)
                            .powf_scalar(2.0)
                            .mean();
                    disj_neg_losses.push(overlap);
                }
                let disj_neg_sum = disj_neg_losses
                    .into_iter()
                    .reduce(|a, b| a + b)
                    .unwrap_or_else(|| Tensor::zeros([1], device));
                losses.push(disj_pos_loss + disj_neg_sum);
            }

            // --- Regularization ---
            if config.reg_factor > 0.0 {
                let bump_reg = current_model
                    .bumps
                    .val()
                    .powf_scalar(2.0)
                    .sum_dim(1)
                    .sqrt()
                    .mean();
                losses.push(bump_reg * config.reg_factor as f64);

                let nr = current_model.role_heads.val().dims()[0];
                if nr > 0 {
                    let rh_reg = current_model.role_heads.val().powf_scalar(2.0).mean();
                    let rt_reg = current_model.role_tails.val().powf_scalar(2.0).mean();
                    losses.push(
                        (rh_reg + rt_reg)
                            * (config.reg_factor as f64 * config.role_reg_mult as f64),
                    );
                }
            }

            // Combine all losses and backward.
            let total_loss = if losses.is_empty() {
                epoch_losses.push(0.0);
                continue;
            } else {
                losses.into_iter().reduce(|a, b| a + b).unwrap()
            };

            let loss_val = total_loss.clone().into_scalar().to_f32();
            if loss_val.is_finite() {
                let grads = GradientsParams::from_grads(total_loss.backward(), &current_model);
                *model = optim.step(current_lr, current_model, grads);
            } else if loss_val.is_nan() || loss_val.is_infinite() {
                eprintln!(
                    "  WARNING: loss diverged at epoch {} (loss={loss_val}). Stopping.",
                    epoch + 1
                );
                epoch_losses.push(loss_val);
                break;
            }

            epoch_losses.push(loss_val);

            if (epoch + 1) % 100 == 0 || epoch == 0 {
                eprintln!(
                    "  epoch {:>5}/{}: loss={loss_val:.4} lr={current_lr:.6}",
                    epoch + 1,
                    config.epochs,
                );
            }

            // Validation checkpoint every 500 epochs (for long runs).
            if config.epochs >= 1000 && (epoch + 1) % 500 == 0 {
                let val_score = self.quick_val(model, &nf2_axioms, &nf3_axioms, nc, dim, device);
                if val_score > best_val_score {
                    best_val_score = val_score;
                    best_params = Some(save_checkpoint(model, device));
                    best_epoch = epoch + 1;
                }
            }
        }

        // Restore best checkpoint.
        if let Some(ref params) = best_params {
            eprintln!(
                "  Restoring best checkpoint from epoch {best_epoch} (val score={best_val_score:.4})"
            );
            restore_checkpoint(model, params, device);
        }

        epoch_losses
    }

    /// Quick validation: evaluate NF2+NF3 MRR on a sample of training axioms.
    fn quick_val(
        &self,
        model: &BurnElModel<B>,
        nf2_axioms: &[(usize, usize)],
        nf3_axioms: &[(usize, usize, usize)],
        nc: usize,
        dim: usize,
        device: &B::Device,
    ) -> f32 {
        let centers = extract_2d(model.concept_centers.val(), device);

        let val_n2 = 200.min(nf2_axioms.len());
        let nf2_mrr = if val_n2 > 0 {
            let mut rr_sum = 0.0f32;
            for &(sub, sup) in &nf2_axioms[..val_n2] {
                if sub >= nc || sup >= nc {
                    continue;
                }
                let query = &centers[sub * dim..(sub + 1) * dim];
                let (rank, _) = l2_rank(query, &centers, nc, dim, sup, &[sub]);
                rr_sum += 1.0 / rank as f32;
            }
            rr_sum / val_n2 as f32
        } else {
            0.0
        };

        let val_n3 = 200.min(nf3_axioms.len());
        let nf3_mrr = if val_n3 > 0 {
            let bump_vecs = extract_2d(model.bumps.val(), device);
            let role_heads = extract_2d(model.role_heads.val(), device);
            let nr = model.role_heads.val().dims()[0];
            let mut rr_sum = 0.0f32;
            for &(sub, role, filler) in &nf3_axioms[..val_n3] {
                if sub >= nc || filler >= nc || role >= nr {
                    continue;
                }
                let sub_off = sub * dim;
                let rh_off = role * dim * 2;
                let target_dist_sq = {
                    let bump_off = filler * dim;
                    let mut d = 0.0f32;
                    for i in 0..dim {
                        let bumped = centers[sub_off + i] + bump_vecs[bump_off + i];
                        let diff = bumped - role_heads[rh_off + i];
                        d += diff * diff;
                    }
                    d
                };
                let mut rank = 1usize;
                for c in 0..nc {
                    let bump_off = c * dim;
                    let mut d = 0.0f32;
                    for i in 0..dim {
                        let bumped = centers[sub_off + i] + bump_vecs[bump_off + i];
                        let diff = bumped - role_heads[rh_off + i];
                        d += diff * diff;
                    }
                    if d < target_dist_sq {
                        rank += 1;
                    }
                }
                rr_sum += 1.0 / rank as f32;
            }
            rr_sum / val_n3 as f32
        } else {
            0.0
        };

        nf2_mrr + nf3_mrr
    }

    /// Extract concept centers as a flat Vec for CPU-based evaluation.
    pub fn extract_centers(model: &BurnElModel<B>, device: &B::Device) -> Vec<f32> {
        extract_2d(model.concept_centers.val(), device)
    }

    /// Extract concept offsets (abs applied) as a flat Vec.
    pub fn extract_offsets(model: &BurnElModel<B>, device: &B::Device) -> Vec<f32> {
        extract_2d(model.concept_offsets.val().abs(), device)
    }

    /// Extract bump vectors as a flat Vec.
    pub fn extract_bumps(model: &BurnElModel<B>, device: &B::Device) -> Vec<f32> {
        extract_2d(model.bumps.val(), device)
    }

    /// Extract role heads as a flat Vec.
    pub fn extract_role_heads(model: &BurnElModel<B>, device: &B::Device) -> Vec<f32> {
        extract_2d(model.role_heads.val(), device)
    }

    /// Extract role tails as a flat Vec.
    pub fn extract_role_tails(model: &BurnElModel<B>, device: &B::Device) -> Vec<f32> {
        extract_2d(model.role_tails.val(), device)
    }

    /// Evaluate all NF types using center-distance ranking (Box2EL protocol).
    ///
    /// Returns `(nf2, nf1, nf3, nf4)` where each is `(h@1, h@10, mrr)`.
    #[allow(clippy::type_complexity)]
    pub fn evaluate(
        model: &BurnElModel<B>,
        ontology: &Ontology,
        dim: usize,
        device: &B::Device,
    ) -> (
        (f32, f32, f32),
        (f32, f32, f32),
        (f32, f32, f32),
        (f32, f32, f32),
    ) {
        let centers = extract_2d(model.concept_centers.val(), device);
        let offsets = extract_2d(model.concept_offsets.val().abs(), device);
        let bump_vecs = extract_2d(model.bumps.val(), device);
        let role_heads_data = extract_2d(model.role_heads.val(), device);
        let nc = ontology.concept_names.len();
        let nr = ontology.role_names.len();

        // Group test axioms.
        let mut nf2_test: Vec<(usize, usize)> = Vec::new();
        let mut nf1_test: Vec<(usize, usize, usize)> = Vec::new();
        let mut nf3_test: Vec<(usize, usize, usize)> = Vec::new();
        let mut nf4_test: Vec<(usize, usize, usize)> = Vec::new();

        for ax in &ontology.axioms {
            match *ax {
                Axiom::SubClassOf { sub, sup } => nf2_test.push((sub, sup)),
                Axiom::Intersection { c1, c2, target } => nf1_test.push((c1, c2, target)),
                Axiom::ExistentialRight { sub, role, filler } => {
                    nf3_test.push((sub, role, filler));
                }
                Axiom::Existential {
                    role,
                    filler,
                    target,
                } => nf4_test.push((role, filler, target)),
                _ => {}
            }
        }

        let eval_nf2 = evaluate_nf2(&centers, nc, dim, &nf2_test);
        let eval_nf1 = evaluate_nf1(&centers, &offsets, nc, dim, &nf1_test);
        let eval_nf3 = evaluate_nf3(
            &centers,
            &bump_vecs,
            &role_heads_data,
            nc,
            nr,
            dim,
            &nf3_test,
        );
        let eval_nf4 = evaluate_nf4(
            &centers,
            &bump_vecs,
            &role_heads_data,
            nc,
            nr,
            dim,
            &nf4_test,
        );

        eprintln!("  +---------+-----------------------------+");
        eprintln!("  |  NF     |  H@1     H@10     MRR       |");
        eprintln!("  +---------+-----------------------------+");
        for (label, m) in [
            ("NF2", eval_nf2),
            ("NF1", eval_nf1),
            ("NF3", eval_nf3),
            ("NF4", eval_nf4),
        ] {
            eprintln!(
                "  | {label:<7} | {:.3}    {:.3}    {:.4}    |",
                m.0, m.1, m.2,
            );
        }
        eprintln!("  +---------+-----------------------------+");

        (eval_nf2, eval_nf1, eval_nf3, eval_nf4)
    }
}

// ---------------------------------------------------------------------------
// Free functions: loss computation
// ---------------------------------------------------------------------------

/// Get concept boxes (center, abs(offset)) for given IDs.
fn concept_boxes<B: Backend>(
    model: &BurnElModel<B>,
    ids: &Tensor<B, 1, Int>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let centers = model.concept_centers.val().select(0, ids.clone());
    let offsets = model.concept_offsets.val().select(0, ids.clone()).abs();
    (centers, offsets)
}

/// Get role box (center, abs(offset)) from packed `[nr, dim*2]` tensor.
fn role_box<B: Backend>(
    model: &BurnElModel<B>,
    ids: &Tensor<B, 1, Int>,
    head: bool,
    dim: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let embed = if head {
        model.role_heads.val().select(0, ids.clone())
    } else {
        model.role_tails.val().select(0, ids.clone())
    };
    let bs = embed.dims()[0];
    let centers = embed.clone().slice([0..bs, 0..dim]);
    let offsets = embed.slice([0..bs, dim..dim * 2]).abs();
    (centers, offsets)
}

/// Inclusion loss: `sqrt(sum(relu(|c_a - c_b| + o_a - o_b - margin)^2, dim=-1) + eps)`.
///
/// Returns `[bs, 1]` tensor of L2 norms.
fn inclusion_loss<B: Backend>(
    centers_a: Tensor<B, 2>,
    offsets_a: Tensor<B, 2>,
    centers_b: Tensor<B, 2>,
    offsets_b: Tensor<B, 2>,
    margin: f32,
) -> Tensor<B, 2> {
    let diffs = (centers_a - centers_b).abs();
    let violation = (diffs + offsets_a - offsets_b - margin).clamp_min(0.0);
    (violation.powf_scalar(2.0).sum_dim(1) + 1e-8).sqrt()
}

/// Negative loss: `sqrt(sum(relu(|c_a - c_b| - o_a - o_b + margin)^2, dim=-1) + eps)`.
///
/// Returns `[bs, 1]` tensor.
fn neg_loss_fn<B: Backend>(
    centers_a: Tensor<B, 2>,
    offsets_a: Tensor<B, 2>,
    centers_b: Tensor<B, 2>,
    offsets_b: Tensor<B, 2>,
    margin: f32,
) -> Tensor<B, 2> {
    let diffs = (centers_a - centers_b).abs();
    let gap = (diffs - offsets_a - offsets_b + margin).clamp_min(0.0);
    (gap.powf_scalar(2.0).sum_dim(1) + 1e-8).sqrt()
}

// ---------------------------------------------------------------------------
// Free functions: checkpoint
// ---------------------------------------------------------------------------

/// Save model parameters as flat Vec<f32> for checkpoint.
fn save_checkpoint<B: Backend>(
    model: &BurnElModel<B>,
    device: &B::Device,
) -> Vec<(String, Vec<f32>, Vec<usize>)> {
    vec![
        (
            "concept_centers".into(),
            extract_2d(model.concept_centers.val(), device),
            model.concept_centers.val().dims().to_vec(),
        ),
        (
            "concept_offsets".into(),
            extract_2d(model.concept_offsets.val(), device),
            model.concept_offsets.val().dims().to_vec(),
        ),
        (
            "bumps".into(),
            extract_2d(model.bumps.val(), device),
            model.bumps.val().dims().to_vec(),
        ),
        (
            "role_heads".into(),
            extract_2d(model.role_heads.val(), device),
            model.role_heads.val().dims().to_vec(),
        ),
        (
            "role_tails".into(),
            extract_2d(model.role_tails.val(), device),
            model.role_tails.val().dims().to_vec(),
        ),
    ]
}

/// Restore model parameters from checkpoint.
fn restore_checkpoint<B: AutodiffBackend>(
    model: &mut BurnElModel<B>,
    params: &[(String, Vec<f32>, Vec<usize>)],
    device: &B::Device,
) {
    for (name, data, shape) in params {
        let t = Tensor::<B, 1>::from_data(data.as_slice(), device)
            .reshape([shape[0], shape[1]])
            .require_grad();
        let param = Param::initialized(ParamId::new(), t);
        match name.as_str() {
            "concept_centers" => model.concept_centers = param,
            "concept_offsets" => model.concept_offsets = param,
            "bumps" => model.bumps = param,
            "role_heads" => model.role_heads = param,
            "role_tails" => model.role_tails = param,
            _ => {}
        }
    }
}

/// Extract a 2D tensor to a flat Vec<f32>.
fn extract_2d<B: Backend>(tensor: Tensor<B, 2>, device: &B::Device) -> Vec<f32> {
    let _ = device; // device is needed for some backends
    let data = tensor.into_data();
    data.as_slice::<f32>()
        .expect("tensor should be f32")
        .to_vec()
}

// ---------------------------------------------------------------------------
// Free functions: CPU evaluation (matches CandleElTrainer protocol)
// ---------------------------------------------------------------------------

/// L2 rank of `target` among all concepts.
fn l2_rank(
    query: &[f32],
    centers: &[f32],
    nc: usize,
    dim: usize,
    target: usize,
    exclude: &[usize],
) -> (usize, f32) {
    let target_dist_sq = {
        let off = target * dim;
        let mut d = 0.0f32;
        for i in 0..dim {
            let diff = query[i] - centers[off + i];
            d += diff * diff;
        }
        d
    };

    let mut rank = 1usize;
    for c in 0..nc {
        if c == target || exclude.contains(&c) {
            continue;
        }
        let off = c * dim;
        let mut d = 0.0f32;
        for i in 0..dim {
            let diff = query[i] - centers[off + i];
            d += diff * diff;
        }
        if d < target_dist_sq {
            rank += 1;
        }
    }
    (rank, target_dist_sq.sqrt())
}

fn metrics_from_ranks(ranks: &[usize]) -> (f32, f32, f32) {
    if ranks.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = ranks.len() as f32;
    let h1 = ranks.iter().filter(|&&r| r == 1).count() as f32 / n;
    let h10 = ranks.iter().filter(|&&r| r <= 10).count() as f32 / n;
    let mrr: f32 = ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / n;
    (h1, h10, mrr)
}

fn evaluate_nf2(
    centers: &[f32],
    nc: usize,
    dim: usize,
    test_axioms: &[(usize, usize)],
) -> (f32, f32, f32) {
    let mut ranks = Vec::with_capacity(test_axioms.len());
    for &(sub, sup) in test_axioms {
        if sub >= nc || sup >= nc {
            continue;
        }
        let query = &centers[sub * dim..(sub + 1) * dim];
        let (rank, _) = l2_rank(query, centers, nc, dim, sup, &[sub]);
        ranks.push(rank);
    }
    metrics_from_ranks(&ranks)
}

fn evaluate_nf1(
    centers: &[f32],
    offsets: &[f32],
    nc: usize,
    dim: usize,
    test_axioms: &[(usize, usize, usize)],
) -> (f32, f32, f32) {
    let mut ranks = Vec::with_capacity(test_axioms.len());
    let mut inter_center = vec![0.0f32; dim];
    for &(c1, c2, d) in test_axioms {
        if c1 >= nc || c2 >= nc || d >= nc {
            continue;
        }
        let c1_off = c1 * dim;
        let c2_off = c2 * dim;
        for i in 0..dim {
            let min1 = centers[c1_off + i] - offsets[c1_off + i];
            let max1 = centers[c1_off + i] + offsets[c1_off + i];
            let min2 = centers[c2_off + i] - offsets[c2_off + i];
            let max2 = centers[c2_off + i] + offsets[c2_off + i];
            let inter_min = min1.max(min2);
            let inter_max = max1.min(max2).max(inter_min);
            inter_center[i] = (inter_min + inter_max) / 2.0;
        }
        let (rank, _) = l2_rank(&inter_center, centers, nc, dim, d, &[c1, c2]);
        ranks.push(rank);
    }
    metrics_from_ranks(&ranks)
}

fn evaluate_nf3(
    centers: &[f32],
    bump_vecs: &[f32],
    role_heads: &[f32],
    nc: usize,
    nr: usize,
    dim: usize,
    test_axioms: &[(usize, usize, usize)],
) -> (f32, f32, f32) {
    let mut ranks = Vec::with_capacity(test_axioms.len());
    for &(sub, role, filler) in test_axioms {
        if sub >= nc || filler >= nc || role >= nr {
            continue;
        }
        let sub_off = sub * dim;
        let rh_off = role * dim * 2;

        let target_dist_sq = {
            let bump_off = filler * dim;
            let mut d = 0.0f32;
            for i in 0..dim {
                let bumped = centers[sub_off + i] + bump_vecs[bump_off + i];
                let diff = bumped - role_heads[rh_off + i];
                d += diff * diff;
            }
            d
        };

        let mut rank = 1usize;
        for c in 0..nc {
            let bump_off = c * dim;
            let mut d = 0.0f32;
            for i in 0..dim {
                let bumped = centers[sub_off + i] + bump_vecs[bump_off + i];
                let diff = bumped - role_heads[rh_off + i];
                d += diff * diff;
            }
            if d < target_dist_sq {
                rank += 1;
            }
        }
        ranks.push(rank);
    }
    metrics_from_ranks(&ranks)
}

fn evaluate_nf4(
    centers: &[f32],
    bump_vecs: &[f32],
    role_heads: &[f32],
    nc: usize,
    nr: usize,
    dim: usize,
    test_axioms: &[(usize, usize, usize)],
) -> (f32, f32, f32) {
    let mut ranks = Vec::with_capacity(test_axioms.len());
    let mut query = vec![0.0f32; dim];
    for &(role, filler, target) in test_axioms {
        if filler >= nc || target >= nc || role >= nr {
            continue;
        }
        let rh_off = role * dim * 2;
        let bump_off = filler * dim;
        for i in 0..dim {
            query[i] = role_heads[rh_off + i] - bump_vecs[bump_off + i];
        }
        let (rank, _) = l2_rank(&query, centers, nc, dim, target, &[]);
        ranks.push(rank);
    }
    metrics_from_ranks(&ranks)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_burn_el_model_creates() {
        let device = Default::default();
        let model = BurnElTrainer::<TestBackend>::init_model(100, 5, 32, &device);
        assert_eq!(model.concept_centers.val().dims(), [100, 32]);
        assert_eq!(model.bumps.val().dims(), [100, 32]);
        assert_eq!(model.role_heads.val().dims(), [5, 64]);
    }

    #[test]
    fn test_burn_el_fit_runs() {
        let device = Default::default();
        let trainer = BurnElTrainer::<TestBackend>::new();
        let mut model = BurnElTrainer::<TestBackend>::init_model(20, 3, 16, &device);

        let mut ont = Ontology::new();
        for i in 0..20 {
            ont.concept(&format!("C{i}"));
        }
        for i in 0..3 {
            ont.role(&format!("R{i}"));
        }
        ont.axioms.push(Axiom::SubClassOf { sub: 0, sup: 1 });
        ont.axioms.push(Axiom::SubClassOf { sub: 2, sup: 3 });
        ont.axioms.push(Axiom::Intersection {
            c1: 0,
            c2: 2,
            target: 4,
        });
        ont.axioms.push(Axiom::ExistentialRight {
            sub: 5,
            role: 0,
            filler: 6,
        });
        ont.axioms.push(Axiom::Existential {
            role: 1,
            filler: 7,
            target: 8,
        });

        let config = BurnElConfig {
            dim: 16,
            epochs: 50,
            batch_size: 4,
            negative_samples: 1,
            lr: 0.01,
            reg_factor: 0.0,
            ..Default::default()
        };

        let losses = trainer.fit(&mut model, &ont, &config, &device);
        assert_eq!(losses.len(), 50);
        assert!(losses[0].is_finite());
        assert!(
            losses.last().unwrap() < &losses[0],
            "loss should decrease: first={}, last={}",
            losses[0],
            losses.last().unwrap()
        );
    }

    #[test]
    fn test_burn_el_eval_works() {
        let device = Default::default();
        let trainer = BurnElTrainer::<TestBackend>::new();
        let mut model = BurnElTrainer::<TestBackend>::init_model(20, 2, 16, &device);

        let mut ont = Ontology::new();
        for i in 0..20 {
            ont.concept(&format!("C{i}"));
        }
        ont.role("R0");
        ont.role("R1");
        for i in 0..15 {
            ont.axioms.push(Axiom::SubClassOf {
                sub: i,
                sup: (i + 1) % 20,
            });
        }

        let config = BurnElConfig {
            dim: 16,
            epochs: 200,
            batch_size: 8,
            negative_samples: 2,
            lr: 0.01,
            reg_factor: 0.0,
            ..Default::default()
        };

        let _losses = trainer.fit(&mut model, &ont, &config, &device);

        let (nf2, _, _, _) = BurnElTrainer::<TestBackend>::evaluate(&model, &ont, 16, &device);
        assert!(
            nf2.2 > 0.0,
            "MRR should be positive on training data, got {}",
            nf2.2
        );
        eprintln!(
            "BurnElTrainer eval: H@1={:.3} H@10={:.3} MRR={:.4}",
            nf2.0, nf2.1, nf2.2
        );
    }

    #[test]
    fn test_burn_el_checkpoint_roundtrip() {
        let device = Default::default();
        let model = BurnElTrainer::<TestBackend>::init_model(10, 2, 8, &device);

        let saved = save_checkpoint(&model, &device);
        let mut restored = BurnElTrainer::<TestBackend>::init_model(10, 2, 8, &device);
        restore_checkpoint(&mut restored, &saved, &device);

        let orig_centers = extract_2d(model.concept_centers.val(), &device);
        let rest_centers = extract_2d(restored.concept_centers.val(), &device);
        for (a, b) in orig_centers.iter().zip(rest_centers.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "checkpoint roundtrip failed: {a} != {b}"
            );
        }
    }

    #[test]
    fn test_burn_el_all_nf_types() {
        let device = Default::default();
        let trainer = BurnElTrainer::<TestBackend>::new();

        let mut ont = Ontology::new();
        // Create a small hierarchy.
        let animal = ont.concept("Animal");
        let dog = ont.concept("Dog");
        let cat = ont.concept("Cat");
        let pet = ont.concept("Pet");
        let living = ont.concept("Living");
        for i in 5..15 {
            ont.concept(&format!("C{i}"));
        }
        let has_part = ont.role("hasPart");
        let part_of = ont.role("partOf");

        // NF2: Dog ⊑ Animal, Cat ⊑ Animal
        ont.axioms.push(Axiom::SubClassOf {
            sub: dog,
            sup: animal,
        });
        ont.axioms.push(Axiom::SubClassOf {
            sub: cat,
            sup: animal,
        });
        // NF1: Dog ⊓ Pet ⊑ Living
        ont.axioms.push(Axiom::Intersection {
            c1: dog,
            c2: pet,
            target: living,
        });
        // NF3: Dog ⊑ ∃hasPart.Animal
        ont.axioms.push(Axiom::ExistentialRight {
            sub: dog,
            role: has_part,
            filler: animal,
        });
        // NF4: ∃partOf.Dog ⊑ Animal
        ont.axioms.push(Axiom::Existential {
            role: part_of,
            filler: dog,
            target: animal,
        });
        // DISJ: Dog ⊓ Cat ⊑ ⊥
        ont.axioms.push(Axiom::Disjoint { a: dog, b: cat });

        let mut model = BurnElTrainer::<TestBackend>::init_model(15, 2, 16, &device);

        let config = BurnElConfig {
            dim: 16,
            epochs: 100,
            batch_size: 4,
            negative_samples: 2,
            lr: 0.01,
            reg_factor: 0.1,
            ..Default::default()
        };

        let losses = trainer.fit(&mut model, &ont, &config, &device);
        assert!(losses[0].is_finite());
        assert!(losses.last().unwrap().is_finite());

        let (nf2, nf1, nf3, nf4) =
            BurnElTrainer::<TestBackend>::evaluate(&model, &ont, 16, &device);
        eprintln!(
            "All NF types: NF2 MRR={:.3}, NF1 MRR={:.3}, NF3 MRR={:.3}, NF4 MRR={:.3}",
            nf2.2, nf1.2, nf3.2, nf4.2
        );
    }
}
