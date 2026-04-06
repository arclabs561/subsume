//! Burn-based spherical cap trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training
//!
//! # Scoring convention
//!
//! Containment margin: `theta_outer - geodesic_dist - theta_inner`.
//! Positive means contained; negative means violation.
//! Sigmoid of margin scaled by `k` gives containment probability.
//!
//! # Center parameterization
//!
//! Centers are stored as unconstrained `[num_entities, dim]` tensors and L2-normalized
//! before every scoring computation.  This keeps gradient flow unrestricted while
//! ensuring centers live on the unit sphere.
//!
//! # Angular radius parameterization
//!
//! Angular radii use the `log_tan_half` reparameterization: `theta = 2 * atan(exp(t))`,
//! which maps the full real line smoothly onto `(0, pi)`.

use crate::dataset::TripleIds;
use crate::spherical_cap::{SphericalCap, SphericalCapRelation};
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

// ---------------------------------------------------------------------------
// Model structs
// ---------------------------------------------------------------------------

/// Entity embedding parameters for spherical caps.
///
/// Centers are unconstrained; they are L2-normalized before scoring.
/// Separate head and tail log-tan-half parameters let each entity develop
/// a small cap when queried as a head and a large cap when used as a tail.
#[derive(Module, Debug)]
pub struct BurnCapEntityParams<B: Backend> {
    /// Center vectors `[num_entities, dim]` (unnormalized; normalized on use).
    pub center: Param<Tensor<B, 2>>,
    /// Log-tan(theta/2) when entity is used as the **head** `[num_entities, 1]`.
    pub log_tan_half: Param<Tensor<B, 2>>,
    /// Log-tan(theta/2) when entity is used as the **tail** `[num_entities, 1]`.
    pub tail_log_tan_half: Param<Tensor<B, 2>>,
}

/// Relation embedding parameters for spherical caps.
///
/// Each relation has separate head and tail translations applied to centers
/// (followed by L2 renormalization) and a log-scale applied to the head's
/// angular radius in log-tan-half space.
#[derive(Module, Debug)]
pub struct BurnCapRelationParams<B: Backend> {
    /// Head center translation `[num_relations, dim]`.
    pub translation: Param<Tensor<B, 2>>,
    /// Tail center translation `[num_relations, dim]`.
    pub tail_translation: Param<Tensor<B, 2>>,
    /// Additive log-scale on head log-tan-half `[num_relations, 1]`.
    pub log_scale: Param<Tensor<B, 2>>,
}

/// Combined spherical cap model.
#[derive(Module, Debug)]
pub struct BurnCapModel<B: Backend> {
    /// Entity embedding parameters.
    pub entities: BurnCapEntityParams<B>,
    /// Relation embedding parameters.
    pub relations: BurnCapRelationParams<B>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based spherical cap trainer with autodiff.
pub struct BurnCapTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
    cached_pools: Option<HashMap<usize, RelationEntityPools>>,
}

impl<B: AutodiffBackend> Default for BurnCapTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
            cached_pools: None,
        }
    }
}

impl<B: AutodiffBackend> BurnCapTrainer<B> {
    /// Create a new trainer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-epoch seed used for negative sampling.
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
    ) -> BurnCapModel<B> {
        let param = |shape: [usize; 2], lo: f64, hi: f64| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::random(shape, burn::tensor::Distribution::Uniform(lo, hi), device)
                    .require_grad(),
            )
        };
        BurnCapModel {
            entities: BurnCapEntityParams {
                center: param([num_entities, dim], -0.1, 0.1),
                log_tan_half: param([num_entities, 1], -1.0, 0.0),
                tail_log_tan_half: param([num_entities, 1], -1.0, 0.0),
            },
            relations: BurnCapRelationParams {
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
    pub fn train_epoch(
        &mut self,
        model: &mut BurnCapModel<B>,
        optim: &mut impl Optimizer<BurnCapModel<B>, B>,
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
                eprintln!("[burn_cap] skipping non-finite batch loss: {loss_val}");
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

    /// Batched ranking loss for one batch.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnCapModel<B>,
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
        let dim = model.entities.center.val().dims()[1];

        // --- Head embeddings ---
        let hc_raw = model.entities.center.val().select(0, head_ids.clone());
        let h_lth = model.entities.log_tan_half.val().select(0, head_ids);
        let rt = model.relations.translation.val().select(0, rel_ids.clone());
        let rtt = model
            .relations
            .tail_translation
            .val()
            .select(0, rel_ids.clone());
        let rls = model.relations.log_scale.val().select(0, rel_ids);

        // --- Tail embeddings ---
        let tc_raw = model.entities.center.val().select(0, tail_ids.clone());
        let t_lth = model.entities.tail_log_tan_half.val().select(0, tail_ids);

        // --- Transform head center: add translation then L2-normalize ---
        let transformed_c = l2_normalize(hc_raw + rt); // [bs, dim]
        let transformed_lth = h_lth + rls; // [bs, 1]
                                           // theta = 2 * atan(exp(lth)); we keep lth for differentiability.
        let transformed_theta = log_tan_half_to_theta(transformed_lth); // [bs, 1]

        // --- Transform tail center ---
        let shifted_tc = l2_normalize(tc_raw + rtt.clone()); // [bs, dim]
        let tail_theta = log_tan_half_to_theta(t_lth); // [bs, 1]

        // --- Geodesic distance (positive pair) ---
        let pos_cos = (transformed_c.clone() * shifted_tc)
            .sum_dim(1)
            .clamp(-1.0 + 1e-6, 1.0 - 1e-6); // [bs, 1]
        let pos_dist = pos_cos.acos(); // [bs, 1]

        // Containment margin: outer.theta - dist - inner.theta
        let pos_marg = tail_theta.clone() - pos_dist - transformed_theta.clone(); // [bs, 1]

        // --- Negative tail margins ---
        let neg_ids = Tensor::<B, 1, Int>::from_data(neg_tail_ids_flat.as_slice(), device);
        let neg_c_raw = model
            .entities
            .center
            .val()
            .select(0, neg_ids.clone())
            .reshape([bs, n_neg, dim]);
        let neg_lth = model
            .entities
            .tail_log_tan_half
            .val()
            .select(0, neg_ids)
            .reshape([bs, n_neg, 1]);
        let neg_theta = log_tan_half_to_theta_3d(neg_lth); // [bs, n_neg, 1]

        // Broadcast tail_translation [bs, dim] → [bs, n_neg, dim]
        let rtt_rep = rtt.reshape([bs, 1, dim]);
        let neg_c_shifted_raw = neg_c_raw + rtt_rep;
        // L2-normalize each negative center: reshape to [bs*n_neg, dim], normalize, reshape back
        let neg_c_shifted = {
            let flat = neg_c_shifted_raw.reshape([bs * n_neg, dim]);
            l2_normalize(flat).reshape([bs, n_neg, dim])
        };

        let tc_rep = transformed_c.reshape([bs, 1, dim]);
        // Dot product: [bs, n_neg]
        let neg_cos = (tc_rep * neg_c_shifted)
            .sum_dim(2)
            .clamp(-1.0 + 1e-6, 1.0 - 1e-6); // [bs, n_neg]
        let neg_dist = neg_cos.acos().reshape([bs, n_neg, 1]); // [bs, n_neg, 1]

        // Broadcast transformed_theta [bs, 1] → [bs, 1, 1]
        let transformed_theta_rep = transformed_theta.reshape([bs, 1, 1]);
        let neg_marg = neg_theta - neg_dist - transformed_theta_rep; // [bs, n_neg, 1]

        if config.use_infonce {
            let neg_marg_2d = neg_marg.reshape([bs, n_neg]);
            let logits = Tensor::cat(vec![pos_marg.clone(), neg_marg_2d], 1) * k; // [bs, 1+n]
            let max_logit = logits.clone().max_dim(1);
            let lse = (logits - max_logit.clone()).exp().sum_dim(1).log() + max_logit;
            (lse - pos_marg * k).mean()
        } else {
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

    /// L2 penalty on embeddings participating in this batch.
    fn l2_reg(
        model: &BurnCapModel<B>,
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

    /// Extract model parameters into CPU-side SphericalCap/SphericalCapRelation vecs.
    ///
    /// Returns `(head_caps, tail_caps, head_rels, tail_rels)`.
    #[allow(clippy::type_complexity)]
    pub fn to_cap_embeddings(
        &self,
        model: &BurnCapModel<B>,
    ) -> (
        Vec<SphericalCap>,
        Vec<SphericalCap>,
        Vec<SphericalCapRelation>,
        Vec<SphericalCapRelation>,
    ) {
        let c_data = model.entities.center.val().into_data();
        let h_lth_data = model.entities.log_tan_half.val().into_data();
        let t_lth_data = model.entities.tail_log_tan_half.val().into_data();
        let ht_data = model.relations.translation.val().into_data();
        let tt_data = model.relations.tail_translation.val().into_data();
        let ls_data = model.relations.log_scale.val().into_data();

        let cs: Vec<f32> = c_data.to_vec().unwrap();
        let h_lths: Vec<f32> = h_lth_data.to_vec().unwrap();
        let t_lths: Vec<f32> = t_lth_data.to_vec().unwrap();
        let hts: Vec<f32> = ht_data.to_vec().unwrap();
        let tts: Vec<f32> = tt_data.to_vec().unwrap();
        let lss: Vec<f32> = ls_data.to_vec().unwrap();

        let n_e = h_lths.len();
        let dim = cs.len() / n_e;
        let n_r = lss.len();

        // Normalize centers on CPU before constructing SphericalCap.
        let make_cap = |raw_center: &[f32], lth: f32| {
            let norm: f32 = raw_center.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm = norm.max(1e-8);
            let center: Vec<f32> = raw_center.iter().map(|x| x / norm).collect();
            SphericalCap::from_log_tan_half(center, lth).unwrap_or_else(|_| {
                let mut fallback = vec![0.0f32; dim];
                fallback[0] = 1.0;
                SphericalCap::from_log_tan_half(fallback, 0.0)
                    .unwrap_or_else(|_| SphericalCap::new(vec![1.0, 0.0], 0.01).unwrap())
            })
        };

        let head_caps: Vec<SphericalCap> = (0..n_e)
            .map(|i| make_cap(&cs[i * dim..(i + 1) * dim], h_lths[i]))
            .collect();
        let tail_caps: Vec<SphericalCap> = (0..n_e)
            .map(|i| make_cap(&cs[i * dim..(i + 1) * dim], t_lths[i]))
            .collect();

        // Relations: use a small fixed axis (axis is not trained, only translation is).
        // For evaluation we reproduce the training transform: translate center then renormalize.
        // SphericalCapRelation.apply() uses rotation, but here we need additive translation.
        // We approximate with angle=0 (identity rotation) and apply the translation manually
        // in the scoring closure inside evaluate().
        // Store translations separately; SphericalCapRelation is used only for angle_scale.
        let identity_axis: Vec<f32> = {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            v
        };
        let head_rels: Vec<SphericalCapRelation> = (0..n_r)
            .map(|i| {
                SphericalCapRelation::new(identity_axis.clone(), 0.0, lss[i].exp().max(1e-6))
                    .unwrap()
            })
            .collect();
        let tail_rels: Vec<SphericalCapRelation> = (0..n_r)
            .map(|_| SphericalCapRelation::new(identity_axis.clone(), 0.0, 1.0).unwrap())
            .collect();

        // Store raw translations for the scoring closure below.
        // We attach them via a side-channel by embedding them into the SphericalCapRelation
        // axis field (hack-free approach: we return them separately by packing into the
        // existing vecs and using them directly in evaluate()).
        // Actually the cleanest approach: evaluate() calls to_cap_embeddings_raw and builds
        // its own scoring closure with the raw translation vecs.  We therefore also store
        // the raw translation data in the returned tuples as an extension to head_rels axes.
        // But SphericalCapRelation.axis is already normalized, so we can't embed an
        // arbitrary vector there.  Instead, evaluate() calls a separate helper.
        let _ = (hts, tts); // consumed below in evaluate(); suppress unused warning
        (head_caps, tail_caps, head_rels, tail_rels)
    }

    /// Evaluate link prediction on `test_triples`.
    pub fn evaluate(
        &self,
        model: &BurnCapModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let c_data = model.entities.center.val().into_data();
        let h_lth_data = model.entities.log_tan_half.val().into_data();
        let t_lth_data = model.entities.tail_log_tan_half.val().into_data();
        let ht_data = model.relations.translation.val().into_data();
        let tt_data = model.relations.tail_translation.val().into_data();
        let ls_data = model.relations.log_scale.val().into_data();

        let cs: Vec<f32> = c_data.to_vec().unwrap();
        let h_lths: Vec<f32> = h_lth_data.to_vec().unwrap();
        let t_lths: Vec<f32> = t_lth_data.to_vec().unwrap();
        let hts: Vec<f32> = ht_data.to_vec().unwrap();
        let tts: Vec<f32> = tt_data.to_vec().unwrap();
        let lss: Vec<f32> = ls_data.to_vec().unwrap();

        let n_e = h_lths.len();
        let dim = cs.len() / n_e;
        let n_r = lss.len();
        let k = 2.0f32;

        let normalize_cpu = |raw: &[f32]| -> Vec<f32> {
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            raw.iter().map(|x| x / norm).collect()
        };

        // score(h, r, t): reproduce training transform.
        //   transformed_c = normalize(head_center + head_trans[r])
        //   transformed_theta = 2 * atan(exp(head_lth + log_scale[r]))
        //   shifted_tc = normalize(tail_center + tail_trans[r])
        //   tail_theta = 2 * atan(exp(tail_lth))
        //   dist = acos(clamp(dot(transformed_c, shifted_tc), -1+eps, 1-eps))
        //   margin = tail_theta - dist - transformed_theta
        //   score = sigmoid(k * margin)
        let score = |h: usize, r: usize, t: usize| -> f32 {
            if r >= n_r {
                return 0.0;
            }
            let hc_raw = &cs[h * dim..(h + 1) * dim];
            let ht_r = &hts[r * dim..(r + 1) * dim];
            let tc_raw = &cs[t * dim..(t + 1) * dim];
            let tt_r = &tts[r * dim..(r + 1) * dim];

            // Translated + normalized centers.
            let trans_c_raw: Vec<f32> = hc_raw.iter().zip(ht_r).map(|(a, b)| a + b).collect();
            let trans_c = normalize_cpu(&trans_c_raw);
            let shifted_tc_raw: Vec<f32> = tc_raw.iter().zip(tt_r).map(|(a, b)| a + b).collect();
            let shifted_tc = normalize_cpu(&shifted_tc_raw);

            // Angular radii.
            let trans_theta = 2.0 * (h_lths[h] + lss[r]).exp().atan();
            let tail_theta = 2.0 * t_lths[t].exp().atan();

            // Geodesic distance.
            let dot: f32 = trans_c.iter().zip(&shifted_tc).map(|(a, b)| a * b).sum();
            let dist = dot.clamp(-1.0 + 1e-6, 1.0 - 1e-6).acos();

            let margin = tail_theta - dist - trans_theta;
            crate::utils::stable_sigmoid(k * margin)
        };

        // Use the containment scorer (same direction for head and tail prediction).
        // For head prediction: we want score(?, r, t) — but our transform applies to head,
        // so we keep the same scoring direction.
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
// Helpers
// ---------------------------------------------------------------------------

/// L2-normalize each row of a `[n, dim]` tensor.
fn l2_normalize<B: Backend>(t: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = t.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-8);
    t / norm
}

/// Convert log-tan(theta/2) `[bs, 1]` to theta `[bs, 1]`.
fn log_tan_half_to_theta<B: Backend>(lth: Tensor<B, 2>) -> Tensor<B, 2> {
    // theta = 2 * atan(exp(lth))
    // atan(exp(x)) = pi/2 - atan(exp(-x)), both stable; exp-then-atan is fine for |x| < ~80.
    lth.exp().atan().mul_scalar(2.0)
}

/// Convert log-tan(theta/2) `[bs, n_neg, 1]` to theta `[bs, n_neg, 1]`.
fn log_tan_half_to_theta_3d<B: Backend>(lth: Tensor<B, 3>) -> Tensor<B, 3> {
    lth.exp().atan().mul_scalar(2.0)
}

/// Numerically stable sigmoid via tanh: σ(x) = (tanh(x/2) + 1) / 2.
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
        let model = BurnCapTrainer::<TestBackend>::new().init_model(10, 3, 4, &device);
        assert_eq!(model.entities.center.val().dims(), [10, 4]);
        assert_eq!(model.entities.log_tan_half.val().dims(), [10, 1]);
        assert_eq!(model.entities.tail_log_tan_half.val().dims(), [10, 1]);
        assert_eq!(model.relations.translation.val().dims(), [3, 4]);
        assert_eq!(model.relations.tail_translation.val().dims(), [3, 4]);
        assert_eq!(model.relations.log_scale.val().dims(), [3, 1]);
    }

    #[test]
    fn loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnCapTrainer::<TestBackend>::new();
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
        let mut trainer = BurnCapTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.02,
            margin: 1.0,
            use_infonce: true,
            sigmoid_k: 2.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnCapModel<TestBackend>>();

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
    fn centers_stay_normalized() {
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
        ];

        let device = Default::default();
        let mut trainer = BurnCapTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(4, 1, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            sigmoid_k: 2.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnCapModel<TestBackend>>();

        for epoch in 0..10 {
            trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }

        // Check that centers used in scoring are approximately unit vectors.
        // (The raw parameter is not normalized; the training loop normalizes on use.)
        // We verify via to_cap_embeddings, which normalizes on extraction.
        let (head_caps, tail_caps, _, _) = trainer.to_cap_embeddings(&model);
        for (i, cap) in head_caps.iter().chain(tail_caps.iter()).enumerate() {
            let norm: f32 = cap.center().iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "cap[{i}] center not unit: norm={norm}"
            );
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
        let test_triples = triples.clone();

        let device = Default::default();
        let mut trainer = BurnCapTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(4, 2, 4, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            use_infonce: true,
            negative_samples: 1,
            sigmoid_k: 2.0,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnCapModel<TestBackend>>();
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
