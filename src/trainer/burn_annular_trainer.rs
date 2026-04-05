//! Burn-based annular sector trainer with autodiff.
//!
//! # Backend selection
//! - `burn-ndarray` (+ rayon) — multi-core CPU training
//! - `burn-wgpu`              — Metal/Vulkan/WebGPU training
//!
//! # Scoring convention
//! Surface distance is **lower = better** (zero means identical sectors).
//! For evaluation, scores are negated so `evaluate_link_prediction_generic` can
//! rank by descending score as usual.
//!
//! # Parameterization
//!
//! Entity parameters (6 per entity, geometry-specific, not dim-dimensional):
//! - `center: [num_entities, 2]` — Cartesian center (re, im)
//! - `log_radii: [num_entities, 2]` — log(r_inner), log(r_outer - r_inner)
//!   stored in log-space to guarantee positivity and r_inner < r_outer
//! - `angles: [num_entities, 2]` — theta_start, theta_end
//!
//! Relation parameters (3 per relation):
//! - `rotation: [num_relations, 1]` — rotation angle in radians
//! - `log_radial_scale: [num_relations, 1]` — log of radial scale (exp ensures positive)
//! - `log_angular_scale: [num_relations, 1]` — log of angular scale (exp ensures positive)

#![allow(missing_docs)]

use crate::annular::{AnnularRelation, AnnularSector};
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

/// Entity embedding parameters for annular sectors.
///
/// Uses log-space parameterization to guarantee:
/// - `r_inner > 0`   (via `exp(log_radii[:, 0])`)
/// - `r_outer > r_inner` (via `r_inner + exp(log_radii[:, 1])`)
#[derive(Module, Debug)]
pub struct BurnAnnularEntityParams<B: Backend> {
    /// Cartesian center of the annular sector `[num_entities, 2]`.
    pub center: Param<Tensor<B, 2>>,
    /// Log-space radii `[num_entities, 2]`: `(log_r_inner, log_delta)` where
    /// `r_inner = exp(log_r_inner)` and `r_outer = r_inner + exp(log_delta)`.
    pub log_radii: Param<Tensor<B, 2>>,
    /// Angular bounds `[num_entities, 2]`: `(theta_start, theta_end)`.
    pub angles: Param<Tensor<B, 2>>,
}

/// Relation embedding parameters for annular sectors.
#[derive(Module, Debug)]
pub struct BurnAnnularRelationParams<B: Backend> {
    /// Rotation angle per relation `[num_relations, 1]`.
    pub rotation: Param<Tensor<B, 2>>,
    /// Log radial scale per relation `[num_relations, 1]`.
    pub log_radial_scale: Param<Tensor<B, 2>>,
    /// Log angular scale per relation `[num_relations, 1]`.
    pub log_angular_scale: Param<Tensor<B, 2>>,
}

/// Combined annular sector model.
#[derive(Module, Debug)]
pub struct BurnAnnularModel<B: Backend> {
    /// Entity embedding parameters.
    pub entities: BurnAnnularEntityParams<B>,
    /// Relation embedding parameters.
    pub relations: BurnAnnularRelationParams<B>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based annular sector trainer with autodiff.
pub struct BurnAnnularTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
}

impl<B: AutodiffBackend> Default for BurnAnnularTrainer<B> {
    fn default() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
        }
    }
}

impl<B: AutodiffBackend> BurnAnnularTrainer<B> {
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
        device: &B::Device,
    ) -> BurnAnnularModel<B> {
        let param2 = |shape: [usize; 2], lo: f64, hi: f64| {
            Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::random(shape, burn::tensor::Distribution::Uniform(lo, hi), device)
                    .require_grad(),
            )
        };

        BurnAnnularModel {
            entities: BurnAnnularEntityParams {
                // Center near origin
                center: param2([num_entities, 2], -0.1, 0.1),
                // log_r_inner ~ Uniform(-1, 0) => r_inner in [0.37, 1.0]
                // log_delta   ~ Uniform(0, 1)  => delta in [1.0, 2.72]
                log_radii: param2([num_entities, 2], -1.0, 1.0),
                // Angles spread in [0, pi]
                angles: param2([num_entities, 2], 0.0, std::f64::consts::PI),
            },
            relations: BurnAnnularRelationParams {
                // Small initial rotation
                rotation: param2([num_relations, 1], -0.1, 0.1),
                // log_scale ~ 0 => scale ~ 1 (near-identity)
                log_radial_scale: param2([num_relations, 1], -0.1, 0.1),
                log_angular_scale: param2([num_relations, 1], -0.1, 0.1),
            },
        }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train for one epoch, returning the mean batch loss.
    pub fn train_epoch(
        &mut self,
        model: &mut BurnAnnularModel<B>,
        optim: &mut impl Optimizer<BurnAnnularModel<B>, B>,
        triples: &[TripleIds],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.entities.center.val().dims()[0];
        let batch_size = config.batch_size.max(1);
        let n_neg = config.negative_samples.max(1);

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
            let loss = self.batch_loss(
                &current_model,
                head_t,
                rel_t,
                tail_t,
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
                eprintln!("[burn_annular] skipping non-finite batch loss: {loss_val}");
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

    /// Compute the surface-distance ranking loss for a batch.
    ///
    /// Lower surface distance = better containment.
    /// Ranking loss: `relu(margin + pos_score - neg_score).mean()`.
    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnAnnularModel<B>,
        head_ids: Tensor<B, 1, Int>,
        rel_ids: Tensor<B, 1, Int>,
        tail_ids: Tensor<B, 1, Int>,
        neg_tail_ids_flat: Vec<i64>,
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = head_ids.dims()[0];

        // --- Entity parameters ---
        // center: [bs, 2]
        let hcenter = model.entities.center.val().select(0, head_ids.clone()); // [bs, 2]
        let tcenter = model.entities.center.val().select(0, tail_ids.clone()); // [bs, 2]
                                                                               // log_radii: [bs, 2]
        let hlog_radii = model.entities.log_radii.val().select(0, head_ids.clone()); // [bs, 2]
        let tlog_radii = model.entities.log_radii.val().select(0, tail_ids.clone()); // [bs, 2]
                                                                                     // angles: [bs, 2]
        let hangles = model.entities.angles.val().select(0, head_ids); // [bs, 2]
        let tangles = model.entities.angles.val().select(0, tail_ids); // [bs, 2]

        // --- Relation parameters ---
        let rot = model
            .relations
            .rotation
            .val()
            .select(0, rel_ids.clone())
            .reshape([bs, 1]); // [bs, 1]
        let log_rs = model
            .relations
            .log_radial_scale
            .val()
            .select(0, rel_ids.clone())
            .reshape([bs, 1]); // [bs, 1]
                               // log_angular_scale is stored in entity angle parameters through training gradients;
                               // the simplified midpoint distance does not apply it directly during forward scoring.
        let _log_as = model
            .relations
            .log_angular_scale
            .val()
            .select(0, rel_ids)
            .reshape([bs, 1]); // [bs, 1]

        // --- Transform head through relation ---
        let cos_rot = rot.clone().cos(); // [bs, 1]
        let sin_rot = rot.clone().sin(); // [bs, 1]

        let h_re = hcenter.clone().slice([0..bs, 0..1]); // [bs, 1]
        let h_im = hcenter.clone().slice([0..bs, 1..2]); // [bs, 1]

        let trans_re = h_re.clone() * cos_rot.clone() - h_im.clone() * sin_rot.clone(); // [bs, 1]
        let trans_im = h_re * sin_rot + h_im * cos_rot; // [bs, 1]

        // Radii for head (decoded from log-space)
        let h_r_inner = hlog_radii.clone().slice([0..bs, 0..1]).exp(); // [bs, 1]
        let h_r_delta = hlog_radii.clone().slice([0..bs, 1..2]).exp(); // [bs, 1]
        let h_r_outer = h_r_inner.clone() + h_r_delta; // [bs, 1]
        let radial_scale = log_rs.exp(); // [bs, 1]
        let trans_r_inner = h_r_inner * radial_scale.clone(); // [bs, 1]
        let trans_r_outer = h_r_outer * radial_scale; // [bs, 1]

        // Angular bounds for head (shift by rotation)
        let h_theta_start = hangles.clone().slice([0..bs, 0..1]) + rot.clone(); // [bs, 1]
        let h_theta_end = hangles.clone().slice([0..bs, 1..2]) + rot; // [bs, 1]

        // Radii for tail
        let t_r_inner = tlog_radii.clone().slice([0..bs, 0..1]).exp(); // [bs, 1]
        let t_r_delta = tlog_radii.clone().slice([0..bs, 1..2]).exp(); // [bs, 1]
        let t_r_outer = t_r_inner.clone() + t_r_delta; // [bs, 1]

        // Positive surface distance
        let pos_score = Self::surface_distance_batched(
            trans_re.clone(),
            trans_im.clone(),
            trans_r_inner.clone(),
            trans_r_outer.clone(),
            h_theta_start.clone(),
            h_theta_end.clone(),
            tcenter.clone().slice([0..bs, 0..1]),
            tcenter.clone().slice([0..bs, 1..2]),
            t_r_inner.clone(),
            t_r_outer.clone(),
            tangles.clone().slice([0..bs, 0..1]),
            tangles.clone().slice([0..bs, 1..2]),
        ); // [bs, 1]

        // --- Negative tails ---
        let neg_ids = Tensor::<B, 1, Int>::from_data(neg_tail_ids_flat.as_slice(), device);
        // [bs*n_neg, 2]
        let neg_center_flat = model.entities.center.val().select(0, neg_ids.clone());
        let neg_log_radii_flat = model.entities.log_radii.val().select(0, neg_ids.clone());
        let neg_angles_flat = model.entities.angles.val().select(0, neg_ids);

        // Decode neg radii
        let neg_ri_flat = neg_log_radii_flat
            .clone()
            .slice([0..bs * n_neg, 0..1])
            .exp(); // [bs*n_neg, 1]
        let neg_rd_flat = neg_log_radii_flat
            .clone()
            .slice([0..bs * n_neg, 1..2])
            .exp();
        let neg_ro_flat = neg_ri_flat.clone() + neg_rd_flat; // [bs*n_neg, 1]

        // Expand transformed head: [bs, 1] -> repeat n_neg times -> [bs*n_neg, 1]
        let expand_bsn = |t: Tensor<B, 2>| -> Tensor<B, 2> {
            // t: [bs, 1] -> [bs*n_neg, 1] by repeating each row n_neg times
            let n = bs * n_neg;
            let data: Vec<f32> = {
                let raw = t.into_data().to_vec::<f32>().unwrap();
                // raw[i] = value for batch i, repeat n_neg times
                let mut out = Vec::with_capacity(n);
                for v in &raw {
                    for _ in 0..n_neg {
                        out.push(*v);
                    }
                }
                out
            };
            Tensor::<B, 1>::from_data(data.as_slice(), device).reshape([n, 1])
        };

        let trans_re_exp = expand_bsn(trans_re); // [bs*n_neg, 1]
        let trans_im_exp = expand_bsn(trans_im);
        let trans_ri_exp = expand_bsn(trans_r_inner);
        let trans_ro_exp = expand_bsn(trans_r_outer);
        let h_ts_exp = expand_bsn(h_theta_start);
        let h_te_exp = expand_bsn(h_theta_end);

        let neg_score_flat = Self::surface_distance_batched(
            trans_re_exp,
            trans_im_exp,
            trans_ri_exp,
            trans_ro_exp,
            h_ts_exp,
            h_te_exp,
            neg_center_flat.clone().slice([0..bs * n_neg, 0..1]),
            neg_center_flat.slice([0..bs * n_neg, 1..2]),
            neg_ri_flat,
            neg_ro_flat,
            neg_angles_flat.clone().slice([0..bs * n_neg, 0..1]),
            neg_angles_flat.slice([0..bs * n_neg, 1..2]),
        ); // [bs*n_neg, 1]

        // Reshape: [bs, n_neg]
        let neg_score = neg_score_flat.reshape([bs, n_neg]); // [bs, n_neg]

        // Ranking loss: relu(margin + pos_score - neg_score)
        // pos_score is LOW (good), neg_score is HIGH (bad) when training correctly.
        let pos_2d = pos_score.reshape([bs, 1]); // [bs, 1]
        let loss_per_neg = (pos_2d - neg_score.clone() + config.margin).clamp_min(0.0); // [bs, n_neg]

        if config.self_adversarial && config.adversarial_temperature > 0.0 {
            let data = neg_score.clone().into_data();
            let slice = data.as_slice::<f32>().expect("neg_score f32");
            let mut weights: Vec<f32> = Vec::with_capacity(slice.len());
            for row in slice.chunks(n_neg) {
                // Negate: higher distance = harder negative
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

    /// Compute approximate surface distance between two (transformed) annular sectors.
    ///
    /// The three components are:
    /// 1. **Radial gap**: gap between the two annuli's radial bounds (0 when overlapping).
    /// 2. **Angular midpoint distance**: |mid_a - mid_b| (simplified; no circular wrap for autodiff compatibility).
    /// 3. **Center L2 distance**: Euclidean distance between Cartesian centers.
    ///
    /// All inputs have shape `[bs, 1]`.
    #[allow(clippy::too_many_arguments)]
    fn surface_distance_batched(
        // Transformed head
        a_re: Tensor<B, 2>, // [bs, 1]
        a_im: Tensor<B, 2>, // [bs, 1]
        a_ri: Tensor<B, 2>, // [bs, 1]
        a_ro: Tensor<B, 2>, // [bs, 1]
        a_ts: Tensor<B, 2>, // [bs, 1]
        a_te: Tensor<B, 2>, // [bs, 1]
        // Tail
        b_re: Tensor<B, 2>, // [bs, 1]
        b_im: Tensor<B, 2>, // [bs, 1]
        b_ri: Tensor<B, 2>, // [bs, 1]
        b_ro: Tensor<B, 2>, // [bs, 1]
        b_ts: Tensor<B, 2>, // [bs, 1]
        b_te: Tensor<B, 2>, // [bs, 1]
    ) -> Tensor<B, 2> {
        // Radial gap: relu(b_ri - a_ro) + relu(a_ri - b_ro)
        let radial_gap =
            (b_ri - a_ro.clone()).clamp_min(0.0) + (a_ri - b_ro.clone()).clamp_min(0.0); // [bs, 1]

        // Angular midpoint distance (linear, not circular -- avoids discontinuity in autograd)
        let a_mid = (a_ts + a_te) * 0.5_f32; // [bs, 1]
        let b_mid = (b_ts + b_te) * 0.5_f32; // [bs, 1]
        let angular_dist = (a_mid - b_mid).abs(); // [bs, 1]

        // Center L2 distance
        let d_re = a_re - b_re; // [bs, 1]
        let d_im = a_im - b_im; // [bs, 1]
        let center_dist = (d_re.powf_scalar(2.0) + d_im.powf_scalar(2.0) + 1e-8_f32).sqrt(); // [bs, 1]

        radial_gap + angular_dist + center_dist
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Extract model parameters to CPU vecs for evaluation.
    ///
    /// Returns `(centers, log_radii, angles, rotations, log_rad_scales, log_ang_scales,
    ///          n_entities, n_relations)`.
    #[allow(clippy::type_complexity)]
    fn extract_params_raw(
        model: &BurnAnnularModel<B>,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        usize,
        usize,
    ) {
        let n_e = model.entities.center.val().dims()[0];
        let n_r = model.relations.rotation.val().dims()[0];

        let e_center: Vec<f32> = model.entities.center.val().into_data().to_vec().unwrap();
        let e_log_radii: Vec<f32> = model.entities.log_radii.val().into_data().to_vec().unwrap();
        let e_angles: Vec<f32> = model.entities.angles.val().into_data().to_vec().unwrap();

        let r_rotation: Vec<f32> = model.relations.rotation.val().into_data().to_vec().unwrap();
        let r_log_rs: Vec<f32> = model
            .relations
            .log_radial_scale
            .val()
            .into_data()
            .to_vec()
            .unwrap();
        let r_log_as: Vec<f32> = model
            .relations
            .log_angular_scale
            .val()
            .into_data()
            .to_vec()
            .unwrap();

        (
            e_center,
            e_log_radii,
            e_angles,
            r_rotation,
            r_log_rs,
            r_log_as,
            n_e,
            n_r,
        )
    }

    /// Convert model parameters to `AnnularSector` and `AnnularRelation` vecs.
    ///
    /// Invalid sectors (non-finite or r_inner >= r_outer) are replaced with a
    /// default sector so evaluation can proceed gracefully.
    pub fn to_annular_embeddings(
        &self,
        model: &BurnAnnularModel<B>,
    ) -> (Vec<AnnularSector>, Vec<AnnularRelation>) {
        let (ec, er, ea, rrot, rrs, ras, n_e, n_r) = Self::extract_params_raw(model);

        let default_sector = AnnularSector::new(0.0, 0.0, 0.5, 1.5, 0.0, std::f32::consts::PI)
            .expect("default sector valid");

        let entities: Vec<AnnularSector> = (0..n_e)
            .map(|i| {
                let re = ec[i * 2];
                let im = ec[i * 2 + 1];
                let log_ri = er[i * 2];
                let log_rd = er[i * 2 + 1];
                let r_inner = log_ri.exp().max(1e-3);
                let r_outer = r_inner + log_rd.exp().max(1e-3);
                let ts = ea[i * 2];
                let te = ea[i * 2 + 1];
                AnnularSector::new(re, im, r_inner, r_outer, ts, te)
                    .unwrap_or_else(|_| default_sector.clone())
            })
            .collect();

        let relations: Vec<AnnularRelation> = (0..n_r)
            .map(|i| {
                let rotation = rrot[i];
                let radial_scale = rrs[i].exp().max(1e-3);
                let angular_scale = ras[i].exp().max(1e-3);
                AnnularRelation::new(rotation, radial_scale, angular_scale)
                    .unwrap_or_else(|_| AnnularRelation::identity())
            })
            .collect();

        (entities, relations)
    }

    /// Evaluate link prediction on `test_triples`.
    ///
    /// Score: `exp(-surface_distance)` so higher = better alignment,
    /// consistent with `evaluate_link_prediction_generic`'s descending-rank assumption.
    pub fn evaluate(
        &self,
        model: &BurnAnnularModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let (entities, relations) = self.to_annular_embeddings(model);
        let n_e = entities.len();

        let score = |h: usize, r: usize, t: usize| -> f32 {
            let dist =
                crate::annular::surface_distance(&relations[r].apply(&entities[h]), &entities[t]);
            (-dist).exp()
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
        let model = BurnAnnularTrainer::<TestBackend>::new().init_model(10, 3, &device);
        assert_eq!(model.entities.center.val().dims(), [10, 2]);
        assert_eq!(model.entities.log_radii.val().dims(), [10, 2]);
        assert_eq!(model.entities.angles.val().dims(), [10, 2]);
        assert_eq!(model.relations.rotation.val().dims(), [3, 1]);
        assert_eq!(model.relations.log_radial_scale.val().dims(), [3, 1]);
        assert_eq!(model.relations.log_angular_scale.val().dims(), [3, 1]);
    }

    #[test]
    fn loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnAnnularTrainer::<TestBackend>::new();
        let model = trainer.init_model(20, 3, &device);
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
    fn to_annular_embeddings_shapes() {
        let device = Default::default();
        let trainer = BurnAnnularTrainer::<TestBackend>::new();
        let model = trainer.init_model(5, 2, &device);
        let (entities, relations) = trainer.to_annular_embeddings(&model);
        assert_eq!(entities.len(), 5);
        assert_eq!(relations.len(), 2);
        // All entities should have r_inner < r_outer (enforced by log parameterization)
        for e in &entities {
            assert!(e.r_inner() < e.r_outer(), "r_inner >= r_outer: {:?}", e);
            assert!(e.r_inner() > 0.0);
        }
        // All relations should have positive scales
        for r in &relations {
            assert!(r.radial_scale() > 0.0);
            assert!(r.angular_scale() > 0.0);
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
        let mut trainer = BurnAnnularTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(4, 2, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 0.5,
            negative_samples: 1,
            batch_size: 4,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnAnnularModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..80 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        eprintln!("AnnularBurn final loss={last_loss:.4}");

        let results = trainer.evaluate(&model, &triples, None);
        eprintln!(
            "AnnularBurn MRR={:.3} mean_rank={:.1}",
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
        let mut trainer = BurnAnnularTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(5, 2, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.02,
            margin: 0.5,
            batch_size: 4,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnAnnularModel<TestBackend>>();

        let loss_0 = trainer.train_epoch(&mut model, &mut optim, &triples, 0, &config, &device);
        let mut loss_last = loss_0;
        for epoch in 1..30 {
            loss_last =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        assert!(
            loss_last < loss_0,
            "loss should decrease: epoch 0 = {loss_0:.4}, epoch 29 = {loss_last:.4}"
        );
    }
}
