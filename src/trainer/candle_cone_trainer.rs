//! GPU-accelerated cone embedding trainer via candle autograd.
//!
//! Trains cone embeddings (ConE, Zhang & Wang NeurIPS 2021) for knowledge graph
//! completion using per-dimension angular containment scoring.
//!
//! # Architecture
//!
//! Entity embeddings: `axes [num_entities, dim]` (angular positions in [-pi, pi])
//! + `raw_apertures [num_entities, dim]` (sigmoid -> apertures in (0, pi)).
//! Relation offsets: optional `rel_offset [num_relations, dim]` axis translations.
//!
//! ConE distance: per-dimension, check if entity angle falls within query cone
//! (axis +/- aperture). Inside dims contribute `cen * dist_to_axis`, outside
//! dims contribute distance to nearest boundary. Lower = better containment.

use candle_core::{Device, Result, Tensor, Var};
use std::f32::consts::PI;

/// Log-sigmoid: `ln(sigmoid(x)) = -softplus(-x, 1)`, numerically stable.
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let clamped = neg_x.clamp(-20.0, 20.0)?;
    let exp_neg = clamped.exp()?;
    let one = Tensor::ones_like(&exp_neg)?;
    one.add(&exp_neg)?.log()?.neg()
}

/// GPU-accelerated cone embedding trainer.
pub struct CandleConeTrainer {
    /// Entity axis angles: `[num_entities, dim]`.
    pub axes: Var,
    /// Entity raw apertures: `[num_entities, dim]` (sigmoid * pi -> actual apertures).
    pub raw_apertures: Var,
    /// Per-relation axis translation offsets: `[num_relations, dim]`.
    pub rel_offset: Option<Var>,
    /// Embedding dimension.
    pub dim: usize,
    /// Number of entities.
    pub num_entities: usize,
    /// Number of relations.
    pub num_relations: usize,
    /// Inside-distance weight (ConE default: 0.02).
    pub cen: f32,
    /// Device.
    pub device: Device,
}

impl CandleConeTrainer {
    /// Create a new cone trainer.
    ///
    /// Axes initialized uniform in [-pi, pi], apertures initialized to mid-range.
    pub fn new(
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        cen: f32,
        device: &Device,
    ) -> Result<Self> {
        // Uniform [-pi, pi] initialization for axes
        let axes_raw = Tensor::rand(-1.0_f32, 1.0, (num_entities, dim), device)?;
        let axes = Var::from_tensor(&axes_raw.affine(PI as f64, 0.0)?)?;

        // Initialize raw_apertures to 0 (sigmoid(0) = 0.5, so aperture = pi/2)
        let raw_aper = Tensor::zeros((num_entities, dim), candle_core::DType::F32, device)?;
        let raw_apertures = Var::from_tensor(&raw_aper)?;

        let rel_offset = if num_relations > 0 {
            let nr = num_relations.max(1);
            let r = Tensor::rand(-0.1_f32, 0.1, (nr, dim), device)?;
            Some(Var::from_tensor(&r)?)
        } else {
            None
        };

        Ok(Self {
            axes,
            raw_apertures,
            rel_offset,
            dim,
            num_entities,
            num_relations,
            cen,
            device: device.clone(),
        })
    }

    /// Get entity axes for given IDs.
    fn entity_axes(&self, ids: &Tensor) -> Result<Tensor> {
        self.axes.as_tensor().index_select(ids, 0)
    }

    /// Get entity apertures (sigmoid * pi) for given IDs.
    fn entity_apertures(&self, ids: &Tensor) -> Result<Tensor> {
        let raw = self.raw_apertures.as_tensor().index_select(ids, 0)?;
        // sigmoid(raw) * pi -> apertures in (0, pi)
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_raw = raw.neg()?;
        let exp_neg = neg_raw.clamp(-20.0, 20.0)?.exp()?;
        let one = Tensor::ones_like(&exp_neg)?;
        let sig = one.broadcast_div(&one.add(&exp_neg)?)?;
        sig.affine(PI as f64, 0.0)
    }

    /// Get relation offsets for given IDs.
    fn rel_offsets(&self, ids: &Tensor) -> Result<Option<Tensor>> {
        match &self.rel_offset {
            Some(r) => Ok(Some(r.as_tensor().index_select(ids, 0)?)),
            None => Ok(None),
        }
    }

    /// ConE distance score: lower = better containment of entity by query.
    ///
    /// query_axes, query_apertures, entity_axes: all `[batch, dim]`.
    fn cone_distance(
        query_axes: &Tensor,
        query_apertures: &Tensor,
        entity_axes: &Tensor,
        cen: f32,
    ) -> Result<Tensor> {
        // dist_to_axis = |sin((entity - query_axis) / 2)|
        let diff = entity_axes.sub(query_axes)?;
        let half_diff = diff.affine(0.5, 0.0)?;
        let dist_to_axis = half_diff.sin()?.abs()?;

        // dist_base = |sin(aperture / 2)|
        let half_aper = query_apertures.affine(0.5, 0.0)?;
        let dist_base = half_aper.sin()?.abs()?;

        // Inside mask: dist_to_axis < dist_base
        let inside = dist_to_axis.lt(&dist_base)?;

        // Inside distance: dist_to_axis (weighted by cen)
        let dist_in = dist_to_axis.affine(cen as f64, 0.0)?;

        // Outside distance: min(|sin((entity - (axis-aper))/2)|, |sin((entity - (axis+aper))/2)|)
        let delta1 = entity_axes.sub(&query_axes.sub(query_apertures)?)?;
        let delta2 = entity_axes.sub(&query_axes.add(query_apertures)?)?;
        let d1 = delta1.affine(0.5, 0.0)?.sin()?.abs()?;
        let d2 = delta2.affine(0.5, 0.0)?.sin()?.abs()?;
        let dist_out = d1.minimum(&d2)?;

        // Combine: inside dims use dist_in, outside dims use dist_out
        let inside_f = inside.to_dtype(candle_core::DType::F32)?;
        let outside_f = inside_f.affine(-1.0, 1.0)?; // 1 - inside

        let combined = inside_f
            .mul(&dist_in)?
            .add(&outside_f.mul(&dist_out)?)?;

        // Sum across dimensions -> [batch]
        combined.sum(1)
    }

    /// Train with AdamW optimizer and log-sigmoid loss.
    ///
    /// Returns per-epoch average losses.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &self,
        triples: &[(usize, usize, usize)],
        epochs: usize,
        lr: f64,
        batch_size: usize,
        margin: f32,
        negative_samples: usize,
    ) -> Result<Vec<f32>> {
        let ne = self.num_entities as u64;
        let n = triples.len();

        use candle_nn::{AdamW, Optimizer, ParamsAdamW};

        let mut vars: Vec<Var> = vec![self.axes.clone(), self.raw_apertures.clone()];
        if let Some(r) = &self.rel_offset {
            vars.push(r.clone());
        }

        let params = ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut opt = AdamW::new(vars, params)?;

        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut rng: u64 = 42;
        let lcg = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *s >> 33
        };

        for epoch in 0..epochs {
            // Cosine LR schedule
            let progress = epoch as f64 / epochs as f64;
            let current_lr = lr * (0.01 + 0.99 * (1.0 + (PI as f64 * progress).cos()) / 2.0);
            opt.set_learning_rate(current_lr);

            let bs = batch_size.min(n);
            let mut h_ids = Vec::with_capacity(bs);
            let mut t_ids = Vec::with_capacity(bs);
            let mut r_ids = Vec::with_capacity(bs);
            for _ in 0..bs {
                let idx = lcg(&mut rng) as usize % n;
                let (h, r, t) = triples[idx];
                h_ids.push(h as u32);
                r_ids.push(r as u32);
                t_ids.push(t as u32);
            }

            let h_t = Tensor::from_vec(h_ids, (bs,), &self.device)?;
            let t_t = Tensor::from_vec(t_ids.clone(), (bs,), &self.device)?;
            let r_t = Tensor::from_vec(r_ids.clone(), (bs,), &self.device)?;

            // Get embeddings
            let h_axes = self.entity_axes(&h_t)?;
            let h_aper = self.entity_apertures(&h_t)?;
            let t_axes = self.entity_axes(&t_t)?;

            // Apply relation offset to head axes
            let q_axes = if let Some(offsets) = self.rel_offsets(&r_t)? {
                h_axes.add(&offsets)?
            } else {
                h_axes
            };

            // Positive scores: head cone should contain tail
            let pos_scores = Self::cone_distance(&q_axes, &h_aper, &t_axes, self.cen)?;

            // Negative sampling: corrupt tails
            let total_neg = bs * negative_samples;
            let neg_ids: Vec<u32> = (0..total_neg)
                .map(|_| (lcg(&mut rng) % ne) as u32)
                .collect();
            let neg_t = Tensor::from_vec(neg_ids, (total_neg,), &self.device)?;
            let neg_axes = self.entity_axes(&neg_t)?;

            // Expand query to match negative batch: [bs, dim] -> [total_neg, dim]
            let q_axes_exp = q_axes.repeat((negative_samples, 1))?;
            let h_aper_exp = h_aper.repeat((negative_samples, 1))?;
            let neg_scores = Self::cone_distance(&q_axes_exp, &h_aper_exp, &neg_axes, self.cen)?;

            // Negative loss: -log(sigmoid(neg - margin))
            let neg_shifted = neg_scores.affine(1.0, -(margin as f64))?;
            let neg_loss = log_sigmoid(&neg_shifted)?.neg()?.mean(0)?;

            // Positive loss: -log(sigmoid(margin - pos))
            let pos_shifted = pos_scores.affine(-1.0, margin as f64)?;
            let pos_loss = log_sigmoid(&pos_shifted)?.neg()?.mean(0)?;

            let total_loss = pos_loss.add(&neg_loss)?;
            let loss_val = total_loss.to_scalar::<f32>()?;
            opt.backward_step(&total_loss)?;

            epoch_losses.push(loss_val);

            if loss_val.is_nan() || loss_val.is_infinite() {
                eprintln!("  WARNING: loss diverged at epoch {}. Stopping.", epoch + 1);
                break;
            }

            if (epoch + 1) % 100 == 0 || epoch == 0 {
                eprintln!(
                    "  epoch {:>5}/{epochs}: loss={loss_val:.4} lr={current_lr:.6}",
                    epoch + 1
                );
            }
        }

        Ok(epoch_losses)
    }

    /// Score (head, tail) pairs. Lower = better containment.
    pub fn score(&self, head_ids: &Tensor, tail_ids: &Tensor) -> Result<Tensor> {
        let h_axes = self.entity_axes(head_ids)?;
        let h_aper = self.entity_apertures(head_ids)?;
        let t_axes = self.entity_axes(tail_ids)?;
        Self::cone_distance(&h_axes, &h_aper, &t_axes, self.cen)
    }

    /// Score (head, tail, relation) triples. Lower = better containment.
    pub fn score_with_rel(
        &self,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        rel_ids: &Tensor,
    ) -> Result<Tensor> {
        let h_axes = self.entity_axes(head_ids)?;
        let h_aper = self.entity_apertures(head_ids)?;
        let t_axes = self.entity_axes(tail_ids)?;

        let q_axes = if let Some(offsets) = self.rel_offsets(rel_ids)? {
            h_axes.add(&offsets)?
        } else {
            h_axes
        };

        Self::cone_distance(&q_axes, &h_aper, &t_axes, self.cen)
    }

    /// Score all entities as tails for a given head.
    pub fn score_all_tails(
        &self,
        head_id: usize,
        rel_id: Option<usize>,
    ) -> Result<Tensor> {
        let h_t = Tensor::from_vec(vec![head_id as u32], (1,), &self.device)?;
        let h_axes = self.entity_axes(&h_t)?;
        let h_aper = self.entity_apertures(&h_t)?;

        let q_axes = if let Some(rid) = rel_id {
            let r_t = Tensor::from_vec(vec![rid as u32], (1,), &self.device)?;
            if let Some(offsets) = self.rel_offsets(&r_t)? {
                h_axes.add(&offsets)?
            } else {
                h_axes
            }
        } else {
            h_axes
        };

        // Broadcast: expand query to [num_entities, dim]
        let q_axes_exp = q_axes.broadcast_as((self.num_entities, self.dim))?;
        let q_aper_exp = h_aper.broadcast_as((self.num_entities, self.dim))?;

        let all_ids: Vec<u32> = (0..self.num_entities as u32).collect();
        let all_t = Tensor::from_vec(all_ids, (self.num_entities,), &self.device)?;
        let all_axes = self.entity_axes(&all_t)?;

        Self::cone_distance(&q_axes_exp, &q_aper_exp, &all_axes, self.cen)
    }

    /// Score all entities as heads for a given tail (and optional relation).
    pub fn score_all_heads(
        &self,
        tail_id: usize,
        rel_id: Option<usize>,
    ) -> Result<Tensor> {
        let t_t = Tensor::from_vec(vec![tail_id as u32], (1,), &self.device)?;
        let t_axes = self.entity_axes(&t_t)?;
        let t_axes_exp = t_axes.broadcast_as((self.num_entities, self.dim))?;

        // Get all entity axes + apertures as potential heads
        let all_ids: Vec<u32> = (0..self.num_entities as u32).collect();
        let all_t = Tensor::from_vec(all_ids, (self.num_entities,), &self.device)?;
        let all_axes = self.entity_axes(&all_t)?;
        let all_aper = self.entity_apertures(&all_t)?;

        let q_axes = if let Some(rid) = rel_id {
            let r_t = Tensor::from_vec(vec![rid as u32], (1,), &self.device)?;
            if let Some(offsets) = self.rel_offsets(&r_t)? {
                let offset_exp = offsets.broadcast_as((self.num_entities, self.dim))?;
                all_axes.add(&offset_exp)?
            } else {
                all_axes
            }
        } else {
            all_axes
        };

        Self::cone_distance(&q_axes, &all_aper, &t_axes_exp, self.cen)
    }

    /// Evaluate link prediction (filtered setting).
    ///
    /// Returns `(mrr, hits_at_1, hits_at_3, hits_at_10, mean_rank)`.
    pub fn evaluate(
        &self,
        test_triples: &[(usize, usize, usize)],
        all_triples: &[(usize, usize, usize)],
    ) -> Result<(f32, f32, f32, f32, f32)> {
        use std::collections::{HashMap, HashSet};

        let mut filter_hr: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
        let mut filter_tr: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
        for &(h, r, t) in all_triples {
            filter_hr.entry((h, r)).or_default().insert(t);
            filter_tr.entry((t, r)).or_default().insert(h);
        }

        let mut reciprocal_ranks = Vec::with_capacity(test_triples.len() * 2);
        let mut hits1 = 0u32;
        let mut hits3 = 0u32;
        let mut hits10 = 0u32;
        let mut total_rank = 0u64;

        for &(h, r, t) in test_triples {
            // Tail prediction
            let tail_scores: Vec<f32> = self.score_all_tails(h, Some(r))?.to_vec1()?;
            let correct_score = tail_scores[t];
            let filter_set = filter_hr.get(&(h, r));
            let mut rank = 1u32;
            for (eid, &s) in tail_scores.iter().enumerate() {
                if eid == t {
                    continue;
                }
                if let Some(known) = filter_set {
                    if known.contains(&eid) {
                        continue;
                    }
                }
                if s < correct_score {
                    rank += 1;
                }
            }
            reciprocal_ranks.push(1.0 / rank as f32);
            total_rank += rank as u64;
            if rank <= 1 { hits1 += 1; }
            if rank <= 3 { hits3 += 1; }
            if rank <= 10 { hits10 += 1; }

            // Head prediction
            let head_scores: Vec<f32> = self.score_all_heads(t, Some(r))?.to_vec1()?;
            let correct_score = head_scores[h];
            let filter_set = filter_tr.get(&(t, r));
            let mut rank = 1u32;
            for (eid, &s) in head_scores.iter().enumerate() {
                if eid == h {
                    continue;
                }
                if let Some(known) = filter_set {
                    if known.contains(&eid) {
                        continue;
                    }
                }
                if s < correct_score {
                    rank += 1;
                }
            }
            reciprocal_ranks.push(1.0 / rank as f32);
            total_rank += rank as u64;
            if rank <= 1 { hits1 += 1; }
            if rank <= 3 { hits3 += 1; }
            if rank <= 10 { hits10 += 1; }
        }

        let n = reciprocal_ranks.len() as f32;
        let mrr = reciprocal_ranks.iter().sum::<f32>() / n;
        let h1 = hits1 as f32 / n;
        let h3 = hits3 as f32 / n;
        let h10 = hits10 as f32 / n;
        let mr = total_rank as f32 / n;

        Ok((mrr, h1, h3, h10, mr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_trainer_creates() {
        let device = Device::Cpu;
        let trainer = CandleConeTrainer::new(100, 5, 32, 0.02, &device).unwrap();
        assert_eq!(trainer.num_entities, 100);
        assert_eq!(trainer.num_relations, 5);
        assert_eq!(trainer.dim, 32);
    }

    #[test]
    fn test_cone_trainer_fits() {
        let device = Device::Cpu;
        let trainer = CandleConeTrainer::new(10, 1, 8, 0.02, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 0, 3), (4, 0, 5), (6, 0, 7)];
        let losses = trainer.fit(&triples, 50, 0.01, 4, 3.0, 2).unwrap();
        assert_eq!(losses.len(), 50);
        // Loss should decrease over training
        let first_avg: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let last_avg: f32 = losses[45..].iter().sum::<f32>() / 5.0;
        assert!(
            last_avg < first_avg,
            "loss should decrease: first_avg={first_avg}, last_avg={last_avg}"
        );
    }

    #[test]
    fn test_cone_score() {
        let device = Device::Cpu;
        let trainer = CandleConeTrainer::new(10, 0, 4, 0.02, &device).unwrap();
        let h = Tensor::from_vec(vec![0u32, 1], (2,), &device).unwrap();
        let t = Tensor::from_vec(vec![2u32, 3], (2,), &device).unwrap();
        let scores = trainer.score(&h, &t).unwrap();
        assert_eq!(scores.dims(), &[2]);
    }

    #[test]
    fn test_cone_score_all_tails() {
        let device = Device::Cpu;
        let trainer = CandleConeTrainer::new(10, 0, 4, 0.02, &device).unwrap();
        let scores = trainer.score_all_tails(0, None).unwrap();
        assert_eq!(scores.dims(), &[10]);
    }
}
