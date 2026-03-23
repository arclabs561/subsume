//! GPU-accelerated box embedding trainer via candle autograd.
//!
//! Trains box embeddings for knowledge graph completion using:
//! - Per-dimension containment violation scoring (L1 norm, scales to any dim)
//! - Log-sigmoid negative sampling loss (BoxE-style)
//! - Self-adversarial negative weighting (RotatE-style)
//! - Head + tail corruption
//! - Zero-copy GPU training loop (data preloaded, negatives generated on device)
//!
//! # Architecture
//!
//! Entity embeddings: `mu` (centers) + `log_delta` (log-widths), both `[num_entities, dim]`.
//! Relation embeddings: optional `rel_offset` translations, `[num_relations, dim]`.
//! Box bounds: `min = mu - delta/2`, `max = mu + delta/2`, where
//! `delta = softplus(exp(log_delta), beta)`.
//!
//! The scoring function sums per-dimension containment violations:
//! `score(h, t) = sum_d(relu(min_t - min_h) + relu(max_h - max_t))`.
//! Lower score = head box better contains tail box.

use candle_core::{Device, Result, Tensor, Var};

/// GPU-accelerated box embedding trainer.
pub struct CandleBoxTrainer {
    /// Entity center positions: `[num_entities, dim]`.
    pub mu: Var,
    /// Entity log-widths: `[num_entities, dim]`.
    pub log_delta: Var,
    /// Per-relation translation offsets: `[num_relations, dim]`.
    pub rel_offset: Option<Var>,
    /// Embedding dimension.
    pub dim: usize,
    /// Number of entities.
    pub num_entities: usize,
    /// Number of relations (0 if identity).
    pub num_relations: usize,
    /// Gumbel beta parameter for softplus width transform.
    pub beta: f32,
    /// Weight for inside distance (0.0 = pure containment violation).
    ///
    /// BoxE uses both outside distance (penalty for protrusion) and inside
    /// distance (penalty for being off-center when contained). Setting this
    /// to ~0.02-0.1 enables the inside term.
    pub inside_weight: f32,
    /// Device (CPU, CUDA, or Metal).
    pub device: Device,
}

impl CandleBoxTrainer {
    /// Create a new trainer with random initialization.
    ///
    /// Set `num_relations` to 0 for identity (no relation transforms).
    pub fn new(
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        beta: f32,
        device: &Device,
    ) -> Result<Self> {
        let scale = (2.0 / dim as f64).sqrt();
        let mu = Var::from_tensor(&Tensor::randn(
            0.0_f32,
            scale as f32,
            (num_entities, dim),
            device,
        )?)?;
        let log_delta =
            Var::from_tensor(&Tensor::randn(-1.0_f32, 0.1, (num_entities, dim), device)?)?;

        let rel_offset = if num_relations > 0 {
            Some(Var::from_tensor(&Tensor::randn(
                0.0_f32,
                0.01,
                (num_relations, dim),
                device,
            )?)?)
        } else {
            None
        };

        Ok(Self {
            mu,
            log_delta,
            rel_offset,
            dim,
            num_entities,
            num_relations,
            beta,
            inside_weight: 0.0,
            device: device.clone(),
        })
    }

    /// Set the inside distance weight (BoxE-style).
    ///
    /// Default is 0.0 (pure containment violation). Values in 0.02-0.1 are typical.
    #[must_use]
    pub fn with_inside_weight(mut self, weight: f32) -> Self {
        self.inside_weight = weight;
        self
    }

    /// Compute (min_all, max_all) for the entire entity table.
    fn entity_bounds(&self) -> Result<(Tensor, Tensor)> {
        let exp_ld = self.log_delta.as_tensor().exp()?;
        let delta = softplus(&exp_ld, self.beta)?;
        let hw = delta.affine(0.5, 0.0)?;
        let mu = self.mu.as_tensor();
        let min_all = mu.sub(&hw)?;
        let max_all = mu.add(&hw)?;
        Ok((min_all, max_all))
    }

    /// Per-dimension distance: containment violation + weighted inside distance.
    ///
    /// Violation (outside distance): `relu(min_h - min_t) + relu(max_t - max_h)` per dim.
    /// Positive when tail protrudes from head. Zero when fully contained.
    /// Lower score = better containment (matches loss function convention).
    ///
    /// Inside distance (BoxE-style): `|center_t - center_h| / width_h` per dim,
    /// masked to dimensions where the tail is fully contained (violation=0).
    /// Discriminates among contained entities by penalizing off-center placement.
    ///
    /// Combined: `violation + alpha * inside`.
    ///
    /// `alpha = 0` recovers pure containment violation.
    fn distance(
        min_h: &Tensor,
        max_h: &Tensor,
        min_t: &Tensor,
        max_t: &Tensor,
        alpha: f32,
    ) -> Result<Tensor> {
        // Containment violation: positive when tail protrudes from head.
        let lower_violation = min_h.sub(min_t)?.relu()?;
        let upper_violation = max_t.sub(max_h)?.relu()?;
        let violation = lower_violation.add(&upper_violation)?;

        if alpha == 0.0 {
            return Ok(violation);
        }

        // Inside distance: |center_t - center_h| / width_h, masked where contained.
        let center_h = min_h.add(max_h)?.affine(0.5, 0.0)?;
        let center_t = min_t.add(max_t)?.affine(0.5, 0.0)?;
        let center_dist = center_t.sub(&center_h)?.abs()?;

        let width_h = max_h.sub(min_h)?.clamp(1e-6, f64::INFINITY)?;
        let inside_normed = center_dist.div(&width_h)?;

        // Soft mask: 1 where contained (violation=0), 0 where protruding.
        let mask = violation.affine(-10.0, 0.0)?.exp()?;
        let inside_masked = inside_normed.mul(&mask)?;

        violation.add(&inside_masked.affine(alpha as f64, 0.0)?)
    }

    /// Score a batch from pre-computed entity bounds.
    ///
    /// Returns per-sample scores (lower = better containment).
    fn batch_score(
        &self,
        min_all: &Tensor,
        max_all: &Tensor,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        rel_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut min_h = min_all.index_select(head_ids, 0)?;
        let mut max_h = max_all.index_select(head_ids, 0)?;
        let min_t = min_all.index_select(tail_ids, 0)?;
        let max_t = max_all.index_select(tail_ids, 0)?;

        if let (Some(ref rel_var), Some(rel)) = (&self.rel_offset, rel_ids) {
            let offset = rel_var.as_tensor().index_select(rel, 0)?;
            min_h = min_h.add(&offset)?;
            max_h = max_h.add(&offset)?;
        }

        let dist = Self::distance(&min_h, &max_h, &min_t, &max_t, self.inside_weight)?;
        dist.sum(1)
    }

    /// Log-sigmoid negative sampling loss.
    fn ns_loss(
        pos_scores: &Tensor,
        neg_scores: &Tensor,
        margin: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let margin_p = Tensor::full(margin, pos_scores.shape(), device)?;
        let margin_n = Tensor::full(margin, neg_scores.shape(), device)?;

        let pos_term = log_sigmoid(&margin_p.sub(pos_scores)?)?;
        let neg_term = log_sigmoid(&neg_scores.sub(&margin_n)?)?;

        pos_term.mean(0)?.add(&neg_term.mean(0)?)?.neg()
    }

    /// Self-adversarial negative sampling loss.
    ///
    /// Weights each negative by `softmax(alpha * neg_score)` (detached).
    /// Concentrates gradient on hard negatives the model currently scores well.
    fn self_adversarial_ns_loss(
        pos_scores: &Tensor,
        neg_scores: &Tensor,
        margin: f32,
        adv_temp: f32,
        num_neg: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let margin_p = Tensor::full(margin, pos_scores.shape(), device)?;

        let pos_term = log_sigmoid(&margin_p.sub(pos_scores)?)?;

        // Reshape negatives: [batch * num_neg] -> [batch, num_neg]
        let batch_size = pos_scores.dims()[0];
        let neg_2d = neg_scores.reshape((batch_size, num_neg))?;

        // Softmax weights (detached -- no gradient through weights)
        let weights = candle_nn::ops::softmax(&neg_2d.affine(adv_temp as f64, 0.0)?.detach(), 1)?;

        // Weighted negative loss per batch element
        let margin_2d = Tensor::full(margin, (batch_size, num_neg), device)?;
        let neg_term_2d = log_sigmoid(&neg_2d.sub(&margin_2d)?)?;
        let weighted = weights.mul(&neg_term_2d)?.sum(1)?; // [batch]

        pos_term.mean(0)?.add(&weighted.mean(0)?)?.neg()
    }

    /// Score a batch of (head, tail) pairs.
    pub fn score(&self, head_ids: &Tensor, tail_ids: &Tensor) -> Result<Tensor> {
        let (min_all, max_all) = self.entity_bounds()?;
        self.batch_score(&min_all, &max_all, head_ids, tail_ids, None)
    }

    /// Score with relation translation applied to head.
    pub fn score_with_rel(
        &self,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        rel_ids: &Tensor,
    ) -> Result<Tensor> {
        let (min_all, max_all) = self.entity_bounds()?;
        self.batch_score(&min_all, &max_all, head_ids, tail_ids, Some(rel_ids))
    }

    /// Train with AdamW, log-sigmoid loss, and optional self-adversarial weighting.
    ///
    /// Triples are `(head_id, relation_id, tail_id)`. Corrupts both head and tail.
    /// Returns per-epoch average losses.
    ///
    /// Set `adversarial_temperature` to 0.0 to disable self-adversarial weighting.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &self,
        train_triples: &[(usize, usize, usize)],
        epochs: usize,
        lr: f64,
        batch_size: usize,
        margin: f32,
        negative_samples: usize,
        adversarial_temperature: f32,
    ) -> Result<Vec<f32>> {
        use candle_nn::{AdamW, Optimizer, ParamsAdamW};

        let mut vars = vec![self.mu.clone(), self.log_delta.clone()];
        if let Some(ref rel) = self.rel_offset {
            vars.push(rel.clone());
        }

        let params = ParamsAdamW {
            lr,
            weight_decay: 0.0, // BoxE uses no weight decay
            ..Default::default()
        };
        let mut opt = AdamW::new(vars, params)?;
        let n = train_triples.len();
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut rng: u64 = 42;

        // Cosine LR schedule: lr decays from `lr` to `lr * 0.01` over all epochs.
        let lr_min = lr * 0.01;

        let lcg = |s: &mut u64| -> usize {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as usize
        };

        // Preload training data as device tensors
        let all_heads: Vec<u32> = train_triples.iter().map(|t| t.0 as u32).collect();
        let all_rels: Vec<u32> = train_triples.iter().map(|t| t.1 as u32).collect();
        let all_tails: Vec<u32> = train_triples.iter().map(|t| t.2 as u32).collect();
        let heads_gpu = Tensor::from_vec(all_heads, (n,), &self.device)?;
        let rels_gpu = Tensor::from_vec(all_rels, (n,), &self.device)?;
        let tails_gpu = Tensor::from_vec(all_tails, (n,), &self.device)?;

        let mut indices: Vec<u32> = (0..n as u32).collect();
        let use_self_adv = adversarial_temperature > 0.0;

        for epoch in 0..epochs {
            for i in (1..n).rev() {
                let j = lcg(&mut rng) % (i + 1);
                indices.swap(i, j);
            }

            let perm = Tensor::from_vec(indices.clone(), (n,), &self.device)?;
            let heads_shuf = heads_gpu.index_select(&perm, 0)?;
            let rels_shuf = rels_gpu.index_select(&perm, 0)?;
            let tails_shuf = tails_gpu.index_select(&perm, 0)?;

            // Cosine LR schedule
            let progress = epoch as f64 / epochs.max(1) as f64;
            let current_lr =
                lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos());
            opt.set_learning_rate(current_lr);

            let mut total_loss = 0.0f32;
            let mut batch_count = 0usize;

            // Compute entity bounds once per epoch for speed.
            // Staleness from not recomputing after each backward_step is
            // acceptable -- standard practice in PyKEEN/DGL-KE.
            let (min_all, max_all) = self.entity_bounds()?;

            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let bs = batch_end - batch_start;

                let h_t = heads_shuf.narrow(0, batch_start, bs)?;
                let r_t = rels_shuf.narrow(0, batch_start, bs)?;
                let t_t = tails_shuf.narrow(0, batch_start, bs)?;

                let rel_ref = if self.num_relations > 0 {
                    Some(&r_t)
                } else {
                    None
                };

                let pos_scores = self.batch_score(&min_all, &max_all, &h_t, &t_t, rel_ref)?;

                // Corrupt both head and tail (half each)
                let total_neg = bs * negative_samples;
                let half_neg = total_neg / 2;

                // Tail corruption: keep head, randomize tail
                let neg_rand_t =
                    Tensor::rand(0.0_f32, self.num_entities as f32, (half_neg,), &self.device)?;
                let neg_t_ids = neg_rand_t.to_dtype(candle_core::DType::U32)?;
                let neg_h_for_t = h_t
                    .repeat((half_neg.div_ceil(bs),))?
                    .narrow(0, 0, half_neg)?;
                let neg_r_for_t = r_t
                    .repeat((half_neg.div_ceil(bs),))?
                    .narrow(0, 0, half_neg)?;

                // Head corruption: keep tail, randomize head
                let neg_rand_h = Tensor::rand(
                    0.0_f32,
                    self.num_entities as f32,
                    (total_neg - half_neg,),
                    &self.device,
                )?;
                let neg_h_ids = neg_rand_h.to_dtype(candle_core::DType::U32)?;
                let remaining = total_neg - half_neg;
                let neg_t_for_h = t_t
                    .repeat((remaining.div_ceil(bs),))?
                    .narrow(0, 0, remaining)?;
                let neg_r_for_h = r_t
                    .repeat((remaining.div_ceil(bs),))?
                    .narrow(0, 0, remaining)?;

                // Concatenate all negatives
                let all_neg_h = Tensor::cat(&[&neg_h_for_t, &neg_h_ids], 0)?;
                let all_neg_t = Tensor::cat(&[&neg_t_ids, &neg_t_for_h], 0)?;
                let all_neg_r = Tensor::cat(&[&neg_r_for_t, &neg_r_for_h], 0)?;

                let neg_rel_ref = if self.num_relations > 0 {
                    Some(&all_neg_r)
                } else {
                    None
                };
                let neg_scores =
                    self.batch_score(&min_all, &max_all, &all_neg_h, &all_neg_t, neg_rel_ref)?;

                let loss = if use_self_adv {
                    Self::self_adversarial_ns_loss(
                        &pos_scores,
                        &neg_scores,
                        margin,
                        adversarial_temperature,
                        negative_samples,
                        &self.device,
                    )?
                } else {
                    Self::ns_loss(&pos_scores, &neg_scores, margin, &self.device)?
                };

                opt.backward_step(&loss)?;

                total_loss += loss.to_scalar::<f32>()?;
                batch_count += 1;
            }

            let avg = total_loss / batch_count.max(1) as f32;
            epoch_losses.push(avg);

            if (epoch + 1) % 50 == 0 || epoch == 0 {
                eprintln!("  epoch {:>4}/{epochs}: avg_loss = {avg:.6}", epoch + 1);
            }
        }

        Ok(epoch_losses)
    }

    /// Score all entities as potential tails for a (head, relation) query.
    ///
    /// Returns a `[num_entities]` tensor of L1 containment violation scores.
    /// Lower score = better containment (head box better contains tail box).
    /// Uses the same scoring signal as training.
    pub fn score_all_tails(&self, head_id: usize, rel_id: Option<usize>) -> Result<Tensor> {
        let (min_all, max_all) = self.entity_bounds()?;

        // Head bounds: [1, dim]
        let h_idx = Tensor::from_vec(vec![head_id as u32], (1,), &self.device)?;
        let mut min_h = min_all.index_select(&h_idx, 0)?; // [1, dim]
        let mut max_h = max_all.index_select(&h_idx, 0)?;

        // Apply relation offset
        if let (Some(ref rel_var), Some(rid)) = (&self.rel_offset, rel_id) {
            let r_idx = Tensor::from_vec(vec![rid as u32], (1,), &self.device)?;
            let offset = rel_var.as_tensor().index_select(&r_idx, 0)?;
            min_h = min_h.add(&offset)?;
            max_h = max_h.add(&offset)?;
        }

        // Broadcast head to [num_entities, dim]
        let min_h = min_h.broadcast_as((self.num_entities, self.dim))?;
        let max_h = max_h.broadcast_as((self.num_entities, self.dim))?;

        // Distance against all entities as tails
        let dist = Self::distance(&min_h, &max_h, &min_all, &max_all, self.inside_weight)?;
        dist.sum(1) // [num_entities]
    }

    /// Evaluate link prediction using L1 violation scoring (matches training signal).
    ///
    /// Returns `(mrr, hits_at_1, hits_at_3, hits_at_10, mean_rank)`.
    /// Uses filtered setting: known-true triples are excluded from ranking.
    pub fn evaluate(
        &self,
        test_triples: &[(usize, usize, usize)],
        all_triples: &[(usize, usize, usize)],
    ) -> Result<(f32, f32, f32, f32, f32)> {
        use std::collections::{HashMap, HashSet};

        // Build filter: for each (head, rel), collect known tail entities.
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
            // Tail prediction: score all entities as tails for (h, r, ?)
            let tail_scores: Vec<f32> = self.score_all_tails(h, Some(r))?.to_vec1()?;

            let correct_score = tail_scores[t];
            let filter_set = filter_hr.get(&(h, r));

            // Rank: count entities with strictly better (lower) score, excluding filtered.
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
            if rank <= 1 {
                hits1 += 1;
            }
            if rank <= 3 {
                hits3 += 1;
            }
            if rank <= 10 {
                hits10 += 1;
            }

            // Head prediction: score all entities as heads for (?, r, t)
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
            if rank <= 1 {
                hits1 += 1;
            }
            if rank <= 3 {
                hits3 += 1;
            }
            if rank <= 10 {
                hits10 += 1;
            }
        }

        let n = reciprocal_ranks.len() as f32;
        let mrr = reciprocal_ranks.iter().sum::<f32>() / n;
        let h1 = hits1 as f32 / n;
        let h3 = hits3 as f32 / n;
        let h10 = hits10 as f32 / n;
        let mr = total_rank as f32 / n;

        Ok((mrr, h1, h3, h10, mr))
    }

    /// Score all entities as potential heads for a (tail, relation) query.
    ///
    /// Returns a `[num_entities]` tensor of L1 containment violation scores.
    /// For head prediction: we check how well each candidate head contains the given tail.
    pub fn score_all_heads(&self, tail_id: usize, rel_id: Option<usize>) -> Result<Tensor> {
        let (min_all, max_all) = self.entity_bounds()?;

        // Tail bounds: [1, dim]
        let t_idx = Tensor::from_vec(vec![tail_id as u32], (1,), &self.device)?;
        let min_t = min_all.index_select(&t_idx, 0)?; // [1, dim]
        let max_t = max_all.index_select(&t_idx, 0)?;

        // Broadcast tail to [num_entities, dim]
        let min_t = min_t.broadcast_as((self.num_entities, self.dim))?;
        let max_t = max_t.broadcast_as((self.num_entities, self.dim))?;

        // All entities as candidate heads
        let mut min_h = min_all.clone();
        let mut max_h = max_all.clone();

        // Apply relation offset to all candidate heads
        if let (Some(ref rel_var), Some(rid)) = (&self.rel_offset, rel_id) {
            let r_idx = Tensor::from_vec(vec![rid as u32], (1,), &self.device)?;
            let offset = rel_var.as_tensor().index_select(&r_idx, 0)?; // [1, dim]
            let offset = offset.broadcast_as((self.num_entities, self.dim))?;
            min_h = min_h.add(&offset)?;
            max_h = max_h.add(&offset)?;
        }

        let dist = Self::distance(&min_h, &max_h, &min_t, &max_t, self.inside_weight)?;
        dist.sum(1) // [num_entities]
    }
}

/// Log-sigmoid: `ln(sigmoid(x)) = -softplus(-x, 1)`, numerically stable.
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let sp = softplus(&neg_x, 1.0)?;
    sp.neg()
}

/// Softplus: `(1/beta) * ln(1 + exp(beta * x))`, numerically stable.
fn softplus(x: &Tensor, beta: f32) -> Result<Tensor> {
    let scaled = x.affine(beta as f64, 0.0)?;
    let clamped = scaled.clamp(-20.0, 20.0)?;
    let exp_scaled = clamped.exp()?;
    let one = Tensor::ones_like(&exp_scaled)?;
    let log1p = one.add(&exp_scaled)?.log()?;
    log1p.affine(1.0 / beta as f64, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_trainer_creates() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(100, 5, 32, 10.0, &device).unwrap();
        assert_eq!(trainer.num_entities, 100);
        assert_eq!(trainer.num_relations, 5);
        assert!(trainer.rel_offset.is_some());
    }

    #[test]
    fn test_candle_trainer_no_relations() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(100, 0, 32, 10.0, &device).unwrap();
        assert!(trainer.rel_offset.is_none());
    }

    #[test]
    fn test_score_computes() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 0, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let tails = Tensor::from_vec(vec![3u32, 4, 5], (3,), &device).unwrap();
        let scores = trainer.score(&heads, &tails).unwrap();
        let vals: Vec<f32> = scores.to_vec1().unwrap();
        for &v in &vals {
            assert!(v >= 0.0, "score should be non-negative, got {v}");
            assert!(v.is_finite(), "score should be finite");
        }
    }

    #[test]
    fn test_score_with_relations() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 3, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let tails = Tensor::from_vec(vec![3u32, 4, 5], (3,), &device).unwrap();
        let rels = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let scores = trainer.score_with_rel(&heads, &tails, &rels).unwrap();
        let vals: Vec<f32> = scores.to_vec1().unwrap();
        for &v in &vals {
            assert!(v >= 0.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ns_loss_backward() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 3, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1], (2,), &device).unwrap();
        let tails = Tensor::from_vec(vec![2u32, 3], (2,), &device).unwrap();
        let rels = Tensor::from_vec(vec![0u32, 1], (2,), &device).unwrap();
        let neg_tails = Tensor::from_vec(vec![4u32, 5, 6, 7], (4,), &device).unwrap();
        let neg_heads = Tensor::from_vec(vec![0u32, 1, 0, 1], (4,), &device).unwrap();
        let neg_rels = Tensor::from_vec(vec![0u32, 1, 0, 1], (4,), &device).unwrap();
        let (min_all, max_all) = trainer.entity_bounds().unwrap();

        let pos = trainer
            .batch_score(&min_all, &max_all, &heads, &tails, Some(&rels))
            .unwrap();
        let neg = trainer
            .batch_score(&min_all, &max_all, &neg_heads, &neg_tails, Some(&neg_rels))
            .unwrap();
        let loss = CandleBoxTrainer::ns_loss(&pos, &neg, 3.0, &device).unwrap();

        let grads = loss.backward().unwrap();
        assert_eq!(grads.get(trainer.mu.as_tensor()).unwrap().dims(), &[10, 8]);
        assert_eq!(
            grads
                .get(trainer.rel_offset.as_ref().unwrap().as_tensor())
                .unwrap()
                .dims(),
            &[3, 8]
        );
    }

    #[test]
    fn test_score_shape() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 0, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let tails = Tensor::from_vec(vec![3u32, 4, 5], (3,), &device).unwrap();
        let scores = trainer.score(&heads, &tails).unwrap();
        assert_eq!(scores.dims(), &[3]);
    }

    #[test]
    fn test_fit_loss_decreases() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(20, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let losses = trainer.fit(&triples, 200, 0.05, 4, 3.0, 2, 0.0).unwrap();
        let first_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let last_avg: f32 = losses[losses.len() - 10..].iter().sum::<f32>() / 10.0;
        assert!(
            last_avg < first_avg,
            "loss should decrease: first_10_avg={first_avg}, last_10_avg={last_avg}",
        );
    }

    #[test]
    fn test_fit_self_adversarial() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(20, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        // Should not panic with self-adversarial enabled
        let losses = trainer.fit(&triples, 50, 0.05, 4, 3.0, 4, 2.0).unwrap();
        assert_eq!(losses.len(), 50);
        assert!(losses[0].is_finite());
    }

    #[test]
    fn test_score_all_tails() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 2, 8, 10.0, &device).unwrap();
        let scores = trainer.score_all_tails(0, Some(0)).unwrap();
        assert_eq!(scores.dims(), &[10]);
        let vals: Vec<f32> = scores.to_vec1().unwrap();
        for &v in &vals {
            assert!(v >= 0.0, "violation scores should be non-negative");
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_score_all_heads() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 2, 8, 10.0, &device).unwrap();
        let scores = trainer.score_all_heads(3, Some(1)).unwrap();
        assert_eq!(scores.dims(), &[10]);
        let vals: Vec<f32> = scores.to_vec1().unwrap();
        for &v in &vals {
            assert!(v >= 0.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_evaluate_after_training() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 2, 8, 10.0, &device).unwrap();
        let train = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let _losses = trainer.fit(&train, 100, 0.05, 4, 3.0, 4, 0.0).unwrap();
        let (mrr, h1, h3, h10, mr) = trainer.evaluate(&train, &train).unwrap();
        assert!((0.0..=1.0).contains(&mrr), "MRR={mrr}");
        assert!((0.0..=1.0).contains(&h1));
        assert!((0.0..=1.0).contains(&h3));
        assert!((0.0..=1.0).contains(&h10));
        assert!(mr >= 1.0, "mean rank should be >= 1, got {mr}");
    }

    #[test]
    fn test_cosine_lr_schedule() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 0, 4, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 0, 3)];
        // Short run -- just check it doesn't panic with the LR schedule
        let losses = trainer.fit(&triples, 20, 0.1, 2, 3.0, 2, 0.0).unwrap();
        assert_eq!(losses.len(), 20);
        // Loss should decrease (LR starts high, decays)
        let first = losses[0];
        let last = losses[losses.len() - 1];
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn test_distance_polarity() {
        // Scoring = containment VIOLATION (lower = better containment).
        // relu(min_h - min_t) + relu(max_t - max_h) per dimension.
        let device = Device::Cpu;

        // Contained: tail [7,8] inside head [0,10]. Violation = 0.
        let min_h = Tensor::new(&[[0.0f32, 0.0]], &device).unwrap();
        let max_h = Tensor::new(&[[10.0f32, 10.0]], &device).unwrap();
        let min_t = Tensor::new(&[[7.0f32, 7.0]], &device).unwrap();
        let max_t = Tensor::new(&[[8.0f32, 8.0]], &device).unwrap();
        let contained = CandleBoxTrainer::distance(&min_h, &max_h, &min_t, &max_t, 0.0)
            .unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(contained < 1e-6, "contained tail should have ~0 violation, got {contained}");

        // Protruding: tail [4, 12] extends beyond head [0,10].
        // upper violation per dim = relu(12 - 10) = 2. Total = 4.
        let min_t2 = Tensor::new(&[[4.0f32, 4.0]], &device).unwrap();
        let max_t2 = Tensor::new(&[[12.0f32, 12.0]], &device).unwrap();
        let protruding = CandleBoxTrainer::distance(&min_h, &max_h, &min_t2, &max_t2, 0.0)
            .unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((protruding - 4.0).abs() < 1e-5, "expected violation=4.0, got {protruding}");

        // Lower = better containment.
        assert!(contained < protruding);
    }

    #[test]
    fn test_inside_distance_discriminates_contained() {
        // Both tails fully contained (violation=0), but one is centered, one is at the edge.
        // Inside distance should distinguish them (lower = more centered = better).
        let device = Device::Cpu;
        let min_h = Tensor::new(&[[0.0f32, 0.0]], &device).unwrap();
        let max_h = Tensor::new(&[[10.0f32, 10.0]], &device).unwrap();

        let min_t_center = Tensor::new(&[[4.0f32, 4.0]], &device).unwrap();
        let max_t_center = Tensor::new(&[[6.0f32, 6.0]], &device).unwrap();
        let min_t_edge = Tensor::new(&[[8.0f32, 8.0]], &device).unwrap();
        let max_t_edge = Tensor::new(&[[9.0f32, 9.0]], &device).unwrap();

        // Without inside weight: both have 0 violation, indistinguishable.
        let center_base = CandleBoxTrainer::distance(
            &min_h, &max_h, &min_t_center, &max_t_center, 0.0,
        ).unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        let edge_base = CandleBoxTrainer::distance(
            &min_h, &max_h, &min_t_edge, &max_t_edge, 0.0,
        ).unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(center_base < 1e-6 && edge_base < 1e-6, "both should have ~0 violation");

        // With inside weight: centered scores lower (better), edge scores higher.
        let center_inside = CandleBoxTrainer::distance(
            &min_h, &max_h, &min_t_center, &max_t_center, 0.1,
        ).unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        let edge_inside = CandleBoxTrainer::distance(
            &min_h, &max_h, &min_t_edge, &max_t_edge, 0.1,
        ).unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(
            center_inside < edge_inside,
            "centered ({center_inside}) should score lower (better) than edge ({edge_inside})"
        );
    }

    #[test]
    fn test_inside_weight_monotonic() {
        // For off-center contained pair, higher inside_weight => higher score.
        let device = Device::Cpu;
        let min_h = Tensor::new(&[[0.0f32, 0.0]], &device).unwrap();
        let max_h = Tensor::new(&[[10.0f32, 10.0]], &device).unwrap();
        let min_t = Tensor::new(&[[7.0f32, 7.0]], &device).unwrap();
        let max_t = Tensor::new(&[[8.0f32, 8.0]], &device).unwrap();

        let s0 = CandleBoxTrainer::distance(&min_h, &max_h, &min_t, &max_t, 0.0)
            .unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        let s1 = CandleBoxTrainer::distance(&min_h, &max_h, &min_t, &max_t, 0.1)
            .unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        let s2 = CandleBoxTrainer::distance(&min_h, &max_h, &min_t, &max_t, 0.5)
            .unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap()[0];

        assert!(s0 < 1e-6, "base violation should be 0 for contained pair");
        assert!(s1 > s0, "inside_weight=0.1 should add inside penalty: s0={s0}, s1={s1}");
        assert!(s2 > s1, "inside_weight=0.5 should add more: s1={s1}, s2={s2}");
    }

    #[test]
    fn test_fit_with_inside_weight() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(20, 2, 8, 10.0, &device)
            .unwrap()
            .with_inside_weight(0.05);
        assert!((trainer.inside_weight - 0.05).abs() < 1e-6);
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let losses = trainer.fit(&triples, 100, 0.05, 4, 3.0, 4, 0.0).unwrap();
        let first_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let last_avg: f32 = losses[losses.len() - 10..].iter().sum::<f32>() / 10.0;
        assert!(
            last_avg < first_avg,
            "loss should decrease with inside_weight: first_10={first_avg}, last_10={last_avg}"
        );
    }
}
