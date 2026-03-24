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
    /// Volume regularization weight. Penalizes large entity boxes to
    /// prevent the trivial solution (all boxes grow to contain everything).
    /// Typical values: 0.0001-0.001.
    pub vol_reg: f32,
    /// How often to recompute entity bounds during training (in batches).
    ///
    /// 0 = once per epoch (fast but stale). N = every N batches.
    /// For WN18RR (~340 batches/epoch), try 50-100.
    pub bounds_every: usize,
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
        // Initialize log_delta near 0 so delta = softplus(exp(0), beta) ≈ 1.0.
        // This gives half-width ~0.5, creating overlapping boxes at init. With tiny
        // boxes (log_delta=-1), almost all pairs have zero violation and zero gradient.
        let log_delta =
            Var::from_tensor(&Tensor::randn(0.0_f32, 0.1, (num_entities, dim), device)?)?;

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
            vol_reg: 0.0,
            bounds_every: 0,
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

    /// Set volume regularization weight. Penalizes large boxes.
    #[must_use]
    pub fn with_vol_reg(mut self, weight: f32) -> Self {
        self.vol_reg = weight;
        self
    }

    /// Set bounds recomputation frequency (batches). 0 = once per epoch.
    #[must_use]
    pub fn with_bounds_every(mut self, n: usize) -> Self {
        self.bounds_every = n;
        self
    }

    /// Compute (min_all, max_all) for the entire entity table.
    ///
    /// These bounds are used for index_select lookups, not directly
    /// differentiated. The autograd graph flows through `batch_score`
    /// which recomputes bounds for the selected entities.
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

    /// Compute bounds for specific entities (avoids full-table computation).
    fn entity_bounds_for(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let mu_sel = self.mu.as_tensor().index_select(ids, 0)?;
        let ld_sel = self.log_delta.as_tensor().index_select(ids, 0)?;
        let delta = softplus(&ld_sel.exp()?, self.beta)?;
        let hw = delta.affine(0.5, 0.0)?;
        let min = mu_sel.sub(&hw)?;
        let max = mu_sel.add(&hw)?;
        Ok((min, max))
    }

    /// Score a batch by computing bounds only for the needed entities.
    ///
    /// Returns per-sample scores (lower = better containment).
    /// This is the fast path used in training -- avoids full-table entity_bounds.
    fn batch_score_direct(
        &self,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        rel_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (mut min_h, mut max_h) = self.entity_bounds_for(head_ids)?;
        let (min_t, max_t) = self.entity_bounds_for(tail_ids)?;

        if let (Some(ref rel_var), Some(rel)) = (&self.rel_offset, rel_ids) {
            let offset = rel_var.as_tensor().index_select(rel, 0)?;
            min_h = min_h.add(&offset)?;
            max_h = max_h.add(&offset)?;
        }

        let dist = Self::distance(&min_h, &max_h, &min_t, &max_t, self.inside_weight)?;
        dist.sum(1)
    }

    /// Score a batch from pre-computed entity bounds (used by public score API).
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

            // Entity bounds are now computed per-batch inside batch_score
            // (only for the entities in the batch, not the full table).
            // The bounds_every mechanism is no longer needed for training
            // since each batch computes its own fresh bounds.

            // Precompute all negative entity IDs for the epoch (one big rand).
            // This avoids per-batch Tensor::rand + dtype cast + clamp overhead.
            let ne = self.num_entities;
            let num_batches = n.div_ceil(batch_size);
            let neg_per_batch = batch_size * negative_samples;
            let total_neg_epoch = num_batches * neg_per_batch;
            let all_neg_ids = Tensor::rand(0.0_f32, ne as f32, (total_neg_epoch,), &self.device)?
                .to_dtype(candle_core::DType::U32)?
                .clamp(0u32, (ne - 1) as u32)?;

            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let bs = batch_end - batch_start;

                let h_t = heads_shuf.narrow(0, batch_start, bs)?;
                let r_t = rels_shuf.narrow(0, batch_start, bs)?;
                let t_t = tails_shuf.narrow(0, batch_start, bs)?;

                // Build all head/tail/rel IDs for positives + negatives in one tensor.
                // One batch_score_direct call = one autograd graph (2x faster).
                let total_neg = bs * negative_samples;
                let half_neg = total_neg / 2;
                let neg_offset = batch_count * neg_per_batch;
                let remaining = total_neg - half_neg;

                // Slice precomputed negative IDs for this batch.
                let batch_neg = all_neg_ids.narrow(0, neg_offset, total_neg)?;
                let neg_tail_ids = batch_neg.narrow(0, 0, half_neg)?;
                let neg_head_ids = batch_neg.narrow(0, half_neg, remaining)?;

                // Build fused head/tail/rel: [positives, neg_tail_corrupt, neg_head_corrupt]
                let neg_h_for_t = h_t
                    .repeat((half_neg.div_ceil(bs),))?
                    .narrow(0, 0, half_neg)?;
                let neg_r_for_t = r_t
                    .repeat((half_neg.div_ceil(bs),))?
                    .narrow(0, 0, half_neg)?;
                let neg_t_for_h = t_t
                    .repeat((remaining.div_ceil(bs),))?
                    .narrow(0, 0, remaining)?;
                let neg_r_for_h = r_t
                    .repeat((remaining.div_ceil(bs),))?
                    .narrow(0, 0, remaining)?;

                let fused_h = Tensor::cat(&[&h_t, &neg_h_for_t, &neg_head_ids], 0)?;
                let fused_t = Tensor::cat(&[&t_t, &neg_tail_ids, &neg_t_for_h], 0)?;

                let (fused_scores, rel_ref) = if self.num_relations > 0 {
                    let fused_r = Tensor::cat(&[&r_t, &neg_r_for_t, &neg_r_for_h], 0)?;
                    (
                        self.batch_score_direct(&fused_h, &fused_t, Some(&fused_r))?,
                        true,
                    )
                } else {
                    (self.batch_score_direct(&fused_h, &fused_t, None)?, false)
                };
                let _ = rel_ref; // used only for control flow above

                // Split fused scores back into positive and negative.
                let pos_scores = fused_scores.narrow(0, 0, bs)?;
                let neg_scores = fused_scores.narrow(0, bs, total_neg)?;

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

                // Volume regularization: penalize mean box width in this batch.
                // Prevents all boxes from growing to contain everything (trivial solution).
                // Uses exp(log_delta) (always positive) instead of raw log_delta
                // (which can be negative, inverting the penalty).
                let loss = if self.vol_reg > 0.0 {
                    let batch_entities = Tensor::cat(&[&h_t, &t_t], 0)?;
                    let batch_ld = self
                        .log_delta
                        .as_tensor()
                        .index_select(&batch_entities, 0)?;
                    let batch_width = batch_ld.exp()?;
                    let vol_penalty = batch_width.mean_all()?.affine(self.vol_reg as f64, 0.0)?;
                    loss.add(&vol_penalty)?
                } else {
                    loss
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

    /// Find the k entities most likely to subsume (contain) the given entity.
    ///
    /// Returns `(entity_ids, scores)` sorted by ascending score (best first).
    /// "Entity A subsumes entity B" means A's box contains B's box.
    /// So we find heads whose boxes best contain the query entity (as tail).
    pub fn query_subsumers(
        &self,
        entity_id: usize,
        k: usize,
        rel_id: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let scores: Vec<f32> = self.score_all_heads(entity_id, rel_id)?.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = k.min(indexed.len());
        let ids: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
        let vals: Vec<f32> = indexed[..k].iter().map(|(_, s)| *s).collect();
        Ok((ids, vals))
    }

    /// Find the k entities most likely to be subsumed by (contained in) the given entity.
    ///
    /// Returns `(entity_ids, scores)` sorted by ascending score (best first).
    /// "Entity A subsumes entity B" means A's box contains B's box.
    /// So we find tails whose boxes are best contained by the query entity (as head).
    pub fn query_subsumed(
        &self,
        entity_id: usize,
        k: usize,
        rel_id: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let scores: Vec<f32> = self.score_all_tails(entity_id, rel_id)?.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = k.min(indexed.len());
        let ids: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
        let vals: Vec<f32> = indexed[..k].iter().map(|(_, s)| *s).collect();
        Ok((ids, vals))
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
            .unwrap()
            .sum(1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(
            contained < 1e-6,
            "contained tail should have ~0 violation, got {contained}"
        );

        // Protruding: tail [4, 12] extends beyond head [0,10].
        // upper violation per dim = relu(12 - 10) = 2. Total = 4.
        let min_t2 = Tensor::new(&[[4.0f32, 4.0]], &device).unwrap();
        let max_t2 = Tensor::new(&[[12.0f32, 12.0]], &device).unwrap();
        let protruding = CandleBoxTrainer::distance(&min_h, &max_h, &min_t2, &max_t2, 0.0)
            .unwrap()
            .sum(1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(
            (protruding - 4.0).abs() < 1e-5,
            "expected violation=4.0, got {protruding}"
        );

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
        let center_base =
            CandleBoxTrainer::distance(&min_h, &max_h, &min_t_center, &max_t_center, 0.0)
                .unwrap()
                .sum(1)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
        let edge_base = CandleBoxTrainer::distance(&min_h, &max_h, &min_t_edge, &max_t_edge, 0.0)
            .unwrap()
            .sum(1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(
            center_base < 1e-6 && edge_base < 1e-6,
            "both should have ~0 violation"
        );

        // With inside weight: centered scores lower (better), edge scores higher.
        let center_inside =
            CandleBoxTrainer::distance(&min_h, &max_h, &min_t_center, &max_t_center, 0.1)
                .unwrap()
                .sum(1)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
        let edge_inside = CandleBoxTrainer::distance(&min_h, &max_h, &min_t_edge, &max_t_edge, 0.1)
            .unwrap()
            .sum(1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
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
            .unwrap()
            .sum(1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let s1 = CandleBoxTrainer::distance(&min_h, &max_h, &min_t, &max_t, 0.1)
            .unwrap()
            .sum(1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let s2 = CandleBoxTrainer::distance(&min_h, &max_h, &min_t, &max_t, 0.5)
            .unwrap()
            .sum(1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];

        assert!(
            s0 < 1e-6,
            "base violation should be 0 for contained pair, got {s0}"
        );
        assert!(
            s1 > s0,
            "inside_weight=0.1 should add inside penalty: s0={s0}, s1={s1}"
        );
        assert!(
            s2 > s1,
            "inside_weight=0.5 should add more: s1={s1}, s2={s2}"
        );
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

    #[test]
    fn test_positive_scores_decrease_with_training() {
        // After training, positive triples should have LOWER violation scores
        // than before training (model learns containment).
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];

        // Score positives before training.
        let h = Tensor::from_vec(vec![0u32, 2, 4, 6], (4,), &device).unwrap();
        let r = Tensor::from_vec(vec![0u32, 1, 0, 1], (4,), &device).unwrap();
        let t = Tensor::from_vec(vec![1u32, 3, 5, 7], (4,), &device).unwrap();
        let scores_before: Vec<f32> = trainer
            .score_with_rel(&h, &t, &r)
            .unwrap()
            .to_vec1()
            .unwrap();
        let avg_before: f32 = scores_before.iter().sum::<f32>() / scores_before.len() as f32;

        // Train.
        let _losses = trainer.fit(&triples, 200, 0.05, 4, 3.0, 4, 0.0).unwrap();

        // Score positives after training.
        let scores_after: Vec<f32> = trainer
            .score_with_rel(&h, &t, &r)
            .unwrap()
            .to_vec1()
            .unwrap();
        let avg_after: f32 = scores_after.iter().sum::<f32>() / scores_after.len() as f32;

        assert!(
            avg_after < avg_before,
            "positive violation should decrease with training: before={avg_before}, after={avg_after}"
        );
    }

    #[test]
    fn test_negative_scores_higher_than_positive() {
        // After training, random negatives should have higher (worse) scores
        // than positive triples on average.
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let _losses = trainer.fit(&triples, 300, 0.05, 4, 3.0, 4, 0.0).unwrap();

        // Score positive triples.
        let h_pos = Tensor::from_vec(vec![0u32, 2, 4, 6], (4,), &device).unwrap();
        let r_pos = Tensor::from_vec(vec![0u32, 1, 0, 1], (4,), &device).unwrap();
        let t_pos = Tensor::from_vec(vec![1u32, 3, 5, 7], (4,), &device).unwrap();
        let pos_scores: Vec<f32> = trainer
            .score_with_rel(&h_pos, &t_pos, &r_pos)
            .unwrap()
            .to_vec1()
            .unwrap();
        let avg_pos: f32 = pos_scores.iter().sum::<f32>() / pos_scores.len() as f32;

        // Score random negative triples (random tails).
        let h_neg = Tensor::from_vec(vec![0u32, 2, 4, 6], (4,), &device).unwrap();
        let r_neg = Tensor::from_vec(vec![0u32, 1, 0, 1], (4,), &device).unwrap();
        let t_neg = Tensor::from_vec(vec![8u32, 9, 0, 2], (4,), &device).unwrap();
        let neg_scores: Vec<f32> = trainer
            .score_with_rel(&h_neg, &t_neg, &r_neg)
            .unwrap()
            .to_vec1()
            .unwrap();
        let avg_neg: f32 = neg_scores.iter().sum::<f32>() / neg_scores.len() as f32;

        assert!(
            avg_neg > avg_pos,
            "negatives ({avg_neg}) should score higher (worse) than positives ({avg_pos})"
        );
    }

    #[test]
    fn test_evaluate_mrr_improves_with_training() {
        // MRR should improve after training vs random init.
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];

        let (mrr_before, _, _, _, _) = trainer.evaluate(&triples, &triples).unwrap();

        let _losses = trainer.fit(&triples, 300, 0.05, 4, 3.0, 4, 0.0).unwrap();

        let (mrr_after, _, _, _, _) = trainer.evaluate(&triples, &triples).unwrap();

        assert!(
            mrr_after > mrr_before,
            "MRR should improve with training: before={mrr_before}, after={mrr_after}"
        );
    }

    #[test]
    fn test_scores_nonnegative() {
        // All scores should be non-negative (violation + inside distance >= 0).
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(20, 3, 16, 10.0, &device)
            .unwrap()
            .with_inside_weight(0.1);
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 2, 5)];
        let _losses = trainer.fit(&triples, 50, 0.05, 3, 3.0, 2, 0.0).unwrap();

        // Score all pairs for all relations.
        for h in 0..20 {
            let scores = trainer.score_all_tails(h, Some(0)).unwrap();
            let vals: Vec<f32> = scores.to_vec1().unwrap();
            for (tid, &v) in vals.iter().enumerate() {
                assert!(v >= -1e-6, "score({h}, 0, {tid}) = {v} should be >= 0");
                assert!(v.is_finite(), "score({h}, 0, {tid}) = {v} should be finite");
            }
        }
    }

    #[test]
    fn test_gradient_flows_through_inside_distance() {
        // Verify autograd propagates through the inside distance (soft mask + division).
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 0, 4, 10.0, &device)
            .unwrap()
            .with_inside_weight(0.1);
        let triples = vec![(0, 0, 1), (2, 0, 3)];
        // If gradient doesn't flow, this would panic or produce NaN loss.
        let losses = trainer.fit(&triples, 20, 0.05, 2, 3.0, 2, 0.0).unwrap();
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "all losses should be finite"
        );
    }

    #[test]
    fn test_vol_reg_constrains_box_size() {
        // Volume regularization should prevent boxes from growing too large.
        let device = Device::Cpu;

        // Train without vol_reg.
        let t1 = CandleBoxTrainer::new(10, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let _l1 = t1.fit(&triples, 200, 0.05, 4, 3.0, 4, 0.0).unwrap();
        let ld1: Vec<f32> = t1
            .log_delta
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let avg_ld1: f32 = ld1.iter().sum::<f32>() / ld1.len() as f32;

        // Train with vol_reg.
        let t2 = CandleBoxTrainer::new(10, 2, 8, 10.0, &device)
            .unwrap()
            .with_vol_reg(0.1);
        let _l2 = t2.fit(&triples, 200, 0.05, 4, 3.0, 4, 0.0).unwrap();
        let ld2: Vec<f32> = t2
            .log_delta
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let avg_ld2: f32 = ld2.iter().sum::<f32>() / ld2.len() as f32;

        // With vol_reg, average log_delta should be smaller (boxes don't grow as much).
        assert!(
            avg_ld2 < avg_ld1,
            "vol_reg should constrain box growth: without={avg_ld1}, with={avg_ld2}"
        );
    }

    #[test]
    fn test_query_subsumers_returns_k_results() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(20, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let _losses = trainer.fit(&triples, 100, 0.05, 4, 3.0, 4, 0.0).unwrap();

        let (ids, scores) = trainer.query_subsumers(1, 5, Some(0)).unwrap();
        assert_eq!(ids.len(), 5);
        assert_eq!(scores.len(), 5);
        // Scores should be sorted ascending (lower = better containment).
        for w in scores.windows(2) {
            assert!(w[0] <= w[1], "scores should be sorted ascending");
        }
    }

    #[test]
    fn test_query_subsumed_returns_k_results() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(20, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let _losses = trainer.fit(&triples, 100, 0.05, 4, 3.0, 4, 0.0).unwrap();

        let (ids, scores) = trainer.query_subsumed(0, 5, Some(0)).unwrap();
        assert_eq!(ids.len(), 5);
        assert_eq!(scores.len(), 5);
        for w in scores.windows(2) {
            assert!(w[0] <= w[1], "scores should be sorted ascending");
        }
    }

    #[test]
    fn test_query_subsumers_trained_finds_correct_head() {
        // After training (0, rel=0, 1), entity 0 should subsume entity 1.
        // So query_subsumers(1) should rank entity 0 near the top.
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 2, 8, 10.0, &device).unwrap();
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let _losses = trainer.fit(&triples, 300, 0.05, 4, 3.0, 4, 0.0).unwrap();

        let (ids, _scores) = trainer.query_subsumers(1, 10, Some(0)).unwrap();
        // Entity 0 should be in the top 10 subsumers of entity 1.
        assert!(
            ids.contains(&0),
            "entity 0 should subsume entity 1 after training, got top-10: {ids:?}"
        );
    }
}
