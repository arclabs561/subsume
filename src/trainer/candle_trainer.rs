//! Candle-based box embedding trainer with autograd.
//!
//! Uses candle tensors for GPU-accelerated training with automatic
//! differentiation. Matches the probabilistic containment loss from
//! the ndarray trainer: `-ln P(B ⊆ A)` where `P(B ⊆ A) = Vol(A ∩ B) / Vol(B)`,
//! with softplus-smoothed intersection volumes for gradient flow through
//! disjoint boxes.
//!
//! # Architecture
//!
//! Entity embeddings are stored as two `Var` tensors:
//! - `mu`: center positions, shape `[num_entities, dim]`
//! - `log_delta`: log-widths, shape `[num_entities, dim]`
//!
//! Box bounds are derived via:
//! - `min = mu - delta/2`, `max = mu + delta/2`
//! - `delta = softplus(exp(log_delta), beta)` (Gumbel parameterization)
//!
//! # Memory optimization
//!
//! The entity table transforms (`exp -> softplus -> bounds`) are computed
//! once per batch step into `(min_all, max_all)`, then batch-indexed via
//! `index_select`. This keeps the autograd graph O(1) in entity count
//! rather than O(negative_samples).

use candle_core::{Device, Result, Tensor, Var};

/// Candle-based box embedding trainer with probabilistic containment loss.
pub struct CandleBoxTrainer {
    /// Entity center positions: `[num_entities, dim]`.
    pub mu: Var,
    /// Entity log-widths: `[num_entities, dim]`.
    pub log_delta: Var,
    /// Per-relation translation offsets: `[num_relations, dim]`.
    /// `None` if no relations are used (identity transform).
    pub rel_offset: Option<Var>,
    /// Embedding dimension.
    pub dim: usize,
    /// Number of entities.
    pub num_entities: usize,
    /// Number of relations (0 if identity).
    pub num_relations: usize,
    /// Gumbel beta parameter.
    pub beta: f32,
    /// Device (CPU or CUDA).
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
            device: device.clone(),
        })
    }

    /// Compute (min_all, max_all) for the entire entity table.
    ///
    /// Returns `[num_entities, dim]` tensors. Call once per batch step,
    /// then use `index_select` for batch-sized lookups.
    fn entity_bounds(&self) -> Result<(Tensor, Tensor)> {
        let exp_ld = self.log_delta.as_tensor().exp()?;
        let delta = softplus(&exp_ld, self.beta)?;
        let hw = delta.affine(0.5, 0.0)?;
        let mu = self.mu.as_tensor();
        let min_all = mu.sub(&hw)?;
        let max_all = mu.add(&hw)?;
        Ok((min_all, max_all))
    }

    /// Per-dimension containment violation between head and tail boxes.
    ///
    /// For each dimension d:
    ///   `violation_d = max(0, max(min_t_d - min_h_d, 0) + max(max_h_d - max_t_d, 0))`
    ///
    /// This measures how far head is from containing tail, per dimension.
    /// Returns `[batch, dim]` tensor of violations.
    fn containment_violation(
        min_h: &Tensor,
        max_h: &Tensor,
        min_t: &Tensor,
        max_t: &Tensor,
    ) -> Result<Tensor> {
        // Head's min must be <= tail's min (head contains tail from below)
        let lower_violation = min_t.sub(min_h)?.relu()?;
        // Head's max must be >= tail's max (head contains tail from above)
        let upper_violation = max_h.sub(max_t)?.relu()?;
        // Combined per-dimension violation
        lower_violation.add(&upper_violation)
    }

    /// Score a batch from pre-computed entity bounds.
    ///
    /// Returns per-dim containment violation summed over dimensions (L1 score).
    /// Lower = head better contains tail.
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

        // Apply relation translation to head
        if let (Some(ref rel_var), Some(rel)) = (&self.rel_offset, rel_ids) {
            let offset = rel_var.as_tensor().index_select(rel, 0)?;
            min_h = min_h.add(&offset)?;
            max_h = max_h.add(&offset)?;
        }

        let violation = Self::containment_violation(&min_h, &max_h, &min_t, &max_t)?;
        // L1 norm across dimensions (sum, not mean -- matches BoxE)
        violation.sum(1)
    }

    /// Negative sampling loss with log-sigmoid (BoxE-style).
    ///
    /// `loss = -mean(log_sigmoid(margin - pos_score))
    ///         -mean(log_sigmoid(neg_score - margin))`
    ///
    /// Lower score = better containment. Positive triples should score
    /// below margin; negatives should score above margin.
    fn ns_loss(
        pos_scores: &Tensor,
        neg_scores: &Tensor,
        margin: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let margin_p = Tensor::full(margin, pos_scores.shape(), device)?;
        let margin_n = Tensor::full(margin, neg_scores.shape(), device)?;

        // Positive: log sigmoid(margin - score) -- want score < margin
        let pos_term = log_sigmoid(&margin_p.sub(pos_scores)?)?;
        // Negative: log sigmoid(score - margin) -- want score > margin
        let neg_term = log_sigmoid(&neg_scores.sub(&margin_n)?)?;

        // Minimize -(mean(pos_term) + mean(neg_term))
        let loss = pos_term.mean(0)?.add(&neg_term.mean(0)?)?.neg()?;
        Ok(loss)
    }

    /// Score a batch of (head, tail) pairs: lower = better containment.
    ///
    /// Returns L1 containment violation per pair.
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

    /// Train on triples with AdamW optimizer.
    ///
    /// Triples are `(head_id, relation_id, tail_id)`. Shuffles each epoch.
    /// Returns per-epoch average losses.
    pub fn fit(
        &self,
        train_triples: &[(usize, usize, usize)],
        epochs: usize,
        lr: f64,
        batch_size: usize,
        margin: f32,
        negative_samples: usize,
    ) -> Result<Vec<f32>> {
        use candle_nn::{AdamW, Optimizer, ParamsAdamW};

        let mut vars = vec![self.mu.clone(), self.log_delta.clone()];
        if let Some(ref rel) = self.rel_offset {
            vars.push(rel.clone());
        }

        let params = ParamsAdamW {
            lr,
            weight_decay: 1e-4,
            ..Default::default()
        };
        let mut opt = AdamW::new(vars, params)?;
        let n = train_triples.len();
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut rng: u64 = 42;

        let lcg = |s: &mut u64| -> usize {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as usize
        };

        // Preload ALL training data as device tensors (zero CPU-GPU transfer in loop)
        let all_heads: Vec<u32> = train_triples.iter().map(|t| t.0 as u32).collect();
        let all_rels: Vec<u32> = train_triples.iter().map(|t| t.1 as u32).collect();
        let all_tails: Vec<u32> = train_triples.iter().map(|t| t.2 as u32).collect();
        let heads_gpu = Tensor::from_vec(all_heads, (n,), &self.device)?;
        let rels_gpu = Tensor::from_vec(all_rels, (n,), &self.device)?;
        let tails_gpu = Tensor::from_vec(all_tails, (n,), &self.device)?;

        // CPU-side permutation (cheap -- just u32 indices)
        let mut indices: Vec<u32> = (0..n as u32).collect();

        for epoch in 0..epochs {
            // Fisher-Yates shuffle (CPU -- fast for index permutation)
            for i in (1..n).rev() {
                let j = lcg(&mut rng) % (i + 1);
                indices.swap(i, j);
            }

            // Upload permutation once per epoch
            let perm = Tensor::from_vec(indices.clone(), (n,), &self.device)?;
            let heads_shuf = heads_gpu.index_select(&perm, 0)?;
            let rels_shuf = rels_gpu.index_select(&perm, 0)?;
            let tails_shuf = tails_gpu.index_select(&perm, 0)?;

            let mut total_loss = 0.0f32;
            let mut batch_count = 0usize;

            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let bs = batch_end - batch_start;

                // Compute entity bounds per step (params change after backward)
                let (min_all, max_all) = self.entity_bounds()?;

                // Slice batch from pre-shuffled GPU tensors (no CPU-GPU transfer)
                let h_t = heads_shuf.narrow(0, batch_start, bs)?;
                let r_t = rels_shuf.narrow(0, batch_start, bs)?;
                let t_t = tails_shuf.narrow(0, batch_start, bs)?;

                let rel_ref = if self.num_relations > 0 {
                    Some(&r_t)
                } else {
                    None
                };

                // Positive scores
                let pos_scores =
                    self.batch_score(&min_all, &max_all, &h_t, &t_t, rel_ref)?;

                // Generate negative tails on device (no CPU-GPU transfer)
                let total_neg = bs * negative_samples;
                let neg_rand = Tensor::rand(
                    0.0_f32,
                    self.num_entities as f32,
                    (total_neg,),
                    &self.device,
                )?;
                let neg_t = neg_rand.to_dtype(candle_core::DType::U32)?;

                // Repeat head/rel for each negative sample
                let neg_h = h_t.repeat((negative_samples,))?;
                let neg_r = r_t.repeat((negative_samples,))?;

                let neg_rel_ref = if self.num_relations > 0 {
                    Some(&neg_r)
                } else {
                    None
                };
                let neg_scores = self.batch_score(
                    &min_all, &max_all, &neg_h, &neg_t, neg_rel_ref,
                )?;

                let loss = Self::ns_loss(&pos_scores, &neg_scores, margin, &self.device)?;
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
}

/// Log-sigmoid: `ln(sigmoid(x)) = -softplus(-x, 1)`, numerically stable.
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    // log_sigmoid(x) = x - softplus(x, 1) = -softplus(-x, 1)
    let neg_x = x.neg()?;
    let sp = softplus(&neg_x, 1.0)?;
    sp.neg()
}

/// Softplus: `(1/beta) * ln(1 + exp(beta * x))`, numerically stable.
///
/// For large beta, this approaches `max(0, x)` (sharp ReLU approximation).
/// Matches `crate::utils::softplus`.
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
        assert_eq!(trainer.dim, 32);
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
        let (min_all, max_all) = trainer.entity_bounds().unwrap();
        let scores = trainer
            .batch_score(&min_all, &max_all, &heads, &tails, None)
            .unwrap();
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
        let (min_all, max_all) = trainer.entity_bounds().unwrap();
        let scores = trainer
            .batch_score(&min_all, &max_all, &heads, &tails, Some(&rels))
            .unwrap();
        let vals: Vec<f32> = scores.to_vec1().unwrap();
        for &v in &vals {
            assert!(v >= 0.0, "score should be non-negative, got {v}");
            assert!(v.is_finite(), "score should be finite");
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
        let mu_grad = grads.get(trainer.mu.as_tensor()).unwrap();
        assert_eq!(mu_grad.dims(), &[10, 8]);

        let rel_grad = grads
            .get(trainer.rel_offset.as_ref().unwrap().as_tensor())
            .unwrap();
        assert_eq!(rel_grad.dims(), &[3, 8]);
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
        let losses = trainer.fit(&triples, 200, 0.05, 4, 0.1, 2).unwrap();
        let first_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let last_avg: f32 = losses[losses.len() - 10..].iter().sum::<f32>() / 10.0;
        assert!(
            last_avg < first_avg,
            "loss should decrease: first_10_avg={first_avg}, last_10_avg={last_avg}",
        );
    }
}
