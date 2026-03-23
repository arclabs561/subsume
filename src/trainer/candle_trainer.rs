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
            device: device.clone(),
        })
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

    /// Per-dimension containment violation (head containing tail).
    fn containment_violation(
        min_h: &Tensor,
        max_h: &Tensor,
        min_t: &Tensor,
        max_t: &Tensor,
    ) -> Result<Tensor> {
        let lower_violation = min_t.sub(min_h)?.relu()?;
        let upper_violation = max_h.sub(max_t)?.relu()?;
        lower_violation.add(&upper_violation)
    }

    /// Score a batch from pre-computed entity bounds (L1 containment violation).
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

        let violation = Self::containment_violation(&min_h, &max_h, &min_t, &max_t)?;
        violation.sum(1)
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
        let weights = candle_nn::ops::softmax(
            &neg_2d.affine(adv_temp as f64, 0.0)?.detach(),
            1,
        )?;

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

            let mut total_loss = 0.0f32;
            let mut batch_count = 0usize;

            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let bs = batch_end - batch_start;

                let (min_all, max_all) = self.entity_bounds()?;

                let h_t = heads_shuf.narrow(0, batch_start, bs)?;
                let r_t = rels_shuf.narrow(0, batch_start, bs)?;
                let t_t = tails_shuf.narrow(0, batch_start, bs)?;

                let rel_ref = if self.num_relations > 0 {
                    Some(&r_t)
                } else {
                    None
                };

                let pos_scores =
                    self.batch_score(&min_all, &max_all, &h_t, &t_t, rel_ref)?;

                // Corrupt both head and tail (half each)
                let total_neg = bs * negative_samples;
                let half_neg = total_neg / 2;

                // Tail corruption: keep head, randomize tail
                let neg_rand_t = Tensor::rand(
                    0.0_f32,
                    self.num_entities as f32,
                    (half_neg,),
                    &self.device,
                )?;
                let neg_t_ids = neg_rand_t.to_dtype(candle_core::DType::U32)?;
                let neg_h_for_t = h_t.repeat(((half_neg + bs - 1) / bs,))?.narrow(0, 0, half_neg)?;
                let neg_r_for_t = r_t.repeat(((half_neg + bs - 1) / bs,))?.narrow(0, 0, half_neg)?;

                // Head corruption: keep tail, randomize head
                let neg_rand_h = Tensor::rand(
                    0.0_f32,
                    self.num_entities as f32,
                    (total_neg - half_neg,),
                    &self.device,
                )?;
                let neg_h_ids = neg_rand_h.to_dtype(candle_core::DType::U32)?;
                let remaining = total_neg - half_neg;
                let neg_t_for_h = t_t.repeat(((remaining + bs - 1) / bs,))?.narrow(0, 0, remaining)?;
                let neg_r_for_h = r_t.repeat(((remaining + bs - 1) / bs,))?.narrow(0, 0, remaining)?;

                // Concatenate all negatives
                let all_neg_h = Tensor::cat(&[&neg_h_for_t, &neg_h_ids], 0)?;
                let all_neg_t = Tensor::cat(&[&neg_t_ids, &neg_t_for_h], 0)?;
                let all_neg_r = Tensor::cat(&[&neg_r_for_t, &neg_r_for_h], 0)?;

                let neg_rel_ref = if self.num_relations > 0 {
                    Some(&all_neg_r)
                } else {
                    None
                };
                let neg_scores = self.batch_score(
                    &min_all, &max_all, &all_neg_h, &all_neg_t, neg_rel_ref,
                )?;

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
}
