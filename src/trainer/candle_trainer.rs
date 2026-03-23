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
//! The loss is `-ln(Vol(A ∩ B) / Vol(B))` for positives,
//! `max(0, P_overlap - margin)^2` for negatives.

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

    /// Get box half-widths from log_delta via softplus.
    fn get_half_width(&self) -> Result<Tensor> {
        let exp_ld = self.log_delta.as_tensor().exp()?;
        let delta = softplus(&exp_ld, self.beta)?;
        delta.affine(0.5, 0.0)
    }

    /// Get (min, max) bounds for a batch of entity IDs.
    fn get_bounds(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let mu = self.mu.as_tensor().index_select(ids, 0)?;
        let hw = self.get_half_width()?.index_select(ids, 0)?;
        let min = mu.sub(&hw)?;
        let max = mu.add(&hw)?;
        Ok((min, max))
    }

    /// Compute softplus-smoothed intersection volume in log space.
    ///
    /// Returns `ln(Vol(A ∩ B))` computed as sum of `ln(softplus(hi - lo, beta))`
    /// per dimension. This avoids the numerical issues of multiplying many
    /// small softplus values directly.
    fn log_intersection_volume(
        &self,
        min_a: &Tensor,
        max_a: &Tensor,
        min_b: &Tensor,
        max_b: &Tensor,
    ) -> Result<Tensor> {
        // lo = max(min_a, min_b), hi = min(max_a, max_b)
        let lo = min_a.maximum(min_b)?;
        let hi = max_a.minimum(max_b)?;
        let diff = hi.sub(&lo)?;
        // softplus(diff, beta) per element, then log, then sum over dims.
        // Add eps before log to handle disjoint boxes where softplus -> 0.
        let sp = softplus(&diff, self.beta)?;
        let log_sp = sp.clamp(1e-30, f32::MAX)?.log()?;
        log_sp.sum(1) // [batch]
    }

    /// Compute log-volume of boxes for a batch of entity IDs.
    fn log_volume(&self, ids: &Tensor) -> Result<Tensor> {
        let hw = self.get_half_width()?.index_select(ids, 0)?;
        // volume = prod(2 * hw) per entity, log_vol = sum(ln(2 * hw))
        let width = hw.affine(2.0, 0.0)?;
        let log_w = width.clamp(1e-30, f32::MAX)?.log()?;
        log_w.sum(1)
    }

    /// Probabilistic containment loss for positive triples.
    ///
    /// Loss = `-ln P(B ⊆ A)` where `P(B ⊆ A) = Vol(A ∩ B) / Vol(B)`.
    /// When relation IDs are provided, head boxes are translated by the
    /// relation offset before computing containment.
    pub fn positive_loss(
        &self,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        rel_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (mut min_h, mut max_h) = self.get_bounds(head_ids)?;
        let (min_t, max_t) = self.get_bounds(tail_ids)?;

        // Apply relation translation to head box
        if let (Some(ref rel_var), Some(rel)) = (&self.rel_offset, rel_ids) {
            let offset = rel_var.as_tensor().index_select(rel, 0)?;
            min_h = min_h.add(&offset)?;
            max_h = max_h.add(&offset)?;
        }

        let log_vol_int = self.log_intersection_volume(&min_h, &max_h, &min_t, &max_t)?;
        let log_vol_t = self.log_volume(tail_ids)?;

        // -ln(P) = -(log_vol_int - log_vol_t), clamped to [0, 10]
        let neg_log_prob = log_vol_t.sub(&log_vol_int)?;
        neg_log_prob.clamp(0.0, 10.0)?.mean(0)
    }

    /// Negative loss: penalize overlap between head and corrupted tail.
    ///
    /// Loss = `max(0, P_overlap - margin)^2` where `P_overlap = Vol(A ∩ B) / Vol(B)`.
    pub fn negative_loss(
        &self,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        rel_ids: Option<&Tensor>,
        margin: f32,
    ) -> Result<Tensor> {
        let (mut min_h, mut max_h) = self.get_bounds(head_ids)?;
        let (min_t, max_t) = self.get_bounds(tail_ids)?;

        if let (Some(ref rel_var), Some(rel)) = (&self.rel_offset, rel_ids) {
            let offset = rel_var.as_tensor().index_select(rel, 0)?;
            min_h = min_h.add(&offset)?;
            max_h = max_h.add(&offset)?;
        }

        let log_vol_int = self.log_intersection_volume(&min_h, &max_h, &min_t, &max_t)?;
        let log_vol_t = self.log_volume(tail_ids)?;

        // P = exp(log_vol_int - log_vol_t), clamped to [0, 1]
        let log_prob = log_vol_int.sub(&log_vol_t)?;
        let prob = log_prob.clamp(-20.0, 0.0)?.exp()?;

        // max(0, prob - margin)^2
        let margin_t = Tensor::full(margin, prob.shape(), &self.device)?;
        let violation = prob.sub(&margin_t)?.relu()?;
        violation.sqr()?.mean(0)
    }

    /// Score a batch of (head, tail) pairs: lower = better containment.
    ///
    /// Returns `-ln P(B ⊆ A)` per pair.
    pub fn score(&self, head_ids: &Tensor, tail_ids: &Tensor) -> Result<Tensor> {
        let (min_h, max_h) = self.get_bounds(head_ids)?;
        let (min_t, max_t) = self.get_bounds(tail_ids)?;

        let log_vol_int = self.log_intersection_volume(&min_h, &max_h, &min_t, &max_t)?;
        let log_vol_t = self.log_volume(tail_ids)?;

        // -ln P = log_vol_t - log_vol_int
        log_vol_t.sub(&log_vol_int)
    }

    /// Score with relation translation applied to head.
    pub fn score_with_rel(
        &self,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        rel_ids: &Tensor,
    ) -> Result<Tensor> {
        let (mut min_h, mut max_h) = self.get_bounds(head_ids)?;
        let (min_t, max_t) = self.get_bounds(tail_ids)?;

        if let Some(ref rel_var) = self.rel_offset {
            let offset = rel_var.as_tensor().index_select(rel_ids, 0)?;
            min_h = min_h.add(&offset)?;
            max_h = max_h.add(&offset)?;
        }

        let log_vol_int = self.log_intersection_volume(&min_h, &max_h, &min_t, &max_t)?;
        let log_vol_t = self.log_volume(tail_ids)?;
        log_vol_t.sub(&log_vol_int)
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

        // Shuffled index for epoch iteration
        let mut indices: Vec<usize> = (0..n).collect();

        for epoch in 0..epochs {
            // Fisher-Yates shuffle using LCG
            for i in (1..n).rev() {
                let j = lcg(&mut rng) % (i + 1);
                indices.swap(i, j);
            }

            let mut total_loss = 0.0f32;
            let mut batch_count = 0usize;

            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let bs = batch_end - batch_start;

                let heads: Vec<u32> = (batch_start..batch_end)
                    .map(|i| train_triples[indices[i]].0 as u32)
                    .collect();
                let rels: Vec<u32> = (batch_start..batch_end)
                    .map(|i| train_triples[indices[i]].1 as u32)
                    .collect();
                let tails: Vec<u32> = (batch_start..batch_end)
                    .map(|i| train_triples[indices[i]].2 as u32)
                    .collect();

                let h_t = Tensor::from_vec(heads.clone(), (bs,), &self.device)?;
                let r_t = Tensor::from_vec(rels.clone(), (bs,), &self.device)?;
                let t_t = Tensor::from_vec(tails, (bs,), &self.device)?;

                let rel_ref = if self.num_relations > 0 {
                    Some(&r_t)
                } else {
                    None
                };

                // Positive loss
                let pos_loss = self.positive_loss(&h_t, &t_t, rel_ref)?;

                // Negative samples (corrupt tail)
                let mut neg_loss_total =
                    Tensor::zeros((), candle_core::DType::F32, &self.device)?;
                for _ in 0..negative_samples {
                    let neg_tails: Vec<u32> = (0..bs)
                        .map(|_| (lcg(&mut rng) % self.num_entities) as u32)
                        .collect();
                    let nt_t = Tensor::from_vec(neg_tails, (bs,), &self.device)?;
                    let nl = self.negative_loss(&h_t, &nt_t, rel_ref, margin)?;
                    neg_loss_total = neg_loss_total.add(&nl)?;
                }

                let loss = pos_loss.add(&neg_loss_total)?;
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

/// Softplus: `(1/beta) * ln(1 + exp(beta * x))`, numerically stable.
///
/// For large beta, this approaches `max(0, x)` (sharp ReLU approximation).
/// Matches `crate::utils::softplus`.
fn softplus(x: &Tensor, beta: f32) -> Result<Tensor> {
    // Scale: beta * x
    let scaled = x.affine(beta as f64, 0.0)?;
    // Clamp for numerical stability, then log1p(exp(scaled)) / beta
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
    fn test_positive_loss_computes() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 0, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let tails = Tensor::from_vec(vec![3u32, 4, 5], (3,), &device).unwrap();
        let loss = trainer.positive_loss(&heads, &tails, None).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val >= 0.0, "loss should be non-negative, got {val}");
        assert!(val.is_finite(), "loss should be finite");
    }

    #[test]
    fn test_positive_loss_with_relations() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 3, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let tails = Tensor::from_vec(vec![3u32, 4, 5], (3,), &device).unwrap();
        let rels = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let loss = trainer
            .positive_loss(&heads, &tails, Some(&rels))
            .unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val >= 0.0, "loss should be non-negative, got {val}");
        assert!(val.is_finite(), "loss should be finite");
    }

    #[test]
    fn test_backward_computes_gradients() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 3, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1], (2,), &device).unwrap();
        let tails = Tensor::from_vec(vec![2u32, 3], (2,), &device).unwrap();
        let rels = Tensor::from_vec(vec![0u32, 1], (2,), &device).unwrap();
        let loss = trainer
            .positive_loss(&heads, &tails, Some(&rels))
            .unwrap();

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
        // Simple triples: entity 0 --rel0--> entity 1, entity 2 --rel1--> entity 3
        let triples = vec![(0, 0, 1), (2, 1, 3), (4, 0, 5), (6, 1, 7)];
        let losses = trainer.fit(&triples, 200, 0.05, 4, 0.1, 2).unwrap();
        // Compare first 10 avg vs last 10 avg for robustness
        let first_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let last_avg: f32 = losses[losses.len() - 10..].iter().sum::<f32>() / 10.0;
        assert!(
            last_avg < first_avg,
            "loss should decrease: first_10_avg={first_avg}, last_10_avg={last_avg}",
        );
    }
}
