//! Candle-based box embedding trainer with autograd.
//!
//! Uses candle tensors for GPU-accelerated training with automatic
//! differentiation. This replaces manual gradient computation for
//! scalability to high dimensions and large datasets.
//!
//! # Architecture
//!
//! Entity embeddings are stored as two `Var` tensors:
//! - `mu`: center positions, shape `[num_entities, dim]`
//! - `log_delta`: log-widths, shape `[num_entities, dim]` (delta = exp(log_delta))
//!
//! The loss for a positive triple (h, r, t) is the containment loss:
//! `||relu(|mu_h - mu_t| + delta_h - delta_t)||^2`
//!
//! Gumbel softplus is applied to convert log_delta to actual widths:
//! `delta = beta * softplus((exp(log_delta)) / beta)`

use candle_core::{Device, Result, Tensor, Var};

/// Candle-based box embedding trainer.
pub struct CandleBoxTrainer {
    /// Entity center positions: `[num_entities, dim]`.
    pub mu: Var,
    /// Entity log-widths: `[num_entities, dim]`.
    pub log_delta: Var,
    /// Embedding dimension.
    pub dim: usize,
    /// Number of entities.
    pub num_entities: usize,
    /// Gumbel beta parameter.
    pub beta: f32,
    /// Device (CPU or CUDA).
    pub device: Device,
}

impl CandleBoxTrainer {
    /// Create a new trainer with random initialization.
    pub fn new(num_entities: usize, dim: usize, beta: f32, device: &Device) -> Result<Self> {
        // Xavier-like initialization
        let scale = (2.0 / dim as f64).sqrt();
        let mu = Var::from_tensor(&Tensor::randn(
            0.0_f32,
            scale as f32,
            (num_entities, dim),
            device,
        )?)?;
        // Initialize log_delta to small positive values (narrow boxes)
        let log_delta =
            Var::from_tensor(&Tensor::randn(-1.0_f32, 0.1, (num_entities, dim), device)?)?;

        Ok(Self {
            mu,
            log_delta,
            dim,
            num_entities,
            beta,
            device: device.clone(),
        })
    }

    /// Compute containment loss for a batch of positive triples.
    ///
    /// For each (head, tail) pair, loss = ||relu(|mu_h - mu_t| + delta_h - delta_t)||^2
    /// where delta = softplus(exp(log_delta)) * beta.
    pub fn containment_loss(
        &self,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        margin: f32,
    ) -> Result<Tensor> {
        let mu_h = self.mu.as_tensor().index_select(head_ids, 0)?;
        let mu_t = self.mu.as_tensor().index_select(tail_ids, 0)?;

        let delta_h = self.get_delta()?.index_select(head_ids, 0)?;
        let delta_t = self.get_delta()?.index_select(tail_ids, 0)?;

        // |mu_h - mu_t| + delta_h - delta_t - margin
        let diff = mu_h.sub(&mu_t)?.abs()?;
        let violation = diff.add(&delta_h)?.sub(&delta_t)?;
        let margin_t = Tensor::full(margin, violation.shape(), &self.device)?;
        let shifted = violation.sub(&margin_t)?;

        // relu then L2 norm squared per triple
        let relu = shifted.relu()?;
        let sq = relu.sqr()?;
        let per_triple = sq.sum(1)?; // sum over dims

        per_triple.mean(0) // mean over batch
    }

    /// Compute negative containment loss (push negatives apart).
    ///
    /// For negative (head, tail) pairs, loss = relu(margin - violation)
    pub fn negative_loss(
        &self,
        head_ids: &Tensor,
        tail_ids: &Tensor,
        margin: f32,
    ) -> Result<Tensor> {
        let mu_h = self.mu.as_tensor().index_select(head_ids, 0)?;
        let mu_t = self.mu.as_tensor().index_select(tail_ids, 0)?;

        let delta_h = self.get_delta()?.index_select(head_ids, 0)?;
        let delta_t = self.get_delta()?.index_select(tail_ids, 0)?;

        let diff = mu_h.sub(&mu_t)?.abs()?;
        let violation = diff.add(&delta_h)?.sub(&delta_t)?;

        // Negative: we WANT violation to be positive (boxes not contained)
        // Loss = relu(margin - ||violation||) -- penalize if negatives look contained
        let norm_sq = violation.sqr()?.sum(1)?;
        let norm = norm_sq.sqrt()?;
        let margin_t = Tensor::full(margin, norm.shape(), &self.device)?;
        let neg_loss = margin_t.sub(&norm)?.relu()?;

        neg_loss.mean(0)
    }

    /// Get widths (delta) from log_delta via softplus.
    fn get_delta(&self) -> Result<Tensor> {
        // delta = softplus(exp(log_delta)) for numerical stability
        let exp_ld = self.log_delta.as_tensor().exp()?;
        softplus(&exp_ld, self.beta)
    }

    /// Score a batch of triples: lower = more contained.
    pub fn score(&self, head_ids: &Tensor, tail_ids: &Tensor) -> Result<Tensor> {
        let mu_h = self.mu.as_tensor().index_select(head_ids, 0)?;
        let mu_t = self.mu.as_tensor().index_select(tail_ids, 0)?;
        let delta_h = self.get_delta()?.index_select(head_ids, 0)?;
        let delta_t = self.get_delta()?.index_select(tail_ids, 0)?;

        let diff = mu_h.sub(&mu_t)?.abs()?;
        let violation = diff.add(&delta_h)?.sub(&delta_t)?.relu()?;
        violation.sqr()?.sum(1)?.sqrt()
    }

    /// Train on triples with AdamW optimizer.
    ///
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

        let params = ParamsAdamW {
            lr,
            weight_decay: 1e-4,
            ..Default::default()
        };
        let mut opt = AdamW::new(vec![self.mu.clone(), self.log_delta.clone()], params)?;
        let n = train_triples.len();
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut rng: u64 = 42;

        let lcg = |s: &mut u64| -> usize {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as usize
        };

        for epoch in 0..epochs {
            let mut total_loss = 0.0f32;
            let mut batch_count = 0usize;

            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let batch = &train_triples[batch_start..batch_end];
                let bs = batch.len();

                let heads: Vec<u32> = batch.iter().map(|&(h, _, _)| h as u32).collect();
                let tails: Vec<u32> = batch.iter().map(|&(_, _, t)| t as u32).collect();

                let h_t = Tensor::from_vec(heads, (bs,), &self.device)?;
                let t_t = Tensor::from_vec(tails, (bs,), &self.device)?;

                // Positive loss
                let pos_loss = self.containment_loss(&h_t, &t_t, 0.0)?;

                // Negative samples
                let mut neg_loss_total = Tensor::zeros((), candle_core::DType::F32, &self.device)?;
                for _ in 0..negative_samples {
                    let neg_tails: Vec<u32> = (0..bs)
                        .map(|_| (lcg(&mut rng) % self.num_entities) as u32)
                        .collect();
                    let nt_t = Tensor::from_vec(neg_tails, (bs,), &self.device)?;
                    let nl = self.negative_loss(&h_t, &nt_t, margin)?;
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

/// Softplus: log(1 + exp(x/beta)) * beta, numerically stable.
fn softplus(x: &Tensor, beta: f32) -> Result<Tensor> {
    let scaled = x.affine(1.0 / beta as f64, 0.0)?;
    // For large x: softplus(x) ≈ x. For small x: softplus(x) = log(1 + exp(x)).
    // candle doesn't have a built-in softplus, so we use log1p(exp(x)) with clamping.
    let exp_scaled = scaled.clamp(-20.0, 20.0)?.exp()?;
    let one = Tensor::ones_like(&exp_scaled)?;
    let log1p = one.add(&exp_scaled)?.log()?;
    log1p.affine(beta as f64, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_trainer_creates() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(100, 32, 10.0, &device).unwrap();
        assert_eq!(trainer.num_entities, 100);
        assert_eq!(trainer.dim, 32);
    }

    #[test]
    fn test_containment_loss_computes() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let tails = Tensor::from_vec(vec![3u32, 4, 5], (3,), &device).unwrap();
        let loss = trainer.containment_loss(&heads, &tails, 0.0).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val >= 0.0, "loss should be non-negative, got {val}");
        assert!(val.is_finite(), "loss should be finite");
    }

    #[test]
    fn test_backward_computes_gradients() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1], (2,), &device).unwrap();
        let tails = Tensor::from_vec(vec![2u32, 3], (2,), &device).unwrap();
        let loss = trainer.containment_loss(&heads, &tails, 0.1).unwrap();

        // backward should work without panicking
        let grads = loss.backward().unwrap();

        let mu_grad = grads.get(trainer.mu.as_tensor()).unwrap();
        assert_eq!(mu_grad.dims(), &[10, 8]);
    }

    #[test]
    fn test_score_shape() {
        let device = Device::Cpu;
        let trainer = CandleBoxTrainer::new(10, 8, 10.0, &device).unwrap();
        let heads = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device).unwrap();
        let tails = Tensor::from_vec(vec![3u32, 4, 5], (3,), &device).unwrap();
        let scores = trainer.score(&heads, &tails).unwrap();
        assert_eq!(scores.dims(), &[3]);
    }
}
