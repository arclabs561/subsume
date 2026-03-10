//! TaxoBell MLP encoder with candle autograd.
//!
//! Maps pre-computed text embeddings to diagonal Gaussian boxes via two
//! small MLPs (one for center, one for offset/sigma). Training uses
//! candle's automatic differentiation for exact gradients.
//!
//! Architecture per MLP:
//! ```text
//! input (embed_dim) → Linear → ReLU → Linear → output (box_dim)
//! ```
//!
//! The offset MLP output is passed through softplus to ensure positive sigma.
//!
//! # References
//!
//! - TaxoBell (WWW 2026, arXiv:2601.09633), Section 4.1: Gaussian box encoder

use crate::taxobell::{CombinedLossResult, TaxoBellConfig};
use crate::BoxError;
use candle_core::{DType, Device, Result as CResult, Tensor, Var, D};

// ---------------------------------------------------------------------------
// Numerically stable softplus
// ---------------------------------------------------------------------------

/// Softplus: ln(1 + exp(x)), numerically stable.
fn softplus(x: &Tensor) -> CResult<Tensor> {
    // softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
    let zero = x.zeros_like()?;
    let relu_x = x.maximum(&zero)?;
    let neg_abs = x.abs()?.neg()?;
    let log_term = neg_abs
        .exp()?
        .broadcast_add(&Tensor::new(&[1.0f32], x.device())?.broadcast_as(x.shape())?)?;
    relu_x.add(&log_term.log()?)
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// A two-layer MLP with learnable `Var` parameters for autograd.
///
/// `Linear(in, hidden) → ReLU → Linear(hidden, out)`.
/// Weights are Xavier-uniform initialized.
#[derive(Debug)]
pub struct Mlp {
    w1: Var,
    b1: Var,
    w2: Var,
    b2: Var,
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
}

impl Mlp {
    /// Create a new MLP with Xavier-uniform initialization.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        device: &Device,
    ) -> Result<Self, BoxError> {
        let map_err = |e: candle_core::Error| BoxError::Internal(e.to_string());

        let scale1 = (6.0 / (input_dim + hidden_dim) as f64).sqrt() as f32;
        let w1 = Var::rand(-scale1, scale1, (hidden_dim, input_dim), device).map_err(map_err)?;
        let b1 = Var::zeros(hidden_dim, DType::F32, device).map_err(map_err)?;

        let scale2 = (6.0 / (hidden_dim + output_dim) as f64).sqrt() as f32;
        let w2 = Var::rand(-scale2, scale2, (output_dim, hidden_dim), device).map_err(map_err)?;
        let b2 = Var::zeros(output_dim, DType::F32, device).map_err(map_err)?;

        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            output_dim,
        })
    }

    /// Forward pass: `Linear → ReLU → Linear`.
    ///
    /// `x`: `[batch, input_dim]` or `[input_dim]`.
    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        // h = relu(x @ W1^T + b1)
        let h = x
            .matmul(&self.w1.as_tensor().t()?)?
            .broadcast_add(self.b1.as_tensor())?
            .relu()?;
        // out = h @ W2^T + b2
        h.matmul(&self.w2.as_tensor().t()?)?
            .broadcast_add(self.b2.as_tensor())
    }

    /// Collect all learnable `Var` parameters.
    pub fn vars(&self) -> Vec<&Var> {
        vec![&self.w1, &self.b1, &self.w2, &self.b2]
    }

    /// Total number of learnable parameters.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.hidden_dim * self.input_dim
            + self.hidden_dim
            + self.output_dim * self.hidden_dim
            + self.output_dim
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// TaxoBell encoder: two MLPs mapping text embeddings to Gaussian box parameters.
///
/// - `center_mlp`: embedding → mu (center of Gaussian)
/// - `offset_mlp`: embedding → raw offset → softplus → sigma (std dev)
#[derive(Debug)]
pub struct TaxoBellEncoder {
    /// MLP producing center (mu).
    pub center_mlp: Mlp,
    /// MLP producing raw offset (pre-softplus sigma).
    pub offset_mlp: Mlp,
    /// Embedding input dimension.
    pub embed_dim: usize,
    /// Box output dimension.
    pub box_dim: usize,
    /// Device (CPU or GPU).
    pub device: Device,
}

impl TaxoBellEncoder {
    /// Create a new encoder with Xavier-initialized MLPs.
    pub fn new(
        embed_dim: usize,
        hidden_dim: usize,
        box_dim: usize,
        device: &Device,
    ) -> Result<Self, BoxError> {
        let center_mlp = Mlp::new(embed_dim, hidden_dim, box_dim, device)?;
        let offset_mlp = Mlp::new(embed_dim, hidden_dim, box_dim, device)?;
        Ok(Self {
            center_mlp,
            offset_mlp,
            embed_dim,
            box_dim,
            device: device.clone(),
        })
    }

    /// Encode a batch of embeddings into (mu, sigma) tensors.
    ///
    /// - Input: `[batch, embed_dim]`
    /// - Returns: `(mu, sigma)` each `[batch, box_dim]` where `sigma > 0`.
    pub fn encode(&self, embeddings: &Tensor) -> CResult<(Tensor, Tensor)> {
        let mu = self.center_mlp.forward(embeddings)?;
        let raw_offset = self.offset_mlp.forward(embeddings)?;
        let sigma = softplus(&raw_offset)?;
        Ok((mu, sigma))
    }

    /// Encode a single embedding into a `GaussianBox` (for evaluation).
    pub fn encode_one(&self, embedding: &[f32]) -> Result<crate::gaussian::GaussianBox, BoxError> {
        let map_err = |e: candle_core::Error| BoxError::Internal(e.to_string());
        let t = Tensor::new(embedding, &self.device).map_err(map_err)?;
        let t = t.unsqueeze(0).map_err(map_err)?; // [1, embed_dim]
        let (mu, sigma) = self.encode(&t).map_err(map_err)?;
        let mu_vec: Vec<f32> = mu.squeeze(0).map_err(map_err)?.to_vec1().map_err(map_err)?;
        let sigma_vec: Vec<f32> = sigma
            .squeeze(0)
            .map_err(map_err)?
            .to_vec1()
            .map_err(map_err)?;
        crate::gaussian::GaussianBox::new(mu_vec, sigma_vec)
    }

    /// Collect all learnable `Var` parameters from both MLPs.
    pub fn vars(&self) -> Vec<&Var> {
        let mut v = self.center_mlp.vars();
        v.extend(self.offset_mlp.vars());
        v
    }

    /// Total number of learnable parameters.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.center_mlp.num_params() + self.offset_mlp.num_params()
    }
}

// ---------------------------------------------------------------------------
// Tensor-based loss functions (differentiable)
// ---------------------------------------------------------------------------

/// Bhattacharyya coefficient between batches of diagonal Gaussians.
///
/// Inputs: `[batch, dim]` tensors. Returns: `[batch]` BC values in `[0, 1]`.
fn bhattacharyya_coeff_batch(
    mu1: &Tensor,
    s1: &Tensor,
    mu2: &Tensor,
    s2: &Tensor,
) -> CResult<Tensor> {
    let v1 = s1.sqr()?; // [batch, dim]
    let v2 = s2.sqr()?;
    // sigma_avg = (v1 + v2) / 2
    let sigma_avg = v1.add(&v2)?.affine(0.5, 0.0)?;
    let mu_diff = mu1.sub(mu2)?;

    // BD = 0.25 * sum((mu_diff^2 / sigma_avg), dim=-1)
    //    + 0.5 * sum(ln(sigma_avg), dim=-1)
    //    - 0.25 * sum(ln(v1), dim=-1)
    //    - 0.25 * sum(ln(v2), dim=-1)
    let t1 = mu_diff
        .sqr()?
        .div(&sigma_avg)?
        .sum(D::Minus1)?
        .affine(0.25, 0.0)?;
    let t2 = sigma_avg.log()?.sum(D::Minus1)?.affine(0.5, 0.0)?;
    let t3 = v1.log()?.sum(D::Minus1)?.affine(0.25, 0.0)?;
    let t4 = v2.log()?.sum(D::Minus1)?.affine(0.25, 0.0)?;

    let bd = t1.add(&t2)?.sub(&t3)?.sub(&t4)?;
    bd.neg()?.exp() // BC = exp(-BD)
}

/// KL divergence KL(q || p) between batches of diagonal Gaussians.
///
/// Inputs: `[batch, dim]` tensors. Returns: `[batch]` KL values >= 0.
fn kl_divergence_batch(
    mu_q: &Tensor,
    s_q: &Tensor,
    mu_p: &Tensor,
    s_p: &Tensor,
) -> CResult<Tensor> {
    let vq = s_q.sqr()?;
    let vp = s_p.sqr()?;
    let mu_diff = mu_p.sub(mu_q)?;

    // KL = 0.5 * sum(vq/vp + (mu_p - mu_q)^2/vp - 1 + ln(vp/vq), dim=-1)
    let ratio = vq.div(&vp)?;
    let mu_term = mu_diff.sqr()?.div(&vp)?;
    let log_term = vp.div(&vq)?.log()?;

    // per_dim = ratio + mu_term + log_term - 1
    let per_dim = ratio.add(&mu_term)?.add(&log_term)?.affine(1.0, -1.0)?;
    per_dim.sum(D::Minus1)?.affine(0.5, 0.0)
}

/// Log-volume of diagonal Gaussians: sum(ln(sigma), dim=-1).
///
/// Input: `sigma [batch, dim]`. Returns: `[batch]`.
fn log_volume_batch(sigma: &Tensor) -> CResult<Tensor> {
    sigma.log()?.sum(D::Minus1)
}

/// Volume regularization (Eq. 13): per-dim squared hinge floor on variance.
///
/// `L_reg = (1/d) * sum(max(0, min_var - sigma^2)^2, dim=-1)`
fn volume_reg_batch(sigma: &Tensor, min_var: f32) -> CResult<Tensor> {
    let var = sigma.sqr()?;
    let zero = var.zeros_like()?;
    // gap = min_var - var
    let gap = var.affine(-1.0, min_var as f64)?;
    let hinge = gap.maximum(&zero)?;
    let d = sigma.dim(D::Minus1)? as f64;
    hinge.sqr()?.sum(D::Minus1)?.affine(1.0 / d, 0.0)
}

/// Sigma ceiling (Eq. 14): per-dim linear hinge ceiling on variance.
///
/// `L_clip = (1/d) * sum(max(0, sigma^2 - max_var), dim=-1)`
fn sigma_ceiling_batch(sigma: &Tensor, max_var: f32) -> CResult<Tensor> {
    let var = sigma.sqr()?;
    let zero = var.zeros_like()?;
    // gap = var - max_var
    let gap = var.affine(1.0, -(max_var as f64))?;
    let hinge = gap.maximum(&zero)?;
    let d = sigma.dim(D::Minus1)? as f64;
    hinge.sum(D::Minus1)?.affine(1.0 / d, 0.0)
}

// ---------------------------------------------------------------------------
// Combined loss (returns Tensor for backward + CombinedLossResult for logging)
// ---------------------------------------------------------------------------

/// Compute the four-component TaxoBell loss on tensor batches.
///
/// Returns `(total_loss_tensor, breakdown)` where `total_loss_tensor` supports
/// `.backward()` for autograd.
#[allow(clippy::too_many_arguments)]
fn combined_loss_tensor(
    mu_child: &Tensor,
    s_child: &Tensor,
    mu_parent: &Tensor,
    s_parent: &Tensor,
    mu_anchor: &Tensor,
    s_anchor: &Tensor,
    mu_pos: &Tensor,
    s_pos: &Tensor,
    mu_neg: &Tensor,
    s_neg: &Tensor,
    mu_all: &Tensor,
    s_all: &Tensor,
    config: &TaxoBellConfig,
) -> CResult<(Tensor, CombinedLossResult)> {
    let device = mu_child.device();
    let eps_val = 1e-7f64;
    let one_minus_eps = 1.0 - eps_val;

    // --- L_sym: symmetric BCE on Bhattacharyya coefficient ---
    let n_neg = mu_neg.dim(0)?;
    let l_sym_scalar;
    let l_sym_t;
    if n_neg > 0 {
        let bc_pos = bhattacharyya_coeff_batch(mu_anchor, s_anchor, mu_pos, s_pos)?;
        let bc_neg = bhattacharyya_coeff_batch(mu_anchor, s_anchor, mu_neg, s_neg)?;

        // Clamp for log stability
        let eps_t = Tensor::new(&[eps_val as f32], device)?.broadcast_as(bc_pos.shape())?;
        let one_me = Tensor::new(&[one_minus_eps as f32], device)?.broadcast_as(bc_pos.shape())?;
        let bc_pos_c = bc_pos.maximum(&eps_t)?.minimum(&one_me)?;
        let bc_neg_c = bc_neg.maximum(&eps_t)?.minimum(&one_me)?;

        // -log(bc_pos) - log(1 - bc_neg)
        let term1 = bc_pos_c.log()?.neg()?;
        let one_t = Tensor::new(&[1.0f32], device)?.broadcast_as(bc_neg_c.shape())?;
        let term2 = one_t.sub(&bc_neg_c)?.log()?.neg()?;
        let per_sample = term1.add(&term2)?;
        l_sym_t = per_sample.mean_all()?;
        l_sym_scalar = l_sym_t.to_vec0::<f32>()?;
    } else {
        l_sym_t = Tensor::new(0.0f32, device)?;
        l_sym_scalar = 0.0;
    }

    // --- L_asym: asymmetric KL containment ---
    let n_edges = mu_child.dim(0)?;
    let l_asym_scalar;
    let l_asym_t;
    if n_edges > 0 {
        let kl = kl_divergence_batch(mu_child, s_child, mu_parent, s_parent)?;
        // L_align = relu(kl - margin)
        let zero_edges = kl.zeros_like()?;
        let l_align = kl
            .affine(1.0, -(config.asymmetric_margin as f64))?
            .maximum(&zero_edges)?;

        let l_diverge = if config.asymmetric_diverge_c > 0.0 {
            let kl_rev = kl_divergence_batch(mu_parent, s_parent, mu_child, s_child)?;
            let lv_parent = log_volume_batch(s_parent)?;
            let lv_child = log_volume_batch(s_child)?;
            let d_rep = lv_parent.sub(&lv_child)?;
            // relu(C * d_rep - kl_rev)
            let zero_d = d_rep.zeros_like()?;
            d_rep
                .affine(config.asymmetric_diverge_c as f64, 0.0)?
                .sub(&kl_rev)?
                .maximum(&zero_d)?
        } else {
            kl.zeros_like()?
        };

        // L_asym = mean(L_align + lambda * L_diverge)
        let asym_per = l_align.add(&l_diverge.affine(config.diverge_lambda as f64, 0.0)?)?;
        l_asym_t = asym_per.mean_all()?;
        l_asym_scalar = l_asym_t.to_vec0::<f32>()?;
    } else {
        l_asym_t = Tensor::new(0.0f32, device)?;
        l_asym_scalar = 0.0;
    }

    // --- L_reg + L_clip on all boxes ---
    let n_all = mu_all.dim(0)?;
    let l_reg_scalar;
    let l_reg_t;
    let l_clip_scalar;
    let l_clip_t;
    if n_all > 0 {
        l_reg_t = volume_reg_batch(s_all, config.min_var)?.mean_all()?;
        l_clip_t = sigma_ceiling_batch(s_all, config.max_var)?.mean_all()?;
        l_reg_scalar = l_reg_t.to_vec0::<f32>()?;
        l_clip_scalar = l_clip_t.to_vec0::<f32>()?;
    } else {
        l_reg_t = Tensor::new(0.0f32, device)?;
        l_clip_t = Tensor::new(0.0f32, device)?;
        l_reg_scalar = 0.0;
        l_clip_scalar = 0.0;
    }

    // --- Total: weighted sum ---
    let total = l_sym_t
        .affine(config.alpha as f64, 0.0)?
        .add(&l_asym_t.affine(config.beta as f64, 0.0)?)?
        .add(&l_reg_t.affine(config.gamma as f64, 0.0)?)?
        .add(&l_clip_t.affine(config.delta as f64, 0.0)?)?;

    let total_scalar = total.to_vec0::<f32>()?;

    let result = CombinedLossResult {
        total: total_scalar,
        l_sym: l_sym_scalar,
        l_asym: l_asym_scalar,
        l_reg: l_reg_scalar,
        l_clip: l_clip_scalar,
    };

    Ok((total, result))
}

// ---------------------------------------------------------------------------
// AMSGrad optimizer state
// ---------------------------------------------------------------------------

struct AmsGrad {
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    v_hat: Vec<Tensor>,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,
}

impl AmsGrad {
    fn new(vars: &[&Var], beta1: f64, beta2: f64, eps: f64) -> CResult<Self> {
        let mut m = Vec::with_capacity(vars.len());
        let mut v = Vec::with_capacity(vars.len());
        let mut v_hat = Vec::with_capacity(vars.len());
        for var in vars {
            let z = var.as_tensor().zeros_like()?;
            m.push(z.clone());
            v.push(z.clone());
            v_hat.push(z);
        }
        Ok(Self {
            m,
            v,
            v_hat,
            beta1,
            beta2,
            eps,
            t: 0,
        })
    }

    fn step(
        &mut self,
        vars: &[&Var],
        grads: &candle_core::backprop::GradStore,
        lr: f64,
    ) -> CResult<()> {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);

        for (i, var) in vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                // m = beta1 * m + (1 - beta1) * g
                self.m[i] = self.m[i]
                    .affine(self.beta1, 0.0)?
                    .add(&grad.affine(1.0 - self.beta1, 0.0)?)?;
                // v = beta2 * v + (1 - beta2) * g^2
                let v_new = self.v[i]
                    .affine(self.beta2, 0.0)?
                    .add(&grad.sqr()?.affine(1.0 - self.beta2, 0.0)?)?;
                self.v[i] = v_new.clone();
                // v_hat = max(v_hat, v) [AMSGrad]
                self.v_hat[i] = self.v_hat[i].maximum(&v_new)?;
                // m_hat = m / (1 - beta1^t)
                let m_hat = self.m[i].affine(1.0 / bc1, 0.0)?;
                // update = lr * m_hat / (sqrt(v_hat) + eps)
                let denom = self.v_hat[i].sqrt()?.affine(1.0, self.eps)?;
                let update = m_hat.affine(lr, 0.0)?.div(&denom)?;
                // param -= update
                let new_val = var.as_tensor().sub(&update)?;
                var.set(&new_val)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Config, result types (plain Rust, no candle dependency)
// ---------------------------------------------------------------------------

/// Configuration for TaxoBell training.
#[derive(Debug, Clone)]
pub struct TaxoBellTrainingConfig {
    /// Learning rate for AMSGrad.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Number of negative samples per positive edge.
    pub num_negatives: usize,
    /// Loss function configuration.
    pub loss_config: TaxoBellConfig,
    /// Hidden dimension for the encoder MLPs.
    pub hidden_dim: usize,
    /// Output box dimension.
    pub box_dim: usize,
    /// RNG seed for negative sampling.
    pub seed: u64,
    /// Number of warmup epochs (linear LR warmup).
    pub warmup_epochs: usize,
}

impl Default for TaxoBellTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 100,
            num_negatives: 3,
            loss_config: TaxoBellConfig::default(),
            hidden_dim: 64,
            box_dim: 16,
            seed: 42,
            warmup_epochs: 5,
        }
    }
}

/// Result of evaluating a trained TaxoBell model on a set of edges.
#[derive(Debug, Clone)]
pub struct TaxoBellEvalResult {
    /// Mean Reciprocal Rank: average of 1/rank for each test edge.
    pub mrr: f32,
    /// Hits@1: fraction of test edges ranked first.
    pub hits_at_1: f32,
    /// Hits@3: fraction of test edges ranked in top 3.
    pub hits_at_3: f32,
    /// Hits@10: fraction of test edges ranked in top 10.
    pub hits_at_10: f32,
}

/// Snapshot of training state at a given epoch.
#[derive(Debug, Clone)]
pub struct TrainingSnapshot {
    /// Epoch number.
    pub epoch: usize,
    /// Combined loss breakdown.
    pub loss: CombinedLossResult,
    /// Current learning rate.
    pub lr: f32,
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Train a TaxoBell encoder on a taxonomy dataset.
///
/// Uses candle autograd for exact gradient computation. Returns the trained
/// encoder and per-epoch loss snapshots.
///
/// # Arguments
///
/// * `embeddings` - pre-computed text embeddings, one per node
/// * `edges` - training edges as `(parent_id, child_id)` pairs
/// * `all_node_ids` - list of all node IDs (for negative sampling)
/// * `node_index` - maps node ID → index into `embeddings`
/// * `config` - training configuration
pub fn train_taxobell(
    embeddings: &[Vec<f32>],
    edges: &[(usize, usize)],
    all_node_ids: &[usize],
    node_index: &std::collections::HashMap<usize, usize>,
    config: &TaxoBellTrainingConfig,
) -> Result<(TaxoBellEncoder, Vec<TrainingSnapshot>), BoxError> {
    let map_err = |e: candle_core::Error| BoxError::Internal(e.to_string());
    let device = Device::Cpu;
    let embed_dim = embeddings[0].len();

    // Load all embeddings as a single tensor [n_nodes, embed_dim].
    let flat: Vec<f32> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();
    let n_nodes = embeddings.len();
    let all_embeds = Tensor::from_vec(flat, (n_nodes, embed_dim), &device).map_err(map_err)?;

    // Build index arrays for edges.
    let child_indices: Vec<u32> = edges
        .iter()
        .map(|&(_, child_id)| node_index[&child_id] as u32)
        .collect();
    let parent_indices: Vec<u32> = edges
        .iter()
        .map(|&(parent_id, _)| node_index[&parent_id] as u32)
        .collect();
    let all_indices: Vec<u32> = (0..n_nodes as u32).collect();

    let child_idx_t =
        Tensor::from_vec(child_indices.clone(), edges.len(), &device).map_err(map_err)?;
    let parent_idx_t =
        Tensor::from_vec(parent_indices.clone(), edges.len(), &device).map_err(map_err)?;
    let all_idx_t = Tensor::from_vec(all_indices, n_nodes, &device).map_err(map_err)?;

    // Create encoder.
    let encoder = TaxoBellEncoder::new(embed_dim, config.hidden_dim, config.box_dim, &device)?;

    // AMSGrad optimizer.
    let vars = encoder.vars();
    let mut opt = AmsGrad::new(&vars, 0.9, 0.999, 1e-8).map_err(map_err)?;
    let mut snapshots = Vec::with_capacity(config.epochs);

    // Xorshift RNG for negative sampling.
    let mut rng_state = config.seed.wrapping_add(1);
    let mut rng_next = |bound: usize| -> usize {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as usize) % bound
    };

    let n_edges = edges.len();
    let n_neg = config.num_negatives;
    let n_total_neg = n_edges * n_neg;

    for epoch in 0..config.epochs {
        // LR schedule: linear warmup + cosine decay.
        let lr = crate::optimizer::get_learning_rate(
            epoch,
            config.epochs,
            config.learning_rate,
            config.warmup_epochs,
        );

        // Sample negatives for this epoch.
        let neg_node_indices: Vec<u32> = (0..n_total_neg)
            .map(|_| {
                let node_id = all_node_ids[rng_next(all_node_ids.len())];
                node_index[&node_id] as u32
            })
            .collect();
        // Anchor/positive indices: repeat each edge's child/parent n_neg times.
        let anchor_indices: Vec<u32> = child_indices
            .iter()
            .flat_map(|&idx| std::iter::repeat(idx).take(n_neg))
            .collect();
        let pos_indices: Vec<u32> = parent_indices
            .iter()
            .flat_map(|&idx| std::iter::repeat(idx).take(n_neg))
            .collect();

        // --- Forward pass: encode all nodes once ---
        let (mu_all, s_all) = encoder.encode(&all_embeds).map_err(map_err)?;

        // Gather child, parent, anchor, positive, negative.
        let mu_child = mu_all.index_select(&child_idx_t, 0).map_err(map_err)?;
        let s_child = s_all.index_select(&child_idx_t, 0).map_err(map_err)?;
        let mu_parent = mu_all.index_select(&parent_idx_t, 0).map_err(map_err)?;
        let s_parent = s_all.index_select(&parent_idx_t, 0).map_err(map_err)?;

        let (mu_anchor, s_anchor, mu_pos, s_pos, mu_neg_t, s_neg_t) = if n_total_neg > 0 {
            let anchor_t =
                Tensor::from_vec(anchor_indices, n_total_neg, &device).map_err(map_err)?;
            let pos_t = Tensor::from_vec(pos_indices, n_total_neg, &device).map_err(map_err)?;
            let neg_t =
                Tensor::from_vec(neg_node_indices, n_total_neg, &device).map_err(map_err)?;

            (
                mu_all.index_select(&anchor_t, 0).map_err(map_err)?,
                s_all.index_select(&anchor_t, 0).map_err(map_err)?,
                mu_all.index_select(&pos_t, 0).map_err(map_err)?,
                s_all.index_select(&pos_t, 0).map_err(map_err)?,
                mu_all.index_select(&neg_t, 0).map_err(map_err)?,
                s_all.index_select(&neg_t, 0).map_err(map_err)?,
            )
        } else {
            let empty = Tensor::zeros((0, config.box_dim), DType::F32, &device).map_err(map_err)?;
            (
                empty.clone(),
                empty.clone(),
                empty.clone(),
                empty.clone(),
                empty.clone(),
                empty,
            )
        };

        // Regularization uses all nodes.
        let mu_all_reg = mu_all.index_select(&all_idx_t, 0).map_err(map_err)?;
        let s_all_reg = s_all.index_select(&all_idx_t, 0).map_err(map_err)?;

        // --- Compute loss ---
        let (loss_t, loss_result) = combined_loss_tensor(
            &mu_child,
            &s_child,
            &mu_parent,
            &s_parent,
            &mu_anchor,
            &s_anchor,
            &mu_pos,
            &s_pos,
            &mu_neg_t,
            &s_neg_t,
            &mu_all_reg,
            &s_all_reg,
            &config.loss_config,
        )
        .map_err(map_err)?;

        snapshots.push(TrainingSnapshot {
            epoch,
            loss: loss_result,
            lr,
        });

        // --- Backward + AMSGrad step ---
        let grads = loss_t.backward().map_err(map_err)?;
        let vars = encoder.vars();
        opt.step(&vars, &grads, lr as f64).map_err(map_err)?;
    }

    Ok((encoder, snapshots))
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluate a trained encoder on a set of edges.
///
/// For each `(parent_id, child_id)`, ranks the true parent among all candidates
/// by KL divergence (lower KL = better containment = higher rank).
pub fn evaluate_taxobell(
    encoder: &TaxoBellEncoder,
    embeddings: &[Vec<f32>],
    test_edges: &[(usize, usize)],
    all_node_ids: &[usize],
    node_index: &std::collections::HashMap<usize, usize>,
) -> Result<TaxoBellEvalResult, BoxError> {
    if test_edges.is_empty() {
        return Ok(TaxoBellEvalResult {
            mrr: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
        });
    }

    // Encode all nodes (no grad needed).
    let boxes: Vec<crate::gaussian::GaussianBox> = all_node_ids
        .iter()
        .map(|&id| {
            let idx = node_index[&id];
            encoder.encode_one(&embeddings[idx])
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut reciprocal_ranks = Vec::with_capacity(test_edges.len());
    let mut hits1 = 0usize;
    let mut hits3 = 0usize;
    let mut hits10 = 0usize;

    for &(parent_id, child_id) in test_edges {
        let child_idx = node_index[&child_id];
        let child_box = encoder.encode_one(&embeddings[child_idx])?;

        let mut scores: Vec<(usize, f32)> = all_node_ids
            .iter()
            .enumerate()
            .map(|(pos, &cand_id)| {
                let kl =
                    crate::gaussian::kl_divergence(&child_box, &boxes[pos]).unwrap_or(f32::MAX);
                (cand_id, kl)
            })
            .collect();

        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let rank = scores
            .iter()
            .position(|&(id, _)| id == parent_id)
            .map(|r| r + 1)
            .unwrap_or(scores.len());

        reciprocal_ranks.push(1.0 / rank as f32);
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

    let n = test_edges.len() as f32;
    Ok(TaxoBellEvalResult {
        mrr: reciprocal_ranks.iter().sum::<f32>() / n,
        hits_at_1: hits1 as f32 / n,
        hits_at_3: hits3 as f32 / n,
        hits_at_10: hits10 as f32 / n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device {
        Device::Cpu
    }

    #[test]
    fn mlp_forward_shape() {
        let mlp = Mlp::new(8, 16, 4, &device()).unwrap();
        let input = Tensor::ones((3, 8), DType::F32, &device()).unwrap();
        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.dims(), &[3, 4]);
    }

    #[test]
    fn encoder_produces_positive_sigma() {
        let enc = TaxoBellEncoder::new(8, 16, 4, &device()).unwrap();
        let embed = Tensor::ones((2, 8), DType::F32, &device()).unwrap();
        let (_mu, sigma) = enc.encode(&embed).unwrap();
        let vals: Vec<Vec<f32>> = sigma.to_vec2().unwrap();
        for row in &vals {
            for &s in row {
                assert!(s > 0.0, "sigma must be positive, got {s}");
            }
        }
    }

    #[test]
    fn encoder_different_inputs_different_outputs() {
        let enc = TaxoBellEncoder::new(8, 16, 4, &device()).unwrap();
        let a = Tensor::zeros((1, 8), DType::F32, &device()).unwrap();
        let b = Tensor::ones((1, 8), DType::F32, &device()).unwrap();
        let (mu_a, _) = enc.encode(&a).unwrap();
        let (mu_b, _) = enc.encode(&b).unwrap();
        let va: Vec<f32> = mu_a.squeeze(0).unwrap().to_vec1().unwrap();
        let vb: Vec<f32> = mu_b.squeeze(0).unwrap().to_vec1().unwrap();
        assert_ne!(va, vb, "different inputs should produce different centers");
    }

    #[test]
    fn backward_produces_gradients() {
        let enc = TaxoBellEncoder::new(4, 8, 2, &device()).unwrap();
        let embed = Tensor::ones((2, 4), DType::F32, &device()).unwrap();
        let (mu, sigma) = enc.encode(&embed).unwrap();
        // Simple loss: sum of mu + sum of sigma
        let loss = mu
            .sum_all()
            .unwrap()
            .add(&sigma.sum_all().unwrap())
            .unwrap();
        let grads = loss.backward().unwrap();
        // Every Var should have a gradient.
        for var in enc.vars() {
            assert!(
                grads.get(var.as_tensor()).is_some(),
                "missing gradient for var"
            );
        }
    }

    #[test]
    fn train_loss_decreases() {
        let node_ids = vec![0usize, 1, 2];
        let edges = vec![(0, 1), (0, 2)];
        let embeddings = vec![
            vec![0.0, 0.5, 1.0, 0.2],
            vec![1.0, 0.0, 0.5, 0.8],
            vec![0.5, 1.0, 0.0, 0.3],
        ];
        let node_index: std::collections::HashMap<usize, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let config = TaxoBellTrainingConfig {
            learning_rate: 5e-3,
            epochs: 30,
            num_negatives: 1,
            hidden_dim: 8,
            box_dim: 4,
            seed: 42,
            warmup_epochs: 3,
            ..Default::default()
        };

        let (_, snapshots) =
            train_taxobell(&embeddings, &edges, &node_ids, &node_index, &config).unwrap();

        assert_eq!(snapshots.len(), 30);
        let first = snapshots[0].loss.total;
        let last = snapshots.last().unwrap().loss.total;
        assert!(first.is_finite());
        assert!(last.is_finite());
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn evaluate_returns_valid_metrics() {
        let node_ids = vec![0usize, 1, 2, 3];
        let embeddings = vec![
            vec![0.0, 0.5],
            vec![1.0, 0.0],
            vec![0.5, 1.0],
            vec![0.2, 0.8],
        ];
        let node_index: std::collections::HashMap<usize, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let encoder = TaxoBellEncoder::new(2, 4, 2, &Device::Cpu).unwrap();
        let test_edges = vec![(0, 1), (0, 2)];
        let result =
            evaluate_taxobell(&encoder, &embeddings, &test_edges, &node_ids, &node_index).unwrap();
        assert!(result.mrr >= 0.0 && result.mrr <= 1.0);
        assert!(result.hits_at_1 >= 0.0 && result.hits_at_1 <= 1.0);
        assert!(result.hits_at_10 >= 0.0 && result.hits_at_10 <= 1.0);
    }

    #[test]
    fn evaluate_empty_edges() {
        let node_ids = vec![0usize, 1];
        let embeddings = vec![vec![0.0, 0.5], vec![1.0, 0.0]];
        let node_index: std::collections::HashMap<usize, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let encoder = TaxoBellEncoder::new(2, 4, 2, &Device::Cpu).unwrap();
        let result = evaluate_taxobell(&encoder, &embeddings, &[], &node_ids, &node_index).unwrap();
        assert_eq!(result.mrr, 0.0);
    }

    #[test]
    fn encode_one_matches_batch() {
        let enc = TaxoBellEncoder::new(4, 8, 2, &device()).unwrap();
        let embed = vec![0.5f32, -0.3, 1.0, 0.2];

        // Encode via batch path.
        let t = Tensor::new(&embed[..], &device())
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let (mu_batch, sigma_batch) = enc.encode(&t).unwrap();
        let mu_b: Vec<f32> = mu_batch.squeeze(0).unwrap().to_vec1().unwrap();
        let sigma_b: Vec<f32> = sigma_batch.squeeze(0).unwrap().to_vec1().unwrap();

        // Encode via single path.
        let gb = enc.encode_one(&embed).unwrap();

        for (a, b) in mu_b.iter().zip(gb.mu.iter()) {
            assert!((a - b).abs() < 1e-5, "mu mismatch: {a} vs {b}");
        }
        for (a, b) in sigma_b.iter().zip(gb.sigma.iter()) {
            assert!((a - b).abs() < 1e-5, "sigma mismatch: {a} vs {b}");
        }
    }
}
