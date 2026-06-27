//! Burn-based TaxoBell Gaussian-box encoder with autodiff.
//!
//! Burn port of the candle `taxobell_encoder`: maps pre-computed text
//! embeddings to diagonal Gaussian boxes via two small MLPs (one for the
//! center mu, one for the offset that becomes sigma after softplus), trained
//! with the four-component TaxoBell loss (symmetric Bhattacharyya BCE,
//! asymmetric KL containment, volume regularization, sigma ceiling).
//!
//! Architecture per MLP: `Linear(embed, hidden) -> ReLU -> Linear(hidden, box)`.
//!
//! This is the Burn counterpart of the candle path; it exists so that the
//! encoder trains on any Burn backend (ndarray/wgpu/tch) and composes with
//! Burn-native tooling. The loss math is identical to `taxobell_encoder.rs`.
//!
//! # References
//! - TaxoBell (WWW 2026, arXiv:2601.09633), Section 4.1: Gaussian box encoder

#![allow(missing_docs)]

use crate::taxobell::{CombinedLossResult, TaxoBellConfig};
use crate::BoxError;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::ToElement;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{activation, Int, Tensor};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Numerically stable softplus
// ---------------------------------------------------------------------------

/// Softplus: `ln(1 + exp(x))`, numerically stable as `relu(x) + ln(1 + exp(-|x|))`.
fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().clamp_min(0.0) + x.abs().neg().exp().add_scalar(1.0).log()
}

// ---------------------------------------------------------------------------
// Encoder model
// ---------------------------------------------------------------------------

/// Two-layer MLP: `Linear -> ReLU -> Linear`.
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(input_dim, hidden_dim).init(device),
            lin2: LinearConfig::new(hidden_dim, output_dim).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.lin2.forward(activation::relu(self.lin1.forward(x)))
    }
}

/// TaxoBell encoder: two MLPs mapping text embeddings to Gaussian box parameters.
///
/// - `center`: embedding -> mu (center of the Gaussian)
/// - `offset`: embedding -> raw offset -> softplus -> sigma (std dev, positive)
#[derive(Module, Debug)]
pub struct BurnTaxoBellEncoder<B: Backend> {
    center: Mlp<B>,
    offset: Mlp<B>,
    box_dim: usize,
}

impl<B: Backend> BurnTaxoBellEncoder<B> {
    /// Create a new encoder.
    pub fn new(embed_dim: usize, hidden_dim: usize, box_dim: usize, device: &B::Device) -> Self {
        Self {
            center: Mlp::new(embed_dim, hidden_dim, box_dim, device),
            offset: Mlp::new(embed_dim, hidden_dim, box_dim, device),
            box_dim,
        }
    }

    /// Encode a batch of embeddings `[batch, embed_dim]` into `(mu, sigma)`,
    /// each `[batch, box_dim]` with `sigma > 0`.
    pub fn encode(&self, embeddings: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mu = self.center.forward(embeddings.clone());
        let sigma = softplus(self.offset.forward(embeddings));
        (mu, sigma)
    }

    /// Box output dimension.
    #[must_use]
    pub fn box_dim(&self) -> usize {
        self.box_dim
    }
}

// ---------------------------------------------------------------------------
// Differentiable loss components (operate on [batch, dim], reduce to [batch, 1])
// ---------------------------------------------------------------------------

/// Bhattacharyya coefficient between batches of diagonal Gaussians. Returns `[batch, 1]` in `[0, 1]`.
fn bhattacharyya_coeff<B: Backend>(
    mu1: Tensor<B, 2>,
    s1: Tensor<B, 2>,
    mu2: Tensor<B, 2>,
    s2: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let v1 = s1.powf_scalar(2.0);
    let v2 = s2.powf_scalar(2.0);
    let sigma_avg = (v1.clone() + v2.clone()).mul_scalar(0.5);
    let mu_diff = mu1 - mu2;

    // BD = 0.25*sum(mu_diff^2/sigma_avg) + 0.5*sum(ln(sigma_avg)) - 0.25*sum(ln v1) - 0.25*sum(ln v2)
    let t1 = (mu_diff.powf_scalar(2.0) / sigma_avg.clone())
        .sum_dim(1)
        .mul_scalar(0.25);
    let t2 = sigma_avg.log().sum_dim(1).mul_scalar(0.5);
    let t3 = v1.log().sum_dim(1).mul_scalar(0.25);
    let t4 = v2.log().sum_dim(1).mul_scalar(0.25);

    let bd = t1 + t2 - t3 - t4;
    bd.neg().exp() // BC = exp(-BD)
}

/// KL divergence `KL(q || p)` between batches of diagonal Gaussians. Returns `[batch, 1]` >= 0.
fn kl_divergence<B: Backend>(
    mu_q: Tensor<B, 2>,
    s_q: Tensor<B, 2>,
    mu_p: Tensor<B, 2>,
    s_p: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let vq = s_q.powf_scalar(2.0);
    let vp = s_p.powf_scalar(2.0);
    let mu_diff = mu_p - mu_q;

    let ratio = vq.clone() / vp.clone();
    let mu_term = mu_diff.powf_scalar(2.0) / vp.clone();
    let log_term = (vp / vq).log();

    // per_dim = ratio + mu_term + log_term - 1; KL = 0.5 * sum(per_dim)
    (ratio + mu_term + log_term)
        .sub_scalar(1.0)
        .sum_dim(1)
        .mul_scalar(0.5)
}

/// Log-volume of diagonal Gaussians: `sum(ln(sigma))`. Returns `[batch, 1]`.
fn log_volume<B: Backend>(sigma: Tensor<B, 2>) -> Tensor<B, 2> {
    sigma.log().sum_dim(1)
}

/// Volume regularization (squared hinge floor on variance). Returns `[batch, 1]`.
fn volume_reg<B: Backend>(sigma: Tensor<B, 2>, min_var: f32) -> Tensor<B, 2> {
    let d = sigma.dims()[1] as f32;
    let var = sigma.powf_scalar(2.0);
    // gap = min_var - var; hinge = relu(gap); sum(hinge^2)/d
    let hinge = var
        .mul_scalar(-1.0)
        .add_scalar(min_var as f64)
        .clamp_min(0.0);
    hinge.powf_scalar(2.0).sum_dim(1).div_scalar(d as f64)
}

/// Sigma ceiling (linear hinge ceiling on variance). Returns `[batch, 1]`.
fn sigma_ceiling<B: Backend>(sigma: Tensor<B, 2>, max_var: f32) -> Tensor<B, 2> {
    let d = sigma.dims()[1] as f32;
    let var = sigma.powf_scalar(2.0);
    // gap = var - max_var; hinge = relu(gap); sum(hinge)/d
    let hinge = var.add_scalar(-(max_var as f64)).clamp_min(0.0);
    hinge.sum_dim(1).div_scalar(d as f64)
}

// ---------------------------------------------------------------------------
// Combined loss
// ---------------------------------------------------------------------------

/// Gathered box parameters for one training step.
struct LossInputs<B: Backend> {
    mu_child: Tensor<B, 2>,
    s_child: Tensor<B, 2>,
    mu_parent: Tensor<B, 2>,
    s_parent: Tensor<B, 2>,
    mu_anchor: Tensor<B, 2>,
    s_anchor: Tensor<B, 2>,
    mu_pos: Tensor<B, 2>,
    s_pos: Tensor<B, 2>,
    mu_neg: Tensor<B, 2>,
    s_neg: Tensor<B, 2>,
    /// All boxes' sigma `[n_nodes, box_dim]`, for the volume/ceiling regularizers.
    s_all: Tensor<B, 2>,
}

/// Compute the four-component TaxoBell loss. Returns `(total_loss, breakdown)`.
///
/// `total_loss` is a scalar `[1]` tensor supporting `.backward()`.
fn combined_loss<B: Backend>(
    inp: LossInputs<B>,
    config: &TaxoBellConfig,
) -> (Tensor<B, 1>, CombinedLossResult) {
    let eps = 1e-7f64;
    let one_minus_eps = 1.0 - eps;

    // --- L_sym: symmetric BCE on Bhattacharyya coefficient ---
    let bc_pos = bhattacharyya_coeff(
        inp.mu_anchor.clone(),
        inp.s_anchor.clone(),
        inp.mu_pos,
        inp.s_pos,
    )
    .clamp(eps, one_minus_eps);
    let bc_neg = bhattacharyya_coeff(inp.mu_anchor, inp.s_anchor, inp.mu_neg, inp.s_neg)
        .clamp(eps, one_minus_eps);
    // -log(bc_pos) - log(1 - bc_neg)
    let term1 = bc_pos.log().neg();
    let term2 = bc_neg.mul_scalar(-1.0).add_scalar(1.0).log().neg();
    let l_sym = (term1 + term2).mean();

    // --- L_asym: asymmetric KL containment ---
    let kl = kl_divergence(
        inp.mu_child.clone(),
        inp.s_child.clone(),
        inp.mu_parent.clone(),
        inp.s_parent.clone(),
    );
    // L_align = relu(kl - margin)
    let l_align = kl
        .clone()
        .add_scalar(-(config.asymmetric_margin as f64))
        .clamp_min(0.0);

    let l_diverge = if config.asymmetric_diverge_c > 0.0 {
        let kl_rev = kl_divergence(
            inp.mu_parent.clone(),
            inp.s_parent.clone(),
            inp.mu_child.clone(),
            inp.s_child.clone(),
        );
        let d_rep = log_volume(inp.s_parent.clone()) - log_volume(inp.s_child.clone());
        // relu(C * d_rep - kl_rev)
        (d_rep.mul_scalar(config.asymmetric_diverge_c as f64) - kl_rev).clamp_min(0.0)
    } else {
        kl.zeros_like()
    };
    let l_asym = (l_align + l_diverge.mul_scalar(config.diverge_lambda as f64)).mean();

    // --- L_reg + L_clip on all boxes ---
    let l_reg = volume_reg(inp.s_all.clone(), config.min_var).mean();
    let l_clip = sigma_ceiling(inp.s_all, config.max_var).mean();

    // --- Total: weighted sum ---
    let total = l_sym.clone().mul_scalar(config.alpha as f64)
        + l_asym.clone().mul_scalar(config.beta as f64)
        + l_reg.clone().mul_scalar(config.gamma as f64)
        + l_clip.clone().mul_scalar(config.delta as f64);

    let breakdown = CombinedLossResult {
        total: total.clone().into_scalar().to_f32(),
        l_sym: l_sym.into_scalar().to_f32(),
        l_asym: l_asym.into_scalar().to_f32(),
        l_reg: l_reg.into_scalar().to_f32(),
        l_clip: l_clip.into_scalar().to_f32(),
    };

    (total, breakdown)
}

// ---------------------------------------------------------------------------
// Training config / result types (plain Rust, backend-agnostic)
// ---------------------------------------------------------------------------

/// Configuration for Burn TaxoBell encoder training.
#[derive(Debug, Clone)]
pub struct BurnTaxoBellTrainingConfig {
    /// Learning rate for Adam.
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

impl Default for BurnTaxoBellTrainingConfig {
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

/// Result of evaluating a trained encoder via KL-ranking.
#[derive(Debug, Clone)]
pub struct BurnTaxoBellEvalResult {
    /// Mean Reciprocal Rank.
    pub mrr: f32,
    /// Hits@1.
    pub hits_at_1: f32,
    /// Hits@3.
    pub hits_at_3: f32,
    /// Hits@10.
    pub hits_at_10: f32,
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Train a Burn TaxoBell encoder on a taxonomy dataset.
///
/// Returns the trained encoder and the per-epoch total loss.
///
/// * `embeddings` - pre-computed text embeddings, one per node
/// * `edges` - training edges as `(parent_id, child_id)` pairs
/// * `all_node_ids` - all node IDs (for negative sampling)
/// * `node_index` - maps node ID -> index into `embeddings`
pub fn train_taxobell_burn<B: AutodiffBackend>(
    embeddings: &[Vec<f32>],
    edges: &[(usize, usize)],
    all_node_ids: &[usize],
    node_index: &HashMap<usize, usize>,
    config: &BurnTaxoBellTrainingConfig,
    device: &B::Device,
) -> Result<(BurnTaxoBellEncoder<B>, Vec<f32>), BoxError> {
    if embeddings.is_empty() || edges.is_empty() {
        return Err(BoxError::Internal(
            "train_taxobell_burn requires non-empty embeddings and edges".into(),
        ));
    }
    let embed_dim = embeddings[0].len();
    let n_nodes = embeddings.len();

    let flat: Vec<f32> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();
    let all_embeds =
        Tensor::<B, 1>::from_data(flat.as_slice(), device).reshape([n_nodes, embed_dim]);

    let child_indices: Vec<i64> = edges.iter().map(|&(_, c)| node_index[&c] as i64).collect();
    let parent_indices: Vec<i64> = edges.iter().map(|&(p, _)| node_index[&p] as i64).collect();

    let child_idx = Tensor::<B, 1, Int>::from_data(child_indices.as_slice(), device);
    let parent_idx = Tensor::<B, 1, Int>::from_data(parent_indices.as_slice(), device);
    let encoder =
        BurnTaxoBellEncoder::<B>::new(embed_dim, config.hidden_dim, config.box_dim, device);
    let mut model = encoder;
    let mut optim = AdamConfig::new()
        .with_epsilon(1e-8)
        .init::<B, BurnTaxoBellEncoder<B>>();

    let n_edges = edges.len();
    let n_neg = config.num_negatives.max(1);
    let n_total_neg = n_edges * n_neg;

    let mut rng = fastrand::Rng::with_seed(config.seed.wrapping_add(1));
    let mut losses = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr = crate::optimizer::get_learning_rate(
            epoch,
            config.epochs,
            config.learning_rate,
            config.warmup_epochs,
        );

        // Sample negatives + anchor/positive index arrays (anchor=child, positive=parent).
        let neg_indices: Vec<i64> = (0..n_total_neg)
            .map(|_| {
                let node_id = all_node_ids[rng.usize(0..all_node_ids.len())];
                node_index[&node_id] as i64
            })
            .collect();
        let anchor_indices: Vec<i64> = child_indices
            .iter()
            .flat_map(|&idx| std::iter::repeat_n(idx, n_neg))
            .collect();
        let pos_indices: Vec<i64> = parent_indices
            .iter()
            .flat_map(|&idx| std::iter::repeat_n(idx, n_neg))
            .collect();

        let anchor_idx = Tensor::<B, 1, Int>::from_data(anchor_indices.as_slice(), device);
        let pos_idx = Tensor::<B, 1, Int>::from_data(pos_indices.as_slice(), device);
        let neg_idx = Tensor::<B, 1, Int>::from_data(neg_indices.as_slice(), device);

        // Forward: encode all nodes once.
        let (mu_all, s_all) = model.encode(all_embeds.clone());

        let gather = |t: &Tensor<B, 2>, idx: &Tensor<B, 1, Int>| t.clone().select(0, idx.clone());

        let inp = LossInputs {
            mu_child: gather(&mu_all, &child_idx),
            s_child: gather(&s_all, &child_idx),
            mu_parent: gather(&mu_all, &parent_idx),
            s_parent: gather(&s_all, &parent_idx),
            mu_anchor: gather(&mu_all, &anchor_idx),
            s_anchor: gather(&s_all, &anchor_idx),
            mu_pos: gather(&mu_all, &pos_idx),
            s_pos: gather(&s_all, &pos_idx),
            mu_neg: gather(&mu_all, &neg_idx),
            s_neg: gather(&s_all, &neg_idx),
            s_all: s_all.clone(),
        };

        let (loss, breakdown) = combined_loss(inp, &config.loss_config);
        losses.push(breakdown.total);

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr as f64, model, grads);
    }

    Ok((model, losses))
}

/// Evaluate a trained encoder: rank the true parent of each test edge among all
/// candidates by KL divergence (lower KL = better containment).
pub fn evaluate_taxobell_burn<B: Backend>(
    encoder: &BurnTaxoBellEncoder<B>,
    embeddings: &[Vec<f32>],
    test_edges: &[(usize, usize)],
    all_node_ids: &[usize],
    node_index: &HashMap<usize, usize>,
    device: &B::Device,
) -> Result<BurnTaxoBellEvalResult, BoxError> {
    if test_edges.is_empty() {
        return Ok(BurnTaxoBellEvalResult {
            mrr: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
        });
    }
    let embed_dim = embeddings[0].len();
    let n_nodes = embeddings.len();
    let flat: Vec<f32> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();
    let all_embeds =
        Tensor::<B, 1>::from_data(flat.as_slice(), device).reshape([n_nodes, embed_dim]);
    let (mu_all, s_all) = encoder.encode(all_embeds);
    let mu: Vec<f32> = mu_all.into_data().to_vec().unwrap();
    let sigma: Vec<f32> = s_all.into_data().to_vec().unwrap();
    let box_dim = encoder.box_dim;

    let boxes: Vec<crate::gaussian::GaussianBox> = all_node_ids
        .iter()
        .map(|&id| {
            let i = node_index[&id];
            crate::gaussian::GaussianBox::new(
                mu[i * box_dim..(i + 1) * box_dim].to_vec(),
                sigma[i * box_dim..(i + 1) * box_dim].to_vec(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut rr = Vec::with_capacity(test_edges.len());
    let (mut h1, mut h3, mut h10) = (0usize, 0usize, 0usize);

    for &(parent_id, child_id) in test_edges {
        let ci = node_index[&child_id];
        let child_box = crate::gaussian::GaussianBox::new(
            mu[ci * box_dim..(ci + 1) * box_dim].to_vec(),
            sigma[ci * box_dim..(ci + 1) * box_dim].to_vec(),
        )?;
        let mut scores: Vec<(usize, f32)> = all_node_ids
            .iter()
            .enumerate()
            .filter(|(_, &cand)| cand != child_id)
            .map(|(pos, &cand)| {
                let kl =
                    crate::gaussian::kl_divergence(&child_box, &boxes[pos]).unwrap_or(f32::MAX);
                (cand, kl)
            })
            .collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let rank = scores
            .iter()
            .position(|&(id, _)| id == parent_id)
            .map(|r| r + 1)
            .unwrap_or(scores.len());
        rr.push(1.0 / rank as f32);
        if rank <= 1 {
            h1 += 1;
        }
        if rank <= 3 {
            h3 += 1;
        }
        if rank <= 10 {
            h10 += 1;
        }
    }

    let n = test_edges.len() as f32;
    Ok(BurnTaxoBellEvalResult {
        mrr: rr.iter().sum::<f32>() / n,
        hits_at_1: h1 as f32 / n,
        hits_at_3: h3 as f32 / n,
        hits_at_10: h10 as f32 / n,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray>;

    fn device() -> <TestBackend as Backend>::Device {
        Default::default()
    }

    #[test]
    fn encoder_produces_positive_sigma() {
        let dev = device();
        let enc = BurnTaxoBellEncoder::<TestBackend>::new(8, 16, 4, &dev);
        let x = Tensor::<TestBackend, 2>::ones([2, 8], &dev);
        let (_mu, sigma) = enc.encode(x);
        let vals: Vec<f32> = sigma.into_data().to_vec().unwrap();
        for s in vals {
            assert!(s > 0.0, "sigma must be positive, got {s}");
        }
    }

    #[test]
    fn encoder_different_inputs_different_outputs() {
        let dev = device();
        let enc = BurnTaxoBellEncoder::<TestBackend>::new(8, 16, 4, &dev);
        let a = Tensor::<TestBackend, 2>::zeros([1, 8], &dev);
        let b = Tensor::<TestBackend, 2>::ones([1, 8], &dev);
        let (mu_a, _) = enc.encode(a);
        let (mu_b, _) = enc.encode(b);
        let va: Vec<f32> = mu_a.into_data().to_vec().unwrap();
        let vb: Vec<f32> = mu_b.into_data().to_vec().unwrap();
        assert_ne!(va, vb, "different inputs should produce different centers");
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
        let node_index: HashMap<usize, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let config = BurnTaxoBellTrainingConfig {
            learning_rate: 5e-3,
            epochs: 40,
            num_negatives: 1,
            hidden_dim: 8,
            box_dim: 4,
            seed: 42,
            warmup_epochs: 3,
            ..Default::default()
        };

        let (_enc, losses) = train_taxobell_burn::<TestBackend>(
            &embeddings,
            &edges,
            &node_ids,
            &node_index,
            &config,
            &device(),
        )
        .unwrap();

        assert_eq!(losses.len(), 40);
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(first.is_finite() && last.is_finite());
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
        let node_index: HashMap<usize, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let enc = BurnTaxoBellEncoder::<TestBackend>::new(2, 4, 2, &device());
        let test_edges = vec![(0, 1), (0, 2)];
        let r = evaluate_taxobell_burn::<TestBackend>(
            &enc,
            &embeddings,
            &test_edges,
            &node_ids,
            &node_index,
            &device(),
        )
        .unwrap();
        assert!(r.mrr >= 0.0 && r.mrr <= 1.0);
        assert!(r.hits_at_10 >= 0.0 && r.hits_at_10 <= 1.0);
    }
}
