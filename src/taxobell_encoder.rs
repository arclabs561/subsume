//! TaxoBell MLP encoder: text embeddings → Gaussian box parameters.
//!
//! Maps pre-computed text embeddings to diagonal Gaussian boxes via two
//! small MLPs (one for center, one for offset). The offset MLP output is
//! passed through softplus to ensure positive sigma values.
//!
//! Architecture per MLP:
//! ```text
//! input (embed_dim) → Linear → ReLU → Linear → output (box_dim)
//! ```
//!
//! # References
//!
//! - TaxoBell (WWW 2026, arXiv:2601.09633), Section 4.1: Gaussian box encoder

use crate::gaussian::GaussianBox;
use crate::optimizer::AMSGradState;
use crate::taxobell::{CombinedLossResult, TaxoBellConfig, TaxoBellLoss};
use crate::BoxError;

/// A two-layer MLP: `Linear(in, hidden) → ReLU → Linear(hidden, out)`.
///
/// Weights are stored in row-major order.
#[derive(Debug, Clone)]
pub struct Mlp {
    /// Weight matrix for the first layer, shape `[hidden_dim, input_dim]` (row-major).
    pub w1: Vec<f32>,
    /// Bias vector for the first layer, length `hidden_dim`.
    pub b1: Vec<f32>,
    /// Weight matrix for the second layer, shape `[output_dim, hidden_dim]` (row-major).
    pub w2: Vec<f32>,
    /// Bias vector for the second layer, length `output_dim`.
    pub b2: Vec<f32>,
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
}

impl Mlp {
    /// Create a new MLP with Xavier-uniform initialization.
    ///
    /// Uses a simple xorshift64 PRNG seeded by `seed` to avoid pulling in `rand`.
    #[must_use]
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut state = seed.wrapping_add(1);
        let mut next_f32 = move || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1, 1]
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        // Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        let scale1 = (6.0 / (input_dim + hidden_dim) as f32).sqrt();
        let w1: Vec<f32> = (0..hidden_dim * input_dim)
            .map(|_| next_f32() * scale1)
            .collect();
        let b1 = vec![0.0; hidden_dim];

        let scale2 = (6.0 / (hidden_dim + output_dim) as f32).sqrt();
        let w2: Vec<f32> = (0..output_dim * hidden_dim)
            .map(|_| next_f32() * scale2)
            .collect();
        let b2 = vec![0.0; output_dim];

        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Forward pass: `Linear → ReLU → Linear`.
    #[must_use]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input_dim);

        // Layer 1: hidden = ReLU(W1 * input + b1)
        let mut hidden = self.b1.clone();
        for h in 0..self.hidden_dim {
            let row_start = h * self.input_dim;
            let mut sum = hidden[h];
            for j in 0..self.input_dim {
                sum += self.w1[row_start + j] * input[j];
            }
            hidden[h] = sum.max(0.0); // ReLU
        }

        // Layer 2: output = W2 * hidden + b2
        let mut output = self.b2.clone();
        for o in 0..self.output_dim {
            let row_start = o * self.hidden_dim;
            let mut sum = output[o];
            for h in 0..self.hidden_dim {
                sum += self.w2[row_start + h] * hidden[h];
            }
            output[o] = sum;
        }

        output
    }

    /// Total number of learnable parameters.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    /// Collect all parameters into a flat vector (for optimizer updates).
    #[must_use]
    pub fn params_flat(&self) -> Vec<f32> {
        let mut p = Vec::with_capacity(self.num_params());
        p.extend_from_slice(&self.w1);
        p.extend_from_slice(&self.b1);
        p.extend_from_slice(&self.w2);
        p.extend_from_slice(&self.b2);
        p
    }

    /// Set parameters from a flat vector.
    pub fn set_params_flat(&mut self, params: &[f32]) {
        debug_assert_eq!(params.len(), self.num_params());
        let mut offset = 0;
        let n = self.w1.len();
        self.w1.copy_from_slice(&params[offset..offset + n]);
        offset += n;
        let n = self.b1.len();
        self.b1.copy_from_slice(&params[offset..offset + n]);
        offset += n;
        let n = self.w2.len();
        self.w2.copy_from_slice(&params[offset..offset + n]);
        offset += n;
        let n = self.b2.len();
        self.b2.copy_from_slice(&params[offset..offset + n]);
    }
}

/// TaxoBell encoder: two MLPs mapping text embeddings to Gaussian box parameters.
///
/// - `center_mlp`: text embedding → center (mu) of the Gaussian
/// - `offset_mlp`: text embedding → raw offset, then softplus → sigma
///
/// The resulting `GaussianBox` has `mu = center_mlp(embed)` and
/// `sigma = softplus(offset_mlp(embed))`.
#[derive(Debug, Clone)]
pub struct TaxoBellEncoder {
    /// MLP producing the center (mu) of the Gaussian box.
    pub center_mlp: Mlp,
    /// MLP producing the raw offset (pre-softplus sigma) of the Gaussian box.
    pub offset_mlp: Mlp,
    /// Embedding dimension (input to both MLPs).
    pub embed_dim: usize,
    /// Box dimension (output of both MLPs).
    pub box_dim: usize,
}

impl TaxoBellEncoder {
    /// Create a new encoder with Xavier-initialized MLPs.
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - dimension of input text embeddings
    /// * `hidden_dim` - hidden layer width for both MLPs
    /// * `box_dim` - dimension of the output Gaussian box
    /// * `seed` - RNG seed for weight initialization
    #[must_use]
    pub fn new(embed_dim: usize, hidden_dim: usize, box_dim: usize, seed: u64) -> Self {
        Self {
            center_mlp: Mlp::new(embed_dim, hidden_dim, box_dim, seed),
            offset_mlp: Mlp::new(embed_dim, hidden_dim, box_dim, seed.wrapping_add(7919)),
            embed_dim,
            box_dim,
        }
    }

    /// Encode a text embedding into a Gaussian box.
    ///
    /// Returns `GaussianBox` with `mu = center_mlp(embed)` and
    /// `sigma = softplus(offset_mlp(embed))`.
    pub fn encode(&self, embedding: &[f32]) -> Result<GaussianBox, BoxError> {
        let center = self.center_mlp.forward(embedding);
        let raw_offset = self.offset_mlp.forward(embedding);
        GaussianBox::from_center_offset(center, raw_offset)
    }

    /// Total number of learnable parameters across both MLPs.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.center_mlp.num_params() + self.offset_mlp.num_params()
    }

    /// Collect all parameters into a flat vector.
    #[must_use]
    pub fn params_flat(&self) -> Vec<f32> {
        let mut p = self.center_mlp.params_flat();
        p.extend(self.offset_mlp.params_flat());
        p
    }

    /// Set all parameters from a flat vector.
    pub fn set_params_flat(&mut self, params: &[f32]) {
        let split = self.center_mlp.num_params();
        self.center_mlp.set_params_flat(&params[..split]);
        self.offset_mlp.set_params_flat(&params[split..]);
    }
}

/// Configuration for TaxoBell training.
#[derive(Debug, Clone)]
pub struct TaxoBellTrainingConfig {
    /// Learning rate for AMSGrad.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Step size for numerical gradient estimation (finite differences).
    pub grad_epsilon: f32,
    /// Number of negative samples per positive edge.
    pub num_negatives: usize,
    /// Loss function configuration.
    pub loss_config: TaxoBellConfig,
    /// Hidden dimension for the encoder MLPs.
    pub hidden_dim: usize,
    /// Output box dimension.
    pub box_dim: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Number of warmup epochs (linear LR warmup).
    pub warmup_epochs: usize,
}

impl Default for TaxoBellTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 100,
            grad_epsilon: 1e-4,
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

/// Train a TaxoBell encoder on a taxonomy dataset.
///
/// # Arguments
///
/// * `embeddings` - pre-computed text embeddings, one per node (indexed by node position)
/// * `edges` - training edges as `(parent_id, child_id)` pairs
/// * `all_node_ids` - list of all node IDs (for negative sampling)
/// * `config` - training configuration
///
/// # Returns
///
/// The trained encoder and a vec of per-epoch loss snapshots.
pub fn train_taxobell(
    embeddings: &[Vec<f32>],
    edges: &[(usize, usize)],
    all_node_ids: &[usize],
    node_index: &std::collections::HashMap<usize, usize>,
    config: &TaxoBellTrainingConfig,
) -> (TaxoBellEncoder, Vec<TrainingSnapshot>) {
    let embed_dim = embeddings[0].len();
    let mut encoder =
        TaxoBellEncoder::new(embed_dim, config.hidden_dim, config.box_dim, config.seed);
    let loss_fn = TaxoBellLoss::new(config.loss_config.clone());

    let n_params = encoder.num_params();
    let mut opt_state = AMSGradState::new(n_params, config.learning_rate);
    let mut snapshots = Vec::with_capacity(config.epochs);

    // Simple xorshift for negative sampling
    let mut rng_state = config.seed.wrapping_add(1);
    let mut rng_next = |bound: usize| -> usize {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as usize) % bound
    };

    for epoch in 0..config.epochs {
        // Learning rate schedule: linear warmup + cosine decay
        let lr = crate::optimizer::get_learning_rate(
            epoch,
            config.epochs,
            config.learning_rate,
            config.warmup_epochs,
        );
        opt_state.set_lr(lr);

        // --- Compute loss for the current parameters ---
        let loss_result = compute_batch_loss(
            &encoder,
            embeddings,
            edges,
            all_node_ids,
            node_index,
            &loss_fn,
            config.num_negatives,
            &mut rng_next,
        );

        snapshots.push(TrainingSnapshot {
            epoch,
            loss: loss_result,
            lr,
        });

        // --- Numerical gradients via forward finite differences ---
        let params = encoder.params_flat();
        let base_loss = loss_result.total;
        let mut grads = vec![0.0f32; n_params];

        for i in 0..n_params {
            let mut params_plus = params.clone();
            params_plus[i] += config.grad_epsilon;
            encoder.set_params_flat(&params_plus);

            let loss_plus = compute_batch_loss(
                &encoder,
                embeddings,
                edges,
                all_node_ids,
                node_index,
                &loss_fn,
                config.num_negatives,
                &mut rng_next,
            )
            .total;

            grads[i] = (loss_plus - base_loss) / config.grad_epsilon;
        }

        // Restore original params before update
        encoder.set_params_flat(&params);

        // --- AMSGrad update ---
        opt_state.t += 1;
        let t = opt_state.t as f32;
        let bias_correction1 = 1.0 - opt_state.beta1.powf(t);

        let mut new_params = params;
        for i in 0..n_params {
            let g = grads[i];
            opt_state.m[i] = opt_state.beta1 * opt_state.m[i] + (1.0 - opt_state.beta1) * g;
            let v_new = opt_state.beta2 * opt_state.v[i] + (1.0 - opt_state.beta2) * g * g;
            opt_state.v[i] = v_new;
            opt_state.v_hat[i] = opt_state.v_hat[i].max(v_new);

            let m_hat = opt_state.m[i] / bias_correction1;
            let update = opt_state.lr * m_hat / (opt_state.v_hat[i].sqrt() + opt_state.epsilon);
            new_params[i] -= update;
            if !new_params[i].is_finite() {
                new_params[i] = 0.0;
            }
        }

        encoder.set_params_flat(&new_params);
    }

    (encoder, snapshots)
}

/// Evaluate a trained encoder on a set of edges.
///
/// For each `(parent_id, child_id)` edge, computes the KL-divergence rank of the
/// true parent among all candidate nodes. Lower KL → better containment → higher rank.
pub fn evaluate_taxobell(
    encoder: &TaxoBellEncoder,
    embeddings: &[Vec<f32>],
    test_edges: &[(usize, usize)],
    all_node_ids: &[usize],
    node_index: &std::collections::HashMap<usize, usize>,
) -> TaxoBellEvalResult {
    if test_edges.is_empty() {
        return TaxoBellEvalResult {
            mrr: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
        };
    }

    // Pre-encode all nodes.
    let boxes: Vec<GaussianBox> = all_node_ids
        .iter()
        .map(|&id| {
            let idx = node_index[&id];
            encoder.encode(&embeddings[idx]).unwrap()
        })
        .collect();

    let mut reciprocal_ranks = Vec::with_capacity(test_edges.len());
    let mut hits1 = 0usize;
    let mut hits3 = 0usize;
    let mut hits10 = 0usize;

    for &(parent_id, child_id) in test_edges {
        let child_idx = node_index[&child_id];
        let child_box = encoder.encode(&embeddings[child_idx]).unwrap();

        // Score all candidate parents by KL(child || candidate): lower = better.
        let mut scores: Vec<(usize, f32)> = all_node_ids
            .iter()
            .enumerate()
            .map(|(pos, &cand_id)| {
                let kl =
                    crate::gaussian::kl_divergence(&child_box, &boxes[pos]).unwrap_or(f32::MAX);
                (cand_id, kl)
            })
            .collect();

        // Sort by ascending KL (best parent first).
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find rank of true parent (1-indexed).
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
    TaxoBellEvalResult {
        mrr: reciprocal_ranks.iter().sum::<f32>() / n,
        hits_at_1: hits1 as f32 / n,
        hits_at_3: hits3 as f32 / n,
        hits_at_10: hits10 as f32 / n,
    }
}

/// Compute the combined TaxoBell loss over a batch of edges with negative sampling.
fn compute_batch_loss(
    encoder: &TaxoBellEncoder,
    embeddings: &[Vec<f32>],
    edges: &[(usize, usize)],
    all_node_ids: &[usize],
    node_index: &std::collections::HashMap<usize, usize>,
    loss_fn: &TaxoBellLoss,
    num_negatives: usize,
    rng_next: &mut impl FnMut(usize) -> usize,
) -> CombinedLossResult {
    if edges.is_empty() {
        return CombinedLossResult {
            total: 0.0,
            l_sym: 0.0,
            l_asym: 0.0,
            l_reg: 0.0,
            l_clip: 0.0,
        };
    }

    // Encode all referenced nodes.
    let mut box_cache: std::collections::HashMap<usize, GaussianBox> =
        std::collections::HashMap::new();
    let mut ensure_box = |id: usize, enc: &TaxoBellEncoder| -> GaussianBox {
        box_cache
            .entry(id)
            .or_insert_with(|| {
                let idx = node_index[&id];
                enc.encode(&embeddings[idx]).unwrap()
            })
            .clone()
    };

    // Build positive pairs and negative triples.
    let mut all_boxes_vec: Vec<GaussianBox> = Vec::new();

    // Positives: (child, parent)
    let mut pos_pairs: Vec<(GaussianBox, GaussianBox)> = Vec::with_capacity(edges.len());
    for &(parent_id, child_id) in edges {
        let child_box = ensure_box(child_id, encoder);
        let parent_box = ensure_box(parent_id, encoder);
        pos_pairs.push((child_box, parent_box));
    }

    // Negatives: for each edge, sample random negative nodes for symmetric triplet loss.
    let mut neg_triples: Vec<(GaussianBox, GaussianBox, GaussianBox)> = Vec::new();
    for &(parent_id, child_id) in edges {
        let anchor = ensure_box(child_id, encoder);
        let positive = ensure_box(parent_id, encoder);
        for _ in 0..num_negatives {
            let neg_id = all_node_ids[rng_next(all_node_ids.len())];
            let negative = ensure_box(neg_id, encoder);
            neg_triples.push((anchor.clone(), positive.clone(), negative));
        }
    }

    // Collect all unique boxes for regularization.
    for (_, gb) in &box_cache {
        all_boxes_vec.push(gb.clone());
    }

    // Build references for the loss function.
    let positives_ref: Vec<(&GaussianBox, &GaussianBox)> =
        pos_pairs.iter().map(|(c, p)| (c, p)).collect();
    let negatives_ref: Vec<(&GaussianBox, &GaussianBox, &GaussianBox)> =
        neg_triples.iter().map(|(a, p, n)| (a, p, n)).collect();
    let all_ref: Vec<&GaussianBox> = all_boxes_vec.iter().collect();

    loss_fn
        .combined_loss(&positives_ref, &negatives_ref, &all_ref)
        .unwrap_or(CombinedLossResult {
            total: 0.0,
            l_sym: 0.0,
            l_asym: 0.0,
            l_reg: 0.0,
            l_clip: 0.0,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_forward_shape() {
        let mlp = Mlp::new(8, 16, 4, 42);
        let input = vec![1.0; 8];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mlp_params_roundtrip() {
        let mlp = Mlp::new(4, 8, 2, 123);
        let params = mlp.params_flat();
        assert_eq!(params.len(), mlp.num_params());

        let mut mlp2 = Mlp::new(4, 8, 2, 999); // different init
        mlp2.set_params_flat(&params);
        assert_eq!(mlp2.params_flat(), params);
    }

    #[test]
    fn encoder_produces_valid_gaussian() {
        let encoder = TaxoBellEncoder::new(8, 16, 4, 42);
        let embed = vec![0.5; 8];
        let gb = encoder.encode(&embed).unwrap();
        assert_eq!(gb.dim(), 4);
        for &s in &gb.sigma {
            assert!(s > 0.0, "sigma must be positive, got {s}");
        }
    }

    #[test]
    fn encoder_params_roundtrip() {
        let encoder = TaxoBellEncoder::new(8, 16, 4, 42);
        let params = encoder.params_flat();
        assert_eq!(params.len(), encoder.num_params());

        let mut encoder2 = TaxoBellEncoder::new(8, 16, 4, 999);
        encoder2.set_params_flat(&params);
        assert_eq!(encoder2.params_flat(), params);
    }

    #[test]
    fn encoder_different_inputs_different_outputs() {
        let encoder = TaxoBellEncoder::new(8, 16, 4, 42);
        let a = encoder.encode(&vec![0.0; 8]).unwrap();
        let b = encoder.encode(&vec![1.0; 8]).unwrap();
        // Outputs should differ (not a degenerate encoder).
        assert_ne!(
            a.mu, b.mu,
            "different inputs should produce different centers"
        );
    }

    #[test]
    fn train_loss_decreases() {
        // Tiny taxonomy: root → A, root → B
        let node_ids = vec![0usize, 1, 2];
        let edges = vec![(0, 1), (0, 2)]; // parent=0, children=1,2
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
            learning_rate: 1e-2,
            epochs: 20,
            grad_epsilon: 1e-3,
            num_negatives: 1,
            hidden_dim: 8,
            box_dim: 4,
            seed: 42,
            warmup_epochs: 2,
            ..Default::default()
        };

        let (_, snapshots) = train_taxobell(&embeddings, &edges, &node_ids, &node_index, &config);

        assert_eq!(snapshots.len(), 20);
        let first_loss = snapshots[0].loss.total;
        let last_loss = snapshots.last().unwrap().loss.total;
        assert!(
            last_loss <= first_loss + 0.1,
            "loss should not increase significantly: first={first_loss}, last={last_loss}"
        );
        assert!(first_loss.is_finite());
        assert!(last_loss.is_finite());
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

        let encoder = TaxoBellEncoder::new(2, 4, 2, 42);
        let test_edges = vec![(0, 1), (0, 2)];

        let result = evaluate_taxobell(&encoder, &embeddings, &test_edges, &node_ids, &node_index);
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

        let encoder = TaxoBellEncoder::new(2, 4, 2, 42);
        let result = evaluate_taxobell(&encoder, &embeddings, &[], &node_ids, &node_index);
        assert_eq!(result.mrr, 0.0);
    }
}
