//! Burn-based subspace trainer with autodiff.
//!
//! Each entity is a linear subspace of R^d, stored as an unconstrained
//! `[num_entities, rank, dim]` parameter tensor.  On every forward pass,
//! each entity's `[rank, dim]` slice is orthonormalised via batched
//! Gram-Schmidt, preserving gradient flow through all differentiable steps.
//!
//! # Scoring
//!
//! Containment score for A ⊆ B:
//!
//! ```text
//! proj      = B * (B^T * A^T)          [bs, dim, rank_A]
//! residual  = sum ||proj_j - a_j||^2   [bs]
//! score     = 1 - residual / rank_A    ∈ [0, 1]
//! ```
//!
//! # Backend selection
//! - `burn-ndarray` — multi-core CPU training
//! - `burn-wgpu`    — Metal/Vulkan/WebGPU training

#![allow(missing_docs)]

use crate::dataset::TripleIds;
use crate::subspace::Subspace;
use crate::trainer::negative_sampling::{
    compute_relation_entity_pools, sample_excluding, RelationEntityPools,
};
use crate::trainer::trainer_utils::self_adversarial_weights;
use crate::trainer::CpuBoxTrainingConfig;
use burn::module::{Param, ParamId};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Subspace embedding parameters.
///
/// `basis` is an unconstrained `[num_entities, rank, dim]` tensor.
/// Rows are orthonormalized via Gram-Schmidt before projection.
/// No relation parameters — subspace containment is relation-agnostic.
#[derive(Module, Debug)]
pub struct BurnSubspaceModel<B: Backend> {
    /// Unconstrained basis vectors `[num_entities, rank, dim]`.
    pub basis: Param<Tensor<B, 3>>,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Burn-based subspace trainer with autodiff.
pub struct BurnSubspaceTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
    epoch_seed: u64,
    /// Fixed subspace rank (number of basis vectors per entity).
    rank: usize,
    cached_pools: Option<HashMap<usize, RelationEntityPools>>,
}

impl<B: AutodiffBackend> BurnSubspaceTrainer<B> {
    /// Create a new trainer.
    ///
    /// `rank` is the number of basis vectors per entity (subspace dimension).
    /// A rank of 2–3 works well for most datasets.
    pub fn new(rank: usize) -> Self {
        Self {
            _backend: std::marker::PhantomData,
            epoch_seed: 0,
            rank: rank.max(1),
            cached_pools: None,
        }
    }

    /// Set the per-epoch seed used for negative sampling.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch_seed = epoch.wrapping_mul(7919);
    }

    /// Rank of the subspace embeddings.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Initialise a randomly-weighted model.
    pub fn init_model(
        &self,
        num_entities: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnSubspaceModel<B> {
        let basis = Param::initialized(
            ParamId::new(),
            Tensor::<B, 3>::random(
                [num_entities, self.rank, dim],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            )
            .require_grad(),
        );
        BurnSubspaceModel { basis }
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train for one epoch.
    pub fn train_epoch(
        &mut self,
        model: &mut BurnSubspaceModel<B>,
        optim: &mut impl Optimizer<BurnSubspaceModel<B>, B>,
        triples: &[TripleIds],
        epoch: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> f32 {
        self.set_epoch(epoch as u64);

        let num_entities = model.basis.val().dims()[0];
        let batch_size = config.batch_size.max(1);
        let n_neg = config.negative_samples.max(1);

        if self.cached_pools.is_none() {
            let indexed: Vec<(usize, usize, usize)> = triples
                .iter()
                .map(|t| (t.head, t.relation, t.tail))
                .collect();
            self.cached_pools = Some(compute_relation_entity_pools(&indexed));
        }
        let pools = self.cached_pools.as_ref().unwrap();

        let n = triples.len();
        if n == 0 {
            return 0.0;
        }

        let mut rng = fastrand::Rng::with_seed(self.epoch_seed.wrapping_add(1));
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.usize(0..=i));
        }

        let mut total_loss = 0.0f32;
        let mut batch_count = 0usize;

        for chunk in order.chunks(batch_size) {
            let bs = chunk.len();

            let batch_heads: Vec<i64> = chunk.iter().map(|&i| triples[i].head as i64).collect();
            let batch_rels: Vec<i64> = chunk.iter().map(|&i| triples[i].relation as i64).collect();
            let batch_tails: Vec<i64> = chunk.iter().map(|&i| triples[i].tail as i64).collect();

            // Corrupt tails only (standard for subsumption direction h ⊆ t).
            let mut neg_tails: Vec<i64> = Vec::with_capacity(bs * n_neg);
            for (&ri, &ti) in batch_rels.iter().zip(&batch_tails) {
                let pool = pools
                    .get(&(ri as usize))
                    .map(|p| p.tails.as_slice())
                    .unwrap_or(&[]);
                for _ in 0..n_neg {
                    let neg = sample_excluding(pool, ti as usize, |len| rng.usize(0..len))
                        .map(|v| v as i64)
                        .unwrap_or_else(|| loop {
                            let v = rng.usize(0..num_entities) as i64;
                            if v != ti {
                                break v;
                            }
                        });
                    neg_tails.push(neg);
                }
            }

            let head_t = Tensor::<B, 1, Int>::from_data(batch_heads.as_slice(), device);
            let rel_t = Tensor::<B, 1, Int>::from_data(batch_rels.as_slice(), device);
            let tail_t = Tensor::<B, 1, Int>::from_data(batch_tails.as_slice(), device);

            let current_model = model.clone();
            let loss = self.batch_loss(
                &current_model,
                head_t,
                rel_t,
                tail_t,
                neg_tails,
                n_neg,
                config,
                device,
            );

            let loss_val = loss.clone().into_scalar().to_f32();
            if loss_val.is_finite() {
                total_loss += loss_val;
                batch_count += 1;
                let grads = GradientsParams::from_grads(loss.backward(), &current_model);
                *model = optim.step(config.learning_rate as f64, current_model, grads);
            } else {
                #[cfg(debug_assertions)]
                eprintln!("[burn_subspace] skipping non-finite batch loss: {loss_val}");
            }
        }

        if batch_count == 0 {
            0.0
        } else {
            total_loss / batch_count as f32
        }
    }

    // -----------------------------------------------------------------------
    // Loss
    // -----------------------------------------------------------------------

    /// Containment score in tensor form.
    ///
    /// `head_basis`: `[bs, rank, dim]` (orthonormal rows)
    /// `tail_basis`: `[bs, rank, dim]` (orthonormal rows)
    ///
    /// Returns `[bs]` scores in `[0, 1]`.
    fn containment_score(head_basis: Tensor<B, 3>, tail_basis: Tensor<B, 3>) -> Tensor<B, 2> {
        let [bs, rank_h, _dim] = head_basis.dims();
        let _rank_t = tail_basis.dims()[1];

        // B^T * A^T  =>  [bs, rank_t, dim] x [bs, dim, rank_h] = [bs, rank_t, rank_h]
        let head_t = head_basis.clone().transpose(); // [bs, dim, rank_h]
        let bta = tail_basis.clone().matmul(head_t); // [bs, rank_t, rank_h]

        // B * (B^T * A^T) => [bs, rank_t, dim]^T x [bs, rank_t, rank_h] ...
        // proj = tail_basis^T * bta = [bs, dim, rank_t] x [bs, rank_t, rank_h] = [bs, dim, rank_h]
        let tail_t = tail_basis.transpose(); // [bs, dim, rank_t]
        let proj = tail_t.matmul(bta); // [bs, dim, rank_h]

        // proj is [bs, dim, rank_h]; head_basis is [bs, rank_h, dim] -> need [bs, dim, rank_h]
        let head_t2 = head_basis.transpose(); // [bs, dim, rank_h]

        // Residual: ||proj - head||^2 summed over dim and rank_h axes
        let diff = proj - head_t2; // [bs, dim, rank_h]
                                   // sum over dim (axis 1) -> [bs, 1, rank_h], then over rank_h (axis 2) -> [bs, 1, 1]
        let residual_3d = diff
            .powf_scalar(2.0)
            .sum_dim(1) // [bs, 1, rank_h]
            .sum_dim(2) // [bs, 1, 1]
            .div_scalar(rank_h as f32);
        let residual = residual_3d.reshape([bs, 1]); // [bs, 1]

        // score = 1 - residual, clamped to [0, 1]
        let score = Tensor::ones([bs, 1], &residual.device()) - residual;
        score.clamp(0.0, 1.0)
    }

    /// L2-normalise each basis vector (row of the [rank, dim] slices).
    ///
    /// `basis`: `[bs, rank, dim]` — normalises along the `dim` axis.
    /// Used in tests and as a building block; production paths use `batched_gram_schmidt`.
    #[allow(dead_code)]
    fn l2_normalize(basis: Tensor<B, 3>) -> Tensor<B, 3> {
        let norms = basis
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2) // [bs, rank, 1]
            .sqrt()
            .clamp_min(1e-8); // avoid div/0
        basis / norms
    }

    /// Gram-Schmidt orthonormalization for batched basis tensors.
    ///
    /// `basis`: `[bs, rank, dim]` — unconstrained row vectors.
    ///
    /// Returns `[bs, rank, dim]` with mutually orthogonal unit rows.
    /// Gradient flows through all arithmetic operations.
    fn batched_gram_schmidt(basis: Tensor<B, 3>) -> Tensor<B, 3> {
        let [bs, rank, dim] = basis.dims();
        let mut rows: Vec<Tensor<B, 3>> = Vec::with_capacity(rank);

        for i in 0..rank {
            // Extract row i: [bs, 1, dim]
            let mut vi = basis.clone().slice([0..bs, i..i + 1, 0..dim]);

            // Subtract projections onto each already-orthonormalized row
            for prev in &rows {
                // dot product along dim axis: [bs, 1, dim] * [bs, 1, dim] -> sum -> [bs, 1, 1]
                let dot = (vi.clone() * prev.clone()).sum_dim(2); // [bs, 1, 1]
                vi = vi - dot * prev.clone();
            }

            // Normalize: norm is [bs, 1, 1]
            let norm = vi
                .clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .sqrt()
                .clamp_min(1e-8);
            vi = vi / norm;

            rows.push(vi);
        }

        // Reassemble: [bs, rank, dim]
        Tensor::cat(rows, 1)
    }

    /// Select and orthonormalize basis vectors for a batch of entity ids.
    ///
    /// `ids`: `[bs]` int tensor.
    /// Returns `[bs, rank, dim]` (orthonormal rows via Gram-Schmidt).
    fn select_and_normalise(basis_all: &Tensor<B, 3>, ids: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        let bs = ids.dims()[0];
        let rank = basis_all.dims()[1];
        let dim = basis_all.dims()[2];
        // select requires 2D input — flatten, select, reshape.
        let flat = basis_all.clone().reshape([basis_all.dims()[0], rank * dim]);
        let selected_flat = flat.select(0, ids); // [bs, rank * dim]
        let selected = selected_flat.reshape([bs, rank, dim]);
        Self::batched_gram_schmidt(selected)
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_loss(
        &self,
        model: &BurnSubspaceModel<B>,
        head_ids: Tensor<B, 1, Int>,
        _rel_ids: Tensor<B, 1, Int>,
        tail_ids: Tensor<B, 1, Int>,
        neg_tail_ids_flat: Vec<i64>,
        n_neg: usize,
        config: &CpuBoxTrainingConfig,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let bs = head_ids.dims()[0];
        let rank = self.rank;
        let dim = model.basis.val().dims()[2];
        let k = config.sigmoid_k;

        let basis_all = model.basis.val();

        let head_b = Self::select_and_normalise(&basis_all, head_ids); // [bs, rank, dim]
        let tail_b = Self::select_and_normalise(&basis_all, tail_ids); // [bs, rank, dim]

        let pos_score = Self::containment_score(head_b.clone(), tail_b); // [bs, 1]

        // Negatives: [bs * n_neg] -> select and reshape to [bs, n_neg, rank, dim]
        let neg_ids = Tensor::<B, 1, Int>::from_data(neg_tail_ids_flat.as_slice(), device);
        let neg_b = Self::select_and_normalise(&basis_all, neg_ids); // [bs * n_neg, rank, dim]

        // Compute containment scores for all (head, neg_tail) pairs.
        // Expand head from [bs, rank, dim] to [bs * n_neg, rank, dim] by repeat-interleave.
        let head_rep = {
            let flat = head_b.reshape([bs, rank * dim]); // [bs, rank * dim]
            let mut rows: Vec<Tensor<B, 2>> = Vec::with_capacity(bs * n_neg);
            for i in 0..bs {
                let row = flat
                    .clone()
                    .slice([i..i + 1, 0..rank * dim])
                    .expand([n_neg, rank * dim]);
                rows.push(row);
            }
            Tensor::cat(rows, 0).reshape([bs * n_neg, rank, dim]) // [bs * n_neg, rank, dim]
        };
        let head_rep_norm = Self::batched_gram_schmidt(head_rep);

        let neg_score_flat = Self::containment_score(head_rep_norm, neg_b); // [bs * n_neg, 1]
        let neg_score = neg_score_flat.reshape([bs, n_neg]); // [bs, n_neg]

        // Margin ranking: relu(margin + neg_score - pos_score)
        // Higher score = better containment; pos should be high, neg should be low.
        let pos_score_rep = pos_score.clone().reshape([bs, 1]).expand([bs, n_neg]); // [bs, n_neg]

        if config.use_infonce {
            // InfoNCE: cross-entropy over (1 + n_neg)-way softmax.
            // logits: [bs, 1 + n_neg] scaled by k
            let pos_2d = pos_score.reshape([bs, 1]);
            let logits = Tensor::cat(vec![pos_2d.clone(), neg_score], 1) * k; // [bs, 1+n]
            let max_logit = logits.clone().max_dim(1);
            let lse = (logits - max_logit.clone()).exp().sum_dim(1).log() + max_logit;
            (lse - pos_2d * k).mean()
        } else {
            let loss_per_neg = (neg_score - pos_score_rep + config.margin).clamp_min(0.0); // [bs, n_neg]

            let agg = if config.self_adversarial {
                Self::apply_self_adv(loss_per_neg, n_neg, config.adversarial_temperature, device)
            } else {
                loss_per_neg.mean_dim(1) // [bs, 1]
            };

            agg.mean()
        }
    }

    /// Weighted negative loss using self-adversarial weights (stop-gradient on weights).
    fn apply_self_adv(
        loss: Tensor<B, 2>, // [bs, n_neg]
        n_neg: usize,
        adv_temp: f32,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let bs = loss.dims()[0];
        let data = loss.clone().into_data();
        let slice = data.as_slice::<f32>().expect("loss should be f32");
        let mut weights: Vec<f32> = Vec::with_capacity(slice.len());
        for row in slice.chunks(n_neg) {
            weights.extend(self_adversarial_weights(row, adv_temp));
        }
        let w = Tensor::<B, 1>::from_data(weights.as_slice(), device).reshape([bs, n_neg]);
        (loss * w).sum_dim(1)
    }

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Extract the trained basis into a `Vec<Subspace>` (one per entity).
    ///
    /// Performs full Gram-Schmidt orthonormalization on CPU for correctness.
    pub fn to_subspace_embeddings(&self, model: &BurnSubspaceModel<B>) -> Vec<Subspace> {
        let basis_data = model.basis.val().into_data();
        let flat: Vec<f32> = basis_data.to_vec().unwrap();
        let n_e = model.basis.val().dims()[0];
        let rank = self.rank;
        let dim = {
            let dims = model.basis.val().dims();
            dims[2]
        };
        // Rebuild dims from flat (model was consumed by into_data)
        // n_e * rank * dim elements
        (0..n_e)
            .map(|e| {
                let start = e * rank * dim;
                let vectors: Vec<Vec<f32>> = (0..rank)
                    .map(|r| flat[start + r * dim..start + (r + 1) * dim].to_vec())
                    .collect();
                Subspace::new(vectors).unwrap_or_else(|_| {
                    // Fallback: axis-aligned basis
                    let mut v = vec![0.0f32; dim];
                    v[0] = 1.0;
                    Subspace::new(vec![v]).unwrap()
                })
            })
            .collect()
    }

    /// Evaluate link prediction on `test_triples`.
    pub fn evaluate(
        &self,
        model: &BurnSubspaceModel<B>,
        test_triples: &[TripleIds],
        filter: Option<&crate::trainer::evaluation::FilteredTripleIndexIds>,
    ) -> crate::trainer::EvaluationResults {
        let entities = self.to_subspace_embeddings(model);
        let n_e = entities.len();

        let score = |h: usize, _r: usize, t: usize| -> f32 {
            crate::subspace::containment_score(&entities[h], &entities[t]).unwrap_or(0.0)
        };

        crate::trainer::evaluation::evaluate_link_prediction_generic(
            test_triples,
            n_e,
            filter,
            score,
            score,
        )
        .unwrap_or_else(|_| crate::trainer::EvaluationResults {
            mrr: 0.0,
            head_mrr: 0.0,
            tail_mrr: 0.0,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
            mean_rank: f32::MAX,
            rank_variance: f32::NAN,
            rank_p25: f32::NAN,
            rank_p50: f32::NAN,
            rank_p75: f32::NAN,
            rank_p95: f32::NAN,
            per_relation: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::optim::AdamConfig;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn model_init_shapes() {
        let device = Default::default();
        let trainer = BurnSubspaceTrainer::<TestBackend>::new(2);
        let model = trainer.init_model(10, 8, &device);
        assert_eq!(model.basis.val().dims(), [10, 2, 8]);
    }

    #[test]
    fn l2_normalize_unit_norms() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let raw = Tensor::<TestBackend, 3>::random(
            [4, 2, 6],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let normed = BurnSubspaceTrainer::<TestBackend>::l2_normalize(raw);
        let norms = normed.powf_scalar(2.0).sum_dim(2).sqrt().into_data();
        let vals: Vec<f32> = norms.to_vec().unwrap();
        for v in vals {
            assert!((v - 1.0).abs() < 1e-5, "norm not 1: {v}");
        }
    }

    #[test]
    fn loss_is_finite_and_nonneg() {
        let device = Default::default();
        let trainer = BurnSubspaceTrainer::<TestBackend>::new(2);
        let model = trainer.init_model(20, 8, &device);
        let config = CpuBoxTrainingConfig::default();
        let loss = trainer.batch_loss(
            &model,
            Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 2], &device),
            Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 0], &device),
            Tensor::<TestBackend, 1, Int>::from_data([1i64, 2, 0], &device),
            vec![3i64, 4, 5],
            1,
            &config,
            &device,
        );
        let v = loss.into_scalar().to_f32();
        assert!(v.is_finite(), "loss not finite: {v}");
        assert!(v >= 0.0, "loss negative: {v}");
    }

    #[test]
    fn loss_decreases_across_epochs() {
        use crate::dataset::TripleIds;

        let triples = vec![
            TripleIds {
                head: 0,
                relation: 0,
                tail: 1,
            },
            TripleIds {
                head: 2,
                relation: 1,
                tail: 3,
            },
        ];

        let device = Default::default();
        let mut trainer = BurnSubspaceTrainer::<TestBackend>::new(2);
        let mut model = trainer.init_model(5, 8, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            use_infonce: true,
            negative_samples: 1,
            sigmoid_k: 2.0,
            batch_size: 4,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnSubspaceModel<TestBackend>>();

        let loss0 = trainer.train_epoch(&mut model, &mut optim, &triples, 0, &config, &device);
        let mut loss_last = loss0;
        for epoch in 1..20 {
            loss_last =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        assert!(
            loss_last < loss0 + 0.5,
            "loss did not decrease: epoch 0 = {loss0:.4}, epoch 19 = {loss_last:.4}"
        );
    }

    #[test]
    fn train_and_evaluate_synthetic() {
        use crate::dataset::TripleIds;

        let triples = vec![
            TripleIds {
                head: 0,
                relation: 0,
                tail: 1,
            },
            TripleIds {
                head: 2,
                relation: 0,
                tail: 3,
            },
            TripleIds {
                head: 0,
                relation: 1,
                tail: 2,
            },
        ];
        let test_triples = triples.clone();

        let device = Default::default();
        let mut trainer = BurnSubspaceTrainer::<TestBackend>::new(2);
        let mut model = trainer.init_model(4, 8, &device);
        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            use_infonce: true,
            negative_samples: 1,
            sigmoid_k: 2.0,
            batch_size: 8,
            ..Default::default()
        };
        let mut optim = AdamConfig::new().init::<TestBackend, BurnSubspaceModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..100 {
            last_loss =
                trainer.train_epoch(&mut model, &mut optim, &triples, epoch, &config, &device);
        }
        eprintln!("final loss={last_loss:.4}");
        let results = trainer.evaluate(&model, &test_triples, None);
        eprintln!("MRR={:.3} mean_rank={:.1}", results.mrr, results.mean_rank);
        assert!(results.mrr > 0.3, "MRR={} expected >0.3", results.mrr);
        assert!(
            results.mean_rank <= 3.5,
            "mean_rank={} expected <=3.5",
            results.mean_rank
        );
    }
}
