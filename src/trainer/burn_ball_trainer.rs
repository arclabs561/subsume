//! Burn-based ball embedding trainer with autodiff.
//!
//! This is the first burn-based trainer in subsume, serving as the template
//! for migrating all other trainers from candle/ndarray to burn.
//!
//! # Backend selection
//! - `burn-ndarray` — CPU training (default)
//! - `burn-wgpu` — GPU training on AMD/Intel/WebGPU
//! - `burn-tch` — GPU training via PyTorch (CUDA)

use crate::ball::{Ball, BallRelation};
use crate::dataset::Triple;
use crate::trainer::CpuBoxTrainingConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

/// Entity parameters for ball embeddings.
#[derive(Module, Debug)]
pub struct BurnBallEntityParams<B: Backend> {
    pub center: Tensor<B, 2>,
    pub log_radius: Tensor<B, 2>,
}

/// Relation parameters for ball embeddings.
#[derive(Module, Debug)]
pub struct BurnBallRelationParams<B: Backend> {
    pub translation: Tensor<B, 2>,
    pub log_scale: Tensor<B, 2>,
}

/// Combined model for burn training.
#[derive(Module, Debug)]
pub struct BurnBallModel<B: Backend> {
    pub entities: BurnBallEntityParams<B>,
    pub relations: BurnBallRelationParams<B>,
}

/// Burn-based ball trainer using autodiff.
pub struct BurnBallTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> BurnBallTrainer<B> {
    /// Create a new burn ball trainer.
    pub fn new() -> Self {
        Self {
            _backend: std::marker::PhantomData,
        }
    }

    /// Initialize the full model.
    pub fn init_model(
        &self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        device: &B::Device,
    ) -> BurnBallModel<B> {
        BurnBallModel {
            entities: BurnBallEntityParams {
                center: Tensor::<B, 2>::random(
                    [num_entities, dim],
                    burn::tensor::Distribution::Uniform(-0.1, 0.1),
                    device,
                ),
                log_radius: Tensor::<B, 2>::random(
                    [num_entities, 1],
                    burn::tensor::Distribution::Uniform(-1.0, 0.0),
                    device,
                ),
            },
            relations: BurnBallRelationParams {
                translation: Tensor::<B, 2>::random(
                    [num_relations, dim],
                    burn::tensor::Distribution::Uniform(-0.01, 0.01),
                    device,
                ),
                log_scale: Tensor::<B, 2>::random(
                    [num_relations, 1],
                    burn::tensor::Distribution::Uniform(-0.1, 0.1),
                    device,
                ),
            },
        }
    }

    /// Compute the ball embedding loss for a batch of triples.
    pub fn compute_loss(
        &self,
        model: BurnBallModel<B>,
        head_ids: Tensor<B, 1, Int>,
        rel_ids: Tensor<B, 1, Int>,
        tail_ids: Tensor<B, 1, Int>,
        neg_tail_ids: Tensor<B, 1, Int>,
        margin: f32,
        k: f32,
    ) -> Tensor<B, 1> {
        let head_center = model.entities.center.clone().select(0, head_ids.clone());
        let head_log_radius = model.entities.log_radius.clone().select(0, head_ids);
        let rel_translation = model
            .relations
            .translation
            .clone()
            .select(0, rel_ids.clone());
        let rel_log_scale = model.relations.log_scale.clone().select(0, rel_ids);
        let tail_center = model.entities.center.clone().select(0, tail_ids.clone());
        let tail_log_radius = model.entities.log_radius.clone().select(0, tail_ids);
        let neg_tail_center = model
            .entities
            .center
            .clone()
            .select(0, neg_tail_ids.clone());
        let neg_tail_log_radius = model.entities.log_radius.clone().select(0, neg_tail_ids);

        let transformed_center = head_center + rel_translation;
        let transformed_log_radius = head_log_radius + rel_log_scale;

        let pos_diff = transformed_center.clone() - tail_center.clone();
        let pos_dist = pos_diff.powf_scalar(2.0).sum_dim(1).sqrt();
        let pos_radius_inner = transformed_log_radius.clone().exp();
        let pos_radius_outer = tail_log_radius.clone().exp();
        let pos_margin = pos_radius_outer - pos_dist - pos_radius_inner;

        let neg_diff = transformed_center - neg_tail_center;
        let neg_dist = neg_diff.powf_scalar(2.0).sum_dim(1).sqrt();
        let neg_radius_inner = transformed_log_radius.exp();
        let neg_radius_outer = neg_tail_log_radius.exp();
        let neg_margin = neg_radius_outer - neg_dist - neg_radius_inner;

        let pos_prob = sigmoid(pos_margin * k);
        let neg_prob = sigmoid(neg_margin * k);

        // Loss = margin - ln(pos_prob) + ln(neg_prob), clamped to >= 0
        // Minimized when pos_prob is large and neg_prob is small
        let pos_loss = pos_prob.clamp(1e-6, 1.0 - 1e-6).log().neg();
        let neg_score = neg_prob.clamp(1e-6, 1.0 - 1e-6).log().neg();
        let ranking_loss = (margin + pos_loss - neg_score).clamp_min(0.0);

        ranking_loss.mean()
    }

    /// Train for one epoch on a set of triples.
    ///
    /// Pass in the optimizer to preserve state across epochs.
    pub fn train_epoch(
        &self,
        model: &mut BurnBallModel<B>,
        optim: &mut impl Optimizer<BurnBallModel<B>, B>,
        triples: &[Triple],
        config: &CpuBoxTrainingConfig,
        entity_to_idx: &HashMap<String, usize>,
        relation_to_idx: &HashMap<String, usize>,
        device: &B::Device,
    ) -> f32 {
        let num_entities = model.entities.center.dims()[0];
        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        // Simple RNG for negative sampling (seeded for reproducibility)
        let mut rng = fastrand::Rng::with_seed(42);

        for triple in triples {
            let head_idx = match entity_to_idx.get(&triple.head) {
                Some(&i) => i,
                None => continue,
            };
            let rel_idx = match relation_to_idx.get(&triple.relation) {
                Some(&i) => i,
                None => continue,
            };
            let tail_idx = match entity_to_idx.get(&triple.tail) {
                Some(&i) => i,
                None => continue,
            };

            let mut neg_tail_idx = rng.usize(0..num_entities);
            while neg_tail_idx == tail_idx {
                neg_tail_idx = rng.usize(0..num_entities);
            }

            let head_ids = Tensor::<B, 1, Int>::from_data([head_idx as i64], device);
            let rel_ids = Tensor::<B, 1, Int>::from_data([rel_idx as i64], device);
            let tail_ids = Tensor::<B, 1, Int>::from_data([tail_idx as i64], device);
            let neg_tail_ids = Tensor::<B, 1, Int>::from_data([neg_tail_idx as i64], device);

            let loss = self.compute_loss(
                model.clone(),
                head_ids,
                rel_ids,
                tail_ids,
                neg_tail_ids,
                config.margin,
                10.0,
            );

            let loss_val = loss.clone().into_scalar().to_f32();
            total_loss += loss_val;
            count += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            let new_model = optim.step(config.learning_rate as f64, model.clone(), grads);
            *model = new_model;
        }

        if count == 0 {
            0.0
        } else {
            total_loss / count as f32
        }
    }

    /// Convert burn model parameters back to Ball/BallRelation types.
    pub fn to_ball_embeddings(&self, model: &BurnBallModel<B>) -> (Vec<Ball>, Vec<BallRelation>) {
        let center_data = model.entities.center.clone().into_data();
        let log_radius_data = model.entities.log_radius.clone().into_data();
        let trans_data = model.relations.translation.clone().into_data();
        let log_scale_data = model.relations.log_scale.clone().into_data();

        let center_slice: &[f32] = center_data.as_slice::<f32>().expect("center should be f32");
        let log_radius_slice: &[f32] = log_radius_data
            .as_slice::<f32>()
            .expect("log_radius should be f32");
        let trans_slice: &[f32] = trans_data
            .as_slice::<f32>()
            .expect("translation should be f32");
        let log_scale_slice: &[f32] = log_scale_data
            .as_slice::<f32>()
            .expect("log_scale should be f32");

        let dims = model.entities.center.dims();
        let num_entities = dims[0];
        let dim = dims[1];
        let num_relations = model.relations.translation.dims()[0];

        let entities: Vec<Ball> = (0..num_entities)
            .map(|i| {
                let center: Vec<f32> = center_slice[i * dim..(i + 1) * dim].to_vec();
                let log_radius = log_radius_slice[i];
                Ball::from_log_radius(center, log_radius).unwrap()
            })
            .collect();

        let relations: Vec<BallRelation> = (0..num_relations)
            .map(|i| {
                let translation: Vec<f32> = trans_slice[i * dim..(i + 1) * dim].to_vec();
                let log_scale = log_scale_slice[i];
                BallRelation::new(translation, log_scale.exp()).unwrap()
            })
            .collect();

        (entities, relations)
    }
}

fn sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.neg().exp().recip()
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

    #[test]
    fn burn_model_init() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let model = trainer.init_model(10, 3, 4, &device);
        assert_eq!(model.entities.center.dims(), [10, 4]);
        assert_eq!(model.entities.log_radius.dims(), [10, 1]);
        assert_eq!(model.relations.translation.dims(), [3, 4]);
        assert_eq!(model.relations.log_scale.dims(), [3, 1]);
    }

    #[test]
    fn burn_loss_is_finite() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let model = trainer.init_model(20, 3, 4, &device);

        let head_ids = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 2], &device);
        let rel_ids = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 0], &device);
        let tail_ids = Tensor::<TestBackend, 1, Int>::from_data([1i64, 2, 0], &device);
        let neg_tail_ids = Tensor::<TestBackend, 1, Int>::from_data([3i64, 4, 5], &device);

        let loss =
            trainer.compute_loss(model, head_ids, rel_ids, tail_ids, neg_tail_ids, 1.0, 10.0);

        let loss_val = loss.into_scalar().to_f32();
        assert!(loss_val.is_finite(), "loss not finite: {loss_val}");
        assert!(loss_val >= 0.0, "loss negative: {loss_val}");
    }

    #[test]
    fn burn_training_step_reduces_loss() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(10, 3, 4, &device);

        let head_ids = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1], &device);
        let rel_ids = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1], &device);
        let tail_ids = Tensor::<TestBackend, 1, Int>::from_data([2i64, 3], &device);
        let neg_tail_ids = Tensor::<TestBackend, 1, Int>::from_data([4i64, 5], &device);

        let loss0 = trainer.compute_loss(
            model.clone(),
            head_ids.clone(),
            rel_ids.clone(),
            tail_ids.clone(),
            neg_tail_ids.clone(),
            1.0,
            10.0,
        );
        let loss0_val = loss0.clone().into_scalar().to_f32();

        // Gradient step
        let grads = loss0.backward();
        let mut optim = AdamConfig::new().init::<TestBackend, BurnBallModel<TestBackend>>();
        let grads = GradientsParams::from_grads(grads, &model);
        let new_model = optim.step(1e-2_f64, model, grads);
        model = new_model;

        let loss1 =
            trainer.compute_loss(model, head_ids, rel_ids, tail_ids, neg_tail_ids, 1.0, 10.0);
        let loss1_val = loss1.into_scalar().to_f32();

        assert!(
            loss1_val <= loss0_val + 0.5,
            "Loss increased significantly: {loss0_val} -> {loss1_val}"
        );
    }

    #[test]
    fn burn_to_ball_embeddings() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let model = trainer.init_model(5, 2, 4, &device);

        let (entities, relations) = trainer.to_ball_embeddings(&model);
        assert_eq!(entities.len(), 5);
        assert_eq!(relations.len(), 2);
        assert_eq!(entities[0].dim(), 4);
        assert!(entities[0].radius() > 0.0);
    }

    #[test]
    fn burn_train_and_evaluate_synthetic() {
        use crate::dataset::{TripleIds, Vocab};
        use crate::trainer::ball_trainer::BallTrainer;

        let mut vocab = Vocab::default();
        let e0 = vocab.intern("e0".to_string());
        let e1 = vocab.intern("e1".to_string());
        let e2 = vocab.intern("e2".to_string());
        let e3 = vocab.intern("e3".to_string());

        let triples = vec![
            Triple {
                head: "e0".to_string(),
                relation: "r0".to_string(),
                tail: "e1".to_string(),
            },
            Triple {
                head: "e2".to_string(),
                relation: "r0".to_string(),
                tail: "e3".to_string(),
            },
            Triple {
                head: "e0".to_string(),
                relation: "r1".to_string(),
                tail: "e2".to_string(),
            },
        ];

        let test_triples = vec![
            TripleIds {
                head: e0,
                relation: 0,
                tail: e1,
            },
            TripleIds {
                head: e2,
                relation: 0,
                tail: e3,
            },
            TripleIds {
                head: e0,
                relation: 1,
                tail: e2,
            },
        ];

        let entity_map: HashMap<String, usize> = [
            ("e0".to_string(), 0),
            ("e1".to_string(), 1),
            ("e2".to_string(), 2),
            ("e3".to_string(), 3),
        ]
        .into_iter()
        .collect();
        let relation_map: HashMap<String, usize> = [("r0".to_string(), 0), ("r1".to_string(), 1)]
            .into_iter()
            .collect();

        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let mut model = trainer.init_model(4, 2, 4, &device);

        let config = CpuBoxTrainingConfig {
            learning_rate: 0.05,
            margin: 1.0,
            ..Default::default()
        };

        let mut optim = AdamConfig::new().init::<TestBackend, BurnBallModel<TestBackend>>();

        let mut last_loss = f32::MAX;
        for epoch in 0..50 {
            let loss = trainer.train_epoch(
                &mut model,
                &mut optim,
                &triples,
                &config,
                &entity_map,
                &relation_map,
                &device,
            );
            if epoch % 10 == 0 {
                eprintln!("Burn Epoch {epoch}: loss={loss:.4}");
            }
            last_loss = loss;
        }
        eprintln!("Burn Final loss: {last_loss:.4}");

        // Convert to Ball/BallRelation and evaluate with the existing eval infra
        let (entities, relations) = trainer.to_ball_embeddings(&model);
        let ball_trainer = BallTrainer::new(42);
        let results = ball_trainer.evaluate(&entities, &relations, &test_triples, None);

        eprintln!(
            "Burn MRR: {}, Mean rank: {}",
            results.mrr, results.mean_rank
        );

        assert!(
            results.mrr > 0.3,
            "Burn MRR after training = {}, expected > 0.3",
            results.mrr
        );
        assert!(
            results.mean_rank <= 3.0,
            "Burn Mean rank = {}, expected <= 3.0",
            results.mean_rank
        );
    }
}
