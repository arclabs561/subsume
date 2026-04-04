//! Burn-based ball embedding trainer (work-in-progress).
//!
//! This module is a scaffold for the burn migration. The full training loop
//! requires burn 0.20 API adjustments. The geometry types (Ball, BallRelation)
//! are already burn-ready since they use `Vec<f32>` and serde serialization.
//!
//! # Migration path
//! 1. Update burn to latest version (API changes frequently)
//! 2. Use `burn_ndarray::NdArray<f32, i64>` with correct type parameters
//! 3. Use `AdamConfig::new().init()` for optimizer initialization
//! 4. Clone tensors before use in `select()` (burn 0.20 consumes on select)
//! 5. Wire into the existing training pipeline

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

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

/// Burn-based ball trainer (work-in-progress).
pub struct BurnBallTrainer<B: AutodiffBackend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> BurnBallTrainer<B> {
    pub fn new() -> Self {
        Self { _backend: std::marker::PhantomData }
    }

    /// Initialize entity and relation parameters.
    pub fn init_params(
        &self,
        num_entities: usize,
        num_relations: usize,
        dim: usize,
        device: &B::Device,
    ) -> (BurnBallEntityParams<B>, BurnBallRelationParams<B>) {
        (
            BurnBallEntityParams {
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
            BurnBallRelationParams {
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
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn burn_params_init() {
        let device = Default::default();
        let trainer = BurnBallTrainer::<TestBackend>::new();
        let (entities, relations) = trainer.init_params(10, 3, 4, &device);
        assert_eq!(entities.center.dims(), [10, 4]);
        assert_eq!(entities.log_radius.dims(), [10, 1]);
        assert_eq!(relations.translation.dims(), [3, 4]);
        assert_eq!(relations.log_scale.dims(), [3, 1]);
    }
}
