//! # subsume-core
//!
//! Framework-agnostic traits for geometric box embeddings.
//!
//! This crate defines the core abstractions for box embeddings, hypercube embeddings,
//! and other geometric representations used for modeling containment, entailment, and
//! hierarchical relationships in NLP and knowledge graphs.
//!
//! # Key Concepts
//!
//! - **Box Embeddings**: Axis-aligned hyperrectangles that encode containment relationships
//! - **Gumbel Boxes**: Probabilistic boxes with Gumbel noise for training stability
//! - **Containment**: One box "subsumes" another (entailment, hierarchical relationships)
//! - **Overlap**: Mutual probability of two boxes representing the same entity
//!
//! # Design Philosophy
//!
//! This crate is **framework-agnostic**. It defines traits that can be implemented
//! by any tensor/array library:
//!
//! - `subsume-candle`: Implementation using `candle_core::Tensor`
//! - `subsume-ndarray`: Implementation using `ndarray::Array1`
//! - Future: `subsume-tch` (PyTorch), `subsume-burn`, etc.
//!
//! # Example
//!
//! ```rust,ignore
//! use subsume_core::{Box, GumbelBox};
//!
//! // Framework-agnostic box operations
//! fn compute_entailment<B: Box>(premise: &B, hypothesis: &B, temp: B::Scalar) -> Result<B::Scalar, subsume_core::BoxError> {
//!     premise.containment_prob(hypothesis, temp)
//! }
//! ```
//!
//! # Research Background
//!
//! Based on:
//! - Vilnis et al. (2018): "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
//! - Lee et al. (2022): "Box Embeddings for Event-Event Relation Extraction" (BERE)
//! - Messner et al. (2022): "Temporal Knowledge Graph Completion with Box Embeddings" (BoxTE)
//! - Chen et al. (2021): "Uncertainty-Aware Knowledge Graph Embeddings" (UKGE)

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod box_trait;
pub mod gumbel;
pub mod embedding;
pub mod utils;
pub mod training;

pub use box_trait::{Box, BoxError};
pub use gumbel::GumbelBox;
pub use embedding::{BoxEmbedding, BoxCollection};
pub use utils::{
    clamp_temperature, clamp_temperature_default,
    MIN_TEMPERATURE, MAX_TEMPERATURE,
    stable_sigmoid, gumbel_membership_prob,
    sample_gumbel, map_gumbel_to_bounds,
    log_space_volume, volume_regularization,
    temperature_scheduler, volume_containment_loss,
    volume_overlap_loss,
};
pub use training::{
    metrics::{mean_reciprocal_rank, hits_at_k, mean_rank, ndcg},
    diagnostics::{TrainingStats, LossComponents},
    quality::{VolumeDistribution, ContainmentAccuracy, IntersectionTopology},
    calibration::{expected_calibration_error, brier_score},
};

