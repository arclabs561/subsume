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
//! - **Box Embeddings**: Axis-aligned hyperrectangles that encode containment relationships.
//!   Unlike vector embeddings, boxes naturally model hierarchical structures through geometric
//!   containment. The volume of a box represents the "scope" or "granularity" of the concept.
//!
//! - **Gumbel Boxes**: Probabilistic boxes that solve the **local identifiability problem**.
//!   By modeling coordinates as Gumbel random variables, Gumbel boxes ensure that every parameter
//!   contributes to the loss, providing dense gradients throughout training. This is essential
//!   for learning, as hard boxes create "flat regions" where gradients vanish.
//!
//! - **Containment/Subsumption**: One box "subsumes" another when it contains it
//!   (entailment, hierarchical relationships). The term "subsume" comes from formal logic,
//!   where subsumption means one statement is more general than another. If box A contains
//!   box B (B ⊆ A), then A subsumes B — the more general concept contains the more specific one.
//!
//! - **Overlap**: The probability that two boxes have non-empty intersection. This measures
//!   whether two boxes represent related but distinct entities (e.g., "dog" and "cat" are
//!   both animals but distinct species). High overlap suggests relatedness; low overlap
//!   suggests disjointness or mutual exclusivity.
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
pub mod distance;
pub mod embedding;
pub mod gumbel;
pub mod training;
pub mod utils;

pub use box_trait::{Box, BoxError};
pub use distance::{boundary_distance, depth_distance, depth_similarity, vector_to_box_distance};
pub use embedding::{BoxCollection, BoxEmbedding};
pub use gumbel::GumbelBox;
pub use training::{
    calibration::{
        adaptive_calibration_error, brier_score, expected_calibration_error, reliability_diagram,
        ReliabilityDiagram,
    },
    diagnostics::{
        DepthStratifiedGradientFlow, GradientFlowAnalysis, LossComponents, PhaseDetector,
        RelationStratifiedTrainingStats, TrainingPhase, TrainingStats,
    },
    metrics::{
        hits_at_k, mean_rank, mean_reciprocal_rank, ndcg, DepthMetrics, FrequencyMetrics,
        RelationMetrics, StratifiedMetrics,
    },
    quality::{
        kl_divergence, AsymmetryMetrics, ContainmentAccuracy, ContainmentHierarchy,
        DimensionalityUtilization, GeneralizationMetrics, IntersectionTopology,
        TopologicalStability, VolumeConservation, VolumeDistribution,
    },
};
pub use utils::validation;
pub use utils::{
    clamp_temperature, clamp_temperature_default, gumbel_membership_prob, is_cross_pattern,
    is_perfectly_nested, log_space_volume, map_gumbel_to_bounds, safe_init_bounds, sample_gumbel,
    stable_sigmoid, suggested_min_separation, temperature_scheduler, volume_containment_loss,
    volume_overlap_loss, volume_regularization, MAX_TEMPERATURE, MIN_TEMPERATURE,
};
