//! # subsume
//!
//! Geometric box embeddings for modeling containment, entailment, and overlap.
//!
//! `subsume` provides framework-agnostic traits and concrete backends for
//! **box embeddings** -- axis-aligned hyperrectangles in d-dimensional space that
//! encode hierarchical relationships through geometric containment. If box A
//! contains box B (B ⊆ A), then A *subsumes* B: the more general concept
//! contains the more specific one.
//!
//! # Getting Started
//!
//! | Goal | Start here |
//! |------|-----------|
//! | Understand the core abstraction | [`Box`] trait, [`BoxError`] |
//! | Use probabilistic (Gumbel) boxes | [`GumbelBox`] trait, [`gumbel`] module |
//! | Use octagon embeddings (box + diagonal constraints) | [`Octagon`] trait, [`octagon`] module |
//! | Fuzzy query answering (t-norms) | [`TNorm`], [`TConorm`], [`fuzzy`] module |
//! | Load a knowledge graph dataset | [`Dataset`], [`Triple`] |
//! | Train box embeddings (ndarray) | [`ndarray_backend`], [`TrainingConfig`] |
//! | Evaluate with link prediction | [`evaluate_link_prediction`], [`training::metrics`] |
//!
//! # Key Concepts
//!
//! **Box embeddings** represent concepts as hyperrectangles. Unlike point vectors,
//! boxes have volume, which encodes generality: a broad concept ("animal") is a
//! large box containing smaller boxes ("dog", "cat").
//!
//! **Gumbel boxes** solve the *local identifiability problem* of hard boxes by
//! modeling coordinates as Gumbel random variables. This ensures dense gradients
//! throughout training -- hard boxes create flat regions where gradients vanish.
//!
//! **Containment probability** measures entailment (P(B ⊆ A)), while **overlap
//! probability** measures relatedness without strict hierarchy. These two scores
//! are the primary outputs of box embedding models.
//!
//! # Module Organization
//!
//! ## Core traits and geometry
//!
//! - [`box_trait`] -- the [`Box`] trait: containment, overlap, volume
//! - [`gumbel`] -- the [`GumbelBox`] trait: probabilistic box operations
//! - [`octagon`] -- octagon embeddings: boxes + diagonal constraints (IJCAI 2024)
//! - [`cone`] -- angular cone embeddings for subsumption with negation
//! - [`hyperbolic`] -- Poincare ball embeddings for tree-like hierarchies
//! - [`hyperbolic_box`] -- boxes in hyperbolic space (high-capacity hierarchies)
//! - [`quasimetric`] -- interval quasimetric embeddings (asymmetric reachability)
//! - [`sheaf`] -- sheaf neural networks for transitivity/consistency on graphs
//! - [`gaussian`] -- diagonal Gaussian box embeddings (KL, Bhattacharyya)
//!
//! ## Representations and scoring
//!
//! - [`center_offset`] -- center+offset <-> min/max coordinate conversion
//! - [`boxe`] -- BoxE scoring model (Abboud et al., NeurIPS 2020)
//! - [`distance`] -- depth-based, boundary, Query2Box, and vector-to-box distance metrics
//! - [`embedding`] -- [`BoxCollection`] and [`BoxEmbedding`] collection traits
//! - [`fuzzy`] -- t-norms, t-conorms, and negation for fuzzy query answering (FuzzQE)
//!
//! ## Ontology and taxonomy
//!
//! - [`el`] -- EL++ ontology embedding primitives (Box2EL / TransBox)
//! - [`taxonomy`] -- TaxoBell-format taxonomy dataset loader
//! - [`taxobell`] -- TaxoBell combined training loss
//!
//! ## Training and evaluation
//!
//! - [`dataset`] -- load WN18RR, FB15k-237, YAGO3-10, and similar KG datasets
//! - [`trainable`] -- [`TrainableBox`] and [`TrainableCone`] with learnable parameters
//! - [`trainer`] -- negative sampling, loss computation, link prediction evaluation
//! - [`training`] -- metrics (MRR, Hits@k), calibration, diagnostics, quality analysis
//! - [`optimizer`] -- AMSGrad state management
//! - [`utils`] -- numerical stability (log-space volume, stable sigmoid, temperature)
//!
//! ## Backends (feature-gated)
//!
//! - [`ndarray_backend`] -- `NdarrayBox`, `NdarrayGumbelBox`, optimizer, scheduler
//!   (feature = `ndarray-backend`, **on by default**)
//! - `candle_backend` -- `CandleBox`, `CandleGumbelBox` with GPU support
//!   (feature = `candle-backend`)
//!
//! # Feature Flags
//!
//! | Feature | Default | Provides |
//! |---------|---------|----------|
//! | `ndarray-backend` | yes | [`ndarray_backend`] module, enables `rand` |
//! | `candle-backend` | no | `candle_backend` module (GPU via candle) |
//! | `rand` | no | Negative sampling utilities in [`trainer`] |
//! | `plotting` | no | Reserved for visualization support |
//!
//! # Example
//!
//! ```rust,ignore
//! use subsume::{Box, GumbelBox};
//!
//! // Framework-agnostic: works with NdarrayBox, CandleBox, or your own impl
//! fn compute_entailment<B: Box>(
//!     premise: &B,
//!     hypothesis: &B,
//!     temp: B::Scalar,
//! ) -> Result<B::Scalar, subsume::BoxError> {
//!     premise.containment_prob(hypothesis, temp)
//! }
//! ```
//!
//! # References
//!
//! - Vilnis et al. (2018), "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
//! - Abboud et al. (2020), "BoxE: A Box Embedding Model for Knowledge Base Completion"
//! - Dasgupta et al. (2020), "Improving Local Identifiability in Probabilistic Box Embeddings"
//! - Chen et al. (2021), "Uncertainty-Aware Knowledge Graph Embeddings" (UKGE)
//! - Lee et al. (2022), "Box Embeddings for Event-Event Relation Extraction" (BERE)
//! - Cao et al. (2024, ACM Computing Surveys), "KG Embedding: A Survey from the
//!   Perspective of Representation Spaces" -- positions box/cone/octagon embeddings
//!   within the broader KGE taxonomy (Euclidean, hyperbolic, complex, geometric)
//! - Yang & Chen (2025), "RegD: Achieving Hyperbolic-Like Expressiveness with Arbitrary
//!   Euclidean Regions" -- source of the depth/boundary dissimilarity metrics in [`distance`]

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// ---------------------------------------------------------------------------
// Core traits and geometry
// ---------------------------------------------------------------------------

/// Core [`Box`] trait: containment probability, overlap, volume, and intersection.
pub mod box_trait;

/// BoxE scoring model for knowledge base completion (Abboud et al., NeurIPS 2020).
pub mod boxe;

/// Center-offset <-> min/max coordinate conversion for box parameterization.
pub mod center_offset;

/// Cone embeddings: angular containment on the unit sphere, with negation support.
pub mod cone;

/// Octagon embeddings: axis-aligned polytopes with diagonal constraints (IJCAI 2024).
pub mod octagon;

/// Knowledge graph dataset loading (WN18RR, FB15k-237, YAGO3-10, and similar formats).
pub mod dataset;

/// Distance metrics: depth-based (RegD), boundary, and vector-to-box distances.
pub mod distance;

/// Collection traits for managing batches of box embeddings.
pub mod embedding;

/// [`GumbelBox`] trait: probabilistic boxes with Gumbel-distributed coordinates.
pub mod gumbel;

/// Poincare ball embeddings for tree-like hierarchical structures.
pub mod hyperbolic;

/// Box embeddings in hyperbolic space for high-capacity hierarchy modeling.
pub mod hyperbolic_box;

/// AMSGrad optimizer state and learning rate utilities.
pub mod optimizer;

/// Interval quasimetric embeddings (IQE) for asymmetric reachability.
pub mod quasimetric;

/// Sheaf neural networks: algebraic consistency enforcement on graphs.
pub mod sheaf;

/// Learnable box and cone representations with gradient-compatible parameters.
pub mod trainable;

/// Training loop utilities: negative sampling, loss kernels, link prediction evaluation.
pub mod trainer;

/// Evaluation and diagnostics: rank metrics, calibration, gradient flow, quality analysis.
pub mod training;

/// Numerical stability: log-space volume, stable sigmoid, temperature scheduling.
pub mod utils;

/// Diagonal Gaussian box embeddings for taxonomy expansion (TaxoBell).
pub mod gaussian;

/// EL++ ontology embedding primitives (Box2EL / TransBox).
pub mod el;

/// Fuzzy set-theoretic operators: t-norms, t-conorms, and negation (FuzzQE).
pub mod fuzzy;

/// Taxonomy dataset loading for the TaxoBell format (`.terms` / `.taxo` / `dic.json`).
pub mod taxonomy;

/// TaxoBell combined training loss for taxonomy expansion.
pub mod taxobell;

// ---------------------------------------------------------------------------
// Re-exports: primary traits and types
// ---------------------------------------------------------------------------

/// The core box embedding trait. Start here.
pub use box_trait::{Box, BoxError};

/// The probabilistic box trait (Gumbel-distributed coordinates).
pub use gumbel::GumbelBox;

// Re-exports: geometry variants
pub use cone::{Cone, ConeError};
pub use hyperbolic::{
    hierarchy_preserved, pairwise_distances, Curvature, HyperbolicError, HyperbolicPoint,
    PoincareBallPoint,
};

// Re-exports: data loading
pub use dataset::{Dataset, DatasetError, DatasetStats, Triple};

// Re-exports: representations and scoring
pub use boxe::{boxe_loss, boxe_point_score, boxe_score, Bump};
pub use center_offset::{center_offset_to_min_max, min_max_to_center_offset};
pub use distance::{
    boundary_distance, depth_distance, depth_similarity, query2box_distance,
    vector_to_box_distance,
};
pub use embedding::{BoxCollection, BoxEmbedding};

// Re-exports: training
pub use optimizer::{get_learning_rate, AMSGradState};
pub use trainable::{TrainableBox, TrainableCone};
pub use trainer::{
    compute_cone_analytical_gradients, compute_cone_pair_loss, evaluate_link_prediction,
    log_training_result, ConeEmbeddingTrainer, EvaluationResults, HyperparameterSearch,
    NegativeSamplingStrategy, TrainingConfig, TrainingResult,
};

/// Negative sampling utilities (requires the `rand` feature).
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
pub use trainer::{
    generate_negative_samples, generate_negative_samples_from_pool_with_rng,
    generate_negative_samples_from_sorted_pool_with_rng, generate_negative_samples_with_rng,
    SortedEntityPool,
};

// Re-exports: evaluation and quality
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

// Re-exports: sheaf
pub use sheaf::{
    consistency_score, diffuse_until_convergence, DenseRestriction, DiffusionConfig, LaplacianType,
    RestrictionMap, SheafBuilder, SheafEdge, SheafError, SheafGraph, SimpleSheafGraph, Stalk,
    VecStalk,
};

// Re-exports: Gaussian boxes
pub use gaussian::{
    bhattacharyya_coefficient, bhattacharyya_distance,
    kl_divergence as gaussian_kl_divergence, sigma_ceiling_loss, sigma_clipping_loss,
    volume_regularization as gaussian_volume_regularization, GaussianBox,
};

// Re-exports: taxonomy
pub use taxonomy::{TaxonomyDataset, TaxonomyNode};

// Re-exports: TaxoBell loss
pub use taxobell::{CombinedLossResult, TaxoBellConfig, TaxoBellLoss};

// Re-exports: EL++ ontology
pub use el::{
    compose_roles, disjointness_loss, el_inclusion_loss, existential_box,
    intersection_nonempty_loss, translate,
};

// Re-exports: fuzzy operators
pub use fuzzy::{
    fuzzy_negation, tconorm_lukasiewicz, tconorm_max, tconorm_probabilistic, tnorm_lukasiewicz,
    tnorm_min, tnorm_product, TConorm, TNorm,
};

// Re-exports: octagon
pub use octagon::{DiagonalBounds, Octagon, OctagonError};

// Re-exports: utilities
pub use utils::validation;
pub use utils::{
    bessel_log_volume, bessel_side_length, clamp_temperature, clamp_temperature_default,
    gumbel_lse_max, gumbel_lse_min, gumbel_membership_prob, is_cross_pattern,
    is_perfectly_nested, log_space_volume, map_gumbel_to_bounds, safe_init_bounds, sample_gumbel,
    softplus, stable_logsumexp, stable_sigmoid, suggested_min_separation, temperature_scheduler,
    volume_containment_loss, volume_overlap_loss, volume_regularization, EULER_GAMMA,
    MAX_TEMPERATURE, MIN_TEMPERATURE,
};

// ---------------------------------------------------------------------------
// Feature-gated backends
// ---------------------------------------------------------------------------

/// Ndarray backend: `NdarrayBox`, `NdarrayGumbelBox`, optimizer, and learning rate scheduler.
///
/// This is the default backend. Enable with `features = ["ndarray-backend"]`.
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
#[cfg(feature = "ndarray-backend")]
pub mod ndarray_backend;

/// Candle backend: `CandleBox`, `CandleGumbelBox` with GPU acceleration.
///
/// Enable with `features = ["candle-backend"]`.
#[cfg_attr(docsrs, doc(cfg(feature = "candle-backend")))]
#[cfg(feature = "candle-backend")]
pub mod candle_backend;
