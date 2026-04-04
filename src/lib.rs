//! # subsume
//!
//! Geometric region embeddings for subsumption, entailment, and logical query answering.
//!
//! `subsume` provides framework-agnostic traits and concrete backends for
//! geometric embeddings -- boxes, cones, octagons, Gaussians, and hyperbolic
//! intervals -- that encode hierarchical relationships through geometric
//! containment. If region A contains region B (B ⊆ A), then A *subsumes* B:
//! the more general concept contains the more specific one.
//!
//! # Getting Started
//!
//! | Goal | Start here |
//! |------|-----------|
//! | Understand the core abstraction | [`Box`] trait, [`BoxError`] |
//! | Use probabilistic (Gumbel) boxes | [`NdarrayGumbelBox`](ndarray_backend::NdarrayGumbelBox) |
//! | Use octagon embeddings (box + diagonal constraints) | [`NdarrayOctagon`](ndarray_backend::ndarray_octagon::NdarrayOctagon), [`octagon`] module |
//! | Fuzzy query answering (t-norms) | [`fuzzy::TNorm`], [`fuzzy::TConorm`], [`fuzzy`] module |
//! | Load a knowledge graph dataset | [`Dataset`], [`Triple`] |
//! | Train box embeddings (CPU) | [`BoxEmbeddingTrainer`], [`TrainingConfig`] |
//! | Train box embeddings (GPU) | [`CandleBoxTrainer`] (feature = `candle-backend`) |
//! | Evaluate with link prediction | [`evaluate_link_prediction`], [`CandleBoxTrainer::evaluate`](trainer::candle_trainer::CandleBoxTrainer::evaluate) |
//!
//! # Why regions instead of points?
//!
//! Point embeddings (TransE, RotatE) work for link prediction but cannot encode
//! containment, volume, or set operations. Regions become necessary when the task
//! requires:
//!
//! - **Subsumption**: box A inside box B means A is-a B
//! - **Generality**: large volume = broad concept, small volume = specific
//! - **Intersection**: combining two concepts (A ∧ B) yields a valid region
//! - **Negation**: cone complement is another cone (FOL queries with ¬)
//!
//! For standard triple scoring, points are simpler and equally accurate. For
//! ontology completion (EL++), taxonomy expansion, and logical query answering,
//! regions are structurally required.
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
//! - [`octagon`] -- octagon error types (implementations in [`ndarray_backend`])
//! - [`cone`] -- cone error types (implementations in [`ndarray_backend`])
//! - `hyperbolic` -- Poincare ball embeddings for tree-like hierarchies (feature-gated)
//! - [`sheaf`] -- sheaf neural networks for transitivity/consistency on graphs
//! - [`gaussian`] -- diagonal Gaussian box embeddings (KL, Bhattacharyya)
//!
//! ## Representations and scoring
//!
//! - [`distance`] -- Query2Box distance scoring
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
//! - [`trainable`] -- [`trainable::TrainableBox`] and [`trainable::TrainableCone`] with learnable parameters
//! - [`trainer`] -- negative sampling, loss computation, link prediction evaluation.
//!   Includes [`CandleBoxTrainer`] for GPU training
//!   with AdamW, cosine LR, self-adversarial NS, and filtered evaluation.
//! - [`metrics`] -- rank-based metrics (MRR, Hits@k, Mean Rank)
//! - [`optimizer`] -- AMSGrad state management
//! - [`utils`] -- numerical stability (log-space volume, stable sigmoid, Gumbel operations)
//!
//! ## Backends (feature-gated)
//!
//! - [`ndarray_backend`] -- `NdarrayBox`, `NdarrayGumbelBox`, distance functions
//!   (feature = `ndarray-backend`, **on by default**)
//! - `candle_backend` -- `CandleBox`, `CandleGumbelBox` with GPU support
//!   (feature = `candle-backend`)
//!
//! # Feature Flags
//!
//! | Feature | Default | Provides |
//! |---------|---------|----------|
//! | `ndarray-backend` | yes | [`ndarray_backend`] module (also enables `rand`) |
//! | `candle-backend` | no | `candle_backend` module (GPU via candle) |
//! | `cuda` | no | CUDA GPU support (implies `candle-backend`) |
//! | `rand` | yes (via `ndarray-backend`) | Negative sampling utilities in [`trainer`] |
//! | `hyperbolic` | no | `hyperbolic` module (Poincare ball via `hyperball` + `skel`) |
//! | `petgraph` | no | `petgraph_adapter` module (convert petgraph graphs to datasets) |
//! | `sheaf` | no | [`sheaf`] module (sheaf diffusion primitives) |
//! | `lattix` | no | [`lattix_bridge`] module (RDF/Turtle/CSV/JSON-LD KG loading via `lattix`) |
//! | `rankops` | no | Re-exports [`rankops`] (rank fusion, nDCG, MAP) |
//! | `spherical` | no | [`spherical`] module (unit-sphere embeddings, experimental) |
//! | `density` | no | [`density`] + [`density_el`] modules (density matrix embeddings, experimental) |
//!
//! # Example
//!
//! ```rust,ignore
//! // Rename to avoid shadowing std::boxed::Box
//! use subsume::Box as BoxRegion;
//!
//! // Framework-agnostic: works with NdarrayBox, CandleBox, or your own impl
//! fn compute_entailment<B: BoxRegion>(
//!     premise: &B,
//!     hypothesis: &B,
//! ) -> Result<f32, subsume::BoxError> {
//!     premise.containment_prob(hypothesis)
//! }
//! ```
//!
//! # References
//!
//! - Vilnis et al. (2018), "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
//! - Abboud et al. (2020), "BoxE: A Box Embedding Model for Knowledge Base Completion"
//! - Li et al. (2019), "Smoothing the Geometry of Probabilistic Box Embeddings" (ICLR 2019)
//! - Dasgupta et al. (2020), "Improving Local Identifiability in Probabilistic Box Embeddings"
//! - Chen et al. (2021), "Uncertainty-Aware Knowledge Graph Embeddings" (UKGE)
//! - Lee et al. (2022), "Box Embeddings for Event-Event Relation Extraction" (BERE)
//! - Cao et al. (2024, ACM Computing Surveys), "KG Embedding: A Survey from the
//!   Perspective of Representation Spaces" -- positions box/cone/octagon embeddings
//!   within the broader KGE taxonomy (Euclidean, hyperbolic, complex, geometric)
//! - Bourgaux et al. (2024, KR), "Knowledge Base Embeddings: Semantics and Theoretical Properties"
//! - Lacerda et al. (2024, TGDK), "Strong Faithfulness for ELH Ontology Embeddings"
//! - Yang & Chen (2025), "RegD: Achieving Hyperbolic-Like Expressiveness with Arbitrary
//!   Euclidean Regions" -- source of the depth/boundary dissimilarity metrics in [`distance`]

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// ---------------------------------------------------------------------------
// Core traits and geometry
// ---------------------------------------------------------------------------

/// Core [`Box`] trait: containment probability, overlap, volume, and intersection.
pub mod box_trait;

/// Cone embeddings: angular containment on the unit sphere, with negation support.
pub mod cone;

/// Octagon embeddings: axis-aligned polytopes with diagonal constraints (IJCAI 2024).
pub mod octagon;

/// Knowledge graph dataset loading (WN18RR, FB15k-237, YAGO3-10, and similar formats).
pub mod dataset;

/// Distance metrics: Query2Box distance scoring.
pub mod distance;

/// Poincare ball embeddings for tree-like hierarchical structures.
///
/// Requires the `hyperbolic` feature (uses `ndarray::ArrayView1` for
/// interoperability with the `hyperball` and `skel` crates).
#[cfg(feature = "hyperbolic")]
#[cfg_attr(docsrs, doc(cfg(feature = "hyperbolic")))]
pub mod hyperbolic;

/// AMSGrad optimizer state and learning rate utilities.
pub mod optimizer;

/// Sheaf neural networks: algebraic consistency enforcement on graphs.
#[cfg(feature = "sheaf")]
#[cfg_attr(docsrs, doc(cfg(feature = "sheaf")))]
pub mod sheaf;

/// Learnable box and cone representations with gradient-compatible parameters.
pub mod trainable;

/// Training loop utilities: negative sampling, loss kernels, link prediction evaluation.
pub mod trainer;

/// Rank-based evaluation metrics (MRR, Hits@k, Mean Rank).
pub mod metrics;

/// Re-export rankops for rank fusion, IR evaluation (nDCG, MAP), and reranking.
#[cfg(feature = "rankops")]
#[cfg_attr(docsrs, doc(cfg(feature = "rankops")))]
pub use rankops;

/// Numerical stability: log-space volume, stable sigmoid, Gumbel operations.
pub mod utils;

/// Ball embeddings for subsumption via Euclidean containment.
///
/// Concepts are solid balls `(center, radius)` in R^d. Containment:
/// `||c_A - c_B|| + r_A <= r_B`. Supports SpherE-style relation transforms
/// (translate + scale) and RegD depth/boundary dissimilarity scoring.
///
/// References:
/// - SpherE (SIGIR 2024, arXiv:2404.19130): ball embeddings for set retrieval
/// - RegD (Jan 2025, arXiv:2501.17518): balls isometric to hyperbolic space
pub mod ball;

/// Spherical cap embeddings for subsumption on the unit sphere.
///
/// Concepts are regions on S^{d-1} defined by a center (unit vector)
/// and an angular radius. Containment: `angle(c_A, c_B) + theta_A <= theta_B`.
/// This is the spherical analog of ball containment in Euclidean space.
pub mod spherical_cap;

/// Subspace embeddings for logical operations (conjunction, disjunction, negation).
///
/// Concepts are linear subspaces of R^d, represented by orthonormal bases.
/// Containment via projection, intersection via common subspace, negation
/// via orthogonal complement.
///
/// Reference: Moreira et al. (2025), arXiv:2508.16687
pub mod subspace;

/// Full-covariance Gaussian embeddings (rotated ellipsoids).
///
/// Concepts are multivariate Gaussians with full covariance, parameterized
/// via Cholesky decomposition. Supports KL divergence (asymmetric containment)
/// and Bhattacharyya distance (symmetric overlap).
pub mod ellipsoid;

/// TransBox: EL++-closed box embeddings with translational composition.
///
/// Concepts and roles as boxes with additive composition. Handles many-to-many
/// relations and complex role compositions while preserving EL++ semantics.
///
/// Reference: Yang, Chen, Sattler (2024), arXiv:2410.14571
pub mod transbox;

/// Annular sector embeddings for knowledge graph completion.
///
/// Concepts as ring-shaped regions in the complex plane. Combines rotation-based
/// relations with region uncertainty, handling 1-N/N-1/N-N relations.
///
/// Reference: Zhu & Zeng (2025), arXiv:2506.11099
pub mod annular;

/// Diagonal Gaussian box embeddings for taxonomy expansion (TaxoBell).
pub mod gaussian;

/// Density matrix region embeddings (pure-state quantum embeddings).
///
/// Represents concepts as rank-1 density matrices in a complex Hilbert space.
/// Subsumption via Loewner order, distance via fidelity / Bures metric.
/// Reference: Garg et al. (2019), "Quantum Embedding of Knowledge for Reasoning" (NeurIPS).
#[cfg(feature = "density")]
#[cfg_attr(docsrs, doc(cfg(feature = "density")))]
pub mod density;

/// Spherical knowledge graph embeddings on the unit sphere.
///
/// Entities are points on `S^{d-1}` (unit vectors). Relations are axis-angle
/// rotations. Scoring uses geodesic (great-circle) distance.
#[cfg(feature = "spherical")]
#[cfg_attr(docsrs, doc(cfg(feature = "spherical")))]
pub mod spherical;

/// Density matrix EL++ training losses: NF1-NF4 and disjointness losses
/// using fidelity-based scoring on pure-state density matrices.
#[cfg(all(feature = "rand", feature = "density"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "rand", feature = "density"))))]
pub mod density_el;

/// EL++ ontology embedding primitives (Box2EL / TransBox).
pub mod el;

/// EL++ normalized axiom dataset loader (GALEN, GO, Anatomy formats).
pub mod el_dataset;

/// EL++ ontology embedding primitives for cones (angular containment).
pub mod cone_el;

/// Composable cone query operators for first-order logical query answering.
#[cfg(feature = "ndarray-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
pub mod cone_query;

/// EL++ ontology embedding training: axiom parsing, training loop, evaluation.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod el_training;

/// Fuzzy set-theoretic operators: t-norms, t-conorms, and negation (FuzzQE).
pub mod fuzzy;

/// Taxonomy dataset loading for the TaxoBell format (`.terms` / `.taxo` / `dic.json`).
pub mod taxonomy;

/// TaxoBell combined training loss for taxonomy expansion.
pub mod taxobell;

/// TaxoBell MLP encoder and training loop with candle autograd.
///
/// Requires the `candle-backend` feature.
#[cfg(feature = "candle-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "candle-backend")))]
pub mod taxobell_encoder;

// ---------------------------------------------------------------------------
// Re-exports: primary traits and types
// ---------------------------------------------------------------------------

/// The core box embedding trait. Start here.
///
/// This trait shares its name with [`std::boxed::Box`]. To avoid shadowing, use one of:
/// - `use subsume::Box as BoxRegion;` (recommended)
/// - Qualify calls as `subsume::Box` or `<T as subsume::Box>::method()`
pub use box_trait::{Box, BoxError};

/// Convenience alias for the [`Box`] trait that avoids shadowing [`std::boxed::Box`].
///
/// `use subsume::BoxRegion;` is equivalent to `use subsume::Box as BoxRegion;`.
pub use box_trait::Box as BoxRegion;

// Re-exports: geometry errors
pub use cone::ConeError;
#[cfg_attr(docsrs, doc(cfg(feature = "hyperbolic")))]
#[cfg(feature = "hyperbolic")]
pub use hyperbolic::{
    hierarchy_preserved, pairwise_distances, Curvature, HyperbolicError, PoincareBallPoint,
};
pub use octagon::OctagonError;
#[cfg(feature = "sheaf")]
#[cfg_attr(docsrs, doc(cfg(feature = "sheaf")))]
pub use sheaf::SheafError;

// Re-exports: data loading
pub use dataset::{Dataset, DatasetError, Triple};

// Re-exports: training
pub use trainer::{
    compute_relation_cardinalities, evaluate_link_prediction, BoxEmbeddingTrainer,
    ConeEmbeddingTrainer, CpuBoxTrainingConfig, EvaluationResults, RelationCardinality,
    RelationTransform, TrainingConfig, TrainingResult,
};

// Re-export: CandleBoxTrainer (GPU training)
#[cfg(feature = "candle-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "candle-backend")))]
pub use trainer::candle_trainer::CandleBoxTrainer;

// Re-export: ndarray (public dependency -- appears in NdarrayBox/NdarrayGumbelBox/NdarrayCone API)
#[cfg(feature = "ndarray-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
pub use ndarray;

// Re-exports: evaluation metrics
pub use metrics::{adjusted_mean_rank, hits_at_k, mean_rank, mean_reciprocal_rank};

// Re-exports: Ball embeddings
pub use ball::{Ball, BallEmbedding, BallRelation};

// Re-exports: Spherical cap embeddings
pub use spherical_cap::{SphericalCap, SphericalCapEmbedding, SphericalCapRelation};

// Re-exports: Subspace embeddings
pub use subspace::{Subspace, SubspaceEmbedding};

// Re-exports: Ellipsoid (full-covariance Gaussian) embeddings
pub use ellipsoid::Ellipsoid;

// Re-exports: Gaussian boxes
pub use gaussian::GaussianBox;

// Re-exports: Density matrix embeddings
#[cfg(feature = "density")]
#[cfg_attr(docsrs, doc(cfg(feature = "density")))]
pub use density::DensityRegion;

// Re-exports: Spherical embeddings
#[cfg(feature = "spherical")]
#[cfg_attr(docsrs, doc(cfg(feature = "spherical")))]
pub use spherical::{SphericalEmbedding, SphericalPoint, SphericalRelation};

// Re-exports: Density matrix EL++ training
#[cfg_attr(docsrs, doc(cfg(all(feature = "rand", feature = "density"))))]
#[cfg(all(feature = "rand", feature = "density"))]
pub use density_el::{
    disjointness_loss_density, nf1_loss_density, train_density_el, DensityElConfig, DensityElResult,
};

// Re-exports: EL++ training
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[cfg(feature = "rand")]
pub use el_training::{
    evaluate_subsumption, train_el_embeddings, Axiom, ElTrainingConfig, ElTrainingResult, Ontology,
};

// Re-exports: cone EL++ primitives
pub use cone_el::{
    compose_cone_roles, cone_disjointness_loss, cone_existential, cone_inclusion_loss,
};

// Re-exports: cone query operators
#[cfg(feature = "ndarray-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
pub use cone_query::{cone_containment_score, ConeQuery};

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
/// Provides box and Gumbel box operations. Cone, octagon, and Gaussian
/// geometries are available through the ndarray backend only.
///
/// Enable with `features = ["candle-backend"]`.
#[cfg_attr(docsrs, doc(cfg(feature = "candle-backend")))]
#[cfg(feature = "candle-backend")]
pub mod candle_backend;

/// Adapter for constructing datasets from [`petgraph`] graphs.
///
/// Requires the `petgraph` feature.
#[cfg(feature = "petgraph")]
#[cfg_attr(docsrs, doc(cfg(feature = "petgraph")))]
pub mod petgraph_adapter;

/// Bridge from [`lattix`] knowledge graphs to subsume datasets.
///
/// Converts lattix KGs (loaded from N-Triples, Turtle, CSV, JSON-LD)
/// into subsume datasets for training.
#[cfg(feature = "lattix")]
#[cfg_attr(docsrs, doc(cfg(feature = "lattix")))]
pub mod lattix_bridge;
