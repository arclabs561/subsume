//! Region embeddings for entailment and set containment.
//!
//! `subsume` represents concepts as geometric regions. A general concept
//! contains the regions for its more specific concepts, so containment becomes
//! the scoring operation for hierarchy, ontology, and set-query tasks.
//!
//! # Example
//!
//! ```rust
//! use ndarray::array;
//! use subsume::{ndarray_backend::NdarrayBox, HyperBox};
//!
//! # fn main() -> Result<(), subsume::BoxError> {
//! let premise = NdarrayBox::new(array![0., 0., 0.], array![1., 1., 1.], 1.0)?;
//! let hypothesis = NdarrayBox::new(array![0.2, 0.2, 0.2], array![0.8, 0.8, 0.8], 1.0)?;
//!
//! let p = premise.containment_prob(&hypothesis)?;
//! assert!(p > 0.9);
//! # Ok(())
//! # }
//! ```
//!
//! # Choosing a Geometry
//!
//! | Task | Start with |
//! | --- | --- |
//! | Containment hierarchy | `ndarray_backend::NdarrayBox` or `NdarrayGumbelBox` |
//! | Logical queries with negation | Cone or subspace |
//! | Taxonomy expansion with uncertainty | Gaussian boxes |
//! | EL++ ontology completion | `el`, `transbox` |
//! | Tree-like hierarchies in low dimension | Hyperbolic intervals or balls |
//!
//! Scores are meaningful within a geometry. Do not compare raw scores across
//! geometries without calibration.
//!
//! # Modules
//!
//! - `box_trait`: `HyperBox`, containment, overlap, volume, and intersection.
//! - `ndarray_backend`: default CPU box and Gumbel-box implementations.
//! - `dataset`, `trainer`, `metrics`: knowledge-graph loading, training, and
//!   rank-based evaluation.
//! - `el`: EL++ ontology embedding losses.
//! - `gaussian`, `cone`, `octagon`, `hyperbolic`: additional geometries.
//!
//! # Limits
//!
//! - For ordinary link prediction, point embeddings are often simpler.
//! - Region scores from different geometries are not directly comparable.
//! - Several geometry trainers are research paths, not recommended defaults.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// ---------------------------------------------------------------------------
// Core traits and geometry
// ---------------------------------------------------------------------------

/// Core [`HyperBox`] trait: containment probability, overlap, volume, and intersection.
pub mod box_trait;
pub mod region;

/// Cone embeddings: angular containment on the unit sphere, with negation support.
pub mod cone;

/// Octagon embeddings: axis-aligned polytopes with diagonal constraints (IJCAI 2024).
pub mod octagon;

/// Knowledge graph dataset loading (WN18RR, FB15k-237, YAGO3-10, and similar formats).
#[cfg(feature = "kge")]
#[cfg_attr(docsrs, doc(cfg(feature = "kge")))]
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
#[cfg(feature = "kge")]
#[cfg_attr(docsrs, doc(cfg(feature = "kge")))]
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

/// EL++ ontology embedding training: axiom parsing, training loop, evaluation.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod el_training;

/// Conjunctive least-common-ancestor queries over faithful EL++ box embeddings
/// via containment-gated proximity to the join box.
pub mod clqa;

/// Taxonomy dataset loading for the TaxoBell format (`.terms` / `.taxo` / `dic.json`).
#[cfg(feature = "kge")]
#[cfg_attr(docsrs, doc(cfg(feature = "kge")))]
pub mod taxonomy;

/// TaxoBell combined training loss for taxonomy expansion.
pub mod taxobell;

// ---------------------------------------------------------------------------
// Re-exports: primary traits and types
// ---------------------------------------------------------------------------

/// The core box (axis-aligned hyperrectangle) embedding trait. Start here.
///
/// Named `HyperBox` (not `Box`) so it does not shadow [`std::boxed::Box`].
/// "Box" in the geometric-embedding literature means an n-dimensional
/// hyperrectangle; this trait models exactly that.
pub use box_trait::{BoxError, HyperBox};

/// The unified region abstraction over all geometries (boxes, balls,
/// ellipsoids, subspaces). See [`region::Region`].
pub use region::Region;

/// Deprecated alias for [`HyperBox`]. The trait was renamed from `Box` to
/// `HyperBox` so it no longer shadows [`std::boxed::Box`]; switch to `HyperBox`.
#[deprecated(
    since = "0.13.0",
    note = "renamed to `HyperBox` (the old name shadowed std::boxed::Box); use `HyperBox`"
)]
pub use box_trait::HyperBox as Box;

/// Convenience alias for the [`HyperBox`] trait, retained for compatibility.
pub use box_trait::HyperBox as BoxRegion;

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
#[cfg(feature = "kge")]
pub use dataset::{Dataset, DatasetError, Triple};

// Re-exports: training (always available)
pub use trainer::{
    compute_relation_cardinalities, BoxEmbeddingTrainer, CpuBoxTrainingConfig, EvaluationResults,
    RelationCardinality, RelationTransform, TrainingConfig, TrainingResult,
};

// Re-exports: training (requires kge feature)
#[cfg(feature = "kge")]
#[cfg_attr(docsrs, doc(cfg(feature = "kge")))]
pub use trainer::{evaluate_link_prediction, ConeEmbeddingTrainer};

// Re-export: ndarray (public dependency -- appears in NdarrayBox/NdarrayGumbelBox/NdarrayCone API)
#[cfg(feature = "ndarray-backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray-backend")))]
pub use ndarray;

// Re-exports: evaluation metrics
#[cfg(feature = "kge")]
#[cfg_attr(docsrs, doc(cfg(feature = "kge")))]
pub use metrics::{adjusted_mean_rank, hits_at_k, mean_rank, mean_reciprocal_rank};

// Re-exports: Ball embeddings
pub use ball::{Ball, BallEmbedding, BallRelation};

// Re-exports: Spherical cap embeddings
pub use spherical_cap::{SphericalCap, SphericalCapEmbedding, SphericalCapRelation};

// Re-exports: Subspace embeddings
pub use subspace::{Subspace, SubspaceEmbedding};

// Re-exports: Ellipsoid (full-covariance Gaussian) embeddings
pub use ellipsoid::Ellipsoid;

// Re-exports: Annular sector embeddings
pub use annular::{AnnularEmbedding, AnnularRelation, AnnularSector};

// Re-exports: TransBox embeddings
pub use transbox::{TransBoxConcept, TransBoxModel, TransBoxRole};

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
    TrainedElModel,
};

// Re-exports: cone EL++ primitives
pub use cone_el::{
    compose_cone_roles, cone_disjointness_loss, cone_existential, cone_inclusion_loss,
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
#[cfg(feature = "kge")]
#[cfg_attr(docsrs, doc(cfg(feature = "kge")))]
pub mod lattix_bridge;
