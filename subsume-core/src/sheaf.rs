//! # Sheaf Neural Networks
//!
//! Algebraic structures for enforcing transitivity and consistency in graphs.
//!
//! A **sheaf** on a graph assigns vector spaces (stalks) to nodes and linear maps
//! (restriction maps) to edges. The key insight: if data is "consistent" across
//! the graph, the Dirichlet energy is zero. Non-zero energy indicates inconsistency.
//!
//! # Why Sheaves for Coreference?
//!
//! Coreference requires transitivity: if A=B and B=C, then A=C.
//! Traditional approaches enforce this post-hoc (transitive closure).
//! Sheaf networks enforce it **at the gradient level** during training.
//!
//! ```text
//! Mention A ──[restriction]──> Mention B ──[restriction]──> Mention C
//!     │                            │                            │
//!   stalk_A                     stalk_B                      stalk_C
//!     │                            │                            │
//!     └── If A=B=C, stalks should be "compatible" under restrictions
//! ```
//!
//! # Mathematical Background
//!
//! Given a graph G = (V, E), a **cellular sheaf** F assigns:
//! - To each vertex v: a vector space F(v) (the "stalk")
//! - To each edge e = (u,v): a linear map F_{u←e}: F(u) → F(e) (restriction)
//!
//! The **sheaf Laplacian** L_F is defined as:
//!
//! L_F = δ^T · δ
//!
//! where δ is the coboundary operator. The **Dirichlet energy** is:
//!
//! E(x) = x^T L_F x = Σ_{(u,v) ∈ E} ||F_{u←e}(x_u) - F_{v←e}(x_v)||²
//!
//! Low energy means the signal x is "consistent" across the sheaf.
//!
//! # Relationship to Graph Neural Networks
//!
//! | Model | Message | Aggregation | Transitivity |
//! |-------|---------|-------------|--------------|
//! | GCN | Identity | Sum | Implicit |
//! | GAT | Attention-weighted | Sum | Implicit |
//! | Sheaf | Restriction maps | Laplacian diffusion | **Explicit** |
//!
//! Sheaf neural networks generalize GNNs by learning edge-specific linear maps
//! instead of using identity or scalar weights.
//!
//! # Implementation
//!
//! This module provides framework-agnostic traits. Implementations live in:
//! - `subsume-ndarray`: CPU-based with ndarray
//! - `subsume-candle`: GPU-accelerated with candle
//!
//! # References
//!
//! - Hansen & Ghrist (2019): "Toward a spectral theory of cellular sheaves"
//! - Bodnar et al. (2022): "Neural Sheaf Diffusion" (ICLR)
//! - Barbero et al. (2022): "Sheaf Neural Networks with Connection Laplacians"

use std::collections::HashMap;
use std::fmt::Debug;

/// Error type for sheaf operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SheafError {
    /// Node not found in the graph.
    NodeNotFound(usize),
    /// Edge not found in the graph.
    EdgeNotFound(usize, usize),
    /// Dimension mismatch in linear map.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Invalid restriction map.
    InvalidRestriction(String),
    /// Computation error.
    ComputationError(String),
}

impl std::fmt::Display for SheafError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "Node {} not found", id),
            Self::EdgeNotFound(u, v) => write!(f, "Edge ({}, {}) not found", u, v),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::InvalidRestriction(msg) => write!(f, "Invalid restriction: {}", msg),
            Self::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl std::error::Error for SheafError {}

/// A restriction map (linear transformation) on an edge.
///
/// For edge (u, v), this maps from the stalk at u to the edge space.
/// The restriction map captures "how information flows" along the edge.
///
/// For coreference: if mentions u and v are coreferent, their stalks
/// should map to the same point in the edge space.
pub trait RestrictionMap: Clone + Debug {
    /// Scalar type for the map.
    type Scalar: Clone + Debug;
    /// Vector type for input/output.
    type Vector: Clone + Debug;

    /// Input dimension (stalk dimension at source node).
    fn in_dim(&self) -> usize;

    /// Output dimension (edge space dimension).
    fn out_dim(&self) -> usize;

    /// Apply the restriction map to a stalk vector.
    fn apply(&self, x: &Self::Vector) -> Result<Self::Vector, SheafError>;

    /// Apply the transpose (adjoint) of the restriction map.
    /// Used in Laplacian computation.
    fn apply_transpose(&self, x: &Self::Vector) -> Result<Self::Vector, SheafError>;

    /// Get the matrix representation (for debugging/serialization).
    fn as_matrix(&self) -> Vec<Vec<Self::Scalar>>;

    /// Frobenius norm of the map (for regularization).
    fn frobenius_norm(&self) -> Self::Scalar;
}

/// A stalk (vector space) at a node.
///
/// For coreference: the stalk at a mention node contains its embedding.
pub trait Stalk: Clone + Debug {
    /// Scalar type.
    type Scalar: Clone + Debug;
    /// Vector type.
    type Vector: Clone + Debug;

    /// Dimension of the stalk.
    fn dim(&self) -> usize;

    /// Get the current value (signal on the stalk).
    fn value(&self) -> &Self::Vector;

    /// Set the value.
    fn set_value(&mut self, v: Self::Vector) -> Result<(), SheafError>;

    /// Zero vector in this stalk.
    fn zero(&self) -> Self::Vector;
}

/// Edge data in a sheaf graph.
#[derive(Debug, Clone)]
pub struct SheafEdge<R: RestrictionMap> {
    /// Source node ID.
    pub source: usize,
    /// Target node ID.
    pub target: usize,
    /// Restriction map from source stalk to edge space.
    pub restriction_source: R,
    /// Restriction map from target stalk to edge space.
    pub restriction_target: R,
    /// Edge weight (optional, for weighted Laplacian).
    pub weight: f32,
}

/// A sheaf on a graph.
///
/// This is the main data structure for sheaf neural networks.
/// It assigns stalks to nodes and restriction maps to edges.
pub trait SheafGraph: Debug {
    /// Scalar type.
    type Scalar: Clone + Debug + Default;
    /// Vector type.
    type Vector: Clone + Debug;
    /// Restriction map type.
    type Restriction: RestrictionMap<Scalar = Self::Scalar, Vector = Self::Vector>;
    /// Stalk type.
    type Stalk: Stalk<Scalar = Self::Scalar, Vector = Self::Vector>;

    /// Number of nodes.
    fn num_nodes(&self) -> usize;

    /// Number of edges.
    fn num_edges(&self) -> usize;

    /// Get stalk at node.
    fn stalk(&self, node: usize) -> Result<&Self::Stalk, SheafError>;

    /// Get mutable stalk at node.
    fn stalk_mut(&mut self, node: usize) -> Result<&mut Self::Stalk, SheafError>;

    /// Get edge data.
    fn edge(
        &self,
        source: usize,
        target: usize,
    ) -> Result<&SheafEdge<Self::Restriction>, SheafError>;

    /// Iterate over all edges.
    fn edges(&self) -> impl Iterator<Item = &SheafEdge<Self::Restriction>>;

    /// Get neighbors of a node.
    fn neighbors(&self, node: usize) -> Result<Vec<usize>, SheafError>;

    /// Compute Dirichlet energy for the current stalk values.
    ///
    /// E(x) = Σ_{(u,v) ∈ E} w_{uv} ||R_u(x_u) - R_v(x_v)||²
    ///
    /// Low energy means consistent signal across the sheaf.
    fn dirichlet_energy(&self) -> Result<Self::Scalar, SheafError>;

    /// Compute the sheaf Laplacian action on a given node.
    ///
    /// (L_F x)_v = Σ_{u ~ v} R_v^T (R_v x_v - R_u x_u)
    fn laplacian_at(&self, node: usize) -> Result<Self::Vector, SheafError>;

    /// Perform one step of sheaf diffusion.
    ///
    /// x_{t+1} = x_t - α * L_F * x_t
    ///
    /// This smooths the signal according to sheaf structure.
    fn diffusion_step(&mut self, step_size: Self::Scalar) -> Result<(), SheafError>;
}

/// Builder for constructing sheaf graphs.
pub trait SheafBuilder: Default {
    /// The sheaf graph type being built.
    type Graph: SheafGraph;
    /// Scalar type.
    type Scalar;
    /// Vector type.
    type Vector;

    /// Add a node with initial stalk value.
    fn add_node(&mut self, stalk_value: Self::Vector) -> usize;

    /// Add an edge with restriction maps.
    ///
    /// # Arguments
    /// - `source`: Source node ID
    /// - `target`: Target node ID
    /// - `restriction_source`: Map from source stalk to edge space
    /// - `restriction_target`: Map from target stalk to edge space
    /// - `weight`: Edge weight (default 1.0)
    fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        restriction_source: <Self::Graph as SheafGraph>::Restriction,
        restriction_target: <Self::Graph as SheafGraph>::Restriction,
        weight: f32,
    ) -> Result<(), SheafError>;

    /// Build the sheaf graph.
    fn build(self) -> Result<Self::Graph, SheafError>;
}

// =============================================================================
// Sheaf Laplacian Types
// =============================================================================

/// Describes the structure of a sheaf Laplacian.
///
/// The Laplacian can be:
/// - **Connection Laplacian**: Uses orthogonal restriction maps (preserves norms)
/// - **General Laplacian**: Arbitrary linear maps
/// - **Diagonal Laplacian**: Scalar weights only (reduces to graph Laplacian)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianType {
    /// Orthogonal restriction maps (O(d) valued).
    Connection,
    /// General linear maps (GL(d) valued).
    General,
    /// Scalar weights (diagonal, equivalent to weighted graph Laplacian).
    Diagonal,
}

/// Configuration for sheaf diffusion.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Number of diffusion steps.
    pub num_steps: usize,
    /// Step size (learning rate for diffusion).
    pub step_size: f32,
    /// Whether to normalize the Laplacian.
    pub normalize: bool,
    /// Type of Laplacian to use.
    pub laplacian_type: LaplacianType,
    /// Regularization weight for restriction maps.
    pub restriction_regularization: f32,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            num_steps: 5,
            step_size: 0.1,
            normalize: true,
            laplacian_type: LaplacianType::General,
            restriction_regularization: 0.01,
        }
    }
}

// =============================================================================
// Simple In-Memory Implementation (f32, Vec<f32>)
// =============================================================================

/// Simple restriction map using a dense matrix.
#[derive(Debug, Clone)]
pub struct DenseRestriction {
    /// Matrix data in row-major order.
    pub data: Vec<f32>,
    /// Number of rows (output dimension).
    pub rows: usize,
    /// Number of columns (input dimension).
    pub cols: usize,
}

impl DenseRestriction {
    /// Create a new restriction map.
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Result<Self, SheafError> {
        if data.len() != rows * cols {
            return Err(SheafError::DimensionMismatch {
                expected: rows * cols,
                actual: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    /// Create an identity restriction (for same-dimension stalks).
    pub fn identity(dim: usize) -> Self {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self {
            data,
            rows: dim,
            cols: dim,
        }
    }

    /// Create a random orthogonal restriction map.
    ///
    /// Uses QR decomposition of random matrix.
    /// Useful for connection Laplacians.
    #[cfg(feature = "rand")]
    pub fn random_orthogonal(dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        // Generate random matrix
        let mut data: Vec<f32> = (0..dim * dim).map(|_| rng.random::<f32>() - 0.5).collect();

        // Simple Gram-Schmidt orthogonalization
        for i in 0..dim {
            // Normalize column i
            let mut norm = 0.0;
            for j in 0..dim {
                norm += data[j * dim + i] * data[j * dim + i];
            }
            norm = norm.sqrt();
            if norm > 1e-6 {
                for j in 0..dim {
                    data[j * dim + i] /= norm;
                }
            }

            // Subtract projections from remaining columns
            for k in (i + 1)..dim {
                let mut dot = 0.0;
                for j in 0..dim {
                    dot += data[j * dim + i] * data[j * dim + k];
                }
                for j in 0..dim {
                    data[j * dim + k] -= dot * data[j * dim + i];
                }
            }
        }

        Self {
            data,
            rows: dim,
            cols: dim,
        }
    }
}

impl RestrictionMap for DenseRestriction {
    type Scalar = f32;
    type Vector = Vec<f32>;

    fn in_dim(&self) -> usize {
        self.cols
    }

    fn out_dim(&self) -> usize {
        self.rows
    }

    fn apply(&self, x: &Self::Vector) -> Result<Self::Vector, SheafError> {
        if x.len() != self.cols {
            return Err(SheafError::DimensionMismatch {
                expected: self.cols,
                actual: x.len(),
            });
        }

        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i * self.cols + j] * x[j];
            }
        }
        Ok(result)
    }

    fn apply_transpose(&self, x: &Self::Vector) -> Result<Self::Vector, SheafError> {
        if x.len() != self.rows {
            return Err(SheafError::DimensionMismatch {
                expected: self.rows,
                actual: x.len(),
            });
        }

        let mut result = vec![0.0; self.cols];
        for j in 0..self.cols {
            for i in 0..self.rows {
                result[j] += self.data[i * self.cols + j] * x[i];
            }
        }
        Ok(result)
    }

    fn as_matrix(&self) -> Vec<Vec<Self::Scalar>> {
        let mut matrix = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix[i][j] = self.data[i * self.cols + j];
            }
        }
        matrix
    }

    fn frobenius_norm(&self) -> Self::Scalar {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Simple stalk holding a Vec<f32>.
#[derive(Debug, Clone)]
pub struct VecStalk {
    value: Vec<f32>,
}

impl VecStalk {
    /// Create a new stalk with given value.
    pub fn new(value: Vec<f32>) -> Self {
        Self { value }
    }
}

impl Stalk for VecStalk {
    type Scalar = f32;
    type Vector = Vec<f32>;

    fn dim(&self) -> usize {
        self.value.len()
    }

    fn value(&self) -> &Self::Vector {
        &self.value
    }

    fn set_value(&mut self, v: Self::Vector) -> Result<(), SheafError> {
        if v.len() != self.value.len() {
            return Err(SheafError::DimensionMismatch {
                expected: self.value.len(),
                actual: v.len(),
            });
        }
        self.value = v;
        Ok(())
    }

    fn zero(&self) -> Self::Vector {
        vec![0.0; self.value.len()]
    }
}

/// Simple in-memory sheaf graph.
#[derive(Debug, Clone)]
pub struct SimpleSheafGraph {
    stalks: Vec<VecStalk>,
    edges: Vec<SheafEdge<DenseRestriction>>,
    adjacency: HashMap<usize, Vec<usize>>,
}

impl SimpleSheafGraph {
    /// Create a new empty sheaf graph.
    pub fn new() -> Self {
        Self {
            stalks: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Add a node with initial stalk value.
    pub fn add_node(&mut self, value: Vec<f32>) -> usize {
        let id = self.stalks.len();
        self.stalks.push(VecStalk::new(value));
        self.adjacency.insert(id, Vec::new());
        id
    }

    /// Add an edge with restriction maps.
    pub fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        restriction_source: DenseRestriction,
        restriction_target: DenseRestriction,
        weight: f32,
    ) -> Result<(), SheafError> {
        if source >= self.stalks.len() {
            return Err(SheafError::NodeNotFound(source));
        }
        if target >= self.stalks.len() {
            return Err(SheafError::NodeNotFound(target));
        }

        // Verify dimensions
        if restriction_source.in_dim() != self.stalks[source].dim() {
            return Err(SheafError::DimensionMismatch {
                expected: self.stalks[source].dim(),
                actual: restriction_source.in_dim(),
            });
        }
        if restriction_target.in_dim() != self.stalks[target].dim() {
            return Err(SheafError::DimensionMismatch {
                expected: self.stalks[target].dim(),
                actual: restriction_target.in_dim(),
            });
        }
        if restriction_source.out_dim() != restriction_target.out_dim() {
            return Err(SheafError::InvalidRestriction(
                "Source and target restrictions must have same output dimension".into(),
            ));
        }

        self.edges.push(SheafEdge {
            source,
            target,
            restriction_source,
            restriction_target,
            weight,
        });

        self.adjacency.entry(source).or_default().push(target);
        self.adjacency.entry(target).or_default().push(source);

        Ok(())
    }
}

impl Default for SimpleSheafGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SheafGraph for SimpleSheafGraph {
    type Scalar = f32;
    type Vector = Vec<f32>;
    type Restriction = DenseRestriction;
    type Stalk = VecStalk;

    fn num_nodes(&self) -> usize {
        self.stalks.len()
    }

    fn num_edges(&self) -> usize {
        self.edges.len()
    }

    fn stalk(&self, node: usize) -> Result<&Self::Stalk, SheafError> {
        self.stalks.get(node).ok_or(SheafError::NodeNotFound(node))
    }

    fn stalk_mut(&mut self, node: usize) -> Result<&mut Self::Stalk, SheafError> {
        self.stalks
            .get_mut(node)
            .ok_or(SheafError::NodeNotFound(node))
    }

    fn edge(
        &self,
        source: usize,
        target: usize,
    ) -> Result<&SheafEdge<Self::Restriction>, SheafError> {
        self.edges
            .iter()
            .find(|e| {
                (e.source == source && e.target == target)
                    || (e.source == target && e.target == source)
            })
            .ok_or(SheafError::EdgeNotFound(source, target))
    }

    fn edges(&self) -> impl Iterator<Item = &SheafEdge<Self::Restriction>> {
        self.edges.iter()
    }

    fn neighbors(&self, node: usize) -> Result<Vec<usize>, SheafError> {
        self.adjacency
            .get(&node)
            .cloned()
            .ok_or(SheafError::NodeNotFound(node))
    }

    fn dirichlet_energy(&self) -> Result<Self::Scalar, SheafError> {
        let mut energy = 0.0;

        for edge in &self.edges {
            let x_u = self.stalks[edge.source].value();
            let x_v = self.stalks[edge.target].value();

            let r_u = edge.restriction_source.apply(x_u)?;
            let r_v = edge.restriction_target.apply(x_v)?;

            // ||R_u(x_u) - R_v(x_v)||²
            let diff_sq: f32 = r_u
                .iter()
                .zip(r_v.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();

            energy += edge.weight * diff_sq;
        }

        Ok(energy)
    }

    fn laplacian_at(&self, node: usize) -> Result<Self::Vector, SheafError> {
        let stalk = self.stalk(node)?;
        let mut result = stalk.zero();

        for edge in &self.edges {
            let (is_source, other) = if edge.source == node {
                (true, edge.target)
            } else if edge.target == node {
                (false, edge.source)
            } else {
                continue;
            };

            let x_node = self.stalks[node].value();
            let x_other = self.stalks[other].value();

            let (r_node, r_other) = if is_source {
                (&edge.restriction_source, &edge.restriction_target)
            } else {
                (&edge.restriction_target, &edge.restriction_source)
            };

            // R_node(x_node) - R_other(x_other)
            let r_x_node = r_node.apply(x_node)?;
            let r_x_other = r_other.apply(x_other)?;

            let diff: Vec<f32> = r_x_node
                .iter()
                .zip(r_x_other.iter())
                .map(|(a, b)| a - b)
                .collect();

            // R_node^T (diff)
            let contrib = r_node.apply_transpose(&diff)?;

            // Accumulate weighted contribution
            for (i, c) in contrib.iter().enumerate() {
                result[i] += edge.weight * c;
            }
        }

        Ok(result)
    }

    fn diffusion_step(&mut self, step_size: Self::Scalar) -> Result<(), SheafError> {
        // Compute Laplacian at all nodes first (to avoid borrowing issues)
        let laplacians: Vec<Vec<f32>> = (0..self.num_nodes())
            .map(|i| self.laplacian_at(i))
            .collect::<Result<_, _>>()?;

        // Update all stalks: x = x - step_size * L_F * x
        for (i, lap) in laplacians.into_iter().enumerate() {
            let stalk = &mut self.stalks[i];
            let new_value: Vec<f32> = stalk
                .value()
                .iter()
                .zip(lap.iter())
                .map(|(x, l)| x - step_size * l)
                .collect();
            stalk.set_value(new_value)?;
        }

        Ok(())
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Compute consistency score for a sheaf graph.
///
/// Returns 1.0 for perfect consistency (zero energy), decreasing toward 0.
pub fn consistency_score(graph: &impl SheafGraph<Scalar = f32>) -> Result<f32, SheafError> {
    let energy = graph.dirichlet_energy()?;
    // Use exponential decay: exp(-energy)
    Ok((-energy).exp())
}

/// Run sheaf diffusion until convergence or max iterations.
pub fn diffuse_until_convergence(
    graph: &mut SimpleSheafGraph,
    config: &DiffusionConfig,
    tolerance: f32,
) -> Result<usize, SheafError> {
    let mut prev_energy = graph.dirichlet_energy()?;

    for step in 0..config.num_steps {
        graph.diffusion_step(config.step_size)?;
        let energy = graph.dirichlet_energy()?;

        if (prev_energy - energy).abs() < tolerance {
            return Ok(step + 1);
        }
        prev_energy = energy;
    }

    Ok(config.num_steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_restriction() {
        let r = DenseRestriction::identity(3);
        let x = vec![1.0, 2.0, 3.0];
        let y = r.apply(&x).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn test_restriction_transpose() {
        // 2x3 matrix
        let r = DenseRestriction::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = r.apply(&x).unwrap();
        assert_eq!(y.len(), 2);

        let z = vec![1.0, 1.0];
        let w = r.apply_transpose(&z).unwrap();
        assert_eq!(w.len(), 3);
        // Transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]]
        // [1,4] · [1,1] = 5, [2,5] · [1,1] = 7, [3,6] · [1,1] = 9
        assert_eq!(w, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_simple_sheaf_graph() {
        let mut graph = SimpleSheafGraph::new();

        // Two nodes with 2D stalks
        let n0 = graph.add_node(vec![1.0, 0.0]);
        let n1 = graph.add_node(vec![0.0, 1.0]);

        // Identity restrictions (simplest case)
        let r = DenseRestriction::identity(2);
        graph.add_edge(n0, n1, r.clone(), r.clone(), 1.0).unwrap();

        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);

        // Dirichlet energy should be ||[1,0] - [0,1]||² = 2
        let energy = graph.dirichlet_energy().unwrap();
        assert!((energy - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_diffusion_reduces_energy() {
        let mut graph = SimpleSheafGraph::new();

        // Three nodes forming a chain
        let n0 = graph.add_node(vec![1.0, 0.0]);
        let n1 = graph.add_node(vec![0.5, 0.5]);
        let n2 = graph.add_node(vec![0.0, 1.0]);

        let r = DenseRestriction::identity(2);
        graph.add_edge(n0, n1, r.clone(), r.clone(), 1.0).unwrap();
        graph.add_edge(n1, n2, r.clone(), r.clone(), 1.0).unwrap();

        let initial_energy = graph.dirichlet_energy().unwrap();

        // Run diffusion
        for _ in 0..10 {
            graph.diffusion_step(0.1).unwrap();
        }

        let final_energy = graph.dirichlet_energy().unwrap();
        assert!(
            final_energy < initial_energy,
            "Diffusion should reduce energy"
        );
    }

    #[test]
    fn test_consistency_score() {
        let mut graph = SimpleSheafGraph::new();

        // Two nodes with identical stalks
        graph.add_node(vec![1.0, 2.0]);
        graph.add_node(vec![1.0, 2.0]);

        let r = DenseRestriction::identity(2);
        graph.add_edge(0, 1, r.clone(), r.clone(), 1.0).unwrap();

        // Perfect consistency: score should be 1.0
        let score = consistency_score(&graph).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }
}
