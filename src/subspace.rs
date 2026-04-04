//! Subspace embeddings for knowledge representation.
//!
//! Each concept is represented as a linear subspace of R^d, parameterized
//! by an orthonormal basis. Subsumption is modeled via subspace inclusion:
//! subspace A is contained in subspace B iff every vector in A's span is
//! also in B's span, equivalently `P_B * P_A = P_A` where P are projection matrices.
//!
//! # Motivation
//!
//! Subspace embeddings are the most expressive geometry for logical operations:
//!
//! - **Conjunction** (A ∧ B): intersection of subspaces
//! - **Disjunction** (A ∨ B): span of combined bases
//! - **Negation** (¬A): orthogonal complement
//!
//! No other region-based embedding supports all three natively.
//!
//! # References
//!
//! - Moreira et al. (2025), "Native Logical and Hierarchical Representations
//!   with Subspace Embeddings" (arXiv:2508.16687, Microsoft Research)
//! - Min et al. (2020), "TransINT: Embedding Implication Rules in Knowledge
//!   Graphs with Isomorphic Intersections of Linear Subspaces" (arXiv:2007.00271)
//!
//! # Containment formula
//!
//! ```text
//! A ⊆ B  ⟺  ||P_B * A_basis - A_basis||_F^2 ≈ 0
//! ```
//!
//! where P_B is the projection matrix onto B's subspace. For numerical
//! stability, we check if each basis vector of A projects onto B with
//! norm close to 1.
#![allow(missing_docs)]

use crate::BoxError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A linear subspace of R^d, represented by an orthonormal basis.
///
/// The basis vectors are stored as columns of a d × k matrix (flattened
/// in row-major order), where k = rank is the dimension of the subspace.
///
/// # Containment
///
/// Subspace A is contained in subspace B iff range(A) ⊆ range(B),
/// checked via projection: `||P_B * v|| ≈ ||v||` for each basis vector v of A.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Subspace {
    /// Orthonormal basis vectors stored as d × k matrix in row-major order.
    /// `basis[i * k + j]` = component i of basis vector j.
    basis: Vec<f32>,
    /// Ambient dimension (rows of the basis matrix).
    dim: usize,
    /// Rank of the subspace (columns of the basis matrix).
    rank: usize,
}

/// Collection of subspace embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubspaceEmbedding {
    entities: Vec<Subspace>,
    dim: usize,
}

// ---------------------------------------------------------------------------
// Subspace construction and accessors
// ---------------------------------------------------------------------------

impl Subspace {
    /// Create a new subspace from a set of basis vectors.
    ///
    /// The basis is automatically orthonormalized via Gram-Schmidt.
    /// Linearly dependent vectors are removed (rank may be less than input).
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any basis vector has
    /// wrong length. Returns [`BoxError::InvalidBounds`] if all vectors
    /// are zero or contain non-finite values.
    pub fn new(vectors: Vec<Vec<f32>>) -> Result<Self, BoxError> {
        if vectors.is_empty() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: 0.0,
            });
        }
        let dim = vectors[0].len();
        for (_i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
            for (j, &x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(BoxError::InvalidBounds {
                        dim: j,
                        min: x as f64,
                        max: x as f64,
                    });
                }
            }
        }

        // Gram-Schmidt orthonormalization
        let mut ortho: Vec<Vec<f32>> = Vec::new();
        for v in &vectors {
            let mut u = v.clone();
            // Subtract projections onto existing orthogonal vectors
            for o in &ortho {
                let dot: f32 = u.iter().zip(o.iter()).map(|(&a, &b)| a * b).sum();
                for (ui, &oi) in u.iter_mut().zip(o.iter()) {
                    *ui -= dot * oi;
                }
            }
            // Normalize
            let norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut u {
                    *x /= norm;
                }
                ortho.push(u);
            }
        }

        if ortho.is_empty() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: 0.0,
            });
        }

        let rank = ortho.len();
        // Flatten in row-major: basis[i * rank + j] = component i of vector j
        let mut basis = vec![0.0f32; dim * rank];
        for j in 0..rank {
            for i in 0..dim {
                basis[i * rank + j] = ortho[j][i];
            }
        }

        Ok(Self { basis, dim, rank })
    }

    /// Create a subspace from a pre-orthonormalized basis matrix.
    ///
    /// # Safety
    ///
    /// The caller must ensure the basis is orthonormal. No validation is performed.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if `basis.len() != dim * rank`.
    pub unsafe fn from_orthonormal(
        basis: Vec<f32>,
        dim: usize,
        rank: usize,
    ) -> Result<Self, BoxError> {
        if basis.len() != dim * rank {
            return Err(BoxError::DimensionMismatch {
                expected: dim * rank,
                actual: basis.len(),
            });
        }
        Ok(Self { basis, dim, rank })
    }

    /// Ambient dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Rank (dimension of the subspace).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Basis matrix as d × k in row-major order.
    pub fn basis(&self) -> &[f32] {
        &self.basis
    }

    /// Get the j-th basis vector.
    pub fn basis_vector(&self, j: usize) -> Vec<f32> {
        assert!(j < self.rank);
        (0..self.dim)
            .map(|i| self.basis[i * self.rank + j])
            .collect()
    }

    /// Get mutable access to all basis vectors as Vec<Vec<f32>> (for training).
    ///
    /// Returns a copy of the basis vectors. After modification, use
    /// [`Subspace::set_basis`] to apply changes and re-orthonormalize.
    pub fn basis_mut(&self) -> Vec<Vec<f32>> {
        (0..self.rank)
            .map(|j| {
                (0..self.dim)
                    .map(|i| self.basis[i * self.rank + j])
                    .collect()
            })
            .collect()
    }

    /// Replace the basis vectors and re-orthonormalize (for training).
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::InvalidBounds`] if all vectors are zero.
    pub fn set_basis(&mut self, vectors: Vec<Vec<f32>>) -> Result<(), BoxError> {
        if vectors.is_empty() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: 0.0,
            });
        }
        // Gram-Schmidt orthonormalization
        let mut ortho: Vec<Vec<f32>> = Vec::new();
        for v in &vectors {
            let mut u = v.clone();
            for o in &ortho {
                let dot: f32 = u.iter().zip(o.iter()).map(|(&a, &b)| a * b).sum();
                for (ui, &oi) in u.iter_mut().zip(o.iter()) {
                    *ui -= dot * oi;
                }
            }
            let norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut u {
                    *x /= norm;
                }
                ortho.push(u);
            }
        }
        if ortho.is_empty() {
            return Err(BoxError::InvalidBounds {
                dim: 0,
                min: 0.0,
                max: 0.0,
            });
        }
        let rank = ortho.len();
        let mut basis = vec![0.0f32; self.dim * rank];
        for j in 0..rank {
            for i in 0..self.dim {
                basis[i * rank + j] = ortho[j][i];
            }
        }
        self.basis = basis;
        self.rank = rank;
        Ok(())
    }

    /// Log-volume proxy: `rank * ln(rank) - dim * ln(dim)`.
    ///
    /// Higher rank = larger subspace = more general concept.
    /// This is a monotonic proxy for the Grassmannian volume.
    #[must_use]
    pub fn log_volume(&self) -> f32 {
        (self.rank as f32) * (self.rank as f32).ln() - (self.dim as f32) * (self.dim as f32).ln()
    }
}

// ---------------------------------------------------------------------------
// SubspaceEmbedding
// ---------------------------------------------------------------------------

impl SubspaceEmbedding {
    /// Create a new subspace embedding model.
    ///
    /// # Errors
    ///
    /// Returns [`BoxError::DimensionMismatch`] if any entity has wrong dimension.
    pub fn new(entities: Vec<Subspace>, dim: usize) -> Result<Self, BoxError> {
        for e in &entities {
            if e.dim() != dim {
                return Err(BoxError::DimensionMismatch {
                    expected: dim,
                    actual: e.dim(),
                });
            }
        }
        Ok(Self { entities, dim })
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[must_use]
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    pub fn entities(&self) -> &[Subspace] {
        &self.entities
    }
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// Containment score: how well subspace A is contained in subspace B.
///
/// ```text
/// score = 1 - (1/k) * sum_j ||P_B * a_j - a_j||^2
/// ```
///
/// where a_j are the basis vectors of A and P_B is the projection onto B.
/// Returns a value in [0, 1]: 1 means A ⊆ B, 0 means A is orthogonal to B.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the subspaces differ in ambient dimension.
pub fn containment_score(a: &Subspace, b: &Subspace) -> Result<f32, BoxError> {
    if a.dim != b.dim {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim,
            actual: b.dim,
        });
    }

    let k = a.rank;
    let mut total_residual = 0.0f32;

    for j in 0..k {
        // Get basis vector a_j
        let a_j: Vec<f32> = (0..a.dim).map(|i| a.basis[i * a.rank + j]).collect();

        // Project a_j onto B: P_B * a_j = B_basis * (B_basis^T * a_j)
        let proj = project_onto_subspace(&a_j, b);

        // Residual: ||P_B * a_j - a_j||^2
        let residual: f32 = proj
            .iter()
            .zip(a_j.iter())
            .map(|(&p, &a)| {
                let d = p - a;
                d * d
            })
            .sum();
        total_residual += residual;
    }

    let avg_residual = total_residual / (k as f32);
    Ok((1.0 - avg_residual).max(0.0).min(1.0))
}

/// Distance between two subspaces via principal angles.
///
/// ```text
/// d(A, B) = sqrt(sum_i sin^2(theta_i))
/// ```
///
/// where theta_i are the principal angles between A and B.
/// Returns 0 when subspaces are identical, sqrt(min(rank_A, rank_B)) when orthogonal.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the subspaces differ in ambient dimension.
pub fn subspace_distance(a: &Subspace, b: &Subspace) -> Result<f32, BoxError> {
    if a.dim != b.dim {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim,
            actual: b.dim,
        });
    }

    let k = a.rank.min(b.rank);
    if k == 0 {
        return Ok(0.0);
    }

    // Compute M = A^T * B (rank_a × rank_b matrix)
    let mut m = vec![0.0f32; a.rank * b.rank];
    for i in 0..a.rank {
        for j in 0..b.rank {
            let mut dot = 0.0f32;
            for r in 0..a.dim {
                dot += a.basis[r * a.rank + i] * b.basis[r * b.rank + j];
            }
            m[i * b.rank + j] = dot;
        }
    }

    // Compute M^T * M (rank_b × rank_b) — its eigenvalues are cos^2(theta_i)
    // For simplicity, compute singular values via power iteration on M^T * M.
    // Since we only need sum sin^2(theta_i) = k - sum cos^2(theta_i) = k - ||M||_F^2
    // when the bases are orthonormal, ||M||_F^2 = sum sigma_i^2 = sum cos^2(theta_i).
    // This holds because A and B have orthonormal bases.
    let frob_sq: f32 = m.iter().map(|v| v * v).sum();
    let sin_sq_sum = (k as f32) - frob_sq;
    Ok(sin_sq_sum.max(0.0).sqrt())
}

/// Intersection of two subspaces.
///
/// Returns the subspace spanned by vectors in both A and B.
/// Computed via SVD of the concatenated basis: find the null space
/// of [A_basis | -B_basis].
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the subspaces differ in ambient dimension.
pub fn intersection(a: &Subspace, b: &Subspace) -> Result<Subspace, BoxError> {
    if a.dim != b.dim {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim,
            actual: b.dim,
        });
    }

    // To find the intersection of A and B, we find vectors that are
    // simultaneously in both subspaces. Algorithm:
    // 1. Project each basis vector of A onto B
    // 2. Keep the projected vectors (which are in B by construction)
    // 3. Check if they are also in A (by projecting back onto A)
    // 4. Vectors that survive both projections with high fidelity are in A ∩ B
    //
    // More robust: iterate projection P_B * P_A until convergence.
    // Each application filters out components not in both subspaces.
    let mut candidates: Vec<Vec<f32>> = Vec::new();

    // Start with A's basis, repeatedly apply P_B then P_A
    for j in 0..a.rank {
        let mut v: Vec<f32> = (0..a.dim).map(|i| a.basis[i * a.rank + j]).collect();

        // Iterate alternating projections (converges to intersection)
        for _ in 0..10 {
            v = project_onto_subspace(&v, b);
            v = project_onto_subspace(&v, a);
        }

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.1 {
            // Normalize
            for x in &mut v {
                *x /= norm;
            }
            candidates.push(v);
        }
    }

    if candidates.is_empty() {
        // Trivial intersection — return a minimal 1D subspace as a sentinel.
        // A proper implementation would support 0-dimensional subspaces.
        let mut v = vec![0.0; a.dim];
        v[0] = 1.0;
        return Subspace::new(vec![v]);
    }

    Subspace::new(candidates)
}

/// Union (span) of two subspaces.
///
/// Returns the subspace spanned by the combined bases of A and B.
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the subspaces differ in ambient dimension.
pub fn union(a: &Subspace, b: &Subspace) -> Result<Subspace, BoxError> {
    if a.dim != b.dim {
        return Err(BoxError::DimensionMismatch {
            expected: a.dim,
            actual: b.dim,
        });
    }

    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for j in 0..a.rank {
        vectors.push((0..a.dim).map(|i| a.basis[i * a.rank + j]).collect());
    }
    for j in 0..b.rank {
        vectors.push((0..b.dim).map(|i| b.basis[i * b.rank + j]).collect());
    }

    Subspace::new(vectors)
}

/// Orthogonal complement of a subspace.
///
/// Returns the subspace of all vectors orthogonal to the given subspace.
/// Computed via Gram-Schmidt: extend the basis to a full basis of R^d,
/// then take the complement.
///
/// # Errors
///
/// Returns [`BoxError::Internal`] if the complement cannot be computed.
pub fn orthogonal_complement(s: &Subspace) -> Result<Subspace, BoxError> {
    let complement_rank = s.dim - s.rank;
    if complement_rank == 0 {
        // Full space: complement is trivial (0-dimensional)
        let mut v = vec![0.0; s.dim];
        v[0] = 1.0;
        return Subspace::new(vec![v]);
    }

    // Start with standard basis vectors, orthogonalize against s's basis
    let mut complement: Vec<Vec<f32>> = Vec::new();
    for i in 0..s.dim {
        if complement.len() >= complement_rank {
            break;
        }
        let mut v = vec![0.0; s.dim];
        v[i] = 1.0;

        // Orthogonalize against s's basis
        for j in 0..s.rank {
            let s_j: Vec<f32> = (0..s.dim).map(|r| s.basis[r * s.rank + j]).collect();
            let dot: f32 = v.iter().zip(s_j.iter()).map(|(&a, &b)| a * b).sum();
            for (vi, &sj) in v.iter_mut().zip(s_j.iter()) {
                *vi -= dot * sj;
            }
        }

        // Orthogonalize against existing complement vectors
        for c in &complement {
            let dot: f32 = v.iter().zip(c.iter()).map(|(&a, &b)| a * b).sum();
            for (vi, &ci) in v.iter_mut().zip(c.iter()) {
                *vi -= dot * ci;
            }
        }

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
            complement.push(v);
        }
    }

    if complement.is_empty() {
        return Err(BoxError::Internal(
            "could not find orthogonal complement vectors".into(),
        ));
    }

    Subspace::new(complement)
}

/// Negation score: how orthogonal are two subspaces?
///
/// ```text
/// neg_score = 1 - containment_score(A, B)
/// ```
///
/// High score means A is mostly orthogonal to B (good for disjointness).
///
/// # Errors
///
/// Returns [`BoxError::DimensionMismatch`] if the subspaces differ in ambient dimension.
pub fn negation_score(a: &Subspace, b: &Subspace) -> Result<f32, BoxError> {
    let containment = containment_score(a, b)?;
    Ok(1.0 - containment)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Project a vector onto a subspace.
///
/// ```text
/// P_B * v = B_basis * (B_basis^T * v)
/// ```
fn project_onto_subspace(v: &[f32], subspace: &Subspace) -> Vec<f32> {
    let dim = subspace.dim;
    let rank = subspace.rank;

    // Compute coefficients: B_basis^T * v
    let mut coeffs = vec![0.0f32; rank];
    for j in 0..rank {
        for i in 0..dim {
            coeffs[j] += subspace.basis[i * rank + j] * v[i];
        }
    }

    // Reconstruct: B_basis * coeffs
    let mut result = vec![0.0f32; dim];
    for j in 0..rank {
        for i in 0..dim {
            result[i] += subspace.basis[i * rank + j] * coeffs[j];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subspace_new_1d() {
        let s = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        assert_eq!(s.dim(), 3);
        assert_eq!(s.rank(), 1);
    }

    #[test]
    fn subspace_new_2d() {
        let s = Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap();
        assert_eq!(s.dim(), 3);
        assert_eq!(s.rank(), 2);
    }

    #[test]
    fn subspace_orthonormalizes() {
        // Non-orthogonal input should be orthonormalized
        let s = Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![1.0, 1.0, 0.0]]).unwrap();
        assert_eq!(s.rank(), 2);
        // Check orthonormality
        let v0 = s.basis_vector(0);
        let v1 = s.basis_vector(1);
        let norm0: f32 = v0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dot: f32 = v0.iter().zip(v1.iter()).map(|(&a, &b)| a * b).sum();
        assert!((norm0 - 1.0).abs() < 1e-5);
        assert!((norm1 - 1.0).abs() < 1e-5);
        assert!(dot.abs() < 1e-5);
    }

    #[test]
    fn subspace_removes_dependent_vectors() {
        let s = Subspace::new(vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0], // dependent
        ])
        .unwrap();
        assert_eq!(s.rank(), 1, "dependent vector should be removed");
    }

    #[test]
    fn subspace_rejects_all_zero() {
        assert!(Subspace::new(vec![vec![0.0, 0.0]]).is_err());
    }

    #[test]
    fn subspace_rejects_non_finite() {
        assert!(Subspace::new(vec![vec![f32::NAN, 0.0]]).is_err());
    }

    #[test]
    fn subspace_rejects_dimension_mismatch() {
        assert!(Subspace::new(vec![vec![1.0, 0.0], vec![1.0]]).is_err());
    }

    // --- Containment ---

    #[test]
    fn containment_identical_is_one() {
        let s = Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap();
        let score = containment_score(&s, &s).unwrap();
        assert!(score > 0.99, "identical containment = {score}, expected ~1");
    }

    #[test]
    fn containment_subspace_is_one() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap();
        let score = containment_score(&a, &b).unwrap();
        assert!(score > 0.99, "subspace containment = {score}, expected ~1");
    }

    #[test]
    fn containment_orthogonal_is_zero() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![0.0, 0.0, 1.0]]).unwrap();
        let score = containment_score(&a, &b).unwrap();
        assert!(
            score < 0.01,
            "orthogonal containment = {score}, expected ~0"
        );
    }

    #[test]
    fn containment_dimension_mismatch() {
        let a = Subspace::new(vec![vec![1.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        assert!(containment_score(&a, &b).is_err());
    }

    // --- Distance ---

    #[test]
    fn distance_identical_is_zero() {
        let s = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let d = subspace_distance(&s, &s).unwrap();
        assert!(d < 1e-5, "identical distance = {d}, expected 0");
    }

    #[test]
    fn distance_orthogonal_is_one() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![0.0, 0.0, 1.0]]).unwrap();
        let d = subspace_distance(&a, &b).unwrap();
        assert!(
            (d - 1.0).abs() < 1e-5,
            "orthogonal distance = {d}, expected 1"
        );
    }

    // --- Logical operations ---

    #[test]
    fn intersection_of_nested_is_smaller() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap();
        let inter = intersection(&a, &b).unwrap();
        assert_eq!(inter.rank(), 1);
    }

    #[test]
    fn intersection_of_orthogonal_is_trivial() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![0.0, 0.0, 1.0]]).unwrap();
        let inter = intersection(&a, &b).unwrap();
        // Orthogonal subspaces have trivial intersection
        assert_eq!(inter.rank(), 1); // degenerate 1D
    }

    #[test]
    fn union_increases_rank() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![0.0, 1.0, 0.0]]).unwrap();
        let u = union(&a, &b).unwrap();
        assert_eq!(u.rank(), 2);
    }

    #[test]
    fn union_of_nested_is_larger() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap();
        let u = union(&a, &b).unwrap();
        assert_eq!(u.rank(), 2);
    }

    #[test]
    fn test_orthogonal_complement() {
        let s = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let c = orthogonal_complement(&s).unwrap();
        assert_eq!(c.rank(), 2); // 3 - 1 = 2
                                 // Check orthogonality
        let v0 = c.basis_vector(0);
        let dot: f32 = s
            .basis_vector(0)
            .iter()
            .zip(v0.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        assert!(dot.abs() < 1e-5, "complement not orthogonal: dot = {dot}");
    }

    #[test]
    fn negation_orthogonal_is_one() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![0.0, 0.0, 1.0]]).unwrap();
        let score = negation_score(&a, &b).unwrap();
        assert!(score > 0.99, "orthogonal negation = {score}, expected ~1");
    }

    #[test]
    fn negation_identical_is_zero() {
        let s = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let score = negation_score(&s, &s).unwrap();
        assert!(score < 0.01, "identical negation = {score}, expected ~0");
    }

    // --- Log volume ---

    #[test]
    fn log_volume_increases_with_rank() {
        let a = Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        let b = Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap();
        assert!(
            b.log_volume() > a.log_volume(),
            "higher rank should have higher log_volume"
        );
    }

    // --- Embedding model ---

    #[test]
    fn embedding_model_construction() {
        let entities = vec![
            Subspace::new(vec![vec![1.0, 0.0, 0.0]]).unwrap(),
            Subspace::new(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).unwrap(),
        ];
        let model = SubspaceEmbedding::new(entities, 3).unwrap();
        assert_eq!(model.num_entities(), 2);
        assert_eq!(model.dim(), 3);
    }

    #[test]
    fn embedding_model_rejects_dim_mismatch() {
        let entities = vec![Subspace::new(vec![vec![1.0, 0.0]]).unwrap()];
        assert!(SubspaceEmbedding::new(entities, 3).is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_subspace(dim: usize, max_rank: usize) -> impl Strategy<Value = Subspace> {
        (1..=max_rank).prop_flat_map(move |rank| {
            prop::collection::vec(prop::collection::vec(-10.0f32..10.0, dim), rank)
                .prop_filter_map("valid subspace", |vectors| Subspace::new(vectors).ok())
        })
    }

    fn arb_subspace_pair(
        dim: usize,
        max_rank: usize,
    ) -> impl Strategy<Value = (Subspace, Subspace)> {
        (arb_subspace(dim, max_rank), arb_subspace(dim, max_rank))
    }

    proptest! {
        #[test]
        fn prop_containment_in_unit_interval(
            (a, b) in arb_subspace_pair(4, 3)
        ) {
            let s = containment_score(&a, &b).unwrap();
            prop_assert!(s >= -1e-6, "containment_score < 0: {s}");
            prop_assert!(s <= 1.0 + 1e-6, "containment_score > 1: {s}");
        }

        #[test]
        fn prop_distance_nonneg(
            (a, b) in arb_subspace_pair(4, 3)
        ) {
            let d = subspace_distance(&a, &b).unwrap();
            prop_assert!(d >= -1e-6, "subspace_distance < 0: {d}");
        }

        #[test]
        fn prop_distance_symmetric(
            (a, b) in arb_subspace_pair(4, 3)
        ) {
            let d_ab = subspace_distance(&a, &b).unwrap();
            let d_ba = subspace_distance(&b, &a).unwrap();
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-3,
                "distance should be symmetric: {d_ab} != {d_ba}"
            );
        }

        #[test]
        fn prop_self_containment_is_one(
            s in arb_subspace(4, 3)
        ) {
            let score = containment_score(&s, &s).unwrap();
            prop_assert!(score > 0.99, "self containment = {score}, expected ~1");
        }

        #[test]
        fn prop_negation_in_unit_interval(
            (a, b) in arb_subspace_pair(4, 3)
        ) {
            let s = negation_score(&a, &b).unwrap();
            prop_assert!(s >= -1e-6, "negation_score < 0: {s}");
            prop_assert!(s <= 1.0 + 1e-6, "negation_score > 1: {s}");
        }

        #[test]
        fn prop_orthogonal_complement_rank(
            s in arb_subspace(4, 3)
        ) {
            let c = orthogonal_complement(&s).unwrap();
            prop_assert_eq!(
                c.rank() + s.rank(),
                s.dim(),
                "complement rank + subspace rank should equal dim"
            );
        }

        #[test]
        fn prop_union_rank_at_least_max(
            (a, b) in arb_subspace_pair(4, 3)
        ) {
            let u = union(&a, &b).unwrap();
            prop_assert!(u.rank() >= a.rank(), "union rank should be >= a rank");
            prop_assert!(u.rank() >= b.rank(), "union rank should be >= b rank");
        }

        #[test]
        fn prop_basis_is_orthonormal(
            s in arb_subspace(4, 3)
        ) {
            for i in 0..s.rank() {
                let v = s.basis_vector(i);
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                prop_assert!((norm - 1.0).abs() < 1e-4, "basis[{i}] norm = {norm}");
                for j in (i + 1)..s.rank() {
                    let w = s.basis_vector(j);
                    let dot: f32 = v.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
                    prop_assert!(dot.abs() < 1e-4, "basis[{i}] . basis[{j}] = {dot}");
                }
            }
        }
    }
}
