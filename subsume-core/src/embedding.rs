//! Box embedding collections and training utilities.

use crate::{Box, BoxError};

/// A collection of box embeddings with associated metadata.
///
/// This trait allows framework-agnostic operations on collections of boxes,
/// such as batch containment queries, hierarchical clustering, etc.
pub trait BoxEmbedding<B: Box> {
    /// Get the number of boxes in the collection.
    fn len(&self) -> usize;

    /// Check if the collection is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a box by index.
    ///
    /// # Errors
    ///
    /// Returns `BoxError::Internal` if index is out of bounds.
    fn get(&self, index: usize) -> Result<&B, BoxError>;

    /// Compute pairwise containment probabilities.
    ///
    /// Returns a matrix where `result[i][j] = P(box[j] ⊆ box[i])`.
    ///
    /// # Errors
    ///
    /// Returns `BoxError` if any boxes have dimension mismatches.
    fn containment_matrix(
        &self,
        temperature: B::Scalar,
    ) -> Result<Vec<Vec<B::Scalar>>, BoxError>;

    /// Find boxes that contain a given box.
    ///
    /// Returns indices of boxes where `containment_prob(box[i], query) > threshold`.
    ///
    /// # Errors
    ///
    /// Returns `BoxError` if query has dimension mismatch with any box.
    fn containing_boxes(
        &self,
        query: &B,
        threshold: B::Scalar,
        temperature: B::Scalar,
    ) -> Result<Vec<usize>, BoxError>;

    /// Find boxes contained by a given box.
    ///
    /// Returns indices of boxes where `containment_prob(query, box[i]) > threshold`.
    ///
    /// # Errors
    ///
    /// Returns `BoxError` if query has dimension mismatch with any box.
    fn contained_boxes(
        &self,
        query: &B,
        threshold: B::Scalar,
        temperature: B::Scalar,
    ) -> Result<Vec<usize>, BoxError>;
}

/// A simple collection of boxes implemented as a vector.
///
/// This is a basic implementation of `BoxEmbedding` that stores boxes in a `Vec`.
/// For large-scale use, consider implementing more efficient data structures
/// (e.g., spatial indexes, hierarchical clustering).
///
/// # Example
///
/// ```rust,ignore
/// // This example requires a backend implementation (e.g., subsume-ndarray)
/// use subsume_core::{BoxEmbedding, BoxCollection};
/// use subsume_ndarray::NdarrayBox;
/// use ndarray::array;
///
/// let mut collection = BoxCollection::new();
/// collection.push(NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0)?);
/// collection.push(NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0)?);
///
/// let matrix = collection.containment_matrix(1.0)?;
/// # Ok::<(), subsume_core::BoxError>(())
/// ```
#[derive(Debug, Clone)]
pub struct BoxCollection<B: Box> {
    boxes: Vec<B>,
}

impl<B: Box> BoxCollection<B> {
    /// Create a new empty collection.
    pub fn new() -> Self {
        Self { boxes: Vec::new() }
    }

    /// Create a collection from a vector of boxes.
    pub fn from_vec(boxes: Vec<B>) -> Self {
        Self { boxes }
    }

    /// Add a box to the collection.
    pub fn push(&mut self, box_: B) {
        self.boxes.push(box_);
    }

    /// Get all boxes as a slice.
    pub fn as_slice(&self) -> &[B] {
        &self.boxes
    }

    /// Convert into a vector of boxes.
    pub fn into_vec(self) -> Vec<B> {
        self.boxes
    }
}

impl<B: Box> Default for BoxCollection<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Box> BoxEmbedding<B> for BoxCollection<B> {
    fn len(&self) -> usize {
        self.boxes.len()
    }

    fn get(&self, index: usize) -> Result<&B, BoxError> {
        self.boxes.get(index).ok_or_else(|| BoxError::Internal(format!("Index {index} out of bounds")))
    }

    fn containment_matrix(
        &self,
        temperature: B::Scalar,
    ) -> Result<Vec<Vec<B::Scalar>>, BoxError> {
        let n = self.len();
        let mut matrix = Vec::with_capacity(n);
        
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            let box_i = self.get(i)?;
            
            for j in 0..n {
                let box_j = self.get(j)?;
                let prob = box_i.containment_prob(box_j, temperature)?;
                row.push(prob);
            }
            
            matrix.push(row);
        }
        
        Ok(matrix)
    }

    fn containing_boxes(
        &self,
        query: &B,
        threshold: B::Scalar,
        temperature: B::Scalar,
    ) -> Result<Vec<usize>, BoxError> {
        let mut results = Vec::new();
        
        for (i, box_) in self.boxes.iter().enumerate() {
            let prob = box_.containment_prob(query, temperature)?;
            if prob > threshold {
                results.push(i);
            }
        }
        
        Ok(results)
    }

    fn contained_boxes(
        &self,
        query: &B,
        threshold: B::Scalar,
        temperature: B::Scalar,
    ) -> Result<Vec<usize>, BoxError> {
        let mut results = Vec::new();
        
        for (i, box_) in self.boxes.iter().enumerate() {
            let prob = query.containment_prob(box_, temperature)?;
            if prob > threshold {
                results.push(i);
            }
        }
        
        Ok(results)
    }
}

impl<B: Box> From<Vec<B>> for BoxCollection<B> {
    fn from(boxes: Vec<B>) -> Self {
        Self::from_vec(boxes)
    }
}

