//! Box embedding collections and training utilities.

use crate::{Box, BoxError};

/// A simple collection of boxes implemented as a vector.
///
/// Provides batch operations over boxes: containment/overlap matrices,
/// spatial queries (containing, contained, overlapping, nearest), and
/// bounding box computation.
///
/// # Example
///
/// ```rust,ignore
/// use subsume::BoxCollection;
/// use ndarray::array;
///
/// let mut collection = BoxCollection::new();
/// collection.push(NdarrayBox::new(array![0.0, 0.0], array![1.0, 1.0], 1.0)?);
/// collection.push(NdarrayBox::new(array![0.2, 0.2], array![0.8, 0.8], 1.0)?);
///
/// let matrix = collection.containment_matrix(1.0)?;
/// # Ok::<(), subsume::BoxError>(())
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

    /// Get the number of boxes in the collection.
    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a box by index.
    pub fn get(&self, index: usize) -> Result<&B, BoxError> {
        self.boxes
            .get(index)
            .ok_or_else(|| BoxError::Internal(format!("Index {index} out of bounds")))
    }

    /// Compute pairwise containment probabilities.
    pub fn containment_matrix(
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

    /// Find boxes that contain a given box.
    pub fn containing_boxes(
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

    /// Find boxes contained by a given box.
    pub fn contained_boxes(
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

    /// Compute pairwise overlap probabilities.
    pub fn overlap_matrix(&self, temperature: B::Scalar) -> Result<Vec<Vec<B::Scalar>>, BoxError> {
        let n = self.len();
        let mut matrix = Vec::with_capacity(n);

        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            let box_i = self.get(i)?;

            for j in 0..n {
                let box_j = self.get(j)?;
                let prob = box_i.overlap_prob_fast(box_j, temperature)?;
                row.push(prob);
            }

            matrix.push(row);
        }

        Ok(matrix)
    }

    /// Find boxes that overlap with a given box.
    pub fn overlapping_boxes(
        &self,
        query: &B,
        threshold: B::Scalar,
        temperature: B::Scalar,
    ) -> Result<Vec<usize>, BoxError> {
        let mut results = Vec::new();

        for (i, box_) in self.boxes.iter().enumerate() {
            let prob = box_.overlap_prob_fast(query, temperature)?;
            if prob > threshold {
                results.push(i);
            }
        }

        Ok(results)
    }

    /// Find the k nearest boxes by distance.
    pub fn nearest_boxes(&self, query: &B, k: usize) -> Result<Vec<usize>, BoxError> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut distances: Vec<(usize, B::Scalar)> = Vec::new();

        for (i, box_) in self.boxes.iter().enumerate() {
            let dist = box_.distance(query)?;
            distances.push((i, dist));
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(distances.into_iter().take(k).map(|(idx, _)| idx).collect())
    }

    /// Compute the bounding box of all boxes in the collection.
    pub fn bounding_box(&self) -> Result<B, BoxError>
    where
        B: Clone,
    {
        if self.is_empty() {
            return Err(BoxError::Internal(
                "Cannot compute bounding box of empty collection".to_string(),
            ));
        }

        let first_box = self.get(0)?;
        let mut bbox = first_box.clone();

        for i in 1..self.len() {
            let box_i = self.get(i)?;
            bbox = bbox.union(box_i)?;
        }

        Ok(bbox)
    }
}

impl<B: Box> Default for BoxCollection<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Box> From<Vec<B>> for BoxCollection<B> {
    fn from(boxes: Vec<B>) -> Self {
        Self::from_vec(boxes)
    }
}
